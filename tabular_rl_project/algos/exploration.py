"""Verhaltenspolicies und Explorationsstrategien.

Aus Übungsblatt 6 (feste Verhaltenspolicies für Q-Learning) und
Übungsblatt 7 Aufgabe 2 (epsilon-greedy, konstant und abnehmend).

    - uniform_random_policy(env)
    - epsilon_greedy(Q, env, eps)         → stochastische Policy (S, A)
    - epsilon_greedy_action(Q, s, env, eps, rng)  → einzelne Aktion
    - epsilon_schedule_factory(schedule)  → Policy-Funktion mit Schedule

Alle Funktionen respektieren ``env.allowed_actions``:
Unerlaubte Aktionen erhalten keine Wahrscheinlichkeitsmasse.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..envs.mdp_base import FiniteMDP
from .schedules import Schedule


def uniform_random_policy(env: FiniteMDP) -> np.ndarray:
    """Gleichverteilte stochastische Policy über alle erlaubten Aktionen.

    Parameters
    ----------
    env : FiniteMDP
        Umgebung mit definierten ``allowed_actions``.

    Returns
    -------
    policy : np.ndarray
        Shape (S, A); nur erlaubte Aktionen erhalten Wahrscheinlichkeitsmasse.
    """
    S, A = env.n_states, env.n_actions
    policy = np.zeros((S, A))
    for s in env.states:
        acts = env.allowed_actions[s]
        policy[s, acts] = 1.0 / len(acts)
    return policy


def epsilon_greedy(Q: np.ndarray, env: FiniteMDP, eps: float) -> np.ndarray:
    """Epsilon-greedy Policy als (S, A)-Wahrscheinlichkeitsmatrix.

    Mit Wahrscheinlichkeit ``eps`` gleichverteilt über erlaubte Aktionen,
    mit ``(1 - eps)`` greedy bzgl. Q. Bei Gleichstand gewinnt die erste
    erlaubte Aktion mit dem höchsten Q-Wert (niedrigster Index).

    Parameters
    ----------
    Q : np.ndarray
        Q-Tabelle Shape (S, A).
    env : FiniteMDP
        Umgebung mit ``allowed_actions``.
    eps : float
        Explorationsrate ∈ [0, 1].

    Returns
    -------
    policy : np.ndarray
        Shape (S, A).
    """
    S, A = env.n_states, env.n_actions
    policy = np.zeros((S, A))
    for s in env.states:
        acts = env.allowed_actions[s]
        n_acts = len(acts)
        best = acts[int(np.argmax(Q[s, acts]))]
        policy[s, acts] = eps / n_acts
        policy[s, best] += 1.0 - eps
    return policy


def epsilon_greedy_action(
    Q: np.ndarray,
    s: int,
    env: FiniteMDP,
    eps: float,
    rng: np.random.Generator,
) -> int:
    """Zieht genau eine Aktion nach der Epsilon-greedy-Strategie.

    Effizienter als die vollständige Policy-Matrix, wenn nur eine
    einzelne Aktion benötigt wird (Online-Sampling in TD/Q-Learning).

    Parameters
    ----------
    Q : np.ndarray
        Q-Tabelle Shape (S, A).
    s : int
        Aktueller Zustand.
    env : FiniteMDP
        Umgebung mit ``allowed_actions``.
    eps : float
        Explorationsrate ∈ [0, 1].
    rng : np.random.Generator
        Zufallsgenerator (``np.random.default_rng``).

    Returns
    -------
    int
        Gewählte Aktion.
    """
    acts = env.allowed_actions[s]
    if rng.random() < eps:
        return int(rng.choice(acts))
    return int(acts[int(np.argmax(Q[s, acts]))])


def epsilon_schedule_factory(
    schedule: Schedule,
) -> Callable[[np.ndarray, FiniteMDP, int], np.ndarray]:
    """Erzeugt eine Policy-Funktion mit schedule-basiertem Epsilon.

    Parameters
    ----------
    schedule : Schedule
        Callable ``(n: int) -> float``, liefert epsilon_n für Schritt n.

    Returns
    -------
    Callable[[np.ndarray, FiniteMDP, int], np.ndarray]
        ``policy_fn(Q, env, n)`` → stochastische Policy (S, A).
        n ist der globale Schrittzähler, den der Aufrufer pflegt.
    """
    def _policy(Q: np.ndarray, env: FiniteMDP, n: int) -> np.ndarray:
        eps = float(schedule(n))
        return epsilon_greedy(Q, env, eps)
    return _policy
