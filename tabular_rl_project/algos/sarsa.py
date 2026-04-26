"""SARSA (Skript Algorithm 20, Übungsblatt 7 Aufgabe 2).

On-policy TD-Kontrolle: die Verhaltenspolicy ist epsilon-greedy bzgl. der
aktuellen Q-Tabelle, und exakt dieselbe Policy wird als Zielpolicy verwendet:

    Q(s,a) ← Q(s,a) + α_n · (r + γ · Q(s',a') − Q(s,a))
    mit a' ~ π_{ε}(·|s')

SARSA konvergiert gegen Q^{π_ε} (on-policy). Mit epsilon → 0 nähert sich
Q^{π_ε} der optimalen Q* an. Gegenüber Q-Learning (off-policy) vorsichtiger
an Klippen / Risikobereichen (Blatt 7 Aufg. 2, Cliff-Walk-Vergleich).

alpha und epsilon dürfen float (konstant) oder Schedule ``(n:int)→float`` sein.
    - alpha:   per-(s,a)-Besuchszähler
    - epsilon: per-Episoden-Zähler
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..envs.mdp_base import FiniteMDP
from .exploration import epsilon_greedy_action
from .schedules import Schedule


def sarsa(
    env: FiniteMDP,
    alpha: Union[float, Schedule],
    epsilon: Union[float, Schedule] = 0.1,
    gamma: Optional[float] = None,
    n_episodes: int = 1000,
    max_steps: int = 1000,
    seed: Optional[int] = None,
    eval_every: Optional[int] = None,
    eval_fn: Optional[Callable[[np.ndarray], Any]] = None,
) -> Tuple[np.ndarray, Dict]:
    """SARSA (Skript Algorithmus 20).

    Parameters
    ----------
    env : FiniteMDP
        Umgebung; Transitionen via env.step().
    alpha : float or Schedule
        Schrittweite: float → konstant; Callable (n)→float → per-(s,a)-Besuch.
    epsilon : float or Schedule
        Explorationsrate: float → konstant;
        Callable (ep)→float → per-Episode (ep=1 für erste Episode).
    gamma : float, optional
        Diskontierung; None → env.gamma.
    n_episodes : int
        Anzahl Trainings-Episoden.
    max_steps : int
        Maximale Episodenlänge.
    seed : int, optional
        Seed für np.random.default_rng.
    eval_every : int, optional
        Auswertungsintervall in Episoden; eval_fn muss ebenfalls gesetzt sein.
    eval_fn : Callable[[np.ndarray], Any], optional
        Auswertungsfunktion; Ergebnisse in info["eval_history"].

    Returns
    -------
    Q : np.ndarray
        Gelernte Q-Tabelle, Shape (S, A).
    info : dict
        "episodes", "total_steps", "visit_counts", "episode_returns",
        "eval_history".
    """
    gamma = gamma if gamma is not None else env.gamma
    rng = np.random.default_rng(seed)
    alpha_fn: Schedule = alpha if callable(alpha) else (lambda n: float(alpha))  # type: ignore[arg-type]
    eps_fn: Schedule = epsilon if callable(epsilon) else (lambda n: float(epsilon))  # type: ignore[arg-type]

    Q = np.zeros((env.n_states, env.n_actions))
    N = np.zeros((env.n_states, env.n_actions), dtype=np.int64)
    episode_returns: List[float] = []
    eval_history: List[Tuple[int, Any]] = []
    total_steps = 0

    for ep in range(n_episodes):
        s = env.reset()
        eps = float(eps_fn(ep + 1))
        # SARSA benötigt a schon vor dem ersten Step
        a = epsilon_greedy_action(Q, s, env, eps, rng) if not env.is_terminal(s) else 0
        ep_return = 0.0
        discount = 1.0

        for _ in range(max_steps):
            if env.is_terminal(s):
                break

            s_next, r, done = env.step(a)
            total_steps += 1

            # Nächste Aktion aus on-policy (gleiche epsilon-greedy Policy)
            a_next = (
                epsilon_greedy_action(Q, s_next, env, eps, rng)
                if not env.is_terminal(s_next)
                else 0
            )

            N[s, a] += 1
            lr = alpha_fn(int(N[s, a]))
            # Q[terminal, a_next] = 0 per Initialisierung (korrekt)
            Q[s, a] += lr * (r + gamma * Q[s_next, a_next] - Q[s, a])

            ep_return += discount * r
            discount *= gamma
            s, a = s_next, a_next
            if done:
                break

        episode_returns.append(ep_return)

        if eval_every is not None and eval_fn is not None and (ep + 1) % eval_every == 0:
            eval_history.append((ep + 1, eval_fn(Q)))

    return Q, {
        "episodes": n_episodes,
        "total_steps": total_steps,
        "visit_counts": N.copy(),
        "episode_returns": episode_returns,
        "eval_history": eval_history,
    }
