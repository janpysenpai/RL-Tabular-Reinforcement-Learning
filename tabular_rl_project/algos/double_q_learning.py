"""Double Q-Learning (Übungsblatt 7 Aufgabe 6; Blatt 8 Aufgabe 4 f).

Adressiert die Overestimation-Bias von Q-Learning: E[max_i X_i] ≥ max_i E[X_i].
Zwei unabhängige Q-Tabellen Q1 und Q2 werden abwechselnd aktualisiert:

    Mit Wahrscheinlichkeit 0.5:
        A* = argmax_{a'} Q1(s', a')          [Selektion über Q1]
        Q1(s,a) ← Q1(s,a) + α · (r + γ · Q2(s', A*) − Q1(s,a))   [Evaluation über Q2]

    Sonst:
        A* = argmax_{a'} Q2(s', a')
        Q2(s,a) ← Q2(s,a) + α · (r + γ · Q1(s', A*) − Q2(s,a))

Verhaltenspolicy: epsilon-greedy bzgl. Q_avg = (Q1 + Q2) / 2.

Durch die Entkopplung von Selektion und Evaluation wird die Maximierungs-Bias
aus E[Q2(s', argmax Q1(s', ·))] ≈ E[Q(s', a*)] eliminiert (Hado van Hasselt 2010).

alpha und epsilon dürfen float oder Schedule ``(n:int)→float`` sein:
    - alpha:   pro (s,a)-Tabellenpaar, d.h. N1[s,a] resp. N2[s,a]
    - epsilon: per-Episode
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..envs.mdp_base import FiniteMDP
from .exploration import epsilon_greedy_action
from .schedules import Schedule


def double_q_learning(
    env: FiniteMDP,
    alpha: Union[float, Schedule],
    epsilon: Union[float, Schedule] = 0.1,
    gamma: Optional[float] = None,
    n_episodes: int = 1000,
    max_steps: int = 1000,
    seed: Optional[int] = None,
    eval_every: Optional[int] = None,
    eval_fn: Optional[Callable[[np.ndarray], Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Double Q-Learning.

    Parameters
    ----------
    env : FiniteMDP
        Umgebung; Transitionen via env.step().
    alpha : float or Schedule
        Schrittweite pro (s,a)-Besuch der jeweiligen Tabelle.
    epsilon : float or Schedule
        Explorationsrate für die Verhaltenspolicy (epsilon-greedy bzgl. Q_avg).
        float → konstant; Callable (ep)→float → per-Episode.
    gamma : float, optional
        Diskontierung; None → env.gamma.
    n_episodes : int
        Anzahl Trainings-Episoden.
    max_steps : int
        Maximale Episodenlänge.
    seed : int, optional
        Seed für np.random.default_rng.
    eval_every : int, optional
        Auswertungsintervall; eval_fn muss ebenfalls gesetzt sein.
    eval_fn : Callable[[np.ndarray], Any], optional
        Erhält Q_avg = (Q1+Q2)/2; Ergebnisse in info["eval_history"].

    Returns
    -------
    Q1 : np.ndarray
        Erste Q-Tabelle, Shape (S, A).
    Q2 : np.ndarray
        Zweite Q-Tabelle, Shape (S, A).
    Q_avg : np.ndarray
        (Q1 + Q2) / 2, Shape (S, A).
    info : dict
        "episodes", "total_steps", "visit_counts_Q1", "visit_counts_Q2",
        "episode_returns", "eval_history".
    """
    gamma = gamma if gamma is not None else env.gamma
    rng = np.random.default_rng(seed)
    alpha_fn: Schedule = alpha if callable(alpha) else (lambda n: float(alpha))  # type: ignore[arg-type]
    eps_fn: Schedule = epsilon if callable(epsilon) else (lambda n: float(epsilon))  # type: ignore[arg-type]

    Q1 = np.zeros((env.n_states, env.n_actions))
    Q2 = np.zeros((env.n_states, env.n_actions))
    N1 = np.zeros((env.n_states, env.n_actions), dtype=np.int64)
    N2 = np.zeros((env.n_states, env.n_actions), dtype=np.int64)
    episode_returns: List[float] = []
    eval_history: List[Tuple[int, Any]] = []
    total_steps = 0

    for ep in range(n_episodes):
        s = env.reset()
        eps = float(eps_fn(ep + 1))
        ep_return = 0.0
        discount = 1.0

        for _ in range(max_steps):
            if env.is_terminal(s):
                break

            Q_avg = 0.5 * (Q1 + Q2)
            a = epsilon_greedy_action(Q_avg, s, env, eps, rng)

            s_next, r, done = env.step(a)
            total_steps += 1

            acts_next = env.allowed_actions[s_next]

            if rng.random() < 0.5:
                # Aktualisiere Q1, evaluiere mit Q2
                a_star = acts_next[int(np.argmax(Q1[s_next, acts_next]))]
                N1[s, a] += 1
                lr = alpha_fn(int(N1[s, a]))
                Q1[s, a] += lr * (r + gamma * Q2[s_next, a_star] - Q1[s, a])
            else:
                # Aktualisiere Q2, evaluiere mit Q1
                a_star = acts_next[int(np.argmax(Q2[s_next, acts_next]))]
                N2[s, a] += 1
                lr = alpha_fn(int(N2[s, a]))
                Q2[s, a] += lr * (r + gamma * Q1[s_next, a_star] - Q2[s, a])

            ep_return += discount * r
            discount *= gamma
            s = s_next
            if done:
                break

        episode_returns.append(ep_return)

        if eval_every is not None and eval_fn is not None and (ep + 1) % eval_every == 0:
            Q_avg_ep = 0.5 * (Q1 + Q2)
            eval_history.append((ep + 1, eval_fn(Q_avg_ep)))

    Q_avg_final = 0.5 * (Q1 + Q2)
    return Q1, Q2, Q_avg_final, {
        "episodes": n_episodes,
        "total_steps": total_steps,
        "visit_counts_Q1": N1.copy(),
        "visit_counts_Q2": N2.copy(),
        "episode_returns": episode_returns,
        "eval_history": eval_history,
    }
