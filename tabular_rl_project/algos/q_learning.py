"""Q-Learning (Skript Algorithm 18 / 19, Übungsblatt 6 Aufgabe 4 c)).

Off-policy stochastische Kontrolle. Verhaltenspolicy kann epsilon-greedy,
uniform oder ein beliebiger Callable sein; die Zielpolicy ist immer greedy:

    Q(s,a) ← Q(s,a) + α_n · (r + γ · max_{a'} Q(s',a') − Q(s,a))

Da der Zielpolicy-Term unabhängig von der Verhaltenspolicy ist, konvergiert
Q-Learning (unter Robbins-Monro) gegen Q* (off-policy, Satz 4.x).

alpha und epsilon dürfen float (konstant) oder Schedule ``(n:int)→float`` sein.
    - alpha:   per-(s,a)-Besuchszähler
    - epsilon: per-Episoden-Zähler (globaler Schrittzähler optional)

Mehrdeutigkeit: Ob epsilon per Episode oder per Schritt abfällt, beeinflusst
die Konvergenzgeschwindigkeit. Hier: per Episode (n = ep+1 bei ep in 0..N-1).

Bezug Submission:
    - Blatt 8 Aufgabe 4 d)  Stepsize/Exploration-Schedules
    - Blatt 8 Aufgabe 4 f)  optimale Hyperparameter im 4x4-Grid
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..envs.mdp_base import FiniteMDP
from .exploration import epsilon_greedy_action
from .schedules import Schedule


def q_learning(
    env: FiniteMDP,
    alpha: Union[float, Schedule],
    epsilon: Union[float, Schedule] = 0.1,
    gamma: Optional[float] = None,
    behaviour_policy: Union[str, Callable] = "eps_greedy",
    n_episodes: int = 1000,
    max_steps: int = 1000,
    seed: Optional[int] = None,
    eval_every: Optional[int] = None,
    eval_fn: Optional[Callable[[np.ndarray], Any]] = None,
) -> Tuple[np.ndarray, Dict]:
    """Q-Learning (Skript Alg. 18 / 19).

    Parameters
    ----------
    env : FiniteMDP
        Umgebung; Transitionen via env.step().
    alpha : float or Schedule
        Schrittweite: float → konstant; Callable (n)→float → per-(s,a)-Besuch.
    epsilon : float or Schedule
        Explorationsrate für "eps_greedy": float → konstant;
        Callable (ep)→float → per-Episode (ep=1 für erste Episode).
    gamma : float, optional
        Diskontierung; None → env.gamma.
    behaviour_policy : {"eps_greedy", "uniform"} or Callable
        Verhaltenspolicy:
            "eps_greedy" → epsilon_greedy_action(Q, s, env, eps, rng)
            "uniform"    → uniform random aus allowed_actions
            Callable     → ``(Q, s, env, rng) → int``
    n_episodes : int
        Anzahl Trainings-Episoden.
    max_steps : int
        Maximale Episodenlänge.
    seed : int, optional
        Seed für np.random.default_rng.
    eval_every : int, optional
        Falls gesetzt: alle eval_every Episoden wird eval_fn(Q) aufgerufen.
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
        ep_return = 0.0
        discount = 1.0

        for _ in range(max_steps):
            if env.is_terminal(s):
                break

            if behaviour_policy == "eps_greedy":
                a = epsilon_greedy_action(Q, s, env, eps, rng)
            elif behaviour_policy == "uniform":
                a = int(rng.choice(env.allowed_actions[s]))
            else:
                a = behaviour_policy(Q, s, env, rng)

            s_next, r, done = env.step(a)
            total_steps += 1

            N[s, a] += 1
            lr = alpha_fn(int(N[s, a]))
            # max Q(s',a') über erlaubte Aktionen; Q[terminal,:] = 0 per Init
            acts_next = env.allowed_actions[s_next]
            max_q_next = float(Q[s_next, acts_next].max())
            Q[s, a] += lr * (r + gamma * max_q_next - Q[s, a])

            ep_return += discount * r
            discount *= gamma
            s = s_next
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
