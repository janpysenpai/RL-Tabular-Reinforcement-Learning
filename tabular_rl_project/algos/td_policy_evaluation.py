"""Sample-basierte (TD(0)) Policy Evaluation (Skript Algorithmen 17 & 18).

Totally asynchrone In-Place-Updates nach jedem Übergang (kein Batch):
    V(S_t) ← V(S_t) + α_n · (R + γ · V(S_{t+1}) − V(S_t))          [Alg. 17]
    Q(S_t,A_t) ← Q(S_t,A_t) + α_n · (R + γ · V_π(S_{t+1}) − Q(S_t,A_t))
        mit V_π(s') = Σ_{a'} π(a'|s') · Q(s',a')                     [Alg. 18]

alpha kann ein float (konstante Schrittweite) oder ein Schedule-Callable
``(n: int) → float`` sein; n ist der per-Zustand- (V-Version) bzw.
per-(s,a)-Besuchszähler (Q-Version).

Konvergenz: Mit Robbins-Monro-Schedules (z.B. one_over_n()) konvergiert
TD(0) fast sicher gegen V^π bzw. Q^π. Konstante alpha konvergieren gegen
einen Fixpunkt mit Bias O(alpha · σ²).

Aus Übungsblatt 6 Aufgabe 4 b).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..envs.mdp_base import FiniteMDP
from .schedules import Schedule


def td0_policy_evaluation_V(
    env: FiniteMDP,
    policy: np.ndarray,
    alpha: Union[float, Schedule],
    gamma: Optional[float] = None,
    n_episodes: int = 1000,
    max_steps: int = 1000,
    seed: Optional[int] = None,
    snapshot_every: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """TD(0) Policy Evaluation für V (Skript Algorithmus 17).

    Parameters
    ----------
    env : FiniteMDP
        Umgebung; Transitionen via env.step().
    policy : np.ndarray
        Stochastische Policy, Shape (S, A).
    alpha : float or Schedule
        Schrittweite: float → konstant; Callable (n) → float →
        per-Zustand-Besuchszähler (Robbins-Monro möglich).
    gamma : float, optional
        Diskontierung; None → env.gamma.
    n_episodes : int
        Anzahl der Trainings-Episoden.
    max_steps : int
        Maximale Episodenlänge.
    seed : int, optional
        Seed für np.random.default_rng.
    snapshot_every : int, optional
        Snapshots von V alle snapshot_every Episoden in info["snapshots"].

    Returns
    -------
    V : np.ndarray
        Geschätzte Wertfunktion V^π, Shape (S,).
    info : dict
        "episodes", "total_steps", "visit_counts", "snapshots".
    """
    gamma = gamma if gamma is not None else env.gamma
    rng = np.random.default_rng(seed)
    alpha_fn: Schedule = alpha if callable(alpha) else (lambda n: float(alpha))  # type: ignore[arg-type]

    V = np.zeros(env.n_states)
    N = np.zeros(env.n_states, dtype=np.int64)
    total_steps = 0
    snapshots: List[Tuple[int, np.ndarray]] = []

    for ep in range(n_episodes):
        s = env.reset()
        for _ in range(max_steps):
            if env.is_terminal(s):
                break
            acts = env.allowed_actions[s]
            probs = policy[s, acts]
            probs = probs / probs.sum()
            a = int(rng.choice(acts, p=probs))
            s_next, r, done = env.step(a)
            total_steps += 1

            N[s] += 1
            lr = alpha_fn(int(N[s]))
            V[s] += lr * (r + gamma * V[s_next] - V[s])

            s = s_next
            if done:
                break

        if snapshot_every is not None and (ep + 1) % snapshot_every == 0:
            snapshots.append((ep + 1, V.copy()))

    info: Dict = {
        "episodes": n_episodes,
        "total_steps": total_steps,
        "visit_counts": N.copy(),
    }
    if snapshot_every is not None:
        info["snapshots"] = snapshots

    return V, info


def td0_policy_evaluation_Q(
    env: FiniteMDP,
    policy: np.ndarray,
    alpha: Union[float, Schedule],
    gamma: Optional[float] = None,
    n_episodes: int = 1000,
    max_steps: int = 1000,
    seed: Optional[int] = None,
    snapshot_every: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """TD(0) Policy Evaluation für Q (Skript Algorithmus 18).

    Q-Update via erwartetem Bootstrap über die Zielpolicy:
        Q(s,a) ← Q(s,a) + α · (r + γ · Σ_{a'} π(a'|s') · Q(s',a') − Q(s,a))

    Dies entspricht Expected-SARSA für Policy Evaluation (nicht Control).
    alpha ist per-(s,a)-Besuchszähler.

    Parameters identisch zu td0_policy_evaluation_V.

    Returns
    -------
    Q : np.ndarray
        Geschätzte Aktionswertfunktion Q^π, Shape (S, A).
    info : dict
        Wie td0_policy_evaluation_V; "visit_counts" hat Shape (S, A).
    """
    gamma = gamma if gamma is not None else env.gamma
    rng = np.random.default_rng(seed)
    alpha_fn: Schedule = alpha if callable(alpha) else (lambda n: float(alpha))  # type: ignore[arg-type]

    Q = np.zeros((env.n_states, env.n_actions))
    N = np.zeros((env.n_states, env.n_actions), dtype=np.int64)
    total_steps = 0
    snapshots: List[Tuple[int, np.ndarray]] = []

    for ep in range(n_episodes):
        s = env.reset()
        for _ in range(max_steps):
            if env.is_terminal(s):
                break
            acts = env.allowed_actions[s]
            probs = policy[s, acts]
            probs = probs / probs.sum()
            a = int(rng.choice(acts, p=probs))
            s_next, r, done = env.step(a)
            total_steps += 1

            N[s, a] += 1
            lr = alpha_fn(int(N[s, a]))
            v_next = float((policy[s_next] * Q[s_next]).sum())
            Q[s, a] += lr * (r + gamma * v_next - Q[s, a])

            s = s_next
            if done:
                break

        if snapshot_every is not None and (ep + 1) % snapshot_every == 0:
            snapshots.append((ep + 1, Q.copy()))

    info: Dict = {
        "episodes": n_episodes,
        "total_steps": total_steps,
        "visit_counts": N.copy(),
    }
    if snapshot_every is not None:
        info["snapshots"] = snapshots

    return Q, info
