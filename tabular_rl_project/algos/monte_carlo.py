"""Monte-Carlo Policy Evaluation (Skript Algorithmen 14 & 15; Blatt 6 Alg. 1).

Algorithmus 14 (every-visit): jeder Besuch von s (bzw. (s,a)) in einer
Episode geht in die Schätzung ein.

Algorithmus 15 (first-visit): nur der erste Besuch von s (bzw. (s,a))
pro Episode.

Inkrementelles Mittel (Blatt 6, Alg. 1):
    V(s) ← V(s) + (1/N(s)) · (G_t − V(s))
wobei N(s) der Besuchszähler von s ist. Kein explizites alpha erforderlich.

Diskontierung: gamma=None → env.gamma.

Mehrdeutigkeit: Ein Schritt (s, a, r) bedeutet hier "von s Aktion a
ausführen → Reward r erhalten → in s' landen". Der Return G_t akkumuliert
ab Reward r_t aufwärts: G_t = r_t + γ·r_{t+1} + ... Reward des
Terminalzustands selbst ist 0 (absorbierend).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..envs.mdp_base import FiniteMDP


def _generate_episode(
    env: FiniteMDP,
    policy: np.ndarray,
    rng: np.random.Generator,
    max_steps: int,
) -> List[Tuple[int, int, float]]:
    """Generiert eine Episode als Liste von (state, action, reward)-Tupeln.

    Endet beim ersten Erreichen eines Terminalzustands oder nach max_steps.
    """
    s = env.reset()
    trajectory: List[Tuple[int, int, float]] = []
    for _ in range(max_steps):
        if env.is_terminal(s):
            break
        acts = env.allowed_actions[s]
        probs = policy[s, acts]
        probs = probs / probs.sum()  # Normalisierung gegen Rundungsfehler
        a = int(rng.choice(acts, p=probs))
        s_next, r, done = env.step(a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory


def _compute_returns(rewards: List[float], gamma: float) -> np.ndarray:
    """Berechnet G_t = Σ_{k=0}^{T-1-t} γ^k · r_{t+k} für alle t rückwärts."""
    T = len(rewards)
    G = np.empty(T)
    g = 0.0
    for t in reversed(range(T)):
        g = rewards[t] + gamma * g
        G[t] = g
    return G


def mc_policy_evaluation_V(
    env: FiniteMDP,
    policy: np.ndarray,
    gamma: Optional[float] = None,
    n_episodes: int = 1000,
    first_visit: bool = True,
    max_steps: int = 1000,
    seed: Optional[int] = None,
    snapshot_every: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """Monte-Carlo Policy Evaluation für V (Skript Alg. 14 / 15).

    Parameters
    ----------
    env : FiniteMDP
        Umgebung; Episoden via env.reset() / env.step().
    policy : np.ndarray
        Stochastische Policy, Shape (S, A), Zeilen über erlaubte Aktionen = 1.
    gamma : float, optional
        Diskontierung; None → env.gamma.
    n_episodes : int
        Anzahl der Trainings-Episoden.
    first_visit : bool
        True = First-Visit MC (Alg. 15), False = Every-Visit MC (Alg. 14).
    max_steps : int
        Maximale Episodenlänge (Abbruch ohne Terminal-State führt zu
        abgeschnittenen Returns; vernachlässigbar für max_steps >> E[T]).
    seed : int, optional
        Seed für np.random.default_rng; sichert Reproduzierbarkeit der
        Aktionsauswahl (Umgebungstransitionen nutzen globalen np-State).
    snapshot_every : int, optional
        Alle snapshot_every Episoden wird eine V-Kopie in
        info["snapshots"] abgelegt (Liste von (episode_idx, V)).

    Returns
    -------
    V : np.ndarray
        Geschätzte Wertfunktion V^π, Shape (S,).
    info : dict
        "episodes", "total_steps", "visit_counts" (Shape S,),
        "snapshots" (nur wenn snapshot_every gesetzt).
    """
    gamma = gamma if gamma is not None else env.gamma
    rng = np.random.default_rng(seed)

    V = np.zeros(env.n_states)
    N = np.zeros(env.n_states, dtype=np.int64)
    total_steps = 0
    snapshots: List[Tuple[int, np.ndarray]] = []

    for ep in range(n_episodes):
        trajectory = _generate_episode(env, policy, rng, max_steps)
        total_steps += len(trajectory)

        if not trajectory:
            continue

        rewards = [r for _, _, r in trajectory]
        G = _compute_returns(rewards, gamma)

        if first_visit:
            visited: set = set()
            for t, (s, _a, _r) in enumerate(trajectory):
                if s not in visited:
                    visited.add(s)
                    N[s] += 1
                    V[s] += (G[t] - V[s]) / N[s]
        else:
            for t, (s, _a, _r) in enumerate(trajectory):
                N[s] += 1
                V[s] += (G[t] - V[s]) / N[s]

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


def mc_policy_evaluation_Q(
    env: FiniteMDP,
    policy: np.ndarray,
    gamma: Optional[float] = None,
    n_episodes: int = 1000,
    first_visit: bool = True,
    max_steps: int = 1000,
    seed: Optional[int] = None,
    snapshot_every: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """Monte-Carlo Policy Evaluation für Q (analog Alg. 14/15 für (s,a)-Paare).

    First-Visit: erstes Auftreten jedes (s,a)-Paars pro Episode zählt.
    Every-Visit: alle Auftreten zählen.

    Parameters
    ----------
    env, policy, gamma, n_episodes, first_visit, max_steps, seed, snapshot_every
        Wie bei mc_policy_evaluation_V.

    Returns
    -------
    Q : np.ndarray
        Geschätzte Aktionswertfunktion Q^π, Shape (S, A).
    info : dict
        Wie mc_policy_evaluation_V; "visit_counts" hat Shape (S, A).
    """
    gamma = gamma if gamma is not None else env.gamma
    rng = np.random.default_rng(seed)

    Q = np.zeros((env.n_states, env.n_actions))
    N = np.zeros((env.n_states, env.n_actions), dtype=np.int64)
    total_steps = 0
    snapshots: List[Tuple[int, np.ndarray]] = []

    for ep in range(n_episodes):
        trajectory = _generate_episode(env, policy, rng, max_steps)
        total_steps += len(trajectory)

        if not trajectory:
            continue

        rewards = [r for _, _, r in trajectory]
        G = _compute_returns(rewards, gamma)

        if first_visit:
            visited_sa: set = set()
            for t, (s, a, _r) in enumerate(trajectory):
                key = (s, a)
                if key not in visited_sa:
                    visited_sa.add(key)
                    N[s, a] += 1
                    Q[s, a] += (G[t] - Q[s, a]) / N[s, a]
        else:
            for t, (s, a, _r) in enumerate(trajectory):
                N[s, a] += 1
                Q[s, a] += (G[t] - Q[s, a]) / N[s, a]

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
