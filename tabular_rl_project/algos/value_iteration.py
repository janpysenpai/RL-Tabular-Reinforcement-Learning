"""Value Iteration (Skript Algorithm 6, Übungsblatt 5 Aufgabe 4).

V-Version: wendet den Bellman-Optimalitätsoperator T* wiederholt auf V an.
Q-Version: wendet T* auf Q an (Bellman-Q-Operator).

Optionen:
    gamma    Diskontierung (None → env.gamma).
    sync     True  = synchrone Updates (alle Zustände mit alten Werten, Algorithm 6).
             False = totally asynchronous (In-Place, Algorithm-8-Stil).
    max_iter Maximale Iterationszahl.
    tol      Abbruchtoleranz für ||V_{k+1}-V_k||_inf; None = kein Toleranzabbruch.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..envs.mdp_base import FiniteMDP


# ------------------------------------------------------------------
# Bellman-Hilfsfunktionen (vektorisiert)
# ------------------------------------------------------------------

def _q_from_v(V: np.ndarray, P: np.ndarray, R: np.ndarray, gamma: float) -> np.ndarray:
    """Q(s,a) = R(s,a) + γ * Σ_{s'} P(s,a,s') * V(s').  Shape: (S, A)."""
    return R + gamma * np.einsum("sap,p->sa", P, V)


def greedy_policy_from_v(
    env: FiniteMDP, V: np.ndarray, gamma: Optional[float] = None
) -> np.ndarray:
    """Berechnet die Greedy-Policy aus V als (S, A)-Array (one-hot).

    Nur erlaubte Aktionen werden berücksichtigt.
    """
    gamma = gamma if gamma is not None else env.gamma
    Q = _q_from_v(V, env.transition_probabilities, env.expected_rewards, gamma)
    S, A = env.n_states, env.n_actions
    policy = np.zeros((S, A))
    for s in env.states:
        acts = env.allowed_actions[s]
        best = acts[int(np.argmax(Q[s, acts]))]
        policy[s, best] = 1.0
    return policy


def greedy_policy_from_q(env: FiniteMDP, Q: np.ndarray) -> np.ndarray:
    """Berechnet die Greedy-Policy direkt aus Q als (S, A)-Array (one-hot)."""
    S, A = env.n_states, env.n_actions
    policy = np.zeros((S, A))
    for s in env.states:
        acts = env.allowed_actions[s]
        best = acts[int(np.argmax(Q[s, acts]))]
        policy[s, best] = 1.0
    return policy


# ------------------------------------------------------------------
# Value Iteration — V-Version
# ------------------------------------------------------------------

def value_iteration_V(
    env: FiniteMDP,
    gamma: Optional[float] = None,
    sync: bool = True,
    max_iter: int = 1000,
    tol: Optional[float] = 1e-9,
) -> Tuple[np.ndarray, Dict]:
    """Value Iteration auf V (Skript Algorithm 6).

    Args:
        env:      Finite MDP mit bekannter Dynamik.
        gamma:    Diskontierung; None → env.gamma.
        sync:     True = synchron (Jacobi), False = totally async (Gauss-Seidel-Stil).
        max_iter: Maximale Iterationszahl.
        tol:      Abbruchtoleranz; None = kein Toleranzabbruch.

    Returns:
        V:    Optimale Wertfunktion, Shape (S,).
        info: {"iterations": int, "delta_history": list}.
    """
    gamma = gamma if gamma is not None else env.gamma
    P = env.transition_probabilities
    R = env.expected_rewards

    V = np.zeros(env.n_states)
    delta_history: List[float] = []

    for iteration in range(max_iter):
        if sync:
            Q = _q_from_v(V, P, R, gamma)
            V_new = Q.max(axis=1)
            delta = float(np.abs(V_new - V).max())
            V = V_new
        else:
            delta = 0.0
            for s in env.states:
                v_old = V[s]
                V[s] = float((R[s] + gamma * (P[s] @ V)).max())
                delta = max(delta, abs(V[s] - v_old))

        delta_history.append(delta)
        if tol is not None and delta < tol:
            break

    return V, {"iterations": iteration + 1, "delta_history": delta_history}


# ------------------------------------------------------------------
# Value Iteration — Q-Version
# ------------------------------------------------------------------

def value_iteration_Q(
    env: FiniteMDP,
    gamma: Optional[float] = None,
    sync: bool = True,
    max_iter: int = 1000,
    tol: Optional[float] = 1e-9,
) -> Tuple[np.ndarray, Dict]:
    """Value Iteration auf Q (Bellman-Q-Optimalitätsoperator).

    Q_new(s,a) = R(s,a) + γ * Σ_{s'} P(s,a,s') * max_{a'} Q(s',a').

    Args und Returns analog zu value_iteration_V; gibt Q shape (S, A) zurück.
    """
    gamma = gamma if gamma is not None else env.gamma
    P = env.transition_probabilities
    R = env.expected_rewards

    Q = np.zeros((env.n_states, env.n_actions))
    delta_history: List[float] = []

    for iteration in range(max_iter):
        if sync:
            max_Q = Q.max(axis=1)                    # (S,)
            Q_new = R + gamma * np.einsum("sap,p->sa", P, max_Q)
            delta = float(np.abs(Q_new - Q).max())
            Q = Q_new
        else:
            delta = 0.0
            for s in env.states:
                q_old = Q[s].copy()
                max_Q = Q.max(axis=1)
                Q[s] = R[s] + gamma * (P[s] @ max_Q)
                delta = max(delta, float(np.abs(Q[s] - q_old).max()))

        delta_history.append(delta)
        if tol is not None and delta < tol:
            break

    return Q, {"iterations": iteration + 1, "delta_history": delta_history}
