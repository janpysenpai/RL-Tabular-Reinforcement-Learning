"""Iterative Policy Evaluation (Skript Algorithm 7 + Algorithm 8 async).

V-Version: Bellman-Erwartungsoperator T^pi auf V.
Q-Version: T^pi auf Q.

Konvergenzgarantie für gamma < 1 (Theorem 3.3.2). Für gamma = 1 kann der
Algorithmus divergieren — Gegenbeispiel in Übungsblatt 5 Aufgabe 1.

Optionen:
    sync  True  = synchron (Algorithm 7): alle Zustände mit alten Werten.
          False = totally asynchronous (Algorithm 8): sofortige In-Place-Updates.
    tol   None  = kein Toleranzabbruch (läuft exakt max_iter Schritte).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..envs.mdp_base import FiniteMDP


def ipe_V(
    env: FiniteMDP,
    policy: np.ndarray,
    gamma: Optional[float] = None,
    sync: bool = True,
    max_iter: int = 1000,
    tol: Optional[float] = 1e-9,
) -> Tuple[np.ndarray, Dict]:
    """Iterative Policy Evaluation für V (Skript Algorithm 7 / 8).

    Args:
        env:      Finite MDP mit bekannter Dynamik.
        policy:   Stochastische Policy, Shape (S, A), Zeilen summieren zu 1.
        gamma:    Diskontierung; None → env.gamma.
        sync:     True = synchron, False = totally asynchronous.
        max_iter: Maximale Iterationszahl.
        tol:      Abbruchtoleranz ||V_{k+1}-V_k||_inf; None = kein Toleranzabbruch.

    Returns:
        V:    V^pi, Shape (S,).
        info: {"iterations": int, "delta_history": list}.
    """
    gamma = gamma if gamma is not None else env.gamma
    P = env.transition_probabilities
    R = env.expected_rewards

    V = np.zeros(env.n_states)
    delta_history: List[float] = []

    for iteration in range(max_iter):
        if sync:
            # Q_pi(s,a) = R(s,a) + γ * Σ_{s'} P(s,a,s') * V(s')
            Q = R + gamma * np.einsum("sap,p->sa", P, V)
            # V_pi(s) = Σ_a pi(a|s) * Q_pi(s,a)
            V_new = (policy * Q).sum(axis=1)
            delta = float(np.abs(V_new - V).max())
            V = V_new
        else:
            delta = 0.0
            for s in env.states:
                v_old = V[s]
                q_s = R[s] + gamma * (P[s] @ V)
                V[s] = float((policy[s] * q_s).sum())
                delta = max(delta, abs(V[s] - v_old))

        delta_history.append(delta)
        if tol is not None and delta < tol:
            break

    return V, {"iterations": iteration + 1, "delta_history": delta_history}


def ipe_Q(
    env: FiniteMDP,
    policy: np.ndarray,
    gamma: Optional[float] = None,
    sync: bool = True,
    max_iter: int = 1000,
    tol: Optional[float] = 1e-9,
) -> Tuple[np.ndarray, Dict]:
    """Iterative Policy Evaluation für Q.

    Q_pi_new(s,a) = R(s,a) + γ * Σ_{s'} P(s,a,s') * Σ_{a'} pi(a'|s') * Q_pi(s',a').

    Args und Returns analog zu ipe_V; gibt Q shape (S, A) zurück.
    """
    gamma = gamma if gamma is not None else env.gamma
    P = env.transition_probabilities
    R = env.expected_rewards

    Q = np.zeros((env.n_states, env.n_actions))
    delta_history: List[float] = []

    for iteration in range(max_iter):
        if sync:
            # V_pi(s') = Σ_{a'} pi(a'|s') * Q(s',a')
            V_pi = (policy * Q).sum(axis=1)             # (S,)
            Q_new = R + gamma * np.einsum("sap,p->sa", P, V_pi)
            delta = float(np.abs(Q_new - Q).max())
            Q = Q_new
        else:
            delta = 0.0
            for s in env.states:
                q_old = Q[s].copy()
                V_pi = (policy * Q).sum(axis=1)
                Q[s] = R[s] + gamma * (P[s] @ V_pi)
                delta = max(delta, float(np.abs(Q[s] - q_old).max()))

        delta_history.append(delta)
        if tol is not None and delta < tol:
            break

    return Q, {"iterations": iteration + 1, "delta_history": delta_history}
