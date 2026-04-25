"""Dynamic Programming für finite-time MDPs (Skript Algorithms 12 & 13).

Übungsblatt 5 Aufgabe 6: Backward Induction für endlichen Horizont T.

WICHTIG: Finite-time DP arbeitet OHNE Diskontierung (gamma = 1 implizit).
Der Diskontierungsfaktor env.gamma wird hier nicht verwendet.

Algorithmus 12 — finite_time_policy_evaluation:
    Berechnet V^pi_t(s) rückwärts von t=T bis t=0.

Algorithmus 13 — finite_time_optimal_control:
    Berechnet V*_t(s) und optimale Policy pi*_t(s) per Backward Induction.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..envs.mdp_base import FiniteMDP


def finite_time_policy_evaluation(
    env: FiniteMDP,
    policy: np.ndarray,
    T: int,
    terminal_value: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Policy Evaluation für endlichen Horizont T (Skript Algorithm 12).

    Keine Diskontierung — gamma = 1 implizit.

    V_t(s) = Σ_a pi_t(a|s) * [R(s,a) + Σ_{s'} P(s,a,s') * V_{t+1}(s')]

    Args:
        env:            Finite MDP.
        policy:         Stationäre Policy, Shape (S, A). Für zeitabhängige Policies
                        kann policy[t] übergeben werden (dann Shape (T, S, A)).
        T:              Zeithorizont.
        terminal_value: V_T(s), Shape (S,); None → 0 für alle s.

    Returns:
        V: Shape (T+1, S). V[t, s] = V^pi_t(s).
    """
    S = env.n_states
    P = env.transition_probabilities
    R = env.expected_rewards

    V = np.zeros((T + 1, S))
    if terminal_value is not None:
        V[T] = terminal_value

    # Stationäre oder zeitabhängige Policy erkennen
    time_dependent = policy.ndim == 3  # Shape (T, S, A)

    for t in range(T - 1, -1, -1):
        pi_t = policy[t] if time_dependent else policy
        # Q_t(s,a) = R(s,a) + Σ_{s'} P(s,a,s') * V_{t+1}(s')
        Q_t = R + np.einsum("sap,p->sa", P, V[t + 1])
        V[t] = (pi_t * Q_t).sum(axis=1)

    return V


def finite_time_optimal_control(
    env: FiniteMDP,
    T: int,
    terminal_value: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimale Steuerung für endlichen Horizont T (Skript Algorithm 13).

    Keine Diskontierung — gamma = 1 implizit.

    V*_t(s) = max_a [R(s,a) + Σ_{s'} P(s,a,s') * V*_{t+1}(s')]

    Args:
        env:            Finite MDP.
        T:              Zeithorizont.
        terminal_value: V*_T(s), Shape (S,); None → 0 für alle s.

    Returns:
        V:      Shape (T+1, S). V[t, s] = V*_t(s).
        policy: Shape (T, S), dtype int. policy[t, s] = optimale Aktion in (t, s).
    """
    S = env.n_states
    P = env.transition_probabilities
    R = env.expected_rewards

    V = np.zeros((T + 1, S))
    policy = np.zeros((T, S), dtype=int)

    if terminal_value is not None:
        V[T] = terminal_value

    for t in range(T - 1, -1, -1):
        Q_t = R + np.einsum("sap,p->sa", P, V[t + 1])
        V[t] = Q_t.max(axis=1)
        policy[t] = Q_t.argmax(axis=1)

    return V, policy
