"""Mini-MDP zur algorithmischen Validierung (nicht für Experimente).

3 Zustände {0, 1, 2}, 2 Aktionen {0, 1}, gamma=0.9.

Übergänge (deterministisch):
    s=0, a=0 → s=0, r=0
    s=0, a=1 → s=1, r=0
    s=1, a=0 → s=0, r=0
    s=1, a=1 → s=2, r=1   (Zustand 2 ist absorbierend/terminal)
    s=2, *   → s=2, r=0   (absorbierend)

Analytische Lösung (Bellman-Optimalitätsgleichung, gamma=0.9):

    V*(2) = 0       (terminal)
    V*(1) = 1.0     (a=1 optimal: r=1 + 0.9*V*(2) = 1.0)
    V*(0) = 0.9     (a=1 optimal: 0 + 0.9*V*(1) = 0.9)

    Q*(s, a):
        Q*(0, 0) = 0 + 0.9 * V*(0) = 0.81
        Q*(0, 1) = 0 + 0.9 * V*(1) = 0.90
        Q*(1, 0) = 0 + 0.9 * V*(0) = 0.81
        Q*(1, 1) = 1 + 0.9 * V*(2) = 1.00
        Q*(2, *)             = 0.00

Optimale Policy: pi*(0)=1, pi*(1)=1 (jeweils Aktion 1).
"""

from __future__ import annotations

import numpy as np

from .mdp_base import FiniteMDP

# Referenzwerte für Tests
V_STAR = np.array([0.9, 1.0, 0.0])
Q_STAR = np.array([[0.81, 0.90], [0.81, 1.00], [0.00, 0.00]])


class _ValidationMDP(FiniteMDP):
    """Minimale FiniteMDP-Subklasse; Matrizen werden direkt gesetzt."""

    def _build_transition_matrix(self) -> None:
        pass  # Daten werden im Konstruktor direkt gesetzt


def build_validation_mdp() -> FiniteMDP:
    """Gibt das vorberechnete 3-Zustands-MDP zurück."""
    S, A = 3, 2

    P = np.zeros((S, A, S))
    R = np.zeros((S, A))

    # s=0
    P[0, 0, 0] = 1.0; R[0, 0] = 0.0
    P[0, 1, 1] = 1.0; R[0, 1] = 0.0
    # s=1
    P[1, 0, 0] = 1.0; R[1, 0] = 0.0
    P[1, 1, 2] = 1.0; R[1, 1] = 1.0
    # s=2: terminal, absorbierend
    P[2, 0, 2] = 1.0
    P[2, 1, 2] = 1.0

    env = _ValidationMDP(gamma=0.9)
    env.states = list(range(S))
    env.actions = list(range(A))
    env.start_state = 0
    env.terminal_states = {2}
    env.allowed_actions = {s: list(range(A)) for s in range(S)}
    env.transition_probabilities = P
    env.expected_rewards = R
    env._current_state = 0

    return env
