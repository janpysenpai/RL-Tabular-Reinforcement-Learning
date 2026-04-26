"""Sammlung kleiner, analytisch lösbarer MDPs für Algorithmus-Tests.

Jede Factory-Funktion gibt einen FiniteMDP zurück und dokumentiert V*.

    make_two_state_loop  — langer Horizont / hohes Gamma
    make_sparse_chain    — Sparse Reward / Exploration
    make_noisy_gridworld — Stochastische Übergänge (Slip)
    make_bias_mdp_for_double_q — Overestimation-Bias (Jensen-Ungleichung)

Der Bias-MDP wird auch in test_sample_based_control.py verwendet; er ist
hier zentralisiert, damit der Test-Code direkt importieren kann.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .mdp_base import FiniteMDP


# ------------------------------------------------------------------
# Hilfsbasis: einfacher nicht-abstrakter FiniteMDP
# ------------------------------------------------------------------

class _SimpleMDP(FiniteMDP):
    """Interner FiniteMDP, bei dem _build_transition_matrix leer ist
    (Daten werden extern direkt gesetzt)."""

    def _build_transition_matrix(self) -> None:
        pass  # Matrizen werden vom Konstruktor der Factory direkt gesetzt


# ------------------------------------------------------------------
# a) Zwei-Zustands-Loop
# ------------------------------------------------------------------

def make_two_state_loop(p_stay: float = 0.9, gamma: float = 0.99) -> FiniteMDP:
    """Zwei-Zustands-MDP mit einer Aktion — stresst langen Horizont / hohes Gamma.

    Zustände:
        s=0: nicht-terminal, eine Aktion
            P(0→0) = p_stay, P(0→1) = 1-p_stay, R=+1
        s=1: terminal (absorbierend), R=0

    Analytische Lösung (Bellman-Optimalitätsgleichung):
        V*(1) = 0
        V*(0) = 1 + gamma * p_stay * V*(0)
        V*(0) = 1 / (1 - gamma * p_stay)

    Beispiel: p_stay=0.9, gamma=0.99 → V*(0) ≈ 9.17.

    Args:
        p_stay: Wahrscheinlichkeit, in s=0 zu bleiben (0 < p_stay < 1).
        gamma:  Diskontierungsfaktor.

    Returns:
        FiniteMDP mit 2 Zuständen und 1 Aktion.
    """
    S, A = 2, 1
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))

    # s=0, a=0
    P[0, 0, 0] = p_stay
    P[0, 0, 1] = 1.0 - p_stay
    R[0, 0] = 1.0

    # s=1: terminal, absorbierend
    P[1, 0, 1] = 1.0
    R[1, 0] = 0.0

    env = _SimpleMDP(gamma=gamma)
    env.states = [0, 1]
    env.actions = [0]
    env.start_state = 0
    env.terminal_states = {1}
    env.allowed_actions = {0: [0], 1: [0]}
    env.transition_probabilities = P
    env.expected_rewards = R
    env._current_state = 0
    return env


# ------------------------------------------------------------------
# b) Sparse-Reward-Kette
# ------------------------------------------------------------------

def make_sparse_chain(n_states: int = 10, gamma: float = 0.95) -> FiniteMDP:
    """Lineare Kette mit Sparse Reward — stresst Exploration.

    Zustände: {0, 1, ..., n_states-1}
    Aktionen: 0=links, 1=rechts
        - rechts: s → min(s+1, n_states-1)
        - links:  s → max(s-1, 0)
    Reward: 0 überall, außer +1 beim Betreten von s=n_states-1 (terminal).
    s=n_states-1 ist Terminal (absorbierend).

    Analytische Lösung (Greedy-Opt.-Policy: immer rechts):
        V*(n_states-1) = 0   (terminal)
        V*(s)          = gamma^(n_states-1-s)   für s < n_states-1

    Args:
        n_states: Anzahl Zustände (incl. Terminal).
        gamma:    Diskontierungsfaktor.

    Returns:
        FiniteMDP mit n_states Zuständen und 2 Aktionen.
    """
    S, A = n_states, 2
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    terminal = n_states - 1

    for s in range(S):
        if s == terminal:
            P[s, 0, s] = 1.0
            P[s, 1, s] = 1.0
            continue
        # links (a=0)
        s_left = max(s - 1, 0)
        P[s, 0, s_left] = 1.0
        R[s, 0] = 0.0

        # rechts (a=1)
        s_right = s + 1
        P[s, 1, s_right] = 1.0
        R[s, 1] = 1.0 if s_right == terminal else 0.0

    env = _SimpleMDP(gamma=gamma)
    env.states = list(range(S))
    env.actions = [0, 1]
    env.start_state = 0
    env.terminal_states = {terminal}
    env.allowed_actions = {s: [0, 1] for s in range(S)}
    env.transition_probabilities = P
    env.expected_rewards = R
    env._current_state = 0
    return env


# ------------------------------------------------------------------
# c) Verrauschte GridWorld (Slip)
# ------------------------------------------------------------------

def make_noisy_gridworld(
    rows: int = 5,
    cols: int = 5,
    slip_prob: float = 0.2,
    gamma: float = 0.9,
) -> FiniteMDP:
    """Stochastische GridWorld mit Slip — stresst Stochastizität.

    Mit Wahrscheinlichkeit slip_prob wird die gewählte Aktion durch eine
    zufällige (uniform über alle 4 Aktionen) ersetzt. Ziel in der unteren
    rechten Ecke (rows-1, cols-1) mit Reward +1. wall_behavior="stay".

    Übergänge:
        P(s, a, s') = (1-slip_prob)*P_det(s,a,s')
                    + slip_prob * (1/4) * sum_{a'} P_det(s,a',s')

    Analytische V* nicht geschlossen — wird via value_iteration bestimmt.

    Args:
        rows, cols:  Gitterdimensionen.
        slip_prob:   Wahrscheinlichkeit für zufällige Aktion (0..1).
        gamma:       Diskontierungsfaktor.

    Returns:
        FiniteMDP mit rows*cols Zuständen und 4 Aktionen.
    """
    DELTA: Dict[int, Tuple[int, int]] = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def _rc(r: int, c: int) -> int:
        return r * cols + c

    def _in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    S, A = rows * cols, 4
    terminal = _rc(rows - 1, cols - 1)

    # Deterministische Übergangsmatrix P_det[s, a, s']
    P_det = np.zeros((S, A, S))
    R_det = np.zeros((S, A))

    for s in range(S):
        r, c = divmod(s, cols)
        if s == terminal:
            for a in range(A):
                P_det[s, a, s] = 1.0
            continue
        for a in range(A):
            dr, dc = DELTA[a]
            nr, nc = r + dr, c + dc
            ns = _rc(nr, nc) if _in_bounds(nr, nc) else s
            P_det[s, a, ns] = 1.0
            R_det[s, a] = 1.0 if ns == terminal else 0.0

    # Stochastische Übergangsmatrix: Slip-Mischung
    P_slip = (
        (1.0 - slip_prob) * P_det
        + slip_prob * P_det.mean(axis=1, keepdims=True)
    )
    # Expected Rewards: gleiche Mischung
    R_slip = (
        (1.0 - slip_prob) * R_det
        + slip_prob * R_det.mean(axis=1, keepdims=True)
    )

    env = _SimpleMDP(gamma=gamma)
    env.states = list(range(S))
    env.actions = list(range(A))
    env.start_state = _rc(0, 0)
    env.terminal_states = {terminal}
    env.allowed_actions = {s: list(range(A)) for s in range(S)}
    env.transition_probabilities = P_slip
    env.expected_rewards = R_slip
    env._current_state = env.start_state
    return env


# ------------------------------------------------------------------
# d) Bias-MDP für Double Q-Learning
# ------------------------------------------------------------------

class BiasedMDP(FiniteMDP):
    """Zwei-Zustands-MDP mit normal-verteiltem Reward — zeigt Jensen-Bias.

    Zustand 0 → Zustand 1 (terminal) mit jeder Aktion.
    Reward R ~ N(reward_mean, reward_std).
    Wahre Q*(0, a) = reward_mean für alle a.

    Der Overestimation-Bias entsteht weil E[max_a Q_hat(0,a)] > reward_mean
    bei endlich vielen Samples (Jensen-Ungleichung, E[max] ≥ max[E]).
    Double Q-Learning entkoppelt Selektion und Evaluation und reduziert ihn.

    Analytische Lösung:
        V*(0) = Q*(0, a) = reward_mean   (für alle a)
        V*(1) = 0                         (terminal)
    """

    def __init__(
        self,
        n_acts: int = 10,
        reward_mean: float = -0.1,
        reward_std: float = 1.0,
        gamma: float = 0.9,
        env_seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            n_acts:       Anzahl Aktionen in Zustand 0.
            reward_mean:  Erwarteter Reward (wahres Q*(0,a)).
            reward_std:   Standardabweichung des Reward-Rauschens.
            gamma:        Diskontierungsfaktor.
            env_seed:     Seed für den internen Reward-RNG.
        """
        super().__init__(gamma)
        self._env_rng = np.random.default_rng(env_seed)
        self._reward_mean = reward_mean
        self._reward_std = reward_std

        self.states = [0, 1]
        self.actions = list(range(n_acts))
        self.start_state = 0
        self.terminal_states = {1}
        self.allowed_actions = {0: list(range(n_acts)), 1: [0]}
        self._build_transition_matrix()
        self._current_state = 0

    def _build_transition_matrix(self) -> None:
        n_acts = len(self.actions)
        P = np.zeros((2, n_acts, 2))
        R = np.zeros((2, n_acts))
        # s=0: jede Aktion führt zu s=1
        P[0, :, 1] = 1.0
        R[0, :] = self._reward_mean
        # s=1: terminal, absorbierend
        P[1, :, 1] = 1.0
        self.transition_probabilities = P
        self.expected_rewards = R

    def _sample_reward(self, state: int, action: int, next_state: int) -> float:
        """Stochastischer N(mean, std)-Reward aus Zustand 0."""
        if state == 0:
            return float(self._env_rng.normal(self._reward_mean, self._reward_std))
        return 0.0


def make_bias_mdp_for_double_q(
    n_acts: int = 10,
    reward_mean: float = -0.1,
    reward_std: float = 1.0,
    gamma: float = 0.9,
    env_seed: Optional[int] = None,
) -> BiasedMDP:
    """Factory für den BiasedMDP (Overestimation-Bias-Demo).

    Analytische Lösung:
        V*(0) = reward_mean,  V*(1) = 0,
        Q*(0, a) = reward_mean für alle a.

    Args:
        n_acts:       Anzahl Aktionen.
        reward_mean:  Wahres Q*(0, a).
        reward_std:   Reward-Standardabweichung.
        gamma:        Diskontierungsfaktor.
        env_seed:     Seed für Reward-Sampling.

    Returns:
        BiasedMDP-Instanz.
    """
    return BiasedMDP(
        n_acts=n_acts,
        reward_mean=reward_mean,
        reward_std=reward_std,
        gamma=gamma,
        env_seed=env_seed,
    )
