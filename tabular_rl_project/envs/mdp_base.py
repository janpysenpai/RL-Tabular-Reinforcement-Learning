"""Abstrakte Basisklasse für endliche MDPs.

Aus Übungsblatt 4 (Aufgabe 5): Jede Umgebung soll als Klasse mit
folgenden Attributen vorliegen, damit modellbasierte Algorithmen
(Value Iteration, Policy Evaluation, ...) sie direkt nutzen können:

    states                   Liste aller Zustandsindizes
    actions                  Liste aller Aktionsindizes
    allowed_actions          Dict s -> Liste erlaubter Aktionen in Zustand s
    transition_probabilities np.ndarray (S, A, S'),  P[s, a, s']
    expected_rewards         np.ndarray (S, A),      r[s, a] = E[R | s, a]
    start_state              int (Startindex)
    terminal_states          set of int
    gamma                    float, Diskontierungsfaktor

Spielmechanik:
    reset()                  -> int  (Startzustand)
    step(action)             -> (next_state, reward, done)
    mc_estimate_reward(s, a, n) -> float
    is_terminal(s)           -> bool
    validate()               prüft Zeilensummen von P
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


class FiniteMDP(ABC):
    """Abstrakte Basisklasse für endliche, diskontierte MDPs."""

    # Subklassen müssen diese Attribute im __init__ setzen, bevor
    # _build_transition_matrix aufgerufen wird.
    states: List[int]
    actions: List[int]
    allowed_actions: Dict[int, List[int]]
    transition_probabilities: np.ndarray  # Shape (S, A, S')
    expected_rewards: np.ndarray          # Shape (S, A)
    start_state: int
    terminal_states: set
    gamma: float

    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma
        self._current_state: int = 0

    # ------------------------------------------------------------------
    # Abstrakte Methode — von Subklassen zu implementieren
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_transition_matrix(self) -> None:
        """Befüllt transition_probabilities, expected_rewards und allowed_actions."""

    # ------------------------------------------------------------------
    # Spielmechanik
    # ------------------------------------------------------------------

    def reset(self) -> int:
        """Setzt Umgebung auf start_state zurück und gibt ihn zurück."""
        self._current_state = self.start_state
        return self._current_state

    def step(self, action: int) -> Tuple[int, float, bool]:
        """Führt action im aktuellen Zustand aus.

        Rückgabe: (next_state, reward, done).
        Im Terminalzustand: sofort (s, 0.0, True) ohne Transition.
        """
        s = self._current_state
        if self.is_terminal(s):
            return s, 0.0, True

        if action not in self.allowed_actions[s]:
            raise ValueError(f"Aktion {action} in Zustand {s} nicht erlaubt.")

        probs = self.transition_probabilities[s, action]
        next_state = int(np.random.choice(self.n_states, p=probs))
        reward = self._sample_reward(s, action, next_state)

        self._current_state = next_state
        done = self.is_terminal(next_state)
        return next_state, reward, done

    def _sample_reward(self, state: int, action: int, next_state: int) -> float:
        """Reward für (s, a, s'). Subklassen überschreiben für stochastische Rewards."""
        return float(self.expected_rewards[state, action])

    def is_terminal(self, state: int) -> bool:
        """True wenn state ein Terminalzustand ist."""
        return state in self.terminal_states

    def mc_estimate_reward(self, state: int, action: int, n: int = 1000) -> float:
        """Monte-Carlo-Schätzer für E[R | S=state, A=action].

        Simuliert n Übergänge und mittelt die erhaltenen Rewards.
        Der interne Zustand wird nach dem Aufruf wiederhergestellt.
        """
        saved = self._current_state
        rewards = np.empty(n)
        for i in range(n):
            self._current_state = state
            _, r, _ = self.step(action)
            rewards[i] = r
        self._current_state = saved
        return float(rewards.mean())

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Prüft, dass P[s, a, :] für alle erlaubten (s, a) zu 1 summiert."""
        P = self.transition_probabilities
        for s in self.states:
            for a in self.allowed_actions[s]:
                row_sum = P[s, a].sum()
                assert abs(row_sum - 1.0) < 1e-9, (
                    f"P[{s}, {a}, :] summiert zu {row_sum:.10f}, erwartet 1."
                )

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_actions(self) -> int:
        return len(self.actions)
