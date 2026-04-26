"""Cliff Walking Environment (Sutton & Barto, Beispiel 6.6).

4×12 Grid mit Start unten links, Ziel unten rechts. Die untere Zeile
zwischen Start und Ziel (Spalten 1–10) ist der Cliff: Betreten gibt
Reward cliff_reward und teleportiert sofort zum Start (nicht terminal).

Reward-Konvention (wie S&B):
    - Alle Schritte:      step_reward  (default -1)
    - Cliff-Betreten:     cliff_reward (default -100), Teleport zu Start
    - Ziel-Betreten:      step_reward  (episode endet)

Entscheidung: Das Ziel erhält step_reward=-1 beim Betreten, danach endet
die Episode. Diese Konvention entspricht S&B Kapitel 6.5 ("reward is -1
on all transitions"). Alternativ wäre goal_reward=0 denkbar; es ändert
nur die Skala von V* nicht die Policy.

Verwendung im Submission-Task c): SARSA vs. Q-Learning vergleichen.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .mdp_base import FiniteMDP


class CliffWalk(FiniteMDP):
    """Cliff Walking als FiniteMDP (Sutton & Barto, Beispiel 6.6)."""

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    _DELTA: Dict[int, Tuple[int, int]] = {
        0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1),
    }
    _ARROW: Dict[int, str] = {0: "^", 1: "v", 2: "<", 3: ">"}

    def __init__(
        self,
        rows: int = 4,
        cols: int = 12,
        cliff_reward: float = -100.0,
        step_reward: float = -1.0,
        gamma: float = 1.0,
        wall_behavior: str = "stay",
    ) -> None:
        """Erzeugt eine Cliff-Walking-Umgebung.

        Args:
            rows, cols:    Gitterdimensionen (default 4×12).
            cliff_reward:  Reward beim Betreten einer Cliff-Zelle.
            step_reward:   Standard-Reward für alle anderen Schritte
                           (incl. Betreten des Ziels).
            gamma:         Diskontierungsfaktor (default 1.0 wie in S&B).
            wall_behavior: "stay" (an Wand bleiben) oder "forbidden".
        """
        super().__init__(gamma)

        if wall_behavior not in ("stay", "forbidden"):
            raise ValueError(f"wall_behavior muss 'stay' oder 'forbidden' sein.")

        self.rows = rows
        self.cols = cols
        self.cliff_reward = cliff_reward
        self.step_reward = step_reward
        self.wall_behavior = wall_behavior

        self.states = list(range(rows * cols))
        self.actions = [self.UP, self.DOWN, self.LEFT, self.RIGHT]
        self.start_state = self._rc(rows - 1, 0)
        self.goal_state = self._rc(rows - 1, cols - 1)
        self.terminal_states = {self.goal_state}
        # Cliff: untere Zeile zwischen Start und Ziel
        self.cliff_states = {self._rc(rows - 1, c) for c in range(1, cols - 1)}

        self._build_transition_matrix()
        self._current_state = self.start_state

    # ------------------------------------------------------------------
    # Koordinaten-Hilfsmethoden
    # ------------------------------------------------------------------

    def _rc(self, row: int, col: int) -> int:
        return row * self.cols + col

    def _to_rc(self, state: int) -> Tuple[int, int]:
        return divmod(state, self.cols)

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    # ------------------------------------------------------------------
    # Übergangsmatrix
    # ------------------------------------------------------------------

    def _build_transition_matrix(self) -> None:
        """Befüllt P (S, A, S') und R (S, A).

        Cliff-Zellen:
            Jeder Schritt der in eine Cliff-Zelle führen würde, landet
            stattdessen beim Start mit Reward=cliff_reward.
        Terminalzustand (Ziel):
            Absorbierend; P[goal, *, goal] = 1, R=0.
        Cliff-Zellen selbst:
            Alle Aktionen führen zurück zum Start (der Agent befindet
            sich in der Praxis nie dauerhaft in einer Cliff-Zelle).
        """
        S = len(self.states)
        A = len(self.actions)
        P = np.zeros((S, A, S))
        R = np.zeros((S, A))
        allowed: Dict[int, List[int]] = {}

        for s in self.states:
            row, col = self._to_rc(s)

            # Terminalzustand: absorbierend
            if s in self.terminal_states:
                for a in self.actions:
                    P[s, a, s] = 1.0
                allowed[s] = list(self.actions)
                continue

            # Cliff-Zellen: immer zurück zum Start
            if s in self.cliff_states:
                for a in self.actions:
                    P[s, a, self.start_state] = 1.0
                    R[s, a] = self.cliff_reward
                allowed[s] = list(self.actions)
                continue

            valid: List[int] = []
            for a in self.actions:
                dr, dc = self._DELTA[a]
                nr, nc = row + dr, col + dc
                in_bounds = self._in_bounds(nr, nc)

                if not in_bounds and self.wall_behavior == "forbidden":
                    continue

                valid.append(a)

                if not in_bounds:
                    # Wand-Semantik: bleibe in s
                    ns = s
                    R[s, a] = self.step_reward
                else:
                    ns_cand = self._rc(nr, nc)
                    if ns_cand in self.cliff_states:
                        # Cliff-Betreten: Teleport zu Start
                        ns = self.start_state
                        R[s, a] = self.cliff_reward
                    else:
                        ns = ns_cand
                        R[s, a] = self.step_reward
                P[s, a, ns] = 1.0

            allowed[s] = valid

        self.transition_probabilities = P
        self.expected_rewards = R
        self.allowed_actions = allowed

    # ------------------------------------------------------------------
    # Visualisierung
    # ------------------------------------------------------------------

    def visualize_layout(
        self,
        ax: Optional[plt.Axes] = None,
        title: str = "Cliff Walk",
    ) -> plt.Axes:
        """Zeichnet das Grid-Layout mit Cliff, Start und Ziel."""
        if ax is None:
            _, ax = plt.subplots(figsize=(self.cols * 0.8, self.rows * 1.2))

        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_aspect("equal")
        ax.set_xticks(range(self.cols + 1))
        ax.set_yticks(range(self.rows + 1))
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_title(title, fontsize=11)

        for s in self.states:
            r, c = self._to_rc(s)
            y = self.rows - 1 - r

            if s == self.start_state:
                facecolor, label = "#aed6f1", "S"
            elif s == self.goal_state:
                facecolor, label = "#a9dfbf", "G"
            elif s in self.cliff_states:
                facecolor, label = "#e74c3c", "C"
            else:
                facecolor, label = "#fdfefe", f"({r},{c})"

            rect = mpatches.FancyBboxPatch(
                (c + 0.05, y + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=facecolor, edgecolor="#555555", linewidth=0.8,
            )
            ax.add_patch(rect)
            ax.text(c + 0.5, y + 0.5, label, ha="center", va="center",
                    fontsize=7, color="#2c3e50")

        for x in range(self.cols + 1):
            ax.axvline(x, color="#cccccc", linewidth=0.5)
        for yy in range(self.rows + 1):
            ax.axhline(yy, color="#cccccc", linewidth=0.5)

        legend_patches = [
            mpatches.Patch(facecolor="#aed6f1", edgecolor="#555555", label="Start"),
            mpatches.Patch(facecolor="#a9dfbf", edgecolor="#555555", label="Ziel"),
            mpatches.Patch(facecolor="#e74c3c", edgecolor="#555555", label="Cliff"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.8)
        return ax

    def visualize_policy(
        self,
        Q: np.ndarray,
        ax: Optional[plt.Axes] = None,
        title: str = "Policy",
    ) -> plt.Axes:
        """Zeichnet die Greedy-Policy aus Q als Pfeile."""
        ax = self.visualize_layout(ax=ax, title=title)
        for s in self.states:
            if s in self.terminal_states or s in self.cliff_states:
                continue
            r, c = self._to_rc(s)
            y = self.rows - 1 - r
            acts = self.allowed_actions[s]
            best = acts[int(np.argmax(Q[s, acts]))]
            ax.text(c + 0.5, y + 0.5, self._ARROW[best],
                    ha="center", va="center", fontsize=10, color="#1a5276")
        return ax
