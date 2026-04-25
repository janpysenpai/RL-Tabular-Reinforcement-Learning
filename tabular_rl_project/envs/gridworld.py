"""Konfigurierbare Grid World (Übungsblatt 4, Aufgabe 5a).

Konfigurierbar:
    - Gittergröße (rows x cols)
    - Belohnungsstruktur: Reward beim Betreten jeder Zelle (deterministisch);
      stochastische Rewards (Normal, Binomial) kommen in einem späteren Schritt.
    - Wandverhalten: "stay" (Agent bleibt stehen) oder "forbidden" (Aktion verboten)
    - Wind, Slippery, Random Noise: noch NICHT implementiert (geplant für spätere Schritte)

Zustände werden als Integerpfade row * cols + col kodiert (row=0 oben).
Aktionen: 0=oben, 1=unten, 2=links, 3=rechts.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .mdp_base import FiniteMDP


class GridWorld(FiniteMDP):
    """Deterministische m×n Grid World als Subklasse von FiniteMDP."""

    # Aktionskonstanten
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    ACTION_NAMES = {0: "oben", 1: "unten", 2: "links", 3: "rechts"}
    _DELTA: Dict[int, Tuple[int, int]] = {
        0: (-1, 0),  # oben:  Zeile nimmt ab
        1: (1, 0),   # unten: Zeile nimmt zu
        2: (0, -1),  # links
        3: (0, 1),   # rechts
    }
    _ARROW: Dict[int, str] = {0: "^", 1: "v", 2: "<", 3: ">"}

    def __init__(
        self,
        rows: int,
        cols: int,
        start: Tuple[int, int],
        terminal_states: List[Tuple[int, int]],
        cell_rewards: Optional[Dict[Tuple[int, int], float]] = None,
        default_reward: float = 0.0,
        gamma: float = 1.0,
        wall_behavior: str = "stay",
    ) -> None:
        """Erstellt eine deterministsche Grid World.

        Args:
            rows, cols:       Gitterdimensionen.
            start:            (row, col) des Startzustands.
            terminal_states:  Liste von (row, col)-Paaren, die Terminal sind.
            cell_rewards:     Dict {(row, col): reward} — Reward beim Betreten.
                              Nicht genannte Zellen erhalten default_reward.
            default_reward:   Standard-Reward für nicht explizit gelistete Zellen.
            gamma:            Diskontierungsfaktor.
            wall_behavior:    "stay" (an Wand bleiben) oder "forbidden" (Aktion verboten).
        """
        super().__init__(gamma)

        if wall_behavior not in ("stay", "forbidden"):
            raise ValueError(f"wall_behavior muss 'stay' oder 'forbidden' sein, nicht '{wall_behavior}'.")

        self.rows = rows
        self.cols = cols
        self.wall_behavior = wall_behavior
        self.default_reward = default_reward

        self.states = list(range(rows * cols))
        self.actions = [self.UP, self.DOWN, self.LEFT, self.RIGHT]
        self.start_state = self._rc(start[0], start[1])
        self.terminal_states = {self._rc(r, c) for r, c in terminal_states}

        # Reward beim Betreten jedes Zustands (state-Index -> float)
        self._entry_reward: Dict[int, float] = {}
        if cell_rewards:
            for (r, c), rw in cell_rewards.items():
                self._entry_reward[self._rc(r, c)] = rw

        self._build_transition_matrix()
        self._current_state = self.start_state

    # ------------------------------------------------------------------
    # Koordinaten-Hilfsmethoden
    # ------------------------------------------------------------------

    def _rc(self, row: int, col: int) -> int:
        """(row, col) -> Zustandsindex."""
        return row * self.cols + col

    def _to_rc(self, state: int) -> Tuple[int, int]:
        """Zustandsindex -> (row, col)."""
        return divmod(state, self.cols)

    def _next_rc(self, row: int, col: int, action: int) -> Tuple[int, int]:
        """Gibt die Zielzelle nach action zurück (noch vor Wandbehandlung)."""
        dr, dc = self._DELTA[action]
        return row + dr, col + dc

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    # ------------------------------------------------------------------
    # Übergangsmatrix aufbauen
    # ------------------------------------------------------------------

    def _build_transition_matrix(self) -> None:
        """Befüllt transition_probabilities (S, A, S') und expected_rewards (S, A)."""
        S = len(self.states)
        A = len(self.actions)

        P = np.zeros((S, A, S))
        R = np.zeros((S, A))
        allowed: Dict[int, List[int]] = {}

        for s in self.states:
            row, col = self._to_rc(s)

            if s in self.terminal_states:
                # Terminalzustand absorbiert: Transition bleibt in s, Reward = 0
                for a in self.actions:
                    P[s, a, s] = 1.0
                allowed[s] = list(self.actions)
                continue

            valid: List[int] = []
            for a in self.actions:
                nr, nc = self._next_rc(row, col, a)
                in_bounds = self._in_bounds(nr, nc)

                if not in_bounds and self.wall_behavior == "forbidden":
                    continue  # Aktion an dieser Zelle verboten

                valid.append(a)
                ns = self._rc(nr, nc) if in_bounds else s  # stay-Semantik
                P[s, a, ns] = 1.0
                R[s, a] = self._entry_reward.get(ns, self.default_reward)

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
        title: str = "Grid World",
        special_labels: Optional[Dict[Tuple[int, int], str]] = None,
    ) -> plt.Axes:
        """Zeichnet das Grid-Layout (Zellen, Start, Terminals).

        Args:
            ax:              Matplotlib-Achse; wird neu erstellt wenn None.
            title:           Achsentitel.
            special_labels:  Zusätzliche Beschriftungen {(row, col): "Label"}.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(self.cols * 1.2, self.rows * 1.2))

        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_aspect("equal")
        ax.set_xticks(range(self.cols + 1))
        ax.set_yticks(range(self.rows + 1))
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_title(title, fontsize=11)

        start_rc = self._to_rc(self.start_state)
        terminal_rcs = {self._to_rc(t) for t in self.terminal_states}

        for s in self.states:
            r, c = self._to_rc(s)
            # y-Koordinate: row 0 oben → in Plotkoordinaten rows-1-r unten
            y = self.rows - 1 - r

            if (r, c) == start_rc:
                facecolor = "#aed6f1"  # blau: Start
                label = "S"
            elif (r, c) in terminal_rcs:
                rw = self._entry_reward.get(s, self.default_reward)
                facecolor = "#a9dfbf"  # grün: Ziel
                label = f"G\n(R={rw:.2f})"
            else:
                facecolor = "#fdfefe"  # weiß: normal
                rw = self._entry_reward.get(s, self.default_reward)
                label = f"({r},{c})\nr={rw:.2f}" if rw != self.default_reward else f"({r},{c})"

            if special_labels and (r, c) in special_labels:
                label = special_labels[(r, c)]

            rect = mpatches.FancyBboxPatch(
                (c + 0.05, y + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=facecolor, edgecolor="#555555", linewidth=0.8,
            )
            ax.add_patch(rect)
            ax.text(
                c + 0.5, y + 0.5, label,
                ha="center", va="center", fontsize=8, color="#2c3e50",
            )

        # Gitterlinien
        for x in range(self.cols + 1):
            ax.axvline(x, color="#cccccc", linewidth=0.5)
        for y in range(self.rows + 1):
            ax.axhline(y, color="#cccccc", linewidth=0.5)

        # Legende
        legend_patches = [
            mpatches.Patch(facecolor="#aed6f1", edgecolor="#555555", label="Start"),
            mpatches.Patch(facecolor="#a9dfbf", edgecolor="#555555", label="Terminal / Ziel"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.8)
        ax.set_xlabel(f"Spalten (0..{self.cols - 1})", fontsize=8)
        ax.set_ylabel(f"Zeilen (0..{self.rows - 1})", fontsize=8)

        return ax

    def visualize_policy(
        self,
        policy: Dict[int, int],
        ax: Optional[plt.Axes] = None,
        title: str = "Policy",
        save_path: Optional[str] = None,
    ) -> plt.Axes:
        """Zeichnet eine deterministische Policy als Pfeile ins Grid.

        Args:
            policy:    Dict {state: action} für alle Nicht-Terminalzustände.
            ax:        Matplotlib-Achse; wird neu erstellt wenn None.
            title:     Achsentitel.
            save_path: Pfad zum Speichern (optional).
        """
        ax = self.visualize_layout(ax=ax, title=title)

        for s, a in policy.items():
            r, c = self._to_rc(s)
            if s in self.terminal_states:
                continue
            y = self.rows - 1 - r
            ax.text(
                c + 0.5, y + 0.5, self._ARROW[a],
                ha="center", va="center", fontsize=14, color="#1a5276",
            )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        return ax
