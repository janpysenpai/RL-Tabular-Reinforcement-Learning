"""4×4 GridWorld für Abgabe (Übungsblatt 8, Aufgabe 4f).

Layout (row 0 oben, col 0 links):

    (0,0)F  (0,1).  (0,2).  (0,3)S
    (1,0).  (1,1).  (1,2).  (1,3).
    (2,0).  (2,1).  (2,2)SR (2,3)SR
    (3,0).  (3,1)G  (3,2)SR (3,3)SR

    F  = Fake Goal  (0,0): terminal,   R = 0.65 (deterministisch)
    S  = Start      (0,3): Startzustand
    G  = Goal       (3,1): terminal,   R = 1.0  (deterministisch)
    SR = Stochastic Region (2,2),(2,3),(3,2),(3,3):
              ohne Noise: E[R] = -0.05
              mit Noise:  R ~ {-2.1, +2.0} mit p=0.5 je
    .  = Default:
              ohne Noise: E[R] = 0.0
              mit Noise:  R ~ {-0.05, +0.05} mit p=0.5 je

gamma = 0.9, wall_behavior = "stay".

Zwei Varianten wählbar über noise=False/True:
    noise=False: alle Rewards deterministisch (expected_rewards).
    noise=True:  SR-Zellen und Defaults stochastisch; Terminals deterministisch.

Modellbasierte Algorithmen nutzen expected_rewards (korrekt für beide
Varianten da E[R] identisch). Sample-basierte Algorithmen erhalten den
tatsächlichen Reward via _sample_reward (variiert bei noise=True).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .mdp_base import FiniteMDP


# ------------------------------------------------------------------
# Layout-Konstanten
# ------------------------------------------------------------------

ROWS, COLS = 4, 4

FAKE_GOAL_RC = (0, 0)   # F — Fake Goal
START_RC     = (0, 3)   # S — Start
GOAL_RC      = (3, 1)   # G — Goal (wahres Ziel)

SR_RCS = {(2, 2), (2, 3), (3, 2), (3, 3)}  # Stochastic Region

FAKE_GOAL_REWARD = 0.65
GOAL_REWARD      = 1.0

# Stochastic Region: Reward ∈ {-2.1, +2.0}, p=0.5
SR_OUTCOMES   = np.array([-2.1, 2.0])
SR_PROBS      = np.array([0.5, 0.5])
SR_EXPECTED   = float(SR_OUTCOMES @ SR_PROBS)         # -0.05

# Default-Reward: {-0.05, +0.05}, p=0.5
DEF_OUTCOMES  = np.array([-0.05, 0.05])
DEF_PROBS     = np.array([0.5, 0.5])
DEF_EXPECTED  = float(DEF_OUTCOMES @ DEF_PROBS)       # 0.0

_DELTA: Dict[int, Tuple[int, int]] = {
    0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1),   # UP,DOWN,LEFT,RIGHT
}


def _rc_to_s(r: int, c: int) -> int:
    return r * COLS + c


def _s_to_rc(s: int) -> Tuple[int, int]:
    return divmod(s, COLS)


class GridWorld4x4Submission(FiniteMDP):
    """Spezifisches 4×4 GridWorld-Layout für Abgabe (Blatt 8, Aufg. 4f)."""

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    _ARROW: Dict[int, str] = {0: "^", 1: "v", 2: "<", 3: ">"}

    def __init__(
        self,
        noise: bool = False,
        gamma: float = 0.9,
        seed: Optional[int] = None,
    ) -> None:
        """Erstellt das Submission-GridWorld.

        Args:
            noise:  False → deterministische Rewards (Expected Values).
                    True  → stochastische Rewards für SR- und Default-Zellen.
            gamma:  Diskontierungsfaktor (default 0.9).
            seed:   Seed für den Reward-RNG (nur relevant bei noise=True).
        """
        super().__init__(gamma)
        self.noise = noise
        self._rng = np.random.default_rng(seed)

        self.rows = ROWS
        self.cols = COLS
        self.states = list(range(ROWS * COLS))
        self.actions = [self.UP, self.DOWN, self.LEFT, self.RIGHT]
        self.start_state = _rc_to_s(*START_RC)
        self.terminal_states = {_rc_to_s(*FAKE_GOAL_RC), _rc_to_s(*GOAL_RC)}
        self.sr_states = {_rc_to_s(r, c) for r, c in SR_RCS}

        self._build_transition_matrix()
        self._current_state = self.start_state

    # ------------------------------------------------------------------
    # Übergangsmatrix
    # ------------------------------------------------------------------

    def _build_transition_matrix(self) -> None:
        S, A = ROWS * COLS, 4
        P = np.zeros((S, A, S))
        R = np.zeros((S, A))
        allowed: Dict[int, List[int]] = {}

        for s in self.states:
            r, c = _s_to_rc(s)

            if s in self.terminal_states:
                for a in range(A):
                    P[s, a, s] = 1.0
                allowed[s] = list(range(A))
                continue

            valid: List[int] = []
            for a in range(A):
                dr, dc = _DELTA[a]
                nr, nc = r + dr, c + dc
                in_bounds = (0 <= nr < ROWS) and (0 <= nc < COLS)

                valid.append(a)
                ns = _rc_to_s(nr, nc) if in_bounds else s
                P[s, a, ns] = 1.0
                R[s, a] = self._expected_reward_for(ns)

            allowed[s] = valid

        self.transition_probabilities = P
        self.expected_rewards = R
        self.allowed_actions = allowed

    @staticmethod
    def _expected_reward_for(ns: int) -> float:
        """Erwarteter Reward beim Betreten von Zustand ns."""
        if ns == _rc_to_s(*FAKE_GOAL_RC):
            return FAKE_GOAL_REWARD
        if ns == _rc_to_s(*GOAL_RC):
            return GOAL_REWARD
        if ns in {_rc_to_s(r, c) for r, c in SR_RCS}:
            return SR_EXPECTED
        return DEF_EXPECTED

    # ------------------------------------------------------------------
    # Stochastisches Sampling
    # ------------------------------------------------------------------

    def _sample_reward(self, state: int, action: int, next_state: int) -> float:
        """Gibt den tatsächlichen Reward zurück.

        noise=False: deterministisch (expected_rewards).
        noise=True:  Terminal-Rewards deterministisch; SR und Default stochastisch.
        """
        if not self.noise:
            return float(self.expected_rewards[state, action])

        # Terminale Zustände: deterministisch
        if next_state in self.terminal_states:
            return self._expected_reward_for(next_state)

        # Stochastic Region
        if next_state in self.sr_states:
            return float(self._rng.choice(SR_OUTCOMES, p=SR_PROBS))

        # Default
        return float(self._rng.choice(DEF_OUTCOMES, p=DEF_PROBS))

    # ------------------------------------------------------------------
    # Visualisierung
    # ------------------------------------------------------------------

    def visualize_layout(
        self,
        ax: Optional[plt.Axes] = None,
        title: str = "4×4 Submission GridWorld",
    ) -> plt.Axes:
        """Zeichnet das Layout mit Farbkodierung aller Zell-Typen."""
        if ax is None:
            _, ax = plt.subplots(figsize=(COLS * 1.5, ROWS * 1.5))

        ax.set_xlim(0, COLS)
        ax.set_ylim(0, ROWS)
        ax.set_aspect("equal")
        ax.set_xticks(range(COLS + 1))
        ax.set_yticks(range(ROWS + 1))
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_title(title, fontsize=11)

        _fake_goal_s = _rc_to_s(*FAKE_GOAL_RC)
        _goal_s      = _rc_to_s(*GOAL_RC)
        _start_s     = self.start_state

        COLORS = {
            "fake_goal": "#f0b27a",   # orange: Fake Goal
            "goal":      "#a9dfbf",   # grün:   Ziel
            "start":     "#aed6f1",   # blau:   Start
            "sr":        "#d7bde2",   # lila:   SR
            "default":   "#fdfefe",   # weiß:   Default
        }

        for s in self.states:
            r, c = _s_to_rc(s)
            y = ROWS - 1 - r

            if s == _fake_goal_s:
                fc = COLORS["fake_goal"]
                label = f"F\nR={FAKE_GOAL_REWARD}"
            elif s == _goal_s:
                fc = COLORS["goal"]
                label = f"G\nR={GOAL_REWARD}"
            elif s == _start_s:
                fc = COLORS["start"]
                label = "S"
            elif s in self.sr_states:
                fc = COLORS["sr"]
                label = "SR\nR∈{-2.1,2}"
            else:
                fc = COLORS["default"]
                label = f"({r},{c})"

            rect = mpatches.FancyBboxPatch(
                (c + 0.05, y + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=fc, edgecolor="#555555", linewidth=0.8,
            )
            ax.add_patch(rect)
            ax.text(c + 0.5, y + 0.5, label,
                    ha="center", va="center", fontsize=8, color="#2c3e50")

        for x in range(COLS + 1):
            ax.axvline(x, color="#cccccc", linewidth=0.5)
        for yy in range(ROWS + 1):
            ax.axhline(yy, color="#cccccc", linewidth=0.5)

        patches = [
            mpatches.Patch(facecolor=COLORS["start"],     label="Start (S)"),
            mpatches.Patch(facecolor=COLORS["fake_goal"], label=f"Fake Goal (R={FAKE_GOAL_REWARD})"),
            mpatches.Patch(facecolor=COLORS["goal"],      label=f"Goal (R={GOAL_REWARD})"),
            mpatches.Patch(facecolor=COLORS["sr"],        label="SR: R∈{-2.1,+2}, p=0.5"),
            mpatches.Patch(facecolor=COLORS["default"],   label="Default: E[R]=0"),
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.9)
        return ax

    def visualize_policy(
        self,
        Q: np.ndarray,
        ax: Optional[plt.Axes] = None,
        title: str = "Greedy Policy",
    ) -> plt.Axes:
        """Zeichnet die Greedy-Policy aus Q als Pfeile."""
        ax = self.visualize_layout(ax=ax, title=title)
        for s in self.states:
            if s in self.terminal_states:
                continue
            r, c = _s_to_rc(s)
            y = ROWS - 1 - r
            acts = self.allowed_actions[s]
            best = acts[int(np.argmax(Q[s, acts]))]
            ax.text(c + 0.5, y + 0.5, self._ARROW[best],
                    ha="center", va="center", fontsize=14, color="#1a5276")
        return ax
