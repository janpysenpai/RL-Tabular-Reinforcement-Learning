"""Multi-Step Bandit (Übungsblatt 4, Aufgabe 5b).

Struktur:
    - Startzustand s0 mit k Branches (Aktionen).
    - Branch b_i besteht aus m_i deterministischen Schritten.
    - An jedem Knoten s_{b_i, j} stehen k_{b_i,j} Aktionen zur Wahl.
    - Alle Aktionen an einem Knoten führen zum gleichen Folge-Zustand
      (deterministische Transition); Reward hängt von der Aktion ab.
    - Rewards: deterministisch (float), Normal- oder Binomialverteilt.

Konfiguration über branch_specs:
    branch_specs[i][j][a]  = Reward-Spec für Branch i, Schritt j, Aktion a.
    Reward-Spec: float  |  ("normal", mu, sigma)  |  ("binomial", n, p)

Zustandsnummerierung:
    0               = Wurzel s0
    offsets[i]      = erster aktiver Knoten von Branch i  (falls m_i > 0)
    offsets[i]+m_i  = Terminalzustand von Branch i

Für einen Branch mit m_i=0 Schritten ist offsets[i] direkt der Terminal.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .mdp_base import FiniteMDP

# Reward-Spec-Typ: float oder ("normal", mu, sigma) oder ("binomial", n, p)
RewardSpec = Union[float, Tuple[Any, ...]]


def _expected(spec: RewardSpec) -> float:
    """Erwartungswert eines Reward-Specs."""
    if isinstance(spec, (int, float)):
        return float(spec)
    kind = spec[0]
    if kind == "normal":
        return float(spec[1])
    if kind == "binomial":
        return float(spec[1] * spec[2])
    raise ValueError(f"Unbekannter Reward-Typ: '{kind}'")


def _sample(spec: RewardSpec) -> float:
    """Zieht einen Reward gemäß Spec."""
    if isinstance(spec, (int, float)):
        return float(spec)
    kind = spec[0]
    if kind == "normal":
        return float(np.random.normal(spec[1], spec[2]))
    if kind == "binomial":
        return float(np.random.binomial(int(spec[1]), spec[2]))
    raise ValueError(f"Unbekannter Reward-Typ: '{kind}'")


class MultiStepBandit(FiniteMDP):
    """Multi-Step Bandit als Subklasse von FiniteMDP.

    Args:
        branch_specs:    branch_specs[i][j] = Liste von Reward-Specs für alle
                         Aktionen an Knoten j von Branch i.
                         Leere Branch-Liste (m_i=0) → sofortiger Terminal.
        root_rewards:    Reward-Spec pro Branch-Aktion am Wurzelknoten.
                         Default: default_reward für alle Branches.
        default_reward:  Fallback-Reward für nicht explizit gelistete Aktionen.
        gamma:           Diskontierungsfaktor.
    """

    def __init__(
        self,
        branch_specs: List[List[List[RewardSpec]]],
        root_rewards: Optional[List[RewardSpec]] = None,
        default_reward: float = 0.0,
        gamma: float = 1.0,
    ) -> None:
        super().__init__(gamma)

        k = len(branch_specs)
        if root_rewards is None:
            root_rewards = [default_reward] * k

        # Zustandsoffsets berechnen
        offsets: List[int] = []
        offset = 1  # Zustand 0 = Wurzel
        for i in range(k):
            offsets.append(offset)
            offset += len(branch_specs[i]) + 1  # m_i aktive + 1 Terminal

        total_states = offset

        # Terminalzustände: offsets[i] + m_i für jeden Branch
        terminal_states: set = set()
        for i in range(k):
            terminal_states.add(offsets[i] + len(branch_specs[i]))

        self.states = list(range(total_states))
        self.terminal_states = terminal_states
        self.start_state = 0

        # Maximale Aktionsanzahl über alle Knoten
        max_actions = k
        for steps in branch_specs:
            for node_spec in steps:
                max_actions = max(max_actions, len(node_spec))
        self.actions = list(range(max_actions))

        # Reward-Specs speichern: _reward_dists[state][action] = spec
        self._reward_dists: Dict[int, Dict[int, RewardSpec]] = {}
        for i, spec in enumerate(root_rewards):
            self._reward_dists.setdefault(0, {})[i] = spec
        for i, steps in enumerate(branch_specs):
            for j, node_spec in enumerate(steps):
                s = offsets[i] + j
                for a, spec in enumerate(node_spec):
                    self._reward_dists.setdefault(s, {})[a] = spec

        self._branch_specs = branch_specs
        self._offsets = offsets
        self._k = k
        self._default_reward = default_reward

        self._build_transition_matrix()
        self._current_state = self.start_state

    # ------------------------------------------------------------------
    # Übergangsmatrix
    # ------------------------------------------------------------------

    def _build_transition_matrix(self) -> None:
        S = len(self.states)
        A = len(self.actions)

        P = np.zeros((S, A, S))
        R = np.zeros((S, A))
        allowed: Dict[int, List[int]] = {}

        # Terminalzustände: absorbierend
        for t in self.terminal_states:
            for a in self.actions:
                P[t, a, t] = 1.0
            allowed[t] = list(self.actions)

        # Wurzel: Aktion i → ersten aktiven Knoten von Branch i (oder Terminal)
        for i in range(self._k):
            ns = self._offsets[i]  # = Terminal falls m_i=0, sonst erster aktiver Knoten
            P[0, i, ns] = 1.0
            spec = self._reward_dists.get(0, {}).get(i, self._default_reward)
            R[0, i] = _expected(spec)
        allowed[0] = list(range(self._k))

        # Branch-Knoten
        for i, steps in enumerate(self._branch_specs):
            m_i = len(steps)
            for j, node_spec in enumerate(steps):
                s = self._offsets[i] + j
                ns = self._offsets[i] + j + 1  # nächster Knoten oder Terminal
                n_act = len(node_spec)
                for a in range(n_act):
                    P[s, a, ns] = 1.0
                    spec = self._reward_dists.get(s, {}).get(a, self._default_reward)
                    R[s, a] = _expected(spec)
                allowed[s] = list(range(n_act))

        self.transition_probabilities = P
        self.expected_rewards = R
        self.allowed_actions = allowed

    # ------------------------------------------------------------------
    # Stochastischer Reward-Sampler
    # ------------------------------------------------------------------

    def _sample_reward(self, state: int, action: int, next_state: int) -> float:
        spec = self._reward_dists.get(state, {}).get(action, self._default_reward)
        return _sample(spec)

    # ------------------------------------------------------------------
    # Visualisierung
    # ------------------------------------------------------------------

    def visualize(
        self,
        ax: Optional[plt.Axes] = None,
        title: str = "Multi-Step Bandit",
        save_path: Optional[str] = None,
    ) -> plt.Axes:
        """Zeichnet den Bandit als Baum (Wurzel links, Terminals rechts)."""
        if ax is None:
            fig_h = max(3, self._k * 1.4)
            _, ax = plt.subplots(figsize=(8, fig_h))

        ax.set_title(title, fontsize=11)
        ax.axis("off")

        # y-Koordinaten: Terminals gleichmäßig verteilt
        terminal_list = sorted(self.terminal_states)
        n_terminals = len(terminal_list)
        term_y: Dict[int, float] = {
            t: (n_terminals - 1 - idx) / max(n_terminals - 1, 1)
            for idx, t in enumerate(terminal_list)
        }

        # Knoten-Positionen berechnen
        pos: Dict[int, Tuple[float, float]] = {}
        pos[0] = (0.0, 0.5)  # Wurzel

        for i in range(self._k):
            m_i = len(self._branch_specs[i])
            terminal = self._offsets[i] + m_i
            ty = term_y[terminal]

            # Aktive Knoten gleichmäßig zwischen x=0.3 und x=0.7
            for j in range(m_i):
                s = self._offsets[i] + j
                x = 0.3 + j * (0.4 / max(m_i, 1))
                pos[s] = (x, ty)

            pos[terminal] = (1.0, ty)

        # Kanten zeichnen
        def draw_edge(s: int, ns: int, label: str) -> None:
            x0, y0 = pos[s]
            x1, y1 = pos[ns]
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
            )
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my + 0.02, label, ha="center", va="bottom", fontsize=7, color="#1a5276")

        # Wurzel-Kanten
        for i in range(self._k):
            m_i = len(self._branch_specs[i])
            ns = self._offsets[i]  # erster aktiver oder Terminal
            spec = self._reward_dists.get(0, {}).get(i, self._default_reward)
            draw_edge(0, ns, f"a{i}: r={_spec_label(spec)}")

        # Branch-Kanten
        for i, steps in enumerate(self._branch_specs):
            m_i = len(steps)
            for j, node_spec in enumerate(steps):
                s = self._offsets[i] + j
                ns = self._offsets[i] + j + 1
                labels = [f"a{a}:{_spec_label(sp)}" for a, sp in enumerate(node_spec)]
                draw_edge(s, ns, "  ".join(labels))

        # Knoten zeichnen
        for s, (x, y) in pos.items():
            is_root = s == 0
            is_term = s in self.terminal_states
            color = "#aed6f1" if is_root else ("#a9dfbf" if is_term else "#fdfefe")
            circle = plt.Circle((x, y), 0.04, color=color, ec="#555555", lw=0.8, zorder=3)
            ax.add_patch(circle)
            label = "s0" if is_root else (f"T{sorted(self.terminal_states).index(s)}" if is_term else f"s{s}")
            ax.text(x, y, label, ha="center", va="center", fontsize=7, zorder=4)

        ax.set_xlim(-0.1, 1.15)
        ax.set_ylim(-0.1, 1.1)

        legend_patches = [
            mpatches.Patch(facecolor="#aed6f1", edgecolor="#555555", label="Wurzel"),
            mpatches.Patch(facecolor="#a9dfbf", edgecolor="#555555", label="Terminal"),
            mpatches.Patch(facecolor="#fdfefe", edgecolor="#555555", label="Aktiver Knoten"),
        ]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=7)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        return ax


def _spec_label(spec: RewardSpec) -> str:
    """Kurze Textdarstellung eines Reward-Specs für Plots."""
    if isinstance(spec, (int, float)):
        return f"{float(spec):.2f}"
    if spec[0] == "normal":
        return f"N({spec[1]},{spec[2]})"
    if spec[0] == "binomial":
        return f"Bin({spec[1]},{spec[2]})"
    return str(spec)
