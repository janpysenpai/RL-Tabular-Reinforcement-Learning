"""Demo: Value Iteration auf einem 4x4 GridWorld (Skript Algorithm 6).

Heatmap von V*(s) mit eingezeichneter Greedy-Policy.
Hyperparameter: gamma=0.9, sync=True, tol=1e-9.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ...envs.gridworld import GridWorld
from ...algos.value_iteration import value_iteration_V, greedy_policy_from_v

FIGURES_DIR = Path(__file__).parents[3] / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

GAMMA = 0.9
SYNC = True
TOL = 1e-9


def _plot_v_heatmap_policy(
    env: GridWorld,
    V: np.ndarray,
    policy: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    rows, cols = env.rows, env.cols
    fig, ax = plt.subplots(figsize=(cols * 1.5, rows * 1.5))

    cmap = plt.cm.YlGn
    v_non_term = [V[s] for s in env.states if s not in env.terminal_states]
    vmin = min(v_non_term) if v_non_term else 0.0
    vmax = max(v_non_term) if v_non_term else 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    arrow_delta = {0: (0, 0.28), 1: (0, -0.28), 2: (-0.28, 0), 3: (0.28, 0)}

    for s in env.states:
        r, c = env._to_rc(s)
        y = rows - 1 - r  # Plotkoordinate: Zeile 0 oben

        is_start = s == env.start_state
        is_term = s in env.terminal_states

        if is_term:
            facecolor = "#27ae60"
        elif is_start:
            facecolor = "#2980b9"
        else:
            facecolor = cmap(norm(V[s]))

        rect = mpatches.Rectangle(
            (c, y), 1, 1,
            facecolor=facecolor, edgecolor="#555555", linewidth=0.8,
        )
        ax.add_patch(rect)

        # V*-Wert
        text_color = "white" if (is_start or is_term) else "black"
        ax.text(c + 0.5, y + 0.72, f"{V[s]:.3f}",
                ha="center", va="center", fontsize=8, color=text_color)

        # Zellenbezeichnung
        label = "S" if is_start else ("G" if is_term else f"({r},{c})")
        ax.text(c + 0.5, y + 0.28, label,
                ha="center", va="center", fontsize=7,
                color="white" if (is_start or is_term) else "#444444")

        # Policy-Pfeil (nicht in Terminalzustand)
        if not is_term:
            action = int(np.argmax(policy[s]))
            dx, dy = arrow_delta[action]
            ax.annotate(
                "", xy=(c + 0.5 + dx, y + 0.5 + dy),
                xytext=(c + 0.5, y + 0.5),
                arrowprops=dict(arrowstyle="->", color="#1a5276", lw=1.5),
            )

    # Gitterlinien
    for x in range(cols + 1):
        ax.axvline(x, color="#888888", linewidth=0.5)
    for yy in range(rows + 1):
        ax.axhline(yy, color="#888888", linewidth=0.5)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.set_xticks(range(cols + 1))
    ax.set_yticks(range(rows + 1))
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_xlabel(f"Spalten (0..{cols - 1})", fontsize=8)
    ax.set_ylabel(f"Zeilen (0..{rows - 1})", fontsize=8)
    ax.set_title(title, fontsize=9, pad=8)

    legend_patches = [
        mpatches.Patch(facecolor="#2980b9", edgecolor="#555555", label="Start S"),
        mpatches.Patch(facecolor="#27ae60", edgecolor="#555555", label="Ziel G"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7)

    # Farbskala
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="V*(s)", fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    env = GridWorld(
        rows=4, cols=4,
        start=(0, 0),
        terminal_states=[(3, 3)],
        cell_rewards={(3, 3): 1.0},
        default_reward=0.0,
        gamma=GAMMA,
        wall_behavior="stay",
    )

    V, info = value_iteration_V(env, gamma=GAMMA, sync=SYNC, tol=TOL)
    policy = greedy_policy_from_v(env, V, gamma=GAMMA)

    print("=== Value Iteration — 4x4 GridWorld ===")
    print(f"Hyperparameter: gamma={GAMMA}, sync={SYNC}, tol={TOL}")
    print(f"Konvergenz nach {info['iterations']} Iterationen")
    print(f"V*(Start=(0,0)) = {V[env.start_state]:.6f}")
    print(f"V*(Goal=(3,3))  = {V[env.terminal_states.copy().pop()]:.6f}")
    print()
    print("V* (Zeilen 0..3, Spalten 0..3):")
    print(V.reshape(env.rows, env.cols).round(4))

    title = (
        f"Value Iteration (Alg. 6) — 4x4 GridWorld\n"
        f"gamma={GAMMA}, sync={SYNC}, tol={TOL}, "
        f"Iterationen={info['iterations']}"
    )
    save_path = FIGURES_DIR / "demo_value_iteration.png"
    _plot_v_heatmap_policy(env, V, policy, title, save_path)
    print(f"\nPlot gespeichert: {save_path}")


if __name__ == "__main__":
    main()
