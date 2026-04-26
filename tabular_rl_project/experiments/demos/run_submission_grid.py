"""Demo: Value Iteration auf dem 4×4 Submission-GridWorld (Blatt 8, Aufg. 4f).

Vergleicht zwei Varianten:
    noise=False  deterministischer Reward (E[R])
    noise=True   stochastischer Reward (SR- und Default-Zellen)

Zeigt für beide Varianten:
    - V*-Heatmap mit Zahlenwerten
    - Greedy-Policy als Pfeile

Hyperparameter:
    gamma = 0.9
    tol   = 1e-9
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ...envs.gridworld_4x4_submission import GridWorld4x4Submission, _rc_to_s, GOAL_RC
from ...algos.value_iteration import value_iteration_V, greedy_policy_from_v

FIGURES_DIR = Path(__file__).parent / ".." / ".." / ".." / "figures"

ROWS, COLS = 4, 4


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        ("ohne Noise (deterministisch)", False),
        ("mit Noise (stochastisch)", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for col_idx, (label, noise) in enumerate(configs):
        env = GridWorld4x4Submission(noise=noise, gamma=0.9)
        V, info = value_iteration_V(env, tol=1e-9)
        pi = greedy_policy_from_v(env, V)

        print(f"\n[{label}]  {info['iterations']} Iterationen")
        print(f"  V*(Start (0,3)) = {V[env.start_state]:.4f}")
        print(f"  V*(Goal  (3,1)) = {V[_rc_to_s(*GOAL_RC)]:.4f}")

        # -- Heatmap --
        ax_heat = axes[0, col_idx]
        grid_V = V.reshape(ROWS, COLS)
        im = ax_heat.imshow(grid_V, cmap="viridis", origin="upper", vmin=grid_V.min(), vmax=grid_V.max())
        ax_heat.set_title(f"V* — {label}", fontsize=10)
        ax_heat.set_xticks(range(COLS))
        ax_heat.set_yticks(range(ROWS))
        ax_heat.set_xlabel("Spalte")
        ax_heat.set_ylabel("Zeile")
        for r in range(ROWS):
            for c in range(COLS):
                ax_heat.text(c, r, f"{grid_V[r,c]:.3f}",
                             ha="center", va="center", fontsize=8,
                             color="white" if grid_V[r,c] < grid_V.max() / 2 else "black")
        fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

        # -- Policy --
        ax_pol = axes[1, col_idx]
        env.visualize_policy(pi, ax=ax_pol, title=f"Greedy Policy — {label}")

    fig.suptitle(
        "4×4 Submission GridWorld — Value Iteration (γ=0.9)",
        fontsize=12,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "submission_grid_vi.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGespeichert: {out}")
