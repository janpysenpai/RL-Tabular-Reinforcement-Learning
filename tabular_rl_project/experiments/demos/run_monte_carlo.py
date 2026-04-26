"""Demo: First-Visit vs. Every-Visit MC Policy Evaluation (Skript Alg. 14 & 15).

4x4 GridWorld (gamma=0.9, Start (0,0), Ziel (3,3), R=1).
Uniforme Policy als Auswertungspolicy.

Erzeugte Plots (figures/):
    monte_carlo_convergence.png   — ||V_mc - V_ipe||_inf vs. Episodenzahl
    monte_carlo_heatmaps.png      — V-Heatmaps beider Varianten nach 5000 Ep.

Hyperparameter:
    n_episodes  = 5000
    max_steps   = 500
    gamma       = 0.9
    seed        = 42
    snapshot_every = 100
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ...envs.gridworld import GridWorld
from ...algos.iterative_policy_evaluation import ipe_V
from ...algos.monte_carlo import mc_policy_evaluation_V
from ...algos.exploration import uniform_random_policy

FIGURES_DIR = Path(__file__).parent / ".." / ".." / ".." / "figures"

# ------------------------------------------------------------------
# Hyperparameter
# ------------------------------------------------------------------
N_EPISODES = 5000
MAX_STEPS = 500
GAMMA = 0.9
SEED = 42
SNAPSHOT_EVERY = 100


def build_env() -> GridWorld:
    return GridWorld(
        rows=4, cols=4,
        start=(0, 0),
        terminal_states=[(3, 3)],
        cell_rewards={(3, 3): 1.0},
        default_reward=0.0,
        gamma=GAMMA,
        wall_behavior="stay",
    )


def _v_to_grid(V: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Formt V (S,) in eine (rows, cols)-Matrix um, Zeile 0 oben."""
    return V.reshape(rows, cols)


def plot_convergence(
    snapshots_first: list,
    snapshots_every: list,
    V_ref: np.ndarray,
    save_path: Path,
) -> None:
    """Konvergenzkurve ||V_mc(k) - V_ref||_inf vs. Episodenzahl."""
    eps_first = [ep for ep, _ in snapshots_first]
    err_first = [float(np.abs(V - V_ref).max()) for _, V in snapshots_first]

    eps_every = [ep for ep, _ in snapshots_every]
    err_every = [float(np.abs(V - V_ref).max()) for _, V in snapshots_every]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(eps_first, err_first, label="First-Visit MC (Alg. 15)", linewidth=1.8)
    ax.semilogy(eps_every, err_every, label="Every-Visit MC (Alg. 14)",
                linestyle="--", linewidth=1.8)
    ax.set_xlabel("Episoden")
    ax.set_ylabel(r"$\|V_{\mathrm{MC}}^{(k)} - V_{\mathrm{IPE}}\|_\infty$")
    ax.set_title(
        f"MC Konvergenz — 4x4 GridWorld "
        f"(γ={GAMMA}, max_steps={MAX_STEPS}, seed={SEED})"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gespeichert: {save_path}")


def plot_heatmaps(
    V_first: np.ndarray,
    V_every: np.ndarray,
    V_ref: np.ndarray,
    rows: int,
    cols: int,
    save_path: Path,
) -> None:
    """Drei Heatmaps: First-Visit, Every-Visit, IPE-Referenz."""
    vmin = min(V_first.min(), V_every.min(), V_ref.min())
    vmax = max(V_first.max(), V_every.max(), V_ref.max())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    titles = [
        f"First-Visit MC\n(n={N_EPISODES} Ep., seed={SEED})",
        f"Every-Visit MC\n(n={N_EPISODES} Ep., seed={SEED})",
        "IPE (Referenz)",
    ]
    arrays = [V_first, V_every, V_ref]

    for ax, title, V in zip(axes, titles, arrays):
        grid = _v_to_grid(V, rows, cols)
        im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="viridis", origin="upper")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Spalte")
        ax.set_ylabel("Zeile")
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))

        for r in range(rows):
            for c in range(cols):
                ax.text(c, r, f"{grid[r, c]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if grid[r, c] < (vmin + vmax) / 2 else "black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"V^π — Uniforme Policy, 4x4 GridWorld (γ={GAMMA})", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gespeichert: {save_path}")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    env = build_env()
    policy = uniform_random_policy(env)

    # IPE-Referenz
    V_ipe, _ = ipe_V(env, policy, tol=1e-9)
    print(f"V_ipe (IPE-Referenz): {V_ipe.round(4)}")

    # First-Visit MC mit Snapshots
    print(f"Starte First-Visit MC ({N_EPISODES} Episoden, seed={SEED}) ...")
    V_first, info_first = mc_policy_evaluation_V(
        env, policy,
        n_episodes=N_EPISODES,
        first_visit=True,
        max_steps=MAX_STEPS,
        seed=SEED,
        snapshot_every=SNAPSHOT_EVERY,
    )
    err_first = float(np.abs(V_first - V_ipe).max())
    print(f"  Final |V_first - V_ipe|_inf = {err_first:.4f}")
    print(f"  Gesamtschritte: {info_first['total_steps']:,}")

    # Every-Visit MC mit Snapshots
    print(f"Starte Every-Visit MC ({N_EPISODES} Episoden, seed={SEED}) ...")
    V_every, info_every = mc_policy_evaluation_V(
        env, policy,
        n_episodes=N_EPISODES,
        first_visit=False,
        max_steps=MAX_STEPS,
        seed=SEED,
        snapshot_every=SNAPSHOT_EVERY,
    )
    err_every = float(np.abs(V_every - V_ipe).max())
    print(f"  Final |V_every - V_ipe|_inf = {err_every:.4f}")
    print(f"  Gesamtschritte: {info_every['total_steps']:,}")

    # Plots
    plot_convergence(
        info_first["snapshots"],
        info_every["snapshots"],
        V_ipe,
        save_path=FIGURES_DIR / "monte_carlo_convergence.png",
    )

    plot_heatmaps(
        V_first, V_every, V_ipe,
        rows=env.rows, cols=env.cols,
        save_path=FIGURES_DIR / "monte_carlo_heatmaps.png",
    )

    print("Demo abgeschlossen.")
