"""Demo: Value Iteration auf den Extreme MDPs (Sanity-Check).

Führt auf jedem Extreme-MDP value_iteration_V aus und zeigt V* sowie
(wo analytisch bekannt) den Vergleich mit der exakten Lösung.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ...envs.extreme_mdps import (
    make_two_state_loop,
    make_sparse_chain,
    make_noisy_gridworld,
    make_bias_mdp_for_double_q,
)
from ...algos.value_iteration import value_iteration_V

FIGURES_DIR = Path(__file__).parent / ".." / ".." / ".." / "figures"


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # a) Zwei-Zustands-Loop
    # ------------------------------------------------------------------
    p_stay, gamma_loop = 0.9, 0.99
    env_loop = make_two_state_loop(p_stay=p_stay, gamma=gamma_loop)
    V_loop, _ = value_iteration_V(env_loop, tol=1e-9)
    V_analytic = 1.0 / (1.0 - gamma_loop * p_stay)
    print(f"[TwoStateLoop] p_stay={p_stay}, gamma={gamma_loop}")
    print(f"  V*(0) = {V_loop[0]:.6f}  (analytisch: {V_analytic:.6f})")
    print(f"  V*(1) = {V_loop[1]:.6f}  (analytisch: 0)")

    # ------------------------------------------------------------------
    # b) Sparse Chain
    # ------------------------------------------------------------------
    n_chain, gamma_chain = 10, 0.95
    env_chain = make_sparse_chain(n_states=n_chain, gamma=gamma_chain)
    V_chain, _ = value_iteration_V(env_chain, tol=1e-9)
    print(f"\n[SparseChain] n={n_chain}, gamma={gamma_chain}")
    for s in range(n_chain):
        expected = gamma_chain ** (n_chain - 2 - s) if s < n_chain - 1 else 0.0
        print(f"  V*({s}) = {V_chain[s]:.6f}  (analytisch: {expected:.6f})")

    # ------------------------------------------------------------------
    # c) Noisy GridWorld
    # ------------------------------------------------------------------
    env_noisy = make_noisy_gridworld(rows=5, cols=5, slip_prob=0.2, gamma=0.9)
    V_noisy, info_noisy = value_iteration_V(env_noisy, tol=1e-9)
    print(f"\n[NoisyGridWorld] 5x5, slip=0.2, gamma=0.9 — {info_noisy['iterations']} Iterationen")
    print(f"  V*(start=0)  = {V_noisy[0]:.4f}")
    print(f"  V*(goal=24)  = {V_noisy[24]:.4f}")

    # ------------------------------------------------------------------
    # d) Bias MDP
    # ------------------------------------------------------------------
    mean_r = -0.1
    env_bias = make_bias_mdp_for_double_q(n_acts=10, reward_mean=mean_r)
    V_bias, _ = value_iteration_V(env_bias, tol=1e-9)
    print(f"\n[BiasMDP] n_acts=10, reward_mean={mean_r}")
    print(f"  V*(0) = {V_bias[0]:.6f}  (analytisch: {mean_r:.6f})")
    print(f"  V*(1) = {V_bias[1]:.6f}  (analytisch: 0)")

    # ------------------------------------------------------------------
    # Plot: V* der Sparse Chain und NoisyGridWorld
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(n_chain), V_chain, color="#5dade2")
    axes[0].set_xlabel("Zustand")
    axes[0].set_ylabel("V*(s)")
    axes[0].set_title(f"Sparse Chain V* (n={n_chain}, γ={gamma_chain})")
    axes[0].grid(axis="y", alpha=0.4)
    for s in range(n_chain):
        axes[0].text(s, V_chain[s] + 0.01, f"{V_chain[s]:.2f}",
                     ha="center", va="bottom", fontsize=7)

    grid_V = V_noisy.reshape(5, 5)
    im = axes[1].imshow(grid_V, cmap="viridis", origin="upper")
    axes[1].set_title("Noisy GridWorld V* (5×5, slip=0.2, γ=0.9)")
    axes[1].set_xlabel("Spalte")
    axes[1].set_ylabel("Zeile")
    axes[1].set_xticks(range(5))
    axes[1].set_yticks(range(5))
    for r in range(5):
        for c in range(5):
            axes[1].text(c, r, f"{grid_V[r,c]:.2f}",
                         ha="center", va="center", fontsize=8,
                         color="white" if grid_V[r,c] < grid_V.max()/2 else "black")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("Extreme MDPs — V* via Value Iteration", fontsize=11)
    fig.tight_layout()
    out = FIGURES_DIR / "extreme_mdps_v_star.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGespeichert: {out}")
