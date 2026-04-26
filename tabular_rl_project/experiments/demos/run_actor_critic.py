"""Demo: General Actor-Critic (Skript Alg. 11) mit verschiedenen Critics.

Vergleicht drei Critic-Varianten auf 4x4 GridWorld (gamma=0.9):
    - ipe_critic(steps=1)    1 IPE-Schritt pro Outer-Iteration
    - ipe_critic(steps=5)    5 IPE-Schritte
    - ipe_critic("exact")    IPE bis Konvergenz (äquivalent zu Policy Iteration)

Zeigt: Outer-Iterationen bis Konvergenz + finaler Vergleich mit pi*_VI.

Hyperparameter:
    n_outer_iter  = 50
    gamma         = 0.9
    actor_update  = "greedy"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ...envs.gridworld import GridWorld
from ...algos.value_iteration import value_iteration_V, greedy_policy_from_v
from ...algos.actor_critic import actor_critic, make_ipe_critic

FIGURES_DIR = Path(__file__).parent / ".." / ".." / ".." / "figures"

N_OUTER = 50
GAMMA = 0.9


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


def _agreement(policy: np.ndarray, pi_star: np.ndarray, env: GridWorld) -> int:
    return sum(
        1 for s in env.states
        if np.argmax(policy[s, env.allowed_actions[s]]) ==
           np.argmax(pi_star[s, env.allowed_actions[s]])
    )


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    env = build_env()
    V_star, _ = value_iteration_V(env, tol=1e-9)
    pi_star = greedy_policy_from_v(env, V_star)

    configs = [
        ("ipe_critic(steps=1)", make_ipe_critic(steps=1)),
        ("ipe_critic(steps=5)", make_ipe_critic(steps=5)),
        ('ipe_critic("exact")', make_ipe_critic(steps="exact")),
    ]

    print(f"{'Critic':<25}  {'Outer-Iter':>10}  {'Konvergiert':>12}  {'Übereinstimmung':>16}")
    print("-" * 70)

    results = []
    for label, critic_fn in configs:
        V, policy, info = actor_critic(
            env,
            critic=critic_fn,
            actor_update="greedy",
            gamma=GAMMA,
            n_outer_iter=N_OUTER,
        )
        agree = _agreement(policy, pi_star, env)
        results.append((label, info["outer_iterations"], info["converged"], agree))
        print(
            f"{label:<25}  {info['outer_iterations']:>10}  "
            f"{'Ja' if info['converged'] else 'Nein':>12}  "
            f"{agree}/16"
        )

    # V-Heatmaps für alle drei Critics
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, (label, critic_fn) in zip(axes, configs):
        V, policy, info = actor_critic(
            env, critic=critic_fn, actor_update="greedy",
            gamma=GAMMA, n_outer_iter=N_OUTER,
        )
        grid_V = V.reshape(env.rows, env.cols)
        im = ax.imshow(grid_V, cmap="viridis", origin="upper")
        ax.set_title(
            f"{label}\n"
            f"({info['outer_iterations']} Outer-Iter, "
            f"{'konv.' if info['converged'] else 'nicht konv.'})",
            fontsize=9,
        )
        ax.set_xlabel("Spalte")
        ax.set_ylabel("Zeile")
        ax.set_xticks(range(env.cols))
        ax.set_yticks(range(env.rows))
        for r in range(env.rows):
            for c in range(env.cols):
                ax.text(c, r, f"{grid_V[r, c]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if grid_V[r, c] < grid_V.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Actor-Critic V^π — 4x4 GridWorld (γ={GAMMA}, actor=greedy)",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "actor_critic_comparison.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nGespeichert: {out}")
