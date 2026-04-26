"""Demo: SARSA auf 4x4 GridWorld (Skript Alg. 20, Übungsblatt 7 Aufg. 2).

Vergleich: konstantes Epsilon vs. linear abnehmend.
Zeigt: Konvergenz der erlernten Greedy-Policy zur optimalen Policy (pi*_VI)
       und den diskontieren Training-Return.

Hyperparameter:
    n_episodes    = 5000
    max_steps     = 300
    gamma         = 0.9
    alpha         = 0.1 (konstant)
    eps_const     = 0.15
    eps_decay     = linear_decay(0.5, 0.01, 5000)
    seed          = 42
    smooth_window = 100
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ...envs.gridworld import GridWorld
from ...algos.value_iteration import value_iteration_V, greedy_policy_from_v
from ...algos.sarsa import sarsa
from ...algos.schedules import constant, linear_decay

FIGURES_DIR = Path(__file__).parent / ".." / ".." / ".." / "figures"

N_EPISODES = 5000
MAX_STEPS = 300
GAMMA = 0.9
SEED = 42
SMOOTH = 100


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


def _smooth(arr: list, w: int) -> np.ndarray:
    return np.convolve(np.array(arr, dtype=float), np.ones(w) / w, mode="valid")


def _visualize_policy(
    env: GridWorld, Q: np.ndarray, ax: plt.Axes, title: str
) -> None:
    """Zeichnet die Greedy-Policy als Pfeile + V-Werte."""
    ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_aspect("equal")
    ax.set_xticks(range(env.cols + 1))
    ax.set_yticks(range(env.rows + 1))
    ax.tick_params(labelleft=False, labelbottom=False)
    ax.set_title(title, fontsize=9)

    for s in env.states:
        r, c = divmod(s, env.cols)
        y = env.rows - 1 - r
        acts = env.allowed_actions[s]
        if env.is_terminal(s):
            ax.text(c + 0.5, y + 0.5, "G", ha="center", va="center",
                    fontsize=12, color="green", fontweight="bold")
        else:
            best_a = acts[int(np.argmax(Q[s, acts]))]
            ax.text(c + 0.5, y + 0.5, ARROWS[best_a],
                    ha="center", va="center", fontsize=14, color="#1a5276")

    for x in range(env.cols + 1):
        ax.axvline(x, color="#cccccc", linewidth=0.5)
    for yy in range(env.rows + 1):
        ax.axhline(yy, color="#cccccc", linewidth=0.5)


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    env = build_env()
    V_star, _ = value_iteration_V(env, tol=1e-9)
    pi_star = greedy_policy_from_v(env, V_star)

    configs = [
        ("SARSA konst. ε=0.15", constant(0.15)),
        ("SARSA ε→0.01 (linear decay)", linear_decay(0.5, 0.01, N_EPISODES)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for i, (label, eps_sched) in enumerate(configs):
        Q, info = sarsa(
            env,
            alpha=constant(0.1),
            epsilon=eps_sched,
            n_episodes=N_EPISODES,
            max_steps=MAX_STEPS,
            seed=SEED,
        )

        smoothed = _smooth(info["episode_returns"], SMOOTH)
        ep_axis = np.arange(SMOOTH, N_EPISODES + 1)
        axes[0].plot(ep_axis, smoothed, label=label, linewidth=1.6)

        agree = sum(
            1 for s in env.states
            if np.argmax(Q[s, env.allowed_actions[s]]) ==
               np.argmax(pi_star[s, env.allowed_actions[s]])
        )
        print(f"{label}: finale Übereinstimmung {agree}/16 mit pi*_VI")

        if i == 1:
            _visualize_policy(env, Q, axes[2], f"Greedy Policy\n({label})")

    # Policy des ersten Configs
    Q_const, _ = sarsa(env, alpha=constant(0.1), epsilon=constant(0.15),
                       n_episodes=N_EPISODES, max_steps=MAX_STEPS, seed=SEED)
    _visualize_policy(env, Q_const, axes[1], "Greedy Policy\n(SARSA konst. ε=0.15)")

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel(f"Return (EMA w={SMOOTH})")
    axes[0].set_title(
        f"SARSA Return — 4x4 GridWorld (γ={GAMMA}, α=0.1, seed={SEED})"
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.4)

    fig.suptitle(
        f"SARSA: konst. ε vs. linear decay (n={N_EPISODES}, max_steps={MAX_STEPS})",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "sarsa_comparison.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gespeichert: {out}")
