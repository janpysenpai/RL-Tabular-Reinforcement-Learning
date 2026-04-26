"""Demo: Q-Learning auf 4x4 GridWorld (Skript Alg. 18 / 19).

Vergleich: konstantes alpha/epsilon vs. polynomial-decay alpha.
Metriken über Episoden:
    - Geglätteter Training-Return (EMA)
    - |argmax Q - pi*_VI|-Übereinstimmungsrate

Hyperparameter:
    n_episodes    = 5000
    max_steps     = 300
    gamma         = 0.9
    alpha_const   = 0.1        epsilon_const = 0.15
    alpha_poly    = polynomial(0.7)
    epsilon_poly  = linear_decay(0.5, 0.02, 5000)
    seed          = 42
    smooth_window = 100
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ...envs.gridworld import GridWorld
from ...algos.value_iteration import value_iteration_V, greedy_policy_from_v
from ...algos.q_learning import q_learning
from ...algos.schedules import constant, polynomial, linear_decay

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


def _agreement_curve(Q_snapshots: list, pi_star: np.ndarray, env: GridWorld) -> list:
    """Gibt Übereinstimmungsrate (Anteil 0..1) pro Snapshot zurück."""
    rates = []
    for ep, Q in Q_snapshots:
        match = sum(
            1 for s in env.states
            if np.argmax(Q[s, env.allowed_actions[s]]) ==
               np.argmax(pi_star[s, env.allowed_actions[s]])
        )
        rates.append((ep, match / env.n_states))
    return rates


def _smooth(arr: list, w: int) -> np.ndarray:
    arr = np.array(arr, dtype=float)
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="valid")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    env = build_env()

    # Optimale Policy (Ground Truth)
    V_star, _ = value_iteration_V(env, tol=1e-9)
    pi_star = greedy_policy_from_v(env, V_star)
    print(f"V*(start) = {V_star[env.start_state]:.4f}")

    configs = [
        ("const α=0.1, ε=0.15",
         constant(0.1), constant(0.15)),
        ("poly α=1/n^0.7, ε→0.02",
         polynomial(0.7), linear_decay(0.5, 0.02, N_EPISODES)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for label, alpha_sched, eps_sched in configs:
        # Snapshots für Übereinstimmungskurve
        snap_every = 50
        Q_snaps: list = []

        def _eval_fn(Q: np.ndarray) -> np.ndarray:
            Q_snaps.append(Q.copy())
            return float(sum(
                1 for s in env.states
                if np.argmax(Q[s, env.allowed_actions[s]]) ==
                   np.argmax(pi_star[s, env.allowed_actions[s]])
            ) / env.n_states)

        Q, info = q_learning(
            env,
            alpha=alpha_sched,
            epsilon=eps_sched,
            n_episodes=N_EPISODES,
            max_steps=MAX_STEPS,
            seed=SEED,
            eval_every=snap_every,
            eval_fn=_eval_fn,
        )

        returns = info["episode_returns"]
        eval_hist = info["eval_history"]  # list of (ep, agreement_rate)

        # Return-Kurve
        smoothed = _smooth(returns, SMOOTH)
        ep_axis = np.arange(SMOOTH, N_EPISODES + 1)
        axes[0].plot(ep_axis, smoothed, label=label, linewidth=1.6)

        # Übereinstimmungskurve
        eps_agree, rates_agree = zip(*eval_hist)
        axes[1].plot(eps_agree, rates_agree, label=label, linewidth=1.6)

        final_agree = rates_agree[-1]
        print(f"{label}: finale Übereinstimmung = {final_agree:.2%}")

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel(f"Return (EMA w={SMOOTH})")
    axes[0].set_title(
        f"Q-Learning Return — 4x4 GridWorld (γ={GAMMA}, seed={SEED})"
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.4)

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Übereinstimmung mit pi*_VI")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("argmax Q vs. pi*_VI — Anteil korrekte Zustände")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.4)

    fig.suptitle(
        f"Q-Learning: const vs. decay Schedule (n={N_EPISODES}, max_steps={MAX_STEPS})",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "q_learning_comparison.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gespeichert: {out}")
