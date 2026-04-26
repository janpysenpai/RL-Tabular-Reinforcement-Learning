"""Demo: SARSA vs. Q-Learning auf Cliff Walk (Sutton & Barto Bsp. 6.6).

Zeigt den klassischen Effekt:
    - Q-Learning (off-policy): wählt den kürzesten (cliff-nahen) Pfad,
      hat aber im Training viele Cliff-Stürze durch epsilon-Exploration.
    - SARSA (on-policy): meidet den Cliff durch epsilon-Exploration,
      wählt den sichereren, etwas längeren Pfad.

Hyperparameter:
    n_episodes  = 500
    max_steps   = 200
    alpha       = 0.1  (konstant)
    epsilon     = 0.1  (konstant)
    gamma       = 1.0
    seeds       = 5
    smooth_window = 10
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ...envs.cliff_walk import CliffWalk
from ...algos.q_learning import q_learning
from ...algos.sarsa import sarsa
from ...algos.schedules import constant

FIGURES_DIR = Path(__file__).parent / ".." / ".." / ".." / "figures"

N_EPISODES  = 500
MAX_STEPS   = 200
ALPHA       = 0.1
EPSILON     = 0.1
GAMMA       = 1.0
N_SEEDS     = 5
SMOOTH      = 10


def _smooth(arr: list, w: int) -> np.ndarray:
    return np.convolve(np.array(arr, dtype=float), np.ones(w) / w, mode="valid")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    env = CliffWalk(gamma=GAMMA)

    # Mittlerer Return über Seeds
    returns_q = np.zeros(N_EPISODES)
    returns_s = np.zeros(N_EPISODES)

    Q_last = None
    S_last = None

    for seed in range(N_SEEDS):
        Q, info_q = q_learning(env, alpha=constant(ALPHA), epsilon=constant(EPSILON),
                               n_episodes=N_EPISODES, max_steps=MAX_STEPS, seed=seed)
        returns_q += np.array(info_q["episode_returns"])

        S, info_s = sarsa(env, alpha=constant(ALPHA), epsilon=constant(EPSILON),
                          n_episodes=N_EPISODES, max_steps=MAX_STEPS, seed=seed)
        returns_s += np.array(info_s["episode_returns"])

        if seed == 0:
            Q_last, S_last = Q, S

    returns_q /= N_SEEDS
    returns_s /= N_SEEDS

    # ------------------------------------------------------------------
    # Plot 1: Return-Verlauf
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ep_axis = np.arange(SMOOTH, N_EPISODES + 1)
    axes[0].plot(ep_axis, _smooth(returns_q, SMOOTH), label="Q-Learning", linewidth=1.6)
    axes[0].plot(ep_axis, _smooth(returns_s, SMOOTH), label="SARSA", linewidth=1.6)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel(f"Return (Mittelwert {N_SEEDS} Seeds, Glättung w={SMOOTH})")
    axes[0].set_title(
        f"Cliff Walk — Return (γ={GAMMA}, α={ALPHA}, ε={EPSILON})"
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.4)
    axes[0].set_ylim(-200, 0)

    # ------------------------------------------------------------------
    # Plot 2: Policies nebeneinander
    # ------------------------------------------------------------------
    ax_q = fig.add_subplot(1, 2, 2)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))

    for ax, Q_pol, label in [(axes2[0], Q_last, "Q-Learning"), (axes2[1], S_last, "SARSA")]:
        env.visualize_policy(Q_pol, ax=ax, title=f"Greedy Policy — {label}")

    fig2.suptitle(
        f"Cliff Walk — Greedy Policies (seed=0, n={N_EPISODES}, α={ALPHA}, ε={EPSILON})",
        fontsize=11,
    )
    fig2.tight_layout()

    fig.tight_layout()
    out1 = FIGURES_DIR / "cliff_walk_returns.png"
    out2 = FIGURES_DIR / "cliff_walk_policies.png"
    fig.savefig(str(out1), dpi=150, bbox_inches="tight")
    fig2.savefig(str(out2), dpi=150, bbox_inches="tight")
    plt.close("all")

    print(f"Gespeichert: {out1}")
    print(f"Gespeichert: {out2}")

    # Finale Greedy-Policy-Länge (Anzahl Schritte ohne epsilon)
    for label, Q_pol in [("Q-Learning", Q_last), ("SARSA", S_last)]:
        env.reset()
        steps = 0
        s = env.start_state
        while not env.is_terminal(s) and steps < 50:
            acts = env.allowed_actions[s]
            a = acts[int(np.argmax(Q_pol[s, acts]))]
            s, _, done = env.step(a)
            steps += 1
            if done:
                break
        env._current_state = env.start_state
        print(f"{label}: Greedy-Pfad {steps} Schritte (optimal=13)")
