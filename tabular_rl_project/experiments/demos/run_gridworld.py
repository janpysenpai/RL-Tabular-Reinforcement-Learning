"""Demo: GridWorld instanziieren, Schritte rollen, Layout plotten.

Baut ein 4x4-Grid (Start oben links, Goal unten rechts, R=1),
druckt P- und r-Formen, rollt 5 zufällige Schritte und validiert
die Übergangsmatrix sowie eine Greedy-Policy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Paket-Import (3 Ebenen hoch: demos -> experiments -> tabular_rl_project)
from ...envs.gridworld import GridWorld

FIGURES_DIR = Path(__file__).parents[3] / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SEED = 42


def main() -> None:
    np.random.seed(SEED)

    # ------------------------------------------------------------------
    # Umgebung: 4x4-Grid, Start (0,0) oben links, Goal (3,3) unten rechts
    # ------------------------------------------------------------------
    env = GridWorld(
        rows=4,
        cols=4,
        start=(0, 0),
        terminal_states=[(3, 3)],
        cell_rewards={(3, 3): 1.0},
        default_reward=0.0,
        gamma=1.0,
        wall_behavior="stay",
    )

    # ------------------------------------------------------------------
    # Form von P und r
    # ------------------------------------------------------------------
    print("=== Übergangsmatrix und Reward-Tabelle ===")
    print(f"P.shape  = {env.transition_probabilities.shape}  (Zustände x Aktionen x Zustände)")
    print(f"r.shape  = {env.expected_rewards.shape}           (Zustände x Aktionen)")
    print(f"Zustände = {env.n_states},  Aktionen = {env.n_actions}")
    print(f"Startindex: {env.start_state}, Terminalindizes: {env.terminal_states}")
    print()

    # ------------------------------------------------------------------
    # Zeilensummen-Assertion
    # ------------------------------------------------------------------
    env.validate()
    print("validate() bestanden: P[s, a, :] summiert fuer alle (s, a) zu 1.")
    print()

    # ------------------------------------------------------------------
    # 5 zufällige Schritte
    # ------------------------------------------------------------------
    print("=== 5 zufaellige Schritte ===")
    state = env.reset()
    for step_idx in range(5):
        action = int(np.random.choice(env.allowed_actions[state]))
        next_state, reward, done = env.step(action)
        r, c = env._to_rc(state)
        nr, nc = env._to_rc(next_state)
        a_name = env.ACTION_NAMES[action]
        print(
            f"  Schritt {step_idx + 1}: s=({r},{c}) [{state:2d}]  "
            f"a={a_name:<6}  r={reward:.2f}  "
            f"s'=({nr},{nc}) [{next_state:2d}]  done={done}"
        )
        state = next_state
        if done:
            break
    print()

    # ------------------------------------------------------------------
    # Greedy-Policy "immer rechts, sonst runter" — muss in <= 6 Schritten
    # ins Goal führen (optimaler Pfad: 3 rechts + 3 runter = 6 Schritte)
    # ------------------------------------------------------------------
    print("=== Greedy-Policy Validierung ===")
    env.reset()
    steps_taken = 0
    greedy_state = env.start_state
    env._current_state = greedy_state

    while not env.is_terminal(greedy_state):
        r, c = env._to_rc(greedy_state)
        # Bevorzuge rechts, dann unten (deterministisch)
        if c < env.cols - 1:
            action = env.RIGHT
        else:
            action = env.DOWN
        greedy_state, _, _ = env.step(action)
        steps_taken += 1
        assert steps_taken <= 6, (
            f"Greedy-Policy hat nach {steps_taken} Schritten noch kein Ziel erreicht."
        )

    assert env.is_terminal(greedy_state), "Greedy-Policy hat Zielzustand nicht erreicht."
    print(f"Greedy-Policy erreicht Goal in {steps_taken} Schritten (erwartet: 6). OK.")
    print()

    # ------------------------------------------------------------------
    # Plot: Grid-Layout speichern
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    env.visualize_layout(
        ax=ax,
        title="4x4 GridWorld (deterministisch)\nStart=(0,0), Goal=(3,3), R=1, gamma=1.0",
        special_labels={(0, 0): "S\n(Start)", (3, 3): "G\n(R=1.0)"},
    )
    fig.tight_layout()
    save_path = FIGURES_DIR / "demo_gridworld.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Layout gespeichert unter: {save_path}")


if __name__ == "__main__":
    main()
