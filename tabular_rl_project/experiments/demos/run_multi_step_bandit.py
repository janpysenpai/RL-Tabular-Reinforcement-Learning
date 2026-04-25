"""Demo: MultiStepBandit nach Abbildung 2 aus Übungsblatt 4.

Struktur (vgl. Abbildung 2):
    s0 → links  (2 Schritte): Schritt 1 r=0, Schritt 2: 3 Aktionen ~ N(1.0, 0.5)
    s0 → rechts (2 Schritte): Schritt 1 r=0, Schritt 2: 2 Aktionen ~ N(2.0, 0.5)
    s0 → runter (0 Schritte): sofort terminal, r=0
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ...envs.multi_step_bandit import MultiStepBandit

FIGURES_DIR = Path(__file__).parents[3] / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SEED = 42


def main() -> None:
    np.random.seed(SEED)

    env = MultiStepBandit(
        branch_specs=[
            # Branch 0 (links): 2 Schritte
            [
                [0.0],                                                            # Schritt 0: 1 Aktion, r=0
                [("normal", 1.0, 0.5), ("normal", 1.0, 0.5), ("normal", 1.0, 0.5)],  # Schritt 1: 3 Aktionen
            ],
            # Branch 1 (rechts): 2 Schritte
            [
                [0.0],                                          # Schritt 0: 1 Aktion, r=0
                [("normal", 2.0, 0.5), ("normal", 2.0, 0.5)],  # Schritt 1: 2 Aktionen
            ],
            # Branch 2 (runter): 0 Schritte → sofort Terminal
            [],
        ],
        root_rewards=[0.0, 0.0, 0.0],
        default_reward=0.0,
        gamma=1.0,
    )

    # ------------------------------------------------------------------
    # Struktur ausgeben
    # ------------------------------------------------------------------
    print("=== MultiStepBandit Struktur ===")
    print(f"Zustände:        {env.n_states}  (Indizes: {env.states})")
    print(f"Terminalzustände:{sorted(env.terminal_states)}")
    print(f"Max. Aktionen:   {env.n_actions}")
    print(f"P.shape:         {env.transition_probabilities.shape}")
    print(f"r.shape:         {env.expected_rewards.shape}")
    print()
    print("Erlaubte Aktionen pro Zustand:")
    for s in env.states:
        tag = " [Terminal]" if env.is_terminal(s) else ""
        print(f"  s={s}: {env.allowed_actions[s]}{tag}")
    print()

    # Zeilensummen prüfen
    env.validate()
    print("validate() bestanden: P[s, a, :] summiert fuer alle (s, a) zu 1.")
    print()

    # ------------------------------------------------------------------
    # 3 vollständige Trajektorien rollen
    # ------------------------------------------------------------------
    print("=== 3 Trajektorien ===")
    for ep in range(3):
        state = env.reset()
        trajectory = []
        total_r = 0.0
        while not env.is_terminal(state):
            action = int(np.random.choice(env.allowed_actions[state]))
            next_state, reward, done = env.step(action)
            trajectory.append((state, action, reward))
            total_r += reward
            state = next_state
        traj_str = "  ".join(f"s{s}→a{a}(r={r:.2f})" for s, a, r in trajectory)
        print(f"  Episode {ep + 1}: {traj_str}  | Gesamt-Reward: {total_r:.2f}")
    print()

    # ------------------------------------------------------------------
    # MC-Schätzer für ausgewählte (s, a)-Paare
    # ------------------------------------------------------------------
    print("=== MC-Schätzer E[R | s, a] (n=2000) ===")
    test_pairs = [
        (0, 0, "Wurzel → links"),
        (0, 1, "Wurzel → rechts"),
        (0, 2, "Wurzel → runter"),
        (2, 0, "links Schritt2, Aktion 0"),
        (2, 1, "links Schritt2, Aktion 1"),
        (5, 0, "rechts Schritt2, Aktion 0"),
    ]
    for s, a, desc in test_pairs:
        if a in env.allowed_actions[s]:
            mc = env.mc_estimate_reward(s, a, n=2000)
            ex = env.expected_rewards[s, a]
            print(f"  {desc:<30} E[R]={ex:.3f}  MC={mc:.3f}")
    print()

    # ------------------------------------------------------------------
    # Plot speichern
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    env.visualize(
        ax=ax,
        title="Multi-Step Bandit (Übungsblatt 4, Abb. 2)\n"
              "links: N(1.0,0.5) x3 | rechts: N(2.0,0.5) x2 | runter: Terminal",
        save_path=str(FIGURES_DIR / "demo_multi_step_bandit.png"),
    )
    plt.close(fig)
    print(f"Plot gespeichert unter: {FIGURES_DIR / 'demo_multi_step_bandit.png'}")


if __name__ == "__main__":
    main()
