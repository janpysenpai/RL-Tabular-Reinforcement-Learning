"""Submission b)  Finite-time vs. discounted MDPs.

Zwei konkrete Beispiele zeigen, wann die Planungshorizont-Wahl die optimale
Policy qualitativ ändert.

Beispiel 1 — Zwei-Wege-MDP (6 Zustände):
    s=0 (Start) hat zwei Aktionen:
        A=0 → Nahziel (s=1, Terminal, R=0.4) in 1 Schritt
        A=1 → Fernziel (s=5, Terminal, R=1.0) in 4 Schritten
    Finite-time T=1: nur Nahziel erreichbar → wählt A (V*_0=0.4).
    Finite-time T=4: Fernziel erreichbar, keine Diskontierung → wählt B (V*_0=1.0).
    Diskontiert γ=0.3: V*(0→A)=0.3·0.4=0.12 > V*(0→B)=0.3^3·1.0=0.027 → wählt A.
    Diskontiert γ=0.9: V*(0→B)=0.9^3·1.0=0.729 > V*(0→A)=0.9·0.4=0.36 → wählt B.

Beispiel 2 — Submission GridWorld (4×4):
    Finite-time T=3: nur Fake Goal F erreichbar (3 LEFT-Schritte), V*_0=0.65.
    Finite-time T=5: wahres Ziel G erreichbar (5 Schritte), keine Diskontierung,
                     V*_0=1.0 > V*_0=0.65 → wählt G.
    Diskontiert γ=0.9: V*(start→F)=0.9^3·0.65≈0.474,
                       V*(start→G)=0.9^5·1.0≈0.590 → wählt G.

Umgebung: 6-Zustands-MDP + 4×4 Submission GridWorld, γ=0.9.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ...envs.gridworld_4x4_submission import (
    GridWorld4x4Submission,
    START_RC, FAKE_GOAL_RC, GOAL_RC,
)
from ...envs.extreme_mdps import _SimpleMDP
from ...algos.finite_time_dp import finite_time_optimal_control
from ...algos.value_iteration import value_iteration_V

FIGURES_DIR = Path(__file__).parent / ".." / ".." / ".." / "figures" / "submission"
RESULTS_DIR = Path(__file__).parent / ".." / ".." / ".." / "results" / "submission"

GAMMA = 0.9
NEAR_REWARD = 0.4
FAR_REWARD  = 1.0
FAR_STEPS   = 4  # Anzahl Schritte von s=0 zu s=5 (Fernziel)


def _build_two_path_mdp(gamma: float = GAMMA) -> _SimpleMDP:
    """6-Zustands-MDP mit Nahziel (R=0.4 in 1 Schritt) und Fernziel (R=1.0 in 4 Schritten)."""
    S, A = 6, 2
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))

    # s=0: a=0 → Nahziel (s=1), a=1 → erster Schritt zum Fernziel (s=2)
    P[0, 0, 1] = 1.0;  R[0, 0] = NEAR_REWARD
    P[0, 1, 2] = 1.0;  R[0, 1] = 0.0
    # s=1: Terminal (Nahziel), absorbierend
    P[1, :, 1] = 1.0
    # s=2,3,4: Zwischenzustände, beide Aktionen → nächster Zustand
    for s, ns in [(2, 3), (3, 4)]:
        P[s, 0, ns] = 1.0
        P[s, 1, ns] = 1.0
    # s=4: → Terminal (Fernziel, s=5)
    P[4, 0, 5] = 1.0;  R[4, 0] = FAR_REWARD
    P[4, 1, 5] = 1.0;  R[4, 1] = FAR_REWARD
    # s=5: Terminal (Fernziel), absorbierend
    P[5, :, 5] = 1.0

    env = _SimpleMDP(gamma=gamma)
    env.states = list(range(S))
    env.actions = [0, 1]
    env.start_state = 0
    env.terminal_states = {1, 5}
    env.allowed_actions = {s: [0, 1] for s in range(S)}
    env.transition_probabilities = P
    env.expected_rewards = R
    env._current_state = 0
    return env


def _finite_v0_over_t(env: _SimpleMDP, t_max: int) -> List[float]:
    """Berechnet V*_0(s=0) für T=1..t_max unter finite-time (keine Diskontierung)."""
    vals = []
    for t in range(1, t_max + 1):
        V, _ = finite_time_optimal_control(env, T=t)
        vals.append(float(V[0, 0]))  # V[t=0, s=0]
    return vals


def _discounted_v0(env: _SimpleMDP, gamma: float) -> float:
    """Berechnet V*(s=0) unter discounted infinite-horizon VI."""
    env_copy = _SimpleMDP(gamma=gamma)
    env_copy.states = env.states
    env_copy.actions = env.actions
    env_copy.start_state = env.start_state
    env_copy.terminal_states = env.terminal_states
    env_copy.allowed_actions = env.allowed_actions
    env_copy.transition_probabilities = env.transition_probabilities
    env_copy.expected_rewards = env.expected_rewards
    env_copy._current_state = 0
    V, _ = value_iteration_V(env_copy, gamma=gamma, tol=1e-12)
    return float(V[0])


def _rc(r: int, c: int) -> int:
    return r * 4 + c


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------ Beispiel 1: Zwei-Wege-MDP ------
    env2p = _build_two_path_mdp(gamma=GAMMA)
    T_MAX = 8
    t_vals = list(range(1, T_MAX + 1))
    finite_v0 = _finite_v0_over_t(env2p, T_MAX)

    gammas_disc = [0.3, 0.6, 0.9]
    disc_v0 = {g: _discounted_v0(env2p, g) for g in gammas_disc}

    # ------ Beispiel 2: Submission GridWorld ------
    sg = GridWorld4x4Submission(noise=False, gamma=GAMMA)
    V_disc, _ = value_iteration_V(sg, tol=1e-12)
    start_s  = _rc(*START_RC)
    fake_s   = _rc(*FAKE_GOAL_RC)
    goal_s   = _rc(*GOAL_RC)

    # Finite-time für T=1..8
    sg_finite_start = []
    sg_finite_policy = []
    for t in range(1, T_MAX + 1):
        V_t, pi_t = finite_time_optimal_control(sg, T=t)
        sg_finite_start.append(float(V_t[0, start_s]))
        sg_finite_policy.append(int(pi_t[0, start_s]))  # Aktion bei t=0

    # ------ Plot ------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Zwei-Wege-MDP
    ax = axes[0]
    ax.plot(t_vals, finite_v0, "o-", lw=2, color="C0", label="Finite-time T")
    for g, col in zip(gammas_disc, ["C3", "C1", "C2"]):
        ax.axhline(disc_v0[g], linestyle="--", color=col, alpha=0.8,
                   label=f"Diskontiert γ={g}  (V*={disc_v0[g]:.3f})")
    ax.axhline(NEAR_REWARD, linestyle=":", color="gray", alpha=0.6, label=f"Nahziel-R={NEAR_REWARD}")
    ax.axhline(FAR_REWARD,  linestyle=":", color="black", alpha=0.6, label=f"Fernziel-R={FAR_REWARD}")
    ax.set_xlabel("Zeithorizont T")
    ax.set_ylabel("V*(s=0)")
    ax.set_title("Zwei-Wege-MDP\n(Nahziel R=0.4 in 1 Schritt, Fernziel R=1.0 in 4 Schritten)")
    ax.legend(fontsize=8, loc="center right")
    ax.set_xticks(t_vals)
    ax.grid(alpha=0.4)
    ax.annotate("T=1: nur\nNahziel\nerreichbar",
                xy=(1, NEAR_REWARD), xytext=(2, 0.55),
                arrowprops=dict(arrowstyle="->", color="C0"), fontsize=8, color="C0")
    ax.annotate("T=4: Fernziel\nerreichbar\n(kein Diskont.)",
                xy=(4, FAR_REWARD), xytext=(5, 0.85),
                arrowprops=dict(arrowstyle="->", color="C0"), fontsize=8, color="C0")

    # Panel 2: Submission GridWorld
    ax = axes[1]
    colors_sg = ["C3" if v == 0.0 else ("C2" if v >= FAR_REWARD * 0.98 else "C0")
                 for v in sg_finite_start]
    ax.bar(t_vals, sg_finite_start, color=["C0"] * len(t_vals), alpha=0.7, label="V*_0(Start) finite-time T")
    ax.axhline(float(V_disc[start_s]), linestyle="--", color="C2", lw=2,
               label=f"Diskontiert γ={GAMMA}  (V*={V_disc[start_s]:.3f})")
    ax.axhline(0.65, linestyle=":", color="C3", alpha=0.7, label="Fake Goal R=0.65")
    ax.axhline(1.0,  linestyle=":", color="black", alpha=0.7, label="True Goal R=1.0")

    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    for t, v, a in zip(t_vals, sg_finite_start, sg_finite_policy):
        if v > 0.01:
            ax.text(t, v + 0.02, ACTION_NAMES.get(a, "?"), ha="center", fontsize=7, color="#2c3e50")

    ax.set_xlabel("Zeithorizont T")
    ax.set_ylabel("V*_0(Start)")
    ax.set_title("4×4 Submission GridWorld\n(F=Fake Goal R=0.65, 3 Schritte; G=Goal R=1.0, 5 Schritte)")
    ax.legend(fontsize=8)
    ax.set_xticks(t_vals)
    ax.grid(alpha=0.4)

    fig.suptitle(
        "Aufgabe b) — Finite-time vs. Diskontiert: Wann wechselt die optimale Policy?",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "task_b_finite_vs_discounted.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot:  {out}")

    # ------ Konsolen-Zusammenfassung ------
    print("\n=== Aufgabe b) — Finite-time vs. Diskontiert ===")
    print("\nZwei-Wege-MDP (Nahziel R=0.4 in 1 Schritt, Fernziel R=1.0 in 4 Schritten):")
    print(f"  Finite T=1: V*(s=0)={finite_v0[0]:.3f}  → Nahziel (A=0)")
    print(f"  Finite T=4: V*(s=0)={finite_v0[3]:.3f}  → Fernziel (A=1, kein Diskont.)")
    for g in gammas_disc:
        pref = "Nahziel" if disc_v0[g] <= NEAR_REWARD + 0.01 else "Fernziel"
        print(f"  Diskontiert γ={g}: V*(s=0)={disc_v0[g]:.3f}  → {pref}")

    print(f"\nSubmission GridWorld (F=R=0.65 in 3 Schritten, G=R=1.0 in 5 Schritten):")
    for t in [3, 4, 5, 6]:
        v = sg_finite_start[t - 1]
        a = ACTION_NAMES.get(sg_finite_policy[t - 1], "?")
        print(f"  Finite T={t}: V*_0(Start)={v:.3f}  Aktion @ t=0: {a}")
    print(f"  Diskontiert γ={GAMMA}: V*(Start)={V_disc[start_s]:.3f}")

    out_json = RESULTS_DIR / "task_b.json"
    with open(out_json, "w") as f:
        json.dump({
            "example1_two_path_mdp": {
                "near_reward": NEAR_REWARD, "far_reward": FAR_REWARD,
                "finite_v0_by_T": {t: v for t, v in zip(t_vals, finite_v0)},
                "discounted_v0_by_gamma": {str(g): disc_v0[g] for g in gammas_disc},
            },
            "example2_submission_grid": {
                "gamma": GAMMA,
                "finite_v0_start_by_T": {t: v for t, v in zip(t_vals, sg_finite_start)},
                "discounted_v0_start": float(V_disc[start_s]),
            },
        }, f, indent=2)
    print(f"JSON:  {out_json}")


if __name__ == "__main__":
    main()
