"""Submission f)  Optimale Hyperparameter für Q-/Double-Q auf dem 4×4-Grid.

Umgebung: GridWorld4x4Submission (noise=False und noise=True), γ=0.9.

    F . . S       F = Fake Goal (R=0.65, terminal, oben links)
    . . . .       S = Start (oben rechts)
    . . SR SR     G = Goal (R=1.0, terminal, unten zweite Spalte)
    . G SR SR     SR = Stochastic Region (R∈{-2.1,+2.0}, p=0.5)
                  Default: E[R]=0 (noise=F) / R∈{-0.05,+0.05} p=0.5 (noise=T)

Suchraum (Grid Search):
    alpha  ∈ {0.05, 0.1, 0.3, 0.5}
    epsilon ∈ {0.05, 0.1, 0.2, 0.3}
    → 4×4 = 16 Kombinationen, je N_SEEDS=3 Seeds, Q-Learning und Double-Q.

Bewertungsmetrik:
    - Finale mittlere Episode Return (letzte 200 Episoden, mean ± std über Seeds)
    - Policy-Korrektheit: Anteil Zustände mit V*-optimaler Aktion

Vorgehen:
    1. Grid Search mit konstantem alpha/epsilon, 2000 Episoden.
    2. Beste Kombination: final_return.mean() maximal.
    3. Separater Lauf mit besten Params: Lernkurve + Policy-Plot.

Hyperparameter: N_SEEDS=3, N_EPISODES=2000, MAX_STEPS=200, last_n=200.
"""

from __future__ import annotations

import json
import matplotlib.patches as mpatches
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ...envs.gridworld_4x4_submission import GridWorld4x4Submission, START_RC
from ...algos.value_iteration import value_iteration_V
from ...algos.q_learning import q_learning
from ...algos.double_q_learning import double_q_learning
from ...algos.schedules import constant

FIGURES_DIR = Path(__file__).parent / ".." / ".." / ".." / "figures" / "submission"
RESULTS_DIR = Path(__file__).parent / ".." / ".." / ".." / "results" / "submission"

GAMMA      = 0.9
N_SEEDS    = 3
SEEDS      = list(range(N_SEEDS))
N_EPISODES = 2000
MAX_STEPS  = 200
LAST_N     = 200  # Letzte N Episoden für Return-Metrik

ALPHA_GRID   = [0.05, 0.1, 0.3, 0.5]
EPSILON_GRID = [0.05, 0.1, 0.2, 0.3]


def _rc(r: int, c: int) -> int:
    return r * 4 + c


def _policy_correct_rate(Q: np.ndarray, Q_star: np.ndarray, env) -> float:
    """Anteil Nicht-Terminal-Zustände mit Q*-optimaler Aktion."""
    correct, total = 0, 0
    for s in env.states:
        if env.is_terminal(s):
            continue
        acts = env.allowed_actions[s]
        best_q = float(Q_star[s, acts].max())
        chosen = acts[int(np.argmax(Q[s, acts]))]
        if Q_star[s, chosen] >= best_q - 1e-9:
            correct += 1
        total += 1
    return correct / total if total > 0 else 1.0


def _grid_search(noise: bool, algo: str) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Grid Search über alpha × epsilon.

    Returns:
        mean_grid: shape (len(ALPHA_GRID), len(EPSILON_GRID)) — finaler mean return.
        std_grid:  shape gleich — std über Seeds.
        best_alpha, best_epsilon: beste Kombination.
    """
    A, E = len(ALPHA_GRID), len(EPSILON_GRID)
    mean_grid = np.zeros((A, E))
    std_grid  = np.zeros((A, E))

    for ai, alpha in enumerate(ALPHA_GRID):
        for ei, epsilon in enumerate(EPSILON_GRID):
            returns_per_seed = []
            for seed in SEEDS:
                env = GridWorld4x4Submission(noise=noise, gamma=GAMMA, seed=seed * 31)
                if algo == "ql":
                    _, info = q_learning(
                        env, alpha=constant(alpha), epsilon=constant(epsilon),
                        n_episodes=N_EPISODES, max_steps=MAX_STEPS, seed=seed,
                    )
                    ep_returns = info["episode_returns"]
                else:
                    _, _, _, info = double_q_learning(
                        env, alpha=constant(alpha), epsilon=constant(epsilon),
                        n_episodes=N_EPISODES, max_steps=MAX_STEPS, seed=seed,
                    )
                    ep_returns = info["episode_returns"]
                returns_per_seed.append(float(np.mean(ep_returns[-LAST_N:])))
            mean_grid[ai, ei] = float(np.mean(returns_per_seed))
            std_grid[ai, ei]  = float(np.std(returns_per_seed))

    best_idx = np.unravel_index(np.argmax(mean_grid), mean_grid.shape)
    best_alpha   = ALPHA_GRID[best_idx[0]]
    best_epsilon = EPSILON_GRID[best_idx[1]]
    return mean_grid, std_grid, best_alpha, best_epsilon


def _learning_curve(noise: bool, algo: str, alpha: float, epsilon: float,
                    seeds: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Lernkurve für beste Params (geglättet, mean±std über seeds)."""
    all_returns = []
    for seed in seeds:
        env = GridWorld4x4Submission(noise=noise, gamma=GAMMA, seed=seed * 31)
        if algo == "ql":
            _, info = q_learning(env, alpha=constant(alpha), epsilon=constant(epsilon),
                                  n_episodes=N_EPISODES, max_steps=MAX_STEPS, seed=seed)
        else:
            _, _, _, info = double_q_learning(
                env, alpha=constant(alpha), epsilon=constant(epsilon),
                n_episodes=N_EPISODES, max_steps=MAX_STEPS, seed=seed,
            )
        all_returns.append(info["episode_returns"])
    L = min(len(r) for r in all_returns)
    mat = np.array([r[:L] for r in all_returns])
    return mat.mean(axis=0), mat.std(axis=0)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------ Referenz V* ------
    env_ref = GridWorld4x4Submission(noise=False, gamma=GAMMA)
    V_star, _ = value_iteration_V(env_ref, tol=1e-12)
    start_s = _rc(*START_RC)
    P, R = env_ref.transition_probabilities, env_ref.expected_rewards
    Q_star = R + GAMMA * np.einsum("sap,p->sa", P, V_star)

    results_json: Dict = {}

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    noise_configs = [(False, "noise=False"), (True, "noise=True")]

    for row_idx, (noise, noise_label) in enumerate(noise_configs):
        for col_idx, algo in enumerate(["ql", "dql"]):
            algo_label = "Q-Learning" if algo == "ql" else "Double Q-Learning"
            print(f"Grid Search: {algo_label}, {noise_label} ...")
            mean_grid, std_grid, best_a, best_e = _grid_search(noise, algo)

            # Heatmap
            ax = axes[row_idx, col_idx]
            im = ax.imshow(mean_grid, origin="upper", aspect="auto",
                           vmin=mean_grid.min(), vmax=mean_grid.max())
            plt.colorbar(im, ax=ax, label="Mittl. Return (letzte 200 Ep.)")
            ax.set_xticks(range(len(EPSILON_GRID)))
            ax.set_xticklabels([str(e) for e in EPSILON_GRID])
            ax.set_yticks(range(len(ALPHA_GRID)))
            ax.set_yticklabels([str(a) for a in ALPHA_GRID])
            ax.set_xlabel("epsilon")
            ax.set_ylabel("alpha")
            ax.set_title(f"{algo_label}\n{noise_label}\nBest: α={best_a}, ε={best_e}")

            # Markiere Bestpunkt mit rotem Rahmen (kein Überlapp mit Textwerten)
            bi = ALPHA_GRID.index(best_a)
            ei = EPSILON_GRID.index(best_e)
            ax.add_patch(mpatches.Rectangle(
                (ei - 0.5, bi - 0.5), 1.0, 1.0,
                fill=False, edgecolor="red", linewidth=3.0, zorder=5,
            ))

            # Annotiere alle Zellen mit mean ± std
            for ai, a_val in enumerate(ALPHA_GRID):
                for eii, e_val in enumerate(EPSILON_GRID):
                    ax.text(eii, ai,
                            f"{mean_grid[ai, eii]:.3f}\n±{std_grid[ai, eii]:.3f}",
                            ha="center", va="center", fontsize=6.5, color="white",
                            fontweight="bold")

            results_json[f"{algo}_{noise_label}"] = {
                "best_alpha": best_a, "best_epsilon": best_e,
                "best_mean_return": float(mean_grid[bi, ei]),
                "best_std_return":  float(std_grid[bi, ei]),
            }

        # Lernkurve mit besten Params (letztes Algo = Double-Q, nehme QL-Best für Vergleich)
        best_a_ql, best_e_ql = results_json[f"ql_{noise_label}"]["best_alpha"], \
                                results_json[f"ql_{noise_label}"]["best_epsilon"]
        best_a_dq, best_e_dq = results_json[f"dql_{noise_label}"]["best_alpha"], \
                                results_json[f"dql_{noise_label}"]["best_epsilon"]

        print(f"Lernkurve ({noise_label}): QL α={best_a_ql} ε={best_e_ql}, DQ α={best_a_dq} ε={best_e_dq}")
        ql_m, ql_s   = _learning_curve(noise, "ql",  best_a_ql, best_e_ql,  SEEDS)
        dql_m, dql_s = _learning_curve(noise, "dql", best_a_dq, best_e_dq, SEEDS)

        ax = axes[row_idx, 2]
        x = np.arange(1, min(len(ql_m), len(dql_m)) + 1)
        Lx = len(x)
        ax.plot(x, ql_m[:Lx],  label=f"Q-Learn α={best_a_ql} ε={best_e_ql}",  color="C0", lw=1.6)
        ax.fill_between(x, ql_m[:Lx] - ql_s[:Lx], ql_m[:Lx] + ql_s[:Lx], alpha=0.2, color="C0")
        ax.plot(x, dql_m[:Lx], label=f"Double-Q α={best_a_dq} ε={best_e_dq}", color="C2", lw=1.6)
        ax.fill_between(x, dql_m[:Lx] - dql_s[:Lx], dql_m[:Lx] + dql_s[:Lx], alpha=0.2, color="C2")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Return")
        ax.set_title(f"Beste Params — {noise_label}\n({N_SEEDS} Seeds)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.4)

    fig.suptitle(
        f"Aufgabe f) — Grid Search Q-Learning vs. Double-Q: 4×4 Submission Grid, γ={GAMMA}",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "task_f_4x4_gridworld.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot:  {out}")

    # ------ Konsolen-Zusammenfassung ------
    print(f"\n=== Aufgabe f) — Grid Search Ergebnisse ===")
    for key, val in results_json.items():
        print(f"  {key}: best α={val['best_alpha']}, ε={val['best_epsilon']}, "
              f"return={val['best_mean_return']:.4f}±{val['best_std_return']:.4f}")

    print(f"\nMethodik:")
    print(f"  - Suchraum: α∈{ALPHA_GRID}, ε∈{EPSILON_GRID}")
    print(f"  - Verfahren: Exhaustive Grid Search")
    print(f"  - Metrik: mittlerer Episode Return über letzte {LAST_N} Episoden")
    print(f"  - {N_SEEDS} Seeds × {N_EPISODES} Episoden × 16 Kombinationen × 2 Algorithmen × 2 Noise-Varianten")

    results_json["hyperparameters"] = {
        "gamma": GAMMA, "n_seeds": N_SEEDS, "n_episodes": N_EPISODES,
        "max_steps": MAX_STEPS, "last_n_for_metric": LAST_N,
        "alpha_grid": ALPHA_GRID, "epsilon_grid": EPSILON_GRID,
    }
    out_json = RESULTS_DIR / "task_f.json"
    with open(out_json, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"JSON:  {out_json}")


if __name__ == "__main__":
    main()
