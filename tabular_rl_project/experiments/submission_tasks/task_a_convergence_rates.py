"""Submission a)  Convergence Rates.

Empirisch: Konvergenzraten iterativer Methoden mit bekannter Dynamik
(Value Iteration, Iterative Policy Evaluation) vs. sample-basierte
Algorithmen (MC, TD(0), Q-Learning) gemessen in ‖V_k − V*‖∞.

Linkes Panel — model-basiert:
    VI und IPE (uniform policy): Fehler vs. DP-Iteration k.
    Gestrichelte Linie: theoretische Rate γ^k · ‖V_0 − V*‖.

Rechtes Panel — sample-basiert:
    MC (first-visit), TD(0), Q-Learning: Fehler vs. Episode.
    Mittelwert ± 1σ über N_SEEDS Seeds als shaded region.

Umgebung: 4×4 GridWorld, deterministisch, γ=0.9.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ...envs.gridworld import GridWorld
from ...algos.value_iteration import value_iteration_V
from ...algos.iterative_policy_evaluation import ipe_V
from ...algos.monte_carlo import mc_policy_evaluation_V
from ...algos.td_policy_evaluation import td0_policy_evaluation_V
from ...algos.q_learning import q_learning
from ...algos.schedules import constant
from ...algos.exploration import uniform_random_policy

FIGURES_DIR = Path(__file__).parent / ".." / ".." / ".." / "figures" / "submission"
RESULTS_DIR = Path(__file__).parent / ".." / ".." / ".." / "results" / "submission"

# ------------------------------------------------------------------
# Hyperparameter
# ------------------------------------------------------------------
GAMMA        = 0.9
N_SEEDS      = 5
SEEDS        = list(range(N_SEEDS))
N_EPISODES   = 3000
SNAPSHOT_EP  = 50
MAX_STEPS    = 200
MODEL_ITER   = 60

QUICK_MODE = False  # True: N_EPISODES→600, N_SEEDS→3


def _build_env() -> GridWorld:
    return GridWorld(
        rows=4, cols=4, start=(0, 0),
        terminal_states=[(3, 3)],
        cell_rewards={(3, 3): 1.0},
        default_reward=0.0,
        gamma=GAMMA, wall_behavior="stay",
    )


def _vi_error_curve(env: GridWorld, V_star: np.ndarray, n_iter: int) -> List[float]:
    P, R = env.transition_probabilities, env.expected_rewards
    V = np.zeros(env.n_states)
    errors = []
    for _ in range(n_iter):
        V = (R + GAMMA * np.einsum("sap,p->sa", P, V)).max(axis=1)
        errors.append(float(np.abs(V - V_star).max()))
    return errors


def _ipe_error_curve(env: GridWorld, policy: np.ndarray,
                     V_pi: np.ndarray, n_iter: int) -> List[float]:
    P, R = env.transition_probabilities, env.expected_rewards
    V = np.zeros(env.n_states)
    errors = []
    for _ in range(n_iter):
        Q = R + GAMMA * np.einsum("sap,p->sa", P, V)
        V = (policy * Q).sum(axis=1)
        errors.append(float(np.abs(V - V_pi).max()))
    return errors


def _align(lists: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    L = min(len(e) for e in lists)
    mat = np.array([e[:L] for e in lists])
    return mat.mean(axis=0), mat.std(axis=0)


def main() -> None:
    n_ep   = 600  if QUICK_MODE else N_EPISODES
    n_seeds = 3   if QUICK_MODE else N_SEEDS
    seeds  = SEEDS[:n_seeds]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env        = _build_env()
    V_star, _  = value_iteration_V(env, tol=1e-12)
    uniform_pi = uniform_random_policy(env)
    V_pi, _    = ipe_V(env, uniform_pi, tol=1e-12)

    # ------ Model-basiert ------
    vi_errs  = _vi_error_curve(env, V_star, MODEL_ITER)
    ipe_errs = _ipe_error_curve(env, uniform_pi, V_pi, MODEL_ITER)
    vi_theory = [vi_errs[0] * (GAMMA ** k) for k in range(MODEL_ITER)]

    # ------ Sample-basiert ------
    ep_axis = list(range(SNAPSHOT_EP, n_ep + 1, SNAPSHOT_EP))

    mc_runs, td_runs, ql_runs = [], [], []
    for seed in seeds:
        _, info = mc_policy_evaluation_V(
            env, uniform_pi, n_episodes=n_ep, max_steps=MAX_STEPS,
            seed=seed, snapshot_every=SNAPSHOT_EP,
        )
        mc_runs.append([float(np.abs(V - V_pi).max()) for _, V in info["snapshots"]])

        _, info = td0_policy_evaluation_V(
            env, uniform_pi, alpha=constant(0.1),
            n_episodes=n_ep, max_steps=MAX_STEPS,
            seed=seed, snapshot_every=SNAPSHOT_EP,
        )
        td_runs.append([float(np.abs(V - V_pi).max()) for _, V in info["snapshots"]])

        ql_errs_seed: List[float] = []
        def _ql_eval(Q: np.ndarray, _e: list = ql_errs_seed) -> float:
            v = float(np.abs(Q.max(axis=1) - V_star).max())
            _e.append(v)
            return v
        q_learning(env, alpha=constant(0.1), epsilon=constant(0.3),
                   n_episodes=n_ep, max_steps=MAX_STEPS,
                   seed=seed, eval_every=SNAPSHOT_EP, eval_fn=_ql_eval)
        ql_runs.append(ql_errs_seed)

    mc_m, mc_s = _align(mc_runs)
    td_m, td_s = _align(td_runs)
    ql_m, ql_s = _align(ql_runs)

    # ------ Plot ------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ks = np.arange(1, MODEL_ITER + 1)
    ax.semilogy(ks, vi_errs,  label="Value Iteration", lw=2)
    ax.semilogy(ks, ipe_errs, label="IPE (uniform π)", lw=2)
    ax.semilogy(ks, vi_theory, "k--", alpha=0.5, label=f"γ^k · ε₀ (γ={GAMMA})")
    ax.set_xlabel("DP-Iteration k")
    ax.set_ylabel("‖V_k − V*‖∞")
    ax.set_title(f"Modellbasiert (γ={GAMMA})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4, which="both")

    ax = axes[1]
    for ep_ax, m, s, label, col in [
        (ep_axis[:len(mc_m)], mc_m, mc_s, f"MC first-visit ({n_seeds} Seeds)", "C0"),
        (ep_axis[:len(td_m)], td_m, td_s, f"TD(0) α=0.1 ({n_seeds} Seeds)",  "C1"),
        (ep_axis[:len(ql_m)], ql_m, ql_s, f"Q-Learning ε=0.3 ({n_seeds} Seeds)", "C2"),
    ]:
        ax.semilogy(ep_ax, m, label=label, color=col, lw=1.6)
        ax.fill_between(ep_ax, np.maximum(m - s, 1e-10), m + s, alpha=0.2, color=col)
    ax.set_xlabel("Episode")
    ax.set_ylabel("‖V̂ − V^π‖∞  (MC/TD)  bzw.  ‖max_a Q − V*‖∞  (QL)")
    ax.set_title(f"Samplebasiert (γ={GAMMA}, snap alle {SNAPSHOT_EP} Ep.)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4, which="both")

    fig.suptitle(
        f"Aufgabe a) — Konvergenzraten: 4×4 GridWorld, γ={GAMMA}, n_seeds={n_seeds}",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "task_a_convergence_rates.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot:  {out}")

    # ------ Konsolen-Zusammenfassung ------
    vi_conv  = next((i+1 for i,e in enumerate(vi_errs)  if e < 1e-6), MODEL_ITER)
    ipe_conv = next((i+1 for i,e in enumerate(ipe_errs) if e < 1e-6), MODEL_ITER)
    print(f"\n=== Aufgabe a) — Konvergenzraten ===")
    print(f"VI  konvergiert in {vi_conv:2d} Iters auf ε<1e-6  (Theorie: -log(1e-6)/log(1/γ) ≈ {-np.log(1e-6)/np.log(1/GAMMA):.0f})")
    print(f"IPE konvergiert in {ipe_conv:2d} Iters auf ε<1e-6")
    print(f"MC  finaler Fehler ({n_ep} Ep.): {mc_m[-1]:.4f} ± {mc_s[-1]:.4f}")
    print(f"TD  finaler Fehler ({n_ep} Ep.): {td_m[-1]:.4f} ± {td_s[-1]:.4f}")
    print(f"QL  finaler Fehler ({n_ep} Ep.): {ql_m[-1]:.4f} ± {ql_s[-1]:.4f}")

    out_json = RESULTS_DIR / "task_a.json"
    with open(out_json, "w") as f:
        json.dump({
            "hyperparameters": {
                "gamma": GAMMA, "n_episodes": n_ep, "n_seeds": n_seeds,
                "seeds": seeds, "snapshot_every": SNAPSHOT_EP,
                "alpha_td_ql": 0.1, "epsilon_ql": 0.3, "max_steps": MAX_STEPS,
            },
            "results": {
                "vi_iters_to_1e6": vi_conv,
                "ipe_iters_to_1e6": ipe_conv,
                "mc_final_mean": float(mc_m[-1]),  "mc_final_std": float(mc_s[-1]),
                "td_final_mean": float(td_m[-1]),  "td_final_std": float(td_s[-1]),
                "ql_final_mean": float(ql_m[-1]),  "ql_final_std": float(ql_s[-1]),
            },
            "n_seeds": n_seeds, "seeds": seeds,
        }, f, indent=2)
    print(f"JSON:  {out_json}")


if __name__ == "__main__":
    main()
