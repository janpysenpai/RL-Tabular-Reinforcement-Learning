"""Submission d)  Stepsize- und Exploration-Schedules für Q-Learning.

Fragestellung: Wie beeinflussen alpha- und epsilon-Schedules die Lernkurve?
Welche Daumenregeln lassen sich aus den Experimenten ableiten?

Experiment 1 — Alpha-Schedules (epsilon=const 0.1 fixiert):
    - constant(0.5)          Zu groß → Q pendelt, konvergiert nicht richtig.
    - constant(0.1)          Klassischer Kompromiss.
    - one_over_n()           Robbins-Monro; langsam aber sicher.
    - polynomial(0.75)       Zwischen 1/n und konstant.

Experiment 2 — Epsilon-Schedules (alpha=const 0.1 fixiert):
    - constant(0.3)          Starke Exploration, behält commit-Problem.
    - constant(0.05)         Geringe Exploration, nutzt früh aus.
    - linear_decay(0.5→0.01, T=1500)  Wechsel Exploration→Exploitation.
    - uniform (behaviour_policy="uniform", epsilon ignoriert)

Umgebung: 4×4 Submission GridWorld (noise=False), γ=0.9.
Metrik: V_hat(Start) = max_a Q(Start, a) aller N_SEEDS Seeds (Mittelwert ± Std).

Hyperparameter: N_SEEDS=5, N_EPISODES=3000, SNAPSHOT_EP=50.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ...envs.gridworld_4x4_submission import GridWorld4x4Submission, START_RC
from ...algos.value_iteration import value_iteration_V
from ...algos.q_learning import q_learning
from ...algos.schedules import constant, one_over_n, polynomial, linear_decay

FIGURES_DIR = Path(__file__).parent / ".." / ".." / ".." / "figures" / "submission"
RESULTS_DIR = Path(__file__).parent / ".." / ".." / ".." / "results" / "submission"

GAMMA       = 0.9
N_SEEDS     = 5
SEEDS       = list(range(N_SEEDS))
N_EPISODES  = 3000
SNAP_EP     = 50
MAX_STEPS   = 200

QUICK_MODE = False  # True: N_EPISODES→800, N_SEEDS→3


def _rc(r: int, c: int) -> int:
    return r * 4 + c


def _run_schedules(
    schedules: List[Tuple[str, Callable, Callable, str]],
    seeds: List[int],
    n_ep: int,
    v_star_start: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Trainiert Q-Learning für jeden Schedule + Seed, gibt mean/std zurück.

    schedules: Liste von (label, alpha_schedule, epsilon_schedule, behaviour_policy).
    Returns: {label: (mean_curve, std_curve)}, Kurvenachse = Episode-Snapshots.
    """
    start_s = _rc(*START_RC)
    results = {}

    for label, alpha_fn, eps_fn, bpolicy in schedules:
        runs = []
        for seed in seeds:
            env = GridWorld4x4Submission(noise=False, gamma=GAMMA)
            snap_vals: List[float] = []
            def _eval(Q, _buf=snap_vals, _s=start_s):
                _buf.append(float(Q[_s, :].max()))
                return _buf[-1]
            q_learning(env, alpha=alpha_fn, epsilon=eps_fn,
                       behaviour_policy=bpolicy,
                       n_episodes=n_ep, max_steps=MAX_STEPS, seed=seed,
                       eval_every=SNAP_EP, eval_fn=_eval)
            runs.append(snap_vals)
        L = min(len(r) for r in runs)
        mat = np.array([r[:L] for r in runs])
        results[label] = (mat.mean(axis=0), mat.std(axis=0))

    return results


def main() -> None:
    n_ep    = 800  if QUICK_MODE else N_EPISODES
    n_seeds = 3    if QUICK_MODE else N_SEEDS
    seeds   = SEEDS[:n_seeds]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Optimaler Wert via VI
    env_ref = GridWorld4x4Submission(noise=False, gamma=GAMMA)
    V_star, _ = value_iteration_V(env_ref, tol=1e-12)
    start_s   = _rc(*START_RC)
    v_star_start = float(V_star[start_s])
    print(f"V*(Start) via VI = {v_star_start:.4f}")

    ep_axis = np.arange(SNAP_EP, n_ep + 1, SNAP_EP)

    # ------ Experiment 1: Alpha-Schedules ------
    alpha_schedules = [
        ("α=const(0.5)",    constant(0.5),        constant(0.1), "eps_greedy"),
        ("α=const(0.1)",    constant(0.1),        constant(0.1), "eps_greedy"),
        ("α=1/n",           one_over_n(),         constant(0.1), "eps_greedy"),
        ("α=1/n^0.75",      polynomial(0.75),     constant(0.1), "eps_greedy"),
    ]
    print("Experiment 1: Alpha-Schedules ...")
    alpha_results = _run_schedules(alpha_schedules, seeds, n_ep, v_star_start)

    # ------ Experiment 2: Epsilon-Schedules ------
    eps_schedules = [
        ("ε=const(0.3)",                constant(0.1), constant(0.3),                        "eps_greedy"),
        ("ε=const(0.05)",               constant(0.1), constant(0.05),                       "eps_greedy"),
        (f"ε=decay(0.5→0.01,T={int(n_ep/2)})", constant(0.1), linear_decay(0.5, 0.01, n_ep // 2), "eps_greedy"),
        ("uniform behaviour",           constant(0.1), constant(0.1),                        "uniform"),
    ]
    print("Experiment 2: Epsilon-Schedules ...")
    eps_results = _run_schedules(eps_schedules, seeds, n_ep, v_star_start)

    # ------ Plot ------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = [f"C{i}" for i in range(4)]

    for ax, results, title_suffix in [
        (axes[0], alpha_results, f"Alpha-Schedules (ε=const(0.1), {n_seeds} Seeds)"),
        (axes[1], eps_results,   f"Epsilon-Schedules (α=const(0.1), {n_seeds} Seeds)"),
    ]:
        for (label, (m, s)), col in zip(results.items(), colors):
            x = ep_axis[:len(m)]
            ax.plot(x, m, label=label, color=col, lw=1.6)
            ax.fill_between(x, m - s, m + s, alpha=0.15, color=col)
        ax.axhline(v_star_start, linestyle="--", color="black", alpha=0.6,
                   label=f"V*(Start)={v_star_start:.3f}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("max_a Q(Start, a)")
        ax.set_title(title_suffix)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.4)

    fig.suptitle(
        f"Aufgabe d) — Q-Learning Schedules: 4×4 Submission Grid, γ={GAMMA}, {n_ep} Ep.",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "task_d_qlearning_hyperparams.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot:  {out}")

    # ------ Konsolen-Zusammenfassung + Daumenregeln ------
    print(f"\n=== Aufgabe d) — Q-Learning Schedules ===")
    print(f"V*(Start) Referenz (VI): {v_star_start:.4f}")
    print("\nAlpha-Schedules (finaler V_hat):")
    for label, (m, _) in alpha_results.items():
        print(f"  {label:25s}: {m[-1]:.4f}")
    print("\nEpsilon-Schedules (finaler V_hat):")
    for label, (m, _) in eps_results.items():
        print(f"  {label:35s}: {m[-1]:.4f}")

    print("\nDaumenregeln:")
    print("  Alpha: Konstantes α konvergiert schneller aber rauscht mehr;")
    print("         α=1/n garantiert Konvergenz aber ist langsamer.")
    print("         Empirische Empfehlung: α=0.1 (konstant) für stationäre Umgebungen,")
    print("         α=1/n^0.75 als guter Kompromiss.")
    print("  Epsilon: Zu kleine ε → frühe Ausbeutung, kann in suboptimales Q stecken bleiben.")
    print("            Decaying ε kombiniert Exploration und Exploitation zeitlich.")
    print("            Uniform Behaviour: maximale Exploration, langsame Policy-Qualität.")

    out_json = RESULTS_DIR / "task_d.json"
    with open(out_json, "w") as f:
        json.dump({
            "hyperparameters": {
                "gamma": GAMMA, "n_episodes": n_ep, "n_seeds": n_seeds,
                "snapshot_every": SNAP_EP, "max_steps": MAX_STEPS,
            },
            "v_star_start": v_star_start,
            "alpha_results": {
                label: {"final_mean": float(m[-1]), "final_std": float(s[-1])}
                for label, (m, s) in alpha_results.items()
            },
            "eps_results": {
                label: {"final_mean": float(m[-1]), "final_std": float(s[-1])}
                for label, (m, s) in eps_results.items()
            },
        }, f, indent=2)
    print(f"JSON:  {out_json}")


if __name__ == "__main__":
    main()
