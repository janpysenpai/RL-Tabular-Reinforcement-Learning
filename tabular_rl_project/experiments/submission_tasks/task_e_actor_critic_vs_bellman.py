"""Submission e)  General Actor-Critic vs. direkte Bellman-Kontrolle.

Verglichen werden fünf Verfahren auf der 4×4 Submission GridWorld (noise=False):

    1. Value Iteration (VI)          — modellbasiert, direkte Bellman-Optimalität
    2. Q-Learning                    — sample-basiert, Bellman-Optimalitätsoperator
    3. Actor-Critic + IPE-Critic     — modellbasiert Critic (exakt, IPE bis Tol)
    4. Actor-Critic + TD(0)-Critic   — sample-basierter Critic
    5. Actor-Critic + MC-Critic      — sample-basierter Critic

Metriken:
    - V_hat(Start) = max_a Q(Start, a) nach jedem Outer-/Trainings-Schritt
      (für VI: nach jeder DP-Iteration; für QL: alle SNAP_EP Episoden;
       für AC-Varianten: nach jedem Outer-Schritt)
    - Finaler Policy-Fehler: Anteil Zustände mit suboptimaler Aktion (vs. V*-Greedy)

Linkes Panel: V_hat(Start) über "Aufwand" (VI-Iter / QL-Episoden / AC-Outer-Iter).
Rechtes Panel: Balkendiagramm finaler Policy-Korrektheit (Anteil optimal).

Hyperparameter:
    γ=0.9, N_SEEDS=5 (QL und AC-TD0/MC), max 200 Outer-Iters (AC),
    TD0/MC Critic: 200 Episoden pro Outer-Schritt, α=0.1, ε=0.1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ...envs.gridworld_4x4_submission import GridWorld4x4Submission, START_RC
from ...algos.value_iteration import value_iteration_V
from ...algos.q_learning import q_learning
from ...algos.actor_critic import actor_critic, make_ipe_critic, make_td0_critic, make_mc_critic
from ...algos.schedules import constant

FIGURES_DIR = Path(__file__).parent / ".." / ".." / ".." / "figures" / "submission"
RESULTS_DIR = Path(__file__).parent / ".." / ".." / ".." / "results" / "submission"

GAMMA          = 0.9
N_SEEDS        = 5
SEEDS          = list(range(N_SEEDS))
QL_EPISODES    = 3000
QL_SNAP        = 50
MAX_STEPS_QL   = 200
AC_OUTER_ITER  = 30
CRITIC_EP      = 300  # Episoden pro Outer-Schritt für TD0/MC Critic
ALPHA          = 0.1
EPSILON        = 0.1


def _rc(r: int, c: int) -> int:
    return r * 4 + c


def _policy_correct_rate(Q: np.ndarray, Q_star: np.ndarray, env) -> float:
    """Anteil Nicht-Terminal-Zustände mit Q*-optimaler Aktion."""
    correct = 0
    total = 0
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


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------ Referenz: V* via VI ------
    env_ref = GridWorld4x4Submission(noise=False, gamma=GAMMA)
    V_star, _ = value_iteration_V(env_ref, tol=1e-12)
    start_s = _rc(*START_RC)
    Q_star = env_ref.expected_rewards + GAMMA * np.einsum(
        "sap,p->sa", env_ref.transition_probabilities, V_star
    )
    v_star_start = float(V_star[start_s])

    # ------ 1. Value Iteration (Track V_hat(Start)) ------
    print("Variante 1: Value Iteration ...")
    P = env_ref.transition_probabilities
    R = env_ref.expected_rewards
    V_vi = np.zeros(env_ref.n_states)
    vi_curve: List[float] = []
    for _ in range(60):
        V_vi = (R + GAMMA * np.einsum("sap,p->sa", P, V_vi)).max(axis=1)
        vi_curve.append(float(V_vi[start_s]))
    Q_vi = R + GAMMA * np.einsum("sap,p->sa", P, V_vi)
    vi_correct = _policy_correct_rate(Q_vi, Q_star, env_ref)

    # ------ 2. Q-Learning (mehrere Seeds) ------
    print("Variante 2: Q-Learning ...")
    ql_runs: List[List[float]] = []
    ql_q_final = None
    for seed in SEEDS:
        env = GridWorld4x4Submission(noise=False, gamma=GAMMA)
        snaps: List[float] = []
        def _snap(Q, _buf=snaps, _s=start_s):
            _buf.append(float(Q[_s, :].max()))
            return _buf[-1]
        Q_out, _ = q_learning(env, alpha=constant(ALPHA), epsilon=constant(EPSILON),
                               n_episodes=QL_EPISODES, max_steps=MAX_STEPS_QL,
                               seed=seed, eval_every=QL_SNAP, eval_fn=_snap)
        ql_runs.append(snaps)
        if seed == SEEDS[0]:
            ql_q_final = Q_out
    L_ql = min(len(r) for r in ql_runs)
    ql_mat = np.array([r[:L_ql] for r in ql_runs])
    ql_m, ql_s = ql_mat.mean(axis=0), ql_mat.std(axis=0)
    ql_correct = _policy_correct_rate(ql_q_final, Q_star, env_ref)
    ql_ep_ax = np.arange(QL_SNAP, QL_EPISODES + 1, QL_SNAP)[:L_ql]

    # ------ 3. Actor-Critic + IPE (exakter Critic) ------
    print("Variante 3: Actor-Critic + IPE ...")
    ipe_curve: List[float] = []
    env_ipe = GridWorld4x4Submission(noise=False, gamma=GAMMA)
    V_ac_ipe, pi_ipe, info_ipe = actor_critic(
        env_ipe, critic=make_ipe_critic("exact"), gamma=GAMMA,
        n_outer_iter=AC_OUTER_ITER,
    )
    # Rekonstruiere Kurve: track V_hat nach jedem Outer-Schritt
    for ci in info_ipe["critic_infos"]:
        pass  # V nach jedem Schritt nicht direkt verfügbar
    # Direkter Re-Run mit Tracking
    ipe_curve2: List[float] = []
    V_ipe_t = np.zeros(env_ipe.n_states)
    from ...algos.iterative_policy_evaluation import ipe_V
    from ...algos.exploration import uniform_random_policy
    pi = uniform_random_policy(env_ipe)
    from ...algos.value_iteration import greedy_policy_from_v, _q_from_v
    for _ in range(AC_OUTER_ITER):
        V_ipe_t, _ = ipe_V(env_ipe, pi, gamma=GAMMA, tol=1e-9)
        ipe_curve2.append(float(V_ipe_t[start_s]))
        pi_new = greedy_policy_from_v(env_ipe, V_ipe_t, gamma=GAMMA)
        if np.array_equal(pi, pi_new):
            break
        pi = pi_new
    Q_ipe = _q_from_v(V_ipe_t, P, R, GAMMA)
    ipe_correct = _policy_correct_rate(Q_ipe, Q_star, env_ipe)

    # ------ 4. Actor-Critic + TD(0) (mehrere Seeds) ------
    print("Variante 4: Actor-Critic + TD(0) ...")
    ac_td_runs: List[List[float]] = []
    ac_td_q_final = None
    for seed in SEEDS:
        env_td = GridWorld4x4Submission(noise=False, gamma=GAMMA)
        V_td_ac, pi_td, _ = actor_critic(
            env_td, critic=make_td0_critic(n_episodes=CRITIC_EP, alpha=ALPHA, seed=seed),
            gamma=GAMMA, n_outer_iter=AC_OUTER_ITER,
        )
        # V nach jedem outer: direkte Kurve nicht möglich über actor_critic; sample über n outer
        # Wir laufen step-by-step
        from ...algos.td_policy_evaluation import td0_policy_evaluation_V
        ac_td_curve: List[float] = []
        pi_t = uniform_random_policy(env_td)
        V_t = np.zeros(env_td.n_states)
        rng_s = np.random.default_rng(seed)
        for _ in range(AC_OUTER_ITER):
            sub_seed = int(rng_s.integers(0, 2**31))
            V_t, _ = td0_policy_evaluation_V(
                env_td, pi_t, alpha=constant(ALPHA), gamma=GAMMA,
                n_episodes=CRITIC_EP, max_steps=MAX_STEPS_QL, seed=sub_seed,
            )
            ac_td_curve.append(float(V_t[start_s]))
            pi_new = greedy_policy_from_v(env_td, V_t, gamma=GAMMA)
            if np.array_equal(pi_t, pi_new):
                break
            pi_t = pi_new
        ac_td_runs.append(ac_td_curve)
        if seed == SEEDS[0]:
            ac_td_q_final = _q_from_v(V_t, P, R, GAMMA)

    L_ac_td = min(len(r) for r in ac_td_runs)
    ac_td_mat = np.array([r[:L_ac_td] for r in ac_td_runs])
    ac_td_m, ac_td_s = ac_td_mat.mean(axis=0), ac_td_mat.std(axis=0)
    ac_td_correct = _policy_correct_rate(ac_td_q_final, Q_star, env_ref)

    # ------ 5. Actor-Critic + MC (mehrere Seeds) ------
    print("Variante 5: Actor-Critic + MC ...")
    ac_mc_runs: List[List[float]] = []
    ac_mc_q_final = None
    for seed in SEEDS:
        env_mc = GridWorld4x4Submission(noise=False, gamma=GAMMA)
        from ...algos.monte_carlo import mc_policy_evaluation_V
        ac_mc_curve: List[float] = []
        pi_t = uniform_random_policy(env_mc)
        V_t = np.zeros(env_mc.n_states)
        rng_s = np.random.default_rng(seed)
        for _ in range(AC_OUTER_ITER):
            sub_seed = int(rng_s.integers(0, 2**31))
            V_t, _ = mc_policy_evaluation_V(
                env_mc, pi_t, gamma=GAMMA,
                n_episodes=CRITIC_EP, max_steps=MAX_STEPS_QL, seed=sub_seed,
            )
            ac_mc_curve.append(float(V_t[start_s]))
            pi_new = greedy_policy_from_v(env_mc, V_t, gamma=GAMMA)
            if np.array_equal(pi_t, pi_new):
                break
            pi_t = pi_new
        ac_mc_runs.append(ac_mc_curve)
        if seed == SEEDS[0]:
            ac_mc_q_final = _q_from_v(V_t, P, R, GAMMA)

    L_ac_mc = min(len(r) for r in ac_mc_runs)
    ac_mc_mat = np.array([r[:L_ac_mc] for r in ac_mc_runs])
    ac_mc_m, ac_mc_s = ac_mc_mat.mean(axis=0), ac_mc_mat.std(axis=0)
    ac_mc_correct = _policy_correct_rate(ac_mc_q_final, Q_star, env_ref)

    # ------ Plot ------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Konvergenz V_hat(Start)
    ax = axes[0]
    ax.plot(np.arange(1, len(vi_curve) + 1), vi_curve,
            label="Value Iteration (DP-Iters)", color="C0", lw=2)
    ax.plot(ql_ep_ax, ql_m, label=f"Q-Learning ({N_SEEDS} Seeds, α={ALPHA})", color="C1", lw=1.6)
    ax.fill_between(ql_ep_ax, ql_m - ql_s, ql_m + ql_s, alpha=0.15, color="C1")
    ax.plot(np.arange(1, len(ipe_curve2) + 1) * CRITIC_EP, ipe_curve2,
            label="AC + IPE-Critic (exakt)", color="C2", lw=2, marker="o", ms=4)
    x_ac_td = np.arange(1, L_ac_td + 1) * CRITIC_EP
    ax.plot(x_ac_td, ac_td_m, label=f"AC + TD(0)-Critic ({N_SEEDS} Seeds)", color="C3", lw=1.6)
    ax.fill_between(x_ac_td, ac_td_m - ac_td_s, ac_td_m + ac_td_s, alpha=0.15, color="C3")
    x_ac_mc = np.arange(1, L_ac_mc + 1) * CRITIC_EP
    ax.plot(x_ac_mc, ac_mc_m, label=f"AC + MC-Critic ({N_SEEDS} Seeds)", color="C4", lw=1.6)
    ax.fill_between(x_ac_mc, ac_mc_m - ac_mc_s, ac_mc_m + ac_mc_s, alpha=0.15, color="C4")
    ax.axhline(v_star_start, linestyle="--", color="black", alpha=0.6,
               label=f"V*(Start)={v_star_start:.3f}")
    ax.set_xlabel("Effektive Episoden (AC: Outer × Critic-Ep, VI: DP-Iters)")
    ax.set_ylabel("V_hat(Start)")
    ax.set_title(f"Konvergenz V(Start): Actor-Critic vs. Bellman-Kontrolle (γ={GAMMA})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.4)

    # Panel 2: Finaler Policy-Fehler
    ax = axes[1]
    methods = ["VI", "Q-Learning", "AC+IPE", "AC+TD(0)", "AC+MC"]
    corrects = [vi_correct, ql_correct, ipe_correct, ac_td_correct, ac_mc_correct]
    cols = ["C0", "C1", "C2", "C3", "C4"]
    bars = ax.bar(methods, corrects, color=cols, alpha=0.8)
    ax.bar_label(bars, fmt="%.2f", fontsize=9)
    ax.axhline(1.0, linestyle="--", color="black", alpha=0.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Anteil Zustände mit optimaler Aktion")
    ax.set_title(f"Finale Policy-Korrektheit (Referenz: V*-Greedy, Seed=0)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "Aufgabe e) — General Actor-Critic vs. Bellman-Kontrolle: 4×4 Submission Grid",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "task_e_actor_critic_vs_bellman.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot:  {out}")

    # ------ Konsolen-Zusammenfassung ------
    print(f"\n=== Aufgabe e) — Actor-Critic vs. Bellman-Kontrolle ===")
    print(f"V*(Start) Referenz (VI): {v_star_start:.4f}")
    for m, c, v in zip(methods, corrects,
                       [vi_curve[-1], ql_m[-1], ipe_curve2[-1], ac_td_m[-1], ac_mc_m[-1]]):
        print(f"  {m:12s}: V_hat(Start)={v:.4f}, Policy-Korrekt={c:.2%}")

    out_json = RESULTS_DIR / "task_e.json"
    with open(out_json, "w") as f:
        json.dump({
            "hyperparameters": {
                "gamma": GAMMA, "n_seeds": N_SEEDS,
                "ql_episodes": QL_EPISODES, "ql_snap": QL_SNAP,
                "ac_outer_iter": AC_OUTER_ITER, "critic_ep": CRITIC_EP,
                "alpha": ALPHA, "epsilon": EPSILON,
            },
            "v_star_start": v_star_start,
            "results": {
                m: {"final_v_hat": float(v), "policy_correct": float(c)}
                for m, v, c in zip(
                    methods,
                    [vi_curve[-1], float(ql_m[-1]), ipe_curve2[-1], float(ac_td_m[-1]), float(ac_mc_m[-1])],
                    corrects,
                )
            },
        }, f, indent=2)
    print(f"JSON:  {out_json}")


if __name__ == "__main__":
    main()
