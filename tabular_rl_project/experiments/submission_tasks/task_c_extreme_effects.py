"""Submission c)  Extreme MDPs — drei algorithmische Effekte.

Effekt 1 — Backpropagation (Sparse-Chain-MDP):
    Value Iteration auf einer 10-Zustands-Kette mit Sparse Reward am Ende.
    V[s=0] propagiert sich nur langsam über Iterationen rückwärts.
    Plot: V_k(s) für alle s über Iterationen k — Wärmebild.

Effekt 2 — Robust RL (Cliff Walk, Sutton & Barto Beispiel 6.6):
    SARSA (on-policy) vs. Q-Learning (off-policy).
    Q-Learning findet die kürzere aber riskante Kliff-Kanten-Route.
    SARSA lernt die sichere aber längere Umgehungsstrategie.
    Plot: Episode Return (geglättet) über Trainingsepisoden, N Seeds.

Effekt 3 — Overestimation Bias (BiasedMDP):
    In einem 2-Zustands-MDP mit 10 Aktionen und normalverteiltem Reward
    (E[R]=−0.1, σ=1.0) überschätzt Q-Learning E[max_a Q(0,a)] > Q*(0,a).
    Double Q-Learning reduziert diesen Jensen-Bias.
    Plot: Q(0,:).max() über Episoden, Vergleich Q-Learning vs. Double-Q.

Umgebungen:
    Backprop:  make_sparse_chain(n_states=10, gamma=0.95)
    Robust:    CliffWalk(rows=4, cols=12, gamma=1.0)
    Bias:      make_bias_mdp_for_double_q(n_acts=10, reward_mean=-0.1, reward_std=1.0)

Hyperparameter: N_SEEDS=5, N_EPISODES=500 (Cliff/Bias), alpha=0.1, epsilon=0.1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ...envs.extreme_mdps import make_sparse_chain, make_bias_mdp_for_double_q
from ...envs.cliff_walk import CliffWalk
from ...algos.value_iteration import value_iteration_V
from ...algos.q_learning import q_learning
from ...algos.sarsa import sarsa
from ...algos.double_q_learning import double_q_learning
from ...algos.schedules import constant

FIGURES_DIR = Path(__file__).parent / ".." / ".." / ".." / "figures" / "submission"
RESULTS_DIR = Path(__file__).parent / ".." / ".." / ".." / "results" / "submission"

N_SEEDS    = 5
SEEDS      = list(range(N_SEEDS))
N_EP_CLIFF = 500
N_EP_BIAS  = 1000
MAX_STEPS  = 300
ALPHA      = 0.1
EPSILON    = 0.1
SMOOTH_W   = 20  # Glättungsfenster für Episode Returns


def _smooth(x: List[float], w: int) -> np.ndarray:
    arr = np.array(x, dtype=float)
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="valid")


def _run_backprop() -> np.ndarray:
    """Gibt V_k(s) als Array shape (n_iter, n_states) zurück."""
    env = make_sparse_chain(n_states=10, gamma=0.95)
    P, R = env.transition_probabilities, env.expected_rewards
    gamma = env.gamma
    V = np.zeros(env.n_states)
    n_iter = 30
    history = []
    for _ in range(n_iter):
        Q = R + gamma * np.einsum("sap,p->sa", P, V)
        V = Q.max(axis=1)
        history.append(V.copy())
    return np.array(history)  # (n_iter, S)


def _run_cliff(n_ep: int, seeds: List[int]):
    """SARSA vs Q-Learning auf Cliff Walk.

    Returns (sarsa_runs, ql_runs) — jede Liste enthält episode_returns pro Seed.
    """
    sarsa_runs, ql_runs = [], []
    for seed in seeds:
        env = CliffWalk(rows=4, cols=12, gamma=1.0)
        _, info = sarsa(env, alpha=constant(ALPHA), epsilon=constant(EPSILON),
                        n_episodes=n_ep, max_steps=MAX_STEPS, seed=seed)
        sarsa_runs.append(info["episode_returns"])

        env2 = CliffWalk(rows=4, cols=12, gamma=1.0)
        _, info2 = q_learning(env2, alpha=constant(ALPHA), epsilon=constant(EPSILON),
                               n_episodes=n_ep, max_steps=MAX_STEPS, seed=seed)
        ql_runs.append(info2["episode_returns"])
    return sarsa_runs, ql_runs


def _runs_mean_std(runs, smooth_w=SMOOTH_W):
    """Mittelwert und Std über Seeds nach optionaler Glättung."""
    smoothed = [_smooth(r, smooth_w) for r in runs]
    L = min(len(s) for s in smoothed)
    mat = np.array([s[:L] for s in smoothed])
    return mat.mean(axis=0), mat.std(axis=0)


def _run_bias(n_ep: int, seeds: List[int]):
    """Q-Learning vs Double-Q auf BiasedMDP.

    Verfolgt max_a Q(0,a) über Episoden via eval_fn.
    """
    TRUE_Q = -0.1
    ql_max_runs, dql_max_runs = [], []

    for seed in seeds:
        env_ql = make_bias_mdp_for_double_q(
            n_acts=10, reward_mean=TRUE_Q, reward_std=1.0, gamma=0.9, env_seed=seed * 7
        )
        ql_errs: List[float] = []
        def _ql_fn(Q, _buf=ql_errs):
            _buf.append(float(Q[0, :].max()))
            return _buf[-1]
        q_learning(env_ql, alpha=constant(ALPHA), epsilon=constant(EPSILON),
                   n_episodes=n_ep, max_steps=10, seed=seed,
                   eval_every=10, eval_fn=_ql_fn)
        ql_max_runs.append(ql_errs)

        env_dq = make_bias_mdp_for_double_q(
            n_acts=10, reward_mean=TRUE_Q, reward_std=1.0, gamma=0.9, env_seed=seed * 7
        )
        dql_errs: List[float] = []
        def _dql_fn(Q_avg, _buf=dql_errs):
            _buf.append(float(Q_avg[0, :].max()))
            return _buf[-1]
        double_q_learning(env_dq, alpha=constant(ALPHA), epsilon=constant(EPSILON),
                          n_episodes=n_ep, max_steps=10, seed=seed,
                          eval_every=10, eval_fn=_dql_fn)
        dql_max_runs.append(dql_errs)

    return ql_max_runs, dql_max_runs, TRUE_Q


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Effekt 1: Backpropagation ...")
    backprop = _run_backprop()

    print("Effekt 2: Robust RL (Cliff Walk) ...")
    sarsa_runs, ql_runs = _run_cliff(N_EP_CLIFF, SEEDS)
    sarsa_m, sarsa_s = _runs_mean_std(sarsa_runs)
    ql_m, ql_s       = _runs_mean_std(ql_runs)

    print("Effekt 3: Overestimation Bias ...")
    ql_max_runs, dql_max_runs, true_q = _run_bias(N_EP_BIAS, SEEDS)

    L_bias = min(min(len(r) for r in ql_max_runs), min(len(r) for r in dql_max_runs))
    ql_bias_m  = np.array([r[:L_bias] for r in ql_max_runs]).mean(axis=0)
    ql_bias_s  = np.array([r[:L_bias] for r in ql_max_runs]).std(axis=0)
    dql_bias_m = np.array([r[:L_bias] for r in dql_max_runs]).mean(axis=0)
    dql_bias_s = np.array([r[:L_bias] for r in dql_max_runs]).std(axis=0)
    bias_ep_ax = np.arange(1, L_bias + 1) * 10

    # ------ Plot ------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel 1: Backpropagation
    ax = axes[0]
    im = ax.imshow(backprop.T, origin="lower", aspect="auto",
                   extent=[1, backprop.shape[0], -0.5, backprop.shape[1] - 0.5],
                   cmap="YlOrRd")
    fig.colorbar(im, ax=ax, label="V_k(s)")
    ax.set_xlabel("VI-Iteration k")
    ax.set_ylabel("Zustand s")
    ax.set_title(f"Backpropagation\n(Sparse Chain, n=10, γ=0.95)")
    ax.set_yticks(range(10))
    ax.set_yticklabels([str(s) for s in range(10)])

    # Panel 2: Robust RL
    ax = axes[1]
    ep_axis_cliff = np.arange(1, len(sarsa_m) + 1) + (SMOOTH_W - 1)
    ax.plot(ep_axis_cliff, sarsa_m, label=f"SARSA α={ALPHA}, ε={EPSILON}", color="C0", lw=1.6)
    ax.fill_between(ep_axis_cliff,
                    np.maximum(sarsa_m - sarsa_s, -200), sarsa_m + sarsa_s,
                    alpha=0.2, color="C0")
    ax.plot(ep_axis_cliff, ql_m, label=f"Q-Learning α={ALPHA}, ε={EPSILON}", color="C1", lw=1.6)
    ax.fill_between(ep_axis_cliff,
                    np.maximum(ql_m - ql_s, -200), ql_m + ql_s,
                    alpha=0.2, color="C1")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return (geglättet, w=20)")
    ax.set_title(f"Robust RL — Cliff Walk\n({N_SEEDS} Seeds, α={ALPHA}, ε={EPSILON})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.4)
    ax.set_ylim(-150, 0)

    # Panel 3: Overestimation Bias
    ax = axes[2]
    ax.plot(bias_ep_ax, ql_bias_m, label=f"Q-Learning ({N_SEEDS} Seeds)", color="C1", lw=1.6)
    ax.fill_between(bias_ep_ax,
                    ql_bias_m - ql_bias_s, ql_bias_m + ql_bias_s,
                    alpha=0.2, color="C1")
    ax.plot(bias_ep_ax, dql_bias_m, label=f"Double Q-Learning ({N_SEEDS} Seeds)", color="C2", lw=1.6)
    ax.fill_between(bias_ep_ax,
                    dql_bias_m - dql_bias_s, dql_bias_m + dql_bias_s,
                    alpha=0.2, color="C2")
    ax.axhline(true_q, linestyle="--", color="black", alpha=0.7,
               label=f"Q*(0,a)={true_q} (True Value)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("max_a Q_hat(0,a)")
    ax.set_title(f"Overestimation Bias\n(BiasedMDP, n_acts=10, E[R]={true_q}, σ=1.0)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.4)

    fig.suptitle(
        "Aufgabe c) — Drei Extreme-MDP-Effekte: Backprop, Robust RL, Overestimation Bias",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "task_c_extreme_effects.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot:  {out}")

    # ------ Konsolen-Zusammenfassung ------
    print("\n=== Aufgabe c) — Extreme MDPs ===")
    print(f"Backprop: V[s=0] nach 1 Iter={backprop[0, 0]:.4f}, nach 10={backprop[9, 0]:.4f}, nach 30={backprop[-1, 0]:.4f}")
    print(f"Cliff Walk: SARSA finaler Return={sarsa_m[-1]:.2f}, Q-Learning={ql_m[-1]:.2f}")
    print(f"Bias (finaler max Q): Q-Learning={ql_bias_m[-1]:.4f}, Double-Q={dql_bias_m[-1]:.4f}, True={true_q:.4f}")

    out_json = RESULTS_DIR / "task_c.json"
    with open(out_json, "w") as f:
        json.dump({
            "hyperparameters": {
                "n_seeds": N_SEEDS, "alpha": ALPHA, "epsilon": EPSILON,
                "n_ep_cliff": N_EP_CLIFF, "n_ep_bias": N_EP_BIAS,
            },
            "results": {
                "backprop_v_s0_iter30": float(backprop[-1, 0]),
                "cliff_sarsa_final_return": float(sarsa_m[-1]),
                "cliff_ql_final_return": float(ql_m[-1]),
                "bias_ql_final_max_q": float(ql_bias_m[-1]),
                "bias_dql_final_max_q": float(dql_bias_m[-1]),
                "bias_true_q": float(true_q),
            },
        }, f, indent=2)
    print(f"JSON:  {out_json}")


if __name__ == "__main__":
    main()
