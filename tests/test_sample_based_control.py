"""Tests für modellfreie Control-Algorithmen (Schritt 3b).

Zeitbudget < 30 s (diese Datei allein):
    Q-Learning / SARSA: 5000 Ep., max_steps=300, 3 Seeds auf 4x4 GridWorld
    Double Q Bias:      200 Ep., 50 Seeds, epsilon=1.0 (uniform), 10 Aktionen
    Actor-Critic:       n_outer=30, ipe_critic(steps=5), 2 Seeds

Korrektheitskriterien:
    - Q-Learning: argmax Q ≥14/16 Zustände stimmen mit pi*_VI überein
    - SARSA eps→0: wie oben
    - Double Q: mittlere Overestimation-Bias < Q-Learning-Bias  (50 Seeds)
    - Actor-Critic: konvergiert, ≥14/16 Übereinstimmung mit pi*_VI

Hinweis zu Exploration:
    Mit Q-Initialisierung 0 und wall_behavior="stay" wählt argmax bei Gleichstand
    immer Aktion 0 (UP), die an Ecke (0,0) den Agenten festhält. Deshalb wird
    epsilon=0.5 (konstant) für Q-Learning und linear_decay(1.0, 0.05) für SARSA
    verwendet, um ausreichende Anfangsexploration sicherzustellen.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabular_rl_project.envs._validation_mdp import build_validation_mdp
from tabular_rl_project.envs.gridworld import GridWorld
from tabular_rl_project.envs.extreme_mdps import make_bias_mdp_for_double_q
from tabular_rl_project.algos.value_iteration import value_iteration_V
from tabular_rl_project.algos.q_learning import q_learning
from tabular_rl_project.algos.sarsa import sarsa
from tabular_rl_project.algos.double_q_learning import double_q_learning
from tabular_rl_project.algos.actor_critic import actor_critic, make_ipe_critic
from tabular_rl_project.algos.schedules import constant, linear_decay


# ------------------------------------------------------------------
# Hilfsmittel
# ------------------------------------------------------------------

def _build_grid() -> GridWorld:
    return GridWorld(
        rows=4, cols=4,
        start=(0, 0),
        terminal_states=[(3, 3)],
        cell_rewards={(3, 3): 1.0},
        default_reward=0.0,
        gamma=0.9,
        wall_behavior="stay",
    )


def _q_star(env: FiniteMDP) -> np.ndarray:
    """Berechnet Q* via Value Iteration + Bellman-Gleichung."""
    V, _ = value_iteration_V(env, tol=1e-9)
    P = env.transition_probabilities
    R = env.expected_rewards
    return R + env.gamma * np.einsum("sap,p->sa", P, V)


def _policy_agreement(Q: np.ndarray, Q_star: np.ndarray, env: FiniteMDP) -> int:
    """Anzahl Zustände, in denen argmax Q eine Q*-optimale Aktion wählt.

    Prüft ob die gewählte Aktion den optimalen Q*-Wert erreicht (Toleranz 1e-9).
    So werden Tie-Breaking-Unterschiede (z.B. DOWN vs RIGHT bei gleich optimalem
    Q*-Wert) nicht als Fehler gewertet.
    """
    count = 0
    for s in env.states:
        acts = env.allowed_actions[s]
        chosen = acts[int(np.argmax(Q[s, acts]))]
        best_q = Q_star[s, acts].max()
        if Q_star[s, chosen] >= best_q - 1e-9:
            count += 1
    return count


# ------------------------------------------------------------------
# Fixtures (module-scope für schnelle Mehrfach-Parametrisierung)
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def grid_env():
    return _build_grid()


@pytest.fixture(scope="module")
def Q_star_grid(grid_env):
    return _q_star(grid_env)


# ------------------------------------------------------------------
# 1. Q-Learning auf 4x4 GridWorld
# ------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_q_learning_agrees_with_vi(grid_env, Q_star_grid, seed):
    """argmax Q wählt in >=14/16 Zuständen eine Q*-optimale Aktion.

    epsilon=0.5 nötig: Bei Q=0 wählt argmax immer action 0 (UP),
    die an Ecke (0,0) durch wall_behavior='stay' den Agenten festhält.
    Höheres Epsilon sichert ausreichende Exploration aller Zustände.
    """
    Q, _ = q_learning(
        grid_env,
        alpha=constant(0.1),
        epsilon=constant(0.5),
        n_episodes=5000,
        max_steps=300,
        seed=seed,
    )
    agree = _policy_agreement(Q, Q_star_grid, grid_env)
    assert agree >= 14, (
        f"seed={seed}: Q-Learning Übereinstimmung {agree}/16 < 14/16"
    )


# ------------------------------------------------------------------
# 2. SARSA mit abnehmender Epsilon → pi*_VI
# ------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sarsa_eps_decay_agrees_with_vi(grid_env, Q_star_grid, seed):
    """SARSA mit linear_decay epsilon (1.0→0.05) lernt Q*-optimale Policy (>=14/16).

    Verglichen wird ob die gewählte Aktion einen Q*-optimalen Wert hat.
    Tie-Breaking (z.B. DOWN vs RIGHT bei identischem Q*) gilt als korrekt.
    """
    N_EP = 5000
    Q, _ = sarsa(
        grid_env,
        alpha=constant(0.1),
        epsilon=linear_decay(1.0, 0.05, N_EP),
        n_episodes=N_EP,
        max_steps=300,
        seed=seed,
    )
    agree = _policy_agreement(Q, Q_star_grid, grid_env)
    assert agree >= 14, (
        f"seed={seed}: SARSA Übereinstimmung {agree}/16 < 14/16"
    )


# ------------------------------------------------------------------
# 3. Double Q-Learning: geringere Overestimation als Q-Learning
# ------------------------------------------------------------------

def test_double_q_less_overestimation():
    """Double Q-Learning hat kleinere Overestimation als Q-Learning.

    Bias-MDP aus envs.extreme_mdps: 10 Aktionen, Reward N(-0.1, 1.0).
    Wahre Q*(0,a) = -0.1. Epsilon=1.0 (uniform) stellt gleiche
    Besuchsfrequenz aller Aktionen sicher.
    Mit 50 Seeds muss E[max_double] < E[max_single] gelten.
    """
    N_ACTS = 10
    MEAN_R = -0.1
    SIGMA_R = 1.0
    N_EP = 200       # bewusst wenig: Bias vor Konvergenz messen
    N_SEEDS = 50
    ALPHA = 0.1

    max_single = []
    max_double = []

    for seed in range(N_SEEDS):
        env_q = make_bias_mdp_for_double_q(N_ACTS, MEAN_R, SIGMA_R, gamma=0.9, env_seed=seed)
        Q, _ = q_learning(
            env_q,
            alpha=constant(ALPHA),
            epsilon=constant(1.0),   # uniform exploration → gleiche Besuchszahlen
            n_episodes=N_EP,
            max_steps=5,
            seed=seed,
        )
        max_single.append(float(Q[0, :].max()))

        env_dq = make_bias_mdp_for_double_q(N_ACTS, MEAN_R, SIGMA_R, gamma=0.9, env_seed=seed)
        _, _, Q_avg, _ = double_q_learning(
            env_dq,
            alpha=constant(ALPHA),
            epsilon=constant(1.0),
            n_episodes=N_EP,
            max_steps=5,
            seed=seed,
        )
        max_double.append(float(Q_avg[0, :].max()))

    mean_single = float(np.mean(max_single))
    mean_double = float(np.mean(max_double))
    assert mean_double < mean_single, (
        f"Double Q (mean max={mean_double:.4f}) ist NICHT kleiner als "
        f"Q-Learning (mean max={mean_single:.4f}). "
        f"Bias-MDP: N_ACTS={N_ACTS}, mean_r={MEAN_R}, sigma_r={SIGMA_R}, "
        f"N_EP={N_EP}, N_SEEDS={N_SEEDS}."
    )


# ------------------------------------------------------------------
# 4. Actor-Critic mit ipe_critic(steps=5) → pi*_VI
# ------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1])
def test_actor_critic_ipe5_agrees_with_vi(grid_env, Q_star_grid, seed):
    """actor_critic(ipe_critic(steps=5)) konvergiert zu Q*-optimaler Policy (>=14/16)."""
    V, policy, info = actor_critic(
        grid_env,
        critic=make_ipe_critic(steps=5),
        actor_update="greedy",
        n_outer_iter=30,
        seed=seed,
    )
    assert info["converged"], "Actor-Critic hat nicht konvergiert"
    agree = _policy_agreement(policy, Q_star_grid, grid_env)
    assert agree >= 14, (
        f"seed={seed}: Actor-Critic Übereinstimmung {agree}/16 < 14/16"
    )


def test_actor_critic_exact_matches_vi(grid_env, Q_star_grid):
    """actor_critic(ipe_exact) muss in allen 16 Zuständen eine Q*-optimale Policy finden."""
    V, policy, info = actor_critic(
        grid_env,
        critic=make_ipe_critic(steps="exact"),
        actor_update="greedy",
        n_outer_iter=50,
    )
    assert info["converged"]
    agree = _policy_agreement(policy, Q_star_grid, grid_env)
    assert agree == 16, f"Exact critic: Übereinstimmung {agree}/16 erwartet 16/16"


# ------------------------------------------------------------------
# 5. Smoke-Tests
# ------------------------------------------------------------------

def test_q_learning_val_mdp():
    """Q-Learning findet optimale Aktionen (a=1) auf Validierungs-MDP."""
    env = build_validation_mdp()
    Q, _ = q_learning(env, alpha=constant(0.1), epsilon=constant(0.1),
                      n_episodes=5000, max_steps=50, seed=0)
    assert np.argmax(Q[0, :]) == 1, f"Q[0,:] = {Q[0,:]}"
    assert np.argmax(Q[1, :]) == 1, f"Q[1,:] = {Q[1,:]}"


def test_sarsa_val_mdp():
    """SARSA (eps=0.01) findet optimale Aktionen auf Validierungs-MDP."""
    env = build_validation_mdp()
    Q, _ = sarsa(env, alpha=constant(0.1), epsilon=constant(0.01),
                 n_episodes=5000, max_steps=50, seed=0)
    assert np.argmax(Q[0, :]) == 1
    assert np.argmax(Q[1, :]) == 1


def test_double_q_val_mdp():
    """Double Q-Learning findet optimale Aktionen auf Validierungs-MDP."""
    env = build_validation_mdp()
    _, _, Q_avg, _ = double_q_learning(
        env, alpha=constant(0.1), epsilon=constant(0.1),
        n_episodes=5000, max_steps=50, seed=0,
    )
    assert np.argmax(Q_avg[0, :]) == 1
    assert np.argmax(Q_avg[1, :]) == 1


def test_episode_returns_length():
    """episode_returns in info hat korrekte Länge für alle Control-Algos."""
    env = build_validation_mdp()
    _, info = q_learning(env, alpha=constant(0.1), n_episodes=100, seed=0)
    assert len(info["episode_returns"]) == 100

    _, info_s = sarsa(env, alpha=constant(0.1), n_episodes=100, seed=0)
    assert len(info_s["episode_returns"]) == 100

    _, _, _, info_d = double_q_learning(env, alpha=constant(0.1), n_episodes=100, seed=0)
    assert len(info_d["episode_returns"]) == 100


def test_eval_fn_called_correctly():
    """eval_fn wird genau n_episodes // eval_every mal aufgerufen."""
    env = build_validation_mdp()
    calls = []
    Q_out, info = q_learning(
        env, alpha=constant(0.1), n_episodes=500, seed=0,
        eval_every=100, eval_fn=lambda Q: calls.append(1) or 0.0,
    )
    assert len(info["eval_history"]) == 5
    assert len(calls) == 5
