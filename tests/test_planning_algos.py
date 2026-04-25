"""Tests für modellbasierte Planungsalgorithmen (Schritt 2).

Validierungsbasis: build_validation_mdp() — analytische V*, Q* im Modul-Docstring.
Zusatz: 4x4 GridWorld (Start (0,0), Goal (3,3), R=1, gamma=1.0).
"""

import numpy as np
import pytest

from tabular_rl_project.envs._validation_mdp import build_validation_mdp, V_STAR, Q_STAR
from tabular_rl_project.envs.gridworld import GridWorld
from tabular_rl_project.algos.value_iteration import value_iteration_V, value_iteration_Q
from tabular_rl_project.algos.iterative_policy_evaluation import ipe_V, ipe_Q
from tabular_rl_project.algos.policy_iteration import policy_iteration
from tabular_rl_project.algos.finite_time_dp import (
    finite_time_policy_evaluation,
    finite_time_optimal_control,
)

ATOL = 1e-6


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def val_env():
    return build_validation_mdp()


@pytest.fixture
def grid_env():
    """gamma=1.0 — nur für VI geeignet (IPE divergiert bei gamma=1, Blatt 5 A1)."""
    return GridWorld(
        rows=4, cols=4,
        start=(0, 0),
        terminal_states=[(3, 3)],
        cell_rewards={(3, 3): 1.0},
        default_reward=0.0,
        gamma=1.0,
        wall_behavior="stay",
    )


@pytest.fixture
def grid_env_09():
    """gamma=0.9 — für PI/IPE-Tests (Kontraktionsbedingung erfüllt)."""
    return GridWorld(
        rows=4, cols=4,
        start=(0, 0),
        terminal_states=[(3, 3)],
        cell_rewards={(3, 3): 1.0},
        default_reward=0.0,
        gamma=0.9,
        wall_behavior="stay",
    )


def _uniform_policy(env):
    S, A = env.n_states, env.n_actions
    policy = np.zeros((S, A))
    for s in env.states:
        acts = env.allowed_actions[s]
        policy[s, acts] = 1.0 / len(acts)
    return policy


def _optimal_policy_val(env):
    """Deterministische Optimal-Policy für das Validierungs-MDP."""
    S, A = env.n_states, env.n_actions
    policy = np.zeros((S, A))
    policy[0, 1] = 1.0   # s=0: a=1
    policy[1, 1] = 1.0   # s=1: a=1
    policy[2, 0] = 1.0   # s=2: terminal, beliebig
    return policy


# ------------------------------------------------------------------
# Value Iteration — Validierungs-MDP
# ------------------------------------------------------------------

def test_vi_v_sync_accuracy(val_env):
    V, info = value_iteration_V(val_env, tol=1e-9)
    assert np.allclose(V, V_STAR, atol=ATOL), f"V={V}, erwartet={V_STAR}"
    assert info["iterations"] < 50


def test_vi_q_sync_accuracy(val_env):
    Q, info = value_iteration_Q(val_env, tol=1e-9)
    assert np.allclose(Q, Q_STAR, atol=ATOL), f"Q={Q}, erwartet={Q_STAR}"
    assert info["iterations"] < 50


def test_vi_v_sync_vs_async(val_env):
    V_sync, _ = value_iteration_V(val_env, sync=True, tol=1e-9)
    V_async, _ = value_iteration_V(val_env, sync=False, tol=1e-9)
    assert np.allclose(V_sync, V_async, atol=ATOL)


def test_vi_q_sync_vs_async(val_env):
    Q_sync, _ = value_iteration_Q(val_env, sync=True, tol=1e-9)
    Q_async, _ = value_iteration_Q(val_env, sync=False, tol=1e-9)
    assert np.allclose(Q_sync, Q_async, atol=ATOL)


# ------------------------------------------------------------------
# Iterative Policy Evaluation — Validierungs-MDP
# ------------------------------------------------------------------

def test_ipe_v_uniform_vs_linear_system(val_env):
    """IPE mit uniformer Policy muss (I - gamma*P_pi)V = r_pi lösen."""
    policy = _uniform_policy(val_env)
    V, _ = ipe_V(val_env, policy, tol=1e-9)

    gamma = val_env.gamma
    P = val_env.transition_probabilities
    R = val_env.expected_rewards
    S = val_env.n_states

    P_pi = np.einsum("sa,sap->sp", policy, P)
    r_pi = (policy * R).sum(axis=1)
    V_exact = np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)

    assert np.allclose(V, V_exact, atol=ATOL)


def test_ipe_v_optimal_policy_matches_vi(val_env):
    """IPE mit optimaler Policy soll V* aus VI reproduzieren."""
    policy = _optimal_policy_val(val_env)
    V_ipe, _ = ipe_V(val_env, policy, tol=1e-9)
    V_vi, _ = value_iteration_V(val_env, tol=1e-9)
    assert np.allclose(V_ipe, V_vi, atol=ATOL)


def test_ipe_q_uniform_consistent_with_ipe_v(val_env):
    """Q-Version und V-Version müssen konsistente Ergebnisse liefern."""
    policy = _uniform_policy(val_env)
    V, _ = ipe_V(val_env, policy, tol=1e-9)
    Q, _ = ipe_Q(val_env, policy, tol=1e-9)
    # V_pi(s) = sum_a pi(a|s) * Q_pi(s,a)
    V_from_Q = (policy * Q).sum(axis=1)
    assert np.allclose(V, V_from_Q, atol=ATOL)


# ------------------------------------------------------------------
# Policy Iteration — Validierungs-MDP
# ------------------------------------------------------------------

def test_pi_exact_critic(val_env):
    V, policy, info = policy_iteration(val_env, critic="exact")
    assert np.allclose(V, V_STAR, atol=ATOL)
    assert np.argmax(policy[0]) == 1, "Optimale Aktion in s=0 muss a=1 sein"
    assert np.argmax(policy[1]) == 1, "Optimale Aktion in s=1 muss a=1 sein"


def test_pi_fixed_steps_critic(val_env):
    V, policy, _ = policy_iteration(val_env, critic=5)
    assert np.allclose(V, V_STAR, atol=ATOL)
    assert np.argmax(policy[0]) == 1
    assert np.argmax(policy[1]) == 1


def test_pi_callable_critic_smoke(val_env):
    """Smoke-Test: callable critic findet optimale Policy."""
    V, policy, _ = policy_iteration(val_env, critic=lambda k: k + 1)
    assert np.allclose(V, V_STAR, atol=ATOL)
    assert np.argmax(policy[0]) == 1
    assert np.argmax(policy[1]) == 1


def test_pi_use_q(val_env):
    """use_q=True gibt konsistentes Q* zurück."""
    Q, policy, _ = policy_iteration(val_env, critic="exact", use_q=True)
    assert np.allclose(Q, Q_STAR, atol=ATOL)


# ------------------------------------------------------------------
# Finite-Time DP — Validierungs-MDP (T=2, Algorithm 13)
# ------------------------------------------------------------------

def test_finite_time_optimal_control_t2(val_env):
    """Analytisch:
        V*_2 = [0, 0, 0]
        V*_1 = [0, 1, 0]
        V*_0 = [1, 1, 0]
        pi*_0 = [1, 1, 0]   (a=1 optimal in s=0 und s=1)
    """
    V, pi = finite_time_optimal_control(val_env, T=2)

    assert np.allclose(V[2], [0.0, 0.0, 0.0], atol=ATOL)
    assert np.allclose(V[1], [0.0, 1.0, 0.0], atol=ATOL)
    assert np.allclose(V[0], [1.0, 1.0, 0.0], atol=ATOL)
    assert pi[0, 0] == 1, "pi*_0(s=0) muss a=1 sein"
    assert pi[0, 1] == 1, "pi*_0(s=1) muss a=1 sein"


def test_finite_time_policy_eval_consistent(val_env):
    """Policy Evaluation mit optimaler Policy soll V*_t reproduzieren."""
    policy = _optimal_policy_val(val_env)
    V_eval = finite_time_policy_evaluation(val_env, policy, T=2)
    V_opt, _ = finite_time_optimal_control(val_env, T=2)
    # Unter optimaler Policy gilt V^pi = V* für finite-time
    assert np.allclose(V_eval, V_opt, atol=ATOL)


# ------------------------------------------------------------------
# GridWorld-Tests
# ------------------------------------------------------------------

def test_vi_v_gridworld_start_value(grid_env):
    """VI muss V*(Start) = 1.0 finden (gamma=1.0, Goal R=1)."""
    V, _ = value_iteration_V(grid_env, tol=1e-9, max_iter=200)
    assert np.isclose(V[grid_env.start_state], 1.0, atol=ATOL)


def test_pi_gridworld_reaches_goal(grid_env_09):
    """PI (gamma=0.9) muss eine Policy finden, die das Ziel in ≤6 Schritten erreicht.

    Hinweis: gamma=1.0 ist für IPE ein Sonderfall (kein eindeutiger Fixpunkt),
    daher wird hier gamma=0.9 verwendet (Übungsblatt 5 Aufgabe 1).
    """
    V, policy, _ = policy_iteration(grid_env_09, critic="exact")

    # Greedy-Rollout: optimaler Pfad hat Länge 6 (3 rechts + 3 runter)
    state = grid_env_09.reset()
    for _ in range(7):
        if grid_env_09.is_terminal(state):
            break
        state, _, _ = grid_env_09.step(int(np.argmax(policy[state])))

    assert grid_env_09.is_terminal(state), (
        f"Policy hat Ziel nach 6 Schritten nicht erreicht (letzter Zustand: {state})"
    )
