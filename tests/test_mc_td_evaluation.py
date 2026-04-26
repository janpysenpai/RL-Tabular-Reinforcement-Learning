"""Tests für MC und TD(0) Policy Evaluation (Schritt 3a).

Validierung gegen ipe_V auf _validation_mdp und 4x4 GridWorld.

Zeitbudget < 30 s gesamt:
  _validation_mdp : 5000 Episoden × 5 Seeds (Episoden sehr kurz, ~6 Schritte E[T])
  GridWorld 4x4   : 3000 Episoden × 3 Seeds, max_steps=500, Toleranz 0.15

Toleranzen:
  |V_mc_first - V_ipe|_inf < 0.05  (validation_mdp)
  |V_mc_every - V_ipe|_inf < 0.05
  |V_td0      - V_ipe|_inf < 0.05
  Alle drei Kriterien gelten auch für GridWorld mit tol = 0.15.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabular_rl_project.envs._validation_mdp import build_validation_mdp
from tabular_rl_project.envs.gridworld import GridWorld
from tabular_rl_project.algos.iterative_policy_evaluation import ipe_V
from tabular_rl_project.algos.monte_carlo import mc_policy_evaluation_V
from tabular_rl_project.algos.td_policy_evaluation import td0_policy_evaluation_V
from tabular_rl_project.algos.schedules import one_over_n
from tabular_rl_project.algos.exploration import uniform_random_policy

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def val_env():
    return build_validation_mdp()


@pytest.fixture(scope="module")
def grid_env():
    return GridWorld(
        rows=4, cols=4,
        start=(0, 0),
        terminal_states=[(3, 3)],
        cell_rewards={(3, 3): 1.0},
        default_reward=0.0,
        gamma=0.9,
        wall_behavior="stay",
    )


@pytest.fixture(scope="module")
def val_policy(val_env):
    return uniform_random_policy(val_env)


@pytest.fixture(scope="module")
def grid_policy(grid_env):
    return uniform_random_policy(grid_env)


@pytest.fixture(scope="module")
def V_ipe_val(val_env, val_policy):
    V, _ = ipe_V(val_env, val_policy, tol=1e-9)
    return V


@pytest.fixture(scope="module")
def V_ipe_grid(grid_env, grid_policy):
    V, _ = ipe_V(grid_env, grid_policy, tol=1e-9)
    return V


# ------------------------------------------------------------------
# Validation MDP — 5 Seeds, 5000 Episoden, tol = 0.05
# ------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_mc_first_visit_val_mdp(val_env, val_policy, V_ipe_val, seed):
    V_mc, _ = mc_policy_evaluation_V(
        val_env, val_policy,
        n_episodes=5000, first_visit=True,
        max_steps=100, seed=seed,
    )
    err = float(np.abs(V_mc - V_ipe_val).max())
    assert err < 0.05, f"seed={seed}: |V_mc_first - V_ipe|_inf = {err:.4f}"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_mc_every_visit_val_mdp(val_env, val_policy, V_ipe_val, seed):
    V_mc, _ = mc_policy_evaluation_V(
        val_env, val_policy,
        n_episodes=5000, first_visit=False,
        max_steps=100, seed=seed,
    )
    err = float(np.abs(V_mc - V_ipe_val).max())
    assert err < 0.05, f"seed={seed}: |V_mc_every - V_ipe|_inf = {err:.4f}"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_td0_val_mdp(val_env, val_policy, V_ipe_val, seed):
    # TD(0) mit one_over_n() hat höhere Varianz als MC (Bootstrap-Ziel statt
    # echtem Return) → 10 000 Episoden für zuverlässige 0.05-Grenze.
    # Trotzdem <2 s: Validierungs-MDP hat E[T] ≈ 6 Schritte.
    V_td, _ = td0_policy_evaluation_V(
        val_env, val_policy,
        alpha=one_over_n(),
        n_episodes=10_000, max_steps=100, seed=seed,
    )
    err = float(np.abs(V_td - V_ipe_val).max())
    assert err < 0.05, f"seed={seed}: |V_td0 - V_ipe|_inf = {err:.4f}"


# ------------------------------------------------------------------
# GridWorld 4x4 — 3 Seeds, 3000 Episoden, tol = 0.15
# ------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mc_first_visit_gridworld(grid_env, grid_policy, V_ipe_grid, seed):
    V_mc, _ = mc_policy_evaluation_V(
        grid_env, grid_policy,
        n_episodes=3000, first_visit=True,
        max_steps=500, seed=seed,
    )
    err = float(np.abs(V_mc - V_ipe_grid).max())
    assert err < 0.15, f"seed={seed}: |V_mc_first - V_ipe|_inf = {err:.4f}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mc_every_visit_gridworld(grid_env, grid_policy, V_ipe_grid, seed):
    V_mc, _ = mc_policy_evaluation_V(
        grid_env, grid_policy,
        n_episodes=3000, first_visit=False,
        max_steps=500, seed=seed,
    )
    err = float(np.abs(V_mc - V_ipe_grid).max())
    assert err < 0.15, f"seed={seed}: |V_mc_every - V_ipe|_inf = {err:.4f}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_td0_gridworld(grid_env, grid_policy, V_ipe_grid, seed):
    V_td, _ = td0_policy_evaluation_V(
        grid_env, grid_policy,
        alpha=one_over_n(),
        n_episodes=3000, max_steps=500, seed=seed,
    )
    err = float(np.abs(V_td - V_ipe_grid).max())
    assert err < 0.15, f"seed={seed}: |V_td0 - V_ipe|_inf = {err:.4f}"


# ------------------------------------------------------------------
# Smoke-Tests: Q-Versionen konsistent mit V-Versionen
# ------------------------------------------------------------------

def test_mc_Q_consistent_with_V(val_env, val_policy):
    """Q^π und V^π müssen V_pi(s) = Σ_a π(a|s)*Q_pi(s,a) erfüllen."""
    from tabular_rl_project.algos.monte_carlo import mc_policy_evaluation_Q
    V, _ = mc_policy_evaluation_V(val_env, val_policy, n_episodes=5000, seed=7)
    Q, _ = mc_policy_evaluation_Q(val_env, val_policy, n_episodes=5000, seed=7)
    V_from_Q = (val_policy * Q).sum(axis=1)
    assert np.abs(V - V_from_Q).max() < 0.1


def test_td0_Q_consistent_with_V(val_env, val_policy):
    """TD(0) Q^π und V^π müssen ebenfalls konsistent sein."""
    from tabular_rl_project.algos.td_policy_evaluation import td0_policy_evaluation_Q
    V, _ = td0_policy_evaluation_V(val_env, val_policy, alpha=one_over_n(),
                                    n_episodes=5000, seed=7)
    Q, _ = td0_policy_evaluation_Q(val_env, val_policy, alpha=one_over_n(),
                                    n_episodes=5000, seed=7)
    V_from_Q = (val_policy * Q).sum(axis=1)
    assert np.abs(V - V_from_Q).max() < 0.1


# ------------------------------------------------------------------
# Schedule-Sanity
# ------------------------------------------------------------------

def test_schedules_basic():
    from tabular_rl_project.algos.schedules import constant, one_over_n, polynomial, linear_decay
    assert constant(0.1)(1) == pytest.approx(0.1)
    assert constant(0.1)(1000) == pytest.approx(0.1)
    assert one_over_n()(1) == pytest.approx(1.0)
    assert one_over_n()(2) == pytest.approx(0.5)
    assert polynomial(0.5)(4) == pytest.approx(0.5)
    ld = linear_decay(1.0, 0.0, 11)
    assert ld(1) == pytest.approx(1.0)
    assert ld(6) == pytest.approx(0.5)
    assert ld(11) == pytest.approx(0.0)
    assert ld(100) == pytest.approx(0.0)


def test_epsilon_greedy_sums_to_one(val_env):
    from tabular_rl_project.algos.exploration import epsilon_greedy
    Q = np.zeros((val_env.n_states, val_env.n_actions))
    policy = epsilon_greedy(Q, val_env, eps=0.1)
    for s in val_env.states:
        assert policy[s].sum() == pytest.approx(1.0)
