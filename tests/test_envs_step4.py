"""Validierungstests für Schritt-4-Environments (Cliff Walk, Extreme MDPs,
Submission GridWorld).

Zeitbudget < 20 s.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabular_rl_project.envs.cliff_walk import CliffWalk
from tabular_rl_project.envs.extreme_mdps import (
    make_two_state_loop,
    make_sparse_chain,
    make_noisy_gridworld,
    make_bias_mdp_for_double_q,
)
from tabular_rl_project.envs.gridworld_4x4_submission import (
    GridWorld4x4Submission,
    FAKE_GOAL_RC, GOAL_RC, START_RC, SR_RCS,
    FAKE_GOAL_REWARD, GOAL_REWARD, SR_EXPECTED, DEF_EXPECTED,
    _rc_to_s,
)
from tabular_rl_project.algos.value_iteration import value_iteration_V


# ------------------------------------------------------------------
# Hilfsmethode: P-Zeilensummen
# ------------------------------------------------------------------

def _check_row_sums(env) -> None:
    P = env.transition_probabilities
    for s in env.states:
        for a in env.allowed_actions[s]:
            total = P[s, a].sum()
            assert abs(total - 1.0) < 1e-9, (
                f"P[{s},{a},:] summiert zu {total:.10f}"
            )


# ==================================================================
# 1. Cliff Walk
# ==================================================================

class TestCliffWalk:
    @pytest.fixture(scope="class")
    def cw(self):
        return CliffWalk()

    def test_validate(self, cw):
        cw.validate()

    def test_row_sums(self, cw):
        _check_row_sums(cw)

    def test_dimensions(self, cw):
        assert cw.n_states == 48
        assert cw.n_actions == 4
        assert len(cw.cliff_states) == 10

    def test_start_goal(self, cw):
        assert cw.start_state == cw._rc(3, 0)
        assert cw.goal_state == cw._rc(3, 11)
        assert cw.is_terminal(cw.goal_state)
        assert not cw.is_terminal(cw.start_state)

    def test_cliff_redirect(self, cw):
        """Schritt in Cliff-Zelle → Teleport zu Start, R=-100."""
        # Von (3,0) nach rechts → wäre (3,1) = Cliff
        s_start = cw._rc(3, 0)
        ns = int(cw.transition_probabilities[s_start, cw.RIGHT].argmax())
        assert ns == cw.start_state, f"Cliff-Betreten soll zu Start teleportieren, ns={ns}"
        assert cw.expected_rewards[s_start, cw.RIGHT] == -100.0

    def test_cliff_not_terminal(self, cw):
        for cliff_s in cw.cliff_states:
            assert not cw.is_terminal(cliff_s), f"Cliff-Zelle {cliff_s} darf nicht terminal sein"

    def test_goal_step_reward(self, cw):
        """Schritt in Ziel gibt step_reward=-1."""
        # Von (2,11) nach unten → Ziel (3,11)
        s_above_goal = cw._rc(2, 11)
        assert cw.expected_rewards[s_above_goal, cw.DOWN] == cw.step_reward

    def test_step_mechanics(self, cw):
        """step() in Cliff → zurück bei Start, nicht done."""
        cw._current_state = cw._rc(3, 0)
        ns, r, done = cw.step(cw.RIGHT)
        assert ns == cw.start_state
        assert r == -100.0
        assert not done

    def test_step_goal_terminal(self, cw):
        """step() in Ziel → done=True."""
        cw._current_state = cw._rc(2, 11)
        ns, r, done = cw.step(cw.DOWN)
        assert ns == cw.goal_state
        assert done

    def test_wall_stay(self, cw):
        """Schritt gegen Wand bleibt im aktuellen Zustand."""
        s = cw._rc(0, 0)
        ns = int(cw.transition_probabilities[s, cw.UP].argmax())
        assert ns == s

    def test_reset(self, cw):
        assert cw.reset() == cw.start_state


# ==================================================================
# 2. Extreme MDPs
# ==================================================================

class TestTwoStateLoop:
    def test_validate(self):
        make_two_state_loop().validate()

    def test_row_sums(self):
        _check_row_sums(make_two_state_loop())

    @pytest.mark.parametrize("p_stay,gamma", [(0.9, 0.99), (0.5, 0.9)])
    def test_v_star_analytical(self, p_stay, gamma):
        """V*(0) = 1 / (1 - gamma * p_stay)."""
        env = make_two_state_loop(p_stay=p_stay, gamma=gamma)
        V, _ = value_iteration_V(env, tol=1e-9)
        expected = 1.0 / (1.0 - gamma * p_stay)
        assert abs(V[0] - expected) < 1e-5, f"V*(0)={V[0]:.6f} ≠ {expected:.6f}"
        assert abs(V[1]) < 1e-9


class TestSparseChain:
    def test_validate(self):
        make_sparse_chain().validate()

    def test_row_sums(self):
        _check_row_sums(make_sparse_chain())

    def test_v_star_analytical(self):
        """V*(s) = gamma^(n-2-s) für s < n-1; V*(n-1) = 0.

        V*(n-2) = R[n-2, right] + gamma*V*(n-1) = 1.0 (terminal reward beim Betreten).
        V*(s)   = gamma^(n-2-s) für s < n-1.
        """
        n = 10
        gamma = 0.95
        env = make_sparse_chain(n_states=n, gamma=gamma)
        V, _ = value_iteration_V(env, tol=1e-9)
        for s in range(n - 1):
            expected = gamma ** (n - 2 - s)
            assert abs(V[s] - expected) < 1e-5, f"V*({s})={V[s]:.6f} ≠ {expected:.6f}"
        assert abs(V[n - 1]) < 1e-9


class TestNoisyGridworld:
    def test_validate(self):
        make_noisy_gridworld().validate()

    def test_row_sums(self):
        _check_row_sums(make_noisy_gridworld())

    def test_goal_absorbing(self):
        env = make_noisy_gridworld()
        goal = env.n_states - 1
        assert env.is_terminal(goal)
        # Von Ziel: bleibt im Ziel
        P = env.transition_probabilities
        assert abs(P[goal, :, goal].mean() - 1.0) < 1e-9

    def test_slip_prob_effect(self):
        """Mit slip=0 entspricht noisy_gridworld einer deterministischen GridWorld."""
        env = make_noisy_gridworld(slip_prob=0.0)
        env.validate()
        # Jede Zeile P[s,a,:] ist ein One-Hot
        P = env.transition_probabilities
        for s in env.states:
            for a in env.allowed_actions[s]:
                assert P[s, a].max() == 1.0, f"P[{s},{a},:] nicht deterministisch"


class TestBiasMDP:
    def test_validate(self):
        make_bias_mdp_for_double_q().validate()

    def test_row_sums(self):
        _check_row_sums(make_bias_mdp_for_double_q())

    def test_v_star(self):
        """V*(0) = reward_mean, V*(1) = 0."""
        mean = -0.1
        env = make_bias_mdp_for_double_q(reward_mean=mean)
        V, _ = value_iteration_V(env, tol=1e-9)
        assert abs(V[0] - mean) < 1e-5, f"V*(0)={V[0]:.6f} ≠ {mean}"
        assert abs(V[1]) < 1e-9

    def test_stochastic_sampling(self):
        """_sample_reward liefert Werte nahe reward_mean (MC-Schätzung)."""
        mean = -0.1
        env = make_bias_mdp_for_double_q(n_acts=2, reward_mean=mean, reward_std=1.0, env_seed=0)
        samples = [env._sample_reward(0, 0, 1) for _ in range(5000)]
        assert abs(np.mean(samples) - mean) < 0.05, (
            f"MC-Schätzung {np.mean(samples):.4f} zu weit von {mean}"
        )


# ==================================================================
# 3. Submission GridWorld
# ==================================================================

class TestSubmissionGrid:
    @pytest.fixture(scope="class")
    def sg(self):
        return GridWorld4x4Submission(noise=False)

    @pytest.fixture(scope="class")
    def sg_noise(self):
        return GridWorld4x4Submission(noise=True, seed=42)

    def test_validate(self, sg):
        sg.validate()

    def test_validate_noise(self, sg_noise):
        sg_noise.validate()

    def test_row_sums(self, sg):
        _check_row_sums(sg)

    def test_layout_start(self, sg):
        assert sg.start_state == _rc_to_s(*START_RC)

    def test_layout_terminals(self, sg):
        assert _rc_to_s(*FAKE_GOAL_RC) in sg.terminal_states
        assert _rc_to_s(*GOAL_RC) in sg.terminal_states
        assert sg.n_states == 16
        assert len(sg.terminal_states) == 2

    def test_fake_goal_reward(self, sg):
        """Betreten von F (0,0) gibt R=0.65."""
        fg = _rc_to_s(*FAKE_GOAL_RC)
        # von (0,1)=1, LEFT=2 → (0,0)=0
        assert abs(sg.expected_rewards[1, 2] - FAKE_GOAL_REWARD) < 1e-9

    def test_goal_reward(self, sg):
        """Betreten von G (3,1) gibt R=1.0."""
        # von (3,0)=12, RIGHT=3 → (3,1)=13
        assert abs(sg.expected_rewards[12, 3] - GOAL_REWARD) < 1e-9

    def test_sr_expected_reward(self, sg):
        """Betreten einer SR-Zelle: E[R] = -0.05."""
        # von (2,1)=9, RIGHT=3 → (2,2)=10 (SR)
        assert abs(sg.expected_rewards[9, 3] - SR_EXPECTED) < 1e-9

    def test_default_expected_reward(self, sg):
        """Default-Zellen: E[R] = 0.0."""
        # von (0,2)=2, LEFT=2 → (0,1)=1 (default)
        assert abs(sg.expected_rewards[2, 2] - DEF_EXPECTED) < 1e-9

    def test_sr_stochastic_sampling(self, sg_noise):
        """Mit noise=True sampelt SR-Reward aus {-2.1, 2.0}."""
        sg_noise._current_state = 9  # (2,1)
        samples = [sg_noise._sample_reward(9, 3, 10) for _ in range(1000)]
        uniq = set(round(x, 5) for x in samples)
        assert -2.1 in uniq or any(abs(v - (-2.1)) < 1e-3 for v in uniq)
        assert 2.0 in uniq or any(abs(v - 2.0) < 1e-3 for v in uniq)
        assert abs(np.mean(samples) - SR_EXPECTED) < 0.15

    def test_deterministic_without_noise(self, sg):
        """Ohne noise: _sample_reward = expected_reward."""
        sg._current_state = 9
        r = sg._sample_reward(9, 3, 10)
        assert abs(r - sg.expected_rewards[9, 3]) < 1e-9

    def test_terminal_reward_deterministic_with_noise(self, sg_noise):
        """Terminal-Reward bleibt deterministisch bei noise=True."""
        fg = _rc_to_s(*FAKE_GOAL_RC)
        r = sg_noise._sample_reward(1, 2, fg)
        assert abs(r - FAKE_GOAL_REWARD) < 1e-9

    def test_value_iteration_converges(self, sg):
        """value_iteration_V läuft auf dem Submission-Grid ohne Fehler."""
        V, info = value_iteration_V(sg, tol=1e-9)
        assert info["iterations"] > 0
        # V*(Start) muss zwischen 0 und GOAL_REWARD liegen
        assert 0.0 < V[sg.start_state] < GOAL_REWARD

    def test_reset(self, sg):
        assert sg.reset() == sg.start_state
