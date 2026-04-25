"""Exact Policy Iteration / Greedy Actor-Critic (Skript Algorithm 10).

Wechselt zwischen Policy Evaluation (Critic) und greedy Policy Improvement
(Actor) bis die Policy stabil bleibt.

Critic-Konfiguration:
    critic="exact"     IPE bis Konvergenz (tol=1e-9).
    critic=int         Genau n IPE-Schritte pro Outer-Iteration.
    critic=callable    critic(outer_iter) liefert die Schrittzahl
                       (deckt „functionally dependent steps" ab).

use_q=True             Gibt Q* statt V* zurück; Policy Improvement intern
                       über V→Q-Konversion, äquivalent zu Q-Iteration.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from ..envs.mdp_base import FiniteMDP
from .iterative_policy_evaluation import ipe_V
from .value_iteration import greedy_policy_from_v


def _uniform_policy(env: FiniteMDP) -> np.ndarray:
    """Uniforme Policy über alle erlaubten Aktionen, Shape (S, A)."""
    S, A = env.n_states, env.n_actions
    policy = np.zeros((S, A))
    for s in env.states:
        acts = env.allowed_actions[s]
        policy[s, acts] = 1.0 / len(acts)
    return policy


def policy_iteration(
    env: FiniteMDP,
    gamma: Optional[float] = None,
    critic: Union[str, int, Callable] = "exact",
    max_outer_iter: int = 100,
    use_q: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Policy Iteration (Skript Algorithm 10).

    Args:
        env:            Finite MDP mit bekannter Dynamik.
        gamma:          Diskontierung; None → env.gamma.
        critic:         Steuert die Critic-Schritte:
                        "exact" → IPE bis tol=1e-9 konvergiert;
                        int     → exakt n IPE-Schritte;
                        callable → critic(k) gibt Schrittzahl in Outer-Iteration k.
        max_outer_iter: Maximale Anzahl Policy-Improvement-Schritte.
        use_q:          True → gibt Q (S, A) zurück statt V (S,).

    Returns:
        result: V* shape (S,) oder Q* shape (S, A) je nach use_q.
        policy: Greedy-Policy, Shape (S, A), one-hot.
        info:   {"outer_iterations": int}.
    """
    gamma = gamma if gamma is not None else env.gamma
    policy = _uniform_policy(env)

    V = np.zeros(env.n_states)
    outer_iter = 0

    for outer_iter in range(max_outer_iter):
        # ---- Critic: Policy Evaluation ----
        if critic == "exact":
            V, _ = ipe_V(env, policy, gamma=gamma, tol=1e-9)
        elif isinstance(critic, int):
            V, _ = ipe_V(env, policy, gamma=gamma, max_iter=critic, tol=None)
        elif callable(critic):
            n_steps = int(critic(outer_iter))
            V, _ = ipe_V(env, policy, gamma=gamma, max_iter=n_steps, tol=None)
        else:
            raise ValueError(f"Unbekannter critic-Typ: {critic!r}")

        # ---- Actor: Greedy Policy Improvement ----
        new_policy = greedy_policy_from_v(env, V, gamma=gamma)

        if np.array_equal(policy, new_policy):
            break
        policy = new_policy

    if use_q:
        P = env.transition_probabilities
        R = env.expected_rewards
        Q = R + gamma * np.einsum("sap,p->sa", P, V)
        return Q, policy, {"outer_iterations": outer_iter + 1}

    return V, policy, {"outer_iterations": outer_iter + 1}
