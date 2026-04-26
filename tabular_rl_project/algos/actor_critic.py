"""General Actor-Critic (Skript Algorithm 11, Übungsblatt 7 Aufgabe 3).

Kombiniert einen beliebigen Critic (Policy Evaluation) mit einem
Policy-Improvement-Schritt (Actor):

    Outer-Loop:
        1. Critic:  V ← critic(env, policy, gamma)   (beliebige Schätzung V^π)
        2. Actor:   policy ← greedy(V) oder ε-greedy(Q_from_V)
        3. Abbruch wenn Policy stabil

Für "greedy" Actor wird die Greedy-Policy über das Modell (P, R) berechnet
(modellbasierter Actor, Gleichung Q(s,a) = R(s,a) + γ·Σ P(s,a,s')·V(s')).

Critic-Factories für bequeme Parametrisierung:
    make_ipe_critic(steps)      ipe_V mit genau steps Iterationen (oder "exact")
    make_td0_critic(...)        td0_policy_evaluation_V
    make_mc_critic(...)         mc_policy_evaluation_V

Ziel des Submission-Tasks e): Vergleich mit reiner stochastischer Kontrolle
(Bellman Optimalitäts-Operator / Value Iteration).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..envs.mdp_base import FiniteMDP
from .iterative_policy_evaluation import ipe_V
from .value_iteration import greedy_policy_from_v, _q_from_v
from .exploration import epsilon_greedy


# ------------------------------------------------------------------
# Critic-Factories
# ------------------------------------------------------------------

def make_ipe_critic(
    steps: Union[int, str] = "exact",
    sync: bool = True,
) -> Callable[[FiniteMDP, np.ndarray, float], Tuple[np.ndarray, Dict]]:
    """Critic-Callable basierend auf Iterativer Policy Evaluation.

    Parameters
    ----------
    steps : int or "exact"
        Anzahl IPE-Iterationen pro Critic-Aufruf;
        "exact" → konvergiert bis tol=1e-9.
    sync : bool
        Synchrones (True) oder totally asynchrones (False) Update.

    Returns
    -------
    Callable[[FiniteMDP, np.ndarray, float], Tuple[np.ndarray, dict]]
        ``critic(env, policy, gamma) → (V, info)``
    """
    def _critic(env: FiniteMDP, policy: np.ndarray, gamma: float) -> Tuple[np.ndarray, Dict]:
        if steps == "exact":
            return ipe_V(env, policy, gamma=gamma, sync=sync, tol=1e-9)
        return ipe_V(env, policy, gamma=gamma, sync=sync,
                     max_iter=int(steps), tol=None)
    return _critic


def make_td0_critic(
    n_episodes: int = 500,
    alpha: Union[float, Any] = 0.1,
    max_steps: int = 500,
    seed: Optional[int] = None,
) -> Callable[[FiniteMDP, np.ndarray, float], Tuple[np.ndarray, Dict]]:
    """Critic-Callable basierend auf TD(0) Policy Evaluation.

    Jeder Critic-Aufruf bekommt einen deterministisch abgeleiteten Seed
    (via Seed-Generator), damit die Outer-Loop reproducierbar bleibt.

    Parameters
    ----------
    n_episodes, alpha, max_steps
        Parameter für td0_policy_evaluation_V.
    seed : int, optional
        Basis-Seed; jeder Critic-Aufruf bekommt einen anderen Sub-Seed.
    """
    from .td_policy_evaluation import td0_policy_evaluation_V
    _seed_gen = np.random.default_rng(seed)

    def _critic(env: FiniteMDP, policy: np.ndarray, gamma: float) -> Tuple[np.ndarray, Dict]:
        sub_seed = int(_seed_gen.integers(0, 2**31))
        return td0_policy_evaluation_V(env, policy, alpha=alpha, gamma=gamma,
                                       n_episodes=n_episodes, max_steps=max_steps,
                                       seed=sub_seed)
    return _critic


def make_mc_critic(
    n_episodes: int = 500,
    first_visit: bool = True,
    max_steps: int = 500,
    seed: Optional[int] = None,
) -> Callable[[FiniteMDP, np.ndarray, float], Tuple[np.ndarray, Dict]]:
    """Critic-Callable basierend auf Monte-Carlo Policy Evaluation.

    Parameters
    ----------
    n_episodes, first_visit, max_steps
        Parameter für mc_policy_evaluation_V.
    seed : int, optional
        Basis-Seed; jeder Critic-Aufruf bekommt einen anderen Sub-Seed.
    """
    from .monte_carlo import mc_policy_evaluation_V
    _seed_gen = np.random.default_rng(seed)

    def _critic(env: FiniteMDP, policy: np.ndarray, gamma: float) -> Tuple[np.ndarray, Dict]:
        sub_seed = int(_seed_gen.integers(0, 2**31))
        return mc_policy_evaluation_V(env, policy, gamma=gamma,
                                      n_episodes=n_episodes, first_visit=first_visit,
                                      max_steps=max_steps, seed=sub_seed)
    return _critic


# ------------------------------------------------------------------
# Actor-Critic Hauptfunktion
# ------------------------------------------------------------------

def _uniform_policy(env: FiniteMDP) -> np.ndarray:
    S, A = env.n_states, env.n_actions
    policy = np.zeros((S, A))
    for s in env.states:
        acts = env.allowed_actions[s]
        policy[s, acts] = 1.0 / len(acts)
    return policy


def actor_critic(
    env: FiniteMDP,
    critic: Callable[[FiniteMDP, np.ndarray, float], Tuple[np.ndarray, Dict]],
    actor_update: str = "greedy",
    gamma: Optional[float] = None,
    n_outer_iter: int = 100,
    epsilon: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """General Actor-Critic (Skript Algorithmus 11).

    Parameters
    ----------
    env : FiniteMDP
        Umgebung; wird für den modellbasierten Actor-Schritt (P, R) genutzt.
    critic : Callable
        ``critic(env, policy, gamma) → (V, info)``
        Schätzt V^π für die aktuelle Policy. Kann ipe_V, td0 oder mc sein.
    actor_update : {"greedy", "eps_greedy"}
        Policy-Verbesserungsregel:
            "greedy"     → deterministisch, argmax_a Q(s,a) aus Modell
            "eps_greedy" → epsilon-greedy bzgl. Q(s,a) aus Modell
        Beide Varianten berechnen Q via Q(s,a) = R(s,a) + γ·Σ P·V (modellbasiert).
    gamma : float, optional
        Diskontierung; None → env.gamma.
    n_outer_iter : int
        Maximale Anzahl Policy-Improvement-Schritte (Outer-Loop).
    epsilon : float
        Explorationsrate für "eps_greedy" Actor (ignoriert bei "greedy").
    seed : int, optional
        Seed für den rng bei "eps_greedy"-Actor (nur relevant für Tiebreaking;
        Critic-Factories haben eigene Seeds).

    Returns
    -------
    V : np.ndarray
        Wertfunktion der stabilen Policy, Shape (S,).
    policy : np.ndarray
        Stabile (verbesserte) Policy, Shape (S, A).
    info : dict
        "outer_iterations"    Anzahl durchgeführter Outer-Schritte.
        "converged"           True wenn Policy vor n_outer_iter stabil.
        "policy_stable_at"    Outer-Iteration des ersten stabilen Schritts.
        "critic_infos"        Liste der info-Dicts vom Critic pro Outer-Iteration.
    """
    gamma = gamma if gamma is not None else env.gamma
    policy = _uniform_policy(env)

    V = np.zeros(env.n_states)
    converged = False
    stable_at = n_outer_iter
    critic_infos: List[Dict] = []

    for outer in range(n_outer_iter):
        # ---- Critic: Policy Evaluation ----
        V, c_info = critic(env, policy, gamma)
        critic_infos.append(c_info)

        # ---- Actor: Policy Improvement via Modell ----
        Q = _q_from_v(V, env.transition_probabilities, env.expected_rewards, gamma)

        if actor_update == "greedy":
            new_policy = greedy_policy_from_v(env, V, gamma=gamma)
        elif actor_update == "eps_greedy":
            new_policy = epsilon_greedy(Q, env, epsilon)
        else:
            raise ValueError(f"Unbekannter actor_update: {actor_update!r}")

        if np.array_equal(policy, new_policy):
            converged = True
            stable_at = outer + 1
            policy = new_policy
            break

        policy = new_policy

    return V, policy, {
        "outer_iterations": outer + 1,
        "converged": converged,
        "policy_stable_at": stable_at,
        "critic_infos": critic_infos,
    }
