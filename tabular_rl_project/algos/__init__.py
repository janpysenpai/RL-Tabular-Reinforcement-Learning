"""Tabular-RL-Algorithmen.

Modellbasiert (Skript Kap. 3, bekannte Übergangswahrscheinlichkeiten):
    value_iteration               Algorithm 6
    iterative_policy_evaluation   Algorithm 7 (sync) + Algorithm 8 (async)
    policy_iteration              Algorithm 10 (exact actor-critic)
    finite_time_dp                Algorithms 12 & 13 (finite-time MDPs)

Sample-basiert (Skript Kap. 2/4):
    monte_carlo                   Algorithm 14/15 + every-visit (Blatt 6 Alg. 1)
    td_policy_evaluation          Algorithm 17/18 (totally async, V & Q)
    q_learning                    Algorithm 18 (Q-Learning)
    double_q_learning             Double-Q-Learning (Bias Problem, Blatt 7/8)
    sarsa                         Algorithm 20
    actor_critic                  General Actor-Critic (Blatt 7 Aufg. 3, Algorithm 11)

Hilfsmodule:
    schedules                     Stepsize- und Epsilon-Zeitpläne (1/n, konstant, ...).
    exploration                   Verhaltenspolicies: uniform, epsilon-greedy, ...
"""
