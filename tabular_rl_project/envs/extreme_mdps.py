"""„Extreme" MDPs für Submission-Task c) (Blatt 8 Aufgabe 4 c)).

Drei Effekte sollen jeweils maximal sichtbar gemacht werden:

    1. Backpropagation
       Lange Korridor-/Ketten-MDPs mit Reward erst am Ende — zeigt, wie
       Information zellweise rückwärts propagiert wird.
       (Verbindung zu Blatt 5 Aufgabe 5 und Blatt 6 Aufgabe 5.)

    2. Robust Reinforcement Learning
       Cliff-/Risk-Variante (SARSA vs. Q-Learning), siehe cliff_walk.py.

    3. Overestimation Bias
       Multi-Step Bandit (oder Grid mit stark verrauschten Aktionen) bei
       dem max E[X_i] gegenüber E[max X_i] auseinanderläuft. Verbindung
       zu Blatt 7 Aufgabe 6 (Bias Problem) und Double Q-Learning.
"""

# TODO: build_chain_mdp(length: int)               # Backpropagation
# TODO: build_overestimation_mdp(n_actions: int)   # Bias Problem
# (build_cliff_walk lebt in cliff_walk.py)
