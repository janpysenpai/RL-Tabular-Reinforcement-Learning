"""Sample-basierte (TD(0)) Policy Evaluation (Skript Algorithms 17 & 18).

Totally asynchrone Updates aus Erfahrungstrajektorien:
    V(S_t) <- V(S_t) + alpha_n * (R + gamma * V(S_{t+1}) - V(S_t))
und analog für Q.

Aus Übungsblatt 6 Aufgabe 4 b).
"""

# TODO: td0_policy_evaluation_V / _Q
