"""Q-Learning (Skript Algorithm 18 / 19, Übungsblatt 6 Aufgabe 4 c)).

Off-policy stochastische Kontrolle mit fester Verhaltenspolicy
(uniform random / epsilon-greedy aus exploration.py).

Updates:
    Q(s, a) <- Q(s, a) + alpha_n(s, a) * (r + gamma * max_a' Q(s', a') - Q(s, a))

Bezug Submission:
    - Blatt 8 Aufgabe 4 d)  Stepsize/Exploration-Schedules
    - Blatt 8 Aufgabe 4 f)  optimale Hyperparameter im 4x4-Grid
"""

# TODO
