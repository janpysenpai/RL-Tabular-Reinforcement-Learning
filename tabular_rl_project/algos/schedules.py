"""Stepsize- und Exploration-Schedules (Übungsblatt 6, Aufgabe 4 Helper).

Wird sowohl von TD-/Q-Learning-/SARSA-Algorithmen für alpha_n
als auch von eps-greedy für epsilon_n verwendet.

Mindestens:
    - constant(c)
    - one_over_n()                   alpha_n = 1/n
    - polynomial(rate)               alpha_n = 1 / n**rate
    - linear_decay(start, end, T)
"""

# TODO: Schedule-Funktionen / kleine Klasse implementieren.
