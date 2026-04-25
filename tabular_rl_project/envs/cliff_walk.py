"""Windy Cliff Walk (Übungsblatt 7, Aufgabe 5).

12 x 4 Grid:
    - Start oben links, Goal oben rechts (Reward 100).
    - Obere Reihe (zwischen Start und Goal) sind Cliff-Felder (Reward -100).
    - Default-Reward -1.
    - Wind: mit Wahrscheinlichkeit p_wind nach oben (Richtung Cliff).

Wird im Submission-Task c) (Robust RL) eingesetzt, um SARSA vs. Q-Learning
zu vergleichen.
"""

# TODO: build_cliff_walk(p_wind: float) -> GridWorld
