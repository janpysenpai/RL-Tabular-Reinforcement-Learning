"""Spezifische 4x4 Grid World für die Abgabe (Blatt 8, Aufgabe 4f).

Layout:
    F . . S      F = Fake Goal  (oben links,  Reward 0.65)
    . . . .      S = Start       (oben rechts)
    . . SR SR    G = Goal        (unten, zweite Spalte v. links, Reward 1.0)
    . G SR SR    SR = Stochastic Region 2x2 unten rechts
                      (Reward -2.1 / 2 mit gleicher Wahrscheinlichkeit)

Default-Reward: -0.05 / 0.05 mit gleicher Wahrscheinlichkeit.
Discount:       gamma = 0.9
Mit und ohne Random Noise zu untersuchen.

Wird als vorkonfigurierte Instanz von GridWorld bereitgestellt.
"""

# TODO: build_submission_gridworld(noise: float = 0.0) -> GridWorld
