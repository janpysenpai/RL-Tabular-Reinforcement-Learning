"""Submission f)  Optimale Hyperparameter für Q-/Double-Q auf dem 4x4-Grid.

Umgebung: envs.gridworld_4x4_submission.build_submission_gridworld(noise=...)

    F . . S       F = Fake Goal (R = 0.65)
    . . . .       S = Start
    . . SR SR     G = Goal (R = 1.0), zweite Spalte unten
    . G SR SR     SR = Stochastic Region (R = -2.1 oder 2 mit p = 1/2 je)
                  Default-Reward: -0.05 / 0.05 mit p = 1/2 je
                  gamma = 0.9
                  einmal mit, einmal ohne Random Noise

Vorgehen für die Hyperparameter-Optimierung dokumentieren:
    - Suchraum (alpha-Schedule, eps-Schedule, behaviour policy, ...)
    - Verfahren (Grid Search / Bandit-artig / Bayesian).
    - Bewertungsmetrik (Episodic Return / Korrektheitsrate / Q(start)).
    - Random-Seeds, Anzahl Wiederholungen.
"""

# TODO
if __name__ == "__main__":
    pass
