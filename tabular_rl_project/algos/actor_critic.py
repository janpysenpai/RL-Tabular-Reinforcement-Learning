"""General Actor-Critic (Skript Algorithm 11, Übungsblatt 7 Aufgabe 3).

Kombiniert ein beliebiges Policy-Evaluation-Schema (Critic) mit einem
Policy-Improvement-Schritt (Actor). Bisherige Algorithmen verwendeten
implizit greedy Improvement; hier soll der Critic frei wählbar sein
(MC, TD(0), n-step TD, Q-Learning, ...).

Ziel des Submission-Tasks e) ist der Vergleich gegen reine
Stochastische-Kontrolle-Algorithmen (Bellman Optimality Operator).
"""

# TODO
