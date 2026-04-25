"""SARSA (Skript Algorithm 20, Übungsblatt 7 Aufgabe 2).

On-policy Variante von TD-Kontrolle:
    Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s', a') - Q(s,a))
mit a' ~ pi(. | s'), wobei pi epsilon-greedy bzgl. Q ist (konstant
oder mit Schedule).
"""

# TODO
