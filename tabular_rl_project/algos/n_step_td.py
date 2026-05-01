"""n-step TD Policy Evaluation (Übungsblatt 8 Aufgabe 3 a).

Update startet erst nach n Schritten der MDP-Trajektorie:
    G_t^{(n)} = R_t + gamma R_{t+1} + ... + gamma^{n-1} R_{t+n-1}
              + gamma^n V(S_{t+n})
    V(S_t) <- V(S_t) + alpha * (G_t^{(n)} - V(S_t))
und analog für Q.

Nicht Teil der gewählten Submission-Tasks; Modul ist als Platzhalter vorhanden.
"""
