"""Multi-Step Bandit (Übungsblatt 4, Aufgabe 5b).

    - Startzustand s0 mit k Branches.
    - Branch b_i besteht aus m_i deterministisch durchlaufenen Schritten.
    - Pro Schritt s_{b_i, j} stehen k_{b_i, j} Aktionen zur Wahl.
    - Default-Reward; pro Aktion lassen sich custom (deterministische
      oder zufällige) Rewards setzen.

Wird in den Submission-Tasks für die Exploration- und Bias-Vergleiche
mit Q-Learning / SARSA / Double-Q wiederverwendet.
"""

# TODO: MultiStepBandit(FiniteMDP) implementieren.
