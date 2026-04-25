# RL-Tabular-Reinforcement-Learning

Abgabeprojekt zu **Übungsblatt 8, Aufgabe 4** ("Tabular Reinforcement Learning")
der Vorlesung *Reinforcement Learning* (Prof. Dr. Leif Döring, Universität
Mannheim, FSS 2026). Aufbauend auf den Programmieraufgaben aus den
Übungsblättern 4–7. Theorie-Referenzen beziehen sich auf das mitgelieferte
`RL_Skript.pdf`, hauptsächlich Kapitel 2 und 3.

## Projektstruktur

```
tabular_rl_project/
├── envs/              Umgebungen (MDPs)
│   ├── mdp_base.py                    Abstrakte Finite-MDP-Klasse
│   ├── gridworld.py                   Konfigurierbare m×n Grid World (Blatt 4)
│   ├── multi_step_bandit.py           Multi-Step Bandit (Blatt 4)
│   ├── gridworld_4x4_submission.py    4×4-Layout aus Blatt 8 f)
│   ├── cliff_walk.py                  Windy Cliff Walk (Blatt 7)
│   └── extreme_mdps.py                "Extreme" MDPs für Blatt 8 c)
│
├── algos/             Algorithmen
│   ├── schedules.py                   α- und ε-Schedules
│   ├── exploration.py                 ε-greedy etc.
│   ├── value_iteration.py             Skript Algorithm 6
│   ├── iterative_policy_evaluation.py Skript Algorithm 7 (sync) + 8 (async)
│   ├── policy_iteration.py            Skript Algorithm 10 (Greedy AC)
│   ├── finite_time_dp.py              Skript Algorithms 12 & 13
│   ├── monte_carlo.py                 First-/Every-Visit MC
│   ├── td_policy_evaluation.py        Skript Algorithms 17 & 18
│   ├── q_learning.py
│   ├── double_q_learning.py
│   ├── sarsa.py                       Skript Algorithm 20
│   ├── n_step_td.py                   Blatt 8 Aufgabe 3 a)
│   ├── n_step_sarsa.py                Blatt 8 Aufgabe 3 b)
│   └── actor_critic.py                General Actor-Critic (Algorithm 11)
│
├── utils/             Visualisierung, Online-Evaluation, Bias-Metriken, Plotting
│
└── experiments/
    ├── demos/                         Mini-Runs pro Algo/Env
    └── submission_tasks/              Skripte für die Abgabe-Subtasks
        ├── task_a_convergence_rates.py
        ├── task_b_finite_vs_discounted.py
        ├── task_c_extreme_effects.py
        ├── task_d_qlearning_hyperparams.py
        ├── task_e_actor_critic_vs_bellman.py
        └── task_f_4x4_gridworld.py
```

Auf Top-Level liegen außerdem:

- `Übungsaufgaben + Skript/` — Original-PDFs (Skript + Übungsblätter).
- `figures/` — generierte Plots.
- `notebooks/` — optional für explorative Auswertung.
- `requirements.txt`, `.gitignore`.

## Abgabe-Aufgabe (Blatt 8, Aufgabe 4)

Vier von sechs Subtasks (a–f) sind verpflichtend. Die zu deckenden Algorithmen
sind:

- Value Iteration und Iterative Policy Evaluation (finite-time und discounted)
- Monte-Carlo Policy Evaluation
- Sample-basierte Policy Evaluation
- Q-Learning und SARSA
- General Actor-Critic

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+ empfohlen. Verwendete Pakete und Hyperparameter werden in den
einzelnen Submission-Skripten dokumentiert (Anforderung des Übungsblatts).
