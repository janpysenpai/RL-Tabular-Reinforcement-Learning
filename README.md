# Tabular Reinforcement Learning

Implementation zentraler tabellarischer RL-Algorithmen in Python — von modellbasierten
Dynamic-Programming-Verfahren bis hin zu sample-basierten Methoden wie Q-Learning,
SARSA und Actor-Critic. Entwickelt im Rahmen der Vorlesung *Reinforcement Learning*
an der Universität Mannheim (FSS 2026).

## Überblick

Das Projekt deckt die folgenden Algorithmen-Familien ab:

- **Dynamic Programming:** Value Iteration, Iterative Policy Evaluation (synchron & asynchron), Policy Iteration, Finite-Time DP
- **Monte-Carlo:** First-Visit und Every-Visit Policy Evaluation
- **Temporal Difference:** TD(0), Q-Learning, Double Q-Learning, SARSA, n-Step TD, n-Step SARSA
- **Actor-Critic:** General Actor-Critic

Alle Algorithmen laufen auf einer eigens implementierten `FiniteMDP`-Basisklasse ohne
externe RL-Frameworks — ausschließlich NumPy, SciPy und Matplotlib.

## Projektstruktur

```
tabular_rl_project/
├── envs/              Umgebungen (MDPs)
│   ├── mdp_base.py                    Abstrakte Finite-MDP-Basisklasse
│   ├── gridworld.py                   Konfigurierbare m×n Grid World
│   ├── multi_step_bandit.py           Multi-Step Bandit
│   ├── gridworld_4x4_submission.py    4×4-Szenario mit stochastischen Regionen
│   ├── cliff_walk.py                  Windy Cliff Walk
│   └── extreme_mdps.py                MDPs für Backpropagation / Overestimation
│
├── algos/             Algorithmen
│   ├── schedules.py                   α- und ε-Schedules (konstant, 1/n, ...)
│   ├── exploration.py                 Explorations-Policies (ε-greedy, ...)
│   ├── value_iteration.py
│   ├── iterative_policy_evaluation.py synchron + asynchron
│   ├── policy_iteration.py
│   ├── finite_time_dp.py
│   ├── monte_carlo.py
│   ├── td_policy_evaluation.py
│   ├── q_learning.py
│   ├── double_q_learning.py
│   ├── sarsa.py
│   ├── n_step_td.py
│   ├── n_step_sarsa.py
│   └── actor_critic.py
│
├── utils/             Visualisierung, Online-Evaluation, Bias-Metriken, Plotting
│
└── experiments/
    ├── demos/                         Sanity-Checks pro Algorithmus / Umgebung
    └── submission_tasks/              Vollständige Experiment-Skripte
        ├── task_a_convergence_rates.py
        ├── task_b_finite_vs_discounted.py
        ├── task_c_extreme_effects.py
        ├── task_d_qlearning_hyperparams.py
        ├── task_e_actor_critic_vs_bellman.py
        └── task_f_4x4_gridworld.py
```

Generierte Plots landen in `figures/`, optionale Auswertungs-Notebooks in `notebooks/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+ empfohlen.

## Demo ausführen

```bash
python -m tabular_rl_project.experiments.demos.run_gridworld
```
