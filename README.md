# Tabular Reinforcement Learning

Implementierung tabellarischer RL-Algorithmen in Python — von modellbasierten
Dynamic-Programming-Verfahren bis zu sample-basierten Methoden wie Q-Learning,
SARSA und Actor-Critic. Entwickelt im Rahmen der Vorlesung *Reinforcement
Learning* (Prof. Dr. Leif Döring, Universität Mannheim, FSS 2026),
Übungsblatt 8 Aufgabe 4. Theoretische Grundlage: Vorlesungsskript Kapitel 2 & 3.

Alle Algorithmen laufen auf einer eigens implementierten `FiniteMDP`-Basisklasse
ohne externe RL-Frameworks — ausschließlich NumPy und Matplotlib.

---

## Algorithmen

| Familie | Implementierung |
|---|---|
| Dynamic Programming | Value Iteration, Iterative Policy Evaluation (sync + async), Policy Iteration, Finite-Time DP |
| Monte-Carlo | First-Visit (Alg. 15) und Every-Visit (Alg. 14) Policy Evaluation |
| Temporal Difference | TD(0), Q-Learning, Double Q-Learning, SARSA |
| Actor-Critic | General Actor-Critic |

---

## Projektstruktur

```
tabular_rl_project/
├── envs/                   Umgebungen (FiniteMDP-Basisklasse, GridWorld,
│                           Multi-Step Bandit, Cliff Walk, Extreme MDPs,
│                           4×4-Submission-Grid mit stochastischen Regionen)
├── algos/                  Algorithmen (DP, MC, TD, Q-Learning, SARSA,
│                           Actor-Critic) + Schedules + Exploration
├── utils/                  Online-Evaluation (Pause-Evaluate-Resume)
└── experiments/
    ├── demos/              Sanity-Checks: ein Demo-Skript pro Algo/Env
    ├── submission_tasks/   Vollständige Experiment-Skripte task_a – task_f
    └── run_all_submission.py   Master-Skript: alle Tasks nacheinander

figures/submission/         Abgabe-Plots (task_a – task_f)
results/submission/         Numerische Ergebnisse als JSON (task_a – task_f)
tests/                      Pytest-Tests (98 Tests)
```

---

## Installation

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11 oder neuer empfohlen.

---

## Tests ausführen

```bash
pytest tests/
```

Erwartete Ausgabe: **98 Tests grün**, Laufzeit ca. 30 Sekunden.

---

## Demos ausführen

```bash
python -m tabular_rl_project.experiments.demos.run_gridworld
python -m tabular_rl_project.experiments.demos.run_value_iteration
python -m tabular_rl_project.experiments.demos.run_multi_step_bandit
python -m tabular_rl_project.experiments.demos.run_monte_carlo
python -m tabular_rl_project.experiments.demos.run_q_learning
python -m tabular_rl_project.experiments.demos.run_sarsa
python -m tabular_rl_project.experiments.demos.run_cliff_walk
python -m tabular_rl_project.experiments.demos.run_actor_critic
python -m tabular_rl_project.experiments.demos.run_extreme_mdps
python -m tabular_rl_project.experiments.demos.run_submission_grid
```

Plots landen in `figures/`.

---

## Submission reproduzieren

```bash
python -m tabular_rl_project.experiments.run_all_submission
```

Führt alle sechs Submission-Tasks (a–f) nacheinander aus.
Erwartete Laufzeit: ca. 50 Sekunden.

Für einen schnellen Testlauf mit reduzierten Episoden/Seeds:

```bash
python -m tabular_rl_project.experiments.run_all_submission --quick
```

Erwartete Laufzeit im Quick-Modus: ca. 28 Sekunden.

Einzelne Tasks können mit `--tasks a c f` ausgewählt werden.

**Output-Pfade:**

| Inhalt | Pfad |
|---|---|
| Plots (PNG) | `figures/submission/task_X_*.png` |
| Numerische Ergebnisse | `results/submission/task_X.json` |

---

## Hinweis zur KI-Unterstützung

Die technische Implementierung wurde mit Unterstützung von Claude (Anthropic)
durchgeführt. Konzeption der Experimente, Auswahl der Algorithmen und
Hyperparameter, inhaltliche Validierung der Ergebnisse sowie alle
Submission-Texte stammen vom Autor.

---

## Autor

Jan Salama und Malte Schröter, Universität Mannheim, FSS 2026
