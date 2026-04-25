# CLAUDE.md — Projektkontext für Claude Code

Dieses Repo ist ein Studienprojekt von Jan zur Vorlesung *Reinforcement
Learning* (Prof. Dr. Leif Döring, Universität Mannheim, FSS 2026). Die
Ordnerstruktur unter `tabular_rl_project/` steht bereits — die Aufgabe ist
**Implementation**, nicht Design.

## Was zu liefern ist

**Abgabeaufgabe: Übungsblatt 8, Aufgabe 4** ("Programming task: Tabular
Reinforcement Learning"), Abgabe bis 04.05.2026, Gruppen bis 3 Personen.

Pflicht ist die Implementation der folgenden Algorithmen-Familien:

- Value Iteration und Iterative Policy Evaluation (finite-time und discounted)
- Monte-Carlo Policy Evaluation
- Sample-basierte Policy Evaluation
- Q-Learning und SARSA
- General Actor-Critic

Aus den sechs Subtasks a)–f) müssen vier abgegeben werden. Die Stubs liegen
in `tabular_rl_project/experiments/submission_tasks/` — Jan wird vier davon
auswählen, alle sechs sind aber als Stub vorbereitet.

## Quellmaterial

- `Übungsaufgaben + Skript/RL_Skript.pdf` — Vorlesungsskript. **Relevant sind
  hauptsächlich Kapitel 2 und 3.**
- `Übungsaufgaben + Skript/ub_04_26.pdf` bis `ub_08_26.pdf` — die Übungsblätter,
  auf denen die Aufgabe inhaltlich aufbaut.

Die Algorithmen werden in den Docstrings der Modul-Stubs jeweils per
**Skript-Algorithmus-Nummer** und **Blatt+Aufgabe** referenziert. Bevor du
einen Algorithmus codest, schlag die Referenz im PDF nach (Read-Tool).

## Projektstruktur (steht bereits)

```
tabular_rl_project/
├── envs/        Umgebungen (FiniteMDP-Basisklasse, GridWorld, Multi-Step Bandit, ...)
├── algos/       Modellbasiert + Sample-basiert + Schedules/Exploration
├── utils/       Visualisierung, Online-Evaluation, Bias-Metriken, Plotting
└── experiments/
    ├── demos/             Mini-Runs pro Algo/Env (Sanity-Checks)
    └── submission_tasks/  task_a … task_f (Skripte für die Abgabe)
```

Die Datei-Docstrings beschreiben die Schnittstelle und den Bezug zur
Vorlesung. Lies sie als ersten Schritt.

## Kritische Anforderungen aus den Übungsblättern (nicht übersehen)

- **`FiniteMDP`-Klasse** muss als Attribute haben: `transition_probabilities`,
  `start_state`, `terminal_states`, `allowed_actions`, `expected_rewards`.
  (Übungsblatt 4, Aufgabe 5.)
- **GridWorld** ist konfigurierbar: Größe, Reward-Struktur (det. + stochastisch:
  Normal **und** Binomial), Wand-Verhalten, Wind, Slippery, Random Noise.
- **Modellbasierte Algorithmen** müssen je in V- und Q-Version, jeweils
  synchron **und** totally asynchronous (Algorithm 8) verfügbar sein.
- **Algorithmen** müssen entweder „n Schritte" oder „bis Toleranz" laufen
  können.
- **Schedules** für `α` (Stepsize) und `ε`: konstant + funktional abnehmend
  (mind. `1/n`).
- **Online-Evaluation während Training** (Blatt 6 Aufgabe 5): pause →
  evaluate für k Schritte → resume. Metriken: average score, correct action
  rate, Q(start), Runtime.
- **Bias-Metriken** (Blatt 7 Aufgabe 6): summed total bias, summed squared
  bias, jeweils auch an ausgewählten state-action-Paaren.
- **Submission 4f-Layout** (4×4 GridWorld):
  - F (Fake Goal, R = 0.65) oben links, S (Start) oben rechts
  - G (Goal, R = 1.0) unten, **zweite Spalte von links**
  - SR (Stochastic Region, R ∈ {-2.1, 2} mit p = 1/2 je) **2×2 unten rechts**
  - Default-Reward: -0.05 / 0.05 mit p = 1/2 je
  - γ = 0.9, einmal mit, einmal ohne Random Noise.

Außerdem (Übungsblatt verlangt explizit):

- Python-Version und genutzte Pakete dokumentieren.
- Plots klar beschriftet.
- Verwendete Hyperparameter pro Experiment angeben.

## Arbeitsweise

**Bottom-up.** Reihenfolge der Implementation:

1. `envs/mdp_base.py` → `envs/gridworld.py` (deterministisch zuerst, dann
   Wind/Slip/Noise) → `envs/multi_step_bandit.py`. Pro Umgebung ein Demo
   in `experiments/demos/`.
2. Modellbasierte Algos: `value_iteration.py` → `iterative_policy_evaluation.py`
   → `policy_iteration.py` → `finite_time_dp.py`. Validieren gegen einen
   festen Mini-MDP, für den V\* analytisch bekannt ist.
3. Sample-basierte Algos: `monte_carlo.py` → `td_policy_evaluation.py` →
   `q_learning.py` → `sarsa.py` → `double_q_learning.py` → `actor_critic.py`.
   Ground Truth für Validierung sind die Modellbasierten aus Schritt 2.
4. `cliff_walk.py`, `extreme_mdps.py`, `gridworld_4x4_submission.py`.
5. Submission-Skripte. Wahrscheinlich a, c, d, f — Jan entscheidet.
6. Final Pass: alles auf einmal laufen lassen, Plots in `figures/`,
   README + dieses File aktualisieren falls nötig.

**Verifikation.** Pro implementierter Funktion einen kleinen Sanity-Check:
einfache MDPs, bekannte Lösungen, Vergleich gegen Modellbasierte. Tests
liegen — falls geschrieben — als `test_*.py` neben dem zu testenden Modul.

**Reproduzierbarkeit.** Random-Seed pro Experiment explizit setzen; Ergebnisse
bei Wiederholung übereinstimmen lassen.

## Konventionen

- **Sprache der Docstrings:** Deutsch (kurze, präzise; nicht überladen). Code
  und Variablennamen Englisch.
- **Code-Stil:** Standard PEP 8, Type Hints wo sinnvoll, NumPy-Vektorisierung
  bevorzugt. Keine Fancy-Frameworks (kein Gym/Gymnasium o. Ä.) — alles aus
  NumPy/SciPy/Matplotlib.
- **Plots:** matplotlib, Achsen und Legende immer beschriftet, Titel mit
  Hyperparameter-Hinweis. Speichern unter `figures/<task>_<inhalt>.png`.
- **Imports:** relativ innerhalb des Pakets (`from ..envs.gridworld import GridWorld`).
- **Keine Emojis** im Code oder in den Plots.
- **Keine Bullet-Listen in Antworten an Jan**, wenn Prosa ausreicht.

## Was Jan dir sagen wird

Jan arbeitet bottom-up und in kleinen Häppchen. Erwarte Aufträge der Form:
„implementiere X gemäß Skript-Algorithmus Y, validiere mit Z, erzeug Plot W".
**Nicht** „mach Aufgabe c)" — solche Aufträge sollst du in Subschritte
zerlegen und Rückfragen stellen, bevor du Hyperparameter erfindest.

Bei Hyperparameter-Optimierung (Aufgabe f) keine Werte raten — entweder eine
echte Suche fahren (Grid / Random Search mit mehreren Seeds) oder explizit
nachfragen.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+ empfohlen. Pakete in `requirements.txt`.
