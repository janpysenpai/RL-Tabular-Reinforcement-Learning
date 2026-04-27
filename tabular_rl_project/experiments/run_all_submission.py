"""Master-Skript: alle 6 Submission-Tasks nacheinander ausführen.

Startet task_a bis task_f und gibt eine abschließende Zusammenfassung aus.
QUICK_MODE kann via Kommandozeile aktiviert werden:
    python -m tabular_rl_project.experiments.run_all_submission --quick

Alle Plots landen in figures/submission/, alle JSON-Ergebnisse in results/submission/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _set_quick(module_name: str) -> None:
    """Setzt QUICK_MODE=True in einem bereits importierten Modul."""
    import importlib
    mod = sys.modules.get(module_name)
    if mod is not None and hasattr(mod, "QUICK_MODE"):
        mod.QUICK_MODE = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Alle 6 Submission-Tasks ausführen.")
    parser.add_argument("--quick", action="store_true",
                        help="QUICK_MODE: weniger Episoden/Seeds für schnellen Testlauf.")
    parser.add_argument("--tasks", nargs="+", choices=list("abcdef"),
                        default=list("abcdef"),
                        help="Nur bestimmte Tasks ausführen (z.B. --tasks a c f).")
    args = parser.parse_args()

    tasks_to_run = sorted(set(args.tasks))
    print("=" * 60)
    print("Submission-Skripte: Übungsblatt 8 Aufgabe 4")
    print(f"Tasks: {', '.join(tasks_to_run)}")
    if args.quick:
        print("QUICK_MODE aktiviert.")
    print("=" * 60)

    TASK_MODULES = {
        "a": "tabular_rl_project.experiments.submission_tasks.task_a_convergence_rates",
        "b": "tabular_rl_project.experiments.submission_tasks.task_b_finite_vs_discounted",
        "c": "tabular_rl_project.experiments.submission_tasks.task_c_extreme_effects",
        "d": "tabular_rl_project.experiments.submission_tasks.task_d_qlearning_hyperparams",
        "e": "tabular_rl_project.experiments.submission_tasks.task_e_actor_critic_vs_bellman",
        "f": "tabular_rl_project.experiments.submission_tasks.task_f_4x4_gridworld",
    }

    timings = {}
    errors  = {}

    for task in tasks_to_run:
        modname = TASK_MODULES[task]
        print(f"\n{'─'*60}")
        print(f"Starte Task {task.upper()} ...")
        t0 = time.perf_counter()
        try:
            import importlib
            mod = importlib.import_module(modname)
            if args.quick and hasattr(mod, "QUICK_MODE"):
                mod.QUICK_MODE = True
            mod.main()
            elapsed = time.perf_counter() - t0
            timings[task] = elapsed
            print(f"Task {task.upper()} abgeschlossen in {elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            errors[task] = str(exc)
            print(f"FEHLER in Task {task.upper()} nach {elapsed:.1f}s: {exc}", file=sys.stderr)

    # ------ Abschlusszusammenfassung ------
    print(f"\n{'='*60}")
    print("Abschlusszusammenfassung")
    print(f"{'='*60}")
    figures_dir = Path(__file__).parent.parent.parent / "figures" / "submission"
    results_dir = Path(__file__).parent.parent.parent / "results" / "submission"

    for task in tasks_to_run:
        status = "FEHLER" if task in errors else "OK"
        t_str  = f"{timings.get(task, 0):.1f}s" if task in timings else "---"
        print(f"  Task {task.upper()}: {status}  ({t_str})")

    if errors:
        print(f"\n{len(errors)} Task(s) fehlgeschlagen: {list(errors.keys())}")
    else:
        print("\nAlle Tasks erfolgreich.")

    if figures_dir.exists():
        plots = sorted(figures_dir.glob("task_*.png"))
        print(f"\nPlots ({len(plots)}):")
        for p in plots:
            print(f"  {p}")
    if results_dir.exists():
        jsons = sorted(results_dir.glob("task_*.json"))
        print(f"\nJSON-Ergebnisse ({len(jsons)}):")
        for j in jsons:
            print(f"  {j}")


if __name__ == "__main__":
    main()
