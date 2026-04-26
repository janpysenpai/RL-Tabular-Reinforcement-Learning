"""Online-Evaluation während des Trainings (Übungsblatt 6, Aufgabe 5).

Muster: Training für m Schritte → Pause → k Episoden auswerten → Resume.

    - evaluate_policy_returns(env, policy, n_eval_episodes, max_steps, gamma, seed)
      → Dict mit avg_return und Episode-Returns-Array

    - online_evaluator(eval_every, eval_fn)
      → OnlineEvaluator (Callable + Logger)

Metriken in den zurückgegebenen Dicts:
    avg_return          Durchschnittlicher diskontierter Return
    returns             Alle Episode-Returns (np.ndarray)
    runtime             Messzeit in Sekunden (wird von OnlineEvaluator ergänzt)
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..envs.mdp_base import FiniteMDP


def evaluate_policy_returns(
    env: FiniteMDP,
    policy: np.ndarray,
    n_eval_episodes: int = 50,
    max_steps: int = 1000,
    gamma: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Wertet eine Policy für n_eval_episodes Episoden aus.

    Parameters
    ----------
    env : FiniteMDP
        Umgebung; wird pro Episode per env.reset() zurückgesetzt.
    policy : np.ndarray
        Stochastische Policy, Shape (S, A).
    n_eval_episodes : int
        Anzahl der Auswertungsepisoden.
    max_steps : int
        Maximale Episodenlänge.
    gamma : float, optional
        Diskontierungsfaktor; None → env.gamma.
    seed : int, optional
        Seed für Reproduzierbarkeit der Aktionsauswahl.

    Returns
    -------
    dict
        "avg_return"  float — Durchschnittlicher diskontierter Return.
        "returns"     np.ndarray — Returns aller n_eval_episodes Episoden.
    """
    gamma = gamma if gamma is not None else env.gamma
    rng = np.random.default_rng(seed)
    returns = np.zeros(n_eval_episodes)

    for i in range(n_eval_episodes):
        s = env.reset()
        g = 0.0
        discount = 1.0
        for _ in range(max_steps):
            if env.is_terminal(s):
                break
            acts = env.allowed_actions[s]
            probs = policy[s, acts]
            probs = probs / probs.sum()
            a = int(rng.choice(acts, p=probs))
            s, r, done = env.step(a)
            g += discount * r
            discount *= gamma
            if done:
                break
        returns[i] = g

    return {"avg_return": float(returns.mean()), "returns": returns}


class OnlineEvaluator:
    """Pause-Evaluate-Resume-Callback (Übungsblatt 6, Aufgabe 5).

    Wird bei jedem Trainingsschritt aufgerufen. Führt eval_fn alle
    eval_every Schritte aus und protokolliert die Ergebnisse.

    Attributes
    ----------
    history : list of (step, metrics)
        Chronologisches Protokoll der Auswertungsergebnisse.
    """

    def __init__(self, eval_every: int, eval_fn: Callable[..., Any]) -> None:
        self.eval_every = eval_every
        self.eval_fn = eval_fn
        self.history: List[Tuple[int, Any]] = []
        self._call_count: int = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Optional[Any]:
        """Inkrementiert Schrittzähler; wertet aus wenn Intervall erreicht.

        Alle Argumente werden an eval_fn weitergegeben.

        Returns
        -------
        Ergebnis von eval_fn, oder None wenn kein Auswertungsschritt.
        """
        self._call_count += 1
        if self._call_count % self.eval_every == 0:
            t0 = time.perf_counter()
            metrics = self.eval_fn(*args, **kwargs)
            runtime = time.perf_counter() - t0
            if isinstance(metrics, dict):
                metrics["runtime"] = runtime
            self.history.append((self._call_count, metrics))
            return metrics
        return None

    def reset(self) -> None:
        """Setzt Schrittzähler und History zurück."""
        self._call_count = 0
        self.history.clear()


def online_evaluator(eval_every: int, eval_fn: Callable[..., Any]) -> OnlineEvaluator:
    """Erstellt einen OnlineEvaluator (Callable + Logger).

    Parameters
    ----------
    eval_every : int
        Trainingsschritte zwischen zwei Auswertungen.
    eval_fn : Callable
        Auswertungsfunktion; erhält dieselben Argumente wie der Evaluator.
        Kann z.B. eine Lambda sein, die über die aktuelle Q/V-Tabelle schließt.

    Returns
    -------
    OnlineEvaluator
        Callable: ``evaluator(step)`` → Optional[metrics].
        Logger:   ``evaluator.history`` → List[(step, metrics)].

    Beispiel
    --------
    >>> eval_fn = lambda: evaluate_policy_returns(env, current_policy)
    >>> ev = online_evaluator(1000, eval_fn)
    >>> for step in range(50_000):
    ...     # Training-Schritt ...
    ...     ev()   # wertet bei Schritten 1000, 2000, ... aus
    """
    return OnlineEvaluator(eval_every, eval_fn)
