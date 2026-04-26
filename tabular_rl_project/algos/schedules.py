"""Schrittweiten- und Explorations-Schedules (Übungsblatt 6, Aufgabe 4).

Alle Schedules sind Callables der Signatur ``(n: int) -> float``,
wobei n der Besuchszähler (visit count) des jeweiligen Zustands oder
(s,a)-Paars ist. n >= 1 wird vorausgesetzt.

    - constant(alpha)              α_n = alpha (konstant)
    - one_over_n()                 α_n = 1/n   (Robbins-Monro erfüllt)
    - polynomial(rate)             α_n = 1/n^rate
    - linear_decay(start, end, T)  linear von start auf end über T Schritte

Schedule = Callable[[int], float]
"""

from __future__ import annotations

from typing import Callable

Schedule = Callable[[int], float]


def constant(alpha: float) -> Schedule:
    """Konstante Schrittweite α_n = alpha für alle n.

    Parameters
    ----------
    alpha : float
        Schrittweite, typischerweise ∈ (0, 1).

    Returns
    -------
    Schedule
        Callable ``(n: int) -> float``.
    """
    def _schedule(n: int) -> float:
        return alpha
    return _schedule


def one_over_n() -> Schedule:
    """Harmonische Schrittweite α_n = 1/n.

    Erfüllt die Robbins-Monro-Bedingungen (``Σ α_n = ∞``, ``Σ α_n² < ∞``),
    die fast sichere Konvergenz von TD(0) und MC garantieren.

    Returns
    -------
    Schedule
        Callable ``(n: int) -> float``.
    """
    def _schedule(n: int) -> float:
        return 1.0 / max(n, 1)
    return _schedule


def polynomial(rate: float) -> Schedule:
    """Polynomielle Schrittweite α_n = 1/n^rate.

    Robbins-Monro erfüllt für rate ∈ (0.5, 1]; für rate=1 identisch
    mit ``one_over_n()``.

    Parameters
    ----------
    rate : float
        Abfallexponent > 0 (empfohlen: 0.5 < rate ≤ 1).

    Returns
    -------
    Schedule
        Callable ``(n: int) -> float``.
    """
    def _schedule(n: int) -> float:
        return 1.0 / max(n, 1) ** rate
    return _schedule


def linear_decay(start: float, end: float, T: int) -> Schedule:
    """Lineare Abnahme von start (n=1) auf end (n=T), danach konstant bei end.

    α_n = start + (end - start) · min(n-1, T-1) / (T-1)  für T > 1.

    Mehrdeutigkeit: Für T ≤ 1 wird sofort end zurückgegeben
    (kein sinnvoller Interpolationsbereich).

    Parameters
    ----------
    start : float
        Anfangswert bei n=1.
    end : float
        Endwert ab n=T.
    T : int
        Schrittanzahl bis zum Endwert (inklusive).

    Returns
    -------
    Schedule
        Callable ``(n: int) -> float``.
    """
    if T <= 1:
        def _schedule_flat(n: int) -> float:
            return end
        return _schedule_flat

    def _schedule(n: int) -> float:
        t = min(n - 1, T - 1)
        return start + (end - start) * t / (T - 1)
    return _schedule
