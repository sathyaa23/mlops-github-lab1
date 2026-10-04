"""Small data-analytics utility functions.

This module replaces the original `calculator.py` from the MLOps
Github_Labs/Lab1. It provides descriptive statistics, feature scaling,
IQR-based outlier detection, and a tiny CSV loader so the lab's testing
(pytest + unittest) and CI (GitHub Actions) demonstrate something closer
to real data-engineering work than arithmetic helpers.
"""
from __future__ import annotations

import csv
from typing import List, Sequence


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean() requires at least one value")
    return sum(values) / len(values)


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median() requires at least one value")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2


def variance(values: Sequence[float], sample: bool = True) -> float:
    n = len(values)
    if n < 2:
        raise ValueError("variance() requires at least two values")
    mu = mean(values)
    ss = sum((x - mu) ** 2 for x in values)
    return ss / (n - 1) if sample else ss / n


def std(values: Sequence[float], sample: bool = True) -> float:
    return variance(values, sample) ** 0.5


def min_max_scale(values: Sequence[float]) -> List[float]:
    if not values:
        raise ValueError("min_max_scale() requires at least one value")
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0 for _ in values]
    return [(x - lo) / (hi - lo) for x in values]


def z_score_normalize(values: Sequence[float]) -> List[float]:
    mu = mean(values)
    sigma = std(values)
    if sigma == 0:
        return [0.0 for _ in values]
    return [(x - mu) / sigma for x in values]


def _percentile(sorted_vals: Sequence[float], pct: float) -> float:
    if not sorted_vals:
        raise ValueError("percentile of empty sequence")
    k = (len(sorted_vals) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def detect_outliers_iqr(values: Sequence[float]) -> List[float]:
    """Return values outside 1.5*IQR of Q1/Q3 (standard Tukey fences)."""
    if len(values) < 4:
        return []
    s = sorted(values)
    q1 = _percentile(s, 25)
    q3 = _percentile(s, 75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return [x for x in values if x < lo or x > hi]


def summary(values: Sequence[float]) -> dict:
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "std": std(values),
        "min": min(values),
        "max": max(values),
        "outliers": detect_outliers_iqr(values),
    }


def load_column(path: str, column: str) -> List[float]:
    """Load one numeric column from a CSV file as a list of floats."""
    out: List[float] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if column not in (reader.fieldnames or []):
            raise KeyError(f"column {column!r} not found in {path}")
        for row in reader:
            out.append(float(row[column]))
    return out
