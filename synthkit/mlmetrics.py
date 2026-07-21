"""SYNTH_B1: ML metrics — stdlib, exact, ceiling-aware.

AUROC via the rank formulation (Mann-Whitney) with average ranks
for ties; Brier score; accuracy/precision/recall at a threshold.

The synthkit-only capability: because outcomes are GENERATED from a
known model, every row has a true probability — so ceiling metrics
(the score of the Bayes-optimal classifier that knows the true
probabilities) are computable for any dataset synthkit makes. A
vendor's 0.71 means something entirely different against a ceiling
of 0.74 than against 0.92.

Python 3.8 compatible. Stdlib only.
"""
from __future__ import annotations

from typing import Dict, List, Sequence


class MetricInputError(ValueError):
    pass


def _check(scores: Sequence[float], labels: Sequence[int]) -> None:
    if len(scores) != len(labels):
        raise MetricInputError(
            "scores and labels must align: {} vs {}".format(
                len(scores), len(labels)))
    if not scores:
        raise MetricInputError("empty inputs")
    for y in labels:
        if y not in (0, 1, True, False):
            raise MetricInputError(
                "labels must be binary, got {!r}".format(y))


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Probability a random positive outscores a random negative
    (ties count half). Rank-based; O(n log n)."""
    _check(scores, labels)
    n_pos = sum(1 for y in labels if y)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise MetricInputError(
            "AUROC needs both classes present ({} pos, {} neg)"
            .format(n_pos, n_neg))
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and \
                scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels) if y)
    u = rank_sum - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def brier(probs: Sequence[float], labels: Sequence[int]) -> float:
    _check(probs, labels)
    return sum((p - (1.0 if y else 0.0)) ** 2
               for p, y in zip(probs, labels)) / len(probs)


def at_threshold(scores: Sequence[float], labels: Sequence[int],
                 threshold: float = 0.5) -> Dict[str, float]:
    _check(scores, labels)
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels):
        pred = s >= threshold
        if pred and y:
            tp += 1
        elif pred:
            fp += 1
        elif y:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {
        "accuracy": (tp + tn) / len(scores),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def ceiling_auroc(true_probs: Sequence[float],
                  labels: Sequence[int]) -> float:
    """The Bayes-optimal AUROC on this realized dataset: the score
    of a classifier that outputs the TRUE generating probability
    for every row. No real model can beat it except by luck."""
    return auroc(true_probs, labels)


# ===================================================================
# Uncertainty — every rate is k-of-n, and verdicts should know it.
# A live bake-off flipped a bar-edge verdict between identical
# runs because 0.105 on n=19 (CI roughly [0.03, 0.29]) never
# contained enough information to resolve a bar of 0.10.
# ===================================================================

import math as _math


def wilson_interval(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion. Behaves
    sanely at the edges (k=0, k=n) where the normal approximation
    lies."""
    if n <= 0:
        return (0.0, 1.0)
    if not (0 <= k <= n):
        raise MetricInputError(
            "k must be within [0, n], got k={} n={}".format(k, n))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * _math.sqrt(
        p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def format_rate(k: int, n: int, z: float = 1.96) -> str:
    lo, hi = wilson_interval(k, n, z)
    p = k / n if n else 0.0
    return "{:.3f} [{:.3f}, {:.3f}] n={}".format(p, lo, hi, n)


def required_n(p_obs: float, bar: float, z: float = 1.96):
    """Approximate sample size at which a Wilson-style interval
    around p_obs would exclude the bar — the PRESCRIPTION: how
    much more data resolves an inconclusive verdict. None when
    the observed rate sits on the bar (no n resolves a tie)."""
    gap = abs(p_obs - bar)
    if gap < 1e-9:
        return None
    p = min(max(p_obs, 1e-6), 1 - 1e-6)
    n = (z * z * p * (1 - p)) / (gap * gap)
    return int(_math.ceil(n))


def auroc_interval(a: float, n_pos: int, n_neg: int,
                   z: float = 1.96):
    """Hanley-McNeil confidence interval for AUROC."""
    if n_pos <= 0 or n_neg <= 0:
        return (0.0, 1.0)
    a = min(max(a, 1e-6), 1 - 1e-6)
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a)
           + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    half = z * _math.sqrt(max(var, 0.0))
    return (max(0.0, a - half), min(1.0, a + half))
