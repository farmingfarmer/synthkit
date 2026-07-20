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
