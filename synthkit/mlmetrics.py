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

from typing import Dict, List, Optional, Sequence


class MetricInputError(ValueError):
    pass


def _check(scores: Sequence[float], labels: Sequence[int]) -> None:
    if len(scores) != len(labels):
        raise MetricInputError(
            "scores and labels must align: {} vs {}".format(
                len(scores), len(labels)))
    # LENGTH, NOT TRUTHINESS. `if not scores` raises
    # "The truth value of an array with more than one element is
    # ambiguous" on a numpy array - and an array is the natural
    # thing to hand a metric. The caller then gets a message
    # about truth values from a function they asked for an
    # AUROC, which costs an hour to trace to a guard that was
    # only ever meant to catch an empty input.
    if len(scores) == 0:
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


def _check_ordinal(scores: Sequence[float],
                   labels: Sequence[int]) -> None:
    """Same contract as `_check` but for an ORDERED label.

    `len(scores) == 0` rather than `not scores`, because a numpy
    array raises on truthiness - the fault this module already
    records, and a caller asking for a concordance would get a
    message about ambiguous truth values from a guard meant to
    catch an empty input.
    """
    if len(scores) == 0 or len(labels) == 0:
        raise MetricInputError("no rows given")
    if len(scores) != len(labels):
        raise MetricInputError(
            "scores and labels differ in length: {} against "
            "{}".format(len(scores), len(labels)))
    for y in labels:
        try:
            iv = int(y)
        except (TypeError, ValueError):
            raise MetricInputError(
                "an ordered label must be a whole number, got "
                "{!r}".format(y))
        if float(y) != float(iv):
            raise MetricInputError(
                "an ordered label must be a whole number, got "
                "{!r}".format(y))


def ordinal_c(scores: Sequence[float],
              labels: Sequence[int]) -> Optional[float]:
    """Concordance between a score and an ORDERED label.

    WHY NOT AUROC. A placement score is asked to separate
    discharge from observation from inpatient - three levels with
    a real order - and collapsing that to one binary question
    throws away the distinction a reviewer cares about most. A
    tool that tells inpatient from everything else perfectly
    while confusing discharge with observation would read as
    flawless.

    So: over every pair of rows whose labels DIFFER, the share
    the score orders correctly, ties counting half. This is the
    pairwise generalisation, and it is computed by WEIGHTING THIS
    MODULE'S OWN `auroc` over each pair of classes rather than by
    a second implementation of the same arithmetic - which also
    makes it exactly AUROC when there are two classes, asserted
    rather than claimed.

    None when fewer than two classes are present: with no
    discordant pair there is nothing to be concordant about, and
    that is an absent measurement rather than a perfect one.
    """
    # `_check` enforces BINARY labels, which is right for auroc
    # and brier and must stay that way - an ordered label needs
    # its own validation rather than a loosened guard that
    # protects two other metrics.
    _check_ordinal(scores, labels)
    lab = [int(v) for v in labels]
    classes = sorted(set(lab))
    if len(classes) < 2:
        return None
    total_w = 0.0
    acc = 0.0
    for ai in range(len(classes)):
        for bi in range(ai + 1, len(classes)):
            a, b = classes[ai], classes[bi]
            sub_s, sub_y = [], []
            for sc, y in zip(scores, lab):
                if y == a:
                    sub_s.append(float(sc))
                    sub_y.append(0)
                elif y == b:
                    sub_s.append(float(sc))
                    sub_y.append(1)
            n_a = sub_y.count(0)
            n_b = sub_y.count(1)
            if not n_a or not n_b:
                continue
            w = float(n_a * n_b)
            acc += w * auroc(sub_s, sub_y)
            total_w += w
    if total_w <= 0:
        return None
    return acc / total_w


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
