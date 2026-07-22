"""SYNTH_C1: the autosolver — synthkit competes on its own data.

Because synthkit generates unlimited labeled training data, it can
train its OWN baseline and stand it next to any vendor's number:

    ceiling 0.87 / synthkit baseline 0.83 / vendor 0.71

Blinding is structural: the baseline trains on a shifted-seed
table and never sees test rows' labels or the answer key. It is a
deliberately modest model — stdlib logistic regression over
sniffed-and-normalized features — because its job is to be the
FLOOR: if a vendor cannot beat thirty lines of gradient descent
trained in seconds, the ceiling report writes the meeting summary.

Also here: `autoclean`, the heuristic reference cleaner (strip,
case-fold to column convention, unify date formats, de-format
numbers, flag exact duplicates) — the floor for cleaning
campaigns.

Deterministic end to end. Python 3.8 compatible. Stdlib only.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")
_DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%B %d, %Y",
                 "%Y.%m.%d"]
_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}
_MISSING = {"", "null", "n/a", "na", "?", "none"}


def _clean_number(raw: str) -> Optional[float]:
    s = str(raw).strip().replace("$", "").replace(",", "")
    s = s.replace(" . ", ".").replace(" ", "")
    if _NUM_RE.match(s):
        return float(s)
    return None


def _clean_date(raw: str) -> Optional[float]:
    from datetime import datetime
    s = str(raw).strip()
    for fmt in _DATE_FORMATS:
        try:
            return float(datetime.strptime(s, fmt).toordinal())
        except ValueError:
            continue
    return None


def _clean_bool(raw: str) -> Optional[float]:
    s = str(raw).strip().lower()
    if s in _TRUE:
        return 1.0
    if s in _FALSE:
        return 0.0
    return None


@dataclass
class _Column:
    name: str
    kind: str                      # numeric|date|bool|categorical
    mean: float = 0.0
    std: float = 1.0
    categories: List[str] = field(default_factory=list)

    def width(self) -> int:
        return len(self.categories) if self.kind == "categorical" \
            else 1


class FeatureEncoder:
    """Sniffs column types from TRAIN rows only; encodes any rows
    into a fixed-width standardized feature vector. Deterministic;
    ignores columns starting with underscore."""

    def __init__(self, max_categories: int = 8):
        self.max_categories = max_categories
        self.columns: List[_Column] = []

    def fit(self, rows: List[Dict[str, str]]) -> "FeatureEncoder":
        if not rows:
            raise ValueError("cannot fit an encoder on zero rows")
        names = [k for k in rows[0] if not k.startswith("_")]
        for name in names:
            raw = [str(r.get(name, "")) for r in rows]
            present = [v for v in raw
                       if v.strip().lower() not in _MISSING]
            n = max(len(present), 1)
            nums = [_clean_number(v) for v in present]
            dates = [_clean_date(v) for v in present]
            bools = [_clean_bool(v) for v in present]
            num_rate = sum(1 for v in nums if v is not None) / n
            date_rate = sum(1 for v in dates if v is not None) / n
            bool_rate = sum(1 for v in bools if v is not None) / n
            if bool_rate > 0.9:
                kind, vals = "bool", [v for v in bools
                                      if v is not None]
            elif date_rate > 0.6:
                kind, vals = "date", [v for v in dates
                                      if v is not None]
            elif num_rate > 0.6:
                kind, vals = "numeric", [v for v in nums
                                         if v is not None]
            else:
                counts: Dict[str, int] = {}
                for v in present:
                    key = v.strip().lower()
                    counts[key] = counts.get(key, 0) + 1
                cats = [c for c, _cnt in sorted(
                    counts.items(),
                    key=lambda kv: (-kv[1], kv[0]))][
                        :self.max_categories]
                self.columns.append(_Column(
                    name=name, kind="categorical",
                    categories=cats))
                continue
            mean = sum(vals) / len(vals) if vals else 0.0
            var = (sum((v - mean) ** 2 for v in vals) / len(vals)
                   if vals else 0.0)
            self.columns.append(_Column(
                name=name, kind=kind, mean=mean,
                std=math.sqrt(var) or 1.0))
        return self

    def encode(self, row: Dict[str, str]) -> List[float]:
        out: List[float] = []
        for col in self.columns:
            raw = str(row.get(col.name, ""))
            if col.kind == "categorical":
                key = raw.strip().lower()
                out.extend(1.0 if key == c else 0.0
                           for c in col.categories)
                continue
            val = {"numeric": _clean_number,
                   "date": _clean_date,
                   "bool": _clean_bool}[col.kind](raw)
            if val is None:
                val = col.mean          # train-mean imputation
            out.append((val - col.mean) / col.std)
        return out


class LogisticBaseline:
    """Full-batch gradient descent, L2, fixed schedule, zero
    randomness. The floor, not the frontier."""

    def __init__(self, lr: float = 0.5, epochs: int = 300,
                 l2: float = 1e-3):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.encoder = FeatureEncoder()
        self.w: List[float] = []
        self.b = 0.0

    def fit(self, rows: List[Dict[str, str]],
            labels: List[int]) -> "LogisticBaseline":
        self.encoder.fit(rows)
        x = [self.encoder.encode(r) for r in rows]
        y = [1.0 if v else 0.0 for v in labels]
        n, d = len(x), len(x[0]) if x else 0
        self.w = [0.0] * d
        self.b = 0.0
        for _ in range(self.epochs):
            grad_w = [0.0] * d
            grad_b = 0.0
            for xi, yi in zip(x, y):
                z = self.b + sum(w * v
                                 for w, v in zip(self.w, xi))
                p = 1.0 / (1.0 + math.exp(-max(min(z, 30), -30)))
                err = p - yi
                grad_b += err
                for j, v in enumerate(xi):
                    grad_w[j] += err * v
            self.b -= self.lr * grad_b / n
            for j in range(d):
                self.w[j] -= self.lr * (
                    grad_w[j] / n + self.l2 * self.w[j])
        return self

    def score(self, rows: List[Dict[str, str]]) -> List[float]:
        out = []
        for r in rows:
            xi = self.encoder.encode(r)
            z = self.b + sum(w * v for w, v in zip(self.w, xi))
            out.append(1.0 / (1.0 + math.exp(
                -max(min(z, 30), -30))))
        return out


class LinearBaseline:
    """Ridge regression by full-batch gradient descent on the
    same mess-tolerant encoder; labels standardized for training,
    predictions returned in label units. Deterministic."""

    def __init__(self, lr: float = 0.1, epochs: int = 400,
                 l2: float = 1e-3):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.encoder = FeatureEncoder()
        self.w: List[float] = []
        self.b = 0.0
        self.y_mean = 0.0
        self.y_std = 1.0

    def fit(self, rows, labels) -> "LinearBaseline":
        self.encoder.fit(rows)
        x = [self.encoder.encode(r) for r in rows]
        n = len(x)
        self.y_mean = sum(labels) / n
        var = sum((v - self.y_mean) ** 2 for v in labels) / n
        self.y_std = math.sqrt(var) or 1.0
        y = [(v - self.y_mean) / self.y_std for v in labels]
        d = len(x[0]) if x else 0
        self.w = [0.0] * d
        self.b = 0.0
        for _ in range(self.epochs):
            grad_w = [0.0] * d
            grad_b = 0.0
            for xi, yi in zip(x, y):
                err = (self.b + sum(w * v for w, v
                                    in zip(self.w, xi))) - yi
                grad_b += err
                for j, v in enumerate(xi):
                    grad_w[j] += err * v
            self.b -= self.lr * grad_b / n
            for j in range(d):
                self.w[j] -= self.lr * (
                    grad_w[j] / n + self.l2 * self.w[j])
        return self

    def predict(self, rows) -> List[float]:
        out = []
        for r in rows:
            xi = self.encoder.encode(r)
            z = self.b + sum(w * v for w, v in zip(self.w, xi))
            out.append(z * self.y_std + self.y_mean)
        return out


def autosolver_regress(**kwargs) -> Callable:
    """The regress-campaign-contract baseline."""
    def solve(train_rows, train_labels, test_rows):
        model = LinearBaseline(**kwargs)
        model.fit(train_rows, [float(v) for v in train_labels])
        return model.predict(test_rows)
    solve.__name__ = "synthkit-baseline-regress"
    return solve


def autosolver(**kwargs) -> Callable:
    """The campaign-contract solver: train on the blinded train
    split, score the test split."""
    def solve(train_rows, train_labels, test_rows):
        model = LogisticBaseline(**kwargs)
        model.fit(train_rows, train_labels)
        return model.score(test_rows)
    solve.__name__ = "synthkit-baseline"
    return solve


# ===================================================================
# autoclean — the heuristic reference cleaner
# ===================================================================

def autoclean(rows: List[Dict[str, str]]
              ) -> List[Dict[str, str]]:
    """Normalizes what heuristics can honestly normalize:
    whitespace, casing (to each column's dominant convention),
    date formats (to ISO), numeric formatting ($, thousands
    separators, spaced decimals); flags exact duplicate rows.
    Never guesses at missing or wrong values — that honesty keeps
    its overcorrection near zero."""
    from datetime import datetime
    if not rows:
        return []
    names = [k for k in rows[0] if not k.startswith("_")]
    case_mode: Dict[str, str] = {}
    for name in names:
        tally = {"lower": 0, "upper": 0, "title": 0}
        for r in rows:
            v = str(r.get(name, "")).strip()
            if not v or _clean_number(v) is not None:
                continue
            if v == v.lower():
                tally["lower"] += 1
            elif v == v.upper():
                tally["upper"] += 1
            elif v == v.title():
                tally["title"] += 1
        case_mode[name] = max(tally, key=lambda k: tally[k]) \
            if any(tally.values()) else "none"

    out: List[Dict[str, str]] = []
    seen: Dict[Tuple, int] = {}
    for r in rows:
        row: Dict[str, str] = {}
        for name in names:
            v = str(r.get(name, "")).strip()
            if v.strip().lower() in _MISSING:
                row[name] = v
                continue
            d = _clean_date(v)
            if d is not None:
                row[name] = datetime.fromordinal(
                    int(d)).date().isoformat()
                continue
            num = _clean_number(v)
            if num is not None:
                row[name] = ("{:.4f}".format(num)
                             .rstrip("0").rstrip(".")
                             if "." in repr(num) else str(num))
                if num == int(num):
                    row[name] = str(int(num))
                continue
            b = _clean_bool(v)
            if b is not None and v.lower() not in ("true",
                                                   "false"):
                row[name] = "True" if b else "False"
                continue
            mode = case_mode.get(name, "none")
            if mode == "lower":
                row[name] = v.lower()
            elif mode == "upper":
                row[name] = v.upper()
            elif mode == "title":
                row[name] = v.title()
            else:
                row[name] = v
        key = tuple(row[n] for n in names)
        if key in seen:
            row["_duplicate"] = "true"
        seen[key] = seen.get(key, 0) + 1
        out.append(row)
    return out


# ===================================================================
# The showdown
# ===================================================================

@dataclass
class ShowdownTier:
    tier: str
    ceiling: float
    baseline_auroc: float
    vendor_auroc: float
    vendor_passed: bool

    @property
    def verdict(self) -> str:
        if self.vendor_auroc >= self.baseline_auroc:
            return "vendor beats the baseline"
        return "vendor loses to a stdlib baseline"


@dataclass
class ShowdownResult:
    title: str
    vendor_name: str
    tiers: List[ShowdownTier]

    def format_text(self) -> str:
        lines = ["SHOWDOWN: {} vs synthkit baseline — {}".format(
            self.vendor_name, self.title)]
        for t in self.tiers:
            lines.append(
                "  {:<14} ceiling {:.3f} / baseline {:.3f} / "
                "{} {:.3f}  [{}] {}".format(
                    t.tier, t.ceiling, t.baseline_auroc,
                    self.vendor_name, t.vendor_auroc,
                    "PASS" if t.vendor_passed else "FAIL",
                    t.verdict))
        return "\n".join(lines)


def run_showdown(campaign, vendor: Callable,
                 vendor_name: str = "vendor",
                 train_seed_offset: int = 1000) -> ShowdownResult:
    from .campaign import _judge
    from .tableeval import (evaluate_prediction,
                            resolve_prediction_metric)
    from .tableplan import plan_table
    from .tablespec import TableSpec
    if campaign.goal != "predict":
        raise ValueError("showdowns run on predict campaigns")
    baseline = autosolver()
    tiers: List[ShowdownTier] = []
    for tier in campaign.tiers:
        spec = TableSpec.from_json(tier.spec_json)
        train_spec = TableSpec.from_json(tier.spec_json)
        train_spec.master_seed += train_seed_offset
        train_bp = plan_table(train_spec)
        test_bp = plan_table(spec)
        outcome = campaign.outcome
        n_train = len(train_bp.clean_rows)
        n_test = len(test_bp.clean_rows)
        train_rows = [
            {k: v for k, v in row.items() if k != outcome}
            for row in train_bp.dirty_rows[:n_train]]
        train_labels = [
            1 if train_bp.clean_rows[r][outcome] == "True" else 0
            for r in range(n_train)]
        test_rows = [
            {k: v for k, v in row.items() if k != outcome}
            for row in test_bp.dirty_rows[:n_test]]
        v_scores = list(vendor(train_rows, train_labels,
                               test_rows))
        b_scores = baseline(train_rows, train_labels, test_rows)
        v_rep = evaluate_prediction(test_bp, outcome, v_scores,
                                    vendor_name)
        b_rep = evaluate_prediction(test_bp, outcome, b_scores,
                                    "synthkit-baseline")
        tr = _judge(tier.conditions,
                    lambda m: resolve_prediction_metric(v_rep, m))
        tiers.append(ShowdownTier(
            tier=tier.name,
            ceiling=round(v_rep.ceiling_auroc, 4),
            baseline_auroc=round(b_rep.auroc, 4),
            vendor_auroc=round(v_rep.auroc, 4),
            vendor_passed=tr.passed,
        ))
    return ShowdownResult(title=campaign.title,
                          vendor_name=vendor_name, tiers=tiers)
