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
            avg_len = sum(len(v) for v in present) / n
            if avg_len > 40:
                # free-text column: not a tabular feature. The
                # tabular baseline stays text-blind by design —
                # mining these is the hybrid solver's job.
                continue
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
                uniq = len(set(v.strip().lower()
                               for v in present)) / n
                if n >= 20 and uniq > 0.5:
                    # identifier-like column (names, MRNs):
                    # mostly-unique strings are memorization
                    # bait, not features — live transparency
                    # view caught "patient_name = jules kim"
                    # wearing a risk weight.
                    continue
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


# Introspection: the most recent fit of each built-in solver
# publishes its learned weights here so the bench can show
# "what the model actually found" — feature names aligned to
# the encoder's exact vector order (categorical one-hots
# expanded; mined note terms renamed to their human meaning).
LAST_FIT: Dict[str, Any] = {}


def _feature_names(encoder, miner=None):
    names = []
    for col in encoder.columns:
        if col.kind == "categorical":
            names.extend("{} = {}".format(col.name, c)
                         for c in col.categories)
        elif col.name.startswith("txmine_") and miner \
                and miner.terms:
            idx = int(col.name.split("_")[1])
            term = miner.terms[idx // 2]
            side = ("mentioned" if idx % 2 == 0 else
                    "mentioned but NEGATED (no/denies...)")
            names.append('note phrase "{}" {}'.format(
                term, side))
        else:
            names.append(col.name)
    return names


def _publish_fit(tag, model, miner=None):
    try:
        names = _feature_names(model.encoder, miner)
        if len(names) != len(model.w):
            LAST_FIT[tag] = {"error": "name/weight mismatch"}
            return
        ranked = sorted(zip(names, model.w),
                        key=lambda t: -abs(t[1]))
        LAST_FIT[tag] = {
            "kind": "logistic regression (standard library, "
                    "deterministic, L2-regularized)",
            "n_features": len(names),
            "text_terms": len(miner.terms) if miner else 0,
            "top": [{"name": n, "weight": round(w, 3),
                     "direction": "raises risk" if w > 0
                     else "lowers risk"}
                    for n, w in ranked[:12]
                    if abs(w) > 1e-6]}
    except Exception as e:                 # never break a run
        LAST_FIT[tag] = {"error": str(e)}


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
        _publish_fit("autosolver", model)
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
        if abs(self.vendor_auroc - self.baseline_auroc) < 1e-9:
            return "vendor matches the baseline"
        if self.vendor_auroc > self.baseline_auroc:
            return "vendor beats the baseline"
        return "vendor loses to a stdlib baseline"


@dataclass
class ShowdownResult:
    title: str
    vendor_name: str
    tiers: List[ShowdownTier]
    baseline_name: str = "synthkit baseline"

    def format_text(self) -> str:
        lines = ["SHOWDOWN: {} vs {} — {}".format(
            self.vendor_name, self.baseline_name,
            self.title)]
        for t in self.tiers:
            lines.append(
                "  {:<14} ceiling {:.3f} / baseline {:.3f} / "
                "{} {:.3f}  [bar: {}] {}".format(
                    t.tier, t.ceiling, t.baseline_auroc,
                    self.vendor_name, t.vendor_auroc,
                    "PASS" if t.vendor_passed else "FAIL",
                    t.verdict))
        return "\n".join(lines)


_NEG_TOKENS = {"no", "not", "denies", "denied", "without",
               "negative", "never", "adherent"}
_TOKEN_RE = None


def _tokens(text: str):
    import re as _re
    global _TOKEN_RE
    if _TOKEN_RE is None:
        _TOKEN_RE = _re.compile(r"[a-z]+")
    return _TOKEN_RE.findall(text.lower())


class TextMiner:
    """Learns discriminative note phrases FROM TRAINING DATA —
    no access to the spec. Unigrams + bigrams, minimum document
    frequency, ranked by |label-rate lift| * sqrt(support);
    features are presence bits with a negation window: a term
    within 3 tokens after a negation cue does not count. Built
    for hybrid tables where the outcome signal hides in a note
    column; deterministic."""

    def __init__(self, top_k: int = 40, min_df: int = 5):
        self.top_k = top_k
        self.min_df = min_df
        self.text_cols: List[str] = []
        self.terms: List[str] = []

    @staticmethod
    def _text_columns(rows):
        if not rows:
            return []
        out = []
        for name in rows[0]:
            if name.startswith("_"):
                continue
            vals = [str(r.get(name, "")) for r in rows[:200]]
            if vals and sum(len(v) for v in vals) / len(vals) > 40:
                out.append(name)
        return out

    @staticmethod
    def _grams(toks):
        for t in toks:
            yield t
        for a, b in zip(toks, toks[1:]):
            yield a + " " + b

    @staticmethod
    def _present(term, toks):
        """Returns (affirmed, negated) presence. Negated
        occurrences are a SEPARATE feature, not a suppression:
        clinical risk is often phrased negatively ("no home
        support", "no transportation") and only the training
        labels can say which sign a negated mention carries."""
        parts = term.split(" ")
        n = len(parts)
        affirmed = negated = False
        for i in range(len(toks) - n + 1):
            if toks[i:i + n] == parts:
                lo = max(0, i - 3)
                if any(t in _NEG_TOKENS
                       for t in toks[lo:i]):
                    negated = True
                else:
                    affirmed = True
        return affirmed, negated

    def fit(self, rows, labels) -> "TextMiner":
        self.text_cols = self._text_columns(rows)
        if not self.text_cols:
            return self
        base = sum(1.0 for y in labels if y) / max(len(labels), 1)
        df: Dict[str, int] = {}
        pos: Dict[str, int] = {}
        docs = []
        for r, y in zip(rows, labels):
            toks = []
            for c in self.text_cols:
                toks.extend(_tokens(str(r.get(c, ""))))
            grams = set(self._grams(toks))
            docs.append((toks, y))
            for g in grams:
                df[g] = df.get(g, 0) + 1
                if y:
                    pos[g] = pos.get(g, 0) + 1
        scored = []
        for g, d in df.items():
            if d < self.min_df or d > 0.9 * len(rows):
                continue
            rate = pos.get(g, 0) / d
            lift = abs(rate - base) * (d ** 0.5)
            scored.append((lift, g))
        scored.sort(key=lambda t: (-t[0], t[1]))
        self.terms = [g for _s, g in scored[:self.top_k]]
        return self

    def encode(self, rows) -> List[List[float]]:
        out = []
        for r in rows:
            toks = []
            for c in self.text_cols:
                toks.extend(_tokens(str(r.get(c, ""))))
            feats: List[float] = []
            for t in self.terms:
                aff, neg = self._present(t, toks)
                feats.append(1.0 if aff else 0.0)
                feats.append(1.0 if neg else 0.0)
            out.append(feats)
        return out


def _augment(rows, feats):
    out = []
    for r, f in zip(rows, feats):
        r2 = dict(r)
        for i, v in enumerate(f):
            r2["txmine_{:02d}".format(i)] = \
                "1" if v else "0"
        out.append(r2)
    return out


def autosolver_hybrid():
    """The doctor's own model: tabular features through the
    standard encoder PLUS mined note features injected as
    numeric pseudo-columns. Blind to the spec; everything it
    knows about the text it learned from training labels."""
    def solve(train_rows, train_labels, test_rows):
        miner = TextMiner().fit(train_rows, train_labels)
        aug_train = _augment(train_rows,
                             miner.encode(train_rows))
        aug_test = _augment(test_rows,
                            miner.encode(test_rows))
        model = LogisticBaseline().fit(aug_train, train_labels)
        _publish_fit("autosolver_hybrid", model, miner)
        return model.score(aug_test)
    return solve


def hybrid(train_rows, train_labels, test_rows):
    """Plain-callable form for CLI dotted paths:
    --solver synthkit.autosolver:hybrid"""
    return autosolver_hybrid()(train_rows, train_labels,
                               test_rows)


def run_showdown(campaign, vendor: Callable,
                 vendor_name: str = "vendor",
                 train_seed_offset: int = 1000,
                 baseline: Optional[Callable] = None,
                 baseline_name: str = "synthkit baseline") -> ShowdownResult:
    from .campaign import _judge
    from .tableeval import (evaluate_prediction,
                            resolve_prediction_metric)
    from .tableplan import plan_table
    from .tablespec import TableSpec
    if campaign.goal != "predict":
        raise ValueError("showdowns run on predict campaigns")
    if baseline is None:
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
                          vendor_name=vendor_name,
                          tiers=tiers,
                          baseline_name=baseline_name)
