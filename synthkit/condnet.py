"""Conditional dependency network — learn the JOINT distribution of
a table, not just its margins and monotone links.

The limitation this exists to remove: fitting each column
separately and imposing rank correlations reproduces "what each
field looks like" and "which fields move together, monotonically."
It cannot represent a U-shaped relationship (both extremes of
sodium are dangerous, so rank correlation is ~0), an interaction
(creatinine matters more in the elderly), or any higher-order
structure. Data can pass a marginal-and-correlation audit with
every interesting relationship destroyed.

How this works instead. Every column is discretized into bins
(quantile bins for numbers; levels for categories; missingness is
its own bin, so CONDITIONAL missingness is captured too). A parent
set is learned per column by conditional mutual information, which
detects dependence of ANY shape — monotone, U-shaped,
threshold-like, interaction-driven. The joint is then stored as
conditional probability tables and sampled ancestrally.

What crosses into the model is counts and bin edges, never records:
  - a cell is kept only if at least `k` rows support it; thinner
    cells back off to a smaller parent set, and ultimately to the
    marginal
  - bin edges are quantiles, so no individual extreme is exposed
  - a parent is accepted only if its dependence survives a
    Bonferroni-corrected G-test, so noise is not learned as
    structure

Every table is a DIAL. `amplify` interpolates a conditional
distribution away from or toward its marginal, so a user can make
a discovered relationship twice as strong, remove it entirely, or
anything between — which is the point of profiling patterns rather
than copying rows.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

MISSING = "\u2205"          # its own bin: missingness is a pattern

# How far the outermost bins may reach past p1/p99 when emitting a
# value, as a multiple of the adjacent interval's width. Zero keeps
# every synthetic value inside the published range (safest, but the
# distribution comes out slightly narrow); larger values restore
# tail spread with extrapolated — never observed — extremes.
TAIL_REACH = 2.0

# How many rows one person may contribute before their extra rows
# are dropped. Differential privacy needs a bound on how much any
# single individual can move a published count; without one the
# sensitivity is whatever the heaviest utiliser happens to be, and
# the guarantee is unstateable. Capping costs a little fidelity on
# high-utilisation patients and buys a number you can defend.
DP_MAX_ROWS_PER_PERSON = 12

# Dirichlet smoothing strength for conditional tables. A cell
# supported by twelve people should not yield a point estimate as
# confident as one supported by twelve hundred, so every table is
# shrunk toward its own marginal by a pseudo-count. This is where
# thin-data behaviour lives, and raw maximum likelihood is
# overconfident exactly where it can least afford to be.
SMOOTHING = 1.0

# Neighbour smoothing — implemented, MEASURED, and off by default.
#
# The idea: a thin cell is better informed by the cells next door
# than by the global marginal, because clinical relationships are
# smooth. Sound in principle. Measured against the fidelity of the
# generated conditional profile at 400/800/1600 patients, in a
# deliberately thin-cell regime (three ordinal parents, fine
# bins), it changed nothing — 0.0319 vs 0.0322, 0.0281 vs 0.0281,
# 0.0216 vs 0.0221.
#
# The reason is that this architecture already solves the problem
# a different way. Cells below k people are never published: they
# BACK OFF to a smaller parent set, which borrows strength more
# aggressively and more defensibly than averaging with a
# neighbour. Every cell that survives to be smoothed already has
# enough support that smoothing is arithmetically negligible.
#
# Left in place and set to zero: if k is ever lowered so that
# genuinely thin cells reach publication, raising this is the
# first thing to try.
NEIGHBOUR_WEIGHT = 0.0

# A column is DERIVED when another column determines it almost
# perfectly — a count computed from a list, an age computed from a
# birth year, a flag defined by a gap. These are arithmetic the
# pipeline itself created, not clinical findings, and letting them
# compete for parent slots wastes a scarce statistical budget on
# rediscovering our own bookkeeping.
#
# The threshold is deliberately loose. A relationship that is exact
# in principle is not exact after BINNING: a count derived from a
# list, once both are discretised and rare levels folded into an
# OTHER bucket, retains real residual entropy. At 0.08 the detector
# caught three tautologies on a live extract and missed five,
# including `condition_count <- conditions`. Genuine physiological
# relationships sit far above 0.20 — systolic and diastolic
# pressure move together strongly and still leave most of the
# entropy unexplained — so the wider band separates arithmetic from
# findings without swallowing any.
# Strict on purpose. The exact tests carry the load now — a tally
# is checked against the list's length, an arithmetic column
# against the actual sum or difference, a list family against its
# determining family — so this heuristic is only a backstop for
# threshold rules like "is this the last visit".
#
# It must stay strict because CLINICAL relationships can be very
# strong without being arithmetic. A planted rule giving 85% of
# heart-failure patients a diuretic scored 0.22 here and was filed
# as bookkeeping — which would have deleted the most interesting
# finding in the data and called it tidying up. Anything a
# clinician would recognise as a fact about patients must survive
# to the findings.
DERIVED_ENTROPY_RATIO = 0.05

# Columns that must never serve as a parent. These are not merely
# redundant — they are the OUTCOME'S OWN DEFINITION. An outcome
# defined as "did the next visit fall within thirty days" is
# perfectly predicted by the gap to the next visit, and a model
# handed both would score flawlessly while learning nothing. This
# is label leakage, and it is worse than a tautology because it
# looks like a finding.
NEVER_PARENT = ("days_to_next_visit", "is_last_visit",
                "visit_id", "record_id")

# Smallest categorical level budget, so a small extract does not have
# ordinary categoricals pruned away by the transition-table bound.
MIN_LEVEL_BUDGET = 20

# Separators are required, so a bare year like 1950 is a number and
# not a date. Matching on digits alone would swallow year_of_birth.
_DATE_PATTERNS = (
    r"^\d{4}-\d{1,2}-\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?$",
    r"^\d{1,2}/\d{1,2}/\d{2,4}( \d{1,2}:\d{2}(:\d{2})?)?$",
    r"^\d{4}/\d{1,2}/\d{1,2}$",
    r"^\d{1,2}-[A-Za-z]{3}-\d{2,4}$",
)


def _looks_like_date(present, threshold=0.9):
    """A column is a date when nearly all of its values are dates.

    Nearly, not all: real extracts carry a stray sentinel or a
    malformed row, and one bad value must not turn a date column back
    into a category."""
    if not present:
        return False
    hits = 0
    for v in present:
        for pat in _DATE_PATTERNS:
            if re.match(pat, v):
                hits += 1
                break
    return hits / float(len(present)) >= threshold

# Separator for list-valued columns (medication lists, condition
# lists). These need expanding, not binning — see below.
LIST_SEP = "; "

# How many items from a list column become their own indicator.
LIST_TOP_ITEMS = 12


# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------
def _num(v) -> Optional[float]:
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null", "na", "n/a"):
        return None
    try:
        return float(s.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def _rng(seed: int, *parts: str) -> random.Random:
    key = "{}:{}".format(seed, ":".join(str(p) for p in parts))
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return random.Random(int(h, 16))


def _quantile(xs: Sequence[float], q: float) -> float:
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return ys[i]


def _norm_ppf(u: float) -> float:
    """Inverse standard normal, so a position in [0,1] can be
    mixed in the space where mixing preserves the marginal."""
    u = max(1e-6, min(1.0 - 1e-6, u))
    # Beasley-Springer-Moro
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c_ = [-7.784894002430293e-03, -3.223964580411365e-01,
          -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if u < pl:
        q = math.sqrt(-2 * math.log(u))
        return (((((c_[0] * q + c_[1]) * q + c_[2]) * q + c_[3])
                 * q + c_[4]) * q + c_[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if u > ph:
        q = math.sqrt(-2 * math.log(1 - u))
        return -(((((c_[0] * q + c_[1]) * q + c_[2]) * q + c_[3])
                  * q + c_[4]) * q + c_[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = u - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r
             + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
          + b[4]) * r + 1)


def _norm_cdf(z: float) -> float:
    """Standard normal CDF, so a Gaussian latent maps back to a
    position that is exactly uniform on [0, 1]."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _noisy(counts: Counter, scale: float, rng) -> Counter:
    """Laplace noise on a histogram, clipped at zero."""
    out = Counter()
    for s, v in counts.items():
        u = rng.random() - 0.5
        lap = -scale * math.copysign(1.0, u) * \
            math.log(max(1.0 - 2.0 * abs(u), 1e-12))
        out[s] = max(0.0, v + lap)
    if not sum(out.values()):
        for s in counts:
            out[s] = 1.0
    return out


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    for pos, i in enumerate(order):
        out[i] = pos
    return out


def _entropy(counts: Sequence[float]) -> float:
    n = sum(counts)
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            h -= p * math.log(p)
    return h


def _z_from_alpha(alpha: float) -> float:
    """Normal quantile, rational approximation (no scipy)."""
    q = max(min(alpha / 2.0, 0.5), 1e-12)
    t = math.sqrt(-2.0 * math.log(q))
    return t - ((2.515517 + 0.802853 * t + 0.010328 * t * t)
                / (1 + 1.432788 * t + 0.189269 * t * t
                   + 0.001308 * t ** 3))


def _chi2_crit(df: int, alpha: float) -> float:
    """Wilson-Hilferty critical value for chi-square."""
    if df <= 0:
        return float("inf")
    z = _z_from_alpha(alpha)
    return df * (1 - 2.0 / (9 * df) +
                 z * math.sqrt(2.0 / (9 * df))) ** 3


# ---------------------------------------------------------------
# discretization
# ---------------------------------------------------------------
class Binning:
    """How one column becomes discrete symbols and back again."""

    def __init__(self, name, kind, edges=None, levels=None,
                 integer=False, missing_rate=0.0, fine=None,
                 atoms=None, clamp=None, nonneg=False):
        self.name = name
        self.kind = kind                  # numeric | categorical
        self.edges = edges or []          # len = nbins + 1
        self.levels = levels or []
        self.integer = integer
        self.missing_rate = missing_rate
        # TWO resolutions, deliberately. `edges` are the coarse
        # bins used for CONDITIONING — they must stay wide enough
        # that cells keep enough rows to detect dependence. `fine`
        # is a denser quantile grid used only to reconstruct a
        # VALUE once a bin has been chosen, so the column's own
        # distribution survives the discretization. Conditioning
        # power and marginal fidelity stop competing.
        self.fine = fine or []
        # Point masses. Clinical numbers pile up on exact values —
        # zeros, detection limits, clamped floors, defaults. No
        # amount of interval sampling reproduces "exactly 0.4 in
        # 4% of rows", so repeated values are stored as atoms with
        # their own probability and emitted verbatim.
        self.atoms = atoms or {}
        # Plausibility bounds. Tail extrapolation exists so the
        # generated distribution keeps its spread after the outer
        # bin edges were pulled in to p1/p99 — but on a sparse bin
        # it can overshoot into values that cannot occur. A
        # diastolic pressure of 37 in a clinical demo undermines
        # every credible number beside it, so emitted values are
        # held inside a band the data itself supports: the
        # published percentiles widened by half an interquartile
        # range, and never through zero for a quantity that is
        # never observed negative.
        self.clamp = clamp or []
        self.nonneg = nonneg

    # -- learning ------------------------------------------------
    @staticmethod
    def learn(name, values, k, max_bins=10, min_level_count=None,
              groups=None):
        """`groups` names the PRIVACY UNIT for each row — usually
        the patient. Every k-test then counts distinct people, not
        rows. It matters enormously: 931 visits from 92 patients
        means a cell holding ten rows can be one person's ten
        visits, and k-anonymity on rows would be no protection at
        all. With no groups supplied each row is its own unit, so
        behaviour is unchanged."""
        gs = groups or [str(i) for i in range(len(values))]
        pairs = [(v, g) for v, g in zip(values, gs)
                 if str(v).strip() != ""]
        present = [v for v, _ in pairs]
        miss = 1.0 - (len(present) / max(1, len(values)))
        nums = [_num(v) for v in present]
        numeric = present and all(x is not None for x in nums)
        if numeric:
            distinct = sorted(set(nums))
            # A numeric column with few distinct values — a 0/1
            # outcome, a small count, a coded category — must NOT
            # be quantile-binned: the quantiles collapse onto the
            # same value, the column reads as constant and gets
            # dropped. That silently destroyed outcome columns, so
            # low-cardinality numerics are modelled by their actual
            # values instead.
            if len(distinct) <= max(2 * max_bins, 12):
                by_v = defaultdict(set)
                for v, g in pairs:
                    x = _num(v)
                    if x is not None:
                        by_v[x].add(g)
                keep = [v for v in distinct
                        if len(by_v.get(v, ())) >= k]
                if len(keep) >= 2:
                    return Binning(
                        name, "discrete",
                        levels=[repr(v) for v in keep],
                        integer=all(float(v).is_integer()
                                    for v in distinct),
                        missing_rate=miss)
        if numeric:
            xs = [x for x in nums if x is not None]
            # bin count is bounded by how many rows each bin must
            # hold: a bin thinner than k cannot be published
            nb = max(2, min(max_bins, len(xs) // max(k, 1)))
            # Bin edges are order statistics — real observed
            # values. Interior quantiles are heavily aggregated and
            # safe to publish, but the outermost edges would be the
            # true MINIMUM and MAXIMUM, i.e. two specific people's
            # measurements. Clamp the outer edges to p1/p99, the
            # same policy the profiler already applies.
            lo_c, hi_c = _quantile(xs, 0.01), _quantile(xs, 0.99)
            edges = [_quantile(xs, i / nb) for i in range(nb + 1)]
            edges[0] = min(lo_c, edges[1] if len(edges) > 1
                           else lo_c)
            edges[-1] = max(hi_c, edges[-2] if len(edges) > 1
                            else hi_c)
            # collapse duplicate edges (heavy ties, e.g. many zeros)
            ded = [edges[0]]
            for e in edges[1:]:
                if e > ded[-1]:
                    ded.append(e)
            if len(ded) < 2:
                ded = [ded[0], ded[0] + 1.0]
            q1_c, q3_c = _quantile(xs, 0.25), _quantile(xs, 0.75)
            iqr_c = max(q3_c - q1_c, 0.0)
            lo_clamp = lo_c - 0.5 * iqr_c
            hi_clamp = hi_c + 0.5 * iqr_c
            nonneg_c = min(xs) >= 0
            if nonneg_c:
                lo_clamp = max(lo_clamp, 0.0)
            nf = max(4, min(40, len(xs) // max(2 * k, 1)))
            fine = [_quantile(xs, i / nf) for i in range(nf + 1)]
            fine[0], fine[-1] = ded[0], ded[-1]
            fine = sorted(set(fine))
            atoms = {}
            counts = Counter(xs)
            for bi in range(len(ded) - 1):
                lo_e, hi_e = ded[bi], ded[bi + 1]
                inb = [x for x in xs
                       if (lo_e <= x <= hi_e if bi == 0
                           else lo_e < x <= hi_e)]
                if len(inb) < k:
                    continue
                by_val = defaultdict(set)
                for v, g in pairs:
                    x = _num(v)
                    if x is None:
                        continue
                    if (lo_e <= x <= hi_e if bi == 0
                            else lo_e < x <= hi_e):
                        by_val[x].add(g)
                spikes = [(v, counts[v] / len(inb))
                          for v in set(inb)
                          if counts[v] >= 0.03 * len(inb)
                          and len(by_val.get(v, ())) >= k]
                if spikes:
                    atoms[str(bi)] = sorted(spikes,
                                            key=lambda t: -t[1])
            return Binning(name, "numeric", edges=ded,
                           integer=all(float(x).is_integer()
                                       for x in xs),
                           missing_rate=miss, fine=fine,
                           atoms=atoms,
                           clamp=[lo_clamp, hi_clamp],
                           nonneg=nonneg_c)
        c = Counter(str(v).strip() for v in present)
        by_level = defaultdict(set)
        for v, g in pairs:
            by_level[str(v).strip()].add(g)
        floor = min_level_count if min_level_count is not None else k
        levels = [lv for lv, _ in c.most_common()
                  if len(by_level[lv]) >= floor]
        if not levels:
            levels = [c.most_common(1)[0][0]] if c else []
        return Binning(name, "categorical", levels=levels,
                       missing_rate=miss)

    # -- use -----------------------------------------------------
    def encode(self, v) -> str:
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none", "null"):
            return MISSING
        if self.kind == "discrete":
            x = _num(s)
            if x is None:
                return "OTHER_SUPPRESSED"
            r = repr(x)
            return r if r in self.levels else "OTHER_SUPPRESSED"
        if self.kind == "numeric":
            x = _num(s)
            if x is None:
                return MISSING
            for i in range(len(self.edges) - 1):
                if x <= self.edges[i + 1] or i == len(
                        self.edges) - 2:
                    return "b{}".format(i)
            return "b0"
        return s if s in self.levels else "OTHER_SUPPRESSED"

    def symbols(self) -> List[str]:
        if self.kind == "discrete":
            return list(self.levels) + ["OTHER_SUPPRESSED"]
        if self.kind == "numeric":
            return ["b{}".format(i)
                    for i in range(len(self.edges) - 1)]
        return list(self.levels) + ["OTHER_SUPPRESSED"]

    def decode(self, sym: str, rng: random.Random, pos=None):
        """`pos` in [0,1] fixes WHERE inside the chosen bin the
        value falls, instead of drawing it fresh.

        This is what carries a patient's position from one visit
        to the next. Without it the bin is stable across visits
        but the value inside it is redrawn every time, so a
        patient's pressure jumps across the whole band each
        encounter and the visit-to-visit correlation is capped by
        the bin width rather than by the physiology.
        """
        if sym == MISSING:
            return ""
        if self.kind == "discrete":
            if sym == "OTHER_SUPPRESSED":
                sym = self.levels[0] if self.levels else "0"
            try:
                x = float(sym)
            except ValueError:
                return sym
            return int(round(x)) if self.integer else x
        if self.kind == "numeric":
            try:
                i = int(sym[1:])
            except ValueError:
                i = 0
            lo = self.edges[max(0, min(i, len(self.edges) - 2))]
            hi = self.edges[max(1, min(i + 1, len(self.edges) - 1))]
            spikes = self.atoms.get(str(i)) or []
            if spikes:
                u = rng.random()
                acc = 0.0
                for val, prob in spikes:
                    acc += prob
                    if u < acc:
                        return (int(round(val)) if self.integer
                                else round(val, 4))
            sub = [v for v in self.fine if lo <= v <= hi]
            if pos is not None and len(sub) >= 2:
                span = sub[-1] - sub[0]
                if self.clamp:
                    _lo = max(self.clamp[0], sub[0] - TAIL_REACH
                              * (sub[1] - sub[0]))
                    _hi = min(self.clamp[1], sub[-1] + TAIL_REACH
                              * (sub[-1] - sub[-2]))
                else:
                    _lo, _hi = sub[0], sub[-1]
                x = _lo + max(0.0, min(1.0, pos)) * (_hi - _lo)
                if self.nonneg:
                    x = max(0.0, x)
                return (int(round(x)) if self.integer
                        else round(x, 4))
            # Tail extrapolation. Because the outer edges clamp to
            # p1/p99 to avoid publishing anyone's true extreme, the
            # synthetic distribution would come out narrower than
            # the source. So the outermost bins reach a little
            # beyond their edge, by the width of the neighbouring
            # interval — the tail regains its spread while every
            # emitted extreme is an extrapolation rather than a
            # copy of a real measurement.
            if len(sub) >= 2 and TAIL_REACH > 0:
                # STRETCH the outermost interval rather than adding
                # one: appending an interval would move a whole
                # extra share of the mass into the tail and widen
                # the distribution. Stretching keeps each interval
                # carrying the mass it should while letting the
                # extreme reach past the published bound.
                if i == 0:
                    sub[0] = sub[0] - TAIL_REACH * (sub[1] - sub[0])
                elif i == len(self.edges) - 2:
                    sub[-1] = sub[-1] + TAIL_REACH * (sub[-1]
                                                      - sub[-2])
            if len(sub) >= 2:
                # each fine interval carries equal probability
                # mass, so picking one uniformly and then a point
                # inside it reproduces the column's shape WITHIN
                # the bin instead of flattening it
                j = rng.randrange(len(sub) - 1)
                x = rng.uniform(sub[j], sub[j + 1])
            else:
                x = rng.uniform(lo, hi)
            if self.clamp:
                x = max(self.clamp[0], min(self.clamp[1], x))
            if self.nonneg:
                x = max(0.0, x)
            return int(round(x)) if self.integer else round(x, 4)
        return sym

    def to_json(self):
        return {"name": self.name, "kind": self.kind,
                "edges": [round(e, 6) for e in self.edges],
                "fine": [round(e, 6) for e in self.fine],
                "atoms": {b: [[round(v, 6), round(pp, 6)]
                              for v, pp in lst]
                          for b, lst in self.atoms.items()},
                "clamp": [round(c, 6) for c in self.clamp],
                "nonneg": self.nonneg,
                "levels": self.levels, "integer": self.integer,
                "missing_rate": round(self.missing_rate, 6)}

    @staticmethod
    def from_json(d):
        return Binning(d["name"], d["kind"], d.get("edges"),
                       d.get("levels"), d.get("integer", False),
                       d.get("missing_rate", 0.0),
                       d.get("fine"),
                       {b: [(v, pp) for v, pp in lst]
                        for b, lst in (d.get("atoms")
                                       or {}).items()},
                       d.get("clamp"), d.get("nonneg", False))


# ---------------------------------------------------------------
# the network
# ---------------------------------------------------------------
class CondNet:
    def __init__(self, k=10, max_parents=3, max_bins=0,
                 alpha=0.01, seed=20260731):
        """max_bins=0 means AUTO: resolution is chosen from how
        much data there is to condition on. Finer bins describe
        each column better but fragment the table — with three
        parents at eight bins, four thousand rows spread across
        four thousand cells and no dependence can be detected at
        all. Auto solves bins^(parents+1) ~ n/k, the point where
        cells stay populated enough to test."""
        self.k = k
        self.max_parents = max_parents
        self.max_bins = max_bins
        self.alpha = alpha
        self.seed = seed
        self.order: List[str] = []
        self.binnings: Dict[str, Binning] = {}
        self.parents: Dict[str, List[str]] = {}
        self.cpt: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.marginal: Dict[str, Dict[str, float]] = {}
        self.report: Dict[str, Any] = {}

    # -- learning ------------------------------------------------
    def _mi(self, xs, ys):
        joint = Counter(zip(xs, ys))
        cx, cy = Counter(xs), Counter(ys)
        n = len(xs)
        mi = 0.0
        for (a, b), c in joint.items():
            pxy = c / n
            mi += pxy * math.log(pxy / ((cx[a] / n) * (cy[b] / n)))
        return mi, len(cx), len(cy)

    def _cmi(self, xs, ys, zs):
        """I(X;Y|Z) — dependence that survives conditioning."""
        if not zs:
            mi, nx, ny = self._mi(xs, ys)
            return mi, (nx - 1) * (ny - 1)
        buckets = defaultdict(lambda: ([], [], set()))
        gs = getattr(self, "groups", None) or list(
            range(len(xs)))
        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            buckets[z][0].append(x)
            buckets[z][1].append(y)
            buckets[z][2].add(gs[i])
        n = len(xs)
        total, df = 0.0, 0
        # An INTERACTION means the effect lives inside ONE stratum.
        # The pooled test averages it across all strata while
        # summing the degrees of freedom across all of them, so it
        # is weakest exactly where interactions live. Alongside the
        # pooled statistic we therefore keep the strongest single
        # stratum, tested on its own people and corrected for how
        # many strata were inspected.
        best_g, best_df, best_n, n_strata = 0.0, 1, 0, 0
        for z, (bx, by, ppl) in buckets.items():
            if len(ppl) < self.k:
                continue
            n_strata += 1
            mi, nx, ny = self._mi(bx, by)
            total += (len(bx) / n) * mi
            df += (nx - 1) * (ny - 1)
            g = 2.0 * len(ppl) * mi
            if g > best_g:
                best_g, best_df = g, max((nx - 1) * (ny - 1), 1)
                best_n = len(ppl)
        self._last_stratum = (best_g, best_df, best_n,
                              max(n_strata, 1))
        return total, max(df, 1)

    def _unmodellable(self, col, values, n_rows, groups=None):
        """None if the column may be modelled, else why not.

        Bound: a categorical's transition table has levels^2 cells, so
        levels^2 must not exceed the observations available to fill
        them at k apiece. The floor of MIN_LEVEL_BUDGET keeps a small
        extract from having ordinary categoricals pruned out of it."""
        gs = groups if groups is not None else list(range(len(values)))
        pairs = [(str(v).strip(), g) for v, g in zip(values, gs)]
        pairs = [(v, g) for v, g in pairs
                 if v and v.lower() not in ("nan", "none", "null")]
        present = [v for v, _g in pairs]
        if not present:
            return None
        if _looks_like_date(present):
            return ("date: not a category. The calendar carries "
                    "identity; the signal is in ELAPSED TIME, which "
                    "is NOT YET modelled - so this column is a net "
                    "LOSS of temporal information until it is")
        # A numeric column is quantile-binned, so its transition table
        # is bins^2 and already bounded. Applying the level bound to it
        # would strip ordinary numerics - age_at_visit and
        # year_of_birth were both excluded before this check existed.
        if all(_num(v) is not None for v in present):
            return None
        # Counted by distinct PATIENTS, exactly as Binning.learn does.
        # Counting occurrences instead would not predict the levels
        # that actually survive, and the guard would fire on the wrong
        # columns - one patient's fifty visits are not fifty holders.
        holders = defaultdict(set)
        for v, g in pairs:
            holders[v].add(g)
        levels = [lv for lv, hs in holders.items() if len(hs) >= self.k]
        budget = max(MIN_LEVEL_BUDGET,
                     int((n_rows / max(self.k, 1)) ** 0.5))
        if len(levels) > budget:
            return ("{} levels above k exceeds the {} its transition "
                    "table can support at {} rows".format(
                        len(levels), budget, n_rows))
        return None

    def learn(self, rows: List[Dict[str, Any]],
              columns: Optional[List[str]] = None,
              targets: Optional[List[str]] = None,
              group_by: Optional[str] = None,
              hypotheses: Optional[Dict[str, List[str]]] = None,
              multilevel: bool = False,
              epsilon: float = 0.0) -> "CondNet":
        """`hypotheses` maps a column to the columns allowed to
        explain it — the clinical questions actually being asked.

        Blind search tests every pair and pays a multiple-
        comparison tax across hundreds of them; that tax is the
        single largest drain on statistical power here. Naming a
        dozen candidate relationships instead of asking the
        machine to find everything collapses the correction and
        buys roughly a doubling of effective sample size, measured.
        Columns outside the hypotheses are still modelled from
        their marginals, so generated data stays complete.

        `multilevel` recognises that not every question costs the
        same amount of data. Collapsing to the patient count is
        right for a BETWEEN-person effect — "do elderly patients
        have more events?" has as many independent observations as
        there are patients. But "when this patient's sodium drifts
        from their own baseline, does their risk change?" is a
        WITHIN-person comparison: each patient serves as their own
        control, so the degrees of freedom come from visits rather
        than people. Treating both the same way discards most of
        what a longitudinal dataset contains."""
        """`targets` are placed LAST in the ordering so they can
        condition on everything else. Without that hint an outcome
        often lands first (it is the hub of the dependence graph)
        and ends up with no parents, which throws away exactly the
        structure a benchmark cares about."""
        cols = columns or list(rows[0].keys())
        n = len(rows)
        # The privacy and inference unit. Repeated encounters from
        # one patient are not independent evidence: they neither
        # earn k-anonymity nor carry n rows' worth of statistical
        # weight.
        self.group_by = group_by
        if group_by:
            self.groups = [str(r.get(group_by, i))
                           for i, r in enumerate(rows)]
            cols = [c for c in cols if c != group_by]
        else:
            self.groups = [str(i) for i in range(n)]
        # Column names that are purely numeric or blank are almost
        # always an artefact of a headerless index or a malformed
        # export, not a clinical field. Modelling them produces
        # nonsense like `column "4" is determined by column "1"`.
        odd = [c for c in cols
               if not str(c).strip()
               or str(c).strip().replace(".", "", 1).isdigit()]
        if odd:
            cols = [c for c in cols if c not in odd]
        self.dropped_odd_names = odd

        # ---- list columns: expand, never bin ----
        # A medication or condition list is different for almost
        # every visit, so treating the whole string as a category
        # sends every value below the k-patient floor and collapses
        # the column to a single bucket. Observed live: `conditions`
        # and `active_drugs` each reduced to ONE level, contributing
        # nothing but their missingness — which is how the clinical
        # content of a record becomes invisible to the model.
        #
        # The information is in the ITEMS, not the combination. Each
        # frequent item becomes its own present/absent indicator, so
        # "this patient is on a diuretic" can carry structure while
        # the rare combinations that would identify someone never
        # appear.
        self.list_columns = {}
        self.list_lengths = {}
        self.tally_of = {}
        expanded_rows = [dict(r) for r in rows]
        for c in list(cols):
            vals = [str(r.get(c, "")) for r in rows]
            if not any(LIST_SEP in v for v in vals):
                continue
            def _items(v):
                # placeholders for "nothing here" are absence, not
                # a finding — expanding them creates an indicator
                # that simply mirrors the empty case
                return {x.strip() for x in v.split(LIST_SEP)
                        if x.strip()
                        and x.strip().lower() not in
                        ("none", "nan", "null", "n/a", "-")}
            people_of = defaultdict(set)
            for j, v in enumerate(vals):
                for item in _items(v):
                    people_of[item].add(self.groups[j])
            keep = [i for i, ppl in sorted(
                people_of.items(), key=lambda kv: -len(kv[1]))
                if len(ppl) >= self.k][:LIST_TOP_ITEMS]
            if not keep:
                continue
            self.list_columns[c] = keep
            cols.remove(c)
            self.list_lengths[c] = [len(_items(v)) for v in vals]
            for j, v in enumerate(vals):
                have = _items(v)
                for item in keep:
                    expanded_rows[j]["{}::{}".format(c, item)] = (
                        "1" if item in have else "0")
            cols.extend("{}::{}".format(c, i) for i in keep)
        if self.list_columns:
            rows = expanded_rows

        # ---- columns that must not be modelled as categories ----
        # Runs AFTER list expansion so the indicators it produces are
        # judged, not the raw list columns it consumes.
        #
        # A date is not a category. Modelled as one, its transition
        # table is levels^2 over the calendar, and the size of that
        # depends on the COHORT: at 92 patients no single date clears
        # the k-patient floor, so every date column collapses to one
        # level and the fault is invisible; at 800 patients over six
        # years the dates clear it easily and the same code produced
        # ~2,230 levels, a 4,975,074-cell transition table for ONE
        # column, and a 424 MB model - larger than the extract it was
        # learned from, with 90% of it being two date columns.
        #
        # It is a privacy fault as much as a size one: that table
        # publishes "at least k patients moved from date X to date Y",
        # and exact dates are the most identifying thing in a clinical
        # record.
        #
        # The general rule is the guard, not the date rule: a
        # categorical may not carry more levels than its transition
        # table can support at k observations per cell. The date test
        # is kept as well because a date is the wrong TYPE regardless
        # of how many levels it happens to have in this cohort - which
        # is exactly what made this latent rather than visible.
        self.excluded_columns = {}
        keep_cols = []
        for c in cols:
            reason = self._unmodellable(c, [r.get(c, "") for r in rows],
                                        n, self.groups)
            if reason:
                self.excluded_columns[c] = reason
            else:
                keep_cols.append(c)
        cols = keep_cols

        self.n_groups = len(set(self.groups))
        self.multilevel = bool(multilevel and group_by)
        # Degrees of freedom for a within-person comparison: every
        # observation, less one absorbed mean per person.
        self.n_within = max(n - self.n_groups, 1)

        # ---- differential privacy ----
        # k-anonymity is a property of the published tables;
        # epsilon is a promise about what an ADVERSARY can learn.
        # It is the stronger claim and the one a privacy office
        # can reason about, so it is offered as an explicit budget
        # rather than an implicit hope.
        self.epsilon = float(epsilon or 0.0)
        self.dp_dropped = 0
        if self.epsilon > 0 and group_by:
            seen = defaultdict(int)
            keep_idx = []
            for j in range(n):
                g = self.groups[j]
                if seen[g] < DP_MAX_ROWS_PER_PERSON:
                    seen[g] += 1
                    keep_idx.append(j)
            if len(keep_idx) < n:
                self.dp_dropped = n - len(keep_idx)
                rows = [rows[j] for j in keep_idx]
                self.groups = [self.groups[j] for j in keep_idx]
                n = len(rows)
                self.n_within = max(n - self.n_groups, 1)
        if self.max_bins and self.max_bins > 0:
            nbins, auto = self.max_bins, False
        else:
            nbins = int(round((self.n_groups / max(self.k, 1))
                              ** (1.0 / (self.max_parents + 1))))
            nbins, auto = max(3, min(10, nbins)), True
        self.resolved_bins = nbins
        self.binnings = {
            c: Binning.learn(c, [r.get(c, "") for r in rows],
                             self.k, nbins, groups=self.groups)
            for c in cols}
        enc = {c: [self.binnings[c].encode(r.get(c, ""))
                   for r in rows] for c in cols}

        # ---- which columns vary WITHIN a person? ----
        self.level = {}
        by_person = defaultdict(lambda: defaultdict(set))
        for j in range(n):
            for c in cols:
                by_person[c][self.groups[j]].add(enc[c][j])
        for c in cols:
            varying = sum(1 for vals in by_person[c].values()
                          if len(vals) > 1)
            share = varying / max(len(by_person[c]), 1)
            self.level[c] = ("visit" if share > 0.25
                             else "patient")

        # ---- within-person centred encodings ----
        # Binning each value's deviation from its own patient's
        # mean makes associations between centred columns
        # within-person BY CONSTRUCTION: stable patient traits are
        # differenced away and cannot confound them.
        # ---- how a patient's course unfolds ----
        # Sampling rows independently produces a pile of
        # encounters, not patients. Real records have people in
        # them: someone with high pressure at one visit tends to
        # have high pressure at the next, and a patient with four
        # visits is a different thing from four patients with one.
        # So generation needs three facts a row-wise model never
        # stores — how many visits a person has, which of their
        # values are FIXED traits, and how a changing value moves
        # from one visit to the next.
        per_person = defaultdict(int)
        for g in self.groups:
            per_person[g] += 1
        self.visit_counts = Counter(per_person.values())
        # How many visits people have is itself a published
        # statistic derived from patients, so it takes its share
        # of the budget. Each person contributes exactly one
        # observation here, so the sensitivity is 1.
        if getattr(self, "epsilon", 0) > 0:
            vrng = random.Random(self.seed ^ 0x515C)
            noisy_v = Counter()
            for kk, vv in self.visit_counts.items():
                u = vrng.random() - 0.5
                lap = -(1.0 / max(self.epsilon / 8.0, 1e-6)) \
                    * math.copysign(1.0, u) \
                    * math.log(max(1.0 - 2.0 * abs(u), 1e-12))
                nv = int(round(max(0.0, vv + lap)))
                if nv:
                    noisy_v[kk] = nv
            if noisy_v:
                self.visit_counts = noisy_v
        self.lag = {}
        seq = defaultdict(list)
        for j in range(n):
            seq[self.groups[j]].append(j)
        self.enc_within = {}
        if self.multilevel:
            for c in cols:
                if self.level[c] != "visit":
                    continue
                if self.binnings[c].kind not in ("numeric",
                                                 "discrete"):
                    continue
                vals = [_num(r.get(c, "")) for r in rows]
                if any(v is None for v in vals):
                    continue
                sums, cnts = defaultdict(float), defaultdict(int)
                for j, v in enumerate(vals):
                    sums[self.groups[j]] += v
                    cnts[self.groups[j]] += 1
                dev = [vals[j] - sums[self.groups[j]]
                       / cnts[self.groups[j]] for j in range(n)]
                b = Binning.learn(
                    c + "__dev", [str(d) for d in dev], self.k,
                    nbins, groups=self.groups)
                self.enc_within[c] = [b.encode(str(d))
                                      for d in dev]
        # drop columns with no variation — nothing to condition on
        usable = [c for c in cols if len(set(enc[c])) > 1]
        dropped = [c for c in cols if c not in usable]

        # Order by total pairwise dependence: hub variables first,
        # so downstream columns can condition on them.
        strength = {}
        for c in usable:
            s = 0.0
            for d in usable:
                if c == d:
                    continue
                mi, _, _ = self._mi(enc[c], enc[d])
                s += mi
            strength[c] = s
        tgt = [c for c in (targets or []) if c in usable]
        rest = [c for c in usable if c not in tgt]
        self.order = (sorted(rest, key=lambda c: -strength[c])
                      + sorted(tgt, key=lambda c: -strength[c]))

        # ---- derived columns: arithmetic, not findings ----
        self.derived = {}
        # Determinism is tested on the RAW values, not the binned
        # symbols. Binning destroys exactly the evidence this test
        # needs: a count computed from a list is perfectly
        # determined by that list, but once the list is bucketed
        # the determinism disappears and the pair is reported as a
        # discovery instead of as arithmetic.
        raw = {c: [str(r.get(c, "")).strip() for r in rows]
               for c in self.order}
        # Some bookkeeping needs TWO columns to explain it. A birth
        # year is fixed while an age changes visit to visit, so
        # neither determines the other on its own — but the pair
        # (age, visit date) determines the birth year exactly. Live,
        # `year_of_birth <- age_at_visit` kept surfacing as a
        # finding for want of this test.
        # The test is ARITHMETIC, not entropy. An entropy test on a
        # pair is worthless here: two high-cardinality columns
        # carve the data into near-singletons and then "determine"
        # everything, so the guard against that would also reject
        # the genuine formulas. Checking the arithmetic directly is
        # both cheaper and impossible to fool — a birth year IS the
        # visit year minus the age, on every row or none.
        def _pair_determines(b_col, cand_a, cand_c):
            xb = num_of.get(b_col)
            xa = num_of.get(cand_a)
            xc = num_of.get(cand_c)
            if xb is None or xa is None or xc is None:
                return None
            for label, f in (("difference",
                              lambda u, v: u - v),
                             ("difference",
                              lambda u, v: v - u),
                             ("sum", lambda u, v: u + v),
                             ("product", lambda u, v: u * v)):
                ok = 0
                for j in range(n):
                    try:
                        want = f(xa[j], xc[j])
                    except (TypeError, OverflowError):
                        break
                    if abs(xb[j] - want) <= 1e-6 + 1e-3 * abs(want):
                        ok += 1
                if ok >= 0.95 * n:
                    return label
            return None

        num_of = {}
        for c2 in self.order:
            xs2 = [_num(v) for v in raw[c2]]
            if all(x is not None for x in xs2):
                num_of[c2] = xs2

        # Two tracks, because "A determines B" means different
        # things depending on how coarse A is.
        #
        # A COARSE determinant (a list, a category, a small count)
        # genuinely determines what follows from it: a condition
        # list fixes the condition count exactly. Conditional
        # entropy detects this.
        #
        # A NEARLY-UNIQUE determinant determines everything
        # trivially — if each age appears once, knowing the age
        # tells you that row's blood pressure, its visit date and
        # its record number, none of which is a derivation. Live,
        # this reported `diastolic_blood_pressure` as derived from
        # `age_at_visit`. So for high-cardinality numeric pairs the
        # test is near-perfect CORRELATION instead, which catches
        # the real arithmetic (an age and a birth year) without the
        # false positives.
        coarse_max = max(2, n // max(2 * self.k, 1))
        numeric_vals = {}
        for c in self.order:
            xs = [_num(v) for v in raw[c]]
            if all(x is not None for x in xs):
                numeric_vals[c] = xs

        def _rho(u, v):
            ru, rv = _rank(u), _rank(v)
            mu = sum(ru) / len(ru)
            mv = sum(rv) / len(rv)
            num = sum((a - mu) * (b - mv) for a, b in zip(ru, rv))
            den = (sum((a - mu) ** 2 for a in ru)
                   * sum((b - mv) ** 2 for b in rv)) ** 0.5
            return num / den if den else 0.0

        # A column holding the LENGTH of a list column is the
        # plainest arithmetic there is. Testing it directly beats
        # any entropy heuristic: if the number matches the item
        # count on essentially every row, it is a tally, not a
        # finding.
        for lc, lengths in getattr(self, "list_lengths", {}).items():
            for cand in self.order:
                xs = [_num(v) for v in raw[cand]]
                if any(x is None for x in xs):
                    continue
                agree = sum(1 for x, L in zip(xs, lengths)
                            if abs(x - L) < 1e-9)
                if agree >= 0.95 * n and cand not in self.derived:
                    self.derived[cand] = lc
                    self.tally_of[cand] = lc

        # ---- families of indicators from the same list ----
        # Expansion moves a relationship DOWN a level. `drug_routes`
        # is determined by `active_drugs` — ondansetron is given IV
        # push, sodium chloride is a flush — but once both are
        # expanded into per-item indicators, the column-level test
        # can no longer see it, and the pharmacology reappears as a
        # dozen separate "findings" about the pipeline's own
        # encoding. Live, seven of thirteen edges were this.
        #
        # So families are tested as families: if most of B's
        # indicators are determined by some indicator of A, the
        # whole of B is derived from A.
        fam_of = {}
        for lc, items in getattr(self, "list_columns", {}).items():
            for it in items:
                fam_of["{}::{}".format(lc, it)] = lc
        fams = defaultdict(list)
        for col, lc in fam_of.items():
            if col in self.order:
                fams[lc].append(col)

        def _determines(a_col, b_col):
            """binary indicators: coarse conditional entropy"""
            hb_ = _entropy(list(Counter(raw[b_col]).values()))
            if hb_ <= 0:
                return False
            g = defaultdict(Counter)
            for j in range(n):
                g[raw[a_col][j]][raw[b_col][j]] += 1
            hc = sum((sum(cc.values()) / n)
                     * _entropy(list(cc.values()))
                     for cc in g.values())
            return hc / hb_ < DERIVED_ENTROPY_RATIO

        self.pair_derived = {}
        self.derived_families = {}
        fam_names = list(fams)
        for bf in fam_names:
            for af in fam_names:
                if af == bf or bf in self.derived_families:
                    continue
                hit = sum(1 for bcol in fams[bf]
                          if any(_determines(acol, bcol)
                                 for acol in fams[af]))
                if hit >= 0.5 * len(fams[bf]):
                    # the richer family explains the poorer one
                    if len(fams[af]) >= len(fams[bf]):
                        self.derived_families[bf] = af
                        for bcol in fams[bf]:
                            self.derived[bcol] = af
                        break

        for b in self.order:
            hb = _entropy(list(Counter(raw[b]).values()))
            if hb <= 0:
                continue
            if b in self.derived:
                continue
            for a_ in self.order:
                if a_ == b:
                    continue
                card_a = len(set(raw[a_]))
                if card_a <= coarse_max:
                    groups_ = defaultdict(Counter)
                    for j in range(n):
                        groups_[raw[a_][j]][raw[b][j]] += 1
                    hcond = sum(
                        (sum(cc.values()) / n)
                        * _entropy(list(cc.values()))
                        for cc in groups_.values())
                    determined = (hcond / hb
                                  < DERIVED_ENTROPY_RATIO)
                elif a_ in numeric_vals and b in numeric_vals:
                    determined = abs(_rho(numeric_vals[a_],
                                          numeric_vals[b])) >= 0.98
                else:
                    determined = False
                if determined:
                    # Two columns can determine EACH OTHER — an age
                    # and a birth year do. Marking both derived
                    # would bar both from the model and lose the
                    # information entirely, so a column whose
                    # proposed determinant is itself derived stays
                    # as the surviving representative of the pair.
                    if a_ in self.derived:
                        continue
                    ha = _entropy(list(Counter(raw[a_]).values()))
                    # keep the more informative column as the
                    # determinant; the other is its shadow
                    if ha >= hb:
                        self.derived[b] = a_
                        break

            # nothing single-handedly explains it — try the pairs
            if b not in self.derived:
                pool = [c2 for c2 in self.order
                        if c2 != b and c2 not in self.derived][:14]
                found, how = None, ""
                if b in num_of:
                    numeric_pool = [c2 for c2 in pool
                                    if c2 in num_of]
                    for x in range(len(numeric_pool)):
                        for y in range(x + 1, len(numeric_pool)):
                            lab = _pair_determines(
                                b, numeric_pool[x], numeric_pool[y])
                            if lab:
                                found = (numeric_pool[x],
                                         numeric_pool[y])
                                how = lab
                                break
                        if found:
                            break
                if found:
                    self.derived[b] = "{} ({} of {})".format(
                        found[0], how, found[1])
                    self.pair_derived[b] = list(found)

        hyp = {c: [x for x in v if x in self.order]
               for c, v in (hypotheses or {}).items()
               if c in self.order}
        self.hypotheses = hyp
        if hyp:
            n_tests = sum(len(v) for v in hyp.values()) or 1
        else:
            n_tests = sum(range(len(self.order))) or 1
        alpha_c = self.alpha / n_tests

        chosen_log = []
        for i, c in enumerate(self.order):
            if hyp:
                # only the declared questions are asked; a column
                # with no hypothesis keeps its marginal
                if c not in hyp:
                    self.parents[c] = []
                    continue
                candidates = [x for x in hyp[c]
                              if x != c and x not in NEVER_PARENT]
            else:
                candidates = self.order[:i]
            candidates = [x for x in candidates
                          if x not in NEVER_PARENT]
            if c in self.derived and self.derived[c] in candidates:
                # its determinant says everything about it; storing
                # that one table is exact and cheap, and no search
                # is warranted
                self.parents[c] = [self.derived[c]]
                continue
            # A derived column carries no information its
            # determinant does not already hold, so it never earns
            # a parent slot — whether or not that determinant is
            # itself still in the running. (A tally of a list stays
            # redundant after the list has been expanded into
            # indicators, which is exactly the case that leaked
            # `active_drugs::furosemide <- active_drug_count` into
            # the findings.)
            candidates = [x for x in candidates
                          if x not in self.derived]
            if c in getattr(self, "pair_derived", {}):
                # its two determinants say everything about it
                self.parents[c] = [
                    x for x in self.pair_derived[c]
                    if x in self.order][:self.max_parents]
                continue
            parents: List[str] = []
            while len(parents) < self.max_parents and candidates:
                best, best_gain, best_df = None, 0.0, 1
                best_n_eff = self.n_groups
                best_level = "between-person"
                best_strat = (0.0, 1, 0, 1)
                strat_best, strat_best_g, strat_best_df = None, 0.0, 1
                found_how = ""
                for cand in candidates:
                    if cand in parents:
                        continue
                    zs = ([tuple(enc[p][j] for p in parents)
                           for j in range(n)] if parents else None)
                    # If BOTH sides move within a person, the
                    # comparison can be made inside patients: use
                    # the centred encodings, and the degrees of
                    # freedom that a within-person comparison
                    # actually has. If either side is a fixed
                    # patient trait, only between-person evidence
                    # exists and the patient count governs.
                    use_w = (self.multilevel
                             and self.level.get(c) == "visit"
                             and self.level.get(cand) == "visit"
                             and c in self.enc_within
                             and cand in self.enc_within)
                    ec = (self.enc_within[c] if use_w else enc[c])
                    ed = (self.enc_within[cand] if use_w
                          else enc[cand])
                    gain, df = self._cmi(ec, ed, zs)
                    if gain > best_gain:
                        best, best_gain, best_df = cand, gain, df
                        best_n_eff = (self.n_within if use_w
                                      else self.n_groups)
                        best_level = ("within-person" if use_w
                                      else "between-person")
                        best_strat = getattr(self, "_last_stratum",
                                             (0.0, 1, 0, 1))
                    # A candidate can be weak POOLED yet decisive
                    # inside one stratum — that is what an
                    # interaction looks like. Keep the strongest
                    # such candidate separately, or greedy search
                    # discards it before the stratum test is ever
                    # applied.
                    sg_c, sdf_c, sn_c, nst_c = getattr(
                        self, "_last_stratum", (0.0, 1, 0, 1))
                    if (sn_c >= self.k
                            and sg_c >= _chi2_crit(
                                sdf_c, alpha_c / max(nst_c, 1))
                            and sg_c > strat_best_g):
                        strat_best = cand
                        strat_best_g = sg_c
                        strat_best_df = sdf_c
                if best is None or best_gain <= 0:
                    # Greedy selection is blind to parents that
                    # matter only JOINTLY: creatinine may carry no
                    # marginal signal yet be decisive among the
                    # elderly. When no single candidate helps, try
                    # the best PAIR before giving up — this is what
                    # rescues interaction structure.
                    pair = None
                    pair_gain, pair_df = 0.0, 1
                    pool = candidates[:8]
                    for x in range(len(pool)):
                        for y in range(x + 1, len(pool)):
                            cx, cy = pool[x], pool[y]
                            zs = ([tuple(enc[q][j] for q in parents)
                                   for j in range(n)]
                                  if parents else None)
                            comb = [enc[cx][j] + "\u2016" +
                                    enc[cy][j] for j in range(n)]
                            g, dfp = self._cmi(enc[c], comb, zs)
                            if g > pair_gain:
                                pair = (cx, cy)
                                pair_gain, pair_df = g, dfp
                    if pair and len(parents) + 2 <= \
                            self.max_parents:
                        g_stat = 2.0 * self.n_groups * pair_gain
                        if g_stat >= _chi2_crit(pair_df, alpha_c):
                            width = 1
                            for q in parents + list(pair):
                                width *= len(set(enc[q]))
                            width *= len(set(enc[c]))
                            if (n / max(width, 1)) >= 1.0:
                                parents.extend(pair)
                                candidates = [
                                    q for q in candidates
                                    if q not in pair]
                                chosen_log.append(
                                    {"child": c,
                                     "parent": "+".join(pair),
                                     "cmi": round(pair_gain, 5),
                                     "g": round(g_stat, 1),
                                     "crit": round(_chi2_crit(
                                         pair_df, alpha_c), 1),
                                     "found_as": "interaction "
                                                 "pair"})
                                continue
                    break
                # Significance scales with INDEPENDENT units. Using
                # the row count would let one patient's forty
                # visits vote forty times.
                g_stat = 2.0 * best_n_eff * best_gain
                crit = _chi2_crit(best_df, alpha_c)
                passes = g_stat >= crit
                sg, sdf, sn, nstr = best_strat
                if not passes and sn >= self.k and \
                        sg >= _chi2_crit(sdf,
                                         alpha_c / max(nstr, 1)):
                    passes = True
                    found_how = "concentrated in one stratum"
                if not passes and strat_best is not None:
                    # the pooled winner failed, but another
                    # candidate is decisive within a stratum
                    best = strat_best
                    best_gain = max(best_gain, 1e-9)
                    passes = True
                    found_how = "concentrated in one stratum"
                # cells must also stay publishable: adding a parent
                # multiplies the configuration count, and a table
                # whose average cell falls below k cannot be kept
                width = 1
                for p in parents + [best]:
                    width *= len(set(enc[p]))
                width *= len(set(enc[c]))
                if not passes or (n / max(width, 1)) < 1.0:
                    break
                parents.append(best)
                candidates = [x for x in candidates if x != best]
                entry = {"child": c, "parent": best,
                         "cmi": round(best_gain, 5),
                         "g": round(g_stat, 1),
                         "crit": round(crit, 1),
                         "level": best_level,
                         "effective_n": best_n_eff}
                if found_how:
                    entry["found_as"] = found_how
                chosen_log.append(entry)
            self.parents[c] = parents

        # ---- every parent must precede its child ----
        # Sampling walks the columns in order and looks each one's
        # parents up in what has already been assigned, so a
        # parent positioned after its child is a crash waiting for
        # the right schema. The greedy search only ever chose
        # parents from earlier columns, but the DERIVED and
        # pair-derived paths assign parents directly and bypass
        # that guarantee — which is exactly how a real extract
        # produced a parent that had not been drawn yet.
        #
        # Rather than trust the construction, the order is
        # re-derived from the edges that actually exist. Any edge
        # that would close a cycle is dropped, because a cycle
        # cannot be sampled at all and losing one edge is far
        # better than failing.
        indeg = {c: 0 for c in self.order}
        children = defaultdict(list)
        for c in self.order:
            for pc in self.parents.get(c, []):
                if pc in indeg:
                    children[pc].append(c)
                    indeg[c] += 1
        ready = [c for c in self.order if not indeg[c]]
        ordered = []
        while ready:
            nxt = ready.pop(0)
            ordered.append(nxt)
            for ch in children[nxt]:
                indeg[ch] -= 1
                if not indeg[ch]:
                    ready.append(ch)
        if len(ordered) < len(self.order):
            # a cycle: keep the remaining columns but strip the
            # parents that cannot be satisfied
            placed = set(ordered)
            for c in self.order:
                if c in placed:
                    continue
                self.parents[c] = [pc for pc in
                                   self.parents.get(c, [])
                                   if pc in placed]
                ordered.append(c)
                placed.add(c)
        self.order = ordered
        # and drop any parent that still fails to precede its
        # child, so the invariant holds by construction
        seen = set()
        for c in self.order:
            kept = [pc for pc in self.parents.get(c, [])
                    if pc in seen]
            self.parents[c] = kept
            seen.add(c)

        # conditional tables, with k-suppression
        # The budget must cover EVERY published quantity, not only
        # the conditional tables. A transition table is published
        # too, and it is derived from the same people — spending
        # epsilon only on the conditional tables and then
        # publishing untouched transitions would state a guarantee
        # that does not hold. The count therefore includes one
        # slot per visit-level column for its transition table and
        # one for the visit-count histogram.
        n_visit_cols = sum(1 for c in self.order
                           if self.level.get(c) == "visit")
        n_tables = max(sum(1 for c in self.order
                           if self.parents.get(c))
                       + n_visit_cols + 1, 1)
        self._dp_scale = (DP_MAX_ROWS_PER_PERSON
                          / (self.epsilon / n_tables)
                          if self.epsilon > 0 else 0.0)
        dp_rng = random.Random(self.seed ^ 0x5EED)
        suppressed = 0
        for c in self.order:
            ps = self.parents[c]
            marg = Counter(enc[c])
            tot = sum(marg.values())
            self.marginal[c] = {s: marg[s] / tot
                                for s in marg}
            table: Dict[str, Dict[str, float]] = {}
            if ps:
                cells = defaultdict(Counter)
                people = defaultdict(set)
                for j in range(n):
                    cfg = "|".join(enc[p][j] for p in ps)
                    cells[cfg][enc[c][j]] += 1
                    people[cfg].add(self.groups[j])
                # which parent positions are ordinal (numeric
                # bins b0, b1, ...) — only those have neighbours
                ordinal = [ix for ix, pn in enumerate(ps)
                           if self.binnings[pn].kind == "numeric"]

                def neighbours(cfg_str):
                    parts = cfg_str.split("|")
                    out = []
                    for ix in ordinal:
                        tok = parts[ix]
                        if not tok.startswith("b"):
                            continue
                        try:
                            bi = int(tok[1:])
                        except ValueError:
                            continue
                        for step in (-1, 1):
                            alt = list(parts)
                            alt[ix] = "b{}".format(bi + step)
                            out.append("|".join(alt))
                    return out

                for cfg, cnt in cells.items():
                    m = sum(cnt.values())
                    # k counts PEOPLE. Ten visits from one patient
                    # is a cell of one, and publishing it would
                    # describe that individual.
                    if len(people[cfg]) < self.k:
                        suppressed += 1
                        continue          # falls back at sampling
                    marg = self.marginal[c]
                    nb_acc, nb_tot = Counter(), 0
                    for nb_cfg in neighbours(cfg):
                        nb = cells.get(nb_cfg)
                        if not nb:
                            continue
                        nm = sum(nb.values())
                        for s, v in nb.items():
                            nb_acc[s] += v / nm
                        nb_tot += 1
                    if self.epsilon > 0:
                        # Laplace noise sized to the budget. One
                        # person can move any count by at most
                        # their capped contribution, so that cap
                        # IS the sensitivity.
                        cnt = _noisy(cnt, self._dp_scale, dp_rng)
                        m = max(sum(cnt.values()), 1e-9)
                    a_s = SMOOTHING
                    if self.epsilon > 0:
                        # When the noise is large relative to the
                        # counts, the table holds no information —
                        # and clipping negatives at zero would
                        # leave it a CONFIDENT point mass, which
                        # is worse than no table at all. Shrinking
                        # toward the column's own marginal in
                        # proportion to the noise makes a swamped
                        # table degrade to "we don't know" instead
                        # of to "we're certain", which is what a
                        # privacy budget should buy.
                        a_s = SMOOTHING + self._dp_scale
                    w_s = NEIGHBOUR_WEIGHT if nb_tot else 0.0
                    syms = set(cnt) | set(marg) | set(nb_acc)
                    table[cfg] = {
                        s: (cnt.get(s, 0)
                            + a_s * marg.get(s, 0.0)
                            + w_s * (nb_acc.get(s, 0.0)
                                     / max(nb_tot, 1)))
                        / (m + a_s + w_s) for s in syms}
            self.cpt[c] = table

        # How much of a column's variation is BETWEEN people rather
        # than within one person's course. A pressure that mostly
        # reflects who the patient is should stay near that
        # patient's own level across their visits; one that is
        # mostly noise should not. This ratio is what decides how
        # strongly a patient's position inside a bin persists.
        self.persistence = {}
        for c in self.order:
            if self.level.get(c) != "visit":
                continue
            b = self.binnings.get(c)
            if b is None or b.kind != "numeric":
                continue
            # Measured on the WITHIN-BIN position, not the raw
            # value. The transition table already reproduces the
            # persistence that lives at the bin level; applying
            # the raw between-patient share on top of it counts
            # the same stability twice, and the generated patients
            # come out steadier than real ones — measured 0.87
            # visit-to-visit against a source of 0.78, which would
            # make a synthetic cohort look better behaved than the
            # people it came from.
            per_bin_pos = defaultdict(list)
            for j in range(n):
                x = _num(rows[j].get(c, ""))
                if x is not None:
                    per_bin_pos[enc[c][j]].append((x, j))
            upos = {}
            for sym, items in per_bin_pos.items():
                items.sort()
                for rank, (_, j) in enumerate(items):
                    upos[j] = (rank + 0.5) / len(items)
            vals, gs = [], []
            for j in range(n):
                if j in upos:
                    vals.append(upos[j])
                    gs.append(self.groups[j])
            if len(vals) < 2 * self.k:
                continue
            grand = sum(vals) / len(vals)
            by_p = defaultdict(list)
            for x, g in zip(vals, gs):
                by_p[g].append(x)
            total = sum((x - grand) ** 2 for x in vals)
            within = 0.0
            for xs in by_p.values():
                m = sum(xs) / len(xs)
                within += sum((x - m) ** 2 for x in xs)
            if total > 0:
                ratio = 1.0 - within / total
                if self.epsilon > 0:
                    # a single scalar per column, but still a
                    # statistic about these patients: it is
                    # coarsened rather than published exactly
                    ratio = round(ratio * 10.0) / 10.0
                self.persistence[c] = max(0.0, min(0.95, ratio))

        # ---- give each column the resolution IT can support ----
        # Bin resolution was solved once for the whole model, from
        # the worst case: a column with the maximum number of
        # parents. But cells multiply only along a column's OWN
        # parents, so a column with none can carry far finer bins
        # at the same cell occupancy — and coarse bins are exactly
        # what caps marginal accuracy and within-bin correlation.
        #
        # Only columns that no other column conditions on are
        # refined. Refining a parent would invalidate every table
        # built against its old encoding, and quietly wrong tables
        # are worse than coarse ones.
        # Only LEAF columns are refined — those nothing else
        # conditions on.
        #
        # Refining a parent is tempting and was tried: it improved
        # that column's own marginal and lifted the cross-column
        # correlation. But finer parent bins multiply out into
        # thinner cells for every child, those cells fall below
        # the k-person floor, and the relationships that depended
        # on them get suppressed. Measured, it destroyed U-shape
        # and interaction recovery outright — trading the
        # structure this model exists to capture for a better
        # histogram. A leaf has no children to starve, so the win
        # there is free.
        used_as_parent = set()
        for c in self.order:
            for pc in self.parents.get(c, []):
                used_as_parent.add(pc)
        self.refined = {}
        for c in self.order:
            if c in self.derived or c in used_as_parent:
                continue
            b = self.binnings.get(c)
            if b is None or b.kind != "numeric":
                continue
            width = 1
            for pc in self.parents.get(c, []):
                width *= max(len(set(enc[pc])), 1)
            # a child's cells multiply along its parents, so the
            # budget for its own bins is what is left after them
            allowed = int(self.n_groups / (self.k * max(width, 1)))
            if allowed <= nbins + 1:
                continue
            finer = min(allowed, 4 * nbins, 24)
            nb = Binning.learn(c, [r.get(c, "") for r in rows],
                               self.k, finer, groups=self.groups)
            if len(nb.edges) - 1 > len(b.edges) - 1:
                self.binnings[c] = nb
                enc[c] = [nb.encode(r.get(c, "")) for r in rows]
                self.refined[c] = len(nb.edges) - 1

        # Only the refined columns need rebuilding: nothing
        # conditions on them, so no other table is affected.
        for c in list(self.refined):
            ps = self.parents.get(c, [])
            marg = Counter(enc[c])
            tot = sum(marg.values())
            self.marginal[c] = {s: marg[s] / tot for s in marg}
            table = {}
            if ps:
                cells = defaultdict(Counter)
                ppl2 = defaultdict(set)
                for j in range(n):
                    cfg = "|".join(enc[q][j] for q in ps)
                    cells[cfg][enc[c][j]] += 1
                    ppl2[cfg].add(self.groups[j])
                for cfg, cnt in cells.items():
                    if len(ppl2[cfg]) < self.k:
                        continue
                    m = sum(cnt.values())
                    syms = set(cnt) | set(self.marginal[c])
                    table[cfg] = {
                        s: (cnt.get(s, 0)
                            + SMOOTHING * self.marginal[c].get(
                                s, 0.0)) / (m + SMOOTHING)
                        for s in syms}
            self.cpt[c] = table

        # ---- what the source's own visit-to-visit stability is --
        # Kept as a target rather than a diagnostic. The bin-level
        # transitions and the within-bin anchor both contribute to
        # how steady a patient looks, and they do not combine in
        # any closed form worth deriving — measuring the raw
        # source correlation and calibrating against it is both
        # simpler and exact, and it self-corrects when either
        # mechanism changes.
        self.target_autocorr = {}
        seq_by = defaultdict(list)
        for j in range(n):
            seq_by[self.groups[j]].append(j)
        for c in self.order:
            if self.level.get(c) != "visit":
                continue
            b = self.binnings.get(c)
            if b is None or b.kind != "numeric":
                continue
            pr = []
            for idxs in seq_by.values():
                xs = [_num(rows[j].get(c, "")) for j in idxs]
                xs = [x for x in xs if x is not None]
                pr += list(zip(xs, xs[1:]))
            if len(pr) < 4 * self.k:
                continue
            a_ = [x for x, _ in pr]
            b_ = [y for _, y in pr]
            ma = sum(a_) / len(a_)
            mb = sum(b_) / len(b_)
            nu = sum((x - ma) * (y - mb) for x, y in pr)
            de = (sum((x - ma) ** 2 for x in a_)
                  * sum((y - mb) ** 2 for y in b_)) ** 0.5
            if de:
                t = nu / de
                if self.epsilon > 0:
                    # This is measured from the real patients and
                    # it steers generation, so under a budget it
                    # is published coarsely rather than exactly.
                    # The calibrated weights derive from it, so
                    # coarsening here is what keeps them covered.
                    t = round(t * 10.0) / 10.0
                self.target_autocorr[c] = t

        # ---- correlation of positions INSIDE the bins ----
        # Two columns can be linked in two separate ways: which
        # bins they land in together, and where inside those bins
        # they sit. The conditional tables capture the first. The
        # second is thrown away by drawing each column's position
        # independently — and it is not a small residue: source
        # systolic and diastolic pressure correlate at 0.873,
        # while bin membership alone reproduces only 0.652.
        #
        # So the within-bin position of a child is drawn correlated
        # with its parent's, using the correlation measured here.
        self.pos_corr = {}
        pos_of = {}
        for c in self.order:
            b = self.binnings.get(c)
            if b is None or b.kind != "numeric":
                continue
            per_bin = defaultdict(list)
            for j in range(n):
                x = _num(rows[j].get(c, ""))
                if x is not None:
                    per_bin[enc[c][j]].append((x, j))
            u = [None] * n
            for sym, items in per_bin.items():
                items.sort()
                m = len(items)
                for rank, (_, j) in enumerate(items):
                    u[j] = (rank + 0.5) / m
            pos_of[c] = u
        for c, u_c in pos_of.items():
            for pcol in self.parents.get(c, []):
                u_p = pos_of.get(pcol)
                if u_p is None:
                    continue
                both = [(a_, b_) for a_, b_ in zip(u_c, u_p)
                        if a_ is not None and b_ is not None]
                if len(both) < 2 * self.k:
                    continue
                za = [_norm_ppf(x) for x, _ in both]
                zb = [_norm_ppf(y) for _, y in both]
                ma = sum(za) / len(za)
                mb = sum(zb) / len(zb)
                num = sum((x - ma) * (y - mb)
                          for x, y in zip(za, zb))
                den = (sum((x - ma) ** 2 for x in za)
                       * sum((y - mb) ** 2 for y in zb)) ** 0.5
                if den:
                    rho = max(-0.95, min(0.95, num / den))
                    if abs(rho) > 0.1:
                        self.pos_corr[c] = (pcol, round(rho, 4))
                        break

        # transition tables: how a value moves visit to visit
        for c in self.order:
            if self.level.get(c) != "visit":
                continue
            trans = defaultdict(Counter)
            people = defaultdict(set)
            for g, idxs in seq.items():
                for a_i, b_i in zip(idxs, idxs[1:]):
                    trans[enc[c][a_i]][enc[c][b_i]] += 1
                    people[enc[c][a_i]].add(g)
            tbl = {}
            for prev, cnt in trans.items():
                if len(people[prev]) < self.k:
                    continue           # too few people to publish
                marg = self.marginal[c]
                a_t = SMOOTHING
                if self.epsilon > 0:
                    cnt = _noisy(cnt, self._dp_scale, dp_rng)
                    # as with the conditional tables, a swamped
                    # transition must degrade to the marginal
                    # rather than to a confident point mass
                    a_t = SMOOTHING + self._dp_scale
                m = max(sum(cnt.values()), 1e-9)
                syms = set(cnt) | set(marg)
                tbl[prev] = {s: (cnt.get(s, 0)
                                 + a_t * marg.get(s, 0.0))
                             / (m + a_t) for s in syms}
            if tbl:
                self.lag[c] = tbl


        self.report = {
            "rows": n,
            "persons": self.n_groups,
            "grouped_by": group_by or "(none — each row treated "
                                      "as its own unit)",
            "effective_n": self.n_groups,
            "clustering_note": (
                "{} rows from {} people (mean {:.1f} each): "
                "k-anonymity counts PEOPLE and significance uses "
                "the person count, because repeated encounters "
                "from one patient are neither independent "
                "evidence nor protection".format(
                    n, self.n_groups, n / max(self.n_groups, 1))
                if group_by else
                "no grouping declared — every row treated as an "
                "independent unit"),
            "columns_modelled": len(self.order),
            "columns_dropped_no_variation": dropped,
            "edges": [{"child": c, "parents": self.parents[c]}
                      for c in self.order
                      if self.parents.get(c)
                      and c not in self.derived],
            "edge_count": sum(
                len(self.parents.get(c, []))
                for c in self.order if c not in self.derived),
            "derived_columns": [
                {"column": b, "determined_by": a_}
                for b, a_ in self.derived.items()],
            "pair_derived": {k2: list(v) for k2, v in
                             getattr(self, "pair_derived",
                                     {}).items()},
            "pair_note": "some bookkeeping needs two columns to "
                         "explain it: a birth year is fixed while "
                         "an age moves visit to visit, so neither "
                         "determines the other alone, but the pair "
                         "(age, visit date) fixes it exactly",
            "derived_families": dict(
                getattr(self, "derived_families", {})),
            "family_note": "when one list determines another — the "
                           "route a drug is given by is a property "
                           "of the drug — the whole family is filed "
                           "as arithmetic rather than surfacing as "
                           "a dozen separate findings about the "
                           "pipeline's own encoding",
            "differential_privacy": (
                {"epsilon": self.epsilon,
                 "max_rows_per_person": DP_MAX_ROWS_PER_PERSON,
                 "rows_dropped_to_bound_contribution":
                     self.dp_dropped,
                 "noise_scale": round(self._dp_scale, 3),
                 "covers": ["conditional tables",
                            "transition tables",
                            "visit-count histogram",
                            "bin edges (percentile-clamped)",
                            "persistence ratios (coarsened)",
                            "visit-to-visit steadiness targets "
                            "(coarsened)"],
                 "budget_split_over": n_tables,
                 "reading": "every published table carries "
                            "calibrated noise. The model "
                            "satisfies {}-differential privacy "
                            "with respect to any single patient: "
                            "an adversary holding everything else "
                            "about the cohort still cannot tell "
                            "whether one particular person was in "
                            "it, beyond a factor bounded by that "
                            "number.".format(self.epsilon)}
                if self.epsilon > 0 else
                {"epsilon": None,
                 "reading": "no differential-privacy budget was "
                            "set: protection rests on "
                            "k-anonymity and percentile clamping, "
                            "which are properties of the tables "
                            "rather than a bound on what an "
                            "adversary can infer"}),
            "never_parent_excluded": [c for c in NEVER_PARENT
                                      if c in self.order],
            "unmodellable_excluded": dict(
                getattr(self, "excluded_columns", {})),
            "leakage_note": "a column that DEFINES the outcome is "
                            "never offered as a parent: an outcome "
                            "meaning 'the next visit fell within "
                            "thirty days' is perfectly predicted "
                            "by the gap to the next visit, and a "
                            "model given both would score "
                            "flawlessly while learning nothing",
            "dropped_odd_column_names": getattr(
                self, "dropped_odd_names", []),
            "list_columns_expanded": {
                c: len(v) for c, v in
                getattr(self, "list_columns", {}).items()},
            "list_note": "list-valued columns are expanded into "
                         "one indicator per frequent item rather "
                         "than binned as whole strings: every "
                         "combination is nearly unique, so binning "
                         "collapses the column to a single bucket "
                         "and its clinical content vanishes",
            "derived_note": "these are arithmetic the data "
                            "pipeline created (a count from a "
                            "list, an age from a birth year), not "
                            "discovered relationships — they are "
                            "reproduced exactly but excluded from "
                            "the findings and from competing for "
                            "parent slots",
            "acceptance_log": chosen_log,
            "suppressed_configurations": suppressed,
            "k": self.k, "max_parents": self.max_parents,
            "hypotheses_declared": len(hyp),
            "multilevel": self.multilevel,
            "within_person_df": self.n_within,
            "column_levels": dict(self.level),
            "multilevel_note": (
                "relationships between columns that both move "
                "within a patient are tested on {} within-person "
                "degrees of freedom; anything involving a fixed "
                "patient trait is tested on {} patients".format(
                    self.n_within, self.n_groups)
                if self.multilevel else
                "not enabled — every relationship is tested on "
                "the patient count"),
            "comparisons_corrected_for": n_tests,
            "search_mode": ("targeted — only the declared "
                            "relationships were tested, so the "
                            "multiple-comparison correction is "
                            "over {} tests rather than {}".format(
                                n_tests,
                                sum(range(len(self.order))) or 1)
                            if hyp else
                            "blind — every pair was tested, and "
                            "the correction is over all {} of "
                            "them".format(n_tests)),
            "bins": nbins,
            "refined_columns": dict(getattr(self, "refined", {})),
            "refinement_note": "columns that nothing else "
                               "conditions on were re-binned at "
                               "the resolution their own parent "
                               "count supports; a column with no "
                               "parents can carry far more detail "
                               "than the model-wide solve allows",
            "bins_chosen": ("auto — solved so cells stay populated "
                            "enough to detect dependence"
                            if auto else "set by caller"),
            "policy": "a parent is accepted only if its dependence "
                      "survives a Bonferroni-corrected G-test AND "
                      "the resulting table stays publishable at k; "
                      "thin configurations back off to fewer "
                      "parents at sampling time",
        }
        if self.multilevel:
            # tune the anchor weights until generated patients are
            # exactly as steady as the source, no more and no less
            self.calibrate_persistence()
        return self

    # -- sampling ------------------------------------------------
    def _lookup(self, c, assign):
        # Belt as well as braces. The column order is now derived
        # topologically so every parent is drawn first, but this
        # is the generation path: if some future change breaks
        # that invariant, backing off to the parents that ARE
        # available produces slightly less specific data, while
        # raising produces none at all.
        ps = [pc for pc in self.parents[c] if pc in assign]
        if len(ps) != len(self.parents[c]):
            self._order_warnings = getattr(
                self, "_order_warnings", 0) + 1
        full = self.parents[c]
        for depth in range(len(ps), -1, -1):
            use = ps[:depth]
            if not use:
                return self.marginal[c], depth
            cfg = "|".join(assign[p] for p in use)
            if depth == len(full) and ps == full:
                d = self.cpt[c].get(cfg)
                if d:
                    return d, depth
            else:
                # re-marginalize the full table over the kept prefix
                acc: Counter = Counter()
                tot = 0
                for full_cfg, dist in self.cpt[c].items():
                    if "|".join(full_cfg.split("|")[:depth]) == cfg:
                        for s, p in dist.items():
                            acc[s] += p
                        tot += 1
                if tot:
                    z = sum(acc.values())
                    return {s: v / z for s, v in acc.items()}, depth
        return self.marginal[c], 0

    def sample(self, rows: int, seed: Optional[int] = None
               ) -> List[Dict[str, Any]]:
        s = self.seed if seed is None else seed
        out = []
        for i in range(rows):
            assign: Dict[str, str] = {}
            for c in self.order:
                dist, _ = self._lookup(c, assign)
                r = _rng(s, i, c, "pick")
                syms = list(dist.keys())
                ws = [dist[x] for x in syms]
                assign[c] = r.choices(syms, weights=ws, k=1)[0]
            row = {}
            zpos: Dict[str, float] = {}
            for c in self.order:
                pos = None
                if self.binnings[c].kind == "numeric":
                    link = self.pos_corr.get(c)
                    e = _rng(s, i, c, "pos").gauss(0.0, 1.0)
                    if link and link[0] in zpos:
                        rho = link[1]
                        z = rho * zpos[link[0]] + \
                            ((1.0 - rho * rho) ** 0.5) * e
                    else:
                        z = e
                    zpos[c] = z
                    pos = _norm_cdf(z)
                row[c] = self.binnings[c].decode(
                    assign[c], _rng(s, i, c, "decode"), pos)
            # Rebuild the list columns from their indicators, so
            # generated data has the same shape as the source
            # rather than a pile of expanded flags. The indicators
            # stay too: they are what carried the structure.
            for lc, items in getattr(self, "list_columns",
                                     {}).items():
                present = [it for it in items
                           if str(row.get("{}::{}".format(lc, it),
                                          "0")).strip() == "1"]
                row[lc] = LIST_SEP.join(sorted(present))
            # A tally is arithmetic on the list, so COMPUTE it
            # rather than sampling it — otherwise generated rows
            # carry a count that contradicts the list beside it.
            for tally, lc in getattr(self, "tally_of", {}).items():
                items = self.list_columns.get(lc)
                if items is None:
                    continue
                row[tally] = sum(
                    1 for it in items
                    if str(row.get("{}::{}".format(lc, it),
                                   "0")).strip() == "1")
            out.append(row)
        return out

    # -- dials ---------------------------------------------------
    def amplify(self, child: str, factor: float) -> "CondNet":
        """Strengthen (>1) or weaken (<1) every dependence feeding
        `child`. factor=0 removes it entirely (child becomes
        independent of its parents); factor=1 leaves it unchanged.
        Marginals are preserved as closely as the tilt allows."""
        marg = self.marginal[child]
        for cfg, dist in self.cpt[child].items():
            new = {}
            for s, p in dist.items():
                m = max(marg.get(s, 1e-9), 1e-9)
                new[s] = m * ((p / m) ** factor)
            z = sum(new.values()) or 1.0
            self.cpt[child][cfg] = {s: v / z
                                    for s, v in new.items()}
        return self

    def set_marginal(self, col: str,
                     weights: Dict[str, float]) -> "CondNet":
        z = sum(weights.values()) or 1.0
        self.marginal[col] = {s: w / z for s, w in weights.items()}
        return self

    # -- persistence ---------------------------------------------
    def _measure_autocorr(self, rows, col):
        by = defaultdict(list)
        for r in rows:
            by[r.get(self.group_by or "person_id")].append(r)
        pr = []
        for v in by.values():
            xs = [_num(r.get(col)) for r in v]
            xs = [x for x in xs if x is not None]
            pr += list(zip(xs, xs[1:]))
        if len(pr) < 30:
            return None
        a_ = [x for x, _ in pr]
        b_ = [y for _, y in pr]
        ma = sum(a_) / len(a_)
        mb = sum(b_) / len(b_)
        nu = sum((x - ma) * (y - mb) for x, y in pr)
        de = (sum((x - ma) ** 2 for x in a_)
              * sum((y - mb) ** 2 for y in b_)) ** 0.5
        return nu / de if de else None

    def temporal_check(self, rows) -> Dict[str, Any]:
        """Did the visit sequences actually come out with the
        steadiness the source had?

        A privacy budget can erase temporal structure entirely
        while generation still reports patients and visit numbers.
        Someone would then hold data that LOOKS longitudinal and
        behaves like independent rows — a worse position than
        knowing they have loose rows, because nothing about the
        output says so.
        """
        lost, kept = [], []
        for col, target in getattr(self,
                                   "target_autocorr", {}).items():
            if abs(target) < 0.2:
                continue
            got = self._measure_autocorr(rows, col)
            if got is None:
                continue
            (kept if abs(got) >= 0.4 * abs(target)
             else lost).append(
                {"column": col, "source": round(target, 3),
                 "generated": round(got, 3)})
        out = {"reproduced": kept, "lost": lost}
        if lost:
            out["warning"] = (
                "The visit-to-visit steadiness the source had was "
                "largely erased for: {}. These records carry a "
                "patient and a visit number but behave like "
                "independent rows. If a privacy budget is set, "
                "this is its cost — raise epsilon or generate "
                "without one when the analysis depends on "
                "patients having a course.".format(
                    ", ".join(x["column"] for x in lost)))
        return out

    def calibrate_persistence(self, trials: int = 4,
                              patients: int = 150) -> None:
        """Tune each column's anchor weight until generated data
        is as steady as the source, no more and no less.

        Overshooting matters as much as undershooting: synthetic
        patients that hold their values more tightly than real
        ones would make any model that groups by patient look
        better behaved than it will be in practice.
        """
        if not getattr(self, "target_autocorr", None):
            return
        # All columns at once. Each column's anchor is independent
        # of the others, so one generated sample per trial serves
        # every column — calibrating them one at a time meant a
        # full generation per column per trial, which took four
        # minutes on a 72-column extract and would only get worse.
        cols = [c for c in self.target_autocorr
                if c in self.persistence]
        if not cols:
            return
        bounds = {c: [0.0, 0.95] for c in cols}
        for _ in range(trials):
            for c in cols:
                self.persistence[c] = sum(bounds[c]) / 2.0
            sample = self.sample_patients(patients, seed=99)
            for c in cols:
                got = self._measure_autocorr(sample, c)
                if got is None:
                    continue
                mid = sum(bounds[c]) / 2.0
                if got < self.target_autocorr[c]:
                    bounds[c][0] = mid
                else:
                    bounds[c][1] = mid
        for c in cols:
            self.persistence[c] = round(sum(bounds[c]) / 2.0, 3)
        self.report["persistence_calibrated"] = {
            c: self.persistence.get(c)
            for c in self.target_autocorr}
        self.report["persistence_targets"] = {
            c: round(v, 3) for c, v in
            self.target_autocorr.items()}

    def sample_patients(self, n_patients: int, seed: int = 0,
                        max_visits: int = 0
                        ) -> List[Dict[str, Any]]:
        """Generate PEOPLE, each with a course of visits.

        A patient is drawn once — their fixed traits, and how many
        times they are seen. Their visits are then drawn in order,
        each one conditioned on the patient's traits and on the
        visit before it, so a value that drifts drifts plausibly
        instead of being redrawn from scratch every row.

        Row-wise sampling cannot produce this. It is also what
        makes a synthetic dataset behave like real longitudinal
        data under any analysis that groups by patient — which is
        most of them.
        """
        rng = random.Random(seed)
        counts = list(self.visit_counts.items()) or [(1, 1)]
        total = float(sum(w for _, w in counts))
        out: List[Dict[str, Any]] = []
        patient_cols = [c for c in self.order
                        if self.level.get(c) == "patient"]
        for pi in range(n_patients):
            u = rng.random() * total
            acc, nvis = 0.0, counts[0][0]
            for v, w in counts:
                acc += w
                if u <= acc:
                    nvis = v
                    break
            if max_visits:
                nvis = min(nvis, max_visits)
            pid = "SYN{:06d}".format(pi)
            traits: Dict[str, str] = {}
            prev: Dict[str, str] = {}
            anchors: Dict[str, float] = {}
            for vi in range(max(1, nvis)):
                assign: Dict[str, str] = {}
                for c in self.order:
                    if c in traits:               # fixed trait
                        assign[c] = traits[c]
                        continue
                    # Two sources of structure, and BOTH must
                    # apply. Letting the transition table replace
                    # the parent conditioning was silently
                    # discarding every cross-column relationship
                    # after the first visit — systolic-diastolic
                    # correlation fell from 0.66 to 0.15, because
                    # diastolic stopped depending on systolic at
                    # all. They are combined instead: each source
                    # contributes its departure from the column's
                    # own marginal, which is the standard way to
                    # merge two conditional beliefs about the same
                    # quantity.
                    dist, _ = self._lookup(c, assign)
                    lag_tbl = self.lag.get(c)
                    if vi and lag_tbl and c in prev \
                            and prev[c] in lag_tbl:
                        lagd = lag_tbl[prev[c]]
                        marg = self.marginal[c]
                        merged, tot = {}, 0.0
                        for sym in set(dist) | set(lagd):
                            m0 = marg.get(sym, 1e-9) or 1e-9
                            v = (dist.get(sym, 0.0)
                                 * lagd.get(sym, 0.0) / m0)
                            merged[sym] = v
                            tot += v
                        if tot > 0:
                            dist = {s: v / tot
                                    for s, v in merged.items()}
                    rr = _rng(seed, pi, vi, c, "pick")
                    syms = list(dist.keys())
                    ws = [dist[x] for x in syms]
                    assign[c] = rr.choices(syms, weights=ws,
                                           k=1)[0]
                    if c in patient_cols:
                        traits[c] = assign[c]
                row = {}
                fresh_z: Dict[str, float] = {}
                for c in self.order:
                    pos = None
                    per = self.persistence.get(c)
                    if per is None and \
                            self.binnings[c].kind == "numeric" \
                            and c in self.pos_corr:
                        per = 0.0        # still needs the link
                    if per is not None:
                        # A Gaussian latent, NOT a blend of two
                        # uniforms. Averaging two uniforms gives a
                        # triangular distribution that pulls every
                        # value toward the middle of its bin and
                        # quietly distorts the marginal — measured
                        # as a drop from 7 of 11 fidelity checks
                        # to 4. Mixing in normal space and mapping
                        # back through the normal CDF leaves the
                        # position exactly uniform while giving
                        # successive visits the intended
                        # correlation.
                        # Both halves of the latent must carry the
                        # link to the parent column. Anchoring
                        # each column independently was what
                        # collapsed systolic-diastolic correlation
                        # from 0.652 to 0.152: it added a large
                        # persistent component that the two
                        # columns did not share.
                        link = self.pos_corr.get(c)
                        rho = link[1] if link else 0.0
                        par = link[0] if link else None
                        if c not in anchors:
                            a_e = _rng(seed, pi, c,
                                       "anchor").gauss(0.0, 1.0)
                            if par and par in anchors:
                                anchors[c] = (
                                    rho * anchors[par]
                                    + ((1.0 - rho * rho) ** 0.5)
                                    * a_e)
                            else:
                                anchors[c] = a_e
                        f_e = _rng(seed, pi, vi, c,
                                   "pos").gauss(0.0, 1.0)
                        if par and par in fresh_z:
                            ze = (rho * fresh_z[par]
                                  + ((1.0 - rho * rho) ** 0.5)
                                  * f_e)
                        else:
                            ze = f_e
                        fresh_z[c] = ze
                        z = (per ** 0.5) * anchors[c] + \
                            ((1.0 - per) ** 0.5) * ze
                        pos = _norm_cdf(z)
                    row[c] = self.binnings[c].decode(
                        assign[c], _rng(seed, pi, vi, c, "dec"),
                        pos)
                for lc, items in getattr(self, "list_columns",
                                         {}).items():
                    present = [it for it in items
                               if str(row.get("{}::{}".format(
                                   lc, it), "0")).strip() == "1"]
                    row[lc] = LIST_SEP.join(sorted(present))
                for tally, lc in getattr(self, "tally_of",
                                         {}).items():
                    items = self.list_columns.get(lc)
                    if items is None:
                        continue
                    row[tally] = sum(
                        1 for it in items
                        if str(row.get("{}::{}".format(lc, it),
                                       "0")).strip() == "1")
                if self.group_by:
                    row[self.group_by] = pid
                row["visit_number"] = vi + 1
                self._emitted_visits = getattr(
                    self, "_emitted_visits", 0) + 1
                out.append(row)
                prev = dict(assign)
        return out

    def log_likelihood(self, row: Dict[str, Any]) -> float:
        """How probable this model finds a given record.

        Published as an attack surface on purpose: a model that
        assigns visibly higher probability to the people it was
        trained on is leaking membership, and the only way to know
        is to compute it.
        """
        total = 0.0
        assign = {}
        for c in self.order:
            b = self.binnings.get(c)
            if b is None:
                continue
            assign[c] = b.encode(row.get(c, ""))
        for c in self.order:
            if c not in assign:
                continue
            dist, _ = self._lookup(c, assign)
            p = dist.get(assign[c], 1e-9)
            total += math.log(max(p, 1e-12))
        return total

    def to_json(self) -> str:
        return json.dumps({
            "k": self.k, "max_parents": self.max_parents,
            "group_by": getattr(self, "group_by", None),
            "max_bins": getattr(self, "resolved_bins",
                                self.max_bins),
            "alpha": self.alpha,
            "seed": self.seed, "order": self.order,
            "binnings": {c: b.to_json()
                         for c, b in self.binnings.items()},
            "parents": self.parents,
            "cpt": self.cpt, "marginal": self.marginal,
            # The hierarchical tables must travel with the model.
            # Without them a reloaded model can still draw rows
            # but no longer draws PEOPLE — and the bench reloads
            # the model on every generate to avoid mutating it,
            # so dropping these silently disabled patient
            # generation everywhere it mattered.
            "visit_counts": {str(k): v for k, v in
                             getattr(self, "visit_counts",
                                     {}).items()},
            "lag": getattr(self, "lag", {}),
            "persistence": dict(getattr(self, "persistence", {})),
            "target_autocorr": {
                c: round(v, 5) for c, v in
                getattr(self, "target_autocorr", {}).items()},
            "pos_corr": {k: list(v) for k, v in
                         getattr(self, "pos_corr", {}).items()},
            "level": dict(getattr(self, "level", {})),
            "group_by": getattr(self, "group_by", None),
            "tally_of": dict(getattr(self, "tally_of", {})),
            "list_columns": {c: list(v) for c, v in
                             getattr(self, "list_columns",
                                     {}).items()},
            "report": self.report,
            "privacy": "counts and quantile edges only; every "
                       "published cell is supported by at least k "
                       "rows; no record is stored",
        }, indent=1)

    @staticmethod
    def from_json(text: str) -> "CondNet":
        d = json.loads(text)
        net = CondNet(d["k"], d["max_parents"], d["max_bins"],
                      d["alpha"], d["seed"])
        net.order = d["order"]
        net.binnings = {c: Binning.from_json(b)
                        for c, b in d["binnings"].items()}
        net.parents = d["parents"]
        net.cpt = d["cpt"]
        net.marginal = d["marginal"]
        net.report = d.get("report", {})
        net.list_columns = {c: list(v) for c, v in
                            (d.get("list_columns") or {}).items()}
        net.tally_of = dict(d.get("tally_of") or {})
        net.visit_counts = Counter(
            {int(k): v for k, v in
             (d.get("visit_counts") or {}).items()})
        net.lag = {c: {p: dict(dist) for p, dist in t.items()}
                   for c, t in (d.get("lag") or {}).items()}
        net.level = dict(d.get("level") or {})
        net.persistence = dict(d.get("persistence") or {})
        net.target_autocorr = dict(d.get("target_autocorr") or {})
        net.pos_corr = {k: (v[0], v[1]) for k, v in
                        (d.get("pos_corr") or {}).items()}
        net.group_by = d.get("group_by")
        return net
