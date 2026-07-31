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
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

MISSING = "\u2205"          # its own bin: missingness is a pattern

# How far the outermost bins may reach past p1/p99 when emitting a
# value, as a multiple of the adjacent interval's width. Zero keeps
# every synthetic value inside the published range (safest, but the
# distribution comes out slightly narrow); larger values restore
# tail spread with extrapolated — never observed — extremes.
TAIL_REACH = 2.0

# Dirichlet smoothing strength for conditional tables. A cell
# supported by twelve people should not yield a point estimate as
# confident as one supported by twelve hundred, so every table is
# shrunk toward its own marginal by a pseudo-count. This is where
# thin-data behaviour lives, and raw maximum likelihood is
# overconfident exactly where it can least afford to be.
SMOOTHING = 1.0

# A column is DERIVED when another column determines it almost
# perfectly — a count computed from a list, an age computed from a
# birth year, a flag defined by a gap. These are arithmetic the
# pipeline itself created, not clinical findings, and letting them
# compete for parent slots wastes a scarce statistical budget on
# rediscovering our own bookkeeping.
DERIVED_ENTROPY_RATIO = 0.08


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
                 atoms=None):
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
                           atoms=atoms)
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

    def decode(self, sym: str, rng: random.Random):
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
            return int(round(x)) if self.integer else round(x, 4)
        return sym

    def to_json(self):
        return {"name": self.name, "kind": self.kind,
                "edges": [round(e, 6) for e in self.edges],
                "fine": [round(e, 6) for e in self.fine],
                "atoms": {b: [[round(v, 6), round(pp, 6)]
                              for v, pp in lst]
                          for b, lst in self.atoms.items()},
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
                                       or {}).items()})


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

    def learn(self, rows: List[Dict[str, Any]],
              columns: Optional[List[str]] = None,
              targets: Optional[List[str]] = None,
              group_by: Optional[str] = None) -> "CondNet":
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
        self.n_groups = len(set(self.groups))
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
        for b in self.order:
            hb = _entropy(list(Counter(enc[b]).values()))
            if hb <= 0:
                continue
            for a_ in self.order:
                if a_ == b:
                    continue
                groups_ = defaultdict(Counter)
                for j in range(n):
                    groups_[enc[a_][j]][enc[b][j]] += 1
                hcond = sum(
                    (sum(cc.values()) / n)
                    * _entropy(list(cc.values()))
                    for cc in groups_.values())
                if hcond / hb < DERIVED_ENTROPY_RATIO:
                    ha = _entropy(list(Counter(enc[a_]).values()))
                    # keep the more informative column as the
                    # determinant; the other is its shadow
                    if ha >= hb:
                        self.derived[b] = a_
                        break

        tests = 0
        for i, c in enumerate(self.order):
            tests += min(i, 1) * i
        tests = max(tests, 1)
        alpha_c = self.alpha / max(
            1, sum(range(len(self.order))) or 1)

        chosen_log = []
        for i, c in enumerate(self.order):
            candidates = self.order[:i]
            if c in self.derived and self.derived[c] in candidates:
                # its determinant says everything about it; storing
                # that one table is exact and cheap, and no search
                # is warranted
                self.parents[c] = [self.derived[c]]
                continue
            # do not let a derived shadow compete against the
            # column that determines it — they carry the same
            # information and only one slot should be spent
            shadows = {b for b, a_ in self.derived.items()
                       if a_ in candidates}
            candidates = [x for x in candidates if x not in shadows]
            parents: List[str] = []
            while len(parents) < self.max_parents and candidates:
                best, best_gain, best_df = None, 0.0, 1
                best_strat = (0.0, 1, 0, 1)
                strat_best, strat_best_g, strat_best_df = None, 0.0, 1
                found_how = ""
                for cand in candidates:
                    if cand in parents:
                        continue
                    zs = ([tuple(enc[p][j] for p in parents)
                           for j in range(n)] if parents else None)
                    gain, df = self._cmi(enc[c], enc[cand], zs)
                    if gain > best_gain:
                        best, best_gain, best_df = cand, gain, df
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
                g_stat = 2.0 * self.n_groups * best_gain
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
                         "crit": round(crit, 1)}
                if found_how:
                    entry["found_as"] = found_how
                chosen_log.append(entry)
            self.parents[c] = parents

        # conditional tables, with k-suppression
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
                for cfg, cnt in cells.items():
                    m = sum(cnt.values())
                    # k counts PEOPLE. Ten visits from one patient
                    # is a cell of one, and publishing it would
                    # describe that individual.
                    if len(people[cfg]) < self.k:
                        suppressed += 1
                        continue          # falls back at sampling
                    marg = self.marginal[c]
                    a_s = SMOOTHING
                    syms = set(cnt) | set(marg)
                    table[cfg] = {
                        s: (cnt.get(s, 0) + a_s * marg.get(s, 0.0))
                        / (m + a_s) for s in syms}
            self.cpt[c] = table

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
            "bins": nbins,
            "bins_chosen": ("auto — solved so cells stay populated "
                            "enough to detect dependence"
                            if auto else "set by caller"),
            "policy": "a parent is accepted only if its dependence "
                      "survives a Bonferroni-corrected G-test AND "
                      "the resulting table stays publishable at k; "
                      "thin configurations back off to fewer "
                      "parents at sampling time",
        }
        return self

    # -- sampling ------------------------------------------------
    def _lookup(self, c, assign):
        ps = self.parents[c]
        for depth in range(len(ps), -1, -1):
            use = ps[:depth]
            if not use:
                return self.marginal[c], depth
            cfg = "|".join(assign[p] for p in use)
            if depth == len(ps):
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
            for c in self.order:
                row[c] = self.binnings[c].decode(
                    assign[c], _rng(s, i, c, "decode"))
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
        return net
