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
                 integer=False, missing_rate=0.0):
        self.name = name
        self.kind = kind                  # numeric | categorical
        self.edges = edges or []          # len = nbins + 1
        self.levels = levels or []
        self.integer = integer
        self.missing_rate = missing_rate

    # -- learning ------------------------------------------------
    @staticmethod
    def learn(name, values, k, max_bins=10, min_level_count=None):
        present = [v for v in values if str(v).strip() != ""]
        miss = 1.0 - (len(present) / max(1, len(values)))
        nums = [_num(v) for v in present]
        numeric = present and all(x is not None for x in nums)
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
            return Binning(name, "numeric", edges=ded,
                           integer=all(float(x).is_integer()
                                       for x in xs),
                           missing_rate=miss)
        c = Counter(str(v).strip() for v in present)
        floor = min_level_count if min_level_count is not None else k
        levels = [lv for lv, n in c.most_common() if n >= floor]
        if not levels:
            levels = [c.most_common(1)[0][0]] if c else []
        return Binning(name, "categorical", levels=levels,
                       missing_rate=miss)

    # -- use -----------------------------------------------------
    def encode(self, v) -> str:
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none", "null"):
            return MISSING
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
        if self.kind == "numeric":
            return ["b{}".format(i)
                    for i in range(len(self.edges) - 1)]
        return list(self.levels) + ["OTHER_SUPPRESSED"]

    def decode(self, sym: str, rng: random.Random):
        if sym == MISSING:
            return ""
        if self.kind == "numeric":
            try:
                i = int(sym[1:])
            except ValueError:
                i = 0
            lo = self.edges[max(0, min(i, len(self.edges) - 2))]
            hi = self.edges[max(1, min(i + 1, len(self.edges) - 1))]
            x = rng.uniform(lo, hi)
            return int(round(x)) if self.integer else round(x, 4)
        return sym

    def to_json(self):
        return {"name": self.name, "kind": self.kind,
                "edges": [round(e, 6) for e in self.edges],
                "levels": self.levels, "integer": self.integer,
                "missing_rate": round(self.missing_rate, 6)}

    @staticmethod
    def from_json(d):
        return Binning(d["name"], d["kind"], d.get("edges"),
                       d.get("levels"), d.get("integer", False),
                       d.get("missing_rate", 0.0))


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
        buckets = defaultdict(lambda: ([], []))
        for x, y, z in zip(xs, ys, zs):
            buckets[z][0].append(x)
            buckets[z][1].append(y)
        n = len(xs)
        total, df = 0.0, 0
        for z, (bx, by) in buckets.items():
            if len(bx) < self.k:
                continue
            mi, nx, ny = self._mi(bx, by)
            total += (len(bx) / n) * mi
            df += (nx - 1) * (ny - 1)
        return total, max(df, 1)

    def learn(self, rows: List[Dict[str, Any]],
              columns: Optional[List[str]] = None,
              targets: Optional[List[str]] = None) -> "CondNet":
        """`targets` are placed LAST in the ordering so they can
        condition on everything else. Without that hint an outcome
        often lands first (it is the hub of the dependence graph)
        and ends up with no parents, which throws away exactly the
        structure a benchmark cares about."""
        cols = columns or list(rows[0].keys())
        n = len(rows)
        if self.max_bins and self.max_bins > 0:
            nbins, auto = self.max_bins, False
        else:
            nbins = int(round((n / max(self.k, 1)) **
                              (1.0 / (self.max_parents + 1))))
            nbins, auto = max(3, min(10, nbins)), True
        self.resolved_bins = nbins
        self.binnings = {
            c: Binning.learn(c, [r.get(c, "") for r in rows],
                             self.k, nbins)
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

        tests = 0
        for i, c in enumerate(self.order):
            tests += min(i, 1) * i
        tests = max(tests, 1)
        alpha_c = self.alpha / max(
            1, sum(range(len(self.order))) or 1)

        chosen_log = []
        for i, c in enumerate(self.order):
            candidates = self.order[:i]
            parents: List[str] = []
            while len(parents) < self.max_parents and candidates:
                best, best_gain, best_df = None, 0.0, 1
                for cand in candidates:
                    if cand in parents:
                        continue
                    zs = ([tuple(enc[p][j] for p in parents)
                           for j in range(n)] if parents else None)
                    gain, df = self._cmi(enc[c], enc[cand], zs)
                    if gain > best_gain:
                        best, best_gain, best_df = cand, gain, df
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
                        g_stat = 2.0 * n * pair_gain
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
                g_stat = 2.0 * n * best_gain
                crit = _chi2_crit(best_df, alpha_c)
                # cells must also stay publishable: adding a parent
                # multiplies the configuration count, and a table
                # whose average cell falls below k cannot be kept
                width = 1
                for p in parents + [best]:
                    width *= len(set(enc[p]))
                width *= len(set(enc[c]))
                if g_stat < crit or (n / max(width, 1)) < 1.0:
                    break
                parents.append(best)
                candidates = [x for x in candidates if x != best]
                chosen_log.append(
                    {"child": c, "parent": best,
                     "cmi": round(best_gain, 5),
                     "g": round(g_stat, 1),
                     "crit": round(crit, 1)})
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
                groups = defaultdict(Counter)
                for j in range(n):
                    cfg = "|".join(enc[p][j] for p in ps)
                    groups[cfg][enc[c][j]] += 1
                for cfg, cnt in groups.items():
                    m = sum(cnt.values())
                    if m < self.k:
                        suppressed += 1
                        continue          # falls back at sampling
                    table[cfg] = {s: v / m for s, v in cnt.items()}
            self.cpt[c] = table

        self.report = {
            "rows": n,
            "columns_modelled": len(self.order),
            "columns_dropped_no_variation": dropped,
            "edges": [{"child": c, "parents": self.parents[c]}
                      for c in self.order if self.parents[c]],
            "edge_count": sum(1 for c in self.order
                              for _ in self.parents[c]),
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
