"""Score generated data against the source it was profiled from —
on FIDELITY (does it reproduce the patterns?) and on PRIVACY (does
it avoid reproducing the people?).

    python scripts/fidelity_report.py --source tidy.csv
                                      --synthetic generated.csv
                                      [--profile profile.json]
                                      [--report]

FIDELITY compares, column by column and pair by pair:
  numeric marginals      Kolmogorov-Smirnov distance (max gap
                         between the two cumulative distributions)
  categorical marginals  total variation distance (half the summed
                         absolute difference in level shares)
  missingness            absolute difference in missing rate
  joint structure        rank correlation of every confirmed pair,
                         source vs synthetic
  conditional structure  standardized stratum shifts, source vs
                         synthetic

PRIVACY asks the only question that matters: is any synthetic
record too close to a real person? It measures
  exact matches         must be zero
  nearest-neighbor     distance from each synthetic record to its
                        closest source record, compared against the
                        source's OWN internal nearest-neighbor
                        distances. Synthetic records must sit no
                        closer to real records than real records
                        already sit to each other — otherwise the
                        generator is echoing individuals.

A scorecard that cannot fail is worthless, so both halves are
validated against deliberate failures in the smoke suite.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict

LIST_SEP = "; "
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:                       # console-safe on Windows terminals
    from _console import console_safe
    console_safe()
except Exception:
    pass

csv.field_size_limit(1 << 22)

# Tolerances are NOT magic numbers. Two independent draws from the
# same process differ by chance, and how much they differ depends on
# how many rows you drew. So every tolerance here is derived from the
# sampling distribution: the question asked is "is this difference
# larger than a fresh sample of the same process would produce?"
ALPHA_C = 1.63          # two-sample KS critical constant, alpha=.01
Z = 3.0                 # sigma for proportion-based comparisons


def ks_tolerance(n1, n2):
    if n1 < 2 or n2 < 2:
        return 1.0
    return max(0.03, ALPHA_C * math.sqrt((n1 + n2) / (n1 * n2)))


def prop_tolerance(p_, n1, n2):
    if n1 < 2 or n2 < 2:
        return 1.0
    se = math.sqrt(max(p_ * (1 - p_), 0.01)
                   * (1.0 / n1 + 1.0 / n2))
    return max(0.02, Z * se)


def corr_tolerance(n1, n2):
    if n1 < 6 or n2 < 6:
        return 1.0
    se = math.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    return max(0.05, Z * se)


# A generated cell can be "missing" without being empty: the spec's
# mess layer writes tokens like N/A or unknown. Counting only empty
# strings would understate synthetic missingness badly.
MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "-",
                  "--", "unknown", "unk", "missing", "?", ".",
                  "not recorded", "not available"}
BOOL_MAP = {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0,
            "t": 1.0, "f": 0.0}


def is_missing(v):
    return (v or "").strip().lower() in MISSING_TOKENS


def parse_day(v):
    v = (v or "").strip()
    for f in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(v, f).toordinal()
        except ValueError:
            continue
    return None


def num(v):
    v = (v or "").strip()
    if is_missing(v):
        return None
    low = v.lower()
    if low in BOOL_MAP:          # True/False vs 1/0 are the SAME
        return BOOL_MAP[low]     # column in two costumes
    try:
        return float(v.replace(",", "").replace("$", ""))
    except ValueError:
        d = parse_day(v)         # dates compare as day numbers,
        return float(d) if d else None   # never as 600 categories


def read(p):
    with Path(p).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ks_distance(a, b):
    """Max gap between two empirical CDFs."""
    if not a or not b:
        return 1.0
    sa, sb = sorted(a), sorted(b)
    allv = sorted(set(sa) | set(sb))
    i = j = 0
    d = 0.0
    for v in allv:
        while i < len(sa) and sa[i] <= v:
            i += 1
        while j < len(sb) and sb[j] <= v:
            j += 1
        d = max(d, abs(i / len(sa) - j / len(sb)))
    return d


def tvd(a, b):
    ca, cb = Counter(a), Counter(b)
    na, nb = sum(ca.values()) or 1, sum(cb.values()) or 1
    keys = set(ca) | set(cb)
    return 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in keys)


def ranks(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for p, i in enumerate(o):
        r[i] = p
    return r


def spearman(xs, ys):
    if len(xs) < 3:
        return 0.0
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    n = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    d = math.sqrt(sum((a - mx) ** 2 for a in rx)
                  * sum((b - my) ** 2 for b in ry))
    return n / d if d else 0.0


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pctl(xs, p):
    if not xs:
        return 0.0
    ys = sorted(xs)
    return ys[min(len(ys) - 1, max(0, int(round(p * (len(ys) - 1)))))]


# ---------------------------------------------------------------
def nn_distances(A, B, numcols, catcols, stats, sample, seed=5):
    """Distance from each row of A to its nearest row in B."""
    rng = random.Random(seed)
    a = A if len(A) <= sample else rng.sample(A, sample)
    b = B if len(B) <= sample else rng.sample(B, sample)
    out = []
    for ra in a:
        best = None
        for rb in b:
            tot = 0.0
            dims = 0
            for c in numcols:
                x, y = num(ra.get(c)), num(rb.get(c))
                if x is None or y is None:
                    continue
                s = stats[c] or 1.0
                tot += ((x - y) / s) ** 2
                dims += 1
            for c in catcols:
                x = (ra.get(c) or "").strip()
                y = (rb.get(c) or "").strip()
                if not x and not y:
                    continue
                tot += 0.0 if x == y else 1.0
                dims += 1
            if dims == 0:
                continue
            d = math.sqrt(tot / dims)
            if best is None or d < best:
                best = d
                if best == 0.0:
                    break
        if best is not None:
            out.append(best)
    return out




# ===============================================================
# JOINT STRUCTURE BEYOND CORRELATION
# ---------------------------------------------------------------
# Marginals and rank correlations can all match while every
# interesting relationship is gone. A U-shaped dependence has rank
# correlation near zero; an interaction is invisible to any
# pairwise measure. These tests compare the SHAPE of dependence —
# how one column's behavior changes across bins of another, and
# how that changes again across a third — so the scorecard can
# fail for the reasons that actually matter.
# ===============================================================
def _edges(xs, nb):
    """Quantile bin edges from the SOURCE, reused for both sides
    so the comparison is like-for-like."""
    ys = sorted(xs)
    if not ys:
        return []
    eg = [ys[min(len(ys) - 1, int(round(i / nb * (len(ys) - 1))))]
          for i in range(nb + 1)]
    ded = [eg[0]]
    for e in eg[1:]:
        if e > ded[-1]:
            ded.append(e)
    return ded if len(ded) >= 2 else []


def _bin_of(x, edges):
    if x is None or not edges:
        return None
    for i in range(len(edges) - 1):
        if x <= edges[i + 1]:
            return i
    return len(edges) - 2


def _keyer(col, rows, nb):
    """Return a function mapping a row to a bin label for `col`,
    plus the number of distinct labels."""
    vals = [num(r.get(col)) for r in rows]
    numeric = sum(1 for v in vals if v is not None) > 0.9 * len(
        [r for r in rows if not is_missing(r.get(col))])
    if numeric:
        eg = _edges([v for v in vals if v is not None], nb)
        if not eg:
            return None, 0
        return (lambda r: _bin_of(num(r.get(col)), eg)), \
            len(eg) - 1
    levels = [lv for lv, _ in Counter(
        (r.get(col) or "").strip() for r in rows
        if not is_missing(r.get(col))).most_common(nb)]
    if len(levels) < 2:
        return None, 0
    lut = set(levels)
    return (lambda r: ((r.get(col) or "").strip()
                       if (r.get(col) or "").strip() in lut
                       else None)), len(levels)


def _profile(rows, keyfn, target, positive=None):
    """mean of `target` within each bin of the key column."""
    acc = defaultdict(list)
    for r in rows:
        b = keyfn(r)
        if b is None:
            continue
        if positive is not None:
            v = 1.0 if (r.get(target) or "").strip() == positive \
                else 0.0
        else:
            v = num(r.get(target))
            if v is None:
                continue
        acc[b].append(v)
    return {b: (len(v), sum(v) / len(v)) for b, v in acc.items()
            if len(v) >= 8}


def _shape(prof, scale=1.0):
    """Is this profile monotone, or does it turn? A turn must beat
    SAMPLING NOISE, not a fixed number: with sixty rows a bin, the
    means wobble by chance and any profile can look like it turns.
    Requiring the swing to exceed three standard errors is what
    keeps the count of `turning` relationships honest — a fixed sd
    threshold reported 268 of them in data drawn independently."""
    """monotone, or does the profile turn? A turning profile is
    exactly what rank correlation cannot represent."""
    keys = sorted(prof.keys(), key=lambda k: str(k))
    ys = [prof[k][1] for k in keys]
    if len(ys) < 3:
        return "flat"
    ups = sum(1 for i in range(len(ys) - 1) if ys[i + 1] > ys[i])
    downs = len(ys) - 1 - ups
    ns = [prof[k][0] for k in keys]
    span_sd = (max(ys) - min(ys)) / (scale or 1.0)
    hi_i = ys.index(max(ys))
    lo_i = ys.index(min(ys))
    se = math.sqrt(1.0 / max(ns[hi_i], 1) + 1.0 / max(ns[lo_i], 1))
    if span_sd < max(0.15, 3.0 * se):
        # a profile that barely moves is not a relationship;
        # calling it "turning" would flood the report with noise
        return "flat"
    if ups and downs:
        span = max(ys) - min(ys)
        interior = max(ys[1:-1]) if len(ys) > 2 else ys[0]
        if span > 0 and (interior >= max(ys) - 1e-9
                         or min(ys[1:-1]) <= min(ys) + 1e-9):
            return "turning"
    return "monotone"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--profile", default="")
    ap.add_argument("-o", "--out", default="fidelity_report.json")
    ap.add_argument("--nn-sample", type=int, default=400)
    ap.add_argument("--group-by", default="",
                    help="column identifying the PATIENT. "
                         "Repeated visits from one person are not "
                         "independent draws, so tolerances are "
                         "computed from the number of people "
                         "rather than the number of rows.")
    ap.add_argument("--target", default="",
                    help="outcome column: interactions affecting "
                         "it are tested explicitly")
    ap.add_argument("--deep-bins", type=int, default=4)
    ap.add_argument("--max-pairs", type=int, default=400)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    S, G = read(a.source), read(a.synthetic)
    if not S or not G:
        sys.exit("empty input")
    shared = [c for c in S[0] if c in G[0]
              if c != a.group_by]

    # Effective sample size. Repeated visits from one patient are
    # not independent draws, so a tolerance derived from the row
    # count would fail honest data for having exactly the
    # clustered structure it was asked to reproduce.
    def _eff(rowset):
        if not a.group_by or not rowset \
                or a.group_by not in rowset[0]:
            return len(rowset)
        return max(len({r.get(a.group_by) for r in rowset}), 1)

    eff_s, eff_g = _eff(S), _eff(G)

    numcols, catcols = [], []
    for c in shared:
        vals = [num(r.get(c)) for r in S
                if not is_missing(r.get(c))]
        vals = [v for v in vals if v is not None]
        present = sum(1 for r in S if not is_missing(r.get(c)))
        if present and len(vals) / present > 0.9:
            numcols.append(c)
        elif present:
            catcols.append(c)

    marg, fails = [], 0
    for c in numcols:
        xs = [num(r.get(c)) for r in S]
        ys = [num(r.get(c)) for r in G]
        xs = [x for x in xs if x is not None]
        ys = [y for y in ys if y is not None]
        d = ks_distance(xs, ys)
        tol = ks_tolerance(
            max(int(len(xs) * eff_s / max(len(S), 1)), 2),
            max(int(len(ys) * eff_g / max(len(G), 1)), 2))
        ok = d <= tol
        fails += 0 if ok else 1
        marg.append({"column": c, "type": "numeric", "metric": "KS",
                     "value": round(d, 4),
                     "tolerance": round(tol, 4),
                     "pass": ok,
                     "source_mean": round(mean(xs), 3),
                     "synth_mean": round(mean(ys), 3),
                     "source_sd": round(sd(xs), 3),
                     "synth_sd": round(sd(ys), 3)})
    for c in catcols:
        xs = [(r.get(c) or "").strip() for r in S
              if not is_missing(r.get(c))]
        ys = [(r.get(c) or "").strip() for r in G
              if not is_missing(r.get(c))]

        # A LIST column cannot match as a whole string, and should
        # not. Its combinations are near-unique per visit, and only
        # items shared by enough patients are ever published — the
        # rare ones are dropped deliberately, because a
        # combination unique to one person identifies them. Judging
        # such a column on whole-string agreement reports a
        # failure for the privacy rule doing its job, and a
        # scorecard that cries wolf teaches people to ignore it.
        #
        # The meaningful comparison is at the ITEM level: does each
        # item appear at about the right rate?
        listish = (sum(1 for v in xs[:400] if LIST_SEP in v)
                   > 0.15 * max(len(xs[:400]), 1))
        if listish:
            def items_of(vals):
                out = Counter()
                for v in vals:
                    for it in {t.strip() for t in v.split(LIST_SEP)
                               if t.strip()}:
                        out[it] += 1
                return out
            si, gi = items_of(xs), items_of(ys)
            common = [i for i in si if si[i] >= 0.02 * len(xs)]
            worst, worst_item = 0.0, ""
            for it in common:
                dd = abs(si[it] / max(len(xs), 1)
                         - gi.get(it, 0) / max(len(ys), 1))
                if dd > worst:
                    worst, worst_item = dd, it
            ltol = prop_tolerance(0.2, len(xs), len(ys))
            ok = worst <= ltol
            fails += 0 if ok else 1
            marg.append({
                "column": c, "type": "list", "metric":
                "worst item rate difference",
                "value": round(worst, 4),
                "tolerance": round(ltol, 4), "pass": ok,
                "worst_item": worst_item,
                "items_compared": len(common),
                "source_items": len(si), "synth_items": len(gi),
                "note": "compared item by item, not as whole "
                        "strings: rare combinations are suppressed "
                        "on purpose, so exact agreement on the "
                        "full list would mean the privacy rule "
                        "had failed"})
            continue
        d = tvd(xs, ys)
        big = max((Counter(xs).most_common(1) or [("", 0)])[0][1]
                  / max(1, len(xs)), 0.05)
        tol = prop_tolerance(big, len(xs), len(ys))
        ok = d <= tol
        fails += 0 if ok else 1
        marg.append({"column": c, "type": "categorical",
                     "metric": "TVD", "value": round(d, 4),
                     "tolerance": round(tol, 4), "pass": ok,
                     "source_levels": len(set(xs)),
                     "synth_levels": len(set(ys))})
    miss = []
    for c in shared:
        ms = sum(1 for r in S if is_missing(r.get(c))) / len(S)
        mg = sum(1 for r in G if is_missing(r.get(c))) / len(G)
        d = abs(ms - mg)
        tol = prop_tolerance((ms + mg) / 2 or 0.02, len(S), len(G))
        ok = d <= tol
        fails += 0 if ok else 1
        miss.append({"column": c, "source_rate": round(ms, 4),
                     "synth_rate": round(mg, 4),
                     "delta": round(d, 4),
                     "tolerance": round(tol, 4), "pass": ok})

    # joint structure
    pairs = []
    if a.profile:
        P = json.loads(Path(a.profile).read_text(encoding="utf-8"))
        conf = [c for c in P["joint"]["numeric_correlations"]
                if c.get("confirmed")]
    else:
        conf = []
        for i in range(len(numcols)):
            for j in range(i + 1, len(numcols)):
                conf.append({"a": numcols[i], "b": numcols[j]})
    for pr in conf:
        ca, cb = pr["a"], pr["b"]
        if ca not in shared or cb not in shared:
            continue

        def both(rows):
            out = [(num(r.get(ca)), num(r.get(cb))) for r in rows]
            return [(x, y) for x, y in out
                    if x is not None and y is not None]
        ps, pg = both(S), both(G)
        if len(ps) < 10 or len(pg) < 10:
            continue
        rs = spearman([x for x, _ in ps], [y for _, y in ps])
        rg = spearman([x for x, _ in pg], [y for _, y in pg])
        d = abs(rs - rg)
        # A redundant (near-perfect) pair was DECLINED at compile
        # time — the draft says to derive one column from the other
        # with a rule instead. Its absence is a design decision, so
        # it is reported as OWED rather than scored as a failure.
        declined = pr.get("redundant", False)
        ctol = corr_tolerance(len(ps), len(pg))
        ok = d <= ctol
        if not declined:
            fails += 0 if ok else 1
        pairs.append({"a": ca, "b": cb, "source": round(rs, 3),
                      "synthetic": round(rg, 3),
                      "delta": round(d, 3),
                      "tolerance": round(ctol, 3),
                      "pass": ok if not declined else None,
                      "declined_as_redundant": declined,
                      "note": ("not imposed by design — add a "
                               "derived rule to restore it"
                               if declined else "")})

    strata = []
    if a.profile:
        P = json.loads(Path(a.profile).read_text(encoding="utf-8"))
        for s in P["joint"]["conditional_shifts"]:
            if not s.get("confirmed"):
                continue
            c, g, lv = s["column"], s["given"], s["level"]
            if c not in shared or g not in shared:
                continue

            def shift(rows):
                base = [num(r.get(c)) for r in rows]
                base = [x for x in base if x is not None]
                if len(base) < 10 or sd(base) == 0:
                    return None
                sub = [num(r.get(c)) for r in rows
                       if (r.get(g) or "").strip() == lv]
                sub = [x for x in sub if x is not None]
                if len(sub) < 5:
                    return None
                return (mean(sub) - mean(base)) / sd(base)
            ss, sg = shift(S), shift(G)
            if ss is None or sg is None:
                continue
            d = abs(ss - sg)
            stol = max(0.15, Z * math.sqrt(1.0 / max(len(S), 1)
                                           + 1.0 / max(len(G), 1))
                       * 4)
            ok = d <= stol
            fails += 0 if ok else 1
            strata.append({"column": c, "given": g, "level": lv,
                           "source": round(ss, 3),
                           "synthetic": round(sg, 3),
                           "delta": round(d, 3), "pass": ok})

    # ------------- shape of dependence (nonlinear) -------------
    nb = a.deep_bins
    shaped, nonlinear_found = [], 0
    tested_pairs = 0
    skipped_noise = 0
    cand_t = [c for c in numcols if c in shared]
    cat_t = [c for c in catcols if c in shared]
    for src_col in shared:
        if tested_pairs >= a.max_pairs:
            break
        kf, klen = _keyer(src_col, S, nb)
        if kf is None:
            continue
        for tgt in cand_t:
            if tgt == src_col or tested_pairs >= a.max_pairs:
                continue
            ps = _profile(S, kf, tgt)
            if len(ps) < 3:
                continue
            pg = _profile(G, kf, tgt)
            if len(pg) < 3:
                continue
            base = [num(r.get(tgt)) for r in S]
            base = [x for x in base if x is not None]
            scale = sd(base) or 1.0
            common = [b for b in ps if b in pg]
            if len(common) < 3:
                continue
            # Only score a profile whose SOURCE shape is real. If
            # the source's own variation across bins is within
            # sampling noise, there is no relationship there — and
            # demanding that synthetic data reproduce noise is not
            # a fidelity test, it is a coin flip counted as a
            # failure. The profiler refuses to IMPOSE unconfirmed
            # structure; the scorecard refuses to SCORE it.
            sy = [ps[b][1] for b in common]
            sn = [ps[b][0] for b in common]
            src_span = (max(sy) - min(sy)) / scale
            src_se = math.sqrt(
                1.0 / max(sn[sy.index(max(sy))], 1)
                + 1.0 / max(sn[sy.index(min(sy))], 1))
            if src_span < 3.0 * src_se:
                skipped_noise += 1
                continue
            dev = max(abs(ps[b][1] - pg[b][1]) / scale
                      for b in common)
            tol = 3.0 * max(
                math.sqrt(1.0 / min(ps[b][0] for b in common)
                          + 1.0 / min(pg[b][0] for b in common)),
                0.05)
            tested_pairs += 1
            shape = _shape(ps, scale)
            ok = dev <= tol
            if shape == "turning":
                nonlinear_found += 1
            if shape == "turning" or not ok:
                fails += 0 if ok else 1
                shaped.append({
                    "given": src_col, "column": tgt,
                    "source_shape": shape,
                    "synthetic_shape": _shape(pg, scale),
                    "max_deviation_sd": round(dev, 3),
                    "tolerance": round(tol, 3), "pass": ok,
                    "source_profile": [round(ps[b][1], 3)
                                       for b in sorted(
                                           common,
                                           key=lambda x: str(x))],
                    "synthetic_profile": [round(pg[b][1], 3)
                                          for b in sorted(
                                              common,
                                              key=lambda x: str(x))
                                          ]})

    # ---------------- interactions -----------------------------
    inter = []
    if a.target and a.target in shared:
        tgt = a.target
        pos = None
        tv = [num(r.get(tgt)) for r in S]
        if sum(1 for v in tv if v is not None) < 0.9 * len(S):
            c0 = Counter((r.get(tgt) or "").strip() for r in S
                         if not is_missing(r.get(tgt)))
            pos = c0.most_common()[-1][0] if len(c0) >= 2 else None
        pool = [c for c in shared if c != tgt][:14]
        for i in range(len(pool)):
            for j in range(len(pool)):
                if i == j:
                    continue
                cx, cz = pool[i], pool[j]
                kx, _ = _keyer(cx, S, nb)
                kz, _ = _keyer(cz, S, 2)
                if kx is None or kz is None:
                    continue
                devs = []
                shapes_s = []
                for zval in (0, 1):
                    sel_s = [r for r in S if kz(r) == zval]
                    sel_g = [r for r in G if kz(r) == zval]
                    if len(sel_s) < 40 or len(sel_g) < 40:
                        continue
                    p_s = _profile(sel_s, kx, tgt, pos)
                    p_g = _profile(sel_g, kx, tgt, pos)
                    common = [b for b in p_s if b in p_g]
                    if len(common) < 3:
                        continue
                    base = [(1.0 if (r.get(tgt) or "").strip()
                             == pos else 0.0) if pos
                            else num(r.get(tgt)) for r in S]
                    base = [x for x in base if x is not None]
                    sc = sd(base) or 1.0
                    devs.append(max(abs(p_s[b][1] - p_g[b][1]) / sc
                                    for b in common))
                    shapes_s.append(
                        (max(p_s[b][1] for b in common)
                         - min(p_s[b][1] for b in common)) / sc)
                if len(devs) == 2 and len(shapes_s) == 2:
                    # an interaction exists when the effect of cx
                    # on the target has a materially different SIZE
                    # in the two halves of cz
                    gap = abs(shapes_s[0] - shapes_s[1])
                    if gap < 0.25:
                        continue
                    ok = max(devs) <= 0.35
                    fails += 0 if ok else 1
                    inter.append({
                        "target": tgt, "varies_with": cx,
                        "moderated_by": cz,
                        "effect_size_low_stratum": round(
                            shapes_s[0], 3),
                        "effect_size_high_stratum": round(
                            shapes_s[1], 3),
                        "max_deviation_sd": round(max(devs), 3),
                        "pass": ok})
        inter.sort(key=lambda d: -abs(
            d["effect_size_high_stratum"]
            - d["effect_size_low_stratum"]))
        inter = inter[:12]

    # ---------------- privacy ----------------
    keys = [c for c in shared]
    sig_s = Counter(tuple((r.get(c) or "").strip() for c in keys)
                    for r in S)
    exact = sum(1 for r in G
                if tuple((r.get(c) or "").strip() for c in keys)
                in sig_s)
    stats = {c: (sd([num(r.get(c)) for r in S
                     if num(r.get(c)) is not None]) or 1.0)
             for c in numcols}
    d_gs = nn_distances(G, S, numcols, catcols, stats, a.nn_sample)
    d_ss = nn_distances(S, S, numcols, catcols, stats, a.nn_sample,
                        seed=6)
    # source-to-source includes the record itself at distance 0;
    # strip one zero per row so we compare like with like
    d_ss = sorted(d_ss)
    d_ss = d_ss[sum(1 for d in d_ss if d == 0.0):] or d_ss
    # Calibrated closeness test. If the synthetic set behaves like an
    # independent draw, about 5% of its records should fall below
    # the source's own 5th-percentile nearest-neighbor distance.
    # Substantially more than that means the generator is hugging
    # real individuals — the interpolation failure mode.
    cut = pctl(d_ss, 0.05) if d_ss else 0.0
    below = (sum(1 for d in d_gs if d < cut) / len(d_gs)
             if d_gs else 0.0)
    priv_ok = exact == 0 and below <= 0.15
    privacy = {
        "exact_matches": exact,
        "exact_matches_pass": exact == 0,
        "synthetic_to_source_nn": {
            "min": round(min(d_gs), 4) if d_gs else None,
            "p05": round(pctl(d_gs, 0.05), 4) if d_gs else None,
            "median": round(pctl(d_gs, 0.5), 4) if d_gs else None},
        "source_internal_nn": {
            "min": round(min(d_ss), 4) if d_ss else None,
            "p05": round(pctl(d_ss, 0.05), 4) if d_ss else None,
            "median": round(pctl(d_ss, 0.5), 4) if d_ss else None},
        "closeness_test": {
            "source_p05_distance": round(cut, 4),
            "synthetic_fraction_below": round(below, 4),
            "expected_if_independent": 0.05,
            "fails_above": 0.15},
        "verdict": "PASS" if priv_ok else "FAIL",
        "reading": "synthetic records must sit no closer to real "
                   "records than real records sit to each other",
    }

    total = (len(marg) + len(miss) + len(pairs) + len(strata)
             + len(shaped) + len(inter))
    report = {
        "source": {"file": Path(a.source).name, "rows": len(S)},
        "synthetic": {"file": Path(a.synthetic).name,
                      "rows": len(G)},
        "tolerance_policy": "derived from the sampling distribution "
                            "at each column's own sample size — a "
                            "difference fails only if it exceeds "
                            "what a fresh draw of the same process "
                            "would produce",
        "summary": {"checks": total, "failed": fails,
                    "passed": total - fails,
                    "fidelity_verdict": "PASS" if fails == 0
                    else "FAIL",
                    "privacy_verdict": privacy["verdict"]},
        "marginals": marg, "missingness": miss,
        "correlations": pairs, "conditional_shifts": strata,
        "dependence_shape": {
            "pairs_tested": tested_pairs,
            "pairs_skipped_as_noise": skipped_noise,
            "nonlinear_relationships_in_source": nonlinear_found,
            "detail": shaped[:60],
            "reading": "a `turning` profile cannot be represented "
                       "by a rank correlation at all — these are "
                       "the relationships a marginals-and-"
                       "correlations generator silently destroys"},
        "interactions": {
            "target": a.target,
            "detail": inter,
            "reading": "an interaction is present when one "
                       "column's effect on the target has a "
                       "different SIZE in different strata of a "
                       "third column"},
        "privacy": privacy,
    }
    Path(a.out).write_text(json.dumps(report, indent=1),
                           encoding="utf-8")
    print("WROTE {} : {}/{} fidelity checks passed | privacy {}"
          .format(a.out, total - fails, total, privacy["verdict"]))

    if a.report:
        print("\n=== FIDELITY: DOES THE SYNTHETIC DATA CARRY THE "
              "SAME PATTERNS? ===")
        print("\n-- each column's own distribution --")
        for m in marg:
            mark = "ok " if m["pass"] else "OFF"
            if m["type"] == "numeric":
                print("  {} {:26s} KS {:.3f} (<= {:.2f})   mean "
                      "{:.2f} vs {:.2f}   sd {:.2f} vs {:.2f}"
                      .format(mark, m["column"], m["value"],
                              m["tolerance"], m["source_mean"],
                              m["synth_mean"], m["source_sd"],
                              m["synth_sd"]))
            elif m["type"] == "list":
                print("  {} {:26s} worst item rate diff {:.3f} "
                      "(<= {:.2f})   {} items compared, {} of {} "
                      "published".format(
                          mark, m["column"], m["value"],
                          m["tolerance"], m["items_compared"],
                          m["synth_items"], m["source_items"]))
            else:
                print("  {} {:26s} TVD {:.3f} (<= {:.2f})   levels "
                      "{} vs {}".format(
                          mark, m["column"], m["value"],
                          m["tolerance"], m["source_levels"],
                          m["synth_levels"]))
        bad_miss = [m for m in miss if not m["pass"]]
        print("\n-- missingness -- {} of {} columns within "
              "sampling tolerance".format(
                  len(miss) - len(bad_miss), len(miss)))
        for m in bad_miss[:6]:
            print("  OFF {:26s} source {:.1%} vs synthetic {:.1%}"
                  .format(m["column"], m["source_rate"],
                          m["synth_rate"]))
        print("\n-- how the fields move together --")
        if pairs:
            for pr in pairs:
                mark = ("OWED" if pr["declined_as_redundant"]
                        else ("ok  " if pr["pass"] else "OFF "))
                print("  {}{:22s} <-> {:22s} rho {:+.3f} vs "
                      "{:+.3f}  (delta {:.3f}){}".format(
                          mark, pr["a"], pr["b"], pr["source"],
                          pr["synthetic"], pr["delta"],
                          "  " + pr["note"] if pr["note"] else ""))
        else:
            print("  no confirmed correlations to reproduce")
        if strata:
            print("\n-- conditional structure --")
            for s in strata:
                print("  {} {:20s} given {} = {:18s} {:+.2f} vs "
                      "{:+.2f} sd".format(
                          "ok " if s["pass"] else "OFF",
                          s["column"], s["given"], s["level"],
                          s["source"], s["synthetic"]))
        print("\n-- shape of dependence (what correlation "
              "cannot see) --")
        print("  {} relationships real enough to score ({} more "
              "were within sampling noise and skipped); {} of "
              "them TURN".format(tested_pairs, skipped_noise,
                                 nonlinear_found))
        if shaped:
            for d in shaped[:8]:
                print("  {} {:20s} across {:18s} [{}] source {} "
                      "-> synth {}".format(
                          "ok " if d["pass"] else "OFF",
                          d["column"], d["given"],
                          d["source_shape"],
                          "/".join("{:.2f}".format(x) for x in
                                   d["source_profile"][:5]),
                          "/".join("{:.2f}".format(x) for x in
                                   d["synthetic_profile"][:5])))
        else:
            print("  every profiled relationship is monotone and "
                  "reproduced within tolerance")
        if a.target:
            print("\n-- interactions on `{}` --".format(a.target))
            if inter:
                for d in inter[:6]:
                    print("  {} effect of {:18s} differs across "
                          "{:18s} ({:.2f} vs {:.2f} sd) — "
                          "deviation {:.2f}".format(
                              "ok " if d["pass"] else "OFF",
                              d["varies_with"], d["moderated_by"],
                              d["effect_size_low_stratum"],
                              d["effect_size_high_stratum"],
                              d["max_deviation_sd"]))
            else:
                print("  no interaction of material size found in "
                      "the source")

        print("\n=== PRIVACY: DOES IT AVOID REPRODUCING THE "
              "PEOPLE? ===")
        print("  exact record matches: {}  ({})".format(
            exact, "none — required" if exact == 0
            else "LEAK — a synthetic row equals a real row"))
        if d_gs and d_ss:
            print("  nearest-neighbor distance to a real record")
            print("      synthetic -> source   min {:.3f}  p05 "
                  "{:.3f}  median {:.3f}".format(
                      min(d_gs), pctl(d_gs, .05), pctl(d_gs, .5)))
            print("      source -> source      min {:.3f}  p05 "
                  "{:.3f}  median {:.3f}".format(
                      min(d_ss), pctl(d_ss, .05), pctl(d_ss, .5)))
            print("      {:.1%} of synthetic records fall below the "
                  "source's own 5th-percentile distance "
                  "({:.3f})".format(below, cut))
            print("      an independent draw would put ~5% there; "
                  "above 15% means the generator is hugging real "
                  "individuals")
        print("\n  FIDELITY {}   PRIVACY {}".format(
            report["summary"]["fidelity_verdict"],
            privacy["verdict"]))


if __name__ == "__main__":
    main()
