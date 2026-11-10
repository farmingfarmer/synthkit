"""Total variation distance between the source and the synthetic.

WHY THIS EXISTS, BESIDE THE GATE. Every pair criterion here is a
PASS/FAIL on a correlation delta over the pairs the SOURCE relates
at 0.1 or more. That has two costs. It is blind to the pairs the
source does NOT relate - a generator that invents structure where
there was none is not penalised - and it is a threshold, so it
moves in steps while the thing underneath moves continuously. The
real extract sits 0.3 points from one bar on a measure whose own
seed spread is four points, which makes that verdict close to a
coin flip.

TVD is what the commercial tools report. MOSTLY AI's accuracy is
the mean of univariate and bivariate (and trivariate) total
variation distance over DISCRETISED empirical distributions, across
ALL pairs rather than only related ones; SDV's quality report uses
contingency similarity for categorical pairs, which is the same
quantity. Reporting it puts our output on the axis a buyer will
compare it on, and gives a number that moves when the engine moves.

    TVD = 0.5 * sum |p_source - p_synthetic|

over the cells of a discretised distribution. 0 is identical, 1 is
disjoint. `accuracy = 1 - TVD` is the form those tools publish.

WHAT THIS IS NOT. It is REPORTED, not gated. Turning it into a
criterion would change release verdicts on a threshold nobody has
measured a basis for, and this project does not move a bar and a
number in the same commit.

NO CELL IS PUBLISHED. The report carries aggregate distances and
column names - never a contingency table, whose cells are counts of
people. A bin holding three patients describes those three, which
is the same reason the deck suppresses sparse source bins.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ten bins per numeric column: enough to see a shifted mode, few
# enough that a 55,000-row table still has hundreds of rows per
# cell in a pairwise grid.
N_BINS = 10
# A categorical with more levels than this is folded to its top
# ones plus an `other` - a 2,000-level code column would otherwise
# make every cell a singleton and every TVD read ~1.0, which is a
# statement about cardinality rather than fidelity.
MAX_LEVELS = 20
MIN_ROWS = 50


def _edges(s: pd.Series) -> Optional[np.ndarray]:
    v = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    if v.size < MIN_ROWS:
        return None
    qs = np.unique(np.quantile(v, np.linspace(0, 1, N_BINS + 1)))
    return qs if len(qs) >= 3 else None


def _codes(s: pd.Series, spec) -> np.ndarray:
    """One integer per row: the bin it falls in, -1 for missing.

    THE EDGES COME FROM THE SOURCE and are applied to BOTH tables.
    Binning each table by its own quantiles would compare a
    source decile against a synthetic decile and call two
    differently-shaped columns identical - the comparison has to
    be on one ruler.
    """
    kind, ref = spec
    if kind == "num":
        v = pd.to_numeric(s, errors="coerce").to_numpy(float)
        out = np.full(len(v), -1, dtype=int)
        ok = np.isfinite(v)
        out[ok] = np.clip(np.digitize(v[ok], ref[1:-1]), 0,
                          len(ref) - 2)
        return out
    txt = s.astype(str).fillna("")
    out = np.full(len(txt), len(ref), dtype=int)   # the `other` bin
    pos = dict((lv, i) for i, lv in enumerate(ref))
    vals = txt.to_numpy()
    for i, x in enumerate(vals):
        j = pos.get(x)
        if j is not None:
            out[i] = j
    return out


def _spec(s: pd.Series):
    """How a column is discretised, decided on the SOURCE."""
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() >= 0.5:
        e = _edges(s)
        return ("num", e) if e is not None else None
    lv = (s.astype(str).fillna("").value_counts()
          .index[:MAX_LEVELS].tolist())
    return ("cat", lv) if len(lv) >= 2 else None


def _tvd_1d(a: np.ndarray, b: np.ndarray, nb: int) -> float:
    pa = np.bincount(a[a >= 0], minlength=nb).astype(float)
    pb = np.bincount(b[b >= 0], minlength=nb).astype(float)
    if pa.sum() <= 0 or pb.sum() <= 0:
        return float("nan")
    pa /= pa.sum()
    pb /= pb.sum()
    return float(0.5 * np.abs(pa - pb).sum())


def _tvd_2d(a1, a2, b1, b2, n1, n2) -> float:
    ok_a = (a1 >= 0) & (a2 >= 0)
    ok_b = (b1 >= 0) & (b2 >= 0)
    if ok_a.sum() < MIN_ROWS or ok_b.sum() < MIN_ROWS:
        return float("nan")
    ha = np.bincount(a1[ok_a] * n2 + a2[ok_a],
                     minlength=n1 * n2).astype(float)
    hb = np.bincount(b1[ok_b] * n2 + b2[ok_b],
                     minlength=n1 * n2).astype(float)
    ha /= ha.sum()
    hb /= hb.sum()
    return float(0.5 * np.abs(ha - hb).sum())


def measure(Xs: pd.DataFrame, Xg: pd.DataFrame,
            skip=None, max_pairs: int = 4000) -> Dict[str, Any]:
    """Univariate and bivariate TVD over the columns BOTH carry.

    `skip` drops scaffolding and the grouping key - search columns
    are not the operator's data, and this project already counts
    them apart everywhere else.
    """
    skip = set(skip or ())
    cols = [c for c in Xs.columns
            if c in Xg.columns and c not in skip]
    specs, codes_s, codes_g, nbins = {}, {}, {}, {}
    for c in cols:
        sp = _spec(Xs[c])
        if sp is None:
            continue
        specs[c] = sp
        codes_s[c] = _codes(Xs[c], sp)
        codes_g[c] = _codes(Xg[c], sp)
        nbins[c] = (len(sp[1]) - 1 if sp[0] == "num"
                    else len(sp[1]) + 1)
    use = sorted(specs)
    uni = []
    for c in use:
        d = _tvd_1d(codes_s[c], codes_g[c], nbins[c])
        if np.isfinite(d):
            uni.append({"column": c, "tvd": round(d, 5)})

    bi: List[Dict[str, Any]] = []
    n = len(use)
    budget = max_pairs
    for i in range(n):
        for j in range(i + 1, n):
            if budget <= 0:
                break
            a, b = use[i], use[j]
            d = _tvd_2d(codes_s[a], codes_s[b], codes_g[a],
                        codes_g[b], nbins[a], nbins[b])
            budget -= 1
            if np.isfinite(d):
                bi.append({"a": a, "b": b, "tvd": round(d, 5)})
        if budget <= 0:
            break
    skipped = max(0, (n * (n - 1)) // 2 - len(bi))

    def agg(rows, key="tvd"):
        v = [r[key] for r in rows]
        return (float(np.mean(v)) if v else float("nan"),
                float(np.max(v)) if v else float("nan"))

    u_mean, u_max = agg(uni)
    b_mean, b_max = agg(bi)
    worst = sorted(bi, key=lambda r: -r["tvd"])[:15]
    return {
        "columns_measured": len(use),
        "univariate": {
            "mean_tvd": None if not np.isfinite(u_mean)
            else round(u_mean, 5),
            "worst_tvd": None if not np.isfinite(u_max)
            else round(u_max, 5),
            "accuracy": None if not np.isfinite(u_mean)
            else round(1.0 - u_mean, 5),
            "n": len(uni),
            "worst": sorted(uni, key=lambda r: -r["tvd"])[:10],
        },
        "bivariate": {
            "mean_tvd": None if not np.isfinite(b_mean)
            else round(b_mean, 5),
            "worst_tvd": None if not np.isfinite(b_max)
            else round(b_max, 5),
            "accuracy": None if not np.isfinite(b_mean)
            else round(1.0 - b_mean, 5),
            "n": len(bi),
            "pairs_not_measured": int(skipped),
            "worst": worst,
        },
        # The headline the commercial tools publish, so ours can be
        # put beside theirs without a translation step.
        "accuracy": None if not (np.isfinite(u_mean)
                                 and np.isfinite(b_mean))
        else round(1.0 - 0.5 * (u_mean + b_mean), 5),
        "note": "total variation distance on discretised "
                "distributions, over EVERY pair of the operator's "
                "columns - not only the pairs the source relates. "
                "0 is identical. `accuracy` is 1 - TVD, the form "
                "MOSTLY AI and SDV report. REPORTED, NOT GATED: "
                "turning it into a criterion would move release "
                "verdicts on a threshold nobody has measured a "
                "basis for. No contingency cell is published - "
                "the cells are counts of people.",
    }
