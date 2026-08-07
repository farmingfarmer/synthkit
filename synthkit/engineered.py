"""Manufacture the candidate features a pairwise search cannot see,
without manufacturing all of them.

THE PROBLEM. A pure interaction - one whose factors carry no effect on
their own - is invisible to a pairwise-first search at any sample size.
condnet does have a pair path for exactly this case, but its pool is
`candidates[:8]` and `self.order` is sorted by TOTAL PAIRWISE
DEPENDENCE. A factor with no marginal signal sorts last and is never
in the pool. The one mechanism built to catch interactions selects its
candidates by the statistic that is blind to them.

WHY NOT JUST MAKE EVERY PRODUCT. At 173 columns that is 14,878
features, each multiplying the comparison count the correction is paid
over. The combinatorial wall is real and no selection rule removes it;
what a rule can do is spend a bounded budget where the blind spot
actually is.

THE RULE. Learn once. Whatever has NO edge after that pass is, by
definition, what the pairwise search could not explain - which is
precisely where a pure interaction hides. Manufacture centred products
among those columns only, learn again, and let split-sample
confirmation referee the additions.

Centred, because that is what makes an exclusive-or visible:
(a - median_a) * (b - median_b) is negative exactly when one factor is
high and the other low. An uncentred product cannot express it.

COVERAGE IS PARTIAL AND SAID SO. This finds interactions among
unexplained columns. An interaction between two columns that each have
some other relationship will still be missed, and the report says how
many pairs were tried against how many exist.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

PRODUCT_SEP = "__x__"


def _num(v):
    try:
        return float(str(v).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def _blank(v):
    s = str(v or "").strip()
    return (not s) or s.lower() in ("nan", "none", "null")


def unexplained_columns(net, rows, min_coverage=0.10):
    """Columns with no edge after a pass, and enough data to carry one.

    These are what the search could not explain. A pure interaction's
    factors are always here, because neither has a marginal signal to
    be explained by."""
    touched = set()
    for e in net.report.get("edges", []):
        touched.add(e.get("child"))
        touched.update(e.get("parents") or [])
    # A DERIVED column and its determinant are explained too, and they
    # never appear as an edge - a column deterministic in another is
    # recorded as arithmetic rather than as a finding. Missing that
    # put a pair with a correlation of 0.98 into the unexplained set
    # and manufactured a product from it.
    for c, src in (getattr(net, "derived", {}) or {}).items():
        touched.add(c)
        base = str(src).split(" (")[0]
        touched.add(base)
    for c, pair in (getattr(net, "pair_derived", {}) or {}).items():
        touched.add(c)
        touched.update(pair)
    out = []
    n = len(rows)
    for c in (rows[0] if rows else {}):
        if c in touched or PRODUCT_SEP in c or c.endswith("__prev"):
            continue
        present = [r.get(c) for r in rows if not _blank(r.get(c))]
        if len(present) < min_coverage * n:
            continue
        if any(_num(v) is None for v in present[:2000]):
            continue
        if len(set(_num(v) for v in present[:2000])) < 3:
            continue
        out.append(c)
    return sorted(out)


def add_product_features(rows: List[Dict[str, Any]],
                         columns: List[str],
                         budget: int = 400,
                         ) -> Tuple[List[Dict[str, Any]],
                                    Dict[str, Any]]:
    """Centred pairwise products among `columns`, up to `budget`."""
    cols = [c for c in columns if c]
    possible = len(cols) * (len(cols) - 1) // 2
    if not cols or possible == 0:
        return rows, {"added": 0, "pairs_possible": 0,
                      "pairs_tried": 0,
                      "note": "no unexplained columns to pair"}

    med = {}
    for c in cols:
        vals = sorted(_num(r.get(c)) for r in rows
                      if not _blank(r.get(c)))
        vals = [v for v in vals if v is not None]
        med[c] = vals[len(vals) // 2] if vals else 0.0

    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j]))
    truncated = len(pairs) > budget
    pairs = pairs[:budget]

    out = []
    for r in rows:
        r2 = dict(r)
        for a, b in pairs:
            va, vb = _num(r.get(a)), _num(r.get(b))
            r2[a + PRODUCT_SEP + b] = (
                "" if (va is None or vb is None)
                else round((va - med[a]) * (vb - med[b]), 6))
        out.append(r2)
    return out, {
        "added": len(pairs),
        "pairs_possible": possible,
        "pairs_tried": len(pairs),
        "budget": budget,
        "truncated": truncated,
        "columns": cols,
        "note": ("products are CENTRED, which is what makes an "
                 "exclusive-or visible: the product is negative "
                 "exactly when one factor is high and the other low. "
                 "Coverage is partial - an interaction between two "
                 "columns that each already have some relationship is "
                 "still missed."
                 + (" BUDGET REACHED: {} of {} pairs tried.".format(
                     len(pairs), possible) if truncated else "")),
    }


def base_columns(name: str) -> Optional[Tuple[str, str]]:
    """The two columns a product feature was built from."""
    if PRODUCT_SEP in name:
        a, b = name.split(PRODUCT_SEP, 1)
        return a, b
    return None
