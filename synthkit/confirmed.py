"""Structure discovered on one set of patients, confirmed on another.

A search that tests thousands of pairs will find structure that lives
only in the patients it happened to look at. Measured on a real
extract: 58 relationships found, 16 reproduced on patients the search
never saw. The other 42 were shaping generated data and nothing was
refereeing them - a threshold cannot, because it is part of the search.

So the model is built in three steps rather than one:

  DISCOVER   search a training half for candidate structure
  CONFIRM    re-test each candidate on held-out patients, where no
             correction applies: each arrives as a single
             pre-specified hypothesis and the holdout does not search
  REFIT      learn again on ALL the data, restricted to what survived

The refit matters. Discovery on half the patients is cheap; parameter
estimation should still use every one of them. Declaring the confirmed
structure as `hypotheses` does exactly that, and collapses the
correction at the same time, which is honest here because the
structure has already been validated out of sample.

The split is BY PATIENT. Visits from one person are not independent,
so a row-wise split leaks the same patient into both halves and
confirms nearly anything.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from .condnet import CondNet, _chi2_crit

JOIN = "‖"


def split_by_patient(rows, group, frac=0.7, seed=20260731):
    """Zero patient overlap, deterministic for a given seed."""
    pids = sorted(set(str(r.get(group, "")) for r in rows))
    rnd = random.Random(seed)
    rnd.shuffle(pids)
    keep = set(pids[:int(len(pids) * frac)])
    tr = [r for r in rows if str(r.get(group, "")) in keep]
    te = [r for r in rows if str(r.get(group, "")) not in keep]
    return tr, te, len(keep), len(pids) - len(keep)


def mutual_information(xs, ys):
    n = len(xs)
    if n == 0:
        return 0.0, 1
    joint = Counter(zip(xs, ys))
    cx, cy = Counter(xs), Counter(ys)
    mi = 0.0
    for (a, b), c in joint.items():
        pxy = c / float(n)
        mi += pxy * math.log(pxy / ((cx[a] / float(n))
                                    * (cy[b] / float(n))))
    return max(0.0, mi), max(1, (len(cx) - 1) * (len(cy) - 1))


def test_edge(net, rows, child, parents, group, alpha):
    """The search's own G-test, on data the search never saw.

    G = 2 * n * MI against chi-square. n counts PATIENTS: ten visits
    from one person are not ten independent observations, and counting
    rows here would confirm almost anything."""
    bc = net.binnings.get(child)
    if bc is None or not parents:
        return None
    ec = [bc.encode(r.get(child, "")) for r in rows]
    cols = []
    for p in parents:
        bp = net.binnings.get(p)
        if bp is None:
            return None
        cols.append([bp.encode(r.get(p, "")) for r in rows])
    comb = [JOIN.join(t) for t in zip(*cols)]
    n_pat = len(set(str(r.get(group, "")) for r in rows))
    mi, df = mutual_information(ec, comb)
    g = 2.0 * n_pat * mi
    crit = _chi2_crit(df, alpha)
    return {"g": round(g, 3), "crit": round(crit, 3), "df": df,
            "patients": n_pat, "reproduced": bool(g >= crit)}


def learn_confirmed(rows: List[Dict[str, Any]],
                    group_by: str,
                    k: int = 10,
                    max_parents: int = 3,
                    alpha: float = 0.01,
                    train_frac: float = 0.7,
                    seed: int = 20260731,
                    multilevel: bool = True,
                    epsilon: float = 0.0,
                    correction: str = "bonferroni",
                    targets: Optional[List[str]] = None,
                    feature_sources: Optional[
                        Dict[str, List[str]]] = None) -> CondNet:
    """Discover, confirm out of sample, then refit on everything.

    Returns a CondNet whose structure is exactly what reproduced on
    held-out patients, with the evidence for each relationship in
    `report["confirmation"]`."""
    tr, te, n_tr, n_te = split_by_patient(rows, group_by, train_frac,
                                          seed)
    if n_tr < 2 * k or n_te < 2 * k:
        raise ValueError(
            "confirmation needs enough patients on both sides: {} "
            "train and {} held out at k={}. Lower k, or learn without "
            "confirmation.".format(n_tr, n_te, k))

    # Engineered features must be declared on BOTH passes. The
    # scout needs them to keep a feature parent-only and to settle
    # a product's direction; the refit needs them to REGENERATE
    # each feature from the data it makes rather than draw it from
    # a marginal, which would report the pattern without producing
    # it.
    scout = CondNet(k=k, max_parents=max_parents,
                    correction=correction).learn(
        tr, group_by=group_by, multilevel=multilevel,
        targets=targets, feature_sources=feature_sources)

    kept, dropped = [], []
    for e in scout.report.get("edges", []):
        child = e.get("child")
        parents = list(e.get("parents") or [])
        if not parents:
            continue
        found = test_edge(scout, tr, child, parents, group_by, alpha)
        held = test_edge(scout, te, child, parents, group_by, alpha)
        if found is None or held is None:
            continue
        rec = {"child": child, "parents": parents,
               "found_on": found, "held_out": held}
        (kept if held["reproduced"] else dropped).append(rec)

    # Refit on EVERY patient, restricted to what survived. Discovery
    # can afford half the data; parameter estimation should not.
    hyp = defaultdict(list)
    for e in kept:
        for p in e["parents"]:
            if p not in hyp[e["child"]]:
                hyp[e["child"]].append(p)
    net = CondNet(k=k, max_parents=max_parents).learn(
        rows, group_by=group_by, multilevel=multilevel,
        epsilon=epsilon, targets=targets,
        feature_sources=feature_sources,
        hypotheses=(dict(hyp) or None))

    net.report["confirmation"] = {
        "train_patients": n_tr,
        "holdout_patients": n_te,
        "patient_overlap": 0,
        "holdout_alpha": alpha,
        "discovered": len(kept) + len(dropped),
        "confirmed": len(kept),
        "not_reproduced": len(dropped),
        "confirmed_edges": kept,
        "dropped_edges": dropped,
        "note": "structure discovered on {} patients and re-tested on "
                "{} the search never saw; parameters then fitted on "
                "all {}. No correction applies on the holdout - each "
                "edge is one pre-specified hypothesis and the holdout "
                "does not search.".format(n_tr, n_te, n_tr + n_te),
    }
    return net
