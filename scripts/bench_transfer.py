"""Which fidelity properties actually move the ANSWER?

    python scripts/bench_transfer.py --src EXTRACT.csv --run RUNDIR
                                     --group-by person_id
                                     --target COLUMN [--seeds 0,1,2]

WHY THIS EXISTS, AND IT IS THE MISSING PIECE. This project reports
coverage, centre, spread, steadiness, clustering, pair sign,
inversion, identity tightness, constraints, set tokens, interactions
and heteroscedasticity. Every one of them is a PROXY. The product's
actual claim is narrower and much more useful: point a vendor's model
at data shaped like yours and the number it gets will tell you
something about the number it would get on yours.

Nothing has ever measured the relationship between the proxies and
that claim. So every defect is treated as equally urgent, the queue of
properties is unbounded, and there is no criterion for stopping. This
turns the question into an experiment.

WHAT IT MEASURES. For a set of candidate models:

  truth      score of model M trained on REAL, tested on held-out REAL
  estimate   score of model M trained on SYNTHETIC, tested on the
             SAME held-out REAL

The product ranks vendors, so the quantity that matters is whether
those two orderings AGREE - Spearman across the candidates - not
whether any single score is close. A synthetic set that makes every
model look 10% worse but keeps the order is doing its job.

THEN ONE PROPERTY IS BROKEN AT A TIME, through the dials, and the
agreement is re-measured. A property whose destruction does not move
the ranking is one this instrument does not need to chase.

THE POSITIVE CONTROL IS WHAT MAKES THE REST WORTH READING. `no
relationships` sets every relationship strength to zero, which leaves
the marginals perfect and the structure absent. If THAT does not
collapse the agreement, the measurement is insensitive and no other
row here means anything - the same reason the attribute-disclosure
work trains an adversary on a generator that republishes records.

NUMERIC FEATURES ONLY, said plainly rather than left to be discovered:
categorical columns are dropped before fitting, so this measures
transfer on the numeric half of the table.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthkit.generate import generate            # noqa: E402


def candidates():
    """Deliberately DIFFERENT learners, or there is no ranking.

    A panel of near-identical models produces a near-arbitrary order
    and the Spearman below would measure noise. These differ in the
    one way the fixture's planted structure cares about: whether they
    can represent a nonlinearity at all."""
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import (HistGradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    return [
        ("mean", lambda: DummyRegressor(strategy="mean")),
        ("ridge", lambda: Ridge(alpha=1.0)),
        ("stump", lambda: DecisionTreeRegressor(max_depth=2,
                                                random_state=0)),
        ("tree", lambda: DecisionTreeRegressor(max_depth=6,
                                               random_state=0)),
        ("knn", lambda: KNeighborsRegressor(n_neighbors=15)),
        ("forest", lambda: RandomForestRegressor(
            n_estimators=60, random_state=0, n_jobs=1)),
        ("boost", lambda: HistGradientBoostingRegressor(
            max_iter=120, random_state=0)),
    ]


def xy(frame, feats, target):
    X = frame[feats].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(frame[target], errors="coerce")
    ok = y.notna()
    return X[ok].to_numpy(dtype=float), y[ok].to_numpy(dtype=float)


def fit_score(train, test, feats, target):
    """R2 on the held-out REAL rows, for every candidate."""
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import r2_score
    from sklearn.pipeline import make_pipeline
    Xtr, ytr = xy(train, feats, target)
    Xte, yte = xy(test, feats, target)
    out = {}
    for name, make in candidates():
        if len(ytr) < 50:
            out[name] = float("nan")
            continue
        m = make_pipeline(SimpleImputer(strategy="median"), make())
        m.fit(Xtr, ytr)
        out[name] = float(r2_score(yte, m.predict(Xte)))
    return out


def agreement(truth, est, top=None):
    """Spearman between the two orderings, over shared candidates.

    `top` RESTRICTS TO THE COMPETITIVE MODELS, and that is the number
    worth reading. Across the full panel the order is easy: a mean
    predictor at 0.00 and a kNN at -0.12 sit far below the rest, and
    any synthetic set that preserves "tree beats nothing" reproduces
    most of the ordering while telling a vendor nothing they did not
    know. The question the product is actually asked is which of the
    CONTENDERS is best - four models inside 0.08 of each other - and
    separating those is a far harder test."""
    from scipy.stats import spearmanr
    keys = [k for k in truth
            if np.isfinite(truth[k]) and np.isfinite(est.get(k, np.nan))]
    if top:
        keys = sorted(keys, key=lambda k: -truth[k])[:top]
    if len(keys) < 3:
        return float("nan")
    a = [truth[k] for k in keys]
    b = [est[k] for k in keys]
    return float(spearmanr(a, b).correlation)


def degrade(bp, how):
    """One property broken, through the dials the product exposes."""
    b = copy.deepcopy(bp)
    cols = b.get("columns") or {}
    if how == "faithful":
        return b
    if how == "no relationships":
        for r in b.get("relationships") or []:
            r.setdefault("dials", {})["strength"] = 0.0
        return b
    for c, spec in cols.items():
        d = spec.setdefault("dials", {})
        if how == "no persistence":
            d["persistence"] = 0.0
        elif how == "no missingness":
            d["coverage"] = 1.0
        elif how == "spread x1.4" and spec.get("kind") == "numeric":
            d["scale"] = 1.4
        elif how == "centre + 0.5sd" and spec.get("kind") == "numeric":
            sd = float(((spec.get("marginal") or {}).get("sd")) or 0.0)
            if not sd:
                m = spec.get("marginal") or {}
                v = m.get("v") or []
                sd = (float(v[-1]) - float(v[0])) / 4.0 if len(v) > 1 \
                    else 0.0
            d["shift"] = 0.5 * sd
        elif how == "no clustering":
            d["missing_clustering"] = 0.0
    return b


# How many of the best models the hard comparison uses.
TOP = 4

ARMS = ["faithful", "no persistence", "no missingness", "no clustering",
        "spread x1.4", "centre + 0.5sd", "no relationships"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--group-by", default="person_id")
    ap.add_argument("--target", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--holdout", type=float, default=0.35)
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    if len(seeds) < 2:
        print("refusing to run on one seed", file=sys.stderr)
        return 2

    print("src      {}".format(Path(a.src).name))
    print("run      {}".format(Path(a.run).name))
    print("target   {}   group-by {}".format(a.target, a.group_by))
    print("seeds    {}".format(", ".join(str(s) for s in seeds)))
    print()

    bp = json.loads((Path(a.run) / "blueprint.json").read_text(
        encoding="utf-8"))
    df = pd.read_csv(a.src)
    if a.target not in df.columns:
        print("no such column: {}".format(a.target), file=sys.stderr)
        return 2

    # SPLIT BY PATIENT, never by row. Two visits of one person share
    # almost everything, so a row split leaks the answer across it and
    # every score comes out flattering.
    # `.copy()` IS LOAD-BEARING on pandas 3: to_numpy returns a
    # read-only view there and shuffle raises. Third instance of
    # the class - and this file escaped the pandas-3 net run
    # because benches are not smokes.
    pids = df[a.group_by].drop_duplicates().to_numpy().copy()
    rs = np.random.RandomState(0)
    rs.shuffle(pids)
    cut = int(len(pids) * (1.0 - a.holdout))
    tr_ids, te_ids = set(pids[:cut]), set(pids[cut:])
    real_tr = df[df[a.group_by].isin(tr_ids)]
    real_te = df[df[a.group_by].isin(te_ids)]

    feats = [c for c in df.columns
             if c not in (a.group_by, a.target)
             and pd.api.types.is_numeric_dtype(df[c])]
    print("{} numeric features, {} real train rows, {} held-out real "
          "rows".format(len(feats), len(real_tr), len(real_te)))

    truth = fit_score(real_tr, real_te, feats, a.target)
    print()
    print("TRUTH - trained on real, tested on held-out real:")
    for k, v in sorted(truth.items(), key=lambda kv: -kv[1]):
        print("   {:<8} r2 {:+.3f}".format(k, v))
    spread = max(truth.values()) - min(truth.values())
    print()
    if spread < 0.05:
        print("WARNING: the candidates score within {:.3f} of each "
              "other on real data, so there is barely an ordering to "
              "reproduce and every number below is noise. Pick a "
              "target with more structure.".format(spread))
        print()

    n_pat = int(df[a.group_by].nunique())
    rows = {}
    for arm in ARMS:
        b = degrade(bp, arm)
        for seed in seeds:
            g = generate(b, n_patients=n_pat, seed=seed)
            gf = [c for c in feats if c in g.columns]
            est = fit_score(g, real_te, gf, a.target)
            rows.setdefault(arm, []).append(
                (agreement(truth, est),
                 float(np.nanmean([est[k] for k in est])),
                 agreement(truth, est, top=TOP)))
        print("  {:<18} done".format(arm))

    print()
    print("{:<18}{:>20}{:>20}{:>16}".format(
        "", "rank all", "rank top-{}".format(TOP), "mean r2"))
    print("-" * 74)
    base = base_top = None
    for arm in ARMS:
        ag = [r[0] for r in rows[arm]]
        sc = [r[1] for r in rows[arm]]
        tp = [r[2] for r in rows[arm]]
        if arm == "faithful":
            base = float(np.nanmean(ag))
            base_top = float(np.nanmean(tp))
        print("{:<18}{:>20}{:>20}{:>16}".format(
            arm,
            "{:+.2f} ({:+.2f},{:+.2f})".format(
                float(np.nanmean(ag)), min(ag), max(ag)),
            "{:+.2f} ({:+.2f},{:+.2f})".format(
                float(np.nanmean(tp)), min(tp), max(tp)),
            "{:+.3f}".format(float(np.nanmean(sc)))))
    print()
    print("`rank top-{}` is the one to read: across the whole panel a "
          "mean predictor and a".format(TOP))
    print("broken kNN sit far below everything, so most of the order "
          "survives any damage.")
    print("The product is asked which CONTENDER wins, and those sit "
          "within a few points.")
    print()
    ctrl = float(np.nanmean([r[0] for r in rows["no relationships"]]))
    if base is not None and not (ctrl < base - 0.15):
        print("READ NOTHING FROM THIS RUN. The positive control - "
              "every relationship")
        print("strength set to zero, so the structure is gone and the "
              "marginals are intact -")
        print("scored {:+.2f} against {:+.2f} faithful. If destroying "
              "the structure does not".format(ctrl, base))
        print("move the ranking, this measurement cannot detect a "
              "fidelity defect of any")
        print("kind and the other rows are noise.")
    else:
        print("The positive control moved the ranking from {:+.2f} to "
              "{:+.2f}, so this".format(base, ctrl))
        print("measurement can see structure loss. A row that did NOT "
              "move is therefore")
        print("evidence about that property, not about the "
              "instrument.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
