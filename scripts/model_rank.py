"""Rank a synthetic column by a model fitted on the REAL data.

THE IDEA, and it is a good one: take the real extract, hold out
one column, fit a model on everything else to predict it, and use
that model to set the column in the synthetic file. The engine's
own worst marginal gap is on `procedure_quantity`, which puts 93%
of rows on the value 1 where the source puts 53%, so a mechanism
that knows what the other columns imply about it is exactly the
right instrument.

TWO THINGS MAKE THE OBVIOUS VERSION WRONG, and both are on this
project's record already.

THE MODEL IS THE DISCLOSURE. A boosted tree fitted on 55,428 real
rows has leaves defined by a handful of real patients; a leaf
reached by three people encodes those three, which is what the k
rule exists to prevent. This project has MEASURED that surface:
the neural engine trains on records and its weights-leak adversary
reads 0.776 - a fail - while its output-only adversary reads 0.514
to 0.551 with a verbatim-republish control at 1.000 CAUGHT. So the
model may never be written down, and this tool is built so that it
cannot be: it is fitted, used and discarded inside one process,
and NOTHING it writes contains a parameter of it. What leaves is a
column of numbers.

AND A PREDICTED VALUE DESTROYS THE MARGINAL. `procedure_quantity`
has a point mass at 1; a regression prediction plus noise is
continuous and rounding cannot land a point mass - this file
records it as "a point mass is not a shape a curve can make", and
it is already why the column drifted, because the engine ALREADY
draws it from a relationship. So the model is used for the ORDER
and never for the value: it decides WHICH rows get more, and the
PUBLISHED MARGINAL decides WHAT the numbers are. The values come
from the blueprint's own grid through the engine's own
`_draw_numeric`, so nothing is disclosed that was not already
published, and the marginal is landed by construction rather than
hoped for.

AND IT REFUSES WHEN THE MODEL CANNOT RANK. A model with no skill
on held-out REAL patients orders the rows by noise, and arranging
a column by noise is worse than leaving it alone - it would
destroy whatever relationships the engine had got right while
reporting that something was done.

Writes nothing without --apply, and re-measures the WHOLE gate
either way, because a repair judged only on the criterion it aims
at is how a 24-point regression once shipped as a fix.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402

from synthkit import gate as _g                 # noqa: E402
from synthkit import marginals as _mg           # noqa: E402
from synthkit import sets as _sets              # noqa: E402

# A ranking has to be better than noise to be worth applying.
# Spearman on held-out REAL patients, which is the quantity the
# arrangement actually uses.
MIN_SKILL = 0.10


def _spear(a, b):
    a = pd.Series(np.asarray(a, dtype=float)).rank()
    b = pd.Series(np.asarray(b, dtype=float)).rank()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _present(series):
    s = series.astype(str).str.strip()
    return ~s.isin(("", "nan", "None", "none", "NaN"))


def fit_ranker(real, target, group_by=None, seed=0):
    """Fit on the REAL data and report held-out skill.

    THE SPLIT IS BY PATIENT. Splitting rows puts the same person
    on both sides, so the skill measured is the same records twice
    and reads high - this project records that exact fault for the
    close ceiling.

    Returns (model, features, skill, rows_fitted). The model is
    returned to the CALLER IN MEMORY and is never serialised.
    """
    from dialback import _design
    from sklearn.ensemble import HistGradientBoostingRegressor

    y_all = pd.to_numeric(real[target], errors="coerce")
    have = y_all.notna().to_numpy()
    if int(have.sum()) < 200:
        return None, [], None, int(have.sum())

    drop = {target}
    for c in real.columns:
        if _sets.is_scaffolding(c):
            drop.add(c)
    X_all, _ = _design(real, target, drop)

    if group_by and group_by in real.columns:
        ids = pd.unique(real[group_by])
        rng = np.random.RandomState(seed)
        keep = set(rng.choice(ids, size=max(1, int(len(ids) * 0.7)),
                              replace=False))
        tr_mask = real[group_by].isin(keep).to_numpy()
    else:
        rng = np.random.RandomState(seed)
        tr_mask = rng.rand(len(real)) < 0.7

    tr = have & tr_mask
    te = have & (~tr_mask)
    if int(tr.sum()) < 100 or int(te.sum()) < 50:
        return None, [], None, int(have.sum())

    m = HistGradientBoostingRegressor(max_iter=200,
                                      random_state=seed)
    m.fit(X_all[tr], y_all.to_numpy()[tr])
    skill = _spear(m.predict(X_all[te]), y_all.to_numpy()[te])
    return m, list(X_all.columns), skill, int(have.sum())


def rank_column(model, features, gen, target, marginal,
                revalue=True, seed=0, random_order=False):
    """Arrange the target by the model's predicted order.

    The model supplies the ORDER. The VALUES come from the
    blueprint's published grid through the engine's own
    `_draw_numeric` - not from the model, and not from a second
    implementation of the quantile draw living here, which is how
    two halves come to disagree about what a published grid means.

    Only rows the generated file already marks PRESENT are
    touched, so coverage is identical afterwards.
    """
    from dialback import _design
    import synthkit.generate as _G

    drop = {target}
    for c in gen.columns:
        if _sets.is_scaffolding(c):
            drop.add(c)
    Xg, _ = _design(gen, target, drop)
    for f in features:
        if f not in Xg.columns:
            Xg[f] = 0.0
    Xg = Xg[features]

    pred = model.predict(Xg)
    here = _present(gen[target]).to_numpy()
    n_here = int(here.sum())
    if not n_here:
        return None, 0, None

    if revalue:
        rng = np.random.RandomState(seed + 7)
        pool = _G._draw_numeric(marginal, rng.rand(n_here))
    else:
        pool = pd.to_numeric(gen.loc[here, target],
                             errors="coerce").to_numpy()
        pool = np.nan_to_num(pool, nan=float(np.nanmedian(pool)))

    # the smallest predicted row gets the smallest value - or,
    # in the CONTROL ARM, nobody does, so the marginal lands the
    # same way and the gate says what the model was worth
    if random_order:
        pred = np.random.RandomState(seed + 991).rand(len(gen))
    order = np.argsort(np.argsort(pred[here]))
    placed = np.sort(np.asarray(pool, dtype=float))[order]

    out = gen[target].astype(object).to_numpy().copy()
    integral = bool(np.all(np.mod(
        np.asarray(pool, dtype=float), 1.0) == 0.0))
    idx = np.where(here)[0]
    for j, pos in enumerate(idx):
        v = placed[j]
        out[pos] = (str(int(round(v))) if integral
                    else "{:.4f}".format(v).rstrip("0").rstrip("."))
    rep = gen.copy()
    rep[target] = out
    achieved = _spear(pred[here],
                      pd.to_numeric(pd.Series(out[here]),
                                    errors="coerce"))
    return rep, n_here, achieved


# Names that would mean a fitted model had been written down. The
# dict assembled below is built from scalars, so this cannot fire
# today - which is exactly why it is a PURE FUNCTION tested on a
# planted dict rather than an inline loop no mutation can reach. A
# guard nothing can break is dead code implying a protection that
# lives somewhere else; this one is a tripwire for the day
# somebody adds a field, and it is reachable.
MODEL_PARAM_NAMES = ("tree_", "coef_", "_predictor", "nodes",
                     "estimators_", "feature_importances_")


def model_leak(obj):
    """The first model-parameter name found, or None."""
    blob = json.dumps(obj, default=str)
    for bad in MODEL_PARAM_NAMES:
        if bad in blob:
            return bad
    return None


def _marg_row(fid, col):
    for r in ((fid.get("marginals") or {}).get("columns") or []):
        if r.get("column") == col:
            return r
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="rank a synthetic column by a model fitted on "
                    "the real data; the model is used for the "
                    "ORDER and discarded")
    ap.add_argument("run")
    ap.add_argument("--src", required=True)
    ap.add_argument("--column", required=True)
    ap.add_argument("--group-by", default="person_id")
    ap.add_argument("--time-col", default=None)
    ap.add_argument("--order-only", action="store_true",
                    help="keep the values the generator produced "
                         "and only rearrange them; the marginal "
                         "cannot move, so this fixes "
                         "relationships and not shape")
    ap.add_argument("--random-order", action="store_true",
                    help="THE CONTROL ARM. Draw the same values "
                         "from the same published grid and "
                         "assign them in RANDOM order. The "
                         "marginal must land identically, "
                         "because a marginal does not care who "
                         "got which value - so whatever this "
                         "arm leaves on the gate is what the "
                         "MODEL is actually buying, and if it "
                         "leaves nothing the model is not "
                         "earning its place")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    from fidelity_deck import read_frame
    from synthkit.pipeline import compare

    run = Path(a.run)
    gen_p = run / "generated.csv"
    bp_p = run / "blueprint.json"
    if not gen_p.exists() or not bp_p.exists():
        print("STOPPED: {} needs generated.csv and "
              "blueprint.json".format(run), file=sys.stderr)
        return 2
    gen = read_frame(gen_p)
    bp = json.loads(bp_p.read_text(encoding="utf-8"))
    src = read_frame(a.src)

    # THE PROVENANCE GUARD, from the one implementation. A
    # BASENAME CANNOT IDENTIFY A FILE - six files of this name
    # and this row count sit on the machine that holds the data -
    # so the check prefers the content digest and SAYS when it
    # had only the row count to go on.
    from synthkit import provenance as _prov
    if not _prov.guard(run, a.src, len(src)):
        return 2

    col = a.column
    if col not in gen.columns:
        print("STOPPED: no column {!r} in the generated file"
              .format(col), file=sys.stderr)
        return 2
    if col not in src.columns:
        print("STOPPED: no column {!r} in the source, so no model "
              "can be fitted for it".format(col), file=sys.stderr)
        return 2
    marginal = ((bp.get("columns") or {}).get(col) or {}).get(
        "marginal") or {}
    if not a.order_only and marginal.get("type") != "quantiles":
        print("STOPPED: {} publishes no quantile grid, so there "
              "is no marginal to draw the values from. Pass "
              "--order-only to rearrange the values the "
              "generator produced instead.".format(col),
              file=sys.stderr)
        return 2

    print("model-rank on {} of {}".format(col, run.name))
    print("  the model is fitted here, used here, and discarded "
          "here - nothing written below contains a parameter of "
          "it, and what leaves is a column of numbers")

    model, feats, skill, nfit = fit_ranker(
        src, col, group_by=a.group_by, seed=a.seed)
    if model is None:
        print("  REFUSED: only {} source row(s) carry a value for "
              "{} - too few to fit and hold out a measurable "
              "skill".format(nfit, col))
        return 1
    print("  fitted on {} real row(s) across {} feature(s)".format(
        nfit, len(feats)))
    print("  held-out skill on real PATIENTS held back: "
          "{:+.4f}".format(skill))
    if abs(skill) < MIN_SKILL:
        print("  REFUSED: a model with no skill orders the rows "
              "by NOISE, and arranging a column by noise destroys "
              "whatever the engine had right while reporting that "
              "something was done. Floor is {:.2f}.".format(
                  MIN_SKILL))
        return 1

    rep, n_here, achieved = rank_column(
        model, feats, gen, col, marginal,
        revalue=not a.order_only, seed=a.seed,
        random_order=a.random_order)
    if rep is None:
        print("  REFUSED: the generated file marks no row present "
              "for {}, so there is nothing to arrange".format(col))
        return 1
    if a.random_order:
        print("  CONTROL ARM: the values are assigned in RANDOM "
              "order, so the marginal lands the same way and "
              "whatever the gate loses here is what the model "
              "was buying")
    print("  arranged {} present row(s); values from {}".format(
        n_here, "the generator's own draw (--order-only)"
        if a.order_only else "the published quantile grid"))
    print("  the arrangement tracks the model at {:+.4f}".format(
        achieved))

    before_cov = float(_present(gen[col]).mean())
    after_cov = float(_present(rep[col]).mean())
    print("  coverage {:.4f} -> {:.4f} (the pass touches only "
          "rows already present)".format(before_cov, after_cov))

    tcol = a.time_col
    fid1 = compare(src, gen, bp, a.group_by, tcol)
    fid2 = compare(src, rep, bp, a.group_by, tcol)
    m1, m2 = _marg_row(fid1, col), _marg_row(fid2, col)
    print("")
    print("  {} against its own source marginal: {} -> {}".format(
        col,
        None if not m1 else round(m1.get("source_vs_output"), 4),
        None if not m2 else round(m2.get("source_vs_output"), 4)))

    print("")
    print("  THE WHOLE GATE, before and after - a repair judged "
          "only on what it aims at is how a regression ships as "
          "a fix:")
    v1, v2 = _g.assess(fid1), _g.assess(fid2)
    for c1, c2 in zip(v1["criteria"], v2["criteria"]):
        mark = "same" if c1["detail"] == c2["detail"] else "MOVED"
        print("    {:24s} {:28s} -> {:28s} {}".format(
            c1["name"][:24], str(c1["detail"])[:28],
            str(c2["detail"])[:28], mark))
    print("    {:24s} {} -> {}".format(
        "MET", v1.get("met"), v2.get("met")))

    if not a.apply:
        print("")
        print("  nothing written. Pass --apply to write a "
              "drop-in run directory.")
        return 0

    out = Path(a.out) if a.out else Path(
        str(run) + "_modelrank")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(run, out)
    rep.to_csv(out / "generated.csv", index=False)
    fid2["remeasured"] = {
        "by": "model_rank.py",
        "column": col,
        "model_skill_on_held_out_real_patients": round(skill, 6),
        "values_from": ("the generator's own draw"
                        if a.order_only
                        else "the published quantile grid"),
        "model_published": False,
        "note": ("the model was fitted, used and discarded in one "
                 "process; no parameter of it is recorded here, "
                 "and the values came from the blueprint's own "
                 "published grid rather than from the model"),
    }
    (out / "fidelity.json").write_text(
        json.dumps(fid2, indent=1, default=str), encoding="utf-8")
    leak = model_leak(fid2)
    if leak:
        print("STOPPED: the written fidelity carries {!r}, which "
              "looks like a model parameter. The model must not "
              "leave this process.".format(leak),
              file=sys.stderr)
        return 2
    print("")
    print("  wrote {} - a drop-in run directory; the ORIGINAL is "
          "untouched".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
