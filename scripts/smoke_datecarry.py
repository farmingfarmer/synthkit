"""A date's children, carried through the within-patient reorder.

THE LOSS, REPRODUCED BEFORE IT WAS FIXED. The reorder sorts each
patient's date VALUES down their rows and leaves every other
column where it was - so a child drawn from a date's curve keeps
its value while the parent's moves, and the pairing it was drawn
to satisfy is broken. Read off the engine's own probe on this
fixture: rho 0.9785 at `after-size-identity` and 0.8612 at
`after-date-reorder`. On the operator's columns the real
extract's curve trace records `condition_count <-
visit_start_date` DROPPED at that exact pass.

AND THE POOLED READING UNDERSTATES IT BADLY, which is why the
first measurement of this looked small. The reorder only ever
moves values WITHIN a patient, so a pooled correlation is held
up by the between-patient half it cannot touch - 0.9802 in
source against 0.8612 generated. Measured WITHIN the patient the
same pair reads 0.8716 against 0.1069: the temporal
relationship is not weakened, it is gone.

WHY CARRYING IS THE ONLY FIX AVAILABLE. The permutation is
FORCED - it is whatever sorts the dates - so if the pairing is to
survive, the child must move by the same permutation. Nothing
else is free to choose.

AND WHY IT IS SAFE. That permutation never moves a value out of
its patient, so a carried child keeps its multiset PER PATIENT
and therefore every patient-level statistic of it exactly: the
patient's mean is identical, which this suite asserts rather
than argues. What moves is within-patient pairing and nothing
else.

THE SAFE RULE DECLINES MORE THAN IT CARRIES, and that is
measured rather than hidden: a child with a non-date parent too
would have THAT pairing broken by the carry, so it is declined
and NAMED. On the fixture below `load` has only the date as a
parent and is carried; a one-line change to the fixture gives it
a co-parent and it is declined.
"""
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import synthkit.blueprint as B                          # noqa: E402
import synthkit.generate as G                           # noqa: E402
from rhythm_read import numeric_days                    # noqa: E402
from synthkit.discover import discover                  # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def frame(n_pat=150, n_vis=8, seed=5, coparent=False):
    """A date-driven child, plus an unmodelled bystander.

    `load` depends on the DATE and nothing else, so the safe rule
    carries it. `tag` is driven by a PATIENT-LEVEL term, which is
    what makes it the control: it cannot become a date-driven
    co-parent of `load` however the seed falls, and whatever the
    carry costs a pair it was not aimed at shows up there.

    THE FIRST VERSION TIED `tag` TO THE DATE WEAKLY, and that put
    it right on discovery's inclusion boundary - it became a
    co-parent of `load` on one draw and not on another, so
    whether the fixture could exercise the carry at all depended
    on the seed. A fixture whose property is a coin flip is one
    this project has a rule about.

    With `coparent=True` the child gains a second, NON-date
    driver, which is the shape the safe rule must decline.
    """
    rng = np.random.default_rng(seed)
    base = np.datetime64("2014-01-01")
    rows = []
    for p in range(n_pat):
        lvl = float(rng.normal(0, 1))
        start = int(rng.integers(0, 2200))
        for v in range(n_vis):
            d = start + int(rng.integers(0, 900))
            extra = 6.0 * lvl if coparent else 0.0
            row = {
                "person_id": "P%03d" % p,
                "visit_number": v + 1,
                "visit_start_date": str(
                    base + np.timedelta64(d, "D")),
                "load": float(0.02 * d + extra
                              + rng.normal(0, 3)),
                "tag": float(5.0 * lvl + rng.normal(0, 1)),
            }
            if coparent:
                row["drive"] = float(6.0 * lvl
                                     + rng.normal(0, 0.5))
            rows.append(row)
    return pd.DataFrame(rows)


def built(df):
    return B.build(df, discover(df, group_by="person_id"),
                   group_by="person_id",
                   time_col="visit_start_date")


def num(s):
    """Dates are READ as numbers where they are numeric and
    stamped otherwise. `to_numeric` alone returns all-NaN on a
    rendered date, which is the wrong-type fault this project
    records four times - and it read `nan` on the first pass of
    this very measurement."""
    v = numeric_days(s)
    return pd.to_numeric(s, errors="coerce") if v is None else v


def within(g, a, b):
    """Rank correlation after each patient's mean is removed.

    THE READING THAT MATTERS HERE. The reorder moves values only
    within a patient, so a POOLED correlation is propped up by
    the between-patient half it never touches and understates
    the loss by an order of magnitude.
    """
    x, y = num(g[a]), num(g[b])
    gg = g["person_id"]
    xd = x - x.groupby(gg).transform("mean")
    yd = y - y.groupby(gg).transform("mean")
    ok = xd.notna() & yd.notna()
    return float(spearmanr(xd[ok], yd[ok]).statistic)


def pooled(g, a, b):
    x, y = num(g[a]), num(g[b])
    ok = x.notna() & y.notna()
    return float(spearmanr(x[ok], y[ok]).statistic)


def in_order_share(g):
    d = num(g["visit_start_date"])
    t = pd.DataFrame({"g": g["person_id"].to_numpy(),
                      "d": d.to_numpy()})
    return float(t.groupby("g")["d"].apply(
        lambda x: bool(x.is_monotonic_increasing)).mean())


def main():
    df = frame()
    bp = built(df)
    cols = bp.get("columns") or {}
    dates = [c for c, v in cols.items()
             if (v or {}).get("date")
             and (v or {}).get("kind") == "numeric"]

    # ---- THE PREDICATE, ON PLANTED CONTRACTS ---------------
    plan = {"relationships": [
        {"child": "kid", "parents": ["d1"]},
        {"child": "d1", "parents": ["kid"]},
        {"child": "twodates", "parents": ["d1", "d2"]},
        {"child": "hub", "parents": ["d1", "other"]},
        {"child": "free", "parents": ["other"]}]}
    carry = G.date_driven_children(plan, ["d1", "d2"])
    decl = G.declined_date_children(plan, ["d1", "d2"])
    check("a child whose published parents are ALL dates is "
          "carried, and one with a non-date parent too is NOT - "
          "carrying that one would trade its date pairing for "
          "the other, and which to prefer is not a decision this "
          "pass can make: carried {}".format(carry),
          carry == ["kid", "twodates"])
    check("...and the declined child is NAMED with the parent "
          "that disqualified it, because a count cannot be acted "
          "on and this list is what says whether the safe rule "
          "reaches anything on a given blueprint: {}".format(
              decl),
          decl == {"hub": ["other"]})
    check("...and a DATE is never carried as a child of itself, "
          "nor a column no date drives - the first would be "
          "double-moved and the second has nothing to preserve",
          "d1" not in carry and "free" not in carry
          and "free" not in decl)
    check("...and a blueprint with no relationships yields "
          "nothing rather than raising, so an older artifact "
          "still generates",
          G.date_driven_children({}, ["d1"]) == []
          and G.declined_date_children({}, ["d1"]) == {})

    # ---- THE FIXTURE REPRODUCES THE LOSS -------------------
    check("the fixture carries the shape, or everything below "
          "measures nothing - a date axis, a child the safe rule "
          "carries, and a bystander discovery does not model: "
          "dates {}, carried {}".format(
              dates, G.date_driven_children(bp, dates)),
          dates == ["visit_start_date"]
          and G.date_driven_children(bp, dates) == ["load"]
          and not G.declined_date_children(bp, dates))

    off = G.generate(bp, seed=11, carry_date_children=False)
    on = G.generate(bp, seed=11, carry_date_children=True)

    w_src = within(df, "visit_start_date", "load")
    w_off = within(off, "visit_start_date", "load")
    w_on = within(on, "visit_start_date", "load")
    check("the reorder DESTROYS the within-patient date "
          "relationship, reproduced before the fix so the fix "
          "cannot quietly become a no-op: source {:.4f}, carry "
          "off {:.4f}".format(w_src, w_off),
          w_src > 0.6 and w_off < 0.25)
    check("...and the POOLED reading understates it by an order "
          "of magnitude, which is why the first measurement of "
          "this looked small - the reorder moves values only "
          "within a patient, so the between-patient half props "
          "the pooled number up: pooled {:.4f} against within "
          "{:.4f}".format(pooled(off, "visit_start_date", "load"),
                          w_off),
          pooled(off, "visit_start_date", "load") > 0.8)
    # THE BAR IS PROXIMITY TO SOURCE, NOT A ROUND NUMBER. A
    # first cut demanded `> 0.9` and went red on a recovery of
    # 0.8897 against a source of 0.8567 - which is essentially
    # exact. Worse, that bar would have PASSED a fix that
    # overshot to 0.99 on a source of 0.5, which is the
    # manufactured-relationship fault this project treats as
    # worse than a missing one.
    check("CARRYING THE CHILD RECOVERS IT, to within 0.1 of the "
          "source rather than to a round number: {:.4f} against "
          "a source {:.4f}, from {:.4f}".format(
              w_on, w_src, w_off),
          abs(w_on - w_src) < 0.1 and w_on > w_off + 0.5)

    # ---- AND THE PRICE, WHICH IS WHAT MAKES IT SAFE --------
    a = sorted(num(off["load"]).round(9).dropna().tolist())
    b = sorted(num(on["load"]).round(9).dropna().tolist())
    check("the carried child holds the IDENTICAL multiset, so "
          "its published marginal cannot have moved - the "
          "instrument the repairs, the dials and the cyclic "
          "sweep all rest on",
          a == b)
    pm_off = num(off["load"]).groupby(
        off["person_id"]).mean().round(9)
    pm_on = num(on["load"]).groupby(
        on["person_id"]).mean().round(9)
    check("...and every patient's MEAN of it is identical, "
          "exactly - the permutation never moves a value out of "
          "its patient, so icc, patient means and every "
          "between-patient correlation are preserved by "
          "construction rather than by luck",
          bool((pm_off == pm_on).all()))
    moved = int((num(off["load"]).to_numpy()
                 != num(on["load"]).to_numpy()).sum())
    check("...and it is not a no-op: {} of {} rows hold a "
          "different value, so the two arms are genuinely two "
          "arms".format(moved, len(on)),
          moved > len(on) // 2)
    # THE COST, OVER EVERY PAIR THAT IS NOT THE ONE AIMED AT.
    # A single hand-picked bystander only measures the pair
    # somebody thought of; this walks them all and reports the
    # worst, so a cost anywhere shows up.
    others = [c for c in on.columns
              if c not in ("person_id", "visit_start_date")]
    worst, pair = 0.0, None
    for i, x in enumerate(others):
        for y in others[i + 1:]:
            d = abs(within(on, x, y) - within(off, x, y))
            if d > worst:
                worst, pair = d, (x, y)
    check("...and no pair OTHER than the one aimed at moves "
          "within the patient by more than 0.05 - walked over "
          "every pair rather than one hand-picked bystander, "
          "because a single control only measures the pair "
          "somebody thought of: worst {:.4f} on {}".format(
              worst, pair),
          worst < 0.05)
    check("...and the dates are STILL in visit order on every "
          "patient, which is the whole reason the reorder exists "
          "- a fix that bought the curve by undoing the pass "
          "would be worse than the loss: {:.4f} / {:.4f}".format(
              in_order_share(off), in_order_share(on)),
          in_order_share(off) == 1.0
          and in_order_share(on) == 1.0)

    # ---- OFF BY DEFAULT -----------------------------------
    base = G.generate(bp, seed=11)
    check("the default is OFF - byte-identical to an arm that "
          "names the flag False, asserted rather than argued, "
          "because the reorder itself was measured on a paired "
          "REAL arm as costing ZERO close pairs and this fixes a "
          "loss those criteria never saw",
          G.CARRY_DATE_CHILDREN is False
          and base.to_csv(index=False) == off.to_csv(index=False))

    # ---- THE DECLINED LIST IS REPORTED EITHER WAY ----------
    df2 = frame(coparent=True)
    bp2 = built(df2)
    d2 = [c for c, v in (bp2.get("columns") or {}).items()
          if (v or {}).get("date")
          and (v or {}).get("kind") == "numeric"]
    decl2 = G.declined_date_children(bp2, d2)
    check("a child with a NON-DATE co-parent is declined on a "
          "real blueprint, not just a planted one - the fixture "
          "gains one driver and the safe rule stops reaching it: "
          "{}".format(decl2 or "nothing declined"),
          bool(decl2))
    rep_off, rep_on = {}, {}
    G.generate(bp2, seed=11, report=rep_off,
               carry_date_children=False)
    G.generate(bp2, seed=11, report=rep_on,
               carry_date_children=True)
    r_off = rep_off.get("dates_put_in_visit_order") or {}
    r_on = rep_on.get("dates_put_in_visit_order") or {}
    check("...and the run REPORTS what it declined EVEN WITH THE "
          "FLAG OFF, because a reader who leaves it off is "
          "entitled to know what it would have reached - and a "
          "blueprint where the safe rule reaches nothing is the "
          "measurement that sends the next person to a joint "
          "re-rank instead",
          bool(r_off.get("declined_children"))
          and bool(r_on.get("declined_children")))
    check("...and the declined child is NOT carried with the "
          "flag on, which is the refusal doing the work rather "
          "than the list being decoration",
          not any(c in (r_on.get("carried_children") or [])
                  for c in decl2))

    # ---- THE RUN RECORDS THE ARGUMENT ---------------------
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src.csv")
        df.to_csv(src, index=False)
        run = os.path.join(td, "run")
        p = subprocess.run(
            [sys.executable,
             str(ROOT / "scripts" / "run_discovery.py"),
             "--src", src, "--group-by", "person_id",
             "--time-col", "visit_start_date",
             "--out", run, "--generate",
             "--carry-date-children"],
            capture_output=True, text=True, cwd=str(ROOT),
            timeout=1800)
        import json
        prov = {}
        pp = os.path.join(run, "provenance.json")
        if os.path.exists(pp):
            prov = json.load(io.open(pp, encoding="utf-8"))
        ga = ((prov.get("settings") or {}).get("generate_args")
              or {})
        check("the flag reaches the engine THROUGH THE CLI and "
              "the run RECORDS it - a run whose provenance cannot "
              "say whether this was on is a run nobody can pair "
              "against another: exit {}".format(p.returncode),
              p.returncode == 0
              and ga.get("carry_date_children") is True)

    print()
    if FAIL:
        print("{} of {} checks failed.".format(FAIL, PASS + FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
