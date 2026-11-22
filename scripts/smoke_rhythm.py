"""A patient's visits come in episodes, not at an even spacing.

WHY THIS EXISTS. `dynamics tracked` read 34 of 92 and was taken as
"the engine models no patient trajectory". Measured with the
own-column split it is 15 of 20, and THREE of the five remaining
misses are one mechanism: the two date columns and `age_at_visit`,
which an identity binds to them. Measured against the source:

    quantity          source   generated
    patient span       2,478       2,313
    mean gap            34.6        31.4
    SAME-DAY gaps      13.0%        3.5%

The span and the mean land; the SPACING does not. A patient holds
a random sample of the global date pool, so the twelve rows
sharing a date land on twelve different people and nobody gets a
same-day encounter.

THE FIX IS A REALLOCATION, which is the only form worth building:
which patient holds which DRAWN date changes, so the multiset -
and therefore the published marginal - cannot move.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def episodic(seed=11, n_pat=160):
    """An episodic cohort with TWO date columns.

    The second one is not decoration. The reallocation carries
    every other date column along at this row's own offset, and
    that loop body runs only from the SECOND date column - so a
    one-date fixture cannot reach it at all. With one date column
    the pass shipped a `pd.to_datetime` in that branch and the real
    extract refused it on the first run: `ufunc 'subtract' cannot
    use operands with types dtype('<M8[ns]') and dtype('float64')`.
    A fixture must contain the thing the check is about, and "a
    date column" was not specific enough.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for p in range(n_pat):
        t = rng.randint(0, 2000)
        for _ in range(rng.randint(4, 9)):
            t += int(rng.exponential(160))
            for _ in range(rng.randint(1, 5)):
                los = int(rng.randint(0, 6))
                rows.append({
                    "person_id": "P%03d" % p,
                    "visit_start_date":
                        (pd.Timestamp("2016-01-01")
                         + pd.Timedelta(days=t)).strftime("%Y-%m-%d"),
                    "visit_end_date":
                        (pd.Timestamp("2016-01-01")
                         + pd.Timedelta(days=t + los)).strftime(
                             "%Y-%m-%d"),
                    "sev": float(rng.normal(10 + 0.01 * t, 2)),
                    "n_proc": int(rng.poisson(2))})
    return pd.DataFrame(rows)


def rhythm_of(frame, col="visit_start_date"):
    d = pd.to_datetime(frame[col], errors="coerce")
    s = pd.DataFrame({"g": frame["person_id"].astype(str),
                      "d": d}).sort_values(["g", "d"])
    gaps, spans = [], []
    for _, grp in s.groupby("g", sort=False):
        v = grp["d"].to_numpy()
        if len(v) > 1:
            gaps.extend((v[1:] - v[:-1]).astype(
                "timedelta64[D]").astype(float).tolist())
            spans.append(float((v[-1] - v[0]).astype(
                "timedelta64[D]").astype(float)))
    g = np.asarray(gaps, dtype=float)
    return (float((g == 0).mean()) if len(g) else 0.0,
            float(np.mean(spans)) if spans else 0.0)


def main():
    from synthkit import blueprint as B, discover as D, generate as G

    df = episodic()
    cat = D.discover(df, group_by="person_id")
    bp = B.build(df, cat, group_by="person_id",
                 time_col="visit_start_date", k=10)
    vr = (bp.get("patients") or {}).get("visit_rhythm")

    check("the blueprint publishes a visit RHYTHM - the span and "
          "the count were already there and the SPACING was not, "
          "and the spacing is the part that is wrong",
          bool(vr) and "same_day_share" in vr)
    check("...and it is counted in PATIENTS, never rows of visits "
          "- one person seen two hundred times would otherwise "
          "supply the whole distribution alone",
          vr.get("counted_in", "").startswith("patients"))
    check("...and it publishes a SHAPE rather than anybody's "
          "dates: a share and quantiles, no date values",
          set(vr) >= {"same_day_share", "gap_quantiles"}
          and not any("date" in str(k).lower() for k in vr))

    src_same, src_span = rhythm_of(df)
    check("the fixture is genuinely episodic, so the check is "
          "about the thing it names: same-day share {:.3f}"
          .format(src_same), src_same > 0.3)

    off_bp = copy.deepcopy(bp)
    off_bp["patients"]["visit_rhythm"] = None
    rep_off, rep_on = {}, {}
    out_off = G.generate(off_bp, seed=5, report=rep_off)
    # THE FLAGGED ARM IS WRAPPED, because a RAISE here kills the
    # suite before any `FAIL:` line and the net judges on exactly
    # that marker - a crash and a check that never ran look
    # identical. The first cut of this pass raised on the real
    # extract (`ufunc 'subtract' cannot use operands with types
    # dtype('<M8[ns]') and dtype('float64')`) and the mutation
    # reproducing it took the whole suite down rather than turning
    # one check red.
    out_on, raised = None, None
    try:
        out_on = G.generate(copy.deepcopy(bp), seed=5,
                            report=rep_on, visit_rhythm=True)
    except Exception as exc:  # noqa: BLE001
        raised = type(exc).__name__
    check("the flagged arm GENERATES at all - a pass that raises "
          "is a pass nobody can measure{}".format(
              "" if raised is None else ": RAISED " + raised),
          out_on is not None)
    if out_on is None:
        print()
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    off_same, off_span = rhythm_of(out_off)
    on_same, on_span = rhythm_of(out_on)

    check("WITHOUT the pass a patient almost never gets a same-day "
          "encounter - {:.3f} against a source {:.3f}"
          .format(off_same, src_same), off_same < 0.1)
    check("WITH it the same-day share rises sharply toward the "
          "source: {:.3f} -> {:.3f} (source {:.3f})"
          .format(off_same, on_same, src_same),
          on_same > off_same * 4)

    # THE CLAIM THE WHOLE PASS RESTS ON.
    vrp = rep_on.get("visit_rhythm") or {}
    # ASSERTED FROM THE FRAMES, NOT FROM THE REPORT. Reading
    # `multiset_preserved` back is checking what the pass SAYS: a
    # mutation hardcoding it True broke no check. The property is
    # that the two arms - same seed, one flag apart - hold the
    # SAME dates in a different arrangement.
    a = np.sort(pd.to_datetime(out_off["visit_start_date"],
                               errors="coerce").dropna()
                .astype("int64").to_numpy())
    b = np.sort(pd.to_datetime(out_on["visit_start_date"],
                               errors="coerce").dropna()
                .astype("int64").to_numpy())
    check("the column's MULTISET is preserved exactly - the two "
          "arms hold the SAME dates, measured on the frames "
          "rather than read back from the report",
          len(a) == len(b) and bool((a == b).all()))
    check("...and the ARRANGEMENT genuinely differs, so this is a "
          "reallocation rather than a pass that did nothing",
          not np.array_equal(
              pd.to_datetime(out_off["visit_start_date"],
                             errors="coerce").to_numpy(),
              pd.to_datetime(out_on["visit_start_date"],
                             errors="coerce").to_numpy()))
    check("...and the pass still REPORTS the claim, so a reader "
          "of the run sees it too",
          vrp.get("multiset_preserved") is True)

    # THE SECOND DATE COLUMN TRAVELS WITH THE FIRST, which is the
    # branch a one-date fixture cannot reach at all. The real
    # extract declares `visit_start_date <-> visit_end_date` at
    # skill 0.9999, so a reallocation that moved one and not the
    # other would destroy a near-identity - and the first cut
    # crashed outright on this branch rather than getting it
    # subtly wrong, which is the lucky version.
    def _gap(fr):
        a = pd.to_datetime(fr["visit_start_date"], errors="coerce")
        b = pd.to_datetime(fr["visit_end_date"], errors="coerce")
        return (b - a).dt.days.to_numpy(dtype=float)

    g_off, g_on = _gap(out_off), _gap(out_on)
    check("the second date column travels with the first, so each "
          "row keeps its OWN offset - the end-minus-start multiset "
          "is identical across the two arms",
          np.array_equal(np.sort(g_off[np.isfinite(g_off)]),
                         np.sort(g_on[np.isfinite(g_on)])))
    # AND THE NEIGHBOURING PROPERTY IS ASSERTED IN THE SAME
    # MEASUREMENT, which is where the first version of this check
    # went wrong. It demanded the offset never be NEGATIVE and went
    # red against correct code: on this fixture generation puts
    # 27.9% of rows with the end BEFORE the start - in the CONTROL
    # arm too, identically, because this call does not enforce
    # constraints and the source's own offsets run 0 to 5 days.
    # That is a pre-existing property `--enforce-constraints`
    # exists for, and demanding it here would have blamed the
    # reallocation for something it did not do.
    ok_off = g_off[np.isfinite(g_off)]
    ok_on = g_on[np.isfinite(g_on)]
    check("...and the share of rows whose end precedes their start "
          "is UNCHANGED by the reallocation - {:.4f} off against "
          "{:.4f} on, so the pass neither causes nor repairs it"
          .format(float((ok_off < 0).mean()),
                  float((ok_on < 0).mean())),
          abs(float((ok_off < 0).mean())
              - float((ok_on < 0).mean())) < 1e-12)

    # AND THE COST IS REPORTED RATHER THAN DISCOVERED LATER.
    check("the run REPORTS the span it produced, so a rhythm "
          "bought by spending the patient's range is visible: "
          "asked {:.3f} got {}".format(
              vrp.get("same_day_asked", 0), vrp.get("same_day_got")),
          vrp.get("patient_span_mean") is not None
          and vrp.get("same_day_got") is not None)

    # OFF BY DEFAULT, asserted through generate rather than argued.
    rep_d = {}
    G.generate(copy.deepcopy(bp), seed=5, report=rep_d)
    check("the pass is OFF by default - it is a measured TRADE "
          "(span {:.0f} against {:.0f} off) and this project does "
          "not default on fixture evidence".format(on_span, off_span),
          "visit_rhythm" not in rep_d)

    # A BLUEPRINT WITHOUT THE RECORD MUST STILL GENERATE.
    old = copy.deepcopy(bp)
    old["patients"].pop("visit_rhythm", None)
    rep_o = {}
    G.generate(old, seed=5, report=rep_o, visit_rhythm=True)
    check("a blueprint written before the rhythm existed still "
          "generates with the flag ON - absent is not empty, and "
          "an older artifact must still load",
          "visit_rhythm" not in rep_o)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
