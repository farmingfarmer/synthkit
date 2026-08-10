"""Discover, explain, blueprint and generate - one command, on a
machine this code has never run on.

    python scripts/run_discovery.py --src TIDY.csv --out DIR

Writes into DIR, which the person running it chooses:

    catalogue.json   every claim with its evidence
    blueprint.json   the editable contract - marginals, effect curves,
                     dynamics, and a dial beside each measured value
    findings.txt     the same thing in sentences, for reading
    generated.csv    synthetic rows            (with --generate)
    fidelity.json    source against generated  (with --generate)

WRITTEN TO RUN WHERE IT CANNOT BE DEBUGGED. Progress prints per column
with a running clock, because 170 columns take minutes and a silent
process looks hung. Every failure that can be anticipated is caught
and explained in a sentence rather than a traceback. Nothing is
written outside DIR.

NOTHING HERE UPLOADS, PRINTS OR RETURNS A RECORD. The catalogue and
blueprint hold aggregates only - quantiles, level shares, response
curves, correlations. findings.txt names columns and shapes. If any of
it is to leave the machine, read it first; that is a decision for the
person holding the data, not for this script.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

T0 = time.time()


def say(msg):
    print("[{:7.1f}s] {}".format(time.time() - T0, msg), flush=True)


def die(msg):
    print("\nSTOPPED: {}".format(msg), file=sys.stderr, flush=True)
    raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser(
        description="Find patterns, explain them, and generate data "
                    "carrying them.")
    ap.add_argument("--src", required=True,
                    help="tidy CSV: one row per visit")
    ap.add_argument("--out", required=True,
                    help="output directory (you choose it; it is "
                         "created if missing)")
    ap.add_argument("--group-by", default="person_id",
                    help="the patient id column")
    ap.add_argument("--time-col", default="",
                    help="visit ordering column; detected if omitted")
    ap.add_argument("--lags", action="store_true",
                    help="add col__prev features so a LAGGED "
                         "cross-column effect can be found at all")
    ap.add_argument("--generate", action="store_true",
                    help="also write synthetic rows and compare them "
                         "against the source")
    ap.add_argument("--patients", type=int, default=0,
                    help="how many patients to generate (default: as "
                         "many as the source had)")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="read only the first N rows - for a quick "
                         "first pass before committing to the full "
                         "extract")
    ap.add_argument("--exclude", default="",
                    help="comma-separated columns to leave out "
                         "entirely - bookkeeping the wrangler "
                         "computed rather than anything the clinic "
                         "recorded")
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--holdout", type=float, default=0.3,
                    help="share of PATIENTS held out; every number "
                         "reported is measured on them")
    a = ap.parse_args()

    # ECHO WHAT ACTUALLY ARRIVED, before anything else runs.
    #
    # This terminal has mangled the command line three times - a
    # wrapped paste once put `--src` inside the filename, and twice a
    # flag simply never reached the program while the log looked
    # normal. Reading the settings back is the difference between
    # diagnosing that in one line and losing a round trip guessing
    # whether the code or the invocation was at fault.
    say("invocation: --src {} --out {}{}{}{}{}{}".format(
        a.src, a.out,
        " --group-by " + a.group_by if a.group_by != "person_id"
        else "",
        " --time-col " + a.time_col if a.time_col
        else " (no --time-col: the axis will be DETECTED)",
        " --lags" if a.lags else "",
        " --generate" if a.generate else "",
        " --exclude " + a.exclude if a.exclude else ""))

    try:
        import numpy  # noqa: F401
        import pandas as pd
        import sklearn  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as e:
        die("a required package is missing ({}). Install them with:\n"
            "    python -m pip install -r requirements.txt".format(e))

    src = Path(a.src)
    if not src.exists():
        die("no such file: {}".format(src))
    out = Path(a.out)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        die("cannot create the output directory {}: {}".format(out, e))

    from synthkit import blueprint as B
    from synthkit.discover import discover
    from synthkit.generate import generate

    say("reading {}".format(src))
    try:
        df = pd.read_csv(src, dtype=str, keep_default_na=False,
                         encoding="utf-8-sig",
                         nrows=(a.max_rows or None))
    except Exception as e:
        die("could not read {} as CSV: {}".format(src, e))
    if a.group_by not in df.columns:
        die("--group-by {!r} is not a column in {}. Columns are: {}"
            .format(a.group_by, src.name,
                    ", ".join(list(df.columns)[:25])))
    say("{} rows x {} columns, {} patients".format(
        len(df), df.shape[1], df[a.group_by].nunique()))

    drop = [c.strip() for c in a.exclude.split(",") if c.strip()]
    unknown = [c for c in drop if c not in df.columns]
    if unknown:
        die("--exclude names {} which {} not in the file"
            .format(", ".join(unknown),
                    "is" if len(unknown) == 1 else "are"))
    if drop:
        df = df.drop(columns=drop)
        say("excluded {} column(s): {}".format(len(drop),
                                               ", ".join(drop)))

    time_col = a.time_col or None
    if a.lags:
        from synthkit.temporal import (add_lag_features,
                                       column_time_kind,
                                       detect_time_column)
        rows = df.to_dict("records")
        tc, tk, ev = detect_time_column(rows, a.group_by)
        # AN EXPLICIT --time-col MUST ACTUALLY BE USED. It was carried
        # into the blueprint and then ignored when the lag features
        # were built, which are what every temporal statistic is
        # measured on - so passing it changed nothing and the message
        # still named the detected column. Detection picks the date
        # with the most distinct values, which chose visit_END_date on
        # a real extract; an end date misorders overlapping stays.
        if time_col:
            if time_col not in df.columns:
                die("--time-col {!r} is not a column in {}"
                    .format(time_col, src.name))
            kind = column_time_kind(rows, a.group_by, time_col)
            if kind is None:
                die("--time-col {!r} does not look like a date or a "
                    "sequence, so it cannot order visits"
                    .format(time_col))
            if time_col != tc:
                say("time axis: {} ({}) AS REQUESTED - detection "
                    "would have chosen {}".format(time_col, kind, tc))
            else:
                say("time axis: {} ({}) - as requested, and detection "
                    "agrees".format(time_col, kind))
            tc, tk = time_col, kind
        else:
            say("time axis: {} ({}) - {}".format(tc or "file order",
                                                 tk, ev))
            time_col = tc
        rows, lrep = add_lag_features(rows, a.group_by, tc, tk)
        df = pd.DataFrame(rows)
        say("{} columns earned a lag feature -> {} columns".format(
            len(lrep["lagged_columns"]), df.shape[1]))
    elif not time_col:
        from synthkit.temporal import detect_time_column
        tc, tk, ev = detect_time_column(df.to_dict("records"),
                                        a.group_by)
        time_col = tc
        say("time axis: {} ({})".format(tc or "file order", tk))

    n_cols = df.shape[1]
    say("searching {} columns - this is the slow part".format(n_cols))

    def progress(i, n, col):
        if i and i % 10 == 0:
            done = max(i, 1)
            rate = (time.time() - T0) / done
            say("  {}/{} columns, about {:.0f}s left ({})".format(
                i, n, rate * (n - i), col))

    cat = discover(df, group_by=a.group_by, seed=a.seed,
                   holdout_frac=a.holdout, progress=progress)
    say("{} claims | {} unexplained | {} skipped | {} identifiers "
        "dropped".format(len(cat["claims"]), len(cat["unexplained"]),
                         len(cat["skipped"]),
                         len(cat.get("identifiers_dropped") or [])))
    (out / "catalogue.json").write_text(
        json.dumps(cat, indent=1), encoding="utf-8")

    say("building the blueprint")
    bp = B.build(df, cat, group_by=a.group_by, time_col=time_col)
    probs = B.validate(bp)
    if probs:
        say("WARNING - the blueprint it just built does not validate, "
            "which is a bug rather than an edit:")
        for p in probs[:10]:
            say("    {}".format(p))
    (out / "blueprint.json").write_text(
        json.dumps(bp, indent=1), encoding="utf-8")
    (out / "findings.txt").write_text(render(bp), encoding="utf-8")
    # A MIRROR IS NEVER A LOSS, whatever else is true of it. The
    # relationship is generated the other way round, so a mirrored
    # edge whose child kept no other parent still reaches the data.
    orphans = [d for d in _will_drop(bp)
               if not d.get("harmless")
               and not d.get("child_keeps_parents")]
    if orphans:
        say("{} relationship(s) will NOT reach the generated data - "
            "see the end of findings.txt".format(len(orphans)))
    say("wrote catalogue.json, blueprint.json, findings.txt")

    if not a.generate:
        say("done. Read findings.txt first; edit blueprint.json to "
            "tune, then re-run with --generate")
        return

    say("generating")
    rep = {}
    g = generate(bp, n_patients=(a.patients or None), seed=a.seed,
                 report=rep)
    g.to_csv(out / "generated.csv", index=False, encoding="utf-8")
    say("{} rows for {} patients -> generated.csv".format(
        rep["rows"], rep["patients"]))
    if rep.get("edges_dropped"):
        say("{} relationship(s) dropped to keep the graph "
            "sampleable".format(len(rep["edges_dropped"])))

    say("comparing source against generated")
    fid = compare(df, g, bp, a.group_by, time_col)
    fid["generation"] = rep
    (out / "fidelity.json").write_text(
        json.dumps(fid, indent=1), encoding="utf-8")
    s = fid["summary"]
    say("coverage within 0.05 on {}/{} columns".format(
        s["coverage_ok"], s["columns"]))
    say("centre within 10% of spread on {}/{} numeric columns".format(
        s["centre_ok"], s["numeric"]))
    say("persistence within 0.15 on {}/{} numeric columns".format(
        s["lag1_ok"], s["numeric_dynamic"]))
    say("clustering within 0.15 on {}/{} partly-covered "
        "columns".format(s["cluster_ok"], s["partly_covered"]))
    say("done -> {}".format(out))


def _will_drop(bp):
    """What the sampler will have to discard, worked out before it
    runs, so the reader is told in the document they actually read."""
    try:
        from synthkit.blueprint import resolve
        from synthkit.generate import _order
        return _order(resolve(bp))[2]
    except Exception:
        return []


def render(bp):
    """The blueprint in sentences. This is the part meant to be read
    by somebody who has never seen the data."""
    L = ["WHAT THE DATA SAYS", "=" * 60, ""]
    p = bp.get("patients") or {}
    if p.get("count"):
        L.append("{} patients, {} rows, {:.1f} visits each on "
                 "average.".format(p["count"], p["rows"],
                                   p["visits"]["mean"]))
    ex = bp.get("excluded") or {}
    if ex.get("identifiers"):
        L.append("Excluded as identifiers: {}".format(
            ", ".join(ex["identifiers"])))
    L.append("")
    L.append("RELATIONSHIPS FOUND, strongest first. Every number was "
             "measured on")
    L.append("patients the search never saw.")
    L.append("")
    rels = sorted(bp.get("relationships") or [],
                  key=lambda r: -(r.get("evidence") or {}).get(
                      "skill_out_of_sample", 0))
    # A catalogue is direction-agnostic, so `age <- year_of_birth` and
    # `year_of_birth <- age` are both true and both reported. That is
    # correct in the data and noise on the page: the same relationship
    # twice. The stronger direction is shown once and the mirror is
    # named as symmetric rather than dropped without saying so.
    seen, shown = {}, []
    for r in rels:
        key = frozenset([r["child"]] + list(r["parents"]))
        if key in seen:
            seen[key]["_mirrored"] = True
            continue
        seen[key] = r
        shown.append(r)
    rels = shown
    # ARITHMETIC IS SEPARATED FROM DISCOVERY, not hidden. A column
    # computed from another sits at the top of any ranking by skill
    # and pushes the real findings off the first page. Both are
    # printed; only one is called a finding.
    arith = [r for r in rels
             if (r.get("evidence") or {}).get("near_deterministic")]
    rels = [r for r in rels if r not in arith]
    for r in rels:
        ev = r["evidence"]
        L.append("{} <- {}".format(r["child"], ", ".join(r["parents"])))
        L.append("    explains {:.0%} of it, on {} held-out "
                 "patients{}".format(
                     ev["skill_out_of_sample"],
                     ev.get("holdout_patients"),
                     " (holds in both directions)"
                     if r.get("_mirrored") else ""))
        for col, e in (ev.get("effect") or {}).items():
            if e.get("description"):
                L.append("    {}".format(e["description"]))
        it = ev.get("interaction")
        if it and it.get("description"):
            L.append("    [{}] {}".format(it["pattern"],
                                          it["description"]))
        L.append("")
    if not rels:
        L.append("(none)")
        L.append("")
    if arith:
        L.append("")
        L.append("NEAR-DETERMINISTIC, listed separately. Each of "
                 "these is one column")
        L.append("computed from another - a subtraction, a threshold, "
                 "a restatement. True,")
        L.append("and not a discovery. The cut is at 97% and is a "
                 "rule of thumb, not a law:")
        L.append("on this data arithmetic measured 98-100% and the "
                 "strongest genuine")
        L.append("relationship measured 96%. Check any that surprise "
                 "you.")
        L.append("")
        for r in arith:
            ev = r["evidence"]
            L.append("{} <- {}   ({:.0%})".format(
                r["child"], ", ".join(r["parents"]),
                ev["skill_out_of_sample"]))
            for col, e in (ev.get("effect") or {}).items():
                if e.get("description"):
                    L.append("    {}".format(e["description"]))
            L.append("")
    L.append("")
    L.append("COLUMNS NOTHING EXPLAINED: {}".format(
        ", ".join(ex.get("unexplained") or []) or "(none)"))
    dropped = _will_drop(bp)
    mirrors = [d for d in dropped if d.get("harmless")]
    trimmed = [d for d in dropped if not d.get("harmless")
               and d.get("partial")]
    orphans = [d for d in dropped if not d.get("harmless")
               and not d.get("partial")
               and not d.get("child_keeps_parents")]
    other = [d for d in dropped if d not in mirrors
             and d not in trimmed and d not in orphans]
    L.append("")
    L.append("WHAT WILL NOT REACH THE GENERATED DATA")
    L.append("-" * 60)
    L.append("A catalogue may hold a relationship in both directions "
             "and may hold loops.")
    L.append("A sampler cannot: something has to be drawn first. "
             "{} were dropped,".format(len(dropped)))
    L.append("in {} kinds - and the counts below add up to that "
             "total.".format(
                 sum(1 for g in (mirrors, trimmed, orphans, other)
                     if g)))
    L.append("")
    if mirrors:
        L.append("{} RESTATEMENTS of structure already taken - the "
                 "same dependence,".format(len(mirrors)))
        L.append("generated the other way round. Nothing is missing.")
        L.append("")
    if trimmed:
        # Reported because it IS a partial loss. The column is still
        # explained, but one specific parent's contribution to it is
        # not in the data - and the first version of this section
        # omitted the category entirely, so ten of twelve drops on the
        # real extract were invisible to the reader.
        L.append("{} had SOME PARENTS REMOVED to break a loop. The "
                 "column is still".format(len(trimmed)))
        L.append("explained by the parents that remain, but the ones "
                 "listed here contribute")
        L.append("nothing to it in the generated data:")
        L.append("")
        for d in sorted(trimmed, key=lambda x: -x.get("skill", 0)):
            L.append("    {} lost {}   (kept {})".format(
                d["child"], ", ".join(d["parents"]),
                ", ".join(d.get("kept_parents") or []) or "nothing"))
        L.append("")
    if other:
        L.append("{} were dropped for other reasons:".format(
            len(other)))
        for d in other:
            L.append("    {} <- {}   ({})".format(
                d["child"], ", ".join(d["parents"]),
                d["why"].split(";")[0]))
        L.append("")
    if orphans:
        L.append("{} left their column with NO parent at all. Each of "
                 "these was shown".format(len(orphans)))
        L.append("above as a finding and is NOT in generated.csv - "
                 "the column is drawn from")
        L.append("its own distribution and nothing else:")
        L.append("")
        for d in sorted(orphans, key=lambda x: -x.get("skill", 0)):
            L.append("    {} <- {}   ({:.0%}, {})".format(
                d["child"], ", ".join(d["parents"]),
                d.get("skill", 0.0), d["why"].split(";")[0]))
        L.append("")
    if not dropped:
        L.append("(nothing was dropped)")
        L.append("")
    L.append("")
    L.append("TO CHANGE ANY OF IT, edit blueprint.json. Each column "
             "and each")
    L.append("relationship has a `dials` block beside the measured "
             "value. null means")
    L.append("use what was measured. strength 0 removes a "
             "relationship, 1 is as")
    L.append("measured, above 1 extrapolates past anything observed.")
    return "\n".join(L)


def compare(df, g, bp, group_by, time_col):
    """Source against generated, column by column."""
    import numpy as np
    import pandas as pd
    from synthkit.discover import prepare
    from synthkit.dynamics import measure

    Xs, _ = prepare(df, group_by)
    Xg, _ = prepare(g, group_by)
    gt = "visit_number" if "visit_number" in g.columns else None
    ds = measure(df, Xs, group_by, time_col)
    dg = measure(g, Xg, group_by, gt)

    cols, n_ok = [], dict(cov=0, ctr=0, lag=0, clu=0, num=0, dyn=0,
                          part=0)
    for c in Xs.columns:
        if c not in Xg.columns:
            continue
        s, q = Xs[c], Xg[c]
        row = {"column": c,
               "coverage_source": round(float(s.notna().mean()), 4),
               "coverage_generated": round(float(q.notna().mean()), 4)}
        row["coverage_delta"] = round(
            row["coverage_generated"] - row["coverage_source"], 4)
        if abs(row["coverage_delta"]) <= 0.05:
            n_ok["cov"] += 1
        if pd.api.types.is_numeric_dtype(s) and \
                pd.api.types.is_numeric_dtype(q):
            n_ok["num"] += 1
            sd = float(s.std() or 0.0)
            row.update({
                "mean_source": round(float(s.mean()), 4),
                "mean_generated": round(float(q.mean()), 4),
                "sd_source": round(sd, 4),
                "sd_generated": round(float(q.std() or 0.0), 4)})
            if sd > 0 and abs(row["mean_generated"]
                              - row["mean_source"]) <= 0.1 * sd:
                n_ok["ctr"] += 1
            a_, b_ = ds.get(c, {}), dg.get(c, {})
            row.update({"icc_source": a_.get("icc"),
                        "icc_generated": b_.get("icc"),
                        "lag1_source": a_.get("lag1_total"),
                        "lag1_generated": b_.get("lag1_total")})
            if a_.get("lag1_total") is not None and \
                    b_.get("lag1_total") is not None:
                n_ok["dyn"] += 1
                if abs(a_["lag1_total"] - b_["lag1_total"]) <= 0.15:
                    n_ok["lag"] += 1
        if 0.02 < row["coverage_source"] < 0.98:
            n_ok["part"] += 1
            cs = (ds.get(c) or {}).get("missing_clustering")
            cg = (dg.get(c) or {}).get("missing_clustering")
            row.update({"clustering_source": cs,
                        "clustering_generated": cg})
            if cs is not None and cg is not None and \
                    abs(cs - cg) <= 0.15:
                n_ok["clu"] += 1
        cols.append(row)

    return {
        "columns": cols,
        "summary": {
            "columns": len(cols), "numeric": n_ok["num"],
            "numeric_dynamic": n_ok["dyn"],
            "partly_covered": n_ok["part"],
            "coverage_ok": n_ok["cov"], "centre_ok": n_ok["ctr"],
            "lag1_ok": n_ok["lag"], "cluster_ok": n_ok["clu"],
        },
        "note": "coverage_ok counts columns within 0.05 of the "
                "source's share of present values; centre_ok within "
                "a tenth of the column's own spread; lag1_ok and "
                "cluster_ok within 0.15 of the source statistic. "
                "These are aggregates - no record is reproduced here.",
    }


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted - nothing further was written",
              file=sys.stderr)
        raise SystemExit(130)
