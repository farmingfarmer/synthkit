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
    ap.add_argument("--ordinal", action="append", default=[],
                    metavar="COL=a,b,c",
                    help="declare an ORDER on a categorical, lowest "
                         "first: --ordinal severity=mild,moderate,"
                         "severe. Repeatable. Nothing infers this - "
                         "mild/moderate/severe and north/south/east/"
                         "west look identical to any test on the "
                         "strings, and inventing an order the data "
                         "never declared is worse than missing it")
    ap.add_argument("--emit-spec", action="store_true",
                    help="also write tablespec.json - the fitted "
                         "marginals as an authorable TableSpec, so a "
                         "campaign can run on measured shape instead "
                         "of guessed. Relationships and dynamics do "
                         "NOT cross; the file says so itself")
    ap.add_argument("--enforce-constraints", action="store_true",
                    help="repair orderings the source never broke, "
                         "such as a visit ending before it began, by "
                         "SWAPPING the two values - which leaves both "
                         "columns' distributions exactly as they "
                         "were. Off by default because it changes the "
                         "output")
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
    say("invocation: --src {} --out {}{}{}{}{}{}{}{}{}".format(
        a.src, a.out,
        " --group-by " + a.group_by if a.group_by != "person_id"
        else "",
        " --time-col " + a.time_col if a.time_col
        else " (no --time-col: the axis will be DETECTED)",
        " --lags" if a.lags else "",
        " --generate" if a.generate else "",
        " --enforce-constraints" if a.enforce_constraints else "",
        "".join(" --ordinal " + o for o in a.ordinal),
        " --emit-spec" if a.emit_spec else "",
        " --exclude " + a.exclude if a.exclude else ""))

    try:
        import numpy  # noqa: F401
        import pandas as pd
        import sklearn  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as e:
        die("a required package is missing ({}). Install them with:\n"
            "    python -m pip install -r requirements.txt".format(e))

    ordinals = {}
    for item in a.ordinal:
        if "=" not in item:
            die("--ordinal wants COL=level1,level2,... - got {!r}"
                .format(item))
        col, _, rest = item.partition("=")
        levels = [x.strip() for x in rest.split(",") if x.strip()]
        if len(levels) < 2:
            die("--ordinal {} needs at least two levels, lowest first"
                .format(col))
        if len(set(levels)) != len(levels):
            die("--ordinal {} repeats a level".format(col))
        ordinals[col.strip()] = levels

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
    unknown_ord = [c for c in ordinals if c not in df.columns]
    if unknown_ord:
        die("--ordinal names {} which {} not a column in {}"
            .format(", ".join(unknown_ord),
                    "is" if len(unknown_ord) == 1 else "are",
                    src.name))
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
                   holdout_frac=a.holdout, progress=progress,
                   ordinals=ordinals)
    say("{} claims | {} unexplained | {} skipped | {} identifiers "
        "dropped".format(len(cat["claims"]), len(cat["unexplained"]),
                         len(cat["skipped"]),
                         len(cat.get("identifiers_dropped") or [])))
    (out / "catalogue.json").write_text(
        json.dumps(cat, indent=1), encoding="utf-8")

    say("building the blueprint")
    bp = B.build(df, cat, group_by=a.group_by, time_col=time_col,
                 ordinals=ordinals)
    # WHICH COLUMNS WERE READ AS DATES, AND WHAT IT COST.
    #
    # A date that does not parse becomes missing, and missing is
    # exactly the direction a whole column was once destroyed in while
    # coverage still read 1.0. On the tidy fixture 1.9% of
    # visit_start_date is impossible - 2002-09-31, 2002-02-29 in a
    # non-leap year - and real extracts carry typing errors too. Under
    # the 0.05 coverage bar it would pass silently, so it is said out
    # loud here rather than left in blueprint.json for nobody to open.
    dated = [(c, v["date"]) for c, v in (bp.get("columns") or {}).items()
             if v.get("date")]
    if dated:
        say("read as dates: {}".format(", ".join(
            "{} ({})".format(c, d["format"]) for c, d in dated)))
        for c, d in dated:
            if d.get("unparsed_share"):
                say("  WARNING {}: {:.2%} of present values are not "
                    "valid dates and became MISSING - check for a "
                    "second format or impossible days".format(
                        c, d["unparsed_share"]))
            if d.get("ambiguous"):
                say("  WARNING {}: day/month order is AMBIGUOUS; read "
                    "as {} over {}".format(c, d["parsed_format"],
                                           ", ".join(d["alternatives"])))
            if d.get("floored_to_day"):
                say("  {}: a time of day was present and floored to "
                    "the day".format(c))
    # HOW EVERY COLUMN WAS TYPED, before an hour is spent on it.
    #
    # A column read as the wrong type is the most expensive fault this
    # tool has: it is silent, it survives every per-column check, and
    # it is only visible by opening the blueprint. Dates went that way
    # at 88% sentinel; a currency column and a time-of-day column have
    # since gone the same way at 100%. One glance at this list would
    # have caught all three.
    say("column types:")
    destroyed = []
    for c, spec in sorted((bp.get("columns") or {}).items()):
        mg = spec.get("marginal") or {}
        if spec.get("kind") == "numeric":
            what = "numeric"
            if spec.get("date"):
                what = "date {}".format(spec["date"]["format"])
            elif mg.get("integral"):
                what = "numeric (whole numbers)"
        elif mg.get("type") == "suppressed":
            what = "SUPPRESSED - too few patients to publish"
        elif mg.get("type") == "list":
            # Named here rather than warned about. It was a warning
            # while nothing could model these; now that the set is
            # modelled as a set, the honest report is what it became.
            sz = mg.get("set_size") or {}
            avg = sum(float(v) * float(p_) for v, p_
                      in zip(sz.get("v") or [1], sz.get("p") or [1.0]))
            what = ("SET of {} tokens, {:.1f} per row, from {} "
                    "distinct combinations".format(
                        len(mg.get("tokens") or []), avg,
                        mg.get("distinct_combinations", 0)))
        else:
            n = len(mg.get("levels") or [])
            sent = float(mg.get("sentinel_share") or 0.0)
            what = "categorical, {} level(s)".format(n)
            if sent >= 0.5:
                what += "  <-- {:.0%} is the `__other__` SENTINEL".format(
                    sent)
                destroyed.append((c, sent))
        say("  {:<28} {}".format(c, what))
    if destroyed:
        say("")
        say("STOP AND READ THIS. {} column(s) are mostly or entirely "
            "the sentinel:".format(len(destroyed)))
        for c, sent in sorted(destroyed, key=lambda x: -x[1]):
            say("  {}: {:.0%} `__other__` - too many distinct values "
                "to publish as labels".format(c, sent))
        say("  These columns are DESTROYED in the output, and every "
            "per-column check will still pass,")
        say("  because coverage counts whether a value is present and "
            "the sentinel is present.")
        say("  A number wearing punctuation does this - currency, a "
            "percent sign, a time of day, an")
        say("  identifier. Check whether these are really categories "
            "before trusting anything downstream.")

    # IS A SUPPRESSED CATEGORICAL ACTUALLY LIST-VALUED?
    #
    # Four columns lost 10-28% of their content to level suppression
    # on the real run - conditions, procedures, active_drugs,
    # drug_routes. If a patient has several drugs recorded in one
    # field, then every distinct COMBINATION becomes its own level,
    # the combinations explode, and almost all of them fall under k.
    # The column would then be suppressed not because its values are
    # rare but because it is the wrong shape for a single categorical
    # - the same class of fault as a date modelled as 200 labels.
    #
    # That is a HYPOTHESIS about those four columns and cannot be
    # settled from a machine with no clinical data on it. So the run
    # measures it: how many present values carry a separator, and how
    # much of the column was suppressed. One run confirms or kills it.
    listy = []
    for c, spec in (bp.get("columns") or {}).items():
        mg = spec.get("marginal") or {}
        lost = ((mg.get("suppressed_levels") or {}).get("share") or 0.0)
        lost += ((mg.get("tail") or {}).get("share_omitted") or 0.0)
        # Only columns that fell back to being ONE LABEL. A column
        # already modelled as a set is not a problem to report.
        if mg.get("type") != "levels" or lost < 0.05:
            continue
        if c not in df.columns:
            continue
        vals = df[c].astype(str).str.strip()
        vals = vals[vals.str.len() > 0]
        if not len(vals):
            continue
        for sep in (";", "|", ","):
            share = float(vals.str.contains(sep, regex=False).mean())
            if share >= 0.3:
                listy.append((c, sep, share, lost,
                              int(vals.nunique())))
                break
    if listy:
        say("these look LIST-VALUED but could NOT be modelled as sets "
            "- too few tokens cleared k, so every combination is "
            "still its own level:")
        for c, sep, share, lost, nun in listy:
            say("  {}: {:.0%} of values contain {!r}, {} distinct "
                "combinations, {:.0%} of the column suppressed or "
                "truncated".format(c, share, sep, nun, lost))
        say("  (modelling these as a set of indicators rather than one "
            "label would recover most of that)")

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
    # WHAT MADE THIS, so a file found in six months can be placed.
    #
    # Nothing recorded which extract, which code, which seed or which
    # flags produced an output directory. A synthetic file that
    # circulates without that is one somebody eventually analyses
    # believing it is real, or compares against a run it has nothing
    # to do with.
    #
    # The source is recorded by NAME, not by path. The path on the
    # machine holding the extract contains a work login, and this file
    # is the one most likely to travel with the output - the same
    # reason `smoke_no_personal` bans that pattern from everything
    # tracked.
    import datetime as _dt
    try:
        from synthkit.gui import build_fingerprint
        code = build_fingerprint()
    except Exception:
        code = None
    prov = {
        "made_at": _dt.datetime.now().replace(microsecond=0).isoformat(),
        "code_fingerprint": code,
        "blueprint_version": bp.get("blueprint_version"),
        "source": {
            "name": src.name,
            "rows_read": int(len(df)),
            "columns_read": int(df.shape[1]),
            "patients": int(df[a.group_by].nunique()),
            "max_rows": a.max_rows or None,
        },
        "settings": {
            "group_by": a.group_by,
            "time_col_requested": a.time_col or None,
            "time_col_used": time_col,
            "lags": bool(a.lags),
            "generate": bool(a.generate),
            "seed": a.seed,
            "holdout": a.holdout,
            "excluded": drop,
        },
        "privacy": {"k": (bp.get("privacy") or {}).get("k")},
        "note": "the source is named, not pathed, on purpose: this "
                "file travels with the output and a path on that "
                "machine carries a work login.",
    }
    if a.emit_spec:
        from synthkit.bridge import blueprint_to_tablespec
        made = blueprint_to_tablespec(bp, title=src.stem)
        (out / "tablespec.json").write_text(
            json.dumps(made, indent=1), encoding="utf-8")
        car = made["carried"]
        say("wrote tablespec.json - {} column(s) crossed, {} dropped"
            .format(len(made["tablespec"]["columns"]),
                    len(car["columns_dropped"])))
        say("  what did NOT cross: {}".format(
            ", ".join(car["did_not_cross"][:5])))
        say("  outcomes are EMPTY by construction - a campaign grades "
            "against planted signal")
        say("  whose answer is known, and no fitted blueprint can "
            "supply one. Author them.")
    (out / "provenance.json").write_text(
        json.dumps(prov, indent=1), encoding="utf-8")
    say("wrote catalogue.json, blueprint.json, findings.txt, "
        "provenance.json")

    if not a.generate:
        say("done. Read findings.txt first; edit blueprint.json to "
            "tune, then re-run with --generate")
        return

    say("generating")
    rep = {}
    g = generate(bp, n_patients=(a.patients or None), seed=a.seed,
                 report=rep,
                 enforce_constraints=a.enforce_constraints)
    g.to_csv(out / "generated.csv", index=False, encoding="utf-8")
    say("{} rows for {} patients -> generated.csv".format(
        rep["rows"], rep["patients"]))
    if rep.get("edges_dropped"):
        say("{} relationship(s) dropped to keep the graph "
            "sampleable".format(len(rep["edges_dropped"])))

    say("comparing source against generated")
    fid = compare(df, g, bp, a.group_by, time_col,
                  ordinals=ordinals)
    fid["generation"] = rep
    (out / "fidelity.json").write_text(
        json.dumps(fid, indent=1), encoding="utf-8")
    # THE READABLE HALF, appended to the document people actually
    # open. findings.txt is written before generation, so this is the
    # only place the per-column verdicts can reach it.
    with (out / "findings.txt").open("a", encoding="utf-8") as fh:
        fh.write(render_verdicts(fid) + "\n")
    s = fid["summary"]
    say("coverage within 0.05 on {}/{} columns".format(
        s["coverage_ok"], s["columns"]))
    say("centre within 10% of spread on {}/{} numeric columns".format(
        s["centre_ok"], s["numeric"]))
    # A COUNT CANNOT BE DIAGNOSED. 18/34 was reported once with
    # nothing to say which mechanism moved them, so the misses are
    # split here by whether the known one - a skewed column whose
    # extreme segment had no published tail mean to shape it - could
    # account for them at all.
    misses = [(c["column"], c["centre_miss"]) for c in fid["columns"]
              if c.get("centre_miss")]
    if misses:
        skewed = [m for _c, m in misses
                  if abs(m.get("skew_source") or 0.0) >= 1.0]
        unshaped = [m for _c, m in misses
                    if not m["tail_shape_published"]]
        say("  of the {} that missed: {} are skewed (|skew| >= 1), "
            "{} had no tail shape published".format(
                len(misses), len(skewed), len(unshaped)))
        say("  worst: " + ", ".join(
            "{} {:.2f}sd {}".format(c, m["by_sd"],
                                    "skew {:.1f}".format(m["skew_source"])
                                    if m.get("skew_source") is not None
                                    else "")
            for c, m in sorted(misses, key=lambda x: -x[1]["by_sd"])[:4]))
    say("spread within 25% on {}/{} numeric columns".format(
        s["spread_ok"], s["numeric"]))
    narrow = [(c["column"], c["spread_miss"]) for c in fid["columns"]
              if c.get("spread_miss")]
    if narrow:
        # A SHORTFALL THE BOUND EXPLAINS IS NOT A FAULT. Naming the
        # share of the column that sits outside the published bound is
        # what stops this line being read as a bug when it is the k
        # rule doing its job.
        bounded = [m for _c, m in narrow
                   if (m.get("share_of_magnitude_outside_bounds") or 0)
                   > 0.1]
        say("  of the {} that missed: {} lose most of their magnitude "
            "to the published bound (privacy, not a fault)".format(
                len(narrow), len(bounded)))
        say("  worst: " + ", ".join(
            "{} {:.0%} of source{}".format(
                c, m["ratio"],
                "" if not m.get("share_of_magnitude_outside_bounds")
                else " ({:.0%} beyond bound)".format(
                    m["share_of_magnitude_outside_bounds"]))
            for c, m in sorted(narrow, key=lambda x: x[1]["ratio"])[:4]))
    say("persistence within 0.15 on {}/{} numeric columns".format(
        s["lag1_ok"], s["numeric_dynamic"]))
    say("clustering within 0.15 on {}/{} partly-covered "
        "columns".format(s["cluster_ok"], s["partly_covered"]))
    say("RELATIONSHIPS: {}/{} keep their direction, {}/{} land within "
        "0.2".format(s["pairs_sign_ok"], s["pairs"],
                     s["pairs_close"], s["pairs"]))
    if s.get("constraints_checked"):
        say("orderings the source never broke, held on {}/{} in the "
            "generated data".format(s["constraints_held"],
                                    s["constraints_checked"]))
        for c in fid.get("constraints") or []:
            if c["holds_in_generated"] < 0.999:
                say("  {} <= {} broken on {} rows ({:.1%}){}".format(
                    c["lhs"], c["rhs"], c["rows_violating"],
                    1.0 - c["holds_in_generated"],
                    "" if a.enforce_constraints
                    else " - --enforce-constraints repairs this by "
                         "swapping the pair, which leaves both "
                         "distributions untouched"))
    if s.get("categorical_compared"):
        say("categorical and mixed associations kept on {}/{} pairs "
            "- these were never measured before".format(
                s["categorical_kept"], s["categorical_compared"]))
        for r in (fid["relationships"].get("categorical_weakened")
                  or [])[:4]:
            say("  {} <- {}: {} {:.2f} -> {:.2f}".format(
                r["child"], r["parent"], r["measure"],
                r["source"], r["generated"]))
    if s.get("deterministic_compared"):
        say("near-deterministic identities hold on {}/{} pairs".format(
            s["deterministic_kept"], s["deterministic_compared"]))
        for r in (fid["relationships"].get("deterministic_loosened")
                  or [])[:4]:
            say("  {} <- {}: tightness {:.2f} -> {:.2f} - the identity "
                "is looser in the generated data".format(
                    r["child"], r["parent"], r["tightness_source"],
                    r["tightness_generated"]))
    if s["pairs_inverted"]:
        say("  {} relationship(s) came out INVERTED - the opposite "
            "sign to the source. This is worse than a missing one; it "
            "reads as a finding:".format(s["pairs_inverted"]))
        for r in fid["relationships"]["inverted"][:6]:
            say("    {} ~ {}   source {:+.3f}   generated {:+.3f}"
                .format(r["child"], r["parent"], r["source"],
                        r["generated"]))
    say("done -> {}".format(out))


def _will_drop(bp):
    """What the sampler will have to discard, worked out before it
    runs, so the reader is told in the document they actually read."""
    return _will_drop_full(bp)[0]


def _will_drop_full(bp):
    """(dropped, reconnected). The repair list too, because a parent
    removed from one relationship and reconnected by a direct edge is
    NOT missing from the data, and the report has to be able to tell
    the reader which of the two happened to each one."""
    try:
        from synthkit.blueprint import resolve
        from synthkit.generate import _order
        o = _order(resolve(bp))
        return o[2], o[3]
    except Exception:
        return [], []


def verdicts(fid):
    """One line per column that failed something, naming what.

    THE COUNTS DO NOT TELL A READER WHICH COLUMNS TO DISTRUST. A run
    reports coverage 42/42, centre 29/33, spread 30/33, persistence
    29/33 and clustering 18/18, and every one of those is a different
    subset. Answering "can I use this column" meant opening
    fidelity.json and cross-referencing five lists by hand, which is
    why nobody did it.

    Per column is also the honest granularity for the question people
    actually ask. "Trust these 27, be careful with these 6" is a
    statement this evidence supports; "the data is good" is not."""
    rel = fid.get("relationships") or {}
    inverted = {}
    for r in (rel.get("inverted") or []):
        inverted.setdefault(r["child"], []).append(r["parent"])
        inverted.setdefault(r["parent"], []).append(r["child"])
    loosened = {}
    for r in (rel.get("deterministic_loosened") or []):
        loosened[r["child"]] = r
    out = []
    for c in fid.get("columns") or []:
        name, notes = c["column"], []
        sent = c.get("sentinel_share")
        if sent and sent >= 0.5:
            notes.append(
                "{:.0%} of it is the `__other__` sentinel - too many "
                "distinct values to publish as labels, so this column "
                "is destroyed in the output. Check whether it is "
                "really a category".format(sent))
        cov = c.get("coverage_delta")
        if cov is not None and abs(cov) > 0.05:
            notes.append("present on {:+.0%} of rows against the "
                         "source".format(cov))
        cm = c.get("centre_miss")
        if cm:
            notes.append("average off by {:.2f} of its own spread"
                         .format(cm["by_sd"]))
        sm = c.get("spread_miss")
        if sm:
            beyond = sm.get("share_of_magnitude_outside_bounds")
            notes.append(
                "spread {:.0%} of source{}".format(
                    sm["ratio"],
                    "" if not beyond or beyond <= 0.1 else
                    " ({:.0%} of it sits beyond what k allows to be "
                    "published, so this one is privacy rather than a "
                    "fault)".format(beyond)))
        ls, lg = c.get("lag1_source"), c.get("lag1_generated")
        if ls is not None and lg is not None and abs(lg - ls) > 0.15:
            notes.append("steadiness {:.2f} against {:.2f}"
                         .format(lg, ls))
        if name in inverted:
            notes.append("INVERTED against {} - the generated "
                         "relationship runs the opposite way to the "
                         "source".format(", ".join(
                             sorted(set(inverted[name]))[:3])))
        if name in loosened:
            r = loosened[name]
            notes.append("an identity it is computed from is looser "
                         "here: {:.2f} against {:.2f}".format(
                             r["tightness_generated"],
                             r["tightness_source"]))
        if notes:
            out.append((name, notes))
    return out


def render_verdicts(fid):
    s = fid["summary"]
    bad = verdicts(fid)
    L = ["", "", "WHICH COLUMNS TO BE CAREFUL WITH", "-" * 60]
    total = s.get("columns") or 0
    L.append("{} of {} columns passed every check this run makes. The "
             "rest are".format(total - len(bad), total))
    L.append("named here with what went wrong, because a count cannot "
             "tell you which")
    L.append("column it was about.")
    L.append("")
    if not bad:
        L.append("(none - every column passed)")
        return "\n".join(L)
    for name, notes in sorted(bad):
        L.append("    {}".format(name))
        for n in notes:
            L.append("        {}".format(n))
    L.append("")
    L.append("A column absent from this list passed coverage, centre, "
             "spread, steadiness")
    L.append("and every relationship it takes part in. That is not the "
             "same as being")
    L.append("fit for any particular purpose - see fidelity.json for "
             "the numbers behind")
    L.append("each line.")
    return "\n".join(L)


def _were(n, singular="was", plural="were"):
    """Agreement, so a run with one of something does not read like a
    typo in the one document meant for someone who has never seen the
    data."""
    return singular if n == 1 else plural


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
    dropped, reconnected = _will_drop_full(bp)
    direct = set(frozenset([r["child"], r["parent"]])
                 for r in reconnected)
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
             "{} {} dropped,".format(len(dropped),
                                     _were(len(dropped))))
    L.append("in {} kinds - and the counts below add up to that "
             "total.".format(
                 sum(1 for g in (mirrors, trimmed, orphans, other)
                     if g)))
    L.append("")
    if mirrors:
        L.append("{} {} of structure already taken - the "
                 "same dependence,".format(
                     len(mirrors),
                     "RESTATEMENT" if len(mirrors) == 1
                     else "RESTATEMENTS"))
        L.append("generated the other way round. Nothing is missing.")
        L.append("")
    if trimmed:
        # Reported because it IS a partial loss - and the first
        # version of this section omitted the category entirely, so
        # ten of twelve drops on the real extract were invisible.
        #
        # BUT NOT EVERY REMOVED PARENT IS A LOST ONE, and saying so
        # was the next mistake. This section read "the ones listed
        # here contribute nothing to it in the generated data" over a
        # list that included `age_at_visit lost year_of_birth`, while
        # fidelity.json recorded `parents_lost: []` for that same drop
        # because the pair is generated the other way round. Two
        # shipped files disagreeing about the same relationship. The
        # split below is per PARENT rather than per relationship,
        # since one drop can do both.
        moved, gone = [], []
        for d in sorted(trimmed, key=lambda x: -x.get("skill", 0)):
            # `parents_lost` is computed after pair repair has run. An
            # older report without the field cannot claim anything
            # survived, so it falls back to naming them all.
            lost = d.get("parents_lost")
            if lost is None:
                lost = list(d["parents"])
            kept = ", ".join(d.get("kept_parents") or []) or "nothing"
            for p in d["parents"]:
                if p in lost:
                    gone.append("    {} lost {}   (kept {})".format(
                        d["child"], p, kept))
                else:
                    moved.append("    {} <- {}   ({})".format(
                        d["child"], p,
                        "kept as a direct edge"
                        if frozenset([d["child"], p]) in direct
                        else "generated the other way round"))
        L.append("{} had SOME PARENTS REMOVED to break a loop, {} "
                 "parent(s) between them.".format(
                     len(trimmed), len(moved) + len(gone)))
        L.append("Removing one is not the same as losing it, so both "
                 "kinds are named")
        L.append("below and {} + {} adds back to {}.".format(
            len(moved), len(gone), len(moved) + len(gone)))
        L.append("")
        if moved:
            L.append("{} {} STILL IN THE DATA, just not inside this "
                     "relationship - the".format(
                         len(moved), _were(len(moved), "IS", "ARE")))
            L.append("pair reaches the output another way:")
            L.append("")
            L.extend(moved)
            L.append("")
        if gone:
            L.append("{} {} GONE. The column is still explained by "
                     "the parents that remain,".format(
                         len(gone), _were(len(gone), "IS", "ARE")))
            L.append("but these contribute nothing to it in the "
                     "generated data:")
            L.append("")
            L.extend(gone)
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


def _cramers_v(a, b):
    """Association between two CATEGORICAL columns, 0 to 1.

    Spearman cannot see this at all - it coerces both sides to numeric,
    gets NaN, and the pair is skipped in silence. Measured on a
    perfectly associated pair, `_pair_fidelity` compared ZERO of them,
    which means every categorical relationship the catalogue reports
    has been going unchecked: gender, race, ethnicity, visit_type,
    admitted_from and the four list-shaped columns on the real
    extract.

    V is biased upward when a column has many levels, and that is
    tolerable here because the SAME bias lands on the source and the
    generated side and the number that matters is the difference."""
    import numpy as np
    import pandas as pd
    ct = pd.crosstab(a, b)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return None
    obs = ct.to_numpy(dtype=float)
    n = obs.sum()
    if n < 30:
        return None
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(exp > 0, (obs - exp) ** 2 / exp, 0.0))
    denom = n * (min(obs.shape) - 1)
    if denom <= 0:
        return None
    return float(np.sqrt(max(chi2 / denom, 0.0)))


def _eta(num, cat):
    """Correlation ratio: how much of a NUMERIC column's spread is
    explained by which level of a categorical it sits in. 0 to 1, and
    unsigned - a category has no direction to invert."""
    import numpy as np
    import pandas as pd
    d = pd.DataFrame({"y": pd.to_numeric(num, errors="coerce"),
                      "g": cat.astype(str)}).dropna()
    if len(d) < 30 or d["g"].nunique() < 2:
        return None
    y = d["y"].to_numpy(dtype=float)
    grand = y.mean()
    ss_tot = float(((y - grand) ** 2).sum())
    if ss_tot <= 0:
        return None
    ss_b = 0.0
    for _lv, grp in d.groupby("g", observed=False)["y"]:
        ss_b += len(grp) * (float(grp.mean()) - grand) ** 2
    return float(np.sqrt(max(min(ss_b / ss_tot, 1.0), 0.0)))


def _tightness(fr, child, parents):
    """How much of the child is determined by ALL its parents at once.

    1 minus the share of the child's spread left over after a least
    squares fit on the parent set. Near 1 means the column is computed
    from them.

    MEASURED AGAINST THE WHOLE PARENT SET, not one parent at a time.
    The first version binned a single parent, reported 3/3 identities
    intact, and was wrong: `age_at_visit` is the visit year MINUS the
    year of birth, so neither parent determines it alone and the
    source never looked tight to begin with. The identity held on
    21.4% of generated rows while that check called it kept.

    Least squares because these are arithmetic - the catalogue calls
    them "a subtraction, a threshold, a restatement" - and a
    subtraction is exactly linear. A threshold will read looser than
    it is, which errs toward reporting a problem rather than hiding
    one."""
    import numpy as np
    import pandas as pd
    cols = [child] + [p for p in parents if p != child]
    if any(c not in fr.columns for c in cols):
        return None
    d = fr[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 50:
        return None
    y = d[child].to_numpy(dtype=float)
    sd = float(y.std())
    if sd <= 0:
        return None
    X = d[cols[1:]].to_numpy(dtype=float)
    X = np.column_stack([X, np.ones(len(X))])
    try:
        beta, _r, _rk, _sv = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    resid = y - X.dot(beta)
    return round(max(0.0, 1.0 - float(resid.std()) / sd), 4)


def _pair_fidelity(Xs, Xg, bp):
    """Did the RELATIONSHIPS survive, not just the columns?

    Everything else here checks one column at a time - its coverage,
    its centre, its steadiness. A table can pass all of it and carry
    no structure between columns at all, which is the classic way a
    synthetic generator looks right and is useless.

    Worse than absent is INVERTED. A conditional effect curve used
    without the parents it was conditioned on generated systolic
    against diastolic at -0.720 where the source had +0.888, and every
    per-column check passed while it did. A number with the wrong sign
    reads as a finding.

    Spearman rather than Pearson: a relationship may be a threshold or
    a saturation rather than a line, and rank correlation follows any
    monotone shape. A genuinely non-monotone pair reads near zero in
    BOTH tables, so comparing them stays fair."""
    import pandas as pd

    seen, rows = set(), []
    for rel in (bp.get("relationships") or []):
        child = rel.get("child")
        for par in (rel.get("parents") or []):
            key = frozenset([child, par])
            if key in seen or child == par:
                continue
            if child not in Xs.columns or par not in Xs.columns:
                continue
            if child not in Xg.columns or par not in Xg.columns:
                continue
            seen.add(key)

            # WHICH MEASURE THE PAIR NEEDS. Spearman only speaks for
            # two numeric columns; used on anything else it coerces to
            # NaN and the pair vanishes without a word. Categorical
            # pairs get Cramer's V and mixed pairs get the correlation
            # ratio, both unsigned - a category has no direction, so
            # they are reported as strength kept or lost rather than
            # as inverted.
            num_c = pd.api.types.is_numeric_dtype(Xs[child])
            num_p = pd.api.types.is_numeric_dtype(Xs[par])
            if num_c and num_p:
                kind = "numeric"

                def assoc(fr):
                    a = pd.to_numeric(fr[child], errors="coerce")
                    b = pd.to_numeric(fr[par], errors="coerce")
                    m = a.notna() & b.notna()
                    if int(m.sum()) < 30:
                        return None
                    return float(a[m].corr(b[m], method="spearman"))
            elif not num_c and not num_p:
                kind = "categorical"

                def assoc(fr):
                    return _cramers_v(fr[child].astype(str),
                                      fr[par].astype(str))
            else:
                kind = "mixed"

                def assoc(fr):
                    if num_c:
                        return _eta(fr[child], fr[par])
                    return _eta(fr[par], fr[child])
            rs, rg = assoc(Xs), assoc(Xg)
            if rs is None or rg is None:
                continue
            ev = rel.get("evidence") or {}
            eff = (ev.get("effect") or {}).get(par) or {}
            # WHY A PAIR CAME OUT INVERTED, on the record. Discovery
            # already knows: it fits each parent a second time on its
            # own, and sets `reverses_when_controlled` when that curve
            # disagrees in SIGN with the conditional one. That is the
            # documented cause - mean arterial pressure is (S + 2D)/3,
            # so holding it fixed diastolic falls as systolic rises,
            # and generating from the conditional curve produced
            # -0.720 where the source had +0.888. Carrying the flag
            # here means an inverted pair names its own mechanism
            # instead of leaving the reader to rediscover it.
            row = {"child": child, "parent": par,
                   "kind": kind,
                   "measure": {"numeric": "spearman",
                               "categorical": "cramers_v",
                               "mixed": "correlation_ratio"}[kind],
                   "source": round(rs, 4),
                   "generated": round(rg, 4),
                   "delta": round(rg - rs, 4),
                   "reverses_when_controlled": bool(
                       eff.get("reverses_when_controlled")),
                   "near_deterministic": bool(
                       ev.get("near_deterministic"))}
            # HOW TIGHT THE RELATIONSHIP IS, not just which way it
            # leans. A near-deterministic pair is an ARITHMETIC
            # IDENTITY - age is the visit year minus the year of
            # birth - and correlation cannot see it break. Measured on
            # a fixture where the identity held on 100% of source
            # rows: it held on 23.7% of generated rows while the
            # correlation stayed strong, both means stayed right, and
            # every check in this file passed. Someone opening the
            # file would find patients whose age contradicts their
            # birth year.
            #
            # Tightness is what survives that: bin the parent, remove
            # the within-bin median, and see how much of the child's
            # spread is left. An identity leaves almost none.
            if row["near_deterministic"]:
                ps = [x for x in (rel.get("parents") or [])]
                row["parents_used"] = ps
                row["tightness_source"] = _tightness(Xs, child, ps)
                row["tightness_generated"] = _tightness(Xg, child, ps)
            rows.append(row)

    # THE NUMERIC COUNTERS KEEP THEIR OLD MEANING. Folding the new
    # pairs into `compared` would silently change what "25/28 keep
    # their direction" refers to and make this run incomparable with
    # the last one. They are counted separately instead.
    numeric_rows = [r for r in rows if r["kind"] == "numeric"]
    other_rows = [r for r in rows if r["kind"] != "numeric"]
    strong_other = [r for r in other_rows if r["source"] >= 0.15]
    kept_other = [r for r in strong_other
                  if r["generated"] >= 0.6 * r["source"]]
    strong = [r for r in numeric_rows if abs(r["source"]) >= 0.1]
    inverted = [r for r in strong
                if r["source"] * r["generated"] < 0
                and abs(r["generated"]) >= 0.1]
    det = [r for r in rows if r.get("near_deterministic")
           and r.get("tightness_source") is not None
           and r.get("tightness_generated") is not None]
    det_kept = [r for r in det
                if r["tightness_generated"] >= r["tightness_source"] - 0.2]
    return {
        "categorical_compared": len(strong_other),
        "categorical_kept": len(kept_other),
        "categorical_weakened": sorted(
            [r for r in strong_other if r not in kept_other],
            key=lambda r: r["source"] - r["generated"], reverse=True),
        "deterministic_compared": len(det),
        "deterministic_kept": len(det_kept),
        "deterministic_loosened": sorted(
            [r for r in det if r not in det_kept],
            key=lambda r: r["tightness_source"] - r["tightness_generated"],
            reverse=True),
        "compared": len(strong),
        "sign_kept": sum(1 for r in strong
                         if r["source"] * r["generated"] > 0),
        "close": sum(1 for r in strong if abs(r["delta"]) <= 0.2),
        "inverted": sorted(inverted,
                           key=lambda r: r["source"] - r["generated"],
                           reverse=True),
        "worst": sorted(rows, key=lambda r: -abs(r["delta"]))[:10],
        "pairs": rows,
        "note": "Spearman on pairs the blueprint related, counting "
                "only those the SOURCE relates at 0.1 or more. "
                "`inverted` is the serious column: a generated "
                "relationship with the opposite sign reads as a "
                "finding and is worse than one that is missing.",
    }


def compare(df, g, bp, group_by, time_col, ordinals=None):
    """Source against generated, column by column."""
    import numpy as np
    import pandas as pd
    from synthkit.discover import prepare
    from synthkit.dynamics import measure

    # THE SAME TYPING ON BOTH SIDES AND IN THE BLUEPRINT. A
    # declared ordinal is numeric in the contract, so comparing
    # it as a category here would measure a different column
    # from the one that was generated.
    Xs, _, _, _ = prepare(df, group_by, ordinals=ordinals)
    Xg, _, _, _ = prepare(g, group_by, ordinals=ordinals)
    gt = "visit_number" if "visit_number" in g.columns else None
    ds = measure(df, Xs, group_by, time_col)
    dg = measure(g, Xg, group_by, gt)

    cols, n_ok = [], dict(cov=0, ctr=0, spr=0, lag=0, clu=0, num=0, dyn=0,
                          part=0)
    pairs = _pair_fidelity(Xs, Xg, bp)
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
        _mg = (((bp.get("columns") or {}).get(c) or {})
               .get("marginal") or {})
        if _mg.get("sentinel_share"):
            row["sentinel_share"] = _mg["sentinel_share"]
        if pd.api.types.is_numeric_dtype(s) and \
                pd.api.types.is_numeric_dtype(q):
            n_ok["num"] += 1
            sd = float(s.std() or 0.0)
            row.update({
                "mean_source": round(float(s.mean()), 4),
                "mean_generated": round(float(q.mean()), 4),
                "sd_source": round(sd, 4),
                "sd_generated": round(float(q.std() or 0.0), 4)})
            # SPREAD WAS RECORDED AND NEVER ASSERTED. Coverage,
            # centre, persistence and clustering all had a bar; the
            # standard deviation sat in the file with nothing checking
            # it. On the 800-patient run `6690_2` came out at 27% of
            # its source spread - three quarters of the column's
            # variance gone - and passed every check, because its
            # CENTRE was fine at 7.05 against 7.05.
            #
            # A shortfall is not automatically a fault. Reproduced
            # here: when five patients hold values above the published
            # bound and 54% of the column's magnitude sits up there,
            # the k rule removes it deliberately and the spread drops
            # to a quarter. That is the protection working. So the
            # miss is reported WITH the share of the source that sits
            # outside the published bound, which is what separates
            # "privacy did this" from "the sampler did this".
            sd_g = float(q.std() or 0.0)
            if sd > 0 and abs(sd_g / sd - 1.0) <= 0.25:
                n_ok["spr"] += 1
            elif sd > 0:
                mg = (((bp.get("columns") or {}).get(c) or {})
                      .get("marginal") or {})
                vv = mg.get("v") or []
                out_of_bounds = None
                if vv:
                    # SHARE OF VARIANCE, NOT OF MAGNITUDE.
                    #
                    # Squared raw values are dominated by the mean on
                    # any column that does not sit near zero, so a
                    # clipped tail looks like nothing. Measured on an
                    # spo2-shaped column centred at 98.6 whose low
                    # tail the k rule removes: this reported 0.0%
                    # where the honest answer is 26.5%, and a real
                    # run then read `spo2 48% of source (0% beyond
                    # bound)` and sent me looking for a sampler bug
                    # that was the privacy rule all along.
                    #
                    # Spread is a statement about deviation from the
                    # centre, so the attribution has to be too.
                    lo_b, hi_b = float(vv[0]), float(vv[-1])
                    sv = s.astype(float)
                    total = float(((sv - sv.mean()) ** 2).sum())
                    beyond = sv[(sv > hi_b) | (sv < lo_b)]
                    out_of_bounds = (
                        round(float(((beyond - sv.mean()) ** 2).sum())
                              / total, 4) if total > 0 else None)
                row["spread_miss"] = {
                    "ratio": round(sd_g / sd, 4),
                    "direction": ("generated wider" if sd_g > sd
                                  else "generated narrower"),
                    "share_of_magnitude_outside_bounds": out_of_bounds,
                    "tail_shape_published": bool(
                        "tail_mean_high" in mg or "tail_mean_low" in mg),
                }
            if sd > 0 and abs(row["mean_generated"]
                              - row["mean_source"]) <= 0.1 * sd:
                n_ok["ctr"] += 1
            elif sd > 0:
                # WHY THIS ONE MISSED, not just that it did. The known
                # mechanism is the quantile grid's extreme segment
                # being drawn as a straight line to the safe bound,
                # which overshoots on a skewed column and is corrected
                # by the published tail mean. So a miss is only
                # EXPLAINED by that mechanism if the column is skewed
                # AND the correction was unavailable. Sixteen columns
                # missed this bar on the 800-patient run with nothing
                # to say why, and a count cannot be diagnosed twice.
                mg = (((bp.get("columns") or {}).get(c) or {})
                      .get("marginal") or {})
                row["centre_miss"] = {
                    "by_sd": round(abs(row["mean_generated"]
                                       - row["mean_source"]) / sd, 4),
                    "direction": ("generated above source"
                                  if row["mean_generated"]
                                  > row["mean_source"]
                                  else "generated below source"),
                    "skew_source": round(float(s.skew()), 4)
                    if len(s) > 2 else None,
                    "tail_shape_published": bool(
                        "tail_mean_high" in mg
                        or "tail_mean_low" in mg),
                    "integral": bool(mg.get("integral")),
                }
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

    # DID THE ORDERINGS SURVIVE? A constraint is a statement about
    # each ROW, and every other number in this file is a statement
    # about a distribution, so nothing here could see one break.
    cons = []
    for con in (bp.get("constraints") or []):
        lhs, rhs = con["lhs"], con["rhs"]
        if lhs not in Xg.columns or rhs not in Xg.columns:
            continue
        a_ = pd.to_numeric(Xg[lhs], errors="coerce")
        b_ = pd.to_numeric(Xg[rhs], errors="coerce")
        m_ = a_.notna() & b_.notna()
        if int(m_.sum()) < 30:
            continue
        held = float((a_[m_] <= b_[m_]).mean())
        cons.append({"lhs": lhs, "op": "<=", "rhs": rhs,
                     "holds_in_source": con["holds_in_source"],
                     "holds_in_generated": round(held, 6),
                     "rows_violating": int((~(a_[m_] <= b_[m_])).sum())})
    return {
        "columns": cols,
        "constraints": cons,
        "relationships": pairs,
        "summary": {
            "columns": len(cols), "numeric": n_ok["num"],
            "numeric_dynamic": n_ok["dyn"],
            "partly_covered": n_ok["part"],
            "constraints_checked": len(cons),
            "constraints_held": sum(1 for c in cons
                                    if c["holds_in_generated"] >= 0.999),
            "coverage_ok": n_ok["cov"], "centre_ok": n_ok["ctr"],
            "spread_ok": n_ok["spr"],
            "lag1_ok": n_ok["lag"], "cluster_ok": n_ok["clu"],
            "pairs": pairs["compared"],
            "pairs_sign_ok": pairs["sign_kept"],
            "pairs_close": pairs["close"],
            "pairs_inverted": len(pairs["inverted"]),
            "categorical_compared": pairs["categorical_compared"],
            "categorical_kept": pairs["categorical_kept"],
            "deterministic_compared": pairs["deterministic_compared"],
            "deterministic_kept": pairs["deterministic_kept"],
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
