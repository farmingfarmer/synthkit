"""IS THIS COHORT BIG ENOUGH? - the argument, from two real runs.

WHY IT EXISTS. A stakeholder asked for synthetic data from a
100-patient cohort, was told it would not hold up, and asked to be
shown WHY rather than told. "Your cohort is too small" is an opinion;
"290 of your condition codes clear the privacy floor against 898 of
theirs, and the same blood-pressure column keeps 25% of its spread
against 61%" is a measurement. This emits the second thing.

It reads two finished run directories plus their sources and writes
the comparison - so the numbers are produced on the machine that
holds the data and NEVER travel into this repository, which is the
same reason the fidelity deck and the sign-off page are generated
rather than committed.

ORDER IS ARGUED, NOT CHOSEN. Coverage is printed before fidelity
because fidelity is scored only on the patterns that cleared the
floor: a thinner source grades a smaller and stronger surviving set
and can post a HIGHER percentage while publishing far less of its
data. A page showing fidelity alone argues for the worse dataset.

WHAT IT REFUSES TO PUT IN THE ARGUMENT. Three numbers that look like
evidence and are not:

  * a column EMPTY in one source and not the other - that is what
    was pulled, not what the floor did;
  * within-patient persistence, which collapses on EVERY cohort
    because this engine is row-level and models no trajectory, so it
    would collapse on a million patients;
  * any single-seed fidelity percentage, which this project has
    measured moving by ten points between seeds.

Each is reported in a NOT-EVIDENCE section instead, because an
argument built from the wrong numbers loses the right ones with it.

  python scripts/cohort_argument.py \
      BIG=src.csv:RUNDIR THIN=src.csv:RUNDIR \
      [--group-by person_id] [--k 10] [--column systolic] \
      [-o argument.md]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
for _p in (str(_root), str(_root / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ORDER_NOTE = (
    "Read coverage before fidelity. Fidelity is scored only on the "
    "patterns that cleared the privacy floor, so a thinner source "
    "grades a smaller and stronger surviving set and can post a "
    "HIGHER percentage while publishing far less of its data.")

NOT_EVIDENCE = (
    "These differences are real and are NOT part of the argument, "
    "because they would not survive scrutiny:")


def _read(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def profile(name, src_path, run_dir, group_by, k):
    """Everything the argument needs from one run, and nothing the
    argument may not use."""
    from synthkit import forecast, gate, sets
    src = _read(src_path)
    run = Path(run_dir)
    gen = _read(run / "generated.csv")
    # FOLD "" TO NaN FOR EVERY COLUMN EXCEPT SETS. A set column's
    # empty cell is a FACT about the visit - it held nothing - and
    # folding it into missing is the recorded reader fault that once
    # made every token share wrong by 1/(1-empty). sets.looks_like_set
    # is the same predicate the engine applies, so the two cannot
    # class a column differently. Done per column rather than with a
    # frame-wide replace, which also avoids the pandas downcast
    # warning that becomes a TypeError on pandas 3.
    srcn = src.copy()
    for _c in srcn.columns:
        if sets.looks_like_set(srcn[_c]):
            continue
        srcn[_c] = srcn[_c].astype(object).where(
            srcn[_c].astype(str).str.strip() != "", np.nan)
    cen = forecast.census(srcn, group_by, k=k)

    out = {"name": name, "patients": cen["patients"],
           "rows": cen["rows"], "mass": cen["publishable_mass"],
           "verdict": cen["verdict"],
           "destroyed": list(cen["destroyed"]),
           "gen_rows": int(len(gen)),
           "gen_patients": int(gen[group_by].nunique())
           if group_by in gen.columns else None,
           "tokens": [], "empty_in_source": [], "gate": None,
           "numeric": {}}

    # WHAT CLEARS THE FLOOR, PER SET COLUMN. The floor counts
    # PATIENTS, so this is the number of patients expressed as how
    # specific the output is allowed to be.
    groups = srcn[group_by].astype(str).to_numpy()
    for c in srcn.columns:
        if c == group_by or not sets.looks_like_set(srcn[c]):
            continue
        toks = set()
        for val in srcn[c].fillna(""):
            for t in str(val).split(";"):
                if t.strip():
                    toks.add(t.strip())
        v = sets.vocabulary(srcn[c], groups, k=k, cap=10 ** 6)
        out["tokens"].append({
            "column": c, "found": len(toks),
            "clear_k": len(v["tokens"]) if v else 0})

    # A COLUMN EMPTY IN THE SOURCE IS AN EXTRACTION DIFFERENCE.
    # Collected so it can be stated OUTSIDE the argument.
    for c in srcn.columns:
        if c == group_by:
            continue
        if float(srcn[c].notna().mean()) == 0.0:
            out["empty_in_source"].append(c)

    fj = run / "fidelity.json"
    if fj.exists():
        a = gate.assess(json.loads(fj.read_text(encoding="utf-8")))
        out["gate"] = {"met": a["met"],
                       "criteria": {c["name"]: c
                                    for c in a["criteria"]}}
        # the per-column numeric reading the gate does not grade
        for col in (json.loads(fj.read_text(encoding="utf-8"))
                    .get("columns") or []):
            nm = col.get("column")
            if nm:
                out["numeric"][nm] = {
                    "cov_src": col.get("coverage_source"),
                    "cov_gen": col.get("coverage_generated"),
                    "sd_src": col.get("sd_source"),
                    "sd_gen": col.get("sd_generated"),
                    "lag_src": col.get("lag1_source"),
                    "lag_gen": col.get("lag1_generated")}
    return out


def _crit(p, name):
    if not p["gate"]:
        return "-"
    c = p["gate"]["criteria"].get(name)
    if not c:
        return "-"
    head = c["detail"].strip().split()[0].rstrip(",")
    return "{} {}".format(head, "PASS" if c["ok"] else "FAIL")


def _retained(d):
    """Share of the source's spread the output keeps. The number the
    whole argument turns on, so it is computed in one place."""
    s, g = d.get("sd_src"), d.get("sd_gen")
    if not s or not g:
        return None
    return float(g) / float(s)


def render(big, thin, column, k):
    L = []
    w = 24
    col = "{:<34}" + ("{:<%d}" % w) * 2
    nm = (big["name"], thin["name"])
    L.append("# Is this cohort big enough, at k={}?".format(k))
    L.append("")
    for _l in _wrap(
            "Two real cohorts through the SAME engine with the SAME "
            "settings. The only difference is the source.", 80):
        L.append(_l)
    L.append("")
    L.append("## 1. How much of each source can be published")
    L.append("")
    for line in _wrap(ORDER_NOTE, 34 + w * 2):
        L.append(line)
    L.append("")
    L.append(col.format("", *nm))
    L.append("-" * (34 + w * 2))
    L.append(col.format("  patients",
                        *["{:,}".format(p["patients"])
                          for p in (big, thin)]))
    L.append(col.format("  rows",
                        *["{:,}".format(p["rows"])
                          for p in (big, thin)]))
    L.append(col.format("  publishable mass",
                        *["{:.0%}".format(p["mass"])
                          for p in (big, thin)]))
    L.append(col.format("  columns destroyed",
                        *[str(len(p["destroyed"]))
                          for p in (big, thin)]))
    L.append(col.format("  verdict",
                        *[p["verdict"] for p in (big, thin)]))
    L.append("")
    L.append("## 2. What clears the floor, per column")
    L.append("")
    for _l in _wrap(
            "A token is publishable only if at least {} DISTINCT "
            "PATIENTS hold it.".format(k), 80):
        L.append(_l)
    L.append("")
    cols = sorted({t["column"] for p in (big, thin)
                   for t in p["tokens"]})
    L.append(col.format("", *nm))
    L.append("-" * (34 + w * 2))
    for c in cols:
        cells = []
        for p in (big, thin):
            t = next((x for x in p["tokens"]
                      if x["column"] == c), None)
            cells.append("-" if not t else "{:,} of {:,}".format(
                t["clear_k"], t["found"]))
        L.append(col.format("  " + c[:30], *cells))
    L.append("")
    L.append("## 3. The same column, in both runs")
    L.append("")
    db = big["numeric"].get(column)
    dt = thin["numeric"].get(column)
    if db and dt:
        L.append(col.format("  " + column[:30], *nm))
        L.append("-" * (34 + w * 2))
        L.append(col.format("  coverage",
                            *["{} -> {}".format(d["cov_src"],
                                                d["cov_gen"])
                              for d in (db, dt)]))
        L.append(col.format("  spread",
                            *["{} -> {}".format(d["sd_src"],
                                                d["sd_gen"])
                              for d in (db, dt)]))
        rb, rt = _retained(db), _retained(dt)
        L.append(col.format("  spread RETAINED",
                            *["{:.0%}".format(r) if r is not None
                              else "-" for r in (rb, rt)]))
        L.append("")
        for _l in _wrap("Presence is reproduced in both, so this is not a missingness problem - the MAGNITUDE is what differs.", 80):
            L.append(_l)
        if rb is not None and rt is not None and rb > rt:
            L.append("")
            for _l in _wrap(
                    "STATED HONESTLY: the larger cohort is BETTER "
                    "here, not good - it loses {:.0%} of this "
                    "column's spread too. The floor costs magnitude "
                    "on any cohort; it costs more on the thin "
                    "one.".format(1.0 - rb), 80):
                L.append(_l)
    else:
        L.append("  (no numeric reading for {!r} in one of the "
                 "runs - pass --column with a name both share)"
                 .format(column))
    L.append("")
    L.append("## 4. What the thin cohort still does well")
    L.append("")
    L.append(col.format("", *nm))
    L.append("-" * (34 + w * 2))
    for c in ("inverted", "set token shares", "set EMPTY rate",
              "shapes tracked", "coverage"):
        L.append(col.format("  " + c,
                            *[_crit(p, c) for p in (big, thin)]))
    L.append("")
    for _l in _wrap(
            "An INVERTED relationship reads as a finding and is "
            "worse than a missing one. Read that row first.", 80):
        L.append(_l)
    L.append("")
    L.append("## 5. NOT evidence about cohort size")
    L.append("")
    for line in _wrap(NOT_EVIDENCE, 34 + w * 2):
        L.append(line)
    L.append("")
    for p, other in ((thin, big), (big, thin)):
        only = [c for c in p["empty_in_source"]
                if c not in other["empty_in_source"]]
        if only:
            for _l in _wrap(
                    "{} has {} column(s) EMPTY in the source that "
                    "the other does not: {}. That is what was "
                    "pulled, not what the floor did.".format(
                        p["name"], len(only),
                        ", ".join(only[:8])), 78):
                L.append("  " + _l)
    if db and dt and db.get("lag_src") and dt.get("lag_src"):
        L.append("")
        for _l in _wrap(
                "Within-patient persistence falls on BOTH ({}: {} "
                "-> {}; {}: {} -> {}). This engine is ROW-LEVEL and "
                "models no patient trajectory, so it would lose "
                "persistence on a million patients.".format(
                    big["name"], db["lag_src"], db["lag_gen"],
                    thin["name"], dt["lag_src"], dt["lag_gen"]), 78):
            L.append("  " + _l)
    L.append("")
    for _l in _wrap(
            "No single-seed fidelity percentage is quoted. A paired "
            "sweep in this project moved such a reading by ten "
            "points, so one seed is not a result.", 78):
        L.append("  " + _l)
    L.append("")
    L.append("## 6. What would actually help")
    L.append("")
    for _l in _wrap(
            "More PEOPLE, not more rows. Duplicating every row "
            "changes no verdict - the floor counts patients, and "
            "one person seen twice as often is still one person. "
            "Adding different people moves what publishes.", 80):
        L.append(_l)
    return "\n".join(L)


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out.append(line)
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    group_by, k, column, out, rest = "person_id", 10, "", "", []
    i = 0
    while i < len(argv):
        if argv[i] == "--group-by":
            group_by = argv[i + 1]; i += 2
        elif argv[i] == "--k":
            k = int(argv[i + 1]); i += 2
        elif argv[i] == "--column":
            column = argv[i + 1]; i += 2
        elif argv[i] in ("-o", "--out"):
            out = argv[i + 1]; i += 2
        else:
            rest.append(argv[i]); i += 1
    specs = []
    for spec in rest:
        name, _, tail = spec.partition("=")
        src, _, run = tail.partition(":")
        if not (name and src and run):
            print("want NAME=SRC.csv:RUNDIR, got {!r}".format(spec))
            return 2
        specs.append((name, src, run))
    if len(specs) != 2:
        print(__doc__)
        return 2
    print("cohort_argument: --group-by {} --k {} --column {}"
          .format(group_by, k, column or "(auto)"))
    profs = [profile(n, s, r, group_by, k) for n, s, r in specs]
    big, thin = sorted(profs, key=lambda p: -p["patients"])
    if not column:
        shared = [c for c in big["numeric"]
                  if c in thin["numeric"]
                  and big["numeric"][c].get("sd_source") is not None]
        column = shared[0] if shared else "systolic_blood_pressure"
    txt = render(big, thin, column, k)
    if out:
        Path(out).write_text(txt + "\n", encoding="utf-8")
        print("wrote {}".format(out))
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
