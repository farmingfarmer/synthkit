"""A data dictionary for the file we actually send.

GENERATED FROM THE ARTIFACTS, NEVER WRITTEN BY HAND. A dictionary
typed out beside a dataset drifts from it the first time anything
changes, and the reader has no way to tell. Every row here is read
from `blueprint.json` for the contract and from the generated CSV
for what the file actually contains - so a column that moved, a
coverage that changed or a vocabulary that thinned shows up
without anybody remembering to update prose.

THE PRIVACY SENTENCES ARE QUOTED, NOT PARAPHRASED. The blueprint
already states what the k rule does in its own words; re-wording a
privacy claim in a second place is how two halves come to say
different things about the same guarantee.

AND THE LIMITS SECTION IS THE POINT. A recipient who is handed
clean-looking columns will assume the fields they need are there.
What this dataset does NOT carry - a principal diagnosis, a
present-on-admission flag, a discharge disposition - decides
whether a DRG can be assigned at all, and saying so here is
cheaper than having it discovered against a vendor's output.

    python scripts/data_dictionary.py RUNDIR
"""
import argparse
import csv
import html
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd                                     # noqa: E402
from synthkit.sets import is_scaffolding                # noqa: E402

# STRUCTURAL ABSENCES A REVIEWER MUST BE TOLD ABOUT.
#
# These are HUMAN-WRITTEN and marked as such, because the absence
# of a field is not something code can prove to be meaningful -
# a script can say "no column is named this", which is a
# different and weaker statement. Each one is here because it
# changes what a recipient can do with the file.
STRUCTURAL_NOTES = [
    ("No principal diagnosis",
     "Diagnoses arrive as an UNORDERED set. No field designates "
     "the condition chiefly responsible for the admission, so "
     "MS-DRG assignment cannot be performed from this file."),
    ("No present-on-admission flag",
     "Nothing distinguishes a condition present on admission from "
     "one arising during the stay, so CC/MCC exclusion logic and "
     "hospital-acquired-condition rules cannot be applied."),
    ("No discharge disposition",
     "Where the patient went is not carried, so transfers cannot "
     "be separated from discharges and readmission measures "
     "cannot exclude planned transfers."),
    ("Codes are not described",
     "Condition codes are carried as codes only. No ICD-10-CM "
     "description dictionary travels with this file."),
    ("Statistical, not clinical, coherence",
     "Co-occurrence between codes is reproduced as measured. "
     "Clinical narrative coherence is NOT modelled, so an "
     "individual synthetic encounter may carry a combination that "
     "is statistically plausible and clinically unusual."),
]


def kind_of(spec):
    m = spec.get("marginal") or {}
    t = m.get("type")
    if t == "list":
        return "set (semicolon-separated)"
    if t == "quantiles":
        return "numeric"
    if t == "levels":
        return "categorical"
    return t or spec.get("kind") or "unknown"


def rows_for(bp, gen):
    cols = bp.get("columns") or {}
    ids = {}
    for it in (bp.get("arithmetic_identities") or []):
        if it.get("child"):
            ids[it["child"]] = "computed from " + ", ".join(
                it.get("parents") or [])
    out = []
    for name in gen.columns:
        spec = cols.get(name) or {}
        m = spec.get("marginal") or {}
        present = gen[name].astype(str).str.strip()
        meas = float((present != "").mean())
        row = {
            "column": name,
            "type": kind_of(spec) if spec else "identifier / index",
            "level": spec.get("level") or "",
            "coverage_published": (
                "" if spec.get("coverage") is None
                else "{:.4f}".format(float(spec["coverage"]))),
            "coverage_in_this_file": "{:.4f}".format(meas),
            "detail": "",
            "notes": ids.get(name, ""),
        }
        if m.get("type") == "quantiles":
            g = m.get("grid") or []
            if g:
                row["detail"] = ("published range {} to {} "
                                 "(k-anonymous aggregates, not "
                                 "observed extremes)".format(
                                     g[0], g[-1]))
        elif m.get("type") == "levels":
            row["detail"] = "{} published level(s)".format(
                len(m.get("levels") or []))
        elif m.get("type") == "list":
            pub = len(m.get("values") or m.get("tokens") or [])
            seen = set()
            sep = m.get("separator") or ";"
            for v in present:
                if v:
                    seen.update(x for x in v.split(sep) if x)
            row["detail"] = ("{} token(s) published, {} distinct "
                             "in this file".format(pub, len(seen)))
        if spec.get("derived_from"):
            row["notes"] = ("derived from "
                            + str(spec["derived_from"]))
        out.append(row)
    return out


def count_vs_set(bp, gen):
    """Where a count column names a set's length, how often they
    actually agree in the file a recipient receives.

    Derived rather than asserted: the source's own count was taken
    before de-duplication on the real extract, so the two are
    equal on well under half of rows there. A tool reading both
    fields will see that disagreement and it should not discover
    it on its own.
    """
    out = []
    cols = bp.get("columns") or {}
    for name, spec in cols.items():
        m = spec.get("marginal") or {}
        if m.get("type") != "list" or name not in gen.columns:
            continue
        sep = m.get("separator") or ";"
        length = gen[name].astype(str).apply(
            lambda v: 0 if not v.strip() else len(
                [x for x in v.split(sep) if x]))
        for cand in gen.columns:
            if cand == name or not cand.endswith(("_count", "__n")):
                continue
            cv = pd.to_numeric(gen[cand].replace("", None),
                               errors="coerce")
            both = cv.notna()
            if int(both.sum()) < 50:
                continue
            agree = float((cv[both] == length[both]).mean())
            if agree > 0.2:
                out.append((cand, name, agree, int(both.sum())))
    return out


def build(rundir):
    rundir = Path(rundir)
    bp = json.loads((rundir / "blueprint.json").read_text(
        encoding="utf-8"))
    gen = pd.read_csv(rundir / "generated.csv", dtype=str,
                      keep_default_na=False)
    rows = rows_for(bp, gen)
    priv = bp.get("privacy") or {}
    dropped = sorted(c for c in (bp.get("columns") or {})
                     if c not in gen.columns and is_scaffolding(c))
    return bp, gen, rows, priv, dropped, count_vs_set(bp, gen)


def render(bp, gen, rows, priv, dropped, cvs, src_name):
    e = html.escape
    p = []
    p.append("<h1>Synthetic dataset - data dictionary</h1>")
    p.append("<p class=sub>{} column(s), {} row(s). Generated from "
             "the run's own blueprint and the file itself, so it "
             "cannot drift from what is being sent.</p>"
             .format(len(gen.columns), len(gen)))
    if src_name:
        p.append("<p class=sub>Modelled on: {}</p>".format(e(src_name)))
    p.append("<h2>Fields</h2><table><tr><th>Column</th><th>Type</th>"
             "<th>Level</th><th>Coverage<br>(published)</th>"
             "<th>Coverage<br>(this file)</th><th>Detail</th>"
             "<th>Notes</th></tr>")
    for r in rows:
        p.append("<tr>" + "".join(
            "<td>{}</td>".format(e(str(r[k]))) for k in
            ("column", "type", "level", "coverage_published",
             "coverage_in_this_file", "detail", "notes")) + "</tr>")
    p.append("</table>")

    p.append("<h2>What this dataset does NOT carry</h2>")
    p.append("<p class=sub>Reviewed statements, not derived ones - "
             "the absence of a field is not something a script can "
             "prove to be meaningful.</p><dl>")
    for head, body in STRUCTURAL_NOTES:
        p.append("<dt>{}</dt><dd>{}</dd>".format(e(head), e(body)))
    p.append("</dl>")

    p.append("<h2>Measured limits</h2><ul>")
    for r in rows:
        if "published," in r["detail"] and "distinct" in r["detail"]:
            p.append("<li><b>{}</b>: {}</li>".format(
                e(r["column"]), e(r["detail"])))
    for cand, setc, agree, n in cvs:
        p.append("<li><b>{}</b> names the length of <b>{}</b> and "
                 "agrees with it on {:.1%} of the {} rows that "
                 "carry both. A tool reading both fields will see "
                 "that disagreement.</li>".format(
                     e(cand), e(setc), agree, n))
    if dropped:
        p.append("<li>{} search-scaffolding column(s) exist in the "
                 "contract and are deliberately NOT in this file; "
                 "nothing reported about them is about data you "
                 "received.</li>".format(len(dropped)))
    p.append("</ul>")

    p.append("<h2>Privacy posture</h2>")
    p.append("<p class=sub>Quoted from the blueprint, not "
             "paraphrased.</p><dl>")
    for key in ("k", "counted_in", "numeric_bounds", "rare_levels",
                "visits_per_patient", "NOT_a_formal_guarantee"):
        if priv.get(key) is not None:
            p.append("<dt>{}</dt><dd>{}</dd>".format(
                e(key), e(str(priv[key]))))
    p.append("</dl>")
    css = ("body{font:15px/1.55 -apple-system,Segoe UI,sans-serif;"
           "max-width:1100px;margin:2rem auto;padding:0 1rem;"
           "color:#14161a}h1{font-size:1.6rem}h2{margin-top:2rem;"
           "border-bottom:1px solid #d8dbe0;padding-bottom:.3rem}"
           ".sub{color:#5b6270}table{border-collapse:collapse;"
           "width:100%;font-size:13px}th,td{border:1px solid #d8dbe0;"
           "padding:.35rem .5rem;text-align:left;vertical-align:top}"
           "th{background:#f3f4f6}dt{font-weight:600;margin-top:.6rem}"
           "dd{margin:.2rem 0 0 1.2rem;color:#333}"
           "@media(prefers-color-scheme:dark){body{background:#14161a;"
           "color:#e8eaed}th{background:#22252b}th,td{border-color:#333}"
           ".sub{color:#9aa1ad}dd{color:#c9ced6}}")
    return ("<!doctype html><meta charset=utf-8><title>Data "
            "dictionary</title><style>%s</style>%s"
            % (css, "".join(p)))


def main():
    ap = argparse.ArgumentParser(
        description="a data dictionary generated from a run")
    ap.add_argument("rundir")
    a = ap.parse_args()
    rd = Path(a.rundir)
    if not (rd / "blueprint.json").exists() or \
            not (rd / "generated.csv").exists():
        print("STOPPED: {} needs blueprint.json and generated.csv"
              .format(rd))
        return 2
    src = ""
    pv = rd / "provenance.json"
    if pv.exists():
        src = ((json.loads(pv.read_text(encoding="utf-8"))
                .get("source") or {}).get("name")) or ""
    bp, gen, rows, priv, dropped, cvs = build(rd)
    (rd / "data_dictionary.html").write_text(
        render(bp, gen, rows, priv, dropped, cvs, src),
        encoding="utf-8")
    with io.open(str(rd / "data_dictionary.csv"), "w",
                 encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote data_dictionary.html and data_dictionary.csv "
          "- {} column(s)".format(len(rows)))
    for cand, setc, agree, n in cvs:
        print("  NOTE {} agrees with len({}) on {:.1%} of {} rows"
              .format(cand, setc, agree, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
