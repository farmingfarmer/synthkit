"""Smoke: several tables, keys from the values, integrity kept.

WHY THIS EXISTS. The fitted path has always been single-table,
so an operator with patients, visits and labs had to join them
by hand first - and a hand join is where row counts and
missingness quietly change before this tool ever sees them.
M3's gate is three multi-table fixtures, keys auto-detected,
referential integrity preserved.

THE NEGATIVE CONTROLS ARE THE POINT. A key detector that finds
a key in everything also finds one in nothing, so the suite
plants columns that MUST NOT be keys beside the ones that must:
a continuous measurement (unique on every row, and the first
version of this module made one a primary key and would have
linked tables on it), a two-value flag, and a table unrelated
to the others.

AND THE HEADERS LIE ON PURPOSE. One fixture names its key
`subj` and puts lab values in a column called `patient_id`,
because this project's rule for PHI is that values decide and
headers only assist, and a key detector has no business
following a different rule.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from synthkit.multitable import (                     # noqa: E402
    assign_keys, check_integrity, detect, integrity, render)

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:3d}  {}".format(PASS, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def star(seed=3):
    """One parent, two children, with planted orphans.

    The orphans are deliberate: a real extract has visits whose
    patient was filtered out by somebody's cohort query, and a
    generator that emits 100% integrity where the source had
    97.8% has invented a cleanliness the data never had.
    """
    rng = np.random.default_rng(seed)
    n = 200
    pat = pd.DataFrame({
        "mrn": ["P{:05d}".format(i) for i in range(n)],
        "sex": rng.choice(["F", "M"], n),
        "yob": rng.integers(1940, 2005, n)})
    vn = 900
    vis = pd.DataFrame({
        "visit_id": ["V{:06d}".format(i) for i in range(vn)],
        "mrn": (list(rng.choice(pat["mrn"], vn - 20))
                + ["P99999"] * 20),
        "los": rng.integers(1, 14, vn)})
    rx = pd.DataFrame({
        "rx_id": ["R{:06d}".format(i) for i in range(400)],
        "mrn": rng.choice(pat["mrn"], 400),
        "dose": np.round(rng.normal(10, 3, 400), 2)})
    return {"patients": pat, "visits": vis, "meds": rx}


def chain(seed=5):
    """patients -> visits -> labs, each a child of the last."""
    rng = np.random.default_rng(seed)
    n = 150
    pat = pd.DataFrame({
        "mrn": ["Q{:05d}".format(i) for i in range(n)],
        "site": rng.choice(["a", "b", "c"], n)})
    vis = pd.DataFrame({
        "visit_id": ["W{:06d}".format(i) for i in range(600)],
        "mrn": rng.choice(pat["mrn"], 600)})
    lab = pd.DataFrame({
        "lab_id": ["L{:07d}".format(i) for i in range(2000)],
        "visit_id": rng.choice(vis["visit_id"], 2000),
        # A MEASUREMENT, unique on nearly every row - the
        # column the first version of this module made a key.
        "value": np.round(rng.normal(5, 1, 2000), 4)})
    return {"patients": pat, "visits": vis, "labs": lab}


def lying_headers(seed=7):
    """The key is `subj`; `patient_id` holds lab values.

    Integer keys too, because a patient number is usually an
    integer and a detector that refuses those rejects the most
    common key there is.
    """
    rng = np.random.default_rng(seed)
    n = 180
    pat = pd.DataFrame({
        "subj": list(range(90000, 90000 + n)),
        "patient_id": np.round(rng.normal(5, 1, n), 3),
        "flag": rng.choice(["y", "n"], n)})
    enc = pd.DataFrame({
        "enc": ["E{:06d}".format(i) for i in range(700)],
        "subj": rng.choice(pat["subj"], 700),
        "admitting": rng.choice(pat["subj"], 700)})
    return {"people": pat, "encounters": enc}


def main():
    # ---- FIXTURE ONE: A STAR
    fr = star()
    found = detect(fr)
    keys = {n: t.primary_key for n, t in found["tables"].items()}
    check("a star schema's three keys are found from the "
          "values: {}".format(keys),
          keys == {"patients": "mrn", "visits": "visit_id",
                   "meds": "rx_id"})
    pairs = {(lk.child, lk.parent) for lk in found["links"]}
    check("...and both children are linked to the parent, by "
          "value overlap rather than by column name: {}"
          .format(sorted(pairs)),
          ("visits", "patients") in pairs
          and ("meds", "patients") in pairs)
    src_int = integrity(fr, found["links"])
    vis_row = [r for r in src_int if r["child"] == "visits"][0]
    check("...and the 20 planted ORPHANS are measured in the "
          "SOURCE - {} of {} visits have a parent ({:.1%}). A "
          "generator reporting 100% without this line is "
          "taking credit for somebody else's cleaning"
          .format(vis_row["with_parent"], vis_row["rows"],
                  vis_row["share"]),
          vis_row["orphans"] == 20)

    gen = assign_keys(fr, found, seed=1)
    gen_int = check_integrity(gen, found)
    check("the generated tables keep referential integrity BY "
          "CONSTRUCTION - every child key is drawn from keys "
          "that already exist, so there is no step where a "
          "child can point at a parent that is not there",
          all(r["share"] >= 0.97 for r in gen_int))
    gvis = [r for r in gen_int if r["child"] == "visits"][0]
    check("...and the orphan SHARE is reproduced rather than "
          "cleaned away: source {:.1%}, generated {:.1%}"
          .format(vis_row["share"], gvis["share"]),
          abs(gvis["share"] - vis_row["share"]) < 0.02)
    check("...and NOT ONE real key is republished - an MRN is "
          "a direct identifier, and a generated file carrying "
          "one has republished it however synthetic everything "
          "around it is",
          not (set(gen["patients"]["mrn"].astype(str))
               & set(fr["patients"]["mrn"].astype(str))))
    check("...and the orphans point at keys that belong to "
          "NOBODY, so an orphan is an orphan rather than a "
          "real identifier that happens not to match",
          any(str(v).startswith("ORPHAN")
              for v in gen["visits"]["mrn"]))

    # ---- FIXTURE TWO: A CHAIN, AND A MEASUREMENT THAT IS NOT A KEY
    fr2 = chain()
    f2 = detect(fr2)
    k2 = {n: t.primary_key for n, t in f2["tables"].items()}
    check("a chain's keys are found at every level: {}"
          .format(k2),
          k2 == {"patients": "mrn", "visits": "visit_id",
                 "labs": "lab_id"})
    check("...and a CONTINUOUS MEASUREMENT is not a key, even "
          "though it is unique on every row and present on "
          "every row - the first version of this module made a "
          "lab value a primary key, and that key would then "
          "have been used to LINK TABLES",
          "value" not in f2["tables"]["labs"].pk_candidates)
    p2 = {(lk.child, lk.parent) for lk in f2["links"]}
    check("...and the chain is linked end to end: {}"
          .format(sorted(p2)),
          ("visits", "patients") in p2
          and ("labs", "visits") in p2)
    g2 = assign_keys(fr2, f2, seed=2)
    check("...and the generated chain holds integrity at every "
          "level",
          all(r["share"] >= 0.999
              for r in check_integrity(g2, f2)))

    # ---- FIXTURE THREE: HEADERS THAT LIE
    fr3 = lying_headers()
    f3 = detect(fr3)
    check("the key is found as `subj` even though another "
          "column is NAMED patient_id - values decide and "
          "headers only assist, which is the rule `scrub` "
          "already uses for PHI",
          f3["tables"]["people"].primary_key == "subj")
    check("...and the column NAMED patient_id, which holds lab "
          "values, is not a key candidate at all",
          "patient_id"
          not in f3["tables"]["people"].pk_candidates)
    check("...and an INTEGER key works, because a patient "
          "number usually is one and a detector that refuses "
          "those rejects the most common key there is",
          f3["tables"]["people"].primary_key == "subj"
          and str(fr3["people"]["subj"].iloc[0]).isdigit())
    check("...and TWO columns pointing at the same parent is "
          "reported as something to CHECK - it can be real "
          "(admitting and discharging clinician) and it is "
          "also what a mis-detection looks like",
          any("mis-detection" in w for w in f3["warnings"]))

    # ---- NEGATIVE CONTROLS
    rng = np.random.default_rng(11)
    unrelated = pd.DataFrame({
        "code": ["X{:04d}".format(i) for i in range(120)],
        "w": rng.normal(0, 1, 120)})
    f4 = detect({"patients": fr["patients"],
                 "codes": unrelated})
    check("two UNRELATED tables produce NO links - a detector "
          "that finds a key in everything also finds one in "
          "nothing",
          not f4["links"])
    check("...and the report SAYS so rather than printing an "
          "empty section, naming the likely cause",
          "NO LINKS FOUND" in render(f4, []))
    twoval = pd.DataFrame({"flag": rng.choice(["y", "n"], 300),
                           "n": rng.normal(0, 1, 300)})
    f5 = detect({"patients": fr["patients"], "flags": twoval})
    check("a two-value column is not a key - it matches any "
          "other two-value column by accident, which is the "
          "commensurability rule the constraint finder already "
          "applies",
          f5["tables"]["flags"].primary_key is None)

    # ---- AND IT REACHES THE CLI
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for name, f in fr.items():
            f.to_csv(d / (name + ".csv"), index=False)
        out = d / "schema.json"
        r = subprocess.run(
            [sys.executable, "-m", "synthkit.cli", "schema",
             str(d / "patients.csv"), str(d / "visits.csv"),
             str(d / "meds.csv"), "--json-out", str(out)],
            capture_output=True, text=True, cwd=str(ROOT),
            timeout=900)
        check("`synthkit schema` reads the tables and reports "
              "the schema - a capability wired to nothing is "
              "this repo's named silent-failure class",
              r.returncode == 0
              and "SCHEMA FOUND FROM THE VALUES" in r.stdout
              and "REFERENTIAL INTEGRITY IN THE SOURCE"
              in r.stdout)
        got = json.loads(out.read_text(encoding="utf-8"))
        check("...and the JSON carries the keys, the links "
              "with their match SHARE, and the integrity - a "
              "link holding on 94% of rows must read as 94% "
              "rather than as a yes",
              got["tables"]["patients"]["primary_key"] == "mrn"
              and any(0.0 < lk["match"] < 1.0
                      for lk in got["links"])
              and got["integrity"])
        r2 = subprocess.run(
            [sys.executable, "-m", "synthkit.cli", "schema",
             str(d / "patients.csv")],
            capture_output=True, text=True, cwd=str(ROOT),
            timeout=900)
        check("...and ONE table is refused with the reason and "
              "the command that does fit - with one table "
              "there is nothing to link",
              r2.returncode != 0
              and "at least two tables" in (r2.stdout + r2.stderr)
              and "synthkit types" in (r2.stdout + r2.stderr))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
