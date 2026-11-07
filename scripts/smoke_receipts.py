"""Checks for `synthkit receipts` - per-claim receipts (REC-1).

WHAT THESE ARE FOR. Goal 3 is "every pattern found, explained, with
receipts" and the receipts were the part that stayed PLANNED for
months. The evidence already existed, spread across three artifacts
and two of them prose, so auditing ONE pattern meant holding all
three in your head at once.

THE LOAD-BEARING CHECKS ARE NOT "IT PRINTS SOMETHING". They are:
every number is READ from the run's artifacts rather than
recomputed, so the receipt and the gate cannot disagree; every
receipt carries a LIMITS section, because one that only carried the
evidence would read as a certificate; and the command is run as a
SUBPROCESS, because a content check cannot see a dead code path -
the first version of this command raised NameError on `os` at the
write step and the printed output looked perfect right up to the
traceback, which stderr showed BEFORE the buffered stdout.
"""

import io
import json
import os as _os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))

from synthkit import receipts                       # noqa: E402
from synthkit import sets as _sets                  # noqa: E402

PASS = FAIL = 0
ROOT = str(Path(__file__).resolve().parent.parent)


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def fixture(path, n_pat=160, n_vis=4, seed=5):
    """A set column with a size partner and a real driver.

    THE FIXTURE MUST CARRY A CLAIM ON A SEARCH COLUMN AND A CLAIM
    ON A COLUMN THE OPERATOR RECEIVES. The receipt's sharpest
    statement is that a claim whose child is a `__has__` indicator
    cannot be checked in the output at all, and a fixture with no
    set column could not reach it.
    """
    rng = np.random.default_rng(seed)
    n = n_pat * n_vis
    pid = np.repeat(["P{:04d}".format(i) for i in range(n_pat)],
                    n_vis)
    vocab = ["tok{:02d}".format(i) for i in range(10)]
    sev = np.round(rng.normal(5, 1.6, n), 2)
    procs, cnt = [], []
    for i in range(n):
        # SIZE RIDES SEVERITY, so there is a genuine claim to
        # receipt rather than noise dressed as a finding.
        lam = max(0.0, (sev[i] - 2.0) / 2.5)
        kk = int(min(4, rng.poisson(lam)))
        if kk == 0:
            procs.append("")
            cnt.append(0)
        else:
            procs.append(";".join(
                rng.choice(vocab, size=kk, replace=False)))
            cnt.append(kk)
    pd.DataFrame({
        "person_id": pid,
        "severity": sev,
        "procedures": procs,
        "procedure_count": cnt,
    }).to_csv(path, index=False)


def main():
    with tempfile.TemporaryDirectory() as td:
        src = _os.path.join(td, "src.csv")
        run = _os.path.join(td, "run")
        fixture(src)
        r = subprocess.run(
            [sys.executable, "-m", "synthkit.cli", "fit",
             "--src", src, "--out", run, "--group-by",
             "person_id", "--generate", "--seed", "3"],
            cwd=ROOT, capture_output=True, text=True)
        check("the fixture fits and generates, so there is a real "
              "run to receipt (exit {})".format(r.returncode),
              r.returncode == 0)
        for name in ("catalogue.json", "blueprint.json",
                     "fidelity.json"):
            check("...and the run wrote {} - a receipt is an "
                  "ASSEMBLY of these and cannot be built without "
                  "them".format(name),
                  _os.path.exists(_os.path.join(run, name)))

        doc = receipts.build(run)
        check("the run yields receipts, one per published claim "
              "({} claim(s))".format(doc["claims"]),
              doc["claims"] > 0
              and len(doc["receipts"]) == doc["claims"])

        # ---- NOTHING IS RECOMPUTED. The whole argument for this
        #      module is that a second implementation is how the
        #      shipped artifact and the report come to disagree,
        #      so every number must be traceable to an artifact.
        cat = json.load(io.open(_os.path.join(run, "catalogue.json"),
                                encoding="utf-8"))
        bycl = dict((c.get("child"), c) for c in cat["claims"])
        agree = 0
        for rec in doc["receipts"]:
            src_claim = bycl.get(rec["child"])
            if src_claim is None:
                continue
            if (rec["strength"]["skill_out_of_sample"]
                    == src_claim.get("skill")
                    and rec["support"]["train_rows"]
                    == src_claim.get("n_train_rows")
                    and rec["support"]["holdout_patients"]
                    == src_claim.get("holdout_patients")):
                agree += 1
        check("every receipt's skill and support are the "
              "catalogue's OWN numbers, to the digit ({} of {}) - "
              "a receipt that recomputed them could disagree with "
              "the gate reading the same run"
              .format(agree, doc["claims"]),
              agree == doc["claims"])

        fid = json.load(io.open(_os.path.join(run, "fidelity.json"),
                                encoding="utf-8"))
        fpairs = dict(
            ((p.get("child"), p.get("parent")), p)
            for p in fid["relationships"]["pairs"])
        checked = mism = 0
        for rec in doc["receipts"]:
            for sv in rec["survival"]:
                p = fpairs.get((rec["child"], sv["parent"]))
                if p is None:
                    continue
                checked += 1
                if (sv["source"] != p.get("source")
                        or sv["generated"] != p.get("generated")):
                    mism += 1
        check("...and every survival reading is fidelity.json's "
              "own source and generated value ({} compared, {} "
              "mismatched)".format(checked, mism),
              checked > 0 and mism == 0)

        # ---- THE LIMITS SECTION IS THE POINT
        check("EVERY receipt carries a LIMITS section - one that "
              "only carried the evidence would read as a "
              "certificate",
              all(r["limits"] for r in doc["receipts"]))
        allim = "\n".join(
            l for r in doc["receipts"] for l in r["limits"])
        check("...and every one says ASSOCIATION, NOT CAUSATION, "
              "because the direction was chosen by out-of-sample "
              "skill and that says which way PREDICTS better, not "
              "which way causes",
              all(any("ASSOCIATION, NOT CAUSATION" in l
                      for l in r["limits"])
                  for r in doc["receipts"]))
        check("...and every one says ONE COHORT: the holdout is "
              "held-out PATIENTS from the same extract, which "
              "tests new people here and nothing about another "
              "site",
              all(any("ONE COHORT" in l for l in r["limits"])
                  for r in doc["receipts"]))
        check("...and every one names the k it was screened at, "
              "so a magnitude short of its source can be read as "
              "the privacy rule working rather than a sampler "
              "fault",
              all(any("k-SCREENED" in l for l in r["limits"])
                  for r in doc["receipts"]))
        check("...and every one says TRIMS ARE NOT IN THIS "
              "RECEIPT, rather than leaving a reader to assume "
              "a trimmed parent was lost - trimmed is not lost, "
              "and the field does not exist to assert it from",
              all(any("TRIMS ARE NOT" in l for l in r["limits"])
                  for r in doc["receipts"]))

        # ---- THE SHARPEST STATEMENT, and the fixture reaches it
        scaff = [r for r in doc["receipts"]
                 if _sets.is_scaffolding(str(r["child"]))]
        check("the fixture really produces a claim whose CHILD is "
              "a search column ({} of {}) - the receipt's sharpest "
              "statement is about those, and a fixture without a "
              "set column could not reach it"
              .format(len(scaff), doc["claims"]),
              len(scaff) > 0)
        check("...and such a claim says it CANNOT BE CHECKED in "
              "the generated file at all, because the column is "
              "dropped before the file is written - nobody can "
              "read a column that is not there",
              all(any("THE CHILD IS A SEARCH COLUMN" in l
                      for l in r["limits"]) for r in scaff))
        check("...and the predicate is `sets.is_scaffolding`, the "
              "same one the pipeline uses - two files deciding "
              "separately what counts as scaffolding is how the "
              "two halves come to disagree",
              _sets.is_scaffolding("procedures__has__tok00")
              and _sets.is_scaffolding("procedures__n")
              and not _sets.is_scaffolding("procedure_count"))

        # ---- WEAK IS SAID, NOT HIDDEN
        weak = [r for r in doc["receipts"]
                if isinstance(r["strength"]["skill_out_of_sample"],
                              float)
                and r["strength"]["skill_out_of_sample"]
                < receipts.WEAK_SKILL]
        if weak:
            check("a claim under the {:.2f} skill bar is called "
                  "WEAK in its own limits rather than suppressed, "
                  "because a thin finding and no finding are "
                  "different facts ({} such)"
                  .format(receipts.WEAK_SKILL, len(weak)),
                  all(any("WEAK CLAIM" in l for l in r["limits"])
                      for r in weak))
        else:
            check("no claim here falls under the weak bar, so "
                  "that branch is NOT exercised by this fixture - "
                  "said out loud rather than counted as covered",
                  True)

        # ---- SURVIVAL VERDICTS, and absent is not clean
        check("the survival verdicts are counted across all "
              "parent pairs, so a reader gets a total to check "
              "the rows against: {}"
              .format(", ".join("{} {}".format(v, k) for k, v
                                in sorted(
                                    doc["survival_verdicts"]
                                    .items()))),
              sum(doc["survival_verdicts"].values())
              == sum(len(r["survival"]) for r in doc["receipts"]))
        check("...and a pair the run never related reads NOT "
              "MEASURED rather than passing silently - absent is "
              "not the same fact as fine",
              all(sv["verdict"] != "kept"
                  for r in doc["receipts"] for sv in r["survival"]
                  if sv["source"] is None))

        # ---- A RUN WITH NO fidelity.json
        bare = _os.path.join(td, "bare")
        _os.makedirs(bare)
        io.open(_os.path.join(bare, "catalogue.json"), "w",
                encoding="utf-8").write(json.dumps(
                    {"claims": cat["claims"][:1],
                     "holdout": cat.get("holdout")}))
        bdoc = receipts.build(bare)
        check("a run with NO fidelity.json still yields receipts, "
              "and the render says no survival was measured for "
              "ANY claim - absent, not clean",
              bdoc["generated_compared"] is False
              and "absent, not clean" in receipts.render(bdoc))

        # ---- THE COMMAND, AS A SUBPROCESS. A content check
        #      cannot see a dead code path: the first version
        #      raised NameError on `os` at the write step, and the
        #      printed output looked perfect right up to the
        #      traceback.
        rc = subprocess.run(
            [sys.executable, "-m", "synthkit.cli", "receipts",
             run], cwd=ROOT, capture_output=True, text=True)
        check("`synthkit receipts RUNDIR` exits 0 with an EMPTY "
              "stderr - the first version printed a perfect "
              "report and then raised NameError at the write, "
              "which only a subprocess check can see",
              rc.returncode == 0 and rc.stderr.strip() == "")
        dest = _os.path.join(run, "receipts.json")
        check("...and it writes receipts.json into the run "
              "directory, beside the artifacts it was assembled "
              "from", _os.path.exists(dest))
        written = json.load(io.open(dest, encoding="utf-8"))
        check("...and the written artifact carries the same claim "
              "count as the library call, so the file and the "
              "screen cannot disagree",
              written["claims"] == doc["claims"])
        check("...and the report states that nothing was "
              "recomputed, which is the whole argument for "
              "assembling rather than re-deriving",
              "recomputing nothing" in written["note"])
        rc2 = subprocess.run(
            [sys.executable, "-m", "synthkit.cli", "receipts", td],
            cwd=ROOT, capture_output=True, text=True)
        check("a directory with no catalogue.json is REFUSED by "
              "name and exits 2, rather than printing an empty "
              "report that reads as a run with no patterns",
              rc2.returncode == 2
              and "no catalogue.json" in rc2.stderr)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
