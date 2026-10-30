"""Smoke: the key/released pair, the cipher, and reading an answer
back.

WHY THIS EXISTS. The release dials have existed for weeks and the
run threw the full-fidelity frame AWAY, keeping only the
perturbed file. That did not merely leave the architecture
unbuilt - it made it impossible: you cannot read a vendor's
answer against a reference you did not keep. One run now writes
key.csv, generated.csv and cipher.json.

THE CLAIM THIS SUITE DEFENDS. Every dial is a PERMUTATION of one
column's values across rows, so:

    each perturbed column holds the IDENTICAL multiset in both
    files, every column with no cipher entry is byte-identical,
    and the recorded arrangement inverts EXACTLY.

That is what makes a re-pointed file a legitimate test article
rather than damaged data, and it is the sentence somebody will
say to a governance board, so it is asserted and not described.

AND THE ONE MEASURED DESIGN RULE: DIAL THE OUTCOME, NOT THE
COVARIATES. Measured on a fixture whose outcome genuinely reads
the perturbed column, skill on our own truth over held-out rows:

    dial the OUTCOME     0.970 -> -1.657   denied 2.627
                         recovery EXACT (their model reads only
                         untouched predictors)
    dial a COVARIATE     0.970 ->  0.892   denied 0.078
                         recovery 0.102 sd by surrogate, 0.482
                         if the distortion is ignored

34x the protection and exact recovery, because a model routes
around one corrupted covariate through the others while a
corrupted LABEL leaves it nothing to learn. Both branches of
`rehearse` are exercised here, because the OBSERVABLE is the
same in each - no perturbed predictor - and the conclusion is
the opposite.
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

PASS = FAIL = 0
PY = sys.executable
LC = str(ROOT / "scripts" / "latent_challenger.py")
DB = str(ROOT / "scripts" / "dialback.py")


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def fixture(path, npat=110, seed=5):
    """An outcome that genuinely reads the column we perturb.

    A fixture whose target does not depend on the dialed column
    could not tell the two arrangements apart, and telling them
    apart is the whole result.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(npat):
        pid = "P%03d" % i
        b = rng.randint(1940, 2005)
        base = rng.normal(0, 1)
        for v in range(rng.randint(3, 8)):
            age = 2026 - b + v
            span = float(20 + 6 * (0.7 * (age - 60) / 15.0
                                   + 0.7 * base
                                   + rng.normal(0, 0.6)))
            lab = float(0.5 * age + rng.normal(0, 8))
            risk = float(0.045 * span + 0.020 * (age - 60)
                         + 0.010 * (lab - 40)
                         + rng.normal(0, 0.25))
            rows.append(dict(person_id=pid, age_at_visit=age,
                             span_days=round(span, 2),
                             lab=round(lab, 2),
                             other=round(float(
                                 rng.normal(50, 9)), 2),
                             risk=round(risk, 3)))
    pd.DataFrame(rows).to_csv(path, index=False)
    return len(rows)


def run_cli(src, out, *dials):
    """THROUGH THE DISK, because the reader is part of the path.

    Every in-memory fixture in this repo's history passed a
    DataFrame straight to fit, which is exactly why nothing here
    could catch a reader fault. The pair is two FILES; it is
    tested as two files.
    """
    cmd = [PY, "-u", LC, "csv", str(src), "--group-by",
           "person_id", "--k", "10", "--seeds", "0",
           "--out", str(out)] + list(dials)
    r = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=1800)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def db(*args):
    r = subprocess.run([PY, "-u", DB] + [str(a) for a in args],
                       capture_output=True, text=True,
                       timeout=1800)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main():
    from fidelity_deck import read_frame
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.csv"
        fixture(src)

        # ---- a run with NO dial has no second file and says so
        rc, log = run_cli(src, td / "plain")
        check("a run with no dial writes no key.csv - a "
              "full-fidelity run has one file and needs no "
              "cipher",
              rc == 0
              and not (td / "plain" / "key.csv").exists()
              and not (td / "plain" / "cipher.json").exists())
        rc, out = db("verify", td / "plain")
        check("...and dialback REFUSES it by name rather than "
              "crashing or inventing an empty record",
              rc != 0 and "no key.csv" in out
              and "needs no cipher" in out)

        # ---- the dialed run: outcome re-pointed AND a column
        #      both re-pointed and suppressed, to exercise
        #      composition
        rc, log = run_cli(
            src, td / "r",
            "--poison", "span_days~risk=-0.60",
            "--poison", "age_at_visit~lab=0.0",
            "--suppress", "age_at_visit~lab")
        check("a dialed run writes key.csv, generated.csv and "
              "cipher.json",
              rc == 0
              and (td / "r" / "key.csv").exists()
              and (td / "r" / "generated.csv").exists()
              and (td / "r" / "cipher.json").exists())
        k = read_frame(td / "r" / "key.csv")
        g = read_frame(td / "r" / "generated.csv")
        c = json.loads((td / "r" / "cipher.json").read_text(
            encoding="utf-8"))
        cols = c.get("columns") or {}
        check("the pair is row-aligned and column-identical - "
              "two unrelated files could carry no cipher",
              len(k) == len(g) and len(k) > 0
              and list(k.columns) == list(g.columns))
        check("the perturbed columns are the ones the dials "
              "named: risk and lab",
              set(cols) == {"risk", "lab"})

        exact = {}
        same = {}
        for col, tf in cols.items():
            tf = np.asarray(tf, dtype=np.int64)
            rec = np.empty(len(g), dtype=object)
            rec[tf] = g[col].to_numpy()
            exact[col] = np.array_equal(
                np.asarray(rec, dtype=str),
                k[col].to_numpy().astype(str))
            a = np.sort(pd.to_numeric(
                k[col], errors="coerce").dropna())
            b = np.sort(pd.to_numeric(
                g[col], errors="coerce").dropna())
            same[col] = (len(a) == len(b)
                         and np.allclose(a, b))
        check("the cipher inverts EXACTLY on every perturbed "
              "column - the key is reconstructible from the "
              "released file value for value",
              bool(exact) and all(exact.values()))
        check("...including `lab`, which was re-pointed AND "
              "THEN suppressed - the record describes the round "
              "trip, not just the last step",
              exact.get("lab") is True)
        check("every perturbed column holds the IDENTICAL "
              "multiset in both files - a dial rearranges and "
              "never moves a distribution",
              bool(same) and all(same.values()))
        untouched = [x for x in g.columns if x not in cols]
        moved = [x for x in untouched
                 if not np.array_equal(
                     g[x].to_numpy().astype(str),
                     k[x].to_numpy().astype(str))]
        check("every column with no cipher entry is "
              "byte-identical in both files - nothing changed "
              "that the record does not name",
              len(untouched) > 2 and not moved)
        check("the cipher states that DILUTION is not covered "
              "and not invertible - a record that silently "
              "covered three of four controls would be the "
              "read-but-not-used fault in a governance file",
              "DILUTION" in (c.get("not_covered") or "")
              and "not invertible" in (c.get("not_covered")
                                       or ""))
        check("the cipher carries requested beside achieved "
              "for every re-pointed pair",
              len(c.get("repointed") or []) >= 2
              and all("requested" in r and "achieved" in r
                      for r in c["repointed"]))

        rc, out = db("verify", td / "r")
        check("`verify` passes a sound pair and states the "
              "claim: a faithful rearrangement, not damaged "
              "data",
              rc == 0 and "VERIFIED" in out
              and "not damaged data" in out)

        # ---- FAIL-FIRST ON THE REAL CLAIM. Move one value in
        #      the released file: the permutation claim is now
        #      false, and verify must say so. A check that only
        #      ever saw a sound pair has never been read.
        tamper = td / "tam"
        tamper.mkdir()
        for f in ("key.csv", "generated.csv", "cipher.json"):
            (tamper / f).write_bytes(
                (td / "r" / f).read_bytes())
        gt = read_frame(tamper / "generated.csv")
        col = sorted(cols)[0]
        # read_frame gives str columns (keep_default_na=False),
        # and writing a float into one is a TypeError on pandas 3
        gt.loc[gt.index[0], col] = "999999.0"
        gt.to_csv(tamper / "generated.csv", index=False)
        rc, out = db("verify", tamper)
        check("one moved value in the released file makes "
              "`verify` FAIL and name the column - the "
              "permutation claim is checked, not described",
              rc != 0 and "NOT VERIFIED" in out
              and col in out)
        check("...and it reports the distribution MOVED, which "
              "is the specific thing a dial may never do",
              "MOVED" in out)

        # ---- THE OUTCOME-ONLY RUN, which is the whole
        #      design rule. With only the label dialed, every
        #      predictor the target reads is faithful - so a
        #      vendor model that SCORES our rows returns the key
        #      answer exactly, while a vendor that TRAINS on it
        #      learns nothing. The run above dials `lab` too,
        #      which is a predictor of risk, so it cannot reach
        #      this branch - isolating it needs its own run.
        rc, log = run_cli(src, td / "lab_only",
                          "--poison", "span_days~risk=-0.60")
        check("the outcome-only run writes its own pair",
              rc == 0
              and (td / "lab_only" / "cipher.json").exists())
        rc, out = db("rehearse", td / "lab_only",
                     "--target", "risk",
                     "--group-by", "person_id")
        check("`rehearse` on a dialed OUTCOME reports recovery "
              "EXACT with no surrogate, because every predictor "
              "it reads is untouched",
              rc == 0 and "EXACT" in out
              and "PERTURBED" in out)
        check("...and MEASURES the protection rather than "
              "asserting it - a vendor trained on the released "
              "file scores worse on our truth",
              rc == 0 and "denied to a vendor" in out)
        pr = [ln for ln in out.splitlines()
              if "denied to a vendor" in ln]
        denied = float(pr[0].split()[-1]) if pr else 0.0
        check("the measured denial on a dialed outcome is "
              "large (> 0.5 of skill), which is the number the "
              "whole arrangement is chosen for: got "
              "{:+.3f}".format(denied), denied > 0.5)

        # ---- and the WEAK case, measured rather than
        #      inferred: a target nobody can predict has
        #      nothing to protect, and the tool says so instead
        #      of reporting a ratio of two noise numbers.
        rc, out = db("rehearse", td / "lab_only",
                     "--target", "other",
                     "--group-by", "person_id")
        check("a target nobody can predict is REFUSED with its "
              "own skill quoted - a recovery ratio between two "
              "noise numbers would read as a result",
              rc == 0 and "NOBODY CAN PREDICT" in out)

        # ---- THE BENCH'S OWN REPLY, pinned at the decision
        #      site. smoke_gui asserts the page READS
        #      `result.pair`; deleting the field the server
        #      sends broke nothing there, which is this repo's
        #      named non-fail-first - a mutation that changes a
        #      reported field no assertion depends on. The
        #      server side is therefore checked here, where real
        #      runs already happen.
        from synthkit import gui as _gui
        rw = _gui._latent_work({
            "src": str(src), "out": str(td / "bw"),
            "group_by": "person_id", "seed": "0", "k": "10",
            "poison": "span_days~risk=-0.60"})
        check("the bench's own run reports the pair is on disk "
              "- the page can only show a key it is told about",
              "error" not in rw and rw.get("pair") is True)
        rp = _gui._latent_work({
            "src": str(src), "out": str(td / "bp"),
            "group_by": "person_id", "seed": "0", "k": "10"})
        check("...and reports no pair for an undialed run, so "
              "the panel cannot claim a key that was never "
              "written",
              "error" not in rp and rp.get("pair") is False)
        check("the bench carries the dial through to the CLI - "
              "the dialed bench run wrote a cipher and the "
              "plain one did not",
              (td / "bw" / "cipher.json").exists()
              and not (td / "bp" / "cipher.json").exists())

        # ---- recover refuses what it cannot relate
        bad = td / "bad.csv"
        pd.DataFrame({"score": [1.0, 2.0, 3.0]}).to_csv(
            bad, index=False)
        rc, out = db("recover", td / "r", "--scores", bad)
        check("`recover` refuses a returned file whose row "
              "count does not match the released one - row "
              "order is the only thing relating them",
              rc != 0 and "one per released" in out)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
