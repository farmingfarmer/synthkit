"""`regate.py` re-measures two finished runs with TODAY'S code.

WHY THIS SUITE EXISTS. The script had no checks at all, and a number
it produced was about to be reported to a stakeholder - the same fault
this project recorded for `peek.py`, where "the tool written so the
operator would not hand-write a one-liner was itself never run by
anything".

WHAT IS LOAD-BEARING HERE is the ENGINE guard. A whole night went into
a "twelve-pair regression" that was two different engines:
`run_deliverable` carries `settings.engine = "neural"` and `run_n9g`
carries no engine field, because the rules pipeline does not write one.
Four control arms were then built and every one came back innocent -
they had to, since each switched off a RULES mechanism while the
baseline was a NEURAL file - and a five-run bisection was queued
against a difference no commit causes.
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

from regate import (engine_of, shared_close, pair_map,   # noqa: E402
                    provenance_rows)
from synthkit import blueprint as B                    # noqa: E402
from synthkit.discover import discover                 # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def write_prov(d, obj):
    d.mkdir(parents=True, exist_ok=True)
    (d / "provenance.json").write_text(
        json.dumps(obj), encoding="utf-8")
    return d


def fid(pairs):
    """A fidelity record shaped the way `pair_map` READS it.

    Asserted against the real reader rather than invented: the first
    cut of this helper used a top-level `pairs` key, `pair_map`
    returned nothing, and `shared_close` came back None - the
    wrong-key fault this project records nine times, committed inside
    the suite written about a comparison going wrong.
    """
    return {"relationships": {"pairs": [
        {"child": c, "parent": p, "source": s, "generated": g}
        for (c, p, s, g) in pairs]}}


def main():
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)

        # ---- engine_of: the guard that would have saved the night --
        neural = write_prov(t / "neu", {
            "blueprint_version": 1, "source": {"rows_read": 600},
            "settings": {"group_by": "person_id", "engine": "neural"},
            "contract": {}})
        rules = write_prov(t / "rul", {
            "made_at": "x", "build": {}, "blueprint_version": 1,
            "source": {"rows_read": 600},
            "settings": {"group_by": "person_id", "generate": True,
                         "seed": 5, "holdout": 0.3, "excluded": ""},
            "privacy": {"k": 10}, "note": "..."})
        mystery = write_prov(t / "mys", {
            "blueprint_version": 1, "source": {"rows_read": 600},
            "settings": {"group_by": "person_id"}})

        check("a NEURAL run is identified from its own provenance - "
              "`settings.engine` is written by the neural engine and "
              "by nothing else in the package",
              engine_of(neural) == "neural")

        check("...and a RULES run is identified from the fields the "
              "rules pipeline writes (`privacy` and `note` at the "
              "top level), because that path records no engine name "
              "at all",
              engine_of(rules) == "rules")

        # ABSENT IS NOT THE SAME FACT AS MATCHING. A provenance that
        # looks like neither must not be guessed at - guessing is how
        # a cross-engine comparison gets published as a regression.
        check("...and a provenance that looks like NEITHER returns "
              "nothing rather than defaulting to one of them - "
              "absent is not the same fact as matching",
              engine_of(mystery) is None)

        check("...and a run with no provenance at all returns "
              "nothing rather than raising",
              engine_of(t / "nothing_here") is None)

        # ---- shared_close: the number that was misread -------------
        a = fid([("y", "x", 0.90, 0.88),      # close in A
                 ("q", "p", 0.50, 0.45),      # close in A
                 ("m", "n", 0.70, 0.10)])     # not close in A
        b = fid([("y", "x", 0.90, 0.30),      # A holds, B loses
                 ("q", "p", 0.50, 0.47),      # close in both
                 ("m", "n", 0.70, 0.68)])     # B gains
        r = shared_close(a, b)

        check("the helper's own fixture matches what `pair_map` "
              "READS - a fixture that shares the reader's invented "
              "key agrees with it and disagrees with the artifact",
              len(pair_map(a)) == 3)

        check("`close` is counted on the pairs BOTH runs relate - "
              "each run's percentage is over its own blueprint "
              "vocabulary, so only the intersection compares",
              r["shared"] == 3 and r["a_close"] == 2
              and r["b_close"] == 2)

        # THE NET HIDES THE TRAFFIC. One lost and one gained nets to
        # zero, and a reader told only the net would look for one
        # mechanism where there are two.
        check("...and the pairs that FLIP are NAMED, not just "
              "counted: a net of zero here is one pair lost and one "
              "gained, which a single number cannot say",
              len(r["lost"]) == 1 and r["lost"][0][0] == ("y", "x"))

        check("...and a flip carries its source value and BOTH "
              "generated values, so a reader can see whether the "
              "pair faded or inverted",
              abs(r["lost"][0][1] - 0.90) < 1e-9
              and abs(r["lost"][0][2] - 0.88) < 1e-9
              and abs(r["lost"][0][3] - 0.30) < 1e-9)

        check("...and two runs sharing no related pair report "
              "nothing rather than dividing by zero",
              shared_close(fid([("y", "x", 0.9, 0.8)]),
                           fid([("b", "a", 0.9, 0.8)])) is None)

        # ---- the guard, end to end through the CLI -----------------
        # A REAL blueprint, because `regate` RE-MEASURES: it reads
        # blueprint.json and generated.csv and runs the comparison
        # itself, so a hand-written fidelity file would never reach
        # the code path this guard lives in.
        rng = np.random.default_rng(5)
        n = 600
        x = rng.normal(0, 1, n)
        df = pd.DataFrame({
            "person_id": ["P%03d" % (i // 3) for i in range(n)],
            "x": np.round(x, 4),
            "z": np.round(rng.normal(0, 1, n), 4),
            "y": np.round(2.0 * x + rng.normal(0, 0.2, n), 4)})
        src = t / "src.csv"
        df.to_csv(src, index=False)
        cat = discover(df, group_by="person_id", targets=["y"])
        bp = B.build(df, cat, group_by="person_id")

        for d in (t / "neu", t / "rul"):
            (d / "blueprint.json").write_text(
                json.dumps(bp), encoding="utf-8")
            df.to_csv(d / "generated.csv", index=False)

        rv = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "regate.py"),
             str(t / "neu"), str(t / "rul"), "--src", str(src)],
            capture_output=True, text=True, cwd=str(ROOT))
        out = rv.stdout + rv.stderr

        check("the CLI NAMES the engine of each run it was given - "
              "a cross-engine number must not be readable without "
              "seeing that it is one",
              "neural" in out and "rules" in out)

        check("...and when the two engines DIFFER it says so in "
              "words, and says a difference is NOT a regression and "
              "not attributable to any commit",
              "DIFFERENT ENGINES" in out
              and "NOT a regression" in out)

        check("...and it names the criteria that are worse than "
              "incomparable across engines, because the neural path "
              "never consults published surfaces",
              "interaction surfaces" in out
              and "by construction" in out)

        # A GUARD THAT FIRES ON EVERYTHING IS AS USELESS AS ONE THAT
        # CANNOT FIRE - this project has said so about a
        # contradiction check that flagged 55 of 60 healthy columns.
        rv2 = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "regate.py"),
             str(t / "neu"), str(t / "neu"), "--src", str(src)],
            capture_output=True, text=True, cwd=str(ROOT))
        check("...and two runs from the SAME engine produce NO such "
              "warning - a guard that fires on everything proves "
              "nothing about the runs it blessed",
              "DIFFERENT ENGINES" not in (rv2.stdout + rv2.stderr))

        # ---- the wrong-source refusal, which already existed -------
        small = t / "small.csv"
        df.head(20).to_csv(small, index=False)
        rv3 = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "regate.py"),
             str(t / "neu"), "--src", str(small)],
            capture_output=True, text=True, cwd=str(ROOT))
        check("a run whose recorded source row count does not match "
              "the source handed in is REFUSED by name - a run "
              "measured against an extract that never produced it "
              "looks entirely reasonable",
              "refusing" in (rv3.stdout + rv3.stderr))

        # AND THE KEY IS ASSERTED AGAINST THE WRITERS, because this
        # guard had never fired on a real run. It looked for
        # `source.rows` and `source.n_rows`; BOTH writers emit
        # `source.rows_read`. A guard that cannot fire, in the one
        # place a whole session was already lost to measuring a run
        # against an extract that never produced it.
        _pipe = (ROOT / "synthkit" / "pipeline.py").read_text(
            encoding="utf-8")
        _lat = (ROOT / "scripts" / "latent_challenger.py").read_text(
            encoding="utf-8")
        check("the row-count guard READS the key both writers "
              "actually emit (`source.rows_read`) - it looked for "
              "`source.rows`, which nothing writes, so the refusal "
              "had never fired on any real run",
              provenance_rows(write_prov(t / "pk", {
                  "source": {"rows_read": 1234}})) == 1234)

        check("...and both writers are asserted to still emit it, so "
              "a rename breaks this suite rather than silently "
              "turning the guard back off",
              '"rows_read"' in _pipe and '"rows_read"' in _lat)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
