"""Checks for the Expert Determination evidence package (ED-1).

THE LOAD-BEARING CHECK IS ON THE FIELD NAMES, ASSERTED AGAINST
THE BATTERIES' LIVE OUTPUT. The first cut of `assess` read
`arms`, `membership` and `attribute` out of one invented
`attack.json`. The real batteries write TWO files with different
shapes, so every element would have read OWED on a run that had
the evidence - and the package would have told an expert that
measurements this project has published for months did not
exist. Seventh instance of the wrong-key fault here, and this
suite applies the prescribed fix: call the batteries and assert
the names, never remember them.

AND ON `OWED` BEING A FIRST-CLASS ANSWER. A package that looks
complete and is not is worse than no package, because an expert
signs against evidence that does not exist. Three ways that goes
wrong are pinned: one adversary is not a threat model, one seed
is a draw rather than a measurement, and an excess with no
positive control is the check-that-cannot-fail shape.

The tool must also never call itself a determination. That is
the same discipline as refusing to describe an output as
differentially private when only the weights are accounted, and
a check asserts the refusal travels in the data rather than in
one renderer's prose.
"""

import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from synthkit import determination as DET            # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def full_artifacts(td):
    """A run directory plus battery outputs, all present."""
    json.dump({"summary": {"columns": 12}, "nn_ratio": 1.15},
              io.open(os.path.join(td, "fidelity.json"), "w"))
    json.dump({"marginals": {"x": {}},
               "bounds_are_k_anonymous": True},
              io.open(os.path.join(td, "blueprint.json"), "w"))
    json.dump({"rows": 100},
              io.open(os.path.join(td, "provenance.json"), "w"))
    mem = {"seeds": [11, 23, 37], "worst_auc_max": 0.551,
           "verdict": "PASS",
           "runs": [{"nearest_neighbor": {"auc": 0.514,
                                          "attacker_sees": "rows"},
                     "likelihood": {"auc": 0.479,
                                    "attacker_sees": "weights"},
                     "worst_auc": 0.514, "verdict": "PASS"}]}
    att = {"seeds": [{"excess_over_control": -0.009}],
           "positive_control": [{"excess_over_control": 0.145}]}
    scope = ("SCOPE. One cohort of 800 patients from one extract "
             "at k=10, output-only release. The determination does "
             "not extend to a different cohort, a different k, or "
             "a release of the trained network. Residual risk: the "
             "effect curves, interaction surfaces and dynamics "
             "have not been audited by the attack battery, and "
             "k-anonymity on what is published is not differential "
             "privacy.")
    return mem, att, scope


def main():
    # ---- 1. THE FIELD NAMES, AGAINST A LIVE CALL
    import membership_new_path as MN
    from synthkit.attack import attribute_disclosure
    _live = MN.run_once(60, 4, 11, 10, sets=0)
    check("the MEMBERSHIP field names come from a LIVE battery "
          "run, not from memory: run_once returns {} - and "
          "`assess` reads nearest_neighbor.auc, likelihood.auc "
          "and worst_auc out of exactly those"
          .format(sorted(k for k in _live
                         if k in ("nearest_neighbor",
                                  "likelihood", "worst_auc",
                                  "verdict"))),
          "nearest_neighbor" in _live and "likelihood" in _live
          and "auc" in _live["nearest_neighbor"]
          and "auc" in _live["likelihood"]
          and _live.get("worst_auc") is not None
          and _live.get("verdict"))

    def _rows(n, seed):
        r = np.random.default_rng(seed)
        return [{"sex": int(r.integers(0, 2)),
                 "site": int(r.integers(0, 3)),
                 "age": float(r.normal(60, 12)),
                 "hiv": int(r.integers(0, 2))} for _ in range(n)]
    _ad = attribute_disclosure("hiv", ["sex", "site", "age"],
                               _rows(400, 1), _rows(400, 2),
                               _rows(400, 3), seed=0)
    check("...and the ATTRIBUTE field names likewise - a live "
          "call returns excess_over_control and control_accuracy "
          "among {} keys, and the package's status turns on the "
          "CONTROL being present rather than on the excess "
          "alone".format(len(_ad)),
          "excess_over_control" in _ad
          and "control_accuracy" in _ad
          and "synthetic_accuracy" in _ad)

    # ---- 2. BOTH BRANCHES ARE REACHABLE
    with tempfile.TemporaryDirectory() as td:
        mem, att, scope = full_artifacts(td)
        _bare = DET.assess(
            fidelity=json.load(io.open(os.path.join(
                td, "fidelity.json"))),
            blueprint=json.load(io.open(os.path.join(
                td, "blueprint.json"))),
            provenance=json.load(io.open(os.path.join(
                td, "provenance.json"))))
        _full = DET.assess(
            fidelity=json.load(io.open(os.path.join(
                td, "fidelity.json"))),
            blueprint=json.load(io.open(os.path.join(
                td, "blueprint.json"))),
            provenance=json.load(io.open(os.path.join(
                td, "provenance.json"))),
            membership=mem, attribute=att,
            scrub_inventory=[{"category": "names"}],
            scope=scope)
    check("a run WITHOUT the batteries reads {} of {} with {} "
          "OWED, and one WITH them reads {} of {} and READY - "
          "both branches reachable, which the first cut was not: "
          "`residual` was unconditionally false, so `ready` "
          "could never be true and the every-element branch was "
          "dead code inside a tool built to catch exactly that"
          .format(_bare["elements_present"],
                  _bare["elements_total"],
                  len(_bare["elements_owed"]),
                  _full["elements_present"],
                  _full["elements_total"]),
          _bare["ready"] is False
          and len(_bare["elements_owed"]) >= 4
          and _full["ready"] is True
          and not _full["elements_owed"])

    # ---- 3. THE COUNTS ARE DERIVED FROM THE ROWS
    check("the totals are DERIVED from the element rows rather "
          "than written beside them - {} present and {} owed "
          "against {} rows, and the two add up"
          .format(_full["elements_present"],
                  len(_full["elements_owed"]),
                  len(_full["elements"])),
          _full["elements_present"]
          + len(_full["elements_owed"])
          == len(_full["elements"])
          == _full["elements_total"]
          and _bare["elements_present"]
          + len(_bare["elements_owed"])
          == _bare["elements_total"])

    # ---- 4. THE THREE WAYS `OWED` MUST STILL FIRE
    _one_arm = dict(mem)
    _one_arm["runs"] = [{"nearest_neighbor": {"auc": 0.51},
                         "worst_auc": 0.51, "verdict": "PASS"}]
    _r1 = DET.assess(membership=_one_arm)
    check("ONE ADVERSARY IS NOT A THREAT MODEL: a membership "
          "summary carrying only the output-surface arm reads "
          "threat_model OWED. The output surface and the weights "
          "surface are different recipients and this project's "
          "recorded numbers differ sharply between them, so one "
          "arm answers a question nobody asked",
          "threat_model" in _r1["elements_owed"])
    _one_seed = dict(mem, seeds=[11])
    _r2 = DET.assess(membership=_one_seed)
    check("...ONE SEED IS A DRAW, NOT A MEASUREMENT: a summary "
          "with a single seed reads membership OWED. The battery "
          "itself refuses fewer than two, so a summary with one "
          "did not come from it",
          "membership" in _r2["elements_owed"])
    _no_ctrl = {"seeds": [{"excess_over_control": -0.009}],
                "positive_control": []}
    _r3 = DET.assess(attribute=_no_ctrl)
    check("...and AN EXCESS WITH NO POSITIVE CONTROL reads "
          "attribute OWED - a clean excess whose leaky-generator "
          "arm never ran is the check-that-cannot-fail shape, "
          "and this project has recorded a reassuring number "
          "that was worth nothing for exactly that reason",
          "attribute" in _r3["elements_owed"])
    _r4 = DET.assess(scope="one cohort, one k")
    check("...and A ONE-LINE SCOPE IS NOT A SCOPE: residual risk "
          "stays OWED until a written statement exists, because "
          "it is the one element a RUN cannot supply - a scope "
          "nobody wrote is not a scope, and deriving one would "
          "be inventing the bound on somebody else's "
          "determination",
          "residual" in _r4["elements_owed"]
          and _r4["scope_chars"] < 200)

    # ---- 5. IT MUST NEVER CALL ITSELF A DETERMINATION
    _txt = "\n".join(DET.plain(_full))
    check("the package refuses to call itself a determination, "
          "in the DATA rather than one renderer's prose, so no "
          "renderer can drop it - and it says no source record "
          "travels with it",
          len(DET.NOT_A_DETERMINATION) >= 3
          and "EVIDENCE PACKAGE, not a determination"
          in " ".join(DET.NOT_A_DETERMINATION)
          and "No source record travels"
          in " ".join(DET.NOT_A_DETERMINATION)
          and all(line in _txt
                  for line in DET.NOT_A_DETERMINATION))
    check("...and a READY package says READINESS rather than "
          "anything stronger - the determination is still a "
          "qualified person's to make",
          "READINESS, not a determination" in _txt
          and "qualified person" in _txt)
    check("...and it is organised around the STATUTE's own two "
          "requirements, quoted, so a reader is not mapping our "
          "headings onto theirs",
          "164.514(b)(1)" in DET.STATUTE
          and "documents the methods and results" in DET.STATUTE
          and DET.STATUTE in _txt)

    # ---- 6. EVERY ELEMENT SAYS WHY IT IS ASKED
    check("every one of the {} elements names the EVIDENCE that "
          "answers it and WHY it is asked - a status a reader "
          "cannot go and check is a status taken on trust"
          .format(len(DET.ELEMENTS)),
          len(DET.ELEMENTS) >= 9
          and all(len(e["evidence"]) > 40 and len(e["why"]) > 60
                  and e["need"] for e in DET.ELEMENTS)
          and len(set(e["key"] for e in DET.ELEMENTS))
          == len(DET.ELEMENTS))

    # ---- 7. THE CLI, THROUGH THE PROCESS BOUNDARY
    with tempfile.TemporaryDirectory() as td:
        mem, att, scope = full_artifacts(td)
        mp = os.path.join(td, "mem.json")
        ap = os.path.join(td, "att.json")
        sp = os.path.join(td, "scope.txt")
        json.dump(mem, io.open(mp, "w"))
        json.dump(att, io.open(ap, "w"))
        io.open(sp, "w").write(scope)
        _bad = subprocess.run(
            [sys.executable, "-m", "synthkit.cli",
             "determination", td],
            capture_output=True, text=True, cwd=str(ROOT))
        _good = subprocess.run(
            [sys.executable, "-m", "synthkit.cli",
             "determination", td, "--membership", mp,
             "--attribute", ap, "--scope", sp, "--scrubbed"],
            capture_output=True, text=True, cwd=str(ROOT))
        _missing = subprocess.run(
            [sys.executable, "-m", "synthkit.cli",
             "determination",
             os.path.join(td, "nope")],
            capture_output=True, text=True, cwd=str(ROOT))
        _wrote = os.path.exists(os.path.join(
            td, "determination.json"))
    check("the CLI exits 3 on an INCOMPLETE package and 0 on a "
          "complete one ({} and {}) - the same reasoning as "
          "`scrub --apply` refusing while a fresh detect still "
          "finds something: a document that says it is ready "
          "while half the work is outstanding is the worst "
          "outcome available"
          .format(_bad.returncode, _good.returncode),
          _bad.returncode == 3 and _good.returncode == 0)
    check("...and it REFUSES a missing run directory by name "
          "with exit 2 rather than a traceback",
          _missing.returncode == 2
          and "STOPPED" in _missing.stderr
          and "Traceback" not in _missing.stderr)
    check("...and it writes determination.json beside the run, "
          "so the package is an artifact rather than a terminal "
          "scroll", _wrote)
    check("...and the incomplete run NAMES what is owed in its "
          "output rather than only counting it, because a count "
          "a reader cannot act on is the fault this project "
          "records about every totals line",
          "OWED:" in _bad.stdout
          and "membership" in _bad.stdout
          and "signing against nothing" in _bad.stdout)

    print()
    if FAIL:
        print("{} of {} checks failed.".format(FAIL, PASS + FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
