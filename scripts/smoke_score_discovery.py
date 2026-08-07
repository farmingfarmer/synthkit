"""Smoke: precision and recall against planted truth.

The scorer decides what every future change is worth, so its own
failure modes matter. Two in particular, both learned the hard way:
an edge learned in the opposite orientation is the SAME relationship,
and a relationship that cannot be found by construction must not be
counted against recall without being named.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def run(*args):
    return subprocess.run([sys.executable] + list(args),
                          capture_output=True, text=True, cwd=str(ROOT))


TRUTH = {
    "relationships": [
        {"child": "y1", "parents": ["x1"], "kind": "linear"},
        {"child": "y2", "parents": ["x2"], "kind": "nonlinear-u"},
        {"child": "y3", "parents": ["a3", "b3"], "kind": "interaction"},
        {"child": "y4", "parents": ["a4", "b4"],
         "kind": "interaction-no-main-effect",
         "note": "a pairwise search cannot see this at any n"},
    ],
    "noise_columns": ["noise_00", "noise_01"],
}


def score(td, edges, name):
    t = Path(td) / "truth.json"
    t.write_text(json.dumps(TRUTH), encoding="utf-8")
    m = Path(td) / (name + ".json")
    m.write_text(json.dumps({"report": {"edges": edges}}),
                 encoding="utf-8")
    return run("scripts/score_discovery.py", "--truth", str(t),
               "--model", str(m))


def main():
    with tempfile.TemporaryDirectory() as td:
        # everything found, and y1 found in the OPPOSITE orientation
        r = score(td, [
            {"child": "x1", "parents": ["y1"]},
            {"child": "y2", "parents": ["x2"]},
            {"child": "y3", "parents": ["a3", "b3"]},
            {"child": "y4", "parents": ["a4", "b4"]},
        ], "all")
        check("scorer runs", r.returncode == 0)
        check("an edge learned in the OPPOSITE orientation counts as "
              "found - dependence is symmetric and the arrow comes "
              "from the column ordering",
              "recall 100%" in r.stdout)

        # nothing found
        r0 = score(td, [], "none")
        check("an empty edge set scores zero recall rather than "
              "crashing", "recall 0%" in r0.stdout)

        # the expected miss is excluded from the adjusted figure
        r1 = score(td, [
            {"child": "y1", "parents": ["x1"]},
            {"child": "y2", "parents": ["x2"]},
            {"child": "y3", "parents": ["a3", "b3"]},
        ], "noxor")
        check("a relationship that cannot be found by construction is "
              "NAMED as an expected miss, not silently forgiven",
              "MISSED (expected)" in r1.stdout)
        check("...and the adjusted recall excludes it, so a known "
              "blind spot does not read as a failure",
              "recall 75%" in r1.stdout
              and "excluding known blind spots 100%" in r1.stdout)

        # partial: one parent of a two-parent relationship
        r2 = score(td, [{"child": "y3", "parents": ["a3"]}],
                   "partial")
        check("finding ONE parent of a two-parent relationship is "
              "partial, not found - the interaction is the finding",
              "partial : y3" in r2.stdout)

        # noise
        r3 = score(td, [
            {"child": "y1", "parents": ["x1"]},
            {"child": "noise_00", "parents": ["y2"]},
        ], "noisy")
        check("an edge touching a noise column is counted as "
              "unambiguously wrong", "noise edges 1" in r3.stdout)
        check("an edge among real columns that was not planted is "
              "reported but NOT judged - the fixture has incidental "
              "structure and calling it a false positive would be "
              "wrong", "unplanted edges 0" in r3.stdout)
        r4 = score(td, [{"child": "x1", "parents": ["x2"]}],
                   "unplanted")
        check("...and such an edge is counted under unplanted",
              "unplanted edges 1" in r4.stdout
              and "noise edges 0" in r4.stdout)

        # recall by kind
        check("recall is broken out BY KIND, since a search that "
              "finds every linear relationship and no interaction "
              "has one capability and lacks another",
              "linear 1/1" in r1.stdout
              and "interaction 1/1" in r1.stdout)

        # alternative input shapes
        t = Path(td) / "truth.json"
        cm = Path(td) / "confirmed.json"
        cm.write_text(json.dumps({"report": {"confirmation": {
            "confirmed_edges": [{"child": "y1", "parents": ["x1"]}]},
            "edges": [{"child": "noise_00", "parents": ["y2"]}]}}),
            encoding="utf-8")
        r5 = run("scripts/score_discovery.py", "--truth", str(t),
                 "--model", str(cm))
        check("a CONFIRMED edge set is preferred over the raw one "
              "when both are present, so a confirmed model is scored "
              "on what it kept",
              "read as: confirmed" in r5.stdout
              and "noise edges 0" in r5.stdout)

        # guards
        r6 = run("scripts/score_discovery.py", "--truth",
                 str(Path(td) / "nope.json"), "--model", str(cm))
        check("a missing file fails readably",
              r6.returncode != 0
              and "not found" in (r6.stdout + r6.stderr)
              and "Traceback" not in (r6.stderr or ""))
        empty = Path(td) / "empty.json"
        empty.write_text(json.dumps({"relationships": []}),
                         encoding="utf-8")
        r7 = run("scripts/score_discovery.py", "--truth", str(empty),
                 "--model", str(cm))
        check("truth with nothing planted says so rather than "
              "reporting a vacuous 100%",
              r7.returncode != 0
              and "no planted relationships" in (r7.stdout + r7.stderr))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
