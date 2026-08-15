"""Smoke: what a broken loop actually cost, said accurately.

THE FAULT THIS REPRODUCES, from the 800-patient run. A cycle forces
the sampler to draw something first, and whatever is drawn first loses
the parents that are not drawn yet. findings.txt listed those parents
under

    the ones listed here contribute nothing to it in the generated data

and one of the lines under that heading was

    age_at_visit lost year_of_birth   (kept admitted_from)

while fidelity.json recorded `parents_lost: []` for that same drop,
because pair repair had reconnected it and the dependence reaches the
output the other way round. Two shipped files disagreeing about one
relationship, which is the signal this project treats as the finding.

The fixture is that loop with those names and those skills, and it
reproduces the line exactly: `age_at_visit lost year_of_birth (kept
admitted_from)`, `parents_lost []`, `reconnected [year_of_birth]`.

THE SPLIT IS PER PARENT, not per relationship, because one drop can
lose one parent and keep another. And it is checked in BOTH
directions - a report that called every removed parent "still
connected" would be as wrong as the one that called them all lost, so
a second fixture plants a parent that repair genuinely cannot
reconnect and asserts it is named as gone.

WHAT IS NOT FIXED HERE, and was measured rather than assumed.

Which column gets drawn first is still decided by the order the
blueprint happens to list its columns, not by evidence. That is a real
defect: the same graph with the same measured skills gives a different
answer depending on how the file was written, and generate.py's own
docstring says the arrow is chosen by out-of-sample skill.

The obvious repair - cut where the immediate skill loss is smallest -
was written, and then measured with `scripts/pair_fidelity_sweep.py`
against the tidy fixture at 300 patients over seeds 11, 23, 37, 53:

    sign kept   16.8 (16-17) against 17.2 (17-18) for the old order
    close       15.8 (15-16) against 16.2 (15-17)
    INVERTED     0-1, on two seeds, against 0-0 - never, before

Every per-seed delta was zero or negative, and it introduced SIGN
INVERSIONS on two of four seeds where the arbitrary order produced
none. A greedy cheapest-cut is locally optimal and globally worse: on
seed 37 it trimmed 12 relationships where the old order trimmed 10.
So the ordering was left as it is and the finding recorded. Anyone
retrying it should run that sweep first, and should expect to beat
0 inversions rather than to improve the mean.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_discovery import _will_drop_full, render         # noqa: E402
from synthkit.blueprint import resolve                    # noqa: E402
from synthkit.generate import _order                      # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def rel(child, imps, skill):
    return {"child": child, "parents": [c for c, _ in imps],
            "evidence": {"skill_out_of_sample": skill,
                         "importance": dict(imps),
                         "effect": {}},
            "dials": {"strength": None}}


def blueprint(col_order):
    """The age_at_visit / year_of_birth loop off the real run.

    Skills and importances are the ones that run reported: the 98%
    edge carries year_of_birth at high importance and admitted_from at
    low, and the back edge that closes the loop is weaker."""
    cols = dict(
        (c, {"kind": "numeric", "level": "visit", "coverage": 1.0,
             "marginal": {"type": "quantiles", "q": [0.0, 1.0],
                          "v": [0.0, 1.0]},
             "dials": {}})
        for c in col_order)
    return {
        "blueprint_version": 1,
        "columns": cols,
        "patients": {"count": 800, "rows": 55428,
                     "visits": {"mean": 69.3}},
        "relationships": [
            rel("age_at_visit",
                [("year_of_birth", 0.90), ("admitted_from", 0.10)],
                0.98),
            rel("year_of_birth",
                [("gender", 0.30), ("age_at_visit", 0.70)], 0.55),
        ],
    }


AGE_FIRST = ["age_at_visit", "admitted_from", "gender",
             "year_of_birth"]


def main():
    bp = blueprint(AGE_FIRST)
    _o, _p, dropped, _r, _d = _order(resolve(bp))

    # ---- THE FIXTURE REPRODUCES THE REAL LINE ---------------------
    check("THE FIXTURE REPRODUCES THE CONDITION: the graph is a real "
          "cycle, so a parent genuinely has to be given up - a "
          "fixture without one would pass whatever the report said",
          len(dropped) == 1 and dropped[0].get("partial"))
    d = dropped[0]
    check("...and it reproduces the exact drop the 800-patient run "
          "reported: age_at_visit lost year_of_birth, keeping "
          "admitted_from",
          d["child"] == "age_at_visit"
          and d["parents"] == ["year_of_birth"]
          and d.get("kept_parents") == ["admitted_from"])
    check("THE TWO FILES DISAGREED HERE: the parent findings.txt "
          "called lost is recorded as reconnected, because the pair "
          "is generated the other way round",
          d.get("parents_lost") == []
          and d.get("reconnected") == ["year_of_birth"])

    # ---- THE REPORT NOW AGREES WITH THE FIELD ---------------------
    text = render(bp)
    section = text[text.index("WHAT WILL NOT REACH THE GENERATED DATA"):]
    check("findings.txt no longer prints the line that contradicted "
          "fidelity.json",
          "age_at_visit lost year_of_birth" not in section)
    check("...and it says where the pair DID end up, in the reader's "
          "own terms rather than leaving them to infer it",
          "generated the other way round" in section
          or "kept as a direct edge" in section)
    check("the relationship is still NAMED - reporting less is not "
          "the fix, reporting it accurately is",
          "age_at_visit <- year_of_birth" in section)

    # ---- THE TOTALS A READER CAN CHECK ---------------------------
    m = re.search(r"(\d+) had SOME PARENTS REMOVED to break a loop, "
                  r"(\d+) parent", section)
    check("the section counts PARENTS, not only relationships - one "
          "relationship can lose two, and a reader cannot check a "
          "list against a count that measures something else",
          m is not None)
    if m:
        n_par = int(m.group(2))
        m2 = re.search(r"below and (\d+) \+ (\d+) adds back to (\d+)",
                       section)
        check("...and still-connected plus gone adds back to that "
              "total, the rule that exists because ten of twelve "
              "drops were once invisible",
              m2 is not None
              and int(m2.group(1)) + int(m2.group(2)) == n_par
              and int(m2.group(3)) == n_par)

    # ---- AND A GENUINE LOSS IS STILL CALLED ONE ------------------
    # The opposite error. A split that reported everything as
    # still-connected would pass every check above and tell the
    # reader nothing was ever lost. Here `p` loses `r`, and no repair
    # can reconnect that pair: repair needs a claim in the other
    # direction to borrow r's curve from, and nothing ever said
    # `r <- p`.
    lonely = blueprint(AGE_FIRST)
    for extra in ("p", "r", "z"):
        lonely["columns"][extra] = dict(lonely["columns"]["gender"])
    lonely["relationships"] = [
        rel("age_at_visit", [("z", 0.10), ("p", 0.90)], 0.98),
        rel("p", [("r", 0.50), ("gender", 0.50)], 0.50),
        rel("r", [("age_at_visit", 0.90)], 0.60),
    ]
    d2, _rec2 = _will_drop_full(lonely)
    trimmed2 = [x for x in d2 if x.get("partial")]
    really_lost = sorted(p for x in trimmed2
                         for p in (x.get("parents_lost") or []))
    t2 = render(lonely)
    s2 = t2[t2.index("WHAT WILL NOT REACH THE GENERATED DATA"):]
    check("THE OPPOSITE FIXTURE REPRODUCES A REAL LOSS: parents no "
          "repair can reconnect, so the next check is testing "
          "something - {}".format(really_lost),
          really_lost == ["p", "r"])
    check("...and they are reported as GONE, in their own section",
          ("ARE GONE" in s2 or "IS GONE" in s2) and "p lost r" in s2)
    check("every parent named as lost is one the sampler actually "
          "removed, in both directions of the split",
          all(p in (x.get("parents") or [])
              for x in trimmed2
              for p in (x.get("parents_lost") or [])))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
