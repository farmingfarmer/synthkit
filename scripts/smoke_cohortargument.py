"""Checks for cohort_argument - the page that argues a cohort is too
thin to publish, from two real runs.

The page's whole value is that its numbers are the RIGHT numbers. So
the checks are mostly about what it REFUSES to use: a column empty in
one source is an extraction difference, within-patient persistence
collapses on every cohort because the engine is row-level, and a
single-seed percentage is not a result. Each must appear in the
NOT-evidence section and nowhere else.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
for _p in (str(_root), str(_root / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PASS = 0
FAIL = 0


def check(name, ok):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print("FAIL: {}".format(name))


def _cohort(tmp, tag, patients, per, vocab, empty_cols=(),
            sd=20.0, seed=1):
    """A source and a run directory. `vocab` sets how many distinct
    tokens exist, so a wide vocabulary on few patients is the starved
    shape; `empty_cols` plants columns absent on EVERY row, which is
    the extraction difference the page must refuse to use."""
    r = np.random.RandomState(seed)
    rows = []
    for p in range(patients):
        pid = "P{:04d}".format(p)
        for _ in range(per):
            row = {"person_id": pid,
                   "codes": ";".join(sorted(set(
                       "t{:04d}".format(r.randint(vocab))
                       for _ in range(3)))),
                   "systolic": round(r.normal(120, sd), 1)}
            for c in empty_cols:
                row[c] = ""
            rows.append(row)
    d = Path(tmp) / tag
    d.mkdir(parents=True, exist_ok=True)
    sp = d / "src.csv"
    pd.DataFrame(rows).to_csv(sp, index=False)
    run = d / "run"
    run.mkdir(exist_ok=True)
    src = pd.read_csv(sp, dtype=str, keep_default_na=False)
    src.to_csv(run / "generated.csv", index=False)
    # a fidelity record shaped like a real one
    (run / "fidelity.json").write_text(json.dumps({
        "summary": {"pairs": 10, "pairs_sign_ok": 10,
                    "pairs_close": 9, "pairs_inverted": 0,
                    "coverage_ok": 3, "coverage_total": 3},
        "columns": [{"column": "systolic",
                     "coverage_source": 1.0,
                     "coverage_generated": 1.0,
                     "sd_source": sd,
                     "sd_generated": sd * (0.6 if patients > 200
                                           else 0.25),
                     "lag1_source": 0.5,
                     "lag1_generated": 0.15}]}),
        encoding="utf-8")
    return str(sp), str(run)


def main():
    import cohort_argument as C

    with tempfile.TemporaryDirectory() as tmp:
        # the THIN cohort also carries two columns absent on every
        # row - the extraction difference
        # SAME wide vocabulary on both, so the ONLY reason more
        # tokens clear the floor is more PATIENTS. The first cut gave
        # the big cohort a 40-token vocabulary, so only 40 could ever
        # clear and the thin one read HIGHER at 161 - the fixture
        # argued the opposite of the point.
        big_s, big_r = _cohort(tmp, "big", 400, 6, 900, seed=2)
        thin_s, thin_r = _cohort(tmp, "thin", 40, 60, 900,
                                 empty_cols=("gender",
                                             "year_of_birth"),
                                 seed=3)
        pb = C.profile("BIG", big_s, big_r, "person_id", 10)
        pt = C.profile("THIN", thin_s, thin_r, "person_id", 10)

        check("the patient-poor source publishes a smaller share of "
              "its data", pt["mass"] < pb["mass"])
        check("...and fewer of its tokens clear the floor, which is "
              "the mechanism stated as a count",
              pt["tokens"] and pb["tokens"]
              and pt["tokens"][0]["clear_k"]
              < pb["tokens"][0]["clear_k"])
        check("a column empty on EVERY source row is detected, "
              "because it has to be kept OUT of the argument",
              set(pt["empty_in_source"]) == {"gender",
                                             "year_of_birth"}
              and not pb["empty_in_source"])

        txt = C.render(pb, pt, "systolic", 10)

        check("coverage is argued BEFORE fidelity, so the "
              "denominator is read before any percentage",
              txt.index("How much of each source can be published")
              < txt.index("The same column, in both runs"))
        check("...and the reason that ordering matters is stated in "
              "words where a reader meets it",
              "HIGHER percentage while publishing far less"
              in txt)

        # THE NUMBER THE ARGUMENT TURNS ON
        check("spread RETAINED is computed and shown for both, as "
              "a share of source - 60% against 25% here",
              "spread RETAINED" in txt
              and "60%" in txt and "25%" in txt)
        check("...and the larger cohort is called BETTER, NOT GOOD, "
              "with its own loss named - overclaiming the control "
              "is the easiest thing to refute",
              "BETTER" in txt and "not good" in txt
              and "40%" in txt)

        # WHAT IT MUST REFUSE
        ne = txt[txt.index("NOT evidence about cohort size"):]
        check("the empty columns appear ONLY in the NOT-evidence "
              "section, named, with the reason",
              "gender" in ne
              and "what was pulled, not what the floor did" in ne
              and "gender" not in txt[:txt.index(
                  "NOT evidence about cohort size")])
        check("persistence is refused too, and the reason given is "
              "the ENGINE rather than the cohort - it would "
              "collapse on a million patients",
              "ROW-LEVEL" in ne
              and "million" in ne)
        check("...and the page states that no single-seed "
              "percentage is quoted, since a sweep moved such a "
              "reading by ten points",
              "No single-seed fidelity percentage is quoted" in ne)

        # the inverted row must be pointed at, not buried
        check("INVERTED is called out as the row to read first - an "
              "inverted relationship reads as a finding",
              "INVERTED relationship reads as a finding" in txt)
        check("the remedy names PEOPLE rather than rows, and says "
              "why duplicating rows cannot help",
              "More PEOPLE, not more rows" in txt
              and "still one person" in txt)

        # ---- A WINDOWS PATH MUST SURVIVE ARGUMENT PARSING ----
        # The first cut took NAME=SRC:RUN and split on the FIRST
        # colon, so a path beginning "C:" parsed the drive letter as
        # the whole source. Every check passed anyway, because every
        # fixture used POSIX temp paths with no drive letter - the
        # tool exists to run on the machine that holds the data, and
        # that machine is Windows. Cohorts are named with their own
        # flags now and nothing is parsed out of a path; this asserts
        # a drive-lettered path reaches the file check INTACT, by
        # requiring the refusal to name the whole path back.
        import io as _io
        import contextlib as _cl
        _buf = _io.StringIO()
        with _cl.redirect_stdout(_buf):
            _rc = C.main(["--name", "BIG",
                          "--src", r"C:\Users\x\dev\big.csv",
                          "--run", r"C:\Users\x\dev\bigrun",
                          "--name", "THIN",
                          "--src", r"C:\Users\x\dev\thin.csv",
                          "--run", r"C:\Users\x\dev\thinrun"])
        _said = _buf.getvalue()
        check("a drive-lettered Windows path reaches the file check "
              "WHOLE - the refusal names the full path back, where "
              "the old colon-spec would have reported just 'C'",
              _rc == 2
              and r"C:\Users\x\dev\big.csv" in _said
              and "no source file for BIG" in _said)
        check("...and an unknown argument is refused by NAME rather "
              "than silently ignored, which is how a flag that "
              "never reaches the program looks identical to a "
              "broken one",
              C.main(["--nope", "1"]) == 2)

        longest = max(len(line) for line in txt.splitlines())
        check("no row overflows the table it is printed in "
              "({} chars)".format(longest), longest <= 120)

    print("\n{} checks, {} failed".format(PASS + FAIL, FAIL))
    if FAIL:
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
