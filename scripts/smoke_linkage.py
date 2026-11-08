"""Checks for linkage risk (LK-1).

THE ELEMENT THIS CLOSES IS OUR OWN. 45 CFR 164.514(b)(1) asks
whether information could identify someone "alone OR IN
COMBINATION WITH OTHER REASONABLY AVAILABLE INFORMATION", and
`synthkit.determination` cited two adversaries for that element -
one seeing generated rows, one assuming the weights leaked - and
NEITHER holds external data. The package was asking the
regulation's question and answering a narrower one.

THE LOAD-BEARING CHECKS ARE ON THE THINGS THAT WOULD MAKE A
CLEAN NUMBER WORTHLESS, and there are three, which are three
different faults and so get three different verdicts:

  - the CONTROL did not beat the non-member arm: no control at
    all, and this project has recorded a reassuring 0.500 that
    was worth nothing for exactly that;
  - the control beat it but only barely: no POWER. A verbatim
    republish is undetectable under the caller's bands, so CLEAN
    would be a pass from a test that cannot fail;
  - the honest excess is NEGATIVE: the adversary did WORSE on
    members, so there is no advantage to express as a share -
    and the first cut printed "-818%" for this case.

AND THE BAR IS RELATIVE, because the absolute one was ABOVE ITS
OWN CEILING. Over six seeds a verbatim republish produces an
excess of only 0.026-0.104 sd, since a selection of at least
five rows averages the target's own value with four other
people's. A 0.10 absolute bar would have missed most republishes
- the same error as a campaign tier demanding 0.700 over a 0.698
ceiling.
"""

import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthkit import linkage as LK                     # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def people(n, seed):
    """A population where the sensitive value GENUINELY depends
    on the quasi-identifiers.

    That dependence is the point: it means population structure
    alone helps the adversary, so a raw accuracy would read as
    disclosure and only the EXCESS over a non-member arm can
    answer the question. A fixture without it would let a
    broken measurement look right.
    """
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        sex = int(r.integers(0, 2))
        site = int(r.integers(0, 4))
        age = float(r.normal(60, 12))
        hb = (13.5 - 1.2 * sex + 0.02 * (age - 60)
              + 0.3 * site + r.normal(0, 0.9))
        out.append({"sex": sex, "site": site, "age": age,
                    "hb": hb})
    return out


BANDS = {"age": [40, 50, 60, 70, 80]}
QUASI = ["sex", "site", "age"]


def main():
    mem, non = people(500, 1), people(500, 2)

    # ---- 1. THE FIXTURE CARRIES THE THING THE CHECK IS ABOUT
    import numpy as _np
    _hb = _np.array([r["hb"] for r in mem])
    _sex = _np.array([r["sex"] for r in mem])
    check("the fixture's sensitive column GENUINELY depends on "
          "the quasi-identifiers (mean hb {:.2f} for sex 0 "
          "against {:.2f} for sex 1) - so population structure "
          "alone helps the adversary, which is why a raw "
          "accuracy cannot answer the question and only the "
          "EXCESS over a non-member arm can"
          .format(float(_hb[_sex == 0].mean()),
                  float(_hb[_sex == 1].mean())),
          abs(float(_hb[_sex == 0].mean())
              - float(_hb[_sex == 1].mean())) > 0.5)

    # ---- 2. FIVE OUTCOMES, FIVE REASONS
    #
    # The fifth turned up on the REAL EXTRACT: at fine banding
    # the control fired cleanly at +0.082 and the HONEST arm
    # scored ZERO rows, because no member's quasi combination
    # had five matching rows in the released file. Before that
    # branch existed the reason given was "the control did not
    # beat the non-member arm" - which would send a reader to
    # fix a control that was working. Adding the branch then
    # RECLASSIFIED the tiny-release fixture here, which is the
    # same fault in miniature and had been mislabelled all
    # along.
    _honest = LK.with_control(mem, non, people(1500, 9), QUASI,
                              "hb", BANDS)
    _leaky = LK.with_control(mem, non, mem + people(100, 77),
                             QUASI, "hb", BANDS)
    _tiny = LK.with_control(mem, non, people(20, 5), QUASI,
                            "hb", BANDS)
    # NO CONTROL AT ALL: a single coarse quasi makes the
    # republish make the adversary WORSE off, so nothing is
    # caught.
    _coarse = LK.with_control(mem, non, people(1500, 9),
                              ["sex"], "hb", {})
    # NO POWER: the control is POSITIVE but tiny (+0.008 sd,
    # under the 0.02 floor), so a republish is barely detectable
    # and CLEAN would be a pass from a test that cannot fail.
    # This arrangement had to be SEARCHED FOR - the branch was
    # unreachable on the fixtures already here, and an
    # unreachable branch inside a tool built to catch
    # unreachable behaviour is a joke at its own expense.
    _nopower = LK.with_control(mem, non, people(1500, 9),
                               ["sex", "site"], "hb", {})
    # A MESSAGE MUST NOT FORMAT WHAT ITS MUTATION REMOVES. The
    # first version formatted `share_of_control` as a
    # percentage, and the mutation that disables banding makes
    # it None - so the suite CRASHED with a TypeError inside the
    # check's own label instead of reporting a failure. Exit 1
    # either way, but no FAIL line for a grep to find, which is
    # the shape this project records about crashed suites.
    def _pct(v):
        return "n/a" if v is None else "{:.0%}".format(v)
    check("an HONEST synthetic release reads CLEAN at {} of "
          "what a verbatim republish achieves, and a release "
          "that is MOSTLY THE MEMBERS reads LEAKS at {} - the "
          "measurement separates them on the same cohort with "
          "the same bands"
          .format(_pct(_honest["share_of_control"]),
                  _pct(_leaky["share_of_control"])),
          _honest["verdict"] == "CLEAN"
          and _leaky["verdict"] == "LEAKS"
          and _honest["share_of_control"] is not None
          and _leaky["share_of_control"] is not None
          and _leaky["share_of_control"]
          > _honest["share_of_control"] * 3)
    check("...a release too small to form selections reads "
          "UNREADABLE because the HONEST arm scored {} rows - "
          "nothing matched, so there was nothing to measure. "
          "This check used to assert 'the control did not beat "
          "the non-member arm', which was the wrong reason for "
          "this fixture all along and was only noticed when the "
          "real extract produced the same pair: a control "
          "firing cleanly at +0.082 beside an honest arm that "
          "scored zero"
          .format(_tiny["honest"]["members"]["n_scored"]),
          _tiny["verdict"] == "UNREADABLE"
          and _tiny["honest"]["members"]["n_scored"] == 0
          and "scored NO rows" in _tiny["why"]
          and "not safety" in _tiny["why"])
    check("...a single COARSE quasi-identifier also reads "
          "UNREADABLE, but for a DIFFERENT reason: there the "
          "republish makes the adversary WORSE off ({:+.4f} "
          "sd), so nothing was caught at all"
          .format(_coarse["control"]["excess_in_sd"] or 0.0),
          _coarse["verdict"] == "UNREADABLE"
          and _coarse["control_caught"] is False)
    check("...and a control that is POSITIVE but TINY ({:+.4f} "
          "sd, under the {:.2f} floor) reads UNREADABLE for a "
          "THIRD reason - no POWER: a republish is barely "
          "detectable under those bands, so CLEAN would be a "
          "pass from a test that cannot fail. This arrangement "
          "had to be SEARCHED FOR, because the branch was "
          "unreachable on the fixtures already here and an "
          "unreachable branch inside a tool built to catch "
          "unreachable behaviour is a joke at its own expense"
          .format(_nopower["control"]["excess_in_sd"] or 0.0,
                  LK.CONTROL_MIN),
          _nopower["verdict"] == "UNREADABLE"
          and _nopower["control_caught"] is True
          and _nopower["powered"] is False
          and "no POWER" in _nopower["why"])
    check("...and the FOUR outcomes keep FOUR REASONS rather "
          "than one word, because they take different actions - "
          "fix the control, use finer bands, turn the generator "
          "down, or report clean. Collapsing them would be the "
          "blocked-versus-unchanged fault",
          len(set((_honest["why"], _leaky["why"],
                   _tiny["why"], _nopower["why"]))) == 4
          and _coarse["why"] != _nopower["why"])

    # ---- 3. CAUGHT REQUIRES A POSITIVE CONTROL EXCESS
    #
    # The first cut tested only control > honest, so with a
    # negative honest excess a NEGATIVE control excess read as
    # caught - and the output said "the control CAUGHT the
    # republish but only by -0.0310 sd", a verdict contradicting
    # the number in its own sentence.
    check("`control_caught` requires the control's OWN excess to "
          "be positive, not merely larger than the honest arm's "
          "- a republish that makes the adversary WORSE off has "
          "not been caught, and the first cut printed a verdict "
          "contradicting its own number",
          _coarse["control_caught"] is False
          and (_coarse["control"]["excess_in_sd"] or 0) <= 0
          and _honest["control_caught"] is True
          and _honest["control"]["excess_in_sd"] > 0)

    # ---- 4. THE BAR IS RELATIVE, NOT ABOVE ITS OWN CEILING
    _ctl = []
    for s in range(6):
        m2, n2 = people(500, 100 + s), people(500, 200 + s)
        r = LK.with_control(m2, n2, people(1500, 900 + s),
                            QUASI, "hb", BANDS)
        _ctl.append(r["control"]["excess_in_sd"])
    check("a VERBATIM REPUBLISH reaches only {:.3f} to {:.3f} sd "
          "of excess over six seeds, because a selection of at "
          "least {} rows averages the target's own value with "
          "others' - so an ABSOLUTE bar of 0.10 would have been "
          "ABOVE ITS OWN CEILING, the same error as a campaign "
          "tier demanding 0.700 over a 0.698 ceiling"
          .format(min(_ctl), max(_ctl), LK.MIN_SELECTION),
          max(_ctl) < 0.20 and min(_ctl) > 0.0
          and LK.CONTROL_MIN < min(_ctl))
    check("...so the verdict is a FRACTION of the control, "
          "measured on the same cohort with the same bands, "
          "which self-calibrates: a coarser banding lowers both "
          "arms together",
          "share_of_control" in _honest
          and _honest["leak_share_bar"] == LK.LEAK_SHARE
          and 0.0 < LK.LEAK_SHARE < 1.0)
    _wins = 0
    for s in range(6):
        m2, n2 = people(500, 100 + s), people(500, 200 + s)
        r = LK.with_control(m2, n2, people(1500, 900 + s),
                            QUASI, "hb", BANDS)
        if r["control_caught"] is True:
            _wins += 1
    check("...and the control beats the honest arm on {} of 6 "
          "seeds - one reading of a control is a draw from a "
          "distribution, which is this project's oldest rule"
          .format(_wins), _wins == 6)

    # ---- 5. A THIN SELECTION IS SKIPPED, NOT SCORED
    check("a selection under {} rows is SKIPPED and COUNTED "
          "({} member rows here), not scored - with one or two "
          "matching rows the estimate is whichever rows were "
          "drawn, and an empty selection is a measurement that "
          "did not happen rather than a release that protected "
          "anybody"
          .format(LK.MIN_SELECTION,
                  _honest["honest"]["members"]["n_thin"]),
          LK.MIN_SELECTION >= 5
          and "n_thin" in _honest["honest"]["members"])
    check("...and a numeric quasi-identifier is BANDED, because "
          "an adversary holding an exact float matches almost "
          "nothing - which would read as SAFETY",
          LK.quasi_key({"age": 63.2}, ["age"], BANDS)
          == LK.quasi_key({"age": 67.9}, ["age"], BANDS)
          and LK.quasi_key({"age": 63.2}, ["age"], {})
          != LK.quasi_key({"age": 67.9}, ["age"], {}))
    check("...and a sensitive column with NO SPREAD is refused "
          "rather than divided by, because an error in units of "
          "its spread is undefined and a column everybody "
          "shares cannot single anybody out",
          LK.linkage_risk(
              [{"a": 1, "s": 5.0}] * 20,
              [{"a": 1, "s": 5.0}] * 20,
              [{"a": 1, "s": 5.0}] * 20,
              ["a"], "s").get("usable") is False)

    # ---- 6. THE DETERMINATION PACKAGE READS IT
    from synthkit import determination as DET
    _keys = [e["key"] for e in DET.ELEMENTS]
    check("the determination package carries LINKAGE as its own "
          "element, and `threat_model` now SAYS its two "
          "adversaries hold no external data - the package was "
          "asking the regulation's question and answering a "
          "narrower one",
          "linkage" in _keys
          and "NEITHER holds external data" in
          [e["evidence"] for e in DET.ELEMENTS
           if e["key"] == "threat_model"][0]
          and "ALONE OR IN COMBINATION" in
          [e["why"] for e in DET.ELEMENTS
           if e["key"] == "linkage"][0])
    check("...and linkage evidence counts only when its CONTROL "
          "CAUGHT the republish, so an UNREADABLE arm reads "
          "OWED rather than present",
          "linkage" not in DET.assess(
              linkage=_honest)["elements_owed"]
          and "linkage" in DET.assess(
              linkage=_tiny)["elements_owed"])

    # ---- 7. THE CLI, THROUGH THE PROCESS BOUNDARY
    def _frame(n, seed):
        r = np.random.default_rng(seed)
        pid = ["P{:04d}".format(i // 3) for i in range(n)]
        sex = r.integers(0, 2, n)
        site = r.integers(0, 4, n)
        age = r.normal(60, 12, n)
        hb = (13.5 - 1.2 * sex + 0.02 * (age - 60)
              + 0.3 * site + r.normal(0, 0.9, n))
        return pd.DataFrame({"person_id": pid, "sex": sex,
                             "site": site, "age": age.round(1),
                             "hb": hb.round(2)})
    with tempfile.TemporaryDirectory() as td:
        s_ = os.path.join(td, "src.csv")
        r_ = os.path.join(td, "rel.csv")
        o_ = os.path.join(td, "lk.json")
        _frame(1500, 1).to_csv(s_, index=False)
        _frame(1500, 9).to_csv(r_, index=False)
        base = [sys.executable,
                str(ROOT / "scripts" / "linkage_risk.py")]
        ok = subprocess.run(
            base + [s_, r_, "--quasi", "sex,site,age",
                    "--sensitive", "hb", "--band",
                    "age=40,50,60,70,80", "-o", o_],
            capture_output=True, text=True, cwd=str(ROOT))
        nocol = subprocess.run(
            base + [s_, r_, "--quasi", "nope",
                    "--sensitive", "hb"],
            capture_output=True, text=True, cwd=str(ROOT))
        badband = subprocess.run(
            base + [s_, r_, "--quasi", "sex", "--sensitive",
                    "hb", "--band", "age"],
            capture_output=True, text=True, cwd=str(ROOT))
        wrote = os.path.exists(o_)
        out = ok.stdout
    check("the CLI runs end to end and writes its JSON, with "
          "the reader's own note printed - a reader that does "
          "not say what it did is the silent-wrong-type fault "
          "waiting to happen",
          wrote and "as delimited text" in out
          and "VERDICT:" in out)
    check("...and it REFUSES a missing quasi-identifier column "
          "and a malformed --band by name, with exit 2 and no "
          "traceback - a feature that arrives by removing an "
          "error path has deleted a guard",
          nocol.returncode == 2
          and "STOPPED" in nocol.stderr
          and "Traceback" not in nocol.stderr
          and badband.returncode == 2
          and "Traceback" not in badband.stderr)
    check("...and it says the error is in units of the "
          "sensitive column's own SPREAD, so a number is "
          "comparable across columns instead of making a wide "
          "column look risky",
          "units of the sensitive column's own" in out
          and "population structure" in out)

    # ---- THE SCAN: FINDING THE WINDOW IS A SEARCH
    #
    # On the real extract the measurement failed at BOTH ends
    # and for different reasons - coarse bands left a verbatim
    # republish undetectable (+0.0108, under the floor), fine
    # bands left the honest arm with nothing scored - and
    # finding the window between them by hand costs a round trip
    # per guess to the machine that holds the extract.
    import linkage_risk as LR

    def _frame(n, seed, rows_per=1):
        r = np.random.default_rng(seed)
        m = n * rows_per
        sex = r.integers(0, 2, m)
        site = r.integers(0, 4, m)
        age = r.normal(60, 12, m)
        hb = (13.5 - 1.2 * sex + 0.02 * (age - 60)
              + 0.3 * site + r.normal(0, 0.9, m))
        return pd.DataFrame({
            "person_id": ["P{:05d}".format(i // rows_per)
                          for i in range(m)],
            # sex and site as STRINGS, so the ladder matches
            # them exactly and only bands what it can cut
            "sex": np.where(sex == 1, "F", "M"),
            "site": ["s{}".format(x) for x in site],
            "age": age.round(1), "hb": hb.round(2)})

    with tempfile.TemporaryDirectory() as _std:
        _s1 = os.path.join(_std, "src.csv")
        _r1 = os.path.join(_std, "rel.csv")
        _frame(1000, 1).to_csv(_s1, index=False)
        _frame(1500, 9).to_csv(_r1, index=False)
        _base = [sys.executable,
                 str(ROOT / "scripts" / "linkage_risk.py")]
        _found = subprocess.run(
            _base + [_s1, _r1, "--quasi", "sex,site,age",
                     "--sensitive", "hb", "--scan"],
            capture_output=True, text=True, cwd=str(ROOT))
        # ...and a cohort where NO arrangement works: three
        # rows per person, every quasi numeric so the ladder
        # bands them all, which is the shape the real extract
        # produced.
        _s2 = os.path.join(_std, "src2.csv")
        _r2 = os.path.join(_std, "rel2.csv")
        _f2 = _frame(1500, 1, rows_per=3)
        _f2["sex"] = (_f2["sex"] == "F").astype(int)
        _f2["site"] = _f2["site"].str[1:].astype(int)
        _f2.to_csv(_s2, index=False)
        _g2 = _frame(1500, 9, rows_per=3)
        _g2["sex"] = (_g2["sex"] == "F").astype(int)
        _g2["site"] = _g2["site"].str[1:].astype(int)
        _g2.to_csv(_r2, index=False)
        _none = subprocess.run(
            _base + [_s2, _r2, "--quasi", "sex,site,age",
                     "--sensitive", "hb", "--scan"],
            capture_output=True, text=True, cwd=str(ROOT))
        _allcat = subprocess.run(
            _base + [_s1, _r1, "--quasi", "sex,site",
                     "--sensitive", "hb", "--scan"],
            capture_output=True, text=True, cwd=str(ROOT))

    check("`--scan` walks arrangements coarse to fine and FINDS "
          "a usable one where the cohort allows it, exit {} - "
          "which is a search rather than a guess, and each "
          "guess by hand costs a round trip to the machine that "
          "holds the extract".format(_found.returncode),
          _found.returncode == 0
          and "MOST SENSITIVE USABLE" in _found.stdout
          and _found.stdout.count("CLEAN") >= 2)
    check("...and it picks the MOST SENSITIVE one, not the "
          "first. The control's excess IS the sensitivity - "
          "what a verbatim republish achieves under those bands "
          "- so the largest is the test most able to see a real "
          "leak. The first version returned the coarsest and "
          "so the weakest available",
          "16 bands" in _found.stdout
          and "LARGEST" in _found.stdout)
    check("...and where NO arrangement is usable it says so as "
          "a finding about the COHORT, exit {}, rather than "
          "reporting the least-bad number as clean"
          .format(_none.returncode),
          _none.returncode == 3
          and "NO ARRANGEMENT IN THIS LADDER IS USABLE"
          in _none.stdout
          and "not a" in _none.stdout
          and "stays OWED" in _none.stdout)
    check("...and both ends of the trade are PRINTED per "
          "arrangement, so a reader sees why it failed rather "
          "than a verdict - selections, honest rows scored, the "
          "control's excess and the honest excess",
          all(h in _none.stdout for h in
              ("selections", "scored", "control", "honest"))
          and _none.stdout.count("UNREADABLE") >= 3)
    check("...and it REFUSES when nothing can be banded, "
          "because the ladder only moves numeric columns and "
          "would otherwise report seven identical rows as a "
          "search",
          _allcat.returncode == 2
          and "at least one quasi" in _allcat.stderr
          and "Traceback" not in _allcat.stderr)
    # AND A SUCCESSFUL SCAN IS ITS OWN RECORD.
    with tempfile.TemporaryDirectory() as _rd:
        _s3 = os.path.join(_rd, "src.csv")
        _r3 = os.path.join(_rd, "rel.csv")
        _o3 = os.path.join(_rd, "scan.json")
        _frame(1000, 1).to_csv(_s3, index=False)
        _frame(1500, 9).to_csv(_r3, index=False)
        _rec = subprocess.run(
            [sys.executable,
             str(ROOT / "scripts" / "linkage_risk.py"),
             _s3, _r3, "--quasi", "sex,site,age",
             "--sensitive", "hb", "--scan", "-o", _o3],
            capture_output=True, text=True, cwd=str(ROOT))
        _doc = json.load(io.open(_o3)) \
            if os.path.exists(_o3) else {}
    from synthkit import determination as _DET2
    check("a successful scan NAMES the band edges it chose "
          "({}), because 're-run at that resolution' without "
          "saying at what edges costs the operator the search "
          "twice - and each guess on the real extract is a "
          "round trip to the machine that holds it"
          .format("--band" in _rec.stdout),
          "--band" in _rec.stdout
          and "age=" in _rec.stdout
          and bool(_doc.get("bands_used")))
    check("...and the scan FILE is itself the record: it "
          "carries the chosen arrangement's full result at the "
          "top level, so `determination --linkage` reads it "
          "directly rather than the operator running the whole "
          "measurement again just to produce a consumable file",
          _doc.get("control_caught") is True
          and (_doc.get("honest") or {}).get("excess_in_sd")
          is not None
          and "linkage" not in _DET2.assess(
              linkage=_doc, source_name=_s3)["elements_owed"])
    check("...and each scanned row keeps its own full result "
          "too, so a reader can see the arrangement that was "
          "NOT chosen and why",
          all(r.get("result") for r in _doc.get("scan", []))
          and len(_doc.get("scan", [])) >= 5)

    check("the ladder crosses the MIDDLE rather than jumping "
          "between the two values somebody tried by hand "
          "({}) - the real extract failed at 2-ish bands and at "
          "13-ish, and the window is between"
          .format(", ".join(str(b) for b in LR.BAND_STEPS)),
          len(LR.BAND_STEPS) >= 5
          and min(LR.BAND_STEPS) <= 2
          and max(LR.BAND_STEPS) >= 12
          and list(LR.BAND_STEPS) == sorted(LR.BAND_STEPS))

    print()
    if FAIL:
        print("{} of {} checks failed.".format(FAIL, PASS + FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
