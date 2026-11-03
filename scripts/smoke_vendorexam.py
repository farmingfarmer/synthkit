"""Smoke: route A end to end - plant an answer, grade exactly.

WHY THIS EXISTS. `dialback` names three routes to a vendor's
value and recommends ROUTE A - we hold the model, or they score
our file - because it has no approximation error at all. Route A
was the only one with NO TOOLING: it existed as advice. What was
missing was never the scoring, it was the ANSWER, and
`semisynth.plant_frame` had been able to supply one since it was
written while being called by nothing but its own smoke test.

THE THREE NUMBERS, and what each is for:
    CEILING  what a model achieves HAVING SEEN our truth.
             Nothing can legitimately beat it.
    FLOOR    what a model achieves having trained on the
             RELEASED file. This is what the dials bought.
    VENDOR   what came back, placed between the two.

AND ABOVE THE CEILING IS A BROKEN EXAM, NOT A GOOD SCORE. The
first cut printed "they are at the ceiling" for a vendor sitting
0.344 ABOVE it - a verdict contradicting the number two lines
above. Either the planted answer reached them or their scores
were in-sample; both invalidate the result, and it has to be
loud before anybody quotes it.
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
VE = str(ROOT / "scripts" / "vendor_exam.py")


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def fixture(path, npat=260, seed=5):
    """Enough rows and a strong enough signal to SEPARATE.

    THE FIRST VERSION COULD NOT REACH THE THING THE CHECKS
    ARE ABOUT. At 110 patients the planted effects left a
    ceiling of 0.502 against a floor of 0.539 - both AUC
    noise around chance, with the order between them a coin
    flip - so the separation checks failed on a healthy
    tool. An exam fixture has to produce an exam."""
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
            rows.append(dict(person_id=pid, age_at_visit=age,
                             span_days=round(span, 2),
                             lab=round(lab, 2),
                             other=round(float(
                                 rng.normal(50, 9)), 2)))
    pd.DataFrame(rows).to_csv(path, index=False)


def run(*args, timeout=1800):
    r = subprocess.run([PY, "-u"] + [str(a) for a in args],
                       capture_output=True, text=True,
                       timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main():
    from fidelity_deck import read_frame
    from dialback import _design
    from synthkit.semisynth import plant_frame, verify_frame
    from sklearn.ensemble import HistGradientBoostingClassifier

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.csv"
        fixture(src)
        rd = td / "r"
        rc, log = run(LC, "csv", src, "--group-by", "person_id",
                      "--k", "10", "--seeds", "0", "--out", rd,
                      "--poison", "age_at_visit~span_days=-0.5")
        check("a dialed run produces the pair the exam needs",
              rc == 0 and (rd / "key.csv").exists()
              and (rd / "blueprint.json").exists())

        # ---- THE CONTRACT, ASSERTED AGAINST A LIVE CALL.
        # The first cut of vendor_exam read chk["recovered"]
        # with "requested"/"recovered" fields and printed
        # NOTHING on a healthy exam, because verify_frame
        # returns `effects` / `requested_in_sds` /
        # `recovered_in_sds`. That is this repository's recorded
        # wrong-key fault - a reader and its fixture agreeing
        # with each other and both disagreeing with the function
        # actually being called. Asserted live, so a rename
        # breaks THIS rather than somebody's report.
        k = read_frame(rd / "key.csv")
        bp = json.loads((rd / "blueprint.json").read_text(
            encoding="utf-8"))
        pf, prec = plant_frame(k, bp,
                               {"age_at_visit": 1.4, "lab": -0.9},
                               name="exam_outcome",
                               prevalence=0.25, seed=0)
        vf = verify_frame(pf, prec)
        check("verify_frame's contract is `effects` with "
              "`requested_in_sds` and `recovered_in_sds` - "
              "asserted LIVE, because guessing these names "
              "printed an empty table on a healthy exam",
              isinstance(vf.get("effects"), list)
              and vf["effects"]
              and "requested_in_sds" in vf["effects"][0]
              and "recovered_in_sds" in vf["effects"][0])

        # ---- set the exam
        rc, out = run(VE, "set", rd, "--effects",
                      "age_at_visit=+1.4,lab=-0.9",
                      "--prevalence", "0.25")
        check("`set` plants the answer and writes all three "
              "files - the key we keep, the released file we "
              "send, and the answer key",
              rc == 0
              and (rd / "key_planted.csv").exists()
              and (rd / "released_planted.csv").exists()
              and (rd / "answer_key.json").exists())
        ak = json.loads((rd / "answer_key.json").read_text(
            encoding="utf-8"))
        check("...and the prevalence lands near what was asked, "
              "which is the intercept solve doing its job: "
              "asked {:.2f} got {:.3f}".format(
                  ak["prevalence_requested"],
                  ak["prevalence_achieved"]),
              abs(ak["prevalence_achieved"]
                  - ak["prevalence_requested"]) < 0.05)
        check("...and the planted effects are RECOVERED near "
              "the values asked for, printed requested beside "
              "recovered",
              "asked +1.40" in out and "recovered" in out
              and all(abs(float(r["recovered_in_sds"])
                          - float(r["requested_in_sds"])) < 0.25
                      for r in (ak.get("recovered") or [])))
        check("...and the CEILING sits above the FLOOR, so the "
              "exam can separate a vendor from a model that "
              "learned only from what we re-pointed",
              float(ak["ceiling"]) > float(ak["floor"]) + 0.02)
        check("...and the released file carries the outcome "
              "RE-POINTED rather than omitted - the measured "
              "rule is dial the outcome, and an absent label "
              "is not a lie, it is a missing column",
              "exam_outcome" in read_frame(
                  rd / "released_planted.csv").columns)

        kp = read_frame(rd / "key_planted.csv")
        rp = read_frame(rd / "released_planted.csv")
        Xk, _ = _design(kp, "exam_outcome", set())
        Xr, _ = _design(rp, "exam_outcome", set())
        yk = pd.to_numeric(kp["exam_outcome"],
                           errors="coerce").to_numpy()
        yr = pd.to_numeric(rp["exam_outcome"],
                           errors="coerce").to_numpy()

        # a vendor who TRAINED on what we sent
        m = HistGradientBoostingClassifier(
            max_iter=120, random_state=1).fit(Xr, yr)
        pd.DataFrame({"score": m.predict_proba(Xk)[:, 1]}).to_csv(
            td / "v_trained.csv", index=False)
        rc, out = run(VE, "grade", rd, "--scores",
                      td / "v_trained.csv")
        vend = float([l for l in out.splitlines()
                      if "VENDOR" in l][0].split()[-1])
        check("a vendor who TRAINED on the released file grades "
              "near the floor, which is the number the dials "
              "exist to produce: got {:.3f} against a floor of "
              "{:.3f} and a ceiling of {:.3f}".format(
                  vend, ak["floor"], ak["ceiling"]),
              rc == 0 and vend < float(ak["ceiling"]) - 0.05)

        # a vendor whose scores are IN-SAMPLE - above the ceiling
        m2 = HistGradientBoostingClassifier(
            max_iter=120, random_state=1).fit(Xk, yk)
        pd.DataFrame({"score": m2.predict_proba(Xk)[:, 1]}).to_csv(
            td / "v_leaky.csv", index=False)
        rc, out = run(VE, "grade", rd, "--scores",
                      td / "v_leaky.csv")
        check("a vendor scoring ABOVE the ceiling is called a "
              "BROKEN EXAM and not a good result - nothing "
              "legitimately beats a model that saw the answer",
              rc == 0 and "ABOVE THE CEILING" in out
              and "broken exam" in out)
        check("...and the reason is named so somebody can act "
              "on it: they saw the truth, or the scores were "
              "in-sample",
              "in-sample" in out)
        check("...and it does NOT also print the at-the-ceiling "
              "praise - a verdict must not contradict the "
              "number printed above it",
              "They are at the ceiling" not in out)

        rc, out = run(VE, "grade", rd, "--scores",
                      td / "v_trained.csv", "--bar", "0.99")
        check("a bar above the exam's own ceiling is called a "
              "MISCALIBRATED BAR rather than a failed vendor",
              rc == 0 and "MISCALIBRATED BAR" in out)

        rc, out = run(VE, "grade", rd, "--scores",
                      td / "v_trained.csv", "--bar", "0.01")
        check("...and a reachable bar is judged normally",
              rc == 0 and "MET" in out
              and "MISCALIBRATED" not in out)

        # refusals
        rc, out = run(VE, "grade", td, "--scores",
                      td / "v_trained.csv")
        check("`grade` refuses a directory with no answer key, "
              "by name - a score is only gradeable against a "
              "truth somebody planted",
              rc != 0 and "no answer_key.json" in out)
        pd.DataFrame({"score": [1.0, 2.0]}).to_csv(
            td / "short.csv", index=False)
        rc, out = run(VE, "grade", rd, "--scores",
                      td / "short.csv")
        check("...and refuses a return whose row count does not "
              "match the exam",
              rc != 0 and "one per row" in out)
        # ---- AN INCOHERENT EXAM MUST BE REFUSED, NOT
        #      WARNED ABOUT AND WRITTEN. Observed on a thin
        #      fixture: ceiling 0.502 against floor 0.539 -
        #      the best achievable score sitting UNDER what a
        #      re-pointed file gives you, which is not a weak
        #      exam but a meaningless one. The first cut
        #      printed a warning and wrote the answer key
        #      anyway, producing an artifact that looks
        #      usable and grades nobody.
        #
        #      Reached by planting on `other`, which is pure
        #      noise, so both numbers are chance and their
        #      order is a coin flip. Seed 1 is PINNED because
        #      it is one that lands the wrong way round - the
        #      condition cannot be forced, only found.
        import shutil as _sh
        inc = td / "inc"
        _sh.copytree(rd, inc)
        for _f in ("answer_key.json", "key_planted.csv",
                   "released_planted.csv"):
            if (inc / _f).exists():
                (inc / _f).unlink()
        rc, out = run(VE, "set", inc, "--effects",
                      "other=+0.2", "--seed", "1")
        check("an exam whose CEILING sits at or below its "
              "FLOOR is REFUSED and writes nothing - that is "
              "not a weak exam, it is an incoherent one",
              rc != 0 and "REFUSED: ceiling" in out
              and not (inc / "answer_key.json").exists())
        check("...and it says what to do about it rather "
              "than only naming the fault",
              "Plant harder or bring more rows" in out)

        rc, out = run(VE, "set", rd, "--effects", "nope=+0.9")
        check("...and `set` refuses an exam with nothing "
              "plantable, NAMING the effect it could not "
              "place rather than writing an answer key "
              "with no answer in it",
              rc != 0 and "no effect could be planted" in out
              and "nope" in out)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
