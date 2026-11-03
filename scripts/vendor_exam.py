"""ROUTE A, end to end: plant an answer, release a lie, grade exactly.

THE ROUTE THIS SERVES, and why it is the recommended one.
`dialback.py` names three ways to get value back from a vendor.
Route B fits a surrogate to returned scores and its error is a
property of the column you dialed - 0.00 sd on a coarse one,
0.25 on a fine one, measured. Route C restates an aggregate.
ROUTE A has no error at all: we hold the model, or they score
our own file, and the number we read is the number their model
computes. It was also, until now, the only route with NO
TOOLING - it existed as advice in a docstring.

WHAT WAS MISSING WAS NOT THE SCORING. It was the ANSWER. A
vendor's score is only gradeable against a truth somebody
knows, and nobody knows the truth in real data - which is the
reason `semisynth.plant_frame` exists and the reason it has
been called by nothing but its own smoke test since it was
written. This joins it to the key/released pair:

    key.csv        faithful covariates + the TRUE outcome,
                   planted by us, never shared
    generated.csv  the same faithful covariates + the outcome
                   RE-POINTED, which is what leaves
    answer_key.json  what we planted, what it recovered, and
                   the arrangement between the two outcomes

A vendor who TRAINS on the released file fits an outcome we
invented. A vendor who SCORES our rows reads covariates that
are untouched, so their answer is about the real structure -
and we can grade it, because we planted the truth.

THE THREE NUMBERS, in report_card's vocabulary so the two
halves of this repository say the same words:

    CEILING   what a model achieves against our truth having
              SEEN our truth. Nobody can beat it; a bar above
              it is a miscalibrated bar, not a failed vendor.
    FLOOR     what a model achieves having trained on the
              RELEASED file. This is what the dials bought.
    VENDOR    what they actually returned, placed between.

DIAL THE OUTCOME, NOT THE COVARIATES - measured, and the whole
reason this works. A corrupted label leaves a trainer nothing
while leaving a scorer everything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from dialback import _design, _model, load_pair      # noqa: E402
from fidelity_deck import read_frame                 # noqa: E402
from synthkit.semisynth import plant_frame, verify_frame  # noqa: E402

OUTCOME = "exam_outcome"


def _auc(y, s):
    """Rank AUC without sklearn's import cost, ties handled."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0)
                 / (pos.sum() * neg.sum()))


def _split(n, seed=0):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * 0.7)
    return idx[:cut], idx[cut:]


def set_exam(run, effects, prevalence=0.25, seed=0):
    run = Path(run)
    k, g, c, err = load_pair(run)
    if err:
        print("STOP: " + err)
        return 1
    bp_path = run / "blueprint.json"
    if not bp_path.exists():
        print("STOP: no blueprint.json in {} - plant_frame "
              "standardizes against the published centres and "
              "spreads, so an effect declared in sds needs "
              "it.".format(run))
        return 1
    bp = json.loads(bp_path.read_text(encoding="utf-8"))

    try:
        planted, rec = plant_frame(k, bp, effects,
                                   name=OUTCOME,
                                   prevalence=prevalence,
                                   seed=seed)
    except ValueError as e:
        print("STOP: {}".format(e))
        return 1
    if rec.get("refused"):
        for r in rec["refused"]:
            print("  refused {}: {}".format(r.get("effect"),
                                            r.get("why")))
    kept = [u["effect"] for u in (rec.get("calibration") or [])]
    if not kept:
        print("STOP: nothing plantable in {} - an exam with no "
              "planted effect has no answer to grade "
              "against.".format(effects))
        return 1

    y = pd.to_numeric(planted[OUTCOME], errors="coerce")
    print("exam planted on {}".format(run))
    print("  effects kept: {}".format(", ".join(kept)))
    print("  prevalence    requested {:.3f}   achieved {:.3f}"
          .format(prevalence, float(y.mean())))
    # THE KEYS ARE verify_frame's, NOT ONES THAT SOUND RIGHT.
    # The first cut of this read chk["recovered"] with
    # "requested"/"recovered" fields and printed NOTHING on a
    # healthy exam - the same wrong-key fault this repository
    # already records for jointcheck, where a renderer and its
    # fixture agreed with each other and both disagreed with
    # the function being called. The contract is `effects`,
    # `requested_in_sds`, `recovered_in_sds`, and the suite
    # asserts it against a LIVE call so a rename breaks the
    # checks rather than somebody's report.
    chk = verify_frame(planted, rec)
    for row in (chk.get("effects") or []):
        print("    {:<22} asked {:+.2f}   recovered {:+.2f}"
              .format(row.get("effect", "?"),
                      float(row.get("requested_in_sds", 0)),
                      float(row.get("recovered_in_sds", 0))))

    # THE OUTCOME TRAVELS TO THE RELEASED FILE AND IS THEN
    # RE-POINTED. The released rows are the key rows with some
    # columns rearranged, so the outcome copies across
    # row-for-row - and then we shuffle it, which is the dial
    # applied where the measurement says it belongs.
    rng = np.random.RandomState(seed + 991)
    perm = rng.permutation(len(planted))
    rel = g.copy()
    rel[OUTCOME] = y.to_numpy()[perm]

    # CEILING and FLOOR, both scored on the SAME held-out key
    # rows against the SAME truth. Anything else is comparing
    # two different questions.
    Xk, _ = _design(planted, OUTCOME, set())
    Xr, _ = _design(rel, OUTCOME, set())
    tr, te = _split(len(planted), seed)
    truth = y.to_numpy()[te]
    from sklearn.ensemble import HistGradientBoostingClassifier

    def _fit(X, lab):
        m = HistGradientBoostingClassifier(max_iter=120,
                                           random_state=0)
        m.fit(X.iloc[tr], lab[tr])
        return _auc(truth, m.predict_proba(Xk.iloc[te])[:, 1])

    ceiling = _fit(Xk, y.to_numpy())
    floor = _fit(Xr, rel[OUTCOME].to_numpy())

    # AN EXAM WHOSE CEILING SITS AT OR BELOW ITS FLOOR IS NOT
    # AN EXAM, and the first cut WARNED and wrote the answer
    # key anyway - producing an artifact that looks usable and
    # grades nobody. On a thin fixture it read ceiling 0.502
    # against floor 0.539, which is not merely weak but
    # INCOHERENT: the best possible score below the score you
    # get from the file we deliberately re-pointed. Both
    # numbers are AUC noise around chance, and the ordering
    # between them is a coin flip. Refuse, and say what to do.
    if ceiling <= floor:
        print("")
        print("  REFUSED: ceiling {:.3f} is AT OR BELOW floor "
              "{:.3f}. That is not a weak exam, it is an "
              "incoherent one - the best achievable score "
              "cannot sit under what a re-pointed file gives "
              "you, so both numbers are noise around chance "
              "and their order is a coin flip. Nothing was "
              "written. Either the planted effects are too "
              "weak for these covariates to carry, or there "
              "are too few rows for an AUC to settle. Plant "
              "harder or bring more rows.".format(
                  ceiling, floor))
        return 1

    planted.to_csv(run / "key_planted.csv", index=False)
    rel.to_csv(run / "released_planted.csv", index=False)
    ak = {
        "what_this_is":
            "the answer to the exam. The vendor sees "
            "released_planted.csv, whose outcome is a "
            "rearrangement we invented. The truth is in "
            "key_planted.csv and never leaves.",
        "outcome": OUTCOME,
        "effects_requested": effects,
        "prevalence_requested": prevalence,
        "prevalence_achieved": round(float(y.mean()), 4),
        "recovered": chk.get("effects"),
        "refused": rec.get("refused"),
        "outcome_permutation": [int(x) for x in perm],
        "ceiling": round(float(ceiling), 4),
        "floor": round(float(floor), 4),
        "seed": seed,
    }
    (run / "answer_key.json").write_text(
        json.dumps(ak, indent=2), encoding="utf-8")

    print("")
    print("  CEILING {:.3f}   a model that SAW our truth. Nobody "
          "beats it, and a bar above it is a miscalibrated bar "
          "rather than a failed vendor.".format(ceiling))
    print("  FLOOR   {:.3f}   a model trained on the RELEASED "
          "file, scored on our truth. This is what the dials "
          "bought.".format(floor))
    print("  room between them: {:.3f}".format(ceiling - floor))
    if ceiling - floor < 0.05:
        print("  THE EXAM BARELY SEPARATES: ceiling and floor "
              "are within 0.05, so a vendor and a model that "
              "learned only from the re-pointed file score "
              "almost alike. Written, because it is coherent - "
              "but read a grade off it with that room in mind.")
    print("")
    print("  wrote key_planted.csv (KEEP), released_planted.csv "
          "(this is what you send) and answer_key.json (KEEP).")
    return 0


def grade(run, scores_csv, score_col=None, bar=None):
    run = Path(run)
    ak_path = run / "answer_key.json"
    if not ak_path.exists():
        print("STOP: no answer_key.json in {} - run `set` "
              "first. A vendor score is only gradeable against "
              "a truth somebody planted.".format(run))
        return 1
    ak = json.loads(ak_path.read_text(encoding="utf-8"))
    kp = read_frame(run / "key_planted.csv")
    sc = read_frame(scores_csv)
    if len(sc) != len(kp):
        print("STOP: {} has {} rows, the exam has {}. The "
              "returned scores must be one per row, in the "
              "order of the file you sent.".format(
                  scores_csv, len(sc), len(kp)))
        return 1
    if score_col is None:
        num = [x for x in sc.columns
               if pd.to_numeric(sc[x],
                                errors="coerce").notna().mean()
               > 0.9]
        if len(num) != 1:
            print("STOP: say which column holds the score with "
                  "--score-col. Numeric: {}".format(
                      ", ".join(num) or "(none)"))
            return 1
        score_col = num[0]
    y = pd.to_numeric(kp[ak["outcome"]], errors="coerce").to_numpy()
    s = pd.to_numeric(sc[score_col], errors="coerce").to_numpy()
    _, te = _split(len(kp), int(ak.get("seed", 0)))
    vend = _auc(y[te], s[te])
    ceiling = float(ak["ceiling"])
    floor = float(ak["floor"])
    print("grading {} against the planted answer".format(scores_csv))
    print("")
    print("  CEILING {:.3f}".format(ceiling))
    print("  VENDOR  {:.3f}".format(vend))
    print("  FLOOR   {:.3f}".format(floor))
    print("  gap to ceiling {:+.3f}".format(vend - ceiling))
    print("")
    if bar is not None:
        bar = float(bar)
        if bar > ceiling:
            print("  MISCALIBRATED BAR: {:.3f} sits ABOVE this "
                  "exam's own ceiling of {:.3f}, so no solver "
                  "on earth clears it. That is a fault in the "
                  "bar, not in the vendor.".format(bar, ceiling))
        else:
            print("  bar {:.3f}: {}".format(
                bar, "MET" if vend >= bar else "NOT MET"))
    # ABOVE THE CEILING IS NOT A GOOD SCORE, IT IS A BROKEN
    # EXAM - and the first cut of this printed "they are at
    # the ceiling" for a vendor sitting 0.344 ABOVE it, which
    # is a verdict contradicting the number printed two lines
    # above it. Nobody can beat a model that saw the answer,
    # so there are only two explanations and both invalidate
    # the result: they saw the planted truth, or their scores
    # were computed on rows they had fitted. Loud, before
    # anybody quotes it as a win.
    if vend > ceiling + 0.02:
        print("  THEY SCORED ABOVE THE CEILING ({:+.3f}). That "
              "is not a good result - it is a broken exam. The "
              "ceiling is what a model achieves HAVING SEEN "
              "our truth, so nothing can legitimately beat it. "
              "Either the planted answer reached them, or the "
              "scores were computed in-sample on rows their "
              "model had already fitted. Do not quote this "
              "until you know which.".format(vend - ceiling))
    elif vend <= floor + 1e-9:
        print("  They scored at or below the FLOOR - no better "
              "than a model that learned from the file we "
              "deliberately re-pointed.")
    elif vend >= ceiling - 0.02 and vend <= ceiling + 0.02:
        print("  They are at the ceiling. Note what that does "
              "and does not mean: they read the covariates "
              "well. It says nothing about whether they could "
              "have TRAINED on what we sent - that is the "
              "floor's job, and it is {:.3f}.".format(floor))
    print("")
    print("  WHAT THIS GRADES: a planted outcome on synthetic "
          "covariates. It is narrower than 'does this work on "
          "our patients' - the covariates are measured, the "
          "answer is invented, and the exam says so about "
          "itself.")
    return 0


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        print("usage:")
        print("  vendor_exam.py set RUNDIR --effects "
              "\"col=+0.8,col2=-0.5\" [--prevalence P]")
        print("  vendor_exam.py grade RUNDIR --scores FILE.csv "
              "[--score-col NAME] [--bar X]")
        return 2
    cmd, run = argv[0], argv[1]
    kw = {}
    i = 2
    while i < len(argv) - 1:
        if argv[i].startswith("--"):
            kw[argv[i][2:].replace("-", "_")] = argv[i + 1]
            i += 2
        else:
            i += 1
    if cmd == "set":
        if "effects" not in kw:
            print("STOP: set needs --effects \"col=+0.8,...\"")
            return 2
        eff = {}
        for part in kw["effects"].split(","):
            if "=" not in part:
                print("STOP: --effects wants col=SD, got "
                      "{}".format(part))
                return 2
            nm, val = part.rsplit("=", 1)
            try:
                eff[nm.strip()] = float(val)
            except ValueError:
                print("STOP: {} is not a number".format(val))
                return 2
        return set_exam(run, eff,
                        prevalence=float(kw.get("prevalence",
                                                0.25)),
                        seed=int(kw.get("seed", 0)))
    if cmd == "grade":
        if "scores" not in kw:
            print("STOP: grade needs --scores FILE.csv")
            return 2
        return grade(run, kw["scores"],
                     score_col=kw.get("score_col"),
                     bar=kw.get("bar"))
    print("STOP: unknown command {} - set / grade".format(cmd))
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
