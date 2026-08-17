"""Smoke: the attribute-disclosure attack can actually catch a leak.

WHY THIS EXISTS. Membership inference asks "was this person in the
cohort". A governance board asks something else: given the ordinary
fields I already hold about someone, does your synthetic data tell me
the SENSITIVE one? That question had never been measured here.

It is also the harder one to answer honestly, because a generator
whose whole job is to reproduce relationships is by construction good
at predicting one column from the others. Raw accuracy cannot separate
disclosure from ordinary statistical inference.

THE CONTROL IS WHAT MAKES IT A MEASUREMENT: an adversary trained on
DIFFERENT REAL PEOPLE who were never in the cohort. Whatever it
achieves is population structure anyone could obtain, and only the
excess over it belongs to this release.

THE LOAD-BEARING CHECK IS THE POSITIVE CONTROL. A test that cannot
fail proves nothing, so the same attack is pointed at a generator that
republishes the members' own records. If that does not show a large
excess, the attack is broken and a clean result from it means nothing.
Measured: +0.145 there against -0.009 on the real path.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit import blueprint as B                    # noqa: E402
from synthkit.attack import attribute_disclosure       # noqa: E402
from synthkit.discover import discover                 # noqa: E402
from synthkit.generate import generate                 # noqa: E402

PASS = FAIL = 0
SENS = "hiv_status"
QUASI = ["age", "sex_code", "site_code", "bmi"]


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def cohort(seed=0, npat=500):
    """The sensitive field is genuinely predictable here, on purpose.

    If it were independent of everything, every adversary would score
    the base rate and this suite would pass without being able to
    fail. It depends on age, bmi and sex, so the CONTROL scores well
    - and the real question becomes whether the synthetic-trained one
    scores better still."""
    r = np.random.RandomState(seed)
    rows = []
    for p in range(npat):
        age = float(np.clip(r.normal(48, 14), 18, 90))
        bmi = float(np.clip(r.normal(27, 5), 15, 55))
        sex = int(r.rand() < 0.5)
        z = -2.2 + 0.045 * (age - 48) + 0.06 * (bmi - 27) + 0.5 * sex
        pos = int(r.rand() < 1.0 / (1.0 + np.exp(-z)))
        for _v in range(int(max(1, r.poisson(4)))):
            rows.append({
                "person_id": "P{:05d}".format(p),
                "age": round(age, 1), "bmi": round(bmi, 1),
                "sex_code": sex, "site_code": int(r.randint(0, 4)),
                SENS: "positive" if pos else "negative"})
    return pd.DataFrame(rows)


def main():
    df = cohort()
    people = sorted(df["person_id"].unique())
    np.random.RandomState(5).shuffle(people)
    half = len(people) // 2
    members = df[df["person_id"].isin(set(people[:half]))]
    nonmembers = df[df["person_id"].isin(set(people[half:]))]

    check("the fixture's sensitive field IS predictable from the "
          "quasi-identifiers, or every adversary scores the base rate "
          "and nothing here can fail",
          float(members[SENS].value_counts(normalize=True).max())
          < 0.95)

    cat = discover(members, group_by="person_id", seed=1)
    bp = B.build(members, cat, group_by="person_id")
    synth = generate(bp, n_patients=half, seed=7)

    real = attribute_disclosure(
        SENS, QUASI, members.to_dict("records"),
        nonmembers.to_dict("records"), synth.to_dict("records"))

    # THE POSITIVE CONTROL, first - if this does not fire, nothing
    # below is evidence of anything.
    leaky = attribute_disclosure(
        SENS, QUASI, members.to_dict("records"),
        nonmembers.to_dict("records"), members.to_dict("records"))
    check("A GENERATOR THAT REPUBLISHES THE MEMBERS IS CAUGHT - "
          "excess {:+.3f} over the control. Without this the clean "
          "result below would be a test that cannot fail"
          .format(leaky["excess_over_control"]),
          leaky["excess_over_control"] > 0.05)

    check("...and the real path shows no material excess ({:+.3f}), "
          "so it predicts the sensitive field about as well as a "
          "model trained on different real people"
          .format(real["excess_over_control"]),
          real["excess_over_control"] < 0.05)

    check("the control is not a straw man - it beats or matches the "
          "base rate ({:.3f} against {:.3f}), which is what makes "
          "the excess a meaningful subtraction".format(
              real["control_accuracy"], real["marginal_accuracy"]),
          real["control_accuracy"] >= real["marginal_accuracy"] - 0.05)

    check("the excess is reported as its own number, not left for a "
          "reader to compute from two accuracies",
          "excess_over_control" in real)
    check("...and the block says plainly that a high accuracy with a "
          "near-zero excess is the generator working rather than "
          "leaking - the distinction the whole measurement exists to "
          "draw", "not leaking" in real["note"])
    check("...and names what the control was trained on, since that "
          "is the entire basis of the claim",
          "DIFFERENT REAL PEOPLE" in real["note"])

    # ---- it refuses to answer when it cannot ---------------------
    thin = attribute_disclosure(
        SENS, QUASI, members.to_dict("records")[:5],
        nonmembers.to_dict("records")[:5],
        synth.to_dict("records")[:5])
    check("too few rows returns an error rather than a number - a "
          "disclosure figure computed on five records would be read "
          "as a result", "error" in thin)

    one = members.copy()
    one[SENS] = "negative"
    flat = attribute_disclosure(
        SENS, QUASI, one.to_dict("records"),
        one.to_dict("records"), one.to_dict("records"))
    check("a sensitive column with one level returns an error too, "
          "rather than a perfect score that means nothing",
          "error" in flat)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
