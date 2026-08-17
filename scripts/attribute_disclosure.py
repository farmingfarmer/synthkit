"""Does the release help an attacker guess a SENSITIVE field?

    python scripts/attribute_disclosure.py [--patients 700] [--seeds 3]
                                           [-o attribute_disclosure.json]

`membership_new_path.py` answers "was this person in the cohort". That
is not the question a governance board asks. Theirs is: given the
ordinary fields I already have about someone, does your synthetic data
tell me the sensitive one?

And it is the harder question to pass, because a generator whose whole
job is to reproduce relationships faithfully is by construction good
at predicting one column from the others. Raw accuracy therefore
cannot answer it.

THE CONTROL IS THE MEASUREMENT. Three predictors of the same sensitive
column, scored on the same real member records:

    marginal     the base rate; knows nothing about the row
    control      trained on DIFFERENT REAL PEOPLE, never in the cohort
    synthetic    trained on the rows we published

Whatever the control achieves is population structure that anyone with
any sample of that population could obtain. Only the EXCESS over it is
attributable to this release.

AND A LEAKY GENERATOR MUST FAIL IT. A pass means nothing from a test
that cannot fail, so the same attack is run against a generator that
emits the members' own records verbatim. If that does not show a large
excess, the measurement is broken and neither number should be
believed.
"""
import argparse
import json
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

SENSITIVE = "hiv_status"
QUASI = ["age", "sex_code", "site_code", "visits_seen", "bmi"]


def cohort(seed, npat):
    """A cohort where the sensitive field IS genuinely predictable.

    That is deliberate. If the sensitive column were independent of
    everything, both adversaries would score the base rate and the
    test would pass without being able to fail. Here it depends on
    age and bmi, so the control adversary scores well - and the
    question becomes whether the SYNTHETIC one scores better still."""
    r = np.random.RandomState(seed)
    rows = []
    for p in range(npat):
        age = float(np.clip(r.normal(48, 14), 18, 90))
        bmi = float(np.clip(r.normal(27, 5), 15, 55))
        sex = int(r.rand() < 0.5)
        site = int(r.randint(0, 4))
        z = -2.2 + 0.045 * (age - 48) + 0.06 * (bmi - 27) + 0.5 * sex
        pos = int(r.rand() < 1.0 / (1.0 + np.exp(-z)))
        n_vis = int(max(1, r.poisson(4)))
        for _v in range(n_vis):
            rows.append({
                "person_id": "P{:05d}".format(p),
                "age": round(age, 1), "bmi": round(bmi, 1),
                "sex_code": sex, "site_code": site,
                "visits_seen": n_vis,
                SENSITIVE: "positive" if pos else "negative"})
    return pd.DataFrame(rows)


def one_seed(seed, npat, say):
    df = cohort(seed, npat)
    people = sorted(df["person_id"].unique())
    rs = np.random.RandomState(seed + 99)
    rs.shuffle(people)
    half = len(people) // 2
    mem_ids, non_ids = set(people[:half]), set(people[half:])
    members = df[df["person_id"].isin(mem_ids)]
    nonmembers = df[df["person_id"].isin(non_ids)]

    cat = discover(members, group_by="person_id", seed=seed + 1)
    bp = B.build(members, cat, group_by="person_id")
    synth = generate(bp, n_patients=len(mem_ids), seed=seed + 7)

    real = attribute_disclosure(
        SENSITIVE, QUASI,
        members.to_dict("records"), nonmembers.to_dict("records"),
        synth.to_dict("records"), seed=seed)

    # THE POSITIVE CONTROL. A generator that publishes the members'
    # own records is the worst case, and the attack must see it.
    leaky = attribute_disclosure(
        SENSITIVE, QUASI,
        members.to_dict("records"), nonmembers.to_dict("records"),
        members.to_dict("records"), seed=seed)

    say("seed {}  marginal {:.3f}  control {:.3f}  synthetic {:.3f}"
        "  EXCESS {:+.3f}   |  leaky generator excess {:+.3f}".format(
            seed, real["marginal_accuracy"], real["control_accuracy"],
            real["synthetic_accuracy"], real["excess_over_control"],
            leaky["excess_over_control"]))
    return real, leaky


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--patients", type=int, default=700)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("-o", "--out", default="")
    a = ap.parse_args()

    def say(m):
        print(m, flush=True)

    say("attribute disclosure: {} from {}".format(SENSITIVE, QUASI))
    say("")
    reals, leaks = [], []
    for s in range(a.seeds):
        r, k = one_seed(s, a.patients, say)
        reals.append(r)
        leaks.append(k)

    exc = [r["excess_over_control"] for r in reals]
    lex = [k["excess_over_control"] for k in leaks]
    say("")
    say("EXCESS over the control, across {} seeds: mean {:+.3f}, "
        "range {:+.3f} to {:+.3f}".format(len(exc), float(np.mean(exc)),
                                          min(exc), max(exc)))
    say("the same attack on a generator that republishes the members: "
        "mean {:+.3f}".format(float(np.mean(lex))))
    say("")
    if float(np.mean(lex)) < 0.03:
        say("THE POSITIVE CONTROL DID NOT FIRE. A leaky generator "
            "should be caught by this attack, and it was not - so "
            "neither number above means anything. Do not report them.")
        rc = 1
    elif float(np.mean(exc)) > 0.05:
        say("EXCESS IS MATERIAL. The release predicts the sensitive "
            "field better than population structure alone, on people "
            "who were in the cohort.")
        rc = 1
    else:
        say("No material excess: the synthetic data predicts the "
            "sensitive field about as well as a model trained on "
            "different real people, which is the generator working "
            "rather than leaking.")
        rc = 0

    say("")
    say("WHAT THIS IS NOT. One cohort shape, one k, one sensitive "
        "column, one adversary family. A floor, not a certificate, "
        "and not a release decision for a different cohort.")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"seeds": reals, "positive_control": leaks},
            indent=1), encoding="utf-8")
        say("wrote {}".format(a.out))
    return rc


if __name__ == "__main__":
    sys.exit(main())
