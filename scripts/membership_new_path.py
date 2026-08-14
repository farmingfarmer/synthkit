"""Can an attacker tell who was in the cohort, on the NEW path?

    python scripts/membership_new_path.py [--patients 600] [--seeds 3]
                                          [-o membership_new.json]

`privacy_curve.py` answers this for CondNet. The discover / blueprint /
generate path has never been asked, and `CONVENTIONS.md` has said so
since the pivot: "no membership-inference test has been run against
this path". That sentence is a design argument standing where evidence
should be, and this is the evidence.

THE SETUP IS THE WHOLE TEST. Patients are split in two. The blueprint
is fitted on the MEMBERS only and synthetic rows are generated from
it. The attacker is then handed exactly what leaves the machine - the
blueprint and the synthetic rows - and asked to say which of a mixed
pile of real records were members. Non-members come from the same
population, or the attack measures the difference between two
populations rather than a leak.

TWO ADVERSARIES, and the worse one is reported:

  nearest neighbour   score a record by how close the closest
                      synthetic row is. Needs no model at all
  likelihood          score a record by how probable the published
                      blueprint finds it. Reads the density straight
                      off the artefact

0.5 is a coin flip. The bands are deliberately generous to the
attacker - a claim of privacy should survive a strict reading, not a
flattering one.

WHAT THIS DOES NOT SETTLE. One cohort shape, one k, one fixture. It is
a floor, not a certificate, and a PASS here does not license a release
decision on a different cohort.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from _console import console_safe
    console_safe()
except Exception:
    pass


def cohort(n_patients, n_visits, seed):
    import numpy as np
    import pandas as pd
    r = np.random.RandomState(seed)
    g = np.repeat(np.arange(n_patients), n_visits)
    n = len(g)
    sev = np.round(60 + 25 * r.normal(0, 1, n), 1)
    return pd.DataFrame({
        "person_id": ["P{:05d}".format(x) for x in g],
        "severity": sev,
        "lab": np.round(r.lognormal(1.0, 0.7, n), 2),
        "bmi": np.round(24 + 0.05 * sev + r.normal(0, 3, n), 1),
        "site": r.choice(["A", "B", "C"], n, p=[.5, .3, .2]),
    })


def run_once(n_patients, n_visits, seed, k):
    import numpy as np
    from synthkit import blueprint as B
    from synthkit.attack import BlueprintLikelihood, membership_audit
    from synthkit.generate import generate

    df = cohort(n_patients, n_visits, seed)
    ids = sorted(df["person_id"].unique())
    rng = np.random.RandomState(seed + 1)
    rng.shuffle(ids)
    half = len(ids) // 2
    member_ids, non_ids = set(ids[:half]), set(ids[half:])
    members_df = df[df["person_id"].isin(member_ids)]
    nonmembers_df = df[df["person_id"].isin(non_ids)]

    # FITTED ON MEMBERS ONLY. Everything the attacker sees comes from
    # this half; the other half is the control.
    bp = B.build(members_df, {"claims": [], "unexplained": [],
                              "skipped": []},
                 group_by="person_id", k=k)
    synth = generate(bp, n_patients=len(member_ids), seed=seed + 2)

    audit = membership_audit(
        BlueprintLikelihood(bp),
        members_df.to_dict("records"),
        nonmembers_df.to_dict("records"),
        synthetic=synth.to_dict("records"))
    audit["k"] = k
    audit["patients_fitted"] = len(member_ids)
    return audit


def main():
    ap = argparse.ArgumentParser(
        description="Membership inference against the fitted path.")
    ap.add_argument("--patients", type=int, default=600)
    ap.add_argument("--visits", type=int, default=8)
    ap.add_argument("--seeds", default="11,23,37")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("-o", "--out", default="")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    if len(seeds) < 2:
        sys.exit("at least two seeds: one attack AUC is a draw from a "
                 "distribution, not a measurement")

    runs = []
    for s in seeds:
        r = run_once(a.patients, a.visits, s, a.k)
        runs.append(r)
        print("seed {:<5} nn {:.3f}  likelihood {:.3f}  worst {:.3f}  "
              "{}".format(s, r.get("nearest_neighbour", {}).get("auc",
                                                                0.5),
                          r.get("likelihood", {}).get("auc", 0.5),
                          r["worst_auc"], r["verdict"]))
    worst = [r["worst_auc"] for r in runs]
    summary = {
        "patients_fitted": runs[0]["patients_fitted"],
        "k": a.k,
        "seeds": seeds,
        "worst_auc_mean": round(statistics.mean(worst), 4),
        "worst_auc_max": round(max(worst), 4),
        "verdict": ("PASS" if max(worst) < 0.60 else
                    "MARGINAL" if max(worst) < 0.70 else "FAIL"),
        "reading": "0.5 is a coin flip. This is one cohort shape at "
                   "one k on one fixture - a floor, not a "
                   "certificate, and not a release decision for a "
                   "different cohort.",
        "runs": runs,
    }
    print()
    print("worst AUC over {} seeds: mean {:.3f}, max {:.3f} -> {}"
          .format(len(seeds), summary["worst_auc_mean"],
                  summary["worst_auc_max"], summary["verdict"]))
    print(summary["reading"])
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1),
                               encoding="utf-8")
        print("wrote {}".format(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
