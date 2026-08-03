"""What does privacy cost, and what does it buy?

    python scripts/privacy_curve.py [--patients 400] [--seeds 3]
                                    [-o privacy_curve.json]
                                    [--report]

Two questions a privacy office actually asks, answered on the same
axis:

  what does it COST   how much of the real structure survives at
                      each budget
  what does it BUY    how well an adversary can tell who was in
                      the cohort at each budget

Both are measured, not argued. The cost is the planted
relationship's strength in the generated data; the buy is the AUC
of the strongest membership attack we can mount, where 0.5 is a
coin flip.

The point of the artifact is that these can be read together: if a
budget costs nothing and the attack already fails without it, the
budget is cheap insurance. If the attack succeeds without a
budget, it is not optional.
"""
from __future__ import annotations

import argparse
import json
import random
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
from synthkit.condnet import CondNet          # noqa: E402
from synthkit.attack import membership_audit  # noqa: E402


def cohort(n_patients, seed, link=0.9):
    r = random.Random(seed)
    rows = []
    for pid in range(n_patients):
        chf = r.random() < 0.35
        bp = r.gauss(132, 16)
        for _ in range(r.randint(2, 10)):
            sbp = r.gauss(bp, 8)
            cs = [c for c in ("dm", "htn", "ckd")
                  if r.random() < 0.4]
            if chf:
                cs.append("chf")
            dr = [d for d in ("lisinopril", "metformin")
                  if r.random() < 0.5]
            if chf and r.random() < link:
                dr.append("furosemide")
            rows.append({
                "person_id": "P%04d" % pid,
                "systolic_blood_pressure": round(sbp, 1),
                "diastolic_blood_pressure":
                    round(0.55 * sbp + r.gauss(0, 4), 1),
                "conditions": "; ".join(sorted(cs)) or "none",
                "active_drugs": "; ".join(sorted(dr)) or "none"})
    return rows


def link_strength(rows):
    def has(row, term):
        return term in (str(row.get("conditions", ""))
                        + str(row.get("active_drugs", ""))).lower()
    a = [r for r in rows if has(r, "chf")]
    b = [r for r in rows if not has(r, "chf")]
    if not a or not b:
        return 0.0
    return (sum(1 for r in a if has(r, "furosemide")) / len(a)
            - sum(1 for r in b if has(r, "furosemide")) / len(b))


def one(eps, patients, seed):
    rows = cohort(patients, seed)
    byp = {}
    for r in rows:
        byp.setdefault(r["person_id"], []).append(r)
    pids = sorted(byp)
    random.Random(seed).shuffle(pids)
    half = len(pids) // 2
    members = [r for p in pids[:half] for r in byp[p]]
    nonmem = [r for p in pids[half:] for r in byp[p]]
    net = CondNet(k=10, max_parents=3).learn(
        members, group_by="person_id", epsilon=eps)
    syn = net.sample(1500, seed=seed + 1)
    audit = membership_audit(net, members, nonmem, syn,
                             nn_sample=120)
    return {"link_source": link_strength(members),
            "link_generated": link_strength(syn),
            "attack_auc": audit["worst_auc"],
            "likelihood_auc": audit["likelihood"]["auc"],
            "verdict": audit["verdict"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epsilons",
                    default="none,10,3,1,0.3,0.1")
    ap.add_argument("-o", "--out", default="privacy_curve.json")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    eps_list = [None if e.strip().lower() == "none" else float(e)
                for e in a.epsilons.split(",")]

    results = []
    for eps in eps_list:
        runs = [one(eps or 0.0, a.patients, 500 + s)
                for s in range(a.seeds)]
        results.append({
            "epsilon": eps,
            "link_source": round(statistics.mean(
                x["link_source"] for x in runs), 3),
            "link_generated": round(statistics.mean(
                x["link_generated"] for x in runs), 3),
            "attack_auc": round(statistics.mean(
                x["attack_auc"] for x in runs), 3),
            "verdict": max(x["verdict"] for x in runs)})
        print("  epsilon {:>6} done".format(
            "none" if eps is None else eps), flush=True)

    payload = {
        "config": {"patients": a.patients, "seeds": a.seeds},
        "results": results,
        "reading": "link_generated against link_source is what "
                   "the budget COSTS. attack_auc is what it BUYS, "
                   "where 0.5 means an adversary cannot tell who "
                   "was in the cohort.",
        "caveat": "measured on a synthetic cohort with a planted "
                  "relationship, because a real one cannot supply "
                  "known truth. The shape of the tradeoff is the "
                  "finding; the exact numbers belong to this "
                  "fixture."}
    Path(a.out).write_text(json.dumps(payload, indent=1),
                           encoding="utf-8")
    print("\nWROTE {}".format(a.out))

    if a.report:
        print("\n" + "=" * 66)
        print("WHAT A PRIVACY BUDGET COSTS, AND WHAT IT BUYS")
        print("=" * 66)
        print("  {:>9}  {:>16}  {:>14}  {:>9}".format(
            "epsilon", "structure kept", "attack AUC", "verdict"))
        print("  " + "-" * 60)
        for r in results:
            kept = (r["link_generated"] / r["link_source"]
                    if r["link_source"] else 0.0)
            print("  {:>9}  {:>15.0%}  {:>14.3f}  {:>9}".format(
                "none" if r["epsilon"] is None else r["epsilon"],
                kept, r["attack_auc"], r["verdict"]))
        print("\n  0.5 on the attack is a coin flip: the adversary "
              "learned nothing about who was in the cohort.")
        print("  100% structure kept means the planted "
              "relationship survived generation intact.")


if __name__ == "__main__":
    main()
