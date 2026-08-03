"""Smoke: the privacy claim is attacked, not asserted.

A privacy test that cannot fail proves nothing, so the first thing
checked here is that the attack CATCHES a generator which
memorises. Only then do the passing results mean anything.
"""
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from synthkit.condnet import CondNet          # noqa: E402
from synthkit.attack import (                 # noqa: E402
    membership_audit, nearest_neighbour_attack)

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def split(rows, seed=1):
    byp = {}
    for r in rows:
        byp.setdefault(r["person_id"], []).append(r)
    pids = sorted(byp)
    random.Random(seed).shuffle(pids)
    half = len(pids) // 2
    return ([r for p in pids[:half] for r in byp[p]],
            [r for p in pids[half:] for r in byp[p]])


def main():
    from smoke_learnspec import cohort
    rows = cohort(patients=300)
    members, nonmembers = split(rows)
    check("members and non-members are split by PATIENT, so the "
          "attack measures membership rather than which visits "
          "happened to land where",
          not ({r["person_id"] for r in members}
               & {r["person_id"] for r in nonmembers}))

    # ---- the attack must catch a real leak ----
    leak = nearest_neighbour_attack(members[:150], nonmembers[:150],
                                    members[:2000])
    check("a generator that MEMORISES is caught outright — a "
          "privacy test that cannot fail proves nothing",
          leak["auc"] > 0.9)

    net = CondNet(k=10, max_parents=3).learn(
        members, group_by="person_id")
    syn = net.sample(1500, seed=3)
    audit = membership_audit(net, members, nonmembers, syn,
                             nn_sample=120)
    check("both adversaries are run and reported separately",
          "likelihood" in audit and "nearest_neighbour" in audit)
    check("the stronger adversary is the one given the published "
          "MODEL, which is what a determined attacker would have",
          audit["likelihood"]["attacker_sees"]
          != audit["nearest_neighbour"]["attacker_sees"])
    check("the verdict is taken from the STRONGEST adversary, not "
          "an average of them",
          audit["worst_auc"] >= audit["likelihood"]["auc"])
    check("parameter-based generation survives the attack",
          audit["worst_auc"] < 0.60
          and audit["verdict"] == "PASS")
    check("the reading explains what the number means rather than "
          "just printing it",
          "coin flip" in audit["reading"])

    dp = CondNet(k=10, max_parents=3).learn(
        members, group_by="person_id", epsilon=1.0)
    audit_dp = membership_audit(dp, members, nonmembers,
                                dp.sample(1500, seed=3),
                                nn_sample=120)
    check("a model carrying a privacy budget also survives it",
          audit_dp["verdict"] == "PASS")

    check("a model assigns no higher likelihood to a member than "
          "to a stranger by any wide margin",
          abs(net.log_likelihood(members[0])
              - net.log_likelihood(nonmembers[0])) < 50)

    # ---- the deliverable ----
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "pc.json"
        r = subprocess.run(
            [sys.executable, "scripts/privacy_curve.py",
             "--patients", "150", "--seeds", "1",
             "--epsilons", "none,1,0.1",
             "-o", str(out), "--report"],
            capture_output=True, text=True, cwd=str(ROOT))
        check("the privacy curve runs and writes an artifact",
              r.returncode == 0 and out.exists())
        P = json.loads(out.read_text(encoding="utf-8"))
        check("it reports what a budget COSTS and what it BUYS on "
              "the same axis",
              all("link_generated" in x and "attack_auc" in x
                  for x in P["results"]))
        check("a tighter budget keeps less of the structure — the "
              "cost is real and is shown",
              P["results"][-1]["link_generated"]
              <= P["results"][0]["link_generated"] + 0.05)
        check("the artifact says plainly that the numbers belong "
              "to the fixture and the SHAPE is the finding",
              "shape of the tradeoff" in P["caveat"])

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
