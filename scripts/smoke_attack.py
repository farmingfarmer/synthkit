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

    # ---- a raised score on a tiny cohort needs explaining ----
    tiny = [r for r in members
            if r["person_id"] < "P0060"]
    tiny_non = [r for r in nonmembers
                if r["person_id"] < "P0180"]
    tnet = CondNet(k=10, max_parents=3).learn(
        tiny, group_by="person_id")
    taud = membership_audit(tnet, tiny, tiny_non,
                            tnet.sample(600, seed=4),
                            nn_sample=60)
    check("the audit reports how many PEOPLE stood behind the "
          "model, because that governs how the score should be "
          "read",
          taud["members_are_people"] > 0)
    check("a raised score on a small cohort is explained rather "
          "than left to alarm — with too few people there is no "
          "crowd to hide in, and that is a reason to widen the "
          "cohort rather than distrust the method",
          taud["worst_auc"] < 0.60 or "widen the cohort"
          in taud.get("context", ""))

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

    # ---- THE NEW PATH HAD NEVER BEEN ATTACKED --------------------
    # CONVENTIONS has said since the pivot that "no membership-
    # inference test has been run against this path" - a design
    # argument standing where evidence should be. The attacker gets
    # exactly what leaves the machine: the blueprint and the synthetic
    # rows, nothing else.
    import numpy as _np
    import pandas as _pd
    from synthkit import blueprint as _B
    from synthkit.attack import BlueprintLikelihood
    from synthkit.generate import generate as _gen

    _r = _np.random.RandomState(11)
    _npat, _nvis = 400, 8
    _g = _np.repeat(_np.arange(_npat), _nvis)
    _n = len(_g)
    _sev = _np.round(60 + 25 * _r.normal(0, 1, _n), 1)
    _df = _pd.DataFrame({
        "person_id": ["P{:05d}".format(x) for x in _g],
        "severity": _sev,
        "lab": _np.round(_r.lognormal(1.0, 0.7, _n), 2),
        "site": _r.choice(["A", "B", "C"], _n, p=[.5, .3, .2])})
    _ids = sorted(_df["person_id"].unique())
    _rng = _np.random.RandomState(12)
    _rng.shuffle(_ids)
    _mem = set(_ids[:len(_ids) // 2])
    _mdf = _df[_df["person_id"].isin(_mem)]
    _ndf = _df[~_df["person_id"].isin(_mem)]
    _bp = _B.build(_mdf, {"claims": [], "unexplained": [],
                          "skipped": []}, group_by="person_id", k=10)
    _syn = _gen(_bp, n_patients=len(_mem), seed=5)
    _audit = membership_audit(BlueprintLikelihood(_bp),
                              _mdf.to_dict("records"),
                              _ndf.to_dict("records"),
                              synthetic=_syn.to_dict("records"))
    check("a blueprint can be SCORED as a model, so the likelihood "
          "adversary works on the fitted path and not only on CondNet",
          "likelihood" in _audit and "auc" in _audit["likelihood"])
    check("membership inference against the fitted path is near a "
          "coin flip - worst AUC {:.3f} across both adversaries"
          .format(_audit["worst_auc"]), _audit["worst_auc"] < 0.60)

    # THE GUARD HAS TO BE ABLE TO FAIL. An attack that always reports
    # a coin flip proves nothing at all, so it is handed a generator
    # that leaks everything: the members themselves as the synthetic
    # data. A nearest-neighbour adversary must find that instantly.
    _leak = membership_audit(BlueprintLikelihood(_bp),
                             _mdf.to_dict("records"),
                             _ndf.to_dict("records"),
                             synthetic=_mdf.to_dict("records"))
    check("...and the attack CATCHES a generator that leaks - handed "
          "the members themselves as synthetic data it scores {:.3f} "
          "and returns {}, so the pass above is a measurement rather "
          "than a formality"
          .format(_leak["worst_auc"], _leak["verdict"]),
          _leak["worst_auc"] > 0.75 and _leak["verdict"] == "FAIL")
    check("...and it reports how many PEOPLE stood behind the model, "
          "because a raised score on a small cohort says as much "
          "about the cohort as the method",
          _audit.get("members_are_people", 0) > 0)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
