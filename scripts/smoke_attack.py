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
    membership_audit, nearest_neighbor_attack)

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
    leak = nearest_neighbor_attack(members[:150], nonmembers[:150],
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
          "likelihood" in audit and "nearest_neighbor" in audit)
    check("the stronger adversary is the one given the published "
          "MODEL, which is what a determined attacker would have",
          audit["likelihood"]["attacker_sees"]
          != audit["nearest_neighbor"]["attacker_sees"])
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
    # data. A nearest-neighbor adversary must find that instantly.
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

    # ---- THE AUDIT COULD NOT SEE A SET COLUMN AT ALL ----
    #
    # Two blind spots, both of which made this suite's PASS mean less
    # than it read.
    #
    # `BlueprintLikelihood` branched on `quantiles` and `levels` and
    # had no `list` branch, so a set column contributed its coverage
    # term and nothing else. Measured: a row holding a p=0.900 token
    # and a row holding a p=0.001 token scored 0.000000 apart. A rare
    # token is exactly what singles a person out, and the marginal
    # now publishes every token that clears k rather than the sixty
    # most common - so the audit was blind to the part of the release
    # that GREW.
    #
    # The nearest-neighbor half compared the whole combination
    # STRING for equality. On a column drawing four tokens from
    # hundreds that is false on essentially every pair, so the term
    # added a constant to every distance and canceled. A verbatim
    # republish was still caught - the strings match - which is why
    # the existing leak control above never exposed it. A PARTIAL
    # leak was not: handed each member's own tokens with ONE swapped
    # out, the old distance scored 0.500, a coin flip, against 0.998
    # with Jaccard. That is a generator republishing somebody's
    # condition list almost verbatim and an audit calling it clean.
    import numpy as _np
    _NT = 400
    _zp = 1.0 / (_np.arange(1, _NT + 1) ** 0.62)
    _zp = _zp / _zp.sum()

    def _mk(n, rs):
        out = []
        for _ in range(n):
            k = max(3, int(rs.poisson(5)))
            out.append(sorted("t{:03d}".format(j) for j in rs.choice(
                _NT, size=min(k, _NT), replace=False, p=_zp)))
        return out

    _rs = _np.random.RandomState(1)
    _mem, _non = _mk(300, _rs), _mk(300, _rs)
    _M = [{"tags": ";".join(t)} for t in _mem]
    _N = [{"tags": ";".join(t)} for t in _non]

    _bl = {"columns": {"tags": {"coverage": 1.0, "marginal": {
        "type": "list", "separator": ";",
        "tokens": [{"value": "t1", "p": 0.9},
                   {"value": "t2", "p": 0.001}]}}}}
    _net = BlueprintLikelihood(_bl)
    _gap = abs(_net.log_likelihood({"tags": "t1"})
               - _net.log_likelihood({"tags": "t2"}))
    check("the likelihood attack SEES a published token's rarity "
          "(a 900x rarity difference moves the score by {:.3f}) - "
          "with no list branch it moved it by exactly 0.000000"
          .format(_gap),
          _gap > 1.0)

    _rs2 = _np.random.RandomState(7)
    _part = []
    for _t in _mem:
        _t2 = list(_t)
        _t2[_rs2.randint(len(_t2))] = "t{:03d}".format(_rs2.randint(_NT))
        _part.append({"tags": ";".join(sorted(set(_t2)))})
    _pa = nearest_neighbor_attack(_M, _N, _part)["auc"]
    check("...and a PARTIAL set leak is caught - each member's own "
          "tokens with one swapped scores {:.3f}, where comparing "
          "the combination string scored 0.500".format(_pa),
          _pa > 0.90)

    # THE NEGATIVE CONTROL, or the check above only proves that
    # everything with a set column scores high.
    _hon = nearest_neighbor_attack(
        _M, _N, [{"tags": ";".join(t)}
                 for t in _mk(300, _np.random.RandomState(9))])["auc"]
    check("...while an HONEST set generator drawing fresh from the "
          "same population is NOT flagged ({:.3f}) - without this "
          "the check above would pass on an attack that shouts at "
          "every set column".format(_hon),
          _hon < 0.60)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
