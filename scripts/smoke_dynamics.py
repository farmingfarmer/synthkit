"""Smoke: missingness clusters and values persist, and fixing that
breaks nothing next to it.

THE FIXTURE REPRODUCES THE FAULT FIRST. Both properties are planted at
values derived from the real extract's measured statistics, and the
suite asserts the OLD behaviour fails on them before asserting the new
behaviour passes. A guard verified only against a shape invented here
passes while failing on the shape that exists.

  clustered missingness   a 44%-covered vital, planted with excess
                          clustering 0.55 - the real extract's
                          44%-covered columns retained 40-47% of their
                          source steadiness against 74-93% for
                          complete ones, and coverage was the whole
                          difference
  a persistent value      icc 0.5 with within-patient drift, the
                          split the pooled autocorrelation cannot see

AND THE NEIGHBOURING PROPERTY IS CHECKED IN THE SAME TEST. Steadiness
and missingness were fixed and broken in one commit once, because only
one of them was measured afterwards. Every dynamics check here is
paired with the marginal, the coverage and the relationship it could
have damaged.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit import blueprint as B                    # noqa: E402
from synthkit.discover import discover, prepare        # noqa: E402
from synthkit.dynamics import (icc1, measure,          # noqa: E402
                               missing_clustering, pooled_lag1,
                               within_from)
from synthkit.generate import generate                 # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


COVERAGE, EXCESS, ICC = 0.44, 0.55, 0.50


def source(n_pat=320, n_vis=10, seed=17):
    r = np.random.RandomState(seed)
    rows = []
    # the two-state chain the generator must invert
    a = COVERAGE + (1 - COVERAGE) * EXCESS
    b = COVERAGE * (1 - EXCESS)
    for p in range(n_pat):
        level = r.normal(0, np.sqrt(ICC))       # the patient's own
        prev_seen = r.random_sample() < COVERAGE
        w = r.normal(0, 1)
        for v in range(n_vis):
            w = 0.7 * w + np.sqrt(1 - 0.49) * r.normal(0, 1)
            val = 100.0 + 10.0 * (level + np.sqrt(1 - ICC) * w)
            thr = a if prev_seen else b
            prev_seen = r.random_sample() < thr
            x = r.uniform(0, 10)
            rows.append({
                "person_id": "P{:04d}".format(p),
                "visit_no": v + 1,
                "vital": round(val, 3) if prev_seen else "",
                # a plain relationship that must survive untouched
                "x": round(x, 4),
                "y": round(3.0 * x + r.normal(0, 1.0), 4),
            })
    return pd.DataFrame(rows)


def stats(df, col, time_col="visit_no", group="person_id"):
    # NOT a rename of the generated frame: `visit_no` is itself a
    # modelled column, so renaming visit_number onto it makes two
    # columns share a name and every later lookup returns a frame.
    X, _, _, _ = prepare(df, group)
    d = measure(df, X, group, time_col)
    return d[col]


def main():
    df = source()

    # ---- the fixture really does carry the faults ----------------
    src = stats(df, "vital")
    check("the fixture's missingness genuinely CLUSTERS, derived from "
          "the real extract's 44%-covered columns rather than "
          "invented - without this the guard is tested against a "
          "shape that does not exist",
          src["missing_clustering"] > 0.35)
    check("...and its value genuinely PERSISTS, with BOTH sources "
          "present - a patient level, and drift on top of it. The "
          "pooled lag-1 must exceed the icc, or there is nothing for "
          "a transition mechanism to add and the split is untested",
          src["icc"] > 0.3 and src["lag1_total"] > src["icc"] + 0.1
          and src["within_lag1"] > 0.3)

    cat = discover(df, group_by="person_id", seed=6)
    bp = B.build(df, cat, group_by="person_id", time_col="visit_no")
    dyn = bp["columns"]["vital"]["dynamics"]
    check("the blueprint MEASURES clustering back to what the chain "
          "was planted with, so the number is the statistic and not a "
          "label - and this is what caught a lexicographic sort that "
          "put visit 10 before visit 2 and read 0.55 as 0.44",
          abs(dyn["missing_clustering"] - EXCESS) < 0.10)
    check("...and separates the patient level from the drift, which "
          "one pooled figure cannot do",
          dyn["icc"] > 0.3 and dyn["within_lag1"] > 0.3
          and dyn["lag1_total"] > dyn["icc"])

    # ---- OLD behaviour must FAIL these ---------------------------
    # Dialling both to zero is exactly what the first generator did.
    off = json.loads(json.dumps(bp))
    off["columns"]["vital"]["dials"]["missing_clustering"] = 0.0
    off["columns"]["vital"]["dials"]["persistence"] = 0.0
    g_off = generate(off, n_patients=320, seed=3)
    s_off = stats(g_off, "vital", "visit_number")
    check("WITHOUT the fix the generated missingness does NOT cluster "
          "- the fault is reproduced, so the check below can fail",
          s_off["missing_clustering"] < 0.15)
    check("...and WITHOUT it the value does not persist either",
          s_off["icc"] < 0.15 and s_off["lag1_total"] < 0.15)

    # ---- WITH the modelling --------------------------------------
    g = generate(bp, n_patients=320, seed=3)
    gs = stats(g, "vital", "visit_number")
    check("generated missingness now CLUSTERS like the source - runs "
          "of unmeasured visits, not a scatter with the right total",
          abs(gs["missing_clustering"] - src["missing_clustering"])
          < 0.18)
    check("the patient's own LEVEL is reproduced - and it is the "
          "SOLVED anchor weight that gets this, not the measured icc "
          "used directly, which overshot to 0.807 against a source of "
          "0.689 because the drift adds between-patient variance of "
          "its own",
          abs(gs["icc"] - src["icc"]) < 0.12)
    check("and so is the total persistence, which is what "
          "visit-to-visit steadiness actually means",
          abs(gs["lag1_total"] - src["lag1_total"]) < 0.10)

    # ---- the neighbouring properties, in the same test -----------
    sv = pd.to_numeric(df["vital"], errors="coerce")
    gv = pd.to_numeric(g["vital"], errors="coerce")
    check("COVERAGE is untouched by the clustering - the chain's "
          "stationary share is the coverage, so the two are separate "
          "dials rather than one confounded one",
          abs(gv.notna().mean() - sv.notna().mean()) < 0.06)
    check("the MARGINAL is untouched by the persistence - steadiness "
          "enters as correlation between uniforms, so the values that "
          "come out are distributed as before",
          abs(gv.mean() - sv.mean()) < 2.0
          and abs(gv.std() - sv.std()) < 2.0)

    def corr(d, a, b):
        s = pd.DataFrame({"a": pd.to_numeric(d[a], errors="coerce"),
                          "b": pd.to_numeric(d[b], errors="coerce")}
                         ).dropna()
        return float(s.a.corr(s.b))
    check("the RELATIONSHIP between other columns still survives - "
          "steadiness and missingness were fixed and broken in one "
          "commit once, because only one was measured afterwards",
          corr(g, "x", "y") > 0.6 * corr(df, "x", "y"))

    # ---- the dials must move these too ---------------------------
    up = json.loads(json.dumps(bp))
    up["columns"]["vital"]["dials"]["missing_clustering"] = 0.9
    gu = generate(up, n_patients=320, seed=3)
    su = stats(gu, "vital", "visit_number")
    check("a clustering dial makes missingness lumpier when asked",
          su["missing_clustering"] > gs["missing_clustering"] + 0.15)
    guv = pd.to_numeric(gu["vital"], errors="coerce")
    check("...without moving coverage, which is the point of "
          "measuring EXCESS over independence rather than a raw match "
          "rate confounded by how often the column is present",
          abs(guv.notna().mean() - sv.notna().mean()) < 0.06)

    # ---- the statistics themselves are not vacuous ---------------
    r = np.random.RandomState(1)
    n = 4000
    grp = np.repeat(np.arange(400), 10)
    check("independent presence measures ZERO excess clustering, so "
          "the statistic is not just reporting coverage back",
          missing_clustering(r.random_sample(n) < 0.44,
                             np.arange(n - 1)[grp[:-1] == grp[1:]],
                             np.arange(1, n)[grp[:-1] == grp[1:]])
          < 0.08)
    check("a column with no between-patient structure measures ICC "
          "near zero", icc1(r.normal(0, 1, n), grp) < 0.1)
    idx = np.arange(n - 1)[grp[:-1] == grp[1:]]
    check("...and white noise carries no lag-1 correlation either",
          abs(pooled_lag1(r.normal(0, 1, n), idx, idx + 1)) < 0.1)
    check("the within-AR is DERIVED from the pooled lag-1 against the "
          "icc, and inverts exactly - measuring it by centring on "
          "each patient's own mean is attenuated, and feeding that "
          "back attenuates it a second time until the property "
          "vanishes from the output",
          abs(within_from(0.5 + 0.5 * 0.7, 0.5) - 0.7) < 1e-9)

    # ---- BEING MEASURED IS ITSELF A SIGNAL ------------------------
    # Presence was drawn from a coverage share and a clustering dial
    # and nothing else, so whether a lab existed on a row was
    # independent of everything on that row. In an extract the test
    # was ordered BECAUSE the patient was unwell. Its own fixture,
    # because adding a column to a shared one has already cost three
    # unrelated checks this week.
    rp = np.random.RandomState(11)
    NPM, NVM = 400, 25
    gm = np.repeat(np.arange(NPM), NVM)
    nm = len(gm)
    sev = np.round(60 + 25 * rp.normal(0, 1, nm), 1)
    pr = 1.0 / (1.0 + np.exp(-(sev - 75) / 8.0))
    lact = np.where(rp.random_sample(nm) < pr,
                    np.round(rp.lognormal(0.3, 0.5, nm), 2), np.nan)
    dfm = pd.DataFrame({"person_id": ["P{:04d}".format(x) for x in gm],
                        "severity": sev, "lactate": lact,
                        "noise": np.round(rp.normal(0, 1, nm), 3)})
    bpm = B.build(dfm, {"claims": [], "unexplained": [], "skipped": []},
                  group_by="person_id")
    pm = bpm["columns"]["lactate"].get("presence")
    check("what makes a column get MEASURED is discovered - {} drives "
          "whether lactate exists at all".format(
              None if not pm else pm["parent"]),
          pm is not None and pm["parent"] == "severity")
    check("...and every published bin is backed by k PATIENTS, like "
          "every other number that leaves this machine",
          pm is not None and pm.get("bins_are_k_anonymous") == 10)
    check("a column whose presence depends on NOTHING gets no model, "
          "or the mechanism would be invented everywhere",
          bpm["columns"]["noise"].get("presence") is None)

    bp_off = dict(bpm)
    cols_off = dict((k2, dict(v)) for k2, v in bpm["columns"].items())
    cols_off["lactate"].pop("presence", None)
    bp_off["columns"] = cols_off
    g_on = generate(bpm, n_patients=NPM, seed=5)
    g_off = generate(bp_off, n_patients=NPM, seed=5)

    def sev_gap(fr):
        a = pd.to_numeric(fr["lactate"], errors="coerce")
        s = pd.to_numeric(fr["severity"], errors="coerce")
        pres = a.notna()
        return float(s[pres].mean() - s[~pres].mean())

    def cov_of(fr):
        return float(pd.to_numeric(fr["lactate"],
                                   errors="coerce").notna().mean())
    gs, go, gn = sev_gap(dfm), sev_gap(g_off), sev_gap(g_on)
    check("THE FIXTURE REPRODUCES THE FAULT: without a presence model "
          "the measured and unmeasured rows differ by {:+.2f} in "
          "severity where the source differs by {:+.2f} - the signal "
          "is simply gone".format(go, gs), abs(go) < 0.15 * abs(gs))
    check("...and with one it comes back at {:+.2f}".format(gn),
          abs(gn - gs) < 0.35 * abs(gs))
    check("COVERAGE IS NOT MOVED BY IT - informative missingness "
          "changes WHICH rows are present, not how many: {:.3f} "
          "against {:.3f}".format(cov_of(g_on), cov_of(dfm)),
          abs(cov_of(g_on) - cov_of(dfm)) <= 0.05)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
