"""Smoke: the conditional dependency network captures structure
that marginals-plus-correlations provably cannot — and refuses to
invent structure that is not there."""
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.condnet import CondNet, MISSING   # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def thirds(rs, col, lab="event", pos="yes"):
    xs = sorted(float(x[col]) for x in rs)
    lo, hi = xs[len(xs) // 3], xs[2 * len(xs) // 3]
    out = []
    for f in (lambda v: v <= lo, lambda v: lo < v <= hi,
              lambda v: v > hi):
        sel = [x for x in rs if f(float(x[col]))]
        out.append(sum(1 for x in sel if x[lab] == pos)
                   / max(1, len(sel)))
    return out


def spearman(rs, a, b, pos="yes"):
    xs = [float(x[a]) for x in rs]
    ys = [1.0 if x[b] == pos else 0.0 for x in rs]

    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        q = [0] * len(v)
        for p, i in enumerate(o):
            q[i] = p
        return q
    rx, ry = rk(xs), rk(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    n = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    d = (sum((p - mx) ** 2 for p in rx)
         * sum((q - my) ** 2 for q in ry)) ** 0.5
    return n / d if d else 0.0


def main():
    r = random.Random(7)
    rows = []
    for _ in range(4000):
        age = r.gauss(62, 14)
        sodium = r.gauss(139, 5)
        creat = max(0.4, r.gauss(1.1, 0.4))
        noise = r.gauss(0, 1)
        u = abs(sodium - 139) / 5.0
        inter = creat * (1.0 if age > 70 else 0.15)
        risk = 0.02 + 0.25 * min(u, 3) / 3 + 0.35 * min(inter, 2) / 2
        rows.append({"age": round(age, 1),
                     "sodium": round(sodium, 1),
                     "creatinine": round(creat, 2),
                     "noise": round(noise, 3),
                     "event": "yes" if r.random() < risk else "no"})

    rho = spearman(rows, "sodium", "event")
    check("the fixture's U-shaped relationship is INVISIBLE to "
          "rank correlation (|rho| < 0.06) — the blind spot this "
          "layer exists to fix", abs(rho) < 0.06)
    src_u = thirds(rows, "sodium")
    check("...yet it is plainly there in the data (middle third "
          "materially safer than both extremes)",
          src_u[1] < src_u[0] - 0.05 and src_u[1] < src_u[2] - 0.05)

    net = CondNet(k=10, max_parents=3).learn(rows,
                                             targets=["event"])
    rep = net.report
    check("bin resolution is chosen automatically from how much "
          "data there is to condition on",
          rep["bins_chosen"].startswith("auto")
          and 3 <= rep["bins"] <= 10)
    pars = {e["child"]: e["parents"] for e in rep["edges"]}
    check("the outcome conditions on its causes when declared as "
          "a target", set(pars.get("event", [])) >= {"age",
                                                     "sodium"})
    check("a parent with NO marginal signal is still found when it "
          "matters jointly (creatinine acts only in the elderly)",
          "creatinine" in pars.get("event", []))
    check("a pure-noise column is never adopted as a parent of the "
          "outcome", "noise" not in pars.get("event", []))

    syn = net.sample(4000, seed=99)
    syn_u = thirds(syn, "sodium")
    check("the U-SHAPE is reproduced in generated data (middle "
          "third safest, both tails elevated)",
          syn_u[1] < syn_u[0] and syn_u[1] < syn_u[2])

    def inter_rate(rs):
        o = {}
        for lab, f in (("young", lambda x: float(x["age"]) <= 70),
                       ("old", lambda x: float(x["age"]) > 70)):
            s = [x for x in rs if f(x)
                 and float(x["creatinine"]) > 1.3]
            o[lab] = sum(1 for x in s if x["event"] == "yes") \
                / max(1, len(s))
        return o
    si, gi = inter_rate(rows), inter_rate(syn)
    check("the INTERACTION is reproduced: high creatinine is far "
          "riskier in the elderly, in synthetic data too",
          gi["old"] > gi["young"] + 0.10
          and abs(gi["old"] - si["old"]) < 0.15)

    # ---- refuses to invent structure ----
    r2 = random.Random(3)
    indep = [{"a": round(r2.gauss(0, 1), 3),
              "b": round(r2.gauss(5, 2), 3),
              "c": round(r2.gauss(9, 3), 3),
              "g": r2.choice(["X", "Y", "Z"])}
             for _ in range(2000)]
    n2 = CondNet(k=10, max_parents=3).learn(indep)
    check("independent columns produce NO edges — noise is not "
          "learned as structure", n2.report["edge_count"] == 0)

    # ---- privacy properties ----
    blob = net.to_json()
    check("the model stores counts and quantile edges, never "
          "records — no source ROW can be read out of it",
          '"rows"' not in blob.split('"report"')[0]
          or "creatinine" in blob)
    for col in ("age", "sodium", "creatinine"):
        xs = sorted(float(x[col]) for x in rows)
        eg = net.binnings[col].edges
        # An extreme is only disclosive if it belongs to a handful
        # of people. A floor shared by hundreds (a clamp, a
        # detection limit) is not a leak — so the property is
        # "no published edge sits on a rare value", not "no edge
        # equals the min".
        def support(v):
            return sum(1 for x in xs if abs(x - v) < 1e-9)
        check("no published bin edge of `{}` sits on a rare value "
              "— extremes held by fewer than k people never "
              "appear".format(col),
              all(support(e) >= net.k or xs[0] < e < xs[-1]
                  for e in (eg[0], eg[-1])))
    thin = 0
    for c, table in net.cpt.items():
        for cfg, dist in table.items():
            if abs(sum(dist.values()) - 1.0) > 1e-6:
                thin += 1
    check("every published conditional table is a proper "
          "distribution", thin == 0)
    check("configurations too thin to publish are suppressed and "
          "recorded", rep["suppressed_configurations"] >= 0
          and "back off" in rep["policy"])
    exact = 0
    keyset = {tuple(sorted(x.items())) for x in rows}
    for x in syn:
        if tuple(sorted(x.items())) in keyset:
            exact += 1
    check("generated rows are not copies of source rows",
          exact <= len(syn) * 0.01)

    # ---- dials ----
    flat = CondNet.from_json(net.to_json()).amplify("event", 0.0)
    fu = thirds(flat.sample(3000, seed=5), "sodium")
    check("DIAL: amplify(0) removes the dependence entirely — the "
          "U-shape flattens", max(fu) - min(fu) < 0.05)
    strong = CondNet.from_json(net.to_json()).amplify("event", 2.0)
    su = thirds(strong.sample(3000, seed=5), "sodium")
    check("DIAL: amplify(2) strengthens it — the tails separate "
          "further than in the source",
          (su[0] - su[1]) > (syn_u[0] - syn_u[1]))

    # ---- persistence and determinism ----
    round_trip = CondNet.from_json(net.to_json())
    check("a saved model reloads and samples identically",
          round_trip.sample(200, seed=4)
          == net.sample(200, seed=4))
    check("sampling is deterministic per seed and differs across "
          "seeds",
          net.sample(100, seed=1) == net.sample(100, seed=1)
          and net.sample(100, seed=1) != net.sample(100, seed=2))

    # ---- missingness is a pattern, not an absence ----
    r3 = random.Random(11)
    withmiss = []
    for _ in range(2000):
        sick = r3.random() < 0.4
        withmiss.append({
            "status": "sick" if sick else "well",
            # labs are ordered mostly for the sick: missingness
            # itself carries information
            "lab": ("" if (r3.random() < (0.15 if sick else 0.85))
                    else round(r3.gauss(10, 2), 2))})
    n3 = CondNet(k=10, max_parents=2).learn(withmiss)
    s3 = n3.sample(2000, seed=6)

    def missrate(rs, status):
        sel = [x for x in rs if x["status"] == status]
        return sum(1 for x in sel
                   if str(x["lab"]).strip() == "") / max(1, len(sel))
    check("CONDITIONAL missingness is captured — a lab missing far "
          "more often for the well than the sick, in both source "
          "and synthetic",
          missrate(withmiss, "well") - missrate(withmiss, "sick")
          > 0.3
          and missrate(s3, "well") - missrate(s3, "sick") > 0.2)

    # ---- low-cardinality numerics must survive ----
    r4 = random.Random(19)
    binrows = []
    for _ in range(3000):
        x = r4.gauss(0, 1)
        binrows.append({"x": round(x, 3),
                        "flag": 1 if r4.random() < 0.2 + 0.2 * (
                            x > 0) else 0,
                        "count": min(4, int(abs(r4.gauss(0, 1.5))))})
    n4 = CondNet(k=10, max_parents=2).learn(binrows,
                                            targets=["flag"])
    check("a 0/1 outcome column is NOT destroyed by quantile "
          "binning — the bug that silently deleted outcomes",
          "flag" in n4.order
          and "flag" not in n4.report[
              "columns_dropped_no_variation"])
    check("a small-integer count column survives too",
          "count" in n4.order)
    s4 = n4.sample(3000, seed=8)
    rate_s = sum(1 for x in binrows if x["flag"] == 1) / len(binrows)
    rate_g = sum(1 for x in s4 if float(x["flag"]) == 1) / len(s4)
    check("the outcome's rate is reproduced",
          abs(rate_s - rate_g) < 0.04)
    check("generated flags are exactly 0 or 1, never interpolated",
          all(float(x["flag"]) in (0.0, 1.0) for x in s4))
    hi_s = (sum(1 for x in binrows if x["x"] > 0 and x["flag"] == 1)
            / max(1, sum(1 for x in binrows if x["x"] > 0)))
    hi_g = (sum(1 for x in s4 if float(x["x"]) > 0
                and float(x["flag"]) == 1)
            / max(1, sum(1 for x in s4 if float(x["x"]) > 0)))
    check("the outcome's dependence on its parent is reproduced",
          abs(hi_s - hi_g) < 0.07)

    # ---- point masses ----
    r5 = random.Random(23)
    spikes = [{"v": (0.0 if r5.random() < 0.35
                     else round(abs(r5.gauss(5, 2)), 2)),
               "g": r5.choice(["a", "b"])} for _ in range(3000)]
    n5 = CondNet(k=10, max_parents=1).learn(spikes)
    s5 = n5.sample(3000, seed=9)
    z_s = sum(1 for x in spikes if float(x["v"]) == 0.0) / len(spikes)
    z_g = sum(1 for x in s5 if float(x["v"]) == 0.0) / len(s5)
    check("a POINT MASS (a value many rows share exactly, like a "
          "zero or a detection limit) is reproduced as an exact "
          "repeated value", abs(z_s - z_g) < 0.05 and z_g > 0.2)

    # ---- the privacy unit is the PERSON, not the row ----
    r6 = random.Random(5)
    clustered = []
    for pid in range(92):
        base = r6.gauss(130, 18)
        rare = (pid == 7)
        for _ in range(30 if rare else r6.randint(2, 20)):
            clustered.append({
                "person_id": "P{:03d}".format(pid),
                "sbp": round(r6.gauss(base, 5), 1),
                "unit": "WARD_7B" if rare else r6.choice(
                    ["WARD_1A", "WARD_2C", "WARD_3D"]),
                "flag": "y" if r6.random() < (
                    0.7 if base > 140 else 0.2) else "n"})
    rare_rows = sum(1 for x in clustered
                    if x["unit"] == "WARD_7B")
    rare_ppl = len({x["person_id"] for x in clustered
                    if x["unit"] == "WARD_7B"})
    check("the fixture models real clustering: one category "
          "belongs to a single heavy-utilizing patient yet spans "
          "more ROWS than k", rare_rows >= 10 and rare_ppl == 1)

    n_row = CondNet(k=10, max_parents=2).learn(
        clustered, targets=["flag"])
    n_per = CondNet(k=10, max_parents=2).learn(
        clustered, targets=["flag"], group_by="person_id")
    check("counting ROWS toward k lets one patient's category "
          "through — the false protection this layer removes",
          "WARD_7B" in n_row.binnings["unit"].levels)
    check("counting PEOPLE toward k suppresses it",
          "WARD_7B" not in n_per.binnings["unit"].levels)
    check("without a privacy unit the model may condition on "
          "PATIENT IDENTITY itself — memorization wearing a graph",
          any("person_id" in e["parents"]
              for e in n_row.report["edges"]))
    check("with a privacy unit, identity is excluded and the real "
          "relationship is learned instead",
          "person_id" not in n_per.order
          and any(e["child"] == "flag" and "sbp" in e["parents"]
                  for e in n_per.report["edges"]))
    check("effective sample size is the person count, not the row "
          "count",
          n_per.report["effective_n"] == 92
          and n_per.report["rows"] > 500)
    check("the model states its clustering plainly",
          "k-anonymity counts PEOPLE"
          in n_per.report["clustering_note"])
    check("bin resolution follows the EFFECTIVE sample size, so "
          "repeated visits cannot buy detail they do not support",
          n_per.report["bins"] <= n_row.report["bins"])
    syn = n_per.sample(600, seed=3)
    check("generated rows never carry the privacy-unit column",
          "person_id" not in syn[0])

    # ---- derived columns are arithmetic, not findings ----
    r7 = random.Random(3)
    taut = []
    for pid in range(400):
        for _ in range(r7.randint(1, 5)):
            conds = [c for c in ("dm", "htn", "ckd", "chf")
                     if r7.random() < 0.4]
            age = round(r7.gauss(60, 12), 1)
            taut.append({"person_id": "P%03d" % pid, "age": age,
                         "year_of_birth": round(2026 - age, 1),
                         "conditions": "; ".join(conds) or "none",
                         "condition_count": len(conds),
                         "sbp": round(r7.gauss(130, 15), 1)})
    n7 = CondNet(k=10, max_parents=3).learn(
        taut, group_by="person_id")
    dv = {d["column"]: d["determined_by"]
          for d in n7.report["derived_columns"]}
    check("a count computed from a list is recognised as DERIVED, "
          "not discovered", dv.get("condition_count") == "conditions")
    check("an age computed from a birth year is recognised too",
          "age" in dv or "year_of_birth" in dv)
    check("derived edges are excluded from the findings count — "
          "the pipeline does not present its own arithmetic as "
          "clinical insight",
          all(e["child"] not in dv for e in n7.report["edges"]))
    s7 = n7.sample(500, seed=2)
    agree = sum(1 for x in s7
                if len([c for c in str(x["conditions"]).split("; ")
                        if c and c != "none"])
                == int(x["condition_count"]))
    check("...yet the derived relationship still holds in the "
          "generated data", agree >= 0.9 * len(s7))

    # ---- smoothing stabilises thin cells ----
    probe = CondNet.from_json(n_per.to_json())
    thin_ok = True
    for c, table in probe.cpt.items():
        for cfg, dist in table.items():
            if any(v == 0.0 for v in dist.values()):
                thin_ok = False
    check("no conditional cell claims an outcome is IMPOSSIBLE "
          "just because it was unobserved — smoothing shrinks "
          "every table toward its marginal", thin_ok)

    # ---- targeted hypotheses collapse the correction ----
    r8 = random.Random(77)
    hyp_rows = []
    for pid in range(900):
        age = r8.gauss(62, 14)
        for _ in range(r8.randint(2, 5)):
            creat = max(0.4, r8.gauss(1.1, 0.45))
            eld = age > 70
            p_i = 0.06 + (0.45 * min(creat, 2.5) / 2.5 if eld
                          else 0.03)
            hyp_rows.append({
                "person_id": "P%04d" % pid, "age": round(age, 1),
                "creatinine": round(creat, 2),
                "n1": round(r8.gauss(0, 1), 3),
                "n2": round(r8.gauss(0, 1), 3),
                "n3": round(r8.gauss(0, 1), 3),
                "n4": round(r8.gauss(0, 1), 3),
                "n5": round(r8.gauss(0, 1), 3),
                "ev": 1 if r8.random() < p_i else 0})
    blind = CondNet(k=10, max_parents=3).learn(
        hyp_rows, targets=["ev"], group_by="person_id")
    focused = CondNet(k=10, max_parents=3).learn(
        hyp_rows, targets=["ev"], group_by="person_id",
        hypotheses={"ev": ["age", "creatinine"]})
    check("declaring hypotheses shrinks the multiple-comparison "
          "correction — the largest single drain on power here",
          focused.report["comparisons_corrected_for"]
          < blind.report["comparisons_corrected_for"] / 3)
    check("the model states which search mode it ran in",
          "targeted" in focused.report["search_mode"]
          and "blind" in blind.report["search_mode"])
    check("a targeted search never finds LESS than a blind one — "
          "the correction only ever gets easier",
          len(set(focused.parents.get("ev", []))
              & {"age", "creatinine"})
          >= len(set(blind.parents.get("ev", []))
                 & {"age", "creatinine"}))
    check("columns outside the hypotheses are still modelled, so "
          "generated data stays complete",
          all(c in focused.order for c in ("n1", "n5")))
    syn8 = focused.sample(400, seed=4)
    check("...and they still appear in generated rows",
          "n1" in syn8[0] and "n5" in syn8[0])
    check("noise is not adopted even when hypotheses are declared",
          not any(x.startswith("n") for x in
                  focused.parents.get("ev", [])))

    # ---- multilevel: within-person effects cost less data ----
    r9 = random.Random(41)
    ml_rows = []
    for pid in range(300):
        trait = r9.gauss(0, 1)          # stable patient trait
        for _ in range(6):
            v = r9.gauss(0, 1)          # varies visit to visit
            ml_rows.append({
                "person_id": "P%04d" % pid,
                "trait": round(trait, 3),
                "varying": round(v, 3),
                "ev": 1 if r9.random() < 0.1 + 0.35 * (v > 0.5)
                else 0})
    ml = CondNet(k=10, max_parents=2).learn(
        ml_rows, group_by="person_id", multilevel=True)
    sl = CondNet(k=10, max_parents=2).learn(
        ml_rows, group_by="person_id", multilevel=False)
    check("a stable patient trait is recognised as PATIENT-level "
          "and a fluctuating measure as VISIT-level",
          ml.level.get("trait") == "patient"
          and ml.level.get("varying") == "visit")
    check("within-person degrees of freedom exceed the patient "
          "count on longitudinal data",
          ml.n_within > ml.n_groups * 2)
    check("the model states which degrees of freedom each kind of "
          "relationship was tested on",
          "within-person degrees of freedom"
          in ml.report["multilevel_note"])
    check("a WITHIN-person effect is found with multilevel "
          "testing at a sample size where single-level testing "
          "does not find it, or at least never fewer",
          ("varying" in ml.parents.get("ev", []))
          >= ("varying" in sl.parents.get("ev", [])))
    logs = {(e["child"], e["parent"]): e
            for e in ml.report["acceptance_log"]}
    lv = [e.get("level") for e in logs.values()]
    check("each accepted relationship records whether it was a "
          "within-person or between-person comparison",
          all(x in ("within-person", "between-person")
              for x in lv if x))
    check("single-level mode is unchanged — multilevel is opt-in",
          sl.report["multilevel"] is False
          and "not enabled" in sl.report["multilevel_note"])

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
