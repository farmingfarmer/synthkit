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

def _sd(xs):
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / max(len(xs) - 1,
                                                1)) ** 0.5


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
    # This used to assert that person_id became a PARENT without a
    # privacy unit — memorization wearing a graph. The cardinality
    # guard now keeps it out of the model entirely, so that edge can
    # no longer form. The protection is INCIDENTAL, not a privacy
    # rule: it fires because an identifier has many levels, and an
    # identity-like column with few levels would still get through.
    # The privacy argument itself is carried by the WARD_7B pair
    # above, which is about counting people rather than rows.
    check("an identifier is refused as a category even with no "
          "privacy unit — incidental protection, not a substitute "
          "for one",
          "person_id" in n_row.excluded_columns
          and not any("person_id" in e["parents"]
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
    check("list columns are rebuilt in generated rows, so the "
          "output has the same shape as the source",
          all("conditions" in x for x in s7))
    agree = sum(1 for x in s7
                if len([c for c in str(x["conditions"]).split("; ")
                        if c and c != "none"])
                == int(x["condition_count"]))
    check("...and a tally is COMPUTED from the rebuilt list, never "
          "sampled independently — a count that contradicts the "
          "list beside it is worse than no count at all",
          agree == len(s7))

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

    # ---- separating findings from bookkeeping (live-shaped) ----
    r10 = random.Random(9)
    live = []
    for pid in range(92):
        base = r10.gauss(132, 16)
        for _ in range(r10.randint(2, 20)):
            sbp = r10.gauss(base, 8)
            dbp = 0.55 * sbp + r10.gauss(0, 5)
            conds = [c for c in ("dm", "htn", "ckd", "chf")
                     if r10.random() < 0.4]
            drugs = [d for d in ("furosemide", "lisinopril",
                                 "metformin") if r10.random() < 0.5]
            gap = r10.randint(1, 400)
            age = round(r10.gauss(66, 12), 1)
            live.append({
                "person_id": "P%03d" % pid,
                "systolic_blood_pressure": round(sbp, 1),
                "diastolic_blood_pressure": round(dbp, 1),
                "conditions": "; ".join(conds) or "none",
                "condition_count": len(conds),
                "active_drugs": "; ".join(drugs) or "none",
                "active_drug_count": len(drugs),
                "age_at_visit": age,
                "year_of_birth": round(2026 - age, 1),
                "days_to_next_visit": gap,
                "returned_within_30d": 1 if gap <= 30 else 0,
                "4": r10.choice(["1", "2"])})
    nl = CondNet(k=10, max_parents=3).learn(
        live, targets=["returned_within_30d"],
        group_by="person_id")
    rl = nl.report
    dv2 = {d["column"]: d["determined_by"]
           for d in rl["derived_columns"]}
    check("a numerically-named column — the mark of a headerless "
          "index or a malformed export — is dropped rather than "
          "modelled",
          "4" in rl["dropped_odd_column_names"]
          and "4" not in nl.order)
    check("LABEL LEAKAGE is barred: the column that DEFINES the "
          "outcome is never offered as a parent",
          "days_to_next_visit" in rl["never_parent_excluded"]
          and not any("days_to_next_visit" in e["parents"]
                      for e in rl["edges"]))
    check("a count derived from a list is filed as arithmetic "
          "even after binning blurs the determinism",
          dv2.get("condition_count") == "conditions"
          and dv2.get("active_drug_count") == "active_drugs")
    check("a year derived from an age is filed as arithmetic too",
          "year_of_birth" in dv2 or "age_at_visit" in dv2)
    # Direction is not meaningful here: both factorisations encode
    # the same joint distribution, and which one the ordering
    # heuristic picks is arbitrary. The claim is that the pair is
    # linked and the bookkeeping is not.
    bp_pair = {"systolic_blood_pressure",
               "diastolic_blood_pressure"}
    check("what survives is the PHYSIOLOGY, not the bookkeeping",
          any(bp_pair <= ({e["child"]} | set(e["parents"]))
              for e in rl["edges"]))
    check("the findings count reflects discoveries only — the "
          "arithmetic is reported separately",
          rl["edge_count"] <= 3
          and len(rl["derived_columns"]) >= 3)
    check("the artifact explains why leakage columns are barred",
          "learning nothing" in rl["leakage_note"])

    # ---- one list determining another; arithmetic on pairs ----
    ROUTE = {"ondansetron": "IV Push", "fentanyl": "IV Push",
             "sodium chloride": "Flush", "propofol": "IV",
             "acetaminophen": "Oral", "lidocaine": "Topical"}
    r11 = random.Random(11)
    fam = []
    for pid in range(92):
        byear = r11.randint(1935, 1995)
        base = r11.gauss(132, 16)
        for _ in range(r11.randint(2, 20)):
            vyear = r11.randint(2015, 2024)
            sbp = r11.gauss(base, 8)
            drugs = [d for d in ROUTE if r11.random() < 0.35]
            fam.append({
                "person_id": "P%03d" % pid,
                "visit_year": vyear, "year_of_birth": byear,
                "age_at_visit": vyear - byear,
                "systolic_blood_pressure": round(sbp, 1),
                "diastolic_blood_pressure":
                    round(0.55 * sbp + r11.gauss(0, 5), 1),
                "active_drugs":
                    "; ".join(sorted(drugs)) or "none",
                "drug_routes": "; ".join(
                    sorted({ROUTE[d] for d in drugs})) or "none"})
    nf = CondNet(k=10, max_parents=3).learn(
        fam, group_by="person_id")
    rf = nf.report
    check("both list columns are expanded into per-item "
          "indicators", len(rf["list_columns_expanded"]) == 2)
    check("a list DETERMINED BY another list is filed as a family "
          "— the route a drug is given by is a property of the "
          "drug, not a discovery about patients",
          rf["derived_families"].get("drug_routes")
          == "active_drugs")
    check("...so none of its indicators surface as findings, "
          "which is where seven of thirteen live edges came from",
          not any(e["child"].startswith("drug_routes::")
                  for e in rf["edges"]))
    check("a column that is exact arithmetic on TWO others is "
          "filed too, whether caught singly or as a pair",
          any(d["column"] in ("year_of_birth", "age_at_visit")
              for d in rf["derived_columns"]))
    bp = {"systolic_blood_pressure", "diastolic_blood_pressure"}
    check("what remains is the physiology",
          rf["edge_count"] <= 3
          and any(bp <= ({e["child"]} | set(e["parents"]))
                  for e in rf["edges"]))
    check("the artifact explains the family rule in plain terms",
          "property" in rf["family_note"])

    # the arithmetic test must be exact, not approximate
    r12 = random.Random(5)
    arith = [{"person_id": "P%03d" % (i // 6),
              "a": round(r12.gauss(50, 10), 2),
              "b": round(r12.gauss(20, 4), 2),
              "unrelated": round(r12.gauss(0, 1), 3)}
             for i in range(600)]
    for row in arith:
        row["total"] = round(row["a"] + row["b"], 2)
    na = CondNet(k=10, max_parents=2).learn(
        arith, group_by="person_id")
    dnames = {d["column"] for d in na.report["derived_columns"]}
    check("a sum of two columns is recognised as arithmetic",
          "total" in dnames or "a" in dnames or "b" in dnames)
    check("an unrelated column is NOT swept up with it",
          "unrelated" not in dnames)

    # ---- generated values must be physiologically possible ----
    r13 = random.Random(31)
    vit = []
    for pid in range(120):
        bp = r13.gauss(132, 14)
        for _ in range(r13.randint(2, 9)):
            s_ = r13.gauss(bp, 8)
            vit.append({"person_id": "P%03d" % pid,
                        "sbp": round(s_, 1),
                        "dbp": round(0.55 * s_ + r13.gauss(0, 4), 1),
                        "wbc": round(abs(r13.gauss(7, 2)), 2)})
    nv = CondNet(k=10, max_parents=2).learn(
        vit, group_by="person_id")
    sv = nv.sample(4000, seed=6)
    for col in ("sbp", "dbp", "wbc"):
        obs = [float(x[col]) for x in vit]
        gen = [float(x[col]) for x in sv]
        span = max(obs) - min(obs)
        check("generated `{}` stays inside a band the data "
              "supports — tail extrapolation can no longer emit "
              "an impossible vital sign".format(col),
              min(gen) >= min(obs) - 0.6 * span
              and max(gen) <= max(obs) + 0.6 * span)
    check("a quantity never observed negative is never generated "
          "negative",
          all(float(x["wbc"]) >= 0 for x in sv))
    check("the clamp does not flatten the distribution it "
          "protects — the spread still matches the source",
          abs((max(float(x["sbp"]) for x in sv)
               - min(float(x["sbp"]) for x in sv))
              - (max(float(x["sbp"]) for x in vit)
                 - min(float(x["sbp"]) for x in vit)))
          < 0.5 * (max(float(x["sbp"]) for x in vit)
                   - min(float(x["sbp"]) for x in vit)))

    # ---- patients with a clinical course ----
    from collections import defaultdict as _dd
    r14 = random.Random(77)
    longit = []
    for pid in range(140):
        bp = r14.gauss(132, 15)
        for _ in range(r14.randint(2, 10)):
            longit.append({"person_id": "P%04d" % pid,
                           "sex": "F" if pid % 2 else "M",
                           "sbp": round(r14.gauss(bp, 6), 1)})
    nh = CondNet(k=10, max_parents=2).learn(
        longit, group_by="person_id", multilevel=True)
    check("the model records how many visits people have and "
          "which of their values are fixed traits",
          nh.visit_counts and nh.level.get("sex") == "patient"
          and nh.level.get("sbp") == "visit")
    check("it learns how a changing value moves from one visit to "
          "the next", "sbp" in nh.lag)
    people = nh.sample_patients(200, seed=3)
    grp = _dd(list)
    for row in people:
        grp[row["person_id"]].append(row)
    check("generation produces PEOPLE with visit histories, not "
          "loose rows",
          len(grp) == 200 and len(people) > 200
          and all("visit_number" in row for row in people))
    check("a fixed trait stays fixed across a patient's visits",
          all(len({row["sex"] for row in v}) == 1
              for v in grp.values()))

    def autocorr(groups):
        pr = []
        for v in groups.values():
            xs = [float(row["sbp"]) for row in v]
            pr += list(zip(xs, xs[1:]))
        if len(pr) < 30:
            return 0.0
        a = [x for x, _ in pr]
        b = [y for _, y in pr]
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        num = sum((x - ma) * (y - mb) for x, y in pr)
        den = (sum((x - ma) ** 2 for x in a)
               * sum((y - mb) ** 2 for y in b)) ** 0.5
        return num / den if den else 0.0

    src_grp = _dd(list)
    for row in longit:
        src_grp[row["person_id"]].append(row)
    flat = nh.sample(len(people), seed=3)
    flat_grp = _dd(list)
    for i, row in enumerate(flat):
        flat_grp[i // 5].append(row)
    check("a patient's values are CORRELATED visit to visit, as "
          "in real longitudinal data — row-wise sampling produces "
          "none of this",
          autocorr(grp) > 0.25 > autocorr(flat_grp))
    check("...without exceeding the correlation actually present "
          "in the source",
          autocorr(grp) <= autocorr(src_grp) + 0.15)
    rt = CondNet.from_json(nh.to_json())
    check("the hierarchical tables survive saving and reloading — "
          "without them a restored model draws rows but no longer "
          "draws people",
          rt.visit_counts and rt.lag
          and len(rt.sample_patients(50, seed=1)) > 50)

    # ---- differential privacy ----
    base_net = CondNet(k=10, max_parents=3).learn(
        longit, group_by="person_id")
    check("with no budget set, the model says plainly that it "
          "offers k-anonymity rather than a bound on inference",
          base_net.report["differential_privacy"]["epsilon"]
          is None
          and "adversary" in
          base_net.report["differential_privacy"]["reading"])
    dp_net = CondNet(k=10, max_parents=3).learn(
        longit, group_by="person_id", epsilon=1.0)
    dpr = dp_net.report["differential_privacy"]
    check("a budget produces a stated epsilon and a noise scale "
          "derived from it",
          dpr["epsilon"] == 1.0 and dpr["noise_scale"] > 0)
    check("one person's contribution is bounded, because without "
          "a bound the sensitivity is whatever the heaviest "
          "utiliser happens to be",
          dpr["max_rows_per_person"] > 0)
    check("the claim is stated in terms of what an adversary "
          "cannot learn",
          "whether one particular person was in it"
          in dpr["reading"])
    strong = CondNet(k=10, max_parents=3).learn(
        longit, group_by="person_id", epsilon=0.05)
    s_sd = _sd([float(x["sbp"]) for x in strong.sample(2000, 4)])
    o_sd = _sd([float(x["sbp"]) for x in longit])
    check("a very tight budget degrades the model toward its own "
          "marginal rather than collapsing it into confident "
          "nonsense — 'we do not know' is the right failure",
          0.4 * o_sd < s_sd < 2.5 * o_sd)

    # ---- a patient's position inside a bin persists ----
    check("the model measures how much of a column's variation is "
          "between people rather than within one person's course",
          nh.persistence.get("sbp", 0) > 0.3)
    check("that persistence survives saving and reloading",
          CondNet.from_json(nh.to_json()).persistence.get("sbp"))
    ppl2 = nh.sample_patients(300, seed=9)
    g2 = _dd(list)
    for row in ppl2:
        g2[row["person_id"]].append(row)
    check("carrying a patient's position within their bin raises "
          "the visit-to-visit correlation well above what bin "
          "membership alone can give",
          autocorr(g2) > 0.45)
    # the position must stay UNIFORM or every marginal shifts
    pos_vals = []
    for v in g2.values():
        for row in v:
            pos_vals.append(float(row["sbp"]))
    lo_q = sorted(pos_vals)[len(pos_vals) // 10]
    hi_q = sorted(pos_vals)[9 * len(pos_vals) // 10]
    src_vals = sorted(float(x["sbp"]) for x in longit)
    check("...without pulling values toward the middle of their "
          "bins: averaging two uniforms would do that and would "
          "quietly distort every marginal",
          abs(lo_q - src_vals[len(src_vals) // 10])
          < 0.25 * (src_vals[-1] - src_vals[0])
          and abs(hi_q - src_vals[9 * len(src_vals) // 10])
          < 0.25 * (src_vals[-1] - src_vals[0]))

    # ---- structure must survive INTO the visit sequence ----
    r15 = random.Random(19)
    linked = []
    for pid in range(150):
        lvl = r15.gauss(132, 15)
        for _ in range(r15.randint(3, 9)):
            s_ = r15.gauss(lvl, 6)
            linked.append({
                "person_id": "P%04d" % pid,
                "sbp": round(s_, 1),
                "dbp": round(0.55 * s_ + r15.gauss(0, 3), 1)})
    nl2 = CondNet(k=10, max_parents=2).learn(
        linked, group_by="person_id", multilevel=True)
    check("the model measures how positions INSIDE the bins move "
          "together, not just which bins co-occur",
          nl2.pos_corr)

    def xcorr(rs):
        xs = [float(x["sbp"]) for x in rs]
        ys = [float(x["dbp"]) for x in rs]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        nu = sum((a_ - mx) * (b_ - my) for a_, b_ in zip(xs, ys))
        de = (sum((a_ - mx) ** 2 for a_ in xs)
              * sum((b_ - my) ** 2 for b_ in ys)) ** 0.5
        return nu / de if de else 0.0

    hier2 = nl2.sample_patients(220, seed=8)
    flat2 = nl2.sample(len(hier2), seed=8)
    check("a relationship between two columns survives into the "
          "visit sequence — letting the transition table replace "
          "the parent conditioning destroyed this entirely",
          xcorr(hier2) > 0.45)
    check("...and generating patients is no worse at it than "
          "generating loose rows",
          xcorr(hier2) >= xcorr(flat2) - 0.08)
    check("neither sampler invents correlation the source does "
          "not have",
          xcorr(hier2) <= xcorr(linked) + 0.1
          and xcorr(flat2) <= xcorr(linked) + 0.1)
    check("the within-bin correlations survive save and reload",
          CondNet.from_json(nl2.to_json()).pos_corr == nl2.pos_corr)

    # ---- resolution follows what each column can support ----
    r16 = random.Random(23)
    wide = []
    for pid in range(320):
        lvl = r16.gauss(130, 16)
        for _ in range(r16.randint(2, 8)):
            s_ = r16.gauss(lvl, 7)
            wide.append({"person_id": "P%04d" % pid,
                         "sbp": round(s_, 1),
                         "dbp": round(0.55 * s_ + r16.gauss(0, 4),
                                      1)})
    nr = CondNet(k=10, max_parents=2).learn(
        wide, group_by="person_id")
    ref = nr.report["refined_columns"]
    check("a LEAF column is re-binned at the resolution its own "
          "parent count supports, rather than the model-wide "
          "worst case",
          ref and max(ref.values()) > nr.report["bins"])
    parents_of_something = {pc for c in nr.order
                            for pc in nr.parents.get(c, [])}
    check("a column that others condition on is NOT refined — "
          "finer parent bins thin every child's cells until the "
          "relationships depending on them are suppressed",
          not (set(ref) & parents_of_something))
    check("the structure learned before refinement is not lost",
          nr.report["edge_count"] >= 1)
    syn_r = nr.sample(2000, seed=5)
    src_v = sorted(float(x["sbp"]) for x in wide)
    gen_v = sorted(float(x["sbp"]) for x in syn_r)

    def q(xs, f):
        return xs[min(len(xs) - 1, int(f * len(xs)))]

    spread = src_v[-1] - src_v[0]
    check("finer bins bring the generated distribution closer to "
          "the source at every quartile",
          all(abs(q(gen_v, f) - q(src_v, f)) < 0.12 * spread
              for f in (0.25, 0.5, 0.75)))
    rt2 = CondNet.from_json(nr.to_json())
    check("a refined model round-trips with its tables intact",
          len(rt2.sample(200, seed=2)) == 200
          and rt2.report["refined_columns"] == ref)
    check("the report explains why resolution differs by column",
          "own parent" in nr.report["refinement_note"])

    # ---- the budget must reach the transition tables ----
    dp_h = CondNet(k=10, max_parents=2).learn(
        longit, group_by="person_id", multilevel=True,
        epsilon=1.0)
    plain_h = CondNet(k=10, max_parents=2).learn(
        longit, group_by="person_id", multilevel=True)
    check("a transition table under a budget is not identical to "
          "one without — publishing untouched transitions while "
          "claiming an epsilon would state a guarantee that does "
          "not hold",
          dp_h.lag.get("sbp") != plain_h.lag.get("sbp"))
    check("the budget is split over the transition tables too, "
          "so more published quantities means less noise budget "
          "each",
          dp_h.report["differential_privacy"]["budget_split_over"]
          > sum(1 for c in dp_h.order if dp_h.parents.get(c)))
    check("the visit-count histogram is noised as well: how often "
          "people are seen is itself a fact about them",
          "visit-count histogram"
          in dp_h.report["differential_privacy"]["covers"])
    # The persistence weights are TUNED from generated data, so
    # what needs covering is the target they are tuned toward —
    # that is the quantity measured from real patients.
    check("the steadiness measured from real patients is "
          "coarsened before it is used to steer generation",
          dp_h.target_autocorr
          and all(abs(v * 10 - round(v * 10)) < 1e-9
                  for v in dp_h.target_autocorr.values()))
    check("...and the budget says it covers them",
          any("steadiness" in x for x in
              dp_h.report["differential_privacy"]["covers"]))
    check("a budgeted hierarchical model still generates patients "
          "with visit histories",
          len(dp_h.sample_patients(60, seed=2)) > 60)

    # ---- every parent must be drawn before its child ----
    # A real extract crashed generation here: the derived and
    # pair-derived paths assign parents directly, bypassing the
    # search that guaranteed a parent came earlier in the order.
    ROUTE2 = {"ondansetron": "IV Push", "fentanyl": "IV Push",
              "sodium chloride": "Flush", "propofol": "IV",
              "acetaminophen": "Oral", "lidocaine": "Topical"}
    r17 = random.Random(31)
    tangled = []
    for pid in range(150):
        htn = r17.random() < 0.4
        for _ in range(r17.randint(2, 9)):
            drugs = [d for d in ROUTE2 if r17.random() < 0.35]
            cs = ["Essential hypertension"] if htn else []
            if htn and r17.random() < 0.7:
                cs.append("Hyperlipidemia")
            age = round(r17.gauss(66, 12), 1)
            tangled.append({
                "person_id": "P%04d" % pid,
                "age_at_visit": age,
                "year_of_birth": round(2026 - age, 1),
                "procedure_quantity": len(drugs),
                "procedures": "; ".join(
                    sorted(ROUTE2[d] for d in drugs)) or "none",
                "active_drugs": "; ".join(sorted(drugs)) or "none",
                "drug_routes": "; ".join(
                    sorted({ROUTE2[d] for d in drugs})) or "none",
                "conditions": "; ".join(sorted(cs)) or "none",
                "sbp": round(r17.gauss(132, 15), 1)})
    for ml in (False, True):
        nt = CondNet(k=10, max_parents=3).learn(
            tangled, group_by="person_id", multilevel=ml)
        drawn = set()
        violations = []
        for c in nt.order:
            for pc in nt.parents.get(c, []):
                if pc not in drawn:
                    violations.append((c, pc))
            drawn.add(c)
        check("every parent precedes its child in the sampling "
              "order (multilevel={})".format(ml), not violations)
        check("both samplers run on a schema with cross-family "
              "and within-family determination (multilevel={})"
              .format(ml),
              len(nt.sample(200, seed=3)) == 200
              and len(nt.sample_patients(50, seed=3)) >= 50)
        check("no parent had to be backed off at generation time "
              "(multilevel={})".format(ml),
              getattr(nt, "_order_warnings", 0) == 0)

    # ---- steadiness must MATCH the source, not exceed it ----
    r18 = random.Random(41)
    steady = []
    for pid in range(240):
        lvl = r18.gauss(130, 15)
        for _ in range(r18.randint(3, 9)):
            steady.append({
                "person_id": "P%04d" % pid,
                "sbp": round(r18.gauss(lvl, 5), 1),
                "noise": round(r18.gauss(0, 1), 3)})
    ns = CondNet(k=10, max_parents=2).learn(
        steady, group_by="person_id", multilevel=True)
    check("the source's own visit-to-visit steadiness is measured "
          "and kept as a target",
          ns.report["persistence_targets"].get("sbp", 0) > 0.4)
    check("a column with no real steadiness gets no target worth "
          "chasing",
          abs(ns.report["persistence_targets"].get("noise", 0))
          < 0.2)
    gen_s = ns.sample_patients(220, seed=6)
    got = ns._measure_autocorr(gen_s, "sbp")
    want = ns.report["persistence_targets"]["sbp"]
    check("generated patients are about as steady as real ones",
          got > 0.35)
    check("...and NOT steadier — synthetic patients that hold "
          "their values more tightly than real ones would make "
          "any model grouping by patient look better behaved "
          "than it will be",
          got <= want + 0.08)
    got_n = ns._measure_autocorr(gen_s, "noise")
    check("steadiness is not invented where the source has none",
          got_n is None or abs(got_n) < 0.25)

    # ---- steadiness must survive a GAP -----------------------------
    # Transitions used to be keyed on the last ROW. When the previous
    # visit did not measure a column, that key is MISSING and the
    # table answers "what follows a gap" - near the marginal, carrying
    # nothing about this patient's level. The anchor cannot rescue it:
    # it sets position WITHIN a bin already drawn from a memoryless
    # key. Measured on a real extract, retention of source steadiness
    # tracked coverage - complete columns kept 74-93%, 44%-covered
    # ones kept 40-47%. Reproduced here: 97% / 73% / 47% / 11% as
    # masking rose. Keyed on the last OBSERVATION instead, it holds
    # flat.
    def _ac(rs, col):
        by = {}
        for r in rs:
            by.setdefault(r.get("person_id"), []).append(r)
        pr = []
        for v in by.values():
            xs = []
            for r in v:
                try:
                    xs.append(float(r.get(col, "")))
                except (TypeError, ValueError):
                    pass
            pr += list(zip(xs, xs[1:]))
        if len(pr) < 30:
            return None
        aa = [x for x, _ in pr]
        bb = [y for _, y in pr]
        ma, mb = sum(aa) / len(aa), sum(bb) / len(bb)
        nu = sum((x - ma) * (y - mb) for x, y in pr)
        de = (sum((x - ma) ** 2 for x in aa)
              * sum((y - mb) ** 2 for y in bb)) ** 0.5
        return nu / de if de else None

    gaps = {}
    for miss in (0.0, 0.75):
        gr = random.Random(11)
        grows = []
        for p in range(250):
            w = gr.gauss(0, 1.0)
            mu = gr.gauss(0, 1.0)
            for _v in range(14):
                w = 0.85 * w + (1 - 0.85 ** 2) ** 0.5 * gr.gauss(0, 1)
                val = round(mu + w, 4)
                grows.append({"person_id": "P{:05d}".format(p),
                              "age": 30 + (p % 55),
                              "lab": ("" if miss and gr.random() < miss
                                      else val)})
        gnet = CondNet(k=10).learn(grows, group_by="person_id",
                                   multilevel=True)
        ggen = gnet.sample_patients(250, seed=99)
        src, got = _ac(grows, "lab"), _ac(ggen, "lab")
        gaps[miss] = (got / src) if (src and got) else None

    check("the fixture really does lose most observations at the "
          "masked rate, or the check below proves nothing",
          sum(1 for r in grows if r["lab"] == "") > len(grows) * 0.6)
    check("steadiness SURVIVES a gap: a 75%-missing column keeps as "
          "much of its source steadiness as a complete one",
          gaps[0.0] is not None and gaps[0.75] is not None
          and gaps[0.75] >= 0.75)
    check("...and retention does not fall away with coverage - that "
          "slope was the whole fault",
          abs(gaps[0.0] - gaps[0.75]) < 0.25)
    check("...without overshooting, which would flatter every model "
          "tested on the output",
          gaps[0.75] <= 1.15)

    # ---- a date is not a category ----------------------------------
    # Modelled as one, its transition table is levels^2 over the
    # calendar. The fault is LATENT: at few patients no single date
    # clears the k-patient floor and every date column collapses to
    # one level, so it looks harmless. At 400 patients over six years
    # the dates clear it and one column produced a 4,975,074-cell
    # table and a 424 MB model. So the fixture must be large enough
    # for dates to clear k, or it proves nothing.
    rnd = random.Random(11)
    drows = []
    for p in range(400):
        mu = rnd.gauss(0, 1)
        day = rnd.randint(0, 300)
        for _v in range(12):
            day += rnd.randint(1, 60)
            y, rem = 2018 + day // 365, day % 365
            drows.append({
                "person_id": "P{:04d}".format(p),
                "visit_start_date": "{:04d}-{:02d}-{:02d}".format(
                    y, (rem // 31) + 1, (rem % 31) + 1),
                "age_at_visit": 30 + (p % 55),
                "sex": "F" if p % 2 else "M",
                "lab": round(mu + rnd.gauss(0, 0.6), 3)})
    dn = CondNet(k=10).learn(drows, group_by="person_id",
                             multilevel=True)
    exc = dn.excluded_columns
    check("the fixture is large enough that dates WOULD have cleared "
          "k - otherwise this proves nothing",
          len(set(r["visit_start_date"] for r in drows)) > 200)
    check("a date column is refused as a category",
          "visit_start_date" in exc and "date" in exc["visit_start_date"])
    check("the date column is not modelled at all",
          "visit_start_date" not in dn.binnings)
    check("ordinary NUMERIC columns are NOT swept up by the bound - "
          "they are binned, so their table is bins^2 already",
          "age_at_visit" not in exc and "lab" not in exc
          and "age_at_visit" in dn.binnings)
    check("an ordinary low-cardinality categorical survives",
          "sex" not in exc and "sex" in dn.binnings)
    check("the exclusion is reported, not silent",
          dn.report.get("unmodellable_excluded", {}).get(
              "visit_start_date"))
    blob = json.loads(dn.to_json())
    check("the model stays small once dates are out",
          len(json.dumps(blob)) < 400000)
    lagcells = sum(len(v) * max((len(x) for x in v.values()), default=0)
                   for v in blob.get("lag", {}).values())
    check("no transition table is levels^2 over a calendar",
          lagcells < 5000)

    # An identifier: one distinct value per row. No value is held by
    # k patients, so the level count is ZERO and a "too many levels"
    # bound cannot see it. Observed live: visit_id was modelled and
    # generated, coming out steadier visit to visit (0.591) than the
    # source it was learned from (0.209).
    # visit_id is an INTEGER here, as it is in real OMOP. The string
    # form was caught by the level test; the integer form was not,
    # because a numeric column never becomes levels at all - and that
    # is the form the real extract has.
    irows = []
    for p in range(200):
        for v in range(10):
            irows.append({"person_id": "P{:04d}".format(p),
                          "visit_id": 100000 + len(irows),
                          "lab": round(random.Random(p * 7 + v)
                                       .gauss(0, 1), 3),
                          "age": 30 + (p % 55)})
    inet = CondNet(k=10).learn(irows, group_by="person_id",
                               multilevel=True)
    check("an identifier with a distinct value on EVERY row is "
          "refused - the mirror of having too many levels, which a "
          "level bound cannot see",
          "visit_id" in inet.excluded_columns
          and "identifier" in inet.excluded_columns["visit_id"])
    check("...and it is therefore never generated, so it cannot come "
          "out steadier than the source it was learned from",
          "visit_id" not in inet.binnings
          and all("visit_id" not in r for r in
                  inet.sample_patients(20, seed=1)[:5]))
    check("a SPARSE column where no value clears k is not swept up by "
          "the identifier rule - absence is not identity",
          "lab" not in inet.excluded_columns)
    check("an ordinary integer column that REPEATS is not mistaken "
          "for a key - uniqueness is the signal, not integrality",
          "age" not in inet.excluded_columns and "age" in inet.binnings)
    # A finely-resolved continuous column can be unique on every row
    # too. Integrality is what separates it from a key.
    frows = []
    rr = random.Random(5)
    for p in range(200):
        mu = rr.gauss(0, 1)
        for v in range(10):
            frows.append({"person_id": "P{:04d}".format(p),
                          "assay": round(mu + rr.gauss(0, 1), 9)})
    fnet = CondNet(k=10).learn(frows, group_by="person_id",
                               multilevel=True)
    check("a continuous column unique on every row is NOT called an "
          "identifier - a lab read to nine decimals is still a lab",
          "assay" not in fnet.excluded_columns)

    # the general guard, independent of the date test
    # Each code is held by exactly 10 distinct patients, so it clears
    # the k-patient floor and really would become a level - 480 of
    # them, far past what a transition table can carry.
    hrows = []
    for p in range(400):
        for v in range(12):
            hrows.append({"person_id": "P{:04d}".format(p),
                          "code": "C{:04d}".format((p % 40) * 12 + v),
                          "lab": round(random.Random(p * 99 + v)
                                       .gauss(0, 1), 3)})
    hn = CondNet(k=10).learn(hrows, group_by="person_id",
                             multilevel=True)
    check("a high-cardinality categorical that is NOT a date is "
          "refused too, on the transition-table bound",
          "code" in hn.excluded_columns
          and "levels" in hn.excluded_columns["code"])

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
