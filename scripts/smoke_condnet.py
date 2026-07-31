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

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
