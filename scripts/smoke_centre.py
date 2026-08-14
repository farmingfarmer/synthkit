"""Smoke: the generated column sits where the source column sat.

THE GAP THIS ATTACKS. The 800-patient run put centre within a tenth of
the column's own spread on 18 of 34 numeric columns. Sixteen columns
were somewhere else entirely and nothing said why.

WHAT IT IS, measured rather than guessed. Generation inverts the
published quantile grid by interpolating between knots, which assumes
the density is UNIFORM between them. Two places that is badly wrong:

  the top segment      the grid's last two knots on a heavy-tailed
                       column were 316.1 at q=0.99 and 2175.0 at the
                       safe maximum. The true mean of that segment is
                       529.0; a straight line draws 1241.7. That ONE
                       segment supplied 7.13 of a 9.97 excess in the
                       column mean - all nine other segments together
                       supplied 2.84

  the rounding cut     a count column 73.8% zeros steps from 0 at
                       q=0.50 to 1 at q=0.75. The line ramps across
                       that quarter of the draw and everything past
                       the halfway point rounds up: 11.3 points of
                       mass moved off zero and the mean read 0.4165
                       against 0.3019

WHAT IS MEASURED AND WHAT IS HYPOTHESIS. `span_days` is the one column
whose failure is confirmed from the run - mean 0.0891 against 0.3503
generated, sd 1.2605, an sd/mean of 14 that is the zero-inflation
signature the rounding cut produces. The heavy-tail mechanism is a
HYPOTHESIS about the other fifteen: it is the dominant error on every
skewed shape tested here, and clinical extracts are full of them, but
which of those sixteen columns it accounts for can only be settled on
the data machine. Both mechanisms are asserted here on shapes that
reproduce them; neither is a claim about how many of the sixteen move.

THE OLD PATH IS RUN SIDE BY SIDE, not described. Every check below
computes what the previous code computed - np.interp then np.round,
which is the whole of the old `_draw_numeric` - and asserts it fails
the same bar the new path clears. A fixture that only exercised the
fix would prove the fix runs, not that it was needed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit import blueprint as B                      # noqa: E402
from synthkit.blueprint import _numeric_marginal         # noqa: E402
from synthkit.discover import (discover,             # noqa: E402
                                prepare)
from synthkit.generate import _draw_numeric, generate    # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


# fidelity.json, run of 2026-08-11: centre_ok is |mean_generated -
# mean_source| <= 0.1 * sd_source, and span_days missed it at
# mean 0.0891 -> 0.3503 against sd 1.2605.
BAR = 0.10
N_PAT, N_VIS = 800, 70
GROUPS = np.repeat(np.arange(N_PAT), N_VIS)
N = len(GROUPS)


def old_draw(m, u):
    """The whole of the previous `_draw_numeric`, kept here so the
    fault is reproduced rather than asserted from memory."""
    out = np.interp(u, np.asarray(m["q"], dtype=float),
                    np.asarray(m["v"], dtype=float))
    return np.round(out) if m.get("integral") else out


def err_of(gen, s):
    sd = float(s.std())
    return abs(float(np.mean(gen)) - float(s.mean())) / sd if sd else 0.0


def main():
    r = np.random.RandomState(7)
    u = r.random_sample(200000)

    shapes = [
        ("heavy tail", r.lognormal(3.0, 1.2, N), True),
        ("count, 74% zero", r.poisson(0.3, N).astype(float), True),
        ("left-skewed", 200 - r.lognormal(3.0, 1.0, N), False),
        ("normal", r.normal(120, 15, N), False),
        ("bimodal", np.concatenate([r.normal(60, 5, N // 2),
                                    r.normal(140, 5, N - N // 2)]),
         False),
    ]
    marg, src = {}, {}
    for name, raw, _ in shapes:
        s = pd.Series(np.asarray(raw, dtype=float))
        src[name] = s
        marg[name] = _numeric_marginal(s, GROUPS, k=10)

    # ---- THE FAULT, RUN NOT DESCRIBED -----------------------------
    broke = [n for n, _, must_fail in shapes if must_fail
             and err_of(old_draw(marg[n], u), src[n]) > BAR]
    check("THE FIXTURE REPRODUCES THE FAULT: the old draw misses the "
          "centre bar on {} - a fixture where it passed would prove "
          "nothing about the fix".format(", ".join(broke) or "nothing"),
          len(broke) == 2)
    check("...and it misses it in the direction the run reported - "
          "generated ABOVE source, which is what span_days did at "
          "0.3503 against 0.0891",
          all(float(np.mean(old_draw(marg[n], u)))
              > float(src[n].mean())
              for n, _, mf in shapes if mf))

    # ---- THE FIX --------------------------------------------------
    for name, _raw, _mf in shapes:
        e = err_of(_draw_numeric(marg[name], u), src[name])
        check("{:<16} centre within {:.0%} of spread ({:.3f})"
              .format(name, BAR, e), e <= BAR)

    # ---- THE NEIGHBOURING PROPERTIES ------------------------------
    # A draw can be made to hit the mean and be wrong everywhere else.
    for name in ("heavy tail", "left-skewed"):
        s, m = src[name], marg[name]
        old_sd = float(np.std(old_draw(m, u)))
        new_sd = float(np.std(_draw_numeric(m, u)))
        check("{}: SPREAD improves too, not just the centre - "
              "{:.1f} against a source {:.1f}, from {:.1f}"
              .format(name, new_sd, float(s.std()), old_sd),
              abs(new_sd - float(s.std())) < abs(old_sd - float(s.std())))

    s, m = src["count, 74% zero"], marg["count, 74% zero"]
    gen = pd.Series(_draw_numeric(m, u))
    tv = s.value_counts(normalize=True)
    gv = gen.value_counts(normalize=True)
    idx = tv.index.union(gv.index)
    tvd_new = 0.5 * float((tv.reindex(idx, fill_value=0)
                           - gv.reindex(idx, fill_value=0)).abs().sum())
    go = pd.Series(old_draw(m, u)).value_counts(normalize=True)
    idx2 = tv.index.union(go.index)
    tvd_old = 0.5 * float((tv.reindex(idx2, fill_value=0)
                           - go.reindex(idx2, fill_value=0)).abs().sum())
    check("THE WHOLE DISTRIBUTION IMPROVES, not only the statistic "
          "being matched: total variation against the source falls "
          "{:.3f} -> {:.3f}. Moving the rounding cut to hit a mean "
          "while leaving the shape wrong would be the failure mode "
          "here".format(tvd_old, tvd_new),
          tvd_new < tvd_old)
    check("...and a whole-number column still holds whole numbers",
          bool(np.all(np.mod(_draw_numeric(m, u), 1.0) == 0.0)))

    # Compared OLD against NEW on the same marginal and the same
    # uniforms, not against an absolute number. A fixed threshold here
    # measures how lumpy this particular fixture draw happened to be -
    # the first version demanded 0.01 and failed at 0.013 on a shape
    # the fix does not touch, which is the fixture's own sampling
    # error and not a regression. Monte-carlo noise on the mean is
    # 1/sqrt(draws) of a standard deviation, so the tolerance is that.
    noise = 1.0 / np.sqrt(len(u))
    unchanged = []
    for name in ("normal", "bimodal"):
        eo = err_of(old_draw(marg[name], u), src[name])
        en = err_of(_draw_numeric(marg[name], u), src[name])
        unchanged.append((name, eo, en, en <= eo + 3 * noise))
    check("a column that was ALREADY right is LEFT ALONE - {} - so "
          "the fix does not pay for skew with a bias everywhere else"
          .format(", ".join("{} {:.4f}->{:.4f}".format(n, a, b)
                            for n, a, b, _ in unchanged)),
          all(ok for _, _, _, ok in unchanged))
    check("generated values still respect the published bound, which "
          "is what keeps one patient's extreme out of the output",
          float(np.max(_draw_numeric(marg["heavy tail"], u)))
          <= marg["heavy tail"]["v"][-1] + 1e-9)

    # ---- AND THE COLUMN STILL BEHAVES IN A REAL TABLE -------------
    n_pat = 260
    rows = []
    rr = np.random.RandomState(3)
    for p in range(n_pat):
        base = float(rr.normal(0, 1))
        for _v in range(6):
            drug = float(rr.lognormal(2.5 + 0.4 * base, 1.1))
            rows.append({
                "person_id": "P{:04d}".format(p),
                "dose": round(drug, 3),
                "visits_billed": int(rr.poisson(0.4)),
                "weight": round(float(rr.normal(78 + 6 * base, 9)), 2),
            })
    df = pd.DataFrame(rows)
    cat = discover(df, group_by="person_id", seed=4)
    bp = B.build(df, cat, group_by="person_id")
    g = generate(bp, n_patients=n_pat, seed=4)
    for col in ("dose", "visits_billed", "weight"):
        s2 = pd.to_numeric(df[col])
        q2 = pd.to_numeric(g[col], errors="coerce").dropna()
        e = abs(float(q2.mean()) - float(s2.mean())) / float(s2.std())
        check("end to end, {} lands within {:.0%} of spread ({:.3f})"
              .format(col, BAR, e), e <= BAR)
    # ---- SPREAD IS ASSERTED NOW, AND EXPLAINED WHEN IT MISSES ----
    # `6690_2` came out at 27% of its source spread on the real run
    # and passed every check, because its centre was fine. A shortfall
    # is not automatically a fault though: when the variance belongs
    # to fewer than k patients, the published bound removes it on
    # purpose. Both cases are planted here.
    rr2 = np.random.RandomState(8)
    n2 = 400 * 30
    g2 = np.repeat(np.arange(400), 30)
    tame = rr2.normal(50, 8, n2)
    # THREE patients out of 400, well under k=10, so the published
    # bound is the mean of ten patient maxima of which seven are tiny
    # - it lands far below the real top and takes the variance with
    # it. Measured across configurations: five such patients only
    # drops the spread to 86% and would not exercise this at all;
    # three drops it to 15%, which is the shape 6690_2 showed.
    few = np.isin(g2, [0, 1, 2])
    wild = np.where(few, rr2.lognormal(5.2, 1.3, n2),
                    rr2.lognormal(0.6, 0.6, n2))
    df2 = pd.DataFrame({"person_id": ["P{:04d}".format(x) for x in g2],
                        "tame": np.round(tame, 2),
                        "wild": np.round(wild, 3)})
    bp2 = B.build(df2, {"claims": [], "unexplained": [], "skipped": []},
                  group_by="person_id")
    gen2 = generate(bp2, n_patients=400, seed=8)
    from run_discovery import compare as _compare
    fid2 = _compare(df2, gen2, bp2, "person_id", None)
    by = dict((c["column"], c) for c in fid2["columns"])
    check("a well-behaved column keeps its SPREAD, which nothing "
          "checked until now - sd was recorded and never asserted",
          not by["tame"].get("spread_miss"))
    wm = by["wild"].get("spread_miss") or {}
    check("a column whose variance belongs to 3 patients loses it - "
          "spread falls to {:.0%} of source - and the report SAYS SO "
          "instead of passing silently"
          .format(wm.get("ratio", 1.0) if wm else 1.0),
          bool(wm) and wm.get("ratio", 1.0) < 0.5)
    check("...and attributes it to the published bound rather than to "
          "the sampler - {:.0%} of that column's magnitude sits "
          "outside what k allows to be published"
          .format(wm.get("share_of_magnitude_outside_bounds") or 0),
          (wm.get("share_of_magnitude_outside_bounds") or 0) > 0.1)

    # ---- AN ARITHMETIC IDENTITY IS NOT A CORRELATION --------------
    # age is the visit year minus the year of birth. Correlation stays
    # strong when that breaks, both means stay right, and every other
    # check in the fidelity report passes - measured, the identity held
    # on 21.4% of generated rows while the report called it fine.
    rr3 = np.random.RandomState(9)
    npat3, nvis3 = 300, 20
    g3 = np.repeat(np.arange(npat3), nvis3); n3 = len(g3)
    yob = np.repeat(np.round(1962 + 16 * rr3.normal(0, 1, npat3)), nvis3)
    vy = 2010 + np.floor(rr3.random_sample(n3) * 14)
    df3 = pd.DataFrame({"person_id": ["P{:04d}".format(x) for x in g3],
                        "year_of_birth": yob, "visit_year": vy,
                        "age_at_visit": vy - yob,
                        "hr": np.round(78 + 10 * rr3.normal(0, 1, n3))})
    bp3 = B.build(df3, discover(df3, group_by="person_id", seed=3),
                  group_by="person_id")
    g3out = generate(bp3, n_patients=npat3, seed=3)
    fid3 = _compare(df3, g3out, bp3, "person_id", None)
    s3 = fid3["summary"]
    ga = pd.to_numeric(g3out["age_at_visit"], errors="coerce")
    gv = (pd.to_numeric(g3out["visit_year"], errors="coerce")
          - pd.to_numeric(g3out["year_of_birth"], errors="coerce"))
    held = float((ga == gv).mean())
    check("THE FIXTURE BREAKS THE IDENTITY: age equals visit year "
          "minus year of birth on 100% of source rows and {:.1%} of "
          "generated ones".format(held), held < 0.6)
    check("...and every OTHER check still passes on it, which is why "
          "this one had to exist - centre {}/{} and spread {}/{}"
          .format(s3["centre_ok"], s3["numeric"],
                  s3["spread_ok"], s3["numeric"]),
          s3["centre_ok"] == s3["numeric"])
    check("the near-deterministic check CATCHES it: {}/{} identities "
          "held".format(s3["deterministic_kept"],
                        s3["deterministic_compared"]),
          s3["deterministic_compared"] > 0
          and s3["deterministic_kept"] < s3["deterministic_compared"])
    loose = fid3["relationships"]["deterministic_loosened"]
    check("...and names which one, with how far it slipped - {}"
          .format(", ".join("{} <- {} {:.2f}->{:.2f}".format(
              x["child"], x["parent"], x["tightness_source"],
              x["tightness_generated"]) for x in loose[:2])),
          loose and all(x["tightness_source"]
                        > x["tightness_generated"] for x in loose))

    # ---- CATEGORICAL PAIRS WERE NEVER MEASURED AT ALL -----------
    # `_pair_fidelity` coerced both sides to numeric and took
    # Spearman, so a categorical pair became NaN and was skipped in
    # silence. Measured on a perfectly associated pair: ZERO compared.
    # Every categorical relationship the catalogue reports has been
    # going unchecked.
    from run_discovery import _pair_fidelity
    r4 = np.random.RandomState(1)
    n4 = 3000
    sex = r4.choice(["M", "F"], n4)
    site = np.where(sex == "M", r4.choice(["A", "B"], n4, p=[.85, .15]),
                    r4.choice(["A", "B"], n4, p=[.15, .85]))
    df4 = pd.DataFrame({
        "person_id": ["P{:03d}".format(i // 3) for i in range(n4)],
        "sex": sex, "site": site,
        "bmi": np.round(np.where(sex == "M", r4.normal(27, 3, n4),
                                 r4.normal(23, 3, n4)), 1)})
    X4, _i4, _d4, _q4 = prepare(df4, "person_id")
    bp4 = {"columns": {}, "relationships": [
        {"child": "site", "parents": ["sex"], "evidence": {}},
        {"child": "bmi", "parents": ["sex"], "evidence": {}}]}
    same = _pair_fidelity(X4, X4, bp4)
    check("categorical and mixed pairs are COMPARED now - {} of them, "
          "where a numeric-only measure compared none"
          .format(same["categorical_compared"]),
          same["categorical_compared"] == 2)
    check("...and an association that survives is reported as kept",
          same["categorical_kept"] == 2)
    shuffled = X4.copy()
    shuffled["sex"] = (shuffled["sex"].sample(frac=1.0, random_state=2)
                       .to_numpy())
    broke = _pair_fidelity(X4, shuffled, bp4)
    check("...and one that is DESTROYED is caught, which a Spearman "
          "on coerced text could never do - {} of {} kept"
          .format(broke["categorical_kept"],
                  broke["categorical_compared"]),
          broke["categorical_kept"] == 0
          and len(broke["categorical_weakened"]) == 2)
    check("...naming the measure each pair was judged by, since a "
          "category has no direction to invert",
          {w["measure"] for w in broke["categorical_weakened"]}
          == {"cramers_v", "correlation_ratio"})
    check("the NUMERIC counters keep their old meaning, so this run "
          "stays comparable with earlier ones",
          same["compared"] == 0 and "sign_kept" in same)

    # ---- A CONSTRAINT IS A STATEMENT ABOUT A ROW ----------------
    # Every other number in the report describes a DISTRIBUTION.
    # "a visit does not end before it begins" describes each row, and
    # nothing could see it break.
    r5 = np.random.RandomState(3)
    npat5, nvis5 = 250, 10
    g5 = np.repeat(np.arange(npat5), nvis5)
    n5 = len(g5)
    start = r5.randint(0, 1500, n5).astype(float)
    span = np.where(r5.random_sample(n5) < 0.12, r5.randint(1, 20, n5), 0)
    df5 = pd.DataFrame({
        "person_id": ["P{:04d}".format(x) for x in g5],
        "visit_start": start, "visit_end": start + span,
        "age_at_visit": np.round(58 + 16 * r5.normal(0, 1, n5)),
        "year_of_birth": np.round(1962 + 16 * r5.normal(0, 1, n5))})
    bp5 = B.build(df5, {"claims": [], "unexplained": [], "skipped": []},
                  group_by="person_id")
    cons = bp5.get("constraints") or []
    check("an ordering the source never breaks is DISCOVERED - {}"
          .format(", ".join("{} <= {}".format(c["lhs"], c["rhs"])
                            for c in cons) or "none"),
          any(c["lhs"] == "visit_start" and c["rhs"] == "visit_end"
              for c in cons))
    check("...and a pair on DISJOINT scales is not mistaken for one: "
          "age is below year_of_birth on every row and means nothing",
          not any({c["lhs"], c["rhs"]} == {"age_at_visit",
                                           "year_of_birth"}
                  for c in cons))
    plain = generate(bp5, n_patients=npat5, seed=4)
    fixed = generate(bp5, n_patients=npat5, seed=4,
                     enforce_constraints=True)

    def broke_share(fr):
        a_ = pd.to_numeric(fr["visit_start"], errors="coerce")
        b_ = pd.to_numeric(fr["visit_end"], errors="coerce")
        m_ = a_.notna() & b_.notna()
        return float((a_[m_] > b_[m_]).mean())
    check("THE FIXTURE REPRODUCES THE FAULT: without repair the "
          "ordering breaks on {:.0%} of generated rows"
          .format(broke_share(plain)), broke_share(plain) > 0.1)
    check("...and --enforce-constraints puts it back",
          broke_share(fixed) == 0.0)
    check("...by SWAPPING, so both columns hold exactly the same "
          "multiset of values and their distributions are untouched",
          sorted(pd.to_numeric(plain["visit_start"]).tolist()
                 + pd.to_numeric(plain["visit_end"]).tolist())
          == sorted(pd.to_numeric(fixed["visit_start"]).tolist()
                    + pd.to_numeric(fixed["visit_end"]).tolist()))

    check("coverage is untouched by any of it", all(
        abs(float(pd.to_numeric(g[c], errors="coerce").notna().mean())
            - float(pd.to_numeric(df[c]).notna().mean())) <= 0.05
        for c in ("dose", "visits_billed", "weight")))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
