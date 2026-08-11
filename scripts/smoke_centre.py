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
from synthkit.discover import discover                   # noqa: E402
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
