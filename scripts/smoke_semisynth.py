"""Smoke: measured covariates with a planted answer key.

WHY THIS EXISTS. The `ceiling / baseline / vendor` framing is the
strongest thing in this tool, and it has only ever run on AUTHORED
specs - so every vendor number so far describes performance on data
whose covariates AND signal were invented. `bridge.py` carries the
measured marginals across and stops at `outcomes`, correctly: nobody
knows the answer in real data.

Planting one closes it. Real covariate distributions, a label whose
coefficients were chosen here, and therefore a computable ceiling.

TWO CHECKS ARE LOAD-BEARING.

THE EFFECT MUST COME BACK OUT AT THE SIZE IT WENT IN. A planted
coefficient that generates something else is worse than none, because
the whole point is that the answer is known. Effects are declared in
SDs and converted with the blueprint's own spread - a raw coefficient
of 0.5 means one thing on a creatinine of 1.1 +/- 0.35 and saturates
the logit on a glucose of 105 +/- 28.

AND THE COVARIATES MUST STILL BE THE MEASURED ONES. If planting a
label quietly changed the marginals, this would be an authored spec
with extra steps.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit import blueprint as B                    # noqa: E402
from synthkit import semisynth as S                    # noqa: E402
from synthkit.bridge import blueprint_to_tablespec     # noqa: E402
from synthkit.discover import discover                 # noqa: E402
from synthkit.tableplan import plan_table              # noqa: E402
from synthkit.tablespec import TableSpec               # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def source(seed=0, npat=500, nvis=6):
    """Columns on deliberately different scales.

    A creatinine of 1.1 +/- 0.35 beside a glucose of 105 +/- 28 is
    what makes raw coefficients meaningless and sd-scaled ones
    necessary - and a fixture whose columns share a scale could not
    show the difference."""
    r = np.random.RandomState(seed)
    g = np.repeat(np.arange(npat), nvis)
    n = len(g)
    return pd.DataFrame({
        "person_id": ["P{:04d}".format(i) for i in g],
        "creatinine": np.round(np.abs(r.normal(1.1, 0.35, n)), 3),
        "glucose": np.round(r.normal(105, 28, n), 1),
        "age": np.round(r.normal(63, 14, n)).astype(int),
        "sex": [str(x) for x in r.choice(["F", "M"], n)]})


def main():
    df = source()
    bp = B.build(df, discover(df, group_by="person_id", seed=1),
                 group_by="person_id")
    bridged = blueprint_to_tablespec(bp, title="fitted")

    check("the bridge leaves outcomes empty, which is the gap this "
          "closes", not (bridged["tablespec"].get("outcomes") or []))

    effects = {"creatinine": 0.9, "glucose": -0.5, "sex=M": 0.4}
    planted = S.plant(bridged, effects, prevalence=0.25)
    oc = planted["tablespec"]["outcomes"]
    check("an outcome is planted", len(oc) == 1)

    # ---- the coefficients are scaled by MEASURED spread ----------
    coefs = oc[0]["coefficients"]
    cal = dict((c["effect"], c) for c in planted["planted"]["calibration"])
    src_sd_cr = float(df["creatinine"].std())
    src_sd_gl = float(df["glucose"].std())
    check("creatinine's coefficient is its sd effect divided by the "
          "MEASURED spread ({:.4g} for 0.9 sd on sd {:.3f})".format(
              coefs["creatinine"], src_sd_cr),
          abs(coefs["creatinine"] - 0.9 / src_sd_cr)
          < 0.12 * abs(0.9 / src_sd_cr))
    check("...and glucose's likewise ({:.4g} on sd {:.1f}) - the same "
          "raw coefficient on both would saturate one and do nothing "
          "to the other".format(coefs["glucose"], src_sd_gl),
          abs(abs(coefs["glucose"]) - 0.5 / src_sd_gl)
          < 0.12 * (0.5 / src_sd_gl))
    check("...and the spread each was scaled by is RECORDED, so the "
          "conversion can be checked rather than trusted",
          abs(cal["creatinine"]["measured_spread"] - src_sd_cr)
          < 0.1 * src_sd_cr)
    check("a categorical level scales by sqrt(p(1-p)), so a rare "
          "level does not carry a meaningless coefficient",
          abs(coefs["sex=M"] - 0.4 / 0.5) < 0.15)

    # ---- refusals are reported, never silent ---------------------
    bad = S.plant(bridged, {"creatinine": 0.5, "nosuchcol": 1.0,
                            "sex=Z": 1.0})
    why = [r["effect"] for r in bad["planted"]["refused"]]
    check("an effect on a column that does not exist is REFUSED and "
          "named, not dropped - a planted effect that silently is not "
          "there makes every number after it wrong",
          "nosuchcol" in why)
    check("...and so is a level that was never published, which may "
          "have been suppressed for privacy", "sex=Z" in why)
    check("...and the block carries a WARNING, so a spec with a "
          "missing effect cannot look complete",
          "WARNING" in bad["planted"])
    try:
        S.plant(bridged, {"nosuchcol": 1.0})
        raised = False
    except ValueError:
        raised = True
    check("planting NOTHING raises rather than emitting an outcome "
          "with no coefficients", raised)

    # ---- the table it produces -----------------------------------
    spec = TableSpec.from_json(json.dumps(
        dict(planted["tablespec"], rows=4000)))
    check("the emitted spec VALIDATES - a planted outcome must not "
          "produce a spec the rest of the tool refuses",
          not spec.validate())

    tp = plan_table(spec)
    got = S.verify(planted, tp)
    check("prevalence lands near what was asked - {:.1%} against the "
          "25% requested".format(got["achieved_prevalence"]),
          got["prevalence_hit"] is True)
    check("...and a CEILING is computable, which is the whole point "
          "of an answer key ({:.3f})".format(got["ceiling_auroc"]),
          0.6 < got["ceiling_auroc"] < 0.95)

    rows = pd.DataFrame(tp.clean_rows)
    for c in ("creatinine", "glucose", "age"):
        rows[c] = pd.to_numeric(rows[c], errors="coerce")

    # THE COVARIATES ARE STILL THE MEASURED ONES.
    for c in ("creatinine", "glucose", "age"):
        s_m, s_s = float(df[c].astype(float).mean()), \
            float(df[c].astype(float).std())
        g_m, g_s = float(rows[c].mean()), float(rows[c].std())
        check("{} keeps its measured centre and spread through "
              "planting - {:.2f}/{:.2f} against {:.2f}/{:.2f}".format(
                  c, g_m, g_s, s_m, s_s),
              abs(g_m - s_m) < 0.15 * s_s and abs(g_s - s_s)
              < 0.25 * s_s)

    # THE EFFECT COMES BACK OUT AT THE SIZE IT WENT IN.
    from sklearn.linear_model import LogisticRegression
    y = np.array([1 if str(r) == "True" else 0
                  for r in rows["outcome"]])
    z = np.column_stack([
        ((rows[c] - rows[c].mean()) / rows[c].std()).to_numpy()
        for c in ("creatinine", "glucose", "age")]
        + [(rows["sex"] == "M").astype(float).to_numpy()])
    fit = LogisticRegression(max_iter=3000).fit(z, y)
    b_cr, b_gl, b_age, b_sex = fit.coef_[0]
    check("creatinine's planted effect is recovered in sd units - "
          "{:+.2f} against the +0.90 planted".format(b_cr),
          abs(b_cr - 0.9) < 0.3)
    check("...and glucose's, including its SIGN - {:+.2f} against "
          "-0.50".format(b_gl), abs(b_gl - (-0.5)) < 0.3)
    check("...and the categorical level, on its raw indicator - "
          "{:+.2f} against the {:+.2f} planted".format(
              b_sex, coefs["sex=M"]),
          abs(b_sex - coefs["sex=M"]) < 0.35)
    check("a column with NO planted effect recovers none ({:+.2f}) - "
          "without this the checks above would pass on a model that "
          "found something everywhere".format(b_age),
          abs(b_age) < 0.2)

    # ---- it says what it is --------------------------------------
    text = S.describe(planted)
    check("the block states plainly that the covariates are measured "
          "and the outcome is not",
          "measured" in text.lower() and "planted" in text.lower())
    check("...and the spec carries that too, because a spec that "
          "looks measured throughout and is half invented is the same "
          "failure as a column that looks present and is sentinel",
          "PLANTED" in json.dumps(planted["planted"]))
    check("...and says effects are in SDs, so nobody reads a "
          "coefficient as a raw-unit one",
          "standard deviation" in
          planted["planted"]["effects_are_in_sds"])

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
