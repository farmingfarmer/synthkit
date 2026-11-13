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

    # THE CLI PATH EXISTS NOW - bridge and plant were reachable
    # only from Python, so the end-to-end loop on the data machine
    # had no road. Exercised as the operator runs them: subprocess,
    # echoing what was planted and REFUSING what it cannot find.
    import json as _json
    import subprocess as _sp
    import sys as _sys
    import tempfile as _tf
    from pathlib import Path as _P
    _root = _P(__file__).resolve().parent.parent
    with _tf.TemporaryDirectory() as _td:
        _run = _P(_td) / "run"
        _run.mkdir()
        (_run / "blueprint.json").write_text(
            _json.dumps(bp), encoding="utf-8")
        _rb = _sp.run([_sys.executable, "-m", "synthkit.cli",
                       "bridge", str(_run)],
                      capture_output=True, text=True,
                      cwd=str(_root))
        check("`synthkit bridge RUNDIR` writes tablespec.json from "
              "an existing run - no 40-minute refit for a file the "
              "blueprint already implies",
              _rb.returncode == 0
              and (_run / "tablespec.json").exists()
              and "column(s) crossed" in _rb.stdout
              and "did NOT cross" in _rb.stdout)
        _exam = _P(_td) / "exam.json"
        _rp = _sp.run([_sys.executable, "-m", "synthkit.cli",
                       "plant", "--spec",
                       str(_run / "tablespec.json"),
                       "--effect", "creatinine=0.9",
                       "--effect", "nosuchcol=1.0",
                       "--prevalence", "0.25",
                       "-o", str(_exam)],
                      capture_output=True, text=True,
                      cwd=str(_root))
        check("`synthkit plant` echoes what it planted in sd, "
              "REFUSES the unknown column by name, and writes the "
              "exam spec beside the full answer-key record",
              _rp.returncode == 0
              and "planted creatinine: +0.90 sd" in _rp.stdout
              and "REFUSED nosuchcol" in _rp.stdout
              and _exam.exists()
              and _exam.with_suffix(".planted.json").exists()
              and (_json.loads(_exam.read_text())
                   .get("outcomes") or []))
        _ra = _sp.run([_sys.executable, "-m", "synthkit.cli",
                       "plant", "--spec",
                       str(_run / "tablespec.json"),
                       "--effect", "nosuchcol=1.0",
                       "-o", str(_P(_td) / "bad.json")],
                      capture_output=True, text=True,
                      cwd=str(_root))
        check("...and a plant whose every effect is refused exits "
              "non-zero - an exam with no planted signal grades "
              "nothing",
              _ra.returncode != 0
              and "no effect could be planted" in
              (_ra.stdout + _ra.stderr)
              and "Traceback" not in (_ra.stdout + _ra.stderr))

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
        check("{} keeps its measured center and spread through "
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

    # ONE SD OF WHAT THE RECIPIENT HOLDS, not of the contract.
    #
    # `plant_frame` used the published trapezoid over the quantile
    # grid, which is what the SAMPLER would draw from and not what
    # the frame in hand actually carries - so a recovered
    # coefficient came out smaller than the one planted by the
    # ratio of the two spreads. The bias was always there and
    # became visible only when the k-screened knots smoothed the
    # grid slightly and the recovery crossed a tolerance it had
    # been sitting just inside.
    import numpy as _np2
    import pandas as _pd2
    from synthkit import blueprint as _B2
    from synthkit import semisynth as _SS
    _r = _np2.random.default_rng(13)
    _n = 6000
    _g = _np2.repeat(_np2.arange(_n // 3), 3)
    _fr = _pd2.DataFrame({
        "pid": ["P%04d" % i for i in _g],
        "a": _r.normal(50, 9, _n),
        "b": _r.normal(0, 1, _n)})
    _cat = {"claims": [], "unexplained": [], "skipped": []}
    _bp2 = _B2.build(_fr, _cat, group_by="pid")
    _out_fr, _pl = _SS.plant_frame(
        _fr, _bp2, {"a": 0.9, "b": -0.5}, name="y",
        prevalence=0.3, seed=5)
    _rec = _SS.verify_frame(_out_fr, _pl)
    _miss = max(abs(float(e["recovered_in_sds"])
                    - float(e["requested_in_sds"]))
                for e in _rec["effects"])
    check("a planted effect is recovered within 0.2 sd of what "
          "was asked, standardised on the frame rather than on "
          "the published grid (worst miss {:.4f})".format(_miss),
          _miss < 0.2)
    _cal = _pl.get("calibration") or []
    _num = [u for u in _cal if "=" not in u["effect"]]
    check("...and the record carries BOTH the frame's own centre "
          "and spread and the published pair, so a reader can see "
          "which was used and by how much they differ",
          bool(_num)
          and all("measured_spread" in u and "published_spread" in u
                  for u in _num))
    check("...and the two genuinely differ on this frame, so the "
          "distinction is not cosmetic",
          any(abs(float(u["measured_spread"])
                  - float(u["published_spread"])) > 1e-6
              for u in _num))
    # AN OLDER RECORD STILL VERIFIES. Absent is not the same fact
    # as zero, and a planted record written before this change
    # carries only the published pair.
    _old = dict(_pl)
    _old["calibration"] = [
        dict((k, v) for k, v in u.items()
             if k not in ("measured_center", "measured_spread"))
        for u in _cal]
    _rec2 = _SS.verify_frame(_out_fr, _old)
    check("a planted record written BEFORE this change still "
          "verifies, against the published pair it was built with "
          "rather than raising on a missing key",
          len(_rec2["effects"]) == len(_rec["effects"]))

    # ASSERT THE DECISION, NOT THE OUTCOME. The mutation that made
    # `verify_frame` ignore the measured pair broke no check,
    # because on this frame the two spreads are close enough that
    # the recovery still landed inside the tolerance. So the
    # preference is made observable: a record whose measured
    # spread is deliberately DOUBLED must move the recovered
    # number, and it can only do that if the measured pair is the
    # one being used.
    _bent = dict(_pl)
    _bent["calibration"] = [
        (dict(u, measured_spread=float(u["measured_spread"]) * 2.0)
         if "measured_spread" in u else dict(u))
        for u in _cal]
    _rec3 = _SS.verify_frame(_out_fr, _bent)
    _moved = max(
        abs(float(a["recovered_in_sds"]) - float(b["recovered_in_sds"]))
        for a, b in zip(_rec["effects"], _rec3["effects"]))
    check("`verify_frame` USES the measured pair rather than "
          "merely carrying it - doubling the recorded spread moves "
          "the recovered coefficient by {:.4f}, which it could not "
          "do if the published pair were being read".format(_moved),
          _moved > 0.1)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
