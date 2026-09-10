"""Smoke: the fixture reproduces the MEASURED shape and knows its
own answers.

A fixture is only worth having if it is faithful. Every property
asserted here has a measured counterpart from the real extract, named
in make_tidy_fixture, and the assertion is against that number rather
than against a guess.
"""
import csv
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from make_tidy_fixture import VITALS, LABS, VISIT_Q   # noqa: E402
from emit_fixture_spec import icc as icc_of           # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def run(*args):
    return subprocess.run([sys.executable] + list(args),
                          capture_output=True, text=True, cwd=str(ROOT))


def load(d):
    with (d / "tidy_visits_labeled.csv").open(
            encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "fx"
        r = run("scripts/make_tidy_fixture.py", "-o", str(d),
                "--patients", "300", "--max-visits", "200")
        if r.returncode:
            print(r.stdout, r.stderr)
        check("fixture generator runs", r.returncode == 0)
        rows = load(d)
        truth = json.loads((d / "ground_truth.json")
                           .read_text(encoding="utf-8"))
        by = defaultdict(list)
        for x in rows:
            by[x["person_id"]].append(x)

        # ---- the visit distribution -------------------------------
        vc = sorted(len(v) for v in by.values())
        p50 = vc[len(vc) // 2]
        check("visits per patient reproduce the measured median of 31",
              abs(p50 - 31) <= 3)
        check("the heavy tail survives - p90 near the measured 179",
              abs(vc[int(0.90 * len(vc))] - 179) <= 25)
        check("a handful of patients carry hundreds of visits, as they "
              "do in the real extract", vc[-1] >= 150)

        # ---- coverage and per-patient level -----------------------
        cov_err, icc_err, n = 0.0, 0.0, 0
        worst_cov = ("", 0.0)
        for spec in VITALS:
            name, cov, _cl, icc = spec[0], spec[1], spec[2], spec[3]
            present = [x for x in rows if str(x.get(name, "")).strip()]
            got_cov = len(present) / float(len(rows))
            cov_err += abs(got_cov - cov)
            if abs(got_cov - cov) > worst_cov[1]:
                worst_cov = (name, abs(got_cov - cov))
            groups = []
            for g in by.values():
                xs = [num(x.get(name)) for x in g]
                xs = [v for v in xs if v is not None]
                if xs:
                    groups.append(xs)
            got_icc = icc_of(groups)
            if got_icc is not None:
                icc_err += abs(got_icc - icc)
                n += 1
        check("every column's COVERAGE lands on its measured value "
              "(mean error under 2 points)",
              cov_err / len(VITALS) < 0.02)
        check("...including the 0.2%-coverage invasive pressures, "
              "which a naive missingness model overshoots badly",
              worst_cov[1] < 0.05)
        check("per-patient level lands near its measured ICC (mean "
              "error under 0.15) - the anchor is what steadiness "
              "rests on", n and icc_err / n < 0.15)

        # ---- missingness clusters ---------------------------------
        name = "heart_rate_monitored"
        same = tot = 0
        for g in by.values():
            st = [bool(str(x.get(name, "")).strip()) for x in g]
            for i in range(len(st) - 1):
                tot += 1
                same += 1 if st[i] == st[i + 1] else 0
        # Independence must be computed from the coverage MEASURED
        # here, not the target. They differ by a point or two, and
        # using the target makes the excess wrong by more than the
        # excess itself.
        cov = sum(1 for x in rows
                  if str(x.get(name, "")).strip()) / float(len(rows))
        ind = cov * cov + (1 - cov) ** 2
        exc = (same / float(tot) - ind) / (1 - ind)
        # The real extract's EXCESS clustering runs 0.09-0.20. The raw
        # match rate of 0.904 for this column is almost all coverage:
        # independence alone gives 0.894.
        check("missingness CLUSTERS above what independence gives - "
              "the property whose absence broke fix 3a, and the "
              "measured excess here is 0.094 not the raw 0.904",
              0.03 < exc < 0.30)

        # ---- ground truth -----------------------------------------
        kinds = {r_["kind"] for r_ in truth["relationships"]}
        check("the planted answers cover every pattern KIND, so recall "
              "can be broken out by kind rather than averaged",
              {"linear", "nonlinear-u", "threshold",
               "interaction-with-main-effect",
               "interaction-no-main-effect",
               "lagged-cross-column", "subgroup-conditional",
               "saturating", "three-way", "heterogeneous",
               "simpsons-reversal"} <= kinds)
        check("the no-main-effect interaction is recorded as expected "
              "to be MISSED, so the blind spot is measured rather "
              "than assumed",
              any("cannot see this" in (r_.get("note") or "")
                  for r_ in truth["relationships"]))
        check("noise columns are named, so a false positive is "
              "identifiable rather than arguable",
              len(truth["noise_columns"]) >= 5)

        # the planted relationships must actually be IN the data
        def corr(xs, ys):
            m1, m2 = sum(xs) / len(xs), sum(ys) / len(ys)
            nu = sum((a - m1) * (b - m2) for a, b in zip(xs, ys))
            de = (sum((a - m1) ** 2 for a in xs)
                  * sum((b - m2) ** 2 for b in ys)) ** 0.5
            return nu / de if de else 0.0
        lx = [num(x["planted_linear_x"]) for x in rows]
        ly = [num(x["planted_linear_y"]) for x in rows]
        check("the planted LINEAR relationship is really there",
              corr(lx, ly) > 0.9)
        ux = [num(x["planted_ushape_x"]) for x in rows]
        uy = [num(x["planted_ushape_y"]) for x in rows]
        check("the planted U-SHAPE is really there, and is INVISIBLE "
              "to correlation - which is the point of it",
              abs(corr(ux, uy)) < 0.10
              and corr([abs(v - 0.5) for v in ux], uy) > 0.8)
        # The Simpson pair must genuinely REVERSE, or it is not the
        # test it claims to be: negative inside each stratum, positive
        # when pooled.
        sx = [num(x["planted_simpson_x"]) for x in rows]
        sy = [num(x["planted_simpson_y"]) for x in rows]
        pooled = corr(sx, sy)
        within = []
        for g in ("0", "1"):
            gx = [num(x["planted_simpson_x"]) for x in rows
                  if str(x["planted_simpson_g"]) == g]
            gy = [num(x["planted_simpson_y"]) for x in rows
                  if str(x["planted_simpson_g"]) == g]
            if len(gx) > 30:
                within.append(corr(gx, gy))
        check("the Simpson pair really does REVERSE - positive pooled, "
              "negative inside every stratum - or it is not testing "
              "what it claims",
              pooled > 0.3 and within and all(w < -0.3 for w in within))
        # The lagged pair must be lagged: y follows the PREVIOUS x, so
        # the same-visit correlation is near zero.
        lx = [num(x["planted_lag_x"]) for x in rows]
        ly = [num(x["planted_lag_y"]) for x in rows]
        check("the lagged pair shows NO same-visit correlation, which "
              "is what makes it invisible to a within-visit search",
              abs(corr(lx, ly)) < 0.10)
        n0 = [num(x["noise_00"]) for x in rows]
        check("a noise column really is unrelated to a planted one",
              abs(corr(n0, ly)) < 0.10)

        # ---- harder -----------------------------------------------
        d2 = Path(td) / "fx2"
        run("scripts/make_tidy_fixture.py", "-o", str(d2),
            "--patients", "300", "--max-visits", "200",
            "--harder", "2.0")
        rows2 = load(d2)
        c1 = sum(1 for x in rows
                 if str(x.get("spo2", "")).strip()) / float(len(rows))
        c2 = sum(1 for x in rows2
                 if str(x.get("spo2", "")).strip()) / float(len(rows2))
        check("--harder makes the data harder rather than different: "
              "the same column comes out sparser", c2 < c1 * 0.75)

        r3 = run("scripts/make_tidy_fixture.py", "-o",
                 str(Path(td) / "x"), "--harder", "0")
        check("an impossible setting fails readably",
              r3.returncode != 0 and "Traceback" not in (r3.stderr or ""))

    # THE FIXTURE COULD NOT REACH THE SHAPE THAT BROKE ON REAL DATA.
    #
    # Its set columns drew twelve common items and a long rare tail,
    # so exactly TWELVE tokens ever cleared the k floor. No cap on
    # the published vocabulary could bind, and nothing here could
    # show what one costs - which is how a cap of 60 came to send
    # every published share about 3x high on the extract while the
    # whole net stayed green. The extract has 1,050 of 7,974 tokens
    # above k on `conditions`; this fixture had 12 of 2,709.
    #
    # `--set-vocab N` adds a MIDDLE band: common enough to clear k,
    # outside the top twelve. How many actually clear it depends on
    # patients x visits, so this asserts the CONSEQUENCE - that a
    # 60-cap now binds - rather than the requested number.
    import pandas as _pd
    from synthkit import sets as _S
    from synthkit.blueprint import MAX_LEVELS_KEPT as _CAP
    with tempfile.TemporaryDirectory() as _td:
        _d3 = Path(_td) / "vocab"
        run("scripts/make_tidy_fixture.py", "-o", str(_d3),
            "--patients", "300", "--max-visits", "10",
            "--seed", "11", "--set-vocab", "150")
        _f3 = _d3 / "tidy_visits_labeled.csv"
        check("--set-vocab writes a fixture", _f3.exists())
        if _f3.exists():
            # READ IT THE WAY THE PIPELINE DOES. With pandas defaults
            # a blank field becomes NaN and every present-but-empty
            # set row disappears - which is exactly how this fixture
            # was measured as having none, three times, and reported
            # as blind when it was not.
            _df3 = _pd.read_csv(_f3, encoding="utf-8-sig",
                                low_memory=False, dtype=str,
                                keep_default_na=False)
            _g3 = _df3["person_id"].values
            _above = {}
            for _c in ("conditions", "active_drugs", "procedures"):
                _v = _S.vocabulary(_df3[_c], _g3, k=10, cap=0)
                _above[_c] = _v["tokens_above_k"] if _v else 0
            check("...where a set column now clears the k floor with "
                  "more tokens than the level cap ({}): conditions "
                  "{}, active_drugs {} - it was 12 for every column "
                  "before, so no cap could bind".format(
                      _CAP, _above["conditions"], _above["active_drugs"]),
                  _above["conditions"] > _CAP
                  and _above["active_drugs"] > _CAP)
            _emp = float((_df3["procedures"].astype(str).str.strip()
                          == "").mean())
            check("...and the present-but-EMPTY rows are still there "
                  "({:.0%} of `procedures`, against 80.7% on the "
                  "extract) - read with pandas defaults this reads "
                  "0%, which is what made it look blind".format(_emp),
                  _emp > 0.5)

    # THE REPRODUCTION FIXTURES CONTAIN THE MECHANISMS THEY CLAIM.
    # Both were aimed at statistics MEASURED on the real extract
    # (2026-09-10): the colinear pressure family behind the seed-37
    # inversions and departed surfaces, and the attribution
    # triangle behind the SHAP flip. A fixture that misses its aim
    # reproduces a different fault than the one that exists.
    sys.path.insert(0, str(Path(__file__).parent))
    from repro_mechanisms import build_pressure, build_triangle
    _pf = build_pressure()
    check("the pressure fixture hits its measured aims - S-D near "
          "+0.888, the two weak negatives near -0.304 and -0.126, "
          "and a tight MAP identity",
          0.80 <= _pf["systolic"].corr(_pf["diastolic"]) <= 0.95
          and -0.40 <= _pf["span_days"].corr(_pf["map_calc"])
          <= -0.20
          and -0.25 <= _pf["proc_count"].corr(_pf["diastolic"])
          <= -0.05
          and float(((_pf["systolic"] + 2 * _pf["diastolic"]) / 3
                     - _pf["map_calc"]).abs().mean()) < 1.0)
    _tf = build_triangle()
    check("the triangle fixture is mutually correlated (a cycle "
          "for discovery) with the lead driver dominant, and both "
          "builders are deterministic",
          _tf["span_like"].corr(_tf["proc_like"]) > 0.7
          and _tf["span_like"].corr(_tf["drug_like"]) > 0.6
          and _tf["proc_like"].corr(_tf["drug_like"]) > 0.6
          and build_pressure().equals(_pf)
          and build_triangle().equals(_tf))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
