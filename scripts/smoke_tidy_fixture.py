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
        cov = 0.056
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
               "interaction-no-main-effect"} <= kinds)
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

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
