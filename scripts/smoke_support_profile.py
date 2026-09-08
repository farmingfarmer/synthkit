"""Smoke: the support profiler tiers columns by what they can carry,
and quantifies the variable-lag measurement error.

Every column in the fixture is planted with a known answer, so a
wrong tier is a failure rather than a judgement call:

  complete_ar     complete, decaying AR(1)   -> longitudinal, lag err 0
  sparse_ar       same process, 75% missing  -> lag error NEGATIVE
  onceonly        one visit per patient      -> cross_sectional
  rare            present for 20 patients    -> presence
  toorare         present for 5 patients     -> drop
  constant        one value throughout       -> drop
  degenerate      99.5% one value            -> drop
  cat_complete    categorical, complete      -> longitudinal
"""
import csv
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PASS = FAIL = 0
SENTINEL = "ZZVALUESENTINEL"
N_PAT, N_VIS = 200, 12
RHO = 0.9


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


PLANTED = ["complete_ar", "sparse_ar", "onceonly", "rare", "toorare",
           "constant", "degenerate", "cat_complete", "meds",
           "birth_year", "walk", "stable", "stable2", "stable3",
           "int_repeats", "assay_unique"]
# visit_id and visit_start_date are described too. They used to be
# skipped as "identifiers" while condnet modeled them, which is
# exactly where a 424 MB fault hid. Only the group column is skipped.
COLS = PLANTED + ["visit_id", "visit_start_date"]


def build(path):
    rnd = random.Random(5)
    rows = []
    for p in range(N_PAT):
        w = rnd.gauss(0, 1.0)
        mu = rnd.gauss(0, 0.3)
        walk = 0.0
        anchor = rnd.gauss(0, 1.0)
        for v in range(N_VIS):
            walk += rnd.gauss(0, 0.25)
            w = RHO * w + math.sqrt(1 - RHO ** 2) * rnd.gauss(0, 1.0)
            val = round(mu + w, 4)
            r = {"person_id": "P{:04d}".format(p),
                 # INTEGER, as in real OMOP. The string form was
                 # caught by the level test; the integer form was
                 # not, and that is the form the real extract has.
                 "visit_id": 100000 + len(rows),
                 # An integer that repeats, and a continuous value
                 # unique on every row - neither is a key.
                 "int_repeats": 30 + (p % 55),
                 "assay_unique": round(rnd.gauss(0, 1), 9),
                 # Visits PAIR UP on dates. Real patients are seen
                 # twice in a day; mimic never was, so a bare
                 # tuple sort over (day, value) never had to
                 # compare a float against a None and the crash
                 # could not appear here.
                 "visit_start_date": "2021-{:02d}-{:02d}".format(
                     ((v // 2) % 12) + 1, ((v // 2) % 27) + 1),
                 "complete_ar": val,
                 "sparse_ar": val if rnd.random() > 0.75 else "",
                 "onceonly": round(rnd.gauss(0, 1), 3) if v == 0 else "",
                 "rare": round(rnd.gauss(0, 1), 3) if p < 20 else "",
                 "toorare": round(rnd.gauss(0, 1), 3) if p < 5 else "",
                 "constant": 7,
                 "degenerate": (SENTINEL if rnd.random() < 0.995
                                else "other"),
                 "cat_complete": SENTINEL + ("A" if p % 2 else "B"),
                 # list-valued: condnet expands, never bins whole
                 "meds": "; ".join(
                     [SENTINEL + "drugA", SENTINEL + "drugB"][:1 + p % 2]),
                 # PATIENT-LEVEL constant: identical at every visit, so
                 # its autocorrelation is 1.0 at every lag. Including
                 # it in the decay curve drags the median to 1.0.
                 "birth_year": 1950 + (p % 60),
                 # DRIFTING: no stable patient level, but each visit
                 # is close to the one before. Steadiness here is
                 # recent history, and it decays with elapsed time.
                 "walk": round(walk, 4),
                 # ANCHORED: steady because of who the patient is.
                 # Cannot decay - there is no dynamics to decay.
                 "stable": round(anchor + rnd.gauss(0, 0.15), 4),
                 # Two more anchored columns, so the anchored group
                 # clears the three-column floor and the comparison
                 # against the drifting group actually runs. With one
                 # column it was silently skipped.
                 "stable2": round(anchor * 2 + rnd.gauss(0, 0.2), 4),
                 "stable3": round(anchor * 0.5 + 10
                                  + rnd.gauss(0, 0.1), 4)}
            rows.append(r)
    with path.open("w", newline="", encoding="utf-8") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w_.writeheader()
        w_.writerows(rows)
    return rows


def main():
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "tidy.csv"
        fixture_rows = build(src)
        outp = Path(td) / "support.json"
        r = run("scripts/support_profile.py", "--in", str(src),
                "-o", str(outp), "--report")
        if r.returncode:
            print(r.stdout, r.stderr)
        check("profiler runs", r.returncode == 0)
        o = json.loads(outp.read_text(encoding="utf-8"))
        by = dict((c["column"], c) for c in o["columns"])
        check("EVERY column is profiled except the group column - no "
              "hand-kept skip list to hide behind",
              set(by) == set(COLS))
        check("identifier and date columns are described rather than "
              "skipped, since the model does not skip them",
              by["visit_id"]["tier"] == "drop"
              and "visit_start_date" in by)
        check("an INTEGER key is caught - the numeric path skipped "
              "the identifier test entirely, and a real visit_id is "
              "an integer",
              "identifier" in by["visit_id"]["reason"]
              and by["visit_id"]["integer_valued"] is True)
        check("an integer that REPEATS is not mistaken for a key",
              by["int_repeats"]["tier"] != "drop")
        check("a continuous value unique on every row is not either - "
              "integrality is what separates a key from a fine assay",
              by["assay_unique"]["tier"] != "drop")
        check("patients and visits counted exactly",
              o["patients"] == N_PAT and o["rows"] == N_PAT * N_VIS)

        blob = (r.stdout or "") + json.dumps(o)
        check("NO LEAK: no column value reaches the output",
              SENTINEL not in blob)

        # ---------- tiers ----------
        check("a complete AR(1) column is modellable over time",
              by["complete_ar"]["tier"] == "longitudinal")
        check("a complete CATEGORICAL column is modellable over time "
              "too - adjacency is presence, not parseability",
              by["cat_complete"]["tier"] == "longitudinal"
              and by["cat_complete"]["adjacent_pairs"] > 0)
        check("a column present at one visit per patient has no pairs "
              "and drops to cross-sectional",
              by["onceonly"]["tier"] == "cross_sectional"
              and by["onceonly"]["adjacent_pairs"] == 0)
        check("a column held by 20 patients is demoted to presence, "
              "not deleted",
              by["rare"]["tier"] == "presence"
              and by["rare"]["patients_present"] == 20)
        check("a column below k patients is dropped",
              by["toorare"]["tier"] == "drop"
              and "k=" in by["toorare"]["reason"])
        check("a constant column is dropped",
              by["constant"]["tier"] == "drop"
              and by["constant"]["reason"] == "constant")
        check("a degenerate column is dropped for its top share, not "
              "its coverage",
              by["degenerate"]["tier"] == "drop"
              and "degenerate" in by["degenerate"]["reason"]
              and by["degenerate"]["visit_coverage"] == 1.0)
        check("tier counts add up to the column count",
              sum(o["tier_counts"].values()) == len(COLS))

        # ---------- coverage measured at both grains ----------
        check("coverage is reported per visit AND per patient, and "
              "they differ where it matters",
              by["onceonly"]["patient_coverage"] == 1.0
              and abs(by["onceonly"]["visit_coverage"]
                      - 1.0 / N_VIS) < 0.01)
        check("sparse column coverage is measured correctly",
              0.20 < by["sparse_ar"]["visit_coverage"] < 0.30)

        # ---------- the variable-lag error ----------
        ca = by["complete_ar"]
        check("a COMPLETE column has zero variable-lag error by "
              "construction",
              abs(ca.get("lag_error", 1.0)) < 0.005)
        sa = by["sparse_ar"]
        check("the same process at 75% missing is measured as LESS "
              "steady than it is - the target is understated",
              sa.get("lag_error") is not None
              and sa["lag_error"] < -0.05)
        check("the understatement is large enough to matter",
              sa["autocorr_as_measured"] < sa["autocorr_adjacent"] - 0.05
              and ca["autocorr_adjacent"] > 0.8)
        affected = [x["column"] for x in o["variable_lag_affected"]]
        check("the affected column is named in the report, the "
              "complete one is not",
              "sparse_ar" in affected and "complete_ar" not in affected)
        check("the report explains which direction the error goes",
              "UNDERSTATED" in r.stdout)

        # ---------- pairs, the thing the model gates on ----------
        check("adjacent pairs are counted separately from loose pairs, "
              "and sparsity destroys the adjacent ones",
              sa["adjacent_pairs"] < sa["loose_pairs"] / 2
              and ca["adjacent_pairs"] == ca["loose_pairs"])

        # ---------- same-day visits and real calendar gaps ----------
        # Both of these crashed or silently mismeasured on the real
        # extract while passing here, because mimic has one visit per
        # date per patient and no month boundaries in play.
        import support_profile as SP
        check("a patient seen TWICE IN A DAY does not crash the "
              "elapsed-time pass",
              r.returncode == 0 and "lag_curve" in o)
        one = [x for x in fixture_rows if x["person_id"] == "P0000"]
        check("the fixture really does contain same-day visits, or "
              "the check above proves nothing",
              len(set(x["visit_start_date"] for x in one)) < len(one))
        gaps = [("2021-02-28", "2021-03-01", 1),
                ("2021-01-31", "2021-02-01", 1),
                ("2020-12-31", "2021-01-01", 1),
                ("2021-01-01", "2022-01-01", 365)]
        check("elapsed days use real calendar arithmetic, so pairs "
              "land in the right bucket across month and year ends",
              all(SP.parse_day(b_) - SP.parse_day(a_) == want
                  for a_, b_, want in gaps))
        check("an impossible date is rejected rather than counted",
              SP.parse_day("2021-02-30") is None
              and SP.parse_day("not-a-date") is None)

        # ---------- the decay curve must not be contaminated -------
        lc = o["lag_curve"]
        check("a PATIENT-LEVEL constant is kept out of the decay "
              "curve - it sits at 1.0 in every bucket and drags the "
              "median to 1.0",
              "birth_year" in lc["excluded_patient_level"]
              and "birth_year" not in lc["by_column"])
        check("a column that really does vary within a patient IS "
              "used", "complete_ar" not in lc["excluded_patient_level"])
        check("the report says how many columns the curve rests on",
              lc["columns_used"] >= 1
              and "actually vary within a patient" in r.stdout)
        bb = by["birth_year"]
        check("a patient-level constant is 100% between-patient "
              "variance", bb["between_patient_variance"] > 0.99)
        ca_ = by["complete_ar"]
        check("a within-varying column has a between-patient share "
              "well below 1, which is the room elapsed time has to "
              "work in",
              ca_["between_patient_variance"] < 0.9)
        check("the ceiling on what elapsed time can buy is stated",
              "can decay at all" in r.stdout)

        if lc.get("decay"):
            base = lc["decay_base_bucket"]
            check("every decay row is normalized against ONE base "
                  "bucket, so the rows are comparable rather than "
                  "each resting on its own baseline",
                  base is not None
                  and any(d["bucket"] == base
                          and abs(d["median_ratio"] - 1.0) < 1e-9
                          for d in lc["decay"]))

        # Two columns on very different scales, each with NO real
        # persistence. Pooling their value pairs into one correlation
        # would read near 1.0 purely because the pairs cluster by
        # column; a median of per-column correlations reads near 0.
        scaled = Path(td) / "scaled.csv"
        rnd2 = random.Random(9)
        with scaled.open("w", newline="", encoding="utf-8") as f:
            w2 = csv.DictWriter(f, fieldnames=[
                "person_id", "visit_start_date", "big", "small"])
            w2.writeheader()
            for p_ in range(120):
                for v_ in range(8):
                    w2.writerow({
                        "person_id": "P{:04d}".format(p_),
                        "visit_start_date": "2021-{:02d}-{:02d}".format(
                            (v_ % 12) + 1, (v_ % 27) + 1),
                        "big": round(1700 + rnd2.gauss(0, 30), 2),
                        "small": round(3.7 + rnd2.gauss(0, 0.4), 3)})
        so = Path(td) / "scaled.json"
        run("scripts/support_profile.py", "--in", str(scaled),
            "-o", str(so))
        sj = json.loads(so.read_text(encoding="utf-8"))
        pooled = sj["lag_curve"].get("pooled") or []
        check("columns on different scales with no persistence do NOT "
              "read as persistent - the curve medians per-column "
              "correlations instead of pooling raw pairs",
              bool(pooled)
              and all(abs(b_["autocorr"]) < 0.35 for b_ in pooled))

        # ---------- the between-patient estimator ------------------
        # ICC(1), not 1 - SSW/SST. The naive share is biased upward
        # when patients have few observations - at two apiece it read
        # 0.754 where the truth was 0.500 - and that bias flows
        # straight into how much room within-patient dynamics appear
        # to have.
        import support_profile as SP2
        rnd3 = random.Random(4)
        for sb, sw, per, truth in ((1.0, 1.0, 3, 0.5),
                                   (0.5, 1.0, 3, 0.2)):
            icc_rows = {}
            for p_ in range(400):
                mu = rnd3.gauss(0, sb)
                icc_rows["P%d" % p_] = [{"v": mu + rnd3.gauss(0, sw)}
                                        for _ in range(per)]
            got = SP2.variance_split(icc_rows, "v")
            check("between-patient share recovers a known ICC of {} "
                  "from only {} observations per patient".format(
                      truth, per),
                  got is not None and abs(got - truth) < 0.08)
        # ---------- the two populations must not be averaged --------
        dcols = dict((x["column"], x)
                     for x in o["steadiness_decomposition"]["columns"])
        check("a DRIFTING column - no stable patient level, each "
              "visit close to the last - is classified by rho, not "
              "by how steady it looks",
              dcols["walk"]["rho"] >= 0.5
              and dcols["walk"]["r1"] > 0.8)
        check("an ANCHORED column - steady because of who the patient "
              "is - lands on the other side despite a HIGHER r(lag1)",
              dcols["stable"]["rho"] < 0.5
              and dcols["stable"]["r1"] > dcols["walk"]["r1"])
        check("the two are told apart by rho even though r(lag1) "
              "would rank them the wrong way round",
              dcols["stable"]["between"] > dcols["walk"]["between"])
        dd = o["steadiness_decomposition"].get("drift_decay")
        ad = o["steadiness_decomposition"].get("anchored_decay")
        # Asserted present, not skipped when missing. With only one
        # anchored column the comparison below quietly did not run,
        # which is the same fault as a fixture that cannot reproduce
        # what it is meant to test.
        check("BOTH populations are measured separately, so the "
              "comparison is not silently skipped",
              bool(dd) and bool(ad)
              and len(dd["columns"]) >= 3 and len(ad["columns"]) >= 3)
        if dd and ad:
            last_d = dd["decay"][-1]["median_ratio"]
            last_a = ad["decay"][-1]["median_ratio"]
            check("drifting columns DECAY and anchored ones do not - "
                  "the split the median was hiding",
                  last_d < last_a)
        check("near-deterministic columns are named so they can be "
              "derived rather than learned",
              isinstance(o["steadiness_decomposition"]
                         .get("near_deterministic"), list))

        # ---------- list columns read as what condnet does ---------
        check("a list column tiers as EXPAND, not drop - condnet "
              "expands it into per-item indicators",
              by["meds"]["tier"] == "expand"
              and by["meds"]["list_valued"] is True)

        # ---------- the field the tier rule gates on is shown -------
        check("patients_adjacent is in the printed table, since it is "
              "what decides the tier",
              "patAdj" in r.stdout)
        check("patients_adjacent never exceeds patients_2plus - a "
              "pair needs two observations",
              all(c["patients_adjacent"] <= c["patients_2plus"]
                  for c in o["columns"]))

        # ---------- model sizing ----------
        mp = Path(td) / "model.json"
        mp.write_text(json.dumps({
            "lag": {"a": {"1": {"1": 0.5, "2": 0.5}},
                    "b": {"1": {"1": 1.0}}},
            "binnings": {"a": [1, 2, 3]}}), encoding="utf-8")
        out2 = Path(td) / "s2.json"
        r2 = run("scripts/support_profile.py", "--in", str(src),
                 "--model", str(mp), "-o", str(out2))
        o2 = json.loads(out2.read_text(encoding="utf-8"))
        keys = dict((k["key"], k) for k in o2["model"]["keys"])
        # Shares must never over-attribute: the parts cannot exceed
        # the whole. They fall short of 1.0 by the JSON key and brace
        # overhead, which is large on a toy model and negligible on a
        # real one, so the useful assertion is the upper bound.
        check("the model artifact is sized by top-level key, heaviest "
              "first, without over-attributing",
              keys["lag"]["bytes"] > keys["binnings"]["bytes"]
              and o2["model"]["keys"][0]["key"] == "lag"
              and sum(k["share"] for k in o2["model"]["keys"]) <= 1.0)
        # A model can be enormous and have learned nothing. Size and
        # cell counts cannot tell the difference; the edge list can.
        mp2 = Path(td) / "model2.json"
        mp2.write_text(json.dumps({
            "lag": {"a": {"1": {"1": 1.0}}},
            "report": {"edges": [{"child": "y", "parents": ["x"]},
                                 {"child": "q", "parents": ["r", "s"]}],
                       "columns_modeled": 9, "bins": 3,
                       "effective_n": 120, "rows": 900,
                       "comparisons_corrected_for": 36,
                       "search_mode": "blind",
                       "derived_columns": [
                           {"column": "total", "determined_by": "part"}],
                       "rows": 900, "persons": 120}}),
            encoding="utf-8")
        o3 = Path(td) / "s3.json"
        r3 = run("scripts/support_profile.py", "--in", str(src),
                 "--model", str(mp2), "-o", str(o3))
        L = json.loads(o3.read_text(encoding="utf-8"))["model"]["learned"]
        check("the model report says WHAT IT LEARNED, not only how big "
              "it is - a large model can have found nothing",
              L["edge_count"] == 2
              and {"child": "q", "parents": ["r", "s"]} in L["edges"]
              and "y <- x" in r3.stdout)
        check("blind search is named along with the correction it "
              "pays, since that is the main thing suppressing edges",
              L["comparisons_corrected_for"] == 36
              and "blind search" in r3.stdout)
        nar = json.loads(o3.read_text(encoding="utf-8"))["model"] \
            .get("narrative") or {}
        check("the findings are narrated in plain English, reusing "
              "learnspec rather than restating its phrasing here",
              any("moves with" in f["sentence"]
                  for f in nar.get("findings", []))
              and any(f["kind"] == "interaction"
                      for f in nar.get("findings", [])))
        check("arithmetic the pipeline itself created is filed apart "
              "from findings, not counted as discovery",
              len(nar.get("findings", [])) == 2
              and len(nar.get("bookkeeping", [])) == 1
              and "computed from" in nar["bookkeeping"][0]["sentence"]
              and "NOT findings" in r3.stdout)
        check("the caveat carries its CONDITIONS - the correction "
              "actually paid, and that a no-main-effect interaction "
              "is never found at any n",
              "36 comparisons" in nar.get("caveat", "")
              and "no effect on their own" in nar.get("caveat", ""))
        dl = json.loads(o3.read_text(encoding="utf-8"))["model"] \
            .get("dials") or []
        check("every relationship is exposed as something the user "
              "can turn, starting at as-found",
              len(dl) == 2
              and all(d["factor"] == 1.0 and d["guide"] for d in dl)
              and any("given" in d["label"] for d in dl))

        check("a model with no report degrades instead of crashing",
              json.loads(out2.read_text(encoding="utf-8"))["model"]
              .get("learned") is None)

        check("the heaviest column inside the heaviest key is named",
              keys["lag"]["heaviest_column"] == "a"
              and keys["lag"]["cells"] == 3)

        # ---------- guards ----------
        r3 = run("scripts/support_profile.py", "--in",
                 str(Path(td) / "nope.csv"))
        check("a missing input fails readably",
              r3.returncode != 0
              and "not found" in (r3.stdout + r3.stderr)
              and "Traceback" not in (r3.stderr or ""))
        bad = Path(td) / "bad.csv"
        bad.write_text("a,b\n1,2\n", encoding="utf-8")
        r4 = run("scripts/support_profile.py", "--in", str(bad))
        check("a file with no patient column names the column it "
              "wanted",
              r4.returncode != 0
              and "person_id" in (r4.stdout + r4.stderr))
        empty = Path(td) / "empty.csv"
        empty.write_text("person_id,x\n", encoding="utf-8")
        r5 = run("scripts/support_profile.py", "--in", str(empty))
        check("a header with no rows fails readably",
              r5.returncode != 0
              and "no rows" in (r5.stdout + r5.stderr))
        r6 = run("scripts/support_profile.py", "--in", str(src),
                 "-o", str(Path(td) / "s6.json"),
                 "--min-pair-patients", "100000")
        o6 = json.loads((Path(td) / "s6.json").read_text(
            encoding="utf-8"))
        check("thresholds actually move the tiers, so they are "
              "reviewable rather than baked in",
              o6["tier_counts"]["longitudinal"] == 0)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
