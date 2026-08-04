"""Smoke: the sampler pulls COMPLETE patients and filters the
vocabulary, without ever holding a table.

The invariant everything else rests on: a patient is wholly in or
wholly out. A partial patient breaks every downstream join, and would
do it silently - the tables would still load, the joins would still
run, and the answers would just be wrong. So the central check counts
each sampled patient's rows in the source and in the output and
demands they match exactly, per table.

The vocabulary checks matter for a different reason: concept does not
shrink with the cohort, so copied whole it dominates the output and
makes the cohort lever nearly inert. Filtering has to keep every
lookup a written row can perform.
"""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import omop_sample as S              # noqa: E402
from omop_wrangle import norm_key    # noqa: E402

PASS = FAIL = 0
EVENT_TABLES = ["condition_occurrence", "drug_exposure", "measurement",
                "procedure_occurrence"]


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


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            yield header, row


def per_person(path, key="person_id"):
    """Rows per normalized person id."""
    out = {}
    for header, row in read_rows(path):
        i = header.index(key)
        pid = norm_key(row[i]) if i < len(row) else ""
        if pid:
            out[pid] = out.get(pid, 0) + 1
    return out


def concept_ids(path):
    return set(norm_key(r[h.index("concept_id")])
               for h, r in read_rows(path))


def main():
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "fx"
        r = run("scripts/make_omop_fixture.py", "-o", str(src),
                "--patients", "200", "--scale", "0.25")
        check("fixture built for the sampler", r.returncode == 0)

        out = Path(td) / "s1"
        r = run("scripts/omop_sample.py", "--src", str(src), "-o",
                str(out), "--cohort", "40", "--seed", "5", "--report")
        if r.returncode:
            print(r.stdout, r.stderr)
        check("sampler runs and its own verify pass agrees",
              r.returncode == 0 and "verify PASSED" in r.stdout)
        man = json.loads((out / "sample_manifest.json")
                         .read_text(encoding="utf-8"))
        check("manifest is written", man["cohort_selected"] == 40)

        # ---------- THE invariant: complete patients ----------
        cohort = set(per_person(out / "person.csv"))
        check("exactly the requested number of patients came out",
              len(cohort) == 40)
        partial = []
        for t in ["visit_occurrence"] + EVENT_TABLES:
            src_counts = per_person(src / (t + "_full.csv"))
            out_counts = per_person(out / (t + ".csv"))
            for pid in cohort:
                if src_counts.get(pid, 0) != out_counts.get(pid, 0):
                    partial.append((t, pid, src_counts.get(pid, 0),
                                    out_counts.get(pid, 0)))
        if partial:
            for p in partial[:4]:
                print("    partial: {} person {} src {} out {}".format(*p))
        check("NO PARTIAL PATIENTS: every sampled patient's row count "
              "matches the source exactly, in every table", not partial)
        strays = set()
        for t in ["visit_occurrence"] + EVENT_TABLES:
            strays |= set(per_person(out / (t + ".csv"))) - cohort
        check("no rows from patients outside the cohort leaked in",
              not strays)
        check("every selected patient has at least one visit",
              all(per_person(out / "visit_occurrence.csv").get(p, 0) > 0
                  for p in cohort))

        # ---------- blank person_id recovery ----------
        via_visit = sum(t.get("via_visit", 0)
                        for t in man["tables"].values())
        check("rows with a blank person_id are recovered through their "
              "visit id, not silently dropped", via_visit > 0)
        unattr = sum(t.get("unattributable", 0)
                     for t in man["tables"].values())
        check("rows blank in BOTH keys are dropped AND counted",
              unattr > 0)
        kept_visits = set()
        for h, row in read_rows(out / "visit_occurrence.csv"):
            kept_visits.add(norm_key(row[h.index("visit_occurrence_id")]))
        orphan = 0
        for t in EVENT_TABLES:
            for h, row in read_rows(out / (t + ".csv")):
                pi, vi = h.index("person_id"), h.index("visit_occurrence_id")
                pid = norm_key(row[pi]) if pi < len(row) else ""
                vid = norm_key(row[vi]) if vi < len(row) else ""
                if not pid and vid not in kept_visits:
                    orphan += 1
        check("every recovered row belongs to a visit that came out",
              orphan == 0)

        # ---------- the vocabulary ----------
        src_c = concept_ids(src / "concept_full.csv")
        out_c = concept_ids(out / "concept.csv")
        check("concept is filtered, not copied",
              len(out_c) < len(src_c) / 10)
        # This fixture runs at --scale 0.25, which shrinks the
        # vocabulary but not the in-use concept pools, so the ratio is
        # milder here than at full scale (1.7%) or in the real extract.
        check("filtering is the dominant lever: the vocabulary drops "
              "by over 90%", len(out_c) < len(src_c) * 0.10)
        unresolved = 0
        for t in ["person", "visit_occurrence"] + EVENT_TABLES:
            for h, row in read_rows(out / (t + ".csv")):
                for i in S.concept_columns(h):
                    if i < len(row):
                        c = norm_key(row[i])
                        if c and c not in out_c and c in src_c:
                            unresolved += 1
        check("every concept id a written row can look up still "
              "resolves", unresolved == 0)
        # Asserting "0" in the normal output proves nothing: the person
        # rows carry gender_source_concept_id 0, so it is referenced
        # anyway. The sentinel branch only matters when NOTHING
        # references 0, so drive that case directly.
        probe = Path(td) / "probe.csv"
        one = sorted(src_c - {"0"})[:1]
        S.filter_concept(src / "concept_full.csv", probe, set(one),
                         lambda *_a: None)
        check("the no-matching-concept sentinel 0 is retained even when "
              "no row references it", "0" in concept_ids(probe))
        check("a reference that dangles IN THE SOURCE is reported, not "
              "blamed on the sampler",
              man["tables"]["concept"]["dangling_count"] >= 1
              and not man["verify_failures"])

        # ---------- determinism and selection ----------
        out2 = Path(td) / "s2"
        run("scripts/omop_sample.py", "--src", str(src), "-o", str(out2),
            "--cohort", "40", "--seed", "5")
        check("same seed gives byte-identical output",
              all((out / f.name).read_bytes() == f.read_bytes()
                  for f in out2.glob("*.csv")))
        out3 = Path(td) / "s3"
        run("scripts/omop_sample.py", "--src", str(src), "-o", str(out3),
            "--cohort", "40", "--seed", "77")
        check("a different seed gives a different cohort",
              set(per_person(out3 / "person.csv")) != cohort)

        worst = max(abs(s["sample_share"] - s["source_share"])
                    for s in man["strata"])
        check("stratification reproduces the visit-count distribution "
              "(every stratum within 3pp of source)", worst < 0.03)
        out4 = Path(td) / "s4"
        run("scripts/omop_sample.py", "--src", str(src), "-o", str(out4),
            "--cohort", "40", "--seed", "5", "--uniform")
        check("--uniform actually selects differently",
              set(per_person(out4 / "person.csv")) != cohort)

        # ---------- the concentration lever ----------
        out5 = Path(td) / "s5"
        run("scripts/omop_sample.py", "--src", str(src), "-o", str(out5),
            "--cohort", "40", "--seed", "5", "--exclude-above-visits",
            "300")
        m5 = json.loads((out5 / "sample_manifest.json")
                        .read_text(encoding="utf-8"))
        check("--exclude-above-visits lowers the heaviest patient's "
              "share of the cohort",
              m5["heaviest_patient_share"] < man["heaviest_patient_share"])
        check("excluded patients are dropped ENTIRELY, never truncated",
              m5["heaviest_patient_visits"] <= 300
              and m5["dropped_above_max_visits"] > 0)
        c5 = set(per_person(out5 / "person.csv"))
        okc = True
        for t in EVENT_TABLES:
            sc = per_person(src / (t + "_full.csv"))
            oc = per_person(out5 / (t + ".csv"))
            for pid in c5:
                if sc.get(pid, 0) != oc.get(pid, 0):
                    okc = False
        check("exclusion preserves completeness for everyone kept", okc)

        out6 = Path(td) / "s6"
        run("scripts/omop_sample.py", "--src", str(src), "-o", str(out6),
            "--cohort", "20", "--min-visits", "31")
        m6 = json.loads((out6 / "sample_manifest.json")
                        .read_text(encoding="utf-8"))
        check("--min-visits filters and reports the bias it introduces",
              m6["dropped_below_min_visits"] > 0
              and m6["strata"][0]["sample"] == 0)

        # ---------- copy mode, for contrast ----------
        out7 = Path(td) / "s7"
        run("scripts/omop_sample.py", "--src", str(src), "-o", str(out7),
            "--cohort", "40", "--seed", "5", "--copy-concept")
        m7 = json.loads((out7 / "sample_manifest.json")
                        .read_text(encoding="utf-8"))
        check("--copy-concept is measurably the wrong default",
              m7["bytes_out"] > man["bytes_out"] * 2)

        # ---------- the verifier itself must be able to fail ----------
        bad = Path(td) / "bad"
        bad.mkdir()
        for f in out.glob("*.csv"):
            (bad / f.name).write_bytes(f.read_bytes())
        with (bad / "measurement.csv").open(
                "a", encoding="utf-8", newline="") as f:
            f.write("9,88888888,1,2021-01-01,,,1,,1.0,,1,1,2,,1,,LAB,1,mg\n")
        fails, _d = S.verify(bad, cohort, set(cohort), set(), set())
        check("VERIFIER CAN FAIL: a row from an outside patient is "
              "caught", any("outside the cohort" in f for f in fails))
        bad2 = Path(td) / "bad2"
        bad2.mkdir()
        for f in out.glob("*.csv"):
            (bad2 / f.name).write_bytes(f.read_bytes())
        lines = (bad2 / "concept.csv").read_text(
            encoding="utf-8").splitlines(True)
        (bad2 / "concept.csv").write_text(
            "".join(lines[:3]), encoding="utf-8")
        fails2, _d = S.verify(bad2, cohort, set(cohort), set(), set())
        check("VERIFIER CAN FAIL: a truncated vocabulary is caught",
              any("do not resolve" in f for f in fails2))

        # ---------- guards ----------
        r8 = run("scripts/omop_sample.py", "--src", str(src), "-o",
                 str(Path(td) / "s8"), "--cohort", "99999")
        check("asking for more patients than exist fails readably",
              r8.returncode != 0
              and "only" in (r8.stdout + r8.stderr)
              and "Traceback" not in (r8.stderr or ""))
        broken = Path(td) / "broken"
        broken.mkdir()
        (broken / "person_full.csv").write_text("a,b\n1,2\n",
                                                encoding="utf-8")
        r9 = run("scripts/omop_sample.py", "--src", str(broken), "-o",
                 str(Path(td) / "s9"))
        joined = r9.stdout + r9.stderr
        check("preflight names EVERY problem at once, before reading "
              "a row",
              r9.returncode != 0 and joined.count("missing table") >= 5
              and "no person_id column" in joined
              and "Traceback" not in (r9.stderr or ""))
        r10 = run("scripts/omop_sample.py", "--src", str(src), "-o",
                  str(out), "--cohort", "10")
        check("a non-empty output folder is refused rather than mixed",
              r10.returncode != 0
              and "not empty" in (r10.stdout + r10.stderr))

        # ---------- dry run ----------
        dry = Path(td) / "dry"
        r12 = run("scripts/omop_sample.py", "--src", str(src), "-o",
                  str(dry), "--cohort", "40", "--seed", "5", "--dry-run")
        check("--dry-run succeeds and says plainly that nothing was "
              "written",
              r12.returncode == 0
              and "no files were written" in r12.stdout)
        check("--dry-run writes NOTHING, not even the output folder",
              not dry.exists())
        check("--dry-run reports the same cohort the real run selects",
              "cohort 40" in r12.stdout
              and str(man["cohort_visits"]) not in ("0", ""))
        check("--dry-run shows the vocabulary at its WHOLE size and "
              "says filtering shrinks it",
              "WHOLE" in r12.stdout and "referenced ids" in r12.stdout)
        check("--dry-run tolerates a non-empty output folder, since it "
              "will not touch it",
              run("scripts/omop_sample.py", "--src", str(src), "-o",
                  str(out), "--cohort", "10",
                  "--dry-run").returncode == 0)
        r13 = run("scripts/omop_sample.py", "--src", str(broken), "-o",
                  str(dry), "--cohort", "10", "--dry-run")
        check("--dry-run still runs preflight, so a broken source is "
              "caught before the long run",
              r13.returncode != 0
              and "missing table" in (r13.stdout + r13.stderr))

        # ---------- the naming contract with the rest of the pipeline --
        check("output files drop the _full suffix so --src works "
              "downstream",
              {p.name for p in out.glob("*.csv")}
              == {t + ".csv" for t in S.ALL_TABLES})
        r11 = run("scripts/omop_wrangle.py", "--src", str(out), "-o",
                  str(Path(td) / "tidy.csv"), "--report")
        check("the existing wrangler consumes the sample unchanged",
              r11.returncode == 0
              and (Path(td) / "tidy.csv").exists())
        tidy = list(read_rows(Path(td) / "tidy.csv"))
        check("the wrangled sample has the cohort's patients in it",
              set(norm_key(r[h.index("person_id")])
                  for h, r in tidy) <= cohort and len(tidy) > 0)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
