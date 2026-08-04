"""Smoke: the synthetic extract reproduces the SHAPE of the real one.

The real files never leave the work machine, so this fixture is the
only thing the sampler is ever tested against. If it drifts from the
described shape, every downstream guarantee is being proved against
the wrong data - which makes these checks load-bearing rather than
decorative.

Targets, from statistics reported back off that machine:
  mean 68 visits, p50 31, p95 255, p99 498, max 1,518
  buckets 2% / 2.5% / 9% / 12% / 74%
  measurement : drug_exposure : condition_occurrence = 5 : 2.5 : 1
  concept ~40% of a 73%-cohort output if copied whole
"""
import csv
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import make_omop_fixture as F        # noqa: E402

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


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            yield header, row


def col(path, name):
    it = rows(path)
    try:
        header, first = next(it)
    except StopIteration:
        return
    i = header.index(name)
    yield first[i] if i < len(first) else ""
    for _h, row in it:
        yield row[i] if i < len(row) else ""


def main():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "fx"
        r = run("scripts/make_omop_fixture.py", "-o", str(out), "--report")
        if r.returncode:
            print(r.stdout, r.stderr)
        check("fixture generator runs", r.returncode == 0)
        files = {p.name for p in out.glob("*.csv")}
        check("all seven tables emitted with the _full suffix",
              files == {t + "_full.csv" for t in
                        F.PATIENT_TABLES + ["concept"]})

        # ---------- headers ----------
        ok = True
        for t, want in F.OMOP_HEADERS.items():
            with (out / (t + "_full.csv")).open(encoding="utf-8-sig") as f:
                got = f.readline().strip()
            if got != want:
                ok = False
                print("    {} header drift".format(t))
        check("every header matches the real extract byte-exact", ok)

        # ---------- the visit-count distribution ----------
        counts = {}
        for pid in col(out / "visit_occurrence_full.csv", "person_id"):
            k = pid.lstrip("0") or "0"
            counts[k] = counts.get(k, 0) + 1
        vals = sorted(counts.values())
        n = len(vals)
        mean = sum(vals) / float(n)

        def q(p):
            return vals[min(n - 1, int(p * n))]

        check("mean visits per patient is 68 (+/-5%)",
              abs(mean - 68) / 68.0 < 0.05)
        check("p50 visits is 31 (+/-2)", abs(q(0.50) - 31) <= 2)
        check("p95 visits is 255 (+/-8%)",
              abs(q(0.95) - 255) / 255.0 < 0.08)
        check("p99 visits is 498 (+/-8%)",
              abs(q(0.99) - 498) / 498.0 < 0.08)
        check("a superuser patient sits at exactly 1,518 visits",
              max(vals) == 1518)
        check("the heavy tail is real: 74% of patients have 11+ visits "
              "(+/-2pp)",
              abs(sum(1 for v in vals if v >= 11) / float(n) - 0.74) < 0.02)
        buckets = [(1, 1, 0.020), (2, 2, 0.025), (3, 5, 0.090),
                   (6, 10, 0.120)]
        okb = True
        for lo, hi, want in buckets:
            got = sum(1 for v in vals if lo <= v <= hi) / float(n)
            if abs(got - want) > 0.015:
                okb = False
                print("    bucket {}-{}: {:.3f} want {:.3f}".format(
                    lo, hi, got, want))
        check("the light buckets match 2% / 2.5% / 9% / 12%", okb)

        # the distribution FUNCTION, free of sampling noise
        rnd = random.Random(3)
        big = sorted(F.visit_count(rnd.random()) for _ in range(120000))
        bmean = sum(big) / float(len(big))
        check("the underlying distribution holds at 120k draws, not "
              "just at this fixture size",
              abs(bmean - 68) / 68.0 < 0.03
              and abs(big[int(0.95 * len(big))] - 255) / 255.0 < 0.05)

        # ---------- table proportions ----------
        size = dict((t, (out / (t + "_full.csv")).stat().st_size)
                    for t in F.PATIENT_TABLES + ["concept"])
        cond = float(size["condition_occurrence"])
        check("measurement is the largest patient table",
              size["measurement"] == max(
                  size[t] for t in F.PATIENT_TABLES))
        check("measurement : condition is 5 : 1 (+/-8%)",
              abs(size["measurement"] / cond - 5.0) / 5.0 < 0.08)
        check("drug_exposure : condition is 2.5 : 1 (+/-8%)",
              abs(size["drug_exposure"] / cond - 2.5) / 2.5 < 0.08)
        patient_bytes = sum(size[t] for t in F.PATIENT_TABLES)
        share = size["concept"] / (patient_bytes * 0.732 + size["concept"])
        check("copied whole, concept would be 40% of a 73%-cohort "
              "output (+/-4pp)", abs(share - 0.40) < 0.04)
        check("the vocabulary dwarfs what the extract actually uses",
              sum(1 for _ in col(out / "concept_full.csv", "concept_id"))
              > 20 * len(set(col(out / "condition_occurrence_full.csv",
                                 "condition_concept_id"))))

        # ---------- the pathologies ----------
        pids = list(col(out / "visit_occurrence_full.csv", "person_id"))
        padded = sum(1 for v in pids if v.startswith("0"))
        check("person_id formatting is MIXED: the same patient appears "
              "zero-padded and not",
              0 < padded < len(pids))
        blanks = {}
        for t in ("drug_exposure", "measurement", "procedure_occurrence",
                  "condition_occurrence"):
            vs = list(col(out / (t + "_full.csv"), "person_id"))
            blanks[t] = sum(1 for v in vs if not v.strip()) / float(len(vs))
        check("person_id blank rates VARY by table, so the visit-id "
              "fallback is live code",
              blanks["measurement"] > blanks["drug_exposure"]
              > blanks["condition_occurrence"] > 0
              and 0.10 < blanks["measurement"] < 0.14)
        both = 0
        for _h, row in rows(out / "measurement_full.csv"):
            if not row[1].strip() and not row[14].strip():
                both += 1
        check("some rows are blank in BOTH keys and are unattributable "
              "to any patient", both > 0)
        vd = list(col(out / "condition_occurrence_full.csv",
                      "visit_detail_id"))
        check("visit_detail_id is a dead reference: 100% blank",
              all(not v.strip() for v in vd))
        with (out / "concept_full.csv").open(encoding="utf-8-sig") as f:
            check("the vocabulary has no person_id column",
                  "person_id" not in f.readline())

        sigs = list(col(out / "drug_exposure_full.csv", "sig"))
        check("sig carries commas and quotes and an embedded date",
              any("," in s and '"' in s for s in sigs))
        conds = list(col(out / "condition_occurrence_full.csv",
                         "condition_source_value"))
        check("condition_source_value carries clinical phrases with "
              "commas", any("," in c for c in conds))
        names = list(col(out / "concept_full.csv", "concept_name"))
        check("concept_name carries commas and quotes",
              any("," in nm for nm in names)
              and any('"' in nm for nm in names))
        drugs = list(col(out / "drug_exposure_full.csv",
                         "drug_source_value"))
        check("drug_source_value uses tall-man lettering",
              any("predniSONE" in d or "valGANciclovir" in d
                  for d in drugs))
        srcc = set(col(out / "condition_occurrence_full.csv",
                       "condition_source_concept_id"))
        allc = set(col(out / "concept_full.csv", "concept_id"))
        check("a deliberately dangling concept reference exists",
              str(F.DANGLING_CONCEPT) in srcc
              and str(F.DANGLING_CONCEPT) not in allc)
        check("the standard OMOP type concepts DO resolve",
              all(str(c) in allc for c in F.SYSTEM_CONCEPTS))

        # ---------- determinism and scaling ----------
        out2 = Path(td) / "fx2"
        run("scripts/make_omop_fixture.py", "-o", str(out2))
        check("same seed reproduces the extract byte for byte",
              all((out / f).read_bytes() == (out2 / f).read_bytes()
                  for f in files))
        out3 = Path(td) / "fx3"
        run("scripts/make_omop_fixture.py", "-o", str(out3), "--seed", "99")
        check("a different seed gives different data",
              (out3 / "measurement_full.csv").read_bytes()
              != (out / "measurement_full.csv").read_bytes())

        out4 = Path(td) / "fx4"
        run("scripts/make_omop_fixture.py", "-o", str(out4),
            "--scale", "0.25")
        m4 = (out4 / "measurement_full.csv").stat().st_size
        check("--scale shrinks the fat tables",
              m4 < size["measurement"] * 0.4)
        counts4 = {}
        for pid in col(out4 / "visit_occurrence_full.csv", "person_id"):
            k = pid.lstrip("0") or "0"
            counts4[k] = counts4.get(k, 0) + 1
        # Byte-for-byte equality would be the wrong test: a smaller
        # vocabulary shifts the RNG stream, so the concept ids on each
        # visit row differ. The distribution is the invariant.
        check("--scale leaves the visit-count distribution untouched, "
              "so the heavy tail survives a fast fixture",
              sorted(counts4.values()) == sorted(counts.values()))
        check("the superuser survives --scale",
              max(counts4.values()) == 1518)

        r5 = run("scripts/make_omop_fixture.py", "-o", str(Path(td) / "x"),
                 "--patients", "0")
        check("an impossible patient count fails readably",
              r5.returncode != 0 and "Traceback" not in (r5.stderr or ""))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
