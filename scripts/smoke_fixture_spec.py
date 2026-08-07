"""Smoke: the spec carries the shape and none of the values.

Two things have to hold at once. It must describe an extract well
enough that a fixture built from it reproduces the faults that hid
here before - clustered missingness, integer identifiers, high-
cardinality categoricals, per-patient level. And it must leak nothing:
no category name, no free text, no date, no percentile of an
identifier.
"""
import csv
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PASS = FAIL = 0
SENT = "ZZSPECSENTINEL"


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


def build(path, n_pat=240, n_vis=12):
    rnd = random.Random(6)
    rows = []
    vid = 500000
    for p in range(n_pat):
        anchor = rnd.gauss(0, 1.0)
        ms = False
        for v in range(n_vis):
            vid += 1
            # missingness CLUSTERS, as it does clinically
            ms = rnd.random() < (0.85 if ms else 0.20)
            rows.append({
                "person_id": "P{:05d}".format(p),
                "visit_id": vid,
                "visit_start_date": "2021-{:02d}-{:02d}".format(
                    (v % 12) + 1, (v % 27) + 1),
                # steady because of WHO the patient is
                "anchored": round(anchor + rnd.gauss(0, 0.15), 4),
                "clustered_lab": ("" if ms
                                  else round(rnd.gauss(50, 8), 3)),
                "cat_wide": SENT + "_L{}".format(rnd.randint(0, 120)),
                "cat_narrow": SENT + ("_A" if p % 2 else "_B"),
                "meds": "; ".join(
                    SENT + "_d{}".format(rnd.randint(0, 40))
                    for _ in range(1 + p % 3)),
                "free_text": SENT + " a long clinical phrase, quoted",
            })
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "t.csv"
        build(src)
        r = run("scripts/emit_fixture_spec.py", "--in", str(src))
        if r.returncode:
            print(r.stdout, r.stderr)
        check("spec emitter runs", r.returncode == 0)
        s = r.stdout
        lines = s.splitlines()

        # ---- leaks -------------------------------------------------
        check("NO LEAK: not one category name, list item or free-text "
              "value appears anywhere in the spec", SENT not in s)
        check("NO LEAK: no date value is emitted", "2021-" not in s)
        vline = [x for x in lines if x.strip().startswith("visit_id ")]
        check("an identifier is NAMED as one rather than described by "
              "its range",
              vline and "kind identifier" in vline[0])
        check("...and no percentile of it is emitted, since those are "
              "real ids with nothing to learn from",
              not any("p1p50p99" in x for x in lines
                      if "visit_id" in x))

        # ---- shape -------------------------------------------------
        def block(col):
            """A column's own line plus its continuation lines, and
            nothing from the column after it."""
            got, on = [], False
            for x in lines:
                if x.startswith("  " + col + " kind"):
                    on, got = True, [x]
                elif on and x.startswith("    "):
                    got.append(x)
                elif on:
                    break
            return " | ".join(got)

        def field(col, key):
            for part in block(col).split("|"):
                p = part.strip().split()
                if len(p) >= 2 and p[0] == key:
                    return p[1]
            return None

        check("clustered missingness is measured, and reads well "
              "above the 0.5 that independent missingness gives",
              float(field("clustered_lab", "cluster") or 0) > 0.7)
        check("a fully present column is distinguishable by coverage",
              float(field("anchored", "cov") or 0) > 0.99
              and float(field("clustered_lab", "cov") or 1) < 0.6)
        check("per-patient level is carried as ICC, so a fixture can "
              "reproduce steadiness rather than guess at it",
              float(field("anchored", "icc") or 0) > 0.5)
        check("a wide categorical is distinguished from a narrow one "
              "by levels above k, which is what a transition table "
              "actually costs",
              int(field("cat_wide", "levels_above_k") or 0)
              > 5 * int(field("cat_narrow", "levels_above_k") or 1))
        check("a list column is named as one and carries its item "
              "count",
              "meds kind list" in s and "items_mean" in s)
        check("the visit-count distribution is carried by quantile",
              any(x.startswith("visits p50 ") for x in lines)
              and any(x.startswith("visits max ") for x in lines))

        # ---- format ------------------------------------------------
        check("every line fits the width, so nothing wraps in a "
              "terminal on the way here",
              all(len(x) <= 78 for x in lines))
        check("every number carries its own key, so a wrapped line "
              "cannot attach a value to the wrong field",
              all(len(p.strip().split()) % 2 == 0 or "kind" in p
                  for x in lines if "|" in x
                  for p in x.split("|")))

        # ---- guards ------------------------------------------------
        r2 = run("scripts/emit_fixture_spec.py", "--in",
                 str(Path(td) / "nope.csv"))
        check("a missing input fails readably",
              r2.returncode != 0
              and "not found" in (r2.stdout + r2.stderr)
              and "Traceback" not in (r2.stderr or ""))
        bad = Path(td) / "bad.csv"
        bad.write_text("a,b\n1,2\n", encoding="utf-8")
        r3 = run("scripts/emit_fixture_spec.py", "--in", str(bad))
        check("a file with no patient column names what it wanted",
              r3.returncode != 0
              and "person_id" in (r3.stdout + r3.stderr))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
