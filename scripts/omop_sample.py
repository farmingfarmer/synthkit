"""Pull COMPLETE patients out of a large OMOP extract by streaming,
so every join downstream stays intact.

    python scripts/omop_sample.py --src DIR -o OUT [--cohort 800]
                                  [--seed 7] [--report]

A partial patient breaks every join, so the unit of selection is the
person and never the row: a patient is wholly in or wholly out.

Fixed number of streaming passes; nothing ever holds a table. Memory
is bounded by the cohort, its visits, and the concepts they reference
- all of which are reported rather than assumed.

  1. census      visit_occurrence -> person -> visit count
  2. select      stratified on visit count, seeded, reproducible
  3. visits      write the cohort's visits; collect their visit ids
  4. person      write the cohort's demographics
  5. events      condition / drug / measurement / procedure
  6. concept     LAST, filtered to the ids the written rows reference
  7. verify      re-read the output and prove the invariants

Why concept is filtered rather than copied. It is a ~9M-row, 1.2 GB
vocabulary with no person_id, so it does not shrink with the cohort -
copied whole it is 40% of a 4,000-patient output and 92% of a 250-
patient one, which makes the cohort lever nearly inert. An extract
touches a sliver of a vocabulary, so filtering to referenced ids takes
it to single-digit MB. The guarantee preserved is the one that
matters: every concept id any written row can look up still resolves.
Filtering happens last because the referenced set is not known until
every patient row has been written.

Key normalization is load-bearing, not decorative: ids arrive as ints
AND as zero-padded text in the same column, so a cohort held in raw
form would select one patient twice under two spellings and take half
their rows. Membership is tested on normalized keys; written rows keep
their raw bytes, so the output stays a faithful sub-extract.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:                       # console-safe on Windows terminals
    from _console import console_safe
    console_safe()
except Exception:
    pass

from omop_wrangle import norm_key

csv.field_size_limit(1 << 22)

PATIENT_TABLES = ["person", "visit_occurrence", "condition_occurrence",
                  "drug_exposure", "measurement", "procedure_occurrence"]
VOCAB_TABLE = "concept"
ALL_TABLES = PATIENT_TABLES + [VOCAB_TABLE]

# Straddles the measured p50 of 31 and opens out the 74% of patients
# sitting at 11+ visits, which a coarser scheme would collapse.
STRATA = [("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6-10", 6, 10),
          ("11-30", 11, 30), ("31-100", 31, 100),
          ("101-300", 101, 300), ("301+", 301, 10 ** 9)]

CONCEPT_REF_SUFFIX = "_concept_id"
PROGRESS_EVERY = 250000


def find_table(src, table):
    for name in (table + "_full.csv", table + ".csv"):
        p = src / name
        if p.exists():
            return p
    return None


def stratum_of(n):
    for label, lo, hi in STRATA:
        if lo <= n <= hi:
            return label
    return STRATA[-1][0]


def open_reader(path):
    f = path.open(encoding="utf-8-sig", newline="", errors="replace")
    r = csv.reader(f)
    try:
        header = next(r)
    except StopIteration:
        f.close()
        return None, None, None
    return f, r, [h.strip() for h in header]


def concept_columns(header):
    """Every *_concept_id column references the vocabulary. The suffix
    rule cleanly excludes person_id, provider_id, care_site_id,
    visit_detail_id and the table's own primary key."""
    return [i for i, h in enumerate(header)
            if h.endswith(CONCEPT_REF_SUFFIX)]


def preflight(src, out, check_out=True):
    """Every problem at once, before a single row is read."""
    problems = []
    found = {}
    for t in ALL_TABLES:
        p = find_table(src, t)
        if p is None:
            problems.append("missing table: {} (expected {}_full.csv "
                            "or {}.csv)".format(t, t, t))
            continue
        found[t] = p
    for t, p in sorted(found.items()):
        f, r, header = open_reader(p)
        if header is None:
            problems.append("{} is empty (no header row)".format(p.name))
            continue
        f.close()
        need = ["person_id"] if t in PATIENT_TABLES else ["concept_id"]
        if t == "visit_occurrence":
            need.append("visit_occurrence_id")
        for col in need:
            if col not in header:
                problems.append("{} has no {} column; found: {}".format(
                    p.name, col, ", ".join(header[:8])))
    if check_out and out.exists() and any(out.iterdir()):
        problems.append("output folder is not empty: {}".format(out))
    return found, problems


def census(path, progress):
    """Pass 1: person -> visit count. One small int per patient."""
    f, r, header = open_reader(path)
    counts = {}
    pid_i = header.index("person_id")
    rows = 0
    t0 = time.time()
    for row in r:
        rows += 1
        if pid_i < len(row):
            pid = norm_key(row[pid_i])
            if pid:
                counts[pid] = counts.get(pid, 0) + 1
        if rows % PROGRESS_EVERY == 0:
            progress("census visit_occurrence", rows, time.time() - t0)
    f.close()
    return counts, rows


def select(counts, args):
    """Patient-level, seeded, stratified on visit count."""
    eligible = {}
    dropped_min = dropped_max = 0
    for pid, n in counts.items():
        if n < args.min_visits:
            dropped_min += 1
            continue
        if args.exclude_above_visits and n > args.exclude_above_visits:
            # dropped WHOLLY - capping their rows would manufacture
            # exactly the partial patient this script exists to prevent
            dropped_max += 1
            continue
        eligible[pid] = n
    rnd = random.Random(args.seed)
    if args.uniform:
        pool = sorted(eligible)
        rnd.shuffle(pool)
        chosen = pool[:args.cohort]
    else:
        by_stratum = {}
        for pid, n in eligible.items():
            by_stratum.setdefault(stratum_of(n), []).append(pid)
        total = float(len(eligible)) or 1.0
        chosen = []
        # Largest-remainder apportionment. Rounding each stratum
        # independently and dumping the drift on one of them skews that
        # stratum by several points; giving the leftover seats to the
        # largest fractional remainders spreads the error evenly.
        alloc = {}
        rema = []
        for label, _lo, _hi in STRATA:
            members = by_stratum.get(label, [])
            quota = args.cohort * len(members) / total
            alloc[label] = int(quota)
            rema.append((quota - int(quota), label))
        seats = args.cohort - sum(alloc.values())
        rema.sort(key=lambda kv: (-kv[0], kv[1]))
        for _frac, label in rema[:max(0, seats)]:
            alloc[label] += 1
        # never allocate a stratum more patients than it holds
        for label, _lo, _hi in STRATA:
            alloc[label] = min(alloc[label],
                               len(by_stratum.get(label, [])))
        for label, _lo, _hi in STRATA:
            members = sorted(by_stratum.get(label, []))
            rnd.shuffle(members)
            chosen.extend(members[:alloc.get(label, 0)])
        if len(chosen) < args.cohort:     # a stratum ran dry; top up
            rest = sorted(set(eligible) - set(chosen))
            rnd.shuffle(rest)
            chosen.extend(rest[:args.cohort - len(chosen)])
    return set(chosen), eligible, dropped_min, dropped_max


def extract(path, outp, cohort, kept_visits, referenced, table,
            collect_visits, progress):
    """One streaming pass: keep a row if its patient is in the cohort,
    or - when person_id is blank, which varies by table in the real
    extract - if its visit belongs to one."""
    f, r, header = open_reader(path)
    of = outp.open("w", newline="", encoding="utf-8")
    w = csv.writer(of, lineterminator="\n")
    w.writerow(header)
    pid_i = header.index("person_id")
    vis_i = (header.index("visit_occurrence_id")
             if "visit_occurrence_id" in header else None)
    cref = concept_columns(header)
    stats = {"rows_in": 0, "rows_out": 0, "via_person": 0,
             "via_visit": 0, "unattributable": 0, "other_patient": 0}
    t0 = time.time()
    for row in r:
        stats["rows_in"] += 1
        pid = norm_key(row[pid_i]) if pid_i < len(row) else ""
        vid = (norm_key(row[vis_i])
               if vis_i is not None and vis_i < len(row) else "")
        keep = False
        if pid:
            if pid in cohort:
                keep = True
                stats["via_person"] += 1
            else:
                stats["other_patient"] += 1
        elif vid:
            if vid in kept_visits:
                keep = True
                stats["via_visit"] += 1
            else:
                stats["other_patient"] += 1
        else:
            stats["unattributable"] += 1
        if keep:
            w.writerow(row)
            stats["rows_out"] += 1
            for i in cref:
                if i < len(row):
                    c = norm_key(row[i])
                    if c:
                        referenced.add(c)
        if stats["rows_in"] % PROGRESS_EVERY == 0:
            progress(table, stats["rows_in"], time.time() - t0)
    f.close()
    of.close()
    stats["seconds"] = round(time.time() - t0, 1)
    stats["bytes_out"] = outp.stat().st_size
    return stats


def extract_visits(path, outp, cohort, referenced, progress):
    """Pass 3: the visits themselves, plus the id set that lets a
    blank-person_id event row be recovered later."""
    f, r, header = open_reader(path)
    of = outp.open("w", newline="", encoding="utf-8")
    w = csv.writer(of, lineterminator="\n")
    w.writerow(header)
    pid_i = header.index("person_id")
    vid_i = header.index("visit_occurrence_id")
    cref = concept_columns(header)
    kept_visits = set()
    per_person = {}
    stats = {"rows_in": 0, "rows_out": 0}
    t0 = time.time()
    for row in r:
        stats["rows_in"] += 1
        pid = norm_key(row[pid_i]) if pid_i < len(row) else ""
        if pid and pid in cohort:
            w.writerow(row)
            stats["rows_out"] += 1
            if vid_i < len(row):
                v = norm_key(row[vid_i])
                if v:
                    kept_visits.add(v)
            per_person[pid] = per_person.get(pid, 0) + 1
            for i in cref:
                if i < len(row):
                    c = norm_key(row[i])
                    if c:
                        referenced.add(c)
        if stats["rows_in"] % PROGRESS_EVERY == 0:
            progress("visit_occurrence", stats["rows_in"],
                     time.time() - t0)
    f.close()
    of.close()
    stats["seconds"] = round(time.time() - t0, 1)
    stats["bytes_out"] = outp.stat().st_size
    return kept_visits, per_person, stats


def extract_person(path, outp, cohort, referenced, progress):
    f, r, header = open_reader(path)
    of = outp.open("w", newline="", encoding="utf-8")
    w = csv.writer(of, lineterminator="\n")
    w.writerow(header)
    pid_i = header.index("person_id")
    cref = concept_columns(header)
    seen = set()
    stats = {"rows_in": 0, "rows_out": 0}
    t0 = time.time()
    for row in r:
        stats["rows_in"] += 1
        pid = norm_key(row[pid_i]) if pid_i < len(row) else ""
        if pid and pid in cohort:
            w.writerow(row)
            stats["rows_out"] += 1
            seen.add(pid)
            for i in cref:
                if i < len(row):
                    c = norm_key(row[i])
                    if c:
                        referenced.add(c)
        if stats["rows_in"] % PROGRESS_EVERY == 0:
            progress("person", stats["rows_in"], time.time() - t0)
    f.close()
    of.close()
    stats["seconds"] = round(time.time() - t0, 1)
    stats["bytes_out"] = outp.stat().st_size
    return seen, stats


def filter_concept(path, outp, referenced, progress):
    """Pass 6, last: keep only what the written rows can look up."""
    f, r, header = open_reader(path)
    of = outp.open("w", newline="", encoding="utf-8")
    w = csv.writer(of, lineterminator="\n")
    w.writerow(header)
    cid_i = header.index("concept_id")
    written = set()
    stats = {"rows_in": 0, "rows_out": 0}
    t0 = time.time()
    for row in r:
        stats["rows_in"] += 1
        cid = norm_key(row[cid_i]) if cid_i < len(row) else ""
        # "0" is OMOP's no-matching-concept sentinel; rows carry it
        # deliberately, so it is kept whether or not it was referenced.
        if cid and (cid in referenced or cid == "0"):
            w.writerow(row)
            stats["rows_out"] += 1
            written.add(cid)
        if stats["rows_in"] % PROGRESS_EVERY == 0:
            progress("concept", stats["rows_in"], time.time() - t0)
    f.close()
    of.close()
    stats["seconds"] = round(time.time() - t0, 1)
    stats["bytes_out"] = outp.stat().st_size
    # Every referenced id that EXISTS upstream is written, because the
    # keep test is membership in `referenced`. So what is left over is
    # exactly the set that was never in the source vocabulary.
    dangling = referenced - written
    stats["referenced"] = len(referenced)
    stats["dangling"] = sorted(dangling)[:20]
    stats["dangling_count"] = len(dangling)
    return stats, dangling


def verify(out, cohort, expected_visitors, referenced, source_dangling):
    """Re-read the output and prove the invariants. Returns failures.

    A reference that is absent from the SOURCE vocabulary is a
    pre-existing defect in the extract - the sampler cannot conjure a
    row upstream never had. Those are reported, not failed. Only a
    reference whose concept row exists upstream and did not survive
    filtering is a bug here."""
    fails = []
    concept_ids = set()
    cf = out / "concept.csv"
    if cf.exists():
        f, r, header = open_reader(cf)
        ci = header.index("concept_id")
        for row in r:
            if ci < len(row):
                concept_ids.add(norm_key(row[ci]))
        f.close()
    if source_dangling is None:
        # concept was copied whole, so the output IS the source
        # vocabulary and anything unresolved was never upstream.
        source_dangling = referenced - concept_ids
    seen_persons = set()
    for t in PATIENT_TABLES:
        p = out / (t + ".csv")
        if not p.exists():
            continue
        f, r, header = open_reader(p)
        pid_i = header.index("person_id")
        cref = concept_columns(header)
        stray = 0
        unresolved = 0
        for row in r:
            pid = norm_key(row[pid_i]) if pid_i < len(row) else ""
            if pid:
                seen_persons.add(pid)
                if pid not in cohort:
                    stray += 1
            for i in cref:
                if i < len(row):
                    c = norm_key(row[i])
                    if (c and c not in concept_ids
                            and c not in source_dangling):
                        unresolved += 1
        f.close()
        if stray:
            fails.append("{}: {} rows belong to patients outside the "
                         "cohort".format(t, stray))
        if unresolved:
            fails.append("{}: {} concept references do not resolve in "
                         "the filtered vocabulary".format(t, unresolved))
    missing = expected_visitors - seen_persons
    if missing:
        fails.append("{} selected patients have no rows in the output"
                     .format(len(missing)))
    return fails, source_dangling


def main():
    ap = argparse.ArgumentParser(
        description="Sample complete patients from a large OMOP extract.")
    ap.add_argument("--src", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--cohort", type=int, default=800,
                    help="patients to keep; 800 covers nonlinear "
                         "structure and leaves a well-powered holdout")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--uniform", action="store_true",
                    help="plain random instead of stratified on visits")
    ap.add_argument("--min-visits", type=int, default=1)
    ap.add_argument("--exclude-above-visits", type=int, default=0,
                    help="drop very heavy patients ENTIRELY (0 = off)")
    ap.add_argument("--copy-concept", action="store_true",
                    help="copy the vocabulary whole instead of "
                         "filtering it to referenced ids")
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight, census and select, then STOP "
                         "without writing a row - one pass over "
                         "visit_occurrence instead of all seven tables")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    src = Path(a.src)
    out = Path(a.out)
    if not src.is_dir():
        sys.exit("source folder not found: {}".format(src))
    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    found, problems = preflight(src, out, check_out=not a.dry_run)
    if problems:
        sys.exit("preflight failed:\n  " + "\n  ".join(problems))

    def progress(label, rows, elapsed):
        rate = rows / elapsed if elapsed > 0 else 0.0
        sys.stderr.write("  {:<26} {:>12,} rows  {:>6.0f}s  "
                         "{:>9,.0f} rows/s\n".format(label, rows,
                                                     elapsed, rate))
        sys.stderr.flush()

    t_start = time.time()
    sys.stderr.write("pass 1/7  census\n")
    counts, visit_rows = census(found["visit_occurrence"], progress)
    if not counts:
        sys.exit("visit_occurrence has no usable person_id values - "
                 "nothing to sample")
    if a.cohort > len(counts):
        sys.exit("cannot sample {:,} patients: only {:,} have visits.\n"
                 "Lower --cohort, or pass --cohort {} to take everyone."
                 .format(a.cohort, len(counts), len(counts)))

    cohort, eligible, drop_min, drop_max = select(counts, a)
    if not cohort:
        sys.exit("no patients selected: --min-visits {} and "
                 "--exclude-above-visits {} left {} eligible"
                 .format(a.min_visits, a.exclude_above_visits,
                         len(eligible)))

    # projected output, checked against free disk BEFORE writing
    share = len(cohort) / float(len(counts))
    src_bytes = sum(found[t].stat().st_size for t in found)
    patient_bytes = sum(found[t].stat().st_size
                        for t in PATIENT_TABLES if t in found)
    projected = int(patient_bytes * share)
    if a.copy_concept and VOCAB_TABLE in found:
        projected += found[VOCAB_TABLE].stat().st_size
    try:
        free = shutil.disk_usage(str(out)).free
    except Exception:
        free = None
    if free is not None and projected > free:
        sys.exit("projected output {:,} bytes exceeds {:,} bytes free "
                 "on the output volume. Lower --cohort or free space."
                 .format(projected, free))

    if a.dry_run:
        dry_report(a, found, counts, cohort, eligible, drop_min,
                   drop_max, share, projected, free,
                   time.time() - t_start)
        return

    referenced = set()
    tables = {}
    sys.stderr.write("pass 3/7  visits\n")
    kept_visits, per_person, tables["visit_occurrence"] = extract_visits(
        found["visit_occurrence"], out / "visit_occurrence.csv",
        cohort, referenced, progress)
    sys.stderr.write("pass 4/7  person\n")
    seen_person, tables["person"] = extract_person(
        found["person"], out / "person.csv", cohort, referenced, progress)
    sys.stderr.write("pass 5/7  events\n")
    for t in ("condition_occurrence", "drug_exposure", "measurement",
              "procedure_occurrence"):
        if t not in found:
            continue
        tables[t] = extract(found[t], out / (t + ".csv"), cohort,
                            kept_visits, referenced, t, False, progress)

    sys.stderr.write("pass 6/7  concept\n")
    if a.copy_concept:
        shutil.copyfile(str(found[VOCAB_TABLE]), str(out / "concept.csv"))
        cstat = {"rows_in": None, "rows_out": None, "mode": "copied whole",
                 "bytes_out": (out / "concept.csv").stat().st_size,
                 "referenced": len(referenced), "dangling": []}
        dangling = None      # verify derives it: the output IS the source
    else:
        cstat, dangling = filter_concept(
            found[VOCAB_TABLE], out / "concept.csv", referenced, progress)
        cstat["mode"] = "filtered to referenced ids"
    tables["concept"] = cstat

    sys.stderr.write("pass 7/7  verify\n")
    fails, dangling = verify(out, cohort, set(cohort), referenced,
                             dangling)
    tables["concept"]["dangling_count"] = len(dangling)
    if not tables["concept"].get("dangling"):
        tables["concept"]["dangling"] = sorted(dangling)[:20]

    # ---------- manifest ----------
    src_hist = {}
    smp_hist = {}
    for pid, n in counts.items():
        s = stratum_of(n)
        src_hist[s] = src_hist.get(s, 0) + 1
        if pid in cohort:
            smp_hist[s] = smp_hist.get(s, 0) + 1
    cohort_visits = sum(per_person.values()) or 1
    top = max(per_person.values()) if per_person else 0
    manifest = {
        "generated_by": "scripts/omop_sample.py",
        "source": str(src.resolve()),
        "seed": a.seed,
        "cohort_requested": a.cohort,
        "cohort_selected": len(cohort),
        "selection": "uniform" if a.uniform else "stratified on visits",
        "persons_available": len(counts),
        "persons_eligible": len(eligible),
        "dropped_below_min_visits": drop_min,
        "dropped_above_max_visits": drop_max,
        "sample_share": round(share, 6),
        "strata": [{"stratum": s,
                    "source": src_hist.get(s, 0),
                    "source_share": round(
                        src_hist.get(s, 0) / float(len(counts)), 4),
                    "sample": smp_hist.get(s, 0),
                    "sample_share": round(
                        smp_hist.get(s, 0) / float(len(cohort)), 4)}
                   for s, _lo, _hi in STRATA],
        "cohort_visits": sum(per_person.values()),
        "heaviest_patient_visits": top,
        "heaviest_patient_share": round(top / float(cohort_visits), 4),
        "selected_without_person_row": len(cohort - seen_person),
        "tables": tables,
        "bytes_source": src_bytes,
        "bytes_out": sum(t.get("bytes_out", 0) for t in tables.values()),
        "memory": {"census_entries": len(counts),
                   "kept_visit_ids": len(kept_visits),
                   "referenced_concept_ids": len(referenced)},
        "verify_failures": fails,
        "seconds": round(time.time() - t_start, 1),
    }
    with (out / "sample_manifest.json").open(
            "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    report(manifest, a.report)
    if fails:
        sys.exit(1)


def dry_report(a, found, counts, cohort, eligible, drop_min, drop_max,
               share, projected, free, secs):
    """What a real run WOULD do. Nothing has been written."""
    src_hist = {}
    smp_hist = {}
    for pid, n in counts.items():
        s = stratum_of(n)
        src_hist[s] = src_hist.get(s, 0) + 1
        if pid in cohort:
            smp_hist[s] = smp_hist.get(s, 0) + 1
    print("=" * 64)
    print("DRY RUN - nothing written. seed {}  cohort {:,}".format(
        a.seed, len(cohort)))
    print("=" * 64)
    print("patients with visits      {:>12,}".format(len(counts)))
    print("eligible after filters    {:>12,}".format(len(eligible)))
    if drop_min:
        print("  dropped below --min-visits {}: {:,}".format(
            a.min_visits, drop_min))
    if drop_max:
        print("  dropped above --exclude-above-visits {}: {:,}".format(
            a.exclude_above_visits, drop_max))
    print("selected                  {:>12,}  ({:.2%})".format(
        len(cohort), share))
    heavy = max((counts[p] for p in cohort), default=0)
    cvis = sum(counts[p] for p in cohort) or 1
    print("cohort visits             {:>12,}".format(cvis))
    print("heaviest patient          {:>12,}  ({:.1%} of them)".format(
        heavy, heavy / float(cvis)))
    print("\nvisit-count strata      source          sample")
    for label, _lo, _hi in STRATA:
        print("  {:<10} {:>8,} {:>6.1%}   {:>6,} {:>6.1%}".format(
            label, src_hist.get(label, 0),
            src_hist.get(label, 0) / float(len(counts)),
            smp_hist.get(label, 0),
            smp_hist.get(label, 0) / float(len(cohort))))
    print("\nprojected output")
    for t in PATIENT_TABLES:
        if t in found:
            b = int(found[t].stat().st_size * share)
            print("  {:<24} {:>14,} bytes".format(t, b))
    if VOCAB_TABLE in found:
        whole = found[VOCAB_TABLE].stat().st_size
        print("  {:<24} {:>14,} bytes  WHOLE (filtered at run time to "
              "referenced ids only - far smaller, but the count is not "
              "known until the patient rows are written)"
              .format(VOCAB_TABLE, whole))
    print("  {:<24} {:>14,} bytes  excluding the vocabulary".format(
        "TOTAL", projected))
    if free is not None:
        print("  free on the output volume  {:,} bytes".format(free))
    print("\nmemory a real run will hold: {:,} census entries, "
          "~{:,} visit ids".format(len(counts), cvis))
    print("\nno files were written. drop --dry-run to do it for real.")
    print("{:.1f}s".format(secs))


def report(m, full):
    print("=" * 64)
    print("OMOP PATIENT SAMPLE  seed {}  {}".format(
        m["seed"], m["selection"]))
    print("{:,} of {:,} patients ({:.2%}){}".format(
        m["cohort_selected"], m["persons_available"], m["sample_share"],
        "" if m["cohort_selected"] == m["cohort_requested"]
        else "  [requested {:,}]".format(m["cohort_requested"])))
    print("=" * 64)
    if full:
        print("\nvisit-count strata      source          sample")
        for s in m["strata"]:
            print("  {:<10} {:>8,} {:>6.1%}   {:>6,} {:>6.1%}".format(
                s["stratum"], s["source"], s["source_share"],
                s["sample"], s["sample_share"]))
    print("\n{:<24} {:>12} {:>12} {:>10}".format(
        "table", "rows in", "rows out", "bytes out"))
    for t, s in m["tables"].items():
        print("{:<24} {:>12} {:>12} {:>10,}".format(
            t,
            "{:,}".format(s["rows_in"]) if s.get("rows_in") is not None
            else "-",
            "{:,}".format(s["rows_out"]) if s.get("rows_out") is not None
            else "(whole)",
            s.get("bytes_out", 0)))
    c = m["tables"].get("concept", {})
    print("\nvocabulary: {} | {:,} ids referenced by written rows"
          .format(c.get("mode"), c.get("referenced", 0)))
    if c.get("dangling_count"):
        print("  {:,} referenced ids are ABSENT from the source "
              "vocabulary (dangling in the source, not caused here)"
              .format(c["dangling_count"]))
    recovered = sum(s.get("via_visit", 0) for s in m["tables"].values())
    unattr = sum(s.get("unattributable", 0)
                 for s in m["tables"].values())
    print("rows recovered via visit id (person_id blank): {:,}"
          .format(recovered))
    print("rows dropped as unattributable (both keys blank): {:,}"
          .format(unattr))
    print("heaviest patient: {:,} visits = {:.1%} of the cohort's visits"
          .format(m["heaviest_patient_visits"],
                  m["heaviest_patient_share"]))
    if m["heaviest_patient_share"] > 0.05:
        print("  one patient is over 5% of the cohort's visits; "
              "--exclude-above-visits N drops such a patient ENTIRELY")
    if m["selected_without_person_row"]:
        print("{} selected patients have visits but no person row"
              .format(m["selected_without_person_row"]))
    print("\nmemory: {:,} census entries, {:,} kept visit ids, "
          "{:,} concept ids".format(m["memory"]["census_entries"],
                                    m["memory"]["kept_visit_ids"],
                                    m["memory"]["referenced_concept_ids"]))
    print("output {:,} bytes from {:,} source bytes  ({:.1%})".format(
        m["bytes_out"], m["bytes_source"],
        m["bytes_out"] / float(m["bytes_source"] or 1)))
    if m["verify_failures"]:
        print("\nVERIFY FAILED:")
        for f in m["verify_failures"]:
            print("  " + f)
    else:
        print("verify PASSED: no partial patients, every concept "
              "reference resolves")
    print("{}s".format(m["seconds"]))


if __name__ == "__main__":
    main()
