"""Measure a large OMOP extract WITHOUT reading it into memory and
WITHOUT printing a single record. Standard library only; runs on the
locked-down Windows laptop where the real extract lives.

    python scripts/omop_census.py --src DIR [-o census.json]
                                  [--report] [--cohort 4000]

This exists because the extract cannot leave that machine and cannot
be read here. It streams every table once and emits STATISTICS ONLY
— counts, rates, lengths, histograms — so the numbers can be pasted
back and used to size scripts/omop_sample.py correctly.

The privacy contract, enforced structurally rather than by habit:

  - Column VALUES are never printed. Text columns yield length
    distributions and character-class counts, never content.
  - The one exception is category LEVELS, and only for columns on
    an explicit whitelist (gender, route, unit ...), only when the
    column has <= --max-levels distinct values, and only for levels
    appearing at least k times. Identifier-like and free clinical
    text columns (person_source_value, sig, concept_name, the
    *_source_value clinical phrase columns) are additionally named
    in a deny list and can never emit levels whatever the counts.
  - Headers ARE printed in full. A header is schema, not data, and
    getting the exact truth is the point of this pass.

Memory is bounded by DISTINCT PERSONS, never by rows. That bound is
itself the main thing being measured, so it is capped and reported
rather than trusted: past --max-track distinct persons the per-person
tracking stops and says so instead of consuming the machine.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:                       # console-safe on Windows terminals
    from _console import console_safe
    console_safe()
except Exception:
    pass

# Same normalization the wrangler joins on, imported rather than
# copied so the two can never drift: a census that counts persons
# differently from the sampler would size it wrong.
from omop_wrangle import norm_key

csv.field_size_limit(1 << 22)


# ---------------------------------------------------------------
# what the extract is made of
# ---------------------------------------------------------------
PATIENT_TABLES = ["person", "visit_occurrence", "condition_occurrence",
                  "drug_exposure", "measurement",
                  "procedure_occurrence"]
VOCAB_TABLES = ["concept"]
ALL_TABLES = PATIENT_TABLES + VOCAB_TABLES

# Levels may be emitted ONLY for these. Everything else is counted.
LEVEL_WHITELIST = {
    "gender_source_value", "race_source_value", "ethnicity_source_value",
    "visit_source_value", "admitted_from_source_value",
    "discharged_to_source_value", "route_source_value",
    "unit_source_value", "dose_unit_source_value",
    "condition_status_source_value", "stop_reason",
    "domain_id", "vocabulary_id", "concept_class_id",
    "standard_concept", "invalid_reason",
}
# Belt and braces: identifier-like or free clinical text. Named here
# so that adding one to the whitelist by accident still cannot leak.
NEVER_LEVELS = {
    "person_source_value", "sig", "lot_number", "concept_name",
    "concept_code", "condition_source_value", "drug_source_value",
    "procedure_source_value", "measurement_source_value",
    "modifier_source_value", "value_source_value",
}

KEY_SUFFIX = "_id"
NUMERIC_COLS = {
    "value_as_number", "range_low", "range_high", "quantity",
    "refills", "days_supply", "year_of_birth", "month_of_birth",
    "day_of_birth",
}
DATE_FORMATS = [
    ("%Y-%m-%d", "YYYY-MM-DD"),
    ("%m/%d/%Y", "MM/DD/YYYY"),
    ("%d/%m/%Y", "DD/MM/YYYY"),
    ("%m/%d/%y", "MM/DD/YY"),
    ("%Y/%m/%d", "YYYY/MM/DD"),
    ("%Y-%m-%d %H:%M:%S", "YYYY-MM-DD hh:mm:ss"),
    ("%Y-%m-%d %H:%M", "YYYY-MM-DD hh:mm"),
    ("%Y-%m-%dT%H:%M:%S", "YYYY-MM-DDThh:mm:ss"),
    ("%m/%d/%Y %H:%M", "MM/DD/YYYY hh:mm"),
    ("%m/%d/%Y %H:%M:%S", "MM/DD/YYYY hh:mm:ss"),
    ("%H:%M:%S", "hh:mm:ss"),
    ("%H:%M", "hh:mm"),
]

PROGRESS_EVERY = 250000


# ---------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------
def is_date_col(col: str) -> bool:
    return (col.endswith("_date") or col.endswith("_datetime")
            or col == "measurement_time")


def is_key_col(col: str) -> bool:
    return col.endswith(KEY_SUFFIX)


def classify(col: str) -> str:
    if is_date_col(col):
        return "date"
    if col in NUMERIC_COLS:
        return "numeric"
    if is_key_col(col):
        return "key"
    return "text"


def pct_from_counter(counter, q):
    """Exact percentile over a Counter of integer observations."""
    total = sum(counter.values())
    if not total:
        return None
    rank = q * total
    seen = 0
    for k in sorted(counter):
        seen += counter[k]
        if seen >= rank:
            return k
    return max(counter)


def mean_from_counter(counter):
    total = sum(counter.values())
    if not total:
        return None
    return sum(k * v for k, v in counter.items()) / float(total)


def human_bytes(n):
    if n is None:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024.0 or u == "TB":
            return "{:.1f} {}".format(f, u)
        f /= 1024.0
    return "{:.1f} TB".format(f)


def machine_memory():
    """Total/available RAM, best effort, both platforms."""
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual",
                             ctypes.c_ulonglong)]
            st = MS()
            st.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullTotalPhys), int(st.ullAvailPhys)
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        avail = None
        try:
            avail = (os.sysconf("SC_PAGE_SIZE")
                     * os.sysconf("SC_AVPHYS_PAGES"))
        except (ValueError, OSError):
            pass
        return int(total), (int(avail) if avail else None)
    except Exception:
        return None, None


def iter_rows(reader, errors):
    """Yield rows, counting rather than dying on malformed ones."""
    while True:
        try:
            row = next(reader)
        except StopIteration:
            return
        except csv.Error:
            errors[0] += 1
            if errors[0] > 1000:
                return
            continue
        yield row


class PersonCounter(object):
    """rows-per-person, with a hard memory ceiling that reports itself."""

    def __init__(self, cap):
        self.cap = cap
        self.counts = {}
        self.capped = False
        self.rows_after_cap = 0

    def add(self, pid):
        if pid in self.counts:
            self.counts[pid] += 1
        elif self.capped or len(self.counts) >= self.cap:
            self.capped = True
            self.rows_after_cap += 1
        else:
            self.counts[pid] = 1

    def histogram(self):
        h = Counter()
        for v in self.counts.values():
            h[v] += 1
        return h


class ColStat(object):
    """Per-column aggregates. Holds no value, only shapes of values."""

    def __init__(self, name):
        self.name = name
        self.kind = classify(name)
        self.blank = 0
        self.lengths = Counter()
        self.newline = 0
        self.delim = 0
        self.quote = 0
        self.date_fmt = Counter()
        self.numeric_ok = 0
        self.numeric_bad = 0
        self.levels = Counter()
        self.levels_abandoned = False
        self.levels_seen = 0
        self.lead_zero = 0
        self.non_digit = 0
        self.quoted_key = 0
        self.present = 0
        self.undecodable = 0

    def observe(self, v, max_levels):
        if v is None:
            v = ""
        s = v.strip()
        if not s or s.lower() in ("nan", "none", "null"):
            self.blank += 1
            return
        self.present += 1
        if "�" in v:     # bytes that were not valid UTF-8
            self.undecodable += 1
        if self.kind == "key":
            raw = v.strip()
            if raw != raw.strip("'\""):
                self.quoted_key += 1
            body = raw.strip("'\"")
            if len(body) > 1 and body[0] == "0":
                self.lead_zero += 1
            if not body.isdigit():
                self.non_digit += 1
            return
        if self.kind == "date":
            self.date_fmt[self._date_shape(s)] += 1
            return
        if self.kind == "numeric":
            try:
                float(s.replace(",", ""))
                self.numeric_ok += 1
            except ValueError:
                self.numeric_bad += 1
            return
        # text: shape only
        self.lengths[len(v)] += 1
        if "\n" in v or "\r" in v:
            self.newline += 1
        if "," in v:
            self.delim += 1
        if '"' in v:
            self.quote += 1
        if self.name in LEVEL_WHITELIST and self.name not in NEVER_LEVELS:
            if not self.levels_abandoned:
                if s in self.levels or len(self.levels) < max_levels * 4:
                    self.levels[s] += 1
                    self.levels_seen = len(self.levels)
                else:
                    self.levels_abandoned = True
                    self.levels_seen = len(self.levels)
                    self.levels.clear()

    @staticmethod
    def _date_shape(s):
        for fmt, label in DATE_FORMATS:
            try:
                datetime.strptime(s, fmt)
                return label
            except ValueError:
                continue
        return "UNPARSED"

    def report(self, rows, max_levels, k):
        d = {"column": self.name, "kind": self.kind,
             "blank": self.blank,
             "blank_rate": round(self.blank / float(rows), 4)
             if rows else None}
        if self.undecodable:
            d["undecodable_bytes_in"] = self.undecodable
        if self.kind == "key":
            d["leading_zero"] = self.lead_zero
            d["non_digit"] = self.non_digit
            d["quoted"] = self.quoted_key
            # Uniform zero-padding cannot collide; a MIXED rate is the
            # only case where two spellings can be one person.
            if self.present and 0 < self.lead_zero < self.present:
                d["format_MIXED"] = True
                d["leading_zero_rate"] = round(
                    self.lead_zero / float(self.present), 4)
        elif self.kind == "date":
            d["formats"] = dict(self.date_fmt.most_common())
        elif self.kind == "numeric":
            d["parseable"] = self.numeric_ok
            d["unparseable"] = self.numeric_bad
        else:
            if self.lengths:
                d["len_mean"] = round(mean_from_counter(self.lengths), 1)
                d["len_p50"] = pct_from_counter(self.lengths, 0.50)
                d["len_p99"] = pct_from_counter(self.lengths, 0.99)
                d["len_max"] = max(self.lengths)
            d["embedded_newline"] = self.newline
            d["contains_comma"] = self.delim
            d["contains_quote"] = self.quote
            d["distinct_note"] = ("levels withheld: not whitelisted"
                                  if self.name not in LEVEL_WHITELIST
                                  else None)
            if (self.name in LEVEL_WHITELIST
                    and self.name not in NEVER_LEVELS):
                if self.levels_abandoned or len(self.levels) > max_levels:
                    d["distinct_note"] = (
                        "levels withheld: over {} distinct"
                        .format(max_levels))
                    d["distinct_at_least"] = self.levels_seen
                else:
                    kept = {lv: c for lv, c in self.levels.items()
                            if c >= k}
                    supp = sum(c for lv, c in self.levels.items()
                               if c < k)
                    if supp:
                        kept["OTHER_SUPPRESSED"] = supp
                    d["levels"] = dict(
                        sorted(kept.items(), key=lambda kv: -kv[1]))
                    d["distinct_note"] = None
        return d


# ---------------------------------------------------------------
def find_table(src: Path, table: str):
    for name in (table + "_full.csv", table + ".csv"):
        p = src / name
        if p.exists():
            return p
    return None


def scan_table(path, table, args, progress):
    """One streaming pass. Returns a stats dict and, for the patient
    tables, the per-person row counter that sizes the sampler."""
    size = path.stat().st_size
    stat = {"table": table, "file": path.name, "bytes": size,
            "bytes_human": human_bytes(size)}
    errors = [0]
    people = PersonCounter(args.max_track)
    rows = 0
    pid_changes = 0
    prev_pid = None
    monotone = True
    prev_num = None
    t0 = time.time()

    with path.open("rb") as fb:
        head = fb.read(3)
    stat["bom"] = (head == b"\xef\xbb\xbf")
    with path.open("rb") as fb:
        first = fb.readline()
    stat["crlf"] = first.endswith(b"\r\n")

    with path.open(encoding="utf-8-sig", newline="",
                   errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            stat["error"] = "file is empty"
            return stat, people
        header = [h.strip() for h in header]
        stat["header"] = header
        stat["columns"] = len(header)
        cols = [ColStat(h) for h in header]
        idx = {h: i for i, h in enumerate(header)}
        pid_i = idx.get("person_id")
        ragged_short = 0
        ragged_long = 0

        for row in iter_rows(reader, errors):
            rows += 1
            if len(row) != len(header):
                if len(row) < len(header):
                    ragged_short += 1
                else:
                    ragged_long += 1
            for i, cs in enumerate(cols):
                cs.observe(row[i] if i < len(row) else "", args.max_levels)
            if pid_i is not None and pid_i < len(row):
                raw = row[pid_i].strip()
                pid = norm_key(raw)
                if pid:
                    people.add(pid)
                    if pid != prev_pid:
                        if prev_pid is not None:
                            pid_changes += 1
                        prev_pid = pid
                        if monotone:
                            try:
                                n = int(pid)
                            except ValueError:
                                monotone = False
                            else:
                                if prev_num is not None and n < prev_num:
                                    monotone = False
                                prev_num = n
            if rows % PROGRESS_EVERY == 0:
                progress(table, rows, time.time() - t0)
            if args.limit and rows >= args.limit:
                stat["LIMITED"] = args.limit
                break

    elapsed = time.time() - t0
    stat["rows"] = rows
    stat["parse_errors"] = errors[0]
    stat["ragged_short"] = ragged_short
    stat["ragged_long"] = ragged_long
    stat["seconds"] = round(elapsed, 1)
    stat["bytes_per_row"] = round(size / float(rows), 1) if rows else None
    stat["columns_detail"] = [c.report(rows, args.max_levels, args.k)
                              for c in cols]

    if pid_i is not None:
        stat["has_person_id"] = True
        stat["person_id_position"] = pid_i + 1
        stat["distinct_persons"] = len(people.counts)
        pid_col = stat["columns_detail"][pid_i]
        stat["person_id_format_mixed"] = bool(pid_col.get("format_MIXED"))
        stat["person_id_leading_zero"] = pid_col.get("leading_zero", 0)
        stat["person_tracking_capped"] = people.capped
        stat["rows_after_person_cap"] = people.rows_after_cap
        if people.counts:
            h = people.histogram()
            stat["rows_per_person"] = {
                "mean": round(mean_from_counter(h), 2),
                "p50": pct_from_counter(h, 0.50),
                "p95": pct_from_counter(h, 0.95),
                "p99": pct_from_counter(h, 0.99),
                "max": max(people.counts.values()),
            }
        stat["grouped_by_person"] = {
            "id_changes": pid_changes,
            "changes_per_distinct": (
                round(pid_changes / float(max(1, len(people.counts))), 3)),
            "monotone_person_id": monotone,
        }
    else:
        stat["has_person_id"] = False
    return stat, people


def visit_histogram(people):
    """Visits-per-person, bucketed — the shape hierarchical
    generation learns from, and the thing a sample must preserve."""
    buckets = [("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5),
               ("6-10", 6, 10), ("11+", 11, 10 ** 9)]
    out = []
    total = len(people.counts)
    for label, lo, hi in buckets:
        n = sum(1 for v in people.counts.values() if lo <= v <= hi)
        out.append({"visits": label, "persons": n,
                    "share": round(n / float(total), 4) if total else 0.0})
    return out


# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Statistics-only census of a large OMOP extract.")
    ap.add_argument("--src", required=True,
                    help="folder holding the OMOP CSVs")
    ap.add_argument("-o", "--out", default="census.json")
    ap.add_argument("--report", action="store_true",
                    help="print the full human-readable report")
    ap.add_argument("--cohort", type=int, default=4000,
                    help="project sample sizes for this many patients")
    ap.add_argument("--k", type=int, default=10,
                    help="suppress category levels seen fewer than k times")
    ap.add_argument("--max-levels", type=int, default=50,
                    help="withhold levels for columns above this many "
                         "distinct values")
    ap.add_argument("--max-track", type=int, default=2000000,
                    help="stop per-person tracking past this many "
                         "distinct persons rather than exhaust RAM")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N rows per table (dry run)")
    a = ap.parse_args()

    src = Path(a.src)
    if not src.exists():
        sys.exit("source folder not found: {}".format(src))
    if not src.is_dir():
        sys.exit("--src must be a folder, got a file: {}".format(src))

    found = {}
    missing = []
    for t in ALL_TABLES:
        p = find_table(src, t)
        if p is None:
            missing.append(t)
        else:
            found[t] = p
    if not found:
        sys.exit("no OMOP tables found in {}\n"
                 "expected files named <table>_full.csv or <table>.csv "
                 "for: {}".format(src, ", ".join(ALL_TABLES)))
    if missing:
        sys.stderr.write(
            "WARNING: {} table(s) not found and skipped: {}\n"
            .format(len(missing), ", ".join(missing)))

    def progress(table, rows, elapsed):
        rate = rows / elapsed if elapsed > 0 else 0.0
        sys.stderr.write("  {:<24} {:>12,} rows  {:>6.0f}s  "
                         "{:>9,.0f} rows/s\n"
                         .format(table, rows, elapsed, rate))
        sys.stderr.flush()

    total_ram, avail_ram = machine_memory()
    try:
        du = shutil.disk_usage(str(src))
        disk = {"total": du.total, "free": du.free}
    except Exception:
        disk = {"total": None, "free": None}

    census = {
        "generated_by": "scripts/omop_census.py",
        "source": str(src.resolve()),
        "contract": "statistics only - no record value is ever emitted; "
                    "headers are schema and are emitted in full",
        "settings": {"k": a.k, "max_levels": a.max_levels,
                     "max_track": a.max_track, "limit": a.limit,
                     "cohort": a.cohort},
        "machine": {"os": os.name, "python": sys.version.split()[0],
                    "ram_total": total_ram, "ram_available": avail_ram,
                    "disk_total": disk["total"], "disk_free": disk["free"]},
        "tables_missing": missing,
        "tables": [],
    }

    visit_people = None
    person_table_count = None
    t_start = time.time()
    for t in ALL_TABLES:
        if t not in found:
            continue
        sys.stderr.write("scanning {} ...\n".format(found[t].name))
        sys.stderr.flush()
        try:
            stat, people = scan_table(found[t], t, a, progress)
        except MemoryError:
            sys.exit("OUT OF MEMORY scanning {}.\n"
                     "Re-run with a lower --max-track (currently {}); "
                     "the per-person tracking is the only structure "
                     "that grows with the extract."
                     .format(found[t].name, a.max_track))
        census["tables"].append(stat)
        if t == "visit_occurrence":
            visit_people = people
        if t == "person":
            person_table_count = stat.get("distinct_persons")

    # ---------- cohort sizing: what a sample would actually cost ----
    sizing = {"cohort": a.cohort}
    if visit_people is not None and visit_people.counts:
        n_persons = len(visit_people.counts)
        sizing["persons_with_visits"] = n_persons
        sizing["persons_in_person_table"] = person_table_count
        sizing["visit_histogram"] = visit_histogram(visit_people)
        vh = visit_people.histogram()
        sizing["visits_per_person"] = {
            "mean": round(mean_from_counter(vh), 2),
            "p50": pct_from_counter(vh, 0.50),
            "p95": pct_from_counter(vh, 0.95),
            "p99": pct_from_counter(vh, 0.99),
            "max": max(visit_people.counts.values()),
        }
        sizing["multi_visit_persons"] = sum(
            1 for v in visit_people.counts.values() if v >= 2)
        share = min(1.0, a.cohort / float(n_persons)) if n_persons else 0.0
        sizing["sample_share"] = round(share, 6)
        sizing["cohort_feasible"] = a.cohort <= n_persons
        proj = []
        total_out = 0
        for st in census["tables"]:
            if st["table"] in VOCAB_TABLES:
                # Has no person_id, so it does not shrink with the
                # cohort. This is its WHOLE size - the upper bound, and
                # what makes the cohort lever nearly inert on its own.
                # scripts/omop_sample.py filters it to the ids the
                # sampled rows actually reference, which is far smaller.
                b = st["bytes"]
                proj.append({"table": st["table"],
                             "mode": "whole (sampler filters)",
                             "bytes": b, "bytes_human": human_bytes(b)})
                total_out += b
            elif st.get("has_person_id"):
                b = int(st["bytes"] * share)
                proj.append({"table": st["table"], "mode": "sampled",
                             "rows_estimate": int(st["rows"] * share),
                             "bytes": b, "bytes_human": human_bytes(b)})
                total_out += b
        sizing["projected_output"] = proj
        sizing["projected_output_bytes"] = total_out
        sizing["projected_output_human"] = human_bytes(total_out)
        if disk["free"] is not None and total_out > disk["free"]:
            sizing["DISK_WARNING"] = (
                "projected output {} exceeds free disk {}"
                .format(human_bytes(total_out), human_bytes(disk["free"])))
        # the memory the sampler itself will need
        sizing["sampler_census_entries"] = n_persons
        sizing["sampler_kept_visits_estimate"] = int(
            sum(visit_people.counts.values()) * share)
    else:
        sizing["error"] = ("visit_occurrence not scanned - cohort sizing "
                           "unavailable")
    census["sizing"] = sizing
    census["seconds_total"] = round(time.time() - t_start, 1)

    outp = Path(a.out)
    with outp.open("w", encoding="utf-8") as f:
        json.dump(census, f, indent=1, sort_keys=False)

    print_report(census, a.report)


# ---------------------------------------------------------------
def print_report(census, full):
    m = census["machine"]
    print("=" * 66)
    print("OMOP CENSUS - statistics only, no record values")
    print("source: {}".format(census["source"]))
    print("machine: {} python {} | RAM {} total, {} free | "
          "disk {} free".format(
              m["os"], m["python"], human_bytes(m["ram_total"]),
              human_bytes(m["ram_available"]),
              human_bytes(m["disk_free"])))
    if census["tables_missing"]:
        print("MISSING TABLES: {}".format(
            ", ".join(census["tables_missing"])))
    print("=" * 66)

    print("\n-- FILES " + "-" * 56)
    print("{:<24} {:>12} {:>10} {:>5} {:>7}".format(
        "table", "rows", "size", "cols", "errors"))
    for t in census["tables"]:
        print("{:<24} {:>12,} {:>10} {:>5} {:>7}".format(
            t["table"], t.get("rows", 0), t["bytes_human"],
            t.get("columns", 0), t.get("parse_errors", 0)))
        if t.get("LIMITED"):
            print("    (LIMITED to {} rows - not a full census)"
                  .format(t["LIMITED"]))
        if t.get("ragged_short") or t.get("ragged_long"):
            print("    ragged rows: {} short, {} long".format(
                t.get("ragged_short"), t.get("ragged_long")))
        if t.get("bom") or t.get("crlf"):
            marks = []
            if t.get("bom"):
                marks.append("BOM")
            if t.get("crlf"):
                marks.append("CRLF")
            print("    {}".format(" + ".join(marks)))

    print("\n-- HEADERS (exact) " + "-" * 46)
    for t in census["tables"]:
        print("{}:".format(t["file"]))
        print("  " + ",".join(t.get("header", [])))

    print("\n-- PERSON KEYS " + "-" * 51)
    for t in census["tables"]:
        if not t.get("has_person_id"):
            print("{:<24} no person_id column".format(t["table"]))
            continue
        g = t["grouped_by_person"]
        print("{:<24} col {:>2} | {:>9,} distinct | grouped {} "
              "(changes/distinct {}) | monotone {}".format(
                  t["table"], t["person_id_position"],
                  t["distinct_persons"],
                  "yes" if g["changes_per_distinct"] <= 1.05 else "no",
                  g["changes_per_distinct"],
                  "yes" if g["monotone_person_id"] else "no"))
        if t.get("person_id_format_mixed"):
            print("    person_id FORMAT MIXED: {} values zero-padded - "
                  "two spellings may be one person, normalization is "
                  "load-bearing".format(t["person_id_leading_zero"]))
        if t.get("person_tracking_capped"):
            print("    CAPPED: distinct persons exceeded --max-track; "
                  "{} rows not attributed".format(
                      t["rows_after_person_cap"]))
        rp = t.get("rows_per_person")
        if rp:
            print("    rows/person  mean {} p50 {} p95 {} p99 {} max {}"
                  .format(rp["mean"], rp["p50"], rp["p95"], rp["p99"],
                          rp["max"]))

    if not full:
        print("\n(run with --report for per-column detail)")
        print_sizing(census)
        return

    print("\n-- BLANK RATES, KEY HYGIENE, TEXT SHAPE " + "-" * 26)
    for t in census["tables"]:
        print("{}:".format(t["table"]))
        for c in t.get("columns_detail", []):
            bits = ["{:<34}".format(c["column"]),
                    "blank {:>6.1%}".format(c["blank_rate"] or 0.0)]
            if c["kind"] == "key":
                bits.append("lead0 {} nondigit {} quoted {}".format(
                    c["leading_zero"], c["non_digit"], c["quoted"]))
            elif c["kind"] == "date":
                bits.append("formats " + ", ".join(
                    "{}={}".format(k, v)
                    for k, v in list(c["formats"].items())[:4]))
            elif c["kind"] == "numeric":
                bits.append("parseable {} unparseable {}".format(
                    c["parseable"], c["unparseable"]))
            else:
                if "len_max" in c:
                    bits.append("len p50 {} p99 {} max {}".format(
                        c["len_p50"], c["len_p99"], c["len_max"]))
                if c.get("embedded_newline"):
                    bits.append("NEWLINES {}".format(c["embedded_newline"]))
                if c.get("contains_comma"):
                    bits.append("commas {}".format(c["contains_comma"]))
                if c.get("levels"):
                    bits.append("levels {}".format(
                        json.dumps(c["levels"], ensure_ascii=True)))
                elif c.get("distinct_note"):
                    bits.append(c["distinct_note"])
            if c.get("undecodable_bytes_in"):
                bits.append("UNDECODABLE {}".format(
                    c["undecodable_bytes_in"]))
            print("  " + "  ".join(bits))

    print_sizing(census)


def print_sizing(census):
    s = census["sizing"]
    print("\n-- COHORT SIZING " + "-" * 49)
    if "error" in s:
        print(s["error"])
    else:
        print("persons with >=1 visit        {:>12,}".format(
            s["persons_with_visits"]))
        if s.get("persons_in_person_table") is not None:
            print("persons in person table       {:>12,}".format(
                s["persons_in_person_table"]))
        print("persons with >=2 visits       {:>12,}".format(
            s["multi_visit_persons"]))
        v = s["visits_per_person"]
        print("visits/person  mean {} p50 {} p95 {} p99 {} max {}"
              .format(v["mean"], v["p50"], v["p95"], v["p99"], v["max"]))
        print("visit-count histogram:")
        for b in s["visit_histogram"]:
            print("  {:>5} visits  {:>10,} persons  {:>6.1%}".format(
                b["visits"], b["persons"], b["share"]))
        print("\nfor a {:,}-patient sample ({:.3%} of persons):".format(
            s["cohort"], s["sample_share"]))
        if not s["cohort_feasible"]:
            print("  IMPOSSIBLE: only {:,} persons have visits".format(
                s["persons_with_visits"]))
        for p in s["projected_output"]:
            print("  {:<24} {:<14} {:>10}".format(
                p["table"], p["mode"], p["bytes_human"]))
        print("  {:<24} {:<14} {:>10}".format(
            "TOTAL", "", s["projected_output_human"]))
        if s.get("DISK_WARNING"):
            print("  DISK WARNING: {}".format(s["DISK_WARNING"]))
        print("\nsampler memory: {:,} census entries + ~{:,} kept visit "
              "ids".format(s["sampler_census_entries"],
                           s["sampler_kept_visits_estimate"]))
    print("\nscanned in {}s".format(census["seconds_total"]))


if __name__ == "__main__":
    main()
