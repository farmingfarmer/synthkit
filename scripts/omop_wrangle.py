"""Wrangle an OMOP CSV extract into ONE TIDY ROW PER VISIT for
synthkit Phase 2 profiling. Standard library only — runs on the
locked-down Windows laptop and a development machine identically.

    python scripts/omop_wrangle.py --src DIR [-o tidy_visits.csv]
                                   [--report]

Source may be the mimic set (data/mimic_omop) or the real
extract folder (e.g. %USERPROFILE%\\Downloads). This is the local
re-implementation of the Redshift cohort -> visits -> facts
pattern: person demographics joined to each visit, with
per-visit medication / condition / procedure lists and a WIDE
pivot of the measurement panel.

Key normalization: visit ids arrive as ints AND as leading-zero
text in the SAME column (observed in the real extract). Every
key is normalized (strip, drop leading zeros) before joining, or
rows silently vanish.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:                       # console-safe on Windows terminals
    from _console import console_safe
    console_safe()
except Exception:
    pass

csv.field_size_limit(1 << 22)


# ---------------------------------------------------------------
# normalization helpers
# ---------------------------------------------------------------
def norm_key(v: str) -> str:
    """Visit/person keys: '023735 and 23735 must join."""
    v = (v or "").strip().strip("'\"")
    if not v or v.lower() == "nan":
        return ""
    v = v.lstrip("0")
    return v or "0"


def clean(v: str) -> str:
    """Literal 'nan' strings are a pandas export artifact, not data."""
    v = (v or "").strip()
    return "" if v.lower() in ("nan", "none", "null") else v


def parse_date(v: str):
    v = clean(v)
    if not v:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y", "%d/%m/%Y"):
        try:
            from datetime import datetime
            return datetime.strptime(v, fmt).date()
        except ValueError:
            continue
    return None


def snake(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", clean(name).lower()).strip("_")
    return s or "unnamed"


def num(v: str):
    v = clean(v)
    if not v:
        return None
    try:
        return float(v.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def read(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="folder holding the OMOP CSVs")
    ap.add_argument("-o", "--out", default="tidy_visits.csv")
    ap.add_argument("--max-labs", type=int, default=24,
                    help="pivot the N most common measurements")
    ap.add_argument("--report", action="store_true",
                    help="print a wrangle report to stdout")
    a = ap.parse_args()
    src = Path(a.src)
    if not src.exists():
        sys.exit("source folder not found: {}".format(src))

    rep = []

    def note(msg):
        rep.append(msg)

    # ---------- person ----------
    persons = {}
    for r in read(src / "person.csv"):
        pid = norm_key(r.get("person_id", ""))
        if not pid:
            continue
        yob = num(r.get("year_of_birth", ""))
        persons[pid] = {
            "year_of_birth": int(yob) if yob else "",
            "gender": clean(r.get("gender_source_value", "")),
            "race": clean(r.get("race_source_value", "")),
            "ethnicity": clean(r.get("ethnicity_source_value", "")),
        }
    note("person.csv: {} patients".format(len(persons)))

    # ---------- visits (the spine) ----------
    visits = {}
    dupes = 0
    for r in read(src / "visit_occurrence.csv"):
        vid = norm_key(r.get("visit_occurrence_id", ""))
        if not vid:
            continue
        if vid in visits:
            dupes += 1
            continue
        start = parse_date(r.get("visit_start_date", ""))
        end = parse_date(r.get("visit_end_date", ""))
        visits[vid] = {
            "visit_id": vid,
            "person_id": norm_key(r.get("person_id", "")),
            "visit_start_date": start,
            "visit_end_date": end,
            "visit_type": clean(r.get("visit_source_value", "")),
            "admitted_from": clean(
                r.get("admitted_from_source_value", "")),
            "span_days": ((end - start).days
                          if start and end else ""),
        }
    note("visit_occurrence.csv: {} visits ({} duplicate ids skipped)"
         .format(len(visits), dupes))

    # ---------- conditions per visit ----------
    cond = defaultdict(list)
    orphan_c = 0
    for r in read(src / "condition_occurrence.csv"):
        vid = norm_key(r.get("visit_occurrence_id", ""))
        name = clean(r.get("concept_name", "")) or \
            clean(r.get("condition_source_value", ""))
        if not vid or vid not in visits:
            orphan_c += 1
            continue
        if name and name.lower() != "no matching concept":
            cond[vid].append(name)
    note("condition_occurrence.csv: {} visits carry conditions "
         "({} rows without a resolvable visit)"
         .format(len(cond), orphan_c))

    # ---------- procedures per visit ----------
    proc = defaultdict(list)
    proc_qty = defaultdict(float)
    orphan_p = 0
    for r in read(src / "procedure_occurrence.csv"):
        vid = norm_key(r.get("visit_occurrence_id", ""))
        if not vid or vid not in visits:
            orphan_p += 1
            continue
        name = clean(r.get("concept_name", "")) or \
            clean(r.get("procedure_source_value", ""))
        if name:
            proc[vid].append(name)
        q = num(r.get("quantity", ""))
        if q:
            proc_qty[vid] += q
    note("procedure_occurrence.csv: {} visits carry procedures "
         "({} unresolvable)".format(len(proc), orphan_p))

    # ---------- drugs ACTIVE AT the visit date ----------
    # exposure spans are era-like; a drug counts for a visit when
    # the visit date falls inside [start, end] for that patient.
    by_person = defaultdict(list)
    for r in read(src / "drug_exposure.csv"):
        pid = norm_key(r.get("person_id", ""))
        s = parse_date(r.get("drug_exposure_start_date", ""))
        e = parse_date(r.get("drug_exposure_end_date", "")) or s
        name = clean(r.get("concept_name", "")) or \
            clean(r.get("drug_source_value", ""))
        if not pid or not s or not name:
            continue
        if name.lower() == "no matching concept":
            continue
        by_person[pid].append((s, e, name,
                               clean(r.get("route_source_value", ""))))
    drugs = defaultdict(list)
    routes = defaultdict(set)
    for vid, v in visits.items():
        d = v["visit_start_date"]
        if not d:
            continue
        for (s, e, name, route) in by_person.get(v["person_id"], []):
            if s <= d <= e:
                drugs[vid].append(name)
                if route:
                    routes[vid].add(route)
    note("drug_exposure.csv: {} patients with exposures; {} visits "
         "have >=1 active drug".format(len(by_person), len(drugs)))

    # ---------- measurements: EAV -> wide ----------
    counts = defaultdict(int)
    raw_meas = []
    for r in read(src / "measurement.csv"):
        vid = norm_key(r.get("visit_occurrence_id", ""))
        if not vid or vid not in visits:
            continue
        name = clean(r.get("measurement_source_value", ""))
        val = num(r.get("value_as_number", ""))
        if not name or val is None:
            continue
        unit = clean(r.get("unit_source_value", ""))
        key = snake(name) if not unit else snake(name)
        counts[key] += 1
        raw_meas.append((vid, key, val,
                         num(r.get("range_low", "")),
                         num(r.get("range_high", ""))))
    top = [k for k, _ in sorted(counts.items(),
                                key=lambda kv: -kv[1])][:a.max_labs]
    top_set = set(top)
    wide = defaultdict(dict)
    out_of_range = 0
    for vid, key, val, lo, hi in raw_meas:
        if key not in top_set:
            continue
        wide[vid][key] = val          # last value wins per visit
        if lo is not None and hi is not None and not (lo <= val <= hi):
            out_of_range += 1
    note("measurement.csv: {} distinct measures; pivoting top {}; "
         "{} values outside their own reference range"
         .format(len(counts), len(top), out_of_range))

    # ---------- assemble ----------
    cols = (["visit_id", "person_id", "visit_start_date",
             "visit_end_date", "span_days", "visit_type",
             "admitted_from", "age_at_visit", "year_of_birth",
             "gender", "race", "ethnicity"]
            + top
            + ["condition_count", "conditions",
               "procedure_count", "procedures", "procedure_quantity",
               "active_drug_count", "active_drugs", "drug_routes"])
    rows_out = []
    for vid, v in sorted(visits.items(),
                         key=lambda kv: (kv[1]["person_id"],
                                         kv[1]["visit_start_date"]
                                         or date(1900, 1, 1))):
        p = persons.get(v["person_id"], {})
        age = ""
        if p.get("year_of_birth") and v["visit_start_date"]:
            age = v["visit_start_date"].year - int(p["year_of_birth"])
        row = {
            "visit_id": vid,
            "person_id": v["person_id"],
            "visit_start_date": (v["visit_start_date"].isoformat()
                                 if v["visit_start_date"] else ""),
            "visit_end_date": (v["visit_end_date"].isoformat()
                               if v["visit_end_date"] else ""),
            "span_days": v["span_days"],
            "visit_type": v["visit_type"],
            "admitted_from": v["admitted_from"],
            "age_at_visit": age,
            "year_of_birth": p.get("year_of_birth", ""),
            "gender": p.get("gender", ""),
            "race": p.get("race", ""),
            "ethnicity": p.get("ethnicity", ""),
            "condition_count": len(cond.get(vid, [])),
            "conditions": "; ".join(sorted(set(cond.get(vid, [])))),
            "procedure_count": len(proc.get(vid, [])),
            "procedures": "; ".join(sorted(set(proc.get(vid, [])))),
            "procedure_quantity": (proc_qty.get(vid, "") or ""),
            "active_drug_count": len(set(drugs.get(vid, []))),
            "active_drugs": "; ".join(sorted(set(drugs.get(vid, [])))),
            "drug_routes": "; ".join(sorted(routes.get(vid, set()))),
        }
        for k in top:
            row[k] = wide.get(vid, {}).get(k, "")
        rows_out.append(row)

    outp = Path(a.out)
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_out)
    note("WROTE {} : {} rows x {} columns".format(
        outp, len(rows_out), len(cols)))

    # coverage stats — what a profiler will actually find
    filled = {c: sum(1 for r in rows_out if str(r.get(c, "")) != "")
              for c in cols}
    note("column fill rates (top labs): " + ", ".join(
        "{} {:.0%}".format(c, filled[c] / max(1, len(rows_out)))
        for c in top[:6]))
    note("visits with any drug: {:.0%} | any condition: {:.0%} | "
         "any procedure: {:.0%}".format(
             filled["active_drugs"] / max(1, len(rows_out)),
             filled["conditions"] / max(1, len(rows_out)),
             filled["procedures"] / max(1, len(rows_out))))
    print("\n".join("  " + m for m in rep) if a.report
          else rep[-1])


if __name__ == "__main__":
    main()
