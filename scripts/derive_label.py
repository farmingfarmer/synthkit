"""Derive an outcome column on the tidy per-visit dataset.

    python scripts/derive_label.py --in tidy_visits.csv
                                   [-o tidy_visits_labeled.csv]
                                   [--window 30] [--report]

Default outcome: `returned_within_<W>d` — for each visit, did the
SAME patient have another visit within W days? Computed per person
in date order; a patient's final visit is labeled 0 (no known
return) and flagged in `is_last_visit` so it can be excluded if a
right-censoring-clean cohort is wanted.

This is a DERIVED label on observed timing — it is NOT planted
truth. On data with no embedded causality it is unpredictable by
construction, which is exactly what the null diagnostic measures.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:                       # console-safe on Windows terminals
    from _console import console_safe
    console_safe()
except Exception:
    pass

csv.field_size_limit(1 << 22)


def parse_date(v: str):
    v = (v or "").strip()
    if not v:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(v, fmt).date()
        except ValueError:
            continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("-o", "--out", default="")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    src = Path(a.src)
    if not src.exists():
        sys.exit("not found: {}".format(src))
    out = Path(a.out) if a.out else src.with_name(
        src.stem + "_labeled.csv")
    label = "returned_within_{}d".format(a.window)

    with src.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("no rows in {}".format(src))

    by_person = defaultdict(list)
    for i, r in enumerate(rows):
        by_person[r.get("person_id", "")].append(i)

    gaps = []
    undated = 0
    for pid, idxs in by_person.items():
        dated = [(parse_date(rows[i].get("visit_start_date", "")), i)
                 for i in idxs]
        undated += sum(1 for d, _ in dated if d is None)
        dated = sorted((d, i) for d, i in dated if d is not None)
        for k, (d, i) in enumerate(dated):
            if k + 1 < len(dated):
                gap = (dated[k + 1][0] - d).days
                rows[i]["days_to_next_visit"] = gap
                rows[i][label] = 1 if 0 <= gap <= a.window else 0
                rows[i]["is_last_visit"] = 0
                gaps.append(gap)
            else:
                rows[i]["days_to_next_visit"] = ""
                rows[i][label] = 0
                rows[i]["is_last_visit"] = 1
        for d, i in [(x, y) for x, y in
                     [(parse_date(rows[j].get("visit_start_date", "")), j)
                      for j in idxs] if x is None]:
            rows[i]["days_to_next_visit"] = ""
            rows[i][label] = ""
            rows[i]["is_last_visit"] = ""

    cols = list(rows[0].keys())
    for extra in ("days_to_next_visit", label, "is_last_visit"):
        if extra not in cols:
            cols.append(extra)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    pos = sum(1 for r in rows if str(r.get(label)) == "1")
    lastv = sum(1 for r in rows if str(r.get("is_last_visit")) == "1")
    n = len(rows)
    print("WROTE {} : {} rows, outcome `{}`".format(out, n, label))
    print("  prevalence: {}/{} = {:.1%}".format(pos, n, pos / n))
    if a.report:
        gaps.sort()
        if gaps:
            def pct(p):
                return gaps[min(len(gaps) - 1, int(p * len(gaps)))]
            print("  patients: {} | visits/patient median {:.0f}"
                  .format(len(by_person),
                          sorted(len(v) for v in by_person.values())
                          [len(by_person) // 2]))
            print("  gap days  p10 {} | median {} | p90 {}"
                  .format(pct(.10), pct(.50), pct(.90)))
        print("  last visits (labeled 0, right-censored): {} "
              "({:.1%})".format(lastv, lastv / n))
        if undated:
            print("  visits without a usable date: {}".format(undated))
        print("  NOTE: prevalence outside ~5-40% makes for a weak "
              "exam; tune --window to land it sensibly.")


if __name__ == "__main__":
    main()
