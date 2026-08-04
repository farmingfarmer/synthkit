"""Measure what each column can SUPPORT, before deciding what to
model. Read-only, statistics only, standard library.

    python scripts/support_profile.py --in tidy_visits.csv
                                      [--model condnet_model.json]
                                      [-o support.json] [--report]

Column names are schema and are printed. No column VALUE is ever
printed - only counts, coverage, correlations and distinct counts.

Why coverage alone is the wrong primitive
-----------------------------------------
"Measured at 40% of visits" conflates two opposite situations: a panel
drawn at every visit for 40% of patients, and one drawn at 40% of
visits for everyone. The first is a cohort-defining variable, the
second a sampling artifact, and a patient-level generative model needs
them treated differently. So coverage is reported per visit AND per
patient.

More importantly, what the longitudinal model actually gates on is
PAIRS. condnet needs >= 4k pairs to set a steadiness target
(condnet.py:1601) and >= 30 to measure one (condnet.py:1977), and
those pairs must come from enough distinct patients to be publishable
at k. A column at 50% visit coverage spread thinly across patients
yields almost no pairs at all. This measures the pairs directly rather
than a proxy for them.

The variable-lag column
-----------------------
condnet measures steadiness by dropping missing values and pairing
what is left, so "the next present value" can be many visits later.
This reports both:

  as-measured   drop missing, pair what remains (what the code does)
  adjacent      pair only CONSECUTIVE visits where BOTH are present

Their difference is the measurement error the current code carries on
that column. It is zero by construction wherever a column is complete.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:                       # console-safe on Windows terminals
    from _console import console_safe
    console_safe()
except Exception:
    pass

csv.field_size_limit(1 << 22)

ID_COLS = {"visit_id", "person_id", "visit_start_date",
           "visit_end_date"}
LIST_SEP = "; "

TIERS = ["longitudinal", "cross_sectional", "presence", "drop"]


def num(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def corr(pairs):
    if len(pairs) < 2:
        return None
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    nu = sum((x - ma) * (y - mb) for x, y in pairs)
    de = (sum((x - ma) ** 2 for x in a)
          * sum((y - mb) ** 2 for y in b)) ** 0.5
    if not de:
        return None
    return nu / de


def classify(s, args):
    """Which role can this column's support sustain?

    Deliberately a demotion ladder rather than a single cutoff. A
    sparse column still carries the fact that the test was ordered,
    which is often the clinical signal and costs almost nothing in
    model size or in how identifying the record becomes."""
    if s["patients_present"] < args.k:
        return "drop", "fewer than k={} patients".format(args.k)
    if s["distinct"] <= 1:
        return "drop", "constant"
    if s["top_share"] is not None and s["top_share"] >= args.degenerate:
        return "drop", "degenerate: one value is {:.0%} of it".format(
            s["top_share"])
    if (s["patients_adjacent"] >= args.min_pair_patients
            and s["adjacent_pairs"] >= 4 * args.k):
        return "longitudinal", "{} patients give {} adjacent pairs".format(
            s["patients_adjacent"], s["adjacent_pairs"])
    if s["patients_present"] >= args.min_cross_patients:
        return "cross_sectional", (
            "{} patients have a value, but only {} give an adjacent "
            "pair".format(s["patients_present"],
                          s["patients_adjacent"]))
    return "presence", (
        "only {} patients have a value; the fact of measurement "
        "survives, the value does not".format(s["patients_present"]))


def profile_column(col, rows_by_patient, n_rows, args):
    present_rows = 0
    pat_present = set()
    pat_counts = defaultdict(int)
    values = []
    loose_pairs = []
    adj_pairs = []
    pat_adjacent = set()
    numeric_ok = 0
    is_listish = False
    n_adj_present = [0]

    for pid, rows in rows_by_patient.items():
        # Adjacency is about PRESENCE, so it must be tracked
        # separately from numeric parseability - otherwise every
        # categorical column reads as having no pairs at all and can
        # never be modelled over time.
        present_seq = []
        numeric_seq = []
        for r in rows:
            raw = r.get(col, "")
            s = str(raw).strip() if raw is not None else ""
            if not s or s.lower() in ("nan", "none", "null"):
                present_seq.append(False)
                numeric_seq.append(None)
                continue
            present_rows += 1
            pat_present.add(pid)
            pat_counts[pid] += 1
            if LIST_SEP in s:
                is_listish = True
            if len(values) < args.max_sample:
                values.append(s)
            x = num(s)
            if x is not None:
                numeric_ok += 1
            present_seq.append(True)
            numeric_seq.append(x)
        got = [x for x in numeric_seq if x is not None]
        loose_pairs += list(zip(got, got[1:]))
        made = False
        for i in range(len(present_seq) - 1):
            if present_seq[i] and present_seq[i + 1]:
                made = True
                if (numeric_seq[i] is not None
                        and numeric_seq[i + 1] is not None):
                    adj_pairs.append((numeric_seq[i],
                                      numeric_seq[i + 1]))
                n_adj_present[0] += 1
        if made:
            pat_adjacent.add(pid)

    n_pat = len(rows_by_patient)
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    top_share = (max(counts.values()) / float(len(values))
                 if values else None)
    s = {
        "column": col,
        "visit_coverage": round(present_rows / float(n_rows), 4)
        if n_rows else 0.0,
        "patient_coverage": round(len(pat_present) / float(n_pat), 4)
        if n_pat else 0.0,
        "rows_present": present_rows,
        "patients_present": len(pat_present),
        "patients_2plus": sum(1 for c in pat_counts.values() if c >= 2),
        "patients_adjacent": len(pat_adjacent),
        "adjacent_pairs": n_adj_present[0],
        "adjacent_numeric_pairs": len(adj_pairs),
        "loose_pairs": len(loose_pairs),
        "distinct": len(counts),
        "top_share": round(top_share, 4) if top_share is not None
        else None,
        "numeric": numeric_ok > 0 and numeric_ok == present_rows,
        "list_valued": is_listish,
    }
    if s["numeric"] and len(adj_pairs) >= 30 and len(loose_pairs) >= 30:
        a = corr(adj_pairs)
        lo = corr(loose_pairs)
        s["autocorr_adjacent"] = round(a, 4) if a is not None else None
        s["autocorr_as_measured"] = round(lo, 4) if lo is not None \
            else None
        if a is not None and lo is not None:
            s["lag_error"] = round(lo - a, 4)
    tier, why = classify(s, args)
    s["tier"] = tier
    s["reason"] = why
    return s


def profile_model(path):
    """Size of a saved condnet by top-level key. Sizes and counts
    only - no table contents."""
    raw = path.read_text(encoding="utf-8")
    total = len(raw)
    try:
        blob = json.loads(raw)
    except ValueError:
        return {"bytes": total, "error": "not parseable as JSON"}
    if not isinstance(blob, dict):
        return {"bytes": total, "error": "top level is not an object"}
    out = {"bytes": total, "keys": []}
    for k, v in blob.items():
        n = len(json.dumps(v))
        entry = {"key": k, "bytes": n,
                 "share": round(n / float(total), 4)}
        if isinstance(v, dict):
            entry["entries"] = len(v)
            worst = None
            cells = 0
            for ck, cv in v.items():
                c = 0
                if isinstance(cv, dict):
                    for _kk, vv in cv.items():
                        c += len(vv) if isinstance(vv, dict) else 1
                cells += c
                if worst is None or c > worst[1]:
                    worst = (ck, c)
            if cells:
                entry["cells"] = cells
                entry["heaviest_column"] = worst[0]
                entry["heaviest_cells"] = worst[1]
        out["keys"].append(entry)
    out["keys"].sort(key=lambda d: -d["bytes"])
    return out


def main():
    ap = argparse.ArgumentParser(
        description="What can each column support? Statistics only.")
    ap.add_argument("--in", dest="inp", required=True,
                    help="tidy one-row-per-visit CSV")
    ap.add_argument("--model", help="condnet_model.json to size up")
    ap.add_argument("-o", "--out", default="support.json")
    ap.add_argument("--group", default="person_id")
    ap.add_argument("--order-by", default="visit_start_date",
                    help="column defining visit order within a patient")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-pair-patients", type=int, default=50,
                    help="patients contributing an adjacent pair "
                         "before a column may be modelled over time")
    ap.add_argument("--min-cross-patients", type=int, default=50,
                    help="patients with a value before a column may "
                         "be modelled cross-sectionally")
    ap.add_argument("--degenerate", type=float, default=0.99,
                    help="one value holding this share is degenerate")
    ap.add_argument("--max-sample", type=int, default=200000,
                    help="cap on values held per column for distinct "
                         "counting")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    inp = Path(a.inp)
    if not inp.exists():
        sys.exit("input not found: {}".format(inp))
    with inp.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if a.group not in header:
            sys.exit("--group column {!r} not in {}: found {}".format(
                a.group, inp.name, ", ".join(header[:8])))
        rows = list(reader)
    if not rows:
        sys.exit("{} has a header but no rows".format(inp.name))

    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r.get(a.group, "")].append(r)
    if a.order_by in header:
        for v in by_patient.values():
            v.sort(key=lambda r: str(r.get(a.order_by, "")))
        ordered = True
    else:
        ordered = False

    cols = [c for c in header if c not in ID_COLS and c != a.group]
    stats = [profile_column(c, by_patient, len(rows), a) for c in cols]

    out = {
        "generated_by": "scripts/support_profile.py",
        "input": str(inp.resolve()),
        "contract": "statistics only - no column value is emitted",
        "rows": len(rows),
        "patients": len(by_patient),
        "visit_order_applied": ordered,
        "settings": {"k": a.k,
                     "min_pair_patients": a.min_pair_patients,
                     "min_cross_patients": a.min_cross_patients,
                     "degenerate": a.degenerate},
        "columns": stats,
        "tier_counts": dict((t, sum(1 for s in stats if s["tier"] == t))
                            for t in TIERS),
    }
    lag = [s for s in stats if s.get("lag_error") is not None
           and abs(s["lag_error"]) > 0.05]
    out["variable_lag_affected"] = [
        {"column": s["column"], "as_measured": s["autocorr_as_measured"],
         "adjacent": s["autocorr_adjacent"], "error": s["lag_error"],
         "visit_coverage": s["visit_coverage"]} for s in lag]
    if a.model:
        mp = Path(a.model)
        if not mp.exists():
            sys.exit("--model not found: {}".format(mp))
        out["model"] = profile_model(mp)

    Path(a.out).write_text(json.dumps(out, indent=1),
                           encoding="utf-8")
    report(out, a.report)


def report(o, full):
    print("=" * 70)
    print("COLUMN SUPPORT PROFILE - statistics only")
    print("{:,} visits | {:,} patients | {} columns{}".format(
        o["rows"], o["patients"], len(o["columns"]),
        "" if o["visit_order_applied"]
        else "  [NO visit-order column: adjacency is file order]"))
    print("=" * 70)
    tc = o["tier_counts"]
    print("\nwhat the data can support")
    for t in TIERS:
        print("  {:<18} {:>4} columns".format(t, tc.get(t, 0)))
    print("\n{:<28} {:>6} {:>6} {:>7} {:>7} {:>8} {:>16}".format(
        "column", "vis%", "pat%", "pat>=2", "adjpr", "distinct",
        "tier"))
    for s in sorted(o["columns"],
                    key=lambda s: (TIERS.index(s["tier"]),
                                   -s["visit_coverage"])):
        if not full and s["tier"] == "drop":
            continue
        print("{:<28} {:>5.0%} {:>6.0%} {:>7} {:>7} {:>8} {:>16}".format(
            s["column"][:28], s["visit_coverage"],
            s["patient_coverage"], s["patients_2plus"],
            s["adjacent_pairs"], s["distinct"], s["tier"]))
    if not full:
        print("  ({} dropped columns hidden; --report shows them)"
              .format(tc.get("drop", 0)))

    vl = o["variable_lag_affected"]
    print("\nvariable-lag measurement error (|as-measured - adjacent| "
          "> 0.05): {} columns".format(len(vl)))
    if vl:
        print("{:<28} {:>12} {:>10} {:>8} {:>7}".format(
            "column", "as-measured", "adjacent", "error", "vis%"))
        for s in sorted(vl, key=lambda s: abs(s["error"]),
                        reverse=True)[:15]:
            print("{:<28} {:>12.3f} {:>10.3f} {:>+8.3f} {:>6.0%}".format(
                s["column"][:28], s["as_measured"], s["adjacent"],
                s["error"], s["visit_coverage"]))
        print("  a NEGATIVE error means the steadiness target the "
              "model calibrates against is UNDERSTATED")

    m = o.get("model")
    if m:
        print("\nmodel artifact: {:,} bytes".format(m["bytes"]))
        if m.get("error"):
            print("  {}".format(m["error"]))
        for e in m.get("keys", [])[:8]:
            extra = ""
            if e.get("cells"):
                extra = ("  {:,} cells, heaviest column {} with {:,}"
                         .format(e["cells"], e["heaviest_column"],
                                 e["heaviest_cells"]))
            print("  {:<22} {:>14,} B {:>6.1%}{}".format(
                e["key"], e["bytes"], e["share"], extra))


if __name__ == "__main__":
    main()
