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
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:                       # console-safe on Windows terminals
    from _console import console_safe
    console_safe()
except Exception:
    pass

csv.field_size_limit(1 << 22)

# Nothing is excluded by name. An earlier version skipped visit_id and
# the two date columns as "identifiers", and condnet was modelling all
# three - which is precisely where a 424 MB fault hid for as long as it
# did. The profiler must describe everything the model might touch and
# let the tier rules say what is unusable; a hand-kept skip list fails
# open exactly as NEVER_PARENT does.
ID_COLS = set()
LIST_SEP = "; "

# Elapsed-day buckets for the decay curve. If autocorrelation falls
# geometrically across these, persistence can be modelled with one
# decay constant per column; if it plateaus or steps, the transition
# table has to be keyed on the gap instead, which costs far more.
LAG_BUCKETS = [(0, 7), (8, 30), (31, 90), (91, 180), (181, 365),
               (366, 10 ** 6)]

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


def classify(s, args, n_rows):
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
    # The same bound condnet applies, so the tiers PREDICT what the
    # model will actually do rather than describing something else.
    # Without it an identifier passes every rule: visit_id has full
    # coverage and an adjacent pair for every visit, and tiered as
    # longitudinal.
    lv = s.get("levels_above_k")
    if lv is not None:
        # Too unique to model is the mirror of too many levels, and a
        # "too many levels" bound cannot see it: visit_id is distinct
        # on every row, so NO value clears k and its level count is
        # zero. condnet collapses such a column to a single fallback
        # level, which carries nothing. Whether anything survives
        # depends on presence: a column measured everywhere says
        # nothing by being present, a sparse one still does.
        if lv == 0:
            if s["visit_coverage"] >= 0.99:
                return "drop", ("no value is held by k patients and it "
                                "is present on every row - an "
                                "identifier, not a variable")
            return "presence", ("no value is held by k patients; only "
                                "the fact of measurement survives")
        budget = max(20, int((n_rows / max(args.k, 1)) ** 0.5))
        if lv > budget:
            return "drop", ("{} levels above k exceeds the {} a "
                            "transition table can support".format(
                                lv, budget))
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
    return s


def parse_day(v):
    """A real day number, or None.

    Uses calendar arithmetic rather than y*372+m*31+d. That shortcut
    is wrong by up to three days at every month boundary and drifts by
    about a month across a year, which lands pairs in the wrong
    elapsed-time bucket - and the whole point of the bucket curve is
    to decide how elapsed time should enter the model."""
    s = str(v or "").strip()[:10]
    if len(s) != 10:
        return None
    for sep, order in (("-", (0, 1, 2)), ("/", (2, 0, 1))):
        parts = s.split(sep)
        if len(parts) == 3:
            try:
                y = int(parts[order[0]])
                m = int(parts[order[1]])
                d = int(parts[order[2]])
                return date(y, m, d).toordinal()
            except ValueError:
                return None      # covers Feb 30 and friends
    return None


def lag_curve(rows_by_patient, cols, date_col, args):
    """Autocorrelation by ELAPSED-TIME bucket, pooled over columns.

    This is the measurement that decides how elapsed time enters the
    model. Row adjacency is not time: two consecutive visits can be a
    week or a year apart, and the transition tables currently treat
    those identically."""
    buckets = dict((b, []) for b in LAG_BUCKETS)
    per_col = {}
    for col in cols:
        cb = dict((b, []) for b in LAG_BUCKETS)
        for _pid, rows in rows_by_patient.items():
            seq = []
            for r in rows:
                d = parse_day(r.get(date_col, ""))
                x = num(r.get(col, ""))
                if d is not None:
                    seq.append((d, x))
            # Sort on the DAY only. A bare tuple sort falls through to
            # the second element whenever two visits share a date, and
            # that element is None for a missing observation - so a
            # patient seen twice in one day crashes it. Common in a
            # real extract, absent from mimic.
            seq.sort(key=lambda t: t[0])
            for i in range(len(seq)):
                if seq[i][1] is None:
                    continue
                for j in range(i + 1, len(seq)):
                    if seq[j][1] is None:
                        continue
                    gap = seq[j][0] - seq[i][0]
                    for b in LAG_BUCKETS:
                        if b[0] <= gap <= b[1]:
                            cb[b].append((seq[i][1], seq[j][1]))
                            break
                    break          # nearest later observation only
        got = {}
        for b in LAG_BUCKETS:
            if len(cb[b]) >= 30:
                c = corr(cb[b])
                if c is not None:
                    got[b] = (round(c, 4), len(cb[b]))
                    buckets[b] += cb[b]
        if len(got) >= 2:
            per_col[col] = dict(
                ("{}-{}".format(b[0], b[1]), v) for b, v in got.items())
    pooled = []
    for b in LAG_BUCKETS:
        if len(buckets[b]) >= 30:
            c = corr(buckets[b])
            if c is not None:
                pooled.append({"bucket": "{}-{}".format(b[0], b[1]),
                               "autocorr": round(c, 4),
                               "pairs": len(buckets[b])})
    # Pooling raw correlations across columns is misleading: a
    # patient-level constant sits at 1.0 in every bucket and flattens
    # the average, hiding whatever decay the varying columns have. So
    # normalise each column against its own shortest bucket and report
    # the median RATIO - that is decay shape, independent of level.
    ratios = defaultdict(list)
    for _c, got in per_col.items():
        keys = [b for b in LAG_BUCKETS
                if "{}-{}".format(b[0], b[1]) in got]
        if len(keys) < 2:
            continue
        base = got["{}-{}".format(keys[0][0], keys[0][1])][0]
        if abs(base) < 0.1:
            continue           # nothing to decay from
        for b in keys:
            lbl = "{}-{}".format(b[0], b[1])
            ratios[lbl].append(got[lbl][0] / base)
    decay = []
    for b in LAG_BUCKETS:
        lbl = "{}-{}".format(b[0], b[1])
        if ratios.get(lbl):
            xs = sorted(ratios[lbl])
            decay.append({"bucket": lbl,
                          "median_ratio": round(xs[len(xs) // 2], 4),
                          "columns": len(xs)})
    return {"pooled": pooled, "decay": decay, "by_column": per_col}


def would_be_levels(rows_by_patient, col, k):
    """Levels condnet would keep for this column if it treated it as a
    categorical: distinct values held by at least k patients. The
    transition table it implies is that count squared."""
    holders = defaultdict(set)
    numericish = 0
    present = 0
    for pid, rows in rows_by_patient.items():
        for r in rows:
            s = str(r.get(col, "") or "").strip()
            if not s or s.lower() in ("nan", "none", "null"):
                continue
            present += 1
            holders[s].add(pid)
            if num(s) is not None:
                numericish += 1
    if present and numericish == present:
        return None            # numeric: binned, not levelled
    return sum(1 for v in holders.values() if len(v) >= k)


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
    ap.add_argument("--date-col", default="visit_start_date",
                    help="date column used to measure ELAPSED time "
                         "between visits, not just their order")
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
    for s in stats:
        lv = would_be_levels(by_patient, s["column"], a.k)
        if lv is not None:
            s["levels_above_k"] = lv
            s["transition_cells"] = lv * lv
        # Tiering happens AFTER the level count exists, so the
        # cardinality rule can see it.
        tier, why = classify(s, a, len(rows))
        s["tier"] = tier
        s["reason"] = why

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
    numeric_cols = [s["column"] for s in stats if s["numeric"]]
    if a.date_col in header:
        out["lag_curve"] = lag_curve(by_patient, numeric_cols,
                                     a.date_col, a)
    else:
        out["lag_curve"] = {"error": "no {} column; elapsed time "
                                     "cannot be measured".format(
                                         a.date_col)}
    adj = sorted(s["adjacent_pairs"] for s in stats)
    if adj:
        out["adjacent_pair_distribution"] = {
            "min": adj[0], "p25": adj[len(adj) // 4],
            "p50": adj[len(adj) // 2],
            "p75": adj[(3 * len(adj)) // 4], "max": adj[-1],
            "possible": len(rows) - len(by_patient)}
    if a.model:
        mp = Path(a.model)
        if not mp.exists():
            sys.exit("--model not found: {}".format(mp))
        out["model"] = profile_model(mp)
        # The coverage gap. visit_start_date and visit_end_date hid
        # here: described by nobody, modelled by condnet. Diff both
        # ways rather than assume the gap is closed.
        modelled = set()
        try:
            blob = json.loads(mp.read_text(encoding="utf-8"))
            for key in ("binnings", "marginal", "level"):
                v = blob.get(key)
                if isinstance(v, dict):
                    modelled |= set(v)
        except ValueError:
            pass
        if modelled:
            described = set(s["column"] for s in stats)
            out["coverage_gap"] = {
                "modelled_not_described": sorted(modelled - described),
                "described_not_modelled": sorted(described - modelled),
                "both": len(modelled & described),
            }

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

    lc = o.get("lag_curve", {})
    if lc.get("pooled"):
        print("\nautocorrelation by ELAPSED DAYS (pooled over "
              "{} numeric columns)".format(len(lc.get("by_column", {}))))
        print("  {:<12} {:>10} {:>12}".format("days apart", "autocorr",
                                              "pairs"))
        for b_ in lc["pooled"]:
            print("  {:<12} {:>10.3f} {:>12,}".format(
                b_["bucket"], b_["autocorr"], b_["pairs"]))
        if lc.get("decay"):
            print("  decay shape, each column normalised against its "
                  "own shortest bucket (median ratio):")
            for d in lc["decay"]:
                print("    {:<12} {:>8.3f}  from {} columns".format(
                    d["bucket"], d["median_ratio"], d["columns"]))
        print("  falling geometrically -> one decay constant per "
              "column is enough")
        print("  plateauing or stepping -> the transition table has "
              "to be keyed on the gap")
    elif lc.get("error"):
        print("\n" + lc["error"])

    ap_ = o.get("adjacent_pair_distribution")
    if ap_:
        print("\nadjacent pairs per column: min {:,} p25 {:,} p50 {:,} "
              "p75 {:,} max {:,}  (of {:,} possible)".format(
                  ap_["min"], ap_["p25"], ap_["p50"], ap_["p75"],
                  ap_["max"], ap_["possible"]))

    heavy = [s for s in o["columns"] if s.get("transition_cells")]
    heavy.sort(key=lambda s: -s["transition_cells"])
    if heavy and heavy[0]["transition_cells"] > 100:
        print("\ncategoricals by transition cost (levels above k, "
              "squared)")
        for s in heavy[:8]:
            print("  {:<28} {:>6} levels {:>12,} cells".format(
                s["column"][:28], s["levels_above_k"],
                s["transition_cells"]))

    cg = o.get("coverage_gap")
    if cg:
        print("\ncoverage gap against the model")
        print("  described AND modelled     {}".format(cg["both"]))
        print("  MODELLED, NOT DESCRIBED    {}{}".format(
            len(cg["modelled_not_described"]),
            "  <-- blind spot" if cg["modelled_not_described"] else ""))
        for c in cg["modelled_not_described"][:12]:
            print("      {}".format(c))
        print("  described, not modelled    {}".format(
            len(cg["described_not_modelled"])))

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
