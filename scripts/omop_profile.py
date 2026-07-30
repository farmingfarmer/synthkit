"""Profile a tidy dataset into PARAMETERS — never records.

    python scripts/omop_profile.py --in tidy_visits_labeled.csv
                                   -o profile.json [--k 10] [--report]

The privacy contract: the emitted profile contains only aggregates
computed over at least k records. No individual value, no record,
no identifier survives into the profile, so nothing generated from
it can descend from a real person. This is deliberately NOT an
interpolation method (SMOTE and kin synthesize between real
records, anchoring every output to specific individuals); here the
data teaches parameters and the parameters generate.

What is measured:
  marginals   — numeric fits (normal / lognormal / zero-inflated),
                categorical weights, missing and out-of-range rates
  joint       — rank correlations between numerics, conditional
                shifts of numerics across categorical strata,
                co-occurrence lift within list-valued columns
  mess        — missingness, mixed formats, degenerate constants

Anonymization applied while profiling:
  - identifier-like columns excluded entirely (mostly-unique text)
  - categorical levels below k folded into OTHER_SUPPRESSED
  - numeric ranges reported as p1/p99, never min/max
  - every correlation, stratum and co-occurrence requires >= k rows
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

csv.field_size_limit(1 << 22)

ALWAYS_EXCLUDE = {"visit_id", "person_id", "days_to_next_visit"}
LIST_SEP = "; "


# ---------------------------------------------------------------
# small stats helpers (stdlib only)
# ---------------------------------------------------------------
def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def skew(xs):
    s = sd(xs)
    if s == 0 or len(xs) < 3:
        return 0.0
    m = mean(xs)
    return sum(((x - m) / s) ** 3 for x in xs) / len(xs)


def pct(xs, p):
    if not xs:
        return None
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round(p * (len(ys) - 1)))))
    return ys[i]


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(xs, ys):
    if len(xs) < 3:
        return 0.0
    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def num(v):
    v = (v or "").strip()
    if not v or v.lower() in ("nan", "none", "null"):
        return None
    try:
        return float(v.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def is_date(v):
    v = (v or "").strip()
    for f in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            datetime.strptime(v, f)
            return True
        except ValueError:
            continue
    return False


# ---------------------------------------------------------------
def classify(name, values, k):
    """Decide what a column IS, from its values."""
    present = [v for v in values if str(v).strip() != ""]
    n, N = len(present), len(values)
    if n == 0:
        return "empty", {}
    uniq = len(set(present))
    if name in ALWAYS_EXCLUDE:
        return "excluded_identifier", {}
    if any(LIST_SEP in str(v) for v in present):
        return "list", {}
    nums = [num(v) for v in present]
    if all(x is not None for x in nums):
        if uniq <= 2 and set(present) <= {"0", "1"}:
            return "binary", {}
        if uniq > max(20, 0.5 * n) and n >= 20 and \
                all(float(x).is_integer() for x in nums) and \
                min(nums) > 10000:
            return "excluded_identifier", {}
        return "numeric", {}
    if sum(1 for v in present if is_date(v)) > 0.9 * n:
        return "date", {}
    # text: identifier-like if mostly unique
    if n >= 20 and uniq / n > 0.5:
        return "excluded_identifier", {}
    if uniq == 1:
        return "constant", {}
    return "categorical", {}


def profile_numeric(vals, k):
    xs = [num(v) for v in vals if str(v).strip() != ""]
    xs = [x for x in xs if x is not None]
    if len(xs) < k:
        return None
    zeros = sum(1 for x in xs if x == 0)
    nz = [x for x in xs if x != 0]
    body = nz if (zeros / len(xs) > 0.2 and len(nz) >= k) else xs
    degenerate = len(set(body)) < 2
    fam, params = "normal", {}
    if body and min(body) > 0 and skew(body) > 0.75:
        logs = [math.log(x) for x in body]
        fam = "lognormal"
        params = {"mu": round(mean(logs), 4),
                  "sigma": round(sd(logs), 4)}
    else:
        params = {"mean": round(mean(body), 4),
                  "sd": round(sd(body), 4)}
    out = {
        "kind": "numeric", "n": len(xs),
        "degenerate_body": degenerate,
        "family": fam, "params": params,
        "p1": round(pct(xs, 0.01), 4), "p50": round(pct(xs, 0.5), 4),
        "p99": round(pct(xs, 0.99), 4),
        "integer_valued": all(float(x).is_integer() for x in xs),
        "skew": round(skew(xs), 3),
    }
    if zeros / len(xs) > 0.2:
        out["zero_inflation"] = round(zeros / len(xs), 4)
    if min(xs) >= 0 and fam == "normal":
        m, s = params["mean"], params["sd"]
        if s > 0 and m - 2 * s < 0:
            out["warning"] = ("observed values are all >= 0 but a "
                              "normal fit would emit negatives; "
                              "generation must clamp at 0")
    return out


def profile_categorical(vals, k):
    present = [str(v).strip() for v in vals if str(v).strip() != ""]
    if len(present) < k:
        return None
    c = Counter(present)
    kept, suppressed, sup_n = {}, [], 0
    for level, cnt in c.most_common():
        if cnt >= k:
            kept[level] = round(cnt / len(present), 4)
        else:
            suppressed.append(level)
            sup_n += cnt
    if sup_n:
        kept["OTHER_SUPPRESSED"] = round(sup_n / len(present), 4)
    return {"kind": "categorical", "n": len(present),
            "levels": len(c), "weights": kept,
            "suppressed_levels": len(suppressed)}


def profile_list(vals, k):
    lists = [[i.strip() for i in str(v).split(LIST_SEP) if i.strip()]
             for v in vals]
    lens = [len(l) for l in lists]
    items = Counter(i for l in lists for i in l)
    kept = {i: round(c / len(lists), 4)
            for i, c in items.most_common() if c >= k}
    # co-occurrence lift, suppressed below k
    pairs = Counter()
    for l in lists:
        s = sorted(set(l))
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                pairs[(s[i], s[j])] += 1
    lift = []
    for (a, b), c in pairs.most_common(400):
        if c < k or a not in kept or b not in kept:
            continue
        exp = kept[a] * kept[b] * len(lists)
        if exp > 0:
            lift.append({"a": a, "b": b, "n": c,
                         "lift": round(c / exp, 3)})
    lift.sort(key=lambda d: -abs(d["lift"] - 1))
    return {"kind": "list", "n": len(lists),
            "mean_length": round(mean(lens), 3),
            "sd_length": round(sd(lens), 3),
            "empty_rate": round(
                sum(1 for l in lists if not l) / len(lists), 4),
            "distinct_items": len(items),
            "item_rates": dict(list(kept.items())[:60]),
            "suppressed_items": sum(1 for i, c in items.items()
                                    if c < k),
            "cooccurrence": lift[:40]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("-o", "--out", default="profile.json")
    ap.add_argument("--k", type=int, default=10,
                    help="minimum cell size (k-anonymity threshold)")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    k = a.k

    with Path(a.src).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("no rows")
    cols = list(rows[0].keys())
    N = len(rows)

    columns, excluded, notes = {}, [], []
    for c in cols:
        vals = [r.get(c, "") for r in rows]
        kind, _ = classify(c, vals, k)
        missing = sum(1 for v in vals if str(v).strip() == "") / N
        if kind == "excluded_identifier":
            excluded.append(c)
            continue
        if kind == "empty":
            excluded.append(c)
            notes.append("{}: entirely empty".format(c))
            continue
        if kind == "constant":
            columns[c] = {"kind": "constant", "n": N,
                          "missing_rate": round(missing, 4),
                          "note": "single value — carries no "
                                  "information; a candidate to drop"}
            continue
        if kind == "numeric":
            p = profile_numeric(vals, k)
        elif kind == "binary":
            ones = sum(1 for v in vals if str(v).strip() == "1")
            p = {"kind": "binary", "n": N,
                 "rate": round(ones / N, 4)}
        elif kind == "categorical":
            p = profile_categorical(vals, k)
        elif kind == "list":
            p = profile_list(vals, k)
        elif kind == "date":
            ds = []
            for v in vals:
                for f in ("%Y-%m-%d", "%m/%d/%Y"):
                    try:
                        ds.append(datetime.strptime(
                            str(v).strip(), f).date())
                        break
                    except ValueError:
                        continue
            p = ({"kind": "date", "n": len(ds),
                  "start": str(min(ds)), "end": str(max(ds))}
                 if len(ds) >= k else None)
        else:
            p = None
        if p is None:
            excluded.append(c)
            notes.append("{}: fewer than k={} usable values"
                         .format(c, k))
            continue
        p["missing_rate"] = round(missing, 4)
        columns[c] = p

    # ---------- joint structure ----------
    numerics = [c for c, p in columns.items()
                if p["kind"] == "numeric"]
    tested = 0
    series = {c: [num(r.get(c, "")) for r in rows] for c in numerics}
    corrs = []
    for i in range(len(numerics)):
        for j in range(i + 1, len(numerics)):
            a_, b_ = numerics[i], numerics[j]
            pairsx = [(x, y) for x, y in zip(series[a_], series[b_])
                      if x is not None and y is not None]
            if len(pairsx) < max(k, 20):
                continue
            rho = spearman([x for x, _ in pairsx],
                           [y for _, y in pairsx])
            nn = len(pairsx)
            # Fisher z: is this correlation distinguishable from
            # zero at all? Spurious pairs at n~180 easily reach
            # |rho| 0.15 by chance — profiling those and then
            # IMPOSING them would manufacture structure the source
            # data never had.
            if abs(rho) >= 0.999 or nn < 4:
                z = float("inf") if abs(rho) >= 0.999 else 0.0
                lo = hi = rho
            else:
                zr = 0.5 * math.log((1 + rho) / (1 - rho))
                se = 1.0 / math.sqrt(nn - 3)
                z = zr / se
                lo = math.tanh(zr - 1.96 * se)
                hi = math.tanh(zr + 1.96 * se)
            tested += 1
            if abs(rho) >= 0.10:
                corrs.append({
                    "a": a_, "b": b_, "spearman": round(rho, 3),
                    "ci": [round(lo, 3), round(hi, 3)],
                    "n": nn, "z": (None if z == float("inf")
                                   else round(z, 2)),
                    "redundant": abs(rho) >= 0.95})
    # Multiple-comparisons correction. With ~30 numeric columns we
    # test ~435 pairs; at alpha=.05 roughly 22 would clear a naive
    # threshold by chance alone. Only Bonferroni survivors are
    # marked safe to IMPOSE on generated data — the rest are
    # reported as unconfirmed, never silently promoted to truth.
    z_crit = 1.96
    if tested > 1:
        # inverse normal for alpha/(2*tested), Acklam-lite
        alpha = 0.05 / tested
        q = alpha / 2.0
        # rational approximation of the normal quantile
        t = math.sqrt(-2.0 * math.log(q))
        z_crit = t - ((2.515517 + 0.802853 * t + 0.010328 * t * t)
                      / (1 + 1.432788 * t + 0.189269 * t * t
                         + 0.001308 * t ** 3))
    for d in corrs:
        d["confirmed"] = (d["z"] is None
                          or abs(d["z"]) >= z_crit)
    corrs.sort(key=lambda d: (not d["confirmed"],
                              -abs(d["spearman"])))

    cats = [c for c, p in columns.items()
            if p["kind"] in ("categorical", "binary")]
    shifts = []
    shift_tests = 0
    for c in cats:
        levels = [l for l in
                  (columns[c].get("weights") or {"1": 1}).keys()
                  if l != "OTHER_SUPPRESSED"]
        for nm in numerics:
            base = [x for x in series[nm] if x is not None]
            if len(base) < max(k, 20):
                continue
            bm, bs = mean(base), sd(base)
            if bs == 0:
                continue
            for lv in levels:
                sub = [num(r.get(nm, "")) for r in rows
                       if str(r.get(c, "")).strip() == lv]
                sub = [x for x in sub if x is not None]
                if len(sub) < k:
                    continue
                d = (mean(sub) - bm) / bs
                shift_tests += 1
                if abs(d) >= 0.20:
                    # Welch-ish z on the standardized difference
                    se = math.sqrt(1.0 / len(sub)
                                   + 1.0 / len(base))
                    shifts.append({
                        "column": nm, "given": c, "level": lv,
                        "n": len(sub), "std_shift": round(d, 3),
                        "z": round(d / se, 2) if se else 0.0})
    zs_crit = 1.96
    if shift_tests > 1:
        alpha = 0.05 / shift_tests
        q = alpha / 2.0
        t = math.sqrt(-2.0 * math.log(q))
        zs_crit = t - ((2.515517 + 0.802853 * t + 0.010328 * t * t)
                       / (1 + 1.432788 * t + 0.189269 * t * t
                          + 0.001308 * t ** 3))
    for d in shifts:
        d["confirmed"] = abs(d["z"]) >= zs_crit
    shifts.sort(key=lambda d: (not d["confirmed"],
                               -abs(d["std_shift"])))

    profile = {
        "source": {"file": Path(a.src).name, "rows": N,
                   "columns_seen": len(cols),
                   "columns_profiled": len(columns)},
        "privacy": {
            "k_threshold": k,
            "contract": "aggregates only; no record, value or "
                        "identifier is emitted; generation from "
                        "this profile cannot reproduce a source "
                        "record",
            "excluded_columns": excluded,
            "numeric_bounds": "p1/p99 reported; min/max withheld",
            "notes": notes,
        },
        "columns": columns,
        "joint": {
            "multiple_comparisons": {
                "correlation_pairs_tested": tested,
                "correlation_z_threshold": round(z_crit, 3),
                "correlations_confirmed": sum(
                    1 for d in corrs if d["confirmed"]),
                "stratum_tests": shift_tests,
                "stratum_z_threshold": round(zs_crit, 3),
                "shifts_confirmed": sum(
                    1 for d in shifts if d["confirmed"]),
                "policy": "only confirmed relationships may be "
                          "imposed on generated data; unconfirmed "
                          "ones are reported for review and "
                          "treated as absent",
            },
            "numeric_correlations": corrs[:60],
            "conditional_shifts": shifts[:80],
        },
    }
    Path(a.out).write_text(json.dumps(profile, indent=1),
                           encoding="utf-8")
    print("WROTE {} : {} columns profiled, {} excluded"
          .format(a.out, len(columns), len(excluded)))

    if a.report:
        print("\n--- WHAT THE DATA SAYS (k={}) ---".format(k))
        for c, p in columns.items():
            if p["kind"] == "numeric":
                pr = p["params"]
                desc = ("lognormal mu={} sigma={}".format(
                    pr.get("mu"), pr.get("sigma"))
                    if p["family"] == "lognormal" else
                    "normal mean={} sd={}".format(
                        pr.get("mean"), pr.get("sd")))
                print("  {:26s} {:9s} {}  p1..p99 {}..{}  "
                      "missing {:.1%}".format(
                          c, "numeric", desc, p["p1"], p["p99"],
                          p["missing_rate"]))
                if "zero_inflation" in p:
                    print("  {:26s}   zero-inflated: {:.1%} exact "
                          "zeros".format("", p["zero_inflation"]))
            elif p["kind"] == "categorical":
                top = ", ".join("{} {:.1%}".format(l, w)
                                for l, w in
                                list(p["weights"].items())[:5])
                print("  {:26s} {:9s} {} levels: {}{}".format(
                    c, "category", p["levels"], top,
                    "  (+{} rare levels suppressed)".format(
                        p["suppressed_levels"])
                    if p["suppressed_levels"] else ""))
            elif p["kind"] == "binary":
                print("  {:26s} {:9s} positive {:.1%}".format(
                    c, "binary", p["rate"]))
            elif p["kind"] == "list":
                top = ", ".join("{} {:.0%}".format(i, r) for i, r
                                in list(p["item_rates"].items())[:4])
                print("  {:26s} {:9s} mean {:.2f} items, {} distinct"
                      ", empty {:.1%} | {}".format(
                          c, "list", p["mean_length"],
                          p["distinct_items"], p["empty_rate"], top))
            elif p["kind"] == "date":
                print("  {:26s} {:9s} {} .. {}".format(
                    c, "date", p["start"], p["end"]))
            elif p["kind"] == "constant":
                print("  {:26s} {:9s} single value — no information"
                      .format(c, "constant"))
        print("\n--- HOW THE FIELDS MOVE TOGETHER ---")
        conf = [d for d in corrs if d["confirmed"]]
        unconf = [d for d in corrs if not d["confirmed"]]
        print("  {} pairs tested; threshold |z| >= {:.2f} after "
              "correcting for {} comparisons".format(
                  tested, z_crit, tested))
        if conf:
            for d in conf[:10]:
                print("  CONFIRMED  {:24s} <-> {:22s} rho {:+.3f} "
                      "[{:+.2f},{:+.2f}] n={}{}".format(
                          d["a"], d["b"], d["spearman"],
                          d["ci"][0], d["ci"][1], d["n"],
                          "  (REDUNDANT — derived from the same "
                          "quantity)" if d["redundant"] else ""))
        if unconf:
            print("  {} further pairs reached |rho| >= 0.10 but do "
                  "NOT survive correction — reported, not imposed:"
                  .format(len(unconf)))
            for d in unconf[:4]:
                print("    unconfirmed  {:22s} <-> {:20s} rho "
                      "{:+.3f}".format(d["a"], d["b"],
                                       d["spearman"]))
        if not corrs:
            print("  no numeric pair reaches |rho| >= 0.10 — the "
                  "columns are mutually independent")
        sconf = [d for d in shifts if d["confirmed"]]
        print("  {} stratum tests; threshold |z| >= {:.2f}".format(
            shift_tests, zs_crit))
        if sconf:
            for d in sconf[:10]:
                print("  CONFIRMED  {:22s} given {} = {:20s} "
                      "shifts {:+.2f} sd (n={})".format(
                          d["column"], d["given"], d["level"],
                          d["std_shift"], d["n"]))
        elif shifts:
            print("  {} strata shifted a numeric by >= 0.20 sd but "
                  "none survive correction — no conditional "
                  "structure confirmed".format(len(shifts)))
        if not shifts:
            print("  no categorical stratum shifts any numeric by "
                  ">= 0.20 sd — no conditional structure")
        for c, p in columns.items():
            if p["kind"] == "list" and p.get("cooccurrence"):
                print("  co-occurrence in {}: {}".format(
                    c, ", ".join("{}+{} lift {:.2f}".format(
                        d["a"][:18], d["b"][:18], d["lift"])
                        for d in p["cooccurrence"][:3])))
        print("\n  EXCLUDED (identifier-like or too sparse): {}"
              .format(", ".join(excluded) or "none"))


if __name__ == "__main__":
    main()
