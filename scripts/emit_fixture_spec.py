"""Emit the SHAPE of a real extract, so a faithful fixture can be
built on a machine that will never see it.

    python scripts/emit_fixture_spec.py --in tidy_visits_labeled.csv
                                        [--model condnet_model.json]
                                        [-o spec.txt] [--width 64]

STATISTICS ONLY, and stricter than the profiler: no category level
names, no free text, no dates, no values of any kind - counts, rates,
correlations and shapes. A fixture does not need to know that a level
is called OUTPATIENT, only that the column has nine levels and the
largest holds 85% of it. Column NAMES are emitted, because a fixture
with different column names cannot reproduce a schema-shaped fault.

Why this exists: every defect that reached the work machine hid
because the fixture here could not contain it - dates that only clear
k above a few hundred patients, an integer identifier where the
fixture used a string, missingness that clusters where the fixture
made it independent. A fixture built from what was MEASURED cannot
have that class of blind spot. A fixture built from imagination
always can.

Output is one fact per line, hard-wrapped, nothing aligned - five
consecutive pastes of tabular output arrived with columns shifted.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:                       # console-safe on Windows terminals
    from _console import console_safe
    console_safe()
except Exception:
    pass

csv.field_size_limit(1 << 22)
LIST_SEP = "; "
DATE_HINT = ("date", "datetime")


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def blank(v):
    s = str(v or "").strip()
    return (not s) or s.lower() in ("nan", "none", "null")


def icc(groups):
    """Between-group share of variance, ANOVA form. Biased upward if
    computed as 1 - SSW/SST, which is why this uses mean squares."""
    allv = [x for g in groups for x in g]
    k, n = len(groups), len(allv)
    if k < 2 or n <= k:
        return None
    grand = sum(allv) / n
    ssb = ssw = 0.0
    for g in groups:
        m = sum(g) / len(g)
        ssb += len(g) * (m - grand) ** 2
        ssw += sum((x - m) ** 2 for x in g)
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - sum(len(g) ** 2 for g in groups) / float(n)) / (k - 1)
    den = msb + (n0 - 1) * msw
    if den <= 0:
        return None
    return max(0.0, min(1.0, (msb - msw) / den))


def lag1(groups):
    pr = []
    for g in groups:
        pr += list(zip(g, g[1:]))
    if len(pr) < 30:
        return None
    a = [x for x, _ in pr]
    b = [y for _, y in pr]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    nu = sum((x - ma) * (y - mb) for x, y in pr)
    de = (sum((x - ma) ** 2 for x in a)
          * sum((y - mb) ** 2 for y in b)) ** 0.5
    return nu / de if de else None


def pack(bits, width, indent="  ", cont="    "):
    """Lay key-value groups onto lines without ever splitting one."""
    lines, cur, pre = [], [], indent
    for bit in bits:
        trial = pre + " | ".join(cur + [bit])
        if cur and len(trial) > width:
            lines.append(pre + " | ".join(cur))
            cur, pre = [bit], cont
        else:
            cur.append(bit)
    if cur:
        lines.append(pre + " | ".join(cur))
    return lines


def main():
    ap = argparse.ArgumentParser(
        description="Emit an extract's shape, statistics only.")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--model", help="condnet_model.json, for the "
                                    "relationship shapes")
    ap.add_argument("-o", "--out", default="")
    ap.add_argument("--group", default="person_id")
    ap.add_argument("--k", type=int, default=10)
    # Wider than run_summary is safe here because every number
    # carries its own key: "cov 0.44", not a column in a table. A
    # wrapped line cannot attach a value to the wrong field, which is
    # the failure that corrupted five earlier pastes.
    ap.add_argument("--width", type=int, default=78)
    a = ap.parse_args()

    inp = Path(a.inp)
    if not inp.exists():
        sys.exit("input not found: {}".format(inp))
    with inp.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if a.group not in header:
            sys.exit("--group column {!r} not found; header starts "
                     "{}".format(a.group, ", ".join(header[:6])))
        rows = list(reader)
    if not rows:
        sys.exit("{} has a header but no rows".format(inp.name))

    by = defaultdict(list)
    for r in rows:
        by[r.get(a.group, "")].append(r)
    n_rows, n_pat = len(rows), len(by)

    out = ["SPEC {}".format(inp.name),
           "rows {}".format(n_rows),
           "patients {}".format(n_pat)]
    vc = sorted(len(v) for v in by.values())
    out.append("visits per patient mean {:.1f}".format(
        n_rows / float(n_pat)))
    for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99):
        out.append("visits p{:.0f} {}".format(
            q * 100, vc[min(len(vc) - 1, int(q * len(vc)))]))
    out.append("visits max {}".format(vc[-1]))
    out.append("columns {}".format(len(header)))
    out.append("--- one line per column ---")

    for col in header:
        if col == a.group:
            continue
        present = [r for r in rows if not blank(r.get(col))]
        cov = len(present) / float(n_rows)
        pats = len(set(r.get(a.group) for r in present))
        vals = [str(r.get(col)).strip() for r in present]
        nums = [num(v) for v in vals]
        numeric = bool(vals) and all(x is not None for x in nums)
        listish = any(LIST_SEP in v for v in vals[:2000])
        distinct = len(set(vals))
        holders = defaultdict(set)
        for r in present:
            holders[str(r.get(col)).strip()].add(r.get(a.group))
        above_k = sum(1 for h in holders.values() if len(h) >= a.k)

        # how much missingness CLUSTERS: share of adjacent visit pairs
        # whose present/absent state matches. 0.5 is independent.
        same = tot = 0
        for g in by.values():
            st = [not blank(r.get(col)) for r in g]
            for i in range(len(st) - 1):
                tot += 1
                same += 1 if st[i] == st[i + 1] else 0
        clus = (same / float(tot)) if tot else None

        kind = ("list" if listish else
                "numeric" if numeric else "categorical")
        if any(h in col for h in DATE_HINT) and not numeric:
            kind = "date-like"
        # An integer unique on essentially every row is a key. A
        # fixture needs to know THAT, not its range - and percentiles
        # of an identifier are real values with nothing to learn from.
        is_key = (numeric and vals
                  and distinct >= 0.999 * len(vals)
                  and cov >= 0.99
                  and all(float(x).is_integer() for x in nums[:5000]))
        if is_key:
            kind = "identifier"
        bits = ["{} kind {}".format(col, kind),
                "cov {:.3f}".format(cov),
                "patcov {:.3f}".format(pats / float(n_pat)),
                "distinct {}".format(distinct),
                "levels_above_k {}".format(above_k)]
        if clus is not None:
            # EXCESS over what independence alone would give. A column
            # that is 98% missing matches on 96% of adjacent pairs by
            # chance, so the raw match rate says "heavily clustered"
            # about a column that is not clustered at all. 0 is
            # independent, 1 is perfectly run-length clustered.
            ind = cov * cov + (1.0 - cov) * (1.0 - cov)
            exc = ((clus - ind) / (1.0 - ind)) if ind < 1.0 else 0.0
            bits.append("clusterx {:.3f}".format(max(0.0, exc)))
            bits.append("matchrate {:.3f}".format(clus))
        if numeric:
            groups = []
            for g in by.values():
                xs = [num(r.get(col)) for r in g]
                xs = [x for x in xs if x is not None]
                if xs:
                    groups.append(xs)
            i = icc(groups)
            l1 = lag1(groups)
            if i is not None:
                bits.append("icc {:.3f}".format(i))
            if l1 is not None:
                bits.append("lag1 {:.3f}".format(l1))
            ok = sorted(x for x in nums if x is not None)
            if len(ok) >= 20 and not is_key:
                # spread only, never min/max: p1..p99 and a log flag
                p1 = ok[int(0.01 * len(ok))]
                p50 = ok[len(ok) // 2]
                p99 = ok[min(len(ok) - 1, int(0.99 * len(ok)))]
                rng = p99 - p1
                bits.append("p1p50p99 {:.4g} {:.4g} {:.4g}".format(
                    p1, p50, p99))
                bits.append("integral {}".format(
                    all(float(x).is_integer() for x in ok[:5000])))
                if p1 > 0 and rng > 0:
                    sk = (p99 - p50) / max(p50 - p1, 1e-9)
                    bits.append("skew {:.2f}".format(sk))
        if listish:
            lens = [len([i for i in v.split(LIST_SEP) if i.strip()])
                    for v in vals[:20000]]
            items = Counter()
            for v in vals[:20000]:
                for it in v.split(LIST_SEP):
                    if it.strip():
                        items[it.strip()] += 1
            bits.append("items_mean {:.2f}".format(
                sum(lens) / float(len(lens)) if lens else 0))
            bits.append("distinct_items {}".format(len(items)))
            bits.append("items_above_k {}".format(
                sum(1 for _i, c in items.items() if c >= a.k)))
        # Break only BETWEEN key-value groups, never inside one.
        # textwrap breaks on any space, which can split "p1p50p99 8
        # 44 83" across lines and separate a value from its key -
        # exactly the misattribution this format exists to prevent.
        out.extend(pack(bits, a.width))

    if a.model:
        mp = Path(a.model)
        if not mp.exists():
            sys.exit("--model not found: {}".format(mp))
        rep = (json.loads(mp.read_text(encoding="utf-8"))
               .get("report") or {})
        edges = rep.get("edges") or []
        out.append("--- relationship shapes ---")
        out.append("edges {}".format(len(edges)))
        widths = Counter(len(e.get("parents") or []) for e in edges)
        for w in sorted(widths):
            out.append("edges with {} parent(s) {}".format(
                w, widths[w]))
        out.append("bins {}".format(rep.get("bins")))
        out.append("corrected over {}".format(
            rep.get("comparisons_corrected_for")))
        for e in edges:
            out.append("  edge {} <- {}".format(
                e.get("child"), ", ".join(e.get("parents") or [])))

    text = "\n".join(
        line if line.startswith(("  ", "    ")) else
        "\n".join(textwrap.wrap(
            line, width=a.width, subsequent_indent="    ",
            break_long_words=True, break_on_hyphens=False) or [""])
        for line in out)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
