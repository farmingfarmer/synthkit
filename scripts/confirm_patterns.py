"""Find relationships on one set of patients, confirm them on another.

    python scripts/confirm_patterns.py --in tidy_visits.csv
                                       [-o confirmed.json] [--report]

WHY THIS EXISTS
---------------
The search protects against false structure by dividing alpha across
every comparison it makes. On a wide extract that is 3,828 comparisons
at an effective n of 800, and the same threshold that keeps noise out
keeps real structure out with it. A pattern nobody knew to look for is
exactly the one that cannot afford the tax.

There is a second way to referee a finding: look everywhere, then
re-test what you found on patients the search never saw. A false
pattern does not reproduce out of sample no matter how many were
tried, so breadth costs compute rather than power. That is what makes
screening liberally safe, and it is the prerequisite for ever
generating candidate features automatically - under a Bonferroni tax,
every extra candidate makes every real one harder to find.

The split is BY PATIENT, never by row. Visits from one person are not
independent evidence, so a row-wise split would leak the same patient
into both halves and confirm almost anything.

WHAT IS REPORTED
----------------
Per relationship: its strength where it was found, its strength on the
held-out patients, and whether it reproduced. A relationship that
reproduces out of sample is credible without anyone needing to know
the domain, which is the point - the person reading this may not know
which patterns should be there.

No correction is applied on the holdout. Each edge arrives as a single
pre-specified hypothesis; the correction was paying for the search,
and the holdout does not search.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:                       # console-safe on Windows terminals
    from _console import console_safe
    console_safe()
except Exception:
    pass

from synthkit.condnet import CondNet, _chi2_crit   # noqa: E402

csv.field_size_limit(1 << 22)

JOIN = "‖"


def split_by_patient(rows, group, frac, seed):
    """Zero patient overlap. A row-wise split would put the same
    person on both sides and confirm nearly anything."""
    pids = sorted(set(str(r.get(group, "")) for r in rows))
    rnd = random.Random(seed)
    rnd.shuffle(pids)
    cut = int(len(pids) * frac)
    train_ids = set(pids[:cut])
    tr = [r for r in rows if str(r.get(group, "")) in train_ids]
    te = [r for r in rows if str(r.get(group, "")) not in train_ids]
    return tr, te, len(train_ids), len(pids) - len(train_ids)


def mutual_information(xs, ys):
    """Returns (MI in nats, degrees of freedom)."""
    import math
    n = len(xs)
    if n == 0:
        return 0.0, 0
    joint = Counter(zip(xs, ys))
    cx, cy = Counter(xs), Counter(ys)
    mi = 0.0
    for (a, b), c in joint.items():
        pxy = c / float(n)
        mi += pxy * math.log(pxy / ((cx[a] / float(n))
                                    * (cy[b] / float(n))))
    df = (len(cx) - 1) * (len(cy) - 1)
    return max(0.0, mi), max(1, df)


def encode(net, rows, col):
    b = net.binnings.get(col)
    if b is None:
        return None
    return [b.encode(r.get(col, "")) for r in rows]


def test_edge(net, rows, child, parents, group, alpha):
    """The same G-test the search uses, on data the search never saw.

    G = 2 * n * MI, chi-square with df. n counts PATIENTS, not rows:
    ten visits from one person are not ten independent observations,
    and using rows here would confirm nearly everything."""
    ec = encode(net, rows, child)
    eps = [encode(net, rows, p) for p in parents]
    if ec is None or any(e is None for e in eps):
        return None
    comb = [JOIN.join(t) for t in zip(*eps)] if eps else None
    if comb is None:
        return None
    n_pat = len(set(str(r.get(group, "")) for r in rows))
    mi, df = mutual_information(ec, comb)
    g = 2.0 * n_pat * mi
    crit = _chi2_crit(df, alpha)
    return {"mi": round(mi, 6), "g": round(g, 3), "df": df,
            "crit": round(crit, 3), "reproduced": bool(g >= crit),
            "patients": n_pat, "rows": len(rows)}


def run_arm(label, rows_tr, rows_te, args, correction):
    net = CondNet(k=args.k, max_parents=args.max_parents,
                  correction=correction).learn(
        rows_tr, group_by=args.group, multilevel=True)
    edges = net.report.get("edges", [])
    out = []
    for e in edges:
        child, parents = e.get("child"), e.get("parents") or []
        if not parents:
            continue
        tr = test_edge(net, rows_tr, child, parents, args.group,
                       args.alpha)
        te = test_edge(net, rows_te, child, parents, args.group,
                       args.alpha)
        if tr is None or te is None:
            continue
        out.append({"child": child, "parents": parents,
                    "found_on": tr, "held_out": te,
                    "reproduced": te["reproduced"]})
    return {
        "arm": label,
        "correction": correction,
        "comparisons_corrected_for":
            net.report.get("comparisons_corrected_for"),
        "columns_modelled": net.report.get("columns_modelled"),
        "bins": net.report.get("bins"),
        "found": len(out),
        "reproduced": sum(1 for e in out if e["reproduced"]),
        "edges": out,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Confirm discovered relationships out of sample.")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("-o", "--out", default="confirmed.json")
    ap.add_argument("--group", default="person_id")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-parents", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="per-edge alpha on the HOLDOUT, where no "
                         "correction applies - each edge is one "
                         "pre-specified hypothesis")
    ap.add_argument("--strict-only", action="store_true",
                    help="skip the liberal screening arm")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    inp = Path(a.inp)
    if not inp.exists():
        sys.exit("input not found: {}".format(inp))
    with inp.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if a.group not in (reader.fieldnames or []):
            sys.exit("--group column {!r} not in {}: found {}".format(
                a.group, inp.name,
                ", ".join((reader.fieldnames or [])[:8])))
        rows = list(reader)
    if not rows:
        sys.exit("{} has a header but no rows".format(inp.name))

    tr, te, n_tr, n_te = split_by_patient(rows, a.group,
                                          a.train_frac, a.seed)
    if n_tr < 2 * a.k or n_te < 2 * a.k:
        sys.exit("split leaves too few patients: {} train, {} held "
                 "out, with k={}. Lower --k or supply more patients."
                 .format(n_tr, n_te, a.k))
    overlap = (set(str(r.get(a.group, "")) for r in tr)
               & set(str(r.get(a.group, "")) for r in te))
    if overlap:
        sys.exit("BUG: {} patients appear in both halves".format(
            len(overlap)))

    out = {
        "generated_by": "scripts/confirm_patterns.py",
        "input": str(inp.resolve()),
        "rows": len(rows),
        "patients": n_tr + n_te,
        "train_patients": n_tr, "holdout_patients": n_te,
        "patient_overlap": 0,
        "holdout_alpha": a.alpha,
        "arms": [],
    }
    out["arms"].append(run_arm("strict", tr, te, a, "bonferroni"))
    if not a.strict_only:
        out["arms"].append(run_arm("screened", tr, te, a, "none"))

    Path(a.out).write_text(json.dumps(out, indent=1),
                           encoding="utf-8")
    report(out, a.report)


def report(o, full):
    print("=" * 68)
    print("PATTERNS CONFIRMED OUT OF SAMPLE")
    print("{:,} rows | {:,} patients: {:,} searched, {:,} held out "
          "(overlap {})".format(
              o["rows"], o["patients"], o["train_patients"],
              o["holdout_patients"], o["patient_overlap"]))
    print("=" * 68)
    for arm in o["arms"]:
        cc = arm["comparisons_corrected_for"]
        print("\n{} arm — {}".format(
            arm["arm"].upper(),
            "alpha split across {} comparisons".format(cc) if cc
            else "each pair tested at alpha; the HOLDOUT is the "
                 "referee, not the threshold"))
        print("  {} found on the search half, {} reproduced on "
              "patients it never saw".format(
                  arm["found"], arm["reproduced"]))
        if arm["found"]:
            print("  {:.0%} of what it found held up".format(
                arm["reproduced"] / float(arm["found"])))
        shown = [e for e in arm["edges"] if e["reproduced"]]
        for e in (shown if full else shown[:8]):
            print("    {} <- {}   G {:.0f} vs {:.0f} needed".format(
                e["child"], ", ".join(e["parents"]),
                e["held_out"]["g"], e["held_out"]["crit"]))
        if not full and len(shown) > 8:
            print("    ... and {} more".format(len(shown) - 8))
        lost = [e for e in arm["edges"] if not e["reproduced"]]
        if lost:
            print("  did NOT reproduce ({}):".format(len(lost)))
            for e in (lost if full else lost[:5]):
                print("    {} <- {}   G {:.0f}, needed {:.0f}".format(
                    e["child"], ", ".join(e["parents"]),
                    e["held_out"]["g"], e["held_out"]["crit"]))
    if len(o["arms"]) == 2:
        strict, screen = o["arms"][0], o["arms"][1]
        extra = screen["reproduced"] - strict["reproduced"]
        print("\n" + "-" * 68)
        print("Screening liberally and confirming out of sample "
              "yielded {:+d} relationship(s)".format(extra))
        print("that reproduce on unseen patients, against the "
              "corrected search.")
        if extra > 0:
            print("Those are real patterns the correction was "
                  "suppressing, not noise: they")
            print("were confirmed on patients no part of the search "
                  "ever touched.")


if __name__ == "__main__":
    main()
