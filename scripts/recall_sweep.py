"""Recall across SEEDS, because one seed cannot tell you anything.

    python scripts/recall_sweep.py --seeds 11,23,37 [--patients 800]
                                   [--confirm] [--max-visits 30]

Measured: recall on the same fixture varies 71%-86% across seeds with
nothing changed but the seed. That is a spread of 15 points from noise
alone, so a single-seed comparison cannot detect any improvement
smaller than that - and a one-relationship difference, which is what
15 points is at seven planted relationships, is exactly the size of
effect these changes produce.

This was learned the hard way: confirmation appeared to raise recall
from 71% to 86% at one seed. Three more seeds showed plain and
confirmed identical every time. The effect was noise and the
mechanism proposed for it was fiction.

Report the range, not a point.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
try:                       # console-safe on Windows terminals
    from _console import console_safe
    console_safe()
except Exception:
    pass


def recall_of(truth_path, model_path):
    r = subprocess.run(
        [sys.executable, "scripts/score_discovery.py",
         "--truth", str(truth_path), "--model", str(model_path)],
        capture_output=True, text=True, cwd=str(ROOT))
    base = adj = None
    for ln in r.stdout.splitlines():
        if ln.startswith("recall excluding"):
            adj = float(ln.split()[-1].rstrip("%")) / 100.0
        elif ln.startswith("recall "):
            base = float(ln.split()[-1].rstrip("%")) / 100.0
    return base, adj


def summarize(name, vals):
    ok = [v for v in vals if v is not None]
    if not ok:
        return "{}: no runs".format(name)
    lo, hi = min(ok), max(ok)
    mean = sum(ok) / len(ok)
    return ("{}: mean {:.0%}  range {:.0%}-{:.0%}  over {} seeds"
            .format(name, mean, lo, hi, len(ok)))


def main():
    ap = argparse.ArgumentParser(
        description="Recall across seeds, with its range.")
    ap.add_argument("--seeds", default="11,23,37")
    ap.add_argument("--patients", type=int, default=800)
    ap.add_argument("--max-visits", type=int, default=30)
    ap.add_argument("--confirm", action="store_true",
                    help="also run the confirmed path and compare")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--lags", action="store_true",
                    help="detect the time axis and engineer x__prev")
    ap.add_argument("--products", action="store_true",
                    help="second pass: centered products among what "
                         "the first pass left unexplained")
    # Measured at the real extract's width: the first pass leaves 22
    # columns unexplained, so 231 pairs exist and 400 covers all of
    # them. 150 covered 65% and the planted interaction fell in the
    # missing third - the budget, not the method, decided it.
    ap.add_argument("--product-budget", type=int, default=600)
    ap.add_argument("--bins", type=int, default=0,
                    help="override the auto bin count; 0 keeps auto, "
                         "which solves bins^(parents+1)~n/k from the "
                         "PATIENT count and gives 3 at this scale")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    if len(seeds) < 2:
        sys.exit("at least two seeds are needed: a single seed cannot "
                 "distinguish a change from noise, and the measured "
                 "spread is 15 points")

    from synthkit.condnet import CondNet
    plain_b, plain_a, conf_b, conf_a = [], [], [], []
    with tempfile.TemporaryDirectory() as td:
        for seed in seeds:
            d = Path(td) / "s{}".format(seed)
            subprocess.run(
                [sys.executable, "scripts/make_tidy_fixture.py",
                 "-o", str(d), "--patients", str(a.patients),
                 "--max-visits", str(a.max_visits),
                 "--seed", str(seed)],
                capture_output=True, cwd=str(ROOT))
            tidy = d / "tidy_visits_labeled.csv"
            if not tidy.exists():
                sys.exit("fixture generation failed for seed "
                         "{}".format(seed))
            with tidy.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            fs = {}
            if a.lags:
                from synthkit.temporal import (
                    detect_time_column, add_lag_features)
                tc, tk, _ = detect_time_column(rows, "person_id")
                rows, _lr = add_lag_features(rows, "person_id", tc, tk)
            if a.products:
                from synthkit.engineered import (
                    unexplained_columns, add_product_features,
                    feature_sources)
                scout = CondNet(k=a.k, max_bins=a.bins).learn(
                    rows, group_by="person_id", multilevel=True,
                    feature_sources=feature_sources(rows))
                un = unexplained_columns(scout, rows)
                rows, _pr = add_product_features(
                    rows, un, budget=a.product_budget)
            if a.lags or a.products:
                from synthkit.engineered import feature_sources
                fs = feature_sources(rows)
            net = CondNet(k=a.k, max_bins=a.bins).learn(
                rows, group_by="person_id", multilevel=True,
                feature_sources=fs)
            (d / "plain.json").write_text(net.to_json(),
                                          encoding="utf-8")
            b, adj = recall_of(d / "ground_truth.json",
                               d / "plain.json")
            plain_b.append(b)
            plain_a.append(adj)
            line = "seed {:<5} bins {:<3} plain {:>4.0%}".format(
                seed, net.report.get("bins"),
                b if b is not None else 0)
            if a.confirm:
                from synthkit.confirmed import learn_confirmed
                cn = learn_confirmed(rows, "person_id", k=a.k)
                (d / "conf.json").write_text(cn.to_json(),
                                             encoding="utf-8")
                cb, ca = recall_of(d / "ground_truth.json",
                                   d / "conf.json")
                conf_b.append(cb)
                conf_a.append(ca)
                line += "   confirmed {:>4.0%}".format(
                    cb if cb is not None else 0)
            print(line)
            sys.stdout.flush()

    print()
    print(summarize("plain    ", plain_b))
    print(summarize("plain adj", plain_a))
    if a.confirm:
        print(summarize("confirmed", conf_b))
        print(summarize("confirm adj", conf_a))
        pb = [v for v in plain_b if v is not None]
        cb = [v for v in conf_b if v is not None]
        if pb and cb:
            d_ = sum(cb) / len(cb) - sum(pb) / len(pb)
            spread = max(max(pb) - min(pb), max(cb) - min(cb))
            print()
            print("difference {:+.0%}, seed spread {:.0%}".format(
                d_, spread))
            if abs(d_) <= spread:
                print("NOT DISTINGUISHABLE from noise at these seeds.")
            else:
                print("larger than the seed spread; still worth more "
                      "seeds before relying on it.")


if __name__ == "__main__":
    main()
