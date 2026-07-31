"""Null diagnostic: can ANY model predict a derived outcome on data
that carries no planted causality?

    python scripts/null_diagnostic.py --in tidy_visits_labeled.csv
                                      --label returned_within_30d

Trains synthkit's own solvers on a PERSON-GROUPED split (the same
patient never appears on both sides — with repeated visits per
person, a naive row split inflates every score) and reports AUROC
with a confidence interval.

Reading the result:
  ~0.50 with the interval straddling it  -> no learnable structure.
      Correct and expected for realistic-looking data with no
      planted truth. It is the argument for planting: an exam whose
      answers were never written cannot be graded.
  clearly >0.55 with the interval excluding 0.50 -> the source data
      DOES embed structure worth profiling; worth investigating
      which columns carry it before authoring outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
csv.field_size_limit(1 << 22)

from synthkit.autosolver import (          # noqa: E402
    autosolver, autosolver_hybrid, LAST_FIT)
from synthkit.mlmetrics import auroc, auroc_interval  # noqa: E402

# columns that would leak the answer or memorize identity
DROP = {"days_to_next_visit", "is_last_visit", "visit_id",
        "person_id", "visit_end_date"}


def grouped_split(rows, key="person_id", frac=0.7, seed="phase2"):
    """Split by PERSON, not by row."""
    def bucket(pid):
        h = hashlib.sha256((seed + ":" + str(pid)).encode())
        return int(h.hexdigest()[:8], 16) / 0xFFFFFFFF
    train, test = [], []
    for r in rows:
        (train if bucket(r.get(key, "")) < frac else test).append(r)
    return train, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--label", default="returned_within_30d")
    ap.add_argument("--drop-last-visits", action="store_true",
                    help="exclude right-censored final visits")
    a = ap.parse_args()

    with Path(a.src).open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if str(r.get(a.label, "")) in ("0", "1")]
    if a.drop_last_visits:
        rows = [r for r in rows
                if str(r.get("is_last_visit", "")) != "1"]
    if not rows:
        sys.exit("no labeled rows found")

    train, test = grouped_split(rows)
    ytr = [r[a.label] == "1" for r in train]
    yte = [1 if r[a.label] == "1" else 0 for r in test]

    def features(rs):
        return [{k: v for k, v in r.items()
                 if k not in DROP and k != a.label} for r in rs]

    n_pos, n_neg = sum(yte), len(yte) - sum(yte)
    if n_pos == 0 or n_neg == 0:
        # A label with one class is not a hard failure — it is a
        # finding: the chosen window produced no variation. Say so
        # and stop cleanly instead of crashing the pipeline.
        print("rows {} | test {} | positives {}".format(
            len(rows), len(yte), n_pos))
        print("\nDEGENERATE LABEL: the test split contains only "
              "one class, so no model can be scored.")
        print("  This usually means the outcome window is wrong "
              "for this data's rhythm — widen or narrow it "
              "(--window) and re-derive. It is a property of the "
              "source, not an error.")
        return
    print("rows {} | train {} ({} patients) | test {} ({} patients)"
          .format(len(rows), len(train),
                  len({r['person_id'] for r in train}),
                  len(test), len({r['person_id'] for r in test})))
    print("test prevalence: {}/{} = {:.1%}"
          .format(n_pos, len(yte), n_pos / max(1, len(yte))))
    overlap = ({r["person_id"] for r in train}
               & {r["person_id"] for r in test})
    print("patient overlap between train and test: {} (must be 0)"
          .format(len(overlap)))
    print()

    for name, factory in (("autosolver (tabular)", autosolver),
                          ("autosolver_hybrid (tabular + text)",
                           autosolver_hybrid)):
        scores = factory()(features(train), ytr, features(test))
        a_ = auroc(scores, yte)
        lo, hi = auroc_interval(a_, n_pos, n_neg)
        verdict = ("no learnable structure (interval includes 0.50)"
                   if lo <= 0.50 <= hi else
                   "STRUCTURE FOUND (interval excludes 0.50)")
        print("  {:36s} AUROC {:.3f}  [{:.3f}, {:.3f}]  -> {}"
              .format(name, a_, lo, hi, verdict))
        tag = ("autosolver" if factory is autosolver
               else "autosolver_hybrid")
        fit = LAST_FIT.get(tag) or {}
        for t in (fit.get("top") or [])[:3]:
            print("        strongest weight: {:+.3f}  {}"
                  .format(t["weight"], t["name"]))


if __name__ == "__main__":
    main()
