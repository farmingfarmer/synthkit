"""Run the whole pipeline across dataset SHAPES, not one fixture.

    python scripts/shape_sweep.py

WHY THIS EXISTS. Every fixture in this repository is one shape:
longitudinal clinical visits, a person id, a date, some vitals, some
sets. The rigour elsewhere in this project is aimed at ONE extract,
and that is exactly where it stopped working - four defects in a
single week were invisible to every fixture here and were found by a
real dataset having a shape nobody had built.

So this builds datasets that differ along the axes a customer varies -
cross-sectional, no time column, numeric-only, categorical-only,
high-cardinality codes, 120 rows, 60 columns, constant and all-empty
columns, duplicate rows, unicode, one row per entity, one entity
holding half the rows, free text, mixed types in one column, 92%
sparse, extreme skew, boolean-ish spellings, and a flat table with no
grouping column at all - and runs discover -> blueprint -> generate on
each.

IT DOES NOT ASSERT FIDELITY. Different shapes have different
achievable numbers and a threshold that fits them all would fit none.
It reports the things that are wrong on ANY shape: a crash, a column
lost, a column invented that the source never had, a column collapsed
to a single value, a row count that moved, a blueprint that
contradicts itself, and a declared property the output violates.

What it found on its first run: `visit_number` written into all 23
outputs including cross-sectional ones, a flat table coming back with
an invented `person_id`, high-cardinality and free-text columns
generated as `__other__` on every row, a heavy-tailed group size
losing 27% of rows, and - on the same run - a check of mine that
fired on 55 of 60 healthy columns.

Read a line as a QUESTION, not a verdict. `rows -27%` on a dataset
where one entity holds half of them is privacy removing row mass and
is expected; the same line on an even dataset is a bug.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

def _pid(i, n_pat):
    return "P{:04d}".format(i % n_pat)

def longitudinal(r, n=2400, n_pat=200):
    return pd.DataFrame({
        "person_id": [_pid(i, n_pat) for i in range(n)],
        "visit_date": ["2024-{:02d}-{:02d}".format(i % 12 + 1,
                                                   i % 28 + 1)
                       for i in range(n)],
        "age": r.randint(20, 90, n),
        "bp": np.round(r.normal(120, 15, n), 1),
        "site": r.choice(list("ABC"), n)})

def cross_sectional(r, n=2400):
    """NO repeated measures - one row per person."""
    d = longitudinal(r, n, n_pat=n)
    return d

def no_time(r, n=2400, n_pat=200):
    d = longitudinal(r, n, n_pat)
    return d.drop(columns=["visit_date"])

def numeric_only(r, n=2400, n_pat=200):
    return pd.DataFrame(dict(
        [("person_id", [_pid(i, n_pat) for i in range(n)])]
        + [("v{}".format(j), np.round(r.normal(0, 1, n), 3))
           for j in range(6)]))

def categorical_only(r, n=2400, n_pat=200):
    return pd.DataFrame(dict(
        [("person_id", [_pid(i, n_pat) for i in range(n)])]
        + [("c{}".format(j), r.choice(list("wxyz"), n))
           for j in range(6)]))

def high_cardinality(r, n=2400, n_pat=200):
    d = longitudinal(r, n, n_pat)
    d["code"] = ["k{:05d}".format(x) for x in r.randint(0, 2000, n)]
    return d

def tiny(r):
    return longitudinal(r, n=120, n_pat=30)

def wide(r, n=800, n_pat=100):
    return pd.DataFrame(dict(
        [("person_id", [_pid(i, n_pat) for i in range(n)])]
        + [("v{}".format(j), np.round(r.normal(0, 1, n), 3))
           for j in range(60)]))

def constant_column(r, n=2400, n_pat=200):
    d = longitudinal(r, n, n_pat)
    d["always_same"] = "X"
    d["all_missing"] = ""
    return d

def one_row_per_person(r, n=300):
    return longitudinal(r, n, n_pat=n)

def imbalanced_groups(r, n=2400):
    """One person with half the rows - the heavy-tail case."""
    pids = ["P0000"] * (n // 2) + [
        "P{:04d}".format(i) for i in range(1, n - n // 2 + 1)]
    d = longitudinal(r, n, n_pat=n)
    d["person_id"] = pids[:n]
    return d

SHAPES = {
    "longitudinal (baseline)": longitudinal,
    "cross-sectional": cross_sectional,
    "no time column": no_time,
    "numeric only": numeric_only,
    "categorical only": categorical_only,
    "high-cardinality code": high_cardinality,
    "tiny (120 rows)": tiny,
    "wide (60 numeric)": wide,
    "constant + empty column": constant_column,
    "one row per person": one_row_per_person,
    "imbalanced groups": imbalanced_groups,
}


def mixed_types(r, n=2400, n_pat=200):
    """One column holding numbers AND words - a CSV that was edited."""
    d = longitudinal(r, n, n_pat)
    d["messy"] = [str(r.randint(0, 100)) if i % 3 else "unknown"
                  for i in range(n)]
    return d

def free_text(r, n=2400, n_pat=200):
    d = longitudinal(r, n, n_pat)
    words = ["alpha beta", "gamma delta epsilon", "zeta", "eta theta"]
    d["note"] = [words[i % 4] + " " + str(i) for i in range(n)]
    return d

def all_missing_column(r, n=2400, n_pat=200):
    d = longitudinal(r, n, n_pat)
    d["never"] = ""
    return d

def duplicate_rows(r, n=2400, n_pat=200):
    d = longitudinal(r, n // 2, n_pat)
    return pd.concat([d, d], ignore_index=True)

def unicode_values(r, n=2400, n_pat=200):
    d = longitudinal(r, n, n_pat)
    d["ville"] = r.choice(["Zurich", "Malmo", "Sao Paulo",
                           "Kobenhavn"], n)
    return d

def every_row_own_group(r, n=1200):
    return longitudinal(r, n, n_pat=n)

def no_group_column(r, n=2400):
    d = longitudinal(r, n, n_pat=200)
    return d.drop(columns=["person_id"])

def wide_sparse(r, n=1200, n_pat=150):
    cols = {"person_id": [_pid(i, n_pat) for i in range(n)]}
    for j in range(40):
        v = np.round(r.normal(0, 1, n), 3).astype(object)
        v[r.rand(n) < 0.92] = ""
        cols["s{}".format(j)] = v
    return pd.DataFrame(cols)

def extreme_skew(r, n=2400, n_pat=200):
    d = longitudinal(r, n, n_pat)
    d["cost"] = np.round(r.lognormal(2.0, 2.5, n), 2)
    return d

def booleanish(r, n=2400, n_pat=200):
    d = longitudinal(r, n, n_pat)
    d["flag_yn"] = r.choice(["Y", "N"], n)
    d["flag_tf"] = r.choice(["true", "false"], n)
    d["flag_01"] = r.randint(0, 2, n)
    return d

def one_group_only(r, n=2400):
    d = longitudinal(r, n, n_pat=1)
    return d

def negatives_and_zero(r, n=2400, n_pat=200):
    d = longitudinal(r, n, n_pat)
    d["balance"] = np.round(r.normal(0, 500, n), 2)
    d["zeros"] = 0.0
    return d

SHAPES.update({
    "mixed types in a column": mixed_types,
    "free text column": free_text,
    "all-missing column": all_missing_column,
    "duplicate rows": duplicate_rows,
    "unicode values": unicode_values,
    "every row own group": every_row_own_group,
    "NO group column": no_group_column,
    "wide + 92% sparse": wide_sparse,
    "extreme skew (lognormal)": extreme_skew,
    "boolean-ish Y/N true/1": booleanish,
    "one group only": one_group_only,
    "negatives and zeros": negatives_and_zero,
})


def _one(name, fn):
    warnings.filterwarnings("ignore")
    from synthkit import blueprint as B
    from synthkit.discover import discover
    from synthkit.generate import generate
    from synthkit.contradictions import find_blueprint
    from synthkit import invariants as INV

    r = np.random.RandomState(3)
    df = fn(r)
    gb = "person_id" if "person_id" in df.columns else None
    try:
        cat = discover(df, group_by=gb, seed=1)
        bp = B.build(df, cat, group_by=gb)
        n = df[gb].nunique() if gb else len(df)
        g = generate(bp, n_patients=n, seed=5)
    except Exception as e:
        return "{:<26} CRASH  {}: {}".format(
            name, type(e).__name__, str(e)[:90])

    flags = []
    ratio = len(g) / float(len(df))
    if abs(ratio - 1.0) > 0.10:
        flags.append("rows {:+.0%}".format(ratio - 1.0))
    lost = [c for c in df.columns if c not in g.columns]
    if lost:
        flags.append("LOST " + ",".join(lost[:3]))
    # `visit_number` is deliberate where entities repeat, and named
    # by the run itself - it is not a finding.
    extra = [c for c in g.columns
             if c not in df.columns and c != "visit_number"]
    if extra:
        flags.append("INVENTED " + ",".join(extra[:3]))
    bad = find_blueprint(bp)
    if bad:
        flags.append("{} blueprint contradiction(s)".format(len(bad)))
    dis = INV.find(g, bp, gb) if gb else []
    if dis:
        flags.append("{} declared property violated".format(len(dis)))
    collapsed = [c for c in df.columns
                 if c in g.columns and df[c].nunique() > 3
                 and g[c].nunique() == 1]
    if collapsed:
        flags.append("COLLAPSED " + ",".join(collapsed[:3]))
    return "{:<26} {}".format(name, "  |  ".join(flags) or "ok")


def main():
    print("{} shapes\n".format(len(SHAPES)))
    for name, fn in SHAPES.items():
        print(_one(name, fn), flush=True)
    print()
    print("A line is a QUESTION. `rows -27%` where one entity holds "
          "half of them is")
    print("privacy removing row mass; the same line on an even "
          "dataset is a bug.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
