"""Everything worth reading off a finished run, in ONE pass.

WHY THIS EXISTS. Reading a real run over the screen bridge costs a
round trip per question, and each one is a chance to mangle a
command or misread a digit. This asks every question at once and
writes the answers to a FILE, so the whole thing crosses through
`sayn.py` and every number arrives spelled.

It RECOMPUTES nothing the run already decided: the gate comes from
`synthkit.gate`, the fidelity figures from `fidelity.json`. What it
adds is the handful of properties that have no home in either -
whether an arithmetic identity actually holds in the written file,
whether a patient's dates agree with the order the file numbers
its visits by, and how much of the cohort's span one patient
occupies.

    python scripts/after_run.py RUNDIR [--src TIDY.csv]
    python scripts/after_run.py RUNDIR --src X.csv > out.txt
    python scripts/sayn.py out.txt
"""

import argparse
import json
import os
import sys
import warnings

# A mixed-format date column makes pandas warn once per call, and
# over the screen bridge that noise crowds out the answers.
warnings.filterwarnings("ignore", message=".*Could not infer format.*")
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthkit import dates as _dates                # noqa: E402
from synthkit import gate as G                      # noqa: E402


def load(run):
    out = {}
    for name in ("blueprint", "fidelity", "provenance"):
        f = run / "{}.json".format(name)
        out[name] = (json.loads(f.read_text(encoding="utf-8"))
                     if f.exists() else None)
    return out


def say(k, v):
    print("{:<34} {}".format(k, v))


def visit_order_col(frame):
    """The column THIS file declares its visit sequence by."""
    if "visit_number" in frame.columns:
        return "visit_number"
    for c in frame.columns:
        if "date" in c.lower():
            return c
    return None


def row_order_share(frame, col, gid):
    if col not in frame.columns or gid not in frame.columns:
        return None
    d = pd.to_datetime(frame[col], errors="coerce")
    if not d.notna().any():
        d = pd.to_numeric(frame[col], errors="coerce")
    g = d.groupby(frame[gid].astype(str))
    return float(g.apply(lambda s: s.is_monotonic_increasing).mean())


def span_ratio(frame, col, gid):
    """How much of the cohort's range one entity occupies."""
    if col not in frame.columns or gid not in frame.columns:
        return None
    d = pd.to_datetime(frame[col], errors="coerce")
    if not d.notna().any():
        return None
    g = d.groupby(frame[gid].astype(str))
    w = (g.max() - g.min()).dt.days.mean()
    t = (d.max() - d.min()).days
    if not t:
        return None
    return float(w) / float(t)


def identity_hold(frame, rec, bp_cols=None):
    """Does a published arithmetic identity hold in the FILE?

    Asked of the written output rather than of the engine's own
    report, because the question a reader has is whether the file
    they received contradicts itself.
    """
    child = rec.get("child")
    pa, pb = (rec.get("parents") or [None, None])[:2]
    for c in (child, pa, pb):
        if c not in frame.columns:
            return None
    # THE ENGINE'S OWN CONVERTER, because the weights are in the
    # units the blueprint published them in - ORDINAL DAYS since
    # `dates.DATE_ORIGIN`. The first version here divided a
    # datetime by 8.64e13 nanoseconds, which is wrong on pandas 3
    # where a datetime need not be nanosecond resolution at all:
    # the check read the identity as holding on 0.0% of rows
    # where a direct `(end - start).dt.days == los` read 100%.
    # Two measurements of one thing disagreeing is this project's
    # alarm, and the one with its own unit assumption was wrong.
    def num(c):
        spec = ((bp_cols.get(c) or {}).get("date")
                if bp_cols else None)
        if spec:
            return _dates.to_ordinal(frame[c],
                                     spec.get("format")
                                     or "%Y-%m-%d")
        return pd.to_numeric(frame[c], errors="coerce")
    w = rec.get("weights") or [0.0, 0.0]
    b = float(rec.get("intercept") or 0.0)
    got = num(child)
    want = float(w[0]) * num(pa) + float(w[1]) * num(pb) + b
    k = got.notna() & want.notna()
    if int(k.sum()) < 10:
        return None
    # A WHOLE-NUMBER CHILD IS COMPARED AT HALF A UNIT, because
    # that is what rounding can move it by.
    #
    # This used 2% of the child's spread for everything, and on
    # the real extract that read `age_at_visit` as holding on
    # 0.6267 of rows while the engine's own report said the
    # column had been computed on all of them. Two measurements
    # of one thing disagreeing is this project's alarm: for an
    # integral child the tier ROUNDS, so a row whose un-rounded
    # error was 0.4 is written 0.4 away from the prediction and
    # is correct, while 2% of a nineteen-year spread is 0.38 and
    # calls it broken. The tolerance was the artifact, not the
    # file.
    _diff = (got[k] - want[k]).abs()
    _whole = bool((got[k] % 1 == 0).all())
    if _whole:
        return float((_diff <= 0.5).mean())
    sd = float(got[k].std()) or 1.0
    return float((_diff <= 0.02 * sd).mean())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--src", default="")
    a = ap.parse_args()
    run = Path(a.run)
    if not run.is_dir():
        print("STOPPED: {} is not a directory".format(run))
        return 2
    art = load(run)
    bp, fid = art["blueprint"], art["fidelity"]
    if bp is None or fid is None:
        print("STOPPED: the run is missing blueprint.json or "
              "fidelity.json - this reads a FINISHED run")
        return 2

    gen_f = run / "generated.csv"
    gen = (pd.read_csv(gen_f, dtype=str, keep_default_na=False)
           if gen_f.exists() else None)
    gid = ((bp.get("patients") or {}).get("id_column")
           or "person_id")

    print("=== THE GATE, from synthkit.gate, recomputing nothing ===")
    res = G.assess(fid)
    for c in res["criteria"]:
        say(c["name"], "{}  {}".format(
            "PASS" if c["ok"] else "FAIL", c["detail"][:88]))
    say("met", res.get("met"))

    print()
    print("=== THE TIME AXIS AND THE ORDER THE FILE DECLARES ===")
    tc = (bp.get("patients") or {}).get("time_column")
    say("blueprint time_column", repr(tc))
    if gen is not None:
        say("declared visit order column",
            repr(visit_order_col(gen)))
        for c in sorted(gen.columns):
            if "date" not in c.lower():
                continue
            say("{} in row order".format(c),
                row_order_share(gen, c, gid))
            say("{} span ratio, generated".format(c),
                span_ratio(gen, c, gid))

    print()
    print("=== ARITHMETIC IDENTITIES, ASKED OF THE WRITTEN FILE ===")
    ids = bp.get("arithmetic_identities") or []
    say("identities published", len(ids))
    for rec in ids:
        say("{} == f({}, {}) holds on".format(
            rec["child"], rec["parents"][0], rec["parents"][1]),
            identity_hold(gen, rec, bp.get("columns"))
            if gen is not None else None)

    print()
    print("=== WHAT THE RUN SAID ABOUT ITS OWN WORK ===")
    for key in ("lag_only_parents",):
        v = (bp.get("excluded") or {}).get(key)
        say(key, len(v) if isinstance(v, list) else v)
    s = fid.get("summary") or {}
    for key in ("numeric", "centre_ok", "spread_ok",
                "coverage_ok", "compared", "sign_kept",
                "close", "pairs_inverted", "pairs_own_inverted"):
        if key in s:
            say("fidelity.{}".format(key), s[key])

    if a.src:
        print()
        print("=== THE SOURCE, FOR THE PROPERTIES THAT NEED IT ===")
        src = pd.read_csv(a.src, dtype=str, keep_default_na=False)
        say("source rows", len(src))
        say("source entities", src[gid].nunique()
            if gid in src.columns else None)
        for c in sorted(src.columns):
            if "date" not in c.lower():
                continue
            say("{} span ratio, source".format(c),
                span_ratio(src, c, gid))
    print()
    print("A RATIO near 1.0 means an entity's rows are spread "
          "across the whole cohort; near 0 means an episode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
