"""The rhythm of a patient's visits, read off two arms and the source.

WHY THIS EXISTS. `--visit-rhythm` reallocates WHICH patient holds
which drawn date, so the claim it makes about itself is narrow and
checkable: the spacing between a patient's visits moves toward the
source's, and the date column's own multiset does NOT move at all.
The gate cannot see either - `dynamics tracked` grades a derived
persistence statistic, and no criterion asks what a patient's gaps
look like.

So this reads the three quantities the pass is about, on the SOURCE
and on each arm's written file, and crosses the arms against each
other on the one property that makes the pass safe:

  SAME-DAY SHARE   the share of consecutive within-patient gaps
                   that are zero days - a run of visits in one day
  GAP MEDIAN       the middle gap, in days
  PATIENT SPAN     first visit to last, averaged over patients

  MULTISET         the two arms' date column sorted - IDENTICAL or
                   MOVED. Both arms draw at the same seed, so the
                   values are the same bag either way and the pass
                   only re-attaches them. MOVED means it is not a
                   reallocation, whatever else it achieved.

Every figure crosses in WORDS, because the digits channel off that
screen is not trustworthy: a zero has read back as @, 6, 8 and 9.

    python scripts/rhythm_read.py SRC.csv RUN_OFF RUN_ON
"""
import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from scripts.sayn import annotate  # noqa: E402


def read_table(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False,
                       low_memory=False)


def numeric_days(s):
    """A date column is read as NUMBERS where it is numeric.

    The engine's own date columns are days, and `_dcols` requires a
    numeric kind because the sort is numeric. Reading them with
    `to_datetime` is the wrong-type fault: days-since-epoch parse as
    NANOSECONDS and every patient lands on one instant.
    """
    d = pd.to_numeric(s, errors="coerce")
    if d.notna().sum() >= 20:
        return d.astype(float)
    t = pd.to_datetime(s, errors="coerce")
    if t.notna().sum() < 20:
        return None
    return (t - t.min()).dt.days.astype(float)


def rhythm(df, gid, dcol):
    """same-day share, gap median, mean patient span - or None."""
    d = numeric_days(df[dcol])
    if d is None:
        return None
    f = pd.DataFrame({"g": df[gid].astype(str), "d": d}).dropna()
    gaps = []
    spans = []
    for _, blk in f.groupby("g", sort=False):
        v = blk["d"].to_numpy(copy=True)
        if v.size < 2:
            continue
        v.sort()
        gaps.extend((v[1:] - v[:-1]).tolist())
        spans.append(float(v[-1] - v[0]))
    if not gaps:
        return None
    gs = pd.Series(gaps)
    return {"same_day_share": float((gs == 0).mean()),
            "gap_median": float(gs.median()),
            "gap_p99": float(gs.quantile(0.99)),
            "patient_span_mean": (
                float(pd.Series(spans).mean()) if spans else None),
            "patients": len(spans),
            "gaps": len(gaps)}


def multiset_same(a, b, dcol):
    """The one property that makes the pass a reallocation."""
    if dcol not in a.columns or dcol not in b.columns:
        return None
    x = numeric_days(a[dcol])
    y = numeric_days(b[dcol])
    if x is None or y is None:
        return None
    xs = x.dropna().sort_values().to_numpy()
    ys = y.dropna().sort_values().to_numpy()
    if xs.shape != ys.shape:
        return False
    return bool((xs == ys).all())


def recorded(run):
    """What the run itself says the pass did - absent is not zero."""
    p = Path(run) / "fidelity.json"
    if not p.exists():
        return None
    fid = json.loads(io.open(p, encoding="utf-8").read())
    gen = fid.get("generation") or {}
    return gen.get("visit_rhythm")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--group-by", default="person_id")
    ap.add_argument("--date-col", default=None)
    a = ap.parse_args(argv)

    src = read_table(a.src)
    gid = a.group_by
    if gid not in src.columns:
        print("STOPPED: the source has no column {!r} to group "
              "patients by".format(gid))
        return 2

    dcol = a.date_col
    if dcol is None:
        # The time axis the blueprint recorded, where a run carries
        # one - a second guess at what the date column IS is how two
        # halves come to disagree.
        for r in a.runs:
            bp = Path(r) / "blueprint.json"
            if bp.exists():
                b = json.loads(io.open(bp, encoding="utf-8").read())
                t = (b.get("patients") or {}).get("time_column")
                if t:
                    dcol = t
                    break
    if dcol is None or dcol not in src.columns:
        print("STOPPED: no date column (looked for {!r}); pass "
              "--date-col".format(dcol))
        return 2

    lines = []
    lines.append("RHYTHM on {} (grouped by {})".format(dcol, gid))
    lines.append("")

    rows = [("SOURCE", rhythm(src, gid, dcol))]
    frames = {}
    for r in a.runs:
        g = Path(r) / "generated.csv"
        if not g.exists():
            lines.append("{}: no generated.csv".format(Path(r).name))
            continue
        df = read_table(g)
        frames[r] = df
        rows.append((Path(r).name, rhythm(df, gid, dcol)))

    lines.append("{:<22} {:>10} {:>10} {:>10} {:>12}".format(
        "", "same-day", "gap med", "gap p99", "span mean"))
    for name, m in rows:
        if m is None:
            lines.append("{:<22} NOT READABLE - the date column "
                         "could not be read as days".format(name))
            continue
        lines.append("{:<22} {:>10.4f} {:>10.1f} {:>10.1f} {:>12.1f}"
                     .format(name, m["same_day_share"],
                             m["gap_median"], m["gap_p99"],
                             m["patient_span_mean"]))
    lines.append("")

    # THE REALLOCATION CLAIM, measured between the two arms rather
    # than read back from the field the pass writes. A check that
    # reads the report is asserting the report.
    names = list(frames)
    if len(names) >= 2:
        same = multiset_same(frames[names[0]], frames[names[1]], dcol)
        if same is None:
            lines.append("MULTISET: NOT READABLE on these two files")
        elif same:
            lines.append(
                "MULTISET: IDENTICAL - the two arms hold the same "
                "bag of dates, so the pass re-attached them and "
                "moved no value")
        else:
            lines.append(
                "MULTISET: MOVED - the arms do NOT hold the same "
                "bag of dates, so this is not a reallocation and "
                "the published marginal is not safe by construction")
    else:
        lines.append("MULTISET: not compared - needs two arms")
    lines.append("")

    for r in a.runs:
        rec = recorded(r)
        if rec is None:
            lines.append("{}: the run RECORDS no visit_rhythm - "
                         "absent, which is a different fact from "
                         "the pass running and doing nothing"
                         .format(Path(r).name))
        else:
            lines.append("{}: RECORDED {}".format(
                Path(r).name,
                ", ".join("{}={}".format(k, rec[k])
                          for k in sorted(rec))))

    print(annotate("\n".join(lines)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
