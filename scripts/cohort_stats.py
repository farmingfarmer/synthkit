"""Per-person statistics of two cohorts, side by side - the reading
somebody asks for when they want to know whether a synthetic cohort
"has the same patterns" as a real one.

WHY IT IS NOT A `df.describe()`. The question is asked about PEOPLE
("average encounters per patient") while the table is rows, so every
statistic here is computed per patient first and summarised second.
Pooling rows answers a different question and answers it wrongly: a
person seen 200 times would carry two hundred times the weight of a
person seen once.

AND WHY THE EXTREMES ARE NOT THE EXTREMES. A per-patient maximum IS
one individual's value - the single busiest person in the cohort -
and this file's own rule is that no published number may be an
individual's value. So the reported low and high are the MEAN of the
k most extreme patients, exactly as the blueprint's bounds are. The
mean, median and share are aggregates over everybody and are
published as measured. Which is which is printed, because a column
of numbers whose rules differ silently is worse than no column.

  python scripts/cohort_stats.py NAME=cohort.csv NAME=other.csv \
      --group-by person_id --k 10 [--time-col visit_start_date]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
for _p in (str(_root), str(_root / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

NOTE = (
    "HOW TO READ THIS. Every number is computed per PATIENT and then "
    "summarised, because the question is about people and the table "
    "is rows. mean / median / share are aggregates over the whole "
    "cohort and are printed as measured. low / high are NOT the "
    "cohort's true minimum and maximum - a single patient's own value "
    "may not be published, so each is the MEAN of the k most extreme "
    "patients. A real cohort and a synthetic one are screened the "
    "same way here, so the two columns stay comparable.")


def _read(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def _bound(vals, k, high):
    """The mean of the k most extreme patients - never one person's
    own value. Mirrors the blueprint's published-bound rule."""
    v = np.sort(np.asarray(vals, dtype=float))
    if len(v) < k:
        return None
    tail = v[-k:] if high else v[:k]
    return float(np.mean(tail))


def _summary(per_patient, k):
    """per_patient: one number per patient. Returns the publishable
    summary of that distribution, or None when too few patients."""
    v = np.asarray([x for x in per_patient if x is not None and
                    not (isinstance(x, float) and np.isnan(x))],
                   dtype=float)
    if len(v) < k:
        return None
    return {"n": int(len(v)),
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "sd": float(np.std(v)),
            "low": _bound(v, k, high=False),
            "high": _bound(v, k, high=True)}


def _token_count(series):
    """Distinct TOKENS, not distinct combination strings. A set
    column's levels are its tokens - this file's recurring lesson,
    and counting the joined string reports one shared code as none."""
    toks = set()
    for val in series.fillna(""):
        for t in str(val).split(";"):
            if t and t.strip():
                toks.add(t.strip())
    return len(toks)


def person_stats(name, path, group_by="person_id", k=10, time_col=None):
    from synthkit import sets
    df = _read(path)
    if group_by not in df.columns:
        raise ValueError(
            "no column named {!r} in {} - the per-person reading "
            "counts distinct PATIENTS, so it cannot run without it; "
            "the columns are: {}".format(
                group_by, path, ", ".join(map(str, df.columns[:12]))))
    g = df.groupby(group_by, sort=False)
    out = {"name": name, "patients": int(df[group_by].nunique()),
           "rows": int(len(df)), "k": k, "stats": {}, "notes": []}

    # visits per patient - the "average encounters" question itself.
    # ASSIGNED ONLY WHEN IT EXISTS: a key mapping to None reads as a
    # statistic to anything downstream, and the below-floor cohort is
    # exactly the case where there is none.
    enc = _summary(g.size().tolist(), k)
    if enc:
        out["stats"]["encounters per patient"] = enc

    # set columns: distinct tokens a patient holds, and how often the
    # column is empty for them
    set_cols = [c for c in df.columns
                if c != group_by and sets.looks_like_set(df[c])]
    for c in sorted(set_cols):
        per = [_token_count(sub[c]) for _, sub in g]
        s = _summary(per, k)
        if s:
            out["stats"]["distinct {} per patient".format(c)] = s
        share = [float(np.mean([1.0 if not str(x).strip() else 0.0
                                for x in sub[c]])) for _, sub in g]
        s2 = _summary(share, k)
        if s2:
            out["stats"]["share of visits with no {}".format(c)] = s2

    # span in days, only when a usable date column was NAMED. Detecting
    # one is good enough to order rows and not good enough to publish a
    # statistic from somebody's extract.
    if time_col:
        if time_col not in df.columns:
            out["notes"].append(
                "no column named {!r}, so span in days is NOT "
                "reported rather than guessed from another "
                "column".format(time_col))
        else:
            per = []
            for _, sub in g:
                t = pd.to_datetime(sub[time_col], errors="coerce")
                t = t.dropna()
                per.append(float((t.max() - t.min()).days)
                           if len(t) >= 2 else np.nan)
            s = _summary(per, k)
            if s:
                out["stats"]["span in days per patient"] = s
            bad = pd.to_datetime(df[time_col], errors="coerce").isna()
            if bad.any():
                out["notes"].append(
                    "{:.1%} of {} values did not parse as a date"
                    .format(float(bad.mean()), time_col))
    else:
        out["notes"].append(
            "span in days needs --time-col; it is not reported "
            "because no date column was named")

    if out["patients"] < k:
        out["notes"].append(
            "ONLY {} PATIENTS, BELOW THE FLOOR OF {} - no per-person "
            "extreme can be published at all for this cohort"
            .format(out["patients"], k))
    return out


def render(profiles):
    L = []
    w = 26
    names = [p["name"] for p in profiles]
    col = "{:<36}" + ("{:<%d}" % w) * len(names)
    L.append("THE SAME QUESTION ASKED OF EVERY COHORT: PER PATIENT")
    L.append("")
    L.append(col.format("", *names))
    L.append("-" * (36 + w * len(names)))
    L.append(col.format("  patients",
                        *["{:,}".format(p["patients"]) for p in profiles]))
    L.append(col.format("  rows",
                        *["{:,}".format(p["rows"]) for p in profiles]))
    keys = []
    for p in profiles:
        for key in p["stats"]:
            if key not in keys:
                keys.append(key)
    for key in keys:
        L.append("")
        L.append("  " + key)
        for field, label in (("mean", "mean (measured)"),
                             ("median", "median (measured)"),
                             ("sd", "spread (measured)"),
                             ("low", "low (mean of k lowest)"),
                             ("high", "high (mean of k highest)")):
            cells = []
            for p in profiles:
                s = p["stats"].get(key)
                cells.append("-" if not s or s.get(field) is None
                             else "{:,.2f}".format(s[field]))
            L.append(col.format("      " + label, *cells))
    notes = [(p["name"], n) for p in profiles for n in p["notes"]]
    if notes:
        L.append("")
        L.append("WHAT IS NOT REPORTED, AND WHY")
        for nm, n in notes:
            for line in _wrap("{}: {}".format(nm, n),
                              34 + w * len(names)):
                L.append("  " + line)
    L.append("")
    for line in _wrap(NOTE, 36 + w * len(names)):
        L.append(line)
    return "\n".join(L)


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out.append(line)
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    group_by, k, time_col, rest = "person_id", 10, None, []
    i = 0
    while i < len(argv):
        if argv[i] == "--group-by":
            group_by = argv[i + 1]; i += 2
        elif argv[i] == "--k":
            k = int(argv[i + 1]); i += 2
        elif argv[i] == "--time-col":
            time_col = argv[i + 1]; i += 2
        else:
            rest.append(argv[i]); i += 1
    specs = []
    for spec in rest:
        name, _, path = spec.partition("=")
        if not (name and path):
            print("want NAME=cohort.csv, got {!r}".format(spec))
            return 2
        specs.append((name, path))
    if not specs:
        print(__doc__)
        return 2
    print("cohort_stats: --group-by {} --k {} --time-col {}".format(
        group_by, k, time_col))
    profs = [person_stats(n, p, group_by, k, time_col) for n, p in specs]
    print(render(profs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
