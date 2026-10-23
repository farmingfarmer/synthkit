"""Two sources, one engine - what did the DATA decide?

WHY IT EXISTS, AND THE TRAP IT EXISTS TO STOP. Running the same engine
on an 800-patient and a 100-patient cut of the same generator, the
SMALLER cohort scored HIGHER on the gate: close 91.4% against 87.5%.
Read alone that says the thin cohort is better. It is not. At 100
patients far fewer patterns clear the privacy floor, so the gate grades
a smaller and stronger surviving set - the run is faithful about less.
Measured on the same pair: 61% of the data lands on a publishable value
against 75%, and the output carries 12 distinct condition codes against
60.

A percentage whose denominator moved is not a comparison. So this page
never prints a fidelity number without the coverage number beside it,
and says that rule out loud where a reader will see it.

  python scripts/compare_sources.py NAME=SRC.csv:RUNDIR NAME=SRC:RUN ...
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
for _p in (str(_root), str(_root / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

WARNING = (
    "READ THE TWO BLOCKS TOGETHER. Fidelity is scored only on the "
    "patterns that cleared the privacy floor, so a thinner source can "
    "post a HIGHER percentage while publishing far less of its data. "
    "Coverage says how much survived; fidelity says how faithful the "
    "survivors are. Neither is the answer alone.")


def _read(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False).replace("", np.nan)


def _set_cols(df, gid):
    from synthkit import sets
    return [c for c in df.columns
            if c != gid and sets.looks_like_set(df[c])]


def _vocab(df, col):
    toks, sizes = set(), []
    for v in df[col].fillna(""):
        t = [x for x in str(v).split(";") if x]
        toks.update(t)
        sizes.append(len(t))
    return len(toks), (float(np.mean(sizes)) if sizes else 0.0)


def profile(name, src_path, run_dir, group_by="person_id", k=10):
    from synthkit import forecast, gate
    src = _read(src_path)
    run = Path(run_dir)
    gen = _read(run / "generated.csv")
    cen = forecast.census(src, group_by, k=k)
    out = {"name": name, "patients": cen["patients"], "rows": cen["rows"],
           "mass": cen["publishable_mass"], "verdict": cen["verdict"],
           "n_pub": cen["n_publishable"], "n_cols": cen["n_columns"],
           "sets": [], "gate": None}
    for c in _set_cols(src, group_by):
        if c not in gen.columns:
            continue
        sv, ss = _vocab(src, c)
        gv, gs = _vocab(gen, c)
        out["sets"].append({"column": c, "src_tokens": sv, "gen_tokens": gv,
                            "src_per_row": ss, "gen_per_row": gs})
    fj = run / "fidelity.json"
    if fj.exists():
        a = gate.assess(json.loads(fj.read_text(encoding="utf-8")))
        out["gate"] = {"met": a["met"], "pairs": a["pairs"],
                       "criteria": {c["name"]: c for c in a["criteria"]}}
    return out


def _frac(crit):
    """The raw n/d, so a denominator can never hide inside a percent."""
    if not crit:
        return "-", None
    # TAKE THE LEADING TOKEN ONLY. The criteria do not share a shape -
    # `close` reads "28/32 = 87.5%  (need...)", `inverted` reads
    # "0 - absolute...", `shapes tracked` reads "46/69 effect curves
    # within 0.35 sd - ..." - and splitting on " =" let two of them
    # drag a sentence into a 24-character column.
    head = crit["detail"].strip().split()[0].rstrip(",")
    try:
        n, d = head.split("/")
        return head, 100.0 * int(n) / int(d)
    except Exception:
        return head, None


def render(profiles):
    L = []
    w = 24
    names = [p["name"] for p in profiles]
    col = "{:<34}" + "{:<%d}" % w * len(names)
    L.append("SAME ENGINE, DIFFERENT SOURCES")
    L.append("")
    L.append(col.format("", *names))
    L.append("-" * (34 + w * len(names)))
    L.append("HOW MUCH THE SOURCE CAN PUBLISH")
    L.append(col.format("  patients", *["{:,}".format(p["patients"]) for p in profiles]))
    L.append(col.format("  rows", *["{:,}".format(p["rows"]) for p in profiles]))
    L.append(col.format("  publishable mass",
                        *["{:.0%}".format(p["mass"]) for p in profiles]))
    L.append(col.format("  columns keeping anything",
                        *["{}/{}".format(p["n_pub"], p["n_cols"]) for p in profiles]))
    L.append(col.format("  verdict", *[p["verdict"] for p in profiles]))
    cols = sorted({s["column"] for p in profiles for s in p["sets"]})
    if cols:
        L.append("")
        L.append("WHAT REACHED THE OUTPUT  (distinct values, and per row)")
        for c in cols:
            cells = []
            for p in profiles:
                s = next((x for x in p["sets"] if x["column"] == c), None)
                cells.append("-" if not s else "{} of {:,}".format(
                    s["gen_tokens"], s["src_tokens"]))
            L.append(col.format("  " + c[:30], *cells))
            cells = []
            for p in profiles:
                s = next((x for x in p["sets"] if x["column"] == c), None)
                cells.append("-" if not s else "{:.2f} vs {:.2f} src".format(
                    s["gen_per_row"], s["src_per_row"]))
            L.append(col.format("", *cells))
    if any(p["gate"] for p in profiles):
        L.append("")
        L.append("FIDELITY OF WHAT SURVIVED  (read with the block above)")
        L.append(col.format("  relationships scored",
                            *[str(p["gate"]["pairs"]) if p["gate"] else "-"
                              for p in profiles]))
        for nm in ("direction kept", "close", "inverted", "coverage",
                   "set token shares", "shapes tracked",
                   "interaction surfaces"):
            cells = []
            for p in profiles:
                if not p["gate"]:
                    cells.append("-")
                    continue
                head, pc = _frac(p["gate"]["criteria"].get(nm))
                cells.append(head if pc is None
                             else "{} ({:.0f}%)".format(head, pc))
            L.append(col.format("  " + nm, *cells))
        L.append(col.format("  GATE",
                            *["MET" if p["gate"]["met"] else "NOT MET"
                              if p["gate"] else "-" for p in profiles]))
    L.append("")
    for line in _wrap(WARNING, 34 + w * len(names)):
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
    group_by, k, specs = "person_id", 10, []
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--group-by":
            group_by = argv[i + 1]; i += 2
        elif argv[i] == "--k":
            k = int(argv[i + 1]); i += 2
        else:
            rest.append(argv[i]); i += 1
    for spec in rest:
        name, _, tail = spec.partition("=")
        src, _, run = tail.partition(":")
        if not (name and src and run):
            print("want NAME=SRC.csv:RUNDIR, got {!r}".format(spec))
            return 2
        specs.append((name, src, run))
    if len(specs) < 2:
        print(__doc__)
        return 2
    profs = [profile(n, s, r, group_by, k) for n, s, r in specs]
    print(render(profs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
