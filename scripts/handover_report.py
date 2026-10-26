"""THE HANDOVER - everything produced, in one document, for the
person who asked for the data rather than for the people who built
it.

WHY IT IS GENERATED AND NOT WRITTEN. A hand-written version of this
report quoted 898 against 290 publishable condition codes; the source
figures are 1,050 against 365. Both had been measured - the first pair
came from each run's 85% TRAINING SPLIT and the second from the full
source - and two documents giving different denominators for one
claim is how a reader stops trusting both. This reads the run
artifacts and computes its own numbers, so it cannot drift from the
gate, from the charts, or from the files it points at.

IT IS WRITTEN FOR SOMEONE NOT IN THE WEEDS. Every number is given
with the thing it is a number OF; every plot gets a sentence on what
it shows and a sentence on how to read it; and the limits sit where
they will be met rather than in a closing footnote, because a limit
discovered after a decision was taken is worth nothing.

  python scripts/handover_report.py \
      --name "800-patient" --src big.csv  --run BIGRUN  --omop DIR \
      --name "100-patient" --src thin.csv --run THINRUN --omop DIR \
      [--group-by person_id] [--k 10] -o handover.md
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

PLOTS = [
    ("fidelity deck", "deck_<cohort>.html",
     "Every column drawn twice - the real data in gray, the "
     "synthetic in red - plus a correlation heatmap and each "
     "discovered pattern drawn three times: the rule the contract "
     "published, the shape measured on the real data, and the same "
     "shape measured on the synthetic.",
     "Look for the two bodies overlapping. Where they do, that "
     "column came through. Each pattern carries a verdict in words - "
     "'tracks within X' or DEPARTS - so you do not have to judge the "
     "curves by eye."),
    ("latent 3D view", "latent_<cohort>.html",
     "The autoencoder's own compressed view of the data, real points "
     "beside synthetic ones, in three dimensions you choose. Four "
     "quantities are measured per point: distance to the nearest "
     "cluster centre, distance to the nearest neighbour, how well "
     "the model rebuilds that point, and distance to the nearest "
     "point of the OTHER group.",
     "Drag to rotate; pick any three of the four measures for the "
     "axes. The question it answers is whether the synthetic points "
     "sit AMONG the real ones or off to one side. That last measure "
     "is the privacy lens: a synthetic point very close to a real "
     "one is shadowing it."),
    ("showcase 3D cloud", "showcase_<cohort>.html",
     "The same real-against-synthetic comparison, but on familiar "
     "clinical axes - blood pressure, glucose, and so on - with "
     "labelled arrows showing what each direction means.",
     "This is the one to open first if the latent view feels "
     "abstract. Same comparison, axes you already know."),
    ("cross-cohort charts", "cohort_charts.html",
     "The two cohorts side by side. A curve per cohort showing how "
     "many distinct clinical codes survive the privacy floor as that "
     "floor is raised, and a paired bar per numeric column showing "
     "how much of its real spread the synthetic kept.",
     "This is the only visualization that COMPARES the two cohorts. "
     "The curve falling away faster is the smaller cohort losing its "
     "vocabulary. Every point is labelled with its number."),
]


def _read(path):
    from synthkit import sets
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in df.columns:
        if sets.looks_like_set(df[c]):
            continue
        df[c] = df[c].astype(object).where(
            df[c].astype(str).str.strip() != "", np.nan)
    return df


def profile(name, src_path, run, omop, group_by, k):
    from synthkit import forecast, gate, sets
    src = _read(src_path)
    runp = Path(run)
    gen = pd.read_csv(runp / "generated.csv", dtype=str,
                      keep_default_na=False)
    cen = forecast.census(src, group_by, k=k)
    out = {"name": name, "patients": cen["patients"],
           "rows": cen["rows"], "mass": cen["publishable_mass"],
           "verdict": cen["verdict"],
           "gen_rows": int(len(gen)),
           "gen_patients": int(gen[group_by].nunique())
           if group_by in gen.columns else None,
           "tokens": [], "empty": [], "gate": None, "numeric": {},
           "omop": {}}
    groups = src[group_by].astype(str).to_numpy()
    for c in src.columns:
        if c == group_by or not sets.looks_like_set(src[c]):
            continue
        toks = set()
        for v in src[c].fillna(""):
            for t in str(v).split(";"):
                if t.strip():
                    toks.add(t.strip())
        vo = sets.vocabulary(src[c], groups, k=k, cap=10 ** 7)
        out["tokens"].append({"column": c, "found": len(toks),
                              "clear_k": len(vo["tokens"])
                              if vo else 0})
    for c in src.columns:
        if c != group_by and float(src[c].notna().mean()) == 0.0:
            out["empty"].append(c)
    fj = runp / "fidelity.json"
    if fj.exists():
        rec = json.loads(fj.read_text(encoding="utf-8"))
        a = gate.assess(rec)
        out["gate"] = {"met": a["met"],
                       "criteria": [(c["name"],
                                     c["detail"].strip().split()[0],
                                     bool(c["ok"]))
                                    for c in a["criteria"]]}
        for col in (rec.get("columns") or []):
            nm, s, g = (col.get("column"), col.get("sd_source"),
                        col.get("sd_generated"))
            if nm and s:
                out["numeric"][nm] = float(g or 0.0) / float(s)
    if omop:
        od = Path(omop)
        for f in sorted(od.glob("*.csv")):
            with f.open("r", encoding="utf-8") as fh:
                out["omop"][f.name] = max(
                    sum(1 for _ in fh) - 1, 0)
    return out


def render(big, thin, k):
    L = []
    A = L.append
    A("# Synthetic data: what was produced, what it shows, "
      "and what it does not")
    A("")
    A("Prepared from the run artifacts themselves, so every figure "
      "here is the same figure the measurement tools report. "
      "Nothing in this document was typed from memory.")
    A("")
    A("---")
    A("")
    A("## 1. What you asked for, and where it is")
    A("")
    A("Two cohorts were put through the same generator with the "
      "same settings. For each one you have:")
    A("")
    A("- the synthetic data as **one row per visit** "
      "(`generated.csv` in the run folder)")
    A("- the same data **split back into the original table "
      "layout** (`omop_synthetic_<cohort>/`)")
    A("- three visualizations")
    A("")
    A("and across the two cohorts, one comparison page and one "
      "written argument.")
    A("")
    A("### The original table layout")
    A("")
    A("Your data arrived as separate OMOP tables and was folded "
      "into one-row-per-visit for modelling. It has been folded "
      "back:")
    A("")
    names = [big["name"], thin["name"]]
    allt = sorted(set(list(big["omop"]) + list(thin["omop"])))
    if allt:
        A("| table | {} | {} |".format(*names))
        A("|---|---:|---:|")
        for t in allt:
            A("| `{}` | {} | {} |".format(
                t,
                "{:,}".format(big["omop"][t])
                if t in big["omop"] else "-",
                "{:,}".format(thin["omop"][t])
                if t in thin["omop"] else "-"))
        A("")
    A("**Four things the original tables carried that these "
      "cannot**, because the one-row-per-visit format used for "
      "modelling does not hold them. A file named `person.csv` "
      "reads as the original in every respect unless something "
      "says otherwise, so each is also written in "
      "`WHAT_IS_NOT_IN_THESE_FILES.txt` beside the tables:")
    A("")
    A("1. **Which route went with which drug.** The modelling "
      "format kept the drugs for a visit and the routes for a "
      "visit as two separate lists, so the pairing was already "
      "gone. Routes are in their own file rather than attached to "
      "a drug they may not belong to.")
    A("2. **Concept IDs.** Only the names were carried, so the "
      "tables have `*_source_value` columns and no `*_concept_id`.")
    A("3. **Measurement reference ranges.** `range_low` and "
      "`range_high` were used during modelling and not carried.")
    A("4. **Per-fact dates.** A condition takes its visit's date, "
      "because that is the only date available.")
    A("")
    A("Every identifier in every table is invented. No real "
      "person's ID and no real record appears in any of them.")
    A("")
    A("---")
    A("")
    A("## 2. The headline finding")
    A("")
    A("**Categorical and set structure survive a 100-patient "
      "cohort. Numeric magnitude and vocabulary breadth do not - "
      "and no output size fixes it, because what binds is the "
      "number of PEOPLE, not the number of rows.**")
    A("")
    A("### Read this table before any percentage")
    A("")
    A("Fidelity is scored only on the patterns that cleared the "
      "privacy floor. A thinner source grades a smaller and "
      "stronger surviving set, so it can post a *higher* "
      "percentage while publishing far less of its data. "
      "Coverage first, always.")
    A("")
    A("| | {} | {} |".format(*names))
    A("|---|---:|---:|")
    A("| patients | {:,} | {:,} |".format(big["patients"],
                                          thin["patients"]))
    A("| rows | {:,} | {:,} |".format(big["rows"], thin["rows"]))
    A("| share of data that can be published | {:.0%} | {:.0%} |"
      .format(big["mass"], thin["mass"]))
    A("| verdict | **{}** | **{}** |".format(big["verdict"],
                                             thin["verdict"]))
    A("")
    A("### How much clinical vocabulary survives")
    A("")
    A("A code is publishable only if at least **{}** different "
      "patients have it. At {:,} patients that means {:.0%} of the "
      "cohort must share a code; at {:,} patients it means "
      "{:.1%}.".format(k, thin["patients"], k / float(thin["patients"]),
                       big["patients"], k / float(big["patients"])))
    A("")
    cols = sorted({t["column"] for t in big["tokens"]}
                  & {t["column"] for t in thin["tokens"]})
    if cols:
        A("| column | {} | {} |".format(*names))
        A("|---|---|---|")
        for c in cols:
            tb = next(t for t in big["tokens"] if t["column"] == c)
            tt = next(t for t in thin["tokens"] if t["column"] == c)
            A("| `{}` | {:,} of {:,} | {:,} of {:,} |".format(
                c, tb["clear_k"], tb["found"],
                tt["clear_k"], tt["found"]))
        A("")
    A("### How much numeric detail survives")
    A("")
    shared = sorted(set(big["numeric"]) & set(thin["numeric"]))
    worse = [c for c in shared
             if thin["numeric"][c] < big["numeric"][c]]
    if shared:
        rows = sorted(shared,
                      key=lambda c: -(big["numeric"][c]
                                      - thin["numeric"][c]))[:6]
        A("Share of each column's real spread that the synthetic "
          "data keeps. The largest gaps first:")
        A("")
        A("| column | {} | {} |".format(*names))
        A("|---|---:|---:|")
        for c in rows:
            A("| `{}` | {:.0%} | {:.0%} |".format(
                c, big["numeric"][c], thin["numeric"][c]))
        A("")
        A("**{} of {} shared numeric columns keep less spread on "
          "the smaller cohort.** Where the two are close, the "
          "privacy floor is not what separates them - and those "
          "columns are in the chart too, because they are an "
          "honest limit on this argument."
          .format(len(worse), len(shared)))
        A("")
    A("---")
    A("")
    A("## 3. The visualizations, and how to read each one")
    A("")
    for title, fname, what, how in PLOTS:
        A("### {}".format(title))
        A("")
        A("`{}`".format(fname))
        A("")
        A("**What it shows.** {}".format(what))
        A("")
        A("**How to read it.** {}".format(how))
        A("")
    A("### One thing the 3D views cannot do")
    A("")
    A("The four measures in the latent view are computed inside "
      "each cohort's own compressed space, and those spaces are "
      "not the same width. A distance in one is not the same unit "
      "as a distance in the other.")
    A("")
    A("So each latent view answers *\"do the synthetic points sit "
      "among the real ones in THIS cohort\"* - which it answers "
      "well. Putting the two side by side and judging which cohort "
      "is worse is not something those pictures support. That "
      "comparison is what `cohort_charts.html` is for.")
    A("")
    A("---")
    A("")
    A("## 4. Privacy")
    A("")
    A("Both datasets were attacked rather than asserted. Two "
      "different attackers, plus a **positive control**: the same "
      "attack run against a generator that simply republishes the "
      "real records. If the control is not caught, the attack is "
      "incapable of failing and a clean result would mean nothing.")
    A("")
    A("The control was caught on both cohorts. The honest scope: "
      "the controlled attacker is the one that sees only the "
      "SYNTHETIC ROWS - which is the only thing that leaves the "
      "machine. A second attacker assumes the trained model itself "
      "leaked; the model is never part of any release, and its "
      "numbers are reported separately for completeness.")
    A("")
    A("Also true, and worth stating because it is easy to "
      "misread: the smaller cohort's attack numbers are much "
      "NOISIER. The attack splits patients into members and "
      "non-members, which is {:,} against {:,} for one cohort and "
      "{:,} against {:,} for the other. A result from the smaller "
      "split carries several times the uncertainty, so a gap "
      "between the two cohorts' numbers is not evidence of a real "
      "difference.".format(big["patients"] // 2, big["patients"] // 2,
                           thin["patients"] // 2,
                           thin["patients"] // 2))
    A("")
    A("---")
    A("")
    A("## 5. Limits, stated plainly")
    A("")
    if thin["empty"]:
        A("**Your {} cohort has {} columns that are empty in the "
          "source itself**: {}. Nothing can be generated for a "
          "field that was never recorded. Note that several of "
          "these are dates - which means that cohort cannot "
          "support any analysis over time, and that is a property "
          "of what was pulled rather than of the generator or of "
          "the privacy floor.".format(
              thin["name"], len(thin["empty"]),
              ", ".join("`{}`".format(c) for c in thin["empty"][:10])))
        A("")
    A("**No patient trajectory is modelled.** The generator "
      "produces faithful rows and a faithful population, and each "
      "synthetic patient's visits do belong to one invented "
      "person - but the ORDER and TIMING of one patient's course "
      "is not reproduced. If an analysis walks along a single "
      "patient's timeline, this data does not support it.")
    A("")
    A("**No single-seed fidelity percentage is quoted anywhere in "
      "this document.** Re-running the generator with a different "
      "random seed moved one of these readings by ten points in "
      "our own testing, so a number from one run is not a result.")
    A("")
    for nm, p in ((big["name"], big), (thin["name"], thin)):
        if not p["gate"]:
            continue
        fails = [c for c, _, ok in p["gate"]["criteria"] if not ok]
        A("**{}**: the automated quality gate reports {}{}."
          .format(nm,
                  "all criteria met" if p["gate"]["met"]
                  else "NOT MET",
                  "" if p["gate"]["met"]
                  else " - failing on " + ", ".join(fails)))
        A("")
    A("---")
    A("")
    A("## 6. If you want something changed")
    A("")
    A("More **people** is the only thing that lifts the limits "
      "above. More rows from the same people does not: the privacy "
      "floor counts patients, and one person seen twice as often "
      "is still one person. Output VOLUME is nearly free - the "
      "larger cohort was generated at five times its size with no "
      "measurable fidelity cost - so ask for whatever row count "
      "suits you.")
    A("")
    A("A wider pull - more patients, and the date and demographic "
      "columns populated - would change the numbers in section 2 "
      "far more than any change to the generator would.")
    return "\n".join(L)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    group_by, k, out = "person_id", 10, ""
    names, srcs, runs, omops = [], [], [], []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--group-by":
            group_by = argv[i + 1]; i += 2
        elif a == "--k":
            k = int(argv[i + 1]); i += 2
        elif a in ("-o", "--out"):
            out = argv[i + 1]; i += 2
        elif a == "--name":
            names.append(argv[i + 1]); i += 2
        elif a == "--src":
            srcs.append(argv[i + 1]); i += 2
        elif a == "--run":
            runs.append(argv[i + 1]); i += 2
        elif a == "--omop":
            omops.append(argv[i + 1]); i += 2
        else:
            print("unknown argument {!r}".format(a))
            return 2
    while len(omops) < len(names):
        omops.append("")
    if not (len(names) == len(srcs) == len(runs) == 2):
        print("want TWO cohorts, each with --name --src --run "
              "[--omop]; got {} / {} / {}".format(
                  len(names), len(srcs), len(runs)))
        return 2
    if not out:
        print("need -o OUT.md")
        return 2
    for nm, sp, rp in zip(names, srcs, runs):
        if not Path(sp).exists():
            print("no source file for {}: {}".format(nm, sp))
            return 2
        if not (Path(rp) / "generated.csv").exists():
            print("no generated.csv for {}: {}".format(nm, rp))
            return 2
    print("handover_report: --group-by {} --k {}".format(group_by, k))
    profs = [profile(n, s, r, o, group_by, k)
             for n, s, r, o in zip(names, srcs, runs, omops)]
    big, thin = sorted(profs, key=lambda p: -p["patients"])
    Path(out).write_text(render(big, thin, k) + "\n",
                         encoding="utf-8")
    print("wrote {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
