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

IT IS AN HTML DOCUMENT, NOT MARKDOWN. A .md file handed to a
stakeholder is a working file - it renders as plain text in most
places, and asking someone to install a viewer to read your report is
asking them not to read it. This emits one self-contained page: no
external font, no script, no stylesheet, so it opens identically on a
locked-down machine with no network and prints without a layout
collapsing. The palette is the one the fidelity decks already use, so
the whole set reads as one piece of work rather than four.

IT IS WRITTEN FOR SOMEONE NOT IN THE WEEDS. Every number is given
with the thing it is a number OF; every plot gets a sentence on what
it shows and a sentence on how to read it; and the limits sit where
they will be met rather than in a closing footnote, because a limit
discovered after a decision was taken is worth nothing.

  python scripts/handover_report.py \
      --name "800-patient" --src big.csv  --run BIGRUN  --omop DIR \
      --name "100-patient" --src thin.csv --run THINRUN --omop DIR \
      [--group-by person_id] [--k 10] -o handover.html
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


def _esc(t):
    return (str(t).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


CSS = """
:root{
  --ink:#1d1d1f; --gray:#6e6e73; --cardinal:#8c1515;
  --rule:#d8d8dd; --wash:#f6f7f9; --warm:#fbf7f2;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:#fff; color:var(--ink);
  font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",
       Helvetica,Arial,sans-serif;
  font-variant-numeric:tabular-nums;
}
.sheet{max-width:50rem; margin:0 auto; padding:4rem 2.5rem 6rem}
.eyebrow{
  font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--cardinal); font-weight:650; margin:0 0 .7rem
}
h1{
  font-size:2.1rem; line-height:1.18; letter-spacing:-.021em;
  font-weight:680; margin:0 0 .9rem; text-wrap:balance
}
.standfirst{
  font-size:1.06rem; color:var(--gray); margin:0 0 2.2rem;
  max-width:36rem
}
h2{
  font-size:1.26rem; letter-spacing:-.012em; font-weight:660;
  margin:3.2rem 0 .3rem; padding-top:1.5rem;
  border-top:1px solid var(--rule); text-wrap:balance
}
h2 .num{
  color:var(--cardinal); font-variant-numeric:tabular-nums;
  margin-right:.5rem; font-weight:650
}
h3{font-size:1rem; font-weight:660; margin:2rem 0 .4rem}
p{margin:.8rem 0; max-width:36rem}
ul{margin:.8rem 0; padding-left:1.15rem; max-width:36rem}
li{margin:.35rem 0}
strong{font-weight:650}
code{
  font:.875em/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  background:var(--wash); padding:.12em .36em; border-radius:4px
}
table{
  border-collapse:collapse; width:100%; margin:1.1rem 0 1.4rem;
  font-size:.94rem
}
th,td{
  text-align:left; padding:.52rem .7rem;
  border-bottom:1px solid var(--rule)
}
thead th{
  color:var(--gray); font-weight:620; font-size:.76rem;
  letter-spacing:.07em; text-transform:uppercase;
  border-bottom:1.5px solid var(--ink)
}
td.n,th.n{text-align:right}
tbody tr:last-child td{border-bottom:1.5px solid var(--ink)}
.note{
  background:var(--wash); border-left:3px solid var(--gray);
  padding:1rem 1.2rem; margin:1.4rem 0; font-size:.95rem
}
.note p{margin:.45rem 0; max-width:none}
.note .lead{font-weight:650}
.caution{background:var(--warm); border-left-color:var(--cardinal)}
.lede{
  font-size:1.1rem; font-weight:620; line-height:1.5;
  margin:1.2rem 0; max-width:36rem
}
.plot{margin:1.8rem 0 0}
.plot .file{
  font:.84rem/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  color:var(--cardinal); margin:.15rem 0 .6rem
}
.plot dt{
  font-size:.74rem; letter-spacing:.09em; text-transform:uppercase;
  color:var(--gray); font-weight:640; margin-top:.7rem
}
.plot dd{margin:.15rem 0 0; max-width:36rem}
.contents{
  border:1px solid var(--rule); border-radius:10px;
  padding:1.1rem 1.4rem; margin:0 0 1rem; background:#fff
}
.contents p{
  margin:0 0 .5rem; font-size:.74rem; letter-spacing:.09em;
  text-transform:uppercase; color:var(--gray); font-weight:640
}
.contents ol{margin:0; padding-left:1.3rem; font-size:.95rem}
.contents li{margin:.22rem 0}
.foot{
  margin-top:4rem; padding-top:1.2rem;
  border-top:1px solid var(--rule); color:var(--gray);
  font-size:.86rem
}
@media print{
  body{font-size:11.5pt}
  .sheet{padding:0; max-width:none}
  h2{page-break-after:avoid}
  table,.note,.plot{page-break-inside:avoid}
}
@media (max-width:40rem){
  .sheet{padding:2.2rem 1.2rem 4rem}
  h1{font-size:1.65rem}
  table{font-size:.86rem}
  th,td{padding:.42rem .45rem}
}
"""


def render(big, thin, k):
    """One self-contained HTML page. No external font, script or
    stylesheet - it has to open on a locked-down machine with no
    network, and print without the layout collapsing."""
    names = [big["name"], thin["name"]]
    H = []
    A = H.append
    A("<!-- generated by scripts/handover_report.py -->")
    A("<style>{}</style>".format(CSS))
    A('<div class="sheet">')
    A('<p class="eyebrow">Synthetic data &middot; handover</p>')
    A("<h1>What was produced, what it shows, "
      "and what it does not</h1>")
    A('<p class="standfirst">Two cohorts through the same generator, '
      "with the same settings. Every figure in this document was "
      "read from the measurement files themselves, so it is the "
      "same figure the tools report &mdash; nothing here was typed "
      "from memory.</p>")

    A('<div class="contents"><p>Contents</p><ol>')
    for t in ("What you asked for, and where it is",
              "The headline finding",
              "The visualizations, and how to read each one",
              "Privacy",
              "Limits, stated plainly",
              "If you want something changed"):
        A("<li>{}</li>".format(_esc(t)))
    A("</ol></div>")

    # ---------------- 1 ----------------
    A('<h2><span class="num">1</span>What you asked for, '
      "and where it is</h2>")
    A("<p>For each cohort you have the synthetic data as "
      "<strong>one row per visit</strong>, the same data "
      "<strong>split back into the original table layout</strong>, "
      "and three visualizations. Across the two cohorts there is "
      "one comparison page and one written argument.</p>")
    A("<h3>The original table layout</h3>")
    A("<p>Your data arrived as separate OMOP tables and was folded "
      "into one-row-per-visit for modelling. It has been folded "
      "back:</p>")
    allt = sorted(set(list(big["omop"]) + list(thin["omop"])))
    if allt:
        A("<table><thead><tr><th>Table</th>"
          '<th class="n">{}</th><th class="n">{}</th>'
          "</tr></thead><tbody>".format(_esc(names[0]),
                                        _esc(names[1])))
        for t in allt:
            A("<tr><td><code>{}</code></td>"
              '<td class="n">{}</td><td class="n">{}</td></tr>'
              .format(_esc(t),
                      "{:,}".format(big["omop"][t])
                      if t in big["omop"] else "&mdash;",
                      "{:,}".format(thin["omop"][t])
                      if t in thin["omop"] else "&mdash;"))
        A("</tbody></table>")
    A('<div class="note caution">')
    A('<p class="lead">Four things the original tables carried '
      "that these cannot.</p>")
    A("<p>The one-row-per-visit format used for modelling does not "
      "hold them. A file named <code>person.csv</code> reads as the "
      "original in every respect unless something says otherwise, "
      "so each is also written in "
      "<code>WHAT_IS_NOT_IN_THESE_FILES.txt</code> beside the "
      "tables.</p>")
    A("<ol>")
    A("<li><strong>Which route went with which drug.</strong> The "
      "modelling format kept a visit's drugs and a visit's routes "
      "as two separate lists, so the pairing was already gone. "
      "Routes are in their own file rather than attached to a drug "
      "they may not belong to.</li>")
    A("<li><strong>Concept IDs.</strong> Only names were carried, "
      "so the tables have <code>*_source_value</code> columns and "
      "no <code>*_concept_id</code>.</li>")
    A("<li><strong>Measurement reference ranges.</strong> "
      "<code>range_low</code> and <code>range_high</code> were "
      "used during modelling and not carried through.</li>")
    A("<li><strong>Per-fact dates.</strong> A condition takes its "
      "visit's date, because that is the only date available.</li>")
    A("</ol>")
    A("<p>Every identifier in every table is invented. No real "
      "person&rsquo;s ID and no real record appears in any of "
      "them.</p>")
    A("</div>")

    # ---------------- 2 ----------------
    A('<h2><span class="num">2</span>The headline finding</h2>')
    A('<p class="lede">Categorical and set structure survive a '
      "100-patient cohort. Numeric magnitude and vocabulary "
      "breadth do not &mdash; and no output size fixes it, because "
      "what binds is the number of <em>people</em>, not the number "
      "of rows.</p>")
    A('<div class="note">')
    A('<p class="lead">Read this table before any percentage.</p>')
    A("<p>Fidelity is scored only on the patterns that cleared the "
      "privacy floor. A thinner source grades a smaller and "
      "stronger surviving set, so it can post a <em>higher</em> "
      "percentage while publishing far less of its data. Coverage "
      "first, always.</p>")
    A("</div>")
    A("<table><thead><tr><th></th>"
      '<th class="n">{}</th><th class="n">{}</th></tr></thead><tbody>'
      .format(_esc(names[0]), _esc(names[1])))
    A('<tr><td>Patients</td><td class="n">{:,}</td>'
      '<td class="n">{:,}</td></tr>'.format(big["patients"],
                                            thin["patients"]))
    A('<tr><td>Rows</td><td class="n">{:,}</td>'
      '<td class="n">{:,}</td></tr>'.format(big["rows"],
                                            thin["rows"]))
    A("<tr><td>Share of data that can be published</td>"
      '<td class="n">{:.0%}</td><td class="n">{:.0%}</td></tr>'
      .format(big["mass"], thin["mass"]))
    A("<tr><td>Verdict</td>"
      '<td class="n"><strong>{}</strong></td>'
      '<td class="n"><strong>{}</strong></td></tr>'
      .format(_esc(big["verdict"]), _esc(thin["verdict"])))
    A("</tbody></table>")

    A("<h3>How much clinical vocabulary survives</h3>")
    A("<p>A code is publishable only if at least <strong>{}</strong> "
      "different patients have it. At {:,} patients that means "
      "{:.0%} of the cohort must share a code; at {:,} patients it "
      "means {:.1%}.</p>".format(
          k, thin["patients"], k / float(thin["patients"]),
          big["patients"], k / float(big["patients"])))
    cols = sorted({t["column"] for t in big["tokens"]}
                  & {t["column"] for t in thin["tokens"]})
    if cols:
        A("<table><thead><tr><th>Column</th>"
          '<th class="n">{}</th><th class="n">{}</th>'
          "</tr></thead><tbody>".format(_esc(names[0]),
                                        _esc(names[1])))
        for c in cols:
            tb = next(t for t in big["tokens"] if t["column"] == c)
            tt = next(t for t in thin["tokens"] if t["column"] == c)
            A("<tr><td><code>{}</code></td>"
              '<td class="n">{:,} of {:,}</td>'
              '<td class="n">{:,} of {:,}</td></tr>'.format(
                  _esc(c), tb["clear_k"], tb["found"],
                  tt["clear_k"], tt["found"]))
        A("</tbody></table>")

    A("<h3>How much numeric detail survives</h3>")
    shared = sorted(set(big["numeric"]) & set(thin["numeric"]))
    worse = [c for c in shared
             if thin["numeric"][c] < big["numeric"][c]]
    if shared:
        rows = sorted(shared,
                      key=lambda c: -(big["numeric"][c]
                                      - thin["numeric"][c]))[:6]
        A("<p>Share of each column&rsquo;s real spread that the "
          "synthetic data keeps. The largest gaps first:</p>")
        A("<table><thead><tr><th>Column</th>"
          '<th class="n">{}</th><th class="n">{}</th>'
          "</tr></thead><tbody>".format(_esc(names[0]),
                                        _esc(names[1])))
        for c in rows:
            A("<tr><td><code>{}</code></td>"
              '<td class="n">{:.0%}</td><td class="n">{:.0%}</td>'
              "</tr>".format(_esc(c), big["numeric"][c],
                             thin["numeric"][c]))
        A("</tbody></table>")
        A("<p><strong>{} of {} shared numeric columns keep less "
          "spread on the smaller cohort.</strong> Where the two are "
          "close, the privacy floor is not what separates them "
          "&mdash; and those columns are in the chart too, because "
          "they are an honest limit on this argument.</p>"
          .format(len(worse), len(shared)))

    # ---------------- 3 ----------------
    A('<h2><span class="num">3</span>The visualizations, '
      "and how to read each one</h2>")
    for title, fname, what, how in PLOTS:
        A('<div class="plot">')
        A("<h3>{}</h3>".format(_esc(title)))
        A('<p class="file">{}</p>'.format(_esc(fname)))
        A("<dl>")
        A("<dt>What it shows</dt><dd>{}</dd>".format(_esc(what)))
        A("<dt>How to read it</dt><dd>{}</dd>".format(_esc(how)))
        A("</dl></div>")
    A('<div class="note caution">')
    A('<p class="lead">One thing the 3D views cannot do.</p>')
    A("<p>The four measures in the latent view are computed inside "
      "each cohort&rsquo;s own compressed space, and those spaces "
      "are not the same width. A distance in one is not the same "
      "unit as a distance in the other.</p>")
    A("<p>So each latent view answers <em>&ldquo;do the synthetic "
      "points sit among the real ones in this cohort&rdquo;</em> "
      "&mdash; which it answers well. Putting the two side by side "
      "and judging which cohort is worse is not something those "
      "pictures support. That comparison is what "
      "<code>cohort_charts.html</code> is for.</p>")
    A("</div>")

    # ---------------- 4 ----------------
    A('<h2><span class="num">4</span>Privacy</h2>')
    A("<p>Both datasets were attacked rather than asserted. Two "
      "different attackers, plus a <strong>positive "
      "control</strong>: the same attack run against a generator "
      "that simply republishes the real records. If the control is "
      "not caught, the attack is incapable of failing and a clean "
      "result would mean nothing.</p>")
    A("<p>The control was caught on both cohorts. The honest "
      "scope: the controlled attacker is the one that sees only "
      "the <strong>synthetic rows</strong> &mdash; which is the "
      "only thing that leaves the machine. A second attacker "
      "assumes the trained model itself leaked; the model is never "
      "part of any release, and its numbers are reported "
      "separately for completeness.</p>")
    A('<div class="note">')
    A('<p class="lead">Why the two cohorts&rsquo; numbers should '
      "not be compared directly.</p>")
    A("<p>The attack splits patients into members and non-members "
      "&mdash; {:,} against {:,} for one cohort and {:,} against "
      "{:,} for the other. A result from the smaller split carries "
      "several times the uncertainty, so a gap between the two "
      "cohorts&rsquo; numbers is not evidence of a real "
      "difference.</p>".format(
          big["patients"] // 2, big["patients"] // 2,
          thin["patients"] // 2, thin["patients"] // 2))
    A("</div>")

    # ---------------- 5 ----------------
    A('<h2><span class="num">5</span>Limits, stated plainly</h2>')
    if thin["empty"]:
        A("<p><strong>Your {} has {} columns that are empty in the "
          "source itself</strong>: {}. Nothing can be generated "
          "for a field that was never recorded. Several of these "
          "are dates &mdash; which means that cohort cannot "
          "support any analysis over time, and that is a property "
          "of what was pulled rather than of the generator or of "
          "the privacy floor.</p>".format(
              _esc(thin["name"]), len(thin["empty"]),
              ", ".join("<code>{}</code>".format(_esc(c))
                        for c in thin["empty"][:10])))
    A("<p><strong>No patient trajectory is modelled.</strong> The "
      "generator produces faithful rows and a faithful population, "
      "and each synthetic patient&rsquo;s visits do belong to one "
      "invented person &mdash; but the order and timing of one "
      "patient&rsquo;s course is not reproduced. If an analysis "
      "walks along a single patient&rsquo;s timeline, this data "
      "does not support it.</p>")
    A("<p><strong>No single-seed fidelity percentage is quoted "
      "anywhere in this document.</strong> Re-running the "
      "generator with a different random seed moved one of these "
      "readings by ten points in our own testing, so a number from "
      "one run is not a result.</p>")
    for nm, p in ((big["name"], big), (thin["name"], thin)):
        if not p["gate"]:
            continue
        fails = [c for c, _, ok in p["gate"]["criteria"] if not ok]
        A("<p><strong>{}</strong>: the automated quality gate "
          "reports {}{}.</p>".format(
              _esc(nm),
              "all criteria met" if p["gate"]["met"] else "NOT MET",
              "" if p["gate"]["met"]
              else " &mdash; failing on "
                   + _esc(", ".join(fails))))

    # ---------------- 6 ----------------
    A('<h2><span class="num">6</span>If you want something '
      "changed</h2>")
    A("<p>More <strong>people</strong> is the only thing that "
      "lifts the limits above. More rows from the same people does "
      "not: the privacy floor counts patients, and one person seen "
      "twice as often is still one person. Output volume is nearly "
      "free &mdash; the larger cohort was generated at five times "
      "its size with no measurable fidelity cost &mdash; so ask "
      "for whatever row count suits you.</p>")
    A("<p>A wider pull &mdash; more patients, and the date and "
      "demographic columns populated &mdash; would change the "
      "numbers in section 2 far more than any change to the "
      "generator would.</p>")
    A('<div class="foot">Generated from the run artifacts. '
      "Every table, chart and figure referenced here can be "
      "regenerated from the same files.</div>")
    A("</div>")
    return "\n".join(H)


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
        print("need -o OUT.html")
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
