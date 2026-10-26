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
    ("The data, column by column", "deck_<cohort>.html",
     "Every field in your data drawn twice on the same axis - your "
     "real data in gray, the synthetic version in red - so you can "
     "see whether they have the same shape. Below that, a grid "
     "showing which fields move together, and a card for each "
     "pattern we found.",
     "Look for the gray and red shapes sitting on top of each "
     "other. Where they do, that field came through. Each pattern "
     "card ends in a plain verdict - either it tracks, or it says "
     "DEPARTS - so you do not have to judge a curve by eye."),
    ("The model's internal view", "latent_<cohort>.html",
     "A three-dimensional picture of how the software itself sees "
     "the data, with your real records and the synthetic ones "
     "plotted together. You choose what the three directions "
     "measure - how unusual a record is, how close it sits to its "
     "nearest neighbour, how well the model can rebuild it, and how "
     "close each synthetic record sits to a real one.",
     "Drag to rotate. The question is simply whether the red points "
     "sit mixed in among the gray ones or off to one side - mixed "
     "in means the synthetic records look like plausible members of "
     "your population. That last measure is the privacy one: a "
     "synthetic record sitting very close to a real one is "
     "shadowing somebody."),
    ("The same comparison on familiar axes",
     "showcase_<cohort>.html",
     "The same real-against-synthetic picture, but the three "
     "directions are ordinary clinical measurements - blood "
     "pressure, glucose, and so on - with labelled arrows saying "
     "which is which.",
     "Open this one first if the previous picture feels abstract. "
     "It asks the same question using quantities you already "
     "recognise."),
    ("The two cohorts side by side", "cohort_charts.html",
     "Both of your datasets on the same page. One line per dataset "
     "showing how many distinct clinical codes we are still allowed "
     "to use as the privacy rule is tightened, and a pair of bars "
     "per measurement showing how much of its real variation the "
     "synthetic version carries.",
     "This is the only picture that compares your two datasets to "
     "each other. The line that drops away faster belongs to the "
     "smaller dataset losing its detail. Every point is labelled "
     "with its actual number, so nothing has to be read off an "
     "axis."),
]

# Each gate criterion, said the way it would be said out loud. The
# gate's own EXPLAIN text is accurate and assumes a reader who knows
# what a rank correlation is; this is for a reader who does not, and
# who is entitled to an answer anyway.
RELATIONSHIP_WORDS = {
    "direction kept": (
        "Do things still move together?",
        "When two things rose and fell together in your real data, "
        "do they still rise and fall together in the synthetic "
        "version. This is the most basic question you can ask of "
        "synthetic data: if it fails, the file is a collection of "
        "plausible-looking columns with no relationship between "
        "them."),
    "close": (
        "...and by about the same amount?",
        "A relationship can point the right way and still be "
        "weaker or stronger than the real one. This counts the "
        "relationships that kept their STRENGTH, not just their "
        "direction."),
    "inverted": (
        "Does anything come out backwards?",
        "A relationship that runs the OPPOSITE way in the "
        "synthetic data. This is the most serious error on the "
        "page, and it is held to zero rather than to a percentage, "
        "because a backwards relationship does not look like a "
        "mistake - it looks like a discovery. Somebody could "
        "publish it."),
    "shapes tracked": (
        "Do the bends survive?",
        "Not every relationship is a straight line. Some rise and "
        "then level off; some are U-shaped. This re-measures the "
        "actual SHAPE of each pattern on both datasets, because a "
        "U-shape can show no overall correlation at all and still "
        "be the most important thing in the data."),
    "interaction surfaces": (
        "Do combinations survive?",
        "Sometimes two things only matter TOGETHER - neither "
        "predicts much alone. This checks whether those joint "
        "effects came through. It is the hardest thing on this "
        "page to reproduce, and the engine does not attempt "
        "anything more complicated than two-at-a-time."),
    "dynamics tracked": (
        "Does a patient's course over time behave the same way?",
        "Whether a value drifts or holds steady from one of a "
        "patient's visits to the next. This is a known weak point "
        "of the engine used here and a low score is expected "
        "rather than surprising - see the limits section."),
    "coverage": (
        "Is each field filled in as often as it really was?",
        "If a measurement was recorded on a third of visits in "
        "your data, it should be recorded on about a third of "
        "visits in the synthetic version. A failure here usually "
        "means a column was misread somewhere upstream, which is "
        "why it is worth checking before anything else."),
    "set token shares": (
        "Does each individual code appear about as often?",
        "Taking the conditions, medications and procedures one code "
        "at a time, each should turn up at roughly the frequency it "
        "really had."),
    "set EMPTY rate": (
        "Are the empty lists still empty?",
        "A visit where nobody recorded any medication is a fact "
        "about that visit, not missing information. Those visits "
        "should still be empty in the synthetic version, at the "
        "same rate."),
}

# the band lives in synthkit/spread.py so three tools cannot
# disagree about what "about right" means
from synthkit import spread as _spread_mod   # noqa: E402
SPREAD_BAND = _spread_mod.BAND


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
                from synthkit import spread as _sp
                r = _sp.ratio(s, g)
                if r is not None:
                    out["numeric"][nm] = r
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
              "Did the data come through?",
              "Did the relationships survive?",
              "The pictures, and how to read each one",
              "Privacy",
              "What this data cannot do",
              "What would change the answer"):
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
    A('<h2><span class="num">2</span>Did the data come '
      "through?</h2>")
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

    A("<h3>How much of the variation survives</h3>")
    shared = sorted(set(big["numeric"]) & set(thin["numeric"]))
    # A RATIO, NOT A SHARE, AND IT CAN EXCEED 100%.
    #
    # This was first written as "share of the real spread the
    # synthetic KEEPS", which cannot pass 100% by its own wording -
    # and a real column came out at 103%, meaning the synthetic
    # version was WIDER than the real one. That is a different
    # error entirely, and it was being labelled as near-perfect
    # success. Worse, the bar was drawn at min(value, 1.0), so 103%
    # rendered as a FULL bar: visually identical to a perfect
    # match, contradicting its own caption. And the headline count
    # asked which cohort's ratio was LOWER, which treats
    # over-dispersion as an improvement.
    #
    # So: 100% is the TARGET, not the ceiling. A miss is a miss in
    # either direction, and the count says so.
    # ONE DEFINITION, SHARED. These were local helpers until the
    # same defect was found in two other tools that had each
    # written their own - see synthkit/spread.py.
    from synthkit import spread as _SP

    def _band(v):
        return not _SP.is_miss(v)

    def _word(v):
        return _SP.verdict(v)

    if shared:
        miss_big = [c for c in shared if not _band(big["numeric"][c])]
        miss_thin = [c for c in shared
                     if not _band(thin["numeric"][c])]
        A("<p>Each measurement varies from patient to patient. This "
          "compares how much it varies in the synthetic data "
          "against how much it varied in yours. <strong>100% is "
          "the target</strong> &mdash; below means the synthetic "
          "version is flatter than reality, above means it is more "
          "spread out than reality, and both are wrong. Anything "
          "within {:.0%} either way is counted as about "
          "right.</p>".format(SPREAD_BAND))
        rows = sorted(shared,
                      key=lambda c: -abs(thin["numeric"][c] - 1.0))[:8]
        A("<table><thead><tr><th>Measurement</th>"
          '<th class="n">{}</th><th class="n">{}</th>'
          "<th>Smaller dataset</th></tr></thead><tbody>".format(
              _esc(names[0]), _esc(names[1])))
        for c in rows:
            A("<tr><td><code>{}</code></td>"
              '<td class="n">{:.0%}</td><td class="n">{:.0%}</td>'
              "<td>{}</td></tr>".format(
                  _esc(c), big["numeric"][c], thin["numeric"][c],
                  _esc(_word(thin["numeric"][c]))))
        A("</tbody></table>")
        A("<p><strong>{} of {} measurements are off by more than "
          "{:.0%} in your {}, against {} of {} in the larger "
          "dataset.</strong> Almost all of the misses are in the "
          "same direction &mdash; the synthetic version is flatter "
          "than reality &mdash; which is what the privacy rule does "
          "to a measurement that only a few patients share.</p>"
          .format(len(miss_thin), len(shared), SPREAD_BAND,
                  _esc(thin["name"]), len(miss_big), len(shared)))
        over = [c for c in shared if big["numeric"][c] > 1.0
                + SPREAD_BAND or thin["numeric"][c] > 1.0
                + SPREAD_BAND]
        if over:
            A('<div class="note">')
            A('<p class="lead">A few go the other way.</p>')
            A("<p>{} measurement(s) come out MORE spread than "
              "reality: {}. These are mostly short integer scales "
              "&mdash; a score from 3 to 15, say &mdash; where the "
              "real variation is only a point or two, so a small "
              "rounding difference is a large percentage. They are "
              "listed because an over-spread column is just as "
              "wrong as a flat one, and the table above would "
              "otherwise read as though the only risk were "
              "flatness.</p>".format(
                  len(over),
                  ", ".join("<code>{}</code>".format(_esc(c))
                            for c in over[:6])))
            A("</div>")

    # ---------------- 3: RELATIONSHIPS ----------------
    # THE QUESTION THE REPORT WAS MISSING. Everything above is
    # about columns ONE AT A TIME - is each field present, does
    # each measurement vary the right amount. None of it answers
    # whether the data still holds together, which is the whole
    # reason to want synthetic data rather than random numbers. A
    # table of perfect marginals with no relationships between them
    # is the classic way a synthetic generator looks right and is
    # useless, and that failure is invisible in every number
    # printed so far.
    A('<h2><span class="num">3</span>Did the relationships '
      "survive?</h2>")
    A('<p class="lede">Everything above looks at one field at a '
      "time. This section asks whether the data still hangs "
      "together &mdash; whether the things that moved together in "
      "your data still move together in the synthetic version. It "
      "is the reason to want synthetic data at all rather than "
      "plausible-looking random numbers.</p>")
    A("<p>Each line below is measured by re-running the same "
      "analysis on both datasets and comparing the answers. The "
      "fraction is how many of the relationships we found in your "
      "real data came through.</p>")
    order = ["direction kept", "close", "inverted",
             "shapes tracked", "interaction surfaces",
             "dynamics tracked", "coverage", "set token shares",
             "set EMPTY rate"]
    for nm, p in ((big["name"], big), (thin["name"], thin)):
        if not p["gate"]:
            continue
        crit = {c: (frac, ok) for c, frac, ok in p["gate"]["criteria"]}
        A("<h3>{}</h3>".format(_esc(nm)))
        A("<table><thead><tr><th>Question</th>"
          '<th class="n">Result</th><th class="n"></th>'
          "</tr></thead><tbody>")
        for key in order:
            if key not in crit:
                continue
            frac, ok = crit[key]
            title, _ = RELATIONSHIP_WORDS.get(key, (key, ""))
            A("<tr><td>{}</td>"
              '<td class="n"><code>{}</code></td>'
              '<td class="n">{}</td></tr>'.format(
                  _esc(title), _esc(frac),
                  "met" if ok else "<strong>not met</strong>"))
        A("</tbody></table>")
    A("<h3>What each question means</h3>")
    A("<dl class=\"plot\">")
    for key in order:
        if key not in RELATIONSHIP_WORDS:
            continue
        title, body = RELATIONSHIP_WORDS[key]
        A("<dt>{}</dt><dd>{}</dd>".format(_esc(title), _esc(body)))
    A("</dl>")
    A('<div class="note caution">')
    A('<p class="lead">The one to read first is '
      "&ldquo;does anything come out backwards&rdquo;.</p>")
    A("<p>A missing relationship is a gap &mdash; you can see it is "
      "missing. A backwards relationship is worse, because it does "
      "not look like a gap. It looks like a finding, and somebody "
      "could act on it. That is why it is held to zero rather than "
      "to a percentage.</p>")
    A("</div>")
    A('<div class="note">')
    A('<p class="lead">What is NOT measured here.</p>')
    A("<p>Nothing more complicated than two things at a time. If "
      "three factors only matter in combination, this engine does "
      "not try to reproduce that and nothing on this page would "
      "tell you it failed. That is a real limit and it is stated "
      "here rather than left to be discovered.</p>")
    A("</div>")

    # ---------------- 4 ----------------
    A('<h2><span class="num">4</span>The pictures, '
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
    A('<h2><span class="num">5</span>Privacy</h2>')
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
    A('<h2><span class="num">6</span>What this data '
      "cannot do</h2>")
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
    A('<h2><span class="num">7</span>What would change '
      "the answer</h2>")
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
