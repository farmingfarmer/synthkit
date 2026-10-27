"""DID THE MULTI-VARIABLE RELATIONSHIPS SURVIVE? - the question the
other reports cannot answer.

WHY IT EXISTS. Every fidelity number shipped before this works by
NAMING a pattern first: a pair's rank correlation, a published curve,
a two-way product. That can only find what somebody thought to look
for, and the gate's `interaction surfaces` criterion asks a narrower
question still - did the PUBLISHED two-variable surfaces survive? An
engine that publishes no surfaces reads 0/N on that criterion, which
means NOT APPLICABLE and looks exactly like total failure.

The measured reality on real data is almost the opposite of what that
reading suggests: the latent engine holds the source's top two-way
interactions at their own magnitude and reproduces driver attribution
at 93/7 against a source 94/6, while the rules engine reads 0.000 on
every one of them. Neither fact is visible in any criterion.

THE HEADLINE HERE IS TRAIN-ON-SYNTHETIC-TEST-ON-REAL, because it is
the only measure that cannot miss an interaction nobody enumerated. A
model trained on the synthetic table and scored on REAL held-out rows
has to have recovered whatever structure it needed, at whatever
order, from whatever combination of columns. No naming required.

SHAP IS USED FOR WHAT SHAP IS FOR - ranking which features drive a
target, and whether the SAME ones drive it in the synthetic table.
Its interaction values are second-order only, so it is the wrong
instrument for "however many variables" and the right one for "which
variables, and in what order of importance".

THE TRAP THIS PAGE REFUSES TO FALL INTO. A retention ratio on a
target that was never predictable is noise dressed as a result: a
target the real data predicts at 0.04 and the synthetic predicts at
0.03 reports 75% retained and means nothing. Targets are therefore
gated on their REAL-data skill first, and the ones that fail that
gate are listed as unpredictable rather than scored.

  python scripts/relationship_report.py \
      --src real.csv --run RUNDIR [--group-by person_id] \
      [--targets N] [--name LABEL] -o relationships.html
"""
import json
import sys
from pathlib import Path

from synthkit.discover import _is_identifier

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
for _p in (str(_root), str(_root / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

INK = "#1d1d1f"
GRAY = "#6e6e73"
CARDINAL = "#8c1515"
RULE = "#d8d8dd"
WASH = "#f6f7f9"
WARM = "#fbf7f2"

# A target whose REAL-data skill is below this is not predictable
# from the other columns in the first place, so there is no
# relationship to retain and a retention ratio on it is noise.
PREDICTABLE_FLOOR = 0.10

# Retained skill at or above this is reported as holding; below it
# the relationship is reported as degraded. Stated wherever used.
HOLD_BAND = 0.80

# THE TRAP AT THE OTHER END OF THE SCALE. The floor above refuses a
# target nobody could predict. This refuses a target ANYBODY can: a
# column the real data predicts at 1.000 is an arithmetic identity,
# not a relationship - age_at_visit IS the visit year minus
# year_of_birth, and the generator COPIES it by design. It will read
# 100% retained on every run, forever, and it is evidence about the
# identity tier rather than about relationships. Scored naively it
# sits at the top of the page as the most reassuring row on it.
IDENTITY_CEILING = 0.99

# Driver agreement is a COSINE between the two tables' importance
# vectors, not a yes/no. Same band as retention, so one number means
# one thing across the page.
DRIVER_AGREE = 0.80

# The keys jointcheck.predictability actually emits. Named here
# because reading the wrong three blanked an entire section of a
# document that went out to a stakeholder.
K_REAL, K_SYN, K_AGREE = "skill_real", "skill_syn", "driver_agreement"


def _read(path):
    from synthkit import sets
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in df.columns:
        if sets.looks_like_set(df[c]):
            continue
        df[c] = df[c].astype(object).where(
            df[c].astype(str).str.strip() != "", np.nan)
    return df


def pick_targets(real, group_by, n=6):
    """Which columns are worth asking about.

    NOT chosen by hand. A column is only interesting here if it has
    multi-variable structure to lose, so candidates are ranked by how
    much they vary and how present they are - and the real test of
    whether the choice was right is the predictability gate below,
    which throws out anything the real data cannot predict anyway."""
    cand = []
    n_rows = len(real)
    for c in real.columns:
        if c == group_by:
            continue
        if _is_identifier(real[c], n_rows):
            continue
        v = pd.to_numeric(real[c], errors="coerce")
        pres = float(v.notna().mean())
        if pres < 0.5 or v.nunique() < 12:
            continue
        sd = float(v.std() or 0.0)
        if sd <= 0:
            continue
        cand.append((c, pres * v.nunique()))
    cand.sort(key=lambda t: -t[1])
    return [c for c, _ in cand[:n]]


def measure(src_path, run, group_by, n_targets, seed=0):
    from synthkit import jointcheck as JC
    real = _read(src_path)
    synth = _read(Path(run) / "generated.csv")
    targets = pick_targets(real, group_by, n_targets)
    out = {"targets": [], "three_way": None, "distinguish": None,
            "predict": None, "n_real": int(len(real)),
            "n_synth": int(len(synth))}
    for t in targets:
        try:
            u = JC.utility(real, synth, t, group_by, seed)
        except Exception as e:
            out["targets"].append({"target": t,
                                   "error": str(e)[:120]})
            continue
        trtr = u.get("train_real_test_real")
        tstr = u.get("train_synthetic_test_real")
        out["targets"].append({
            "target": t, "trtr": trtr, "tstr": tstr,
            "retained": u.get("retained"),
            "predictable": (trtr is not None
                            and trtr >= PREDICTABLE_FLOOR),
            "identity": (trtr is not None
                         and trtr >= IDENTITY_CEILING)})
    for key, fn in (("three_way",
                     lambda: JC.three_way(real, synth, seed=seed,
                                          group_by=group_by)),
                    ("distinguish",
                     lambda: JC.distinguishability(real, synth,
                                                   group_by, seed)),
                    ("predict",
                     lambda: JC.predictability(real, synth,
                                               group_by,
                                               seed=seed))):
        try:
            out[key] = fn()
        except Exception as e:
            out[key] = {"error": "{}: {}".format(
                type(e).__name__, e)}
    return out


CSS = """
*{box-sizing:border-box}
body{margin:0;background:#fff;color:#1d1d1f;
 font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",
 Helvetica,Arial,sans-serif;font-variant-numeric:tabular-nums}
.sheet{max-width:50rem;margin:0 auto;padding:4rem 2.5rem 6rem}
.eyebrow{font-size:.72rem;letter-spacing:.14em;
 text-transform:uppercase;color:#8c1515;font-weight:650;
 margin:0 0 .7rem}
h1{font-size:2rem;line-height:1.2;letter-spacing:-.02em;
 font-weight:680;margin:0 0 .9rem;text-wrap:balance}
.standfirst{font-size:1.05rem;color:#6e6e73;margin:0 0 2rem;
 max-width:36rem}
h2{font-size:1.24rem;font-weight:660;margin:3rem 0 .3rem;
 padding-top:1.4rem;border-top:1px solid #d8d8dd}
h2 .num{color:#8c1515;margin-right:.5rem}
h3{font-size:1rem;font-weight:660;margin:1.8rem 0 .4rem}
p,ul{margin:.8rem 0;max-width:36rem}
li{margin:.3rem 0}
code{font:.875em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
 background:#f6f7f9;padding:.12em .36em;border-radius:4px}
table{border-collapse:collapse;width:100%;margin:1.1rem 0 1.3rem;
 font-size:.94rem}
th,td{text-align:left;padding:.5rem .7rem;
 border-bottom:1px solid #d8d8dd}
thead th{color:#6e6e73;font-weight:620;font-size:.76rem;
 letter-spacing:.07em;text-transform:uppercase;
 border-bottom:1.5px solid #1d1d1f}
td.n,th.n{text-align:right}
tbody tr:last-child td{border-bottom:1.5px solid #1d1d1f}
.note{background:#f6f7f9;border-left:3px solid #6e6e73;
 padding:1rem 1.2rem;margin:1.3rem 0;font-size:.95rem}
.note p{margin:.45rem 0;max-width:none}
.note .lead{font-weight:650}
.caution{background:#fbf7f2;border-left-color:#8c1515}
.lede{font-size:1.1rem;font-weight:620;line-height:1.5;
 margin:1.1rem 0;max-width:36rem}
.bar{display:inline-block;height:.62rem;background:#8c1515;
 vertical-align:middle;border-radius:2px}
.barbg{display:inline-block;width:7rem;height:.62rem;
 background:#ececed;vertical-align:middle;border-radius:2px;
 margin-right:.45rem}
.foot{margin-top:3.5rem;padding-top:1.1rem;
 border-top:1px solid #d8d8dd;color:#6e6e73;font-size:.86rem}
@media print{.sheet{padding:0;max-width:none}h2{page-break-after:avoid}
 table,.note{page-break-inside:avoid}}
@media (max-width:40rem){.sheet{padding:2.2rem 1.2rem 4rem}
 h1{font-size:1.6rem}table{font-size:.86rem}th,td{padding:.4rem .4rem}}
"""


def _esc(t):
    return (str(t).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _bar(frac):
    frac = max(0.0, min(float(frac or 0.0), 1.0))
    return ('<span class="barbg"><span class="bar" '
            'style="width:{:.0f}%"></span></span>'.format(frac * 100))


def render(m, label):
    H = []
    A = H.append
    A("<!doctype html>")
    A('<html lang="en"><head><meta charset="utf-8">')
    A('<meta name="viewport" content="width=device-width,'
      'initial-scale=1">')
    A("<title>Did the relationships survive? &mdash; {}"
      "</title>".format(_esc(label)))
    A("<!-- generated by scripts/relationship_report.py -->")
    A("<style>{}</style>".format(CSS))
    A("</head><body>")
    A('<div class="sheet">')
    A('<p class="eyebrow">Synthetic data &middot; '
      "relationships</p>")
    A("<h1>Did the relationships between variables "
      "survive?</h1>")
    A('<p class="standfirst">Not one field at a time &mdash; '
      "whether the <em>connections</em> between fields came "
      "through. {} real rows against {} synthetic.</p>".format(
          "{:,}".format(m["n_real"]), "{:,}".format(m["n_synth"])))

    # ---------------- 1. TSTR ----------------
    A('<h2><span class="num">1</span>The test that cannot miss '
      "anything</h2>")
    A('<p class="lede">Train a model on the synthetic data. Score '
      "it on <em>real</em> rows it has never seen. Compare that "
      "against the same model trained on real data.</p>")
    A("<p>If a model learned from synthetic data performs on real "
      "patients, then whatever relationships it needed survived "
      "&mdash; however many fields were involved, and whether or "
      "not anybody thought to look for them. Every other measure "
      "has to name a pattern before it can check it. This one "
      "does not.</p>")
    ident = [t for t in m["targets"] if t.get("identity")]
    good = [t for t in m["targets"]
            if t.get("predictable") and not t.get("identity")]
    weak = [t for t in m["targets"]
            if not t.get("predictable") and "error" not in t]
    if good:
        A("<table><thead><tr><th>Predicting</th>"
          '<th class="n">From real data</th>'
          '<th class="n">From synthetic</th>'
          '<th class="n">Kept</th><th></th>'
          "</tr></thead><tbody>")
        for t in sorted(good, key=lambda x: -(x["retained"] or 0)):
            r = t["retained"] or 0.0
            A("<tr><td><code>{}</code></td>"
              '<td class="n">{:.3f}</td><td class="n">{:.3f}</td>'
              '<td class="n">{:.0%}</td><td>{} {}</td></tr>'.format(
                  _esc(t["target"]), t["trtr"], t["tstr"], r,
                  _bar(r),
                  "holds" if r >= HOLD_BAND else "degraded"))
        A("</tbody></table>")
        A("<p>Both columns are scored on the <strong>same "
          "held-out real rows</strong>, so they are directly "
          "comparable. &ldquo;Kept&rdquo; is the synthetic-trained "
          "score as a share of the real-trained one; at or above "
          "{:.0%} the relationship is reported as holding.</p>"
          .format(HOLD_BAND))
    else:
        A('<div class="note caution"><p>No target cleared the '
          "predictability gate below, so there is no retention to "
          "report. That is a statement about the source data, not "
          "about the synthetic copy.</p></div>")
    if weak:
        A('<div class="note">')
        A('<p class="lead">Fields the real data cannot predict '
          "either &mdash; excluded rather than scored.</p>")
        A("<p>A retention figure on an unpredictable field is noise "
          "dressed as a result: if the real data predicts something "
          "at 0.04 and the synthetic predicts it at 0.03, that is "
          "&ldquo;75% kept&rdquo; and it means nothing. These "
          "scored below {:.2f} on real data, so there was no "
          "relationship to carry over: {}.</p>".format(
              PREDICTABLE_FLOOR,
              ", ".join("<code>{}</code> ({:.3f})".format(
                  _esc(t["target"]), t["trtr"] or 0.0)
                  for t in weak[:8])))
        A("</div>")
    if ident:
        A('<div class="note">')
        A('<p class="lead">Fields that are pure arithmetic '
          "&mdash; reported separately, because a perfect score "
          "here would be misleading.</p>")
        A("<p>A patient&rsquo;s age is their visit year minus "
          "their birth year. That is not a pattern the generator "
          "had to learn; it is a sum, and the generator is built "
          "to carry it across exactly. The real data predicts "
          "these at or above {:.2f}, so they come back at nearly "
          "100% on every run and always will. Counting them "
          "alongside the rest would put the most reassuring "
          "numbers on this page in front of the reader while "
          "telling them nothing about whether the "
          "<em>relationships</em> survived: {}.</p>".format(
              IDENTITY_CEILING,
              ", ".join("<code>{}</code> ({:.3f} &rarr; {:.3f})"
                        .format(_esc(t["target"]), t["trtr"] or 0.0,
                                t["tstr"] or 0.0)
                        for t in ident[:8])))
        A("</div>")

    # ---------------- 2. drivers ----------------
    A('<h2><span class="num">2</span>Are the same things doing '
      "the driving?</h2>")
    pr = m.get("predict") or {}
    rows = pr.get("columns") if isinstance(pr, dict) else None
    if isinstance(pr, dict) and pr.get("error"):
        A('<div class="note"><p>This measure did not run: '
          "<code>{}</code>. It is reported as absent rather than "
          "left blank.</p></div>".format(_esc(pr["error"])))
    elif rows:
        A("<p>A field can be just as predictable in the synthetic "
          "data while being predicted by the <em>wrong things</em>. "
          "This checks whether the most important drivers are the "
          "same ones, using the same ranking method on both "
          "tables.</p>")
        A("<table><thead><tr><th>Field</th>"
          '<th class="n">Predictable, real</th>'
          '<th class="n">Predictable, synthetic</th>'
          '<th class="n">Driver overlap</th>'
          "<th></th></tr></thead><tbody>")
        for r in rows[:12]:
            agr = r.get(K_AGREE)
            A("<tr><td><code>{}</code></td>"
              '<td class="n">{}</td><td class="n">{}</td>'
              '<td class="n">{}</td><td>{}</td></tr>'.format(
                  _esc(r.get("column")),
                  _fmt(r.get(K_REAL)), _fmt(r.get(K_SYN)),
                  _fmt(agr),
                  "&mdash;" if agr is None else
                  ("same drivers" if float(agr) >= DRIVER_AGREE
                   else "<strong>different drivers</strong>")))
        A("</tbody></table>")

    # ---------------- 3. three-way ----------------
    A('<h2><span class="num">3</span>Combinations of three</h2>')
    tw = m.get("three_way") or {}
    if isinstance(tw, dict) and tw.get("error"):
        A('<div class="note"><p>This measure did not run: '
          "<code>{}</code>.</p></div>".format(_esc(tw["error"])))
    else:
        A("<p>Sometimes three fields only matter together &mdash; "
          "no pair of them explains the effect. This asks whether "
          "a three-way combination adds anything beyond all three "
          "of its pairs, and whether the synthetic data kept "
          "it.</p>")
        trios = tw.get("trios") or []
        if trios:
            A("<table><thead><tr><th>Three fields together</th>"
              '<th class="n">Effect in real data</th>'
              '<th class="n">In synthetic</th><th></th>'
              "</tr></thead><tbody>")
            for r in trios[:10]:
                g0 = r.get("source") or 0.0
                g1 = r.get("synthetic") or 0.0
                kept = (g1 / g0) if g0 else None
                A("<tr><td><code>{}</code></td>"
                  '<td class="n">{}</td><td class="n">{}</td>'
                  "<td>{}</td></tr>".format(
                      _esc(" + ".join(r.get("trio") or [])
                           + " \u2192 " + str(r.get("child", ""))),
                      _fmt(g0), _fmt(g1),
                      "&mdash;" if kept is None else
                      ("kept" if kept >= HOLD_BAND
                       else "<strong>lost</strong>")))
            A("</tbody></table>")
            A("<p>The two numbers are how much the three fields "
              "explain <em>together</em> that no pair of them "
              "explains on its own. A combination that matters in "
              "the real data and not in the synthetic copy is one "
              "the generator did not carry.</p>")
        else:
            A('<div class="note"><p>No three-field combination in '
              "the real data added anything beyond its pairs, so "
              "there was none to carry over. That is a statement "
              "about the source data.</p></div>")
    A('<div class="note caution">')
    A('<p class="lead">Nothing above three fields at a time is '
      "measured anywhere.</p>")
    A("<p>If four factors only matter in combination, no number on "
      "this page would tell you whether that survived. The "
      "train-on-synthetic test in section 1 is the only measure "
      "here that would notice indirectly &mdash; a model needing "
      "that structure would simply score worse &mdash; which is "
      "why it leads.</p>")
    A("</div>")

    # ---------------- 4. what gives it away ----------------
    A('<h2><span class="num">4</span>What gives the synthetic '
      "data away?</h2>")
    dg = m.get("distinguish") or {}
    if isinstance(dg, dict) and dg.get("error"):
        A('<div class="note"><p>This measure did not run: '
          "<code>{}</code>.</p></div>".format(_esc(dg["error"])))
    else:
        A("<p>A classifier is trained to tell real rows from "
          "synthetic ones. If it cannot, the two are "
          "interchangeable. If it can, the fields it relies on "
          "are the ones still carrying a tell &mdash; which is a "
          "to-do list, not a verdict.</p>")
        _auc = dg.get("auc")
        A('<p class="lede">The classifier scored '
          "<strong>{}</strong>, and its own verdict is "
          "<strong>{}</strong>. A score of 0.50 means it could not "
          "tell real from synthetic at all; 1.00 means it always "
          "could. {} of {} fields were individually "
          "indistinguishable.</p>".format(
              _fmt(_auc), _esc(str(dg.get("verdict", "&mdash;"))),
              dg.get("columns_indistinguishable", "&mdash;"),
              dg.get("columns_measured", "&mdash;")))
        tells = dg.get("tells") or []
        if tells:
            A("<table><thead><tr><th>Field the classifier "
              'leaned on</th><th class="n">How much</th>'
              "</tr></thead><tbody>")
            for item in tells[:12]:
                try:
                    cname, cval = item[0], item[1]
                except Exception:
                    cname, cval = str(item), None
                A("<tr><td><code>{}</code></td>"
                  '<td class="n">{}</td></tr>'.format(
                      _esc(str(cname)), _fmt(cval)))
            A("</tbody></table>")
            A("<p>These are the fields still carrying a tell. They "
              "are the to-do list &mdash; fix the top of this "
              "table and the classifier loses its easiest "
              "evidence.</p>")
        A('<div class="note">')
        A("<p>Read this one with care in both directions. Over "
          "many fields, differences far too small to matter "
          "anywhere compound into an easy win for the classifier, "
          "so a high score does not by itself mean the data is "
          "unusable. The <em>named fields</em> are the useful "
          "part.</p>")
        A("</div>")

    A('<div class="foot">Every figure here was computed from the '
      "real and synthetic tables at the time this page was "
      "written. None of it measures privacy &mdash; a generator "
      "that copied its training data would score perfectly on "
      "every number above, which is why privacy is tested "
      "separately and by attack.</div>")
    A("</div>")
    A("</body></html>")
    return "\n".join(H)


def _fmt(v):
    return "&mdash;" if v is None else "{:.3f}".format(float(v))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    src = run = out = ""
    group_by, n, label = "person_id", 6, "cohort"
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--src":
            src = argv[i + 1]; i += 2
        elif a == "--run":
            run = argv[i + 1]; i += 2
        elif a == "--group-by":
            group_by = argv[i + 1]; i += 2
        elif a == "--targets":
            n = int(argv[i + 1]); i += 2
        elif a == "--name":
            label = argv[i + 1]; i += 2
        elif a in ("-o", "--out"):
            out = argv[i + 1]; i += 2
        elif a in ("-h", "--help"):
            print((__doc__ or "").strip())
            return 0
        else:
            print("unknown argument {!r}".format(a))
            return 2
    if not (src and run and out):
        print("need --src, --run and -o")
        return 2
    if not Path(src).exists():
        print("no source file: {}".format(src))
        return 2
    if not (Path(run) / "generated.csv").exists():
        print("no generated.csv in {}".format(run))
        return 2
    print("relationship_report: --group-by {} --targets {}".format(
        group_by, n))
    m = measure(src, run, group_by, n)
    Path(out).write_text(render(m, label) + "\n", encoding="utf-8")
    side = Path(out).with_suffix(".json")
    side.write_text(json.dumps({"label": label, "source": src,
                                "run": str(run), "measured": m},
                               indent=1, default=str) + "\n",
                    encoding="utf-8")
    print("wrote {}".format(out))
    print("wrote {}".format(side))
    return 0


if __name__ == "__main__":
    sys.exit(main())
