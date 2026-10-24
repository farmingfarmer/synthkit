"""THE PATIENT-COUNT EFFECT, DRAWN - two cohorts, one engine.

WHY IT EXISTS. Nothing in this repository compares two SOURCES
visually. The deck's --compare puts two SCALES of one source side by
side; the duel puts two ENGINES on one source. The question "is this
cohort big enough" is about neither, and it had only a text answer.

THE CHART THAT CARRIES THE ARGUMENT is the TOKEN SURVIVAL CURVE: how
many tokens clear the floor as the floor RISES, one line per cohort.
It draws the mechanism instead of asserting it - the floor counts
PATIENTS, so a cohort with fewer people loses its vocabulary faster
as k climbs. A single k is one point and can always be argued with; a
curve shows the shape.

Beside it, PAIRED SPREAD RETENTION over every shared numeric column,
because one column's 61%-against-25% is an anecdote until you can see
whether it holds across the others - and if some columns retain
EQUALLY well, that is an honest limit on the argument and belongs on
the page rather than in a footnote.

WHAT IT REFUSES, for the same reason cohort_argument refuses them: a
column EMPTY in one source is an extraction difference and not the
floor's doing; within-patient persistence collapses on every cohort
because this engine is row-level; and no single-seed fidelity
percentage is quoted. Each is stated in a NOT-EVIDENCE block.

  python scripts/cohort_charts.py \
      --name "800-patient" --src big.csv  --run BIGRUN \
      --name "100-patient" --src thin.csv --run THINRUN \
      [--group-by person_id] [--floors 2,5,10,20,40] -o charts.html
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

INK = "#1d1d1f"
GRAY = "#6e6e73"
RED = "#8c1515"
RULE = "#d2d2d7"
WASH = "#f5f7fa"

DEFAULT_FLOORS = (2, 5, 10, 20, 40)


def _read(path):
    """The set-aware fold, for the fourth time in this repository -
    a blanket ""->NaN turns an EMPTY SET into missing, and an empty
    set is a fact about the visit. Every consumer that hand-rolled
    its own read has had this bug; `sets.looks_like_set` is the
    predicate the engine itself applies."""
    from synthkit import sets
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in df.columns:
        if sets.looks_like_set(df[c]):
            continue
        df[c] = df[c].astype(object).where(
            df[c].astype(str).str.strip() != "", np.nan)
    return df


def survival(src, group_by, floors):
    """Tokens clearing the floor, per set column, as the floor rises.

    THE FLOOR COUNTS PATIENTS, which is the whole mechanism: at 100
    patients a token needs 10% prevalence to survive k=10; at 800 it
    needs 1.25%. Computed through sets.vocabulary so this page and
    the generator cannot disagree about what is publishable."""
    from synthkit import sets
    groups = src[group_by].astype(str).to_numpy()
    out = {}
    for c in src.columns:
        if c == group_by or not sets.looks_like_set(src[c]):
            continue
        row = []
        for k in floors:
            v = sets.vocabulary(src[c], groups, k=k, cap=10 ** 7)
            row.append(len(v["tokens"]) if v else 0)
        out[c] = row
    return out


def retention(run):
    """Share of each numeric column's source spread the output kept.

    Read off the run's own fidelity record rather than recomputed, so
    the chart and the gate cannot disagree about the same column."""
    fj = Path(run) / "fidelity.json"
    if not fj.exists():
        return {}
    rec = json.loads(fj.read_text(encoding="utf-8"))
    out = {}
    for col in (rec.get("columns") or []):
        nm, s, g = (col.get("column"), col.get("sd_source"),
                    col.get("sd_generated"))
        if nm and s:
            out[nm] = {"src": float(s), "gen": float(g or 0.0),
                       "kept": float(g or 0.0) / float(s),
                       "cov_src": col.get("coverage_source"),
                       "lag_src": col.get("lag1_source"),
                       "lag_gen": col.get("lag1_generated")}
    return out


def empty_cols(src, group_by):
    return [c for c in src.columns
            if c != group_by and float(src[c].notna().mean()) == 0.0]


def profile(name, src_path, run, group_by, floors):
    src = _read(src_path)
    return {"name": name,
            "patients": int(src[group_by].nunique()),
            "rows": int(len(src)),
            "survival": survival(src, group_by, floors),
            "retention": retention(run),
            "empty": empty_cols(src, group_by)}


# ---------------------------------------------------------------
# drawing
# ---------------------------------------------------------------

def _curve_svg(floors, series, title):
    """Two lines, log-ish y, with every point labelled. Points are
    labelled because a curve read off an axis is an impression and
    the number is the claim."""
    W, H = 660, 300
    L, R, T, B = 64, 18, 34, 46
    mx = max([max(v) for _, v, _ in series] + [1])
    def X(i):
        return L + (W - L - R) * (i / float(max(len(floors) - 1, 1)))
    def Y(v):
        return H - B - (H - T - B) * (v / float(mx))
    p = ['<svg viewBox="0 0 {} {}" width="100%" '
         'role="img" aria-label="{}">'.format(W, H, title)]
    p.append('<rect x="0" y="0" width="{}" height="{}" '
             'fill="#fff"/>'.format(W, H))
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = Y(mx * frac)
        p.append('<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" '
                 'stroke="{}" stroke-width="1"/>'.format(
                     L, y, W - R, y, RULE))
        p.append('<text x="{}" y="{:.1f}" font-size="11" fill="{}" '
                 'text-anchor="end">{:,}</text>'.format(
                     L - 8, y + 4, GRAY, int(round(mx * frac))))
    for i, k in enumerate(floors):
        p.append('<text x="{:.1f}" y="{}" font-size="11" fill="{}" '
                 'text-anchor="middle">k={}</text>'.format(
                     X(i), H - B + 18, GRAY, k))
    for label, vals, color in series:
        pts = " ".join("{:.1f},{:.1f}".format(X(i), Y(v))
                       for i, v in enumerate(vals))
        p.append('<polyline points="{}" fill="none" stroke="{}" '
                 'stroke-width="2.5"/>'.format(pts, color))
        for i, v in enumerate(vals):
            p.append('<circle cx="{:.1f}" cy="{:.1f}" r="3.5" '
                     'fill="{}"/>'.format(X(i), Y(v), color))
            p.append('<text x="{:.1f}" y="{:.1f}" font-size="10" '
                     'fill="{}" text-anchor="middle">{:,}</text>'
                     .format(X(i), Y(v) - 9, color, v))
    p.append('<text x="{}" y="20" font-size="12" fill="{}">{}</text>'
             .format(L, INK, title))
    p.append('<text x="{}" y="{}" font-size="11" fill="{}">the '
             'privacy floor, in patients -&gt;</text>'.format(
                 L, H - 6, GRAY))
    p.append("</svg>")
    return "".join(p)


def _retain_svg(rows, names):
    """Paired bars, one pair per column, retention as a share of
    source spread. Sorted by the GAP so the columns that separate
    the cohorts are read first - and columns that retain equally
    are visible at the other end rather than cropped away."""
    if not rows:
        return "<p>no shared numeric column carries a spread in both runs</p>"
    rowh, W, L = 26, 660, 230
    H = 44 + rowh * len(rows)
    p = ['<svg viewBox="0 0 {} {}" width="100%" role="img" '
         'aria-label="spread retained per column">'.format(W, H)]
    p.append('<rect x="0" y="0" width="{}" height="{}" '
             'fill="#fff"/>'.format(W, H))
    bw = W - L - 70
    for frac in (0.25, 0.5, 0.75, 1.0):
        x = L + bw * frac
        p.append('<line x1="{:.1f}" y1="28" x2="{:.1f}" y2="{}" '
                 'stroke="{}" stroke-width="1"/>'.format(
                     x, x, H - 10, RULE))
        p.append('<text x="{:.1f}" y="22" font-size="10" fill="{}" '
                 'text-anchor="middle">{:.0%}</text>'.format(
                     x, GRAY, frac))
    for j, (col, a, b) in enumerate(rows):
        y = 34 + rowh * j
        p.append('<text x="0" y="{}" font-size="11" fill="{}">{}</text>'
                 .format(y + 11, INK, col[:34]))
        for idx, (val, color) in enumerate(((a, GRAY), (b, RED))):
            if val is None:
                continue
            yy = y + idx * 9
            p.append('<rect x="{}" y="{}" width="{:.1f}" height="7" '
                     'fill="{}"/>'.format(L, yy, bw * min(val, 1.0),
                                          color))
            p.append('<text x="{:.1f}" y="{}" font-size="9.5" '
                     'fill="{}">{:.0%}</text>'.format(
                         L + bw * min(val, 1.0) + 5, yy + 7,
                         color, val))
    p.append('<text x="0" y="14" font-size="11" fill="{}">'
             '<tspan fill="{}">{}</tspan> vs '
             '<tspan fill="{}">{}</tspan> - share of the source '
             'spread kept</text>'.format(INK, GRAY, names[0],
                                         RED, names[1]))
    p.append("</svg>")
    return "".join(p)


def build(pa, pb, floors):
    """pa is the LARGER cohort by patients - fixed by the caller so
    the gray/cardinal pairing means the same thing on every chart."""
    cols = sorted(set(pa["survival"]) & set(pb["survival"]))
    H = ['<!-- generated by scripts/cohort_charts.py -->',
         '<style>body{{font-family:-apple-system,BlinkMacSystemFont,'
         '"Segoe UI",Helvetica,Arial,sans-serif;color:{ink};'
         'background:#fff;margin:0;font-size:15px;line-height:1.55}}'
         '.wrap{{max-width:760px;margin:0 auto;padding:36px 28px}}'
         'h1{{font-size:23px;letter-spacing:-.02em;margin:0 0 4px}}'
         'h2{{font-size:15px;margin:30px 0 6px}}'
         '.sub{{color:{gray};font-size:13px;margin:0 0 22px}}'
         '.card{{border:1px solid {rule};border-radius:12px;'
         'padding:16px 18px;margin:14px 0}}'
         '.note{{background:{wash};border-radius:10px;padding:12px 16px;'
         'font-size:13.5px;color:{ink}}}'
         'table{{border-collapse:collapse;width:100%;font-size:13.5px}}'
         'td,th{{text-align:left;padding:5px 8px;border-bottom:1px '
         'solid {rule}}}'
         'th{{color:{gray};font-weight:600}}'
         '.num{{font-variant-numeric:tabular-nums;text-align:right}}'
         '</style>'.format(ink=INK, gray=GRAY, rule=RULE, wash=WASH),
         '<div class="wrap">',
         '<h1>The patient-count effect, drawn</h1>',
         '<p class="sub">Two real cohorts, one engine, identical '
         'settings. The only difference is the source.</p>']

    H.append('<div class="note">Read coverage before fidelity. '
             'Fidelity is scored only on the patterns that cleared '
             'the privacy floor, so a thinner source grades a '
             'smaller and stronger surviving set and can post a '
             '<b>higher</b> percentage while publishing far less of '
             'its data.</div>')

    H.append("<h2>How much each cohort holds</h2>")
    H.append('<div class="card"><table><tr><th></th>'
             '<th class="num">{}</th><th class="num">{}</th></tr>'
             .format(pa["name"], pb["name"]))
    for lab, key, fmt in (("patients", "patients", "{:,}"),
                          ("rows", "rows", "{:,}")):
        H.append('<tr><td>{}</td><td class="num">{}</td>'
                 '<td class="num">{}</td></tr>'.format(
                     lab, fmt.format(pa[key]), fmt.format(pb[key])))
    H.append("</table></div>")

    H.append("<h2>Token survival as the floor rises</h2>")
    H.append('<p class="sub">A token is publishable only when at '
             'least k distinct PATIENTS hold it. At 100 patients '
             'k=10 means 10% prevalence; at 800 it means 1.25%. '
             'That is the whole mechanism, and this is its '
             'shape.</p>')
    for c in cols:
        H.append('<div class="card">')
        H.append(_curve_svg(
            floors,
            [(pa["name"], pa["survival"][c], GRAY),
             (pb["name"], pb["survival"][c], RED)], c))
        H.append("</div>")

    H.append("<h2>Spread retained, per column</h2>")
    H.append('<p class="sub">One column separating the cohorts is '
             'an anecdote. Columns that retain EQUALLY well are '
             'shown too, because they are an honest limit on the '
             'argument.</p>')
    shared = sorted(set(pa["retention"]) & set(pb["retention"]))
    rows = [(c, pa["retention"][c]["kept"], pb["retention"][c]["kept"])
            for c in shared]
    rows.sort(key=lambda r: -(abs((r[1] or 0) - (r[2] or 0))))
    H.append('<div class="card">')
    H.append(_retain_svg(rows, (pa["name"], pb["name"])))
    H.append("</div>")
    if rows:
        worse = sum(1 for _, a, b in rows if b < a)
        H.append('<div class="note">{} of {} shared numeric '
                 'column(s) retain LESS spread on the smaller '
                 'cohort. Where the two are close, the floor is '
                 'not what separates them.</div>'
                 .format(worse, len(rows)))

    H.append("<h2>Not evidence about cohort size</h2>")
    H.append('<div class="note">')
    for p, other in ((pb, pa), (pa, pb)):
        only = [c for c in p["empty"] if c not in other["empty"]]
        if only:
            H.append('<p><b>{}</b> has {} column(s) empty in the '
                     'SOURCE that the other does not: {}. That is '
                     'what was pulled, not what the floor did.</p>'
                     .format(p["name"], len(only),
                             ", ".join(only[:10])))
    lag = [(c, pa["retention"][c], pb["retention"][c])
           for c in shared
           if pa["retention"][c].get("lag_src")
           and pb["retention"][c].get("lag_src")]
    if lag:
        c, ra, rb = lag[0]
        H.append('<p>Within-patient persistence falls on '
                 '<b>both</b> ({}: {} &rarr; {}; {}: {} &rarr; '
                 '{}). This engine is row-level and models no '
                 'patient trajectory, so it would lose persistence '
                 'on a million patients.</p>'.format(
                     pa["name"], ra["lag_src"], ra["lag_gen"],
                     pb["name"], rb["lag_src"], rb["lag_gen"]))
    H.append('<p>No single-seed fidelity percentage is quoted. A '
             'paired sweep in this project moved such a reading by '
             'ten points, so one seed is not a result.</p>')
    H.append("</div>")

    H.append('<h2>What would help</h2>')
    H.append('<div class="note">More <b>people</b>, not more rows. '
             'Duplicating every row changes no verdict - the floor '
             'counts patients, and one person seen twice as often '
             'is still one person.</div>')
    H.append("</div>")
    return "\n".join(H)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    group_by, out = "person_id", ""
    floors = list(DEFAULT_FLOORS)
    names, srcs, runs = [], [], []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--group-by":
            group_by = argv[i + 1]; i += 2
        elif a == "--floors":
            floors = [int(x) for x in argv[i + 1].split(",")]; i += 2
        elif a in ("-o", "--out"):
            out = argv[i + 1]; i += 2
        elif a == "--name":
            names.append(argv[i + 1]); i += 2
        elif a == "--src":
            srcs.append(argv[i + 1]); i += 2
        elif a == "--run":
            runs.append(argv[i + 1]); i += 2
        else:
            print("unknown argument {!r} - see --help".format(a))
            return 2
    if not (len(names) == len(srcs) == len(runs) == 2):
        print("want TWO cohorts, each with --name --src --run; got "
              "{} name(s), {} source(s), {} run(s)".format(
                  len(names), len(srcs), len(runs)))
        return 2
    if not out:
        print("need -o OUT.html")
        return 2
    for nm, sp, rp in zip(names, srcs, runs):
        if not Path(sp).exists():
            print("no source file for {}: {}".format(nm, sp))
            return 2
        if not (Path(rp) / "fidelity.json").exists():
            print("no fidelity.json in the run for {}: {}"
                  .format(nm, rp))
            return 2
    print("cohort_charts: --group-by {} --floors {}".format(
        group_by, ",".join(str(f) for f in floors)))
    profs = [profile(n, s, r, group_by, floors)
             for n, s, r in zip(names, srcs, runs)]
    pa, pb = sorted(profs, key=lambda p: -p["patients"])
    Path(out).write_text(build(pa, pb, floors) + "\n",
                         encoding="utf-8")
    print("wrote {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
