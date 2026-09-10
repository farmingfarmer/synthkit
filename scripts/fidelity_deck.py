"""The roadshow artifact: original vs synthetic, column by column.

    python scripts/fidelity_deck.py --src REAL.csv --run RUNDIR \\
        -o deck.html [--group-by person_id] \\
        [--compare "5x=OTHER_RUNDIR" ...]

One self-contained HTML - no dependencies, works from disk on a
locked-down machine. For every column the operator gave us: a
statistics table (original beside synthetic), an overlaid
distribution, and for the numeric set a pair of correlation heatmaps.
Ends with the gate, and with a scale-comparison table when --compare
runs are given (the "does 5x degrade it" question).

WHY IT EXISTS. Every number in it already existed - in fidelity.json,
findings.txt and the blueprint - and the person who asked for it is a
statistician who said, correctly, that nobody reads JSON. This is the
same information drawn the way the field draws it (original vs
synthetic profiling), in the house palette: near-black text, gray for
the original series, cardinal red for the synthetic one, white
everywhere else.

ONE THING NO VENDOR PAGE SHOWS: this report is itself k-screened.
A histogram of real data is a set of bin counts, and a bin holding
fewer than k patients describes a group small enough to gossip about.
Source bins backed by fewer than k patients are suppressed and the
suppression is printed. The synthetic side needs no screening -
nothing in it belongs to anyone.

Reads the source the way the pipeline does (blank stays "", never
NaN) and takes column kinds from the BLUEPRINT, not from re-typing -
the report describes the contract that was actually used.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from _console import console_safe
    console_safe()
except Exception:
    pass

INK = "#1d1d1f"          # near-black
GRAY = "#6e6e73"         # the ORIGINAL series
RED = "#8c1515"          # cardinal - the SYNTHETIC series
RULE = "#d2d2d7"
NEG = "#44546a"          # heatmap negative pole

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',
  Helvetica,Arial,sans-serif;color:%(ink)s;background:#fff;
  margin:0;font-size:14px;line-height:1.5}
.wrap{max-width:1060px;margin:0 auto;padding:48px 32px}
h1{font-size:28px;font-weight:700;letter-spacing:-.02em;margin:0}
h2{font-size:19px;font-weight:700;margin:40px 0 6px;
  padding-top:22px;border-top:1px solid %(rule)s}
h3{font-size:15px;font-weight:700;margin:18px 0 4px}
.sub{color:%(gray)s;margin:4px 0 0}
.legend{margin:10px 0 0;color:%(gray)s;font-size:12.5px}
.legend b{font-weight:700}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;
  vertical-align:-1px;margin-right:4px}
table{border-collapse:collapse;font-size:12.5px;margin:8px 0}
th{font-weight:600;text-align:left;color:%(gray)s;
  border-bottom:1px solid %(rule)s;padding:4px 14px 4px 0}
td{padding:3px 14px 3px 0;border-bottom:1px solid #f0f0f2;
  font-variant-numeric:tabular-nums}
td.num{text-align:right}
.card{display:grid;grid-template-columns:330px 1fr;gap:26px;
  align-items:start;padding:14px 0 6px}
.gate{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}
.chip{font-size:11.5px;font-weight:700;padding:4px 10px;
  border-radius:6px;letter-spacing:.02em}
.chip.ok{background:#f2f2f4;color:%(ink)s}
.chip.bad{background:%(red)s;color:#fff}
.note{color:%(gray)s;font-size:12px;margin:6px 0}
.small{font-size:11.5px;color:%(gray)s}
svg text{font-family:inherit}
.footer{margin-top:44px;padding-top:16px;border-top:1px solid
  %(rule)s;color:%(gray)s;font-size:12px}

/* ---- the charts are ALIVE ----
   Every chart reveals itself as it scrolls into view, and press-
   and-hold pulls the original and the synthetic apart so the eye
   can hold each shape alone - release, and they glide back into
   overlap, which IS the fidelity claim, performed. One long
   ease-out everywhere: nothing snaps, nothing bounces twice. */
svg.live{cursor:grab;touch-action:pan-y}
svg.live:active{cursor:grabbing}
.ser{transition:transform .85s cubic-bezier(.22,1,.36,1),
  opacity .85s cubic-bezier(.22,1,.36,1)}
svg.live:not(.on) .ser.src{opacity:0;transform:translateY(-10px)}
svg.live:not(.on) .ser.syn{opacity:0;transform:translateY(10px)}
svg.live:not(.on) .ser.pub{opacity:0}
svg.live.apart .ser.src{transform:translateY(-16px)}
svg.live.apart .ser.syn{transform:translateY(16px)}
svg.live.apart .ser.pub{opacity:.25}
svg.live.apart:not(.hm) > text,
svg.live.apart:not(.hm) > line{opacity:.15;
  transition:opacity .5s}
svg.live:not(.hm) > text,svg.live:not(.hm) > line{
  transition:opacity .5s}
svg.hm .hm-self,svg.hm .hm-other{
  transition:opacity .7s cubic-bezier(.22,1,.36,1)}
svg.hm .hm-other{opacity:0}
svg.hm.apart .hm-self{opacity:0}
svg.hm.apart .hm-other{opacity:1}
path.draw{transition:stroke-dashoffset 1.4s
  cubic-bezier(.4,0,.2,1)}
.hold-hint{color:%(gray)s;font-size:12px;margin:8px 0 0;
  font-style:italic}
.gissue{margin:14px 0;padding:12px 18px;background:#FDFAFA;
  border:1px solid #efd9d9;border-left:5px solid %(red)s;
  border-radius:0 12px 12px 0}
.gissue h3{margin:2px 0 4px;color:%(red)s}
@media (prefers-reduced-motion:reduce){
  .ser,path.draw{transition:none}
  svg.live:not(.on) .ser{opacity:1;transform:none}}
""" % {"ink": INK, "gray": GRAY, "red": RED, "rule": RULE}

# The deck's only script: reveal-on-scroll and press-and-hold.
# Inline and dependency-free, because this file must animate
# identically as a shared artifact on a machine with no network.
MOTION_JS = """<script>
(function(){
var mq=window.matchMedia&&window.matchMedia(
  '(prefers-reduced-motion: reduce)').matches;
var lives=[].slice.call(document.querySelectorAll('svg.live'));
/* curves draw themselves on: length-based dash, offset to zero */
lives.forEach(function(sv){
  [].slice.call(sv.querySelectorAll('path.draw')).forEach(
    function(pa){
      try{var L=pa.getTotalLength();
        pa.style.strokeDasharray=L;
        pa.style.strokeDashoffset=mq?0:L;}catch(e){}});});
function on(sv){
  sv.classList.add('on');
  [].slice.call(sv.querySelectorAll('path.draw')).forEach(
    function(pa){pa.style.strokeDashoffset=0;});}
if(mq||!window.IntersectionObserver){lives.forEach(on);}
else{var io=new IntersectionObserver(function(es){
    es.forEach(function(e){if(e.isIntersecting){
      on(e.target);io.unobserve(e.target);}});},
  {threshold:.25});
  lives.forEach(function(sv){io.observe(sv);});}
/* press-and-hold: apart while held; a quick click performs the
   separation once and lets it settle back on its own */
lives.forEach(function(sv){
  var t0=0,timer=null;
  sv.addEventListener('pointerdown',function(ev){
    ev.preventDefault();t0=Date.now();
    if(timer){clearTimeout(timer);timer=null;}
    sv.classList.add('apart');});
  function release(){
    if(!sv.classList.contains('apart'))return;
    var held=Date.now()-t0;
    if(held<250){timer=setTimeout(function(){
      sv.classList.remove('apart');},900);}
    else{sv.classList.remove('apart');}}
  sv.addEventListener('pointerup',release);
  sv.addEventListener('pointercancel',release);
  sv.addEventListener('pointerleave',release);});
})();
</script>"""


def esc(x):
    return html.escape(str(x))


def fnum(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    if isinstance(x, float):
        if abs(x) >= 1000:
            return "{:,.0f}".format(x)
        return "{:.{}f}".format(x, nd).rstrip("0").rstrip(".")
    return "{:,}".format(x) if isinstance(x, int) else str(x)


def read_frame(path):
    import pandas as pd
    return pd.read_csv(path, dtype=str, keep_default_na=False,
                       encoding="utf-8-sig", low_memory=False)


def numeric(series):
    import pandas as pd
    s = series.replace("", None)
    return pd.to_numeric(s, errors="coerce")


def stats_rows(vs, vg):
    import numpy as np

    def one(v):
        ok = v.dropna()
        if not len(ok):
            return {}
        a = ok.to_numpy(dtype=float)
        return {
            "rows": int(v.shape[0]),
            "missing": float(v.isna().mean()),
            "distinct": int(ok.nunique()),
            "mean": float(a.mean()), "sd": float(a.std()),
            "p5": float(np.percentile(a, 5)),
            "p25": float(np.percentile(a, 25)),
            "median": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95)),
            "skew": (float(((a - a.mean()) ** 3).mean()
                     / (a.std() ** 3)) if a.std() else 0.0),
        }
    s, g = one(vs), one(vg)
    order = [("rows", 0), ("missing", 3), ("distinct", 0),
             ("mean", 3), ("sd", 3), ("p5", 2), ("p25", 2),
             ("median", 2), ("p75", 2), ("p95", 2), ("skew", 2)]
    out = []
    for k, nd in order:
        a, b = s.get(k), g.get(k)
        if k == "missing":
            a = None if a is None else round(100 * a, 2)
            b = None if b is None else round(100 * b, 2)
            k = "missing %"
        out.append((k, fnum(a, nd), fnum(b, nd)))
    return out


def hist_svg(vs, vg, pat_counts, k, date_labels=None):
    """Overlaid histogram: original gray, synthetic cardinal.
    Source bins with fewer than k patients are dropped and counted."""
    import numpy as np
    a = vs.dropna().to_numpy(dtype=float)
    b = vg.dropna().to_numpy(dtype=float)
    if not len(a) and not len(b):
        return "", 0
    lo = min(a.min() if len(a) else b.min(),
             b.min() if len(b) else a.min())
    hi = max(a.max() if len(a) else b.max(),
             b.max() if len(b) else a.max())
    if hi <= lo:
        hi = lo + 1.0
    nb = 24
    edges = np.linspace(lo, hi, nb + 1)
    ca, _ = np.histogram(a, bins=edges)
    cb, _ = np.histogram(b, bins=edges)
    # k-screen the SOURCE bins by patients
    suppressed = 0
    if pat_counts is not None:
        for i in range(nb):
            if 0 < ca[i] and pat_counts[i] < k:
                ca[i] = 0
                suppressed += 1
    fa = ca / max(ca.sum(), 1)
    fb = cb / max(cb.sum(), 1)
    top = max(float(fa.max()), float(fb.max()), 1e-9)
    W, H, PAD = 620, 170, 26
    bw = (W - 2 * PAD) / nb
    # Each series lives in ONE <g>, so the whole distribution can
    # move as a body: press-and-hold pulls original and synthetic
    # apart, release lets them settle back into overlap. Interleaved
    # rects cannot do that.
    parts = ['<svg class="live" viewBox="0 0 {} {}" width="{}" '
             'height="{}">'.format(W, H + 34, W, H + 34)]
    parts.append('<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" '
                 'stroke="{3}"/>'.format(PAD, H, W - PAD, RULE))
    src_r, syn_r = [], []
    for i in range(nb):
        x = PAD + i * bw
        ha = fa[i] / top * (H - 14)
        hb = fb[i] / top * (H - 14)
        if ha > 0:
            src_r.append(
                '<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" '
                'height="{:.1f}" fill="{}"/>'.format(
                    x + 1, H - ha, bw - 2, ha, GRAY))
        if hb > 0:
            syn_r.append(
                '<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" '
                'height="{:.1f}" fill="{}" fill-opacity="0.55"/>'
                .format(x + 1, H - hb, bw - 2, hb, RED))
    parts.append('<g class="ser src">' + "".join(src_r) + "</g>")
    parts.append('<g class="ser syn">' + "".join(syn_r) + "</g>")
    for t, frac, anch in ((lo, 0.0, "start"),
                          ((lo + hi) / 2, 0.5, "middle"),
                          (hi, 1.0, "end")):
        label = (date_labels(t) if date_labels else fnum(float(t)))
        parts.append(
            '<text x="{:.1f}" y="{}" font-size="10.5" fill="{}" '
            'text-anchor="{}">{}</text>'.format(
                PAD + frac * (W - 2 * PAD), H + 16, GRAY, anch,
                esc(label)))
    parts.append("</svg>")
    return "".join(parts), suppressed


def bar_pairs_svg(rows):
    """Horizontal paired bars: (label, orig_share, syn_share)."""
    W, RH = 620, 21
    H = RH * len(rows) + 8
    top = max([max(o, s) for _, o, s in rows] + [1e-9])
    parts = ['<svg class="live" viewBox="0 0 {} {}" width="{}" '
             'height="{}">'.format(W, H, W, H)]
    LAB = 170
    src_r, syn_r = [], []
    for i, (label, o, s) in enumerate(rows):
        y = i * RH + 4
        parts.append('<text x="{}" y="{}" font-size="11" fill="{}" '
                     'text-anchor="end">{}</text>'.format(
                         LAB - 8, y + 12, INK, esc(label[:24])))
        wo = o / top * (W - LAB - 96)
        ws = s / top * (W - LAB - 96)
        src_r.append('<rect x="{}" y="{}" width="{:.1f}" height="6" '
                     'fill="{}"/>'.format(LAB, y + 2, wo, GRAY))
        syn_r.append('<rect x="{}" y="{}" width="{:.1f}" height="6" '
                     'fill="{}"/>'.format(LAB, y + 9, ws, RED))
        parts.append('<text x="{:.1f}" y="{}" font-size="10" '
                     'fill="{}">{:.1%} / {:.1%}</text>'.format(
                         LAB + max(wo, ws) + 6, y + 12, GRAY, o, s))
    parts.append('<g class="ser src">' + "".join(src_r) + "</g>")
    parts.append('<g class="ser syn">' + "".join(syn_r) + "</g>")
    parts.append("</svg>")
    return "".join(parts)


def heatmap_pair(names, ms, mg):
    """Two Spearman heatmaps side by side, shared scale."""
    n = len(names)
    cell = max(9, min(20, int(360 / max(n, 1))))
    lab = 110
    w1 = lab + n * cell + 8

    def color(v):
        if v is None or not math.isfinite(v):
            return "#ffffff"
        t = max(-1.0, min(1.0, v))
        if t >= 0:
            # white -> cardinal
            r1, g1, b1 = 0x8c, 0x15, 0x15
        else:
            r1, g1, b1 = 0x44, 0x54, 0x6a
            t = -t
        r = int(255 + (r1 - 255) * t)
        g = int(255 + (g1 - 255) * t)
        b = int(255 + (b1 - 255) * t)
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    def cells(m, ox):
        out = []
        for i in range(n):
            for j in range(n):
                out.append(
                    '<rect x="{}" y="{}" width="{}" height="{}" '
                    'fill="{}"/>'.format(
                        ox + lab + j * cell, 22 + i * cell,
                        cell - 1, cell - 1, color(m[i][j])))
        return "".join(out)

    def one(m, other, title, other_title, ox):
        # Press-and-hold crossfades the OTHER matrix over this one,
        # in place, so any pair that differs blinks into view. The
        # title rides inside each layer - during the fade the half
        # must be labeled by what it is SHOWING, not by what it
        # usually shows.
        t = ('<text x="{}" y="12" font-size="12" font-weight="700" '
             'fill="{}">{}</text>')
        parts = []
        for i, nm in enumerate(names):
            parts.append(
                '<text x="{}" y="{}" font-size="9" fill="{}" '
                'text-anchor="end">{}</text>'.format(
                    ox + lab - 4, 22 + i * cell + cell * 0.7, GRAY,
                    esc(nm[:16])))
        parts.append('<g class="hm-self">'
                     + t.format(ox + lab, INK, esc(title))
                     + cells(m, ox) + '</g>')
        parts.append('<g class="hm-other">'
                     + t.format(ox + lab, INK, esc(other_title))
                     + cells(other, ox) + '</g>')
        return "".join(parts)

    H = 30 + n * cell
    return ('<svg class="live hm" viewBox="0 0 {w} {h}" '
            'width="{w}" height="{h}">'
            .format(w=2 * w1 + 30, h=H)
            + one(ms, mg, "Original", "Synthetic", 0)
            + one(mg, ms, "Synthetic", "Original", w1 + 30)
            + "</svg>")


def shap_candidates(claims, src, gen):
    """Which claims can carry SHAP attribution: a non-scaffolding
    child present in both written files, with at least TWO drivers
    that are also present, numeric and non-scaffolding.

    On the real extract the top patterns are token-indicator pairs
    whose columns live only inside the search, plus date-driven
    children - so the first version, which screened only the top-8
    drawn cards, found nothing and rendered NOTHING, the exact
    silent absence the omission note exists to prevent. This walks
    EVERY claim by skill instead, and the caller states the empty
    case out loud."""
    from synthkit import sets as _sets
    out, seen = [], set()
    for cl in sorted(claims or [],
                     key=lambda c: -(c.get("skill") or 0)):
        child = cl.get("child")
        if (not child or child in seen
                or _sets.is_scaffolding(child)
                or child not in src.columns
                or child not in gen.columns):
            continue
        if numeric(src[child]).notna().mean() < 0.5:
            continue
        parents = []
        for pr in (cl.get("predictors") or []):
            c = pr.get("column")
            if (not c or c == child or c in parents
                    or _sets.is_scaffolding(c)
                    or c not in src.columns
                    or c not in gen.columns):
                continue
            if numeric(src[c]).notna().mean() < 0.5:
                continue
            parents.append(c)
        if len(parents) >= 2:
            seen.add(child)
            out.append((float(cl.get("skill") or 0), child,
                        parents[:6]))
    return out


def gate_issues_html(fid, verdict, bp=None):
    """Every failed gate criterion, highlighted and ENUMERATED.

    The gate chips said FAIL and stopped; the operator asked,
    rightly, where the misses were and how large. For the
    relationship criteria the offenders are named one by one with
    their magnitudes, because fidelity.json carries every pair; the
    column and set criteria point at the exact charts below that
    draw them. Returns "" when the gate is met - highlighting under
    a green gate is noise."""
    failed = [c for c in (verdict.get("criteria") or [])
              if not c.get("ok")]
    if not failed:
        return ""
    rel = fid.get("relationships") or {}
    rows = [r for r in (rel.get("pairs") or [])
            if r.get("kind") == "numeric"
            and abs(r.get("source") or 0) >= 0.1]

    # WHICH COLUMNS LOSE MAGNITUDE TO THE k BOUND - so a drifted
    # pair on such a column is MARKED as partly the privacy rule
    # working by design, not a fault to chase. The operator asked
    # for exactly this: the k-rule information beside the drift,
    # not a note telling them to go look for it.
    kclip = {}
    for cname, spec in ((bp or {}).get("columns") or {}).items():
        m = (spec or {}).get("marginal") or {}
        v = m.get("share_of_magnitude_outside_bounds")
        if v and v > 0.1:
            kclip[cname] = float(v)

    def krule(r):
        hits = [(c, kclip[c]) for c in (r["child"], r["parent"])
                if c in kclip]
        if not hits:
            return ""
        return " &middot; ".join(
            "{} loses {:.0%} of its magnitude to the k bound - "
            "by design".format(esc(c), v) for c, v in hits)

    def pair_table(rs, note_fn):
        out = ['<table><tr><th>relationship</th><th>original</th>'
               '<th>synthetic</th><th>drift</th><th></th>'
               '<th>k-rule?</th></tr>']
        for r in rs:
            out.append(
                '<tr><td>{} &larr; {}</td><td class="num">{}</td>'
                '<td class="num">{}</td><td class="num">{:+.2f}</td>'
                '<td>{}</td><td>{}</td></tr>'.format(
                    esc(r["child"]), esc(r["parent"]),
                    fnum(r["source"]), fnum(r["generated"]),
                    r["delta"], note_fn(r), krule(r)))
        out.append("</table>")
        return "".join(out)

    parts = ['<h2>The gate, enumerated &mdash; what missed, '
             'and by how much</h2>']
    for c in failed:
        name = c["name"]
        parts.append('<div class="gissue"><h3>FAIL &middot; {} '
                     '&mdash; {}</h3>'.format(
                         esc(name), esc(c.get("detail") or "")))
        if name == "close":
            bad = sorted([r for r in rows
                          if abs(r["delta"]) > 0.2],
                         key=lambda r: -abs(r["delta"]))
            parts.append(
                '<p class="note">Every drifted pair, largest '
                'first. These are exactly the cells that blink '
                'when you press and hold the correlation heatmaps '
                'below; the strongest also have pattern cards '
                'drawing the drift as two curves.</p>')
            parts.append(pair_table(
                bad, lambda r: ("direction lost"
                                if r["source"] * r["generated"] <= 0
                                else ("faded" if abs(r["generated"])
                                      < abs(r["source"])
                                      else "overshot"))))
        elif name == "direction kept":
            bad = sorted([r for r in rows
                          if r["source"] * r["generated"] <= 0],
                         key=lambda r: -abs(r["source"]))
            parts.append(
                '<p class="note">Pairs whose direction did not '
                'survive, strongest source relationship first.</p>')
            parts.append(pair_table(bad, lambda r: "sign lost"))
        elif name == "inverted":
            parts.append(
                '<p class="note">STOP: these generated '
                'relationships run OPPOSITE to the source and '
                'would read as findings. Do not circulate this '
                'file.</p>')
            parts.append(pair_table(
                rel.get("inverted") or [], lambda r: "INVERTED"))
        elif name == "shapes tracked":
            bad = sorted([r for r in (fid.get("shapes") or {})
                          .get("claims") or []
                          if not r.get("tracks")],
                         key=lambda r: -(r.get("gap_sd") or 0))
            parts.append(
                '<p class="note">Curves whose SHAPE departed by '
                'more than 0.35 of the child\'s own spread - '
                'drift the rank correlation cannot see. The '
                'pattern cards below draw the strongest of these '
                'as gray-vs-cardinal curves.</p>')
            parts.append(
                '<table><tr><th>relationship</th><th>shape</th>'
                '<th>gap (sd)</th></tr>'
                + "".join(
                    '<tr><td>{} &larr; {}</td><td>{}</td>'
                    '<td class="num">{:.2f}</td></tr>'.format(
                        esc(r["child"]), esc(r["parent"]),
                        esc(r.get("shape") or ""), r["gap_sd"])
                    for r in bad) + "</table>")
        elif name == "interaction surfaces":
            bad = sorted([r for r in (fid.get("surfaces") or {})
                          .get("surfaces") or []
                          if not r.get("tracks")],
                         key=lambda r: -(r.get("gap_sd") or 0))
            parts.append(
                '<p class="note">Two-variable interactions whose '
                'joint response drifted by more than 0.35 sd over '
                'the cells both tables can measure.</p>')
            parts.append(
                '<table><tr><th>child</th><th>pair</th>'
                '<th>cells</th><th>gap (sd)</th></tr>'
                + "".join(
                    '<tr><td>{}</td><td>{} &times; {}</td>'
                    '<td class="num">{}</td>'
                    '<td class="num">{:.2f}</td></tr>'.format(
                        esc(r["child"]), esc(r["pair"][0]),
                        esc(r["pair"][1]), r["cells_compared"],
                        r["gap_sd"])
                    for r in bad) + "</table>")
        elif name == "coverage":
            parts.append(
                '<p class="note">A column is present at the wrong '
                'rate. Scan the per-column stats tables below: the '
                'missing % row shows original beside synthetic, '
                'and the offenders are the rows where the two '
                'disagree by more than 5 points. A miss here '
                'usually means a column was read as the wrong '
                'type.</p>')
        elif name == "set token shares":
            parts.append(
                '<p class="note">A published token generates at '
                'the wrong frequency. The paired bars on each set '
                'column below draw original beside synthetic share '
                'per token - the offenders are the visibly '
                'unequal bar pairs.</p>')
        elif name == "set EMPTY rate":
            parts.append(
                '<p class="note">Rows that hold an empty list in '
                'the source are not empty at the same rate in the '
                'synthetic file - the set column sections below '
                'state both rates side by side.</p>')
        parts.append("</div>")
    return "".join(parts)


# THE CURVE MEASUREMENT IS SHARED. synthkit.curvecheck is the one
# implementation; the pipeline records the gaps for the gate and
# this deck draws from the identical functions - a second copy here
# is how the two would come to disagree.
from synthkit.curvecheck import cond_curve, shape_gap  # noqa: E402


def curve_svg(pub, src_pts, gen_pts):
    """Published curve (dashed ink), source (gray), synthetic
    (cardinal) - three tellings of one relationship."""
    xs = ([p[0] for p in pub] + [p[0] for p in src_pts]
          + [p[0] for p in gen_pts])
    ys = ([p[1] for p in pub] + [p[1] for p in src_pts]
          + [p[1] for p in gen_pts])
    if not xs:
        return ""
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 <= x0:
        x1 = x0 + 1
    if y1 <= y0:
        y1 = y0 + 1
    W, H, PAD = 620, 190, 34

    def X(v):
        return PAD + (v - x0) / (x1 - x0) * (W - 2 * PAD)

    def Y(v):
        return H - 18 - (v - y0) / (y1 - y0) * (H - 40)

    def path(pts):
        return "M" + " L".join("{:.1f} {:.1f}".format(X(a), Y(b))
                               for a, b in pts)
    parts = ['<svg class="live" viewBox="0 0 {} {}" width="{}" '
             'height="{}">'.format(W, H + 6, W, H + 6)]
    parts.append('<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" '
                 'stroke="{3}"/>'.format(PAD, H - 18, W - PAD, RULE))
    if pub:
        parts.append('<g class="ser pub"><path d="{}" fill="none" '
                     'stroke="{}" stroke-width="1.3" '
                     'stroke-dasharray="5 4" opacity="0.75"/></g>'
                     .format(path(pub), INK))
    if src_pts:
        g = ['<path class="draw" d="{}" fill="none" stroke="{}" '
             'stroke-width="2.2"/>'.format(path(src_pts), GRAY)]
        for a, b in src_pts:
            g.append('<circle cx="{:.1f}" cy="{:.1f}" r="2.6" '
                     'fill="{}"/>'.format(X(a), Y(b), GRAY))
        parts.append('<g class="ser src">' + "".join(g) + "</g>")
    if gen_pts:
        g = ['<path class="draw" d="{}" fill="none" stroke="{}" '
             'stroke-width="2.2" opacity="0.85"/>'.format(
                 path(gen_pts), RED)]
        for a, b in gen_pts:
            g.append('<circle cx="{:.1f}" cy="{:.1f}" r="2.6" '
                     'fill="{}" opacity="0.85"/>'.format(
                         X(a), Y(b), RED))
        parts.append('<g class="ser syn">' + "".join(g) + "</g>")
    for v, anch in ((x0, "start"), (x1, "end")):
        parts.append('<text x="{:.1f}" y="{}" font-size="10.5" '
                     'fill="{}" text-anchor="{}">{}</text>'.format(
                         X(v), H + 2, GRAY, anch, fnum(float(v))))
    for v in (y0, y1):
        parts.append('<text x="{}" y="{:.1f}" font-size="10.5" '
                     'fill="{}">{}</text>'.format(
                         2, Y(v) + 4, GRAY, fnum(float(v))))
    parts.append("</svg>")
    return "".join(parts)


def spearman_matrix(df, cols):
    import numpy as np
    vals = {c: numeric(df[c]) for c in cols}
    n = len(cols)
    m = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if j < i:
                m[i][j] = m[j][i]
                continue
            a, b = vals[cols[i]], vals[cols[j]]
            ok = a.notna() & b.notna()
            if int(ok.sum()) < 30:
                m[i][j] = None
                continue
            ar = a[ok].rank()
            br = b[ok].rank()
            with np.errstate(all="ignore"):
                r = float(np.corrcoef(ar, br)[0, 1])
            m[i][j] = r if math.isfinite(r) else None
    return m


class _A(object):
    """A tiny args stand-in so build_deck can be called in-process
    (the bench) with the fields argparse fills at the CLI."""

    def __init__(self, src, run, group_by, compare):
        self.src = src
        self.run = run
        self.group_by = group_by
        self.compare = list(compare or [])


def build_deck(src_path, run_dir, group_by="person_id",
               compares=None):
    """Return the deck HTML as a string. The CLI writes it to a
    file; the bench serves it into an iframe - one code path, so the
    picture the roadshow ships and the picture the UI shows cannot
    drift. Raises ValueError with a plain message rather than
    exiting, because inside the bench server sys.exit would take the
    whole process down."""
    a = _A(src_path, run_dir, group_by, compares)

    import numpy as np
    import pandas as pd
    from synthkit import gate as _gate
    from synthkit import sets as _sets

    run = Path(a.run)
    bp = json.loads((run / "blueprint.json").read_text(
        encoding="utf-8"))
    fid = json.loads((run / "fidelity.json").read_text(
        encoding="utf-8"))
    src = read_frame(a.src)
    gen = read_frame(run / "generated.csv")
    k = 10
    for spec in (bp.get("columns") or {}).values():
        m = (spec or {}).get("marginal") or {}
        k = int(m.get("bounds_are_k_anonymous") or k)
        break

    gid = a.group_by if a.group_by in src.columns else None
    prov = {}
    if (run / "provenance.json").exists():
        prov = json.loads((run / "provenance.json").read_text(
            encoding="utf-8"))
    build = prov.get("build")
    build = (build.get("id") if isinstance(build, dict) else build) \
        or "unknown"

    cols = []
    for c, spec in (bp.get("columns") or {}).items():
        if _sets.is_scaffolding(c) or c == gid:
            continue
        m = (spec or {}).get("marginal") or {}
        if m.get("type") in ("quantiles", "levels", "list") \
                and c in src.columns and c in gen.columns:
            cols.append((c, m))

    body = []
    npat_s = src[gid].nunique() if gid else len(src)
    npat_g = gen[gid].nunique() if gid and gid in gen.columns \
        else len(gen)
    body.append('<h1>Original vs synthetic</h1>')
    body.append('<p class="sub">{} &middot; {} rows / {} patients '
                'original &middot; {} rows / {} patients synthetic '
                '&middot; build {}</p>'.format(
                    esc(Path(a.src).name), fnum(len(src)),
                    fnum(npat_s), fnum(len(gen)), fnum(npat_g),
                    esc(build)))
    body.append('<p class="legend"><span class="sw" style='
                '"background:{}"></span><b>original</b> &nbsp; '
                '<span class="sw" style="background:{}"></span>'
                '<b>synthetic</b> &nbsp;&mdash;&nbsp; every '
                'published extreme is an average over at least {} '
                'patients, and source histogram bins backed by '
                'fewer than {} patients are suppressed from this '
                'report.</p>'.format(GRAY, RED, k, k))
    body.append('<p class="hold-hint">Press and hold any chart: '
                'the original and the synthetic pull apart so you '
                'can read each shape alone - release, and watch '
                'them settle back into overlap. On the correlation '
                'heatmaps, holding crossfades each side into the '
                'other, so any pair that differs blinks into '
                'view.</p>')

    # the gate
    try:
        verdict = _gate.assess(fid)
        chips = "".join(
            '<span class="chip {}">{} {}</span>'.format(
                "ok" if c["ok"] else "bad",
                "PASS" if c["ok"] else "FAIL", esc(c["name"]))
            for c in verdict["criteria"])
        body.append('<div class="gate">{}</div>'.format(chips))
        body.append(gate_issues_html(fid, verdict, bp))
    except Exception:
        pass

    total_suppressed = 0
    numeric_cols = []
    for c, m in cols:
        kind = m.get("type")
        body.append('<h2>{}</h2>'.format(esc(c)))
        if kind == "quantiles":
            vs, vg = numeric(src[c]), numeric(gen[c])
            date_fmt = None
            if vs.notna().mean() < 0.5:
                ds = pd.to_datetime(src[c].replace("", None),
                                    errors="coerce",
                                    format="%Y-%m-%d")
                if ds.notna().mean() > 0.5:
                    vs = pd.Series(ds.map(
                        lambda d: d.toordinal()
                        if d == d else None), dtype=float)
                    dg = pd.to_datetime(gen[c].replace("", None),
                                        errors="coerce",
                                        format="%Y-%m-%d")
                    vg = pd.Series(dg.map(
                        lambda d: d.toordinal()
                        if d == d else None), dtype=float)
                    import datetime as _dt

                    def date_fmt(o):
                        try:
                            return _dt.date.fromordinal(
                                int(o)).isoformat()
                        except Exception:
                            return "-"
            else:
                numeric_cols.append(c)
            pat = None
            if gid:
                okv = vs.dropna()
                if len(okv):
                    edges = np.linspace(
                        min(float(okv.min()),
                            float(vg.dropna().min())
                            if vg.notna().any() else float(okv.min())),
                        max(float(okv.max()),
                            float(vg.dropna().max())
                            if vg.notna().any() else float(okv.max())),
                        25)
                    binidx = np.digitize(okv.to_numpy(dtype=float),
                                         edges[1:-1])
                    gg = src.loc[okv.index, gid]
                    pat = [gg[binidx == i].nunique()
                           for i in range(24)]
            svg, sup = hist_svg(vs, vg, pat, k,
                                date_labels=date_fmt)
            total_suppressed += sup
            rows = stats_rows(vs, vg)
            tbl = ('<table><tr><th></th><th>original</th>'
                   '<th>synthetic</th></tr>'
                   + "".join('<tr><td>{}</td><td class="num">{}'
                             '</td><td class="num">{}</td></tr>'
                             .format(esc(r[0]), r[1], r[2])
                             for r in rows) + "</table>")
            note = ('<div class="note">{} source bin(s) suppressed '
                    '(fewer than {} patients)</div>'.format(sup, k)
                    if sup else "")
            body.append('<div class="card"><div>{}</div>'
                        '<div>{}{}</div></div>'.format(
                            tbl, svg, note))
        elif kind == "levels":
            def shares(series):
                v = series.replace("", None).dropna()
                return v.value_counts(normalize=True)
            ss, sg = shares(src[c]), shares(gen[c])
            levels = list(ss.index[:12])
            rows = [(lv, float(ss.get(lv, 0.0)),
                     float(sg.get(lv, 0.0))) for lv in levels]
            supp = (m.get("suppressed_levels") or {}).get("count")
            note = ('<div class="note">{} level(s) suppressed by '
                    'the k rule and never published</div>'
                    .format(supp) if supp else "")
            body.append(bar_pairs_svg(rows) + note)
        elif kind == "list":
            toks = m.get("tokens") or []
            sep = m.get("separator") or ";"

            def tok_share(series, t):
                v = series.fillna("").astype(str)
                return float(v.apply(
                    lambda cell: t in [x.strip() for x in
                                       cell.split(sep)]).mean())
            top = toks[:12]
            rows = [(t["value"], tok_share(src[c], t["value"]),
                     tok_share(gen[c], t["value"])) for t in top]
            body.append('<h3>most common items (share of all '
                        'rows)</h3>')
            body.append(bar_pairs_svg(rows))
            es = float((src[c].fillna("").astype(str).str.strip()
                        == "").mean())
            eg = float((gen[c].fillna("").astype(str).str.strip()
                        == "").mean())
            unpub = m.get("tokens_found", 0) - len(toks)
            body.append(
                '<div class="note">empty on {:.1%} of original '
                'rows, {:.1%} of synthetic &middot; {} rarer '
                'item(s) fall below the k floor and are never '
                'published</div>'.format(
                    es, eg, max(unpub, 0)))

    # THE PATTERNS THEMSELVES - the section no vendor page has.
    #
    # Profiling tools compare CORRELATIONS: one number per pair,
    # which a threshold, a saturation and a straight line can all
    # share. Discovery measured the SHAPE of every relationship on
    # held-out patients, so this draws each one three times: the
    # published curve from the contract (dashed ink), the shape
    # measured on the original (gray), and the same measurement on
    # the synthetic (cardinal). If generation kept the pattern, the
    # cardinal line sits on the gray one - including the bends no
    # correlation can see. Source bins under k patients are
    # suppressed here exactly as in the histograms.
    cat_path = run / "catalogue.json"
    if cat_path.exists():
        cat = json.loads(cat_path.read_text(encoding="utf-8"))
        cards = []
        for cl in sorted(cat.get("claims") or [],
                         key=lambda c: -(c.get("skill") or 0)):
            child = cl.get("child")
            if not child or _sets.is_scaffolding(child)                     or child not in src.columns:
                continue
            vs_c = numeric(src[child])
            vg_c = numeric(gen[child])
            if vs_c.notna().mean() < 0.5:
                continue
            for pr in (cl.get("predictors") or [])[:2]:
                par = pr.get("column")
                eff = pr.get("effect") or {}
                if not par or par not in src.columns                         or _sets.is_scaffolding(par):
                    continue
                if (eff.get("grid_kind") or "numeric") != "numeric":
                    continue
                vs_p = numeric(src[par])
                vg_p = numeric(gen[par])
                if vs_p.notna().mean() < 0.5:
                    continue
                pub = list(zip(eff.get("grid") or [],
                               eff.get("response") or []))
                r = shape_gap(vs_p, vs_c, vg_p, vg_c,
                              src[gid] if gid else None, k)
                if r is None:
                    continue
                gap, sp, gp = r
                svg = curve_svg(pub, sp, gp)
                shape = eff.get("shape") or "association"
                desc = eff.get("description") or ""
                skill = float(cl.get("skill") or 0)
                cards.append((skill, child, par, shape, desc, svg,
                              gap, cl.get("predictors") or [],
                              cl.get("interaction")))
        if cards:
            body.append('<h2>The patterns, drawn</h2>')
            body.append('<p class="note">Each relationship three '
                        'times: the published curve from the '
                        'contract (dashed), the shape measured on '
                        'the original (gray), and the same '
                        'measurement on the synthetic (cardinal). '
                        'A faithful pattern is a cardinal line on a '
                        'gray one - including the bends a '
                        'correlation cannot see.</p>')
        for (skill, child, par, shape, desc, svg, gap, preds,
                _cit) in cards[:8]:
            body.append('<h3>{} &larr; {}</h3>'.format(
                esc(child), esc(par)))
            verdict = ("the synthetic curve tracks the original "
                       "within {:.2f} of a standard deviation"
                       .format(gap) if gap < 0.35 else
                       "the synthetic curve DEPARTS from the "
                       "original by up to {:.2f} standard "
                       "deviations - read this one closely"
                       .format(gap))
            body.append('<p class="note">A {} relationship; '
                        'confirmed on held-out patients (skill '
                        '{:.0%}). {} {}</p>'.format(
                            esc(shape), skill,
                            esc(desc[:160]), verdict))
            imps = [(q.get("column"), float(q.get("importance")
                                            or 0))
                    for q in preds if q.get("column")
                    and not _sets.is_scaffolding(q.get("column"))]
            tot = sum(i for _, i in imps) or 1.0
            if len(imps) > 1:
                body.append('<div class="small">what drives it: '
                            + " &middot; ".join(
                                "{} {:.0%}".format(esc(nm), i / tot)
                                for nm, i in imps[:4]) + "</div>")
            # SAY THE MULTIVARIATE POSITION OUT LOUD, per pattern.
            # The operator looked for it and could not find it:
            # how many variables act jointly, whether a two-way
            # surface is published, and that higher-order is not
            # modeled - stated, not implied by absence.
            n_drv = len(imps)
            if n_drv > 1:
                _pr = list((_cit or {}).get("pair") or [])
                if len(_pr) == 2:
                    body.append(
                        '<div class="small">this is a '
                        'MULTI-VARIABLE pattern: {} variables act '
                        'jointly (the model is non-linear), and '
                        'the contract publishes a two-way '
                        'interaction surface for <b>{} &times; '
                        '{}</b> - its survival is gated. '
                        'Interactions above two-way are not '
                        'modeled.</div>'.format(
                            n_drv, esc(_pr[0]), esc(_pr[1])))
                else:
                    body.append(
                        '<div class="small">this is a '
                        'MULTI-VARIABLE pattern: {} variables act '
                        'jointly through a non-linear model, '
                        'combined additively per variable - no '
                        'two-way surface cleared the interaction '
                        'bar for this child. Interactions above '
                        'two-way are not modeled.</div>'.format(
                            n_drv))
            body.append(svg)


    # THE DRIVERS, ATTRIBUTED - SHAP on BOTH tables. Importance
    # said which parents matter; SHAP says how much of each row's
    # prediction each parent carries, and its interaction values
    # rank which PAIRS act jointly - the question the operator
    # asked ("what about interactions across more than two
    # variables, non-linear?"). Attribution is computed on each
    # table separately, so the paired bars answer a fidelity
    # question too: does the synthetic data distribute the driving
    # the same way the original does? Optional dependency, stated
    # when absent - a silently missing section reads as "nothing
    # to show".
    try:
        import shap as _shap
        _HAS_SHAP = True
    except Exception:
        _HAS_SHAP = False
    if not _HAS_SHAP:
        body.append(
            '<h2>The drivers, attributed</h2>'
            '<p class="note">The optional <code>shap</code> package '
            'is not installed on this machine, so driver '
            'attribution and interaction ranking are omitted - '
            'run <code>pip install shap</code> and rebuild this '
            'report to see them. Nothing else on this page '
            'depends on it.</p>')
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        _seen_children = [(c, ps) for _sk, c, ps in
                          shap_candidates(cat.get("claims"),
                                          src, gen)]
        if not _seen_children:
            body.append(
                '<h2>The drivers, attributed</h2>'
                '<p class="note">No pattern here has two or more '
                'numeric drivers present in the written files, so '
                'there is nothing to attribute - the strongest '
                'patterns in this run live on list-token '
                'indicators, which exist only inside the search, '
                'or on single or date drivers. This is a stated '
                'limit of the attribution, not an empty '
                'result.</p>')
        if _seen_children:
            body.append('<h2>The drivers, attributed &mdash; '
                        'SHAP, on both tables</h2>')
            body.append(
                '<p class="sub">For each pattern, a compact model '
                'is trained on each table and every prediction is '
                'split among the drivers (mean |SHAP|). Matching '
                'bars mean the synthetic data distributes the '
                'driving the way the original does. The strongest '
                'jointly-acting pair comes from SHAP interaction '
                'values - and whether the contract publishes a '
                'surface for that pair is stated, because an '
                'unmodeled interaction is a limitation to name, '
                'not to hide.</p>')
        for child, parents in _seen_children[:5]:
            def _shares(frame):
                import numpy as _n
                cols = [numeric(frame[p]) for p in parents]
                yv = numeric(frame[child])
                ok = yv.notna()
                for c in cols:
                    ok &= c.notna()
                if int(ok.sum()) < 120:
                    return None, None
                X = _n.column_stack(
                    [c[ok].to_numpy(dtype=float)
                     for c in cols])[:4000]
                y = yv[ok].to_numpy(dtype=float)[:4000]
                m = HistGradientBoostingRegressor(
                    max_iter=60, random_state=0).fit(X, y)
                ex = _shap.TreeExplainer(m)
                sv = _n.abs(ex.shap_values(X[:1000])).mean(0)
                tot = float(sv.sum()) or 1.0
                return [float(v) / tot for v in sv], (ex, X)
            ss, sx = _shares(src)
            gs, _ = _shares(gen)
            if ss is None or gs is None:
                continue
            rows_b = list(zip(parents, ss, gs))
            rows_b.sort(key=lambda r: -r[1])
            body.append('<h3>{} &mdash; who drives it</h3>'.format(
                esc(child)))
            body.append(bar_pairs_svg(rows_b))
            # the strongest jointly-acting pair, on the original
            try:
                import numpy as _n
                ex, X = sx
                iv = _n.abs(ex.shap_interaction_values(
                    X[:300])).mean(0)
                best, bv = None, 0.0
                for i in range(len(parents)):
                    for j in range(i + 1, len(parents)):
                        if iv[i, j] > bv:
                            best, bv = (i, j), float(iv[i, j])
                if best and bv > 0:
                    pa, pb = parents[best[0]], parents[best[1]]
                    pub_pairs = set()
                    for rel in (bp.get("relationships") or []):
                        if rel.get("child") != child:
                            continue
                        it = (rel.get("evidence") or {}).get(
                            "interaction") or {}
                        pr = it.get("pair") or []
                        if len(pr) == 2:
                            pub_pairs.add(frozenset(pr))
                    modeled = frozenset((pa, pb)) in pub_pairs
                    body.append(
                        '<p class="note">Strongest jointly-acting '
                        'pair (SHAP interaction, original): '
                        '<b>{} &times; {}</b> &mdash; {}</p>'
                        .format(esc(pa), esc(pb),
                                'the contract publishes a surface '
                                'for this pair, and its survival '
                                'is gated above.' if modeled else
                                'the contract does NOT publish a '
                                'surface for this pair - a joint '
                                'effect the generator will not '
                                'reproduce. A limitation, '
                                'stated.'))
            except Exception:
                pass

    # correlations
    if len(numeric_cols) >= 3:
        body.append('<h2>Correlation, original vs synthetic</h2>')
        ms = spearman_matrix(src, numeric_cols)
        mg = spearman_matrix(gen, numeric_cols)
        deltas = [abs((ms[i][j] or 0) - (mg[i][j] or 0))
                  for i in range(len(numeric_cols))
                  for j in range(i + 1, len(numeric_cols))
                  if ms[i][j] is not None and mg[i][j] is not None]
        body.append(heatmap_pair(numeric_cols, ms, mg))
        if deltas:
            body.append('<p class="note">rank correlation, all '
                        'numeric pairs - largest difference '
                        '{:.2f}, mean difference {:.3f} across {} '
                        'pairs</p>'.format(
                            max(deltas),
                            sum(deltas) / len(deltas),
                            len(deltas)))

    # scale comparison
    if a.compare:
        body.append('<h2>Does scale degrade it?</h2>')
        rows = ['<tr><th>run</th><th>rows</th><th>patients</th>'
                '<th>direction kept</th><th>close</th>'
                '<th>inverted</th><th>coverage</th>'
                '<th>gate</th></tr>']

        def one_row(label, rdir):
            # A MISSING COMPARE RUN GETS A SENTENCE, NOT A STACK.
            # The operator hit a raw FileNotFoundError here because
            # the 5x directory had not been generated yet - the
            # refusal now says what to do, and the 1x deck they
            # already wrote is not held hostage to it.
            fp2 = Path(rdir) / "fidelity.json"
            if not fp2.exists():
                raise ValueError(
                    "no fidelity.json in {} - the '{}' run does not "
                    "exist yet. Generate it first (synthkit fit ... "
                    "--out {} --generate), or drop this compare; the "
                    "deck without it already built."
                    .format(rdir, label, rdir))
            f2 = json.loads(fp2.read_text(encoding="utf-8"))
            s2 = f2.get("summary") or {}
            g2 = read_frame(Path(rdir) / "generated.csv")
            np2 = (g2[a.group_by].nunique()
                   if a.group_by in g2.columns else len(g2))
            try:
                met = _gate.assess(f2)["met"]
            except Exception:
                met = False
            pairs = s2.get("pairs") or 0
            return ('<tr><td>{}</td><td class="num">{}</td>'
                    '<td class="num">{}</td><td class="num">{}/{}'
                    '</td><td class="num">{}/{}</td>'
                    '<td class="num">{}</td><td class="num">{}/{}'
                    '</td><td>{}</td></tr>'.format(
                        esc(label), fnum(len(g2)), fnum(np2),
                        s2.get("pairs_sign_ok"), pairs,
                        s2.get("pairs_close"), pairs,
                        s2.get("pairs_inverted"),
                        s2.get("coverage_ok"), s2.get("columns"),
                        "MET" if met else "NOT MET"))
        rows.append(one_row("1x (this report)", run))
        for spec2 in a.compare:
            label, _, rdir = spec2.partition("=")
            rows.append(one_row(label, rdir))
        body.append("<table>{}</table>".format("".join(rows)))
        body.append('<p class="note">same blueprint, different '
                    'patient counts - the metrics that matter are '
                    'proportions, so a stable row means no '
                    'degradation.</p>')

    body.append('<div class="footer">Every published number in the '
                'contract behind this data describes at least {} '
                'patients; the stored extremes are averages over '
                'the {} most extreme patients, never one '
                "person's value. {} histogram bin(s) in this "
                'report were suppressed for the same reason. '
                'k-anonymous publication is not differential '
                'privacy; attack results are a floor, not a '
                'certificate.</div>'.format(k, k, total_suppressed))

    return ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>Original vs synthetic</title><style>" + CSS
            + "</style></head><body><div class='wrap'>"
            + "".join(body) + "</div>" + MOTION_JS
            + "</body></html>"), {
        "columns": len(cols), "suppressed": total_suppressed}


def main():
    ap = argparse.ArgumentParser(
        description="Original vs synthetic, drawn.")
    ap.add_argument("--src", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--group-by", default="person_id")
    ap.add_argument("--compare", action="append", default=[],
                    help='LABEL=RUNDIR, repeatable')
    a = ap.parse_args()
    try:
        doc, meta = build_deck(a.src, a.run, a.group_by, a.compare)
    except ValueError as e:
        sys.exit("STOPPED: {}".format(e))
    out = Path(a.out)
    out.write_text(doc, encoding="utf-8")
    print("wrote {} ({} KB, {} columns, {} source bins suppressed "
          "by the k rule)".format(out, out.stat().st_size // 1024,
                                  meta["columns"], meta["suppressed"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
