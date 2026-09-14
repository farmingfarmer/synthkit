"""The report card - one page a decision-maker reads unaided.

    python scripts/report_card.py CAMPAIGNDIR --planted PLANTED.json --showdown SHOWDOWN.json -o CARD.html

The ceiling / baseline / vendor line, the answer that was planted,
the bar verdicts with the bar itself judged, and what this exam
does and does not claim - assembled from artifacts the run already
wrote, never recomputed, so the card and the terminal cannot
disagree. House palette, self-contained, no network.

A BAR ABOVE ITS OWN CEILING IS CALLED OUT BY NAME. The first real
run failed its as-specified tier at 0.683 against a 0.700 bar -
while the tier's ceiling was 0.698, so no solver on earth clears
it. An exam that knows its ceiling can say "miscalibrated bar,
near-optimal solver" in one line; this card says it wherever it
happens.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

INK = "#1d1d1f"
GRAY = "#6e6e73"
RED = "#8c1515"
GOLD = "#B3995D"
RULE = "#e3e3e6"

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',
  Helvetica,Arial,sans-serif;color:%(ink)s;background:#fff;
  margin:0;font-size:14px;line-height:1.5}
.wrap{max-width:960px;margin:0 auto;padding:48px 32px}
h1{font-size:28px;font-weight:700;letter-spacing:-.02em;margin:0}
h2{font-size:18px;font-weight:700;margin:36px 0 6px;
  padding-top:20px;border-top:1px solid %(rule)s}
.sub{color:%(gray)s;margin:4px 0 0}
table{border-collapse:collapse;font-size:13px;margin:12px 0}
table.wide{width:100%%}
td.num{padding-left:28px}
th{font-weight:600;text-align:left;color:%(gray)s;
  border-bottom:1px solid %(rule)s;padding:6px 14px 6px 0}
td{padding:6px 14px 6px 0;border-bottom:1px solid #f0f0f2;
  font-variant-numeric:tabular-nums}
td.num{text-align:right}
.verdict{font-weight:700}
.pass{color:#2e7d55}.fail{color:%(red)s}
.callout{margin:14px 0;padding:12px 18px;background:#FDFCF8;
  border:1px solid #e9e0cb;border-left:5px solid %(gold)s;
  border-radius:0 12px 12px 0}
.note{color:%(gray)s;font-size:12.5px;margin:6px 0}
.footer{margin-top:40px;padding-top:16px;border-top:1px solid
  %(rule)s;color:%(gray)s;font-size:12px}
.bar{height:10px;background:#EFF1F4;border-radius:5px;
  position:relative;margin:2px 0}
.bar i{position:absolute;top:0;bottom:0;border-radius:5px;
  display:block}
""" % {"ink": INK, "gray": GRAY, "red": RED, "gold": GOLD,
       "rule": RULE}


def esc(x):
    return html.escape(str(x))


def build_card(campaign_dir, planted_path, showdown_path):
    """Return (html, meta) - the card, from artifacts only."""
    cd = Path(campaign_dir)
    camp = json.loads((cd / "campaign.json").read_text(
        encoding="utf-8"))
    result = json.loads((cd / "result.json").read_text(
        encoding="utf-8"))
    planted = json.loads(Path(planted_path).read_text(
        encoding="utf-8"))
    showdown = json.loads(Path(showdown_path).read_text(
        encoding="utf-8"))
    blk = planted.get("planted") or {}
    effects = blk.get("effects_in_sds") or {}
    vendor = showdown.get("vendor") or "vendor"

    measured = {t["tier"]: t for t in (result.get("tiers") or [])}
    body = []
    body.append("<h1>Vendor evaluation report card</h1>")
    body.append('<p class="sub">{} &middot; solver graded against '
                'a PLANTED answer on measured covariates &middot; '
                'baseline: {} &middot; vendor seat: {}</p>'.format(
                    esc(camp.get("title") or ""),
                    esc(result.get("solver") or "?"), esc(vendor)))

    # the planted answer - the reason a ceiling exists at all
    body.append("<h2>The answer that was planted</h2>")
    body.append('<p class="note">Outcome `{}` ({}), requested '
                'prevalence {:.0%}. Effects are in standard '
                'deviations of each covariate; every other column '
                'is a null by construction.</p>'.format(
                    esc(blk.get("outcome") or "outcome"),
                    esc(blk.get("kind") or "logistic"),
                    float(blk.get("requested_prevalence") or 0)))
    body.append("<table><tr><th>covariate</th>"
                "<th>planted effect (sd)</th></tr>"
                + "".join(
                    '<tr><td>{}</td><td class="num">{:+.2f}</td>'
                    "</tr>".format(esc(k), float(v))
                    for k, v in effects.items())
                + "</table>")

    # the line: ceiling / baseline / vendor, with bar judged
    body.append("<h2>Ceiling / baseline / vendor</h2>")
    rows = []
    miscal = []
    for t in showdown.get("tiers") or []:
        name = t.get("tier")
        m = (measured.get(name) or {}).get("measured") or {}
        ceiling = float(t.get("ceiling") or 0)
        base = float(t.get("baseline") or 0)
        vend = float(t.get("vendor") or 0)
        ok = bool(t.get("passed"))
        # the bar itself, read from the recorded evidence
        ev = (measured.get(name) or {}).get("evidence") or ""
        bar = None
        for ln in ev.splitlines():
            if "predict.auroc =" in ln and "required >=" in ln:
                try:
                    bar = float(ln.split("required >=")[1]
                                .split(")")[0])
                except ValueError:
                    pass
        bar_above = bar is not None and bar > ceiling
        if bar_above:
            miscal.append((name, bar, ceiling))
        gap = m.get("predict.auroc_gap")
        rows.append(
            '<tr><td>{}</td><td class="num">{:.3f}</td>'
            '<td class="num">{:.3f}</td><td class="num">{:.3f}</td>'
            '<td class="num">{}</td>'
            '<td class="num">{}</td>'
            '<td class="verdict {}">{}</td></tr>'.format(
                esc(name), ceiling, base, vend,
                "{:.3f}".format(gap) if gap is not None else "-",
                "{:.2f}".format(bar) if bar is not None else "-",
                "pass" if ok else "fail",
                "PASS" if ok else ("FAIL (bar &gt; ceiling)"
                                   if bar_above else "FAIL")))
    body.append('<table class="wide"><tr><th>tier</th>'
                "<th>ceiling</th>"
                "<th>baseline</th><th>vendor</th>"
                "<th>gap to ceiling</th><th>bar</th>"
                "<th>verdict</th></tr>" + "".join(rows)
                + "</table>")
    body.append('<p class="note">The ceiling is the best score the '
                'planted answer permits ANY solver; it is '
                'computable only because the answer is known. The '
                'gap - what a solver leaves on the table - is the '
                'solver-quality verdict; the bar is a business '
                'threshold somebody chose.</p>')

    for name, bar, ceiling in miscal:
        body.append(
            '<div class="callout"><b>The {} bar is above its own '
            "ceiling</b> &mdash; {:.2f} demanded where the planted "
            "signal permits at most {:.3f}. No solver on earth "
            "clears it; this is a miscalibrated bar, not a failed "
            "solver, and the gap column is the honest reading. "
            "Re-run with the bar beneath the ceiling, or judge on "
            "gap.</div>".format(esc(name), bar, ceiling))

    body.append("<h2>What this measures, and what it does not</h2>")
    body.append(
        '<p class="note">The covariates are measured from the real '
        "extract through the k-anonymous bridge; the OUTCOME is "
        "invented on purpose, because a grade needs an answer "
        "known in advance. Effect shapes, interactions and "
        "dynamics did not cross the bridge. What a model is asked "
        "here is NARROWER than &ldquo;does this work on our "
        "data&rdquo;, and this card says so about itself. A "
        "vendor replaces the vendor seat with their own code; "
        "nothing else changes.</p>")
    body.append('<div class="footer">Assembled from campaign.json, '
                "result.json, the planted record and showdown.json "
                "- the numbers are the run's own, never recomputed "
                "here. k-anonymous publication is not differential "
                "privacy; attack results are a floor, not a "
                "certificate.</div>")

    doc = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
           "<title>Vendor evaluation report card</title><style>"
           + CSS + "</style></head><body><div class='wrap'>"
           + "".join(body) + "</div></body></html>")
    return doc, {"tiers": len(rows), "miscalibrated": len(miscal)}


def main():
    ap = argparse.ArgumentParser(
        description="The report card, from a finished campaign.")
    ap.add_argument("campaign_dir")
    ap.add_argument("--planted", required=True)
    ap.add_argument("--showdown", required=True)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    for p, what in ((Path(a.campaign_dir) / "result.json",
                     "campaign result - run campaign-run first"),
                    (Path(a.planted), "planted record"),
                    (Path(a.showdown), "showdown json")):
        if not p.exists():
            sys.exit("STOPPED: no {} at {}".format(what, p))
    doc, meta = build_card(a.campaign_dir, a.planted, a.showdown)
    Path(a.out).write_text(doc, encoding="utf-8")
    print("wrote {} ({} tier(s), {} miscalibrated bar(s) called "
          "out)".format(a.out, meta["tiers"], meta["miscalibrated"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
