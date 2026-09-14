"""The sign-off page - one roll-up a decision-maker signs.

    python scripts/signoff.py RUNDIR -o SIGNOFF.html

Everything a release decision needs, on one page, from artifacts
the run already wrote - fidelity.json, blueprint.json,
provenance.json - never recomputed, so this page and the Verdict
station cannot disagree (the gate verdicts come from synthkit.gate,
the ONE place they are defined). The privacy posture is stated in
the non-overclaiming words, with the measured counts beside it, and
the signature block says exactly what signing accepts.
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
GREEN = "#2e7d55"
RULE = "#e3e3e6"

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',
  Helvetica,Arial,sans-serif;color:%(ink)s;background:#fff;
  margin:0;font-size:14px;line-height:1.55}
.wrap{max-width:880px;margin:0 auto;padding:48px 32px}
h1{font-size:26px;font-weight:700;letter-spacing:-.02em;margin:0}
h2{font-size:17px;font-weight:700;margin:32px 0 6px;
  padding-top:18px;border-top:1px solid %(rule)s}
.sub{color:%(gray)s;margin:4px 0 0}
.gate{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}
.chip{font-size:11.5px;font-weight:700;padding:4px 10px;
  border-radius:6px}
.chip.ok{background:#f2f2f4;color:%(ink)s}
.chip.bad{background:%(red)s;color:#fff}
.met{font-size:15px;font-weight:750;margin:8px 0}
.met.ok{color:%(green)s}.met.bad{color:%(red)s}
table{border-collapse:collapse;font-size:13px;margin:10px 0}
th{font-weight:600;text-align:left;color:%(gray)s;
  border-bottom:1px solid %(rule)s;padding:5px 18px 5px 0}
td{padding:5px 18px 5px 0;border-bottom:1px solid #f0f0f2;
  font-variant-numeric:tabular-nums}
.note{color:%(gray)s;font-size:12.5px;margin:6px 0}
.sig{margin-top:28px;padding:16px 20px;border:1px solid
  %(rule)s;border-radius:12px}
.sig .line{margin:22px 0 4px;border-bottom:1px solid %(ink)s;
  width:320px;height:18px}
.sig .lbl{color:%(gray)s;font-size:11.5px}
.footer{margin-top:36px;padding-top:14px;border-top:1px solid
  %(rule)s;color:%(gray)s;font-size:12px}
""" % {"ink": INK, "gray": GRAY, "red": RED, "green": GREEN,
       "rule": RULE}


def esc(x):
    return html.escape(str(x))


def build_signoff(run_dir):
    """Return (html, meta) from the run's own artifacts."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from synthkit import gate as _gate
    run = Path(run_dir)
    fid = json.loads((run / "fidelity.json").read_text(
        encoding="utf-8"))
    bp = json.loads((run / "blueprint.json").read_text(
        encoding="utf-8"))
    prov = {}
    if (run / "provenance.json").exists():
        prov = json.loads((run / "provenance.json").read_text(
            encoding="utf-8"))
    verdict = _gate.assess(fid)
    s = fid.get("summary") or {}
    contradictions = len(fid.get("contradictions") or [])
    disobedience = len(fid.get("disobedience") or [])

    # privacy counts, straight off the contract
    k = 10
    invented, suppressed, unpub = [], [], []
    for c, spec in (bp.get("columns") or {}).items():
        m = (spec or {}).get("marginal") or {}
        k = int(m.get("bounds_are_k_anonymous") or k)
        if m.get("invented_labels"):
            invented.append(c)
        if m.get("suppressed_levels"):
            suppressed.append(c)
        v = m.get("unpublishable_row_share")
        if v:
            unpub.append((c, float(v)))

    build = prov.get("build")
    build_id = (build.get("id", "unknown")
                if isinstance(build, dict) else str(build or "?"))
    src = prov.get("source") or {}

    body = []
    body.append("<h1>Release sign-off</h1>")
    body.append('<p class="sub">run `{}` &middot; build {} '
                "&middot; source: {} row(s), {} patient(s) "
                "&middot; this page is assembled from the run's "
                "own artifacts and recomputes nothing.</p>".format(
                    esc(run.name), esc(build_id),
                    esc(src.get("rows_read", "?")),
                    esc(src.get("patients", "?"))))

    # the gate
    body.append("<h2>The gate</h2>")
    body.append('<p class="met {}">{}</p>'.format(
        "ok" if verdict["met"] else "bad",
        "MET on all {} criteria".format(len(verdict["criteria"]))
        if verdict["met"] else
        "NOT MET &mdash; " + ", ".join(
            c["name"] for c in verdict["criteria"]
            if not c["ok"])))
    body.append('<div class="gate">' + "".join(
        '<span class="chip {}">{} {}</span>'.format(
            "ok" if c["ok"] else "bad",
            "PASS" if c["ok"] else "FAIL", esc(c["name"]))
        for c in verdict["criteria"]) + "</div>")
    body.append("<table><tr><th>criterion</th><th>reading</th>"
                "</tr>" + "".join(
                    "<tr><td>{}</td><td>{}</td></tr>".format(
                        esc(c["name"]), esc(c["detail"]))
                    for c in verdict["criteria"]) + "</table>")
    body.append('<p class="note">{} contradiction(s) among '
                "published numbers; {} disobeyed declaration(s). "
                "The same criteria, from the same module, drive "
                "the Verdict station and scripts/m0_gate.py - the "
                "three cannot disagree.</p>".format(
                    contradictions, disobedience))

    # privacy posture - the non-overclaiming words, with counts
    body.append("<h2>Privacy posture</h2>")
    parts = []
    parts.append("Every number the contract publishes describes at "
                 "least {} patients; stored extremes are averages "
                 "over the {} most extreme patients, never one "
                 "person's value.".format(k, k))
    if suppressed:
        parts.append("{} column(s) had sub-{} levels suppressed "
                     "({}).".format(len(suppressed), k,
                                    esc(", ".join(suppressed[:4]))))
    if invented:
        parts.append("{} column(s) publish INVENTED labels because "
                     "no real label cleared the floor ({}).".format(
                         len(invented),
                         esc(", ".join(invented[:4]))))
    if unpub:
        parts.append("List columns carry rows whose every token "
                     "sits below the floor: " + ", ".join(
                         "{} {:.0%}".format(esc(c), v)
                         for c, v in unpub[:4]) + ".")
    body.append('<p class="note">' + " ".join(parts) + "</p>")
    body.append('<p class="note"><b>k-anonymous publication is '
                "NOT differential privacy.</b> Membership and "
                "attribute-disclosure attacks are run with "
                "positive controls - an attack that cannot catch a "
                "cheating generator proves nothing - and their "
                "results are a floor measured on one cohort, not a "
                "certificate. Effect curves, interaction surfaces "
                "and dynamics have not been audited to the same "
                "standard as the published marginals.</p>")

    # fidelity summary
    body.append("<h2>Fidelity, summarized</h2>")

    def frac(a, b):
        return "{} / {}".format(a if a is not None else "?",
                                b if b is not None else "?")
    rows = [
        ("columns present at the source rate",
         frac(s.get("coverage_ok"), s.get("columns"))),
        ("numeric centers within a tenth of spread",
         frac(s.get("centre_ok"), s.get("numeric"))),
        ("relationships keeping direction",
         frac(s.get("pairs_sign_ok"), s.get("pairs"))),
        ("relationships within 0.2",
         frac(s.get("pairs_close"), s.get("pairs"))),
        ("inverted relationships",
         str(s.get("pairs_inverted"))),
        ("effect-curve shapes tracked",
         frac(s.get("shapes_ok"), s.get("shapes_compared"))),
        ("interaction surfaces tracked",
         frac(s.get("surfaces_ok"), s.get("surfaces_compared"))),
        ("set token shares at source frequency",
         frac(s.get("set_tokens_ok"),
              s.get("set_tokens_compared"))),
        ("set EMPTY rates matched",
         frac(s.get("set_empty_ok"), s.get("set_empty_compared"))),
    ]
    body.append("<table><tr><th>measure</th><th>reading</th></tr>"
                + "".join("<tr><td>{}</td><td>{}</td></tr>".format(
                    esc(a), esc(b)) for a, b in rows)
                + "</table>")

    # limitations - restated, not hidden
    body.append("<h2>Limitations, stated</h2>")
    lims = ["One cohort, one seed: every number above is a floor, "
            "not a certificate, and moves with the seed."]
    for c in verdict["criteria"]:
        if not c["ok"]:
            lims.append("`{}` did not meet its bar ({}); the "
                        "dashboard enumerates the misses by "
                        "name.".format(c["name"], c["detail"]))
    body.append('<p class="note">' + " ".join(
        esc(x) for x in lims) + "</p>")

    # the signature block
    body.append('<div class="sig"><b>What signing accepts.</b> '
                "The release of the GENERATED file and this page. "
                "No record from the source travels with either; "
                "the contract behind them publishes aggregates "
                "over at least {} patients. The reviewer has read "
                "the gate, the privacy posture and the "
                "limitations above."
                '<div class="line"></div>'
                '<div class="lbl">reviewed by</div>'
                '<div class="line"></div>'
                '<div class="lbl">date</div></div>'.format(k))
    body.append('<div class="footer">Assembled from '
                "fidelity.json, blueprint.json and provenance.json "
                "in `{}` - the numbers are the run's own, never "
                "recomputed here.</div>".format(esc(run.name)))

    doc = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
           "<title>Release sign-off</title><style>" + CSS
           + "</style></head><body><div class='wrap'>"
           + "".join(body) + "</div></body></html>")
    return doc, {"met": verdict["met"],
                 "criteria": len(verdict["criteria"])}


def main():
    ap = argparse.ArgumentParser(
        description="One sign-off page from a finished run.")
    ap.add_argument("rundir")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    run = Path(a.rundir)
    for name in ("fidelity.json", "blueprint.json"):
        if not (run / name).exists():
            sys.exit("STOPPED: no {} in {} - run `synthkit fit` "
                     "with --generate first".format(name, run))
    try:
        doc, meta = build_signoff(run)
    except ValueError as e:
        sys.exit("STOPPED: {}".format(e))
    Path(a.out).write_text(doc, encoding="utf-8")
    print("wrote {} - gate {} on {} criteria".format(
        a.out, "MET" if meta["met"] else "NOT MET",
        meta["criteria"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
