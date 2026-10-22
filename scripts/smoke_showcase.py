"""Checks for the showcase slide (scripts/showcase.py) - the premium,
self-contained page that spins the latent space and then shows the same
performance plots the deck draws.

The page is built here, from a tiny fast fixture run through the REAL
build path, and asserted on: the spinning cloud carries baked 3-D
points, the plots render, the correlation heatmap's toggle layer is
hidden (so its titles do not garble into one static frame), the gate
scorecard appears only when a fidelity.json is present, and the whole
thing reaches NO external host - the CSP an artifact runs under forbids
it, and a bench iframe inherits the same rule.
"""
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
for _p in (str(_root), str(_root / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PASS = 0
FAIL = 0


def check(name, ok):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print("FAIL: {}".format(name))


def _fixture(tmp, tag="a", cols=3, patients=40, visits=6, seed=5):
    """A small clinic-shaped frame, fit and generated through the real
    LatentGen, written as a run directory build_showcase can read."""
    from latent_challenger import LatentGen
    r = np.random.RandomState(seed)
    n = patients * visits
    gid = np.repeat(np.arange(patients), visits)
    data = {"person_id": ["P{:03d}".format(i) for i in gid]}
    age = r.normal(60, 12, n)
    data["age"] = np.round(age, 1)
    if cols >= 2:
        data["bp"] = np.round(0.5 * age + r.normal(70, 8, n), 1)
    if cols >= 3:
        data["hr"] = np.round(r.normal(75, 10, n), 1)
    data["sex"] = np.where(gid % 2 == 0, "F", "M")
    df = pd.DataFrame(data)
    base = Path(tmp) / tag
    base.mkdir(exist_ok=True)
    src = base / "src.csv"
    df.to_csv(src, index=False)
    g = LatentGen(seed=seed).fit(df, "person_id")
    gen = g.generate(len(df), seed=seed)
    run = base / "run"
    run.mkdir(exist_ok=True)
    gen.to_csv(run / "generated.csv", index=False)
    return str(src), str(run)


def _pos(html):
    m = re.search(r"var POS=(\[[^\]]*\]);", html)
    return json.loads(m.group(1)) if m else None


def main():
    import showcase

    with tempfile.TemporaryDirectory() as tmp:
        src, run = _fixture(tmp, tag="a")

        # --- no fidelity.json: page still builds, no scorecard --------
        html0 = showcase.build_showcase(run, src, "person_id", seed=5)
        check("builds without a fidelity.json (chips are conditional)",
              isinstance(html0, str) and len(html0) > 2000
              and 'class="crit' not in html0)

        # --- with a minimal fidelity.json: scorecard appears ----------
        fid = {"summary": {"pairs": 6, "pairs_sign_ok": 6,
                           "pairs_close": 5, "pairs_inverted": 0,
                           "columns": 4, "coverage_ok": 4}}
        (Path(run) / "fidelity.json").write_text(
            json.dumps(fid), encoding="utf-8")
        html = showcase.build_showcase(run, src, "person_id", seed=5)

        check("the gate scorecard renders when fidelity.json is present",
              'class="crit' in html and "direction kept" in html
              and ("MET" in html or "REVIEW" in html))

        # --- the spinning cloud carries baked 3-D points --------------
        pos = _pos(html)
        check("the latent cloud bakes a non-empty flat xyz array",
              pos is not None and len(pos) > 0 and len(pos) % 3 == 0)
        check("the WebGL canvas and draw loop are present",
              'id="gl"' in html and "drawArrays" in html
              and "requestAnimationFrame" in html)
        check("the color-mode toggle (real-vs-synthetic / fidelity) "
              "is wired",
              'id="mode"' in html and "GRP" in html and "FID" in html)

        # --- performance plots render (deck's own SVG, retinted) ------
        check("overlaid distribution plots render",
              'class="pcard"' in html
              and html.count('<svg class="plot"') >= 1)
        check("the correlation heatmap renders as a static pair",
              'class="plot hm"' in html
              and "original (left) vs synthetic (right)" in html)
        check("the heatmap's toggle layer is hidden so its titles do "
              "not garble into one frame",
              ".pcard svg.hm .hm-other{display:none;}" in html)

        # --- self-contained: reaches NO external host -----------------
        low = html.lower()
        check("the page is self-contained - no external host, CDN, "
              "link or remote script",
              "http://" not in low and "https://" not in low
              and "cdn" not in low and "<link" not in low
              and "<script src" not in low)

        # --- honest posture travels on the page ----------------------
        check("the honest privacy posture and the demo-data "
              "disclaimer are on the page",
              "membership-inference" in html
              and "invented demo data" in html)

        # --- thin data (latent dim may be < 3) still renders a cloud --
        src2, run2 = _fixture(tmp, tag="b", cols=2,
                              patients=30, visits=5, seed=9)
        html2 = showcase.build_showcase(run2, src2, "person_id", seed=9)
        pos2 = _pos(html2)
        check("thin data still yields a 3-coordinate cloud (PCA dim "
              "guard)",
              pos2 is not None and len(pos2) > 0 and len(pos2) % 3 == 0)

    print("\n{} checks, {} failed".format(PASS + FAIL, FAIL))
    if FAIL:
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
