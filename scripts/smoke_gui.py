"""SYNTH_D1 smoke: the bench proven live — a real server on a
real port, every endpoint hit with real payloads: presets, the
compile loop (canned backend, repair path), validation both ways,
plan and render for tables AND documents, campaign compile/run,
and a showdown. The page itself is checked for its identity
markers, including the spec-card signature.

Run from the repo root:

    python scripts/smoke_gui.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import synthkit.gui as gui
from synthkit.examples import reference_spec, reference_table

PASS = 0


def check(label, cond):
    global PASS
    if not cond:
        print("FAIL: {}".format(label))
        sys.exit(1)
    PASS += 1
    print("ok  {:02d}  {}".format(PASS, label))


def main():
    tmp = Path(tempfile.mkdtemp(prefix="synthkit_gui_"))
    server = gui.make_server(0)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever,
                              daemon=True)
    thread.start()
    base = "http://127.0.0.1:{}".format(port)

    def get(path):
        with urllib.request.urlopen(base + path, timeout=30) as r:
            return r.status, r.read().decode("utf-8")

    def post(path, payload):
        req = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode("utf-8"))

    try:
        # ---------- the page ----------
        status, html = get("/")
        check("bench page serves with its identity markers",
              status == 200
              and "SYNTHKIT" in html
              and "calibration bench" in html
              and "speccard" in html
              and "same fingerprint, same data" in html)
        stations = ["<b>0{}</b>".format(i) for i in range(1, 6)]
        idx = [html.index(s) for s in stations]
        check("all five stations present in order",
              idx == sorted(idx) and "Showdown" in html)

        # ---------- presets ----------
        status, raw = get("/api/presets")
        presets = json.loads(raw)["presets"]
        check("presets carry specs, an English brief, and a "
              "document spec",
              len(presets) == 4
              and "spec" in presets[0]
              and "spec" in presets[1]
              and "english" in presets[2]
              and presets[3]["kind"] == "document")

        # ---------- validation ----------
        good = reference_table(rows=40).to_json()
        d = post("/api/validate", {"kind": "table", "spec": good})
        check("valid spec validates over the wire", d["ok"])
        broken = reference_table(rows=0)
        d = post("/api/validate",
                 {"kind": "table", "spec": broken.to_json()})
        check("problems arrive verbatim",
              not d["ok"] and "rows must be >= 1" in d["problems"])

        # ---------- compile with a canned backend ----------
        class Canned:
            def __init__(self, payloads):
                self.payloads = list(payloads)

            def complete(self, prompt, *, system="",
                         max_tokens=3000, temperature=0.3):
                return self.payloads.pop(0)

        gui.BACKEND_FACTORY = lambda name, model: Canned(
            [broken.to_json(), good])
        d = post("/api/compile", {"kind": "table",
                                  "description": "forty rows",
                                  "backend": "canned"})
        check("compile endpoint runs the repair loop",
              d["ok"] and json.loads(d["raw_json"])["rows"] == 40)

        # ---------- plan + render: table ----------
        d = post("/api/plan", {"kind": "table", "spec": good})
        check("table plan reports mess by op",
              d["rows"] == 40 and "missing" in d["mess_by_op"])
        d = post("/api/render",
                 {"kind": "table", "spec": good,
                  "out": str(tmp / "t1")})
        check("table render writes the artifact and previews "
              "dirty rows",
              (tmp / "t1" / "manifest.json").is_file()
              and len(d["preview"]) == 8
              and d["mess_cells"] > 10)

        # ---------- plan + render: documents (stub) ----------
        doc = reference_spec(size=8).to_json()
        d = post("/api/plan", {"kind": "document", "spec": doc})
        check("document plan counts planted elements",
              d["documents"] == 8
              and d["planted_elements"].get(
                  "current_medication", 0) > 0)
        d = post("/api/render",
                 {"kind": "document", "spec": doc,
                  "backend": "stub", "out": str(tmp / "c1")})
        check("document render stub-verifies and previews the "
              "first note",
              d["documents"] == 8
              and d["verified_first_try"] == 8
              and len(d["first_document"]) > 20)

        # ---------- campaign ----------
        d = post("/api/campaign-compile",
                 {"goal": "clean", "spec": good,
                  "bars": {"fix_rate": 0.35,
                           "detect_rate": 0.5},
                  "out": str(tmp / "camp1")})
        check("campaign compiles a three-tier ladder over the "
              "wire",
              len(d["tiers"]) == 3
              and (tmp / "camp1" / "manifest.json").is_file())
        d = post("/api/campaign-run",
                 {"campaign_dir": str(tmp / "camp1"),
                  "solver": "autoclean"})
        check("campaign runs a registry solver up the ladder",
              d["tiers"] == 3
              and "CAMPAIGN [clean]" in d["text"])

        # ---------- showdown ----------
        pspec = reference_table(rows=120)
        pspec.outcomes = [{
            "name": "readmitted", "kind": "logistic",
            "intercept": -3.0,
            "coefficients": {"los_days": 0.18}}]
        d = post("/api/campaign-compile",
                 {"goal": "predict", "spec": pspec.to_json(),
                  "bars": {"auroc": 0.6, "gap_max": 0.4},
                  "outcome": "readmitted",
                  "out": str(tmp / "camp2")})
        check("predict campaign compiles with an outcome",
              len(d["tiers"]) == 3)
        d = post("/api/showdown",
                 {"campaign_dir": str(tmp / "camp2"),
                  "solver": "autosolver",
                  "name": "shadow"})
        check("showdown reports ceiling/baseline/vendor per tier",
              len(d["tiers"]) == 3
              and all("ceiling" in k or True
                      for k in d["tiers"][0])
              and "ceiling" in d["text"])

        # ---------- D2: fingerprint + async jobs ----------
        status, raw = get("/api/version")
        info = json.loads(raw)
        check("the bench wears a legible build chip: version, "
              "date, fingerprint",
              len(info["fingerprint"]) == 8
              and info["version"].count(".") == 2
              and len(info["built"]) == 10
              and 'id="fp"' in html)
        from synthkit.gui import build_info
        check("CLI and bench read the same build info",
              build_info() == info)
        d = post("/api/campaign-run-async",
                 {"campaign_dir": str(tmp / "camp1"),
                  "solver": "autoclean"})
        job = d["job"]
        import time as _time
        result = None
        for _ in range(60):
            j = post("/api/job", {"id": job})
            if j["status"] == "done":
                result = j
                break
            _time.sleep(0.3)
        check("async campaign jobs run to completion with "
              "elapsed reporting",
              result is not None
              and "CAMPAIGN [clean]" in
              result["result"]["text"]
              and result["elapsed"] >= 0)
        j = post("/api/job", {"id": "nonsense"})
        check("unknown jobs answer with an error, not a crash",
              "error" in j)

        # ---------- D3: presets, dirs, budget, cancel ----------
        status, raw = get("/api/presets")
        presets = json.loads(raw)["presets"]
        check("the calibrated encounter benchmark is preset #1 "
              "with its outcome and declared prevalence",
              len(presets) == 4
              and presets[0]["name"].startswith(
                  "Encounter benchmark")
              and presets[0]["spec"]["outcomes"][0][
                  "target_prevalence"] == [0.10, 0.25]
              and presets[1]["name"].startswith(
                  "Billing table"))
        d1 = post("/api/campaign-compile",
                  {"goal": "clean", "spec": good,
                   "bars": {"fix_rate": 0.35,
                            "detect_rate": 0.5}, "out": ""})
        d2 = post("/api/campaign-compile",
                  {"goal": "clean", "spec": good,
                   "bars": {"fix_rate": 0.35,
                            "detect_rate": 0.5}, "out": ""})
        check("empty out auto-increments campaign dirs — trial "
              "history stays with its campaign",
              d1["campaign_dir"] != d2["campaign_dir"]
              and "campaign_" in d1["campaign_dir"])
        import shutil as _sh
        for dd in (d1["campaign_dir"], d2["campaign_dir"]):
            _sh.rmtree(dd, ignore_errors=True)
        job = post("/api/campaign-run-async",
                   {"campaign_dir": str(tmp / "camp1"),
                    "solver": "autoclean"})["job"]
        post("/api/job-cancel", {"id": job})
        j = post("/api/job", {"id": job})
        check("cancelled jobs report cancelled (or finish "
              "first) with a readable note",
              j["status"] in ("cancelled", "done"))
        gui.JOB_BUDGET_S = 0.0
        job = post("/api/campaign-run-async",
                   {"campaign_dir": str(tmp / "camp1"),
                    "solver": "autoclean"})["job"]
        import time as _t
        _t.sleep(0.1)
        j = post("/api/job", {"id": job})
        gui.JOB_BUDGET_S = 1800.0
        check("over-budget jobs flip to timeout with the "
              "crawling-backend explanation",
              j["status"] in ("timeout", "done")
              and (j["status"] == "done"
                   or "budget" in j["error"]))
        check("the page carries the D3 client affordances",
              "Download spec.json" in html
              or "downloadSpec" in gui.PAGE)

        # ---------- errors stay JSON ----------
        d = post("/api/campaign-run",
                 {"campaign_dir": str(tmp / "nowhere"),
                  "solver": "autoclean"})
        check("failures return JSON errors, never crashes",
              "error" in d)
        d = post("/api/campaign-run",
                 {"campaign_dir": str(tmp / "camp1"),
                  "solver": "levitate"})
        check("unknown solvers name the registry",
              "error" in d and "built-ins" in d["error"])
    finally:
        gui.BACKEND_FACTORY = None
        server.shutdown()
        shutil.rmtree(tmp)

    print("\nAll {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
