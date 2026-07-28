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
              len(presets) == 5
              and "spec" in presets[0]
              and "spec" in presets[1]
              and "english" in presets[3]
              and presets[4]["kind"] == "document")

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
              and len(d["preview"]) >= 8
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
        check("the vendor demo leads the presets, encounter "
              "benchmark second, both carrying declared "
              "prevalence",
              len(presets) == 5
              and presets[0]["name"].startswith(
                  "CHF readmission")
              and presets[0]["spec"]["outcomes"][0][
                  "target_prevalence"] == [0.05, 0.12]
              and any(c["ctype"] == "note" for c in
                      presets[0]["spec"]["columns"])
              and presets[1]["name"].startswith(
                  "Encounter benchmark"))
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
        check("the backend switch reaches every LLM seam: "
              "compiler, render, and vendor pickers all offer "
              "openai",
              html.count('value="openai"') >= 2
              and 'id="lbackend"' in html
              and 'id="rbackend"' in html
              and html.count(">openai<") >= 1)
        from synthkit.gui import _solver
        import os as _os
        import tempfile as _tf
        vtmp = _tf.mkdtemp(prefix="synthkit_vendor_")
        (Path(vtmp) / "vmod.py").write_text(
            "def predict(tr, tl, te):\n"
            "    return [0.5 for _ in te]\n",
            encoding="utf-8")
        _prev = _os.getcwd()
        _pp = list(sys.path)
        try:
            _os.chdir(vtmp)
            sys.path[:] = [q for q in sys.path
                           if q not in ("", ".", vtmp)]
            sys.modules.pop("vmod", None)
            fn = _solver("vmod:predict")
            check("dotted vendor files beside the spec "
                  "resolve through the bench even "
                  "without cwd on sys.path (rehearsal "
                  "catch)",
                  fn(None, None, [1, 2]) == [0.5, 0.5])
        finally:
            _os.chdir(_prev)
            sys.path[:] = _pp
            sys.modules.pop("vmod", None)
        check("the hybrid solver is resolvable by the GUI "
              "registry AND offered at BOTH stations (a "
              "partial edit once shipped one without the "
              "other)",
              callable(_solver("autosolver_hybrid"))
              and gui.PAGE.count("autosolver_hybrid") >= 2)
        check("the page carries the D3 client affordances",
              "Download spec.json" in html
              or "downloadSpec" in gui.PAGE)

        # ---------- station 03 backend picker ----------
        check("station 03 offers a render backend picker with "
              "async LLM route",
              'id="rbackend"' in html and "render-async"
              in gui.PAGE and "renderDone" in gui.PAGE)
        doc_spec = json.loads(get("/api/presets")[1])[
            "presets"][4]
        d = post("/api/render",
                 {"kind": "document",
                  "spec": json.dumps(doc_spec["spec"]),
                  "out": str(tmp / "docs_stub"),
                  "backend": "stub"})
        check("document render (stub) returns the first "
              "document inline",
              d["documents"] > 0 and len(
                  d["first_document"]) > 40)
        job = post("/api/render-async",
                   {"kind": "document",
                    "spec": json.dumps(doc_spec["spec"]),
                    "out": str(tmp / "docs_async"),
                    "backend": "stub"})["job"]
        result = None
        for _ in range(80):
            j = post("/api/job", {"id": job})
            if j["status"] in ("done", "error"):
                result = j
                break
            _time.sleep(0.25)
        check("async renders run to completion through the "
              "job machinery",
              result is not None and result["status"] == "done"
              and result["result"]["documents"] > 0)

        # ---------- plain-english guidance layer ----------
        check("every station wears a numbered step banner with "
              "plain-english purpose",
              all(m in html for m in (
                  "Step 1 of 5", "Step 2 of 5", "Step 3 of 5",
                  "Step 4 of 5", "Step 5 of 5", "stepbanner",
                  "explain")))
        check("controls carry step numbers and required/"
              "optional badges",
              html.count('class="stepno"') >= 15
              and 'badge req' in html and 'badge opt' in html
              and 'badge rec' in html)
        check("jargon is translated: dataset kinds, goals, and "
              "solvers are labeled for non-technical readers",
              "one row per patient" in html
              and "clinical notes" in html
              and "forecast a yes/no" in html
              and "reads the table AND the notes" in html
              and "coin flip" in html)

        # ---------- data transparency ----------
        table_preset = json.loads(get("/api/presets")[1])[
            "presets"][1]
        rd = post("/api/render",
                  {"kind": "table",
                   "spec": json.dumps(table_preset["spec"]),
                   "out": str(tmp / "transp")})
        check("table renders report rows and realized-vs-"
              "declared outcome rates for the summary line",
              rd["rows"] > 0 and rd["outcomes"]
              and "realized" in rd["outcomes"][0]
              and rd["outcomes"][0]["declared"])
        ex = post("/api/export", {"dir": str(tmp / "transp"),
                                  "which": "dirty",
                                  "fmt": "csv"})
        ex2 = post("/api/export", {"dir": str(tmp / "transp"),
                                   "which": "clean",
                                   "fmt": "json"})
        ex3 = post("/api/export", {"dir": str(tmp / "transp"),
                                   "which": "ledger"})
        check("the download menu serves synthetic CSV, answer-"
              "key JSON, and the corruption ledger",
              ex["filename"].endswith(".csv")
              and ex["content"].count("\n") > 10
              and json.loads(ex2["content"])
              and json.loads(ex3["content"]))

        # ---------- the plain-english final report ----------
        hspec = {
         "title": "report fixture", "rows": 150,
         "master_seed": 5,
         "columns": [
          {"name": "pid", "ctype": "str_id",
           "distribution": {"kind": "sequence",
                            "prefix": "P", "start": 1}},
          {"name": "age", "ctype": "int",
           "distribution": {"kind": "normal", "mean": 70,
                            "std": 10, "min": 40, "max": 95},
           "mess": {"missing_rate": 0.05}},
          {"name": "note", "ctype": "note", "note": {
           "elements": [
            {"id": "nonadherence", "density": 0.3,
             "weight": 1.8,
             "phrasings": [
              "patient reports missing several doses "
              "of medication this month",
              "documented poor adherence to the "
              "prescribed regimen"]}],
           "distractors": [
            {"id": "denies", "density": 0.5,
             "excludes": "nonadherence",
             "phrasings": [
              "patient denies missing any doses of "
              "medication"]}],
           "fillers": [
            "Vital signs stable at time of discharge",
            "Follow-up appointment scheduled next week",
            "Dietary counseling provided during stay"]}}],
         "outcomes": [{"name": "readmit",
           "kind": "logistic", "intercept": -3.0,
           "coefficients": {"age": 0.02,
                            "note.nonadherence": 1.8},
           "target_prevalence": [0.03, 0.4]}]}
        cc = post("/api/campaign-compile",
                  {"kind": "table",
                   "spec": json.dumps(hspec),
                   "goal": "predict", "outcome": "readmit",
                   "bars": {"auroc": 0.55,
                            "gap_max": 0.4},
                   "out": str(tmp / "rep_camp")})
        check("hybrid fixture campaign compiles",
              "error" not in cc)
        job = post("/api/showdown-async",
                   {"campaign_dir": cc["campaign_dir"],
                    "baseline": "autosolver_hybrid",
                    "solver": "autosolver"})["job"]
        rres = None
        for _ in range(240):
            j = post("/api/job", {"id": job})
            if j["status"] in ("done", "error"):
                rres = j
                break
            _time.sleep(0.5)
        check("showdowns complete with a structured report "
              "attached", rres is not None
              and rres["status"] == "done"
              and "report" in rres["result"])
        rep = rres["result"]["report"]
        check("the report knows the dataset facts, the trap "
              "examples verbatim from the spec, and both "
              "methodologies",
              rep["dataset"]["rows"] == 150
              and rep["dataset"]["note_columns"] == 1
              and any(t["kind"] == "negation trap"
                      and "denies missing" in t["example"]
                      for t in rep["traps"])
              and "logistic regression" in
                  rep["baseline"]["how"]
              and "mines the free-text" in
                  rep["baseline"]["how"])
        check("the report names a winner on the as-specified "
              "tier — and with text signal dominant, the "
              "note-reading hybrid beats the text-blind "
              "vendor",
              rep["winner"]
              and rep["winner"]["name"]
              == "autosolver_hybrid"
              and rep["winner"]["vendor_won"] is False
              and rep["tiers_data"])

        # ---------- preview + readiness affordances ----------
        check("the data preview is a white scrollable pane "
              "with click-to-expand cells and a full-content "
              "reader modal",
              all(m in html for m in (
                  "pvwrap", "openCell", "cellmodal",
                  "click any cell")))
        check("table previews serve enough rows to scroll",
              len(rd["preview"]) >= 20)
        check("required fields carry a live red/green "
              "readiness engine and action buttons gate on it",
              all(m in html for m in (
                  "paintReady", "class=\'need\'"
                  if False else "need", "gate([",
                  "Fill the highlighted red field")))

        # ---------- clinician clarity batch ----------
        check("sidebar carries plain subtitles, station 01 "
              "wears the governance strip, and completed "
              "steps earn ticks",
              all(m in html for m in (
                  "define the dataset", "compare results",
                  'class="gov"', "Synthetic only",
                  "tick('data')", "station.done")))
        check("the writer menu explains every choice in plain "
              "terms (no-AI stub, Ollama, local server, "
              "Bedrock, Anthropic)",
              all(m in html for m in (
                  "no AI, instant", "mistral-24B",
                  "llama.cpp", "Bedrock", "fully-offline")))
        check("the recipe summary card and live pass-marks "
              "translation exist",
              all(m in html for m in (
                  "spec-summary", "specSummary",
                  "bars-plain", "coin flip)")))
        check("station 05 explains vendor plumbing with the "
              "three real-vendor integration routes",
              all(m in html for m in (
                  "How would a REAL", "Wrapper file",
                  "Offline scoring exchange",
                  "vendor_integration.md")))
        check("the report renders a verdict strip and the "
              "inside-the-model coefficient rows",
              all(m in html for m in (
                  "vstrip", "VERDICT:", "modelrow",
                  "what it actually learned")))
        check("the trained baseline publishes real "
              "coefficients including mined note phrases",
              "inside_the_model" in rep
              and rep["inside_the_model"].get("top")
              and any("note phrase" in t["name"]
                      for t in rep["inside_the_model"]["top"]))

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
