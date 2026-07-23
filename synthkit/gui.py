"""SYNTH_D1: the GUI — a calibration bench for data claims.

`synthkit gui` serves a single-page app on 127.0.0.1 wrapping the
whole pipeline: plain English -> compiled spec (with the
assisted-never-trusted repair loop and the human gate as a visible
review step) -> validate -> render + preview -> campaigns ->
showdown. Preset buttons inject the reference cases. The persistent
SPEC CARD shows the live sha256 fingerprint of the current spec —
determinism made visible: same fingerprint, same data, forever.

Stdlib only (http.server + a self-contained page). Binds loopback
exclusively. Solvers resolve from a built-in registry or a dotted
path. `BACKEND_FACTORY` is injectable for tests.

Python 3.8 compatible.
"""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, Optional

BACKEND_FACTORY = None  # tests may inject: fn(name, model) -> backend

_FINGERPRINT = None


def build_info() -> dict:
    """version · built date · fingerprint — the legible answer to
    'am I on the latest?'. Compare against `synthkit version` in
    a terminal: the CLI reads the same disk, so a match means
    the bench serves what is installed."""
    import datetime
    from . import __version__
    pkg = Path(__file__).parent
    newest = max(path.stat().st_mtime
                 for path in pkg.glob("*.py"))
    return {
        "version": __version__,
        "built": datetime.date.fromtimestamp(
            newest).isoformat(),
        "fingerprint": build_fingerprint(),
    }


def build_fingerprint() -> str:
    """Short sha over every module's bytes — the bench wears it
    in the wordmark so 'am I on the latest?' is a glance, not a
    ritual."""
    global _FINGERPRINT
    if _FINGERPRINT is None:
        import hashlib
        h = hashlib.sha256()
        pkg = Path(__file__).parent
        for path in sorted(pkg.glob("*.py")):
            h.update(path.read_bytes())
        _FINGERPRINT = h.hexdigest()[:8]
    return _FINGERPRINT


_JOBS: Dict[str, dict] = {}


JOB_BUDGET_S = 1800.0   # a campaign should not outlive this


def _start_job(fn, payload: dict) -> str:
    import time
    import uuid
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {"status": "running",
                     "started": time.time(),
                     "cancelled": False}

    def work():
        try:
            result = fn(payload)
            if "error" in result:
                _JOBS[job_id].update(status="error",
                                     error=result["error"])
            else:
                _JOBS[job_id].update(status="done",
                                     result=result)
        except Exception as e:
            _JOBS[job_id].update(status="error", error=str(e))

    threading.Thread(target=work, daemon=True).start()
    return job_id


def api_job(payload: dict) -> dict:
    import time
    job = _JOBS.get(payload.get("id", ""))
    if job is None:
        return {"error": "unknown job"}
    elapsed = round(time.time() - job["started"], 1)
    status = job["status"]
    if status == "running" and job.get("cancelled"):
        status = "cancelled"
    elif status == "running" and elapsed > JOB_BUDGET_S:
        status = "timeout"
    out = {"status": status, "elapsed": elapsed}
    if status == "done":
        out["result"] = job["result"]
    elif status == "error":
        out["error"] = job["error"]
    elif status == "timeout":
        out["error"] = ("job exceeded its {}s budget — the "
                        "backend is likely crawling or wedged "
                        "(a wedged ollama once served a 6-minute "
                        "run for 106 minutes). The thread may "
                        "still finish in the background; safe "
                        "to move on.".format(int(JOB_BUDGET_S)))
    elif status == "cancelled":
        out["error"] = ("cancelled — the current backend call "
                        "may run to completion in the "
                        "background, then stop")
    return out


def api_job_cancel(payload: dict) -> dict:
    job = _JOBS.get(payload.get("id", ""))
    if job is None:
        return {"error": "unknown job"}
    job["cancelled"] = True
    return {"ok": True}


def _backend(name: str, model: str):
    if BACKEND_FACTORY is not None:
        return BACKEND_FACTORY(name, model)
    from .cli import _backend as cli_backend
    return cli_backend(name, model)


# ===================================================================
# Solver registry
# ===================================================================

def _solver(name: str) -> Callable:
    from .autosolver import (autoclean, autosolver,
                             autosolver_regress)
    from .examples import regex_extract, strip_cleaner
    registry: Dict[str, Callable] = {
        "autoclean": autoclean,
        "strip_cleaner": strip_cleaner,
        "autosolver": autosolver(),
        "autosolver_regress": autosolver_regress(),
        "regex_extract": regex_extract,
    }
    if name in registry:
        return registry[name]
    if ":" in name:
        import importlib
        mod, fn = name.split(":", 1)
        return getattr(importlib.import_module(mod), fn)
    raise ValueError(
        "unknown solver `{}` — built-ins: {}".format(
            name, ", ".join(sorted(registry))))


# ===================================================================
# API operations
# ===================================================================

_CALIBRATED_ENCOUNTER = """{
  "title": "Inpatient encounters (readmission study)",
  "rows": 400,
  "master_seed": 12345,
  "duplicate_rate": 0.03,
  "columns": [
    {"name": "patient_id", "ctype": "str_id",
     "distribution": {"kind": "sequence", "prefix": "P",
                      "start": 20000}},
    {"name": "patient_name", "ctype": "person_name",
     "mess": {"case_rate": 0.05, "space_rate": 0.02}},
    {"name": "age", "ctype": "int",
     "distribution": {"kind": "normal", "mean": 62, "std": 15,
                      "min": 18, "max": 100},
     "mess": {"missing_rate": 0.12}},
    {"name": "admitting_department", "ctype": "category",
     "distribution": {"kind": "categorical",
       "choices": ["internal medicine", "cardiology",
                   "oncology", "emergency"],
       "weights": [0.5, 0.2, 0.2, 0.1]},
     "mess": {"typo_rate": 0.05}},
    {"name": "length_of_stay", "ctype": "int",
     "distribution": {"kind": "mixture",
       "components": [{"kind": "uniform", "min": 1, "max": 4},
                      {"kind": "uniform", "min": 10,
                       "max": 18}],
       "weights": [0.75, 0.25]},
     "mess": {"outlier_rate": 0.01, "outlier_factor": 2}},
    {"name": "total_charges", "ctype": "float",
     "mess": {"outlier_rate": 0.02, "outlier_factor": 40.0}},
    {"name": "admission_date", "ctype": "date",
     "distribution": {"kind": "date_range",
                      "start": "2026-01-01",
                      "end": "2026-06-30"},
     "mess": {"format_rate": 0.05}},
    {"name": "discharge_date", "ctype": "date",
     "mess": {"wrong_rate": 0.02}}
  ],
  "rules": [
    {"kind": "date_after", "earlier": "admission_date",
     "later": "discharge_date", "days_from": "length_of_stay"},
    {"kind": "derived", "target": "total_charges",
     "source": "length_of_stay", "factor": 2100,
     "noise_sigma": 0.15}
  ],
  "outcomes": [
    {"name": "readmitted", "kind": "logistic",
     "intercept": -3.8,
     "coefficients": {"length_of_stay": 0.1, "age": 0.02,
                      "admitting_department=oncology": 0.8},
     "target_prevalence": [0.10, 0.25]}
  ]
}"""


def api_presets() -> dict:
    from .examples import reference_spec, reference_table
    import json as _json
    calibrated = _CALIBRATED_ENCOUNTER
    return {"presets": [
        {"name": "Encounter benchmark (calibrated)",
         "kind": "table",
         "description": "the 400-row readmission study from the "
                        "live compiler benchmark: coupled dates, "
                        "derived charges, declared prevalence",
         "spec": _json.loads(calibrated)},
        {"name": "Billing table (reference)",
         "kind": "table",
         "description": "billing extract with bimodal stays, "
                        "cost derived from stay, every mess tier",
         "spec": json.loads(reference_table(rows=150).to_json())},
        {"name": "Lab results table",
         "kind": "table",
         "description": "compile this from English to watch the "
                        "assisted-never-trusted loop work",
         "english": "a 300-row lab results extract: patient id, "
                    "ordering department weighted toward internal "
                    "medicine, collection and result dates where "
                    "results follow collection by one to three "
                    "days, potassium normally distributed around "
                    "4.1, ten percent missing results, occasional "
                    "wrong-value dates, a few duplicate rows"},
        {"name": "Progress notes corpus",
         "kind": "document",
         "description": "the reference clinical-notes vertical "
                        "with the discontinued-med trap",
         "spec": json.loads(reference_spec(size=20).to_json())},
    ]}


def _load_spec(kind: str, raw: str):
    if kind == "table":
        from .tablespec import TableSpec
        return TableSpec.from_json(raw)
    from .spec import DataSpec
    return DataSpec.from_json(raw)


def api_validate(payload: dict) -> dict:
    from .spec import SpecError
    from .tablespec import TableSpecError
    try:
        spec = _load_spec(payload["kind"], payload["spec"])
    except Exception as e:
        return {"ok": False,
                "problems": "not parseable: {}".format(e)}
    try:
        spec.validate()
        return {"ok": True, "problems": ""}
    except (SpecError, TableSpecError) as e:
        return {"ok": False, "problems": str(e)}


def api_compile(payload: dict) -> dict:
    from .compiler import compile_spec, compile_table_spec
    backend = _backend(payload.get("backend", "ollama"),
                       payload.get("model", ""))
    fn = compile_table_spec if payload["kind"] == "table" \
        else compile_spec
    result = fn(payload["description"], backend, retries=1)
    return {"ok": result.ok, "raw_json": result.raw_json,
            "problems": result.problems or ""}


def api_lint(payload: dict) -> dict:
    from .lint import lint_corpus, lint_table
    if payload.get("kind") == "table":
        from .tablespec import TableSpec
        report = lint_table(
            TableSpec.from_json(payload["spec"]),
            description=payload.get("description", ""))
    else:
        from .spec import DataSpec
        report = lint_corpus(
            DataSpec.from_json(payload["spec"]),
            description=payload.get("description", ""))
    return {"ok": report.ok, "text": report.format_text()}


def api_plan(payload: dict) -> dict:
    spec = _load_spec(payload["kind"], payload["spec"])
    if payload["kind"] == "table":
        from .tableplan import plan_table
        bp = plan_table(spec)
        ops: Dict[str, int] = {}
        for m in bp.ledger:
            ops[m.op] = ops.get(m.op, 0) + 1
        return {"rows": len(bp.clean_rows),
                "dirty_rows": len(bp.dirty_rows),
                "columns": bp.columns,
                "mess_by_op": ops,
                "outcomes": sorted(bp.true_probs)}
    from .planner import plan_corpus
    spec.validate()
    blueprints = plan_corpus(spec)
    counts: Dict[str, int] = {}
    for bp in blueprints:
        for note in bp.notes:
            for el in note.elements:
                counts[el.element_id] = counts.get(
                    el.element_id, 0) + 1
    return {"documents": len(blueprints),
            "planted_elements": counts}


def api_render(payload: dict) -> dict:
    spec = _load_spec(payload["kind"], payload["spec"])
    out = Path(payload.get("out") or "gui_runs/run_001")
    if payload["kind"] == "table":
        from .tableplan import plan_table, write_table
        bp = plan_table(spec)
        run_dir = write_table(out, spec, bp)
        return {"run_dir": str(run_dir),
                "columns": bp.columns,
                "preview": bp.dirty_rows[:8],
                "mess_cells": len([m for m in bp.ledger
                                   if m.op != "duplicate"])}
    from .corpus_io import write_corpus
    from .harness import StubBackend
    from .planner import plan_corpus
    from .renderer import render_corpus
    spec.validate()
    blueprints = plan_corpus(spec)
    backend = (StubBackend()
               if payload.get("backend", "stub") == "stub"
               else _backend(payload["backend"],
                             payload.get("model", "")))
    documents, report = render_corpus(spec, blueprints, backend)
    run_dir = write_corpus(out, spec, blueprints, documents,
                           report, backend_name=payload.get(
                               "backend", "stub"))
    first = documents[sorted(documents)[0]] if documents else ""
    return {"run_dir": str(run_dir),
            "documents": len(documents),
            "first_document": first[:800],
            "verified_first_try": report.first_try}


def api_campaign_compile(payload: dict) -> dict:
    from .campaign import compile_campaign, write_campaign
    spec = _load_spec(
        "table" if payload["goal"] in ("clean", "predict",
                       "regress")
        else "document", payload["spec"])
    camp = compile_campaign(payload["goal"], spec,
                            bars=payload.get("bars") or {},
                            outcome=payload.get("outcome", ""))
    out = payload.get("out", "")
    if not out:
        # Auto-increment: each compile gets its own directory so
        # trial history stays with the campaign that produced it
        # (a live tour found regress arms filed under a clean
        # campaign's tiers).
        base = Path("gui_runs")
        base.mkdir(exist_ok=True)
        n = 1
        while (base / "campaign_{:03d}".format(n)).exists():
            n += 1
        out = base / "campaign_{:03d}".format(n)
    out = Path(out)
    write_campaign(out, camp)
    return {"campaign_dir": str(out),
            "title": camp.title,
            "tiers": [{"name": t.name, "notes": t.notes,
                       "conditions": t.conditions}
                      for t in camp.tiers]}


def api_campaign_run(payload: dict) -> dict:
    from .campaign import load_campaign, run_campaign, \
        write_campaign
    camp = load_campaign(Path(payload["campaign_dir"]))
    extractor = None
    if payload["solver"] == "llm_extract":
        if camp.goal != "extract":
            return {"error": "llm_extract runs on extract "
                             "campaigns only"}
        from .llmvendor import LLMExtractor
        from .spec import DataSpec
        spec = DataSpec.from_json(camp.tiers[0].spec_json)
        extractor = LLMExtractor(
            _backend(payload.get("backend", "ollama"),
                     payload.get("model", "")),
            spec, samples=int(payload.get("samples", 1)),
            name="llm-vendor",
            extra_system=payload.get("extra_system", ""))
        solver = extractor
    else:
        solver = _solver(payload["solver"])
    result = run_campaign(camp, solver,
                          solver_name=payload["solver"])
    write_campaign(Path(payload["campaign_dir"]), camp, result,
                   result_name=payload.get("name")
                   or payload["solver"])
    text = result.format_text()
    if extractor is not None:
        text += "\n" + extractor.stats_line()
    return {"highest_passed": result.highest_passed,
            "tiers": len(result.tier_results),
            "text": text}


def api_showdown(payload: dict) -> dict:
    from .autosolver import run_showdown
    from .campaign import load_campaign
    camp = load_campaign(Path(payload["campaign_dir"]))
    vendor = _solver(payload["solver"])
    result = run_showdown(camp, vendor,
                          vendor_name=payload.get("name")
                          or payload["solver"])
    return {"text": result.format_text(),
            "tiers": [{"tier": t.tier, "ceiling": t.ceiling,
                       "baseline": t.baseline_auroc,
                       "vendor": t.vendor_auroc,
                       "passed": t.vendor_passed}
                      for t in result.tiers]}


_ROUTES = {
    "/api/presets": lambda payload: api_presets(),
    "/api/validate": api_validate,
    "/api/compile": api_compile,
    "/api/lint": api_lint,
    "/api/plan": api_plan,
    "/api/render": api_render,
    "/api/campaign-compile": api_campaign_compile,
    "/api/campaign-run": api_campaign_run,
    "/api/campaign-run-async": lambda payload: {
        "job": _start_job(api_campaign_run, payload)},
    "/api/showdown-async": lambda payload: {
        "job": _start_job(api_showdown, payload)},
    "/api/job": api_job,
    "/api/job-cancel": api_job_cancel,
    "/api/showdown": api_showdown,
}


# ===================================================================
# Server
# ===================================================================

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes,
              ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type",
                         ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html")
        elif self.path == "/api/presets":
            self._send(200, json.dumps(
                api_presets()).encode("utf-8"))
        elif self.path == "/api/version":
            self._send(200, json.dumps(
                build_info()).encode("utf-8"))
        else:
            self._send(404, b'{"error": "not found"}')

    def do_POST(self):
        handler = _ROUTES.get(self.path)
        if handler is None:
            self._send(404, b'{"error": "not found"}')
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(
                self.rfile.read(length).decode("utf-8") or "{}")
            result = handler(payload)
            self._send(200, json.dumps(result).encode("utf-8"))
        except Exception as e:
            self._send(200, json.dumps(
                {"error": str(e)}).encode("utf-8"))


def make_server(port: int = 0) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), _Handler)


def run_gui(port: int = 8377, open_browser: bool = True) -> None:
    server = make_server(port)
    url = "http://127.0.0.1:{}/".format(server.server_port)
    print("synthkit bench -> {}   (Ctrl-C to stop)".format(url))
    if open_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


# ===================================================================
# The page — calibration-bench identity, mono-forward, spec card
# ===================================================================

PAGE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>synthkit bench</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bench:#EDF0F3; --panel:#FFFFFF; --ink:#17222C; --dim:#5C6B78;
  --rule:#CBD4DC; --enamel:#0E6E64; --enamel-press:#0A544C;
  --pass:#17803D; --fail:#B3261E; --chip:#DFE6EB;
  --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  --sans:'IBM Plex Sans',system-ui,sans-serif;
}
*{box-sizing:border-box;margin:0}
body{background:var(--bench);color:var(--ink);
  font-family:var(--sans);font-size:14px;line-height:1.5}
.frame{display:grid;grid-template-columns:216px 1fr;
  min-height:100vh}
/* ---- station rail ---- */
nav{border-right:1px solid var(--rule);padding:20px 0}
.wordmark{font-family:var(--mono);font-weight:600;
  letter-spacing:.14em;padding:0 20px 18px;font-size:15px}
.wordmark small{display:block;color:var(--dim);font-weight:400;
  letter-spacing:.08em;font-size:10px;margin-top:2px}
.station{display:flex;gap:12px;align-items:baseline;width:100%;
  padding:11px 20px;border:0;background:none;text-align:left;
  font-family:var(--sans);font-size:14px;color:var(--dim);
  cursor:pointer;border-left:3px solid transparent}
.station b{font-family:var(--mono);font-size:11px;font-weight:500}
.station.active{color:var(--ink);border-left-color:var(--enamel);
  background:var(--panel)}
.station:focus-visible{outline:2px solid var(--enamel)}
/* ---- bench ---- */
main{padding:24px 32px;max-width:1080px}
header.bar{display:flex;justify-content:space-between;
  align-items:flex-start;gap:16px;margin-bottom:20px}
h1{font-family:var(--mono);font-size:19px;font-weight:600}
h1 span{color:var(--dim);font-weight:400}
/* the signature: the spec card */
#speccard{font-family:var(--mono);font-size:12px;
  background:var(--panel);border:1px solid var(--rule);
  border-top:3px solid var(--enamel);padding:10px 14px;
  min-width:250px}
#speccard .fp{font-size:15px;font-weight:600;
  letter-spacing:.06em}
#speccard .meta{color:var(--dim);margin-top:2px}
#speccard.empty{border-top-color:var(--rule);color:var(--dim)}
section{display:none}
section.active{display:block}
.panel{background:var(--panel);border:1px solid var(--rule);
  padding:18px;margin-bottom:16px}
.eyebrow{font-family:var(--mono);font-size:10px;
  letter-spacing:.16em;color:var(--dim);text-transform:uppercase;
  margin-bottom:10px}
textarea,input,select{width:100%;font-family:var(--mono);
  font-size:12.5px;border:1px solid var(--rule);padding:9px;
  background:#FBFCFD;color:var(--ink)}
textarea:focus,input:focus,select:focus{
  outline:2px solid var(--enamel);outline-offset:-1px}
textarea{resize:vertical}
label{display:block;font-size:12px;color:var(--dim);
  margin:10px 0 4px}
button.act{font-family:var(--mono);font-size:12.5px;
  font-weight:500;letter-spacing:.04em;background:var(--enamel);
  color:#fff;border:0;padding:9px 16px;cursor:pointer;
  margin:12px 8px 0 0}
button.act:hover{background:var(--enamel-press)}
button.act:focus-visible{outline:2px solid var(--ink)}
button.ghost{background:none;color:var(--enamel);
  border:1px solid var(--enamel)}
button.ghost:hover{background:#E7F0EF}
.presets{display:flex;gap:8px;flex-wrap:wrap}
.presets button{font-family:var(--mono);font-size:12px;
  background:var(--chip);border:1px solid var(--rule);
  padding:7px 12px;cursor:pointer;color:var(--ink)}
.presets button:hover{border-color:var(--enamel)}
pre.out{font-family:var(--mono);font-size:12px;line-height:1.55;
  background:#FBFCFD;border:1px solid var(--rule);padding:14px;
  overflow:auto;max-height:420px;white-space:pre-wrap;
  margin-top:12px}
pre.out:empty{display:none}
.outlabel{font-family:var(--mono);font-size:10px;
  letter-spacing:.16em;color:var(--dim);
  text-transform:uppercase;margin-top:14px}
.outlabel:has(+pre.out:empty){display:none}
pre.out .ok{color:var(--pass)} pre.out .bad{color:var(--fail)}
.verdict-pass{color:var(--pass);font-weight:600}
.verdict-fail{color:var(--fail);font-weight:600}
table.preview{border-collapse:collapse;font-family:var(--mono);
  font-size:11.5px;margin-top:12px;width:100%}
table.preview th,table.preview td{border:1px solid var(--rule);
  padding:5px 8px;text-align:left;max-width:160px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
table.preview th{background:var(--chip);
  font-family:var(--mono);font-weight:500}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.hint{color:var(--dim);font-size:12px;margin-top:8px}
@media(max-width:860px){.frame{grid-template-columns:1fr}
  nav{display:flex;overflow-x:auto;border-right:0;
  border-bottom:1px solid var(--rule)}
  .row2{grid-template-columns:1fr}}
@media(prefers-reduced-motion:no-preference){
  section.active{animation:in .16s ease-out}
  @keyframes in{from{opacity:.4}to{opacity:1}}}
</style></head><body>
<div class="frame">
<nav>
  <div class="wordmark">SYNTHKIT<small>calibration bench<br>
    <span id="fp" title="version &middot; built &middot; build
    fingerprint; compare with `synthkit version`">loading
    build...</span></small></div>
  <button class="station active" data-s="describe"><b>01</b> Describe</button>
  <button class="station" data-s="spec"><b>02</b> Spec</button>
  <button class="station" data-s="data"><b>03</b> Data</button>
  <button class="station" data-s="campaign"><b>04</b> Campaign</button>
  <button class="station" data-s="showdown"><b>05</b> Showdown</button>
</nav>
<main>
<header class="bar">
  <h1 id="title">Describe <span>— plain English in</span></h1>
  <div id="speccard" class="empty">
    <div class="fp">no spec loaded</div>
    <div class="meta">describe one or load a preset</div>
  </div>
</header>

<section id="s-describe" class="active">
  <div class="panel">
    <div class="eyebrow">presets</div>
    <div class="presets" id="presets"></div>
    <div class="hint">Presets load a finished spec into station 02,
    or drop example English below to compile fresh.</div>
  </div>
  <div class="panel">
    <div class="eyebrow">english &rarr; spec</div>
    <textarea id="english" rows="5"
      placeholder="a 300-row lab results extract: patient id, ordering department weighted toward internal medicine, ten percent missing results, occasional wrong-value dates..."></textarea>
    <div class="row2">
      <div><label for="kind">dataset kind</label>
        <select id="kind"><option value="table">table</option>
        <option value="document">document corpus</option></select></div>
      <div><label for="backend">compiler model</label>
        <select id="backend"><option value="ollama">ollama (local)</option></select></div>
    </div>
    <button class="act" onclick="compileSpec()">Compile spec</button>
    <div class="hint">Compiled specs are drafts. Validation problems
    are shown verbatim; nothing renders until the spec validates and
    you have reviewed it — that pause is the safety story.</div>
    <div class="outlabel">readout &mdash; results only, not editable</div>
    <pre class="out" id="compile-out"></pre>
  </div>
</section>

<section id="s-spec">
  <div class="panel">
    <div class="eyebrow">the spec — review before rendering</div>
    <textarea id="spec" rows="22" spellcheck="false"
      placeholder="No spec yet — describe one or load a preset."></textarea>
    <button class="act" onclick="validateSpec()">Validate</button>
    <button class="act ghost" onclick="planSpec()">Plan (dry run)</button>
    <button class="act ghost" onclick="lintSpec()">Semantic lint</button>
    <button class="act ghost" onclick="downloadSpec()">Download spec.json</button>
    <div class="outlabel">readout &mdash; results only, not editable</div>
    <pre class="out" id="spec-out"></pre>
  </div>
</section>

<section id="s-data">
  <div class="panel">
    <div class="eyebrow">render &rarr; auditable artifact</div>
    <label for="outdir">output directory</label>
    <input id="outdir" value="gui_runs/run_001">
    <button class="act" onclick="renderSpec()">Render data</button>
    <div class="hint">Tables write dirty.csv + clean.csv +
    ledger.json under an integrity manifest. Document corpora render
    stub-verified here; use the CLI with ollama for live prose.</div>
    <div class="outlabel">readout &mdash; results only, not editable</div>
    <pre class="out" id="render-out"></pre>
    <div id="render-preview"></div>
  </div>
</section>

<section id="s-campaign">
  <div class="panel">
    <div class="eyebrow">goal &rarr; tier ladder</div>
    <div class="row2">
      <div><label for="goal">goal</label>
        <select id="goal"><option>clean</option>
        <option>predict</option><option>regress</option>
        <option>extract</option></select></div>
      <div><label for="outcome">outcome (predict only)</label>
        <input id="outcome" placeholder="readmitted"></div>
    </div>
    <label for="bars">bars (k=v, comma-separated)</label>
    <input id="bars" value="fix_rate=0.9,detect_rate=0.5">
    <button class="act" onclick="campaignCompile()">Compile ladder</button>
    <label for="solver">solver</label>
    <select id="solver"><option>autoclean</option>
      <option>strip_cleaner</option><option>autosolver</option>
      <option>autosolver_regress</option>
      <option>regex_extract</option>
      <option value="llm_extract">llm_extract (ollama as the
      vendor)</option></select>
    <div class="row2">
      <div><label for="samples">llm samples (majority vote)</label>
        <input id="samples" value="1"></div>
      <div><label for="intervention">llm intervention
        (extra system prompt)</label>
        <input id="intervention" placeholder="optional"></div>
    </div>
    <button class="act" onclick="campaignRun()">Run ladder</button>
    <div class="outlabel">readout &mdash; results only, not editable</div>
    <pre class="out" id="campaign-out"></pre>
  </div>
</section>

<section id="s-showdown">
  <div class="panel">
    <div class="eyebrow">vendor vs the synthkit baseline</div>
    <div class="hint">Runs on a compiled predict campaign. Every tier
    reports ceiling / baseline / vendor — the whole meeting in one
    line per tier.</div>
    <label for="vendor">vendor solver (built-in or pkg.mod:fn)</label>
    <input id="vendor" value="autosolver">
    <button class="act" onclick="showdown()">Run showdown</button>
    <div class="outlabel">readout &mdash; results only, not editable</div>
    <pre class="out" id="showdown-out"></pre>
  </div>
</section>
</main></div>
<script>
const titles={describe:['Describe','plain English in'],
  spec:['Spec','review before rendering'],
  data:['Data','the auditable artifact'],
  campaign:['Campaign','a goal becomes a ladder'],
  showdown:['Showdown','ceiling / baseline / vendor']};
let campaignDir='';
document.querySelectorAll('.station').forEach(btn=>{
  btn.onclick=()=>{
    document.querySelectorAll('.station').forEach(
      b=>b.classList.remove('active'));
    document.querySelectorAll('section').forEach(
      s=>s.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('s-'+btn.dataset.s)
      .classList.add('active');
    const t=titles[btn.dataset.s];
    document.getElementById('title').innerHTML=
      t[0]+' <span>&mdash; '+t[1]+'</span>';
  };});
async function api(path,body){
  const r=await fetch(path,{method:body?'POST':'GET',
    headers:{'Content-Type':'application/json'},
    body:body?JSON.stringify(body):undefined});
  return r.json();}
async function sha256(text){
  const buf=await crypto.subtle.digest('SHA-256',
    new TextEncoder().encode(text));
  return[...new Uint8Array(buf)].map(
    b=>b.toString(16).padStart(2,'0')).join('');}
async function refreshCard(){
  const card=document.getElementById('speccard');
  const raw=document.getElementById('spec').value.trim();
  if(!raw){card.className='empty';card.innerHTML=
    '<div class="fp">no spec loaded</div>'+
    '<div class="meta">describe one or load a preset</div>';
    return;}
  let title='(unparsed)',rows='';
  try{const d=JSON.parse(raw);title=d.title||'(untitled)';
    rows=d.rows?d.rows+' rows':(d.corpus&&d.corpus.size?
      d.corpus.size+' docs':'');}catch(e){}
  const fp=(await sha256(raw)).slice(0,12);
  card.className='';
  card.innerHTML='<div class="fp">spec:'+fp+'</div>'+
    '<div class="meta">'+title+(rows?' &middot; '+rows:'')+
    ' &middot; same fingerprint, same data</div>';}
document.getElementById('spec')
  .addEventListener('input',refreshCard);
function setSpec(obj){
  document.getElementById('spec').value=
    JSON.stringify(obj,null,2);
  refreshCard();}
function out(id,text,cls){const el=document.getElementById(id);
  el.innerHTML='';const span=document.createElement('span');
  if(cls)span.className=cls;span.textContent=text;
  el.appendChild(span);}
async function loadPresets(){
  const d=await api('/api/presets');
  const box=document.getElementById('presets');
  d.presets.forEach(p=>{
    const b=document.createElement('button');
    b.textContent=p.name;b.title=p.description;
    b.onclick=()=>{
      if(p.spec){setSpec(p.spec);
        document.querySelector('[data-s=spec]').click();}
      else{document.getElementById('english').value=p.english;
        document.getElementById('kind').value=p.kind;}};
    box.appendChild(b);});}
async function compileSpec(){
  out('compile-out','compiling via local model...');
  const d=await api('/api/compile',{
    description:document.getElementById('english').value,
    kind:document.getElementById('kind').value,
    backend:document.getElementById('backend').value});
  if(d.error){out('compile-out',d.error,'bad');return;}
  try{setSpec(JSON.parse(d.raw_json));}catch(e){}
  out('compile-out',d.ok?
    'Compiled clean. Review the draft at station 02 before '+
    'rendering.':'Draft saved with problems — the human gate is '+
    'yours:\n\n'+d.problems,d.ok?'ok':'bad');}
async function validateSpec(){
  const d=await api('/api/validate',{
    kind:guessKind(),spec:document.getElementById('spec').value});
  out('spec-out',d.ok?'Spec validates. A validated spec must '+
    'plan — that is the contract.':d.problems,d.ok?'ok':'bad');}
function guessKind(){
  try{const d=JSON.parse(
    document.getElementById('spec').value);
    return d.columns?'table':'document';}catch(e){
    return 'table';}}
function downloadSpec(){
  const raw=document.getElementById('spec').value;
  if(!raw.trim()){out('spec-out',
    'Nothing to download - no spec loaded.','bad');return;}
  let name='spec';
  try{name=(JSON.parse(raw).title||'spec')
    .toLowerCase().replace(/[^a-z0-9]+/g,'_')
    .replace(/^_+|_+$/g,'');}catch(e){}
  const blob=new Blob([raw],{type:'application/json'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download=name+'.json';a.click();
  URL.revokeObjectURL(a.href);}
async function lintSpec(){
  const d=await api('/api/lint',{kind:guessKind(),
    spec:document.getElementById('spec').value,
    description:document.getElementById('english').value});
  out('spec-out',d.error?d.error:d.text,
    d.error?'bad':(d.ok?'ok':''));}
async function planSpec(){
  const d=await api('/api/plan',{kind:guessKind(),
    spec:document.getElementById('spec').value});
  out('spec-out',d.error?d.error:
    JSON.stringify(d,null,2),d.error?'bad':'');}
async function renderSpec(){
  out('render-out','rendering...');
  const d=await api('/api/render',{kind:guessKind(),
    spec:document.getElementById('spec').value,
    out:document.getElementById('outdir').value});
  if(d.error){out('render-out',d.error,'bad');return;}
  if(d.preview){
    out('render-out','wrote '+d.run_dir+'  ('+
      d.mess_cells+' mess cells, ledgered)');
    const cols=d.columns;
    let html='<table class="preview"><tr>'+cols.map(
      c=>'<th>'+c+'</th>').join('')+'</tr>';
    d.preview.forEach(r=>{html+='<tr>'+cols.map(
      c=>'<td>'+(r[c]===''?'&empty;':String(r[c])
      .replace(/&/g,'&amp;').replace(/</g,'&lt;'))+'</td>')
      .join('')+'</tr>';});
    document.getElementById('render-preview').innerHTML=
      html+'</table>';}
  else{out('render-out','wrote '+d.run_dir+'  ('+d.documents+
    ' docs, '+d.verified_first_try+' verified first try)\n\n'+
    d.first_document);
    document.getElementById('render-preview').innerHTML='';}}
async function campaignCompile(){
  const bars={};document.getElementById('bars').value
    .split(',').forEach(p=>{const[k,v]=p.split('=');
    if(k&&v)bars[k.trim()]=parseFloat(v);});
  out('campaign-out','compiling ladder...');
  const d=await api('/api/campaign-compile',{
    goal:document.getElementById('goal').value,
    spec:document.getElementById('spec').value,
    bars:bars,outcome:document.getElementById('outcome').value,
    out:''});
  if(d.error){out('campaign-out',d.error,'bad');return;}
  campaignDir=d.campaign_dir;saveSession();
  out('campaign-out',d.title+' -> '+d.campaign_dir+'\n\n'+d.tiers.map((t,i)=>
    'tier '+(i+1)+' `'+t.name+'`: '+t.notes+'\n    '+
    t.conditions.join('\n    ')).join('\n'));}
let activeJob=null;
async function cancelJob(){
  if(activeJob)await api('/api/job-cancel',{id:activeJob});}
async function poll(jobId,outId,render){
  activeJob=jobId;
  const t=setInterval(async()=>{
    const j=await api('/api/job',{id:jobId});
    if(j.status==='running'){
      let msg='working... '+j.elapsed+'s';
      if(j.elapsed>300)msg+='  (long run: llm ladders take '
        +'minutes; Cancel abandons it)';
      out(outId,msg);
      const el=document.getElementById(outId);
      const b=document.createElement('button');
      b.className='act ghost';b.textContent='Cancel job';
      b.style.marginLeft='12px';b.onclick=cancelJob;
      el.appendChild(b);}
    else{clearInterval(t);activeJob=null;
      if(j.status==='error'||j.status==='timeout'
        ||j.status==='cancelled'){out(outId,j.error,'bad');}
      else{render(j.result);}}},1500);}

async function campaignRun(){
  if(!campaignDir){out('campaign-out',
    'Compile a ladder first.','bad');return;}
  out('campaign-out','walking the ladder... 0s');
  const d=await api('/api/campaign-run-async',{
    campaign_dir:campaignDir,
    solver:document.getElementById('solver').value,
    samples:parseInt(document.getElementById(
      'samples').value)||1,
    extra_system:document.getElementById(
      'intervention').value});
  poll(d.job,'campaign-out',r=>out('campaign-out',r.text,
    r.highest_passed===r.tiers?'ok':''));}
async function showdown(){
  if(!campaignDir){out('showdown-out',
    'Compile a predict campaign at station 04 first.','bad');
    return;}
  out('showdown-out','running both solvers up the ladder... 0s');
  const d=await api('/api/showdown-async',{
    campaign_dir:campaignDir,
    solver:document.getElementById('vendor').value});
  poll(d.job,'showdown-out',r=>out('showdown-out',r.text,''));}
const PERSIST=['spec','english','kind','goal','outcome',
  'bars','solver','vendor','outdir','samples','intervention'];
function saveSession(){
  const state={campaignDir:campaignDir};
  PERSIST.forEach(id=>{const el=document.getElementById(id);
    if(el)state[id]=el.value;});
  try{localStorage.setItem('synthkit_bench',
    JSON.stringify(state));}catch(e){}}
function restoreSession(){
  try{
    const raw=localStorage.getItem('synthkit_bench');
    if(!raw)return;
    const state=JSON.parse(raw);
    PERSIST.forEach(id=>{const el=document.getElementById(id);
      if(el&&state[id]!==undefined)el.value=state[id];});
    if(state.campaignDir)campaignDir=state.campaignDir;
    refreshCard();
  }catch(e){}}
PERSIST.forEach(id=>{const el=document.getElementById(id);
  if(el)el.addEventListener('input',saveSession);});
restoreSession();
loadPresets();
api('/api/version').then(d=>{
  document.getElementById('fp').textContent=
    'v'+d.version+' \u00b7 '+d.built+' \u00b7 '+d.fingerprint;});
</script></body></html>
"""
