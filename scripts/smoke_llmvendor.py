"""SYNTH_V1 smoke: the LLM-as-vendor adapter proven — vendor
prompts carry elements but NEVER distractors, JSON salvage survives
prose and fences, malformed output is tolerated and counted,
majority voting suppresses one-off hallucinations, and the adapter
runs an entire extract campaign end to end with reliability stats.

Run from the repo root:

    python scripts/smoke_llmvendor.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthkit.campaign import compile_campaign, run_campaign
from synthkit.evaluator import evaluate
from synthkit.examples import reference_spec
from synthkit.harness import StubBackend
from synthkit.llmvendor import LLMExtractor, _find_json_array
from synthkit.planner import plan_corpus
from synthkit.renderer import render_corpus

PASS = 0


def check(label, cond):
    global PASS
    if not cond:
        print("FAIL: {}".format(label))
        sys.exit(1)
    PASS += 1
    print("ok  {:02d}  {}".format(PASS, label))


class Canned:
    """Scripted backend; repeats the last payload when exhausted."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts = []
        self.systems = []

    def complete(self, prompt, *, system="", max_tokens=1200,
                 temperature=0.2):
        self.prompts.append(prompt)
        self.systems.append(system)
        if len(self.payloads) > 1:
            return self.payloads.pop(0)
        return self.payloads[0]


def main():
    spec = reference_spec(size=6)

    # ---------- prompt hygiene ----------
    backend = Canned(['[]'])
    ex = LLMExtractor(backend, spec, name="probe")
    ex.extract("doc_00000", "some note text")
    system = backend.systems[0]
    check("vendor prompt lists every target element with its "
          "description",
          "allergy_flag" in system
          and "current_medication" in system
          and "followup_appointment" in system)
    check("vendor prompt NEVER mentions the distractors — no "
          "coaching the defendant",
          "discontinued" not in system.lower()
          and "distractor" not in system.lower()
          and "trap" not in system.lower())
    check("the document text travels in the task prompt",
          "some note text" in backend.prompts[0])

    # ---------- JSON salvage ----------
    check("bare arrays parse",
          _find_json_array('[{"a": 1}]') == [{"a": 1}])
    check("fenced arrays parse",
          _find_json_array(
              'Sure!\n```json\n[{"a": 1}]\n```\nHope that helps!')
          == [{"a": 1}])
    check("arrays embedded in prose parse",
          _find_json_array(
              'The extractions are [{"a": 1}, {"b": 2}] as '
              'requested.') == [{"a": 1}, {"b": 2}])
    check("hopeless output yields None",
          _find_json_array("I cannot help with that.") is None)

    # ---------- tolerance + stats ----------
    backend = Canned([
        'garbage {not json',
        '[{"category": "current_medication", "text": '
        '"lisinopril 10mg"}, "stray-string", '
        '{"category": "", "text": "no category"}]',
    ])
    ex = LLMExtractor(backend, spec, name="messy")
    first = ex.extract("d0", "note one")
    second = ex.extract("d1", "note two")
    check("malformed calls yield empty and are counted",
          first == [] and ex.stats["malformed"] >= 1)
    check("bad items inside good arrays are skipped, good ones "
          "kept",
          len(second) == 1
          and second[0].text == "lisinopril 10mg"
          and second[0].category == "current_medication")

    # ---------- majority voting ----------
    stable = ('[{"category": "current_medication", "text": '
              '"metformin 500mg"}]')
    with_hallucination = (
        '[{"category": "current_medication", "text": '
        '"metformin 500mg"}, {"category": "allergy_flag", '
        '"text": "penicillin rash"}]')
    backend = Canned([stable, with_hallucination, stable])
    ex = LLMExtractor(backend, spec, samples=3, name="voter")
    out = ex.extract("d0", "note")
    check("majority voting keeps the stable extraction",
          len(out) == 1 and out[0].text == "metformin 500mg")
    check("one-off hallucinations are voted out and counted",
          ex.stats["voted_out"] == 1)
    check("reliability stats line reads like a verdict",
          "3 call(s)" in ex.stats_line()
          and "voted out" in ex.stats_line())

    # ---------- honest evaluation over a real corpus ----------
    blueprints = plan_corpus(spec)
    documents, _ = render_corpus(spec, blueprints, StubBackend())
    backend = Canned(['[]'])
    ex = LLMExtractor(backend, spec, name="mute-vendor")
    report = evaluate(blueprints, documents, ex)
    check("a vendor that extracts nothing scores an honest zero",
          report.overall_recall == 0.0
          and ex.stats["empty"] == len(documents))

    # An oracle-ish canned vendor: echo the planted medication
    # value per document, scripted from the blueprints (the test
    # cheats so the PLUMBING can be judged; a real LLM cannot).
    payloads = []
    for bp in blueprints:
        meds = [el.value for note in bp.notes
                for el in note.elements
                if el.element_id == "current_medication"
                and el.value]
        payloads.append(json.dumps(
            [{"category": "current_medication", "text": m}
             for m in meds]))
    backend = Canned(payloads + ['[]'])
    ex = LLMExtractor(backend, spec, name="scripted-oracle")
    report = evaluate(blueprints, documents, ex)
    check("a scripted-perfect vendor scores full medication "
          "recall through the adapter",
          report.elements["current_medication"].recall == 1.0)

    # ---------- full campaign ----------
    camp = compile_campaign("extract", reference_spec(size=8),
                            bars={"recall": 0.99})
    backend = Canned(['[]'])
    ex = LLMExtractor(backend, spec, name="mute-vendor")
    result = run_campaign(camp, ex, "mute-vendor")
    check("an LLMExtractor walks the extract campaign unchanged",
          result.highest_passed == 0
          and "CAMPAIGN [extract]" in result.format_text())

    # ---------- CLI wiring ----------
    import contextlib
    import io as _io
    import shutil
    import tempfile
    import synthkit.cli as cli_mod
    from synthkit.cli import main as cli
    tmp = Path(tempfile.mkdtemp(prefix="synthkit_v_"))
    spec_path = tmp / "doc.json"
    spec_path.write_text(reference_spec(size=6).to_json(),
                         encoding="utf-8")

    def run_cli(argv):
        out = _io.StringIO()
        err = _io.StringIO()
        with contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            try:
                rc = cli(argv)
            except SystemExit as e:
                rc = int(e.code or 0)
        return rc, out.getvalue(), err.getvalue()

    rc, out, _e = run_cli(["campaign-compile", "--goal",
                           "extract", "--spec", str(spec_path),
                           "-o", str(tmp / "c1")])
    check("extract campaign compiles for the vendor run",
          rc == 0)
    real_backend = cli_mod._backend
    cli_mod._backend = lambda name, model: Canned(['[]'])
    try:
        rc, out, _e = run_cli(["campaign-run", str(tmp / "c1"),
                               "--llm", "--name", "mute-llm"])
    finally:
        cli_mod._backend = real_backend
    check("CLI --llm runs the vendor ladder and prints "
          "reliability",
          rc == 1
          and "CAMPAIGN [extract]" in out
          and "vendor reliability:" in out)
    rc, _out, err = run_cli(["campaign-run", str(tmp / "c1")])
    check("campaign-run without a solver or --llm is refused "
          "with directions",
          rc == 2 and "--llm" in err)

    # ---------- GUI wiring ----------
    import json as _json
    import threading
    import urllib.request
    import synthkit.gui as gui
    server = gui.make_server(0)
    threading.Thread(target=server.serve_forever,
                     daemon=True).start()
    base = "http://127.0.0.1:{}".format(server.server_port)
    gui.BACKEND_FACTORY = lambda name, model: Canned(['[]'])
    try:
        req = urllib.request.Request(
            base + "/api/campaign-run",
            data=_json.dumps({
                "campaign_dir": str(tmp / "c1"),
                "solver": "llm_extract"}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = _json.loads(r.read().decode("utf-8"))
    finally:
        gui.BACKEND_FACTORY = None
        server.shutdown()
    check("GUI llm_extract solver walks the ladder with stats "
          "in the report",
          d.get("highest_passed") == 0
          and "vendor reliability:" in d.get("text", ""))
    shutil.rmtree(tmp)

    print("\nAll {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
