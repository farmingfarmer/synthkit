"""Build the STANDALONE synthkit notebook: every module inlined as a
cell, in dependency order, with intra-package imports stripped
(one shared namespace needs none), followed by the live walkthrough
and a validation cell. The notebook is GENERATED from the module
sources — regenerate after any library change and the demo cannot
drift:

    python scripts/build_notebook.py
    -> notebooks/synthkit_standalone.ipynb

Python 3.8 compatible.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "synthkit"
OUT = REPO / "notebooks" / "synthkit_standalone.ipynb"

# Dependency order matters: each cell may only use names defined by
# earlier cells.
MODULES = [
    ("spec.py", "The DataSpec — the reviewable contract"),
    ("compiler.py", "LLM backends (Ollama / Anthropic / Bedrock) "
                    "and the spec compiler"),
    ("planner.py", "The Planner — deterministic blueprints, "
                   "ground truth first"),
    ("renderer.py", "The Renderer — verified generation with "
                    "targeted retries and fallback"),
    ("corpus_io.py", "Corpus I/O — the reproducible, "
                     "integrity-checked run artifact"),
    ("evaluator.py", "The Evaluator — exact, sliced scoring "
                     "against planted truth"),
    ("harness.py", "The Harness — hypotheses become measured "
                   "experiments"),
    ("examples.py", "Examples — the reference vertical and a "
                    "naive real extractor"),
]

_REL_IMPORT_RE = re.compile(r"^\s*from\s+\.[\w.]*\s+import\b")


def strip_relative_imports(source: str) -> str:
    """Remove intra-package imports (top-level AND function-local),
    consuming multi-line parenthesized forms."""
    out = []
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if _REL_IMPORT_RE.match(line):
            depth = line.count("(") - line.count(")")
            while depth > 0:
                i += 1
                depth += lines[i].count("(") - lines[i].count(")")
            out.append(re.match(r"^\s*", line).group(0)
                       + "pass  # intra-package import inlined above")
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def md(*lines):
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in lines]}


def code(source: str):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in source.splitlines()]}


def build() -> dict:
    cells = [
        md("# synthkit — fully self-contained notebook",
           "",
           "The ENTIRE library inlined below, generated from the",
           "module sources by `scripts/build_notebook.py` — nothing",
           "to install, no repo required: run top to bottom in any",
           "JupyterLab (SageMaker Studio included).",
           "",
           "Pipeline: plain-English spec -> deterministic planning",
           "(blueprints ARE the ground truth) -> verified rendering",
           "(code verifier around every LLM call, deterministic",
           "fallback) -> exact sliced evaluation -> measured",
           "experiments.",
           "",
           "PHI posture: there is NO ingestion path anywhere below.",
           "Distributions are specified, never fitted; nothing real",
           "ever enters the generator."),
    ]
    for fname, title in MODULES:
        src = (PKG / fname).read_text(encoding="utf-8")
        cells.append(md("## Library: {}".format(title),
                        "",
                        "*Source of truth: `synthkit/{}` — this cell "
                        "is generated, not hand-edited.*".format(fname)))
        cells.append(code(strip_relative_imports(src)))

    cells += [
        md("---", "# Walkthrough", "",
           "Everything below runs deterministically (StubBackend).",
           "Flip `RUN_BEDROCK = True` for realistic prose (needs",
           "`bedrock:InvokeModel` on the execution role)."),
        code("RUN_BEDROCK = False\n"
             "BEDROCK_MODEL_ID = \"anthropic.claude-sonnet-4-6-v1:0\"\n"
             "BEDROCK_REGION = \"us-west-2\"\n"
             "CORPUS_DIR = \"corpus/notebook_run_001\""),
        md("## 1. Spec and plan"),
        code("spec = reference_spec(size=20, master_seed=42)\n"
             "spec.validate()\n"
             "blueprints = plan_corpus(spec)\n"
             "import json as _json\n"
             "print(_json.dumps(corpus_stats(blueprints), indent=2))"),
        md("## 2. Verified render (deterministic)"),
        code("documents, render_report = render_corpus(\n"
             "    spec, blueprints, StubBackend())\n"
             "print(render_report.format_text())\n"
             "print()\n"
             "print(documents[\"doc_00000\"][:400])"),
        md("## 3. Evaluate the naive extractor",
           "",
           "Replace `regex_extract` with an adapter around any real",
           "model: `(doc_id, text) -> [Extraction]`."),
        code("report = evaluate(\n"
             "    blueprints, documents,\n"
             "    FunctionExtractor(regex_extract, \"regex-naive\"),\n"
             "    reference_rules())\n"
             "print(report.format_text())"),
        md("## 4. A measured experiment"),
        code("exp = Experiment(\n"
             "    name=\"allergy bar + discontinued-med trap\",\n"
             "    spec=spec,\n"
             "    extractor=FunctionExtractor(regex_extract,\n"
             "                                \"regex-naive\"),\n"
             "    conditions=[\n"
             "        Condition.parse(\n"
             "            \"elements.allergy_flag.recall >= 0.85\"),\n"
             "        Condition.parse(\n"
             "            \"distractors.discontinued_medication\"\n"
             "            \".fp_rate <= 0.10\"),\n"
             "    ],\n"
             "    rules=reference_rules(),\n"
             ")\n"
             "result = run_experiment(exp)\n"
             "print(result.finding())"),
        md("## 5. Persist and reload with integrity"),
        code("from pathlib import Path as _P\n"
             "run_dir = write_corpus(_P(CORPUS_DIR), spec,\n"
             "                       blueprints, documents,\n"
             "                       render_report, \"stub\")\n"
             "_s2, _b2, _d2, _m = load_corpus(run_dir)\n"
             "assert _d2 == documents\n"
             "print(\"corpus verified ->\", run_dir)"),
        md("## 6. In-notebook open-source LLM (gated)",
           "",
           "Generation INSIDE this notebook's process — no server,",
           "no service. First run downloads weights from the",
           "Hugging Face hub (~1-3GB; use a local/S3 path as",
           "`model_id` in air-gapped environments). The 0.5B model",
           "runs on CPU instances; prefer a GPU instance for 1.5B+.",
           "The verifier + retry + fallback wrap it like any",
           "backend — a weak model degrades measurably, never",
           "breaks the corpus."),
        code("RUN_LOCAL_LLM = False\n"
             "LOCAL_MODEL_ID = \"Qwen/Qwen2.5-0.5B-Instruct\"\n"
             "if RUN_LOCAL_LLM:\n"
             "    import sys as _sys\n"
             "    !{_sys.executable} -m pip install -q transformers "
             "torch accelerate\n"
             "    hf_backend = HFLocalBackend(model_id=LOCAL_MODEL_ID)\n"
             "    hf_docs, hf_rr = render_corpus(spec, blueprints,\n"
             "                                   hf_backend)\n"
             "    print(hf_rr.format_text())\n"
             "    print(evaluate(\n"
             "        blueprints, hf_docs,\n"
             "        FunctionExtractor(regex_extract,\n"
             "                          \"regex-naive\"),\n"
             "        reference_rules()).format_text())\n"
             "else:\n"
             "    print(\"RUN_LOCAL_LLM is False — skipped.\")"),
        md("## 7. Bedrock (gated) — realistic prose"),
        code("if RUN_BEDROCK:\n"
             "    backend = BedrockBackend(\n"
             "        model_id=BEDROCK_MODEL_ID,\n"
             "        region=BEDROCK_REGION)\n"
             "    live_docs, live_rr = render_corpus(\n"
             "        spec, blueprints, backend)\n"
             "    print(live_rr.format_text())\n"
             "    print(evaluate(\n"
             "        blueprints, live_docs,\n"
             "        FunctionExtractor(regex_extract,\n"
             "                          \"regex-naive\"),\n"
             "        reference_rules()).format_text())\n"
             "else:\n"
             "    print(\"RUN_BEDROCK is False — skipped.\")"),
        md("## 8. Self-validation",
           "",
           "A compact assertion suite over the inlined library —",
           "the notebook proves itself on every full run."),
        code("corpus_b = plan_corpus(reference_spec(size=20,\n"
             "                                      master_seed=42))\n"
             "assert all(a.to_json() == b.to_json()\n"
             "           for a, b in zip(blueprints, corpus_b)), \\\n"
             "    \"determinism\"\n"
             "def _perfect(doc_id, text):\n"
             "    bp = next(b for b in blueprints\n"
             "              if b.doc_id == doc_id)\n"
             "    cats = {\"current_medication\":\n"
             "            \"current medication\",\n"
             "            \"followup_appointment\": \"follow-up\",\n"
             "            \"allergy_flag\": \"allergy\"}\n"
             "    return [Extraction(cats[e.element_id],\n"
             "                       e.value or e.phrasing)\n"
             "            for e in bp.notes[0].elements]\n"
             "_r = evaluate(blueprints, documents,\n"
             "              FunctionExtractor(_perfect, \"perfect\"),\n"
             "              reference_rules())\n"
             "assert abs(_r.overall_recall - 1.0) < 1e-9, \\\n"
             "    \"perfect recall\"\n"
             "assert _r.distractors[\n"
             "    \"discontinued_medication\"].false_positives == 0\n"
             "assert render_report.fallbacks == 0, \"verified render\"\n"
             "v = verify_note(documents[\"doc_00000\"],\n"
             "                blueprints[0].notes[0],\n"
             "                spec.unstructured_fields[0])\n"
             "assert v.ok, \"verifier\"\n"
             "print(\"SELF-VALIDATION: all assertions passed.\")"),
    ]
    return {"cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3",
                               "language": "python",
                               "name": "python3"},
                "language_info": {"name": "python",
                                  "version": "3.10"},
            },
            "nbformat": 4, "nbformat_minor": 5}


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    n_code = sum(1 for c in nb["cells"]
                 if c["cell_type"] == "code")
    print("wrote {} ({} cells, {} code)".format(
        OUT.relative_to(REPO), len(nb["cells"]), n_code))
    return 0


if __name__ == "__main__":
    sys.exit(main())
