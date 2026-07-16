"""SYNTH_V1 CLI: the demo-ready surface over the library.

    synthkit compile  -d "fifty progress notes..." -o spec.json
    synthkit plan     spec.json
    synthkit render   spec.json -o corpus/run_001 --backend ollama
    synthkit evaluate corpus/run_001 --extractor pkg.mod:fn
    synthkit experiment --backend stub

Backends: stub (deterministic, no LLM), ollama (local, default
model mistral-small3.1), anthropic, bedrock. Extractors load by
dotted path 'package.module:function' — any callable
(doc_id, text) -> [Extraction].

Python 3.8 compatible.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def _backend(name: str, model: str = ""):
    from .harness import StubBackend
    if name == "stub":
        return StubBackend()
    if name == "ollama":
        from .compiler import OllamaBackend
        return OllamaBackend(model=model or "mistral-small3.1")
    if name == "anthropic":
        from .compiler import AnthropicBackend
        return AnthropicBackend(model=model or "claude-sonnet-4-6")
    if name == "hf":
        from .compiler import HFLocalBackend
        return HFLocalBackend(
            model_id=model or "Qwen/Qwen2.5-1.5B-Instruct")
    if name == "bedrock":
        from .compiler import BedrockBackend
        if not model:
            print("bedrock requires --model <model_id>",
                  file=sys.stderr)
            sys.exit(2)
        return BedrockBackend(model_id=model)
    print("unknown backend: {}".format(name), file=sys.stderr)
    sys.exit(2)


def _load_extractor(dotted: str):
    from .evaluator import FunctionExtractor
    if ":" not in dotted:
        print("extractor must be 'package.module:function'",
              file=sys.stderr)
        sys.exit(2)
    mod_name, fn_name = dotted.split(":", 1)
    fn = getattr(importlib.import_module(mod_name), fn_name)
    return FunctionExtractor(fn, dotted)


def cmd_compile(args) -> int:
    from .compiler import compile_spec
    description = args.description
    if args.file:
        description = Path(args.file).read_text(encoding="utf-8")
    if not description:
        print("provide -d/--description or -f/--file",
              file=sys.stderr)
        return 2
    backend = _backend(args.backend, args.model)
    result = compile_spec(description, backend, retries=1)
    out = Path(args.out)
    out.write_text(result.raw_json, encoding="utf-8")
    if result.ok:
        print("compiled OK -> {}".format(out))
        print("review the spec before rendering.")
        return 0
    print("compiled with problems -> {} (draft saved)".format(out))
    print("PROBLEMS:\n{}".format(result.problems))
    print("fix the JSON by hand, then `synthkit plan {}`.".format(out))
    return 1


def cmd_plan(args) -> int:
    from .planner import corpus_stats, plan_corpus
    from .spec import DataSpec, SpecError
    spec = DataSpec.from_json(
        Path(args.spec).read_text(encoding="utf-8"))
    try:
        blueprints = plan_corpus(spec)
    except SpecError as e:
        print("spec invalid:\n{}".format(e), file=sys.stderr)
        return 1
    print(json.dumps(corpus_stats(blueprints), indent=2))
    return 0


def cmd_render(args) -> int:
    from .corpus_io import write_corpus
    from .planner import plan_corpus
    from .renderer import render_corpus
    from .spec import DataSpec
    spec = DataSpec.from_json(
        Path(args.spec).read_text(encoding="utf-8"))
    blueprints = plan_corpus(spec)
    backend = _backend(args.backend, args.model)
    print("rendering {} document(s) via {}...".format(
        len(blueprints), backend.name))
    documents, report = render_corpus(spec, blueprints, backend,
                                      attempts=args.attempts)
    print(report.format_text())
    run_dir = write_corpus(Path(args.out), spec, blueprints,
                           documents, report,
                           backend_name=backend.name)
    print("corpus -> {}".format(run_dir))
    return 0


def cmd_evaluate(args) -> int:
    from .corpus_io import load_corpus
    from .evaluator import MatchRule, evaluate
    spec, blueprints, documents, _m = load_corpus(Path(args.run_dir))
    extractor = _load_extractor(args.extractor)
    rules = None
    if args.rules:
        raw = json.loads(Path(args.rules).read_text(encoding="utf-8"))
        rules = {k: MatchRule(**v) for k, v in raw.items()}
    elif args.reference_rules:
        from .examples import reference_rules
        rules = reference_rules()
    report = evaluate(blueprints, documents, extractor, rules)
    print(report.format_text())
    if args.json_out:
        Path(args.json_out).write_text(report.to_json(),
                                       encoding="utf-8")
        print("report -> {}".format(args.json_out))
    return 0


def cmd_experiment(args) -> int:
    from .examples import allergy_bar_experiment
    from .harness import run_experiment
    extractor_fn = None
    if args.extractor:
        extractor_fn = _load_extractor(args.extractor)._fn
    exp = allergy_bar_experiment(
        backend=_backend(args.backend, args.model),
        size=args.size, extractor_fn=extractor_fn)
    print("running experiment '{}' ({} docs, backend {})...".format(
        exp.name, args.size, args.backend))
    result = run_experiment(exp)
    print(result.finding())
    print()
    print(result.eval_report.format_text())
    return 0 if result.passed else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="synthkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("compile",
                       help="plain English -> DataSpec JSON")
    p.add_argument("-d", "--description", default="")
    p.add_argument("-f", "--file", default="")
    p.add_argument("-o", "--out", default="spec.json")
    p.add_argument("--backend", default="ollama")
    p.add_argument("--model", default="")
    p.set_defaults(fn=cmd_compile)

    p = sub.add_parser("plan",
                       help="validate a spec, print corpus stats")
    p.add_argument("spec")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("render",
                       help="spec -> verified corpus on disk")
    p.add_argument("spec")
    p.add_argument("-o", "--out", default="corpus/run_001")
    p.add_argument("--backend", default="ollama")
    p.add_argument("--model", default="")
    p.add_argument("--attempts", type=int, default=3)
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("evaluate",
                       help="corpus + extractor -> sliced report")
    p.add_argument("run_dir")
    p.add_argument("--extractor", required=True,
                   help="dotted path 'pkg.mod:function'")
    p.add_argument("--rules", default="",
                   help="rules JSON: {id: {mode, categories}}")
    p.add_argument("--reference-rules", action="store_true",
                   help="use the reference vertical's rules")
    p.add_argument("--json-out", default="")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("experiment",
                       help="run the canned allergy-bar experiment")
    p.add_argument("--backend", default="stub")
    p.add_argument("--model", default="")
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--extractor", default="",
                   help="dotted path; default: the naive regex")
    p.set_defaults(fn=cmd_experiment)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
