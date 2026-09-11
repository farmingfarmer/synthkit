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
    if name == "openai":
        from .openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(model=model or "local")
    if name == "bedrock":
        from .compiler import BedrockBackend
        if not model:
            print("bedrock requires --model <model_id>",
                  file=sys.stderr)
            sys.exit(2)
        return BedrockBackend(model_id=model)
    print("unknown backend: {}".format(name), file=sys.stderr)
    sys.exit(2)


def _cwd_importable():
    """Console entry points do not put the working directory on
    sys.path, so `synthkit evaluate --extractor mymod:fn` could
    never find mymod beside the spec. (Found live on the development machine the
    first time a custom extractor met the installed command.)"""
    import os
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def _load_extractor(dotted: str):
    _cwd_importable()
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


def cmd_table_compile(args) -> int:
    from .compiler import compile_table_spec
    description = args.description
    if args.file:
        description = Path(args.file).read_text(encoding="utf-8")
    if not description:
        print("provide -d/--description or -f/--file",
              file=sys.stderr)
        return 2
    backend = _backend(args.backend, args.model)
    result = compile_table_spec(description, backend, retries=1)
    out = Path(args.out)
    out.write_text(result.raw_json, encoding="utf-8")
    if result.ok:
        print("compiled OK -> {}".format(out))
        from .lint import lint_table
        print(lint_table(result.spec,
                         description=description).format_text())
        print("review the spec before rendering.")
        return 0
    print("compiled with problems -> {} (draft saved)".format(out))
    print("PROBLEMS:\n{}".format(result.problems))
    return 1


# ---- THE FITTED PATH ------------------------------------------
#
# It was a 1,526-line script under `scripts/`, which `pip install`
# does not ship - so the capability the README leads with was
# reachable only from a git checkout, and `synthkit --help` listed
# eighteen subcommands, none of them this one.
#
# The flags are NOT restated here. `pipeline.build_parser()` defines
# them once and argparse adopts it with `parents=`; two copies of a
# flag list is how they come to disagree, and this project already
# has a rule about a flag that is accepted in one place and ignored
# in another.


def cmd_fit(args) -> int:
    from .pipeline import main as pipeline_main
    return pipeline_main(args=args) or 0


def cmd_types(args) -> int:
    """The cheap look, promoted to its own verb.

    Every silent fault this tool has had was a column read as the
    wrong type, and the listing that catches them used to run after
    five minutes of discovery. It is seconds, and on a dataset nobody
    has opened it should be the first thing run."""
    from .pipeline import main as pipeline_main
    args.types_only = True
    return pipeline_main(args=args) or 0


def cmd_dials(args) -> int:
    """What is tunable on a blueprint, and what it currently is."""
    import json as _json
    from . import dials as _d
    try:
        bp = _json.loads(Path(args.blueprint).read_text(
            encoding="utf-8"))
    except (OSError, ValueError) as e:
        print("could not read {}: {}".format(args.blueprint, e),
              file=sys.stderr)
        return 2
    cols = bp.get("columns") or {}
    print("dials on {} ({} columns)".format(args.blueprint, len(cols)))
    print()
    print("  patients.{{{}}}".format(",".join(_d.PATIENT_DIALS)))
    for name in sorted(cols):
        c = cols[name]
        set_now = dict((k, v) for k, v in (c.get("dials") or {}).items()
                       if v is not None and v != "n/a")
        numeric = c.get("kind") == "numeric"
        avail = [x for x in _d.COLUMN_DIALS
                 if numeric or x not in ("shift", "scale")]
        print("  {:<34} {}{}".format(
            name, ",".join(avail),
            "" if not set_now else "   SET: " + _json.dumps(set_now)))
    print()
    print("Set one with:  synthkit fit --src ... --out ... --generate "
          "--dial COLUMN.NAME=VALUE")
    print("What you asked for and what arrived are both reported in "
          "findings.txt.")
    return 0


def cmd_version(args) -> int:
    from .gui import build_info
    info = build_info()
    print("synthkit v{version} \u00b7 built {built} \u00b7 "
          "{fingerprint}".format(**info))
    print("(the bench wordmark shows the same line; a match "
          "means it serves this install)")
    return 0


def cmd_corpus_lint(args) -> int:
    from .lint import lint_corpus
    from .spec import DataSpec
    spec = DataSpec.from_json(
        Path(args.spec).read_text(encoding="utf-8"))
    description = args.description
    if args.file:
        description = Path(args.file).read_text(encoding="utf-8")
    report = lint_corpus(spec, description=description)
    print(report.format_text())
    return 0 if report.ok else 1


def cmd_relational_render(args) -> int:
    from .relational import (RelationalSpec, plan_relational,
                             write_relational)
    spec = RelationalSpec.from_json(
        Path(args.spec).read_text(encoding="utf-8"))
    rbp = plan_relational(spec)
    run_dir = write_relational(Path(args.out), spec, rbp)
    print("relational -> {}".format(run_dir))
    for name, bp in rbp.blueprints.items():
        print("  {}: {} clean / {} dirty row(s)".format(
            name, len(bp.clean_rows), len(bp.dirty_rows)))
    print("  {} orphan(s) planted across {} link(s), ledgered"
          .format(len(rbp.link_ledger), len(spec.links)))
    return 0


def cmd_table_lint(args) -> int:
    from .lint import lint_table
    from .tablespec import TableSpec
    spec = TableSpec.from_json(
        Path(args.spec).read_text(encoding="utf-8"))
    description = args.description
    if args.file:
        description = Path(args.file).read_text(encoding="utf-8")
    report = lint_table(spec, description=description)
    print(report.format_text())
    return 0 if report.ok else 1


def cmd_table_plan(args) -> int:
    from .tableplan import plan_table
    from .tablespec import TableSpec, TableSpecError
    spec = TableSpec.from_json(
        Path(args.spec).read_text(encoding="utf-8"))
    try:
        bp = plan_table(spec)
    except TableSpecError as e:
        print("spec invalid:\n{}".format(e), file=sys.stderr)
        return 1
    ops = {}
    for m in bp.ledger:
        ops[m.op] = ops.get(m.op, 0) + 1
    print(json.dumps({
        "rows": len(bp.clean_rows),
        "dirty_rows": len(bp.dirty_rows),
        "columns": bp.columns,
        "mess_by_op": ops,
    }, indent=2))
    return 0


def cmd_table_render(args) -> int:
    from .tableplan import plan_table, write_table
    from .tablespec import TableSpec
    spec = TableSpec.from_json(
        Path(args.spec).read_text(encoding="utf-8"))
    bp = plan_table(spec)
    run_dir = write_table(Path(args.out), spec, bp)
    print("table -> {} ({} dirty row(s), {} mess cell(s))".format(
        run_dir, len(bp.dirty_rows),
        len([m for m in bp.ledger if m.op != "duplicate"])))
    return 0


def cmd_table_evaluate(args) -> int:
    from .tableeval import evaluate_cleaning
    from .tableplan import load_table
    _spec, bp = load_table(Path(args.run_dir))
    if ":" not in args.cleaner:
        print("cleaner must be 'package.module:function'",
              file=sys.stderr)
        return 2
    mod_name, fn_name = args.cleaner.split(":", 1)
    fn = getattr(importlib.import_module(mod_name), fn_name)
    cleaned = fn([dict(r) for r in bp.dirty_rows])
    report = evaluate_cleaning(bp, cleaned, args.cleaner)
    print(report.format_text())
    if args.json_out:
        Path(args.json_out).write_text(report.to_json(),
                                       encoding="utf-8")
    return 0


def _parse_bars(raw: str):
    bars = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        key, _, val = part.partition("=")
        bars[key.strip()] = float(val)
    return bars


def cmd_campaign_compile(args) -> int:
    from .campaign import CampaignError, compile_campaign, \
        write_campaign
    raw = Path(args.spec).read_text(encoding="utf-8")
    if args.goal in ("clean", "predict", "regress"):
        from .tablespec import TableSpec
        base = TableSpec.from_json(raw)
    else:
        from .spec import DataSpec
        base = DataSpec.from_json(raw)
    try:
        camp = compile_campaign(args.goal, base,
                                bars=_parse_bars(args.bars),
                                outcome=args.outcome)
    except CampaignError as e:
        print("campaign invalid: {}".format(e), file=sys.stderr)
        return 1
    run_dir = write_campaign(Path(args.out), camp)
    print("campaign [{}] -> {} ({} tier(s))".format(
        camp.goal, run_dir, len(camp.tiers)))
    for i, tier in enumerate(camp.tiers):
        print("  tier {} `{}`: {}".format(i + 1, tier.name,
                                          tier.notes))
    return 0


def cmd_campaign_run(args) -> int:
    from .campaign import load_campaign, run_campaign, \
        write_campaign
    _cd = Path(args.campaign_dir)
    if not (_cd / "manifest.json").exists():
        # the operator hit this as a raw FileNotFoundError after a
        # one-letter path typo - compile had written to a sibling
        # directory and nothing said so
        print("STOPPED: no manifest.json in {} - either "
              "campaign-compile has not run, or it wrote to a "
              "differently-spelled directory. `dir {}` shows what "
              "is actually there.".format(_cd, _cd.parent),
              file=sys.stderr)
        return 2
    camp = load_campaign(_cd)
    extractor = None
    if args.llm:
        if camp.goal != "extract":
            print("--llm runs an LLM vendor on extract "
                  "campaigns only", file=sys.stderr)
            return 2
        from .llmvendor import LLMExtractor
        from .spec import DataSpec
        spec = DataSpec.from_json(camp.tiers[0].spec_json)
        extra = args.llm_extra_system
        if extra.startswith("@"):
            extra = Path(extra[1:]).read_text(encoding="utf-8")
        extractor = LLMExtractor(
            _backend(args.llm_backend, args.llm_model), spec,
            samples=args.llm_samples,
            name=args.name or "llm-vendor",
            extra_system=extra)
        solver = extractor
    elif ":" in args.solver:
        mod_name, fn_name = args.solver.split(":", 1)
        _cwd_importable()
        solver = getattr(importlib.import_module(mod_name),
                         fn_name)
    elif not args.solver:
        print("pass --solver (a registry name like autosolver, or "
              "package.module:function) or --llm", file=sys.stderr)
        return 2
    else:
        # the registry names the bench uses (autosolver,
        # autosolver_hybrid, ...) work here too - the operator on
        # the data machine should not need a dotted path for the
        # built-ins
        from .gui import _solver as _gui_solver
        try:
            solver = _gui_solver(args.solver)
        except ValueError as e:
            print("STOPPED: {}".format(e), file=sys.stderr)
            return 2
    arm = args.name or args.solver or "llm-vendor"
    result = run_campaign(camp, solver, solver_name=arm)
    write_campaign(Path(args.campaign_dir), camp, result,
                   result_name=arm)
    print(result.format_text())
    if extractor is not None:
        print(extractor.stats_line())
    return 0 if result.highest_passed == len(
        result.tier_results) else 1


def cmd_showdown(args) -> int:
    from .autosolver import run_showdown
    from .campaign import load_campaign
    _cd = Path(args.campaign_dir)
    if not (_cd / "manifest.json").exists():
        print("STOPPED: no manifest.json in {} - either "
              "campaign-compile has not run, or it wrote to a "
              "differently-spelled directory. `dir {}` shows what "
              "is actually there.".format(_cd, _cd.parent),
              file=sys.stderr)
        return 2
    camp = load_campaign(_cd)
    if ":" in args.solver:
        _cwd_importable()
        mod_name, fn_name = args.solver.split(":", 1)
        fn = getattr(importlib.import_module(mod_name), fn_name)
    else:
        from .gui import _solver as _gui_solver
        try:
            fn = _gui_solver(args.solver)
        except ValueError as e:
            print("STOPPED: {}".format(e), file=sys.stderr)
            return 2
    kwargs = {}
    if args.baseline:
        from .gui import _solver as _gui_solver
        kwargs["baseline"] = _gui_solver(args.baseline)
        kwargs["baseline_name"] = args.baseline
    result = run_showdown(camp, fn,
                          vendor_name=args.name or args.solver,
                          **kwargs)
    print(result.format_text())
    if args.json_out:
        import json as _json
        Path(args.json_out).write_text(_json.dumps({
            "vendor": result.vendor_name,
            "tiers": [{"tier": t.tier, "ceiling": t.ceiling,
                       "baseline": t.baseline_auroc,
                       "vendor": t.vendor_auroc,
                       "passed": t.vendor_passed}
                      for t in result.tiers],
        }, indent=2), encoding="utf-8")
    return 0


def cmd_gui(args) -> int:
    from .gui import run_gui
    run_gui(port=args.port, open_browser=not args.no_browser)
    return 0


def _console_safe():
    """Windows consoles may be cp1252/cp437; never let a
    middle-dot crash a command."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def cmd_bridge(args) -> int:
    """A fitted run -> TableSpec, without refitting. `fit
    --emit-spec` does this at fit time; a 40-minute refit to get a
    file the blueprint already implies is the kind of round trip
    this CLI exists to remove."""
    import json as _json
    from .bridge import blueprint_to_tablespec
    run = Path(args.rundir)
    bp_path = run / "blueprint.json"
    if not bp_path.exists():
        print("no blueprint.json in {} - run `synthkit fit` "
              "first".format(run), file=sys.stderr)
        return 2
    bp = _json.loads(bp_path.read_text(encoding="utf-8"))
    made = blueprint_to_tablespec(bp, title=run.name)
    out = Path(args.out) if args.out else run / "tablespec.json"
    out.write_text(_json.dumps(made, indent=1), encoding="utf-8")
    car = made["carried"]
    print("wrote {} - {} column(s) crossed, {} dropped".format(
        out, len(made["tablespec"]["columns"]),
        len(car["columns_dropped"])))
    print("did NOT cross: {}".format(
        ", ".join(car["did_not_cross"]) or "(nothing)"))
    return 0


def cmd_plant(args) -> int:
    """Plant a KNOWN outcome on a bridged spec - the step that lets
    the two halves meet with an answer key. Effects are declared in
    STANDARD DEVIATIONS of each covariate; `col=0.8` or, for a
    categorical level, `col=LEVEL=0.8`."""
    import json as _json
    from . import semisynth
    raw = _json.loads(Path(args.spec).read_text(encoding="utf-8"))
    bridged = raw if "tablespec" in raw else {"tablespec": raw}
    effects = {}
    for e in (args.effect or []):
        name, beta = e.rsplit("=", 1)
        try:
            effects[name] = float(beta)
        except ValueError:
            print("--effect needs col=BETA or col=LEVEL=BETA; got "
                  "{!r}".format(e), file=sys.stderr)
            return 2
    if not effects:
        print("at least one --effect is required - an exam with "
              "no planted signal grades nothing", file=sys.stderr)
        return 2
    try:
        planted = semisynth.plant(
            bridged, effects, name=args.outcome_name,
            kind=args.kind, prevalence=args.prevalence)
    except ValueError as e:
        # all effects refused raises - a sentence, not a stack
        print("STOPPED: {}".format(e), file=sys.stderr)
        return 2
    blk = planted.get("planted") or {}
    used = blk.get("effects_in_sds") or {}
    for name, beta in used.items():
        print("planted {}: {:+.2f} sd".format(name, float(beta)))
    for r in blk.get("refused") or []:
        print("REFUSED {}: {}".format(r["effect"], r["why"]))
    cal = blk.get("calibration") or {}
    if cal:
        print("requested prevalence {:.0%}; the intercept was "
              "SOLVED, not centered - achieved is reported by "
              "semisynth.verify after render".format(
                  float(blk.get("requested_prevalence") or 0)))
    if not used:
        print("every effect was refused - nothing planted",
              file=sys.stderr)
        return 2
    out = Path(args.out)
    out.write_text(_json.dumps(planted["tablespec"], indent=1),
                   encoding="utf-8")
    rec = out.with_suffix(".planted.json")
    rec.write_text(_json.dumps(planted, indent=1),
                   encoding="utf-8")
    print("wrote {} (the exam spec) and {} (the full record - "
          "the answer key lives there)".format(out, rec))
    return 0


def main(argv=None) -> int:
    _console_safe()
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

    p = sub.add_parser("table-compile",
                       help="plain English -> TableSpec JSON")
    p.add_argument("-d", "--description", default="")
    p.add_argument("-f", "--file", default="")
    p.add_argument("-o", "--out", default="table_spec.json")
    p.add_argument("--backend", default="ollama")
    p.add_argument("--model", default="")
    p.set_defaults(fn=cmd_table_compile)

    from .pipeline import build_parser as _fit_parser
    p = sub.add_parser(
        "fit", parents=[_fit_parser(add_help=False)],
        help="learn a blueprint from a real extract, and optionally "
             "generate from it")
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser(
        "types", parents=[_fit_parser(add_help=False)],
        help="how every column was read, in seconds - run this "
             "first on a dataset nobody has opened")
    p.set_defaults(fn=cmd_types)

    p = sub.add_parser("dials",
                       help="what is tunable on a blueprint")
    p.add_argument("blueprint")
    p.set_defaults(fn=cmd_dials)

    p = sub.add_parser("version",
                       help="version, build date, and "
                            "fingerprint of this install")
    p.set_defaults(fn=cmd_version)

    p = sub.add_parser("corpus-lint",
                       help="semantic lint for document specs")
    p.add_argument("spec")
    p.add_argument("-d", "--description", default="")
    p.add_argument("-f", "--file", default="")
    p.set_defaults(fn=cmd_corpus_lint)

    p = sub.add_parser("relational-render",
                       help="plan and write a multi-table run "
                            "with join mess")
    p.add_argument("spec")
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_relational_render)

    p = sub.add_parser("table-lint",
                       help="semantic lint: does the spec mean "
                            "what you meant?")
    p.add_argument("spec")
    p.add_argument("-d", "--description", default="")
    p.add_argument("-f", "--file", default="")
    p.set_defaults(fn=cmd_table_lint)

    p = sub.add_parser("table-plan",
                       help="validate a table spec, print stats")
    p.add_argument("spec")
    p.set_defaults(fn=cmd_table_plan)

    p = sub.add_parser("table-render",
                       help="table spec -> dirty/clean/ledger run")
    p.add_argument("spec")
    p.add_argument("-o", "--out", default="tables/run_001")
    p.set_defaults(fn=cmd_table_render)

    p = sub.add_parser("table-evaluate",
                       help="table run + cleaner -> sliced report")
    p.add_argument("run_dir")
    p.add_argument("--cleaner", required=True,
                   help="dotted path 'pkg.mod:function'")
    p.add_argument("--json-out", default="")
    p.set_defaults(fn=cmd_table_evaluate)

    p = sub.add_parser("campaign-compile",
                       help="goal + base spec -> tier ladder")
    p.add_argument("--goal", required=True,
                   choices=["clean", "predict", "regress",
                            "extract"])
    p.add_argument("--spec", required=True)
    p.add_argument("--outcome", default="")
    p.add_argument("--bars", default="",
                   help="k=v pairs, e.g. fix_rate=0.9,auroc=0.7")
    p.add_argument("-o", "--out", default="campaigns/run_001")
    p.set_defaults(fn=cmd_campaign_compile)

    p = sub.add_parser("gui",
                       help="launch the calibration bench "
                            "(local web UI)")
    p.add_argument("--port", type=int, default=8377)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(fn=cmd_gui)

    p = sub.add_parser("showdown",
                       help="vendor vs the synthkit baseline up "
                            "a predict ladder")
    p.add_argument("campaign_dir")
    p.add_argument("--solver", required=True,
                   help="dotted path 'pkg.mod:function'")
    p.add_argument("--name", default="")
    p.add_argument("--json-out", default="")
    p.add_argument("--baseline", default="",
                   help="baseline solver (registry name or pkg.mod:fn); default autosolver")
    p.set_defaults(fn=cmd_showdown)

    p = sub.add_parser("trials",
                       help="compare recorded arms of a campaign")
    p.add_argument("campaign_dir")
    p.set_defaults(fn=lambda a: (
        print(__import__("synthkit.campaign",
                         fromlist=["format_trials"])
              .format_trials(Path(a.campaign_dir))) or 0))

    p = sub.add_parser("campaign-run",
                       help="run a solver up the ladder")
    p.add_argument("campaign_dir")
    p.add_argument("--solver", default="",
                   help="dotted path 'pkg.mod:function'")
    p.add_argument("--llm", action="store_true",
                   help="run an LLM as the vendor "
                        "(extract campaigns)")
    p.add_argument("--llm-backend", default="ollama")
    p.add_argument("--llm-model", default="")
    p.add_argument("--llm-samples", type=int, default=1)
    p.add_argument("--llm-extra-system", default="",
                   help="intervention text appended to the "
                        "vendor system prompt (@file to read "
                        "from a file)")
    p.add_argument("--name", default="")
    p.set_defaults(fn=cmd_campaign_run)

    p = sub.add_parser("bridge",
                       help="fitted run -> tablespec.json, without "
                            "refitting")
    p.add_argument("rundir")
    p.add_argument("-o", "--out", default=None)
    p.set_defaults(fn=cmd_bridge)

    p = sub.add_parser("plant",
                       help="plant a known outcome on a bridged "
                            "spec - effects in sd, e.g. --effect "
                            "creatinine=0.8")
    p.add_argument("--spec", required=True,
                   help="tablespec.json from `synthkit bridge`")
    p.add_argument("--effect", action="append", default=[],
                   help="col=BETA or col=LEVEL=BETA, repeatable")
    p.add_argument("--outcome-name", default="outcome")
    p.add_argument("--kind", default="logistic",
                   choices=["logistic", "linear"])
    p.add_argument("--prevalence", type=float, default=0.25)
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_plant)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
