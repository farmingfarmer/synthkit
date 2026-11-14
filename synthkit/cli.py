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


def cmd_forecast(args) -> int:
    """What can this dataset support? - asked BEFORE generating.

    The operator's question was "tell me what I can and cannot do in
    terms of levers". The answer is structural and cheap: how much of
    this source clears the privacy floor, which is what decides
    fidelity. Output SIZE is reported as what it is - measured to 5x,
    unmeasured above it - because a forecast that invents a percentage
    is worse than one that admits the gap."""
    import pandas as pd
    from .forecast import census, plain, sizes
    df = pd.read_csv(args.src, dtype=str, keep_default_na=False)
    df = df.replace("", None)
    cen = census(df, args.group_by, k=args.k)
    mult = [float(x) for x in args.multipliers.split(",")] \
        if args.multipliers else None
    print(plain(cen, sizes(cen, mult)))
    return 0


def cmd_types(args) -> int:
    """The cheap look, promoted to its own verb.

    Every silent fault this tool has had was a column read as the
    wrong type, and the listing that catches them used to run after
    five minutes of discovery. It is seconds, and on a dataset nobody
    has opened it should be the first thing run."""
    import tempfile
    from .pipeline import main as pipeline_main
    args.types_only = True
    if getattr(args, "out", None):
        return pipeline_main(args=args) or 0
    with tempfile.TemporaryDirectory() as td:
        args.out = td
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
    rels = bp.get("relationships") or []
    if rels:
        print()
        print("  relationships ({} edge(s)) - dial: strength"
              .format(sum(len(r.get("parents") or [])
                          for r in rels)))
        shown = 0
        for r in rels:
            for par in (r.get("parents") or []):
                st = (r.get("dials") or {}).get("strength")
                print("  {}<-{}{}".format(
                    r.get("child"), par,
                    "" if st is None
                    else "   SET: strength={}".format(st)))
                shown += 1
                if shown >= 20:
                    break
            if shown >= 20:
                break
        rest = sum(len(r.get("parents") or []) for r in rels) - shown
        if rest > 0:
            print("  ... and {} more edge(s)".format(rest))
    print()
    print("Set one with:  synthkit fit --src ... --out ... --generate "
          "--dial COLUMN.NAME=VALUE")
    print("A relationship: --dial \"CHILD<-PARENT.strength=0.5\" "
          "(0 removes it; the achieved correlation is reported)")
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


def cmd_batch(args) -> int:
    """Source in, artifacts out, nothing else.

    The whole point of this command is the SHAPE: one
    direction for data, one allow-list for results, and a
    dry run that says what would leave before anything does.
    """
    import tempfile
    from . import cloud
    print("SYNTHKIT BATCH")
    for line in cloud.describe():
        print("  {}".format(line))
    print("")
    work = Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as _tmp:
            local_src = cloud.fetch_source(args.src, _tmp)
            print("  source: {}".format(
                "pulled into the compute"
                if cloud.is_s3(args.src) else "read in place"))
            if args.dry_run:
                print("  DRY RUN - the fit was not started. "
                      "What WOULD be published from {}: {}"
                      .format(work, ", ".join(cloud.ALLOWED)))
                return 0
            from .pipeline import main as _fit
            argv = ["--src", str(local_src),
                    "--out", str(work),
                    "--group-by", args.group_by,
                    "--generate"]
            if args.max_rows:
                argv += ["--max-rows", str(args.max_rows)]
            rc = _fit(argv)
            if rc:
                print("STOPPED: the fit exited {} - nothing "
                      "is published from a run that did not "
                      "finish".format(rc), file=sys.stderr)
                return rc
    except cloud.CloudRefusal as e:
        print("STOPPED: {}".format(e), file=sys.stderr)
        return 2
    if not args.publish:
        print("")
        print("  nothing published - pass --publish to send "
              "the artifacts somewhere")
        return 0
    try:
        wrote = cloud.publish(work, args.publish)
    except cloud.CloudRefusal as e:
        print("STOPPED: {}".format(e), file=sys.stderr)
        return 2
    print("")
    print("  published {} artifact(s):".format(len(wrote)))
    for w in wrote:
        print("    {}".format(w))
    return 0


def cmd_receipts(args) -> int:
    """One receipt per published claim, from a run's artifacts.

    Goal 3 is "every pattern found, explained, with receipts" and
    the receipts were the part that stayed PLANNED. The evidence
    existed - skill and support in `catalogue.json`, the effect
    shape and importances in `blueprint.json`, survival in
    `fidelity.json` - spread across three files, two of them
    prose, so auditing ONE pattern meant holding all three in
    your head. This assembles it per claim and recomputes
    nothing.
    """
    import os
    from . import receipts
    try:
        doc = receipts.build(args.rundir)
    except ValueError as e:
        print("STOPPED: {}".format(e), file=sys.stderr)
        return 2
    print(receipts.render(doc))
    dest = args.json_out or os.path.join(args.rundir,
                                         "receipts.json")
    import json as _json
    with open(dest, "w") as fh:
        _json.dump(doc, fh, indent=2, default=str)
    print("")
    print("wrote {}".format(dest))
    return 0


def cmd_determination(args) -> int:
    """The evidence package for a HIPAA Expert Determination.

    THE EVIDENCE EXISTED AND NOBODY HAD ASSEMBLED IT.
    `docs/COMPETITIVE.md` lists Expert Determination as a flat
    `no` against a vendor that has one, and notes it is "not a
    model change: a documentation and evidence package" - most
    of which this project has measured for months, spread
    across six files. A reader deciding whether the
    re-identification risk is very small had to hold all of it
    at once.

    IT REFUSES TO CALL ITSELF A DETERMINATION. 164.514(b)(1)
    puts the determination in a qualified person's hands; a tool
    claiming one would be the same fault as an output described
    as differentially private when only the weights are
    accounted. Exit 3 when any element is OWED, because a
    package that looks complete and is not would have an expert
    signing against evidence that does not exist.
    """
    import json as _json
    import os
    from . import determination as _det

    def _read_path(p):
        if not os.path.exists(p):
            print("note: no such file, treated as OWED: {}"
                  .format(p))
            return None
        try:
            with open(p) as fh:
                return _json.load(fh)
        except (ValueError, OSError) as e:
            print("note: could not read {} ({}), treated as "
                  "OWED".format(p, e))
            return None

    def _read(name):
        return _read_path(os.path.join(args.rundir, name)) \
            if os.path.exists(os.path.join(args.rundir, name)) \
            else None

    if not os.path.isdir(args.rundir):
        print("STOPPED: no such run directory: {}".format(
            args.rundir), file=sys.stderr)
        return 2

    inv = None
    if args.scrubbed:
        try:
            from . import scrub as _scrub
            inv = _scrub.inventory()
        except Exception as e:                  # noqa: BLE001
            print("note: could not read the Safe Harbor "
                  "inventory ({})".format(e))

    scope = None
    if args.scope:
        if not os.path.exists(args.scope):
            print("STOPPED: no such scope file: {}".format(
                args.scope), file=sys.stderr)
            return 2
        with open(args.scope) as fh:
            scope = fh.read()

    rep = _det.assess(
        fidelity=_read("fidelity.json"),
        blueprint=_read("blueprint.json"),
        provenance=_read("provenance.json"),
        # The batteries write their own files with their own
        # shapes; the operator points at what they ran rather
        # than the tool guessing a filename.
        membership=(_read_path(args.membership)
                    if args.membership else None),
        attribute=(_read_path(args.attribute)
                   if args.attribute else None),
        linkage=(_read_path(args.linkage)
                 if args.linkage else None),
        scrub_inventory=inv,
        scope=scope,
        # The run's own source, so evidence measured on a
        # different dataset reads FIXTURE rather than present.
        source_name=((_read("provenance.json") or {})
                     .get("source", {}) or {}).get("name"),
        # ...and the run's own CAPS, so evidence measured under
        # different settings reads CONFIG rather than present.
        run_config=((_read("provenance.json") or {})
                    .get("contract") or {}))

    for line in _det.plain(rep):
        print(line)

    dest = args.json_out or os.path.join(
        args.rundir, "determination.json")
    with open(dest, "w") as fh:
        _json.dump(rep, fh, indent=2, default=str)
    print("")
    print("wrote {}".format(dest))
    # AND THE PAGE, because the person deciding whether to seek
    # a determination will not run a CLI - this project's own
    # lesson about the reconstruction gap, which lived in a
    # terminal until somebody asked why nothing SHOWED it.
    if args.html:
        html = _det.render_html(rep)
        with open(args.html, "w") as fh:
            fh.write(html)
        print("wrote {} ({:,} bytes)".format(args.html,
                                             len(html)))
    # A NON-ZERO EXIT ON AN INCOMPLETE PACKAGE. The same
    # reasoning as `scrub --apply` refusing when a fresh detect
    # still finds something: a document that says it is ready
    # while half the work is outstanding is the worst outcome
    # available.
    return 0 if rep["ready"] else 3


def cmd_relate(args) -> int:
    """Fit and generate a LINKED set of tables.

    THE CAPABILITY EXISTED AND NOTHING CALLED IT. `detect`,
    `cross_couplings` and `assign_keys` were reachable only
    from Python and invoked by nothing but their own suite - so
    the cross-table fidelity work had no operator path at all,
    and `synthkit schema` reported a schema it could not act
    on. That is this project's most-recorded fault, which is
    why this command exists.

    Each table is still fitted on its OWN, which is the honest
    limit: what crosses a link is the KEY ASSIGNMENT, chosen so
    the couplings measured in the source come back. A child's
    VALUES are not re-drawn from its parent.
    """
    import pandas as pd
    from . import blueprint as _bp
    from .discover import discover
    from .generate import generate
    from .intake import IntakeError, read_table
    from .multitable import (assign_keys, check_integrity,
                             cross_couplings, detect, integrity)
    frames = {}
    for f in args.files:
        path = Path(f)
        try:
            fr, how = read_table(path)
        except IntakeError as e:
            print("STOPPED: {}".format(e), file=sys.stderr)
            return 2
        name = path.stem
        if name in frames:
            print("STOPPED: two files are both called {!r} - "
                  "tables are named after their files, so "
                  "rename one".format(name), file=sys.stderr)
            return 2
        frames[name] = fr
        print("  read {}".format(how))
    if len(frames) < 2:
        print("STOPPED: relating needs at least two tables - "
              "with one there is nothing to link. Use "
              "`synthkit fit` for a single table.",
              file=sys.stderr)
        return 2

    overrides = {}
    for spec in (args.group_by or []):
        if "=" not in spec:
            print("STOPPED: --group-by takes TABLE=COLUMN, got "
                  "{!r}".format(spec), file=sys.stderr)
            return 2
        t, _, c = spec.partition("=")
        if t not in frames:
            print("STOPPED: no table called {!r} - the tables "
                  "here are {}".format(
                      t, ", ".join(sorted(frames))),
                  file=sys.stderr)
            return 2
        if c not in frames[t].columns:
            print("STOPPED: {!r} has no column {!r}".format(t, c),
                  file=sys.stderr)
            return 2
        overrides[t] = c

    found = detect(frames)
    src_integ = integrity(frames, found["links"])
    cps = [] if args.no_couple else cross_couplings(
        frames, found, k=args.k)
    print("")
    print("{} table(s), {} link(s), {} cross-table "
          "coupling(s)".format(len(frames), len(found["links"]),
                               len(cps)))

    # WHICH COLUMN GROUPS A TABLE IS A DECISION, SO IT IS
    # PRINTED AND OVERRIDABLE. A silently chosen grouping
    # column is the read-but-not-used fault waiting to happen:
    # every temporal and per-entity statistic is measured
    # against it, so guessing wrong changes every number
    # without changing any message.
    fk_of = {}
    for lk in found["links"]:
        fk_of.setdefault(lk.child, lk.column)
    gen, blueprints, grouped_by = {}, {}, {}
    for name, fr in frames.items():
        if name in overrides:
            gb, why = overrides[name], "you asked for it"
        elif name in fk_of:
            gb = fk_of[name]
            why = ("its foreign key - a child's entity is its "
                   "parent")
        else:
            pk = found["tables"][name].primary_key
            gb = pk
            why = ("its primary key - no parent, so one row "
                   "per entity")
        if gb is None:
            print("STOPPED: {} has no foreign key and no "
                  "primary key was found, so there is nothing "
                  "to group it by. Pass --group-by {}=COLUMN."
                  .format(name, name), file=sys.stderr)
            return 2
        print("  {}: grouped by `{}` ({})".format(name, gb, why))
        bp = _bp.build(fr, discover(fr, group_by=gb,
                                    seed=args.seed),
                       group_by=gb, k=args.k)
        gen[name] = generate(bp, n_patients=fr[gb].nunique(),
                             seed=args.seed)
        # KEPT, so the table can be GRADED afterwards. The
        # blueprint and the grouping column were loop-locals
        # and thrown away, which is why this command could
        # generate a set and never say whether any table in it
        # was faithful.
        blueprints[name] = bp
        grouped_by[name] = gb

    rep = []
    out = assign_keys(gen, found, seed=args.seed,
                      couplings=cps or None, report_into=rep)

    print("")
    for r in rep:
        if r.get("joint") is False:
            # A REFUSAL IS A VERDICT AND MUST BE READ AS ONE.
            print("  {} -> {}: NOT JOINTLY REACHABLE "
                  "(rank-one share {:.2f})".format(
                      r.get("child", "?"), r.get("parent", "?"),
                      r["carried"]))
            print("    enforced {}".format(r.get("enforced")))
            for miss in r.get("not_reachable", []):
                print("    cannot come with it: {}".format(miss))
            print("    why: {}".format(r.get("why", "")))
        else:
            print("  {} -> {}: joint, rank-one share "
                  "{:.2f}".format(r.get("child", "?"),
                                  r.get("parent", "?"),
                                  r["carried"]))
            for pr in r.get("pairs", []):
                print("    {} ~ {}  asked {:+.3f}  landed "
                      "{:+.3f}".format(
                          pr["child_column"],
                          pr["parent_column"],
                          pr["target"], pr["achieved"]))

    gen_integ = check_integrity(out, found)
    src_by = dict((d["child"] + "." + d["column"], d)
                  for d in src_integ)
    print("")
    print("referential integrity, source against generated:")
    for d in gen_integ:
        key = d["child"] + "." + d["column"]
        was = src_by.get(key, {}).get("share")
        print("  {} -> {}: {} source, {:.1%} generated "
              "({} orphan(s))"
              .format(key, d["parent"],
                      "{:.1%}".format(was) if was is not None
                      else "not measured",
                      d["share"], d["orphans"]))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, fr in out.items():
        dest = outdir / (name + ".csv")
        fr.to_csv(dest, index=False)
        print("  wrote {} ({} rows)".format(dest, len(fr)))
    # ---- AND NOW GRADE EACH TABLE, because generating
    # without a verdict is the capability-wired-to-nothing
    # fault at the verdict level.
    #
    # `relate` reported referential integrity and the
    # cross-table couplings and called NO gate at all - so an
    # operator got linked tables with no answer to "is each
    # table faithful?", while `gate.assess` sat there able to
    # answer it from a fidelity record this command never
    # wrote. Every single-table path produces one; the
    # multi-table path did not.
    from . import gate as _gate
    from .pipeline import compare as _compare
    per_table, verdicts, ungraded = {}, {}, {}
    print("")
    print("per-table fidelity, source against generated:")
    for name, fr in out.items():
        gb = grouped_by.get(name)
        src_fr = frames.get(name)
        if src_fr is None or gb is None or gb not in fr.columns:
            print("  {}: NOT GRADED - no grouping column in "
                  "the output".format(name))
            ungraded[name] = "no grouping column"
            continue
        try:
            fid = _compare(src_fr, fr, blueprints[name], gb,
                           None)
        except Exception as e:                  # noqa: BLE001
            # A TABLE THAT CANNOT BE GRADED SAYS SO BY NAME. A
            # silent absence would read as a table with nothing
            # wrong, which is the worst kind of clean.
            print("  {}: NOT GRADED - {}: {}".format(
                name, type(e).__name__, str(e)[:70]))
            ungraded[name] = "{}: {}".format(
                type(e).__name__, str(e)[:70])
            continue
        per_table[name] = fid
        # THE GATE REFUSES A RUN THAT RELATED NO PAIRS, and it
        # is right to: "that is a finding in itself, not a
        # pass". A narrow table - two columns with nothing
        # between them - reaches that, so the refusal is
        # REPORTED rather than allowed to crash the command,
        # and the table counts as ungraded rather than passing.
        try:
            res = _gate.assess(fid)
        except ValueError as e:
            print("  {}: NOT GRADED - {}".format(
                name, str(e)[:110]))
            ungraded[name] = str(e)
            continue
        verdicts[name] = res
        bad = [c["name"] for c in res["criteria"]
               if not c.get("ok")]
        print("  {}: {}".format(
            name,
            "MET on all {} criteria".format(len(res["criteria"]))
            if not bad else "NOT MET on {} of {}: {}".format(
                len(bad), len(res["criteria"]),
                ", ".join(bad))))
        with open(str(outdir / (name + "_fidelity.json")),
                  "w") as fh:
            __import__("json").dump(fid, fh, indent=1,
                                    default=str)
    if verdicts or ungraded:
        from .multitable import set_verdict as _set_verdict
        _all_met = _set_verdict(verdicts, ungraded)
        print("")
        print("THE SET'S VERDICT IS THE WORST OF ITS TABLES: "
              "{}".format("MET" if _all_met else "NOT MET"))
        print("A linked release is read as one thing, so a set "
              "whose weakest table fails is not a set that "
              "passed -")
        if ungraded:
            print("and an UNGRADED table is not a passing one "
                  "either: {} was not graded, so the set "
                  "cannot read MET. Excluding it would be the "
                  "worst kind of clean."
                  .format(", ".join(sorted(ungraded))))

    import json as _json
    with open(str(outdir / "relate.json"), "w") as fh:
        _json.dump({
            "tables": dict((n, {"rows": len(f),
                                "columns": list(f.columns)})
                           for n, f in out.items()),
            # THE VERDICT TRAVELS WITH THE SET, per table and
            # as the worst of them, so a reader does not have
            # to open four files and take the minimum by hand.
            "per_table_verdict": dict(
                (n, {"met": all(c.get("ok")
                                for c in v["criteria"]),
                     "criteria": v["criteria"]})
                for n, v in verdicts.items()),
            # AN UNGRADED TABLE CANNOT LET THE SET READ MET.
            # Taking the minimum over only the GRADED ones
            # would let a table nobody measured pass by being
            # absent from the list, which is the worst kind of
            # clean.
            # ONE rule, one place. Computing it twice is how
            # the printed verdict and the written one come to
            # disagree, which this project records about a
            # privacy threshold applied with one tail bound and
            # reported with another.
            "set_verdict_met": _set_verdict(verdicts, ungraded),
            "tables_not_graded": ungraded,
            "links": [{"child": lk.child, "column": lk.column,
                       "parent": lk.parent,
                       "parent_key": lk.parent_key}
                      for lk in found["links"]],
            "couplings": cps,
            "coupling_report": rep,
            "integrity_source": src_integ,
            "integrity_generated": gen_integ,
            "k": args.k, "seed": args.seed,
        }, fh, indent=2, default=str)
    print("  wrote {}".format(outdir / "relate.json"))
    print("")
    print("WHAT CROSSED AND WHAT DID NOT: each table is fitted "
          "on its own, and what carries a cross-table "
          "relationship is the KEY ASSIGNMENT - a child is "
          "attached to a parent whose value fits. A child's "
          "VALUES are never re-drawn from its parent, and the "
          "correlations one link can carry have rank one, "
          "because one foreign key is one choice per row.")
    return 0


def cmd_schema(args) -> int:
    """Find the keys and links across several tables.

    DETECTION ONLY. It reads, measures and reports - nothing is
    written unless --json-out is given, because a tool that
    reshapes somebody's extract on the strength of a guessed
    key is the silent-wrong-answer fault at its most
    expensive.
    """
    from .intake import IntakeError, read_table
    from .multitable import detect, integrity, render
    frames = {}
    for f in args.files:
        path = Path(f)
        try:
            fr, how = read_table(path)
        except IntakeError as e:
            print("STOPPED: {}".format(e), file=sys.stderr)
            return 2
        name = path.stem
        if name in frames:
            print("STOPPED: two files are both called {!r} - "
                  "tables are named after their files, so "
                  "rename one".format(name), file=sys.stderr)
            return 2
        frames[name] = fr
        print("  read {}".format(how))
    if len(frames) < 2:
        print("STOPPED: schema detection needs at least two "
              "tables - with one there is nothing to link. Use "
              "`synthkit types` for a single table.",
              file=sys.stderr)
        return 2
    found = detect(frames)
    integ = integrity(frames, found["links"])
    print("")
    print(render(found, integ))
    if args.json_out:
        import json as _json
        payload = {
            "tables": {n: {"rows": len(t.frame),
                           "columns": list(t.frame.columns),
                           "primary_key": t.primary_key,
                           "candidates": t.pk_candidates,
                           "notes": t.notes}
                       for n, t in found["tables"].items()},
            "links": [{"child": lk.child, "column": lk.column,
                       "parent": lk.parent,
                       "parent_key": lk.parent_key,
                       "match": round(lk.match, 4),
                       "orphans": lk.orphans}
                      for lk in found["links"]],
            "warnings": found["warnings"],
            "integrity": integ,
        }
        Path(args.json_out).write_text(
            _json.dumps(payload, indent=1), encoding="utf-8")
        print("")
        print("wrote {}".format(args.json_out))
    # A SCHEMA WITH NO LINKS IS A FINDING, NOT A SUCCESS.
    return 0 if found["links"] else 1


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
    # THE COMPETITOR ARM, OFF BY DEFAULT AND SAID WHEN ON.
    # `autosolver` is stdlib logistic regression and honest
    # about it, so a vendor clearing it has shown nothing;
    # --competitor adds what we could actually do, which is
    # the bar a claim has to clear. Default off because it
    # trains three model families per tier and an operator
    # should choose to pay for that.
    if getattr(args, "competitor", None):
        from .gui import _solver as _gui_solver
        try:
            kwargs["competitor"] = _gui_solver(args.competitor)
        except ValueError as e:
            print("STOPPED: {}".format(e), file=sys.stderr)
            return 2
        kwargs["competitor_name"] = args.competitor
    result = run_showdown(camp, fn,
                          vendor_name=args.name or args.solver,
                          **kwargs)
    print(result.format_text())
    if args.json_out:
        import json as _json
        payload = _json.dumps({
            "vendor": result.vendor_name,
            "tiers": [{"tier": t.tier, "ceiling": t.ceiling,
                       "baseline": t.baseline_auroc,
                       "competitor": t.competitor_auroc,
                       "competitor_name": t.competitor_name,
                       "verdict": t.verdict,
                       "vendor": t.vendor_auroc,
                       "passed": t.vendor_passed}
                      for t in result.tiers],
        }, indent=2)
        try:
            Path(args.json_out).write_text(payload,
                                           encoding="utf-8")
        except OSError as e:
            # The operator hit PermissionError here AFTER a clean
            # showdown - a locked file, or a folder wearing the
            # filename - and got a traceback that read as the run
            # failing. The run is printed above and did not fail.
            print("STOPPED writing {}: {}. The showdown above "
                  "still ran - the numbers printed are real. "
                  "Close whatever holds that file (or remove a "
                  "folder by that name) and re-run with the same "
                  "or a fresh --json-out to capture the "
                  "JSON.".format(args.json_out, e),
                  file=sys.stderr)
            return 2
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


def cmd_notes(args) -> int:
    """Render clinical notes over a generated cohort, and write
    the answer key separately.

    WHY THE KEY IS A SEPARATE FILE. The ledger says what every
    note actually asserts - which corruptions were applied, where
    the prose and the structured record disagree, and which facts
    live only in the text. That is the answer key to the exam, so
    `notes.csv` is the thing that travels to a recipient and
    `ledger.json` is the thing that stays, exactly as the release
    dials keep a key beside what they publish. Shipping both would
    be marking your own homework for somebody else.

    NO PHRASING COMES FROM A REAL NOTE. Facts are declared and
    rendered, so no sentence can be traced to a patient - which is
    what makes a note shareable at all, and is why this is
    generated from the SYNTHETIC cohort rather than learned from
    real text.
    """
    import json
    import pandas as pd
    from synthkit import notefacts as NF
    from synthkit.transcribe import CORRUPTIONS, TranscribeSpec, \
        transcribe

    src = args.csv
    if args.run:
        cand = Path(args.run) / "generated.csv"
        if not cand.exists():
            print("no generated.csv in {} - point --run at a run "
                  "directory or pass --csv".format(args.run),
                  file=sys.stderr)
            return 2
        src = str(cand)
    if not src:
        print("give --run RUNDIR or --csv FILE", file=sys.stderr)
        return 2
    try:
        df = pd.read_csv(src, dtype=str, keep_default_na=False)
    except OSError as e:
        print("could not read {}: {}".format(src, e),
              file=sys.stderr)
        return 2
    print("notes: reading {} ({} rows, {} columns)".format(
        src, len(df), len(df.columns)))

    if args.visit_type:
        if args.visit_column not in df.columns:
            print("no column named {!r} to filter on - the "
                  "columns are: {}".format(
                      args.visit_column,
                      ", ".join(map(str, df.columns[:12]))),
                  file=sys.stderr)
            return 2
        wanted = [v.strip().upper()
                  for v in args.visit_type.split(",")]
        before = len(df)
        df = df[df[args.visit_column].astype(str).str.upper()
                .isin(wanted)].reset_index(drop=True)
        print("scope: {} -> {} rows where {} is one of {}".format(
            before, len(df), args.visit_column,
            ", ".join(wanted)))
        if df.empty:
            print("nothing matched that scope - no notes written",
                  file=sys.stderr)
            return 2

    desc = None
    if args.descriptions:
        try:
            desc = json.load(open(args.descriptions,
                                  encoding="utf-8"))
        except (OSError, ValueError) as e:
            print("could not read {}: {}".format(
                args.descriptions, e), file=sys.stderr)
            return 2
        print("descriptions: {} code(s) supplied".format(len(desc)))
    else:
        print("descriptions: none supplied, so a diagnosis reads "
              "as its bare code - pass --descriptions for prose")

    gb = args.group_by if args.group_by in df.columns else None
    frame, facts, plan = NF.build_facts(
        df, group_by=gb, k=args.k,
        tokens_per_set=args.tokens_per_set,
        descriptions=desc, measures=args.measures)

    if not facts:
        print("no facts could be derived from this frame, so "
              "there is nothing for a note to say", file=sys.stderr)
        for col, why in plan["skipped_sets"].items():
            print("  {}: {}".format(col, why), file=sys.stderr)
        return 2

    kinds = {}
    for f in facts:
        kinds[f.kind] = kinds.get(f.kind, 0) + 1
    print("facts: {} total - {}".format(
        len(facts),
        ", ".join("{} {}".format(v, k)
                  for k, v in sorted(kinds.items()))))
    print("tokens mentioned are held by at least {} patients, "
          "counted in patients and never rows - a note is a "
          "release".format(args.k))
    for col, why in plan["skipped_sets"].items():
        print("  set skipped: {} - {}".format(col, why))
    if plan["not_charted"]:
        print("  not a charted measurement, so absent from the "
              "vitals line: {}".format(
                  ", ".join(plan["not_charted"][:8])))
    if plan["unlabelled"]:
        print("  no declared clinical phrasing, so left out "
              "rather than given prose invented from a code: "
              "{}".format(", ".join(plan["unlabelled"][:8])))

    spec = TranscribeSpec(facts=facts,
                          rates=NF.default_rates(args.rate),
                          seed=args.seed)
    noted, ledgers = transcribe(frame.to_dict("records"), spec,
                                note_column=args.note_column)
    final = NF.reconcile(pd.DataFrame(noted), plan)

    applied = {}
    for L in ledgers:
        for e in L:
            for c in e.get("corruptions", []):
                applied[c] = applied.get(c, 0) + 1
    missing = [c for c in CORRUPTIONS if c not in applied]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final.to_csv(out / "notes.csv", index=False)
    json.dump({"facts": [f.__dict__ for f in facts],
               "rates": spec.rates, "seed": spec.seed,
               "k": args.k, "source": src,
               "corruptions_applied": applied,
               "corruptions_never_applied": missing,
               "plan": {k: v for k, v in plan.items()
                        if k != "indicators"},
               "ledger": ledgers},
              open(out / "ledger.json", "w", encoding="utf-8"),
              indent=1, default=str)

    print()
    print("wrote {} - {} rows, {} columns, the note in "
          "{!r}".format(out / "notes.csv", len(final),
                        len(final.columns), args.note_column))
    print("wrote {} - the ANSWER KEY. This one does NOT travel "
          "with the notes.".format(out / "ledger.json"))
    print()
    print("documentation problems planted, by kind:")
    for c in CORRUPTIONS:
        print("  {:22s} {}".format(c, applied.get(c, 0)))
    if missing:
        print("  NEVER APPLIED: {} - a grader cannot report on a "
              "failure type the text does not contain".format(
                  ", ".join(missing)))
    return 0


def cmd_grounded(args) -> int:
    """Census a tool's claims against the record it was given.

    The question a vendor's safety claim makes - every statement
    traces back to the chart - is answerable exactly here, because
    we wrote the chart. Recall and the per-corruption breakdown
    come from `transcribe.score_extraction`, which this calls
    rather than reimplements; what it adds is the claim that
    matches no fact at all, which that loop cannot see.
    """
    import json
    import pandas as pd
    from synthkit import grounding as GR
    from synthkit.transcribe import FactSpec, score_extraction

    try:
        key = json.load(open(args.ledger, encoding="utf-8"))
    except (OSError, ValueError) as e:
        print("could not read {}: {}".format(args.ledger, e),
              file=sys.stderr)
        return 2
    if "ledger" not in key:
        print("{} is not a ledger from `synthkit notes` - it has "
              "no `ledger` list".format(args.ledger),
              file=sys.stderr)
        return 2
    ledgers = key["ledger"]
    facts = [FactSpec(**f) for f in key.get("facts", [])]
    plan = key.get("plan", {})

    try:
        claims_raw = json.load(open(args.claims, encoding="utf-8"))
    except (OSError, ValueError) as e:
        print("could not read {}: {}".format(args.claims, e),
              file=sys.stderr)
        return 2
    claims = _claims_per_row(claims_raw, len(ledgers))
    if claims is None:
        print("claims must be a list of {} per-row lists, or a "
              "flat list of objects each carrying `row` - got {} "
              "entries".format(len(ledgers), len(claims_raw)),
              file=sys.stderr)
        return 2

    frame = None
    if args.notes:
        try:
            frame = pd.read_csv(args.notes, dtype=str,
                                keep_default_na=False)
        except OSError as e:
            print("could not read {}: {}".format(args.notes, e),
                  file=sys.stderr)
            return 2
        if len(frame) != len(ledgers):
            print("{} has {} rows and the ledger has {} - these "
                  "are not the same run".format(
                      args.notes, len(frame), len(ledgers)),
                  file=sys.stderr)
            return 2

    desc = None
    if args.descriptions:
        desc = json.load(open(args.descriptions, encoding="utf-8"))

    rep = GR.audit(ledgers, claims, frame=frame, facts=facts,
                   plan=plan, descriptions=desc)
    print("grounding: {} positive claim(s) over {} row(s)".format(
        rep["claims_censused"], len(ledgers)))
    if frame is None:
        print("  NOTE: no --notes given, so the STRUCTURED record "
              "could not be consulted and a claim read straight "
              "off a code set will be reported as ungrounded")
    for cat in GR.CATEGORIES:
        print("  {:14s} {}".format(cat, rep["counts"][cat]))
    print("  grounded share {}".format(rep["grounded_share"]))
    print()
    print(rep["verdict"])
    for ex in rep["ungrounded_examples"][:10]:
        print("  ungrounded  row {:>6}  {}".format(
            ex["row"], ex["label"]))
    for ex in rep["contradicted_examples"][:10]:
        print("  DENIED by the prose  row {:>6}  {}".format(
            ex["row"], ex["label"]))

    sf = GR.source_followed(ledgers, claims)
    print()
    if sf["disagreements"]:
        print("where the prose and the record DISAGREE by "
              "construction ({} mentions), the tool followed:"
              .format(sf["disagreements"]))
        for k, v in sf["followed"].items():
            print("  {:12s} {}".format(k, v))
    else:
        print("no planted disagreement in this run, so which "
              "source the tool trusts is not measured here")

    scored = score_extraction(
        ledgers, [[{"fact": g.get("fact"),
                    "present": g.get("present", True),
                    "value": g.get("value"),
                    "current": g.get("current", True),
                    "certain": g.get("certain", True)}
                   for g in (row or [])] for row in claims])
    print()
    print("recall by documentation problem (from "
          "transcribe.score_extraction):")
    for tag in sorted(scored):
        r = scored[tag]
        print("  {:22s} mentions {:>6}  recall {}  false "
              "positives {}".format(
                  tag, r["mentions"], r["recall"],
                  r["false_positive_rate"]))

    if args.out:
        json.dump({"grounding": rep, "source_followed": sf,
                   "by_corruption": scored},
                  open(args.out, "w", encoding="utf-8"),
                  indent=1, default=str)
        print()
        print("wrote {}".format(args.out))
    return 0


def _claims_per_row(raw, n):
    """Accept per-row lists or a flat list carrying `row`.

    A vendor returns whichever shape their tool produces, and
    refusing one of them would make the audit unusable for a
    reason that has nothing to do with grounding.
    """
    if isinstance(raw, dict):
        raw = raw.get("claims", raw.get("rows"))
    if not isinstance(raw, list):
        return None
    if raw and isinstance(raw[0], dict) and "row" in raw[0]:
        out = [[] for _ in range(n)]
        for g in raw:
            try:
                i = int(g["row"])
            except (KeyError, TypeError, ValueError):
                return None
            if 0 <= i < n:
                out[i].append(g)
        return out
    if len(raw) != n:
        return None
    return [r or [] for r in raw]


def cmd_codemap(args) -> int:
    """Build a code-to-phrasing map, screened by what was already
    released.

    The screen is the point. A map built from the raw source names
    every code the source contains - including the ones the k rule
    refused to publish - and a list of rare codes says a patient
    in this cohort has one. Restricted to codes the released
    cohort already carries, it discloses nothing new.
    """
    import json
    import pandas as pd
    from synthkit import codemap as CM

    paths = list(args.table or [])
    if args.omop:
        for name in ("condition_occurrence", "procedure_occurrence",
                     "drug_exposure", "measurement",
                     "observation", "device_exposure"):
            cand = Path(args.omop) / (name + ".csv")
            if cand.exists():
                paths.append(str(cand))
    if not paths:
        print("give --omop DIR or --table FILE (repeatable)",
              file=sys.stderr)
        return 2

    tables = []
    for pth in paths:
        try:
            frame = pd.read_csv(pth, dtype=str,
                                keep_default_na=False,
                                low_memory=False)
        except OSError as e:
            print("could not read {}: {}".format(pth, e),
                  file=sys.stderr)
            return 2
        tables.append((Path(pth).name, frame))
        print("read {} ({} rows)".format(Path(pth).name,
                                         len(frame)))

    released = None
    if args.cohort:
        try:
            coh = pd.read_csv(args.cohort, dtype=str,
                              keep_default_na=False)
        except OSError as e:
            print("could not read {}: {}".format(args.cohort, e),
                  file=sys.stderr)
            return 2
        cols = args.set_column or None
        released = CM.cohort_codes(coh, cols or coh.columns,
                                   sep=args.sep)
        print("cohort {} carries {} distinct token(s)".format(
            args.cohort, len(released)))
    else:
        print("NO --cohort GIVEN, so the map is NOT screened. It "
              "will name codes the k rule refused to publish and "
              "must not be handed to a recipient.")

    rep = CM.build(tables, released)
    for why in rep["tables_skipped"]:
        print("  skipped {}".format(why))
    if rep["screened"]:
        print("codes in the source {} -> kept {}, withheld "
              "{}".format(rep["codes_in_source"],
                          rep["codes_kept"],
                          rep["codes_withheld"]))
    print(rep["note"])
    if not rep["descriptions"]:
        print("no code carried a description, so there is nothing "
              "to write", file=sys.stderr)
        return 2
    json.dump(rep["descriptions"],
              open(args.out, "w", encoding="utf-8"),
              indent=1, sort_keys=True)
    print("wrote {} - {} entries".format(
        args.out, len(rep["descriptions"])))
    return 0


def cmd_sample(args) -> int:
    """A PATIENT-level subsample, for live demonstrations and
    quick reads - all rows of N randomly chosen patients, chosen
    by seed. Sampling ROWS instead would break visit counts,
    persistence and every within-patient statistic the fit
    measures; a patient either travels whole or stays behind.
    The values pass through byte-for-byte - read the way the
    pipeline reads (a blank stays a blank), written with no
    reformatting - because a sampler that touches values is a
    second parser waiting to disagree with the first."""
    import numpy as np
    import pandas as pd
    try:
        df = pd.read_csv(args.csv, dtype=str,
                         keep_default_na=False)
    except OSError as e:
        print("could not read {}: {}".format(args.csv, e),
              file=sys.stderr)
        return 2
    if args.group_by not in df.columns:
        print("no column named {!r} in {} - the columns are: {}"
              .format(args.group_by, args.csv,
                      ", ".join(map(str, df.columns[:12]))),
              file=sys.stderr)
        return 2
    ids = pd.unique(df[args.group_by])
    print("sample: --patients {} --seed {} from {} ({} rows, "
          "{} patients)".format(args.patients, args.seed,
                                args.csv, len(df), len(ids)))
    if args.patients >= len(ids):
        print("only {} patient(s) exist, so all of them travel - "
              "this is a copy, not a sample".format(len(ids)))
        keep = set(ids)
    else:
        rng = np.random.RandomState(args.seed)
        keep = set(rng.choice(ids, size=args.patients,
                              replace=False))
    out = df[df[args.group_by].isin(keep)]
    out.to_csv(args.out, index=False)
    print("wrote {} - {} rows for {} patient(s), whole patients "
          "only".format(args.out, len(out),
                        out[args.group_by].nunique()))
    return 0


def cmd_gate(args) -> int:
    """The M0 gate verdict on a finished run - the same lines
    scripts/m0_gate.py prints, from the same synthkit.gate
    renderer, reachable wherever the package is installed. The
    test kit is run from its own folder, where `scripts/` does
    not exist; a verb the install carries is the `synthkit fit`
    lesson applied to the gate."""
    import json as _json
    from . import gate as _gate
    run = Path(args.rundir)
    fp = run / "fidelity.json"
    if not fp.exists():
        print("no fidelity.json in {} - run `synthkit fit` with "
              "--generate first".format(run), file=sys.stderr)
        return 2
    try:
        verdict = _gate.assess(
            _json.loads(fp.read_text(encoding="utf-8")))
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    lines, code = _gate.report(verdict, run)
    print("\n".join(lines))
    return code


def cmd_signoff(args) -> int:
    """The sign-off page, as a verb. The page builder lives in
    scripts/signoff.py beside the repository; this resolves it the
    way `exam` resolves the report card, and says so in a sentence
    when it is not there rather than tracebacking."""
    import subprocess as _sp
    script = (Path(__file__).resolve().parent.parent
              / "scripts" / "signoff.py")
    if not script.exists():
        print("scripts/signoff.py is not beside this install - "
              "run from the repository checkout, where the page "
              "builder lives", file=sys.stderr)
        return 2
    q = _sp.run([sys.executable, str(script), args.rundir,
                 "-o", args.out], capture_output=True, text=True)
    msg = (q.stdout or q.stderr).strip()
    if msg:
        print(msg.splitlines()[-1])
    return q.returncode


def cmd_scrub(args) -> int:
    """Structured-field PHI detection - report, and on request
    drop. Exit 1 when PHI-shaped columns are present and nothing
    was done about them, because a detector used in a pipeline
    must be usable as a gate."""
    import pandas as pd
    from . import scrub
    if args.inventory:
        # THE COVERAGE TABLE IS PRINTABLE WITHOUT A FILE,
        # because the question "what does this reach" is asked
        # before anybody has a table to point it at - and a
        # governance reader should not have to run a scan on
        # real records to find out what the scanner looks for.
        inv = scrub.inventory()
        for c in inv["categories"]:
            print("{:>2}  {:<14}  {}".format(
                c["number"], c["status"], c["what"]))
            print("    {}".format(c["note"]))
        print()
        print("{} categories: {}".format(
            inv["total"], ", ".join(
                "{} {}".format(v, k)
                for k, v in sorted(inv["counts"].items()))))
        print("A category marked `not covered` is a GAP THIS "
              "TOOL NAMES, not one it has ruled out. A scan "
              "that reports nothing has reported nothing about "
              "those.")
        return 0
    # MAKING THE CSV OPTIONAL FOR --inventory REMOVED
    # ARGPARSE'S OWN REFUSAL, so a bare `scrub` fell through to
    # pandas and raised a ValueError about a NoneType buffer -
    # a traceback where there had been a clean "required
    # argument". The convenience ate a guard, which this
    # project has recorded once already for campaign-run.
    if not args.csv:
        print("STOP: scrub needs a CSV. Pass the table to "
              "scan, or --inventory to print the coverage "
              "table with no file.", file=sys.stderr)
        return 2
    df = pd.read_csv(args.csv, dtype=str,
                     keep_default_na=False)
    rep = scrub.detect(df, group_by=args.group_by)
    for f in rep["findings"]:
        print("PHI  {}: {}".format(f["column"], f["why"]))
    for o in rep["out_of_scope"]:
        print("OUT OF SCOPE  {}: {}".format(o["column"],
                                            o["why"]))
    print("clear: {} of {} column(s)".format(
        len(rep["clear"]), rep["columns_seen"]))
    print(rep["note"])
    if args.apply:
        # TWO REMEDIES, BOTH APPLIED. Dropping alone would
        # leave the Safe Harbor generalisations undone, and
        # dropping a service date or an age instead would
        # destroy span, length-of-stay and a covariate every
        # model uses. So what must go is dropped, and what must
        # be coarsened is coarsened - and both are REPORTED,
        # because a file that says it was de-identified while
        # half the work is outstanding is the worst outcome
        # here.
        out_df, dropped = scrub.apply(df, rep)
        out_df, actions = scrub.generalise(out_df, rep)
        out_df.to_csv(args.apply, index=False)
        print("dropped {} column(s) -> {}".format(
            len(dropped), args.apply))
        for act in actions:
            print("  generalised {}: {}".format(
                act["column"], act["did"]))
        left = [f for f in scrub.detect(
            out_df, group_by=args.group_by)["findings"]]
        # A REVIEW ITEM IS NOT A TOOL DEFECT, AND MUST NOT BE
        # REPORTED AS ONE.
        #
        # The refusal below exists because a file that says it
        # was de-identified while half the work is outstanding
        # is the worst outcome available - and it was written
        # when every remedy was drop or generalise, both of
        # which this command performs. A `review` finding is
        # explicitly NOT performed: free-text redaction is a
        # governance decision and this tool does not make it.
        #
        # Folding the two together would have made `--apply`
        # report a defect forever on any table with a clinical
        # note, which is both wrong and the kind of permanent
        # red that teaches people to ignore an exit code. They
        # get separate states: 1 means the tool failed to do
        # what it claimed, 3 means a human decision is still
        # outstanding.
        owed = [f for f in left
                if f.get("remedy") == "review"]
        broke = [f for f in left
                 if f.get("remedy") != "review"]
        if broke:
            print("STILL FLAGGED after applying: {} - this is "
                  "a defect in the tool, not in your data"
                  .format(", ".join(f["column"] for f in broke)),
                  file=sys.stderr)
            return 1
        if owed:
            print("NOT CERTIFIED: {} still carr{} identifier "
                  "shapes inside FREE TEXT, which this tool "
                  "reports and does not redact. That is a "
                  "decision for whoever owns the data, not a "
                  "default - see the finding's counts above. "
                  "Everything with an automatic remedy was "
                  "applied."
                  .format(", ".join(f["column"] for f in owed),
                          "ies" if len(owed) == 1 else "y"),
                  file=sys.stderr)
            return 3
        return 0
    if rep["findings"]:
        print("PHI-shaped columns present and nothing was "
              "dropped - re-run with --apply OUT.csv, or remove "
              "them yourself before fitting", file=sys.stderr)
        return 1
    return 0


def cmd_exam(args) -> int:
    """The whole evaluation loop as ONE command - goal 7's
    repeatability. bridge -> plant -> ladder -> baseline ->
    showdown (-> report card when the repository's scripts are
    present), each stage echoing what it did and refusing in a
    sentence. The first end-to-end run took five commands and one
    path typo cost a day; a loop that exists should be one
    command."""
    import json as _json
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = Path(args.rundir)

    print("[1/6] bridge")
    bp_path = run / "blueprint.json"
    if not bp_path.exists():
        print("STOPPED: no blueprint.json in {} - run `synthkit "
              "fit` first".format(run), file=sys.stderr)
        return 2
    from .bridge import blueprint_to_tablespec
    made = blueprint_to_tablespec(
        _json.loads(bp_path.read_text(encoding="utf-8")),
        title=run.name)
    (out / "tablespec.json").write_text(
        _json.dumps(made, indent=1), encoding="utf-8")
    print("  {} column(s) crossed, {} dropped; did NOT cross: "
          "{}".format(len(made["tablespec"]["columns"]),
                      len(made["carried"]["columns_dropped"]),
                      ", ".join(made["carried"]
                                ["did_not_cross"][:3]) + ", ..."))

    print("[2/6] plant")
    from . import semisynth
    effects = {}
    for e in (args.effect or []):
        name, _, beta = e.rpartition("=")
        try:
            effects[name] = float(beta)
        except ValueError:
            print("--effect needs col=BETA; got {!r}".format(e),
                  file=sys.stderr)
            return 2
    if not effects:
        print("at least one --effect is required - an exam with "
              "no planted signal grades nothing", file=sys.stderr)
        return 2
    try:
        planted = semisynth.plant(
            made, effects, name=args.outcome_name,
            kind=args.kind, prevalence=args.prevalence)
    except ValueError as e:
        print("STOPPED: {}".format(e), file=sys.stderr)
        return 2
    blk = planted.get("planted") or {}
    for nm, beta in (blk.get("effects_in_sds") or {}).items():
        print("  planted {}: {:+.2f} sd".format(nm, float(beta)))
    for r_ in blk.get("refused") or []:
        print("  REFUSED {}: {}".format(r_["effect"], r_["why"]))
    (out / "exam_spec.json").write_text(
        _json.dumps(planted["tablespec"], indent=1),
        encoding="utf-8")
    (out / "exam_spec.planted.json").write_text(
        _json.dumps(planted, indent=1), encoding="utf-8")

    print("[3/6] ladder")
    from .campaign import CampaignError, compile_campaign,         load_campaign, run_campaign, write_campaign
    from .tablespec import TableSpec
    base = TableSpec.from_json(
        (out / "exam_spec.json").read_text(encoding="utf-8"))
    try:
        camp = compile_campaign("predict", base,
                                bars=_parse_bars(args.bars),
                                outcome=args.outcome_name)
    except CampaignError as e:
        print("STOPPED: campaign invalid: {}".format(e),
              file=sys.stderr)
        return 2
    camp_dir = write_campaign(out / "campaign", camp)
    print("  {} tier(s) -> {}".format(len(camp.tiers), camp_dir))

    print("[4/6] baseline: {}".format(args.baseline))
    from .gui import _solver as _gui_solver
    try:
        base_solver = _gui_solver(args.baseline)
        vend_solver = _gui_solver(args.solver)             if ":" not in args.solver else None
    except ValueError as e:
        print("STOPPED: {}".format(e), file=sys.stderr)
        return 2
    camp2 = load_campaign(Path(camp_dir))
    result = run_campaign(camp2, base_solver,
                          solver_name=args.baseline)
    write_campaign(Path(camp_dir), camp2, result,
                   result_name=args.baseline)
    print("  highest tier passed: {}".format(
        result.highest_passed))

    print("[5/6] showdown: {} vs {}".format(args.solver,
                                            args.baseline))
    from .autosolver import run_showdown
    if vend_solver is None:
        _cwd_importable()
        mod_name, fn_name = args.solver.split(":", 1)
        vend_solver = getattr(
            importlib.import_module(mod_name), fn_name)
    sd = run_showdown(camp2, vend_solver,
                      vendor_name=args.solver,
                      baseline=base_solver,
                      baseline_name=args.baseline)
    print(sd.format_text())
    payload = _json.dumps({
        "vendor": sd.vendor_name,
        "tiers": [{"tier": t.tier, "ceiling": t.ceiling,
                   "baseline": t.baseline_auroc,
                   "vendor": t.vendor_auroc,
                   "passed": t.vendor_passed}
                  for t in sd.tiers]}, indent=2)
    (out / "showdown.json").write_text(payload, encoding="utf-8")

    print("[6/6] report card")
    card_script = (Path(__file__).resolve().parent.parent
                   / "scripts" / "report_card.py")
    if card_script.exists():
        import subprocess as _sp
        q = _sp.run([sys.executable, str(card_script),
                     str(camp_dir),
                     "--planted",
                     str(out / "exam_spec.planted.json"),
                     "--showdown", str(out / "showdown.json"),
                     "-o", str(out / "report_card.html")],
                    capture_output=True, text=True)
        print("  " + (q.stdout or q.stderr).strip()
              .splitlines()[-1])
    else:
        print("  scripts/report_card.py is not beside this "
              "install - build the card from the repository "
              "checkout; showdown.json above is its input")
    print()
    print("exam -> {}".format(out))
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
        "forecast",
        help="what this dataset can support before you generate: what "
             "clears the privacy floor, and which output sizes are "
             "honest")
    p.add_argument("src", help="the source CSV")
    p.add_argument("--group-by", default="person_id",
                   help="the patient/entity id column - the floor "
                        "counts PATIENTS, never rows")
    p.add_argument("--k", type=int, default=10,
                   help="privacy floor: distinct patients behind every "
                        "published pattern (default 10)")
    p.add_argument("--multipliers", default=None,
                   help="output sizes to report, e.g. 1,5,10,20")
    p.set_defaults(fn=cmd_forecast)

    p = sub.add_parser(
        "types", parents=[_fit_parser(add_help=False)],
        help="how every column was read, in seconds - run this "
             "first on a dataset nobody has opened")
    # A LOOK SHOULD NOT DEMAND A DESTINATION. `types` reads and
    # prints; requiring --out made the test kit's FIRST command an
    # argparse error. The flag stays honored when given (the report
    # lands there); omitted, a temporary directory is used and
    # discarded.
    for a in p._actions:
        if "--out" in getattr(a, "option_strings", ()):
            a.required = False
            a.default = None
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
    p.add_argument("--competitor", default="",
                   help="also run a COMPETITOR arm, e.g. "
                        "automl. The default baseline is "
                        "stdlib logistic regression and says "
                        "so, which a vendor can clear without "
                        "showing anything; the competitor is "
                        "what we could actually do. Off by "
                        "default because it trains several "
                        "model families per tier")
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

    p = sub.add_parser("sample",
                       help="a patient-level subsample of a CSV "
                            "- whole patients only, seeded, for "
                            "live demos and quick reads")
    p.add_argument("csv")
    p.add_argument("--group-by", required=True)
    p.add_argument("--patients", type=int, required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_sample)

    p = sub.add_parser("gate",
                       help="the M0 gate verdict on a finished "
                            "run, with next steps on a FAIL")
    p.add_argument("rundir")
    p.set_defaults(fn=cmd_gate)

    p = sub.add_parser("signoff",
                       help="the one-page sign-off from a "
                            "finished run")
    p.add_argument("rundir")
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_signoff)

    p = sub.add_parser("batch",
                       help="run a fit end to end in a cloud "
                            "account: source in, named "
                            "artifacts out, nothing else")
    p.add_argument("--src", required=True,
                   help="the source table, a local path or an "
                        "s3:// URI. It comes IN to the "
                        "compute and is never sent anywhere.")
    p.add_argument("--out", required=True,
                   help="a local working directory for the run")
    p.add_argument("--publish", default="",
                   help="where the ARTIFACTS go - a local "
                        "directory or an s3:// prefix. Only "
                        "the allow-listed files leave; the "
                        "source cannot, by any flag.")
    p.add_argument("--group-by", default="person_id")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="say what WOULD leave and stop. A "
                        "transfer nobody previewed is one "
                        "nobody can object to.")
    p.set_defaults(fn=cmd_batch)

    p = sub.add_parser("schema",
                       help="read several tables, find the keys "
                            "from the VALUES, and report the "
                            "links and referential integrity")
    p.add_argument("files", nargs="+",
                   help="the tables, as csv/xlsx/jsonl/parquet. "
                        "Each table is named after its file.")
    p.add_argument("--json-out", default="",
                   help="write the detected schema as JSON")
    p.set_defaults(fn=cmd_schema)

    p = sub.add_parser("relate",
                       help="fit and generate a LINKED set of "
                            "tables, enforcing the cross-table "
                            "couplings measured in the source")
    p.add_argument("files", nargs="+")
    p.add_argument("--out", required=True, metavar="DIR")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--group-by", action="append",
                   metavar="TABLE=COLUMN",
                   help="override the grouping column for one "
                        "table; repeatable")
    p.add_argument("--no-couple", action="store_true",
                   help="assign keys WITHOUT enforcing the "
                        "cross-table couplings - the control "
                        "arm, for measuring what the "
                        "enforcement is worth")
    p.set_defaults(fn=cmd_relate)

    p = sub.add_parser("determination",
                       help="assemble the evidence package for "
                            "a HIPAA Expert Determination from "
                            "a run directory - and say what is "
                            "still OWED")
    p.add_argument("rundir")
    p.add_argument("--membership", default=None, metavar="JSON",
                   help="output of scripts/membership_new_path.py "
                        "-o FILE")
    p.add_argument("--attribute", default=None, metavar="JSON",
                   help="output of "
                        "scripts/attribute_disclosure.py -o FILE")
    p.add_argument("--linkage", default=None, metavar="JSON",
                   help="output of scripts/linkage_risk.py -o "
                        "FILE - the statute's 'in COMBINATION "
                        "with other reasonably available "
                        "information' half, which the two "
                        "attack adversaries do not cover")
    p.add_argument("--scope", default=None, metavar="FILE",
                   help="a WRITTEN scope and residual-risk "
                        "statement; residual risk is the one "
                        "element a run cannot supply")
    p.add_argument("--html", default=None, metavar="FILE",
                   help="also write the package as a "
                        "self-contained page, for the person "
                        "deciding whether to seek a "
                        "determination")
    p.add_argument("--scrubbed", action="store_true",
                   help="the source went through `synthkit "
                        "scrub`, so the Safe Harbor inventory "
                        "is part of the record")
    p.add_argument("--json-out", default=None, metavar="FILE",
                   help="where to write determination.json; "
                        "defaults to the run directory")
    p.set_defaults(fn=cmd_determination)

    p = sub.add_parser("receipts",
                       help="one receipt per published claim "
                            "from a run directory: evidence, "
                            "support, survival, and what each "
                            "claim does NOT establish")
    p.add_argument("rundir")
    p.add_argument("--json-out", default=None, metavar="FILE",
                   help="where to write receipts.json; defaults "
                        "to the run directory")
    p.set_defaults(fn=cmd_receipts)

    p = sub.add_parser("scrub",
                       help="detect (and with --apply, drop) "
                            "PHI-shaped columns in a structured "
                            "table before fitting")
    p.add_argument("csv", nargs="?")
    p.add_argument("--group-by", default=None)
    p.add_argument("--apply", default=None, metavar="OUT_CSV")
    p.add_argument("--inventory", action="store_true",
                   help="print the 18 Safe Harbor categories "
                        "with this tool's coverage status and "
                        "exit; needs no CSV")
    p.set_defaults(fn=cmd_scrub)

    p = sub.add_parser("exam",
                       help="the whole evaluation loop, one "
                            "command: bridge, plant, ladder, "
                            "baseline, showdown, report card")
    p.add_argument("rundir")
    p.add_argument("--effect", action="append", default=[],
                   help="col=BETA in sd, repeatable")
    p.add_argument("--outcome-name", default="outcome")
    p.add_argument("--kind", default="logistic",
                   choices=["logistic", "linear"])
    p.add_argument("--prevalence", type=float, default=0.25)
    p.add_argument("--bars", default="auroc=0.65",
                   help="k=v pairs; the default sits under the "
                        "usual ceilings on purpose")
    p.add_argument("--solver", default="autosolver_hybrid")
    p.add_argument("--baseline", default="autosolver")
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_exam)

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

    p = sub.add_parser("notes",
                       help="render clinical notes over a "
                            "generated cohort, with the answer "
                            "key written separately")
    p.add_argument("--run", help="a run directory; its "
                                 "generated.csv is used")
    p.add_argument("--csv", help="a generated cohort csv instead")
    p.add_argument("-o", "--out", required=True,
                   help="directory for notes.csv and ledger.json")
    p.add_argument("--group-by", default="person_id",
                   help="the patient column; k counts patients")
    p.add_argument("--k", type=int, default=10,
                   help="a note may mention only a token held by "
                        "at least this many PATIENTS")
    p.add_argument("--tokens-per-set", type=int, default=8)
    p.add_argument("--measures", type=int, default=6)
    p.add_argument("--rate", type=float, default=0.25,
                   help="per-corruption rate; 0 renders clean "
                        "prose with no documentation problems")
    p.add_argument("--descriptions",
                   help="json mapping a code to its clinical "
                        "phrasing; without it a diagnosis reads "
                        "as its bare code")
    p.add_argument("--visit-type",
                   help="restrict to these visit types, comma "
                        "separated, e.g. INPATIENT")
    p.add_argument("--visit-column", default="visit_type")
    p.add_argument("--note-column", default="clinical_note")
    p.add_argument("--seed", type=int, default=20260731)
    p.set_defaults(fn=cmd_notes)

    p = sub.add_parser("grounded",
                       help="census a tool's claims against the "
                            "record it was given - is every "
                            "statement supported?")
    p.add_argument("--ledger", required=True,
                   help="ledger.json from `synthkit notes` - the "
                        "answer key")
    p.add_argument("--claims", required=True,
                   help="json: one list per row of "
                        "{label, present}, or a flat list whose "
                        "objects carry `row`")
    p.add_argument("--notes",
                   help="notes.csv, so the STRUCTURED record can "
                        "be consulted too; without it a claim "
                        "read off a code set reads as ungrounded")
    p.add_argument("--descriptions",
                   help="the same code-to-phrasing json the notes "
                        "were rendered with")
    p.add_argument("-o", "--out", help="write the report as json")
    p.set_defaults(fn=cmd_grounded)

    p = sub.add_parser("codemap",
                       help="build a code-to-phrasing map for "
                            "note rendering, screened by what the "
                            "released cohort already carries")
    p.add_argument("--omop", help="a directory of OMOP-shaped csvs")
    p.add_argument("--table", action="append", default=[],
                   help="one table carrying a code beside a name, "
                        "repeatable")
    p.add_argument("--cohort",
                   help="the released cohort csv; codes absent "
                        "from it are DROPPED, because naming a "
                        "code the k rule withheld discloses that "
                        "a patient here has it")
    p.add_argument("--set-column", action="append", default=[],
                   help="restrict the cohort scan to these "
                        "columns")
    p.add_argument("--sep", default=";")
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_codemap)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
