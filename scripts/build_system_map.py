"""Build the Smart Problem List system atlas.

Generates docs/system_map.html: a single self-contained interactive
drill-down map of the whole system, from pipeline overview down to
live source code.

The map never paraphrases code. Every component names a real symbol in
a real file; source is extracted at build time via ast (Python) or a
keyed text scan (YAML). If a symbol moves or vanishes, the build FAILS
LOUDLY. That refusal is the map's accuracy guarantee: rebuild after
any code change.

Usage:
    python scripts/build_system_map.py
"""

import ast
import datetime
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "docs", "system_map.html")
MAX_SNIPPET_LINES = 90


def drift(symbol, path):
    sys.exit("MAP DRIFT: `%s` not found in %s -- update scripts/build_system_map.py" % (symbol, path))


def truncate(lines):
    if len(lines) <= MAX_SNIPPET_LINES:
        return lines
    kept = lines[:MAX_SNIPPET_LINES]
    kept.append("# ... %d more lines" % (len(lines) - MAX_SNIPPET_LINES))
    return kept


def extract_py(rel_path, dotted):
    """Extract a function, class, method, or module-level assignment."""
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        drift(dotted, rel_path)
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()
    tree = ast.parse(source)
    src_lines = source.splitlines()

    parts = dotted.split(".")

    def find_in(body, name):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                return node
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        return node
        return None

    node = None
    body = tree.body
    for part in parts:
        node = find_in(body, part)
        if node is None:
            drift(dotted, rel_path)
        body = getattr(node, "body", [])

    start = node.lineno
    if getattr(node, "decorator_list", None):
        start = min(d.lineno for d in node.decorator_list)
    end = node.end_lineno
    lines = src_lines[start - 1:end]

    indent = len(lines[0]) - len(lines[0].lstrip())
    if indent:
        lines = [ln[indent:] if len(ln) >= indent else ln for ln in lines]
    return "\n".join(truncate(lines))


def extract_yaml(rel_path, dotted):
    """Extract a top-level or one-level-nested key block from a YAML file."""
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        drift(dotted, rel_path)
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    parts = dotted.split(".")
    key = parts[-1]
    indent = 2 * (len(parts) - 1)
    pattern = re.compile(r"^" + (" " * indent) + re.escape(key) + r":")

    start = None
    for i, line in enumerate(lines):
        if pattern.match(line):
            start = i
            break
    if start is None:
        drift(dotted, rel_path)

    block = [lines[start]]
    for line in lines[start + 1:]:
        stripped = line.strip()
        if stripped and not line.startswith(" " * (indent + 1)) and not stripped.startswith("#"):
            break
        block.append(line)
    while block and not block[-1].strip():
        block.pop()
    if indent:
        block = [ln[indent:] if len(ln) >= indent else ln for ln in block]
    return "\n".join(truncate(block))


def comp(name, file, symbol, blurb, lang="py"):
    return {"name": name, "file": file, "symbol": symbol, "blurb": blurb, "lang": lang}


ATLAS = {
    "project": "SMART PROBLEM LIST",
    "input_label": "a stale PROBLEM extract \u2192",
    "output_label": "\u2192 a clinician-ready review queue",
    "domains": [
        {
            "id": "intake", "name": "INTAKE", "kind": "stage", "color": "#2E6E8E",
            "stage": "01 INTAKE",
            "note": "xlsx rows in; dates parsed,\n12/31/2100 sentinel unmasked",
            "blurb": "Reads the Cerner Millennium PROBLEM extract without trusting its shape: headers are matched case-insensitively, dates arrive as Excel datetimes or half a dozen string formats, and the mysterious 12/31/00 end date is unmasked as Millennium's end-of-time sentinel rather than treated as data.",
            "modules": [
                {"name": "date semantics", "file": "spl/ingest.py", "components": [
                    comp("parse_date", "spl/ingest.py", "parse_date",
                         "Extracts of this table come from CCL, Discern Analytics, or ad-hoc queries, and each renders dates differently. This tries real datetimes first, then six string formats, and returns None rather than guessing -- a wrong date here would corrupt every window downstream."),
                    comp("is_sentinel_end_date", "spl/ingest.py", "is_sentinel_end_date",
                         "The founding design call of the project. Every row showed END_EFFECTIVE_DT_TM = 12/31/00, which is Millennium's end-of-time sentinel 12/31/2100 rendered with a two-digit year. Treating it as a null sentinel (not corrupt data to be fixed) is what makes the whole stale-problem framing coherent. Handles the true 2100 value, the 2000/1900 misparse, and empty cells."),
                ]},
                {"name": "extract loader", "file": "spl/ingest.py", "components": [
                    comp("EXPECTED_COLUMNS", "spl/ingest.py", "EXPECTED_COLUMNS",
                         "The 13 columns confirmed from the real extract layout. Only SOURCE_STRING and ACTIVE_STATUS_DT_TM are hard requirements -- everything else degrades gracefully, because the next extract will not have the same shape as this one."),
                    comp("load_problem_rows", "spl/ingest.py", "load_problem_rows",
                         "Builds one dict per row, parses every date column, and computes IS_OPEN_ACTIVE: sentinel end date AND status ACTIVE (code 3301). That flag is the scope gate -- rows with a real end date are out of scope by definition, and estimating on them would be noise."),
                ]},
            ],
        },
        {
            "id": "resolve", "name": "RESOLVE", "kind": "stage", "color": "#6E5A9E",
            "stage": "02 RESOLVE",
            "note": "free-text SOURCE_STRING collapsed\nto canonical condition classes",
            "blurb": "Collapses the free-text chaos of SOURCE_STRING into canonical condition classes through four tiers ordered by trust: exact alias, token rules, SOURCE_IDENTIFIER propagation, and an optional local LLM. Five observed sinusitis spellings become one class; a mangled string recovers through its shared code.",
            "modules": [
                {"name": "string hygiene", "file": "spl/normalize.py", "components": [
                    comp("clean", "spl/normalize.py", "clean",
                         "Built for the trailing periods in the real data: 'Acute bacterial sinusitis.' and 'Scaphoid fracture of wrist.' both carry punctuation that would break exact matching. Lowercase, strip trailing punctuation, collapse whitespace -- nothing more aggressive, so the cleaned string stays recognizable to a reviewer."),
                    comp("tokens", "spl/normalize.py", "tokens",
                         "Reduces a cleaned string to its word set. This is the substrate for token rules: 'acute maxillary sinusitis' and 'acute pansinusitis' share the tokens that matter (acute, sinusitis) and differ only in anatomy tokens the rules ignore."),
                ]},
                {"name": "match tiers", "file": "spl/normalize.py", "components": [
                    comp("TokenRule", "spl/normalize.py", "TokenRule",
                         "Required-tokens plus any-of-tokens matching. The acute_sinusitis rule requires only {acute, sinusitis}, which is why all five observed anatomic variants (maxillary, frontal, pan-, bacterial, plain) collapse to one duration class without enumerating them."),
                    comp("resolve_row", "spl/normalize.py", "resolve_row",
                         "Tier order is trust order: exact alias match (confidence 1.0) before token rules (0.85). A string that exactly matches a curated alias needs no inference; token matching is the controlled fallback for spellings the dictionary has not seen yet."),
                    comp("resolve_all", "spl/normalize.py", "resolve_all",
                         "The code-propagation tier. Rows sharing a SOURCE_IDENTIFIER with a string-resolved row inherit its class -- but only when every vote for that code agrees (len(votes)==1). In the sample run this recovered 'Ac. max. sinus infection', unmatchable by any string rule, through its shared SNMJ01 code."),
                ]},
                {"name": "local llm fallback", "file": "spl/llm_fallback.py", "components": [
                    comp("PROMPT_TEMPLATE", "spl/llm_fallback.py", "PROMPT_TEMPLATE",
                         "Strict JSON output, temperature 0, and an explicit UNKNOWN bias: the model must choose UNKNOWN unless the string clearly denotes a known key. A creative LLM is exactly what a clinical mapping pipeline does not want."),
                    comp("OllamaClassifier", "spl/llm_fallback.py", "OllamaClassifier",
                         "Off by default (--llm to enable), talks only to localhost Ollama, and whitelist-validates the answer against known keys. Runs on-machine so no PHI-bearing context ever leaves the host -- the hospital-employer constraint that shaped the whole tier."),
                ]},
            ],
        },
        {
            "id": "estimate", "name": "ESTIMATE", "kind": "stage", "color": "#2F7D5B",
            "stage": "03 ESTIMATE",
            "note": "resolution windows + one of\nfour statuses per open problem",
            "blurb": "Pure date arithmetic against the knowledge base: min/max resolution dates off ACTIVE_STATUS_DT_TM, then one of four statuses against the as-of date. Nothing here ever writes to a source field -- estimates live in EST_* columns and closure remains a human decision.",
            "modules": [
                {"name": "classification", "file": "spl/estimate.py", "components": [
                    comp("estimate_row", "spl/estimate.py", "estimate_row",
                         "The four-status contract: LIKELY_RESOLVED, POSSIBLY_ACTIVE, CHRONIC_NO_EXPIRY, NEEDS_REVIEW. Red flags and chronic conditions exit before any window math. The build's only test failure happened here: unmapped red-flagged rows carried None confidence and crashed a comparison -- now pinned to an explicit 0.0 with red flags still surfaced in the rationale."),
                    comp("estimate_all", "spl/estimate.py", "estimate_all",
                         "Deliberately trivial batch wrapper. Every decision lives in estimate_row so a single row is independently testable -- which is how the pytest fixtures exercise the classifier without touching xlsx at all."),
                ]},
            ],
        },
        {
            "id": "report", "name": "REPORT", "kind": "stage", "color": "#B0662A",
            "stage": "04 REPORT",
            "note": "color-coded review workbook;\nlive formulas, sources untouched",
            "blurb": "Writes the clinician-facing workbook: AUGMENTED (original columns plus EST_* fields, status color-coded, END_EFFECTIVE_DT_TM deliberately preserved), SUMMARY with live COUNTIF formulas, REVIEW_QUEUE oldest-first, and UNMAPPED as the dictionary-growth backlog.",
            "modules": [
                {"name": "sheet writers", "file": "spl/report.py", "components": [
                    comp("_value_for", "spl/report.py", "_value_for",
                         "Prefers the parsed datetime, falls back to the raw cell. The raw fallback matters: an unparseable date still appears in the review sheet as whatever the extract carried, so nothing silently disappears from a clinician's view."),
                    comp("_write_rows", "spl/report.py", "_write_rows",
                         "Applies mm/dd/yyyy formats to date columns and the status color fill to EST_STATUS only -- one colored cell per row, not a painted row, so the sheet stays readable when a reviewer sorts or filters it."),
                    comp("STATUS_FILLS", "spl/report.py", "STATUS_FILLS",
                         "Green for likely-resolved, amber for possibly-active, red for needs-review, blue for chronic. Standard Excel conditional-format colors, chosen so the sheet reads correctly to people who live in Excel all day."),
                ]},
                {"name": "workbook assembly", "file": "spl/report.py", "components": [
                    comp("write_report", "spl/report.py", "write_report",
                         "Builds all four sheets. SUMMARY counts are live COUNTIF formulas against the AUGMENTED sheet, never Python-computed literals, so the workbook recalculates if a reviewer edits rows. The sample run's 9 formulas recalculated with zero errors and matched the CLI counts exactly (16/2/4/5 across 27 rows)."),
                ]},
            ],
        },
        {
            "id": "knowledge", "name": "CLINICAL KNOWLEDGE", "kind": "rail", "color": "#8E3B4A",
            "stage": "KNOWLEDGE RAIL",
            "note": "the sign-off artifact: durations, aliases, red flags",
            "blurb": "The clinical judgment of the system, deliberately kept out of code. durations.yaml is the artifact clinical stakeholders review and sign off on; the loader turns it into alias indexes and red-flag detection consumed by RESOLVE and ESTIMATE alike.",
            "modules": [
                {"name": "durations.yaml", "file": "knowledge/durations.yaml", "components": [
                    comp("acute_sinusitis", "knowledge/durations.yaml", "conditions.acute_sinusitis",
                         "The condition that motivated the project. Viral courses resolve in 7-10 days, bacterial up to 4 weeks; the 14-60 day window is deliberately conservative so an uncomplicated case is essentially never flagged early. Eight aliases cover every observed anatomic spelling.", "yaml"),
                    comp("scaphoid_fracture", "knowledge/durations.yaml", "conditions.scaphoid_fracture",
                         "6-12 weeks of immobilization is the textbook course, but scaphoid nonunion risk is real -- so the honest max is 180 days, not 90. The example of why these windows need a clinician's sign-off rather than an engineer's confidence.", "yaml"),
                    comp("red_flag_tokens", "knowledge/durations.yaml", "red_flag_tokens",
                         "chronic, recurrent, nonunion and friends invalidate the simple acute window even when the head concept matches. 'Chronic sinusitis' must never inherit acute sinusitis's 60-day window -- these tokens route it to NEEDS_REVIEW instead.", "yaml"),
                ]},
                {"name": "kb loader", "file": "spl/normalize.py", "components": [
                    comp("KnowledgeBase", "spl/normalize.py", "KnowledgeBase",
                         "Loads the YAML once, builds the cleaned-alias index, and exposes red-flag detection over both token sets and substrings (catching hyphenated forms like 'non-union'). Everything downstream consumes this object, never the raw YAML."),
                ]},
            ],
        },
        {
            "id": "quality", "name": "QUALITY GATES", "kind": "rail", "color": "#4A6B3B",
            "stage": "QUALITY RAIL",
            "note": "19 tests; the observed problem strings are the fixtures",
            "blurb": "The 19-test pytest suite uses the exact strings observed in the real data as fixtures, plus a 27-row synthetic extract that mirrors the described layout column-for-column. The suite caught the project's only build-time bug before any data did.",
            "modules": [
                {"name": "smoke fixtures", "file": "tests/test_pipeline.py", "components": [
                    comp("sinusitis variants", "tests/test_pipeline.py", "TestNormalization.test_all_sinusitis_variants_map_to_one_class",
                         "Parametrized over the five sinusitis spellings quoted from the real data, asserting they all collapse to one class. If a normalize refactor breaks anatomic collapsing, this fails first."),
                    comp("2013 window math", "tests/test_pipeline.py", "TestEstimation.test_2013_sinusitis_is_likely_resolved",
                         "Pins the exact window dates for the canonical example: diagnosed 9/25/13, min resolution 10/9/13, max 11/24/13, LIKELY_RESOLVED as of 2026. Exact-date assertions mean an off-by-one in timedelta math cannot hide."),
                    comp("never mutate sources", "tests/test_pipeline.py", "TestEstimation.test_source_dates_never_mutated",
                         "The architectural invariant as a test: after a full pipeline run, END_EFFECTIVE_DT_TM still carries the sentinel. If anyone ever makes the estimator write back, this test is the tripwire."),
                ]},
                {"name": "synthetic extract", "file": "tools/make_sample_data.py", "components": [
                    comp("ROWS", "tools/make_sample_data.py", "ROWS",
                         "27 fabricated rows spanning every classification path: the real example strings and 2013-era dates, two recent cases that must stay POSSIBLY_ACTIVE, four chronics that must stay untouched, red-flag traps, and one deliberately mangled string that only code propagation can recover."),
                    comp("main", "tools/make_sample_data.py", "main",
                         "Writes the extract with the described layout exactly: all rows ACTIVE/3301, every END_EFFECTIVE_DT_TM set to the 12/31/2100 sentinel, ONSET and LIFE_CYCLE dates sparsely populated the way the real data is."),
                ]},
            ],
        },
        {
            "id": "orchestration", "name": "ORCHESTRATION", "kind": "rail", "color": "#56606B",
            "stage": "ORCHESTRATION RAIL",
            "note": "one command: extract in, review workbook out",
            "blurb": "One CLI command runs the whole journey: load, resolve, estimate, report, and print the status counts. The --as-of flag makes runs reproducible for tests and demos; --llm opts in to the local fallback tier.",
            "modules": [
                {"name": "cli runner", "file": "run_poc.py", "components": [
                    comp("main", "run_poc.py", "main",
                         "The pipeline in eleven effective lines: KB, rows, resolve, estimate, report. The printed status counts were how the sample run's 16/2/4/5 split was first verified against the workbook's COUNTIF formulas -- two independent counts of the same classification."),
                ]},
            ],
        },
    ],
}


def build_pipeline_svg(domains):
    """Layer 1: the data's journey, left to right, rails beneath."""
    stages = [d for d in domains if d["kind"] == "stage"]
    rails = [d for d in domains if d["kind"] == "rail"]

    node_w, node_h, gap = 158, 60, 74
    left = 226
    flow_y = 112
    parts = []
    parts.append('<svg id="pipeline" viewBox="0 0 1180 %d" xmlns="http://www.w3.org/2000/svg">'
                 % (238 + len(rails) * 42 + 16))
    parts.append('<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" '
                 'markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#5C6B78"/></marker></defs>')

    parts.append('<text x="%d" y="%d" class="io-label" text-anchor="end">%s</text>'
                 % (left - 22, flow_y + 4, ATLAS["input_label"]))

    for i, d in enumerate(stages):
        x = left + i * (node_w + gap)
        cy = flow_y - node_h // 2
        parts.append('<g class="clickable" onclick="showDomain(\'%s\')">' % d["id"])
        parts.append('<rect x="%d" y="%d" width="%d" height="%d" rx="6" fill="#FFFFFF" '
                     'stroke="%s" stroke-width="1.4"/>' % (x, cy, node_w, node_h, d["color"]))
        num, name = d["stage"].split(" ", 1)
        parts.append('<text x="%d" y="%d" class="stage-num" text-anchor="middle" fill="%s">%s</text>'
                     % (x + node_w // 2, cy + 22, d["color"], num))
        parts.append('<text x="%d" y="%d" class="stage-name" text-anchor="middle">%s</text>'
                     % (x + node_w // 2, cy + 43, name))
        parts.append('</g>')

        if i < len(stages) - 1:
            parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#5C6B78" '
                         'stroke-width="1.4" marker-end="url(#arr)"/>'
                         % (x + node_w, flow_y, x + node_w + gap - 9, flow_y))

        above = (i % 2 == 0)
        note_y = flow_y - node_h // 2 - 34 if above else flow_y + node_h // 2 + 30
        tie_y1 = cy if above else cy + node_h
        tie_y2 = note_y + 12 if above else note_y - 12
        parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1" opacity="0.45"/>'
                     % (x + node_w // 2, tie_y1, x + node_w // 2, tie_y2, d["color"]))
        note_lines = d["note"].split("\n")
        for j, line in enumerate(note_lines):
            parts.append('<text x="%d" y="%d" class="stage-note" text-anchor="middle">%s</text>'
                         % (x + node_w // 2, note_y + j * 15, line))

    last_x = left + (len(stages) - 1) * (node_w + gap) + node_w
    parts.append('<text x="%d" y="%d" class="io-label">%s</text>'
                 % (last_x + 22, flow_y + 4, ATLAS["output_label"]))

    rail_left, rail_right = left, last_x
    rail_top = flow_y + node_h // 2 + 66
    for i, d in enumerate(stages):
        x = left + i * (node_w + gap) + node_w // 2
        parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#9AA7B2" stroke-width="1" '
                     'stroke-dasharray="2,4" opacity="0.6"/>'
                     % (x, flow_y + node_h // 2 + 32, x, rail_top - 6))

    for r, d in enumerate(rails):
        y = rail_top + r * 42
        parts.append('<g class="clickable" onclick="showDomain(\'%s\')">' % d["id"])
        parts.append('<rect x="%d" y="%d" width="%d" height="32" rx="4" fill="#FFFFFF" '
                     'stroke="#C9D2DA" stroke-width="1"/>' % (rail_left, y, rail_right - rail_left))
        parts.append('<rect x="%d" y="%d" width="4" height="32" rx="2" fill="%s"/>' % (rail_left, y, d["color"]))
        parts.append('<text x="%d" y="%d" class="rail-name" fill="%s">%s</text>'
                     % (rail_left + 16, y + 20, d["color"], d["name"]))
        parts.append('<text x="%d" y="%d" class="rail-note" text-anchor="end">%s</text>'
                     % (rail_right - 14, y + 20, d["note"]))
        parts.append('</g>')

    parts.append('</svg>')
    return "".join(parts)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Problem List -- system atlas</title>
<style>
:root {
  --bg: #EDF0F3; --panel: #FFFFFF; --line: #C9D2DA;
  --ink: #17222C; --dim: #5C6B78;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace;
}
* { box-sizing: border-box; margin: 0; }
body { background: var(--bg); color: var(--ink); font-family: var(--mono); font-size: 14px; }
header { display: flex; align-items: baseline; gap: 14px; padding: 18px 28px 10px;
  border-bottom: 1px solid var(--line); background: var(--panel); flex-wrap: wrap; }
.wordmark { font-weight: 700; font-size: 17px; letter-spacing: 0.06em; }
.tagline { color: var(--dim); font-size: 12px; }
.stamp { color: var(--dim); font-size: 12px; margin-left: auto; }
#crumbs { padding: 10px 28px 0; text-align: right; font-size: 12px; color: var(--dim); min-height: 26px; }
#crumbs a { color: var(--ink); text-decoration: none; border-bottom: 1px dotted var(--dim); cursor: pointer; }
#crumbs a:hover { border-bottom-style: solid; }
main { max-width: 1220px; margin: 0 auto; padding: 8px 20px 30px; }
.view { animation: fade .16s ease; }
@keyframes fade { from { opacity: 0 } to { opacity: 1 } }
@media (prefers-reduced-motion: reduce) { .view { animation: none } }
.hidden { display: none; }

#pipeline { width: 100%; height: auto; display: block; }
.clickable { cursor: pointer; }
.clickable:hover rect { filter: brightness(0.965); }
.io-label { font: 700 14px var(--mono); fill: var(--ink); }
.stage-num { font: 700 11px var(--mono); letter-spacing: 0.12em; }
.stage-name { font: 700 13px var(--mono); fill: var(--ink); }
.stage-note { font: 11.5px var(--mono); fill: var(--dim); }
.rail-name { font: 700 11.5px var(--mono); letter-spacing: 0.08em; }
.rail-note { font: 11px var(--mono); fill: var(--dim); }

.domain-head { display: flex; align-items: baseline; gap: 12px; margin: 18px 2px 4px; flex-wrap: wrap; }
.domain-dot { width: 11px; height: 11px; border-radius: 3px; align-self: center; flex: none; }
.domain-title { font-weight: 700; font-size: 15px; letter-spacing: 0.06em; }
.domain-blurb { color: var(--dim); font-size: 12.5px; max-width: 860px; margin: 4px 2px 10px; line-height: 1.55; }
#curves { width: 100%; height: 64px; display: block; }
.cards { display: flex; gap: 16px; flex-wrap: wrap; }
.card { background: var(--panel); border: 1px solid var(--line); border-left-width: 4px;
  border-radius: 6px; padding: 13px 16px; min-width: 218px; cursor: pointer; }
.card:hover { filter: brightness(0.975); }
.card .m-name { font-weight: 700; margin-bottom: 5px; }
.card .m-meta { color: var(--dim); font-size: 11.5px; line-height: 1.6; }

.explorer { display: flex; gap: 16px; align-items: flex-start; margin-top: 12px; }
.chips { width: 250px; flex: none; display: flex; flex-direction: column; gap: 8px; }
.chip { text-align: left; background: var(--panel); border: 1px solid var(--line);
  border-left-width: 4px; border-radius: 5px; padding: 9px 12px; font: 700 12.5px var(--mono);
  color: var(--ink); cursor: pointer; }
.chip:hover { filter: brightness(0.975); }
.chip.active { background: #F2F5F8; }
.chip:focus-visible, .card:focus-visible { outline: 2px solid var(--ink); outline-offset: 1px; }
.codebox { flex: 1; background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
  min-width: 0; overflow: hidden; }
.codebox .cb-head { display: flex; gap: 10px; align-items: baseline; padding: 10px 16px;
  border-bottom: 1px solid var(--line); font-size: 12px; flex-wrap: wrap; }
.cb-path { color: var(--dim); }
.cb-symbol { font-weight: 700; }
.cb-blurb { padding: 12px 16px; color: var(--dim); font-size: 12.5px; line-height: 1.6;
  border-bottom: 1px solid var(--line); }
pre { padding: 14px 16px; overflow-x: auto; font: 12.5px/1.55 var(--mono); }
.tok-c { color: #2F7D3B; font-style: italic; }
.tok-s { color: #A0522D; }
.tok-k { color: #1F5FA8; font-weight: 700; }

.steps { margin: 10px 2px 8px; max-width: 920px; }
.atlas-steps { margin-top: 2px; }
.steps-title { font-size: 11px; letter-spacing: 0.14em; color: var(--dim); margin-bottom: 9px; font-weight: 700; }
.step { display: flex; gap: 11px; margin: 8px 0; line-height: 1.55; font-size: 12.5px; align-items: baseline; }
.step-num { flex: none; min-width: 23px; height: 23px; border-radius: 6px; color: #FFFFFF;
  font-weight: 700; font-size: 11.5px; display: inline-flex; align-items: center;
  justify-content: center; transform: translateY(4px); }
.step-what { font-weight: 700; }
.step-why { color: var(--dim); }
.cb-walk { padding: 12px 16px; border-bottom: 1px solid var(--line); }
.cb-walk .steps { margin: 0; }

footer { border-top: 1px solid var(--line); background: var(--panel); color: var(--dim);
  font-size: 12px; padding: 12px 28px; display: flex; gap: 18px; flex-wrap: wrap; }
@media (max-width: 760px) { .explorer { flex-direction: column } .chips { width: 100% } }
</style>
</head>
<body>
<header>
  <span class="wordmark">SMART PROBLEM LIST</span>
  <span class="tagline">system atlas &middot; click nodes to descend</span>
  <span class="stamp">__STAMP__</span>
</header>
<div id="crumbs"></div>
<main>
  <div id="view-atlas" class="view">__PIPELINE_SVG__
__PLAIN_STEPS__</div>
  <div id="view-domain" class="view hidden"></div>
  <div id="view-module" class="view hidden"></div>
</main>
<footer>
  <span>__COUNTS__</span>
  <span>rebuild: <b>python scripts/build_system_map.py</b> &mdash; rerun after any code change; a moved symbol fails the build, and that refusal is this map's accuracy guarantee</span>
</footer>
<script type="application/json" id="atlas-data">__ATLAS_JSON__</script>
<script>
var DATA = JSON.parse(document.getElementById('atlas-data').textContent);
var byId = {};
DATA.domains.forEach(function (d) { byId[d.id] = d; });

function esc(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

var PY_KW = /\b(def|class|return|if|elif|else|for|while|in|not|and|or|None|True|False|import|from|try|except|raise|with|as|lambda|is|continue|break|pass|global|assert|yield)\b/g;

function hl(code, lang) {
  var out = esc(code);
  out = out.replace(/("{3}|'{3}|"|')((?:\\.|(?!\1)[\s\S])*?)\1|(#[^\n]*)/g, function (m, q, body, comment) {
    if (comment !== undefined) return '<span class="tok-c">' + comment + '</span>';
    return '<span class="tok-s">' + m + '</span>';
  });
  if (lang === 'py') {
    out = out.split(/(<span[\s\S]*?<\/span>)/).map(function (part) {
      if (part.indexOf('<span') === 0) return part;
      return part.replace(PY_KW, '<span class="tok-k">$1</span>');
    }).join('');
  }
  return out;
}

function stepsHtml(steps, color, title) {
  var items = steps.map(function (s, i) {
    return '<div class="step"><span class="step-num" style="background:' + color + '">' + (i + 1) + '</span>' +
      '<span><span class="step-what">' + esc(s.t) + '</span> ' +
      '<span class="step-why">&mdash; ' + esc(s.w) + '</span></span></div>';
  }).join('');
  return '<div class="steps"><div class="steps-title">' + esc(title) + '</div>' + items + '</div>';
}

function setCrumbs(items) {
  var el = document.getElementById('crumbs');
  el.innerHTML = items.map(function (it, i) {
    if (i === items.length - 1 || !it.fn) return '<span>' + esc(it.label) + '</span>';
    return '<a onclick="' + it.fn + '">' + esc(it.label) + '</a>';
  }).join(' / ');
}

function show(id) {
  ['view-atlas', 'view-domain', 'view-module'].forEach(function (v) {
    var el = document.getElementById(v);
    el.classList.toggle('hidden', v !== id);
    if (v === id) { el.classList.remove('view'); void el.offsetWidth; el.classList.add('view'); }
  });
}

function showAtlas() {
  setCrumbs([{ label: 'atlas' }]);
  show('view-atlas');
}

function showDomain(id) {
  var d = byId[id];
  var n = d.modules.length;
  var curves = d.modules.map(function (m, i) {
    var x = (i + 0.5) * (1000 / n);
    return '<path d="M500,0 C500,40 ' + x + ',18 ' + x + ',62" fill="none" stroke="' +
      d.color + '" stroke-width="1.2" opacity="0.5"/>';
  }).join('');
  var cards = d.modules.map(function (m, i) {
    return '<div class="card" tabindex="0" role="button" style="border-left-color:' + d.color +
      '" onclick="showModule(\'' + id + '\',' + i + ')" onkeydown="if(event.key===\'Enter\')showModule(\'' + id + '\',' + i + ')">' +
      '<div class="m-name">' + esc(m.name) + '</div>' +
      '<div class="m-meta">' + m.components.length + ' component' + (m.components.length === 1 ? '' : 's') +
      '<br>' + esc(m.file) + '</div></div>';
  }).join('');
  document.getElementById('view-domain').innerHTML =
    '<div class="domain-head"><span class="domain-dot" style="background:' + d.color + '"></span>' +
    '<span class="domain-title">' + esc(d.name) + '</span>' +
    '<span class="tagline">' + esc(d.stage) + '</span></div>' +
    '<p class="domain-blurb">' + esc(d.blurb) + '</p>' +
    stepsHtml(d.steps, d.color, 'WHAT HAPPENS HERE, STEP BY STEP') +
    '<svg id="curves" viewBox="0 0 1000 64" preserveAspectRatio="none">' + curves + '</svg>' +
    '<div class="cards">' + cards + '</div>';
  setCrumbs([{ label: 'atlas', fn: 'showAtlas()' }, { label: d.name.toLowerCase() }]);
  show('view-domain');
}

function showModule(id, mi) {
  var d = byId[id], m = d.modules[mi];
  var chips = m.components.map(function (c, ci) {
    return '<button class="chip" id="chip-' + ci + '" style="border-left-color:' + d.color +
      '" onclick="selectComp(\'' + id + '\',' + mi + ',' + ci + ')">' + esc(c.name) + '</button>';
  }).join('');
  document.getElementById('view-module').innerHTML =
    '<div class="domain-head"><span class="domain-dot" style="background:' + d.color + '"></span>' +
    '<span class="domain-title">' + esc(m.name) + '</span>' +
    '<span class="tagline">' + esc(m.file) + '</span></div>' +
    '<div class="explorer"><div class="chips">' + chips + '</div>' +
    '<div class="codebox"><div class="cb-head"><span class="cb-path" id="cb-path"></span>' +
    '<span class="cb-symbol" id="cb-symbol"></span></div>' +
    '<div class="cb-blurb" id="cb-blurb"></div><div class="cb-walk" id="cb-walk"></div><pre id="cb-code"></pre></div></div>';
  setCrumbs([
    { label: 'atlas', fn: 'showAtlas()' },
    { label: d.name.toLowerCase(), fn: "showDomain('" + id + "')" },
    { label: m.name }
  ]);
  show('view-module');
  selectComp(id, mi, 0);
}

function selectComp(id, mi, ci) {
  var d = byId[id], m = d.modules[mi], c = m.components[ci];
  m.components.forEach(function (_, i) {
    document.getElementById('chip-' + i).classList.toggle('active', i === ci);
  });
  document.getElementById('cb-path').textContent = c.file;
  document.getElementById('cb-symbol').textContent = c.symbol;
  document.getElementById('cb-blurb').textContent = c.blurb;
  document.getElementById('cb-walk').innerHTML = stepsHtml(c.walk, d.color, 'PLAIN ENGLISH WALKTHROUGH');
  document.getElementById('cb-code').innerHTML = hl(c.code, c.lang);
}

showAtlas();
</script>
</body>
</html>
"""


def step(t, w):
    return {"t": t, "w": w}


NARRATIVE = {
    "domains": {
        "intake": {
            "plain": step("Read the hospital's spreadsheet and keep every problem still marked open",
                          "we can only judge what we can reliably read, and closed problems don't need our help"),
            "steps": [
                step("Open the spreadsheet, however it arrives",
                     "column headers are matched loosely, so a slightly different export still loads"),
                step("Read every date carefully",
                     "dates arrive in many different spellings, and one misread date would corrupt every estimate downstream"),
                step("Spot the fake end date",
                     "12/31/2100 is Cerner's placeholder for 'no end date was ever recorded' -- it is not a real date"),
                step("Keep only the open, active problems",
                     "rows that already have a real end date are someone else's success story; judging them would just add noise"),
            ],
        },
        "resolve": {
            "plain": step("Work out which medical condition each row actually means",
                          "the same illness is written a dozen different ways by different people"),
            "steps": [
                step("Tidy the text",
                     "the data really contains entries like 'Acute sinusitis.' with a stray period; tidying makes equals look equal"),
                step("Try the exact dictionary first",
                     "a curated list of known names; a perfect match needs no guessing at all"),
                step("Then try word patterns",
                     "'acute' plus 'sinusitis' catches the maxillary, frontal, and pan- variants without listing every spelling"),
                step("Then borrow from sibling rows",
                     "rows sharing the same source code inherit an answer we already trust -- but only when every such row agrees"),
                step("Only then, optionally, ask a local AI",
                     "for the leftovers only, on this machine only, choosing only from our known list -- it can never invent a condition"),
            ],
        },
        "estimate": {
            "plain": step("Ask how long that condition normally lasts, and compare with today",
                          "a sinus infection that has supposedly lasted thirteen years is not believable"),
            "steps": [
                step("Look up the normal course of the condition",
                     "the clinical knowledge file says how long each condition usually takes to resolve"),
                step("Check for warning words first",
                     "'chronic', 'recurrent', or 'nonunion' mean the simple timeline doesn't apply -- those rows go straight to a human"),
                step("Leave lifelong conditions alone",
                     "an open-ended diabetes or hypertension entry is medically correct, not stale"),
                step("Do the calendar math",
                     "diagnosis date plus normal course gives the window in which the problem should have resolved"),
                step("Compare the window with today",
                     "far past the window means likely resolved; still inside it means possibly active"),
                step("Never touch the original record",
                     "every estimate lives in new columns; closing a problem remains a human decision"),
            ],
        },
        "report": {
            "plain": step("Hand a color-coded review workbook to a human",
                          "the system suggests -- a clinician decides"),
            "steps": [
                step("Rebuild the spreadsheet with our columns added",
                     "reviewers see every original field untouched, side by side with the estimates"),
                step("Color each verdict",
                     "green likely resolved, amber possibly active, red needs review, blue chronic -- readable at a glance"),
                step("Build the review queue, oldest first",
                     "human attention is the scarce resource, so it goes to the longest-standing questions first"),
                step("List everything we couldn't name",
                     "unmapped condition names, ranked by how often they occur, show exactly where the dictionary should grow next"),
                step("Keep the summary counts live",
                     "the totals are real spreadsheet formulas, so they recalculate if a reviewer edits rows"),
            ],
        },
        "knowledge": {
            "plain": step("Keep every medical assumption in one reviewable file",
                          "clinicians can check and approve the rules without reading a line of code"),
            "steps": [
                step("Keep medicine out of the code",
                     "condition durations, name variants, and warning words all live in one plain-text file"),
                step("Make that file the sign-off artifact",
                     "a clinical stakeholder reviews and approves this file, and that approval is what makes the tool deployable"),
                step("Load it once, use it everywhere",
                     "both the naming step and the estimating step draw on the same approved rules -- one source of truth"),
            ],
        },
        "quality": {
            "plain": step("Guard every behavior with automated tests",
                          "changes cannot silently break the logic -- 19 checks run in under a second"),
            "steps": [
                step("Test with the real problem strings",
                     "the exact spellings observed in the data are the test cases, so the tests defend against reality"),
                step("Pin the math to exact dates",
                     "the tests demand specific resolution dates, so an off-by-one-day bug cannot hide"),
                step("Guard the golden rule with a tripwire",
                     "a dedicated test fails the build if the pipeline ever writes into the medical record's own fields"),
                step("Rehearse everything on invented data",
                     "a 27-row synthetic extract covers every path through the system using zero real patient data"),
            ],
        },
        "orchestration": {
            "plain": step("Run the whole journey with one command",
                          "anyone can reproduce a result -- extract in, review workbook out"),
            "steps": [
                step("Take just two file names",
                     "the input extract and the output workbook; an optional 'pretend today is...' date makes demos repeatable"),
                step("Run the four stages in a fixed order",
                     "load, name, estimate, report -- the same journey every single time"),
                step("Print the verdict counts at the end",
                     "an independent tally to check against the workbook's own formulas -- two counts of the same truth"),
            ],
        },
    },
    "walkthroughs": {
        "spl/ingest.py::parse_date": [
            step("If it's already a real date, keep it", "Excel sometimes stores true dates, and we never second-guess what is already right"),
            step("Otherwise try six known date spellings", "different export tools write dates in different formats"),
            step("If none fit, answer 'unknown'", "an honest blank is safer than a guessed date in a clinical tool"),
        ],
        "spl/ingest.py::is_sentinel_end_date": [
            step("Treat an empty cell as 'no end date recorded'", "absence of a date is information, not an error"),
            step("Treat year 2100 as 'no end date recorded'", "that is Cerner's end-of-time placeholder"),
            step("Also catch 12/31/2000 and 12/31/1900", "that is what the placeholder becomes when its two-digit year is misread"),
            step("Anything else counts as a genuine end date", "and a row with a real ending is not our problem to solve"),
        ],
        "spl/ingest.py::EXPECTED_COLUMNS": [
            step("Name the 13 columns we know from the real extract", "this is the confirmed shape of the source data"),
            step("Let missing columns become blanks, not crashes", "the next extract will not match this one exactly"),
            step("Require only two columns absolutely", "the condition name and the diagnosis date -- without those, nothing can be judged"),
        ],
        "spl/ingest.py::load_problem_rows": [
            step("Match column headers loosely", "capitalization and stray spaces should never break an import"),
            step("Refuse to run if the two essential columns are missing", "refusing loudly beats misreading quietly"),
            step("Parse every date column once, up front", "so the rest of the system never touches raw date text"),
            step("Flag each row as open-and-active or not", "placeholder end date plus ACTIVE status means the row is in scope"),
        ],
        "spl/normalize.py::clean": [
            step("Lowercase everything", "'Acute' and 'acute' must be the same word"),
            step("Strip trailing periods and commas", "the data genuinely contains 'Acute bacterial sinusitis.' with a period"),
            step("Collapse doubled spaces", "invisible differences should not defeat matching"),
            step("Change nothing else", "the cleaned text must stay recognizable to a human reviewer"),
        ],
        "spl/normalize.py::tokens": [
            step("Split the cleaned text into individual words", "matching works on words, not exact sentences"),
            step("Keep them as a set", "word order stops mattering while the meaning remains"),
        ],
        "spl/normalize.py::TokenRule": [
            step("Each rule names its required words", "for example: must contain both 'acute' and 'sinusitis'"),
            step("Some rules add an either/or list", "a wrist fracture must mention one of the wrist bones, whichever one"),
            step("A rule fires only when all its words are present", "nothing subtler, nothing surprising, nothing to debug at 2am"),
        ],
        "spl/normalize.py::resolve_row": [
            step("Try the exact dictionary first", "a perfect match earns full trust and needs no inference"),
            step("Try word patterns second", "slightly less trust, much wider net"),
            step("If neither matches, say so honestly", "'unknown' routes the row to a human, and that is correct behavior, not failure"),
        ],
        "spl/normalize.py::resolve_all": [
            step("Resolve every row by its text first", "the trustworthy tiers get first claim on every answer"),
            step("Record which source codes received which answers", "codes tie spelling variants of the same concept together"),
            step("Share answers across rows with the same code", "but only when every such row agrees -- one dissent cancels the share"),
            step("Optionally send the leftovers to the local AI", "the last resort tier, and only when explicitly switched on"),
            step("Attach a confidence to every answer", "the tier that answered determines how much we trust it, and reviewers see that number"),
        ],
        "spl/llm_fallback.py::PROMPT_TEMPLATE": [
            step("Show the AI the text and our list of known conditions", "it picks from a menu; it does not write freely"),
            step("Demand a one-field JSON answer", "no essays, no explanations, nothing to misread"),
            step("Tell it that UNKNOWN is a good answer", "in a hospital tool, abstaining beats guessing"),
        ],
        "spl/llm_fallback.py::OllamaClassifier": [
            step("Stay off unless explicitly switched on", "the default pipeline is fully deterministic"),
            step("Talk only to this machine", "patient-adjacent text never leaves the building"),
            step("Run with zero randomness", "the same question always gets the same answer"),
            step("Check every answer against our known list", "an invented condition name is silently discarded"),
        ],
        "spl/estimate.py::estimate_row": [
            step("Skip rows that aren't open", "they already have a real ending and are out of scope"),
            step("Send unknown conditions to a human", "with a stated confidence of zero -- no bluffing"),
            step("Send warning-word rows to a human", "chronic and recurrent cases do not follow the simple timeline"),
            step("Leave lifelong conditions alone", "an open-ended diabetes entry is correct, and flagging it would destroy trust"),
            step("Otherwise, do the calendar math", "diagnosis date plus normal course, compared against today"),
            step("Write the reasoning down", "every verdict carries a sentence a reviewer can actually read"),
        ],
        "spl/estimate.py::estimate_all": [
            step("Apply the exact same judgment to every row", "consistency is the point of automating this"),
            step("Stay deliberately tiny", "all the thinking lives in one testable place, one row at a time"),
        ],
        "spl/report.py::_value_for": [
            step("Prefer the cleanly parsed date", "so Excel can sort and filter properly"),
            step("Fall back to whatever the extract said", "nothing silently disappears from a reviewer's view, not even a malformed date"),
        ],
        "spl/report.py::_write_rows": [
            step("Write each row with its verdict columns attached", "originals and estimates side by side"),
            step("Format dates as real dates", "so sorting and filtering behave the way Excel users expect"),
            step("Color only the verdict cell", "one colored cell per row keeps the sheet readable when filtered"),
        ],
        "spl/report.py::STATUS_FILLS": [
            step("Green means likely resolved, amber possibly active, red needs review, blue chronic", "four verdicts, four familiar colors"),
            step("Use standard Excel colors", "instantly readable to people who live in spreadsheets all day"),
        ],
        "spl/report.py::write_report": [
            step("Sheet one: everything original, plus our verdicts", "the full picture, nothing hidden, nothing altered"),
            step("Sheet two: summary counts as live formulas", "they recalculate if a reviewer edits the rows"),
            step("Sheet three: the review queue, oldest first", "the actual to-do list this tool exists to produce"),
            step("Sheet four: what we couldn't name, ranked by count", "the map of where the dictionary should grow next"),
            step("Throughout: original fields untouched", "the workbook is advice, never action"),
        ],
        "knowledge/durations.yaml::conditions.acute_sinusitis": [
            step("Normal course: two weeks to two months", "deliberately wide, so a real uncomplicated case is almost never flagged early"),
            step("Eight known spellings listed", "every variant observed in the data maps to this one entry"),
            step("Confidence 0.90", "high, because the medicine here is well established"),
        ],
        "knowledge/durations.yaml::conditions.scaphoid_fracture": [
            step("Textbook healing: six to twelve weeks", "the standard immobilization course"),
            step("Our maximum: six months", "this particular bone famously heals badly; honesty beats neatness"),
            step("The poster child for clinical sign-off", "these numbers must be a clinician's, not an engineer's"),
        ],
        "knowledge/durations.yaml::red_flag_tokens": [
            step("Seven warning words: chronic, recurrent, nonunion and friends", "each one signals the simple timeline does not apply"),
            step("Any one of them cancels the automatic estimate", "'chronic sinusitis' must never borrow acute sinusitis's two-month window"),
            step("Flagged rows always go to a human", "caution is the default whenever the words hint at complexity"),
        ],
        "spl/normalize.py::KnowledgeBase": [
            step("Read the clinical file once at startup", "one read, one truth, for the whole run"),
            step("Build a fast lookup of every known spelling", "so exact matching is instant"),
            step("Offer warning-word checking to the whole system", "every stage asks the same authority the same way"),
            step("Serve as the only door to the clinical rules", "nothing downstream ever reads the raw file itself"),
        ],
        "tests/test_pipeline.py::TestNormalization.test_all_sinusitis_variants_map_to_one_class": [
            step("Feed in all five real-world sinusitis spellings", "the exact strings quoted from the actual data"),
            step("Demand they all land in one condition class", "anatomic variants share one clinical timeline"),
            step("Fail instantly if a future change breaks this", "the first alarm against a careless refactor"),
        ],
        "tests/test_pipeline.py::TestEstimation.test_2013_sinusitis_is_likely_resolved": [
            step("Use the founding example: sinusitis diagnosed 9/25/2013", "the row that started this whole project"),
            step("Demand the exact window: Oct 9 to Nov 24, 2013", "not roughly right -- exactly right"),
            step("Make an off-by-one-day bug impossible to hide", "exact-date assertions leave it nowhere to live"),
        ],
        "tests/test_pipeline.py::TestEstimation.test_source_dates_never_mutated": [
            step("Run the entire pipeline over a row", "the full journey, not a shortcut"),
            step("Check the original end date is completely unchanged", "byte for byte, placeholder and all"),
            step("Stand guard over the project's core promise", "if anyone ever makes this tool write into the record, this test fails first"),
        ],
        "tools/make_sample_data.py::ROWS": [
            step("27 invented rows, zero real patients", "the entire demo runs without touching patient data"),
            step("Every path through the system is covered", "old infections, fresh cases, lifelong conditions, warning words"),
            step("One string is deliberately garbled", "it proves code-sharing rescues rows that text matching cannot"),
        ],
        "tools/make_sample_data.py::main": [
            step("Write the file in exactly the real extract's shape", "same 13 columns, same quirks"),
            step("Give every row the 12/31/2100 placeholder", "faithfully reproducing the very problem this project exists to detect"),
        ],
        "run_poc.py::main": [
            step("Take two file names and an optional 'pretend today is...' date", "nothing else to remember"),
            step("Run the four stages in a fixed order", "load, name, estimate, report -- every time"),
            step("Print the verdict counts at the end", "an independent tally against the workbook's own formulas"),
            step("Stay eleven effective lines long", "the whole pipeline visible at a glance, nothing hidden"),
        ],
    },
}


def merge_narrative():
    for d in ATLAS["domains"]:
        nar = NARRATIVE["domains"].get(d["id"])
        if not nar:
            sys.exit("NARRATIVE DRIFT: no plain-English narrative for domain `%s` -- update NARRATIVE" % d["id"])
        d["plain"] = nar["plain"]
        d["steps"] = nar["steps"]
        for m in d["modules"]:
            for c in m["components"]:
                key = "%s::%s" % (c["file"], c["symbol"])
                walk = NARRATIVE["walkthroughs"].get(key)
                if not walk:
                    sys.exit("NARRATIVE DRIFT: no plain-English walkthrough for `%s` -- update NARRATIVE" % key)
                c["walk"] = walk


def build_plain_steps(domains):
    """Layer 1: the whole pipeline as numbered plain-English steps."""
    stages = [d for d in domains if d["kind"] == "stage"]
    rails = [d for d in domains if d["kind"] == "rail"]
    parts = ['<div class="steps atlas-steps"><div class="steps-title">THE PIPELINE IN PLAIN ENGLISH</div>']
    n = 0
    for d in stages + rails:
        n += 1
        suffix = "" if d["kind"] == "stage" else " (runs alongside every step)"
        parts.append(
            '<div class="step"><span class="step-num" style="background:%s">%d</span>'
            '<span><span class="step-what">%s%s</span> '
            '<span class="step-why">&mdash; %s</span></span></div>'
            % (d["color"], n, d["plain"]["t"], suffix, d["plain"]["w"]))
    parts.append('</div>')
    return "".join(parts)


def build():
    merge_narrative()
    total_modules = 0
    total_components = 0
    for d in ATLAS["domains"]:
        for m in d["modules"]:
            total_modules += 1
            for c in m["components"]:
                total_components += 1
                extractor = extract_yaml if c["lang"] == "yaml" else extract_py
                c["code"] = extractor(c["file"], c["symbol"])

    version = "unknown"
    init_path = os.path.join(ROOT, "spl", "__init__.py")
    with open(init_path, "r", encoding="utf-8") as fh:
        m = re.search(r'__version__\s*=\s*"([^"]+)"', fh.read())
        if m:
            version = m.group(1)

    stamp = "v%s &middot; built %s" % (version, datetime.date.today().isoformat())
    counts = "%d domains &middot; %d modules &middot; %d components" % (
        len(ATLAS["domains"]), total_modules, total_components)

    atlas_json = json.dumps(ATLAS, indent=None).replace("</", "<\\/")

    html = (HTML_TEMPLATE
            .replace("__PIPELINE_SVG__", build_pipeline_svg(ATLAS["domains"]))
            .replace("__PLAIN_STEPS__", build_plain_steps(ATLAS["domains"]))
            .replace("__STAMP__", stamp)
            .replace("__COUNTS__", counts)
            .replace("__ATLAS_JSON__", atlas_json))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("Wrote %s (%.1f KB)" % (os.path.relpath(OUT_PATH, ROOT), len(html) / 1024.0))
    return total_components


def self_check(expected_components):
    with open(OUT_PATH, "r", encoding="utf-8") as fh:
        html = fh.read()
    m = re.search(r'<script type="application/json" id="atlas-data">(.*?)</script>', html, re.S)
    assert m, "self-check: embedded JSON block not found"
    data = json.loads(m.group(1).replace("<\\/", "</"))
    found = 0
    for d in data["domains"]:
        assert d.get("plain") and d.get("steps"), "self-check: no narrative for domain %s" % d["id"]
        for mod in d["modules"]:
            for c in mod["components"]:
                assert c.get("code", "").strip(), "self-check: empty code for %s" % c["symbol"]
                assert c.get("walk"), "self-check: no walkthrough for %s" % c["symbol"]
                found += 1
    assert found == expected_components, "self-check: %d components embedded, expected %d" % (found, expected_components)
    for marker in ("function showDomain", "function showModule", "function selectComp",
                   "function showAtlas", "id=\"pipeline\"", "function stepsHtml",
                   "PLAIN ENGLISH", "atlas-steps"):
        assert marker in html, "self-check: missing marker %r" % marker
    print("Self-check passed: %d components, all carrying live code; interactive markers present." % found)


if __name__ == "__main__":
    n = build()
    self_check(n)
