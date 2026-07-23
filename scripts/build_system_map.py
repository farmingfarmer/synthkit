"""Builds docs/system_map.html — the interactive synthkit atlas.

A single self-contained HTML file (no dependencies, works from
disk, offline, on a locked-down machine): a hub of eight colored
domain nodes -> each domain's modules -> each module's key
components with the ACTUAL source code, extracted from the live
files by ast at build time so the map can never drift from the
code. Rebuild after any change:

    python scripts/build_system_map.py

The generator fails loudly if any referenced symbol has moved or
vanished — the map is a smoke check on its own accuracy.
"""
from __future__ import annotations

import ast
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "synthkit"
OUT = ROOT / "docs" / "system_map.html"

# ---------------------------------------------------------------
# The atlas: domain -> modules -> components (symbol extraction)
# Each component: (label, filename, symbol, blurb). symbol is a
# top-level def/class name, or "Class.method".
# ---------------------------------------------------------------

DOMAINS = [
 {"id": "spec",
  "stage": 'English in', "note": 'a paragraph becomes a strict,\nhuman-reviewed contract', "label": "Spec Layer", "color": "#1B5FAA",
  "blurb": "Plain English becomes a strict, reviewable contract. "
           "The compiler is assisted-never-trusted: LLM drafts, "
           "validator teaches, human gates. Live benchmark: 6 "
           "problems -> 2 -> 2 -> 0 across four runs of one "
           "paragraph.",
  "modules": [
   {"file": "compiler.py", "name": "compiler",
    "blurb": "English -> spec with one repair round; every "
             "validator message feeds back as curriculum.",
    "components": [
     ("The table architect prompt", "compiler.py",
      "_TABLE_COMPILER_SYSTEM",
      "The textbook mistral learns from. Grew a chapter every "
      "time a live compile failed: outcomes, days_from, "
      "rule-vs-distribution, sigma semantics, "
      "target_prevalence."),
     ("compile_table_spec", "compiler.py", "compile_table_spec",
      "Draft, validate, feed problems back verbatim, retry "
      "once, then hand the human whatever survived — with the "
      "problems attached."),
    ]},
   {"file": "tablespec.py", "name": "tablespec",
    "blurb": "The table schema: columns, distributions, mess "
             "rates, rules, outcomes. Validation reports EVERY "
             "problem at once, in teaching voice.",
    "components": [
     ("TableSpec.validate", "tablespec.py", "TableSpec.validate",
      "Total validation with teaching messages — each grew from "
      "a real confusion: rule-kinds-in-distributions, bool "
      "derived targets, dollar-amount sigmas, "
      "target_prevalence bounds."),
    ]},
   {"file": "spec.py", "name": "spec (documents)",
    "blurb": "The document-corpus schema: plantable elements, "
             "distractor traps, style axes, difficulty tiers.",
    "components": [
     ("TargetElement", "spec.py", "TargetElement",
      "A plantable fact the outside model SHOULD extract — "
      "density, difficulty, phrasings."),
     ("Distractor", "spec.py", "Distractor",
      "The trap: content that LOOKS extractable and must NOT "
      "be. The discontinued-medication trap caught a real "
      "model at 26%."),
    ]},
  ]},
 {"id": "truth",
  "stage": 'plant the truth', "note": 'clean data + ledgered mess,\ndeterministic to the cell', "label": "Truth Planning", "color": "#6B3FA0",
  "blurb": "Deterministic generation where the blueprint IS the "
           "ground truth. Cell-level seeds; every corruption "
           "ledgered; same spec, same data, forever.",
  "modules": [
   {"file": "tableplan.py", "name": "tableplan",
    "blurb": "Clean truth first, then mess — mutually exclusive "
             "per cell, priority-ordered, every op a signed "
             "ledger entry.",
    "components": [
     ("Cell-seed determinism", "tableplan.py", "_cell_rng",
      "sha256(master:row:column) -> independent RNG per cell. "
      "The column-independence law: editing one column never "
      "reshuffles another."),
     ("Rules: days_from date gaps", "tableplan.py",
      "_apply_rules",
      "date_after with days_from makes discharge = admission + "
      "los. Built because a live compile kept asking for a "
      "coupling the schema could not say."),
     ("Outcomes: logistic and linear", "tableplan.py",
      "plan_table",
      "Planted causality. Logistic stores true probabilities "
      "(the AUROC ceiling); linear stores the noiseless signal "
      "(the exact R^2 ceiling)."),
    ]},
   {"file": "planner.py", "name": "planner (documents)",
    "blurb": "Blueprints decide every document's contents before "
             "any prose exists.",
    "components": [
     ("plan_corpus", "planner.py", "plan_corpus",
      "Elements, distractors, styles, difficulties — all drawn "
      "deterministically per document."),
    ]},
   {"file": "relational.py", "name": "relational",
    "blurb": "Multi-table truth. Seeded foreign keys; join mess "
             "in tiers: random orphans, near-key orphans, and "
             "format drift — the precision trap.",
    "components": [
     ("plan_relational", "relational.py", "plan_relational",
      "Layered determinism: per-table seeds, seeded fk "
      "assignment, orphan/drift injection into DIRTY rows only "
      "— clean truth keeps referential integrity always."),
     ("Orphan styles + drift", "relational.py", "_make_orphan",
      "'near' orphans are one digit-transposition from a real "
      "key; drift mangles VALID references so naive set-"
      "checkers false-flag good data (precision 0.35 in the "
      "demo, 1.00 after normalizing)."),
    ]},
  ]},
 {"id": "render",
  "stage": 'make artifacts', "note": 'csv / corpora under\nintegrity manifests', "label": "Rendering", "color": "#3D5A6C",
  "blurb": "Blueprints become artifacts under integrity "
           "manifests. Documents render via LLM with verify-"
           "and-retry; tables write dirty/clean/ledger; tamper "
           "one byte and loading refuses.",
  "modules": [
   {"file": "renderer.py", "name": "renderer",
    "blurb": "LLM prose, never trusted: every note verified "
             "against its blueprint, retried, or deterministically "
             "fallback-rendered.",
    "components": [
     ("render_corpus", "renderer.py", "render_corpus",
      "The verify-retry-fallback loop; the RenderReport counts "
      "first-try successes honestly."),
    ]},
   {"file": "tableplan.py", "name": "table artifacts",
    "blurb": "dirty.csv + clean.csv + ledger.json under a "
             "sha256 manifest.",
    "components": [
     ("write_table / integrity", "tableplan.py", "write_table",
      "Every artifact hashed into manifest.json; load_table "
      "verifies before trusting."),
    ]},
  ]},
 {"id": "eval",
  "stage": 'score vs ceiling', "note": 'exact scoring against the\nledger and the possible', "label": "Evaluation & Ceilings",
  "color": "#B3541E",
  "blurb": "Scoring against the ledger, never heuristics — and "
           "against the CEILING: the best any solver could "
           "honestly do, known because the truth is planted.",
  "modules": [
   {"file": "tableeval.py", "name": "tableeval",
    "blurb": "Cleaning fix-rates by op, prediction AUROC vs "
             "ceiling, regression R^2 vs exact ceiling.",
    "components": [
     ("evaluate_cleaning", "tableeval.py", "evaluate_cleaning",
      "Same-length contract; fix/overcorrection sliced by op "
      "and column; wrong-detection and dup-flag rates."),
     ("evaluate_prediction", "tableeval.py",
      "evaluate_prediction",
      "AUROC against the ceiling computed from generating "
      "probabilities — the gap is the sentence."),
     ("evaluate_regression", "tableeval.py",
      "evaluate_regression",
      "ceiling_r2 = Var(signal)/Var(y): exact by construction. "
      "The demo's baseline sits 0.005-0.007 under it."),
    ]},
   {"file": "evaluator.py", "name": "evaluator (documents)",
    "blurb": "Recall per element, false positives per "
             "distractor — the trap rate that characterized a "
             "real model.",
    "components": [
     ("evaluate", "evaluator.py", "evaluate",
      "Extractions matched against planted truth; distractor "
      "hits counted as the fp_rate a vendor meeting needs."),
    ]},
   {"file": "mlmetrics.py", "name": "mlmetrics",
    "blurb": "Stdlib metrics plus the uncertainty layer: every "
             "rate knows its k-of-n.",
    "components": [
     ("wilson_interval", "mlmetrics.py", "wilson_interval",
      "Exact at the edges. Built after 2-of-19 (CI 0.03-0.31) "
      "flipped a verdict a naked 0.105 had claimed."),
     ("required_n — the prescription", "mlmetrics.py",
      "required_n",
      "How much MORE data resolves an inconclusive bar. "
      "Executable, because synthkit generates data: the "
      "resolution run at prescribed n=250/tier closed the "
      "study DECISIVE."),
     ("auroc_interval (Hanley-McNeil)", "mlmetrics.py",
      "auroc_interval",
      "Class-split confidence for AUROC lines."),
    ]},
  ]},
 {"id": "campaign",
  "stage": 'verdict out', "note": 'tiered trials -> DECISIVE\nor a prescription', "label": "Campaigns & Verdicts",
  "color": "#A02F44",
  "blurb": "A goal becomes a three-tier ladder with bars. "
           "Verdicts are three-way: DECISIVE when the interval "
           "clears the bar, INCONCLUSIVE with a prescription "
           "when it straddles. Trial history is append-only.",
  "modules": [
   {"file": "campaign.py", "name": "campaign",
    "blurb": "clean / predict / regress / extract ladders; "
             "blinded shifted-seed training; interval-aware "
             "judgment; the trials comparison table.",
    "components": [
     ("The three-way judge", "campaign.py", "_judge",
      "Point pass/fail gates the ladder (compat); every "
      "evidence line carries [ci lo-hi n: DECISIVE] or "
      "INCONCLUSIVE with n~X to resolve."),
     ("Blinded predict tiers", "campaign.py",
      "_run_predict_tier",
      "Train on master_seed+1000, score the tier table: the "
      "solver never sees test labels or the answer key."),
     ("Append-only trials", "campaign.py", "write_campaign",
      "results/trial_NNN_<arm>.json — a live bake-off once "
      "clobbered its control arm; never again."),
    ]},
  ]},
 {"id": "solvers",
  "stage": 'take the test', "note": 'baselines, cleaners, and\nany LLM in the vendor chair', "label": "Solvers & Vendors",
  "color": "#4C7A34",
  "blurb": "synthkit competes on its own data: mess-tolerant "
           "encoding, stdlib baselines, an honest reference "
           "cleaner — and any LLM in the vendor chair, blind-"
           "prompted and reliability-counted.",
  "modules": [
   {"file": "autosolver.py", "name": "autosolver",
    "blurb": "The floor every vendor must beat. Baseline within "
             "0.011 of the 0.678 ceiling on the encounter "
             "cohort.",
    "components": [
     ("FeatureEncoder", "autosolver.py", "FeatureEncoder",
      "Sniffs numeric/date/bool/categorical THROUGH the mess "
      "from train rows; imputes to train means; standardizes; "
      "deterministic."),
     ("autoclean", "autosolver.py", "autoclean",
      "Fixes what heuristics honestly can (space, case, "
      "formats), flags exact dups, never guesses — "
      "overcorrection near zero."),
     ("run_showdown", "autosolver.py", "run_showdown",
      "ceiling / baseline / vendor per tier — the meeting "
      "summary as a function."),
    ]},
   {"file": "llmvendor.py", "name": "llmvendor",
    "blurb": "The chair a real model sat in: 26% trap rate "
             "characterized, intervened to 3.7% [1.6-8.4], "
             "1,662 calls, 0 malformed.",
    "components": [
     ("LLMExtractor", "llmvendor.py", "LLMExtractor",
      "Blind prompts (elements, never distractors), JSON "
      "salvage, majority voting, reliability stats, and the "
      "extra_system intervention seam."),
     ("_find_json_array", "llmvendor.py", "_find_json_array",
      "Salvages arrays from fences and prose; hopeless output "
      "counts as malformed instead of crashing."),
    ]},
  ]},
 {"id": "gates",
  "stage": '', "note": 'validation, semantic lint,\nstatistics — guarding every stage', "label": "Quality Gates", "color": "#B7791F",
  "blurb": "Three layers of gate: validation (lawful?), "
           "semantic lint (meant?), statistics (known?). Every "
           "check here was a human catch first.",
  "modules": [
   {"file": "lint.py", "name": "lint",
    "blurb": "Probe-based: plan real data, inspect realized "
             "properties. Tables L1-L6, corpora D1-D5.",
    "components": [
     ("lint_table", "lint.py", "lint_table",
      "L1 realized prevalence vs declared target (caught a "
      "67%-vs-15-20% intercept pre-render), L2 derived "
      "magnitudes (the 10^150-dollar stay), L3 rule coupling, "
      "L4 fossils, L5 ignored name pools, L6 dropped mess "
      "clauses."),
     ("lint_corpus", "lint.py", "lint_corpus",
      "D1 never-planted elements, D2 unloaded traps, D3 id "
      "collisions, D4 flat styles, D5 trap-language "
      "coverage."),
    ]},
  ]},
 {"id": "iface",
  "stage": '', "note": 'CLI - bench - notebooks - Bedrock:\nthe doors into every stage', "label": "Interfaces & Backends",
  "color": "#0E6E64",
  "blurb": "One library, four doors: CLI, the calibration "
           "bench, self-contained notebooks, and pluggable LLM "
           "backends including AWS Bedrock.",
  "modules": [
   {"file": "gui.py", "name": "gui — the bench",
    "blurb": "Five stations, spec-card fingerprint, async jobs "
             "with budget+cancel, auto-incrementing campaign "
             "dirs, session persistence, the build chip.",
    "components": [
     ("build_info — the chip", "gui.py", "build_info",
      "version - built date - fingerprint; matches `synthkit "
      "version` so 'am I current' is a glance. Caught a real "
      "install defect on first use."),
     ("Async jobs with a budget", "gui.py", "api_job",
      "A wedged ollama once served a 6-minute run for 106 "
      "minutes; jobs now time out readably and can be "
      "cancelled."),
    ]},
   {"file": "cli.py", "name": "cli",
    "blurb": "Everything scriptable: compile, render, lint, "
             "campaigns, trials, showdown, version.",
    "components": [
     ("campaign-run --llm", "cli.py", "cmd_campaign_run",
      "The vendor chair from the terminal: backend, model, "
      "samples, @file interventions, reliability line."),
    ]},
   {"file": "bedrock.py", "name": "bedrock",
    "blurb": "The work door: Converse API, injectable client, "
             "fully smoked without AWS.",
    "components": [
     ("BedrockBackend", "bedrock.py", "BedrockBackend",
      "Same contract as Ollama; boto3 deferred to first real "
      "call; the compiler, renderer, and vendor adapter take "
      "it unchanged."),
    ]},
  ]},
]


def extract_symbol(path: Path, symbol: str) -> str:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines()
    parts = symbol.split(".")

    def find(body, name):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef,
                                 ast.AsyncFunctionDef)) \
                    and node.name == name:
                return node
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) \
                            and t.id == name:
                        return node
        return None

    node = find(tree.body, parts[0])
    if node is not None and len(parts) == 2:
        node = find(node.body, parts[1])
    if node is None:
        raise SystemExit("MAP DRIFT: `{}` not found in {} — "
                         "update build_system_map.py".format(
                             symbol, path.name))
    start = node.lineno - 1
    if getattr(node, "decorator_list", None):
        start = node.decorator_list[0].lineno - 1
    end = node.end_lineno
    snippet = "\n".join(lines[start:end])
    if len(snippet) > 4200:
        snippet = "\n".join(lines[start:start + 90]) + \
            "\n    # ... ({} more lines in {}) ...".format(
                end - start - 90, path.name)
    return snippet


def build_data() -> dict:
    total_components = 0
    for dom in DOMAINS:
        for mod in dom["modules"]:
            for comp in mod["components"]:
                label, fname, symbol, blurb = comp
                code = extract_symbol(PKG / fname, symbol)
                mod.setdefault("built", []).append({
                    "label": label, "symbol": symbol,
                    "file": fname, "blurb": blurb,
                    "code": code})
                total_components += 1
            mod["components"] = mod.pop("built")
    from synthkit.gui import build_info
    info = build_info()
    return {"domains": DOMAINS, "build": info,
            "components": total_components}


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>synthkit — system atlas</title>
<style>
:root{--bench:#EDF0F3;--panel:#FFF;--ink:#17222C;--dim:#5C6B78;
 --rule:#CBD4DC;--enamel:#0E6E64;
 --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
 --sans:'IBM Plex Sans',system-ui,sans-serif}
*{box-sizing:border-box;margin:0}
body{background:var(--bench);color:var(--ink);
 font-family:var(--sans);font-size:14px;line-height:1.5;
 padding:22px 28px}
header{display:flex;justify-content:space-between;
 align-items:baseline;margin-bottom:14px}
.wordmark{font-family:var(--mono);font-weight:600;
 letter-spacing:.14em;font-size:16px}
.wordmark small{display:block;color:var(--dim);font-weight:400;
 letter-spacing:.08em;font-size:10px}
#crumbs{font-family:var(--mono);font-size:12px;color:var(--dim)}
#crumbs a{color:var(--enamel);cursor:pointer;
 text-decoration:none}
#crumbs a:hover{text-decoration:underline}
.stage{background:var(--panel);border:1px solid var(--rule);
 padding:18px;min-height:560px}
svg text{font-family:var(--mono)}
.node{cursor:pointer}
.node:hover .halo{opacity:.25}
.node .halo{opacity:0;transition:opacity .12s}
.blurb{max-width:760px;color:var(--dim);font-size:13px;
 margin:10px 0 0}
.explorer{display:grid;grid-template-columns:290px 1fr;gap:16px}
.chip{border:1px solid var(--rule);border-left:4px solid
 var(--accent,#888);padding:10px 12px;margin-bottom:8px;
 cursor:pointer;background:#FBFCFD}
.chip:hover{border-color:var(--accent,#888)}
.chip.active{background:#fff;border-color:var(--accent,#888);
 box-shadow:0 1px 0 var(--rule)}
.chip b{font-family:var(--mono);font-size:12.5px;display:block}
.chip small{color:var(--dim);font-size:11px}
.codebox{border:1px solid var(--rule)}
.codehead{font-family:var(--mono);font-size:11px;
 background:var(--bench);border-bottom:1px solid var(--rule);
 padding:8px 12px;color:var(--dim)}
.codehead b{color:var(--ink)}
pre.code{font-family:var(--mono);font-size:11.8px;
 line-height:1.55;padding:14px;overflow:auto;max-height:520px;
 background:#FCFDFE;white-space:pre}
.tok-c{color:#4C7A34;font-style:italic}
.tok-s{color:#A02F44}
.tok-k{color:#1B5FAA;font-weight:600}
.tok-d{color:#6B3FA0;font-weight:600}
.compblurb{padding:10px 12px;font-size:12.5px;color:var(--ink);
 background:#F4F7F6;border-bottom:1px solid var(--rule)}
footer{margin-top:12px;font-family:var(--mono);font-size:11px;
 color:var(--dim);display:flex;justify-content:space-between}
@media(prefers-reduced-motion:no-preference){
 .stage>*{animation:in .16s ease-out}
 @keyframes in{from{opacity:.5}to{opacity:1}}}
</style></head><body>
<header>
 <div class="wordmark">SYNTHKIT<small>system atlas &middot;
  click nodes to descend &middot; @@FP@@</small></div>
 <div id="crumbs"></div>
</header>
<div class="stage" id="stage"></div>
<footer><span>@@COUNTS@@</span>
 <span>rebuild: python scripts/build_system_map.py</span></footer>
<script id="atlas" type="application/json">@@DATA@@</script>
<script>
var DATA=JSON.parse(document.getElementById('atlas').textContent);
var stage=document.getElementById('stage');
var crumbs=document.getElementById('crumbs');
function esc(s){return String(s).replace(/&/g,'&amp;')
 .replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function hi(code){
 var s=esc(code);
 s=s.replace(/(&quot;|')((?:\\.|(?!\1)[^\\\n])*)(\1)/g,
  function(m){return '<span class="tok-s">'+m+'</span>';});
 s=s.replace(/(^|\n)([ \t]*#[^\n]*)/g,
  function(m,a,b){return a+'<span class="tok-c">'+b+
   '</span>';});
 s=s.replace(/\b(def|class|return|if|elif|else|for|while|try|except|finally|with|import|from|raise|yield|lambda|not|and|or|in|is|None|True|False|assert|continue|break|pass|global)\b/g,
  '<span class="tok-k">$1</span>');
 s=s.replace(/\b(self|cls)\b/g,
  '<span class="tok-d">$1</span>');
 return s;}
function setCrumbs(parts){
 var out=[];
 for(var i=0;i<parts.length;i++){
  var p=parts[i];
  if(i<parts.length-1){
   out.push('<a data-act="'+p.act+'" data-i="'+
    (p.i===undefined?'':p.i)+'">'+p.t+'</a>');}
  else{out.push('<b>'+p.t+'</b>');}}
 crumbs.innerHTML=out.join(' &nbsp;/&nbsp; ');}
function hub(){
 setCrumbs([{t:'atlas'}]);
 var FLOW=['spec','truth','render','solvers','eval','campaign'];
 var RAILS=['gates','iface'];
 var byId={};
 for(var i=0;i<DATA.domains.length;i++){
  byId[DATA.domains[i].id]=[DATA.domains[i],i];}
 var W=1060,H=600,y=250,x0=118,dx=(W-2*x0)/(FLOW.length-1);
 var s='<svg viewBox="0 0 '+W+' '+H+'" width="100%">';
 s+='<defs><marker id="arr" viewBox="0 0 10 10" refX="9" '+
  'refY="5" markerWidth="7" markerHeight="7" orient="auto">'+
  '<path d="M0 0 L10 5 L0 10 z" fill="#8A97A3"/></marker>'+
  '</defs>';
 s+='<text x="'+(x0-84)+'" y="'+(y-26)+'" font-size="10" '+
  'fill="#5C6B78">an English</text>'+
  '<text x="'+(x0-84)+'" y="'+(y-13)+'" font-size="10" '+
  'fill="#5C6B78">paragraph</text>'+
  '<line x1="'+(x0-56)+'" y1="'+y+'" x2="'+(x0-44)+'" y2="'+y+
  '" stroke="#8A97A3" stroke-width="1.6" '+
  'marker-end="url(#arr)"/>';
 var k;
 for(k=0;k<FLOW.length-1;k++){
  var xa=x0+k*dx+46,xb=x0+(k+1)*dx-52;
  s+='<line x1="'+xa+'" y1="'+y+'" x2="'+xb+'" y2="'+y+
   '" stroke="#8A97A3" stroke-width="1.6" '+
   'marker-end="url(#arr)"/>';}
 var xe=x0+(FLOW.length-1)*dx;
 s+='<line x1="'+(xe+46)+'" y1="'+y+'" x2="'+(xe+62)+'" y2="'+
  y+'" stroke="#8A97A3" stroke-width="1.6" '+
  'marker-end="url(#arr)"/>'+
  '<text x="'+(xe+70)+'" y="'+(y-6)+'" font-size="10" '+
  'fill="#5C6B78">a measured</text>'+
  '<text x="'+(xe+70)+'" y="'+(y+7)+'" font-size="10" '+
  'fill="#5C6B78">verdict</text>';
 for(k=0;k<FLOW.length;k++){
  var pair=byId[FLOW[k]],d=pair[0],di=pair[1];
  var x=x0+k*dx,above=(k%2===0);
  s+='<g class="node" data-act="domain" data-i="'+di+'">'+
   '<circle class="halo" cx="'+x+'" cy="'+y+
   '" r="56" fill="'+d.color+'"/>'+
   '<circle cx="'+x+'" cy="'+y+'" r="44" fill="'+d.color+
   '"/>'+
   '<text x="'+x+'" y="'+(y-1)+'" text-anchor="middle" '+
   'fill="#fff" font-size="10.5" font-weight="600">0'+
   (k+1)+'</text>'+
   '<text x="'+x+'" y="'+(y+13)+'" text-anchor="middle" '+
   'fill="#fff" font-size="9" opacity=".95">'+d.stage+
   '</text></g>';
  var ly=above?y-118:y+82;
  s+='<line x1="'+x+'" y1="'+(above?y-56:y+56)+'" x2="'+x+
   '" y2="'+(above?ly+30:ly-12)+'" stroke="'+d.color+
   '" stroke-width="1.2" opacity=".5"/>';
  s+='<text x="'+x+'" y="'+ly+'" text-anchor="middle" '+
   'font-size="11.5" font-weight="600" fill="#17222C">'+
   d.label+'</text>';
  var noteLines=d.note.split('\n');
  for(var q=0;q<noteLines.length;q++){
   s+='<text x="'+x+'" y="'+(ly+15+q*13)+
    '" text-anchor="middle" font-size="9.5" '+
    'fill="#5C6B78">'+noteLines[q]+'</text>';}}
 for(var r=0;r<RAILS.length;r++){
  var rp=byId[RAILS[r]],rd=rp[0],ri=rp[1];
  var ry=442+r*72;
  s+='<g class="node" data-act="domain" data-i="'+ri+'">'+
   '<rect class="halo" x="'+(x0-58)+'" y="'+(ry-26)+
   '" width="'+(W-2*x0+116)+'" height="52" rx="8" fill="'+
   rd.color+'"/>'+
   '<rect x="'+(x0-50)+'" y="'+(ry-20)+'" width="'+
   (W-2*x0+100)+'" height="40" rx="5" fill="#fff" stroke="'+
   rd.color+'" stroke-width="2"/>'+
   '<text x="'+(x0-34)+'" y="'+(ry+4)+'" font-size="11.5" '+
   'font-weight="600" fill="'+rd.color+'">'+rd.label+
   '</text>'+
   '<text x="'+(W-x0+34)+'" y="'+(ry+4)+'" font-size="9.5" '+
   'text-anchor="end" fill="#5C6B78">'+
   rd.note.replace(/\n/g,' ')+'</text></g>';
  for(k=0;k<FLOW.length;k++){
   var tx=x0+k*dx;
   s+='<line x1="'+tx+'" y1="'+(ry-20)+'" x2="'+tx+'" y2="'+
    (r===0?y+140:ry-52)+'" stroke="'+rd.color+
    '" stroke-width="1" opacity=".18"/>';}}
 s+='</svg>';
 stage.innerHTML=s+'<div class="blurb">The pipeline reads '+
  'left to right: what happens to your paragraph. The two '+
  'rails below guard and expose every stage. Every node is '+
  'clickable: domain &rarr; modules &rarr; components '+
  '&rarr; the actual source.</div>';}
function domainView(i){
 var d=DATA.domains[i];
 setCrumbs([{t:'atlas',act:'hub'},{t:d.label}]);
 var W=980,H=340,cx=180,cy=H/2;
 var s='<svg viewBox="0 0 '+W+' '+H+'" width="100%">';
 var n=d.modules.length;
 for(var j=0;j<n;j++){
  var m=d.modules[j];
  var y=60+j*((H-100)/Math.max(n-1,1)),x=560;
  s+='<path d="M '+(cx+58)+' '+cy+' C 380 '+cy+', 380 '+y+
   ', '+(x-96)+' '+y+'" stroke="'+d.color+
   '" stroke-width="1.4" fill="none" opacity=".4"/>';
  s+='<g class="node" data-act="module" data-i="'+i+
   '" data-j="'+j+'">'+
   '<rect class="halo" x="'+(x-104)+'" y="'+(y-30)+
   '" width="208" height="60" rx="6" fill="'+d.color+'"/>'+
   '<rect x="'+(x-96)+'" y="'+(y-24)+'" width="192" '+
   'height="48" rx="4" fill="#fff" stroke="'+d.color+
   '" stroke-width="2"/>'+
   '<text x="'+x+'" y="'+(y-3)+'" text-anchor="middle" '+
   'fill="#17222C" font-size="12" font-weight="600">'+
   m.name+'</text>'+
   '<text x="'+x+'" y="'+(y+13)+'" text-anchor="middle" '+
   'fill="'+d.color+'" font-size="9.5">'+
   m.components.length+' component'+
   (m.components.length>1?'s':'')+' &middot; '+m.file+
   '</text></g>';}
 s+='<circle cx="'+cx+'" cy="'+cy+'" r="58" fill="'+d.color+
  '"/>'+
  '<text x="'+cx+'" y="'+(cy+4)+'" text-anchor="middle" '+
  'fill="#fff" font-size="11.5" font-weight="600">'+
  d.label+'</text></svg>';
 stage.innerHTML=s+'<div class="blurb">'+d.blurb+'</div>';}
function moduleView(i,j){
 var d=DATA.domains[i],m=d.modules[j];
 setCrumbs([{t:'atlas',act:'hub'},
  {t:d.label,act:'domain',i:i},{t:m.name}]);
 var chips='';
 for(var k=0;k<m.components.length;k++){
  var c=m.components[k];
  chips+='<div class="chip" style="border-left-color:'+
   d.color+'" data-act="show" data-i="'+i+'" data-j="'+j+
   '" data-k="'+k+'" id="chip'+k+'">'+
   '<b>'+esc(c.label)+'</b><small>'+c.file+
   ' &middot; '+esc(c.symbol)+'</small></div>';}
 stage.innerHTML='<div class="explorer"><div>'+
  '<div class="blurb" style="margin:0 0 12px">'+m.blurb+
  '</div>'+chips+'</div><div id="codepane"></div></div>';
 showComp(i,j,0);}
function showComp(i,j,k){
 var d=DATA.domains[i],c=d.modules[j].components[k];
 var chipEls=document.querySelectorAll('.chip');
 for(var q=0;q<chipEls.length;q++){
  chipEls[q].className=(q===k)?'chip active':'chip';}
 document.getElementById('codepane').innerHTML=
  '<div class="codebox"><div class="codehead">synthkit/'+
  c.file+' &nbsp;&middot;&nbsp; <b>'+esc(c.symbol)+
  '</b></div>'+
  '<div class="compblurb">'+esc(c.blurb)+'</div>'+
  '<pre class="code">'+hi(c.code)+'</pre></div>';}
function findAct(el){
 while(el&&el!==document){
  if(el.getAttribute&&el.getAttribute('data-act')){
   return el;}
  el=el.parentNode;}
 return null;}
document.addEventListener('click',function(ev){
 var el=findAct(ev.target);
 if(!el)return;
 var act=el.getAttribute('data-act');
 var i=parseInt(el.getAttribute('data-i')||'0',10);
 var j=parseInt(el.getAttribute('data-j')||'0',10);
 var k=parseInt(el.getAttribute('data-k')||'0',10);
 if(act==='hub')hub();
 else if(act==='domain')domainView(i);
 else if(act==='module')moduleView(i,j);
 else if(act==='show')showComp(i,j,k);});
hub();
</script></script></body></html>
"""


def main():
    data = build_data()
    counts = ("{} domains · {} modules · {} components · "
              "270 checks / 14 suites").format(
        len(data["domains"]),
        sum(len(d["modules"]) for d in data["domains"]),
        data["components"])
    fp = "v{version} · {built} · {fingerprint}".format(
        **data["build"])
    page = (TEMPLATE
            .replace("@@DATA@@", json.dumps(
                {"domains": data["domains"]})
                .replace("</", "<\\/"))
            .replace("@@FP@@", html.escape(fp))
            .replace("@@COUNTS@@", counts))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print("system map -> {} ({} components, {} KB)".format(
        OUT, data["components"], len(page) // 1024))


if __name__ == "__main__":
    main()
