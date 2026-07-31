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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:                       # console-safe on Windows terminals
    from _console import console_safe
    console_safe()
except Exception:
    pass

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
     ("Note columns: text with planted signal", "tablespec.py",
      "_validate_note",
      "Plain English: a table can carry a free-text note field "
      "whose hidden phrases genuinely drive the outcome. This "
      "validator teaches the rules: elements need phrasings, "
      "densities, weights; distractors can carry `excludes` so "
      "denial traps only appear in patients WITHOUT the risk."),
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
     ("Hybrid notes: prose that moves the logit",
      "tableplan.py", "_build_note",
      "The capstone mechanism. Each patient's note is "
      "assembled from seeded phrase draws; which risk phrases "
      "landed feeds the outcome directly — so the achievable "
      "ceiling INCLUDES the text, and any model that cannot "
      "read hits a lower wall by construction. Measured: "
      "ceiling 0.82, note-blind 0.60, note-reading 0.72."),
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
     ("TextMiner: reading the notes, honestly",
      "autosolver.py", "TextMiner",
      "Plain English: from training records ALONE it learns "
      "which words and phrases predict the outcome — no "
      "peeking at the recipe. Negated mentions ('denies X') "
      "are kept as separate evidence from 'X', because "
      "clinical risk is often phrased negatively ('no home "
      "support'). That split alone bought +0.03 AUROC."),
     ("autosolver_hybrid: the doctor's own model",
      "autosolver.py", "autosolver_hybrid",
      "Tabular features PLUS mined note features into one "
      "logistic regression. The free challenger that beat "
      "the text-blind vendor by 0.12 on the capstone cohort."),
     ("Model transparency: real coefficients",
      "autosolver.py", "_publish_fit",
      "Every fit publishes its actual learned weights with "
      "human names — categorical one-hots expanded, mined "
      "phrases labeled ('note phrase \"kg\" mentioned — "
      "raises risk'). Building this caught patient NAMES "
      "wearing risk weights; identifier-like columns are now "
      "excluded from features."),
     ("run_showdown", "autosolver.py", "run_showdown",
      "ceiling / baseline / vendor per tier, with a "
      "SELECTABLE baseline (hybrid as the floor) and a "
      "[bar:] verdict split from the baseline comparison — "
      "the meeting summary as a function."),
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
     ("The linear salvage scanner", "llmvendor.py",
      "_first_balanced_array",
      "Replaced a backtracking regex that spun FOREVER on a "
      "3B model's unclosed-bracket rambles — one bug, three "
      "live incidents (frozen ticker, idle server, Ctrl-C "
      "traceback). String-aware bracket scan: 0.0001s on the "
      "killer input, 17 rambles counted calmly in its first "
      "real outing."),
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
    "blurb": "Five numbered steps a clinician can walk without "
             "an engineer: plain-English banners, required/"
             "optional badges, live red/green readiness that "
             "GATES the buttons, jargon-free labels ('a table "
             "— one row per patient'), a governance strip, "
             "and depth styling (sheened raised buttons, "
             "inset editables, shadowed cards) so the eye "
             "knows what is pressable. Plus: async jobs with "
             "budget+cancel, fingerprint chip, persistence.",
    "components": [
     ("build_info — the chip", "gui.py", "build_info",
      "version - built date - fingerprint; matches `synthkit "
      "version` so 'am I current' is a glance. Caught a real "
      "install defect on first use."),
     ("Async jobs with a budget", "gui.py", "api_job",
      "A wedged ollama once served a 6-minute run for 106 "
      "minutes; jobs now time out readably and can be "
      "cancelled."),
     ("Data transparency: preview + downloads", "gui.py",
      "api_export",
      "The moment data exists: metric cards (records / "
      "corruptions / outcome-rate vs promise), a white "
      "scrollable preview with click-to-read-any-cell (the "
      "notes, full screen), and one-click downloads — messy "
      "CSV/JSON, the clean answer key, the corruption "
      "ledger."),
     ("The plain-English final report", "gui.py",
      "_showdown_report",
      "After the showdown: what happened in plain terms, the "
      "hidden traps WITH example phrases pulled live from "
      "the spec, both methodologies, the trained model's "
      "actual coefficients as weight bars, color-coded "
      "per-tier verdicts, and the WINNER card on top."),
    ]},
   {"file": "openai_compat.py", "name": "openai_compat",
    "blurb": "The open-source door. 'OpenAI-compatible' names "
             "the message FORMAT the local-serving world "
             "standardized on — llama.cpp, LM Studio, vLLM, "
             "llamafile — NOT the company: no ChatGPT, no "
             "account, traffic stays on the machine. This one "
             "backend put a $0 Llama-3B on the demo laptop.",
    "components": [
     ("OpenAICompatBackend", "openai_compat.py",
      "OpenAICompatBackend",
      "One class covering every OpenAI-dialect server; "
      "injectable transport, both response dialects, "
      "llamafile-citing errors. Field-proven: the ThinkPad "
      "rendered its first live clinical prose through it "
      "(7 verified first try / 1 retried / 0 fallbacks)."),
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
 {"id": "phase2", "label": "Learning From Real Data",
  "stage": True, "color": "#7B4B94",
  "blurb": "Phase 2. Everything above INVENTS a population from a "
           "recipe. This stage learns one from an existing extract "
           "— fitting distributions, mess rates and the "
           "relationships between fields — and hands back an "
           "editable recipe. The privacy line is absolute: real "
           "data teaches PARAMETERS, and generation never touches "
           "a record.",
  "modules": [
   {"file": "../scripts/omop_wrangle.py", "name": "wrangler",
    "blurb": "Six clinical tables collapsed into one tidy row per "
             "visit: key normalisation, a temporal drug join, and "
             "an EAV lab panel pivoted wide.",
    "components": [
     ("Tidy one row per visit", "../scripts/omop_wrangle.py",
      "main",
      "Joins person, visits, conditions, procedures, drugs and "
      "measurements. Keys arrive as ints AND as leading-zero text "
      "in the same column, so every key is normalised or rows "
      "silently vanish; orphans are counted and reported rather "
      "than dropped in silence."),
    ]},
   {"file": "../scripts/omop_profile.py", "name": "profiler",
    "blurb": "Measures patterns as parameters, never records. "
             "k-anonymity counts PEOPLE; extremes are withheld; "
             "nothing unconfirmed is imposed.",
    "components": [
     ("Profile: parameters, never records",
      "../scripts/omop_profile.py", "main",
      "Fits each column, measures missingness and contamination, "
      "and reports the joint structure. Every published number is "
      "an aggregate over at least k PATIENTS — the fix after a "
      "single heavy utiliser's thirty visits was found clearing a "
      "row-based threshold while describing one person."),
     ("Contamination-aware fitting",
      "../scripts/omop_profile.py", "profile_numeric",
      "A gaussian column with 4% wild outliers reads as skewed "
      "and gets mis-fitted as lognormal, generating the wrong "
      "shape entirely. The family is chosen first, the body is "
      "trimmed in the space where that family is symmetric, and "
      "the extremes are declared separately as a mess rate."),
    ]},
   {"file": "condnet.py", "name": "condnet — the joint model",
    "blurb": "Learns the JOINT distribution rather than margins "
             "plus monotone links, so U-shapes and interactions "
             "survive. Every conditional table is a dial.",
    "components": [
     ("CondNet: the joint distribution", "condnet.py", "CondNet",
      "Discretises every column, learns parent sets by conditional "
      "mutual information, stores conditional tables and samples "
      "ancestrally. Reproduced a U-shaped relationship whose rank "
      "correlation was -0.019 — invisible to any correlation-based "
      "method."),
     ("Missingness as a pattern", "condnet.py", "Binning",
      "Absence gets its own bin, so 'this lab is ordered mainly "
      "for the sick' is captured as structure. Point masses — "
      "zeros, detection limits, clamped floors — are stored as "
      "atoms and emitted verbatim, because interval sampling can "
      "never produce an exact repeated value."),
     ("The amplify dial", "condnet.py", "CondNet.amplify",
      "A geometric tilt of each conditional toward or away from "
      "its marginal. Factor 0 deletes a discovered relationship, "
      "1 leaves it, 2 sharpens it — the strength changes while "
      "the SHAPE is preserved, which is what makes a found "
      "pattern testable at difficulties the source never had."),
     ("Person-aware learning", "condnet.py", "CondNet.learn",
      "The privacy and inference unit is the patient. k counts "
      "people, significance uses the person count, and the "
      "identity column is excluded — without which the model "
      "conditioned on patient identity itself, which is "
      "memorisation wearing a graph."),
    ]},
   {"file": "../scripts/fidelity_report.py", "name": "scorecard",
    "blurb": "Does the synthetic data carry the same patterns, and "
             "does it avoid reproducing the people? Both halves "
             "validated against deliberate failures.",
    "components": [
     ("Fidelity and privacy scored together",
      "../scripts/fidelity_report.py", "main",
      "Marginals, missingness, correlations, the SHAPE of each "
      "dependence and the interactions — with tolerances derived "
      "from the sampling distribution rather than fixed "
      "thresholds. Relationships that are only noise in the source "
      "are skipped, because reproducing randomness is not "
      "fidelity."),
     ("The nearest-neighbour privacy test",
      "../scripts/fidelity_report.py", "nn_distances",
      "Synthetic records must sit no closer to real records than "
      "real records sit to each other. Catches the interpolation "
      "failure mode — near-copies with zero exact matches — which "
      "an exact-match test alone would pass."),
    ]},
   {"file": "../scripts/power_sweep.py", "name": "power curve",
    "blurb": "How many patients does each kind of structure need? "
             "Measured against planted truth, because with real "
             "data you never know the answer.",
    "components": [
     ("Patients needed, per structure type",
      "../scripts/power_sweep.py", "main",
      "Plants a monotone relationship, a U-shape, a threshold and "
      "two interactions, then measures recovery across cohort "
      "sizes. Turns 'we need more data' into a specification: "
      "nonlinear structure at ~800 patients, interactions at "
      "~3,200 — and zero false positives at every size."),
    ]},
  ]},
 {"id": "narrative", "label": "Notes & Extraction",
  "stage": True, "color": "#B45309",
  "blurb": "Clinical facts rendered into messy prose, with a "
           "ledger of what each note actually asserts — so an "
           "extractor can be graded by the KIND of mess that beat "
           "it. Nothing here learns language from real notes: "
           "facts are authored and rendered, so no phrasing can "
           "trace to a patient.",
  "modules": [
   {"file": "transcribe.py", "name": "transcribe",
    "blurb": "Structured facts become prose, corrupted by a "
             "taxonomy rather than by undifferentiated noise.",
    "components": [
     ("Placement: where each fact is allowed to live",
      "transcribe.py", "transcribe_row",
      "A benchmark whose facts appear identically in a column and "
      "a note measures nothing — a model can ignore the text and "
      "score perfectly. So each fact declares itself structured-"
      "only, text-only, agreeing, or DISAGREEING, which is the "
      "hard part of chart abstraction and the part worth "
      "grading."),
     ("The corruption taxonomy", "transcribe.py", "CORRUPTIONS",
      "Nine kinds of mess, each testing a distinct competence: "
      "abbreviation, simple and compound negation, hedging, "
      "historical mentions, copy-forward staleness, transposed "
      "digits, missing units — and the scope trap, where a "
      "negation sits in the same clause but does not reach the "
      "finding."),
     ("Grading by corruption",
      "transcribe.py", "score_extraction",
      "One overall recall hides everything that matters. Sliced by "
      "corruption it reads 'recall 0.91, but 0.00 on scope traps', "
      "which names a missing competence instead of issuing a "
      "mark."),
    ]},
   {"file": "noteextract.py", "name": "the note vendor chair",
    "blurb": "A language model held to a narrow contract — "
             "present, current, certain — and graded on it.",
    "components": [
     ("NoteExtractor: the tested seat",
      "noteextract.py", "NoteExtractor",
      "Asks only what the NOTE asserts, blind to which "
      "corruptions were planted. Unusable replies, empty arrays "
      "and invented fact keys are each counted separately: a "
      "model that cannot hold the answer format is a procurement "
      "finding, not a harness problem."),
     ("A reader with no model attached",
      "noteextract.py", "ScriptedBackend",
      "Deterministic stand-ins for a keyword matcher, a "
      "clause-scoped reader and a format-breaking model, so the "
      "grading path can be demonstrated with no network — and so "
      "each failure mode is provably detected."),
    ]},
  ]},
]


# =====================================================================
# NARRATIVE — the plain-English layer, authored separately and merged
# at build time. If a domain or component lacks its narrative, the
# build REFUSES ("NARRATIVE DRIFT") — same accuracy contract as code
# drift. Steps are [bold action, dim why] pairs; language rules: no
# jargon, no symbol names, concrete over abstract, real incidents.
# =====================================================================
NARRATIVE = {
 "pipeline": {
  "spec": "so the request is exact, reviewable, and approved "
          "by a person before anything exists",
  "truth": "because grading is only honest when the answer "
           "key exists before the data does",
  "render": "so a model faces realistic mess while the exact "
            "truth sits safely beside it",
  "eval": "so every score is measured against planted truth "
          "AND the best score possible",
  "campaign": "so one lucky result can never pass — verdicts "
              "come from three difficulties with statistics "
              "attached",
  "solvers": "so every vendor claim is measured against a "
             "free, transparent challenger in the identical "
             "seat",
  "gates": "runs alongside every step — lawful recipe, kept "
           "promises, honest uncertainty",
  "phase2": "so a benchmark can be shaped like the hospital's "
            "own data instead of an engineer's guess — while "
            "generation still consumes parameters, never records",
  "narrative": "because the clinical value that vendors compete "
               "over is locked in free text, and an exam that "
               "only has columns cannot test for it",
  "iface": "runs alongside every step — the bench, the "
           "command line, and AI connections on your terms",
 },
 "domains": {
  "spec": {
   "plain": "Turn a plain-English request into an exact, "
            "reviewable recipe for the data.",
   "steps": [
    ["Read the paragraph describing the data someone needs",
     "a doctor writes what records should look like — no code, "
     "no forms"],
    ["Draft a formal recipe from it",
     "an AI writes the first draft, but drafts can contain "
     "mistakes — one small model invented three impossible "
     "settings in a single live run"],
    ["Check the draft and explain every problem in plain terms",
     "the checker lists everything wrong at once, in teaching "
     "voice, so the fix is obvious"],
    ["Wait for a person to approve",
     "nothing is ever created from an unapproved recipe — the "
     "pause is the safety feature, not a delay"],
   ]},
  "truth": {
   "plain": "Build the answer key first: decide every value, "
            "every flaw, and every hidden trap before any data "
            "exists.",
   "steps": [
    ["Decide every clean value from the recipe",
     "each cell gets its own dice roll, seeded, so the same "
     "recipe gives the same data forever — on any machine"],
    ["Plant the outcome's true causes",
     "who gets readmitted is decided by known factors with "
     "known strengths, so the best possible score is known "
     "exactly"],
    ["Write the clinical notes with risk phrases hidden inside",
     "real risk lives in free text — 'ran out of furosemide "
     "two weeks ago' — and here those phrases genuinely drive "
     "the outcome"],
    ["Lace in traps and deliberate flaws, keeping a ledger",
     "denials appear only in patients WITHOUT the risk, old "
     "findings look current, values go missing or mistyped — "
     "and every corruption is written down"],
   ]},
  "render": {
   "plain": "Turn the plan into real files: the messy dataset, "
            "the clean answer key, and the corruption ledger.",
   "steps": [
    ["Write the messy dataset a model would actually face",
     "with the typos, gaps, and duplicates a real hospital "
     "extract has"],
    ["Write the clean answer key beside it",
     "same records, every flaw repaired, every truth known — "
     "the exam's grading sheet"],
    ["Optionally let an AI phrase the free-text notes",
     "for more natural prose — but every AI-written note is "
     "checked against the answer key and corrected if it "
     "drifts; the demo laptop's small model needed one "
     "correction in eight notes"],
   ]},
  "eval": {
   "plain": "Score any model's answers against the planted "
            "truth — and against the best score possible.",
   "steps": [
    ["Grade answers cell by cell against the answer key",
     "no opinions: the truth was planted, so grading is exact"],
    ["Compare every score to the known ceiling",
     "a 0.72 means something different when the maximum "
     "possible is 0.82 — the ceiling keeps everyone honest"],
    ["Attach how-sure-are-we ranges to every number",
     "small samples get wide ranges; the instrument says "
     "'not enough data to be sure' instead of pretending"],
   ]},
  "campaign": {
   "plain": "Run the same exam at three difficulty levels and "
            "return a verdict, not just a number.",
   "steps": [
    ["Build one exam at three difficulties from the recipe",
     "gentler, exactly as specified, and adversarial — so a "
     "model can't pass by luck on one easy draw"],
    ["Train contestants on a separate practice population",
     "answers are hidden during the test; nobody studies the "
     "actual exam"],
    ["Judge each pass mark with statistical honesty",
     "verdicts are PASS, FAIL, or INCONCLUSIVE — and "
     "inconclusive comes with a prescription: how much more "
     "data would settle it"],
    ["File every result permanently",
     "append-only records, so every claim in a meeting traces "
     "to a file"],
   ]},
  "solvers": {
   "plain": "Field our own free, transparent model — and give "
            "any vendor's model the identical seat.",
   "steps": [
    ["Standardize the messy table automatically",
     "dates in five formats become one; dollar signs and "
     "typos are cleaned; names and record numbers are "
     "excluded from evidence — they once snuck in wearing "
     "risk weights"],
    ["Learn which note phrases predict the outcome",
     "from training records alone — and 'denies missing "
     "doses' is kept separate from 'missing doses', because "
     "in medicine the denial often means the opposite"],
    ["Train a simple, explainable risk model",
     "classic logistic regression: every learned weight can "
     "be shown and named — no black box on our side"],
    ["Seat any vendor's model in the same chair",
     "same training data, same hidden answers — one AI "
     "vendor was measured falling for 26% of the traps, "
     "then improved to under 4% and re-verified"],
    ["Count when a tested AI fails to follow instructions",
     "a small model gave unusable answers on 17 of 24 calls "
     "— that unreliability is itself a finding, counted "
     "calmly instead of crashing"],
   ]},
  "gates": {
   "plain": "Guard rails that run alongside every step: is the "
            "recipe lawful, does the data keep its promises, "
            "and are we sure?",
   "steps": [
    ["Check the recipe is complete and lawful",
     "before anything is created"],
    ["Generate a sample and measure it against the promises",
     "'readmission was promised at 5-12% and lands at 9.3%' "
     "— the data means what the paragraph said"],
    ["Refuse to overclaim",
     "when the numbers can't support a verdict, the "
     "instrument says so and prescribes the sample size that "
     "would"],
   ]},
  "phase2": {
   "plain": "Learn what an existing extract looks like, and hand "
            "back a recipe someone can edit.",
   "steps": [
    ["Collapse the hospital's tables into one row per visit",
     "keys arrive in mixed formats, facts live in six places, "
     "and labs arrive one row per test — all of which has to be "
     "reconciled before anything can be measured"],
    ["Measure the patterns as numbers, never as records",
     "distributions, how often values go missing, how often they "
     "are implausible, and how the fields move together — every "
     "figure covering at least ten patients"],
    ["Learn the relationships that are actually there",
     "including the ones a correlation cannot see: a lab that is "
     "dangerous at BOTH extremes, or a drug that only matters in "
     "the elderly"],
    ["Refuse to learn the ones that are not",
     "with hundreds of possible pairings, some always look "
     "convincing by chance; those are reported and never imposed"],
    ["Hand back a recipe where every number is a dial",
     "so a person can keep the shape of the real data and make "
     "any pattern in it stronger, weaker, or absent"],
    ["Say how many patients each kind of pattern needs",
     "measured against data where the answer is known — which is "
     "the only honest way to say whether a dataset is big "
     "enough"],
   ]},
  "narrative": {
   "plain": "Write the facts into a messy clinical note, and keep "
            "a record of what that note really says.",
   "steps": [
    ["Decide where each fact is allowed to appear",
     "some only in the columns, some only in the note, and some "
     "in both while DISAGREEING — which is the situation that "
     "makes chart review hard"],
    ["Write it the way clinicians actually write",
     "abbreviations, denials, hedges, findings copied forward "
     "from last month, transposed numbers, missing units"],
    ["Record what each sentence truly asserts",
     "so a denial counts as absence, a hedge counts as "
     "uncertainty, and a carried-forward value counts as stale — "
     "the answer key for the text"],
    ["Grade a reader by the KIND of mess that beat it",
     "one overall score would average everything into a number "
     "that names nothing; per-category it reads 'handles "
     "negation, cannot reason about scope'"],
    ["Count a model that cannot answer at all, separately",
     "failing to hold the answer format is a different problem "
     "from being wrong, and buyers need to know which one they "
     "are looking at"],
   ]},
  "iface": {
   "plain": "The doors in: a five-step bench a clinician can "
            "walk alone, a command line for engineers, and AI "
            "connections that stay on your terms.",
   "steps": [
    ["Guide anyone through five numbered steps",
     "plain-English banners, red/green readiness on every "
     "required field, and buttons that refuse politely until "
     "the fields are ready"],
    ["Show the data the moment it exists",
     "metric cards, a scrollable preview where any cell — "
     "especially a full clinical note — opens to read, and "
     "one-click downloads of the data, the answer key, and "
     "the ledger"],
    ["End with a report a stakeholder can repeat",
     "what was tested, the traps with real example phrases, "
     "what our model actually learned (its real "
     "coefficients), and a named winner"],
    ["Connect to AI on your terms",
     "a free local model on the laptop itself, the "
     "hospital's governed cloud, or none at all — and "
     "'OpenAI-compatible' names a message format, not the "
     "company: nothing goes to ChatGPT"],
   ]},
 },
 "walkthroughs": {
  "compiler.py::_TABLE_COMPILER_SYSTEM": [
   ["Hold the textbook the drafting AI studies",
    "every rule it must follow when turning a paragraph into "
    "a recipe"],
   ["Grow a chapter after every real mistake",
    "each time a live draft went wrong, the lesson was added "
    "— the textbook is a diary of failures that can't repeat"],
  ],
  "compiler.py::compile_table_spec": [
   ["Ask the AI for a draft recipe",
    "from the plain-English paragraph"],
   ["Have the checker mark every problem",
    "and hand the marked-up list straight back to the AI for "
    "one repair attempt"],
   ["Give a person whatever survived, problems attached",
    "the human gate: nothing proceeds unapproved — in one "
    "live demo the gate caught three invented settings"],
  ],
  "tablespec.py::TableSpec.validate": [
   ["Read the entire recipe and list every problem at once",
    "not one error at a time — the whole homework, marked"],
   ["Explain each problem in teaching voice",
    "each message grew from a real person's real confusion"],
  ],
  "tablespec.py::_validate_note": [
   ["Check the free-text note plan is complete",
    "the hidden phrases need wording variants, how often "
    "they appear, and how strongly they matter"],
   ["Check the traps are wired correctly",
    "a denial like 'denies missing doses' must be set to "
    "appear only in patients who truly didn't"],
  ],
  "spec.py::TargetElement": [
   ["Describe one fact a tested model SHOULD find",
    "how often it appears, how hard it is, and the different "
    "ways it can be phrased"],
  ],
  "spec.py::Distractor": [
   ["Describe one trap a tested model must NOT fall for",
    "text that looks extractable but is wrong to extract — "
    "one real AI fell for a trap like this 26% of the time"],
  ],
  "tableplan.py::_cell_rng": [
   ["Give every single cell its own private dice",
    "seeded by recipe, row, and column"],
   ["Guarantee editing one column never reshuffles another",
    "so a small recipe change doesn't silently change "
    "everything else"],
  ],
  "tableplan.py::_apply_rules": [
   ["Enforce relationships between fields",
    "discharge dates follow admission dates by the length of "
    "stay; charges derive from days at a daily rate"],
  ],
  "tableplan.py::plan_table": [
   ["Build every clean record first",
    "the answer key exists before any mess does"],
   ["Decide each patient's outcome from planted causes",
    "and remember the exact probabilities — that is how the "
    "best possible score is known"],
   ["Then corrupt deliberately, writing every act down",
    "missing values, typos, duplicates — each one a signed "
    "ledger entry"],
  ],
  "tableplan.py::_build_note": [
   ["Assemble each patient's note from seeded phrase draws",
    "which risk phrases landed is decided here, patient by "
    "patient"],
   ["Feed those phrases into the outcome itself",
    "so reading the notes is genuinely worth points: "
    "measured, a note-blind model tops out at 0.60 where "
    "0.82 is achievable"],
   ["Keep denial traps out of truly at-risk patients",
    "'denies missing any doses' appears only where it's "
    "true — the trap that punishes keyword matching"],
  ],
  "planner.py::plan_corpus": [
   ["Decide every document's contents before any prose",
    "which facts, which traps, which writing style — all "
    "drawn deterministically per document"],
  ],
  "relational.py::plan_relational": [
   ["Plan several linked tables that agree with each other",
    "patients, visits, bills — sharing consistent IDs"],
  ],
  "relational.py::_make_orphan": [
   ["Deliberately break some links between tables",
    "the way real exports do — including near-miss IDs that "
    "are one character off"],
  ],
  "renderer.py::render_corpus": [
   ["Write each document, then verify it against its plan",
    "AI prose is checked fact by fact"],
   ["Retry or fall back when the writer drifts",
    "a wrong note is rewritten; a hopeless one is replaced "
    "deterministically — the demo cannot be derailed by a "
    "bad writing day"],
  ],
  "tableplan.py::write_table": [
   ["Write the four artifacts of a run",
    "messy data, clean answer key, corruption ledger, and a "
    "manifest that fingerprints them all"],
  ],
  "tableeval.py::evaluate_cleaning": [
   ["Grade a cleaning attempt cell by cell",
    "fixed, missed, or overcorrected — sliced by corruption "
    "type"],
  ],
  "tableeval.py::evaluate_prediction": [
   ["Score predictions on ranking quality",
    "did the model put the truly at-risk patients ahead of "
    "the rest — and by how much versus the ceiling"],
  ],
  "tableeval.py::evaluate_regression": [
   ["Score numeric predictions against planted signal",
    "with the exact best-possible accuracy known in advance"],
  ],
  "evaluator.py::evaluate": [
   ["Grade text extraction against the planted facts",
    "found the real ones? fell for the traps? sliced by "
    "difficulty, style, and phrasing"],
  ],
  "mlmetrics.py::wilson_interval": [
   ["Attach an honest how-sure range to every rate",
    "3 misses out of 17 doesn't mean 17.6% exactly — it "
    "means somewhere between 6% and 41%, and the instrument "
    "says so"],
  ],
  "mlmetrics.py::required_n": [
   ["Prescribe the sample size that would settle it",
    "when a verdict is inconclusive, the answer isn't a "
    "shrug — it's 'test this many more'"],
  ],
  "mlmetrics.py::auroc_interval": [
   ["Put a how-sure range on the ranking score itself",
    "so 0.72 versus 0.60 can be called a real gap, not "
    "noise"],
  ],
  "campaign.py::_judge": [
   ["Turn a measured number into a verdict",
    "PASS, FAIL, or INCONCLUSIVE against the declared pass "
    "mark, using the how-sure range — never the bare number"],
  ],
  "campaign.py::_run_predict_tier": [
   ["Run one difficulty tier end to end",
    "generate that tier's population, train the contestant "
    "on a separate practice draw, score blind, judge"],
  ],
  "campaign.py::write_campaign": [
   ["File the exam and every result permanently",
    "append-only: trials accumulate, nothing is overwritten "
    "— the paper trail a meeting can stand on"],
  ],
  "autosolver.py::FeatureEncoder": [
   ["Recognize what each messy column really is",
    "numbers, dates, yes/nos, categories — through the "
    "typos and formats"],
   ["Refuse to use names and record numbers as evidence",
    "mostly-unique identifiers are memorization bait; one "
    "build caught 'patient_name = jules kim' wearing a risk "
    "weight"],
  ],
  "autosolver.py::autoclean": [
   ["Fix only what can be fixed honestly",
    "spacing, casing, formats, exact duplicates — and never "
    "guess at a missing value"],
  ],
  "autosolver.py::TextMiner": [
   ["Read every training note and find predictive phrases",
    "using only the training answers — never the recipe"],
   ["Keep denials separate from mentions",
    "'denies missing doses' and 'missing doses' are "
    "different evidence; in medicine the denial often points "
    "the other way"],
  ],
  "autosolver.py::autosolver_hybrid": [
   ["Combine the table's numbers with the notes' phrases",
    "into one simple risk model"],
   ["Beat the text-blind vendor with it",
    "by 0.12 on the capstone cohort — the value of reading, "
    "measured"],
  ],
  "autosolver.py::_publish_fit": [
   ["Publish the model's actual learned weights, named",
    "'note phrase kg mentioned — raises risk' beside "
    "ejection fraction — the answer to 'but what did it "
    "actually learn?'"],
  ],
  "autosolver.py::run_showdown": [
   ["Give both models the identical exam",
    "same data, same hidden answers, three difficulties"],
   ["Report ceiling, our model, and the vendor per tier",
    "with the pass-mark verdict kept separate from who beat "
    "whom — one line per tier, meeting-ready"],
  ],
  "llmvendor.py::LLMExtractor": [
   ["Put an AI model in the tested seat, blind",
    "it's told what to find, never what the traps are"],
   ["Count its unreliability honestly",
    "unusable answers are tallied, majority voting is "
    "available, and an instruction tweak can be A/B tested"],
  ],
  "llmvendor.py::_find_json_array": [
   ["Salvage structured answers from messy AI replies",
    "answers arrive wrapped in prose and formatting; "
    "hopeless ones count as failures instead of crashing"],
  ],
  "llmvendor.py::_first_balanced_array": [
   ["Scan long rambling replies in one safe pass",
    "the previous approach could spin forever on one bad "
    "reply — it froze a live run three different ways before "
    "being replaced; the fix handled 17 rambles in "
    "microseconds"],
  ],
  "lint.py::lint_table": [
   ["Generate a sample and measure it against the promises",
    "the realized readmission rate versus the promised "
    "range — this catch is where 'the data means what you "
    "said' comes from"],
  ],
  "lint.py::lint_corpus": [
   ["Do the same promise-check for document collections",
    "are the facts, traps, and styles landing at their "
    "declared rates"],
  ],
  "gui.py::build_info": [
   ["Stamp the page with exactly which build is running",
    "one glance against the command line answers 'am I "
    "current' — it caught a real stale install on day one"],
  ],
  "gui.py::api_job": [
   ["Run long work in the background with a budget",
    "a stuck AI once served a 6-minute job for 106 minutes; "
    "now jobs tick visibly, time out readably, and can be "
    "cancelled"],
  ],
  "gui.py::api_export": [
   ["Hand over the run's files on click",
    "the messy data, the clean answer key, and the "
    "corruption ledger — CSV or JSON, straight to the "
    "browser"],
  ],
  "gui.py::_showdown_report": [
   ["Assemble the plain-English final report",
    "dataset facts, trap examples quoted from the actual "
    "recipe, both methodologies, the model's real learned "
    "weights"],
   ["Name a winner and show why",
    "color-coded per-tier cards, winner on top — the slide "
    "that writes itself"],
  ],
  "openai_compat.py::OpenAICompatBackend": [
   ["Speak the standard local-AI message format",
    "one connector covers llama.cpp, LM Studio, vLLM and "
    "more — the format is named after OpenAI but nothing is "
    "sent to them; it works with the network unplugged"],
   ["Power the $0 demo",
    "a 3-billion-parameter model on the demo laptop wrote "
    "verified clinical prose through this door"],
  ],
  "cli.py::cmd_campaign_run": [
   ["Run the whole exam from a terminal",
    "everything the bench does, scriptable — including the "
    "AI vendor chair with reliability counting"],
  ],
  "../scripts/omop_wrangle.py::main": [
   ["Join six clinical tables into one row per visit",
    "demographics, visits, conditions, procedures, medications "
    "and labs all have to meet in one place before anything can "
    "be measured"],
   ["Repair the join keys before trusting them",
    "the same identifier arrives as a number in one table and as "
    "text with a leading zero in another; without normalising "
    "them those visits silently vanish"],
   ["Count what could not be matched, out loud",
    "orphaned records are reported rather than quietly dropped, "
    "because a silent loss looks exactly like clean data"],
  ],
  "../scripts/omop_profile.py::main": [
   ["Measure every column as numbers rather than copying it",
    "what shape it has, how often it is missing, how often it is "
    "implausible"],
   ["Require at least ten PATIENTS behind every figure",
    "ten visits can be one person, and a rule counting visits "
    "would have published one heavy user's ward as a category"],
   ["Report how the fields move together, and how sure we are",
    "with hundreds of pairs tested, some always look convincing "
    "by chance; those are shown but never imposed"],
  ],
  "../scripts/omop_profile.py::profile_numeric": [
   ["Decide what shape a column really has",
    "before trimming anything, because a long tail can be the "
    "shape itself rather than contamination"],
   ["Separate genuine outliers from the shape",
    "four percent of wild values makes an ordinary column look "
    "skewed and get described wrongly; the extremes are declared "
    "as their own rate instead"],
  ],
  "condnet.py::CondNet": [
   ["Turn every column into bands",
    "so the relationship between any two of them can be stored as "
    "a simple table of what follows what"],
   ["Find which fields actually explain which",
    "using a measure that detects dependence of ANY shape, not "
    "just the straight-line kind"],
   ["Store the answer as tables and draw new patients from them",
    "a table can describe any pattern at all — including one that "
    "reverses direction, which a correlation reports as nothing"],
  ],
  "condnet.py::Binning": [
   ["Give 'missing' its own band",
    "because whether a test was ordered is itself informative — a "
    "lab absent for the healthy and present for the sick is a "
    "pattern worth reproducing"],
   ["Keep exact repeated values exactly",
    "zeros, detection limits and default readings pile up on one "
    "number, and drawing from a range would never reproduce that"],
   ["Never publish anyone's true minimum or maximum",
    "the extremes belong to individuals; the bands stop short of "
    "them and the tails are extrapolated instead"],
  ],
  "condnet.py::CondNet.amplify": [
   ["Make any discovered pattern stronger, weaker, or absent",
    "turn it off entirely and see whether a vendor still claims "
    "to find it; double it and see whether they catch it"],
   ["Change the strength without changing the shape",
    "an amplified U-curve stays a U-curve with deeper troughs, so "
    "the test gets harder in the way you intended"],
  ],
  "condnet.py::CondNet.learn": [
   ["Treat the patient as the unit, not the visit",
    "repeated visits from one person are neither ten people's "
    "worth of protection nor ten people's worth of evidence"],
   ["Exclude patient identity from the model entirely",
    "without this the model learned to condition on WHO the "
    "patient was, which is memorisation with extra steps"],
   ["Ask the cheaper question where the data allows it",
    "'when this patient's own reading drifts, does their risk "
    "change' is answered by their visits; 'do these patients "
    "differ from those' is answered only by the patient count"],
  ],
  "../scripts/fidelity_report.py::main": [
   ["Compare every column's shape, source against synthetic",
    "and set the bar by how much two samples of the same size "
    "would differ by chance, rather than by a made-up threshold"],
   ["Check that the RELATIONSHIPS survived, not just the columns",
    "data can match on every field individually and have lost "
    "every connection between them"],
   ["Skip anything the source itself cannot confirm",
    "asking synthetic data to reproduce noise is a coin flip "
    "counted as a failure"],
  ],
  "../scripts/fidelity_report.py::nn_distances": [
   ["Measure how close each invented record sits to a real one",
    "and compare that against how close real records already sit "
    "to each other"],
   ["Fail when the invented records hug real people",
    "which catches methods that blend real patients together — "
    "they produce no exact copies and still leak"],
  ],
  "../scripts/power_sweep.py::main": [
   ["Plant relationships of known kinds and known strengths",
    "a straight one, one that reverses, one with a cut-point, and "
    "two that only appear in combination"],
   ["Grow the cohort until each is reliably recovered",
    "which converts 'we need more data' into a number per kind of "
    "pattern"],
   ["Count how often pure noise is mistaken for a finding",
    "an instrument that finds structure everywhere is worthless; "
    "this one found none at any size"],
  ],
  "transcribe.py::transcribe_row": [
   ["Decide where each fact is allowed to appear",
    "if every fact sits in both the column and the note, a model "
    "can ignore the writing entirely and still score perfectly"],
   ["Let the note and the record disagree on purpose",
    "reconciling a form that says one thing against a note that "
    "says another is the actual work of chart review"],
   ["Write down what the sentence really claims",
    "which is the answer key: a denial means absent, a hedge "
    "means unconfirmed, a copied-forward number means stale"],
  ],
  "transcribe.py::CORRUPTIONS": [
   ["Name every kind of mess separately",
    "shorthand, denials, denials covering two findings at once, "
    "hedging, old findings, copied-forward numbers, transposed "
    "digits, missing units"],
   ["Include the trap that defeats the obvious fix",
    "'no improvement in heart failure' contains a denial that "
    "does not reach the finding — any reader looking for a "
    "negative word nearby gets it exactly backwards"],
  ],
  "transcribe.py::score_extraction": [
   ["Grade every mention against what the note really said",
    "not against the structured record, so reading is what is "
    "being measured"],
   ["Report the result by the kind of mess involved",
    "'recall 0.91 overall, 0.00 on scope traps' names a missing "
    "skill; a single average names nothing"],
  ],
  "noteextract.py::NoteExtractor": [
   ["Ask the model three questions per fact",
    "is it true of this patient, is it current, and is it stated "
    "plainly — which is where clinical reading actually goes "
    "wrong"],
   ["Never tell it what the traps are",
    "the questions define the answer format; naming the planted "
    "mess would make the exam meaningless"],
   ["Count replies that cannot be used at all",
    "a model that will not hold the answer format is a different "
    "problem from a model that is wrong, and buyers need to see "
    "both"],
  ],
  "noteextract.py::ScriptedBackend": [
   ["Stand in for a model with no model attached",
    "a keyword matcher, a careful reader and one that answers in "
    "prose — so the whole grading path runs offline"],
   ["Prove each failure is actually detected",
    "a scorecard that cannot fail is worthless, so every failure "
    "mode has a stand-in that triggers it deliberately"],
  ],
  "bedrock.py::BedrockBackend": [
   ["Connect to the hospital's governed AWS models",
    "open-weight and Claude models inside existing "
    "governance — the sanctioned route when local isn't "
    "enough"],
  ],
 },
}


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
    # ---- merge the narrative layer; refuse loudly on drift --
    seen_keys = set()
    for dom in DOMAINS:
        narr = NARRATIVE["domains"].get(dom["id"])
        if not narr or not narr.get("plain") \
                or not narr.get("steps"):
            sys.exit("NARRATIVE DRIFT: no plain-English entry "
                     "for domain `{}` — update NARRATIVE"
                     .format(dom["id"]))
        dom["plain"] = narr["plain"]
        dom["steps"] = narr["steps"]
        for mod in dom["modules"]:
            for comp in mod["components"]:
                key = "{}::{}".format(comp["file"],
                                      comp["symbol"])
                walk = NARRATIVE["walkthroughs"].get(key)
                if not walk:
                    sys.exit("NARRATIVE DRIFT: no plain-"
                             "English walkthrough for `{}` — "
                             "update NARRATIVE".format(key))
                comp["walk"] = walk
                seen_keys.add(key)
    stale = set(NARRATIVE["walkthroughs"]) - seen_keys
    if stale:
        sys.exit("NARRATIVE DRIFT: walkthrough(s) for "
                 "unknown component(s): {}".format(
                     ", ".join(sorted(stale))))
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
/* ===== NARRATIVE LAYER ===== */
.narr{margin-top:18px;max-width:860px}
.narrhead{font-family:var(--mono);font-size:11.5px;
 letter-spacing:.12em;color:#5C6B78;font-weight:700;
 margin-bottom:8px}
.pstep{display:flex;gap:11px;align-items:flex-start;
 margin:8px 0;font-size:14px;line-height:1.5}
.nchip{flex:none;min-width:25px;height:25px;border-radius:7px;
 color:#fff;font-weight:800;display:flex;align-items:center;
 justify-content:center;font-size:13px;margin-top:1px;
 box-shadow:inset 0 1px 0 rgba(255,255,255,.3),
            0 2px 4px rgba(23,34,44,.25)}
.pstep .why{color:#5C6B78}
.walkwrap{padding:11px 14px;background:#F6FAF8;
 border-bottom:1px solid var(--rule)}
.walkwrap .pstep{font-size:13.5px;margin:6px 0}
/* ===== DEPTH & READABILITY LAYER (matches the bench) ===== */
body{font-size:15.5px;line-height:1.55}
.stage{border-radius:12px;
 box-shadow:0 1px 2px rgba(23,34,44,.06),
            0 6px 18px rgba(23,34,44,.08)}
.blurb{font-size:14px;color:#3d4c56;max-width:820px}
.chip{border-radius:10px;
 background:linear-gradient(180deg,#ffffff,#f5f8f7);
 box-shadow:inset 0 1px 0 rgba(255,255,255,.85),
            0 2px 5px rgba(23,34,44,.10);
 transition:transform .06s,box-shadow .06s}
.chip:hover{transform:translateY(-1px);
 box-shadow:0 5px 12px rgba(23,34,44,.16)}
.chip.active{box-shadow:inset 0 2px 5px rgba(23,34,44,.10)}
.chip b{font-size:13.5px}
.chip small{font-size:12px;color:#4a5a54}
.node .plate{filter:drop-shadow(0 3px 5px rgba(23,34,44,.22))}
.node:hover .plate{filter:drop-shadow(0 6px 10px
 rgba(23,34,44,.3))}
#crumbs{font-size:13px}
#crumbs a{font-weight:700}
.codebox{border-radius:10px;overflow:hidden;
 box-shadow:0 3px 10px rgba(23,34,44,.12)}
pre.code{font-size:12.5px}
.compblurb{font-size:13.5px;line-height:1.55}
.codehead{font-size:12px}
footer{font-size:12px}
</style></head><body>
<header>
 <div class="wordmark">SYNTHKIT<small>system atlas v3 &middot;
  plain-English narrative layer &middot;
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
function stepsHtml(steps,color,head){
 var h='<div class="narr"><div class="narrhead">'+head+
  '</div>';
 for(var i=0;i<steps.length;i++){
  h+='<div class="pstep"><span class="nchip" '+
   'style="background:'+color+'">'+(i+1)+'</span><span><b>'+
   esc(steps[i][0])+'</b><span class="why"> &mdash; '+
   esc(steps[i][1])+'</span></span></div>';}
 return h+'</div>';}
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
   '<circle class="plate" cx="'+x+'" cy="'+y+'" r="44" fill="'+d.color+
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
   '<rect class="plate" x="'+(x0-50)+'" y="'+(ry-20)+'" width="'+
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
 var narr='<div class="narr"><div class="narrhead">'+
  'THE PIPELINE IN PLAIN ENGLISH</div>';
 var order=[],r;
 for(r=0;r<DATA.domains.length;r++){
  if(DATA.domains[r].stage)order.push(r);}
 for(r=0;r<DATA.domains.length;r++){
  if(!DATA.domains[r].stage)order.push(r);}
 for(var q=0;q<order.length;q++){
  var dd=DATA.domains[order[q]];
  var bold=dd.plain.replace(/\.$/,'');
  if(!dd.stage)bold+=' (runs alongside every step)';
  narr+='<div class="pstep"><span class="nchip" '+
   'style="background:'+dd.color+'">'+(q+1)+
   '</span><span><b>'+esc(bold)+'</b><span class="why">'+
   ' &mdash; '+esc(DATA.pipeline[dd.id])+
   '</span></span></div>';}
 narr+='</div>';
 stage.innerHTML=s+'<div class="blurb">The pipeline reads '+
  'left to right: what happens to your paragraph. The two '+
  'rails below guard and expose every stage. Every node is '+
  'clickable: domain &rarr; modules &rarr; components '+
  '&rarr; the actual source.</div>'+narr;}
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
   '<rect class="plate" x="'+(x-96)+'" y="'+(y-24)+'" width="192" '+
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
 stage.innerHTML=s+'<div class="blurb">'+d.blurb+'</div>'+
  stepsHtml(d.steps,d.color,
   'WHAT HAPPENS HERE, STEP BY STEP');}
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
  '<div class="walkwrap">'+
  stepsHtml(c.walk,d.color,'PLAIN ENGLISH WALKTHROUGH')+
  '</div>'+
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
    l1 = len(data["domains"])
    l2 = sum(len(d["steps"]) for d in data["domains"])
    l3 = sum(len(c["walk"]) for d in data["domains"]
             for m in d["modules"] for c in m["components"])
    counts = ("{} domains · {} modules · {} components · "
              "{} narrative steps ({} + {} + {})").format(
        len(data["domains"]),
        sum(len(d["modules"]) for d in data["domains"]),
        data["components"], l1 + l2 + l3, l1, l2, l3)
    fp = "v{version} · {built} · {fingerprint}".format(
        **data["build"])
    page = (TEMPLATE
            .replace("@@DATA@@", json.dumps(
                {"domains": data["domains"],
                 "pipeline": NARRATIVE["pipeline"]})
                .replace("</", "<\\/"))
            .replace("@@FP@@", html.escape(fp))
            .replace("@@COUNTS@@", counts))
    OUT.parent.mkdir(exist_ok=True)
    # ---- structural self-check: the page must carry the
    #      narrative layer or the build fails ----
    for marker in ("THE PIPELINE IN PLAIN ENGLISH",
                   "WHAT HAPPENS HERE, STEP BY STEP",
                   "PLAIN ENGLISH WALKTHROUGH",
                   'class="nchip"', "atlas v3"):
        if marker not in page:
            sys.exit("SELF-CHECK FAILED: page missing "
                     "narrative marker {!r}".format(marker))
    embedded = json.loads(page.split(
        '<script id="atlas" type="application/json">', 1)[1]
        .split("</script>", 1)[0]
        .replace("<" + "\\/", "</"))
    for d in embedded["domains"]:
        assert d.get("plain") and d.get("steps"), d["id"]
        for m in d["modules"]:
            for c in m["components"]:
                assert c.get("walk"), c["symbol"]
    OUT.write_text(page, encoding="utf-8")
    print("system map -> {} ({} components, {} KB)".format(
        OUT, data["components"], len(page) // 1024))
    print("narrative: {} steps  (L1 {} · L2 {} · L3 {})"
          .format(l1 + l2 + l3, l1, l2, l3))


if __name__ == "__main__":
    main()
