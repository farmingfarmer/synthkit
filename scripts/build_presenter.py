"""Build the PRESENTER COMPANION — a private, single-file HTML
teleprompter for demoing synthkit. Mirrors the atlas's drill
structure (hub -> domains -> key components) but every node
carries: SAY THIS (verbatim script), WHY IT'S BUILT THIS WAY,
SCALING & FUTURE CONFIGURATION, and LIKELY QUESTIONS with
ready answers (costs, LLMs, PHI, enterprise).

    python scripts/build_presenter.py  ->  docs/presenter_companion.html
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / \
    "presenter_companion.html"

# =====================================================================
# All content is authored here. say = verbatim sentences (list);
# why / scale = short paragraphs; faq = [question, answer] pairs.
# =====================================================================
P = {
 "hub": {
  "say": [
   "A vendor wants to sell us a model that predicts thirty-day "
   "readmission in heart failure, claiming an AUROC of at "
   "least 0.75. We can't test them on real patients — no "
   "answer key, and it's all PHI.",
   "So we built the exam instead of finding one: a synthetic "
   "population where every value, every flaw, and every truth "
   "was planted on purpose. We know the maximum achievable "
   "score before anyone sits the test.",
   "In the next twenty minutes you'll watch a paragraph of "
   "plain English become a thousand-patient dataset with "
   "messy clinical notes, watch our own free model learn to "
   "read those notes, and watch the vendor's model take the "
   "identical exam — and you'll see who wins and exactly why.",
   "Everything runs on this laptop. No cloud, no real data, "
   "no cost.",
  ],
  "faq": [
   ["What did this system cost to build and what does it "
    "cost to run?",
    "Build: engineering time only — it is pure Python "
    "standard library, zero licensed components, zero "
    "external packages in the core. Run: $0 on this laptop; "
    "the deterministic writer and our models are CPU-only. "
    "The only metered option is cloud AI, which is optional "
    "and priced per use (see the Interfaces node)."],
   ["Is any real patient data involved, ever?",
    "No. Nothing connects to Cerner or any clinical system. "
    "Every record is generated from a recipe; the 'patients' "
    "have no source individuals. That's also why we can hand "
    "the dataset to a vendor freely."],
   ["Which AI models does this use?",
    "Three distinct seats, all optional: (1) a drafting AI "
    "that turns English into a recipe — a human approves "
    "everything; (2) an optional writer that phrases the "
    "clinical notes — verified line-by-line against the "
    "answer key; (3) any AI can sit in the TESTED seat as a "
    "vendor. The scoring, generation, and our own risk model "
    "use no AI at all — they're deterministic code."],
   ["How long does a full evaluation take?",
    "On this laptop: data generation is instant, our model "
    "trains in about a minute, the full three-difficulty "
    "showdown lands in two to three. An analyst's first "
    "custom dataset, from paragraph to verdict, is an "
    "afternoon including review."],
   ["Why should we trust results on synthetic data?",
    "Because it's the only data where the truth is KNOWN. "
    "Synthetic testing doesn't replace clinical validation — "
    "it comes first: it filters vendor claims cheaply and "
    "safely before any real-data pilot, and it can be made "
    "adversarial in ways real data can't."],
  ]},
 "domains": [
  {"id": "spec", "label": "Spec Layer", "color": "#1B5FAA",
   "say": [
    "Everything starts with a paragraph. Watch: this is the "
    "actual English we'll use — one record per patient, a "
    "bimodal ejection fraction, ten percent missing ages, a "
    "discharge note with risk factors hidden in messy "
    "language, and a readmission rate promised between five "
    "and twelve percent.",
    "An AI drafts the formal recipe from that paragraph — "
    "and here's the part I want you to remember: the draft "
    "is never trusted. A validator marks every problem in "
    "plain teaching language, and a person approves before "
    "anything is created. In one live run, a small AI "
    "invented three impossible settings — and the gate "
    "caught all three.",
   ],
   "why": "Assisted-never-trusted. LLMs are excellent "
    "drafters and unreliable finishers, so the design gives "
    "them the first word and never the last. Every validator "
    "message doubles as curriculum — failures literally "
    "become chapters in the drafting AI's instructions.",
   "scale": "Enterprise: the recipe IS the governance "
    "artifact — a JSON file that can be versioned, "
    "code-reviewed, and approved in a normal PR workflow. "
    "The drafting AI is swappable per deployment: local "
    "models for private drafting, Bedrock for a governed "
    "route, or none at all (analysts can author recipes "
    "directly or start from the preset library, which is "
    "how a template culture forms).",
   "faq": [
    ["Does the drafting AI see any sensitive information?",
     "It sees only the English paragraph the analyst wrote. "
     "With a local model, that text never leaves the "
     "machine."],
    ["What does compilation cost?",
     "Local models: $0. On Bedrock, a compile is a few "
     "thousand tokens — fractions of a cent on open-weight "
     "models, a cent or two on frontier models. Negligible "
     "at any scale."],
    ["Can non-technical staff really write these?",
     "They write the paragraph; the system writes the "
     "recipe; the validator explains problems in plain "
     "terms. The demo dataset's paragraph is on screen — "
     "it's prose, not code."],
   ],
   "components": [
    {"name": "The human gate",
     "say": "Nothing an AI drafts becomes real without a "
      "person clicking approve — and the validator arms that "
      "person with a complete, plain-English problem list.",
     "why": "The failure mode of AI tooling is silent "
      "confident error. The gate converts every AI mistake "
      "into a visible teaching moment instead.",
     "scale": "At enterprise scale the gate maps onto "
      "existing review culture: recipes in git, approvals in "
      "PRs, a named owner per template.",
     "faq": [
      ["What if the AI's draft is wrong in a way the "
       "validator misses?",
       "Two more gates follow: the semantic lint generates a "
       "sample and measures it against the promises, and the "
       "human sees a full preview before any evaluation "
       "runs."]]},
    {"name": "The recipe as a contract",
     "say": "This fingerprint at the top — that's the "
      "recipe's identity. Same fingerprint, same data, "
      "forever, on any machine. If I change one character, "
      "watch the fingerprint change.",
     "why": "Reproducibility is the difference between 'we "
      "measured' and 'we remember'. Every result in this "
      "demo traces to a fingerprinted recipe.",
     "scale": "Auditors and vendors receive the recipe file "
      "itself — anyone can regenerate the exact evaluation "
      "dataset and verify our numbers independently.",
     "faq": [
      ["Could a vendor game the test if they have the "
       "recipe?",
       "They receive the recipe only after evaluation, if at "
       "all — and campaigns can regenerate fresh populations "
       "from new seeds at any time, which is exactly what "
       "the three-difficulty exam already does."]]},
   ]},
  {"id": "truth", "label": "Truth Planning",
   "color": "#6B3FA0",
   "say": [
    "Here's the trick that makes honest grading possible: "
    "the answer key is built BEFORE the data. Every cell "
    "gets its own seeded dice roll; who gets readmitted is "
    "decided by known causes with known strengths.",
    "And the causes live where they live in real medicine — "
    "in the notes. 'Ran out of furosemide two weeks ago.' "
    "'Lives alone, no home support.' Those phrases "
    "genuinely drive the outcome, which means a model that "
    "can't read tops out at 0.60 while 0.82 is achievable. "
    "We didn't assert that — we measured it.",
    "And the traps: 'denies missing any doses' appears ONLY "
    "in patients who truly didn't. A model matching "
    "keywords without understanding negation gets those "
    "exactly backwards.",
   ],
   "why": "Planted causality is what separates this from "
    "'realistic-looking fake data'. Realism is necessary "
    "but not sufficient — the point is a KNOWN ceiling and "
    "a KNOWN answer for every record.",
   "scale": "The mechanism is domain-agnostic: sepsis "
    "deterioration, denial-of-claims, med-rec errors — any "
    "outcome with tabular-plus-text evidence. New domains "
    "are new recipes, not new code. Note phrasing libraries "
    "can be grown per specialty and shared as templates.",
   "faq": [
    ["How realistic is the mess, really?",
     "Every corruption type came from real extracts: mixed "
     "date formats, unit typos, casing drift, duplicated "
     "rows, implausible outliers, and contradictory prose. "
     "And unlike real data, every single corruption is "
     "ledgered — we can grade a cleaning tool cell by "
     "cell."],
    ["Could we tune difficulty for different vendors?",
     "Yes — that's built in. Every campaign runs three "
     "difficulty tiers automatically, and the recipe's "
     "densities, weights, and mess rates are all dials."],
    ["What does generation cost at scale?",
     "Effectively zero: it's deterministic CPU code. A "
     "thousand patients generate in seconds on a laptop; a "
     "million is minutes on any server."],
   ],
   "components": [
    {"name": "Notes that move the outcome",
     "say": "This is the capstone mechanism. When we render "
      "the data you'll see one patient 'admits missing "
      "several doses' and another 'denies missing any' — "
      "and only one of those patients carries the extra "
      "risk. The exam rewards reading.",
     "why": "Because that's where the vendor market "
      "actually splits: plenty of tools model the tabular "
      "vitals; the clinical value locked in notes is the "
      "hard part, so the benchmark makes it worth exactly "
      "0.22 of AUROC — measured.",
     "scale": "Phrase libraries per specialty (cardiology, "
      "onc, ED) become shared assets; clinicians contribute "
      "phrasings, which is a natural engagement point.",
     "faq": [
      ["Who decided the risk factors and weights?",
       "For the demo, clinical common sense. For production "
       "use, that's a clinician-governance question — and "
       "the recipe format makes their decisions explicit, "
       "reviewable, and versioned."]]},
    {"name": "The corruption ledger",
     "say": "Every deliberate flaw is written down — this "
      "download here is the ledger. Seven hundred "
      "corruptions in this run, each one signed: what cell, "
      "what kind, what the truth was.",
     "why": "It turns 'our tool handles messy data' from a "
      "slogan into a gradeable claim.",
     "scale": "The ledger is also the audit story: any "
      "regulator or vendor can verify the mess was "
      "deliberate, bounded, and disclosed.",
     "faq": [
      ["Can vendors see the ledger?",
       "After evaluation, sure — it's the explanation of "
       "their errors. Before, no: the ledger is the answer "
       "key's twin."]]},
   ]},
  {"id": "render", "label": "Rendering", "color": "#3D5A6C",
   "say": [
    "One click writes three artifacts: the messy dataset a "
    "model actually faces, the clean answer key, and the "
    "ledger. There's the preview — null cells, a typo'd "
    "unit, mixed date formats — and click any note to read "
    "it full screen.",
    "The prose you're reading was written by the built-in "
    "deterministic writer — instant, free, identical every "
    "run. If we want more natural language, we flip one "
    "dropdown and a local AI writes it instead — and every "
    "AI-written sentence is verified against the answer key "
    "and corrected if it drifts.",
   ],
   "why": "Determinism by default, AI by choice, "
    "verification always. The demo can never be derailed by "
    "a model having a bad day — worst case, the "
    "deterministic fallback guarantees completion.",
   "scale": "The writer seam is where cloud scale plugs in: "
    "the same recipe renders through a local 3B model, a "
    "24B model, or Bedrock's catalog by changing one "
    "selection. Batch rendering of large corpora "
    "parallelizes trivially because every document is "
    "independently seeded.",
   "faq": [
    ["What does AI-written rendering cost?",
     "Local models: $0 — this laptop's 3-billion-parameter "
     "model wrote eight verified notes in about ninety "
     "seconds, offline. Cloud: a 1,000-note corpus is "
     "roughly a few hundred thousand tokens — single-digit "
     "dollars on Bedrock's open-weight models, order of "
     "magnitude more on frontier models. Check current AWS "
     "pricing for exact figures."],
    ["How do you know the AI writer didn't change the "
     "facts?",
     "Every note is checked fact-by-fact against its "
     "blueprint. On this laptop's live run: seven verified "
     "first try, one caught and corrected, zero failures. "
     "The check is the feature."],
   ],
   "components": [
    {"name": "Verify-retry-fallback",
     "say": "Nothing an AI writes is trusted: verified, "
      "retried if wrong, replaced deterministically if "
      "hopeless. The render report shows those counts every "
      "run — the safety net performs in public.",
     "why": "It converts 'AI quality' from a risk into a "
      "measured, visible statistic.",
     "scale": "The same verification loop is model-agnostic "
      "— it's what makes swapping writers safe.",
     "faq": [
      ["What happens with a much worse AI writer?",
       "More retries, more fallbacks, same guaranteed "
       "output — and the report tells you the writer was "
       "weak, which is itself useful procurement data."]]},
   ]},
  {"id": "eval", "label": "Evaluation & Ceilings",
   "color": "#B7791F",
   "say": [
    "Scores here mean more than usual, for one reason: we "
    "know the ceiling. This population supports at most "
    "0.82 — not estimated, known, because we planted the "
    "causes.",
    "And every number carries a how-sure range. When a trap "
    "rate is three-out-of-seventeen, the instrument says "
    "'somewhere between 6 and 41 percent' — it refuses to "
    "pretend precision it doesn't have.",
   ],
   "why": "Uncertainty honesty is a design principle, not a "
    "footnote: verdicts are computed on confidence "
    "intervals, never bare point estimates.",
   "scale": "The metrics library is pure stdlib and "
    "portable anywhere — the same honest scoring runs in a "
    "CI pipeline, a notebook, or this bench.",
   "faq": [
    ["Why does the ceiling matter so much?",
     "Because 0.72 is a poor score against 0.95 and an "
     "excellent one against 0.82. Without a known ceiling, "
     "vendors get judged against imagination."],
    ["What's the cost of evaluation?",
     "Zero marginal: deterministic CPU arithmetic, "
     "milliseconds per model per tier."],
   ],
   "components": [
    {"name": "INCONCLUSIVE with a prescription",
     "say": "When there isn't enough data to call it, the "
      "verdict says INCONCLUSIVE — and prescribes exactly "
      "how many more samples would settle it. The "
      "instrument tells you the price of certainty.",
     "why": "The most dangerous evaluation output is a "
      "confident answer from insufficient data.",
     "scale": "That prescription is also the budgeting "
      "tool: it converts statistical power into a concrete "
      "'generate N more' instruction.",
     "faq": [
      ["Has that prescription been validated?",
       "Yes — live: a trap-rate finding was re-run at the "
       "instrument's own prescribed sample size and "
       "confirmed at the tighter interval."]]},
   ]},
  {"id": "campaign", "label": "Campaigns & Verdicts",
   "color": "#17803D",
   "say": [
    "One click builds the same exam at three difficulties — "
    "gentler, exactly-as-specified, and adversarial — so "
    "nobody passes by luck on one easy draw.",
    "Contestants train on a separate practice population; "
    "answers stay hidden during the test. And every result "
    "files permanently — every claim in this room traces to "
    "a file on disk.",
   ],
   "why": "Tiering plus blinding is what makes this an "
    "instrument rather than a demo: the same protocol, "
    "every time, for every contestant.",
   "scale": "Campaign directories are plain files — they "
    "archive, ship, and diff. An enterprise rollout is a "
    "shared campaign library plus a results registry, no "
    "database required.",
   "faq": [
    ["Can we standardize one exam across all vendor "
     "evaluations?",
     "Yes — that's the intended use: a recipe plus pass "
     "marks becomes the department's standing exam, "
     "versioned like any policy document."],
   ],
   "components": [
    {"name": "The append-only trial record",
     "say": "Results never overwrite — they accumulate. "
      "Here's the trials view: every run, every arm, every "
      "verdict, timestamped.",
     "why": "Paper trails are the difference between "
      "evaluation and anecdote.",
     "scale": "The record format is trivially exportable to "
      "any GRC or reporting system.",
     "faq": [
      ["Who can modify past results?",
       "Nobody through the tool — the write path only "
       "appends. File-system permissions handle the rest, "
       "same as any governed artifact."]]},
   ]},
  {"id": "solvers", "label": "Solvers & Vendors",
   "color": "#4C7A34",
   "say": [
    "Before any vendor is judged, we field our own "
    "challenger — free, transparent, built in about a "
    "minute. It standardizes the messy table, then mines "
    "the notes: from training records alone it learns which "
    "phrases predict readmission — and it keeps 'denies "
    "missing doses' separate from 'missing doses', because "
    "in medicine the denial usually points the other way.",
    "And it's not a black box: after the showdown you'll "
    "see its actual learned weights, named in English — "
    "'note phrase kg mentioned, raises risk' sitting right "
    "beside ejection fraction.",
    "The vendor's model gets the identical seat: same "
    "training data, same hidden answers. Ours reads; "
    "theirs doesn't; the gap is 0.12 — measured.",
   ],
   "why": "A vendor should never be compared to nothing. "
    "The free-challenger floor reframes every procurement "
    "conversation: beat the model we get for free, on data "
    "where we know the answers.",
   "scale": "The vendor seat accepts any model via a "
    "ten-line wrapper: their library, their API, or an "
    "offline scoring exchange — three routes documented, "
    "answer key never leaves home. AI vendors get an "
    "additional honesty meter: unusable answers are "
    "counted, and one small model failed to follow "
    "instructions on 17 of 24 calls — a procurement "
    "finding in itself.",
   "faq": [
    ["What model is 'our challenger' exactly?",
     "Logistic regression — the most defensible model in "
     "clinical statistics — over automatically standardized "
     "table features plus label-mined note phrases. "
     "Standard library only, deterministic, every weight "
     "inspectable."],
    ["Why not use a fancier model as the baseline?",
     "The floor should be cheap, transparent, and "
     "reproducible — its job is to be undeniable, not "
     "maximal. Anything a vendor sells must clear it; a "
     "fancier internal model can always be seated as a "
     "second contestant."],
    ["What did testing an actual AI vendor cost?",
     "Locally: $0 and an afternoon — 1,662 calls to a "
     "local model with zero malformed answers. The same "
     "study through Bedrock's open-weight models would be "
     "single-digit dollars; through frontier cloud models, "
     "tens."],
   ],
   "components": [
    {"name": "The note miner",
     "say": "It reads every training note and finds "
      "predictive phrases using only the training answers — "
      "it has never seen the recipe. What it found on this "
      "cohort: weight-gain language, lives-alone language, "
      "medication-refill language. It rediscovered the "
      "clinical truth from prose.",
     "why": "Learning from labels rather than the recipe "
      "keeps the demonstration honest — our model wins by "
      "reading, not by cheating.",
     "scale": "The same miner generalizes to any note "
      "field; specialty stopword and negation lists are the "
      "only tuning surface.",
     "faq": [
      ["Is this NLP? Do we need GPUs?",
       "It's classical text statistics — counts, "
       "correlations, a negation window. CPU, milliseconds, "
       "no GPU anywhere in the core system."]]},
    {"name": "Model transparency",
     "say": "These bars are the model's real trained "
      "coefficients — not a summary, the actual weights. "
      "Building this view caught a real bug: patient NAMES "
      "had snuck in as risk factors. Now identifiers are "
      "excluded from evidence categorically.",
     "why": "If we demand explainability from vendors, our "
      "own challenger must over-deliver on it.",
     "scale": "The same introspection publishes to any "
      "reporting surface — it's a data structure, not a "
      "screen.",
     "faq": [
      ["Can the vendor's internals be shown the same "
       "way?",
       "Only if they expose them — and their refusal is "
       "itself informative. We judge black boxes on scores; "
       "we judge our own model on scores AND internals."]]},
    {"name": "The AI reliability meter",
     "say": "When an AI model sits in the tested seat, we "
      "also count how often it fails to follow the answer "
      "format. One 3-billion-parameter model: 17 unusable "
      "answers out of 24 — while a 24-billion model went "
      "1,662 for 1,662. Capability tiers, measured.",
     "why": "'Can write clinical text' and 'can be trusted "
      "to extract from it' turn out to be different skills "
      "— a 71-point reliability gap on the same model, "
      "same corpus, same afternoon.",
     "scale": "This is the exact test to run on any vendor "
      "claiming a small efficient model handles clinical "
      "extraction.",
     "faq": [
      ["Did the unreliable model crash the system?",
       "No — every ramble was counted calmly in "
       "microseconds. It did expose a performance bug in "
       "an earlier version, which is now fixed and "
       "regression-tested; the test suite is 324 checks "
       "green on both machines."]]},
   ]},
  {"id": "gates", "label": "Quality Gates",
   "color": "#B7791F",
   "say": [
    "Three gates run alongside everything: is the recipe "
    "lawful, does the data keep its promises, and are we "
    "statistically sure. Here's the middle one live: the "
    "lint generated a sample and reports the realized "
    "readmission rate — 9.3 percent, inside the promised 5 "
    "to 12. The data means what the paragraph said.",
   ],
   "why": "Every check in the gates was a human catch "
    "first — the suite is institutionalized experience.",
   "scale": "Gates run headless in CI: recipes can be "
    "linted on every commit, so a template library stays "
    "trustworthy as it grows.",
   "faq": [
    ["What stops someone shipping a bad recipe?",
     "The validator refuses unlawful ones; the lint flags "
     "broken promises with numbers; and the human gate "
     "means someone owns the approval."],
   ],
   "components": [
    {"name": "Semantic lint",
     "say": "Validation asks 'is it lawful'; lint asks 'is "
      "it TRUE' — it generates a probe sample and measures "
      "the promises. It once caught a prevalence drift a "
      "human review had missed.",
     "why": "Specs can be lawful and still wrong; only "
      "generated data can testify.",
     "scale": "Probe size is a dial: bigger probes, "
      "tighter guarantees, still seconds of CPU.",
     "faq": [
      ["Does lint slow the workflow?",
       "It's one click and a few seconds — and it's "
       "optional-but-recommended, marked exactly that way "
       "in the interface."]]},
   ]},
  {"id": "iface", "label": "Interfaces & Backends",
   "color": "#0E6E64",
   "say": [
    "Everything you've watched ran through the five-step "
    "bench — numbered steps, red-green readiness on every "
    "required field, plain-English explanations "
    "everywhere. A clinician can walk this alone; the "
    "command line does everything the bench does for "
    "engineers and CI.",
    "And the AI connections are on our terms: a free local "
    "model on this very laptop, the hospital's governed "
    "AWS, or none at all. One clarification I'll make "
    "before anyone asks: the option labeled "
    "'OpenAI-compatible' is a message FORMAT the "
    "local-AI world standardized on — nothing here talks "
    "to ChatGPT; it works with the network unplugged.",
   ],
   "why": "Adoption dies at the interface. The bench "
    "invests in the first five minutes of a non-technical "
    "user's experience, because that's where tools live or "
    "die.",
   "scale": "The backend seam is the whole cloud story in "
    "one dropdown: local 3B ($0, offline) -> local 24B "
    "($0, workstation) -> Bedrock open weights "
    "(HIPAA-eligible, cents) -> a shared vLLM cluster "
    "(near-zero marginal at scale). Same recipes, same "
    "exams, every price point.",
   "faq": [
    ["What would enterprise-wide AI usage cost?",
     "The core system is $0 forever. For AI rendering and "
     "compiling: local models cost nothing but hardware "
     "already owned; Bedrock open-weight usage for a "
     "typical analyst — a few compiles and corpora a week "
     "— is cents to low single-digit dollars monthly per "
     "seat; a shared on-prem serving cluster amortizes to "
     "near-zero marginal. The expensive option (frontier "
     "cloud models) is never required. Figures are "
     "order-of-magnitude as of mid-2026 — verify current "
     "AWS pricing before budgeting."],
    ["Is this deployed? What's the footprint?",
     "It runs from a folder — no installer, no database, "
     "no admin rights; it's running on a locked-down "
     "hospital laptop right now. Enterprise deployment is "
     "'copy the folder, pin the version', and the test "
     "suite certifies any machine in thirty seconds."],
    ["What about Epic/Cerner integration?",
     "Deliberately none — this system never touches "
     "clinical systems. It evaluates models BEFORE they "
     "earn access to real data. The output that travels is "
     "paperwork: recipes, datasets, verdicts."],
   ],
   "components": [
    {"name": "The five-step bench",
     "say": "Numbered steps, required-versus-optional "
      "badges, fields that turn green when ready, buttons "
      "that politely refuse until they are — and a final "
      "report a stakeholder can repeat in a meeting I'm "
      "not in.",
     "why": "The tool argues for itself: every label was "
      "rewritten through a clinician's eyes, and every "
      "confusing one we found became a fix and a test.",
     "scale": "It's a single-file server — trivially "
      "hostable on a shared workstation for a whole team.",
     "faq": [
      ["Browser requirements?",
       "Any modern browser, localhost only by default — "
       "nothing is exposed to the network unless "
       "deliberately hosted."]]},
    {"name": "The $0 local AI door",
     "say": "This laptop runs its own 3-billion-parameter "
      "model through a standard local server — it wrote "
      "clinical prose live, offline, for free. The same "
      "door speaks to LM Studio, vLLM, and the hospital's "
      "future serving cluster.",
     "why": "The open-source path removes the budget "
      "conversation from every experiment — iteration "
      "becomes free, so it actually happens.",
     "scale": "One environment variable points the door at "
      "any endpoint: laptop today, shared cluster "
      "tomorrow, no code change.",
     "faq": [
      ["Why not just use ChatGPT?",
       "Governance and cost: local and Bedrock routes keep "
       "data in controlled environments at near-zero "
       "prices — and no capability in this system needs a "
       "frontier model. The instrument's power is in the "
       "planted truth, not the size of any AI."]]},
    {"name": "Bedrock: the governed route",
     "say": "When we want cloud scale, the recipe doesn't "
      "change — the dropdown does. Bedrock serves "
      "open-weight and Claude models inside the "
      "hospital's existing AWS governance, HIPAA-eligible.",
     "why": "Meeting the enterprise where its governance "
      "already lives beats asking for new trust.",
     "scale": "The Bedrock connector is already smoked "
      "end-to-end with injected transports; activation is "
      "an IAM request, not an engineering project.",
     "faq": [
      ["What's the Bedrock cost model?",
       "Pure per-token, no minimums: open-weight models "
       "run fractions of a cent per thousand tokens. The "
       "entire 1,662-call vendor study would have been "
       "single-digit dollars. Verify current pricing — "
       "it only goes down."]]},
   ]},
 ],
}


def build() -> str:
    # ---- structural self-check on the authored content ----
    assert P["hub"]["say"] and len(P["hub"]["faq"]) >= 4
    n_comp = n_faq = 0
    for d in P["domains"]:
        for k in ("id", "label", "color", "say", "why",
                  "scale", "faq", "components"):
            assert d.get(k), (d.get("id"), k)
        n_faq += len(d["faq"])
        for c in d["components"]:
            for k in ("name", "say", "why", "scale", "faq"):
                assert c.get(k), (d["id"], c.get("name"), k)
            n_comp += 1
            n_faq += len(c["faq"])
    page = (TEMPLATE
            .replace("@@DATA@@",
                     json.dumps(P).replace("</", "<\\/"))
            .replace("@@COUNTS@@",
                     "{} domains · {} deep-dive nodes · {} "
                     "FAQ answers".format(
                         len(P["domains"]), n_comp,
                         n_faq + len(P["hub"]["faq"]))))
    for marker in ("SAY THIS", "WHY IT&#x27;S BUILT" if False
                   else "WHY IT", "SCALING",
                   "LIKELY QUESTIONS", "PRESENTER COMPANION"):
        if marker not in page:
            sys.exit("SELF-CHECK FAILED: missing {!r}"
                     .format(marker))
    return page


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>synthkit — presenter companion (private)</title>
<style>
:root{--bench:#20272D;--panel:#2A333B;--ink:#E8EDF1;
 --dim:#9FB0BE;--rule:#3C4852;--enamel:#19B597;
 --mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;
 --sans:'IBM Plex Sans',system-ui,sans-serif}
*{box-sizing:border-box;margin:0}
body{background:var(--bench);color:var(--ink);
 font-family:var(--sans);font-size:16px;line-height:1.6;
 padding:24px 30px}
header{display:flex;justify-content:space-between;
 align-items:baseline;margin-bottom:16px}
.wordmark{font-family:var(--mono);font-weight:600;
 letter-spacing:.14em;font-size:17px}
.wordmark small{display:block;color:var(--enamel);
 font-weight:600;letter-spacing:.1em;font-size:10.5px;
 margin-top:3px}
#crumbs{font-family:var(--mono);font-size:13px;
 color:var(--dim)}
#crumbs a{color:var(--enamel);cursor:pointer;
 text-decoration:none;font-weight:700}
.stage{background:var(--panel);border:1px solid var(--rule);
 border-radius:14px;padding:22px 26px;min-height:520px;
 box-shadow:0 2px 4px rgba(0,0,0,.3),
            0 12px 32px rgba(0,0,0,.35)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,
 minmax(240px,1fr));gap:12px;margin-top:14px}
.card{border-radius:11px;padding:14px 16px;cursor:pointer;
 background:linear-gradient(180deg,#323D46,#28313A);
 border:1px solid var(--rule);
 box-shadow:inset 0 1px 0 rgba(255,255,255,.06),
            0 3px 8px rgba(0,0,0,.3);
 transition:transform .06s,box-shadow .06s}
.card:hover{transform:translateY(-2px);
 box-shadow:0 8px 18px rgba(0,0,0,.45)}
.card b{font-size:15.5px;display:block}
.card small{color:var(--dim);font-size:12.5px}
.dot{display:inline-block;width:11px;height:11px;
 border-radius:4px;margin-right:8px;vertical-align:0}
.sect{margin-top:20px}
.secthead{font-family:var(--mono);font-size:12px;
 letter-spacing:.14em;font-weight:700;margin-bottom:9px}
.say{background:#1B2228;border-left:4px solid var(--accent,
 #19B597);border-radius:0 10px 10px 0;padding:13px 17px;
 margin:9px 0;font-size:16.5px;line-height:1.65;
 box-shadow:inset 0 1px 0 rgba(255,255,255,.04)}
.why,.scale{color:#C9D4DD;font-size:14.5px;max-width:900px;
 padding:2px 2px}
details.faq{background:#242D35;border:1px solid var(--rule);
 border-radius:10px;margin:8px 0;overflow:hidden}
details.faq summary{cursor:pointer;padding:11px 15px;
 font-weight:700;font-size:14.5px;list-style:none}
details.faq summary::before{content:"Q  ";
 color:var(--accent,#19B597);font-family:var(--mono)}
details.faq[open] summary{border-bottom:1px solid var(--rule)}
details.faq .a{padding:11px 15px;color:#C9D4DD;
 font-size:14.5px;line-height:1.6}
details.faq .a::before{content:"A  ";
 color:var(--accent,#19B597);font-family:var(--mono);
 font-weight:700}
footer{margin-top:14px;font-family:var(--mono);font-size:12px;
 color:var(--dim);display:flex;justify-content:space-between}
</style></head><body>
<header>
 <div class="wordmark">SYNTHKIT<small>PRESENTER COMPANION
  &middot; PRIVATE &mdash; for the presenter&#39;s screen
  only</small></div>
 <div id="crumbs"></div>
</header>
<div class="stage" id="stage"></div>
<footer><span>@@COUNTS@@</span>
 <span>rebuild: python scripts/build_presenter.py</span></footer>
<script id="pdata" type="application/json">@@DATA@@</script>
<script>
var P=JSON.parse(document.getElementById('pdata').textContent);
var stage=document.getElementById('stage');
var crumbs=document.getElementById('crumbs');
function esc(s){return String(s).replace(/&/g,'&amp;')
 .replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function crumbHtml(list){
 var out=[];
 for(var i=0;i<list.length;i++){
  out.push(list[i].act?'<a data-act="'+list[i].act+'"'+
   (list[i].i!==undefined?' data-i="'+list[i].i+'"':'')+
   '>'+esc(list[i].t)+'</a>':esc(list[i].t));}
 crumbs.innerHTML=out.join(' &nbsp;/&nbsp; ');}
function says(arr,color){
 var h='<div class="sect"><div class="secthead" '+
  'style="color:'+color+'">SAY THIS &mdash; VERBATIM</div>';
 for(var i=0;i<arr.length;i++){
  h+='<div class="say" style="border-left-color:'+color+
   '">&ldquo;'+esc(arr[i])+'&rdquo;</div>';}
 return h+'</div>';}
function block(head,body,color){
 return '<div class="sect"><div class="secthead" '+
  'style="color:'+color+'">'+head+'</div>'+
  '<div class="'+(head.indexOf('SCALING')===0?
  'scale':'why')+'">'+esc(body)+'</div></div>';}
function faqs(arr,color){
 var h='<div class="sect"><div class="secthead" '+
  'style="color:'+color+'">LIKELY QUESTIONS</div>';
 for(var i=0;i<arr.length;i++){
  h+='<details class="faq" style="--accent:'+color+'">'+
   '<summary>'+esc(arr[i][0])+'</summary>'+
   '<div class="a">'+esc(arr[i][1])+'</div></details>';}
 return h+'</div>';}
function hub(){
 crumbHtml([{t:'companion'}]);
 var h='<div style="font-size:14px;color:var(--dim)">'+
  'The opener &mdash; then click any part of the system for '+
  'its script, rationale, scaling notes, and ready answers.'+
  '</div>'+says(P.hub.say,'#19B597');
 h+='<div class="grid">';
 for(var i=0;i<P.domains.length;i++){
  var d=P.domains[i];
  h+='<div class="card" data-act="domain" data-i="'+i+'">'+
   '<b><span class="dot" style="background:'+d.color+
   '"></span>'+esc(d.label)+'</b><small>'+
   d.components.length+' deep-dive node'+
   (d.components.length>1?'s':'')+' &middot; '+
   d.faq.length+' FAQs</small></div>';}
 h+='</div>'+faqs(P.hub.faq,'#19B597');
 stage.innerHTML=h;}
function domainView(i){
 var d=P.domains[i];
 crumbHtml([{t:'companion',act:'hub'},{t:d.label}]);
 var h=says(d.say,d.color)+
  block('WHY IT IS BUILT THIS WAY',d.why,d.color)+
  block('SCALING &amp; FUTURE CONFIGURATION',d.scale,
   d.color)+
  '<div class="sect"><div class="secthead" style="color:'+
  d.color+'">DEEP-DIVE NODES</div><div class="grid">';
 for(var k=0;k<d.components.length;k++){
  h+='<div class="card" data-act="comp" data-i="'+i+
   '" data-k="'+k+'"><b><span class="dot" '+
   'style="background:'+d.color+'"></span>'+
   esc(d.components[k].name)+'</b><small>'+
   d.components[k].faq.length+' FAQ'+
   (d.components[k].faq.length>1?'s':'')+
   '</small></div>';}
 h+='</div></div>'+faqs(d.faq,d.color);
 stage.innerHTML=h;}
function compView(i,k){
 var d=P.domains[i],c=d.components[k];
 crumbHtml([{t:'companion',act:'hub'},
  {t:d.label,act:'domain',i:i},{t:c.name}]);
 stage.innerHTML=says(c.say instanceof Array?c.say:[c.say],
   d.color)+
  block('WHY IT IS BUILT THIS WAY',c.why,d.color)+
  block('SCALING &amp; FUTURE CONFIGURATION',c.scale,
   d.color)+
  faqs(c.faq,d.color);}
document.addEventListener('click',function(ev){
 var el=ev.target;
 while(el&&el!==document&&
  !(el.getAttribute&&el.getAttribute('data-act'))){
  el=el.parentNode;}
 if(!el||el===document)return;
 var act=el.getAttribute('data-act');
 var i=parseInt(el.getAttribute('data-i'),10);
 var k=parseInt(el.getAttribute('data-k'),10);
 if(act==='hub')hub();
 else if(act==='domain')domainView(i);
 else if(act==='comp')compView(i,k);});
hub();
</script></body></html>
"""


def main():
    page = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print("presenter companion -> {} ({} KB)".format(
        OUT, len(page) // 1024))


if __name__ == "__main__":
    main()
