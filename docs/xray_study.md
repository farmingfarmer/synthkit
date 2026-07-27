# The X-ray corpus study: a two-act demonstration

A complete extraction study on a corpus that did not exist the
day before, run twice - once per renderer - against the same
planted truth. All numbers regenerate from `xray.json` (seed 7)
and `xray_extractor.py`.

## The corpus

24 chest X-ray reports. Three planted elements (acute findings,
device positions, comparison changes) at three difficulties; two
trap types: **negation** ("no evidence of pneumothorax" must not
be extracted as a finding) and **historicity** ("stable calcified
granuloma, unchanged from 2019" is not acute). Rendered two ways
from identical blueprints: the deterministic stub, and
mistral-small via ollama with blueprint verification (21/24
first try, 3 caught and retried, 0 fallbacks).

## Act 1 - the study arc (stub renderer)

| arm | recall | negation fp | history fp | ladder |
|---|---|---|---|---|
| extract_naive | 100% | 100% (17/17) | 100% (16/16) | 1/3 |
| extract_blunt | 62% (acute: 13.6%) | 0% | 0% | 0/3 |
| extract_careful | 100% | 0% | 0% | 3/3 |

The classic arc: the naive extractor falls for every trap; the
blunt intervention (sentence-level suppression) kills the traps
AND the recall - the cure-that-kills failure a real intervention
study must check for; the precise intervention (clause-scoped
guard) passes the full ladder including the adversarial tier.

## Act 2 - the transfer test (mistral renderer)

Same spec, same seed, same seventeen planted negations - a
different pen writes the prose.

| arm | recall | negation fp | history fp |
|---|---|---|---|
| extract_naive | 100% | 100% | 100% |
| extract_careful | 100% | **17.6% (3/17)** | 0% |
| extract_careful_v2 | 100% | **0% (0/17)** | 0% |

The guard that scored a perfect 0% on stub prose leaks 3/17 on
LLM prose: mistral writes negations the template-tuned patterns
never saw ("No pneumothorax or pleural effusion is identified" -
compound subject, no matching pattern). The historicity guard
transfers; the negation guard does not. Against a trap_max=0.1
bar, 3/17 carries a Wilson interval of roughly [6%, 41%] - a
fail that knows its own uncertainty.

**The lesson (this is the slide):** an intervention validated
against one renderer's phrasings can be flattered by them.
Re-render the same planted truth with a different pen and the
honest number appears. Evaluate extraction pipelines across
MULTIPLE renderings of the same blueprint truth.

## Act 3 - the hardened arm

`extract_careful_v2` replaces template patterns with structural
negation (bare "no <...>" clauses, compound subjects, "not
identified/seen/present", "without", "absent", "negative for"),
with one surgical exception: "no significant change" is a
legitimate comparison_change VALUE and must not be treated as a
negation. Verified live on both renderers: catches all three
mistral escapes with recall held at 100% (0/17 negation fp on
the LLM corpus), keeps the real finding in the escape report,
preserves the exception, and regresses cleanly on the stub
corpus (100% recall, 0%/0%). The completed arc:
**0% (stub, flattered) -> 17.6% (mistral, honest) ->
0% (hardened, verified)**.

## Act 4 - the third pen (Llama-3.2-3B, CPU, Windows)

The same spec rendered live on the locked-down demo machine
itself: llama.cpp serving Llama-3.2-3B-Instruct (Q4, CPU-only,
16 GB laptop) through synthkit's openai-compatible backend.
Render report at demo size (8 docs): 7 verified first try, 1
after retry, 0 fallbacks. Evaluation of extract_careful_v2:
recall 100%, negation fp 0.0%, historical fp 0.0% - the
structural guard holds against a third renderer's phrasings.
(Small-n caveat: at 8 docs the trap counts are single-digit, so
a campaign would call these trap lines INCONCLUSIVE against a
0.10 bar; the 24-doc rerun is the verdict-grade version.)

Three renderers, one planted truth, one guard: stub 0%,
mistral-24B 0% (after the 17.6% lesson and the v2 repair),
llama-3B 0%. The open-source cost ladder, demonstrated:
$0 local 3B on the demo laptop -> local 24B on a dev Mac ->
Bedrock open weights when sanctioned -> one dropdown apart.

The demo shows BOTH numbers - the 17.6% cliff and v2's repair -
because the cliff sells the problem and the repair sells the
workflow: characterize, intervene, transfer-test, harden,
re-verify. Two study cycles in one afternoon, on an invented
domain, every number reproducible from a JSON file and a seed.

## Act 5 - the 3B takes the witness stand (and fails it)

The role reversal: the same Llama-3.2-3B that RENDERED the
corpus beautifully (7/1/0 verified) was seated as the tested
extractor - blind prompts, JSON protocol, three tiers, 24 calls
on the CPU-only demo machine.

Verdict: **vendor reliability: 24 calls, 17 malformed (71%)**,
0 empty, 0 voted out. Recall 0.294 / 0.235 / 0.235 across
tiers - DECISIVE fails against the 0.8 bar (ci 0.133-0.531
clear of it). Trap fp lines read 0.000 but INCONCLUSIVE: with
most calls malformed, there were barely any extractions left to
trap, and the instrument says so rather than crediting a
specificity it cannot measure.

The lesson is the ROLE ASYMMETRY: generation and disciplined
structured extraction are different capabilities. A model can
write clinical prose convincingly and still be unusable as an
extraction vendor - and the gap here is 71 points of protocol
reliability on the same model, same corpus, same afternoon.
This is the exact test to run on any vendor claiming a small
efficient model "handles clinical extraction."

Engineering footnote: this run also field-proved the salvage
hardening - the 3B's rambling outputs previously triggered
catastrophic regex backtracking (a live frozen-job incident);
the linear scanner processed all 17 rambles in microseconds,
each becoming a malformed tally instead of an infinite spin.

Open follow-ups: majority voting (samples=3) and a
format-forcing intervention ("respond with ONLY a JSON array")
are the natural next arms - the intervention seam is built for
exactly this question: how much protocol reliability can
prompting buy a 3B?

## Reproduce everything

    synthkit render xray.json -o xray_stub --backend stub
    synthkit render xray.json -o xray_llm --backend ollama   # Mac
    synthkit evaluate xray_stub --extractor xray_extractor:extract_careful
    synthkit evaluate xray_llm  --extractor xray_extractor:extract_careful
    synthkit evaluate xray_llm  --extractor xray_extractor:extract_careful_v2
