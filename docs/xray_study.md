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

The demo shows BOTH numbers - the 17.6% cliff and v2's repair -
because the cliff sells the problem and the repair sells the
workflow: characterize, intervene, transfer-test, harden,
re-verify. Two study cycles in one afternoon, on an invented
domain, every number reproducible from a JSON file and a seed.

## Reproduce everything

    synthkit render xray.json -o xray_stub --backend stub
    synthkit render xray.json -o xray_llm --backend ollama   # Mac
    synthkit evaluate xray_stub --extractor xray_extractor:extract_careful
    synthkit evaluate xray_llm  --extractor xray_extractor:extract_careful
    synthkit evaluate xray_llm  --extractor xray_extractor:extract_careful_v2
