# Bringing synthkit through the front door

A working plan for introducing synthkit at Keck: provenance,
the demo, deployment paths, and the compliance story. Written to
be pasted into an email or spoken from.

## 1. Provenance (say this first, unprompted)

synthkit is personal R&D: designed, built, and validated entirely
on personal hardware and personal time, in a personal
GitHub repository, before and outside any work assignment. It
contains no Keck code, data, or credentials, and has never touched
PHI — it GENERATES synthetic data by construction. Bringing it to
work is a deliberate, documented act (this file, dated commits),
not an ambiguity. If Keck wants it internal, the clean path is a
fork into the org with a license note, keeping the personal repo
as origin.

## 2. The pitch, in three sentences

We regularly need to judge tools and models — data-quality
vendors, extraction pipelines, prediction claims — and we can
never fully test them on real data because real data has no
answer key and lots of PHI. synthkit generates realistic datasets
where every corruption and every signal is planted and ledgered,
then measures any tool against the theoretical ceiling with
confidence intervals. It has already run a complete study: it
characterized a local LLM's clinically dangerous extraction
failure (26% of discontinued meds extracted as current), fixed it
with a measured prompt intervention (to ~4%, recall intact), and
confirmed at a sample size the tool itself prescribed.

## 3. The 15-minute demo (SageMaker or any Jupyter)

Upload ONE file: `notebooks/synthkit_demo.ipynb`. It needs no
pip, no network, no data — the library is inlined and
self-validating. Run all cells, then walk it top to bottom:

1. **Cells 1-N (the library)** — skip past, mentioning only:
   "stdlib-only, 260+ checks in CI, everything deterministic."
2. **The encounter dataset** — 400 rows from an English
   paragraph; point at the mess-by-op dict and the 14.0%
   prevalence: "every corruption here is deliberate and
   ledgered; the prevalence was DECLARED and is now measured."
3. **Semantic lint** — "validation checks lawfulness; this
   checks intent. In live use it caught a miscalibrated
   outcome before a single row rendered."
4. **The ladder** — point at a DECISIVE interval line and an
   INCONCLUSIVE one with its n-prescription: "the instrument
   knows what it knows, and prices what it doesn't in
   generatable rows."
5. **The showdown** — read the sentence aloud: ceiling /
   baseline / vendor. "This is the meeting summary for any
   vendor claim."
6. **Regression + relational** — one line each: R^2 ceilings
   for cost models; the precision trap for referential-
   integrity tools (watch precision 0.35 -> 1.00).
7. **The study summary** — close on it: "this is not a demo
   scenario; this happened, every number regenerates from a
   spec and a seed, and the full writeup is one file."

Leave behind: the notebook + `docs/first_vendor_trial.md`.

## 4. Deployment paths, in order of ceremony

- **Zero-install**: the demo notebook on any SageMaker kernel.
  This is the default and covers most evaluation work.
- **Library on SageMaker**: `pip install` from a wheel built
  locally (`python -m build`), or from the org fork once it
  exists. Core is stdlib-only; nothing to security-review
  beyond the code itself.
- **Bedrock for LLM steps**: `synthkit.bedrock.BedrockBackend`
  speaks the Converse API and drops into the compiler, the
  document renderer, and the LLM-as-vendor adapter unchanged:

      from synthkit.bedrock import BedrockBackend
      backend = BedrockBackend(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")
      compile_table_spec(description, backend)

  The client is injectable, so the integration is smoke-tested
  without AWS; live use needs only the usual bedrock-runtime
  IAM permissions on the notebook role.
- **The bench**: `synthkit gui` binds 127.0.0.1 only — fine on
  a laptop, not intended for shared hosts without a review.

## 5. The compliance story (one paragraph, ready to send)

synthkit produces entirely synthetic data: names come from
built-in pools, values from declared statistical distributions,
and documents from templates or LLM rendering verified against
generated blueprints. No real records are ingested at any point,
so no PHI exists anywhere in the pipeline — which is precisely
its value: it lets us test extraction, cleaning, prediction, and
integrity tools with clinical REALISM (mess, traps, causal
signal) and zero disclosure risk, before those tools ever meet a
patient record.

## 6. First internal use cases, ranked by fit

1. **Vendor bake-offs** — any data-quality or extraction tool
   under consideration walks the ladder before procurement.
2. **LLM extraction hardening** — the mistral study, repeated on
   whatever model/prompt we actually intend to deploy, via
   Bedrock.
3. **Pipeline regression tests** — freeze a spec + seed as a CI
   fixture; a pipeline change that shifts fix-rates or AUROC
   against known truth fails the build.
4. **Readmission-model realism checks** — regression/predict
   ceilings as sanity bounds when someone reports a
   suspiciously good number on real data.
