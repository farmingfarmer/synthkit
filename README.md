# synthkit

Synthetic data creation and testing engine. A user describes a
dataset in plain English; synthkit compiles it to a reviewable
DataSpec, deterministically plans a corpus where every document's
contents are known in advance (the blueprint IS the ground truth),
renders structured fields statistically and unstructured notes via
LLM with verification, and evaluates outside models against exact
planted-element truth — sliced by the axes the planner controlled
(density, difficulty, persona, verbosity, abbreviation, distractors).

No ingestion path exists: distributions are specified, never fitted.
Nothing real ever enters the generator (PHI posture is structural).

## Layout
    synthkit/spec.py       S1: DataSpec schema, validation, JSON round-trip
    synthkit/compiler.py   S1: LLMBackend protocol (Ollama/Anthropic/Bedrock)
                               + plain-English -> DataSpec compiler
    synthkit/planner.py    S2: seeded blueprint expansion, corpus stats
    scripts/smoke_s1_s2.py     17-check smoke, no LLM required

## Pipeline (build order S1 -> S2 -> S4 -> S3 -> S5)
    S1 SpecCompiler    English -> DataSpec (human reviews before use)
    S2 Planner         DataSpec -> blueprints (deterministic; ground truth)
    S3 Renderers       blueprints -> documents (LLM notes + verifier pass)
    S4 Evaluator       outside model vs blueprints -> sliced report
    S5 Forge adapter   hypothesis-shaped test_fn for autonomous testing

## Quick start
    python scripts/smoke_s1_s2.py
