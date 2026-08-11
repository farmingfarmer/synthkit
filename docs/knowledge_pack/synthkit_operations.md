---
id: synthkit_operations
display_name: synthkit operations runbook
type: knowledge
status: active
owner: project-owner
tech_stack: [python-stdlib, llama.cpp, ollama, bedrock]
related: [synthkit]
---

# synthkit operations runbook

How synthkit is actually run, synced, and paid for.

## Key commands

synthkit version prints the build fingerprint (must match the
bench's chip). synthkit render SPEC -o DIR --backend
stub|ollama|openai|bedrock generates data. synthkit table-lint
SPEC probe-checks the recipe's promises. synthkit
campaign-compile --goal predict --spec SPEC --bars
auroc=0.6,gap_max=0.3 -o DIR builds the three-tier exam;
synthkit campaign-run DIR runs it (add --llm --llm-backend
openai for an AI in the tested seat). synthkit showdown DIR
--solver vendor_model:predict --baseline autosolver_hybrid is
the head-to-head. synthkit gui serves the bench at
127.0.0.1:8377. python scripts/run_all_smokes.py is the
324-check certification; scripts/build_system_map.py rebuilds
the atlas; scripts/build_presenter.py the presenter companion.

## Phase 2 commands

python scripts/phase2_pipeline.py --src DIR -o OUTDIR --engine
spec|condnet|both [--transcribe] runs the whole loop: wrangle,
label, diagnose, profile, compile, generate, score. The privacy
unit defaults to person_id and is announced up front. Output
folders are git-ignored because the tidy CSV is the source records
reshaped. scripts/power_sweep.py --focused --multilevel reports
how many patients each kind of structure needs.
scripts/narrative_showdown.py runs the ceiling/reading/blind spine
on transcribed notes. scripts/note_vendor_run.py --backend
stub|ollama|openai|bedrock seats a model in the note-extraction
chair and grades it by corruption type.

## The Windows sync ritual (locked-down Windows laptop)

Activate the verbatim venv, remove the old synthkit folder,
curl the GitHub zipball, then ALWAYS size-check the zip before
extracting — a ~106-byte file means the repo is still private
(this exact failure happened twice; megabytes means real).
Extract, rename, pip install -e . (mandatory after every
removal — the editable link died with the old folder), verify
the fingerprint matches the development machine, run the smoke net. Local LLM
on Windows: llama.cpp's win-cpu-x64 build (fetched via the
GitHub release API) serving llama3.2-3b.gguf on port 8080;
synthkit's openai backend defaults to that address. Terminal
topology for demos: terminal one is the model, terminal two the
bench, terminal three the working terminal.

## Cost ladder (order of magnitude; verify current AWS pricing)

The deterministic core costs nothing, forever. Local models:
$0 — the Windows laptop's 3B renders eight notes in about ninety
seconds offline; the development machine's 24B ran 1,662 test calls without a
single malformed reply. AWS Bedrock open-weight models are
HIPAA-eligible at fractions of a cent per thousand tokens — the
entire vendor study would have cost single-digit dollars.
Frontier cloud models are never required. A typical analyst's
AI usage on open weights runs cents to low single-digit
dollars monthly.
