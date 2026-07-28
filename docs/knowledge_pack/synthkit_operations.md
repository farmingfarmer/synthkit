---
id: synthkit-operations
type: operations
name: synthkit operations (how to run, sync, and pay for it)
part_of: synthkit
updated: 2026-07-28
tags: [runbook, cli, sync-ritual, costs, backends]
---

# synthkit operations

## Canonical commands (CLI)
- synthkit version — build fingerprint (must match the bench chip)
- synthkit table-compile <english> --backend ollama|openai|bedrock|anthropic
- synthkit render <spec.json> -o <dir> --backend stub|ollama|openai|...
- synthkit table-lint <spec.json> — probe-based promise check
- synthkit campaign-compile --goal predict --spec X --bars auroc=0.6,gap_max=0.3 -o DIR
- synthkit campaign-run DIR [--llm --llm-backend openai --samples N --llm-extra-system @file]
- synthkit showdown DIR --solver vendor_model:predict --baseline autosolver_hybrid
- synthkit trials DIR — append-only record
- synthkit gui — the bench at http://127.0.0.1:8377/
- python scripts/run_all_smokes.py — 324-check certification
- python scripts/build_system_map.py — atlas (drift-refusing)
- python scripts/build_presenter.py — presenter companion

## The Windows sync ritual (locked-down ThinkPad)
activate verbatim venv -> cd dev -> rmdir /s /q synthkit ->
curl -L -o synthkit.zip api.github.com/repos/farmingfarmer/synthkit/zipball/main
-> **dir synthkit.zip (MUST be MB; ~106 bytes = repo private)**
-> tar -xf -> ren farmingfarmer-synthkit-* synthkit ->
pip install -e . (mandatory after every rmdir) -> synthkit
version fingerprint match -> run_all_smokes 324 green.
Repo public-flip or Bearer-token header required; flip back.

## Windows local LLM (proven path)
llama.cpp win-cpu-x64 zip via GitHub API one-liner (llamafile's
APE format is AV-blocked); llamacpp\llama-server.exe -m
llama3.2-3b.gguf --port 8080; synthkit backend `openai`
defaults to 127.0.0.1:8080/v1. Terminal topology: T1=model,
T2=bench (8377), T3=working terminal.

## Backend/cost ladder (order-of-magnitude, verify AWS pricing)
- stub: $0, instant, deterministic — default and demo-safe.
- local 3B (llama.cpp, ThinkPad): $0, offline; 8-doc render
  ~90s; extraction-protocol UNRELIABLE (17/24 malformed).
- local 24B (Ollama, Mac): $0; 24-doc render 2–5 min; 0/1,662
  malformed in the vendor chair.
- Bedrock open weights: HIPAA-eligible, fractions of a cent /
  1K tokens; the 1,662-call study ≈ single-digit dollars;
  BedrockBackend smoked, activation = IAM request.
- Anthropic/frontier cloud: never required; tens of dollars for
  study-scale workloads.
- Enterprise seat estimate: cents to low single-digit $/month
  for typical analyst AI usage on open weights; core system $0.

## Demo assets
docs/demo_script.md (six-act runbook, canonical numbers),
docs/demo_prompt.md (the station-01 English paragraph),
readmit_demo.json + GUI preset #1, vendor_model.py (VendorCo),
docs/vendor_integration.md (three real-vendor routes),
docs/DEMO_LLM_WINDOWS.md, docs/system_map.html (atlas v2),
docs/presenter_companion.html (private).
