# Windows port guide (locked-down work laptop)

The established sync ritual, adapted for synthkit, plus what to
expect. Written for cmd/PowerShell on a machine with a working
python venv at `dev\` and no admin rights.

## 1. Pull the repo (no git needed)

Private repo, so the fine-grained read-only token header applies:

```bat
cd %USERPROFILE%\dev
curl -L -H "Authorization: Bearer YOUR_READONLY_TOKEN" -o synthkit.zip https://api.github.com/repos/farmingfarmer/synthkit/zipball/main
tar -xf synthkit.zip
ren farmingfarmer-synthkit-* synthkit
```

(Re-syncing later: delete the old folder first, or extract beside
it and swap.)

## 2. Install + verify identity

```bat
cd %USERPROFILE%\dev\synthkit
pip install -e .
synthkit version
```

The fingerprint must MATCH the Mac's `synthkit version` at the
same commit — that one line proves the zip landed byte-perfect.
If `synthkit` is not found, the venv's Scripts dir is not on
PATH; `python -m synthkit.cli version` works regardless
(module entry: use `python -c "from synthkit.cli import main;
main(['version'])"` if -m is not wired).

## 3. Run the full net

```bat
python scripts\run_all_smokes.py
```

Expect: 14 suites, 270 checks, ALL GREEN, roughly 15-30s. Every
suite is offline and LLM-free by design (LLM paths are tested via
canned backends), so no ollama is required for a green net.

## 4. What works on this machine

- Everything offline: tables, campaigns (built-in solvers),
  lint, relational, showdowns, trials, the bench (`synthkit
  gui`), both notebooks, the system atlas
  (`docs\system_map.html` opens in any browser from disk).
- NOT available without an LLM: live compile from English
  (`--backend ollama`), live document rendering, the LLM-vendor
  chair. On this machine those routes go through Bedrock once
  IAM is sorted (`synthkit.bedrock.BedrockBackend`), or stay
  Mac-only.

## 5. The demo payloads, in order of ceremony

1. `notebooks\synthkit_demo.ipynb` — upload to SageMaker or any
   Jupyter; zero dependencies; runs top to bottom.
2. `docs\system_map.html` — double-click; the architecture tour.
3. `synthkit gui` — the live bench (binds 127.0.0.1 only).
4. `docs\first_vendor_trial.md` + `docs\keck_onboarding.md` —
   the study and the script.

## 6. Windows-specific notes (already handled, listed for trust)

- All file I/O is explicit UTF-8 (audited by ast; four smoke
  stragglers fixed pre-port). cp1252 defaults cannot bite.
- CLI output is console-safe: middle-dots and em-dashes degrade
  to replacements on cp437/cp1252 consoles instead of crashing
  (`_console_safe` in cli.py; verified under
  PYTHONIOENCODING=cp437:strict).
- `.gitattributes` pins LF so the zipball is byte-identical to
  the Mac tree (the fingerprint match depends on it).
- The smoke net is a Python script, not a shell loop — no zsh
  assumed anywhere in the run path.
- If console output still looks garbled on legacy cmd:
  `set PYTHONUTF8=1` before running, or use Windows Terminal.
