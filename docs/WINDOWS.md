# Windows port guide (locked-down work laptop)

The established sync ritual, adapted for synthkit, plus what to
expect. Written for cmd/PowerShell on a machine with a working
python venv at `dev\` and no admin rights.

## 1. Pull the repo (no git needed)

The repository is `git-datasciences-org-main/keck-synthkit` on
Keck's GitHub. It is private, so either use a fine-grained
read-only token or clone over HTTPS with your own credentials.

```bat
cd %USERPROFILE%\dev
rmdir /s /q synthkit
rmdir /s /q git-datasciences-org-main-keck-synthkit-* 2>nul
del /q synthkit.zip 2>nul
curl -L -H "Authorization: Bearer YOUR_READONLY_TOKEN" -o synthkit.zip https://api.github.com/repos/git-datasciences-org-main/keck-synthkit/zipball/main
dir synthkit.zip
```

**Stop and read that size before going further.** A few hundred
kilobytes or more means the download worked. Around 106 bytes
means the token was rejected and what landed is an error page, not
an archive — everything after this point will then fail in ways
that look like something else entirely.

```bat
tar -xf synthkit.zip
for /d %i in (git-datasciences-org-main-keck-synthkit-*) do ren "%i" synthkit
```

Two things about that rename, both learned the hard way. The
extracted folder carries a commit hash, so its name changes every
push and the wildcard is the only stable way to refer to it —
Windows `ren` will not accept a wildcard on its own, hence the
loop. And the `rmdir` of leftover folders above matters: an
interrupted sync leaves an orphan with a different hash, the
wildcard then matches two folders, and the second rename fails
with "a duplicate file name exists".

Inside a `.bat` file the loop variable doubles to `%%i`.

If `tar` reports "Can't create ... Invalid argument" on paths
containing `.venv`, a virtual environment has been committed to
the repository by mistake: Windows cannot create the symlinks
inside it. Fix that at the source rather than here.

## 2. Install + verify identity

```bat
cd %USERPROFILE%\dev\synthkit
pip install -r requirements.txt
pip install -e .
synthkit version
```

**The requirements line is not optional and `pip install -e .`
does not cover it.** `pyproject.toml` declares no dependencies, so
without it numpy, pandas, scikit-learn and scipy are missing and
seven suites — every one covering the discover/blueprint/generate
path — fail on their import line and read as broken code.

The FINGERPRINT must MATCH the development machine's `synthkit
version` at the same commit — that one line proves the zip landed
byte-perfect. Compare the 8 hex characters only: it is a sha256 over
the package's bytes, whereas the `built` date beside it is a file
timestamp and will differ on every machine. A different date is
expected and means nothing.
If `synthkit` is not found, the venv's Scripts dir is not on
PATH; `python -m synthkit.cli version` works regardless
(module entry: use `python -c "from synthkit.cli import main;
main(['version'])"` if -m is not wired).

## 3. Run the full net

```bat
python scripts\run_all_smokes.py
```

Expect: 50 suites, 1336 checks, ALL GREEN. It is no longer quick -
about 9 minutes on the development machine and several times that
here, with the long pauses at `smoke_generate` and `smoke_shapes`.
Each suite prints `running` before it starts and `PASS` when it
finishes, so a pause names the suite it is waiting on.

If you are on an older copy that prints nothing at all until it
exits: that is Python block-buffering stdout when it does not see an
interactive console, not a hang. `python -u scripts\run_all_smokes.py`
forces it out line by line. Every
suite is offline and LLM-free by design (LLM paths are tested via
canned backends), so no ollama is required for a green net.

## 4. Run the discovery pipeline on a real extract

The output directory is YOURS to choose and is created if missing.
Put it OUTSIDE the repository — it holds real-data-derived output,
and an unexpanded `%USERPROFILE%` has previously written such output
into the working tree. Use a full absolute path and check it.

A cheap first pass, to catch a wrong `--src` or the wrong id column
in a minute instead of an hour:

```bat
python -u scripts\run_discovery.py --src C:\FULL\PATH\TO\EXTRACT.csv --out C:\FULL\PATH\YOU\CHOOSE --group-by person_id --max-rows 5000
```

Then the full run:

```bat
python -u scripts\run_discovery.py --src C:\FULL\PATH\TO\EXTRACT.csv --out C:\FULL\PATH\YOU\CHOOSE --group-by person_id --lags --generate
```

Each of those is ONE line. A wrapped paste has three times put a flag
inside a filename or dropped it entirely, and the log looked normal
either way — which is why the run echoes an `invocation:` line first.
Read it back and confirm every flag you typed is in it.

Watch for `read as dates:` naming the date columns and the format
they were read under, and for any `WARNING ... became MISSING`, which
means values did not parse and those rows are now blank.

Written into the directory you chose: `catalogue.json`,
`blueprint.json`, `findings.txt`, and with `--generate` also
`generated.csv` and `fidelity.json`. Read `findings.txt` first.

## 5. What works on this machine

- Everything offline: tables, campaigns (built-in solvers),
  lint, relational, showdowns, trials, the bench (`synthkit
  gui`), both notebooks, the system atlas
  (`docs\system_map.html` opens in any browser from disk).
- NOT available without an LLM: live compile from English
  (`--backend ollama`), live document rendering, the LLM-vendor
  chair. On this machine those routes go through Bedrock once
  IAM is sorted (`synthkit.bedrock.BedrockBackend`), or stay
  development machine only.

## 6. The demo payloads, in order of ceremony

1. `notebooks\synthkit_demo.ipynb` — upload to SageMaker or any
   Jupyter; zero dependencies; runs top to bottom.
2. `docs\system_map.html` — double-click; the architecture tour.
3. `synthkit gui` — the live bench (binds 127.0.0.1 only).
4. `docs\first_vendor_trial.md` + `docs\keck_onboarding.md` —
   the study and the script.

## 7. Windows-specific notes (already handled, listed for trust)

- All file I/O is explicit UTF-8 (audited by ast; four smoke
  stragglers fixed pre-port). cp1252 defaults cannot bite.
- CLI output is console-safe: middle-dots and em-dashes degrade
  to replacements on cp437/cp1252 consoles instead of crashing
  (`_console_safe` in cli.py; verified under
  PYTHONIOENCODING=cp437:strict).
- `.gitattributes` pins LF so the zipball is byte-identical to
  the development tree (the fingerprint match depends on it).
- The smoke net is a Python script, not a shell loop — no POSIX shell
  assumed anywhere in the run path.
- If console output still looks garbled on legacy cmd:
  `set PYTHONUTF8=1` before running, or use Windows Terminal.
