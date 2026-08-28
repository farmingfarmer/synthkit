# Windows port guide (locked-down work laptop)

The established sync ritual, adapted for synthkit, plus what to
expect. Written for cmd/PowerShell on a machine with a working
python venv at `dev\` and no admin rights.

## 1. Pull the repo (no git needed)

**WHICH REPOSITORY, and why this page does not name it.** There are
two. Code is authored on the development machine, PROVEN here against
a real extract, and only then published to the enterprise repository.
So the release repo is DOWNSTREAM of this test and is guaranteed not
to hold the code under test — pulling from it runs the previous
build, and on a machine with no git history a stale build looks
exactly like a current one. Pull from the STAGING repo; publish to
the release repo after this page comes out green.

The staging repo belongs to a personal account, and a tracked file
here reaches the enterprise mirror, so writing that handle into this
page is banned (`smoke_no_personal`). Set it yourself instead. Both
are private, so use a fine-grained read-only token.

```bat
cd %USERPROFILE%\dev
set SYNTHKIT_REPO=OWNER/REPO
set SYNTHKIT_TOKEN=YOUR_READONLY_TOKEN
set SYNTHKIT_DIR=%SYNTHKIT_REPO:/=-%
echo pulling %SYNTHKIT_REPO% into %SYNTHKIT_DIR%
for /d %i in (%SYNTHKIT_DIR%-*) do rmdir /s /q "%i"
rmdir /s /q synthkit
del /q synthkit.zip 2>nul
curl -L -H "Authorization: Bearer %SYNTHKIT_TOKEN%" -o synthkit.zip https://api.github.com/repos/%SYNTHKIT_REPO%/zipball/main
dir synthkit.zip
```

**Read that `echo` line before anything else.** If it prints
`%SYNTHKIT_REPO%` back at you literally rather than an owner and a
repo, the variable never got set and every command after it is
addressing nothing — the same failure an unexpanded `%USERPROFILE%`
caused when it wrote real-data output into the repo. That echo is
there because this terminal has mangled pasted commands three times,
and a flag that never arrives is otherwise indistinguishable from
broken code.

The leftover-folder cleanup is a `for` loop now. `rmdir` accepts no
wildcard, so the previous `rmdir /s /q <pattern>-*` could never have
matched anything and its `2>nul` hid that it was failing every time —
which is exactly the orphan-folder problem described below. This has
not been run on Windows from here; if the loop misbehaves, deleting
the stray folders by hand does the same job.

**Stop and read that size before going further.** A few hundred
kilobytes or more means the download worked. Around 106 bytes
means the token was rejected and what landed is an error page, not
an archive — everything after this point will then fail in ways
that look like something else entirely.

```bat
tar -xf synthkit.zip
for /d %i in (%SYNTHKIT_DIR%-*) do ren "%i" synthkit
```

**THE COMMIT NOW RIDES INSIDE THE ZIP, so there is nothing to
remember.** GitHub builds a zipball by running `git archive`, and
`git archive` substitutes `$Format:` placeholders in files marked
`export-subst`. `BUILD_SHA.txt` is such a file, so the download
already knows which commit it is. Every run echoes it on its first
line and records it in `provenance.json`.

**Read that first line.** It should say `build: <sha> (from archive)`.
If it says `build: UNKNOWN`, this copy did not come from a GitHub
archive and nothing it reports can be tied to a known tree — which is
the failure this exists to prevent, after several rounds of results
were read against fixes the running code did not contain.

This used to be a hand step, and it was skipped on three runs in a
row. If you ever need it — a copy assembled some other way — the
extracted folder's name carries the sha, and these two lines put it
where the code looks. They must come BEFORE the rename, which
destroys that name:

```bat
for /d %i in (%SYNTHKIT_DIR%-*) do (echo %i)>build_id.txt
move /y build_id.txt synthkit\BUILD_ID.txt
```

The `(echo %i)` is parenthesised deliberately. Written bare as
`echo %i>build_id.txt`, cmd reads a trailing digit as a redirection
handle, and a sha ending in a digit would silently lose its last
character — a build id that is quietly wrong is worse than none.

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

**`pip install -e .` now covers the dependencies, and the
requirements line is kept only because this page names it.**
`pyproject.toml` declares numpy, pandas, scikit-learn and scipy, so
one install is enough. It did not used to: they lived in
`requirements.txt` alone, and skipping that line left seven suites —
every one covering the discover/blueprint/generate path — failing on
their import line and reading as broken code. Running both commands
is harmless.

**Python 3.10 or newer.** Those four packages all require it. The
package used to claim 3.8, which did not fail cleanly on an older
interpreter — pip resolved whatever ancient versions still supported
it, and a silently different pandas produces a wrong number rather
than an error. Check with `python --version` before installing.

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

Expect: 67 suites, **1777** checks, ALL GREEN.

**That number is for THIS machine, and it is nine lower than the
development machine's on purpose.** Nine checks in `smoke_buildid`
run `git archive` to prove the commit really is substituted into the
download, and there is no git here to run it with - so they report
SKIPPED and the total comes out at 1777 against 1786 on a
checkout.
Both were measured, the second by extracting an archive and running
the whole net inside it. A total that does not match this page is
worth stopping for. It is no longer quick -
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

**How every column was read, in seconds.** Run this FIRST on a
dataset nobody has opened. Every silent fault this tool has had was a
column read as the wrong type, and this is the listing that catches
them — a date read as 200 categories, a currency column at 85%
sentinel, a clock at 62%. It does no discovery and writes no data.

```bat
synthkit types --src C:\FULL\PATH\TO\EXTRACT.csv --out C:\FULL\PATH\YOU\CHOOSE --group-by person_id
```

A cheap first pass, to catch a wrong `--src` or the wrong id column
in a minute instead of an hour:

```bat
synthkit fit --src C:\FULL\PATH\TO\EXTRACT.csv --out C:\FULL\PATH\YOU\CHOOSE --group-by person_id --max-rows 5000
```

Then the full run:

```bat
synthkit fit --src C:\FULL\PATH\TO\EXTRACT.csv --out C:\FULL\PATH\YOU\CHOOSE --group-by person_id --lags --generate
```

To generate a TUNED version — a column moved, widened, made sparser,
or a different number of patients — add `--dial`. What you asked for
and what actually arrived are both written into `findings.txt`,
because a dial can be capped by the k-anonymous bound or swapped back
by constraint repair, and a silent difference is the failure this
tool exists to refuse.

```bat
synthkit fit --src C:\FULL\PATH\TO\EXTRACT.csv --out C:\FULL\PATH\YOU\CHOOSE --group-by person_id --generate --dial patients.count=500
```

**Four other flags worth knowing before you start.**

`--time-col COLUMN` names the visit-ordering column. It is detected
if omitted, and the run prints which one it chose — read that line.
Every temporal statistic is measured on it.

`--long CONCEPT=VALUE` reshapes a one-row-per-measurement extract to
one column per concept. **The run detects this shape and tells you**
before any discovery happens; if it does, the numbers from a run
without it describe a column that is a mixture of unrelated
quantities and mean very little.

`--enforce-constraints` repairs orderings the source never broke,
such as a visit ending before it began. Off by default because it
changes the output. The constraints are REPORTED either way, so read
them first and turn this on deliberately.

`--refine-sweeps 0` restores the older cycle handling exactly. The
sweeps that re-apply trimmed parents are new and have NOT been
measured on a real extract — if a column that sits in a cycle looks
wrong, this is the first thing to try, and the difference between the
two runs is the evidence.

`synthkit dials C:\FULL\PATH\YOU\CHOOSE\blueprint.json` lists what
that blueprint can be tuned on, so nobody has to open the JSON to
find out.

The older form still works and does the same thing, in case this
page is newer than the copy you pulled:

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
