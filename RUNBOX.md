# RUNBOX — recover the data machine after a forced restart

Updated 2026-09-28. Fresh pull, install, verify, then the bench.
One command per line - this terminal has mangled wrapped pastes
three times. Two STOP-AND-READ points below; they exist because
a silently-wrong pull looks exactly like working code.

## 1. Set up (all local, repo can stay private)

```bat
cd %USERPROFILE%\dev
```

```bat
set SYNTHKIT_REPO=OWNER/REPO
```

```bat
set SYNTHKIT_DIR=%SYNTHKIT_REPO:/=-%
```

```bat
echo pulling %SYNTHKIT_REPO% into %SYNTHKIT_DIR%
```

**STOP AND READ.** That must print an owner and a repo. If it
echoes `%SYNTHKIT_REPO%` back literally, the variable never got
set and every command after it addresses nothing.

## 2. Clear the old copy FIRST (never after the extract)

```bat
for /d %i in (%SYNTHKIT_DIR%-*) do rmdir /s /q "%i"
```

```bat
rmdir /s /q synthkit
```

```bat
del /q synthkit.zip 2>nul
```

## 3. Download - the repo must be PUBLIC for this one line

Flip it public now, run the next command, then flip it back as
soon as `dir` shows a real size.

```bat
curl -L -o synthkit.zip https://api.github.com/repos/%SYNTHKIT_REPO%/zipball/main
```

```bat
dir synthkit.zip
```

**STOP AND READ.** A few hundred KB or more means it worked.
About 106 bytes means what landed is an error page, not an
archive - the repo was still private, or the name is wrong.
Everything after this point would then fail in ways that look
like something else entirely. Fix it here.

Repo can go back to private now.

## 4. Extract and rename

```bat
tar -xf synthkit.zip
```

```bat
for /d %i in (%SYNTHKIT_DIR%-*) do ren "%i" synthkit
```

## 5. Install, then verify identity

```bat
cd %USERPROFILE%\dev\synthkit
```

```bat
pip install -e .
```

```bat
python -m synthkit.cli version
```

Run version AFTER the install - running it before shows the old
build id and reads as a failed pull.

## 6. Prove the tree

```bat
python scripts\run_all_smokes.py
```

**Expect: 69 suites, 1950 checks, ALL GREEN.** That is the
zipball number; a checkout reads 1959 because nine build-id
checks need git. Do not read 1950 as nine failures.

## 7. The bench

```bat
python -m synthkit.cli gui
```

Serves at http://127.0.0.1:8377 and opens the browser. Confirm
the build id in the wordmark matches step 5.

New since your last session: the bench scrolls in layers now
(titles and instructions drift at different rates past the form,
which stays put), and the DUEL station runs both engines with
the neural side producing PATIENTS rather than unlinked rows.

## Optional - driver attribution in the dashboard

```bat
pip install shap
```
