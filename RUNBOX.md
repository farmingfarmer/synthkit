# RUNBOX — from scratch, in order: pull, verify, push, kits

Updated 2026-10-07. Four steps. Each step ends in a CHECKPOINT
that must pass before the next step starts — every failure this
runbook has ever seen was a later step running on an earlier
step's stale output. If a checkpoint fails, the problem is
upstream of where you are standing.

`OWNER/REPO` and `<KECK_REPO_URL>` are placeholders in this
tracked copy. RUNBOX.html (tracked beside this file) is the same
runbook with copy buttons: open it in a browser, type the
owner/name and the enterprise URL into its two boxes once, and
every command fills itself in — the values live in the browser,
never in the file. `%USERPROFILE%` is the only variable that
survives on this terminal — everything else is typed out in
full, one command per line.

---

## STEP 0 — pull the fresh build

Close the bench first. Then, in a new terminal:

```
cd %USERPROFILE%\dev
for /d %i in (OWNER-REPO-*) do rmdir /s /q "%i"
rmdir /s /q synthkit
del /q synthkit.zip 2>nul
```

Make the repo public for the next line only, then private again
once `dir` shows a real size:

```
curl -L -o synthkit.zip https://api.github.com/repos/OWNER/REPO/zipball/main
dir synthkit.zip
```

**STOP AND READ.** Hundreds of KB means it worked; about 106
bytes means an error page landed, not an archive.

```
tar -xf synthkit.zip
for /d %i in (OWNER-REPO-*) do ren "%i" synthkit
```

**CHECKPOINT — which commit are you holding?**

```
type %USERPROFILE%\dev\synthkit\BUILD_SHA.txt
```

Compare the first seven characters against the newest commit on
the repo page. If they differ, the pull brought an old build and
NOTHING BELOW is worth running — a whole day was once spent
rebuilding bundles from a stale extract that one `type` would
have exposed in two seconds.

---

## STEP 1 — install and verify the extract

```
cd %USERPROFILE%\dev\synthkit
pip install -e .
python scripts\run_all_smokes.py
```

**CHECKPOINT: 69 suites, ALL GREEN.** The words are the verdict;
the exact zipball check count for this build is pinned in
docs/WINDOWS.md. A FAILED suite stops everything — copy the
lines that start with FAIL and send them back.

---

## STEP 2 — push to the enterprise repo

Needs: git on this machine, and credentials for the team repo.
The push is a SINGLE CLEAN COMMIT of a filtered tree — never our
history. The filtering is done by the bundle script, which
applies the one exclusion list, scrubs the in-flight items,
resets the build stamp to its placeholder, writes the
lab-notebook stub, and REFUSES to finish unless its own deep
scan prints `self-scan: CLEAN`.

```
cd %USERPROFILE%\dev\synthkit
rmdir /s /q %USERPROFILE%\dev\keck_stage 2>nul
python scripts\make_team_bundle.py -o %USERPROFILE%\dev\keck_stage --wheels skip
```

**CHECKPOINT, two lines.** The build must end `self-scan: CLEAN`
(DIRTY lines mean it refused — stop and report them). Then:

```
type %USERPROFILE%\dev\keck_stage\synthkit_src\BUILD_SHA.txt
```

This must print `$Format:%H$` — the raw placeholder. A 40-hex
string here would poison the team repo's identity (its own
archives would name a commit that is not in it); that happened
once, the verify clone caught it, and this line is why it
cannot happen silently again.

Now the push:

```
cd %USERPROFILE%\dev\keck_stage\synthkit_src
git init -b main
git remote add keck <KECK_REPO_URL>
git add -A
git commit -m "synthkit: two engines, eight-criterion gate, team test kits"
git push keck main --force
```

`--force` is correct only because this repo's convention is
whole-version replacement — confirm nobody commits to it
directly before using it.

**CHECKPOINT — verify from a second clone.** Always delete the
old clone first; re-running inside a stale `keck_check` verifies
the previous push, not this one:

```
cd %USERPROFILE%\dev
rmdir /s /q keck_check 2>nul
git clone <KECK_REPO_URL> keck_check
cd keck_check
pip install -e .
python scripts\run_all_smokes.py
```

Expect **68 suites, ALL GREEN** (one suite fewer than our
checkout: the personal-scan suite stays home by design). This
clone has found a real defect on every shape it was first run
against — it is not ceremony.

---

## STEP 3 — build, rehearse and send the team bundle

Needs: network for the dependency wheels (one pip download).
From the pulled checkout:

```
cd %USERPROFILE%\dev\synthkit
rmdir /s /q %USERPROFILE%\Desktop\synthkit_team_bundle 2>nul
del /q %USERPROFILE%\Desktop\synthkit_team_bundle.zip 2>nul
python scripts\make_team_bundle.py -o %USERPROFILE%\Desktop\synthkit_team_bundle --wheels win
```

**CHECKPOINT.** The build must end `self-scan: CLEAN`, and:

```
dir %USERPROFILE%\Desktop\synthkit_team_bundle
```

must show `START_SYNTHKIT.bat` and `RUN_CHECKS.bat` at the top
level, beside INSTALL.md, kit, synthkit_src and wheels. Missing
launchers mean the extract is stale — back to STEP 0. Then zip:

```
cd %USERPROFILE%\Desktop
tar -a -c -f synthkit_team_bundle.zip synthkit_team_bundle
```

**CHECKPOINT — be the first teammate.** Unzip a copy somewhere
else, double-click `START_SYNTHKIT.bat` (one-time setup, then
the bench opens in the browser), then double-click
`RUN_CHECKS.bat` and expect ALL GREEN. Only a bundle that passed
this rehearsal gets sent. What a teammate then needs is one
sentence: unzip, double-click START_SYNTHKIT.bat — the offline
wheels mean their machine needs no network, only Python 3.12 or
3.14.

---

## STEP 4 — housekeeping

`keck_stage` and `keck_check` are disposable by design — the
runbook recreates both from scratch every time:

```
rmdir /s /q %USERPROFILE%\dev\keck_stage 2>nul
rmdir /s /q %USERPROFILE%\dev\keck_check 2>nul
```

---

## If anything surprises you

Copy the exact terminal output (or photograph the window) and
send it back. Every refusal in these tools states its reason in
a sentence; a traceback or a silent difference is a finding, not
an inconvenience.
