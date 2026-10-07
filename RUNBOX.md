# RUNBOX — pull, push to the enterprise repo, build the kits

Updated 2026-10-02. Three jobs, in order, every command typed on
one line. `OWNER/REPO` and `<KECK_REPO_URL>` are placeholders in
this tracked copy. RUNBOX.html (tracked beside this file) is the
same runbook with copy buttons: open it in a browser, type the
owner/name and the enterprise URL into its two boxes once, and
every command fills itself in - the values live in the browser,
never in the file. `%USERPROFILE%` is the only
variable that survives on this terminal — everything else is
typed out in full.

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
cd %USERPROFILE%\dev\synthkit
pip install -e .
python scripts\run_all_smokes.py
```

**Expect: 69 suites, ALL GREEN** (the check count grows with
every build; ALL GREEN is the verdict — the zipball count for
this build is in docs/WINDOWS.md).

---

## STEP 1 — push to the enterprise repo

Needs: git on this machine, and credentials for the team repo.
The push is a SINGLE CLEAN COMMIT of a filtered tree — never our
history. The filtering is done by the bundle script, which
applies the one exclusion list, scrubs the two in-flight items,
writes the lab-notebook stub, and REFUSES to finish unless its
own deep scan prints `self-scan: CLEAN`. This exact filtered
tree was verified on the development side: scan CLEAN, and its
own net run came back ALL GREEN, 68 suites, 2025 checks.

```
cd %USERPROFILE%\dev\synthkit
rmdir /s /q %USERPROFILE%\dev\keck_stage 2>nul
python scripts\make_team_bundle.py -o %USERPROFILE%\dev\keck_stage --wheels skip
```

**The last line of that command must read `self-scan: CLEAN`.**
If it prints DIRTY lines instead, STOP and report them — the
script will have refused to finish, which is the design. Then:

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

**Verify from a second clone:**

```
cd %USERPROFILE%\dev
git clone <KECK_REPO_URL> keck_check
cd keck_check
pip install -e .
python scripts\run_all_smokes.py
```

Expect 68 suites, ALL GREEN (one suite fewer than our checkout:
the personal-scan suite stays home by design; the buildid suite
reports nine checks SKIPPED off a non-git tree, expected and
stated in its own output).

---

## STEP 2 — build and send the team bundle

Needs: network for the dependency wheels (one pip download).
From the pulled checkout:

```
cd %USERPROFILE%\dev\synthkit
python scripts\make_team_bundle.py -o %USERPROFILE%\Desktop\synthkit_team_bundle --wheels win
```

The build REFUSES to finish unless its own deep scan prints
`self-scan: CLEAN` — if it prints DIRTY lines instead, stop and
report them. Then zip and send:

```
cd %USERPROFILE%\Desktop
tar -a -c -f synthkit_team_bundle.zip synthkit_team_bundle
```

Send `synthkit_team_bundle.zip` through the team channel. What a
teammate gets, needing only Python 3.10+: the enterprise-clean
source, offline Windows wheels (`pip install --no-index` — no
network needed on their machine), the test kit with its printed
answer key, INSTALL.md one command per line, and the honest AI
page (the measure route needs no language model; the hospital
cloud path is named before any local install; Ollama steps
included for machines where it is approved, with the
fully-offline model-copy route).

---

## If anything surprises you

Copy the exact terminal output and send it back. Every refusal
in these tools states its reason in a sentence; a traceback or a
silent difference is a finding, not an inconvenience.
