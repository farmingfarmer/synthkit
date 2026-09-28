# RUNBOX — recover the data machine after a forced restart

Updated 2026-09-28. Fresh pull, install, verify, then the bench.
One command per line - this terminal has mangled wrapped pastes
three times. Two STOP-AND-READ points below; they exist because
a silently-wrong pull looks exactly like working code.

## 1. Set up (all local, repo can stay private)

**WRITE THE OWNER/REPO OUT IN FULL - do not use a variable.**
The `set SYNTHKIT_REPO=...` plus `%SYNTHKIT_REPO%` form in
WINDOWS.md has never once worked on the data machine: the
substitution does not survive that terminal, and what reaches
curl is a URL addressing nothing. Everywhere below that reads
`OWNER/REPO` or `OWNER-REPO`, type the real thing. The only
variable that has ever survived there is `%USERPROFILE%`.

```bat
cd %USERPROFILE%\dev
```

## 2. Clear the old copy FIRST (never after the extract)

```bat
for /d %i in (OWNER-REPO-*) do rmdir /s /q "%i"
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
curl -L -o synthkit.zip https://api.github.com/repos/OWNER/REPO/zipball/main
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
for /d %i in (OWNER-REPO-*) do ren "%i" synthkit
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

**Expect: 69 suites, 1955 checks, ALL GREEN.** That is the
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

## 8. Run the neural engine on its own (the short path)

The **GEN / Neural** station in the left rail runs the engine
with no prior run and no second engine. Four boxes:

- source CSV path - the real extract
- output directory - a NEW folder of yours
- patient / entity column - `person_id`
- seed - `0`

Press **Generate**, wait for it to say the file is written, then
press **Review the output**. The review leads with "does a
patient look like a person", then the combination effects, then
every column drawn twice.

## 9. Or run both engines side by side (the comparison)

Everything below happens in the browser. The station is **DUEL**
in the left rail, under SEE THE FIDELITY, just beneath Dashboard.

Click DUEL, then fill the five boxes TOP TO BOTTOM. Browser
fields do not expand %USERPROFILE%, so type the expanded path -
your local RUNBOX.html carries them ready to copy.

1. **source CSV path** - the real extract
2. **blueprint run directory** - your finished full-extract fit
   (the one holding generated.csv); this is the OTHER engine,
   and the page needs it to compare against
3. **patient / entity column** - `person_id`
4. **latent output directory** - a NEW folder. Do not reuse an
   older one: anything written before 2026-09-25 holds
   patient-less output from the previous build, and the page
   would measure the old engine.
5. **latent seed** - `0`

Then, in order:

- Press **Generate with the latent engine (minutes)**. This is
  the autoencoder training and drawing on the full extract. The
  loader runs with elapsed seconds; it finishes by saying the
  draws are written.
- Press **Draw the duel**. Also a job - it mines the
  interactions out of the real data and measures both engines
  against them. Your last run drew in 198s.

**What the page gives you, top to bottom:** four headline cards
(each engine's kept-direction / kept-strength counts and its
BACKWARDS count), every shared column drawn three ways (gray
original, cardinal rules engine, gold neural engine - press and
hold to fan them apart), "Where each engine goes wrong" as a
grid, "Does a patient still look like a person?" - the new
section, per column, which is where the neural engine's patient
structure is judged - then "The combination effects", and a
closing section on what the page does NOT say about safety.

## Optional - driver attribution in the dashboard

```bat
pip install shap
```
