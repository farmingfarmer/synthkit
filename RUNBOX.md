# RUNBOX — what to type in each box, in order

Updated 2026-09-29. You have the bench open. Do these in order.
Every box below is named exactly as it appears on screen, with
the value to put in it directly underneath.

Paths shown as OWNER/REPO or with %USERPROFILE% are placeholders
here; your local RUNBOX.html on the Desktop has the real ones
typed out, ready to copy. Browser boxes do NOT expand
%USERPROFILE%, so type the full path.

---

## STEP 0 — do you have the right build?

**If you already tried Generate and got a STOPPED message about
`Unable to allocate ... GiB`, that was a real bug and it is
fixed.** Set columns were being one-hot encoded with thousands
of combination strings apiece - 44 columns became 4,326 encoded
ones. Pull first (APPENDIX at the bottom), then start here.


Look at the LEFT RAIL of the bench. Under the heading
`SEE THE FIDELITY` you should see three tiles:

    VIEW   Dashboard
    GEN    Neural          <- this one
    DUEL   Duel

**If `GEN  Neural` is NOT there, stop.** You are on an older
build and nothing below will work. Go to the APPENDIX at the
bottom, pull, reinstall, relaunch, then come back here.

---

## STEP 1 — click the tile

Click **`GEN  Neural`** in the left rail. The heading at the top
should change to `Neural — the engine, on its own`.

---

## STEP 2 — fill the four boxes

The panel is titled **Point at the data**. Four boxes, top to
bottom.

**Box 1 of 4, labelled `source CSV path`**

```
%USERPROFILE%\dev\tidy_visits.csv
```

**Box 2 of 4, labelled `output directory`**

```
%USERPROFILE%\dev\neural_out
```

Use a NEW folder name. If you reuse an old one you will be
looking at output from a previous build.

**Box 3 of 4, labelled `patient / entity column`**

```
person_id
```

**Box 4 of 4, labelled `seed`**

```
0
```

---

## STEP 3 — press Generate

Press the dark button labelled **`Generate`**.

A loader appears with a counter. It is doing two things: training
the autoencoder and drawing records from it, then measuring what
your real data contains so the other stations can read the
result.

**Expect 10 to 25 minutes on the full extract.** The counter
keeps moving. When it finishes the box underneath says:

    written in NNNs. Now press Review the output.

**If it says STOPPED**, the message names what went wrong; send
it to me.

---

## STEP 4 — press Review the output

Press the pale button labelled **`Review the output`**.

Another loader. **Expect 5 to 15 minutes.** When it finishes a
page appears in a frame directly below the buttons, and the box
says:

    measured in NNNs. Gray is the real data, gold is the engine.

---

## STEP 5 — read the page, top to bottom

Scroll the frame. In order you will see:

1. **Four headline cards** - relationships pointing the same
   way, at the same strength, how many came out BACKWARDS
   (this one wants to be zero), and the row count.
2. **Does a patient look like a person?** - per column, how much
   of it belongs to the person rather than the visit, real
   against generated. This is where patient structure is judged.
3. **The whole table at once** - how many columns a model
   CANNOT tell apart from real, the per-column list of the ones
   it can, whether each column is as learnable, whether the
   synthetic covers the real space, and three-column effects.
4. **The combination effects** - effects that only exist when
   two columns act together.
5. **Every column, drawn twice** - press and hold any chart to
   pull real and synthetic apart.
6. **What this page does not say** - the safety statement. Read
   it once.

---

## STEP 6 — the other stations, same directory

The Generate step wrote the files the rest of the bench reads,
so these now work on the SAME output directory:

- **`03 Verdict`** - click it, put your output directory in its
  box, press `Open the run in the output directory from Step 1`.
  Gives the eight-criteria gate.
- **`VIEW Dashboard`** - source CSV in the first box, the same
  output directory in the second, `person_id` in the third.
- **`DUEL Duel`** - only if you also want the side-by-side
  against the rules engine; it needs a finished rules-engine run
  as well.

`ALT Learn` is the first engine entirely and does not apply.

**Send back:** the headline cards, the patient table, and the
whole-table section.

---

# APPENDIX — pull a fresh build (only if STEP 0 sent you here)

One command per line. Close the bench first.

```bat
cd %USERPROFILE%\dev
```

```bat
for /d %i in (OWNER-REPO-*) do rmdir /s /q "%i"
```

```bat
rmdir /s /q synthkit
```

```bat
del /q synthkit.zip 2>nul
```

Make the repo public for the next line only, then private again
once `dir` shows a real size.

```bat
curl -L -o synthkit.zip https://api.github.com/repos/OWNER/REPO/zipball/main
```

```bat
dir synthkit.zip
```

**STOP AND READ.** Hundreds of KB means it worked. About 106
bytes means what landed is an error page, not an archive.

```bat
tar -xf synthkit.zip
```

```bat
for /d %i in (OWNER-REPO-*) do ren "%i" synthkit
```

```bat
cd %USERPROFILE%\dev\synthkit
```

```bat
pip install -e .
```

```bat
python scripts\run_all_smokes.py
```

**Expect: 69 suites, 1965 checks, ALL GREEN.**

```bat
python -m synthkit.cli gui
```

Then go back to STEP 0.
