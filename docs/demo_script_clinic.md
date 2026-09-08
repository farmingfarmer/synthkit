# The live demo, step by step: clinic data, measure route

Every DO is a literal action. Every SAY is a suggested line - say it
in your own words, but say the honest ones (marked **keep**) in some
form, because the demo's credibility is built on them.

Timings assume the rehearsed fit time on the demo machine. Rehearse
once, note the real number, and budget that.

---

## Before anyone arrives (15 minutes, once)

1. DO: pull the current build and run the net. It must say ALL GREEN
   with the count `docs/WINDOWS.md` names for a zipball. If it does
   not, stop - do not demo on a red net.
2. DO: `python scripts\make_demo_clinic.py -o %USERPROFILE%\dev\clinic`
3. DO: run the full fit once, as the understudy:
   `synthkit fit --src %USERPROFILE%\dev\clinic\demo_clinic.csv --out %USERPROFILE%\dev\clinic_backup --group-by person_id --generate`
   Note the wall time. If anything fails live, the Verdict station
   can open `clinic_backup` instead and the demo continues.
4. DO: delete the live run directory if it exists from rehearsal:
   `rmdir /s /q %USERPROFILE%\dev\clinic_live`
5. DO: `synthkit gui` - confirm the build id in the top-left corner
   matches the pull. Click through Source -> Fit -> Verdict once.
6. DO: close every other window. The terminal and the bench are the
   whole show.

---

## Act 1 - make the "real" data in front of them (2 min)

DO: in the terminal:

    python scripts\make_demo_clinic.py -o %USERPROFILE%\dev\clinic

SEE: it prints the row count and then THE ANSWER KEY - nine planted
patterns.

SAY: "This is a fictional clinic dataset we generated just now -
about 260 patients, two thousand visits. For today it plays the role
of a customer's real file, because the real extract takes half an
hour to process and this takes about two.

**keep**: Everything I'm about to show runs identically on our real
800-patient hospital extract - I'll show you that scorecard at the
end, including the one number that still fails there.

The important part is on screen: the answer key. We planted nine
patterns - a straight-line effect, a threshold effect that a simple
correlation cannot see, one *medication* tied to a patient trait,
patients who take no medications at all on 30% of visits, three
deliberately extreme patients, and about 120 rare medications that
privacy must refuse to publish. The system has not seen this key.
The question for the next ten minutes: does it find what we hid, and
does it refuse what it must refuse."

---

## Act 2 - the bench, Source station (2 min)

DO: in the browser (`synthkit gui`), the left rail reads two routes.
Point at them.

SAY: "Two ways in. The top rail invents data from an English
description. The second rail - same step numbers, same colors -
measures data you already have. Today we take the measure route."

DO: click **01 Source** (second rail). Fill the three fields:

    source CSV path:    %USERPROFILE%\dev\clinic\demo_clinic.csv
    output directory:   %USERPROFILE%\dev\clinic_live
    patient column:     person_id     (already filled)

Leave "include lag features" as it is - it does not matter for this
dataset.

DO: click **Check the types (seconds)**.

SEE: every column listed with its type; `meds` as a SET of tokens;
lines saying how many patients stand behind each categorical level.

SAY: "Two seconds, and it has read every column - numbers, dates,
categories, and it recognized the medication list as a *list*, not a
blob of text. **keep**: It also says, per column, whether
anonymization will destroy it - before we spend a minute on
anything. Every silent failure this tool ever had started with a
column read as the wrong type, so this cheap check always runs
first."

---

## Act 3 - Fit station: the run, live (the fit time you rehearsed)

DO: click **02 Fit**. Click **Fit and generate**.

SEE: the log streams - reading, searching columns with a countdown,
building the blueprint, generating, comparing.

SAY (while it runs - this is your narration window):
"It is now doing four things. One: for every column, finding what
explains it - and every claim must hold on *held-out patients* the
model never saw, or it is discarded. Two: writing the contract - and
this is the privacy heart: **keep**: every number in that contract
describes at least ten patients. The most extreme value it will ever
publish is an *average of the ten most extreme people*. Those three
extreme patients we planted? Their values cannot appear. The 120
rare medications? Each is carried by one or two people, so the
system refuses to publish them - and tells us how many it refused.
Three: generating brand-new patients from the contract alone -
**keep**: generation never touches a record, only the contract.
Four: grading itself against the source."

If there is dead air: open the answer key printout in the terminal
and walk one pattern in detail - the threshold: "below stress 7 the
inflammation lab is flat; above 7 it jumps. A correlation coefficient
half-sees this. An effect curve draws it."

---

## Act 4 - the findings, read aloud (3 min)

SEE: when the log says done, scroll the tail of the log (or open
`clinic_live\findings.txt`).

DO: point at these lines, in this order. Each maps to a planted
pattern:

1. `heart_rate` under RELATIONSHIPS - "the straight-line effect:
   found."
2. `crp` - "the threshold: found - and not as a vague correlation;
   the effect curve carries the jump."
3. `meds__has__melatonin` beside `sleep_hours` - "**keep**: it found
   that one specific medication - melatonin - travels with short
   sleepers. Not the medication list as a blob: the individual
   token. That is the set machinery."
4. The meds line saying N token(s) below the k floor cannot be
   published (privacy, not a fault) - "the 120 rare medications:
   refused, counted, and the refusal is printed, not silent."
5. The line about the column that loses magnitude to the published
   bound (privacy, not a fault) - "our three extreme patients: the
   bound clipped them, on purpose, and the report says so - so
   nobody chases a 'bug' that is privacy working."
6. `set columns EMPTY at their source rate` - "30% of visits have no
   medications in the source; the generated file has the same 30%.
   Two weeks ago it did not - a continuous model cannot naturally
   produce 'exactly zero, 30% of the time' - and the fix came from
   this same instrumentation."
7. `near-deterministic identities hold` - "age equals 2026 minus
   birth year, on every generated row."

---

## Act 5 - Verdict station: the gate (2 min)

DO: click **03 Verdict**. Click **Open the run in the output
directory from Step 1**.

SEE: six rows, all PASS. "M0 MET on all criteria."

SAY: "**keep**: This gate is a script with an exit code, not a
slide - the same gate, same thresholds, runs on our real hospital
extract, where it currently reads five of six, and the report says
exactly which criterion fails and by how much. It passing here is
worth something *because* it is willing to fail there. A gate is a
floor, not a certificate - one dataset, one seed."

Also point at the footer: zero contradictions, zero disobeyed
declarations, and the build id - "every run is tied to the exact
code that produced it."

---

## Act 6 - the bridge into the exam (1 min)

DO: still on Verdict, click **Send to the exam (Step 4)**.

SEE: the spec card at the top fills; the panel lists what CROSSED
and what did NOT cross.

SAY: "The measured recipe just became an exam paper - the same
Campaign and Showdown machinery that grades vendor models now points
at data shaped like this clinic. **keep**: And read the second list:
it states what did *not* survive the crossing - effect shapes,
interactions. A recipe that silently lost its relationships would be
worse than one that admits it. Every hand-off in this system
announces its own limits.

What we do NOT do today is run the exam live - an exam needs an
answer column, and honestly deciding what to plant as the answer is
its own step. That end-to-end run is the next milestone, and you can
see it on the map."

DO NOT click Compile ladder unless you have separately rehearsed a
campaign on this spec.

---

## Act 7 - Roadmap close (2 min)

DO: click **MAP Roadmap**.

SAY: "Where this stands against the eight goals we set. The
percentages are our judgment; the numbers beside them are measured.
Generation and self-assessment lead - you just watched both. The
gaps are honest: the PHI-scrubbing layer is not built yet, the
vendor loop you saw the bridge for has not been run end to end, and
on the real extract one fidelity number is still below its bar.
Core complete mid-November on the current trajectory."

Stop there. Questions land better on the Roadmap page than on a
closing summary.

---

## If things go wrong

- Fit errors or runs absurdly long: say "we pre-ran this exact
  command this morning", go to **03 Verdict**, change the output
  directory field to `%USERPROFILE%\dev\clinic_backup`, Open the
  run. Every act from 4 onward proceeds unchanged.
- The GUI misbehaves: the terminal does everything -
  `python scripts\m0_gate.py %USERPROFILE%\dev\clinic_backup` shows
  the same gate; `findings.txt` opens in Notepad.
- Someone asks to see the real extract's numbers: open
  `docs\goals_scorecard.md` - it exists for exactly that question.

## The three honesty rules

1. Say up front that today's file is synthetic playing "real" for
   speed, and that the identical pipeline runs on the real extract.
2. Read the real extract's FAIL (close, 76.5% against 87.9%) out
   loud yourself before anyone asks. Confidence, not apology: the
   willingness to print a FAIL is why the PASSes mean something.
3. Never call the privacy posture "differential privacy" or a
   guarantee. It is k-anonymous publication, attacked with positive
   controls, and the attack numbers are a floor on one cohort - the
   scorecard has the exact wording.
