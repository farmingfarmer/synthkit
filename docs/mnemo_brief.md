# SynthKit — live brief for Mnemo (2026-11-19)

Measured figures only. If asked something not here, say it is not
measured rather than estimating.

## One sentence
SynthKit studies the patterns in real patient records, writes them
down as a readable list, and builds new patients from the list. No
real record is ever copied. It also GRADES stand-in data, including
its own.

## The dataset we can hand over
`dev\handover_rv11\` on the data machine. 11 files; the data itself
is `generated.csv` — 57,684 visits, 800 patients, 44 columns.
Chosen from four runs by a rule written down BEFORE the numbers were
read. A 5x version exists (284,436 visits, identical pattern list).
It is NOT yet anywhere Yesha can reach — that is the AWS request.

## The scorecard, in plain terms
All-or-nothing: only completely right counts as a pass.
- Nothing runs backwards — 0 reversals — PASS (the one we refuse to
  ship on; a reversed relationship is worse than a missing one)
- Every field filled in as often as real life — 42 of 42 — PASS
- "Nothing recorded" as common as real life — 4 of 4 — PASS
- Relationships point the same way — 134 of 151 — FAIL
- ...and are about as strong — 127 of 151 — FAIL
- Individual codes appear as often as they do — 94 of 96 — FAIL
- Relationships that bend keep their bend — 31 of 42 — FAIL
- Effects depending on two things at once — 16 of 57 — FAIL
- A patient's course over time — 40 of 92 — FAIL

151 = relationships found in the real data strong enough to check.
Quality went 43% -> 81% -> 85% on "about as strong", against a
target of 88%. The count grew 94 -> 151, so it is a better score on
a 60% bigger test: 76 came through properly, now 128.

## We made our own tests harder
One check was grading 9 of 281 things — 3% of the work, scored on
that 3%. Another looked at 7 of 75. Both fixed. The first thing the
fixed version found was the largest error in the file (body weight
against age). So some failures are NEW INFORMATION, not new faults.

## Privacy — always quote the control beside the number
- Can anyone tell who was in the original group: we read 0.51-0.55,
  guessing is 0.50, a cheating version reads 1.00 CAUGHT
- Does it help guess something private: -0.006, cheat +0.229 CAUGHT
- Did it memorise: 1.12x, a memorising system 9.7x CAUGHT
Never publish anything fewer than TEN patients share. Counted in
PATIENTS not visits. Cost: about 9 in 10 diagnosis codes cannot be
published — a real limit on rare-condition work.

## Newest capability: clinical notes
We generate the notes too, from the stand-in patient's own record,
AND hold a list of exactly what is true in each. So a note-reading
tool can be marked exactly. On 789 real encounters a simple keyword
reader invented 427 statements; a careless one invented 2,247. Our
own notes make 3,311 statements with ZERO unsupported, and a planted
falsehood was caught 789 of 789.
The trade worth explaining: a keyword reader gets 100% of plain
negations wrong ("patient denies chest pain"). Fix that and it fails
the harder ones — "no family history of diabetes, patient IS
diabetic" — dropping to about 1 in 5. Neither is better; they are
differently wrong, and the test says how.

## Prente: six of nine capabilities testable
CAN TEST: readmission/utilisation flags; structured encounter
summaries; missing-data handling; source-field traceability; speed
and repeatability.
PARTLY: expected vs actual length of stay (we hold actual).
BLOCKED: assigning the billing group (DRG); flagging the
complications that raise the tier; separating main from secondary
diagnosis.
Why blocked: a DRG is set by which diagnosis was the MAIN reason for
admission, which conditions count as complications, procedures, and
where the patient went. Our file has none of those four markers.
TWO of the three are ONE data request away — the hospital system
records whether a condition was present on arrival; it was dropped
when this file was assembled. Same for lab timestamps.
Scope gap: they asked for adult medical patients managed by the
hospitalist team; ours is ~789 inpatient visits, unfiltered.

## If asked about giving Dr D the answer key
Give ALL explanatory variables at full fidelity. Split the delivery:
a development set WITH answers, and a scoring set with the answer
SCRAMBLED. We keep the true values and the exact un-scrambling.
Measured: scrambling the OUTCOME took a vendor from +0.433 to
-0.111 (worse than guessing); scrambling a PREDICTOR denied nothing
(-0.025), because a model routes around one corrupted column.
A score ABOVE the computable ceiling means something leaked.
Say it as being in HIS interest: a blind score is evidence he can
show the next hospital; a score with the answer key in hand is not.

## Never claim
Do not claim an Expert Determination — we have ten of eleven
elements measured; a qualified person writes the eleventh.
Do not call the output differentially private.
Do not quote AWS dollar figures.
