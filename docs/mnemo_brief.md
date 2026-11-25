# SynthKit — live brief for Mnemo (2026-11-25)

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

## AWS: what is done, what is owed, what we are asking for
This is the subject of the meeting with Hari and Brittany. A
written document with the full detail is on both machines:
`SYNTHKIT_AWS_ASKS_2026-11-24.html`.

The shape of it: compute goes to the data. The tool reads ONE
object from S3, writes a fixed list of files to ONE prefix, and
runs as a Fargate task. There is no function in it that sends a
source anywhere, and that absence is deliberate.

DONE, and each of these was run rather than assumed:
- A real upload into a real bucket in the dev account has already
  happened — the dataset and the data dictionary.
- It needed a certificate bundle, because the network intercepts
  encrypted traffic. We never turned verification off.
- The container builds, runs as a non-root user, carries its own
  test net, and holds no patient file of any kind. That was
  checked by scanning the built image, not by reading a config —
  a spreadsheet was found inside it that way, and removed.
- A full fit runs end to end INSIDE the container.
- Both chip types build and run. We name the chip type explicitly
  because a mismatch fails in a way that looks like a broken
  image rather than a wrong setting.

OWED — four things, and every one needs the account, not more
code from us:
- The tool has never run against an S3 source and prefix.
- No run has ever used a task role. Everything so far used a
  pasted session, which is not what a deployment uses.
- The container is not in their registry and no task definition
  is registered.
- We have never written to a bucket that requires encryption.

THE EIGHT ASKS, in the order they unblock things:
1. A bucket, a prefix, and the extract staged at a known key.
2. A task role. We supply the policy — they do not write it.
3. A container registry and permission to push.
4. Does the bucket require encryption, and if so which key.
5. The network: subnets, security group, and whether a private
   subnet can reach S3.
6. A log group and the region.
7. Confirm the execution role and the chip type.
8. Does traffic from inside their network get intercepted the way
   it does on the laptops.
Items 1 to 3 unblock a first run. 4 to 6 are needed before that
run will succeed. 7 and 8 are one-word answers.

WE HAND THEM TWO FINISHED ARTIFACTS so nobody guesses: the
permissions policy and the task definition, both generated from
what the code actually does. Every gap in them is marked and
listed by name.

SIZING, measured on a fit at the real extract's scale: about ten
minutes of wall clock, just under a gigabyte of memory at peak.
The number that matters is the processor time — roughly 1,050
seconds of user time and 1,900 of system time, so the fit used
about five processors at once. On two, it takes three to five
times as long. SAY THIS IF TIMEOUTS COME UP: a timeout set at ten
minutes would kill a healthy run. The task defaults are two
processors and eight gigabytes — nearly nine times the measured
peak, deliberately, because the real extract is wider than the
fixture and this project once died allocating one and a half
gigabytes in a single step. If someone asks why so much headroom,
that is the answer; it is not a measurement of what it needs.
These are measured on a fixture at the extract's scale on a
laptop — NOT a measurement of their cloud.

WHAT CAN EVER LEAVE: sixteen named files, and nothing else. A
request naming anything off that list is refused by name, and a
request mixing allowed and refused files is refused whole — a
half-finished upload that looks complete is worse than a clean
failure. The source extract is not on the list and cannot be
added by a setting.

IF ASKED WHAT GOES WRONG FIRST: the encryption one. A bucket that
requires encryption rejects the upload as a PERMISSIONS error, so
somebody gets sent to the wrong console. The tool tests for that
specifically and before the generic permissions check.

ONE THING WE FOUND AND ARE VOLUNTEERING: the file that records
how a deliberately scrambled release was scrambled was on the
upload list and should not have been. Released file plus that
record reconstructs the withheld values exactly. We demonstrated
it rather than assumed it, removed it, and added a test. No harm
done — the one real upload was the dataset and the dictionary,
and no scrambled release has ever been published. We are raising
it rather than waiting for a reviewer to find it.

## Never claim
Do not claim an Expert Determination — we have ten of eleven
elements measured; a qualified person writes the eleventh.
Do not call the output differentially private.
Do not quote AWS dollar figures.
Do not say SynthKit "runs in AWS" or "is deployed". It runs in a
container on our side and has uploaded to a bucket once. It has
never run inside their account, under a task role, or from their
registry.
Do not promise a date for the first cloud run — it depends on the
eight asks above, not on us.
