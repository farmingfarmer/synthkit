# Sprint plan: 2026-09-24 to 2026-10-08

The previous sprint's five items all closed (end-to-end run, seed
sweep, close-gap verdict, first SHAP reading, report card), plus
the unplanned: goal 6 closed, relationship dials, the PHI scrub
with its real-read correction cycle, the one-command exam runner,
the team kit, and the fragility flag. This sprint is the demo,
the feedback it generates, and the two measured fidelity targets
the last sprint's diagnostics aimed.

Each item states its EXIT CRITERION - the sentence that must be
true on 2026-10-08.

## 1. The demo, delivered (week 1)

Prep is staged: RUNBOX has the final pull and the second-seed
check; SPEAKER carries the rehearsed numbers including the live
inversion beat. Remaining: the second-seed gate read, one full
out-loud rehearsal, the room.

EXIT: the demo happened on the rehearsed build; the room's
questions are captured as additions to the speaker doc or, where
they exposed a gap, as items below.

## 2. Test kits: hand out, absorb, harden (weeks 1-2)

One kit folder per teammate (or one shared zip), TRY_TO_BREAK
stated. The standing rule is the point: anything broken becomes
a check, and the kit gets harder.

EXIT: every reported break is either a new check in the net
(counted: breaks in -> checks added) or a recorded won't-fix
with its reason. A kit session that finds nothing is also a
result - record who tried what.

## 3. Goal 5 - the two measured fidelity targets (weeks 1-2)

(a) THE SMALL-COHORT TOKEN SOLVE. The extract-shaped fixture at
300 patients generates every published token at ~1/3 of its
source share (constant multiplier, 0/12 within tolerance) while
the real 300-patient sample reads 23/23 - the defect lives at
the flat-vocabulary shape. Reproduce minimally, fix the solve,
verify at BOTH shapes plus the 800-patient scale.

(b) THE INVERSION-FRAGILITY LINK. Monday's rehearsal showed the
fragility flag naming the pair that then inverted. Measure on
the second-seed run and the reproduction fixtures whether
flagged-fragile predicts inversion/close-drift beyond chance -
if it does, the flag becomes a gate-adjacent warning and a
trim-priority signal; if not, that is recorded too. Any
discovery-side change must hold pressure at 0/5 seeds failing
surfaces and the bench at its same-day baseline.

EXIT: (a) fixed and verified at three shapes, or the blocking
mechanism named; (b) the correlation measured and written down,
whichever way it reads.

## 3b. The latent challenger - added mid-sprint (2026-09-24)

The operator called the interaction fidelity unacceptable and
proposed a competing autoencoder generator. Built the same day
as a RULER (`scripts/latent_challenger.py`, k-screened training,
memorization tripwire with positive control): on the fixtures it
holds the triangle's attribution dead-on, the pressure family at
63/63 close, and the planted XOR on every seed - the ceiling is
ours, not the problem's.

NEXT: run it against the real 300-patient sample ON THE DATA
MACHINE (output stays there); adapt the membership attack to
grade it honestly; then decide - import its strengths into the
blueprint path (latent-informed surfaces), or promote it as a
second engine behind the full attack battery.

PROGRESS 2026-09-24: the real comparison table EXISTS and is
decisive - blueprint 0.000 on all eight mined interactions and
the driver flip reproduced (94/6 -> 25/75), latent holding both
(93/7, interactions at source magnitude), tripwire clean at
1.52-2.14. The attack ran at the FIXTURE (worst 0.546 PASS,
republish control 1.000 FAIL); still owed at real width.

RESOLVED 2026-09-24: the attack ran at width (77-column
fixture): reconstruction adversary 0.776/0.694 FAIL under
weights-leak, rows-only adversary 0.500 clean - width is where
AE memorization lives, and the narrow-fixture pass did not
transfer. The fork is decided by evidence: LATENT-INFORMED
SURFACES into the blueprint (mine interactions the challenger's
way, publish as k-screened aggregates), not a second release
engine. The challenger remains the ruler.

DELIVERED 2026-09-25: the DUEL station - both engines on one
source, inside the bench, with three graphic sets (per-column
three-body histograms, the gap heatmap, mined-interaction bars)
and both privacy postures on the page. Next: the full-extract
duel READ on the data machine.

NEW EXIT: (a) the full-extract duel run and read - the
300-sample's "blueprint 0.000 on all eight mined interactions"
confirmed or corrected at full width; (b) first latent-informed
surface published end to end - one mined interaction from the
challenger, k-screened, carried in the blueprint, surviving
generation where the blueprint currently reads 0.000.

## 4. Goal 8 - a challenger worth the name (week 2)

The report card half shipped last sprint; the challenger seat is
still our structurally-blinded floor. Build one real challenger
(gradient-boosted, feature-engineering pass, still structurally
blinded from test labels) and put it in the vendor seat of the
existing exams.

EXIT: on the clinic exam and the real-shaped exam, the card
shows floor / challenger / ceiling with the challenger closing a
measured share of the floor-to-ceiling gap - whatever share it
is, stated. Goal 8 re-scored on the number.

## 5. Goal 7 - a real vendor in the seat (scheduling)

The machinery takes any `package.module:function`. Ask the team
- or a teammate's kit-built solver - to sit the exam.

EXIT: one showdown where the vendor seat is not ours, or a named
commitment with a date.

## 6. Stretch: goal 1 multi-table intake, first cut

The last big unstarted piece. Scope the first milestone only:
two tables with a declared key, joined into the tidy shape the
pipeline already eats, with the join REPORTED (rows matched,
orphans counted) rather than silent.

EXIT (stretch): the two-table fixture round-trips, or the design
note exists stating why not yet.

## Small debts carried in

- The bench reads 11/13 on its default config (het unexplained
  at discovery) against a recorded 12/13 - pre-existing drift,
  proven not the flag's doing; find which change moved it.
- `discover.py`'s `Xq[c] = keep[c]` SettingWithCopyWarning spam
  on pandas 2.x - cosmetic in logs, gone in pandas 3, fix at
  leisure.

## Not this sprint, said out loud

The ring-adjacency ceiling and free-text PHI stay parked - one
is the genuine research unknown, the other is a governance gate.
Neither moves on a two-week clock, and pretending otherwise is
how dates slip silently.
