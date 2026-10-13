# Sprint plan: 2026-10-08 to 2026-10-22

Last sprint delivered the demo with its asks shipped same-day,
the self-contained team bundle with double-click launchers, and
the neural engine's fidelity campaign on the real extract -
close 42.6% -> 80.9% against the 87.9 bar, direction 97.9%
PASS, inversions 10 -> 1, every fix behind a prediction on
record. The enterprise channel is live and verified from a
second clone, and three different verify environments each
caught a real defect, which is the argument for keeping them.
Goal 5's two measured targets were NOT touched (superseded in
priority by the neural path) and goals 7/8 and the multi-table
stretch did not move - carried below with that said plainly.

This sprint is the team actually using the thing, the last
certification blocker, and the close bar.

Each item states its EXIT CRITERION - the sentence that must be
true on 2026-10-22.

## 1. The kit feedback loop (weeks 1-2)

The bundles are distributable as of this week - repo for
networked teammates, zip with launchers for locked-down
machines. The standing rule is the point: anything broken
becomes a check, and the kit gets harder.

EXIT: every reported break is either a new check in the net
(counted: breaks in -> checks added) or a recorded won't-fix
with its reason. A kit session that finds nothing is also a
result - record who tried what. Zero sessions by anyone is the
one failing outcome.

## 2. The output-surface attack at width (week 1)

The last certification blocker before the neural engine's
posture can be written down as decided. The nearest-neighbor
adversary reads 0.500 at 77-column width even when handed the
members VERBATIM - a control that cannot catch a republish
certifies nothing. Distance concentration is the named cause;
the attack must be rebuilt on a surface where a republish is
catchable (per-column match counts, rare-combination hits, or
a learned distinguisher - whatever survives width).

EXIT: an output-surface attack whose republish control FAILS
loudly at real width, and the honest arm's reading recorded -
whichever way it reads. With it, the crowning decision written
with its numbers: what is certified, what is not, both
postures stated.

PROGRESS 2026-10-08: the blindness was a PARSE HOLE, not
geometry - float("nan") passed the `is None` guards and tied
every sparse candidate at infinity; a verbatim republish at
0.500 could never have been concentration, and the
contradiction sat unread. Fixed, watched red, and measured at
the 77-column fixture over 3 seeds: honest output arm
0.495-0.556 with the republish control 1.000 CAUGHT on the
same arm, weights arm 0.544-0.574, worst mean 0.560 PASS. The
duel footer now carries the decided posture.

RESOLVED 2026-10-12: the battery ran on the REAL extract.
Output surface 0.514/0.523/0.551 over three seeds with the
republish control 1.000 CAUGHT on the same arm - certified.
Weights arm 0.366/0.479/0.619, one MARGINAL, recorded as-is;
the network never leaves. Item CLOSED; the crowning posture
stands as written, now on the data of record.

## 3. The close bar - 80.9% against 87.9% (weeks 1-2)

The drift list after the quota build names rare-token pairs
and the parked ring-adjacency family. Next mechanism from the
list, fixture first, prediction on record before the real run
- same discipline that carried 42.6 -> 80.9.

EXIT: either close MET on a real run, or the next mechanism
named with its fixture reproduction and the residual gap
attributed (bound-explained / ring-parked / open), stated on
the gate's own denominator.

PROGRESS 2026-10-12: the re-baseline ran (run_oct12): close
76/94 = 80.9%, direction 97.9% PASS, INVERTED 1 - identical to
the prior build, stable, no regression. quota_landing says
enforcement LANDS at scale (40 adopted, 1 off), so the drift
is not the adopted pairs. The inversion has coordinates:
conditional value corr src -0.306 gen +0.229 on a 0.2%-present
parent with a copied-set-size child - the next fixture is that
exact shape. Still wanted from the run: the Dashboard's
gate-issues drifted list, which names the other 18 pairs.

PROGRESS 2026-10-12 (later): the fixture REPRODUCED the
inversion (two seeds of eight) and the instrument found the
real fault - the 94-row channel decodes to its mean (sd 0.2
against 10.5), so the conditional corr was rank-of-noise.
Fixed with the k-banded sparse grid plus jointly-enforced
k-screened rank couplings (one partner under-covers - the
token-quota lesson). Eight seeds: -0.22..-0.36 against source
-0.34, zero inversions, marginal exact. PREDICTION for the
next real run, on record: the INVERTED 1
(procedure_count <- MAP_invasive) clears, the invasive-family
sparse columns regain their spread, and any close pairs whose
columns are sub-5%-presence numerics move toward source;
close pairs on dense columns are NOT claimed. If INVERTED
clears and close still misses the bar, the next mechanism is
named by the fresh drift list, not by this fix.

PREDICTION FAILED 2026-10-12 (run_oct125), AND THE RUN SAID
WHY: the tier's four partner slots filled with the colinear
pressure family and the procedure_count coupling ranked fifth
- the token-quota cap lesson, re-made. Cap removed, decoy
fixture reproduces the shape (sign 7/8, zero inversions).
REVISED prediction, on record: INVERTED 1 clears on the next
real run; close 76 -> 77 at most from this fix alone; the
rest of the gap belongs to the drift list.

## 4. Governance parity for the neural engine (week 2)

The rules engine has a published auditable contract and dials;
the neural engine has neither, and that gap is on the crowning
list. Scope: the run artifacts state what was learned and
under what bounds (k, visit-count cap, token caps, identity
tiers adopted), readable by the same deck pages - not a new
dial system, a written contract of what this run did.

EXIT: a neural run's artifacts carry the contract statement,
the sign-off page renders it, and one check asserts it cannot
be silently absent.

## 5. Goal 8 - a challenger worth the name (carried, week 2)

Unmoved last sprint. Build the gradient-boosted challenger
with a feature-engineering pass, structurally blinded, in the
vendor seat of the existing exams.

EXIT: on the clinic exam and the real-shaped exam, the card
shows floor / challenger / ceiling with the measured share of
the gap closed. Goal 8 re-scored on the number.

## 6. Goal 7 - a real vendor in the seat (carried, scheduling)

The backend deep-dive puts the team in front of the code; the
exam seat takes any `package.module:function`, and a
teammate's kit-built solver counts.

EXIT: one showdown where the vendor seat is not ours, or a
named commitment with a date.

## 7. Goal 5 residuals (carried, explicitly deprioritized)

(a) The small-cohort token solve (rules engine, flat-vocabulary
shape, constant ~1/3 multiplier) and (b) the
inversion-fragility correlation. Both stand exactly as
recorded; both matter less while the neural path leads. Touch
only if an item above closes early.

EXIT: moved or re-parked with this paragraph updated either way.

## Small debts carried in

- The bench reads 11/13 on its default config (het unexplained
  at discovery) against a recorded 12/13 - still unattributed.
- `discover.py` SettingWithCopyWarning spam on pandas 2.x -
  cosmetic, gone on the 3.13/3.14 lanes, fix at leisure.
- The enterprise repo maintenance rule is now written in the
  RUNBOX (push only when a pull brought something the team
  repo should have) - keep it true.

## Not this sprint, said out loud

The ring-adjacency ceiling and free-text PHI stay parked - one
is the genuine research unknown, the other is a governance
gate. The multi-table intake (goal 1) drops OFF the stretch
list this sprint rather than pretending a third time; it comes
back when an item above frees a week.
