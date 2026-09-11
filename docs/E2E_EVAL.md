# The end-to-end evaluation run (goal 7), on the data machine

The loop this page walks: a fitted run of real data becomes an exam
with a KNOWN answer, and solvers are graded against the ceiling that
answer implies. Every command echoes what it did; every refusal is a
sentence, not a stack. Rehearsed end to end on the clinic proxy
before this page was written - ceiling / baseline / vendor came out
0.827 / 0.822 / 0.822 on the strong tier with the gap condition
holding at 0.006.

Paths shown as `RUNDIR` and `EXAMDIR` are yours to choose. One
command per line. `python -m synthkit.cli` and the bare `synthkit`
command are the same program.

## 0. What you need

A finished fitted run (blueprint.json present) of the real extract -
any seed's run directory works.

## 1. Bridge - the fitted run becomes an authorable spec

    python -m synthkit.cli bridge RUNDIR

Writes `RUNDIR\tablespec.json` and prints what crossed and what did
NOT (effect shapes, interactions, dynamics have no vocabulary there;
outcomes never cross - nobody knows the answer in real data, which
is the whole reason for step 2).

## 2. Plant - a known outcome on the measured covariates

Effects are in STANDARD DEVIATIONS of each covariate. Pick two or
three columns the bridge carried, with effect sizes worth
recovering (0.5-1.0 sd), and one you leave out as the null:

    python -m synthkit.cli plant --spec RUNDIR\tablespec.json --effect age_at_visit=0.8 --effect span_days=-0.5 --prevalence 0.25 -o EXAMDIR\exam_spec.json

It echoes each planted effect, REFUSES any column the bridge did
not carry (by name, with why), and writes two files: the exam spec,
and `exam_spec.planted.json` - the full record. THE ANSWER KEY
LIVES IN THAT SECOND FILE; it does not travel with the exam.

## 3. Compile the ladder

    python -m synthkit.cli campaign-compile --goal predict --spec EXAMDIR\exam_spec.json --outcome outcome --bars auroc=0.7 -o EXAMDIR\campaign

Three tiers: amplified, as-declared, attenuated - the bar drops on
the weak tier but the CEILING drops more, and the gap condition is
what still bites.

## 4. Run the baseline

    python -m synthkit.cli campaign-run EXAMDIR\campaign --solver autosolver

Registry names work here (autosolver, autosolver_regress,
autosolver_hybrid, autoclean); a vendor's own code is a dotted path
`package.module:function`. Each tier prints AUROC against the
CEILING the planted answer implies - the gap between them is what a
solver leaves on the table, and the ceiling is computable only
because the answer was planted.

## 5. Showdown - vendor against baseline, with the ceiling

    python -m synthkit.cli showdown EXAMDIR\campaign --solver autosolver_hybrid --baseline autosolver --json-out EXAMDIR\showdown.json

One line per tier: `ceiling / baseline / vendor [bar]`. This is the
line the report card is built from. A real vendor replaces
`autosolver_hybrid` with their dotted path; nothing else changes.

## What this run does and does not claim

The covariates are measured from the real extract through the
k-anonymous bridge; the OUTCOME is invented, on purpose, because a
grade needs an answer known in advance. What a model is asked here
is NARROWER than "does this work on our data", and the spec says so
about itself. Effect shapes, interactions and dynamics did not
cross the bridge - the exam is marginals-and-correlations shaped,
and the did-NOT-cross line from step 1 is part of any honest
report.
