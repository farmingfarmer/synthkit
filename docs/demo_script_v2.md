# The deep-dive walkthrough (v2): live system, then the code

For the session where the audience asked to see the BACK END.
Structure: fifteen minutes of the live system telling its own
story, then the code tour, then the exam loop end to end. Every
DO is a literal action; the honest lines marked **keep** are the
credibility of the whole session, say them in some form.

`docs/demo_script_clinic.md` remains the from-zero script for a
first-time audience; this one assumes the room has seen the basics
and wants depth. Prep is the same ritual: pull, ALL GREEN at the
count `docs/WINDOWS.md` names, rehearse once.

## Before the room fills (15 min)

1. Pull current build; run the net; confirm ALL GREEN.
2. `python scripts\make_demo_clinic.py -o` a scratch folder, and
   run one fit into a live directory so every station has food.
3. Open in tabs: the bench, the two report cards from the real
   exam (the FAIL card and the clean card), the sign-off page.
4. Have `CONVENTIONS.md` and `synthkit/gate.py` open in an editor
   - the code half of the session lives there.

## Act 1 - the dashboard is the argument (5 min)

DO: VIEW Dashboard on the fitted run. Scroll slowly; press and
hold a histogram; press and hold a correlation heatmap.

SAY: "Original gray, synthetic cardinal. Holding a chart pulls
the two apart so you can read each shape alone - and the
heatmaps crossfade so any pair that differs blinks. **keep**: the
histogram bins you see were k-screened before they were drawn -
a fidelity report on real data is itself a release, so bins
backed by fewer than ten patients are suppressed and counted.
The shareable file and this screen are one build; they cannot
disagree."

DO: scroll to "The drivers, attributed" if present.

SAY: "Attribution is computed on BOTH tables - matching bars mean
the synthetic data distributes the driving the way the original
does. On our real extract this section caught the synthetic file
rerouting a pattern's driving from one covariate to another -
invisible to every pairwise number - and that finding is now a
reproduction fixture with a measured gate."

## Act 2 - the gate that explains itself (5 min)

DO: 03 Verdict on the run. Point at a PASS row's explanation,
then a FAIL's what-now panel and its mini station buttons.

SAY: "Eight criteria. Every row, pass or fail, says what it
measures and how to inspect it yourself - a PASS you cannot check
is just a claim. A FAIL comes with its options, cheapest first,
including the exact seed-sweep commands - and 'proceed with the
number stated' is on the list, because **keep**: a gate is a
floor, not a certificate. On our real extract the gate currently
reads NOT MET on close and interaction surfaces, and we can say
exactly why and where."

## Act 3 - the code, in the editor (10 min)

DO: open `synthkit/gate.py`, scroll it top to bottom slowly.

SAY: "One module holds the criteria, the explanations, and the
guidance. The Verdict station, the command-line gate, and the
sign-off page all call this file - the three cannot disagree,
which is a design rule everywhere here: the dashboard and the
shareable deck are one build; shape fidelity is measured by one
function that the pipeline records and the dashboard draws."

DO: open `synthkit/generate.py`, find the synchronized final pass
(search: PING-PONG or synchronized).

SAY: "Every non-obvious decision carries the measurement that
justified it, in the comment, with numbers. This one shipped
after being measured three ways - and the version we tried first
is in the history as a revert WITH its numbers, because **keep**:
here a change to generation is worth nothing until the multi-seed
sweeps say so, and 'measured and reverted' is a respected
outcome."

DO: open `CONVENTIONS.md`, scroll a page of it.

SAY: "The lab notebook. Every rule was paid for by a specific
failure, recorded with its numbers. New contributors read this
before the code."

## Act 4 - the exam loop, end to end (8 min)

DO: in the terminal, on the fitted run - ONE command:

    python -m synthkit.cli exam RUNDIR --effect age_at_visit=0.8 --effect span_days=-0.5 -o EXAMDIR

It echoes all six stages - bridge, plant, ladder, baseline,
showdown, report card (`docs/E2E_EVAL.md` still walks the stages
separately for when one needs varying).

SAY over the output: "The covariates are measured; the OUTCOME is
planted, in standard deviations, because a grade needs an answer
known in advance. That is what makes the ceiling computable -
the best score the planted answer permits ANY solver."

DO: open the two report cards side by side.

SAY: "Same exam, same solver, same data - only the bar moved. The
FAIL card is the one to study: the bar sat ABOVE the tier's own
ceiling, so no solver on earth could clear it, and the card says
so - 'a miscalibrated bar, not a failed solver'. **keep**: no
vendor's self-benchmark can make that distinction, because only
an exam that planted its own answer knows the ceiling. And the
honest scope: what a model is asked here is NARROWER than 'does
this work on our data', and every artifact says so about itself."

## Act 5 - close (2 min)

DO: MAP Roadmap.

SAY: "Percentages are judgments; the numbers beside them are not.
Goal six closed this sprint; the two genuinely open problems -
the attribution flip and the ring-adjacency ceiling - live on
instrumented fixtures with measured gates, which is this
project's way of being honest about what it cannot do yet."

Then hand out the test kit (`docs/TEAM_TESTKIT.md`): "everything
you just saw runs on your machine on invented data - and the kit
ends with a try-to-break list. Anything you break, file it; the
suite grows a check and the kit gets harder."

## If things go wrong

The terminal does everything the bench does - `m0_gate.py`,
`findings.txt`, `report_card.py`, `signoff.py` all run without a
browser. A pre-run backup directory turns any live failure into a
thirty-second detour, and saying "we pre-ran this exact command"
out loud costs nothing and reads as discipline, because it is.
