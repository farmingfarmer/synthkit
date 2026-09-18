# synthkit — working conventions

Synthetic clinical data generator and model-evaluation instrument. Core rule: learn the patterns, never copy the records. Real data teaches parameters; generation never touches a record.

## Verify before claiming

- Run `python scripts/run_all_smokes.py` before claiming anything works. Expect 67 suites, 1905 checks, ALL GREEN **on a checkout**. Off a zipball extract - which is what the data machine runs - it is 1896: nine checks in `smoke_buildid` need git to test the archive path and report SKIPPED without it. Both numbers were measured. Do not quote the checkout number to the data machine; that is how a correct run gets read as a failure.
- **THE DEVELOPMENT MACHINE WAS BEHIND THE DATA MACHINE, and that is
  how a green suite here failed there.** Dev was on Python 3.10 with
  pandas 2.3; the data machine installs fresh and got pandas 3.0.5,
  numpy 2.5, scipy 1.18. Writing a string into a float64 column is a
  FutureWarning on 2.3 and a **TypeError** on 3.0, so `smoke_long`
  passed here and crashed there - on the one machine where a crash
  costs a round trip. Reproduced by building a 3.13 venv and running
  the whole net against it; that was the only code incompatibility,
  and CI now runs the net on 3.10 AND 3.13 so the newest library
  behavior is covered rather than whatever the author happens to
  have. **Test against what the DATA machine will install, not what
  is on this one.**
- **A PACKAGING CHECK MUST ASK ABOUT THE PACKAGE, NOT THE MACHINE.**
  `smoke_packaging` compared our floor against the floors of the
  INSTALLED dependencies, so a machine with newer packages failed a
  suite that had nothing wrong with it - numpy 2.5 and scipy 1.18
  both want 3.12. The defect worth catching is our floor being below
  what our own DECLARED MINIMUMS need, which does not move when
  somebody upgrades. That is asserted from a recorded table, with a
  check that every declared dependency appears in it so adding one
  cannot silently skip. The installed-versions gap is REPORTED as a
  note. Verified both ways: the original `>=3.8` bug still fails, and
  3.13 with pandas 3 passes.
- **`pip install -e .` is enough now, and Python must be 3.10+.**
  The numeric dependencies are declared in `pyproject.toml` rather
  than living only in `requirements.txt`, so the old footgun — skip
  one line, seven suites fail on their import line and read as broken
  code — no longer exists. `requires-python` said `>=3.8` while
  numpy, scikit-learn and scipy all require `>=3.10`; that does not
  fail cleanly on an older interpreter, it resolves ancient versions,
  and a silently different pandas gives a wrong number instead of an
  error. `smoke_packaging` now compares our floor against each
  installed dependency's own, so the day one of them moves it goes
  red here rather than on someone's laptop weeks later.
- **CI runs the whole net, in four shards.** It used to run five
  suites, all of them the authored path, and never installed the
  numeric dependencies — so every check covering discover / blueprint
  / generate ran on one laptop, by hand. Shards are round-robin over
  the sorted suite list, and `smoke_packaging` asserts the split
  covers every suite exactly once, because a shard that quietly drops
  one still reports green. Only an unsharded run prints ALL GREEN.
- **The `smoke_gui` flake is fixed, and it was VACUOUS as well as
  flaky.** The over-budget check set the budget to 0, started a
  campaign, slept 0.1s and accepted `timeout` OR `done`. Instrumented
  over fourteen runs: `done` 9 times — the job finished before the
  budget branch ever ran, so the check passed having tested nothing —
  `timeout` once, and `error` three times. The error was not timing:
  the previous job had been CANCELED but was still writing the same
  campaign directory the next one read, so the integrity check saw a
  half-written `campaign.json`. **Cancellation is not synchronous,
  and two jobs on one campaign directory corrupt each other** — that
  product limitation still stands. `api_job` decides the budget from
  the job record alone, so the branch is now tested on a planted
  record: exact, every time, plus the neighboring checks that a job
  inside its budget is NOT a timeout and that canceled beats
  timeout. 16/16 clean afterwards.
- `py_compile` every Python file you touch.
- Assert count==1 before every string replacement — verify the edit, not just the compile.
- **Read a file before Write overwrites it.** `Write` says "updated"
  rather than "created" when the path exists, and that word is the
  only warning. A new `synthkit/spec.py` silently replaced SYNTH_V1's
  DataSpec and took out 17 suites. `ls` the target, or pick a name the
  repo does not already use.
- Never state a number as measured unless you actually ran it.

## Fixing what the real extract found

The DATA MACHINE has the extract; the DEVELOPMENT MACHINE has only its
statistics. Every rule here exists because a fix passed on the
development machine and did nothing, or did harm, on the data machine.

- **Make the fixture reproduce the fault BEFORE fixing it, and assert
  that it does.** A guard verified only against a shape invented here
  will pass while failing on the shape that exists. A string `visit_id`
  passed; the real one is an integer and the guard never fired.
- **Derive the fixture's key property from a measured statistic, and
  say which one.** Do not assume it. Fix 3a v1 was tested against
  independent missingness and shipped against clustered missingness,
  which no clinical column has — it repaired steadiness and thinned a
  49% missing column to 22%.
- **When a fix changes one property, assert the neighboring property
  in the same test.** Steadiness and missingness were fixed and broken
  in one commit because only one of them was checked.

## What the 800-patient extract said about the new path

Ran 2026-08-10 on run7's tidy file: 55,428 rows, 47 columns, 800
patients. 274s with `--lags` (73 columns). The estimate beforehand was
one to three hours — wrong by roughly 30x, and wrong upward, which is
the direction that stops work happening.

- coverage within 0.05 on 45/45 columns, clustering within 0.15 on
  18/18, persistence within 0.15 on 28/34
- **center within 10% of spread on only 18/34 numeric columns** — one
  mechanism found and fixed, the rest STILL OPEN. A piecewise-linear
  inverse CDF assumes uniform density between knots, and across the
  top segment that is the whole error: knots at 316.1 and 2175.0, true
  segment mean 529.0, straight line 1241.7, and that one segment
  carried 7.13 of a 9.97 excess against 2.84 for the other nine
  together. Same cause in whole-number columns, where rounding at .5
  moved 11.3 points of mass off zero. Fixed by publishing the tail's
  own mean and by placing the rounding cut where the column mean says.
  How many of the sixteen this accounts for IS NOT KNOWN: the tidy
  fixture passes 94% of its columns where the extract passes 53%, so
  it cannot answer. `fidelity.json` now carries a `centre_miss` block
  — miss in sds, direction, source skew, whether a tail shape was
  published — so the next real run diagnoses the remainder instead of
  counting it again.
- 16 of 26 relationships dropped to keep the graph sampleable, which
  is not yet reported per-edge anywhere a reader would look
- the identifier-derivative fix fired on real data: `visit_id` and its
  lag both dropped

## What the blueprint may publish

- **k counts PATIENTS, never rows.** One person seen 200 times can
  supply the ten most extreme rows alone; a row-counted rule would let
  them set the published bound and still call it anonymous.
- **No published number may be an individual's value.** The stored
  minimum and maximum are the MEAN of the k most extreme patients'
  own extremes. Storing the true 0th/100th percentile published one
  person's smallest and one person's largest value, and did so for
  weeks in a file described as aggregates-only.
- **THE TWO HALVES CAN NOW MEET WITH AN ANSWER KEY.** `outcomes`
  could never cross the bridge - nobody knows the answer in real data
  - so every vendor number this instrument has produced was on
  covariates AND signal somebody invented. `semisynth.plant` keeps
  the measured covariates and plants a KNOWN outcome on top, so the
  ceiling is computable on data shaped like the customer's. Effects
  are declared in STANDARD DEVIATIONS and converted with the
  blueprint's own spread: a raw coefficient of 0.5 means one thing on
  a creatinine of 1.1 +/- 0.35 and saturates the logit on a glucose
  of 105 +/- 28. Verified end to end - planted +0.9/-0.5 sd recovered
  at +0.86/-0.44, a column with no planted effect recovers +0.03, and
  the covariate marginals come through unchanged. What a model is
  asked here is NARROWER than "does this work on our data", and the
  spec says so about itself.
- **The intercept is SOLVED, not centered.** `sigmoid(E[z])` is not
  `E[sigmoid(z)]`: centering analytically asked for 25% prevalence and
  produced 29.4%. Bisection over draws from the columns' own
  marginals gives 27.2%, and the residual is the correlations the
  table applies afterwards - so `semisynth.verify` reports achieved
  against requested rather than assuming.
- **ATTRIBUTE DISCLOSURE IS MEASURED NOW, and it needed a CONTROL.**
  Membership inference asks "was this person in the cohort"; a
  governance board asks whether the release helps guess a SENSITIVE
  field from the ordinary ones. Raw accuracy cannot answer that - a
  generator that reproduces relationships faithfully is by
  construction good at predicting one column from the others, and a
  real relationship is revealed by any sample of the population. So a
  second adversary is trained on DIFFERENT REAL PEOPLE who were never
  in the cohort: whatever it achieves is population structure anyone
  could obtain, and only the EXCESS over it belongs to this release.
  Across three seeds the excess is -0.009 (-0.033 to +0.003); the
  same attack on a generator that republishes the members' own
  records reads +0.145. The positive control is what makes the clean
  number worth anything.
- k-anonymity on what is published is NOT differential privacy, and
  the effect curves, interaction surfaces and dynamics have NOT been
  audited the same way. A membership-inference test HAS now been run
  against the new path - `scripts/membership_new_path.py`, both
  adversaries, worst AUC 0.52 where a coin flip is 0.50, and the SAME
  attack scores 1.00 and returns FAIL when handed a generator that
  leaks, so the pass is a measurement and not a formality. It remains
  one cohort shape at one k on one fixture: a floor, not a
  certificate, and not a release decision for a different cohort. Say
  all of that whenever the posture comes up.

- **THE SEED SWEEP RAN ON THE REAL EXTRACT (2026-09-10), and it
  re-aimed the fidelity work.** Three seeds (20260731 / 11 / 37):
  close 76.5% / 80.6% / 80.2% - a four-point spread, every seed
  7-11 points under the 87.9% bar, so the close gap is REAL, not
  noise. Direction passed on all three (93.9-97.7%). Shapes
  survived almost everywhere (8/9, 7/7, 7/7) - the U-shape fear is
  not realized on this extract. centre is LARGELY FIXED: 29/33
  within 10% of spread against 18/34 in August; the four remaining
  misses are all 0.11-0.16 sd. The five spread misses all lose
  their magnitude to the published bound (68-96% beyond) - privacy
  by design, now correctly attributed.
- **TWO INVERSIONS APPEARED ON SEED 37 - the first ever observed
  on real data.** procedure_count ~ diastolic_blood_pressure_invasive
  (source -0.126, generated +0.117) and span_days ~
  mean_arterial_pressure_invasive (source -0.304, generated
  +0.136). Both involve the invasive-pressure family - the MAP
  identity territory where the partial-dependence sign flip was
  first found - and both sources are weak negatives. One seed of
  three: per the gate's own guidance, that file does not
  circulate, and the mechanism is a defect to find, not a
  statistic to average away.
- **INTERACTION SURFACES FAIL CONSISTENTLY: 1/6, 2/6, 1/6 across
  seeds.** The first per-run reading of the new criterion says the
  published two-variable surfaces mostly DO NOT survive generation
  on the real extract, on every seed - while shapes and direction
  survive. The suspect mechanism is already named in the fit log:
  11 relationships trimmed to order the graph, and a surface whose
  parent was trimmed cannot fire at generation. This, not centre
  and not blind close-tuning, is the measured target for fidelity
  work.

- **THE REPRODUCTION FIXTURES RAN (`repro_mechanisms.py`), and the
  scorecard reads: TRIANGLE REPRODUCED, SURFACES REPRODUCED,
  SHAPES OVER-REPRODUCED, INVERSIONS OPEN.** The attribution
  triangle (aimed at the measured 75/25) flips its majority driver
  on 2 of 5 seeds - 77/23 becoming 41/59 and 48/52 - with a mean
  lead-share drift of 20 points: the real extract's flip, on
  demand, and hugely seed-dependent (85/15 on one seed, 41/59 on
  another), matching the single-seed real observation. The
  colinear pressure family fails the surface criterion on 2 of 5
  seeds and the SHAPE criterion on 3 of 5 (as low as 5/12 curves
  tracked) - nastier than the real extract in exactly the
  dimension worth stressing. Any fix is measured against BOTH
  fixtures across seeds, flip count and drift for the triangle,
  with the zero-inversion sweep gate held.
- **INVERSIONS DID NOT REPRODUCE, and two hypotheses died
  cleanly.** Dense colinear family: 0 inversions across 5 seeds.
  Sparse-invasive variant (the one measured difference - the real
  inverted pairs involve columns present on a minority of rows):
  0 across 5 more, while close and shapes got WORSE - sparsity
  stresses the sampler, just not into inversion. Both negatives
  show structure, so they read as mechanism differences, not
  script faults. Remaining known differences: --lags and the full
  44-column graph. NOT pursued by guessing - the real-data seed
  sweep remains the inversion gate, and a fix that heals the
  triangle and surfaces is the likeliest candidate to heal
  inversions too, since all three symptoms sit on the trim
  machinery.

- **THE SWEEPS ARE GAUSS-SEIDEL, AND THE LAST WRITER WAS WINNING.**
  In-place ordered refinement leaves the last-refined cyclic column
  aligned to every neighbour's FINAL values while the first sits on
  stale ones - a privilege assigned by seed accident. Candidate 1,
  ping-pong (alternate the direction each sweep), measured WORSE on
  both reproduction fixtures - triangle flips 2/5 -> 3/5, drift 20
  -> 27 points; pressure surfaces 2/5 -> 3/5 failing - proving the
  bias is total, not directional: turn-taking amplifies variance.
  Reverted with the numbers.
- **ONE SYNCHRONIZED FINAL PASS FIXED SURFACE SURVIVAL.** After the
  ordinary sweeps, every cyclic column is computed once from the
  SAME frozen snapshot and all are committed together (Jacobi where
  the sweeps are Gauss-Seidel). Measured on the pressure fixture,
  five seeds: the surface criterion went from FAILING ON 2 OF 5
  SEEDS TO 0 OF 5, with shapes better-or-equal on every seed (5/12
  -> 6/12 at worst, 9/13 -> 11/13 at best), close never worse, and
  zero inversions. pair_fidelity_sweep against HEAD: exactly
  [0,0,0,0] on sign and close across four seeds - the pass moves
  nothing where the sweeps already converge, which is the
  change-only-where-the-mechanism-lives property, held at the
  stronger level. Acyclic bit-identity untouched by construction.
  The real-extract prediction to check on the next refit:
  `interaction surfaces` read 1/6, 2/6, 1/6 across the three seed
  runs; the mechanism transferring should move it sharply up.
- **STILL OPEN, STATED: the attribution flip and the near-identity
  shape ceiling.** The triangle flip is unmoved by both candidates
  (2/5 seeds, drift ~20-24 points) - it is NOT a sweep-order
  effect; suspect discovery-side orientation. And map_cuff <-
  map_cuff_bmdi still drifts up to 0.68 sd in shape: the sweeps
  deliver ~0.78 adjacent correlation where a near-identity needs
  ~0.97 - the ring-adjacency ceiling, the genuine research item.
  Both live on the fixtures with measured gates.

- **THE SURFACE FIX DID NOT TRANSFER, AND THE PREDICTION WAS ON
  RECORD.** The refit (run_seed11b, build 873eb10) read interaction
  surfaces 2/6 - the same 2/6, and an IDENTICAL gate line to the
  pre-fix run. Two non-transfer hypotheses died the same day: the
  pressure fixture WITH --lags still fixes all surfaces (3/3), and
  it selects no lag parents at all - the fixture cannot reach the
  extract's condition. Per the no-third-guess rule, the diagnosis
  moved into the run directory: `peek.py RUNDIR surfaces` prints
  each published surface with its pair, blueprint parents, gap,
  and every trimmed edge touching the child - flagging a pair
  member absent from the parents (cannot fire in the first pass),
  a lost LAG parent (the cyclic refinement skips the record
  entirely - a lag column can never exist in the output), and a
  lost pair member, by name. One command on the data machine
  answers what a third fixture would only have guessed at.

- **THE SURFACE CHAPTER CLOSED ON THE OPERATOR'S peek OUTPUT
  (2026-09-16).** 72 surfaces published on the real extract; the
  application machinery WORKS - `mean_arterial_pressure_cuff <-
  diastolic x systolic` tracks at 0.22, and `787_2` tracks at
  0.29 DESPITE losing both pair members to trims, which is the
  re-application doing its job on real data. Zero LAG flags in 72
  surfaces: that hypothesis is dead everywhere. What fails is not
  a mechanism: `diastolic <- MAP_cuff x MAP_cuff_bmdi` at 1.38 sd
  is the ring-adjacency ceiling expressed in 2D (near-identical
  variant columns), and the other three sit 0.03-0.14 above the
  0.35 bar - marginal drift, parked with the ring research, not
  chased as bugs.
- **AND THE INSTRUMENT VIOLATED ITS OWN TOTALS RULE: 72
  published, 6 measured, 66 silent.** The unmeasured surfaces
  touch token-indicator columns that exist only inside the
  search, and neither the criterion nor peek said so - a bare
  "?" that read as a bug. measure_surfaces now counts every skip
  by reason (published == measured + token + absent + thin, held
  as an identity in the checks), and peek surfaces prints WHY
  each unmeasured surface is unmeasured plus the total line.
  Measuring token-indicator surfaces from the written files would
  require re-deriving the indicators from the list columns -
  possible, recorded here as future work, not silently absent.

- **THE ATTRIBUTION FLIP IS CHARACTERIZED: a two-regime effect
  governed by discovery's edge inclusion, tested blind on fresh
  seeds (2026-09-17).** On the triangle fixture across eight
  seeds: when discovery keeps the third edge (proc <- drug), the
  generated file's attribution LOCKS to the true driver - 88-93%
  proc on all three such seeds, zero flips. When the edge misses
  the inclusion threshold, attribution DESTABILIZES into a
  40-71% band that contains every flip (2) and the borderline
  (1) across five seeds. Mechanism: with the edge gone,
  generation produces drug as a high-skill CHILD of span, so in
  the written file drug couples to span more tightly than proc
  does, and attribution measured on the output correctly follows
  the file, not the source. The flip is NOT a sampler bug - the
  sampler faithfully renders the blueprint it was given; the
  instability lives at discovery's inclusion threshold. Tested
  the honest way: the predictor read only the blueprint and
  committed before the generated attribution was measured - and
  its strong form FAILED (edge-absent seeds 23 and 42 kept at
  69/31 and 71/29), which is what refined "sufficient" down to
  "necessary". The second factor - where in the band an
  edge-absent seed lands - is OPEN, recorded rather than hunted
  blind. Fix directions when this is picked up: stabilize edge
  inclusion near the threshold (discovery-side), or flag
  families whose attribution is inclusion-sensitive in the
  report so a reader knows the number is seed-fragile.

- **5x EXPANSION HOLDS FIDELITY ON THE REAL EXTRACT
  (2026-09-18).** 288,631 rows for 4,000 patients from the
  800-patient source: direction 92.3% against 1x's 93.9%, close
  77.8% against 76.5% - within seed noise both ways - coverage
  42/42, tokens 63/63, EMPTY 4/4, and center IMPROVED to 31/33
  (worst miss 0.18 sd; more rows steady the bins). All four
  spread misses are bound-explained (68-96% of magnitude beyond
  the k bound - privacy by design). Expansion rides the published
  ratios, as the clinic rehearsal predicted; contraction, not
  expansion, is where patterns would thin.

## Numbers before conclusions

- **Recall on the fixture varies 71%-86% across seeds with nothing
  changed but the seed.** A single-seed comparison cannot detect any
  improvement smaller than 15 points, which is one relationship out of
  seven - the size of effect these changes actually produce. Use
  `scripts/recall_sweep.py` and report the range, never a point.

- **Do not report a derived quantity until a second, independent
  measurement agrees with it.** A pooled correlation across columns of
  different scales measured which column a pair came from. A naive
  between-patient share read 0.754 where the truth was 0.500. Both were
  reported, reasoned from, and withdrawn.
- **A statistic fed back as a generative parameter is attenuated
  twice.** Within-patient lag-1 measured by centering on each patient's
  own mean read 0.453 where the truth was 0.700; generating with 0.453
  and re-measuring gave 0.0, and the property vanished while every
  neighboring check passed. Derive the parameter from an unbiased
  identity, then confirm the output measures back what went in.
- **A flag that is read but not used is worse than no flag.**
  `--time-col` was carried into the blueprint and then ignored when
  the lag features were built — which is what every temporal statistic
  is measured on — so passing it changed nothing and the message still
  named the detected column. Assert the override CHANGED something.
- **Do not add a column to a fixture other checks depend on.** The
  CLI fixture plants `y = 2.5x + noise`, which explains 99% of y - two
  points over the 97% cut that calls a relationship near-deterministic
  rather than a discovery. Adding one unrelated column shifted the
  estimate enough to cross it, the pair moved to a different section
  of the report, and three checks broke that had nothing to do with
  either. Give the new thing its own fixture and its own run.
- **A report that lists categories must state a TOTAL the reader can
  check against them.** The dropped-edge section said "2 were mirrors"
  while the run reported 12 dropped: ten had a parent removed to break
  a loop, a category with no section at all, so ten relationships lost
  a parent and no reader was told. An omitted category is invisible in
  any other way.
- **A partial-dependence curve is CONDITIONAL on the other parents,
  and with correlated columns that can flip its sign.** Mean arterial
  pressure is (S + 2D)/3, so holding it fixed, diastolic falls as
  systolic rises. Using that curve to generate diastolic from systolic
  alone produced -0.720 where the source had +0.888. Inverted output
  is worse than absent: it reads as a finding. Every parent also gets
  a curve measured ON ITS OWN, and generation uses that whenever the
  parents it was conditioned on are not all present.
- **A curve and the skill it is paired with must come from the SAME
  model.** The noise added back is sqrt(1 - skill), so a single-parent
  curve carrying the whole claim's skill removes far too much of it:
  systolic against diastolic generated at 0.978 where the source had
  0.888. Each curve now carries the skill of the model that produced
  it.
- **THE TWO HALVES NOW MEET, and what does not cross is written on
  the artifact.** One half learns a blueprint from a real extract; the
  other grades a vendor model against planted signal in a TableSpec.
  Nothing connected them, so every evaluation so far used marginals
  somebody guessed at. `bridge.py` crosses the measured ones - as a
  new `quantiles` distribution kind, because TableSpec was parametric
  only and fitting a normal to a clinical column is the exact loss the
  fitted path exists to prevent. Measured across the bridge on a
  column skewed 2.83: every decile within 0.03 of a source sd.
  Effect curves, interactions, the relationship graph and the dynamics
  have no vocabulary there and do NOT cross. `outcomes` never can - a
  campaign grades against an answer known in advance, and nobody knows
  the answer in real data. The emitted file says all of this about
  itself, because a spec that looks complete and has silently lost
  every relationship is the same failure as a column that looks
  present and is entirely sentinel.
- **A SUB-NUMBER THAT LOOKS LIKE A VERSION IS NOT A LABEL.** The
  bench numbered its controls 1.1, 4.7, 6.3 - and `6.3` appeared
  TWICE, in the same station, which no amount of reading the page
  would reveal. Letters restart inside each step now, so a repeat or
  a gap is visible on sight and a check can assert the whole
  sequence. Each step also owns a HUE carried by its tab, its banner
  and its lettered chips: with one accent color, nothing told you at
  a glance which step you were in.
- **A STEP MUST SAY WHAT IT NEEDS BEFORE IT RUNS.** Every station
  explained what it was ABOUT and then stopped, so the only way to
  learn what had to exist first was to press the button and read an
  error. `you need` / `you get` on every step, and a `next` line at
  the foot naming where to go.
- **`learn` WAS MISSING FROM THE TITLE MAP, and threw.**
  `titles['learn']` came back undefined, `t[0]` raised a TypeError,
  and the handler died after switching the section - so the station
  worked, the heading kept the PREVIOUS step's name, and nothing said
  so. It was also numbered `06` inside a five-step run, which implied
  you arrive there last when it is another way to BEGIN.
- **TRIMMED IS NOT LOST, AND THE REPORT SAID LOST.** The refinement
  sweeps put the cut parents back, and every line of the run still
  called them dropped - on the real extract that would have told the
  operator sixteen relationships left their data when they had not.
  They have their own heading now, with a parent count, and they are
  removed from the loss section rather than appearing in both. The
  loss section ends in a catch-all bucket, so a record that is
  neither a mirror nor a trim lands there whatever else was done with
  it.
- **AND THE LABEL HAS TO FOLLOW THE FLAG.** `--refine-sweeps 0` is
  the comparison the operator is told to run, and the report claimed
  "re-applied" in that run too - false in exactly the place it was
  sent to look. `_order` takes the flag now, and `smoke_loopbreak`
  asserts the two runs disagree: one names the parent as put back,
  the other as gone.
- **A DIAGNOSTIC MUST MEASURE THE THING IT NAMES.** The spread report
  attributed a shortfall to the k rule by the share of squared RAW
  values beyond the bound, which on any column not sitting near zero
  is dominated by the mean. An spo2-shaped column centered at 98.6,
  whose low tail the rule removes, reported 0.0% where the honest
  answer is 26.5% - and a real run then read `spo2 48% of source (0%
  beyond bound)`, which sent me hunting a sampler bug that was the
  privacy rule all along. Spread is deviation from the center, so its
  attribution has to be too.
- **TWO CONSTRAINTS CAN CONTRADICT EACH OTHER, and the repair will
  let them.** `a <= b` and `b <= a` both hold on every row exactly
  when the columns are identical, so both were discovered and both
  were enforced - the real run swapped 258 rows to satisfy one and
  10,328 to satisfy the other, which undid the first and reported it
  broken on 99.8% of rows. Last one wins and the sampler takes the
  blame. They are one EQUALITY, repaired by copying rather than by
  swapping, because swapping two values that should match only
  exchanges the mismatch.
- **The NARROWER column sets the scale.** Judging commensurability by
  the wider one let anything through beside a broad column:
  `glasgow_coma_score <= age_at_visit` comes within 12 of touching,
  which is nothing next to age's spread and everything next to a coma
  score's. Artifacts still get through - the filter is better, not
  finished - which is why constraints are reported before they are
  enforced and enforcement is opt-in.
- **A CONSTRAINT MUST BE EXACT AND COMMENSURATE.** At a 0.999
  threshold a real run found 77 orderings, most of them scale
  artifacts - `span_days <= spo2` at 0.999838 - and the one that
  mattered was buried among them. A rule the source breaks at all is
  not a rule, so exactness is required; and because the repair is a
  SWAP, the two columns must come close on their own scale. Swapping a
  five-day stay with an oxygen saturation of 97 would destroy both.
- **Put the cheap check first.** Every silent fault this tool has had
  was a column read as the wrong type, and the listing that catches
  them ran after five minutes of discovery. `--types-only` reports it
  in seconds - 1.8s on a small file, 6s on a 77-column one - and
  stops. On a dataset nobody has looked at, run that before spending
  anything.
- **A column read as the wrong type fails SILENTLY, and fixing one
  type does not fix that.** Dates became 200 labels with 88% sentinel
  and every check stayed green, because coverage counts presence and
  the sentinel is present. On a plainly tabular file afterwards, three
  more: `$1,234.56` at 85% sentinel, `14:32` at 62%, `45%` as 60
  unordered levels. The guard is `sentinel_share`, which reports the
  failure whatever causes it, and the run now prints how EVERY column
  was typed - one glance would have caught all four. Parsers for
  currency, percent and clock remove three known causes; the guard
  catches the next one nobody thought of.
- **Do not infer an order that was never declared.** mild/moderate/
  severe has one and north/south/east/west does not, and no test on
  the strings tells them apart. A boolean needs no parser either: two
  levels are modeled correctly as two levels. Inventing structure the
  data never carried is worse than missing it - that is the
  partial-dependence lesson again, and it has to be declared.
- **Being MEASURED is a signal, and it was being thrown away.**
  Presence came from a coverage share and a clustering dial and
  nothing else, so whether a lab existed on a row was independent of
  everything on that row. In an extract the test was ordered BECAUSE
  the patient was unwell. Measured on a fixture where presence depends
  on severity: the source separates measured from unmeasured rows by
  35.8 points of severity, generation reproduced -0.2, and the signal
  was simply absent. Discovery had been naming it all along - the
  presence-only shape in `shapes.describe` - and nothing carried it.
  Now the strongest predictor of presence gets a k-screened curve, and
  generation thresholds correlated uniforms against it: informative,
  coverage unmoved, clustering preserved.
- **The fitted path is `synthkit fit`, and it lives in the package.**
  It was 1,526 lines under `scripts/`, which pip does not install, so
  `pip install synthkit` shipped everything except the capability the
  README leads with. `synthkit fit / types / dials` now exist;
  `scripts/run_discovery.py` is a shim over `synthkit.pipeline` and
  still works, because WINDOWS.md names it. The flags are defined
  ONCE, in `pipeline.build_parser`, and adopted by argparse `parents=`.
- **`--dial` reaches the dials, and the answer comes back MEASURED.**
  They had been reachable only by hand-editing blueprint.json on the
  machine holding the extract. `findings.txt` now carries requested
  against achieved, because a dial can be capped by the k-anonymous
  bound, clipped at 0.98, or swapped back by constraint repair - and
  a silent difference is the `--time-col` failure again. An unknown
  column or dial name is an ERROR, never a shrug.
- **`shift` and `scale` were fighting each other.** Applied as
  `(x + shift) * scale`, the shift came out multiplied by the scale
  and the scale dragged the CENTER with it - shift 12 beside scale
  1.5 on a column centered at 34.8 arrived as +35.8. Both dials had
  passed their checks for as long as they existed, because each check
  set ONE dial and neither asserted the neighboring property. It is
  `(x - center) * scale + center + shift` now, about the PUBLISHED
  center rather than the draw's own, so the effect does not depend on
  the seed.
- **ONE ROW PER MEASUREMENT IS MODELED WRONG, SILENTLY.** A long/EAV
  extract - a concept column and one value column - arrives as a
  numeric column whose distribution is a mixture. Measured on five
  real concept scales: 94% of the pooled variance is BETWEEN concepts,
  and the deciles run 1.1, 73.8, 139.8, which is not any lab.
  Coverage reads 100%, the concept column is a legitimate categorical
  so the sentinel guard has nothing to fire on, and center and spread
  both pass. `longshape.detect` reports it with the correlation ratio
  - the harm stated directly - before any discovery runs, and `--long
  CONCEPT=VALUE` pivots. The pivot key uses the REQUESTED `--time-col`
  and never the detected one: detection is good enough to order rows
  and not good enough to reshape somebody's extract on.
- **SPREAD THAT VARIES WITH THE PREDICTION WAS THROWN AWAY.**
  `sqrt(1 - skill)` is one number for a whole column, and clinical
  data is heteroscedastic. Measured on a fixture whose noise grows 4x
  along a parent, five seeds: source conditional spread grows 1.85x
  (1.77-1.91), generated was FLAT at 1.02x (0.98-1.08), and with a
  published residual profile it reaches 1.52x (1.41-1.64). MARGINAL
  spread reads 1.02 either way, which is why nothing caught it - every
  existing spread check is marginal, and a column can have exactly the
  right overall spread and the wrong spread everywhere in particular.
  The profile is a MULTIPLIER on the overall residual sd, measured out
  of sample, k-screened by patients, normalized so total noise
  variance is unchanged, and NOT published when the column is
  homoscedastic.
- **A one-parent fixture cannot test the child's noise model.** With
  `y <- x` alone the graph is symmetric - `x <- y` scored 0.668
  against 0.626 - so the sampler drew y from its marginal, y never
  went through the relationship path, and the first version of that
  measurement was reading a direction generation does not use. Give
  the child several parents so the orientation is not a coin flip,
  and ASSERT the child is a child.
- **A SET COLUMN IS A BAD CATEGORY AND A GOOD SET OF INDICATORS.**
  `_list_marginal` fixed what was GENERATED from a set; nothing fixed
  what was LEARNED from it, because discovery still saw the
  combination string. `t01;t03;t09` and `t03;t14` share the thing that
  matters and share no level. Measured on a fixture shaped after the
  extract - 2,064 combinations, 49.5% sentinel against 69% there -
  `severity <- conditions` reads r2 0.324 through the capped category
  and 0.733 through the one token that drives it. Each published token
  now becomes its own column plus the set size, screened by PATIENTS
  like every other bound, DERIVED at generation from the set already
  drawn and dropped before the file is written. The vocabulary lives
  in `sets.py` and BOTH halves call it: screening tokens twice, in two
  files, is how the two sides come to disagree again. Co-occurrence
  and informative token SELECTION are still not modeled, and the
  module says so about itself.
- **The expansion is capped and the cap is REPORTED.** Only an
  expanded token can carry a relationship, so a reader who is not told
  how many were expanded cannot tell an absent finding from an
  unexamined one. Scaffolding is also counted APART from the
  operator's own columns in the fidelity summary - a four-column file
  expanded to twenty-seven and reported "coverage within 0.05 on
  27/27 columns", which reads as twenty-seven columns of their data.
- **AN INTERACTION SURFACE NAMES ITS OWN TWO COLUMNS - read them, do
  not assume they are the first two parents.** `pair` is the top two
  by IMPORTANCE and `parents` is the blueprint's order after
  filtering, so they disagree whenever a parent is dropped or ranked
  differently. A surface measured on (a, b) was applied to (c, a): a
  0/1 indicator read against a grid of [1..5] floors onto one row, the
  response comes out nearly constant, and it contributed nothing while
  STILL marking both columns handled - so the 24.5-point curve on the
  real driver never fired. This is not a set defect; sets only made
  the orderings differ.
- **AND ONE SEED HID IT.** The set fix measured 98% of the source
  effect on seed 0 and was ready to ship. Across five seeds: 98, 99,
  100, **-1**, 100. Seed 3 reproduced the exact pre-fix failure. A
  single-seed measurement of a change to `generate.py` is worth
  nothing, which this file already said about row counts and recall
  and now says about effects.
- **Spearman cannot see a CATEGORICAL pair, and said nothing about
  it.** `_pair_fidelity` coerced both sides to numeric and skipped
  what became NaN, so a perfectly associated categorical pair compared
  ZERO of them - and "25/28 relationships keep their direction"
  counted only numeric ones. gender, race, ethnicity, visit_type,
  admitted_from and the four list-shaped columns had never been
  checked at all. Cramer's V and the correlation ratio now cover them,
  counted SEPARATELY so the numeric numbers keep their old meaning.
- **A constraint is a statement about a ROW; everything else here
  describes a distribution.** Nothing could express "a visit does not
  end before it begins", so nothing noticed it break on 49% of
  generated rows. Discovered by looking for orderings the source never
  violates, and repaired by SWAPPING the pair rather than clamping -
  a swap leaves both columns holding the same multiset of values, so
  the center and spread the rest of this file works to get right are
  untouched. Pairs on disjoint scales are not constraints: age is
  below year_of_birth on every row and means nothing.
- **Repair arithmetic BEFORE rendering dates.** The enforcement ran
  after the write-back, compared two strings, got NaN, found no
  violations and reported success while fixing nothing - the second
  time in one week that formatting a date early broke a numeric
  consumer downstream.
- **Pair correlation cannot see a broken IDENTITY.** `age_at_visit`
  is the visit year minus `year_of_birth`, and the catalogue reports
  that at 100%. Generated, the identity held on 21.4% of rows while
  the correlation stayed strong, both means stayed right, and every
  check in the fidelity report passed. Someone opening the file finds
  patients whose age contradicts their birth year. Tightness measures
  it - the child's spread left over after least squares on ALL its
  parents - and it must be the whole parent set: the first version
  binned one parent at a time, called 3/3 identities intact, and was
  measuring nothing, because neither parent determines age alone.
- **A shortfall that privacy explains is not a fault.** `6690_2` came
  out at 27% of its source spread on the real run. Reproduced: when
  three patients out of four hundred hold values above the published
  bound, 86-89% of the column's magnitude sits up there and the k rule
  removes it deliberately. Report the share outside the bound beside
  the ratio, or the next person spends a day chasing the sampler for
  something the privacy rule did on purpose. Five such patients only
  drops spread to 86% and would not exercise the check at all - the
  fixture has to be measured, not assumed.
- **Per-column checks cannot see a broken relationship.** Coverage,
  center, steadiness and clustering all pass on a table with no
  structure between its columns at all — the classic way a synthetic
  generator looks right and is useless. Measure the pairs too, and
  report INVERTED separately: a generated relationship with the
  opposite sign to the source is worse than a missing one, because it
  reads as a finding.
- **An internal contradiction is the signal.** Every measurement error
  here was caught by two numbers disagreeing, never by review — 0.735
  against an asymptote of 0.38; a negative rho that cannot exist.
- **State predictions as hypotheses with a test, not as findings.**
  Dates driving the attack: −0.001. Bonferroni as the main suppressor:
  +0, twice. Cutting the cheapest edge to break a cycle: worse, below.
- **THE SEARCH IS NOT THE WEAK PART, and nothing had measured it.**
  `make_tidy_fixture` has planted a U-shape with zero linear
  correlation, an XOR with no main effect, a Simpson's reversal, a
  three-way and a lagged pair since it was written, and
  `score_discovery` scores recall BY KIND - but it read condnet
  models and confirmation reports only, so the fitted path had never
  been scored against planted truth at all. Pointed at it
  (`bench_new_path.py`): FOUND 12/13 with 0 noise edges, and 10/10 of
  the kinds with an honest survival test SURVIVE generation -
  including the XOR the fixture predicted would be missed. The only
  miss is the lagged pair, which needs `--lags`.
- **THE FIXTURE WAS A FOREST, AND THE REAL GRAPH IS A RING.** The
  planted structure was disjoint pairs and triples, so nothing was
  ever dropped to break a cycle and a survival score of 10/10 said
  nothing about a graph that has them. The real extract drops 16 of
  26 relationships to make itself sampleable, and a relationship that
  is FOUND and then DROPPED is indistinguishable in the output from
  one never found. `--tangled N` adds a ring where every column is
  both parent and child: 27 of 42 dropped, 64%, against the real
  62% - and `heterogeneous`, found and surviving at 0.196 -> 0.141 on
  the forest, vanished entirely at 0.228 -> -0.004. That is the
  mechanism to attack, not the search.
- **SKEW WAS NOT THE MISSING AXIS, and adding it broke the fixture.**
  The clinical columns already carry their measured skew - up to 25.6
  - and this fixture still passes center on 94% of its columns where
  the extract passes 53%, so a skewed marginal is not by itself what
  loses the center. Applied to the planted columns it also took the
  U-shape's linear correlation from ~0 to +0.27, and "zero linear
  correlation" is the property that relationship exists to test - the
  fixture stopped being true of itself. `--skew-planted` touches the
  ring only. What loses the center on 16 of 34 real columns is STILL
  UNKNOWN; the `centre_miss` block exists to diagnose it and has not
  been run on a real extract yet.
- **A CYCLE CANNOT BE ORDERED, BUT IT DOES NOT HAVE TO BE.** The
  sampler trimmed parents until an order existed, and processed the
  leftover columns in whatever order the BLUEPRINT LISTED them - so
  the column listed first lost the most. Measured on a ring of eight
  given IDENTICAL curves and near-identical skill, which must
  therefore come out uniform: adjacent correlation ran 0.443 to 0.735,
  and listing the same relationships in reverse moved a pair by 0.192.
  The blueprint meant the same thing both times.

  The fix is not a better cut - cutting by immediate skill loss was
  tried and reverted, and it made both measures worse AND produced
  inversions. The first pass needs an order to get any values at all;
  once every column holds one, the trimmed parents are applied without
  one. Two sweeps by default.

      order-dependence   0.192 -> 0.021   (a ring, reversed listing)
      spread across pairs 0.304 -> 0.187
      mean adjacent |r|   0.594 -> 0.775

  On `pair_fidelity_sweep`, four seeds, against the same-day baseline:
  sign kept 17.0 -> 16.8, **close 16.2 -> 16.8**, INVERTED 0 -> 0.
  One pair lost its sign on one seed; two moved into close on two
  others. The reverted attempt failed the inversion gate; this one
  holds it.
- **RE-RANKING IS WHAT MAKES A SWEEP SAFE.** Each sweep feeds its own
  output back in, so on a ring the values run away - with curves hot
  enough, a single pass already reaches 9.0 against a published bound
  of 2.6. Taking the exact multiset the marginal produced and changing
  only the ARRANGEMENT means the marginal cannot drift and the sweep
  cannot diverge, by construction. It also pins values back INSIDE
  the k-anonymous bound, which the ordered pass could already exceed.
- **AN ACYCLIC BLUEPRINT MUST COME OUT BIT-IDENTICAL.** Most have no
  cycle; if this changed them it would be a sampler rewrite wearing a
  bug fix's clothes. Asserted, and the assertion was mutation-tested
  against a version that refines every column.
- **TWO MECHANISMS CAN REACH THE SAME PROPERTY, and measuring them
  together credits the wrong one.** Applying the previously-cut
  parents also varies conditional spread: on the heteroscedastic
  fixture the "no residual profile" arm went from 1.02x to 1.57x by
  itself. `smoke_spread` holds `refine_sweeps=0` so it measures the
  profile alone, and checks separately that the two together are not
  worse than either.
- **A change that is obviously right on a hand-built fixture still has
  to be measured on a real one.** Inside a cycle the sampler draws
  columns in whatever order the blueprint lists them, so the same
  graph with the same skills gives a different answer depending on how
  the file was written — a genuine defect, and generate.py claims the
  arrow is chosen by out-of-sample skill. The fix that follows from
  that, cutting where the immediate skill loss is smallest, was built,
  proven order-invariant on a four-column loop, and then measured with
  `scripts/pair_fidelity_sweep.py` at 300 patients over four seeds:

      sign kept   16.8 (16-17)  against  17.2 (17-18) before
      close       15.8 (15-16)  against  16.2 (15-17) before
      INVERTED     0-1, two seeds of four,  against  0-0, never

  Every per-seed delta was zero or negative, and it produced sign
  inversions where the arbitrary order produced none — the outcome the
  pair checks exist to catch. Greedy cheapest-cut is locally optimal
  and globally worse: on one seed it trimmed 12 relationships where
  the old order trimmed 10. Reverted; the ordering defect stands.
  Retry it only against that sweep, and beat 0 inversions rather than
  the mean.
- **A SINGLE RUN CANNOT TELL A BIAS FROM A DRAW.** The 800-patient
  run generated 58,768 rows against 55,428 in the source, +6.0%, and
  that was read as evidence of a systematic overshoot in the
  visit-count draw. Over 40 seeds the delta is mean +1.5%, sd 4.6%,
  range -6.9% to +11.4%: the observed +6.0% sits one standard
  deviation from center and is noise. Visit counts are heavy-tailed,
  so a few hundred patients give the row total a standard error near
  6% all by itself. The same mistake was then made twice more in one
  afternoon - a fixture "reproducing" the inflation at +6.7%, and a
  centre_ok of 84% that reads 90%-100% across twelve seeds. Anything
  drawn from a heavy tail needs a seed sweep before it means anything.
- **Routing visit counts through `_draw_numeric` changes nothing.**
  Measured across 60 seeds on identical uniforms: -0.03 percentage
  points, and the paired per-seed difference does not hold its sign.
  The grid's own linear mean is only +0.71 visits off. Reverted rather
  than shipped, because a change that does nothing still has to be
  read by whoever comes next.
- **`recall_sweep.py` cannot see the new path.** It drives CondNet and
  measures whether DISCOVERY finds planted relationships. For anything
  in discover / blueprint / generate, use `pair_fidelity_sweep.py`,
  which measures whether the relationships survive GENERATION. Reading
  a flat recall sweep as evidence about the sampler measures nothing.

- **A CAP THAT NO FIXTURE REACHES IS AN UNTESTED CAP.** The set
  marginal published `MAX_LEVELS_KEPT` tokens because it shared the
  CATEGORY cap, and a category can spill the rest into `__other__`
  where a set cannot - the dropped tokens' share of the size budget
  is redistributed over the survivors. On the real extract
  `conditions` holds 1,050 tokens above k and published 60, so those
  60 absorbed the whole 4.27-per-row budget and each came out about
  3x too common. `drug_routes`, whose 53 all fit under the cap, read
  1.01x with no token missing: the control was in the same run as
  the fault. Every fixture in this repo has TWELVE tokens above k,
  `--harder` does not move it, so nothing here could reach the cap
  and nothing did for as long as it existed.
- **AND UNCAPPING EXPOSED A SOLVE THE CAP HAD BEEN HIDING.**
  `_token_weights` used an undamped multiplicative update, tested at
  EIGHT tokens, which is inside the range where the raw ratio
  converges. At extract scale it oscillates and the ends SWAP: a
  token published at 0.4514 generated at 0.0227 while one published
  at 0.0159 generated at 0.3657. The misses PAIR OFF, which is how
  the diagnosis was reached - a solve that had merely run out of
  iterations would miss low everywhere. Damped to `** 0.5` over 12
  passes: at the extract's own shape, 0 of 2,500 tokens miss by 0.05
  and the worst is 0.0379. A fix measured only at the scale the old
  test used is not measured.
- **THE PRIVACY AUDIT COULD NOT SEE A SET COLUMN, IN BOTH HALVES.**
  `BlueprintLikelihood` branched on `quantiles` and `levels` only, so
  a `list` marginal contributed its coverage term and nothing else -
  a row holding a p=0.900 token and one holding a p=0.001 token
  scored 0.000000 apart, and rarity is exactly what singles a person
  out. The nearest-neighbor half compared the whole combination
  STRING, which on four tokens drawn from hundreds is false for
  every pair, so the term added a constant and canceled.
  A VERBATIM republish was still caught - the strings match - which
  is why the existing leak control never exposed this. A PARTIAL one
  was not: each member's own tokens with ONE swapped scored 0.500,
  a coin flip, against 0.998 with Jaccard. That is a generator
  republishing somebody's condition list almost verbatim and the
  audit calling it clean.
- **TWO ARMS THAT AGREE TO THREE DECIMALS ARE ONE ARM.** The first
  set-aware membership run returned PASS on both arms with seeds 11
  and 37 identical - which was not reassurance, it was the signature
  of a check that could not fail. Adding a 400-token vocabulary must
  move an attack that reads it. With both halves fixed the arms
  differ (0.510 without sets, 0.496 with) and the no-set arm is
  unchanged, so the recorded baseline stays comparable.
- **A QUASI-IDENTIFIER THAT COERCES TO NOTHING WEAKENS THE
  ADVERSARY, NOT THE RELEASE.** `_encode` runs every quasi column
  through `_num`, which returns None for a category, a date or a
  set, and None becomes NaN - so naming one hands the attacker an
  empty column and the lower accuracy reads as privacy. It raises
  now, naming the column. The recorded -0.009 excess is unaffected:
  that cohort passes sex and site as integer CODES, which is why
  this never fired and why it had to be sought rather than waited
  for.

- **THE SIZE AND THE VOCABULARY MUST DESCRIBE THE SAME
  POPULATION.** Uncapping fixed WHICH tokens are published and left
  the size measured over every token the row held - so a set drawn
  at the source's own size had to fill it from a vocabulary the k
  rule had thinned, and every survivor ran proportionally hot. On
  the real extract after uncapping, `conditions` still gapped 1.32x
  and `procedures` 1.50x, and the only two tokens still outside
  tolerance were both on `procedures` and both HIGH. Publishing the
  PUBLISHABLE size closes it to 1.00x exactly. Generated sets are
  visibly shorter than source sets - 3.25 tokens per row against
  4.27 - and that is the k rule's price, REPORTED in tokens per row
  because that is what somebody opening the file sees.
- **PRESENT-AND-EMPTY IS NOT MISSING, and two numbers beside each
  other disagreed about it for as long as both existed.**
  `blueprint` calls a row present when the cell is `notna`, so an
  empty set counts as covered; `has_token` and `sizes_of` required
  `len > 0` and returned NaN. It cost nothing while empty sets were
  rare and became load-bearing the moment size came from the
  publishable subset, because a row whose tokens all sit below the
  floor now draws an empty set and holds a known ZERO of every
  published token. Reading it as unknown would have buried the k
  rule's cost inside the missingness model.
- **SCAFFOLDING IS NOT THE OPERATOR'S DATA, and the constraint
  section counted it as if it were.** A real run reported "orderings
  the source never broke, held on 843/933" and then listed page
  after page of `procedures__has__Oxygen Therapy <= procedure_count
  broken on 52,197 rows (94.9%)` - columns built for the search and
  DROPPED before the file is written. The orderings that were about
  their data sat buried among them. The fidelity summary already
  separates scaffolding when it counts columns; this did not. The
  predicate lives in `sets.py` and both halves call it, because two
  files deciding separately what counts as scaffolding is how the
  two sides come to disagree.
- **A COUPLED CHANGE CANNOT BE FAIL-FIRSTED BY REVERTING ONE FILE.**
  Reverting `sets.py` alone raised `KeyError` inside `blueprint.py`
  and the suite died before reaching the new checks - a crash proves
  the files are coupled, not that the checks can fail. Perturb the
  BEHAVIOR and leave the interface, then watch the checks go red.

- **AN EMPTY SET MEANS "HELD NOTHING", NEVER "HELD SOMETHING WE
  CANNOT PUBLISH".** A visit with no drugs genuinely has an empty
  routes list - 36% of them on the real extract - and generation
  should say so. A visit whose conditions were ALL below the k floor
  is a different thing: that patient HAD conditions, and emitting an
  empty list asserts they had none. Publishing one token instead is
  also not what they had, but it preserves the fact that they had
  something, which is what every relationship on the column depends
  on. Conflating the two put a sign INVERSION into the pair sweep
  where the previous code had none - 0-1 against 0-0, the same shape
  as the reverted greedy cut - on a fixture whose source is 0% empty
  and whose zero-size mass is entirely sub-k. Separated and reported
  apart, five seeds return 0 inversions and land on the pre-change
  baseline.
- **A CONSTANT MULTIPLIER ACROSS SEVERAL TOKENS IS A DENOMINATOR,
  NOT A SAMPLER.** Five `drug_routes` tokens came out high by 1.505,
  1.574, 1.585, 1.580 and 1.572 - and 1/1.57 = 0.637 is the share of
  visits that had any routes at all. `vocabulary` measured `p` over
  NON-EMPTY rows while `has_token` and the fidelity comparison
  measured over ALL present rows. Reading the five ratios as five
  faults would have sent someone into the sampler; reading them as
  one ratio names the bug in a line.
- **AND THE DEFECT UNDERNEATH IT WAS OLDER THAN THE REGRESSION.**
  `vocabulary` discarded present-but-empty rows entirely, so a visit
  with no drugs had never been modeled at all and generation
  invented routes for it. It surfaced only because a change of mine
  made the two halves disagree; a defect that produces a
  self-consistent wrong answer produces no symptom.
- **THE SHARES CANNOT CATCH A SIZE FAULT.** With the denominator
  fixed and the size floor still at 1, token shares measured back
  within 0.043 while the generated empty share read 0.000 against a
  source of 0.359 - the sampler simply spread the same mass more
  thinly over more rows. Assert the empty share BESIDE the shares;
  neither number alone can see it.
- **A LONG VERIFICATION RUN MUST WRITE ITS RESULTS AS IT GOES.**
  Three multi-hour sweeps were killed under memory pressure, and the
  two that piped through `tail` or `grep` lost every completed seed
  because the filter had buffered them. `python -u` straight to a
  file loses only the seed in flight, and the run resumes from what
  is on disk.

- **THE TYPED FRAME AND THE OUTPUT FILE MUST AGREE ABOUT WHAT
  "PRESENT" MEANS.** `prepare` folds "", "nan", "none" and "null"
  into NaN, which is right for a CATEGORY - three spellings of one
  absence - and wrong for a SET, where an empty list is a fact about
  the visit. Generation writes "" as a present value, so the two
  sides disagreed and `procedures` read coverage 0.193 in the source
  against 1.0 generated, with `procedure_quantity` following it.
  A set column's blanks encode to `sets.EMPTY` now. That marker
  never reaches the output file, which still carries "".
- **AND FOLDING IT AWAY THREW OUT A LARGE FACT ABOUT THE DATA.** On
  the real extract, `procedures` is EMPTY on 80.5% of visits,
  `drug_routes` on 36.0%, `active_drugs` on 31.4% and `conditions`
  on 15.8% - none of it modeled, so generation invented procedures
  for four fifths of the visits that had none. Nothing measured it
  because the rows were discarded before any measurement ran. A
  defect upstream of every check is invisible to all of them.
- **PRESENCE AND ITS COUNT PARTNER ARE DRAWN INDEPENDENTLY, AND IN
  THE SOURCE THEY ARE NOT.** Newly visible once empties existed at
  all: on a procedures-shaped fixture the source is empty on 79.6%
  of PRESENT rows and generation emits 97.3%, because whether the
  column is present is drawn from coverage while its size comes from
  the count. A visit that had procedures always records the list.
  NOT FIXED - recorded so the next person does not read it as a
  sampler fault.
- **A CHECK THAT ASKS FOR SOMETHING THE FUNCTION DOES NOT RETURN
  CANNOT FAIL.** `_sets_empty_seen` looked for a `frame` key in
  `discover`'s result, which has never had one, and returned True
  whatever the encoder did. Written and green in the same minute as
  a genuine check beside it. Assert against the thing itself -
  `prepare` returns the frame - and watch it go red first.

- **READ A CSV THE WAY THE PIPELINE READS IT.** `pipeline` uses
  `keep_default_na=False`, so a blank field is `""`; pandas defaults
  turn it into NaN. Measuring the tidy fixture with defaults showed
  0% present-but-empty set rows and it was reported THREE TIMES as
  blind to a shape it reproduces well - 16.4% / 30.8% / 83.3%
  against the extract's 15.8% / 31.4% / 80.7%. The gate had been
  covering that work all along. A wrong reader is the same class of
  fault as a wrong type, and this file has a rule about that already.
- **THE SEARCH CAP'S COST CANNOT BE MEASURED ON THIS FIXTURE, AND A
  NULL RESULT FROM IT IS NOT EVIDENCE.** A relationship was planted
  on a token at rank 27 - just past `EXPAND_CAP` - with an effect of
  2.55 sd. Discovery missed it at cap 24 AND at cap 40, where the
  token IS expanded, because the token lands on 22 of 2,608 rows and
  ranks 24 and 25 sit at 0.88% and 0.84%. Nothing on 22 rows is
  detectable at any cap. On 55,428 rows the same share is 277 rows,
  which is a different question. `peek.py RUNDIR cap` prints the
  share AND the row count at the boundary so the real run can answer
  it; do not read "raising the cap changed nothing" as "the cap is
  free".
- **THE M0 GATE IS A SCRIPT NOW, NOT A NUMBER IN A CHAT LOG.** It was
  "close at least 116, direction about 124", set when a run related
  132 pairs. The denominator moved to 115 and then 114, so 124
  became larger than the number of pairs that exist - unreachable by
  construction, and unnoticed for two runs because the target lived
  nowhere near the numbers. `scripts/m0_gate.py RUNDIR` restates
  them as proportions of whatever the run relates (93.9% and 87.9%),
  keeps INVERTED absolute, and exits non-zero on failure.
- **peek.py HAD NO CHECKS AT ALL.** The tool written so the operator
  would not hand-write a hundred-character one-liner was itself
  never run by anything. Discipline applied to the numerical path
  and not to the tooling is the gap this file keeps rediscovering.

- **THE SEARCH CAP IS NOT A BUDGETING PROBLEM, AND REALLOCATION WAS
  MEASURED AND REJECTED.** Greedy on raw share took `procedures` from
  24 slots to 8 and its mass coverage from 67% to 52%, summing WORSE
  than flat-24 (2.192 against 2.247); greedy on each column's own
  share summed marginally better (2.266) while making three of four
  columns worse. Covering 80% of every set column needs 521 slots
  against the 96 available - `conditions` alone needs 308 - so
  ninety-six cannot be arranged into five hundred.
- **SO THE QUESTION IS WHICH TOKENS, AND FREQUENCY IS A POOR PROXY -
  BUT IT SHIPS AS A CHOICE, NOT A FIX.** `--expand-by signal` spends
  the same budget on tokens that MEASURE as related to a numeric
  column. On a fixture whose driver sits on 8% of rows against thirty
  tokens at 30%, frequency ranks it last of 31 and no cap keeps it,
  while signal ranks it first and discovery recovers it at skill
  0.598. But the budget is fixed, so every slot spent on signal is
  taken from frequency: on the one paired seed where the cap actually
  bound, signal related 14 pairs against frequency's 15, with zero
  inversions and 100% sign kept BOTH ways. One paired seed settles
  nothing. Default unchanged; the run prints which rule it used.
- **A GATE THAT RETURNS TWO IDENTICAL ARMS IS MEASURING A NO-OP.**
  The first sweep of the selection change came back byte-identical on
  both seeds - because the fixture's set columns hold TWELVE tokens
  above k against a cap of 24, so the branch never ran. That is a
  useful guarantee in itself (where the cap does not bind, the change
  is provably inert) and it is not the gate anyone thought they were
  reading. `pair_fidelity_sweep --set-vocab N` now builds a fixture
  where the cap binds.
- **WHEN A MEASUREMENT SCRIPT RETURNS A CLEAN NEGATIVE, SUSPECT THE
  SCRIPT FIRST.** The signal gate reported "not found" for BOTH arms
  and nearly buried a working change: claims key off `child` and
  `predictors[].column`, and it was reading `target` and `parents`.
  The two real negatives of the same day - the reallocation, and the
  cap's fixture blindness - both survived that check because they
  showed STRUCTURE (specific columns worsening; ranks 24 and 25
  differing by 0.04 points) rather than a flat nothing.
- **PUTTING A CAP INSIDE A BRANCH DELETES THE CAP.** The truncation
  to `EXPAND_CAP` was written inside the `signal` arm, so the DEFAULT
  rule expanded every token - 1,050 search columns per set column on
  the real extract, which reads as a hang rather than a bug. Caught
  in minutes by a check written long before, asserting that the cap
  is reported rather than silent: "28 tokens found, 28 expanded".

- **`--expand-by signal` LOSES ON THE REAL EXTRACT, and the argument
  for it was wrong.** Two runs on the same data differing in that one
  flag: frequency related 115 relationships, signal 87 - a quarter of
  them gone - for 3 points of direction (92.0% against 88.7%) and 2
  of close (64.4% against 62.6%) on the smaller set it keeps, and
  signal also broke the EMPTY rate (active_drugs 31.4% source
  against 37.1% generated, 3/4 where frequency is 4/4). The mass
  argument - that the excluded tokens carry 72.4% of `conditions` -
  predicted the opposite. Default stays frequency; the flag remains
  because the question is now ANSWERED rather than assumed.
- **I VERIFIED THE FEATURE AND NOT THE REFACTOR THAT CARRIED IT.**
  Deferring set expansion so the token screen could see the numeric
  columns moved every `__has__` column to the END of the frame. The
  sweep compared signal against frequency - both on the NEW code -
  and came back byte-identical, which read as a clean pass. Old
  against new was never run. On the real extract the same command on
  the same data then trimmed 13 relationships over 18 columns where
  the previous build trimmed 14 over 19, and `close` fell from 87/114
  to 72/115.
- **AND THE FIXTURE COULD NOT HAVE CAUGHT IT.** Its claims come out
  IDENTICAL under both orderings, because its cap never binds - the
  difference only appears once a vocabulary is large enough for the
  screen to choose. What surfaced it was two of the operator's runs
  disagreeing about how many relationships were trimmed. So the guard
  asserts the POSITION - indicators sit immediately after their
  source column - and not the consequence, which no fixture here can
  reach.
- **DISCOVERY IS ORDER-SENSITIVE, which nothing said out loud.** It
  screens to the top predictors and breaks ties on the order it meets
  them, so where a column sits in the typed frame is not cosmetic.
  Rebuilding the mapping in the original order costs nothing and
  removes the question; reasoning about which orders are safe does
  not.

## What breaks on a dataset that is not this one

The rigour above is aimed at ONE extract. Everything in this section
was found by running the whole pipeline across 23 dataset shapes a
customer could plausibly hand over - cross-sectional, no time column,
numeric-only, categorical-only, high-cardinality codes, 120 rows, 60
columns, constant and all-empty columns, duplicate rows, unicode, one
row per entity, one entity holding half the rows, free text, mixed
types in a column, 92% sparse, extreme skew, boolean-ish spellings,
and a flat table with no grouping column at all. Nothing crashed;
what follows is what came out wrong anyway.

- **A LEVEL THE k RULE CANNOT PUBLISH IS NOT A REASON TO PUBLISH
  NOTHING.** A categorical column whose values are each held by fewer
  than k patients came out as `__other__` on EVERY row - honest, and
  a constant column for anything downstream. Codes, SKUs, postcodes,
  order ids and free text are all that shape, so on an ordinary
  business table a large fraction of the columns were being thrown
  away. What CAN be published is the shape: how many distinct values,
  what share of rows, and the profile of their frequencies with the
  extremes k-screened. The labels are then INVENTED. Measured: 1,436
  distinct in, 1,087 out, ZERO real labels republished, no sentinel.
  A column whose levels do clear k keeps its real labels - the shape
  path is for what cannot be published, not for everything.
- **AND IT HAD TO BE ADDED TO BOTH DRAW PATHS.** A categorical with
  visit-to-visit persistence goes through the sticky draw and a plain
  one does not, and both read `m["levels"]` straight out of the
  blueprint - so the capability landed in one and `note` stayed 100%
  `__other__` while `code`, the same kind of column, generated
  correctly. `effective_levels` is the one vocabulary both call. This
  file already recorded that shape of bug for set tokens.
- **WHAT GOVERNS A CATEGORICAL COLUMN IS PATIENTS PER LEVEL, NOT ITS
  DISTINCT COUNT.** Measured on 200 patients and 2,400 rows: 100
  distinct gives 22.8 patients per level and publishes cleanly, 240
  gives 9.8 and the sentinel appears, 1,436 gives 2.1 and the column
  is destroyed. The crossing is exactly k. `types` reports it now -
  "destroyed" is a verdict, "2 patients per level, 0 of 1,436 clear
  the floor" is the diagnosis, and it says whether aggregating the
  column would rescue it.
- **DO NOT PUT A COLUMN IN SOMEBODY'S FILE THAT THEY DID NOT GIVE
  YOU.** `visit_number` was written into all 23 outputs, including
  cross-sectional data where it is meaningless, and a flat table came
  back carrying an invented `person_id` as well - a structure the
  data never had. It is emitted now only where entities actually
  repeat, and named in the run when it is. Flat data comes back flat.
- **AND "ABSENT" IS NOT "KNOWN TO BE NOTHING".** Inferring
  group-less-ness from a MISSING `id_column` broke every hand-built
  blueprint that simply never set one. `None` present means there was
  no grouping column; the key absent means the blueprint predates the
  question. Same distinction as empty-versus-missing, one level up.
- **A COUNT'S MEAN LIVES IN THE ENTITIES THE k RULE WILL NOT
  PUBLISH.** On a dataset where one entity held half the rows, the
  visit distribution published mean 1.9983 beside a grid of
  [1,...,1,121] which implies 1.60, and the row total came out 27%
  short with nothing saying why. The numeric fix does NOT transfer:
  `_shape_tail` bends the segment between the 99th percentile and the
  maximum, which is 1% of entities - four out of four hundred, always
  under the floor - so it can never fire, and forcing it on the
  degenerate case drove the exponent to 119 and made the shortfall
  -50%. Built, measured, REVERTED. `contradictions.find_blueprint`
  predicts the shortfall in ROWS instead and names privacy as the
  cause.
- **A CHECK THAT FIRES ON EVERYTHING IS AS USELESS AS ONE THAT
  CANNOT.** The first version of that rule divided by the mean for
  every distribution and reported 55 of 60 healthy gaussian columns
  as contradictory - their means sit near zero, so an irrelevant
  0.003 on a column with sd 1.0 became "19%". A COUNT is judged
  against its mean, because that multiplies out to the row total; a
  COLUMN against its spread, which is what "within 10% of spread"
  already means everywhere else here.

- **THREE HYPOTHESES, THREE REFUTED, AND THE DIAGNOSTIC GOES WHERE
  THE DATA IS.** A real extract generated `active_drugs` empty on
  36.8% of rows against 31.4% in its source. Built to reproduce it:
  a set column with a declared size partner (0.306 -> 0.294, tracks
  exactly); the same with the measured sub-k tail, 1,907 tokens of
  which 1,378 below the floor (gap -0.025, the OPPOSITE direction);
  and a fixture with cycles, refinement sweeps on against off
  (identical to three decimals). Every set column on every fixture
  reproduces its source rate within 1.1 points.
  What differs is scale, `--lags` and the real relationship graph -
  too many variables to isolate blind, and a fourth guess would have
  been a fourth wrong one. `peek.py RUNDIR empty` separates the two
  possible causes from the RUN DIRECTORY alone: if the size partner
  is zero at the generated empty rate the count column is the cause,
  and if it sits at the source rate the fault is in the set draw.
  Different fixes; one line decides which.

- **A POINT MASS AT ZERO IS NOT A SHAPE A CURVE CAN MAKE, AND THE
  MEAN HIDES IT.** A zero-inflated count drawn from its own marginal
  comes out right, because the inverse CDF reproduces the spike.
  Drawn as a RELATIONSHIP CHILD it does not: the value arrives as a
  curve prediction plus roughly symmetric noise, which is continuous,
  and rounding cannot land a point mass. Measured: mean 2.9206
  against 2.9204 - exact - with zeros at 30.6% against 38.0%. Every
  center, spread and coverage check passes while the column's shape
  at zero is wrong. The blueprint publishes `zero_share` and the
  zeros go to the LOWEST-predicted rows, so the relationship the
  curve found survives: point mass exact, center pays 0.046 of a sd
  at worst against a 10% standard, spearman within 0.03, and a count
  with no zero inflation is untouched.
- **AND IT DRAGGED A SET COLUMN WITH IT.** The blueprint declares
  `active_drug_count == active_drugs__n`, so the set's size comes
  from the count - and the count's excess zeros made the SET empty on
  36.8% of rows against 31.4% in source. The three set columns with
  NO size partner were within 0.004 in the same run, which is what
  made the attribution safe rather than a story. A fault in a numeric
  column can surface as a fault in a set, and `peek.py RUNDIR empty`
  exists to tell those apart.
- **THE MEAN-INFORMED ROUNDING CUT DOES NOT HELP HERE, AND THAT WAS
  MEASURED.** Routing the relationship path through `_to_integers` -
  the cut the marginal path uses - was built and reverted: on a
  relationship child the noise is roughly symmetric, so plain
  rounding already matches the published mean and the bisection lands
  at exactly 0.5. Both arms came out identical on every case tried.

- **THE BENCH SHOWED THE FIRST ENGINE.** The GUI's Learn station
  drives CondNet, so a demo through the interface would have shown
  none of the fitted path - the same gap the atlas had, in the UI.
  The Fit station is a WINDOW onto `synthkit fit`: it launches the
  identical CLI subprocess and reads the identical artifacts, so
  the bench and the terminal cannot drift apart. The gate criteria
  moved into `synthkit/gate.py` for the same reason - the script
  and the UI both call the one computation. And the Roadmap
  station puts the eight goals, built against planned, in the UI
  itself.
- **A JOB'S BUDGET RIDES ON ITS RECORD.** One global 1800s budget
  would stamp `timeout` on a 33-minute fit of the real extract
  while the run was succeeding - a wrong verdict about a healthy
  job, the class the over-budget check was rebuilt to avoid. Fit
  jobs carry four hours.
- **THE STEPGOAL CHECK BIT ITS OWN AUTHOR, correctly.** The two new
  stations went in without `you need / you get` and a `next`
  footer, and the count check went red until they carried both -
  the conventions enforcing themselves on new UI is exactly what
  the counts are for. Also: provenance's `build` is a DICT, and
  slicing it in the browser would have thrown mid-demo.

- **TWO ROUTES INTO THE SAME PIPELINE WEAR THE SAME NUMBERS AND
  HUES.** The from-scratch route counted 01-05 while learning from
  real data was one unnumbered ALT button - the operator asked why
  one path was laid out and the other was not. The measure route is
  now its own rail: Source/Fit/Verdict, carrying the SAME step
  numbers and colors as Describe/Spec/Data, with Campaign and
  Showdown shared by both routes. Parallel structure is itself the
  explanation.
- **A SECTION'S DOM POSITION IS PART OF ITS BEHAVIOR.** Both new
  stations were first appended AFTER </main> closed - valid HTML,
  every content check green, and the panels rendered at the bottom
  of the page under the side tabs, shown only when their tab added
  `.active`. 123 GUI checks asserted content and none asserted
  position; there is a structural inside-<main> check now, watched
  red against the broken arrangement. Layout cannot be seen from
  here - the operator is the first pair of eyes on it, and said so.

- **A NAV LABEL IS A PROMISE, AND "FOR EITHER ROUTE" WAS FALSE.**
  Only the first engine's Learn output could reach Campaign in the
  bench; the measure rail dead-ended at Verdict while the nav
  claimed a shared road. The bridge from a fitted blueprint into
  the exam existed and was MEASURED - it was never wired to the UI.
  The Verdict station sends the blueprint through
  `bridge.blueprint_to_tablespec` into the same client slot the
  Spec station fills, prints what crossed and what did not, and the
  bridged spec is asserted to VALIDATE as a TableSpec - so Campaign
  and Showdown genuinely serve both routes.
- **AND THE ENDPOINT'S FIRST TEST PRINTED AN ERROR AS A RESULT,
  AGAIN.** The run directory it pointed at had been cleaned, the
  endpoint refused correctly, and the test printed the empty fields
  without checking `error` - third instance of the class in one
  week. Assert `error not in r` FIRST, in every endpoint test.

- **A DECLARED SIZE IDENTITY IS ENFORCED BY COPYING, NOT BY DRAWING
  TWICE.** `count == set__n` holds on every source row because the
  count IS the length - one causal direction. When the count is a
  relationship CHILD of the set's own token indicators it is drawn
  AFTER the set, the set falls back to its published size
  distribution, and two independent draws of one marginal agree by
  chance: the identity held on 23% of generated rows and every
  token <- count relationship read ~0.00 against source values up to
  +0.48. The extract's `procedure_count == procedure_quantity broken
  on 79.5%` is the same wound. Refinement sweeps were suspected and
  measured innocent (23% -> 25% with sweeps off). The partner is now
  OVERWRITTEN with the set's actual size after every column exists -
  the constraint-repair rule again: copying enforces an equality, a
  second draw only exchanges the mismatch. On the demo set: identity
  100%, token-count correlations restored, and the gate went from
  failing close at 5/10 to MET on all six criteria.
- **A DEMO DATASET IS A FIXTURE WITH AN AUDIENCE, and building one
  found a real defect.** `make_demo_clinic.py` plants nine patterns
  chosen so each exercises measured machinery, prints its answer key,
  and fits in ~2 minutes. Its first fit failed the gate, and the
  failure was the size-identity defect above - found the day BEFORE
  the live demo rather than during it, which is the entire argument
  for rehearsing on an instrumented fixture.

- **THE NUMBERS EXISTED; THE PICTURE DID NOT, and the picture is
  the product.** A statistician on the team said, correctly, that
  nobody reads JSON - vendors show original-vs-synthetic profiling
  as stat tables, overlaid histograms and paired correlation
  heatmaps. `scripts/fidelity_deck.py` draws exactly that from
  artifacts that already existed, one self-contained HTML in the
  house palette (near-black, gray original, cardinal synthetic),
  with the six gate chips on top and the privacy posture stated in
  the footer in the non-overclaiming words.
- **A FIDELITY REPORT ON REAL DATA IS ITSELF A RELEASE.** A
  histogram of real data is a set of bin counts, and a bin holding
  three patients describes a group small enough to gossip about.
  The deck suppresses source bins backed by fewer than k patients
  and prints the count it suppressed - 47 on the demo set. No
  vendor page does this; say so when the comparison comes up.
- **"DOES 5x DEGRADE IT" IS A MEASUREMENT, NOT A DEBATE.** The
  deck's --compare table on the demo set: 1x (260 patients, 1,924
  rows) and 5x (1,300 patients, 9,723 rows) both read 10/10
  direction, 10/10 close, 0 inverted, gate MET. Expansion rides the
  published ratios; contraction is where patterns would thin.
- **THE CHECK'S FIXTURE FAILED ON ITS OWN EMPTINESS, again.** One
  numeric column renders no heatmap and no relationships render no
  gate chips, so the deck check failed against a healthy deck. A
  fixture must contain the thing the check is about - this file has
  said so for weeks and it still had to bite its own author.

- **THE FOOLPROOFING NEVER LIVED IN THE HUES.** The bench's
  six-color rainbow existed to make the parallel routes
  unmistakable, but the meaning is carried by words - step numbers,
  rail labels, you-need/you-get banners, next footers - and color
  only reinforces. Two accents replaced eight: CARDINAL is the
  create route, GOLD is the measure route, slate is everything
  shared, ink on white throughout. One bit reads faster than six
  hues, a color-blind reader loses nothing, and the bench now
  shares one palette with the deck and the house decks. Color is
  keyed to a data-route attribute, not the step number, so parallel
  stations still share their NUMBERS while each route wears one
  accent. And gold surfaces carry INK text - white on gold is
  roughly 2.4:1, below any contrast bar. The old hex values are
  asserted GONE, not merely unused, because a surviving token gets
  reused.

- **A CORRELATION IS ONE NUMBER; A PATTERN IS A SHAPE, and the deck
  now draws the shapes.** Vendor profiling compares correlations,
  which a threshold, a saturation and a straight line can all
  share. "The patterns, drawn" renders each discovered relationship
  three times on one axis - the published curve from the contract
  (dashed), the shape measured on the original (gray, k-screened
  bins), the same measurement on the synthetic (cardinal) - with
  the shape named in words, skill on held-out patients, driver
  attribution, and a verdict: "tracks within X sd" or DEPARTS.
  A kept bend is visible; a lost one is flagged.
- **THE DEPARTS FLAG'S FIRST FIRING WAS MY OWN ARTIFACT.** np.interp
  holds the last value flat outside the synthetic curve's range, so
  a source point past the synthetic edge compared against that
  plateau flagged the STRONGEST relationship at 0.39 sd. Caught
  because the flag contradicted the pattern's own Spearman - the
  internal-contradiction rule again. Curves are compared only where
  both exist.
- **THE SOFT PASS, and the check that stayed green for the wrong
  reason.** The operator asked for nudges, not bold: active
  stations became a pale wash with a slim accent bar and ink text,
  chips became tinted outlines. The "tabs are REFLECTIVE" check
  kept passing after the flattening because the old gradient
  strings survive in an earlier CSS layer the final-wins overrides
  defeat - green for the wrong reason is decoration, and it now
  asserts the gentle contract instead.

- **THE DASHBOARD IS A STATION, AND ONE CODE PATH FEEDS BOTH IT AND
  THE FILE.** The deck was refactored into `fidelity_deck.build_deck`
  returning an HTML string; the CLI writes it, and the bench's
  Dashboard station drops it into a sandboxed iframe. So the
  interactive view and the shareable artifact are one picture by
  construction - a second implementation is how the two sides drift.
- **A LINTER SILENTLY REVERTED THE REFACTOR.** The first heredoc
  edit landed, then the whole file rolled back to before it - only
  discovered because `build_deck` would not import. Read+Edit held
  where the heredoc did not; verify a function EXISTS after editing,
  not just that the file compiles.
- **A SUBPROCESS CHECK RAN AFTER ITS TEMP DIR CLOSED.** The
  compare-refusal check was outside the `with TemporaryDirectory()`
  block, so it failed on the 1x run being deleted, not on the
  refusal it meant to test - green would have meant nothing and red
  meant the wrong thing. Moved inside. A check must exercise the
  path it names.
- **THE OPERATOR POINTED AT A HOVER STATE, AND TWICE THE WRONG HALF
  OF THE SCREENSHOT WAS PRESERVED.** What they loved was the button
  going PURE WHITE with raised ink lettering and its black shadow -
  the letterpress moment - not the saturated glossy pill beside it.
  Two passes "protected" the pill and deleted the loved state as
  flattening. When a user points at a screenshot with more than one
  state in it, ask WHICH STATE before shipping a look. The design
  is PORCELAIN LETTERPRESS now: every station, button and chip is
  white, raised, ink-lettered with a dark glyph shadow, at rest;
  hover lifts and deepens; route color survives as accents only
  (number chip, accent bar, active bezel ring). Washes stay a few
  points above pure white; inputs stay inset; wash tokens are set
  ONCE.
- **THE COUTURE PASS: light behaves like light.** One imagined
  source; shadows in many soft layers with long falloff (deeper is
  never darker); beveled edges (hairline brighter above, darker
  below); a two-layer letterpress plume; and ONE flourish - a glint
  gliding across the porcelain on hover. A white light-band is
  INVISIBLE on near-white porcelain: a glint reads only as a bright
  core between faintly shadowed flanks, which a freeze-frame barely
  shows and motion shows plainly. Verified by eye at rest,
  mid-sweep and settled hover before shipping.

- **APPENDED OVERRIDE LAYERS FLATTENED THE ONE THING THAT WAS
  LIKED.** The operator pointed at the raised, glossy, embossed
  station tab and asked for it everywhere, always. Two earlier CSS
  passes had done the opposite - overriding the tab with a pale wash
  and `!important` - so the loved effect was the removed effect.
  Band-aid override layers are fragile twice over: they lose to
  earlier `!important` rules (so the intended change never renders)
  AND they defeat the base style they sit above (so a good default
  is lost). Fixed at SOURCE: the two flattening blocks deleted, the
  tab's gradient + inset highlight + drop shadow + specular sheen +
  embossed text made the permanent style of the action buttons,
  preset chips and cards too, always, not on hover. One source of
  truth per rule.
- **"NO DIFFERENCE AT ALL" WAS A STALE SERVE, NOT A NO-OP.** The
  operator's build id showed glossy saturated tabs while every
  recent push had flattened them - proof the running `synthkit gui`
  was serving an OLD copy. The page is baked into the module at
  import, so a running server serves its start-time version forever
  and the browser caches it. The fix is operational, not code: stop
  the server, `pip install -e .`, relaunch, hard-refresh, and
  confirm the build id in the wordmark matches `synthkit version`.
  The version line exists for exactly this - a UI change that cannot
  be seen is indistinguishable from one that was never made.

- **A BAR THAT MOVES MUST BE HONEST, and the log already knows the
  truth.** The living loader's fit bar is parsed from the run's own
  stage lines and search countdown (`i/N columns`), unit-run in node
  against every real stage transition; when nothing is parseable it
  sweeps as an indeterminate comet rather than inventing a
  percentage. A fake percentage is the `--time-col` failure wearing
  motion.
- **THE MOTION RIDES INSIDE THE ARTIFACT.** The deck's reveal and
  press-and-hold pull-apart are inline CSS/JS in the emitted HTML,
  so the shared file animates identically to the Dashboard station
  with no network - one code path, again. Press-and-hold separates
  original from synthetic; release settles them back into overlap,
  which is the fidelity claim performed. `prefers-reduced-motion`
  is honored in both the bench and the deck.
- **WATCH THE MOTION GUARD FAIL ON THE PRE-MOTION OUTPUT.** The
  animation contract was run against the previous build's deck file
  (red) and the new one (green) before entering the suite - the
  fail-first rule applies to decoration exactly as it does to
  numbers, because a check that cannot fail proves nothing about
  either.

- **A VISUAL CHECK IS RUNNABLE FROM HERE: headless Chrome over the
  DevTools protocol against the REAL server.** Click the stations,
  pose the loader, draw the dashboard through the live API, force
  the held-apart state, and READ the screenshots. One pass caught
  four defects every green suite had missed: a stale CSS rule
  rendering the active tab's text near-black on cardinal (deleted
  at source - superseded rules get deleted, not out-shouted), edge
  axis labels clipping, category share text truncating, and the
  apart state burying the axis labels (they recede during the hold
  now). Layout still gets its first REAL eyes from the operator,
  but "cannot be seen from here" is no longer true.
- **AND THE STALE-SERVE TRAP BIT THE CHECKER ITSELF.** The restarted
  bench failed to bind the port, the old server kept serving, and
  the first "fix verified" screenshot was of the unfixed build.
  Kill by port and re-read the build id before believing a
  screenshot, exactly as the operator is told to.

- **SPELLING IS AMERICAN; CONTRACT NAMES ARE FROZEN.** The prose,
  comments, UI and in-repo identifiers use American English
  (behavior, artifact, anonymization, center, neighbor, canceled).
  Three names are deliberate exceptions because they are the
  ON-DISK CONTRACT, not prose: `catalogue.json` (the artifact file
  every existing run directory holds), the `centre_ok` /
  `centre_sd` / `centre_miss` / `"centre"` keys in fidelity.json
  and blueprints (the centre_miss block exists to be read off the
  NEXT real run), and `docs/e2e/end_to_end.json` (a record keeps
  the key names its run produced). A spelling sweep that renames a
  published key is a format break wearing a copy-edit's clothes;
  migrate contract names only deliberately, with a read-both shim.

- **A GATE THAT SAYS NOT MET MUST ALSO SAY WHAT NOW.** The real
  extract failed `close` (76.5% against 87.9%) and the Verdict
  station stopped talking; the operator asked, correctly, what
  their options were. `gate.NEXT_STEPS` now carries, per criterion,
  what a FAIL means and numbered options cheapest-first - and
  "proceed with the failure stated" is on the list, because a gate
  is a floor, not a release decision. INVERTED says STOP, do not
  send the file. The words live in synthkit.gate ONCE; the bench
  panel and scripts/m0_gate.py both print them, so they cannot
  disagree. A green gate prints no guidance - advice under a PASS
  is noise. And fail-first done wrong: `git stash` of the WHOLE
  tree stashes the new checks too and proves nothing - stash only
  the source files, watch the kept checks go red, pop.

- **FRICTION GETS A LOADER; DONE GETS A LIGHT.** The types check
  sat silent between button press and response - the operator hit
  it - and an audit found six more synchronous gaps with the same
  silence (open-run, bridge, compile, learn, learn-generate,
  ladder). Every await shows the living loader now. And completion
  is SAID, twice: a jade lamp on the rail tile, and a completion
  strip in the panel - "Step complete. You are free to move on -
  next: ..." - whose next-step name is read from the section's own
  nextup footer, so the strip and the footer cannot name different
  destinations. A clean types check completes the Source step
  immediately; it used to wait for the whole fit, which
  contradicted the promise the step makes.

- **RANK CORRELATION CANNOT SEE A U-SHAPE, and the gate now can.**
  A U-shaped relationship has Spearman near zero on BOTH tables, so
  it was excluded from the close criterion entirely - generation
  could flatten it and the gate stayed green. Measured: a planted U
  at |rho| 0.023 reads gap 1.69 sd when lost, 0.0 when kept.
  `synthkit.curvecheck` re-measures every published effect curve as
  a binned conditional mean on both tables (`shapes tracked`,
  gap <= 0.35 sd - the dashboard's own DEPARTS bar, promoted), and
  every published two-variable interaction surface as a k-screened
  3x3 grid (`interaction surfaces`; a planted XOR reads 1.05 sd
  when lost). ONE implementation: pipeline records, gate reads,
  deck draws from the same functions. Both criteria are
  conditional - a run that measured none does not fail on absence -
  and nothing above two-way is measured, because the engine does
  not publish higher-order structure and a metric for what the
  generator cannot produce would only restate a known limitation.
- **THE GATE ENUMERATES ITS FAILURES IN THE DASHBOARD, and every
  criterion explains itself.** A FAIL chip gets a highlighted
  section naming every offender with its magnitude (drifted pairs
  with drift, departed shapes with gap); every Verdict row, PASS or
  FAIL, carries what it measures in the data plus "See it
  yourself" - and station mentions render as MINIATURE STATION
  BUTTONS that jump there ([[station]] tokens in gate.py; the CLI
  prints the station's name via gate.plain). A PASS the operator
  cannot check for themselves is just a claim.

- **SHAP IS OPTIONAL, AND ITS ABSENCE IS STATED.** The dashboard's
  "drivers, attributed" section trains a compact model on EACH
  table and splits every prediction among the drivers (mean
  |SHAP|), so attribution doubles as a fidelity view - matching
  bars mean the synthetic data distributes the driving the way the
  original does. SHAP interaction values rank the strongest
  jointly-acting pair, cross-referenced against whether the
  contract publishes a surface for it - an unmodeled interaction
  is a limitation to NAME. Measured before shipping: TreeExplainer
  supports HistGradientBoosting including interaction values, a
  planted XOR pair ranks first at 16x the runner-up, and the
  install touches nothing in the numeric stack. shap pulls numba,
  so it is an OPTIONAL extra ([explain]); without it the deck
  states the omission and names the install command, and the
  omission path is exercised by force-blocking the import in the
  checks - a silently missing section reads as "nothing to show".
  Three self-caught defects: an `a, b = ...` that shadowed the
  args object two hundred lines later, a dedup guard comparing a
  string against tuples (one child attributed twice, identically),
  and the omission check first written OUTSIDE the temp block -
  the compare-refusal trap, hit a second time.

- **GUIDANCE GIVES COMMANDS, INFORMATION SITS BESIDE THE NUMBER,
  AND PLACEHOLDERS MUST SURVIVE THE RENDERER.** Three operator
  asks in one pass: the seed step now states the exact CLI (two
  seeds, a NEW --out per seed, all other flags identical) instead
  of advising "re-run seeds"; the gate-issues table carries a
  k-rule column marking drifted pairs whose columns lose magnitude
  to the published bound, instead of sending the reader to
  findings.txt; and each multi-driver pattern card states its
  arity, its published surface (or the absence), and that
  higher-order is not modeled - machinery that exists but never
  introduces itself reads as machinery that does not exist. The
  first seed command used <angle bracket> placeholders and the
  bench's innerHTML swallowed them as tags, rendering "--src
  --out" - caught by READING the screenshot. Placeholders in text
  that two renderers consume must be legal in both.
- **THE LOG RENDERS, IT IS NOT REWRITTEN.** The fit stream was a
  wall of text nobody would read; each line is now a styled row -
  stages as headers, warnings red, the countdown collapsed to its
  latest line, done green - and the words are the run's own. The
  renderer was unit-run in node against a real log shape before
  shipping.

- **A STATED-ABSENCE BRANCH DOES NOT COVER THE THIRD CASE.** The
  SHAP section had two branches - installed, and absent-with-a-
  note - and shipped a third, silent one: installed but nothing
  eligible, which is what the real extract hit (its top drawn
  patterns are token-indicator pairs whose columns live only
  inside the search). The candidate pool now walks EVERY claim by
  skill (`shap_candidates`, unit-tested for the token-child and
  date-parent exclusions), and the genuinely-empty case renders a
  stated limit. Enumerate the cases; the one without a branch is
  the one the real data finds. And the fix's own check then hit
  the wrapped-heading lesson - grepping a phrase a string-literal
  line break splits - in a check about stating things.
- **THE FIRST REAL GATE-ISSUES READ GAVE THE ATTACK COORDINATES.**
  Close drift on the real extract is CONCENTRATED in set-token
  relationships (has_ <- has_, has_ <- quantity; k-rule column
  blank, so not bound-explained), while ordinary numeric pairs
  mostly survive. The departed surfaces and both seed-37
  inversions converge on the colinear cuff/invasive pressure
  family - MAP = (S+2D)/3 territory. Fidelity work aims there,
  on fixtures first.

- **THE FIRST REAL SHAP READ FOUND AN ATTRIBUTION FLIP.** On the
  extract, span_days is driven 74/26 by procedure_count over
  active_drug_count; in the synthetic file the driving flips to
  22/78 - and procedure_count itself flips 77/23 to 7/93. The
  pairwise checks cannot see this: each pair can survive as
  "faded" while WHO DRIVES WHOM is reassigned. Mechanistically
  consistent with the trim machinery - a parent cut to order the
  cycle routes its influence through the surviving parent. The
  span_days / procedure_count / active_drug_count triangle is
  reproduction fixture number one, beside the pressure family.
- **AND BOTH STRONGEST JOINT PAIRS ARE UNPUBLISHED.** SHAP
  interaction values rank procedure_count x active_drug_count and
  span_days x active_drug_count as the strongest joint effects in
  the real data - neither has a published surface. Beside the
  1-2/6 survival of the surfaces that ARE published, the
  interaction layer is now fully characterized on real data: what
  is published mostly does not survive, and what matters most is
  not published. Both halves measured, not suspected.

- **THE END-TO-END LOOP IS ASSEMBLED AND REHEARSED
  (docs/E2E_EVAL.md).** Goal 7's prediction held: every part
  existed and the glue did not. `synthkit bridge RUNDIR` writes
  tablespec.json from an existing run (the bridge lived only
  inside `fit --emit-spec`, a 40-minute refit for a file the
  blueprint already implies); `synthkit plant` is the answer-key
  step that was reachable only from Python (effects in sd, echoed;
  unknown columns REFUSED by name; all-refused is a sentence, not
  a stack - it surfaced as a traceback until the rehearsal);
  campaign-run and showdown accept registry solver names. Rehearsed
  on the clinic proxy: bridge -> plant -> three tiers -> showdown,
  ceiling / baseline / vendor 0.827 / 0.822 / 0.822 on the strong
  tier, gap to ceiling 0.006 everywhere - the line the product
  exists to produce, working.
- **AND THE NET CAUGHT THE CONVENIENCE DELETING A REFUSAL.** The
  registry-name path replaced campaign-run's "no solver given"
  error, and smoke_llmvendor's guard went red within one cycle -
  the refusal is restored beside the convenience, and an unknown
  registry name STOPs with the built-ins listed. A feature that
  arrives by overwriting an error path has removed a guard
  somebody planted on purpose.

- **THE END-TO-END RUN HAPPENED (2026-09-11), and the exam judged
  its own bar.** run_seed11 -> bridge -> plant (+0.8 age_at_visit,
  -0.5 span_days, 25%) -> three tiers -> showdown, 55,428 rows:
  strong ceiling 0.786 / baseline 0.758 (gap 0.028, PASS);
  as-specified ceiling 0.698 / baseline 0.683 (gap 0.016, bar
  FAIL); weak ceiling 0.622 / 0.617 (gap 0.005, PASS). The
  as-specified FAIL is the instrument working: the 0.700 bar sits
  ABOVE that tier's 0.698 ceiling, so no solver on earth clears
  it - a miscalibrated bar, visible in one line only because the
  ceiling is computable from the planted answer. The gap condition
  held everywhere, which is the solver-quality verdict. LESSON:
  set bars beneath the ceiling, or judge on gap; a worthwhile
  future guard is campaign machinery warning when a tier's bar
  exceeds its own ceiling.
- **The intercept solve verified at extract scale**: as-specified
  prevalence landed at 13,846/55,428 = 25.0%, exactly as
  requested. Amplified tiers shift prevalence by construction
  (strong tier read 57%) - the coefficients move and the intercept
  stays solved for the declared tier.

- **THE REPORT CARD EXISTS (`scripts/report_card.py`), and it
  judges the bar as well as the solver.** One page from artifacts
  the run already wrote - campaign.json, result.json, the planted
  record, showdown.json - never recomputed, so the card and the
  terminal cannot disagree. The planted answer is stated first,
  because the card's authority rests on the answer being known;
  the ceiling/baseline/vendor table carries the gap AND the bar;
  and a bar above its own ceiling is called out BY NAME as a
  miscalibrated bar, not a failed solver - verified against the
  first real run's exact shape (0.700 demanded over a 0.698
  ceiling). The honesty section rides on the card itself: measured
  covariates, invented outcome, what did not cross, narrower than
  "does this work on our data".

- **THE SIGN-OFF PAGE EXISTS (`scripts/signoff.py`), and it was
  read failing before it shipped.** One roll-up from the run's own
  artifacts - the gate from synthkit.gate (the Verdict station,
  m0_gate and this page cannot disagree), the privacy posture in
  the non-overclaiming words with MEASURED counts beside it, the
  limitations restated rather than hidden, and a signature block
  saying exactly what signing accepts: the GENERATED file and this
  page, no source record traveling with either. Exercised on a
  NOT-MET fixture shaped like the real extract's current gate,
  because a sign-off that has only ever rendered green has never
  been read.

- **THE WALKTHROUGH SHOWS THE CODE, AND THE TEST KIT INVITES THE
  BREAKING.** The team leader asked to see the back end, so
  docs/demo_script_v2.md walks the live system first and then the
  EDITOR - gate.py as the one-module pattern, the synchronized
  pass with its measurement in the comment and its reverted
  predecessor in history, a page of this file - because the
  discipline IS the demo. docs/CODE_TOUR.md is the companion map.
  And `make_testkit.py` assembles the folder a teammate runs and
  tries to break: invented clinic data with its printed answer
  key, START_HERE executed VERBATIM end to end before its checks
  were written, and TRY_TO_BREAK with the standing rule stated -
  anything broken becomes a check, and the kit gets harder. The
  kit carries no code of its own, so it cannot go stale against
  the install, and its checks couple every command it teaches to
  the CLI's own --help.

## Talking to the data machine

- **The data machine may not be a git repository.** `docs/WINDOWS.md`
  section 1 is "Pull the repo (no git needed)": it downloads a
  zipball, extracts it and renames the folder. `git pull`, `git
  remote` and `git status` then all fail with `fatal: not a git
  repository`. Read WINDOWS.md before writing any command for it.
- **It holds real clinical data and the development machine does
  not.** Anything that can be done from either belongs on the one with
  no PHI on it — pushing most of all.

- **Code comes from git; output paths are chosen at run time.** Never
  give an invented directory name as if it were a deliverable — say
  plainly that the operator picks it. Four round trips were lost to `fix5`,
  `run_missingfix` and `test9` reading as things to pull.
- One command per line, no line wrapping: an unexpanded `%USERPROFILE%`
  wrote real-data output into the repo.
- **The data machine's terminal mangles pasted commands.** Three times: once a
  wrapped paste put `--src` inside the filename, twice a flag never
  reached the program while the log looked normal. Any script that
  runs there must ECHO the settings it received, or a missing flag is
  indistinguishable from a broken fix and costs a round trip.

## Code

- Python 3.8 target for the stdlib engine. No walrus, no match
  statements.
- **numpy, pandas and scikit-learn are allowed** as of 2026-08-07: the
  data machine can pip install. `synthkit/discover.py` uses them.
  The older stdlib modules stay stdlib until they are replaced.
- **`getattr(obj, name, default)` on a library attribute is banned.**
  `HistGradientBoosting` has no `feature_importances_`; the getattr
  returned None on every column, the fallback ranked columns by
  POSITION, and recall read 0% while the model scored 0.93 on the same
  target. A silent no-op is worse than a crash. Prefer a screen that
  is the same computation as the measurement, so it cannot no-op.
- Every new capability gets smoke checks that can actually fail. A test that cannot fail proves nothing.
- **Run the new check against the OLD code and watch it fail before
  believing it.** Two guards written the same afternoon could not have
  failed: one searched a wrapped heading for `"contribute nothing"`, a
  phrase the line break splits so it never appears whatever the report
  says; the other looked for a bare 5-digit ordinal when `{:.4g}`
  renders 17920 as `1.792e+04`. Both sat beside a positive assertion
  that did work, so the suite was green and half of each check was
  decoration. Revert the fix, or feed the guard the old output
  directly, and see red first.
- **A fixture must contain the thing the check is about.** A "was
  anything genuinely lost" check ran against a graph where nothing was
  lost, so it passed on an empty list. If the check is `X or not Y`,
  assert Y happened.
- One CLI command per line. The development shell is zsh with BSD
  `sed`, so `sed -i ''` needs its empty argument.

## Environment

Two machines, and the split is the whole reason these conventions
exist.

- The DEVELOPMENT MACHINE holds the code and no clinical data at all.
  All authoring, all fixtures and all pushing happen here.
- The DATA MACHINE holds the real extracts. It takes code from the
  repository and runs it; nothing is authored on it.
- Code that touches real data is therefore written WITHOUT access to
  it: stream, report progress, fail readably, and echo the settings
  actually received.

## Tone

- Direct and concise. No emojis. No over-explanation.
