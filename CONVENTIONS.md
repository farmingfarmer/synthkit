# synthkit — working conventions

Synthetic clinical data generator and model-evaluation instrument. Core rule: learn the patterns, never copy the records. Real data teaches parameters; generation never touches a record.

## Verify before claiming

- Run `python scripts/run_all_smokes.py` before claiming anything works. Expect 67 suites, 1823 checks, ALL GREEN **on a checkout**. Off a zipball extract - which is what the data machine runs - it is 1814: nine checks in `smoke_buildid` need git to test the archive path and report SKIPPED without it. Both numbers were measured. Do not quote the checkout number to the data machine; that is how a correct run gets read as a failure.
- **THE DEVELOPMENT MACHINE WAS BEHIND THE DATA MACHINE, and that is
  how a green suite here failed there.** Dev was on Python 3.10 with
  pandas 2.3; the data machine installs fresh and got pandas 3.0.5,
  numpy 2.5, scipy 1.18. Writing a string into a float64 column is a
  FutureWarning on 2.3 and a **TypeError** on 3.0, so `smoke_long`
  passed here and crashed there - on the one machine where a crash
  costs a round trip. Reproduced by building a 3.13 venv and running
  the whole net against it; that was the only code incompatibility,
  and CI now runs the net on 3.10 AND 3.13 so the newest library
  behaviour is covered rather than whatever the author happens to
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
  the previous job had been CANCELLED but was still writing the same
  campaign directory the next one read, so the integrity check saw a
  half-written `campaign.json`. **Cancellation is not synchronous,
  and two jobs on one campaign directory corrupt each other** — that
  product limitation still stands. `api_job` decides the budget from
  the job record alone, so the branch is now tested on a planted
  record: exact, every time, plus the neighbouring checks that a job
  inside its budget is NOT a timeout and that cancelled beats
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
- **When a fix changes one property, assert the neighbouring property
  in the same test.** Steadiness and missingness were fixed and broken
  in one commit because only one of them was checked.

## What the 800-patient extract said about the new path

Ran 2026-08-10 on run7's tidy file: 55,428 rows, 47 columns, 800
patients. 274s with `--lags` (73 columns). The estimate beforehand was
one to three hours — wrong by roughly 30x, and wrong upward, which is
the direction that stops work happening.

- coverage within 0.05 on 45/45 columns, clustering within 0.15 on
  18/18, persistence within 0.15 on 28/34
- **centre within 10% of spread on only 18/34 numeric columns** — one
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
- **The intercept is SOLVED, not centred.** `sigmoid(E[z])` is not
  `E[sigmoid(z)]`: centring analytically asked for 25% prevalence and
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
  twice.** Within-patient lag-1 measured by centring on each patient's
  own mean read 0.453 where the truth was 0.700; generating with 0.453
  and re-measuring gave 0.0, and the property vanished while every
  neighbouring check passed. Derive the parameter from an unbiased
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
  the artefact.** One half learns a blueprint from a real extract; the
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
  and its lettered chips: with one accent colour, nothing told you at
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
  is dominated by the mean. An spo2-shaped column centred at 98.6,
  whose low tail the rule removes, reported 0.0% where the honest
  answer is 26.5% - and a real run then read `spo2 48% of source (0%
  beyond bound)`, which sent me hunting a sampler bug that was the
  privacy rule all along. Spread is deviation from the centre, so its
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
  score's. Artefacts still get through - the filter is better, not
  finished - which is why constraints are reported before they are
  enforced and enforcement is opt-in.
- **A CONSTRAINT MUST BE EXACT AND COMMENSURATE.** At a 0.999
  threshold a real run found 77 orderings, most of them scale
  artefacts - `span_days <= spo2` at 0.999838 - and the one that
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
  levels are modelled correctly as two levels. Inventing structure the
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
  and the scale dragged the CENTRE with it - shift 12 beside scale
  1.5 on a column centred at 34.8 arrived as +35.8. Both dials had
  passed their checks for as long as they existed, because each check
  set ONE dial and neither asserted the neighbouring property. It is
  `(x - centre) * scale + centre + shift` now, about the PUBLISHED
  centre rather than the draw's own, so the effect does not depend on
  the seed.
- **ONE ROW PER MEASUREMENT IS MODELLED WRONG, SILENTLY.** A long/EAV
  extract - a concept column and one value column - arrives as a
  numeric column whose distribution is a mixture. Measured on five
  real concept scales: 94% of the pooled variance is BETWEEN concepts,
  and the deciles run 1.1, 73.8, 139.8, which is not any lab.
  Coverage reads 100%, the concept column is a legitimate categorical
  so the sentinel guard has nothing to fire on, and centre and spread
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
  of sample, k-screened by patients, normalised so total noise
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
  and informative token SELECTION are still not modelled, and the
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
  the centre and spread the rest of this file works to get right are
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
  centre, steadiness and clustering all pass on a table with no
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
  - and this fixture still passes centre on 94% of its columns where
  the extract passes 53%, so a skewed marginal is not by itself what
  loses the centre. Applied to the planted columns it also took the
  U-shape's linear correlation from ~0 to +0.27, and "zero linear
  correlation" is the property that relationship exists to test - the
  fixture stopped being true of itself. `--skew-planted` touches the
  ring only. What loses the centre on 16 of 34 real columns is STILL
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
  deviation from centre and is noise. Visit counts are heavy-tailed,
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
  out. The nearest-neighbour half compared the whole combination
  STRING, which on four tokens drawn from hundreds is false for
  every pair, so the term added a constant and cancelled.
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
  BEHAVIOUR and leave the interface, then watch the checks go red.

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
  with no drugs had never been modelled at all and generation
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
  on 15.8% - none of it modelled, so generation invented procedures
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
