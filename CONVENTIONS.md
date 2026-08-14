# synthkit — working conventions

Synthetic clinical data generator and model-evaluation instrument. Core rule: learn the patterns, never copy the records. Real data teaches parameters; generation never touches a record.

## Verify before claiming

- Run `python scripts/run_all_smokes.py` before claiming anything works. Expect 52 suites, 1430 checks, ALL GREEN.
- **`pip install -r requirements.txt` first, or seven suites do not
  run.** A machine without numpy, pandas and scikit-learn fails
  `smoke_blueprint`, `smoke_discover`, `smoke_dynamics`,
  `smoke_generate`, `smoke_privacy`, `smoke_run_discovery` and
  `smoke_shapes` on the import line — every suite covering the new
  path — and the runner reports seven FAILs that look like broken
  code. Check the tail says ALL GREEN, not just that it exited.
- **`smoke_gui` loses a race roughly one run in eight**, independently
  of any change: the over-budget job check reads a status 0.1s after
  starting the job and sees `error: Expecting value: line 1 column 1`.
  Measured 7/8 in isolation. A single FAIL there is not evidence of
  anything until it repeats; re-run before believing it.
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
- **A DIAGNOSTIC MUST MEASURE THE THING IT NAMES.** The spread report
  attributed a shortfall to the k rule by the share of squared RAW
  values beyond the bound, which on any column not sitting near zero
  is dominated by the mean. An spo2-shaped column centred at 98.6,
  whose low tail the rule removes, reported 0.0% where the honest
  answer is 26.5% - and a real run then read `spo2 48% of source (0%
  beyond bound)`, which sent me hunting a sampler bug that was the
  privacy rule all along. Spread is deviation from the centre, so its
  attribution has to be too.
- **A CONSTRAINT MUST BE EXACT AND COMMENSURATE.** At a 0.999
  threshold a real run found 77 orderings, most of them scale
  artefacts - `span_days <= spo2` at 0.999838 - and the one that
  mattered was buried among them. A rule the source breaks at all is
  not a rule, so exactness is required; and because the repair is a
  SWAP, the two columns must come close on their own scale. Swapping a
  five-day stay with an oxygen saturation of 97 would destroy both.
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
