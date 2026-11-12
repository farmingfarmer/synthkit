# synthkit — working conventions

Synthetic clinical data generator and model-evaluation instrument. Core rule: learn the patterns, never copy the records. Real data teaches parameters; generation never touches a record.

## Verify before claiming
- **THE SAME DIGEST FAULT TWICE IN ONE DAY, AND THE SECOND TIME
  WAS BECAUSE I DIAGNOSED A CLASS AND FIXED AN INSTANCE
  (2026-11-08 night).** The data machine's net came back with one
  red suite - `smoke_multitable` - on the pinned digest of a
  single-coupling link's output. The libraries were IDENTICAL to
  the development machine's: pandas 3.0.5, numpy 2.5.2,
  scikit-learn 1.9.0. The only difference was **Python 3.14.6
  against 3.14.8**. A digest of a CSV of floats is not portable
  across a PATCH release, let alone across stacks - and hours
  earlier I had written up exactly that lesson for a catalogue
  digest that broke the team's 3.10 lane, then left this one
  standing.
- **THE CLAIM IS NOW MADE WITH TWO ARMS ON ONE MACHINE.** The
  property was never the bytes; it is that `_whiten` is a NO-OP
  where a link has one column. So the check runs the SAME seed
  twice with the whitening replaced by the identity and requires
  the outputs to be byte-identical - both arms here, now, with
  whatever is installed. And it asserts the substitution DOES
  change the output on a link with several colinear targets, so
  the identity is a statement about that link rather than about a
  function that does nothing anywhere. Removing the whitening
  turns four checks red.
- **AND THE AUDIT'S FIRST VERSION HAD THE HOLE IN ITS OWN
  PREMISE.** It scanned only suites containing `hexdigest` - and
  the mutation that put the pinned digest back went UNNOTICED,
  because removing the comparison had removed the word too. It
  scans every suite now. What separates a ban from a false
  positive is LENGTH: a full 40-character git SHA is a
  legitimate identifier and two suites use one on purpose, while
  `hexdigest()[:16]` is the shape of an expected-output pin, so
  12 to 32 characters is the banned form. The scanner also
  asserts it still finds those two 40-char SHAs, which is how a
  pattern that stopped matching would announce itself.
- **AND MY POLL PATTERN SAID NO FAILURE WHILE ONE SAT IN THE
  LOG.** Waiting on the remote net I grepped
  `suite..s.. FAILED` - two characters between "suite" and "s" -
  against a line reading `1 suite(s) FAILED`, which has ONE. So
  the poll reported NO-SUITE-FAIL for as long as I asked it, and
  the failure only surfaced when I read the log's tail instead.
  A diagnostic predicate that does not match the thing it claims
  to find is the same fault as a check that cannot fail, and this
  file already records a PowerShell `^FAIL` mangled to `*FAIL`
  producing a confident wrong verdict from passing lines. Read
  the tail; do not trust a pattern you have not seen match.

- **THREE PUSHES REPORTED SUCCESS AND PUSHED NOTHING, BECAUSE
  `git checkout <sha>` LEAVES A DETACHED HEAD (2026-11-08).**
  Running a control arm meant checking out the previous build:
  `git checkout -q 5466e76`, measure, `git checkout -q 58acc67`.
  That second checkout attaches HEAD to a COMMIT, not to `main` -
  so the next three commits landed on a detached HEAD, `main`
  stayed where it was, and `git push origin main` dutifully
  pushed the unchanged branch and said **"Everything
  up-to-date"**. `git push -q ... && echo PUSHED` printed PUSHED
  every time, truthfully: the push succeeded in pushing nothing.
  `git log --oneline -1` showed my commits throughout, because
  they exist - just not where the remote was being asked to look.
- **FOUND ONLY BECAUSE THE DATA MACHINE COULD NOT FETCH THE
  SHA.** The zipball 404'd on all three archive URLs while the
  API reported the repo present, and `git ls-remote origin main`
  then named a commit three behind. Nothing local would have
  shown it: the working tree was clean, the log was right, and
  the exit code was zero. This file already records the
  file-level version of this trap - `git checkout COMMIT -- PATH`
  stages the file, so the plain restore puts the old one back,
  and "a matching SHA is not a matching TREE". This is the
  branch-level form and it is worse, because the thing that
  silently did nothing was the PUBLISH step.
- **TWO GUARDS, AND NEITHER IS AN EXIT CODE.** Before committing,
  `git branch --show-current` must be NON-EMPTY - an empty answer
  is a detached HEAD and any commit made there is invisible to a
  push. And after pushing, verify against the REMOTE with
  `git ls-remote origin main`, not against the exit status: a
  push that has nothing to do exits 0, which is the
  check-that-cannot-fail shape in the one place it costs a day of
  somebody else's work. Recovered by `git branch -f main <sha>`
  after confirming `main` was an ancestor, so the fast-forward
  could not lose anything.
- **AND THE CONTROL ARM WAS STILL THE RIGHT MOVE** - it is what
  established that the sklearn crash predated today's work. The
  lesson is not to stop checking out old commits; it is to
  `git checkout main` rather than `git checkout <sha>` on the way
  back, and to read the branch name before trusting a commit.

- **A SUITE WRITTEN TO SURVIVE THE TEAM BUNDLE FAILED IN IT, AND
  THE RECORD IT VERIFIES IS THE THING THAT DOES NOT TRAVEL
  (2026-11-08).** `smoke_competitive` checks that every figure
  the comparison page states appears in this project's own
  record - and the bundle ships a SEVEN-LINE STUB in place of
  CONVENTIONS.md, deliberately, because the lab notebook stays in
  the internal repo. So inside the bundle the suite reported
  **44 figures as unrecorded** and went red about nothing, on the
  one tree a teammate actually runs. This file already records
  the lesson for this exact suite - "a suite that cannot run
  inside the filtered bundle is a suite the team cannot run at
  all" - and the previous fix was to stop reading the sprint
  board; the notebook itself being a stub was the half left.
  SKIPPED with a printed NOTE now, the `gated()` pattern: dev
  tree 19 checks, bundle 16 plus two notes, and the delta
  decomposes.
- **THE STUB IS DETECTED FROM ITS OWN SENTENCE, not from its size
  or its heading count** - a proxy would start lying the day
  somebody writes a short record or a long stub.
- **AND A MARKER TOO BROAD WOULD TURN THE CHECK OFF EVERYWHERE
  AND NOTHING WOULD NOTICE**, which a mutation found: replacing
  the stub's sentence with "CONVENTIONS" matches the REAL record
  too, so the gate skips in the development tree as well and the
  suite stays green having stopped checking the thing it exists
  for. A notebook carrying real `##` sections must NOT match the
  marker, and that turns two checks red.
- **AND MY FIRST GUARD HAD ITS PREMISE ON THE WRONG FILE.**
  `record_text()` concatenates the notebook AND the scorecard,
  and the SCORECARD TRAVELS - so the combined text carries real
  sections even in the bundle, and "this record has sections so
  it must not look like a stub" was true here and wrong there.
  Only running the suite in the filtered tree shows that
  distinction; it is now gated on the notebook alone.
- **AND A DIGEST PIN IS LIBRARY-VERSION DEPENDENT, WHICH BROKE
  THE TEAM'S OWN 3.10 LANE.** The sparse-slice fix was checked by
  pinning a sha of discovery's whole catalogue on a frame with
  nothing to drop - proving the change moves nothing. That digest
  reads `543448fe03994fb4` on pandas 3 and `14465360208811da` on
  pandas 2.3, so the check failed on 3.10 while the engine
  behaved identically. **Assert the DECISION, not a digest of the
  output**: discovery now REPORTS which columns were empty inside
  a target's own rows - a fact about the data an operator would
  want - and the check asserts that the drop fires on the sparse
  frame, names the column, and does NOT fire where there is
  nothing to drop. Version-independent, and green on both lanes.
- **AND THE FIXTURE FOR IT WAS SPARSE-AND-HOPEFUL.** It planted a
  column present on 8% of rows and expected it to empty inside
  some target's slice; at 1,920 rows it never did and the check
  went red reporting zero. The condition is that a predictor is
  absent on exactly the rows where the TARGET is observed, so the
  fixture makes it DISJOINT BY CONSTRUCTION - the two columns
  never co-occur. A fixture must contain the thing the check is
  about, and "sparse" was not specific enough.
- **AND THE MUTATION READ `0 FAIL LINES` ON THE NEWER LANE, WHICH
  IS THE CRASHED-SUITE TRAP.** Removing the drop makes the suite
  CRASH on scikit-learn 1.9 - traceback, exit 1, and no `FAIL:`
  line for a grep to count - while on 3.10, where the library
  tolerates an all-NaN feature, a check goes red on the missing
  report. Both detect it; reading the filtered count alone said
  one of them did not. The verdict comes from the SUCCESS MARKER,
  which is absent in both.

- **THE COMPUTABLE CEILING'S CENTRAL CLAIM IS NOW CHECKED, AND
  THE METRIC WOULD NOT TAKE A NUMPY ARRAY (2026-11-08).** The
  claim is that the exam measures the SOLVER and not the
  particular synthetic file it ran on. Three draws of one
  blueprint with one planted answer: ceiling 0.6051 / 0.5842 /
  0.5861, spread **0.0209** on 900 rows - against the real
  extract's spread of **0.006** across two generated files at
  56,065 rows. The property is pinned and the MAGNITUDE is
  named as an n effect, because a tolerance quoted without the
  row count is the kind of number that gets reused at the wrong
  scale.
- **AND `ceiling_auroc` RAISED ON THE OBVIOUS INPUT.**
  `_check` did `if not scores:` - which on a numpy array raises
  "The truth value of an array with more than one element is
  ambiguous" - so a caller asking for an AUROC got a message
  about truth values from a guard only ever meant to catch an
  empty input. Every existing caller happened to pass a list.
  `len(scores) == 0` accepts both and still refuses an empty
  input by name.
- **AND THE MUTATION CRASHED THE SUITE INSTEAD OF FAILING IT,
  in the check written about that very footgun.** The
  stability check is the FIRST caller to hand the metric an
  array, so restoring the truthiness guard raised there -
  before the array check could run - and the suite exited 1
  with no `FAIL:` line. Guarded; both mutations now give a red
  check.
- **AND MY KEEP-FILE PREDATED THE FIX, so the restore put the
  bug BACK.** I copied `mlmetrics.py` aside before applying the
  fix and then restored from that copy after each mutation -
  leaving the broken version in place and reporting IDENTICAL,
  which was true and useless. This file's own rule is to
  restore from a COPY OF THE WORKING STATE; the working state
  is the one AFTER the fix, and taking the copy first is the
  same mistake as restoring from a commit that does not contain
  it.

- **ONE REAL RUN, EVERY CONSUMER DRIVEN AGAINST IT - AND THE
  SUITE CAUGHT TWO OF ITS OWN GUESSES IN THE FIRST MINUTE
  (2026-11-08, AC-1, `smoke_artifact_contract.py`).** The
  wrong-key fault has happened EIGHT times here and every fix
  was applied case by case AFTER the failure. The shape is
  always the same: module A reads module B's dict with names
  that SOUND right, and the checks cannot see it because they
  build B's dict BY HAND with the same invented names - the
  fixture agrees with the reader and both disagree with the only
  thing that matters. So one suite now writes a REAL run
  (`write_run_artifacts`, with the reconstruction measurement
  from the engine's own method) and drives the gate, the
  determination package, the receipts, the deck, `m0_gate.py`,
  `signoff.py` and two CLIs against it. Every assertion is "it
  found SOMETHING", named, because a reader with the wrong key
  does not crash - it returns an empty table and a reassuring
  page.
- **AND ITS OWN FIRST RUN FAILED TWICE, BOTH ON MY GUESSES.** I
  asked a gate criterion for a `have` field - it carries `name`,
  `ok` and `detail`, with the measured pair in the detail text -
  and I called `build_deck(run, src)` when the signature is
  `(src, run)`. A guessed field name and a guessed argument
  order, inside the suite written to stop guesses. Read the
  contract; do not remember it.
- **AND `build_deck`'s DOCSTRING HAD BEEN WRONG FOR AS LONG AS
  ITS SECOND RETURN VALUE EXISTED.** It opens "Return the deck
  HTML as a string" and returns `(html, stats)`. A new caller
  took the result as html and measured `len(...) == 2` - two
  bytes by its own reckoning - with no error, because `len`
  works on a tuple. A stale RETURN contract is the same fault as
  a wrong field name, and it is worse in one way: the docstring
  actively asserts the wrong thing. Corrected, and the stats are
  now checked too - they carry the suppressed-bin count, which a
  caller must print because a suppression nobody is told about
  reads as data that was not there.

- **THE EIGHTH WRONG-KEY FAULT, AND MY OWN CHECKS COULD NOT SEE
  IT BECAUSE THE FIXTURE SHARED THE INVENTED NAMES (2026-11-08).**
  `determination.assess` read `fidelity.nn_ratio`,
  `fidelity.reconstruction_gap` and
  `blueprint.bounds_are_k_anonymous`. A real neural run writes
  `fidelity.reconstruction` - train / holdout / ratio /
  rows_train / rows_holdout / columns - and a whole
  `blueprint.privacy` block (k / counted_in / numeric_bounds /
  ... / NOT_a_formal_guarantee), and **no `nn_ratio` at all**:
  that number belongs to the attack and the challenger, not to
  this artifact. So `memorization` and `aggregates` read OWED on
  a real run that HELD the evidence, and a determination reader
  would have been told two measurements this project publishes
  did not exist.
  Forty passing checks missed it, because every one built the
  artifacts BY HAND with the names I had invented: the fixture
  agreed with the reader and both disagreed with the only thing
  that matters. This file already records that exact fault for
  the relationships page, where twelve rows of em-dashes shipped
  behind forty green checks - and the fix prescribed there is
  the fix applied here.
- **THE SUITE NOW DRIVES THE REAL WRITER.** It fits a small
  cohort, generates, measures the reconstruction gap with the
  ENGINE'S OWN method, calls `write_run_artifacts`, reads the
  files back off disk and asserts against those. Restoring
  either invented path turns it red immediately. A dict
  assembled in the check would carry whatever shape I guessed,
  which is how the fault got in.
- **AND THE K RULE IS CHECKED FOR ITS WORDING, NOT A NUMBER.**
  `privacy.counted_in` reads "patients, never rows", and a k
  counted in ROWS is refused - one person seen two hundred times
  would otherwise set a published bound alone, which is the
  whole reason the rule is written that way. An older artifact
  carrying the old flag still counts, because absent is not the
  same fact as false.
- **AND IT WAS FOUND BY RUNNING THE TOOL ON REAL ARTIFACTS, not
  by any check.** `synthkit determination` against the real
  extract's run read six of eleven elements present with
  linkage, attribute, memorization, aggregates and residual
  OWED - and two of those five were my own bad paths rather than
  missing evidence. A tool whose first real read is also its
  first honest read is the argument for running it on real
  artifacts before trusting its verdict.

- **A POSITIONAL ARRAY WAS BEING INDEXED BY LABEL, AND IT ONLY
  EVER WORKED BECAUSE NOBODY HAD FILTERED (2026-11-08).**
  `sets.patient_counts` and, before it, `vocabulary` did
  `np.asarray(groups)[present.index]` - and `groups` is aligned
  to the ROWS of `raw` while `present.index` holds LABELS. Those
  are the same thing only when the caller's index runs 0..n-1,
  which every caller's did. The attribute battery grew a
  real-extract path, passed `df[mask]` - a boolean-filtered
  half-cohort whose labels are the original row numbers - and
  the real extract raised **IndexError: index 29758 is out of
  bounds for axis 0 with size 29741**. `get_indexer` converts
  labels to positions and is correct either way.
- **THE FAULT IS OLDER THAN THE FUNCTION IT WAS FOUND IN, which
  is worth saying because the function was extracted TODAY.**
  `git show` at the previous build has the same expression in
  `vocabulary` with `txt.index`. Extracting the loop did not
  introduce it; adding a caller that filters is what reached it.
- **AND NO FIXTURE HERE HAD A NON-CONTIGUOUS INDEX.** Three
  shapes are pinned now, and only one of them crashes: a HEAD
  slice's labels happen to stay in range, so it passes on the
  broken code. The shape that reaches the fault is a TAIL slice,
  whose labels exceed its own length. A fixture must contain the
  thing the check is about, and "a filtered frame" was not
  specific enough.
- **AND THE CHECK HAD TO BE WRAPPED TWICE BEFORE IT COULD
  FAIL.** First the call raised, so the mutation crashed the
  suite - a traceback and no `FAIL:` line, which this file
  records as indistinguishable from a check that never ran. Then
  with the call caught, the CONDITION still dereferenced the
  None before its own guard - Python evaluates `and` left to
  right - so the suite printed ONE failure and died where three
  were owed. Every reference guarded, in the condition as well
  as the label, and the mutation now gives three red checks and
  exit 1.


- **A CHECK SUITE REPORTED SUCCESS WITH A FAILURE IN HAND, AND
  SIXTEEN CHECKS HAD BEEN UNENFORCED FOR AS LONG AS THEY
  EXISTED (2026-11-07).** `smoke_latent` carried
  `if FAIL: sys.exit(1)` MID-FILE, with sixteen `check()` calls
  below it and an unconditional
  `print("All N checks passed.")` at the end - so a red check
  in those blocks printed the success marker and exited 0, and
  the net judges on exactly that marker plus the exit code.
  **Both lied at once.** This file already records the shape
  twice - eleven checks vanishing in `smoke_patterndial`, six
  in `smoke_dialmatrix` - and this is the worse form: those
  made checks DISAPPEAR, this made a FAILURE read as a green
  net. Found only because a check pinning a since-fixed cost
  went red and the suite announced success anyway. The verdict
  is at the END now, and the fail-first confirmed it: with the
  red check still in place the suite reports
  `1 of 136 checks failed.` where it used to report
  `All 135 checks passed.`
- **AND THE RED CHECK WAS ONE PUNISHING PROGRESS, THE FOURTH
  THIS MONTH.** It asserted that a within-patient COUNTER is
  HARMED by the trajectory flag - pinned deliberately so the
  cost could not be forgotten - and the counter fix removed the
  cost, so the check went red on an improvement. Rewritten to
  pin the NEW property (the counter is exact in BOTH arms, and
  the engine DETECTED it rather than being told its name)
  rather than deleted, because a counter is exactly the column
  a future change to the trajectory could break again.

- Run `python scripts/run_all_smokes.py` before claiming anything works. Expect **90 suites, 2784 checks**, ALL GREEN **on a checkout** (measured 2026-11-03). Off a zipball extract - which is what the data machine runs - the total is TWELVE LOWER, and the gap is now MEASURED rather than estimated (2026-11-02: 2480 on a checkout against **2468** on the data machine at the same commit, ALL GREEN both ways). It decomposes exactly, with no residue: nine in `smoke_buildid`, which needs git to test the archive path; one for `node`, absent, so the full JS parse cannot run; two for Chrome, absent, so the in-browser load check cannot run. All three report NOTES and pass rather than failing - that is the design, and it is why the count moves. This entry previously said NINE LOWER and warned that a total nobody ran is not a number to quote; somebody has now run it. CONFIRMED AT A SECOND COMMIT the same day: 2483 against **2471**, the same delta of twelve - and that pair is the stronger evidence, because the commit between them ADDED THREE checks on the dev side and all three appeared on BOTH machines. That is `gated()` doing exactly what its docstring promises: a skipped check still emits a line, so the count stays one number across both places and only the genuinely unavailable tooling moves it. Do not quote the checkout number to the data machine; that is how a correct run gets read as a failure.
- **THE DEVELOPMENT MACHINE WAS BEHIND THE DATA MACHINE, and that is
  how a green suite here failed there.** Dev was on Python 3.10 with
  pandas 2.3; the data machine installs fresh and got pandas 3.0.5,
  numpy 2.5, scipy 1.18. Writing a string into a float64 column is a
  FutureWarning on 2.3 and a **TypeError** on 3.0, so `smoke_long`
  passed here and crashed there - on the one machine where a crash
  costs a round trip. Reproduced by building a 3.13 venv and running
  the whole net against it; that was the only code incompatibility,
  and CI now runs the net on 3.10, 3.13 AND 3.14 so the newest library
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
- **THREE REAL ARMS ON ONE RULER, AND THE ROW-CHOOSING PASS IS THE
  ONLY ONE THAT IMPROVES BOTH AXES (2026-11-12, run_sw_ctl /
  _guard / _rows).** Same source, same seed 11, one flag apart,
  every arm on a build whose off-path is verified unchanged by
  diff:
      arm                           pq vs its grid   identity broken
      control                               0.3255            90.1%
      --copy-coverage-guard                 0.0254            97.9%
      --copy-presence-from-source           0.0847            59.0%
  **The control already breaks the identity on 90.1% of rows**, so
  the copy was never delivering the equality in the shipping
  engine - the bound-pinning pass clamps the copied column to its
  own cap, which is the second defect recorded above. That reframes
  both fixes: the guard buys the best marginal and makes a
  already-broken identity worse, while choosing the rows improves
  the marginal 3.8-fold AND repairs a third of the identity.
- **SO THE FIXTURE OVER-PREDICTED THE ROW-CHOOSING PASS AND THE
  REAL ARM CORRECTED IT, which is the sixth time that has
  happened here.** On the fixture it read 0.0078 with the identity
  whole; on the extract it reads 0.0847 with the identity broken on
  59%. The ORDERING of the three arms is what transferred, not the
  magnitudes - and the ordering is what the decision rests on.
- **AND WHAT REMAINS ON THAT COLUMN IS NOW MOSTLY THE CONTRACT'S.**
  After the fix the split is 0.0774 against the source and 0.0847
  against the output, so the published grid's own error is no
  longer the smaller half. Further gain there is a blueprint
  question rather than a sampler one, which is a different and
  smaller fix than the one just made.
- **BOTH STAY OFF AND BOTH STAY AVAILABLE.** The row-choosing pass
  is the one to reach for - better on both axes than shipping
  today - and the guard remains the honest answer where two columns
  do not share a grid shape. Neither is a default on one seed of
  one extract.

- **IT DID NOT HAVE TO BE A TRADE, AND THE THIRD ARM KEEPS BOTH
  (2026-11-12, `--copy-presence-from-source`).** A column carried by
  an `==` constraint EQUALS its source wherever it is present - so
  its own published marginal IS the source's restricted to those
  rows, copying is right, and the thing to reproduce is WHICH rows
  are present. A flat mask reproduces the SOURCE's distribution
  instead, which is the whole fault. The keep probability for a row
  is the ratio of the two published densities at the copied value,
  normalised to the column's own coverage - the shape
  `_presence_p` already uses, with the response DERIVED from the
  grids rather than published.
      arm                           sparse gap   identity   coverage
      copy with a flat mask             0.4449      48.3%     0.1834
      refuse the copy (the guard)       0.0372       1.6%     0.1986
      choose the rows                   0.0078     100.0%     0.1777
  **Better on the marginal than refusing the copy AND the identity
  is whole**, with the coverage held.
- **AND IT RESOLVES THE PIN CONFLICT WITHOUT AIMING AT IT.** The
  baseline's identity reads 48.3% rather than 100% because the
  bound-pinning pass clamps the copied column to its OWN published
  cap - the second defect recorded above. Choosing the rows by the
  density ratio leaves only values the sparse column's own grid
  covers, so the pin has nothing to clamp and the equality
  survives. A fix that removes the need for a second fix is worth
  noticing; it was not designed for that.
- **THE DENSITY RATIO IS A PURE FUNCTION ON PLANTED GRIDS**, so the
  branch that matters is tested without a generate: the keep
  probability is ZERO where the sparse column publishes no mass - a
  row whose copied value cannot occur in that column must not be
  present - positive where it does, averages to the coverage asked
  for, and an unusable grid returns NOTHING rather than a mask
  invented from it. Three mutations each turn it red: a flat mask,
  the ratio inverted, and skipping the renormalisation.
- **OFF BY DEFAULT, and the real arm is the next run.** The guard
  stays too: it is the honest answer where the two columns do not
  share a grid shape, and an operator should be shown the trade
  rather than have it decided by a default.

- **THE PAIRED REAL ARM RAN, AND THE GUARD IS A TRADE RATHER THAN A
  WIN (2026-11-12, run_sw_ctl against run_sw_guard).** Same source,
  same seed 11, same build for the arm and a control verified
  unchanged by diff - every change between the two commits is a new
  constant, a defaulted parameter, report-only recording, or gated
  behind the flag, so no RNG or value moves with it off.
      criterion                       control   --copy-coverage-guard
      procedure_quantity vs its grid   0.3255          0.0254
      procedure_quantity verdict         BOTH        CONTRACT
      procedure_count vs its grid      0.0594          0.0594
      numeric columns off their grid    6 of 31        6 of 31
      of those, the SAMPLER's                2              1
  **A thirteen-fold improvement on the worst column on the file**,
  with its neighbour untouched to four decimals and the
  sampler-faulted count halved. What remains on that column is the
  CONTRACT's 0.0756, which is a different and smaller fix.
- **AND THE COST IS IN THE RUN'S OWN NARRATION: the identity breaks
  on 97.9% of rows.** `procedure_count == procedure_quantity broken
  on 10964 rows` - refusing the copy restores the marginal and
  loses the equality the source has. That is the whole trade, and
  neither side of it is obviously right: a released file whose two
  columns should be equal and are not is the same class as a
  patient whose age contradicts their birth year, and a column
  putting 93% of rows on the value 1 where the source puts 53% is
  the largest fidelity gap on the file.
- **SO THE BETTER FIX IS NAMED RATHER THAN BUILT, because it keeps
  both.** If the sparse column equals the dense one wherever it is
  PRESENT, then its published marginal IS the dense column's
  restricted to those rows - so copying is right and the thing that
  must be reproduced is WHICH rows are present. The source's rule
  is almost certainly a function of the dense column's own value (a
  quantity is recorded when there were procedures), and this engine
  already models informative presence as a parent plus a response
  curve. Driving a copied column's presence from its COPY SOURCE
  would keep the identity AND land the marginal, where the guard
  keeps one at the cost of the other. NOT MEASURED - it is a design
  with a reason, and this file has a rule about the difference.
- **THE GUARD SHIPS OFF, WITH BOTH NUMBERS ON RECORD.** An operator
  choosing between a faithful marginal and a preserved identity
  should be shown the trade rather than have it decided for them by
  a default, which is the same reasoning as the dial that prints
  requested beside achieved.

- **THE GUARD IS BUILT AND THE FIXTURE REPRODUCES THE REAL
  MAGNITUDE (2026-11-12, `--copy-coverage-guard`).** Two columns
  equal where both are present, one covered on every row and one on
  19%, with the sparse column's own marginal measured on its own
  rows and therefore starting higher:
      arm        sparse vs its OWN grid   dense
      guard off              0.4449       0.0112
      guard ON               0.0372       0.0112
  The real column reads **0.3255**, so the fixture is the same
  fault at the same size. The DENSE column is untouched to twelve
  decimals either way - the guard acts on the copy, not on the
  pair - and the mismatch is REPORTED whether or not the guard
  acted, because a copy that quietly replaces a column's marginal
  is exactly what nobody looks for.
- **AND THE CONTROL THAT KEEPS IT HONEST CAUGHT ME CONFLATING TWO
  FAULTS.** The check that columns of the SAME coverage are still
  copied went red - and the guard was innocent: with DIFFERENT
  published bounds the bound-pinning pass clamps the copied column
  to its own cap, so the equality breaks for a reason that has
  nothing to do with coverage. Measured at **0.49 of rows on BOTH
  arms**, which is what showed the check was testing two things at
  once. The control now uses one marginal for both columns.
- **AND THAT IS A SECOND DEFECT, NAMED NOT CHASED.** `_pin_pass`
  re-asserts each column's own published bounds AFTER the copy is
  re-asserted, so an `==` carried by copying is broken again by
  the pin whenever the two columns publish different bounds. That
  is the `a <= b` and `b <= a` family this file already records -
  two repairs that each own one column, where whichever runs last
  wins. Not fixed tonight; measured, and written down.
- **OFF BY DEFAULT, and the paired real arm decides it.** It
  changes which values a shipped column carries. `peek rerank`
  already says the presence pass touches nothing on this
  blueprint, so the arm worth running is this guard - and its
  predicted effect is on exactly one column, which is what makes
  it cheap to read.

- **THE LARGEST FIDELITY GAP ON THE FILE IS AN `==` CONSTRAINT
  BETWEEN TWO COLUMNS OF DIFFERENT COVERAGE, AND IT IS CONFIRMED ON
  THE REAL EXTRACT (2026-11-12, run_sw_ctl).** Four candidates died
  before this one; this is the measurement rather than a fifth
  guess. The run's own artifacts say:
      peek copies    procedure_quantity := procedure_count
      peek sweep     17 of 19 cyclic columns refined; SKIPPED:
                     gender, procedure_quantity - both for
                     "no marginal draw was kept for it"
      peek rerank    procedure_count      coverage 1.00, no presence
                     procedure_quantity   coverage 0.19, presence
      peek marginals vs source   vs output   verdict
                     procedure_quantity   0.0760   0.3255   BOTH
                     procedure_count      0.0317   0.0594   GENERATION
  **An `==` constraint between a column present on EVERY row and
  one present on a fifth of them is not a whole-column identity.**
  The two are equal where both are present, which clears the 0.999
  bar the detector applies - and copying the dense column wholesale
  hands the sparse one the dense one's distribution, masked
  afterwards by a presence model that selects different rows than
  the source's did. The sparse column is then never drawn from its
  own published marginal at all, which is exactly what the sweep
  reports.
- **AND THAT IS WHY EVERY EARLIER CANDIDATE MISSED.** `peek sizes`
  found nothing because this is a CONSTRAINT copy, not the sized-set
  identity. The unmodelled-parent skip was refuted by the predictor.
  The presence-aware re-rank cannot reach it, because a copied
  column has no draw to re-rank onto - the flag would have touched
  ZERO columns on this blueprint, which is what saved a
  forty-five-minute paired arm.
- **AND THE ENGINE IS NOT BROADLY WRONG ABOUT MARGINALS, which is
  the reassuring half and was never measured before.** Of 31 numeric
  columns, **25 track both their own published grid and their
  source**; six do not, and only **two of those are the sampler's -
  and they are the same column twice**, one copied into the other.
  The other four are the CONTRACT's error, which is a different and
  smaller fix. A whole-engine reading like that is worth having
  before anybody concludes from one column that the sampler is
  unreliable.
- **AND `gender` IS THE SECOND SKIP AND IS NOT A FAULT.** The sweep
  keeps a marginal draw for NUMERIC columns only, so a cyclic
  CATEGORICAL column always reports "no marginal draw was kept for
  it". The reason text reads like a defect there and is not one -
  worth rewording before somebody acts on it.

- **THE REFINEMENT SWEEP CAN SKIP A CYCLIC COLUMN ENTIRELY, AND
  THEN NOTHING RE-RANKS IT (2026-11-12).** The sweep needs EVERY
  parent of some relationship record to be a column the engine
  actually draws. A blueprint can name one it never modelled - the
  token cap publishes more indicators than it draws - and then
  `usable` is empty, the sweep skips the column, and the acyclic
  re-rank skips it too for being cyclic. Measured on a ring
  fixture, ONE PARENT NAME apart:
      parent available    refinements 16   share on the mode 0.6479
      parent unmodelled   refinements 14   share on the mode 0.0000
      the published grid                                     0.6238
  Not a weakened marginal - a DESTROYED one, with the refinements
  count as the observable that proves the skip rather than an
  inference. The post-pass picks those up and the column returns to
  **0.6479**, which is not merely moved but IDENTICAL to the arm
  where the parent was available.
- **AND IT IS NOT WHAT THE REAL COLUMN HAS, which is the third
  candidate to die by measurement.** `peek rerank` reads
  `procedure_quantity` on the shipping run as coverage **0.19**, a
  relationship child, with an informative presence model, **CYCLIC
  and NOT blocked by an unmodelled parent** - so the sweep should
  be refining it, and its output still sits 32 points from its own
  published grid. The flag would re-rank **ZERO** columns on that
  blueprint, so a paired arm there would have measured only the
  seed. That read cost seconds and saved forty-five minutes, which
  is the whole reason the predictor was built before the arm was
  run.
- **SO THE RUN NOW SAYS WHAT THE SWEEP DID, PER COLUMN, rather than
  earning a fourth guess.** `refinements_applied` was a COUNT, and
  a count cannot answer a question about WHICH - this file records
  that for a dropped-edge report and for a twelve-pair difference
  guessed at four times before anybody read the names.
  `refinement_refined` and `refinement_skipped` (column -> reason)
  now ride in the report, and no column can appear in both.
- **AND THE CHECK THAT THEY REACH THE RUN DIRECTORY WENT RED ON MY
  OWN AIM.** I pointed it at a fixture that runs `phase2_pipeline`,
  which writes its OWN `fidelity.json`; the generate report rides
  on `synthkit/pipeline.py`, which assigns the whole dict under
  `generation`. Worth the detour: this project already lost a
  measurement to the opposite arrangement, where `generate`
  recorded `sets_informed` for as long as the routing existed and
  `pipeline` copied one other field out of that report, so
  `fidelity.json` never carried it and reading the key off a run
  returned None - which was read as the routing never firing. **A
  diagnostic that only prints to a terminal on the machine holding
  the data is not readable at all.**
- **AND TWO SIZE SAMPLES AGREED BECAUSE MY OWN WAIT RETURNED
  INSTANTLY.** Polling the launched run, the log read 2209 bytes
  twice and I nearly called it stalled - but the `until` loop
  between the samples was waiting on a net that had ALREADY
  finished, so the two reads were a minute apart rather than
  fifteen. Reading the log's TAIL showed it at 50 of 138 columns
  with 630s left, progressing normally. The recorded rule is to
  read a log twice before calling it stopped; the addition is to
  check that the two reads are actually separated in time.

- **THE ACYCLIC RE-RANK NEVER GOT THE FIX THE CYCLIC SWEEP HAS HAD
  SINCE IT WAS WRITTEN, AND IT COSTS A PARTIALLY-COVERED COLUMN ITS
  MARGINAL (2026-11-11, `--presence-rerank`).** The re-rank inside
  the draw loop runs BEFORE the column's presence mask exists - the
  mask is deliberately held back to the end so a child can read a
  parent on every row - so the published marginal is spread across
  EVERY row while the grid describes the PRESENT rows, which is
  where the source was measured. Where presence and value share a
  driver, the surviving rows are not a fair sample. The cyclic
  sweep already solves this and carries the measurement in its own
  comment (**re-ranked on kept rows +0.630 against +0.242 on all
  rows**); the acyclic path keys on `isfinite` instead.
- **REPRODUCED WITH A CONTROL, which is what makes it the mask
  rather than noise.** Share on the mode against the column's own
  published grid, on a fixture whose presence is driven by the
  column's own driver:
      presence response          off        on
      flat (the control)      +0.0176   +0.0142
      keeps LOW-driver rows   +0.1064   +0.0268
      keeps HIGH-driver rows  -0.0880   -0.0023
      keeps low, steeply      +0.1262   +0.0241
  The flat arm barely moves either way; the other two move by a
  tenth and come back. **AND THE SIGN FOLLOWS WHICH ROWS PRESENCE
  KEEPS** - a one-sided fixture would have shown a repair that
  happens to push one way.
- **AND THE TIE-BREAK TURNED A COST INTO A GAIN, which is the part
  worth keeping.** Re-ranking on the column's OWN values breaks the
  point mass's ties arbitrarily and costs association: rank
  correlation with the driver fell **-0.011 on every one of three
  seeds**. The order the RELATIONSHIP produced is still available
  at draw time, has no ties, and carries the association - keyed on
  that instead, rho goes **0.1228 -> 0.2920, 0.1981 -> 0.3002,
  0.0697 -> 0.2310**. So the in-loop re-rank had been destroying
  association among tied values all along, and this recovers it.
  Reverting the tie-break turns that check red.
- **THE NEIGHBOURING PROPERTIES ARE ASSERTED IN THE SAME
  MEASUREMENT, which this file requires.** Coverage is IDENTICAL to
  twelve decimals - the pass re-ranks values and never decides which
  rows are present. A PATIENT-LEVEL column is skipped, because a
  per-row re-rank would put visit-to-visit variation into a fact
  about the person, which is the eighth ordering bug this function
  already records. A column the contract declares EQUAL to another
  still equals it. And a blueprint with no partially-covered column
  generates **BYTE-IDENTICALLY** with the pass on - the change acts
  only where its mechanism lives, asserted as a digest rather than
  argued.
- **FOUR MUTATIONS, EACH RED ON ITS OWN TARGET**: re-ranking on all
  rows rather than the surviving ones (3 red), tie-breaking on the
  tied marginal values (1), defaulting it ON (1), and dropping the
  patient-level and copied guards (2).
- **OFF BY DEFAULT, and the paired real arm decides it.** Fixture
  evidence alone is not the standard here - five real-extract arms
  have each overturned or traded away a fixture-measured gain. What
  is NOT yet known is whether `procedure_quantity` is even reached
  by this path: its gap is **+0.3255** where this fixture's worst is
  +0.1262, so the mask explains at most a third of it on these
  numbers, and the column may be cyclic rather than acyclic. The
  honest statement is that a real defect was found and fixed, not
  that the named column is fixed.

- **THE BIGGEST FIDELITY GAP SPLITS IN TWO, AND THE GENERATION HALF
  IS FOUR TIMES THE CONTRACT HALF (2026-11-11, `peek marginal` on
  run_tail_ctl_current).** `procedure_quantity` has had two
  incompatible explanations on this record with nothing measuring
  which applies. Asked directly - the source, what the run's OWN
  published grid produces through the engine's own draw, and the
  output, on one ruler whose edges come from the source:
      share at the mode (the value 1)
        source                    0.5282
        the PUBLISHED GRID        0.6029
        the OUTPUT                0.9284
      column mean
        source                    7.2706
        the PUBLISHED GRID        7.2707
        the OUTPUT                2.0591
      worst band gap, grid against source   0.0760
      worst band gap, grid against output   0.3255
      verdict                               BOTH
  Every decisive figure read back SPELLED by the machine. **The
  contract is modestly wrong and generation is badly wrong**, and
  the two take opposite fixes - so the work belongs in the sampler,
  which is the opposite of where my own earlier write-up pointed it
  ("a point mass is not a shape a curve can make" is a contract-side
  framing, and the contract carries 7 points of a 40-point gap).
- **THE GRID MEAN MATCHES THE SOURCE MEAN TO FOUR DECIMALS, which is
  what makes the output's 2.06 a measurement rather than an
  impression.** A published grid that reproduces the source's mean
  exactly and an output at 28% of it cannot both be describing the
  same column.
- **AND IT MEANS NO RE-RANK IS LANDING ON THIS COLUMN.** Both
  re-rank paths exist and have since **2026-08-25** - the acyclic
  one at the draw site, the cyclic one in the refinement sweep - and
  either preserves the marginal BY CONSTRUCTION. An output 32 points
  from its own grid is therefore not a weakened re-rank; it is no
  re-rank. So the recorded sentence "a relationship child that is
  never reached by either keeps whatever curve plus noise produces"
  is right in its CONCLUSION and wrong in its PREMISE: it was
  written as though the mechanism were missing, when the mechanism
  has been in the engine for months and something is stopping it
  reaching this column.
- **THREE CANDIDATES DIED THE SAME HOUR, which is the point of
  asking before building.** It is NOT a set-size partner - the
  column does not appear in `peek sizes` at all, on a report
  confirmed non-empty, so the copy-the-set-length path cannot be
  overwriting it. A plain RING reproduces nothing: every column,
  cyclic and acyclic alike, holds its marginal within **0.0093 to
  0.0447** of its own grid. And a fixture built to the column's
  MEASURED shape - integral, present on 19% of rows, a point mass at
  1, a hub that is child of two ring members and parent of two more -
  drifts **-0.07**, which is the WRONG SIGN: the real column sits
  32 points ABOVE its grid and the fixture lands below it.
- **A WRONG-SIGN REPRODUCTION IS INFORMATION, not a failed
  attempt.** It says the mechanism is not "a cyclic hub with low
  coverage loses its marginal", because that shape loses it the
  other way. What remains, named rather than guessed: something
  OVERWRITING the column after the draw (an identity or a
  constraint repair, neither checked yet on this blueprint), or the
  PRESENCE MASK selecting a biased subset - the acyclic re-rank
  keys on `isfinite`, where the sweep deliberately re-ranks "among
  the rows that will survive", and those are different row sets on
  a column present on a fifth of rows.
- **AND A QUANTILE GRID'S ERROR ON A POINT MASS HAS NO FIXED
  DIRECTION, which is worth knowing before anybody "fixes" the
  grid.** Building the fixture, a grid derived from a source's OWN
  quantiles still lost **10 points of a 60% mode**, because the mass
  between the last knot equal to the mode and the next one is spread
  continuously and only a sliver rounds back. The real column's grid
  goes the other way, OVER-representing its mode by 7.5 points. The
  mean-informed rounding cut is the obvious suspect for the sign,
  and it is a suspect rather than a finding.

- **THE DELIVERABLE'S GATE READS DIFFERENTLY DEPENDING ON WHICH
  QUESTION YOU ASK IT, AND THE STORED ANSWER IS THE FLATTERING ONE
  (2026-11-11, run_tail_ctl).** `m0_gate.py RUNDIR`, the deck and
  the sign-off page all read `fidelity.json` OUT OF THE RUN
  DIRECTORY - so they report what was believed at GENERATION time.
  For any run made before a MEASUREMENT fix that is stale, and on
  this run it is not a rounding difference:
      criterion            stored      re-measured
      pairs graded            130              151
      direction kept          121              132
      close                   113              128
      interaction surfaces    4/7            14/53
      set token shares      71/72            94/96
      dynamics tracked      38/89            35/92
  **The stored reading says `surfaces 4/7 = 57%` where today's code
  measures `14/53 = 26%`**, because the criterion used to grade 7
  of 75 published surfaces and report the proportion of THAT. So
  the governance page accompanying the file would have carried the
  flattering number, which is the fault this project refuses
  everywhere else.
- **THE FULL CURRENT READING, every figure crossed in WORDS by the
  machine because the digits channel off that screen is not
  trustworthy:**
      direction kept        132/151 = 87.4%   need 93.9%   FAIL
      close                 128/151 = 84.8%   need 87.9%   FAIL
      INVERTED                    0           absolute     PASS
      coverage                  42/42                      PASS
      set token shares          94/96                      FAIL
      set EMPTY rate              4/4                      PASS
      shapes tracked              6/9                      FAIL
      interaction surfaces      14/53                      FAIL
      dynamics tracked          35/92                      FAIL
  Three of nine, M0 NOT MET - and the one criterion this project
  calls an absolute stop is CLEAR with no repair. On the 29
  columns a researcher actually receives: 25 keep direction, 24
  are close, ZERO inverted. The source row count read back 55,428,
  which is the provenance guard confirming the right extract
  rather than my reading of a path.
- **AND RE-MEASURING NOW WRITES A DROP-IN DIRECTORY RATHER THAN
  BEING A NUMBER IN A TERMINAL.** `regate.py RUN --src SRC
  --refresh-into NEW` copies the run and swaps one file, so the
  deck, the sign-off page and `m0_gate` all read the current
  measurement with no flag and no special case - the arrangement
  this file already records for a repaired output. The ORIGINAL is
  never modified: its `fidelity.json` is the record of what that
  build measured, and rewriting it would destroy the only evidence
  of what was believed at the time. A mutation that writes into the
  original turns a check red.
- **AND THE REFRESHED FILE STAMPS ITSELF, because otherwise it is
  indistinguishable from a generation-time measurement.** A reader
  comparing it against `provenance.json` would see two different
  builds with nothing explaining why. The stamp names the code, the
  stack and the source. `provenance.json` is left alone on purpose:
  it describes the GENERATION, which is unchanged and still true.
- **APPLIED TO THE DELIVERABLE AND VERIFIED FOUR WAYS
  (2026-11-11, run_tail_ctl_current).** The refresh ran on the real
  extract and `m0_gate` against the new directory now reads
  **132/151, 128/151, 0, 42/42, 94/96, 4/4, 6/9, 14/53, 35/92** -
  identical to what `regate` measured, where the same command
  against the original still read the stored `4/7`. Then:
  `signoff.html` and `deck.html` rebuilt from it and asserted to
  carry `14/53` and `128/151` with the stale figure GONE; the
  ORIGINAL run's `fidelity.json` confirmed to carry no
  `remeasured` stamp, so nothing of record was rewritten; and
  `fc /b` says the two `generated.csv` files are BYTE-IDENTICAL.
- **WHICH IS THE SENTENCE WORTH SAYING TO A RECIPIENT: the DATA
  did not change, the honesty of the numbers describing it did.**
  A correction that moved the output as well would be a different
  claim needing a different check, and the byte comparison is what
  separates the two.

- **AND THE LOAD-BEARING CHECK IS THAT A REFRESH IS NOT A COPY.**
  `copytree` has already put the stale file there, so a writer that
  silently does nothing produces a directory that looks perfect and
  carries the old numbers. The fixture PLANTS a sentinel pair in
  the stale fidelity and requires it GONE afterwards; removing the
  write turns two checks red, where an existence check on the
  directory would have stayed green.

- **THE IMAGE COULD NOT RUN ITS OWN NET, AND THAT IS THE ONE THING
  THE DOCKERFILE SAYS IT IS FOR (2026-11-11).** Built for the first
  time and told to run the net - the command `docs/AWS.md`
  instructs a deployment to run before pointing anything at an
  extract - **TEN of 113 suites failed.** A deployer following the
  documentation would have seen ten red suites and reasonably
  concluded the tool was broken. Causes, in the order they were
  found:
      cause                                      suites
      `docs/` was never COPYed into the image          3
      `requirements.txt`, `Dockerfile`,
        `.gitignore`, `readmit_demo.json` likewise     3
      git is not installed, and the suites CRASHED
        on FileNotFoundError rather than skipping      2
      a repository property asserted where there
        is no repository                               1
      my own per-suite runs used the wrong cwd         2
  **The in-image net now reports ONE failing suite where it
  reported ten**, and the one left is `smoke_gui` - the local
  bench, which is not part of the deployment surface. Its
  campaign-compile endpoint cannot write where it wants under a
  non-root user with only `/work` writable, and its vendor-exam
  arm comes back without a vendor. The first is gated with a
  printed reason; the second is where I stopped, because patching
  a bench suite to pass in a container it is not meant to run in
  is unbounded work aimed at the wrong surface. `synthkit batch`
  is the deployment entrypoint and it is verified end to end.
- **AND THE IMAGE HAD NO "MUST NOT TRAVEL" LIST AT ALL, WHICH IS
  THE SERIOUS ONE.** `make_team_bundle.py` keeps `EXCLUDE_FILES`
  for the filtered export - the lab notebook, the sprint board,
  the runbooks that carry a machine's real paths, the demo
  scripts, and the personal scanner that contains the patterns it
  scans for. **A container that may run in a hospital's own AWS
  account is a SECOND way out of this repo and it honoured none of
  it.** My first fix made it worse: I copied `CONVENTIONS.md`
  straight in before catching it. The image now carries the
  documents minus that list, with the bundle's own one-line stub
  in place of the notebook so the suites skip with a note rather
  than failing - the same mechanism, the same sentence.
- **AND THAT LIST NOW EXISTS IN THREE PLACES, which the code
  already calls a known risk and I have made worse.**
  `make_team_bundle.py` says of its own duplication: "if this list
  changes, that file must change with it - stated here because the
  duplication is a known risk, accepted so the manual path has no
  import step." There are now three copies: the bundle, the
  operator export, and `.dockerignore`. A single source of truth
  is OWED, and adding a fourth without saying so would be the
  quiet kind of debt.
- **ONE WRAPPER BEAT PATCHING CALL SITES, after the crash moved
  twice.** `subprocess.run` raises FileNotFoundError when the
  PROGRAM is absent - a different condition from a tree with no
  `.git`, and the one a container is in. I guarded the probe in
  each suite; the crash moved to the next call. I guarded that;
  it moved again. `smoke_buildid` has SIX git calls and
  `smoke_no_personal` four. A single `_run` wrapper per suite, with
  an absent binary reading as returncode 1 and empty output, fixed
  every site at once - and every caller already handled that as
  "not available", so the suites now SKIP with their notes.
  Whack-a-mole is a signal that the guard is at the wrong level.
- **AND THE FALLBACK MUST NOT BE AN EMPTY LIST, which is why
  `smoke_no_personal` delegates rather than returning `[]`.** Its
  `tracked()` enumerates the files the personal-data scan reads.
  Returning nothing where git is absent would make the one scan
  that exists to stop a record leaving pass trivially - this file
  already records that exact fault in `make_team_bundle`, "an
  empty source tree whose self-scan passes trivially, which is
  the worst kind of clean". It now calls that module's own lister,
  which walks the tree: one decision site about what the tracked
  tree IS.
- **AND TWO OF MY SEVEN "FAILURES" WERE MY OWN CWD.**
  `smoke_hybrid` and `smoke_fidelity` read `readmit_demo.json` and
  `scripts/fidelity_deck.py` by RELATIVE path, and the image's
  WORKDIR is `/work` while the repo is `/opt/synthkit`. Run
  directly they fail; run through the net they do not, because
  `run_all_smokes` already passes `cwd=SCRIPTS.parent`. I had been
  testing suites in a way the net never does and reading the
  result as a defect. The net is the authority on whether the net
  passes.
- **THE DELIVERABLE DECISION IS SETTLED, AND IT GOES THE OPPOSITE
  WAY TO THE CONNECTION NUMBERS (2026-11-11).** The head-to-head
  had the neural engine ahead on relationship DIRECTION by twelve
  pairs, clearing a bar the rules engine fails. Asked the other
  question - how much of each field's natural spread survives,
  against the ceiling the privacy rule leaves - on the same source
  and the same blueprint:
      engine    mean spread retained    fields at the limit
      rules                   0.9972              25 of 31
      neural                  0.5847               7 of 31
  **The neural engine's spread collapse is GENERAL, not the one
  column.** `procedure_quantity` was the symptom that led here -
  maximum 14 against a source 2,656 - and the whole-column reading
  says it retains 58% of the permitted spread across 31 fields
  where the rules engine retains 99.7%, with 7 fields at the limit
  against 25.
- **SO `run_tail_ctl` IS THE FILE, and the reasoning is worth
  keeping because it is not the flattering one.** A researcher
  receiving a table cares more that every field has roughly the
  right spread than that twelve extra SCAFFOLDING relationships
  keep their direction - and this file already records that the
  neural engine's twelve-pair win is earned on the scaffolding,
  with the rules engine slightly AHEAD on the 29 columns a
  researcher actually receives. Both arms carry zero inverted
  relationships, which is the absolute stop, so the choice is made
  on spread and on auditability rather than on safety.
- **AND IT CORRECTS THE EMPHASIS OF MY OWN WRITE-UP HOURS
  EARLIER.** I reported the head-to-head as "the neural engine
  keeps twelve more directions and clears that bar" with the
  tail collapse as a caveat on one column. Measured across all 31
  bounded numerics the caveat is the headline: ahead on
  connections, far behind on distributions. A comparison that
  reports one axis and footnotes the other invites the wrong
  decision, which is the denominator fault wearing a different
  coat.
- **THE DEPLOYMENT CONTAINER HAD NEVER BEEN BUILT, AND BUILDING IT
  VERIFIED THE WHOLE PATH EXCEPT THE ONE CALL THAT NEEDS
  CREDENTIALS (2026-11-11).** `Dockerfile`, `.dockerignore`,
  `cloud.py` and `docs/AWS.md` have existed since 2026-11-04 with
  the honest note that none of it had run against a live AWS
  account. What was NOT said is that the image had never been
  built either - so the doc's central claim, that the net ships
  inside it and a deployment can prove the tool works before it is
  pointed at an extract, rested on nothing. Built and exercised:
      entrypoint `synthkit batch --help`            works
      interpreter inside                            3.14.10
      runs as                                       non-root `synth`
      *.csv anywhere in the image                   NONE
      data/ or run_*/ in the image                  NONE
      AWS env vars, ~/.aws                          NONE
  And the documented command runs end to end on a mounted
  extract: `--dry-run` prints the posture, names exactly what
  would leave, says the source is not on that list and cannot be
  added by a flag, and starts nothing. A real publish then wrote
  **six artifacts, all allow-listed, with the source NOT published
  and key.csv NOT published**. The flags in `docs/AWS.md` match
  the parser exactly.
- **AND THE TOOL PRINTS ITS OWN CAVEAT, which is the part worth
  keeping.** The run says `NOT YET RUN AGAINST A LIVE AWS ACCOUNT
  - this is the code and the guard, verified locally` in its own
  output, so a reviewer reading a transcript cannot mistake a
  local verification for a deployed one.
- **THE IMAGE PINS THE INTERPRETER AND NOT THE LIBRARIES, which is
  the layer this project has actually been bitten by.** Measured:
  the image carries **pandas 3.1.1** where the data machine and
  the development machine both run **3.0.5**, because
  `pip install -e .` resolves fresh at build time. The
  Dockerfile's own comment said "the container and the laptop
  cannot drift" - true of Python, false of the libraries, and
  stated in the one place a deployer would read it. Corrected.
  Pinning them would freeze a stack that stops getting security
  updates, which is the worse trade in a hospital account, so the
  versions are REPORTED instead.
- **AND A RUN RECORDED WHICH COMMIT IT WAS AND NOT WHICH STACK IT
  RAN ON.** `provenance.json` carried `code_fingerprint`, `build`,
  source, settings and privacy - and no interpreter or library
  version at all. Both of this project's version faults were
  diagnosed AFTER the fact by asking a machine what was installed,
  which only works while somebody still has that machine. In a
  container the image IS the environment, so two runs of the same
  commit can differ by a library nobody wrote down. `stack` now
  carries python, pandas, numpy, sklearn and scipy. Removing it
  turns two checks red, and the checks assert the recorded values
  equal the INSTALLED ones - a provenance field nobody verifies is
  the read-but-not-used fault in the one file a reviewer trusts.
- **AND I DIAGNOSED THE CONTAINER NET WRONG TWICE, THE SECOND
  TIME INSIDE THE RETRACTION OF THE FIRST.** I read its log
  twenty seconds after launch, saw NINE LINES and no success
  marker, and called the image's central claim broken. Then I
  "retracted" that by blaming my own `nohup ... &` for killing it.
  Both were wrong: the log later held 165 lines and the container
  was running the whole time - it had simply not got there yet.
  **A GROWING LOG AT LINE NINE IS INDISTINGUISHABLE FROM A CRASHED
  ONE AT LINE NINE**, and the only thing that separates them is
  whether it is still growing. I had written that rule down the
  same morning - "check the log is GROWING before walking away
  from a detached launch" - and then applied its converse,
  declaring a live run dead from a single sample. The tell was
  free and I did not take it: two containers were up from one
  image, which is only possible if the first never died. Read a
  log TWICE before calling it stopped.
- **OPTION 3'S PREMISE WAS WRONG, AND MEASURING IT TOOK TWENTY
  MINUTES INSTEAD OF A BUILD (2026-11-11).** The plan was the one
  this file recorded weeks ago - "MINE interactions the
  challenger's way, publish them as k-screened aggregates the
  blueprint samples from" - importing the neural engine's
  interaction fidelity while keeping the rules engine's posture.
  Three measurements killed the premise before any code was
  written.
  (1) **Discovery is not narrow.** I assumed it tested only the top
  two predictors as a pair where the miner tests all pairs among
  the top four. It searches `preds[:max(shape_top, 2)]`, every
  pair, and takes the best by departure from additivity - verified
  on a fixture where the interacting pair is NOT the top two by
  importance (`y = 10z + 6ab`) and discovery published `['b','a']`
  correctly.
  (2) **The screening bar is not the blocker either.** A pure
  interaction with no main effects, one beside a dominant main
  effect, and one beside a main effect three times larger again -
  all three publish.
  (3) **The extract already publishes 75 surfaces.** Publication is
  not the gap.
- **THE GAP IS APPLICATION, AND IT IS THE SAME SIZE ON BOTH
  ENGINES.** Measured on one ruler, both outputs through the rules
  blueprint:
      engine    surfaces measured   carried   share
      rules                   53        14     26%
      neural                  55        19     35%
  So the neural engine carries five more, and **neither carries
  most of what the contract publishes.** There is nothing to
  import: the rules engine already knows about these surfaces and
  does not apply them.
- **AND THE FAILURES ARE NOT MARGINAL DRIFT, WHICH CORRECTS A
  RECORDED CONCLUSION.** This file says of the surface chapter:
  "the application machinery WORKS ... the other three sit
  0.03-0.14 above the 0.35 bar - marginal drift, parked with the
  ring research, not chased as bugs." That was read off SIX
  measured surfaces, which was all the instrument could then see.
  On the fifty-odd now measurable, the surfaces that fail miss by:
      engine    smallest miss   median miss    bar
      rules              0.37          1.10   0.35
      neural             0.36          1.05   0.35
  **The median miss is three times the bar on both engines.** A
  handful sit just over it, as recorded; the typical one does not.
  "Marginal drift, not a mechanism" was true of the six visible
  then and is false of the population.
- **SO THE WORK MOVES FROM DISCOVERY TO GENERATION, which is a
  better-aimed target than the one asked for.** The contract
  carries the structure and the output does not - the same shape
  this file records for the co-occurrence conditionals, where the
  blueprint published a phi of 0.885 and the draw delivered
  nothing. Nothing about mining, publishing or screening is worth
  changing; the question is why `_apply_numeric` and the surface
  pass leave two thirds of a published surface on the floor. The
  recorded suspect is a pair member trimmed to order the graph,
  which `peek surfaces` reports per surface.
- **THE HEAD-TO-HEAD RAN ON ONE RULER, AND NEITHER ENGINE WINS -
  THEY FAIL IN OPPOSITE DIRECTIONS (2026-11-11, run_tail_ctl
  against run_neural_h2h).** Same source
  (`phase2_800_fix4\tidy_visits.csv`, 55,428 rows, 800 patients),
  same seed 11, same k of 10 - and critically BOTH OUTPUTS PUT
  THROUGH THE SAME BLUEPRINT, the one discovery found on the real
  source, so the 151 pairs are identical pairs rather than two
  engines each graded on their own surface. This file records a
  night lost to a cross-engine comparison over different
  vocabularies; one ruler is the fix.
      criterion              rules      neural     bar
      pairs graded             151         151       -
      direction kept      132 87.4%   144 95.4%     142
      close               128 84.8%   131 86.8%     133
      INVERTED                   0           0       0
      own-column pairs          29          29       -
      own-column close          24          22       -
      own-column INVERTED        0           0       0
  **The neural engine keeps twelve more relationship directions
  and CLEARS that bar where the rules engine fails it**, and takes
  close three pairs further while both still miss. The absolute
  stop holds for both.
- **AND THE RULES ENGINE IS BETTER ON THE COLUMNS A RESEARCHER
  ACTUALLY RECEIVES, which is the finding a headline would
  hide.** Of the 151 pairs, 29 are the operator's own columns and
  122 are set-token scaffolding. On those 29 the rules engine
  holds **24** close against the neural engine's **22**. So the
  neural engine's twelve-pair direction win is earned on the
  scaffolding, and on the clinical columns themselves it is
  slightly behind. This file already records that four fifths of
  the graded surface is scaffolding and that every fidelity arm
  moves only there; the same split decides this comparison.
- **AND ON THE WORST COLUMN THE TWO ENGINES FAIL IN OPPOSITE WAYS,
  which is the most useful thing the comparison produced.**
  `procedure_quantity`, share of rows by band:
      band        source     rules    neural
      = 1         0.5282    0.9284    0.7080
      2 to 24     0.4223    0.0684    0.2920
      25 to 87    0.0396    0.0019    0.0000
      over 87     0.0099    0.0013    0.0000
      maximum      2,656         -        14
      delivered         -    0.4367    0.0285
  The neural engine **recovers most of the missing middle** - 29.2%
  in the 2-to-24 band against the rules engine's 6.8%, where the
  source has 42.2% - and has **NO TAIL AT ALL**: nothing above 24,
  a maximum of 14 against a source maximum of 2,656, and a
  delivered spread of **0.0285** against the rules engine's 0.4367.
  A smoothing autoencoder with a narrow bottleneck cannot emit a
  2,656 outlier, and this file already records that it "smooths".
  So the rules engine has the wrong body and keeps a tail; the
  neural engine has a far better body and no tail. Neither is close
  to this column, and the two defects do not overlap.
- **WHICH IS A WARNING ABOUT PRUNE-AFTER ON THE NEURAL PATH.** The
  generate-then-prune design removes records in SPARSE regions,
  which are the tails. On an engine that already emits no tail,
  pruning compounds the one failure it has rather than paying for
  k out of spare fidelity. The order to consider them in is
  therefore: fix the tail first, prune second - and on the rules
  engine, whose tail exists, the priority is the reverse.
- **AND THE COST IS NOT A ROUNDING ERROR: about four hours against
  twenty-two minutes**, of which roughly two hours was the fit and
  the rest a self-measurement a comparison does not need. If the
  neural engine is a release candidate that belongs beside the
  fidelity numbers, not in a footnote.
- **WHAT THE NEURAL ARM'S OWN PRIVACY READINGS SAY, measured on
  this run of the real extract:** held-out reconstruction
  **0.00557** against **0.00527** on the 47,029 rows it trained on,
  a ratio of **1.06** where a memorising network climbs and the
  recorded positive control reads 9.7; nn-ratio **1.04**; and the
  k-aware blur snapped **20.0% of rows**. The output generalises.
  That 20% is also an independent read on the prune-after
  estimate - this file's measured figure was ~25% of rows, from a
  fixture, and the real extract says a fifth at the sampling stage.
- **AND ITS OWN CONSOLE METRIC IS NOT THE GATE, said before anybody
  quotes it.** The run printed `seed 11: sign 476/520 close
  451/520 INVERTED 11`. That is the challenger's uniform pair
  metric over 520 pairs with its own definition - a different
  denominator from the 151 above, and the INVERTED 11 is not the
  gate's zero. Both are true; say which one is being quoted.
- **GENERATE AT FULL FIDELITY AND THEN PRUNE WHAT BREAKS k IS THE
  RIGHT ORDERING FOR AN OUTPUT-ONLY RELEASE, AND IT IS WHAT MAKES
  THE NEURAL ENGINE AUDITABLE (2026-11-11, design note, nothing
  built).** The operator's proposal: stop constraining the draw,
  generate the most faithful table we can, and remove afterwards
  the synthetic records whose patterns rest on fewer than k
  patients. Four reasons it is sound, one of which was not
  obvious.
  (1) A constraint applied at sampling time distorts EVERY row; a
  filter applied afterwards distorts only the rows it removes.
  (2) It is COUNTABLE. "We generated N, removed M for this stated
  reason" is a sentence a reviewer can check, where a sampling
  constraint is a property of a model they must take on trust.
  (3) **IT CLOSES THE ONE OBJECTION TO THE NEURAL ENGINE.** This
  file records the fork as "latent-informed surfaces, not a second
  engine", and the surviving argument after the weights risk is
  dismissed is that a network cannot produce the enumerable
  screened LIST the rules engine's blueprint provides. A
  post-generation support filter produces exactly that list - a
  per-record decision with a stated threshold - applied to the
  OUTPUT rather than to the contract. The two requests are one
  design.
  (4) It composes with any generator, so the engine choice stops
  being a privacy decision.
- **AND IT IS ONLY VALID BECAUSE WE NEVER SHIP THE MODEL, which is
  the operator's own point and is load-bearing rather than
  incidental.** If a sub-k pattern is learned, it is in the
  weights, and the model would reproduce it given further draws -
  so deleting the instances we happened to draw does not remove
  the capability. For a release that included the model, pruning
  would be cosmetic. For an output-only release the released
  artifact genuinely does not contain the pattern, and pruning is
  sufficient. The architecture rests on the model being destroyed
  after generation.
- **THE LOAD-BEARING DECISION IS THE DEFINITION, NOT THE CODE, AND
  THIS FILE ALREADY MEASURES AN 80-FOLD RANGE.** Asking "which
  records rest on fewer than k patients" three ways gave: support
  in the LATENT space **0.3%** of rows, support in FEATURE space
  **3.8%**, and distinct patients within a RADIUS in feature space
  **25.3%**. The radius version is the correct one and this file
  records why - a fixed 40-neighbour count can never isolate a
  small group, because the list spills into the population and a
  cluster of four patients still reports ten. So the filter must be
  specified before it is built, and the honest working figure is
  that **about a quarter of generated rows would be removed**:
  over-generate by a third and discard.
- **IT NEEDS TWO GATES, NOT ONE, AND SUPPORT IS BLIND TO THE
  SECOND.** Support counts patients NEAR a synthetic record, so a
  record sitting in a DENSE region has high support and survives -
  including a record that is a near-copy of one specific patient in
  that region. Support cannot see that; nearest-neighbour DISTANCE
  can, and this project already measures it as the nn-ratio
  tripwire in aggregate. A prune-after release needs both as
  PER-RECORD gates: a support floor for rare-pattern disclosure and
  a distance floor for individual near-copying.
- **THE COST IS THE TAILS, WHICH IS WHERE THE FIDELITY WE WANT
  LIVES, AND IT COULD BE SELF-DEFEATING.** The rows a support
  filter removes are the ones in sparse regions - the heavy
  utilisers, the extreme values, the rare code combinations. So
  pruning pays for k in exactly the place the current engine
  already pays for it, and a quarter of rows removed from the
  sparse end shifts the released distribution toward its centre.
  The failure mode to measure FIRST is that generate-then-prune
  ends up WORSE on the tails than the rules engine, which at least
  publishes a tail shape and draws from it. That comparison is the
  experiment, and it is not a foregone conclusion either way.
- **AND WHETHER TO PRUNE ROWS OR WHOLE PATIENTS IS A REAL CHOICE.**
  Dropping individual visits from a longitudinal table changes a
  patient's visit count and breaks the course the engine just built
  for them; dropping whole synthetic patients preserves it and
  costs more rows. For a table whose unit of analysis is the
  patient, the second is probably right, and it is the kind of
  decision that is cheap to make now and expensive to retrofit.
- **WHAT CANNOT BE MOVED TO THE END: nothing, for an output-only
  release - which is the surprising part.** Every k-type constraint
  this project applies is a statement about what the released table
  contains, and anything of that form can be checked per row and
  the offending rows removed. The published bounds are the apparent
  exception - they exist so no released VALUE is an individual's -
  but a value past what k patients support is also just a row that
  can be dropped. So the current ordering (screen, then generate)
  is not forced by the privacy model; it is forced by the rules
  engine needing an enumerable contract to generate FROM.
- **THE THIN BUCKET DECOMPOSES, AND HALF OF IT LED TO A MARGINAL
  FAILURE FAR LARGER THAN THE SURFACES (2026-11-11, measured on
  run_tail_ctl).** 22 surfaces landed in `skipped_thin_data`, which
  is one number covering three different findings. Attributed:
      thin, generated side could not fill       11
      thin, source could not be binned at all    8
      thin, both binned but share under 4 cells  3
      thin, source cells removed by the k screen 0
  **Zero are the k rule**, which was my first suspect and is
  refuted: a shortfall privacy explains is not a fault, and here
  privacy explains none of it. Eight are the cohort being unable to
  support a grid for that pair, which is a statement about the
  extract rather than about either table.
- **AND ALL ELEVEN OF THE ENGINE'S REDUCE TO TWO COLUMNS.** Named
  by asking which axis `_axis_bins` refuses on the generated side
  and not on the source:
      procedure_quantity                 53 distinct, source 165
      procedures__has__Oxygen Therapy      1 distinct, source 2
  The second is the published-but-never-emitted gap this file
  already records. The first is the whole of the rest.
- **AND THE CAUSE IS NOT THE TAIL AT ALL, WHICH RETIRES TWO
  HYPOTHESES INCLUDING MY OWN MOST RECENT ONE.** I had predicted a
  pile of rows at the published knot, from the band below it
  interpolating and rounding - measured on a fixture at 86 of 151
  rows. On the real extract there are **TWO** source rows at the
  knot and **ZERO** generated ones. No pile. What is there instead,
  `procedure_quantity` by band:
      band        source    generated
      = 1         0.5282       0.9284
      2 to 24     0.4223       0.0684
      25 to 87    0.0396       0.0019
      over 87     0.0099       0.0013
  **The engine puts 93% of rows on the single value 1 where the
  source puts 53%, and the entire body between 2 and 24 - 42% of
  the source's rows - is almost absent.** That is a marginal
  failure in the BODY, not a tail placement question, and every
  hour spent on `tail_mean_high`, `tail_sd_high` and the banded
  grid was aimed at the 1% above the last knot while 42% of the
  column sat in the wrong place.
- **THE MECHANISM IS PRECEDENTED AND THE EXISTING FIX DOES NOT
  COVER IT.** `procedure_quantity` is a relationship CHILD with
  three parents, and a parent of eleven other relationships - a
  hub. This file already records the class: "A POINT MASS IS NOT A
  SHAPE A CURVE CAN MAKE. Drawn as a relationship child the value
  arrives as a curve prediction plus roughly symmetric noise, which
  is continuous, and rounding cannot land a point mass where the
  source had one." The fix recorded there is `zero_share`, which
  publishes the mass AT ZERO and places it on the lowest-predicted
  rows. This column has **no zeros at all**, so `zero_share` is
  absent and nothing fires - its point mass is at ONE. A fix that
  only knows about zero cannot see a count whose mode is one.
- **AND THE REMEDY IS ALREADY IN THE ENGINE FOR OTHER COLUMNS,
  which is what makes it worth building rather than inventing.**
  The cyclic refinement sweep works by "taking the exact multiset
  the marginal produced and changing only the ARRANGEMENT", so the
  marginal cannot drift; sparse numerics rank-map a decoded channel
  through a published grid for the same reason. A relationship
  child that is never reached by either keeps whatever distribution
  the curve plus noise happens to produce. Rank-mapping such a
  child onto its own published marginal - the relationship decides
  the ORDER, the marginal decides the VALUES - preserves the
  marginal by construction and is a permutation, which is the
  instrument the repairs and the dials already rest on.
  NOT CLAIMED: that it fixes the column. What is measured is the
  defect, its size, and that the engine contains the mechanism for
  the class. It changes how every relationship child is drawn, so
  it needs a flag, fixture evidence on several shapes, and a
  paired real arm before it is a default - five arms this week say
  so.
- **AND THE SURFACES WERE THE ROUTE TO IT, NOT THE PRIZE.** Nine of
  the eleven engine-side thin surfaces are unmeasurable because
  this one column's distribution is too concentrated to bin. The
  measurement work did not improve fidelity by itself and is not
  claimed to; what it did was turn "22 surfaces are thin" into one
  named column with a 42-point marginal gap.
- **THE SURFACE CRITERION WAS MEASURING 7 OF 75 AND READING "4/7",
  AND FIXING IT ALSO EXPOSED 21 PAIRS NOBODY WAS GRADING
  (2026-11-11, re-measured on run_tail_ctl).** `interaction
  surfaces 4/7` has been on the record for months. The run's own
  stored accounting says 75 surfaces published, **7 compared**, 67
  skipped as token columns. So the criterion graded 9% of what it
  claimed and reported the proportion of THAT - a criterion wrong
  in the direction nobody checks, by measuring too LITTLE.
      measurement, same run and source, one commit apart
                              before      after
      surfaces published          75         75
      compared                     7         53
      tracked                      4         14
      skipped, token columns      67          0
      skipped, thin data           1         22
      the criterion reads     4/7 57%   14/53 26%
  **The engine carries a quarter of its published two-variable
  surfaces, not a half.** Nothing was generated differently; this
  is what the number always was.
- **AND THE PAIR SURFACE MOVED WITH IT, WHICH NOBODY ASKED FOR AND
  IS THE LARGER FINDING.**
      pairs graded              130        151
      close                 113 86.9%   128 84.8%
      direction kept        121 93.1%   132 87.4%
      INVERTED                     0          0
      own-column pairs            29         29
      own-column close            24         24
      own-column inverted          0          0
  Twenty-one pairs became comparable, and they are far worse than
  the surface that was already being graded: **15 of 21 close (71%
  against 86.9%) and 11 of 21 keeping direction (52% against
  93.1%)**. The absolute stop holds at zero on both counts, and the
  operator's own 29 columns are IDENTICAL - every point of movement
  is scaffolding, which is the pattern this file already records
  for five fidelity arms.
- **SO EVERY RECORDED close AND direction PERCENTAGE ON THIS
  EXTRACT IS NOW OVER A DIFFERENT DENOMINATOR.** 130 became 151.
  The absolute counts are the comparison, not the percentages -
  this file has twice recorded a conclusion drawn from a
  percentage whose denominator moved, and this is the third
  occasion. It also moves the distance to the bars: close needs
  133 of 151 and has 128 (five pairs, not two), and direction
  needs 142 and has 132 (ten pairs, not two).
- **FOUR CAUSES, AND EACH ALONE WOULD HAVE LEFT THE OTHERS
  POINTLESS.** (1) `measure_surfaces` refused on the NAME before
  asking whether the column was present - and `compare` hands it
  the EXPANDED frames, so indicators the blueprint's own
  vocabulary had rebuilt were thrown away beside the ones it could
  not. (2) `blueprint_vocab` collected the tokens `child` and
  `parents` name but not the ones `evidence.interaction.pair`
  names - a third list, because `pair` is the top two by
  IMPORTANCE where `parents` is the filtered order, which this
  file already records costing a surface measured on (a, b) and
  applied to (c, a). (3) A 0/1 indicator cannot yield three
  quantile edges, so every surface with a binary axis fell to
  `skipped_thin_data` even with its columns present; removing the
  refusal alone would have moved 54 surfaces from one skip bucket
  to another and changed nothing. The two-valued rule now lives in
  ONE function that `shapes._grid_for` and
  `curvecheck._axis_bins` both call.
- **AND THE FOURTH IS THE WORST: ONE PREDICATE GAVE TWO ANSWERS
  FOR ONE COLUMN.** `looks_like_set` reads the frame in front of
  it, and what decides it is `_separator`, which needs the
  separator in at least `SEP_SHARE` of the FILLED cells. A
  generated set column holds fewer multi-token cells than its
  source, so the two land either side of that bar:
      procedures, share of filled cells holding a ";"
        source      4,352 of 10,697        0.4068
        generated   2,854 of 11,209        0.2546
      SEP_SHARE                            0.30
  So the SOURCE frame was expanded into indicators and the
  GENERATED frame was not, and every measurement over those
  columns compared a column against nothing - silently, because
  the column was never built to be counted as absent. That one
  classification carried 23 of the 75 surfaces AND all 21 of the
  pairs. The fix is that the CONTRACT decides what a column is:
  when the caller holds a blueprint, the blueprint has already
  declared the column a set and the frame's own thinness must not
  overrule it. Same rule as the TVD bin edges, which come from the
  source and are applied to both tables.
- **THE k SCREEN IS UNTOUCHED, ASSERTED RATHER THAN CLAIMED.**
  `declared` says what the column IS and never which tokens may be
  released; a token held by fewer than k patients is still not
  published. A flag that loosened that would be a privacy change
  wearing a measurement fix.
- **AND THE 22 THAT REMAIN ARE NOW AN HONEST REASON RATHER THAN AN
  INSTRUMENT GAP.** Token-column skips went 67 to ZERO: every
  published surface's columns can now be rebuilt. The 22 land in
  `skipped_thin_data`, which says the OUTPUT holds too few of
  those tokens to fill a grid - a statement about the engine, and
  the next thing to look at.
- **AND MY OWN FIRST WRITE-UP NAMED THE WRONG QUANTITY.** I
  recorded `procedures` as averaging 0.373 tokens per row in the
  source against 0.277 in the output. That is the mean over ALL
  rows; the predicate averages over FILLED rows and in fact turns
  on the separator SHARE. Both numbers were real and neither was
  the one the code tests. Read the predicate before explaining
  what it did.
- **BOTH TAIL FLAGS RAN AS PAIRED REAL-EXTRACT ARMS, AND THE ONE
  THE FIXTURE CALLED A LARGE GAIN COSTS A THIRD OF THE COLUMN IT
  WAS BUILT FOR (2026-11-10, run_tail_ctl / _spread / _bands).**
  Same source (`phase2_800_fix4\tidy_visits.csv`, confirmed at
  55,428 rows x 44 columns over 800 patients), same tree at one
  commit, same seed 11, three arms ONE FLAG apart, run within two
  hours by the same hand. Every figure below was read back SPELLED
  by the machine.
      criterion              control   --tail-spread  --tail-band-knots 4
      direction kept        121/130      122/130        121/130
      close                 113/130      113/130        113/130
      inverted                0 PASS       0 PASS         0 PASS
      coverage                42/42        42/42          42/42
      set token shares        71/72        71/72          71/72
      set EMPTY rate            4/4          4/4            4/4
      shapes tracked            6/9          6/9           8/10
      interaction surfaces      4/7          4/7            4/7
      dynamics tracked        38/89        38/89          38/89
      mean delivered         0.9972       0.9976         0.9551
      at the ceiling, of 31      25           25             23
      procedure_quantity     0.4367       0.4367         0.2954
- **THE CONTROL REPRODUCES THE RECORDED BASELINE EXACTLY, which is
  what makes the other two arms worth reading.** Mean delivered
  **0.9972**, **25 of 31** columns at the ceiling and
  `procedure_quantity` at **0.4367** against a recorded 0.437 -
  three independent figures landing on the record from a fresh run
  on a fresh tree. An arm whose control does not reproduce is an
  arm measuring something else.
- **`--tail-spread` IS INERT EXACTLY WHERE IT WAS AIMED.**
  `procedure_quantity` - the column the whole mechanism was built
  for - reads **0.4367 on both arms, identical to four decimals**,
  because the real extract is one of the shapes where the smooth
  power curve ALREADY exceeds the k-safe floor and the solve
  correctly declines. The fixture predicted that refusal on 5 of 8
  cells; the extract is one of the five. What the flag does buy is
  **one direction pair** (121 to 122) and mean delivered +0.0004,
  from FOUR columns moving by 0.001 to 0.006 - and three of those
  four were already ABOVE 1.0, so moving them up is not obviously
  an improvement. One pair is smaller than this criterion's own
  recorded seed spread, so it is not a gain anybody should bank.
  The flag DID execute: the run printed its line, which is the
  read-but-not-used guard satisfied on real data.
- **`--tail-band-knots` IS REFUTED, AND IT IS THE FIFTH FIXTURE
  GAIN A REAL ARM HAS OVERTURNED.** It FIRED - the published grid
  for `procedure_quantity` goes from 11 knots to 14, so the top
  segment on an 800-patient cohort does hold enough patients to
  band, which was the open question. It buys **two effect curves**
  (shapes tracked 6 of 9 to 8 of 10 - the DENOMINATOR moved, so 6
  against 8 absolute is the comparison) and nothing else on the
  gate. It costs **mean delivered 0.9972 to 0.9551**, two columns
  off the ceiling, and **`procedure_quantity` 0.4367 to 0.2954** -
  a third of the delivered spread of the one column it exists for.
  On a fixture the same flag was the largest single gain measured
  on that column (mean absolute delivered error 0.345 to 0.168,
  better on 4 of 4 seeds).
- **AND THE MECHANISM IS VISIBLE IN THE CONTRACT, not inferred
  from the outcome.** The published aggregates for that column:
      arm        last knot      cap    published tail mean
      control         87.0    983.0                  212.4
      bands          238.3    983.0                  507.4
  The cap does not move, which is the claim that nothing new is
  disclosed, and it holds. But every inserted knot is a
  patient-weighted BAND MEAN, and a patient-weighted figure sits
  BELOW the row distribution it replaces - so adding resolution
  between 87 and 238 describes that region with values lower than
  the rows actually there, and the region flattens. Named as the
  candidate mechanism rather than claimed: what is measured is the
  grid, the aggregates and the cost.
- **SO BOTH SHIP OFF, AND THE SCREEN FOR THE NEXT ONE IS ALREADY
  ON FILE.** This file's own test - does the change ADD a
  mechanism, or REARRANGE what the draw already does - puts both
  of these in the rearrange column, and that is now five arms
  against four. `--tail-spread` stays available because its
  refusal is correct and its cost is zero; `--tail-band-knots`
  stays available because a cohort whose tail is held by more
  patients may read differently, and because it is the only thing
  that has moved `shapes tracked` on this extract.
- **AND A NUMERIC NONCE CANNOT VERIFY A READ IS FRESH ON A CHANNEL
  THAT MANGLES DIGITS.** Polling a long run, I echoed a random
  10-digit nonce and required it back - and it came back with one
  digit flipped, which is the exact failure the nonce exists to
  rule out. A random 8-LETTER nonce then came back with `I` read
  as `T`. The working form is a SHORT nonce from a visually
  unambiguous alphabet compared with a one-character tolerance: a
  stale read returns a completely different string or none, so
  fuzzy matching still separates fresh from stale while surviving
  the channel. Worth having, because one poll did return the
  PREVIOUS command's output and only the nonce caught it.
- **A PUBLISHED TAIL AGGREGATE WAS MEASURED ON ONE SEGMENT AND
  CONSUMED ON ANOTHER, AND NOTHING COULD SEE IT BECAUSE THE
  CAPABILITY THAT REACHES THE FAULT SHIPS OFF (2026-11-10).**
  `tail_mean_high` was measured at `inner[-1]` - the 99th
  percentile - and `_shape_tail` consumes it at `v[-2]`. Those are
  the same value on every blueprint this project has written, so
  the fault was invisible; `TAIL_BAND_KNOTS` inserts a knot and
  they stop being the same. With four bands the engine bent a
  segment whose own clipped mean is **381.43** to average
  **102.24**, and `delivered` fell from **0.65-0.80 to 0.19-0.27**
  on four paired seeds. Both aggregates are now read off the grid
  AS IT STANDS, which is byte-identical at the default by
  construction, and reverting to `inner[-1]` turns three checks
  red.
- **AND THE CAPABILITY IS NOT DEAD CODE, WHICH RETRACTS WHAT I
  WROTE ABOUT IT HOURS EARLIER.** I recorded that `_tail_bands`
  "measured returning `[]` because the top 1% of rows holds ~1% of
  patients" and was ready to delete it under this file's own rule
  about a guard no mutation can break. It returns **three bands**
  on the heavy fixture - the segment holds 69 patients against a k
  of 10 - so the earlier measurement was of some other shape, and
  deleting it would have removed a working capability on the
  strength of a number I did not re-run. Check whether a thing is
  reachable before calling it dead.
- **AND ONCE THE SEGMENTS AGREE, BANDING IS THE LARGEST SINGLE
  GAIN MEASURED ON THIS COLUMN - WHICH IS WHY THE BUG MATTERED.**
  Paired, four generate seeds, mean |delivered - 1|:
      TAIL_BAND_KNOTS 0    0.345
      TAIL_BAND_KNOTS 4    0.168
  Better on 4 of 4 seeds. It still ships OFF: `delivered` has a
  seed range of 0.38 on this very column, four seeds is not a
  range, and two of the banded readings OVERSHOOT (1.097, 1.085),
  which is a different failure from undershooting and is not
  obviously the better one. The paired real arm decides it, and the
  fixture evidence is now worth the machine time to ask.
- **AND THE OBVIOUS TIDY-UP IS A REGRESSION: THE SPREAD IS CLIPPED
  AT THE BOUND AND THE MEAN MUST NOT BE.** Having clipped the
  spread, clipping the mean for symmetry reads WORSE on **0 of 5
  seeds across three fixtures** - mean |delivered - 1| 0.260 to
  0.390, 0.392 to 0.449, 0.213 to 0.382. The reason is the
  DIRECTION each correction pulls: patient-weighting already puts
  the published mean BELOW the segment's own row mean (65.56
  against 82.14), so clipping compounds two conservative
  corrections, while the unclipped SPREAD sits ABOVE anything a
  placement under the cap can reach (172.88 against 98.28).
  **Clip where the figure is unreachable; leave it where it is
  already conservative.** Both halves of that are asserted, and
  clipping the mean turns five checks red.
- **THE TOP SEGMENT'S SPREAD IS PUBLISHED NOW, AND THE FIRST
  VERSION ASKED THE DRAW FOR A NUMBER NO ARRANGEMENT UNDER THE CAP
  COULD REACH (2026-11-10).** The extreme segment publishes its
  mean, and `_shape_tail` solves a power curve's single parameter
  from that mean - so the spread is whatever that leaves rather
  than anything chosen. Two placements agreeing on the mean differ
  by a factor of two in spread. `_tail_sd` publishes it: one
  aggregate over the same patients the cap already averages.
- **AND THE TARGET HAS TO BE REACHABLE BY THE THING BEING ASKED
  FOR IT.** Measured on the segment AS IT IS, a heavy-tailed count
  reads **172.88** where the same segment CLIPPED at its own
  published cap reads **98.28** - the difference being the handful
  of rows above the bound, which the k rule exists to not publish
  and which generation CANNOT EMIT. Asking for the unclipped figure
  made the solve saturate: it drove the placement to its own limit,
  the **99.9th percentile FELL from 119 to 91**, and the apparent
  gain came from pinning mass at the cap rather than from matching
  a shape. Clipped at the bound it is a floor on the thing the draw
  can actually produce. The mutation removing the clip turns SIX
  checks red.
- **AND A PILE AT THE CAP IS NOT AN ARTIFACT, WHICH IS WHY THE
  PLACEMENT IS ALLOWED ONE.** It was the obvious objection - a
  point mass at exactly the published maximum reads as a defect and
  advertises the bound. Measured instead of argued: **5.8% of the
  clipped source's own top segment sits exactly at the cap**,
  because clipping is what puts it there. The family is a power
  curve over (1 - f) of the segment's mass plus f AT the cap, with
  p re-solved so the published mean is held exactly while f buys
  spread - closed form for both moments, checked against sampling
  the same family to within 0.5.
- **THE REFUSAL IS THE COMMON CASE, NOT AN EDGE ONE, AND IT REFUTES
  THE PREMISE I STARTED FROM.** On **5 of 8 measured cells** the
  smooth power curve ALREADY exceeds the k-safe floor, so the solve
  declines and the output is BYTE-IDENTICAL. "The engine picks too
  smooth a placement" is FALSE for those shapes: at the published
  mean the curve gives 80.88 against a published 79.56. Where it
  binds (3 of 8) delivered spread moves **+0.061, +0.030, +0.003** -
  all positive, none negative - with the segment mean, the column
  mean, the zero mass and the drivers' rank correlations all
  unmoved, the last two IDENTICAL to three decimals.
- **AND THE FLOOR CANNOT BE RAISED IN A k-SAFE WAY, which closed off
  the obvious follow-up.** The gap between the patient-weighted
  spread (79.56) and the row-weighted one (98.28) looked like
  within-patient variation, which could be published as a second
  component and added back. Measured over four seeds, the
  within-patient term is **1.6 to 2.3** and adds **0.03** to the
  total. The gap is the WEIGHTING itself: a patient holding many
  extreme rows counts once. That is the k rule working, and a
  shortfall privacy explains is not a fault.
- **SO `procedure_quantity`'s DIAGNOSIS IS REVISED, AND THE
  DOMINANT TERM IS THE k RULE RATHER THAN THE PLACEMENT.** The
  published `tail_mean_high` is itself patient-weighted - **65.56
  against a clipped row mean of 82.14, which is 80%** - so the
  engine is aiming at a mean a fifth low before any placement
  question arises. The second term is not in the tail at all:
  **86 of the 151 generated rows at or above the knot sit EXACTLY
  AT the knot value**, piled there by the 0.95-to-0.99 band
  interpolating 7 to 11 and rounding, against 103 such rows in the
  source. The shaped rows themselves read mean 52.39 on 65 rows.
  Reading the segment's mean without separating those three is what
  made a placement look like the cause.
- **AND THE k-FLOOR CHECK COULD NOT FAIL, TWICE, FOR TWO DIFFERENT
  REASONS.** It asserted that a segment reaching fewer than k
  patients publishes nothing. A cut leaving ONE patient returns
  nothing anyway, because the spread of a single value is NaN; a
  cut ABOVE the published cap returns nothing too, because every
  value clips to the cap and the spread is exactly zero. Either way
  the k floor is deletable with the check still green - and the
  first two versions of this did one of each. The cut is SEARCHED
  FOR now (below the cap, two to nine patients, spread finite and
  positive), the search is asserted to have found one, and the same
  segment at a k below its own patient count must still publish, so
  the refusal is the floor doing the work rather than the data
  being degenerate.
- **AND THE MUTATION THAT FOUND NOTHING WAS A NON-RESULT, AGAIN.**
  Shortening the fail-first's anchor to `if len(per) < k:` made it
  match TWICE, the assert fired, the file was never modified, and
  the run reported **red=0** against UNMUTATED code - which reads
  exactly like a check that cannot fail. Assert count==1 before
  every replacement, including in the mutation.
- **AND THE PAIRED ARM IS EIGHT GENERATE SEEDS ON ONE BLUEPRINT,
  WHICH IS THE ONLY FORM THIS MEASUREMENT HOLDS IN.** Same source,
  same blueprint, one flag apart, varying only the generate seed:
      TAIL_SPREAD off   delivered 0.421 to 0.801, mean 0.592
      TAIL_SPREAD on    delivered 0.446 to 0.856, mean 0.644
      per-seed delta    +0.022 to +0.111, mean +0.052, 8 of 8
  **The range does not cross zero**, which is the standard this
  file sets, and it is fixture evidence so the flag still ships
  off.
- **BUT THE SAME TABLE SAYS `delivered` CANNOT BE QUOTED AS A
  POINT, AND THAT RETIRES A NUMBER THIS FILE ALREADY CARRIES.** On
  ONE blueprint with nothing changed but the generate seed it runs
  **0.421 to 0.801** - a range **0.38 wide**, which is most of the
  distance to 1.00. The spread of a heavy-tailed column is carried
  by its top 1% of rows, so the statistic's numerator rests on a
  few hundred values under a convex placement. The real extract's
  `procedure_quantity` reading of **0.437 is therefore ONE DRAW**
  and supports no conclusion about the engine; nor does the
  reproduction that "matched" it at 0.469 - two draws of a
  statistic this noisy agreeing by chance. The warning now travels
  with EVERY `k_price` reading, including a healthy one, because
  the reading that gets quoted is the flattering one. Stripping it
  turns two checks red.
- **AND THE REPRODUCTION FIXTURE WAS TOO SMALL, WHICH IS A SECOND
  AND INDEPENDENT REASON ITS AGREEMENT MEANT NOTHING - AND MY FIRST
  WRITE-UP OF THAT COMMITTED THE SAME FAULT IT WAS DESCRIBING.** I
  recorded that at 9,000 rows the realised mean of the placement
  "undershoots its own target by about a fifth, converging by
  36,000 rows", from readings of 0.783 / 1.030 / 1.009 / 1.052 at
  four ROW COUNTS. Adding the two scales in between gives 1.176 and
  0.850, so there is no monotone trend, and the check I wrote to
  pin the convergence went RED - correctly. Measured properly, over
  five GENERATE seeds at each scale:
      shaped rows  71   ratios 1.414 0.922 0.717 1.228 0.750
      shaped rows 344   ratios 0.923 0.892 1.042 1.044 1.015
  **The estimator is UNBIASED and its range falls 0.70 to 0.15
  while the centre stays near 1.0.** Nothing undershoots; 0.783 was
  one draw from a distribution 0.70 wide, and I read a
  size-dependent bias off a subsequence of four points. That is the
  single-seed fault one level below `delivered`, committed in the
  entry written to record the single-seed fault - and the only
  reason it was caught is that a check was written for the claim
  rather than prose.
  What survives: the fixture at 9,000 rows CANNOT resolve a tail
  reading at all, so its agreement with the real extract was
  meaningless, and **the real reading's shortfall still has no
  established cause.** Match the ROW COUNT before reading a tail
  reproduction, and read the segment over SEEDS before reading it
  at all.
- **AND THE ONE MECHANISM THAT WOULD HAVE EXPLAINED A CONSTANT WAS
  REFUTED BY MEASURING IT.** Engine-over-contract read **0.707,
  0.720, 0.698, 0.701** across four seeds - a constant multiplier,
  which this file's own rule says is one mechanism rather than four
  faults. The candidate was within-patient clustering: `pq` carries
  **icc 0.98** over three visits each, so the segment's rows would
  come from a third as many independent draws. Dialled to zero the
  segment mean moves **29.07 to 31.04** - nothing. The constancy is
  the sample mean of a convex transform being below its own
  expectation in a TYPICAL draw, not a mechanism, and it
  disappears at scale.
- **WHAT IS CONFIRMED WORKING, because three candidates died and
  something had to be acquitted:** the engine reproduces **every
  published knot exactly** on this column - 0.0, 0.0, 0.0, 0.0,
  2.0, 3.0, 3.0, 5.0, 7.0, 11.0 and the cap at 551 against a
  published 551.4 - so the draw's uniforms are marginally uniform,
  the grid lands, and the loss is entirely INSIDE the top segment.
  And the segment statistic I first read was diluted by the body:
  **86 of the 151 generated rows at or above the knot sit exactly
  AT the knot value**, rounded there from the 0.95-to-0.99 band,
  so slicing on `> knot` undercounts the shaped rows by about a
  quarter and drags their mean down with it. Slice on the SHAPED
  FLAG, not on a value.
- **OFF BY DEFAULT.** Five real-extract arms this week have each
  overturned or traded away a fixture-measured gain, and the
  evidence here is three fixture cells. `TAIL_SPREAD` is the
  constant, the default is BYTE-IDENTICAL on every shape where the
  floor does not bind, a blueprint predating the field generates
  identically, and the paired real arm decides it.
- **CORRECTION TO YESTERDAY'S `ceiling`: IT IS A REFERENCE, NOT A
  MAXIMUM, AND I CALLED IT ONE (2026-11-10).** `k_price.py` reports
  `ceiling = sd(source clipped to the published bounds) / sd(source)`
  and the entry above describes that as "what ANY generator obeying
  the rule can reach". **That is false.** Clipping compresses the
  tail toward the cap; a generator may instead PLACE mass at the cap,
  which is bimodal and carries more variance. Measured on a
  heavy-tailed count fixture, all three consistent with the same
  published facts:
      source sd                          63.41
      sd if the tail is CLIPPED to cap   16.12
      sd with a SMOOTH tail placement    10.25
      sd with an EXTREMAL placement      20.29
  The extremal arrangement EXCEEDS the clipped reference by a quarter
  while matching the published tail mean to two decimals. So
  `delivered` above 1.0 is legitimate rather than noise - and the
  capacity sweep's 1.073 and 1.189 readings, which I waved off, were
  real. The quantity is still the right reference to report and the
  decomposition still separates the rule's price from the engine's;
  what is withdrawn is the word "maximum".
- **AND THE GAP IT EXPOSES IS A 2x AMBIGUITY THE CONTRACT DOES NOT
  RESOLVE.** The top segment publishes its MEAN (`tail_mean_high`)
  and its cap, and nothing about its spread. Two placements matching
  both differ by a factor of two in sd - 10.25 against 20.29 - and
  the engine takes the smooth one, because `_shape_tail` bends a
  line to hit the mean. On a heavy-tailed column that is the wrong
  half of the ambiguity: the source's tail is a few rows far out and
  many near the knot, which is the extremal shape.
- **SO THE FIX IS ONE MORE PUBLISHED AGGREGATE, NOT MORE KNOTS.** The
  obvious move was a k-banded grid inside the top segment, the way
  sparse numerics already get one. It is IMPOSSIBLE under the k rule:
  the top 1% of ROWS holds about 1% of PATIENTS, so at k=10 there are
  not enough people above the 99th percentile to form two bands - on
  the real extract's 800 patients that segment holds roughly eight.
  `_tail_bands` was written, measured returning nothing for exactly
  that reason, and is kept with the finding in its docstring. What IS
  publishable is a k-screened SPREAD for the segment, one aggregate
  over the same patients the cap already averages, which resolves the
  ambiguity without disclosing anything the cap does not.
- **AND THE SHORTFALL IS ENTIRELY ABOVE THE LAST KNOT, which is what
  makes this worth doing rather than guessing.** On the reproduction
  fixture, source against generated: zeros 0.171 against 0.171,
  median 2 against 2, 90th 5 against 5, 99th 10 against 10 - and then
  99.9th **178 against 40** and max **379 against 107**. The body is
  exact and the whole loss sits in the one segment the contract
  describes with a single number.
- **THE PER-REGION k SCREEN IS REFUTED BEFORE IT WAS BUILT: THE
  ENGINE IS ALREADY AT THE PRIVACY RULE'S CEILING (2026-11-10,
  run_tt_off, `scripts/k_price.py`).** The plan was to make the
  published bounds per-region rather than global, on the reasoning
  that uniform caution costs spread on dense columns - and this file
  records every spread miss on the extract as "bound-explained",
  68% to 96% of a column's magnitude beyond the bound. That is true
  and it is only half the question. It says the rule is responsible
  for a shortfall without saying how much spread was REACHABLE and
  how much of that the engine delivered. Measured on the real
  extract, 31 bounded numeric columns:
      mean ceiling     0.8935   what the RULE still allows
      mean delivered   0.9972   what the ENGINE produced of that
      at the ceiling   25 of 31 columns (delivered >= 0.95)
  All three read back SPELLED by the machine. **The engine produces
  99.7% of the spread the rule leaves available**, so there is
  essentially nothing for a cleverer bound construction to recover:
  the remaining ~10.7% is the posture itself, and moving it is a
  governance decision rather than an engineering one.
- **AND THE DECOMPOSITION IS THE REUSABLE PART.** `ceiling` is
  `sd(source clipped to the published bounds) / sd(source)` - what
  ANY generator obeying the rule can reach, since the bounds are the
  mean of the k most extreme patients. `delivered` is
  `sd(generated) / sd(source clipped)`. Before this there was no way
  to tell "the k rule explains it" from "the k rule explains PART of
  it", and the second is the only one a fidelity change can act on.
- **WHAT SURVIVES IS NARROWER AND BETTER AIMED: SIX NAMED COLUMNS.**
  Six of 31 sit below 0.95, and the two worst are
  `procedure_quantity` at **0.437** and `glasgow_coma_score` at
  **0.450** - the engine delivering under half the spread the rule
  allows, on columns whose ceilings are 0.697 and 0.869. Both are
  COUNT-LIKE or ordinal, which is where this file already records a
  point mass being unreachable by a curve plus symmetric noise.
- **AND `procedure_quantity` CARRIES PART OF THE CLOSE GAP TOO,
  which is what makes it the lead rather than one of six.**
  `close_gap` on the same run names it in THREE of the nine
  remaining misses - against `active_drug_count`, against
  `span_days`, and against `conditions__n`. One column appearing in
  both the spread shortfall and a third of the close misses is a
  concentration worth chasing, and it is ADDITIVE work - how a count
  column's spread is produced - rather than the token-placement
  rearrangement that four real arms have now shown to be near
  zero-sum.
- **AND A CEILING ABOVE 1.0 WOULD MEAN THE MEASURE IS WRONG, WHICH
  THE BRIDGE NEARLY CONVINCED ME OF.** The first real read appeared
  to show 2.869 for `glasgow_coma_score`. Clipping is a CONTRACTION,
  so the variance it leaves cannot exceed the original and a ceiling
  above one is impossible - the contradiction is the signal this
  file relies on, and it was right: the screen had rendered "0." as
  "2.", and the words channel settled it at **0.869**. The invariant
  is a check now at three bound widths, which is cheaper than the
  round trip.
- **AND THE FORCING ARM IS THE FOURTH REFUTATION, WHICH TURNS THE
  PATTERN INTO SOMETHING WORTH PLANNING AROUND (2026-11-10,
  run_fi_on against run_tt_off, seed 11, one flag apart,
  byte-identical at default).**
      criterion            control    --force-implications 0.90
      direction kept       121/130    121/130
      close                113/130    111/130
      inverted             0 PASS     0 PASS
      set token shares     71/72      71/72
  **Two close pairs lost and nothing gained.** Read back spelled:
  "one two one", "one one one". And the tier DID execute - the run
  printed `forcing implications at a published conditional of 0.9 or
  above, capped at each token's own share` at 1733s, so this is a
  measured cost rather than a silent skip.
- **AND IT FITS THE MECHANISM EXACTLY, which is why it is worth
  recording rather than just reverting.** Forcing DISPLACES the
  weakest unforced pick - that is how the set keeps the size the
  published grid drew for it - so forcing is a PLACEMENT change like
  the other three, and it pays the same zero-sum among the 101
  scaffolding pairs. The fixture said 4 to 20, 9 to 19 and 13 to 28
  conditionals landed with no share over the bar; the extract says
  the arrangement that buys those costs two close pairs elsewhere.
      change                 close        real-arm verdict
      co-occurrence draw   106 -> 113     the one clear gain
      token-target tier    113 -> 110     refuted
      passes scaling       113 -> 112     traded for a token share
      forcing tier         113 -> 111     refuted
  FOUR arms now, and the only winner remains the one that ADDED a
  mechanism where none existed. Everything that rearranges existing
  placement has cost more than it gained.
- **SO THE NEXT FIDELITY WORK SHOULD BE SCREENED ON THAT TEST BEFORE
  IT IS BUILT: does this change ADD a mechanism, or rearrange what
  the draw already does?** Four arms is not a law and it is not a
  coincidence either. The three losers were each measured good on
  three fixture seeds with their neighbouring properties checked,
  which is the strongest fixture evidence this project knows how to
  produce - so fixture evidence is not the filter. The filter is the
  question above, and it is free to ask.
- **THE PASSES ARM IS ALSO A TRADE, AND THAT MAKES THREE TOKEN
  CHANGES REVERSED OR TRADED BY THE REAL EXTRACT IN ONE NIGHT
  (2026-11-09, run_ps_on against run_tt_off, seed 11, one flag
  apart, byte-identical at default - verified by hashing a
  cooc-carrying blueprint through both trees).**
      criterion            control      --token-solve-scale 0.5
      direction kept       121/130      120/130
      close                113/130      112/130
      set token shares     71/72 FAIL   72/72 PASS
  Read back spelled: "one two zero", "one one two", and the shares
  line as PASS seven two of seven two. So the scaling fixes the one
  drifting token share - flipping a whole criterion to PASS - and
  costs one close pair and one direction pair. The run also PRINTED
  the setting it used, `token solve: 0.5 extra pass(es) per
  published conditional pair, capped at 224`, which is the
  read-but-not-used guard satisfied on real data.
- **THE PATTERN IS THE FINDING, AND IT IS STRUCTURAL.** Three
  changes tonight, each measured good on fixtures and each paired
  on the real extract at one seed:
      token-target tier    close 113 -> 110   direction 121 -> 122
      passes scaling       close 113 -> 112   direction 121 -> 120
      co-occurrence draw   close 106 -> 113   direction 119 -> 121
  Only the third is a clear gain, and it is the one that ADDED a
  mechanism where none existed rather than re-arranging what was
  already there. **The gate's close criterion counts 130 pairs of
  which 101 are set-token search scaffolding and 29 are the
  operator's own columns** - and on the two arms that traded, the
  own-column figures were IDENTICAL (25 direction, 24 close, 0
  inverted). So every point of movement happened among the
  scaffolding pairs, where one token's placement is another's
  displacement.
- **WHICH MEANS IMPROVING `close` BY MOVING TOKEN PLACEMENT IS NEAR
  ZERO-SUM, and that reframes the remaining work.** A change that
  rearranges which row holds which token takes from one pair to
  give to another; the only changes that have ever won here are
  ADDITIVE - a mechanism that was absent, like the conditional draw,
  or the binary-grid fix that gave a sparse indicator a curve it
  never had. Two more arms would be needed to call this a law
  rather than a pattern across three, and it is written as a
  pattern.
- **AND THE THREE ARMS COST ABOUT FOUR HOURS OF MACHINE TIME AND
  WERE WORTH EVERY MINUTE.** Each of the two refuted changes looked
  good on three fixture seeds with the neighbouring properties
  checked. This file already says a control arm is the only thing
  that has ever overturned a fidelity claim here; tonight it did it
  twice in one session, and both times the fixture evidence was the
  strongest kind this project knows how to produce.
- **THE RULES ENGINE CAN FORCE AN IMPLICATION NOW, AND THE BUDGET IS
  THE WHOLE DESIGN (2026-11-09, `--force-implications`).** The real
  extract's unreachable band is the near-deterministic one -
  `cooc_landing` reads 4,471 published pairs with 2,729 landing, and
  the worst read **published 1.0, achieved 0.0**. A multiplicative
  weight cannot reach a conditional of 1.0; the neural engine has
  forced since 2026-09-30 and the rules engine never did. Forcing
  goes inside `_pick_row`, which both the DRAW and the weight SOLVE
  call, so the solve measures a draw that already includes it.
- **AND MY FIRST CUT LANDED EVERY CONDITIONAL AND WRECKED SEVENTEEN
  MARGINALS.** Measured, not imagined:
      arm                   landed      worst share   over 0.05
      no forcing             4 of 74       0.0089         0
      forcing, no budget    69 of 69       0.3073        17
      forcing with budget   20 of 74       0.0097         0
  I had written in the commit that the marginal restore was FREE
  because the solve calls the same function. **That is false**, and
  the measurement is what said so. The reason is the UNION: a token
  implied by many others is forced wherever ANY of them lands, and
  that union can exceed the token's own published share even when
  each implier fits inside it. No weight can take it back - the
  solve can drive the UNFORCED picks to zero and the forced ones
  remain. This project recorded the same union fault in the neural
  engine's tier and I rediscovered it from the other side.
- **SO FORCING IS CAPPED AT EACH TOKEN'S OWN PUBLISHED SHARE, IN
  ROWS.** Past that the pair falls back to the nudge, so the
  conditional lands on as many rows as the marginal can afford and
  no more - which is the honest arrangement rather than a choice
  between the two. Three seeds: **4 to 20, 9 to 19, 13 to 28**
  landed, with ZERO shares over the 0.05 bar in every arm and the
  mean set size unchanged, because a forced token DISPLACES the
  weakest unforced pick rather than extending the row. Removing the
  budget turns three checks red, both marginal checks among them.
- **AND A SECOND BUDGET TEST WAS WRITTEN, PROVEN REDUNDANT, AND
  REMOVED.** The first fail-first removed the test in the
  candidate-gathering step and ALL ELEVEN CHECKS PASSED - because
  the apply loop already refuses an exhausted token. Byte-identical
  md5 with and without it. A guard no mutation can break is dead
  code implying a protection that lives somewhere else, which is
  why this project removed a redundant campaign guard rather than
  leaving it as an untestable comment. Deleted, with the proof in
  its place.
- **AND THE BROAD-TOKEN CAP IS KEPT AND CHECKED, because this file
  already records what happens without it:** forcing a 0.86-share
  token flipped a guarded NEGATIVE correlation to positive, an
  inversion manufactured by a fix for a close verdict. A token over
  half the rows keeps its own placement, measured at 0.7635 against
  a published 0.80 with the tier on - inside the share bar.
- **AND THE BROAD-TOKEN FIXTURE WAS THE THIRD INFEASIBLE ONE OF THE
  NIGHT.** It set one share to 0.80 and left the rest, so the total
  passed what any row can hold - and the CONTROL arm then missed the
  broad token too, at 0.6886 against 0.80, which I first read as
  the forcing dragging it. Rescaling the rest so the shares still
  sum to the mean set size gave 0.7830 on the control. Three times
  in one session: a frame whose targets cannot all be met at once
  measures nothing but its own infeasibility, and the feasibility
  condition for a set vocabulary is that the shares sum to the mean
  SIZE.
- **OFF BY DEFAULT, AND BYTE-IDENTICAL WHEN OFF.** Asserted through
  `generate`, not argued. The token-target tier was reversed by a
  real arm the same night on fixture evidence of the same strength,
  so the paired real arm decides this one too.
- **THE LANDING DIAGNOSTIC RAN ON THE REAL EXTRACT AND REFUTED MY
  OWN DRAW HYPOTHESIS, THEN NAMED A SHARPER ONE (2026-11-09,
  run_tt_off).** `cooc_landing` on the shipping default's own run:
      published pairs      4,471
      scored               4,471
      not scorable             0
      LANDED within 0.15   2,729   (61%)
      mean absolute miss  0.1735
  Read back spelled by the machine - "four four seven one", "two
  seven two nine", "zero point one seven three five". **Most
  published conditionals DO land**, where my synthetic fixture had
  predicted 7% at 400 pairs. So the solve is not broadly broken at
  real scale, and the fixture that said otherwise was asking 0.9 of
  every pair's feasible bound - far more aggressive than what the
  extract actually publishes.
- **AND THE PAIRS THAT FAIL ARE THE NEAR-DETERMINISTIC ONES, WHICH
  IS EXACTLY THE KIND `close` IS MADE OF.** The worst misses read
  **published 1.0, achieved 0.0** - three ophthalmic drugs against
  one procedure code, a perfect implication landing at nothing. A
  source rank correlation of 0.885 IS a near-perfect implication, so
  a single landing rate of 61% can sit beside a band that lands
  NONE and every pair the gate still misses can live in that band.
  The report is BANDED by published conditional strength now, for
  that reason: "most land" was a true summary and the wrong one.
- **AND THE MECHANISM IS THE ONE THE ENGINE'S OWN COMMENT NAMES.** A
  multiplicative weight cannot reach a conditional of 1.0 - "a lift
  is a nudge; P(b|a) is the thing the draw has to LAND, and a
  near-deterministic implication cannot be reached by multiplying a
  weight - it has to be solved against". The neural engine solved
  this with an IMPLICATION TIER that FORCES b wherever an implying a
  was drawn and then restores b's marginal, and this file records
  both of its pitfalls: a broad token must keep its own placement
  (the cap at share 0.5, after forcing Oral flipped a guarded
  negative sign), and the restore must be able to drain (dropping
  only from rows keeping two or more left a route column's excess
  nowhere to go). The rules engine has no such tier. That is the
  next build, and it has a known design and two known traps.
- **WHAT THIS CLOSES: the DRAW-versus-ADOPTION question, with the
  answer being neither of the two things I guessed.** It is not the
  cap (ranking is by abs(phi), the quantity the grader computes),
  not detection (the pair is published at phi 0.88483), and not the
  solve being broadly starved (61% land). It is that the top band
  cannot be reached by the mechanism in use at all.
- **AND THE BAND CHECK CRASHED THE SUITE INSTEAD OF FAILING IT.**
  The mutation collapsing BANDS to one made a label lookup an
  IndexError, so the suite raised and emitted NO `FAIL:` line - and
  the fail-first reported "0 red", which reads as a check that
  cannot fail. This file records that trap repeatedly and the rule
  it prescribes: a message must not index what its mutation
  removes. Guarded; the same mutation now gives two red checks.
- **THE TOKEN-TARGET TIER WAS REFUTED BY THE REAL EXTRACT, AND THE
  LOSS IS ENTIRELY ON SCAFFOLDING (2026-11-09, run_tt_off against
  run_tt_on, seed 11, one flag apart, same tree).** Three fixture
  seeds had read the within-0.2 count 1/4 to 2/4 with signs 4/4 on
  both arms and the mean miss 0.282 to 0.179. The paired real arm:
      criterion              OFF (shipping)   ON (--solve-token-targets)
      inverted               0 PASS           0 PASS
      direction kept         121/130 93.1%    122/130 93.8%
      close                  113/130 86.9%    110/130 84.6%
      coverage               42/42            42/42
      set EMPTY rate         4/4              4/4
      set token shares       71/72            71/72
      shapes tracked         6/9              6/9
      interaction surfaces   4/7              4/7
      dynamics tracked       38/89            38/89
  **The tier costs THREE close pairs and gains ONE direction pair.**
  Every figure read back spelled by the machine - "one two two",
  "one one zero", "one one three", "one three zero".
- **AND THE DAMAGE IS ALL IN THE SCAFFOLDING, which the gate's own
  breakdown says.** Both arms report the SAME own-column figures -
  29 of the 130 pairs are the operator's columns, and those read 25
  direction, 24 close, 0 inverted on BOTH arms - so the 101
  set-token search pairs carry the entire difference. The tier acts
  on token indicators and makes them worse by three while leaving
  the operator's data untouched.
- **SO IT STAYS OFF, and the control arm is what earned that.** This
  is the THIRD time in this file a fixture-measured fidelity gain
  has been overturned by a real-extract arm at the same seed, and
  the first two cost a night each. A fixture told me the tier fixed
  a point-biserial saturation it had correctly diagnosed; the
  extract says the arrangement it buys costs more elsewhere than it
  gains. Locally right, globally worse - the greedy-cut lesson, in
  a new place.
- **THE MECHANISM IT FIXED IS STILL REAL, which is why it ships as a
  flag rather than being deleted.** The point-biserial bound is
  arithmetic: a two-valued child rank-matched to a continuous target
  reaches only `phi(z)/sqrt(p(1-p))` of it, measured at 0.776
  against a predicted 0.7766 at a 0.35 share. What the real arm
  shows is that landing that target more exactly is not worth what
  the tightened arrangement costs the pairs around it.

- **AND THE TOKEN SOLVE IS ITERATION-STARVED AT A REAL VOCABULARY,
  WHICH REFUTES A CLAIM IN THE ENGINE'S OWN COMMENT (2026-11-09).**
  The remaining close misses are cross-set token pairs whose
  structure IS detected and IS published - sodium-chloride against
  Flush at phi 0.88483, conditional 0.954 - so the draw is the
  suspect. Measured on a FEASIBLE 1,827-token vocabulary with 400
  published pairs:
      passes   conditionals landed   mean|miss|   worst share   >0.05
         24          27 of 400          0.696        0.032         0
         96          55 of 400          0.533        0.053         1
        224          93 of 400          0.372        0.039         0
  Monotone, with the token MARGINALS unharmed in every arm. The
  engine's comment said the solve "plateaus" by 30 passes - measured
  on ONE planted pair, and it does not hold at 400.
- **AND IT IS NOT THE WHOLE CAUSE, said rather than implied.** Even
  224 passes land only 23% of the published conditionals. The
  engine's own comment names the remaining candidate - the set SIZE
  is drawn independently of context - and that is a different fix.
- **SHIPPED OFF, AT A RATE OF ZERO, FOR THE REASON THE TIER ABOVE
  JUST DEMONSTRATED.** Both changes act by moving token placement,
  so a fixture gain here carries the same risk that was realised
  there. `--token-solve-scale` is the opt-in, the default is
  BYTE-IDENTICAL to every run made so far - asserted through
  `generate`, not argued - and the paired real arm decides it.
- **AND MY FIRST FIXTURE FOR IT WAS INFEASIBLE IN AGGREGATE, which
  made the engine look broken at every scale.** It gave 1,827 tokens
  an average share of 0.15, so the shares summed to about 274 tokens
  per row against a set size of at most 12. Every token ran cold,
  1,638 of 1,827 marginals missed by more than 0.05 on BOTH arms,
  and the conditional landing read 1 of 10 even at FORTY tokens -
  which I briefly read as "the solve never works". The asks were
  also individually impossible: **P(b|a) cannot exceed p_b / p_a
  while both marginals hold**, and the first version asked 0.95 of
  every pair regardless. Scaling the shares to sum to the mean set
  size and capping each ask at 0.9 of its feasible bound turned
  10 of 10 at 40 tokens into the honest curve above. A frame whose
  targets cannot all be met at once measures nothing but its own
  infeasibility - the same lesson as the dial work's reachable band.
- **THE REMAINING CLOSE MISSES ARE CROSS-SET TOKEN PAIRS, AND BOTH
  OBVIOUS SUSPECTS ARE ACQUITTED BY READING THE ARTIFACTS
  (2026-11-09, run_tt_off, seed 11).** `close_gap` on the shipping
  default's own run reads **"to clear the 87.9% bar this run needs
  TWO more pairs inside 0.2"** - read back spelled by the machine -
  and the misses group as:
      CROSS-set token pair   has_sodium-chloride <- has_Flush  +0.885
                             has_heparin         <- has_Flush  +0.570
                             has_lactate         <- has_IV     +0.583
                             has_lidocaine       <- has_Topical +0.443
      ordinary numeric       procedure_quantity <- active_drug_count
                             procedure_quantity <- span_days
                             procedure_count <- mean_arterial_pressure
      token vs plain column  has_midazolam <- procedure_count  +0.335
      other scaffolding      procedure_quantity <- conditions__n
  So the dominant group is a DRUG IMPLYING ITS ROUTE, which is the
  shape the co-occurrence tier exists for - and that tier is ON.
- **THE CAP IS NOT THE CAUSE, AND NOR IS DETECTION.** This file
  records the cap-excludes-the-measured-pair fault twice, so it was
  the first suspect. `cooccurrence` ranks candidates by **abs(phi)**
  - the correlation the grader itself computes - before applying
  `COOC_MAX_PER_TOKEN`, so the strongest partner cannot be cut.
  Checked rather than argued, by querying the run's own blueprint:
  the sodium-chloride / Flush pair IS published, in BOTH directions,
  at **phi 0.88483** with conditionals of **0.837** and **0.954**.
  Phi matches the source correlation of +0.885 to three decimals.
  **The contract carries the structure and the output does not**,
  which puts the fault in the DRAW rather than in detection or
  adoption - a different fix from raising a cap or loosening a
  screen, and the kind of thing two reverted tiers were built
  without establishing first.
- **AND THE ENGINE'S OWN COMMENT PREDICTED THIS.** The detector
  records `cond` beside `lift` precisely because "a lift is a nudge;
  P(b|a) is the thing the draw has to LAND, and a near-deterministic
  implication cannot be reached by multiplying a weight - it has to
  be solved against". A conditional of 0.954 is near-deterministic.
  The precedented suspect is the solve not converging at extract
  scale: this file already records the token MARGINAL solve being
  verified at EIGHT tokens and oscillating at the extract's own
  vocabulary, where the ends swapped. The conditional solve was
  likewise developed on fixtures carrying a handful of pairs, and
  the extract publishes hundreds.
- **SO THE NEXT STEP IS A DIAGNOSTIC IN THE RUN DIRECTORY, NOT A
  FOURTH GUESS (`scripts/cooc_landing.py`).** For every published
  pair it measures P(b | a) in the WRITTEN file - the indicators are
  derived from the set column the operator actually receives - and
  compares it against the conditional the blueprint published. The
  verdict is a PURE FUNCTION on planted numbers with three branches:
  most pairs failing to land names the DRAW, most landing names the
  ADOPTION gates (because then the drifting pairs are not the
  adopted ones), and nothing scored is its own answer. Same move as
  `quota_landing`, for the same reason.
- **AND NOT SCORABLE IS KEPT APART FROM MISSED, with its reason.** A
  pair whose driving token the output never emitted has no achieved
  conditional at all, and counting that as a miss would blame the
  draw for something the vocabulary did - which is the
  blocked-versus-unchanged distinction this file already draws for
  the corruption ledger.
- **AND MY OWN FIXTURE ASSUMED THE TWO DIRECTIONS OF ONE PAIR SHARE
  A CONDITIONAL.** It planted P(B|A) = 0.50 and expected P(A|B) to
  be 0.50 as well; B is held by 150 rows of which 50 hold A, so the
  honest value is **one third**. The check caught it. The two
  directions being different numbers is the whole reason both are
  published, and I had written that in the detector's own docstring
  before getting it wrong in a fixture.
- **THE CLOSE CEILING ON THE REAL EXTRACT IS 97.9% TO 100.0%, SO
  SAMPLING NOISE EXPLAINS ALMOST NONE OF THE GAP AND THE BAR IS NOT
  MISCALIBRATED (2026-11-09, `phase2_800\tidy_visits.csv`, 55,428
  rows, 800 patients).** Two disjoint halves BY PATIENT, one in as
  the source and the other as the generated table, through the
  gate's own `_pair_fidelity`:
      split 0   pairs 155   close 155   = 100.0%   inverted 0
      split 1   pairs 146   close 143   =  97.9%   inverted 0
  Every figure read back SPELLED by the machine - "one zero zero
  point zero percent", "nine seven point nine percent" - because
  the digits channel off that screen turns a zero into an at-sign.
  **Nothing is generated here**: both sides are real data, and each
  half holds only FOUR HUNDRED patients, so it carries MORE
  sampling noise than the full-cohort comparison the gate makes.
- **SO THE ~11 TO 13 POINT GAP IS THE ENGINE, AND THAT IS THE
  ANSWER TO "WHAT IS STOPPING US".** The shipping engine reads
  close **113/130 = 86.9%** on this extract at seed 11 against a
  ceiling of 97.9% measured with half the data. Two hypotheses this
  retires, both of which this file had entertained: that the 87.9%
  bar is unreachable on this cohort - it is cleared by a wide
  margin with nothing generated at all - and that the close gap is
  substantially seed and sampling noise. It is not. There is real
  headroom, which means the fidelity work is worth doing rather
  than being a chase after a number the data cannot support.
- **WHAT THE CEILING DOES NOT SAY.** It is a LOWER bound on what is
  achievable, so it cannot be read as a target the engine should
  hit: a generator is not a second half of the same cohort, and
  nothing here says the remaining gap is all addressable. It also
  says nothing about the OTHER criteria - shapes tracked, the
  interaction surfaces or the dynamics - which have their own
  ceilings and were not measured.
- **AND THE FIRST READING OF IT CAME BACK IN BARE DIGITS WHILE THE
  TOOL REPORTED SUCCESS.** `close_ceiling.py` imported a function
  named `sayn`, which does not exist - the module exports `spell`
  and `annotate` - inside a `try/except` that fell back to `str`.
  So every figure in the one measurement built to settle a
  strategic question arrived as an OCR artifact by this project's
  own rule, and nothing said so. That is the banned silent-no-op
  shape: `getattr(obj, name, default)` on a library attribute is
  forbidden here for exactly this reason, and a try/except around
  an IMPORT is the same fault wearing different syntax. The import
  is now unguarded - if the speller cannot be loaded the script
  FAILS rather than quietly producing numbers nobody can trust -
  and reinstating the fallback turns two checks red.
- **AND THE SHIPPING DEFAULT'S OWN REAL-EXTRACT READING IS ON
  RECORD, with two criteria far better than anything this file has
  recorded.** Seed 11, co-occurrence ON, token targets OFF:
      inverted               0        PASS, the absolute stop
      coverage               42/42    PASS
      set EMPTY rate         4/4      PASS
      set token shares       71/72
      direction kept         121/130 = 93.1%   (need 93.9%)
      close                  113/130 = 86.9%   (need 87.9%)
      shapes tracked         6/9
      interaction surfaces   4/7
      dynamics tracked       38/89
  `interaction surfaces` and `dynamics tracked` were read back in
  WORDS - "four / seven" and "three eight / eight nine". Against
  this file's recorded history those are the best readings yet:
  surfaces have read 0/5 on the deliverable and 2/5 on the n8g/n9g
  pair, and dynamics 17/87 and 35/89. **The DENOMINATORS moved, so
  the absolute counts are the comparison** - surfaces 0 or 2 to 4,
  dynamics 17 or 35 to 38 - and these are a different seed and 180-odd
  commits from the deliverable, so this is NOT a paired arm and
  nothing here attributes the movement to any one change.
- **AND IT REPRODUCES THE SEED-11 CO-OCCURRENCE ARM EXACTLY**, at
  130 pairs, 121 direction and 113 close - which is an independent
  confirmation that the flipped default is the arm that was
  measured, rather than a near neighbour of it.
- **A TWO-VALUED CHILD SATURATES THE WAY A TWO-VALUED DRIVER DOES,
  AND ONLY THE DRIVER CASE WAS EVER HANDLED (2026-11-09).**
  `_informative_sets` rearranges the drawn sets so a token indicator
  lands where its parents say, and `_place` builds the ordering
  target as `lam * sys + sqrt(1 - lam^2) * noise` - so that target
  correlates with the systematic at exactly `lam`. Rank-matching a
  BINARY child onto it reaches only `ceiling(p)` of that, where
  `ceiling(p) = phi(z)/sqrt(p(1-p))` is the point-biserial bound for
  the token's own share. So every routed indicator landed
  systematically short, by a factor of how common the token is:
      share   ceiling   achieved/target   bisected
       0.10     0.585         0.562        0.876
       0.20     0.700         0.687        0.966
       0.35     0.777         0.776        0.996
       0.50     0.798         0.797        0.998
  measured on a SYNTHETIC frame at the extract's own row count -
  56,065 rows, nine routed children, independent drivers - and NOT
  on the extract itself, which is a run nobody has made yet. **The plain
  column tracks the PREDICTED ceiling to three decimals - 0.776
  measured against 0.7766 predicted at a 0.35 share** - which is two
  independent routes to one number and is what makes this a
  mechanism rather than an observed fade. Because the factor is a
  smooth function of the share it looked like ordinary drift, and a
  single-share fixture could not have told the two apart.
  `generate.py` already carried this solve for the case where the
  PARENT is two-valued, with a comment naming the saturation. The
  CHILD being two-valued is the mirror image, it is universal for
  token indicators, and it was unhandled.
- **THE FIX ASKS THE BISECTION THAT WAS ALREADY THERE.** Rather than
  computing a ceiling, the existing `corr_at` loop - which MEASURES
  the achieved correlation and raises lam until it lands - is handed
  the correlation the claim implies instead of using that number as
  a blend weight. No distributional assumption, no new formula, and
  where the ask exceeds what the share can carry the bisection
  saturates and the shortfall is REAL: a token on a twentieth of
  rows cannot correlate above about 0.45 with anything, whatever the
  arrangement does.
- **AND THE SOLVE ALONE BOUGHT THREE GAINS WITH ONE INVERSION, which
  is not a trade to take.** Raising lam on the earlier children
  fixes the groups more tightly and starves the last one. Paired on
  a four-token fixture at one seed, one file apart, target +0.548:
      order        solve   achieved                        signs
      skill         off    +0.37 +0.32 +0.24 +0.13          4/4
      skill          ON    +0.55 +0.52 +0.35 -0.02          3/4
      least-free    off    +0.37 +0.21 +0.15 +0.34          4/4
      least-free     ON    +0.55 +0.30 +0.08 +0.55          4/4
  Placing the LEAST-FREE token first - the extreme shares, which
  have the fewest arrangements available to them - removes the sign
  loss on every seed, and is the same reasoning the SIZE rule beside
  it already follows: spend the degrees of freedom where they still
  exist. That ordering is part of this tier rather than a tidy-up,
  and the mutation that removes it turns exactly the sign check red.
  Across three seeds: within the gate's own 0.2 band **1/4 -> 2/4**
  on every seed, mean absolute miss **0.282 -> 0.179**, signs 4/4 on
  BOTH arms.
- **AND TWO LARGER IDEAS WERE MEASURED AND REFUSED FIRST, BOTH
  CHEAPLY.** A WHITENED JOINT SOLVE over all the routed children at
  once - `v = R^-1 q`, the algebra the cross-table work measured as
  six times tighter - reads **2.3x WORSE** here: 0 of 6 children
  within 0.2 against greedy's 6 of 6. The reason is structural and
  worth keeping: a foreign key gives ONE choice per row, so the
  correlations across a link have rank one and whitening is the
  whole game, while a set rearrangement is a FULL PERMUTATION that
  sequential sorts exploit once per child - and k independent
  drivers cannot share one axis, which caps every correlation at
  about 1/sqrt(k). And CAPPING the group refinement, so later
  children stay out of singleton groups, is worse at every child
  count (-0.10 mean) and drives the worst child NEGATIVE,
  manufacturing the one verdict this project calls an absolute stop.
- **AND THE STARVATION IS A COUNTING FACT THAT THE REAL EXTRACT'S
  ROW COUNT ABSORBS.** Each placed child DOUBLES the group count, so
  after k children there are 2^k groups and the permutation is
  exhausted once that passes n. Measured at three row counts:
      rows     k   2^k/rows   mean ratio   worst
      4,000    13      2.05       0.677    0.337
      14,000   13      0.59       0.732    0.553
      56,065   13      0.15       0.761    0.689
      56,065   18      4.68       0.674    0.248
  **At the extract's own row count with its own thirteen routed
  children the ratio is 0.761 against an unstarved ceiling of
  0.775** - so there is no extract-scale starvation at thirteen, and
  the shortfall there is the point-biserial bound rather than the
  arrangement running out of room. The problem I first set out to
  fix did not exist; the one underneath it did.
- **OFF BY DEFAULT, ON FIXTURE EVIDENCE ALONE.** Three fixture seeds
  are not the standard a default meets here - the co-occurrence
  default was earned on three seeds OF THE EXTRACT, and this file
  records a fidelity gain that a real control arm later overturned
  after a whole night of work. `--solve-token-targets` is the
  opt-in, the paired real arms decide it, and the run REPORTS each
  child's asked-for correlation beside what it achieved in place, so
  a tier that silently stops landing is visible from the run's own
  artifacts rather than from a fixture.
- **AND MY FIRST CHECK FOR THE MECHANISM ASSERTED THE BOUND ON EVERY
  CHILD AND WENT RED, CORRECTLY.** The point-biserial bound predicts
  an UNCONSTRAINED placement, so it describes the FIRST-placed child
  (+0.3745 measured against +0.3691 predicted) and every later one
  falls BELOW it as the group starvation compounds - 0.32 of its own
  bound by the fourth. Those are two separate true statements, and
  averaging them into one tolerance hid both.
- **AND THE RUN COULD NOT PROVE THE TIER EXECUTED, BECAUSE THE CLI
  CONSUMED ONE FIELD OF THE GENERATE REPORT AND DROPPED THE REST.**
  `generate` has recorded `sets_informed` for as long as the routing
  existed; `pipeline` read `persistence_solved` out of that report
  and nothing else, so `fidelity.json` never carried it. Reading the
  key off a run therefore returned None - and I read that as **"the
  routing never fired on this fixture"** and nearly rebuilt the
  fixture around a fault that did not exist. It had fired, in BOTH
  arms, with five children. That is the wrong-field fault in its
  quietest form: the field was real, the writer was real, and only
  the PLUMBING between them was missing, so the absence looked like
  a measurement. It reaches `fidelity.json` now, with each child's
  asked-for correlation beside what it achieved in place, and it is
  ABSENT rather than empty when nothing was routed.
- **AND PROPAGATION IS NOT A LEAK, which my own check asserted
  wrongly first.** It demanded that every column except the set be
  byte-identical across the two arms, and went red on `severity` -
  which the blueprint declares a CHILD of two token indicators
  (`severity <- ['drugs__has__t_mid', 'drugs__has__t_rare',
  'load']`). A column drawn FROM an indicator must move when the
  sets are rearranged; that is the relationship working. Which is
  which is now read from the blueprint's own graph rather than
  assumed, the same distinction the dial matrix already draws, and
  separately EVERY column's multiset is asserted unchanged - so
  nothing the tier touches or reaches can drift a published
  marginal.
- **THE CONDITIONAL TOKEN DRAW IS ON BY DEFAULT, EARNED ON THREE
  PAIRED SEEDS OF THE REAL EXTRACT - AND THE SCREEN THAT DECIDED
  IT WAS PHI RATHER THAN LIFT (2026-11-09).** About four fifths of
  the pairs the gate relates on this extract are SET-TOKEN pairs,
  and `_token_weights` solved only each token's MARGINAL share -
  so within a row the tokens were drawn INDEPENDENTLY and a drug
  implying its route had nothing to carry it. The draw now
  reweights each pick by the published co-occurrence of what the
  row already holds, including tokens drawn in EARLIER set
  columns, which is the shape auto-regressive tabular generators
  use over discretised columns. Paired through the full CLI, same
  build, same source, ONE FLAG apart, and the SAME denominator on
  each arm so the counts compare directly:
      seed       pairs   direction        close          own inv
      20260731    125    118 -> 120   102 -> 106  (+4)   0 -> 0
      11          130    119 -> 121   106 -> 113  (+7)   0 -> 0
      37          122    115 -> 117    97 -> 102  (+5)   0 -> 0
  **Every seed gains, none degrades, and the absolute stop holds
  at zero own-column inversions on all six arms.** Mean close
  80.9% to 85.1%, direction 93.4% to 95.0%. This file's oldest
  rule is to report the RANGE and never a point; the range here is
  +4 to +7 and does not cross zero, which is the standard a
  DEFAULT has to meet and which the trajectory flag did not.
- **AND THE FIRST ARM READ IT AS A TRADE, WHICH IS WHY THE SCREEN
  MATTERS.** With a LIFT bar the one-seed arm read close 102 -> 99
  and would have been written up as a cost. Switching the
  publication screen to PHI - the binary-pair correlation the
  grader ACTUALLY computes - is what turned it over. Lift and phi
  agree on strong pairs and on rare ones; they part on a MODERATE
  conditional on a COMMON token, where lift divides by a large
  base and sits near 1 while phi clears the gate's own 0.1 relate
  floor. **Screen on the quantity the criterion measures**, which
  is the same lesson as optimising `within_lag1` rather than the
  two measurements it is derived from.
- **AND THE k FLOOR APPLIES TO THE JOINT, NOT TO THE TOKENS.** Each
  token has already cleared the vocabulary's floor; a PAIR held by
  three patients describes those three people however common each
  token is alone. Pinned on a fixture carrying exactly that shape -
  two tokens on hundreds of patients whose joint is held by a
  handful - because no existing fixture had one.
- **AND A SIZE-COUPLING TIER WAS BUILT ALONGSIDE IT AND REMOVED
  ENTIRELY, because its gain was an artifact of my own fixture.**
  Its baseline came from a blueprint built with
  `discover(df, targets=["x"])`, which EXCLUDES the relationships
  the pipeline finds between the set-size and token-indicator
  scaffolding columns - and the engine already draws those through
  its ordinary relationship machinery. On that restricted
  blueprint the baseline read -0.003 and the tier looked like it
  took a planted implication from absent to +0.419. Through the
  REAL CLI the baseline already carries **+0.497**. A measurement
  that does not go through the real code path is not a
  measurement, and this suite now runs the CLI for that reason.
- **WHAT IS NOT CLAIMED: that close CLEARS ITS BAR.** The best arm
  is 86.9% at seed 11 against 87.9%, so M0's close criterion still
  FAILS on every seed - a four-point gain on a criterion that was
  seven points short. Thirteen of the seventeen close misses on
  the record sit on TOKEN INDICATORS and SET SIZES, which are
  search machinery dropped before `generated.csv` is written, so
  no rearrangement of the written file can reach them and the only
  route to them is the draw. Nor is anything claimed for shapes
  tracked, interaction surfaces or dynamics tracked; nothing here
  touches those.
- **TWO SEEDS NOW, BOTH POSITIVE, AND THE SECOND IS BIGGER THAN
  THE FIRST (2026-11-09 overnight).** The conditional token draw,
  paired through the full CLI - same build, same source, same
  seed, ONE FLAG apart, and the SAME denominator each time so the
  numbers compare directly:
      seed        pairs   direction        close        own inv
      20260731     125    118 -> 120   102 -> 106 (+4)   0 -> 0
      11           130    119 -> 121   106 -> 113 (+7)   0 -> 0
  As percentages: 81.6% to 84.8%, and 81.5% to 86.9%. The absolute
  stop holds at zero on every arm. Seed 37 is the third and is
  running; this project's own rule is to report the RANGE and
  never a point, and two seeds is not a range yet.
- **THE BAR'S OWN PROVENANCE IS WORTH SAYING BESIDE THOSE
  NUMBERS.** 87.9% is an old absolute target - "close at least
  116" when a run related 132 pairs - restated as a proportion
  after the denominator moved. It was never set from a measured
  distribution, and the measured three-seed baseline spread is
  **four points** (79.5 / 81.6 / 83.8). A bar inside its own
  measure's noise is a verdict decided by the draw, which is a
  product decision to revisit rather than something to engineer
  against.

- **THE ENGINE ALREADY CARRIED TOKEN STRUCTURE, BY A ROUTE I SPENT
  AN EVENING REDISCOVERING - AND IT IS WRITTEN DOWN IN ITS OWN
  SOURCE (2026-11-09).** An indicator is DERIVED from the set that
  was drawn, so a relationship whose child is `x__has__t` can never
  be applied to the indicator: `generate` hits
  `if c in derived and derived[c] in out: continue` before the
  relationship path. For weeks those relationships were dropped
  outright. They are now ROUTED to the set column, and
  `_informative_sets` REARRANGES which row gets which drawn set so
  the tokens land where their parents say - the multiset of sets is
  untouched, so no marginal can drift.
  That is what produces the baseline I mistook for nothing: on a
  fixture measured through the real path, `heparin <-> Flush` reads
  **+0.497** and the two set sizes correlate at **+0.873** with no
  count column, no size partner and no set column appearing as a
  relationship child. Nothing I added put it there.
- **WHICH MAKES THE TWO MECHANISMS COMPLEMENTARY IN A PRECISE
  WAY.** `_informative_sets` decides WHICH ROW gets a given drawn
  set, after the draw. The conditional token draw decides WHICH
  TOKENS go into a set, during it. The first cannot create a set
  that pairs two tokens the draw never put together; the second
  cannot place a set on the right row. They compose, and the
  measured increment of adding the second on the real extract is
  **+3 close pairs** - small because the first was already doing
  most of the work, and real because it is measured through the
  full CLI on the extract rather than on a blueprint I restricted.
- **AND THE SIZE CORRELATION COMES FROM THE SAME REARRANGEMENT,
  which is the whole reason an explicit size coupling bought
  nothing.** A row that needs a token gets a set containing it, and
  sets containing more tokens are larger - so the sizes correlate
  as a CONSEQUENCE of the token routing rather than as a thing
  anybody modelled. Measured: the real extract's mean set-size
  correlation is 0.1499 before my feature and 0.1520 after, against
  a source 0.4210. The gap is real and it is NOT closed by
  publishing a size conditional; it is bounded by how well the
  rearrangement can satisfy every token relationship at once.

- **MY FIXTURE BASELINE WAS AN ARTIFACT OF MY OWN RESTRICTED
  DISCOVERY, AND IT OVERSTATED TWO FEATURES BY AN ORDER OF
  MAGNITUDE (2026-11-09, found by checking that a mechanism
  FIRED).** Every fixture measurement of the token work built its
  blueprint with `discover(df, group_by=..., targets=["x"])`. That
  restriction EXCLUDES the relationships the pipeline discovers
  between the SCAFFOLDING columns - and those are exactly where
  this structure already lived. A real run's blueprint carries,
  unprompted:
      drug_routes__has__Flush <- ['active_drugs__n',
          'active_drugs__has__acetaminophen',
          'active_drugs__has__albuterol']
      active_drugs__n <- ['drug_routes__n',
          'drug_routes__has__Oral', 'x']
  So the engine ALREADY draws token co-occurrence and set-size
  coupling through its ordinary relationship machinery.
- **THE SAME FIXTURE, MEASURED THROUGH THE REAL CLI PATH:**
      arm          size corr   heparin/Flush   albuterol/Neb
      source         +0.9130        +1.000         +1.000
      neither        +0.8730        +0.497         +0.445
      size only      +0.8804        +0.503         +0.435
      cooccurrence   +0.8730        +0.502         +0.378
      both           +0.8804        +0.509         +0.422
  The baseline already carries HALF the implication and nearly all
  of the size correlation. The numbers I recorded hours earlier -
  "neither -0.003, cooccurrence +0.419, both +0.833" - are
  WITHDRAWN: they describe a blueprint no run produces.
- **SO THE SET-SIZE COUPLING IS REMOVED ENTIRELY.** On the real
  extract it moved the mean set-size correlation from **0.1499 to
  0.1520** against a source **0.4210** - 0.002 of a 0.27 gap - and
  it introduced **2 scaffolding inversions**. On a fixture through
  the real path it is worth +0.007. It is redundant with machinery
  that already exists, and a flag that buys nothing is a thing to
  delete rather than ship: the detector, the published field, the
  draw hook and the CLI flag are all gone.
- **THE CO-OCCURRENCE DRAW STAYS, AND ITS JUSTIFICATION IS THE
  REAL EXTRACT ALONE.** Paired through the full CLI, same build,
  same seed, one flag: close **102/125 to 106/125**, and on the
  re-measured 136-pair shared surface **110 to 113**. On a fixture
  it adds nothing measurable. A feature whose only evidence is the
  extract is exactly the kind this file says to keep OFF by
  default and measure again - which is where it is.
- **AND THE WAY THIS WAS CAUGHT IS THE REUSABLE PART.** The gate
  numbers alone said "size coupling is neutral". Asking whether
  the MECHANISM had fired - mean set-size correlation, source
  against both arms - said it had barely run at all, which is a
  different fact with a different fix. An arm that bought nothing
  and an arm that did nothing are indistinguishable in a verdict
  table, and this project has a rule about that in both
  directions.

- **THE PHI SCREEN TURNED THE REAL-EXTRACT TRADE INTO A WIN, AND
  THE FIXTURE COULD NOT HAVE TOLD ME (2026-11-09, run_cooc2
  against run_base2).** Same build, same source, same seed
  20260731, ONE FLAG apart, same denominator of 125:
      criterion        run_base2        run_cooc2
      direction kept   119/125          120/125
      close            102/125 = 81.6%  106/125 = 84.8%
      inverted             0 PASS           0 PASS
  **close +4 pairs**, where the same mechanism under the old LIFT
  screen read **-3** (102 to 99). The phi screen is what changed,
  and 84.8% is the best close this engine has measured on this
  extract.
- **AND THE FIXTURE SAID THE TWO SCREENS WERE INTERCHANGEABLE.**
  On every fixture shape tried - strong pairs, rare pairs, the
  contention case - lift and phi published the SAME list, and the
  mutation between them broke nothing until a frame was built
  specifically for a moderate conditional on a common token. The
  real extract is full of exactly that shape, which is why a
  one-line screen change moved seven pairs there and nothing on a
  fixture. The justification was alignment - screen on the
  correlation the grader measures - and it was right for a reason
  the fixtures could not show.
- **SO THE EARLIER TRADE VERDICT IS SUPERSEDED, NOT OVERTURNED.**
  `run_cooc_nov9` really did read close 99 against 102: that arm
  carried the lift screen. The flag stays OFF by default until the
  full decomposition is read - `--set-size-coupling` and both
  together are the remaining arms - because a one-seed paired
  reading is what this file has been burned by, and because the
  bar is 87.9% and 84.8% does not clear it.

- **THE SET SIZES WERE UNCOUPLED, AND THAT IS WHAT BOUNDED THE
  TOKEN DRAW (2026-11-09).** A visit with five drugs has more
  routes than one with a single drug. Measured on a planted
  fixture, the source correlates the two set sizes at **+0.915**
  and the generated file at **-0.003** - absent, not weakened,
  because each set column drew its size from its own published
  distribution with no reference to the other.
  It is also what capped the conditional token draw at ~0.42. Of
  the rows holding a drug whose route is implied with certainty,
  58% carried the route - and **among those that missed it, 60%
  had drawn NO routes at all**. No reweighting of WHICH tokens are
  picked can help a row that is picking none, which is why the
  solve plateaued rather than running out of iterations.
- **THE TWO MECHANISMS ARE COMPLEMENTARY, AND THE FOUR-ARM TABLE
  IS WHAT SHOWS IT.** One fixture, one seed, flags apart:
      arm                size corr   implied token carried   phi
      source               +0.915            100.0%        +1.000
      neither              -0.003             28.0%        -0.003
      cooccurrence only    -0.003             58.3%        +0.419
      size coupling only   +0.916             49.5%        +0.284
      BOTH                 +0.916             89.3%        +0.833
  Either alone leaves most of the gap; together they carry **83%
  of a deterministic association** that was previously absent, and
  the size correlation comes back to within 0.001 of source. The
  set's own mean size still lands (1.31 against 1.29).
- **THE CONDITIONING GROUP IS WHAT CLEARS k.** A published
  distribution of "how many routes, given six drugs" describes the
  people who have six drugs - so a size held by six patients is
  six people's set sizes however many rows they have, and it is
  not published. The first check for this could not separate the
  mechanism: every size on the fixture is held by hundreds, so
  dropping the floor published the same table and the mutation
  broke nothing. A frame with a size held by six patients turns it
  red.
- **BOTH SHIP OFF, UNDER SEPARATE FLAGS, because they must be
  measurable apart.** `--token-cooccurrence` and
  `--set-size-coupling`. The real-extract arm for the first was a
  TRADE, and a second mechanism that helps on a fixture is not
  evidence about the extract - which is this file's most expensive
  recurring lesson. Separate flags mean the decomposition can be
  run there the way it was run here.

- **THE SCREEN IS PHI NOW, AND THE FAIL-FIRST CAUGHT ME
  ATTRIBUTING THE FIX TO THE WRONG THING (2026-11-09).** The
  detector screened pairs on LIFT at 1.5. A contention fixture -
  ten drugs all implying one route - published ZERO conditionals
  into that route, and I wrote that up as the lift threshold
  disagreeing with the gate's relate floor. Then the mutation back
  to the lift screen broke NOTHING, which is this file's own
  signal, and the measurement said why: on the properly-built
  fixture BOTH screens publish the same twenty pairs, the real
  lift being **1.568** rather than the 1.47 I had computed by
  hand. **The zero came from a MALFORMED fixture** whose route
  column averaged under 1.2 tokens per row, so `vocabulary`
  correctly did not treat it as a set at all and published no
  tokens for it. I had measured the set-size floor and called it a
  lift threshold.
- **AND WHERE THE TWO SCREENS GENUINELY PART TOOK THREE
  CONSTRUCTIONS TO FIND.** A STRONG pair: both publish. A RARE
  pair: both publish - phi is large when the association is
  strong, however rare the tokens, so "lift explodes on rare
  pairs" is also wrong. They part on a **MODERATE conditional on a
  COMMON token**, where lift divides by a large base and stays
  near 1 while phi does not: measured at P(a) 0.301, P(b) 0.500,
  **phi +0.1648 against a lift of 1.251**. The gate relates that
  pair at 0.1 and the lift screen drops it. Pinned as the check
  the mutation can see, and it turns two red.
- **SO THE JUSTIFICATION IS ALIGNMENT, NOT A RESCUE.** Phi is the
  binary-pair correlation the grader computes, so screening on it
  means the contract carries the structure the criterion asks
  about. That is worth doing on its own reasoning; what is NOT
  true, and was nearly recorded, is that it rescued the contention
  case. Under contention - ten drugs for one route's fixed budget
  - the association goes from **+0.001 to +0.147 against a source
  +0.241** with the route's own share landing at 0.636 against
  0.611, and that holds under either screen.

- **THE CONDITIONAL TOKEN DRAW IS A TRADE ON THE REAL EXTRACT, SO
  IT SHIPS OFF (2026-11-09, run_cooc_nov9 against run_rules_nov9).**
  Paired: same source, same seed 20260731, same settings, one code
  change, SAME denominator of 125 so the percentages compare:
      criterion        run_rules_nov9    run_cooc_nov9
      direction kept   118/125 = 94.4%   120/125 = 96.0%
      close            102/125 = 81.6%    99/125 = 79.2%
      inverted              0 PASS            0 PASS
  Re-measured on the shared 136-pair surface with today's code on
  both sides: close **109 against 107**. Direction up two, close
  down three. **Not a net win there**, and this project has twice
  reverted a tier that shipped on a fixture gain a control arm
  later overturned - so it is behind `--token-cooccurrence`, OFF,
  with the price on record. The blueprint PUBLISHES the structure
  either way; hiding the finding would be a different change.
- **AND THE MECHANISM IS ACTING EXACTLY WHERE IT LIVES, WHICH IS
  WHAT MAKES THE NEGATIVE READABLE.** Four pairs flipped out of
  close and **all four are token pairs, none are plain columns**.
  So this is not a broad disturbance; it is the token surface
  moving, which is the surface it was built for.
- **THEY FADED, THEY DID NOT OVERSHOOT - WHICH REFUTES THE OBVIOUS
  READING.** `has_Sodium... <- has_Flush` reads source **+0.342**,
  before **+0.158**, after **+0.079**; `has_mid... <- has_UBC`
  reads +0.400, +0.204, +0.156. A draw that had over-applied its
  conditionals would push past the source; these went the other
  way. The cause is budget: a token's marginal share is pinned, it
  carries up to TWELVE published conditionals, and placing it in
  one partner's rows takes it out of another's. The neural
  engine's own notes name this - "the cross-column route quotas
  become JOINTLY UNSATISFIABLE" - and the allocation here is
  unprioritised, so a 1.5-lift pair steals budget from a 3.4-lift
  one.
- **ON A FIXTURE WITH FEW CONDITIONALS PER TOKEN IT IS A CLEAN
  WIN**, which is why the fixture could not predict the extract: a
  planted implication goes from **-0.003 (absent) to +0.419
  (carried)** with the worst token share missing by 0.016. Two
  shapes, opposite verdicts, and the one that decides is the one
  with the real contention.

- **TVD OVER EVERY PAIR IS REPORTED NOW, WHICH IS THE AXIS THE
  COMMERCIAL TOOLS PUBLISH ON (2026-11-09).** Every pair criterion
  here is a PASS/FAIL on a correlation delta over the pairs the
  SOURCE relates at 0.1 or more - blind to a generator INVENTING
  structure where there was none, and a threshold, so it moves in
  steps while the thing underneath moves continuously. MOSTLY AI
  reports accuracy as the mean of univariate and bivariate total
  variation distance over DISCRETISED distributions across ALL
  pairs; SDV's quality report uses contingency similarity for
  categorical pairs, the same quantity.
- **THE PROPERTY THAT MAKES IT WORTH CARRYING, measured:**
  shuffling one column leaves every MARGINAL untouched and the
  univariate distance at **0.0000** while the bivariate rises to
  **0.2472** and names the broken pair at 0.713. That is the
  classic way a generator looks right per column and is useless,
  seen by one number.
- **THE EDGES COME FROM THE SOURCE AND ARE APPLIED TO BOTH
  TABLES.** Binning each table by its own quantiles compares a
  source decile against a synthetic decile and calls two
  differently-shaped columns identical - the mutation that does it
  turns a check red.
- **REPORTED, NOT GATED, AND NO CELL IS PUBLISHED.** A bar for it
  would change what a MET means on a threshold nobody has measured
  a basis for, and this project does not move a bar and a number
  in the same commit. The report carries aggregate distances and
  column names only - a contingency cell is a count of people, and
  a bin holding three describes those three.

- **TOKEN CO-OCCURRENCE IS MODELLED NOW, AND IT CLOSES THE LARGEST
  STRUCTURAL GAP THIS ENGINE HAD (2026-11-09).** About **110 of the
  135 pairs** the real extract's gate relates are set-token pairs -
  roughly four fifths of everything graded - and the rules engine's
  draw modelled NONE of it. `_token_weights` solved weights so each
  token's MARGINAL share measured back and there was nothing in it
  about which tokens co-occur, so tokens were drawn independently
  and "a drug implies its route" survived by chance. That is not a
  tuning gap; there was no mechanism to tune.
- **THE APPROACH IS THE STATE OF THE ART, APPLIED TO SET
  MEMBERSHIP.** Auto-regressive tabular generators (TabularARGN,
  which is MOSTLY AI's engine) approximate the joint as a product
  of CONDITIONALS over discretised columns rather than a product of
  marginals. `_pick_row` does that for a row's tokens: each pick
  reweights the remaining candidates by the published structure of
  what the row already holds, including tokens drawn in EARLIER set
  columns - which is where `drug -> route` lives.
- **AND THE CONDITIONAL IS SOLVED, NOT NUDGED.** A published LIFT
  is a multiplier; measured on a planted deterministic pair the
  lift alone reached **+0.18** against a source +1.000. What has to
  LAND is `P(b | a)`, so the detector publishes it and the solve
  feeds it back by the same draw-measure-adjust loop the marginals
  use, with the same damping and for the same reason. **The SOLVE
  AND THE DRAW CALL ONE PICK FUNCTION** - measuring an independent
  draw while generation drew conditionally would leave the
  marginals unlanded and the two halves disagreeing about the
  column's contents, which is the fault this project records more
  than any other.
      pair                     source   no cooc   with cooc
      heparin <-> Flush        +1.000    -0.003      +0.419
      albuterol <-> Neb        +1.000    -0.001      +0.508
      worst token share miss                          0.016
  From ABSENT to carried, with the marginals still landing.
- **THE JOINT IS WHAT CLEARS k, NOT THE TWO TOKENS SEPARATELY.**
  Each token has already passed the vocabulary's own patient floor;
  a PAIR held by three patients describes those three people
  however common each token is alone. The first check for this
  could not separate the mechanism - a mutation screening per token
  broke nothing, because every pair on the fixture cleared both
  ways - so the fixture now plants two tokens held by ~790 patients
  each whose JOINT is held by fewer than ten, and the mutation
  turns it red.
- **AND REPULSION IS PUBLISHED TOO**, closing a gap this file has
  carried for months as "negative token associations are not
  modeled": a lift below 1 is structure, and the draw handles it
  with no special case.
- **THE REMAINING CEILING IS THE SIZE DRAW, MEASURED RATHER THAN
  GUESSED.** The conditional solve plateaus near **0.44** on a
  deterministic pair - 14 passes reach 0.393, 30 reach 0.430 - so
  it is not the solve running out of road. A row's set SIZE is
  still drawn from the published size distribution INDEPENDENTLY of
  context, so a row that needs a route can draw none. Making the
  size conditional is the next layer and it is named here rather
  than left to be rediscovered.
- **WHAT THE VENDORS DO, AND WHERE WE SIT.** SDV's quality report
  is Column Shapes plus Column Pair Trends (correlation similarity
  for numerics, CONTINGENCY similarity for categoricals), and since
  0.27.0 it DISCARDS pairs with no strong pattern in the real data
  - the same rule as our "the source relates it at 0.1 or more".
  MOSTLY AI reports accuracy as the mean of univariate and
  bivariate (and trivariate) **total variation distance on
  discretised distributions**, over ALL pairs rather than only
  related ones. So their headline is a DISTANCE and ours is a
  pass/fail on a correlation delta.
  What we carry that they do not: published bounds that are the
  mean of the k most extreme patients, so no released number is an
  individual's value; INVERTED as an absolute stop; a positive
  control on every privacy measurement; the determination evidence
  package; and the key/cipher arrangement. What they carried that
  we did not is exactly this - conditional modelling of
  co-occurrence - and it is now built.

- **THE FRESH RULES RUN ON THE REAL EXTRACT: THE PREDICTION SCORED
  3 OF 3, AND THE MECHANISM IS VERIFIED AS FIRING (2026-11-09,
  run_rules_nov9, build 2a8149c).** A paired arm against run_n9g -
  same source, same seed 20260731, same settings, no `--lags`, and
  BOTH confirmed as the rules engine by the guard built the same
  day, so this is one engine against itself rather than the
  cross-engine comparison that wasted a night:
      criterion            run_n9g        run_rules_nov9
      direction kept   118/125 = 94.4% P  118/125 = 94.4% P
      close            100/125 = 80.0% F  102/125 = 81.6% F
      inverted             0 PASS             0 PASS
      shapes tracked        6/7                6/7
      interaction surfaces  2/5                2/5
      dynamics tracked     35/89              35/89
  **close +2 pairs and NOTHING ELSE MOVES.** Re-measured on the
  shared surface with today's code on both sides, 136 pairs: 108
  against 109, so +1 there. Two surfaces, both small, both
  positive.
- **AND THE MECHANISM IS CONFIRMED RATHER THAN INFERRED FROM THE
  OUTCOME.** `peek curves` on each run, which is the observable the
  grid fix acts on:
      indicator parent slots        82        82
      of those WITHOUT a curve   11 (13%)   0 (0%)
      parents with no curve, all     13         2
  The blueprint's surface is IDENTICAL - 82 indicator parent slots
  either way - and every one of the eleven curveless ones now
  carries a published curve. A +2 close with the mechanism
  unverified would have been a seed story; this is the fix doing
  the thing it was built to do.
- **THE PREDICTION, SCORED AGAINST WHAT IT ACTUALLY SAID.** It
  claimed the close drift on sparse-token-PARENT pairs would move,
  and explicitly did NOT claim `shapes tracked`, `interaction
  surfaces` or `dynamics tracked`. close moved; those three are
  byte-identical at 6/7, 2/5 and 35/89. It also named the RISK -
  that a parent which previously applied no push now applies one,
  so close could move either way - and the measured answer is that
  it moved up and cost nothing.
- **AND MY OWN EARLIER EXTRAPOLATION WAS WRONG BY A LARGE FACTOR,
  which the same run settles.** The prediction reasoned that "most
  token-indicator PARENTS have been carrying no curve", from 91-92%
  of the fixture's publishable vocabulary sitting under a 2% share.
  The real extract says **11 of 82**, which is 13%. The modelled
  parents come from the frequency top-24 per set column and skew
  dense; the publishable VOCABULARY does not. Those are different
  populations and I read one as the other.
- **WHAT THIS DOES NOT DO: clear the bar.** close reads 81.6%
  against 87.9%, so the rules engine fails that criterion on this
  extract too - as does the neural deliverable at 87.6%. Neither
  engine passes close here, and +2 pairs is a real gain on a gap
  that is about eight pairs wide. The fresh run DOES clear the
  absolute stop unrepaired (inverted 0, where the deliverable needs
  a one-column repair) and carries two interaction surfaces and
  twice the graded dynamics - but those are the ENGINE's
  properties, not this fix's, and comparing them across engines is
  what this file now has a guard against.

- **THE CURVE `close` CANNOT SEE NOW HAS A REPAIR, AND ON THE REAL
  EXTRACT IT REFUSES - WHICH IS THE ANSWER (2026-11-09).** The
  deliverable's one departed shape is `condition_count <-
  visit_start_date`, and it is the case the criterion exists for:
  source rank correlation **+0.2964** against a generated
  **+0.3223**, so `close` passes it comfortably, while the curve
  departs at **0.4955 sd** against a 0.35 bar and the published
  shape is `non-monotone`. No arrangement aimed at a CORRELATION
  can fix that, because the correlation is already right.
  `repair_inversions --shapes` aims at the SOURCE's own curve
  instead. On a planted U-shape whose rank correlation is +0.013 it
  takes the gap from **1.6205 to 0.0414** as a permutation, with
  zero drift on the child's other couplings.
  On the real extract it reports **0.4955 -> 0.5060, SHORT**, with
  the best arrangement available moving other pairs by **0.171**
  against a 0.10 bar - and the whole gate reads SAME on every
  criterion. `condition_count` is a HUB (it is a set-size partner),
  so every arrangement that follows the date curve fights its other
  relationships, and the joint constraint binds before the bar is
  reached. **A measured refusal, not a failure to try.**
- **AND THE FIRST REAL RUN DECLINED FOR THE WRONG REASON, ON THE
  WRONG-TYPE FAULT.** It reported "the source curve has too few
  publishable bins to aim at" about a pair the gate had just
  measured at 0.4955. `compare` runs both tables through `prepare`
  before `measure_shapes` sees them, so a DATE parent is numeric
  THERE; the repair read the raw CSV, `pd.to_numeric` returned
  all-NaN for `visit_start_date`, and the curve came back with too
  few bins. A column read as the wrong type fails SILENTLY, and
  every fixture here had a NUMERIC parent so none could reach it.
  Measured on the frame the gate measures on now, with a
  date-parent fixture pinned: 1.7012 -> 0.3166, the date column
  itself untouched, and the mutation back to the raw frame turns
  four checks red.
- **AND THE SELECTION RULE TOOK THREE ATTEMPTS TO MAKE TESTABLE.**
  A mutation turning the search into a pure gap-minimiser broke NO
  check twice: the first comparison asserted `alpha > 0` on a
  fixture where the smallest gap sits at the same alpha, and the
  second compared arms that differed in the GRID rather than in the
  RULE. `pick_arrangement` is a PURE function over (alpha, gap,
  drift) triples now, asserted on planted arrangements - offered a
  0.05-gap arrangement that moves other pairs by 0.90 and a
  0.30-gap one that moves them 0.02 it must take the second - and
  the same mutation turns three checks red. This project already
  recorded that a guard only became testable once the function was
  made pure; it is the same fix.
- **WHAT THIS SETTLES ABOUT PASSING THE WHOLE GATE.** The repair
  loop reaches the PAIR criteria - `close`, `inverted`, `direction`
  - and it works: the deliverable goes from four of nine to six of
  nine. It does NOT reach `shapes tracked` on this file, for a
  measured reason rather than a missing feature. And it cannot
  reach `interaction surfaces` (0/5, gaps 0.70 to 1.38 against a
  0.35 bar, and **54 of the 61 published surfaces cannot even be
  MEASURED** because they involve token indicators the written file
  does not carry) or `dynamics tracked` (17/87, a row-level engine
  with no trajectory model). Those three are FULL-MARKS criteria on
  large denominators. Six of nine is the honest ceiling for
  rearrangement on this artifact, and saying so is worth more than
  a plan to reach nine.

- **THE TWELVE PAIRS WERE NEVER A REGRESSION: THE DELIVERABLE AND
  EVERY ARM MEASURED AGAINST IT WERE BUILT BY DIFFERENT ENGINES
  (2026-11-09, read off the artifacts).** `run_deliverable`'s
  `provenance.json` carries `settings.engine = "neural"`, and that
  field is written by `latent_challenger.py` and by nothing else -
  `"engine"` does not appear anywhere in the package. `run_n9g`
  carries no engine field and its settings are
  `{generate, seed, holdout, excluded}`, which is exactly what
  `pipeline.py` writes. A SECOND, INDEPENDENT observable agrees:
  the deliverable's top level is
  `blueprint_version, source, settings, contract` where `contract`
  is the neural writer's, and run_n9g's is
  `made_at, code_fingerprint, build, blueprint_version, source,
  settings, privacy, note` where `privacy` and `note` are the rules
  writer's. **So `close` 119 against 107 on 135 shared pairs is the
  NEURAL engine against the RULES engine, not one build against
  another.**
- **WHAT IS WITHDRAWN.** "close DID regress - tonight's output is
  worse by twelve pairs"; "the gap is 182 commits wide"; and the
  reading that four control arms had acquitted every candidate.
  Those arms could not have found anything: every mechanism they
  switched off - the identity detector, the date reorder, identity
  enforcement, the sparse-slice drop - is a RULES-engine mechanism,
  and the baseline they were compared against was a NEURAL file. An
  arm that cannot reach the thing it is testing returns zero and
  means nothing, which this file already says about a cap no fixture
  reaches. **The five-run bisection is cancelled** - there is no
  commit to find, and running it would have spent two hours
  confirming a difference no change causes.
- **AND THE RECORDED CLAIM THAT SET IT UP WAS FALSE.** The run_n8g
  entry says "same source as the deliverable ... same settings as
  that run". The source matched; the ENGINE did not, and nothing
  checked. Two artifacts differing in more ways than the comparison
  assumed is the oldest fault in this file wearing a provenance
  field, and the fix is one line: read `settings.engine` on both
  sides before quoting a cross-run number.
- **THE FLIP NAMES SAY THE SAME THING FROM THE OTHER END, AND THEY
  ARE WHY THE DIFFERENCE IS EXPECTED RATHER THAN MYSTERIOUS.** The
  comparison names **21** pairs the deliverable holds close and the
  rules run loses - not twelve; the net is twelve because the rules
  run also gains some. Of the twenty listed, **FIFTEEN involve a
  `__has__` set-token indicator** and five are plain clinical
  columns. The neural engine has an independent per-token draw, an
  implication tier and conditional token quotas, all built
  specifically for cross-set token pairs; the rules engine has none
  of them. A neural file holding more drug-implies-route pairs close
  is the designed behaviour of that engine, not a defect in the
  other.
- **WHAT STILL STANDS, said separately so the withdrawal does not
  take it with it.** The deliverable's own gate numbers are measured
  on the deliverable and are unaffected: 134/137 direction, 120/137
  close, 1 inverted, 42/42 coverage, and six of nine after the
  one-column repair. The real-extract attack battery that certifies
  the release surface - output adversary 0.514/0.523/0.551 with the
  republish control at 1.000 CAUGHT - is `latent_challenger attack`,
  so it is the NEURAL battery and it is the right evidence for this
  file. What is withdrawn is only the cross-run comparison.
- **AND A CROSS-ENGINE GATE COMPARISON IS WORSE THAN
  INCOMPARABLE ON THREE CRITERIA IN PARTICULAR.**
  `interaction surfaces` grades whether PUBLISHED surfaces survive,
  and the neural engine never consults them - this file already
  records that it fails that criterion BY CONSTRUCTION - so
  "0/5 to 2/5" is not an improvement, it is one engine being graded
  on a contract the other one has. The same applies to
  `shapes tracked`, and `dynamics tracked` is measured along an axis
  the two engines treat differently. Quote those three within one
  engine or not at all.

- **THE SPARSE-TOKEN RELATIONSHIP GOES FROM ABSENT TO CARRIED AT
  ZERO MEASURED COST, AND THE CAP WAS NEVER WHAT BOUND IT
  (2026-11-09).** Three paired arms on one fixture at one seed,
  each differing from its partner by one thing:
      arm                      gen lift   % of source   driver curve
      cap 24                     +0.030        1.2%     not claimed
      cap 60                     +0.035        1.4%     NO CURVE
      cap 60 + grid fix          +2.124       83.9%     range +2.46
      source                     +2.530         ---
  The price, on the pairs both arms relate: cap 60 against cap 60
  plus the fix is **close 42 of 45 on BOTH arms, 45 relationships
  on both, not one pair lost**. A 7.2 sd relationship of the
  extract's largest kind goes from absent to 84% carried and
  nothing else moves.
- **AND IT REFUTED THE FIX I HAD JUST SPENT AN HOUR MEASURING.**
  The obvious reading of the miss was the search cap, and the cap
  arm settles it: at cap 60 the driver IS expanded and IS
  published as a parent BY NAME - `['conditions__has__cond_m40',
  'drug_routes', 'noise_02']` - and the output still carries
  **1.4%** of the relationship, for **3.8x the wall clock** (3586s
  against 933s) and one lost close pair. Raising the cap buys 0.2
  percentage points of a relationship it correctly publishes. The
  cap decides whether a relationship is FOUND; it has nothing to
  do with whether it is CARRIED.
- **THE MECHANISM IS A QUANTILE GRID ON A BINARY COLUMN, AND IT
  WAS FOUND BY READING THE PUBLISHED EVIDENCE RATHER THAN
  GUESSING.** The cap-60 blueprint names the driver in `parents`
  and its `effect` dict holds ONLY `drug_routes` (range 0.075) and
  `noise_02` (range 0.009) - both noise. `_grid_for` sweeps the
  2nd to 98th percentile, so an indicator that is 1 on 0.81% of
  rows has EVERY quantile at 0, `np.unique` collapses the grid to
  one point, `effect_curve` returns None and `discover` skips the
  curve with a bare `continue`. So the contract listed the driver
  as a parent and published no evidence for it, and generation had
  nothing to apply. A binary column has two values by
  construction; quantiles are the wrong tool for it.
- **THE FIX IS ADDITIVE, WHICH IS WHY A ONE-LINE CHANGE INSIDE
  EVERY EFFECT CURVE IN THE PROJECT WAS SAFE TO SHIP.** Above a 2%
  share the quantile grid ALREADY returns [0, 1], so the only
  behaviour that changes is the case that produced no curve at
  all. Checked rather than argued: a three-value column keeps the
  quantile path, a continuous column still gets its eleven points,
  and the whole net reads ALL GREEN at 103 suites and 3,447
  checks. Two mutations bite, and the dense-token check stays
  green under both - which is the additivity as a measurement
  instead of a claim.
- **AND HALF THE MODELLED VOCABULARY WAS IN THAT STATE.** On the
  fixture's own publishable vocabulary **91-92% of the tokens that
  clear k sit under a 2% share** - conditions 120 of 132,
  active_drugs 132 of 144, procedures 120 of 132 - and **36 of the
  72 MODELLED indicators** are under it even though the frequency
  top-24 skews dense by construction. So for as long as the search
  has expanded tokens, roughly half of the ones it models could
  not carry a curve as a parent.
- **WHAT IS NOT ESTABLISHED, AND THE FIXTURE SAYS SO ITSELF: the
  fix is INERT at the shipped default here.** A fourth paired arm
  at cap 24 with the fix reads 43 relationships and **41 close on
  both arms with not one pair lost** - identical - and the reason
  is in the artifact: **ZERO relationships in either blueprint
  have an indicator as a PARENT**. This fixture's tokens carry no
  structure except the one planted driver, and at cap 24 that one
  is not expanded, so there is nothing for the fix to act on. Two
  arms that agree are one arm, and here that is a statement about
  the fixture rather than about the fix.
  Whether the REAL extract has sparse indicator PARENTS is
  therefore the question that decides what this buys there - and
  it is a ONE-MINUTE READ of `run_deliverable/blueprint.json`,
  which is already on disk: count the relationships whose parents
  include a `__has__` column, and how many of those parents have
  no entry in their claim's own `effect`. No fit, no generate. It
  is queued behind the restart dialog with the regate read.

- **PREDICTION FOR THE NEXT REAL RUN, ON RECORD BEFORE IT RUNS AND
  BEFORE THE FIXTURE ARM REPORTED (2026-11-09).** `_grid_for` now
  gives a binary column its two values instead of its quantiles, so
  a token indicator whose share is under 2% carries a published
  curve where it previously carried NONE. Measured on the fixture's
  own publishable vocabulary: **91-92% of the tokens that clear k
  sit under a 2% share** (conditions 120 of 132, active_drugs 132
  of 144, procedures 120 of 132), and **half of the MODELLED
  indicators** are under it even though the frequency top-24 skews
  dense. On the real extract, where 1,827 tokens are published over
  56,065 rows, 2% is 1,121 rows and the great majority of published
  tokens are far below it - so most token-indicator PARENTS have
  been carrying no curve for as long as the search has expanded
  them.
  WHAT IS PREDICTED: the close drift on pairs where a SPARSE TOKEN
  IS THE PARENT should move. The record names that group as the
  largest single one - "a `set__has__token` against an ordinary
  column, 5 of 17" - and names the cross-set token pairs
  (has_sodium-chloride <- has_Flush, 0.88 source against 0.58
  generated) as the residue after the implication tier.
  WHAT IS NOT PREDICTED, and the distinction decides whether the
  fix can reach a pair at all: where the indicator is the CHILD it
  is DERIVED from the set already drawn, so its own grid was never
  consulted and this fix does nothing for it. Nor is anything
  claimed for tokens ABOVE a 2% share - the quantile grid already
  returned [0, 1] for those, which is exactly why the change is
  additive. And `shapes tracked`, `interaction surfaces` and
  `dynamics tracked` are untouched by this.
  AND THE RISK IS NAMED RATHER THAN WAITED FOR: a parent that
  previously applied NO push now applies one, so this is not a
  tightening of something that already worked - it is a new force
  on every sparse-token parent at once. A fix aimed at one
  criterion must say what it might cost the others, and the honest
  statement is that `close` could move either way on pairs whose
  sparse-token parent was previously inert. The run of record is a
  PAIRED arm on one seed with one code change, and the whole gate
  is read, not just the criterion this aims at.

- **THE SEARCH AND THE CONTRACT DISAGREE ABOUT WHICH SET TOKENS
  EXIST, SO A RELATIONSHIP FOUND CORRECTLY CANNOT BE PUBLISHED -
  AND THE CLAIM SURVIVES NAMING A PARENT THAT DOES NOT DRIVE IT
  (2026-11-09).** `--expand-by signal` expands the tokens that
  MEASURE as related to a numeric column; the blueprint's
  modelled indicator columns are the first `EXPAND_CAP` by
  FREQUENCY. On a 300-patient fixture with a 120-token middle
  band, every figure from one paired run:
      the planted driver, found    importance 0.633, skill 0.281
      blueprint parents published  ['drug_routes'] at 0.004
      the driver in `excluded`     NOWHERE - a bare `continue`
      source lift of the driver            +2.530
      generated lift                       +0.046
  The blueprint models exactly **72** indicators, which is
  `24 x 3` and is provably the frequency top-24 per set column -
  asserted as a subset, not inferred from the count. So the
  search spent its budget finding a relationship on a token the
  contract has no column for, the driver was dropped in silence,
  and a co-predictor at **one hundred and fifty times less
  importance** became the published parent.
- **AND THIS IS A CANDIDATE MECHANICAL CAUSE FOR A RECORDED
  MYSTERY, which is why it is worth more than the one pair it
  cost here.** This file records that `--expand-by signal` LOSES
  on the real extract - 115 published relationships against
  frequency's 87, "a quarter of them gone" - with the cause
  never established and the mass argument predicting the
  opposite. A signal-expanded token having no modelled column to
  be published on would produce exactly that: relationships
  found and then dropped at blueprint time. NOT CLAIMED as the
  cause - the magnitude here is ONE pair, because this fixture
  has one token-driven relationship where the extract's surface
  is dominated by them. It is a mechanism confirmed on a fixture
  and a hypothesis about the extract, and the two are different
  things.
- **THE DROP IS REPORTED NOW, WITH THE IMPORTANCE IT CARRIED.**
  `excluded.unmodelled_parents` names the child, the parent and
  its importance, so a reader comparing it against the parents
  that WERE kept can see whether the published claim still means
  what the search found. The importance is the load-bearing
  field: without it the section would say a column was dropped
  and not that the DRIVER was. Restoring the bare `continue`
  turns two checks red; the third pins the claim surviving with
  the weaker parent, which is true either way and is the
  behaviour that makes the report necessary.
- **AND IT IS ASSERTED ON PLANTED INPUT, because no fixture here
  can reach the condition.** It needs the two halves to
  DISAGREE, and a run that expands by frequency has them agree
  by construction - so a fixture-driven check would pass without
  exercising anything, which is the cap-no-fixture-reaches fault
  this file already records. A catalogue naming a ghost
  indicator is handed to `build` directly.

- **A PLANTED RELATIONSHIP OF THE EXTRACT'S LARGEST KIND WAS IN
  NEITHER THE ANSWER KEY NOR THE NOISE LIST, AND THE INSTRUMENT
  THAT GRADES RECALL COULD NOT BUILD IT AT ALL (2026-11-09).**
  `set_driven_y` has been planted for as long as `--set-vocab`
  has existed and appeared in neither list, so the recall score
  could neither credit finding it nor penalise missing it.
  Token-driven relationships are about **110 of the 135 pairs**
  the real extract's gate relates, so the score had no coverage
  of its dominant kind. And `bench_new_path.py` has no
  `--set-vocab` flag, so it could never build a fixture
  containing the column - the kind was unscoreable BY
  CONSTRUCTION. That is the same shape as the `--lags` finding
  recorded the day before, where the capability was in the
  PIPELINE and the instrument that grades recall never built the
  features. Both fixed: the kind is in the key as
  `set-token-driven` (only when a middle band is planted, so the
  DEFAULT fixture's key is untouched and every other suite reads
  it unchanged), and the flag is wired through.
- **AND THE MISS IS THE CAP, NOT DETECTABILITY, WHICH IS A
  DIFFERENT FINDING FROM THE ONE ON FILE.** This file records a
  cap experiment whose conclusion was "nothing on 22 rows is
  detectable at any cap", with an explicit warning not to read
  "raising the cap changed nothing" as "the cap is free".
  Measured here at 300 patients: the driver is on **185 of
  22,910 rows (0.81%)** and lifts the child by **2.53 against a
  planted noise sd of 0.35 - about 7.2 sd** - plentiful and
  strong. It clears the k floor at published rank **57 of 132**
  while `EXPAND_CAP` keeps 24, so no indicator column exists and
  the relationship cannot be examined at all. Which midband
  token survives is a frequency coin flip among ~120 near
  equals: `cond_m47` and `cond_m29` were expanded and `cond_m40`
  was not.
- **AND THE CHECK FOR IT FAILED FIRST ON MY OWN SCALE, TWICE IN
  ONE PROPERTY.** Written at 120 patients and 10 visits - 1,043
  rows - only 65 of 120 midband tokens clear k and the driver
  lands at rank **23, INSIDE the cap**, so that scale cannot
  exercise the thing the check is about and both assertions went
  red against correct code. Four scales were measured rather
  than guessed; 250 patients at 40 visits reaches it in under a
  second (6,388 rows, 132 clear k, driver at rank 61 on 52
  rows). A fixture must contain the thing the check is about,
  and "has a middle band" was not specific enough.
- **AND THE CAP CHECK GOES RED IF THE CAP RISES PAST THE DRIVER,
  WHICH IS THE RIGHT BEHAVIOUR AND ITS MESSAGE SAYS SO.** A
  check that pins "the driver is outside the cap" would
  otherwise read as punishing progress. Red there means the
  fixture needs a DEEPER driver because it has stopped
  exercising the cap - an action, not a verdict on the cap - and
  the failure text states it.

- **THE TWELVE PAIRS DO NOT REPRODUCE ON A FIXTURE, AND THE TWO
  SUSPECTED MECHANISMS ARE ACQUITTED A SECOND TIME ON A
  DIFFERENT SHAPE (2026-11-09).** The deliverable's build
  `203c4ae` was put in a WORKTREE - not a checkout, because a
  checkout is what left HEAD detached and silently pushed
  nothing last night - and both builds were run on ONE
  byte-identical fixture at one seed, 300 patients, 22,910 rows,
  86 columns, a ring of 8 and a 120-token set vocabulary, 156
  columns after expansion in BOTH arms:
      old (203c4ae) related   43     close on shared   41
      new (b64bcb3) related   43     close on shared   41
      shared pairs            43     DELTA              0
      pairs the old holds and the new loses              0
- **AND IT IS NOT A VACUOUS ARM, which is the only thing that
  makes a zero worth reading.** The two generated files have
  DIFFERENT md5s, and the mechanisms are verified as firing
  rather than assumed: the new blueprint publishes the
  `age_at_visit` arithmetic identity and records
  `patients.time_column = visit_start_date`, where the old
  publishes NO identity and has no time column at all - so both
  the enforcement and the within-patient date reorder engage in
  one arm and are absent in the other. The old tree's
  `generate()` has no `reorder_dates` parameter, read off its
  own code object, which is the observable confirming the
  worktree is the pre-date-work build instead of trusting a sha.
- **WHAT THE NEGATIVE DOES AND DOES NOT ESTABLISH.** It is a
  second, independent acquittal of the two mechanisms the four
  real-extract control arms already cleared, on a shape they
  were not measured on, with the firing verified. It does NOT
  locate the twelve: this fixture relates 43 pairs where the
  extract relates 135, and the extract's surface is dominated by
  about 110 set-token pairs that this fixture does not relate at
  all - its tokens carry no structure against the numerics, so
  the one place the recorded drift CONCENTRATES is missing from
  the measurement. Close also sits at 41/43 here, so there is
  little room to lose pairs. A control arm that cannot resolve
  the cost it is watching for is not a control arm, and that is
  said about this one rather than left for somebody to assume.
- **THE BISECTION IS NARROWED FROM EIGHT RUNS TO FIVE, BY
  READING WHICH FILES MOVED.** 184 commits separate the
  deliverable from tonight and 71 touch the package, but a
  single-table fit-and-generate cannot be changed by cli, gui,
  determination, scrub, dp, linkage, automl, multitable, intake
  or jointcheck. Filtering to generate / blueprint / discover /
  sets / pipeline / identities / temporal / dates / tableplan
  leaves **TWENTY**, which is five runs rather than eight -
  about 2.2 hours of machine time instead of 3.5. FOUR MORE
  touch `gate.py` and change the MEASUREMENT rather than the
  output; they are listed separately because re-gating both runs
  with today's code already controls for them, and conflating
  the two sets would send a bisection after the wrong thing.
- **AND THE CHEAPEST READ IS STILL UNRUN AND IS NOT A FIT.**
  Naming WHICH twelve pairs flipped is one `regate.py`
  invocation against two existing run directories - about a
  minute, no fit, no generate - and their shape would name the
  mechanism without any bisection at all. It is queued behind an
  IT security-restart dialog on the data machine, whose two
  buttons are the operator's to press: deferring a security
  update is not a decision a tool makes on somebody's managed
  laptop.
- **ALL FOUR CANDIDATES FOR THE TWELVE PAIRS ARE REFUTED, AND THE
  GAP IS 182 COMMITS WIDE RATHER THAN ONE NIGHT'S WORK
  (2026-11-08 night).** Every arm paired on the same source and
  seed, one thing apart, measured on the pairs both arms relate:
      the identity DETECTOR    107 against 108     1 pair
      the DATE REORDER         108 against 108     ZERO
      identity ENFORCEMENT     108 against 108     ZERO
      the sparse-slice drop    fired on 0 targets of this extract
  The reorder and the enforcement were both suspected on the same
  mechanism - each OVERWRITES or moves a date column against
  every other one - and each costs nothing here. The enforcement
  arm was needed precisely because it survives
  `--no-reorder-dates`, which is why the reorder arm coming back
  identical did not settle it; now both are settled and both are
  innocent.
- **AND THE DELIVERABLE IS TWO WEEKS AND 182 COMMITS OLD, which
  is the fact that reframes the whole question.** Its
  `provenance.json` names build `203c4ae`, dated **2026-10-26**.
  So the honest statement is: `close` on the shared pair set
  reads **119 for the deliverable against 107-108 for every arm
  tonight**, the difference is real, and it is NOT attributable
  to any change made tonight - all four mechanisms were measured
  and acquitted. Where in those 182 commits it lies is unknown.
- **AND NARROWING IT HAS A PRICE WORTH STATING BEFORE ANYBODY
  STARTS.** Each arm is a full fit and generate on the real
  extract, about 26 minutes on the data machine, so a bisection
  over 182 commits is roughly eight runs and three and a half
  hours of machine time - feasible, unattended, and the only
  honest route, because every cheap guess tonight was wrong. A
  linear sweep would be eighty hours and is not the way.
- **WHAT THE FOUR ARMS DID BUY, measured on the same runs:**
  `inverted` 1 to 0 with no repair, `interaction surfaces` 0/5 to
  2/5, `dynamics tracked` 17/87 to 35/89, both arithmetic
  identities holding on **1.0** of rows, and a patient's dates
  agreeing with the order the file numbers them by. Those are
  measured on published surfaces and graded columns, so they
  survive the pair-surface problem that makes `close` percentages
  incomparable. The twelve pairs and these four are both real and
  neither cancels the other.

- **TWO OF THE THREE CANDIDATES FOR THE TWELVE PAIRS ARE DEAD, AND
  THE THIRD NEVER RAN (2026-11-08 night).** Paired arms, same
  source and seed, one thing apart each time, all measured on the
  pairs both arms relate:
      the identity DETECTOR   run_n8g 107  run_n9g 108   (1 pair)
      the DATE REORDER        reorder on 108  off 108    (ZERO)
      the sparse-slice drop   fired on 0 targets of this extract
  The reorder was my named candidate and it costs **nothing** on
  this extract - 108 close on 136 shared pairs with it on and 108
  with it off, which is a control arm answering a question rather
  than a story explaining one. And the discovery-side drop, which
  would have changed the blueprint and could have explained the
  surface difference, reports `targets_with_empty_features` of
  **zero** here: the fix that stopped a crash on `wide + 92%
  sparse` never engages on the real extract at all.
- **SO THE REMAINING UNTESTED CHANGE IS THE IDENTITY ENFORCEMENT
  ITSELF, and it needed an off switch before it could be
  accused.** It OVERWRITES a column - on this extract
  `visit_start_date`, computed from the end date minus the span -
  which moves that column against every other one by exactly the
  mechanism the reorder was suspected of, and it SURVIVES
  `--no-reorder-dates` because it is a separate step. That is why
  the reorder arm came back identical: it never disabled the
  thing doing the overwriting. `--no-identities` is the arm for
  it, with the same reasoning written into its help text, and the
  blueprint still PUBLISHES the identity either way - hiding the
  finding would be a different change.
- **AND ADDING THAT FLAG KILLED EVERY INVOCATION OF THE SCRIPT FOR
  ONE COMMIT, ON THE OLDEST ARGPARSE FAULT THIS FILE RECORDS.**
  The help text said `11.9% of rows`, argparse %-formats help
  strings, and Python 3.14 validates them EAGERLY at
  `add_argument` - so `run_discovery.py` raised
  `ValueError: badly formed help string` before doing anything at
  all, on every call. This file already records the identical
  fault in `m0_gate`'s bar flag, where it took down all eight
  gate checks on the one machine that matters. Doubled, and the
  check now asserts the doubling rather than only the wording.

- **`close` DID REGRESS, AND IT TOOK A SHARED-PAIR MEASUREMENT TO
  SAY SO HONESTLY: 119 AGAINST 107 ON THE 135 PAIRS BOTH RUNS
  RELATE (2026-11-08 night).** Each run's pair surface comes from
  its OWN blueprint's token vocabulary, so the recorded 87.6%
  and tonight's 80.8% are over different sets and no
  re-measurement makes the denominators equal - regated with
  today's code on the same source they still read 148 against
  125. On the INTERSECTION, both measured the same way:
      shared related pairs            135
      run_deliverable          119 close
      run_n9g                  107 close
  **Tonight's output is worse by twelve pairs**, and it is not a
  denominator artifact - the earlier reading, "close has no
  comparable baseline", was true of the percentages and became an
  excuse once the comparable number existed.
  **CORRECTED THE SAME NIGHT: it is NOT a regression I introduced
  today.** I wrote that sentence and then built four control arms,
  and every one of them came back at zero or one pair. The
  deliverable was built at commit `203c4ae` on **2026-10-26** -
  **182 commits** before tonight - so the twelve span two weeks
  of changes, and attributing them to the night's work was a
  story about the most recent thing I had touched.
- **AND THE IDENTITY DETECTOR IS NOT THE CAUSE, which the paired
  arms already settled.** run_n8g (four identities, two spurious)
  against run_n9g (two) differed by ONE close pair on an
  identical surface. Both arms carry the date work, so does the
  start-axis arm, and all three land in the same place. What the
  twelve have in common is still to be established.
- **THE CANDIDATE IS THE DATE WORK, AND IT IS NAMED RATHER THAN
  ASSUMED.** Two things landed today that move a date column's
  values against every non-date column: the within-patient
  reorder, measured on a fixture as costing 0.061 of rank
  correlation on a date-driven numeric, and the identity
  enforcement, which computes one date member from the other two.
  Neither can be switched off, which is exactly why the twelve
  cannot be attributed yet - a cost nobody can turn off is a cost
  nobody can measure. The test is a flag that disables the
  reorder, run paired on the same seed; that is the next step and
  it is a CONTROL ARM, which this file records as the only thing
  that has ever overturned a fidelity claim here.
- **WHAT THE TWELVE DO NOT BUY BACK, said plainly so the trade is
  visible: `inverted` 1 to 0 with no repair, `interaction
  surfaces` 0/5 to 2/5, `dynamics tracked` 17/87 to 35/89, and
  both arithmetic identities holding on 1.0 of rows where the
  record measured one of them at 21.4%.** Whether twelve close
  pairs is worth those four is a decision for an operator and not
  a default, and this file's own rule is that a gain a control arm
  has not checked is not a gain. The four are measured on
  published surfaces and graded columns; the twelve are measured
  on the shared pair set. Both numbers are real.

- **THE TIME-AXIS OVERRIDE SCORED 3 OF 3 ON ITS CLAIMS AND THEN
  ARGUED AGAINST ITSELF (2026-11-08 night, run_n9s).** Three arms
  on the same source and settings, one flag or one code change
  apart. Every figure crossed in words:
      criterion            end axis    end axis    start axis
                           4 ids       2 ids       2 ids
      close              100/125     101/125      100/125
      inverted             0 PASS      0 PASS       0 PASS
      shapes tracked         6/7         6/7          6/8
      interaction surfaces   2/5         2/5          0/4
      dynamics tracked     35/89       35/89        35/90
      start in row order  0.8888          -          1.0000
      identity child      start=1.0   start=1.0    end=1.0
- **THE CLAIMS HELD.** `visit_start_date in row order` went
  **0.8888 to 1.0000**, and `visit_end_date` took its place at
  0.8875 - the mirror image, because start and end are not
  co-monotone on the 11% of patients whose stays overlap. The
  override reached the blueprint rather than being read and
  ignored. And the identity's computed child FLIPPED from the
  start date to the END date, which is the arrangement a fixture
  measured as the right one: `span_days` keeps its published
  marginal and the end date is computed from the ordered start
  plus it, holding on **1.0** of rows.
- **AND THE PRICE IS THE TWO INTERACTION SURFACES THE IDENTITY
  TIER HAD JUST WON: 2/5 BECOMES 0/4.** The denominator moved as
  well, so the axis changes WHICH surfaces discovery publishes as
  well as whether they survive - the dynamics are measured along
  the axis, so every column with persistence is generated
  differently, the pressure family included. The prediction's
  not-claimed clause covered exactly this and it is the half that
  decided the outcome.
- **SO THE DEFAULT STAYS: DETECTION'S CHOICE OF `visit_end_date`
  IS THE ONE THAT SCORES BETTER.** This file records the end date
  as misordering overlapping stays, and that is still true - 11%
  of patients come out with their start dates out of order. But
  the obvious-looking fix costs the only two interaction surfaces
  this engine has ever carried on the real extract, and trading a
  criterion that FAILS either way for one that moved from zero is
  not a trade anybody asked for. The override stays available,
  the trade is on record, and nothing is changed on the strength
  of one seed.

- **PREDICTION FOR THE `--time-col visit_start_date` ARM, ON
  RECORD BEFORE IT LANDS (2026-11-08 night, run_n9s).** Detection
  picks the date with the most distinct values and chose
  `visit_end_date`, which this file already records as misordering
  overlapping stays. That choice now has two further
  consequences, so the override tests three things at once.
- **WHAT IS CLAIMED.** (1) `visit_start_date in row order` goes
  from **0.8888** to essentially 1.0, because the reorder will
  sort by the start date itself rather than carrying it along
  behind the end date - the 11% are the patients whose stays
  overlap, where start and end are not co-monotone. (2) The
  blueprint's `time_column` reads `visit_start_date`, since an
  explicit override must actually be used - a flag that is read
  and ignored is the `--time-col` failure this file names. (3)
  The identity tier then protects the START date instead, so the
  computed child of `start / end / span_days` changes - the END
  date becomes the child, which is the arrangement this project
  measured as the right one on a fixture, because `los` keeps its
  published marginal and the end date is computed from the
  ordered start plus it.
- **WHAT IS NOT CLAIMED: that any gate criterion moves.** The
  dynamics are MEASURED along the axis too, so changing it
  changes both the source figures and the generated ones, and
  `dynamics tracked 35/89` could go either way for reasons that
  are about the measurement rather than the engine. `close` has
  no comparable baseline tonight in any case. If a criterion
  moves, the next question is whether the SOURCE side of it moved
  with the axis - which is a different fact from the output
  improving.

- **THE PAIRED ARM REFUTED MY OWN DIAGNOSIS: THE TWO SPURIOUS
  IDENTITIES COST ONE PAIR, NOT TWENTY (2026-11-08 night, run_n9g
  against run_n8g).** Same source, same settings, same seed, one
  code change - the float-noise floor on the identity detector.
  Every figure crossed in words:
      criterion          run_n8g (4 ids)   run_n9g (2 ids)
      identities published      4                 2
      close             100/125 = 80.0%   101/125 = 80.8%
      inverted              0 PASS            0 PASS
      coverage             42/42             42/42
      shapes tracked         6/7               6/7
      interaction surfaces   2/5               2/5
      dynamics tracked      35/89             35/89
  The fix removes exactly the two spurious identities and moves
  **one close pair**. I had written down that two hub columns
  overwritten from scaffolding was "the obvious candidate for 20
  lost close pairs". It was not.
- **AND THE REASON IS THAT THE SPURIOUS IDENTITY WAS VALUE-NEUTRAL
  BY LUCK.** `procedure_count` IS `procedure_quantity` in the
  source, and the fitted weights are `1.0 * procedure_quantity +
  ~0 * an indicator + ~0` - so computing it wrote back
  approximately the number the column already held. The defect
  was real and nearly inert HERE; a spurious pair whose second
  weight was not ~0 would have done damage. Being harmless by
  arithmetic coincidence is not a reason to keep it.
- **AND THE DELIVERABLE'S `close` IS NOT COMPARABLE TO TONIGHT'S,
  WHICH IS THE HONEST ANSWER TO "DID IT REGRESS".** The recorded
  87.6% (120/137) against tonight's 80.8% (101/125) looks like a
  fall and the pair SETS are different. Diffed by name against
  `run_deliverable`'s own fidelity.json, still on the machine:
      pairs the deliverable recorded      196
      pairs run_n9g recorded              172
      in BOTH                             160
      deliverable only                     36
      run_n9g only                         12
      of the 36, token or set-size          34
  Thirty-four of the thirty-six that dropped out are set
  scaffolding, so the two percentages are measured over
  substantially different surfaces. This run's own prediction
  said to read the ABSOLUTE counts and warned the denominator
  might move; it moved, and the right conclusion is that close
  has no comparable baseline here rather than that it fell.
- **WHAT IS COMPARABLE AND DID IMPROVE, stated separately for
  that reason:** `inverted` goes from 1 FAIL - the one criterion
  this project calls an absolute stop, which the deliverable
  needed a reordered column to clear - to **0 PASS with no
  repair**; `interaction surfaces` from **0/5 to 2/5**, the
  pressure family sitting on the enforced arithmetic; and
  `dynamics tracked` from **17/87 to 35/89**. Those three are
  measured on published surfaces and graded columns rather than
  on the pair set, which is why they survive the denominator
  problem.

- **AND THE CARE-EPISODE MODELLING ITEM IS RETIRED BY A CLEAN
  NEGATIVE: THE ENGINE ALREADY REPRODUCES THE REAL SPREAD.** The
  open item was to publish a patient-level anchor plus a
  k-screened within-patient GAP distribution, because a date
  drawn from the GLOBAL marginal should spread a patient's visits
  across the whole cohort. Measured on the run and its own
  source, both crossed in words:
      visit_start_date span ratio, source     0.2994
      visit_start_date span ratio, generated  0.3053
  Within six thousandths - and CONFIRMED ON A SECOND ARM, the
  `--time-col visit_start_date` run, which reads **0.3144** on
  both of its date columns against the same source 0.2994. Two
  arms with different time axes landing within 0.015 of the
  source is what makes this a measurement rather than one draw. **There is nothing to model here** -
  the engine's date dynamics (a patient-level level with an icc)
  already carry it - so the anchor-and-gaps route is not a
  fidelity gap on this extract and the price it would have cost
  the date column's published marginal is a price for nothing.
  Written down as REFUTED rather than quietly dropped, and
  measured on the extract rather than on either fixture, which
  disagreed with each other and with the data.
- **AND THE END-DATE TIME AXIS HAS A SECOND CONSEQUENCE NOW,
  measured at 0.8888.** Detection picks the date with the most
  distinct values and chose `visit_end_date`, which this file
  already records as misordering overlapping stays. The
  within-patient date reorder therefore sorts by the END date and
  moves the start dates with it, so `visit_start_date` comes out
  in row order on **88.9%** of patients rather than all of them -
  the 11% are the patients whose stays overlap, where start and
  end are not co-monotone. The same choice also decides WHICH
  member of a date identity gets computed, since the tier
  protects the axis: here it made `visit_start_date` the child of
  `visit_end_date - span_days`. A `--time-col visit_start_date`
  arm would test both at once and is the obvious next run.

- **THE FIRST REAL-EXTRACT RUN WITH THE IDENTITY TIER: THE
  ABSOLUTE STOP CLEARS ON ITS OWN, SURFACES GO FROM ZERO, AND
  `close` FALLS - AND THE FALL IS A DEFECT I HAD ALREADY FIXED
  (2026-11-08 night, run_n8g).** Same source as the deliverable
  (`phase2_800\tidy_visits.csv`, 55,428 rows confirmed by a count
  whose verdict crossed as a WORD), same settings, no `--lags`.
  Every figure below read back spelled by the machine:
      criterion          deliverable        run_n8g
      direction kept  134/137 = 97.8% P  118/125 = 94.4% P
      close           120/137 = 87.6% F  100/125 = 80.0% F
      inverted            1 FAIL             0 PASS
      coverage           42/42 P            42/42 P
      shapes tracked      7/8 F              6/7 F
      interaction surfaces 0/5 F             2/5 F
      dynamics tracked   17/87              35/89
  **`inverted` clears without any repair** - the one criterion
  this project calls an absolute stop, which the deliverable
  needed a reordered column to pass. **Interaction surfaces go
  from NONE to two of five**, which is the pressure family
  sitting on the arithmetic exactly as predicted. **Dynamics
  double.**
- **AND THE `close` FALL IS THE TWO SPURIOUS IDENTITIES, WHICH IS
  WHY THIS RUN IS THE BEFORE ARM RATHER THAN A RESULT.** The
  blueprint published FOUR identities and two of them claim a
  child that IS one other column, with dropped set scaffolding as
  the second parent - the float-noise ratio defect recorded
  above, found by this very run and fixed before it finished.
  Both children, `procedure_count` and `active_drug_count`, are
  HUB columns that appear in many token pairs, and overwriting
  them from columns the reader never receives is the obvious
  candidate for 20 lost close pairs. A paired arm on the fixed
  detector is the test; quoting 80.0% as the engine's number
  would be quoting a defect.
- **THE DENOMINATOR MOVED 137 TO 125, so the percentages are not
  the comparison** - which this run's own prediction said to
  expect. In absolute terms close went 120 to 100 and the related
  count 137 to 125.
- **THE IDENTITIES, ASKED OF THE WRITTEN FILE RATHER THAN OF THE
  ENGINE'S REPORT:**
      visit_start_date == span_days + visit_end_date    1.0000
      age_at_visit == f(year_of_birth, visit_end_date)  0.6267
      procedure_count == f(quantity, a drug indicator)    None
      active_drug_count == f(a procedure, active_drugs__n) None
  The first is exact. The two `None`s are the confirmation from
  the output side: their PARENTS are scaffolding dropped before
  the file is written, so the identity cannot be checked in the
  deliverable at all - published, enforced, and unverifiable.
- **CORRECTION, SAME NIGHT: THE 0.6267 WAS MY OWN TOLERANCE, NOT
  THE FILE, AND IT IS WITHDRAWN.** I first read
  `age_at_visit == f(year_of_birth, visit_end_date)` as holding
  on 0.6267 of rows and wrote that up as "the honest number for a
  near-identity". It is an artifact of the reader: it compared
  the written child against the prediction at **2% of the
  column's spread**, which on a nineteen-year spread is 0.38 - and
  the tier ROUNDS a whole-number child, so a row whose un-rounded
  error was 0.4 is written 0.4 away and is correct. Compared at
  half a unit, which is what rounding can move it by, both
  identities on the real extract hold on **1.0** of rows.
  Two measurements of one thing disagreeing is this project's
  alarm and it fired here: the engine's own report said the
  column had been computed on every row while the reader said a
  third of them were wrong. The reader was wrong.
- **AND WHAT HOLDING MEANS HERE IS NARROWER THAN THE CALENDAR,
  which is worth saying since neither number answers it.** What
  holds on 1.0 is the LINEAR identity the blueprint published and
  the tier enforced. Whether `age_at_visit` equals the visit YEAR
  minus the year of birth in the calendar sense is a different
  question, and this extract carries no visit-year column to ask
  it with - the record's **21.4%** figure was an exact-integer
  comparison against such a column on a different frame, so it is
  not the before-number for this 1.0 either. The engine now
  reports `held_before` - how often the DRAWN value already
  satisfied the identity - which is the comparable thing and
  cannot be satisfied by the enforcement itself.

- **THE REAL EXTRACT FOUND A DEFECT IN THE IDENTITY TIER THE SAME
  DAY IT SHIPPED, AND NO FIXTURE HERE COULD REACH IT (2026-11-08
  night).** The first real-extract fit with the tier published
  **FOUR** arithmetic identities, and only two are identities:
      child              parents                  best single
      visit_start_date   span_days, visit_end_date      0.9019
      age_at_visit       year_of_birth, visit_end_date  3.7358
      procedure_count    procedure_quantity, a drug     7.9e-15
      active_drug_count  a procedure, active_drugs__n   8.7e-15
  The bottom two children ARE their first parent, exactly:
  `procedure_count == procedure_quantity` and
  `active_drug_count == active_drugs__n`. The second parent in
  each is dropped SET SCAFFOLDING - a `__has__` indicator that is
  not even in the written file - and it earns nothing.
- **THE CAUSE IS THAT `PAIR_GAIN` IS A RATIO, AND BOTH SIDES OF
  IT WERE FLOAT NOISE.** The rule requires the pair to beat the
  best single parent by a factor of four, which separates a real
  identity from a one-parent one by orders of magnitude when the
  single residual is a MEASUREMENT - 0.90 and 3.74 above. When
  the child is an exact copy of one parent, both residuals are
  numerically zero, a least-squares fit with three free
  parameters lands below one with two, and the ratio passes.
- **AND IT IS A COIN FLIP, which is why one fixture would have
  lied either way.** On the same copy-of-a-column frame at float
  magnitude the pair-over-single ratio reads **0.0125 at one seed
  and 0.267 at another** - so `PAIR_GAIN` refuses it SOMETIMES.
  Every fixture here used integer counts, where both residuals
  are exactly 0.0 and `0 >= 0` refuses correctly; it takes float
  values for the noise to separate at all.
- **THE FIX IS A FLOOR UNDER THE DENOMINATOR, and the rule is now
  a PURE FUNCTION checked on planted numbers.** The single
  parent's residual must be a measurement rather than noise -
  above `1e-9` of the child's own spread - before the ratio is
  trusted. Checked on the real extract's own four pairs: 0.90 and
  3.74 pass, 7.9e-15 and 8.7e-15 are refused whichever way their
  ratio happens to fall, a genuine one-parent relationship is
  still refused by the ratio, and a candidate with no
  single-parent fit is refused rather than treated as a perfect
  one. Three mutations - dropping the floor, dropping the ratio,
  treating a missing single fit as success - each turn checks
  red.
- **AND REFUSING IS THE RIGHT ANSWER, not computing it elsewhere.**
  A child that IS one other column is an EQUALITY, and this
  project already repairs an equality by COPYING rather than
  drawing twice; a count that is a set's length belongs to the
  size identity, which also copies. Had the spurious pair been
  enforced, `procedure_count` would have been computed from
  columns the reader never receives, and two tiers would each
  have owned one column - the shape this file records as `a <= b`
  and `b <= a` undoing each other.

- **THE CARE-EPISODE QUESTION IS ANSWERED ON THE REAL EXTRACT: A
  PATIENT'S VISITS SPAN THIRTY PERCENT OF THE COHORT'S RANGE
  (2026-11-08 night).** Earlier today I named this as a property
  of the real extract that NEITHER fixture here could answer, and
  wrote down the route rather than guessing. Measured on
  `phase2_800\tidy_visits.csv`, every figure crossed in WORDS
  through `sayn.py` because the digits channel off that screen
  turns a leading zero into a nine:
      parsed rows                      55,428
      distinct visit_start_date values  4,692
      mean within-patient span          2,371 days
      cohort span                       7,919 days
      ratio                             0.2994
  Two of those - the row count and the 4,692 distinct dates -
  match this file's own earlier readings of the same extract
  exactly, which is what makes the other two worth quoting:
  reading the right file and parsing it soundly are both
  confirmed before anything is concluded.
- **SO A PATIENT IS NEITHER AN EPISODE NOR THE WHOLE COHORT, AND
  BOTH OF MY FIXTURES WERE WRONG IN OPPOSITE DIRECTIONS.** The
  fixture I built with cumulative gaps makes every patient a
  tight episode (ratio near zero); the repo's extract-shaped
  tidy fixture reads **0.500**; the real extract reads **0.299**
  - about six and a half years per patient inside a
  twenty-two-year cohort. An engine that draws each visit date
  from the GLOBAL marginal tends toward a ratio near 1.0, so the
  gap is real and now has a TARGET rather than a guess: 0.30,
  not the 0.05 one fixture implies nor the 1.0 the other does.
- **AND IT RETIRES THE CORRECTION I MADE THIS MORNING WITHOUT
  OVERTURNING IT.** The withdrawn claim was a 0.74-against-0.98
  within-patient autocorrelation quoted from a fixture of my own
  construction; withdrawing it was right. What was missing was
  the real number, and this is it. The modelling route stands as
  written - a patient-level anchor plus a k-screened within-
  patient GAP distribution, drawn as a cumulative sum - with its
  price unchanged: the date column's global marginal becomes a
  convolution rather than the published quantiles, which the
  gate measures.

- **PREDICTION FOR THE FIRST REAL-EXTRACT RUN WITH THE IDENTITY
  TIER, ON RECORD BEFORE IT RUNS (2026-11-08 night).** Source
  `phase2_800\tidy_visits.csv`, confirmed at **55,428 rows** by a
  count whose verdict crossed as a WORD, which is the file the
  deliverable run was built from. Same settings as that run - no
  `--lags` - so the gate numbers are comparable to its close
  **87.8%** (121/137 after a one-column repair, 120/137 without).
  What is new in this tree: arithmetic identities ENFORCED by the
  rules engine, a patient's dates put in visit order, a lag
  parent no longer folded onto the same visit, and the
  sklearn-1.9 sparse-slice fix.
- **WHAT IS CLAIMED.** (1) The run completes - the sparse fix
  matters here, because this stack is scikit-learn 1.9 and a
  wide frame with sparse columns is exactly what crashed the
  shape sweep. (2) The narration names at least one arithmetic
  identity on this extract, and `age_at_visit` against
  `year_of_birth` is the one to look for, since the record
  measures that identity holding on **21.4%** of generated rows.
  (3) Every patient's `visit_start_date` comes out in the order
  the file numbers its visits, where the engine previously
  agreed on a minority of patients.
- **WHAT IS NOT CLAIMED: that `close` clears 87.9%.** On a
  fixture the identity tier moved `interaction surfaces` 1/2 ->
  2/2 and `shapes tracked` 4/9 -> 5/9 and left close untouched,
  and the real extract's close gap is dominated by CROSS-set-token
  pairs that no identity touches. The pressure family sits on the
  arithmetic, so `interaction surfaces` (0/5 on the record) and
  `shapes tracked` (7/8) are where movement is expected - and
  "expected" is not "claimed", which is the distinction three
  previous predictions here got wrong by stating a magnitude.
- **AND THE DENOMINATOR MAY MOVE, which would make a percentage
  not a comparison.** Enforcing an identity changes a column's
  values, so a pair that was related at 0.1 may cross the floor
  either way. Read the ABSOLUTE counts first; this file has twice
  recorded a conclusion drawn from a percentage whose
  denominator moved.


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

- **THE SCRUB'S FIRST REAL READ GRADED THE DETECTOR, NOT THE
  DATA (2026-09-22).** Of five non-clear lines on the real
  extract, one was a true catch (visit_id, correctly an
  identifier) and three were detector faults: visit dates flagged
  as identifiers (the date regex's digits-and-hyphens shape plus
  4,692 distinct values - the fixture's date control had NINE
  distinct values against a rule that binds at 0.9x the patient
  count, so no fixture run had ever reached the rule), a
  semicolon-joined set column misread as free text by the length
  rule, and the identifier line printing "one per person" beside
  55,428 distinct over 800 patients - a line contradicting the
  numbers it carries. Each fault reproduced from the measured
  statistic (300 distinct dates, 85-char set strings), watched
  red, then fixed: dates are never identifiers, set columns are
  judged by their TOKENS - which also made a set of email
  addresses catchable where the joined string matches no pattern
  - and identifier lines state the measured ratio. The
  cap-no-fixture-reaches lesson, again, in a new module's first
  week.

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
- **TEN OF ELEVEN ELEMENTS NOW MEASURED ON THE REAL EXTRACT, AND
  THE ONE LEFT IS THE ONE A RUN CANNOT SUPPLY (2026-11-08).**
  The linkage element moved from OWED to CLEAN because the
  arrangement was SEARCHED FOR rather than guessed. `--scan`
  walks bandings coarse to fine and reports what each does; on
  the real extract it found the window at **16 bands per numeric
  quasi-identifier**:
      selections                            542
      honest rows scored                 10,863
      control (verbatim republish)       +0.0746
      honest release                     +0.0084
      share of the republish               11.3%
      verdict                              CLEAN
  The control is well over the 0.02 floor, so a verbatim
  republish IS detectable under those bands - which is what
  makes the honest 11.3% worth reading. Every figure was read
  back spelled by the machine.
- **THE TWO HAND-TRIED ARRANGEMENTS BOTH FAILED, AND AT
  OPPOSITE ENDS.** Coarse bands gave 36 selections of ~1,500
  released rows each, where a republish averages the target's
  own value with 1,499 others and the control reached only
  +0.0108. Fine bands with five quasi-identifiers gave 1,852
  selections and the honest arm scored ZERO. The window between
  them is narrow and was never going to be found by guessing -
  each guess costs a round trip to the machine that holds the
  extract, which is the whole argument for the ladder.
- **AND `residual` STAYS OWED BECAUSE IT SHOULD.** It is a
  statement about SCOPE that a person writes; a run cannot
  supply it and deriving one would be inventing the bound on
  somebody else's determination.
  `docs/DETERMINATION_SCOPE.md` carries the measured facts with
  every judgment marked for review. Ten elements of evidence on
  the real cohort, each with the control that makes it worth
  reading, is the strongest position this project has been in
  to ask a qualified person for a determination - and it is
  still an ask.

- **THE EXPERT DETERMINATION RUN OF RECORD EXISTS: NINE OF
  ELEVEN ELEMENTS MEASURED ON THE REAL EXTRACT, AND THE TWO GAPS
  ARE NAMED (2026-11-08).** `synthkit determination` against the
  real 800-patient run with the membership, attribute and
  linkage batteries all pointed at the extract itself:
      [x] exactly what is released, and what is not
      [x] the de-identification method, described
      [x] that direct identifiers were addressed
      [x] the anticipated recipient and what they can SEE of
          the release
      [ ] ...and in COMBINATION with other reasonably available
          information                                     OWED
      [x] measured risk of determining membership
      [x] measured risk of inferring a sensitive field
      [x] that the release does not reproduce individual records
      [x] that the published statistics themselves do not
          disclose
      [x] the information retained, and why it is useful
      [ ] residual risk and the scope of the analysis     OWED
  Every present element is evidence about THIS cohort at THESE
  caps - the package marks anything else FIXTURE or CONFIG and
  refuses to count it.
- **THE NUMBERS BEHIND IT, each with the control that makes it
  worth reading.** Membership at the released configuration
  (token cap 800, three seeds): worst **0.6253 MARGINAL**, mean
  0.5469, driven by the weights-surface adversary, with the
  verbatim-republish control at **1.000 CAUGHT**. The weights
  are never part of any release; the output-surface arm is the
  release surface. Attribute disclosure on the real extract with
  a real diagnosis column as the sensitive field: excess
  **-0.0055** - the adversary does NO better on members than on
  non-members - with the leaky-generator control firing at
  **+0.2289**, a forty-fold separation and the strongest
  attribute result this project has, every previous figure
  having been on a fixture. Memorization: held-out
  reconstruction **1.1196** of training, against a 9.7 positive
  control on a frame with nothing to learn.
- **AND THE TWO GAPS ARE DIFFERENT IN KIND, which is why they
  are not one line.** `linkage` is OWED because the measurement
  is NOT YET MEASURABLE at this cohort size, not because it
  failed - coarse bands leave a verbatim republish undetectable
  (control +0.0108, under the floor) and fine bands leave the
  honest arm with nothing scored (control +0.082, zero rows
  matched). `residual` is OWED because it is the one element a
  RUN cannot supply: somebody has to write the scope, and
  `docs/DETERMINATION_SCOPE.md` is a template with the measured
  facts filled in and every judgment marked for review, not a
  scope.
- **WHAT THIS IS NOT, said here as the tool says it: READINESS
  IS NOT A DETERMINATION.** The determination in 164.514(b)(1)
  is made by a qualified person. Nine elements of evidence
  assembled on the real cohort is the strongest position this
  project has been in to ask for one, and it is still an ask.

- **A FIFTH LINKAGE OUTCOME TURNED UP ON THE REAL EXTRACT, AND
  IT HAD BEEN MISLABELLED IN A FIXTURE ALL ALONG (2026-11-08).**
  With fine bands on the real extract the CONTROL fired cleanly
  at **+0.082** - well over the floor - and the HONEST arm
  scored **ZERO** rows: no member's quasi-identifier combination
  had as many as five matching rows in the released file. The
  verdict was right, UNREADABLE, and the REASON was "the control
  did not beat the non-member arm" - which would send a reader
  to fix a control that was working perfectly. Adding the branch
  then RECLASSIFIED this repo's own tiny-release fixture, which
  had been carrying the wrong reason since it was written.
- **AND IT IS NOT SAFETY, which the verdict now says.** A
  release holding no row that matches a real person's
  combination at that granularity is an interesting finding in
  itself - but an absent measurement is an absent measurement,
  and reporting it as clean would be the worst kind of pass.
- **THE REAL-EXTRACT LINKAGE READING IS THEREFORE: NOT YET
  MEASURABLE AT THIS COHORT SIZE.** Three arrangements were
  tried. Coarse bands (36 selections over 56,065 rows, so ~1,500
  released rows per selection): the control reaches only
  +0.0108, under the 0.02 floor, so a verbatim republish is
  undetectable. Fine bands with categorical quasis (1,852
  selections): the control fires at +0.082 and the honest arm
  scores nothing. There is a window between those where the
  measurement would work, and finding it is the next step - the
  honest statement today is that the linkage element stays OWED
  on this extract, not that it passed.

- **CORRECTION, SAME NIGHT: THE CAP EFFECT WAS A SINGLE-SEED
  CLAIM AND THE SECOND SEED MOVED THE OTHER WAY.** The entry
  below says the real-extract attack's output-surface adversary
  read 0.498 at cap 60 against 0.561 at cap 800 and called that
  0.06 AUC "in exactly the predicted direction". Seed 1 then
  read **0.521 at cap 60 against 0.485 at cap 800** - DOWN by
  0.036. Two seeds, opposite signs:
      seed   cap 60   cap 800   move
         0    0.498     0.561   +0.063
         1    0.521     0.485   -0.036
  So the MAGNITUDE and even the SIGN of the cap's effect on this
  adversary are NOT established. The CONFIG guard still stands
  on its own reasoning - a larger cap publishes more vocabulary
  about members, so the two runs measure different things and
  neither is evidence for the other - but the number I quoted to
  justify it is withdrawn. This is the single-seed fault, in an
  entry written minutes after I recorded the same rule for
  somebody else's benefit, and the only reason it was caught is
  that the third seed was still running and I kept reading.

- **THE ATTRIBUTE BATTERY HAD NO REAL-EXTRACT PATH, SO A
  COMPLETE DETERMINATION RUN WAS IMPOSSIBLE (2026-11-07, ED-3).**
  It only ever ran on its own cohort fixture - and the package
  correctly marks fixture evidence FIXTURE, because a number
  about a different dataset is not evidence about this one. So
  the element could never read present on a real run, by
  construction. `--csv` with `--sensitive` and `--quasi` closes
  it, and the two judgments are REQUIRED rather than guessed:
  which column is sensitive, and what an adversary is assumed to
  hold, decide which attack is being measured, and guessing them
  would measure an attack nobody is worried about.
- **AND IT REFUSES A QUASI-IDENTIFIER THAT COERCES TO NOTHING,
  rather than waiting for the fault this file already records.**
  `_encode` runs every quasi through `_num`, which returns None
  for a category or a date, so naming one hands the adversary an
  EMPTY column and the lower accuracy then reads as PRIVACY.
  Refused by name, with exit 2 and no traceback.
- **AND THE PACKAGE WAS ACCEPTING AN EXCESS WHOSE CONTROL NEVER
  FIRED.** `assess` required a non-empty `positive_control` list
  - which a control that ran and caught NOTHING satisfies. The
  first real-extract run printed "THE POSITIVE CONTROL DID NOT
  FIRE" at +0.019 excess and the element would still have read
  present: the check-that-cannot-fail shape, which the linkage
  element already refuses via `control_caught`. The flag is
  READ, never re-derived - the bar lives at the battery's own
  decision site, and computing it in both places is how a
  privacy threshold came to be applied with one tail bound and
  reported with another.
- **AND AN ABSENT FLAG STILL COUNTS, because absent is not the
  same fact as did-not-fire.** A record predating the flag
  cannot be judged on it; a new one must say True. Same
  distinction this file draws for `id_column` and for empty
  sets, and the mutation that treats absent as failure turns
  four checks red.
- **AND THE CAP MISMATCH WAS QUOTED FROM ONE SEED - SEE THE
  CORRECTION ABOVE, WHICH WITHDRAWS THE NUMBER.** The
  real-extract attack was launched twice on the same data, once
  at the shipped token cap of 60 and once at the 800 the
  released run used. Seed 0: output-surface adversary **0.498 at
  cap 60 against 0.561 at cap 800**, with the weights arm
  0.619 and 0.625. A larger cap publishes more vocabulary about
  members and the number moves by 0.06 AUC in exactly that
  direction - so the CONFIG flag is not a theoretical nicety,
  and an attack measured at a smaller cap understates the risk
  of a larger-cap release.
- **AND TWO MUTATIONS LOOKED LIKE WEAK CHECKS AND WERE MY OWN
  ERRORS, AGAIN.** One changed a message prefix from "STOPPED:"
  to "note:" while the asserted words lived in the same string
  either way - and the check deserved the criticism anyway,
  because it asserted the MESSAGE rather than the decision; it
  now runs the CLI with a categorical quasi-identifier and
  requires exit 2. The other named an anchor that did not exist,
  so the mutation never applied and its zero was a non-result.
  Assert count==1 before every replacement, which is the oldest
  editing rule here.

- **THE STATUTE SAYS "ALONE OR IN COMBINATION WITH OTHER
  REASONABLY AVAILABLE INFORMATION", AND OUR OWN PACKAGE WAS
  ANSWERING THE NARROWER HALF (2026-11-07, LK-1,
  `synthkit/linkage.py`).** The determination package cited two
  adversaries for that element - one that sees generated rows,
  one that assumes the weights leaked - and NEITHER holds any
  external data. So it asked the regulation's question and
  measured something else, which is the fault class this project
  refuses everywhere. Linkage is its own element now, and
  `threat_model` SAYS its adversaries hold nothing external.
- **A MATCH IS NOT AN IDENTIFICATION AGAINST SYNTHETIC OUTPUT,
  so the question had to be translated rather than imported.**
  The classic attack matches a released record to an external
  one and reads off the sensitive field; against synthetic rows
  nobody's record is there, and calling a match an
  identification would overstate the risk as badly as ignoring
  it understates it. What survives: an adversary holding
  quasi-identifiers for a person they can NAME selects the
  released rows matching that combination - does that tell them
  this person's value better than the same procedure applied to
  someone never in the cohort? Only that EXCESS belongs to the
  release, which is the argument the attribute-disclosure
  measure already makes with the external knowledge made
  explicit instead of assumed away.
- **AND MY OWN BAR WAS ABOVE ITS OWN CEILING.** The first cut
  called it a leak above 0.10 sd of excess. Measured over six
  seeds, a VERBATIM REPUBLISH - the worst generator there is -
  produces only **0.026 to 0.104 sd**, because a selection of at
  least five rows averages the target's own value with four
  other people's. A 0.10 bar would therefore have missed most
  republishes: the exact error this project found in a campaign
  tier demanding 0.700 over a 0.698 ceiling and wrote a rule
  about. The verdict is a FRACTION of what the republish
  achieves on the same cohort with the same bands, which
  self-calibrates - a coarser banding lowers both arms together.
- **THREE WAYS A CLEAN NUMBER IS WORTHLESS, AND THEY ARE THREE
  DIFFERENT FAULTS.** The control did not beat the non-member
  arm (no control at all). The control beat it but only barely -
  under a 0.02 floor - so a republish is undetectable under
  those bands and CLEAN would be a pass from a test that cannot
  fail (no POWER). And the honest excess is NEGATIVE, meaning
  the adversary did WORSE on members, so there is no advantage
  to express as a share - the first cut printed "-818%" for that
  case. Each gets its own verdict and its own sentence, because
  each takes a different action.
- **AND `caught` HAD TO REQUIRE A POSITIVE CONTROL EXCESS, not
  merely a larger one.** Testing only `control > honest` let a
  NEGATIVE control excess read as caught, and the output then
  said "the control CAUGHT the republish but only by -0.0310
  sd" - a verdict contradicting the number in its own sentence.
  A republish must HELP the adversary; one that makes them worse
  off has not been caught.
- **AND THE NO-POWER BRANCH HAD TO BE SEARCHED FOR, because no
  fixture here reached it.** After the `caught` fix both failing
  fixtures routed to "did not beat", leaving the power branch
  unreachable - a dead branch inside a tool built to catch
  unreachable behaviour. Found by sweeping arrangements: two
  coarse quasi-identifiers with no banding give a control excess
  of +0.008, positive but under the floor. Pinned there.
- **AND A CHECK'S LABEL MUST NOT FORMAT WHAT ITS MUTATION
  REMOVES.** The banding mutation makes `share_of_control` None,
  and the check formatted it as a percentage - so the suite
  CRASHED with a TypeError inside its own label rather than
  reporting a failure. Exit 1 either way, but no FAIL line for a
  grep to find, which is the crashed-suite shape this file
  already records. Guarded, and the same mutation now produces
  three proper red checks.
- **EVIDENCE MEASURED ON A DIFFERENT DATASET IS NOT EVIDENCE
  ABOUT THIS ONE, AND THE DETERMINATION PACKAGE NOW SEES IT
  (2026-11-07, ED-2).** The attack batteries default to their own
  cohort FIXTURE, and this project's own posture calls a fixture
  result "a floor, not a certificate, and not a release decision
  for a different cohort". A determination is scoped to ONE
  release, so citing a fixture number for it is the strongest
  form of the measured-at-one-shape fault - not a stale number
  but a number about somebody else. An element whose evidence
  names a source that is not the run's reads **FIXTURE**: the
  measurement is real, so it is not OWED, and it does not count
  toward readiness either.
- **AND FIXTURE MUST NOT ALSO READ OWED.** The first cut computed
  the owed list as "not present", which included the fixture
  ones - so the output said FIXTURE on one line and OWED on
  another about the same element. Two statements disagreeing is
  this file's alarm, and inside a governance document it is
  worse than elsewhere. Separated, with a check that the two
  lists cannot intersect.
- **AND IT DOES NOT FLAG ON A HUNCH.** A run with no recorded
  source has nothing to compare against, so the evidence reads
  present rather than suspect - absent is not the same fact as
  mismatched, the same distinction this file draws for
  `id_column` and for empty sets.
- **THE REAL-EXTRACT ATTACK PRINTED AND WROTE NOTHING, so the
  one run with the strongest membership evidence would have read
  OWED.** `latent_challenger.py attack --csv` is the measurement
  at real width - the record's 0.514/0.523/0.551 with the
  republish control at 1.000 came from it - and it had no `--out`,
  so nothing could reach a package that needs a FILE. It writes
  the summary in the shape `membership_new_path.py` already
  writes, deliberately: a third shape for one measurement is how
  a reader comes to need three parsers, and this module had
  already lost a day to reading field names that were never
  written. The record also names WHICH ARM the positive control
  controls, because feeding the members back as synthetic can
  only move the adversary that READS synthetic rows - the
  reconstruction adversary is blind to the output and cannot be
  controlled that way.
- **AND THE WRITE DIED ON A MISSING `io` IMPORT, which is the
  third instance of that exact shape.** `latent_challenger.py`
  does not import `io` at module level, so the attack ran to
  completion, printed its whole measurement, and then raised at
  the write step - the same fault as a receipts command that
  printed a perfect report and died at its write, invisible
  because stderr is unbuffered while piped stdout is not. Plain
  `open` needs no import.

- **THE LAST UNACCOUNTED DP STAGE IS CLOSED, AND ITS COST IS A
  SHAPE COST THAT MORE EPSILON CANNOT BUY OFF (2026-11-07,
  DP-3).** The latent mixture was the one stage that is neither
  a count nor a quantile - a density estimate fitted on codes of
  real rows. DP-EM is not the route: every iteration is a
  release and they compose, so a mixture fitted to convergence
  spends a budget proportional to the iteration count.
  `dp_gaussian_fit` is a SINGLE release instead - clip the
  codes, publish a noisy mean and a noisy covariance, sample a
  Gaussian - and it is written as a drop-in for the engine's own
  `_mixture` so the cost can be measured through the real
  generate path.
- **THE MOMENTS ARE CHEAP AND THAT IS THE TRAP.** On a two-lump
  latent fixture, 800 points:
      arm           gap mass   mean[0]    sd[0]
      source           0.020    -0.013    3.114
      mixture          0.022         -        -
      DP eps 1         0.391    -0.018    3.073
      DP eps 3         0.391    -0.026    3.069
      DP eps 10        0.391    -0.029    3.067
  The mean and the spread come back to three decimals at every
  epsilon, because they are averages over hundreds of points and
  the noise falls away. What is lost is the HOLE a mixture exists
  to leave empty - twenty times the source's gap mass - and no
  moment can see it. The same lesson as a zero-inflated count
  whose mean was exact while 7 points of mass sat in the wrong
  place.
- **AND THE COST IS EPSILON-INDEPENDENT, WHICH IS THE ACTIONABLE
  PART.** 0.391 at epsilon 1, 3 and 10 alike: the loss is the
  MODEL CLASS a single release forces, not the noise. A reader
  told only that DP costs something here would reasonably try a
  larger budget and get nothing for it. The mixture arm tracking
  the source at 0.022 is what makes the attribution safe - a
  one-armed measurement could not tell the mechanism's gap from
  the fixture's.
- **THE EIGENVALUE FLOOR FIRES HARD AT WIDTH, AND REPORTS.** A
  noised covariance need not be positive semi-definite, and
  sampling from one that is not either raises or returns
  nonsense. Measured: 0 eigenvalues floored at 2 dimensions,
  7-11 at 20, and 29-31 at 60 - barely moving with epsilon, so
  again structural rather than budget-limited. A large repair
  means the budget is small for that width and the run says so
  instead of handing back a plausible cloud.
- **AND THE NOISE SYMMETRY IS A PRIVACY PROPERTY, NOT A TIDINESS
  ONE - WHICH THE FIRST CHECK COULD NOT SEE.** The Gram matrix
  is symmetric, so its free entries are the upper triangle and
  the noise must be drawn there and MIRRORED. Drawing
  independent noise on all d-squared entries and symmetrising
  afterwards AVERAGES two draws into every off-diagonal entry
  and cuts that noise by sqrt(2) - delivering LESS noise than
  the sensitivity analysis claims, which is a mechanism
  overstating its own privacy. The released matrix looks
  identical either way, so a check on its shape passed on the
  broken version. The observable is the noise MAGNITUDE across
  seeds on a fixed input: empirical sd 0.1142 against a
  predicted 0.1122, where the averaged version gives 0.0793.
- **AND THE UNIT OF PRIVACY IS A REQUIRED ARGUMENT.** The engine
  fits THREE mixtures: one over patient CENTRES, where one point
  is one patient and the sensitivity is patient-level by
  construction, and two over ROW codes where a patient
  contributes many points. That difference is the whole content
  of the guarantee, so `unit` has no default and four
  near-misses are refused by name. A clip taken from the data is
  also a query on the data, so defaulting it REPORTS that a
  release would want the clip fixed in advance.
- **WHAT IS NOT MEASURED, NAMED: whether the REAL latent cloud
  is lumpy.** The fixture was built lumpy on purpose so the cost
  would be visible at all. Whether an 800-patient clinical
  extract's latent space has holes a mixture is representing is
  a property of that cohort and needs a run on the machine that
  holds it. Quoting the 20x gap as a statement about the extract
  would be the measured-at-one-shape fault.

- **THE EXPERT DETERMINATION EVIDENCE EXISTED AND NOBODY HAD
  ASSEMBLED IT (2026-11-07, ED-1, `synthkit determination`).**
  `docs/COMPETITIVE.md` carried Expert Determination as a flat
  `no` against a vendor that has one, with the note that it is
  "not a model change: a documentation and evidence package" -
  and most of what 45 CFR 164.514(b)(1) asks for has been
  measured here for months, spread across six files. A reader
  deciding whether the re-identification risk is very small had
  to hold all of it at once. That is the same complaint the
  per-claim receipts answered for patterns, one level up.
- **IT REFUSES TO CALL ITSELF A DETERMINATION, AND THAT IS IN
  THE DATA RATHER THAN ONE RENDERER'S PROSE.** The statute puts
  the determination in a qualified person's hands; a tool
  claiming one would be the same fault as an output described as
  differentially private when only the weights are accounted.
  `NOT_A_DETERMINATION` is a tuple every renderer carries, so
  none can drop it, and a READY package says READINESS and
  nothing stronger.
- **`OWED` IS A FIRST-CLASS ANSWER, AND THREE WAYS IT MUST STILL
  FIRE ARE PINNED.** A package that looks complete and is not is
  worse than no package, because an expert signs against
  evidence that does not exist. One adversary is NOT a threat
  model (the output surface and the weights surface are
  different recipients whose recorded numbers differ sharply);
  one seed is a DRAW, not a measurement (the battery itself
  refuses fewer than two); and an excess with no positive
  control is the check-that-cannot-fail shape this project has
  already been burned by. The CLI exits 3 while anything is
  owed, the same reasoning as `scrub --apply` refusing while a
  fresh detect still finds something.
- **AND THE WRONG-KEY FAULT HAPPENED A SEVENTH TIME, IN THE ONE
  PLACE IT WOULD HAVE BEEN WORST.** The first cut read `arms`,
  `membership` and `attribute` out of a single invented
  `attack.json`. The real batteries write TWO files with
  different shapes - `membership_new_path.py` writes
  `worst_auc_max` and a `runs` list whose entries carry
  `nearest_neighbor` and `likelihood`, while
  `attribute_disclosure.py` writes `seeds` and
  `positive_control` - so every element would have read OWED on
  a run that HAD the evidence, and the package would have told
  an expert that measurements this project has published for
  months did not exist. The suite now calls both batteries LIVE
  (about a second each) and asserts the names, which is the fix
  this file already prescribes and which was not applied until
  the fault recurred.
- **AND `residual` WAS UNCONDITIONALLY FALSE, WHICH MADE THE
  READY BRANCH DEAD CODE.** Residual risk is the one element a
  RUN cannot supply - it is a statement about SCOPE that a
  person writes - and marking it permanently owed meant `ready`
  could never be true, so the every-element branch was
  unreachable inside a tool built to catch exactly that. It
  takes a written scope now and refuses a one-line one, because
  a scope nobody wrote is not a scope and deriving one would be
  inventing the bound on somebody else's determination.

- **THE FOUR UNACCOUNTED DP STAGES ARE COUNTS AND QUANTILES, SO
  THE GAP WAS NEVER A RESEARCH PROBLEM - AND MEASURING THE PRICE
  SETTLED THE POSTURE (2026-11-07, DP-2).** The weights epsilon
  covered one stage of five. Four of the rest are patient counts
  (which levels and tokens get modelled, which identities are
  detected) or empirical quantiles (the published bounds, the
  visit-count and set-size grids), and both have standard
  mechanisms. Built: a Laplace count, a Laplace and a Gaussian
  histogram, a selection with a RAISED floor, and quantiles and
  bounds by the exponential mechanism.
- **SENSITIVITY IS PATIENT-LEVEL, AND THE HISTOGRAM IS THE ONE
  THAT IS EASY TO GET WRONG.** One patient moves a single
  token's count by 1 - but they hold MANY tokens, so their
  removal moves many counts at once and the L1 sensitivity is
  the number they hold, UNBOUNDED unless it is clipped. The cap
  is a parameter the caller must IMPOSE and pass, not something
  measured from the data, because measuring it would itself be a
  query on the data. A mechanism whose sensitivity claim is
  wrong does not give a weaker guarantee; it gives none.
- **A SELECTION NEEDS A RAISED FLOOR, NOT NOISED COUNTS.**
  Releasing WHICH items cleared a floor is a different query
  from releasing their counts: an item held by exactly k
  patients appears or vanishes on one person's presence, so the
  SELECTED SET is itself a release, and noising the counts while
  keeping the floor leaves an item at the floor flipping with
  probability near one half. How far it must rise is the result
  (tau 1e-6, delta 1e-5, against this project's floor of TEN
  PATIENTS):
      eps  tokens/patient |   laplace    gaussian    ratio
      1.0              10 |     151.2        92.8    1.63x
      1.0              50 |     716.1       222.8    3.21x
      1.0             200 |    2834.5       535.7    5.29x
      3.0              50 |     278.7       114.3    2.44x
     10.0              50 |     125.6        76.3    1.65x
- **AND GAUSSIAN BEATS LAPLACE BY sqrt(L), WHICH IS NOT A
  REFINEMENT AT THESE WIDTHS.** A patient holding L tokens moves
  the count vector by L in L1 and sqrt(L) in L2; Laplace's
  scale is set by L1 and the Gaussian's by L2. At a vocabulary
  of hundreds it is the difference between a usable floor and an
  unusable one, and the advantage GROWS with width - which is
  the property a check pins, because a mechanism that happened
  to be better on one cell proves nothing.
- **THE POSTURE IS NOW DEFENDED BY A NUMBER RATHER THAN AN
  ARGUMENT.** `scripts/dp_select.py` on an 800-patient
  clinical-shaped vocabulary of 892 tokens, 40 per patient at
  the 95th percentile:
      floor   tokens   share
         10      320  100.0%   the k rule
        196       19    5.9%   DP eps 1
         99       39   12.2%   DP eps 3
         65       57   17.8%   DP eps 10
  **DP selection at epsilon 3 keeps 12% of the publishable
  vocabulary.** The k-anonymous aggregate release is therefore
  not a shortcut taken instead of DP: on a cohort this size DP
  selection at a usable epsilon publishes almost nothing, and a
  generator with no vocabulary is not more private, it is
  broken. DP on the WEIGHTS is affordable and ships; DP on the
  SELECTION is affordable at a cohort size this project does not
  have.
- **TWO HALVES COMPUTED THE SAME PRIVACY THRESHOLD SEPARATELY,
  TWICE, AND ONE OF THEM WAS WRONG.** `select_above` APPLIED the
  raised floor and `required_floor` REPORTED it - with different
  Laplace tail bounds, so the number a reader was shown was not
  the number being used. Found by a check and the implementation
  disagreeing about the arithmetic; the tight one-sided bound is
  b times log(0.5/tau) because the failure that matters is an
  item being INCLUDED when it should not be, while the other
  direction costs utility and leaks nothing. Then the SAME fault
  turned up for the Gaussian: `gaussian_histogram` applied one
  scale and `required_floor` reported another, so a mutation
  scaling the noise by L1 instead of L2 changed the noise, left
  the published floor untouched, and broke NO check. One
  `gaussian_scale` function now, and both are pinned to agree.
- **AND A CAP OF ZERO WOULD HAVE MEANT NO NOISE AT ALL.**
  sqrt(0) is 0, so a zero cap makes the Gaussian sigma ZERO and
  releases the exact counts from a function named for a privacy
  mechanism - a silent total failure, not a weak guarantee. The
  guard existed and was NOT in the refusal check list until a
  mutation removing it broke nothing. It is refused inside
  `gaussian_scale` rather than at each call site, so a future
  caller cannot miss it.
- **AND THE SHELL ATE THE BACKTICKS OUT OF THIS ENTRY ON THE
  FIRST ATTEMPT.** Passing the text as a `python -c` argument
  let zsh read every `` `name` `` as command substitution, so a
  version with every function name STRIPPED was written to the
  file and the helper's own confirmation line then raised on an
  unrelated NameError - a mangled write that announced itself as
  a crash about something else. Reverted and redone by writing
  the entry to a FILE and reading it, so nothing passes through
  shell interpretation. Same family as the wrapped-phrase trap:
  the text that reaches the file is not the text you typed.


- **DIFFERENTIAL PRIVACY RAN ON THE REAL EXTRACT, AND IT COSTS
  TWICE WHAT THE FIXTURE PREDICTED (2026-11-07, epsilon 3.00).**
  Two arms on the 800-patient extract, same split, 46,288
  training rows and 9,140 held-out people, 1,601 encoded
  dimensions. Every figure below was crossed in WORDS before
  being written down, because the digits channel off that screen
  turns a zero into an at-sign:
      arm                 holdout   train   ratio   x plain
      plain               0.00625  0.00559   1.12      1.00
      epsilon 3.00        0.02213  0.02022   1.09      3.54
  The fixture sweep predicted **1.78x** at epsilon 3. And the
  epsilon itself had to be recovered from a second fact: the log
  rendered `eps 3.9000` where `sigma 3.413` is what
  `sigma_for_epsilon(0.05, 1500, 3.0)` returns to the digit
  while 3.90 returns 2.713 - so the filename and the sigma agree
  on 3.00 and the OCR did not.
- **THE HOLDOUT RATIO IS THE REASSURING HALF, AND IT IS WORTH
  STATING BESIDE THE COST.** 1.09 under DP against 1.12 plain on
  the same held-out people: DP generalises marginally BETTER,
  which is what the theory says, measured rather than assumed.
- **WIDTH WAS THE OBVIOUS EXPLANATION AND IT IS REFUTED.** The
  cost expression carries `sqrt(params)` and the sweep varied
  only ROWS, so every recorded number was measured at ONE width
  - 30 dimensions against the extract's 1,601, and
  sqrt(1601/30) = 7.3 looked like more than enough. `--width`
  was built to confirm it and says the opposite:
      width    control       dp    x control
         30    0.02135  0.10976       5.14
        120    0.15821  0.63698       4.03
        480    1.00144  0.94336    REFUSED
  The ratio FALLS as width grows. Hypothesis dead, and recorded
  as dead in the script rather than quietly rewritten.
- **AND THE 480 CELL WAS NEVER MEASURABLE, REPORTED AS 0.94 -
  BETTER THAN THE CONTROL.** On a standardised frame an MSE of
  1.0 means the net predicted the mean, and 1.00144 is what the
  CONTROL reads there: both arms learned nothing and the ratio
  was noise over noise, which is the same fault as a recovery
  ratio between two noise numbers reading as a result. Not
  under-training either - 1500, 6000 and 20000 steps give
  1.00154, 1.00148, 1.00144, identical to four decimals. A cell
  whose control does not learn is REFUSED by name now, which is
  the guard `smoke_dp` already applies to the trainer and the
  sweep was missing.
- **AND A CHECK PINNED THE CLAIM THE MEASUREMENT REFUTED - THE
  FIFTH SUCH CHECK THIS MONTH.** `smoke_dp` asserted that the
  cost script says the price is "dominated by DATASET SIZE", so
  correcting that sentence to name BOTH axes turned the suite
  red. A check written around a HYPOTHESIS goes red when the
  hypothesis is refuted, which is backwards: it now pins that
  both sweeps are stated, that the real-extract numbers carry
  their split, and that the unlearnable cell is refused. The
  refusal is asserted as a DECISION on planted inputs rather
  than as prose - a text check for `CONTROL_MUST_BEAT` passes
  while the guard is bypassed, because the name appears at both
  the definition and the use site, so the one mutation that
  mattered went uncaught until the function was made pure.
- **WHAT IS STILL UNEXPLAINED, NAMED RATHER THAN GUESSED.** The
  extract's plain reconstruction is 0.00625 against the
  fixture's 0.0195 - the real frame is three times EASIER to
  rebuild, being mostly sparse one-hot token indicators - so a
  fixed noise budget may cost relatively more against a smaller
  error. A candidate, not a measurement, and the third guess
  does not get made: it earns a measurement or it stays named.
- **AND THE SCOPE NOTE LANDED ON REAL DATA, WHICH IS THE POINT
  OF IT.** The run printed, on the machine that holds the
  extract, that this is NOT an end-to-end guarantee, that the
  output must not be described as differentially private, and
  that five other stages touch the real data unaccounted - the
  published numeric bounds, the latent mixture fitted on real
  codes, and the k-screened visit-count and set-size grids among
  them. An epsilon that covers the WEIGHTS is evidence about the
  surface the weights-leak adversary reads, and not a release
  certificate.

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

- **THE LATENT CHALLENGER EXISTS, AND ITS FIRST READ SAYS THE
  INTERACTION CEILING IS OURS, NOT THE PROBLEM'S (2026-09-24).**
  `scripts/latent_challenger.py`: a bottleneck autoencoder over
  the k-screened frame (sub-k levels folded, numerics clipped to
  the blueprint's own k bounds BEFORE training, output clipped
  back inside them), a Gaussian mixture over the latents,
  hand-decoded - sklearn only, row-level, NO dynamics, built as
  a fidelity-at-any-cost RULER. Three fixtures, three seeds
  each: the triangle's attribution lands 75-79% against a 76/23
  source where the blueprint wanders 69-90 with a flip lottery;
  the colinear pressure family reads close 63/63 with zero
  inversions where the blueprint's surfaces fail; the planted
  core reads sign 900/900, close 94%, and the XOR interaction
  SURVIVES on every seed (gain 0.56 -> 0.38-0.86, noisy) while
  the 3-way mostly dies. The memorization tripwire (nn-ratio,
  generated-to-training over holdout-to-training) reads
  0.95-1.18 with a positive control that reads 0.00 on a
  verbatim republish.
- **THE LATENT PATH TOOK THE MEMBERSHIP ATTACK AND PASSED AT THE
  FIXTURE, WITH A MORE DANGEROUS ADVERSARY (2026-09-24).** Same
  cohort fixture, same bands as the blueprint's recorded 0.52:
  worst AUC 0.527-0.546 over three seeds - and the likelihood
  adversary here is RECONSTRUCTION ERROR through the trained
  weights, which assumes the weights themselves leak. The
  republish control reads 1.000 FAIL, so the pass is a
  measurement. One simple cohort at one k: a floor of a floor,
  and the attack must re-run at real width before it means more.
- **THE RULER CAUGHT THE WRONG-TYPE FAULT IN ITSELF, and scaling
  found its own inversion cost.** span_days holds seven distinct
  day-counts; the v1 gate sent it down the categorical branch and
  argmax decoding collapsed it to a constant - the named triple's
  shares read 0/0 while everything looked healthy. Numeric-valued
  columns are numeric at any cardinality now, categoricals sample
  proportionally instead of argmax, and capacity scales with the
  frame (an 8-dim bottleneck at 77 columns read close 61%; scaled,
  97-99%). After the fixes, at width: the flip triple's driver
  shares transfer as 35/65 -> 42/58 (ordering held, ~7-point
  drift), the XOR gain is nearly exact (0.545-0.570 against
  0.556) - and the latent path produced 2-4 INVERSIONS among
  ~1,375 pairs where the blueprint holds zero. The ruler has its
  own failure mode, in the failure class this house treats as
  worst; neither path dominates, and the comparison table must
  carry both columns.

- **THE REAL HEAD-TO-HEAD RAN (2026-09-24), AND THE INTERACTION
  VERDICT IS A ROUT.** On the real 300-patient sample, the miner
  found the source's top-8 two-way interactions - all in the
  span_days / condition_count / active_drug_count /
  procedure_count family, the exact territory of the operator's
  SHAP complaint - and the BLUEPRINT READS 0.000 ON EVERY ONE.
  Not weakened: absent. The latent path holds the strong ones at
  source magnitude (0.131 -> 0.133/0.131 on two seeds, one
  overshoot 0.192) and loses only the <=0.02 tail. Driver
  attribution: source 94/6, blueprint 25/75 (the flip, live),
  latent 93/7 / 89/11 / 91/9. By the compare's uniform pair
  metric the blueprint also showed 20 inversions of 443 to the
  latent's 1-6 of 325 (the GATE's own count on the same run is
  1 of its 51 related pairs - different denominator and
  definition, both true, say which one is quoted). The
  memorization tripwire read 1.52-2.14: no copying signal; the
  autoencoder smooths. Still owed before any posture change:
  the membership attack at real width, and dynamics - the
  latent path has none.
- **THE ATTACK AT WIDTH FAILED THE LATENT PATH, AND THE RELEASE
  SURFACE IS THE WHOLE QUESTION (2026-09-24).** On the 77-column
  extract-shaped fixture: reconstruction adversary 0.776 / 0.694
  - FAIL under the weights-leak assumption, where the narrow
  cohort fixture had read 0.546 PASS. Width is where autoencoder
  memorization lives, and a pass measured at the narrow shape
  was worth exactly what the one-scale token solve was. The
  row-level adversary - the attacker who sees only GENERATED
  ROWS - still reads 0.500. So the honest sentence has two
  halves: output-only release, clean at 0.50; any release of the
  model itself, FAIL at 0.78. AND the republish control's worst
  now EQUALS the honest arm's (both dominated by the
  reconstruction adversary, which never reads the synthetic
  rows) - two arms that agree are one arm, so at width the
  control controls only the nn adversary, said out loud.
- **THE FORK IS THEREFORE DECIDED BY EVIDENCE: LATENT-INFORMED
  SURFACES, NOT A SECOND ENGINE.** The challenger's interaction
  fidelity is real and the blueprint's zero is structural - but
  promoting the latent path to a release engine would require
  solving width-scale memorization first. The winning move the
  numbers point at: MINE interactions the challenger's way,
  publish them as k-screened aggregates the blueprint samples
  from - importing the fidelity while keeping the posture that
  passed its attacks. The challenger remains the ruler.

- **THE PARALLAX WAS INVISIBLE BECAUSE IT WAS APPLIED WHERE
  NOTHING SCROLLS (2026-09-28).** Four layers were built,
  measured and pinned - and the operator saw nothing, because
  the layers were the station header and footer while the
  longest scroll in the bench, the Roadmap, is made entirely of
  `.goal` cards that were outside the layer list. Past the first
  screen there was literally nothing moving. A motion effect
  must be applied to the thing that actually travels, and
  "verified at five scroll positions" on a SHORT station proved
  nothing about a long one. Fixed by making a card's head a
  layer (38px against its own body, card 16px the other way) and
  raising every maximum, since the first magnitudes were also
  too timid to see.
- **EVERY TERM IS A DOOR (`synthkit/explain.py`).** The bench is
  full of words carrying real machinery - the k rule, the latent
  space, a membership adversary, an interaction surface - and a
  reader who did not already know them had no way in. One
  registry of 25 entries, served to the bench: a term becomes a
  quiet underline, clicking opens a panel explaining it plainly,
  the panel's own words are doors too, and a trail across the
  top climbs back out. Auto-linking marks the FIRST occurrence
  per section only and refuses to touch anything inside an
  input, a `pre`, a button or an output box. Every `[[slug]]`
  must resolve or the checks fail - a link that goes nowhere is
  worse than no link, because the reader trusted it.
- **THE ROADMAP WAS A WALL OF TEXT, AND THE FIX WAS
  STRUCTURE.** Each goal was one dense paragraph per state with
  the state as an inline prose prefix; the operator could not
  read it. Now each goal is a card with a labelled two-column
  row per state - BUILT / MEASURED / PLANNED / NEXT in their own
  column, body at 62 characters, key numbers bold. The check
  moved with it: it asserts the row LABELS and that the old
  inline `BUILT:` prefix is GONE, rather than asserting a
  formatting that no longer exists.

- **I READ THE REFERENCE'S LAYOUT AND MISSED ITS MOTION, AND THE
  FIX WAS TO MEASURE THE MOTION (2026-09-26).** Handed a scroll
  reference as a gif, I extracted four still frames, redesigned
  the bench from them, and never asked what MOVED - the operator
  had to point out that the page is layered and the layers
  scroll at different rates. A gif is not watchable from here,
  but it is MEASURABLE: cross-correlating horizontal bands
  between consecutive frames recovers each band's vertical shift,
  and on that reference the bands differ by 15-25% (-22px at the
  top against -27px in the middle on one transition). Stills
  answer layout; only a frame diff answers motion, and the
  question "what moves" has to be asked out loud or it is not
  asked at all.
- **DEPTH BY DIFFERENTIAL RATE, WITH THE FORM AS THE ANCHOR.**
  Four layers scroll at distinct bounded rates - the eyebrow
  lags most, then the folded prose, then the display heading,
  while the what-next footer leads slightly. THE PANEL NEVER
  MOVES: a label that drifts away from its own input is a broken
  form and no amount of depth is worth it, so the form is the
  fixed thing everything else moves against. Displacement is
  symmetric about the viewport centre and clamped, so a layer
  returns to true position exactly where the eye reads it and
  nothing accumulates drift. The editorial reference's 15-25%
  would need ~130px of lag across a screen and would visibly
  detach the header from its panel; this lands near 7%, which is
  the honest translation of the effect into a tool. Verified
  live at five scroll positions rather than by eye: banner
  11.8->26.0px, fold 6.5->17.0, heading 2.4->8.8, footer
  11.4->1.6 the other way, panel `none` throughout - and
  `prefers-reduced-motion` returns `none` for all of them.

- **THE NEURAL ENGINE HAS PATIENTS NOW, AND THE MEASURE THAT
  PROVES IT ALSO NAMES WHAT IS STILL WRONG (2026-09-25).** It
  used to emit no `person_id` at all - a table of unlinked
  visits, which on a source averaging 76 visits per patient is
  not degraded longitudinal structure but the absence of one.
  Each row's latent code now splits into the PATIENT's centre
  and that visit's DEVIATION from it; generation draws a
  synthetic patient centre once, draws a k-screened visit count
  for them, and places their visits around that centre, so every
  visit of a generated patient shares a "who". Identities are
  INVENTED (S000001 upward), never a real person's, and the
  visit-count tail is clipped to the mean of the k most extreme
  patients' counts - the blueprint's own bound rule applied to a
  count. Measured on the 77-column fixture: person-level columns
  come through at 0.90-0.96 against a source 1.00, per-visit
  columns read 0.07-0.14 against a source near 0 (a mild
  tendency to make visit-level things too patient-like, the
  residual), and the mean absolute between-share error over 41
  columns is 0.079.
- **AND SHRINKING THE VISIT DEVIATION WAS SWEPT AND LOST.** It
  sounds like pulling a patient's visits tighter around their
  own centre should RAISE the between-patient share; through the
  decoder's nonlinearity it does the opposite. dev_scale 1.0 -
  the measured decomposition, untouched - read mean between-share
  error 0.079, then 0.108, 0.166 and 0.298 at 0.8, 0.6 and 0.4,
  with `close` falling too. A hand-rolled version of the same
  experiment suggested otherwise and was WRONG because it
  skipped the blur in the generate path: a measurement that does
  not go through the real code path is not a measurement.
- **A BETWEEN-PATIENT CHECK ON A PER-ROW COLUMN IS VACUOUS.**
  The first patient-structure check asserted that a column kept
  its between-patient share - on a fixture where every column
  re-rolls per row, so the share is ~0 whether or not the
  generator has patients at all. Zeroing the patient centre
  changed nothing and the check stayed green. The fixture now
  plants `plevel`, a genuine property of the person (0.99 of its
  variance between patients), and the same mutation takes it to
  -0.02 and the check goes red.

- **DENOISING TRAINING BUYS PRIVACY AND FIDELITY AT ONCE, AND IS
  THE DEFAULT (2026-09-25).** The weights-leak membership
  adversary read 0.776 FAIL at 77-column width. Training the
  autoencoder to rebuild a CLEAN record from a CORRUPTED one -
  input noise sd 0.3 - takes it to 0.569 PASS while `close`
  RISES from 1332/1376 to 1354/1376 and inversions fall 4 to 3;
  sd 0.6 reads 0.563 with close 1352. Noise during training
  regularizes, so the network learns the structure instead of
  the records: the same change is the privacy fix and a fidelity
  gain, which is why the default follows the measurement rather
  than a trade-off argument.
- **A SAMPLING-TIME DEFENSE CANNOT MOVE A WEIGHTS-TIME ATTACK,
  AND THE MEASUREMENT SAID SO IN THREE DECIMALS.** The k-aware
  blur governs which latent points are DRAWN; the reconstruction
  adversary scores candidate records straight through encode /
  decode and never reads a generated row. Blur on and blur off
  returned byte-identical AUCs - which by this file's own rule
  means one arm, not two, and here the reason is structural
  rather than a bug. Defenses must be built on the surface the
  attack reads.
- **THE OUTPUT-SURFACE ADVERSARY IS BLIND AT REAL WIDTH, AND ITS
  CONTROL PROVED IT.** The nearest-neighbor attack reads 0.500
  on the honest arm at 77 columns - and 0.500 when handed the
  members VERBATIM as its "synthetic" output. A control that
  cannot catch a literal republish is a check that cannot fail,
  so every reassuring 0.500 that adversary has produced at width
  is worth nothing. Distance concentration in ~150 one-hot
  dimensions is the cause. Found only because the control was
  made to report the arm IT controls: `worst_auc` had been
  reporting the reconstruction number for both arms and hiding
  the blind one behind it. An output-surface attack that
  survives width is now an open item.
- **THE K-AWARE CONTRACT IS ALREADY MET, AND THE MEASURE HAD TO
  BE FIXED TWICE TO SEE IT.** The operator asked for
  reconstruction close on average and NOT close for patterns
  fewer than k records support. Measured properly, the
  generator already does this: rare rows sit 1.09x FURTHER from
  their nearest synthetic row than dense rows do, and the blur
  moves that ratio by -0.00 while costing +0.1% of average
  fidelity. Getting there took two corrections, both the same
  lesson - ask the k question where the pattern lives. Support
  in the LATENT space found sub-k regions on 0.3% of rows;
  moving it to the FEATURE space (a rare COMBINATION OF VALUES,
  which is what k-anonymity has always meant) found 3.8%; and
  counting distinct patients among a FIXED 40 neighbors can
  never isolate a small group at all - the neighbor list spills
  into the population and a cluster of four patients still
  reports ten - so the count is taken within a RADIUS, which
  found 25.3%. The blur ships as insurance that costs nothing,
  with its benefit honestly recorded as unmeasured.
- **AND A SCRIPT THAT ASSERTS THEN WRITES AT THE END LOSES EVERY
  EDIT WHEN A LATER ASSERT FAILS.** An edit script made the
  padding fix and a header rewrite, then died on a later
  assertion - and because the file write is the last statement,
  NONE of it landed while the console showed only the traceback.
  The page was then debugged for a defect that had never been
  applied. Write after each edit, or read the file back before
  believing an edit is in.

- **THE NEURAL ENGINE HAD NEVER SEEN A SHAPE THAT WAS NOT THE
  EXTRACT, AND THE SWEEP FOUND FOUR FAULTS (2026-09-29).**
  `shape_sweep.py --neural` runs it across the same 23 dataset
  shapes the rules engine was swept across long ago. First run:
  one CRASH and three DESTROYED columns. The crash was a table
  with ONE entity - `GaussianMixture` requires at least two rows
  however few components are asked for, so a single patient
  centre cannot use one at all, and the fix is a stand-in that
  says what the distribution of one point is. The three
  collapses were a high-cardinality code, a free-text note and a
  date, all folded entirely into `__other__` because no level
  cleared the k floor - the SAME lesson the rules engine learned
  months ago, relearned in a new engine: a level the k rule
  cannot publish is not a reason to publish nothing, so the
  SHAPE is published (distinct count, frequency profile) and the
  labels are INVENTED. 23 of 23 clean afterwards, with both
  classes pinned as checks and both mutations watched red.
- **THE KIT MUST CARRY WHAT THE TOOL CARRIES.** It taught the
  rules engine only - types, scrub, fit, gate, exam, sign-off -
  while the neural engine was the one being developed and shown
  to the room. And it never told a reader they could point any
  of it at their OWN csv, which is the first thing a teammate
  wants to do. Both are in it now, the second with the 23-shape
  result stated so "it should handle your data" is a measured
  claim rather than a hope, and the try-to-break list asks for
  the 24th shape.

- **A SET COLUMN'S LEVELS ARE ITS COMBINATION STRINGS, AND THE
  NEURAL ENGINE HAD NO CAP ON THEM (2026-09-29).** The blueprint
  has capped categories at 60 since it was written; LatentGen
  kept every level that cleared k. On the real extract that
  turned 44 source columns into 4,326 encoded dimensions and
  training died allocating 1.5 GiB inside the denoising noise.
  Fixed four ways: the same 60-level cap (k first, then
  FREQUENCY, rest to `__other__`), float32 instead of float64
  for the biggest array in the process, the denoising noise
  added IN CHUNKS rather than materialising a second full frame,
  and the encoded width PRINTED with its widest columns named so
  the next blow-up is visible before it is fatal. Measured
  after: a frame whose `conditions` holds 22,393 distinct
  combination strings encodes to 425 dimensions, widest
  categorical 61.
- **AND THE FIRST FIXTURE FOR THAT CHECK COULD NOT REACH THE
  CAP.** It planted 900 level strings held by 2 patients each -
  none cleared the k floor, everything folded to `__other__`,
  the width was 1, and removing the cap entirely changed
  nothing. A level cap is only exercised by levels that QUALIFY:
  120 strings on ~15 patients apiece, and the same mutation then
  reads 120 wide against a cap of 60 and goes red.

- **THE JOINT MEASURES EXIST, AND THE FIRST ONE FOUND THREE
  DEFECTS IN AN HOUR (`synthkit/jointcheck.py`, 2026-09-28).**
  Every fidelity measure before this NAMES a pattern first - a
  pair's rank correlation, a published curve, a two-way product
  - so it can only find what somebody thought to look for. Five
  measures now ask the reverse: distinguishability (can a
  classifier tell the tables apart, and WHAT gives it away),
  per-column predictability (is each column as learnable, and do
  the same drivers drive it), train-on-synthetic-test-on-real,
  three-way interactions, and manifold precision/recall. Within
  an hour of existing, the first one found: two columns emitted
  100% EMPTY (a reconstructed missingness indicator cut at a
  flat 0.5 empties any column absent more than half the time),
  integers emitted as floats, and continuous columns written to
  17 decimal places where the source has 4. Every enumerated
  measure had called that same output good.
- **AND THE SAME WRONG-TYPE BUG, A THIRD TIME, BY SPARSITY.**
  `height` is numeric, 4,155 distinct values, present on 46.6%
  of rows - and a flat `presence >= 0.5` gate sent it down the
  CATEGORICAL branch, where every value fell below the k floor,
  folded to `__other__`, and came out empty. Cardinality was
  fixed once and sparsity was still there. A column whose
  present values are numbers is numeric, however often it is
  present and however few distinct values it has.
- **A JOINT AUC SATURATES, AND READING IT AS A VERDICT WOULD BE
  WRONG TWICE.** Over ~150 encoded dimensions, differences too
  small to matter anywhere compound into 1.00, so the number
  cannot distinguish "slightly off everywhere" from "useless".
  AND the k rule guarantees separability by itself: 92% of real
  rows hold at least one value the published bounds forbid, so
  the classifier wins on privacy alone. Both are handled -
  each real column is CLIPPED into the synthetic's range (clip,
  never drop: dropping whole rows kept 4.8% of them and turned
  the test into "the most central real rows against all
  synthetic ones"), and every column is also scored ALONE, which
  is the comparable, actionable number. On the fixture: 46 of 76
  columns individually indistinguishable, with the rest named.

- **THE NEURAL OUTPUT NOW REACHES THE WHOLE BENCH, AND WIRING
  IT FOUND TWO CONTRACT BUGS (2026-09-28).** The Verdict
  station, the Dashboard and the sign-off page all key off
  artifacts the rules pipeline writes, so the neural engine's
  output could only be read by its own page - the same question
  about the same data, answerable by one instrument. The run now
  writes blueprint.json, catalogue.json, fidelity.json,
  generated.csv and provenance.json, and all three stations
  read it. Found on the way: a missing `catalogue.json` SILENTLY
  deleted the Dashboard's pattern cards (no error, just an
  absent section), and `source` written as a STRING instead of a
  dict killed the sign-off page with `'str' object has no
  attribute 'get'` - the provenance-shape lesson this file
  already recorded once. Both are checks now, both watched red.
- **AND THE RELATIONSHIPS BELONG TO THE SOURCE, NOT TO AN
  ENGINE.** The first cut wrote a blueprint with no claims, and
  the gate refused: no published relationships, no pairs, "the
  gate cannot be read - that is a finding in itself, not a
  pass". Correct, and the fix is that discovery runs on the REAL
  DATA: what the extract contains is a property of the extract,
  so "did this output keep it" is a fair question to put to any
  generator. The neural engine's first full gate reading is
  therefore honest and unflattering - it FAILS shapes (16/40)
  and surfaces (4/11) where the rules engine passes, because
  those criteria judge whether published CURVES survive and this
  engine publishes none. Different questions, both worth
  asking: it wins the combination effects mined from data and
  loses the curves declared in a contract.

- **THE ENGINE GETS ITS OWN STATION; THE CONTEST STAYS A
  RULER (2026-09-28).** Running the neural engine required going
  through the duel, which is backwards: a comparison is how you
  DECIDE between engines, not how you USE one. The Neural
  station generates and reviews with no prior run and no second
  engine - two polled jobs, since each is minutes of real work -
  and its review page is `build_neural`, the duel's own measures
  minus the comparison, so the two pages cannot disagree about
  the same output. The duel remains because the one open
  certification gap means a reference point is still needed, and
  because every future change to the engine has to be measured
  against something.
- **AND IT LEADS WITH THE PATIENT QUESTION.** The single-engine
  page puts "does a patient look like a person" ABOVE the
  per-column charts, asserted by position in the checks. A
  longitudinal table whose rows belong to nobody passes every
  per-column number there is; putting that question last would
  let the page read well while the file was useless.

- **THE DUEL IS A STATION, AND THE GAP IS ITS OWN EXHIBIT
  (2026-09-25).** Both engines against one source on one page:
  every shared column drawn as THREE bodies (gray original,
  cardinal blueprint, gold latent - the third series inherits
  the same press-and-hold fan), a GAP HEATMAP colored by which
  engine drifts further from the source's correlation in each
  pair (cardinal = blueprint worse there, gold = latent worse,
  white = same miss, depth = size of the difference), and the
  mined interactions as per-engine bars. `build_duel` is ONE
  function the CLI writes files with and the bench serves into
  its iframe - the same rule the deck has always had, because a
  second implementation is how the shipped artifact and the
  screen come to disagree. The latent side runs as a bench JOB
  over the same script CLI a terminal would call.
- **AND A NINE-MINUTE REQUEST IS A HUNG BROWSER.** The duel
  endpoint shipped SYNCHRONOUS, then the full-scale timing was
  measured: latent generation 3-7 minutes, the duel page 7-9
  (the mining is real work over every numeric child). Both are
  polled JOBS now with elapsed seconds on the loader -
  discovered by timing the thing rather than by a green suite,
  which is why the timing was run at all. At that scale the
  latent side also read close 618/630 with ONE inversion and
  nn-ratio 1.03, against 2-4 inversions at 300-patient width:
  more patients, better behavior - the opposite of the
  small-cohort token defect, and a reason to read the real
  full-extract duel rather than extrapolating from the sample.
- **A COMPARISON PAGE MUST CARRY THE POSTURE, NOT JUST THE
  SCORES.** The duel's footer states both models in place: the
  blueprint publishes k-screened aggregates and passed its
  attacks; the latent trains on records, FAILS the weights-leak
  adversary at width (0.78) while its rows-only adversary reads
  0.50, and is row-level with no dynamics. It also NAMES any
  column the latent flattens to a constant rather than letting
  an averaged score hide it. A page that showed only fidelity
  would argue for the wrong engine by omission.

- **THE NEURAL ENGINE READ EVERY DATE AND EVERY SET COLUMN AS A
  CATEGORY, AND DESTROYED BOTH (2026-09-29).** Found by reading
  the real extract's dashboard, not by any check.
  `pd.to_numeric` returns NaN for `2013-10-03`, so every date
  fell into the level path: `visit_start_date` went from 4,692
  distinct values and 0% missing to **60 and 79.9%** - sixty
  being the level cap, and the rest folded into `__other__`,
  which is not a date, so the deck read it back as missing. A
  set column is worse: its "levels" are its distinct COMBINATION
  strings, thousands of them, so nearly every row landed on
  `__other__` - a string that is NOT EMPTY and holds NO REAL
  TOKEN. Measured on a fixture with 400 tokens and a heavy tail:
  the old code emitted **one distinct token in the entire file**
  and 100% unparseable dates. On the extract, `procedures` was
  empty on 80.7% of source rows and 0.0% of generated ones, and
  `ondansetron` went 9.2% to 0.0%.
  Both are the same fault: the rules engine has parsed dates and
  expanded set tokens since the beginning, and the second engine
  learned neither. THE TWO HALVES COME TO DISAGREE WHENEVER A
  CAPABILITY IS REBUILT INSTEAD OF CALLED - this file says that
  about token screening and about the two categorical draw
  paths, and it now says it about a whole engine. Dates go
  through `synthkit.dates`; sets go through `synthkit.sets`, one
  publishable-size rule in one function both halves call.
      date unparseable   100%  ->  0%
      distinct tokens       1  ->  50 (of 365; 50 clear k)
      empty share        29.4% ->  36.0%  (source 36.1%)
      worst token share  0.208 ->  0.006
- **AND THE SIZE NEEDED THE POINT-MASS FIX, WHICH THIS FILE
  ALREADY HELD FOR COUNTS.** A set size decoded as a
  standardized float plus rounding cannot land a point mass, so
  36% genuinely-empty rows came out at 29.4% while the mean size
  ran short. Mapping the decoder's own ordering through the
  PUBLISHED size quantiles reproduces the mass exactly and keeps
  whatever ordering the network learned. The remaining 2.16 ->
  1.39 tokens per row is the k rule plus the token cap, printed
  by name in the run.

- **THE DELIVERABLE WAS THE SIZE OF THE TRAINING SPLIT, AND
  NOBODY DECIDED THAT.** `generate(len(tr))` - 85% - so the real
  run's dashboard opened with "55,428 rows / 800 patients
  original, 47,163 rows / 687 patients synthetic", which reads
  as the generator losing an eighth of the cohort. It also made
  the DUEL UNFAIR: the rules engine writes a full-size file and
  this one was scored against it at 85%. Holding 15% of patients
  back is right and stays - the nearest-neighbor tripwire needs
  real people the network never saw - but that is a decision
  about TRAINING, and it silently became a decision about the
  OUTPUT.

- **THE PREDICTION TO CHECK ON THE NEXT REAL RUN, ON RECORD
  BEFORE IT RUNS.** The failing run read: direction 18/21
  (85.7%), close 14/21 (66.7%), INVERTED 2, coverage 38/42,
  surfaces 0/8. Both inversions are `procedure_quantity <-
  condition_count` and `procedure_count <- condition_count` -
  every column in them is a set SIZE PARTNER, and the sets they
  size were being emitted as a single meaningless token. So:
  coverage should move up (`visit_start_date` alone was one of
  the four, at 79.9% missing against 0%), the two inversions
  should go, and the set token-share and EMPTY-rate lines should
  stay green while meaning something for the first time. If the
  inversions SURVIVE a fixed set path, they are not a set defect
  and the search moves to the count columns themselves. Written
  down first so the answer cannot be read backwards out of
  whatever comes back.
- **WHAT IS NOT CLAIMED: the SHAP attribution drift.** On that
  run `span_days` was driven 59.8/40.2 by procedure_count over
  condition_count in the source and 74.1/25.9 in the output - 14
  points. That is far milder than the rules engine's 94/6 ->
  25/75 on the same data, and this engine has NO trim machinery,
  so the rules engine's characterized mechanism (discovery's
  edge inclusion) cannot be the cause here. Nothing above is
  expected to fix it. Open, and named as open.

- **THE GATE GOT WORSE BECAUSE THE DENOMINATOR GREW, AND ONE
  REAL DEFECT WAS UNDERNEATH IT (2026-09-29).** The set/date
  build read direction 89/115, close 49/115, INVERTED 10 against
  the previous run's 18/21, 14/21, INVERTED 2 - and the
  operator reasonably read that as a large regression. It is
  two things at once and they must not be reported as one.
  FIRST, the run related **115 pairs where the previous one
  related 21**: the set columns were emitting a single
  meaningless token, so ninety-odd token-indicator pairs could
  not be compared at all and were silently absent from the
  denominator. In absolute terms direction-kept went 18 -> 89
  and close 14 -> 49. A criterion that was not even present
  before (`set token shares`, 70/71) appeared for the same
  reason. **A percentage whose denominator moved by 5x is not a
  comparison**, and a report that shows only the percentage
  invites the wrong conclusion.
  SECOND, the newly-visible pairs were genuinely failing, and
  for one nameable reason - below.
- **A SIZE IDENTITY THE ENGINE DOES NOT DECLARE STILL HAS TO BE
  FOUND.** `active_drug_count` IS the length of `active_drugs`
  on every source row - one causal direction, correlation
  1.000. The neural engine decodes it as an ordinary number,
  which is a SECOND independent draw of that same quantity, and
  two independent draws of one marginal agree only by chance.
  Measured on a fixture whose source identity holds on 100% of
  rows, the generated file held it on **37.2%**. Every one of
  the ten inversions was a set token indicator against its own
  count partner: the tokens track the set's ACTUAL size
  correctly (+0.277 against a source +0.260) and it is the
  PARTNER that drifted away from them, so a source +0.41 lands
  at -0.14 and reads as INVERTED.
  This file already recorded the identical lesson for the rules
  engine, where the fix was to OVERWRITE the partner with the
  set's real size after every column exists - copying enforces
  an equality, a second draw only exchanges the mismatch. The
  rules engine is TOLD the identity by its blueprint; this
  engine declares nothing, so it has to DETECT it: any numeric
  column equal to a set's length on >=99% of source rows is
  copied at generation. Identity 37.2% -> 100%, and the token
  correlations land on their source values.

- **THE IDENTITY FIX LANDED ON THE REAL EXTRACT AS PREDICTED,
  AND THE REMAINING GAP HAS A NAME (2026-09-29, build a381fd1).**
  INVERTED 10 -> 0, direction kept PASS, coverage PASS, token
  shares PASS, close 49/115 -> 88/115. The recorded prediction
  held exactly. What remains: close 76.5% against the 87.9% bar,
  and the drifted list is dominated by CROSS-set-column token
  pairs - has_sodium-chloride <-> has_Flush at 0.88 source, 0.20
  generated. A drug implies its route; the correlation was IN
  the decoder's activations and the DRAW was randomizing it
  away: Gumbel top-k adds noise of fixed ~1.28 sd to
  log-activations of the same order.
- **SHARPNESS IS THE KNOB, AND ITS SOLVE MUST SCALE WITH IT.**
  `TOKEN_SHARPNESS` multiplies the log-activations before the
  Gumbel; the per-token offset solve recalibrates marginals at
  any sharpness - PROVIDED its pass count grows too. Swept on
  the drug-implies-route fixture, three seeds per cell:
      sharp 1, 12 passes   |cooc drop| 0.276  share 0.008  nn 1.44
      sharp 4, 12 passes   |cooc drop| 0.105  share 0.052  nn 1.30
      sharp 4, 40 passes   |cooc drop| 0.115  share 0.014  nn 1.31
      sharp 8, 80 passes   |cooc drop| 0.097  share 0.015  nn 1.26
  Sharpening alone trades the marginals away (0.052); passes
  proportional to the push restore them. 4/40 ships. The
  memorization tripwire IMPROVED with sharpness rather than
  degrading - checked because a draw that follows the decoder
  more faithfully could in principle copy more.
- **AND THE ARITHMETIC IDENTITIES GET THE SAME TREATMENT AS THE
  SIZE IDENTITY.** MAP is (S+2D)/3 and age_at_visit is the visit
  year minus year_of_birth - arithmetic, not correlation.
  Decoded as ordinary numbers the identity held within 0.1 on
  40.2% of generated rows against 100% in source. Detected from
  the source by least squares over parent pairs (kept only when
  the residual sd is under 2% of the child's spread - a rule the
  source itself breaks is not a rule), enforced by COMPUTING the
  child from its generated parents, first claim wins so
  computations cannot chase each other. 40.2% -> 100%, centre
  exact, neighboring correlations in band.
- **PREDICTION FOR THE NEXT REAL RUN (before it runs):** the
  cross-set-token pairs (has_drug <-> has_route) should carry
  most of the close gap's improvement; the pressure-family
  identity should hold arithmetically; whether the 1/6
  interaction surfaces heal is NOT claimed - the failing
  surfaces involve the near-identical MAP variants, and the
  ring-adjacency ceiling is a recorded research item this fix
  does not address. If close still misses the bar after the
  co-occurrence fix, the remaining drift list names the next
  mechanism.
- **THE SHARPNESS BUILD CREATED AN INVERSION, AND THE MECHANISM
  IS THE COMPETITION, NOT THE SHARPNESS (2026-09-30, build
  429e60b).** The real run read INVERTED 2, led by
  `drug_routes__has__Oral <- active_drug_count` at 0.26 source,
  **-0.64** generated - a regression from the co-occurrence fix.
  Reproducing it took the right fixture: a simple broad token
  did NOT reproduce (corr held at every sharpness); a TWO-REGIME
  fixture did (acute visits: many drugs, none Oral; ambulatory:
  few drugs, Oral) - source corr(has_Oral, count) -0.25, and the
  competitive draw emitted +0.04 to +0.16 at sharpness 1 AND 4.
  ANY within-row top-size-many draw mechanically ties every
  token to the SET SIZE, so a token whose correlation with a
  count runs AGAINST the size mechanism cannot survive it,
  sharpened or not. The offset solve then redistributes the
  broad token into whatever rows its sharpened score wins -
  which on the extract was the small oral-only visits, hence
  the hard negative.
- **THE DRAW IS INDEPENDENT PER TOKEN NOW: each token takes its
  top share-of-rows by ITS OWN score.** No within-row
  competition; marginal shares exact by construction
  (round(p*n) rows per token); the EMPTY mass placed on the
  rows with least total activation; a non-empty row that picked
  nothing gets its single best token; the k size cap enforced
  by dropping weakest picks. Set SIZES are now EMERGENT from
  the picks - which also fixes a second fault found the same
  night: the old size draw rank-mapped a near-flat decoder
  channel through the quantile grid, and the rank of a flat
  channel is NOISE, which is why `condition_count <-
  conditions__n` read 0.89 -> 0.03. Measured: regime fixture
  sign restored (-0.19/-0.21 against source -0.25 on both
  seeds), cooc drop 0.101 at sharpness 4 (competitive: 0.115),
  worst share 0.018, empty gap 0.000, nn 1.28,
  routes_n~count 0.45 -> 0.64.
- **AND THE REPAIR CHAIN NEEDED THREE PASSES OF ITS OWN
  MEASUREMENT.** (1) Forcing a starved row's GLOBAL argmax
  piles the forced mass onto the strongest token - its share
  overshot by 0.098, twice the tolerance - so each over-quota
  token drops its weakest picks from rows keeping >=2.
  (2) Then the top token's count corr read -0.10, because the
  global argmax also CONCENTRATES forced picks on the smallest
  rows; forcing by the row's best PER-TOKEN-STANDARDIZED score
  spreads them. (3) The first cut of that fix read WORSE
  (-0.196), because the standardization mean was taken over the
  empty-mask rows whose scores were set to -1e18 - a sentinel
  in a mean is a poisoned mean, and the z-score collapsed back
  into global argmax. Standardized over live rows only:
  d000~count +0.206 against source +0.260, worst share 0.0027,
  empty exact. An instrument on the RAW picks (corr +0.283
  before any repair, source +0.26) is what confined the fault
  to the repair chain instead of the draw.
- **PREDICTION FOR THE NEXT REAL RUN (before it runs, replacing
  the previous one):** both inversions should clear - Oral by
  the mechanism above; `procedure_count <-
  mean_arterial_pressure_invasive` because procedure_count is
  copied from len(procedures) and sizes now carry latent signal
  instead of rank-of-noise. The size-partner pairs
  (`condition_count <- conditions__n` 0.03, `active_drug_count
  <- drug_routes__n` 0.14) should recover substantially. NOT
  claimed: the near-identical pressure variants (surfaces 0/6,
  `systolic <- MAP_cuff_bmdi` faded) - the ring-adjacency
  ceiling, shared by both engines, research item.

- **THE READER FOLDED EMPTY INTO MISSING, AND THE ENGINE'S OWN
  GATE COULD NOT SEE IT (2026-09-30, found on the morning of the
  demo).** The Dashboard's set sections read "empty on 80.7% of
  original rows, 0.0% of synthetic" with every token bar ~5x its
  source share - beside gate chips reading set EMPTY PASS 4/4
  and token shares PASS 45/45. Two numbers on one page
  disagreeing is this file's alarm, and it was right: the latent
  CLI's `df.replace("", np.nan)` folded present-and-empty set
  cells into MISSING, so the engine learned token shares from a
  source in which the medication-free visits did not exist -
  every share inflated by exactly 1/(1-empty), 5.2x on the
  80.7%-empty procedures - and the size grid lost its zeros, so
  no empty set was ever drawn. The gate PASSED both criteria
  because its reference numbers came from the same folded frame:
  CONSISTENT WRONGNESS IS INVISIBLE FROM INSIDE; only the deck,
  which reads the raw file, disagreed. `prepare()` had solved
  exactly this a month earlier (sets.EMPTY); the latent CLI
  hand-rolled its own reader - the capability-rebuilt-instead-
  of-called fault, again, in the same module family.
- **THE FIX IS A PREDICATE BOTH SIDES SHARE, AND THE CHECK GOES
  THROUGH THE DISK.** `sets.looks_like_set` is the same gate
  `vocabulary` applies, exposed so the reader and the model
  cannot class a column differently; `_read_source` folds "" to
  NaN for every column EXCEPT sets. Reproduced through the real
  path (CLI writes generated.csv, deck reads it): empty share
  0.000 -> 0.363 against a source 0.361, deck section and gate
  now agree. The smoke check runs the CLI against a CSV ON DISK
  - every in-memory fixture had passed a DataFrame straight to
  fit, which is why nothing here could catch a reader fault.
- **RUN-LEVEL NUMBERS DOWNSTREAM OF A READER FAULT ARE NOT
  COMPARABLE.** The 2026-09-30 real run's set-pair drift list,
  its metoclopramide<-count inversion and its close 55/83 were
  all measured against the folded source; re-read them only
  after a re-run on the fixed reader. The
  procedure_count <- MAP_invasive inversion is the piece the
  reader fault does NOT explain - it survived exactly as the
  prior prediction said it might, and is now isolated to the
  sparse invasive family.

- **THE READER-FIX RUN CAME BACK, AND THE PREDICTIONS READ
  ALMOST EXACTLY AS WRITTEN (2026-09-30 pm, build abe13cf).**
  coverage 38/42 -> PASS 42/42 (the four misses were the set
  columns); the metoclopramide inversion gone; the Dashboard's
  set sections and the gate AGREE (procedures empty 80.7% /
  81.1%, token bars at source scale); close 66.3% -> 76.6%
  (72/94); direction 96.8% PASS; the twin surface's gap 1.10 ->
  0.92, still FAIL, correctly unclaimed. And `procedure_count <-
  mean_arterial_pressure_invasive` SURVIVED (-0.31 -> +0.23),
  exactly the branch the prediction reserved: it is real, it is
  not the reader fault's shadow, and it has earned a run-side
  diagnostic rather than a fifth fixture.
- **A DRUG IMPLIES ITS ROUTE, AND INDEPENDENT PER-COLUMN DRAWS
  CANNOT SAY SO.** The remaining close misses cluster on CROSS-
  set-column token pairs (has_sodium-chloride <- has_Flush 0.88
  source, 0.58 generated; has_albuterol <- has_HH Neb 0.66 ->
  0.23) - near-FUNCTIONAL implications: the route column is
  essentially derived from the drug column. Each set column
  draws its tokens from its own activations, so cross-column
  coherence flows only through the latent and arrives at a
  fraction of its strength. No sharpness can fix a relationship
  the draw never consults.
- **THE IMPLICATION TIER: detect P(b|a) >= 0.9 from the source
  (on >= 3k rows holding a, b not near-universal), FORCE b
  where any implying a was drawn, then RESTORE the books** -
  b's quota by dropping its weakest non-forced picks from ANY
  row, rows the restore empties repaired with their best
  remaining token (the base draw's own starved-row rule), and
  the column's empty count restored by emptying the weakest
  unforced rows that forcing made non-empty. The first cut
  dropped only from rows keeping >= 2 - and a route column's
  rows mostly hold ONE route, so the excess could not drain:
  the broad token's share overshot 0.86 -> 0.92 and the
  surplus piled into high-score rows, flipping the regime
  fixture's guarded negative sign to +0.01. The suite's own
  Oral check caught it within the hour. On the
  deterministic drug->route fixture every token pair lands ON
  its source value (worst drop < 0.01 against 0.1-0.2 for the
  draw alone), the empty rate holds at 8.6% vs 8.5%, and the
  size pair n~n improves 0.65 -> 0.71 (source 0.91 - the
  derivation oracle reads 0.85, so more remains on sizes). The
  exactness is structural: forcing reproduces the source's own
  P(b|a)=1 and the restores pin the marginals, so the pair
  correlation is determined to be the source's.
- **A BROAD TOKEN KEEPS ITS OWN PLACEMENT - the tier adopts an
  implication only when the implied token's share is <= 0.5.**
  Forcing b wherever any implier lands replaces b's decoded
  signal with the UNION of independently-placed impliers, and a
  union cannot carry a macro signal: on the two-regime fixture,
  forcing Oral (share 0.86) via its ten implying drugs flipped
  corr(has_Oral, count) from the draw's own -0.13 to +0.19
  against a source -0.25 - an INVERTED verdict, manufactured by
  the fix for a close verdict. The suite's own sign guard
  caught it within the hour. For narrow tokens the pair
  correlation IS the co-occurrence, which forcing fixes; with
  the cap, the implication fixture still lands exact (worst
  drop 0.04) and the regime fixture's Oral sign comes out
  STRONGER than before the tier (-0.13/-0.16), because the
  narrow-token forcing improves the coherence Oral's own
  signal rides on.

- **PREDICTION v3 HELD ON ALL THREE CLAUSES, AND THE INVERSION
  CHAPTER IS CLOSED ON THE REAL EXTRACT (2026-10-13,
  run_oct135, build fcc6941).** The narration shows "reorder
  ran for mean_arterial_pressure_invasive: 114 rows, 12
  partner(s)" - the guard clause that the first two predictions
  lacked, and the reason this one could be believed. The gate:
  INVERTED 1 -> 0 PASS ABSOLUTE, direction 92 -> 93/94 (98.9%),
  close 76 -> 77/94 (81.9%) - the predicted at-most-one. The
  challenger's own uniform inversions nearly halved on every
  seed (31/20/26 -> 17/12/14): the tier reaches pairs beyond
  the gate's list. The pair that survived two fixes and three
  real runs - the sparse-conditional rank-of-noise class - is
  fixed by: a k-banded published grid for the marginal, ALL
  qualifying partners enforced jointly by rearrangement, the
  live mask anchored on the child's own rows, and the
  enforcement SAYING whether it ran. Three failed deployments
  taught three lessons already on file: fixtures must carry the
  real shape (dense decoys, then independent-sparse decoys,
  then co-measured siblings), caps exclude what the gate
  measures, and a fix that cannot prove it executed has not
  been deployed. M0 now fails on close (6 pairs), shapes (7/8)
  and surfaces (0/5, ring-parked) only - the drift list is the
  next coordinate source.

- **THE SECOND FIX ALSO NEVER RAN, AND SEVENTEEN IDENTICAL
  DIGITS WERE THE PROOF (2026-10-13, run_oct13).** The uncapped
  tier's real run came back with the INVERTED diagnostic at
  gen +0.24023546807476087 - BYTE-IDENTICAL to the previous
  build's. Two arms that agree to the last digit are one arm:
  detection now kept the procedure_count coupling (visible in
  the narration), but enforcement required EVERY partner
  present on the same row, and a sparse column's strongest
  partners are its sparse SIBLINGS - co-measured on the same
  ICU visits, overlapping enough to qualify at detection, and
  jointly emptying the all-present intersection at generation.
  The tier silently skipped, in BOTH runs; the first fixture
  missed it because its decoys were dense, and the second
  because its sparse decoys were INDEPENDENT of the child's
  rows - the real siblings are CO-MEASURED, which is both how
  they qualify and how they veto. Enforcement now anchors on
  the child's present rows alone, an absent partner
  contributes its standardized mean instead of a veto, and
  every reorder PRINTS ran-or-skipped with its row and partner
  counts - a silent skip is the read-but-not-used flag class,
  and it cost two round trips to the data machine. Eight seeds
  on the co-measured fixture: -0.202..-0.303 against source
  -0.341, sign 8/8, zero inversions, the tightest arm yet.
  PREDICTION v3, on record: INVERTED 1 clears; the narration
  must show "reorder ran for mean_arterial_pressure_invasive"
  with its row count; close moves at most a point or two from
  this fix alone.

- **THE PARTNER CAP CUT EXACTLY THE PAIR THE GATE MEASURES,
  AND THE CONVENTIONS FILE HAD ALREADY SAID IT WOULD
  (2026-10-12 night, run_oct125).** The sparse tier's first
  real run left the gate IDENTICAL - close 76/94, INVERTED 1,
  the same pair at gen +0.240 - and the run's own narration
  said why: mean_arterial_pressure_invasive's four partner
  slots filled with its colinear pressure siblings at
  +0.4..+0.6, and the -0.31 coupling with procedure_count -
  the pair this tier was BUILT for - ranked fifth and was cut.
  The token-quota entry above records this exact failure
  ("a cap could exclude exactly the partners the gate
  measures") and the cap was removed there; the same mistake
  was re-made in a new tier the same month. The cap is GONE
  (the joint solve bounds total variance at one unit, so
  breadth costs nothing; ridge raised to 1e-3 for the
  colinear family), and the fixture now carries four decoys
  shaped like the pressure family so the measured pair ranks
  fifth - eight seeds: sign kept 7/8, one flat, ZERO
  inversions, marginal exact. The landing fades when many
  partners share one unit of variance (-0.58 against a -0.94
  source on the decoy fixture), so the tier's stated contract
  is SIGN AND SUBSTANCE (>= 0.4x magnitude), not exactness -
  written into the check text itself. REVISED prediction for
  the next real run: INVERTED 1 clears via the now-kept
  procedure_count coupling; close moves at most a point or
  two; the ~17-pair close gap is a separate mechanism that
  the still-unread Dashboard drift list names.

- **THE SPARSE INVERSION IS FIXED ON THE BENCH, AND IT TOOK
  THREE MEASUREMENTS TO AIM (2026-10-12).** The fixture built
  from run_oct12's own diagnostic (0.2% presence, conditional
  corr -0.31, copied-set-size child, coupling +0.11) reproduced
  the inversion two seeds of eight - the fifth construction,
  and the first with those exact coordinates. The instrument
  then named the true fault in one column: GENERATED VALUE SD
  0.1-0.3 AGAINST A SOURCE 10.5 - a 94-training-row channel
  decodes to its mean, and a conditional correlation computed
  on near-constant noise flips sign by seed. Rank-of-noise,
  the recorded set-size class, in a numeric. Two layers fixed
  it. (1) Sparse numerics (presence under 5%) publish a
  K-BANDED GRID - sorted present values walked into bands of
  at least k DISTINCT PATIENTS, each point a band MEAN, never
  an individual's value - and generation rank-maps the decoded
  channel through it WITHIN the rows that survived the missing
  cut: the first cut ranked all rows then masked, and the
  present subset sampled a biased slice (mean 90.7 against
  73.3, sd overshot 2x). (2) The k-screened conditional rank
  couplings are enforced by REARRANGEMENT - and one partner
  under-covers (the token-quota lesson, again): the best
  coupling alone left the second-best to noise, and the
  second-best was the inverting pair. Up to four partners,
  jointly, b = R^-1 rho through the partners' own rank
  correlations. Eight seeds after: cond corr -0.22 to -0.36
  against source -0.34, ZERO inversions, marginal exact. And
  the edit itself hit the assert-then-write-at-the-end trap
  AGAIN - the probe "re-measured" an unchanged file; write
  after each edit, then verify the text is IN the file before
  believing any measurement of it.

- **THE RUNS OF RECORD CAME BACK (2026-10-12, build 2b16b18):
  THE ATTACK IS CERTIFIED ON THE REAL EXTRACT, THE GATE
  REPRODUCES ITS BASELINE, AND THE INVERSION HAS COORDINATES.**
  Attack battery on tidy_visits (3 seeds): output-surface
  adversary 0.514/0.523/0.551 with the republish control at
  1.000 CAUGHT on the same arm - the output surface is now
  certified by an attack that can fail, on the data that
  matters. The weights arm read 0.366/0.479/0.619 - seed-
  dependent, one MARGINAL at 0.619 - recorded as-is; the
  network is never part of any release, and the duel footer
  quotes the real-extract worst now, not the fixture's 0.57.
  Fidelity re-run (run_oct12): gate direction 92/94 = 97.9%
  PASS, close 76/94 = 80.9% FAIL against 87.9% - the same
  numbers as the previous build, a stable baseline, no
  regression - shapes 7/8, surfaces 0/5 (ring-parked),
  INVERTED 1. quota_landing: 40 adopted pairs, 1 off by more
  than 0.15 - ENFORCEMENT LANDS AT EXTRACT SCALE, so the
  remaining close drift is not the adopted pairs. The console's
  own sign/close/INVERTED (22-28 of ~420 uniform pairs) is the
  challenger metric, NOT the gate - different denominator,
  both true, say which is quoted. And the one inversion,
  procedure_count <- mean_arterial_pressure_invasive, now has
  measured coordinates: conditional value corr src -0.306 gen
  +0.229 on a parent PRESENT ON 0.2% OF ROWS, child a copied
  set size, presence share reproduced exactly - a conditional-
  value flip inside a ~90-row subset, a shape no previous
  reproduction fixture had. That is the next fixture.

- **THE WIDTH BLINDNESS WAS A PARSE HOLE WEARING A THEORY, AND
  THE OUTPUT SURFACE IS NOW CERTIFIED AT FIXTURE WIDTH
  (2026-10-08).** The output-surface adversary read 0.500 at 77
  columns even on a VERBATIM REPUBLISH, recorded as "distance
  concentration" - but true concentration cannot put a
  distance-0 copy at 0.500, and that contradiction sat unread.
  The fact: float("nan") PARSES, so a missing cell sailed past
  every `is None` guard as nan, poisoned the distance, and
  `nan < best` being always-False tied every candidate at
  infinity - sparsity was the trigger, width only guarantees
  sparsity. Fixed in `_num` (nan/inf -> None, which also heals
  the scales profiler) plus one semantic change: absent on BOTH
  sides is agreement about absence, so a sparse row is distance
  ZERO from itself - the property the republish control stands
  on. Watched red on the old code. Measured at the 77-column
  extract-shaped fixture (400 patients, 3 seeds, shipping
  defaults): output-surface honest arm 0.495-0.556 with the
  republish control at 1.000 CAUGHT on the same arm;
  weights-leak reconstruction 0.544-0.574; worst mean 0.560,
  max 0.574 - PASS in the recorded bands, beside the
  blueprint's 0.52. THE CROWNING POSTURE, WRITTEN: output-only
  release is the release surface and it reads clean under an
  attack that can fail; the network itself is never part of any
  release and its adversary reads 0.544-0.574 with denoising;
  the duel footer now states exactly this, plus the one still
  true limitation - patient identity is carried (invented,
  0.90-0.96 person-level) but a patient's course over TIME is
  not modeled. Still owed: the same battery on the real extract
  (data machine), which is the run of record.

- **THE OFFLINE INSTALL HAD NEVER BEEN OFFLINE, AND THE EASY
  PATH IS A DOUBLE-CLICK (2026-10-06).** `pip install --no-index
  --find-links wheels -e .` fails on any modern venv - 3.12+
  ships no setuptools, and pip's isolated build fetches it from
  the NETWORK, so the kit's offline promise only ever worked on
  machines with quiet connectivity. Measured both ways on a
  fresh 3.14 venv: without a setuptools wheel the install
  errors, with one it completes - and the first version of that
  measurement lied twice (wheels fetched by the wrong
  interpreter's pip, and `import synthkit` finding the cwd
  package), which is why it was re-run from a neutral directory
  before being believed. setuptools and wheel now ride in the
  wheels folder. And the bundle ships START_SYNTHKIT.bat /
  RUN_CHECKS.bat - venv, offline install, bench, net, every
  step echoed and every exit PAUSED so an error is a readable
  window rather than a vanished one - because the easiest thing
  for a team to run is the thing they double-click.

- **A SUBSTITUTED STAMP IS AN ARCHIVE ARTIFACT, NOT A SOURCE
  FILE, AND STAGING IT POISONED THE TEAM REPO'S IDENTITY
  (2026-10-05).** The first enterprise push's verify clone
  failed smoke_buildid 2/26 - and the checks were RIGHT.
  BUILD_SHA.txt is export-subst, so the data machine's zipball
  arrived with OUR commit's sha already substituted in; the
  stage swept that file verbatim into the keck commit, and a
  repo whose tracked stamp is a bare 40-hex has archives that
  name a commit that is not in it - git archive substitutes
  PLACEHOLDERS, and there was none left to substitute. Staging
  now resets the stamp to the raw `$Format:%H$`; the fixture
  plants a substituted stamp and asserts the reset (watched
  red); and the full chain - substituted stamp in, placeholder
  staged, single-commit clone, smoke_buildid on 3.14 - was run
  here before asking the operator to push again. Three verify
  environments have now each found a real defect (zipball:
  empty git enumeration; 3.14: the argparse help crash; the
  enterprise clone: this) - the verify clone is not ceremony.

- **THE DATA MACHINE INSTALLED PYTHON 3.14, AND A LATENT CRASH
  IN A HELP STRING TOOK DOWN ALL EIGHT GATE CHECKS (2026-10-02).**
  smoke_fidelity failed 8/107 there and passed everywhere here -
  same libraries to the digit, so the interpreter was the only
  suspect left, and a local 3.14 venv reproduced all eight
  exactly. argparse %-formats help text, so the "93.9%" that
  `.format` put into m0_gate's bar-flag help was a ValueError
  WAITING FOR WHOEVER ASKED FOR HELP - latent on every Python
  since the flags shipped, because no check ever rendered -h.
  Python 3.14 validates help strings EAGERLY at add_argument, so
  the script died on EVERY invocation, and every check that
  crosses the m0_gate process boundary went red at once - eight
  failures, one defect. The literal percent is doubled now, -h
  is rendered through the process boundary as a check (watched
  red against the old script on 3.10 - the crash was always
  reachable, 3.14 only removed the need to ask), CI gained a
  3.14 lane, and the team bundle ships cp312 AND cp314 wheels,
  because the cp312-only set could not install on the Python
  the data machine actually has. The same lesson as pandas 3,
  one layer down: test against the INTERPRETER the data machine
  will install, not just the libraries.

- **A RUNBOOK THAT CANNOT REACH THE MACHINE IT RUNS ON IS A
  MAC-SIDE DECORATION (2026-10-02).** The copy-button RUNBOX.html
  lived on the Desktop because its baked-in real values are
  banned from tracked files - so the one page built for the data
  machine's terminal could never get there. The operator asked
  what use it was, correctly. RUNBOX.html is TRACKED now and
  carries only the placeholders; the page itself substitutes -
  two input boxes, values held in localStorage, every copy
  button yields the real command after typing the owner/name
  once per machine. The substitution logic was unit-run in node
  before shipping, the file is on the bundle's exclusion list
  beside RUNBOX.md, and the scan passes because the secret
  stays in the browser, never in the file.

- **THE BUNDLE'S GIT PATH WAS EMPTY ON THE ONE MACHINE THAT
  MATTERS, AND THE WALK FALLBACK REFUSES STRAYS (2026-10-02).**
  `git ls-files` on the data machine's zipball extract returns
  NOTHING - an empty source tree whose self-scan passes
  trivially, the worst kind of clean, caught by dry-running the
  flow as the data machine would run it. The fallback walks the
  extract (a zipball IS the tracked tree), was verified to
  produce a tree IDENTICAL to the git path, and refuses BY NAME
  any .csv outside the tracked data roots - on the machine that
  holds real extracts, sweeping an unexpected file into a
  bundle is the one failure that must be loud. The scanner also
  flagged its own rule-text (the exclusion list names the
  excluded files; the pattern names the banned words) once the
  script became tracked - exempted by FILE for the rules file,
  by LINE everywhere else, both stated. And RUNBOX was
  rewritten to delegate the enterprise filtering to this one
  script rather than fifteen manual commands, because the
  manual block's own scan line contained the banned words and
  would have failed the tracked-file scan - the rules and the
  procedure now live in one place. Two more from running the net
  AS the data machine would: the smoke proving the git-less path
  enumerated the tree with `git ls-files` itself - the check for
  git-lessness REQUIRED git, and crashed on the extract instead
  of failing - so it calls the script's own `_list_files()` and
  asserts the enumeration is non-empty; and that lister would
  raise FileNotFoundError on a machine with no git PROGRAM (the
  data machine, not merely no .git directory), so the subprocess
  call is wrapped and an absent binary reads as an empty list,
  which the walk fallback then serves.

- **THE BUNDLE SCANS ITSELF, AND ITS FIRST CATCH WAS ITS OWN
  TEST (2026-10-01).** `make_team_bundle.py` assembles the
  self-contained team bundle - the kit, the enterprise-filtered
  source (one exclusion list, with a stub CONVENTIONS.md
  because ten code comments reference the notebook by name),
  offline Windows wheels pulled cross-platform (--platform
  win_amd64, 65 MB covers numpy/pandas/sklearn/scipy), and an
  honest AI page: the measure route needs NO language model,
  Describe-from-English needs one of four backends, and the
  hospital cloud path is named before any local install. The
  bundle REFUSES to finish if its own deep scan hits - and the
  first thing the scan caught was the new smoke check whose
  assertions name the excluded files; the `noqa: bundle-scan`
  marker exempts exactly those lines, the same pattern the
  personal scan already uses. The dry-run export was also
  executed and verified here first: scan CLEAN, and the
  filtered tree's own net run - a push whose tree was never run
  is a push on hope.

- **THE FRONT DOOR AND THE GATE DIAL (2026-10-01, follow-ups
  from the demo).** The bench opened into step 01's FORM - a
  first-time kit user met input boxes before anyone told them
  what the tool is. A HOME landing page is the default station
  now: the one-sentence contract, both routes, the privacy
  posture in one line, and the ?-door reading model; exactly
  one section may start active, asserted, because two active
  sections render stacked (the appended-after-main fault
  wearing a default). And the gate's two PROPORTIONAL criteria
  are a dial: assess(fid, bars=...) with the bench fields and
  m0_gate flags riding the one implementation. The policy
  rules, stated everywhere they bind: a chosen bar is printed
  as YOUR bar beside the recorded default (a MET that does not
  name its contract is a number nobody can trust); bars clamp
  to [0.5, 1.0]; INVERTED is not a dial at any setting, and
  the full-marks criteria stay full marks.

- **THE TEAM DEMO LANDED, AND ITS ASKS SHIPPED THE SAME DAY
  (2026-10-01).** Three asks from the room. (1) A USER-SETTABLE
  k WITH A FLOOR: --k on the csv CLI and a fifth box on the
  Generate panel; raising k is a stated trade (the k=40 bound
  sits strictly inside the k=10 bound - asserted, because a
  dial that is read but not used is the --time-col failure);
  lowering below 10 is REFUSED with the same sentence at the
  bench and the CLI, before any training. The chosen k rides
  the blueprint's `bounds_are_k_anonymous`, which every deck
  page already reads - so "fewer than N patients" went dynamic
  for free. One bug en route: run_csv's return value was
  DISCARDED by the dispatcher, so the floor refusal exited 0 -
  a refused run reporting success. (2) TEMPLATES for people not
  in the know: three editable-English presets (a
  fill-in-the-blanks skeleton, a scheduling shape, a sensor
  shape); inserting them mid-list broke a positional
  presets[4] in a check, now selected by KIND. (3) The Keck
  push: a single clean filtered commit, never our history;
  exclusion list and both command blocks (Mac and data-machine,
  one command per line) in the operator's export file; the
  export runs 68 suites green because smoke_no_personal - which
  contains the scan patterns themselves - stays home.

- **THE QUOTA RUN MOVED ONE PAIR, THE FIXTURES LAND EXACT, AND
  THE RUN NOW MEASURES ITS OWN ENFORCEMENT (2026-09-30 night,
  build 0991b3e).** close 73/94 -> 74/94 while the rare-wide
  fixture lands all eight pairs within 0.04 - and through the
  FULL CLI disk path within 0.06 (12 quotas adopted, close
  135/136). The build-to-build pair wiggles on the extract
  (ondansetron 0.54 -> 0.36, docusate 0.08 -> 0.14, mixed
  directions) read as single-seed draw variance, which this
  file already forbids chasing. The honest open question -
  does adoption or enforcement differ at extract scale - is
  one no fixture here has answered, so the run answers it
  itself: `quota_landing` in fidelity.json measures every
  adopted token against its top conditioner on BOTH tables and
  prints the verdict. If those land while close does not move,
  the drift pairs are not the adopted pairs and the ADOPTION
  gates are the question; if they do not land, ENFORCEMENT
  breaks at scale. Same move as the inverted diagnostic, same
  reason: a fifth fixture would be a fifth guess.
- **AND THE CONDITIONER CAP IS GONE.** Top-6-by-gain could
  exclude exactly the partners the gate measures (a route on
  the extract is implied by a dozen drugs) while the rest-fill
  excluded their rows. All qualifying conditioners now; the
  budget bounds the total on its own. The cap-harm fixture
  built to prove it could NOT reproduce harm - recorded as a
  negative, the uncapping ships on coverage grounds, not on a
  measured rescue.

- **THE 23-SHAPE SWEEP WAS RE-RUN AFTER THE FIDELITY TIERS
  LANDED (2026-09-30 evening): 23 of 23 clean.** The kit's
  "handles your data" claim was measured before the reader fix,
  the size/arithmetic/near-identity tiers, the independent
  token draw and the conditional quotas existed - a claim like
  that does not stay true by argument. Re-measured on the
  current engine: every shape clean, every new tier inert where
  its structure is absent, which is the property that makes the
  fidelity work GENERALIZE - each tier detects from the source
  and enforces at generation, so on a retail table "a drug
  implies its route" is "a product implies its category" and on
  data with none of it the tiers do nothing. Known residual
  sensitivities, stated: the adoption thresholds were swept on
  extract-shaped fixtures; conditioners come from EARLIER file
  columns only; negative token associations are not modeled.

- **THE IMPLICATION-TIER RUN MOVED ONE PAIR, AND THE DRIFT LIST
  NAMED WHY (2026-09-30 build bd0f7ba).** close 72/94 -> 73/94:
  the near-deterministic pairs cleared (sodium-chloride <->
  Flush LEFT the list; n~n 0.58 -> 0.63) and eleven MODERATE
  pairs on RARE tokens remained - docusate <-> Rectal 0.4 ->
  0.08, tacrolimus <-> NG-tube 0.39 -> 0.1 - all BELOW the
  P(b|a) >= 0.9 bar, where hard forcing would overshoot a 0.6
  conditional to 1.0.
- **THE CLASS REPRODUCES ONLY WHEN THE TOKENS ARE RARE AND THE
  FRAME IS WIDE.** Moderate conditionals on COMMON tokens
  survive the draw even at width (measured, worst drop ~0.1);
  the same conditionals on tokens at 1.5-6% share fade to half
  strength at width (source +0.37/+0.72/+0.62 -> +0.17/+0.35/
  +0.41) - a rare token's thin activation cannot place it among
  its partner's rows by itself. Two negatives and a positive,
  all needed before the mechanism had its true name: not
  "co-occurrence", but RARE-token co-occurrence at width.
- **CONDITIONAL QUOTAS SUBSUME THE IMPLICATION TIER.** Each
  adopted token b gets a quota SPLIT: round(q * |trigger|) of
  its exactly-cj picks inside its conditioner's rows (best by
  b's own score), the rest outside - any q lands by
  construction and q ~ 1 IS the old forcing. Adoption: q
  exceeding base by >= 0.15 and 1.5x, base <= 0.5 (the Oral
  lesson retained), conditioners from EARLIER plan columns only
  so a trigger reads a column that will not move again.
- **AND ONE CONDITIONER PER TOKEN UNDER-COVERS, measured before
  it shipped.** One route is preferred by several drugs: with a
  single best conditioner, IV-Push-conditioned-on-ondansetron
  leaves its acetaminophen and metoclopramide pairs exactly as
  faded as before - the single-conditioner cut landed only the
  pairs whose token happened to pick THEM (3 of 8). Up to six
  conditioners per token now, strongest first, overlaps counted
  once through the running pick state, budget capped at cj. On
  the rare-wide fixture ALL EIGHT pairs land on source (worst
  drop 0.04, both seeds).

- **THE PRESSURE FADE REPRODUCED ONLY AT WIDTH, AND THE SWEEP
  EXONERATED THE SAMPLER (2026-09-30).** Dense twins do NOT
  reproduce it (the engine OVER-correlates them, 0.989 vs
  0.967); sparse twins at 8 columns do not either (sign kept,
  mild fade). At ~25 columns / ~150 encoded dims - the extract's
  shape - the sparse second-device twin faded 0.975 -> 0.77-0.88,
  seed-dependent, which is the real run's `systolic <-
  MAP_cuff_bmdi 0.5 -> 0.13` class. A WIDER bottleneck made it
  WORSE (0.67-0.72 at b=64), which acquitted the encoder and
  indicted the sampler - and then the sampler sweep acquitted
  the sampler too: GMM at x1/x3/x6 components and a KDE sampler
  at bw 0.3/0.5/0.7 all land in the same 0.70-0.87 band, with
  the fade following the TRAINING seed across every arm. The
  attenuation lives in the network's handling of the sparse
  channel, and no density estimate downstream can restore it.
- **THE CURE IS THE IDENTITY TIER, ONE NOTCH RELAXED.** A
  second-device twin is an identity plus device noise: rho 0.97,
  residual ~3% of spread. `_near_identities` detects pairs at
  |pearson| >= 0.95 on >= 200 both-present rows with residual sd
  under 0.35 of the child's spread, child = the SPARSER column,
  and generation computes the twin from its generated sibling
  PLUS noise at the measured residual sd, only where both are
  present. Wide fixture: 0.973/0.977 against source 0.975 on
  both seeds - exact by construction - with close, sign,
  INVERTED and the reconstruction guard unmoved, and the twin's
  own spread and presence rate held (the shift/scale rule).
- **ONE SET CONFLATING TWO ROLES SILENTLY DISABLED THE TIER.**
  The exact tier's parents were seeded into the same exclusion
  set that bans near-CHILDREN - correct, re-deriving an exact
  parent would break the arithmetic just enforced - but the
  same set also gated near-PARENTS, so map_cuff (exact parent
  AND the twin's sibling) was banned from the one role the tier
  exists for, and detection returned empty while every gate
  passed by hand. Two exclusions, two sets, and the check
  asserts the exact parent IS the near parent.
- **PREDICTION FOR THE NEXT REAL RUN (before it runs,
  superseding the previous one):** with the reader fix, the four
  coverage misses (the set columns, whose folded source read
  19-84% present against a generated 100%) should clear, the
  set-pair drift list re-baselines, and the metoclopramide
  inversion class goes. With the near tier, `systolic <-
  MAP_cuff_bmdi` and its family land at source strength and the
  pressure surfaces (0/6) get their first honest chance - the
  near tier is expected to convert at least the surfaces whose
  pair members are twins; surfaces on genuinely joint effects
  are NOT claimed. `procedure_count <- MAP_invasive` did not
  reproduce on FOUR constructions - narrow sparse (sign kept
  -0.21), wide weak, and wide strong across three seeds
  (-0.18/-0.24/-0.35 against source -0.225) - so no fixture
  claim is made. The plausible account is that the pair was
  itself DOWNSTREAM of the reader fault: procedure_count is
  len(procedures), procedures is the 80.7%-empty column whose
  empties the folded reader erased from both the vocabulary and
  the gate's reference numbers, so both sides of that
  correlation were measured on a distorted conditional subset.
  If it survives the re-run on the fixed reader, it earns a
  peek diagnostic on the data machine, not a fifth fixture.

- **A NET RUN CONCURRENT WITH EDITS VALIDATES A MIXED STATE.**
  The 2020-check ALL GREEN finished while the near tier was
  being written; suites read files at their own start, so that
  green covers neither the before nor the after. The final
  state got its own clean net before the push.

- **SMOTE WAS CONSIDERED AND REJECTED, WITH REASONS THAT ARE
  MEASUREMENTS.** Asked directly whether SMOTE-style
  interpolation plus k-means outlier scoring should replace or
  augment the autoencoder. No: (1) a SMOTE row is a convex
  combination of TWO REAL PEOPLE's records - nearer to copying
  than sampling a fitted latent, so the nn-ratio tripwire and
  the membership adversary both get worse by construction, on
  the surface where this engine already carries its FAIL; (2)
  interpolation is undefined for what this data is made of -
  categories, dates, sets ("Oral;IV" has no midpoint) - which is
  why the encode step exists at all; (3) the measured failures
  are not density-estimation failures: co-occurrence was a DRAW
  defect (fixed above), surfaces are the colinear pressure
  family. (4) The k-means-distance outlier idea is already
  implemented in a stronger form: the k-aware blur counts
  DISTINCT PATIENTS within a radius in feature space - distance
  to a centroid cannot tell four near-identical patients far
  from a centroid apart from one genuinely isolated person, and
  the radius count is the k question asked directly. Recorded so
  the next "should we SMOTE it" starts from these numbers.

- **THE HOLDOUT EXISTED AND NOTHING ASKED IT THE OVERFITTING
  QUESTION (2026-09-29).** 15% of PATIENTS have been held out
  since the engine was written, and the only use of them was
  `nn_ratio` - a DISTANCE ratio between the GENERATED file and
  the training rows, which never puts a held-out record through
  the network at all. So "is it memorizing rather than learning"
  had a split ready for it and no measurement on it: a
  capability present and wired to nothing, which is this file's
  own recurring fault. `reconstruction_gap` encodes and decodes
  both groups and compares squared error. Asked by the operator,
  not found by a check.
- **AND IT NEEDED A POSITIVE CONTROL, BECAUSE A RATIO NEAR 1.0
  IS ALSO WHAT A BROKEN MEASURE RETURNS.** The first control -
  a wide bottleneck with denoising off, on STRUCTURED data -
  read 1.296 against the default's 1.155, which separates
  nothing: the network generalizes on structured data whatever
  width you give it. The control that works removes the thing
  being generalized TO - a frame of iid noise, where any low
  training error IS memorization.
      structured, shipping default           1.155
      structured, denoising off              1.444
      PURE NOISE, wide bottleneck, no denoise  9.739
      PURE NOISE, shipping default            1.799
  The shipping configuration cannot memorize even when
  memorizing is the only way to score well. The 1.155 against
  1.444 is also an INDEPENDENT confirmation of the denoising
  result, which was previously known only through the
  membership AUC (0.776 -> 0.569).
- **AND THE ARTIFACTS WERE WRITTEN FROM THE TRAINING SPLIT.**
  `write_run_artifacts(tr, ...)` discovered the relationships on
  85% of the data while the Dashboard's own header counted
  100%, so the gate and the page it sits on used different
  denominators. The relationships belong to the SOURCE.

- **AND THE NUMBER NOBODY DREW WAS READ AS A NUMBER NOBODY
  MEASURED.** The reconstruction gap lived in the terminal and
  fidelity.json, and the operator asked why nothing SHOWS how
  training and held-out error compare - correctly, because a
  measurement that never reaches a page might as well not
  exist for the person deciding. The neural page now draws it:
  three headline cards (train error, holdout error, ratio with
  the 9.7-vs-1.8 control stated) and per-column paired bars,
  gray training beside gold held-out, largest first, with any
  column whose holdout error runs 1.5x past its training error
  flagged - that is where memorization would show first.

- **AND THE DASHBOARD COULD NOT REACH THE EXPLANATIONS AT ALL.**
  It is served into a SANDBOXED IFRAME, where the bench's panel
  is unreachable by construction, and the same HTML is a file
  people send to each other - so the one page an operator
  actually reads was the one page with no doors on it. The
  panel travels INSIDE the deck now, fed from the same
  `synthkit.explain` registry: one dictionary, two renderers,
  because the hosts genuinely differ while the words must not.
  Eleven new entries for the words the gate tables actually
  use - drift, faded, overshot, sign lost, rank correlation,
  coverage, set token shares, suppressed bins, skill - since a
  reader meeting "INVERTED" or "-0.55 faded" in a table had no
  way in.

- **THE AUTO-LINKER IS NOT ENOUGH WHERE THE MACHINERY IS THE
  SUBJECT.** Explanations are marked on the FIRST occurrence of
  a phrase per section, so a reader who wants to know how the
  engine works must first guess that the words in front of them
  are clickable. On Generate and Verdict the doors are now
  listed openly as a labelled strip of `?` buttons, and every
  slug on a strip is asserted to resolve - the same rule
  `[[slug]]` already had, because a door that opens on nothing
  is worse than no door.

- **A RAIL IS A CLAIM ABOUT WHICH ENGINE THE TOOL LEADS WITH.**
  The autoencoder sat under "see the fidelity" beside the
  Dashboard while the rules engine held step 02 of the measure
  route - so the path an operator is walked down generated with
  the engine the duel exists to show is beaten. The
  autoencoder is step 02 now (it already measures the source's
  patterns before generating, which is what the slot means);
  the rules engine, Learn and the Duel are grouped as "the
  other engine, for comparison". Asserted on the TILE and on
  `gate.STATIONS` together, because guidance that says "go to
  step 02" over a rail that numbers it differently is the
  two-halves-disagree fault wearing navigation.
- **AND FAIL-FIRST HIT THE COUPLED-CHANGE TRAP AGAIN.**
  Reverting gui.py AND gate.py together made `smoke_gui` die on
  a KeyError at the first new check, proving the files are
  coupled rather than that the checks can fail. Reverting only
  the BEHAVIOR - gui.py, keeping the new station table - showed
  all five predicates red.

- **EIGHT PUBLISHED SURFACES IS A PUBLISHING CHOICE, AND THE
  DASHBOARD LET IT READ AS WHAT THE DATA HOLDS.** The gate's
  `interaction surfaces 0/8` counts the surfaces the CONTRACT
  published; `_mine_interactions` finds joint effects nobody
  published, by screening parents on MODEL IMPORTANCE rather
  than correlation - and it had lived on the Neural and Duel
  pages only, never on the Dashboard, which is where anyone
  actually looks. It is on the Dashboard now, at 24 rather than
  8, with a per-row "published?" column and a TOTAL of how many
  have no surface. The unpublished ones are the interesting
  half: SHAP had already ranked the two strongest jointly-acting
  pairs on the real extract and NEITHER had a published surface.
  A joint effect the contract cannot express is a limitation to
  NAME, and it cannot be named while it is not measured.

- **THE BLUEPRINT'S ZERO IS STRUCTURAL, NOT A TUNING MISS.** The
  engine publishes interaction surfaces only where discovery
  finds and keeps them, applies them through curves - and the
  product-gain metric shows the OUTPUT carries none of it on
  this sample. The fork this forces: mine interactions the
  challenger's way and PUBLISH them as k-screened surfaces the
  blueprint samples from (latent-informed surfaces - keeps the
  aggregates-only privacy posture), or run two engines behind
  two stated postures. Decide on the attack-at-width result,
  not before.

- **AND ITS PRIVACY MODEL IS DIFFERENT IN KIND, SAID EVERY TIME:
  IT TRAINS ON RECORDS.** The k-screen narrows what the network
  sees; it does not prevent memorization, which is record-level.
  The blueprint's membership result (0.52 where a cheat FAILS at
  1.00) belongs to the blueprint alone. Until the SAME attack
  battery passes on this path, its output is a MEASUREMENT of
  headroom - a ruler beside the generator, never a release. The
  decision it forces is now quantified instead of argued: the
  interactions the operator sees inverted in SHAP are
  representable at this privacy posture; the open question is
  the price of getting them at ours.

- **THE ATTRIBUTION FLIP IS SURGERY-PROVEN AND SHARPER THAN ITS
  FIRST CHARACTERIZATION (2026-09-23).** Eight instrumented seeds
  on the triangle: cross-edges live at 0.2-0.9% of their claim's
  top importance and flicker with the seed; both-absent seeds sit
  STABLE at 69-71/29-31 against a 77/23 source, edge-present
  seeds lock at 86-90, and the one flip (39/61) is the
  ASYMMETRIC case - the reverse edge drug<-proc included alone.
  Causality by blueprint surgery: removing that ONE parent
  restored 70/30, removing the whole relationship gave the same
  70/30 - while dialing the edge's strength to ZERO changed
  nothing (39/61 -> 40/60). The harm is the edge's PRESENCE in
  the graph - trim decisions, cycle ordering, coupling - not its
  curve.
- **CUTTING THE BORDERLINE PARENTS WAS MEASURED AND REVERTED,
  WITH THE NUMBERS.** A 2%-of-top relative floor stabilized the
  triangle completely - 0 flips in 8 seeds, the identical edge
  set on every seed - and cost the pressure fixture its surfaces
  (0/5 seeds failing -> 3/5), with drop lists showing
  hemoglobin-family parents cut at 1-1.7% relative. The bench
  read 11/13 under the floor and the SAME 11/13 after the
  revert, identical misses both ways - that delta against the
  recorded 12/13 is pre-existing drift on HEAD (het is now
  unexplained at discovery on the default fixture config), NOT
  the floor's doing, and the misattribution was caught only by
  re-running the bench after the revert. Locally right,
  globally worse - the greedy-cut lesson on the discovery side.
  What shipped instead is the FLAG: claims name their
  `attribution_fragile_parents` (kept, with share-of-top), the
  fit narration says "N borderline parent(s) flagged -
  attribution involving them is seed-fragile", and a future cut
  must hold pressure at 0/5 seeds failing and the bench at its
  same-day baseline, not just win the triangle.
- **THE TOKEN-WEIGHT SOLVE UNDERSHOOTS AT SMALL-COHORT /
  HUGE-VOCABULARY SHAPE, BY A CONSTANT (OPEN, 2026-09-23).** A
  300-patient sample of the extract-shaped fixture (active_drugs:
  1,626 tokens, ~24 above the cap, most of the vocabulary near
  the k floor) generated EVERY published token at almost exactly
  one third of its source share - twelve deltas between -0.061
  and -0.069 on sources near 0.095, `set token shares 0/12`. A
  constant multiplier across tokens is a denominator or a solve,
  not twelve faults; the damped solve was verified at the
  800-patient extract shape and never at this one - the
  measured-only-at-one-scale lesson, again. NOT fixed; reproduce
  at this shape before touching the solve, and expect the LIVE
  demo's 300-patient rehearsal to show it on the real
  active_drugs.
- **THE LIVE-FIT DEMO MATH: SET EXPANSION, NOT ROWS, OWNS THE
  SEARCH BUDGET.** On the 300-patient fixture sample, 85 file
  columns became 143 searched after token expansion, and the fit
  blew past 10 minutes; excluding three of four set columns cut
  the search to 104 and the fit landed. The August "274s with
  --lags" number predates set expansion and must not be used for
  estimates. Live-demo configuration: whole-patient `synthkit
  sample` plus `--exclude` on the heavy set columns, REHEARSED
  once for minutes and gate verdict before any room sees it -
  and timing measured on a contended machine is not timing.

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

- **THE RELATIONSHIP DIAL IS PER-EDGE, AND ITS FIRST CUT KILLED A
  NEIGHBOR.** `--dial "CHILD<-PARENT.strength=0.5"` reaches one
  DIRECTED edge. The first implementation set the record-level
  target_strength, which scales EVERY parent at once - measured:
  dialing span<-proc to 0 left the named pair at 0.907 (alive via
  the reverse record) while collapsing the innocent span~drug to
  0.007. The shift/scale lesson, again: each dial's check must
  assert the NEIGHBORING property. Per-edge routing
  (dials.edge_strength -> target_edge_strength, applied at all
  three generation sites) measures correctly: both directions at
  0 remove the pair (0.919 -> 0.056) while the neighbor holds
  (0.555), strength=1.0 is bit-identical to an undialed run, and
  a single-edge 0 with a live reverse edge reports as
  INFORMATIONAL naming the reverse edge - render was showing
  hit=None as MISS under a note explaining the number, and a
  verdict must not contradict the note beneath it. Unknown edges
  error with did-you-mean; `synthkit dials` lists every edge.

## Numbers before conclusions
- **THE CAPACITY AXIS WAS SWEPT AND IT IS NEARLY FLAT FOR `close`,
  WITH ONE SEED-ROBUST GAIN ON THE ABSOLUTE STOP (2026-11-10).** The
  proposal was to train a generalising and a MEMORISING autoencoder
  and interpolate between their feature spaces to find the best
  fidelity that still generalises. Two things had to be measured
  before any of that was worth building: whether fidelity rises with
  capacity at all, and whether the privacy measures go first.
  Swept on an extract-shaped fixture (7,811 rows, 220 patients,
  ~1,375 uniform pairs), three seeds per width, scored on the
  challenger's own pair metric AND the nn tripwire AND the
  train/holdout reconstruction ratio together:
      width   close, three seeds        mean      inversions
        32    97.75 / 97.09 / 97.53    97.46%     8, 7, 6
       128    98.62 / 98.03 / 97.24    97.96%     3, 3, 5
       256    96.58 / 98.47 / 98.03    97.69%     6, 4, 5
  **The within-width seed spread is 0.66 to 1.89 points and the
  between-width differences are about 0.5** - so on `close` the axis
  above the default is flat, and the single-seed readings that
  looked like a +10-pair peak at 128 and a -29-pair collapse at 256
  were both draws. This project's oldest rule caught both.
- **WHAT IS SEED-ROBUST IS THE INVERSION COUNT: 6 to 8 at width 32
  against 3 to 5 at 128, with NON-OVERLAPPING ranges.** That is the
  one verdict this project calls an absolute stop, and it is the
  only thing the capacity axis moved. Quoted on the CHALLENGER's
  uniform pair metric over ~1,375 pairs, which is a different
  denominator and definition from the gate's own inverted count -
  say which one is being quoted.
- **AND THE DEFAULT SITS EXACTLY AT THE KNEE, BY A CAP NOBODY HAS
  REVISITED.** `self.b = int(max(8, min(32, self.d // 6)))` - a HARD
  cap at 32 whatever the frame. At the fixture's d=547 the rule
  wants 91 and gets 32; at the real extract's d=1,601 it wants 266
  and gets 32, eight times lower. So the engine has been running at
  the knee of this curve on every real run, which is why the axis
  looks flat: the cheap part was already being taken.
- **AND RECONSTRUCTION ERROR CANNOT FIND THE SPOT, WHICH REFUTES THE
  SELECTION CRITERION THE PROPOSAL RESTED ON.** Training error falls
  **4x** from width 32 to 256 (0.0438 to 0.0119) across a range
  where `close` does not move outside its own seed noise. A criterion
  that improves monotonically while the graded surface stays flat
  cannot locate an optimum on that surface. The selection has to be
  scored on the GATE, which this file already says about optimising
  `within_lag1` rather than the two measurements it is derived from.
- **AND THE PRIVACY MEASURES NEVER DEGRADED IN THE RANGE, which is
  the half of the proposal that would have decided it.** nn-ratio
  0.990 to 1.016 across every arm, against **0.00** for a verbatim
  republish; train/holdout 1.076 to 1.113, against **9.7** for the
  recorded pure-noise positive control. So nothing in this range is
  near the memorising regime at all - which means an interpolation
  between a generalising and a memorising model has almost no
  fidelity to arbitrate over here, because the generalising end
  already has it.
- **AND THE OUTLIER SCREEN THE PROPOSAL NEEDED WAS TESTED AND IS A
  MEDIOCRE PROXY FOR A MEASURE THIS ENGINE ALREADY HAS.** The idea
  was to use per-record reconstruction error to find the records
  whose patterns fewer than k patients support, keep the rest, and
  reproduce them faithfully - the k-ceiling half of which is sound
  and is this project's own posture. Measured against
  `support_of_rows`, which counts DISTINCT TRAINING PATIENTS within
  a radius in FEATURE space:
      width    sub-k share   AUROC(error finds sub-k)   err ratio
         8        0.202              0.781                1.24
        32        0.202              0.738                1.21
       256        0.202              0.791                1.27
  **My own predicted flaw was REFUTED** - I expected the AUROC to
  collapse as capacity rose, because a memorising model rebuilds a
  unique record well, and it does not: the error falls everywhere
  and the RATIO survives. What the measurement does say is that the
  proxy is moderate, not sharp - an error ratio of 1.2 and an AUROC
  near 0.78 would keep a real share of sub-k records and discard a
  real share of well-supported ones - and that the exact quantity is
  already computed directly and already wired in as the k-aware
  blur. There is no reason to screen on a 0.78 proxy for something
  measured at 1.0 by definition.
- **WHAT THE SWEEP DOES LEAVE: raising the cap for the INVERSION
  gain, on a real arm.** It is the one thing three seeds agreed on,
  it costs nothing measurable in privacy, and it is a capacity
  change rather than a token-placement rearrangement - so it is not
  subject to the near zero-sum this file records for the last four
  token tiers. The cost is a longer fit and more memory at d=1,601,
  which is the shape that once died allocating 1.5 GiB.
- **HOW MUCH OF `close` IS SAMPLING NOISE IS COMPUTABLE FROM THE
  SOURCE ALONE, AND NOBODY HAD ASKED (2026-11-09,
  `scripts/close_ceiling.py`).** `close` has sat 1 to 8 points under
  its 87.9% bar on the real extract for months, and this file already
  records that the bar's provenance is an old absolute target
  restated as a proportion rather than one set from a measured
  distribution. Before spending more on the engine it is worth
  knowing what the number could be AT BEST - and that needs no fit
  and no generate. Split the cohort into two disjoint halves BY
  PATIENT, put one in as the source and the other as the generated
  table, and run the gate's OWN `_pair_fidelity` over them. Nothing
  has been generated, so every pair that fails is failing on
  sampling noise: two halves of the same real data disagreeing about
  a correlation by more than 0.2. A pair that cannot agree with
  ITSELF across a split cannot be expected to agree between a source
  and any synthetic file.
- **IT IS A LOWER BOUND, AND WHICH VERDICT THAT PERMITS IS STATED
  RATHER THAN LEFT TO A READER.** Each half holds half the patients,
  so it carries MORE noise than the full-cohort comparison the gate
  actually makes. A split-half reading BELOW the bar is therefore
  decisive - the bar is unreachable before the engine is involved -
  while a reading above it is INCONCLUSIVE and says so in those
  words. This project has twice drawn a conclusion from a number
  whose denominator or scale it had not checked, and a lower bound
  read as a measurement would be the third.
- **THE SPLIT IS BY PATIENT, which is the one way this instrument
  could lie in the flattering direction.** Splitting ROWS puts the
  same person on both sides, so the agreement measured is the same
  records twice and the ceiling reads high. The mutation that splits
  by row turns two checks red.
- **AND THE VERDICT IS A PURE FUNCTION ON PLANTED NUMBERS, because
  no fixture here reaches the UNREACHABLE branch.** Discovery
  publishes only the pairs it FINDS, so a cohort weak enough to
  disagree across halves is also one where few pairs reach the
  surface at all - measured over six shapes, the lowest split-half
  reading reached was 93.3%. A branch no fixture can reach is the
  dead-branch fault this project already found in a linkage verdict,
  so the decision is asserted where it is MADE, and NO-PAIRS is its
  own answer rather than being folded into either verdict - an
  absent measurement is not a clean one.
- **AND THE BLUEPRINT IS REQUIRED RATHER THAN OPTIONAL.**
  `_pair_fidelity` iterates the blueprint's own relationships, so
  without one there are ZERO pairs and the tool would report a
  ceiling over nothing - the worst kind of clean. It also has to be
  the surface the GATE measures, which this file records costing a
  week when two runs counted pairs over different token
  vocabularies and the difference was read as a regression.
- **AND THE 2% BAR ALONE CLAIMED A ONE-PARENT NEAR-IDENTITY AND
  DESTROYED THE COLUMN THE DIFFERENCE IS.** Four checks that had
  been reproducing a constraint violation for months went red the
  moment the tier shipped, and they were right. A fixture here
  has `visit_end = visit_start + span` with span 0 on 88% of rows
  and 1-20 on the rest: against a start spread of 430 days the
  residual is 3.6, which is **0.8% of the spread** and clears a
  2% bar - so `visit_end == visit_start + 0*anything + c`
  qualified, generation COMPUTED the end date as the start plus a
  constant, and the length of stay vanished. There was no
  violation left for the constraint-repair checks to reproduce,
  which is how a fix that destroys information can read as a fix.
- **THE PAIR MUST EARN ITS SECOND PARENT, AND THE SEPARATION IS
  ORDERS OF MAGNITUDE RATHER THAN A TUNING QUESTION.** Residual
  of the pair over residual of the best SINGLE parent:
      end ~ start + age   (age is noise)        1.000
      end ~ start + los   (the real identity)   5.6e-14
      map ~ systolic + diastolic                1.9e-15
  A pair that does not beat its best single parent by a factor of
  four is a one-parent relationship wearing two, and the record
  carries `best_single_residual_sd` so a reader can see the
  second parent earned its place rather than taking the word
  `pair` on trust.
- **AND A WHOLE-NUMBER CHILD HAS TO BE ROUNDED OR THE IDENTITY IS
  FALSE ON EVERY ROW WHILE A TOLERANCE SAYS IT HOLDS.**
  `age_at_visit` is the visit year minus the year of birth and
  both are integers, so the age is one too - and least squares
  returns weights of 0.9999999999 with a tiny intercept, so the
  computed value came out at 54.99999999. The near-deterministic
  measure, which uses a tolerance, reported 3 of 3 identities
  HELD; exact comparison reported 0%. Two measurements of one
  thing disagreeing is this project's alarm, and here the
  tolerant one was the flattering one.
- **AND THREE CHECKS WERE REWRITTEN RATHER THAN DELETED.** The
  `age_at_visit` fixture existed to BREAK the identity - it held
  on 21.4% of generated rows while every other number in the
  report read fine - and it now holds on **100.0%**. The checks
  pin the enforcement instead, and the tightness MEASURE is
  tested separately on a frame where the break is PLANTED rather
  than on whatever the engine happens to produce. That is the
  stronger arrangement: the measure exists to catch breakage from
  any cause, including a future change that stops enforcing.
- **AND MY OWN BAR CHECK WAS WRONG THREE TIMES RUNNING.** First
  it loosened the residual bar on the full frame and expected the
  strong relationship to be claimed - first-claim-wins had
  already consumed its parent. Then it loosened the bar on a
  single-parent relationship, which `PAIR_GAIN` refuses first.
  Then it used `3a + 2b + 1.2n`, whose pair-over-best-single
  reads 0.52 and fails `PAIR_GAIN` too. Only a fixture where the
  second parent DOMINATES (ratio 0.15) leaves the residual bar as
  the one thing refusing it. Each red was a premise of mine, not
  a defect, and each took a measurement to see.

- **THE RULES ENGINE HAD NEVER ENFORCED AN ARITHMETIC IDENTITY -
  ONLY MEASURED ONE - AND IT IS THE ENGINE THAT SHIPS
  (2026-11-08, `synthkit/identities.py`).** `mean_arterial_
  pressure` IS `(systolic + 2*diastolic)/3`; a length of stay IS
  the end date minus the start. A generator that draws the child
  as an ordinary number makes a SECOND independent draw of a
  quantity the data already determines, so the identity holds by
  chance. The neural path has detected and enforced these since
  2026-09-29; the rules path had `tightness`, which MEASURES the
  breakage and reports it, and acted on none of it. Measured on a
  length-of-stay fixture: `end - start == los` held on 100% of
  source rows and **11.9%** of generated ones. A capability
  present in the half that does not ship and absent from the half
  that does is this project's most repeated fault, and the fix it
  prescribes is one module both halves CALL - applied here, with
  the neural engine's own inline loop replaced by the call and
  its 144 checks green.
- **THE RESULT, paired on one fixture, one file apart:**
      measure                        source   before    after
      end - start == los             1.0000   0.1185   1.0000
      interaction surfaces              -      1/2      2/2
      shapes tracked                    -      4/9      5/9
      los mean (sd)                4.45 (2.28)  4.45 (2.36)  4.45 (2.36)
  Two gate criteria improve, one of them to a PASS, with `los`'s
  own marginal untouched and direction / close / inverted /
  coverage unchanged. The pressure-family interaction surfaces
  sit exactly on this, which the neural engine's own entry
  predicted.
- **WHICH MEMBER OF THE IDENTITY IS COMPUTED DECIDES WHETHER THE
  COLUMN IS RIGHT, AND IT COST TWO MEASUREMENTS TO LEARN.** Any
  rearrangement of an exact identity is the same constraint, so
  the choice looks arbitrary. It is not. Protecting EVERY date
  made `los` the child of `end - start`: the identity then held
  on 100% of rows while `los` came out with **mean 5.49 against a
  source 4.45 and sd 4.92 against 2.28** - the arithmetic was
  true and the column was wrong, because the generated dates are
  not the source's PAIRED dates so their difference is not
  distributed like a length of stay. Protecting only the TIME
  AXIS makes the END DATE the child instead, and then all three
  agree: `los` keeps its published marginal, the end date is
  computed from the ordered start plus it.
- **AND THE TIME AXIS MUST STILL BE PROTECTED, because two
  repairs that each own one of the same columns undo each
  other.** The first detection claimed
  `visit_start_date == visit_end_date - los`, which would
  overwrite the dates the within-patient reordering had just put
  in visit order - and whichever ran last would win. This file
  already records that shape: `a <= b` and `b <= a` both
  enforced, 258 rows swapped one way and 10,328 the other, the
  first repair reported broken on 99.8% of rows.
- **THE LOAD-BEARING CHECK IS THE REFUSAL.** A detector that
  called every strong relationship an identity would replace its
  curve and its noise with a straight line, which is a worse
  output than a broken identity. The bar is a residual under 2%
  of the child's own spread - a rule the source nearly satisfies
  is not a rule - and the check proves the bar does the
  separating rather than the fixture: on a frame carrying only
  the strong relationship, nothing is claimed at 2% and `strong`
  IS claimed at 90%. Loosening the bar turns FIVE checks red.
- **AND THAT CHECK'S FIRST VERSION HAD A WRONG PREMISE.** It
  loosened the bar on the FULL frame and expected `strong` to be
  claimed - but first-claim-wins had already consumed `systolic`,
  so `strong` had no parent pair left and the loosened bar
  changed nothing about it. Red against correct code. The frame
  carrying only the strong relationship isolates the bar.

- **A PATIENT'S DATES GO IN VISIT ORDER NOW, AND MY FIRST READING
  OF WHY IT MATTERED WAS WRONG (2026-11-08).** `visit_number` is
  emitted as 1..n in ROW order, which ASSERTS a sequence; a date
  column is drawn from its marginal, which does not. Measured on
  a two-date fixture, the generated start dates came out in row
  order on **0.2% of patients**. Two columns in one file
  disagreeing about the order of a patient's visits is this
  project's own alarm.
- **THE GATE WAS NEVER THE VICTIM, AND I NEARLY RECORDED THAT IT
  WAS.** The first measurement read the output's within-patient
  course by DATE and found 0.4026 against a source 0.6456, where
  reading it by ROW order gives 0.6647 - and I was one step from
  writing up that the gate had been understating the engine's
  dynamics and that `dynamics tracked 17/87` was substantially a
  measurement artifact. It is not: `pipeline.compare` already
  sets `gt = "visit_number" if "visit_number" in g.columns`, so
  it has been reading the output in the right order all along.
  **The whole gate is IDENTICAL on both arms of this change -
  seven criteria unmoved, including `dynamics tracked 1/4`.**
  What saved it was running the gate rather than reasoning about
  it, which is the only reason the claim did not ship.
- **WHAT IT FIXES IS THE FILE, for whoever reads it next** - and
  that includes the lagged-parent path, which applies its curve
  in ROW order, so without this the two readings of one file
  disagree about whether a lagged relationship is there at all.
  Start dates in row order: **0.2% -> 100%**.
- **THE VALUES MOVE, NOT THE ROWS, and that is the whole design.**
  Reordering rows would preserve every within-row relationship
  exactly - and the persistence lives in the ROW layout, so it
  would destroy the thing this repairs. Sorting the date VALUES
  keeps the layout.
- **AND ONE PERMUTATION SERVES EVERY DATE COLUMN, which is what
  makes a date-to-date identity survive.** The real extract
  declares `visit_start_date <-> visit_end_date` at skill 0.9999;
  permuting the start dates while the end dates stayed would
  destroy it. Measured: start~end **+0.9988 in BOTH arms**,
  bit-identical, and with constraint enforcement on `end >=
  start` holds on **100.00%** of rows after the reorder because
  each row keeps its own pair.
- **THE PRICE IS ONE NUMBER AND IT IS STATED.** A relationship
  between a date and a NON-date does change: a planted
  date-driven numeric read +0.9730 in source, +0.9643 before and
  **+0.9029** after. The `end - start == los` identity was
  already broken at 11.9% before this and reads 9.3% after -
  a pre-existing defect that a date-difference identity is not
  enforced, named here rather than attributed to the reorder.
- **AND THE TIME AXIS IS RECORDED IN THE BLUEPRINT, because the
  dynamics were MEASURED along it.** Without it the sampler would
  need its own detection rule, and two halves deciding separately
  what the time axis is is how they come to disagree - this file
  says that about token screening, the scaffolding predicate and
  the two categorical draw paths. `patients.time_column` is None
  when the source had no axis, and then nothing is reordered.
- **AND MY OWN CHECKS FAILED TWICE FOR BEING WRONG RATHER THAN
  FINDING ANYTHING.** One asserted `end >= start` unconditionally
  and went red against correct code - it holds on 92.7% of rows
  in BOTH arms without enforcement, so the property to assert is
  that the reorder PRESERVES it, with enforcement on. The other
  claimed to check "a permutation within each patient" and
  actually compared only the column's LENGTH, which cannot fail:
  a mutation replacing the grouped `lexsort` with a GLOBAL
  `argsort` - which moves dates between patients and changes
  every patient's multiset - passed all 45 checks. It is a unit
  check on the function with planted input now.
- **AND THAT FIXTURE THEN COULDN'T REACH THE FAULT EITHER.** Its
  two entities had A's dates entirely BELOW B's, so a global sort
  gives the same answer as a grouped one and the mutation still
  passed. With the ranges OVERLAPPING it turns three checks red.
  Twice in one fix, the same lesson: a fixture must contain the
  thing the check is about, and "two entities with dates" was not
  specific enough.
- **A PATIENT'S VISITS MAY NOT BE A CARE EPISODE, AND NO FIXTURE
  HERE CAN SAY - WHICH I FOUND BY CHECKING MY OWN NUMBER.** With
  the dates ordered, their within-patient autocorrelation read
  **0.74 against a source 0.98**, and the obvious account was
  that the engine draws each visit date from the GLOBAL marginal
  so a generated patient's dates spread across the cohort instead
  of clustering. That 0.98 came from a fixture I BUILT with
  cumulative 3-40 day gaps, where every patient starts on the
  same date - so its patients are episodes by construction.
  Measured on the repo's own extract-shaped fixture instead:
  `visit_start_date` has **icc 0.0062**, with a mean
  within-patient span of 53,383 days against a cohort span of
  106,780 - **half the total range**. That fixture's patients are
  not episodes at all, so an output that spreads their dates is
  FAITHFUL there, and the 0.98 describes my construction rather
  than this data.
  Whether a real patient's visits cluster in time is therefore a
  property of the real extract and NEITHER fixture here can
  answer it. The modelling option - publish a patient-level
  anchor plus a k-screened within-patient GAP distribution and
  draw the dates as a cumulative sum - is written down as the
  route, with its price named in advance: the date column's
  global marginal would become the convolution of anchor and
  gaps rather than the published quantiles, which the gate
  measures. Quoting the 0.74 as a general limitation would be
  the measured-at-one-shape fault, and it was one sentence away
  from being quoted.

- **THE LAGGED RELATIONSHIP SURVIVES NOW, AND THE CURVE WAS RIGHT
  ALL ALONG - ONLY THE COLUMN IT WAS READ FROM WAS WRONG
  (2026-11-08).** The fold's victim was never the SHAPE.
  `discover` measures the curve on the whole group, so on a lagged
  finding that curve IS the lag's: on a planted fixture whose
  same-visit pair reads **0.00**, the published curve for `x` comes
  back `increasing` with a response range of **37** over a child of
  sd 10, which no same-visit curve could be. So the evidence
  already described `y` against the PREVIOUS visit's `x`, and
  generation was applying it to the current one.
  `blueprint.build` marks such a parent in
  `evidence.lag_parents` - it is still a PARENT, because dropping
  it removes the invented pattern by removing the finding too,
  which is what my first version did - and `_apply_numeric` reads
  that parent's value from the patient's previous row.
      pair                      source    before     after
      x~y (same visit)         -0.0038   +0.7901   +0.0149
      x_(t-1) ~ y (lagged)     +0.8871   -0.0020   +0.7870
      u~w (honest same-visit)  +0.9698   +0.9645   +0.9645
  **An absent temporal relationship is now carried at 89% of
  source**, with the invented same-visit pattern gone and the
  honest same-visit relationship bit-identical.
- **AND IT IS INERT WHERE ITS MECHANISM IS ABSENT, ASSERTED AS A
  DIGEST.** The whole path hangs off `evidence.lag_parents`, which
  only a run that BUILT lag features can produce - so on every
  blueprint this project has ever written the branch cannot fire.
  A blueprint predating the field generates BYTE-IDENTICALLY to
  one carrying it empty: absent is not a different fact from empty
  here, and an old artifact must still load. That is the property
  that makes a four-call-site change to the sampler core safe to
  ship, and it is a check rather than an argument.
- **AND A PATIENT'S VISIT ROWS ARE CONTIGUOUS AND IN ORDER, WHICH
  IS WHY NO NEW MACHINERY WAS NEEDED.** `pidx = repeat(arange(
  n_pat), counts)`, so the previous visit is the previous ROW as
  long as it belongs to the same patient. A first visit has no
  previous value and gets NaN, which `_curve_delta` already turns
  into no push at all - the same thing it does for a missing
  parent, which is what a patient's first visit genuinely is.
- **AND THE MEASUREMENT EXPOSED A SEPARATE DEFECT IN THE OUTPUT:
  TWO COLUMNS DISAGREE ABOUT THE ORDER OF A PATIENT'S VISITS.**
  The first reading of the fix said the lag transferred at only
  +0.31, and the reason was the ordering the measurement used.
  Measured on the generated file:
      visit_number in row order          100.0% of patients
      visit_start_date in row order        0.8% of patients
      lagged pair read by ROW order     +0.8614 (source +0.8898)
      lagged pair read by DATE order    +0.3109
  **FIXED the same day, and the reading below that the GATE was
  affected is WITHDRAWN - `pipeline.compare` already orders the
  generated frame by `visit_number`.** See the entry above.
  `visit_number` is emitted as 1..n in row order, which ASSERTS a
  sequence; the date column is drawn from its marginal, which does
  not. A consumer sorting by date and a consumer sorting by the
  index get different sequences, and 99.2% of patients carry at
  least one inversion. **That is present with or without `--lags`**
  and is this project's own alarm - two numbers in one file
  disagreeing - so it is named as its own open item rather than
  folded into the lag work. The instrument now reads each table by
  the order ITS OWN FILE declares (the index where there is one,
  the dates otherwise) rather than picking whichever ordering
  flatters the number.
- **AND THE SOURCE'S ROW ORDER IS NOT THE ANSWER EITHER.** The
  obvious fix - sort each patient's dates so row order and date
  order agree - was checked against the source first, and the
  tidy fixture's own rows are monotone within a patient on only
  **47.5%** of patients. The source file does not encode visit
  order in its rows, and it has no visit index at all; the DATES
  are its only sequence. So the fix is the other direction -
  `visit_number` should follow the generated dates rather than the
  row layout - and it is not free: the real extract declares
  `visit_start_date <-> visit_end_date` at skill 0.9999, so
  permuting one date column within a patient while the other
  stays would destroy a near-identity. Measured first, not
  assumed, and left as the next step.

- **THE 0.97 HAS ITS CAUSE: A LAG PARENT WAS BEING REPLACED BY ITS
  BASE COLUMN, SO AN ACROSS-VISIT RELATIONSHIP WAS APPLIED AT THE
  SAME VISIT (2026-11-08).** The run's own diagnostic printed
  `blueprint planted_lag_y <- ['planted_lag_x']` - the SAME-VISIT
  column, where the finding was on `planted_lag_x__prev`.
  `discover._source` folds `x__prev` to `x` so the catalogue reads
  in the operator's own column names, which is right and which its
  docstring defends on exactly those grounds. It also erases WHEN,
  and that is the whole substance of a lagged finding. The claim
  RECORDS the engineered members it stood for in `via`, and
  `blueprint.build` never read that field - the read-but-not-used
  fault, with a measured consequence. Paired on a planted fixture,
  one file apart:
      pair                 source     old      new
      x~y (same visit)    -0.0038  +0.7901  -0.0106
      u~w (honest)        +0.9698  +0.9645  +0.9591
  The invented pattern is gone and the honest same-visit
  relationship is unmoved. The refusal is REPORTED in
  `excluded.lag_only_parents` with the measured same-visit figure
  that justified it, because a category with no section is
  invisible in any other way.
- **THE TEST IS ASKED OF THE SOURCE, NOT ASSUMED, which is why the
  fold is kept.** A lag-carried parent whose BASE column genuinely
  relates to the child at the same visit still counts - the fold
  is only wrong when it changes what the claim is about. On the
  same fixture `w <- u` is lag-carried too (`via: ['u__prev']`)
  and is kept at a source +0.9698.
- **AND MY FIRST VERSION'S OWN JUSTIFICATION WAS FALSE.** The
  `_lag_carried` docstring said a lag-only `via` means "the base
  column itself was not in the feature set". It is not: the base
  column is a member of its own group whenever it is a feature,
  and `via` merely excludes it. So the flag says the importance
  MIGHT be the lag's and nothing more - which would have been the
  only justification for dropping on the flag alone, and is why
  the decision has to be a measurement.
- **AND RANK CORRELATION CANNOT SEE A U-SHAPE, so deciding on it
  alone deletes the non-monotone same-visit relationships this
  project plants on purpose.** Found because a mutation making
  `_lag_carried` fire on EVERY predictor broke no check: the thing
  saving it was `bool(via)`, which is not a statement about the
  source at all. A k-screened binned conditional mean is the
  second measure - the same definition of a bend the deck and the
  shape criterion already use, so there is one in this codebase
  rather than two - and either may keep a parent. Pinned on a
  fixture whose U-shape has NO linear component: rank correlation
  **-0.032**, bend **2.52 sd**, parent kept.
- **AND AN INTERACTION-ONLY PARENT HAS NEITHER, so the first fix
  deleted an XOR parent and would have shipped.** An interaction
  with no main effect has near-zero rank correlation AND near-zero
  bend in each parent alone - that is the definition of it.
  Measured on a pure-product fixture (`c = 8ab`) with `--lags` on:
  `c <- ['a', 'b']` became **`c <- ['a']`**, dropping `b` at rho
  +0.014 and bend 0.33 against a 0.35 bar. Half an interaction
  pair cannot fire at generation, so that is the same fault class
  the drop exists to prevent, committed by the drop. A parent the
  claim's own `interaction.pair` names is kept whatever either
  measure says, and with the escape the blueprint matches the
  no-lags one exactly.
- **THREE ESCAPES, THREE MUTATIONS, ALL LOAD-BEARING.** Removing
  the pair escape turns the XOR check red, removing the bend turns
  the U-shape check red, and disabling the drop turns three red
  with the invented correlation printed in the message. The
  ordinary same-visit path is untouched: with no lag features
  there is no lag member in any group, so the branch cannot fire.

- **THE BENCH SCORED A KIND AS A PERMANENT MISS FOR MONTHS AND
  THE REASON WAS THE INSTRUMENT, NOT THE ENGINE - AND TURNING IT
  ON MAKES THE OUTPUT WORSE (2026-11-08, M4).**
  `bench_new_path.py` is the instrument that scores the fitted
  path against planted truth, and its one PERMANENT miss was
  `lagged-cross-column`, with its own note saying the kind
  "needs --lags, which exists". It exists in the PIPELINE. The
  bench called `discover` on the raw frame and never built the
  lag features at all, so the kind could not be found by
  construction - a capability present in one half and wired to
  nothing in the half that grades it, which is this project's
  most repeated fault, here inside the measuring instrument.
  With the same construction `synthkit fit --lags` uses - called,
  not rebuilt - over the SAME five seeds:
      seed   without --lags   with --lags
         0        11/13          12/13
         1        12/13          13/13
         2        11/13          12/13
         3        11/13          12/13
         4        11/13          12/13
  **Exactly one kind on every seed**, which is the lagged one and
  not a general improvement - asserted as a property so a future
  change cannot report a broad gain through this door. M4's gate
  asked for >= 12/13 including lagged, and that is met on all
  five seeds.
- **AND IT IS NOT A FREE POINT: THE OUTPUT ACQUIRES A SAME-VISIT
  RELATIONSHIP THE SOURCE DOES NOT HAVE.** On the same runs, the
  pair the fixture relates only ACROSS visits:
      seed   source same-visit   generated, --lags   without
         0        +0.003               0.971          0.001
         1        +0.000               0.969         -0.003
         2        -0.001               0.973          0.003
         4        -0.004               0.971         -0.002
  Without the flag the generated same-visit correlation is
  correctly ZERO; with it the output reads **0.97** on every
  seed. An INVENTED relationship is worse than a missing one
  because it reads as a finding - this project's own rule about
  inversions - so the kind now FAILS survival rather than
  scoring the point, and the verdict names the two numbers.
- **AND THE LAG ITSELF STILL DOES NOT SURVIVE, which took a
  different measurement to see.** The old survival test for this
  kind compared the SAME-VISIT pair, which the fixture plants at
  zero by design, so it correctly declined to score and the row
  read `--` for as long as it existed. Rebuilding the lag from
  each table's OWN ordering gives source **+0.98** against a
  generated **-0.002**: the temporal structure is absent, not
  degraded, which is what a row-level engine should be expected
  to do and had never been stated as a number.
- **AND THE EMITTED `__prev` COLUMN MUST NOT BE TRUSTED FOR THAT
  TEST.** With `--lags` the search sees `x__prev` and the
  blueprint then models it as an ordinary numeric, so generation
  would DRAW it rather than carry the previous row's `x` - and a
  test reading `y ~ x__prev` off the output would pass on a file
  whose lag column contradicts its own previous visit. That is
  the broken-identity class this file already records for
  `age_at_visit` against `year_of_birth`, where the pair
  correlation stayed strong while the identity held on 21% of
  rows. The lag is recomputed on both tables; a mutation that
  reads the column instead turns the check red.
- **AND THE RUN SAYS 55 LAG COLUMNS IN, ZERO OUT - ABSENT IS A
  FINDING, NOT A BLANK.** The first version of the lag
  diagnostic printed its header and NO ROWS, because it required
  the column in both frames and not one reaches the written
  file. A header with nothing under it reads as a broken
  diagnostic where the emptiness IS the answer: a blueprint
  claim whose parent is not in the output cannot fire however
  well the search found it.
- **AND MY OWN DIAGNOSTIC THEN COMMITTED THE NINTH WRONG-KEY
  FAULT, ON ITS FIRST READ.** It printed `blueprint
  planted_lag_y <- []` and I spent a measurement hunting an
  empty-predictor bug - building a fixture, planting such a
  record, and finding generation unaffected. The record is
  unaffected because it does not exist: a BLUEPRINT relationship
  carries **`parents`, a list of NAMES**, where
  `predictors[].column` is the CATALOGUE's shape. This file
  already records that exact pair of names being read backwards,
  and the prescribed fix - assert the key against a LIVE built
  blueprint - is applied here: `smoke_bench` builds one, asserts
  `parents` is a list of strings, asserts `predictors` is ABSENT,
  asserts at least one record is NON-EMPTY so a wrong key cannot
  print an empty list and be believed, and asserts the
  diagnostic reads that key. Restoring the wrong key turns it
  red.
- **WHAT IS NOT CLAIMED: the mechanism behind the 0.97.** The
  lag parent is absent from the output and the claim cannot
  fire, so the invented same-visit correlation comes from
  something else, and the empty-predictor hypothesis is dead by
  measurement. Per this file's no-third-guess rule it earns the
  run's own diagnostic rather than a fourth story, and that
  diagnostic now prints the child's real parents.
  **SUPERSEDED THE SAME DAY: the mechanism is the fold, and it is
  fixed - see the entry above.** The invented pattern came from
  `discover._source` replacing a lag parent with its base column
  so the claim was applied at the SAME visit. What remains open
  is the other half: making the lagged relationship SURVIVE,
  which needs a lag column DERIVED at generation rather than
  drawn - the identity-by-copying rule - and that is a
  generation-ORDER change rather than a post-pass, because the
  child is drawn FROM the lag.

- **A CHAIN CARRIES ITSELF BY COMPOSITION, SO THE REACH BOUND WAS
  NEVER THE GAP - AND THE REAL DEFECT WAS A WEIGHT VECTOR
  (2026-11-08, MT-5, `scripts/hop_depth.py`).** The scorecard
  left two things NAMED rather than measured: four or more hops,
  and whether "the rank-one limit compounds along a chain". Only
  one of those is about depth, and neither is what the
  measurement found.
      hops   source   coupled   control   share
         1    0.900     0.879     0.006    0.98
         2    0.810     0.769     0.003    0.95
         3    0.729     0.676     0.002    0.93
         4    0.654     0.607    -0.005    0.93
  on a five-table chain with **only the DIRECT couplings
  enforced** (`VIA_HOPS = 1`). The deepest cell is the one worth
  quoting: a FOUR-hop correlation of **+0.607** against a source
  **+0.654**, with the control arm at -0.005. A four-hop coupling lands at 93%
  of source without anything detecting it, because each link's
  own enforcement COMPOSES, and the control arm reads zero at
  every hop. At the shipped bound of 3 the same cells read 0.98
  / 0.95 / 0.93 / 0.92 - **the multi-hop detection adds nothing
  on a chain.**
- **WHICH DOES NOT MAKE IT USELESS: IT IS LOAD-BEARING WHERE
  COMPOSITION CANNOT HAPPEN.** On MT-3's shape - an intermediate
  table carrying no numeric to compose THROUGH - there is no
  direct coupling at all and the deep one is the only carrier:
  source **+0.897**, one-hop detection 0.010, multi-hop
  **+0.817**, control -0.013. So the two shapes need opposite things, and
  neither measurement is a statement about the other.
- **THE DEFECT THE CHAIN EXPOSED IS THAT COLINEAR TARGETS WERE
  COLLAPSING ONTO ONE NUMBER WHILE `carried` READ 1.00.** A
  chain builds colinear parent columns by construction: the
  order frame carries `q2`, plus `q1` from one link up and `q0`
  from two, correlated 0.89 and 0.79 WITH EACH OTHER - so one
  link's target is three different numbers against three
  near-twins. The joint solve weighted the parent score by the
  SVD's right singular vector, which is the TARGET DIRECTION and
  not the SCORE WEIGHTS that reach it. Achieved is approximately
  `lambda a b'` with `b = R v / sqrt(v' R v)` for the columns'
  OWN correlation matrix R, so `v = R^-1 q` and setting `v = q`
  is correct only when the columns are uncorrelated. A plain
  weighted sum over near-twins correlates nearly EQUALLY with
  all of them.
      result link, targets 0.891 / 0.795 / 0.704
      svd weights   0.807  0.832  0.795    sq err 0.0166
      whitened      0.923  0.824  0.731    sq err 0.0025
  The achieved SPREAD collapsed from the target's 0.187 to
  **0.010** - three different targets delivered as one number.
  Over five seeds and ten multi-target links whitening improves
  EVERY ONE: mean squared error per link 0.0119 -> 0.0019, worst
  single pair 0.1328 -> 0.0351, and the error on the DIRECT
  coupling - the adjacency a schema declares, rather than one
  derived through it - 0.0705 -> 0.0296.
- **AND `carried` WAS RIGHT THE WHOLE TIME, which is why the
  metric is not being changed.** It reads 1.00 because the
  direction IS reachable: with one child column the achievable
  target vector is `R v` over all v, which spans the space. The
  metric answered its question correctly and the weights did not
  reach what it promised. A 1x3 target matrix also has rank one
  by construction, so `carried` is structurally 1.00 for any
  single-child-column link - true, and worth knowing before
  reading it as reassurance.
- **AND A SINGLE-COUPLING LINK IS BYTE-IDENTICAL, pinned as a
  digest.** With one column there is no correlation matrix to
  solve through, so the change acts only where its mechanism
  lives - and a digest is the only form of that claim a check
  can hold without carrying the previous implementation beside
  it. Reverting the two whitened lines turns three checks red
  with the collapsed spread printed in the failure message.
- **THE RIDGE WAS ADDED, A FIXTURE WAS BUILT TO JUSTIFY IT, AND
  THE FIXTURE REFUTED IT.** By analogy with the within-table
  conditional tier a ridge went into the solve, and on the
  WEIGHTS ALONE it looks necessary: at a parent correlation of
  0.9995 with targets 0.90 / 0.88 the unridged direction gives a
  parent score correlating 0.823 / 0.805 with its own parents
  against the ridged 0.977 / 0.969, because `R^-1 q` points
  along the near-null direction, which is the DIFFERENCE of two
  twins. The ordinary chain's 0.89 cannot reach that, so
  removing the ridge broke no check - and a near-twin fixture
  was built so it would.
      parent |r|    ridged   unridged
        0.9500      0.0332     0.0334
        0.9990      0.0328     0.0327
        0.99999     0.0330     0.0330
        0.9999999   0.0329     0.0329
        1.0000      0.0329     0.0329
  **Identical to four decimals at every colinearity including
  EXACT**, and at 0.9995 the unridged arm is BETTER on two of
  three seeds. The solve searches ten noise scales, so a weaker
  direction is re-absorbed by choosing less noise. Removed, with
  the sweep recorded at the site so nobody re-adds it on the
  weights-level reasoning - the same outcome as the dead
  `_STATE_HINT` exclusion. The fixture is kept for the property
  that survived: a coupling must still land on near-twin
  parents.
- **AND THE REACH BOUND WAS DOCUMENTED AS A SETTING AND FROZEN
  AT IMPORT - THE READ-BUT-NOT-USED FAULT IN ITS MOST LITERAL
  FORM.** `hops: int = VIA_HOPS` in the signature, and a default
  argument is evaluated ONCE, so reassigning
  `multitable.VIA_HOPS` changed nothing and the two call sites
  that omit the argument were pinned at three forever. The
  module's own comment said "deeper is a setting, not a
  default". My measurement script then printed **"detection ran
  at VIA_HOPS = 4" over a run that used three** - a label that
  does not follow its flag, which this file already has a rule
  about. Read at CALL time now, the label reads the value BACK
  from the module rather than from argv, and the check asserts
  the OUTCOME - couplings found at bound 0/1/2 go 3, 5, 6 where
  the frozen version gives 6, 6, 6.
- **AND WHAT THREE REACHES IS FOUR LINKS, which the old comment
  understated by one.** `with_inherited(name, 3)` climbs three
  levels of PARENTS, but `cross_couplings` calls it on the
  PARENT side, so a child reaches its parent plus three above
  it. Measured on a five-table chain where ten couplings exist:
  4 / 7 / 9 / 10 / 10 detected at bounds 0 through 4. Three is
  already complete there and raising it detects NOTHING new - so
  "four or more hops are not measured" was answered by
  measurement rather than by a feature.

- **THE REPORT CARD OF RECORD EXISTS, AND THE REAL EXTRACT'S
  FOUR-ARM SHOWDOWN PUT MY OWN COMPETITOR BELOW THE STDLIB FLOOR
  (2026-11-08).** Goal 8's last named item was "the single
  report-card artifact from a real run - unrun rather than
  unbuilt". It is run. The showdown behind it, every figure read
  back spelled by the machine:
      tier            ceiling   floor   automl   vendor   bar
      strong-signal    0.7884  0.7611   0.7841   0.7611  PASS
      as-specified     0.7040  0.6872   0.6955   0.6872  FAIL
      weak-signal      0.6301  0.6248   0.6082   0.6248  PASS
  The competitor comes within **0.0043 of the ceiling** on the
  strong tier - which is the point of having one - and on the
  WEAK tier it reads **0.6082 against the floor's 0.6248**:
  0.017 WORSE than stdlib logistic regression, against the
  "never worse than the floor" claim I recorded the same week
  from fixtures that never reached that regime.
- **THE CAUSE IS SWITCHING ON A DIFFERENCE THE SLICE CANNOT
  RESOLVE.** With little signal the held-out selection slice
  cannot tell the families apart, so a bare argmax picks
  whichever one the noise favoured. The incumbent is the
  SIMPLEST family now and a flexible one must beat it by
  `1/sqrt(min(positives, negatives))` on the slice - the
  approximate standard error of an AUC, so it tightens as the
  slice grows. Measured on six weak arms: worst delta
  **-0.0123 -> -0.0000**, with the xor and curved gains intact
  (floor 0.551/0.542/0.576 against 0.745/0.703/0.716) and the
  selector still naming different families by mechanism.
- **AND MY FIRST ATTEMPT AT THAT FIX WAS UNREACHABLE CODE.** It
  decided inside the loop, and `FAMILIES` is ordered simplest
  LAST - so every candidate was simpler than the incumbent and
  the margin branch never ran. With and without it the nine
  weak arms came out BYTE-IDENTICAL, including the -0.0123, and
  two arms that agree are one arm. Restructured to score every
  family and then choose.
- **AND A BAR TIGHTER THAN EVERY MEASURED GAP IS CALLED OUT
  NOW, which is the milder form of the miscalibration this
  project already names.** The card flagged a bar ABOVE its
  ceiling - September's 0.700 over 0.698. This run hit 0.700
  against a ceiling of 0.704: achievable by 0.004, while every
  solver on that tier left more than that on the table, so the
  FAIL was about where the bar sits. Reported SEPARATELY from
  the strict case, because they have the same fix for different
  reasons.
- **AND THE `nobody passed` GUARD ON THAT TEST WAS REDUNDANT,
  PROVEN RATHER THAN ARGUED.** If any solver cleared the bar
  its own gap is at most the window, and that gap is in the
  list the minimum is taken over - so `window < min(gaps)`
  already implies nobody cleared it. Checked over 200,000
  random (ceiling, bar, three solvers) triples: zero
  counterexamples. Removed rather than left as an untestable
  comment, and the check carries the sweep so the redundancy is
  measured and not asserted.

- **THE VENDOR EXAM RAN AGAIN ON THE REAL EXTRACT AND
  REPRODUCED ITSELF ON A DIFFERENT SYNTHETIC DRAW, WHICH IS THE
  PROPERTY THAT MAKES IT AN INSTRUMENT (2026-11-08).** bridge ->
  plant (+0.8 age_at_visit, -0.5 span_days, 25% prevalence) ->
  three tiers -> campaign-run, on `traj_off` rather than the
  September run's file:
      tier             AUROC   ceiling    gap
      strong-signal    0.761     0.788   0.027
      as-specified     0.687     0.704   0.017
      weak-signal      0.625     0.630   0.005
  Against the 2026-09-11 run on a DIFFERENT generated file:
  0.758/0.786/0.028, 0.683/0.698/0.016, 0.617/0.622/0.005. Every
  tier within 0.006 of its earlier reading, and every GAP within
  0.001. The exam measures the SOLVER, not the draw - which is
  the whole claim of a computable ceiling, and it had never been
  tested across two synthetic files of the same cohort.
- **AND THE MISCALIBRATED BAR TURNED UP AGAIN, with a window
  this time instead of an impossibility.** The as-specified tier
  reads `auroc = 0.687 (required >= 0.700) -> FAILS` while its
  own CEILING is **0.704** - so the bar is achievable by 0.004,
  and the solver's gap is 0.017, four times the window. In
  September the same bar sat ABOVE its ceiling (0.700 over
  0.698) and no solver could clear it. The gap condition HOLDS
  at 0.017 against 0.150, which is the solver-quality verdict
  and is the one to read. A bar that leaves less room than any
  realistic gap is a bar nobody can pass for reasons that have
  nothing to do with the solver, and this project's own rule
  from the first occurrence was to set bars beneath the ceiling
  or judge on gap.
- **AND `campaign-run` EXITING NON-ZERO ON A FAILING TIER IS THE
  INSTRUMENT WORKING, which cost me a chained command.** I ran
  bridge, plant, compile and run joined by `&&`, so the failing
  as-specified tier short-circuited the chain and the done
  marker never appeared - which reads at first glance as a
  crash. It is not: the gate is a floor and it exited to say so.
  Chain with `&` where a non-zero exit is a VERDICT rather than
  an error, or run the steps separately.

- **NO TOLERANCE CAN DISCRIMINATE AT THE HALF, AND TIGHTENING
  ONE TWICE IS HOW THAT SURFACED (2026-11-08).** The scorecard
  check compared the page's stated overall against the mean of
  its eight goals within 0.1 - so a page stating **93.3** against
  a computed **93.2** PASSED while the check's own message
  printed both numbers side by side. Two numbers disagreeing on
  one line is this project's alarm even when something tolerates
  it, and a reader of that output would see a mismatch described
  as ok. Tightening the bar to 0.05 changed NOTHING: the eight
  goals average exactly 93.25, and float arithmetic puts 93.2
  and 93.3 each 0.0499999... away, so both still passed. The
  comparison had to stop being a tolerance - it is now exact
  against the mean ROUNDED THE WAY THE PAGE ROUNDS IT, and both
  neighbouring values turn it red.

- **THE TRAJECTORY RAN ON THE REAL EXTRACT AS A PAIRED CONTROL,
  AND IT IS A TRADE RATHER THAN A WIN - SO IT STAYS OFF
  (2026-11-07, traj_off / traj_ar, build 21254e9).** Same
  extract, same seed, same k and token cap, run by the same hand
  an hour apart; the ONLY difference is the flag. Every figure
  crossed in words:
      arm        dynamics    close     sign   INV   own-INV
      off          21/93   133/148      145     1         0
      ar           28/93   131/149      146     1         0
  **Seven more graded dynamics columns on the real extract, at
  a cost of about two close pairs.** sign kept improved by one;
  INVERTED and the own-column stop are unchanged. The mechanism
  transfers - the recorded `dynamics tracked 17/87` was never
  structural - and it is not free.
- **THE DENOMINATOR MOVED BY ONE, SO THE PERCENTAGES ARE NOT THE
  COMPARISON.** close went 148 to 149 related pairs, which this
  file's own rule says contaminates a percentage; the ABSOLUTE
  counts are what compare, 133 to 131. And the percentage
  matters anyway for the gate: 131/149 is **87.92%** against an
  87.9% bar, passing by two hundredths of a point, where the off
  arm passed at 89.86%. Trading a comfortably-passing criterion
  for a marginally-passing one in order to gain a different
  criterion is a decision for an operator, not a default - and
  this project has twice shipped a "gain" that a control arm
  later overturned. **OFF by default, with the price on record.**
- **THE REAL PATIENT COURSE IS WEAK, AND MY FIXTURES WERE FAR
  MORE LONGITUDINAL THAN THE DATA.** The extract's latent
  deviation lag-1 is **+0.351** over 45,608 consecutive pairs,
  against 0.89 to 0.95 on every fixture the mechanism was
  developed on. The solver landed 0.300, near it, with 1 of 3
  graded columns outside the bar. So the measured ~0.40 ceiling
  is not the binding constraint here - the source simply does
  not ask for much course - and a fixture built at 0.95 was
  answering a harder question than the extract poses.
- **AND NO COUNTER COLUMN WAS DETECTED ON THE EXTRACT, so that
  fix bought nothing there.** It was load-bearing on the
  fixtures - `visit_number` IS the within-patient index and was
  blocking one graded column outright - and the extract has no
  such column meeting the 99% rule. The fix is right and it
  repaired a fault of my own fixtures' making, which is worth
  saying plainly rather than letting it read as a real-data win.
- **AND THE NINE-ARM SWEEP SAID `close` WAS NEVER WORSE, WHICH
  THE REAL EXTRACT CONTRADICTS.** `dyn_sweep` read close 3/3 on
  BOTH arms of every one of nine arms - because its fixtures
  relate only three pairs, so a two-pair cost is beneath what it
  can resolve. The measured-at-one-scale fault, in a sweep I
  built specifically to catch a cost. A control arm that cannot
  resolve the cost it is watching for is not a control arm, and
  the sweep now says so in its own docstring.


- **THE PATIENT COURSE IS MODELLED NOW, THE RHO IS SOLVED, AND
  THE OBJECTIVE TOOK THREE TRIES TO GET RIGHT (2026-11-07,
  DYN-1).** `dynamics tracked 17/87` sat on the sign-off page
  with "structural - the engine is row-level and models no
  patient trajectory" beside it. It is no longer structural: an
  AR(1) on the latent deviation gives a patient a course, and
  `d_t = rho*d_(t-1) + sqrt(1-rho^2)*e_t` preserves any
  stationary covariance exactly, so the marginal is untouched by
  construction.
- **THE GATE GRADES `within_lag1`, NOT THE TWO MEASUREMENTS IT
  IS DERIVED FROM, AND SOLVING FOR THE WRONG ONE MOVES IT
  BACKWARDS.** Measured on one fixture, one seed, same code
  except the solver objective:
      objective                        solved rho   dynamics
      pooled lag-1                        0.520        2/4
      pooled lag-1 and icc                0.500        2/4
      within_lag1, what the gate reads    0.900        4/4
  The first arm drove pooled lag-1 error to 0.026 - BETTER than
  a fixed 0.7 on every column - and scored worse. The second
  landed 0 of 3 columns outside its own bar and still read 2/4.
  `(lag1 - icc) / (1 - icc)` amplifies both errors as icc
  approaches one, so a solver can match each MEASUREMENT to
  within 0.03 of source and move the DERIVED number the wrong
  way. Optimise what the criterion reads.
- **AND THE WORKING OBJECTIVE LANDS ON THE LATENT MEASUREMENT,
  WHICH RETROACTIVELY SAYS THAT MEASUREMENT WAS RIGHT.** It
  solves 0.900 against a measured latent deviation
  autocorrelation of 0.892 - the two wrong objectives were
  steering AWAY from a number the engine already knew. That is
  the internal-agreement signal this file treats as evidence,
  arrived at from the opposite direction.
- **THE ACHIEVED COURSE IS NON-MONOTONE IN RHO, SO A BISECTION
  IS WRONG IN PRINCIPLE HERE.** Measured by fitting ONCE and
  varying rho at generate, on a fixture whose source needs
  0.47-0.54:
      rho    achieved (systolic / severity / diastolic)
      0.30     0.177  0.187  0.172
      0.50     0.278  0.299  0.302
      0.70     0.395  0.402  0.433   <- the peak
      0.85     0.344  0.439  0.395
      0.99     0.062  0.220  0.147
  Past the peak a high rho stops producing a DEVIATION and
  starts producing part of the patient's LEVEL: the
  between-patient share rises and `(lag1 - icc)/(1 - icc)`
  divides the gain back out. Same inversion the dev_scale sweep
  found - and it explains why a FIXED 0.7 beat both wrong
  solves, since 0.7 IS the peak. The grid is dense through
  0.6-0.8 where the turn is.
- **THE CEILING IS ABOUT 0.40 OF WITHIN-PATIENT AR, MEASURED,
  AND IT IS WHY THE GAIN VARIES BY FIXTURE.** The off arm reads
  EXACTLY 0.000 on every column - the course is absent, not
  degraded - and the ar arm reaches 0.27-0.43 whatever rho is
  set to. A source needing 0.16-0.40 clears the 0.15 bar on
  every column; one needing 0.47-0.54 does not. So the varying
  gain across fixtures is the SOURCE REQUIREMENT varying against
  a fixed ceiling, not the mechanism being unreliable - a
  distinction that would otherwise read as seed noise.
- **AND A WITHIN-PATIENT COUNTER WAS BLOCKING THE CRITERION
  OUTRIGHT.** `visit_number` IS the within-patient index -
  within-AR 0.98 by construction - and the engine drew it as an
  ordinary number, so one graded column could never pass however
  good the trajectory was. A column whose value equals its
  position within the patient plus a constant on 99% of rows is
  reproduced BY POSITION now: exact, not modelled at all, and
  DETECTED from the source rather than named, so the same thing
  happens to a column called `seq` or `encounter_no`. It also
  removes the trajectory's one recorded price, which was that
  persisting the deviations made a counter look
  patient-specific.
- **THE SWEEP IS NEVER WORSE ON NINE ARMS, AND IT COSTS NOTHING
  ON THE RELATIONSHIP CRITERIA.** `dyn_sweep.py`: two
  longitudinal shapes plus a cross-sectional control, three
  seeds each, off against ar.
      fixture  seed | dyn off  dyn ar | close off  close ar | inv
      wide        0 |    1/4     2/4  |     3/3       3/3   |  0
      wide        1 |    1/4     1/4  |     3/3       3/3   |  0
      wide        2 |    1/4     1/4  |     3/3       3/3   |  0
      narrow      0 |    2/4     3/4  |     3/3       3/3   |  0
      narrow      1 |    1/4     4/4  |     3/3       3/3   |  0
      narrow      2 |    1/4     3/4  |     3/3       3/3   |  0
      flat      0-2 |    0/0     0/0  |     0/0       0/0   |  0
  Mean 1.17 to 2.17 graded columns, worse on NO arm, `close`
  and `inverted` identical everywhere, and the flat fixture
  grades zero columns - the criterion staying silent on data
  with no course to lose rather than failing on absence.
- **THE DEFAULT STAYS OFF UNTIL THE REAL EXTRACT SAYS
  OTHERWISE.** Nine fixture arms is not a cohort, and the number
  this exists to move - 17/87 - is a real-extract number. The
  solve also costs eight small generations at fit. Turning it on
  is a decision for the real-extract run, not for a fixture
  result, which is this file's own rule about a gain a control
  arm later overturned.
- **AND THE FIRST SWEEP FIXTURE HAD AN EMPTY CONTROL COLUMN.**
  Each column carried its own independent AR, so discovery found
  no relationships at all and `close` read 0/0 on every arm -
  the sweep could see what the trajectory GAINED and not what it
  COST, which is the half that matters. The columns share a
  factor now, so the criteria that must not move have something
  to move.

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
- **THE NEURAL ENGINE'S OWN 23-SHAPE CLAIM HOLDS ON THE SAME
  STACK, WHICH IS WHAT MAKES THE RULES FINDING ATTRIBUTABLE.**
  `shape_sweep.py --neural` reads **23 of 23 clean** on
  scikit-learn 1.9 - it trains an `MLPRegressor` and never puts a
  frame through the histogram binner, so the crash belongs to the
  rules path's estimator rather than to the stack generally. A
  one-armed re-run could not have said that.
- **AND CI RUNS THE NET BUT NOT THE SWEEP, so "it should handle
  your data" was verified only when somebody remembered.** CI
  installs unpinned (`pip install -e .`), so it HAD the newest
  scikit-learn and still could not see this: no suite in the net
  put a wide, sparse frame through discovery.
  `smoke_dataset_shapes.py` now runs the EIGHT shapes that have
  each found a real defect here - about 35 seconds - and says in
  its own docstring that it is a subset, with the count asserted
  so a reader can see how much of the twenty-three it covers.
- **AND A CRASH-ONLY CHECK WOULD PASS ON A USELESS FILE**, so
  each shape must also return every column the operator gave us
  and no column the SOURCE varies arriving as a single value -
  which is the fault class this sweep found on the neural engine,
  where a code, a note and a date all folded into `__other__`.
- **AND I DESTROYED AN EXISTING SUITE FINDING THAT OUT.**
  `smoke_shapes.py` was taken - it is about EFFECT CURVE shapes,
  rising / U-shaped / threshold / saturating - and I overwrote
  its 259 lines. This file already carries the rule: read a file
  before writing over it, or pick a name the repo does not
  already use. **I RAN `ls` ON THE TARGET AND IT PRINTED EXISTS**
  - and the heredoc that destroyed it was queued in the SAME
  command, so the check ran and its answer could not stop
  anything. A guard whose result arrives too late to act on is
  not a guard. Recovered from git, and the new suite is
  `smoke_dataset_shapes.py`.

- **THE 23-SHAPE SWEEP CRASHED, AGAINST A RECORDED "23 OF 23
  CLEAN", AND THE TRIGGER WAS A LIBRARY VERSION (2026-11-08).**
  `wide + 92% sparse` came back
  `CRASH ValueError: window shape cannot be larger than input
  array shape`. It is NOT today's work - the same crash is there
  at the previous commit - and it is not ours either:
  **scikit-learn 1.9.0 raises on ANY all-NaN feature column, at
  any row count**, where earlier versions handled it. The sweep
  was last re-run on the older stack, so the recorded claim went
  stale the day the data machine's libraries moved. That is this
  file's own rule about testing against what the DATA machine
  installs, caught by re-running a sweep rather than by trusting
  a sentence.
- **AND THE ALL-NaN COLUMN IS PRODUCED BY THE SPLIT, NOT BY THE
  FRAME.** Discovery takes its train/holdout split on the rows
  where the TARGET is observed, so a column present on 8% of the
  frame can be entirely absent from that slice. The fit that
  died was **56 rows by 39 columns, 25 of them under five present
  values and one with none at all** - captured by wrapping the
  estimator's `fit` rather than inferred, because a traceback
  through joblib names the library and not the input.
- **THE FIX IS VERSION-INDEPENDENT: a feature with no values in
  the rows being fitted carries no information about them.** It
  is dropped, and if every candidate is empty the column is
  SKIPPED with that reason rather than crashing. Verified not to
  move the engine's answer anywhere else: on a planted frame with
  three real relationships, discovery's whole catalogue is
  BYTE-IDENTICAL with and without the drop, and a frame with no
  sparse column hashes to the same digest - pinned, because a
  change inside discovery could quietly move every number
  downstream of it.
- **THE SWEEP NOW READS 23 OF 23 COMPLETE, 22 `ok`, and the 23rd
  is the DESIGNED privacy effect rather than a failure** -
  `imbalanced groups  rows -27%  |  1 blueprint contradiction`,
  which is the recorded case where one entity holds half the rows
  and the count's mean lives in the entities the k rule will not
  publish. The old "23 of 23 clean" glossed that line; the honest
  form names it.
- **AND `--help` RAN THE WHOLE SWEEP, which is the third
  instance of that fault here.** `shape_sweep.py` dispatched on
  `"--neural" in sys.argv`, so any other argument fell through to
  the default and `--help` printed twenty-three lines that read
  as an answer to whatever was asked. `latent_challenger.py` did
  the same with a five-seed fixture demo. A command that does
  something plausible instead of refusing is the oldest failure
  shape in this file.
- **SO IT IS A STANDING AUDIT NOW, AND IT FOUND TWO MORE.** Any
  script that dispatches by testing membership in `sys.argv` has
  no parser to refuse for it, so it must refuse for itself:
  `peek.py` and `repro_mechanisms.py` both answered a help
  request with exit **2**, which a wrapper reads as failure.
  argparse-based scripts are exempt because argparse already
  does it.
- **AND THE AUDIT'S FIRST VERSION CHECKED THE WORDING.** It
  searched each script's text for "STOPPED" or "unknown", so a
  script that refuses perfectly well without either word failed
  it - `peek.py` did. It RUNS them now and reads the exit code: 0
  for a help request, non-zero for an argument the script does
  not know, and a timeout counts as unguarded because a help
  request that takes thirty seconds is doing the work, which is
  the whole fault. Assert the decision, not the message.


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

- **"DELIBERATE CORRUPTIONS" TOLD A READER NOTHING THEY COULD ACT
  ON, AND REPLACING IT WITH REQUESTED-AGAINST-ACHIEVED EXPOSED A
  CLASS THE OLD COUNT WAS STRUCTURALLY BLIND TO (2026-10-29).**
  The walkthrough asked for "a confidence / percentage of what was
  requested" instead of a count of corruptions. The count was not
  merely unfriendly - it could not see its own worst case. A
  `case_rate` of 0.5 on an INTEGER column fires on every drawn cell
  and changes nothing (a case drift on digits is a no-op), the
  chain's `dirty != cstr` guard writes NO LEDGER ROW, so a count of
  what happened reports the column as untouched - indistinguishable
  from a rate nobody asked for. Measured on a 2,000-row fixture:
  1,000 cells requested, 1,016 draws fired, **0 landed**, and the
  old readout showed nothing at all.
- **THE SHORTFALL HAS EXACTLY TWO CAUSES AND THEY ARE DIFFERENT
  NEWS, so one total would hide both.** `blocked` means a
  higher-priority op claimed the cell - the spec asking for more
  mess on one column than a cell can carry, which is a dial to
  turn down. `unchanged` means the op does not apply to that column
  at all, which is a spec to fix. On the same fixture
  `site` (missing 0.4 + case 0.4) came up short by COMPETITION
  (318 blocked against 153 unchanged, share 0.41) while `mrn`
  came up short entirely by NO-OP (0 blocked, 1,016 unchanged,
  share 0.00). A single "achieved vs requested" number would read
  0.41 and 0.00 as the same kind of miss.
- **AND THE COUNT IS TAKEN WHERE THE DECISION IS MADE, NOT
  RE-DERIVED FROM THE SEEDS.** The ops are an `elif` chain with
  per-op seeded draws, so which op fired is knowable by replaying
  the same RNGs - and that would be a second copy of the priority
  logic, which this file already records as how two numbers come to
  disagree. The chain increments its own counter instead. Note the
  one branch that needed care: `wrong` sets `op_applied` inside an
  inner `if cand != cstr`, so counting beside the assignment would
  have measured fired-AND-CHANGED and hidden the very cause the
  readout exists to name.
- **A CHECK ON A PHRASE TWO STRING LITERALS SPLIT CANNOT PASS,
  however the page reads.** Asserting `"claimed the "` against a
  source that holds `'...claimed '+ 'the cell first'` failed while
  the rendered text was correct - the wrapped-heading trap this
  file already records, hit again in the same hour. Assert a phrase
  that lives inside ONE literal.
- **AND THE FIRST MUTATION THAT "PROVED" THE READOUT PROVED
  NOTHING.** Setting the reported `"fired"` key to the achieved
  count broke no check, because `blocked` and `unchanged` are
  computed from the local variable rather than the dict - the key
  was reported and almost nothing consumed it. The check asserting
  `achieved <= fired` held trivially at 0 <= 0. Strengthened to pin
  the decision-site count directly (`mrn|case` fired > 900), which
  the same mutation then fails. A mutation that changes a reported
  field no assertion depends on is not a fail-first.

- **"WE ARE FAILING ALL MULTIVARIATE INTERACTIONS" WAS THE
  CRITERION'S SCOPE, NOT THE DATA'S CONTENT (2026-10-29).** The
  operator read `interaction surfaces 0/N FAIL` on a neural run and
  reasonably asked whether the engine synthesizes multi-variable
  structure at all. `measure_surfaces` iterates
  `bp["relationships"]` and scores only those carrying
  `evidence.interaction.pair` - surfaces PUBLISHED by discovery on
  the real data. The rules engine applies them at generation; the
  neural engine samples a latent space and **never consults them**,
  so it fails that criterion BY CONSTRUCTION. The engine-neutral
  measure is the mined-interaction table, and it says the opposite:
  on the real head-to-head the rules engine read **0.000 on every
  one of the source's top eight** two-way interactions - absent,
  not weakened - while the neural engine held the strong ones at
  source magnitude. One chip and one table pointing opposite ways,
  and the chip was the one on screen.
- **AND THE GUIDANCE SENT A NEURAL-RUN READER HUNTING MACHINERY
  THAT ENGINE DOES NOT HAVE.** `NEXT_STEPS` told them to check the
  trimmed-parents report for a parent cut to break a cycle - the
  neural engine trims nothing, a fact this file already records.
  Both texts now name whose contract the criterion grades, put the
  mined table FIRST, and mark the trim step rules-engine-only. The
  texts live in `synthkit.gate` once, so the bench panel and
  `m0_gate` cannot disagree about it.
- **WHAT IS GENUINELY MISSING IS THREE-WAY, AND IT IS A SAMPLE-SIZE
  LIMIT BEFORE IT IS AN ENGINE LIMIT.** Two-way survives (XOR holds
  on every seed); the 3-way mostly dies; `three_way` found ZERO
  qualifying three-column combinations across six candidates on the
  real cohort even with token expansion live. The power curve puts
  3-way structure at ~3,200 PATIENTS and this extract has 800 - and
  generating more rows cannot fix it, because synthetic patients
  drawn from an 800-patient model are not independent evidence. Say
  all three parts whenever the interaction question comes up: the
  criterion's scope, the mined result, and the 3,200.

- **THE SHAP SECTION RENDERED "NOTHING TO ATTRIBUTE" ON DATA FULL
  OF DRIVERS, AND THE GATE WAS A PRESENCE SHARE (2026-10-29).**
  `shap_candidates` required `notna().mean() >= 0.5` on the child
  and on every driver - so on a clinical extract it rejected
  exactly the sparse columns worth attributing. **A column whose
  present values are numbers is numeric however often it is
  present**: this file records that lesson three times already
  (span_days at low cardinality, height at 46.6%, an all-empty
  column) and this was a FOURTH instance, in the attribution
  screen. A model needs ROWS, not a share: the gate is now
  `SHAP_MIN_ROWS = 200` present rows. Fixture at 20% presence with
  300 present rows qualifies, and the old gate fails it - watched
  red.
- **AND THE EMPTY-CASE NOTE ASSERTED A CAUSE IT HAD NOT
  MEASURED.** It told the reader "the strongest patterns in this
  run live on list-token indicators", which was true of one run and
  became a standing explanation - a guess that sends somebody
  hunting the wrong thing. The screen now COUNTS its rejections by
  reason and the note prints the census. Same rule as the
  totals-a-reader-can-check-against and the surfaces skip-by-reason
  identity: when a section is empty, say what you examined.

- **A MERGE THAT MOVES PANELS MUST RE-POINT EVERY READER OF THE
  MOVED FIELDS, AND tick() FAILS SILENTLY (2026-10-29).** Folding
  Generate into Source left FOUR dead paths behind, none of which
  raised: `nPayload`, `sizeAsk` and `neuralReview` each read the
  moved station's own `nsrc`/`nout` boxes - blank by default - so
  Generate, the SIZE LEVER and Review all refused with "give the
  source CSV" on a station displaying that path two inches above;
  and `tick('neural')` hit `if(!b)return`, so generating no longer
  lit the lamp or wrote the completion strip. A station key is a
  STRING in four files and nothing type-checks it. The fix is a
  fallback, not a copy step: `nField(override, base)` reads the
  override when filled and the station's own field otherwise,
  because a value that must be COPIED into a second input before
  anything can read it fails silently whenever the copy does not
  run - which is exactly what `carryForward` did once its one call
  site was keyed on a removed station.
- **AND THE CHECK THAT CAUGHT THE OTHER THREE WAS A NEGATIVE
  ASSERTION.** `"getElementById('nsrc').value.trim()" not in html`
  found two further call sites I had not looked at. Asserting the
  OLD pattern is GONE catches every straggler; asserting the new
  one is present only proves one site was fixed.

- **THE DASHBOARD AND LATENT STATIONS ASKED THE SAME THREE
  QUESTIONS TWICE.** Both wanted source CSV, run directory and
  patient column for ONE run, and each kept its own frame, so
  seeing both pictures meant filling six boxes and navigating
  between tabs. Folded into one Visualize station: one panel, one
  button that starts both, and each picture appears when its own
  work finishes - the deck returns from a single request, the
  latent space is a polled job of minutes, so making the fast one
  wait on the slow one would be a choice rather than a constraint.
- **AND THE PLOTS CARRIED A HARD PIXEL WIDTH INTO A NARROWER
  FRAME.** Every deck SVG sets `width="620"` and a viewBox; the
  bench serves the deck into an iframe narrower than the page it
  was sized for, because the rail takes its share - so a histogram
  needed horizontal scrolling to read. With the viewBox already
  set, responsiveness is one CSS rule (`width:100%`, `height:auto`)
  and the plots now also GROW on a wide pane. The width attribute
  was a natural size, never a commitment.
- **AND A LITERAL PERCENT IN THE DECK'S CSS IS A FORMAT STRING
  BUG.** That block is `%`-formatted (`%(ink)s`), so `width:100%`
  raised `TypeError: not enough arguments for format string` - the
  same trap as the argparse help text this file already records,
  one module over. Double it.

- **AN ANCHOR THAT IS A PREFIX OF THE REAL TEXT DOES NOT FAIL, IT
  RELOCATES THE EDIT - AND IT KILLED THE WHOLE BENCH (2026-10-29).**
  An edit anchored on `function deckBuild(){` matched the substring
  AFTER `async `, so the insertion landed between `async` and its
  own declaration. `deckBuild` kept its `await` without being
  async, the inline script became a **SyntaxError**, no handler
  bound, EVERY BUTTON WENT DEAD, and the wordmark sat at "loading
  build..." forever - which an operator reads as a page still
  loading rather than a page that cannot run. Three times in one
  day: `return o` matched `return out` and truncated a return
  statement; a 16-space `op_applied = "..."` matched the tail of a
  20-space line. An exact-count assertion does NOT protect against
  this, because the count is 1 and the match is real - it is just
  in the wrong place. Anchor on a span that INCLUDES the preceding
  token, or assert the result, not the edit.
- **AND 211 CONTENT CHECKS PASSED ON A PAGE THE BROWSER COULD NOT
  RUN.** Every assertion was true: each looked for text that was
  genuinely present in a file that never executed. A page is not
  its HTML - **a content check cannot see a dead script**, and
  this suite had 211 of them and no parse. Three layers now, each
  watched red against the broken page: (1) every page function
  using `await` must be declared `async` - pure Python, always
  runs, names the offender; (2) `node --check` on every inline
  script block, which catches any syntax error rather than this
  one; (3) the page loaded in HEADLESS CHROME, asserting that the
  build stamp script fills has replaced "loading build..." - the
  only one of the three that can catch a RUNTIME error on load.
  With layer 1 disabled, layer 2 still fails. node's and Chrome's
  absence is PRINTED as a note, never silently passed, the way
  `smoke_buildid` reports SKIPPED without git.
- **THE OPERATOR-VISIBLE TELL IS THE RIGHT THING TO ASSERT.** The
  question asked was "it says loading build... is that what is
  preventing anything from working?" - the symptom and the cause
  were one fault, and the check that reproduces the complaint is
  the check that the stamp gets filled. Assert the observable the
  fault actually changes; this file already said that about three
  checks that could not fail, and it applies to the browser too.

- **A CHECK THAT CAN HANG IS WORSE THAN A CHECK THAT CANNOT FAIL,
  BECAUSE IT TAKES THE WHOLE SUITE WITH IT (2026-10-29).** The
  new in-browser check launched headless Chrome with a 180s
  timeout. On the data machine Chrome wants its own profile
  directory, so it never returned, and `smoke_gui` died with
  `subprocess.TimeoutExpired` - the same class as
  `smoke_packaging` asking about the MACHINE instead of the
  package, and it would have broken `RUN_CHECKS.bat` for the whole
  team. It is strictly BEST-EFFORT now: its own
  `--user-data-dir`, `--headless=new`, a 45s timeout, and ANY
  launch problem becomes a printed note. It asserts only when a
  DOM actually came back. Verified both ways - Chrome present: 216
  checks and the assertion holds; Chrome pointed at a path that
  does not exist: the note prints and 214 pass, no crash.
- **AND A CRASHED SUITE EMITS NO `FAIL:` LINE, SO "NO FAILURES"
  IS NOT "PASSED".** Grepping the data machine's log for `^FAIL`
  returned nothing and that read as clean. The suite had not
  passed - it had raised, after `ok 170`-odd, and `check()` never
  got the chance to print anything. This file already records the
  shape (`_root` vs `ROOT` crashed rather than failed; a crashed
  process also exits). The verdict must come from the SUCCESS
  marker - the final `All N checks passed.` line - never from the
  absence of a failure marker.
- **AND AN OCR-MANGLED REGEX SILENTLY CHANGES WHAT A DIAGNOSTIC
  MATCHES.** Two independent signals said the suite had failed and
  BOTH were artifacts: `len(FAIL lines)` came back as `177 6`
  where the real value was `177 0` (the recorded zero-reads-as-six
  problem), and a PowerShell `-Pattern "^FAIL"` arrived as
  `*FAIL`, which then matched the words "failures" and "fails"
  inside PASSING lines and produced a confident `VERDICT-FAILED`.
  A regex typed through that bridge is as untrustworthy as a digit
  read back from it. Prefer a predicate the MACHINE evaluates with
  a literal (`x[:4] == "FAIL"`), print a word, and cross-check
  against the success marker - the mismatch between the two is
  what exposed it.

- **TWO FIDELITY TIERS, BOTH AIMED FROM A MEASUREMENT RATHER THAN
  A GUESS (2026-10-29 night).** The deliverable's gate read close
  120/137 = 87.6% against an 87.9% bar and INVERTED 1 - M0 NOT MET.
  Both failures were given coordinates before anything was changed.
- **A COUNT THAT TRACKS A SIZE WITHOUT EQUALLING IT IS DRAWN
  TWICE.** `peek.py RUN sizes SRC.csv` on the real extract:
  `condition_count` tracks `len(conditions)` at rank **0.95** while
  EQUAL on only **37%** of rows - a count taken before
  de-duplication - so the copy rule, which asks about EQUALITY,
  never fires and the generated pair reads **0.36**. The drift
  table asks about CORRELATION and the enforcement asks about
  EQUALITY, and on this pair those disagree. A fixture at the
  other shape (81% empty, count equal to the length) does NOT
  reproduce it - detected and enforced exactly, corr 1.0000 - so
  the mechanism works and the gap is DETECTION. Copying would be
  wrong here (these columns genuinely differ from the size), so
  the coupling is landed by REARRANGEMENT, which cannot move a
  marginal because it is a permutation. Paired, three seeds:
  src 0.867/0.873/0.872 -> off 0.813/0.814/0.817 -> on
  0.864/0.867/0.873, marginals identical between arms.
- **AND THE ENGINE WAS NOT ENFORCING WHAT THE GATE GRADES.** The
  gate counts a pair as related when the SOURCE reaches 0.1 and
  calls a sign flip on one INVERTED - its one absolute criterion.
  The conditional-coupling tier enforced only |rho| >= 0.25 AND
  only on columns whose marginal rides the k-banded grid
  (presence under 5%). The extract's single inversion is
  `body_mass_index_measured <- active_drug_count`: source
  **+0.105**, generated **-0.110**, child present on **29%** of
  rows - below the strength floor AND above the sparsity gate, in
  the gap between both. At 16,322 both-present rows a correlation
  of 0.105 has a standard error near 0.008, so it is a real
  effect and not noise to shrug at. Reproduced from those
  coordinates: the coupling decays toward zero (+0.118 -> +0.046,
  +0.147 -> -0.010) and a weak coupling attenuated PAST zero
  reads as an inversion - the rank-of-noise class, again.
  Threshold matched to the gate's own cut and the tier widened to
  partially-measured columns; the joint solve is untouched, so a
  column with several partners still lands them together.
  Measured by the GATE'S OWN RULE (only pairs the source relates
  at >= 0.1 count), twelve seeds: **1 of 8 graded -> 0 of 8**.
- **AND THE NEIGHBOURING RELATIONSHIPS ARE BIT-IDENTICAL.** A fix
  that lands one weak pair and degrades three strong ones is
  worse than no fix, so the same fixture carries couplings at
  three strengths plus an unrelated column. strong +0.600 ->
  +0.641 BOTH arms, medium +0.303 -> +0.320 BOTH arms, unrelated
  +0.005 -> +0.023 BOTH arms; only `weak` moved, -0.025 -> +0.001.
  The change acts only where its mechanism lives.
- **PREDICTION FOR THE NEXT REAL RUN, ON RECORD BEFORE IT RUNS.**
  (1) INVERTED 1 -> 0, and the narration must show `reorder ran
  for body_mass_index_measured` with its row count - a tier that
  cannot prove it executed has not been deployed, which cost two
  round trips before. (2) The narration must show `condition_count
  tracks len(conditions) at rank +0.95 ... enforced by
  rearrangement`, and the `condition_count <- conditions__n` drift
  row should close most of its 0.94 -> 0.45 gap. (3) close should
  move UP; whether it clears 87.9% is NOT claimed - it needs
  roughly one net pair and the drift list holds other mechanisms.
  (4) NOT claimed at all: shapes tracked (7/8), interaction
  surfaces (0/5, ring-parked and scope-limited for this engine),
  dynamics tracked (17/87, structural - the engine is row-level
  and models no patient trajectory). Nothing tonight touches
  those, and a fix that cannot name what it does NOT address
  invites the wrong conclusion from whatever comes back.

- **THE PREDICTION SCORED 1 OF 3, AND THE REAL RUN CAUGHT MY OWN
  FIX (2026-10-29 night, runs run_night and run_night2 on the
  800-patient extract).** All three arms below are the same
  source, same settings (k 10, token_cap 800, person_id); only
  the code differs. Both new arms are SEED 0; the deliverable's
  seed is unknown, which is the limit on every comparison here.

      criterion        deliverable      single-partner   joint
      direction kept   134/137 97.8% P  122/130 93.8% F  122/130 93.8% F
      close            120/137 87.6% F  117/130 90.0% P  115/130 88.5% P
      inverted         1 FAIL           1 FAIL           1 FAIL
      coverage         42/42 P          42/42 P          42/42 P

- **WHAT IS PROVEN: the size-coupling MECHANISM, on real data.**
  `peek sizes` on the new output reads `condition_count <-
  conditions` gen r **0.36 -> 0.73** against a source 0.95, and
  the run's own narration names FIVE couplings the engine had
  been drawing twice - including `procedure_count tracks
  len(procedures) at rank +1.00`, which is the dashboard's WORST
  drift row (1.00 -> 0.38). The tier fires, and it is measured by
  OUTCOME rather than by a log line.
- **WHAT IS NOT PROVEN: that the gate is net better.** close went
  FAIL -> PASS on both new arms and direction kept went PASS ->
  FAIL on both. One criterion traded for another is not an
  improvement, and the denominator moved 137 -> 130 in both arms,
  so the percentages are not directly comparable - in ABSOLUTE
  terms close fell 120 -> 117 -> 115 while the denominator fell
  seven. **M0 is NOT MET in any of the three.**
- **AND THE REAL RUN FOUND A DEFECT IN MY OWN TIER THAT THE
  FIXTURE HAD HIDDEN.** A rearrangement that targets ONE coupling
  randomises every other relationship the column has, and a count
  column is a HUB. On a fixture where the count also drives span
  and procedure: control (no tier) size 0.869 -> 0.858 with span
  0.537 -> 0.566; single-partner size 0.869 -> 0.869 with span
  0.537 -> **0.361**. The control is the damning column - there
  the engine ALREADY lands the size coupling unaided, so the tier
  bought nothing and dragged two relationships down. The first
  probe showed a gain only because its fixture had no competing
  partners: locally right, globally worse, the greedy-cut lesson
  again. The joint solve (b = R^-1 rho, which the sparse tier has
  used since it was written) restores span to 0.527 and proc to
  0.420 while still landing the size coupling.
- **BUT THE JOINT FIX DID NOT MOVE THE REAL GATE.** Paired at seed
  0, joint against single-partner: direction kept IDENTICAL at
  122/130, close 117 -> 115. The fixture damage was real and the
  fix removed it; the real extract's direction drop is therefore
  NOT that damage, and is still unattributed - it may be the
  lowered COUPLING_MIN, the size tier, or the seed. **The missing
  arm is the OLD code at seed 0**, which would separate code from
  seed in one run and was not completed tonight.
- **PREDICTION SCORING, honestly.** (1) INVERTED 1 -> 0: **FAILED**
  - it is still 1 on both arms. (2) The narration must name the
  couplings: **HELD** - five of them, by name, with their ranks.
  (3) close moves up, clearing not claimed: **HELD** - it moved up
  and did clear, but at the cost of direction, which the
  prediction did not anticipate and should have. The thing a
  prediction is FOR is catching this: a fix aimed at one
  criterion must say what it might cost the others.

- **THE CONTROL ARM OVERTURNED THE WHOLE NIGHT, AND BOTH TIERS
  ARE REVERTED (2026-10-30).** Everything above about close
  improving was wrong, and only the OLD CODE AT THE SAME SEED
  could show it. All three arms below are seed 0 on the
  800-patient extract, same settings, same denominator of 130 -
  a clean comparison, unlike the deliverable's 137:

      criterion        OLD (control)     size tier        joint
      direction kept   127/130 97.7% P   122/130 93.8% F  122/130 93.8% F
      close            119/130 91.5% P   117/130 90.0% P  115/130 88.5% P
      inverted         1 FAIL            1 FAIL           1 FAIL
      dynamics         20/86             18/86            18/86

  **The old code is better on every criterion.** close PASSES at
  91.5% without any of my work, so the deliverable's close FAIL
  (87.6% over 137 pairs) was a SEED AND DENOMINATOR effect that I
  mistook for a defect my tier had fixed. The tiers cost direction
  kept (127 -> 122, PASS -> FAIL) and cost close as well
  (119 -> 117 -> 115). Reverted.
- **THE MECHANISM WAS REAL AND THE FIX WAS STILL WRONG.** The
  measurement that motivated it holds: `condition_count` tracks
  `len(conditions)` at 0.95 while equal on 37% of rows, and
  enforcing it moved the generated pair 0.36 -> 0.73 on real data.
  Landing that one coupling simply costs more elsewhere than it
  gains. Locally right, globally worse - the greedy-cut lesson,
  now observed at REAL-DATA scale rather than on a fixture, and
  the second time in one night (the single-partner version failed
  the same way against its own control on a hub fixture).
- **WITHOUT THE CONTROL ARM I WOULD HAVE SHIPPED A REGRESSION AS
  AN IMPROVEMENT.** The first reading - "close FAIL -> PASS" -
  was against the DELIVERABLE, a different seed and a different
  denominator. Every rule needed to catch this was already in
  this file: a single-seed comparison is worth nothing, a
  percentage whose denominator moved is not a comparison, and a
  change must be measured against the same-day baseline. The
  control cost one unattended 40-minute run and was the only
  thing that produced the truth. Run it FIRST next time, not last.
- **WHAT SURVIVES: the diagnostic, not the fix.**
  `peek.py RUN sizes SRC.csv` separates an enforcement fault
  (IDENTITY LOST) from a detection gap (TRACKS, NOT EQUAL), which
  is how the mechanism was found at all; and the deck's joint
  measures (per-column predictability with driver agreement, and
  TSTR) which touch no engine code. The engine is byte-identical
  to the build that produced the deliverable.

- **THE ABSOLUTE STOP WAS JUDGING A COLUMN THAT IS NOT IN THE
  FILE (2026-10-30).** Asked whether SynthKit could repair an
  inversion and then re-measure to see if the whole gate improved,
  the repair tool was built - and on the real extract it reported
  `child absent` and had nothing to rearrange. The inverted pair
  was `procedures__n <- mean_arterial_pressure_invasive`, and
  `procedures__n` is a set-SIZE column: search machinery, dropped
  before generated.csv is written. Read back in WORDS so no digit
  could be misread: `pairs_inverted = one`,
  `pairs_own_inverted = zero`. **Nobody can read a column that is
  not there**, so the one verdict that says STOP was counting
  scaffolding. The gate has computed `pairs_own_inverted` all
  along and printed it in a note while the CRITERION judged the
  scaffolding-inclusive total. Same fault this file already
  records for the constraints section. The stop now judges the
  operator's columns; a scaffolding inversion is REPORTED,
  because it can still mark a modelling fault; and a run that
  predates the split falls back to the total, because absent is
  not known-to-be-zero.
- **THIS IS A MEASUREMENT CORRECTION, NOT A FIDELITY GAIN, AND
  THE DIFFERENCE MUST BE SAID OUT LOUD.** Not one value in the
  generated file changed. On run_old0 the gate goes from four
  PASS to **six of nine**, the absolute stop clearing, with the
  residue being shapes tracked 7/8, interaction surfaces 0/5
  (the criterion grades surfaces this engine never consults -
  already recorded) and dynamics tracked 20/86 (structural: the
  engine is row-level and models no trajectory). Reporting that
  as "fidelity improved overnight" would be the same error as
  last night's close reading.
- **AND THE REPAIR TOOL STANDS, UNUSED BUT PROVEN.**
  `repair_inversions.py RUN SRC.csv` rearranges an inverted
  pair's child to recover the source rank correlations - jointly
  across every partner above 0.10, because targeting one pair
  randomises the rest - and then re-measures the WHOLE gate,
  printing every criterion before and after. It writes nothing
  without `--apply`: a repair aimed at one criterion can cost
  others, which is not a hypothesis but what two tiers did last
  night. Verified on a planted inversion: pair -0.119 -> +0.155
  against a source +0.129, two neighbours unmoved at 0.515 and
  0.370, and the repaired column holding the SAME MULTISET as the
  generated one - a permutation cannot move a marginal.

- **REPAIRING THE INVERSION IMPROVED THE WHOLE GATE, AND THE
  MEASUREMENT HAS NO SEED TO ARGUE ABOUT (2026-10-30).** The
  operator asked whether SynthKit could rearrange an inverted
  pair back to the sign it should have and then re-measure to see
  if overall fidelity rose or fell. On run_deliverable, whose
  inversion IS in the operator's own data
  (`body_mass_index_measured <- active_drug_count`, source
  +0.105, generated -0.110, 15,112 both-present rows):

      criterion        before             after              verdict
      direction kept   134/137 = 97.8%    135/137 = 98.5%    improved
      close            120/137 = 87.6% F  121/137 = 88.3% P  FIXED
      inverted         1 FAIL             0 PASS             FIXED
      coverage         42/42              42/42              same
      set token shares 79/79              79/79              same
      set EMPTY rate   4/4                4/4                same
      shapes tracked   7/8                7/8                same
      surfaces         0/5                0/5                same
      dynamics         17/87              17/87              same

  The pair landed at +0.108 against a source +0.105. **Two
  criteria fixed, one improved, nothing degraded** - the
  deliverable goes from four PASS to six of nine, matching
  run_old0.
- **AND THIS IS THE COMPARISON LAST NIGHT'S WORK COULD NOT MAKE.**
  Every claim last night was cross-RUN - different seed, moving
  denominator - which is why close looked fixed when the control
  later showed it was never broken. This is a PAIRED before/after
  on ONE file with ONE column reordered: same rows, same seed,
  same denominator. There is no seed to argue about, which is
  exactly why the operator's framing ("then measure fidelity
  again to see if it got better or worse overall") was the right
  one and should be the default shape for any fidelity change.
- **A PERMUTATION CANNOT MOVE A MARGINAL, SO THE PRIVACY POSTURE
  IS UNTOUCHED.** The repaired column holds the SAME MULTISET the
  generator produced - mean, spread, point masses and the
  k-anonymous bounds are all exactly what was published. Only the
  ORDER changes. Nothing is written without `--apply`.
- **THE DELIVERABLE IS NOT THE BEST ARTIFACT ON THAT MACHINE.**
  Same code, same settings, different seed: run_old0 reads six of
  nine with the absolute stop clear and close at 91.5%, where
  run_deliverable needs the repair to reach the same place and
  sits at 88.3% after it. Worth saying plainly before anyone
  ships the one that happens to be named "deliverable".

- **THE REPAIR IS APPLIED, AND IT IS PROVABLY A PERMUTATION OF
  ONE COLUMN (2026-10-30).** `--apply` wrote
  `generated_repaired.csv` and `fidelity_repaired.json` beside
  the originals, which are untouched. Checked on the machine
  rather than asserted: **ROWS-SAME, MULTISET-SAME,
  ORDER-CHANGED, OTHERCOLS-SAME** - the repaired column holds
  every value the generator produced, only its ORDER differs, and
  every other column is byte-identical. So the published bounds,
  the point masses, the k-anonymous extremes and the whole
  privacy posture are exactly what they were; nothing new is
  disclosed by reordering values that were already released.
- **AND THE RESULT IS A DROP-IN RUN DIRECTORY.** Writing the
  repaired file beside the original is safe but useless to the
  gate, the deck and the bench, which all read `generated.csv`
  and `fidelity.json` from a run directory. `run_deliverable_
  repaired` is a full sibling with those two files swapped, so
  every existing tool consumes it with no flag and no special
  case. Gated through the normal path: **six of nine PASS**,
  direction 135/137 = 98.5%, close 121/137 = 88.3%, inverted 0.
  A repair that only a bespoke command can read is a repair
  nobody will use.

## Talking to the data machine
- **`focus` REPORTED ALREADY-FOREGROUND WHILE A QUEUED READ CAME
  BACK AS ANOTHER PROJECT'S BROWSER SCREEN (2026-11-11).** Three
  calls, three different answers about one machine: `run` refused
  with "cmd prompt not foreground" and a window title full of
  browser tabs; `hands focus` then answered **ok
  (already-foreground)**; a third call refused for a DIFFERENT
  reason - another driver holding the single-flight lock with a
  PowerShell command - and the same call with `--wait` took the
  lock and returned a Copilot page belonging to a different piece
  of work, with **my marker absent**. This file already records
  that the health fields cannot see a console in selection mode;
  the new part is that **`focus`'s own answer cannot see it
  either**, so there are now two signals that report green over a
  screen that is not live.
- **THE MARKER IS WHAT MADE IT VISIBLE, and nothing else would
  have.** The returned text was plausible prose - evidence rules,
  a knowledge pane, a terms line - and the recorded worst case of
  this fault is a read of ANOTHER PROJECT's model-comparison
  output being reported as this run's fidelity. It happened again
  here, and the only thing that caught it was requiring a nonce
  echoed in a SEPARATE call to come back. Do not read a verdict
  off that bridge without one, however healthy `status` and
  `focus` both look.
- **AND THE ANSWER IS TO STOP DRIVING, NOT TO TAKE THE WINDOW.**
  Somebody has a browser up on other work. Forcing the console
  forward means a global window key, which this file forbids
  because the machine is shared, or clicking around a logged-in
  corporate page, which the standing instruction forbids outright.
  A refusal from the focus guard is the guard EARNING its place -
  it is the same guard that once stopped shell commands being
  typed into a colleague's chat window - so the correct response
  is to wait for the console rather than to argue with it.

- **THE PURPOSE-BUILT SUBCOMMAND EXISTED AND I COMPOSED A SLOWER
  EQUIVALENT OUT OF A MORE GENERAL ONE (2026-11-11).** The task was
  the rules engine against the neural engine on the real extract.
  `latent_challenger.py --help` lists, in four words,
  **`compare    this engine against the rules engine`** - it takes
  a finished fit run by `--blueprint-run`, mines the strongest
  two-way interactions from the SOURCE and measures both engines
  against them. I ran `csv` instead, which does a standalone fit,
  generate AND its own full re-discovery of the source. The
  re-discovery is redundant for a comparison and is the expensive
  part: the run took about FOUR HOURS where the fit and generation
  together were about two, and the rules engine does its whole job
  in twenty-two minutes. The output is still usable so nothing was
  wasted, but the extra two hours were a routing error. Read the
  usage before composing a run out of a general subcommand - the
  tool had already been given the exact job.
- **AND THE BRIDGE'S REFUSAL TEXT IS THE ONLY RELIABLE LIVENESS
  SIGNAL WHILE A JOB HOLDS THE CONSOLE, which is worth more than
  any probe.** With a long job running in a console's foreground
  every `hands run` queues behind it, so a probe cannot report
  until the job ends - it cannot tell "working silently" from
  "hung". But the focus guard distinguishes them for free:
      terminal busy (a command appears to be running --
      screen is changing)              -> the job is ALIVE
      cmd prompt not foreground (title '<a browser>')
                                       -> the screen stopped
                                          changing, so the job has
                                          probably ENDED
  The reason CHANGING from the first to the second is how I knew
  the run had finished, after an hour in which no probe could
  answer. Read the refusal rather than grepping past it - this file
  already says to grep for `HANDS refused` FIRST, and this is the
  stronger form: the refusal carries information the output does
  not.
- **AND THE RECOVERY ORDER IS `key esc` THEN `term new`, not
  `focus`.** Measured over several attempts on one stuck console:
  `focus` reported **ok (already-foreground)** while the guard
  immediately refused saying the foreground was a browser, and
  later failed outright with **no cmd channel found after 8
  alt+tab cycles**; `term new` alone failed with **no prompt
  visible after opening a new console**; and `key esc` followed by
  `term new` worked first time. The esc clears a console sitting in
  selection mode - which this file already records as a state the
  health fields cannot see - and only then can a new console come
  up with a visible prompt. Try them in that order.
- **"A PYTHON PROCESS EXISTS" IS NOT EVIDENCE MY JOB IS ALIVE,
  BECAUSE MY OWN PROBES ARE PYTHON (2026-11-11).** Polling a
  launched run with
  `tasklist /fi "imagename eq python.exe" | find "python"` reported
  RUNNING for a job that had died at launch - the process it found
  was the `python -c` of the probe I had just sent. The job's log
  was **8 BYTES** and its output directory did not exist, and both
  of those facts were available the whole time. This is the
  predicate-does-not-match-the-thing fault in a new place: this
  file already records a poll pattern matching its own echoed
  command, and a `^FAIL` grep that matched the word "failures" in
  passing lines. Poll on the ARTIFACT the job is supposed to
  produce, or on a window title the job owns - never on an image
  name a probe shares.
- **AND AN 8-BYTE LOG IS A FAILED LAUNCH, NOT A SLOW START.** The
  detached form that works for the rules engine -
  `start "x" /min cmd /c "python script ... > log 2>&1"` - produced
  8 bytes here, because this command carries a SUBCOMMAND plus a
  quoted positional path and the extra nesting broke the redirect
  inside `cmd /c "..."`. The same shape had worked three times
  earlier the same day with a flags-only command. Check the log is
  GROWING before walking away from a detached launch; a log that
  exists is not a log that is being written.
- **AND A LONG JOB MUST NOT BE RUN IN A CONSOLE'S FOREGROUND, which
  cost four unreliable reads.** Run in the foreground it works
  fine, but every subsequent `hands run` types into the console the
  job occupies, and the screen then returns a MIX of the job's
  output and the probe's - one read came back carrying another
  project's PostgreSQL message. `term new` did not reliably
  separate them. The working pattern is the one that already
  worked: launch detached with a redirect, verify the log grows,
  then poll the artifacts. If a job is already running in the
  foreground, STOP DRIVING until it ends - queued keystrokes
  execute later, and this file records interleaved input returning
  `python. exe € r+ =o: \ a`.
- **AND THE FOREGROUND RUN IS WHAT PROVED THE SETTINGS LANDED,
  which is the one thing it was good for.** With no redirect the
  echo went to the screen and read back: `caps: set tokens 60 ->
  800`, `--group-by person_id --seeds 11 --k 10`, `k-aware blur:
  on`, then `encoded frame: 47029 rows x 1615 dimensions`. Every
  setting the arm needed, confirmed as received rather than
  assumed - which is this file's own rule for that terminal, and
  the detached launch had silently delivered none of it.
- **A COMPOUND `cmd` LINE CAN DROP ITS POWERSHELL HALF
  SILENTLY, AND THE NEXT STEP THEN RUNS AGAINST NOTHING
  (2026-11-07).** `echo MARK & powershell -Command "Expand-Archive
  ...; Rename-Item ..." & if exist ... & echo MARK` returned both
  markers and NO error, and the archive was never expanded - so
  the launch that followed cd'd into a directory that did not
  exist and produced no log at all. Diagnosed by the recorded
  rule rather than by a message: a probe said the tree was
  missing, I ran `Expand-Archive` ALONE, and it printed
  EXPANDED. One command per line on that terminal, which this
  file already says for `%USERPROFILE%` and for wrapped pastes -
  the new part is that the failure is SILENT and the symptom
  appears one step later as an empty log.

- **A DIFFERENT CONFIGURATION IS DIFFERENT EVIDENCE TOO, AND THE
  MISMATCH IS INVISIBLE WITHOUT COMPARING (2026-11-07).** The
  real-extract attack launched at the SHIPPED token cap of 60
  while the run being certified used 800 - caught by reading the
  attack's own first lines ("5,702 tokens found, 533 clear k, 60
  modeled") rather than by any check. It matters in the
  dangerous direction: a larger cap publishes MORE vocabulary
  about members, so an attack measured at a smaller cap
  UNDERSTATES the risk of a larger-cap release. The attack now
  RECORDS its caps and the package compares them against the
  run's own contract.
- **AND THE TWO CAUSES KEEP SEPARATE WORDS, because they take
  different fixes.** FIXTURE means the measurement was made on
  ANOTHER DATASET - re-run it on this cohort. CONFIG means it
  was made on this data under DIFFERENT CAPS - re-run it at
  these settings. Collapsing them into one word would be the
  blocked-versus-unchanged fault this file already records for
  the corruption ledger and for route B's two opposite readings
  of one number.
- **AND AN ENUMERATED STATUS MAP CRASHED THE MOMENT A STATUS WAS
  ADDED.** The renderer held `{present, FIXTURE, OWED}` and
  raised a KeyError on CONFIG - a renderer that must be edited
  whenever a status is added is the thing that breaks when
  somebody adds one. It renders any status now.
- **AND THE STUCK CONSOLE CAME BACK, with the recorded recovery
  working exactly as written.** Three reads returned a frozen
  screen with my markers ABSENT while `status` reported
  reachable, keyboard_online, not busy and awake - the health
  fields cannot see a console in selection mode, which this file
  records. `enabled=False` was the recorded red herring again.
  `key esc` cleared it in one command and the next marker came
  straight back. The marker discipline is what made the
  staleness visible: without requiring my own fresh marker back,
  three stale reads would each have looked like an answer.


- **THE BENCH'S 11/13 WAS A SEED, NOT A REGRESSION, AND A
  SWEEP SETTLED A DEBT THAT HAD BEEN CARRIED UNATTRIBUTED
  (2026-11-06).** The sprint board read "the bench reads 11/13
  on its default config (het unexplained at discovery) against
  a recorded 12/13 - still unattributed". Five seeds:
      seed  FOUND   het source   het found
       0    11/13     0.182         no
       1    12/13     0.247        YES
       2    11/13     0.234         no
       3    11/13     0.230         no
       4    11/13     0.184         no
  **The only kind that moves is `heterogeneous`, and it is
  missed at 0.234 and found at 0.247** - the same strength
  either side of discovery's inclusion cut, which is the
  threshold and not a weak effect. The recorded 12/13 was the
  lucky seed of five, and `smoke_bench`'s own docstring had
  stated it as THE result for months.
- **SO THE INSTRUMENT REPORTS THE RANGE ITSELF.** This file's
  oldest rule about seeds is to report the range and never a
  point, and the bench was printing a point - which is exactly
  how a later reader comes to see 11 as a regression against a
  recorded 12. The note is now a FUNCTION returning lines,
  computed from the recorded sweep, so `smoke_bench` can CALL
  it and check the arithmetic: a content check cannot see a
  dead code path, and a range typed by hand is not a range
  derived from a measurement. Three mutations bite - a
  single-seed sweep, a hardcoded range, and the dropped
  not-a-regression caveat.
- **AND A PERMANENT MISS IS NOT A FRAGILE ONE.**
  `lagged-cross-column` misses on EVERY seed because it needs
  `--lags`; `heterogeneous` misses on four of five because of a
  threshold. Reporting them as one number would invite the
  wrong fix, so the note names both with their reasons.

- **THERE ARE NOW THREE CHECK-COUNT DELTAS AND ALL THREE
  DECOMPOSE WITH NO RESIDUE (2026-11-06).** The rule has always
  been that a gap between two runs must be a number somebody
  ACCOUNTED FOR, because that is how a correct run gets read as
  a failure. Two measurements tonight, and each pair is ONE
  COMMIT - the first write-up of this entry mixed two commits
  and quoted a data-machine count against a local count from a
  later build, which is the wrong-denominator fault in a note
  about counting:
      at fa2afb1   dev py3.14  3054   data machine  3042   -12
      at 53b5cdc   dev py3.14  3060   dev py3.10    3056   - 4
  The TWELVE is the recorded decomposition - nine in
  `smoke_buildid` which needs git for the archive path, one for
  node, two for Chrome, all reporting NOTES and passing rather
  than failing. The FOUR is new and sits entirely in
  `smoke_dialmatrix` (36 against 32), which is exactly the
  `staged()` checks pinned on pandas 3+ because four dial
  verdicts genuinely differ by library version. Verified by
  diffing the per-suite counts rather than by assuming: no
  other suite moves at all.
- **AND THE WHOLE SESSION'S WORK WAS RE-RUN ON THE OLD LANE FOR
  THAT REASON.** Everything tonight was developed and measured
  on 3.14 with pandas 3, which is what the data machine
  installs - and CI also runs 3.10. The mirror of the recorded
  wrong-venv lesson: testing only against the newest stack
  leaves the TEAM's older lane unverified. ALL GREEN on both,
  96 suites either way.

- **THREE HOPS WAS UNREACHABLE, NOT WEAKENED, AND THE FIX WAS
  RECURSION RATHER THAN A SOLVER (2026-11-07, MT-4).** MT-3
  made `with_inherited` carry a PARENT's numeric columns down,
  which reaches a grandparent. It climbed exactly ONE level, so
  on a person -> visit -> order -> result chain a result riding
  the PERSON's age - three links up - was never a column
  anything could see. Measured through the written files on a
  four-table fixture:
      source three-hop creatinine ~ age   +0.976
      before (one level)                  +0.010
      after (recursive, bounded)          +0.981
      control arm, --no-couple            +0.010
  The run's own output said "0 cross-table coupling(s)", which
  reads as none there rather than none looked for - the same
  shape of silence MT-3 fixed one level down.
- **THE PARENT FRAME IS PASSED THROUGH THE SAME FUNCTION BEFORE
  BEING JOINED.** A grandparent's column arrives on the parent
  as a `__via__` column and is then carried one level further,
  so the existing detection and the existing rank-one
  assignment work unchanged on a column whose NAME happens to
  record a longer path. The one-hop version SKIPPED `__via__`
  columns on the parent, which is exactly what made it one-hop;
  carrying them is the whole change.
- **AND THE PATH IS IN THE NAME, because these appear in the
  run's own report.** Naive nesting gives
  `__via__visit_id____via__person_id__age`; the path form
  `__via__visit_id>person_id__age` says the same thing and can
  be read aloud. Collapsing it turns a check red, because two
  different paths could then produce one name.
- **BOUNDED AT THREE, AND A CYCLE IS REFUSED RATHER THAN
  TRUNCATED.** Each level joins an ancestor's frame, so cost
  grows with the chain and a deep schema would price a
  detection pass nobody asked for. Three reaches a
  great-grandparent. The depth bound alone would merely
  truncate a loop in the link graph, so there is a visited-set
  guard as well, with its own check. FOUR or more hops are not
  measured and the rank-one limit compounds along a chain -
  named, not assumed cheap.

- **A VIN IS NOT SHAPELESS - IT CARRIES A CHECK DIGIT, AND THE
  INVENTORY WAS WRONG ABOUT THAT (2026-11-07, PHI-6).** The
  Safe Harbor inventory marked categories 12 and 13 NOT COVERED
  on the reasoning that "a VIN is indistinguishable from any
  other alphanumeric code". That was my own sentence and it was
  wrong. A VIN is seventeen characters, EXCLUDES I, O and Q so
  they cannot be confused with 1 and 0, and position NINE is a
  check digit computed from the other sixteen - as verifiable as
  an SSN's three-two-four and far more specific. Measured
  coincidence floor: random strings from the VIN alphabet clear
  the checksum 3.1% of the time (1 in 33, the modulus being 11
  with X absorbing the 10 case) against a column needing 50%.
  The negative control is the SAME seventeen characters with the
  check digit moved, so only the arithmetic separates them.
- **AND A DEVICE UDI HAS TWO HALVES, ONLY ONE OF WHICH
  IDENTIFIES A UNIT.** `(01)` is the DEVICE identifier - a GTIN,
  a MODEL code, the clinical fact of which device was implanted.
  `(21)` serial, `(10)` lot, `(11)` production date and `(17)`
  expiry are PRODUCTION identifiers: which unit, and when. The
  detector fires on a production identifier and deliberately
  does NOT fire on a DI alone, because a bare GTIN identifies
  nobody and dropping it would remove a covariate to delete
  something that is not an identifier - exactly the trade the
  REMEDY table exists to refuse. The finding says which it saw,
  so a reader making an Expert Determination argument can see
  the tool did not pre-empt it. The mutation that flags a DI
  turns THREE checks red.
- **EVERY SAFE HARBOR CATEGORY IS NOW AT LEAST PARTIALLY
  REACHED - 8 covered, 8 partial, 2 not-in-a-table, ZERO not
  covered - WHICH BROKE TWO OF MY OWN CHECKS.** One asserted
  that vehicle and device identifiers were specifically `not
  covered`; the other that every status word had an instance.
  Both went red when the gap was CLOSED, which is a check that
  punishes progress. Rewritten to pin the PROPERTY instead: any
  category short of covered must NAME what it does not reach,
  whichever those turn out to be. That also exposed three notes
  reading only "same rule as an MRN, same limit" - a
  cross-reference a governance reader has to chase, which is the
  count-a-reader-cannot-act-on fault in a coverage table. Each
  states its own limit now. STILL NOT REACHED, named: license
  plates (no checksum, state-varying) and free-form serials with
  no GS1 syntax.

- **A CHECK ON A DOCUMENT CAN BE SATISFIED BY THAT DOCUMENT'S
  BIBLIOGRAPHY, AND IT TOOK FOUR ATTEMPTS TO MAKE ONE FAIL
  (2026-11-07, `smoke_competitive`).** The check asserts each
  named competitive GAP is still named, so closing one has to be
  a deliberate edit rather than a quiet deletion. Attempt one
  looked for the phrase ANYWHERE and passed, because "Expert
  Determination" appears in a source link's title. Attempt two
  narrowed to the gaps SECTION and passed, because the phrase
  survives there inside the sentence describing what TONIC
  documents - a vendor claim, not our gap. Attempt three moved
  to the scoreboard and the build list and STILL passed,
  because the build-list slice ran to end-of-file and picked up
  the bibliography again. **A source list is evidence FOR a
  page's statements and must never be able to satisfy them**, so
  the Sources section is cut before any prose match. Only
  instrumenting what the check actually saw found it; three
  rounds of reasoning did not.
- **AND THE NUMBER CHECK WAS SELF-FULFILLING IN THE WAY THIS
  FILE ALREADY RECORDS.** It held a hardcoded tuple of our
  figures and asserted each appeared in the record - verifying
  THE LIST, not THE PAGE. Changing a figure in the document
  failed nothing. Figures are EXTRACTED from the prose now and
  checked against the record, which is the oldest rule here
  pointed at the one page most tempted to break it. The
  extractor then needed its own negative control: unanchored, it
  matched `505.01524` as a three-digit suffix inside the arXiv
  id `2505.01524` and reported a figure nobody measured.
- **AND WHITESPACE IS NORMALISED BEFORE EVERY PHRASE MATCH,
  which is the mechanical form of a rule recorded six times.** A
  phrase a LINE BREAK splits cannot be grepped however correctly
  the page reads: "Business Source License" landed with a
  newline inside it and a check went red against a document that
  said exactly the right thing. Reflowing the prose would fix
  one instance; collapsing the whitespace closes the class.

- **OUT OF SCOPE MEANT UNINSPECTED, AND THAT IS A CLEAN BILL OF
  HEALTH FOR WHATEVER NOBODY LOOKED AT (2026-11-07, PHI-7).** A
  column whose average value ran past `LONG_TEXT` was reported
  OUT OF SCOPE by name and never read - honest about the
  DECISION, and it meant nobody knew what was in the most
  information-dense column in the record. The decision about
  redacting a clinical note is governance; the MEASUREMENT is
  ours. The battery runs over the TOKENS now, the same move a
  set column uses, because the joined prose matches no pattern
  even when its pieces do. Two faults found on the first probes:
  an email counted TWICE because only the phone can span a token
  boundary, and a URL counted NEVER because the tokeniser splits
  on `:` and cut `https://` in half.
- **AND THE REPORT MUST NEVER QUOTE WHAT IT FOUND.** Counting is
  safe; printing the span is not. A PHI detector whose own
  output carries the identifier it found has MOVED the leak
  rather than closed it, and this is the one place in the module
  where the matched text is somebody's note. Counts, kinds and
  the ROW SHARE only - enough for a reviewer to tell one stray
  from a systematic field. The check plants a name, a phone and
  a date as module constants and asserts NONE appears anywhere
  in the serialised report; adding an `example` field turns it
  red.
- **THE REMEDY IS A THIRD CLASS, AND IT HAD TO BE.** Dropping a
  clinical note destroys the densest column in the record;
  "generalising" free text is not a defined operation; and
  REDACTING it is exactly the governance decision this tool does
  not make for anybody - a redactor that misses one identifier
  is worse than none, because it hands back false confidence.
  Proper free-text de-identification needs clinical NER, which
  this project does not have and does not pretend to. So
  `review`: the column stays, `pending` keeps saying it is owed,
  and `apply` needs NO special case because it drops only what
  is marked drop - which is the argument for the remedy living
  on the finding rather than in the applier.
- **A REVIEW ITEM IS NOT A TOOL DEFECT, AND THE EXIT CODE SAYS
  SO.** `--apply` refused with exit 1 if a fresh detect still
  found anything - correct while every remedy was drop or
  generalise, both of which it performs. Folding `review` in
  would have made `--apply` report a defect FOREVER on any table
  with a clinical note, which is wrong and is the kind of
  permanent red that teaches people to ignore an exit code. Exit
  1 now means the tool failed to do what it claimed; exit 3
  means a human decision is outstanding, with NOT CERTIFIED and
  a note that everything automatic WAS applied. Nothing
  exercised that until a mutation conflating them found no check
  to break - and the control is a table with NO free-text
  column, which must still reach 0 or exit 3 is a permanent red
  wearing a different number.
- **AND THE IDEMPOTENCE CHECK'S PREMISE CHANGED WITH IT.** It
  asserted a fresh detect finds NOTHING after apply and
  generalise. That is a demand that the tool do the thing it
  refuses to do, so it now asserts nothing with an AUTOMATIC
  remedy remains, and separately that the review finding
  deliberately SURVIVES. A check that insists on work the tool
  declines is worse than no check, because the only way to
  satisfy it is to start redacting.
- **AND THE TEAM BUNDLE'S OWN SELF-SCAN CAUGHT THE NEW SUITE,
  EXACTLY AS IT DID THE LAST TIME.** `smoke_competitive` read
  `docs/SPRINT.md` to verify figures, and the sprint board is on
  the bundle's EXCLUSION list, so the scan went DIRTY and
  refused to finish. The established fix is a `noqa: bundle-scan`
  marker; the better one here was to stop reading it -
  CONVENTIONS and the scorecard are the authoritative record,
  and a suite that cannot run inside the filtered bundle is a
  suite the team cannot run at all.

- **I USED A REPLACE HELPER FOR A PREPEND AND IT ATE THE ANCHOR,
  THEN THE NEXT ENTRY VANISHED SILENTLY (2026-11-07).** Several
  entries above were inserted with a local
  `edit(path, old, new)` that calls `s.replace(old, new)`. Used
  with the anchor as `old` and the entry as `new`, that DELETES
  the anchor - so the heading line `PRINT THE ITEMS, NOT THE
  COUNT` disappeared, leaving a stray fragment, and the
  following entry's `assert s.count(anchor) == 1` then failed
  and threw its whole entry away. **The PHI-6 and COMP-1 entries
  were lost that way and committed and pushed without anybody
  noticing**, because that edit ran in a command I had put in
  the BACKGROUND, so its traceback was never read. Three rules
  this file already holds, all three broken at once: write after
  each edit, read the whole output rather than a filtered view,
  and never background a step whose failure you need to see. The
  helper is a PREPEND now, it asserts the anchor survives the
  write, and the repair was verified by grepping for each
  entry's own title rather than by trusting a print.

- **THE RING-ADJACENCY CEILING DOES NOT EXIST, AND THE RECORDED
  SENTENCE WAS COMPARING TWO DIFFERENT EXPERIMENTS (2026-11-07,
  RING-1).** This file carried, for weeks and with a 2-4 week
  estimate beside it, "the sweeps deliver ~0.78 adjacent
  correlation where a near-identity needs ~0.97 - the
  ring-adjacency ceiling, the genuine research item". The 0.775
  is what `smoke_cycles.ring()` produces AT ITS OWN SKILL, which
  it hardcodes as `0.75 - 0.005*i`. The 0.97 is what a
  near-identity NEEDS. Swept across skill, eight seeds a cell:
      declared skill   adjacent |r| min-mean-max   sweeps OFF
          0.75         0.789  0.794  0.798           0.591
          0.90         0.875  0.878  0.880           0.641
          0.97         0.948  0.949  0.950           0.703
          0.99         0.979  0.980  0.981           0.760
  **The sweeps deliver the skill they were GIVEN, to within
  about 0.02, with no cap anywhere.** The ranges are almost
  degenerate, so this is not a seed effect.
- **THE SHAPE CRITERION AGREES, AND IT NEEDED ITS OWN FLOOR.**
  The binned-conditional-mean gap at skill 0.97 is 0.059 sd
  against a 0.35 DEPARTS bar - and that figure is measured
  BETWEEN TWO INDEPENDENT DRAWS OF THE SAME BLUEPRINT, which is
  the floor any comparison can reach. Without that reference a
  0.06 reads as a small loss rather than as noise, which is how
  a non-problem acquires a work item.
- **AND THE REAL EXTRACT SETTLES IT.** `run_deliverable`'s
  blueprint declares `visit_start_date <-> visit_end_date` at
  skill **0.9999 in BOTH directions** - a genuine near-identity
  inside a cycle, the exact shape the ceiling was supposed to
  describe. Source rho **1.0000** on 55,428 rows; generated rho
  **1.0000** on 51,418. A fixture result alone would have been
  worth less; this is the thing itself.
- **WHAT IS STILL OPEN IS NARROWER, AND THE OLD SYMPTOM CANNOT
  BE RE-MEASURED.** The original complaint was
  `map_cuff <- map_cuff_bmdi` drifting up to 0.68 sd in shape on
  a September cohort. Those columns are NOT in the current
  blueprint - the pressure variants appear there only as weak
  PARENTS at skill 0.07 to 0.36, with no near-identity claim
  between them at all - so the symptom belongs to a cohort that
  no longer exists in measurable form. Because the sampler
  renders whatever skill it is handed, a future shape gap on a
  near-identical pair comes from DISCOVERY's skill assignment or
  from the curve, NOT from the sweeps. That is the re-aimed
  question, and it is a different file.
- **THE FINDING IS PINNED AS CHECKS, NOT PROSE, AND THE
  FAIL-FIRST IS A GENUINE CAP.** `smoke_cycles` now asserts the
  fixture reproduces 0.775 at its own skill (a correction that
  cannot reproduce what it corrects has not earned it), that
  skill 0.97 delivers over 0.90, that the delivered value sits
  within 0.05 of the declared skill at both ends, and that the
  gap between the two ends is real rather than zero. Clamping
  the sampler's own `share` at 0.78 - the mechanism the record
  described - turns THREE of them red.
- **AND THE FIRST FAIL-FIRST FOUND NOTHING BECAUSE I MUTATED
  THE WRONG FUNCTION.** `generate.py` reads
  `skill_out_of_sample` in SEVEN places; I clamped the one in
  `_informative_sets` and the sweep uses `_apply_numeric`. A
  mutation that does not apply looks exactly like a check that
  cannot fail, and this file already says so - the fix is to
  locate the read by FUNCTION before mutating, which takes one
  grep.
- **AND I DESTROYED THE NEW CHECKS WITH `git checkout --` HOURS
  AFTER RECORDING THAT EXACT TRAP.** The rule above is "restore
  from a COPY of the working state, never from a commit, when
  the working state is what you are trying to keep", written
  this same weekend. I then reverted `smoke_cycles.py` to HEAD
  to undo a mutation and took the uncommitted RING-1 checks with
  it. Recovered by re-applying from the same script, verified by
  asserting the text is IN the file, and a copy kept in /tmp
  BEFORE the next mutation cycle. Knowing a rule and having a
  habit are different things; the habit is to copy first.

- **DIFFERENTIAL PRIVACY IS AN OPTION ON THE NEURAL PATH NOW,
  AND THE EPSILON COVERS THE WEIGHTS AND NOT THE RELEASE
  (2026-11-07, DP-1).** `COMPETITIVE.md` named DP as the largest
  single gap against the other tools. `--dp-epsilon` trains the
  autoencoder with DP-SGD - per-example gradient clipping,
  Gaussian noise on the clipped sum, Poisson subsampling - and
  an RDP accountant for the subsampled Gaussian mechanism. OFF
  by default, because it costs something and that is somebody
  else's decision to make.
- **THE ACCOUNTANT IS VERIFIED AGAINST ITSELF FOUR WAYS, because
  a wrong epsilon is a privacy claim somebody will act on.** It
  cannot be checked against a reference implementation here, so:
  at q -> 1 the subsampled bound must REDUCE to the full-batch
  closed form alpha/(2 sigma^2), and does to 1.9e-09; the
  conversion at sigma=1, T=1, delta=1e-5 must hit the
  hand-solved optimum alpha* = 1 + sqrt(2 ln(1/delta)),
  eps 5.2985, and reads 5.3026 on the integer grid; it must sit
  just ABOVE the classical single-shot Gaussian bound
  sqrt(2 ln(1.25/delta))/sigma and CONVERGE to it as sigma grows
  (0.6077 against 0.6056 at sigma 8) - below would mean claiming
  more privacy than the textbook mechanism gives; and the bound
  must amplify under subsampling and be monotone in q and sigma.
  Three mutations - dropping the binomial (1-q) factor, dropping
  the Renyi exponent, a factor of two in the Gaussian RDP - each
  turn FIVE checks red.
- **SUBSAMPLING IS NOT AN OPTIMISATION, IT IS WHY DP TRAINING IS
  POSSIBLE.** Full-batch gradient descent gives
  eps ~ 4.8 sqrt(T)/sigma, so the 800 steps this architecture
  trains for would need a noise multiplier near 136 to reach
  eps 1 - which destroys training. Measured: 800 steps at
  sigma 4 cost eps 61.5 full-batch against 0.355 subsampled at
  q=0.01. The subsampled bound is the closed form for INTEGER
  alpha, used in preference to a tighter bound needing numerical
  integration, because a loose epsilon is honest and a wrong one
  is not.
- **THE TRAINER IS A CONFOUND AND THE CONTROL ARM IS THE SAME
  NETWORK WITH THE NOISE OFF.** `MLPRegressor` exposes no
  per-example gradients, so DP-SGD cannot be applied to the
  shipping trainer at all - it had to be hand-rolled, which
  means a cost measured against the shipping engine would mix
  the price of privacy with the price of my optimiser. Plain SGD
  reached holdout MSE 0.065 where MLPRegressor reached 0.011;
  with ADAM the trainer's own cost falls to 1.10x, small enough
  that the control arm is fair. DP's price is measured against
  THAT arm, and the 1.10x is reported separately.
- **THE PRICE, 51,200 rows, 3 seeds, against the control:**
      epsilon   sigma    holdout MSE    x control
         8.00    1.54         0.0226         1.16
         3.00    3.41         0.0346         1.78
         1.00    9.57         0.0977         5.02
         0.50   18.86         0.1745         8.96
      control (DP off)        0.0195         1.00
  An MSE near 1.0 would mean the autoencoder learned NOTHING on
  a standardised frame, so even eps 0.5 is still learning.
- **AND THE COST IS DOMINATED BY ROW COUNT, NOT BY EPSILON.** At
  a fixed epsilon the signal-to-noise of a DP step goes as
  n/(sqrt(T) sqrt(params)) - independent of the sampling rate,
  LINEAR in the number of rows. Measured at eps 1.0: **18.8x on
  3,200 rows, 17.8x on 12,800, 4.6x on 51,200.** DP is expensive
  on a small cohort and affordable at extract scale, which is
  the opposite of how a feature list reads, and a ratio quoted
  without its cohort size means nothing. The operating point was
  swept too: steps matter and the sampling rate barely does
  (q=0.05/T=1500 reads 4.55x against q=0.50/T=400 at 6.78x).
- **END TO END ON THE REAL ENGINE, eps 3 on a 4,200-row
  fixture**: reconstruction error 0.00462 -> 0.02736, train /
  holdout ratio 1.03 -> 0.97 so both arms generalise, nn-ratio
  0.90 -> 0.85, and the gate's close, sign and INVERTED
  criteria UNMOVED at 6/6, 6/6, 0 on both. **Two arms that agree
  are one arm**: a five-column fixture cannot separate them, so
  that is a limit of the fixture and NOT evidence that DP is
  free - the reconstruction number is the sensitive one and it
  moved by 5.9x.
- **THE TRAINER IS SKLEARN-SHAPED, WHICH IS WHY NOTHING ELSE
  CHANGED.** The engine's `_fwd` reads `net.coefs_` and
  `net.intercepts_` and applies ReLU to every layer but the
  last - exactly this network's layout and rule - so exposing
  those two names makes it a DROP-IN and `_encode`, `_decode`
  and `generate` are untouched. Verified by running the engine's
  own forward path against the module's and asserting they
  agree.
- **AND PER-EXAMPLE NORMS ARE COMPUTED WITHOUT PER-EXAMPLE
  GRADIENTS.** Example i's weight gradient for a layer is the
  outer product of its input activation and its output delta,
  whose squared Frobenius norm is ||a_i||^2 ||g_i||^2 - so the
  total is a sum of products of row norms and no
  batch-by-in-by-out tensor is ever built, which is what makes
  clipping affordable at this width. Checked against the
  explicit outer-product construction to 2.8e-14, because an
  error there would silently change the sensitivity the whole
  accounting assumes.
- **WHAT IS STILL NOT DP, NAMED: five stages.** The k-screen and
  vocabulary selection (which levels get modelled comes from
  patient counts), the published numeric bounds (k-anonymous,
  not DP), the latent mixture (fitted on codes of real rows),
  the visit-count and set-size grids, and the identity tiers
  (detected from real rows). `UNACCOUNTED_STAGES` names them,
  `scope_note` prints them, and the run REFUSES to describe its
  output as differentially private. A half-accounted epsilon is
  the free-text-redactor fault wearing a privacy guarantee, and
  false confidence about privacy is the most expensive kind
  there is.

- **A PATIENT'S COURSE OVER TIME WAS ABSENT BY CONSTRUCTION,
  NOT DEGRADED (2026-11-07, DYN-1).** `COMPETITIVE.md` names
  sequential modelling as the second-largest gap, and the gate
  has read `dynamics tracked 17/87` for months. The mechanism
  is one line: the neural engine draws a patient's visit
  deviations iid and then PERMUTES them, so consecutive visits
  are independent by construction. Measured with the project's
  own instrument on a fixture carrying a planted within-patient
  AR(1):
      col          source within   generated within
      systolic           0.468            0.000
      severity           0.456            0.000
      diastolic          0.379            0.001
  `within` is lag-1 with the between-patient share divided out -
  the patient's course rather than their level. The LEVEL
  survives (icc 0.808 -> 0.733); the course is exactly zero.
- **THE REARRANGEMENT ARM WAS BUILT FIRST, MEASURED, AND
  REMOVED - AND ITS GUIDING ARGUMENT WAS FALSE.** Reordering
  each patient's drawn deviation rows to match an AR(1) score
  recovered within 0.000 -> 0.032 against a source 0.468, about
  5% of the gap. And the reasoning behind it - a PERMUTATION
  cannot move a marginal, the instrument the repairs and dials
  rest on - DOES NOT HOLD HERE: downstream steps rank-map rows
  onto published grids, so reordering rows changes which row
  gets which value and the output multiset genuinely moves.
  Found by measuring the multisets rather than by trusting the
  argument; they differed. Removed rather than left as a second
  option.
- **AND THE FIRST COMPARISON WAS A RESEED WEARING A FLAG.** The
  AR targets drew from the MAIN rng, so every downstream draw -
  the k-blur, the set token picks, the identity tiers - shifted,
  and the with-and-without arms were effectively different
  seeds. Caught because the output multisets differed when a
  permutation of deviations cannot change them. A dedicated
  stream fixes it, and the lesson is the recorded one: a change
  that consumes randomness is not a change you can A/B against
  the same seed.
- **THE AR WALK IN LATENT SPACE RECOVERS IT, AND THE MEASURED
  LATENT RHO IS THE WRONG CALIBRATION.** Walking the deviations
  as d_t = rho d_{t-1} + sqrt(1-rho^2) e_t preserves any
  stationary covariance exactly, so the latent covariance that
  drives the column correlations is untouched. Swept, mean over
  three columns against a source within of 0.434 and icc 0.700:
      rho    mean icc   mean within   icc drift
      0.00      0.700         0.000      +0.000
      0.50      0.747         0.261      +0.047
      0.70      0.800         0.388      +0.100
      0.85      0.850         0.392      +0.151
      0.95      0.925         0.314      +0.226
  The measured latent rho IS 0.95 - and applying it directly is
  strictly worse than 0.85 on BOTH axes. A high rho turns the
  deviation into part of the centre, which is the recorded
  `dev_scale` lesson in reverse: a knob that sounds right does
  the opposite through the decoder's nonlinearity.
- **ONE CRITERION FIXED, ONE BROKEN, AND THE BROKEN ONE IS A
  NAMEABLE COLUMN CLASS.** At rho 0.7, per column against the
  gate's 0.15 bar: lag-1 systolic 0.742 -> 0.831 (FIXED - it was
  failing), severity 0.676 -> 0.805, diastolic 0.651 -> 0.789,
  all better. But `visit_number` - a within-patient COUNTER -
  went icc 0.002 -> 0.365 against a source 0.000, BROKEN,
  because persisting the deviations makes a counter look
  patient-specific. The gate's `dynamics tracked` therefore
  reads 1/4 on every arm: **three arms that agree are one arm**,
  and the reason is one column class rather than the mechanism
  failing.
- **SO IT SHIPS OFF BY DEFAULT WITH THE PRICE PINNED.** A flag
  that silently harmed one column class would be worse than no
  flag. `--trajectory ar:RHO`, with the sweep in its own help
  text, the counter harm asserted by a check so it cannot be
  quietly forgotten, and the default asserted to be OFF. Three
  mutations bite: removing the AR walk (3 red), turning it on by
  default (1), dropping the variance-preserving factor (1).
- **AND THE RHO WANTS SOLVING, WHICH IS THE NAMED RESIDUE.** At
  a FIXED rho the column-level landing is dataset-dependent: on
  a 600-patient fixture rho 0.7 moved every column toward its
  source, and on a 320-patient one it OVERSHOOTS severity
  (source 0.848, off 0.797, on 0.938) while still improving
  systolic. The within-patient recovery is robust; the
  column-level calibration is not. The fix is to SOLVE rho
  against the source's own lag-1 rather than take the latent
  value or a swept constant - the same lesson as the intercept
  being solved rather than centred. A check asserts the
  overshoot HAPPENS so the gap cannot be forgotten, and the
  first version of that check asserted the opposite and went
  red, which is how the gap was found.

- **PRINT THE ITEMS, NOT THE COUNT - AN EMPTY LIST CANNOT BE
  MISREAD WHERE A ZERO CAN (2026-11-06).** Polling a net on the
  data machine, the failure COUNT came back as `8` on one read
  and as `®` on another when the true value was ZERO. This file
  already records zero reading as six and as eight; the fix is
  not a better digit but a different question. Asking the
  machine to PRINT each failing line between two literal
  markers gives `BEGIN-FAIL-LIST` immediately followed by
  `END-FAIL-LIST`, and nothing between two markers is
  unambiguous however the OCR mangles characters. Use it for
  any zero that decides something: a count must be read, a
  bracketed emptiness only has to be seen. The success marker
  is still the verdict - this is how to read the absence of
  failures without trusting a digit.

- **A RAW GITHUB FETCH SERVED THE PRE-PUSH FILE AND REPORTED
  SUCCESS (2026-10-23).** `curl` of
  `raw.githubusercontent.com/OWNER/REPO/main/scripts/x.py` seconds
  after a push returned the file as it was BEFORE the commit -
  byte-for-byte the previous version - while curl exited 0 and the
  file's timestamp updated. Caught only by comparing the fetched
  size against the local file: 169,434 against 170,825, and 169,434
  was exactly the pre-edit copy. The consequence would have been
  precise and invisible: the attack battery had just been given a
  `--max-tokens` lever so it would stop certifying a different
  configuration from the one that ships, and running it against the
  cached file would have re-run the very bug the lever was built to
  fix, then reported the clean number as clearance.
  FETCH BY COMMIT SHA, never by branch - `/<sha>/path` cannot be
  served stale - and verify the byte size against the local file
  before running anything that reads it. A branch URL is a cache
  with a promise attached.

- **THE BRIDGE SAID "REFUSED" AND MY OWN FILTER THREW THE LINE
  AWAY (2026-10-23).** Every read that looked like a stale screen
  today was an EXPLICIT refusal - `HANDS refused: focus guard
  refused: cmd prompt not the foreground window (wrong window or
  selection mode)` - and the read pipeline was
  `... | sed -n '/--- output ---/,$p'`, which drops everything ABOVE
  that marker. The refusal prints above it. So the bridge reported
  the fault clearly, in words, every single time, and the pipeline
  discarded exactly the line that mattered while keeping the
  previously-rendered screen underneath it. Hours went into
  diagnosing a bridge that was never broken: first blamed on
  `enabled=False` from `status` (a red herring - the flag is
  per-process and in-memory and `status` never sets it), then on a
  frozen video feed (disproved by hashing the screen across a known
  repaint). THE RULE: grep for `HANDS refused` FIRST, and only then
  slice to the output section. And snapshots LAG - two probes came
  back together on a third read, so a single read that lacks its
  marker is not yet evidence of anything; settle, then re-probe.
  The guard itself is the good news: with a browser and Teams in the
  foreground it refused rather than typing shell commands into a
  colleague's chat window.
- **THE PRECAUTION THAT SURVIVED THE MISDIAGNOSIS.** Echoing a
  unique marker in a SEPARATE call and requiring it back is still
  the only trustworthy proof a read is live, because the failure
  mode - previously rendered content returned as a result - is the
  same whatever causes it, and at one point that content was ANOTHER
  PROJECT's model-comparison output, which a log read would have
  reported as this run's fidelity. A marker inside the command under
  test is not enough: the bridge prints the command above the
  output, so the echo matches its own text.
- **(SUPERSEDED, KEPT FOR THE SHAPE OF THE ERROR) A STUCK CONSOLE
  RETURNS THE LAST SCREEN AND CALLS IT A RESULT, WHILE EVERY HEALTH
  SIGNAL READS GREEN.** A console
  window left in selection/mark mode freezes its output AND swallows
  typed input, so `hands run` typed nothing, raised nothing, and
  returned whatever was already rendered - every command "succeeded"
  and every number read back was stale. `status` reported
  `reachable=True keyboard_online=True busy=False awake=True`
  throughout, and `key enter` returned `ok` while producing no new
  prompt line, which is impossible at a live prompt and was the first
  real evidence. The screen at that moment held ANOTHER PROJECT's
  model-comparison output, so the next read would have reported that
  project's AUCs as this run's fidelity.
  Two diagnoses were wrong before the right one: `enabled=False` from
  `status` is a RED HERRING (the flag is per-process and in-memory,
  and `status` never sets it), and the feed was not frozen either -
  hashing the screen before and after opening a new window proved it
  repaints. Recovery is `key esc` then `term new`; esc alone did not
  clear it.
  THE RULE: before trusting any read from that bridge, echo a unique
  marker in a SEPARATE call and require it back in the output. A
  marker inside the command under test is not enough - the bridge
  prints the command above the output, so the echo matches its own
  text, which is the same mistake as the poll loop below. And a
  screen hash across a known-repainting action is the test for
  liveness, because the health fields cannot see a frozen window.
  Same class as the read-but-not-used flag and the silent `getattr`
  no-op: a capability that fails by returning something plausible is
  worse than one that crashes.
- **A WAITER THAT GREPS FOR ITS OWN MARKER MATCHES THE COMMAND, NOT
  THE OUTPUT.** A poll loop built as `hands run 'if exist X (echo
  RUNDONE-END)'` piped into `grep -q RUNDONE-END` exited on its
  FIRST iteration and reported a multi-hour run complete while it
  was still training - because the bridge ECHOES the command it ran
  above the output, so the marker was present in the echo whatever
  the machine's state. A check that cannot fail, in the tooling
  again, and it cost a false "complete" that was then acted on. Two
  rules: slice to the output section before matching anything, and
  pick a marker that CANNOT appear in the command - `tasklist`'s
  "No tasks are running" is produced only by the machine, where any
  `echo MARKER` is produced by me. The run's own artifacts are then
  verified separately, because a crashed process also exits.
- **ONE DRIVER AT A TIME, AND A BACKGROUND POLLER IS A SECOND
  DRIVER.** A poll loop left running in the background while reads
  were issued in the foreground interleaved keystrokes into one
  console and returned nonsense (`python. exe € r+ =o: \ a`). The
  machine is also shared: another session was working in a different
  repository, my commands executed from ITS shell and inherited an
  environment variable it had set, and the window-management keys
  used to work around the garbled output (`win+m` minimizes
  everything) would have disrupted that work. Drive from one place,
  never use a global window key, and echo the settings a command
  actually received - an inherited variable is the unannounced
  setting this file already has a rule about.

- **THE SCREEN IS NOT A NUMERIC CHANNEL, AND EVERY NUMBER READ OFF
  IT SO FAR WAS UNVERIFIED (2026-10-23).** The data machine is read
  by screen OCR, and DIGITS DO NOT SURVIVE IT. Tested directly
  rather than assumed: `0123456789` came back as `9123456789` - the
  leading zero became a nine - while in the SAME read all ten
  spelled words (zero one two ... nine) came back perfectly.
  Elsewhere in the same session `0.37` read as `@.37`, `0.30` as
  `0.3@`, and the column name `4544_3` as `4544 3`. A number quoted
  off an OCR pass is therefore an ARTIFACT, not a measurement, and
  this file's own rule is never to state a number as measured unless
  it was. Re-reading does not help: the same pixels give the same
  wrong answer, so two agreeing reads are one read - the recorded
  two-arms-that-agree rule, in a new place. `scripts/sayn.py` pipes
  any output through and repeats every numeric literal in words
  (`close 77/94` -> `[77 = seven seven | 94 = nine four]`). THE
  WORDS ARE THE CHANNEL; the digits beside them are a convenience
  that may arrive corrupted, and when the two disagree the words are
  right. Identifiers are left alone on purpose - spelling `4544_3`
  would bury the numbers that matter in noise. Anything whose
  numbers decide something crosses in words.
- **A WAITER THAT GREPS FOR ITS OWN MARKER MATCHES THE COMMAND, NOT
  THE OUTPUT.** A poll loop built as `hands run 'if exist X (echo
  RUNDONE-END)'` piped into `grep -q RUNDONE-END` exited on its
  FIRST iteration and reported a multi-hour run complete while it
  was still training - because the bridge ECHOES the command it ran
  above the output, so the marker was present in the echo whatever
  the machine's state. A check that cannot fail, in the tooling
  again, and it cost a false "complete" that was then acted on. Two
  rules: slice to the output section before matching anything, and
  pick a marker that CANNOT appear in the command - `tasklist`'s
  "No tasks are running" is produced only by the machine, where any
  `echo MARKER` is produced by me. The run's own artifacts are then
  verified separately, because a crashed process also exits.

- **A SHELL VARIABLE IS ONE MORE THING THAT CAN ARRIVE EMPTY, AND
  ON THAT TERMINAL IT DOES (2026-09-28).** WINDOWS.md built the
  zipball URL out of `set SYNTHKIT_REPO=...` plus
  `%SYNTHKIT_REPO%`, with an echo line to catch the failure - and
  the operator reports it has NEVER once worked: what reaches
  curl is a URL addressing nothing. The echo was a guard against
  a mechanism that should not have been there. Type the owner and
  repo out IN FULL, in the curl line and in both `for /d`
  patterns; `%USERPROFILE%` is the only variable that has ever
  survived there. Same lesson as the wrapped paste and the flag
  that never reached the program: on that terminal, prefer the
  literal string to anything that has to be substituted first.
  (The handle itself is banned from tracked files, so the tracked
  copy carries `OWNER/REPO` and the operator's local RUNBOX
  carries the real one.)

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

- **THE RUN THREW THE KEY AWAY, SO THE ARCHITECTURE WAS
  IMPOSSIBLE RATHER THAN MERELY UNBUILT (2026-10-30).** The
  release dials have existed for weeks and the run wrote ONLY
  the perturbed file - the full-fidelity frame was discarded the
  moment the dials were applied. You cannot read a vendor's
  answer against a reference you did not keep, so the whole
  key/released design had no first file. One run now writes
  three things: `key.csv` (the full-fidelity k-anonymous
  synthetic, kept, never shared), `generated.csv` (the released
  file) and `cipher.json` (which key row each released row took
  each perturbed value from).
- **AND IT WORKS BECAUSE EVERY DIAL IS A PERMUTATION.**
  Re-pointing sorts a column against an invented score;
  suppression shuffles it outright. Both only MOVE values
  between rows, so each perturbed column holds the identical
  multiset in both files and the change is EXACTLY INVERTIBLE -
  verified value-for-value through the disk, and watched red
  against two mutations (an overwriting cipher instead of a
  composing one, and suppression not recording its
  arrangement). That is the sentence to say to a governance
  board: the released file is a faithful rearrangement, not
  damaged data. `dialback.py verify` asserts it rather than
  describing it, and moving ONE value in the released file turns
  it red naming the column.
  Suppression now draws `permutation()` rather than `shuffle()`
  - verified `v[perm] == shuffle(v)` on one RNG state first, so
  the record exists at no cost in output bytes.
- **DILUTION IS NOT IN THE CIPHER AND THE CIPHER SAYS SO.**
  `--release-strength` blends values toward the mean instead of
  moving them, so a diluted column has no permutation. A record
  that silently covered three of four controls would be the
  read-but-not-used fault in the one file a reviewer trusts.
- **DIAL THE OUTCOME, NOT THE COVARIATES - MEASURED, AND IT IS
  THE WHOLE DESIGN RULE.** On a fixture whose outcome genuinely
  reads the dialed column, skill on our own truth over held-out
  rows:
      dial the OUTCOME     0.970 -> -1.657   denied 2.627
                           recovery EXACT
      dial a COVARIATE     0.970 ->  0.892   denied 0.078
                           recovery 0.102 sd (0.482 ignored)
  **34x the protection AND exact recovery.** A model routes
  around one corrupted covariate through the others, so dialing
  a covariate buys almost nothing; a corrupted LABEL leaves a
  trainer nothing to learn. And because a vendor model that
  merely SCORES our rows reads only the covariates - which stay
  faithful - the number it returns about released row p is
  already the number for key row p. Nothing needs undoing.
- **AND THE CIPHER DOES NOT UN-DISTORT A ROW-LEVEL PREDICTION.
  Said out loud, because the obvious reading of "we hold the
  key" is wrong.** A vendor's score for released row p is
  `f(X_p, C_s)` where `C_s` is some OTHER row's value of the
  perturbed column. We want `f(X_p, C_p)`. No arrangement of the
  INPUTS produces that number - only `f` can, because `f` was
  never evaluated there. Three routes, and they are not equally
  sound: (A) they return a model or let us call it, we run it on
  `key.csv`, exact, and the cipher is not needed for it at all -
  **this is the route to put in the contract**; (B) they return
  only per-row scores, a surrogate is fitted to (released rows
  -> scores) and read on the key rows, approximate, error
  MEASURED; (C) they return an aggregate, and the cipher gives
  the exact before/after of every statistic we moved.
- **THE WEAK CASE IS MEASURED, NOT INFERRED FROM THE PLAN.** The
  first cut branched on "no perturbed predictor" to report that
  a setting protects nothing - but in a flat table every column
  predicts every other, so that branch was near-unreachable dead
  code, and the two smoke checks aimed at it both failed on a
  fixture that could not reach it. The verdict now comes off the
  two protection numbers, and a target nobody can predict is
  REFUSED with its own skill quoted - a recovery ratio between
  two noise numbers reads as a result.
- **AND THE MUTATION THAT FOUND NOTHING NAMED THE REAL GAP.**
  Deleting the `pair` field the bench's own run reports broke
  ZERO checks: `smoke_gui` asserted the page READS
  `result.pair`, and nothing asserted the server SENDS it. That
  is this file's own non-fail-first - a mutation that changes a
  reported field no assertion depends on. Pinned at the decision
  site in `smoke_dialback`, which already does real runs, and
  the same mutation now fails two checks. The wrapped-heading
  trap also bit again in the same hour: `<b>Dial the OUTCOME,
  not the\n    covariates.</b>` cannot be grepped as a phrase
  however correctly the page renders.

- **THE REAL-EXTRACT DIALBACK RUN CAME BACK, THE THREE ROUTE
  VERDICTS HELD, AND IT FOUND TWO REPORTING DEFECTS
  (2026-10-30, run_dial1, 800 patients, 56,065 rows, k 10,
  dial `condition_count~span_days=-0.60`).** The prediction
  scored 3 of 3 on the verdicts and was RIGHT to refuse the
  magnitudes.
      verify            rows 56,065 both sides; span_days
                        inverse EXACT, same multiset; 43
                        untouched columns, 0 changed -> VERIFIED
      dialed OUTCOME    key-trained  +0.433
      (span_days)       released     -0.111
                        DENIED       +0.544   recovery EXACT
      dialed PREDICTOR  key-trained  +0.368
      (condition_count) released     +0.392
                        DENIED       -0.025   protects nothing
  **A vendor who trains on the released file goes from +0.433
  to NEGATIVE skill - worse than predicting the mean - while a
  vendor who merely SCORES our rows returns the key answer
  exactly, because every predictor it reads is untouched.** The
  fixture's +2.63 did NOT transfer, exactly as the prediction
  refused to claim: 44 colinear columns give a model far more
  to route around. +0.544 is the real number and it is still
  decisive. The three headline figures were re-read through a
  SECOND channel, spelled by the machine - "PLUS zero POINT
  four three three", "MINUS zero POINT one one one", "PLUS zero
  POINT five four four" - because the digits channel off that
  screen is not trustworthy and these go in the record.
- **A DIAL CAN ASK FOR A CORRELATION THE PAIR CANNOT REPRESENT,
  AND NOTHING SAID SO.** The run asked -0.60 and landed -0.29,
  which reads as a sampler miss. It is not. `span_days` holds
  **FIVE distinct values over 56,065 rows**, and against
  `condition_count`'s 22 the most negative Spearman ANY
  rearrangement can reach is **-0.395** - measured by sorting
  one against the other, not assumed. The ask did not exist;
  the dial reached 73% of the floor that does. The run now
  prints the reachable band with the distinct-value count that
  bounds it, the band is RECORDED beside requested and
  achieved, and `verify` marks an ask that fell outside it.
  Every fixture here had continuous columns, so none could
  reach this - the cap-no-fixture-reaches lesson, in a new
  module, found by the first real run rather than by a check.
- **AND ROUTE B'S VERDICT CONFLATED "TOO DISTORTED TO RECOVER"
  WITH "NOTHING TO RECOVER".** On the dialed-predictor target
  the returned score sat **0.007 sd** from truth and the
  surrogate read 0.110, so the tool printed ROUTE B DOES NOT
  WORK and blamed a corrupted predictor - when the truth is the
  model barely reads the column we moved. Same observable,
  opposite conclusion, the third instance in this module.
  Separated now, and the fix NAMES the agreement: a dial that
  denies nothing (-0.025) and a return that needs no recovery
  (0.007) are the SAME fact, and two numbers agreeing is this
  file's signal rather than a coincidence.
- **THE REAL EXTRACT DOCUMENTS ITS OWN CEILING NOW, AND THE
  HAND MEASUREMENT AND THE IMPLEMENTATION AGREE TO THREE
  DECIMALS (2026-11-02, run_dial2).** The same dial on the
  current build:
      re-pointed condition_count~span_days
        asked -0.60   landed -0.29
        reachable band [-0.395, +0.445], bounded by the 5
        distinct value(s) the dialed column holds
        THE ASK WAS OUTSIDE THAT BAND
  The floor was predicted at **-0.395**, measured here by hand
  by sorting span_days against condition_count in the key file
  before the fix was written; the engine's own solve reports
  **-0.395**. Two independent routes to one number is what
  makes it a measurement rather than an implementation detail.
  NOT predicted and worth keeping: the band is ASYMMETRIC
  (+0.445 up, -0.395 down) - a tie structure does not bound
  both directions equally, and only the floor had been checked.
- **ROUTE B RAN END TO END ON THE REAL RUN, AND ITS NUMBER IS A
  BEST CASE RATHER THAN A TYPICAL ONE (2026-11-02).** `recover`
  had only ever been exercised on its REFUSAL path - the whole
  value-extraction half of the architecture was untested on
  anything. Against run_dial2, with a stand-in vendor scoring
  the released rows and the truth computed on the key rows:
      route B, surrogate                0.041 sd
      the return taken untouched        0.641 sd
  Fifteen times better than ignoring the distortion, on 56,065
  rows. **SAID PLAINLY: the stand-in is LINEAR** (0.4*count +
  0.3*span + noise), which a gradient-boosting surrogate fits
  almost exactly, so 0.041 is the ceiling of this route and not
  what a real vendor model would give. What it does establish
  is that the chain WORKS at extract scale and that route B is
  worth running rather than taking a return as-is. The honest
  order is still route A - get the model, run it on key.csv,
  exact - and `rehearse` remains the thing that says which
  route a given dial setting leaves you.

- **ROUTE B IS A RANGE, AND THE SWEEP REFUTED THE PRIOR IT WAS
  BUILT ON (2026-11-02).** The 0.041 sd first reported for this
  route came from a LINEAR stand-in vendor, quoted as if it
  described the route. `route_b_sweep.py` runs five shapes at
  one dial setting. On the fixture:
      shape          dependence  untouched  route B
      linear              0.439      0.660    0.085
      curved              0.567      0.796    0.079
      interaction         0.845      1.147    0.250
      threshold           0.775      1.143    0.030
      xor-like            1.000      2.112    0.031
  **The guess was that nonlinear would be harder and linear
  would be the ceiling. Both are wrong.** The surrogate is a
  gradient-boosted TREE: a threshold or a sign flip is native
  to it and reads EASY, while a smooth slope is approximated
  by a staircase and reads WORSE. The hard case is a PRODUCT
  across the perturbed and the untouched column - 0.250,
  three times linear - because the surrogate must learn a
  surface from outputs that only ever sampled it along the
  wrong diagonal. The module says it was refuted rather than
  quietly rewriting itself, and a check pins that sentence.
  DEPENDENCE is reported beside the error because a model that
  barely reads the moved column needs no recovery - a
  different fact from a surrogate that recovers well, and the
  two are easy to confuse in one number.
- **GOAL 4'S NAMED NEXT STEP IS BUILT: EVERY DIAL AGAINST
  EVERY NEIGHBOURING PROPERTY (2026-11-02, `dial_matrix.py`).**
  The roadmap asked for "the full per-class verification pass,
  each dial asserting its neighbouring properties", and this
  file says why: shift and scale fought each other for as long
  as both existed because each check set ONE dial and neither
  looked at the other's property; one edge to zero collapsed an
  innocent neighbour 0.555 -> 0.007. The fault is a CROSS
  PRODUCT - dial times property - so the test is one. Each of
  the eight classes runs alone on one seed, and the same
  property vector is measured on every column plus the pair
  correlations between them.
- **A SHIFT ON A PARENT SILENTLY DELETES ITS RELATIONSHIPS, AND
  THE DIAL REPORTS A CLEAN HIT THROUGHOUT.** The matrix's first
  real catch. A child is drawn from a curve published over the
  parent's OBSERVED range and `np.interp` holds flat outside
  it - the plateau already recorded for the deck. Swept on a
  two-column fixture, parent~child Spearman, three seeds:
      shift 0.0 sd  0.968     shift 2.0 sd  0.832
      shift 1.0 sd  0.965     shift 3.0 sd  0.313
                              shift 5.0 sd -0.027
  Not a rounding error at the far end - the whole relationship.
  For the cipher this is the worst shape of fault: dial a shift
  to protect one column and unknowingly delete every pattern it
  takes part in, with the record saying nothing. `dials.verify`
  now reports the share of generated values landing outside the
  published grid and NAMES the children that will have
  flattened, beside the hit. A safe shift reports nothing,
  which is the negative case that makes it worth reading.
- **THREE OF THE MATRIX'S FIRST FOUR FINDINGS WERE ITS OWN
  FAULTS, and this file's rule caught all three.** When a
  measurement script returns a clean negative, suspect the
  script. `patients.count` read NO-OP because the matrix PASSED
  n_patients EXPLICITLY on every arm - overriding the dial it
  was testing. `persistence` read NO-OP because the fixture had
  no within-patient serial correlation at all, so the dial had
  nowhere to land. And `persistence` is a MULTIPLIER on the
  column's own persistence, not a target: asked 0.4 of a base
  0.388 it lands 0.157, exact, and grading it as a target
  called it broken. The matrix records each dial's SEMANTICS
  now - absolute, multiplier, target, move - which is
  documentation that did not exist anywhere.
- **AND THE DISTINCTIONS THE FIRST CUT COLLAPSED.** A NO-OP
  reads CLEAN if only neighbours are checked, so the TARGET is
  asserted too - and the branch has a case that reaches it
  (`scale=1.0`, a setting that changes nothing by definition),
  because every real dial works and an unreachable branch
  inside the tool built to catch unreachable behaviour is a
  joke at its own expense. MOVING is not LANDING: the edge
  dial asked x0.40 and delivered x0.86 while reading CLEAN,
  and the SHORT verdict now names the live REVERSE edge
  carrying the pair. PROPAGATION is not a LEAK: flattening a
  parent's persistence flattens its child's, which is the
  relationship working, so it is reported as propagation -
  read from the blueprint's graph, not assumed - while a
  bystander column the graph never touches is the control that
  makes every other row mean something.
- **THE REAL-BLUEPRINT RUN REFUSED, AND THE REFUSAL WAS THE
  POINT.** Pointed at the extract's own blueprint - 142
  columns, 88 relationships, 40 bystanders against the
  fixture's 4 and 1 - the matrix reported value None, sparse
  None, bystander None and STOPPED: "without a control the
  matrix cannot tell an effect from noise". The guard, written
  for the fixture, is what prevented an all-clean table on real
  data that would have meant nothing. The PICKER was wrong, not
  the blueprint: it read `marginal["kind"]` where the field is
  `marginal["type"]`, so it matched none of 133 quantiles
  columns. Proven by reverting - wrong key gives
  (None, None, None, None), right key picks its columns.
- **THE PATTERN DIAL WROTE NO CIPHER, AND I SHIPPED THAT
  WHILE WRITING PROSE ABOUT INVERTIBILITY (2026-11-03).**
  `--apply` wrote the dialled file and nothing else: a
  rearrangement invertible IN PRINCIPLE and recorded nowhere,
  so undoable by nobody. That is this file's own
  capability-wired-to-nothing fault, committed in the same
  module whose docstring explains why permutations matter.
  `repair` now fills a `record` dict with {child: taken_from}
  and the dial writes it into cipher.json in the same shape
  the engine's own cipher uses, so `dialback verify` reads it
  with no special case - and the entry carries requested
  beside achieved and the band, which makes it an answer key
  rather than a diff.
- **AND THE INVERSE READ EXACTLY FALSE, BECAUSE THE DIAL
  RE-RENDERED THE VALUES.** `repair` assigns NUMERIC values
  back into the frame, so a file written from it carries
  22.289999999999999 where the generated file holds 22.29 -
  the released file was not a byte-level rearrangement of the
  generated one, and the cipher's round trip failed on
  formatting rather than on arrangement. The arrangement IS
  the whole content of the change, so it is applied to the
  UNTOUCHED ORIGINAL STRINGS. Inverse exact afterwards, and
  no column without a cipher entry changed at all.

- **A PATTERN IS ADDRESSED BY ID NOW, AND THE ID IS CONTENT-
  DERIVED (2026-11-03).** `pattern_dials.py RUNDIR --dial
  ID=TARGET` rearranges one named pattern as an output
  permutation, reports the reachable band, names the
  collateral, and writes nothing without `--apply`.
  POSITIONAL IDS WOULD HAVE BEEN A TRAP: `p07` meaning the
  seventh row shifts the moment discovery finds one more
  pattern, so a user who dialed p07 yesterday dials something
  else today with no error and no sign. The id hashes the
  pattern's CONTENT.
- **AND SORTING THE COLUMNS GAVE TWO DIRECTIONS ONE ID.**
  `x <- y` and `y <- x` are different patterns; the first cut
  sorted the column names and handed both the same
  identifier, which is the exact hazard content-derived ids
  exist to prevent. Only a SYMMETRIC kind is sorted now, and
  a check asserts the two directions differ.
- **AND A CHECK GUARDED BY `if` VANISHES INSTEAD OF FAILING.**
  The refusal test was written `if und:` - so a mutation that
  made every pattern dialable removed `und`, SKIPPED the
  check, and the mutation "found nothing". The fixture is
  asserted to contain an undialable pattern FIRST, then the
  refusal is tested unconditionally. A conditional check is a
  check that can disappear.
- **AND THE MUTATION HAD TO HIT THE RIGHT GATE.** Widening
  `DIALABLE` still found nothing, because `dial_pair` ALSO
  requires two columns - so a single-column pattern stayed
  undialable whatever the kind list said. Two gates, one
  mutated: the fail-first was testing a path that was not
  load-bearing. Mutate the gate the behaviour actually passes
  through.
- **AND A PERMUTATION CLAIM ACROSS A FILE WRITE COMPARES TWO
  FILES.** The written-file check compared the CSV against the
  IN-MEMORY frame and read `array_equal False / allclose
  True` - a round trip moves the last bits. `dialback verify`
  reads BOTH sides from disk for exactly this reason, and the
  claim being made is about two files.

- **PER-PATTERN DIALS RUN ON THE REPAIR ENGINE, AND THE
  CONSTRAINT IS GEOMETRIC (2026-11-03).** The product goal is
  a user choosing, per pattern, how to distort it. The engine
  for that already existed: `repair_inversions` rearranges a
  child JOINTLY against all its partners, which is what stops
  a single-pattern dial randomising everything else. It now
  takes a `targets` map overriding one pair's desired rank
  correlation - the same solve, so the two cannot disagree,
  and a permutation either way so the marginal cannot move and
  the cipher can undo it.
- **AND "CHANGE THIS PATTERN, LEAVE THE REST ALONE" IS
  SOMETIMES IMPOSSIBLE.** Where one driver feeds two children
  their couplings to it are LINKED, and no arrangement
  satisfies an arbitrary value for one while pinning the
  other. Measured on exactly that shape:
      target   holding others     letting them move
      -0.50    a~b -0.209         a~b -0.490
      -0.80    a~b -0.297         a~b -0.806
  Letting them move LANDS the ask. Holding them lands about
  40% of it AND drags the held pair anyway, because the solve
  splits the difference and neither target is met. So
  `hold_others` is a real choice the operator makes and the
  report says what it cost - one default hiding this would be
  a dial that silently under-delivers, which is the fault the
  module exists to refuse.
- **AND A PATTERN THE OPERATOR NAMED IS NEVER DROPPED FOR
  BEING WEAK.** The coupling floor screens NOISE partners out
  of the joint solve; a pair somebody dialed is not noise, and
  dropping it for a low source correlation is exactly how a
  dial comes to report success and do nothing.
- **`repair_inversions` HAD NO CHECKS AT ALL** - a script that
  rewrites a generated file with `--apply`, used on the
  deliverable, with nothing automated exercising it. Covered
  now, including the restore path it was written for. And its
  first fixture was VACUOUS: `sample(frac=1.0)` permutes every
  column TOGETHER, so every correlation survived and the
  restore check passed by restoring something never lost. The
  suite's own fixture-validation check caught it; shuffling
  the CHILD alone is what destroys the coupling.

- **CONFIRMED ON THE REAL EXTRACT: span_days HAS THREE
  PARENTS (2026-11-03).** `visit_type, procedure_count,
  procedure_quantity` - and it is also PARENT of
  procedure_count and procedure_quantity, so it sits in a
  cycle. That is the reproduced mechanism, on the data the
  dials would actually be used against: a column generated
  from its parents' curves never reads a shift written into
  its marginal, and which side of a cycle gets drawn from its
  marginal is decided by trim order, which is why the
  behaviour cannot be predicted from the blueprint.
- **SO THE VALUE DIALS ARE INERT ON THE REAL DATA, AND THE
  RELEASE CONTROLS ARE NOT.** Both halves have to be said
  together. `shift` and `scale` land in the blueprint, report
  no problem, and move nothing, because every well-covered
  numeric there is a relationship child. `--poison` and
  `--suppress` are PERMUTATIONS APPLIED TO THE OUTPUT after
  every tier has run - verified landing and inverting exactly
  on 56,065 real rows. The governance controls are sound; a
  different set of dials is not, and nobody should quote
  "dials over every pattern" without that distinction.
- **AND THE PATTERN INVENTORY SAYS THE GAP IS THE SAME FROM
  BOTH ENDS (`pattern_dials.py`).** The product goal is a
  user seeing every pattern found and choosing per pattern how
  to distort it. The inventory lists every pattern the
  blueprint carries with the dial that reaches it, the LAYER
  that dial acts on, and whether it is invertible. On any
  blueprint today: **NOT ONE discovered pattern is reachable
  by an invertible dial.** The invertible dials exist and are
  attached to a pair an operator names BY HAND; every dial
  attached to a discovered pattern is a blueprint dial, the
  layer measured inert. Effect curves and identities have no
  dial at all. The unreachable kinds are NAMED rather than
  summed, because the list is the roadmap and the count is an
  advert.
- **AND WHAT IS ABSENT MAY BE UNDECIDABLE RATHER THAN ABSENT.**
  Three-way structure needs roughly 3,200 patients on this
  project's own power curve and the extract has 800, so
  "no three-way patterns found" is a statement about the
  cohort. The inventory says so where the counts are read,
  because a user asked to dial "all patterns" deserves to know
  which ones nobody could decide.

- **THE REAL BLUEPRINT'S FIRST READING: EVERY VALUE DIAL ON
  span_days READ NO-OP, AND THE MECHANISM REPRODUCES ON A
  FIXTURE (2026-11-03).** 1 of 10 dials moved what it named
  and nothing else. Three hypotheses died before the right
  one: span_days is NOT derived (derived_from is None); "a
  relationship child bypasses its marginal" is refuted BY THE
  REPO'S OWN FIXTURE, where x is a child and its dial works;
  and constraint repair is off by default and that constraint
  is `<=`, not the `==` class that always fires. Reproduced
  locally on four columns, shift 40:
      c    no parents                 +40.00  applied
      b    child, marginal-drawn      +40.00  applied
      a    child, curve-drawn           0.00  NO-OP
      out  child, curve-drawn           0.00  NO-OP
  **A column whose parents survive the cycle trim is generated
  from THEIR CURVES, not from its own marginal, so a shift
  written into the marginal is never read.** The dial reaches
  the blueprint, `apply` reports no problem, and the number
  does not move. WHICH children are curve-drawn is decided by
  cycle-breaking and trim order, so an operator cannot read it
  off the blueprint - and on the real extract EVERY
  well-covered numeric is a relationship child, which is why
  the value dials are inert there.
- **AND THE MISS REPORTED A BARE NUMBER.** `verify` already
  called it - hit False, achieved 0.70 against a requested 40 -
  and said NOTHING about why, which is a verdict a reader
  cannot act on. It now names the relationship and the parents
  the column is generated from, and a dial that DID apply
  carries no such excuse, so the explanation marks a real
  cause rather than printing on every row. Both watched red.
  **This is the fault class the cipher cannot tolerate**: a
  governance control that reports success and does nothing.

- **A MODE WITH NO TEST WAS DEBUGGED SIX TIMES ON THE MACHINE
  THAT HOLDS THE DATA, AND EVERY FAULT WAS LOCAL.** The
  matrix's --blueprint path was written and never exercised
  against anything but the real extract. Six round trips at
  roughly ten minutes each:
      1  marginal key read as `kind`, the field is `type`
      2  picked a `__has__` indicator, which is dropped before
         generated.csv is written
      3  no control at all - "related to nothing" finds nothing
         on a dense graph
      4  picked a date column, whose marginal is quantiles like
         any numeric
      5  the date and scaffolding filters applied to the column
         picks and NOT to the edge pick
      6  a banner that indexed a source frame blueprint mode
         does not have
  Five of the six were reproducible on a fixture of six
  columns and two relationships. The suite stayed GREEN through
  all of them, because every check pointed at the fixture path.
  **A second mode is a second thing to test, not a flag.** The
  blueprint path now has an end-to-end check that builds a
  two-component blueprint, picks all four roles and runs the
  whole matrix - and the fixture needs TWO components because
  the control is more than two hops from the value column, so
  with a single relationship there is nowhere far enough to
  stand.
- **AND THE MUTATION THAT PROVED IT LOOKED LIKE IT FOUND
  NOTHING, TWICE OVER.** Restoring the broken banner was
  greppped for `^FAIL` and came back empty - because a CRASH
  emits a traceback, not a FAIL line, which this file already
  records - and the `exit=0` beside it belonged to the
  pipeline's `tail`, not the suite, which this file also
  already records. Both traps in one command. Read the whole
  output of a mutation, not a filtered view of it.

- **AND THE SECOND REAL-BLUEPRINT ATTEMPT DIED ON
  SCAFFOLDING.** With the key fixed the picker chose
  `conditions__has__ICD10-CM!E11.9` and the run crashed with a
  KeyError - because a `__has__` indicator lives in the
  BLUEPRINT, where the search needs it, and is DROPPED before
  generated.csv is written. This file already records that
  class twice (the constraints section counted scaffolding as
  the operator's data; the gate's absolute stop judged a
  scaffolding inversion), and it already says the predicate
  lives in `sets.py` and both halves must CALL it rather than
  each decide. Called now.
- **AND THE FIXTURE COULD NOT REACH THE FAULT UNTIL ITS REAL
  COLUMNS WERE IN THE GRAPH.** Removing the exclusion broke
  NOTHING at first: scaffolding qualifies as a BYSTANDER -
  quantiles-typed, fully covered, related to nothing - but the
  fixture's own unrelated numerics came first in iteration
  order and were picked safely. On the real blueprint every
  real numeric was in the graph, so the picker fell through to
  the scaffolding. The fixture now relates its two numerics,
  which leaves scaffolding as the only bystander candidate,
  and the same mutation then picks `conditions__n` and goes
  red. A mutation that changes nothing is not proof the check
  works; it is proof the fixture cannot reach it.

- **THAT IS THE THIRD WRONG-KEY FAULT IN ONE DAY**, after
  jointcheck's `skill_real` and verify_frame's
  `requested_in_sds`. The same shape every time: writing the
  field name that sounds right instead of reading the
  contract. The same fix applies and is now applied here - the
  suite asserts the key against a LIVE BUILT BLUEPRINT, so a
  rename breaks the checks rather than a run on the machine
  that holds the extract. When a module reads another module's
  dict, assert the key names live; do not remember them.

- **AND A RED CHECK WAS WALKED PAST.** The suite went red when
  the sparse cases were restored, the `1 FAILED` was printed in
  an output I read, and I continued into the shift
  investigation without fixing it. The NET caught it. The check
  was right - at 2 sd the shift genuinely leaks - and the fix
  is that the matrix carries BOTH sizes, clean at 0.5 sd and
  leaking 0.68 of the pair at 3 sd, with both pinned so the
  detection cannot regress into a clean row.

- **ROUTE A HAD NO TOOLING, AND WHAT WAS MISSING WAS THE
  ANSWER (2026-11-02, `vendor_exam.py`).** `dialback` names
  three routes and recommends A - we hold the model, or they
  score our file - because it has no approximation error at
  all. It was the only route that existed solely as ADVICE IN
  A DOCSTRING. And what it lacked was never the scoring: a
  score is gradeable only against a truth somebody knows, and
  `semisynth.plant_frame` has been able to supply one since it
  was written while being called by NOTHING BUT ITS OWN SMOKE
  TEST - a capability wired to nothing, this file's own
  recurring fault, sitting next to the route that needed it.
  Now joined: plant a known outcome on key.csv, copy it to the
  released file and RE-POINT it there, and grade in
  report_card's vocabulary so the two halves say the same
  words - CEILING (a model that SAW our truth), FLOOR (a model
  trained on the RELEASED file), VENDOR between them.
  Measured: a vendor who trained on what we sent grades 0.483
  against a floor of 0.484 and a ceiling of 0.649.
- **ABOVE THE CEILING IS A BROKEN EXAM, NOT A GOOD SCORE.**
  The first cut printed "they are at the ceiling" for a vendor
  sitting **0.344 ABOVE it** - a verdict contradicting the
  number two lines above, which is this file's named fault.
  Nothing legitimately beats a model that saw the answer, so
  there are two explanations and both invalidate the result:
  the planted truth reached them, or their scores were
  computed in-sample. It is loud now, names both causes, and
  the praise branch is bounded on BOTH sides.
- **AND AN EXAM WHOSE CEILING SITS UNDER ITS FLOOR IS REFUSED,
  NOT WARNED ABOUT.** On a thin fixture it read ceiling 0.502
  against floor 0.539 - the best achievable score BELOW what a
  re-pointed file gives you, which is not a weak exam but a
  meaningless one, both numbers being AUC noise around chance
  with their order a coin flip. The first cut warned and wrote
  the answer key anyway, producing an artifact that looks
  usable and grades nobody. The checks caught it. The
  condition CANNOT BE FORCED, only found: it is reached by
  planting on a pure-noise column and pinning a seed that
  lands the wrong way round, and the pin is named as such.
- **AND THE WRONG-KEY FAULT HAPPENED AGAIN, in the same week
  it was recorded for jointcheck.** `verify_frame` returns
  `effects` / `requested_in_sds` / `recovered_in_sds`; the
  first cut read `recovered` / `requested` / `recovered` and
  printed an EMPTY TABLE on a healthy exam. The suite now
  asserts those names against a LIVE `verify_frame` call, so a
  rename breaks the checks rather than somebody's report -
  the same fix that entry prescribes.

- **THE SWEEP RAN ON THE REAL EXTRACT, AND ONE MEASUREMENT
  EXPLAINS BOTH ITS RESULTS (2026-11-02, run_dial2, 56,065
  rows).**
      shape          dependence  untouched  route B
      linear              0.602      0.305    0.003
      curved              0.924      0.428    0.010
      interaction         0.944      0.315    0.036
      threshold           0.718      0.368    0.007
      xor-like            1.000      0.264    0.000
  `interaction` is still the hard case, so the fixture's
  ORDERING transfers. The MAGNITUDES do not: every shape reads
  three to seven times better than on the fixture, and the
  reason is one number - **86.9% of rows hold the identical
  span_days value in both files**. Five distinct values
  dominated by one, so a permutation mostly swaps a value for
  itself. That single fact explains the impossible ask AND the
  easy recovery: the dial moved 13% of one column, so there
  was little to protect and little to recover. The sweep
  prints the moved share and warns when a coarse column is
  flattering it. Quote these numbers only with that share
  beside them.
- **AND "COARSE" WAS NOT THE SHAPE THE CHECK NEEDED - "COARSE
  AND DOMINATED BY ONE LEVEL" WAS.** The fixture's tied column
  was five even levels cut from a span and moved 63% of rows,
  so it reached the impossible-ask fault and NOT the
  flattered-recovery one, and the new check went red. Rebuilt
  with 93% of rows on a single level it reproduces the real
  86.9% almost exactly. This file's own rule - a fixture must
  contain the thing the check is about - and the wrong
  property was planted because "coarse" sounded sufficient.

- **AND A MUTATION THAT DOES NOT APPLY LOOKS EXACTLY LIKE A
  CHECK THAT CANNOT FAIL.** The first fail-first on the
  worst-shape check found nothing and was nearly reported as a
  vacuous check. The CHECK was fine; my mutation had two
  replacements and I asserted count==1 on the first only, so
  the second silently no-opped and the text never changed.
  This file's oldest editing rule - assert count==1 before
  EVERY replacement - applies to the mutation as much as to
  the fix, and skipping it inverts the conclusion.

- **AND A `cd && (...) || echo ABSENT` CHAIN IS NOT AN EXISTENCE
  TEST.** The first check of that run reported RUNDIR-ABSENT and
  I briefly believed the run had produced nothing - while its
  own log ended "wrote key.csv and cipher.json". Two readings
  disagreeing is this file's alarm and it was right: the
  directory was there, and the malformed cmd chain was the
  liar. `if exist X (...) else (...)` gave the truth
  immediately. A negative from a compound chain is worth less
  than a positive from a simple one.

- **THE ONE MISTAKE THE WHOLE SPLIT EXISTS TO PREVENT HAD NO
  GUARD AT ALL (2026-11-02).** Two machines, real extracts on
  one and fixtures on the other, and every rule in this file
  about writing code without access to the data assumes a real
  extract never reaches this repository. The only thing
  enforcing that was nobody having done it. `smoke_no_personal`
  scans tracked files for a HANDLE or a LAPTOP MODEL; a
  55,428-row clinical extract committed into `data/` contains
  neither and would have passed every scan in the suite. The
  remote going public made that materially worse the same day.
- **TWO GUARDS, BECAUSE THEY CATCH DIFFERENT MISTAKES.** The
  INVENTORY pins every tracked record-bearing file by name, so
  a new `data/real_extract.csv` fails until somebody adds it
  deliberately - which is the moment to ask what it is. The
  CEILING catches a fixture OVERWRITTEN by the real thing under
  its OWN name, which the inventory cannot see. Measured, not
  guessed: the largest tracked fixture is measurement.csv at
  5,653 rows against a real extract's 55,428, so 20,000 sits
  above every fixture and far below anything that could be an
  extract. A third check asserts every name in the inventory is
  still tracked, because a pinned list that matches nothing
  passes trivially - this file's own worst-kind-of-clean.
  Neither guard can tell real data from fake by reading it,
  which is not a thing a checker can do, and neither pretends
  to: they make an unexpected file LOUD. Both watched red - a
  planted new file names itself, and 55,428 rows planted under
  `person.csv` reads "at 55429 rows".
- **AND I RAISED A MIMIC DUA ALARM THAT WAS WRONG.** Seeing
  `data/mimic_omop/` with record-level person, measurement and
  drug rows, I read it as redistributed MIMIC-IV under a
  PhysioNet DUA and told the operator to take the repo private.
  It is generated: `scripts/make_mimic_omop.py` builds it to
  reproduce the shape AND THE MESS of the real extract, from
  schemas transcribed off photographs of the data machine's
  screen because the data has no route out. "OMOP-mimic" means
  OMOP-MIMICKING. Grep for a generator before reading a
  directory name as provenance - the correction cost less than
  the alarm, but the alarm was mine to check first.

- **THE BAND REACHED THE AUDIT RECORD AND THE BENCH, AND
  FAIL-FIRST ON A FAIL-FAST SUITE PROVES ONLY THE FIRST CHECK
  (2026-11-02).** Three gaps closed after the real run: verify
  printed a BARE `[-0.39, +0.39]` while the cipher already
  carried `distinct_child` - a field written and never read,
  this file's own named fault, now named in the line; a cipher
  predating the band printed NOTHING, which reads as "no
  ceiling here" rather than "cannot be said from this record",
  the empty-versus-missing distinction one more time; and the
  BENCH still showed asked beside landed alone, so the one
  surface an operator actually reads was the last place still
  implying a gap is a miss.
- **AND THE MUTATION THAT SHOULD HAVE FAILED THREE CHECKS
  FAILED ONE, WHICH IS A PROPERTY OF THE SUITE, NOT THE FIX.**
  `smoke_gui.check()` calls `sys.exit(1)` on the first failure,
  so reverting the page printed ONE `FAIL:` line, no summary
  line, and an exit code belonging to the pipeline's `tail`.
  Two of the three checks were therefore UNVERIFIED while
  looking verified - the shape this file already records as a
  crashed suite emitting no FAIL line. On a fail-fast suite
  each check must be mutated ALONE; done, and all three go red
  on their own target.

- **AND A ZERO FAIL COUNT MADE THE RUNNER REPORT A GREEN NET
  AS FAILED.** The net printed `ALL GREEN: 82 suites, 2475
  checks` and `exit=0`, and the harness still called the job
  failed - because the command chain ENDED in `grep -c "^FAIL"`,
  which exits 1 when the count is zero. This file already says
  the verdict must come from the SUCCESS marker rather than the
  absence of a failure marker; here the absence of failures was
  itself the failure signal. End a chain on the thing you want
  the status of, or read the marker.

- **WHAT THE MACHINE COST, AND WHAT IS LEFT UNTIDY.** No git,
  no `gh`, `SYNTHKIT_TOKEN` unset, so the documented zipball
  fetch was closed and typing a secret is not done. The two
  changed files went over as a gzip+base64 payload typed in 46
  chunks and verified by md5 AND by the gzip CRC. That leaves
  `dev\synthkit` a HYBRID - `abf66447` plus two files from
  `fd66ece` - which is fine for this measurement and is NOT a
  state to leave: a tree matching no commit is the stale-serve
  fault waiting to happen. Re-fetch properly with the token.
  Also: the read pipeline sliced to `--- output ---` and
  `HANDS refused` prints ABOVE it, so several "empty" reads
  were explicit refusals being discarded - this file already
  says grep the refusal FIRST, and it still cost three probes.

- **PREDICTION FOR THE FIRST REAL-EXTRACT DIALBACK RUN, ON
  RECORD BEFORE IT RUNS (2026-10-30).** 800-patient
  tidy_visits, k 10, one dial: `condition_count~span_days
  =-0.60`, so `span_days` - the column a vendor would be asked
  to model - is the OUTCOME and every predictor stays faithful.
  (1) `verify` PASSES: inverse exact and identical multiset on
  span_days, every other column byte-identical. This is
  structural, so a FAIL would mean the engine moved values
  rather than rearranging them, which is a real finding rather
  than a tuning miss. (2) `rehearse --target span_days` reports
  the target PERTURBED with no perturbed predictor, so recovery
  reads EXACT and the protection number is MEASURED. (3)
  `rehearse --target condition_count` takes the route-B branch,
  because span_days is then a perturbed predictor.
  **NOT CLAIMED: that the fixture's +2.63 denial transfers.**
  That fixture had one dominant driver and clean noise; the
  extract has 44 colinear columns where a model routes around a
  single lost column far more easily, so the absolute denial
  should be SMALLER and could be small enough to matter. NOT
  CLAIMED either: that route B's 0.102 sd recovery transfers -
  it was measured on a five-column frame. The three ROUTE
  VERDICTS are the claim; the magnitudes are the measurement.

- **THE TOKEN CAP IS A MEMORY BUDGET, NOT A PRIVACY RULE - AND IT
  WAS COSTING FIDELITY (2026-10-23).** On an 800-patient tidy
  fixture, `conditions` held 5,191 tokens, **4,710 cleared the k
  floor, and SIXTY were modeled** - while the encoded frame sat at
  437 dimensions against the 4,326 that once died allocating
  1.5 GiB. Ten times the headroom, and 4,650 already-publishable
  tokens discarded. Paired on three seeds at a flat cap of 400,
  `close` moved **+3.1 / +6.2 / +3.1** with INVERTED still 0 and
  direction 32/32, and the output vocabulary went 60 tokens -> 400
  with set sizes closing on the source (active_drugs 1.40 -> 1.61
  per row against 2.25 held). `--max-tokens` / `--max-levels` are
  the lever, and the run PRINTS the budget it used.
- **AND THE SINGLE SEED SAID THE OPPOSITE.** The first reading of
  that change was close 87.5% -> 81.2%, a 6.3-point LOSS, on one
  seed. The paired sweep says +4.2 mean, all three positive. A
  single-seed measurement of a generation change is worth nothing -
  this file has said so about row counts, recall and effects, and
  it now says it about the cap, having nearly rejected a real
  improvement on one run.
- **THE SOLVED WIDTH BUDGET WAS BUILT, MEASURED AND WITHDRAWN.** A
  flat 400 is unsafe on a wider frame, so the cap was made a budget:
  take the largest cap whose total width fits 2,400 dimensions. It
  chose 280 and measured NOTHING - +0.0 / +3.1 / -3.1, mean 0.0,
  signs MIXED, against the flat 400's all-positive. The cause is a
  MODELLING error, not a tuning one: predicting width from
  k-screened levels on the FULL frame overestimates what the encoder
  builds by about 2.5x at every cap (973 against a measured 437 at
  cap 60; 3,222 against 1,308 at cap 400), because the fit sees an
  85% patient split and several counted columns never become one-hot
  at all. Tuning the constant until it happened to pick 400 on this
  one fixture would be the measured-at-one-scale fault wearing a
  budget. The DEFAULT is unchanged; the fix when it is picked up is
  to take true per-column width from the PLAN the encoder already
  builds rather than predicting it.

- **THE GATE CAN BE READ BACKWARDS, AND THE THIN COHORT WINS
  (2026-10-23).** The same engine on an 800- and a 100-patient cut
  of the same generator: the SMALLER cohort scored HIGHER on close,
  **91.4% against 87.5%**, while publishing **64% of its data
  against 87%** and carrying **12 distinct condition codes against
  60**. Fidelity is scored only on the pairs that cleared the
  privacy screen, so a thinner source grades a smaller and stronger
  surviving set - it is faithful about LESS. A comparison that
  prints fidelity without coverage beside it argues for the worse
  dataset. `compare_sources.py` puts coverage ABOVE fidelity and
  states the trap in words where a reader will meet it, and a check
  asserts that ordering.
- **PUBLISHABLE MASS IS THE SIGNAL; COLUMN COUNT HID IT.** Columns
  keeping anything moved 74 -> 71 between those two cuts - three -
  while the share of data landing on a publishable value fell 87% ->
  64%. A column keeps its name on one surviving level out of two
  hundred, so the count flatters.
- **AND A SET COLUMN IS NOT A CATEGORY OF ITS COMBINATION STRINGS.**
  The first census judged sets that way: `t03;t09;t14` and `t03;t14`
  share no "level", every combination is near-unique, and the column
  read 0% publishable however ordinary its TOKENS were - two sources
  that differ completely both came out at exactly 0.000, which is
  how the fault was caught. Sets are screened by TOKEN through
  `sets.vocabulary`, the same module the engine uses, so the census
  and the generator cannot disagree about what is publishable. The
  real pair then separated further: 87% against 64%.

- **THE TOKEN CAP WAS A CEILING ON WHAT A PATIENT COULD HOLD, AND
  LIFTING IT EXPOSED ONE MECHANISM UNDER THREE SEPARATE MISSES
  (2026-10-23).** On a 100-patient cohort whose patients each hold
  about 82 distinct drugs and up to 170, a run at the old cap of 60
  emitted a high bound of EXACTLY 60.00 with the spread collapsed to
  a third of source - a hard ceiling, not a model failure: the
  column had 202 drug tokens clearing k and published 60, so no
  generated patient could hold a 61st. At cap 400 the ceiling is
  gone, 60.00 -> 155.20 against a source 170.30, and the spread
  recovers from 34% of source to 83%.
- **THE PREDICTION WAS WRITTEN FIRST AND SCORED 2 OF 4.** The
  ceiling clause held exactly. The low-end clause held - it was
  predicted NOT to improve, and it worsened. The clause saying
  encounters per patient would barely move FAILED: mean -20%,
  median -33%, high -26%, so the encoded WIDTH couples into the
  visit-count draw through the latent, which was not expected and
  is the finding that failed clause bought. And the column was
  MISNAMED in the first reading - the ceiling was on
  `distinct active_drugs per patient`, read off a block whose
  header had not been checked. Read the header, never the line
  offset.
- **AND THE RESIDUAL IS ONE SHAPE, NOT THREE FAULTS: EVERY
  PER-PATIENT DISTRIBUTION COMES OUT TOO NARROW.** Measured on the
  same run across three independent quantities, source -> generated:
      conditions/patient    spread 90.4 -> 52.8    low 13.2 -> 58.7    high 322.0 -> 235.7
      active_drugs/patient  spread 47.5 -> 39.2    low 10.2 -> 29.1    high 170.3 -> 155.2
      encounters/patient    spread 204.8 -> 139.4  low 17.2 -> 31.9    high 707.2 -> 490.6
  Every spread is short, every low bound too HIGH and every high
  bound too LOW - the light patients are not light enough and the
  heavy ones not heavy enough. Three misses with one signature are
  ONE mechanism, and reporting them as three fidelity gaps would
  send somebody chasing each in turn. The suspect is the mixture
  over PATIENT CENTRES under-dispersing in latent space, which
  would compress every per-patient quantity at once; the test is to
  compare the spread of generated patient centres against the
  training centres DIRECTLY rather than inferring it from the
  decoded columns. NOT the recorded dev_scale sweep, which moved the
  between-patient SHARE and is about within-patient deviation.

- **AN ALL-EMPTY COLUMN COMES OUT AS THE STRING "nan", AND IT COST A
  REAL RUN ITS COVERAGE CRITERION (2026-10-23, OPEN).** A 100-patient
  cohort failed the gate at coverage 34/42 - and the eight failures
  were columns ENTIRELY ABSENT in the source: year_of_birth, gender,
  race, ethnicity and four more, coverage 0.0 in, 1.0 out.
  Reproduced through the CLI on a CSV on disk, because the reader is
  part of the suspect path and an in-memory frame handed straight to
  fit cannot catch it:
      SOURCE  year_of_birth  present on   0.0% of rows
      GEN     year_of_birth  present on 100.0% of rows, values ['nan']
  MECHANISM: an all-empty column has no present value to classify, so
  it falls through the numeric/categorical gate into the CATEGORICAL
  branch, where `astype(str)` turns NaN into the literal level
  "nan". That is a NON-EMPTY STRING, so the fidelity comparison
  counts the column present on every row. Third trigger of the
  wrong-type-branch family this file already records twice -
  span_days collapsing at low cardinality, height folding to
  `__other__` at low presence - and this one is NO PRESENT VALUES AT
  ALL. The 800-patient extract passes coverage 42/42 and could never
  have shown it; it takes a cohort thin enough that whole columns
  arrive empty.
  THE FIX, not yet made: at the gate in `LatentGen.fit`, a column
  with `raw_notna == 0` is neither numeric nor categorical - it is
  EMPTY, and empty is a fact about the data, the same
  empty-versus-missing distinction already drawn for set columns and
  for `id_column`.
  FIXED the same day. A column with `raw_notna == 0` is kept OUT of
  the plan entirely - no encoded dimension, no decoded channel, no
  draw, because there is nothing to learn from a column nobody
  recorded - and written back EMPTY at assembly, as "" rather than
  NaN, which is what the reader folds to missing and what a person
  opening the file expects in a field nobody filled. Measured
  through the CLI on a CSV on disk: year_of_birth 100% present with
  value 'nan' -> 0.0% present, with the column COUNT unchanged, so
  nothing was dropped from the file. The check was pinned as a
  characterization test expecting `{"nan"}` while the fault was open
  - so the shared net stayed green for the team rather than failing
  CI and RUN_CHECKS.bat for a fault that blocked nobody - and arming
  it was the promised one-line flip to `{""}`. Watched RED against
  the removed branch, which matters more than usual here: three
  checks written the same day could not fail, so green-after-the-fix
  proves nothing on its own.
- **AND THE FIRST VERSION OF THAT CHECK COULD NOT FAIL - THE THIRD
  SUCH CHECK IN ONE DAY.** It asked whether the generator INVENTED
  content, with a presence test that listed "nan" as absent - so a
  column full of "nan" read as 0% present and the check passed,
  correctly answering a question nobody needed asked. It went green
  against code the probe had just watched emit the fault. Caught only
  by running the probe and the check side by side and noticing they
  disagreed - the internal-contradiction rule, applied to the
  tooling. The pattern across all three: THE ASSERTION WAS WRITTEN
  AROUND THE HYPOTHESISED MECHANISM RATHER THAN AROUND THE
  OBSERVABLE THE FAULT ACTUALLY CHANGES. The waiter asserted on my
  own marker instead of the machine's output; the token check
  asserted a bound both rules cleared; this one asserted "no
  invented content" instead of "the cell is empty". Assert the
  observable.
- **AND NOT EVERY FAILURE ON A THIN COHORT IS THE k RULE'S DOING.**
  Building the comparison that argues 100 patients is too few at
  k=10, three candidate numbers had to be thrown OUT of the
  argument: the four empty demographic columns are an EXTRACTION
  difference (absent from the pull, nothing to do with cohort size),
  and within-patient persistence falls on BOTH cohorts - 0.4952 ->
  0.1568 on the extract against 0.3939 -> 0.0869 on the thin one -
  because this engine is ROW-LEVEL and models no patient trajectory,
  so it would lose persistence on a million patients. What DOES
  carry the argument, measured: publishable mass 89% against 80%,
  publishable condition codes 898 against 290, and spread retained
  on the same column 61% against 25% (sd 18.44 -> 11.24 against
  19.55 -> 4.82). An argument built from the wrong numbers loses the
  right ones with it.

- **A FIXTURE THAT SHARES THE RENDERER'S WRONG KEYS IS NOT A
  TEST, AND FORTY CHECKS SAW NOTHING (2026-10-26).** Section 2 of
  the relationships page - "are the same things doing the
  driving?" - rendered TWELVE ROWS OF EM-DASHES on the real
  extract. The renderer read `r2_real` / `r2_synth` /
  `drivers_match`; `jointcheck.predictability` emits `skill_real`
  / `skill_syn` / `driver_agreement`. Every cell came back None,
  and a document about to go to a stakeholder looked finished and
  carried no values at all. THE CHECKS PASSED BECAUSE THE FIXTURE
  HELD THE SAME WRONG KEYS: the two halves agreed with each other
  and both disagreed with the only thing that matters, the
  function actually being called. This file already says
  consistent wrongness is invisible from inside - said there
  about a reader that folded empty into missing, and the folded
  gate that passed on its own folded numbers. Same fault, new
  module. The key names are asserted against a LIVE
  `predictability` call now, so a rename in jointcheck breaks the
  suite rather than somebody's report. Measured after: 15 columns
  scored, and `span_days` reads 40% driver agreement against a
  0.80 band - the SHAP attribution drift this file records,
  visible for the first time as a number on a page.
- **AND THE SCORE GATE NEEDED A CEILING AS WELL AS A FLOOR.** The
  page refused to score a target nobody could predict - a real
  0.04 against a synthetic 0.03 is "75% kept" and means nothing.
  It happily scored targets ANYBODY can: `year_of_birth` and
  `age_at_visit` read 1.000 -> 0.999, "100%, holds", sitting at
  the top as the most reassuring rows on the page. `age_at_visit`
  IS the visit year minus `year_of_birth` - arithmetic the
  generator COPIES by design, which will read 100% on every run
  forever. It is evidence about the identity tier, not about
  relationships. Reported apart, with the ORDERING asserted so it
  cannot drift back above the real results. On the real extract
  both fired and were flagged; the 100-patient cohort has none,
  so the section is conditional and correctly absent there.
- **TSTR IS THE ONE FIDELITY NUMBER THE THIN-COHORT TRAP CANNOT
  REVERSE.** This file records that a 100-patient cut scored
  HIGHER on `close` than the 800 while publishing 64% of its data
  against 87% - fidelity is scored only on the pairs that cleared
  the privacy screen, so a thinner source grades a smaller and
  stronger surviving set. Train-on-synthetic-test-on-real scores
  both arms on the SAME held-out real rows, so it cannot be
  flattered that way. Measured on both cohorts, same engine, same
  build: `condition_count` 14% against 1%, `active_drug_count`
  70% against 53%, and `procedure_count` is not predictable in
  the thin cohort AT ALL (below the 0.10 floor) where the extract
  predicts it at 0.639. Six scored targets against four. The one
  reversal, `visit_id` 25% against 55%, is an identifier whose
  real-data skill is 0.22 - quote it with the skill beside it or
  it reads as a win.
- **EVERY DEGRADED TARGET WAS A COUNT COLUMN.**
  `condition_count`, `procedure_count`, `active_drug_count` - all
  set-size partners, copied from set lengths. The multivariate
  loss sits exactly where the k rule bites hardest - the
  measured cohort pair is 898 publishable condition codes
  against 290. Not a sampler defect; the privacy screen,
  surfacing in the one place no per-column number looks.
  (CORRECTED: this entry first read "365 against 1,050", which
  paired the extract's own token count with a number belonging
  to a FIXTURE - 50 of 365 clear k, nothing to do with either
  cohort. Caught while writing a script somebody was going to
  read out loud. "Never state a number as measured unless you
  actually ran it" is the oldest rule in this file and it was
  broken IN this file, in an entry written the same day; a
  number that is merely nearby is not a measurement.)

- **RAISING THE TOKEN CAP PAST 400 BUYS NOTHING, MEASURED PAIRED
  ACROSS THREE SEEDS (2026-10-27).** Nine cells on the real
  800-patient extract, caps 400/800/1600 x seeds 0/1/2, 57 min per
  cell, TSTR retention as the metric. Per-seed verdicts for
  400 -> 800:
      procedure_count     UP / EQ / EQ    mean  6 -> 8
      active_drug_count   UP / DN / EQ    mean 62 -> 63
      condition_count     UP / DN / UP    mean 13 -> 16
  **No target rises on all three seeds and two have a DOWN seed.**
  Mean movements are one to three points, which is inside the noise
  this file already documents for generation changes.
- **AND SEED ZERO SAID THE OPPOSITE, WHICH IS THE WHOLE POINT.**
  The first cell of each cap read every target UP and was reported
  as a direction needing confirmation rather than a result. It was
  noise. This file records the same shape twice already - one seed
  reading -6.3 on this exact lever against a paired +4.2, and a
  centre_ok of 84% that runs 90-100% across twelve seeds. A single
  seed on a generation change is worth nothing, including when it
  agrees with the hypothesis.
- **THE WIDTH DATA SAYS WHY, AND IT IS STRUCTURAL.** Encoded
  dimensions went 1,114 at cap 400, 1,601 at 800, 1,703 at 1600.
  The step from 800 to 1600 admits ONE HUNDRED AND TWO dimensions
  because nearly every token clearing the k floor is already in by
  800 - you cannot publish vocabulary the privacy screen rejects.
  So the ceiling is the k rule, not the cap, and the earlier
  60 -> 400 gain (close +3.1/+6.2/+3.1, paired) was recovering
  genuinely excluded vocabulary where 400 -> 800 has almost none
  left to recover.
- **WHAT THIS CLOSES: the vocabulary hypothesis for multivariate
  loss.** The count columns are the weakest TSTR targets and they
  are set-size partners, so more published tokens was the obvious
  candidate. It is now measured and refuted. The remaining
  multivariate loss is NOT in what the k rule excluded; look at the
  draw and at the three-way machinery instead, where a fixture
  already shows two-way surviving every seed while the 3-way dies.
- **ONE ROW OF THAT SWEEP IS DISCARDED, NOT AVERAGED.** `visit_id`
  appears as a target because the sweep launched BEFORE the
  identifier exclusion shipped. An identifier has no relationships
  to lose and its 26 -> 27 is leakage. Dropping the row is right;
  averaging it in would have let a key vote on a fidelity question.

- **THE NET WAS PAYING FOR THREADS IT COULD NOT USE, AND IT COST
  MORE THAN EVERY OPTIMISATION EVER MADE TO IT (2026-10-27).**
  `three_way` on a 540-row, 7-column fixture: **31.5s with default
  OpenMP threading, 1.3s pinned to one thread.**
  `smoke_relationships` 248s -> 7s. The whole net 1102s -> 633s.
  HistGradientBoosting's per-fit thread setup costs more than the
  arithmetic when the frame is small, and EVERY SUITE HERE RUNS ON
  SMALL FRAMES BY CONSTRUCTION - that is what a fixture is.
- **PINNED FOR THE CHILD SUITES ONLY, AND A CHECK ASSERTS IT NEVER
  REACHES THE LIBRARY.** The real extract is 55,428 rows, where the
  threads genuinely pay; a well-meaning later edit moving the
  setting into `jointcheck` would silently slow every real run
  instead of every test run. So it is set on the SUBPROCESS
  environment in `run_all_smokes`, by `setdefault` so an outside
  value still wins, and a check asserts `OMP_NUM_THREADS` is ABSENT
  from the module. Guard the direction a fix can be wrongly
  generalised, not just the fix.
- **AND I GUESSED TWICE BEFORE MEASURING ONCE.** Convinced the cost
  was my own recent work, I trimmed fixtures 900 -> 540 and cut a
  check's target count 6 -> 2. **Both changed nothing** - 248s then
  241s then 248s. A two-minute profile found a 35x win in a place
  neither guess was near. This file already requires a number
  before a claim; a performance claim is a claim. Measure the cost
  before optimising it, and be suspicious when the suspect happens
  to be whatever you edited last.

- **THE MARGINAL DIAL'S NO-OP WAS THE CYCLIC SWEEP, NOT THE
  CURVE, AND THE NOTE EXPLAINING IT NAMED THE WRONG HALF OF
  THE GENERATOR (2026-11-03).** `dials.py` told a reader that
  a column whose parents survive the cycle trim is generated
  from their curves and so never reads a shift written into
  its marginal. Measured across all four cells with a control
  column in each run, shift +25:
      acyclic, no parents      +25.000
      acyclic, curve-drawn     +25.000
      cyclic,  no parents      +25.000
      cyclic,  curve-drawn       0.000  <-
  Being curve-drawn is not the cause - an acyclic child reads
  the dial perfectly. The cause is the CYCLIC REFINEMENT
  SWEEP, which re-ranks a cyclic column onto the multiset the
  MARGINAL drew; that pool is captured before the relationship
  and so before the dial, so the sweep took the relationship's
  order and the marginal's UNDIALED values and the dial
  vanished with no error. Named in one run by setting
  `refine_sweeps=0`, where the same child moved +25.000.
- **THE FIX MOVES THE POOL WITH THE DIAL, which keeps the
  property the re-rank rests on.** The pool is still exactly
  the multiset the marginal produced, shifted and scaled as a
  whole, so the sweep still cannot drift or diverge - the dial
  moved the marginal, which is what a marginal dial is. Shift
  lands +25.000 with spread unmoved and every other column
  unmoved; scale lands x1.500 with the centre holding.
  Verified BIT-IDENTICAL on every undialed arm, cyclic and
  acyclic, two seeds each - the change acts only where its
  mechanism lives.
- **AND "MEASURED INERT" WAS AN OVERCLAIM THAT THREE PAGES
  REPEATED.** The bench panel, the pattern report and the
  suite docstring all told a reader the recipe layer does not
  take effect. That came from shift/scale on a cyclic column -
  the one cell that ever missed, now fixed - and persistence
  and coverage were never in the measurement. With a control
  in the same run, `persistence` moved a column's
  between-patient share 0.919 -> 0.384 and `coverage` moved
  presence 0.602 -> 0.301 on an ask of 0.30. The distinction
  that survives without qualification is INVERTIBILITY: an
  output dial is a permutation the cipher takes back value for
  value, a recipe dial changes what generation produces and
  undoing it means generating again. The claim now carries a
  CHECK rather than a comment, because a number in prose that
  nothing runs is the class this file already forbids.

- **A SET COLUMN'S TOKENS WERE PATTERNS NOTHING LISTED
  (2026-11-03).** `set_token` sat in the pattern inventory's
  reach table with NO BUILDER EMITTING IT, so a page promising
  "every pattern the fit found" showed a clinical extract's
  drug and condition structure as four rows of
  column_presence. Only tokens that cleared the k floor are
  named - naming a sub-k token would be the inventory
  disclosing what the generator refused to.
- **A SET TOKEN'S SHARE IS NOT THE DIALABLE THING.** Changing
  how often a token appears means writing it onto rows that
  never recorded it: not a rearrangement, not invertible, and
  a lie about what that visit held. What moves exactly and
  reversibly is WHICH ROWS hold it, so the whole set STRING
  travels as a unit and the dial takes a partner column:
  `--dial ID=RHO --against COLUMN`. Measured: asks of -0.5,
  0.0, +0.5 and +0.8 land within 0.001, the column holds the
  identical multiset, and every published token's share drifts
  by 0.0000. The reachable band is set by the token's SHARE,
  not [-1, +1] - a token on 42% of rows reaches +/-0.850.
- **AND THE FIRST CUT BROKE THE DECLARED SIZE IDENTITY, which
  this file has already recorded twice in other engines.**
  `nc == len(conds)` holds on every source row because the
  count IS the length. Rearranging the strings and leaving the
  count behind took it from broken on 0.0% of rows to broken
  on **75.3%** - a file whose condition_count contradicts its
  own condition list. The identity is DECLARED in the
  blueprint's constraints, so it is read rather than guessed,
  and every size partner travels with the set. Identity back
  to 0.0% broken; the partner's own couplings move (+0.260 ->
  -0.239) and are reported, because a column that travels
  silently is a relationship changed behind the operator's
  back.

- **THE INTERACTION DIAL'S OBVIOUS VERSION DOUBLED THE MAIN
  EFFECTS (2026-11-03).** Flattening a joint effect means
  rearranging the child so each (A, B) cell lands on its
  ADDITIVE target, and it works - gain 0.660 -> 0.188 on a
  planted fixture. It also took the main effects from +0.367
  and +0.235 to **+0.790 and +0.501**, because the additive
  target IS the main effects and ranking on it makes them
  maximally clean. An operator who asked to remove an
  interaction would have received a file with twice the linear
  signal.
- **SO THE ARRANGEMENT CARRIES NOISE, AND THE NOISE SCALE IS
  CHOSEN BY SEARCHING BOTH OBJECTIVES AT ONCE:** land the
  asked-for gain, and among the settings that do, take the one
  whose main effects sit closest to where they started. On the
  real run: gain **0.6884 -> 0.0318** with main-effect drift
  **-0.000 / +0.015**.
- **THE TIE TOLERANCE IS THE WHOLE SEARCH.** At three decimals
  of gain error almost nothing ties, so gain decides alone and
  the mains never get a vote - that version picked a setting
  with 0.088 of total drift where its neighbour held 0.015.
  Two decimals is a tenth of the DEPARTS bar the deck already
  uses for a surface, so settings that close are genuinely
  interchangeable on gain and the mains pick between them.
- **AND THE CHECK'S OWN BAR HAD TO BE TIGHTENED TO MEAN
  ANYTHING.** At 0.06 per effect it could not tell the
  two-objective search from one that ignores the mains
  entirely - both land under it, and the mutation that removed
  the mains vote went GREEN. Pinned on TOTAL drift under 0.04,
  which separates 0.015 from 0.088. A check that cannot
  separate a mechanism from its absence is not testing the
  mechanism.
- **SUPPRESSION ONLY, AND THE BAND SAYS SO.** This dial
  flattens a surface; it cannot invent one the column's values
  cannot express, so the ceiling is the gain the file already
  has and an ask above it is called out rather than quietly
  returning the closest thing.

- **A FULL REACH TABLE READS AS COMPLETENESS, AND IT IS NOT
  (2026-11-03).** Both unreachable kinds got dials in one
  session, which emptied the "nothing reaches this" list - the
  exact table this file warned is an advert, now arrived at
  honestly. The remaining gap moved from what a dial cannot
  reach to what the inventory does not NAME, and the report
  says that in those words, because a reader looking at a
  table with no gaps will not think to ask which patterns are
  missing from it.
- **AND ELEVEN CHECKS VANISHED INSTEAD OF FAILING, because
  they were appended after the suite's own `if FAIL:
  sys.exit(1)`.** A mutation anywhere earlier deleted the
  whole block and the suite reported `1 FAILED` where it
  should have reported eight. Moved above the exit, and the
  one value the block cannot run without falls back to a
  stand-in id that cannot resolve - so the dependent checks go
  RED and loud rather than disappearing, which reads the same
  as passing. Same class as the `if und:` guard this file
  already records, and as the checks that could not fail.
- **ASSERT THE DECISION, NOT THE MESSAGE.** The bench check
  for "a set token with no partner is refused" looked for the
  refusal TEXT, which survives with the guard disabled - the
  mutation went green. Pinning the guard EXPRESSION catches
  it. Same move as counting at the decision site rather than
  re-deriving it from the seeds.
- **AND A UI BRANCH NO ROW REACHES IS AN UNTESTED BRANCH.**
  The patterns panel asserted that its "no dial reaches this"
  text was present in the source, and passed - while on the
  fixture's own blueprint four rows rendered a dial's NAME
  beside an empty cell, a named dial with no box and no
  reason. Only a RENDER tells those apart: `patRow` is
  unit-run in node against all four cases, best-effort with a
  printed NOTE when node is absent rather than a silent pass.

- **THE PATTERN DIALS RAN ON THE REAL EXTRACT, AND THE
  CONTROL ARM IS WHAT MAKES THE HEADLINE A MEASUREMENT
  (2026-11-03, run_deliverable, 800 patients, build b023fac).**
  The real blueprint carries **1,827 set-token patterns**, 50
  effect curves, 26 column presences, 23 within-patient, 23
  missing-clustering, 19 relationships, 7 interaction surfaces
  and 2 identities - and before this work the inventory
  emitted NONE of the set tokens, so the largest body of
  structure in the extract was invisible on a page promising
  every pattern the fit found. 72 patterns are reachable only
  by a blueprint dial (26 + 23 + 23, which is the whole of the
  three non-permutation kinds).
- **THE RECORDED NO-OP IS FIXED, AND THE PAIRED ARMS DIFFER IN
  ONE FILE.** Same blueprint, same seed, same patient count,
  only `generate.py` changed:
      old generate.py   asked +0.559   moved +0.000   NO-OP
      new generate.py   asked +0.559   moved +0.559   exact
  on `span_days` - the column the record names. Spread 0.937
  -> 0.937 and ZERO other columns moved their mean. Both
  numbers were read back spelled by the machine ("plus zero
  point zero zero zero" against "plus zero point five five
  nine") because the digits channel off that screen is not
  trustworthy. This is the comparison shape the two reverted
  fidelity tiers lacked: one variable, one blueprint, both
  arms run the same hour.
- **AND THE DIAL CORRECTLY REFUSES THE THREE COLUMNS IT
  CANNOT MOVE.** `procedure_count`, `procedure_quantity` and
  `active_drug_count` carry declared identities, so generation
  computes them from another column and a marginal dial on one
  is overridden BY DESIGN. The first run of the driver picked
  one and read the dial as a no-op - asked +2.342, moved
  +1.628, spread 1.125 -> 1.771 - which is the identity tier
  working, not a dial fault. Grading it would have sent the
  next person hunting a guarantee.
- **THE SET-TOKEN DIAL LANDS AT REAL WIDTH, AND IT LANDED ON
  THE HARDEST TOKEN IN THE FILE.** The driver picked
  `drug_routes / Oral against active_drug_count` - the exact
  two-regime pair this file records as the one the sharpness
  build inverted. Tie **+0.472 -> -0.472**, a full sign flip,
  landed within 0.001 of the ask, with the WORST published
  token share drifting **0.0000** across all 1,827 tokens and
  the column holding the identical multiset. `drug_routes`
  declares no size partner on this blueprint, so the
  partner-travels branch was NOT exercised on real data -
  said out loud, because it is the branch whose absence broke
  the identity on the fixture.
- **THE INTERACTION DIAL EXPOSED A CONTRADICTION IN ITS OWN
  BAND.** On `787_2 from 789_8 x 4544_3` it printed a
  reachable floor of **0.3143** and then landed **0.2404** -
  below its own stated floor. The band was probed over four
  noise scales while the search ran over fifteen, so a band
  computed from a narrower grid than the search is not the
  band. Two numbers on one screen disagreeing is this file's
  alarm and it was right. The floor is now the minimum the
  search ACTUALLY SAW, so it contains the result by
  construction and no later grid change can part them.
  **The fixture cannot reach this**: its gain bottoms out at
  noise scale 1.5, inside the old probe, so the old code
  passes the new check there. The real extract reached it
  because its gain kept falling past the probe's last scale.
  Structural fix, real-run verification.
- **AND THAT SURFACE COULD NOT BE FLATTENED FOR FREE.** Gain
  1.1503 -> 0.2404 cost the main effect -0.274 -> -0.040, a
  drift of +0.235, and the search chose the LAST scale in its
  grid - so the result was bounded by the search rather than
  by the data. The tool stated both, which is what it owes;
  whether the mains can be held while a surface is flattened
  is a property of the data, not a defect, and the first
  version of the driver graded it as a FAILURE directly
  beneath the tool's own correct report. A verdict must not
  contradict the note under it.

- **`git checkout COMMIT -- PATH` STAGES THE FILE, SO THE
  PLAIN RESTORE PUTS THE OLD ONE BACK - AND THE DATA MACHINE
  RAN OLD CODE WHILE EVERY SHA CHECK SAID OTHERWISE
  (2026-11-03).** The control arm for the span_days fix was
  run by checking the previous build's `generate.py` into the
  working tree. `git checkout <commit> -- <path>` writes to
  the INDEX as well, so the `git checkout -- <path>` that was
  meant to undo it restored from the index - the OLD file -
  and a subsequent `git checkout <newsha>` carried the staged
  modification forward. `git rev-parse HEAD` matched the
  intended commit every time; `git diff --stat HEAD` showed
  **30 deletions in generate.py**. Caught only because a
  re-run of a test that had read YES came back NO, which is
  the two-readings-disagree alarm. The restore is
  `git reset HEAD -- PATH` THEN `git checkout HEAD -- PATH`,
  and the verification is `git diff --quiet HEAD -- PATH`
  printing a WORD, not a SHA anybody reads. **A matching SHA
  is not a matching TREE**, which this file already says
  about the hybrid left by a hand-typed payload; here it
  arrived through git itself.
- **AND A RELAY THAT FILTERS THE CAVEAT IS WORSE THAN NO
  RELAY.** The real surface's search chose the LAST scale in
  its grid, so the result was bounded by the SEARCH rather
  than by the data - and the tool said so while the driver
  relaying it dropped the line, because the filter listed
  four prefixes and `NOTE:` was not one of them. The
  boundary case is pinned on the ONE fixture surface that
  reaches it: the y-child surface lands at scale 1.75, well
  inside the grid, so asserting the note there would pass
  whether or not the note existed.
- **A FLAG IS NOT A PATH, AND THE CHECK COULD NOT SEE IT.**
  The driver took `argv[0]` as the run directory, so invoking
  it with the flag alone - which is how the data machine
  invoked it - opened `--skip-generate/blueprint.json` and
  died. The local check passed the path FIRST and the flag
  second, which is the one ordering that works. Both
  orderings are tested now. And the first fail-first on that
  check found nothing twice over: once because the mutation
  was applied without asserting count==1 and silently
  no-opped, and once because the mutated run CRASHED rather
  than failing, printing no `FAIL:` line for the grep to
  find. The helper catches the exception and returns it as
  text now, so a crash becomes a failed check instead of a
  dead suite.

- **THE INTERACTION DIAL HAS TWO CONSTRUCTIONS NOW, AND THE
  SWEEP IS WHAT DECIDED THAT (2026-11-03 night).** The shipped
  one ranks the child on its additive target plus gaussian
  noise. A second starts from the arrangement the file ALREADY
  HAS, makes minimal swaps between grid cells to pull each
  cell's mean onto its additive target, then re-sorts each
  cell's values onto that cell's ORIGINAL ranking - a re-sort
  cannot move a cell mean, so it cannot move the gain, and it
  repairs the within-cell disorder the swaps caused at zero
  cost to the thing being dialled.
- **A SINGLE SEED SAID THE NEW ONE WAS THREE TIMES BETTER ON
  BOTH AXES. THE SWEEP SAID IT WINS OUTRIGHT IN 1 OF 12.**
  Four interaction shapes x three seeds. Neither construction
  dominates - each won six cells under the final rule - so
  both are generated and scored. That is this file's own rule
  about single-seed measurements of a generation change, and
  it came within one commit of shipping a regression as an
  improvement.
- **WHAT THE SWEEP ACTUALLY FOUND IS THAT THE SHIPPED
  ALGORITHM WRECKS WEAK SURFACES.** On the thresh and
  weak-interaction shapes its total main-effect drift ran
  **0.537 to 0.890** against source mains of about 1.0 and
  0.8 - it was not flattening those surfaces, it was emptying
  the column of its linear signal. The swap arm held the same
  cells at 0.012 to 0.031.
- **SO DRIFT IS A BAR, NOT A TIEBREAK.** The old score put
  gain error first and used the mains only to separate
  near-ties, which chose an arm with 0.890 of drift over one
  with 0.012 because the gain read 0.013 against 0.032.
  Wrecking both main effects to move a residual gain by two
  hundredths is not a trade anybody asked for. Arrangements
  that hold the mains within 0.10 are considered FIRST and
  the gain is optimised among those; if none qualify the
  least-damaging one is taken and the run says the ask could
  not be met without cost. Over the same twelve cells the
  **worst drift falls from 0.890 to 0.094** and every cell
  meets the bar.
- **AND THE PRICE IS NAMED: on a weak surface the suppression
  drops to 40-86% where the old rule claimed 90%+.** That is
  the honest trade and the run prints both halves - the share
  of the surface removed, and what the REJECTED construction
  would have reached and at what cost. A 35% suppression
  looks weak until you see the alternative bought 95% by
  moving a main effect a third of its value.
- **ONE THRESHOLD, NOT TWO.** A per-effect warning at 0.06
  fired beside the total-drift bar at 0.10, so a run could
  report the bar MET and the mains MOVED in one breath - a
  verdict contradicting the note under it. The bar is the
  only threshold.
- **AND REMOVING THE BAR BROKE NO CHECK UNTIL THE RIGHT
  FIXTURE EXISTED.** Every surface in the suite was a STRONG
  interaction, where the old construction is harmless, so the
  mutation that deleted the bar passed. The weak-interaction
  fixture is the shape the bar exists for: with it the chosen
  arrangement drifts 0.099, without it 0.397, and the check
  goes red. A fixture must contain the thing the check is
  about, and "an interaction" was not specific enough.
- **THE BOUNDARY NOTE IS A RECORDED COVERAGE GAP.** It fires
  when the chosen arm sits at the last noise scale, meaning
  the result was bounded by the SEARCH rather than the data.
  It has fired once, on the real extract's 787_2 surface. No
  fixture here reaches it - with the drift bar opened the
  high-noise arms still tie on gain at two decimals and lose
  on drift. Writing a check that passes without exercising it
  would be the cap-no-fixture-reaches fault; the coverage
  that exists is the real-run driver, watched printing it.

- **THE TWO-CONSTRUCTION DIAL IS A DOUBLE WIN ON THE REAL
  SURFACE, WHICH THE FIXTURE SWEEP DID NOT PREDICT
  (2026-11-03 night, run_deliverable, 787_2 from 789_8 x
  4544_3).**
      before   gain 1.1503 -> 0.2016   82%   drift 0.254
      after    gain 1.1503 -> 0.1334   88%   drift 0.081
  The main effect it used to wreck (-0.274 -> -0.040) now
  holds at -0.291. The sweep predicted a TRADE - more
  suppression or less drift, not both - and on the real
  surface the drift bar bought both, because the arrangement
  it rejected was not on the efficient frontier at all.
- **AND THE PARTNER BRANCH IS COVERED ON REAL DATA AT LAST.**
  The first real run picked `drug_routes`, which declares no
  size partner, so the repair built for that case never ran.
  Pointed at `active_drugs`, which declares
  `active_drug_count`: the identity reads broken on ZERO rows
  after the dial, unchanged from before, and the partner's
  values verifiably moved - a branch that passes by not
  running is the fault, not the test.
- **AND IT FOUND A DIAL THAT CANNOT WORK AND DID NOT SAY SO.**
  The driver's strongest tie for a partnered column was
  `active_drugs / acetaminophen` AGAINST `active_drug_count`
  - the set's OWN size partner. That partner travels with the
  set so `count == len(set)` survives, so both sides move
  together and their correlation is INVARIANT by
  construction. Asked **-0.425, landed +0.425**, unchanged to
  three decimals, with every other verdict on the page
  reading clean. Two asks that cannot both be met: keep the
  declared identity, or change that pair. It is REFUSED now,
  naming the conflict rather than reporting a landed number
  for an ask it cannot touch - the silent-difference class
  this module exists to refuse, found only by running on a
  blueprint where a set column HAS a declared partner.

- **THE BAND LIVED INSIDE THE PRINTER, SO THE DRIVER GRADED
  AGAINST NO BAND AT ALL (2026-11-03 night).** A token's
  reachable range is set by its SHARE, and that computation
  sat inline in the reporting path. The real-run driver calls
  the rearrangement directly, so it compared a landed -0.332
  against an ask of -0.400 and reported a MISS - a verdict
  about an ask that was never reachable, on a token whose
  floor is near -0.33. `token_band` is a function both halves
  call now, returning floor, ceiling and the share that sets
  them, and the driver grades against the CLIPPED ask while
  saying when the ask was outside. Two halves computing the
  same bound separately is how they come to disagree; this
  file has said so about token screening, about scaffolding
  and about the two categorical draw paths.

- **THE FITTED PATH READS MORE THAN CSV NOW, AND THE CHECK
  THAT MATTERS IS THAT THE FORMATS AGREE (2026-11-03 night,
  `synthkit/intake.py`).** An operator with an .xlsx export or
  a .jsonl dump had to convert it first, and the error when
  they did not was pandas' own - a parser complaint about a
  binary file, which says nothing about what to do next.
  `read_table` dispatches on the extension across
  csv/tsv/txt, jsonl/ndjson/json, xlsx/xlsm and parquet.
- **THE LOAD-BEARING CHECK IS NOT "XLSX OPENS".** The reader is
  part of the suspect path - this file records the wrong-type
  fault four times and empty-versus-missing twice, and both
  are reader faults. So every format is asserted to produce
  the IDENTICAL FRAME to the CSV path. A second reader that
  disagrees about a blank or a number is invisible from
  inside, because the run would be self-consistently wrong.
- **AND IT CAUGHT ONE IMMEDIATELY.** pandas' own `dtype=str`
  renders an integral float 60.0 as `"60"` where the CSV path
  preserves `"60.0"`, so a normal draw that happened to land
  on a whole number disagreed between Excel and CSV. The first
  fix was worse: rendering any integral value without its tail
  repaired the integer column and broke the float one.
  INTEGRALITY IS A PROPERTY OF THE COLUMN, NOT THE CELL - all
  values whole means an integer column and no tails, anything
  fractional means every value keeps its own rendering - and
  the Excel branch reads NATIVELY so there is only one
  renderer in the path rather than pandas' and ours.
- **A WORKBOOK HAS SHEETS AND A CSV DOES NOT.** Reading the
  first tab silently is how an operator gets a run over their
  notes; the run names the sheet it took and what else was in
  the book, and an absent sheet is refused LISTING the real
  ones so the next attempt can succeed. A format whose package
  is missing is refused with the install that fixes it.
- **AND THE NEW SUITE PASSED HERE AND CRASHED THERE, ON AN
  OPEN FILE HANDLE.** `pd.ExcelFile(src)` left open keeps the
  file LOCKED on Windows, so the suite completed every check
  on the development machine and then died with **WinError
  32** cleaning up its own temp directory on the data
  machine - which would have taken `RUN_CHECKS.bat` down for
  the whole team, the same severity as the Chrome hang. POSIX
  does not block a delete on an open handle, so NO amount of
  testing here could have found it. The handle is closed with
  a context manager, and the check that catches a reintroduced
  leak deletes the file right after reading: REAL on Windows,
  vacuous on POSIX, and labelled as such rather than left to
  look like coverage. Third entry in this file about a fault
  that only the other machine can see.

- **THE VENDOR SEAT NEEDED A COMPETITOR, NOT A FLOOR
  (2026-11-03 night, `synthkit/automl.py`).** `autosolver` is
  deliberately modest - stdlib logistic regression, and honest
  about it - so a vendor beating it has shown nothing. The
  competitor tries three families with light feature
  engineering and picks by held-out AUC on a slice carved from
  the TRAINING rows. Measured over three mechanisms x three
  seeds: **never worse than the floor** (0 of 9 arms worse by
  more than 0.01), xor **0.49-0.52 -> 0.88-0.90**, curved
  **0.54-0.58 -> 0.72-0.76**, and a correct TIE on linear
  where it names logistic regression the winner itself.
- **"BEATS THE FLOOR ON ALL THREE MECHANISMS" WAS THE WRONG
  GATE.** A selector that picks the best family must TIE where
  the simple family is best - beating the floor on a linear
  mechanism would mean beating the right answer. The property
  worth asserting is NEVER WORSE, decisively better where the
  floor is blind, and that is what the suite pins.
- **IT CANNOT PEEK, AND PROVING THAT TOOK THREE MUTATIONS.**
  The first keyed the selection on `len(test_rows)`, which is
  identical between the two probe calls, so nothing moved. The
  second added a test-derived term too small to flip a 0.02
  AUC gap. The third decided selection ENTIRELY on the test
  rows and broke three performance checks - but not the
  no-peek check itself, because that probe merely SWAPPED two
  columns and left the test rows statistically similar.
  Constant rows cannot be confused with the real ones, and
  with that probe the same mutation breaks five.
- **AND ONE CHECK WAS WRITTEN AS A LITERAL `True`.** A
  placeholder that reads as coverage, shipped green for one
  run, in a suite whose docstring is about not being fooled.
  It asserts the thing now: the competitor must NAME DIFFERENT
  FAMILIES on different mechanisms, or it is a second floor
  with extra steps.
- **AND THE SHOWDOWN PLACES THE VENDOR AGAINST THE COMPETITOR,
  NOT THE FLOOR.** `--competitor automl` adds a fourth arm and
  the verdict changes with it: above our competitor reads
  "the claim is worth something"; below the floor still reads
  "loses to a stdlib baseline"; and in between is named as
  exactly that - **"better than nothing, worse than us"** -
  rather than rounding up to "beats the baseline", which was
  the old wording for a vendor that had cleared only stdlib
  logistic regression. The arm is OPTIONAL and a tier where
  it did not run keeps the old verdict, because absent is not
  the same fact as scoring zero and an old result must still
  load. The competitor is printed BETWEEN the floor and the
  vendor, where the eye goes; at the end the floor keeps
  reading as the thing to beat.
- **AND THE CARD GROWS THE COLUMN ONLY WHERE THE ARM RAN.** A
  blank competitor cell on every row reads as a competitor
  that scored nothing, which is a different fact from one
  nobody asked to run - the absent-is-not-zero rule again, in
  a report. With the arm present the card names the column,
  places it between the floor and the vendor, and says IN
  WORDS that **the floor is not the bar**: four numbers with
  no sentence leaves a reader to assume the smallest one was
  the target.

- **THE DIAL MATRIX RAN ON THE REAL BLUEPRINT, AND THE REAL
  GRAPH LEAKS WHERE THE FIXTURE DID NOT (2026-11-03 night).**
  Four columns against 88 relationships, where the fixture has
  one:
      span_days.shift              CLEAN
      span_days.scale              CLEAN   x1.50 asked x1.50
      span_days.persistence        NO-OP
      systolic.coverage            CLEAN
      systolic.missing_clustering  LEAKED  own persistence +0.13
      patients.count               LEAKED  age persistence +0.10,
                                           a pair +0.16
      patients.visits_scale        LEAKED  span_days.spread
                                           +0.12
      span_days<-proc.strength     NO-OP
  **`span_days.shift` and `.scale` reading CLEAN is the
  cyclic-sweep fix holding on the real blueprint** through a
  second instrument. THREE dials leak into neighbouring
  properties that were all clean on the one-relationship
  fixture - a dense graph propagates a dial's effect in ways
  a forest cannot show, which is the same lesson the
  fixture-was-a-forest entry records for discovery.
  **CORRECTED AN HOUR LATER BY THE NOISE FLOOR, and the
  correction is the point.** The first version of this entry
  said `patients.visits_scale` moved TWO count columns'
  spread, +0.12 and +0.13. Re-running the UNDIALLED blueprint
  at other seeds puts the floor for `procedure_count.spread`
  at **+0.17**, so that one was the SEED and not the dial;
  `span_days.spread +0.12` survives. One of three reported
  collateral items was noise, written up as a finding,
  exactly as this file warns about every single-seed
  measurement - and caught only because the instrument was
  made to measure itself.
- **AND THE MATRIX WAS COUNTING ITS OWN CONTROL AS A FAULT.**
  `scale@1.0` exists to prove the NO-OP detector can fire - a
  dial asked for x1.0 moves nothing BY DEFINITION - and it
  was listed among the defects, so the real blueprint showed
  three NO-OPs where only two were findings. A control
  counted among the faults makes every real one a third less
  alarming. It is reported as the control that proves the
  detector can fail, and the fault list no longer names it.
- **AND THE MATRIX NOW MEASURES ITS OWN NOISE FLOOR, BECAUSE
  THREE OF THOSE LEAKS MIGHT BE THE SEED.** Every verdict
  compares ONE dialled run against ONE undialled run at one
  seed with a flat tolerance - and `patients.count` draws a
  different number of people while `visits_scale` draws a
  different number of rows, so the whole frame is re-drawn
  and every property moves by whatever re-seeding moves it.
  The three LEAKED verdicts sat at +0.10 to +0.16, which is
  the size a different seed can produce unaided.
  `--noise-seeds N` re-runs the UNDIALLED blueprint at other
  seeds and takes the per-property spread as the floor; a
  move no bigger than that is reported as INSIDE THE NOISE
  FLOOR with the floor printed beside it, not as a finding.
  What the floor absorbed is listed, because a tolerance that
  silently swallows moves is indistinguishable from a
  detector that cannot see them. This file's oldest rule
  about single-seed measurements, applied to the instrument
  rather than to the thing it measures.

- **THE CLOSE GAP ON THE REAL EXTRACT IS ONE PAIR, AND THIRTEEN
  OF THE SEVENTEEN MISSES CANNOT BE REACHED (2026-11-04,
  `close_gap.py` on run_deliverable).** 17 pairs miss close at
  **87.6%** against an 87.9% bar - the gate needs exactly ONE
  more pair inside 0.2. The misses decompose: token against a
  plain column 5, CROSS-set token pair 5, ordinary numeric 4,
  other scaffolding 3, and **ZERO bound-explained** - none of
  this gap is the k rule, so all of it is in principle
  fixable. But only **4 are repairable by rearrangement**: the
  other 13 sit on token indicators and set sizes, which are
  search machinery dropped before the file is written, and
  nobody can rearrange a column that is not there.
- **THE OVERSHOOT-IS-CYCLIC HYPOTHESIS WAS REFUTED BY ITS OWN
  CONTROL.** 14 of the 17 misses are FADES and 3 are
  OVERSHOOTS - the worst being `777_3 <- 6690_2` at source
  +0.444, generated **+0.721**. The sweeps over-correlating
  cyclic neighbours was the obvious suspect, and this file
  records them doing exactly that on dense twins. Measured:
  **ZERO of the three overshot pairs sit in a cycle**, and of
  25 cyclic pairs that PASSED only 8 lean high, which is near
  chance. Clean negative, and the reason to measure before
  building.
- **AND THE REPAIR TOOL WAS COMPARING TWO DIFFERENT
  MEASUREMENTS.** Its `before` came from the run's recorded
  `fidelity.json` and its `after` from a fresh `compare()` -
  and on the real extract those count different pairs. The
  recorded run relates **137**; the re-measure relates **57**,
  because it reads the WRITTEN FILE, which drops the token
  indicators the original search expanded. So the first real
  `--close` run printed `120/137 before, 47/57 after` and a
  verdict column over the top of it. That is two answers to
  two questions wearing a before and an after, and this file's
  own rule says a percentage whose denominator moved is not a
  comparison. **Any earlier claim from this tool rests on the
  same arrangement and should be re-read.** The unrepaired
  frame is measured through the identical call now, both
  denominators are printed, and a mismatch with the recorded
  count is explained rather than left to be misread.
- **THE FIXTURE CANNOT REACH THAT FAULT**, because it has no
  set columns and so no expansion gap - the mutation that
  restored the old arrangement passed. The mismatch is
  PLANTED in the suite instead of faked away, so the branch
  that explains it is exercised.

- **THE JOINT CROSS-TABLE SOLVE LANDS TWO COUPLINGS WHERE
  TAKING THE STRONGEST LANDED ONE - AND ITS REAL FINDING IS
  THAT SOME TARGETS ARE UNREACHABLE BY ARITHMETIC (2026-11-06,
  MT-2).** MT-1 enforced ONE coupling per link, the strongest,
  and the obvious improvement was to solve for all of them at
  once. Paired on one fixture at one seed, two child columns
  riding the SAME parent, only the coupling set differing:
      arm                   lab_value   lab_b   total error
      source                   +0.898  +0.883
      MT-1 strongest only      +0.923  +0.738      0.171
      MT-2 joint               +0.880  +0.872      0.029
  Six times tighter, and the second column is the whole of it -
  enforcing one coupling left `lab_b` at +0.738 against its own
  source +0.883.
- **BUT A SINGLE FOREIGN KEY IS ONE CHOICE PER ROW, SO THE
  CORRELATIONS ACROSS A LINK HAVE RANK ONE BY CONSTRUCTION.**
  Two children riding DIFFERENT parents is a DIAGONAL target,
  which is rank two, and no assignment can produce it. The
  first cut fitted rank one to it anyway and the result was
  worse than MT-1: it landed one pair at **+0.994 against a
  target of +0.914** - an OVERSHOOT - and lost the other
  entirely, **+0.043 against +0.898**. A solve that spends its
  one degree of freedom on whichever direction the singular
  vector happens to point, and reports success, is exactly the
  silent-difference fault this house refuses everywhere else.
- **SO NOT-JOINTLY-REACHABLE IS A VERDICT, NOT A BEST EFFORT.**
  `carried` - the leading singular value over the whole norm -
  says in ADVANCE how much of a target one assignment can hold:
  1.00 for a shared parent, 0.71 for two independent couplings
  of equal size. Below `JOINT_MIN_CARRIED` the solve enforces
  the strongest, NAMES every pair that cannot come with it, and
  states that one foreign key is the reason. Measured on the
  diagonal fixture: the enforced pair lands +0.911 against
  +0.914 and the unreachable one is named rather than summed -
  a count a reader cannot act on was the first version.
- **AND NO PER-TABLE NUMBER CAN DEGRADE, BY CONSTRUCTION.** The
  assignment permutes KEYS, so every child column holds the
  identical multiset it was generated with - asserted in both
  arms rather than argued. The same property the repairs and
  the dials rest on.
- **THE REPORT TRAVELS BY ITS OWN PARAMETER.** The first cut
  smuggled it back as `out["__couplings__"]`, which would hand
  a LIST to anything iterating those values as tables -
  `check_integrity` does - and would have failed somewhere
  else entirely. `report_into` is explicit, and a check asserts
  every value in the returned mapping is still a frame.
- **AND WHICH PATH RAN IS STATED, NOT INFERRED FROM A MISSING
  KEY.** Only the refusal wrote `joint`, so the joint path was
  recognisable only by the field's ABSENCE - the
  empty-versus-missing distinction, in a record. It also made
  the fail-first CRASH instead of fail: the mutation removed
  the key and the check's own message indexed it, so the suite
  raised and printed no `FAIL:` line at all. Both fixed - the
  field is always written, and a message never indexes what its
  mutation removes.

- **A ZIP IS THE ONE PHI KIND WHOSE VALUE SHAPE CANNOT CARRY
  THE DETECTION, AND THE FIRST HEADER PATTERN WAS SILENTLY
  DEAD ON THE MOST COMMON NAME THE COLUMN HAS (2026-11-06,
  PHI-4).** Every other kind in `scrub` has a shape nothing
  else wears - an SSN is three-two-four, an email has an @.
  FIVE DIGITS IS A VISIT COUNT, A LAB RESULT, A ROW ID: the
  most ordinary shape a clinical column can have, so a bare
  five-digit test would flag half an extract. A ZIP therefore
  REQUIRES one of three tells, and the module says that is a
  narrower claim than the others rather than leaving it as a
  weaker version of them: ZIP+4, a PRESERVED LEADING ZERO
  (which proves somebody kept the column as text because the
  digit is part of the code), or the HEADER with five digits
  under it.
- **AND `\bzip\b` CANNOT MATCH `patient_zip`, BECAUSE `_` IS A
  WORD CHARACTER.** There is no boundary before the z, so the
  header tell was dead for the single most common name this
  column has - caught by a negative-control probe, NOT by
  reading the pattern, and `zip_code` matched anyway through a
  different alternative so the bug was invisible from the
  obvious test. An underscore separates words everywhere in a
  database and nowhere in `\b`. The mutation back to the old
  pattern turns SIX checks red.
- **THE LEADING-ZERO BAR IS DELIBERATELY NOT A MAJORITY, AND
  THE FIXTURE HAD TO BE REBUILT TO SHOW IT.** Zero-leading
  areas are New England and a minority of any national column,
  so a `MATCH_SHARE` bar would miss every real ZIP column that
  has them. The first fixture put 2 of 4 values at a leading
  zero - 50%, exactly AT the bar - so the check asserting the
  bar is below `MATCH_SHARE` failed on its own fixture. A
  fixture must carry the thing the check is about, and
  "has leading zeros" was not specific enough.
- **TRUNCATING TO THREE DIGITS IS HALF THE RULE.** A
  three-digit area holding 20,000 people or fewer identifies
  too narrowly to publish at all, and Safe Harbor names
  seventeen prefixes that become 000. A plain truncation skips
  that silently. `RESTRICTED_ZIP3` is encoded rather than
  computed - a population table is not something this project
  can carry or keep current, and a list WRONG about which
  areas are sparse would be worse than no list, so it is the
  published one and its provenance is stated. A column with
  none of them SAYS SO, because absent is not the same fact as
  not-looked-for.
- **AND THE REMEDY IS GENERALISE, NEVER DROP.** Geography
  carries catchment and travel distance - things a model is
  often asked about - and the rule asks for three digits, not
  for the column to go. Dropping would remove the identifier
  by destroying the variable, which is the trade the `REMEDY`
  table exists to refuse. The mutation to `drop` turns three
  checks red.
- **A DETECTOR THAT FINDS NOTHING READS AS A CLEAN BILL OF
  HEALTH, AND IT IS NOT ONE.** It is a clean bill for whatever
  the detector LOOKS AT, and nobody could tell those apart
  from the output - a governance reader had to infer coverage
  from whichever kinds happened to fire on their file.
  `scrub.inventory()` carries all EIGHTEEN Safe Harbor
  categories with a status and a note each: 8 covered, 6
  partial, 2 NOT COVERED, 2 not-in-a-table. The counts are
  DERIVED from the rows, never written beside them. The gaps
  are NAMED - vehicle and device serial numbers have no
  distinctive shape, and the device case is called LIKELY in
  clinical data (OMOP has a device_exposure table) rather than
  left as a footnote. `scrub --inventory` needs no CSV,
  because the question is asked before anybody has a table to
  point a scan at.
- **TWO CATEGORIES WERE MISSING OUTRIGHT RATHER THAN
  CONSIDERED AND REJECTED: web addresses and IP addresses.**
  Both wear shapes nothing else wears, which is the property
  that let them join the value battery instead of needing a
  header. They turn up in audit and device columns rather than
  clinical ones, which is why nobody had looked.
- **AND MAKING THE CSV OPTIONAL FOR `--inventory` ATE
  ARGPARSE'S OWN REFUSAL.** A bare `scrub` fell through to
  pandas and raised `ValueError: Invalid file path or buffer
  object type: <class 'NoneType'>` where there had been a
  clean "required argument" - a traceback instead of a
  refusal, and the exit code read 0 through a pipeline's
  `tail`. A feature that arrives by removing an error path has
  deleted a guard somebody planted on purpose; this file
  already records that for campaign-run.

- **THE WHOLE MULTI-TABLE CAPABILITY WAS REACHABLE ONLY FROM
  PYTHON AND CALLED BY NOTHING BUT ITS OWN SUITE, AND I FOUND
  THAT OUT BY ASKING AFTER IMPROVING IT (2026-11-06,
  `synthkit relate`).** `detect`, `cross_couplings` and
  `assign_keys` had no operator path at all: `cmd_schema`
  imports `detect, integrity, render` and nothing else, so
  `synthkit schema` reported a schema it could not act on, and
  there was no command that generated a linked set. I had just
  spent a session improving the joint solve that NO SHIPPED
  COMMAND INVOKED. This file records the
  capability-wired-to-nothing fault for the holdout, for
  `plant_frame`, for the parallax layers and for the pattern
  dial's cipher; this is the largest instance yet, and the
  check that would have caught it is one line - grep for the
  function's callers outside its suite.
- **`relate` MAKES IT REAL, AND ITS CONTROL ARM IS THE
  EVIDENCE.** `synthkit relate A.csv B.csv --out DIR` reads,
  detects the keys from the VALUES, measures the cross-table
  couplings, fits each table, generates, assigns keys to carry
  the couplings, and writes the set with both integrity
  readings. Measured through the WRITTEN FILES - the reader and
  writer are in the suspect path and an in-memory fixture
  cannot reach them - at one seed, one flag apart:
      source            lab_value +0.892   lab_b +0.875
      relate            lab_value +0.871   lab_b +0.865
      --no-couple       lab_value -0.028   lab_b -0.066
  Without enforcement the cross-table structure is DESTROYED,
  not weakened. `--no-couple` ships as the control arm, named
  as such, because this is the comparison shape two reverted
  fidelity tiers lacked.
- **WHICH COLUMN GROUPS A TABLE IS A DECISION, SO IT IS
  PRINTED AND OVERRIDABLE.** The default is the foreign key
  where there is one - a child's entity is its parent - and the
  primary key otherwise, with the reason stated per table.
  Every per-entity and temporal statistic is measured against
  that column, so guessing wrong changes every number without
  changing any message, which is the `--time-col` fault. A
  `--group-by TABLE=COLUMN` that cannot be honored is REFUSED
  by name rather than ignored.
- **AND THE COMMAND STATES WHAT DOES NOT CROSS.** Each table is
  still fitted on its own; what carries a cross-table
  relationship is the KEY ASSIGNMENT - a child is attached to a
  parent whose value fits - and a child's VALUES are never
  re-drawn from its parent. Printed at the end of every run,
  because a tool that silently models less than its output
  implies is the spec-that-looks-complete fault.
- **AND READING `integrity`'s FIELD AS `match` INSTEAD OF
  `share` WAS THE SIXTH WRONG-KEY FAULT.** It crashed rather
  than printing an empty table, which is the lucky version.
  This file already prescribes the fix - assert the key names
  against a LIVE call - and the lesson keeps being that the
  field name which SOUNDS right is not the contract. Read it;
  do not remember it.

- **ZERO READ AS EIGHT, AND MY OWN PREDICATE MANUFACTURED THE
  SCARE (2026-11-06).** A net running on the data machine was
  polled with `f=[x for x in L if x[:4]=="FAIL" or "FAILED" in
  x]` and came back `FAILCOUNT 8`. Both halves were wrong at
  once. The `"FAILED" in x` clause is far too broad - it
  matches check-label text, which is how a PowerShell `^FAIL`
  mangled to `*FAIL` once produced a confident
  `VERDICT-FAILED` from PASSING lines. And the count itself was
  OCR: re-asked two ways, ZERO lines start with `FAIL` and
  ZERO contain `FAILED`, with the real zero rendering as `@` in
  one read and as `8` in the other. This file already records
  zero reading as six; it also reads as eight. The verdict must
  come from the SUCCESS MARKER - the final `All N checks
  passed.` line - and a diagnostic predicate must be as narrow
  as the thing it claims to find.

- **A PUBLISHED COVERAGE WAS OVERWRITTEN BY AN IDENTITY, AND
  THE GATE COULD SEE IT (2026-11-06).** Looking for the
  recorded presence/count defect, a procedures-shaped fixture
  found a different one. The size identity is DETECTED on rows
  where both columns are present and was ENFORCED wherever the
  SET is present - and a set column cannot be missing at all,
  because its blanks encode to `sets.EMPTY` (an empty set is a
  fact about the visit). So the copy ran on EVERY row and wrote
  a size into rows whose partner the blueprint says should be
  ABSENT:
      procedure_count coverage   source 0.9375
      identity on both-present rows      1.0000
      identity on ALL source rows        0.9375
      generated coverage BEFORE          1.0000   delta +0.0625
      generated coverage AFTER           0.9275   delta -0.0100
  That turns "nobody recorded whether there were procedures"
  into "there were zero" - the empty-versus-missing conflation
  this file treats as a first-class fault, committed by the
  enforcement written to fix a different one. A blueprint
  number that is read and then ignored is the
  read-but-not-used fault, and the delta was over the 0.05 bar,
  so the GATE was already reporting it and nothing was asking.
- **AND THE NEIGHBOURING PROPERTIES WERE ASSERTED IN THE SAME
  MEASUREMENT.** Paired, same seed, only `generate.py`
  differing: the identity still holds on 100% of both-present
  rows; the set's own empty share is BIT-IDENTICAL (-0.0117
  both arms); and the gate's relationship criteria are
  unchanged (compared 11, sign_kept 11, close 11, inverted 0 in
  both). On a control fixture whose partner is present on every
  row the output is BIT-IDENTICAL, so the change acts only
  where its mechanism lives.
- **THE RECORDED PRESENCE/COUNT DEFECT DID NOT REPRODUCE, AND
  IS CLOSED AS FIXED INCIDENTALLY.** It read "the source is
  empty on 79.6% of PRESENT rows and generation emits 97.3%".
  Re-measured on the same shape through the CLI: source 0.826,
  generated 0.814, a 1.2-point gap. The set rewrite -
  independent per-token draw, sizes EMERGENT from the picks,
  one shared empty predicate - fixed it without anybody aiming
  at it. Re-measure a recorded open item before working on it;
  this one had been overtaken.
- **AND THE IDENTITY COPY IS INERT ON EVERY FIXTURE THIS REPO
  HAS, WHICH THE FAIL-FIRST FOUND AND THE SUITE NOW SAYS OUT
  LOUD.** Disabling the overwrite ENTIRELY fails nothing - not
  in `smoke_sets`, and not through the real CLI on the
  procedures fixture, where identity reads 1.0000 and coverage
  0.9275 with the copy and without it. When the blueprint
  DECLARES the identity, each row's set is drawn AT that row's
  count, so the identity arrives from the DRAW and the copy is
  belt-and-braces. Two arms agreeing to four decimals are one
  arm - and here that is the correct reading rather than a
  measurement bug. The shape that would exercise the copy is
  the one the original defect had: the count a relationship
  CHILD of the set's token indicators, drawn AFTER the set. No
  fixture here carries it. The copy STAYS - removing it would
  be a change with no measured benefit and real risk on the one
  shape nothing tests - and the check beside it states that it
  pins the DRAW rather than implying it tests the copy.
- **`git checkout HEAD -- PATH` RESTORED AWAY MY OWN
  UNCOMMITTED FIX.** Mutating `generate.py` to run an
  old-against-new comparison, the restore was
  `git checkout HEAD -- synthkit/generate.py` - and HEAD did
  not contain the fix, which was still uncommitted. The
  comparison was valid; the fix was gone, and only
  `git diff --quiet HEAD` printing CLEAN revealed it. This file
  already records that this command stages as well as writes;
  the new lesson is narrower and worse: **restore from a COPY
  of the working state, never from a commit, when the working
  state is what you are trying to keep.** Re-applied and
  verified the text is IN the file before believing any further
  measurement of it.
- **AND THE DATA MACHINE CONFIRMED THE CHAIN, DELTA EXACTLY
  TWELVE.** The PHI-4 build ran there off a zipball fetched BY
  COMMIT SHA (public remote, no token, byte size verified
  against the local download first): **ALL GREEN, 94 suites,
  2969 checks, 1357s** on Python 3.14.6 with pandas 3.0.5,
  numpy 2.5.2. The local checkout read 2981, so the gap is the
  recorded TWELVE and it still decomposes with no residue. The
  count was crossed in WORDS - "nine four suites, two nine six
  nine checks" - because that delta is the number that decides
  whether a correct run reads as a failure.
- **AND THE NEW DETECTORS PRODUCED ZERO FALSE POSITIVES ON THE
  REAL EXTRACT.** `scrub` on the 55,428-row, 44-column extract:
  **NEWKIND NONE** - no zip_code, url or ip_address finding -
  with the four pre-existing findings unchanged (one
  identifier, two service dates, one age) and 39 clear plus 1
  out of scope, which balances against 44 exactly. The last
  real-extract read of this detector exposed THREE faults the
  fixture could not reach, so a clean read here is evidence
  rather than a formality.

- **THE RECEIPTS EXISTED AS EVIDENCE AND NOT AS AN ARTIFACT,
  AND THE DIFFERENCE IS WHETHER ONE PATTERN CAN BE AUDITED
  (2026-11-06, REC-1, `synthkit receipts RUNDIR`).** Goal 3 is
  "every pattern found, explained, with receipts" and the
  receipts stayed PLANNED for months - not because the evidence
  was missing but because it was SPREAD: skill and support in
  `catalogue.json`, the effect shape and importances in
  `blueprint.json`, survival in `fidelity.json`, and the trim
  note in a page of prose. A reader who wanted to know whether
  ONE pattern was trustworthy had to hold three files in their
  head. A receipt is that assembly done per claim.
- **NOTHING IS RECOMPUTED, AND A CHECK ASSERTS THE DIGITS
  MATCH.** Every number is read from the run's own artifacts,
  because a second implementation is how the shipped artifact
  and the report come to disagree - this file says that about
  token screening, about scaffolding, about the two categorical
  draw paths and about a whole engine. The suite compares every
  receipt's skill and support against the catalogue's own
  values and every survival reading against fidelity.json's,
  to the digit; rounding one of them to two decimals turns it
  red.
- **AND THE LIMITS SECTION IS THE POINT, because a receipt
  that only carried the evidence would read as a
  certificate.** Each one states what it does NOT establish,
  derived per claim from the measured record: ASSOCIATION
  rather than causation (the arrow was chosen by out-of-sample
  skill, which says which direction PREDICTS better); ONE
  COHORT (the holdout is held-out PATIENTS from the same
  extract); a skill under 0.30 called WEAK rather than
  suppressed, because a thin finding and no finding are
  different facts; a seed-fragile attribution NAMED; a sign
  that reverses when co-parents are held fixed; the k it was
  screened at, so a short magnitude reads as the privacy rule
  working; and that TRIMS ARE NOT ASSERTED, because the trim
  record is prose and not a field. Removing the section turns
  SEVEN checks red.
- **THE SHARPEST STATEMENT IS ABOUT A CLAIM NOBODY CAN
  CHECK.** When the child is a `__has__` indicator the claim
  cannot be verified in the output AT ALL - the column is
  search machinery, dropped before the file is written, and
  nobody can read a column that is not there. The receipt says
  so, through `sets.is_scaffolding`, the same predicate the
  pipeline uses. On the fixture 8 of 10 claims are that kind,
  which is why a fixture without a set column could not have
  reached it.
- **AND A CONTENT CHECK CANNOT SEE A DEAD CODE PATH, AGAIN.**
  The first version of the command raised `NameError: name 'os'
  is not defined` at the WRITE step - `cli.py` does not import
  `os` at module level at all - and the printed report looked
  perfect, because stderr is unbuffered while piped stdout is
  block-buffered, so the traceback appeared ABOVE the output I
  was reading through `tail -6`. Found only by checking the
  real exit code and an EMPTY stderr from a subprocess. Read
  the whole output, not a filtered view; this file already says
  that about a mutation and it applies to a feature's first
  run.

- **THE FIRST REAL RECEIPT READ SAYS SIXTY-FOUR OF EIGHTY-EIGHT
  PUBLISHED CLAIMS ARE ON COLUMNS THE OPERATOR NEVER RECEIVES
  (2026-11-06, run_deliverable).** Run through the CLI on the
  data machine, 2,826 lines:
      claims published                     88
      claims whose CHILD is a search column 64
      parent readings                     752
        kept                              175
        not measured                      556
        faded                              16
        INVERTED                            3
        overshot                            2
  Every figure crossed in WORDS - "eight eight", "six four",
  "one seven five", "five five six" - because they would go in
  a report and the digits channel off that screen is not
  trustworthy.
- **THAT IS A KNOWN THING MADE COUNTABLE, AND IT CHANGES HOW
  THE CLAIM COUNT SHOULD BE READ.** This file already says
  scaffolding is not the operator's data - it says it about the
  constraints section, about the fidelity summary's column
  counts, and about the gate's absolute stop judging a
  scaffolding inversion. What was not countable until now is
  the SHARE: nearly three quarters of the published claims sit
  on `__has__` indicators and set sizes, which are dropped
  before `generated.csv` is written. A reader told "88
  relationships found" and not told that 64 of them cannot be
  checked in the file they received has been given a number
  that flatters. The receipts say it per claim and in the
  header total.
- **AND THE 556 UNMEASURED PARENT READINGS ARE MOSTLY THE SAME
  FACT, NOT A SECOND ONE.** A pair is measured on BOTH tables,
  so a parent that is search machinery cannot be compared in
  the output however well the source relates it. Reporting the
  556 as a fidelity gap would be the
  percentage-whose-denominator-moved fault in a new place; it
  is reported beside the scaffolding count so the two can be
  read together. The 3 INVERTED here are parent-pair readings
  across all claims, which is a DIFFERENT denominator from the
  gate's `pairs_own_inverted` - say which one is quoted.

- **A COLUMN ONE HOP FURTHER UP IS STILL A COLUMN THE CHILD
  TRACKS, AND NOTHING WAS LOOKING FOR IT (2026-11-06, MT-3).**
  `cross_couplings` iterates the LINKS, so it only ever compared
  a child against its DIRECT parent. On a person -> visit -> lab
  chain, a lab value riding the PERSON's age spans two links -
  and it was not weakened, it was NEVER SEARCHED FOR. Measured
  on a three-table fixture through the written files:
      source two-hop creatinine ~ age     +0.976
      before (one-hop detection only)     -0.000
      after                               +0.981
      control arm, --no-couple            -0.000
  The run's own output said "0 cross-table coupling(s)", which
  reads as none there rather than none looked for. A missing
  coupling is invisible in exactly the same way a missing
  category is, and this file already has a rule about reporting
  a total a reader can check against.
- **AND THE FIX NEEDED NO SECOND SOLVER, because once a visit
  is attached to a person that visit ROW carries the person's
  age.** Materialising the inherited numeric columns onto the
  parent frame turns a two-hop coupling into an ordinary
  one-hop coupling on a column named `__via__person_id__age`,
  and the existing detection and the existing rank-one
  assignment then work unchanged. `with_inherited` joins
  through whichever foreign key is actually present in the
  frame handed to it, so it runs identically on the SOURCE
  (real keys) and on a GENERATED set (invented keys) - which is
  the only way detection and enforcement can agree about what a
  two-hop coupling IS. One function, both halves, for the same
  reason token screening and the scaffolding predicate each
  live in one place.
- **THE LINK LOOP HAD TO BECOME TOPOLOGICAL, and it was in
  DETECTION order.** That was harmless while every coupling was
  one hop - a child only needs its parent's KEYS, and those are
  invented for every table up front. It stops being harmless
  the moment a child is scored against a column its parent
  INHERITED: the parent's own foreign key must be assigned
  first or the inherited column is simply not there yet.
  Depth is counted by walking up to the root, which also makes
  a cycle impossible to loop on. Reversing the order turns the
  check red.
- **AND NO `__via__` COLUMN REACHES A WRITTEN FILE.** It is
  scaffolding for the assignment, and this project does not put
  a column in somebody's file that they did not give you -
  asserted on every written table, with a mutation that leaks
  them turning it red. Same rule as the `__has__` indicators
  and the invented `visit_number`.

- **THE PAGE THAT STATES HOW COMPLETE THIS PRODUCT IS HAD NO
  CHECKS, AND IT CONTRADICTED ITSELF WITHIN AN HOUR
  (2026-11-06, `smoke_scorecard`).** `docs/goals_scorecard.md`
  carries eight per-goal percentages, a summary list of the
  same eight, and a stated overall - three statements of one
  thing, so they can be checked against each other. Nothing
  read the file. Its headings said goal 1 was at 90% while its
  summary list said 86 and its overall said 90.1, and that
  inconsistency was COMMITTED. Two numbers on one page
  disagreeing is this project's own alarm, and the page
  carrying its completeness claim had no alarm on it - the
  `peek.py had no checks at all` fault, aimed at the progress
  report. Eight checks now: the headings and the summary list
  must be EQUAL, the stated overall must be their mean to one
  decimal, every goal under 100% must carry a named `What
  remains`, and the page must still say in its own words that
  the percentages are JUDGMENTS. Three mutations bite, and the
  first is literally the inconsistency that shipped. It
  deliberately does not grade the percentages: a check that
  tried to verify a judgment would be asserting a feeling.
- **AND THE CAUSE WAS ASSERT-THEN-WRITE-AT-THE-END, AGAIN,
  COMPOUNDED BY A WRAPPED PHRASE.** The edit script made two
  replacements and wrote once at the end; the SECOND assertion
  failed, so the FIRST edit - which had succeeded - was thrown
  away, and the console showed only a traceback while the
  commit went through. This file has recorded that trap twice
  before. The second assertion failed because the phrase I
  searched for spans a LINE BREAK in the file ("rank-one" /
  "ceiling stated as..."), which is the wrapped-heading trap
  recorded four times. Both rules in one failure: write after
  each edit, and anchor inside a single line.

- **THE SCAFFOLDING IS FLATTERING THE CLOSE CRITERION, NOT
  DRAGGING IT DOWN - AND THAT REFUTED THE FIX I WAS ABOUT TO
  MAKE (2026-11-06).** The INVERTED stop was switched to the
  operator's own columns because nobody can read a column that
  is not there; `close` and `direction kept` still judge the
  scaffolding-inclusive total, and `pairs_own_close` existed
  and appeared only in a note. The obvious move was to judge
  close on the operator's columns too, which would also have
  been a route to Goal 5's named item - the unrepaired engine
  reaching the bar. Measured on run_deliverable FIRST, read
  back in words:
      close,     all pairs   120/137 = 87.6%
      close,     own columns   23/27  = 85.2%
      direction, all pairs    134/137 = 97.8%
      direction, own columns   26/27  = 96.3%
      set-token scaffolding pairs         110
  **The own-column number is WORSE.** A 0/1 token indicator
  tied to its own set survives generation better than a
  clinical column does, so the headline is KINDER than the file
  the operator receives - the opposite of what the INVERTED
  precedent led me to expect. The change would not have cleared
  the bar, and proposing it would have been the
  move-the-denominator-to-clear-a-bar mistake while failing on
  its own terms. One measurement, no code, idea dead.
- **SO THE CRITERION STILL JUDGES THE TOTAL, DELIBERATELY, AND
  THE STRICTER NUMBER IS PRINTED BESIDE IT.** The bar was SET
  against the total, so switching the denominator moves a
  release verdict rather than fixing a bug - a product decision
  and not a bug fix, which this file's own rule about printing
  a chosen bar beside the recorded default exists to keep
  honest. What WAS a defect: the own/scaffolding breakdown was
  appended to the `direction kept` line and NOT to `close`,
  which is the criterion where it matters most and the one that
  FAILS. Now on both, with a run that has no scaffolding still
  adding nothing - a note that fires on everything is noise.

- **A CITY NAME AND A RACE CATEGORY ARE THE SAME SHAPE, SO THE
  HEADER IS THE ONLY TELL THERE IS (2026-11-06, PHI-5).** Both
  are short alphabetic strings, both are CONSTANT within a
  patient, both hold a handful of values across a cohort - every
  test that separates a person's name from a drug's gives the
  same answer for `city` and for `race`, which is why the name
  detector had to be taught not to flag gender, race and
  ethnicity. So the geography rule is header-driven, the finding
  SAYS it is header-driven and names its own limit (a renamed
  city column is not caught), and the value shape is asked only
  to be CONSISTENT with a place name - its job is to refuse
  NUMBERS under a city-ish header, not to insist on a single
  word.
- **AND THE REMEDY IS DROP, BECAUSE NO DEFINED COARSENING
  EXISTS.** Safe Harbor names the first three digits of a postal
  code as a publishable unit and names NO equivalent for a place
  name, so the honest options are to remove it or to map it to
  its state - and a city-to-state table is data this project
  cannot carry or keep correct, so shipping one would be a guess
  wearing a lookup. Said in the module rather than implied by a
  silent drop.
- **`ethnicity` ENDS IN `city`, AND THE REMEDY IS DROP.** So
  does `capacity`, `toxicity`, `specificity` and `electricity`.
  A loose header pattern would have DELETED ethnicity from a
  clinical extract - the same class as the first name detector
  flagging gender, race and ethnicity, where `apply` would have
  dropped three covariates. The pattern is underscore-aware
  (`\b` does not see an underscore - the lesson the ZIP tell
  taught, and it matters more here), tested against all of them
  before the detector was written, and the loose version turns
  three checks red.
- **AND AN EXPLICIT STATE EXCLUSION WAS DEAD CODE THAT COULD
  ONLY DO HARM.** A state is not PHI under Safe Harbor, which
  asks for subdivisions SMALLER than a state - so I wrote a
  `_STATE_HINT` exclusion. The mutation removing it failed NO
  check, because `_GEO_HINT` never matched `state` to begin
  with: the rule is achieved by `state` simply NOT BEING IN THE
  TOKEN LIST. Worse, the exclusion would have caused a FALSE
  NEGATIVE on `city_state` - a combined "Pasadena, CA" field,
  common in flat exports and genuinely PHI because it contains
  a city. Removed, the real mechanism asserted instead, and
  restoring the exclusion now turns a check red. A mutation
  that finds nothing means the check is not load-bearing OR the
  code is dead; here it was the code.
- **AND THE FIXTURE CAUGHT MY OWN VALUE PATTERN BEING TOO
  NARROW.** `_PLACEISH` forbade a comma, so `Pasadena, CA`
  scored 0.0 and the combined field was missed - a positive
  control failing on the first run, which is what controls are
  for. Commas allowed; numbers still refused (`1234`, `90033`,
  `Apt 4B` all fail the pattern), so the gate still does the
  one job it has.

- **AND THE GEOGRAPHY RULE READ CLEAN ON THE REAL EXTRACT,
  WITH THE THREE COLUMNS A LOOSE PATTERN WOULD HAVE DELETED
  CONFIRMED CLEAR (2026-11-06).** On the 55,428-row, 44-column
  extract: **GEOFINDINGS NONE**, the four pre-existing findings
  unchanged (one identifier, two service dates, one age), and
  **gender / race / ethnicity all CLEAR** - stated as words, not
  digits. 39 clear plus 1 out of scope plus 4 findings balances
  against 44 exactly. That is the specific false positive the
  FIRST name detector committed on this same file, where `apply`
  would have dropped three covariates, so a clean read here is
  evidence rather than a formality.
- **AND `$(git rev-parse HEAD)` INSIDE `cd ~/ai_copilot`
  RESOLVED ANOTHER REPOSITORY'S HEAD.** The fetch command was
  built in the bridge directory's shell, which is its own git
  repo, so it asked the data machine to download
  `35791b1a...` - a different project's commit - while I had
  just printed `fa2afb13...` two lines above. Caught only by the
  two SHAs disagreeing in one screen. This file already says to
  prefer a LITERAL string on that terminal to anything that has
  to be substituted first; the new part is that the
  substitution can be silently correct for the WRONG
  repository. Compute a sha where the repo is, or type it out.

- **MULTI-TABLE INTAKE EXISTS, AND THE KEYS COME FROM THE
  VALUES (2026-11-04, `synthkit/multitable.py`).** The fitted
  path has always been single-table, so an operator with
  patients, visits and labs had to join them by hand first -
  and a hand join is where row counts and missingness quietly
  change before this tool ever sees them. `synthkit schema`
  reads several tables, finds each primary key, finds the
  links by VALUE OVERLAP, and reports referential integrity.
  Three fixtures: a star, a chain, and one whose headers lie.
- **A CONTINUOUS MEASUREMENT WAS MADE A PRIMARY KEY, AND IT
  WOULD HAVE LINKED TABLES ON IT.** 300 normal draws are
  unique on every row and present on every row, so a
  uniqueness test calls the column a key - and the fixture
  caught it on the first run. Identifiers are strings or WHOLE
  numbers; a value with a fractional part is something
  somebody measured. Integers are deliberately left alone,
  because a patient number usually is one and refusing those
  rejects the most common key there is.
- **ORPHANS ARE KEPT, NOT CLEANED AWAY.** A real extract has
  visits whose patient was filtered out by somebody's cohort
  query. The star fixture plants 20, the source measures
  **97.8%**, and the generated set reads **97.8%** - with the
  orphans pointing at invented `ORPHAN*` keys that belong to
  nobody, so an orphan is an orphan rather than a real
  identifier that happens not to match. Emitting 100% where
  the source had 97.8% would invent a cleanliness the data
  never had and make every downstream count that depends on
  orphans wrong.
- **INTEGRITY IS BY CONSTRUCTION, NOT BY REPAIR.** The parent's
  key is invented first and every child key is DRAWN from keys
  that already exist, so there is no step at which a child can
  point at a parent that is not there - the same reason the
  size identity is enforced by copying rather than by drawing
  twice. And no real key is republished: an MRN is a direct
  identifier, and a file carrying one has republished it
  however synthetic everything around it is.
- **WHAT IS NOT DONE, NAMED: the tables are detected and
  re-keyed, not FITTED TOGETHER.** Each still goes through the
  single-table pipeline alone, so a cross-table relationship -
  a lab value that depends on its visit's length of stay - is
  not modelled. That is the rest of M3.
- **AND THE MISSES THAT ARE NOT IN THE FILE CAN BE REACHED
  THROUGH THEIR SET.** The biggest single group of close
  misses on the real extract is a `set__has__token` against an
  ordinary column - 5 of 17 - and the indicator is NOT in
  generated.csv, so no rearrangement of the written file can
  touch it. But it is DERIVED from the set column, which IS
  written: moving the whole set string as a unit moves the
  indicator with it, which is the rearrangement already
  measured to land within 0.001 of an ask while leaving every
  published token's share untouched. On a fixture with the
  fade planted - the set column shuffled ALONE, because
  shuffling the frame permutes every column together and
  preserves every correlation - close goes **5/7 to 7/7** and
  direction **5/7 to 7/7**, with `set token shares` unmoved
  and nothing BROKEN. The size partner travels with the set,
  so the declared identity survives a repair aimed at a token.
- **AND ON THE REAL EXTRACT THE SET ROUTE WAS MEASURED
  HARMFUL, SO IT IS OFF BY DEFAULT.** It reached all 5 token
  misses and took close from **46/57 to 32/57**, with
  direction 57/57 to 56/57. Moving one set moves every OTHER
  token on it, and a real set column carries hundreds where
  the fixture's carries eight - five repairs cost about
  fourteen pairs. The fixture said 5/7 to 7/7 and the extract
  said the opposite, which is the measured-at-one-scale
  lesson in a new place. The capability stays behind
  `--via-sets` with that measurement in its help text, the
  default SAYS what it did not do and why, and the whole gate
  is re-measured either way. **The whole-gate re-measure is
  what caught it** - a repair judged only on the criterion it
  aimed at would have shipped a 24-point regression as a fix.
- **TWICE IN ONE NIGHT A CHECK ASSERTED THE MESSAGE INSTEAD OF
  THE DECISION.** The default-off check looked for the
  sentence explaining that the set route was skipped - and
  that sentence prints from a SEPARATE branch, so a mutation
  turning the route back on passed while the sets moved
  anyway. The observable is that no set MOVED, and asserting
  that catches it. Same fault as the bench's refusal check
  earlier.

- **WHY THE GATE COUNTS 137 PAIRS AND A RE-MEASURE COUNTS 57:
  ESTABLISHED, NOT GUESSED (2026-11-04).** The run that wrote
  the blueprint expanded its set columns into `__has__`
  indicators with the cap it was given; any later `compare()`
  re-derived them through `prepare()`, which applies the
  MODULE CONSTANT `EXPAND_CAP = 24`. Measured on the real
  extract by `close_gap.py RUNDIR --surface SRC.csv`:
      published tokens   active_drugs 529, conditions 1050,
                         drug_routes 53, procedures 195
                         (1,827 - the inventory's own count)
      prepare() rebuilds at most 24 per set column
      indicators the blueprint's relationships NAME
                         that CANNOT be rebuilt:  37
  Every pair touching one of those 37 was silently skipped, so
  the recorded percentage and any re-measured one were over
  DIFFERENT PAIR SETS - which means the earlier `120/137
  before, 47/57 after` was two questions wearing a before and
  an after.
  **CORRECTED WITHIN THE HOUR: the cap is A cause and NOT the
  whole gap.** This entry first said it was. With the fix the
  real extract's re-measure goes **57 to 69**, not to 137, so
  the cap accounts for twelve pairs of about eighty. The rest
  is the second half of the same question and is named below.
- **THE FIX IS TO MEASURE ON THE SURFACE THE RUN WAS MEASURED
  ON.** `prepare(vocab=...)` takes the vocabulary a caller
  already knows and skips the cap for it; `compare()` passes
  `blueprint_vocab(bp)`, read straight off the artifact. The
  cap exists to choose which tokens become SEARCH columns
  when nobody has decided yet - measuring an existing run is
  a different job. Published tokens have ALREADY cleared the
  k floor, so this changes what is measured and not what is
  disclosed. Verified where the cap binds: 24 rebuilt by
  default against 40 with the vocabulary.
- **AND NO FIXTURE HERE CAN REACH THE CONDITION, which is
  said rather than papered over.** For `compare()` to relate
  MORE pairs with the vocabulary, the blueprint must NAME an
  indicator the default cap cannot rebuild - and on a fixture
  it cannot, because discovery and the re-measure apply the
  SAME cap, so whatever discovery named is rebuildable. The
  real extract names 37 because the run that wrote its
  blueprint expanded 1,827 tokens. The verification runs on
  the machine that has the shape.
- **AND THE REST OF THE GAP IS THAT THE ENGINE NEVER EMITS
  WHAT THE BLUEPRINT PUBLISHES.** A pair is measured on BOTH
  tables, so a token the generated file never carries gives an
  all-zero indicator there and the pair is skipped however the
  source is expanded. Measured on the real extract:
      column        published   in source   in OUTPUT
      active_drugs        529         824         477
      conditions         1050        2721     **800**
      drug_routes          53          71          48
      procedures          195        2376         169
  **`conditions` emits exactly 800 - the engine's token cap -
  against 1050 that cleared k**, so 250 published condition
  tokens appear in no generated row at all. The number was
  read back spelled ("eight zero zero") because it is the one
  that decides. The other three sit under the cap, so their
  smaller losses are a separate and much smaller effect.
- **THE ROUNDING HYPOTHESIS WAS REFUTED BY ITS OWN COUNT.**
  The independent per-token draw places `round(p x n)` rows,
  which would erase any token whose expected count is under a
  half - deterministically, on every seed. Measured: **ZERO**
  published tokens round to zero on this run. A clean
  negative, and the reason to count before fixing.
- **SO THE TWO HALVES DISAGREE ABOUT THE VOCABULARY.** The
  blueprint publishes everything that clears k and the engine
  models a capped subset, so the contract promises 250 tokens
  the output does not contain. That is this file's recurring
  two-halves fault in a new place, and the choice between
  fixing it by RAISING the engine's cap (better coverage,
  more encoded width - the record has 1,601 dimensions at
  800) and by publishing only what the engine can emit
  (cheaper, and makes the artifact honest) is a product
  decision with a cost, not something to settle quietly.
  Note it also contradicts, for THIS column, the earlier
  reading that "nearly every token clearing the k floor is
  already in by 800".

- **THE CLOUD PATH IS A REFUSAL BEFORE IT IS A TRANSFER
  (2026-11-04, `synthkit/cloud.py`, `docs/AWS.md`).** A
  hospital's own AWS account is a legitimate place to run
  this; what it must never become is a route by which records
  leave the account they were already in. So the compute goes
  to the data: `fetch_source` brings a source IN and there is
  NO function that sends one anywhere - that absence is the
  design, not an omission.
- **ONLY NAMED ARTIFACTS LEAVE, AND THE LIST CANNOT BE
  EXTENDED BY A FLAG.** `publish()` refuses anything off the
  allow-list BY NAME. Two refusals matter most: the SOURCE
  EXTRACT, which sits in the same directory as the artifacts
  during a run so "upload the run directory" is the natural
  mistake; and **`key.csv`**, because the key is the
  full-fidelity file the operator KEEPS and publishing it
  would defeat the whole key/released arrangement. A request
  naming both an allowed and a refused file is refused WHOLE -
  a partial upload that looks complete is the
  silent-difference fault this file refuses everywhere else.
- **THE STORE IS A SECOND REAL IMPLEMENTATION, NOT A MOCK.**
  The whole upload path runs against a local directory in the
  checks, so what they exercise is this project's logic rather
  than somebody else's SDK behaving as its author imagined. A
  cloud path tested only against a mock is tested against the
  mock. Three mutations - allowing the source, honouring a
  mixed request partly, and adding `key.csv` to the list -
  each turn the suite red.
- **AND IT HAS NOT RUN AGAINST A LIVE AWS ACCOUNT, SAID IN THE
  MODULE, IN THE DOC AND IN THE TOOL'S OWN OUTPUT.** A
  deployment is not measured until somebody deploys it, which
  is this file's oldest rule applied to infrastructure. The
  container pins `python:3.14-slim` because this project has
  twice been bitten by testing against a different interpreter
  than the one that runs, runs as non-root, excludes `*.csv`,
  `data/` and `run_*/` from the build context - an extract
  swept into a layer travels with every pull of the image -
  and carries the NET, so a deployment can prove the tool
  works before it is pointed at an extract.
- **AND THE RUN NOW COUNTS WHAT IT PUBLISHED AND NEVER
  EMITTED.** `set token shares` skips a token absent from the
  generated frame with a `continue`, so a token the engine
  never emits was invisible to it - not a share that drifted,
  a share with nothing to compare. `set_tokens_never_emitted`
  counts them and NAMES the column with examples, because a
  reader told "45/45 token shares within 0.05" and not told
  that 250 published tokens appear in no generated row has
  been given a pass on a question nobody asked.
- **AND A CLAIM I MADE ABOUT THAT CRITERION WAS WRONG, CAUGHT
  BY ITS OWN CHECK.** The first version said `set token
  shares` is BLIND to a missing token. It is not, where the
  indicator column exists: the share reads zero, the delta is
  the whole source share, and the criterion counts it - 7 of
  8 on the fixture. It was blind on the real extract only
  because the column was never BUILT, `prepare()` having
  rebuilt the vocabulary under EXPAND_CAP. **With the
  vocabulary now read off the blueprint those columns exist,
  which makes the criterion STRICTER - and means it was
  previously FLATTERED by the tokens it could not see.** A
  future real run should be expected to read worse on `set
  token shares` for that reason, and that is a better number,
  not a regression.
- **AND THE VOCABULARY HAD TO BE BOUNDED BEFORE IT REACHED A
  REAL RUN - MY OWN FIX WOULD HAVE KILLED ONE.** Expanding
  every published token builds one indicator column per token
  on BOTH frames: 1,827 columns over 56,065 rows is **1.64 GB
  for the pair**, the same order as the allocation that once
  died inside the denoising noise. Caught by arithmetic
  before it ran, not by a crash on the machine that holds the
  extract. The set is now the UNION of what the pair criteria
  NAME - the 37 that could not be rebuilt, which was the
  whole point - and the first `EXPAND_CAP` by published
  order, so the token-share criterion keeps the surface it
  already had. About 133 columns, 119 MB for the pair. A
  measurement that cannot run is worth less than one that is
  slightly narrow, and what it leaves out is named.

- **`--help` RAN A FIVE-SEED FIXTURE DEMO AND PRINTED A
  RESULT (2026-11-04).** `latent_challenger.py` picked its
  subcommand from `sys.argv[1]` and let anything it did not
  match fall through to the DEFAULT - so `--help`, and any
  bare option, spent minutes fitting a planted fixture and
  then printed numbers, which read as an answer to whatever
  the caller actually asked. A command that does something
  plausible instead of refusing is this file's oldest failure
  shape, and it cost real time before it was noticed. `--help`
  prints the subcommands and exits 0; an unknown one STOPs by
  name, lists what is accepted, and exits non-zero.
- **AND THE CHECK FOR IT HAD TO NOT HANG.** Under the old
  behaviour the probe runs the demo, so a straightforward
  `subprocess.run` check would take MINUTES and then blow the
  suite's timeout - a check that hangs is worse than one that
  cannot fail, because it takes the whole suite with it. The
  timeout is short and expiring it is REPORTED as the fault
  it is, so the mutation turns three checks red in seconds.
- **THE TOKEN CAP IS CONFIRMED AS THE THING THAT BINDS.**
  `provenance.json` for the run of record reads `"token_cap":
  800` and the output carries exactly 800 `conditions` tokens
  against 1,050 that cleared k. The run also records 55,428
  rows and 800 patients, and NOT the seed - which is why a
  raised-cap run must be paired with a CONTROL ARM AT THE
  SAME SEED, run by the same hand, rather than compared
  against the deliverable. That pair is queued, not yet run.

- **I MEASURED A RUN AGAINST AN EXTRACT THAT NEVER PRODUCED
  IT, FOR A WHOLE SESSION, AND EVERY NUMBER LOOKED
  REASONABLE (2026-11-04).** `yesha_tidy\tidy_visits.csv` has
  **23,092 rows**; the run of record reports reading
  **55,428**. Nothing compared the two, so every
  `repair_inversions --close` and `close_gap --surface`
  invocation that night compared `run_deliverable`'s
  generated file against a DIFFERENT, smaller extract. The
  outputs were plausible in every respect - that is the
  danger, and it is the oldest shape in this file wearing a
  file path.
- **WHAT FALLS, AND WHAT DOES NOT.** The source-dependent
  numbers are withdrawn: the re-measured pair counts (57, 69),
  the "37 indicators cannot be rebuilt" count, and every
  before/after gate column from those repair runs. What
  STANDS is what came from the run's own artifacts: the
  close-gap attribution (17 misses at **87.6%**, zero
  bound-explained, 4 repairable, 14 fades against 3
  overshoots, the mechanism groups) read `fidelity.json`
  alone; and "**conditions publishes 1,050 and the output
  carries exactly 800**" took one number from the blueprint
  and the other from the generated file, neither from the
  source. State which artifact a number came from, or a
  wrong file silently reclassifies all of them.
- **THE GUARD IS THE ROW COUNT THE RUN ALREADY RECORDED.**
  Both tools now read `provenance.json` and REFUSE a source
  whose row count differs, naming both numbers. It costs
  nothing and it would have caught this on the first
  command. Watched red against a planted mismatch.
- **AND THE ENGINE'S k SCREEN IS NOT THE BLUEPRINT'S.**
  Measured on the smaller extract while chasing the cap: of
  2,543 `conditions` tokens found, **288 clear k and 288 are
  modeled** - the cap of 800 does not bind there at all. The
  blueprint for the 55,428-row run publishes 1,050. So the
  two halves screen on different populations (the engine on
  its 85% training split), and "how many tokens clear k" is
  not one number. The cap question has to be re-asked on the
  right file.

- **THE MULTI-TABLE INTAKE MET REAL OMOP TABLES AND GOT IT
  RIGHT FIRST TIME (2026-11-04).** `synthkit schema` on the
  data machine's own `visit_occurrence`, `drug_exposure` and
  `procedure_occurrence`:
      visit_occurrence       23,091 rows x 22 cols
                             key visit_occurrence_id
      drug_exposure         111,545 rows x 23 cols
                             key drug_exposure_id
      procedure_occurrence   97,215 rows x 17 cols
                             key procedure_occurrence_id
      drug_exposure.visit_occurrence_id
          -> visit_occurrence   111,545/111,545   100.0%
      procedure_occurrence.visit_occurrence_id
          -> visit_occurrence    97,204/97,204    100.0%
  Every key is the canonical OMOP one and was found from the
  VALUES, not the names. No false link was reported, and no
  link to `person_id` was invented - there is no person table
  in the set, and a column with no parent table is not a
  foreign key however it is named. The 11 `procedure_occurrence`
  rows between 97,215 and 97,204 carry a BLANK visit id and
  are counted as absent rather than as orphans, which is the
  present-versus-missing distinction this file draws for keys
  deliberately and the opposite of the one it draws for sets.

- **AN ALL-EMPTY COLUMN SIZED THE WHOLE FRAME, AND THE ENGINE
  CRASHED ON A REAL EXTRACT (2026-11-04).**
      ValueError: Length of values (21306) does not match
      length of index (23091)
  `n` is the REQUESTED row count. Generating patients
  hierarchically makes the real count the sum of the DRAWN
  visit counts, and those need not add up to the request -
  the visit-count grid is k-screened and its tail CLIPPED, so
  a heavy-tailed cohort undershoots. Every column was built
  at the drawn length except the all-empty one, which used
  `n`; `pd.DataFrame` took the union and `frame.insert` of
  the group column then failed. It needs BOTH an all-empty
  column AND a draw that misses the request, which is why it
  survived every run until one extract had both.
- **AND THE FIXTURE HAD TO PLANT BOTH, including one that is
  easy to get wrong.** A column of empty STRINGS is not
  detected as empty - `""` reads as PRESENT, which is the
  distinction this file draws deliberately for sets - so the
  fixture uses genuine NaN, plus visit counts heavy enough
  that the draw undershoots (515 against 660 asked). Watched
  red: the old code raises the exact ValueError.

- **THE TOKEN CAP WAS RAISED AND IT CHANGED NOTHING - THE CAP
  IS NOT THE LIMITER ON THIS EXTRACT (2026-11-04).** Two arms,
  same seed, same file, differing only in `--max-tokens`:
      cap  800   conditions 2,543 found, 288 clear k, 288 modeled
                 encoded frame 722 dimensions
                 seed 5: sign 263/309 close 224/309 INVERTED 11
                 nn-ratio 1.33, k-blur 36.3%
      cap 1200   IDENTICAL IN EVERY DIGIT
  Two arms that agree to the digit are one arm - and here that
  is the correct conclusion rather than a bug: the parameter
  genuinely has no effect in this range. **The limiter is the
  k screen on the engine's 85% TRAINING SPLIT: 288 of 2,543
  `conditions` tokens clear a k=10 patient floor there.** No
  cap setting can emit a token the k rule will not publish.
- **AND THE 1,050-AGAINST-800 OBSERVATION BELONGS TO A COHORT
  WHOSE SOURCE NO LONGER EXISTS.** It came from
  `run_deliverable`, which recorded reading 55,428 rows; the
  file at that path today has **23,092**. The extract was
  re-pulled or re-filtered at some point, so that run's gate
  numbers describe a cohort that cannot be re-measured, and
  no new run is comparable to it. Any future claim about
  "the real extract" has to say WHICH one.
- **WHAT WOULD ACTUALLY RAISE EMISSION, and none of it is a
  cap.** Lower k - a privacy decision, not an engineering
  one. Or screen on the FULL cohort rather than the training
  split, which would leak holdout information into the
  published vocabulary and is the thing the split exists to
  prevent. Or recover the original extract. The two halves
  screening on different populations - blueprint on the whole
  cohort, engine on its split - is the real disagreement, and
  it is a design question rather than a setting.

- **THE 137-VERSUS-57 ANSWER WAS AN ARTIFACT OF THE WRONG FILE,
  AND THE REAL ANSWER POINTS THE OTHER WAY (2026-11-04).** A
  whole session established that the gate's pair surface
  exceeded a re-measure's because `EXPAND_CAP` stopped the
  token indicators being rebuilt. Every number behind that was
  measured against a 23,092-row extract that never produced the
  run. `find_extract.py` found the real one by ROW COUNT -
  `phase2_800\tidy_visits.csv`, accepted by the provenance
  guard, which is what CONFIRMED it rather than my reading of
  an OCR'd digit. Re-measured there, the pair sets diff by name
  as:
      recorded by the run       196
      related by a re-measure   218
      in BOTH                   196
      RE-MEASURE ONLY            22   all on indicators the
                                      blueprint NAMES
      RECORDED ONLY            NONE
  **The re-measure is a strict SUPERSET: not one recorded pair
  is lost.** A cap removes columns, so it cannot add pairs - the
  SIGN of the difference refutes the whole story in one line of
  arithmetic, with no need to look at a column. What remains is
  the opposite finding: the RUN's own fidelity omitted 22 pairs
  on indicators its own contract names, which by the
  classifier's definition both surfaces should hold. The
  candidate is the token vocabulary the run was built with
  predating the `blueprint_vocab` change - NAMED as a
  candidate, because naming a cause I have not measured is the
  fault that produced this entry in the first place.
- **AND A COUNT CANNOT ANSWER A QUESTION ABOUT WHICH.** Every
  account of this gap for a week was a story told from two
  sizes. `close_gap.py --pairs` diffs the sets BY NAME and
  buckets each difference three ways - an indicator the
  blueprint NAMES (a defect, since both surfaces should hold
  it), an indicator it does NOT name (a re-measure deriving its
  own columns), or plain columns (neither explanation covers
  it). Those are three different findings and one number
  reports none of them.
- **`close_gap.py` HAD NO CHECKS, in a tool whose numbers were
  quoted in a report.** The `peek.py` fault again. Twelve now,
  each watched red, and the load-bearing one is the provenance
  guard - which must refuse BEFORE the expensive `compare()`,
  so the check asserts that no re-measure output appears.

- **THE M0 CLOSE CRITERION CLEARS ON THE REAL EXTRACT, AND THE
  MEASUREMENT HAS NO SEED TO ARGUE ABOUT (2026-11-04).**
  `repair_inversions --close` against run_deliverable on the
  correct source, one file, one column reordered, same
  denominator both sides:
      direction kept  145/148 = 98.0%  ->  146/148 = 98.6%
      close           130/148 = 87.8%  ->  133/148 = 89.9%   FIXED
  The bar is 87.9%, so close goes FAIL -> PASS. The three
  headline figures were re-read through the word channel before
  being written down, because the digits channel off that
  screen is not trustworthy. Five further misses are on TOKEN
  INDICATORS and are left alone: reaching them means rearranging
  the set column they came from, which moves every other token
  on the same set and was measured harmful (close 46/57 ->
  32/57).

- **NO FIXTURE HERE PRODUCES AN IN-FILE CLOSE MISS, AND THAT IS
  WHY A CHECK PASSED BY NOT RUNNING (2026-11-04).**
  `smoke_patterndial`'s `--close` fixture asserted that
  generation would miss close on some pair and then use it. It
  does not: four clean gaussian columns come out with a worst
  delta of 0.095 against the 0.2 cut. Five shapes were measured
  before anything was concluded - a smooth RING of colinear
  columns (0 misses), a SPARSE pair at 25% presence (0), a
  HEAVY-SKEW pair whose tail the k rule takes (0), and a SET
  fixture over nine data/seed arms. **The set shape is the only
  one that misses close at all** - which is exactly where the
  real extract's drift concentrates, so the negative is
  informative - and every miss it produces is on a TOKEN
  INDICATOR, which is scaffolding dropped before generated.csv
  is written and therefore unreachable by a repair. Adding one
  plain column riding the count removed the misses entirely,
  which is this file's own don't-add-a-column rule.
  So the drift is PLANTED now - the child shuffled alone, the
  way the restore fixture beside it already worked - and the
  check cannot be green for want of a fault.

- **THE DIAL MATRIX WAS GIVING THREE WRONG VERDICTS, AND SIX OF
  ITS CHECKS WERE VANISHING RATHER THAN FAILING (2026-11-04).**
  Found by running the net rather than by anything asking. Six
  checks sat AFTER the suite's own `if FAIL: sys.exit(1)`, so
  the six real failures below made them disappear - and a
  vanished check reads exactly like a passing one. This file
  already records that shape from `smoke_patterndial`; the exit
  is at the END now, after every check has had its turn.
- **A DIAL CAN BE WIRED AND WEAK, AND THAT IS NOT A NO-OP.**
  `persistence` asked x0.40 of a base 0.313 and landed 0.263 -
  x0.84, a real move but smaller than the property's own
  tolerance - so the matrix reported THE DIAL MOVED NOTHING.
  Unwired and under-delivering take different fixes and only
  one of them is in the plumbing. The `elif not hit` branch
  preceded `elif short`, so SHORT was unreachable for exactly
  the dials that need it most; a second much smaller
  did-it-budge-at-all threshold separates them, and the row now
  states asked beside landed.
- **A DIAL THAT REDRAWS THE SAMPLE NEEDS A FLOOR MEASURED UNDER
  ITSELF.** `patients.count` read `x.spread +0.12 LEAKED` on
  every run. The global noise floor re-seeds the UNDIALLED run
  at the BASE patient count - the wrong conditions entirely for
  a dial that draws a different number of people, since spread
  is less stable at a different n. So the floor was too tight
  and the resampling it exists to absorb came back as a defect:
  a seed written up as a finding, one level below where this
  file already records it. Those arms re-seed THEMSELVES now,
  always, not on `--noise-seeds` - without it the verdict is
  not conservative, it is wrong. CLEAN afterwards; every other
  arm still pays nothing.
- **AND A NO-OP WITH A CAUSE IS NOT THE SAME FINDING AS AN
  UNEXPLAINED ONE.** `y<-x.strength` moves the pair not at all
  - +0.966 against +0.966, identical, through the dials' own
  entry point, with no problem reported - because discovery
  orients this fixture into a CYCLE and the live REVERSE edge
  carries the pair. The reverse-edge explanation was attached
  only to SHORT, so the one verdict that needed it most printed
  a bare number that reads as a dead dial. It is on NO-OP too
  now, named from the blueprint rather than assumed.
- **AND THE RECORDED 3 sd DESTRUCTION DOES NOT REPRODUCE.**
  **WRONG - SUPERSEDED 2026-11-05, and the error was the
  VENV.** Everything in this bullet was measured on Python 3.10
  with pandas 2.3. On the stack the data machine installs -
  3.14 with pandas 3.0 - the shift DOES leak (`x~y.pair +0.68`)
  and the original check was right. Kept here because the shape
  of the mistake matters: a reproduction attempt on the wrong
  library stack is not a reproduction attempt, and I used it to
  overwrite a correct check. The
  matrix's own comment recorded parent~child Spearman falling
  0.968 -> 0.313 at a 3 sd shift and asserted that row was
  EXPECTED to read LEAKED. Re-measured on this fixture: the
  shift LANDS - x mean 50.3 -> 84.9, the full 3 sd - and the
  pair is UNHARMED, +0.966 dialled against +0.966 undialled. A
  rebuilt fixture reads +0.986 -> +0.971, also inside
  tolerance. Discovery orients both into a cycle, where the
  refinement sweep re-ranks the column onto the multiset the
  marginal drew, so the arrangement survives the shift and the
  relationship with it. The check now asserts CLEAN **and that
  the shift landed**, because a CLEAN produced by a dial doing
  nothing prints identically and is the whole fault class this
  matrix exists to separate.
- **AND THE LEAK DETECTOR'S ONLY POSITIVE CASE HAD EVAPORATED
  WITH THE ENGINE'S IMPROVEMENT.** It rested on that 3 sd row
  leaking. A detector whose positive case can disappear without
  anybody editing it is one that quietly stops being able to
  fail - so the classification is a pure function taking
  planted inputs now (`classify_move`), asserted on all four
  branches: above the floor on an unrelated column is a LEAK,
  the same move on a CHILD is PROPAGATED, under the floor is
  NOISE either way, and with NO floor measured nothing is
  absorbed.
- **AND PROPAGATION IS INFORMATION, NOT A VERDICT.** It was
  folded into the note only under CLEAN, so a dial that
  under-delivered AND propagated reported the propagation
  nowhere - the same value-assigned-downstream-of-its-reader
  fault the block one branch over already carries a comment
  about.

- **THREE CANDIDATES FOR THE 22 PAIRS DIED THE SAME NIGHT, ALL
  BY MEASUREMENT, AND THE FOURTH GUESS IS NOT BEING MADE
  (2026-11-04).** With the pair sets diffed by name - 196
  shared, 22 extra, NONE lost - the question became why the RUN
  omitted 22 pairs on indicators its own blueprint names. The
  candidate was the token vocabulary that run was built with,
  so `close_gap --pairs` was taught to ask where each extra
  pair's token sits in its column's PUBLISHED order:
      rank  6   active_drugs  lidocaine
      rank 11   active_drugs  lidocaine topical
      rank 13   procedures    ICD10-PCS!3EQU3BZ
      ...
      29 token(s) sit INSIDE the cap
  **Every one is well within a capped expansion's reach.** A
  cap of 24 per column would have expanded all of them, so
  neither the cap nor the vocabulary accounts for the
  difference. Both candidates are now refuted by numbers rather
  than argued about.
  The extra pairs are CROSS-SET-COLUMN token pairs -
  `active_drugs__has__metoclopramide <- drug_routes__has__Oral`,
  `active_drugs__has__tacrolimus <- drug_routes__has__NG-tube` -
  the drug-implies-route family this file already tracks.
- **AND THE THIRD CANDIDATE - ORIENTATION - COST TWO LINES AND
  ALSO DIED.** The diff keys on (child, parent), so a pair the
  run recorded the other way round would read as missing from
  it and present in the re-measure, inflating the difference
  with pairs that are in both sets. Measured: ignoring
  direction the re-measure still adds **22**, so none of the
  difference is an orientation artifact - the extra pairs are
  absent from the recorded set in BOTH directions.
  So the characterization is now: **22 cross-set-column token
  pairs, on tokens at published ranks 6 to 13, absent from the
  run's own recorded fidelity in both directions, on indicators
  the blueprint's own relationships name.** Why the run related
  none of them is OPEN. Per this file's no-third-guess rule it
  earns a diagnostic in the run directory rather than a fourth
  hypothesis stated as a finding - and the three that died were
  each cheap to kill, which is the argument for measuring a
  candidate the hour it is thought of.
- **AND THE FIRST READING OF THAT CENSUS WAS A WRONG FIELD
  NAME - THE FIFTH TIME.** It read `token` where a list
  marginal writes `value`, so every lookup returned None and
  all 29 indicators were reported as "not in the published list
  at all" - which reads as a blueprint declaring relationships
  on tokens it does not publish, a far more alarming finding,
  and I was one step from reporting it. What refuted it was the
  CONTRADICTION: the names were `metoclopramide` and `Oral`,
  whose published shares are on record in this file, so "not
  published" could not be true. The prescribed fix is applied -
  the key is asserted against a LIVE built blueprint, so a
  rename breaks the checks rather than somebody's report.
- **AND THE CHECK WRITTEN FOR IT COULD NOT SEE IT, TWICE.**
  First version asserted only that the FIELD exists and never
  went through the builder: restoring the bug produced ZERO
  failures. `token_ranks` is asserted against a PLANTED order
  map, so the step that BUILDS the map was the one piece with
  nothing on it - which is exactly where the bug was. Second
  version was SELF-FULFILLING: it took its token name FROM the
  map, so under the bug the map held `None`, `None` resolved,
  and the check passed. The token comes from the SOURCE data
  now and the mutation goes from zero red to two. Assert the
  observable, and take the input from somewhere the bug cannot
  reach.
- **AND AN INDEX-BASED SPLICE ATE TWO FUNCTION BODIES.**
  Factoring the builder out was done by slicing between two
  found offsets rather than by exact string replacement, and
  the slice removed `published_order`'s body, all of
  `token_ranks`, and the head of `pair_diff` - leaving a
  function whose only statement was a call to itself. It
  COMPILED. This file's anchor rule is about a prefix matching
  in the wrong place; index arithmetic is the same fault with
  no anchor at all. Exact string replacement with count==1,
  every time, including when refactoring.

- **THE CIPHER CHAIN RAN END TO END ON THE REAL EXTRACT, AND IT
  IS THE STRONGEST READING YET (2026-11-05, run_cipher_s0,
  800 patients, k 10, token cap 800, dial
  `condition_count~span_days=-0.60`).** Every figure below was
  read back SPELLED by the machine before being written down,
  because the digits channel off that screen turns a zero into
  an @.
      verify            43 column(s) untouched, 0 changed;
                        every column's own distribution
                        unchanged -> VERIFIED
      asked             -0.60
      landed            -0.24
      reachable band    [-0.284, +0.392], bounded by the FOUR
                        distinct values the dialed column holds
                        - THE ASK WAS OUTSIDE IT, and the dial
                        reached 85% of the floor that exists
      key-trained       +0.690
      released-trained  -0.055
      DENIED            +0.744
      recovery          EXACT, no surrogate
  **A vendor who TRAINS on the released file goes from +0.690
  to NEGATIVE skill - worse than predicting the mean - while a
  vendor who merely SCORES our rows returns the key answer
  already, because every predictor that target reads is
  untouched.** The arithmetic is self-checking: 0.690 against
  -0.055 is 0.745 and the tool reports 0.744, which is two
  numbers agreeing rather than one asserted.
- **AND THE SPAN_DAYS CEILING IS A PROPERTY OF THE COHORT, NOT
  A SAMPLER MISS.** This run's `span_days` holds FOUR distinct
  values where the earlier `run_dial2` held five, and the band
  moved with it - [-0.284, +0.392] against [-0.395, +0.445].
  Same mechanism, different cohort, and the dial states the
  bound with the distinct-value count that causes it, so a
  landed -0.24 against an asked -0.60 reads as an impossible
  ask rather than a failure. Quote the band whenever the dial
  magnitude comes up.
- **THE SNAPSHOT IS PART OF THE METHOD.** The run writes
  `generated.csv` and `cipher.json` per seed into one
  directory, so reading them while a later seed is still
  writing is the two-jobs-on-one-directory corruption this
  file already records. The complete seed-0 trio was COPIED to
  its own directory first and every verification ran there.
  `key.csv` and `generated.csv` came out byte-identical in
  SIZE, which is what a permutation of the same values looks
  like before anything is measured.
- **AND ROUTE B WAS EXERCISED ON THE REAL EXTRACT, WITH ITS
  OWN AGREEMENT CHECK FIRING (2026-11-05).** Pointed at
  `condition_count` - where the dialled column is a PREDICTOR
  rather than the outcome - the chain reads surrogate **0.108
  sd** and the returned score already **0.001 sd** from truth,
  and the tool says so in the right words: *nothing to recover
  here, because this target barely reads the column the dial
  moved, and that is the SAME fact as the protection number
  above, which is why they agree.* That is the conflation this
  file recorded being refused in practice: "too distorted to
  recover" and "nothing to recover" look identical in one
  number and take opposite actions.
  So both routes are now measured on real data, and the design
  rule holds exactly as recorded: dial the OUTCOME (denied
  0.744, recovery exact) and not a covariate (denies nothing,
  needs nothing).

- **THE WORKING TREE ON THE DATA MACHINE WAS GUTTED, AND I SPENT
  A DOZEN PROBES DISBELIEVING THE PROBES THAT SAID SO
  (2026-11-05).** `.venv\Scripts\python` stopped being
  recognized, `synthkit\*.py` matched nothing, and
  `scripts\run_all_smokes.py` was absent - while jobs using
  exactly those had run minutes earlier. Every `if exist` NO was
  CORRECT and I treated each one as a channel artifact, invented
  a new theory per contradiction, and at one point raised a
  false alarm about endpoint security quarantining files. The
  operator checked and there was nothing.
  **WHEN A PROBE DISAGREES WITH YOUR MODEL, RUN THE THING.**
  `python -V` settled in one command what twelve `if exist`
  probes could not, because an execution either works or names
  its failure, where a predicate about a path can only ever
  return yes or no. This file already says a capability that
  fails by returning something plausible is worse than one that
  crashes; here the plausible thing was my own disbelief.
- **A PATH WITH BACKSLASHES DOES NOT RENDER THROUGH THE SCREEN
  CHANNEL AT ALL.** `cd` alone, `type _run.txt` and
  `echo %USERPROFILE%` all came back as "(no output)" for a
  whole night, which reads as a failed command and is in fact a
  successful one whose answer cannot cross. Ask for the
  DIRECTORY NAME instead - `for %i in (.) do @echo AT-%~nxi`
  renders - or convert the separators before printing. Several
  "the command did nothing" conclusions tonight were this.
- **AND A SINGLE DIGIT DOES NOT RENDER EITHER.**
  `python -c "print(1)"` came back empty; the same command
  printing `STACK-OK` came back intact. The recorded rule is
  that the WORDS are the channel and the digits beside them are
  a convenience - that applies to a one-character success flag
  as much as to a measurement, so a probe's answer must be a
  WORD.
- **RECOVERY NEEDED NO INSTALL, BECAUSE THE NUMERIC STACK IS
  SYSTEM-WIDE THERE.** System `python` is 3.14.6 and imports
  numpy, pandas, sklearn and scipy; only the source tree was
  gone. A zipball re-extract at a known commit into a NEW
  directory, run with the system interpreter, restored the
  tooling without touching the broken copy, without pip, and
  without going near the real extract. Verified by importing
  pipeline / gate / generate and then by running the net there.
- **WHAT WAS NEVER AT RISK, AND IT IS THE PART THAT MATTERS.**
  The real extract and every run artifact survived untouched -
  `phase2_800\tidy_visits.csv` plus the labeled copy, the
  profile and the membership audit; `run_cipher_s0` with its
  key, cipher, blueprint, fidelity and generated files. Every
  number reported from tonight is backed by files still on
  disk. The cause of the tree's state is NOT ESTABLISHED and is
  not being guessed at: nothing run against it deletes files -
  the commands only wrote with `curl -o`, copied, and launched
  processes. Recorded as unexplained rather than attributed,
  because attributing it would be the fifth wrong theory in one
  session.

- **I DEVELOPED ALL NIGHT AGAINST THE WRONG VENV, AND THE DATA
  MACHINE'S NET SAID SO IN FOURTEEN FAILURES (2026-11-05).**
  `.venv` here is Python 3.10 with pandas 2.3.3 / numpy 2.2.6 /
  sklearn 1.7.2; the data machine installs **3.14.6 with pandas
  3.0 / numpy 2.5 / sklearn 1.9 / scipy 1.18**. `.venv314`
  ALREADY EXISTED with exactly that stack - the tool was there
  and I used the old one, which is this file's oldest rule
  ignored by its own author. Run the net under `.venv314`
  before believing anything.
- **AND FOUR DIAL VERDICTS DIFFER BY LIBRARY VERSION, SO LAST
  NIGHT'S "CORRECTION" WAS WRONG ON THE STACK THAT SHIPS.** On
  pandas 3 the engine behaves BETTER, and the ORIGINAL checks
  were right:
      x.shift@3sd      LEAKED, x~y.pair +0.68   (3.10: CLEAN)
      x.persistence    CLEAN, x0.42 asked x0.40 (3.10: x0.84)
      y<-x.strength    SHORT, x0.86 asked x0.40 (3.10: NO-OP)
      dials clean      7 of 9                   (3.10: 6 of 9)
  So the recorded 0.968 -> 0.313 shift destruction DOES
  reproduce, the persistence dial LANDS where asked, and the
  edge dial moves-but-does-not-land exactly as the check said.
  I rewrote four correct checks to match an old-library
  artifact and wrote a CONVENTIONS entry asserting the
  destruction did not reproduce. Both are corrected. The arms
  are now pinned on pandas 3+ via `staged()` and print a NOTE
  naming the stack on an older lane - the `gated()` pattern,
  because a skipped check that says nothing reads as a pass.
  WHAT SURVIVES from that work, because it holds on BOTH
  stacks: the per-arm noise floor for sample-changing dials,
  SHORT being reachable at all, propagation surviving a
  non-CLEAN row, the reverse-edge explanation on a NO-OP, the
  pure `classify_move` with planted inputs, and moving the
  suite's `sys.exit` to the END so six checks stopped
  vanishing.
- **A SCRIPT'S sys.path[0] IS ITS OWN DIRECTORY, AND
  `m0_gate.py` WAS THE ONE SCRIPT OF 143 THAT DID NOT ADD THE
  ROOT.** `python scripts/m0_gate.py RUNDIR` - the command in
  its own docstring - died with
  `ModuleNotFoundError: No module named 'synthkit'` on a source
  tree with no install, which is the documented no-git zipball
  flow on the data machine. It took TEN checks down with it and
  every one read as a gate defect. `peek.py` had the same fault
  in a worse form: its insert sat INSIDE one helper, so
  `peek surfaces` and the rest failed while the fixed caller
  passed - a path fix buried in a function fixes one caller and
  leaves the others looking like a different bug. **`python -c`
  hides this entirely, because the CWD is on the path**, which
  is why it never showed up by hand.
- **AND THE CHECK FOR IT WENT VACUOUS, THEN TOO BROAD.** First
  cut ran `peek.py RUNDIR` with no subcommand: peek printed its
  usage and exited without importing anything, so removing the
  insert broke nothing. Fixed by planting a published surface
  with no gap record, which is the branch that imports. Second
  cut then rejected ANY `ModuleNotFoundError` and failed on the
  bare interpreter lacking NUMPY - an environment limitation
  reported as the path defect. The fault being tested is that
  the ROOT is not on the path, so the observable is
  `No module named 'synthkit'` and nothing else. Both mutations
  red now, on an interpreter chosen BECAUSE it has no install.
- **AND THE WAITER MATCHED ITS OWN COMMAND TWICE IN A ROW, SO
  THE RULE NEEDS A MECHANICAL FORM.** This file already says to
  pick a marker that cannot appear in the command, because the
  bridge ECHOES the command above the output. That is advice,
  and advice is what I failed to follow twice inside five
  minutes: a loop grepping for `ALL GREEN` matched the
  `findstr /c:"ALL GREEN"` in its own echo and exited on the
  first iteration, reporting a running net as finished; the
  retry grepping for `suites,` did exactly the same.
  THE MECHANICAL FORM: **make the PATTERN differ from the TEXT
  it matches.** Search the remote file with a regex wildcard -
  `findstr /r /c:"ALL.GREEN"` - and match locally on the literal
  `ALL GREEN`. The command then CANNOT contain the string being
  waited for, so self-matching is impossible by construction
  rather than by remembering. Same trick for any marker: put a
  wildcard where the real text has a space or a bracket.

- **THE ZIPBALL DELTA IS NINETEEN NOW, AND IT STILL DECOMPOSES
  EXACTLY (2026-11-05).** The net reads **2,869** here on the
  shipping stack and **2,850** on the data machine's zipball
  extract at the same commit. This file's rule is that the gap
  is a number somebody accounted for, not a number nobody ran:
      smoke_buildid    26 -> 17   -9   needs git for the
                                       archive path
      smoke_gui       251 -> 248  -3   node absent (1),
                                       Chrome absent (2)
      smoke_packaging  28 -> 21   -7   no interpreter WITHOUT
                                       an install was found
      total                      -19   = 2869 - 2850, no residue
  The previously recorded delta of TWELVE was buildid plus gui;
  the seven are new and they are mine.
- **AND THE SEVEN WERE A GAP IN MY OWN CHECK, ON THE ONE MACHINE
  WHERE IT MATTERS.** The no-install checks need an interpreter
  that CANNOT import synthkit, and the data machine has one -
  its system python carries the numeric stack and no install at
  all. The search looked for `python3.14`, `python3.13` and
  `python3.12`, none of which is a command name on Windows, so
  it found nothing and printed its NOTE: the checks built for
  the zipball flow skipped on the zipball machine. Plain
  `python` and `python3` are candidates now - safe, because a
  candidate is accepted only if importing synthkit FAILS in it,
  so a venv with the package installed is rejected whatever it
  is called. Found by insisting the delta decompose rather than
  by anything going red.

- **TMP-1 IS ANSWERED ON THE REAL EXTRACT, AND THE WITHIN-STAY
  LINE IS BLOCKED RATHER THAN DEAD (2026-11-05).** The gate
  that could have killed the whole temporal item in one query
  came back NO-CLOCK. Every figure read back spelled by the
  machine:
      rows                    882,103
      distinct visits          14,241
      REPEATED                  7,296  visits hold the same
                                       concept twice
      MOVING                    3,907  of those also CHANGE
                                       value
      TIMED                       n/a  THERE IS NO TIME COLUMN
  **The series EXIST** - 51% of visits repeat a concept and in
  3,907 the values actually move - so the extraction did NOT
  collapse them, which was the feared answer. What the
  measurement table has no column for is WHEN: it carries
  measurement_id, person_id, concept, value, units, ranges and
  visit_occurrence_id, and nothing that says time. Order and
  interval are therefore unknowable, and "haemoglobin falls
  2 g/dL in 24h" needs both.
  THE ASK IS NOW CONCRETE AND SMALL: a re-pull including
  `measurement_datetime`. That is a request to whoever runs the
  extraction, not a research item, and it converts an 8-point
  spike from unbuildable into blocked-on-one-column.
- **AND THE PROBE'S FIRST RUN REFUSED, CORRECTLY AND
  USELESSLY.** With no time column it stopped - a clean
  sentence rather than a traceback, which is right - and threw
  away the half of the answer that is knowable. Counting
  repeats and asking whether their values differ needs no
  clock. A missing clock is a DIFFERENT FACT from a collapsed
  extract and calls for a different action, so it has its own
  verdict and its own exit code, and TIMED is reported ABSENT
  rather than zero: a zero would read as "the repeats share one
  instant", which is a third thing again and wrong.
- **THE MEASUREMENT TABLE IS A WIDER COHORT THAN THE FITTED
  EXTRACT, said out loud so nobody pairs the numbers.** 14,241
  distinct visits here against the 23,091 in visit_occurrence
  and the 800 patients / 55,428 rows the gate numbers come
  from. These counts describe what the SOURCE contains, not
  the cohort any run was built on.

- **THE TOKEN CAP IS SOLVED NOW, NOT ASSUMED, AND IT LANDS ON
  THE CAP THE MEASUREMENT ALREADY FOUND GOOD (2026-11-05).**
  Raising the cap 60 -> 400 was measured as a real fidelity
  gain - close **+3.1/+6.2/+3.1** paired over three seeds - and
  it stayed off the default for weeks because a flat 400 is
  unsafe on a wider frame. The budget that would have made it
  safe was built, measured and withdrawn: it PREDICTED the
  encoded width from k-screened levels on the full frame and
  came out about 2.5x high at every cap, so it solved for a cap
  on the wrong number and moved nothing.
  Two things make it solvable now. `plan_width` gives the exact
  width the encoder will build - num 2, cat len(levels), set
  1 + tokens - and `fit(..., plan_only=True)` stops before
  training, so pricing a candidate is a plan build and a sum
  rather than a full fit. **The last attempt predicted because
  measuring looked expensive. It is not: under a second on a
  small frame.**
  On an extract-shaped frame (61,121 rows x 78 columns) the
  ladder walks down and the solve reports its work:
      cap 800 -> width 3066   over budget
      cap 600 -> width 2579   over budget
      cap 400 -> width 1979   TAKEN
  **400 is the cap the paired measurement found good, reached
  by pricing rather than by guessing** - and 800 and 600 are
  refused for exactly the reason that kept a flat default off.
- **THE LADDER'S FLOOR IS THE OLD DEFAULT, so solving can only
  match or improve on what shipped.** It cannot quietly narrow
  a vocabulary, which is the failure mode a budget could
  otherwise introduce. And where the mechanism is absent it is
  INERT: on a frame with no set columns the solved run is
  BYTE-IDENTICAL to the one before it, verified on the same
  fixture and seed.
- **AND THE WIDTH FUNCTION IS CHECKED AGAINST THE MATRIX IT
  DESCRIBES.** `plan_width(plan) == encoder.d` is asserted on a
  fixture whose plan carries the SET branch with 400 planted
  tokens - that is the term a cap moves, and checking width on
  a numeric-only frame would leave the formula that matters
  untested. If the two ever drift, the 2.5x error is back in a
  new place and every budget decision rests on it. Four
  mutations red: the set size dimension dropped, the numeric
  missing flag dropped, the budget ignored, and the ladder
  floor pushed below the old default.

- **PHI-2 LANDED: THE THREE MISSING DETECTORS, AND THE REMEDY
  TRAVELS WITH THE FINDING (2026-11-06).** PHI-1 measured the
  gap; this closes it. `name` now fires (it was a declared kind
  with no detector - a column of person names landed in
  `clear` while KINDS described the class as covered), and
  `service_date` and `age_over_89` are new kinds. All three
  characterization checks from PHI-1 went RED on the first run,
  which is what they were for, and were armed.
- **THE SAFETY-CRITICAL PART WAS NOT THE DETECTION, IT WAS THE
  REMEDY.** `scrub.apply` drops every finding. Adding service
  dates and ages as findings WITHOUT a remedy would have made
  the scrub destroy span, length-of-stay and a covariate every
  model uses - strictly worse than never detecting them. So
  each finding carries `remedy`: drop for the direct
  identifiers, GENERALISE for the two Safe Harbor categories.
  `apply` removes only the drops, `pending()` reports what is
  still owed, and a check asserts the dates SURVIVE `apply`.
  Detected, reported and not acted on is bad; detected and
  silently destroyed is worse.
- **AND WHAT SEPARATES A PERSON'S NAME FROM A DRUG'S IS THAT IT
  BELONGS TO THE PERSON.** Shape alone - letters, one to three
  words - cannot tell `Dana Ibarra` from `Oxygen Therapy` or
  from OMOP's own `standard_concept_name`, and dropping a
  concept column would be a bad trade. A name is CONSTANT
  within a patient and about one distinct value per patient; a
  concept name varies within one by construction. The existing
  scrub suite had planted `Oxygen Therapy` as a false-positive
  control years before this detector existed, and it holds.
- **THE NEGATIVE CONTROL CAUGHT THAT THE TEST CANNOT WORK ON A
  CROSS-SECTIONAL TABLE.** On one row per patient EVERY column
  is trivially constant within a patient, so the discriminator
  discriminates nothing and would flag a concept column as a
  name on any such file. My own fixture was one row per patient
  and the control fired. Where fewer than 10% of patients
  repeat, the header now decides - the way the birth-date rule
  already does - and the finding SAYS which evidence it used.
- **AND A SECOND PATTERN FOR ONE RULE WAS WRITTEN AND
  DELETED.** `_MONTHDAY` was added so a date already reduced to
  a year would not be a finding - then the mutation meant to
  prove it found NOTHING, because `_DATEISH` already requires a
  day component and a bare year never reaches the branch. Two
  patterns for one rule is how two halves come to disagree, so
  the duplicate went and the property is asserted on `_DATEISH`
  itself: loosening it to accept `2019` now turns three checks
  red where the end-to-end control alone could not fail.
- **AND THE REAL EXTRACT GRADED THE DETECTOR AGAIN, WITHIN THE
  HOUR (2026-11-06).** PHI-2's first real read flagged
  **gender, race and ethnicity as person names** - and the
  remedy for `name` is DROP, so `apply` would have deleted
  three covariates from the extract. Every test passed on them
  honestly: alphabetic, one to three words, CONSTANT within
  100% of patients. What they lack is VARIETY - 3, 10 and 7
  distinct values across 800 patients, where a name column
  holds roughly one per person. A floor at 0.2 x patients
  separates them and no fixture here had a low-cardinality
  constant-per-patient column until the extract supplied
  three. The same lesson as the scrub's first real read in
  September, in the same module, on a new kind.
- **AND MY OWN FIXTURE COULD NOT REACH THE FLOOR, TWICE.**
  First it used 6 first names and 5 surnames - 30 combinations
  across 400 patients, well under the floor, so the positive
  check went red when the floor was added. Then the
  replacement used `Nm00`-style placeholders, which are not
  ALPHABETIC at all and the shape test rejected them. A
  fixture must contain the thing the check is about, and
  "a name" turned out to mean three properties at once:
  letters only, one to three words, and about one value per
  person.

- **PHI-3 APPLIES THE GENERALISATION, AND M2 IS COMPLETE
  (2026-11-06).** `scrub.generalise` does the work `pending`
  reported: a service date keeps its YEAR and loses the month
  and day, an age over 89 is capped at 90 - the single
  "90 or above" category the rule asks for - and both COLUMNS
  STAY, which is the entire reason the remedy was never a drop.
  The CLI's `--apply` now does both, reports each action, and
  REFUSES with exit 1 if a fresh detect still finds anything,
  because a file that says it was de-identified while half the
  work is outstanding is the worst outcome available.
- **THE LOAD-BEARING PROPERTY IS IDEMPOTENCE, AND IT CAUGHT THE
  DETECTOR REJECTING ITS OWN REMEDY.** After dropping and
  generalising, a fresh detect must find NOTHING. It found
  `age_at_visit` still flagged - because the rule fired on any
  value `> 89` and the cap writes exactly 90, which is itself
  over 89. Safe Harbor's category IS "90 or above", so a column
  already aggregated there is compliant; the finding is an
  UNAGGREGATED TAIL, meaning any value above 90. A
  de-identification step whose output fails its own detector is
  incoherent, and only the end-to-end check could see it.
- **AND TRUNCATING TWO DATES COSTS THE LENGTH OF STAY UNLESS
  SOMETHING ELSE CARRIES IT.** Setting both service dates to
  their years makes every duration computed from them zero -
  the same destruction the generalise remedy exists to avoid,
  wearing a different name. So before truncating, the tool
  looks for a column that EQUALS the difference between the
  dates - detected by equality on 95% of rows, never by a name
  that sounds right - and the action says either "the span is
  carried by `X`" or that the length of stay is LOST by this
  step. It does not invent the column: this file's rule is not
  to put a column in somebody's file that they did not give
  you.
- **AND PANDAS 3 TURNED THE WRITE-BACK INTO A HARD TypeError.**
  Assigning the integer 90 into a string column raises on
  pandas 3 where 2.x coerced silently - the same shape as the
  FutureWarning-to-TypeError change that once passed here and
  crashed on the data machine. Caught on the development
  machine this time, for one reason: the work was done against
  `.venv314`, the stack the data machine installs. The cap and
  the year are written in the column's own dtype now.

- **MT-1: THE CROSS-TABLE RELATIONSHIP IS RECOVERED, AND THE
  ASSIGNMENT WAS THE ONLY FREE VARIABLE (2026-11-06).** Each
  table is still fitted ALONE - that part is unchanged, and it
  is why a child column that depends on a parent's had no path
  to learn it. What changed is WHICH PARENT a child row points
  at. `assign_keys` drew uniformly, which guaranteed the
  structure was absent rather than merely weak.
      source                     +0.906
      uniform assignment         -0.023
      coupled assignment         +0.917
  The coupling is FOUND in the source, not declared: numeric
  pairs over the join, rank correlation, screened for rows and
  for the gate's own 0.1 floor so noise is not chased across a
  key.
- **IT IS A REARRANGEMENT, WHICH IS WHY IT COSTS NOTHING.** The
  child's values and the parent's values are already drawn and
  carry their marginals; only the attachment is permuted.
  Measured: both value columns come out BYTE-IDENTICAL, and
  referential integrity stays 100% because the pool assigned
  from IS the parent keys that exist. Same instrument as the
  repair and the dials, applied across a link instead of
  within a column.
- **AND THE CONTROL ARM IS IN THE SUITE, not just the result.**
  The same seed with no couplings passed still reads -0.023, so
  the recovery is the coupling and not the seed. This file has
  twice recorded a fidelity "gain" that a control arm later
  overturned; the control ships beside the claim now.
- **THE LIMIT IS NAMED: ONE COUPLING PER LINK, THE STRONGEST.**
  Two columns pulling on one assignment is a joint solve, and
  this file already records how a cap that excluded exactly the
  pair the gate measures is how a fix misses. Written down as
  the next step rather than left to be discovered.
- **AND A MUTATION THAT DOES NOT APPLY LOOKS EXACTLY LIKE A
  CHECK THAT CANNOT FAIL, AGAIN.** The sign-inversion
  fail-first asserted count==1 on a line whose indentation I
  had guessed, the assert fired, the file was never modified -
  and the run that followed reported FAILs: 0 against
  UNMUTATED code. Reported as a non-result and redone against
  the real line, where it turns one check red. The oldest
  editing rule in this file, and it still needs its own guard.
- **AND THE REAL OMOP TABLES FOUND A BUG IN THE MEASUREMENT
  ITSELF, WITHIN THE HOUR.** The first cross-coupling read on
  `measurement` -> `visit_occurrence` returned pairs like
  `value_as_number ~ standard_concept_code` and
  `range_high ~ provider_id` - identifiers and codes, which
  should never have been the strongest couplings across a
  link. The cause: `merge(..., suffixes=("", "__par"))`
  suffixes ONLY the colliding name, so the bare one refers to
  the CHILD. `provider_id` exists in both tables in real OMOP,
  so the parent's column was read from the child and a
  WITHIN-CHILD correlation came back labelled cross-table.
  Every parent column is renamed before the join now, so there
  is nothing to resolve ambiguously.
  THE FIXTURE COULD NOT REACH IT: its two tables share only
  the join key, which is excluded. The control planted now is
  a column named `site` that is CONSTANT in the parent and
  random in the child - reading the wrong side gives a
  different answer every time - and restoring the old merge
  turns it red.
- **AND THE REAL TABLES THEN SHOWED THE COUPLINGS WERE MOSTLY
  IDENTIFIERS.** With the join fixed, the strongest
  cross-table couplings on `measurement` -> `visit_occurrence`
  fell from +0.45 to **+0.16**, and the list was led by
  `measurement_id ~ provider_id`, `value_as_concept_id ~
  care_site_id` and `provider_id ~ care_site_id`. A rank
  correlation between two keys means nothing, and enforcing
  one would attach a child to a parent by ID SIMILARITY -
  worse than not enforcing at all. Both sides are screened
  now: by NAME (`_id`, `_code`, `_concept_id` - OMOP's
  convention is strong enough to use, and it is stated as
  used, the way the birth-date rule states that the header
  decides) and by VALUE, so a renamed key is still caught.
- **AND NEAR-UNIQUENESS WAS THE WRONG VALUE TEST, which the
  suite caught immediately.** The first screen refused any
  near-unique numeric - and a continuous measurement is
  near-unique too, so `lab_value` itself was refused along
  with `measurement_id`. What separates a counter from a
  quantity is being INTEGRAL as well as near-unique: a key has
  no fractional part, while an integer column with few
  distinct values is an ordinary count and passes.
- **AND THE PLANTED `_id` HAD TO BE OFFERABLE FOR ITS OWN
  CHECK TO MEAN ANYTHING.** The first fixture gave
  `provider_id` a modular counter, which correlates with
  nothing - so removing the NAME screen changed no verdict and
  the mutation found nothing. It is a rounded copy of the
  length of stay now, carrying the same strong signal as the
  genuine coupling, so only its name keeps it out and the
  mutation turns it red.
- **WHAT THE REAL READ MEANS FOR MT-1, STATED PLAINLY.** On
  these two tables the clinical cross-table structure is WEAK
  once the identifiers are gone - the surviving pairs sit near
  the 0.1 floor. The mechanism is proven on a planted fixture
  (+0.906 recovered as +0.917) and it is INERT where there is
  nothing to carry, which is the right behaviour; but nobody
  should quote the fixture number as a statement about this
  extract. A visit table of dates and codes has few numeric
  quantities for a measurement to depend on, and the obvious
  candidate - a length of stay - is not a column there.
