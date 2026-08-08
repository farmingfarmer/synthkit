# synthkit — working conventions

Synthetic clinical data generator and model-evaluation instrument. Core rule: learn the patterns, never copy the records. Real data teaches parameters; generation never touches a record.

## Verify before claiming

- Run `python scripts/run_all_smokes.py` before saying anything works. Expect 41 suites, 1115 checks, ALL GREEN.
- `py_compile` every Python file you touch.
- Assert count==1 before every string replacement — verify the edit, not just the compile.
- **Read a file before Write overwrites it.** `Write` says "updated"
  rather than "created" when the path exists, and that word is the
  only warning. A new `synthkit/spec.py` silently replaced SYNTH_V1's
  DataSpec and took out 17 suites. `ls` the target, or pick a name the
  repo does not already use.
- Never state a number as measured unless you actually ran it.

## Fixing what the real extract found

The Windows machine has the data; this machine has only its statistics.
Every rule here exists because a fix passed here and did nothing, or
did harm, there.

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
- **An internal contradiction is the signal.** Every measurement error
  here was caught by two numbers disagreeing, never by review — 0.735
  against an asymptote of 0.38; a negative rho that cannot exist.
- **State predictions as hypotheses with a test, not as findings.**
  Dates driving the attack: −0.001. Bonferroni as the main suppressor:
  +0, twice.

## Talking to the Windows machine

- **Code comes from git; output paths are chosen at run time.** Never
  give an invented directory name as if it were a deliverable — say
  plainly that the user picks it. Four round trips were lost to `fix5`,
  `run_missingfix` and `test9` reading as things to pull.
- One command per line, no line wrapping: an unexpanded `%USERPROFILE%`
  wrote real-data output into the repo.

## Code

- Python 3.8 target for the stdlib engine. No walrus, no match
  statements.
- **numpy, pandas and scikit-learn are allowed** as of 2026-08-07: the
  Windows machine can pip install. `synthkit/discover.py` uses them.
  The older stdlib modules stay stdlib until they are replaced.
- **`getattr(obj, name, default)` on a library attribute is banned.**
  `HistGradientBoosting` has no `feature_importances_`; the getattr
  returned None on every column, the fallback ranked columns by
  POSITION, and recall read 0% while the model scored 0.93 on the same
  target. A silent no-op is worse than a crash. Prefer a screen that
  is the same computation as the measurement, so it cannot no-op.
- Every new capability gets smoke checks that can actually fail. A test that cannot fail proves nothing.
- One CLI command per line, zsh-compatible, BSD `sed -i ''`.

## Environment

- Claude Code runs ONLY on this Mac.
- Real clinical extracts live on a separate Windows machine that pulls from git and runs there. It never runs Claude Code.
- Code that touches real data must be written without access to it: stream, report progress, fail readably.

## Tone

- Direct and concise. No emojis. No over-explanation.
