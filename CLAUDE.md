# synthkit — working conventions

Synthetic clinical data generator and model-evaluation instrument. Core rule: learn the patterns, never copy the records. Real data teaches parameters; generation never touches a record.

## Verify before claiming

- Run `python scripts/run_all_smokes.py` before saying anything works. Expect 31 suites, 929 checks, ALL GREEN.
- `py_compile` every Python file you touch.
- Assert count==1 before every string replacement — verify the edit, not just the compile.
- Never state a number as measured unless you actually ran it.

## Code

- Python 3.8 target. No walrus, no match statements.
- Every new capability gets smoke checks that can actually fail. A test that cannot fail proves nothing.
- One CLI command per line, zsh-compatible, BSD `sed -i ''`.

## Environment

- Claude Code runs ONLY on this Mac.
- Real clinical extracts live on a separate Windows machine that pulls from git and runs there. It never runs Claude Code.
- Code that touches real data must be written without access to it: stream, report progress, fail readably.

## Tone

- Direct and concise. No emojis. No over-explanation.
