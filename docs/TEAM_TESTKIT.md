# The team test kit

One command assembles a folder anyone can run and try to break,
with no real data anywhere near it:

    python scripts/make_testkit.py -o YOUR_FOLDER

The folder contains the invented clinic dataset, its printed
answer key (nine planted patterns), `START_HERE.md` walking every
command from the two-second types check through fit, the bench,
the exam loop, the report card and the sign-off page, and
`TRY_TO_BREAK.md` - a challenge list, because the fastest way to
trust an instrument is to fail to break it yourself.

Prerequisites for the recipient: Python 3.10+, then
`pip install -e .` from the synthkit repository (and optionally
`pip install shap` for driver attribution). The kit carries no
code of its own, so it never goes stale against the install.

Ground rules for kit sessions:

- Everything in the kit is invented; the answer key is printed on
  purpose so every claim can be judged against known truth.
- A refusal should always be a sentence naming the file, the
  cause, and the next command. A raw stack trace is a bug in
  itself - report it even if you caused it on purpose.
- Anything you break becomes a check in the suite, and the kit
  gets harder. That is the point.
