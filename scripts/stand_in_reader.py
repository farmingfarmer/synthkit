"""Stand-in note readers, so the instruments can be shown to
discriminate without waiting for a vendor.

WHY THESE AND NOT A MOCK. A reader that copies the ledger proves
the pipe is connected and nothing else; one that fabricates a
fixed label proves the grounding audit can see invention. Neither
READS THE PROSE, so neither can show whether the instruments
separate readers that fail the way real tools fail. These three
do read it, and their failure profiles are different on purpose:

    keyword          the label appears in the text, so present.
                     What a regex-based extractor does, and it
                     walks straight into every negation.
    negation-aware   ...unless a negation word sits just before
                     it. Fixes the obvious trap and walks into
                     the SCOPE trap, where a negation sits in the
                     clause without reaching the finding.
    greedy           any WORD of the label appears anywhere. Over
                     claims, which is the behaviour the GROUNDING
                     audit exists to catch rather than the
                     per-type scorer.

TWO INSTRUMENTS, TWO FAILURE CLASSES, and that is the thing worth
demonstrating: `grounding.audit` catches INVENTION and
`transcribe.score_extraction` catches MISREADING. A reader that
claims only facts from the known list cannot be ungrounded, so
the grounding census should separate the first two by nothing -
and if it does separate them, one of them is inventing.

NOTHING HERE IS A MODEL. These are deliberately dumb so that what
they get wrong is PREDICTABLE from the corruption that was
planted, which is what makes them a ruler.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODES = ("keyword", "negation-aware", "greedy")

# Words that deny what follows. Deliberately short: a real
# extractor's list is longer and this is meant to be beatable.
NEGATORS = ("no", "not", "without", "denies", "negative",
            "ruled", "free", "absent", "never")
WINDOW = 36          # characters before a match to look in


def _norm(t):
    return " ".join(str(t or "").lower().split())


def _negated_before(text, at):
    before = text[max(0, at - WINDOW):at]
    return any(re.search(r"\b" + w + r"\b", before)
               for w in NEGATORS)


def read_note(text, facts, mode):
    """What this reader claims about one note."""
    low = _norm(text)
    out = []
    for f in facts:
        label = _norm(f.get("label"))
        if not label:
            continue
        if mode == "greedy":
            words = [w for w in re.split(r"[^a-z0-9]+", label)
                     if len(w) > 3]
            hit = any(re.search(r"\b" + re.escape(w) + r"\b", low)
                      for w in words) if words else (
                          label in low)
            at = -1
        else:
            m = re.search(re.escape(label), low)
            hit = m is not None
            at = m.start() if m else -1
        if not hit:
            continue
        if mode == "negation-aware" and at >= 0 \
                and _negated_before(low, at):
            continue
        out.append({"label": f.get("label"),
                    "fact": f.get("column"),
                    "present": True,
                    "current": True,
                    "certain": True})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="produce a stand-in reader's claims over a "
                    "note set")
    ap.add_argument("--notes", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--note-column", default="clinical_note")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)

    import pandas as pd
    try:
        key = json.loads(Path(a.ledger).read_text(encoding="utf-8"))
        notes = pd.read_csv(a.notes, dtype=str,
                            keep_default_na=False)
    except (OSError, ValueError) as e:
        print("could not read: {}".format(e), file=sys.stderr)
        return 2
    if "facts" not in key or "ledger" not in key:
        print("{} is not a ledger from `synthkit notes`".format(
            a.ledger), file=sys.stderr)
        return 2
    if a.note_column not in notes.columns:
        print("no column {!r} in {}".format(a.note_column,
                                            a.notes),
              file=sys.stderr)
        return 2
    if len(notes) != len(key["ledger"]):
        print("{} has {} rows and the ledger has {} - not the "
              "same run".format(a.notes, len(notes),
                                len(key["ledger"])),
              file=sys.stderr)
        return 2

    facts = key["facts"]
    claims = [read_note(t, facts, a.mode)
              for t in notes[a.note_column].astype(str)]
    total = sum(len(c) for c in claims)
    json.dump(claims, open(a.out, "w", encoding="utf-8"))
    print("stand-in reader: {}".format(a.mode))
    print("  {} note(s), {} fact(s) offered, {} claim(s) "
          "made".format(len(notes), len(facts), total))
    if not total:
        print("  NOTHING CLAIMED - this reader found nothing, "
              "which is its own answer and not a clean one")
    print("  wrote {}".format(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
