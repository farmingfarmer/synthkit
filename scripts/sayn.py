"""OCR-SAFE NUMBERS OVER THE SCREEN BRIDGE.

MEASURED, not assumed: the data machine is read by screen OCR, and
DIGITS DO NOT SURVIVE IT. In one read, `0123456789` came back as
`9123456789`; elsewhere `0.37` reads as `@.37`, `0.30` as `0.3@` and
`4544_3` as `4544 3`. In the SAME read, all ten spelled words - zero
one two three four five six seven eight nine - came back perfectly.

A number quoted off an OCR pass is therefore an artifact rather than a
measurement, and the standing rule here is never to state a number as
measured unless it actually was. So anything whose numbers matter is
piped through this, which repeats every numeric literal in words:

    close 77/94        ->  close 77/94 [seven seven slash nine four]
    centre miss -0.27  ->  centre miss -0.27 [minus zero point two seven]

The words are the channel; the digits beside them are a convenience
that may arrive corrupted. When the two disagree, the WORDS are right.

  python scripts/sayn.py FILE [FILE ...]
  ... | python scripts/sayn.py -
"""
import re
import sys

WORD = {"0": "zero", "1": "one", "2": "two", "3": "three",
        "4": "four", "5": "five", "6": "six", "7": "seven",
        "8": "eight", "9": "nine"}

# A numeric literal: an optional sign, digits with optional thousands
# commas, an optional decimal part, an optional percent. Kept
# deliberately narrow - a run of digits inside an identifier like
# `4544_3` is a column NAME, and spelling those would bury the
# numbers that matter in noise.
LITERAL = re.compile(r"(?<![0-9A-Za-z_.])"
                     r"-?[0-9][0-9,]*(?:\.[0-9]+)?%?"
                     r"(?![0-9A-Za-z_])")


def spell(lit):
    """The literal, digit by digit, in words. Digit-by-digit rather
    than 'seventy-seven' on purpose: it is unambiguous at any
    magnitude, needs no grammar, and reads back the same whether the
    number is a count, a ratio or a percentage."""
    out = []
    s = lit
    if s.startswith("-"):
        out.append("minus")
        s = s[1:]
    pct = s.endswith("%")
    if pct:
        s = s[:-1]
    for ch in s:
        if ch in WORD:
            out.append(WORD[ch])
        elif ch == ".":
            out.append("point")
        elif ch == ",":
            continue          # a thousands separator carries no value
    if pct:
        out.append("percent")
    return " ".join(out)


def annotate(text):
    """Every line that holds a numeric literal gains a bracketed
    spelling of each one, in order. Lines with no numbers pass
    through untouched, so prose stays readable."""
    lines = []
    for line in text.splitlines():
        lits = LITERAL.findall(line)
        if not lits:
            lines.append(line)
            continue
        # EACH LITERAL PAIRED WITH ITS SPELLING, IN ORDER. The first
        # cut guessed a separator from the line ("slash" when the line
        # held a /), which put "slash" between a fraction and an
        # unrelated percentage on the same line. Order is all the
        # pairing needs; a guessed separator only invents structure.
        said = ["{} = {}".format(lit, spell(lit)) for lit in lits]
        lines.append("{}   [{}]".format(line.rstrip(),
                                        " | ".join(said)))
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    chunks = []
    for a in argv:
        if a == "-":
            chunks.append(sys.stdin.read())
        else:
            with open(a, "r", encoding="utf-8", errors="replace") as fh:
                chunks.append(fh.read())
    print(annotate("\n".join(chunks)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
