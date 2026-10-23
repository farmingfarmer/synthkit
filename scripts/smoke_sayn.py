"""Checks for sayn - the OCR-safe numeric channel to the data machine.

WHY IT EXISTS, MEASURED. The data machine is read by screen OCR, and
digits do not survive it: `0123456789` came back from one read as
`9123456789`, and elsewhere `0.37` reads as `@.37`, `0.30` as `0.3@`.
In the SAME read all ten spelled words came back perfectly. Since the
standing rule is never to state a number as measured unless it was
measured, every number that matters crosses that bridge in words.
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
for _p in (str(_root), str(_root / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PASS = 0
FAIL = 0


def check(name, ok):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print("FAIL: {}".format(name))


def main():
    import sayn as S

    # ---- THE CHARACTERS OCR ACTUALLY CORRUPTED ----
    # A leading zero became a 9 in the real read, so zero is the
    # single most important digit to carry in words.
    check("every digit spells out, zero included - the digit the "
          "bridge was measured corrupting",
          S.spell("0123456789")
          == "zero one two three four five six seven eight nine")
    check("a decimal point is named, so 0.37 cannot arrive as 037",
          S.spell("0.37") == "zero point three seven")
    check("a minus sign is named - a sign flip is this project's "
          "worst failure class, and it is one character",
          S.spell("-0.27") == "minus zero point two seven")
    check("a percent is named rather than dropped",
          S.spell("81.9%") == "eight one point nine percent")
    check("a thousands comma carries no value and is dropped, so "
          "55,428 reads as its digits",
          S.spell("55,428") == "five five four two eight")

    # DIGIT BY DIGIT, NOT "seventy-seven". Unambiguous at any
    # magnitude and needs no grammar - and a count, a ratio and a
    # percentage all read back the same way.
    check("the spelling is digit by digit, so no magnitude has to be "
          "parsed out of English",
          S.spell("231.00")
          == "two three one point zero zero")

    # ---- WHAT IT MUST NOT TOUCH ----
    # A run of digits inside an identifier is a column NAME. Spelling
    # those would bury the numbers that matter in noise.
    out = S.annotate("centre miss -0.27 sd on 4544_3")
    check("a digit run inside an identifier is left alone - 4544_3 "
          "is a column name, not a measurement",
          "four five four four" not in out
          and "minus zero point two seven" in out)
    check("...and a line carrying no number passes through untouched",
          S.annotate("no numbers on this line")
          == "no numbers on this line")

    # ---- ORDER IS THE PAIRING ----
    # The first cut guessed a separator from the line, which put
    # "slash" between a fraction and an unrelated percentage. Each
    # literal is paired with its own spelling instead.
    out2 = S.annotate("close 77/94 = 81.9%")
    check("each literal is paired with its own spelling in order, so "
          "a corrupted digit still has an unambiguous owner",
          "77 = seven seven" in out2
          and "94 = nine four" in out2
          and "81.9% = eight one point nine percent" in out2)
    check("...and no separator is invented between unrelated numbers "
          "on one line",
          " slash " not in out2)

    # ---- THE WHOLE POINT, END TO END ----
    # A gate line is the thing actually read off that screen.
    gate = S.annotate("close 77/94 = 81.9%  (need 87.9%)\n"
                      "inverted 0 - absolute")
    check("a gate reading crosses in words, including the zero that "
          "an INVERTED verdict turns on",
          "0 = zero" in gate and "eight seven point nine" in gate)

    print("\n{} checks, {} failed".format(PASS + FAIL, FAIL))
    if FAIL:
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
