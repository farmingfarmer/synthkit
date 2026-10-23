"""Checks for compare_sources - the side-by-side that exists to stop a
reader drawing the opposite conclusion from the truth.

The trap, measured: an 800- and a 100-patient cut of the SAME generator
through the SAME engine gave the thin cohort the BETTER headline, close
91% against 88%, while it published 61% of its data against 75% and
carried 12 distinct condition codes against 60. Fidelity is scored only
on what cleared the privacy floor, so a thinner source grades a smaller
and stronger surviving set.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

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


def _make(tmp, tag, patients, per, vocab, seed=1):
    """A source and a matching 'run' directory. `vocab` sets how many
    distinct tokens exist, so a wide vocabulary on few patients is the
    starved shape and a narrow one on many is the healthy shape."""
    r = np.random.RandomState(seed)
    rows = []
    for p in range(patients):
        pid = "P{:04d}".format(p)
        for _ in range(per):
            toks = ";".join(sorted(set(
                "t{:04d}".format(r.randint(vocab)) for _ in range(3))))
            rows.append({"person_id": pid, "codes": toks,
                         "sex": "F" if p % 2 else "M",
                         "val": round(r.normal(100, 10), 2)})
    src = pd.DataFrame(rows)
    d = Path(tmp) / tag
    d.mkdir(parents=True, exist_ok=True)
    sp = d / "src.csv"
    src.to_csv(sp, index=False)
    run = d / "run"
    run.mkdir(exist_ok=True)
    # a stand-in "generated" table: the same shape with a THINNER
    # vocabulary, which is what the k rule does to a real output
    keep = ["t{:04d}".format(i) for i in range(min(vocab, 20))]
    g = src.copy()
    g["codes"] = [";".join(sorted(set(r.choice(keep, 2))))
                  for _ in range(len(g))]
    g.to_csv(run / "generated.csv", index=False)
    return str(sp), str(run)


def main():
    import compare_sources as C

    with tempfile.TemporaryDirectory() as tmp:
        rich_s, rich_r = _make(tmp, "rich", patients=400, per=6, vocab=40)
        thin_s, thin_r = _make(tmp, "thin", patients=40, per=60, vocab=900)

        pr = C.profile("rich", rich_s, rich_r)
        pt = C.profile("thin", thin_s, thin_r)

        check("a profile carries BOTH how much the source can publish "
              "and what reached the output - neither alone is the "
              "answer",
              "mass" in pr and pr["sets"]
              and all("src_tokens" in s and "gen_tokens" in s
                      for s in pr["sets"]))
        check("the patient-poor source publishes a smaller share of "
              "its data than the healthy one",
              pt["mass"] < pr["mass"])

        txt = C.render([pr, pt])
        check("the rendered page puts coverage ABOVE fidelity, so the "
              "denominator is read before the percentage",
              txt.index("HOW MUCH THE SOURCE CAN PUBLISH")
              < txt.index("WHAT REACHED THE OUTPUT"))
        check("...and it states the trap in words, where a reader "
              "comparing two percentages will meet it",
              "READ THE TWO BLOCKS TOGETHER" in txt
              and "can post a HIGHER percentage while" in txt)
        check("output vocabulary is shown against the SOURCE "
              "vocabulary, not alone - '12 of 6,312' is the fact, "
              "'12' is not",
              " of " in txt and "vs" in txt)

        # A criterion whose detail is not a bare fraction must not drag
        # its explanation into the column: `inverted` reads
        # "0 - absolute, ...", `shapes tracked` reads "46/69 effect
        # curves within ...". Both were printing whole sentences.
        class _F:
            pass
        got, pct = C._frac({"detail": "0 - absolute, and it stays"})
        check("a non-fraction criterion renders as its leading token, "
              "not as a sentence", got == "0" and pct is None)
        got2, pct2 = C._frac({"detail": "46/69 effect curves within"})
        check("...and a fraction keeps its denominator visible beside "
              "the percentage",
              got2 == "46/69" and round(pct2) == 67)

        longest = max(len(line) for line in txt.splitlines())
        check("no row overflows the table it is printed in "
              "({} chars)".format(longest), longest <= 140)

    print("\n{} checks, {} failed".format(PASS + FAIL, FAIL))
    if FAIL:
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
