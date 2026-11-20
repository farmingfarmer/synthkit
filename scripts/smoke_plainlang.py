"""The bench explains itself to somebody who has not seen it before.

WHY THIS EXISTS. Reviewed with the team on 2026-11-19 and the ask was
explicit: "rather than just saying [the k rule] let's just explicitly
put that out there", and "the users may extend beyond just our team to
the larger analytics team". The criterion names a clinician meets --
`set token shares`, `interaction surfaces`, `dynamics tracked` -- are
internal vocabulary presented as though it were plain English.

"Plainer" is not a property a test can assert on its own, so this
asserts the three things that CAN be checked: every criterion has a
reader-facing name and a reason, the names reach the caller rather
than living in a renderer, and the internal keys are untouched --
they are the on-disk contract and renaming one is a format break
wearing a copy-edit's clothes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def main():
    from synthkit import gate, explain

    # ---- every criterion is named for a reader ----
    missing = [k for k in gate.EXPLAIN if k not in gate.PLAIN_NAME]
    check("every criterion the gate explains also has a plain-English "
          "name - a reader meeting `set token shares` cannot act on "
          "it: {}".format(", ".join(sorted(missing)) or "none missing"),
          not missing)
    no_why = [k for k in gate.PLAIN_NAME if not gate.plain_why(k)]
    check("...and each says what it is FOR in one line: {}".format(
        ", ".join(sorted(no_why)) or "all have one"),
          not no_why)

    # THE PLAIN NAME MUST NOT BE THE INTERNAL ONE. A map that returns
    # its own key for everything would pass a presence check while
    # changing nothing a reader sees.
    same = [k for k, v in gate.PLAIN_NAME.items() if v.strip() == k]
    check("...and the plain name DIFFERS from the internal one, so "
          "this is a translation rather than a lookup that echoes "
          "its key: {}".format(", ".join(same) or "all differ"),
          not same)

    # ---- the names reach the caller ----
    fid = {"summary": {"pairs": 10, "sign_kept": 9, "close": 8,
                       "inverted": 0, "own_inverted": 0,
                       "coverage_ok": 5, "coverage": 5}}
    crit = gate.assess(fid)["criteria"]
    check("the criteria come back from `assess` CARRYING their plain "
          "name, so the bench, the sign-off page and the terminal "
          "cannot each describe one differently",
          bool(crit) and all(c.get("plain") for c in crit))
    check("...and their reason too", all("why" in c for c in crit))

    # ---- the contract is untouched ----
    check("the INTERNAL names are unchanged - they are the on-disk "
          "contract, and a renamed field would make every existing "
          "run unreadable",
          {"direction kept", "close", "inverted"} <=
          {c["name"] for c in crit})

    # ---- plain() is still the station renderer ----
    # A SECOND `plain` SILENTLY REPLACED IT while this work was being
    # done, and the terminal gate started printing raw [[station]]
    # markup. Nothing but calling it would have caught that.
    rendered = gate.plain("see [[dashboard]] for this")
    check("`gate.plain` still expands station tokens - defining a "
          "second function of that name replaced it once already: "
          "{}".format(rendered[:46]),
          "[[" not in rendered and "Visualize" in rendered)

    # ---- the k rule is stated in words ----
    terms = explain.as_json()
    terms = terms.get("terms", terms)
    kr = str(terms.get("k-rule", {}))
    check("the k rule is stated in WORDS rather than as a letter - "
          "asked for directly in review",
          "ten different" in kr and "patients, never visits" in kr)
    check("...and it states the cost, so a reader is not surprised "
          "later by what cannot be published",
          "rare" in kr.lower())

    # ---- the eighteen categories are enumerated in the bench ----
    page = (ROOT / "synthkit" / "gui.py").read_text(encoding="utf-8")
    for want in ("Social Security numbers", "Full-face photographs",
                 "Device identifiers", "Network (IP) addresses"):
        check("the De-identify station NAMES what it scans for, "
              "category by category: {}".format(want), want in page)
    check("...and says that finding something does not mean deleting "
          "it - dates keep their year, ages become a 90-or-above "
          "group, notes are reported and not touched",
          "does not mean deleting it" in page)

    # ---- the dial deck ----
    # THE PATTERN DIAL'S OWN CONTROL, not any slider on the page.
    # Two other stations already use a range input, so a bare
    # `type="range"` check passed while the pattern dial had been
    # reverted to a text box - green for the wrong reason.
    check("the patterns station renders a CONTROL rather than a "
          "five-character text box - a typed number cannot say what "
          "an ask means",
          '<input type="range" id="pt_\'+p.id+\'' in page
          and "function dCard(p)" in page)
    check("...and each dial states where the real data sits, where "
          "the released file sits now, and what the pair can reach "
          "at all",
          "real data <b>" in page and "released file now <b>" in page
          and "can only reach <b>" in page)
    check("...and every pattern kind has a plain description, so the "
          "card does not print an internal kind name",
          "var DKIND=" in page
          and "Things that move together" in page
          and "A particular code, and which visits carry it" in page)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
