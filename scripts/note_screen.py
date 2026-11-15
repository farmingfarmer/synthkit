"""Does a note mention only what the blueprint published?

A NOTE IS A RELEASE. The k rule governs which tokens a file may
carry - counted in patients, never rows - and prose is not an
exception: a note naming a diagnosis the screen withheld has
undone the screen, and it has done so in the one surface nobody
would think to audit.

THE CHECK IS AGAINST THE BLUEPRINT, NOT THE FACT LIST. The facts
are derived from the published vocabulary, so comparing them
against themselves would be circular and would pass however the
derivation drifted. The published list in `blueprint.json` is the
contract; the ledger says what the notes assert; this compares
the second against the first.

AND IT REPORTS THE DENOMINATOR. "No token leaked" over a note set
that mentions nothing is the worst kind of clean, so the count of
tokens actually mentioned is printed beside the verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthkit import sets as _sets                # noqa: E402


def published_tokens(bp) -> dict:
    """column -> set of published token values."""
    out = {}
    for col, spec in (bp.get("columns") or {}).items():
        m = spec.get("marginal") or {}
        toks = m.get("tokens")
        if not isinstance(toks, list):
            continue
        vals = set()
        for t in toks:
            if isinstance(t, dict) and "value" in t:
                vals.add(str(t["value"]))
        if vals:
            out[col] = vals
    return out


def mentioned_tokens(key) -> list:
    """(column, token) for every fact the notes could mention.

    A fact's column is the derived indicator, so the source
    column and the token are read back off the name through
    `sets`' own helpers rather than by splitting the string here -
    two halves deciding separately what an indicator name means
    is how they come to disagree.
    """
    out = []
    for f in (key.get("facts") or []):
        col = f.get("column") or ""
        src = _sets.source_of(col)
        if not src:
            continue
        tok = col.split(_sets.HAS, 1)[1] if _sets.HAS in col else None
        if tok:
            out.append((src, tok, f.get("label")))
    return out


def screen(bp, key) -> dict:
    pub = published_tokens(bp)
    said = mentioned_tokens(key)
    leaked, ok = [], 0
    for col, tok, label in said:
        allowed = pub.get(col)
        if allowed is None:
            leaked.append({"column": col, "token": tok,
                           "label": label,
                           "why": "the blueprint publishes no "
                                  "token list for this column"})
            continue
        if tok in allowed:
            ok += 1
        else:
            leaked.append({"column": col, "token": tok,
                           "label": label,
                           "why": "not in the published "
                                  "vocabulary"})
    return {
        "tokens_mentioned": len(said),
        "tokens_published": sum(len(v) for v in pub.values()),
        "within_the_screen": ok,
        "leaked": leaked,
        "verdict": _verdict(len(said), leaked),
    }


def _verdict(n, leaked):
    if not n:
        return ("NO TOKEN IS MENTIONED AT ALL, so the screen is "
                "not measured here. That is an absent "
                "measurement and not a clean one.")
    if leaked:
        return ("{} of {} mentioned token(s) are NOT in the "
                "published vocabulary - a note naming what the "
                "file withheld has undone the k screen".format(
                    len(leaked), n))
    return ("CLEAN - all {} mentioned token(s) are in the "
            "published vocabulary".format(n))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="verify a note set mentions only tokens the "
                    "blueprint published")
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("-o", "--out")
    a = ap.parse_args(argv)
    try:
        key = json.loads(Path(a.ledger).read_text(encoding="utf-8"))
        bp = json.loads(Path(a.blueprint).read_text(
            encoding="utf-8"))
    except (OSError, ValueError) as e:
        print("could not read: {}".format(e), file=sys.stderr)
        return 2
    if "facts" not in key:
        print("{} is not a ledger from `synthkit notes`"
              .format(a.ledger), file=sys.stderr)
        return 2
    rep = screen(bp, key)
    print("note screen")
    print("  tokens mentioned by the notes  {}".format(
        rep["tokens_mentioned"]))
    print("  tokens the blueprint publishes {}".format(
        rep["tokens_published"]))
    print("  within the screen              {}".format(
        rep["within_the_screen"]))
    print("")
    print(rep["verdict"])
    for l in rep["leaked"][:20]:
        print("  LEAKED {} / {} ({}) - {}".format(
            l["column"], l["token"], l["label"], l["why"]))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1),
                               encoding="utf-8")
        print("")
        print("wrote {}".format(a.out))
    return 1 if rep["leaked"] else 0


if __name__ == "__main__":
    sys.exit(main())
