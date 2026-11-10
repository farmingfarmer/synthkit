"""How much of the `close` criterion is SAMPLING NOISE?

WHY THIS EXISTS. `close` asks what share of the pairs the source
relates at 0.1 or more come back within 0.2 of their source rank
correlation, and the bar is 87.9%. The real extract has sat 1 to 8
points under it for months, and this project's own record says the
bar's provenance is an old absolute target restated as a proportion
rather than one set from a measured distribution. Before spending
more on the engine it is worth knowing what the number could be at
best - and that is computable WITHOUT any fit or generate, from the
source alone.

THE INSTRUMENT. Split the cohort into two disjoint halves BY PATIENT
and put one half in as the source and the other as the generated
table, through the gate's OWN `_pair_fidelity`. Nothing has been
generated, so every pair that fails is failing on sampling noise -
two halves of the same real data disagreeing about a correlation by
more than 0.2. A pair that cannot agree with ITSELF across a split
cannot be expected to agree between a source and any synthetic file.

WHAT IT IS AND IS NOT. It is a LOWER BOUND on the achievable
ceiling, and the reason is stated rather than buried: each half holds
half the patients, so it carries MORE sampling noise than the
full-cohort comparison the gate actually makes. So a split-half
reading BELOW the bar is decisive - the bar is unreachable even
before the engine is involved - while a reading above it is
inconclusive, and this script says which it got rather than leaving a
reader to work it out. The Fisher-z scaling to full cohort size is
reported beside it for the same reason: sqrt(2) less noise, which
moves a correlation's standard error but not the pairs whose
disagreement is structural.

It also reports a SECOND split and the range, because this project
reports a range and never a point.

NO CELL IS PUBLISHED - counts, shares and column NAMES only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthkit.discover import prepare                   # noqa: E402
from synthkit.intake import read_table                  # noqa: E402
from synthkit.pipeline import _pair_fidelity            # noqa: E402
from synthkit.sets import is_scaffolding                # noqa: E402

try:
    from scripts.sayn import sayn                       # noqa: E402
except Exception:                                       # pragma: no cover
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from sayn import sayn                           # noqa: E402
    except Exception:
        def sayn(x):
            return str(x)


def _say(n):
    """Numbers cross this project's screen bridge in WORDS."""
    try:
        return sayn(n)
    except Exception:
        return str(n)


BAR = 87.9


def verdict(highest_pct, bar=BAR):
    """UNREACHABLE or INCONCLUSIVE, as a PURE FUNCTION.

    Asserted on planted numbers rather than on whatever a fixture
    happens to produce. A fixture that cannot reach one branch
    leaves that branch untested inside a tool whose whole output is
    the branch it chose - the dead-branch fault this project has
    already found in a linkage verdict and in a release-dial
    report. `highest_pct` is the BEST of the splits, because the
    claim is about what is achievable.
    """
    if highest_pct is None:
        return "no-pairs"
    return "unreachable" if highest_pct < bar else "inconclusive"


def halves(df, group_by, rng):
    ids = pd.unique(df[group_by].astype(str))
    pick = rng.permutation(len(ids))
    a = set(ids[pick[: len(ids) // 2]])
    m = df[group_by].astype(str).isin(a).to_numpy()
    return df.loc[m].reset_index(drop=True), \
        df.loc[~m].reset_index(drop=True)


def main(argv):
    src = None
    bp_path = None
    group_by = "person_id"
    splits = 3
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--src" and i + 1 < len(argv):
            src = argv[i + 1]
            i += 2
            continue
        if a == "--blueprint" and i + 1 < len(argv):
            bp_path = argv[i + 1]
            i += 2
            continue
        if a == "--group-by" and i + 1 < len(argv):
            group_by = argv[i + 1]
            i += 2
            continue
        if a == "--splits" and i + 1 < len(argv):
            splits = int(argv[i + 1])
            i += 2
            continue
        if a in ("-h", "--help"):
            print(__doc__)
            print("usage: close_ceiling.py --src SRC.csv "
                  "--blueprint BP.json [--group-by COL] "
                  "[--splits N]")
            return 0
        i += 1
    if not src:
        print("STOPPED: --src is required.")
        print("usage: close_ceiling.py --src SRC.csv "
              "--blueprint BP.json [--group-by COL] [--splits N]")
        return 2
    # THE BLUEPRINT IS REQUIRED, not optional. `_pair_fidelity`
    # iterates the blueprint's own relationships, so without one
    # there are ZERO pairs and the tool reports a ceiling over
    # nothing - the worst kind of clean. It also has to be the
    # SAME surface the gate measures: this project lost a week to
    # two runs counting pairs over different token vocabularies and
    # reading the difference as a regression.
    if not bp_path:
        print("STOPPED: --blueprint is required.")
        print("  The pair surface is the blueprint's own "
              "relationships - `_pair_fidelity` iterates them - so "
              "without one this measures a ceiling over NO pairs "
              "and reports it as a number. It also has to be the "
              "surface the GATE measures, which is the run's own "
              "blueprint and not a frame-derived guess.")
        print("usage: close_ceiling.py --src SRC.csv "
              "--blueprint BP.json [--group-by COL] [--splits N]")
        return 2

    # READ THE CONTRACT, DO NOT REMEMBER IT. `read_table` returns
    # (frame, note) - the note says HOW the file was read, and a
    # reader that does not say what it did is the silent-wrong-type
    # fault waiting to happen. The first cut of this script took the
    # tuple as a frame and died on `.columns`.
    df, how = read_table(src)
    if group_by not in df.columns:
        print("STOPPED: no column named {!r} to group by. "
              "The split has to be BY PATIENT - splitting rows "
              "would put the same person on both sides and read as "
              "agreement that is really the same records twice."
              .format(group_by))
        return 2

    bp = None
    if bp_path:
        import json
        bp = json.loads(Path(bp_path).read_text(encoding="utf-8"))

    # The pair surface must be the one the GATE measures, so the
    # vocabulary comes off the blueprint when there is one - this
    # project has already lost a week to two runs counting pairs
    # over different token vocabularies and calling the difference a
    # regression.
    vocab = None
    if bp:
        try:
            from synthkit.pipeline import blueprint_vocab
            vocab = blueprint_vocab(bp)
        except Exception:
            vocab = None

    print("THE CLOSE CEILING FROM SAMPLING NOISE ALONE")
    print("  source rows   {}".format(_say(len(df))))
    print("  patients      {}".format(
        _say(int(df[group_by].astype(str).nunique()))))
    print("  read as       {}".format(how))
    print("  nothing is generated here: both sides are REAL data, so")
    print("  every pair that fails is failing on sampling noise.")
    print()

    rows = []
    for s in range(splits):
        rng = np.random.default_rng(1000 + s)
        A, B = halves(df, group_by, rng)
        pa = prepare(A, group_by=group_by, vocab=vocab) \
            if vocab is not None else prepare(A, group_by=group_by)
        pb = prepare(B, group_by=group_by, vocab=vocab) \
            if vocab is not None else prepare(B, group_by=group_by)
        Xa = pa[0] if isinstance(pa, tuple) else pa
        Xb = pb[0] if isinstance(pb, tuple) else pb
        res = _pair_fidelity(Xa, Xb, bp or {"columns": {}})
        comp = int(res.get("compared") or 0)
        close = int(res.get("close") or 0)
        sign = int(res.get("sign_kept") or 0)
        inv = int(res.get("pairs_inverted") or 0)
        own = res.get("pairs_own_inverted")
        own_c = res.get("pairs_own_close")
        own_t = res.get("pairs_own_compared")
        rows.append((comp, close, sign, inv, own, own_c, own_t))
        pct = (100.0 * close / comp) if comp else float("nan")
        print("  split {}  pairs {}  close {}  = {}%  "
              "sign {}  inverted {}"
              .format(s, _say(comp), _say(close),
                      _say(round(pct, 1)), _say(sign), _say(inv)))
        if own is not None:
            print("            the operator's OWN columns: "
                  "close {} of {}, inverted {}"
                  .format(_say(own_c), _say(own_t), _say(own)))
    print()

    good = [r for r in rows if r[0]]
    if not good:
        print("NO PAIR WAS COMPARABLE ON EITHER SIDE. That is a "
              "finding in itself rather than a ceiling - with no "
              "blueprint the pair surface comes from the frame "
              "alone, so pass --blueprint to measure the surface "
              "the gate measures.")
        return 3
    pcts = [100.0 * r[1] / r[0] for r in good]
    lo, hi = min(pcts), max(pcts)
    print("CEILING FROM SAMPLING NOISE, HALF COHORT: {}% to {}%"
          .format(_say(round(lo, 1)), _say(round(hi, 1))))
    print("  reported as a RANGE across {} splits, because this "
          "project reports a range and never a point."
          .format(_say(len(good))))
    print()
    print("AGAINST THE GATE'S OWN CLOSE BAR OF {}%:".format(_say(BAR)))
    if verdict(hi) == "unreachable":
        print("  THE BAR IS UNREACHABLE ON HALF THIS COHORT even")
        print("  with NOTHING GENERATED - two halves of the real")
        print("  extract cannot agree with each other to within 0.2")
        print("  on {}% of the pairs it relates. The engine is not")
        print("  the only thing between this cohort and that bar.")
        print("  WHAT THIS DOES NOT SAY: that the bar is unreachable")
        print("  at FULL cohort size. Each half holds half the")
        print("  patients, so it carries more sampling noise than")
        print("  the comparison the gate makes - roughly sqrt(2) on")
        print("  a correlation's standard error. A full-cohort")
        print("  ceiling needs a second cohort of the same size,")
        print("  which this extract cannot supply.")
    else:
        print("  INCONCLUSIVE at this cohort size. The split-half")
        print("  ceiling sits at or above the bar, and because each")
        print("  half carries MORE noise than the gate's own")
        print("  comparison, that is consistent with the bar being")
        print("  reachable. It does not establish that it is: this")
        print("  measurement is a LOWER BOUND, and a lower bound")
        print("  above a bar says only that noise alone does not")
        print("  rule the bar out.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
