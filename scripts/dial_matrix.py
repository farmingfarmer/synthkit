"""Every dial, against every neighbouring property, in one pass.

WHY THIS EXISTS, in this repository's own words: "shift and scale
were fighting each other... Both dials had passed their checks
for as long as they existed, because each check set ONE dial and
neither asserted the neighboring property." The same shape has
bitten the per-edge strength dial - taking span<-proc to zero
collapsed the innocent span~drug from 0.555 to 0.007 - and the
relationship dial, and the set-size partner. Every time, a dial
did what it said AND something else nobody was looking at.

One dial checked by one check finds none of these. The fault is
a CROSS PRODUCT - dial times property - so the test has to be
one too.

WHAT IT DOES. For each dial class, on one column, with
everything else untouched and the SAME SEED on both arms:

    arm 0   generate with no dial at all
    arm 1   generate with ONLY this dial

then measure the SAME property vector on both arms, for EVERY
column, plus the pair correlations between them. A dial is
CLEAN when its own property moved and nothing else did. It
LEAKED when anything else moved, and the leak is named.

WHY IT MATTERS BEYOND TIDINESS. The key/released pair rests on
being able to say exactly what a dial did. A dial that silently
moves a neighbouring property is a dial whose effect cannot be
decoded from the record - the cipher would describe one change
while the file carries two. An un-verified dial is not a
governance control, it is a rumour.

TOLERANCES ARE THE REPOSITORY'S OWN. Center within 0.10 of the
column's spread is the same bar `centre_ok` uses everywhere
else; coverage within 0.03 of a share; correlations within 0.10,
which is the gate's own `close` neighbourhood. They are printed
with the table so a reader can disagree with them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit import blueprint as B                      # noqa: E402
from synthkit import dials as D                          # noqa: E402
from synthkit.discover import discover                   # noqa: E402
from synthkit.generate import generate                   # noqa: E402

TOL = {"centre": 0.10, "spread": 0.10, "coverage": 0.03,
       "persistence": 0.10, "rows": 0.10, "pair": 0.10,
       "missing_clustering": 0.10, "patients": 0.10}


def fixture(seed=3, n_pat=260, n_vis=8):
    """Carries every property a dial can move.

    A fixture that cannot express a property cannot show a dial
    leaking into it - this file's oldest rule. So: a patient
    structure (persistence has somewhere to live), a column with
    real missingness (coverage and clustering do too), a
    relationship (an edge dial has an edge), and an innocent
    bystander column that NO dial names, which is the one most
    likely to catch a leak.
    """
    r = np.random.RandomState(seed)
    g = np.repeat(np.arange(n_pat), n_vis)
    base = r.normal(0, 1, n_pat)[g]
    # WITHIN-PATIENT SERIAL CORRELATION, because `_persistence`
    # centres on each patient's own mean - which removes a
    # patient-level offset entirely. The first version of this
    # fixture planted only a per-patient level, so its
    # within-patient lag-1 was noise by construction and the
    # persistence dial read NO-OP against a fixture that could
    # not express persistence. The fault was the fixture's.
    x = np.empty(len(g))
    sparse = np.empty(len(g))
    miss = np.zeros(len(g), dtype=bool)
    for i in range(n_pat):
        sl = slice(i * n_vis, (i + 1) * n_vis)
        e = r.normal(0, 4, n_vis)
        w = np.empty(n_vis)
        w[0] = e[0]
        for t in range(1, n_vis):
            w[t] = 0.75 * w[t - 1] + e[t]      # AR(1)
        x[sl] = 50 + 10 * base[sl] + w
        sparse[sl] = r.normal(7, 2, n_vis)
        # CLUSTERED missingness: a patient-level propensity plus
        # a run, not independent coin flips. A clustering dial
        # cannot be seen against independent missingness - this
        # file already records that exact fault for fix 3a.
        m = np.zeros(n_vis, dtype=bool)
        if r.random_sample() < 0.5:
            st = r.randint(0, max(1, n_vis - 3))
            m[st:st + r.randint(2, 4)] = True
        miss[sl] = m
    y = 2.5 * x + r.normal(0, 6, len(g))
    z = r.normal(20, 5, len(g))          # the innocent bystander
    sparse[miss] = np.nan
    return pd.DataFrame({
        "person_id": ["P{:04d}".format(i) for i in g],
        "x": np.round(x, 2), "y": np.round(y, 2),
        "z": np.round(z, 2),
        "sparse": np.round(sparse, 2)})


def _persistence(frame, col, group_by):
    """Within-patient lag-1, centred on each patient's own mean.

    Reported as a property to WATCH, never fed back as a
    parameter - this file records that a statistic fed back as a
    generative parameter is attenuated twice.
    """
    v = pd.to_numeric(frame[col], errors="coerce")
    out = []
    for _, idx in frame.groupby(group_by).indices.items():
        s = v.iloc[idx].to_numpy(float)
        s = s[~np.isnan(s)]
        if len(s) < 3:
            continue
        s = s - s.mean()
        if s.std() == 0:
            continue
        out.append(float(np.corrcoef(s[:-1], s[1:])[0, 1]))
    return float(np.nanmean(out)) if out else float("nan")


def _clustering(frame, col, group_by):
    """Lag-1 correlation of the MISSINGNESS INDICATOR, within
    patient. Clustering is about HOW absence is arranged, not
    how much of it there is - the first cut of this matrix used
    coverage as a stand-in for it, which is a different
    property, and so the clustering dial was graded against a
    number it does not govern."""
    v = pd.to_numeric(frame[col], errors="coerce")
    out = []
    for _, idx in frame.groupby(group_by).indices.items():
        m = v.iloc[idx].isna().to_numpy(float)
        if len(m) < 3 or m.std() == 0:
            continue
        out.append(float(np.corrcoef(m[:-1], m[1:])[0, 1]))
    return float(np.nanmean(out)) if out else float("nan")


def properties(frame, cols, group_by):
    p = {}
    for c in cols:
        v = pd.to_numeric(frame[c], errors="coerce")
        p[(c, "centre")] = float(v.mean())
        p[(c, "spread")] = float(v.std())
        p[(c, "coverage")] = float(v.notna().mean())
        p[(c, "persistence")] = _persistence(frame, c, group_by)
        p[(c, "missing_clustering")] = _clustering(
            frame, c, group_by)
    p[("*", "rows")] = float(len(frame) /
                             max(frame[group_by].nunique(), 1))
    # THE COUNT DIAL MOVES THE NUMBER OF PATIENTS, NOT ROWS PER
    # PATIENT - and the first cut graded it against rows per
    # patient, which it leaves at 8.0 either way. A dial judged
    # by a property it does not govern reads NO-OP however well
    # it works.
    p[("*", "patients")] = float(frame[group_by].nunique())
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            va = pd.to_numeric(frame[a], errors="coerce")
            vb = pd.to_numeric(frame[b], errors="coerce")
            m = va.notna() & vb.notna()
            p[("{}~{}".format(a, b), "pair")] = (
                float(va[m].corr(vb[m], method="spearman"))
                if m.sum() > 20 else float("nan"))
    return p


def moved(before, after, key, sd):
    """Did this property move beyond its own tolerance?"""
    name, kind = key
    a, b = before.get(key), after.get(key)
    if a is None or b is None or not np.isfinite(a) \
            or not np.isfinite(b):
        return False, 0.0
    if kind == "centre":
        d = abs(b - a) / (sd or 1.0)
    elif kind in ("spread", "rows", "patients"):
        d = abs(b / a - 1.0) if a else abs(b - a)
    else:
        d = abs(b - a)
    return d > TOL[kind], d


def from_blueprint(path, group_by):
    """Pick the columns a real blueprint can express each dial on.

    A FIXTURE PROVES THE MATRIX WORKS; A REAL BLUEPRINT PROVES
    THE DIALS DO. The fixture has four columns and one
    relationship, chosen so every dial has somewhere to land.
    The extract has forty-four columns and a real graph, where a
    dial has neighbours nobody picked for it - which is the
    whole point, and is where leaks would actually live.

    Chosen from the blueprint rather than named by hand: the
    value column is a well-covered numeric with at least one
    relationship, the sparse one is partially covered, and the
    BYSTANDER is a numeric the graph does not touch at all -
    the control that makes every other row mean something.
    """
    bp = json.loads(Path(path).read_text(encoding="utf-8"))
    cols = bp.get("columns") or {}
    rels = bp.get("relationships") or []
    related = set()
    for r in rels:
        related.add(r.get("child"))
        related.update(r.get("parents") or [])

    def _cov(c):
        return float((cols[c] or {}).get("coverage") or 0.0)

    def _num(c):
        # THE KEY IS `type`, NOT `kind`. The first cut guessed
        # `kind` and matched NOTHING across 142 real columns -
        # value, sparse and bystander all None - so the matrix
        # refused to run. That refusal was the guard working;
        # the guess was the third wrong-key fault of the day,
        # after jointcheck's skill_real and verify_frame's
        # requested_in_sds. The contract is asserted live in
        # the suite now rather than read from memory.
        m = (cols[c] or {}).get("marginal") or {}
        return m.get("type") == "quantiles"

    nums = [c for c in cols if _num(c) and c != group_by]
    val = next((c for c in nums
                if _cov(c) > 0.95 and c in related), None)
    sparse = next((c for c in nums
                   if 0.2 < _cov(c) < 0.8), None)
    bystander = next((c for c in nums
                      if c not in related and _cov(c) > 0.9), None)
    # AN EDGE WHOSE BOTH ENDS ARE NUMERIC. The first pick on
    # the real blueprint was visit_start_date<-visit_end_date,
    # and a rank correlation between two date columns is not
    # the property the strength dial is judged on.
    edge = None
    for r in rels:
        ch, pars = r.get("child"), (r.get("parents") or [])
        if ch in cols and _num(ch):
            par = next((q for q in pars
                        if q in cols and _num(q)), None)
            if par:
                edge = "{}<-{}".format(ch, par)
                break
    return bp, val, sparse, bystander, edge


def run(seed=7, n_pat=260, blueprint=None, group_by="person_id"):
    if blueprint:
        base_bp, val, sparse_c, byst, edge = from_blueprint(
            blueprint, group_by)
        cols = [c for c in (val, sparse_c, byst) if c]
        if edge:
            cols += [c for c in edge.split("<-") if c not in cols]
        print("REAL BLUEPRINT {}".format(blueprint))
        print("  value {}   sparse {}   bystander {}   edge {}"
              .format(val, sparse_c, byst, edge))
        if not val or not byst:
            print("  STOP: this blueprint has no column that can "
                  "carry the value dials, or no bystander the "
                  "graph leaves alone. Without a control the "
                  "matrix cannot tell an effect from noise.")
            return []
        base_gen0 = generate(json.loads(json.dumps(base_bp)),
                             n_patients=n_pat, seed=seed)
        sds = {c: float(pd.to_numeric(base_gen0[c],
                                      errors="coerce").std())
               for c in cols}
        df = None
    else:
        df = fixture()
        cols = ["x", "y", "z", "sparse"]
        val, sparse_c, byst, edge = "x", "sparse", "z", "y<-x"
        cat = discover(df, group_by="person_id", seed=1)
        base_bp = B.build(df, cat, group_by="person_id")
        sds = {c: float(pd.to_numeric(df[c],
                                      errors="coerce").std())
               for c in cols}

    # which columns are DOWNSTREAM of which, read from the
    # blueprint rather than assumed from how the fixture was
    # written - the graph is what decides whether a second
    # column moving is propagation or a leak
    kids = {}
    for rel in (base_bp.get("relationships") or []):
        for par in (rel.get("parents") or []):
            kids.setdefault(par, set()).add(rel.get("child"))

    base_gen = generate(json.loads(json.dumps(base_bp)),
                        n_patients=n_pat, seed=seed)
    before = properties(base_gen, cols, group_by)

    # EVERY dial class, each one alone. The column chosen for
    # each is the one that can express it: `sparse` for the two
    # missingness dials, `x` for the value dials, the x~y edge
    # for strength.
    # EACH DIAL'S SEMANTICS, WRITTEN DOWN - because grading a
    # dial against the wrong ones reads as a NO-OP on a dial
    # that works perfectly. `persistence` is a MULTIPLIER on the
    # column's own serial correlation, not a target value: asked
    # 0.8 against a base of 0.388 it lands 0.313, which is
    # 0.8x to within 0.003. Graded as a target it looks broken.
    # Nothing in the CLI said which dials are absolute and which
    # are multipliers; this table is that answer, and it is
    # ASSERTED rather than described.
    _sd_val = sds.get(val, 1.0) or 1.0
    cases = [
        # TWO SHIFTS, because the dial is clean at one size and
        # destructive at another and a single case can only
        # show one of those. A child is drawn from a curve
        # published over the parent's OBSERVED range, so a
        # small shift stays inside it and a large one walks off
        # the end, where np.interp holds flat. Measured on this
        # fixture, parent~child Spearman over three seeds:
        #     0.0 sd  0.968    2.0 sd  0.832
        #     1.0 sd  0.965    3.0 sd  0.313    5.0 sd  -0.027
        # The 3 sd row is EXPECTED to read LEAKED. It is the
        # matrix catching a real cost, not the matrix failing.
        ("{}.shift".format(val),
         {val: {"shift": 0.5 * _sd_val}}, "absolute"),
        ("{}.shift@3sd".format(val),
         {val: {"shift": 3.0 * _sd_val}}, "absolute"),
        ("{}.scale".format(val), {val: {"scale": 1.5}},
         "multiplier"),
        ("{}.persistence".format(val),
         {val: {"persistence": 0.4}}, "multiplier"),
        ("{}.coverage".format(sparse_c),
         {sparse_c: {"coverage": 0.9}}, "target"),
        ("{}.missing_clustering".format(sparse_c),
         {sparse_c: {"missing_clustering": 0.9}}, "move"),
        ("patients.count", {"patients": {"count": 120}},
         "target"),
        ("patients.visits_scale",
         {"patients": {"visits_scale": 1.5}}, "multiplier"),
        ("{}.strength".format(edge),
         {edge: {"strength": 0.4}}, "multiplier"),
        # A SETTING THAT CHANGES NOTHING, ON PURPOSE. Every real
        # dial here works, so the NO-OP branch had no case that
        # could reach it - an untested branch sitting in the one
        # place built to catch untested behaviour. `scale=1.0`
        # is a legitimate request that moves nothing by
        # definition, and an operator who sets it deserves to be
        # told it did nothing rather than shown a CLEAN row.
        ("{}.scale@1.0".format(val), {val: {"scale": 1.0}},
         "multiplier"),
    ]

    print("DIAL MATRIX - every dial against every neighbouring "
          "property")
    print("  fixture {} patients x {} visits, {} columns, one "
          "relationship".format(
              df["person_id"].nunique(),
              len(df) // df["person_id"].nunique(), len(cols)))
    print("  tolerances: " + "  ".join(
        "{} {:g}".format(k, v) for k, v in TOL.items()))
    print("")
    print("  {:<28} {:<11} {:<9} {}".format(
        "dial", "semantics", "verdict", "what else moved"))
    rows = []
    for label, overlay, sem in cases:
        bp = json.loads(json.dumps(base_bp))
        problems = D.apply(bp, overlay)
        if problems:
            print("  {:<28} {:<11} {:<9} {}".format(
                label, sem, "REFUSED",
                "; ".join(problems)[:60]))
            rows.append({"dial": label, "semantics": sem,
                         "verdict": "REFUSED",
                         "note": "; ".join(problems),
                         "leaks": [], "propagated": [],
                         "landed": ""})
            continue
        # DO NOT OVERRIDE THE DIAL UNDER TEST. The first cut
        # passed n_patients explicitly on every arm, which
        # overrode patients.count and made the working dial read
        # NO-OP. The matrix was testing its own argument.
        kw = {} if "count" in overlay.get("patients", {}) \
            else {"n_patients": n_pat}
        gen = generate(bp, seed=seed, **kw)
        after = properties(gen, cols, group_by)

        tgt_col, _, tgt_dial = label.partition(".")
        tgt_dial = tgt_dial.partition("@")[0]
        # A NO-OP DIAL ALSO MOVES NOTHING, and the first cut of
        # this matrix called that CLEAN. Checking only the
        # neighbours cannot tell "moved exactly what it named"
        # from "did nothing at all" - which is this file's
        # oldest named failure, the flag that is read and not
        # used. The target has to be asserted too.
        targets = []
        leaks = []
        prop = []
        for key in sorted(before, key=lambda k: (k[0], k[1])):
            name, kind = key
            # the dial's OWN property on its OWN column is the
            # thing it is allowed to move
            own = (name == tgt_col
                   and (kind == tgt_dial
                        or (tgt_dial == "shift" and kind == "centre")
                        or (tgt_dial == "scale" and kind == "spread")
                        ))
            if tgt_col == "patients":
                own = kind == ("patients" if tgt_dial == "count"
                               else "rows")
            if tgt_dial == "strength":
                own = kind == "pair" and name == "x~y"
            if own:
                targets.append(key)
                continue
            sd = sds.get(name, 1.0)
            did, d = moved(before, after, key, sd)
            if did:
                # A CHILD INHERITING ITS PARENT'S CHANGE IS NOT
                # A BUG - y is 2.5x plus noise, so flattening
                # x's serial correlation flattens y's too, and
                # that is the relationship doing its job. It is
                # still a SECOND column moving when the operator
                # named one, so it is reported as PROPAGATED
                # rather than hidden or called a leak. The
                # bystander column z is the control: it holds.
                tag = ("propagated" if name in kids.get(
                    tgt_col, ()) else "leak")
                (prop if tag == "propagated" else leaks).append(
                    "{}.{} {:+.2f}".format(name, kind, d))
        hit = any(moved(before, after, k, sds.get(k[0], 1.0))[0]
                  for k in targets)
        # A MULTIPLIER IS GRADED AS A MULTIPLIER. Asked 0.4 of a
        # base of 0.388 is 0.155, and a dial that lands there is
        # exactly right - while a target-value reading of the
        # same number calls it a miss.
        landed = ""
        short = ""
        if sem == "multiplier" and targets and hit:
            k0 = targets[0]
            want = float(list(overlay.values())[0][tgt_dial])
            b0, a0 = before.get(k0), after.get(k0)
            if b0 and np.isfinite(b0) and np.isfinite(a0) \
                    and abs(b0) > 1e-9:
                ratio = a0 / b0
                landed = "  x{:.2f} of base (asked x{:.2f})".format(
                    ratio, want)
                if abs(ratio - want) > 0.15:
                    short = ("landed x{:.2f}, asked x{:.2f}"
                             .format(ratio, want))
        if not targets:
            verdict, note = "NO-TARGET", ("this dial names no "
                                          "property the matrix "
                                          "measures")
        elif not hit:
            verdict, note = "NO-OP", ("the dial moved NOTHING - "
                                      "not even what it names")
        elif short:
            # MOVING IS NOT LANDING. The edge-strength dial
            # asked x0.40 and delivered x0.86 while the matrix
            # called it CLEAN, because "did the property move"
            # and "did it land where asked" are different
            # questions and only the first was being asked. The
            # known cause is a live REVERSE edge carrying the
            # pair - this file records it - and the matrix now
            # surfaces that automatically instead of needing
            # somebody to remember.
            verdict, note = "SHORT", short
        elif leaks:
            verdict, note = "LEAKED", ", ".join(leaks[:4])
        else:
            verdict, note = "CLEAN", "-"
        # ORDERING: the propagation note has to be folded in
        # BEFORE the line is printed. The first cut computed it
        # after, so a real propagation never reached the page -
        # a value assigned downstream of its only reader.
        if prop and verdict == "CLEAN":
            note = "propagated to child: " + ", ".join(prop[:3])
        if verdict == "SHORT" and "<-" in label:
            # A SINGLE-EDGE DIAL WITH A LIVE REVERSE EDGE
            # CANNOT REMOVE THE PAIR, and the number alone
            # reads as a broken dial. Named from the blueprint,
            # not assumed.
            ch, _, pa = label.partition(".")[0].partition("<-")
            rev = any(r.get("child") == pa and ch in
                      (r.get("parents") or [])
                      for r in (base_bp.get("relationships") or []))
            if rev:
                note += (" - the REVERSE edge {}<-{} is still "
                         "live and carries the pair".format(
                             pa, ch))
        print("  {:<28} {:<11} {:<9} {}{}".format(
            label, sem, verdict, note, landed))
        rows.append({"dial": label, "semantics": sem,
                     "verdict": verdict, "note": note,
                     "leaks": leaks, "propagated": prop,
                     "landed": landed.strip()})

    print("")
    bad = [r for r in rows if r["verdict"] != "CLEAN"]
    print("  {} of {} dials move what they name AND nothing "
          "else.".format(len(rows) - len(bad), len(rows)))
    noop = [r["dial"] for r in rows if r["verdict"] == "NO-OP"]
    if noop:
        print("  NO-OP: {} - a dial that moves nothing is worse "
              "than no dial, because the report says it was "
              "applied.".format(", ".join(noop)))
    if bad:
        print("  A LEAK IS NOT A TUNING MISS. The key/released "
              "pair rests on saying exactly what a dial did; a "
              "dial that moves a neighbouring property silently "
              "is one the cipher cannot describe.")
    return rows


if __name__ == "__main__":
    _bp = None
    _gb = "person_id"
    _a = sys.argv[1:]
    for i, tok in enumerate(_a):
        if tok == "--blueprint" and i + 1 < len(_a):
            _bp = _a[i + 1]
        if tok == "--group-by" and i + 1 < len(_a):
            _gb = _a[i + 1]
    rows = run(blueprint=_bp, group_by=_gb)
    sys.exit(0 if all(r["verdict"] in ("CLEAN", "SHORT")
                      for r in rows) else 1)
