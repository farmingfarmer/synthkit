"""Smoke: find the time axis whatever shape it arrived in, and lag
only what is worth lagging.

An automated system cannot be told which column orders its visits, so
the detection is tested against a date, against a bare sequence
number, and against a table with neither - where the honest answer is
to say file order was assumed rather than to pretend.
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.temporal import (                    # noqa: E402
    detect_time_column, add_lag_features, LAG_SUFFIX)

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def build(kind, n_pat=120, n_vis=8, seed=3):
    rnd = random.Random(seed)
    rows = []
    for p in range(n_pat):
        trait = rnd.choice(["A", "B"])
        prev = None
        for v in range(n_vis):
            r = {"person_id": "P{:04d}".format(p),
                 # a fixed trait: its previous value IS its value
                 "sex": trait,
                 # varies within a patient
                 "lab": round(rnd.gauss(50, 8), 3),
                 # only 3% coverage - not worth a lag feature
                 "rare": (round(rnd.gauss(0, 1), 3)
                          if rnd.random() < 0.03 else ""),
                 # y follows the PREVIOUS x
                 "x": round(rnd.uniform(0, 1), 4)}
            r["y"] = round((2.0 * prev if prev is not None
                            else rnd.uniform(0, 2))
                           + rnd.gauss(0, 0.1), 4)
            prev = r["x"]
            if kind == "date":
                r["when"] = "2021-{:02d}-{:02d}".format(
                    (v % 12) + 1, (v % 27) + 1)
            elif kind == "sequence":
                r["visit_no"] = v + 1
            rows.append(r)
    return rows


def main():
    # ---- detection ------------------------------------------------
    col, kind, ev = detect_time_column(build("date"), "person_id")
    check("a DATE column is found and named as the time axis",
          col == "when" and kind == "date-like" and "parse" in ev)
    col, kind, ev = detect_time_column(build("sequence"), "person_id")
    check("a bare SEQUENCE number is found when there is no date - "
          "an automated system cannot be told which column it is",
          col == "visit_no" and kind == "sequence")
    col, kind, ev = detect_time_column(build("none"), "person_id")
    check("with neither, it says FILE ORDER was assumed rather than "
          "inventing an axis",
          col is None and kind == "file-order"
          and "assumption" in ev)
    check("...and says so in words a reader can act on",
          "rather than a measurement" in ev)

    # ---- START before END, measured rather than named -------------
    # Ranking date columns by distinct count chose visit_END_date on a
    # real extract. An end date orders a stay by when it finished, so
    # overlapping admissions come out in the wrong sequence and every
    # lag and dynamics statistic is measured across visits that were
    # never adjacent - measured on a fixture, 41 patients in 200 have
    # a different visit ORDER under the two dates.
    import datetime as _dt

    def two_dates(spread, n_pat=200, seed=2):
        rr = random.Random(seed)
        base = _dt.date(2021, 1, 1)
        out = []
        for p in range(n_pat):
            d = 0
            for _v in range(8):
                d += rr.randint(1, 10)
                st = base + _dt.timedelta(days=d)
                en = st + _dt.timedelta(days=rr.randint(0, spread))
                out.append({"person_id": "P{:03d}".format(p),
                            "visit_start_date": st.isoformat(),
                            "visit_end_date": en.isoformat(),
                            "lab": rr.random()})
        return out

    wide = two_dates(60)
    n_end = len(set(r["visit_end_date"] for r in wide))
    n_start = len(set(r["visit_start_date"] for r in wide))
    check("the end date really does have more distinct values, which "
          "is what made a distinct-count ranking choose it - without "
          "this the fixture cannot reproduce the fault",
          n_end > n_start)
    col, kind, ev = detect_time_column(wide, "person_id")
    check("the START date is chosen anyway, because which column "
          "began the visit is a MEASUREMENT - at or before the other "
          "on nearly every row - and not a guess from its name",
          col == "visit_start_date")
    check("...and the reason says so, with the share it was measured "
          "on", "BEGAN" in ev and "strictly earlier" in ev)

    same = two_dates(0)
    c2, _k2, e2 = detect_time_column(same, "person_id")
    check("two dates that are IDENTICAL on every row order visits the "
          "same way, so neither is claimed to be the beginning - a "
          "swap there would be a measurement nobody made",
          "BEGAN" not in e2)

    # ---- what earns a lag -----------------------------------------
    rows = build("date")
    col, kind, _ = detect_time_column(rows, "person_id")
    out, rep = add_lag_features(rows, "person_id", col, kind)
    lagged = set(rep["lagged_columns"])
    check("a column that VARIES within a patient earns a lag",
          "x" in lagged and "lab" in lagged)
    check("a FIXED trait does not - its previous value is its value, "
          "and the search would rediscover that as a finding",
          "sex" not in lagged)
    check("a column too sparse to carry one does not either",
          "rare" not in lagged)
    check("the time column is not lagged against itself",
          col not in lagged)
    check("the report names the axis it used, so the choice is "
          "visible rather than implicit",
          rep["time_column"] == "when"
          and rep["time_kind"] == "date-like")

    # ---- the feature is correct -----------------------------------
    by = {}
    for r in out:
        by.setdefault(r["person_id"], []).append(r)
    one = by["P0000"]
    check("the FIRST visit has no previous value, and says so with a "
          "blank rather than a fabricated number",
          one[0]["x" + LAG_SUFFIX] == "")
    check("each later visit carries the previous visit's value "
          "exactly",
          all(abs(float(one[i]["x" + LAG_SUFFIX])
                  - float(one[i - 1]["x"])) < 1e-9
              for i in range(1, len(one))))

    # last OBSERVED, not last row
    gap = [{"person_id": "P1", "t": 1, "v": 5.0},
           {"person_id": "P1", "t": 2, "v": ""},
           {"person_id": "P1", "t": 3, "v": ""},
           {"person_id": "P1", "t": 4, "v": 9.0}]
    g2, _ = add_lag_features(gap * 30, "person_id", "t", "sequence",
                             columns=["v"])
    fourth = [r for r in g2 if str(r["v"]) == "9.0"]
    check("the lag carries the last OBSERVED value across a gap, not "
          "the last row's blank - at 11% coverage the previous row is "
          "almost always empty and the feature would be useless",
          fourth and str(fourth[0]["v" + LAG_SUFFIX]) == "5.0")

    # ---- the relationship becomes findable ------------------------
    def corr(xs, ys):
        m1, m2 = sum(xs) / len(xs), sum(ys) / len(ys)
        nu = sum((a - m1) * (b - m2) for a, b in zip(xs, ys))
        de = (sum((a - m1) ** 2 for a in xs)
              * sum((b - m2) ** 2 for b in ys)) ** 0.5
        return nu / de if de else 0.0
    pairs = [(float(r["x" + LAG_SUFFIX]), float(r["y"]))
             for r in out if str(r["x" + LAG_SUFFIX]).strip()]
    same = [(float(r["x"]), float(r["y"])) for r in out]
    check("the engineered feature makes an invisible relationship "
          "VISIBLE: y correlates with x__prev while showing nothing "
          "against x at the same visit",
          corr([a for a, _ in pairs], [b for _, b in pairs]) > 0.9
          and abs(corr([a for a, _ in same],
                       [b for _, b in same])) < 0.15)

    # ---- the report's SHAPE does not move between branches -------
    #
    # `added` read `[]` on the no-rows branch and a COUNT on the
    # main one, for as long as both existed. A caller writing
    # `len(added)` worked on an empty frame and raised on a real
    # one; a caller writing `added > 0` did the reverse. Neither
    # fails at the branch that is wrong, which is what makes a type
    # that moves between branches worse than a type that is wrong -
    # and the bench's --lags path raised a TypeError on the real
    # frame rather than on the fixture that was tested.
    _, empty = add_lag_features([], "person_id", "when", "date-like")
    check("the no-rows report carries the SAME KEYS as a real one, "
          "so a caller cannot be written against one branch and "
          "crash on the other",
          sorted(empty) == sorted(rep))
    check("...and `added` is a COUNT on both branches, never a list "
          "on one and an int on the other",
          isinstance(empty["added"], int)
          and isinstance(rep["added"], int))
    check("a real frame's `added` counts the columns it names, which "
          "is what makes the count checkable at all",
          rep["added"] == len(rep["lagged_columns"]))

    # ---- A LAG PARENT MUST NOT BECOME A SAME-VISIT PARENT -------
    #
    # `discover._source` folds `x__prev` to `x` so the catalogue
    # reads in the operator's own column names - "x moved this",
    # which is what a reader needs. It also erases WHEN, which is
    # the whole substance of a lagged finding, and the claim's own
    # `via` field recorded that and nothing read it. So a
    # relationship discovered on the PREVIOUS visit's value
    # arrived at the blueprint as a predictor named `x` and was
    # applied as a same-visit curve.
    #
    # Measured on the bench with `--lags`, four seeds: the output
    # came back with a same-visit rank correlation of 0.97 on a
    # pair the SOURCE relates at 0.00, where without the flag it
    # correctly reads 0.00. An invented relationship is worse than
    # a missing one, because it reads as a finding.
    #
    # THE FIXTURE CARRIES BOTH KINDS, because a fix that removed
    # the invented pattern by also removing the honest one would
    # pass a check that only looked at the lagged pair.
    import numpy as _np
    import pandas as _pd
    from synthkit.discover import discover as _disc
    from synthkit import blueprint as _B
    from synthkit.generate import generate as _gen

    def _lagfix_fixture(seed=0, n=260, per=8):
        rg = _np.random.default_rng(seed)
        pid = _np.repeat(["P{:04d}".format(i) for i in range(n)],
                         per)
        N = len(pid)
        day = _np.tile(_np.arange(per), n)
        x = rg.normal(50, 10, N)
        xs = x.reshape(n, per)
        sh = _np.concatenate(
            [_np.full((n, 1), _np.nan), xs[:, :-1]], axis=1)
        y = 0.9 * (sh - 50) + 50 + 4.4 * rg.normal(0, 1, (n, per))
        y = _np.where(_np.isnan(y), rg.normal(50, 10, (n, per)), y)
        u = rg.normal(20, 5, N)
        w = 0.9 * (u - 20) * 2 + 40 + 2.2 * rg.normal(0, 1, N)
        return _pd.DataFrame({
            "person_id": pid,
            "visit_start_date": (_pd.to_datetime("2020-01-01")
                                 + _pd.to_timedelta(day, "D")),
            "x": x, "y": y.reshape(-1),
            "u": u, "w": w}).astype(str)

    def _sp(f, a, b):
        pp = _pd.to_numeric(f[a], errors="coerce")
        qq = _pd.to_numeric(f[b], errors="coerce")
        kk = pp.notna() & qq.notna()
        return float(pp[kk].rank().corr(qq[kk].rank()))

    _raw = _lagfix_fixture()
    _rws, _inf = add_lag_features(_raw.to_dict("records"),
                                  "person_id", "visit_start_date",
                                  "date-like")
    _lf = _pd.DataFrame(_rws)
    _s_same = _sp(_lf, "x", "y")
    _s_lag = _sp(_lf, "x__prev", "y")
    _s_honest = _sp(_lf, "u", "w")
    check("the fixture plants a genuinely LAGGED pair - x~y at the "
          "same visit {:+.3f}, x__prev~y {:+.3f} - which is the "
          "shape the fold is wrong about; a pair related at both "
          "offsets could not reach it"
          .format(_s_same, _s_lag),
          abs(_s_same) < 0.1 and abs(_s_lag) >= 0.5)
    check("...and an HONEST same-visit pair beside it (u~w "
          "{:+.3f}), because a fix that removed the invented "
          "pattern by also removing the real one would pass a "
          "check that looked only at the lagged pair"
          .format(_s_honest),
          abs(_s_honest) >= 0.5)

    _cat = _disc(_lf, group_by="person_id", seed=1)
    _bp = _B.build(_lf, _cat, group_by="person_id")
    _rel = dict((r["child"], list(r["parents"]))
                for r in (_bp.get("relationships") or []))
    _drop = _bp["excluded"].get("lag_only_parents") or []
    check("the catalogue DOES find it, reported under the base "
          "name with the lag recorded in `via` - the fold is not "
          "the bug, reading it as same-visit is",
          any(c["child"] == "y"
              and any(p["column"] == "x"
                      and (p.get("via") or []) == ["x__prev"]
                      for p in c["predictors"])
              for c in _cat["claims"]))
    _lagp = dict((r["child"], (r["evidence"].get("lag_parents")
                               or []))
                 for r in (_bp.get("relationships") or []))
    check("...and the blueprint MARKS it as read from the "
          "PREVIOUS visit rather than publishing a same-visit "
          "curve (y's lag parents are {}): the curve discovery "
          "measured IS the lag's, so only the column it is read "
          "from was wrong"
          .format(_lagp.get("y")),
          "x" in (_lagp.get("y") or []))
    check("...and it is still a PARENT ({}), because dropping it "
          "would lose a relationship the data has - the first "
          "version of this fix dropped it, which removed the "
          "invented pattern by removing the finding too"
          .format(_rel.get("y")),
          "x" in (_rel.get("y") or []))
    check("...and the HONEST same-visit parent is NOT marked, so "
          "the marker is a statement about the evidence rather "
          "than about every lag-carried parent (w's lag parents "
          "{})".format(_lagp.get("w")),
          "u" not in (_lagp.get("w") or []))
    check("...and the HONEST pair is KEPT ({} <- {}): the fold is "
          "only wrong when it changes what the claim is about, so "
          "a lag-carried parent whose base column DOES relate at "
          "the same visit still counts"
          .format("w", _rel.get("w")),
          ("u" in (_rel.get("w") or []))
          or ("w" in (_rel.get("u") or [])))

    _g = _gen(_bp, n_patients=260, seed=5)
    _g_same = _sp(_g, "x", "y")
    _g_honest = _sp(_g, "u", "w")

    # THE POINT OF THE WHOLE CHANGE: the lagged relationship is
    # CARRIED now, where it was absent rather than degraded.
    #
    # Read by the order each file DECLARES. A source extract has
    # no visit index, so its dates are its only sequence; the
    # generated file emits `visit_number` as 1..n in row order,
    # which asserts one. Measured on this engine, only 0.8% of
    # generated patients have their DATES in row order while
    # `visit_number` is in row order on 100% of them - so reading
    # the output by its date gives +0.31 where reading it by the
    # column the file itself numbers gives +0.86. That gap is a
    # separate defect IN THE OUTPUT, present with or without
    # `--lags`: two columns in one file disagreeing about the
    # order of a patient's visits.
    def _lagc(f, parent, child):
        ordc = ("visit_number" if "visit_number" in f.columns
                else "visit_start_date")
        q = f[["person_id", ordc, parent, child]].copy()
        q[parent] = _pd.to_numeric(q[parent], errors="coerce")
        q[child] = _pd.to_numeric(q[child], errors="coerce")
        o = _pd.to_numeric(q[ordc], errors="coerce")
        q["_o"] = (o if o.notna().all()
                   else _pd.to_datetime(q[ordc], errors="coerce"))
        q = q.sort_values(["person_id", "_o"], kind="stable")
        q["_p"] = q.groupby("person_id", sort=False)[parent].shift(1)
        kk = q["_p"].notna() & q[child].notna()
        return float(q["_p"][kk].rank().corr(q[child][kk].rank()))

    _lag_s = _lagc(_lf, "x", "y")
    _lag_g = _lagc(_g, "x", "y")
    check("THE LAGGED RELATIONSHIP IS CARRIED: source {:+.3f}, "
          "generated {:+.3f}, each read by the order its own file "
          "declares - it was absent rather than degraded before "
          "(-0.002), because the curve was applied to the current "
          "visit".format(_lag_s, _lag_g),
          abs(_lag_g) >= 0.4 * abs(_lag_s)
          and _np.sign(_lag_g) == _np.sign(_lag_s))
    check("THE OBSERVABLE: the generated same-visit correlation "
          "is {:+.3f} against a source {:+.3f}, where applying "
          "the folded claim produced +0.79 - an invented "
          "relationship reads as a finding, which is worse than "
          "a missing one".format(_g_same, _s_same),
          abs(_g_same) < 0.15)
    check("...and the honest relationship still lands ({:+.3f} "
          "against {:+.3f}), so the change acts only where its "
          "mechanism lives".format(_g_honest, _s_honest),
          abs(_g_honest) >= 0.5 * abs(_s_honest))

    # ---- AND A U-SHAPE AT THE SAME VISIT MUST SURVIVE IT --------
    #
    # RANK CORRELATION CANNOT SEE A U-SHAPE - this project already
    # has that rule for the gate's `close` criterion - so deciding
    # a lag-carried parent on Spearman alone would delete exactly
    # the non-monotone same-visit relationships the fixtures plant
    # on purpose. Found because a mutation making `_lag_carried`
    # fire on EVERY predictor broke no check: the guard that saved
    # it was `bool(via)`, which is not a statement about the
    # source at all.
    def _ushape_fixture(seed=0, n=300, per=6):
        rg = _np.random.default_rng(seed)
        pid = _np.repeat(["Q{:04d}".format(i) for i in range(n)],
                         per)
        N = len(pid)
        day = _np.tile(_np.arange(per), n)
        z = rg.normal(0, 1, N)
        # a genuine SAME-VISIT U-shape: no linear component
        v = 6.0 * (z ** 2) + 0.9 * rg.normal(0, 1, N)
        return _pd.DataFrame({
            "person_id": pid,
            "visit_start_date": (_pd.to_datetime("2021-01-01")
                                 + _pd.to_timedelta(day, "D")),
            "z": z, "v": v}).astype(str)

    _uraw = _ushape_fixture()
    _urws, _ = add_lag_features(_uraw.to_dict("records"),
                                "person_id", "visit_start_date",
                                "date-like")
    _ulf = _pd.DataFrame(_urws)
    _u_rho = _sp(_ulf, "z", "v")
    from synthkit.blueprint import _same_visit_bend as _bend
    _u_bend = _bend(_ulf.apply(_pd.to_numeric, errors="coerce"),
                    "z", "v", _ulf["person_id"].to_numpy(), 10)
    check("the U-shape fixture is INVISIBLE to rank correlation "
          "({:+.3f}) and plain on the bend measure ({:.2f} sd) - "
          "a fixture whose U-shape had a linear component could "
          "not reach this"
          .format(_u_rho, _u_bend or -1.0),
          abs(_u_rho) < 0.15
          and _u_bend is not None and _u_bend >= 0.35)
    _ucat = _disc(_ulf, group_by="person_id", seed=1)
    _ubp = _B.build(_ulf, _ucat, group_by="person_id")
    _urel = dict((r["child"], list(r["parents"]))
                 for r in (_ubp.get("relationships") or []))
    check("...and the lag-carried U-shaped parent is KEPT ({}), "
          "because the bend says the source relates the pair at "
          "the same visit even though Spearman reads zero"
          .format(_urel),
          ("z" in (_urel.get("v") or []))
          or ("v" in (_urel.get("z") or [])))

    # ---- AND AN INTERACTION-ONLY PARENT MUST SURVIVE IT ---------
    #
    # An interaction with NO main effect has near-zero rank
    # correlation AND near-zero bend in each parent ALONE - that
    # is the definition of it, and this project plants exactly
    # that shape on purpose. So neither of the two measures above
    # can keep it, and the first version of this drop deleted one:
    # on a pure-product fixture with `--lags` on,
    # `c <- ['a', 'b']` became `c <- ['a']` at rho +0.014 and
    # bend 0.33 against a 0.35 bar. Half an interaction pair
    # cannot fire at generation, so that is the same fault class
    # the drop exists to prevent, and it would have shipped.
    def _xor_fixture(seed=0, n=300, per=6):
        rg = _np.random.default_rng(seed)
        pid = _np.repeat(["X{:04d}".format(i) for i in range(n)],
                         per)
        N = len(pid)
        day = _np.tile(_np.arange(per), n)
        a = rg.normal(0, 1, N)
        b = rg.normal(0, 1, N)
        c = 8.0 * a * b + 0.8 * rg.normal(0, 1, N)
        return _pd.DataFrame({
            "person_id": pid,
            "visit_start_date": (_pd.to_datetime("2022-01-01")
                                 + _pd.to_timedelta(day, "D")),
            "a": a, "b": b, "c": c}).astype(str)

    _xraw = _xor_fixture()
    _x_rho_a = _sp(_xraw, "a", "c")
    _x_rho_b = _sp(_xraw, "b", "c")
    check("the XOR fixture has NO main effect in either parent "
          "(a~c {:+.3f}, b~c {:+.3f}), which is what makes both "
          "same-visit measures blind to it"
          .format(_x_rho_a, _x_rho_b),
          abs(_x_rho_a) < 0.15 and abs(_x_rho_b) < 0.15)
    _xrws, _ = add_lag_features(_xraw.to_dict("records"),
                                "person_id", "visit_start_date",
                                "date-like")
    _xlf = _pd.DataFrame(_xrws)
    _xbp = _B.build(_xlf, _disc(_xlf, group_by="person_id", seed=1),
                    group_by="person_id")
    _xrel = dict((r["child"], list(r["parents"]))
                 for r in (_xbp.get("relationships") or []))
    check("...and BOTH members of the pair survive with --lags on "
          "(c <- {}), because a parent the claim's own "
          "`interaction.pair` names is kept whatever either "
          "measure says - half a pair cannot fire at generation"
          .format(_xrel.get("c")),
          set(_xrel.get("c") or []) >= {"a", "b"})

    # ---- AND IT IS INERT WITHOUT `--lags` -----------------------
    #
    # The whole lag path hangs off `evidence.lag_parents`, which
    # only a run that BUILT lag features can produce - so on every
    # blueprint this project has ever written the branch cannot
    # fire, and the output must be byte-identical. That is the
    # property that makes a change to the sampler core safe to
    # ship, and it is asserted rather than argued: the same
    # fixture with no lag features, generated twice, against a
    # blueprint whose lag field is forced empty.
    import copy as _copy
    import hashlib as _hl
    _nolag_bp = _B.build(_raw, _disc(_raw, group_by="person_id",
                                     seed=1),
                         group_by="person_id")
    _marks = [r["evidence"].get("lag_parents")
              for r in (_nolag_bp.get("relationships") or [])]
    check("a blueprint built with NO lag features marks no parent "
          "as lagged ({}), so the sampler's lag path cannot fire "
          "on one".format(_marks),
          all(not mk for mk in _marks))
    _stripped = _copy.deepcopy(_nolag_bp)
    for _r in (_stripped.get("relationships") or []):
        _r["evidence"].pop("lag_parents", None)
    _h1 = _hl.sha256(_gen(_nolag_bp, n_patients=200, seed=7)
                     .to_csv(index=False).encode()).hexdigest()
    _h2 = _hl.sha256(_gen(_stripped, n_patients=200, seed=7)
                     .to_csv(index=False).encode()).hexdigest()
    check("...and an OLDER artifact that predates the field "
          "entirely generates byte-identically to one carrying it "
          "empty - absent is not a different fact from empty "
          "here, and an old blueprint must still load",
          _h1 == _h2)

    # ---- A PATIENT'S DATES GO IN VISIT ORDER --------------------
    #
    # `visit_number` is emitted as 1..n in ROW order, which
    # ASSERTS a sequence; a date column is drawn from its marginal,
    # which does not - measured on a two-date fixture, the start
    # dates came out in row order on 0.2% of patients. Two columns
    # in one file disagreeing about the order of a patient's
    # visits is this project's own alarm, and the lagged-parent
    # path applies its curve in ROW order, so without this the
    # two readings of one file disagree about whether a lagged
    # relationship is there.
    #
    # THE GATE WAS NEVER THE VICTIM: `pipeline.compare` already
    # orders the generated frame by `visit_number`, and all seven
    # criteria are identical on both arms. This fixes the FILE.
    def _dates_fixture(seed=0, n=300, per=8):
        rg = _np.random.default_rng(seed)
        pid = _np.repeat(["D{:04d}".format(i) for i in range(n)],
                         per)
        N = len(pid)
        gaps = rg.integers(3, 40, (n, per)).cumsum(axis=1)
        start = gaps.reshape(-1)
        los = rg.integers(1, 9, N)
        return _pd.DataFrame({
            "person_id": pid,
            "visit_start_date": (_pd.to_datetime("2019-01-01")
                                 + _pd.to_timedelta(start, "D")),
            "visit_end_date": (_pd.to_datetime("2019-01-01")
                               + _pd.to_timedelta(start + los, "D")),
            "val": rg.normal(20, 4, N)}).astype(str)

    def _row_order_share(f, col):
        d = _pd.to_datetime(f[col], errors="coerce")
        return float(d.groupby(f["person_id"])
                     .apply(lambda ss: ss.is_monotonic_increasing)
                     .mean())

    _dsrc = _dates_fixture()
    _dbp = _B.build(_dsrc, _disc(_dsrc, group_by="person_id",
                                 seed=1),
                    group_by="person_id",
                    time_col="visit_start_date")
    check("the blueprint RECORDS the time axis it measured the "
          "dynamics along ({!r}), so the sampler does not need a "
          "second detection rule - two halves deciding separately "
          "what the time axis is is how they come to disagree"
          .format((_dbp.get("patients") or {}).get("time_column")),
          (_dbp.get("patients") or {}).get("time_column")
          == "visit_start_date")
    _drep = {}
    _dg = _gen(_dbp, n_patients=300, seed=5, report=_drep)
    _share = _row_order_share(_dg, "visit_start_date")
    check("...and every patient's dates come out in the order the "
          "file numbers them ({:.1%} of patients), where a "
          "marginal draw agrees on almost none"
          .format(_share),
          _share >= 0.99)
    _rec = _drep.get("dates_put_in_visit_order") or {}
    check("...and the run SAYS it did it, naming the axis and the "
          "columns ({}), because a rearrangement nobody is told "
          "about is a change made behind the operator's back"
          .format(sorted(_rec.get("columns") or [])),
          _rec.get("time_column") == "visit_start_date"
          and "visit_end_date" in (_rec.get("columns") or []))

    def _dsp(f, a, b):
        def num(col):
            v = _pd.to_datetime(f[col], errors="coerce")
            return (_pd.to_numeric(v, errors="coerce")
                    if v.notna().any()
                    else _pd.to_numeric(f[col], errors="coerce"))
        u, w = num(a), num(b)
        kk = u.notna() & w.notna()
        return float(u[kk].rank().corr(w[kk].rank()))

    _dd = _dsp(_dg, "visit_start_date", "visit_end_date")
    check("ONE permutation serves every date column, so a "
          "date-to-date near-identity survives (start~end "
          "{:+.4f} against a source {:+.4f}) - the real extract "
          "declares that pair at skill 0.9999, and permuting one "
          "while the other stayed would destroy it"
          .format(_dd, _dsp(_dsrc, "visit_start_date",
                            "visit_end_date")),
          abs(_dd) >= 0.95)
    # AND THE STRONGER FORM: each row keeps its OWN pair, so a
    # per-row constraint the repair enforced is still enforced
    # afterwards. Asserted with enforcement ON, because without
    # it `end >= start` holds on 92.7% of rows in BOTH arms - the
    # first version of this check asserted it unconditionally and
    # went red against code that was behaving correctly, which is
    # a check whose premise was wrong rather than a defect.
    _dge = _gen(_dbp, n_patients=300, seed=5,
                enforce_constraints=True)
    _es = _pd.to_datetime(_dge["visit_start_date"],
                          errors="coerce")
    _ee = _pd.to_datetime(_dge["visit_end_date"], errors="coerce")
    check("...and each row keeps its OWN pair, so a per-row "
          "constraint survives the rearrangement: with "
          "enforcement on, end >= start on {:.2%} of rows AFTER "
          "the dates were reordered"
          .format(float((_ee >= _es).mean())),
          bool((_ee >= _es).all())
          and _row_order_share(_dge, "visit_start_date") >= 0.99)

    # AND IT IS A PERMUTATION WITHIN EACH PATIENT, asserted on
    # the function itself with planted input rather than on a
    # generated frame. The first version of this check compared
    # only the LENGTH of the date column, which cannot fail - and
    # a mutation replacing the grouped `lexsort` with a GLOBAL
    # `argsort`, which moves dates between patients and changes
    # every patient's multiset, broke nothing.
    from synthkit.generate import _dates_in_visit_order as _div
    # THE TWO ENTITIES' DATE RANGES MUST OVERLAP, or a GLOBAL
    # sort gives the same answer as a grouped one and the fixture
    # cannot tell them apart - the first version had A entirely
    # below B, and a mutation replacing the grouped `lexsort`
    # with a plain `argsort` passed all 45 checks.
    _pf = _pd.DataFrame({
        "pid": ["A", "A", "A", "B", "B", "B"],
        "d1": [50.0, 10.0, 30.0, 60.0, 20.0, 40.0],
        "d2": [51.0, 11.0, 31.0, 61.0, 21.0, 41.0],
        "other": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    _before = dict((k, sorted(v)) for k, v in
                   _pf.groupby("pid")["d1"].apply(list).items())
    _pfo = _div(_pf.copy(), "pid", ["d1", "d2"], "d1")
    _after = dict((k, sorted(v)) for k, v in
                  _pfo.groupby("pid")["d1"].apply(list).items())
    check("the reorder is a permutation WITHIN each entity - every "
          "entity's own date multiset is unchanged ({} vs {}) - so "
          "a global sort that moved dates between patients could "
          "not pass".format(_before, _after),
          _before == _after)
    check("...and it actually ORDERS them, which a no-op would "
          "also satisfy the line above: {}"
          .format(list(_pfo["d1"])),
          list(_pfo["d1"]) == [10.0, 30.0, 50.0, 20.0, 40.0,
                               60.0])
    check("...and the SECOND date column follows the SAME "
          "permutation rather than being sorted on its own, which "
          "is what keeps each row's pair together: {}"
          .format(list(_pfo["d2"])),
          list(_pfo["d2"]) == [11.0, 31.0, 51.0, 21.0, 41.0,
                               61.0])
    check("...and a NON-date column does not move - that is the "
          "stated price, and silently moving it would be a "
          "different change: {}".format(list(_pfo["other"])),
          list(_pfo["other"]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
