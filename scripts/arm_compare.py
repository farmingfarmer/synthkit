"""Two runs side by side, with every uncontrolled difference named.

WHY THIS EXISTS. A paired arm is the only thing that has ever
overturned a fidelity claim in this project, and it is also the
thing most easily faked. The record carries three ways it has gone
wrong here, all three producing a table that looked like a result:

  * TWO ENGINES. `run_deliverable` was built by the neural engine
    and every arm measured against it by the rules engine - "close
    119 against 107" was one engine against another, and five
    control arms that could not reach their own mechanism were run
    on the strength of it.
  * A MOVING DENOMINATOR. Two runs count pairs over their own
    blueprint's token vocabulary, so 137 against 125 is not a
    comparison and the percentages are not either.
  * A DEFAULT NOBODY COULD SEE. A control arm run on an older tree
    differs from its partner by every default changed in between,
    and nothing in the run directory said so.

So this refuses the comparisons that cannot be valid, NAMES the
ones that can be read with care, and makes the caller state what
the arm is for. It computes nothing: every figure is read from the
two run directories, because a second implementation of the gate
is how a report and a verdict come to disagree.

    python scripts/arm_compare.py RUN_A RUN_B --testing "TEXT"
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# A DIFFERENT SOURCE OR A DIFFERENT SEED IS NOT AN ARM. Everything
# else is readable with the difference stated; these two make the
# two columns answers to different questions.
FATAL = ("source", "seed")


def _load(run, name):
    p = Path(run) / name
    if not p.exists():
        raise SystemExit(
            "STOPPED: {} has no {} - this is not a finished run"
            .format(run, name))
    return json.loads(p.read_text(encoding="utf-8"))


def identity(run):
    """What a run says about itself, from provenance alone."""
    prov = _load(run, "provenance.json")
    src = prov.get("source") or {}
    sett = prov.get("settings") or {}
    build = prov.get("build")
    if isinstance(build, dict):
        build = build.get("id") or build.get("sha") or str(build)
    return {
        "source": src.get("name"),
        "rows": src.get("rows_read"),
        "patients": src.get("patients"),
        "seed": sett.get("seed"),
        "engine": sett.get("engine") or "rules",
        "build": build,
        "settings": sett,
        "stack": prov.get("stack") or {},
        # EVERY DIAL THE ENGINE WAS HOLDING. Without this two runs
        # could differ in what the sampler DID and agree in every
        # recorded field, which is how a cross-build control arm
        # came to be invisible.
        "dials": prov.get("dials") or {},
        "marginal_metric": None,
    }


def differences(a, b):
    """(fatal, readable) - every field the two runs disagree on.

    A FIELD MISSING ON ONE SIDE IS A DIFFERENCE, not a match. An
    older run that never recorded a setting cannot be assumed to
    have had the current default, which is the whole reason the
    cross-build arm was invalid.
    """
    fatal, readable = [], []
    for k in ("source", "rows", "patients", "seed", "engine"):
        if a.get(k) != b.get(k):
            row = (k, a.get(k), b.get(k))
            (fatal if k in FATAL or k == "engine"
             else readable).append(row)
    if a.get("build") != b.get("build"):
        readable.append(("build", a.get("build"), b.get("build")))
    keys = set(a["settings"]) | set(b["settings"])
    for k in sorted(keys):
        if k == "seed":
            continue
        if a["settings"].get(k) != b["settings"].get(k):
            readable.append(("settings." + k,
                             a["settings"].get(k),
                             b["settings"].get(k)))
    da, db = a.get("dials") or {}, b.get("dials") or {}
    for grp in sorted(set(da) | set(db)):
        ga = (da.get(grp) or {})
        gb = (db.get(grp) or {})
        for k in sorted(set(ga) | set(gb)):
            if ga.get(k) != gb.get(k):
                readable.append(("dials.{}.{}".format(grp, k),
                                 ga.get(k), gb.get(k)))
    if bool(da) != bool(db):
        readable.append(
            ("dials recorded at all", bool(da), bool(db)))
    # THE RULER ITSELF CAN MOVE. The marginal block states the
    # metric it was measured with, and that metric was changed
    # once already - from the worst single band to the share of
    # mass in the wrong band, over a finer set of edges. Two runs
    # measured on different rulers are not comparable on that
    # block however well their flags match, and the two numbers
    # look exactly alike.
    ma = ((a.get("marginal_metric") or "") or "")[:60]
    mb = ((b.get("marginal_metric") or "") or "")[:60]
    if ma != mb:
        readable.append(("marginal metric (the RULER)",
                         ma or "not recorded",
                         mb or "not recorded"))
    for k in ("python", "pandas", "numpy", "sklearn", "scipy"):
        if (a["stack"] or {}).get(k) != (b["stack"] or {}).get(k):
            readable.append(("stack." + k,
                             (a["stack"] or {}).get(k),
                             (b["stack"] or {}).get(k)))
    return fatal, readable


def tiers(run):
    """Which sampler tiers this run RECORDS as having fired.

    A tier that cannot prove it executed has not been deployed -
    two round trips to the machine that holds the extract were
    spent on a tier that silently skipped. The generate report is
    carried into fidelity.json whole, so this is a read.
    """
    fid = _load(run, "fidelity.json")
    gen = fid.get("generation") or {}
    out = {}
    for k, v in sorted(gen.items()):
        if isinstance(v, (list, dict)) and v:
            out[k] = len(v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = v
    return out


def readings(run):
    """The numbers, read - never recomputed."""
    fid = _load(run, "fidelity.json")
    summ = fid.get("summary") or {}
    mg = fid.get("marginals") or {}
    tv = fid.get("tvd") or {}
    # THE PAIR COUNTS LIVE IN `summary`, NOT IN `pairs`.
    #
    # The first version of this read `fid["pairs"]` for
    # `compared`, `sign_kept`, `close` and `inverted` - the names
    # the pair BLOCK uses internally - and `fid["pairs"]` is a
    # dict whose own `pairs` key is a list of per-pair rows. So
    # all five came back None and every pair row was SILENTLY
    # SKIPPED by the "both sides missing" rule, and the first real
    # arm printed coverage, tokens, dynamics and the marginals
    # with no close, no direction and no INVERTED - the criteria
    # the arm was for. The tenth wrong-key fault on the record,
    # inside a tool written in the same session as a check
    # prescribing that the names be asserted against a live call.
    return {
        "related": summ.get("pairs"),
        "direction": summ.get("pairs_sign_ok"),
        "close": summ.get("pairs_close"),
        "inverted": summ.get("pairs_inverted"),
        "own_inverted": summ.get("pairs_own_inverted"),
        "own_related": summ.get("pairs_own"),
        "own_close": summ.get("pairs_own_close"),
        "coverage_ok": summ.get("coverage_ok"),
        "coverage_n": summ.get("coverage_compared"),
        "dynamics_ok": summ.get("dynamics_ok"),
        "dynamics_n": summ.get("dynamics_compared"),
        "token_ok": summ.get("set_tokens_ok"),
        "token_n": summ.get("set_tokens_compared"),
        "marg_n": mg.get("columns_measured"),
        "marg_off": len(mg.get("off_source") or []),
        "marg_off_names": mg.get("off_source") or [],
        "marg_worst": mg.get("worst_source_vs_output"),
        "marg_sampler": len(mg.get("sampler_faults") or []),
        "tvd_accuracy": tv.get("accuracy"),
    }


# THE ABSOLUTE COUNTS ARE THE COMPARISON. A denominator that moved
# makes a percentage a different question, and this file records
# three occasions where a conclusion was drawn from one.
ROWS = [
    ("pairs related", "related", None),
    ("direction kept", "direction", "related"),
    ("close", "close", "related"),
    ("INVERTED", "inverted", None),
    # THE OPERATOR'S OWN COLUMNS, counted apart. Four fifths of
    # the graded surface on this extract is set-token scaffolding
    # dropped before the file is written, and the absolute stop
    # judges the columns a recipient actually receives.
    ("own-column pairs", "own_related", None),
    ("own-column close", "own_close", "own_related"),
    ("own-column INVERTED", "own_inverted", None),
    ("coverage", "coverage_ok", "coverage_n"),
    ("set token shares", "token_ok", "token_n"),
    ("dynamics tracked", "dynamics_ok", "dynamics_n"),
    ("marginal: off the source", "marg_off", "marg_n"),
    ("marginal: the sampler's", "marg_sampler", "marg_n"),
]


def main():
    ap = argparse.ArgumentParser(
        description="two runs side by side, differences named")
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument(
        "--testing", required=True,
        help="what this arm is testing, in your own words. "
             "Required: a table with no stated question is read as "
             "whichever difference the reader notices first.")
    a = ap.parse_args()

    # A HALF-WRITTEN DIRECTORY IS NAMED AS ONE. Without this the
    # missing fidelity surfaces later as a SOURCE mismatch, which
    # sends a reader to check which extract they pointed at - the
    # wrong fault, and the recorded cancellation case left exactly
    # such a directory behind.
    for _r in (a.run_a, a.run_b):
        for _f in ("provenance.json", "fidelity.json"):
            if not (Path(_r) / _f).exists():
                raise SystemExit(
                    "STOPPED: {} has no {} - this is not a "
                    "finished run".format(_r, _f))

    ia, ib = identity(a.run_a), identity(a.run_b)
    for _i, _r in ((ia, a.run_a), (ib, a.run_b)):
        _i["marginal_metric"] = (
            (_load(_r, "fidelity.json").get("marginals") or {})
            .get("metric"))
    fatal, readable = differences(ia, ib)

    print("ARM: {}".format(a.testing))
    print("  A  {}".format(a.run_a))
    print("  B  {}".format(a.run_b))
    print("")
    print("source   {} / {} rows / {} patients".format(
        ia["source"], ia["rows"], ia["patients"]))
    print("seed     {}".format(ia["seed"]))
    print("")

    if fatal:
        for k, x, y in fatal:
            print("STOPPED: {} differs - A {} against B {}"
                  .format(k, x, y))
        print("")
        print("These two runs are not an arm. A different source, "
              "a different seed or a different ENGINE makes the "
              "two columns answers to different questions, and a "
              "table of them reads as a result.")
        sys.exit(2)

    if readable:
        print("UNCONTROLLED DIFFERENCES - {} of them, and every "
              "one is a thing this arm did not mean to test:"
              .format(len(readable)))
        for k, x, y in readable:
            print("  {:<28} A {!r:<22} B {!r}".format(k, x, y))
        print("")
        print("  A build difference is the dangerous one: it "
              "carries every default changed in between, and the "
              "run directory does not list them.")
        print("")
    else:
        print("No difference visible in provenance beyond the "
              "flags below - which is what a paired arm looks "
              "like.")
        print("")

    ta, tb = tiers(a.run_a), tiers(a.run_b)
    only = sorted(set(ta) ^ set(tb))
    if only:
        print("TIERS RECORDED ON ONE SIDE ONLY:")
        for k in only:
            print("  {:<28} A {!s:<10} B {!s}".format(
                k, ta.get(k, "-"), tb.get(k, "-")))
        print("")

    ra, rb = readings(a.run_a), readings(a.run_b)
    missing = []
    print("  {:<28} {:>12} {:>12} {:>9}".format(
        "criterion", "A", "B", "B - A"))
    for label, key, den in ROWS:
        va, vb = ra.get(key), rb.get(key)
        if va is None and vb is None:
            # SAY IT, do not skip it. A row absent from both runs
            # is indistinguishable from a key read under the
            # wrong name, and the first real arm printed no close,
            # no direction and no INVERTED for exactly that
            # reason while looking like a finished table.
            missing.append(label)
            continue
        da = ra.get(den) if den else None
        db = rb.get(den) if den else None
        sa = "{}".format(va) if da is None \
            else "{}/{}".format(va, da)
        sb = "{}".format(vb) if db is None \
            else "{}/{}".format(vb, db)
        delta = ("" if va is None or vb is None
                 else "{:+d}".format(int(vb) - int(va)))
        mark = ""
        if den is not None and da != db:
            mark = "  <- denominator MOVED"
        print("  {:<28} {:>12} {:>12} {:>9}{}".format(
            label, sa, sb, delta, mark))
    for label, key in (("marginal worst", "marg_worst"),
                       ("TVD accuracy", "tvd_accuracy")):
        va, vb = ra.get(key), rb.get(key)
        if va is None or vb is None:
            continue
        print("  {:<28} {:>12.4f} {:>12.4f} {:>+9.4f}".format(
            label, va, vb, vb - va))

    if missing:
        print("")
        print("  NEITHER RUN CARRIES: {}".format(
            ", ".join(missing)))
        print("    A row absent from both is indistinguishable "
              "from a key read under the wrong name, which is why "
              "it is printed rather than skipped.")

    gained = sorted(set(ra["marg_off_names"]) - set(rb["marg_off_names"]))
    lost = sorted(set(rb["marg_off_names"]) - set(ra["marg_off_names"]))
    if gained or lost:
        print("")
        print("  marginal shape, by NAME - a count cannot say "
              "which column moved:")
        if gained:
            print("    B fixed: " + ", ".join(gained))
        if lost:
            print("    B broke: " + ", ".join(lost))

    print("")
    print("Read the ABSOLUTE counts. Where a denominator moved the "
          "percentages are not a comparison, and this project has "
          "three times drawn a conclusion from one that did.")


if __name__ == "__main__":
    main()
