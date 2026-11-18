"""Smoke: can a run be traced to the file that produced it?

THE CASE THIS EXISTS FOR cannot be caught by a row count. The data
machine holds SIX files called `tidy_visits.csv` and all six have
exactly 55,428 rows, so a run recording the basename and the row
count could not be traced to the one that produced it - and a
paired arm rested on somebody's inference about a directory. The
decisive check below builds a TWIN: same shape, same row count,
one value changed.

AND THE RECORD MUST NOT CARRY A PATH. A full path on that machine
holds the operator's username and provenance TRAVELS with a run,
so only the containing directory's NAME is recorded - asserted,
because it is a privacy property and not a tidiness one.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd                              # noqa: E402

from synthkit import provenance as P              # noqa: E402

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
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        a = td / "a.csv"
        b = td / "b.csv"
        a.write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
        b.write_text("x,y\n1,2\n3,5\n", encoding="utf-8")

        da, db = P.file_digest(a), P.file_digest(b)
        check("two files of the SAME shape and row count digest "
              "differently - which is the whole point, because a "
              "row count cannot tell them apart",
              da and db and da != db)
        check("the digest is stable on a second read",
              P.file_digest(a) == da)
        # A CHECK MUST FAIL, NOT CRASH. Letting the call raise
        # makes the mutation a traceback with no FAIL line,
        # which a grep cannot tell from a check that never ran.
        try:
            _missing = P.file_digest(td / "nope.csv")
        except Exception as e:                   # noqa: BLE001
            _missing = "RAISED " + type(e).__name__
        check("a file that cannot be read gives None rather than "
              "raising, so a guard can report it (got "
              "{!r})".format(_missing),
              _missing is None)

        # THE DIGEST IS OF BYTES, which is why it is portable
        # across stacks where a digest of GENERATED float output
        # is not - this project records such a pin breaking
        # across a pandas release and a Python patch version.
        crlf = td / "crlf.csv"
        crlf.write_bytes(b"x,y\r\n1,2\r\n3,4\r\n")
        check("it digests BYTES - the same rows with different "
              "line endings are a different file, which is what "
              "makes it stack-independent",
              P.file_digest(crlf) != da)

        rec = P.source_record(a, 2, 2, patients=2)
        check("the record carries the digest",
              rec.get("digest") == da)
        check("...and the containing directory's NAME",
              rec.get("parent_name") == td.name)
        blob = json.dumps(rec)
        check("NO PATH SEPARATOR appears anywhere in the record - "
              "a full path on that machine carries a username and "
              "provenance travels with a run",
              "/" not in blob and "\\\\" not in blob)
        check("...and the record says what the digest is of, so a "
              "reader is not left to assume it pins output",
              "BYTES" in (rec.get("digest_note") or ""))

        # --- check_source, every branch
        run = td / "run"
        run.mkdir()
        (run / "provenance.json").write_text(
            json.dumps({"source": rec}), encoding="utf-8")
        ok, how, msg = P.check_source(run, a, 2)
        check("the right file is confirmed, and the check reports "
              "that it was EXACT", ok and how == "exact")
        ok, how, msg = P.check_source(run, b, 2)
        check("THE TWIN IS REFUSED - same shape, same row count, "
              "one value changed - and the reason names the "
              "digests", not ok and how == "exact"
              and "digests differ" in msg)

        # --- the fallback, which must SAY it is weaker
        old = td / "old"
        old.mkdir()
        thin = dict(rec)
        thin.pop("digest")
        (old / "provenance.json").write_text(
            json.dumps({"source": thin}), encoding="utf-8")
        ok, how, msg = P.check_source(old, a, 2)
        check("a run predating the digest proceeds on the ROW "
              "COUNT and the check SAYS so - a reader not told "
              "that believes they got the exact answer",
              ok and how == "rows" and "weaker" in msg)
        ok, how, msg = P.check_source(old, b, 2)
        check("...and on that older record the TWIN IS NOT "
              "CAUGHT, which is exactly the hole the digest "
              "closes", ok and how == "rows")
        ok, how, msg = P.check_source(old, a, 999)
        check("...while a wrong row count still is", not ok
              and how == "rows")

        nothing = td / "none"
        nothing.mkdir()
        ok, how, msg = P.check_source(nothing, a, 2)
        check("a run with no provenance proceeds and says nothing "
              "could be confirmed - absent is not a mismatch",
              ok and how == "none")
        (nothing / "provenance.json").write_text(
            "{ not json", encoding="utf-8")
        ok, how, msg = P.check_source(nothing, a, 2)
        check("...and an unreadable provenance is the same answer "
              "rather than a crash", ok and how == "none")

        # --- THE PIPELINE MUST USE THIS RECORD, asserted against
        # a LIVE run rather than remembered. The wrong-key fault
        # has happened nine times here; a second shape for one
        # record is how two halves come to disagree.
        from smoke_modelrank import fixture
        fsrc = td / "f.csv"
        fixture(fsrc)
        rd = td / "fit"
        r = subprocess.run(
            [sys.executable, "-m", "synthkit.cli", "fit",
             "--src", str(fsrc), "--out", str(rd),
             "--group-by", "person_id", "--generate"],
            cwd=str(ROOT), capture_output=True, text=True,
            timeout=900)
        live = {}
        if (rd / "provenance.json").exists():
            live = json.loads(
                (rd / "provenance.json").read_text(
                    encoding="utf-8")).get("source") or {}
        check("a real run records the source through this module "
              "- every field it produces is present: {}".format(
                  sorted(live)),
              r.returncode == 0
              and all(k in live for k in
                      ("digest", "parent_name", "rows_read",
                       "digest_note")))
        check("...and that digest matches the file it read, so "
              "the guard is exact on a run made today",
              live.get("digest") == P.file_digest(fsrc))
        ok, how, _ = P.check_source(rd, fsrc, len(
            pd.read_csv(fsrc)))
        check("...and the guard confirms it EXACTLY",
              ok and how == "exact")

        # ---- EVERY KEYWORD GENERATE WAS CALLED WITH ------------
        #
        # `settings` carried group_by, the time column, lags,
        # generate, seed, holdout and excluded - and NOT ONE of the
        # eight keywords that decide what generation does. Two runs
        # could differ in whether the constraints were enforced,
        # whether the dates were reordered, whether the identities
        # were enforced or how many refinement sweeps ran, and
        # produce provenance files that agree to the byte. That is
        # the same hole `dials` was added to close, one layer over:
        # `dials` enumerates module CONSTANTS and these are per-run
        # ARGUMENTS.
        #
        # Found by building a curve-trace driver that has to
        # reproduce a run's own generate call and discovering there
        # was nothing to read.
        setts = {}
        if (rd / "provenance.json").exists():
            setts = json.loads(
                (rd / "provenance.json").read_text(
                    encoding="utf-8")).get("settings") or {}
        ga = setts.get("generate_args")
        check("a real run records every keyword generate was "
              "called with: {}".format(
                  ", ".join(sorted(ga or [])) or "NOTHING"),
              isinstance(ga, dict) and len(ga) >= 7)

        # THE KEY SET IS ASSERTED AGAINST THE ENGINE'S OWN
        # SIGNATURE, not against a list written here. A parameter
        # added to `generate` and left out of the recorder would
        # otherwise join the things a run cannot say about itself,
        # silently - and this project has nine wrong-key faults on
        # the record, every one of them a name somebody remembered
        # instead of reading.
        import inspect
        from synthkit.generate import generate as _gfn
        want = set(inspect.signature(_gfn).parameters) - {
            "blueprint", "seed", "report"}
        check("...and the recorded keys are EXACTLY generate's own "
              "parameters, so a ninth argument cannot arrive "
              "unrecorded (missing {}, extra {})".format(
                  sorted(want - set(ga or {})) or "none",
                  sorted(set(ga or {}) - want) or "none"),
              set(ga or {}) == want)

        # AND THE VALUES ARE THE DECISION, not the key set. A
        # recorder whose keys are right and whose values are
        # constants would pass every check above.
        from synthkit.pipeline import _generate_args

        class _NS(object):
            def __init__(self, **kw):
                self.__dict__.update(kw)

        base = dict(patients=0, enforce_constraints=False,
                    refine_sweeps=2, no_reorder_dates=False,
                    no_identities=False)
        d_def = _generate_args(_NS(**base))
        d_off = _generate_args(_NS(**dict(base, no_identities=True,
                                          no_reorder_dates=True,
                                          enforce_constraints=True,
                                          refine_sweeps=0)))
        check("the recorder reads the FLAGS rather than returning "
              "constants - a run made with --no-identities records "
              "enforce_identities False where the default records "
              "True",
              d_def["enforce_identities"] is True
              and d_off["enforce_identities"] is False
              and d_def["reorder_dates"] is True
              and d_off["reorder_dates"] is False
              and d_off["enforce_constraints"] is True
              and d_off["refine_sweeps"] == 0)
        check("...and `--patients 0` is recorded as None rather "
              "than 0, because zero there MEANS the blueprint's "
              "own count and recording the digit would be a "
              "statement nobody made",
              d_def["n_patients"] is None
              and _generate_args(
                  _NS(**dict(base, patients=40)))["n_patients"] == 40)

    print()
    if FAIL:
        print("{} of {} checks failed.".format(FAIL, PASS + FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
