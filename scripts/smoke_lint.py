"""SYNTH_L1 smoke: the semantic linter proven — every check
recreates the live incident that motivated it: the 71% prevalence
against a 15-20% ask, the near-single-class degenerate, absurd
derived magnitudes, dropped mess clauses, fossil distributions,
ignored name pools — plus the clean bill of health on a spec that
means what it says.

Run from the repo root:

    python scripts/smoke_lint.py
"""
from __future__ import annotations

import io
import re as _re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthkit.examples import reference_table
from synthkit.lint import lint_table
from synthkit.tablespec import TableSpec, TableSpecError

PASS = 0


def check(label, cond):
    global PASS
    if not cond:
        print("FAIL: {}".format(label))
        sys.exit(1)
    PASS += 1
    print("ok  {:02d}  {}".format(PASS, label))


def encounter(intercept=-3.8, target=None) -> TableSpec:
    spec = reference_table(rows=300, master_seed=9)
    oc = {"name": "readmitted", "kind": "logistic",
          "intercept": intercept,
          "coefficients": {"los_days": 0.1, "age": 0.02,
                           "department=oncology": 0.8}}
    if target:
        oc["target_prevalence"] = target
    spec.outcomes = [oc]
    return spec


def codes(report):
    return [f.code for f in report.findings]


def warns(report):
    return [f.code for f in report.findings if f.level == "WARN"]


def main():
    # ---------- validation of the new field ----------
    bad = encounter(target=[0.9, 0.2])
    try:
        bad.validate()
        check("malformed target_prevalence rejected", False)
    except TableSpecError as e:
        check("malformed target_prevalence rejected",
              "target_prevalence" in str(e))

    # ---------- L1: the 71% incident ----------
    report = lint_table(encounter(intercept=-1.4,
                                  target=[0.15, 0.20]))
    check("the live 71%-vs-target incident warns with intercept "
          "advice",
          "L1-prevalence" in warns(report)
          and "more negative = rarer" in report.format_text())
    report = lint_table(encounter(intercept=-3.8,
                                  target=[0.10, 0.25]))
    check("prevalence inside its target is a clean INFO",
          "L1-prevalence" in codes(report)
          and "L1-prevalence" not in warns(report))
    report = lint_table(encounter(intercept=-9.0))
    check("near-single-class outcomes warn as degenerate",
          "L1-degenerate" in warns(report))
    report = lint_table(encounter())
    check("undeclared targets get the enforcement hint",
          "target_prevalence" in report.format_text())

    # ---------- L2: derived sanity ----------
    spec = encounter()
    for rule in spec.rules:
        if rule.get("kind") == "derived":
            rule["factor"] = 1e15
    report = lint_table(spec)
    check("absurd derived magnitudes warn with the "
          "factor/sigma pointer",
          "L2-derived" in warns(report)
          and "scale belongs in factor" in report.format_text())
    spec = encounter()
    for rule in spec.rules:
        if rule.get("kind") == "derived":
            rule["factor"] = 0.0
    report = lint_table(spec)
    check("all-zero derived columns warn",
          "L2-derived" in warns(report))

    # ---------- L4: fossil distributions ----------
    spec = encounter()
    for col in spec.columns:
        if col.name == "discharge_date":
            col.distribution = {"kind": "date_range",
                                "start": "2026-01-01",
                                "end": "2026-06-30"}
    report = lint_table(spec)
    check("fossil distributions on rule targets surface as INFO",
          "L4-fossil" in codes(report)
          and "L4-fossil" not in warns(report))

    # ---------- L5: ignored name pools ----------
    spec = encounter()
    for col in spec.columns:
        if col.ctype == "person_name":
            col.distribution = {"kind": "categorical",
                                "choices": ["Smith", "Jones"]}
    report = lint_table(spec)
    check("the live Smith-list incident surfaces as INFO",
          "L5-names" in codes(report))

    # ---------- L6: the dropped-outlier incident ----------
    spec = encounter()
    for col in spec.columns:
        col.mess.outlier_rate = 0.0
    report = lint_table(
        spec, description="a few extreme outlier charges and "
                          "some duplicate rows")
    check("mess clauses in the English but absent in the spec "
          "warn",
          "L6-coverage" in warns(report)
          and "outlier" in report.format_text())
    check("clauses that ARE covered stay silent",
          sum(1 for f in report.findings
              if f.code == "L6-coverage") == 1)

    # ---------- the clean bill ----------
    spec = encounter(intercept=-3.8, target=[0.10, 0.25])
    report = lint_table(
        spec, description="duplicate rows, missing ages, typos, "
                          "outliers, wrong values")
    check("a spec that means what it says lints ok",
          report.ok)
    report2 = lint_table(spec, description="duplicate rows")
    check("lint is deterministic",
          report.probe_rows == report2.probe_rows
          and lint_table(spec).format_text()
          == lint_table(spec).format_text())

    # ---------- corpus lint ----------
    from synthkit.examples import reference_spec
    from synthkit.lint import lint_corpus
    doc = reference_spec(size=12)
    report = lint_corpus(doc)
    check("the reference corpus lints clean", report.ok)
    broken = reference_spec(size=12)
    for uf in broken.unstructured_fields:
        for el in uf.target_elements:
            if el.element_id == "followup_appointment":
                el.density = 0.001
    report = lint_corpus(broken)
    check("never-planted elements warn (D1)",
          any(f.code == "D1-never-planted"
              and "followup_appointment" in f.message
              for f in report.findings))
    toothless = reference_spec(size=12)
    for uf in toothless.unstructured_fields:
        for dis in uf.distractors:
            dis.density = 0.001
    report = lint_corpus(toothless, probe_docs=8)
    check("unloaded traps warn (D2)",
          any(f.code == "D2-toothless"
              for f in report.findings))
    report = lint_corpus(
        reference_spec(size=12),
        description="include distractor traps please")
    check("trap-language with traps present stays quiet (D5)",
          not any(f.code == "D5-coverage"
                  for f in report.findings))

    # ---------- CLI + GUI wiring ----------
    import contextlib
    import io as _io
    import json as _json
    import shutil
    import tempfile
    import threading
    import urllib.request
    from synthkit.cli import main as cli
    tmp = Path(tempfile.mkdtemp(prefix="synthkit_l_"))
    good = encounter(intercept=-3.8, target=[0.10, 0.25])
    (tmp / "s.json").write_text(good.to_json(), encoding="utf-8")
    bad = encounter(intercept=-1.4, target=[0.15, 0.20])
    (tmp / "b.json").write_text(bad.to_json(), encoding="utf-8")

    def run_cli(argv):
        out = _io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                rc = cli(argv)
            except SystemExit as e:
                rc = int(e.code or 0)
        return rc, out.getvalue()

    rc, out = run_cli(["table-lint", str(tmp / "s.json")])
    check("CLI lint passes a truthful spec with exit 0",
          rc == 0 and "SEMANTIC LINT" in out)
    rc, out = run_cli(["table-lint", str(tmp / "b.json")])
    check("CLI lint fails the 71% spec with exit 1",
          rc == 1 and "L1-prevalence" in out)

    import synthkit.gui as gui
    server = gui.make_server(0)
    threading.Thread(target=server.serve_forever,
                     daemon=True).start()
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:{}/api/lint".format(
                server.server_port),
            data=_json.dumps({
                "kind": "table",
                "spec": bad.to_json(),
                "description": ""}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = _json.loads(r.read().decode("utf-8"))
    finally:
        server.shutdown()
    check("GUI lint endpoint judges over the wire",
          d["ok"] is False and "L1-prevalence" in d["text"])
    shutil.rmtree(tmp)

    # ---- NO SUITE MAY REACH ITS SUCCESS MARKER WITH A FAILURE
    #
    # `smoke_latent` carried `if FAIL: sys.exit(1)` MID-FILE,
    # with sixteen checks below it and an unconditional
    # "All N checks passed." at the end - so a red check in
    # those blocks printed the success marker AND exited 0, and
    # the net judges on exactly that pair. Both lied. This file
    # already records the shape twice (eleven vanishing checks
    # in smoke_patterndial, six in smoke_dialmatrix); those made
    # checks DISAPPEAR, this made a FAILURE read as green.
    #
    # The property is narrow on purpose. "No checks after the
    # exit" is WRONG - `smoke_dials` DEFINES a check function
    # after its guard and CALLS it before, which no lexical scan
    # can see. What cannot be excused is a success marker
    # reachable with FAIL non-zero, and a fail-fast suite whose
    # `check()` exits by itself can never get there.
    def _check_exits(src):
        m = _re.search(r"\ndef check\(.*?(?=\ndef |\nif __name__|\Z)",
                       src, _re.S)
        return bool(m and "sys.exit" in m.group(0))

    def _marker_guarded(src):
        lines = src.splitlines()
        hits = [j for j, l in enumerate(lines)
                if "All {} checks passed" in l]
        if not hits:
            return True
        i, seen = hits[-1] - 1, []
        while i >= 0 and len(seen) < 3:
            t = lines[i].strip()
            if t and not t.startswith("#"):
                seen.append(t)
            i -= 1
        j = " ".join(seen)
        return "sys.exit(1)" in j and "if FAIL" in j

    _suites = sorted(Path(__file__).resolve().parent
                     .glob("smoke_*.py"))
    _unguarded, _accum = [], 0
    for _f in _suites:
        _src = io.open(str(_f), encoding="utf-8").read()
        if "All {} checks passed" not in _src or _check_exits(_src):
            continue
        _accum += 1
        if not _marker_guarded(_src):
            _unguarded.append(_f.name)
    check("every accumulating suite GUARDS its success marker - "
          "{} of {} suites accumulate rather than exiting from "
          "inside check(), and a reachable \"All N checks "
          "passed.\" is how a red check came to print the "
          "success line AND exit 0, which is the exact pair the "
          "net judges on{}"
          .format(_accum, len(_suites),
                  "" if not _unguarded
                  else " - unguarded: {}".format(_unguarded)),
          _accum >= 20 and not _unguarded)
    check("...and the audit SKIPS fail-fast suites rather than "
          "flagging them, because a `check()` that exits on the "
          "first failure can never reach the marker with FAIL "
          "set - {} of {} suites are that kind"
          .format(len(_suites) - _accum, len(_suites)),
          len(_suites) - _accum >= 10)

    # ---- A SCRIPT MUST REFUSE AN ARGUMENT IT DOES NOT KNOW -----
    #
    # TWICE NOW A COMMAND HAS DONE SOMETHING PLAUSIBLE INSTEAD OF
    # REFUSING. `latent_challenger.py` picked its subcommand from
    # `sys.argv[1]` and let anything unmatched fall through to the
    # default, so `--help` spent minutes fitting a planted fixture
    # and then printed numbers that read as an answer to whatever
    # was asked. `shape_sweep.py` dispatched on
    # `"--neural" in sys.argv`, so `--help` ran the whole 23-shape
    # sweep. Both cost real time before anyone noticed.
    #
    # A STANDING AUDIT, not a list: any script that dispatches by
    # testing MEMBERSHIP in `sys.argv` has no parser to refuse for
    # it, so it must refuse for itself. argparse-based scripts are
    # exempt because argparse already does.
    # ASSERT THE DECISION, NOT THE MESSAGE. The first version of
    # this audit searched each script's TEXT for the word
    # "STOPPED" or "unknown", which is a check on wording: a
    # script that refuses perfectly well without using either
    # word failed it. What matters is the exit code a caller
    # reads - non-zero for an argument the script does not know,
    # ZERO for a help request - and that the help request returns
    # without DOING the work, which is the whole fault.
    import subprocess as _sp
    import time as _tm
    _scripts = sorted((ROOT / "scripts").glob("*.py"))
    _hand, _unguarded = [], []
    for f in _scripts:
        if f.name.startswith("smoke_"):
            continue
        t = io.open(str(f), encoding="utf-8").read()
        if "argparse" in t:
            continue
        if "in sys.argv" not in t and "sys.argv[1]" not in t:
            continue
        _hand.append(f.name)
        _t0 = _tm.time()
        try:
            _h = _sp.run([sys.executable, str(f), "--help"],
                         capture_output=True, text=True,
                         timeout=30)
            _b = _sp.run([sys.executable, str(f), "--definitely-not"],
                         capture_output=True, text=True,
                         timeout=30)
        except _sp.TimeoutExpired:
            # A HELP REQUEST THAT TAKES THIRTY SECONDS IS DOING
            # THE WORK, which is exactly the fault: both
            # `latent_challenger.py` and `shape_sweep.py` ran
            # their whole job for `--help`.
            _unguarded.append("{} (timed out)".format(f.name))
            continue
        _why = []
        if _h.returncode != 0:
            _why.append("--help exits {}".format(_h.returncode))
        if _b.returncode == 0:
            _why.append("unknown argument exits 0")
        if _tm.time() - _t0 > 20:
            _why.append("--help is slow enough to be working")
        if _why:
            _unguarded.append("{} ({})".format(f.name,
                                               "; ".join(_why)))
    check("every script that dispatches on `sys.argv` without a "
          "parser ANSWERS --help with exit 0 and REFUSES an "
          "argument it does not know with a non-zero one - {} "
          "such script(s), unguarded: {}"
          .format(len(_hand), _unguarded or "none"),
          not _unguarded)
    check("...and the audit actually FOUND scripts of that kind "
          "({}), because an audit that matches nothing passes "
          "trivially - which is the worst kind of clean"
          .format(_hand),
          len(_hand) >= 2)

    # ---- NO SUITE MAY PIN A DIGEST OF FLOATING-POINT OUTPUT ----
    #
    # TWICE IN ONE DAY A RECORDED SHA FAILED SOMEBODY ELSE'S
    # MACHINE while the engine behaved identically. A catalogue
    # digest differed between pandas 2.3 and pandas 3 and broke
    # the team's own 3.10 lane; a generated-CSV digest differed
    # between Python 3.14.6 and 3.14.8 - IDENTICAL pandas, numpy
    # and scikit-learn - and broke the data machine. The first
    # time I diagnosed it as a class and then fixed only the
    # instance in front of me, which is why there was a second.
    #
    # Comparing two digests computed in the SAME run is fine and
    # is how both surviving uses work: it is a statement about
    # two arms on one machine. What is banned is a hex literal
    # standing in for an expected value.
    # LENGTH IS WHAT SEPARATES THE TWO, and the scanner must look
    # at EVERY suite rather than only the ones that hash
    # something. The first version of this audit skipped any file
    # without `hexdigest` - and a mutation putting the pinned
    # digest back went unnoticed, because removing the comparison
    # had removed the word too. The hole was in the audit's own
    # premise.
    #
    # A full 40-char git SHA is a legitimate identifier and two
    # suites use one on purpose (a planted folder name, and the
    # commit a personal-data scan is grandfathered through). A
    # TRUNCATED hex literal is what `hexdigest()[:16]` produces,
    # so 12 to 32 characters is the banned shape.
    _pin = _re.compile(r"=\s*[\"']([0-9a-f]{12,32})[\"']")
    _sha40 = _re.compile(r"[\"']([0-9a-f]{40})[\"']")
    _pinned, _legit = [], []
    for f in sorted((ROOT / "scripts").glob("smoke_*.py")):
        t = io.open(str(f), encoding="utf-8").read()
        for m in _pin.finditer(t):
            _pinned.append("{}:{}".format(f.name, m.group(1)[:10]))
        if _sha40.search(t):
            _legit.append(f.name)
    check("no suite pins a TRUNCATED digest as an expected value "
          "- a recorded sha is not portable across a library or "
          "even a Python PATCH release, and two arms on one "
          "machine say the same thing portably ({})"
          .format(_pinned or "none"),
          not _pinned)
    # NON-VACUITY IS ASSERTED ON A PLANTED STRING, NOT ON A FILE
    # COUNT. The first version required TWO suites carrying a
    # 40-char SHA - and the TEAM BUNDLE excludes
    # `smoke_no_personal.py`, which is one of them, because it
    # contains the scan patterns themselves. So a check written
    # to prove the scanner still matches went red inside the
    # filtered tree, which is the second time tonight a premise
    # rested on a file that does not travel. Planting the input
    # makes the check mean the same thing in both trees.
    _probe_banned = 'X = "' + "a" * 16 + '"'
    _probe_ok = 'X = "' + "b" * 40 + '"'
    check("the scanner matches a planted TRUNCATED digest and "
          "does NOT match a planted 40-char SHA, so a pattern "
          "that stopped matching would announce itself here "
          "rather than by counting files",
          bool(_pin.search(_probe_banned))
          and not _pin.search(_probe_ok)
          and bool(_sha40.search(_probe_ok)))
    check("...and it reports which suites carry a real 40-char "
          "SHA on purpose ({}) as information rather than as a "
          "threshold - the bundle ships one fewer of them by "
          "design".format(_legit),
          isinstance(_legit, list))

    print("\nAll {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
