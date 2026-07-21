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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

    print("\nAll {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
