"""SYNTH_V1 CLI smoke: the whole pipeline through the command
surface with the stub backend and the real regex extractor —
plan -> render -> evaluate -> experiment, plus compile's
human-handoff path via a canned-backend monkeypatch. Subprocess-free
(direct main() calls) so exit codes and stdout are assertable.

Run from the repo root:

    python scripts/smoke_cli.py
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthkit.cli import main as cli
from synthkit.examples import reference_spec

PASS = 0


def check(label, cond):
    global PASS
    if not cond:
        print("FAIL: {}".format(label))
        sys.exit(1)
    PASS += 1
    print("ok  {:02d}  {}".format(PASS, label))


def run_cli(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            rc = cli(argv)
        except SystemExit as e:
            rc = int(e.code or 0)
    return rc, out.getvalue()


def main():
    tmp = Path(tempfile.mkdtemp(prefix="synthkit_cli_"))
    spec_path = tmp / "spec.json"
    spec = reference_spec(size=10)
    spec_path.write_text(spec.to_json(), encoding="utf-8")

    # ---------- plan ----------
    rc, out = run_cli(["plan", str(spec_path)])
    stats = json.loads(out)
    check("plan validates and prints corpus stats",
          rc == 0 and stats["documents"] == 10
          and "current_medication" in stats["element_density"])

    bad = tmp / "bad_spec.json"
    broken = reference_spec(size=0)
    bad.write_text(broken.to_json(), encoding="utf-8")
    rc, out = run_cli(["plan", str(bad)])
    check("plan rejects an invalid spec with exit 1", rc == 1)

    # ---------- render (stub backend) ----------
    run_dir = tmp / "corpus" / "run_001"
    rc, out = run_cli(["render", str(spec_path),
                       "-o", str(run_dir), "--backend", "stub"])
    check("render writes a verified corpus via the stub backend",
          rc == 0
          and "verified first try: 10" in out
          and (run_dir / "manifest.json").is_file()
          and len(list((run_dir / "docs").glob("*.txt"))) == 10)

    # ---------- evaluate (real regex extractor by dotted path) ----------
    rc, out = run_cli([
        "evaluate", str(run_dir),
        "--extractor", "synthkit.examples:regex_extract",
        "--reference-rules",
        "--json-out", str(tmp / "report.json"),
    ])
    check("evaluate runs the regex extractor against the corpus",
          rc == 0 and "PER ELEMENT" in out
          and "RECALL BY DIFFICULTY" in out)
    report = json.loads((tmp / "report.json")
                        .read_text(encoding="utf-8"))
    check("the naive regex extractor falls into the "
          "discontinued-med trap",
          report["distractors"]["discontinued_medication"]
          ["false_positives"] > 0)
    check("naive regex recall is honest, not flattering",
          0.0 < report["overall_recall"] < 1.0)

    # ---------- experiment (canned, stub backend) ----------
    rc, out = run_cli(["experiment", "--backend", "stub",
                       "--size", "12"])
    check("canned experiment runs and fails the naive extractor",
          rc == 1 and "FAILED" in out
          and "distractors.discontinued_medication.fp_rate" in out
          and "FAILS" in out)

    # ---------- compile: human-handoff on unrepaired problems ----------
    import synthkit.cli as cli_mod

    class CannedBackend:
        name = "canned"

        def __init__(self, payload):
            self.payload = payload

        def complete(self, prompt, *, system="", max_tokens=2000,
                     temperature=0.3):
            return self.payload

    broken_json = reference_spec(size=0).to_json()
    orig = cli_mod._backend
    cli_mod._backend = lambda name, model="": CannedBackend(broken_json)
    try:
        rc, out = run_cli(["compile", "-d", "anything",
                           "-o", str(tmp / "draft.json"),
                           "--backend", "stub"])
        check("compile hands the human a draft plus problems on "
              "failure",
              rc == 1 and "PROBLEMS" in out
              and (tmp / "draft.json").is_file())
        good_json = reference_spec(size=10).to_json()
        cli_mod._backend = lambda name, model="": CannedBackend(good_json)
        rc, out = run_cli(["compile", "-d", "anything",
                           "-o", str(tmp / "good.json"),
                           "--backend", "stub"])
        check("compile succeeds and asks for human review",
              rc == 0 and "review the spec" in out)
    finally:
        cli_mod._backend = orig

    shutil.rmtree(tmp)
    # ---------- dotted imports from the working dir ----------
    # Console entry points lack cwd on sys.path; emulate that and
    # prove the loaders bootstrap it (live catch on the development machine).
    import os as _os
    tmp = tempfile.mkdtemp(prefix="synthkit_cwd_")
    (Path(tmp) / "mymod.py").write_text(
        "from synthkit.evaluator import Extraction\n"
        "def fn(doc_id, text):\n"
        "    return [Extraction(category='x', text='y')]\n",
        encoding="utf-8")
    prev_cwd = _os.getcwd()
    prev_path = list(sys.path)
    try:
        _os.chdir(tmp)
        sys.path[:] = [q for q in sys.path
                       if q not in ("", ".", tmp)]
        sys.modules.pop("mymod", None)
        from synthkit.cli import _load_extractor
        ex = _load_extractor("mymod:fn")
        check("dotted extractors import from the working "
              "directory even without cwd on sys.path",
              ex.extract("d", "t")[0].category == "x")
    finally:
        _os.chdir(prev_cwd)
        sys.path[:] = prev_path
        sys.modules.pop("mymod", None)

    print("\nAll {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
