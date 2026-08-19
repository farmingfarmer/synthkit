"""Smoke: the diagnostics bundle travels, and the screen can REFUSE.

WHY THIS EXISTS. Four rounds of this project were run off photographs
of a monitor, because the measurements could not leave the machine
holding the extract. A column got paired with the wrong note that way
and the pairing had to be withdrawn. `export_bundle` is the fix, and
a screen that cannot fail would make it worse than nothing: it would
put a stamp of approval on whatever was in the directory.

So every refusal below is exercised against a bundle that CONTAINS
the thing it is about. A guard tested only against clean input passes
on an empty list.
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.export import build, collect, screen   # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def clean_run(d: Path, n_cols=40):
    """A run directory shaped like a real one."""
    (d / "fidelity.json").write_text(json.dumps({
        "columns": [{"column": "c{}".format(i),
                     "coverage_source": 0.9,
                     "coverage_generated": 0.9} for i in range(n_cols)],
        "summary": {"columns": n_cols, "coverage_ok": n_cols},
        "note": "aggregates",
    }), encoding="utf-8")
    (d / "provenance.json").write_text(json.dumps({
        "build": {"source": "archive", "id": "9395a9ca7d39"},
        "source": {"name": "extract.csv", "rows_read": 55428,
                   "columns_read": n_cols},
        "settings": {"group_by": "person_id"},
    }), encoding="utf-8")
    (d / "findings.txt").write_text("WHAT THE DATA SAYS\n",
                                    encoding="utf-8")


def main():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        clean_run(d)

        b, problems = build(d)
        check("a clean run passes the screen, or every refusal below "
              "is untestable", not problems)
        check("...and the bundle carries the three aggregate parts",
              set(b) >= {"fidelity", "provenance", "findings_txt"})
        check("...and does NOT carry the blueprint, the catalogue or "
              "the generated data - this answers what was measured, "
              "and a bundle that grows is one nobody re-reads before "
              "sending it",
              not any(k in b for k in
                      ("blueprint", "catalogue", "generated")))

        # ---- A PATH. The source is recorded by NAME on purpose.
        prov = json.loads((d / "provenance.json").read_text())
        prov["source"]["name"] = "C:\\Users\\NE1002766\\dev\\x.csv"
        (d / "provenance.json").write_text(json.dumps(prov),
                                           encoding="utf-8")
        probs = screen(collect(d))
        check("a Windows path is REFUSED - that is the shape that "
              "carries a work login, and it is why provenance names "
              "the source instead of pathing it",
              any("path" in p for p in probs))

        prov["source"]["name"] = "/Users/someone/dev/x.csv"
        (d / "provenance.json").write_text(json.dumps(prov),
                                           encoding="utf-8")
        check("...and so is a POSIX home path",
              any("path" in p for p in screen(collect(d))))

        # ---- A ROW. Everything here is per-column or per-pair.
        clean_run(d, n_cols=8)          # 8 columns -> 64 pairs
        fid = json.loads((d / "fidelity.json").read_text())
        fid["per_row_residual"] = [0.1] * 5000
        (d / "fidelity.json").write_text(json.dumps(fid),
                                         encoding="utf-8")
        probs = screen(collect(d))
        check("an array longer than the table has column PAIRS is "
              "refused - every legitimate field here is per-column or "
              "per-pair, so a long one is a per-row field somebody "
              "added or a leak, and a human should look either way",
              any("per-ROW" in p for p in probs))
        check("...and the refusal names the field and both counts, "
              "or it cannot be acted on",
              any("per_row_residual" in p and "5000" in p
                  for p in probs))

        # AND IT MUST NOT FIRE ON THE LEGITIMATE LONG ONE. `constraints`
        # is one entry per ordered pair and ran to 241 on a real
        # 83-column run; a cap that refused that would be turned off
        # within a day.
        clean_run(d, n_cols=83)
        fid = json.loads((d / "fidelity.json").read_text())
        fid["constraints"] = [{"lhs": "a", "rhs": "b"}] * 241
        (d / "fidelity.json").write_text(json.dumps(fid),
                                         encoding="utf-8")
        check("...while 241 constraints on an 83-column table pass - "
              "a cap that refused the real run would be disabled "
              "within a day and protect nothing",
              not screen(collect(d)))

        # ---- NO BUILD. Results read against the wrong tree have
        # already cost this project several rounds.
        clean_run(d)
        prov = json.loads((d / "provenance.json").read_text())
        prov["build"] = {"source": "unknown", "id": None}
        (d / "provenance.json").write_text(json.dumps(prov),
                                           encoding="utf-8")
        check("a bundle that cannot name its commit is refused - "
              "undiagnosable, and reading output against a tree that "
              "did not produce it is a mistake already made here",
              any("build id" in p for p in screen(collect(d))))

        # ---- A MISSING PART is NAMED, not silently dropped.
        clean_run(d)
        (d / "fidelity.json").unlink()
        b2 = collect(d)
        check("a run with no fidelity.json says which part is absent "
              "rather than producing a smaller bundle that looks "
              "complete", b2.get("missing") == ["fidelity.json"])

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
