"""What does TODAY'S gate say about an OLD run?

WHY THIS EXISTS. A gate percentage is only a comparison when both
sides are measured over the same pair set, and this project has
twice drawn a conclusion from one that was not. Tonight the
recorded deliverable read `close 120/137 = 87.6%` and a fresh run
read `101/125 = 80.8%` - and diffing the pair sets by name showed
160 in both, 36 deliverable-only, **34 of those 36 set
scaffolding**. Two numbers over different surfaces.

The fix is not to argue about it. An old run still has its
`generated.csv` and its `blueprint.json`, so its gate can be
RE-MEASURED with today's code against today's source, and then
both sides are the same measurement. Nothing is refitted and
nothing is regenerated: the old output is read off disk exactly
as it was written.

    python scripts/regate.py RUNDIR --src TIDY.csv
    python scripts/regate.py RUN_A RUN_B --src TIDY.csv   (both)

The row count the run RECORDED is checked against the source it
is handed, because this project has already measured a run
against an extract that never produced it and every number
looked reasonable.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", message=".*Could not infer format.*")

from synthkit import gate as G                      # noqa: E402
from synthkit.pipeline import compare               # noqa: E402


def provenance_rows(run):
    f = run / "provenance.json"
    if not f.exists():
        return None
    p = json.loads(f.read_text(encoding="utf-8"))
    for key in ("rows", "source_rows", "n_rows"):
        if isinstance(p.get(key), int):
            return p[key]
    src = p.get("source")
    if isinstance(src, dict):
        for key in ("rows", "n_rows"):
            if isinstance(src.get(key), int):
                return src[key]
    return None


def regate(run, src_df, src_rows):
    run = Path(run)
    bpf, genf = run / "blueprint.json", run / "generated.csv"
    if not bpf.exists() or not genf.exists():
        return {"run": run.name,
                "why": "needs blueprint.json and generated.csv - "
                       "this re-measures a FINISHED run"}
    rec = provenance_rows(run)
    if rec is not None and rec != src_rows:
        # A RUN MEASURED AGAINST AN EXTRACT THAT NEVER PRODUCED IT
        # looks entirely reasonable, which is why this refuses by
        # name rather than warning.
        return {"run": run.name,
                "why": "the run recorded {} source rows and this "
                       "source has {} - refusing rather than "
                       "producing a plausible number about the "
                       "wrong file".format(rec, src_rows)}
    bp = json.loads(bpf.read_text(encoding="utf-8"))
    g = pd.read_csv(genf, dtype=str, keep_default_na=False)
    tc = (bp.get("patients") or {}).get("time_column")
    gid = (bp.get("patients") or {}).get("id_column") or "person_id"
    fid = compare(src_df, g, bp, gid, tc)
    res = G.assess(fid)
    out = {"run": run.name, "met": res.get("met"), "criteria": {}}
    for c in res["criteria"]:
        out["criteria"][c["name"]] = {
            "ok": c["ok"], "detail": c["detail"][:70]}
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    src = pd.read_csv(a.src, dtype=str, keep_default_na=False)
    print("source rows {}".format(len(src)))
    print()
    for r in a.runs:
        got = regate(r, src, len(src))
        if "why" in got:
            print("{:<22} SKIPPED: {}".format(got["run"],
                                              got["why"]))
            continue
        print("{} met={}".format(got["run"], got["met"]))
        for name, c in got["criteria"].items():
            print("  {:<22} {:<5} {}".format(
                name[:22], "PASS" if c["ok"] else "FAIL",
                c["detail"]))
        print()
    print("BOTH SIDES ARE NOW THE SAME MEASUREMENT - same source, "
          "same code, same pair surface.")
    print("A difference here is the OUTPUT differing, which a "
          "difference between two recorded runs is not.")
    print()
    print("AND THESE FIGURES NEED NOT MATCH A RUN'S OWN RECORDED "
          "ONES. A run measures the frame it")
    print("generated in memory; this re-reads the written CSV, so "
          "a rendered date comes back as text and")
    print("is parsed again. The comparability on offer is between "
          "the runs listed here, not against")
    print("a number in an old fidelity.json - which is the whole "
          "reason this script exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
