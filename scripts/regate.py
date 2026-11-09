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
        # `rows_read` FIRST, because it is the name BOTH
        # writers actually use - `pipeline.py` and
        # `latent_challenger.py` both emit
        # `source.rows_read`, and this list used to hold only
        # `rows` and `n_rows`. So the refusal that exists to
        # stop a run being measured against an extract that
        # never produced it had NEVER FIRED on a real run: a
        # guard that cannot fire, in the one place a whole
        # session was already lost. The others are kept for
        # older artifacts.
        for key in ("rows_read", "rows", "n_rows"):
            if isinstance(src.get(key), int):
                return src[key]
    return None


def engine_of(run):
    """Which ENGINE built this run, read off its own provenance.

    WHY THIS EXISTS. A whole night went into a "twelve-pair
    regression" that was two different engines: `run_deliverable`
    carries `settings.engine = "neural"` and `run_n9g` carries no
    engine field at all, because `pipeline.py` - the rules path -
    does not write one. Four control arms were then built to find
    the cause and every one came back innocent, which they had to:
    each switched off a RULES mechanism while the baseline was a
    NEURAL file. A five-run bisection was queued against a
    difference no commit causes.

    `neural` is positive evidence. Absent is reported as `rules`
    ONLY when the provenance looks like the rules writer's - it
    writes `privacy` and `note` at the top level - because absent
    on its own is not the same fact as rules, which is the
    distinction this project draws everywhere else.
    """
    f = Path(run) / "provenance.json"
    if not f.exists():
        return None
    try:
        p = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None
    eng = (p.get("settings") or {}).get("engine")
    if isinstance(eng, str) and eng.strip():
        return eng.strip()
    if "privacy" in p and "note" in p:
        return "rules"
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
    out = {"run": run.name, "met": res.get("met"),
           "fid": fid, "criteria": {}}
    for c in res["criteria"]:
        out["criteria"][c["name"]] = {
            "ok": c["ok"], "detail": c["detail"][:70]}
    return out


def pair_map(fid):
    """(child, parent) -> the pair record, for the related ones.

    The gate counts a pair when the SOURCE relates it, so that is
    the set to intersect: a pair one run never related cannot say
    anything about the other.
    """
    out = {}
    for x in ((fid.get("relationships") or {}).get("pairs") or []):
        if x.get("source") is None or x.get("generated") is None:
            continue
        try:
            if abs(float(x["source"])) < 0.1:
                continue
        except (TypeError, ValueError):
            continue
        out[(x.get("child"), x.get("parent"))] = x
    return out


def shared_close(fa, fb, tol=0.2):
    """`close` for both runs, on the pairs BOTH of them relate.

    WHY THIS IS THE ONLY COMPARABLE CLOSE. Each run's pair surface
    comes from its OWN blueprint's token vocabulary, so two runs
    with different blueprints are graded over different sets and
    no amount of re-measuring makes the denominators equal -
    measured tonight as 148 against 125 with the same code and
    the same source. On the INTERSECTION both numbers are over
    one set, and a difference there is the output differing.
    """
    a, b = pair_map(fa), pair_map(fb)
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    def ok(x):
        return abs(float(x["source"]) - float(x["generated"])) <= tol
    # AND WHICH PAIRS FLIPPED, because a count cannot answer a
    # question about WHICH - this project's own lesson, and the
    # reason a twelve-pair difference was guessed at four times
    # before anybody looked at the names.
    flips = []
    for k in keys:
        if ok(a[k]) and not ok(b[k]):
            flips.append((k, float(a[k]["source"]),
                          float(a[k]["generated"]),
                          float(b[k]["generated"])))
    return {"shared": len(keys),
            "a_close": sum(1 for k in keys if ok(a[k])),
            "b_close": sum(1 for k in keys if ok(b[k])),
            "lost": flips}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    src = pd.read_csv(a.src, dtype=str, keep_default_na=False)
    print("source rows {}".format(len(src)))
    print()
    _fids, _labels = {}, {}
    for r in a.runs:
        got = regate(r, src, len(src))
        if "why" in got:
            print("{:<22} SKIPPED: {}".format(got["run"],
                                              got["why"]))
            continue
        if got.get("fid") is not None:
            # KEYED BY PATH, not by directory NAME. Two runs can
            # perfectly well both be called `run_n9g` under
            # different parents, and keying by name would silently
            # collapse them into one and print nothing.
            _fids[str(Path(r).resolve())] = got.pop("fid")
            _labels[str(Path(r).resolve())] = got["run"]
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
    if len(_fids) == 2:
        _names = list(_fids)
        _sh = shared_close(_fids[_names[0]], _fids[_names[1]])
        # Look the paths up BY NAME rather than by position:
        # a skipped run leaves `_fids` and `a.runs` misaligned,
        # and an engine label attached to the wrong run is the
        # exact fault this guard exists to prevent.
        _bypath = dict((Path(r).name, r) for r in a.runs)
        _e0 = engine_of(_bypath.get(_names[0], _names[0]))
        _e1 = engine_of(_bypath.get(_names[1], _names[1]))
        print("=== `close` ON THE PAIRS BOTH RUNS RELATE ===")
        print("  engine: {} = {} | {} = {}".format(
            _names[0], _e0 or "(not recorded)",
            _names[1], _e1 or "(not recorded)"))
        if _e0 and _e1 and _e0 != _e1:
            # NOT A WARNING UNDER THE NUMBER. A cross-engine close
            # figure is not a weaker comparison, it is a different
            # question - and this project already published one as
            # a regression. The numbers are still printed, because
            # suppressing them would hide what was measured, but
            # the verdict above them says what they are.
            print("  THESE RUNS WERE BUILT BY DIFFERENT ENGINES, so "
                  "a difference below is NOT a regression and NOT")
            print("  attributable to any commit - it is one engine "
                  "against another. `interaction surfaces`,")
            print("  `shapes tracked` and `dynamics tracked` are "
                  "worse than incomparable across engines: the")
            print("  neural path never consults published surfaces, "
                  "so it fails that criterion by construction.")
        elif not (_e0 and _e1):
            print("  AT LEAST ONE RUN DOES NOT RECORD ITS ENGINE, so "
                  "whether this is a like-for-like comparison")
            print("  cannot be established from the artifacts. "
                  "Absent is not the same fact as matching.")
        if _sh is None:
            print("  the two runs share no related pair, so there "
                  "is nothing comparable to report")
        else:
            print("  shared related pairs      {}"
                  .format(_sh["shared"]))
            print("  {:<22} {} close".format(
                _labels[_names[0]][:22], _sh["a_close"]))
            print("  {:<22} {} close".format(
                _labels[_names[1]][:22], _sh["b_close"]))
            _lost = _sh.get("lost") or []
            if _lost:
                print("  pairs the FIRST holds and the second "
                      "loses ({}):".format(len(_lost)))
                for (c, pp), sv, av, bv in _lost[:20]:
                    print("    {:<46} src {:+.3f}  "
                          "{:+.3f} -> {:+.3f}".format(
                              "{} <- {}".format(str(c)[:22],
                                                str(pp)[:22]),
                              sv, av, bv))
                if len(_lost) > 20:
                    print("    ... and {} more".format(
                        len(_lost) - 20))
            print("  Each run's own percentage is over its OWN "
                  "blueprint vocabulary, so the")
            print("  denominators differ by construction. This is "
                  "the number that compares.")
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
