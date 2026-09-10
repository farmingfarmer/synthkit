"""Reproduce the two real-extract fidelity mechanisms on fixtures.

    python scripts/repro_mechanisms.py pressure [SEEDS...]
    python scripts/repro_mechanisms.py triangle [SEEDS...]
    python scripts/repro_mechanisms.py build OUTDIR   (CSVs only)

Two mechanisms, measured on the real extract 2026-09-10, that must
be reproduced here before any fix is written - a guard verified
only against a shape invented on this machine will pass while
failing on the shape that exists:

PRESSURE - the colinear family. Both seed-37 inversions and the
worst departed surfaces live in the cuff/invasive pressure columns,
where MAP = (S + 2D)/3 makes every column a near-linear function of
the others. The fixture plants that identity (S-D correlation aimed
at the measured +0.888), a variant chain (cuff, bmdi), and two weak
negatives shaped after the inverted pairs (span ~ MAP at the
measured -0.304, proc ~ D at the measured -0.126). Reported per
seed: inversions by name, surface survival, close.

TRIANGLE - the attribution flip. On the extract, span_days is
driven 74/26 by procedure_count over active_drug_count in the
source and 22/78 in the synthetic file; procedure_count itself
flips 77/23 to 7/93. The fixture plants a three-column cycle with
the measured 75/25 asymmetry and re-measures the SHAP driver
shares on both tables. Reported per seed: source shares, generated
shares, and the flip (whether the majority driver changed).

Every number this script prints is measured from the run it just
made; nothing is assumed.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEEDS = [20260731, 11, 37, 5, 7]
N_PATIENTS = 400
VISITS = 6


def _rng(seed):
    import numpy as np
    return np.random.RandomState(seed)


def build_pressure(seed=0):
    """The colinear pressure family, aimed at measured statistics."""
    import numpy as np
    import pandas as pd
    r = _rng(seed)
    n = N_PATIENTS * VISITS
    pid = np.repeat(np.arange(N_PATIENTS), VISITS)
    # patient-level baselines so persistence exists, like the extract
    d_base = r.normal(70, 8, N_PATIENTS)[pid]
    dia = d_base + r.normal(0, 4, n)
    # aim S-D at the measured +0.888: shared component dominates
    sys_ = 45 + 1.05 * dia + r.normal(0, 5.5, n)
    map_ = (sys_ + 2 * dia) / 3 + r.normal(0, 0.6, n)
    map_cuff = map_ + r.normal(0, 1.6, n)
    map_bmdi = map_cuff + r.normal(0, 1.6, n)
    # the weak negatives, shaped after the two inverted pairs
    z_map = (map_ - map_.mean()) / map_.std()
    span = np.clip(np.round(
        3.2 - 0.304 * 2.2 * z_map
        + r.normal(0, 2.0, n)), 0, None)
    z_dia = (dia - dia.mean()) / dia.std()
    proc = np.clip(np.round(
        2.5 - 0.126 * 1.6 * z_dia
        + r.normal(0, 1.5, n)), 0, None)
    return pd.DataFrame({
        "person_id": ["P{:04d}".format(i) for i in pid],
        "systolic": np.round(sys_, 1),
        "diastolic": np.round(dia, 1),
        "map_calc": np.round(map_, 1),
        "map_cuff": np.round(map_cuff, 1),
        "map_cuff_bmdi": np.round(map_bmdi, 1),
        "span_days": span.astype(int),
        "proc_count": proc.astype(int),
    })


def build_triangle(seed=0):
    """The attribution triangle: y1 driven ~75/25 by y2 over y3,
    all three mutually correlated so the discovered graph cycles."""
    import numpy as np
    import pandas as pd
    r = _rng(seed)
    n = N_PATIENTS * VISITS
    pid = np.repeat(np.arange(N_PATIENTS), VISITS)
    latent = r.normal(0, 1, N_PATIENTS)[pid] + r.normal(0, .4, n)
    y3 = 4 + 1.1 * latent + r.normal(0, 0.9, n)
    y2 = 6 + 1.3 * latent + 0.45 * (y3 - y3.mean()) \
        + r.normal(0, 1.0, n)
    y1 = 3 + 1.0 * (y2 - y2.mean()) + 0.75 * (y3 - y3.mean()) \
        + r.normal(0, 1.1, n)
    return pd.DataFrame({
        "person_id": ["P{:04d}".format(i) for i in pid],
        "span_like": np.round(y1, 2),
        "proc_like": np.round(y2, 2),
        "drug_like": np.round(y3, 2),
    })


def _fit(csv_path, out_dir, seed):
    cmd = [sys.executable, "-m", "synthkit.cli", "fit",
           "--src", str(csv_path), "--out", str(out_dir),
           "--group-by", "person_id", "--seed", str(seed),
           "--generate"]
    print("  fit --seed {} ...".format(seed), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str(ROOT))
    if p.returncode != 0:
        sys.exit("fit failed (seed {}):\n{}".format(
            seed, (p.stdout + p.stderr)[-2000:]))


def _shares(frame, child, parents):
    """SHAP driver shares (falls back to permutation importance if
    shap is absent, and says which it used)."""
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = np.column_stack([frame[p].to_numpy(dtype=float)
                         for p in parents])
    y = frame[child].to_numpy(dtype=float)
    m = HistGradientBoostingRegressor(
        max_iter=60, random_state=0).fit(X, y)
    try:
        import shap
        sv = np.abs(shap.TreeExplainer(m).shap_values(
            X[:1500])).mean(0)
        how = "shap"
    except Exception:
        from sklearn.inspection import permutation_importance
        sv = permutation_importance(
            m, X, y, n_repeats=5,
            random_state=0).importances_mean
        sv = np.clip(sv, 0, None)
        how = "permutation"
    tot = float(sv.sum()) or 1.0
    return [float(v) / tot for v in sv], how


def run_pressure(seeds):
    import pandas as pd
    print("PRESSURE - colinear family: inversions and surface "
          "survival, {} seeds".format(len(seeds)))
    df = build_pressure()
    # the fixture must contain the thing: verify aimed statistics
    r_sd = df["systolic"].corr(df["diastolic"])
    r_span = df["span_days"].corr(df["map_calc"])
    r_proc = df["proc_count"].corr(df["diastolic"])
    print("fixture: S-D r={:.3f} (aim .888) | span~map r={:.3f} "
          "(aim -.304) | proc~dia r={:.3f} (aim -.126)".format(
              r_sd, r_span, r_proc))
    rows = []
    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / "pressure.csv"
        df.to_csv(csv, index=False)
        for seed in seeds:
            out = Path(td) / "run{}".format(seed)
            _fit(csv, out, seed)
            fid = json.loads((out / "fidelity.json").read_text(
                encoding="utf-8"))
            s = fid["summary"]
            inv = fid["relationships"]["inverted"]
            rows.append({
                "seed": seed,
                "close": "{}/{}".format(s["pairs_close"],
                                        s["pairs"]),
                "inverted": len(inv),
                "inverted_pairs": [
                    "{}~{} {:+.2f}->{:+.2f}".format(
                        r_["child"], r_["parent"], r_["source"],
                        r_["generated"]) for r_ in inv],
                "surfaces": "{}/{}".format(
                    s.get("surfaces_ok"),
                    s.get("surfaces_compared")),
                "shapes": "{}/{}".format(
                    s.get("shapes_ok"), s.get("shapes_compared")),
            })
    print()
    for r_ in rows:
        print("  seed {:>9}: close {:>6}  shapes {:>4}  "
              "surfaces {:>4}  inverted {}".format(
                  r_["seed"], r_["close"], r_["shapes"],
                  r_["surfaces"], r_["inverted"]))
        for ip in r_["inverted_pairs"]:
            print("      INVERTED {}".format(ip))
    n_inv = sum(r_["inverted"] for r_ in rows)
    n_surf_bad = sum(
        1 for r_ in rows
        if r_["surfaces"].split("/")[1] not in ("0", "None")
        and r_["surfaces"].split("/")[0] != r_["surfaces"].split("/")[1])
    print()
    print("TOTAL: {} inversion(s) across {} seeds; surface "
          "criterion failed on {} seed(s). Reproduced = any "
          "inversion, or repeated surface failure.".format(
              n_inv, len(seeds), n_surf_bad))
    return rows


def run_triangle(seeds):
    import pandas as pd
    print("TRIANGLE - attribution flip: driver shares on both "
          "tables, {} seeds".format(len(seeds)))
    df = build_triangle()
    src_shares, how = _shares(df, "span_like",
                              ["proc_like", "drug_like"])
    print("fixture source attribution ({}): proc {:.0%} / drug "
          "{:.0%} (aim 75/25)".format(how, *src_shares))
    rows = []
    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / "triangle.csv"
        df.to_csv(csv, index=False)
        for seed in seeds:
            out = Path(td) / "run{}".format(seed)
            _fit(csv, out, seed)
            gen = pd.read_csv(out / "generated.csv")
            gs, _ = _shares(gen, "span_like",
                            ["proc_like", "drug_like"])
            flipped = (src_shares[0] > 0.5) != (gs[0] > 0.5)
            rows.append({"seed": seed, "gen": gs,
                         "flipped": flipped})
    print()
    for r_ in rows:
        print("  seed {:>9}: generated proc {:.0%} / drug {:.0%}"
              "  {}".format(r_["seed"], r_["gen"][0], r_["gen"][1],
                            "FLIPPED" if r_["flipped"] else
                            "majority kept"))
    n_flip = sum(1 for r_ in rows if r_["flipped"])
    print()
    print("TOTAL: majority driver flipped on {} of {} seeds. "
          "Reproduced = any flip, or a mean drift of the lead "
          "share beyond 15 points.".format(n_flip, len(seeds)))
    drift = sum(abs(r_["gen"][0] - src_shares[0])
                for r_ in rows) / len(rows)
    print("mean lead-share drift: {:.0f} points".format(
        100 * drift))
    return rows


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    what = sys.argv[1]
    if what == "build":
        if len(sys.argv) < 3:
            sys.exit("build needs OUTDIR")
        out = Path(sys.argv[2])
        out.mkdir(parents=True, exist_ok=True)
        build_pressure().to_csv(out / "pressure.csv", index=False)
        build_triangle().to_csv(out / "triangle.csv", index=False)
        print("wrote pressure.csv and triangle.csv to {}".format(
            out))
        return 0
    seeds = ([int(x) for x in sys.argv[2:]]
             or list(DEFAULT_SEEDS))
    if what == "pressure":
        run_pressure(seeds)
    elif what == "triangle":
        run_triangle(seeds)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
