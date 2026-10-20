"""The release-dial verification harness, on the IV-Push fixture.

The fixture reproduces the real extract's acute-visit stratum
(drugs and their route co-occurring on ~10% of visits at measured
shares) - the shape the cluster tier was proven on. The harness
sweeps the release-strength dial and one suppression, printing
requested-vs-achieved, because a dial a compliance reader cannot
verify on the page is a knob, not a control.

Lives in the repo, not a scratch folder: the OS tmp cleaner has
eaten proof fixtures twice, and a fixture that proved a mechanism
is part of the mechanism's record.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DRUGS = [("ondansetron", 0.092, 0.77),
         ("acetaminophen", 0.093, 0.60),
         ("metoclopramide", 0.035, 0.40),
         ("pantoprazole", 0.049, 0.30),
         ("acetHYDRO", 0.037, 0.25),
         ("diphenhydrAMINE", 0.082, 0.36)]


def build(seed):
    rs = np.random.RandomState(seed)
    n_pat, mv = 800, 58
    counts = rs.poisson(mv, n_pat) + 1
    pid = np.repeat(np.arange(n_pat), counts)
    n = len(pid)
    sev = rs.normal(0, 1, n_pat)[pid] + rs.normal(0, 0.6, n)
    acute = rs.rand(n) < 0.097
    IN = {"ondansetron": (0.75, 0.017),
          "acetaminophen": (0.62, 0.033),
          "metoclopramide": (0.26, 0.010),
          "pantoprazole": (0.25, 0.025),
          "acetHYDRO": (0.18, 0.020),
          "diphenhydrAMINE": (0.38, 0.045)}
    has = {}
    for name, _share, _c in DRUGS:
        q_in, bg = IN[name]
        has[name] = (acute & (rs.rand(n) < q_in)) \
            | (rs.rand(n) < bg)
    route = (acute & (rs.rand(n) < 0.93)) | (rs.rand(n) < 0.007)
    extra_d = {"filler_d%02d" % i: rs.rand(n)
               < rs.uniform(0.02, 0.06) for i in range(12)}
    extra_r = {"Oral": rs.rand(n) < 0.51, "IV": rs.rand(n) < 0.35,
               "Topical": rs.rand(n) < 0.22,
               "Subcut": rs.rand(n) < 0.18,
               "IV Piggyback": rs.rand(n) < 0.16,
               "Flush": rs.rand(n) < 0.20,
               "IntraMuscular": rs.rand(n) < 0.14,
               "Nasal": rs.rand(n) < 0.12}

    def join(row_sets):
        return [";".join(sorted(t for t, m in row_sets.items()
                                if m[i])) for i in range(n)]
    alld = dict(has)
    alld.update(extra_d)
    allr = dict(extra_r)
    allr["IV Push"] = route
    return pd.DataFrame({
        "person_id": ["P%04d" % i for i in pid],
        "active_drugs": join(alld),
        "drug_routes": join(allr),
        "heart_rate": np.round(80 + 6 * sev
                               + rs.normal(0, 9, n), 1),
        "span_days": np.round(np.exp(
            1.1 + 0.5 * sev + rs.normal(0, 0.4, n))),
    })


def tok(frame, col, t):
    return frame[col].fillna("").str.split(";").apply(
        lambda x, tt=t: tt in x).to_numpy()


def pair_corrs(df):
    out = {}
    hasr = tok(df, "drug_routes", "IV Push")
    for name, _s, _c in DRUGS:
        out[name] = float(np.corrcoef(
            tok(df, "active_drugs", name), hasr)[0, 1])
    return out


def drug_drug(df):
    names = [n for n, _s, _c in DRUGS]
    cols = {n: tok(df, "active_drugs", n) for n in names}
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            out[(names[i], names[j])] = float(np.corrcoef(
                cols[names[i]], cols[names[j]])[0, 1])
    return out


def null_floor(df, seed=7):
    """The combinatorial floor: fully re-deal the SOURCE's set
    columns (tokens pooled and re-dealt by row size) and permute
    its scalar columns, then measure the same pairs. What
    survives is what ANY sets of these sizes exhibit - size
    arithmetic, not institutional pattern - and the dial's honest
    zero is this floor, not 0.0."""
    rs = np.random.RandomState(seed)
    null = df.copy()
    for c in ("active_drugs", "drug_routes"):
        cells = null[c].fillna("").tolist()
        toks = [[t for t in v.split(";") if t] for v in cells]
        pool = [t for ts in toks for t in ts]
        rs.shuffle(pool)
        pos = 0
        out = []
        for ts in toks:
            n9 = len(ts)
            out.append(";".join(sorted(set(
                pool[pos:pos + n9]))))
            pos += n9
        null[c] = out
    for c in ("heart_rate", "span_days"):
        v = null[c].to_numpy(copy=True)
        rs.shuffle(v)
        null[c] = v
    return null


def sweep(seed=0, strengths=(1.0, 0.5, 0.25, 0.0)):
    """One fit, several strengths, one suppression. Returns
    {strength: mean_pair_ratio_vs_source} plus the suppressed
    pair's achieved correlation."""
    from latent_challenger import LatentGen
    df = build(0)
    src = pair_corrs(df)
    dd_src = drug_drug(df)
    nf = null_floor(df)
    nc = pair_corrs(nf)
    nd = drug_drug(nf)
    allp0 = [(nc[k], src[k]) for k in src] \
        + [(nd[k], dd_src[k]) for k in dd_src]
    floor = float(np.mean([a / b for a, b in allp0
                           if abs(b) > 0.15]))
    print("measured combinatorial floor: {:+.3f}x source - "
          "what ANY sets of these sizes exhibit; the dial's "
          "honest zero".format(floor))
    g = LatentGen(seed=seed).fit(df, "person_id")
    ratios = {}
    for s_ in strengths:
        g.release_strength = s_
        gen = g.generate(len(df), seed=1000)
        gc = pair_corrs(gen)
        dd_gen = drug_drug(gen)
        allp = [(gc[k], src[k]) for k in src] \
            + [(dd_gen[k], dd_src[k]) for k in dd_src]
        rr = [a / b for a, b in allp if abs(b) > 0.15]
        ratios[s_] = float(np.mean(rr))
        print("strength {:>4} requested: mean pair strength "
              "achieved {:+.3f}x source".format(s_, ratios[s_]))
    g.release_strength = 1.0
    g.suppress_pairs = [("heart_rate", "span_days")]
    gen = g.generate(len(df), seed=1000)
    g.suppress_pairs = []
    r_sup = float(np.corrcoef(
        pd.to_numeric(gen["heart_rate"], errors="coerce"),
        pd.to_numeric(gen["span_days"], errors="coerce"))[0, 1])
    r_src = float(np.corrcoef(
        pd.to_numeric(df["heart_rate"], errors="coerce"),
        pd.to_numeric(df["span_days"], errors="coerce"))[0, 1])
    print("suppressed heart_rate~span_days: source {:+.3f}, "
          "achieved {:+.3f}".format(r_src, r_sup))
    return ratios, r_sup, r_src, floor


if __name__ == "__main__":
    sweep()
