"""A competing generator, built to MEASURE, not (yet) to ship.

    python scripts/latent_challenger.py triangle [SEEDS...]
    python scripts/latent_challenger.py pressure [SEEDS...]
    python scripts/latent_challenger.py planted  [SEEDS...]

WHY THIS EXISTS. The blueprint path keeps direction on ~94% of
pairs and zero inversions on the runs that circulate - and the
complex interactions still mostly do not survive: published
surfaces read 2/6 on the real extract, and the SHAP attribution
view has shown the generated file rerouting who-drives-whom. The
question this instrument answers is whether that ceiling belongs
to OUR approach or to the problem: a generator with no published
contract at all - an autoencoder over the (k-screened) frame,
sampling its latent space - is the fidelity-at-any-cost ruler.

THE PRIVACY MODEL IS DIFFERENT IN KIND, AND SAYS SO. The blueprint
never touches records at generation; everything published is a
k-screened aggregate, which is why the membership attack reads
0.52 where a cheat reads 1.00. THIS generator trains ON RECORDS.
Folding sub-k levels and clipping numeric tails to the k-anonymous
bounds (both done here, before training) narrows what the network
can see; it does not prevent memorization, which happens at the
record level. Until the SAME attack battery - membership with its
positive control, attribute disclosure with its population
control - passes on this path, its output is a measurement of
headroom and never a release. The nearest-neighbor ratio printed
with every run is the first tripwire, not a certificate.

WHAT IT IS. Row-level: standardize numerics (clipped to the mean
of the k most extreme patients' extremes - the blueprint's own
rule), one-hot categoricals (sub-k levels folded to __other__),
missingness as indicator columns. A bottleneck MLP autoencoder
(sklearn, the data machine's stack - no torch), a Gaussian
mixture fitted over the training latents, sampling from the
mixture, hand-decoding through the trained weights. Row-level
means NO within-patient dynamics and no visit structure - stated
here because a spec that silently lost the dynamics is the same
failure as a column that silently lost its meaning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ONE source of truth for the posture sentence - the run prints
# it, and the governance contract carries it verbatim, because
# two wordings of a privacy posture is how pages disagree.
POSTURE = ("Row-level: no within-patient dynamics. Trains on "
           "records: a MEASUREMENT, not a release.")

# below this presence share a numeric column's decoded channel is
# starved (measured: sd ratio 0.01 at 0.002 presence) and its
# marginal rides a k-banded published grid instead
SPARSE_PRESENCE = 0.05
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthkit import dates, sets    # noqa: E402


def _read_source(path):
    """Read a source CSV the way the engine must see it.

    "" folds to NaN for ordinary columns - three spellings of one
    absence - and is KEPT for set columns, where an empty list is
    a fact about the visit. The blanket fold fed the engine a
    source in which the medication-free visits did not exist:
    token shares inflated by exactly 1/(1-empty) (5x on the 80%-
    empty procedures), the size grid lost its zeros so no empty
    set was ever drawn, and the gate PASSED both, because its
    reference numbers were folded the same way. The deck reads
    the raw file and disagreed - two numbers on one page, which
    is this project's alarm. The predicate is `sets.looks_like_
    set`, the same gate `vocabulary` applies, so the reader and
    the model cannot class a column differently."""
    _p = Path(path)
    if _p.is_dir():
        # a FOLDER, not a file: a terminal user hits the same
        # [Errno 13] the bench user hit - name the CSVs inside
        _csvs = sorted(f.name for f in _p.glob("*.csv"))[:6]
        _hint = (" - this folder holds: " + ", ".join(_csvs)
                 + ". For example: " + str(_p / _csvs[0])
                 if _csvs else " and it holds no .csv files")
        sys.exit("{} is a FOLDER, not a file{}".format(
            path, _hint))
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    setcols = [c for c in raw.columns
               if sets.looks_like_set(raw[c].replace("", np.nan))]
    df = raw.replace("", np.nan)
    for c in setcols:
        df[c] = raw[c]
    return df

K = 10
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
# THE SAME CAP THE BLUEPRINT USES. Without it a set column -
# whose "levels" are its distinct COMBINATION strings - puts
# thousands of one-hot dimensions into the encoded frame: on the
# real extract, 44 source columns became 4,326 encoded ones and
# training died allocating 1.5 GiB. Most-common first, the rest
# folded to __other__.
MAX_LEVELS = 60
# HOW MANY SET TOKENS GET THEIR OWN DIMENSION. A set column is a
# BAD CATEGORY and a GOOD SET OF INDICATORS - the rules engine
# recorded that months ago and this engine had not learned it.
# Capped for the same reason levels are: `conditions` holds 1,050
# tokens above a k=10 patient floor on the real extract, and one
# dimension each would put the encoded frame back where the
# 1.5 GiB allocation died. The cap is REPORTED, because a reader
# who is not told how many tokens were modeled cannot tell an
# absent one from an unexamined one.
MAX_TOKENS = 60
# HOW FAITHFULLY THE TOKEN DRAW FOLLOWS THE DECODER. The Gumbel
# top-k draw adds noise of fixed scale (~1.28 sd) to the LOG
# activations, and the decoder's activation differences are of
# the same order - so the noise drowned the conditional signal.
# A drug implies its route in the source at rho 0.63-0.73 and the
# generated file carried 0.24-0.34: the co-occurrence was IN the
# activations and the draw randomized it away. Multiplying the
# log-activations sharpens the draw; the per-token offset solve
# recalibrates the marginals at ANY sharpness, so token shares
# and the empty rate are unmoved by construction. Value chosen by
# the sweep in the module docstring's checks, with the
# memorization tripwire measured beside it - a sharper draw
# follows the decoder more faithfully, and the decoder was
# trained on records.
# Swept 2026-09-29 on the drug-implies-route fixture (source rho
# 0.56-0.73), three seeds per cell, under the INDEPENDENT
# per-token draw (each token takes its top share-of-rows by its
# own score; no within-row competition):
#     sharp 1   |cooc drop| 0.315  share 0.007  nn 1.48
#     sharp 2   |cooc drop| 0.179  share 0.013  nn 1.44
#     sharp 4   |cooc drop| 0.101  share 0.018  nn 1.28
#     sharp 8   |cooc drop| 0.071  share 0.020  nn 1.25
# Marginal shares are exact by construction here (each token picks
# round(p*n) rows), so sharpness costs only threshold rounding.
# The earlier COMPETITIVE draw (Gumbel top-k with an offset solve)
# was retired after a measured sign flip: any within-row
# competition mechanically ties every token to the SET SIZE, and
# corr(has_Oral, drug_count) read +0.04..+0.16 against a source
# -0.25 at every sharpness - the class behind the real extract's
# has_Oral <- active_drug_count 0.26 -> -0.64 INVERTED.
TOKEN_SHARPNESS = 4.0


def _k_clip(vals: np.ndarray, gid: np.ndarray, k: int):
    """The blueprint's bound rule, applied to the training frame:
    each numeric column is clipped to the MEAN of the k most
    extreme patients' own extremes, so no single person's min or
    max shapes what the network sees."""
    df = pd.DataFrame({"v": vals, "g": gid}).dropna()
    if df["g"].nunique() <= 2 * k:
        return float(np.nanmin(vals)), float(np.nanmax(vals))
    per = df.groupby("g")["v"]
    lo = float(per.min().nsmallest(k).mean())
    hi = float(per.max().nlargest(k).mean())
    return lo, hi


class _OneCloud:
    """A stand-in for a mixture when there are too few samples to
    fit one. `GaussianMixture` requires at least TWO rows however
    few components are asked for, so a table with a single entity
    - one patient centre - cannot use it at all. The distribution
    of one point is that point, plus the spread of everything
    else; this returns exactly that and says so by existing."""

    def __init__(self, pts, sd):
        import numpy as np
        self.pts = np.atleast_2d(pts)
        self.sd = sd

    def sample(self, n):
        import numpy as np
        r = np.random.RandomState(0)
        base = self.pts[r.randint(0, len(self.pts), n)]
        return base + r.normal(0, self.sd, base.shape), None


def _mixture(X, n_components, seed, sd_fallback):
    """A mixture where one can be fitted, and an honest stand-in
    where it cannot."""
    import numpy as np
    from sklearn.mixture import GaussianMixture
    X = np.atleast_2d(X)
    k = max(1, min(int(n_components), len(X)))
    if len(X) < 2:
        return _OneCloud(X, sd_fallback)
    return GaussianMixture(n_components=k,
                           covariance_type="full",
                           reg_covar=1e-4,
                           random_state=seed).fit(X)


class LatentGen:
    def __init__(self, k: int = K, bottleneck: int = None,
                 hidden: int = None, seed: int = 0,
                 k_blur: bool = True, support_ref: int = 12000,
                 support_neighbors: int = 40,
                 denoise: float = 0.3,
                 hierarchical: bool = True,
                 dev_scale: float = 1.0,
                 release_strength: float = 1.0,
                 suppress_pairs: list = None,
                 suppress_top: int = 0,
                 poison_pairs: list = None):
        # defaults so the caps exist before fit() chooses them
        self._cap_tok = MAX_TOKENS
        self._cap_lv = MAX_LEVELS
        # Capacity scales with the frame, decided at fit time. An
        # 8-dim bottleneck tuned on 6-column fixtures LOST a named
        # driver triple entirely at 77 columns (shares read 0/0,
        # close fell to 61%) - a ruler too short for the thing it
        # measures reads as a finding about the thing.
        self.k, self.b, self.h, self.seed = k, bottleneck, hidden, seed
        # THE RELEASE DIALS. Compliance's concern is not privacy
        # (certified) but VALUE EXTRACTION - a vendor training on
        # the patterns a full-fidelity release carries. strength
        # scales every learned relationship toward independence
        # while marginals stay byte-exact (a calibrated fraction
        # of each column is PERMUTED - rearrangement only, the
        # sweeps' own safety property); suppress_pairs names
        # relationships that must not exist at all. Arithmetic
        # identities are recomputed after and therefore EXEMPT -
        # a file where count != len(list) reads as broken, not
        # private - and the contract says so.
        self.release_strength = float(release_strength)
        self.suppress_pairs = list(suppress_pairs or [])
        # AUTO-SUPPRESS THE TOP-K STRONGEST RELATIONSHIPS: the
        # "minimum shareable" control. The global dial (release_
        # strength) weakens the textbook relationships every
        # clinical dataset has exactly as much as the distinctive
        # ones compliance fears a vendor extracting; this instead
        # finds the K strongest institution-specific pairs and
        # removes exactly those, keeping everything else faithful.
        # Compliance cannot name patterns it has never seen, so
        # the tool names them - by strength - and says which it
        # removed.
        self.suppress_top = int(suppress_top or 0)
        self._auto_suppressed = []
        # POISON: force a NAMED relationship to a user-chosen
        # (possibly FALSE) strength. Suppression removes a
        # pattern; poison replaces it with a deliberate lie the
        # user picks - invert a correlation, or plant one that is
        # not real - so a vendor training on the file learns the
        # wrong thing. It stays a valid TEST because we record
        # the true pattern AND the planted one: we hold the
        # answer key, they take a rigged exam only we can grade.
        self.poison_pairs = list(poison_pairs or [])
        self._poison_record = []
        # K-AWARE RECONSTRUCTION. Faithful where many patients
        # stand behind the pattern; deliberately BLURRED where
        # fewer than k do. Screening the INPUTS (sub-k levels
        # folded, tails clipped) bounds what the network is
        # shown; it does nothing about a rare COMBINATION of
        # ordinary values, which is the shape an autoencoder
        # memorizes and the weights-leak adversary reads back.
        self.k_blur = k_blur
        self.support_ref = support_ref
        self.support_neighbors = support_neighbors
        self.blur_report = None
        # DENOISING TRAINING - the defense on the surface that is
        # actually exposed. The k-aware blur above governs what
        # is SAMPLED; the membership adversary that fails at
        # width scores records straight through the WEIGHTS and
        # never reads a generated row, so no sampling rule can
        # move it. Training the network to rebuild a clean record
        # from a corrupted one denies it the exact-record
        # memorization that adversary reads back. Measured, not
        # assumed: see the attack numbers in CONVENTIONS.
        # DEFAULT 0.3, MEASURED. At 77-column width, seed 0:
        # denoise 0.0 -> weights-surface adversary 0.776 FAIL,
        # close 1332/1376, 4 inverted; 0.3 -> 0.569 PASS, close
        # 1354/1376, 3 inverted; 0.6 -> 0.563 PASS, close
        # 1352/1376. Better on BOTH axes at 0.3 - noise during
        # training regularizes, so the network learns the
        # structure instead of the records, which is both the
        # privacy fix and a fidelity gain.
        self.denoise = denoise
        # PATIENT STRUCTURE. Without this the engine returns a
        # table of UNLINKED VISITS: no person_id, so on a source
        # averaging 76 visits per patient nothing about a
        # patient's own history survives - not degraded, absent.
        # The latent code of every row is split into the
        # PATIENT's centre and that visit's DEVIATION from it;
        # generation draws a synthetic patient centre once, then
        # draws that patient's visits around it. Every visit of a
        # generated patient therefore shares a "who", which is
        # what makes within-patient behaviour possible at all.
        self.hierarchical = hierarchical
        # How much of the per-visit deviation to keep. SWEPT on
        # the 77-column fixture, and 1.0 - the measured
        # decomposition, untouched - won on both axes: mean
        # absolute between-patient-share error over 41 columns
        # read 0.079 at 1.0, then 0.108, 0.166 and 0.298 at 0.8,
        # 0.6 and 0.4, with `close` falling too. Shrinking a
        # patient's visits toward their own centre sounded like
        # it should raise the between-share and it does the
        # opposite once the decoder's nonlinearity is in the
        # path. The knob stays at 1.0 and exists so the next
        # person can re-run the sweep rather than re-reason it.
        self.dev_scale = dev_scale

    def fit(self, df: pd.DataFrame, group_by: str):
        from sklearn.mixture import GaussianMixture
        from sklearn.neural_network import MLPRegressor
        self._set_inds = {}
        # `sets.vocabulary` positions its group array by the
        # frame's own index labels, so a frame that arrived
        # filtered would read somebody else's patient ids.
        df = df.reset_index(drop=True)
        gid = df[group_by].astype(str).to_numpy()
        self._group_name = group_by
        cols = [c for c in df.columns if c != group_by]
        # THE WIDTH BUDGET WAS BUILT, MEASURED, AND NOT SHIPPED.
        # The cap being a memory budget rather than a privacy rule is
        # right, and lifting it is a measured WIN: paired on three
        # seeds at a flat 400, `close` moved +3.1 / +6.2 / +3.1 with
        # INVERTED still 0, direction 32/32, and the output vocabulary
        # went 60 tokens -> 400 with set sizes closing on the source
        # (active_drugs 1.40 -> 1.61 per row against 2.25 held).
        #
        # A SOLVED budget was then tried so a wider frame could not
        # recreate the 4,326-dimension allocation that once died. It
        # chose 280 and measured NOTHING: +0.0 / +3.1 / -3.1, mean
        # 0.0, signs mixed, against the flat 400's all-positive. The
        # reason is a modelling error, not a tuning one - predicting
        # width from k-screened levels on the FULL frame overestimates
        # what the encoder actually builds by about 2.5x at every cap
        # (973 against a measured 437 at cap 60; 3,222 against 1,308
        # at cap 400), because the fit sees an 85% patient split and
        # several counted columns never become one-hot at all. A
        # budget spent in the wrong units buys the wrong cap.
        #
        # So the DEFAULT is unchanged and the lever is explicit:
        # --max-tokens is measured good at 400 on an extract-shaped
        # frame. Shipping the solver would be shipping a default whose
        # own measurement says it does nothing. The fix, when this is
        # picked up: take true per-column width from the PLAN the
        # encoder already builds rather than predicting it, then solve
        # and trim - exact, in the units that bind.
        self._cap_tok, self._cap_lv = MAX_TOKENS, MAX_LEVELS
        self.plan = []
        # columns present on NO source row - carried so the written
        # file can say what the source said
        self.empty_cols = []
        mats = []
        for c in cols:
            v = pd.to_numeric(df[c], errors="coerce")
            # A column whose values are NUMBERS is numeric however
            # few of them there are - span_days holds seven
            # distinct day-counts, fell into the categorical
            # branch, and argmax decoding collapsed it to a
            # constant. The wrong-type fault, in this repo's own
            # ruler, found the same way as always: by reading the
            # output.
            # A COLUMN WHOSE PRESENT VALUES ARE NUMBERS IS
            # NUMERIC, however OFTEN it is present and however
            # FEW distinct values it has. Both halves of that
            # were learned the hard way and both by reading
            # output: a seven-value integer column fell into the
            # categorical branch and collapsed to a constant,
            # and then `height` - numeric, 4,155 distinct values,
            # present on 46.6% of rows - failed a flat
            # presence >= 0.5 gate, went down the same branch,
            # folded every value to __other__ below the k floor
            # and came out 100% EMPTY. Sparsity is a fact about
            # a column, not a reason to change its type.
            raw_notna = df[c].notna().mean()
            # A COLUMN PRESENT ON NO ROW IS EMPTY, WHICH IS A FACT
            # ABOUT THE DATA - NOT A TYPE TO GUESS AT.
            #
            # With no present value there is nothing to classify, so
            # the column fell through to the CATEGORICAL branch where
            # astype(str) turned NaN into the literal level "nan".
            # "nan" is a non-empty string, so the fidelity comparison
            # counted the column present on every row: a real
            # 100-patient cohort failed coverage 34/42 on eight
            # columns that were absent in its source, among them its
            # entire date spine (visit_start_date, visit_end_date,
            # span_days, age_at_visit) and four demographics. Third
            # trigger of the wrong-type-branch family this file
            # records - span_days collapsing at low cardinality,
            # height folding to __other__ at low presence - and this
            # one is NO PRESENT VALUES AT ALL.
            #
            # Kept OUT of the plan rather than given a kind: it
            # contributes no encoded dimension, no decoded channel
            # and no draw, because there is nothing to learn. The
            # column is written back EMPTY at assembly, so the
            # output says what the source said. Same
            # empty-versus-missing distinction this file already
            # draws for set columns and for id_column.
            if raw_notna == 0.0:
                self.empty_cols.append(c)
                continue
            is_num = (raw_notna > 0 and v.notna().mean()
                      >= 0.9 * raw_notna and v.nunique() >= 3)
            # A DATE IS A NUMBER WRITTEN AS TEXT, and reading it
            # as a CATEGORY destroys it. `pd.to_numeric` returns
            # NaN for `2013-10-03`, so every date column fell
            # into the categorical branch, kept its 60 most
            # common days as levels and folded 4,632 others into
            # `__other__` - which is not a date, so the deck read
            # it back as missing. Measured on the real extract:
            # `visit_start_date` went from 4,692 distinct values
            # and 0% missing to 60 and 79.9%. The rules engine
            # has parsed dates since the beginning; this engine
            # never did, which is the two-halves-disagree fault
            # this repo keeps rediscovering. Dates go through the
            # NUMERIC path as days since an epoch and are written
            # back in the format the column was read under.
            # A SET COLUMN IS A BAD CATEGORY. Its "levels" are
            # its distinct COMBINATION strings - 2,000+ of them on
            # the real extract - so a 60-level cap sent nearly
            # every row to `__other__`, which is a string that is
            # NOT EMPTY and contains NO REAL TOKEN. Measured on the
            # real extract: `procedures` is empty on 80.7% of
            # source rows and came out empty on 0.0% of generated
            # ones, `active_drugs` 31.4% against 0.0%, and
            # `ondansetron` went from 9.2% of rows to 0.0%. The
            # column looked present and was entirely scaffolding.
            # Encoded as the publishable SIZE plus one indicator
            # per published token, which is what the rules engine
            # has done since `sets.py` was written - and both
            # halves call that one module, so they cannot come to
            # disagree about which tokens clear the floor.
            if not is_num:
                vocab = sets.vocabulary(df[c], gid, k=self.k,
                                        cap=self._cap_tok)
                if vocab:
                    sep = vocab["separator"]
                    toks = [t["value"] for t in vocab["tokens"]]
                    tgt = [float(t["p"]) for t in vocab["tokens"]]
                    szs = sets.publishable_sizes(df[c], sep, toks)
                    smu = float(szs.mean())
                    ssd = float(szs.std()) or 1.0
                    ind = np.column_stack(
                        [sets.has_token(df[c], sep, t)
                         .fillna(0.0).to_numpy() for t in toks])
                    # THE SIZE MARGINAL, AS A QUANTILE GRID.
                    # A POINT MASS AT ZERO IS NOT A SHAPE A
                    # CURVE CAN MAKE - this file records that
                    # for the rules engine's zero-inflated
                    # counts, and a set size is the same object:
                    # 36% of these rows are exactly empty and a
                    # standardized decode plus rounding smears
                    # that mass, measured at 29.4% emitted
                    # against 36.1% in source while the mean
                    # size ran SHORT. Mapping the decoder's own
                    # ordering through the published quantiles
                    # reproduces the mass exactly and keeps
                    # whatever ordering the network learned.
                    grid = np.quantile(
                        szs.dropna().to_numpy(dtype=float),
                        np.linspace(0.0, 1.0, 1001))
                    print("  {}: {} tokens found, {} clear k, "
                          "{} modeled; mean set size {:.2f} "
                          "publishable against {:.2f} held"
                          .format(c, vocab["tokens_found"],
                                  vocab["tokens_above_k"],
                                  len(toks),
                                  vocab["mean_set_size"],
                                  vocab["mean_set_size_source"]))
                    self.plan.append(
                        ("set", c, toks,
                         (sep, smu, ssd, float(szs.max()), tgt,
                          int(vocab["tokens_above_k"]),
                          [float(x) for x in grid])))
                    # kept for cross-column implication
                    # detection below - one indicator matrix per
                    # set column, published tokens only
                    self._set_inds[c] = (toks, ind)
                    mats.append(np.column_stack(
                        [((szs - smu) / ssd).fillna(0.0)
                         .to_numpy()] + [ind]))
                    continue
            dspec = None
            if not is_num:
                dspec = dates.date_kind(df[c])
                if dspec:
                    v = dates.to_ordinal(df[c],
                                         dspec["parsed_format"])
                    is_num = bool(v.notna().any()
                                  and v.nunique() >= 3)
                    if not is_num:
                        dspec = None
            if is_num:
                lo, hi = _k_clip(v.to_numpy(dtype=float), gid,
                                 self.k)
                x = v.clip(lo, hi)
                mu, sd = float(x.mean()), float(x.std()) or 1.0
                filled = ((x - mu) / sd).fillna(0.0)
                miss = v.isna().astype(float)
                # TWO PROPERTIES THE DECODER CANNOT INVENT, so
                # they are carried: what share of the column is
                # ABSENT, and whether its values are WHOLE
                # NUMBERS. Both were found by the
                # distinguishability check, which separated the
                # tables perfectly on them - a column 47%
                # present in reality came out 100% empty because
                # a reconstructed missingness indicator was cut
                # at a flat 0.5, and integer ids came out as
                # floats.
                nn = v.dropna()
                integral = bool(len(nn)) and bool(
                    (nn == nn.round()).all())
                # HOW PRECISELY THE COLUMN IS WRITTEN. The
                # decoder emits full float precision, so a
                # column recorded to 4 decimal places came out
                # with 17 - a fingerprint that separates the two
                # tables perfectly and looks obviously wrong to
                # anyone opening the file. Read off the SOURCE
                # STRINGS, because that is where precision
                # actually lives; a float cannot tell you it was
                # written as 1.2474.
                dec = 0
                if not integral:
                    ss_ = df[c].dropna().astype(str)
                    ss_ = ss_[~ss_.str.contains("e|E", na=False)]
                    if len(ss_):
                        dd = (ss_.str.split(".").str[1]
                              .fillna("").str.len())
                        dec = int(min(int(dd.max()), 8))
                # A COLUMN PRESENT ON A SLIVER OF ROWS GIVES
                # THE DECODER ALMOST NOTHING TO LEARN, and its
                # decoded channel collapses to the mean -
                # measured: a 0.2%-present column generated at
                # sd 0.09 against a source 10.9, and the
                # conditional correlation computed on that
                # near-constant channel is RANK-OF-NOISE, which
                # is where the surviving real-extract inversion
                # lives. For sparse numerics the marginal is
                # published as a K-BANDED grid - sorted present
                # values walked into bands until each holds at
                # least k DISTINCT PATIENTS (k counts patients,
                # never rows), each grid point the band MEAN, so
                # no published number is an individual's value -
                # and generation rank-maps the decoded channel
                # through it, keeping whatever ordering the
                # network learned.
                qgrid = None
                pres_share = float(v.notna().mean())
                if 0 < pres_share < SPARSE_PRESENCE:
                    pv = pd.DataFrame({
                        "v": x[v.notna()],
                        "g": gid[v.notna()]}).sort_values("v")
                    bands, cur, seen = [], [], set()
                    for vv, gg in zip(pv["v"], pv["g"]):
                        cur.append(vv)
                        seen.add(gg)
                        if len(seen) >= self.k:
                            bands.append(float(np.mean(cur)))
                            cur, seen = [], set()
                    if cur and bands:
                        bands[-1] = float(np.mean(
                            [bands[-1]] + cur))
                    if len(bands) >= 2:
                        qgrid = bands
                self.plan.append(("num", c, mu, sd,
                                  (lo, hi, float(miss.mean()),
                                   integral, dec, dspec, qgrid)))
                mats.append(np.column_stack(
                    [filled.to_numpy(), miss.to_numpy()]))
            else:
                s = df[c].astype(str)
                per = pd.DataFrame({"l": s, "g": gid}) \
                    .groupby("l")["g"].nunique()
                # A LEVEL THE K RULE CANNOT PUBLISH IS NOT A
                # REASON TO PUBLISH NOTHING. The rules engine
                # learned this already: codes, SKUs, postcodes
                # and free text all have levels held by fewer
                # than k people, and folding every one of them
                # into `__other__` returns a CONSTANT column -
                # measured on the shape sweep, which collapsed a
                # high-cardinality code, a free-text note and a
                # date. What CAN be published is the SHAPE: how
                # many distinct values, and the profile of their
                # frequencies. The labels are then INVENTED.
                ok = per[per >= self.k].index
                # k first, then FREQUENCY - a column with
                # thousands of qualifying levels still gets only
                # the common ones, and the tail is honest rather
                # than enormous.
                keep = list(s[s.isin(ok)].value_counts()
                            .index[:self._cap_lv])
                shape_only = None
                if not keep:
                    # Nothing clears the floor: publish the shape
                    # and invent the labels, rather than emitting
                    # one value for every row.
                    vc = s.value_counts()
                    n_distinct = int(len(vc))
                    if n_distinct == 0:
                        # A COLUMN WITH NO VALUES AT ALL - every
                        # cell blank, which a de-identified OMOP
                        # export produces when a field was
                        # stripped. There is no shape to publish
                        # and nothing to draw from; emit the
                        # empty string for every row rather than
                        # crashing on choice([]). Found on the
                        # first REAL external cohort, the
                        # shape-sweep class in the wild.
                        keep = [""]
                        shape_only = {"distinct": 0,
                                      "profile": []}
                        lv = pd.Series([""] * len(s),
                                       index=s.index)
                    else:
                        prof = (vc.head(self._cap_lv) / float(len(s))
                                ).tolist()
                        keep = ["{}_{:04d}".format(c[:12], i)
                                for i in range(min(
                                    n_distinct, self._cap_lv))]
                        shape_only = {"distinct": n_distinct,
                                      "profile": prof}
                        lv = pd.Series(
                            np.random.RandomState(
                                self.seed).choice(
                                keep, len(s),
                                p=np.array(prof) / sum(prof)),
                            index=s.index)
                else:
                    lv = s.where(s.isin(keep), "__other__")
                levels = sorted(lv.unique())
                self.plan.append(("cat", c, levels, None,
                                  shape_only))
                mats.append(np.column_stack(
                    [(lv == l).astype(float).to_numpy()
                     for l in levels]))
        # A DECLARED SIZE IDENTITY IS ENFORCED BY COPYING, NOT
        # BY DRAWING TWICE - and this engine declares nothing, so
        # it has to FIND them.
        #
        # `active_drug_count` IS the length of `active_drugs` on
        # every source row: one causal direction, correlation
        # 1.000. Decoded as an ordinary number it becomes a
        # SECOND, independent draw of that same quantity, and two
        # independent draws of one marginal agree only by chance.
        # Measured on a fixture where the source identity holds
        # on 100% of rows, the generated file held it on 37.2%.
        #
        # It is the whole reason the real extract's gate read 10
        # INVERTED: every one of those pairs is a set token
        # indicator against its own count partner, and once the
        # partner stops being the size, a source correlation of
        # +0.4 lands at -0.14. The token indicators track the
        # set's ACTUAL size correctly (+0.277 against a source
        # +0.260) - it is the partner that drifted away from
        # them. This repository already recorded exactly this,
        # for the rules engine, and the fix is the same:
        # COPYING enforces an equality, a second draw only
        # exchanges the mismatch.
        self._size_identities = []
        for it in self.plan:
            if it[0] != "set":
                continue
            sep_ = it[3][0]
            true_n = sets.sizes_of(df[it[1]], sep_)
            for jt in self.plan:
                if jt[0] != "num" or jt[1] == it[1]:
                    continue
                cand = pd.to_numeric(df[jt[1]], errors="coerce")
                both = true_n.notna() & cand.notna()
                if int(both.sum()) < 50:
                    continue
                if float((true_n[both] == cand[both]).mean()) \
                        >= 0.99:
                    self._size_identities.append(
                        (jt[1], it[1], sep_))
                    print("  {} == len({}) on the source: the "
                          "generated column will be COPIED from "
                          "the set, not drawn again"
                          .format(jt[1], it[1]))
        # A LINEAR IDENTITY IS THE SIZE IDENTITY ONE LEVEL UP.
        # mean_arterial_pressure IS (systolic + 2*diastolic)/3,
        # and age_at_visit IS the visit year minus year_of_birth
        # - arithmetic, not correlation. Decoded as ordinary
        # numbers they become independent draws that satisfy the
        # arithmetic only by chance: measured on a MAP fixture,
        # the identity held within 0.1 on 100% of source rows and
        # 40.2% of generated ones. Someone opening the file finds
        # patients whose numbers contradict each other, and the
        # pressure-family interaction surfaces sit exactly on
        # this. Detected from the SOURCE (this engine declares
        # nothing): child ~ w.parents + b by least squares over
        # single parents and pairs, kept only when the residual
        # sd is under 2% of the child's own spread - a rule the
        # source itself breaks is not a rule. Enforced by
        # COMPUTING the child from its generated parents, the
        # same copy-not-redraw rule as the size identity.
        self._linear_identities = []
        num_items = [(it[1],
                      pd.to_numeric(df[it[1]], errors="coerce")
                      if it[4][5] is None else
                      dates.to_ordinal(df[it[1]],
                                       it[4][5]["parsed_format"]))
                     for it in self.plan if it[0] == "num"]
        # detection runs on a bounded sample - it is a property
        # of the columns, not of every row
        _det = df.index if len(df) <= 4000 else \
            np.random.RandomState(self.seed).choice(
                df.index, 4000, replace=False)
        num_s = [(c, v.loc[_det]) for c, v in num_items]
        _claimed = set()
        for ci, (child, cv) in enumerate(num_s):
            csd = float(cv.std()) or 0.0
            if csd <= 0 or child in _claimed:
                continue
            best = None
            for pi, (pa, pv) in enumerate(num_s):
                if pi == ci or pa in _claimed:
                    continue
                for qi in range(pi + 1, len(num_s)):
                    if qi == ci:
                        continue
                    pb, qv = num_s[qi]
                    if pb in _claimed:
                        continue
                    d = pd.DataFrame({"c": cv, "a": pv,
                                      "b": qv}).dropna()
                    if len(d) < 200:
                        continue
                    A = np.column_stack(
                        [d["a"], d["b"], np.ones(len(d))])
                    try:
                        w, *_ = np.linalg.lstsq(
                            A, d["c"].to_numpy(), rcond=None)
                    except np.linalg.LinAlgError:
                        continue
                    resid = d["c"].to_numpy() - A @ w
                    rs = float(resid.std())
                    if rs < 0.02 * csd and (
                            best is None or rs < best[0]):
                        best = (rs, pa, pb,
                                [float(x) for x in w])
            if best is not None:
                _, pa, pb, w = best
                self._linear_identities.append(
                    (child, pa, pb, w))
                # a child computed from parents must not also
                # BE a parent someone else is computed from -
                # first claim wins, deterministic by column
                # order
                _claimed.add(child)
                print("  {} == {:.4g}*{} + {:.4g}*{} + {:.4g} "
                      "on the source: computed from its "
                      "generated parents, not drawn again"
                      .format(child, w[0], pa, w[1], pb, w[2]))
        # A NEAR-IDENTITY IS AN IDENTITY ONE NOTCH RELAXED, and
        # the network cannot carry it. mean_arterial_pressure_
        # cuff_bmdi IS map_cuff measured by a second device -
        # source rho 0.975, residual ~3% of spread - and the
        # decoded twin FADES to 0.70-0.87, seed-dependent. The
        # sampler was exonerated by measurement: GMM at x1/x3/x6
        # components and a KDE sampler at three bandwidths all
        # land in the same band, so the attenuation is in the
        # network's handling of the sparse channel, not the
        # density estimate. Same cure as the exact tier, one
        # notch relaxed: the SPARSER twin is computed from its
        # denser sibling plus noise at the measured residual sd
        # - so the pair lands at its source correlation by
        # construction and the twin's own spread is preserved.
        # Detected only when rho >= 0.95 on >= 200 both-present
        # rows; enforced only where both are present in the
        # generated frame.
        # A DRUG IMPLIES ITS ROUTE, AND INDEPENDENT PER-COLUMN
        # DRAWS CANNOT SAY SO. The real extract's remaining close
        # misses cluster on CROSS-set-column token pairs -
        # has_sodium-chloride <- has_Flush at 0.88 source, 0.58
        # generated - and the relationship is near-FUNCTIONAL:
        # the route column is essentially derived from the drug
        # column. Each set column's tokens follow their own
        # activations, so cross-column coherence flows only
        # through the latent and arrives at a fraction of its
        # strength. Detected from the source: for tokens a in
        # column A and b in column B, keep a -> b when
        # P(b | a) >= 0.9 on at least 3k rows holding a, and b is
        # not near-universal anyway. Generation FORCES b where
        # any implying a was drawn, then restores b's quota and
        # B's empty count, so the marginals stay published while
        # the implication holds by construction.
        # CONDITIONAL QUOTAS, NOT ONLY HARD IMPLICATIONS. The
        # P(b|a) >= 0.9 tier cleared the near-deterministic pairs
        # (sodium-chloride <-> Flush left the drift list) and the
        # extract's remaining fades are MODERATE conditionals on
        # RARE tokens - docusate <-> Rectal at 0.4 -> 0.08,
        # tacrolimus <-> NG-tube at 0.39 -> 0.1 - reproduced at
        # width only when the paired tokens sit at 1.5-6% share:
        # a rare token's thin activation cannot place it among
        # its partner's rows by itself. Hard-forcing a q = 0.6
        # conditional would overshoot it to 1.0, so each adopted
        # token gets a QUOTA SPLIT instead: round(q * |trigger|)
        # of its picks inside its partner's rows, the rest
        # outside - any q lands by construction, and q ~ 1 is
        # just the old forcing. One conditioner per token (its
        # best by excess over base), the conditioner always from
        # an EARLIER plan column so triggers read final state,
        # and a broad token (base > 0.5) keeps its own placement
        # - the Oral lesson, retained.
        self._token_implications = []
        _setcols = [it[1] for it in self.plan if it[0] == "set"]
        for bi, cb in enumerate(_setcols):
            toks_b, ind_b = self._set_inds[cb]
            pb_all = ind_b.mean(axis=0)
            # SEVERAL CONDITIONERS PER TOKEN, because one
            # route is preferred by several drugs: with a single
            # best conditioner, IV Push conditioned on
            # ondansetron alone leaves its acetaminophen and
            # metoclopramide pairs exactly as faded as before -
            # measured: the single-conditioner cut landed only
            # the pairs whose token happened to pick THEM.
            cands = {}
            # the SAME column is a conditioner source too: the
            # extract's acute cluster (ondansetron,
            # acetaminophen, metoclopramide... on the same
            # visits) dissolves under the independent per-token
            # draw - measured src +0.10..+0.47 against gen
            # +0.35 down to -0.10 - and with it the
            # cross-column route quotas become JOINTLY
            # UNSATISFIABLE: the budget shortage was never the
            # allocator, it was the triggers coming apart.
            for ca in _setcols[:bi + 1]:
                toks_a, ind_a = self._set_inds[ca]
                na = ind_a.sum(axis=0)
                co = ind_a.T @ ind_b        # count(a & b)
                for j, b_ in enumerate(toks_b):
                    if pb_all[j] > 0.5:
                        continue
                    for i2 in range(len(toks_a)):
                        if ca == cb and toks_a[i2] == b_:
                            continue
                        if na[i2] < 3 * self.k:
                            continue
                        q = co[i2, j] / max(na[i2], 1.0)
                        gain = q - pb_all[j]
                        if gain < 0.15 or q < 1.5 * pb_all[j]:
                            continue
                        cands.setdefault(b_, []).append(
                            (float(gain), ca, toks_a[i2],
                             float(q)))
            for b_, lst in cands.items():
                lst.sort(reverse=True)
                # ALL qualifying conditioners, not a top-N: on
                # the real extract one route is implied by a
                # dozen drugs, and a partner outside a cap gets
                # no allocation while the rest-fill excludes its
                # rows - the budget caps the total on its own.
                self._token_implications.append(
                    (cb, b_, [(ca, a_, q) for _g, ca, a_, q
                              in lst]))
        if self._token_implications:
            print("  {} conditional token quota(s) detected "
                  "(P(b|a) exceeding base by >= 0.15): enforced "
                  "at generation, marginals exact by "
                  "construction".format(
                      len(self._token_implications)))
        # JOINT CLUSTER PLACEMENT. Pairwise same-column quotas
        # chase each other's base placements - each token
        # reallocates against a snapshot of the others, the
        # mutual overlap converges partway, and the honest
        # (fresh-credit) allocator leaves the cluster pairs
        # FADED where the stale one manufactured them. The
        # cluster is placed as ONE OBJECT instead: the
        # same-column implication pairs are the edges, the
        # connected components are the clusters, and the
        # published aggregates are the share of rows holding
        # >= 2 members (co-presence, the preserved thing
        # itself) and each member's in-cluster mass - all
        # k-screened by distinct patients.
        self._token_clusters = []
        self._rejected_clusters = []
        _samecol_edges = {}
        for cb2, b2, conds2 in self._token_implications:
            for ca2, a2, _q2 in conds2:
                if ca2 == cb2:
                    _samecol_edges.setdefault(cb2, []).append(
                        (a2, b2))
        for cb2, edges in _samecol_edges.items():
            toks2, ind2 = self._set_inds[cb2]
            parent = {}
            def _find(x):
                while parent.get(x, x) != x:
                    parent[x] = parent.get(parent[x], parent[x])
                    x = parent[x]
                return x
            for a2, b2 in edges:
                parent.setdefault(a2, a2)
                parent.setdefault(b2, b2)
                ra, rb = _find(a2), _find(b2)
                if ra != rb:
                    parent[ra] = rb
            comps = {}
            for t2 in parent:
                comps.setdefault(_find(t2), []).append(t2)
            for members in comps.values():
                if len(members) < 2:
                    continue
                mi = [toks2.index(t2) for t2 in members
                      if t2 in toks2]
                if len(mi) < 2:
                    continue
                M = ind2[:, mi] > 0
                active = M.sum(axis=1) >= 2
                n_act = int(active.sum())
                if n_act < 3 * self.k:
                    continue
                if len(set(gid[active])) < self.k:
                    continue
                nrow2 = len(ind2)
                # THE JOINT PATTERN DISTRIBUTION, k-screened:
                # placing members independently within the
                # stratum cannot encode the joint - the
                # decoder's scores COMPETE (trained on rows
                # holding 2-3 of 6 members) and members
                # partitioned the stratum, driving overlap to
                # its forced minimum (measured: mean drift
                # 0.13-0.20 with undershoots to -0.43, WORSE
                # than pairwise chasing). So the combinations
                # themselves are published - each pattern a
                # row-combination of members standing on at
                # least k DISTINCT PATIENTS - and designated
                # rows DRAW patterns. No published pattern is
                # an individual's: each is an aggregate k
                # people share.
                names2 = [toks2[i3] for i3 in mi]
                pat_counts = {}
                act_idx = np.nonzero(active)[0]
                for ri in act_idx:
                    key = tuple(sorted(
                        names2[z] for z in range(len(mi))
                        if M[ri, z]))
                    pat_counts.setdefault(key, []).append(
                        gid[ri])
                patterns = []
                kept_rows = 0
                for key, gids in pat_counts.items():
                    if len(set(gids)) >= self.k:
                        patterns.append(
                            (list(key),
                             len(gids) / float(nrow2)))
                        kept_rows += len(gids)
                if not patterns:
                    continue
                # THE COVERAGE GATE, learned on the real
                # extract the hard way: union-find chained the
                # implication edges into GIANT components - 56
                # members in active_drugs whose k-cleared
                # patterns covered 1.6% of rows against a
                # column averaging 3.7 tokens - and the tier
                # wiped every member and rebuilt only that
                # sliver, with background explicitly AVOIDING
                # co-occurrence: close crashed 84.6 -> 64.9 and
                # direction failed for the first time in the
                # campaign. A cluster whose joint the k rule
                # cannot mostly publish is not a cluster this
                # tier may own; it is REJECTED by name with its
                # numbers, and the pairwise quotas stand.
                cov = kept_rows / float(max(n_act, 1))
                if cov < 0.6:
                    self._rejected_clusters.append(
                        {"column": cb2, "members": len(mi),
                         "patterns": len(patterns),
                         "coverage": round(float(cov), 3)})
                    print("  cluster of {} in {} REJECTED: "
                          "{} patterns cover {}% of its "
                          "co-presence - the k rule cannot "
                          "publish this cluster's joint; "
                          "pairwise quotas stand".format(
                              len(mi), cb2, len(patterns),
                              round(100 * cov, 1)))
                    continue
                act_share = kept_rows / float(nrow2)
                self._token_clusters.append(
                    (cb2, names2, float(act_share),
                     patterns))
                print("  cluster of {} in {}: {} - {} joint "
                      "patterns clear k and are drawn as "
                      "OBJECTS at generation ({}% of rows)"
                      .format(len(mi), cb2,
                              ", ".join(sorted(names2)[:6])
                              + ("..." if len(mi) > 6
                                 else ""),
                              len(patterns),
                              round(100 * act_share, 1)))
        # one mechanism per property: same-column pairwise rules
        # whose BOTH ends sit in one cluster are superseded by
        # the joint placement - two mechanisms moving one
        # property would fight
        # a clustered member must not be a quota TARGET at
        # all: the quota enforcement WIPES its picks and
        # reallocates, silently erasing the pattern placement
        # that ran before it - measured as byte-identical
        # metrics across two different placement builds, the
        # seventeen-digits tell. Clustered members may still
        # SERVE as triggers for other tokens' quotas; their
        # own placement has exactly one owner.
        _clustered = {(cb2, t2)
                      for cb2, mem2, _a2, _m2
                      in self._token_clusters for t2 in mem2}
        self._token_implications = [
            (cb2, b2, conds2)
            for cb2, b2, conds2 in self._token_implications
            if (cb2, b2) not in _clustered]
        self._near_identities = []
        # Two DIFFERENT exclusions, and conflating them silently
        # disabled the tier: a near-CHILD must not be an exact
        # identity's parent (re-deriving it would break the
        # arithmetic just enforced), but an exact parent is a
        # perfectly good near-PARENT - map_cuff both feeds the
        # exact MAP identity and is the sibling the sparse bmdi
        # twin is computed from.
        _no_child = set(_claimed)
        for _c2, _pa2, _pb2, _w9 in self._linear_identities:
            _no_child.add(_pa2)
            _no_child.add(_pb2)
        _near_children = set()
        _num_by = dict(num_items)
        _order = [c for c, _ in num_items]
        for ci, child in enumerate(_order):
            if child in _no_child or child in _near_children:
                continue
            cv = _num_by[child]
            best = None
            for pa in _order:
                if pa == child or pa in _near_children:
                    continue
                pv = _num_by[pa]
                # the CHILD is the sparser column: compute the
                # occasional device from the routine one
                if pv.notna().mean() < cv.notna().mean():
                    continue
                both = cv.notna() & pv.notna()
                if int(both.sum()) < 200:
                    continue
                r_ = float(cv[both].corr(pv[both]))
                if abs(r_) < 0.95:
                    continue
                A2 = np.column_stack(
                    [pv[both], np.ones(int(both.sum()))])
                w2, *_ = np.linalg.lstsq(
                    A2, cv[both].to_numpy(), rcond=None)
                resid = cv[both].to_numpy() - A2 @ w2
                rs = float(resid.std())
                csd = float(cv[both].std()) or 1.0
                if rs < 0.35 * csd and (
                        best is None or rs < best[0]):
                    best = (rs, pa, [float(w2[0]),
                                     float(w2[1])])
            if best is not None:
                rs, pa, w2 = best
                self._near_identities.append(
                    (child, pa, w2, rs))
                _near_children.add(child)
                print("  {} ~= {:.4g}*{} + {:.4g} (resid sd "
                      "{:.3g}) on the source: the sparser twin "
                      "is computed from its sibling plus "
                      "matched noise, not decoded alone"
                      .format(child, w2[0], pa, w2[1], rs))
        # THE SPARSE-CONDITIONAL COUPLINGS: for each column whose
        # marginal rides the k-banded grid, the strongest rank
        # correlation measurable among its PRESENT rows - at
        # least 30 both-present rows AND at least k distinct
        # patients behind them, so the published number is an
        # aggregate k people stand behind. Enforced at
        # generation by rearrangement; detected here so a table
        # without the structure publishes nothing and the tier
        # stays inert.
        self._sparse_couplings = []
        _sparse_cols = [it[1] for it in self.plan
                        if it[0] == "num"
                        and it[4][6] is not None]
        _owned = set(_claimed) | _near_children
        for child in _sparse_cols:
            if child in _owned:
                continue
            cv = _num_by[child]
            # ALL qualifying partners, strongest first, capped -
            # one conditioner under-covers (the token-quota
            # lesson): enforcing only the single best rank
            # coupling left the second-best to noise, and the
            # second-best was the pair the real extract
            # inverts on.
            cands = []
            for pa in _order:
                if pa == child:
                    continue
                pv = _num_by[pa]
                both = cv.notna() & pv.notna()
                if int(both.sum()) < 30:
                    continue
                if len(set(gid[both.to_numpy()])) < self.k:
                    continue
                rho = float(cv[both].corr(pv[both],
                                          method="spearman"))
                if not np.isfinite(rho) or abs(rho) < 0.25:
                    continue
                cands.append((abs(rho), pa, rho))
            cands.sort(reverse=True)
            # NO CAP. The first cut kept the top four by |rho|,
            # and on the real extract the invasive-pressure
            # family filled all four slots at +0.4..+0.6 while
            # the -0.31 coupling with procedure_count - THE pair
            # the gate measures, the one this tier was built
            # for - ranked fifth and was cut: the inversion
            # survived its own fix. The conventions file
            # recorded this exact failure for token-quota
            # conditioners ("a cap could exclude exactly the
            # partners the gate measures") and the cap was
            # removed there; same lesson, same remedy. The
            # joint solve bounds total variance at one unit by
            # construction, so breadth costs nothing.
            if cands:
                self._sparse_couplings.append(
                    (child, [(pa, rho)
                             for _a, pa, rho in cands]))
                print("  {} present on a sliver of rows: rank "
                      "corr {} measured among present rows "
                      "(k-screened), enforced jointly by "
                      "rearrangement at generation".format(
                          child, ", ".join(
                              "{:+.2f} with {}".format(rho, pa)
                              for _a, pa, rho in cands)))
        # float32, not float64: the encoded frame is the biggest
        # array here and half of it is one-hot zeros.
        X = np.hstack(mats).astype(np.float32, copy=False)
        self.d = X.shape[1]
        wide = [(it[1], len(it[2])) for it in self.plan
                if it[0] == "cat" and len(it[2]) > 20]
        print("  encoded frame: {} rows x {} dimensions"
              "{}".format(X.shape[0], X.shape[1],
                          "  (widest: " + ", ".join(
                              "{} {}".format(c, n)
                              for c, n in sorted(
                                  wide, key=lambda t: -t[1])[:3])
                          + ")" if wide else ""))
        if self.b is None:
            self.b = int(max(8, min(32, self.d // 6)))
        if self.h is None:
            self.h = int(max(64, min(256, 2 * self.d)))
        self.net = MLPRegressor(
            hidden_layer_sizes=(self.h, self.b, self.h),
            activation="relu", max_iter=800, tol=1e-5,
            random_state=self.seed, early_stopping=False)
        if self.denoise > 0:
            # IN PLACE, IN CHUNKS. `X + normal(size=X.shape)`
            # allocates a second frame the size of the first and
            # is what actually ran out of memory on the real
            # extract.
            rs2 = np.random.RandomState(self.seed + 5)
            Xn = X.copy()
            step = max(1, 200000 // max(X.shape[1], 1))
            for a0 in range(0, len(Xn), step):
                b0 = min(a0 + step, len(Xn))
                Xn[a0:b0] += rs2.normal(
                    0, self.denoise,
                    (b0 - a0, X.shape[1])).astype(np.float32)
            self.net.fit(Xn, X)
            del Xn
        else:
            self.net.fit(X, X)
        z = self._encode(X)
        # The support index: how many distinct PATIENTS stand
        # behind a region of the latent space. Measured against a
        # bounded reference sample of the training latents -
        # exact nearest-neighbor over every row is quadratic and
        # would double generation time, and the count below is an
        # ESTIMATE, reported as one.
        from sklearn.neighbors import NearestNeighbors as _NN
        codes = pd.factorize(pd.Series(gid))[0]
        rs = np.random.RandomState(self.seed + 77)
        if len(X) > self.support_ref:
            sel = rs.choice(len(X), self.support_ref,
                            replace=False)
        else:
            sel = np.arange(len(X))
        # SUPPORT IS MEASURED WHERE THE PATTERN LIVES - in the
        # FEATURE space, not the latent one. "A pattern fewer
        # than k records support" means a rare COMBINATION OF
        # VALUES; latent density is a different question, and
        # measuring it there found sub-k regions on 0.3% of rows
        # while the weights-leak attack was reading 0.78. The
        # honest space for the k question is the one k-anonymity
        # has always been asked in.
        self._ref_x = X[sel]
        self._ref_codes = codes[sel]
        self._lat_sd = z.std(axis=0)
        self._lat_sd[self._lat_sd == 0] = 1.0
        _nnn = min(self.support_neighbors, len(self._ref_x))
        self._nn_x = _NN(n_neighbors=_nnn).fit(self._ref_x)
        # THE K QUESTION NEEDS A RADIUS, NOT A COUNT. Counting
        # distinct patients among a FIXED number of neighbors
        # cannot find an isolated group: a cluster of four
        # patients still reports ten, because the neighbor list
        # spills into the general population once their own rows
        # run out. A fixed radius - the typical distance to the
        # 40th neighbor - asks the honest question instead: how
        # many distinct patients are actually NEAR this
        # combination of values.
        _d = self._nn_x.kneighbors(
            self._ref_x, n_neighbors=_nnn)[0]
        self._radius = float(np.median(_d[:, -1])) or 1e-6
        # training points that are THEMSELVES k-supported: the
        # latent pool a hopeless draw is snapped into
        own = self.support_of_rows(X)
        self._dense = z[own >= self.k]
        if not len(self._dense):
            self._dense = z
        self.gmm = _mixture(z, min(10, len(df) // 200), self.seed,
                            np.maximum(z.std(axis=0), 1e-6))
        self._Xtrain = X
        if self.hierarchical:
            n_pat = int(codes.max()) + 1
            centres = np.zeros((n_pat, z.shape[1]))
            np.add.at(centres, codes, z)
            counts = np.bincount(codes, minlength=n_pat)
            centres /= np.maximum(counts, 1)[:, None]
            dev = z - centres[codes]
            # A MIXTURE CANNOT HAVE MORE COMPONENTS THAN
            # SAMPLES. A table with ONE entity gives one patient
            # centre, and asking for two crashed the whole run -
            # found by sweeping the shapes a customer could
            # plausibly hand over, not by any fixture here.
            self.gmm_pat = _mixture(centres, min(8, n_pat // 40),
                                    self.seed,
                                    np.maximum(z.std(axis=0),
                                               1e-6))
            self.gmm_dev = _mixture(dev, min(8, len(z) // 400),
                                    self.seed + 1,
                                    np.maximum(dev.std(axis=0),
                                               1e-6))
            # VISIT COUNTS ARE K-SCREENED LIKE EVERYTHING ELSE:
            # one patient with an extreme number of visits must
            # not set the published tail, so counts are clipped
            # to the MEAN of the k most extreme patients' counts
            # - the blueprint's own bound rule, applied to a
            # count instead of a value.
            srt = np.sort(counts)
            kk = min(self.k, max(1, len(srt) // 2))
            lo_c = max(1, int(round(srt[:kk].mean())))
            hi_c = max(lo_c, int(round(srt[-kk:].mean())))
            self._visit_counts = np.clip(counts, lo_c, hi_c)
            self._mean_visits = float(
                self._visit_counts.mean()) or 1.0
        if self.suppress_top > 0:
            # rank the source's numeric pairs by |spearman|, take
            # the strongest, and suppress one child per pair (the
            # SPARSER column, so the fuller one keeps its own
            # relationships) - a pair is removed by shuffling the
            # child, which breaks ALL of the child's edges, so the
            # collateral is counted and stated.
            num = [c for c in df.columns
                   if c != group_by and
                   pd.to_numeric(df[c], errors="coerce"
                                 ).notna().mean() > 0.5]
            ranked = []
            for i in range(len(num)):
                for j in range(i + 1, len(num)):
                    a = pd.to_numeric(df[num[i]],
                                      errors="coerce")
                    b = pd.to_numeric(df[num[j]],
                                      errors="coerce")
                    m = a.notna() & b.notna()
                    if int(m.sum()) < 30:
                        continue
                    r = a[m].corr(b[m], method="spearman")
                    if r is None or not np.isfinite(r):
                        continue
                    ranked.append((abs(float(r)), num[i],
                                   num[j], float(r)))
            ranked.sort(reverse=True)
            chosen, seen = [], set()
            for _ar, ca, cb, rr in ranked:
                # one child per removed pair; the sparser column
                # is the child, so the denser keeps its edges
                child = (cb if df[cb].notna().mean()
                         <= df[ca].notna().mean() else ca)
                if child in seen:
                    continue
                seen.add(child)
                chosen.append((ca, cb, child, round(rr, 3)))
                if len(chosen) >= self.suppress_top:
                    break
            self._auto_suppressed = chosen
            for ca, cb, child, rr in chosen:
                self.suppress_pairs.append((ca, cb))
            if chosen:
                print("  auto-suppress top {}: removing the "
                      "strongest relationship(s) - {}".format(
                          self.suppress_top,
                          ", ".join("{}~{} (r={})".format(
                              ca, cb, rr)
                              for ca, cb, child, rr
                              in chosen)))
        return self

    def support_of_rows(self, Xrows):
        """Distinct training PATIENTS whose records sit closest
        to each row - the k rule asked of a COMBINATION OF
        VALUES, which is what 'a pattern fewer than k records
        support' means. Estimated against a bounded reference
        sample and reported as an estimate."""
        idx = self._nn_x.radius_neighbors(
            Xrows, radius=self._radius, return_distance=False)
        return np.array([len(np.unique(self._ref_codes[i]))
                         if len(i) else 0 for i in idx])

    def _blur_sparse(self, Z, rng, rounds=5):
        """Blur any draw that lands where fewer than k patients
        stand, escalating until it does - and snap the hopeless
        remainder into a k-supported region. The dense majority
        of the table is untouched, which is why average fidelity
        survives while the rare combination does not."""
        n0 = len(Z)
        blurred = np.zeros(n0, dtype=bool)
        for r in range(rounds):
            # decode first: the k question is asked of the ROW
            # the draw would become, not of the code that makes
            # it.
            weak = self.support_of_rows(
                self._decode(np.maximum(Z, 0.0))) < self.k
            if not weak.any():
                break
            blurred |= weak
            Z[weak] = Z[weak] + rng.normal(
                0, 0.3 * (r + 1) * self._lat_sd,
                (int(weak.sum()), Z.shape[1]))
        weak = self.support_of_rows(
            self._decode(np.maximum(Z, 0.0))) < self.k
        n_snap = int(weak.sum())
        if n_snap:
            pick = rng.randint(0, len(self._dense), n_snap)
            Z[weak] = self._dense[pick] + rng.normal(
                0, 0.4 * self._lat_sd, (n_snap, Z.shape[1]))
        return Z, {"rows": n0, "blurred": int(blurred.sum()),
                   "snapped": n_snap}

    def _fwd(self, A, layers):
        W, B = self.net.coefs_, self.net.intercepts_
        for i in layers:
            A = A @ W[i] + B[i]
            if i < len(W) - 1:
                A = np.maximum(A, 0.0)
        return A

    def _encode(self, X):
        return self._fwd(X, [0, 1])

    def _decode(self, Z):
        return self._fwd(Z, [2, 3])

    def generate(self, n: int, seed: int = None,
                 n_patients: int = None) -> pd.DataFrame:
        rng = np.random.RandomState(
            self.seed if seed is None else seed)
        _set_state = {}
        pid = None
        if self.hierarchical:
            P = n_patients or max(
                1, int(round(n / self._mean_visits)))
            centres, _ = self.gmm_pat.sample(P)
            centres = centres[rng.permutation(P)]
            counts = rng.choice(self._visit_counts, P)
            total = int(counts.sum())
            dev, _ = self.gmm_dev.sample(total)
            dev = dev[rng.permutation(total)] * self.dev_scale
            Z = np.repeat(centres, counts, axis=0) + dev
            pid = np.repeat(
                ["S{:06d}".format(i) for i in range(P)], counts)
        else:
            Z, _ = self.gmm.sample(n)
            Z = Z[rng.permutation(n)]
        if self.k_blur:
            Z, self.blur_report = self._blur_sparse(Z, rng)
        X = self._decode(np.maximum(Z, 0.0))
        out = {}
        i = 0
        for item in self.plan:
            kind, c = item[0], item[1]
            if kind == "num":
                (_, _, mu, sd,
                 (lo, hi, miss_p, integral, dec, dspec,
                  qgrid)) = item
                # The decoder is unbounded; the PUBLISHED bound is
                # not. Clipping costs nothing the network ever
                # legitimately learned - it never saw past the
                # k-anonymous bound - and makes the bound hold by
                # construction rather than by hope.
                vals = np.clip(X[:, i] * sd + mu, lo, hi)
                if integral:
                    vals = np.round(vals)
                elif dec:
                    vals = np.round(vals, dec)
                # THE MISSING CUT IS PLACED WHERE THE SOURCE RATE
                # SAYS, not at a flat 0.5. A column absent on 53%
                # of rows drives its reconstructed indicator above
                # 0.5 nearly everywhere, and thresholding there
                # emptied the column completely - present on 47%
                # of real rows and 0% of generated ones. Taking
                # the quantile at the published rate reproduces
                # that rate by construction.
                ind = X[:, i + 1]
                if miss_p <= 0:
                    miss = np.zeros(len(ind), dtype=bool)
                elif miss_p >= 1:
                    miss = np.ones(len(ind), dtype=bool)
                else:
                    cut = float(np.quantile(ind, 1.0 - miss_p))
                    miss = ind >= cut
                v = pd.Series(vals)
                v[miss] = np.nan
                if qgrid is not None and (~miss).sum() >= 2:
                    # rank-map the starved channel through the
                    # k-banded published grid - WITHIN the rows
                    # that survived the missing cut. The first
                    # cut ranked ALL rows and then masked, so
                    # the present subset sampled a biased slice
                    # of the grid: means drifted to 90.7 and
                    # 64.7 against a source 73.3 and the sd
                    # overshot 2x. Ranked among the living, the
                    # present-row marginal IS the grid.
                    live = np.where(~miss)[0]
                    ch = X[live, i]
                    r = ch.argsort().argsort().astype(float)
                    r = (r + 0.5) / len(live)
                    qq = (np.arange(len(qgrid)) + 0.5) / len(qgrid)
                    gv = np.clip(np.interp(r, qq, qgrid), lo, hi)
                    if integral:
                        gv = np.round(gv)
                    elif dec:
                        gv = np.round(gv, dec)
                    v.iloc[live] = gv
                if dspec:
                    # Back to text, in the DAY format - the same
                    # inverse the rules engine uses, so a date
                    # means one thing whichever engine wrote it.
                    v = dates.from_ordinal(
                        v.to_numpy(dtype=float),
                        dspec.get("format") or "%Y-%m-%d",
                        dspec.get("origin") or dates.DATE_ORIGIN)
                out[c] = v
                i += 2
            elif kind == "set":
                toks = item[2]
                (sep, smu, ssd, smax, tgt,
                 _above, grid) = item[3]
                nt = len(toks)
                A = np.clip(X[:, i + 1:i + 1 + nt], 1e-6, 1.0)
                nrow = len(A)
                # EACH TOKEN PICKS ITS OWN ROWS; TOKENS DO NOT
                # COMPETE WITHIN A ROW. The previous draw ranked
                # tokens against each other inside each row and
                # took the top size-many - and any within-row
                # competition mechanically ties every token to
                # the SET SIZE, so a token whose correlation with
                # a count runs AGAINST the size mechanism cannot
                # survive it. Measured on a two-regime fixture
                # (acute visits: many drugs, no Oral; ambulatory:
                # few drugs, Oral): source corr(has_Oral, count)
                # -0.25, and the competitive draw emitted +0.04
                # to +0.16 at EVERY sharpness - a sign flip, the
                # exact class of the real extract's
                # has_Oral <- active_drug_count 0.26 -> -0.64
                # INVERTED. Here each token takes its top
                # p_t-share of rows by ITS OWN score, so its
                # placement follows the decoder's activation for
                # THAT token: marginal shares exact by
                # construction, correlations free to point
                # whichever way the network learned.
                gmb = -np.log(-np.log(
                    rng.rand(nrow, nt) + 1e-12) + 1e-12)
                score = TOKEN_SHARPNESS * np.log(A) + gmb
                # the EMPTY mass is placed where total token
                # activation is lowest - the published share of
                # genuinely-empty rows, on the rows the decoder
                # says hold the least
                grid_arr = np.asarray(grid, dtype=float)
                e_share = float((np.round(grid_arr) <= 0).mean())
                rowmass = A.sum(axis=1) + 1e-9 * rng.rand(nrow)
                n_empty = int(round(e_share * nrow))
                empty_mask = np.zeros(nrow, dtype=bool)
                if n_empty > 0:
                    empty_mask[np.argsort(rowmass)[:n_empty]] \
                        = True
                score[empty_mask] = -1e18
                t_arr = np.clip(np.asarray(tgt, dtype=float),
                                0.0, 1.0)
                pick = np.zeros((nrow, nt), dtype=bool)
                avail = nrow - n_empty
                for j in range(nt):
                    cj = int(round(min(t_arr[j] * nrow, avail)))
                    if cj > 0:
                        idx = np.argpartition(
                            -score[:, j], cj - 1)[:cj]
                        pick[idx, j] = True
                # a non-empty row that picked nothing still holds
                # SOMETHING - an empty set would claim the visit
                # had none, and the zero/nonzero split is already
                # exactly the published one
                none = ~empty_mask & ~pick.any(axis=1)
                if none.any():
                    # THE FORCED PICK IS THE ROW'S BEST TOKEN
                    # RELATIVE TO THAT TOKEN'S OWN SCALE, never
                    # the raw argmax. Raw argmax sends every
                    # starved row to the same globally-strong
                    # token, concentrating the forced mass on
                    # the SMALLEST rows - measured: the top
                    # token's corr with its count partner
                    # flipped +0.26 -> -0.10. Standardized per
                    # token, the forcing spreads across the
                    # vocabulary and each token's quota barely
                    # moves.
                    # ...standardized over the NON-EMPTY rows:
                    # the empty rows' scores were set to -1e18
                    # above, and a mean taken over them poisons
                    # the standardization back into a global
                    # argmax - the exact behavior this exists
                    # to avoid. Watched happen: the fix's first
                    # cut read WORSE than the bug it fixed.
                    live = score[~empty_mask]
                    mu_s = live.mean(axis=0)
                    sd_s = live.std(axis=0)
                    sd_s[sd_s == 0] = 1.0
                    zrel = (score[none] - mu_s) / sd_s
                    best = np.argmax(zrel, axis=1)
                    pick[np.nonzero(none)[0], best] = True
                    # ...and the quota is RESTORED, because the
                    # forced picks pile onto the globally
                    # strongest tokens - measured at +0.098 on
                    # the top token of the 400-token fixture,
                    # twice the share tolerance. Each over-quota
                    # token drops its weakest picks, taken only
                    # from rows that keep at least two, so no
                    # row returns to empty and the marginals
                    # land back on round(p*n).
                    sizes_now = pick.sum(axis=1)
                    for j in np.unique(best):
                        cj = int(round(min(t_arr[j] * nrow,
                                           avail)))
                        over_n = int(pick[:, j].sum()) - cj
                        if over_n <= 0:
                            continue
                        rows_j = np.nonzero(
                            pick[:, j] & (sizes_now >= 2))[0]
                        if not len(rows_j):
                            continue
                        order = rows_j[np.argsort(
                            score[rows_j, j])]
                        drop = order[:over_n]
                        pick[drop, j] = False
                        sizes_now[drop] -= 1
                # the k rule's size cap still binds: a row over
                # the published maximum drops its weakest picks
                cap = int(min(smax, float(nt)))
                sizes_now = pick.sum(axis=1)
                over = np.nonzero(sizes_now > cap)[0]
                for r in over:
                    js = np.nonzero(pick[r])[0]
                    drop = js[np.argsort(score[r, js])][
                        :len(js) - cap]
                    pick[r, drop] = False
                vals = []
                for r in range(nrow):
                    js = np.nonzero(pick[r])[0]
                    vals.append(sep.join(toks[j] for j in js)
                                if len(js) else "")
                out[c] = pd.Series(vals)
                # state kept for the implication pass below -
                # the restore needs the scores, and by frame
                # time they are gone
                _set_state[c] = {"toks": toks, "sep": sep,
                                 "score": score, "pick": pick,
                                 "t_arr": t_arr,
                                 "n_empty": n_empty,
                                 "rowmass": rowmass}
                i += 1 + nt
            else:
                levels = item[2]
                block = np.maximum(X[:, i:i + len(levels)], 1e-9)
                # SAMPLE, do not argmax: argmax hands every row
                # the modal level and the marginal collapses.
                pr = block / block.sum(axis=1, keepdims=True)
                cum = pr.cumsum(axis=1)
                u = rng.rand(len(pr), 1)
                idx = (u > cum).sum(axis=1)
                out[c] = pd.Series(
                    [levels[min(j, len(levels) - 1)]
                     for j in idx])
                i += len(levels)
        # THE CONDITIONAL-QUOTA PASS. Each adopted token b is
        # REASSIGNED: round(q * |trigger|) of its exactly-cj
        # picks go to the rows where its conditioner landed
        # (best by b's own score within them), the rest to the
        # best rows outside. Marginals stay exactly cj by
        # construction; empty rows stay empty (selection is
        # over live rows only); rows the reassignment starves
        # are repaired with their best remaining token, the base
        # draw's own rule. Conditioners come from EARLIER plan
        # columns only, so a trigger always reads a column that
        # will not move again.
        if getattr(self, "_token_clusters", None) \
                and _set_state:
            # THE CLUSTER IS PLACED AS ONE OBJECT: rows are
            # ranked by the decoder's own summed member signal,
            # the top act_share are designated, and every
            # member puts its published in-cluster mass on
            # THOSE rows and its background elsewhere -
            # co-occurrence arises from shared placement, not
            # from tokens chasing each other's layouts.
            # Marginals exact by construction.
            for cb, members, act_share, patterns in \
                    self._token_clusters:
                if cb not in _set_state:
                    continue
                st = _set_state[cb]
                toks_b = st["toks"]
                pick_b = st["pick"]
                score_b = st["score"]
                live = score_b[:, 0] > -1e17
                nrow_b = len(pick_b)
                mi = [toks_b.index(t) for t in members
                      if t in toks_b]
                if len(mi) < 2:
                    continue
                live_idx = np.nonzero(live)[0]
                Z = score_b[live_idx][:, mi]
                mu_z = Z.mean(axis=0)
                sd_z = Z.std(axis=0)
                sd_z[sd_z == 0] = 1.0
                act_score = ((Z - mu_z) / sd_z).sum(axis=1)
                # designated rows draw PATTERNS - the published
                # joint combinations - so co-occurrence is
                # source-stated, not emergent from competing
                # scores. Pattern counts scale from published
                # shares; larger patterns (stronger activity)
                # take the highest-activity rows, keeping the
                # decoder's own signal relevant.
                quota_pat = [(pt, int(round(sh * nrow_b)))
                             for pt, sh in patterns]
                quota_pat = [(pt, n9) for pt, n9 in quota_pat
                             if n9 > 0]
                N = sum(n9 for _pt, n9 in quota_pat)
                N = min(N, len(live_idx))
                desig = live_idx[np.argsort(-act_score)[:N]]
                desig_mask = np.zeros(nrow_b, dtype=bool)
                desig_mask[desig] = True
                pre_ne = pick_b.any(axis=1)
                for jx in mi:
                    pick_b[:, jx] = False
                quota_pat.sort(key=lambda pn: -len(pn[0]))
                pos = 0
                placed_in = {t: 0 for t in members}
                for pt, n9 in quota_pat:
                    rows9 = desig[pos:pos + n9]
                    pos += n9
                    if not len(rows9):
                        break
                    for t in pt:
                        if t in toks_b:
                            pick_b[rows9,
                                   toks_b.index(t)] = True
                            placed_in[t] = placed_in.get(
                                t, 0) + len(rows9)
                # background singles AVOID rows already
                # holding another member - placed by top score
                # they co-occur by accident and the co-presence
                # share overshot its published value (0.108
                # against 0.088); patterns alone own the joint
                member_any = pick_b[:, mi].any(axis=1)
                for jx in mi:
                    t = toks_b[jx]
                    cj = int(round(min(
                        st["t_arr"][jx] * nrow_b,
                        int(live.sum()))))
                    n_bg = cj - placed_in.get(t, 0)
                    if n_bg > 0:
                        oi = np.nonzero(live & ~desig_mask
                                        & ~member_any)[0]
                        o_out = oi[np.argsort(
                            -score_b[oi, jx])]
                        take_bg = o_out[:n_bg]
                        pick_b[take_bg, jx] = True
                        member_any[take_bg] = True
                        short = n_bg - len(take_bg)
                        if short > 0:
                            o2 = np.nonzero(
                                live & ~desig_mask
                                & ~pick_b[:, jx])[0]
                            o2 = o2[np.argsort(
                                -score_b[o2, jx])]
                            pick_b[o2[:short], jx] = True
                starved = live & pre_ne \
                    & ~pick_b.any(axis=1)
                if starved.any():
                    live_sc = score_b[live]
                    mu_b = live_sc.mean(axis=0)
                    sd_b3 = live_sc.std(axis=0)
                    sd_b3[sd_b3 == 0] = 1.0
                    zb = (score_b[starved] - mu_b) / sd_b3
                    bestb = np.argmax(zb, axis=1)
                    pick_b[np.nonzero(starved)[0],
                           bestb] = True
                # THE REPAIR REOPENED CLOSED BOOKS: the wipe
                # empties every base row that held only
                # members, those rows starve, and argmax
                # repair piled thousands onto whichever member
                # the net favored - diphenhydrAMINE's share
                # read 0.211 against a source 0.078 while p11
                # stayed EXACT, so the "faded" correlations
                # were marginal inflation, not joint loss. The
                # recorded repair-overshoot rule applies:
                # restore each member's count by dropping its
                # weakest BACKGROUND picks - pattern rows are
                # never touched.
                for jx in mi:
                    cj = int(round(min(
                        st["t_arr"][jx] * nrow_b,
                        int(live.sum()))))
                    excess = int(pick_b[:, jx].sum()) - cj
                    if excess <= 0:
                        continue
                    cand = np.nonzero(pick_b[:, jx]
                                      & ~desig_mask)[0]
                    if not len(cand):
                        continue
                    weakest = cand[np.argsort(
                        score_b[cand, jx])][:excess]
                    pick_b[weakest, jx] = False
                # SERIALIZE, here, now. The quota loop only
                # re-writes out[cb] for columns it still owns -
                # and a cluster column is no longer among them,
                # so three successive placement builds mutated
                # pick arrays that nothing re-serialized and
                # produced the base draw's strings verbatim:
                # byte-identical metrics, the narration saying
                # "placed", the file never hearing about it.
                sep_c = st["sep"]
                vals_c = []
                for r in range(nrow_b):
                    js = np.nonzero(pick_b[r])[0]
                    vals_c.append(sep_c.join(
                        toks_b[j] for j in js)
                        if len(js) else "")
                out[cb] = pd.Series(vals_c)
                print("  cluster placed for {}: {} members, "
                      "{} patterns on {} designated rows"
                      .format(cb, len(mi), len(quota_pat), N))
        if getattr(self, "_token_implications", None) \
                and _set_state:
            by_b = {}
            for cb, b_, conds in self._token_implications:
                conds = [(ca, a_, q) for ca, a_, q in conds
                         if ca in _set_state]
                if cb in _set_state and conds:
                    by_b.setdefault(cb, []).append(
                        (b_, conds))
            # ONE PASS, in insertion order - which already
            # sequences the within-column cluster before the
            # cross-column routes, so route triggers read the
            # restored cluster. A second pass was measured and
            # REVERTED: symmetric pairs double-dip and the
            # cluster OVERSHOOTS (acetaminophen~pantoprazole
            # src +0.20 to gen +0.45) - manufactured structure
            # with the other sign, the same class as the
            # allocator's manufactured negatives.
            for cb, rules in by_b.items():
                st = _set_state[cb]
                toks_b = st["toks"]
                pick_b = st["pick"]
                score_b = st["score"]
                live = score_b[:, 0] > -1e17
                nrow_b = len(pick_b)
                avail_b = int(live.sum())
                pre_nonempty = pick_b.any(axis=1)
                # same-column triggers read this per-iteration
                # SNAPSHOT, never the live state mid-wipe:
                # symmetric pairs (the acute cluster is full of
                # them) must not read each other's in-flight
                # reallocations - the Gauss-Seidel
                # last-writer-wins fault
                snap_b = pick_b.copy()
                for b_, conds in rules:
                    if b_ not in toks_b:
                        continue
                    jb = toks_b.index(b_)
                    cj = int(round(min(
                        st["t_arr"][jb] * nrow_b, avail_b)))
                    pick_b[:, jb] = False
                    any_trig = np.zeros(nrow_b, dtype=bool)
                    budget = cj
                    # PRO-RATA OVER PASSES, NOT GREEDY BY GAIN.
                    # The greedy order with a hard budget zeroed
                    # whichever conditioners ran last - and the
                    # wipe above had already deleted the decoder
                    # picks their rows used to hold, so a faded
                    # pair became a MANUFACTURED NEGATIVE
                    # (diphenhydrAMINE src +0.36, gen -0.14 on
                    # the fixture; the extract's -0.50 drift at
                    # the head of the close list). When the
                    # budget cannot satisfy every quota, all
                    # needs scale by the same factor; overlap
                    # credit still accrues through the running
                    # pick state across passes.
                    trigs = []
                    for ca, a_, q in conds:
                        sa = _set_state[ca]
                        if a_ not in sa["toks"]:
                            continue
                        ja = sa["toks"].index(a_)
                        src_pick = snap_b if ca == cb \
                            else sa["pick"]
                        trig = src_pick[:, ja] & live
                        any_trig |= trig
                        trigs.append((trig, float(q)))
                    for _pass in range(3):
                        if budget <= 0:
                            break
                        needs = []
                        for trig, q in trigs:
                            n_t = int(trig.sum())
                            have = int((pick_b[:, jb]
                                        & trig).sum())
                            needs.append(max(
                                0, int(round(q * n_t)) - have))
                        total = sum(needs)
                        if total <= 0:
                            break
                        scale = min(1.0, budget / float(total))
                        placed = 0
                        for trig, q in trigs:
                            # FRESH need at placement time, not
                            # the pass-start figure: overlapping
                            # triggers (a comorbid stratum)
                            # otherwise double-fill the overlap
                            # - the first placement lands picks
                            # the second was never credited for,
                            # and P(b|a) OVERSHOOTS its target
                            # (measured: chained ICD pairs src
                            # +0.18..+0.24 generating +0.37..
                            # +0.65; the extract's +0.23 ->
                            # +0.52). The scale stays from pass
                            # start for budget fairness - the
                            # starvation fix - while the credit
                            # is sequential - the overshoot fix.
                            n_t = int(trig.sum())
                            have = int((pick_b[:, jb]
                                        & trig).sum())
                            need = max(0, int(round(q * n_t))
                                       - have)
                            want = int(round(need * scale))
                            if want <= 0:
                                continue
                            free = trig & ~pick_b[:, jb]
                            fi = np.nonzero(free)[0]
                            if not len(fi):
                                continue
                            order = fi[np.argsort(
                                -score_b[fi, jb])]
                            take = order[:min(want, budget)]
                            pick_b[take, jb] = True
                            budget -= len(take)
                            placed += len(take)
                            if budget <= 0:
                                break
                        if placed == 0:
                            break
                    rest = live & ~any_trig
                    want_r = int(min(budget, int(rest.sum())))
                    if want_r > 0:
                        rs_idx = np.nonzero(rest)[0]
                        order = rs_idx[np.argsort(
                            -score_b[rs_idx, jb])]
                        pick_b[order[:want_r], jb] = True
                # starved-row repair: rows the reassignment left
                # with nothing, that the draw had made non-empty
                starved = live & pre_nonempty \
                    & ~pick_b.any(axis=1)
                if starved.any():
                    live_sc = score_b[live]
                    mu_b = live_sc.mean(axis=0)
                    sd_b2 = live_sc.std(axis=0)
                    sd_b2[sd_b2 == 0] = 1.0
                    zb = (score_b[starved] - mu_b) / sd_b2
                    bestb = np.argmax(zb, axis=1)
                    pick_b[np.nonzero(starved)[0], bestb] = True
                sep_b = st["sep"]
                vals_b = []
                for r in range(nrow_b):
                    js = np.nonzero(pick_b[r])[0]
                    vals_b.append(sep_b.join(
                        toks_b[j] for j in js)
                        if len(js) else "")
                out[cb] = pd.Series(vals_b)
        # COLUMNS EMPTY IN THE SOURCE COME BACK EMPTY. Written as
        # "" rather than NaN, because that is what the reader folds
        # back to missing and what a person opening the file expects
        # to see in a field nobody recorded.
        for _ec in getattr(self, "empty_cols", []):
            out[_ec] = pd.Series([""] * n)
        frame = pd.DataFrame(out)
        # THE RELEASE PASS runs before the identity tiers (which
        # recompute their children afterwards, keeping the file
        # internally consistent). Permuting a fraction f of a
        # column scales its pair correlations by (1-f); with both
        # columns diluted the pair carries (1-f)^2, so
        # f = 1 - sqrt(strength) lands pair strength at
        # approximately strength x source. Marginals are EXACT by
        # construction - a permutation moves values, never the
        # multiset. Suppression shuffles the named child FULLY:
        # deleting one edge surgically while keeping its
        # neighbors would leak the edge back through them, so the
        # collateral is taken and STATED rather than hidden.
        rs_rel = np.random.RandomState(
            (seed if seed is not None else self.seed) + 777)
        s_rel = min(1.0, max(0.0, self.release_strength))
        if s_rel < 1.0:
            f_rel = 1.0 - s_rel ** 0.5
            n_rel = len(frame)
            set_cols_rel = {it[1]: it[3][0] for it in self.plan
                            if it[0] == "set"}
            for c_rel in frame.columns:
                m_rel = max(1, int(round(f_rel * n_rel)))
                idx_rel = rs_rel.choice(n_rel, m_rel,
                                        replace=False)
                vals_rel = frame[c_rel].to_numpy(
                    copy=True)
                if c_rel in set_cols_rel:
                    # PERMUTING WHOLE CELLS CANNOT DILUTE A SET
                    # COLUMN: has_a and has_b live in the SAME
                    # cell, so cell shuffling carries their
                    # co-occurrence along intact - measured, the
                    # zero-strength dial left within-column
                    # pairs at 0.9x source while cross-column
                    # pairs died. Tokens are RE-DEALT instead:
                    # the selected rows' tokens pool, shuffle,
                    # and refill by each row's own size - every
                    # token count and every row size exact,
                    # co-occurrence broken.
                    sep_rel = set_cols_rel[c_rel]
                    pool = []
                    sizes = []
                    for v9 in vals_rel[idx_rel]:
                        ts9 = [t for t in str(v9 or "").split(
                            sep_rel) if t]
                        pool.extend(ts9)
                        sizes.append(len(ts9))
                    rs_rel.shuffle(pool)
                    pos9 = 0
                    newcells = []
                    for sz9 in sizes:
                        row_t = pool[pos9:pos9 + sz9]
                        pos9 += sz9
                        newcells.append(sep_rel.join(
                            sorted(set(row_t))))
                    vals_rel[idx_rel] = np.array(
                        newcells, dtype=object)
                else:
                    sub = vals_rel[idx_rel].copy()
                    rs_rel.shuffle(sub)
                    vals_rel[idx_rel] = sub
                frame[c_rel] = vals_rel
            print("  release strength {}: {}% of every column "
                  "permuted - marginals exact, relationships "
                  "diluted toward {}x source".format(
                      s_rel, round(100 * f_rel, 1), s_rel))
        # AFTER EVERY COLUMN EXISTS, never during the draw: the
        # set may be decoded before or after its partner and the
        # identity has to hold either way.
        for cnt_c, set_c, sep_ in getattr(
                self, "_size_identities", []):
            if cnt_c in frame.columns and set_c in frame.columns:
                frame[cnt_c] = sets.sizes_of(
                    frame[set_c], sep_).astype(float)
        # Linear identities, the same rule: computed, not
        # redrawn. The child's own precision and missingness are
        # kept - the arithmetic replaces the VALUE, never the
        # fact of whether it was recorded.
        _plan_by = dict((it[1], it) for it in self.plan)
        for child, pa, pb, w in getattr(
                self, "_linear_identities", []):
            if not all(c in frame.columns
                       for c in (child, pa, pb)):
                continue
            it = _plan_by.get(child)
            dspec_c = it[4][5] if it and it[0] == "num" else None

            def _nm(col):
                jt = _plan_by.get(col)
                dj = jt[4][5] if jt and jt[0] == "num" else None
                if dj:
                    return dates.to_ordinal(
                        frame[col],
                        dj.get("format") or "%Y-%m-%d")
                return pd.to_numeric(frame[col],
                                     errors="coerce")
            va, vb = _nm(pa), _nm(pb)
            vc = _nm(child)
            newv = w[0] * va + w[1] * vb + w[2]
            if it:
                _, _, _, _, (lo_, hi_, _m, integ, dec_,
                             _d, _qg) = it
                newv = newv.clip(lo_, hi_)
                if integ:
                    newv = newv.round()
                elif dec_:
                    newv = newv.round(dec_)
            newv = newv.where(vc.notna() & va.notna()
                              & vb.notna(), vc)
            if dspec_c:
                frame[child] = dates.from_ordinal(
                    newv.to_numpy(dtype=float),
                    dspec_c.get("format") or "%Y-%m-%d")
            else:
                frame[child] = newv
        # Near-identities: the sparser twin re-derived from its
        # sibling plus noise at the measured residual sd, only
        # where BOTH are present - where the sibling is absent
        # the decoded value stands, because a twin of nothing is
        # not computable.
        for child, pa, w2, rs in getattr(
                self, "_near_identities", []):
            if child not in frame.columns \
                    or pa not in frame.columns:
                continue
            it = _plan_by.get(child)
            vc = pd.to_numeric(frame[child], errors="coerce")
            vp = pd.to_numeric(frame[pa], errors="coerce")
            newv = w2[0] * vp + w2[1] + pd.Series(
                rng.normal(0, rs, len(frame)),
                index=frame.index)
            if it and it[0] == "num":
                _, _, _, _, (lo_, hi_, _m, integ, dec_,
                             _d, _qg) = it
                newv = newv.clip(lo_, hi_)
                if integ:
                    newv = newv.round()
                elif dec_:
                    newv = newv.round(dec_)
            frame[child] = newv.where(
                vc.notna() & vp.notna(), vc)
        # THE SPARSE-CONDITIONAL REORDER. A column present on a
        # sliver of rows decodes through a starved channel, so
        # even with its marginal restored by the k-banded grid
        # its ORDERING among present rows is rank-of-noise - and
        # a conditional correlation computed on noise flips sign
        # by seed (measured: src -0.34, eight seeds reading
        # -0.36 to +0.53, two inverted). The coupling is
        # measured on the SOURCE among present rows, k-screened,
        # and enforced here by REARRANGEMENT: the generated
        # present values are reassigned against the partner's
        # ranks to land the published rank correlation. A
        # rearrangement cannot move the marginal - the same
        # property that makes the cyclic sweeps safe.
        for child, partners in getattr(
                self, "_sparse_couplings", []):
            if child not in frame.columns:
                continue
            ps = [(pa, rho) for pa, rho in partners
                  if pa in frame.columns]
            if not ps:
                continue
            vc = pd.to_numeric(frame[child], errors="coerce")
            # LIVE IS THE CHILD'S PRESENT ROWS, NOT THE
            # INTERSECTION. The first cut required every partner
            # present on the same row - and a sparse column's
            # strongest partners are its sparse SIBLINGS, so on
            # the real extract the intersection was empty and
            # the tier silently skipped: two runs, different
            # code, the same diagnostic to seventeen digits.
            # A partner absent on a row contributes its mean
            # (zero after standardizing) instead of vetoing the
            # whole rearrangement.
            live = vc.notna()
            m = int(live.sum())
            if m < 8:
                print("  reorder SKIPPED for {}: only {} "
                      "present rows".format(child, m))
                continue
            idx = frame.index[live]
            cols_z = []
            for pa, _r in ps:
                pv = pd.to_numeric(
                    frame[pa], errors="coerce")[live]
                pm = pv.notna()
                z = np.zeros(m)
                if int(pm.sum()) >= 2:
                    rk = pv[pm].rank(
                        method="average").to_numpy()
                    rk = (rk - rk.mean()) / (rk.std() or 1.0)
                    z[pm.to_numpy()] = rk
                cols_z.append(z)
            P = np.column_stack(cols_z)
            rho_v = np.array([r for _p, r in ps])
            # target score z = P b + residual noise, b solved so
            # corr(z, partner_j) lands each published rho at
            # once: b = R^-1 rho over the partners\' own rank
            # correlations, ridged for near-colinear partners,
            # scaled back if the solve asks for more than one
            # unit of variance.
            R = np.atleast_2d(np.corrcoef(P, rowvar=False))
            # the invasive family is near-colinear; a heavier
            # ridge keeps the uncapped solve from chasing noise
            # in R's small eigenvalues
            bvec = np.linalg.solve(
                R + 1e-3 * np.eye(len(ps)), rho_v)
            v2 = float(bvec @ R @ bvec)
            if v2 > 1.0:
                bvec /= np.sqrt(v2)
                v2 = 1.0
            z = (P @ bvec
                 + np.sqrt(max(1.0 - v2, 0.0))
                 * rng.normal(0, 1, m))
            srt = np.sort(vc[live].to_numpy())
            newv = np.empty(m)
            newv[z.argsort()] = srt
            vc2 = vc.copy()
            vc2.loc[idx] = newv
            frame[child] = vc2.where(vc.notna(), frame[child])
            print("  reorder ran for {}: {} rows, {} "
                  "partner(s)".format(child, m, len(ps)))
        # POISON RUNS LAST too, before suppression: reorder a
        # named child against its parent to hit a user-chosen
        # rank correlation (which may be negative or entirely
        # invented). Same rearrangement the sparse-coupling tier
        # uses - the child's own marginal is preserved exactly,
        # only its ordering against the parent changes - so the
        # column looks real and the relationship is a lie.
        for pa_p, pb_p, rho_p in self.poison_pairs:
            if pa_p not in frame.columns or pb_p not in \
                    frame.columns:
                print("  poison SKIPPED: {}~{} not both "
                      "columns".format(pa_p, pb_p))
                continue
            parent = pd.to_numeric(frame[pa_p], errors="coerce")
            child = pd.to_numeric(frame[pb_p], errors="coerce")
            live = parent.notna() & child.notna()
            m = int(live.sum())
            if m < 8:
                continue
            idx = frame.index[live]
            pr = parent[live].rank(method="average").to_numpy()
            pr = (pr - pr.mean()) / (pr.std() or 1.0)
            z = rho_p * pr + np.sqrt(max(1 - rho_p * rho_p, 0.0)) \
                * rs_rel.normal(0, 1, m)
            srt = np.sort(child[live].to_numpy())
            newv = np.empty(m)
            newv[z.argsort()] = srt
            cc = child.copy()
            cc.loc[idx] = newv
            frame[pb_p] = cc.where(child.notna(), frame[pb_p])
            ach = float(pd.to_numeric(frame[pb_p],
                                      errors="coerce")[live].corr(
                parent[live], method="spearman"))
            self._poison_record.append(
                {"pair": "{}~{}".format(pa_p, pb_p),
                 "requested": rho_p, "achieved": round(ach, 3)})
            print("  POISONED {}~{}: forced to {:+.2f} "
                  "(achieved {:+.2f}) - the column's own "
                  "distribution is intact; the relationship is a "
                  "deliberate lie we recorded".format(
                      pa_p, pb_p, rho_p, ach))
        # SUPPRESSION RUNS LAST - after every identity tier.
        # A suppressed pair is often a near-identity (crown_b =
        # crown_a + noise is exactly the kind of strong
        # relationship worth removing), and the near-identity
        # tier recomputes the child from its sibling: run before
        # it and the tier puts the pattern straight back. Dilution
        # runs BEFORE the tiers (they keep the file consistent and
        # the residual is honest arithmetic); suppression must
        # WIN over them, so it is the final word.
        for pa_s, pb_s in self.suppress_pairs:
            child_s = pb_s if pb_s in frame.columns else (
                pa_s if pa_s in frame.columns else None)
            if child_s is None:
                print("  suppress SKIPPED: neither {} nor {} "
                      "is a column".format(pa_s, pb_s))
                continue
            vals_s = frame[child_s].to_numpy().copy()
            rs_rel.shuffle(vals_s)
            frame[child_s] = vals_s
            print("  suppressed {}~{}: {} fully shuffled - its "
                  "distribution is intact and ALL its "
                  "relationships are gone, stated because a "
                  "single edge cannot be deleted without "
                  "leaking back through its neighbors".format(
                      pa_s, pb_s, child_s))
        if pid is not None:
            # The identity column goes FIRST, and it is invented -
            # S000001 upward - never a real person's id.
            frame.insert(0, self._group_name, pid)
        return frame

    def reconstruction_gap(self, holdout: pd.DataFrame):
        """How much better the network rebuilds the people it
        TRAINED on than people it has never seen.

        THE SPLIT EXISTED AND NOTHING MEASURED THIS. 15% of
        patients are held out, and the only thing asked of them
        was the nearest-neighbor distance ratio - which compares
        the GENERATED file to the training rows and never puts a
        held-out record through the network at all. So the
        question "is the autoencoder memorizing rather than
        learning" had a holdout ready for it and no measurement
        on it, which is this repository's own recurring fault: a
        capability present and not wired to anything.

        Encode and decode both sets and compare squared error. A
        network that has learned the SHAPE of the data rebuilds a
        stranger about as well as a member, so the ratio sits
        near 1. A network that has memorized its training rows
        rebuilds them far better, and the ratio climbs - which is
        the same thing that would make a membership adversary
        work, because reconstruction error IS the signal that
        adversary reads.

        Reported, never silently gated: a number this cheap to
        misread deserves its threshold stated beside it."""
        Xh = self._re_encode_frame(holdout, clip=True)
        if not len(Xh):
            return None
        Xt = self._Xtrain

        def sqerr(A):
            R = self._decode(np.maximum(self._encode(A), 0.0))
            return (R - A) ** 2
        et, eh = sqerr(Xt), sqerr(Xh)
        tr, ho = float(et.mean()), float(eh.mean())
        # PER COLUMN, because one overall ratio cannot say WHERE
        # any gap lives, and the operator asked to SEE the
        # comparison rather than take a single number's word for
        # it. Encoded dimensions are mapped back to the columns a
        # person can name via the plan.
        cols = []
        i = 0
        for item in self.plan:
            kind, c = item[0], item[1]
            w = (2 if kind == "num"
                 else 1 + len(item[2]) if kind == "set"
                 else len(item[2]))
            cols.append({
                "column": c,
                "train": float(et[:, i:i + w].mean()),
                "holdout": float(eh[:, i:i + w].mean())})
            i += w
        return {"train": tr, "holdout": ho,
                "ratio": ho / (tr or 1e-12),
                "rows_train": int(len(Xt)),
                "rows_holdout": int(len(Xh)),
                "columns": cols}

    def nn_ratio(self, gen: pd.DataFrame, holdout_X) -> float:
        """The memorization tripwire. Distance from each generated
        row to its nearest TRAINING row, over the same distance
        for held-out REAL rows. Near 1.0 means the generator sits
        no closer to the training data than fresh real people do;
        well under 1.0 reads as copying and is disqualifying."""
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1).fit(self._Xtrain)
        gX = self._re_encode_frame(gen)
        dg = nn.kneighbors(gX)[0].ravel()
        dh = nn.kneighbors(holdout_X)[0].ravel()
        return float(np.median(dg) / (np.median(dh) or 1.0))

    def _re_encode_frame(self, df, clip=False):
        """A frame back into the encoded space the network was
        trained in.

        `clip` exists because two callers want different things.
        The memorization tripwire compares DISTANCES and has
        always used the raw values, so its recorded numbers stay
        comparable. The reconstruction gap compares ERRORS, and
        there the training rows were clipped to the k-anonymous
        bound before the network ever saw them - so holding the
        holdout to a different rule would charge it for the
        privacy clip and read as overfitting."""
        mats = []
        for item in self.plan:
            kind, c = item[0], item[1]
            if kind == "set":
                toks = item[2]
                sep, smu, ssd = item[3][0], item[3][1], item[3][2]
                szs = sets.publishable_sizes(df[c], sep, toks)
                mats.append(np.column_stack(
                    [((szs - smu) / ssd).fillna(0.0).to_numpy()]
                    + [np.column_stack(
                        [sets.has_token(df[c], sep, t)
                         .fillna(0.0).to_numpy() for t in toks])]))
                continue
            if kind == "num":
                _, _, mu, sd, extra = item
                dsp = extra[5] if len(extra) > 5 else None
                if dsp:
                    # The GENERATED frame carries the day format,
                    # never the parsed one - re-encoding with the
                    # wrong one returns all-NaN and the tripwire
                    # silently measures nothing. A REAL holdout
                    # frame still carries the source's own
                    # format, so both are tried.
                    v = dates.to_ordinal(
                        df[c], dsp.get("format") or "%Y-%m-%d")
                    if not v.notna().any():
                        v = dates.to_ordinal(
                            df[c], dsp["parsed_format"])
                else:
                    v = pd.to_numeric(df[c], errors="coerce")
                if clip:
                    v = v.clip(extra[0], extra[1])
                mats.append(np.column_stack(
                    [((v - mu) / sd).fillna(0.0).to_numpy(),
                     v.isna().astype(float).to_numpy()]))
            else:
                levels = item[2]
                s = df[c].astype(str) \
                    .where(df[c].astype(str).isin(levels),
                           "__other__")
                mats.append(np.column_stack(
                    [(s == l).astype(float).to_numpy()
                     for l in levels]))
        return np.hstack(mats)


# ---------------------------------------------------------------
def _split_patients(df, group_by, seed, frac=0.85):
    rng = np.random.RandomState(seed)
    ids = df[group_by].unique()
    tr = set(rng.choice(ids, int(len(ids) * frac), replace=False))
    m = df[group_by].isin(tr)
    return df[m].reset_index(drop=True), \
        df[~m].reset_index(drop=True)


def _pairs(df, cols):
    out = {}
    for i, a in enumerate(cols):
        for b_ in cols[i + 1:]:
            va = pd.to_numeric(df[a], errors="coerce")
            vb = pd.to_numeric(df[b_], errors="coerce")
            r = va.corr(vb, method="spearman")
            if pd.notna(r):
                out[(a, b_)] = float(r)
    return out


def _pair_score(src, gen):
    sign = close = inv = n = 0
    for p_, r in src.items():
        if p_ not in gen:
            continue
        n += 1
        g = gen[p_]
        if abs(r) >= 0.05:
            if np.sign(g) == np.sign(r) or abs(g) < 0.05:
                sign += 1
            if np.sign(g) == -np.sign(r) and abs(g) >= 0.1 \
                    and abs(r) >= 0.1:
                inv += 1
        else:
            sign += 1
        if abs(g - r) <= 0.2:
            close += 1
    return sign, close, inv, n


def _interaction_r2(df, y, a, b_):
    """How much the PRODUCT term explains beyond the mains - the
    direct read on whether a two-variable interaction exists in
    the file. An XOR has near-zero mains and a large gain."""
    from sklearn.linear_model import LinearRegression
    d = df[[y, a, b_]].apply(
        pd.to_numeric, errors="coerce").dropna()
    if len(d) < 100:
        return np.nan
    X0 = d[[a, b_]].to_numpy()
    X1 = np.column_stack([X0, (d[a] * d[b_]).to_numpy()])
    yv = d[y].to_numpy()
    r0 = LinearRegression().fit(X0, yv).score(X0, yv)
    r1 = LinearRegression().fit(X1, yv).score(X1, yv)
    return float(r1 - r0)


def _shares(frame, child, parents):
    sys.path.insert(0, str(ROOT / "scripts"))
    from repro_mechanisms import _shares as sh
    return sh(frame, child, parents)


def run(which, seeds):
    sys.path.insert(0, str(ROOT / "scripts"))
    from repro_mechanisms import build_pressure, build_triangle
    print("LATENT CHALLENGER - {} - the fidelity-at-any-cost "
          "ruler. Privacy posture: trains on records; output is "
          "a MEASUREMENT, not a release.".format(which))
    rows = []
    for seed in seeds:
        if which == "triangle":
            df = build_triangle()
        elif which == "pressure":
            df = build_pressure()
        else:
            df = _planted_frame(seed)
        tr, ho = _split_patients(df, "person_id", seed)
        g = LatentGen(seed=seed).fit(tr, "person_id")
        gen = g.generate(len(tr), seed=seed + 1000)
        cols = [c for c in df.columns if c != "person_id"]
        s_src = _pairs(tr, cols)
        s_gen = _pairs(gen, cols)
        sign, close, inv, n = _pair_score(s_src, s_gen)
        nn = g.nn_ratio(gen, g._re_encode_frame(ho))
        extra = ""
        if which == "triangle":
            src_sh, _ = _shares(tr, "span_like",
                                ["proc_like", "drug_like"])
            gen_sh, _ = _shares(gen, "span_like",
                                ["proc_like", "drug_like"])
            extra = "attribution src {:.0%}/{:.0%} gen " \
                    "{:.0%}/{:.0%}".format(src_sh[0], src_sh[1],
                                           gen_sh[0], gen_sh[1])
        if which == "planted":
            xs = _interaction_r2(tr, "planted_xor_y",
                                 "planted_xor_a", "planted_xor_b")
            xg = _interaction_r2(gen, "planted_xor_y",
                                 "planted_xor_a", "planted_xor_b")
            ts = _interaction_r2(tr, "planted_3way_y",
                                 "planted_3way_a",
                                 "planted_3way_b")
            tg = _interaction_r2(gen, "planted_3way_y",
                                 "planted_3way_a",
                                 "planted_3way_b")
            extra = "xor-gain src {:.3f} gen {:.3f} | 3way-gain " \
                    "src {:.3f} gen {:.3f}".format(xs, xg, ts, tg)
        print("seed {:>2}: sign {}/{} close {}/{} INVERTED {} | "
              "nn-ratio {:.2f} | {}".format(
                  seed, sign, n, close, n, inv, nn, extra))
        rows.append((sign, close, inv, n, nn))
    ss = sum(r[0] for r in rows)
    cc = sum(r[1] for r in rows)
    ii = sum(r[2] for r in rows)
    nn_ = sum(r[3] for r in rows)
    print()
    print("TOTAL over {} seeds: sign {}/{} close {}/{} INVERTED "
          "{} | median nn-ratio {:.2f}".format(
              len(seeds), ss, nn_, cc, nn_, ii,
              float(np.median([r[4] for r in rows]))))
    print("Read beside the blueprint's numbers on the same "
          "fixture, and remember which of the two passed a "
          "membership attack.")


class ReconstructionAdversary:
    """The likelihood adversary for the latent path, deliberately
    GENEROUS to the attacker: it assumes the trained WEIGHTS leak,
    not just the generated rows, and scores a candidate record by
    how well the autoencoder reconstructs it - members were
    trained on, so they reconstruct better. This is the classic
    autoencoder membership attack, and it is exactly the exposure
    the blueprint path does not have."""

    def __init__(self, gen: LatentGen):
        self.g = gen

    def log_likelihood(self, row) -> float:
        df = pd.DataFrame([row])
        if "person_id" in df.columns:
            df = df.drop(columns=["person_id"])
        X = self.g._re_encode_frame(df)
        R = self.g._decode(self.g._encode(X))
        return -float(np.sum((X - R) ** 2))


def run_attack(seeds, csv=None, group_by="person_id",
               k_blur=True, denoise=0.0):
    """The SAME membership question the blueprint path answered at
    worst 0.52 - asked of the latent path, with the same bands,
    plus the republish control that makes any PASS worth reading.
    Default: the shared cohort fixture, for comparability. With
    --csv: the REAL frame at its real width, which is where an
    autoencoder's memorization risk actually lives - members are
    half the patients, non-members the other half of the SAME
    population."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from membership_new_path import cohort
    from synthkit.attack import membership_audit
    print("MEMBERSHIP ATTACK ON THE LATENT PATH - two "
          "adversaries, worse one reported; the reconstruction "
          "adversary assumes the WEIGHTS leak.")
    print("k-aware blur: {} | denoising training: {}".format(
        "on" if k_blur else "OFF (measurement only)",
        "sd {:.2f}".format(denoise) if denoise else "off"))
    if csv:
        print("frame: {} (real width - rows are subsampled to "
              "1,500 per side for the row-scored adversary)"
              .format(csv))
    base = None
    if csv:
        base = pd.read_csv(csv, dtype=str,
                           keep_default_na=False).replace(
                               "", np.nan)
    worst = []
    for seed in seeds:
        df = base if base is not None else cohort(600, 6, seed)
        ids = sorted(df[group_by].astype(str).unique())
        rng = np.random.RandomState(seed + 1)
        rng.shuffle(ids)
        half = len(ids) // 2
        mem = df[df[group_by].astype(str).isin(set(ids[:half]))]
        non = df[df[group_by].astype(str).isin(set(ids[half:]))]
        g = LatentGen(seed=seed, k_blur=k_blur,
                      denoise=denoise).fit(mem, group_by)
        synth = g.generate(min(len(mem), 4000), seed=seed + 2)
        audit = membership_audit(
            ReconstructionAdversary(g),
            mem.to_dict("records")[:1500],
            non.to_dict("records")[:1500],
            synthetic=synth.to_dict("records"))
        print("seed {:>2}: nn {:.3f}  reconstruction {:.3f}  "
              "worst {:.3f}  {}".format(
                  seed,
                  audit.get("nearest_neighbor", {}).get(
                      "auc", 0.5),
                  audit["likelihood"]["auc"],
                  audit["worst_auc"], audit["verdict"]))
        worst.append(audit["worst_auc"])
    # THE POSITIVE CONTROL - a "generator" that republishes its
    # members must FAIL, or the numbers above are decoration.
    df = base if base is not None else cohort(600, 6, seeds[0])
    ids = sorted(df[group_by].astype(str).unique())
    rng = np.random.RandomState(seeds[0] + 1)
    rng.shuffle(ids)
    half = len(ids) // 2
    mem = df[df[group_by].astype(str).isin(set(ids[:half]))]
    non = df[df[group_by].astype(str).isin(set(ids[half:]))]
    g = LatentGen(seed=seeds[0], k_blur=k_blur,
                  denoise=denoise).fit(mem, group_by)
    leak = membership_audit(
        ReconstructionAdversary(g),
        mem.to_dict("records")[:1500],
        non.to_dict("records")[:1500],
        synthetic=mem.to_dict("records")[:2000])
    print()
    # THE CONTROL MUST REPORT THE ARM IT CONTROLS. Feeding the
    # members back as "synthetic" can only move the adversary
    # that READS synthetic rows - the nearest-neighbor one. The
    # reconstruction adversary scores through the weights and is
    # blind to the output, so `worst_auc` reported the
    # reconstruction number for both arms and the control read
    # PASS the moment denoising lowered it: a control that
    # stopped controlling while still printing a verdict.
    _ctl_nn = (leak.get("nearest_neighbor") or {}).get("auc", 0.5)
    print("republish control (the arm it actually controls - "
          "nearest neighbor on the OUTPUT): {:.3f} {} - a "
          "verbatim republish must be caught here, or the "
          "output-surface numbers above mean nothing".format(
              _ctl_nn, "CAUGHT" if _ctl_nn >= 0.70 else
              "NOT CAUGHT"))
    print("latent path worst over {} seeds: mean {:.3f}, max "
          "{:.3f} - read beside the blueprint's recorded 0.52, "
          "same cohort fixture, same bands.".format(
              len(seeds), float(np.mean(worst)),
              float(np.max(worst))))
    return worst, leak


def run_support(argv):
    """THE DIRECT READ OF THE K-AWARE CONTRACT: is the synthetic
    output CLOSE to the real records that many patients stand
    behind, and FAR from the ones fewer than k stand behind?

    For every real training record, the distance to its nearest
    synthetic row - split by how many distinct patients support
    that record's region. Dense rows want a SMALL distance (that
    is fidelity); rare rows want a LARGE one (that is privacy).
    Both arms - blur on and blur off - run in ONE command,
    because a comparison assembled by eye from two terminals is
    not a comparison.

    This is a different surface from the membership attack on the
    WEIGHTS: that adversary scores records straight through the
    trained network and never reads the output at all, so no
    sampling-time defense can move it. Fixing that one means
    changing TRAINING, and this measurement does not claim to."""
    import argparse
    from sklearn.neighbors import NearestNeighbors
    ap = argparse.ArgumentParser(
        prog="latent_challenger.py support")
    ap.add_argument("csv")
    ap.add_argument("--group-by", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print("support: --group-by {} --seed {}".format(
        a.group_by, a.seed))
    df = pd.read_csv(a.csv, dtype=str,
                     keep_default_na=False).replace("", np.nan)
    tr, _ho = _split_patients(df, a.group_by, a.seed)
    print("{} rows x {} columns, {} patients (training half: "
          "{} rows)".format(len(df), df.shape[1],
                            df[a.group_by].nunique(), len(tr)))
    rows = []
    for blur in (False, True):
        g = LatentGen(seed=a.seed, k_blur=blur).fit(tr,
                                                    a.group_by)
        gen = g.generate(len(tr), seed=a.seed + 1000)
        Xtr = g._Xtrain
        sup = g.support_of_rows(Xtr)
        rare = sup < g.k
        if not rare.any():
            print("  no sub-k regions in this frame - the "
                  "measurement has nothing to measure, stated "
                  "rather than reported as a pass")
            return 0
        nn = NearestNeighbors(n_neighbors=1).fit(
            g._re_encode_frame(gen))
        d = nn.kneighbors(Xtr)[0].ravel()
        dense_d = float(np.median(d[~rare]))
        rare_d = float(np.median(d[rare]))
        br = g.blur_report or {}
        rows.append((blur, dense_d, rare_d, br))
        print("  k-blur {:<3}: dense rows median distance {:.3f} "
              "| RARE rows {:.3f} | ratio {:.2f}x{}".format(
                  "ON" if blur else "off", dense_d, rare_d,
                  rare_d / (dense_d or 1e-9),
                  "  ({:.1%} of draws blurred, {} snapped)".format(
                      br["blurred"] / max(br["rows"], 1),
                      br["snapped"]) if br else ""))
    print("  {} of {} training rows sit in sub-k regions "
          "({:.1%})".format(int(rare.sum()), len(rare),
                            rare.mean()))
    off, on = rows[0], rows[1]
    d_ratio = (on[2] / (on[1] or 1e-9)) - (off[2] / (off[1] or 1e-9))
    print()
    print("VERDICT: the blur moved the rare-to-dense distance "
          "ratio by {:+.2f}x and the dense (average-fidelity) "
          "distance by {:+.1%}. The contract wants the first "
          "POSITIVE and the second near zero.".format(
              d_ratio, (on[1] - off[1]) / (off[1] or 1e-9)))
    return 0


def write_run_artifacts(src, gen, group_by, outdir,
                        time_col=None, source_name="(source)",
                        reconstruction=None, gen_engine=None,
                        k=K):
    """Make the neural output READABLE BY THE REST OF THE BENCH.

    The Verdict station and the Dashboard both key off artifacts
    the rules pipeline writes - blueprint.json, fidelity.json,
    generated.csv - so without them the neural engine's output
    could only be read by its own page. It is the same question
    ('how close is this to the real data?') and it should reach
    the same instruments.

    The blueprint written here describes the SOURCE and publishes
    no claims: it carries the k-screened marginals every
    measurement needs and asserts nothing about relationships,
    because this engine publishes no curves. Criteria that judge
    published claims therefore report as not-measured rather than
    passing on an absence, which is the honest reading."""
    import json
    from synthkit import blueprint as B
    from synthkit import pipeline as P
    from synthkit.discover import discover as _disc
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gen.to_csv(outdir / "generated.csv", index=False)
    # THE RELATIONSHIPS BELONG TO THE SOURCE, NOT TO AN ENGINE.
    # Without them `compare` has no pairs to judge and the gate
    # refuses to read - correctly, since a gate with nothing to
    # measure is not a pass. Discovery runs on the REAL data and
    # asks what it contains; whether a given output kept those
    # relationships is then a fair question to put to any
    # generator, this one included.
    print("  measuring what the real data contains (this is the "
          "slow part) ...")
    cat = _disc(src, group_by=group_by)
    # the CHOSEN k travels on the blueprint, and every page that
    # says "fewer than N patients" reads it from there - one
    # number, stated where it binds
    bp = B.build(src, cat, group_by=group_by, k=k)
    bp["generator"] = "neural"
    (outdir / "blueprint.json").write_text(
        json.dumps(bp, indent=1), encoding="utf-8")
    fid = P.compare(src, gen, bp, group_by, time_col)
    fid["generator"] = "neural"
    # AN INVERSION THAT SURVIVES ITS FIXTURES EARNS A DIAGNOSTIC
    # WHERE THE DATA IS. `procedure_count <- MAP_invasive`
    # refused to reproduce on four constructions and then
    # survived the reader fix on the real extract (-0.31 ->
    # +0.23) - so instead of a fifth guess, every INVERTED pair
    # whose columns exist raw in both frames is DECOMPOSED here
    # and the pieces travel with the run: the overall sign is
    # the sum of a conditional-value part (both present) and a
    # presence-coupling part, and which piece flipped names the
    # mechanism. A pair that inverts because the parent's
    # conditional values lost their signal is a sparse-decode
    # fault; one that inverts because presence landed on the
    # wrong rows is a placement fault. One block decides,
    # from the run directory alone.
    # DID THE QUOTAS LAND? The conditional-quota tier reads
    # exact on every fixture and moved ONE pair on the real
    # extract - so the run now measures its own enforcement:
    # for each adopted token, the corr of (token, its top
    # conditioner) on both tables. If these land while the
    # gate's close criterion does not move, the drift list's
    # pairs are NOT the adopted pairs and the adoption gates
    # are the question; if these do NOT land, the enforcement
    # is broken at extract scale. One block decides, from the
    # run directory.
    tq = getattr(gen_engine, "_token_implications", None) \
        if gen_engine is not None else None
    if tq:
        from synthkit import sets as _S2
        landing = []
        for cb, b_, conds in tq[:40]:
            if not conds or cb not in src.columns:
                continue
            ca, a_, q = conds[0]
            if ca not in src.columns:
                continue
            sep_b = _S2._separator(
                src[cb].dropna().astype(str).str.strip()
                .pipe(lambda t: t[t.str.len() > 0]))
            sep_a = _S2._separator(
                src[ca].dropna().astype(str).str.strip()
                .pipe(lambda t: t[t.str.len() > 0]))
            if not sep_b or not sep_a:
                continue

            def _corr(fr):
                x = _S2.has_token(fr[ca], sep_a, a_).fillna(0)
                y = _S2.has_token(fr[cb], sep_b, b_).fillna(0)
                return (float(x.corr(y, method="spearman"))
                        if x.std() > 0 and y.std() > 0 else None)
            landing.append({
                "token": "{}__has__{}".format(cb, b_),
                "conditioner": "{}__has__{}".format(ca, a_),
                "q": round(q, 3),
                "source": _corr(src), "generated": _corr(gen)})
        if landing:
            fid["quota_landing"] = landing
            miss = [d for d in landing
                    if d["source"] is not None
                    and d["generated"] is not None
                    and abs(d["source"] - d["generated"]) > 0.15]
            print("  quota landing: {} adopted pairs measured, "
                  "{} off by more than 0.15{}".format(
                      len(landing), len(miss),
                      " - " + "; ".join(
                          "{}~{} {:+.2f}->{:+.2f}".format(
                              d["token"][-14:],
                              d["conditioner"][-14:],
                              d["source"], d["generated"])
                          for d in miss[:4]) if miss else ""))
    inv_rows = (fid.get("relationships") or {}).get(
        "inverted") or []
    diags = []
    for r in inv_rows:
        ch, pa = str(r.get("child")), str(r.get("parent"))
        if ch not in src.columns or pa not in src.columns \
                or ch not in gen.columns or pa not in gen.columns:
            diags.append({"child": ch, "parent": pa,
                          "note": "scaffolding pair - measure "
                                  "from the expanded frame"})
            continue

        def _dz(fr):
            cv = pd.to_numeric(fr[ch], errors="coerce")
            pv = pd.to_numeric(fr[pa], errors="coerce")
            both = cv.notna() & pv.notna()
            cond = (float(cv[both].corr(pv[both],
                                        method="spearman"))
                    if int(both.sum()) >= 30 else None)
            pres = pv.notna().astype(float)
            coup = (float(cv.corr(pres, method="spearman"))
                    if pres.nunique() > 1 else None)
            return {"conditional_value_corr": cond,
                    "presence_coupling_corr": coup,
                    "parent_present_share":
                        round(float(pres.mean()), 4),
                    "both_present_rows": int(both.sum())}
        d = {"child": ch, "parent": pa,
             "source": _dz(src), "generated": _dz(gen)}
        diags.append(d)
        sd_, gd_ = d["source"], d["generated"]
        print("  INVERTED diagnostic {} <- {}:".format(ch, pa))
        print("    conditional value corr  src {} gen {}".format(
            sd_["conditional_value_corr"],
            gd_["conditional_value_corr"]))
        print("    presence coupling       src {} gen {}".format(
            sd_["presence_coupling_corr"],
            gd_["presence_coupling_corr"]))
        print("    parent present share    src {} gen {}".format(
            sd_["parent_present_share"],
            gd_["parent_present_share"]))
    if diags:
        fid["inverted_diagnostics"] = diags
    # THE OVERFITTING NUMBER TRAVELS WITH THE RUN, so the Verdict
    # station and the sign-off page can state it rather than
    # asking the reader to go and find the terminal log.
    if reconstruction:
        fid["reconstruction"] = reconstruction
    fid["note"] = ("Produced by the neural engine. The "
                   "relationships judged here were discovered "
                   "from the SOURCE, not published by this "
                   "engine - it publishes no contract of its "
                   "own. The question being asked is whether "
                   "its output kept what the real data "
                   "contains.")
    (outdir / "fidelity.json").write_text(
        json.dumps(fid, indent=1), encoding="utf-8")
    # THE CATALOGUE TOO - the Dashboard draws its pattern cards
    # from it, and without the file that whole section vanishes
    # with no explanation. Discovery already produced it.
    (outdir / "catalogue.json").write_text(
        json.dumps(cat, indent=1), encoding="utf-8")
    # PROVENANCE IN THE SHAPE THE OTHER PAGES READ. `source` is a
    # DICT with rows_read and patients, and `build` is a dict -
    # writing a bare string here made the sign-off page die with
    # `'str' object has no attribute 'get'`, which is the
    # contract-shape lesson this file already records about
    # provenance once before.
    try:
        from synthkit.build_id import build_id as _bi
        _b = _bi()
    except Exception:
        _b = {"id": "unknown"}
    _release_record = (None if gen_engine is None
                       or (float(getattr(gen_engine,
                                         "release_strength",
                                         1.0)) >= 1.0
                           and not getattr(gen_engine,
                                           "suppress_pairs",
                                           None)
                           and not getattr(gen_engine,
                                           "poison_pairs",
                                           None))
                       else _measure_release(
                           src, gen, gen_engine, group_by))
    (outdir / "provenance.json").write_text(json.dumps({
        "generator": "neural (autoencoder)",
        "build": _b,
        "blueprint_version": bp.get("blueprint_version"),
        "source": {
            "name": str(source_name),
            "rows_read": int(len(src)),
            "columns_read": int(src.shape[1]),
            "patients": (int(src[group_by].nunique())
                         if group_by in src.columns else None),
        },
        "settings": {"group_by": group_by,
                     "engine": "neural"},
        # THE GOVERNANCE CONTRACT: what this run learned and
        # under what rules, as a RECORD rather than scrollback.
        # Every field is read off the fitted engine itself, so
        # the contract cannot say something the run did not do.
        "contract": (None if gen_engine is None else {
            "k": int(k),
            "k_blur": bool(getattr(gen_engine, "k_blur",
                                   False)),
            "denoise_sd": float(getattr(gen_engine, "denoise",
                                        0.0)),
            "token_cap": int(gen_engine._cap_tok),
            "level_cap": int(gen_engine._cap_lv),
            "split": {"train_share": 0.85,
                      "holdout_share": 0.15},
            "tiers": {
                "size_identities": len(getattr(
                    gen_engine, "_size_identities", []) or []),
                "linear_identities": len(getattr(
                    gen_engine, "_linear_identities", [])
                    or []),
                "near_identities": len(getattr(
                    gen_engine, "_near_identities", []) or []),
                "sparse_couplings": len(getattr(
                    gen_engine, "_sparse_couplings", [])
                    or []),
                "conditional_quotas": len(getattr(
                    gen_engine, "_token_implications", [])
                    or []),
                "clusters_adopted": [
                    {"column": cb_, "members": len(mem_),
                     "patterns": len(pats_)}
                    for cb_, mem_, _a_, pats_ in getattr(
                        gen_engine, "_token_clusters", [])
                    or []],
                "clusters_rejected": list(getattr(
                    gen_engine, "_rejected_clusters", [])
                    or []),
            },
            # THE RELEASE DIALS, requested AND achieved -
            # measured from this very output, never assumed. A
            # dial a compliance reader cannot verify on the
            # page is a knob, not a control (the --time-col
            # lesson). strength_floor is the combinatorial zero:
            # the correlation ANY file of these shapes carries,
            # below which only arithmetic that MUST hold
            # survives.
            "release": _release_record,
            "posture": POSTURE,
        }),
    }, indent=1), encoding="utf-8")
    return fid


def _measure_release(src, gen, gen_engine, group_by):
    """Achieved dilution and suppression, measured from the
    output. Mean |rank-corr ratio| over pairs strong in source,
    and each suppressed pair's achieved correlation."""
    import numpy as _np
    req = float(getattr(gen_engine, "release_strength", 1.0))
    supp = list(getattr(gen_engine, "suppress_pairs", []) or [])
    cols = [c for c in src.columns
            if c != group_by and c in gen.columns]
    num = [c for c in cols
           if pd.to_numeric(src[c], errors="coerce").notna().mean()
           > 0.5]
    ratios = []
    for i in range(len(num)):
        for j in range(i + 1, len(num)):
            a = pd.to_numeric(src[num[i]], errors="coerce")
            b = pd.to_numeric(src[num[j]], errors="coerce")
            ga = pd.to_numeric(gen[num[i]], errors="coerce")
            gb = pd.to_numeric(gen[num[j]], errors="coerce")
            m = a.notna() & b.notna()
            gm = ga.notna() & gb.notna()
            if m.sum() < 30 or gm.sum() < 30:
                continue
            rs = float(a[m].corr(b[m], method="spearman"))
            if abs(rs) < 0.2:
                continue
            rg = float(ga[gm].corr(gb[gm], method="spearman"))
            if _np.isfinite(rs) and _np.isfinite(rg) and rs:
                ratios.append(rg / rs)
    achieved = (float(_np.mean(ratios)) if ratios else None)
    sup_rec = []
    for pa, pb in supp:
        if pa in gen.columns and pb in gen.columns:
            ga = pd.to_numeric(gen[pa], errors="coerce")
            gb = pd.to_numeric(gen[pb], errors="coerce")
            m = ga.notna() & gb.notna()
            r = (float(ga[m].corr(gb[m])) if m.sum() > 2
                 else None)
            sup_rec.append({"pair": "{}~{}".format(pa, pb),
                            "achieved_corr": r})
    auto = list(getattr(gen_engine, "_auto_suppressed", [])
                or [])
    return {"strength_requested": req,
            "strength_achieved": achieved,
            "suppressed": sup_rec,
            "auto_suppressed_top": [
                {"pair": "{}~{}".format(ca, cb),
                 "source_corr": rr}
                for ca, cb, child, rr in auto],
            "poisoned": list(getattr(
                gen_engine, "_poison_record", []) or []),
            "note": ("strength_achieved is the mean of generated "
                     "vs source pair-correlation ratios over "
                     "pairs the source holds strongly; marginals "
                     "are exact regardless of the dial. What "
                     "survives at strength 0 is arithmetic that "
                     "must hold for the file to be valid, not "
                     "institutional pattern.")}


def run_csv(argv):
    """The real-extract read, for the data machine. Echoes every
    setting it received - a missing flag must be visible."""
    global MAX_TOKENS, MAX_LEVELS
    import argparse
    ap = argparse.ArgumentParser(prog="latent_challenger.py csv")
    ap.add_argument("csv")
    ap.add_argument("--group-by", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=None,
                    help="directory for generated.csv per seed - "
                         "REAL-DERIVED output stays on this "
                         "machine")
    ap.add_argument("--shares", default=None,
                    help="CHILD=PARENT1,PARENT2 - SHAP driver "
                         "shares on a named triple, source vs "
                         "generated")
    ap.add_argument("--no-k-blur", action="store_true",
                    help="disable the k-aware blur - for "
                         "MEASURING what it costs and buys, "
                         "never for a release")
    ap.add_argument("--k", type=int, default=K,
                    help="the k-anonymity floor: no published "
                         "pattern may rest on fewer than k "
                         "distinct patients. {} is the minimum "
                         "and the default; RAISING it trades "
                         "fidelity for a stronger guarantee, "
                         "and the trade is reported, not "
                         "silent".format(K))
    # THE CAPS ARE A MEMORY BUDGET, NOT A PRIVACY RULE, so they are
    # reachable. Measured on an 800-patient tidy fixture: 4,710
    # condition tokens cleared the k floor and SIXTY were modeled,
    # while the encoded frame sat at 437 dimensions against the 4,326
    # that once died allocating 1.5 GiB. A cap that throws away
    # publishable signal with that much headroom is worth sweeping.
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                    help="set-token dimensions per column (default "
                         "{}): a MEMORY budget, not a privacy rule - "
                         "the k floor still decides what may be "
                         "published at all".format(MAX_TOKENS))
    ap.add_argument("--max-levels", type=int, default=MAX_LEVELS,
                    help="categorical level dimensions per column "
                         "(default {})".format(MAX_LEVELS))
    ap.add_argument("--denoise", type=float, default=0.3,
                    help="train as a DENOISING autoencoder with "
                         "this input-noise sd - the defense on "
                         "the weights surface")
    ap.add_argument("--release-strength", type=float,
                    default=1.0,
                    help="0..1 release dial: how much LEARNED "
                         "relationship strength survives into a "
                         "shared file. 1.0 is full fidelity; "
                         "lower dilutes patterns toward "
                         "independence while marginals stay "
                         "EXACT. For value-extraction control "
                         "when sharing - the achieved dilution "
                         "is measured and recorded, not assumed")
    ap.add_argument("--suppress", action="append", default=[],
                    help="A~B: a relationship that must NOT "
                         "exist in the output - the named child "
                         "is fully re-dealt, breaking ALL its "
                         "relationships (collateral stated, "
                         "because one edge cannot be removed "
                         "without leaking back through its "
                         "neighbors). Repeatable")
    ap.add_argument("--poison", action="append", default=[],
                    help="A~B=RHO: force the A-B relationship to "
                         "a chosen, possibly FALSE strength "
                         "(e.g. invert +0.4 to -0.8, or plant "
                         "one that is not real). A vendor "
                         "training on it learns the lie; it "
                         "stays a valid test because the true "
                         "and planted values are both recorded - "
                         "we hold the answer key. Repeatable")
    ap.add_argument("--suppress-top", type=int, default=0,
                    help="the MINIMUM-SHAREABLE control: auto-"
                         "detect the K strongest relationships "
                         "and remove exactly those, keeping "
                         "everything else faithful. For a vendor "
                         "release where the most extractable "
                         "patterns must go but the clinically "
                         "ordinary ones may stay - compliance "
                         "need not name a pattern it has never "
                         "seen; the tool names them by strength "
                         "and records which it removed")
    a = ap.parse_args(argv)
    # The caps are module-level because the encoder reads them while
    # building the frame; a sweep sets them once, here, and the run
    # PRINTS what it used so a reading can never be attributed to the
    # wrong budget.
    if a.max_tokens != MAX_TOKENS or a.max_levels != MAX_LEVELS:
        print("caps: set tokens {} -> {}, levels {} -> {} "
              "(memory budget, not a privacy rule)".format(
                  MAX_TOKENS, a.max_tokens, MAX_LEVELS, a.max_levels))
    MAX_TOKENS, MAX_LEVELS = a.max_tokens, a.max_levels
    _supp = []
    for _sp in a.suppress:
        if "~" not in _sp:
            print("STOP: --suppress wants A~B, got {}".format(
                _sp), file=sys.stderr)
            return 2
        _a, _b = _sp.split("~", 1)
        _supp.append((_a.strip(), _b.strip()))
    if not (0.0 <= a.release_strength <= 1.0):
        print("STOP: --release-strength must be in 0..1",
              file=sys.stderr)
        return 2
    _poison = []
    for _pz in a.poison:
        if "~" not in _pz or "=" not in _pz:
            print("STOP: --poison wants A~B=RHO, got {}".format(
                _pz), file=sys.stderr)
            return 2
        _ab, _rho = _pz.rsplit("=", 1)
        _pa, _pb = _ab.split("~", 1)
        try:
            _r = float(_rho)
        except ValueError:
            print("STOP: --poison RHO not a number: {}".format(
                _rho), file=sys.stderr)
            return 2
        if not (-1.0 <= _r <= 1.0):
            print("STOP: --poison RHO must be in -1..1",
                  file=sys.stderr)
            return 2
        _poison.append((_pa.strip(), _pb.strip(), _r))
    # THE FLOOR IS A FLOOR. The team asked for a user-settable k
    # with clarity on what can and cannot move: HIGHER is a
    # policy choice the operator may make; LOWER is not offered,
    # because below 10 distinct patients a published pattern
    # starts describing individuals, which is the one thing this
    # tool exists to refuse.
    if a.k < K:
        print("STOP: --k {} is below the floor of {}. The floor "
              "exists because a pattern resting on fewer than "
              "{} distinct patients begins to describe "
              "individuals; k may be raised, never lowered."
              .format(a.k, K, K), file=sys.stderr)
        return 2
    seeds = [int(x) for x in a.seeds.split(",")]
    print("csv: --group-by {} --seeds {} --out {} --shares {} "
          "--k {}".format(a.group_by, a.seeds, a.out or "(none)",
                          a.shares or "(none)", a.k))
    df = _read_source(a.csv)
    print("read {}: {} rows x {} columns, {} patients".format(
        a.csv, len(df), df.shape[1],
        df[a.group_by].nunique()))
    print("k-aware blur: {}".format(
        "OFF (measurement only)" if a.no_k_blur else "on"))
    for seed in seeds:
        tr, ho = _split_patients(df, a.group_by, seed)
        g = LatentGen(seed=seed, k=a.k,
                      k_blur=not a.no_k_blur,
                      denoise=a.denoise,
                      release_strength=a.release_strength,
                      suppress_pairs=_supp,
                      suppress_top=a.suppress_top,
                      poison_pairs=_poison).fit(
                          tr, a.group_by)
        # THE DELIVERABLE IS THE SIZE OF THE SOURCE, NOT OF THE
        # TRAINING SPLIT. Holding 15% of patients back is right -
        # the nearest-neighbor tripwire needs real people the
        # network never saw - and sizing the OUTPUT at 85% is a
        # different decision that was never made on purpose. On
        # the real extract it put "55,428 rows / 800 patients
        # original, 47,163 rows / 687 patients synthetic" at the
        # top of the dashboard, which reads as the generator
        # losing an eighth of the cohort. It also made the duel
        # unfair: the rules engine wrote a full-size file and
        # this one wrote 85% of it, and they were scored against
        # each other.
        gen = g.generate(
            len(df), seed=seed + 1000,
            n_patients=int(df[a.group_by].nunique()))
        cols = [c for c in df.columns if c != a.group_by]
        sign, close, inv, n = _pair_score(
            _pairs(tr, cols), _pairs(gen, cols))
        nn = g.nn_ratio(gen, g._re_encode_frame(ho))
        # THE HOLDOUT ANSWERS THE OVERFITTING QUESTION TOO, and
        # for as long as the split existed nothing asked it: the
        # only use of `ho` was a distance ratio on the GENERATED
        # file, which never puts a held-out record through the
        # network at all.
        rg = g.reconstruction_gap(ho)
        extra = ""
        if a.shares:
            child, ps = a.shares.split("=")
            parents = ps.split(",")
            # BOTH frames get the same numeric coercion - the
            # first cut coerced the source and handed the
            # generated frame over raw, where a folded __other__
            # string poisons the column and the shares read 0/0.
            def _numframe(fr):
                return fr.assign(**{
                    c: pd.to_numeric(fr[c], errors="coerce")
                    for c in [child] + parents}).dropna(
                        subset=[child] + parents)
            ss, _ = _shares(_numframe(tr), child, parents)
            gs, _ = _shares(_numframe(gen), child, parents)
            extra = " | shares src " + "/".join(
                "{:.0%}".format(x) for x in ss) + " gen " +                 "/".join("{:.0%}".format(x) for x in gs)
        br = g.blur_report
        blurb = ""
        if br:
            blurb = " | k-blur {:.1%} of rows ({} snapped)".format(
                br["blurred"] / max(br["rows"], 1), br["snapped"])
        print("seed {:>2}: sign {}/{} close {}/{} INVERTED {} | "
              "nn-ratio {:.2f}{}{}".format(seed, sign, n, close,
                                           n, inv, nn, extra,
                                           blurb))
        if rg:
            print("  reconstruction on {} held-out rows (people "
                  "the network never saw) {:.5f} against {:.5f} "
                  "on the {} it trained on - ratio {:.2f}. Near "
                  "1.0 is generalizing; a memorizing network "
                  "rebuilds its own rows far better and climbs. "
                  "On a frame with nothing to learn, a "
                  "wide-bottleneck undefended net reads 9.7 "
                  "where this configuration reads 1.8."
                  .format(rg["rows_holdout"], rg["holdout"],
                          rg["train"], rg["rows_train"],
                          rg["ratio"]))
        if a.out:
            outd = Path(a.out)
            outd.mkdir(parents=True, exist_ok=True)
            gen.to_csv(outd / "latent_gen_seed{}.csv".format(
                seed), index=False)
            print("  wrote {}".format(
                outd / "latent_gen_seed{}.csv".format(seed)))
            # ...and the artifacts the rest of the bench reads,
            # so the Verdict station and the Dashboard work on
            # this output too. Written for the FIRST seed only -
            # a run directory describes one generated file.
            if seed == seeds[0]:
                try:
                    # THE FULL SOURCE, NOT THE TRAINING
                    # SPLIT. The relationships belong to the
                    # data the operator pointed at; measuring
                    # them on 85% of it while the Dashboard's
                    # own header counts 100% is two halves
                    # disagreeing about the denominator.
                    write_run_artifacts(
                        df, gen, a.group_by, outd,
                        source_name=a.csv,
                        reconstruction=rg, gen_engine=g,
                        k=a.k)
                    print("  wrote blueprint.json, fidelity.json "
                          "and generated.csv - the Verdict "
                          "station and the Dashboard can read "
                          "this directory now")
                except Exception as e:
                    print("  could NOT write the bench "
                          "artifacts: {} - the review page still "
                          "works, the other stations will "
                          "refuse".format(e))
    print(POSTURE)


def between_share(df, col, group_by):
    """The share of a column's variance that lives BETWEEN
    patients rather than within one patient's own visits - the
    single number that says whether a longitudinal table has
    people in it. Near 1.0: the column is a property of the
    person (year of birth). Near 0: it is a property of the
    visit and re-rolls every time. A generator with no patient
    structure returns ~0 for EVERY column, because its rows
    belong to nobody."""
    if group_by not in df.columns:
        return float("nan")
    v = pd.to_numeric(df[col], errors="coerce")
    d = pd.DataFrame({"v": v,
                      "g": df[group_by].astype(str)}).dropna()
    if d["g"].nunique() < 5 or len(d) < 50:
        return float("nan")
    tot = float(d["v"].var())
    if not tot or not np.isfinite(tot):
        return float("nan")
    within = float(d.groupby("g")["v"].var().mean())
    if not np.isfinite(within):
        return float("nan")
    return float(1.0 - within / tot)


def _mine_interactions(df, cols, group_by=None, top_n=8,
                       seed=0, max_children=40):
    """MINE the strongest two-variable interactions from the
    SOURCE, so both engines are judged on what the real data
    actually contains rather than on a pair somebody named. Parent
    screening goes through model importance, NOT rank correlation
    - an XOR's parents have Spearman near zero with their child,
    which is precisely why they matter. The gain measured is the
    product term's R2 beyond the two mains: a screen for TWO-way
    interactions, said plainly; higher orders are not mined."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    num = {}
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().mean() >= 0.5 and v.nunique() >= 8:
            num[c] = v
    found = []
    children = list(num)[:max_children]
    for child in children:
        feats = [c for c in num if c != child]
        if len(feats) < 2:
            continue
        d = pd.DataFrame({c: num[c] for c in feats + [child]}
                         ).dropna(subset=[child])
        if len(d) < 200:
            continue
        X = d[feats].to_numpy()
        y = d[child].to_numpy()
        m = HistGradientBoostingRegressor(
            max_iter=60, random_state=seed,
            early_stopping=False).fit(X, y)
        sub = min(len(d), 2000)
        imp = permutation_importance(
            m, X[:sub], y[:sub], n_repeats=3,
            random_state=seed).importances_mean
        top = [feats[i] for i in np.argsort(-imp)[:4]]
        for i, a in enumerate(top):
            for b_ in top[i + 1:]:
                g = _interaction_r2(d, child, a, b_)
                if pd.notna(g) and g > 0.005:
                    found.append((float(g), child, a, b_))
    found.sort(reverse=True)
    return found[:top_n]


def run_compare(argv):
    """THE HEAD-TO-HEAD, one command, one table. The blueprint
    run's generated.csv and the latent path's draws are measured
    against the SAME source frame with the SAME metrics - a
    comparison assembled by eye from two terminals is not a
    comparison. Frames are restricted to the columns all three
    share, and what was dropped is SAID."""
    import argparse
    ap = argparse.ArgumentParser(
        prog="latent_challenger.py compare")
    ap.add_argument("csv")
    ap.add_argument("--group-by", required=True)
    ap.add_argument("--blueprint-run", required=True,
                    help="a finished fit run directory holding "
                         "generated.csv")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--shares", default=None, action="append",
                    help="CHILD=P1,P2 - repeatable")
    ap.add_argument("--interactions", type=int, default=8,
                    help="mine the top-N two-way interactions "
                         "from the SOURCE and measure both "
                         "engines on them; 0 disables")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(",")]
    print("compare: --group-by {} --blueprint-run {} --seeds {} "
          "--shares {} --out {}".format(
              a.group_by, a.blueprint_run, a.seeds,
              a.shares or "(none)", a.out or "(none)"))
    bp_csv = Path(a.blueprint_run) / "generated.csv"
    if not bp_csv.exists():
        print("STOPPED: no generated.csv in {} - point "
              "--blueprint-run at a finished `synthkit fit "
              "--generate` run".format(a.blueprint_run),
              file=sys.stderr)
        return 2
    src = _read_source(a.csv)
    bpg = _read_source(bp_csv)
    common = [c for c in src.columns
              if c in bpg.columns or c == a.group_by]
    dropped = [c for c in src.columns if c not in common]
    if dropped:
        print("columns not in the blueprint output, dropped "
              "from ALL sides so the metrics match: {}".format(
                  ", ".join(dropped[:8])
                  + (", ..." if len(dropped) > 8 else "")))
    src = src[common]
    cols = [c for c in common if c != a.group_by]

    def score(gen, label, g_for_nn=None, ho=None):
        sign, close, inv, n = _pair_score(
            _pairs(src, cols), _pairs(gen, cols))
        extra = ""
        for spec in (a.shares or []):
            child, ps = spec.split("=")
            parents = ps.split(",")

            def nf(fr):
                return fr.assign(**{
                    c: pd.to_numeric(fr[c], errors="coerce")
                    for c in [child] + parents}).dropna(
                        subset=[child] + parents)
            try:
                gs, _ = _shares(nf(gen), child, parents)
                extra += " | {} shares ".format(child) + "/".join(
                    "{:.0%}".format(x) for x in gs)
            except Exception as e:
                extra += " | {} shares unmeasurable ({})".format(
                    child, e)
        nnr = ""
        if g_for_nn is not None:
            nnr = " | nn-ratio {:.2f}".format(
                g_for_nn.nn_ratio(gen,
                                  g_for_nn._re_encode_frame(ho)))
        print("  {:<18} sign {}/{} close {}/{} INVERTED {}{}{}"
              .format(label, sign, n, close, n, inv, nnr, extra))

    for spec in (a.shares or []):
        child, ps = spec.split("=")
        parents = ps.split(",")

        def nf0(fr):
            return fr.assign(**{
                c: pd.to_numeric(fr[c], errors="coerce")
                for c in [child] + parents}).dropna(
                    subset=[child] + parents)
        ss, _ = _shares(nf0(src), child, parents)
        print("  {:<18} {} shares {}".format(
            "source", child, "/".join("{:.0%}".format(x)
                                      for x in ss)))
    score(bpg, "blueprint")
    latent_gens = []
    for seed in seeds:
        tr, ho = _split_patients(src, a.group_by, seed)
        g = LatentGen(seed=seed).fit(tr, a.group_by)
        # THE DELIVERABLE IS THE SIZE OF THE SOURCE, NOT OF THE
        # TRAINING SPLIT. Holding 15% of patients back is right -
        # the nearest-neighbor tripwire needs real people the
        # network never saw - and sizing the OUTPUT at 85% is a
        # different decision that was never made on purpose. On
        # the real extract it put "55,428 rows / 800 patients
        # original, 47,163 rows / 687 patients synthetic" at the
        # top of the dashboard, which reads as the generator
        # losing an eighth of the cohort. It also made the duel
        # unfair: the rules engine wrote a full-size file and
        # this one wrote 85% of it, and they were scored against
        # each other.
        gen = g.generate(
            len(src), seed=seed + 1000,
            n_patients=int(src[a.group_by].nunique()))
        latent_gens.append((seed, gen))
        score(gen, "latent seed {}".format(seed),
              g_for_nn=g, ho=ho)
        if a.out:
            outd = Path(a.out)
            outd.mkdir(parents=True, exist_ok=True)
            gen.to_csv(outd / "latent_gen_seed{}.csv".format(
                seed), index=False)
            print("    wrote {}".format(
                outd / "latent_gen_seed{}.csv".format(seed)))
    if a.interactions:
        print()
        print("THE MINED INTERACTIONS - top {} two-way product "
              "gains found in the SOURCE (parents screened by "
              "model importance, so an XOR's parents are "
              "findable; higher orders not mined). gain = R2 of "
              "the product term beyond the two mains.".format(
                  a.interactions))
        mined = _mine_interactions(src, cols,
                                   top_n=a.interactions)
        if not mined:
            print("  none cleared the 0.005 gain floor - "
                  "stated, not silent.")
        for gv, child, pa, pb in mined:
            gb = _interaction_r2(bpg, child, pa, pb)
            row = "  {} <- {} x {}: src {:.3f} | blueprint "                   "{}".format(child, pa, pb, gv,
                              "{:.3f}".format(gb)
                              if pd.notna(gb) else "n/a")
            for seed, gen in latent_gens:
                gl = _interaction_r2(gen, child, pa, pb)
                row += " | latent s{} {}".format(
                    seed, "{:.3f}".format(gl)
                    if pd.notna(gl) else "n/a")
            print(row)
    print("Same source, same metrics, both engines. The latent "
          "path trains on records and has no dynamics; the "
          "blueprint publishes k-screened aggregates and passed "
          "its attack battery. Fidelity is only one column of "
          "that table.")
    return 0


def _planted_frame(seed):
    """The nonlinear core: the tidy fixture's planted xor, u-shape,
    threshold, 3-way and simpson columns - the kinds the operator's
    complaint is about - kept numeric and small."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            [sys.executable,
             str(ROOT / "scripts" / "make_tidy_fixture.py"),
             "-o", td, "--seed", str(seed)],
            check=True, capture_output=True, cwd=str(ROOT))
        df = pd.read_csv(next(Path(td).glob("*.csv")),
                         keep_default_na=False)
    keep = ["person_id"] + [c for c in df.columns
                            if c.startswith("planted_")
                            and "lag" not in c]
    return df[keep]


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "triangle"
    if which == "csv":
        # the return value IS the exit code - the k-floor
        # refusal returns 2, and a discarded return exits 0,
        # which told the caller a refused run succeeded
        sys.exit(run_csv(sys.argv[2:]))
    elif which == "support":
        sys.exit(run_support(sys.argv[2:]))
    elif which == "compare":
        sys.exit(run_compare(sys.argv[2:]))
    elif which == "attack":
        import argparse
        ap = argparse.ArgumentParser(
            prog="latent_challenger.py attack")
        ap.add_argument("seeds", nargs="*", type=int,
                        default=[0, 1, 2])
        ap.add_argument("--csv", default=None)
        ap.add_argument("--group-by", default="person_id")
        ap.add_argument("--no-k-blur", action="store_true")
        ap.add_argument("--denoise", type=float, default=0.3)
        # THE ATTACK MUST RUN THE CONFIGURATION THAT SHIPS. Without
        # these the battery fitted at the DEFAULT cap while the
        # deliverable was generated at 400, so the certified
        # configuration and the released one were different objects -
        # and width is exactly where an autoencoder's memorization
        # risk lives, so it is the one parameter a membership result
        # cannot be transferred across. The recorded pass was taken
        # at ~150-440 encoded dimensions; a 400-cap extract frame is
        # 1,123.
        ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
        ap.add_argument("--max-levels", type=int, default=MAX_LEVELS)
        aa = ap.parse_args(sys.argv[2:])
        # no `global` here: this block is module level, so the
        # assignment below rebinds the module name directly - and a
        # `global` statement after MAX_TOKENS has already been read
        # by add_argument is a SyntaxError, the same ordering trap
        # this file hit once before with a displaced docstring.
        if aa.max_tokens != MAX_TOKENS or aa.max_levels != MAX_LEVELS:
            print("caps: set tokens {} -> {}, levels {} -> {} "
                  "(the attack runs the shipped configuration)"
                  .format(MAX_TOKENS, aa.max_tokens,
                          MAX_LEVELS, aa.max_levels))
        MAX_TOKENS, MAX_LEVELS = aa.max_tokens, aa.max_levels
        run_attack(aa.seeds or [0, 1, 2], csv=aa.csv,
                   group_by=aa.group_by,
                   k_blur=not aa.no_k_blur,
                   denoise=aa.denoise)
    else:
        seeds = [int(x) for x in sys.argv[2:]] or DEFAULT_SEEDS
        run(which, seeds)
