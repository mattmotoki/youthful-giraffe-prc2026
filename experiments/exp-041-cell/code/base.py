"""The shared base of exp-041: the fold-side v10 panel plus the row attributes
the cell question needs, and the exp-039 measurement packet pointed at this
experiment's `analysis/`.

Nothing is re-implemented.  `exp-039-screens/code/v10.py` is the gated base
(it checks the uploaded v10's SHA-256, reproduces exp-036 arm 2's published
rotation and pins the fold vectors by their own bytes), `rowfeat.py` is the one
masking story (`prc.data.mask_like_ranking` blanks departure
`BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` before any feature is built) and
`packet.py` is the one contract writer.  `packet.EXP` is repointed at this
experiment so the per-date CSVs and the provenance JSON land here; the contract
itself is the same object exp-038 and exp-039 measured on.

Row attributes added here, all from the MASKED month frame and all legal
inputs under `docs/input-rule.md`:

    rwy   RUNWAY_mvt          op    AIRCRAFT_OPERATOR_flt   pier  STAND_mvt[:2]
    typ   AIRCRAFT_TYPE_mvt   wtc   WK_TBL_CAT_flt          seg   MARKET_SEGMENT_flt
    aobt  AOBT_3_flt          lobt  LOBT_flt                sched SCHED_TIME_UTC_mvt
    eobt  EOBT_1_flt          t     MVT_TIME_UTC_mvt (the take-off clock, in the panel)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
S39 = ROOT / "experiments/exp-039-screens/code"
sys.path.insert(0, str(S39))
import v10 as V        # noqa: E402
import rowfeat as R    # noqa: E402
import packet as PK    # noqa: E402
import donor as Dn     # noqa: E402

PK.EXP = EXP           # per-date CSVs and packet JSON land in exp-041/analysis

FOLDS = V.FOLDS
D = V.D

# ---- predeclared, before any number of this experiment was looked at --------
BIN = 600                       # the cell's UTC window, seconds (Q03/Q04's grid)
HOLD_AIRPORTS = ("EHAM", "EDDF")
ELIGIBLE_REGIME = "nm_bulk"
BLOCKS = 5                      # contiguous day blocks, shared across airports
SEED = 41
BOOT = 2000
CLIP = 7200.0                   # clock differences, seconds

ATTR_COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt",
             "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt",
             "WK_TBL_CAT_flt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
             "AOBT_3_flt", "LOBT_flt", "EOBT_1_flt",
             "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]


#: `donor.epoch` handles whatever datetime resolution the parquet carries; a
#: hand-rolled `/1e9` silently reads a microsecond column as 1970 (caught here
#: on the first smoke run).
_epoch = Dn.epoch


def attrs(month: str) -> pd.DataFrame:
    """Departure row attributes of one month, from the ranking-masked frame."""
    raw = D.load_month(month, columns=ATTR_COLS)
    masked, _ = D.mask_like_ranking(raw)
    del raw
    dep = masked[masked["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    del masked
    if dep["BLOCK_TIME_UTC_mvt"].notna().any() or dep[D.TARGET].notna().any():
        raise SystemExit(f"{month}: departure B/y survived the ranking mask")
    out = pd.DataFrame({
        "mid": dep["MVT_ID_mvt"].to_numpy("float64"),
        "adep": dep["ADEP_mvt"].astype("string").fillna("<na>").to_numpy(),
        "rwy": dep["RUNWAY_mvt"].astype("string").fillna("<na>").to_numpy(),
        "op": dep["AIRCRAFT_OPERATOR_flt"].astype("string").fillna("<na>").to_numpy(),
        "pier": dep["STAND_mvt"].astype("string").fillna("").str.slice(0, 2).to_numpy(),
        "typ": dep["AIRCRAFT_TYPE_mvt"].astype("string").fillna("<na>").to_numpy(),
        "wtc": dep["WK_TBL_CAT_flt"].astype("string").fillna("<na>").to_numpy(),
        "seg": dep["MARKET_SEGMENT_flt"].astype("string").fillna("<na>").to_numpy(),
        "t_attr": _epoch(dep["MVT_TIME_UTC_mvt"]),
        "sched": _epoch(dep["SCHED_TIME_UTC_mvt"]),
        "aobt": _epoch(dep["AOBT_3_flt"]),
        "lobt": _epoch(dep["LOBT_flt"]),
        "eobt": _epoch(dep["EOBT_1_flt"])})
    return out


def panel(fold: str, pan: pd.DataFrame) -> pd.DataFrame:
    """The fold's v10 panel with the row attributes joined and the cell key set."""
    a = attrs(R.month_of(fold)).set_index("mid")
    ids = pan["MVT_ID_mvt"].to_numpy("float64")
    a = a.reindex(ids)
    if a[["rwy", "op"]].isna().any().any():
        raise SystemExit(f"{fold}: a panel row has no attribute row in the month")
    m = pan.copy()
    for c in a.columns:
        m[c] = a[c].to_numpy()
    # the attribute frame's own take-off clock must be the panel's, row for row
    off = np.nanmax(np.abs(m["t_attr"].to_numpy("float64")
                           - m["t"].to_numpy("float64")))
    if not (off == 0):
        raise SystemExit(f"{fold}: the attribute take-off clock differs by {off} s")
    m = m.drop(columns=["t_attr"])
    m["bin"] = np.floor(m["t"].to_numpy("float64") / BIN).astype("int64")
    m["apt"] = m["airport"].astype(str).to_numpy()
    m["elig"] = ((m["regime"].to_numpy() == ELIGIBLE_REGIME)
                 & ~np.isin(m["apt"].to_numpy(), HOLD_AIRPORTS))
    m["day"] = V.day(m["t"].to_numpy("float64"))
    return m


#: The shipped current feature block's own categorical codes.  Sol round 1,
#: blocking finding 7: they are NOT per-month codes.  `exp-018-fresh` encodes
#: every category through one persistent append-only vocabulary
#: (`artifacts/exp-018-fresh/cache/vocab.json`, `code/dataset.py:_vocab_encode`),
#: so `c_rwy = 17` is the same runway in July and in January.  `shipped()`
#: asserts that on every load, and no fit here drops them.
SHIPPED_CATS = ("c_apt", "c_rwy", "c_stand", "c_standpfx", "c_actype", "c_wtc",
                "c_seg", "c_ftype", "c_frule", "c_oper", "c_ades", "c_adescty",
                "c_csgn")

#: The K2-enabled feature cache.  Sol round 1, blocking finding 3: v10 serves
#: `refit-bulk` at eight airports and `refit-bulk` is a K2-lineage member, whose
#: frame is built with `PRC_WINDOW=1` -- the 68-column transient block of
#: `experiments/exp-018-fresh/code/window_features.py`.  `assignments/cell.md`
#: step 2 says the same thing in words ("the shipped window block already carries
#: unordered counts AND the K2 transient window").  A base built on the flag-off
#: 89-column cache is therefore NOT matched to v10's own information, so every
#: fit in this experiment reads `<month>-w.parquet`, not `<month>.parquet`.
W_CACHE = ROOT / "artifacts/exp-018-fresh/cache"
LABEL_COLS = ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "TAXITIME_SEC", "y_true")
W_DROP = ("y", "month", "MVT_ID_mvt")      # `q04.W_DROP`, unchanged
_VOCAB: dict | None = None
#: One month at a time.  The K2-enabled block is 160 columns over ~190,000 rows,
#: so caching three months costs about 0.7 GB and this laptop is shared with
#: another worker; a fold reads its own month several times in a row, which is
#: what a one-entry cache is for.
_SHIPPED_CACHE: dict[str, pd.DataFrame] = {}


def vocab() -> dict:
    """The persistent category vocabulary the feature cache was encoded with."""
    global _VOCAB
    if _VOCAB is None:
        import json
        _VOCAB = json.loads((W_CACHE / "vocab.json").read_text())
    return _VOCAB


def shipped_frame(month: str) -> pd.DataFrame:
    """The K2-enabled shipped feature block of one month, keyed by movement id.

    Same guards as `q04.shipped_W` -- no departure label column may appear -- plus
    the cross-month check the categorical codes need: every code must index into
    the persistent vocabulary, which is what makes a cross-month fit legible.
    """
    if month in _SHIPPED_CACHE:
        return _SHIPPED_CACHE[month]
    f = W_CACHE / f"{month}-w.parquet"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} is missing; build it with PRC_WINDOW=1 and "
            "experiments/exp-018-fresh/code/features.py")
    w = pd.read_parquet(f)
    for c in LABEL_COLS:
        if c in w.columns:
            raise ValueError(f"{c} is present in the shipped feature block")
    V_ = vocab()
    for c in SHIPPED_CATS:
        if c in w.columns:
            mx = int(w[c].max())
            if mx >= len(V_[c]):
                raise SystemExit(f"{month}: {c} code {mx} is outside the "
                                 f"{len(V_[c])}-value shared vocabulary")
    w = w.set_index(w["MVT_ID_mvt"].to_numpy("float64"))
    keep = [c for c in w.columns
            if c not in W_DROP and pd.api.types.is_numeric_dtype(w[c])]
    out = w[keep].astype("float32")
    del w
    _SHIPPED_CACHE.clear()
    _SHIPPED_CACHE[month] = out
    return out


def shipped(month: str, ids: np.ndarray, drop_cats: bool = False
            ) -> tuple[np.ndarray, list[str]]:
    """The shipped current feature block for one month, on the panel's rows.

    exp-039's Sol round 2 (blocking finding b1) settled that a matched base for
    these screens is the SHIPPED block, not a hand-built substitute; Sol round 1
    here settled that the shipped block v10 reads is the K2-enabled one.  Built
    by `experiments/exp-018-fresh/code/features.py` from one month's frame alone,
    with departure `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` never read.

    `drop_cats` is kept for callers that want the numeric block alone; nothing in
    this experiment uses it, because the codes are shared across months.
    """
    w = shipped_frame(month)
    if drop_cats:
        w = w[[c for c in w.columns if c not in SHIPPED_CATS]]
    x = w.reindex(ids)
    if x.isna().all(axis=1).any():
        raise SystemExit(f"{month}: a panel row has no shipped feature row")
    return x.to_numpy("float32"), list(x.columns)


def cell_code(m: pd.DataFrame) -> tuple[np.ndarray, int]:
    """airport x UTC 10-minute bin x serving regime, as codes."""
    key = (m["apt"].to_numpy().astype(str) + "|"
           + m["regime"].astype(str).to_numpy() + "|"
           + m["bin"].to_numpy().astype(str))
    code, uniq = pd.factorize(key)
    return code, len(uniq)


def blocks_of(days: np.ndarray) -> np.ndarray:
    u = np.unique(days)
    idx = {d: b for b, chunk in enumerate(np.array_split(u, BLOCKS)) for d in chunk}
    return np.array([idx[d] for d in days])


def boot_ratio(num: np.ndarray, den: np.ndarray, days: np.ndarray, rng,
               boot: int = BOOT) -> tuple[float, float, float]:
    """Day-clustered bootstrap of `sum(num)/sum(den)`; days resampled with replacement.

    Every relation statistic here is a ratio of a pair sum to a pair count, so
    the interval must be on the ratio, not on a per-row mean.
    """
    codes, uniq = pd.factorize(pd.Series(days))
    k = len(uniq)
    n = np.bincount(codes, weights=num, minlength=k)
    d = np.bincount(codes, weights=den, minlength=k)
    if d.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    draws = rng.integers(0, k, size=(boot, k))
    ds = d[draws].sum(1)
    b = np.where(ds > 0, n[draws].sum(1) / np.maximum(ds, 1e-12), np.nan)
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return float(n.sum() / d.sum()), float(lo), float(hi)


def group_sums(code: np.ndarray, ncode: int, r: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """Per-code sum of `r` and count, gathered back onto the rows."""
    s = np.bincount(code, weights=r, minlength=ncode)
    c = np.bincount(code, minlength=ncode).astype("float64")
    return s[code], c[code]


def codes_of(*cols: np.ndarray) -> tuple[np.ndarray, int]:
    """Dense codes for a composite key.

    The key is built by joining the columns with a separator and factorizing the
    strings.  `pd.factorize` on a `MultiIndex` does NOT do this -- it collapsed a
    48,000-cell key to 48 codes on the first smoke run -- so the string form is
    the one used everywhere in this experiment, as it is in exp-039's Q03.
    """
    if len(cols) == 1:
        code, uniq = pd.factorize(np.asarray(cols[0]).astype(str))
        return code, len(uniq)
    key = np.asarray(cols[0]).astype(str)
    for c in cols[1:]:
        key = np.char.add(np.char.add(key, "|"), np.asarray(c).astype(str))
    code, uniq = pd.factorize(key)
    return code, len(uniq)


def write_oof(name: str, panel: dict, preds: dict) -> Path:
    """`COMMON.md`'s out-of-fold record AND the per-row contribution table.

    One parquet per arm, all three folds, in panel row order per fold:

        MVT_ID_mvt  fold  date  y_true  p_v10  y_pred  delta_sse

    so a later experiment can rebase on the arm without rerunning it, and every
    headline number can be recomputed row by row (Sol round 1, blocking finding
    9: the first version wrote date aggregates only).  `y_true`, `p_v10` and
    `y_pred` are float32 and the file is zstd, which is what keeps a
    per-arm-per-row record small enough to commit -- these are predictions in
    seconds, and float32 carries them to about 2e-5 s.
    """
    out = []
    for f in FOLDS:
        m = panel[f]
        y = m["y_true"].to_numpy("float64")
        b = m["p10"].to_numpy("float64")
        q = np.asarray(preds[f], "float64")
        out.append(pd.DataFrame({
            "MVT_ID_mvt": m["MVT_ID_mvt"].to_numpy("float64"),
            "fold": pd.Categorical([f] * len(m), categories=list(FOLDS)),
            "date": pd.Categorical(V.day(m["t"].to_numpy("float64"))),
            "y_true": y.astype("float32"),
            "p_v10": b.astype("float32"),
            "y_pred": q.astype("float32"),
            "delta_sse": ((y - b) ** 2 - (y - q) ** 2).astype("float32")}))
    p = EXP / "oof" / f"exp-041-{name}.parquet"
    pd.concat(out, ignore_index=True).to_parquet(p, index=False,
                                                 compression="zstd")
    return p


def oof_manifest() -> Path:
    """One committed index of the out-of-fold / row-contribution parquets."""
    import json
    d = EXP / "oof"
    out = []
    for f in sorted(d.glob("exp-041-*.parquet")):
        out.append({"file": f"experiments/{EXP.name}/oof/{f.name}",
                    "bytes": f.stat().st_size,
                    "rows": int(pd.read_parquet(f, columns=["MVT_ID_mvt"]).shape[0]),
                    "sha256": PK.sha_file(f)})
    p = d / "MANIFEST.json"
    p.write_text(json.dumps(
        {"note": "one file per measured arm, all three folds, panel row order: "
                 "MVT_ID_mvt, fold, date, y_true, p_v10, y_pred, delta_sse.  "
                 "These are both COMMON.md's out-of-fold record and the per-row "
                 "contribution table behind every headline number.",
         "files": out}, indent=2))
    return p


# ---- the measurement record: row contributions, provenance, board intervals ---
def code_state() -> dict:
    """SHA-256 of THIS experiment's executable files.

    `packet.code_state` hashes exp-039's code directory, because `PK.EXP` moves
    the outputs but not the module's own `HERE` (Sol round 1, blocking finding 9).
    Both dictionaries are written, under different keys.
    """
    return {p.name: PK.sha_file(p) for p in sorted(HERE.glob("*.py"))}


def write_packet(screen: str, rec: dict, pan: dict) -> Path:
    """`packet.write` plus this experiment's own code hashes and row manifest."""
    import json
    p = PK.write(screen, rec, pan)
    d = json.loads(p.read_text())
    d["provenance"]["code_sha256_exp039"] = d["provenance"].pop("code_sha256")
    d["provenance"]["code_sha256"] = code_state()
    mp = oof_manifest()
    #  The manifest indexes EVERY arm of the experiment, so a later screen's run
    #  changes it and the hash a earlier packet recorded no longer matches (Sol
    #  round 2, non-blocking).  Each packet therefore also carries the hashes of
    #  ITS OWN arms' files, which nothing outside this screen can move.
    own = {f.name: PK.sha_file(f)
           for f in sorted((EXP / "oof").glob(f"exp-041-{screen}-*.parquet"))}
    d["provenance"]["row_contributions_manifest"] = {
        "file": f"experiments/{EXP.name}/oof/MANIFEST.json",
        "sha256": PK.sha_file(mp),
        "sha256_note": "a snapshot: the manifest covers every arm of the "
                       "experiment, so a later screen's run changes it",
        "this_screen_oof_sha256": own,
        "note": "per-row contributions live in the out-of-fold parquets, one "
                "per arm; see base.write_oof"}
    d["provenance"]["shipped_feature_cache_sha256"] = {
        m: PK.sha_file(W_CACHE / f"{m}-w.parquet") for m in
        sorted({R.month_of(f) for f in FOLDS})}
    p.write_text(json.dumps(d, indent=2, default=str))
    return p


def board_boot(parts: dict, rng, lengths=(1, 2, 3)) -> dict:
    """A moving-block interval on the PRIMARY endpoint itself.

    The endpoint is `board = sqrt(sum_f share_f * MSE_f)` over the two ranking
    months (`v10.board`), so an interval on it cannot be read off a per-row
    interval: the days are resampled, each fold's MSE is rebuilt from its own
    resampled day sums, and the board is recomputed inside every draw.  Blocks
    are drawn INSIDE a fold and never across a month boundary (`packet`'s own
    round-2 correction).

    `parts[fold] = (days, sse_v10, sse_arm)` over that fold's FULL row set.
    """
    import numpy as _np
    SH = V.Sc.P.SHARES
    per = {}
    for f, (days, s0, s1) in parts.items():
        if f not in SH:
            continue
        codes, uniq = pd.factorize(pd.Series(days))
        order = _np.argsort(uniq.astype(str))
        rank = _np.empty(len(uniq), int)
        rank[order] = _np.arange(len(uniq))
        di = rank[codes]
        k = len(uniq)
        per[f] = (_np.bincount(di, weights=s0, minlength=k),
                  _np.bincount(di, weights=s1, minlength=k),
                  _np.bincount(di, minlength=k).astype("float64"))
    pt0 = float(_np.sqrt(sum(SH[f] * per[f][0].sum() / per[f][2].sum() for f in per)))
    pt1 = float(_np.sqrt(sum(SH[f] * per[f][1].sum() / per[f][2].sum() for f in per)))
    out = {"board_v10": round(pt0, 4), "board_arm": round(pt1, 4),
           "gain": round(pt0 - pt1, 4),
           "note": "shares " + ", ".join(f"{f} {SH[f]}" for f in sorted(per))
                   + "; blocks drawn inside each fold"}
    for L in lengths:
        acc0 = _np.zeros(BOOT); acc1 = _np.zeros(BOOT)
        for f, (n0, n1, c) in per.items():
            k = len(c)
            nb = int(_np.ceil(k / L))
            st = rng.integers(0, max(k - L + 1, 1), size=(BOOT, nb))
            idx = (st[:, :, None] + _np.arange(L)[None, None, :]).reshape(BOOT, -1)
            idx = _np.minimum(idx, k - 1)[:, :k]
            den = _np.maximum(c[idx].sum(1), 1.0)
            acc0 += SH[f] * n0[idx].sum(1) / den
            acc1 += SH[f] * n1[idx].sum(1) / den
        b = _np.sqrt(acc0) - _np.sqrt(acc1)
        lo, hi = _np.percentile(b, [2.5, 97.5])
        out[f"ci_block_{L}d"] = [round(float(lo), 4), round(float(hi), 4)]
    return out
