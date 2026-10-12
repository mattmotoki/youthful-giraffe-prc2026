"""Stand re-occupation between LOBT and the take-off: the sharp screen for
'the aircraft actually left the stand at LOBT, not at the NM off-block'."""
import numpy as np, pandas as pd
import prc.stand_features as sf


def _runs(k):
    if len(k) == 0:
        return np.array([]), np.array([], int), np.array([], int)
    ch = np.flatnonzero(k[1:] != k[:-1]) + 1
    st = np.concatenate([[0], ch]); en = np.concatenate([ch, [len(k)]])
    return k[st], st, en


def reoccupation(mvt: pd.DataFrame, key: str = "key_raw") -> pd.DataFrame:
    """For each departure, arrivals that parked at its stand inside three windows.

    If another aircraft on-blocks at stand S at time X, this departure was gone by
    X. So an arrival inside `(LOBT, AOBT)` says the aircraft left the stand near
    LOBT rather than near the NM off-block: it is direct evidence for
    block-on-LOBT, and unlike the counting bound in exp-003 it needs no
    arrival/departure alternation assumption.
    """
    df = mvt if "key_raw" in mvt.columns else sf.add_stand_keys(mvt)
    is_dep = df["PHASE_mvt"].astype("string") == "DEP"
    dep = df.loc[is_dep]
    arr = df.loc[~is_dep]
    arr = arr[arr["BLOCK_TIME_UTC_mvt"].notna()]
    ns = lambda s: s.dt.as_unit("ns").astype("int64").to_numpy()

    a_key = arr[key].astype(object).fillna("?").astype(str).to_numpy()
    a_t = ns(arr["BLOCK_TIME_UTC_mvt"])
    o = np.argsort(a_key, kind="mergesort")
    a_key, a_t = a_key[o], a_t[o]
    spans = {k: (s, e) for k, s, e in zip(*_runs(a_key))}

    out = pd.DataFrame(index=dep.index, dtype="float64",
                       columns=["reocc_lobt_aobt", "reocc_lobt_mvt", "reocc_aobt_mvt",
                                "reocc_first_gap_lobt", "reocc_last_gap_mvt"])
    L, A, T = ns(dep["LOBT_flt"]), ns(dep["AOBT_3_flt"]), ns(dep["MVT_TIME_UTC_mvt"])
    nat = np.iinfo(np.int64).min
    dk = dep[key].astype(object).fillna("?").astype(str)
    for k, idx in dk.groupby(dk).indices.items():
        se = spans.get(k)
        if se is None:
            out.iloc[idx, :] = [0, 0, 0, np.nan, np.nan]
            continue
        s, e = se
        at = np.sort(a_t[s:e])
        l, a, t = L[idx], A[idx], T[idx]
        lo = np.searchsorted(at, np.where(l == nat, t, l), "right")
        hi_a = np.searchsorted(at, np.where(a == nat, t, a), "left")
        hi_t = np.searchsorted(at, t, "left")
        n_la = np.maximum(hi_a - lo, 0)
        n_lt = np.maximum(hi_t - lo, 0)
        n_at = np.maximum(hi_t - np.searchsorted(at, np.where(a == nat, t, a), "right"), 0)
        first = np.where(n_lt > 0, at[np.clip(lo, 0, len(at) - 1)], np.nan)
        last = np.where(n_lt > 0, at[np.clip(hi_t - 1, 0, len(at) - 1)], np.nan)
        blk = np.column_stack([
            n_la, n_lt, n_at,
            np.where(np.isnan(first), np.nan, (first - np.where(l == nat, t, l)) / 1e9),
            np.where(np.isnan(last), np.nan, (t - last) / 1e9)])
        out.iloc[idx, :] = blk
    return out
