"""exp-168 unit 9: shared frame. The fold rows with the new base B (stack form e9771b94 by the stack's `compose`),
the label kinds of unit 5 and the label-free time differences of a row."""
import hashlib, os, sys
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, f"{ROOT}/experiments/exp-169-stack-plan/code")
from assembly import compose  # noqa: E402  (the stack's builder, read only)
BUNDLE = f"{ROOT}/experiments/exp-169-stack-plan/inputs/unit2-ready/bundle-fold.parquet"
T0 = pd.Timestamp("2025-01-01", tz="UTC")
secs = lambda a, c: (a - c).dt.total_seconds()


def frame(path):
    assert hashlib.sha256(open(BUNDLE, "rb").read()).hexdigest() == "73274b9bb2ef13acc69a5d999bb862903003d2e69f56852bf50abaeb79c7f28f"
    b = pd.read_parquet(BUNDLE).set_index("MVT_ID_mvt"); f = pd.read_parquet(path)
    f["t"] = secs(f.MVT_TIME_UTC_mvt, T0); f["day"] = (f.t // 86400).astype(int)
    f = f.sort_values(["airport", "day", "t", "MVT_ID_mvt"], kind="stable").reset_index(drop=True); bb = b.reindex(f.MVT_ID_mvt).reset_index()
    f["B"] = compose(bb, route="R4mk", noplan="v3cal", practice="selected"); f["Bpre"] = compose(bb, route="R4mk", noplan="v3cal", practice="none")
    for c in ("rule", "extension", "noplan", "practice"): f[c] = bb[c].to_numpy(bool)
    f["practice_status"] = bb.practice_status.to_numpy(); f["practice_weight"] = bb.practice_weight.to_numpy(float)
    f["Fb"] = np.maximum(60, f.F.to_numpy()); assert np.abs(f.Fb.to_numpy() - bb.F.to_numpy()).max() < 1e-6
    f["nm"] = f.LOBT_flt.notna(); f["clock"] = f.cls == "clock"; f["b_s"] = secs(f.BLOCK_TIME_UTC_mvt, f.SCHED_TIME_UTC_mvt)
    f["point"] = secs(f.MVT_TIME_UTC_mvt, f.SCHED_TIME_UTC_mvt)                      # the label of a row with a default block time
    f["taxi_nm"] = secs(f.MVT_TIME_UTC_mvt, f.AOBT_3_flt)                             # NM's taxi time
    f["delay"] = secs(f.AOBT_3_flt, f.SCHED_TIME_UTC_mvt)                             # NM's off-block against the schedule
    f["late_l"] = secs(f.MVT_TIME_UTC_mvt, f.LOBT_flt); f["g_e1"] = secs(f.AOBT_3_flt, f.EOBT_1_flt)
    f["g_m15"] = secs(f.AOBT_3_flt, np.minimum(f.LOBT_flt, f.EOBT_1_flt.fillna(f.LOBT_flt)))
    y = f.y.to_numpy()
    f["kind"] = np.select([~f.nm.to_numpy(), f.b_s.abs().to_numpy() <= 6, np.abs(y - 600) <= 12, y >= 3600], ["no plan", "placeholder", "default 600", "long"], "body")
    f["eB"] = (f.B - f.y) ** 2; f["gB"] = f.w26 * f.eB; f["jan"] = f.month == 1
    f["op"] = f.AIRCRAFT_OPERATOR_flt.fillna(f.FLIGHT_mvt.astype(str).str[:3]).astype(str)
    return f


def auc(score, pos):
    """AUC of `score` for the class `pos` (ties count half)."""
    s = pd.Series(score).rank().to_numpy(); n1 = int(pos.sum()); n0 = len(pos) - n1
    return np.nan if n1 == 0 or n0 == 0 else (s[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_within(score, pos, strata):
    """AUC of `score` inside the strata (pairs of one positive and one negative of the same stratum), pooled over the pairs."""
    num = 0.0; den = 0.0
    for _, ix in pd.Series(np.arange(len(pos))).groupby(strata).groups.items():
        ix = np.asarray(ix); p = pos[ix]; n1 = int(p.sum()); n0 = len(ix) - n1
        if n1 == 0 or n0 == 0: continue
        num += auc(score[ix], p) * n1 * n0; den += n1 * n0
    return (num / den if den else np.nan), den


def boot(f, d, draws=2000, seed=16810):
    """G of the row gains d (s2, squared-error fall) with an interval over airport-days."""
    g = pd.factorize(f.airport.astype(str) + "|" + f.day.astype(str))[0]; K = g.max() + 1; by = np.bincount(g, weights=f.w26.to_numpy() * d, minlength=K); r = np.random.default_rng(seed).integers(0, K, size=(draws, K)); bs = by[r].sum(axis=1)
    return by.sum(), np.quantile(bs, 0.025), np.quantile(bs, 0.975), by.sum() - np.sort(by)[::-1][:5].sum()
