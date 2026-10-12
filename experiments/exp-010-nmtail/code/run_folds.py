"""Fit the NM-tail head per fold and score it against the current best line."""
import argparse, json, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-010-nmtail/code")
sys.path.insert(0, "experiments/exp-003-anatomy/code")
import prc.data as d
import prc.nmtail as nt
import common as C
from features import CAT, BASE_NUM, OCC, NBR

REOCC = d.ROOT / "artifacts" / "exp-010-nmtail"
R1 = d.ROOT / "experiments/exp-001-retro-v1/oof"
BASE_OOF = {"july_loo": R1 / "R0-jloo-july_loo.parquet",
            "december": R1 / "R0-dec-december.parquet",
            "january": R1 / "R0-jan-january.parquet"}
TAIL_OOF = d.ROOT / "experiments/exp-002-tail/oof/exp-002-tail-oof.parquet"

EXTRA = ["gap", "log_gap", "screen_1h", "screen_30m", "reocc_x_screen",
         "reocc_x_screen1h", "mvt_lobt_ratio"]
FEATURES = [c for c in CAT + BASE_NUM + OCC + NBR] + list(nt.SIGNALS) + EXTRA


def frame(m: str) -> pd.DataFrame:
    f = C.month(m, columns=None) if False else pd.read_parquet(
        d.ROOT / "artifacts/exp-003-anatomy/feat" / f"{m}.parquet")
    r = pd.read_parquet(REOCC / f"reocc_{m}.parquet")
    f = f.merge(r, on="MVT_ID_mvt", how="left", validate="one_to_one")
    f["gap"] = -f["aobt_lobt"]
    # explicit forms of the two screens that carry the mechanism, because the
    # tree has to find the interaction otherwise (exp-010 finding 2)
    g = f["gap"].to_numpy(float)
    rd = f["reocc_lobt_aobt_diff"].fillna(0).to_numpy(float)
    f["log_gap"] = np.sign(g) * np.log1p(np.abs(g))
    f["screen_1h"] = (g < -3600).astype("float32")
    f["screen_30m"] = (g < -1800).astype("float32")
    f["reocc_x_screen"] = rd * (g < -1800)
    f["reocc_x_screen1h"] = rd * (g < -3600)
    f["mvt_lobt_ratio"] = f["mvt_lobt"] / np.maximum(f["nm_taxi"], 60.0)
    return f


def fold_frames(fold: str):
    F = d.FOLDS[fold]
    tr = pd.concat([frame(m) for m in F.train_months], ignore_index=True)
    va = frame(F.valid_month)
    return tr, va


def slices(y, pred, has_nm, gap=None):
    nm_tail = has_nm & (y > 3600)
    out = {"rmse_all": d.rmse(y, pred),
           "rmse_le3600": d.rmse(y[y <= 3600], pred[y <= 3600]),
           "rmse_gt3600": d.rmse(y[y > 3600], pred[y > 3600]),
           "rmse_noNM": d.rmse(y[~has_nm], pred[~has_nm]) if (~has_nm).any() else np.nan,
           "rmse_bulk": d.rmse(y[has_nm & (y <= 3600)], pred[has_nm & (y <= 3600)]),
           "rmse_nm_tail": d.rmse(y[nm_tail], pred[nm_tail]),
           "n_nm_tail": int(nm_tail.sum())}
    return out


SCREEN = -1800.0   # gap below this is where the mechanism lives


def run(fold: str, mode: str, name: str, log, rounds=None, lr=None, leaves=None):
    tr, va = fold_frames(fold)
    tr_nm = tr[tr.has_nm & tr.TAXITIME_SEC_mvt.notna()]
    base_mode = mode.replace("_screened", "")
    model = nt.NMTailModel(mode=base_mode, features=FEATURES, categoricals=list(CAT))
    if rounds:
        model.n_estimators = rounds
    if lr:
        model.learning_rate = lr
    if leaves:
        model.num_leaves = leaves
    t0 = time.time()
    model.fit(tr_nm, tr_nm["TAXITIME_SEC_mvt"].to_numpy(float))
    head = model.predict(va)
    if mode.endswith("_screened"):
        # second stage: a specialist fitted only where AOBT is at least 30 min
        # after LOBT, i.e. the 4 % of NM rows that carry the mechanism
        sub = tr_nm[tr_nm["gap"] < SCREEN]
        spec = nt.NMTailModel(mode=base_mode, features=FEATURES, categoricals=list(CAT),
                              num_leaves=31, min_data_in_leaf=20, n_estimators=400,
                              learning_rate=0.05)
        spec.fit(sub, sub["TAXITIME_SEC_mvt"].to_numpy(float))
        m_va = (va["gap"] < SCREEN).to_numpy()
        if m_va.any():
            head = head.copy()
            head[m_va] = spec.predict(va[m_va])
    y = va["TAXITIME_SEC_mvt"].to_numpy(float)
    has_nm = va["has_nm"].to_numpy()
    res = {"name": name, "fold": fold, "mode": mode, "secs": round(time.time() - t0)}

    # head standalone on the NM rows; airport median taxi elsewhere (as exp-003)
    med = tr.loc[tr.TAXITIME_SEC_mvt.notna()].groupby("apt", observed=True)["TAXITIME_SEC_mvt"].median()
    fb = va["apt"].map(med).fillna(tr["TAXITIME_SEC_mvt"].median()).to_numpy()
    pred_head = np.where(has_nm, head, fb)
    res["head"] = slices(y, pred_head, has_nm)

    # against the current best line, where its OOF exists
    for label, path in [("R0", BASE_OOF.get(fold)), ("tail_model", TAIL_OOF)]:
        if path is None or not path.exists():
            continue
        o = pd.read_parquet(path)
        if "fold" in o.columns:
            o = o[o.fold == fold]
        if not len(o):
            continue
        o = o[["MVT_ID_mvt", "y_pred"]].rename(columns={"y_pred": f"p_{label}"})
        j = va[["MVT_ID_mvt"]].merge(o, on="MVT_ID_mvt", how="left")
        base = j[f"p_{label}"].to_numpy(float)
        ok = np.isfinite(base)
        res[f"base_{label}"] = slices(y[ok], base[ok], has_nm[ok])
        swap = np.where(has_nm & ok, head, base)
        res[f"swap_{label}"] = slices(y[ok], swap[ok], has_nm[ok])
        for w in (0.25, 0.5):
            mix = np.where(has_nm & ok, (1 - w) * base + w * head, base)
            res[f"mix{w}_{label}"] = slices(y[ok], mix[ok], has_nm[ok])
    OOF = d.ROOT / "experiments/exp-010-nmtail/oof"
    OOF.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"MVT_ID_mvt": va["MVT_ID_mvt"].to_numpy(), "fold": fold,
                  "y_true": y, "y_pred": pred_head}).to_parquet(
        OOF / f"{name}_{fold}.parquet", index=False)
    res["config"] = {"mode": mode, "n_estimators": model.n_estimators,
                     "learning_rate": model.learning_rate, "num_leaves": model.num_leaves,
                     "min_data_in_leaf": model.min_data_in_leaf, "seed": model.seed,
                     "n_features": len(FEATURES)}
    with open(log, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    print(f"[{name}] {fold} {mode}: head nm_tail={res['head']['rmse_nm_tail']:.0f} "
          f"bulk={res['head']['rmse_bulk']:.1f} all={res['head']['rmse_all']:.0f} "
          f"({res['secs']}s)", flush=True)
    for k in ("base_R0", "swap_R0", "mix0.5_R0", "base_tail_model", "swap_tail_model"):
        if k in res:
            print(f"     {k:18s} nm_tail={res[k]['rmse_nm_tail']:7.0f} all={res[k]['rmse_all']:7.1f} "
                  f"bulk={res[k]['rmse_bulk']:6.1f}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="july,july_loo,december,january")
    ap.add_argument("--modes", default="phi,p")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--leaves", type=int, default=None)
    ap.add_argument("--name", default="H")
    a = ap.parse_args()
    log = d.ROOT / "experiments/exp-010-nmtail/climb.log"
    for mode in a.modes.split(","):
        for fold in a.folds.split(","):
            run(fold, mode, f"{a.name}-{mode}", log, a.rounds, a.lr, a.leaves)
