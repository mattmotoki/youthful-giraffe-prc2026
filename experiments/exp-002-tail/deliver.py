"""The single delivered model: `prc.tail_model.TailModel` measured through
`prc.data.fold_frames`, with the four metrics COMMON.md requires per fold.

Folds: `july`, `july_loo`, `december`, `january`. `july_loo` trains on the other
eleven months of 2025 -- August included -- so July 2025 is predicted with
same-season data available, which is the shape the submission model is in: it
trains on all twelve months and predicts July 2026 with July and August 2025 in
training. The forward `july` fold has no same-season data at all and is therefore
pessimistic for the ranking month; `(july_loo, january)` is the desk's best fold
estimate of the board score.

This is the only script that writes `oof/exp-002-tail-oof.parquet`. The climb
harnesses (`run_all.py`, `run_seg*.py`, `run_final.py`, `run_weather.py`) read a
feature cache and measure mechanisms; this reads the raw months through the
shared contract, so the delivered number is reproducible from `prc/` alone.
"""
from __future__ import annotations
import time
import numpy as np, pandas as pd
import prc.data as D
from prc.tail_model import TailModel

EXP = D.ROOT / "experiments" / "exp-002-tail"
(EXP / "oof").mkdir(exist_ok=True)
NR, NN_RANK = 344841, 388


def metrics(y, p, sig, fold, name):
    y = np.asarray(y, float); p = np.asarray(p, float); e2 = (y - p) ** 2
    tail = y > 3600
    nonm = sig["regime"].to_numpy() != "nm"
    noac = sig["regime"].to_numpy() == "noac"
    r = lambda m: float(np.sqrt(e2[m].mean())) if m.sum() else np.nan
    return dict(fold=fold, model=name, n=len(y), rmse=r(np.ones(len(y), bool)),
                rmse_le3600=r(~tail), rmse_gt3600=r(tail), rmse_nonm=r(nonm),
                rmse_noac=r(noac), n_noac=int(noac.sum()),
                noac_contrib_s=float(np.sqrt(NN_RANK * e2[noac].mean() / NR)) if noac.sum() else np.nan)


def main(out_name="exp-002-tail-oof.parquet"):
    rows, oof = [], []
    for fold in ("july", "july_loo", "december", "january"):
        t0 = time.time()
        train, valid_masked, y_valid = D.fold_frames(fold)
        model = TailModel().fit(train, n_jobs=16)
        pred = model.predict(valid_masked)
        y = y_valid.reindex(pred.index).to_numpy(float)
        rows.append(metrics(y, pred["y_pred"], pred, fold, "TailModel (delivered)"))
        alt = pred["y_pred"].where(pred["y_pred_direct_offset"].isna(), pred["y_pred_direct_offset"])
        rows.append(metrics(y, alt, pred, fold, "TailModel, noac head = direct offset"))
        hard = pred["y_pred"].copy()
        na = pred["regime"] == "noac"
        hard[na] = np.where(pred.loc[na, "p_placeholder"] >= 0.5,
                            pred.loc[na, "y_pred_tail"], pred.loc[na, "y_pred_bulk"])
        rows.append(metrics(y, hard, pred, fold, "TailModel, noac head = hard switch"))
        orc = pred["y_pred"].copy(); orc[na] = y[na.to_numpy()]
        rows.append(metrics(y, orc, pred, fold, "TailModel, noac oracle (ceiling)"))
        oof.append(pd.DataFrame({"MVT_ID_mvt": pred["MVT_ID_mvt"].to_numpy(), "fold": fold,
                                 "y_true": y, "y_pred": pred["y_pred"].to_numpy()}))
        print(f"[{fold}] {time.time()-t0:.0f}s  rmse={rows[-4]['rmse']:.2f}", flush=True)
    out = pd.concat(oof, ignore_index=True)
    path = EXP / "oof" / out_name
    out.to_parquet(path, index=False)
    m = pd.DataFrame(rows)
    if out_name == "exp-002-tail-oof.parquet":
        m.to_csv(EXP / "work" / "metrics_delivered.csv", index=False)
        print(m.round(2).to_string(index=False))
    sha = D.sha256(path)
    print(f"\n{len(out)} OOF rows -> {path}\nsha256 {sha}", flush=True)
    return sha


if __name__ == "__main__":
    import sys
    if "--verify" in sys.argv:
        a = main()
        b = main("_verify_rerun.parquet")
        (EXP / "oof" / "_verify_rerun.parquet").unlink()
        print(f"\nreproducible: {a == b}\n  run 1 {a}\n  run 2 {b}")
        if a != b:
            raise SystemExit("NOT REPRODUCIBLE: the two runs disagree")
    else:
        main()

# --- reproducibility check -------------------------------------------------
# `python deliver.py --verify` runs the whole thing twice and asserts the OOF
# artifact is byte-identical, which is what makes the SHA-256 in RESULTS.md a
# checkable claim. Requires the deterministic LightGBM settings in
# prc/tail_model.MAIN_PARAMS / NOAC_PARAMS.
