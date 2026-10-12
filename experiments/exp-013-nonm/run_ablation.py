"""exp-013 candidate ablation: base plus each off-block proxy family."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
import prc.data as D
import features as F
from segment import rmse, FOLDS

N_JOBS = 6  # docs/COMPUTE.md compute allocation
CACHE = HERE.parents[1] / "artifacts" / "exp-013-nonm"
CACHE.mkdir(parents=True, exist_ok=True)


def month_ids(m: str) -> np.ndarray:
    """MVT_ID for the departure rows, in the order `month()` returns them.

    `features.month_features` resets the index, so the frame's own index is
    positional and must never be written into an OOF as an id.
    """
    raw = D.load_month(m, columns=[D.ID, "PHASE_mvt"])
    return raw.loc[raw["PHASE_mvt"] == "DEP", D.ID].to_numpy()


def month(m: str):
    p = CACHE / f"{m}.parquet"
    if p.exists():
        d = pd.read_parquet(p)
        return d.drop(columns=["__y", "__seg"]), d["__y"].to_numpy(), d["__seg"].to_numpy().astype(bool)
    X, y, seg = F.month_features(m)
    d = X.copy(); d["__y"] = y; d["__seg"] = seg
    d.to_parquet(p, index=False)
    return X, y, seg


def fit_predict(Xtr, ytr, Xva, rounds=600):
    import lightgbm as lgb
    cats = [c for c in F.CATS if c in Xtr.columns]
    for c in cats:
        a, b = Xtr[c].astype(str), Xva[c].astype(str)
        u = pd.Index(pd.unique(np.concatenate([a.to_numpy(), b.to_numpy()])))
        Xtr[c] = pd.Categorical(a, categories=u)
        Xva[c] = pd.Categorical(b, categories=u)
    mdl = lgb.LGBMRegressor(n_estimators=rounds, learning_rate=0.05, num_leaves=63,
                            min_child_samples=40, subsample=0.9, subsample_freq=1,
                            colsample_bytree=0.9, n_jobs=N_JOBS, random_state=7, verbose=-1)
    mdl.fit(Xtr, ytr, categorical_feature=cats)
    return mdl.predict(Xva), mdl


def run(fold: str, configs: dict, out: dict):
    f = D.FOLDS[fold]
    tr = [month(m) for m in f.train_months]
    Xtr_all = pd.concat([t[0] for t in tr], ignore_index=True)
    ytr = np.concatenate([t[1] for t in tr])
    Xva_all, yva, segva = month(f.valid_month)
    res = out.setdefault(fold, {})
    for name, fams in configs.items():
        cols = list(F.BASE) + [c for fam in fams for c in F.FAMILIES[fam]]
        t0 = time.time()
        p, mdl = fit_predict(Xtr_all[cols].copy(), ytr, Xva_all[cols].copy())
        res[name] = {"rmse_segment": rmse(yva[segva], p[segva]),
                     "rmse_all": rmse(yva, p), "n": int(segva.sum()),
                     "n_features": len(cols), "secs": round(time.time() - t0)}
        print(f"[{fold}] {name:16s} segment={res[name]['rmse_segment']:.1f} "
              f"all={res[name]['rmse_all']:.1f} ({res[name]['secs']}s)", flush=True)
        (HERE / "ablation.json").write_text(json.dumps(out, indent=2))
        oof = pd.DataFrame({D.ID: month_ids(f.valid_month), "fold": fold,
                            "y_true": yva, "y_pred": p, "seg": segva})
        oof.to_parquet(HERE / "oof" / f"{name}_{fold}.parquet", index=False)
        if name == "all":
            pd.Series(mdl.feature_importances_, index=cols).sort_values(
                ascending=False).to_csv(HERE / f"importance_all_{fold}.csv", header=["gain"])


CONFIGS = {"base": (), "op": ("op",), "stand": ("stand",), "surface": ("surface",),
           "bracket": ("bracket",), "all": ("op", "stand", "surface", "bracket"),
           # the families whose coverage on the segment is high, without `stand`
           # (21.9 % covered, and it costs the segment 81 s on its own)
           "sb": ("surface", "bracket"),
           "sbo": ("surface", "bracket", "op")}

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    out = json.loads((HERE / "ablation.json").read_text()) if (HERE / "ablation.json").exists() else {}
    only = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--only=")]
    cfg = ({k: CONFIGS[k] for k in only[0].split(",")} if only else CONFIGS)
    for fold in (args or list(FOLDS)):
        run(fold, cfg, out)
