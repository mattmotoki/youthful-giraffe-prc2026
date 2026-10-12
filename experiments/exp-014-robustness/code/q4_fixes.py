"""Q4: label-free prediction-time repairs, measured against their own control.

One fit per (fold, learner) on the UNMODIFIED training frame, then the same
model predicts the validation frame several times -- once raw, once per repair.
The fit is held fixed, so the difference between two columns is the repair and
nothing else. That is also exactly how a repair would be deployed: the model is
already trained on 2025 and the 2026 frame is what gets repaired.

Repairs:
  clip    every numeric feature clipped to the fold's TRAINING min-max.
          CALENDAR INDEX columns are excluded (`month_num`, `day_of_month`):
          on the backward january fold every validation row is out of range on
          `month_num` by construction (train Feb-Dec, validate Jan), and
          clipping it would relabel January as February -- an artefact of the
          fold, not an exposure of the deployment, where both ranking months
          sit inside 2025's 1-12.
  parent  an unseen `airport_stand` / `airport_runway` is replaced by a
          REPRESENTATIVE training level of its parent group (same airport, same
          stand prefix; then same airport), instead of reaching the encoder as
          an unknown. The representative is the group's level whose training
          median residual is closest to the group's row-weighted median, so the
          substitute is the best single stand-in for the parent rather than the
          busiest one.
  both    clip + parent.

Because the folds present these exposures on far fewer rows than the ranking
months do, the whole-fold RMSE is reported to show a repair does no harm, and
the decisive number is measured ON THE AFFECTED ROWS, paired, with a bootstrap.

  python q4_fixes.py --kind lgb --fold january
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames14 as F  # noqa: E402
ROOT = F.ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "exp-008-diversity"))
sys.path.insert(0, str(ROOT / "experiments" / "exp-001-retro-v1"))

import prc.data as D  # noqa: E402
import frames as FR  # noqa: E402
import guard as guard_mod  # noqa: E402
import learners as L  # noqa: E402


def _load(name: str, path: Path):
    """exp-001 and exp-008 both ship a `run.py`; a bare import takes whichever
    directory happens to be first on sys.path. Load exp-008's by its path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R8 = _load("exp008_run", ROOT / "experiments" / "exp-008-diversity" / "run.py")

OUT = F.EXP / "analysis"
CALENDAR = ("month_num", "day_of_month")
PARENT_COLS = ("airport_stand", "airport_runway")
_MARS = re.compile(r"(?<=\d)[LRC]$|(?<=[A-Za-z]\d)[LRC]$")
_PFX = re.compile(r"^([A-Za-z]+|[0-9])")

ARMS = {
    "lgb": dict(kind="lgb", name="L2-hub", variant="plain", monotone=False, huber_delta=1200.0,
                lr=0.05, leaves=255, rounds={"m6": 3225, "m11": 2100}),
    "cat": dict(kind="cat", name="C2-cat", depth=8, cpu=False, lr=0.05,
                rounds={"m6": 3625, "m11": 5325}),
}
BASE = dict(target="residual", clip_floor=60.0, residual_guard=True, seed=7, seed_bag=1,
            select=False, noac_min_rows=30, noac_rounds=None, noac_leaves=15,
            noac_min_data=5, noac_lr=0.05, cpu=False, epochs=12, outdir=None)


# ------------------------------------------------------------- repairs ----
def numeric_bounds(tr: pd.DataFrame, cols) -> dict:
    b = {}
    for c in cols:
        t = tr[c].to_numpy("float64")
        t = t[np.isfinite(t)]
        if len(t):
            b[c] = (float(t.min()), float(t.max()))
    return b


def apply_clip(X: pd.DataFrame, bounds: dict) -> tuple[pd.DataFrame, np.ndarray]:
    out = X.copy()
    touched = np.zeros(len(X), bool)
    for c, (lo, hi) in bounds.items():
        v = out[c].to_numpy("float64")
        m = np.isfinite(v) & ((v < lo) | (v > hi))
        if m.any():
            touched |= m
            out.loc[m, c] = np.clip(v[m], lo, hi)
    return out, touched


def stand_prefix(level: str) -> str:
    """`APT|STAND` -> `APT|<prefix>`; the prefix is stand_features' convention."""
    if "|" not in str(level):
        return str(level)
    apt, st = str(level).split("|", 1)
    base = _MARS.sub("", st.strip())
    m = _PFX.match(base)
    return f"{apt}|{m.group(1) if m else '?'}"


def airport_of(level: str) -> str:
    return str(level).split("|", 1)[0]


def parent_map(tr: pd.DataFrame, va: pd.DataFrame, col: str, resid: np.ndarray) -> dict:
    """Unseen validation level -> a representative TRAINING level of its parent.

    The representative of a group is the level whose training median residual is
    closest to the group's row-weighted median residual, among levels with at
    least 30 training rows (the same floor `prc.stand_features.StandStats` uses).
    """
    lv = tr[col].astype("object")
    d = pd.DataFrame({"lv": lv.to_numpy(), "r": resid})
    d = d[pd.notna(d["lv"])]
    per = d.groupby("lv", observed=True)["r"].agg(["size", "median"])
    per = per[per["size"] >= 30]
    if not len(per):
        return {}
    groups = {"pfx": {}, "apt": {}}
    keyf = {"pfx": stand_prefix, "apt": airport_of}
    for g, f in keyf.items():
        d2 = d.copy()
        d2["g"] = [f(x) for x in d2["lv"]]
        gm = d2.groupby("g", observed=True)["r"].median()
        pk = pd.Series([f(x) for x in per.index], index=per.index, name="g")
        for gval, sub in pk.groupby(pk):
            target = gm.get(gval, np.nan)
            if not np.isfinite(target):
                continue
            cand = per.loc[sub.index, "median"]
            groups[g][gval] = str((cand - target).abs().idxmin())
    seen = set(pd.unique(lv.dropna()))
    out = {}
    for level in pd.unique(va[col].astype("object").dropna()):
        if level in seen:
            continue
        rep = groups["pfx"].get(stand_prefix(level)) or groups["apt"].get(airport_of(level))
        if rep is not None:
            out[str(level)] = rep
    return out


def apply_parent(X: pd.DataFrame, maps: dict) -> tuple[pd.DataFrame, np.ndarray]:
    out = X.copy()
    touched = np.zeros(len(X), bool)
    for col, mp in maps.items():
        if not mp:
            continue
        v = out[col].astype("object")
        m = v.isin(list(mp)).to_numpy()
        if m.any():
            touched |= m
            out.loc[m, col] = v[m].map(mp).to_numpy()
    return out, touched


# ---------------------------------------------------------------- main ----
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=list(ARMS))
    ap.add_argument("--fold", required=True)
    a = ap.parse_args()
    cfg = dict(BASE, **ARMS[a.kind])
    fold = a.fold
    t0 = time.time()

    tr, va = FR.fold_frames(fold)
    feats = FR.feature_columns(tr)
    cats = FR.cat_columns(tr)
    nums = [c for c in feats if c not in cats and c not in CALENDAR]
    Xtr, Xva = tr[feats], va[feats]
    ytr, yva = tr["y"].to_numpy("float64"), va["y"].to_numpy("float64")
    keep = np.isfinite(yva)
    base_tr, base_va = R8.anchors(tr, va)      # anchors from the RAW frame, never repaired
    rtr = ytr - base_tr
    rounds = cfg["rounds"][FR.FAMILY[fold]]
    print(f"[{fold}/{a.kind}] train {Xtr.shape} valid {Xva.shape} rounds {rounds} "
          f"loaded at {time.time()-t0:.0f}s", flush=True)

    bounds = numeric_bounds(tr, nums)
    maps = {c: parent_map(tr, va, c, rtr) for c in PARENT_COLS if c in feats}
    print("[parent] " + json.dumps({c: len(m) for c, m in maps.items()}), flush=True)
    variants = {"raw": (Xva, np.zeros(len(va), bool))}
    Xc, tc = apply_clip(Xva, bounds)
    variants["clip"] = (Xc, tc)
    Xp, tp = apply_parent(Xva, maps)
    variants["parent"] = (Xp, tp)
    Xb, _ = apply_clip(Xp, bounds)
    variants["both"] = (Xb, tc | tp)
    for k, (_, t) in variants.items():
        print(f"[touch] {k}: {int(t.sum())} of {len(va)} validation rows", flush=True)

    lr = R8.make_learner(cfg, cfg["seed"])
    lr.fit(Xtr, rtr, cats, rounds, cfg["seed"])
    print(f"[{fold}/{a.kind}] fitted at {time.time()-t0:.0f}s", flush=True)
    head = None
    mtr = Xtr["noac"].to_numpy().astype(bool)
    if mtr.sum() >= cfg["noac_min_rows"]:
        b_tr = np.nan_to_num(Xtr.loc[mtr, "d_takeoff_sched"].to_numpy("float64"))
        head = R8.make_learner(cfg, cfg["seed"], head=True)
        head.fit(Xtr[mtr], ytr[mtr] - b_tr, cats, R8.NOAC_ROUNDS[cfg["kind"]], cfg["seed"])
        print(f"[{fold}/{a.kind}] noac head fitted at {time.time()-t0:.0f}s", flush=True)

    glo, ghi = R8.guard_bounds(fold)
    av = Xva["d_takeoff_aobt"].to_numpy("float64")
    preds = {}
    for k, (Xv, _) in variants.items():
        p = lr.predict(Xv) + base_va
        mv = Xv["noac"].to_numpy().astype(bool)
        if head is not None and mv.any():
            p = p.copy()
            p[mv] = head.predict(Xv[mv]) + np.nan_to_num(
                Xv.loc[mv, "d_takeoff_sched"].to_numpy("float64"))
        p = np.maximum(p, cfg["clip_floor"])
        p = guard_mod.apply(p, av, glo, ghi, floor=cfg["clip_floor"])
        preds[k] = p
        print(f"[{fold}/{a.kind}] predicted {k} at {time.time()-t0:.0f}s", flush=True)

    no_nm = Xva["nm_missing"].to_numpy().astype(bool)
    noac = Xva["noac"].to_numpy().astype(bool)
    rows = []
    for k, p in preds.items():
        r = R8.decompose(yva[keep], p[keep], no_nm[keep], noac[keep])
        r.update(fold=fold, kind=a.kind, variant=k,
                 n_touched=int(variants[k][1].sum()),
                 delta_rmse=r["rmse"] - 0.0)
        rows.append(r)
    T = pd.DataFrame(rows)
    T["delta_rmse"] = T["rmse"] - float(T.loc[T.variant == "raw", "rmse"].iloc[0])

    # the decisive number: the affected rows only, paired, with a bootstrap
    aff = []
    rng = np.random.default_rng(7)
    for k in ("clip", "parent", "both"):
        t = variants[k][1] & keep
        if t.sum() < 5:
            aff.append(dict(fold=fold, kind=a.kind, variant=k, n=int(t.sum()),
                            note="too few affected rows to measure"))
            continue
        y = yva[t]
        p0, p1 = preds["raw"][t], preds[k][t]
        e0, e1 = (p0 - y) ** 2, (p1 - y) ** 2
        d = e0 - e1                       # positive => the repair helps
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        aff.append(dict(fold=fold, kind=a.kind, variant=k, n=int(t.sum()),
                        rmse_raw=float(np.sqrt(e0.mean())), rmse_fix=float(np.sqrt(e1.mean())),
                        bias_raw=float((p0 - y).mean()), bias_fix=float((p1 - y).mean()),
                        mean_abs_change=float(np.abs(p1 - p0).mean()),
                        mse_gain=float(d.mean()), ci_lo=float(np.quantile(bs, 0.025)),
                        ci_hi=float(np.quantile(bs, 0.975)),
                        rmse_contrib_s=float(np.sign(d.mean()) *
                                             np.sqrt(abs(d.mean()) * t.sum() / len(yva[keep])))))
    A = pd.DataFrame(aff)

    tag = f"{a.kind}-{fold}"
    T.to_csv(OUT / f"q4_fold_{tag}.csv", index=False)
    A.to_csv(OUT / f"q4_affected_{tag}.csv", index=False)
    np.save(F.EXP / "oof" / f"q4-preds-{tag}.npy",
            np.stack([preds[k] for k in ("raw", "clip", "parent", "both")]))
    print("\n=== whole fold ===")
    print(T[["variant", "n_touched", "rmse", "delta_rmse", "rmse_bulk", "rmse_tail",
             "rmse_no_nm", "rmse_noac"]].round(3).to_string(index=False))
    print("\n=== affected rows only (paired) ===")
    print(A.round(3).to_string(index=False))
    print(f"\ndone in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
