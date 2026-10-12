"""exp-148 fold form (Matt, 10-01; docs/assignments/2026-10-01-two-fold-board-plan.md): the level 2
of a stage from its saved q01 and q07 models, without a refit.

    PRC_SOURCE_ROOT=... python fold_serve.py --stage R4 --inputs r4 --split jjrank \
        --member-art DIR [...] --stack DIR --features F --proxy F --out DIR

`--split jjrank`: the 2026 ranking rows, member inputs from the `jjrank` fits (janjul's training
rows); each of q01 and q07 scores every row; the fold form is the mean of the two (each floored
at 60 s, as the stacker serves). Write-once `<out>/<stage>F-jjrank.parquet` (owner `jjrank`,
columns as the stacker's: pred_S, pred_Scat, pred_Sb, anchor, anchor_served, plus each owner's
pred_Sb), read by `serve_r2.py --owner jjrank`.

`--split jjrefit` or `janjul`: the local fold read, q01 on January 2025 and q07 on July 2025, member
inputs from that split; write-once `<out>/<stage>F<split>-month-2025-01.parquet` and `-07`
(owners q01, q07). With `janjul` (the saved member vectors) each must equal the stacker's own
`<stage>-month-2025-0M.parquet`: the check that the saved models and this frame code reproduce
the fit; the record carries the largest difference.

The scored frame is the stacker's (`stacker.members_for`, `frame`, `feature_list`, `cat_frame`);
the feature list must equal the saved fit's record, and both model files must match its hashes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stacker as ST                 # noqa: E402  (sets sys.path for prc and pilot)
import pilot as P                    # noqa: E402

ID = ST.ID
#: the two fold owners of a stage and the month each scores (R4-type stages q01/q07, R2 r01/r07)
OWNERS_OF = {"q": (("q01", 1), ("q07", 7)), "r": (("r01", 1), ("r07", 7))}


def model_of(stack: Path, stage: str, owner: (str), inputs: str, fx: list[str]):
    import catboost as cb
    import lightgbm as lgb
    name = P.consumer(owner).name
    rec = json.loads((stack / f"{stage}-{name}.json").read_text())
    lp, cp = stack / f"{stage}-{name}.model.txt", stack / f"{stage}-{name}.cat.cbm"
    if rec["inputs"] != inputs or rec["features"] != fx:
        raise SystemExit(f"{stage} {owner}: the saved fit has other inputs or features")
    if ST.sha(lp) != rec["model_sha256"] or ST.sha(cp) != rec["cat_model_sha256"]:
        raise SystemExit(f"{stage} {owner}: a model file changed since the fit")
    cm = cb.CatBoostRegressor()
    cm.load_model(str(cp))
    return lgb.Booster(model_file=str(lp)), cm, {"record": str(stack / f"{stage}-{name}.json"),
                                                 "model_sha256": rec["model_sha256"],
                                                 "cat_model_sha256": rec["cat_model_sha256"]}


def predict(bst, cm, sc: pd.DataFrame, fx: list[str]) -> pd.DataFrame:
    f_l, f_c = bst.predict(sc[fx]), cm.predict(ST.cat_frame(sc, fx))
    anc = sc["anchor"].to_numpy("float64")
    return pd.DataFrame({ID: sc[ID].to_numpy("int64"),
                         "pred_S": np.maximum(anc + f_l, ST.FLOOR),
                         "pred_Scat": np.maximum(anc + f_c, ST.FLOOR),
                         "pred_Sb": np.maximum(anc + (f_l + f_c) / 2, ST.FLOOR),
                         "anchor": anc, "anchor_served": np.maximum(anc, ST.FLOOR)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--inputs", required=True, choices=sorted(ST.STAGE))
    ap.add_argument("--split", required=True, choices=("jjrank", "jjrefit", "janjul"))
    ap.add_argument("--member-art", type=Path, action="append", required=True)
    ap.add_argument("--stack", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--proxy", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--owners", choices=sorted(OWNERS_OF), default="q", help="q01/q07 (R4 stages) or r01/r07 (R2)")
    ap.add_argument("--refit-equal", action="append", default=[],
                    help="with --split jjrefit: a member source whose refit equals its saved janjul vector "
                         "(another worker's measured held RMS 0); its janjul file is read")
    a = ap.parse_args()
    OWNERS = OWNERS_OF[a.owners]
    if ST.STAGE[a.inputs] != a.stage:
        raise SystemExit(f"inputs {a.inputs} belong to stage {ST.STAGE[a.inputs]}, not {a.stage}")
    t0 = time.monotonic()
    ST.MEMBERS.update(ST.EXTRA_MEMBERS.get(a.inputs, {}))
    ST.MEMBER_SOURCE.update(ST.SOURCE_OF.get(a.inputs, {}))
    feats = pd.read_parquet(a.features)
    proxy = pd.read_parquet(a.proxy)
    proxy[ID] = proxy[ID].astype("int64")
    fx = ST.feature_list(feats, proxy, a.inputs)
    models = {o: model_of(a.stack, a.stage, o, a.inputs, fx) for o, _ in OWNERS}
    base = {"what": f"exp-148 fold form of {a.stage}: saved q01 and q07 models, split {a.split}",
            "stage": a.stage, "inputs": a.inputs, "models": {o: m[2] for o, m in models.items()},
            "feature_list_sha256": ST.hashlib.sha256(json.dumps(fx).encode()).hexdigest(),
            "features_sha256": ST.sha(a.features), "proxy_sha256": ST.sha(a.proxy),
            "code_sha256": {"fold_serve.py": ST.sha(Path(__file__)), "stacker.py": ST.sha(HERE / "stacker.py")}}
    if a.split == "jjrank":
        path = a.out / f"{a.stage}F-jjrank.parquet"
        if path.exists() or path.with_suffix(".json").exists():
            raise SystemExit(f"{path} exists; write-once")
        mem, rec = ST.members_for(a.member_art, [("jjrank", "jjrank")], "scored")
        sc = ST.frame(mem, feats, proxy)
        p = {o: predict(models[o][0], models[o][1], sc, fx) for o, _ in OWNERS}
        (o1, _), (o7, _) = OWNERS
        if not p[o1][ID].equals(p[o7][ID]) or not np.array_equal(p[o1]["anchor"], p[o7]["anchor"]):
            raise SystemExit(f"{o1} and {o7} scored different rows")
        out = p[o1][[ID, "anchor", "anchor_served"]].copy()
        for c in ("pred_S", "pred_Scat", "pred_Sb"):
            out[c] = np.maximum((p[o1][c].to_numpy() + p[o7][c].to_numpy()) / 2, ST.FLOOR)
        for o, _ in OWNERS:
            out[f"{o}_Sb"] = p[o]["pred_Sb"].to_numpy()
        if len(out) != 344_841 or not np.isfinite(out.drop(columns=["anchor"]).to_numpy()).all():
            raise SystemExit(f"{len(out)} rows or a value not finite")
        d = out[f"{o1}_Sb"].to_numpy() - out[f"{o7}_Sb"].to_numpy()
        record = {**base, "member_artifacts": rec, "rows": int(len(out)),
                  "first_minus_second_owner_Sb": {"owners": [o1, o7], "mean": float(d.mean()),
                                                  "rms": float(np.sqrt(np.mean(d ** 2)))},
                  "seconds": round(time.monotonic() - t0, 1)}
        print(json.dumps({k: record[k] for k in ("rows", "first_minus_second_owner_Sb")}), flush=True)
        print(P.write(a.out, f"{a.stage}F", P.split("jjrank"), "jjrank", out, record), flush=True)
        return
    # local fold read: q01 on January, q07 on July, member inputs from the split
    drop = frozenset(ST.r2_no_features()) if a.owners == "q" else frozenset()   # as the stacker (R4 owners)
    if a.refit_equal and a.split != "jjrefit":
        raise SystemExit("--refit-equal applies to --split jjrefit only")
    mem, rec = ST.members_for(a.member_art, [(a.split, a.split)], "janjul", drop,
                              {s: ("janjul", "janjul") for s in a.refit_equal})
    mem, no_features = ST.drop_no_features(mem, feats)
    mon = feats.set_index(ID).ctx_month.reindex(mem[ID].to_numpy()).to_numpy()
    for o, m in OWNERS:
        cons = P.consumer(o)
        cons.assert_admits(P.split(a.split), what=f"{a.split} for {cons.name}")
        name = f"{a.stage}F{a.split}"
        if (a.out / f"{name}-{cons.name}.parquet").exists():
            raise SystemExit(f"{name}-{cons.name}.parquet exists; write-once")
        sc = ST.frame(mem[mon == m].reset_index(drop=True), feats, proxy)
        out = predict(models[o][0], models[o][1], sc, fx)
        saved = pd.read_parquet(a.stack / f"{a.stage}-{cons.name}.parquet").set_index(ID).reindex(out[ID])
        diff = {c: float(np.nanmax(np.abs(out[c].to_numpy() - saved[c].to_numpy())))
                for c in ("pred_S", "pred_Scat", "pred_Sb", "anchor")}
        rms = float(np.sqrt(np.mean((out["pred_Sb"].to_numpy() - saved["pred_Sb"].to_numpy()) ** 2)))
        record = {**base, "member_artifacts": rec, "refit_equal_janjul": a.refit_equal, "owner": (o), "rows": int(len(out)),
                  "rows_without_features_dropped": no_features,
                  "vs_saved_fit": {"max_abs": diff, "rms_pred_Sb": rms,
                                   "rows_missing_in_saved": int(saved["pred_Sb"].isna().sum())},
                  "seconds": round(time.monotonic() - t0, 1)}
        print(json.dumps({o: record["vs_saved_fit"], "rows": len(out)}), flush=True)
        print(P.write(a.out, name, cons, o, out, record), flush=True)


if __name__ == "__main__":
    main()
