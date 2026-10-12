"""exp-148: clock-loss view of the stacker (SPEC section 4, VALIDATION section 9).

    python experiments/exp-148-short-route/code/clockloss.py --art DIR [--art DIR ...] \
        --features features.parquet --proxy proxy-features.parquet --out DIR [--tag NAME]

For each scheme (`coverage`, `lirf_july`) and each outer owner k, the saved outer(k) stacker
models (`S-outer-k.model.txt`, `.cat.cbm`; `S22-...` when present) re-predict pair k with the
ADS-B-derived inputs blanked on the scheme's dark airport-days: `obs_*`, `v23_*`, the observed
taxi `q_taxi_o` and the `q_No_*` neighbour columns (so the `No_` residuals and the observed-taxi
differences are blank too), exp-137's two row clocks and the six slot-clock proxies
(`clock_*`); the class is ordinary there where it was clock. The anchor's members read no
ADS-B input and stay as they are. Limitation (SPEC section 4): the members L and Lp are not
refitted on masked points; their predictions are the unmasked ones.

Check first: the same code path on unmasked inputs reproduces every saved outer prediction.
Scores: `gscore.score(..., clock_mask=scheme)` of each masked form against the anchor (and
against the masked S22 form when present). Write-once `clockloss-<tag>.json` and the masked
vectors `clockloss-<tag>.parquet`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stacker as ST                 # noqa: E402  (sets sys.path for prc and pilot)
from prc import gscore as GS         # noqa: E402
from prc import weekpair as WP       # noqa: E402
import pilot as P                    # noqa: E402

ID = ST.ID
SCHEMES = ("coverage", "lirf_july")
FORMS = {"": "pred_S", "cat": "pred_Scat", "b": "pred_Sb"}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def masked_inputs(feats: pd.DataFrame, proxy: pd.DataFrame, scheme: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dd = GS.load_clock_loss_days()
    dd = dd[dd.scheme == scheme]
    key = set(zip(dd.airport.astype(str), dd.date.astype(str)))
    dark = np.array([(a, d) in key for a, d in zip(feats.ctx_airport.astype(str), feats.ctx_day.astype(str))])
    f, p = feats.copy(), proxy.copy()
    fcols = [c for c in f.columns if c.startswith(("obs_", "v23_", "q_No_"))] + ["q_taxi_o"]
    f.loc[dark, fcols] = np.nan
    clock = dark & (f.ctx_cls.astype(str).to_numpy() == "clock")
    f.loc[clock, "ctx_cls"] = "ordinary"
    pcols = [c for c in p.columns if c.startswith("clock_")]
    pdark = p[ID].isin(f.loc[dark, ID])
    p.loc[pdark, pcols] = np.nan
    return f, p, {"dark_rows": int(dark.sum()), "clock_to_ordinary": int(clock.sum()),
                  "feature_columns_blanked": fcols, "proxy_columns_blanked": pcols}


def predict(arts: list[Path], stage: str, k: int, sc: pd.DataFrame, fx: list[str]) -> pd.DataFrame:
    import catboost as cb
    import lightgbm as lgb
    d = ST.find(arts, stage, f"outer-{k}").parent
    bst = lgb.Booster(model_file=str(d / f"{stage}-outer-{k}.model.txt"))
    cm = cb.CatBoostRegressor()
    cm.load_model(str(d / f"{stage}-outer-{k}.cat.cbm"))
    f_l, f_c = bst.predict(sc[fx]), cm.predict(ST.cat_frame(sc, fx))
    anc = sc["anchor"].to_numpy("float64")
    return pd.DataFrame({ID: sc[ID].to_numpy("int64"),
                         "pred_S": np.maximum(anc + f_l, ST.FLOOR),
                         "pred_Scat": np.maximum(anc + f_c, ST.FLOOR),
                         "pred_Sb": np.maximum(anc + (f_l + f_c) / 2, ST.FLOOR),
                         "anchor_served": np.maximum(anc, ST.FLOOR)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", type=Path, action="append", required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--proxy", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    js, pq = a.out / f"clockloss-{a.tag}.json", a.out / f"clockloss-{a.tag}.parquet"
    if js.exists() or pq.exists():
        raise SystemExit(f"{js} exists; write-once")
    feats = pd.read_parquet(a.features)
    proxy = pd.read_parquet(a.proxy)
    proxy[ID] = proxy[ID].astype("int64")
    stages = [s for s in ("S", "S22") if all(any((d / f"{s}-outer-{k}.parquet").exists() for d in a.art)
                                             for k in range(1, 6))]
    if "S" not in stages:
        raise SystemExit("no complete set of S-outer-k artifacts")
    rows = GS.load_rows()
    ids_days = rows[rows.month.isin([1, 7])][[ID, "pair"]]
    rec = {"what": "exp-148 clock-loss view (SPEC section 4, VALIDATION section 9)",
           "spec_sha256": {n: sha(HERE.parent / n) for n in ("SPEC.md", "SPEC-r1.md")},
           "code_sha256": {"clockloss.py": sha(Path(__file__)), "stacker.py": sha(HERE / "stacker.py")},
           "features_sha256": sha(a.features), "proxy_sha256": sha(a.proxy),
           "splits_fingerprint": WP.fingerprint(), "stages": stages, "parity": {}, "masks": {}, "scores": {}}
    mem = {}
    for k in range(1, 6):
        mem[k], r = ST.members_for(a.art, [(f"outer-{k}", k)], "scored")
        rec.setdefault("member_artifacts", {}).update(r)
    fx = {}
    for stage in stages:
        saved = {}
        for k in range(1, 6):
            path = ST.find(a.art, stage, f"outer-{k}")
            srec = json.loads(path.with_suffix(".json").read_text())
            if srec["features_sha256"] != sha(a.features) or srec["proxy_sha256"] != sha(a.proxy):
                raise SystemExit(f"{path}: fitted on other feature tables")
            fx.setdefault(stage, srec["features"])
            if srec["features"] != fx[stage]:
                raise SystemExit(f"{stage}: feature lists differ between owners")
            saved[k] = P.load(path, k, what=f"{stage} outer-{k}").set_index(ID)
            got = predict(a.art, stage, k, ST.frame(mem[k], feats, proxy), fx[stage]).set_index(ID)
            dev = {c: float(np.abs(got[c].to_numpy() - saved[k][c].reindex(got.index).to_numpy()).max())
                   for c in FORMS.values()}
            rec["parity"][f"{stage}-outer-{k}"] = dev
            if max(dev.values()) > 1e-6:
                raise SystemExit(f"{stage} outer-{k}: re-prediction differs from the saved artifact {dev}")
    print(json.dumps({"parity": rec["parity"]}), flush=True)

    vecs = []
    y_ids = ids_days[ID].to_numpy("int64")
    y = GS.labels(y_ids)
    for scheme in SCHEMES:
        f_m, p_m, info = masked_inputs(feats, proxy, scheme)
        rec["masks"][scheme] = info
        vec = {}
        for stage in stages:
            per = {k: predict(a.art, stage, k, ST.frame(mem[k], f_m, p_m), fx[stage]) for k in range(1, 6)}
            for f, col in FORMS.items():
                vec[f"{stage}{f}"] = WP.join_outer(per, ids_days, col)
            vec.setdefault("anchor", WP.join_outer(per, ids_days, "anchor_served"))
        v = pd.DataFrame({n: s.reindex(y_ids).to_numpy("float64") for n, s in vec.items()})
        v.insert(0, ID, y_ids)
        v.insert(1, "scheme", scheme)
        vecs.append(v)
        pairs = [(f"S{f}", "anchor") for f in FORMS]
        if "S22" in stages:
            pairs += [(f"S{f}", f"S22{f}") for f in FORMS] + [(f"S22{f}", "anchor") for f in FORMS]
        for cand, base in pairs:
            res = {view: GS.score(y_ids, v[base].to_numpy(), v[cand].to_numpy(), y=y, view=view,
                                  clock_mask=scheme) for view in ("all", "clean")}
            rec["scores"][f"{scheme}:{cand}_vs_{base}"] = res
            iv = res["all"]["interval"]
            print(json.dumps({f"{scheme}:{cand}_vs_{base}": {"G": res["all"]["G"], "lo95": iv["lo95"],
                                                              "hi95": iv["hi95"],
                                                              "darkness": res["all"]["views"]["darkness"]}}),
                  flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    pd.concat(vecs, ignore_index=True).to_parquet(pq, index=False)
    rec["vectors_sha256"] = sha(pq)
    js.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
