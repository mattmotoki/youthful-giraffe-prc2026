"""Predict-only form of the HB3u parts (exp-190 SPEC, design row 8): the saved boosters on the frame's 2026 rows.

    EXP159_SET=12 $PY experiments/exp-190-final-clock/code/predict_head.py --models <fits dir> --out <dir>
        [--train-ref <ranking-mode HU18-jjrank.json of this script>] [--threads 4]

For each part of HB3u (`HU18`, `HU30s2`), the `jjrank` boosters that `fit.py --save-models` wrote to
`<fits dir>/<part>-jjrank.models/` predict the frame's 2026 rows. The output has the form of `fit.py`'s files
(`<out>/<part>-jjrank.{parquet,json,seeds.npz}`), so `merge.py` merges it unchanged. The training rows, the input
list and the target are rebuilt as `fit.py` builds them and checked against the boosters' fit record (rows by
month, pushback inputs, input list); the SHA-256 of the training matrix and target is recorded. With
`--train-ref`, it must equal the recorded value of the reference run: then this frame's training rows are the
rows the boosters were fitted on, and a refit with the same recipe would give the same boosters. Reads the 2025
labels of the training rows (the target), as `fit.py` does. Write-once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

import common as C
import pilot as P                     # noqa: E402  (path set in common)
from fit import targets               # noqa: E402

PARTS = ("HU18", "HU30s2")
SPLIT = "jjrank"
#: keys that pilot.write sets; a copied fit record must not carry them
PILOT_KEYS = ("stage", "split", "owner", "protocol", "splits_fingerprint", "rows", "output_sha256", "host", "python",
              "written_utc")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-ref", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if C.SET != "12":
        raise SystemExit("EXP159_SET=12: the HB3u frame")
    for part in PARTS:
        for ext in ("parquet", "json", "seeds.npz"):
            C.refuse_existing(a.out / f"{part}-{SPLIT}.{ext}")
    t0 = time.time()
    s = P.split(SPLIT)
    if s.token != C.TOKENS[SPLIT]:
        raise SystemExit(f"split {SPLIT}: token {s.token} is not the contract's {C.TOKENS[SPLIT]}")
    f = C.load_frame(with_track=True)
    is25 = f.month.isin(C.MONTHS_2025).to_numpy()
    anchor, z = targets(f, "obs")
    wp = f[[C.TIME]]
    train = is25 & s.label_mask(wp)
    m = ~is25                                              # fit.py: a ranking split scores every 2026 frame row
    if not np.isfinite(z[train]).all():
        raise SystemExit("a training row has no target")
    ref = json.loads(a.train_ref.read_text()) if a.train_ref else None
    for part in PARTS:
        fit_rec = json.loads((a.models / f"{part}-{SPLIT}.json").read_text())
        if fit_rec["arm"] != "A4u" or fit_rec["target"] != "obs" or fit_rec["learner"] != "lightgbm":
            raise SystemExit(f"{part}: not an A4u LightGBM fit of the obs target")
        if fit_rec["fit"]["drop_months"]:
            raise SystemExit(f"{part}: a fit with dropped months")
        by_month = f.loc[train].groupby("month").size().to_dict()
        if int(train.sum()) != fit_rec["train_rows"] or by_month != fit_rec["train_rows_by_month"]:
            raise SystemExit(f"{part}: training rows {int(train.sum())} {by_month} are not the fit's")
        cov = f.loc[train, C.TRACK_PUSH].notna().mean()
        push = [c for c in C.TRACK_PUSH if cov[c] >= 0.05]  # fit.py's --min-coverage default
        feats = C.arm_features("A4u", f.columns, push)
        if feats != fit_rec["features"]:
            raise SystemExit(f"{part}: the input list is not the fit's")
        X = f[feats].to_numpy("float32")
        h = hashlib.sha256()
        h.update("\n".join(feats).encode())
        h.update(np.ascontiguousarray(X[train]).tobytes())
        h.update(np.ascontiguousarray(z[train]).tobytes())
        train_sha = h.hexdigest()
        if ref is not None and train_sha != ref["train_matrix_sha256"]:
            raise SystemExit(f"{part}: the training matrix {train_sha} is not the reference's "
                             f"{ref['train_matrix_sha256']}; the boosters do not fit this frame (refit)")
        seeds = fit_rec["fit"]["seeds"]
        rounds = fit_rec["fit"]["rounds"]
        zz = np.zeros((len(seeds), int(m.sum())), "float32")
        boosters = {}
        mdir = a.models / f"{part}-{SPLIT}.models"
        for k, seed in enumerate(seeds):
            path = mdir / f"seed-{seed}.txt"
            booster = lgb.Booster(model_file=str(path))
            if booster.num_trees() != rounds:
                raise SystemExit(f"{path}: {booster.num_trees()} trees, expected {rounds}")
            zz[k] = booster.predict(X[m], num_threads=a.threads, num_iteration=rounds)
            boosters[str(seed)] = {"path": str(path), "sha256": C.sha(path)}
        pred = f.loc[m, [C.ID, "month", "utc_day", "airport"]].reset_index(drop=True)
        pred[f"pred_{part}"] = anchor[m] + zz.mean(0).astype("float64")
        a.out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.out / f"{part}-{SPLIT}.seeds.npz", ids=pred[C.ID].to_numpy("int64"),
                            anchor=anchor[m], z=zz, seeds=np.array(seeds))
        rec = {k: v for k, v in fit_rec.items() if k not in PILOT_KEYS}
        rec["inputs"] = {"frame_sha256": C.sha(C.FRAME), "track_sha256": C.sha(C.TRACK), "extra_sha256": None,
                         "weather_sha256": None, "actype_table_sha256": C.sha(C.ACTYPE_TABLE)}
        if C.N_DEPARTURES is not None:
            rec["ranking_mode"] = C.RANKING_MODE
        else:
            rec.pop("ranking_mode", None)
        rec.update(domain="ranking2026", trees_used=rounds, train_matrix_sha256=train_sha,
                   predicted_from={"models": str(mdir), "fit_record_sha256": C.sha(a.models / f"{part}-{SPLIT}.json"),
                                   "fit_output_sha256": fit_rec["output_sha256"], "boosters": boosters},
                   train_ref=(None if a.train_ref is None else {"path": str(a.train_ref), "sha256": C.sha(a.train_ref)}),
                   code_sha256={"predict_head.py": C.sha(__file__), "common.py": C.sha(C.__file__),
                                "fit.py": C.sha(Path(__file__).with_name("fit.py"))})
        P.write(a.out, part, s, SPLIT, pred, rec)
        print(f"wrote {part}-{SPLIT}: {len(pred)} rows, train matrix {train_sha[:12]}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
