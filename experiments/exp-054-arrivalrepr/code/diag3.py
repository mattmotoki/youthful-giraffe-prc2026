"""exp-054-arrivalrepr: an UNREGISTERED zero-fit diagnostic, written after iteration 2 was measured.

Question (RESULTS section 7, the assignment's "sparse structure"): where the departure stand is
sparse in the fitting blocks, do the taxi-in factors help?  For each P row of day block b, the
stand's training support is the number of P rows with y > 0 and the same `apt|stand` in the other
four blocks (the rows the block-b fit saw).  Strata: stand missing; support < 100 (LightGBM's
`min_data_per_group`, below which a categorical split cannot isolate the stand); 100 <= support
< 300; support >= 300 (the upper cut was set after a first run found no stand with 1,000).  Per fold and stratum: rows, the
share with a nonzero `f_stand`, and the paired gain (s^2 per row, 1-day interval by exp-050's
`block_ci` on the stratum's rows, seed 54) of `true` over `rowonly` (iteration 1) and of `true_k`
over `rowonly_k`, `shuffle_k`, `gmean_k`, `donor_k` (iteration 2).  Arithmetic on the recorded
predictions (`oof`-equivalent h files) and labels; no fit, no choice.

    .venv/bin/python experiments/exp-054-arrivalrepr/code/diag3.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e54 as E                                                       # noqa: E402

NO, NR = E.NO, E.NR
OUT_DEFAULT = "analysis/iter2-sparse.json"
PAIRS = [("rowonly", "true"), ("rowonly_k", "true_k"), ("shuffle_k", "true_k"), ("gmean_k", "true_k"),
         ("donor_k", "true_k")]
STRATA = ["stand_missing", "support_lt_100", "support_100_299", "support_ge_300"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    OUT = E.EXP / args.out
    NO.refuse_existing([OUT], args.overwrite)
    rec = {"what": "UNREGISTERED zero-fit diagnostic after iteration 2: the factors' paired gain by the "
                   "departure stand's training support", "pairs": [f"{b}_over_{a}" for a, b in PAIRS],
           "code_sha256": {n: E.sha_file(E.HERE / n) for n in ("e54.py", "diag3.py")}, "folds": {}}
    for f in E.FOLDS:
        pan = E.panel(f, E.PANEL_LABEL_FREE + ["y_true"])
        pop = E.population(pan, f)
        blk = E.blocks(pan.day.to_numpy())
        ids = pan.MVT_ID_mvt.to_numpy("float64")
        j1, j2 = json.loads(E.fit_json(f).read_text()), json.loads(E.fit_json(f, "2").read_text())
        for p, j in ((E.h_path(f), j1), (E.h_path(f, "2"), j2)):
            if E.sha_file(p) != j["h_parquet"]["sha256"]:
                raise SystemExit(f"{p} moved since its fit record was written")
        H1, H2 = pd.read_parquet(E.h_path(f)), pd.read_parquet(E.h_path(f, "2"))
        assert np.array_equal(H1.MVT_ID_mvt.to_numpy("float64"), ids) and np.array_equal(H2.MVT_ID_mvt.to_numpy("float64"), ids)
        y, pv = pan.y_true.to_numpy("float64"), pan.p_v22.to_numpy("float64")
        pred = {"rowonly": H1["h_rowonly"], "true": H1["h_true"],
                **{a: H2[f"h_{a}"] for a in ("true_k", "rowonly_k", "shuffle_k", "gmean_k", "donor_k")}}
        pred = {a: np.maximum(E.MIN_PRED, pv + h.to_numpy("float64")) for a, h in pred.items()}
        frame = E.departure_frame(E.fold_month(f))
        Fe = E.row_features(frame).set_index("MVT_ID_mvt").reindex(ids)
        stand = Fe["apt_stand"].to_numpy(object)
        fit_rows = pop & (y > 0) & np.isfinite(y)
        sup = np.zeros(len(ids))
        for b in range(E.K):
            cnt = pd.Series(stand[fit_rows & (blk != b)]).dropna().value_counts()
            m = pop & (blk == b)
            sup[m] = pd.Series(stand[m]).map(cnt).fillna(0).to_numpy("float64")
        missing = pd.isna(pd.Series(stand)).to_numpy()
        strata = {"stand_missing": pop & missing, "support_lt_100": pop & ~missing & (sup < 100),
                  "support_100_299": pop & ~missing & (sup >= 100) & (sup < 300),
                  "support_ge_300": pop & ~missing & (sup >= 300)}
        rep = pd.read_parquet(E.rep_path(f))
        fst = rep[rep["key"] == "stand"].set_index("value")["true"]
        dep = frame[frame["PHASE_mvt"].to_numpy() == "DEP"].reset_index(drop=True)
        dk = E.departure_keys(dep).set_index("MVT_ID_mvt").reindex(ids)
        f_stand = fst.reindex(dk["stand"].to_numpy(object)).fillna(0.0).to_numpy("float64")
        days = pan.day.to_numpy()
        rec["folds"][f] = {}
        for s, m in strata.items():
            out = {"rows": int(m.sum()), "share_f_stand_nonzero": float((f_stand[m] != 0).mean()) if m.any() else None}
            rec["folds"][f][s] = out
            if m.sum() < 2:
                print(f"  [{f}] {s:16s} rows {out['rows']:7,} (too few rows for a gain)", flush=True)
                continue
            for a, b in PAIRS:
                v = (y[m] - pred[a][m]) ** 2 - (y[m] - pred[b][m]) ** 2
                c = NR.block_ci(v, days[m], np.random.default_rng(E.SEED))
                out[f"{b}_over_{a}"] = {"mse_gain_s2_per_row": float(v.mean()), "ci_block_1d": c["ci_block_1d"],
                                        "sse_gain": float(v.sum())}
            print(f"  [{f}] {s:16s} rows {out['rows']:7,} f_stand≠0 {out['share_f_stand_nonzero']:.2f}  "
                  + "  ".join(f"{b}/{a} {out[f'{b}_over_{a}']['mse_gain_s2_per_row']:+.2f} "
                              f"[{out[f'{b}_over_{a}']['ci_block_1d'][0]:+.1f},{out[f'{b}_over_{a}']['ci_block_1d'][1]:+.1f}]"
                              for a, b in PAIRS), flush=True)
        tot = sum(rec["folds"][f][s]["rows"] for s in STRATA)
        assert tot == int(pop.sum()), (tot, int(pop.sum()))
    NO.refuse_existing([OUT], args.overwrite)
    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True, default=float))
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
