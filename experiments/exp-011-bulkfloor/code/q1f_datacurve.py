"""Six training months against eleven, same model family, identical validation rows.

Guarded by a content hash: `experiments/exp-008-diversity/oof` contains eight
groups of byte-identical files (`C-cat-july` == `C1-d8-july`,
`X-xgb-july` == `X1-d8-july`, and six more), so a naive pairing compares two
different families and calls the difference a data effect. Review round 1, B4.
"""
import glob, hashlib, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-011-bulkfloor/code")
import prc.data as d
import common as C

SRC = d.ROOT / "experiments/exp-008-diversity/oof"


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    names = []
    for p in sorted(glob.glob(str(SRC / "*-july.parquet"))):
        n = os.path.basename(p)[:-len("-july.parquet")]
        if (SRC / f"{n}-july_loo.parquet").exists():
            names.append(n)
    # a file that is byte-identical to another family's file is that family's fit
    seen = {}
    for n in names:
        h = md5(SRC / f"{n}-july.parquet")
        seen.setdefault(h, []).append(n)
    dup = {n: v for h, v in seen.items() if len(v) > 1 for n in v}
    m = d.load_month("2025-07", columns=["MVT_ID_mvt", "PHASE_mvt", "FLIGHT_ID_mvt",
                                         "AIRCRAFT_TYPE_mvt"])
    m = m[m.PHASE_mvt == "DEP"]
    rows = []
    for n in names:
        a = pd.read_parquet(SRC / f"{n}-july.parquet")
        b = pd.read_parquet(SRC / f"{n}-july_loo.parquet")
        j = a.merge(b, on="MVT_ID_mvt", suffixes=("_6", "_11")).merge(m, on="MVT_ID_mvt")
        bulk = j.FLIGHT_ID_mvt.notna() & j.AIRCRAFT_TYPE_mvt.notna() & j.y_true_6.le(3600)
        rows.append({"model": n, "n": int(bulk.sum()),
                     "6 months": d.rmse(j.y_true_6[bulk], j.y_pred_6[bulk]),
                     "11 months": d.rmse(j.y_true_6[bulk], j.y_pred_11[bulk]),
                     "usable": n not in dup,
                     "note": "" if n not in dup else
                             f"six-month file is byte-identical to {sorted(set(dup[n]) - {n})[0]}"})
    T = pd.DataFrame(rows)
    T["gain"] = T["6 months"] - T["11 months"]
    T.to_csv(C.ANA / "data_curve.csv", index=False)
    ok = T[T.usable]
    out = ["# Q1(f) — six training months against eleven, same family, same rows\n",
           f"{len(T)} families have OOF for both folds; **{len(T)-len(ok)} are excluded** "
           "because their six-month file is byte-identical to another family's, so the "
           "pair would compare two different models (review round 1, B4).\n",
           C.md_table(T.set_index("model")[["n", "6 months", "11 months", "gain", "usable", "note"]],
                      floatfmt="{:.2f}"), "",
           f"Mean gain over the {len(ok)} usable families: **{ok.gain.mean():+.2f} s** "
           f"(median {ok.gain.median():+.2f}, range {ok.gain.min():+.2f} to {ok.gain.max():+.2f}). "
           f"Including the two contaminated rows would give {T.gain.mean():+.2f} s.\n"]
    C.write("q1f_datacurve.md", "\n".join(out))
    print(T.round(2).to_string(index=False))
    print(f"\nusable mean {ok.gain.mean():+.2f}  all mean {T.gain.mean():+.2f}")


if __name__ == "__main__":
    main()
