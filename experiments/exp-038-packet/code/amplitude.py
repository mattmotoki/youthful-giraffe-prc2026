"""Where the bounded-amplitude gain of Q01's finalists comes from.

Reads the committed packet (`packet/rows-{fold}.parquet`) and nothing else, so
it is also the evidence that the packet is a usable artifact and not a private
intermediate.

For each finalist the amplitude `s` is the LEAVE-ONE-FOLD-OUT one from
`analysis/q01-table.csv`: the value scoring a fold was fitted on the two months
that fold is not.  The served vector is `p = v10 + s*(p_candidate - v10)`, a
convex blend when `0 <= s <= 1`.  The tables below split its paired SSE change
by regime, by airport and by 3-day date block, which is what says whether the
gain is a broad shrinkage or a few rows.

    python amplitude.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import align as A  # noqa: E402

FOLDS = A.FOLDS
TOP = 10


def main() -> None:
    T = pd.read_csv(EXP / "analysis" / "q01-table.csv")
    cols = json.loads((EXP / "packet" / "columns.json").read_text())["d_columns"]
    inv = {v: k for k, v in cols.items()}
    fin = (T[T["candidate"] != cols[inv["v10 (exp-036 refitbulk-mask)"]]]
           .sort_values("gain_board_loo_amp", ascending=False).head(TOP))
    data = {f: pd.read_parquet(EXP / "packet" / f"rows-{f}.parquet") for f in FOLDS}

    reg, apt, blk, chk = [], [], [], []
    for _, row in fin.iterrows():
        name = row["candidate"]
        col = inv[name]
        for f in FOLDS:
            d0 = data[f]
            s = float(row[f"amp_loo_{f}"])
            r = d0["r"].to_numpy("float64")
            d = d0[col].to_numpy("float64") * s
            e10 = r * r
            ec = (r - d) ** 2
            dd = ec - e10
            n = len(dd)
            base = {"candidate": name, "fold": f, "amplitude": s}
            chk.append({**base, "n": n, "n_changed": int((d != 0).sum()),
                        "rmse_v10": float(np.sqrt(e10.mean())),
                        "rmse_blend": float(np.sqrt(ec.mean())),
                        "gain_rmse": float(np.sqrt(e10.mean()) - np.sqrt(ec.mean())),
                        "sse_change": float(dd.sum()),
                        "max_abs_move": float(np.abs(d).max())})
            for key, sink in (("regime", reg), ("airport", apt)):
                g = pd.DataFrame({key: d0[key].to_numpy(), "dsse": dd, "d": d})
                agg = g.groupby(key).agg(n=("d", "size"), n_changed=("d", lambda x: int((x != 0).sum())),
                                         dsse=("dsse", "sum")).reset_index()
                for _, a in agg.iterrows():
                    sink.append({**base, key: a[key], "n": int(a["n"]),
                                 "n_changed": int(a["n_changed"]), "dsse": float(a["dsse"])})
            b3 = (d0["day"].to_numpy() - d0["day"].to_numpy().min()) // 3
            g = pd.DataFrame({"block": b3, "dsse": dd})
            agg = g.groupby("block").agg(n=("dsse", "size"), dsse=("dsse", "sum")).reset_index()
            for _, a in agg.iterrows():
                blk.append({**base, "block3d": int(a["block"]), "n": int(a["n"]),
                            "dsse": float(a["dsse"])})

    pd.DataFrame(chk).to_csv(EXP / "analysis" / "q01-amplitude-folds.csv", index=False)
    pd.DataFrame(reg).to_csv(EXP / "analysis" / "q01-amplitude-regime.csv", index=False)
    pd.DataFrame(apt).to_csv(EXP / "analysis" / "q01-amplitude-airport.csv", index=False)
    B = pd.DataFrame(blk)
    B.to_csv(EXP / "analysis" / "q01-amplitude-blocks.csv", index=False)

    print("=== the blend's paired SSE change by regime (negative = better than v10) ===")
    R = pd.DataFrame(reg)
    print(R.pivot_table(index=["candidate", "regime"], columns="fold", values="dsse",
                        aggfunc="sum").round(0).to_string())
    print("\n=== 3-day blocks: how many of them the blend improves ===")
    w = B.groupby(["candidate", "fold"]).agg(
        blocks=("dsse", "size"), better=("dsse", lambda x: int((x < 0).sum())),
        worst=("dsse", "max"), best=("dsse", "min")).reset_index()
    print(w.to_string(index=False))
    print("\n=== per fold ===")
    print(pd.DataFrame(chk).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
