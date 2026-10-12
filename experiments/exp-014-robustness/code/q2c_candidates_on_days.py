"""Q2c: which candidate is better ON THE EXPOSED DAYS, on the 2025 folds?

Candidate 2 adds `D1-winter`, whose day-state block pushes +71 s on January
2026's out-of-range rows (q1c). exp-007 measured that block as +72 s of HARM on
january's disrupted days; the analogue in q2b says the blend UNDER-predicts on
days past the training envelope. Those two facts point opposite ways, and the
january fold can decide between them, because it contains days of both kinds.

Both candidates are rebuilt on the folds with their own shipped rule -- rotated
NNLS over three members for candidate 1, flat over four plus a fitted five-member
vector inside `nm_late` for candidate 2 -- and scored on the same slices.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames14 as F  # noqa: E402
ROOT = F.ROOT
for p in (str(ROOT), str(ROOT / "experiments" / "exp-012-blend2")):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402
import blend as BL  # noqa: E402
import nmtail_regime as NRG  # noqa: E402
import nmtail_rotate as NR  # noqa: E402

OUT = F.EXP / "analysis"
C1 = ("T-tailv2", "C2-cat", "L2-hub")


def build() -> dict:
    data = {f: NRG.enriched(f) for f in BL.FOLDS}
    names = data[BL.FOLDS[0]]["names"]
    k = len(names)
    idx1 = [names.index(n) for n in C1]
    out = {}
    for f in BL.FOLDS:
        src = [g for g in BL.FOLDS if g != f]
        # candidate 1: rotated NNLS over its three members
        P1 = np.concatenate([data[g]["P"][:, idx1] for g in src])
        y1 = np.concatenate([data[g]["y"] for g in src])
        w1 = NR.weights(P1, y1, "nnls", len(idx1))
        # candidate 2: flat over four outside nm_late, fitted five inside
        selg = {g: (data[g]["regime"] == "nm_late") & data[g]["ok"] for g in src}
        P5 = np.concatenate([data[g]["P"][selg[g]] for g in src])
        y5 = np.concatenate([data[g]["y"][selg[g]] for g in src])
        w5 = NR.weights(P5, y5, "nnls", k)
        w4 = np.full(k - 1, 1.0 / (k - 1))
        d = data[f]
        sel = (d["regime"] == "nm_late") & d["ok"]
        p2 = np.empty(len(d["y"]))
        p2[sel] = d["P"][sel] @ w5
        p2[~sel] = d["P"][~sel][:, :k - 1] @ w4
        out[f] = dict(y=d["y"], c1=np.maximum(d["P"][:, idx1] @ w1, B.FLOOR),
                      c2=np.maximum(p2, B.FLOOR), ids=data[f].get("ids"))
    return out, data


def main() -> None:
    res, data = build()
    FD = pd.read_csv(OUT / "q2b_fold_days.csv", parse_dates=["day"])
    rows = []
    for fold in BL.FOLDS:
        month = D.FOLDS[fold].valid_month
        raw = D.load_month(month, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt",
                                           "MVT_TIME_UTC_mvt", "AOBT_3_flt"])
        raw = raw[raw["PHASE_mvt"] == "DEP"]
        # rebuild the join order nmtail_rotate used, from the L2-hub OOF
        base = NR.load_one(NR.CORE["L2-hub"], fold, "L2-hub")
        j = base[["MVT_ID_mvt"]].merge(pd.DataFrame({
            "MVT_ID_mvt": raw["MVT_ID_mvt"].to_numpy("float64"),
            "apt": raw["ADEP_mvt"].astype(str).to_numpy(),
            "day": raw["MVT_TIME_UTC_mvt"].dt.floor("D").to_numpy(),
            "obs": (raw["MVT_TIME_UTC_mvt"] - raw["AOBT_3_flt"]).dt.total_seconds().to_numpy()}),
            on="MVT_ID_mvt", how="left")
        fd = FD[FD["fold"] == fold][["apt", "day", "over_tr_max", "n"]]
        j = j.merge(fd, on=["apt", "day"], how="left")
        y, c1, c2 = res[fold]["y"], res[fold]["c1"], res[fold]["c2"]
        assert len(j) == len(y), (len(j), len(y))
        ov = j["over_tr_max"].to_numpy("float64")
        rank = j.groupby("apt")["over_tr_max"].rank(pct=True).to_numpy("float64")
        slices = (("whole fold", np.ones(len(y), bool)),
                  ("worst 5 % of days", rank > 0.95),
                  ("days past the training maximum", ov > 0),
                  ("days past it by >30 s", ov > 30))
        for label, m in slices:
            if m.sum() < 20:
                continue
            e1, e2 = c1[m] - y[m], c2[m] - y[m]
            rows.append(dict(fold=fold, slice=label, n=int(m.sum()),
                             rmse_c1=float(np.sqrt((e1 ** 2).mean())),
                             rmse_c2=float(np.sqrt((e2 ** 2).mean())),
                             gain_c2=float(np.sqrt((e1 ** 2).mean()) - np.sqrt((e2 ** 2).mean())),
                             bias_c1=float(e1.mean()), bias_c2=float(e2.mean()),
                             mean_push=float((c2[m] - c1[m]).mean())))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "q2c_candidates_on_days.csv", index=False)
    print("=== candidate 2 against candidate 1, by day-exposure slice ===")
    print("(gain_c2 > 0 means candidate 2 is better on that slice)")
    print(T.round(2).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
