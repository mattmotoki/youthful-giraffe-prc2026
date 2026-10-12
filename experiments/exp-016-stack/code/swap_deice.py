"""Does the stacker's gain survive the blend owner's repaired winter member?

exp-012 adopted `ds_columns="deice"` for the winter member (commit ec54ccf,
2026-09-05): the one bounded day-state column instead of the 25-column block,
worth 51 s on january's most disrupted 5 % of days. The desk's assignment named
"the repaired D1-winter when it lands", and it has -- as OOF only. There are no
ranking predictions for it, so it can be MEASURED here and not SERVED.

Same rotation, same arms, same bag, one member address changed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import members as M  # noqa: E402
import stack as S  # noqa: E402
import verify_r1 as V  # noqa: E402
sys.path.insert(0, str(M.ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

OUT = M.EXP / "analysis"
REPAIRED = "experiments/exp-012-blend2/oof/D1w-deice-{fold}.parquet"


def main() -> None:
    have_rank = (M.ROOT / "artifacts/exp-012/pred/D1w-deice-ranking.parquet").exists()
    print(f"repaired member ranking predictions present: {have_rank}", flush=True)
    old = M.OOF["D1-winter"]
    recs = []
    for tag, path in (("block (25 col, shipped)", old), ("deice (1 col, repaired)", REPAIRED)):
        M.OOF["D1-winter"] = path
        # Pin and verify the swapped-in member. The first version of this script
        # mutated the address without pinning, and the deice files were being
        # written by exp-012's lane six minutes before this ran -- exactly the
        # failure members.py's docstring says the pinning exists to prevent
        # (review round 2, N1). The pin file is per-member-set so a swap cannot
        # reuse the pins of the set it replaces.
        M.PINS = M.EXP / f"pins-swap-{'deice' if path is REPAIRED else 'block'}.json"
        bad = M.verify()
        if bad:
            raise SystemExit(f"{tag}: member set not clean: {bad}")
        data = {f: M.fold_frame(f) for f in M.FOLDS}
        for f in M.FOLDS:
            ap = data[f]
            fit = pd.concat([data[g] for g in M.FOLDS if g != f], ignore_index=True)
            y = ap["y_true"].to_numpy("float64")
            cl = S.clusters(ap)
            ref = S.fit_blend2(fit, ap)
            flat = S.flat4(ap)
            reg = ap["regime"].to_numpy()
            row = dict(member=tag, fold=f, blend2=S.rmse(y, ref), flat4=S.rmse(y, flat),
                       winter_alone=S.rmse(y, np.maximum(ap["D1-winter"].to_numpy("float64"), B.FLOOR)))
            for lab, cols in (("stack-1", V.NOGATE), ("stack-2", V.ALLC)):
                bag = np.maximum(np.mean([V.arm(fit, ap, s, cols)[0] for s in V.SEEDS], axis=0),
                                 B.FLOOR)
                p = np.where(reg == "noac", ref, bag)
                d, lo, hi = S.boot_delta(y, p, ref, cl)
                row[lab] = S.rmse(y, p)
                row[f"{lab}_vs_blend2"] = d
                row[f"{lab}_lo"], row[f"{lab}_hi"] = lo, hi
            recs.append(row)
            print(f"[{tag}] {f}: blend2 {row['blend2']:.3f}  stack-1 {row['stack-1']:.3f}  "
                  f"stack-2 {row['stack-2']:.3f}  (winter alone {row['winter_alone']:.2f})", flush=True)
    M.OOF["D1-winter"] = old
    T = pd.DataFrame(recs)
    T.to_csv(OUT / f"swap_deice{M.TAG}.csv", index=False)
    print("\n=== board estimates ===")
    for tag in T.member.unique():
        s = T[T.member == tag]
        for col in ("blend2", "flat4", "stack-1", "stack-2"):
            be = M.board_estimate({f: s[s.fold == f][col].iloc[0] for f in M.SHARES})
            print(f"  {tag:26s} {col:8s} {be:.3f}")
    print("\n=== stack gain over blend2, day-clustered 95 % ===")
    for r in T.itertuples():
        print(f"  {r.member:26s} {r.fold:9s} stack-1 {r._asdict()["stack-1_vs_blend2"]:+7.3f} "
              f"[{r._asdict()["stack-1_lo"]:+7.3f}, {r._asdict()["stack-1_hi"]:+7.3f}]   "
              f"stack-2 {r._asdict()["stack-2_vs_blend2"]:+7.3f} "
              f"[{r._asdict()["stack-2_lo"]:+7.3f}, {r._asdict()["stack-2_hi"]:+7.3f}]")


if __name__ == "__main__":
    M.config_from_argv()
    main()
