"""How much of stack-5's gain comes from the ANCHOR leakage channel?

This closes ONE of the two channels RESULTS.md section 4a describes.  Read that
section before quoting anything here: an earlier draft of this file called the
variant "truly nested" and said it closed the leak completely, which is wrong.

**Channel A, closed here.**  Scoring fold `f`, the stacker's training anchor on
fold `g` is the blend fitted on the other two folds, and "the other two"
includes `f`, so `f`'s labels reach the training residual through 7 NNLS
weights.  The variant here fits fold `g`'s training anchor on the SINGLE
remaining fold, which is neither `f` nor `g`.  The apply-time anchor for `f` is
unchanged -- fitted on the two training folds, which never included `f`.

**Channel B, still open, and not closable on this machine.**  The stacker's
feature columns are the members' own out-of-fold predictions, and those member
models were fitted on month sets that include `f`.  Scoring july_loo, the
stacker trains on december and january rows whose member features come from
models that saw July.  Closing that needs the level-one members refit per
rotation with the scored month held out of every training set -- the training
data and a fitting machine, neither of which is here.  So the number below
prices channel A alone; it is not a clean out-of-sample figure and must not be
quoted as one.

The nested anchor is also a WEAKER anchor (one fold of fitting rows instead of
two), so a drop here is two effects at once and the honest reading is a lower
bound on the gain, not a corrected estimate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames as F  # noqa: E402
import rule as R  # noqa: E402
import stack5 as S  # noqa: E402
sys.path.insert(0, str(F.ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

B5 = S.B5


def main() -> None:
    data = {f: F.fold_frame(f, B5) for f in F.FOLDS}
    # apply-time anchor for each fold: fitted on the other two (unchanged)
    anchor = {g: R.apply_hybrid(data[g], B5,
                                R.fit_hybrid(pd.concat([data[h] for h in F.FOLDS if h != g],
                                                       ignore_index=True), B5))
              for g in F.FOLDS}
    rows = []
    for f in F.FOLDS:
        ap, y = data[f], data[f]["y_true"].to_numpy("float64")
        ids = ap["MVT_ID_mvt"].to_numpy("float64")
        train = [g for g in F.FOLDS if g != f]
        fit = pd.concat([data[g] for g in train], ignore_index=True)
        ref = np.maximum(anchor[f], B.FLOOR)
        reg = ap["regime"].to_numpy()

        # nested: fold g's training anchor is fitted on the OTHER training fold
        nested = {g: R.apply_hybrid(data[g], B5, R.fit_hybrid(data[other], B5))
                  for g, other in zip(train, train[::-1])}
        a_fit_nested = np.concatenate([nested[g] for g in train])
        a_fit_shipped = np.concatenate([anchor[g] for g in train])

        for lab, a_fit in (("shipped anchor", a_fit_shipped),
                           ("nested anchor", a_fit_nested)):
            bag = np.maximum(np.mean([S.arm(fit, ap, s, a_fit, anchor[f], B5)
                                      for s in S.SEEDS], axis=0), B.FLOOR)
            keep = np.where(reg == "noac", ref, bag)
            rows.append(dict(fold=f, variant=lab,
                             rmse_clean=D.rmse_both(y, keep, ids, f)["rmse_clean"]))
        rows.append(dict(fold=f, variant="blend-5 (anchor)",
                         rmse_clean=D.rmse_both(y, ref, ids, f)["rmse_clean"]))
        print(f"[{f}] " + "  ".join(f"{r['variant']}={r['rmse_clean']:.3f}"
                                    for r in rows[-3:]), flush=True)

    T = pd.DataFrame(rows).pivot(index="variant", columns="fold",
                                 values="rmse_clean")[list(F.FOLDS)]
    T["clean board"] = [F.board_estimate({f: T.loc[v, f] for f in F.SHARES}) for v in T.index]
    T["predict"] = T["clean board"] + 4.0
    T["vs blend-5"] = T["clean board"] - T.loc["blend-5 (anchor)", "clean board"]
    print("\n=== stack-5 keep-noac, shipped anchor against anchor-channel-closed ===")
    print(T.round(3).to_string())
    T.round(4).to_csv(F.EXP / "analysis" / "nested_anchor.csv")
    (F.EXP / "analysis" / "nested_anchor.json").write_text(
        json.dumps(json.loads(T.round(4).to_json(orient="index")), indent=2))


if __name__ == "__main__":
    main()
