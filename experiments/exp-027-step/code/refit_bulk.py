"""The repair for `k2-refit`'s tail failure: refit the anchor off the NM tail only.

`code/candidate.py` measures two readings of "the K2 member in F1-fresh's slot".
`k2-refit`, which refits blend-5's anchor with K2 as well, buys 0.108 s more
board than the shipping candidate and **fails `SPEC.md`'s second kill clause**:
`y > 3600` degrades by 2.90 s on july_loo and 1.10 s on january against a 1 s
allowance, while december gains 18.48 s.

`AGENTS.md`: write the repair attempt before writing the FAIL. The diagnosis is
that the anchor refit is a bulk gain bought partly out of the tail regimes, so
this arm refits the anchor **only where the tail is not**. `rule.fit_hybrid`
returns exactly two weight vectors -- `nm_late`, for rows on the NM join where
the tail specialist is defined, and `_all` for everything else -- so "off the
tail" is the `_all` vector and nothing finer is available without changing the
blend rule, which is not this experiment's to change.

  anchor(row) = blend-K's `_all` weights    where the row is not `nm_late`
                blend-5's `nm_late` weights where it is

Everything else is `candidate.py`'s: the same frames, the same 600 s block, the
same five seeds and 300 rounds, `noac` held at blend-5's own anchor, v7's EHAM
mask, and exp-030's scoring and day-clustered intervals.

    PRC_THREADS=3 python refit_bulk.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import candidate as C  # noqa: E402

F, R, S, SW, Sc = C.F, C.R, C.S, C.SW, C.Sc
B, D = C.B, C.D
FOLDS = C.FOLDS


def mixed_anchor(data: dict) -> dict:
    """blend-K off the NM tail, blend-5 on it, fold by fold."""
    out = {}
    for g in FOLDS:
        src = pd.concat([data[h] for h in FOLDS if h != g], ignore_index=True)
        W5 = R.fit_hybrid(src, C.NAMES5)
        WK = R.fit_hybrid(src, C.NAMESK)
        a5 = R.apply_hybrid(data[g], C.NAMES5, W5)
        aK = R.apply_hybrid(data[g], C.NAMESK, WK)
        _, late = R._split(data[g], list(C.NAMESK))
        out[g] = np.where(late, a5, aK)
    return out


def main() -> None:
    data = C.frames()
    a5 = C.anchors(data, C.NAMES5)
    for f in FOLDS:
        y = data[f]["y_true"].to_numpy("float64")
        d = float(np.sqrt(np.mean((y - a5[f]) ** 2))) - S.PUBLISHED[f]
        if abs(d) > C.GATE:
            raise SystemExit(f"gate A fails on {f}: {d:+.4f}")
    print("[refit-bulk] gate A: blend-5 reproduces on all three folds", flush=True)
    ref5 = {f: np.maximum(a5[f], B.FLOOR) for f in FOLDS}
    am = mixed_anchor(data)
    for f in FOLDS:
        y = data[f]["y_true"].to_numpy("float64")
        print(f"[refit-bulk] mixed anchor {f:9s} "
              f"{float(np.sqrt(np.mean((y - am[f]) ** 2))):9.4f}", flush=True)

    blk = {f: SW.block_for(f, SW.panel_of(data[f]), {"blocks": [C.BLOCK], "win": C.WIN})
           for f in FOLDS}
    ref = Sc.reference()
    v7, _ = C.eham_mask({f: ref[f]["y_pred_v6"].to_numpy("float64") for f in FOLDS}, ref)

    p = C.align(C.rotate(data, C.NAMESK, am, blk, ref5), data, ref)
    pe, n = C.eham_mask(p, ref)
    rec = Sc.score("k2-refit-bulk-eham-out", pe, ref,
                   note="anchor refitted with K2 off the NM tail only, blend-5's "
                        "nm_late weights kept; block depx 600 s; v7's EHAM mask; "
                        "noac held at blend-5's anchor")
    rec["airport_excluded"] = "EHAM"
    rec["n_rows_taken_from_stack5"] = n
    Sc.show(rec)
    C.against_v7(rec, pe, ref, v7)
    v7s = {f: D.rmse_clean(ref[f]["y_true"], v7[f], ref[f]["MVT_ID_mvt"], f) for f in FOLDS}
    print("\n  the second kill clause, y > 3600 against v7 (positive = better):")
    ref30 = json.loads((C.EXP.parents[0] / "exp-030-window2/analysis/arm-eham-out.json")
                       .read_text())
    for f in FOLDS:
        got = rec["slices"][f]["gt3600"]
        was = ref30["slices"][f]["gt3600"]
        print(f"    {f:9s} {was:9.3f} -> {got:9.3f}   {was - got:+7.3f} s"
              f"   {'PASS' if was - got > -1.0 else 'FAIL'}")
    rec["v7_clean"] = {f: round(v7s[f], 4) for f in FOLDS}
    # The configuration block `serve.py` binds to, in `candidate-k2.json`'s
    # shape: everything that decides a number, so a served file cannot be
    # attached to an arm it was not built from (Sol round 3, blocking 4).
    rec.update({"members": list(C.NAMESK), "anchor_members": list(C.NAMES5),
                "anchor": "blend-K off the NM tail, blend-5's nm_late weights on it",
                "block": C.BLOCK, "win": C.WIN, "seeds": list(S.SEEDS),
                "rounds": S.ROUNDS, "threads": C.THREADS,
                "lightgbm": __import__("lightgbm").__version__})
    # exp-035-k2climb round 3, blocking 1: this arm has to be runnable on a
    # CLIMBED member, so the outputs carry `candidate.py`'s own `PRC_TAG` the
    # same way its do.  Unset, TAG is "k2" and every name below is what exp-027
    # committed, byte for byte; set, nothing of exp-027's is touched.
    name = "refit-bulk" if C.TAG == "k2" else f"refit-bulk-{C.TAG}"
    (C.EXP / "analysis" / f"{name}.json").write_text(json.dumps(rec, indent=2))
    for f in FOLDS:
        pd.DataFrame({"MVT_ID_mvt": ref[f]["MVT_ID_mvt"].to_numpy("float64"),
                      "fold": f, "y_true": ref[f]["y_true"].to_numpy("float64"),
                      "y_pred": pe[f], "y_pred_nomask": p[f]}).to_parquet(
            C.EXP / "oof" / f"exp-027-{C.TAG}-refit-bulk-{f}.parquet", index=False)
    print(f"\n  written: analysis/{name}.json", flush=True)


if __name__ == "__main__":
    main()
