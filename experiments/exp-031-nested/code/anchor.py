"""blend-5's own rotation on each column, for the one question the backlog asks.

`docs/BACKLOG.md`'s nested-instrument row asks whether "the stacker's −1.10 s
edge over the blend is real or inflated".  The anchor is fitted and gated inside
every rotation `rotate.py` runs and then thrown away; this script keeps it.  No
LightGBM: `rule.fit_hybrid` / `rule.apply_hybrid` only, so it is seconds, and it
is not a new arm — it is the arm every other arm here is anchored on.

    python anchor.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import instrument as I  # noqa: E402
import rotate as RT  # noqa: E402

F, R, S, B, D = RT.F, RT.R, RT.S, RT.B, RT.D
FOLDS = I.FOLDS


def main() -> None:
    base = I.base_frames(RT.ALL)
    out, clean, allr, ex, bex = {}, {}, {}, {}, {}
    for col in I.COLUMNS:
        clean[col] = {}
        for f in FOLDS:
            data = I.build(col, f, RT.ALL, base)
            y = data[f]["y_true"].to_numpy("float64")
            a5 = np.maximum(RT.anchors(data, RT.C.NAMES5)[f], B.FLOOR)
            # Sol round 2, finding 3: the board is computed from the UNROUNDED
            # fold RMSE; rounding happens once, on the way into the print and
            # the JSON, which also carries the exact values.
            ex.setdefault(col, {})[f] = D.rmse_clean(
                y, a5, data[f]["MVT_ID_mvt"].to_numpy("float64"), f)
            clean[col][f] = round(ex[col][f], 4)
            allr.setdefault(col, {})[f] = round(float(np.sqrt(np.mean((y - a5) ** 2))), 4)
        bex[col] = F.board_estimate({f: ex[col][f] for f in F.SHARES})
        out[col] = round(bex[col], 4)
        print(f"[blend-5 {col:14s}] " + "  ".join(f"{f} {clean[col][f]:9.4f}" for f in FOLDS)
              + f"   board {out[col]:9.4f}")
    # `stack5.PUBLISHED` is `blend-config-5.json`'s ALL-ROWS rotated RMSE, which is
    # what exp-026's own gate compares; the clean figures above drop the rows
    # `prc.data.excluded_ids` names and are a different quantity on january.
    off = {f: round(allr["published"][f] - S.PUBLISHED[f], 4) for f in FOLDS}
    print(f"[gate] published all-rows against blend-config-5.json: {off}")
    if max(abs(v) for v in off.values()) > 1e-3:
        raise SystemExit(f"blend-5 does not reproduce: {off}")

    rot = json.loads((EXP / "analysis" / "rotation.json").read_text())
    print("\n=== the stacker's edge over its own anchor, per column ===")
    rotB = json.loads((EXP / "analysis" / "rotation-nestedanchor.json").read_text())
    edge, edgeB = {}, {}
    for col in I.COLUMNS:
        edge[col] = round(bex[col] - rot["board_exact"][col]["stack-5"], 4)
        # Sol round 2, finding 6: blend-5 has no stacker and so no channel A of
        # its own, but the stacker side of the edge does have one, so the edge is
        # also quoted against the A-closed stack-5.
        edgeB[col] = round(bex[col] - rotB["board_exact"][col]["stack-5"], 4)
        print(f"  {col:14s} blend-5 {out[col]:9.4f}  stack-5 "
              f"{rot['board'][col]['stack-5']:9.4f}  edge {edge[col]:+8.4f}"
              f"   (against A-closed stack-5 {edgeB[col]:+8.4f})")
    (EXP / "analysis" / "anchor.json").write_text(json.dumps(
        {"clean": clean, "clean_exact": ex, "all_rows": allr, "board": out,
         "board_exact": bex, "stack5_minus_blend5": edge,
         "stack5_minus_blend5_anchor_closed": edgeB,
         "gate_published_all_rows": off,
         "note": "blend-5's rotation is the shipped one on every column: it has no "
                 "stacker, so exp-026 §4a's channel A does not arise inside it. "
                 "The edge is quoted against both stack-5 constructions."}, indent=2))
    print("\n  written: analysis/anchor.json")


if __name__ == "__main__":
    main()
