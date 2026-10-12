"""§7bk: promote a hedge fit's raw OOF into `oof/` in the member contract.

    .venv/bin/python work/promote_hedge.py <member> <source.parquet>

Normalises to `MVT_ID_mvt, fold, y_true, y_pred` -- the contract the Kaggle
kernel wrote for F1-fresh and the one `prc.blend.stack_members` reads -- and runs
the same four checks every vector in `oof/` has passed: 162,332 rows, the id set
identical to this lane's 2025-11 block, nothing lost through `stack_members`, no
duplicates or nulls.  Refuses to write if any check fails.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
import prc.blend as B                                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lane                                                  # noqa: E402
lane.install()
FOLD = lane.FOLD
CONTRACT = ["MVT_ID_mvt", "fold", "y_true", "y_pred"]
BLOCK = ROOT / f"artifacts/exp-047/block/{lane.MONTH}.parquet"


def main() -> None:
    member, src = sys.argv[1], Path(sys.argv[2])
    d = pd.read_parquet(src)
    d = d[[c for c in CONTRACT if c in d.columns]].copy()
    d["fold"] = FOLD
    d = d[CONTRACT]
    dst = EXP / "oof" / f"{member}--{FOLD}.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(dst, index=False)

    blk = pd.read_parquet(BLOCK, columns=["MVT_ID_mvt"])
    ids_new = set(d["MVT_ID_mvt"].astype("float64"))
    ids_blk = set(blk["MVT_ID_mvt"].astype("float64"))
    try:
        frame, _ = B.stack_members({member: str(dst)}, ROOT, fold=FOLD)
        ok, lost = True, int(len(d) - len(frame))
    except Exception as e:                                   # noqa: BLE001
        ok, lost = f"{type(e).__name__}: {e}", None
    r = d["y_true"] - d["y_pred"]
    checks = {"rows_equal_block": len(d) == len(blk),
              "ids_identical_to_the_block": ids_new == ids_blk,
              "stack_members_ok": ok is True,
              "no_rows_lost": lost == 0,
              "no_duplicates": int(len(d) - d["MVT_ID_mvt"].nunique()) == 0,
              "no_y_pred_nulls": int(d["y_pred"].isna().sum()) == 0}
    rec = {"member": member, "from": str(src), "path": str(dst.relative_to(ROOT)),
           "rows": int(len(d)), "bytes": dst.stat().st_size,
           "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
           "rmse": round(float((r ** 2).mean() ** 0.5), 4) if r.notna().any() else None,
           "rows_lost_to_the_join": lost, "stack_members": ok, "checks": checks,
           "PASS": all(checks.values())}
    print(json.dumps(rec, indent=1, default=str))
    if not rec["PASS"]:
        dst.unlink()
        raise SystemExit(f"{member}: checks failed, {dst.name} removed")
    print(f"{member} promoted  rows {rec['rows']:,}  rmse {rec['rmse']}  "
          f"sha {rec['sha256'][:16]}")


if __name__ == "__main__":
    main()
