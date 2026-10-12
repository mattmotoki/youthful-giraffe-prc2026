"""Apply a blend config to the published members and stage the prediction vector.

The arithmetic is `prc.blend.build_candidate`'s, call for call, with one change:
where that function ends in `prc.data.write_submission` -- which needs the
submission template -- this one writes `MVT_ID_mvt` and `y_pred` in the members'
own row order.  `finalise.py` turns a staged vector into a real candidate
through `write_submission` once the template exists, and checks it against
`build_candidate`'s own output on the same config.

Splitting it this way is what lets the blend step run the moment the ranking
file lands, without waiting for the template, and lets the fingerprint be
checked (`n`, `min`, `median`, `p99`) before anything is issued.

    python rebuild_blend.py --config ../configs/config-....json --name blend-5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames as F  # noqa: E402
import regime_table as RT  # noqa: E402
sys.path.insert(0, str(F.ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402


def stage(config: dict, name: str, gates: pd.DataFrame | None = None,
          out_dir: Path | None = None) -> dict:
    gates = RT.load() if gates is None else gates
    frame, names = B.stack_members(config["ranking"], F.ROOT)
    m = frame.merge(gates[["MVT_ID_mvt", "regime", "d_takeoff_aobt"]],
                    on="MVT_ID_mvt", how="inner")
    if len(m) != len(gates):
        raise ValueError(f"members cover {len(m)} of {len(gates)} ranking departures")
    if not np.array_equal(m["MVT_ID_mvt"].to_numpy(), frame["MVT_ID_mvt"].to_numpy()):
        raise ValueError("the merge reordered the member rows")
    W = {k: np.array([v[n] for n in names], "float64") for k, v in config["weights"].items()}
    P = m[names].to_numpy("float64").copy()
    a = m["d_takeoff_aobt"].to_numpy("float64")
    floor = float(config.get("floor", B.FLOOR))
    guard = config.get("guard_bounds")
    reg = m["regime"].to_numpy()
    touched_member = 0
    for j, n in enumerate(names):
        bad = ~np.isfinite(P[:, j])
        if bad.any():
            for rg in np.unique(reg[bad]):
                w = np.asarray(W.get(rg, W["_all"]), "float64")[j]
                if w != 0.0:
                    k = int((bad & (reg == rg)).sum())
                    raise ValueError(f"member {n!r} is undefined on {k} {rg} rows but "
                                     f"carries weight {w:.4f} there")
            P[bad, j] = floor
    if config.get("prepare_members", True):
        for j in range(P.shape[1]):
            before = P[:, j].copy()
            P[:, j] = np.maximum(P[:, j], floor)
            if guard:
                P[:, j] = B.apply_residual_guard(P[:, j], a, float(guard[0]), float(guard[1]),
                                                 floor)
            touched_member += int((P[:, j] != before).sum())
    pred = B.apply_weights(P, reg, W)
    pred = np.maximum(pred, floor)
    touched_out = 0
    if guard:
        before = pred.copy()
        pred = B.apply_residual_guard(pred, a, float(guard[0]), float(guard[1]), floor)
        touched_out = int((pred != before).sum())
    out = (out_dir or F.EXP / "candidates") / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"MVT_ID_mvt": m["MVT_ID_mvt"].to_numpy("float64"),
                  "y_pred": pred.astype("float64")}).to_parquet(out, index=False)
    rec = {"name": name, "path": str(out), "sha256": D.sha256(out),
           "n": int(len(pred)), "members": names,
           "min": float(pred.min()), "median": float(np.median(pred)),
           "p99": float(np.quantile(pred, 0.99)), "max": float(pred.max()),
           "n_below_floor": int((pred < floor - 1e-9).sum()),
           "n_nonfinite": int((~np.isfinite(pred)).sum()),
           "guard_touched_members": touched_member, "guard_touched_output": touched_out,
           "regimes": {k: int(v) for k, v in pd.Series(reg).value_counts().items()},
           "member_sha256": {n: D.sha256(F.ROOT / p) for n, p in config["ranking"].items()}}
    out.with_suffix(".json").write_text(json.dumps(rec, indent=2))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--expect", help="JSON with n/min/median/p99 to check against")
    # Both are arguments so `tests/shape_check.py` can exercise this file
    # against a synthetic gates table without writing into candidates/.
    ap.add_argument("--gates", default=None, help="gates table (default: the cached one)")
    ap.add_argument("--out-dir", default=None, help="where to stage (default: candidates/)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    rec = stage(cfg, a.name, RT.load(Path(a.gates)) if a.gates else None,
                Path(a.out_dir) if a.out_dir else None)
    print(json.dumps(rec, indent=2))
    if a.expect:
        want = json.loads(Path(a.expect).read_text())
        bad = {k: (want[k], rec[k]) for k in ("n", "min", "median", "p99")
               if k in want and abs(float(want[k]) - float(rec[k])) > 5e-2}
        # Review round 1: this printed MISMATCH and exited 0, so a caller that
        # tested the exit status was told a bad vector was good.
        if bad:
            raise SystemExit(f"fingerprint MISMATCH {bad}")
        print("fingerprint: MATCH")


if __name__ == "__main__":
    main()
