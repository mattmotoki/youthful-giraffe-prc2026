"""Stage stack-5's served prediction vector. The serve side of `stack5.py`.

`stack5.py` measures stack-5 on the three folds. This builds the object that
would be uploaded, following `exp-016/code/submit4.py` step for step:

  anchor at serve time   blend-5's per-regime hybrid NNLS fitted on ALL THREE
                         folds' OOF, applied to the members' RANKING predictions
  stacker                fitted on the three folds' ROTATED anchors, so its
                         training anchor is out of sample on every row
  noac rows              fall back to the anchor, as stack-1..4 did
  guard                  floor then the residual guard on the served vector

The rotated anchors come from `oof/stack5-{fold}.parquet` (`y_pred_anchor`), so
this script and the rotation cannot disagree about what stack-5 is anchored on.

Like `rebuild_blend.py` it stops short of `prc.data.write_submission`: it writes
`candidates/stack-5.parquet` with `MVT_ID_mvt` and `y_pred`, and `finalise.py`
turns that into `submissions/candidates/exp-026-stack-5.parquet` once the
template exists. The one raw-data dependency is the gates table
(`code/regime_table.py`).

    python serve_stack5.py [--variant keep-noac|full]
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
import rule as R  # noqa: E402
import stack5 as S  # noqa: E402
sys.path.insert(0, str(F.ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

#: The variant that ships. `stack5.py` measures both; keep-noac wins on july and
#: december and ties on january, the same ordering stack-4 found.
DEFAULT_VARIANT = "keep-noac"
GUARD = F.GUARD


def ranking_frame(names, gates: pd.DataFrame) -> pd.DataFrame:
    """Members over the ranking rows, prepared exactly as `build_candidate` does."""
    base = None
    for n in names:
        d = pd.read_parquet(F.ROOT / F.RANKING[n])
        col = "y_pred" if "y_pred" in d.columns else D.TARGET
        d = d[["MVT_ID_mvt", col]].rename(columns={col: n})
        d["MVT_ID_mvt"] = d["MVT_ID_mvt"].astype("float64")
        base = d if base is None else base.merge(d, on="MVT_ID_mvt", how="left")
    m = base.merge(gates[["MVT_ID_mvt", "regime", "d_takeoff_aobt"]],
                   on="MVT_ID_mvt", how="inner")
    if len(m) != len(gates):
        raise ValueError(f"members cover {len(m)} of {len(gates)} ranking departures")
    a = m["d_takeoff_aobt"].to_numpy("float64")
    touched = bound = 0
    for n in names:
        v = m[n].to_numpy("float64")
        ok = np.isfinite(v)
        before = v.copy()
        v[ok] = np.maximum(v[ok], B.FLOOR)
        r = a[ok] - v[ok]
        bound += int(((r < GUARD[0]) | (r > GUARD[1])).sum())
        v[ok] = B.apply_residual_guard(v[ok], a[ok], *GUARD, B.FLOOR)
        touched += int(np.sum(v[ok] != before[ok]))
        m[n] = v
    m.attrs["member_prep_touched"] = touched
    m.attrs["member_prep_bound"] = bound
    return m


def build(gates: pd.DataFrame, variant: str = DEFAULT_VARIANT) -> dict:
    """Build stack-5's served vector. The whole serve path, in one callable.

    `main` writes what this returns; `verify.py` calls it again and compares,
    so the gate that lets a candidate into `submissions/candidates/` runs the
    real path rather than a paraphrase of it.  Returns the ids, the prediction,
    the anchor, the regimes and the diagnostics `main` records.
    """
    names = S.B5
    data = {f: F.fold_frame(f, names) for f in F.FOLDS}
    fit = pd.concat([data[f] for f in F.FOLDS], ignore_index=True)

    # The fitting anchor: the rotation's own, read back rather than recomputed.
    # Equal lengths are NOT enough -- a per-row misalignment between the saved
    # rotation and the fold frames would pass a length check silently and
    # mistrain every stacker. The ids are compared element for element.
    saved = [pd.read_parquet(F.EXP / "oof" / f"stack5-{f}.parquet") for f in F.FOLDS]
    anchor_fit = np.concatenate([s["y_pred_anchor"].to_numpy("float64") for s in saved])
    if len(anchor_fit) != len(fit):
        raise SystemExit("the saved rotation does not match the fold frames; rerun stack5.py")
    saved_ids = np.concatenate([s["MVT_ID_mvt"].to_numpy("float64") for s in saved])
    if not np.array_equal(saved_ids, fit["MVT_ID_mvt"].to_numpy("float64")):
        raise SystemExit("oof/stack5-*.parquet is not row-aligned with the fold frames; "
                         "rerun stack5.py")
    saved_y = np.concatenate([s["y_true"].to_numpy("float64") for s in saved])
    if not np.array_equal(saved_y, fit["y_true"].to_numpy("float64")):
        raise SystemExit("oof/stack5-*.parquet labels disagree with the fold frames")

    rk = ranking_frame(names, gates)
    W = R.fit_hybrid(fit, names)
    anchor_rk = R.apply_hybrid(rk, names, W)
    bag = np.maximum(np.mean([S.arm(fit, rk, s, anchor_fit, anchor_rk, names)
                              for s in S.SEEDS], axis=0), B.FLOOR)
    reg = rk["regime"].to_numpy()
    pred = np.where(reg == "noac", anchor_rk, bag) if variant == "keep-noac" else bag
    pred = np.maximum(pred, B.FLOOR)
    a = rk["d_takeoff_aobt"].to_numpy("float64")
    before = pred.copy()
    # How often the guard truly BINDS -- the residual outside its bounds -- as
    # opposed to how many values merely change. `apply_residual_guard` computes
    # `a - clip(a - y)`, which returns a different float from `y` even when the
    # clip does nothing, so a change count is mostly round-trip noise and reads
    # as an effect that is not there.
    ok_a = np.isfinite(a)
    r_before = a[ok_a] - before[ok_a]
    n_bind = int(((r_before < GUARD[0]) | (r_before > GUARD[1])).sum())
    pred = B.apply_residual_guard(pred, a, *GUARD, B.FLOOR)
    return {"ids": rk["MVT_ID_mvt"].to_numpy("float64"), "pred": pred,
            # the same vector one step earlier: floored, not yet guarded.
            # `tests/serve_check.py` uses it as the control that gives the guard
            # invariant something to fail on.
            "pred_unguarded": before,
            "anchor": anchor_rk, "regime": reg, "d_takeoff_aobt": a,
            "weights": W, "names": names, "variant": variant,
            "guard_bound_output": n_bind,
            "guard_changed_output": int((pred != before).sum()),
            "guard_max_change_output": float(np.abs(pred - before).max()),
            "guard_bound_members": int(rk.attrs["member_prep_bound"]),
            "guard_changed_members": int(rk.attrs["member_prep_touched"]),
            "n_noac_on_anchor": int((reg == "noac").sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=DEFAULT_VARIANT, choices=("keep-noac", "full"))
    ap.add_argument("--name", default="stack-5")
    # A shape test cannot use the real gates table, which does not exist until
    # the data lands, and must not write into candidates/. Both are arguments so
    # `tests/shape_check.py` can exercise every line of this file without
    # leaving an object that looks like a candidate.
    ap.add_argument("--gates", default=None, help="gates table (default: the cached one)")
    ap.add_argument("--out-dir", default=None, help="where to stage (default: candidates/)")
    a = ap.parse_args()
    gates = RT.load(Path(a.gates)) if a.gates else RT.load()
    b = build(gates, a.variant)
    names, pred, reg = b["names"], b["pred"], b["regime"]
    print(f"[serve] members prepared, guard+floor touched "
          f"{b['guard_changed_members']} member-values, of which "
          f"{b['guard_bound_members']} truly bind", flush=True)
    print(f"[serve] anchor weights nm_late "
          f"{ {k: round(v, 4) for k, v in b['weights']['nm_late'].items()} }", flush=True)
    guard_touched = b["guard_changed_output"]

    outdir = Path(a.out_dir) if a.out_dir else F.EXP / "candidates"
    out = outdir / f"{a.name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"MVT_ID_mvt": b["ids"],
                  "y_pred": pred.astype("float64")}).to_parquet(out, index=False)
    rot = json.loads((F.EXP / "analysis" / "stack5.json").read_text())
    rec = {"name": a.name, "path": str(out), "sha256": D.sha256(out),
           "n": int(len(pred)), "members": list(names),
           "variant": a.variant, "arm": rot["arm"], "anchor": rot["anchor"],
           "rotated_clean": rot["rotated_clean"].get(f"stack-5 {a.variant}"),
           "clean_board": rot["clean_board"].get(f"stack-5 {a.variant}"),
           "predict": rot["predict"].get(f"stack-5 {a.variant}"),
           "min": float(pred.min()), "median": float(np.median(pred)),
           "p99": float(np.quantile(pred, 0.99)), "max": float(pred.max()),
           "n_below_floor": int((pred < B.FLOOR - 1e-9).sum()),
           "n_nonfinite": int((~np.isfinite(pred)).sum()),
           "guard_bound_output": b["guard_bound_output"],
           "guard_changed_output": guard_touched,
           "guard_max_change_output": b["guard_max_change_output"],
           "guard_bound_members": b["guard_bound_members"],
           "guard_changed_members": b["guard_changed_members"],
           "n_noac_on_anchor": b["n_noac_on_anchor"],
           "member_sha256": {n: D.sha256(F.ROOT / F.RANKING[n]) for n in names}}
    (outdir / f"{a.name}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
