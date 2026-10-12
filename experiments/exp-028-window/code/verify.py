"""Independent verification of a staged candidate, the way exp-026 verifies stack-5.

`finalise.py` refuses to issue a blend candidate whose values differ from
`prc.blend.build_candidate`'s by more than 0.  This is the same discipline for a
candidate that has no second implementation in the library: the served vector is
**rebuilt here, from the record the candidate itself carries**
(`analysis/served-<name>.json`), by straight-line code that does not call
`serve.py`, and the maximum absolute difference against the file on disk must be
exactly 0.

Also checked, against `data/raw/submitting.parquet` rather than against an
assumption:

  * the id set is the template's, in the template's order;
  * every value is finite, float64, and at or above `prc.blend.FLOOR`;
  * the rows outside the served regimes are stack-5's own bytes, bit for bit;
  * the recorded SHA-256 is the file's SHA-256 now.

    python verify.py [name ...]        # default: every served-*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import panel as P  # noqa: E402
import stack5w as SW  # noqa: E402
R26 = P.ROOT / "experiments/exp-026-rebuild/code"
sys.path.insert(0, str(R26))
import frames as F  # noqa: E402
import regime_table as RT  # noqa: E402
import rule as R  # noqa: E402
import serve_stack5 as SS  # noqa: E402
import stack5 as S  # noqa: E402
sys.path.insert(0, str(P.ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402


def rebuild(rec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(prediction, MVT_ID_mvt, served-by-block mask), from the record alone."""
    names = tuple(rec["members"])
    spec = {"blocks": rec["blocks"], "win": rec["win"]}
    gates = RT.load()
    data = {f: F.fold_frame(f, names) for f in F.FOLDS}
    fit = pd.concat([data[f] for f in F.FOLDS], ignore_index=True)
    a_fit = np.concatenate([pd.read_parquet(F.EXP / "oof" / f"stack5-{f}.parquet")
                            ["y_pred_anchor"].to_numpy("float64") for f in F.FOLDS])
    Xf = pd.concat([SW.block_for(f, data[f], spec) for f in F.FOLDS], ignore_index=True)
    rk = SS.ranking_frame(names, gates)
    a_rk = R.apply_hybrid(rk, names, R.fit_hybrid(fit, names))
    Xa = SW.ranking_blocks(rk, spec)
    bag = np.mean([SW.arm(fit, rk, s, a_fit, a_rk, names, Xf, Xa) for s in S.SEEDS], axis=0)
    bag = B.apply_residual_guard(np.maximum(np.maximum(bag, B.FLOOR), B.FLOOR),
                                 rk["d_takeoff_aobt"].to_numpy("float64"), *F.GUARD, B.FLOOR)
    base = pd.read_parquet(P.ROOT / rec["base"])
    b = pd.Series(base[D.TARGET].to_numpy("float64"),
                  index=base["MVT_ID_mvt"].to_numpy("float64")) \
        .reindex(rk["MVT_ID_mvt"].to_numpy("float64")).to_numpy()
    on = np.isin(rk["regime"].to_numpy(), rec["regimes"])
    return np.where(on, bag, b), rk["MVT_ID_mvt"].to_numpy("float64"), on


def check(name: str) -> dict:
    rec = json.loads((P.EXP / "analysis" / f"served-{name}.json").read_text())
    path = P.ROOT / rec["candidate"]
    got = pd.read_parquet(path)
    tpl = D.load_submitting()
    out = {"name": name, "path": rec["candidate"], "n": int(len(got))}

    # 1. the template contract
    out["ids_match_template_in_order"] = bool(np.array_equal(
        got["MVT_ID_mvt"].to_numpy("float64"), tpl[D.ID].to_numpy("float64")))
    v = got[D.TARGET].to_numpy()
    out["dtype"] = str(v.dtype)
    out["all_finite"] = bool(np.isfinite(v).all())
    out["min"] = float(v.min())
    out["n_below_floor"] = int((v < B.FLOOR - 1e-9).sum())
    out["sha256"] = D.sha256(path)
    out["sha256_matches_record"] = bool(out["sha256"] == rec["sha256"])

    # 2. the independent rebuild
    pred, ids, on = rebuild(rec)
    order = pd.Index(ids).get_indexer(got["MVT_ID_mvt"].to_numpy("float64"))
    if (order < 0).any():
        raise SystemExit(f"{name}: the rebuild is missing {int((order < 0).sum())} ids")
    out["rebuild_max_abs_diff"] = float(np.abs(pred[order] - v).max())

    # 3. the rows outside the served regimes are stack-5's own bytes
    base = pd.read_parquet(P.ROOT / rec["base"])
    b = pd.Series(base[D.TARGET].to_numpy("float64"),
                  index=base["MVT_ID_mvt"].to_numpy("float64")) \
        .reindex(got["MVT_ID_mvt"].to_numpy("float64")).to_numpy()
    off = ~on[order]
    out["n_outside_regimes"] = int(off.sum())
    out["outside_max_abs_diff_vs_stack5"] = float(np.abs(v[off] - b[off]).max()) if off.any() else 0.0
    out["vs_stack5_rms"] = float(np.sqrt(np.mean((v - b) ** 2)))

    out["PASS"] = bool(out["ids_match_template_in_order"] and out["all_finite"]
                       and out["dtype"] == "float64" and out["n_below_floor"] == 0
                       and out["sha256_matches_record"]
                       and out["rebuild_max_abs_diff"] == 0.0
                       and out["outside_max_abs_diff_vs_stack5"] == 0.0)
    return out


def main() -> None:
    names = sys.argv[1:] or sorted(p.stem.removeprefix("served-")
                                   for p in (P.EXP / "analysis").glob("served-*.json"))
    recs = [check(n) for n in names]
    for r in recs:
        print(json.dumps(r, indent=2))
    (P.EXP / "analysis" / "verify.json").write_text(json.dumps(recs, indent=2))
    if not all(r["PASS"] for r in recs):
        raise SystemExit("verification FAILED")
    print(f"\n[verify] {len(recs)} candidate(s) PASS")


if __name__ == "__main__":
    main()
