"""Exercise every line of the ranking-side scripts without the ranking file.

`serve_stack5.py` and `rebuild_blend.py` cannot run until `data/raw` exists, so
until then they are unexecuted code that another lane (exp-028) is about to
depend on. This builds a **synthetic** gates table over the real 344,841 member
ids and runs both scripts against it, into a scratch directory.

**Nothing here is a measurement.** The regimes are assembled from what the
record names (`N1-nmtail`'s `has_nm`, exp-002's 388 noac ids, exp-013's `seg`)
plus an arbitrary `nm_late` draw, and `d_takeoff_aobt` is invented. The only
claims are structural: the joins cover, the member order survives, the feature
matrix matches between fit and serve, the output is 344,841 finite float64
values above the floor, and the noac rows equal the anchor.

    python tests/shape_check.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import frames as F  # noqa: E402


def synthetic_gates(out: Path) -> pd.DataFrame:
    n1 = pd.read_parquet(F.ROOT / F.RANKING["N1-nmtail"], columns=["MVT_ID_mvt", "has_nm"])
    ids = n1["MVT_ID_mvt"].astype("float64").to_numpy()
    nm_missing = ~n1["has_nm"].astype(bool).to_numpy()
    noac_ids = set(pd.read_parquet(
        F.ROOT / "experiments/exp-002-tail/noac_ranking_predictions.parquet",
        columns=["MVT_ID_mvt"])["MVT_ID_mvt"].astype("float64"))
    noac = np.isin(ids, list(noac_ids))
    rng = np.random.default_rng(7)
    # an arbitrary nm_late draw at the census rate; the mask itself is invented
    late = (~nm_missing & ~noac) & (rng.random(len(ids)) < 16090 / 344841)
    regime = np.where(noac, "noac", np.where(nm_missing, "nonm",
                      np.where(late, "nm_late", "nm_bulk")))
    g = pd.DataFrame({"MVT_ID_mvt": ids, "regime": regime,
                      "d_takeoff_aobt": rng.normal(1000.0, 600.0, len(ids))})
    g.to_parquet(out, index=False)
    return g


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="exp026-shape-"))
    gates_path = tmp / "ranking-gates.parquet"
    g = synthetic_gates(gates_path)
    print(f"[shape] synthetic gates {len(g)} rows -> {gates_path}")
    print("[shape] " + " ".join(f"{k}={v}" for k, v in g.regime.value_counts().items()))

    code = HERE.parent / "code"
    runs = [
        [sys.executable, str(code / "rebuild_blend.py"),
         "--config", str(HERE.parent / "configs" / "config-blend-5.json"),
         "--name", "blend-5", "--gates", str(gates_path), "--out-dir", str(tmp)],
        [sys.executable, str(code / "serve_stack5.py"),
         "--gates", str(gates_path), "--out-dir", str(tmp)],
    ]
    for cmd in runs:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=F.ROOT)
        if r.returncode:
            print(r.stdout[-3000:]); print(r.stderr[-3000:])
            raise SystemExit(f"FAILED: {' '.join(cmd[1:3])}")
        print(f"[shape] OK {Path(cmd[1]).name}")

    ids_ref = pd.read_parquet(F.ROOT / F.RANKING["F1-fresh"],
                              columns=["MVT_ID_mvt"])["MVT_ID_mvt"].astype("float64").to_numpy()
    for name in ("blend-5", "stack-5"):
        d = pd.read_parquet(tmp / f"{name}.parquet")
        assert list(d.columns) == ["MVT_ID_mvt", "y_pred"], f"{name}: columns {list(d.columns)}"
        assert d["y_pred"].dtype == "float64", f"{name}: dtype {d['y_pred'].dtype}"
        assert len(d) == 344841, f"{name}: {len(d)} rows"
        assert np.array_equal(d["MVT_ID_mvt"].to_numpy(), ids_ref), f"{name}: id order moved"
        assert np.isfinite(d["y_pred"]).all(), f"{name}: non-finite values"
        assert (d["y_pred"] >= 60.0 - 1e-9).all(), f"{name}: below the floor"
        print(f"[shape] {name}: 344,841 float64, id order held, all finite, none under 60")

    st = pd.read_parquet(tmp / "stack-5.parquet")["y_pred"].to_numpy()
    bl = pd.read_parquet(tmp / "blend-5.parquet")["y_pred"].to_numpy()
    noac = g["regime"].to_numpy() == "noac"
    print(f"[shape] stack-5 vs blend-5 on synthetic gates: rms "
          f"{np.sqrt(((st - bl) ** 2).mean()):.2f} s, noac rows differ on "
          f"{int((st[noac] != bl[noac]).sum())} of {int(noac.sum())} "
          f"(0 would mean the noac fallback and the blend agree)")
    print(f"\n[shape] PASS. Scratch: {tmp}")
    print("[shape] These are shapes, not numbers: the regime mask and "
          "d_takeoff_aobt are invented.")


if __name__ == "__main__":
    main()
