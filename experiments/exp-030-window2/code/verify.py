"""Independent verification of a staged candidate, the way exp-028 §8e verifies its own.

The served vector is **rebuilt from the record the candidate itself carries**
(`analysis/served-<name>.json`) by straight-line code that does not call
`serve.py`, and the maximum absolute difference against the file on disk must be
exactly 0.

**What that 0 proves and what it does not** — exp-028 Sol round 1, Standards 3,
restated because it applies unchanged here, and corrected by this experiment's own
Sol round 1 (blocker 2). For a **derived** candidate (arm 3) the rebuild shares no
model code with `serve.py`: the record names two candidate files, a regime set and
an airport, and the rebuild is `np.where` over their committed bytes. It is not
literally "nothing shared but pandas" — both sides read the same
`ranking-gates.parquet` for the regime and the same raw ranking file for the
airport. So all three inputs are **pinned by SHA-256 and checked before the
rebuild runs** (the two candidates and the gates table), the row order is taken
from `submitting.parquet` rather than from `serve_stack5.ranking_frame`, and what
remains shared is `prc.data`'s parquet reading. For a **fitted** candidate the rebuild calls the same
`blocks.block_for`, `blocks.ranking_blocks` and `stack5w.arm` that `serve.py`
calls, so a 0 proves deterministic replay and correct file assembly and **cannot**
detect a defect inside a shared function; what covers those is exp-028's
`tests/panel_vs_raw.py` (the fitted and served features are the same function,
checked against the real months), the id-order check on the saved rotation, and
the reproduction gate that aborts unless the no-window rotation lands on exp-026's
published stack-5.

Also checked, against `data/raw/submitting.parquet` rather than against an
assumption: the id set is the template's in the template's order; every value is
finite, float64 and at or above `prc.blend.FLOOR`; the **388 `noac` rows are
`exp-026-stack-5.parquet`'s own bytes, bit for bit**; every row the mask does not
serve is likewise; and the recorded SHA-256 is the file's SHA-256 now.

    python verify.py [name ...]        # default: every served-*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP))
import blocks as B2  # noqa: E402
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


def _airports(ids: np.ndarray) -> np.ndarray:
    raw = D.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    dep = raw[raw["PHASE_mvt"] == "DEP"]
    return pd.Series(dep["ADEP_mvt"].astype(str).to_numpy(),
                     index=dep["MVT_ID_mvt"].astype("float64").to_numpy()) \
        .reindex(ids).to_numpy()


def _on_ids(path: Path, ids: np.ndarray) -> np.ndarray:
    d = pd.read_parquet(path)
    return pd.Series(d[D.TARGET].to_numpy("float64"),
                     index=d["MVT_ID_mvt"].to_numpy("float64")).reindex(ids).to_numpy()


def rebuild(rec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(prediction, ids, served-by-block mask, regime), from the record alone.

    Every input the record names is checked against the SHA-256 the record
    carries **before** it is read (Sol round 1, blocker 2).
    """
    names = tuple(rec["members"])
    for what, path, sha in (("base", P.ROOT / rec["base"], rec["base_sha256"]),
                            ("gates", RT.OUT, rec["gates_sha256"])):
        if D.sha256(path) != sha:
            raise SystemExit(f"{what} {path} is not the file the record was built from")
    gates = RT.load()
    if rec["mode"] == "derived":
        # no `serve_stack5.ranking_frame`: the ids are the template's own order and
        # the regime comes straight from the gates table.
        ids = D.load_submitting()[D.ID].to_numpy("float64")
        reg = pd.Series(gates["regime"].to_numpy(),
                        index=gates["MVT_ID_mvt"].astype("float64").to_numpy()) \
            .reindex(ids)
        if reg.isna().any():
            raise SystemExit(f"{int(reg.isna().sum())} template rows have no regime")
        reg = reg.to_numpy()
        src = P.ROOT / rec["source"]
        if D.sha256(src) != rec["source_sha256"]:
            raise SystemExit(f"{rec['source']} is not the file the record was built from")
        block = _on_ids(src, ids)
        base = _on_ids(P.ROOT / rec["base"], ids)
        apt = _airports(ids)
        on = np.isin(reg, rec["regimes"]) & ~np.isin(apt, rec["exclude_airports"])
        return np.where(on, block, base), ids, on, reg
    rk = SS.ranking_frame(names, gates)
    ids = rk["MVT_ID_mvt"].to_numpy("float64")
    if rec["mode"] == "fitted":
        spec = {"blocks": rec["blocks"], "win": rec["win"]}
        data = {f: F.fold_frame(f, names) for f in F.FOLDS}
        fit = pd.concat([data[f] for f in F.FOLDS], ignore_index=True)
        a_fit = np.concatenate([pd.read_parquet(F.EXP / "oof" / f"stack5-{f}.parquet")
                                ["y_pred_anchor"].to_numpy("float64") for f in F.FOLDS])
        Xf = pd.concat([B2.block_for(f, data[f], spec) for f in F.FOLDS], ignore_index=True)
        a_rk = R.apply_hybrid(rk, names, R.fit_hybrid(fit, names))
        Xa = B2.ranking_blocks(rk, spec)
        bag = np.mean([SW.arm(fit, rk, s, a_fit, a_rk, names, Xf, Xa) for s in S.SEEDS], axis=0)
        block = B.apply_residual_guard(np.maximum(bag, B.FLOOR),
                                       rk["d_takeoff_aobt"].to_numpy("float64"),
                                       *F.GUARD, B.FLOOR)
    else:
        raise SystemExit(f"unknown mode {rec['mode']!r}")
    base = _on_ids(P.ROOT / rec["base"], ids)
    apt = _airports(ids)
    reg = rk["regime"].to_numpy()
    on = np.isin(reg, rec["regimes"]) & ~np.isin(apt, rec["exclude_airports"])
    return np.where(on, block, base), ids, on, reg


def check(name: str) -> dict:
    rec = json.loads((EXP / "analysis" / f"served-{name}.json").read_text())
    path = P.ROOT / rec["candidate"]
    got = pd.read_parquet(path)
    tpl = D.load_submitting()
    out = {"name": name, "path": rec["candidate"], "mode": rec["mode"], "n": int(len(got))}

    out["ids_match_template_in_order"] = bool(np.array_equal(
        got["MVT_ID_mvt"].to_numpy("float64"), tpl[D.ID].to_numpy("float64")))
    v = got[D.TARGET].to_numpy()
    out["dtype"] = str(v.dtype)
    out["all_finite"] = bool(np.isfinite(v).all())
    out["min"] = float(v.min())
    out["n_below_floor"] = int((v < B.FLOOR - 1e-9).sum())
    out["sha256"] = D.sha256(path)
    out["sha256_matches_record"] = bool(out["sha256"] == rec["sha256"])

    pred, ids, on, reg = rebuild(rec)
    order = pd.Index(ids).get_indexer(got["MVT_ID_mvt"].to_numpy("float64"))
    if (order < 0).any():
        raise SystemExit(f"{name}: the rebuild is missing {int((order < 0).sum())} ids")
    out["rebuild_max_abs_diff"] = float(np.abs(pred[order] - v).max())

    b = _on_ids(P.ROOT / rec["base"], got["MVT_ID_mvt"].to_numpy("float64"))
    off = ~on[order]
    noac = reg[order] == "noac"
    out["n_noac"] = int(noac.sum())
    out["noac_max_abs_diff_vs_stack5"] = float(np.abs(v[noac] - b[noac]).max())
    out["n_outside_mask"] = int(off.sum())
    out["outside_max_abs_diff_vs_stack5"] = (float(np.abs(v[off] - b[off]).max())
                                             if off.any() else 0.0)
    out["vs_stack5_rms"] = float(np.sqrt(np.mean((v - b) ** 2)))

    out["PASS"] = bool(out["ids_match_template_in_order"] and out["all_finite"]
                       and out["dtype"] == "float64" and out["n_below_floor"] == 0
                       and out["sha256_matches_record"] and out["n"] == 344841
                       and out["n_noac"] == 388
                       and out["rebuild_max_abs_diff"] == 0.0
                       and out["noac_max_abs_diff_vs_stack5"] == 0.0
                       and out["outside_max_abs_diff_vs_stack5"] == 0.0)
    return out


def main() -> None:
    names = sys.argv[1:] or sorted(p.stem.removeprefix("served-")
                                   for p in (EXP / "analysis").glob("served-*.json"))
    recs = [check(n) for n in names]
    for r in recs:
        print(json.dumps(r, indent=2))
    (EXP / "analysis" / "verify.json").write_text(json.dumps(recs, indent=2))
    if not all(r["PASS"] for r in recs):
        raise SystemExit("verification FAILED")
    print(f"\n[verify] {len(recs)} candidate(s) PASS")


if __name__ == "__main__":
    main()
