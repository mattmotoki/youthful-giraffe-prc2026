"""Turn one measured arm into a submission candidate. One command.

    .venv/bin/python experiments/exp-030-window2/serve.py --arm <arm> --name exp-030-<arm>

Two modes, because the arms are of two kinds.

**Fitted** (`--blocks`): the arm has its own stacker, so the whole of exp-028's
serving path runs — `serve_stack5.ranking_frame` prepares the members,
`rule.fit_hybrid` fits the blend-5 anchor on all three folds, `stack5w.arm` fits
the stacker with `stack5.SEEDS/ROUNDS/LGB_BASE`, and the block comes from
`blocks.ranking_blocks`, which is the same function the rotation fitted on.

**Derived** (`--from <candidate>`): the arm is a substitution on an existing
candidate and refits nothing, which is what arm 3 is. The block-side vector is
read from that candidate's own bytes.

In both modes the file is assembled the same way: `pred = block side where the
mask says so, else the base candidate's own bytes`. The mask is the served
regimes minus any excluded airport, so the 388 `noac` rows — and every row of an
excluded airport — are **stack-5's bytes, bit for bit**.

A configuration nothing measured is refused: `--arm` must name an
`analysis/arm-<arm>.json` written by `code/score.py`, and **every input that
decides a byte of the output** must be the one that record was measured on — the
blocks and width in fitted mode, and in derived mode the `--from` and `--base`
files **by SHA-256** as well as the served regimes and the excluded airports.
Sol round 1 (blocker 1) found the first version checking only the blocks and the
airport, which would have let a command attach a measured arm's score and PASS to
different candidate bytes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
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

EXP = HERE
DEST = P.ROOT / "submissions" / "candidates"
WORK = EXP / "work"
ALL_BUT_NOAC = ["nm_bulk", "nm_late", "nonm"]


def ranking_airports(ids: np.ndarray) -> np.ndarray:
    """The departure airport of every ranking row, in `ids` order (`ADEP_mvt`)."""
    raw = D.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt",
                                  "MVT_TIME_UTC_mvt"])
    dep = raw[raw["PHASE_mvt"] == "DEP"]
    s = pd.Series(dep["ADEP_mvt"].astype(str).to_numpy(),
                  index=dep["MVT_ID_mvt"].astype("float64").to_numpy())
    out = s.reindex(ids)
    if out.isna().any():
        raise SystemExit(f"{int(out.isna().sum())} rows have no departure airport")
    return out.to_numpy()


def fitted_block_side(spec: dict, rk: pd.DataFrame) -> tuple[np.ndarray, int, int]:
    """exp-028's serving path, with this experiment's multi-width block."""
    names = S.B5
    data = {f: F.fold_frame(f, names) for f in F.FOLDS}
    fit = pd.concat([data[f] for f in F.FOLDS], ignore_index=True)
    oof = [pd.read_parquet(F.EXP / "oof" / f"stack5-{f}.parquet") for f in F.FOLDS]
    anchor_fit = np.concatenate([d["y_pred_anchor"].to_numpy("float64") for d in oof])
    oof_ids = np.concatenate([d["MVT_ID_mvt"].to_numpy("float64") for d in oof])
    if len(anchor_fit) != len(fit) or not np.array_equal(
            oof_ids, fit["MVT_ID_mvt"].to_numpy("float64")):
        raise SystemExit("the saved rotation's ids are not the fold frames' ids, in order")
    Xf = pd.concat([B2.block_for(f, data[f], spec) for f in F.FOLDS], ignore_index=True)
    W = R.fit_hybrid(fit, names)
    anchor_rk = R.apply_hybrid(rk, names, W)
    Xa = B2.ranking_blocks(rk, spec)
    if len(Xa) != len(rk) or list(Xa.columns) != list(Xf.columns):
        raise SystemExit(f"block mismatch: fit {Xf.shape} against serve {Xa.shape}")
    print(f"[serve] block {'+'.join(spec['blocks'])} {Xa.shape[1]} columns, "
          f"default win {spec['win']:.0f}s", flush=True)
    bag = np.maximum(np.mean([SW.arm(fit, rk, s, anchor_fit, anchor_rk, names, Xf, Xa)
                              for s in S.SEEDS], axis=0), B.FLOOR)
    before = bag.copy()
    bag = B.apply_residual_guard(np.maximum(bag, B.FLOOR),
                                 rk["d_takeoff_aobt"].to_numpy("float64"), *F.GUARD, B.FLOOR)
    return bag, int((bag != before).sum()), int(Xa.shape[1])


def derived_block_side(path: Path, ids: np.ndarray) -> np.ndarray:
    d = pd.read_parquet(path)
    v = pd.Series(d[D.TARGET].to_numpy("float64"),
                  index=d["MVT_ID_mvt"].to_numpy("float64")).reindex(ids).to_numpy()
    if not np.isfinite(v).all():
        raise SystemExit(f"{int((~np.isfinite(v)).sum())} ranking rows are not in {path.name}")
    return v


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="the measured arm, analysis/arm-<arm>.json")
    ap.add_argument("--name", default=None, help="candidate name, default exp-030-<arm>")
    ap.add_argument("--blocks", default=None, help="fitted mode, e.g. depx+depx_w300")
    ap.add_argument("--win", type=float, default=600.0)
    ap.add_argument("--from", dest="src", default=None,
                    help="derived mode: the candidate the block side is read from")
    ap.add_argument("--regimes", default="+".join(ALL_BUT_NOAC))
    ap.add_argument("--exclude-airports", default="", help="e.g. EHAM")
    ap.add_argument("--base", default=str(DEST / "exp-026-stack-5.parquet"),
                    help="every row the mask does not serve is this file's own bytes")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    name = a.name or f"exp-030-{a.arm}"
    if a.src:
        a.src = str(Path(a.src).resolve())
    a.base = str(Path(a.base).resolve())

    rp = EXP / "analysis" / f"arm-{a.arm}.json"
    if not rp.exists():
        raise SystemExit(f"{rp.relative_to(P.ROOT)} is absent: this arm has no measurement.\n"
                         f"Run code/rotate.py and code/measure.py (or code/eham.py) first.")
    measured = json.loads(rp.read_text())
    if bool(a.blocks) == bool(a.src):
        raise SystemExit("give exactly one of --blocks (fitted) and --from (derived)")
    spec = {"blocks": a.blocks.split("+"), "win": a.win} if a.blocks else None
    if spec and measured.get("blocks") != spec["blocks"]:
        raise SystemExit(f"the record was measured on blocks={measured.get('blocks')}, "
                         f"not {spec['blocks']}")
    excl = [x for x in a.exclude_airports.split("+") if x]
    if sorted(excl) != sorted([measured["airport_excluded"]] if measured.get("airport_excluded")
                              else []):
        raise SystemExit(f"the record excludes {measured.get('airport_excluded')!r}, "
                         f"the command excludes {excl}")
    base_path, regimes = Path(a.base), a.regimes.split("+")
    if sorted(regimes) != sorted(measured.get("regimes", ALL_BUT_NOAC)):
        raise SystemExit(f"the record was measured serving {measured.get('regimes')}, "
                         f"the command serves {regimes}")
    con = measured.get("serve")
    if a.src and not con:
        raise SystemExit(f"{rp.name} carries no serving contract; a derived candidate may "
                         f"only be built from an arm whose record pins its inputs")
    if con:
        want = {"from": (a.src, con["from"], con["from_sha256"]),
                "base": (str(base_path), con["base"], con["base_sha256"])}
        for what, (given, path, sha) in want.items():
            if Path(given).resolve() != (P.ROOT / path).resolve():
                raise SystemExit(f"--{what} is {given}, the record pins {path}")
            if D.sha256(given) != sha:
                raise SystemExit(f"--{what} {path} is not the file the arm was measured "
                                 f"against: sha256 {D.sha256(given)[:16]} against {sha[:16]}")
        if sorted(excl) != sorted(con["exclude_airports"]):
            raise SystemExit(f"the contract excludes {con['exclude_airports']}, "
                             f"the command excludes {excl}")

    if not base_path.exists():
        raise SystemExit(f"{base_path} is absent; build stack-5's served vector first "
                         f"(experiments/exp-026-rebuild/code/serve_stack5.py)")
    if not RT.OUT.exists():
        print("[serve] building the gates table (exp-026/code/regime_table.py)", flush=True)
        RT.build()
    rk = SS.ranking_frame(S.B5, RT.load())
    ids = rk["MVT_ID_mvt"].to_numpy("float64")

    if spec:
        block, guard_touched, ncols = fitted_block_side(spec, rk)
    else:
        block, guard_touched, ncols = derived_block_side(Path(a.src), ids), 0, 0

    base = derived_block_side(base_path, ids)
    apt = ranking_airports(ids)
    on = np.isin(rk["regime"].to_numpy(), regimes) & ~np.isin(apt, excl)
    pred = np.where(on, block, base)
    if not np.isfinite(pred).all():
        raise SystemExit(f"{int((~np.isfinite(pred)).sum())} non-finite predictions")
    off = ~on
    if off.any() and float(np.abs(pred[off] - base[off]).max()) != 0.0:
        raise SystemExit("rows the mask does not serve differ from the base")

    rec = {"name": name, "arm": a.arm, "mode": "fitted" if spec else "derived",
           "blocks": spec["blocks"] if spec else None,
           "win": spec["win"] if spec else None,
           "source": str(Path(a.src).relative_to(P.ROOT)) if a.src else None,
           "source_sha256": D.sha256(a.src) if a.src else None,
           "regimes": regimes, "exclude_airports": excl,
           "base": str(base_path.relative_to(P.ROOT)), "base_sha256": D.sha256(base_path),
           "n_served_by_block": int(on.sum()), "n_from_base": int(off.sum()),
           "n_noac": int((rk["regime"].to_numpy() == "noac").sum()),
           "n_at_excluded_airport": int(np.isin(apt, excl).sum()),
           "members": list(S.B5), "variant": "keep-noac",
           "arm_construction": f"{S.ROUNDS} rounds, {len(S.SEEDS)}-seed bag, "
                               f"member columns + the block" if spec else "no refit",
           "anchor": "blend-5 per-regime hybrid NNLS fitted on all three folds",
           "n_cols_block": ncols, "guard_touched_block": guard_touched,
           "n": int(len(pred)), "min": float(pred.min()), "median": float(np.median(pred)),
           "p99": float(np.quantile(pred, 0.99)), "max": float(pred.max()),
           "n_below_floor": int((pred < B.FLOOR - 1e-9).sum()),
           "lightgbm": __import__("lightgbm").__version__,
           "gates_sha256": D.sha256(RT.OUT),
           "member_sha256": {n: D.sha256(P.ROOT / F.RANKING[n]) for n in S.B5},
           "measured": {k: measured[k] for k in
                        ("clean", "clean_board", "vs_v6", "gain_board_vs_v6",
                         "predicted_board", "gate")},
           "vs_stack5_rms": float(np.sqrt(np.mean((pred - base) ** 2))),
           "vs_stack5_n_over_60s": int((np.abs(pred - base) > 60).sum()),
           "wall_s": round(time.time() - t0, 1)}
    if a.dry_run:
        print(json.dumps(rec, indent=2))
        print("[serve] dry run: nothing written into submissions/")
        return
    WORK.mkdir(parents=True, exist_ok=True)
    work = WORK / f"{name}.parquet"
    rec["sha256"] = D.write_submission(pd.Series(pred, index=ids), work)
    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / f"{name}.parquet"
    work.replace(out)
    if D.sha256(out) != rec["sha256"]:
        raise SystemExit("the file changed between the check and the move")
    rec["candidate"] = str(out.relative_to(P.ROOT))
    (EXP / "analysis" / f"served-{name}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
