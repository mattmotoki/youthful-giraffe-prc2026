"""Turn the measured window construction into a submission candidate. One command.

    .venv/bin/python experiments/exp-028-window/serve.py

The candidate is **stack-5 with the window block inside its own stacker**, served
on every regime `analysis/choice.json` names — for the shipping file that is
`nm_bulk`, `nm_late` and `nonm`, everything except the `noac` rows stack-5 already
takes from the anchor.  It is not an arm bolted on top of stack-5: `RESULTS.md` §3
measures the second-layer arm at +0.002 s and the one-layer construction at ten
times that, because a residual model on a residual stacker pays a refit cost the
window columns only buy back.  The regime restriction is `analysis/choice.json`,
measured by `code/restrict.py`.

Every row outside those regimes is served with **exp-026's own stack-5
candidate, byte for byte** — the file the desk would otherwise upload — so this
candidate is literally stack-5 with a substitution on part of the file, and the
two cannot disagree anywhere the block is not served.  The block side goes
through `exp-026`'s serving code wherever that code exists:
`serve_stack5.ranking_frame` prepares the members, `rule.fit_hybrid` fits the
anchor, `stack5.SEEDS/ROUNDS/LGB_BASE` set the stacker.  Nothing about stack-5
is reimplemented here.

What it needs:

  `data/raw/ranking.parquet`, `data/raw/submitting.parquet`
  `experiments/exp-026-rebuild/candidates/ranking-gates.parquet`
      (`exp-026/code/regime_table.py`; built automatically if absent)
  the committed member out-of-fold and ranking parquets, which are in git.

The block and the window width come from `analysis/choice.json`, which
`code/stack5w.py` writes.  The `--blocks/--regimes/--win` overrides exist for the
staged variants, and since Sol round 1 (Standards 8) a configuration with no
rotation record and no restriction record is refused rather than written with
empty metadata: `--unmeasured` is the deliberate escape hatch, and it stamps the
record so a file built that way cannot be mistaken for a measured one.
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
import features as Fe  # noqa: E402
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

DEST = P.ROOT / "submissions" / "candidates"
WORK = P.EXP / "work"
CHOICE = P.EXP / "analysis" / "choice.json"


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="exp-028-window")
    ap.add_argument("--blocks", default=None, help="override, e.g. depc or dep+arr")
    ap.add_argument("--regimes", default=None, help="override, e.g. nm_bulk+nm_late")
    ap.add_argument("--win", type=float, default=None, help="override the window width")
    ap.add_argument("--base", default=str(DEST / "exp-026-stack-5.parquet"),
                    help="stack-5's served vector; every row outside --regimes is taken from it")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--unmeasured", action="store_true",
                    help="write a configuration that has no rotation and no restriction "
                         "record; the served record is stamped measured=null, unmeasured=true")
    a = ap.parse_args()

    if not CHOICE.exists() and not a.blocks:
        raise SystemExit(f"{CHOICE.relative_to(P.ROOT)} is absent; run\n"
                         f"    PRC_VARIANTS=... .venv/bin/python "
                         f"experiments/exp-028-window/code/stack5w.py\n"
                         f"and write the variant it selects into choice.json, or pass --blocks")
    spec = json.loads(CHOICE.read_text()) if CHOICE.exists() else {}
    if a.blocks:
        spec = dict(spec, blocks=a.blocks.split("+"), win=spec.get("win", Fe.WIN))
    if a.regimes:
        spec["regimes"] = a.regimes.split("+")
    if a.win:
        spec["win"] = a.win
    spec.setdefault("win", Fe.WIN)
    spec.setdefault("regimes", ["nm_bulk"])
    names = S.B5

    # Sol round 1, Standards 8: the overrides could write a file whose numbers
    # nothing had measured, and the missing records became empty metadata rather
    # than an error.  Refuse that unless it is asked for in as many words.
    v0, tag0 = "+".join(spec["blocks"]), f"w{int(spec['win'])}"
    have_rot = (P.EXP / "analysis" / f"stack5w-{tag0}.json").exists()
    have_rst = (P.EXP / "analysis" / f"restrict_{v0}_{tag0}.json").exists()
    if not (have_rot and have_rst) and not a.unmeasured:
        raise SystemExit(
            f"no measurement for blocks={v0} win={int(spec['win'])}: "
            f"analysis/stack5w-{tag0}.json {'is present' if have_rot else 'is absent'}, "
            f"analysis/restrict_{v0}_{tag0}.json "
            f"{'is present' if have_rst else 'is absent'}.\n"
            f"Run code/stack5w.py at that width and code/restrict.py for that block, "
            f"or pass --unmeasured to write it anyway.")

    base_path = Path(a.base)
    if not base_path.exists():
        raise SystemExit(
            f"{base_path} is absent. Build stack-5's served vector first:\n"
            f"    .venv/bin/python experiments/exp-026-rebuild/code/regime_table.py\n"
            f"    .venv/bin/python experiments/exp-026-rebuild/code/serve_stack5.py\n"
            f"    .venv/bin/python experiments/exp-026-rebuild/finalise.py --name stack-5\n"
            f"which is the blend lane's own serving path; this experiment substitutes into "
            f"it and never rebuilds it.")

    if not RT.OUT.exists():
        print("[serve] building the gates table (exp-026/code/regime_table.py)", flush=True)
        RT.build()
    gates = RT.load()

    # ---- 1. the fold side: the same frames and anchors the rotation used ----
    data = {f: F.fold_frame(f, names) for f in F.FOLDS}
    fit = pd.concat([data[f] for f in F.FOLDS], ignore_index=True)
    anchor_fit = np.concatenate([
        pd.read_parquet(F.EXP / "oof" / f"stack5-{f}.parquet")["y_pred_anchor"]
        .to_numpy("float64") for f in F.FOLDS])
    if len(anchor_fit) != len(fit):
        raise SystemExit("the saved rotation does not match the fold frames; rerun stack5.py")
    # Sol round 1, Standards 5: length is not alignment.  The saved rotation is
    # written in fold order, so its ids and labels must equal the frames' own.
    oof_ids = np.concatenate([pd.read_parquet(F.EXP / "oof" / f"stack5-{f}.parquet")
                              ["MVT_ID_mvt"].to_numpy("float64") for f in F.FOLDS])
    if not np.array_equal(oof_ids, fit["MVT_ID_mvt"].to_numpy("float64")):
        raise SystemExit("the saved rotation's ids are not the fold frames' ids, in order")
    Xf = pd.concat([SW.block_for(f, data[f], spec) for f in F.FOLDS], ignore_index=True)

    # ---- 2. the ranking side -----------------------------------------------
    rk = SS.ranking_frame(names, gates)
    W = R.fit_hybrid(fit, names)
    anchor_rk = R.apply_hybrid(rk, names, W)
    Xa = SW.ranking_blocks(rk, spec)
    if len(Xa) != len(rk) or list(Xa.columns) != list(Xf.columns):
        raise SystemExit(f"block mismatch: fit {Xf.shape} against serve {Xa.shape}")
    print(f"[serve] {len(rk)} ranking rows, block {'+'.join(spec['blocks'])} "
          f"{Xa.shape[1]} columns, win {spec['win']:.0f}s", flush=True)

    bag = np.maximum(np.mean([SW.arm(fit, rk, s, anchor_fit, anchor_rk, names, Xf, Xa)
                              for s in S.SEEDS], axis=0), B.FLOOR)
    reg = rk["regime"].to_numpy()
    # the block vector gets stack-5's own output steps before it is substituted,
    # so both sides of the substitution have passed the same floor and guard
    before = bag.copy()
    bag = B.apply_residual_guard(np.maximum(bag, B.FLOOR),
                                 rk["d_takeoff_aobt"].to_numpy("float64"), *F.GUARD, B.FLOOR)
    guard_touched = int((bag != before).sum())

    s5 = pd.read_parquet(base_path)
    base = pd.Series(s5[D.TARGET].to_numpy("float64"),
                     index=s5["MVT_ID_mvt"].to_numpy("float64")) \
        .reindex(rk["MVT_ID_mvt"].to_numpy("float64")).to_numpy()
    if not np.isfinite(base).all():
        raise SystemExit(f"{int((~np.isfinite(base)).sum())} ranking rows are not in "
                         f"{base_path.name}")
    on = np.isin(reg, spec["regimes"])
    pred = np.where(on, bag, base)
    if not np.isfinite(pred).all():
        raise SystemExit(f"{int((~np.isfinite(pred)).sum())} non-finite predictions")
    # the rows outside the served regimes must be stack-5 exactly, not nearly
    off = ~on
    if off.any() and float(np.abs(pred[off] - base[off]).max()) != 0.0:
        raise SystemExit("rows outside the served regimes differ from stack-5")
    sp = P.EXP / "analysis" / f"stack5w-w{int(spec['win'])}.json"
    rot = json.loads(sp.read_text()) if sp.exists() else {}
    v = "+".join(spec["blocks"])
    tag = f"w{int(spec['win'])}"
    rp = P.EXP / "analysis" / f"restrict_{v}_{tag}.json"
    rst = json.loads(rp.read_text()) if rp.exists() else []
    rname = {"nm_bulk": "bulk", "nm_bulk+nm_late": "bulk+late",
             "nm_bulk+nonm": "bulk+nonm",
             "nm_bulk+nm_late+nonm": "all-but-noac"}.get("+".join(spec["regimes"]))
    rec = {"name": a.name, "blocks": spec["blocks"], "win": spec["win"],
           "regimes": spec["regimes"], "base": str(base_path.relative_to(P.ROOT)),
           "base_sha256": D.sha256(base_path),
           "n_served_by_block": int(on.sum()), "n_served_by_stack5": int(off.sum()),
           "measured": next((r for r in rst if r["variant"] == rname), None),
           "unmeasured": bool(a.unmeasured and not (rot and rst)),
           "members": list(names), "variant": "keep-noac",
           "arm": f"{S.ROUNDS} rounds, {len(S.SEEDS)}-seed bag, member columns + the block",
           "anchor": "blend-5 per-regime hybrid NNLS fitted on all three folds",
           # the rotation of the UNRESTRICTED block; `measured` above is the row
           # for the regime set actually served, and is the one that describes
           # this file.  Named apart so the two can never be read as one number.
           "unrestricted_rotated_clean": rot.get("clean", {}).get(f"stack-5+{v} keep-noac"),
           "unrestricted_clean_board": rot.get("clean_board", {}).get(f"stack-5+{v} keep-noac"),
           "unrestricted_gain_per_fold": rot.get("gain_per_fold", {}).get(v),
           "unrestricted_three_folds": rot.get("three_folds", {}).get(v),
           "n": int(len(pred)), "n_cols_block": int(Xa.shape[1]),
           "lightgbm": __import__("lightgbm").__version__,
           "min": float(pred.min()), "median": float(np.median(pred)),
           "p99": float(np.quantile(pred, 0.99)), "max": float(pred.max()),
           "n_below_floor": int((pred < B.FLOOR - 1e-9).sum()),
           "guard_touched_block": guard_touched,
           "n_noac_on_anchor": int((reg == "noac").sum()),
           "gates_sha256": D.sha256(RT.OUT),
           "member_sha256": {n: D.sha256(P.ROOT / F.RANKING[n]) for n in names}}
    rec["vs_stack5_rms"] = float(np.sqrt(np.mean((pred - base) ** 2)))
    rec["vs_stack5_n_over_60s"] = int((np.abs(pred - base) > 60).sum())
    rec["vs_stack5_corr"] = float(np.corrcoef(pred, base)[0, 1])
    rec["wall_s"] = round(time.time() - t0, 1)

    if a.dry_run:
        print(json.dumps(rec, indent=2))
        print("[serve] dry run: nothing written into submissions/")
        return
    WORK.mkdir(parents=True, exist_ok=True)
    work = WORK / f"{a.name}.parquet"
    rec["sha256"] = D.write_submission(
        pd.Series(pred, index=rk["MVT_ID_mvt"].to_numpy("float64")), work)
    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / f"{a.name}.parquet"
    work.replace(out)
    if D.sha256(out) != rec["sha256"]:
        raise SystemExit("the file changed between the check and the move")
    rec["candidate"] = str(out.relative_to(P.ROOT))
    (P.EXP / "analysis" / f"served-{a.name}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
