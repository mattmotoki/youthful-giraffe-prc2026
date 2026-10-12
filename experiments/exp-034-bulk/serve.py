"""Turn the exp-034 candidate into a submission file. One command.

    .venv/bin/python experiments/exp-034-bulk/serve.py

The candidate is `assignments/bulk.md`'s: **K2's stacker with the 64-column bulk
member ADDED, the 600 s window block inside the stacker, and v7's EHAM mask.**
`code/candidate.py` measures it on the three folds; this serves the same
construction on the 344,841 ranking rows.

It is `experiments/exp-027-step/serve.py` with three differences, and it calls
that file's own functions for everything else:

1. the stacker's member columns are `candidate.NAMESB` -- K2's seven with
   `B1-bulk64` added before `N1-nmtail` -- while **the anchor stays blend-5 on
   the committed seven**, which is why the 388 `noac` rows keep stack-5's bytes;
2. the bulk ranking column is PARTIAL and is required to be null on exactly the
   ranking rows where `d_takeoff_aobt` is null (`code/land_rank.py`);
3. the served arm must have PASSED `assignments/bulk.md`'s gate against K2 in
   the rotation record, and this script exits rather than serve one that did not.

Every row outside the served regimes, and every EHAM row, is `--base` byte for
byte -- exp-026's own stack-5 candidate -- so the 388 `noac` and 38,182 EHAM
rows are v7's bytes, which the run checks at the end against v7's own file.
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "code"))
sys.path.insert(0, str(ROOT / "experiments/exp-028-window/code"))
sys.path.insert(0, str(ROOT / "experiments/exp-026-rebuild/code"))
import candidate as C  # noqa: E402
import stack5w as SW  # noqa: E402
import frames as F  # noqa: E402
import regime_table as RT  # noqa: E402
import rule as R  # noqa: E402
import serve_stack5 as SS  # noqa: E402
import stack5 as S  # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

DEST = ROOT / "submissions" / "candidates"
REGIMES = ["nm_bulk", "nm_late", "nonm"]
#: v7's own file: the bytes the `noac` and EHAM rows of this candidate must equal.
V7 = DEST / "exp-030-eham-out.parquet"
K2_RANK = "experiments/exp-027-step/candidates/k2-window-ranking.parquet"
K2_RANK_REC = ROOT / "experiments/exp-027-step/analysis/k2-ranking.json"
BULK_RANK = "experiments/exp-034-bulk/candidates/bulk64-ranking.parquet"
BULK_RANK_REC = HERE / "analysis" / "bulk-ranking.json"
#: The measured arms this file may serve, and the member tuple each carries.
ARMS = {"bulk-add-hold": C.NAMESB, "bulk-add-bulkonly-hold": C.NAMESBO}
ROTATION = HERE / "analysis" / "candidate-bulk.json"


def measured_arm(name: str) -> tuple[dict, dict]:
    """The rotation record this serve must match, or a fatal error.

    Everything that decides a number -- block, window, member set, seeds, rounds
    -- is compared against the arm's own record, and the arm must have passed
    `assignments/bulk.md`'s gate against K2.  Without both, a command could
    write a candidate's name over a construction no rotation ever scored, or
    over one that a rotation scored and rejected (exp-032 Sol round 1).
    """
    if not ROTATION.exists():
        raise SystemExit(f"{ROTATION.relative_to(ROOT)} is absent; run the rotation first")
    rec = json.loads(ROTATION.read_text())
    arm = next((a for a in rec["arms"] if a["arm"] == f"{name}-eham-out"), None)
    if arm is None:
        raise SystemExit(f"{ROTATION.name} has no arm {name!r}")
    # Sol round 1, non-blocking 6: the recorded thread count and LightGBM
    # version also decide a number, and were not checked.  Both are trivially
    # satisfiable (`PRC_THREADS`), so both are hard checks.
    want = {"block": C.BLOCK, "win": C.WIN, "seeds": list(S.SEEDS),
            "rounds": S.ROUNDS, "threads": C.THREADS,
            "lightgbm": __import__("lightgbm").__version__}
    bad = {k: (rec.get(k), v) for k, v in want.items() if rec.get(k) != v}
    if bad:
        raise SystemExit(f"this run does not match the measured arm: {bad}")
    key = "members" if name == "bulk-add-hold" else "members_repair"
    if rec[key] != list(ARMS[name]):
        raise SystemExit(f"the record's {key} {rec[key]} is not {list(ARMS[name])}")
    if rec.get("identity"):
        raise SystemExit("the rotation record is an identity run; it scores no candidate")
    if arm.get("airport_excluded") != "EHAM" or arm.get("regimes") != REGIMES:
        raise SystemExit(f"{name} was measured with a different mask: "
                         f"{arm.get('airport_excluded')} / {arm.get('regimes')}")
    g = arm.get("gate_k2") or {}
    if not g.get("PASS"):
        raise SystemExit(f"{name} FAILED the assignment gate against K2 "
                         f"(gain {g.get('gain_board')}, worst fold {g.get('worst_fold')}); "
                         f"it is not servable")
    if arm.get("exploratory"):
        raise SystemExit(f"{name} is recorded as EXPLORATORY; it was not pre-registered "
                         f"and may not be served")
    # Sol round 1, non-blocking 6: `SPEC.md`'s pre-registration says the repair
    # arm ships ONLY if the shipping arm fails and the repair passes.  Checking
    # the selected arm alone would let the repair be served while the shipping
    # arm also passed, which the registration forbids.
    if name != "bulk-add-hold":
        ship = next((a for a in rec["arms"] if a["arm"] == "bulk-add-hold-eham-out"), None)
        if ship is None:
            raise SystemExit("the record has no bulk-add-hold arm to compare against")
        if (ship.get("gate_k2") or {}).get("PASS"):
            raise SystemExit(f"{name} may not be served: SPEC.md's pre-registration ships "
                             f"the repair only when bulk-add-hold FAILS, and it passed")
    return rec, arm


def checked_rank(path: str, rec_path: Path) -> str:
    """A ranking column may reach a candidate only after its own validator ran."""
    if not rec_path.exists():
        raise SystemExit(f"{rec_path.relative_to(ROOT)} is absent; validate the column first")
    r = json.loads(rec_path.read_text())
    if r["out"] != path:
        raise SystemExit(f"the validated column is {r['out']}, not {path}")
    got = D.sha256(ROOT / path)
    if got != r["sha256"]:
        raise SystemExit(f"{path} has changed since it was validated: "
                         f"{got[:8]} against {r['sha256'][:8]}")
    if r["n"] != 344841:
        raise SystemExit(f"{path} has {r['n']} rows")
    return got


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="exp-034-bulk")
    ap.add_argument("--base", default=str(DEST / "exp-026-stack-5.parquet"))
    ap.add_argument("--arm", default="bulk-add-hold", choices=sorted(ARMS))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rec_rot, rec_arm = measured_arm(a.arm)
    names, anch = ARMS[a.arm], C.NAMES5
    F.RANKING[C.K2] = K2_RANK
    F.RANKING[C.BULK] = BULK_RANK
    F.RANKING[C.BULK_ONLY] = BULK_RANK
    k2_sha = checked_rank(K2_RANK, K2_RANK_REC)
    bulk_sha = checked_rank(BULK_RANK, BULK_RANK_REC)
    spec = {"blocks": [C.BLOCK], "win": C.WIN}

    if not RT.OUT.exists():
        RT.build()
    gates = RT.load()

    # ---- 1. the fold side: candidate.py's own frames and the frozen anchor ---
    data = C.frames()
    fit = pd.concat([data[f] for f in F.FOLDS], ignore_index=True)
    saved = [pd.read_parquet(F.EXP / "oof" / f"stack5-{f}.parquet") for f in F.FOLDS]
    oof_ids = np.concatenate([s["MVT_ID_mvt"].to_numpy("float64") for s in saved])
    if not np.array_equal(oof_ids, fit["MVT_ID_mvt"].to_numpy("float64")):
        raise SystemExit("the saved rotation's ids are not the fold frames' ids, in order")
    anchor_fit = np.concatenate([s["y_pred_anchor"].to_numpy("float64") for s in saved])
    Xf = pd.concat([SW.block_for(f, SW.panel_of(data[f]), spec) for f in F.FOLDS],
                   ignore_index=True)

    # ---- 2. the ranking side ------------------------------------------------
    rk = SS.ranking_frame(C.NAMES5 + (C.K2, C.BULK), gates)
    if a.arm == "bulk-add-bulkonly-hold":
        v = rk[C.BULK].to_numpy("float64").copy()
        v[rk["regime"].to_numpy() != "nm_bulk"] = np.nan
        rk[C.BULK_ONLY] = v
    anchor_rk = R.apply_hybrid(rk, anch, R.fit_hybrid(fit, anch))
    if not np.isfinite(rk[C.K2].to_numpy("float64")).all():
        raise SystemExit(f"{int((~np.isfinite(rk[C.K2].to_numpy('float64'))).sum())} "
                         f"ranking rows have no K2 prediction after the join")
    # the partial member: defined on exactly the rows the guard's argument is
    # defined on, the same contract the three folds carry
    okb = np.isfinite(rk[C.BULK].to_numpy("float64"))
    ta = np.isfinite(rk["d_takeoff_aobt"].to_numpy("float64"))
    if not np.array_equal(okb, ta):
        raise SystemExit(f"the bulk member is defined on {int(okb.sum())} ranking rows and "
                         f"d_takeoff_aobt on {int(ta.sum())}; they must be the same rows")
    Xa = SW.ranking_blocks(rk, spec)
    if len(Xa) != len(rk) or list(Xa.columns) != list(Xf.columns):
        raise SystemExit(f"block mismatch: fit {Xf.shape} against serve {Xa.shape}")
    print(f"[serve] {len(rk)} ranking rows, block {C.BLOCK} {Xa.shape[1]} columns, "
          f"win {C.WIN:.0f}s, members {'/'.join(names)}; bulk member defined on "
          f"{int(okb.sum())} ({okb.mean():.4f})", flush=True)

    bag = np.maximum(np.mean([SW.arm(fit, rk, s, anchor_fit, anchor_rk, names, Xf, Xa)
                              for s in S.SEEDS], axis=0), B.FLOOR)
    before = bag.copy()
    bag = B.apply_residual_guard(np.maximum(bag, B.FLOOR),
                                 rk["d_takeoff_aobt"].to_numpy("float64"), *F.GUARD, B.FLOOR)
    guard_touched = int((bag != before).sum())

    s5 = pd.read_parquet(a.base)
    base = pd.Series(s5[D.TARGET].to_numpy("float64"),
                     index=s5["MVT_ID_mvt"].to_numpy("float64")) \
        .reindex(rk["MVT_ID_mvt"].to_numpy("float64")).to_numpy()
    if not np.isfinite(base).all():
        raise SystemExit("some ranking rows are not in the base")
    apt = gates.set_index("MVT_ID_mvt")["airport"].astype(str) \
        .reindex(rk["MVT_ID_mvt"].to_numpy("float64")).to_numpy()
    reg = rk["regime"].to_numpy()
    on = np.isin(reg, REGIMES) & (apt != "EHAM")
    pred = np.where(on, bag, base)
    off = ~on
    if off.any() and float(np.abs(pred[off] - base[off]).max()) != 0.0:
        raise SystemExit("a held row differs from the base")
    if not np.isfinite(pred).all():
        raise SystemExit("non-finite predictions")
    if float(pred.min()) < B.FLOOR:
        raise SystemExit(f"{int((pred < B.FLOOR).sum())} rows below the floor {B.FLOOR}")

    ser = pd.Series(pred, index=rk["MVT_ID_mvt"].to_numpy("float64"))
    out = DEST / f"{a.name}.parquet"
    rec = {"name": a.name, "arm": a.arm, "members": list(names),
           "anchor_members": list(anch), "blocks": spec["blocks"], "win": spec["win"],
           "regimes": REGIMES, "exclude_airports": ["EHAM"],
           "base": str(Path(a.base).relative_to(ROOT)),
           "base_sha256": D.sha256(Path(a.base)),
           "k2_ranking": K2_RANK, "k2_ranking_sha256": k2_sha,
           "bulk_ranking": BULK_RANK, "bulk_ranking_sha256": bulk_sha,
           "bulk_defined_ranking_rows": int(okb.sum()),
           "measured_arm": rec_arm.get("arm"),
           "measured_clean": rec_arm.get("clean"),
           "measured_gate_k2": rec_arm.get("gate_k2"),
           "measured_gain_vs_k2": rec_arm.get("gain_board_vs_k2"),
           "measured_gain_vs_v7": rec_arm.get("gain_board_vs_v7"),
           "rows_served": int(on.sum()), "rows_from_base": int(off.sum()),
           "guard_touched": guard_touched,
           "member_prep_touched": int(rk.attrs.get("member_prep_touched", 0)),
           "pred_min": float(pred.min()), "pred_max": float(pred.max()),
           "seeds": list(S.SEEDS), "rounds": S.ROUNDS,
           "lightgbm": __import__("lightgbm").__version__,
           "wall_s": round(time.time() - t0, 1)}
    if a.dry_run:
        print(json.dumps(rec, indent=2))
        return
    rec["sha256"] = D.write_submission(ser, str(out))

    # ---- 3. the held rows are v7's, at max abs difference 0.0 ---------------
    # `assignments/bulk.md` step 4.  v7 serves EHAM and `noac` from stack-5 and
    # so does this candidate, so the two files must agree on those rows exactly
    # -- not to a tolerance.  v7 is read from its own committed file.
    if not V7.exists():
        raise SystemExit(f"{V7} is absent; the held-row check needs v7's own file")
    got = pd.read_parquet(out)
    v7 = pd.read_parquet(V7)
    if not np.array_equal(got[D.ID].to_numpy("float64"), v7[D.ID].to_numpy("float64")):
        raise SystemExit("the candidate and v7 are not in the same row order")
    gv, wv = got[D.TARGET].to_numpy("float64"), v7[D.TARGET].to_numpy("float64")
    gid = got[D.ID].to_numpy("float64")
    g2 = gates.set_index("MVT_ID_mvt").reindex(gid)
    held = (g2["regime"].to_numpy() == "noac") | (g2["airport"].astype(str).to_numpy() == "EHAM")
    md = float(np.abs(gv[held] - wv[held]).max())
    rec["held_rows"] = int(held.sum())
    rec["n_noac"] = int((g2["regime"].to_numpy() == "noac").sum())
    rec["n_eham"] = int((g2["airport"].astype(str).to_numpy() == "EHAM").sum())
    rec["held_max_abs_diff_vs_v7"] = md
    rec["rows_changed_vs_v7"] = int((gv != wv).sum())
    rec["rms_move_vs_v7"] = float(np.sqrt(np.mean((gv - wv) ** 2)))
    print(f"[held] {int(held.sum())} rows ({rec['n_noac']} noac, {rec['n_eham']} EHAM) "
          f"against v7: max abs difference {md:.17g}", flush=True)
    if md != 0.0:
        raise SystemExit("the held rows are not v7's bytes")
    (HERE / "analysis" / f"served-{a.name}.json").write_text(json.dumps(rec, indent=2))
    print(f"[serve] wrote {out.name}  sha256 {rec['sha256']}  "
          f"{rec['rows_served']} served, {rec['rows_from_base']} from the base; "
          f"{rec['rows_changed_vs_v7']} rows differ from v7 at RMS "
          f"{rec['rms_move_vs_v7']:.2f} s", flush=True)


if __name__ == "__main__":
    main()
