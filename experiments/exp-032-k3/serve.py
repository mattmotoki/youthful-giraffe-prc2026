"""Turn the K3 candidate into a submission file. One command.

    .venv/bin/python experiments/exp-032-k3/serve.py

The candidate is the assignment's: **K2's stacker with the K3 member in
`T-tailv4`'s slot and the K2 member in `F1-fresh`'s, the 600 s window block
inside the stacker, and v7's EHAM mask.**  `code/candidate.py` measures it on
the three folds; this serves the same construction on the 344,841 ranking rows.

It is `experiments/exp-027-step/serve.py` with two differences, and it calls
that file's imports for everything else:

1. the stacker's member columns are `NAMESK23G` -- B5 with `K2-window` in
   F1-fresh's slot and the gated `K3-tail-gated` in T-tailv4's -- while **the anchor stays
   blend-5 on the committed seven**, which is `candidate.py`'s `k3g-hold`
   reading and is why the 388 `noac` rows keep the bytes v6, v7 and v9 served;
2. there are two substituted ranking columns instead of one, each required to
   be the bytes its own producer's record blessed.

Under `--identity` both columns point back at the committed members and this
script must reproduce `exp-030-eham-out.parquet` (v7) to the serving tolerance;
that is the serving-path check and it runs on committed bytes alone.
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
sys.path.insert(0, str(ROOT / "experiments/exp-028-window"))
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
V7 = DEST / "exp-030-eham-out.parquet"
#: The two substituted slots, and the committed member each replaces under
#: `--identity`.  `K3G` is the shipping member: `code/gate_member.py`'s repair.
SLOTS = (C.K2, C.K3G)
IDENT = {C.K2: ("experiments/exp-012-blend2/published/F1-fresh-s1-ranking.parquet",
                "experiments/exp-012-blend2/published/F1-fresh-s1-{fold}.parquet"),
         C.K3G: ("experiments/exp-012-blend2/published/T-tailv4-ranking.parquet",
                 "experiments/exp-009-noac2/oof/exp-009-noac2-oof-v3.parquet")}
#: Each substituted member's ranking column, and the record that blessed it.
#: exp-027's `code/land_rank.py` wrote the first; `code/gate_member.py --rank`
#: writes the second from `code/rank.py`'s window column and the committed
#: `T-tailv4-ranking.parquet`, which is the serving form of the same gate the
#: fold vectors carry.
RANK = {C.K2: ("experiments/exp-027-step/candidates/k2-window-ranking.parquet",
               ROOT / "experiments/exp-027-step/analysis/k2-ranking.json"),
        C.K3G: ("experiments/exp-032-k3/candidates/k3-tail-gated-ranking.parquet",
                HERE / "analysis" / "rank-gated.json")}
ARM = "k3g-hold-eham-out"
N_RANK = 344841


def measured_arm() -> tuple[dict, dict]:
    """The rotation record this serve must match, or a fatal error.

    exp-027's Sol round 3, blocking 4: the member set, the block and the window
    come from the environment, so without this the script could write a
    candidate's name over a construction no rotation ever scored.
    """
    path = HERE / "analysis" / "candidate-k3.json"
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} is absent; run the rotation first")
    rec = json.loads(path.read_text())
    arm = next((a for a in rec["arms"] if a["arm"] == ARM), None)
    if arm is None:
        raise SystemExit(f"{path.name} has no arm {ARM!r}")
    want = {"block": C.BLOCK, "win": C.WIN, "members": list(C.NAMESK23G),
            "seeds": list(S.SEEDS), "rounds": S.ROUNDS}
    missing = [k for k in want if k not in rec]
    if missing:
        raise SystemExit(f"{path.name} does not record {missing}; rerun the rotation")
    # Sol round 1, blocking 3/4: matching the CONSTRUCTION is not enough. The
    # arm must also have PASSED the gate `assignments/k3.md` sets, or this
    # script would happily serve a measured loser. exp-032's arms all fail it,
    # so this is the check that stops the artifact, not the operator's memory.
    g = arm.get("gate_k2")
    if g is None:
        raise SystemExit(f"{path.name} arm {ARM!r} has no `gate_k2`; rerun the "
                         "rotation with the current code/candidate.py")
    if not g.get("PASS"):
        raise SystemExit(
            f"{ARM!r} FAILS the assignment's gate against K2 "
            f"(board gain {g['gain_board']:+.4f} against a {g['bar_board']:+.2f} bar, "
            f"worst fold {g['worst_fold']:+.4f} against a {g['floor_fold']:+.2f} floor). "
            "Nothing may be served from it.")
    bad = {k: (rec[k], want[k]) for k in want if rec[k] != want[k]}
    if bad:
        raise SystemExit(f"this run does not match the measured arm: {bad}")
    if arm.get("airport_excluded") != "EHAM" or arm.get("regimes") != REGIMES:
        raise SystemExit(f"{ARM} was measured with a different mask: "
                         f"{arm.get('airport_excluded')} / {arm.get('regimes')}")
    return rec, arm


def checked_rank_columns(identity: bool) -> dict:
    """The ranking file to serve each substituted member from, and its SHA-256.

    `serve_stack5.ranking_frame` joins members with a left join and LightGBM
    accepts missing features, so a partial or stale column would pass every
    downstream check and still change the candidate (exp-027 Sol round 3,
    blocking 2).  Each column must be the bytes its own producer's record
    blessed, on 344,841 rows.
    """
    out = {}
    for m, (path, recpath) in RANK.items():
        if identity:
            p = IDENT[m][0]
            out[m] = (p, D.sha256(ROOT / p))
            continue
        if not recpath.exists():
            raise SystemExit(f"{recpath.relative_to(ROOT)} is absent; run the producer first")
        r = json.loads(recpath.read_text())
        if r["out"] != path:
            raise SystemExit(f"{recpath.name} blessed {r['out']}, not {path}")
        got = D.sha256(ROOT / path)
        if got != r["sha256"]:
            raise SystemExit(f"{path} has changed since it was validated: "
                             f"{got[:8]} against {r['sha256'][:8]}")
        if r["n"] != N_RANK:
            raise SystemExit(f"{path} has {r['n']} rows")
        out[m] = (path, got)
    return out


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="exp-032-k3")
    ap.add_argument("--base", default=str(DEST / "exp-026-stack-5.parquet"))
    ap.add_argument("--identity", action="store_true",
                    help="serve the committed members in both slots; the result must equal v7")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rec_rot, rec_arm = ({}, {}) if a.identity else measured_arm()
    cols = checked_rank_columns(a.identity)
    for m, (p, _) in cols.items():
        F.RANKING[m] = p
        if a.identity:
            F.OOF[m] = IDENT[m][1]
    for m in SLOTS:
        for d in (F.RANKING[m], F.OOF[m].format(fold="july_loo")):
            if not (ROOT / d).exists():
                raise SystemExit(f"{d} is absent")
    spec = {"blocks": [C.BLOCK], "win": C.WIN}
    names, anch = C.NAMESK23G, C.NAMES5

    if not RT.OUT.exists():
        RT.build()
    gates = RT.load()

    # ---- 1. the fold side: candidate.py's own frames and anchor -------------
    data = C.frames()
    fit = pd.concat([data[f] for f in F.FOLDS], ignore_index=True)
    saved = [pd.read_parquet(F.EXP / "oof" / f"stack5-{f}.parquet") for f in F.FOLDS]
    oof_ids = np.concatenate([s["MVT_ID_mvt"].to_numpy("float64") for s in saved])
    if not np.array_equal(oof_ids, fit["MVT_ID_mvt"].to_numpy("float64")):
        raise SystemExit("the saved rotation's ids are not the fold frames' ids, in order")
    # the rotated blend-5 anchor exp-026 froze, read back rather than refitted
    anchor_fit = np.concatenate([s["y_pred_anchor"].to_numpy("float64") for s in saved])
    Xf = pd.concat([SW.block_for(f, SW.panel_of(data[f]), spec) for f in F.FOLDS],
                   ignore_index=True)

    # ---- 2. the ranking side ------------------------------------------------
    # the ranking frame carries the committed seven AND both substitutes: the
    # anchor is fitted and applied on the committed seven, the stacker reads the
    # substitutes in their own slots.
    rk = SS.ranking_frame(C.NAMES5 + SLOTS, gates)
    anchor_rk = R.apply_hybrid(rk, anch, R.fit_hybrid(fit, anch))
    for m in SLOTS:
        bad = int((~np.isfinite(rk[m].to_numpy("float64"))).sum())
        if bad:
            raise SystemExit(f"{bad} ranking rows have no {m} prediction after the join")
    Xa = SW.ranking_blocks(rk, spec)
    if len(Xa) != len(rk) or list(Xa.columns) != list(Xf.columns):
        raise SystemExit(f"block mismatch: fit {Xf.shape} against serve {Xa.shape}")
    print(f"[serve] {len(rk)} ranking rows, block {C.BLOCK} {Xa.shape[1]} columns, "
          f"win {C.WIN:.0f}s, members {'/'.join(names)}", flush=True)

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

    ser = pd.Series(pred, index=rk["MVT_ID_mvt"].to_numpy("float64"))
    out = DEST / f"{a.name}.parquet"
    rec = {"name": a.name, "identity": bool(a.identity), "members": list(names),
           "anchor_members": list(anch), "blocks": spec["blocks"], "win": spec["win"],
           "regimes": REGIMES, "exclude_airports": ["EHAM"],
           "base": str(Path(a.base).relative_to(ROOT)),
           "base_sha256": D.sha256(Path(a.base)),
           "ranking_columns": {m: {"path": p, "sha256": h} for m, (p, h) in cols.items()},
           "arm": ARM, "anchor_kind": "blend5",
           "measured_arm": rec_arm.get("arm"),
           "measured_clean": rec_arm.get("clean"),
           "measured_gain_vs_v7": rec_arm.get("gain_board_vs_v7"),
           "measured_gain_vs_k2": rec_arm.get("gain_board_vs_k2"),
           "rows_served": int(on.sum()),
           "rows_from_base": int(off.sum()), "guard_touched": guard_touched,
           "member_prep_touched": int(rk.attrs.get("member_prep_touched", 0)),
           "seeds": list(S.SEEDS), "rounds": S.ROUNDS,
           "lightgbm": __import__("lightgbm").__version__,
           "wall_s": round(time.time() - t0, 1)}
    if a.dry_run:
        print(json.dumps(rec, indent=2))
        return
    if a.identity:
        out = DEST / f"{a.name}-identity.parquet"
    rec["sha256"] = D.write_submission(ser, str(out))
    (HERE / "analysis" / f"served-{out.stem}.json").write_text(json.dumps(rec, indent=2))
    print(f"[serve] wrote {out.name}  sha256 {rec['sha256'][:8]}…  "
          f"{rec['rows_served']} served, {rec['rows_from_base']} from the base", flush=True)

    if a.identity:
        if not V7.exists():
            raise SystemExit(f"{V7} is absent; the identity check needs v7's own file")
        got = pd.read_parquet(out)[D.TARGET].to_numpy("float64")
        want = pd.read_parquet(V7)[D.TARGET].to_numpy("float64")
        md = float(np.abs(got - want).max())
        # Not 0.0, and it should not be: v7's own file is arithmetic on two
        # committed candidates while this path REFITS the stacker on a frame
        # carrying two extra member columns, so the same numbers are summed in a
        # different order.  exp-026's serving tolerance, 12 orders below the
        # 0.1025 s rotation noise floor.
        tol = 1e-9
        print(f"[identity] max abs difference against {V7.name}: {md:.10g} s "
              f"(tolerance {tol:g}, {'PASS' if md <= tol else 'FAIL'})", flush=True)
        if md > tol:
            raise SystemExit("the serving path does not reproduce v7")


if __name__ == "__main__":
    main()
