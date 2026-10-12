"""Turn the K2 candidate into a submission file. One command.

    .venv/bin/python experiments/exp-027-step/serve.py

The candidate is the desk's (BOARD READING 7): **stack-5's stacker with the K2
member in F1-fresh's slot, the 600 s window block inside the stacker, and v7's
EHAM mask.**  `code/candidate.py` measures it on the three folds; this serves the
same construction on the 344,841 ranking rows.

It is `experiments/exp-028-window/serve.py` with three differences, and it calls
that file's own functions for everything else:

1. the stacker's member columns are `NAMESK` -- B5 with `K2-window` in
   F1-fresh's slot -- while **the anchor stays blend-5 on the committed seven**,
   which is `candidate.py`'s `k2-hold` reading and is why the 388 `noac` rows
   keep stack-5's bytes;
2. every EHAM departure is served from the base, as v7 does;
3. the K2 ranking column comes from `kernels/f1-window-rank` and is named in
   `PRC_K2_RANK`, so `--identity` can point it back at F1-fresh's own committed
   ranking file.  Under `--identity` this script must reproduce
   `exp-030-eham-out.parquet` byte for byte; that is the serving-path check, and
   it runs on committed bytes alone.

Every row outside the served regimes, and every EHAM row, is `--base` byte for
byte -- exp-026's own stack-5 candidate, the file the desk would otherwise
upload -- so this candidate is stack-5 with a substitution on part of the file.
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
import refit_bulk as RB  # noqa: E402
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
#: Every arm here serves the block on every non-`noac` regime; `noac` is score-feedback rule's.
REGIMES = ["nm_bulk", "nm_late", "nonm"]
#: v7's own file: what `--identity` must reproduce, and the file the K2 candidate
#: replaces on the board.
V7 = DEST / "exp-030-eham-out.parquet"
F1_RANK = "experiments/exp-012-blend2/published/F1-fresh-s1-ranking.parquet"
K2_RANK = "experiments/exp-027-step/candidates/k2-window-ranking.parquet"
#: The measured arms this file is allowed to serve, each with the record that
#: has to agree with the run and the anchor the arm was rotated on.
#:   `blend5`  blend-5 on the committed seven -- the candidate (`RESULTS.md` 16d)
#:   `mixed`   blend-K off the NM tail, blend-5's `nm_late` weights on it, which
#:             is `code/refit_bulk.py`'s repair arm (`RESULTS.md` 16e)
ARMS = {"k2-hold-eham-out": ("candidate-k2.json", "blend5"),
        "k2-refit-bulk-eham-out": ("refit-bulk.json", "mixed")}
#: `code/land_rank.py`'s record: the validated ranking column and its hash.
RANK_REC = HERE / "analysis" / "k2-ranking.json"


def measured_arm(name: str) -> tuple[dict, dict]:
    """The rotation record this serve must match, or a fatal error.

    Sol round 3, blocking 4: `PRC_BLOCK`, `PRC_WIN` and the member set come from
    the environment, so without this the script could write a candidate's name
    over a construction no rotation ever scored.  Everything that decides a
    number is compared against the arm's own record.  `candidate-k2.json` holds
    several arms and carries the configuration once at the top; `refit-bulk.json`
    is one arm and carries its own, so both shapes are read here.
    """
    fname, _ = ARMS[name]
    path = HERE / "analysis" / fname
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} is absent; run the rotation first")
    rec = json.loads(path.read_text())
    if "arms" in rec:
        arm = next((a for a in rec["arms"] if a["arm"] == name), None)
        cfg = rec
    else:
        arm = rec if rec.get("arm") == name else None
        cfg = rec
    if arm is None:
        raise SystemExit(f"{path.name} has no arm {name!r}")
    want = {"block": C.BLOCK, "win": C.WIN, "members": list(C.NAMESK),
            "seeds": list(S.SEEDS), "rounds": S.ROUNDS}
    missing = [k for k in want if k not in cfg]
    if missing:
        raise SystemExit(f"{path.name} does not record {missing}; rerun the rotation")
    bad = {k: (cfg[k], want[k]) for k in want if cfg[k] != want[k]}
    if bad:
        raise SystemExit(f"this run does not match the measured arm: {bad}")
    if arm.get("airport_excluded") != "EHAM" or arm.get("regimes") != REGIMES:
        raise SystemExit(f"{name} was measured with a different mask: "
                         f"{arm.get('airport_excluded')} / {arm.get('regimes')}")
    return rec, arm


def checked_rank_column(identity: bool) -> tuple[str, str]:
    """The ranking file to serve K2 from, and its SHA-256.

    Sol round 3, blocking 2: `serve_stack5.ranking_frame` joins members with a
    left join and LightGBM accepts missing features, so a partial or stale K2
    file would pass every downstream check and still change the candidate.  The
    validation lives in `code/land_rank.py`; this requires that it ran, and that
    the bytes it blessed are the bytes on disk.
    """
    if identity:
        return F1_RANK, D.sha256(ROOT / F1_RANK)
    if not RANK_REC.exists():
        raise SystemExit(f"{RANK_REC.relative_to(ROOT)} is absent; run "
                         f"code/land_rank.py on the kernel's output first")
    r = json.loads(RANK_REC.read_text())
    if r["out"] != K2_RANK:
        raise SystemExit(f"the landed ranking column is {r['out']}, not {K2_RANK}")
    got = D.sha256(ROOT / K2_RANK)
    if got != r["sha256"]:
        raise SystemExit(f"{K2_RANK} has changed since it was validated: "
                         f"{got[:8]} against {r['sha256'][:8]}")
    if r["n"] != 344841:
        raise SystemExit(f"the landed ranking column has {r['n']} rows")
    return K2_RANK, got


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="exp-027-k2")
    ap.add_argument("--base", default=str(DEST / "exp-026-stack-5.parquet"))
    ap.add_argument("--identity", action="store_true",
                    help="serve F1-fresh in K2's slot; the result must equal v7")
    ap.add_argument("--arm", default="k2-hold-eham-out", choices=sorted(ARMS),
                    help="which measured arm to serve; the anchor follows from it")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    anchor_kind = ARMS[a.arm][1]
    rec_rot, rec_arm = ({}, {}) if a.identity else measured_arm(a.arm)
    F.RANKING[C.K2], k2_sha = checked_rank_column(a.identity)
    if a.identity:
        F.OOF[C.K2] = "experiments/exp-012-blend2/published/F1-fresh-s1-{fold}.parquet"
    for d in (F.RANKING[C.K2], F.OOF[C.K2].format(fold="july_loo")):
        if not (ROOT / d).exists():
            raise SystemExit(f"{d} is absent")
    spec = {"blocks": [C.BLOCK], "win": C.WIN}
    regimes = list(REGIMES)
    names, anch = C.NAMESK, C.NAMES5

    if not RT.OUT.exists():
        RT.build()
    gates = RT.load()

    # ---- 1. the fold side: candidate.py's own frames and anchor -------------
    data = C.frames()
    fit = pd.concat([data[f] for f in F.FOLDS], ignore_index=True)
    oof_ids = np.concatenate([pd.read_parquet(F.EXP / "oof" / f"stack5-{f}.parquet")
                              ["MVT_ID_mvt"].to_numpy("float64") for f in F.FOLDS])
    if not np.array_equal(oof_ids, fit["MVT_ID_mvt"].to_numpy("float64")):
        raise SystemExit("the saved rotation's ids are not the fold frames' ids, in order")
    if anchor_kind == "blend5":
        # the rotated blend-5 anchor exp-026 froze, read back rather than refitted
        anchor_fit = np.concatenate([
            pd.read_parquet(F.EXP / "oof" / f"stack5-{f}.parquet")["y_pred_anchor"]
            .to_numpy("float64") for f in F.FOLDS])
    else:
        # `code/refit_bulk.py`'s own rotated mixed anchor, from its own function,
        # so the fit side of the serve is the fit side of the rotation
        am = RB.mixed_anchor(data)
        anchor_fit = np.concatenate([am[f] for f in F.FOLDS])
    Xf = pd.concat([SW.block_for(f, SW.panel_of(data[f]), spec) for f in F.FOLDS],
                   ignore_index=True)

    # ---- 2. the ranking side ------------------------------------------------
    # the ranking frame carries BOTH F1-fresh and K2: the anchor is fitted and
    # applied on the committed seven, the stacker reads K2 in F1-fresh's slot.
    rk = SS.ranking_frame(C.NAMES5 + (C.K2,), gates)
    if anchor_kind == "blend5":
        anchor_rk = R.apply_hybrid(rk, anch, R.fit_hybrid(fit, anch))
    else:
        # the same rule as `refit_bulk.mixed_anchor`, fitted on the three folds
        # pooled instead of rotated, which is what serving does for blend-5 too
        a5 = R.apply_hybrid(rk, C.NAMES5, R.fit_hybrid(fit, C.NAMES5))
        aK = R.apply_hybrid(rk, C.NAMESK, R.fit_hybrid(fit, C.NAMESK))
        _, late = R._split(rk, list(C.NAMESK))
        anchor_rk = np.where(late, a5, aK)
        print(f"[serve] mixed anchor: {int(late.sum())} nm_late rows on blend-5's "
              f"weights, {int((~late).sum())} on blend-K's", flush=True)
    if not np.isfinite(rk[C.K2].to_numpy("float64")).all():
        raise SystemExit(f"{int((~np.isfinite(rk[C.K2].to_numpy('float64'))).sum())} "
                         f"ranking rows have no K2 prediction after the join")
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
    on = np.isin(reg, regimes) & (apt != "EHAM")
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
           "regimes": regimes, "exclude_airports": ["EHAM"],
           "base": str(Path(a.base).relative_to(ROOT)),
           "base_sha256": D.sha256(Path(a.base)),
           "k2_ranking": F.RANKING[C.K2], "k2_ranking_sha256": k2_sha,
           "arm": a.arm, "anchor_kind": anchor_kind,
           "measured_arm": rec_arm.get("arm"),
           "measured_clean": rec_arm.get("clean"),
           "measured_gain_vs_v7": rec_arm.get("gain_board_vs_v7"),
           "rows_served": int(on.sum()),
           "rows_from_base": int(off.sum()), "guard_touched": guard_touched,
           "member_prep_touched": int(rk.attrs.get("member_prep_touched", 0)),
           "seeds": list(S.SEEDS), "rounds": S.ROUNDS,
           "lightgbm": __import__("lightgbm").__version__,
           "wall_s": round(time.time() - t0, 1)}
    if a.dry_run:
        print(json.dumps(rec, indent=2))
        return
    rec["sha256"] = D.write_submission(ser, str(out))
    (HERE / "analysis" / f"served-{a.name}.json").write_text(json.dumps(rec, indent=2))
    print(f"[serve] wrote {out.name}  sha256 {rec['sha256'][:8]}…  "
          f"{rec['rows_served']} served, {rec['rows_from_base']} from the base", flush=True)

    if a.identity:
        if not V7.exists():
            raise SystemExit(f"{V7} is absent; the identity check needs v7's own file")
        got = pd.read_parquet(out)[D.TARGET].to_numpy("float64")
        want = pd.read_parquet(V7)[D.TARGET].to_numpy("float64")
        md = float(np.abs(got - want).max())
        # Not 0.0, and it should not be: v7's own file is arithmetic on two
        # committed candidates while this path REFITS the stacker, and the frame
        # it fits on carries an eighth member column, so the same numbers are
        # summed in a different order.  The tolerance is exp-026's serving
        # tolerance, 1e-9 s, which is 12 orders below the 0.1025 s rotation noise
        # floor and 22 below a second of taxi time.
        tol = 1e-9
        print(f"[identity] max abs difference against {V7.name}: {md:.10g} s "
              f"(tolerance {tol:g}, {'PASS' if md <= tol else 'FAIL'})", flush=True)
        if md > tol:
            raise SystemExit("the serving path does not reproduce v7")


if __name__ == "__main__":
    main()
