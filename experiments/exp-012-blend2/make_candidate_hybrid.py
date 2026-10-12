"""Build the hybrid candidate: flat over the core, a fitted specialist in nm_late.

The desk's blend rule for four members is flat unless a fitted vector beats flat
in EVERY rotation (it does not: NNLS wins only december, by 0.001). The NM-tail
head then earns real and stable weight inside `nm_late` alone -- 0.297 / 0.331 /
0.336 across the three rotations -- and exactly 0.000 in `nm_bulk`. So the
shipped vector is:

  nm_late                       five members, NNLS, fitted on all three months
  nm_bulk / nonm / noac         four members flat at 1/4, N1 at exactly 0

N1 is null on non-NM rows, which `prc.blend` VERSION 5 permits precisely because
its weight there is zero, and verifies rather than assumes.

  python make_candidate_hybrid.py <number>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import blend as BL  # noqa: E402
import nmtail_rotate as NR  # noqa: E402
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "exp-001-retro-v1"))
import guard as guard_mod  # noqa: E402

RANKING = {
    "T-tailv2":  "artifacts/exp-012/pred/T-tailv2-ranking.parquet",
    "F1-fresh":  "artifacts/exp-012/pred/F1-fresh-s1-ranking.parquet",
    "C2-cat":    "artifacts/exp-008/pred/C2-d8p-ranking.parquet",
    "L2-hub":    "artifacts/exp-008/pred/L2-hub-ranking.parquet",
    "D1-winter": "artifacts/exp-012/pred/D1-winter-deice-v1-ranking.parquet",
    "N1-nmtail": "artifacts/exp-012/pred/N1-nmtail-ranking.parquet",
}
if os.environ.get("HYBRID_RANKING"):
    RANKING = json.loads((HERE / os.environ["HYBRID_RANKING"]).read_text())["ranking"]
    print(f"[ranking] {os.environ['HYBRID_RANKING']}: " + ", ".join(RANKING), flush=True)

TARGET = "nm_late"


def main(number: str, outer: str = "flat") -> None:
    """`outer` is the rule OUTSIDE the specialist regime. The desk's rule picks
    it per candidate: flat unless a fitted vector beats flat in EVERY rotation.
    blend-3b shipped flat; blend-4 ships nnls, because the fresh member made the
    general members unequal and fitted then wins all three rotations."""
    data = {f: NR.enriched(f, "phi") for f in BL.FOLDS} if hasattr(NR, "enriched") else None
    import nmtail_regime as NG  # noqa: E402  (enriched lives there)
    data = {f: NG.enriched(f, "phi") for f in BL.FOLDS}
    names = data[BL.FOLDS[0]]["names"]
    k = len(names)

    # shipped weights: fitted on ALL three months, the rotation is how the
    # candidate is scored and not how its weights are chosen
    sel = {f: (data[f]["regime"] == TARGET) & data[f]["ok"] for f in BL.FOLDS}
    P5 = np.concatenate([data[f]["P"][sel[f]] for f in BL.FOLDS])
    y5 = np.concatenate([data[f]["y"][sel[f]] for f in BL.FOLDS])
    w_late = B.fit_weights(P5, y5, "nnls")
    w_rest = (np.array([1.0 / (k - 1)] * (k - 1) + [0.0]) if outer == "flat"
              else np.append(B.fit_weights(
                  np.concatenate([data[f]["P"][~sel[f]][:, :k-1] for f in BL.FOLDS]),
                  np.concatenate([data[f]["y"][~sel[f]] for f in BL.FOLDS]), outer), 0.0))
    print(f"nm_late   ({len(y5):,} rows): " + "  ".join(f"{n}={v:.4f}" for n, v in zip(names, w_late)))
    print(f"elsewhere: " + "  ".join(f"{n}={v:.4f}" for n, v in zip(names, w_rest)))

    # rotated score of exactly this rule
    per, per_core = {}, {}
    for f in BL.FOLDS:
        src = [g for g in BL.FOLDS if g != f]
        Pc = np.concatenate([data[g]["P"][sel[g]] for g in src])
        yc = np.concatenate([data[g]["y"][sel[g]] for g in src])
        wl = B.fit_weights(Pc, yc, "nnls")
        d, s = data[f], sel[f]
        p = np.empty(len(d["y"]))
        p[s] = d["P"][s] @ wl
        wo = (np.full(k - 1, 1.0 / (k - 1)) if outer == "flat" else B.fit_weights(
            np.concatenate([data[g]["P"][~sel[g]][:, :k-1] for g in src]),
            np.concatenate([data[g]["y"][~sel[g]] for g in src]), outer))
        p[~s] = d["P"][~s][:, :k-1] @ wo
        p = np.maximum(p, B.FLOOR)
        per[f] = float(np.sqrt(((d["y"] - p) ** 2).mean()))
        pc = np.maximum(d["P"][:, :k-1] @ wo, B.FLOOR)
        per_core[f] = float(np.sqrt(((d["y"] - pc) ** 2).mean()))
    be, be0 = BL.board_estimate(per), BL.board_estimate(per_core)
    print("rotated: " + "  ".join(f"{f} {per[f]:.2f}" for f in BL.FOLDS)
          + f"   board {be:.3f}   (core4 flat {be0:.3f})")

    lo, hi = guard_mod.bounds_from_months(D.MONTHS, 0.0, 1.0)
    W = {TARGET: dict(zip(names, map(float, w_late)))}
    for r in B.REGIMES:
        if r != TARGET:
            W[r] = dict(zip(names, map(float, w_rest)))
    W["_all"] = W["nm_bulk"]
    cfg = {"ranking": RANKING, "mode": f"hybrid: nnls in {TARGET}, {outer} elsewhere",
           "weights": W, "floor": B.FLOOR, "guard_bounds": [lo, hi],
           "late_cut": B.LATE_CUT, "prepare_members": True, "blend_version": B.VERSION,
           "rotated": {f: round(per[f], 4) for f in BL.FOLDS},
           "board_estimate": round(be, 4),
           "core4_flat_board": round(be0, 4)}
    out = ROOT / "submissions" / "candidates" / f"exp-012-blend-{number}.parquet"
    rec = B.build_candidate(cfg, out, ROOT)
    cfg["candidate"] = rec
    (HERE / f"blend-config-{number}.json").write_text(json.dumps(cfg, indent=2))
    print("\n" + json.dumps(rec, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2",
         sys.argv[2] if len(sys.argv) > 2 else "flat")
