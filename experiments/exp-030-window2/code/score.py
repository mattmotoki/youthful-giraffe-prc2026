"""One scoring function for every arm, against the v6 candidate.

`assignments/window2.md`: the comparison point for every arm is **the v6
candidate** — stack-5 with exp-028's 600 s window block inside its own stacker,
clean 257.6230, board 263.6922 (`docs/LEDGER.md` BOARD READING 6) — and not
stack-5.  So every arm is scored twice here: against stack-5, which is what
exp-028's tables quote and what makes the two experiments comparable, and against
v6, which is what the gate and the board prediction use.

The v6 and stack-5 fold vectors are read back from exp-028's committed rotation
(`oof/exp-028-stack5w-w600-{fold}.parquet`, columns `y_pred_depx` and
`y_pred_stack5`) and gated on exp-028's published numbers before any arm is
scored, exactly as `panel.load` gates the base.  Intervals are
`panel.boot_delta`, which is exp-026's `stack5.boot_delta` byte for byte, on the
clean rows and again with the `noac` rows dropped.

The board prediction is BOARD READING 6's rule: v6 measured 263.6922 with the 388
noac rows held at stack-5's own bytes, and every arm here holds the same 388 rows,
so the noac discount and the +4.28 anchor are common to both and cancel.  The
prediction is **263.6922 minus the arm's clean gain over v6**, quoted at face —
BOARD READING 6 measured this family transferring at 1.68x face, which is upside
this prediction does not claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
W28 = ROOT / "experiments/exp-028-window"
sys.path.insert(0, str(W28 / "code"))
import panel as P  # noqa: E402
import review1 as R1  # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

FOLDS = P.FOLDS
#: exp-028 RESULTS §4 and §8d, the two rows every arm is measured against.
PUBLISHED = {
    "stack-5": {"july_loo": 285.2029, "december": 205.6118, "january": 219.1303,
                "board": 258.0372},
    "v6": {"july_loo": 284.7219, "december": 205.6534, "january": 218.8166,
           "board": 257.6230},
}
V6_BOARD = 263.6922          # measured, BOARD READING 6
V6_OOF = W28 / "oof" / "exp-028-stack5w-w600-{fold}.parquet"
#: The **served** files the two fold-side vectors correspond to, pinned by
#: SHA-256.  A derived arm's record carries this contract and `serve.py` refuses
#: any `--from` / `--base` / `--regimes` that is not it (Sol round 1, blocker 1):
#: without it a command could attach an arm's score to different bytes.
CANDIDATES = {
    "v6": ("submissions/candidates/exp-028-window.parquet",
           "d55b1de9ce8fe7577c8c566c6efc57301d18484242aaa06ce808a111d38dabb0"),
    "stack-5": ("submissions/candidates/exp-026-stack-5.parquet",
                "94f18e47a73265a6db66811ededfc3c935fb2a37fac04bf0cd34d45af708a2a1"),
}
#: Every arm here serves the block on every non-`noac` regime; `noac` is score-feedback rule's.
REGIMES = ["nm_bulk", "nm_late", "nonm"]
#: exp-028 RESULTS §3, corrected under its own Sol round 1: the airport x 600 s
#: oracle on stack-5's residual.  `climb.log`'s gap-closed denominator.
ORACLE = 14.62
GATE = 1e-3
#: The gate `assignments/window2.md` sets for the shipping arm.
GAIN_MIN = 0.10
FOLD_MIN = -0.10


def reference() -> dict:
    """Per fold: the panel, stack-5's vector and v6's, gated on exp-028's numbers."""
    out = {}
    for f in FOLDS:
        o = pd.read_parquet(str(V6_OOF).format(fold=f))
        p = P.load_fold(f)[["MVT_ID_mvt", "airport", "regime", "nm_missing", "noac", "t"]]
        m = o.merge(p, on="MVT_ID_mvt", how="inner")
        if len(m) != len(o) or not np.array_equal(m["MVT_ID_mvt"].to_numpy(),
                                                  o["MVT_ID_mvt"].to_numpy()):
            raise SystemExit(f"{f}: the panel join lost or reordered rows")
        m = m.rename(columns={"y_pred_depx": "y_pred_v6"})
        out[f] = m
    for name, col in (("stack-5", "y_pred_stack5"), ("v6", "y_pred_v6")):
        got = {f: D.rmse_clean(out[f]["y_true"], out[f][col], out[f]["MVT_ID_mvt"], f)
               for f in FOLDS}
        off = {f: round(got[f] - PUBLISHED[name][f], 4) for f in FOLDS}
        if max(abs(v) for v in off.values()) > GATE:
            raise SystemExit(f"{name} does not reproduce exp-028's rotation: {off}")
        if abs(P.board(got) - PUBLISHED[name]["board"]) > GATE:
            raise SystemExit(f"{name} board {P.board(got):.4f} against "
                             f"{PUBLISHED[name]['board']:.4f}")
    return out


def score(name: str, preds: dict, ref: dict, note: str = "") -> dict:
    """The ledger row for one arm.  `preds[fold]` must be in the reference's row order."""
    rec = {"arm": name, "note": note, "clean": {}, "vs_stack5": {}, "vs_v6": {},
           "ci_v6": {}, "ci_v6_no_noac": {}, "slices": {}}
    per, base5, base6 = {}, {}, {}
    for f in FOLDS:
        d, p = ref[f], np.asarray(preds[f], "float64")
        if len(p) != len(d):
            raise SystemExit(f"{f}: {len(p)} predictions against {len(d)} rows")
        y = d["y_true"].to_numpy("float64")
        ids = d["MVT_ID_mvt"].to_numpy("float64")
        a5 = d["y_pred_stack5"].to_numpy("float64")
        a6 = d["y_pred_v6"].to_numpy("float64")
        per[f] = D.rmse_clean(y, p, ids, f)
        base5[f] = D.rmse_clean(y, a5, ids, f)
        base6[f] = D.rmse_clean(y, a6, ids, f)
        cl = P.clusters(d)
        ok = ~d["MVT_ID_mvt"].isin(D.excluded_ids(f)).to_numpy()
        nn = ok & (d["regime"].to_numpy() != "noac")
        rec["ci_v6"][f] = [round(x, 4) for x in
                           P.boot_delta(y[ok], p[ok], a6[ok], pd.factorize(cl[ok])[0])]
        rec["ci_v6_no_noac"][f] = [round(x, 4) for x in
                                   P.boot_delta(y[nn], p[nn], a6[nn], pd.factorize(cl[nn])[0])]
        # COMMON.md's four slices, on the SAME population as the point estimate:
        # clean.  exp-028's Sol round 1 (Standards 4, Spec 7) caught both halves of
        # this — an all-rows slice quoted beside a clean board, and a no-NM mask
        # built from `FLIGHT_ID_mvt` alone instead of COMMON.md's "every `_flt`
        # column null".  `review1.no_nm_mask` is exp-028's corrected mask; the loose
        # one is kept beside it so the difference stays visible.
        e = ((p - y) ** 2)[ok]
        yy = y[ok]
        strict = R1.no_nm_mask(f, ids)[ok]
        bulk = yy <= 3600.0
        rr = lambda m: round(float(np.sqrt(e[m].mean())), 4) if m.any() else float("nan")
        rec["slices"][f] = {
            "n": int(ok.sum()), "rmse": round(per[f], 4),
            "le3600": rr(bulk), "gt3600": rr(~bulk), "no_nm": rr(strict),
            "no_nm_flight_id_only": rr(d["nm_missing"].to_numpy().astype(bool)[ok]),
            "noac": rr((d["regime"].to_numpy() == "noac")[ok]),
            "no_noac": rr((d["regime"].to_numpy() != "noac")[ok]),
            "n_no_nm": int(strict.sum())}
        rec["clean"][f] = round(per[f], 4)
        rec["vs_stack5"][f] = round(base5[f] - per[f], 4)
        rec["vs_v6"][f] = round(base6[f] - per[f], 4)
    rec["clean_board"] = round(P.board(per), 4)
    rec["gain_board_vs_stack5"] = round(P.board(base5) - P.board(per), 4)
    rec["gain_board_vs_v6"] = round(P.board(base6) - P.board(per), 4)
    rec["predicted_board"] = round(V6_BOARD - rec["gain_board_vs_v6"], 4)
    rec["worst_fold_vs_v6"] = round(min(rec["vs_v6"].values()), 4)
    rec["gate_gain"] = bool(rec["gain_board_vs_v6"] >= GAIN_MIN)
    rec["gate_folds"] = bool(rec["worst_fold_vs_v6"] >= FOLD_MIN)
    rec["gate"] = bool(rec["gate_gain"] and rec["gate_folds"])
    rec["regimes"] = list(REGIMES)
    return rec


def show(rec: dict) -> None:
    print(f"\n=== {rec['arm']} ===  {rec['note']}")
    print("  clean      " + "  ".join(f"{f} {rec['clean'][f]:9.4f}" for f in FOLDS)
          + f"   board {rec['clean_board']:9.4f}")
    print("  vs stack-5 " + "  ".join(f"{f} {rec['vs_stack5'][f]:+9.4f}" for f in FOLDS)
          + f"   board {rec['gain_board_vs_stack5']:+9.4f}")
    print("  vs v6      " + "  ".join(f"{f} {rec['vs_v6'][f]:+9.4f}" for f in FOLDS)
          + f"   board {rec['gain_board_vs_v6']:+9.4f}")
    for f in FOLDS:
        d, lo, hi = rec["ci_v6"][f]
        dn, lon, hin = rec["ci_v6_no_noac"][f]
        print(f"    {f:9s} vs v6 {-d:+7.3f} [{-hi:+7.3f}, {-lo:+7.3f}]"
              f"   no-noac {-dn:+7.3f} [{-hin:+7.3f}, {-lon:+7.3f}]")
    print(f"  predicted board {rec['predicted_board']:.4f} "
          f"(v6 measured {V6_BOARD}, at face)")
    print(f"  gate: gain >= {GAIN_MIN} {rec['gate_gain']}, "
          f"no fold below {FOLD_MIN} {rec['gate_folds']}  ->  "
          f"{'PASS' if rec['gate'] else 'FAIL'}")


def save(rec: dict) -> Path:
    p = EXP / "analysis" / f"arm-{rec['arm']}.json"
    p.write_text(json.dumps(rec, indent=2))
    # COMMON.md's format: iteration, mechanism changed, july RMSE, december RMSE,
    # (january RMSE), fraction of the remaining gap closed.  The gap-closed
    # denominator is exp-028 §3's corrected airport x 600 s oracle on stack-5's
    # residual, 14.62 s -- the same denominator exp-028's own climb.log uses, so
    # the two experiments' lines are comparable.  (Sol round 1: the first version
    # of this line logged gains only and omitted the RMSEs and the fraction.)
    log = EXP / "climb.log"
    n = sum(1 for ln in log.read_text().splitlines()
            if ln and not ln.startswith("#")) + 1 if log.exists() else 1
    with open(log, "a") as fh:
        fh.write(f"{n:03d}  {pd.Timestamp.utcnow():%Y-%m-%dT%H:%M}  {rec['arm']:18s} "
                 + "  ".join(f"{f} {rec['clean'][f]:9.4f}" for f in FOLDS)
                 + f"  board {rec['clean_board']:9.4f}"
                 + "  vs v6 " + " ".join(f"{rec['vs_v6'][f]:+.3f}" for f in FOLDS)
                 + f" board {rec['gain_board_vs_v6']:+.3f}"
                 + f"  gap-closed {rec['gain_board_vs_v6'] / ORACLE:+.4f} of the "
                   f"{ORACLE} s window oracle"
                 + f"  predict {rec['predicted_board']:.2f}"
                 f"  gate {'PASS' if rec['gate'] else 'FAIL'}\n")
    return p
