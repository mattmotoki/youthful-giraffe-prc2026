"""exp-046-surfacefit: the fold-side base, and the measurement contract.

**The base is v12 as of 2026-09-08 14:50** (desk, tail rule): v11 with only its K2 member
replaced by `exp-035-k2climb` arm 3b's climbed member, board 262.7418, served file
`submissions/candidates/exp-035-k2climb-v11member.parquet` (SHA-256 `48dd572d…`).  Its
fold-side form is `exp-035`'s §26a composition over vectors that lane already gated --
no member, no rotation and no amplitude is refitted here either:

    head   = oof/cand-k2c-mod-refit-bulk-{fold}.parquet   (gated on §26's rotation)
    masked = head at exp-036's eight kept airports, v7 at EDDF and EHAM
    p12    = masked + S5_AMP * (stack-5 - masked)

and the rebuild is gated on `exp-035` §26c's published fold-side gain over v11
(+0.3762 all-row, +0.4460 clean) through THIS file's own `price()`, so the base cannot
drift from the file the desk uploaded.  **EDDF and EHAM sit at v7 in both compositions,
so p12 == p11 there row for row** -- which is where this lane's observation reaches, so
a held row stays bit-identical to v12 and the arms' gains carry over unchanged.

`p11` and `r11` stay on the panel: `code/reproduce_pilot.py` reproduces the pilot's
held-out table, which was measured against v11 and stays measured against v11.

**v11 is the board head** (`docs/SUBMISSIONS.md` row 11): `v10 + 0.217*(stack-5 - v10)`,
uploaded 2026-09-08, board 263.0298.  Its fold-side form is that same arithmetic over
two gated vectors, so nothing in this experiment is refitted:

    p10 = exp-039 `v10.load()`      (exp-036 arm 2, gated on 284.1572/204.9218/218.1506
                                     and on the uploaded file's SHA-256)
    p11 = p10 + S5_AMP * (stack-5 - p10)

`S5_AMP` is 0.217, the scalar the served file carries, stated in exp-038 §2b BEFORE
that build and never revised on a board reading (score-feedback rule).

The contract, from `exp-038-packet/RESULTS.md` §1, unchanged:

    r = y - p11,  d = p_arm - p11,  A = mean(r*d),  B = mean(d*d),  gain = 2A - B

reported as **board seconds** at the ranking-month shares, on the all-row population
(primary) and the clean one, with december alone beside them, and with day-clustered
intervals.  A row the arm does not move contributes exactly zero.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/base.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "code"))
from imports import load as _load                            # noqa: E402

V = _load("exp-039-screens", "v10")
sys.path.insert(0, str(ROOT))
import prc.data as D                                         # noqa: E402

FOLDS = ["july_loo", "december", "january"]
SHARES = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621}
S5_AMP = 0.217                     # exp-038 §2c, the scalar v11 was served at
BOOT = 2000
SEED = 46
V11_BOARD_CLEAN = 256.91           # docs/SUBMISSIONS.md row 11, for the gate print
V12_FILE = "submissions/candidates/exp-035-k2climb-v11member.parquet"
V12_SHA = "48dd572d61ff2403f13614ce04bdc3a8fd0dbdb5175bef7eb143650ea520ec52"
V12_BOARD = 262.7418               # docs/SUBMISSIONS.md row 12 (BOARD READING 11)
#: exp-035 RESULTS §26c, the fold side of the file the desk uploaded.  The rebuilt
#: base is gated on these through `price()`; `GATE_S` is in board seconds.
V12_FOLD_GAIN_OVER_V11 = {"all_row": 0.3762, "clean": 0.4460}
GATE_S = 0.01
#: exp-035 `code/price_vs_v11.py`, imported rather than re-derived: `load_head` gates
#: the head on its published clean rotation and orders it against the panel.
P35 = _load("exp-035-k2climb", "price_vs_v11")


def board(rmse_by_fold: dict[str, float]) -> float:
    return float(np.sqrt(sum(SHARES[f] * rmse_by_fold[f] ** 2 for f in SHARES)))


def load() -> dict[str, pd.DataFrame]:
    """Per fold: the panel with `pbase`/`rbase` (v12), `p11`/`r11`, and the day key."""
    pan = V.load()
    ref, vec, _ = V.Bs.load()
    head = P35.load_head(ref, P35.CAND_RB_OOF, P35.CAND_RB_PUBLISHED,
                         "the climbed member (refit-bulk anchor), for v12")
    out = {}
    for f in FOLDS:
        d = pan[f].copy()
        s5 = vec["stack-5"][f]
        if len(s5) != len(d):
            raise SystemExit(f"{f}: stack-5 has {len(s5)} rows, the panel {len(d)}")
        d["p11"] = d["p10"].to_numpy("float64") + S5_AMP * (s5 - d["p10"].to_numpy("float64"))
        d["r11"] = d["y_true"].to_numpy("float64") - d["p11"].to_numpy("float64")
        # v12: the SAME composition with the climbed member in the head's place
        if not np.array_equal(ref[f]["MVT_ID_mvt"].to_numpy(),
                              d["MVT_ID_mvt"].to_numpy()):
            raise SystemExit(f"{f}: the head's row order is not the panel's")
        keep = np.isin(d["airport"].astype(str).to_numpy(), V.KEEP)
        masked = np.where(keep, head[f], vec["v7"][f])
        d["p12"] = masked + S5_AMP * (vec["stack-5"][f] - masked)
        d["r12"] = d["y_true"].to_numpy("float64") - d["p12"].to_numpy("float64")
        d["pbase"] = d["p12"].to_numpy("float64")
        d["rbase"] = d["r12"].to_numpy("float64")
        d["day"] = pd.to_datetime(d["t"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
        d["month"] = pd.to_datetime(d["t"], unit="s", utc=True).dt.strftime("%Y-%m")
        out[f] = d
    return out


def paired(r: np.ndarray, d: np.ndarray, days: np.ndarray,
           rng: np.random.Generator | None = None) -> dict:
    """`2A - B` per row with a day-clustered interval, and its board-second form.

    The gain is reported per row of the WHOLE fold, so that an arm which moves few
    rows is not flattered: `v` is zero on an unmoved row and enters the mean.
    """
    v = 2 * r * d - d * d
    rec = {"rows": int(len(r)), "rows_moved": int((d != 0).sum()),
           "A_mean_rd": float(np.mean(r * d)), "B_mean_dd": float(np.mean(d * d)),
           "mse_gain_s2": float(np.mean(v))}
    if rng is not None:
        codes, uniq = pd.factorize(days)
        k = len(uniq)
        n = np.bincount(codes, weights=v, minlength=k)
        c = np.bincount(codes, minlength=k)
        draws = rng.integers(0, k, size=(BOOT, k))
        b = n[draws].sum(1) / c[draws].sum(1)
        lo, hi = np.percentile(b, [2.5, 97.5])
        rec["ci95"] = [float(lo), float(hi)]
    return rec


def board_interval(pan: dict[str, pd.DataFrame], d_by_fold: dict[str, np.ndarray],
                   rng: np.random.Generator, clean: bool = False) -> dict:
    """A day-clustered interval on the BOARD gain, from recombined draws.

    `paired()` gives an interval per fold, which is not the endpoint the assignment
    gates on (Sol round 1, blocking 9).  The board is `sqrt(sum share * MSE)` over the
    two ranking-share folds, so a board interval has to resample the days of BOTH folds
    inside one draw and rebuild the endpoint from the resampled MSEs.  Folds are drawn
    independently, which is the same independence `paired()` assumes within a fold.
    """
    num0, num1, cnt = {}, {}, {}
    for f in SHARES:
        p = pan[f]
        m = p["clean"].to_numpy(bool) if clean else np.ones(len(p), bool)
        y = p["y_true"].to_numpy("float64")[m]
        pb = p["pbase"].to_numpy("float64")[m]
        dd = np.asarray(d_by_fold[f], dtype="float64")[m]
        codes, uniq = pd.factorize(p["day"].to_numpy()[m])
        k = len(uniq)
        s0 = np.bincount(codes, weights=(y - pb) ** 2, minlength=k)
        s1 = np.bincount(codes, weights=(y - pb - dd) ** 2, minlength=k)
        c = np.bincount(codes, minlength=k).astype("float64")
        dr = rng.integers(0, k, size=(BOOT, k))
        num0[f], num1[f], cnt[f] = s0[dr].sum(1), s1[dr].sum(1), c[dr].sum(1)
    mse0 = sum(SHARES[f] * num0[f] / cnt[f] for f in SHARES)
    mse1 = sum(SHARES[f] * num1[f] / cnt[f] for f in SHARES)
    g = np.sqrt(mse0) - np.sqrt(mse1)
    lo, hi = np.percentile(g, [2.5, 97.5])
    return {"gain_s_bootstrap_mean": float(g.mean()), "ci95": [float(lo), float(hi)],
            "draws": BOOT, "clean": bool(clean),
            "share_of_draws_positive": float((g > 0).mean())}


def price(pan: dict[str, pd.DataFrame], d_by_fold: dict[str, np.ndarray],
          rng: np.random.Generator) -> dict:
    """The full contract table for one arm: all-row, clean, december, and the board."""
    out = {"by_fold": {}}
    rm_v11, rm_arm, rm_v11_c, rm_arm_c = {}, {}, {}, {}
    for f in FOLDS:
        p = pan[f]
        r = p["rbase"].to_numpy("float64")
        dd = np.asarray(d_by_fold[f], dtype="float64")
        y, p11 = p["y_true"].to_numpy("float64"), p["pbase"].to_numpy("float64")
        cl = p["clean"].to_numpy(bool)
        days = p["day"].to_numpy()
        rec = {"all_row": paired(r, dd, days, rng),
               "clean": paired(r[cl], dd[cl], days[cl], rng)}
        rec["rmse_base"] = float(np.sqrt(np.mean((y - p11) ** 2)))
        rec["rmse_arm"] = float(np.sqrt(np.mean((y - p11 - dd) ** 2)))
        rec["rmse_base_clean"] = float(D.rmse_clean(p["y_true"], p11, p["MVT_ID_mvt"], f))
        rec["rmse_arm_clean"] = float(D.rmse_clean(p["y_true"], p11 + dd, p["MVT_ID_mvt"], f))
        out["by_fold"][f] = rec
        rm_v11[f], rm_arm[f] = rec["rmse_base"], rec["rmse_arm"]
        rm_v11_c[f], rm_arm_c[f] = rec["rmse_base_clean"], rec["rmse_arm_clean"]
    out["board_all_row"] = {"base": board(rm_v11), "arm": board(rm_arm),
                            "gain_s": board(rm_v11) - board(rm_arm)}
    out["board_clean"] = {"base": board(rm_v11_c), "arm": board(rm_arm_c),
                          "gain_s": board(rm_v11_c) - board(rm_arm_c)}
    out["december_rmse"] = {"base": rm_v11["december"], "arm": rm_arm["december"],
                            "gain_s": rm_v11["december"] - rm_arm["december"]}
    out["board_all_row"]["interval"] = board_interval(pan, d_by_fold, rng, clean=False)
    out["board_clean"]["interval"] = board_interval(pan, d_by_fold, rng, clean=True)
    out["every_fold_interval_clear"] = all(
        out["by_fold"][f]["all_row"].get("ci95", [-1, 1])[0] > 0 for f in FOLDS)
    # WHY the gate reads False, when it does.  A fold on which the arm moves no row at
    # all contributes exactly zero and its interval is the degenerate [0, 0]; that is not
    # the arm failing on that fold, it is the fold having no summaries under the resolved
    # rule (2025-01 carries `summaries_rule_sha256: null` in the harvest manifest, so the
    # matcher fails closed on it).  The gate itself is NOT relaxed by this: it stays the
    # `all(FOLDS)` above, and the second boolean is a diagnostic that says where the
    # failure lives.
    moving = [f for f in FOLDS if np.any(np.asarray(d_by_fold[f]) != 0)]
    out["folds_with_no_moved_row"] = [f for f in FOLDS if f not in moving]
    out["every_moving_fold_interval_clear"] = bool(moving) and all(
        out["by_fold"][f]["all_row"].get("ci95", [-1, 1])[0] > 0 for f in moving)
    # the desk's 2026-09-08 17:15 instruction: no board number of this experiment may be
    # quoted without the basis it was computed on, while a fold is unpriced
    empty = out["folds_with_no_moved_row"]
    out["board_basis"] = (
        "the whole board: every fold priced" if not empty
        else "july half only, january fold unpriced" if empty == ["january"]
        else "partial: no row moved on " + ", ".join(empty))
    # the assignment's gate, as one boolean: an all-row paired gain over v11 with every
    # fold interval clear, AND the board endpoint itself clear of zero
    out["gate_all_row_gain_and_every_interval_clear"] = bool(
        out["board_all_row"]["gain_s"] > 0
        and out["every_fold_interval_clear"]
        and out["board_all_row"]["interval"]["ci95"][0] > 0)
    return out


def gate_v12(pan: dict[str, pd.DataFrame], rng: np.random.Generator) -> dict:
    """v12 rebuilt here must reproduce exp-035 §26c's published fold-side gain over v11.

    `d = p12 - p11` priced through THIS file's own `price()` on a v11 base, so the gate
    exercises the same algebra the arms are priced with and cannot pass by describing
    itself.  A miss is fatal: the base is the one thing in this experiment that is not
    allowed to drift.
    """
    v11 = {f: pan[f].assign(pbase=pan[f].p11, rbase=pan[f].r11) for f in FOLDS}
    d = {f: (pan[f].p12 - pan[f].p11).to_numpy("float64") for f in FOLDS}
    got = price(v11, d, rng)
    rec = {"all_row_gain_s": got["board_all_row"]["gain_s"],
           "clean_gain_s": got["board_clean"]["gain_s"],
           "published": V12_FOLD_GAIN_OVER_V11, "tolerance_s": GATE_S,
           "rows_v12_equals_v11": {f: int((d[f] == 0).sum()) for f in FOLDS},
           "eddf_eham_identical": {
               f: bool(np.all(d[f][np.isin(pan[f].airport.astype(str).to_numpy(),
                                           list(V.HOLD))] == 0)) for f in FOLDS}}
    rec["all_row_off_s"] = rec["all_row_gain_s"] - V12_FOLD_GAIN_OVER_V11["all_row"]
    rec["clean_off_s"] = rec["clean_gain_s"] - V12_FOLD_GAIN_OVER_V11["clean"]
    rec["passes"] = bool(abs(rec["all_row_off_s"]) <= GATE_S
                         and abs(rec["clean_off_s"]) <= GATE_S
                         and all(rec["eddf_eham_identical"].values()))
    if not rec["passes"]:
        raise SystemExit(f"the rebuilt v12 does not reproduce exp-035 §26c: {rec}")
    return rec


if __name__ == "__main__":
    pan = load()
    rep = {"S5_AMP": S5_AMP, "folds": {}}
    rep["v12_gate"] = gate_v12(pan, np.random.default_rng(SEED))
    for f in FOLDS:
        p = pan[f]
        rep["folds"][f] = {
            "rows": int(len(p)),
            "rmse_v10_all": float(np.sqrt(np.mean(p.r ** 2))),
            "rmse_v11_all": float(np.sqrt(np.mean(p.r11 ** 2))),
            "rmse_v12_all": float(np.sqrt(np.mean(p.r12 ** 2))),
            "rmse_v12_clean": float(D.rmse_clean(p.y_true, p.p12, p.MVT_ID_mvt, f)),
            "rmse_v10_clean": float(D.rmse_clean(p.y_true, p.p10, p.MVT_ID_mvt, f)),
            "rmse_v11_clean": float(D.rmse_clean(p.y_true, p.p11, p.MVT_ID_mvt, f)),
            "noac_rows": int(p.noac.sum()),
            "days": int(p.day.nunique()),
        }
    rep["board_v10_clean"] = board({f: rep["folds"][f]["rmse_v10_clean"] for f in SHARES})
    rep["board_v11_clean"] = board({f: rep["folds"][f]["rmse_v11_clean"] for f in SHARES})
    rep["v11_gain_over_v10_clean_s"] = rep["board_v10_clean"] - rep["board_v11_clean"]
    rep["board_v12_clean"] = board({f: rep["folds"][f]["rmse_v12_clean"] for f in SHARES})
    rep["v12_gain_over_v11_clean_s"] = rep["board_v11_clean"] - rep["board_v12_clean"]
    rep["published_v11_clean_board"] = V11_BOARD_CLEAN
    rep["published_v12_board"] = V12_BOARD
    rep["base_served_file"] = {"file": V12_FILE, "sha256": V12_SHA}
    (EXP / "analysis/base.json").write_text(json.dumps(rep, indent=2, sort_keys=True))
    print(json.dumps(rep, indent=2, sort_keys=True))
