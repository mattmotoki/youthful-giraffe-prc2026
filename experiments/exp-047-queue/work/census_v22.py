"""exp-047 §7bz: the §7bd residual census RE-READ on the CURRENT board bytes (v22).

recorded rule, the desk 2026-09-10: the extension lever's next month is priced under the 0.1 s
upload floor, so the lane's next kind of work is a lever that is NOT on the k-curve.
This re-reads the census that named the candidates -- but on the bytes the board holds
today, not on v12 as served, because ten uploads have moved one half of the file and
none of them has moved the other.

WHAT THE CURRENT BYTES ARE, on the fold side:

  complement (no clock OR noac, 148,585 ranking rows)  ->  p12 plus THIS LANE's own
      arm-N update on three cells (v14 and v17); no exp-046 upload since v17 has
      moved one complement row -- measured below, v18..v22 move 0 of 148,585
  with-clock class (196,256 ranking rows)              ->  p12 + d-seasonal-bag10

`d-seasonal-bag10-255-free.npz` is the vector v22's BYTES ship; it is read-only from
exp-046 and reproduces surfacefit's own per-fold rmse_new, so the row order is proved
and not trusted.  It is NOT applied to a complement row here even where the vector is
non-zero there: on the ranking side exp-046's class is exactly `obs_has_clock & ~noac`
(checked, 0 disagreements in 344,841), but on the FOLD side its vectors touch a few
hundred rows this lane's block calls no-clock, and the board follows the ranking class.
That disagreement is reported, not smoothed.

The arm-N update has no saved fold-side vector; `analysis/serve-lit.json` carries its
before/after per fold instead, and its size is reported beside every cell so the
census can be read as current to within it.  The three cells it SERVED are flagged, so
a spent cell is never offered as an unpriced lever.

No fits.  No candidate.  No serve.  An estimator (`base.shared`, exp-041's
self-excluded cross-product) is used for the reachable share exactly as in §7be.
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
sys.path.insert(0, str(EXP / "code"))
sys.path.insert(0, str(ROOT))
import base as BS          # noqa: E402
import prc.data as D       # noqa: E402

BLOCK = ROOT / "artifacts/exp-047/block"
SF = ROOT / "experiments/exp-046-surfacefit"
D_HEAD = SF / "work/d-bundle-ungated1299-guard995.npz"     # v18's bytes
D_NOW = SF / "work/d-seasonal-bag10-255-free.npz"          # v22's bytes
SERVE = SF / "analysis/serve-v25.json"
CLASSV = ROOT / "artifacts/exp-046/class"
UPLOADS = ROOT / "submissions/uploaded"
TAIL = 3600.0
# the three no-clock cells this lane has already SERVED (v14 EDDF/reached +
# LTFM/unreached, v17 adding LEMD/reached under the lit rule) -- analysis/serve-lit.json
# The reach grain of the two serve records is code/arms.py::reached -- `No_n_n_cell > 0`,
# "the network heard at least one other aircraft in the row's cell" -- and NOT the block's
# `reach_class`.  A first cut used `reach_class` and its served/never-served split was
# wrong; the serve records' own ranking-row counts did not reconcile with it.
REACH_COL = "No_n_n_cell"
SERVED_CELLS = {("EDDF", "reached"), ("LEMD", "reached"), ("LTFM", "unreached")}
LIT = None
TWELVE_H = 43200.0
BOARD_RMSE = 245.0207          # BOARD READING 21, the score these bytes scored
FLOOR_S = 0.1                  # recorded rule's upload floor
OUT = EXP / "analysis/census-v22.json"


def cell(r: np.ndarray, y: np.ndarray) -> dict:
    n = int(len(r))
    if not n:
        return {"rows": 0, "rms_s": None, "mean_signed_s": None, "sum_sq": 0.0}
    return {"rows": n, "rms_s": float(np.sqrt(np.mean(r ** 2))),
            "mean_signed_s": float(np.mean(r)), "sum_sq": float(np.sum(r ** 2))}


def board_seconds(share: float, frac: float) -> float:
    """Board seconds from removing `frac` of a cell holding `share` of the board's
    squared error: RMSE*(1 - sqrt(1 - share*frac)), the exact form, not the tangent."""
    return float(BOARD_RMSE * (1.0 - np.sqrt(max(0.0, 1.0 - share * frac))))


def frac_for_floor(share: float) -> float:
    """The fraction of a cell's own squared error that must go to clear the floor."""
    if share <= 0:
        return float("inf")
    need = 1.0 - (1.0 - FLOOR_S / BOARD_RMSE) ** 2
    return float(need / share)


def main() -> None:
    folds = BS.load(with_v12=True)
    zh, zn = np.load(D_HEAD), np.load(D_NOW)
    inc = json.loads(SERVE.read_text())["gate"]["increment_over_the_head"]

    pop, tot, checks = {}, {}, {"per_fold": {}}
    for f, d in folds.items():
        month = BS.MONTH_OF[f]
        blk2 = pd.read_parquet(BLOCK / f"{month}.parquet",
                               columns=["MVT_ID_mvt", "obs_has_clock", REACH_COL])
        blk = blk2
        mv = D.load_month(month, columns=["MVT_ID_mvt", "AIRCRAFT_TYPE_mvt", "PHASE_mvt"])
        mv = mv[mv.PHASE_mvt == "DEP"]
        noac_ids = set(mv.loc[mv["AIRCRAFT_TYPE_mvt"].isna(), "MVT_ID_mvt"].to_numpy())
        ids = d["MVT_ID_mvt"].to_numpy()
        clock = pd.Series(ids).map(
            blk.set_index("MVT_ID_mvt")["obs_has_clock"]).fillna(False).to_numpy(bool)
        noac = np.isin(ids, np.fromiter(noac_ids, dtype=ids.dtype, count=len(noac_ids)))
        comp = (~clock) | noac

        y = d["y_true"].to_numpy("float64")
        r12 = d["r12"].to_numpy("float64")
        dh, dn = zh[f].astype("float64"), zn[f].astype("float64")
        if not (len(dh) == len(dn) == len(r12)):
            raise SystemExit(f"{f}: vector lengths {len(dh)}/{len(dn)} against {len(r12)}")

        # the handshake, measured: how far the two lanes' fold-side classes disagree
        checks["per_fold"][f] = {
            "rows": int(len(r12)),
            "this_lanes_with_clock_rows": int((~comp).sum()),
            "v18_vector_nonzero_on_a_complement_row": int(np.count_nonzero(dh[comp])),
            "v22_vector_nonzero_on_a_complement_row": int(np.count_nonzero(dn[comp])),
            "those_rows_that_are_noac": int(np.count_nonzero(dn[comp & noac])),
            "rmse_head_reproduced": float(np.sqrt(np.mean((r12 - dh) ** 2))),
            "rmse_head_recorded": inc["by_fold"][f]["rmse_head"],
            "rmse_new_reproduced": float(np.sqrt(np.mean((r12 - dn) ** 2))),
            "rmse_new_recorded": inc["by_fold"][f]["rmse_new"]}
        c = checks["per_fold"][f]
        c["max_abs_rmse_disagreement_s"] = max(
            abs(c["rmse_head_reproduced"] - c["rmse_head_recorded"]),
            abs(c["rmse_new_reproduced"] - c["rmse_new_recorded"]))

        # the bytes the board holds now: the class carries v22, the complement is
        # where this lane left it -- no exp-046 upload since v17 has moved one
        r_now = np.where(comp, r12, r12 - dn)
        r_v18 = np.where(comp, r12, r12 - dh)
        tot[f] = {"fold_rows": int(len(r_now)),
                  "board_share": float(BS.SHARES.get(f, 0.0)),
                  "sum_sq_v12": float(np.sum(r12 ** 2)),
                  "sum_sq_v18": float(np.sum(r_v18 ** 2)),
                  "sum_sq_v22": float(np.sum(r_now ** 2)),
                  "rmse_v12": float(np.sqrt(np.mean(r12 ** 2))),
                  "rmse_v18": float(np.sqrt(np.mean(r_v18 ** 2))),
                  "rmse_v22": float(np.sqrt(np.mean(r_now ** 2)))}
        rc = pd.Series(ids).map(blk2.set_index("MVT_ID_mvt")[REACH_COL]).to_numpy("float64")
        reach = np.where(np.nan_to_num(rc, nan=0.0) > 0, "reached", "unreached")
        apt = d["apt"].to_numpy()
        served_N = np.zeros(len(ids), bool)
        for a, rc in SERVED_CELLS:
            served_N |= (apt == a) & (reach == rc)
        served_N &= comp
        pop[f] = {"month": month, "comp": comp, "clock": clock, "noac": noac,
                  "r": r_now, "r12": r12, "y": y, "reach": reach, "servedN": served_N,
                  "apt": apt, "bin": d["bin"].to_numpy(),
                  "day": d["day"].to_numpy(), "ids": ids}

    board_sq = sum(tot[f]["board_share"] * tot[f]["sum_sq_v22"] / tot[f]["fold_rows"]
                   for f in folds)
    board_sq_v12 = sum(tot[f]["board_share"] * tot[f]["sum_sq_v12"] / tot[f]["fold_rows"]
                       for f in folds)

    def bw(f: str, ss: float) -> float:
        return (tot[f]["board_share"] * ss / tot[f]["fold_rows"] / board_sq) if board_sq else 0.0

    def priced(f: str, ss: float) -> dict:
        s = bw(f, ss)
        return {"board_weighted_share": s,
                "board_seconds_if_all_of_it_went": board_seconds(s, 1.0),
                "share_of_its_own_squared_error_needed_for_%.2f_s" % FLOOR_S:
                    min(1.0, frac_for_floor(s)) if s > 0 else None,
                "the_floor_is_out_of_reach_on_this_cell": frac_for_floor(s) > 1.0}

    # band edges: the SAME construction as §7bd, on the pooled complement, tail first
    ally = np.concatenate([p["y"][p["comp"]] for p in pop.values()])
    body = ally[ally <= TAIL]
    edges = [float(x) for x in np.quantile(body, np.linspace(0, 1, 11))]
    edges[0], edges[-1] = float(body.min()), TAIL
    BANDS = ["d%d %.0f-%.0f s" % (i + 1, edges[i], edges[i + 1]) for i in range(10)] \
        + ["tail > %.0f s" % TAIL]

    def band_of(y: np.ndarray) -> np.ndarray:
        b = np.digitize(y, edges[1:-1], right=True)
        return np.where(y > TAIL, 10, b)

    out: dict = {
        "what": __doc__.strip().splitlines()[0],
        "written_before_any_option_was_chosen": True,
        "no_fit": "no arm is fitted, selected or served here; base.shared is an estimator",
        "price_base": {
            "complement": "p12, unchanged -- both served vectors are zero there",
            "with_clock": "p12 + d-seasonal-bag10-255-free.npz (v22's bytes)",
            "v18_for_contrast": "p12 + d-bundle-ungated1299-guard995.npz",
            "board_rmse_these_bytes_scored": BOARD_RMSE,
            "floor_s": FLOOR_S},
        "checks": checks,
        "the_lanes_own_served_update": json.loads(
            (EXP / "analysis/serve-lit.json").read_text())["currency"],
        "fold_totals": tot,
        "board_squared_error_v22": board_sq,
        "board_squared_error_v12_for_contrast": board_sq_v12,
        "the_two_halves": {}, "by_airport": {}, "by_band": {},
        "concentration": {}, "reachable": {}}

    for f, p in pop.items():
        rec = {}
        for name, m in (("complement", p["comp"]), ("with_clock", ~p["comp"])):
            c = cell(p["r"][m], p["y"][m])
            c["share_of_the_folds_rows"] = c["rows"] / tot[f]["fold_rows"]
            c["share_of_the_folds_squared_error"] = c["sum_sq"] / tot[f]["sum_sq_v22"]
            c.update(priced(f, c["sum_sq"]))
            c["sum_sq_at_v12"] = float(np.sum(p["r12"][m] ** 2))
            c["rms_at_v12_s"] = float(np.sqrt(np.mean(p["r12"][m] ** 2)))
            rec[name] = c
        for nm, mm in (("complement_already_served_by_arm_N", p["servedN"]),
                       ("complement_never_served", p["comp"] & ~p["servedN"])):
            c = cell(p["r"][mm], p["y"][mm])
            c.update(priced(f, c["sum_sq"]))
            rec[nm] = c
        rec["no_clock_only"] = cell(p["r"][(~p["clock"]) & ~p["noac"]],
                                    p["y"][(~p["clock"]) & ~p["noac"]])
        rec["no_clock_only"].update(priced(f, rec["no_clock_only"]["sum_sq"]))
        mno = p["noac"]
        rec["noac_any"] = cell(p["r"][mno], p["y"][mno])
        rec["noac_any"].update(priced(f, rec["noac_any"]["sum_sq"]))
        out["the_two_halves"][f] = rec

    for f, p in pop.items():
        for side, m0 in (("complement", p["comp"]), ("with_clock", ~p["comp"])):
            for a in sorted(set(p["apt"][m0])):
                s = m0 & (p["apt"] == a)
                c = cell(p["r"][s], p["y"][s])
                c.update(priced(f, c["sum_sq"]))
                out["by_airport"].setdefault(side, {}).setdefault(str(a), {})[f] = c
            b = band_of(p["y"])
            for i, name in enumerate(BANDS):
                s = m0 & (b == i)
                c = cell(p["r"][s], p["y"][s])
                c.update(priced(f, c["sum_sq"]))
                out["by_band"].setdefault(side, {}).setdefault(name, {})[f] = c

    for f, p in pop.items():
        m0 = p["comp"]
        for a in sorted(set(p["apt"][m0])):
            for rc in sorted(set(p["reach"][m0 & (p["apt"] == a)])):
                s_ = m0 & (p["apt"] == a) & (p["reach"] == rc)
                if not s_.any():
                    continue
                c = cell(p["r"][s_], p["y"][s_])
                c.update(priced(f, c["sum_sq"]))
                c["served_by_arm_N"] = (a, rc) in SERVED_CELLS
                c["body_only"] = cell(p["r"][s_ & (p["y"] <= TAIL)], p["y"][s_ & (p["y"] <= TAIL)])
                c["body_only"].update(priced(f, c["body_only"]["sum_sq"]))
                out.setdefault("complement_by_cell", {}).setdefault(
                    "%s/%s" % (a, rc), {})[f] = c

    # concentration, on the current bytes
    for f, p in pop.items():
        rec = {}
        order = np.argsort(-(p["r"] ** 2))
        for k in (1, 5, 20, 100):
            m = np.zeros(len(p["r"]), bool)
            m[order[:k]] = True
            c = cell(p["r"][m], p["y"][m])
            c["in_the_complement"] = int((m & p["comp"]).sum())
            c.update(priced(f, c["sum_sq"]))
            rec["worst_%d_rows" % k] = c
        m = p["y"] > TWELVE_H
        c = cell(p["r"][m], p["y"][m])
        c["in_the_complement"] = int((m & p["comp"]).sum())
        c.update(priced(f, c["sum_sq"]))
        rec["labels_over_12_hours"] = c
        out["concentration"][f] = rec

    # reachability on the shipped airport x 10-minute grid
    rng = np.random.default_rng(BS.SEED)
    for f, p in pop.items():
        code, ncode = BS.cellcode(p["apt"], p["bin"])
        r, day = p["r"], p["day"]
        named = {
            "complement, y <= 3600": p["comp"] & (p["y"] <= TAIL),
            "complement never served, y <= 3600": p["comp"] & ~p["servedN"] & (p["y"] <= TAIL),
            "complement, tail > 3600": p["comp"] & (p["y"] > TAIL),
            "complement at LFPG, y <= 3600": p["comp"] & (p["apt"] == "LFPG") & (p["y"] <= TAIL),
            "with-clock, y <= 3600": (~p["comp"]) & (p["y"] <= TAIL),
            "with-clock, tail > 3600": (~p["comp"]) & (p["y"] > TAIL)}
        rec = {}
        for name, keep in named.items():
            if keep.sum() < 50:
                continue
            sh = BS.shared(r, code, ncode, day, rng, keep=keep)
            ms = float(np.mean(r[keep] ** 2))
            ss = float(np.sum(r[keep] ** 2))
            sh["mean_sq_s2"] = ms
            sh["board_weighted_share"] = bw(f, ss)
            v = sh["value"]
            sh["reachable_share"] = (v / ms) if v is not None else None
            sh["reachable_share_ci"] = ([c / ms for c in sh["ci"]]
                                        if sh["ci"][0] is not None else [None, None])
            if v is not None:
                sh["board_seconds_at_the_point"] = board_seconds(sh["board_weighted_share"],
                                                                 max(0.0, sh["reachable_share"]))
                sh["board_seconds_at_the_ci_low"] = board_seconds(
                    sh["board_weighted_share"], max(0.0, sh["reachable_share_ci"][0]))
                sh["clears_the_floor_at_the_point"] = sh["board_seconds_at_the_point"] >= FLOOR_S
            rec[name] = sh
        out["reachable"][f] = rec

    # the ranking-side freeze: which upload last moved a complement row
    cls = pd.concat([pd.read_parquet(CLASSV / f"{m}-v25.parquet")
                     for m in ("2026-01", "2026-07")], ignore_index=True)
    rid = cls["MVT_ID_mvt"].to_numpy()
    rcomp = ~cls["occupied_by_exp_046"].to_numpy(bool)

    def upload(v: int) -> np.ndarray:
        s = pd.read_parquet(UPLOADS / f"youthful-giraffe_v{v}.parquet").set_index(
            "MVT_ID_mvt")["TAXITIME_SEC_mvt"]
        return pd.Series(rid).map(s).to_numpy("float64")

    steps, prev, pv = {}, None, None
    for v in (14, 15, 16, 17, 18, 19, 20, 21, 22):
        cur = upload(v)
        if prev is not None:
            steps["v%d_to_v%d" % (pv, v)] = {
                "complement_rows_that_moved": int((prev[rcomp] != cur[rcomp]).sum()),
                "class_rows_that_moved": int((prev[~rcomp] != cur[~rcomp]).sum())}
        prev, pv = cur, v
    out["the_ranking_side_freeze"] = {
        "complement_rows": int(rcomp.sum()), "class_rows": int((~rcomp).sum()),
        "class_definition_agrees_with_this_lanes":
            bool((cls["occupied_by_exp_046"].to_numpy(bool) ==
                  (cls["obs_has_clock"].to_numpy(bool) & ~cls["noac"].to_numpy(bool))).all()),
        "by_step": steps}

    OUT.write_text(json.dumps(out, indent=1, default=str))

    print("=== checks")
    for f, c in checks["per_fold"].items():
        print(" %-9s v18 vec on complement %d, v22 vec on complement %d, rmse agree to %.2e"
              % (f, c["v18_vector_nonzero_on_a_complement_row"],
                 c["v22_vector_nonzero_on_a_complement_row"],
                 c["max_abs_rmse_disagreement_s"]))
    print("\n=== the two halves on the CURRENT bytes")
    print(" %-9s %-11s %8s %8s %9s %9s %9s" % ("fold", "side", "rows", "rms", "of fold e2", "board", "s if all"))
    for f, rec in out["the_two_halves"].items():
        for side in ("complement", "with_clock"):
            c = rec[side]
            print(" %-9s %-11s %8d %8.1f %9.4f %9.4f %9.3f"
                  % (f, side, c["rows"], c["rms_s"], c["share_of_the_folds_squared_error"],
                     c["board_weighted_share"], c["board_seconds_if_all_of_it_went"]))
    print("\n=== reachable, priced")
    for f, rec in out["reachable"].items():
        for name, s in rec.items():
            if s.get("reachable_share") is None:
                continue
            print(" %-9s %-30s board %.4f reach %+.4f [%+.4f,%+.4f] -> %+.3f s (ci low %+.3f)"
                  % (f, name, s["board_weighted_share"], s["reachable_share"],
                     s["reachable_share_ci"][0], s["reachable_share_ci"][1],
                     s["board_seconds_at_the_point"], s["board_seconds_at_the_ci_low"]))
    print("\n=== the ranking-side freeze")
    for k, v in out["the_ranking_side_freeze"]["by_step"].items():
        print("  %-12s complement %6d   class %6d" % (k, v["complement_rows_that_moved"],
                                                      v["class_rows_that_moved"]))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
