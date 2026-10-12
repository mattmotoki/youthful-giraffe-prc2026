"""exp-051, Sol round 1 answers B4 and B5: COMMON.md's four populations and the OOF receipts.  ZERO FIT.

For every key vector of iterations 1 and 2 this prices the served prediction pbase + d on each
fold in COMMON.md's four populations (line 98): all rows, target at or below 3600 s, target
above 3600 s, and the no-NM rows.  The no-NM mask is COMMON.md's definition, no NM join: every
`_flt` column of the raw month null, built from the raw month's departures in the panel's row
order.

For every registered step it also splits G (s2 per fold row, the pricing of price.py and
price2.py) into the rows at or below 3600 s, the rows above, and the no-NM rows, and gives each
registered ratio (I1's T, I2's rho and T2, the sham's rho^S) on the rows at or below 3600 s
beside all rows.  Those restricted ratios are points, record only: they gate nothing.

The vectors are the ones price.py and price2.py priced, rebuilt by their own functions.  Every
all-row fold G must equal its analysis JSON's to 1e-9 before anything is written.

The OOF receipts (COMMON.md:38): oof/i1-<fold>.parquet and oof/i2-<fold>.parquet, with
MVT_ID_mvt, fold, y_true, y_pred (the tested arm at its top level) and one y_pred_<vector>
column per key vector.

    .venv/bin/python experiments/exp-051-transferadv/code/populations.py
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import common as C
from common import A, BA, NO
from prc import data as D
import bagarm as BAG                                                  # noqa: E402
import price as PR
import price2 as P2

OUT = C.ANALYSIS / "populations.json"
OOF = C.EXP / "oof"
BOARD_FOLDS = ("july_loo", "january")
CUT = 3600.0
POPS = ("all", "le3600", "gt3600", "no_nm")


def oof_paths() -> list:
    return [OOF / f"{it}-{f}.parquet" for it in ("i1", "i2") for f in A.FOLDS]


def no_nm_mask(ids: np.ndarray, f: str) -> tuple[np.ndarray, dict]:
    month = A.FOLD_MONTH[f]
    flt = [c for c in pq.read_schema(D.month_file(month)).names if c.endswith("_flt")]
    raw = D.load_month(month, ["MVT_ID_mvt", "PHASE_mvt"] + flt)
    dep = raw[raw.PHASE_mvt == "DEP"]
    direct = pd.Series(dep[flt].isna().all(axis=1).to_numpy(),
                       index=dep.MVT_ID_mvt.to_numpy("float64"))
    if not direct.index.is_unique:
        raise SystemExit(f"{f}: departure ids are not unique")
    if not np.isin(ids, direct.index.to_numpy()).all():
        raise SystemExit(f"{f}: a panel row is not a departure of {month}")
    return direct.reindex(ids).to_numpy(bool), {"month": month, "departures_in_month": int(len(dep)),
                                                  "flt_columns": len(flt)}


def main() -> None:
    ps = argparse.ArgumentParser()
    ps.add_argument("--overwrite", action="store_true")
    a = ps.parse_args()
    t0 = time.time()
    NO.refuse_existing([OUT] + oof_paths(), overwrite=a.overwrite)
    pan_ship, _, _, _ = C.build_panels()
    b1 = json.loads((C.ANALYSIS / "boardlike.json").read_text())
    b2 = json.loads((C.ANALYSIS / "carrier.json").read_text())

    # ---- the key vectors, rebuilt by the pricing scripts' own functions
    hv = {f: C.half_of_rows(pan_ship, f) for f in A.FOLDS}
    V = {}                                                     # V[name] = {fold: d}
    for L in range(4):
        V[f"P{L}"] = C.load_vec(C.PLAIN[L])
        V[f"BL{L}"] = {f: PR.stitched(f, L, hv[f], {}) for f in A.FOLDS}
    for kind in ("P", "C"):
        for L in range(4):
            V[f"{kind}3bag_L{L}"] = BAG.build_bag({s: P2.arm_vec(kind, s, L) for s in P2.K3},
                                                 pan_ship, P2.K3)[0]
    for L in range(4):
        V[f"S1_L{L}"] = P2.arm_vec("S", 1, L)
        if not all(np.array_equal(V[f"S1_L{L}"][f], V[f"P{L}"][f]) for f in ("january", "december")):
            raise SystemExit(f"S1 L{L}: the sham arm differs from member 1 off july")
        for f in ("january", "december"):
            if not np.array_equal(V[f"C3bag_L{L}"][f], V[f"P3bag_L{L}"][f]):
                raise SystemExit(f"L{L} {f}: the carrier bag differs from the plain bag off july")
    F = P2.F
    V["JuneP4"] = {**V["P3"], F: P2.fit_vec("June-P4-s1")}
    V["JuneBL4"] = {**V["P3"], F: np.where(hv[F] == 1, P2.fit_vec("June-BL4-A"), P2.fit_vec("June-BL4-B"))}

    # ---- populations, per fold, in the panel's row order
    rec = {"what": __doc__.strip().splitlines()[0], "cut_s": CUT, "folds": {}, "vectors": {},
           "steps": {}, "restricted_ratios_record_only": {}}
    y, pb, M, r0 = {}, {}, {}, {}
    for f in A.FOLDS:
        p = pan_ship[f]
        ids = p.MVT_ID_mvt.to_numpy("float64")
        y[f], pb[f] = p.y_true.to_numpy("float64"), p.pbase.to_numpy("float64")
        r0[f] = y[f] - pb[f]
        nonm, info = no_nm_mask(ids, f)
        M[f] = {"all": np.ones(len(ids), bool), "le3600": y[f] <= CUT, "gt3600": y[f] > CUT,
                "no_nm": nonm}
        rec["folds"][f] = {**info, "panel_rows": int(len(ids)),
                           "rows": {k: int(m.sum()) for k, m in M[f].items()},
                           "no_nm_rows_above_3600": int((nonm & (y[f] > CUT)).sum()),
                           "board_weight": BA.SHARES.get(f, 0.0)}

    def rmse(name: str, f: str) -> dict:
        e = r0[f] - V[name][f]
        return {k: float(np.sqrt(np.mean(e[m] ** 2))) for k, m in M[f].items()}

    rec["vectors"]["base_d0"] = {f: {k: float(np.sqrt(np.mean(r0[f][m] ** 2))) for k, m in M[f].items()}
                                 for f in A.FOLDS}
    for name in V:
        rec["vectors"][name] = {f: rmse(name, f) for f in A.FOLDS}

    # ---- each registered step split by population; the all-row G is checked against the JSON
    def split(new: str, old: str, f: str) -> dict:
        g = (r0[f] - V[old][f]) ** 2 - (r0[f] - V[new][f]) ** 2
        n = len(g)
        return {k: float(g[m].sum() / n) for k, m in M[f].items()}

    def mse_pt(name: str, f: str, m: np.ndarray | None = None) -> float:
        # the pricing scripts' own arithmetic: day sums by bincount, then their total
        e = (r0[f] - V[name][f]) ** 2
        if m is not None:
            e = e[m]
        cd = pd.factorize(pan_ship[f].day.to_numpy()[m if m is not None else slice(None)])[0]
        return float(np.bincount(cd, weights=e).sum() / len(e))

    worst = 0.0
    for L in (1, 2, 3):
        s = C.STEP[L]
        for arm, new, old, js, key in (
                ("I1_BL", f"BL{L}", f"BL{L-1}", b1["steps"][s]["by_fold_s2_per_row"], "BL"),
                ("I1_P", f"P{L}", f"P{L-1}", b1["steps"][s]["by_fold_s2_per_row"], "P"),
                ("I2_C3", f"C3bag_L{L}", f"C3bag_L{L-1}", None, "C3"),
                ("I2_P3", f"P3bag_L{L}", f"P3bag_L{L-1}", None, "P3"),
                ("I2_S1", f"S1_L{L}", f"S1_L{L-1}", None, "S")):
            ent = {}
            for f in A.FOLDS:
                ent[f] = split(new, old, f)
                g_pt = mse_pt(old, f) - mse_pt(new, f)
                if js is not None:
                    ref = js[f][key]
                elif f == F and key in ("C3", "P3"):
                    ref = b2["steps"][s]["july_s2_per_row"][key]
                elif f == F:
                    ref = b2["steps"][s]["sham_K1"]["july_s2"]["S"]
                elif key == "S":
                    ref = b1["steps"][s]["by_fold_s2_per_row"][f]["P"]
                else:
                    ref = b2["steps"][s][f"{f}_s2_per_row_both_arms"]
                worst = max(worst, abs(g_pt - ref), abs(ent[f]["all"] - ref))
                if abs(g_pt - ref) > 1e-9 or abs(ent[f]["all"] - ref) > 1e-6:
                    raise SystemExit(f"{arm} {s} {f}: G {g_pt!r} / {ent[f]['all']!r} is not the JSON's {ref!r}")
            rec["steps"].setdefault(s, {})[arm] = ent

        # the registered ratios on all rows and on the rows at or below 3600 s (points)
        def board_gain(new: str, old: str, pop: str) -> float:
            def b(name):
                return np.sqrt(sum(BA.SHARES[f] * mse_pt(name, f, M[f][pop] if pop != "all" else None)
                                   for f in BOARD_FOLDS))
            return float(b(old) - b(new))

        rr = {}
        for pop in ("all", "le3600"):
            T = board_gain(f"BL{L}", f"BL{L-1}", pop) / board_gain(f"P{L}", f"P{L-1}", pop)
            T2 = (board_gain(f"C3bag_L{L}", f"C3bag_L{L-1}", pop)
                  / board_gain(f"P3bag_L{L}", f"P3bag_L{L-1}", pop))
            gC = split(f"C3bag_L{L}", f"C3bag_L{L-1}", F)[pop]
            gP = split(f"P3bag_L{L}", f"P3bag_L{L-1}", F)[pop]
            gS = split(f"S1_L{L}", f"S1_L{L-1}", F)[pop]
            gP1 = split(f"P{L}", f"P{L-1}", F)[pop]
            rr[pop] = {"I1_T": T, "I2_rho": gC / gP, "I2_T2": T2, "I2_rho_S": gS / gP1}
        for k, ref in (("I1_T", b1["steps"][s]["T_board_like_over_plain"]), ("I2_rho", b2["steps"][s]["rho"]),
                       ("I2_T2", b2["steps"][s]["T2"]), ("I2_rho_S", b2["steps"][s]["sham_K1"]["rho_S"])):
            worst = max(worst, abs(rr["all"][k] - ref))
            if abs(rr["all"][k] - ref) > 1e-9:
                raise SystemExit(f"{s} {k}: all-row ratio {rr['all'][k]!r} is not the JSON's {ref!r}")
        rec["restricted_ratios_record_only"][s] = rr
    rec["max_abs_diff_against_json"] = worst

    # ---- the OOF receipts
    cols = {"i1": (["BL3"], ["P0", "P1", "P2", "P3", "BL0", "BL1", "BL2"]),
            "i2": (["C3bag_L3"], ["C3bag_L0", "C3bag_L1", "C3bag_L2", "P3bag_L0", "P3bag_L1", "P3bag_L2",
                                  "P3bag_L3", "S1_L0", "S1_L1", "S1_L2", "S1_L3", "JuneP4", "JuneBL4"])}
    rec["oof"] = {}
    OOF.mkdir(parents=True, exist_ok=True)
    for it, (top, rest) in cols.items():
        for f in A.FOLDS:
            df = pd.DataFrame({"MVT_ID_mvt": pan_ship[f].MVT_ID_mvt.to_numpy(), "fold": f,
                               "y_true": y[f], "y_pred": pb[f] + V[top[0]][f]})
            if it == "i1":
                df["half"] = hv[f].astype("int8")
            keep = rest if (it == "i1" or f == F) else [c for c in rest if c.startswith("P3bag")]
            for c in keep:
                df[f"y_pred_{c}"] = pb[f] + V[c][f]
            path = OOF / f"{it}-{f}.parquet"
            NO.refuse_existing([path], overwrite=a.overwrite)
            df.to_parquet(path, index=False, compression="zstd")
            rec["oof"][path.name] = {"rows": int(len(df)), "columns": list(df.columns),
                                     "y_pred_is": top[0], "sha256": C.sha256(path)}
    rec["seconds"] = round(time.time() - t0, 1)
    C.write_json(OUT, rec, overwrite=a.overwrite)

    for f in A.FOLDS:
        fr = rec["folds"][f]
        print(f"{f}: panel {fr['panel_rows']:,} of {fr['departures_in_month']:,} departures; rows "
              f"{fr['rows']}; no-NM above 3600 {fr['no_nm_rows_above_3600']}")
        for name in ["base_d0"] + list(V):
            v = rec["vectors"][name][f]
            print(f"   {name:10} " + "  ".join(f"{k} {v[k]:9.3f}" for k in POPS))
    for s, rr in rec["restricted_ratios_record_only"].items():
        print(s, {pop: {k: round(x, 3) for k, x in d.items()} for pop, d in rr.items()})
    print(f"max |diff| against the JSONs {worst:.2e}; written to {OUT.relative_to(C.ROOT)} "
          f"and {len(rec['oof'])} OOF files ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
