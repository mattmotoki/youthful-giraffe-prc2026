"""exp-055 iteration 2: the historical loss falsification.  ZERO FIT.  READS LABELS.

SPEC section 8 (rule, registration 1, commit 2509842) and section 10 (registration 2, commit
bfbf55e).  Iteration 1's vectors come through `originvec.load`, which re-verifies every ordered
hash.  Labels: the exp-050 panel's `y_true` and `clean`, checked against exp-051's OOF files.

1. Labels and masks (SPEC 10.1): y_true equals both exp-051 OOF files of the fold bit for bit;
   clean against prc.data.excluded_ids; the no-NM mask against Q15's no_nm_sha256.
2. exp-051's gains reproduced with its own arithmetic to 1e-9 s2/row (SPEC 10.2), before any
   statistic.
3. For each point (SPEC 10.3), with p0 = pbase + X_{L-1}, r = y - p0, n = fold rows, sums on H:
   A, B, m, g_hat = 2 m A - m^2 B, g_plain = 2 A - B, g_C, J = |g_plain - g_C| - |g_hat - g_C|,
   and the exact amplitude / alignment split.
4. Intervals on the iteration-1 draws (default_rng(5500 + L), SPEC 4.4 order), day weights on
   every fold row, matched seed multisets for the bag (SPEC 10.4); the rule of SPEC 10.5.
5. COMMON's populations for every predictor of the loss test (SPEC 10.6), checked against
   exp-051's populations.json; the OOF receipt of SPEC 10.7.

Every output is write-once and written only after every check has passed.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-055-originmove/code/lossfals.py
    (a trial run writes under --scratch DIR and no repo file)
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import common as C
import originvec as V
from prc import data as D                                             # noqa: E402

B = 2000
BLOCKS = (1, 2, 3)
GATE_BLOCK = 3
KS = (1, 2, 3)
MULTISETS = list(itertools.combinations_with_replacement(C.SEEDS, 3))
MS = ["".join(map(str, s)) for s in MULTISETS]
CUT = 3600.0
TOL_REPRO = 1e-9
TOL_ROW = 1e-6
TOL_SPLIT = 1e-9
STATS_H = ("ry", "yy", "rx", "xx", "xy")
STATS_OUT = ("ox", "oy")
SCREEN = C.ANALYSIS / "screen.json"
#: iteration 1's record as committed in c947912 (not in the census; pinned here)
SCREEN_SHA = "081c39ed4789c5fc4851fe61808bf1cc28b13d5eee1e1e974a768faa17721ae9"
E51A = C.E51 / "analysis"


def name(p: Path) -> str:
    try:
        return C.rel(p)
    except ValueError:
        return str(p)


def load_labels(f: str, ids: np.ndarray) -> dict:
    C.check(C.PANEL[f])
    p = pd.read_parquet(C.PANEL[f], columns=["MVT_ID_mvt", "y_true", "clean", "regime"])
    if not np.array_equal(p.MVT_ID_mvt.to_numpy("float64"), ids):
        raise SystemExit(f"{f}: the label read is not in the pinned row order")
    y = p.y_true.to_numpy("float64")
    if not np.isfinite(y).all():
        raise SystemExit(f"{f}: y_true is not finite on every row")
    for it in ("i1", "i2"):
        C.check(C.OOF51[(it, f)])
        o = pd.read_parquet(C.OOF51[(it, f)], columns=["MVT_ID_mvt", "y_true"])
        if not (np.array_equal(o.MVT_ID_mvt.to_numpy("float64"), ids) and np.array_equal(o.y_true.to_numpy("float64"), y)):
            raise SystemExit(f"{f}: y_true is not exp-051's {it} OOF y_true bit for bit")
    return {"y": y, "clean": p.clean.to_numpy(bool), "nmreg": p.regime.astype(str).to_numpy(),
            "mvt_raw": p.MVT_ID_mvt.to_numpy()}


def no_nm_mask(ids: np.ndarray, f: str) -> np.ndarray:
    """COMMON's no-NM rows (exp-053 `no_nm_mask`, reproduced)."""
    month = C.FOLD_MONTH[f]
    flt = [c for c in pq.read_schema(D.month_file(month)).names if c.endswith("_flt")]
    raw = D.load_month(month, ["MVT_ID_mvt", "PHASE_mvt"] + flt)
    dep = raw[raw.PHASE_mvt == "DEP"]
    direct = pd.Series(dep[flt].isna().all(axis=1).to_numpy(), index=dep.MVT_ID_mvt.to_numpy("float64"))
    if not direct.index.is_unique or not np.isin(ids, direct.index.to_numpy()).all():
        raise SystemExit(f"{f}: panel ids do not map one to one onto {month}'s departures")
    return direct.reindex(ids).to_numpy(bool)


def mse51(y, pbase, d, codes) -> float:
    """exp-051's arithmetic: squared errors summed by day with bincount, total over n."""
    e = (y - pbase - d) ** 2
    return float(np.bincount(codes, weights=e).sum() / len(e))


def levels_for(f: str, pan: dict, vec: dict) -> dict:
    """Level vectors (L0..L3) of every arm used by a point on fold f."""
    v = vec[f]
    lev = {nm: [v[f"{nm}_L{L}"] for L in range(4)] for nm in ("P1", "BL", "P3bag")}
    if f == "july_loo":
        for nm in ("P2", "P3", "C1", "C2", "C3", "S1", "C3bag"):
            lev[nm] = [v[f"{nm}_L{L}"] for L in range(4)]
        for arm in ("P", "C"):
            for sig, tag in zip(MULTISETS, MS):
                lev[f"{arm}bag_{tag}"] = [
                    C.bag({s: v[f"{arm}{s}_L{L}"] for s in C.SEEDS}, sig, pan["pbase"], pan["usable"], pan["guard"])
                    for L in range(4)]
        for L in range(4):
            if not (np.array_equal(lev["Pbag_123"][L], v[f"P3bag_L{L}"])
                    and np.array_equal(lev["Cbag_123"][L], v[f"C3bag_L{L}"])):
                raise SystemExit("the (1, 2, 3) multiset bag is not the pinned K = 3 bag")
    else:
        lev["P2"] = [v[f"P2_L{L}"] for L in range(4)]
        lev["P3"] = [v[f"P3_L{L}"] for L in range(4)]
    return lev


def points_for(f: str) -> dict:
    """point: (carrier X, plain Y, K, role)"""
    pts = {"stitched": ("BL", "P1", 1, "descriptive (K = 1 stitched half)")}
    if f == "july_loo":
        pts = {f"bag{t}": (f"Cbag_{t}", f"Pbag_{t}", 3, "primary (P1, P2)" if t == "123" else "seed-multiset draw")
               for t in MS}
        pts |= {f"seed{s}": (f"C{s}", f"P{s}", 1, "primary (P3)") for s in C.SEEDS}
        pts["sham"] = ("S1", "P1", 1, "descriptive (damaged-base sham)")
        pts["stitched"] = ("BL", "P1", 1, "descriptive (K = 1 stitched half)")
    return pts


def read_receipt(src) -> pd.DataFrame:
    return pd.read_csv(src, dtype={"day": str, "airport": str}, float_precision="round_trip")


def receipt(key: str, cells: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    text = cells.to_csv(index=False, float_format="%.17g")
    back = read_receipt(io.StringIO(text))
    num = [c for c in cells.columns if c not in ("day", "airport")]
    if not (back[["day", "airport"]].equals(cells[["day", "airport"]])
            and np.array_equal(back[num].to_numpy("float64"), cells[num].to_numpy("float64"))):
        raise SystemExit(f"{key}: the receipt text does not parse back bit for bit")
    return text, back


def day_matrix(cells: pd.DataFrame, days: np.ndarray) -> tuple[np.ndarray, dict]:
    num = [c for c in cells.columns if c not in ("day", "airport")]
    g = cells.groupby("day")[num].sum().reindex(days, fill_value=0)
    return g.to_numpy("float64"), {c: j for j, c in enumerate(num)}


def interval(x: np.ndarray) -> dict:
    x = np.asarray(x, "float64")
    ok = np.isfinite(x)
    drop = int((~ok).sum())
    lo, hi = np.percentile(x[ok], [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi), "sd": float(x[ok].std(ddof=1)), "draws": int(ok.sum()),
            "dropped": drop, "unreliable": bool(drop > 0.01 * len(x))}


def quantities(S, cols: dict, pt: str, s: str, pick=None) -> dict:
    """Every SPEC 10.3 quantity from (weighted) sums; S is 1-D (point) or B x cols (draws)."""
    def g(stat):
        if pick is None:
            return S[..., cols[f"{stat}_{pt}_{s}"]]
        j = np.array([cols[f"{stat}_bag{t}_{s}"] for t in MS])
        return S[np.arange(S.shape[0]), j[pick]]
    n = S[..., cols["n"]]
    ry, yy, rx, xx, xy = (g(k) for k in STATS_H)
    A, Bq = ry / n, yy / n
    m = np.sqrt(xx / yy)
    ghat = 2 * m * A - m ** 2 * Bq
    gplain = 2 * A - Bq
    gC = (2 * rx - xx) / n
    J = np.abs(gplain - gC) - np.abs(ghat - gC)
    cos = xy / np.sqrt(xx * yy)
    amp = (m - 1) * (2 * A - (m + 1) * Bq)
    align = 2 * (rx / n - m * A)
    align_dir = 2 * m * (cos - 1) * A
    align_orth = 2 * (rx - (xy / yy) * ry) / n
    return {"A": A, "B": Bq, "m": m, "g_hat": ghat, "g_plain": gplain, "g_C": gC, "J": J, "cos": cos,
            "err_hat": np.abs(ghat - gC), "err_plain": np.abs(gplain - gC),
            "amplitude": amp, "alignment": align, "alignment_direction": align_dir,
            "alignment_orthogonal": align_orth}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", help="trial run: write every output under this directory, no repo file")
    args = ap.parse_args()
    base = Path(args.scratch).resolve() if args.scratch else C.EXP
    OUT = base / "analysis/lossfals.json"
    RPATH = {f: base / f"analysis/receipts/i2-daysums-{f}.csv" for f in C.FOLDS}
    OOFP = base / "oof/i2-members-july_loo.parquet"
    C.NO.refuse_existing([OUT, OOFP, *RPATH.values()], overwrite=False)
    t0 = time.time()
    np.seterr(divide="ignore", invalid="ignore")
    pan, vec, pin = V.load()
    if C.sha_file(SCREEN) != SCREEN_SHA:
        raise SystemExit("analysis/screen.json is not iteration 1's committed record")
    scr = json.loads(SCREEN.read_text())
    C.check(C.E53 / "analysis/pin.json")
    qpin = json.loads((C.E53 / "analysis/pin.json").read_text())
    rec = {"what": __doc__.strip().splitlines()[0],
           "registration": "SPEC.md section 8 (commit 2509842) and section 10 (commit bfbf55e)",
           "B": B, "blocks": list(BLOCKS), "gate_block": GATE_BLOCK, "multisets": MS,
           "generator": "np.random.default_rng(5500 + L), SPEC 4.4 draw order (ranking starts drawn, unused)",
           "units": "s2 per fold row (n = every row of the fold)", "checks": {}}

    # ---- 1. labels and masks ---------------------------------------------------------------------
    lab, masks = {}, {}
    rec["labels"] = {}
    for f in C.FOLDS:
        ids = pan[f]["ids"]
        lab[f] = load_labels(f, ids)
        nonm = no_nm_mask(ids, f)
        if C.sha_arr(nonm) != qpin["folds"][f]["no_nm_sha256"]:
            raise SystemExit(f"{f}: the no-NM mask is not Q15's")
        exc = D.excluded_ids(C.FOLD_MONTH[f])
        clean_d = ~np.isin(ids, np.array(sorted(exc), dtype="float64")) if exc else np.ones(len(ids), bool)
        y = lab[f]["y"]
        masks[f] = {"all": np.ones(len(ids), bool), "le3600": y <= CUT, "gt3600": y > CUT, "no_nm": nonm,
                    "clean": lab[f]["clean"]}
        masks[f] |= {f"own_{g}": pan[f]["reg"] == g for g in C.REGIMES}
        masks[f] |= {f"nm_{r}": lab[f]["nmreg"] == r for r in sorted(set(lab[f]["nmreg"]))}
        rec["labels"][f] = {"rows": int(len(ids)), "y_true_equals_exp051_oof_i1_i2": True,
                            "no_nm_matches_q15": True,
                            "clean_rows": int(lab[f]["clean"].sum()),
                            "clean_disagreements_with_excluded_ids": int((clean_d != lab[f]["clean"]).sum()),
                            "population_rows": {k: int(m.sum()) for k, m in masks[f].items()}}

    # ---- 2. exp-051's gains, reproduced with its arithmetic --------------------------------------
    C.check(E51A / "carrier.json")
    C.check(E51A / "boardlike.json")
    car = json.loads((E51A / "carrier.json").read_text())["steps"]
    bdl = json.loads((E51A / "boardlike.json").read_text())["steps"]
    lev, codes = {}, {}
    for f in C.FOLDS:
        lev[f] = levels_for(f, pan[f], vec)
        codes[f] = pd.factorize(pan[f]["day"])[0]

    def g51(f, nm, k):
        y, pb = lab[f]["y"], pan[f]["pbase"]
        return mse51(y, pb, lev[f][nm][k - 1], codes[f]) - mse51(y, pb, lev[f][nm][k], codes[f])

    repro, worst = {}, 0.0
    for k in KS:
        s = C.STEP[k]
        want = {("july_loo", "C3bag"): car[s]["july_s2_per_row"]["C3"],
                ("july_loo", "P3bag"): car[s]["july_s2_per_row"]["P3"],
                ("january", "P3bag"): car[s]["january_s2_per_row_both_arms"],
                ("december", "P3bag"): car[s]["december_s2_per_row_both_arms"],
                ("july_loo", "S1"): car[s]["sham_K1"]["july_s2"]["S"],
                ("july_loo", "P1:sham"): car[s]["sham_K1"]["july_s2"]["P1"]}
        for m_ in C.SEEDS:
            want[("july_loo", f"C{m_}")] = car[s]["members_record_only"][str(m_)]["july_s2"]["C"]
            want[("july_loo", f"P{m_}")] = car[s]["members_record_only"][str(m_)]["july_s2"]["P"]
        for f in C.FOLDS:
            want[(f, "BL")] = bdl[s]["by_fold_s2_per_row"][f]["BL"]
            want[(f, "P1:boardlike")] = bdl[s]["by_fold_s2_per_row"][f]["P"]
        for (f, key), ref in want.items():
            got = g51(f, key.split(":")[0], k)
            d = abs(got - ref)
            worst = max(worst, d)
            repro[f"{s}:{f}:{key}"] = {"exp051": ref, "reproduced": got, "abs_diff": d}
            if d > TOL_REPRO:
                raise SystemExit(f"{s} {f} {key}: gain {got!r} is not exp-051's {ref!r} (|diff| {d:.3e})")
    rec["reproduction_exp051"] = {"tolerance_s2_per_row": TOL_REPRO, "max_abs_diff": worst, "gains": repro}
    print(f"exp-051 gains reproduced: {len(repro)} values, max |diff| {worst:.3e} s2/row", flush=True)

    # ---- 3. per-row sums by (day, airport) over every fold row ------------------------------------
    texts, cells, days, D_, ci = {}, {}, {}, {}, {}
    rowcheck = 0.0
    own = {}
    for f in C.FOLDS:
        H = pan[f]["H"]
        y, pb = lab[f]["y"], pan[f]["pbase"]
        key = pd.MultiIndex.from_arrays([pan[f]["day"], pan[f]["ap"]])
        cd, uq = pd.factorize(key, sort=True)
        G = len(uq)
        cols = {"n": np.bincount(cd, minlength=G).astype("int64"),
                "n_H": np.bincount(cd, weights=H.astype("float64"), minlength=G).astype("int64")}
        own[f] = {}
        for pt, (X, Y, K, role) in points_for(f).items():
            for k in KS:
                s = C.STEP[k]
                x0, x1 = lev[f][X][k - 1], lev[f][X][k]
                y0, y1 = lev[f][Y][k - 1], lev[f][Y][k]
                r = y - pb - x0
                dx, dy = x1 - x0, y1 - y0
                part = {"ry": r * dy, "yy": dy * dy, "rx": r * dx, "xx": dx * dx, "xy": dx * dy}
                for st, v in part.items():
                    cols[f"{st}_{pt}_{s}"] = np.bincount(cd, weights=np.where(H, v, 0.0), minlength=G)
                ox = np.where(~H, 2 * r * dx - dx * dx, 0.0)
                ry_ = y - pb - y0
                oy = np.where(~H, 2 * r * dy - dy * dy, 0.0)
                if pt.startswith("bag"):
                    if np.any(dx[~H] != 0) or np.any(dy[~H] != 0):
                        raise SystemExit(f"{f} {pt} {s}: a bag step moves a row outside H")
                else:
                    cols[f"ox_{pt}_{s}"] = np.bincount(cd, weights=ox, minlength=G)
                    cols[f"oy_{pt}_{s}"] = np.bincount(cd, weights=oy, minlength=G)
                # row route against exp-051's gains (carrier all-row; the plain arm at its own p0)
                n = len(y)
                gC_all = float((np.sum(np.where(H, 2 * r * dx - dx * dx, 0.0)) + np.sum(ox)) / n)
                gP_own = float(np.sum(2 * ry_ * dy - dy * dy) / n)
                if not pt.startswith("bag") or pt == "bag123":
                    ck = {"bag123": "C3bag", "sham": "S1", "stitched": "BL"}.get(pt, X)
                    pk = {"bag123": "P3bag", "sham": "P1:sham", "stitched": "P1:boardlike"}.get(pt, Y)
                    for gv, kk in ((gC_all, ck), (gP_own, pk)):
                        ref = repro[f"{s}:{f}:{kk}"]["exp051"]
                        rowcheck = max(rowcheck, abs(gv - ref))
                    own[f][f"{pt}:{s}"] = {"g_C_all_rows": gC_all, "g_P_own_p0_all_rows": gP_own,
                                           "rho_all_rows": gC_all / gP_own}
        df = pd.DataFrame({"day": [a for a, _ in uq], "airport": [b for _, b in uq], **cols})
        texts[f], cells[f] = receipt(f, df)
        days[f] = np.sort(np.unique(pan[f]["day"]))
        D_[f], ci[f] = day_matrix(cells[f], days[f])
        if int(D_[f][:, ci[f]["n"]].sum()) != len(pan[f]["ids"]):
            raise SystemExit(f"{f}: the receipt does not cover every fold row")
    rec["checks"]["row_route_vs_exp051_max_abs_diff"] = rowcheck
    if rowcheck > TOL_ROW:
        raise SystemExit(f"the row-level gains differ from exp-051's by {rowcheck:.3e} s2/row")

    # ---- 4. point quantities ------------------------------------------------------------------------
    rec["points"] = {}
    splitw, mchk = 0.0, 0.0
    Pt = {f: D_[f].sum(0) for f in C.FOLDS}
    scr_arm = {"bag123": "june_bag", "seed1": "june_seed1", "seed2": "june_seed2", "seed3": "june_seed3",
               "sham": "sham"}
    for f in C.FOLDS:
        rec["points"][f] = {}
        for pt, (X, Y, K, role) in points_for(f).items():
            if pt.startswith("bag") and pt != "bag123":
                continue
            ent = {"carrier": X, "plain": Y, "K": K, "role": role, "steps": {}}
            for k in KS:
                s = C.STEP[k]
                q = {a: float(b) for a, b in quantities(Pt[f], ci[f], pt, s).items()}
                # the frozen iteration-1 m
                arm = scr_arm.get(pt, f"stitched_{'july' if f == 'july_loo' else f}")
                m1 = scr["arms"][arm]["steps"][s]["m"]
                mchk = max(mchk, abs(q["m"] - m1))
                # exact identities
                splitw = max(splitw, abs((q["g_C"] - q["g_plain"]) - (q["amplitude"] + q["alignment"])),
                              abs(q["alignment"] - (q["alignment_direction"] + q["alignment_orthogonal"])),
                              abs(q["g_hat"] - q["g_plain"] - q["amplitude"]))
                n = Pt[f][ci[f]["n"]]
                if not pt.startswith("bag"):
                    q["outside_H_g_C"] = float(Pt[f][ci[f][f"ox_{pt}_{s}"]] / n)
                    q["outside_H_plain_step_gain"] = float(Pt[f][ci[f][f"oy_{pt}_{s}"]] / n)
                else:
                    q["outside_H_g_C"] = 0.0
                    q["outside_H_plain_step_gain"] = 0.0
                q["m_iteration1"] = m1
                q |= own[f][f"{pt}:{s}"]
                q["g_P_minus_g_plain"] = q["g_P_own_p0_all_rows"] - q["g_plain"]
                q["J_positive"] = bool(q["J"] > 0)
                q["intervals"] = {}
                ent["steps"][s] = q
            ent["mean_J"] = {"point": float(np.mean([ent["steps"][s]["J"] for s in ent["steps"]])), "intervals": {}}
            rec["points"][f][pt] = ent
    rec["checks"]["m_vs_iteration1_max_abs_diff"] = mchk
    rec["checks"]["split_identities_max_abs_diff_s2"] = splitw
    if mchk > 1e-12:
        raise SystemExit(f"m differs from iteration 1's by {mchk:.3e}")
    if splitw > TOL_SPLIT:
        raise SystemExit(f"a split identity fails by {splitw:.3e} s2/row")

    # ---- 5. intervals on the iteration-1 draws ----------------------------------------------------
    rec["draw_sha256"] = {}
    for L in BLOCKS:
        rng = np.random.default_rng(5500 + L)
        st = {}
        for key in ("july_loo", "2026-07", "january", "2026-01", "december"):
            n_days = len(days[key]) if key in days else 31
            st[key] = C.mbb_starts(rng, n_days, L, B)
        sig = rng.integers(1, 4, size=(B, 3))
        msi = np.array([MULTISETS.index(tuple(sorted(t))) for t in sig.tolist()])
        rec["draw_sha256"][f"L{L}"] = {k: C.sha_arr(v) for k, v in st.items()} | {"seeds": C.sha_arr(sig)}
        for f in C.FOLDS:
            S = C.mbb_weights(st[f], len(days[f]), L) @ D_[f]
            for pt, ent in rec["points"][f].items():
                modes = {"day": None}
                if pt == "bag123":
                    modes["day+seed"] = msi
                for mode, pick in modes.items():
                    Js = []
                    for k in KS:
                        s = C.STEP[k]
                        q = quantities(S, ci[f], pt, s, pick)
                        Js.append(q["J"])
                        ent["steps"][s]["intervals"].setdefault(f"L{L}", {})[mode] = {
                            k_: interval(q[k_]) for k_ in ("J", "g_C", "g_hat", "g_plain", "m", "A", "B",
                                                          "amplitude", "alignment")}
                    ent["mean_J"]["intervals"].setdefault(f"L{L}", {})[mode] = interval(np.mean(np.vstack(Js), 0))
    # the iteration-1 draws are the same draws
    for L in BLOCKS:
        if rec["draw_sha256"][f"L{L}"] != {k: v for k, v in scr["draw_sha256"][f"L{L}"].items() if k != "multiset_counts"}:
            raise SystemExit(f"L{L}: the regenerated draws are not iteration 1's")
    rec["checks"]["draws_equal_iteration1"] = True

    # ---- 6. the rule (SPEC 10.5) -------------------------------------------------------------------
    bag = rec["points"]["july_loo"]["bag123"]
    p1 = bag["mean_J"]["intervals"][f"L{GATE_BLOCK}"]["day+seed"]
    p2 = {s: bag["steps"][s]["J"] for s in bag["steps"]}
    p3 = {f"seed{s}": rec["points"]["july_loo"][f"seed{s}"]["mean_J"]["point"] for s in C.SEEDS}
    rule = {"P1": {"statistic": "mean J of the June bag over v20, v21, v22; 3-day moving blocks plus matched seeds",
                   "point": bag["mean_J"]["point"], "lo": p1["lo"], "hi": p1["hi"], "unreliable": p1["unreliable"],
                   "holds": bool(p1["lo"] > 0)},
            "P2": {"J_by_step": p2, "holds": bool(all(v > 0 for v in p2.values()))},
            "P3": {"mean_J_by_seed": p3, "positive": int(sum(v > 0 for v in p3.values())),
                   "holds": bool(sum(v > 0 for v in p3.values()) >= 2)}}
    rule["verdict"] = "PASS" if all(rule[c]["holds"] for c in ("P1", "P2", "P3")) else "FAIL"
    rule["failed_conditions"] = [c for c in ("P1", "P2", "P3") if not rule[c]["holds"]]
    rec["rule"] = rule

    # ---- 7. COMMON's populations -----------------------------------------------------------------
    C.check(E51A / "populations.json")
    popj = json.loads((E51A / "populations.json").read_text())["vectors"]
    rec["populations"] = {}
    pw = 0.0
    for f in C.FOLDS:
        y, pb = lab[f]["y"], pan[f]["pbase"]
        preds = ["P3bag", "P1", "P2", "P3", "BL"] + (["C3bag", "C1", "C2", "C3", "S1"] if f == "july_loo" else [])
        if f != "july_loo":
            for L in range(4):
                if not np.array_equal(vec[f][f"P3bag_L{L}"], lev[f]["P3bag"][L]):
                    raise SystemExit("the plain bag off July is not the pinned one")
        rec["populations"][f] = {}
        for nm in preds:
            for L in range(4):
                e = y - pb - lev[f][nm][L]
                rec["populations"][f][f"{nm}_L{L}"] = {
                    k: {"rows": int(m.sum()), "rmse_s": float(np.sqrt(np.mean(e[m] ** 2))) if m.any() else None}
                    for k, m in masks[f].items()}
        # exp-051's populations.json
        for L in range(4):
            pairs = {f"P{L}": f"P1_L{L}", f"BL{L}": f"BL_L{L}", f"P3bag_L{L}": f"P3bag_L{L}",
                     f"C3bag_L{L}": f"C3bag_L{L}" if f == "july_loo" else f"P3bag_L{L}",
                     f"S1_L{L}": f"S1_L{L}" if f == "july_loo" else f"P1_L{L}"}
            for k51, mine in pairs.items():
                for pop in ("all", "le3600", "gt3600", "no_nm"):
                    d = abs(popj[k51][f][pop] - rec["populations"][f][mine][pop]["rmse_s"])
                    pw = max(pw, d)
                    if d > 1e-9:
                        raise SystemExit(f"{f} {k51} {pop}: RMSE differs from exp-051's populations.json by {d:.3e}")
    rec["checks"]["populations_vs_exp051_max_abs_diff_s"] = pw

    # ---- 8. the OOF receipt (July member vectors) ------------------------------------------------
    f = "july_loo"
    oof = pd.DataFrame({"MVT_ID_mvt": lab[f]["mvt_raw"], "fold": f, "y_true": lab[f]["y"],
                        "y_pred": pan[f]["pbase"] + lev[f]["C3bag"][3]})
    for nm in ("C1", "C2", "C3", "P2", "P3"):
        for L in range(4):
            oof[f"y_pred_{nm}_L{L}"] = pan[f]["pbase"] + lev[f][nm][L]
    buf = io.BytesIO()
    oof.to_parquet(buf, index=False)
    back = pd.read_parquet(io.BytesIO(buf.getvalue()))
    if not back.equals(oof):
        raise SystemExit("the OOF receipt does not read back")

    # ---- 9. write everything, then read the files back ------------------------------------------
    rec["receipts"] = {}
    for k, p in RPATH.items():
        C.NO.refuse_existing([p], overwrite=False)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(texts[k])
        rec["receipts"][name(p)] = C.sha_file(p)
        if not np.array_equal(day_matrix(read_receipt(p), days[k])[0], D_[k]):
            raise SystemExit(f"{name(p)}: the file on disk does not rebuild the day sums")
    C.NO.refuse_existing([OOFP], overwrite=False)
    OOFP.parent.mkdir(parents=True, exist_ok=True)
    OOFP.write_bytes(buf.getvalue())
    if not pd.read_parquet(OOFP).equals(oof):
        raise SystemExit("the OOF file on disk does not read back")
    rec["oof"] = {"path": name(OOFP), "sha256": C.sha_file(OOFP), "rows": int(len(oof)),
                  "columns": list(oof.columns),
                  "ordered_sha256": {c: C.sha_arr(pan[f]["ids"], oof[c].to_numpy("float64"))
                                     for c in oof.columns if c.startswith("y_pred")}}
    rec["seconds"] = round(time.time() - t0, 1)
    C.write_json(OUT, rec)

    # ---- log ---------------------------------------------------------------------------------------
    for f in C.FOLDS:
        for pt, ent in rec["points"][f].items():
            line = f"{f:9s} {pt:8s} K={ent['K']} "
            for s, q in ent["steps"].items():
                line += (f"| {s} m {q['m']:.3f} A {q['A']:+9.2f} B {q['B']:8.2f} g_C {q['g_C']:+9.2f} "
                         f"g_hat {q['g_hat']:+9.2f} g_plain {q['g_plain']:+9.2f} J {q['J']:+8.2f} cos {q['cos']:.3f} ")
            mj = ent["mean_J"]
            line += f"| mean J {mj['point']:+8.2f} day [{mj['intervals']['L3']['day']['lo']:+.2f}, {mj['intervals']['L3']['day']['hi']:+.2f}]"
            if "day+seed" in mj["intervals"]["L3"]:
                line += f" day+seed [{mj['intervals']['L3']['day+seed']['lo']:+.2f}, {mj['intervals']['L3']['day+seed']['hi']:+.2f}]"
            print(line, flush=True)
    print(f"P1 {rule['P1']['holds']} (mean J {rule['P1']['point']:+.3f}, [{rule['P1']['lo']:+.3f}, {rule['P1']['hi']:+.3f}]); "
          f"P2 {rule['P2']['holds']} ({', '.join(f'{k} {v:+.3f}' for k, v in p2.items())}); "
          f"P3 {rule['P3']['holds']} ({', '.join(f'{k} {v:+.3f}' for k, v in p3.items())})", flush=True)
    print(f"iteration 2 {rule['verdict']}; written {name(OUT)} ({rec['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
