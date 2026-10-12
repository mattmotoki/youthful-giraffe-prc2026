"""exp-055 iteration 1, step 2: the label-free screen.  ZERO FIT, LABEL-FREE.

SPEC sections 4-6 (registration commit 2509842; pins in `analysis/pin.json`, commit 5fe9de1).
Reads the vectors that `pin.py` pinned (through `vectors.load`, which re-verifies every ordered
hash), the ranking inputs and the uploaded bytes.  No `y_true`, no `clean`, no exp-051 loss
record.

1. Steps d_X = X_L - X_{L-1} (L = 1, 2, 3 <-> v20, v21, v22) for every arm of SPEC section 5,
   for both K = 3 bags on all ten seed multisets (bag rule, ascending order), and for the K = 10
   heads; the uploaded steps on ranking H.
2. Receipts: per (day, airport) of H, the row count and every sum of squares and cross product
   used here (`analysis/receipts/i1-daysums-*.csv`, floats written with 17 significant digits).
   Every interval and every point ratio is computed from the receipt text as parsed back (checked
   bit for bit against the in-memory sums); the files on disk are read once more at the end.
3. m, q, I = |1 - q| - |m - q|, the mean over the three steps, m - q and m / q, with the
   moving-block day bootstrap (block L = 1, 2, 3; B = 2,000; default_rng(5500 + L); the SPEC draw
   order) and, for the K = 3 bags, matched seed multisets.
4. The registered rule C1, C2, C3 (SPEC section 6) and the descriptive tables of sections 4.2,
   4.3 and 6.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-055-originmove/code/screen.py
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

import common as C
import originvec as V

B = 2000
BLOCKS = (1, 2, 3)
GATE_BLOCK = 3
KS = (1, 2, 3)
MULTISETS = list(itertools.combinations_with_replacement(C.SEEDS, 3))    # the 10 seed multisets
MS = ["".join(map(str, s)) for s in MULTISETS]
FULL = "123"
REL = 1e-9
#: SPEC section 5.  name: (fold, carrier X, plain Y, K, role)
ARMS = {
    "june_bag": ("july_loo", "Cbag", "Pbag", 3, "primary (gate C1, C2)"),
    "june_seed1": ("july_loo", "C1", "P1", 1, "gate C3"),
    "june_seed2": ("july_loo", "C2", "P2", 1, "gate C3"),
    "june_seed3": ("july_loo", "C3", "P3", 1, "gate C3"),
    "stitched_july": ("july_loo", "BL", "P1", 1, "descriptive (K = 1 held-month check)"),
    "stitched_january": ("january", "BL", "P1", 1, "descriptive (K = 1 held-month check)"),
    "stitched_december": ("december", "BL", "P1", 1, "descriptive (no ranking month, no target)"),
    "sham": ("july_loo", "S1", "P1", 1, "damaged-base negative control, never gates"),
}


def fold_steps(f: str, pan: dict, vec: dict) -> dict:
    v = vec[f]
    lev = {nm: [v[f"{nm}_L{L}"] for L in range(4)] for nm in ("T10", "P1", "BL", "P3bag")}
    if f == "july_loo":
        for nm in ("P2", "P3", "C1", "C2", "C3", "S1"):
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
    st = {nm: {k: x[k] - x[k - 1] for k in KS} for nm, x in lev.items()}
    for nm, dk in st.items():
        if nm.startswith(("T10", "P3bag", "Pbag", "Cbag")):
            for k in KS:
                if np.any(dk[k][~pan["H"]] != 0):
                    raise SystemExit(f"{f} {nm} {C.STEP[k]}: a bag or K = 10 step moves a row outside H")
    return st, lev


def pairs(f: str) -> list:
    pr = [("BL", "P1"), ("P3bag", "T10"), ("P1", "T10")]
    if f == "july_loo":
        pr += [("S1", "P1"), ("P2", "T10"), ("P3", "T10")]
        pr += [(f"C{s}", f"P{s}") for s in C.SEEDS] + [(f"Cbag_{t}", f"Pbag_{t}") for t in MS]
    return pr


def cells_fold(f: str, pan: dict, st: dict) -> pd.DataFrame:
    H = pan["H"]
    cols = {"day": pan["day"][H], "airport": pan["ap"][H], "n": np.ones(int(H.sum()), "int64")}
    for nm, dk in st.items():
        for k in KS:
            cols[f"ss_{nm}_{C.STEP[k]}"] = dk[k][H] ** 2
    for X, Y in pairs(f):
        for k in KS:
            cols[f"x_{X}__{Y}_{C.STEP[k]}"] = st[X][k][H] * st[Y][k][H]
    return pd.DataFrame(cols).groupby(["day", "airport"], sort=True).sum().reset_index()


def cells_rank(sel: np.ndarray, rk: pd.DataFrame, up: dict) -> pd.DataFrame:
    cols = {"day": rk.day.to_numpy()[sel], "airport": rk.airport.to_numpy()[sel],
            "n": np.ones(int(sel.sum()), "int64")}
    for k in KS:
        cols[f"ss_UP_{C.STEP[k]}"] = (up[C.LEVELS10[k]] - up[C.LEVELS10[k - 1]])[sel] ** 2
    return pd.DataFrame(cols).groupby(["day", "airport"], sort=True).sum().reset_index()


def read_receipt(src) -> pd.DataFrame:
    return pd.read_csv(src, dtype={"day": str, "airport": str}, float_precision="round_trip")


def receipt(key: str, cells: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """The receipt text and the table parsed back from it (bit for bit, or a hard stop)."""
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


def kish(x) -> float:
    return C.kish(np.asarray(x, "float64"))


def name(p: Path) -> str:
    try:
        return C.rel(p)
    except ValueError:
        return str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", help="trial run: write every output under this directory, no repo file")
    args = ap.parse_args()
    base = Path(args.scratch).resolve() if args.scratch else C.ANALYSIS
    OUT = base / "screen.json"
    RPATH = {f: base / f"receipts/i1-daysums-{f}.csv" for f in C.FOLDS}
    RPATH |= {m: base / f"receipts/i1-daysums-rank-{m}.csv" for m in C.RMONTHS}
    C.NO.refuse_existing([OUT, *RPATH.values()], overwrite=False)
    t0 = time.time()
    np.seterr(divide="ignore", invalid="ignore")
    pan, vec, pin = V.load()
    qpin = json.loads((C.E53 / "analysis/pin.json").read_text())       # hash keys only
    rec = {"what": __doc__.strip().splitlines()[0],
           "registration": "SPEC.md, commit 2509842; pins analysis/pin.json, commit 5fe9de1",
           "B": B, "blocks": list(BLOCKS), "gate_block": GATE_BLOCK,
           "generator": "np.random.default_rng(5500 + L)",
           "draw_order": ["july_loo fold starts", "2026-07 ranking starts", "january fold starts",
                          "2026-01 ranking starts", "december fold starts", "seed triples (B x 3)"],
           "multisets": MS, "percentiles": [2.5, 97.5], "checks": {}}

    # ---- steps and fold receipts ---------------------------------------------------------------
    steps, levels, cells, days, texts = {}, {}, {}, {}, {}
    for f in C.FOLDS:
        steps[f], levels[f] = fold_steps(f, pan[f], vec)
        days[f] = np.sort(np.unique(pan[f]["day"]))
        texts[f], cells[f] = receipt(f, cells_fold(f, pan[f], steps[f]))

    # ---- ranking side, verified against Q15's pin and pin.py's upload hashes ----------------------
    rk = C.load_ranking()
    rid = rk.MVT_ID_mvt.to_numpy("float64")
    rreg = rk.regime.to_numpy()
    if C.sha_arr(rid) != qpin["ranking"]["ids_sha256"] or \
            C.sha_arr(C.regime_index(rreg)) != qpin["ranking"]["regime_sha256"]:
        raise SystemExit("ranking ids or regime differ from Q15's pin")
    up = C.load_uploads(rid)
    for lv in C.LEVELS10:
        if C.sha_arr(rid, up[lv]) != pin["ordered_sha256"][f"upload_{lv}"]:
            raise SystemExit(f"upload {lv} differs from pin.py's ordered hash")
    rmo = rk.month.to_numpy()
    for m in C.RMONTHS:
        sel = (rmo == m) & (rreg == "H")
        days[m] = np.sort(np.unique(rk.day.to_numpy()[rmo == m]))
        texts[m], cells[m] = receipt(m, cells_rank(sel, rk, up))
    rec["days"] = {k: {"n": int(len(d)), "first": str(d[0]), "last": str(d[-1])} for k, d in days.items()}
    for k, d in days.items():
        if len(d) != 31:
            raise SystemExit(f"{k}: {len(d)} days, not 31")

    D, ci = {}, {}
    for k in days:
        D[k], ci[k] = day_matrix(cells[k], days[k])

    # the day sums against the vectors directly (summation order only)
    worst = 0.0
    for f in C.FOLDS:
        H = pan[f]["H"]
        tot = D[f].sum(0)
        if int(tot[ci[f]["n"]]) != C.H_ROWS[f]:
            raise SystemExit(f"{f}: receipt row count {tot[ci[f]['n']]} is not {C.H_ROWS[f]}")
        for nm, dk in steps[f].items():
            for k in KS:
                direct = float(np.sum(dk[k][H] ** 2))
                got = tot[ci[f][f"ss_{nm}_{C.STEP[k]}"]]
                r = abs(got - direct) / max(direct, 1e-300)
                worst = max(worst, r)
    rec["checks"]["receipt_sums_vs_direct_max_rel"] = worst
    if worst > 1e-12:
        raise SystemExit(f"receipt sums differ from the direct sums by {worst:.3e} relative")

    # ---- point quantities --------------------------------------------------------------------
    P = {k: D[k].sum(0) for k in D}

    def col(S, key, name):
        return S[..., ci[key][name]]

    def q_of(Sf, Sr, f, m, k):
        s = C.STEP[k]
        return np.sqrt((col(Sr, m, f"ss_UP_{s}") / col(Sr, m, "n")) / (col(Sf, f, f"ss_T10_{s}") / col(Sf, f, "n")))

    def m_of(Sf, f, X, Y, k):
        s = C.STEP[k]
        return np.sqrt(col(Sf, f, f"ss_{X}_{s}") / col(Sf, f, f"ss_{Y}_{s}"))

    def imp(m, q):
        return np.abs(1 - q) - np.abs(m - q)

    qpt = {}
    rec["q"] = {}
    for f, m in C.RANK_MONTH.items():
        key = f"{f}->{m}"
        rec["q"][key] = {}
        for k in KS:
            val = float(q_of(P[f], P[m], f, m, k))
            want = pin["q_reproduction"][f"{C.STEP[k]}:{key}"]["q"]
            if abs(val - want) > 1e-12:
                raise SystemExit(f"q {C.STEP[k]} {key}: {val!r} from the receipts, {want!r} in pin.json")
            qpt[(f, k)] = val
            s = C.STEP[k]
            rec["q"][key][s] = {
                "point": val, "pin_json": want, "abs_diff": abs(val - want),
                "rms_ranking_s": float(np.sqrt(col(P[m], m, f"ss_UP_{s}") / col(P[m], m, "n"))),
                "rms_fold_s": float(np.sqrt(col(P[f], f, f"ss_T10_{s}") / col(P[f], f, "n"))),
                "q_airport_mix_q15": pin["q_reproduction"][f"{s}:{key}"]["q_airport_mix"],
                "kish_days_rank_rows": kish(D[m][:, ci[m]["n"]]),
                "kish_days_rank_mass": kish(D[m][:, ci[m][f"ss_UP_{s}"]]),
                "kish_days_fold_rows": kish(D[f][:, ci[f]["n"]]),
                "kish_days_fold_mass_K10": kish(D[f][:, ci[f][f"ss_T10_{s}"]]),
                "intervals": {}}

    def arm_names(arm, tag=FULL):
        f, X, Y, K, _ = ARMS[arm]
        if K == 3:
            return f"{X}_{tag}", f"{Y}_{tag}"
        return X, Y

    rec["arms"] = {}
    for arm, (f, X0, Y0, K, role) in ARMS.items():
        X, Y = arm_names(arm)
        H = pan[f]["H"]
        reg = pan[f]["reg"]
        has_q = f in C.RANK_MONTH
        ar = {"fold": f, "carrier": X, "plain": Y, "K": K, "role": role,
              "target": f"q {f}->{C.RANK_MONTH[f]}" if has_q else None, "H_rows": int(H.sum()), "steps": {}}
        Is = []
        for k in KS:
            s = C.STEP[k]
            mm = float(m_of(P[f], f, X, Y, k))
            dx, dy = steps[f][X][k], steps[f][Y][k]
            direct = float(np.sqrt(np.sum(dx[H] ** 2) / np.sum(dy[H] ** 2)))
            if abs(mm - direct) > 1e-12 * direct:
                raise SystemExit(f"{arm} {s}: m from the receipts {mm!r}, direct {direct!r}")
            sxx, syy = col(P[f], f, f"ss_{X}_{s}"), col(P[f], f, f"ss_{Y}_{s}")
            sxy = col(P[f], f, f"x_{X}__{Y}_{s}")
            r = {"m": mm, "rms_H_carrier_s": float(np.sqrt(sxx / H.sum())),
                 "rms_H_plain_s": float(np.sqrt(syy / H.sum())), "cos": float(sxy / np.sqrt(sxx * syy)),
                 "changed_rows": {nm: {g: int(((d != 0) & (reg == g)).sum()) for g in C.REGIMES}
                                  for nm, d in (("carrier", dx), ("plain", dy))},
                 "outside_H_share_of_step_mass": {nm: {g: float(np.sum(d[reg == g] ** 2) / np.sum(d ** 2))
                                                       for g in ("G", "A", "C")}
                                                  for nm, d in (("carrier", dx), ("plain", dy))},
                 "kish_days_rows": kish(D[f][:, ci[f]["n"]]),
                 "kish_days_mass": {"carrier": kish(D[f][:, ci[f][f"ss_{X}_{s}"]]),
                                    "plain": kish(D[f][:, ci[f][f"ss_{Y}_{s}"]])},
                 "intervals": {}}
            if has_q:
                q = qpt[(f, k)]
                I = float(imp(mm, q))
                # SPEC 4.1 identity, stated and checked (q < 1 on every step)
                ident = (1 - mm) if q <= mm <= 1 else ((1 + mm - 2 * q) if mm < q else (1 - mm))
                if q >= 1 or abs(I - ident) > 1e-12:
                    raise SystemExit(f"{arm} {s}: the SPEC 4.1 identity does not hold")
                r.update({"q": q, "e_carrier": abs(mm - q), "e_unity": abs(1 - q), "I": I,
                          "m_minus_q": mm - q, "m_over_q": mm / q, "I_positive_band": [2 * q - 1, 1.0]})
                Is.append(I)
            ar["steps"][s] = r
        if has_q:
            ar["mean_I"] = {"point": float(np.mean(Is)), "intervals": {}}
        rec["arms"][arm] = ar

    # ---- moving-block day bootstrap with matched seeds -----------------------------------------
    rec["draw_sha256"] = {}
    for L in BLOCKS:
        rng = np.random.default_rng(5500 + L)
        st = {}
        for key in ("july_loo", "2026-07", "january", "2026-01", "december"):
            st[key] = C.mbb_starts(rng, len(days[key]), L, B)
        sig = rng.integers(1, 4, size=(B, 3))
        msi = np.array([MULTISETS.index(tuple(sorted(t))) for t in sig.tolist()])
        rec["draw_sha256"][f"L{L}"] = {k: C.sha_arr(v) for k, v in st.items()} | {"seeds": C.sha_arr(sig)}
        rec["draw_sha256"][f"L{L}"]["multiset_counts"] = {MS[j]: int((msi == j).sum()) for j in range(len(MS))}
        S = {key: C.mbb_weights(st[key], len(days[key]), L) @ D[key] for key in st}
        qd = {(f, k): q_of(S[f], S[m], f, m, k) for f, m in C.RANK_MONTH.items() for k in KS}
        for f, m in C.RANK_MONTH.items():
            for k in KS:
                rec["q"][f"{f}->{m}"][C.STEP[k]]["intervals"][f"L{L}"] = interval(qd[(f, k)])
        for arm, (f, X0, Y0, K, _) in ARMS.items():
            has_q = f in C.RANK_MONTH
            modes = {"day": None}
            if K == 3:
                modes["day+seed"] = msi
            for mode, pick in modes.items():
                Id = []
                for k in KS:
                    s = C.STEP[k]
                    if pick is None:
                        X, Y = arm_names(arm)
                        md = m_of(S[f], f, X, Y, k)
                    else:
                        jx = np.array([ci[f][f"ss_{X0}_{t}_{s}"] for t in MS])
                        jy = np.array([ci[f][f"ss_{Y0}_{t}_{s}"] for t in MS])
                        md = np.sqrt(S[f][np.arange(B), jx[pick]] / S[f][np.arange(B), jy[pick]])
                    blk = rec["arms"][arm]["steps"][s]["intervals"].setdefault(f"L{L}", {})
                    out = {"m": interval(md)}
                    if has_q:
                        qq = qd[(f, k)]
                        Id.append(imp(md, qq))
                        out.update({"I": interval(Id[-1]), "m_minus_q": interval(md - qq),
                                    "m_over_q": interval(md / qq)})
                    blk[mode] = out
                if has_q:
                    Ibar = np.mean(np.vstack(Id), axis=0)
                    rec["arms"][arm]["mean_I"]["intervals"].setdefault(f"L{L}", {})[mode] = interval(Ibar)

    # ---- the registered rule (SPEC section 6) --------------------------------------------------
    bag = rec["arms"]["june_bag"]
    c1 = bag["mean_I"]["intervals"][f"L{GATE_BLOCK}"]["day+seed"]
    c2 = {s: bag["steps"][s]["I"] for s in bag["steps"]}
    c3 = {f"seed{s}": rec["arms"][f"june_seed{s}"]["mean_I"]["point"] for s in C.SEEDS}
    gate = {"C1": {"statistic": "mean I of the June bag over v20, v21, v22; 3-day moving blocks plus matched seeds",
                   "point": bag["mean_I"]["point"], "lo": c1["lo"], "hi": c1["hi"],
                   "unreliable": c1["unreliable"], "holds": bool(c1["lo"] > 0)},
            "C2": {"I_by_step": c2, "holds": bool(all(v > 0 for v in c2.values()))},
            "C3": {"mean_I_by_seed": c3, "positive": int(sum(v > 0 for v in c3.values())),
                   "holds": bool(sum(v > 0 for v in c3.values()) >= 2)}}
    gate["verdict"] = "CLEARS" if all(gate[c]["holds"] for c in ("C1", "C2", "C3")) else "FAILS"
    gate["failed_conditions"] = [c for c in ("C1", "C2", "C3") if not gate[c]["holds"]]
    rec["gate"] = gate

    # the K = 1 held-month direction check (descriptive)
    rec["direction_check_K1"] = {}
    for arm in ("stitched_july", "stitched_january"):
        a = rec["arms"][arm]
        rec["direction_check_K1"][arm] = {
            s: {"m": a["steps"][s]["m"], "q": a["steps"][s]["q"], "m_below_1": bool(a["steps"][s]["m"] < 1),
                "I": a["steps"][s]["I"], "I_positive": bool(a["steps"][s]["I"] > 0),
                "m_interval_day": {Lk: a["steps"][s]["intervals"][Lk]["day"]["m"] for Lk in a["steps"][s]["intervals"]},
                "I_interval_day": {Lk: a["steps"][s]["intervals"][Lk]["day"]["I"] for Lk in a["steps"][s]["intervals"]}}
            for s in a["steps"]}

    # ---- fit-count record (July H) -------------------------------------------------------------
    f = "july_loo"
    nH = C.H_ROWS[f]
    fc = {}
    for k in KS:
        s = C.STEP[k]
        rms = {nm: float(np.sqrt(col(P[f], f, f"ss_{nm}_{s}") / nH)) for nm in ("P1", "P2", "P3", "P3bag", "T10")}
        cos = {}
        for nm in ("P3bag", "P1", "P2", "P3"):
            cos[f"{nm}_vs_T10"] = float(col(P[f], f, f"x_{nm}__T10_{s}")
                                        / np.sqrt(col(P[f], f, f"ss_{nm}_{s}") * col(P[f], f, f"ss_T10_{s}")))
        fc[s] = {"rms_H_plain_step_s": rms, "cos_with_K10_step": cos}
    rec["fit_count_record_july_H"] = fc

    # ---- one-seed multisets (s, s, s) against the raw member (SPEC rejected reading 6) ------------
    H = pan[f]["H"]
    one = {}
    for arm in ("P", "C"):
        for s in C.SEEDS:
            t = f"{s}{s}{s}"
            dl = [float(np.max(np.abs(levels[f][f"{arm}bag_{t}"][L][H] - levels[f][f"{arm}{s}"][L][H])))
                  for L in range(4)]
            rows = [int((levels[f][f"{arm}bag_{t}"][L][H] != levels[f][f"{arm}{s}"][L][H]).sum()) for L in range(4)]
            one[f"{arm}{s}"] = {"max_abs_diff_on_H_s_by_level": dl, "rows_differing_on_H_by_level": rows}
    for s in C.SEEDS:
        t = f"{s}{s}{s}"
        one[f"m_seed{s}"] = {C.STEP[k]: {"bag_rule_sss": float(m_of(P[f], f, f"Cbag_{t}", f"Pbag_{t}", k)),
                                         "raw_members": float(m_of(P[f], f, f"C{s}", f"P{s}", k))} for k in KS}
    rec["one_seed_multiset_vs_raw_member"] = one

    # ---- decompositions (SPEC 4.2) -------------------------------------------------------------
    dec = {"airport_m": {}, "nm_regime_m": {}, "airport_q": {}, "ownership": {}}
    worst_rec = 0.0
    for arm, (f, X0, Y0, K, _) in ARMS.items():
        X, Y = arm_names(arm)
        g = cells[f].groupby("airport").sum(numeric_only=True)
        H = pan[f]["H"]
        nm_ = pan[f]["nm"]
        dec["airport_m"][arm], dec["nm_regime_m"][arm] = {}, {}
        for k in KS:
            s = C.STEP[k]
            sx, sy = g[f"ss_{X}_{s}"], g[f"ss_{Y}_{s}"]
            SX, SY = col(P[f], f, f"ss_{X}_{s}"), col(P[f], f, f"ss_{Y}_{s}")
            for a_, b_ in ((sx.sum(), SX), (sy.sum(), SY)):
                worst_rec = max(worst_rec, abs(a_ - b_) / b_)
            contrib = (sx - sy) / SY
            m2 = SX / SY
            worst_rec = max(worst_rec, abs(contrib.sum() - (m2 - 1)))
            dec["airport_m"][arm][s] = {
                "m2_minus_1": float(m2 - 1), "sum_of_contributions": float(contrib.sum()),
                "by_airport": {a: {"H_rows": int(g.loc[a, "n"]), "contribution_to_m2_minus_1": float(contrib[a]),
                                   "plain_mass_share": float(sy[a] / SY),
                                   "m_airport": float(np.sqrt(sx[a] / sy[a])) if sy[a] > 0 else None}
                               for a in g.index}}
            dx, dy = steps[f][X][k], steps[f][Y][k]
            by = {}
            tot = 0.0
            for r_ in C.NM_REGIMES:
                sel = H & (nm_ == r_)
                a_, b_ = float(np.sum(dx[sel] ** 2)), float(np.sum(dy[sel] ** 2))
                by[r_] = {"H_rows": int(sel.sum()), "contribution_to_m2_minus_1": (a_ - b_) / SY,
                          "plain_mass_share": b_ / SY, "m_regime": float(np.sqrt(a_ / b_)) if b_ > 0 else None}
                tot += (a_ - b_) / SY
            if int(sum(v["H_rows"] for v in by.values())) != int(H.sum()):
                raise SystemExit(f"{f}: the NM regimes do not cover H")
            worst_rec = max(worst_rec, abs(tot - (m2 - 1)))
            dec["nm_regime_m"][arm][s] = {"m2_minus_1": float(m2 - 1), "sum_of_contributions": tot, "by_regime": by}
    for f, m in C.RANK_MONTH.items():
        gf = cells[f].groupby("airport").sum(numeric_only=True)
        gr = cells[m].groupby("airport").sum(numeric_only=True)
        key = f"{f}->{m}"
        dec["airport_q"][key] = {}
        nf, nr = gf["n"].sum(), gr["n"].sum()
        for k in KS:
            s = C.STEP[k]
            num = (gr["n"] / nr) * (gr[f"ss_UP_{s}"] / gr["n"])
            den = (gf["n"] / nf) * (gf[f"ss_T10_{s}"] / gf["n"])
            q2 = num.sum() / den.sum()
            worst_rec = max(worst_rec, abs(q2 - qpt[(f, k)] ** 2) / qpt[(f, k)] ** 2)
            AP = sorted(set(gf.index) | set(gr.index))
            dec["airport_q"][key][s] = {
                "q2": float(qpt[(f, k)] ** 2), "q2_from_airports": float(q2),
                "by_airport": {a: {
                    "rank_H_share": float(gr.loc[a, "n"] / nr) if a in gr.index else 0.0,
                    "fold_H_share": float(gf.loc[a, "n"] / nf) if a in gf.index else 0.0,
                    "rank_mass_share": float(num[a] / num.sum()) if a in gr.index else 0.0,
                    "fold_mass_share": float(den[a] / den.sum()) if a in gf.index else 0.0,
                    "rms_rank_s": float(np.sqrt(gr.loc[a, f"ss_UP_{s}"] / gr.loc[a, "n"])) if a in gr.index else None,
                    "rms_fold_s": float(np.sqrt(gf.loc[a, f"ss_T10_{s}"] / gf.loc[a, "n"])) if a in gf.index else None,
                    "q_airport": (float(np.sqrt((gr.loc[a, f"ss_UP_{s}"] / gr.loc[a, "n"])
                                                / (gf.loc[a, f"ss_T10_{s}"] / gf.loc[a, "n"])))
                                  if a in gr.index and a in gf.index and gf.loc[a, f"ss_T10_{s}"] > 0 else None)}
                    for a in AP}}
    rec["checks"]["decomposition_max_reconciliation_error"] = worst_rec
    if worst_rec > REL:
        raise SystemExit(f"a decomposition does not reconcile: {worst_rec:.3e}")
    for f in C.FOLDS:
        reg = pan[f]["reg"]
        dec["ownership"][f] = {}
        for nm, dk in steps[f].items():
            if nm.startswith(("Pbag_", "Cbag_")) and nm not in ("Pbag_123", "Cbag_123"):
                continue
            dec["ownership"][f][nm] = {
                C.STEP[k]: {g: {"rows_changed": int(((dk[k] != 0) & (reg == g)).sum()),
                                "mass_share": float(np.sum(dk[k][reg == g] ** 2) / np.sum(dk[k] ** 2))}
                            for g in C.REGIMES} for k in KS}
    rec["decompositions"] = dec
    rec["kish_days_fold_rows"] = {f: kish(D[f][:, ci[f]["n"]]) for f in C.FOLDS}
    rec["kish_days_rank_rows"] = {m: kish(D[m][:, ci[m]["n"]]) for m in C.RMONTHS}

    # ---- write the receipts, read the files on disk back, then the record ----------------------------
    rec["receipts"] = {}
    for k, p in RPATH.items():
        C.NO.refuse_existing([p], overwrite=False)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(texts[k])
        rec["receipts"][name(p)] = C.sha_file(p)
        if not np.array_equal(day_matrix(read_receipt(p), days[k])[0], D[k]):
            raise SystemExit(f"{name(p)}: the file on disk does not rebuild the day sums")
    rec["checks"]["day_sums_rebuilt_from_receipt_files"] = True
    rec["seconds"] = round(time.time() - t0, 1)
    C.write_json(OUT, rec)

    # ---- log ---------------------------------------------------------------------------------
    for key, qq in rec["q"].items():
        print(f"q {key}: " + ", ".join(f"{s} {v['point']:.4f} [{v['intervals']['L3']['lo']:.3f}, "
                                         f"{v['intervals']['L3']['hi']:.3f}]" for s, v in qq.items()), flush=True)
    for arm, a in rec["arms"].items():
        line = f"{arm:18s} K={a['K']} "
        for s, r in a["steps"].items():
            iv = r["intervals"]["L3"]["day"]["m"]
            line += f"| {s} m {r['m']:.4f} [{iv['lo']:.3f}, {iv['hi']:.3f}] cos {r['cos']:.3f}"
            if "I" in r:
                line += f" I {r['I']:+.4f}"
        if "mean_I" in a:
            mi = a["mean_I"]
            line += f" | mean I {mi['point']:+.4f} day [{mi['intervals']['L3']['day']['lo']:+.3f}, " \
                    f"{mi['intervals']['L3']['day']['hi']:+.3f}]"
            if "day+seed" in mi["intervals"]["L3"]:
                line += f" day+seed [{mi['intervals']['L3']['day+seed']['lo']:+.3f}, " \
                        f"{mi['intervals']['L3']['day+seed']['hi']:+.3f}]"
        print(line, flush=True)
    print(f"C1 {gate['C1']['holds']} (mean I {gate['C1']['point']:+.4f}, [{gate['C1']['lo']:+.4f}, "
          f"{gate['C1']['hi']:+.4f}]); C2 {gate['C2']['holds']} ({', '.join(f'{k} {v:+.4f}' for k, v in c2.items())}); "
          f"C3 {gate['C3']['holds']} ({', '.join(f'{k} {v:+.4f}' for k, v in c3.items())})", flush=True)
    print(f"screen {gate['verdict']}; written {name(OUT)} ({rec['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
