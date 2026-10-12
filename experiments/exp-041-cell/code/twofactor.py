"""Step 3 -- Q10's two-factor predictor: does the shared error arrive on the
runway channel or on the operator channel?

The queue writes it as: *a minimal predictor routes neighboring AOBT-LOBT
innovations through same-runway/different-operator and same-operator/
different-runway groups, excluding the query.*  The anatomy (T6) already says
what the LABEL side looks like -- `Var(runway | window)` is 5,977 / 4,622 /
3,886 s^2 against `Var(operator | window)` 2,759 / 583 / 2,037 -- so this file
asks whether an OBSERVABLE routed the same way reproduces that split.

The innovation is `u = (AOBT - LOBT) - g(apt, hour, dow)`, with `g` a shrunk
back-off group mean fitted on the OTHER ELEVEN MONTHS: the scoring month never
contributes to its own centring, and neither runway nor operator enters `g`, so
the routed group means keep the two factors the screen is trying to separate.
`AOBT_3_flt` and `LOBT_flt` are ranking-file inputs and neither is the masked
departure block time (`docs/input-rule.md`).

Arms, one learner, one capacity, one seed, one set of rows, four extra columns
each so the one- and two-factor arms are matched:

    g0     the same 19-feature base as step 2                     (the base)
    one    + cell and wide-window innovation means and counts, NOT split   (+4)
    two    + same-runway/different-operator and same-operator/different-
             runway innovation means and counts, query excluded            (+4)
    two_c  + the two-factor block read seven days away              (placebo, +4)

Coverage is checked BEFORE fitting, on the queue's rule: 200 matched pairs per
relation across 20 dates per airport.  Unsupported airports are reported, never
relaxed, and the readout is given twice -- on every eligible row and on the
supported airports alone.

    python twofactor.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import base as B  # noqa: E402
import order as O  # noqa: E402

sys.path.insert(0, str(B.S39))
import q04 as Q4      # noqa: E402
import rowfeat as R   # noqa: E402
import anatomy as AN  # noqa: E402

EXP = B.EXP
FOLDS = B.FOLDS
D = B.D
PK = B.PK

# ---- predeclared -------------------------------------------------------------
PAIRS_PER_RELATION = 200
DATES_PER_RELATION = 20
MIN_GROUP = 30              # the nuisance back-off's minimum cell
CLIP = 7200.0               # AOBT - LOBT, seconds
PLACEBO_DAYS = 7
WIDE_BINS = 4
#: `q04h1` is exp-039 Q04's own `h1` control on this base; `one_q04` and
#: `two_q04` put the one- and two-factor blocks BEYOND it, which is what the
#: assignment's "beyond Q04 and the existing windows" clause tests.
ARMS = ("g0", "one", "two", "two_c", "q04h1", "one_q04", "two_q04")
CAPS = O.CAPS
HEADLINE_CAP = O.HEADLINE_CAP
#: The nuisance levels.  Runway and operator are DELIBERATELY absent: `g` must
#: not absorb the two factors the routed means are meant to separate.
LEVELS = (("apt", "hour", "dow"), ("apt", "hour"), ("apt",))


class Nuisance:
    """Shrunk back-off mean of `AOBT - LOBT`, fitted on months, never on the
    scoring month.  The same shape as exp-039 Q04's arrival nuisance."""

    def fit(self, months: list[str]) -> "Nuisance":
        tot = {lv: None for lv in LEVELS}
        self.g, n = 0.0, 0
        for mth in months:
            a = _delay_frame(mth)
            self.g += float(a["u"].sum()); n += len(a)
            for lv in LEVELS:
                s = a.groupby(list(lv))["u"].agg(["sum", "size"])
                tot[lv] = s if tot[lv] is None else tot[lv].add(s, fill_value=0)
            del a
        self.g /= max(n, 1)
        self.tab = {}
        for lv in LEVELS:
            t = tot[lv]
            self.tab[lv] = (t["sum"] / t["size"]).where(t["size"] >= MIN_GROUP)
        self.n_train = n
        return self

    def predict(self, a: pd.DataFrame) -> np.ndarray:
        out = np.full(len(a), np.nan)
        for lv in LEVELS:
            idx = (pd.MultiIndex.from_arrays([a[c].to_numpy() for c in lv])
                   if len(lv) > 1 else pd.Index(a[lv[0]].to_numpy()))
            v = self.tab[lv].reindex(idx).to_numpy()
            out = np.where(np.isnan(out), v, out)
        return np.where(np.isnan(out), self.g, out)


_DELAY_CACHE: dict[str, pd.DataFrame] = {}


def _delay_frame(month: str) -> pd.DataFrame:
    """`AOBT - LOBT` per departure of a month, with the nuisance's own keys.

    Cached: the same twelve months are read by all three folds' nuisance fits.
    """
    if month in _DELAY_CACHE:
        return _DELAY_CACHE[month]
    a = B.attrs(month)
    u = np.clip(a["aobt"].to_numpy() - a["lobt"].to_numpy(), -CLIP, CLIP)
    t = a["t_attr"].to_numpy()
    out = pd.DataFrame({"u": u, "apt": a["adep"].to_numpy(),
                        "hour": np.floor(np.mod(t, 86400) / 3600),
                        "dow": np.floor(t / 86400) % 7,
                        "mid": a["mid"].to_numpy()})
    out = out[np.isfinite(out["u"])].reset_index(drop=True)
    _DELAY_CACHE[month] = out
    return out


def _skey(*cols) -> np.ndarray:
    """The same string composite key `base.codes_of` factorizes, kept as strings
    so an aggregate built on one set of bins can be LOOKED UP at another."""
    key = np.asarray(cols[0]).astype(str)
    for c in cols[1:]:
        key = np.char.add(np.char.add(key, "|"), np.asarray(c).astype(str))
    return key


def q04_block(month: str, m: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """exp-039 Q04's arrival-innovation columns, on this panel's rows.

    `assignments/cell.md` step 3 says "Q04's h0/h1 controls reused" (Sol round 1,
    blocking finding 4).  `h0` is the matched base every arm here already
    carries; `h1` is that base plus these two columns -- the cell's arrival
    taxi-in innovation and its four-bin wide-window version, with the nuisance
    fitted on the OTHER ELEVEN months.  Rebuilt through Q04's own functions, so
    it is that screen's block and not a lookalike.
    """
    if Q4.BIN != B.BIN:
        raise SystemExit(f"Q04's bin is {Q4.BIN} s and this cell is {B.BIN} s")
    nu = Q4.Nuisance().fit([x for x in D.MONTHS if x != month])
    a = Q4.arrivals(month)
    cs = Q4.cell_stats(a, a["taxi_in"].to_numpy() - nu.predict(a))
    W4 = Q4.wide_window(cs)
    key = pd.MultiIndex.from_arrays([m["apt"].to_numpy(), m["bin"].to_numpy()],
                                    names=["apt", "bin"])
    z = cs["z"].reindex(key).to_numpy()
    zp = W4["z_wide"].reindex(key).to_numpy()
    return np.nan_to_num(z, nan=0.0), np.nan_to_num(zp, nan=0.0)


def routed(code_all, n_all, code_r, n_r, code_o, n_o, code_ro, n_ro, u, have
           ) -> dict:
    """Query-excluded group means of `u` over the four crossover relations.

    Rows whose own `u` is missing still contribute a group mean (they are
    excluded from every sum by construction), so `have` marks the rows that are
    IN the sums, not the rows that receive a feature.
    """
    z = np.where(have, u, 0.0)
    h = have.astype("float64")
    S, _ = B.group_sums(code_all, n_all, z); N, _ = B.group_sums(code_all, n_all, h)
    SR, _ = B.group_sums(code_r, n_r, z);    NR, _ = B.group_sums(code_r, n_r, h)
    SO, _ = B.group_sums(code_o, n_o, z);    NO, _ = B.group_sums(code_o, n_o, h)
    SRO, _ = B.group_sums(code_ro, n_ro, z); NRO, _ = B.group_sums(code_ro, n_ro, h)
    out = {}
    for name, (s, c) in {
            "all": (S - z, N - h),
            "same_rwy_diff_op": (SR - SRO, NR - NRO),
            "diff_rwy_same_op": (SO - SRO, NO - NRO)}.items():
        out[f"f_{name}"] = np.where(c > 0, s / np.maximum(c, 1e-9), np.nan)
        out[f"n_{name}"] = c
    return out


def run_fold(fold: str, pan: pd.DataFrame, rng, preds: dict) -> dict:
    month = R.month_of(fold)
    others = [m for m in D.MONTHS if m != month]
    nu = Nuisance().fit(others)
    df = _delay_frame(month)
    u_by_id = pd.Series(df["u"].to_numpy() - nu.predict(df),
                        index=df["mid"].to_numpy())

    m = B.panel(fold, pan)
    r = m["r"].to_numpy("float64")
    y = m["y_true"].to_numpy("float64")
    ids = m["MVT_ID_mvt"].to_numpy("float64")
    p10 = m["p10"].to_numpy("float64")
    days = m["day"].to_numpy()
    blk = B.blocks_of(days)
    apt = m["apt"].to_numpy(); bins = m["bin"].to_numpy()
    rwy = m["rwy"].to_numpy(); ope = m["op"].to_numpy()
    ei = np.flatnonzero(m["elig"].to_numpy())

    u = u_by_id.reindex(ids).to_numpy()
    have = np.isfinite(u)

    # the neighbour stream is EVERY departure of the month in the cell, whatever
    # its serving regime: all AOBT and LOBT clocks are given at scoring time
    c_all, n_all = B.codes_of(apt, bins)
    c_r, n_r = B.codes_of(apt, bins, rwy)
    c_o, n_o = B.codes_of(apt, bins, ope)
    c_ro, n_ro = B.codes_of(apt, bins, rwy, ope)
    F = routed(c_all, n_all, c_r, n_r, c_o, n_o, c_ro, n_ro, u, have)

    # a wide-window one-factor pair, so the one-factor arm has four columns too
    key = pd.MultiIndex.from_arrays([apt, bins], names=["apt", "bin"])
    su = pd.Series(np.where(have, u, 0.0), index=key).groupby(level=[0, 1]).sum()
    cu = pd.Series(have.astype("float64"), index=key).groupby(level=[0, 1]).sum()
    wide_s = AN.wide_count(su, WIDE_BINS).reindex(key).to_numpy()
    wide_c = AN.wide_count(cu, WIDE_BINS).reindex(key).to_numpy()
    f_wide = np.where(wide_c - have > 0,
                      (wide_s - np.where(have, u, 0.0)) / np.maximum(wide_c - have, 1e-9),
                      np.nan)

    # ---- the placebo: the same two-factor block read SEVEN DAYS AWAY ---------
    #  Sol round 1, blocking finding 6.  The first version binned every query at
    #  its own shifted clock and THEN grouped, which moves the whole month
    #  together: two rows in the same real cell land in the same shifted cell, so
    #  the "placebo" reproduced the real grouping and only relabelled it.  The
    #  aggregates are now built on the REAL bins, once, and each query looks its
    #  own key up at the SHIFTED bin -- the neighbours of another day, with the
    #  query's own row never inside them (its own bin is not the shifted bin), so
    #  no exclusion is needed and none is applied.
    q = O.query_times(m, PLACEBO_DAYS * 86400.0)
    bins_c = np.floor(q / B.BIN).astype("int64")
    zc = np.where(have, u, 0.0); hc = have.astype("float64")

    def _agg(*cols) -> tuple[pd.Series, pd.Series]:
        k = _skey(*cols)
        return (pd.Series(zc).groupby(k).sum(), pd.Series(hc).groupby(k).sum())

    s_r, n_r_ = _agg(apt, bins, rwy)
    s_o, n_o_ = _agg(apt, bins, ope)
    s_ro, n_ro_ = _agg(apt, bins, rwy, ope)
    Fc = {}
    for nm, (sa, na, sb, nb, ka, kb) in {
            "same_rwy_diff_op": (s_r, n_r_, s_ro, n_ro_,
                                 _skey(apt, bins_c, rwy),
                                 _skey(apt, bins_c, rwy, ope)),
            "diff_rwy_same_op": (s_o, n_o_, s_ro, n_ro_,
                                 _skey(apt, bins_c, ope),
                                 _skey(apt, bins_c, rwy, ope))}.items():
        s1 = sa.reindex(ka).fillna(0.0).to_numpy()
        k1 = na.reindex(ka).fillna(0.0).to_numpy()
        s2 = sb.reindex(kb).fillna(0.0).to_numpy()
        k2 = nb.reindex(kb).fillna(0.0).to_numpy()
        Fc[f"f_{nm}"] = np.where(k1 - k2 > 0,
                                 (s1 - s2) / np.maximum(k1 - k2, 1e-9), np.nan)
        Fc[f"n_{nm}"] = k1 - k2
    #  how far the placebo actually moves: the share of rows whose shifted
    #  same-runway group is a DIFFERENT set of neighbours from the real one
    _real_n = F["n_same_rwy_diff_op"]
    placebo_moved = float(np.mean(Fc["n_same_rwy_diff_op"][ei] != _real_n[ei]))

    # ---- the coverage rule, checked BEFORE the fit --------------------------
    cov, supported = {}, []
    for a in sorted(set(apt[ei])):
        ka = np.isin(np.arange(len(m)), ei) & (apt == a)
        cov[a] = {}
        for nm in ("same_rwy_diff_op", "diff_rwy_same_op"):
            pr = float(np.where(ka, F[f"n_{nm}"], 0.0).sum()) / 2.0
            dts = int(len(np.unique(days[ka & (F[f"n_{nm}"] > 0)])))
            cov[a][nm] = {"pairs": int(round(pr)), "dates": dts,
                          "met": bool(pr >= PAIRS_PER_RELATION
                                      and dts >= DATES_PER_RELATION)}
        if all(v["met"] for v in cov[a].values()):
            supported.append(a)
    sup_mask = np.isin(apt, supported)

    ac = AN.arrival_cells(month)
    base, _ = O.base_features(m, ac, month)
    z_q4, zp_q4 = q04_block(month, m)
    one_blk = np.column_stack([F["f_all"], F["n_all"], f_wide, wide_c])
    two_blk = np.column_stack([F["f_same_rwy_diff_op"], F["n_same_rwy_diff_op"],
                               F["f_diff_rwy_same_op"], F["n_diff_rwy_same_op"]])
    two_c_blk = np.column_stack([Fc["f_same_rwy_diff_op"],
                                 Fc["n_same_rwy_diff_op"],
                                 Fc["f_diff_rwy_same_op"],
                                 Fc["n_diff_rwy_same_op"]])
    q4_blk = np.column_stack([z_q4, zp_q4])
    #  One arm matrix at a time (see order.py): seven copies of a 172-column
    #  block do not fit beside another worker on this laptop.
    BLOCK = {"g0": (), "one": (one_blk,), "two": (two_blk,),
             "two_c": (two_c_blk,), "q04h1": (q4_blk,),
             "one_q04": (q4_blk, one_blk), "two_q04": (q4_blk, two_blk)}

    def arm_matrix(name: str) -> np.ndarray:
        b = BLOCK[name]
        return base if not b else np.column_stack(
            [base] + [np.asarray(x, "float32") for x in b])

    out = {"fold": fold, "month": month, "n_eligible": int(len(ei)),
           "nuisance_train_departures": int(nu.n_train),
           "nuisance_months": len(others),
           "u_present_pct": round(100 * float(have[ei].mean()), 2),
           "u_std_s": round(float(np.nanstd(u[ei])), 1),
           "coverage": cov, "supported_airports": supported,
           "unsupported_airports": [a for a in cov if a not in supported],
           "n_rows_supported": int(sup_mask[ei].sum()),
           "n_base_features": int(base.shape[1]),
           "placebo_group_differs_pct": round(100 * placebo_moved, 2),
           "placebo_note": "share of eligible rows whose 7-day-shifted "
                           "same-runway group has a different size from the real "
                           "one; the first version of this placebo scored 0 here, "
                           "because shifting every row together preserves the "
                           "grouping (Sol round 1, blocking finding 6)",
           "q04_h1_columns": 2,
           "feature_coverage_pct": {
               k: round(100 * float(np.isfinite(F[k][ei]).mean()), 2)
               for k in ("f_all", "f_same_rwy_diff_op", "f_diff_rwy_same_op")},
           "arms": {}}

    H = {}
    for name in ARMS:
        Xn = arm_matrix(name)
        for cap in CAPS:
            h = O.fit_arm(Xn, r, ei, blk, cap)
            H[f"{name}@{cap}"] = h
            p_new = p10 + h
            rec = {"cap": cap, "n_features": int(Xn.shape[1])}
            for tag, sel in (("all_eligible", ei),
                             ("supported_airports", ei[sup_mask[ei]])):
                A_ = float((r[sel] * h[sel]).mean()); Bq = float((h[sel] ** 2).mean())
                v = 2 * r[sel] * h[sel] - h[sel] ** 2
                pt, lo, hi = B.boot_ratio(v, np.ones(len(sel)), days[sel], rng)
                rec[tag] = {"n": int(len(sel)), "A_mean_r_h": round(A_, 3),
                            "B_mean_h2": round(Bq, 3), "mse_gain_s2": round(pt, 3),
                            "ci": [round(lo, 3), round(hi, 3)]}
            rec["multiday"] = PK.boot_blocks(2 * r[ei] * h[ei] - h[ei] ** 2,
                                             days[ei], rng)
            rec["rmse_all_v10"] = round(D.rmse(y, p10), 4)
            rec["rmse_all_new"] = round(D.rmse(y, p_new), 4)
            rec["gain_all"] = round(rec["rmse_all_v10"] - rec["rmse_all_new"], 4)
            rec["rmse_clean_v10"] = round(D.rmse_clean(y, p10, ids, fold), 4)
            rec["rmse_clean_new"] = round(D.rmse_clean(y, p_new, ids, fold), 4)
            rec["gain_clean"] = round(rec["rmse_clean_v10"] - rec["rmse_clean_new"], 4)
            out["arms"][f"{name}@{cap}"] = rec
            if cap == HEADLINE_CAP:
                preds.setdefault(name, {})[fold] = p_new
                PK.dates("twofactor", name, fold, days, y, p10, p_new)
        del Xn
    for cap in CAPS:
        for a1, a2 in (("one", "g0"), ("two", "g0"), ("two_c", "g0"),
                       ("two", "one"), ("two", "two_c"),
                       ("q04h1", "g0"), ("one_q04", "q04h1"),
                       ("two_q04", "q04h1"), ("two_q04", "one_q04")):
            k1, k2 = f"{a1}@{cap}", f"{a2}@{cap}"
            h1, h2 = H[k1][ei], H[k2][ei]
            dv = (2 * r[ei] * h1 - h1 ** 2) - (2 * r[ei] * h2 - h2 ** 2)
            pt, lo, hi = B.boot_ratio(dv, np.ones(len(ei)), days[ei], rng)
            out["arms"][k1].setdefault("vs", {})[a2] = {
                "delta_s2": round(pt, 3), "ci": [round(lo, 3), round(hi, 3)],
                "clear_of_zero": bool(lo > 0 or hi < 0),
                "multiday": PK.boot_blocks(dv, days[ei], rng)}
    return out


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(B.SEED)
    pan = B.V.load()
    recs, preds = {}, {}
    for f in FOLDS:
        recs[f] = run_fold(f, pan[f], rng, preds)
        c = recs[f]
        print(f"[{f}] {c['n_eligible']:,} eligible; AOBT-LOBT present on "
              f"{c['u_present_pct']} %, innovation sd {c['u_std_s']} s; "
              f"coverage met at {len(c['supported_airports'])} of "
              f"{len(c['coverage'])} airports "
              f"(unsupported {c['unsupported_airports']})", flush=True)
        print(f"    feature coverage {c['feature_coverage_pct']}", flush=True)
        for k, v in c["arms"].items():
            print(f"    {k:10s} {v['n_features']:2d} feats  2A-B "
                  f"{v['all_eligible']['mse_gain_s2']:+9.2f} "
                  f"[{v['all_eligible']['ci'][0]:+.2f}, "
                  f"{v['all_eligible']['ci'][1]:+.2f}] s^2  (supported "
                  f"{v['supported_airports']['mse_gain_s2']:+.2f})  all-row "
                  f"{v['gain_all']:+.4f} s"
                  + ("\n               vs " + "; ".join(
                      f"{a} {d['delta_s2']:+.1f} [{d['ci'][0]:+.1f}, "
                      f"{d['ci'][1]:+.1f}]{'*' if d['clear_of_zero'] else ''}"
                      for a, d in v["vs"].items()) if "vs" in v else ""),
                  flush=True)
    board = {n: PK.primary(pan, preds[n], f"arm {n} at capacity {HEADLINE_CAP}")
             for n in ARMS}
    #  an interval on the PRIMARY endpoint itself (Sol round 1, blocking
    #  finding 5): the board is a weighted quadrature of two fold RMSEs, so it is
    #  bootstrapped as one quantity with blocks drawn inside each fold
    for n in ARMS:
        parts = {f: (B.V.day(pan[f]["t"].to_numpy("float64")),
                     (pan[f]["y_true"].to_numpy("float64")
                      - pan[f]["p10"].to_numpy("float64")) ** 2,
                     (pan[f]["y_true"].to_numpy("float64")
                      - np.asarray(preds[n][f], "float64")) ** 2)
                 for f in FOLDS}
        board[n]["board_interval"] = B.board_boot(parts, rng)
    out = {"screen": "twofactor",
           "predeclared": {"levels": [list(l) for l in LEVELS],
                           "min_group": MIN_GROUP, "clip_s": CLIP,
                           "placebo_days": PLACEBO_DAYS, "caps": CAPS,
                           "headline_cap": HEADLINE_CAP, "blocks": B.BLOCKS,
                           "seed": B.SEED, "bin_s": B.BIN,
                           "pairs_per_relation": PAIRS_PER_RELATION,
                           "dates_per_relation": DATES_PER_RELATION,
                           "eligible_regime": B.ELIGIBLE_REGIME,
                           "hold_airports": list(B.HOLD_AIRPORTS),
                           "advance_clause": "the second factor predicts current "
                                             "residuals outside the discovery "
                                             "blocks, beyond Q04 and the existing "
                                             "windows"},
           "folds": recs, "primary": board["two"], "boards": board,
           "wall_s": round(time.time() - t0, 1)}
    p = EXP / "analysis" / "twofactor.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    for n in ARMS:
        print(f"  oof: {B.write_oof(f'twofactor-{n}', pan, preds[n])}")
    B.write_packet("twofactor", out, pan)
    print("\n  PRIMARY all-row january/july board, capacity " + HEADLINE_CAP + ":")
    for n in ARMS:
        b = board[n]["board_all_row"]
        bi = board[n]["board_interval"]
        print(f"    {n:8s} {b['v10']:.4f} -> {b['arm']:.4f} ({b['gain']:+.4f} s)"
              f"   clean {board[n]['board_clean']['gain']:+.4f} s"
              f"   1d {bi['ci_block_1d']} 3d {bi['ci_block_3d']}")
    print(f"  written: {p}  ({out['wall_s']} s)")


if __name__ == "__main__":
    main()
