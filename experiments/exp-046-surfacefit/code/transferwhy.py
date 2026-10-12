"""Why the ranking rows carry more than the folds priced -- measured without a 2026 label.

The desk, 2026-09-08 17:45 UTC, after BOARD READING 12: v13 scored **248.4818**, a gain of
**14.26 s** on v12's 262.7418, against a fold-side paired gain of **+8.1280 s** and a
projection of **+11.09 s** (primary) / **+7.99 s** (conservative).  The transfer is
**1.75x** the fold gain and above even the face projection.  Both projections undershot,
so the next one is quoted wrong unless this is understood.

Nothing here is fitted to that reading (score-feedback rule).  The board number is used **once**, at the
end, as a check on a prediction this file makes from 2025 labels and 2026 observations.

**The instrument.**  The fold-side per-row gain is `2*r*d - d^2` with `r = y - pbase`, and
`y` does not exist on 2026.  But `obs_taxi` does: the aircraft's own ground run says how
long it actually taxied, and it is an input, never a label.  So define the **proxy
residual**

    g = obs_taxi - pbase

which exists on both sides, and the **proxy gain** `2*g*d - d^2`.  On the three folds both
the real gain and the proxy gain are computable, so the ratio between them is measurable;
on the two ranking months only the proxy is, and the calibrated proxy predicts what the
board should have moved.  If the proxy gain per row is much larger on 2026 than on 2025,
the base has more to explain there and the undershoot is explained without any appeal to
the board.

Read per airport-month, so a month that is dark at an airport cannot borrow another's:

    reach       rows, matched share, with-clock share, wide share, modal-form share
    quality     |obs_wheels_off - MVT_TIME|: median and the share inside 60 s
    g           the proxy residual on with-clock rows: median, RMS, IQR
    d           the correction actually made: RMS
    gain        the proxy gain per with-clock row, and on the folds the real one beside it

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/transferwhy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parent.parent

# `serve` FIRST: `arms` pulls in `imports.load`, which puts other experiments' code
# directories at the front of sys.path, and one of them also has a `serve.py`
import serve as S                                                     # noqa: E402
import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402

ARM = "c_lgb_shipped"
BOARD_V12 = 262.7418
BOARD_V13 = 248.4818
SHARES = BA.SHARES if hasattr(BA, "SHARES") else {"july_loo": 0.5571321275602379,
                                                  "january": 0.4428678724397621}


def stats(g: np.ndarray) -> dict:
    g = g[np.isfinite(g)]
    if g.size == 0:
        return {"n": 0}
    q1, q2, q3 = np.percentile(g, [25, 50, 75])
    return {"n": int(g.size), "median": float(q2), "iqr": float(q3 - q1),
            "rms": float(np.sqrt(np.mean(g ** 2))),
            "rms_about_the_median": float(np.sqrt(np.mean((g - q2) ** 2)))}


def cell(df: pd.DataFrame, d: np.ndarray, has_label: bool) -> dict:
    """One airport-month: the instrument, the proxy residual, and the two gains."""
    m = df["usable"].to_numpy(bool)
    out = {
        "rows": int(len(df)),
        "matched": int(df.obs_matched.sum()),
        "with_clock": int(m.sum()),
        "match_share": float(df.obs_matched.mean()),
        "clock_share": float(m.mean()),
    }
    if "match_mode" in df:
        out["wide_share_of_matched"] = float(
            (df.match_mode.fillna(0.0) > 0).sum() / max(int(df.obs_matched.sum()), 1))
    if "matched_on_modal" in df:
        mm = df.loc[m, "matched_on_modal"]
        out["modal_form_share_of_with_clock"] = (
            float(mm.mean()) if len(mm) and mm.notna().any() else None)
    if "obs_wheels_off" in df and "mvt_s" in df:
        dt = (df.loc[m, "obs_wheels_off"] - df.loc[m, "mvt_s"]).to_numpy("float64")
        dt = dt[np.isfinite(dt)]
        if dt.size:
            out["clock_offset_median_s"] = float(np.median(dt))
            out["clock_offset_share_within_60s"] = float(np.mean(np.abs(dt) <= 60))
    if not m.any():
        return out
    pb = df.loc[m, "pbase"].to_numpy("float64")
    ob = df.loc[m, "obs_taxi"].to_numpy("float64")
    dd = d[m]
    g = ob - pb
    out["proxy_residual_g"] = stats(g)
    out["update_d"] = stats(dd)
    out["proxy_gain_per_with_clock_row"] = float(np.mean(2 * g * dd - dd ** 2))
    out["moved_rows"] = int(np.count_nonzero(dd))
    # the quality split: a row whose observed take-off sits within a minute of the
    # recorded one is a clean clock; the rest are coarse.  2026 carries far more of the
    # first kind than the 2025 fold months do, and this is where that shows up in money.
    if "obs_wheels_off" in df and "mvt_s" in df:
        dt = (df.loc[m, "obs_wheels_off"] - df.loc[m, "mvt_s"]).to_numpy("float64")
        clean = np.isfinite(dt) & (np.abs(dt) <= 60.0)
        q = {}
        for name, sel in (("clean_clock_within_60s", clean),
                          ("coarse_clock", ~clean)):
            e = {"rows": int(sel.sum())}
            if sel.any():
                if has_label:
                    rr = (df.loc[m, "y_true"].to_numpy("float64") - pb)[sel]
                    e["real_gain_per_row"] = float(
                        np.mean(2 * rr * dd[sel] - dd[sel] ** 2))
                e["d_rms"] = float(np.sqrt(np.mean(dd[sel] ** 2)))
            q[name] = e
        out["by_clock_quality"] = q
    if has_label:
        r = (df.loc[m, "y_true"] - pb).to_numpy("float64")
        out["real_residual_r"] = stats(r)
        out["real_gain_per_with_clock_row"] = float(np.mean(2 * r * dd - dd ** 2))
        out["obs_taxi_minus_y"] = stats(ob - df.loc[m, "y_true"].to_numpy("float64"))
    return out


def main() -> None:
    rec: dict = {
        "what": ("why the ranking months carry more gain per row than the folds priced, "
                 "measured on the observation and never on the board reading"),
        "arm": ARM,
        "board": {"v12": BOARD_V12, "v13": BOARD_V13,
                  "moved_s": round(BOARD_V12 - BOARD_V13, 4),
                  "used": ("once, at the end, as a check on the prediction this file "
                           "makes from 2025 labels and 2026 observations (D11)")},
        "by_airport_month": {}, "by_month": {}, "folds": {}, "ranking": {},
    }

    # ---- the fold side: the real gain and the proxy gain, side by side ---------
    pan = A.panel_with_block(BA.load())
    dz = np.load(EXP / f"work/d-{ARM}.npz")
    for fold, month in A.FOLD_MONTH.items():
        p = pan[fold].copy()
        p["usable"] = A.served_class(p)
        d = dz[fold]
        if len(d) != len(p):
            raise SystemExit(f"{fold}: d has {len(d)} rows and the panel {len(p)}")
        p = p.reset_index(drop=True)
        if "mvt_s" not in p:
            t = pd.read_parquet(S.D.month_file(month),
                                columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"])
            t["mvt_s"] = (t.MVT_TIME_UTC_mvt
                          - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
            p = p.merge(t[["MVT_ID_mvt", "mvt_s"]], on="MVT_ID_mvt", how="left")
        rec["folds"][month] = cell(p, d, has_label=True)
        for a in sorted(p.airport.astype(str).unique()):
            sel = (p.airport.astype(str) == a).to_numpy()
            rec["by_airport_month"][f"{a} {month}"] = cell(
                p[sel].reset_index(drop=True), d[sel], has_label=True)

    # ---- the ranking side: the proxy alone -------------------------------------
    dep, _ = S.ranking_block()
    cand = pd.read_parquet(ROOT / "submissions/candidates/exp-046-surface.parquet")
    base = pd.read_parquet(S.BASE_FILE)
    pb = base.set_index("MVT_ID_mvt")[S.D.TARGET]
    pa = cand.set_index("MVT_ID_mvt")[S.D.TARGET]
    dep = dep.set_index("MVT_ID_mvt")
    dep["pbase"] = pb.reindex(dep.index).to_numpy("float64")
    dep["d"] = (pa.reindex(dep.index).to_numpy("float64") - dep["pbase"].to_numpy())
    dep = dep.reset_index()
    if not np.isfinite(dep.pbase).all():
        raise SystemExit("a ranking departure has no base prediction")
    for m, g in dep.groupby("rank_month"):
        g = g.reset_index(drop=True)
        rec["ranking"][str(m)] = cell(g, g["d"].to_numpy("float64"), has_label=False)
        for a, h in g.groupby("airport"):
            h = h.reset_index(drop=True)
            rec["by_airport_month"][f"{a} {m}"] = cell(
                h, h["d"].to_numpy("float64"), has_label=False)

    # ---- instrument 1, and why it FAILS ----------------------------------------
    # `g = obs_taxi - pbase` was meant to stand in for the residual on a month with no
    # label.  It does not: `obs_taxi` carries a systematic offset of about -400 s against
    # the base on EVERY month, 2025 and 2026 alike, and disagrees with the label itself by
    # 542-604 s RMS on the three folds, against a real residual of 175-242 s.  The
    # observation is an informative FEATURE the model learns to read, not a measurement of
    # the target, so `2*g*d - d^2` is dominated by that offset and its fold ratio to the
    # real gain is 0.65-1.33 with no stable value.  Recorded as a negative, with the
    # numbers that kill it, rather than deleted.
    rec["proxy_instrument"] = {
        "verdict": "FAILS -- not usable to project a gain onto a month with no label",
        "why": ("obs_taxi sits about 400 s BELOW the base prediction at the median on "
                "every month (2025 and 2026 alike) and disagrees with the label by "
                "542-604 s RMS on the folds, against a real residual of 175-242 s RMS"),
        "fold_ratio_real_over_proxy": {
            m: (rec["folds"][m]["real_gain_per_with_clock_row"]
                / rec["folds"][m]["proxy_gain_per_with_clock_row"])
            for m in rec["folds"]
            if rec["folds"][m].get("proxy_gain_per_with_clock_row")},
    }

    # ---- instrument 2: the population, which is what actually moved ---------------
    # Each airport's gain per with-clock row is measured on the FOLD months and applied to
    # the RANKING months' with-clock rows.  That is only honest where the two populations
    # are alike.  They are not: the 2026 harvest is far better covered than the 2025 fold
    # months at four airports, so a per-row gain measured on a thin slice was multiplied
    # by a population many times larger and differently composed.
    board_rows = {m: rec["ranking"][m]["rows"] for m in rec["ranking"]}
    tot = sum(board_rows.values())
    SEASON = {"2026-07": ["2025-07"], "2026-01": ["2025-01", "2025-12"]}
    MIN_FOLD_CLOCK = 200

    def fold_cells(a: str) -> dict:
        return {f: rec["by_airport_month"][f"{a} {f}"]
                for f in ("2025-01", "2025-07", "2025-12")
                if f"{a} {f}" in rec["by_airport_month"]
                and rec["by_airport_month"][f"{a} {f}"].get("with_clock", 0) >= MIN_FOLD_CLOCK}

    variants = {"pooled": {}, "seasonal": {}, "like_for_like": {},
                "quality_mixed": {}}
    QB = ("clean_clock_within_60s", "coarse_clock")
    # the fold gain per row in each quality bin, per airport, pooled over the fold months
    # that have enough rows in that bin; the month-level number is the fallback
    def bin_gain(a: str | None) -> dict:
        g = {}
        for b in QB:
            num = den = 0.0
            for f in ("2025-01", "2025-07", "2025-12"):
                k = f"{a} {f}" if a else f
                c = (rec["by_airport_month"] if a else rec["folds"]).get(k, {})
                e = c.get("by_clock_quality", {}).get(b, {})
                if e.get("rows", 0) >= MIN_FOLD_CLOCK and "real_gain_per_row" in e:
                    num += e["real_gain_per_row"] * e["rows"]
                    den += e["rows"]
            g[b] = (num / den) if den else None
        return g
    global_bin = bin_gain(None)
    coverage = {}
    for m in rec["ranking"]:
        for a in sorted({k.split(" ")[0] for k in rec["by_airport_month"]}):
            rk = rec["by_airport_month"].get(f"{a} {m}")
            if not rk or not rk.get("with_clock"):
                continue
            fc = fold_cells(a)
            if not fc:
                continue
            # pooled: every fold month, weighted by its with-clock rows (what was quoted)
            w = np.array([v["with_clock"] for v in fc.values()], float)
            gpool = float(np.average([v["real_gain_per_with_clock_row"] for v in fc.values()],
                                     weights=w))
            # seasonal: only the fold months of the same season
            fs = {f: v for f, v in fc.items() if f in SEASON[m]}
            gseas = (float(np.average([v["real_gain_per_with_clock_row"] for v in fs.values()],
                                      weights=[v["with_clock"] for v in fs.values()]))
                     if fs else gpool)
            # like for like: the fold month whose CLOCK SHARE at this airport is nearest
            # the ranking month's, because that is the fold whose with-clock population is
            # the same kind of population
            best = min(fc.items(),
                       key=lambda kv: abs(np.log(max(kv[1]["clock_share"], 1e-6)
                                                 / max(rk["clock_share"], 1e-6))))
            variants["pooled"][f"{a} {m}"] = gpool * rk["with_clock"] / tot
            variants["seasonal"][f"{a} {m}"] = gseas * rk["with_clock"] / tot
            variants["like_for_like"][f"{a} {m}"] = (
                best[1]["real_gain_per_with_clock_row"] * rk["with_clock"] / tot)
            # the quality mix: the ranking month's OWN split of clean against coarse
            # clocks, priced at the fold's gain per row in each bin
            ab = bin_gain(a)
            mix = rk.get("by_clock_quality", {})
            qm = 0.0
            for b in QB:
                gb = ab[b] if ab[b] is not None else global_bin[b]
                if gb is None:
                    gb = gpool
                qm += gb * mix.get(b, {}).get("rows", 0) / tot
            variants["quality_mixed"][f"{a} {m}"] = qm
            coverage_extra = {
                "ranking_quality_mix": {b: mix.get(b, {}).get("rows", 0) for b in QB},
                "fold_gain_by_quality": ab}
            coverage[f"{a} {m}"] = {
                "ranking_clock_share": rk["clock_share"],
                "ranking_with_clock": rk["with_clock"],
                "fold_clock_share": {f: v["clock_share"] for f, v in fc.items()},
                "fold_with_clock": {f: v["with_clock"] for f, v in fc.items()},
                "coverage_ratio_vs_the_nearest_fold": (
                    rk["clock_share"] / max(best[1]["clock_share"], 1e-6)),
                "like_for_like_fold": best[0],
                "gain_pooled": gpool, "gain_seasonal": gseas,
                "gain_like_for_like": best[1]["real_gain_per_with_clock_row"],
                **coverage_extra,
            }

    def move(G: float) -> float:
        return float(BOARD_V12 - np.sqrt(max(BOARD_V12 ** 2 - G, 0.0)))

    G_real = BOARD_V12 ** 2 - BOARD_V13 ** 2
    rec["projection_variants"] = {
        k: {"mse_gain_per_ranking_row_s2": float(sum(v.values())),
            "board_move_s": move(sum(v.values())),
            "ratio_to_realised": float(sum(v.values()) / G_real)}
        for k, v in variants.items()}
    rec["projection_variants"]["realised"] = {
        "mse_gain_per_ranking_row_s2": float(G_real), "board_move_s": move(G_real),
        "ratio_to_realised": 1.0,
        "note": ("the board number enters ONLY here, as the check on the three "
                 "projections above; nothing is fitted to it (D11)")}
    rec["coverage_by_airport_month"] = coverage
    rec["what_the_old_projection_said"] = {"primary_s": 11.09, "conservative_s": 7.99,
                                           "realised_s": move(G_real)}

    (EXP / "analysis/transfer-why.json").write_text(
        json.dumps(rec, indent=2, sort_keys=True))

    for k, v in rec["projection_variants"].items():
        print(f"{k:14s} {v['mse_gain_per_ranking_row_s2']:9.1f} s2/row  "
              f"{v['board_move_s']:7.3f} s  x{v['ratio_to_realised']:.3f}")
    print("fold  real/row  proxy/row")
    for m, v in sorted(rec["folds"].items()):
        print(f"  {m}  {v['real_gain_per_with_clock_row']:9.1f}  "
              f"{v['proxy_gain_per_with_clock_row']:9.1f}")
    print("rank  proxy/row  with_clock  g rms   d rms")
    for m, v in sorted(rec["ranking"].items()):
        print(f"  {m}  {v['proxy_gain_per_with_clock_row']:9.1f}  {v['with_clock']:10d}  "
              f"{v['proxy_residual_g']['rms']:6.1f}  {v['update_d']['rms']:6.1f}")
    print("the four airports whose coverage moved most:")
    for k, c in sorted(rec["coverage_by_airport_month"].items(),
                       key=lambda kv: -abs(np.log(max(kv[1]["coverage_ratio_vs_the_nearest_fold"], 1e-6))))[:6]:
        print(f"  {k}  ranking clock {c['ranking_clock_share']:.3f} against fold "
              f"{c['like_for_like_fold']} {c['fold_clock_share'][c['like_for_like_fold']]:.3f}"
              f"  x{c['coverage_ratio_vs_the_nearest_fold']:.2f}")


if __name__ == "__main__":
    main()
