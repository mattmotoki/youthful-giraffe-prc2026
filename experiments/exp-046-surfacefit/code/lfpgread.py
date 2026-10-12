"""LFPG, section 17 row 4, as a READ before any arm (desk, 2026-09-09).

The hold on LFPG is an AIRPORT NAME, and section 12.1 said so honestly: the ranking that
justifies it (served wide-share over fitted wide-share, LFPG 2026-01 at 5.0x against
LSZH's 2.54x) is durable, but the cut between them was drawn after seeing it.  A name is
not a rule, and it refuses every LFPG row with a clock -- 2,061 of them now, not the 1,612
section 17 quoted, because the turnaround pass added 449.

This asks whether an INSTRUMENT can replace the name: do the observation-quality fields
the block already carries -- point count, cadence, gaps, span, stop-at-stand -- tell a
usable LFPG observation from a broken one?

Three rules keep the answer honest:

  * the gate is LABEL-FREE at serving time: it reads only fields present on every ranking
    row, never `y` and never a quantity derived from `y`;
  * its threshold is set on the OTHER airports' rows only.  LFPG never enters the
    selection, so the airport axis is held out (the rule beside section 13.1) and the
    answer cannot be LFPG fitted to itself;
  * the threshold is additionally leave-one-fold-out, so the fold axis is held out too.

Then the gated LFPG subset is priced on the folds LFPG is LIT on -- 2025-07 (396 rows) and
2025-12 (1,005); 2025-01 carries ZERO LFPG rows with a clock and is structurally silent,
which is a fact about the harvest and not a failed fold.  No fit runs: the served arm's
saved update vector is masked, as section 12.1 priced its four rules.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/lfpgread.py
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

import serve as S                                                     # noqa: E402
import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402

#: the arm this lane serves, on the 122c7d84 blocks
ARM = "c_lgb_shipped@leaves_255x300_lr050+guard995"
HELD = "LFPG"
#: v17's board level, for the level currency of section 16.3
LEVEL = 246.6571
N_RANKING = 344841

#: the five fields the desk named, and nothing else.  A wider family is a wider
#: selection; these are pre-declared before any threshold is read.
QUALITY = {
    "seg_points":       "point count -- every point of the ground segment",
    "n_ground_points":  "point count -- the ground points alone",
    "seg_dt_median_s":  "cadence -- the median gap between consecutive points",
    "seg_gap_max_s":    "gaps -- the largest hole in the segment",
    "seg_span_s":       "span -- first point to last",
    "stand_stop_s":     "stop-at-stand -- how long the aircraft sat before it moved",
}
#: read alongside, never in the gate family: these are identity quality, not observation
#: quality, and section 12.1 already priced a rule built on `clock_ok` and refuted it.
BESIDE = ["dt_takeoff", "clock_ok", "obs_ambiguous", "obs_n_candidates",
          "matched_on_modal", "match_mode", "time_to_250m", "seen_before_move_s"]

QUANTILES = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]
OUT = EXP / "analysis/lfpg-read.json"


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return None
    ra = pd.Series(a[m]).rank().to_numpy()
    rb = pd.Series(b[m]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def rms(v: np.ndarray) -> float | None:
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else None


def main() -> None:
    rng = np.random.default_rng(A.SEED)
    pan = A.panel_with_block(BA.load())
    dep, _ = S.ranking_block()
    out = {"what": __doc__.strip().splitlines()[0], "arm": ARM, "held_airport": HELD,
           "level_s": LEVEL,
           "gate_family": QUALITY,
           "rules": {"label_free": "the gate reads only block columns, never y",
                     "airport_axis": f"every threshold is set on rows with airport != {HELD}",
                     "fold_axis": "leave-one-fold-out; the ranking cut uses all three folds"}}

    # ------------------------------------------------- part A: the population
    pop = {}
    for f in A.FOLDS:
        p = pan[f]
        u = p.usable.to_numpy(bool)
        ap = p.airport.to_numpy().astype(str)
        pop[A.FOLD_MONTH[f]] = {
            "fold": f, "rows": int(len(p)), "with_clock": int(u.sum()),
            "lfpg_rows": int((ap == HELD).sum()),
            "lfpg_with_clock": int((u & (ap == HELD)).sum())}
    served = dep.usable.to_numpy(bool) & ~dep.noac.to_numpy(bool)
    rap = dep.airport.to_numpy().astype(str)
    rmo = dep.rank_month.to_numpy().astype(str)
    for m in sorted(set(rmo)):
        pop[m] = {"fold": None, "rows": int((rmo == m).sum()),
                  "with_clock": int((served & (rmo == m)).sum()),
                  "lfpg_rows": int(((rap == HELD) & (rmo == m)).sum()),
                  "lfpg_with_clock": int((served & (rap == HELD) & (rmo == m)).sum())}
    out["population"] = pop
    out["lfpg_held_ranking_rows"] = int((served & (rap == HELD)).sum())
    out["lit_folds"] = [f for f in A.FOLDS
                        if pop[A.FOLD_MONTH[f]]["lfpg_with_clock"] > 0]
    out["dark_folds"] = [f for f in A.FOLDS
                         if pop[A.FOLD_MONTH[f]]["lfpg_with_clock"] == 0]

    # ---------------------------------- part B: quality against the residual
    # THE RESIDUAL OF THE OBSERVATION: how wrong the ground clock is as an estimate of
    # the label.  Label-bearing, so it is the READ and never the gate.
    z = np.load(S.npz_path(ARM))
    full = {f: z[f].astype("float64") for f in A.FOLDS}

    def frame(f: str) -> pd.DataFrame:
        """The usable rows of one fold, carrying the served arm's own update for each
        row, so nothing downstream has to line a vector up by position."""
        p = pan[f]
        m = p.usable.to_numpy(bool)
        d = p.loc[m].copy()
        d["upd"] = full[f][m]
        d["obs_err"] = d.obs_taxi.to_numpy("float64") - d.y_true.to_numpy("float64")
        d["abs_obs_err"] = np.abs(d.obs_err.to_numpy("float64"))
        d["is_lfpg"] = d.airport.to_numpy().astype(str) == HELD
        d["fold"] = f
        return d
    FR = {f: frame(f) for f in A.FOLDS}
    fr = pd.concat(FR.values(), ignore_index=True)
    lf, ot = fr[fr.is_lfpg], fr[~fr.is_lfpg]
    out["observation_residual"] = {
        "definition": "obs_taxi - y_true, in seconds; positive = the clock reads longer than the label",
        "lfpg": {"rows": int(len(lf)), "rms_s": rms(lf.obs_err.to_numpy("float64")),
                 "median_s": float(np.nanmedian(lf.obs_err)) if len(lf) else None,
                 "p90_abs_s": float(np.nanpercentile(lf.abs_obs_err, 90)) if len(lf) else None},
        "other_airports": {"rows": int(len(ot)), "rms_s": rms(ot.obs_err.to_numpy("float64")),
                           "median_s": float(np.nanmedian(ot.obs_err)),
                           "p90_abs_s": float(np.nanpercentile(ot.abs_obs_err, 90))},
        "by_fold_lfpg": {A.FOLD_MONTH[f]: rms(lf[lf.fold == f].obs_err.to_numpy("float64"))
                         for f in A.FOLDS}}

    rel = {}
    for q in list(QUALITY) + BESIDE:
        if q not in fr.columns:
            rel[q] = {"absent": True}
            continue
        rec = {"in_the_gate_family": q in QUALITY,
               "spearman_with_abs_residual": {
                   "lfpg": spearman(lf[q].to_numpy("float64"), lf.abs_obs_err.to_numpy("float64")),
                   "other_airports": spearman(ot[q].to_numpy("float64"), ot.abs_obs_err.to_numpy("float64"))},
               "coverage_lfpg": round(float(np.isfinite(lf[q].to_numpy("float64")).mean()), 4)
                                if len(lf) else None,
               "coverage_other": round(float(np.isfinite(ot[q].to_numpy("float64")).mean()), 4)}
        # the quintile table, cut on the OTHER airports and read on both
        v = ot[q].to_numpy("float64")
        fin = np.isfinite(v)
        if fin.sum() >= 500:
            edges = np.unique(np.nanpercentile(v[fin], [0, 20, 40, 60, 80, 100]))
            if len(edges) >= 3:
                rec["quintiles_cut_on_other_airports"] = []
                for i in range(len(edges) - 1):
                    lo, hi = edges[i], edges[i + 1]
                    sel = lambda d: (d[q].to_numpy("float64") >= lo) & (
                        d[q].to_numpy("float64") <= hi if i == len(edges) - 2
                        else d[q].to_numpy("float64") < hi)
                    so, sl = sel(ot), (sel(lf) if len(lf) else np.zeros(0, bool))
                    rec["quintiles_cut_on_other_airports"].append({
                        "lo": float(lo), "hi": float(hi),
                        "other_rows": int(so.sum()),
                        "other_rms_residual_s": rms(ot.obs_err.to_numpy("float64")[so]),
                        "lfpg_rows": int(sl.sum()) if len(lf) else 0,
                        "lfpg_rms_residual_s": (rms(lf.obs_err.to_numpy("float64")[sl])
                                                if len(lf) and sl.sum() else None)})
        rel[q] = rec
    out["quality_against_the_residual"] = rel
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("part A/B written", flush=True)

    # ------------------------------------------------------- part C: the gate
    # A gate KEEPS a row.  Direction is fixed by the sign of the relation measured on the
    # OTHER airports, never on LFPG: if higher q means a worse observation there, the gate
    # keeps the LOW side, and the cut is that side's quantile among the other airports.
    fold_use = {f: [g for g in A.FOLDS if g != f] for f in A.FOLDS}

    def other_rows(folds: list[str]) -> pd.DataFrame:
        d = pd.concat([FR[g] for g in folds], ignore_index=True)
        return d[~d.is_lfpg]

    def gain_per_row(d: pd.DataFrame) -> float:
        """mean squared-error gain per row: 2*r*d - d^2, the same v as tail_clauses."""
        if not len(d):
            return 0.0
        r, dv = d.rbase.to_numpy("float64"), d.upd.to_numpy("float64")
        return float(np.mean(2 * r * dv - dv * dv))

    # the selection table, built ENTIRELY on the other airports
    sel_rows = other_rows(list(A.FOLDS))
    assert not sel_rows.is_lfpg.any(), "the selection saw a held-airport row"

    cand = []
    for q in QUALITY:
        v = sel_rows[q].to_numpy("float64")
        if not np.isfinite(v).any():
            continue
        sp = spearman(v, sel_rows.abs_obs_err.to_numpy("float64"))
        if sp is None or abs(sp) < 0.02:
            cand.append({"field": q, "skipped": "no usable relation on the other airports",
                         "spearman": sp})
            continue
        keep_low = sp > 0          # higher q -> worse observation -> keep the low side
        for qq in QUANTILES:
            cut = float(np.nanpercentile(v[np.isfinite(v)], qq if keep_low else 100 - qq))
            k = (v <= cut) if keep_low else (v >= cut)
            k &= np.isfinite(v)
            if k.sum() < 200 or (~k).sum() < 200:
                continue
            cand.append({
                "field": q, "keeps": "low" if keep_low else "high",
                "quantile": qq, "cut": cut, "spearman_on_other": sp,
                "other_yield": round(float(k.mean()), 4),
                "other_rms_residual_kept_s": rms(sel_rows.obs_err.to_numpy("float64")[k]),
                "other_rms_residual_dropped_s": rms(sel_rows.obs_err.to_numpy("float64")[~k]),
                "other_mse_gain_per_kept_row_s2": gain_per_row(sel_rows[k]),
                "other_mse_gain_per_dropped_row_s2": gain_per_row(sel_rows[~k]),
                "rows_kept": int(k.sum())})
    # The criterion, pre-declared: a gate is worth having only if DROPPING the rows it
    # rejects raises the total gain -- that is, only if the rejected rows carry NEGATIVE
    # gain.  Score every candidate by the gain per row of the whole other-airport
    # population once the rejected rows are returned to the base, and put "keep
    # everything" in the same table so the answer is allowed to be "no gate pays".
    keep_all = gain_per_row(sel_rows)
    for c in cand:
        if "quantile" not in c:
            continue
        c["other_total_gain_per_population_row_s2"] = (
            c["other_yield"] * c["other_mse_gain_per_kept_row_s2"])
        c["beats_keeping_everything"] = bool(
            c["other_total_gain_per_population_row_s2"] > keep_all)
    out["keep_everything_on_the_other_airports_s2_per_row"] = keep_all
    out["gate_selection_on_the_other_airports"] = cand
    scored = [c for c in cand if "quantile" in c]
    scored.sort(key=lambda c: -c["other_total_gain_per_population_row_s2"])
    out["gate_selection_ranked"] = scored[:12]
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print(f"part C: {len(scored)} candidate gates scored; keep-everything "
          f"{keep_all:+.1f} s2/row", flush=True)
    for c in scored[:6]:
        print(f"   {c['field']:18s} keep {c['keeps']:4s} q{c['quantile']:<3d} "
              f"yield {c['other_yield']:.3f} kept {c['other_mse_gain_per_kept_row_s2']:+9.1f} "
              f"dropped {c['other_mse_gain_per_dropped_row_s2']:+9.1f} "
              f"total {c['other_total_gain_per_population_row_s2']:+9.1f} "
              f"beats_all {c['beats_keeping_everything']}", flush=True)

    if not scored or not scored[0]["beats_keeping_everything"]:
        out["gate_chosen"] = None
        out["reading"] = ("No gate in the pre-declared family beats keeping every row on "
                          "the other airports: the rejected rows do not carry negative "
                          "gain, so the quality fields do not separate a broken "
                          "observation from a usable one on the axis the threshold is "
                          "allowed to see.")
        print("NO GATE PAYS on the other airports", flush=True)
    else:
        out["gate_chosen"] = {k: v for k, v in scored[0].items()}
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    # ------------------------------------- part D: apply at LFPG and price it
    def cut_for(field: str, keeps: str, qq: int, folds: list[str]) -> float:
        d = other_rows(folds)
        v = d[field].to_numpy("float64")
        v = v[np.isfinite(v)]
        return float(np.nanpercentile(v, qq if keeps == "low" else 100 - qq))

    def keep_mask(p: pd.DataFrame, field: str, keeps: str, cut: float) -> np.ndarray:
        v = p[field].to_numpy("float64")
        k = (v <= cut) if keeps == "low" else (v >= cut)
        return k & np.isfinite(v)

    gate = out.get("gate_chosen")
    price_rules = {}
    def price(mask_of) -> dict:
        kept, zeroed = {}, {}
        for f in A.FOLDS:
            d = full[f].copy()
            m = mask_of(f)
            zeroed[f] = int((d[m] != 0).sum())
            d[m] = 0.0
            kept[f] = d
        rec = BA.price(pan, kept, rng)
        return {"board_all_row": rec["board_all_row"], "board_clean": rec["board_clean"],
                "every_fold_interval_clear": rec["every_fold_interval_clear"],
                "every_moving_fold_interval_clear": rec["every_moving_fold_interval_clear"],
                "folds_with_no_moved_row": rec["folds_with_no_moved_row"],
                "by_fold_all_row": {f: rec["by_fold"][f]["all_row"] for f in A.FOLDS},
                "rows_returned_to_the_base_by_fold": zeroed,
                "tail_clauses": {f: A.tail_clauses(pan[f], kept[f])
                                 ["registered_clause_y_over_3600s"] for f in A.FOLDS}}

    def is_lfpg(f):
        return pan[f].airport.to_numpy().astype(str) == HELD

    gated = {}
    if gate:
        for f in A.FOLDS:
            c = cut_for(gate["field"], gate["keeps"], gate["quantile"], fold_use[f])
            gated[f] = keep_mask(pan[f], gate["field"], gate["keeps"], c) \
                       & pan[f].usable.to_numpy(bool) & is_lfpg(f)
        out["gate_cut_by_fold"] = {A.FOLD_MONTH[f]: cut_for(
            gate["field"], gate["keeps"], gate["quantile"], fold_use[f]) for f in A.FOLDS}
        out["gate_yield_on_lfpg_folds"] = {
            A.FOLD_MONTH[f]: {"lfpg_with_clock": int((pan[f].usable.to_numpy(bool)
                                                      & is_lfpg(f)).sum()),
                              "released": int(gated[f].sum())} for f in A.FOLDS}
    else:
        for f in A.FOLDS:
            gated[f] = np.zeros(len(pan[f]), bool)

    rules = {"none": lambda f: np.zeros(len(pan[f]), bool),
             "hold_lfpg_as_it_ships": is_lfpg,
             "hold_lfpg_except_the_gated": lambda f: is_lfpg(f) & ~gated[f],
             "release_only_the_REJECTED_control": lambda f: is_lfpg(f) & gated[f]}
    for name, fn in rules.items():
        rec = price(fn)
        price_rules[name] = rec
        ci = rec["board_all_row"]["interval"]["ci95"]
        t = {f: round(rec["tail_clauses"][f]["rmse_gain_s"], 3) for f in A.FOLDS}
        print(f"{name:34s} all-row {rec['board_all_row']['gain_s']:+8.4f} "
              f"[{ci[0]:+7.3f},{ci[1]:+7.3f}] clean {rec['board_clean']['gain_s']:+8.4f} "
              f"held {rec['rows_returned_to_the_base_by_fold']} tail {t}", flush=True)
        out["priced_rules"] = price_rules
        OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    # the INCREMENT over the hold: the two rules agree on every row that is not a
    # released LFPG row, so the increment is exactly those rows' own update priced
    # against the same base.  The dark fold contributes a structural zero, not a failure.
    inc = price(lambda f: ~gated[f])
    out["increment_over_the_hold"] = inc
    ci = inc["board_all_row"]["interval"]["ci95"]
    print(f"{'INCREMENT over the hold':34s} all-row {inc['board_all_row']['gain_s']:+8.4f} "
          f"[{ci[0]:+7.3f},{ci[1]:+7.3f}] moving-fold clear "
          f"{inc['every_moving_fold_interval_clear']} dark {inc['folds_with_no_moved_row']}",
          flush=True)

    # ---------------------------------------------------- part E: the face
    lit = out["lit_folds"]
    rel = pd.concat([FR[f][FR[f].is_lfpg & gated[f][pan[f].usable.to_numpy(bool)]]
                     for f in A.FOLDS], ignore_index=True) if gate else pd.DataFrame()
    g_measured = gain_per_row(rel) if len(rel) else 0.0
    # the ranking side: the same gate, cut on all three folds' other airports
    n_rel_rank, by_month = 0, {}
    if gate:
        c = cut_for(gate["field"], gate["keeps"], gate["quantile"], list(A.FOLDS))
        kk = keep_mask(dep, gate["field"], gate["keeps"], c) & served & (rap == HELD)
        n_rel_rank = int(kk.sum())
        by_month = {m: int((kk & (rmo == m)).sum()) for m in sorted(set(rmo))}
        out["gate_cut_for_the_ranking_months"] = c

    def face(n: int, g: float) -> dict:
        gg = g * n / N_RANKING
        lvl = LEVEL - (LEVEL ** 2 - gg) ** 0.5 if LEVEL ** 2 > gg else None
        return {"rows": n, "s2_per_released_row": g,
                "board_s_at_x1_2_the_level_currency": lvl,
                "board_s_at_x1_5": lvl * 1.492 if lvl is not None else None}
    out["face"] = {
        "level_s": LEVEL, "level_is": "v17's board reading; nothing is fitted to it (D11)",
        "measured_on": [A.FOLD_MONTH[f] for f in lit],
        "released_fold_rows": int(len(rel)),
        "released_ranking_rows": n_rel_rank, "released_ranking_by_month": by_month,
        "at_the_measured_per_row_gain": face(n_rel_rank, g_measured),
        "note": ("g is MEASURED on the released LFPG rows of the lit folds, not borrowed "
                 "from the served average, so the face is this subset's own worth")}
    out["yield"] = {
        "lfpg_ranking_rows_with_a_clock": int((served & (rap == HELD)).sum()),
        "released_by_the_gate": n_rel_rank,
        "share": (round(n_rel_rank / max(int((served & (rap == HELD)).sum()), 1), 4))}
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    # ------------------------------------------------------------------------
    # part F: the desk's OWN criterion, run separately -- a gate that separates a
    # usable observation from a broken one is a gate that lowers the RESIDUAL of the
    # observation, whether or not the rows it drops were harmful.  Selected on the other
    # airports exactly as above, then applied at LFPG at a LADDER of cuts, so no single
    # threshold is being sold and the whole trade-off is visible.
    best = None
    for c in scored:
        if c["other_yield"] < 0.5:
            continue
        red = (out["observation_residual"]["other_airports"]["rms_s"]
               - c["other_rms_residual_kept_s"])
        if best is None or red > best[0]:
            best = (red, c)
    out["gate_by_residual_separation"] = None if best is None else {
        "field": best[1]["field"], "keeps": best[1]["keeps"],
        "chosen_by": ("largest fall in RMS(obs_taxi - y) among KEPT rows on the other "
                      "airports, at a yield of at least 0.5"),
        "other_rms_residual_all_s": out["observation_residual"]["other_airports"]["rms_s"],
        "other_rms_residual_kept_s": best[1]["other_rms_residual_kept_s"],
        "fall_s": best[0], "at_quantile": best[1]["quantile"]}

    ladder = []
    if best is not None:
        field, keeps = best[1]["field"], best[1]["keeps"]
        for qq in [0] + QUANTILES:
            g = {}
            for f in A.FOLDS:
                if qq == 0:                      # the ceiling: release every LFPG row
                    g[f] = pan[f].usable.to_numpy(bool) & is_lfpg(f)
                else:
                    cc = cut_for(field, keeps, qq, fold_use[f])
                    g[f] = (keep_mask(pan[f], field, keeps, cc)
                            & pan[f].usable.to_numpy(bool) & is_lfpg(f))
            inc = price(lambda f: ~g[f])
            rel = pd.concat([FR[f][FR[f].is_lfpg
                                   & g[f][pan[f].usable.to_numpy(bool)]]
                             for f in A.FOLDS], ignore_index=True)
            gm = gain_per_row(rel)
            if qq == 0:
                cc_rank, kk = None, served & (rap == HELD)
            else:
                cc_rank = cut_for(field, keeps, qq, list(A.FOLDS))
                kk = keep_mask(dep, field, keeps, cc_rank) & served & (rap == HELD)
            n_rank = int(kk.sum())
            whole = price(lambda f: is_lfpg(f) & ~g[f])
            row = {"quantile": qq, "cut_for_the_ranking_months": cc_rank,
                   "released_fold_rows": int(len(rel)),
                   "released_fold_rows_by_fold": {
                       A.FOLD_MONTH[f]: int(g[f].sum()) for f in A.FOLDS},
                   "released_ranking_rows": n_rank,
                   "released_ranking_by_month": {m: int((kk & (rmo == m)).sum())
                                                 for m in sorted(set(rmo))},
                   "lfpg_rms_residual_kept_s": rms(rel.obs_err.to_numpy("float64"))
                                               if len(rel) else None,
                   "lfpg_rms_residual_all_s": out["observation_residual"]["lfpg"]["rms_s"],
                   "mse_gain_per_released_row_s2": gm,
                   "increment_all_row_s": inc["board_all_row"]["gain_s"],
                   "increment_ci95": inc["board_all_row"]["interval"]["ci95"],
                   "increment_moving_fold_clear": inc["every_moving_fold_interval_clear"],
                   "increment_dark_folds": inc["folds_with_no_moved_row"],
                   # THE LIT FOLDS, per fold (desk ruling, 2026-09-09): LFPG's january is
                   # an EMPTY LIT FOLD -- dark by instrument, zero with-clock rows -- and
                   # the registered lit rule says an empty lit fold is not a crossing, so
                   # this release is gated on july_loo and december, december being the
                   # winter fold a winter ranking month is priced on.
                   "increment_by_fold": inc["by_fold_all_row"],
                   "increment_lit_folds_clear": {
                       f: bool(inc["by_fold_all_row"][f]["ci95"][0] > 0)
                       for f in A.FOLDS if f not in inc["folds_with_no_moved_row"]},
                   "whole_update_all_row_s": whole["board_all_row"]["gain_s"],
                   "whole_update_every_fold_clear": whole["every_fold_interval_clear"],
                   "whole_update_tail": {f: whole["tail_clauses"][f]["rmse_gain_s"]
                                         for f in A.FOLDS},
                   "face": face(n_rank, gm)}
            ladder.append(row)
            print(f"  q{qq:<3d} released fold {row['released_fold_rows']:>5} rank "
                  f"{n_rank:>5}  resid {str(round(row['lfpg_rms_residual_kept_s'],1)) if row['lfpg_rms_residual_kept_s'] else '--':>7}"
                  f"  s2/row {gm:>+9.1f}  incr {row['increment_all_row_s']:+.4f} "
                  f"[{row['increment_ci95'][0]:+.4f},{row['increment_ci95'][1]:+.4f}] "
                  f"clear {row['increment_moving_fold_clear']}  face "
                  f"{row['face']['board_s_at_x1_2_the_level_currency']:+.4f}", flush=True)
            out["ladder_on_the_residual_gate"] = ladder
            OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    print(json.dumps({"yield": out["yield"], "face": out["face"]}, indent=1, default=float))
    print("written to", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
