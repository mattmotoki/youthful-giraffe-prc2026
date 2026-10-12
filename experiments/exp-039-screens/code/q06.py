"""Q06 -- the census of recoverable missing joins, before any rematching.

Four questions the queue insists are kept apart, and they are reported apart:

  A  CENSUS      how many of the 388 noac and 4,895 nonm RANKING departures have
                 0, 1 or more than one permitted NM-carrying donor, before and
                 after the declared clock windows.
  B  ACCURACY    on JOINED departures with their own NM exposure removed and the
                 gold `FLIGHT_ID_mvt` retained only for scoring: counterpart
                 existence, candidate recall and accepted-ID agreement, each a
                 different denominator.
  C  STRESS      the same test with EVERY donor carrying the gold id removed, so
                 the only thing the rule can do is accept another occurrence.
                 Wrong acceptance is priced in SQUARED ERROR on labelled 2025
                 rows, not as a match-accuracy percentage.
  D  BENEFIT     the donor-anchor prediction read on naturally unmatched,
                 type-present rows (`nonm`) of the three fold months, against
                 v10 on the shared contract: r = y - p10, d = p_donor - p10,
                 A = mean(r d), B = mean(d^2), gain at full amplitude 2A - B.

The 388 noac rows stay census-only, as the queue says.  Nothing here refits a
member or writes a candidate.

    python q06.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import v10 as V  # noqa: E402
import rowfeat as R  # noqa: E402
import donor as Dn  # noqa: E402
import packet as PK  # noqa: E402

ROOT = V.ROOT
FOLDS = V.FOLDS
D = V.D

SEED = 39
N_JOINED_SAMPLE = 20_000       # predeclared, fixed seed, for B and C
COLS = ["MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt",
        "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
        "AIRCRAFT_TYPE_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"] + Dn.NM_FIELDS


def counts(e: pd.DataFrame, q: pd.DataFrame, tag: str) -> dict:
    """The 0 / 1 / >1 census for one query population, before and after filters."""
    n = len(q)
    out = {"population": tag, "n_queries": n,
           "n_missing_key": int((q["key"].isna() | (q["key"] == "")).sum())}
    for stage, sub in (("key_leg_only", e), ("in_window", e[e["in_window"]]),
                       ("in_window_nm", e[e["in_window"] & e["has_nm"]])):
        g = sub.groupby("q_mid")["fid"].agg(["size", "nunique"])
        got = g.reindex(q["mid"]).fillna(0)
        out[stage] = {
            "donors_0": int((got["size"] == 0).sum()),
            "donors_1": int((got["size"] == 1).sum()),
            "donors_gt1": int((got["size"] > 1).sum()),
            "ids_1": int((got["nunique"] == 1).sum()),
            "ids_gt1": int((got["nunique"] > 1).sum())}
    for cls in ("arrival", "duplicate"):
        s = e[(e["cls"] == cls) & e["in_window"] & e["has_nm"]]
        out[f"in_window_nm_{cls}"] = int(s["q_mid"].nunique())
    out["clock_conflicts"] = int(e.loc[~e["in_window"], "q_mid"].nunique())
    out["boundary_cases"] = int(e.loc[e["boundary"], "q_mid"].nunique())
    return out


def census(rk: pd.DataFrame) -> dict:
    tbl = Dn.table(rk)
    dep = tbl[tbl["phase"] == "DEP"]
    raw_dep = rk[rk["PHASE_mvt"] == "DEP"]
    noac = dep[raw_dep["AIRCRAFT_TYPE_mvt"].isna().to_numpy()]
    nonm = dep[(~np.isfinite(dep["fid"].to_numpy()))
               & raw_dep["AIRCRAFT_TYPE_mvt"].notna().to_numpy()]
    out = {}
    for tag, q in (("noac", noac), ("nonm", nonm)):
        e = Dn.candidates(q, tbl)
        c = counts(e, q, tag)
        acc = Dn.accept(e)
        c["accepted"] = int(len(acc))
        c["accepted_pct"] = round(100 * len(acc) / max(len(q), 1), 2)
        c["accepted_with_aobt"] = int(acc["AOBT_3_flt"].notna().sum()) if len(acc) else 0
        c["accepted_by_class"] = (acc["cls"].value_counts().to_dict() if len(acc)
                                  else {})
        ten = set(dep["adep"].unique())
        c["ades_in_ten_airports"] = int(q["ades"].isin(ten).sum())
        c["ades_in_ten_airports_pct"] = round(
            100 * float(q["ades"].isin(ten).mean()), 2)
        c["accepted_of_ades_in_ten_pct"] = round(
            100 * len(acc) / max(int(q["ades"].isin(ten).sum()), 1), 2)
        out[tag] = c
    return out


def gold_removed(tbl: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """Every row carrying a query's own gold `FLIGHT_ID_mvt`, dropped."""
    gold = set(q["fid"].dropna().to_numpy())
    return tbl[~tbl["fid"].isin(gold)]


def accuracy(rk: pd.DataFrame, rng) -> dict:
    tbl = Dn.table(rk)
    dep = tbl[tbl["phase"] == "DEP"]
    joined = dep[np.isfinite(dep["fid"].to_numpy())]
    # A DEDICATED generator, so that the predeclared fixed-seed sample does not
    # move when an unrelated bootstrap draws from `rng` earlier in the run.
    idx = np.random.default_rng(SEED).choice(
        len(joined), size=min(N_JOINED_SAMPLE, len(joined)), replace=False)
    q = joined.iloc[np.sort(idx)].copy()
    gold = q.set_index("mid")["fid"]

    e = Dn.candidates(q, tbl)
    acc = Dn.accept(e)
    # COUNTERPART EXISTENCE is a property of the table, not of the search: does
    # any OTHER row anywhere carry this query's gold `FLIGHT_ID_mvt`?  Sol round
    # 1, blocking finding 10: the first version computed it after the key and leg
    # filters, which made it a property of the candidate generator and put
    # recall on the wrong denominator.
    rows_per_fid = tbl.groupby("fid")["mid"].nunique()
    exist_mask = rows_per_fid.reindex(q["fid"]).fillna(0).to_numpy() > 1
    exist = int(exist_mask.sum())
    e["is_gold"] = e["fid"].to_numpy() == gold.reindex(e["q_mid"]).to_numpy()
    recall = e.loc[e["is_gold"] & e["in_window"], "q_mid"].nunique()
    in_cand = e.loc[e["is_gold"], "q_mid"].nunique()
    agree = int((acc["fid"].to_numpy()
                 == gold.reindex(acc["q_mid"]).to_numpy()).sum()) if len(acc) else 0
    ten = set(dep["adep"].unique())
    out = {"n_queries": int(len(q)), "sample_seed": SEED,
           "ades_in_ten_airports": int(q["ades"].isin(ten).sum()),
           "ades_in_ten_airports_pct": round(100 * float(q["ades"].isin(ten).mean()), 2),
           "same_leg_row_exists": int(e["q_mid"].nunique()),
           "same_leg_row_in_window": int(e.loc[e["in_window"], "q_mid"].nunique()),
           "counterpart_exists": exist,
           "counterpart_exists_pct": round(100 * exist / len(q), 2),
           "gold_row_survives_key_and_leg": int(in_cand),
           "candidate_recall": int(recall),
           "candidate_recall_pct_of_existing": round(100 * recall / max(exist, 1), 2),
           "candidate_recall_pct_of_all": round(100 * recall / len(q), 2),
           "accepted": int(len(acc)),
           "accepted_pct": round(100 * len(acc) / len(q), 2),
           "accepted_id_agrees": agree,
           "accepted_id_agreement_pct": round(100 * agree / max(len(acc), 1), 2),
           "accepted_unique_ids": int(acc["fid"].nunique()) if len(acc) else 0}

    # C: every donor with the gold id removed
    t2 = gold_removed(tbl, q)
    e2 = Dn.candidates(q, t2)
    acc2 = Dn.accept(e2)
    out["stress"] = {
        "note": "every row carrying a query's own gold FLIGHT_ID_mvt removed",
        "rows_removed": int(len(tbl) - len(t2)),
        "accepted": int(len(acc2)),
        "wrong_acceptance_pct": round(100 * len(acc2) / len(q), 2)}
    return out, q, acc, acc2


def fold_stage(fold: str, panel: pd.DataFrame, rng, preds_out: dict) -> dict:
    """D on the naturally unmatched rows, and C priced in squared error."""
    raw = D.load_month(R.month_of(fold))
    masked, _ = D.mask_like_ranking(raw)
    del raw
    tbl = Dn.table(masked)
    del masked
    dep = tbl[tbl["phase"] == "DEP"]
    p = panel.set_index("MVT_ID_mvt")
    t_of = dep.set_index("mid")["t"]

    out = {"fold": fold, "month": R.month_of(fold)}
    # ---- D: naturally unmatched, type-present rows -------------------------
    nonm_ids = panel.loc[panel["regime"] == "nonm", "MVT_ID_mvt"].to_numpy()
    q = dep[dep["mid"].isin(nonm_ids)]
    e = Dn.candidates(q, tbl)
    acc = Dn.accept(e)
    ten = set(dep["adep"].unique())
    rec = {"n_nonm": int(len(q)), "accepted": int(len(acc)),
           "ades_in_ten_airports": int(q["ades"].isin(ten).sum())}
    if len(acc):
        a = acc[acc["AOBT_3_flt"].notna()]
        rec["accepted_with_aobt"] = int(len(a))
        if len(a):
            mid = a["q_mid"].to_numpy()
            pdon = t_of.reindex(mid).to_numpy() - a["AOBT_3_flt"].to_numpy()
            r = p["r"].reindex(mid).to_numpy()
            p10 = p["p10"].reindex(mid).to_numpy()
            y = p["y_true"].reindex(mid).to_numpy()
            cl = p["clean"].reindex(mid).to_numpy().astype(bool)
            d = pdon - p10
            A = float((r * d).mean()); B = float((d ** 2).mean())
            rec.update({
                "A_mean_r_d": round(A, 1), "B_mean_d2": round(B, 1),
                "full_amplitude_mse_gain": round(2 * A - B, 1),
                "unrestricted_scale": round(A / B, 4) if B > 0 else None,
                "best_scalar_mse_gain": round(A * A / B, 1) if B > 0 else None,
                "rmse_v10_on_these": round(float(np.sqrt((r ** 2).mean())), 1),
                "rmse_donor_anchor": round(float(np.sqrt(((y - pdon) ** 2).mean())), 1),
                # Sol round 1, blocking finding 12: this is a CLEAN share and
                # must use the clean denominator and numerator.
                "sse_share_of_fold_clean_pct": round(
                    100 * float((r[cl] ** 2).sum())
                    / float((panel.loc[panel["clean"], "r"] ** 2).sum()), 3),
                "sse_share_of_fold_all_row_pct": round(
                    100 * float((r ** 2).sum())
                    / float((panel["r"] ** 2).sum()), 3)})
            # Sol round 2, blocking finding e1: the same 1/2/3-day sensitivity
            # every other screen reports.
            rec["multiday"] = PK.boot_blocks(
                2 * r * d - d ** 2, V.day(t_of.reindex(mid).to_numpy()), rng)
            ed, ev = np.abs(y - pdon), np.abs(r)
            rec["abs_err_donor_pctiles"] = [round(float(v), 1) for v in
                                            np.percentile(ed, [50, 75, 90, 99])]
            rec["abs_err_v10_pctiles"] = [round(float(v), 1) for v in
                                          np.percentile(ev, [50, 75, 90, 99])]
            rec["donor_closer_pct"] = round(100 * float((ed < ev).mean()), 1)
            # the bounded repair: a donor anchor clipped to a plausible taxi time
            pc = np.clip(pdon, 60.0, 3600.0)
            dc = pc - p10
            Ac = float((r * dc).mean()); Bc = float((dc ** 2).mean())
            rec["clipped"] = {
                "A_mean_r_d": round(Ac, 1), "B_mean_d2": round(Bc, 1),
                "full_amplitude_mse_gain": round(2 * Ac - Bc, 1),
                "unrestricted_scale": round(Ac / Bc, 4) if Bc > 0 else None,
                "best_scalar_mse_gain": round(Ac * Ac / Bc, 1) if Bc > 0 else None,
                "rmse_clipped_anchor": round(
                    float(np.sqrt(((y - pc) ** 2).mean())), 1)}
    out["donor_anchor_nonm"] = rec

    # ---- C priced: masked-joined rows with the gold id removed -------------
    joined = dep[np.isfinite(dep["fid"].to_numpy())]
    # A DEDICATED generator, so that the predeclared fixed-seed sample does not
    # move when an unrelated bootstrap draws from `rng` earlier in the run.
    idx = np.random.default_rng(SEED).choice(
        len(joined), size=min(N_JOINED_SAMPLE, len(joined)), replace=False)
    qj = joined.iloc[np.sort(idx)].copy()
    t2 = gold_removed(tbl, qj)
    acc2 = Dn.accept(Dn.candidates(qj, t2))
    price = {"n_queries": int(len(qj)), "accepted": int(len(acc2))}
    if len(acc2):
        a = acc2[acc2["AOBT_3_flt"].notna()]
        price["accepted_with_aobt"] = int(len(a))
        if len(a):
            mid = a["q_mid"].to_numpy()
            pdon = t_of.reindex(mid).to_numpy() - a["AOBT_3_flt"].to_numpy()
            y = p["y_true"].reindex(mid).to_numpy()
            p10 = p["p10"].reindex(mid).to_numpy()
            ok = np.isfinite(y)
            price["rmse_v10_on_wrongly_accepted"] = round(
                float(np.sqrt(((y[ok] - p10[ok]) ** 2).mean())), 1)
            price["rmse_wrong_donor_anchor"] = round(
                float(np.sqrt(((y[ok] - pdon[ok]) ** 2).mean())), 1)
            price["mse_cost_s2_per_row"] = round(
                float(((y[ok] - pdon[ok]) ** 2 - (y[ok] - p10[ok]) ** 2).mean()), 1)
    out["gold_removed_priced"] = price
    # The donor-anchor arm as a full prediction vector, so the screen reports the
    # queue's primary endpoint like every other one (Sol round 1, finding 2).
    pv = panel["p10"].to_numpy("float64").copy()
    if len(acc):
        a2 = acc[acc["AOBT_3_flt"].notna()]
        if len(a2):
            pos = pd.Index(panel["MVT_ID_mvt"].to_numpy("float64")).get_indexer(
                a2["q_mid"].to_numpy())
            ok = pos >= 0
            pv[pos[ok]] = (t_of.reindex(a2["q_mid"].to_numpy()).to_numpy()
                           - a2["AOBT_3_flt"].to_numpy())[ok]
    preds_out[fold] = pv
    PK.dates("q06", "donor_anchor", fold, V.day(panel["t"].to_numpy("float64")),
             panel["y_true"].to_numpy("float64"),
             panel["p10"].to_numpy("float64"), pv)
    return out


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rk = D.load_ranking(columns=COLS)
    out = {"screen": "Q06",
           "predeclared": {"arrival_window_s": [0, Dn.ARR_MAX],
                           "duplicate_tolerance_s": Dn.DUP_TOL,
                           "normalisation": "uppercase and whitespace only",
                           "sample": N_JOINED_SAMPLE, "seed": SEED}}
    out["census_ranking"] = census(rk)
    for tag, c in out["census_ranking"].items():
        print(f"[census {tag}] {c['n_queries']:,} rows; key missing "
              f"{c['n_missing_key']}; in-window NM donors: 0 "
              f"{c['in_window_nm']['donors_0']:,} / 1 {c['in_window_nm']['donors_1']:,}"
              f" / >1 {c['in_window_nm']['donors_gt1']:,}; ACCEPTED "
              f"{c['accepted']:,} ({c['accepted_pct']} %), with AOBT "
              f"{c['accepted_with_aobt']:,}", flush=True)
    acc_out, _, _, _ = accuracy(rk, rng)
    out["accuracy_masked_joined"] = acc_out
    print(f"[accuracy] n={acc_out['n_queries']:,}: counterpart exists "
          f"{acc_out['counterpart_exists']:,} ({acc_out['counterpart_exists_pct']} %), "
          f"candidate recall {acc_out['candidate_recall']:,} of those "
          f"({acc_out['candidate_recall_pct_of_existing']} %), accepted "
          f"{acc_out['accepted']:,} ({acc_out['accepted_pct']} %), accepted-id "
          f"agreement {acc_out['accepted_id_agreement_pct']} % on "
          f"{acc_out['accepted_unique_ids']:,} distinct ids; gold-removed wrong "
          f"acceptance {acc_out['stress']['wrong_acceptance_pct']} %", flush=True)
    del rk

    pan = V.load()
    out["folds"], preds_out = {}, {}
    for f in FOLDS:
        out["folds"][f] = fold_stage(f, pan[f], rng, preds_out)
        d = out["folds"][f]["donor_anchor_nonm"]
        print(f"[{f}] nonm {d['n_nonm']:,}; accepted {d['accepted']:,}"
              + (f", with AOBT {d.get('accepted_with_aobt', 0):,}; 2A-B "
                 f"{d.get('full_amplitude_mse_gain')} s^2, scale "
                 f"{d.get('unrestricted_scale')}, v10 RMSE "
                 f"{d.get('rmse_v10_on_these')} vs donor anchor "
                 f"{d.get('rmse_donor_anchor')}" if d.get("accepted_with_aobt")
                 else ""), flush=True)
        g = out["folds"][f]["gold_removed_priced"]
        print(f"      gold-removed: accepted {g['accepted']:,} of "
              f"{g['n_queries']:,}"
              + (f"; wrong-anchor RMSE {g.get('rmse_wrong_donor_anchor')} against "
                 f"v10 {g.get('rmse_v10_on_wrongly_accepted')} "
                 f"({g.get('mse_cost_s2_per_row')} s^2/row)"
                 if g.get("accepted_with_aobt") else ""), flush=True)
    out["primary"] = PK.primary(pan, preds_out,
                                "headline arm: the donor anchor substituted on "
                                "every accepted naturally unmatched nonm row")
    out["wall_s"] = round(time.time() - t0, 1)
    p = EXP / "analysis" / "q06.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    PK.write("q06", out, pan)
    pr = out["primary"]
    print(f"\n  PRIMARY (all-row january/july board): {pr['board_all_row']['v10']:.4f}"
          f" -> {pr['board_all_row']['arm']:.4f} ({pr['board_all_row']['gain']:+.4f} s)"
          f"; clean {pr['board_clean']['v10']:.4f} -> {pr['board_clean']['arm']:.4f} "
          f"({pr['board_clean']['gain']:+.4f} s)")
    print(f"\n  written: {p}  ({out['wall_s']} s)")


if __name__ == "__main__":
    main()
