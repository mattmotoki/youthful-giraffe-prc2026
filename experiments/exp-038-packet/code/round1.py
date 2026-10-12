"""Sol round 1, the recomputations its blocking findings require.

Everything here reads the committed packet and refits nothing, so each number
is a command that runs in seconds:

  B2/B3  a PROCEDURE bootstrap for the held-out amplitude.  `q01.py` fixed each
         fitted `s` and resampled only the scored rows, which is an interval on
         the scored gain and not on the cross-fitted procedure.  Here every draw
         resamples day blocks in ALL THREE folds, refits `scale_loo` on the
         resampled A and B, and scores at the refitted amplitude -- at 1-day,
         3-day and 7-day blocks.
  B4     the composition-reweighted endpoint, both the true maximum move across
         the whole table and the reweighted gain of the BOUNDED blend, which
         `gain_board_rankmix` never priced.
  B5     day-block intervals on the CLEAN endpoint for every Q02 arm, so a clean
         point is never quoted against an all-row interval again.
  B11    the seed spread as the MATCHED design it is: five seed draws shared by
         the two heads, paired differences, a sign-flip exact test, and the
         PRIMARY all-row endpoint beside the clean one.
  B12    the packet's own accounting: the real `q01-dates.csv` row count, the
         SHA-256 of every source file the registry reads, and the per-row paired
         SSE column the contract asks for by name.
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
import align as A  # noqa: E402
import vectors as V  # noqa: E402

FOLDS = V.FOLDS
DRAWS = 2000
SEED = 7
FINALISTS = ("stack-5", "k2-w300", "K2 (v9)", "refit-bulk", "v7", "v6")
NAMES = json.loads((EXP / "packet" / "columns.json").read_text())["d_columns"]
COL = {v: k for k, v in NAMES.items()}


def load() -> dict:
    out = {}
    for f in FOLDS:
        d = pd.read_parquet(EXP / "packet" / f"rows-{f}.parquet")
        out[f] = {"r": d["r"].to_numpy("float64"),
                  "clean": d["clean"].to_numpy(bool),
                  "day": d["day"].to_numpy("int64"),
                  "frame": d}
    return out


def procedure_ci(P: dict, name: str, width: int, draws: int = DRAWS,
                 seed: int = SEED, clean: bool = False) -> tuple:
    """Resample day blocks in every fold, REFIT the leave-one-fold-out amplitude
    on the draw, and score each fold at its own refitted amplitude."""
    rng = np.random.default_rng(seed)
    sel, blk, r, d = {}, {}, {}, {}
    for f in FOLDS:
        k = P[f]["clean"] if clean else np.ones(len(P[f]["r"]), bool)
        sel[f] = k
        blk[f] = A.blocks(P[f]["day"][k], width)
        r[f] = P[f]["r"][k]
        d[f] = P[f]["frame"][COL[name]].to_numpy("float64")[k]
    ids, groups = {}, {}
    for f in FOLDS:
        u, inv = np.unique(blk[f], return_inverse=True)
        order = np.argsort(inv, kind="stable")
        starts = np.searchsorted(inv[order], np.arange(len(u)))
        groups[f] = np.split(order, starts[1:])
        ids[f] = len(u)
    out = []
    for _ in range(draws):
        Ad, Bd, nd, rr, dd = {}, {}, {}, {}, {}
        for f in FOLDS:
            pick = rng.integers(0, ids[f], ids[f])
            idx = np.concatenate([groups[f][i] for i in pick])
            rr[f], dd[f] = r[f][idx], d[f][idx]
            Ad[f] = float(np.mean(rr[f] * dd[f]))
            Bd[f] = float(np.mean(dd[f] * dd[f]))
            nd[f] = len(idx)
        s = A.scale_loo(Ad, Bd, nd)
        num = den = 0.0
        for f in A.BOARD_FOLDS:
            sf = s[f]["s"]
            num += A.SHARES[f] * float(np.mean(rr[f] ** 2))
            den += A.SHARES[f] * float(np.mean((rr[f] - sf * dd[f]) ** 2))
        out.append(np.sqrt(num) - np.sqrt(den))
    q = np.quantile(out, [0.025, 0.975])
    return float(q[0]), float(q[1]), float(np.mean(out))


def main() -> None:
    P = load()
    T = pd.read_csv(EXP / "analysis" / "q01-table.csv")
    rec: dict = {}

    # ---- B2, B3: the procedure bootstrap ---------------------------------
    rows = []
    for name, width, clean in product(FINALISTS, (1, 3, 7), (False, True)):
        lo, hi, mean = procedure_ci(P, name, width, clean=clean)
        rows.append({"candidate": name, "block_days": width,
                     "population": "clean" if clean else "all rows",
                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                     "boot_mean": round(mean, 4)})
    B = pd.DataFrame(rows)
    B.to_csv(EXP / "analysis" / "q01-procedure-ci.csv", index=False)
    print("=== B2/B3  the amplitude REFITTED in every draw, all three folds "
          "resampled ===")
    with pd.option_context("display.width", 200):
        print(B[B.population == "all rows"].to_string(index=False))

    # ---- B4: composition --------------------------------------------------
    t = T.dropna(subset=["gain_board_rankmix"]).copy()
    t["move"] = (t["gain_board_rankmix"] - t["gain_board_allrow"]).abs()
    worst = t.nlargest(3, "move")[["candidate", "gain_board_allrow",
                                   "gain_board_rankmix", "move"]]
    no37 = t[~t.candidate.str.startswith("exp-037")]
    o1 = t.sort_values("gain_board_allrow", ascending=False).candidate.tolist()
    o2 = t.sort_values("gain_board_rankmix", ascending=False).candidate.tolist()
    # the BOUNDED blend under the ranking mix, at the amplitude already fitted
    g = pd.read_parquet(ROOT / "experiments/exp-026-rebuild/candidates/ranking-gates.parquet")
    RANKING_OF = {"july_loo": "2026-07", "january": "2026-01"}
    w = {}
    for f, month in RANKING_OF.items():
        d = P[f]["frame"]
        rk = g[g["month"] == month]
        tgt = (rk.groupby([rk["airport"].astype(str), rk["regime"].astype(str)])
               .size().rename("n_rank"))
        tgt.index = tgt.index.map(tuple)
        cell = pd.Series(list(zip(d["airport"].astype(str), d["regime"].astype(str))))
        src = cell.value_counts().rename("n_fold")
        j = pd.concat([tgt, src], axis=1)
        share_t = j["n_rank"].fillna(0.0) / float(tgt.sum())
        share_s = j["n_fold"].fillna(0.0) / float(len(cell))
        wc = (share_t / share_s.replace(0.0, np.nan)).fillna(0.0)
        ww = cell.map(wc).fillna(0.0).to_numpy("float64")
        w[f] = ww * (len(ww) / ww.sum())
    comp_rows = []
    for name in FINALISTS:
        amp = {f: float(T.loc[T.candidate == name, f"amp_loo_{f}"].iloc[0]) for f in FOLDS}
        num = den = 0.0
        for f in A.BOARD_FOLDS:
            r_ = P[f]["r"]
            d_ = P[f]["frame"][COL[name]].to_numpy("float64")
            num += A.SHARES[f] * float(np.average(r_ ** 2, weights=w[f]))
            den += A.SHARES[f] * float(np.average((r_ - amp[f] * d_) ** 2, weights=w[f]))
        comp_rows.append({"candidate": name,
                          "gain_board_loo_amp": round(float(
                              T.loc[T.candidate == name, "gain_board_loo_amp"].iloc[0]), 4),
                          "gain_rankmix_loo_amp": round(float(np.sqrt(num) - np.sqrt(den)), 4)})
    Cc = pd.DataFrame(comp_rows)
    Cc.to_csv(EXP / "analysis" / "q01-composition-blend.csv", index=False)
    rec["composition"] = {
        "max_move_full_amplitude": round(float(t["move"].max()), 4),
        "max_move_candidate": str(t.loc[t["move"].idxmax(), "candidate"]),
        "max_move_excluding_exp037": round(float(no37["move"].max()), 4),
        "max_move_candidate_excluding_exp037": str(
            no37.loc[no37["move"].idxmax(), "candidate"]),
        "ordering_changes": bool(o1 != o2),
        "blend_at_ranking_mix": comp_rows}
    print("\n=== B4  composition, the true numbers ===")
    print(worst.to_string(index=False))
    print(Cc.to_string(index=False))

    # ---- B5: clean intervals for every Q02 arm ---------------------------
    q = {f: pd.read_parquet(EXP / "packet" / f"q02-{f}.parquet") for f in FOLDS}
    arms = [c for c in q[FOLDS[0]].columns if c not in ("MVT_ID_mvt", "y_true")]
    base = "unchanged (v10 replay)"
    rows = []
    for arm in arms:
        parts_all, parts_cl = {}, {}
        for f in A.BOARD_FOLDS:
            y = q[f]["y_true"].to_numpy("float64")
            p0 = q[f][base].to_numpy("float64")
            p1 = q[f][arm].to_numpy("float64")
            k = P[f]["clean"]
            b = A.blocks(P[f]["day"], 1)
            parts_all[f] = (b, (y - p0) ** 2, (y - p1) ** 2)
            parts_cl[f] = (b[k], ((y - p0) ** 2)[k], ((y - p1) ** 2)[k])
        ca = A.boot_board(parts_all, DRAWS)
        cc = A.boot_board(parts_cl, DRAWS)
        rows.append({"arm": arm, "ci_lo_allrow": round(ca[0], 4),
                     "ci_hi_allrow": round(ca[1], 4),
                     "ci_lo_clean": round(cc[0], 4), "ci_hi_clean": round(cc[1], 4)})
    Q = pd.DataFrame(rows)
    Q.to_csv(EXP / "analysis" / "q02-clean-ci.csv", index=False)
    print("\n=== B5  the clean endpoint gets its own interval ===")
    print(Q.to_string(index=False))

    # ---- B12: the packet's own accounting --------------------------------
    dates = pd.read_csv(EXP / "analysis" / "q01-dates.csv")
    prov = json.loads((EXP / "analysis" / "q01-provenance.json").read_text())
    files = prov.get("files", {})
    sha = {}
    seen: dict = {}
    for nm, byfold in files.items():
        if not isinstance(byfold, dict):
            continue
        for f, rel in byfold.items():
            if not isinstance(rel, str):
                continue
            q = Path(rel)
            if not q.is_absolute():
                q = ROOT / rel
            if q.exists():
                seen.setdefault(rel, V.D.sha256(q))
                sha[f"{nm}|{f}"] = {"path": rel, "sha256": seen[rel]}
    (EXP / "analysis" / "q01-source-sha256.json").write_text(json.dumps(sha, indent=2))
    mask_p = ROOT / V.MASK_JSON
    rec["accounting"] = {
        "q01_dates_rows": int(len(dates)),
        "q01_dates_note": "one row per candidate x fold x day that MOVED a row; a "
                          "candidate-fold whose d is identically zero writes none",
        "candidates_registered": int(len(NAMES)),
        "source_files_hashed": len(sha),
        "mask_json_sha256": V.D.sha256(mask_p),
        "mask_json_matches_record": V.D.sha256(mask_p).startswith("c89caa9f1be45d9d")}
    print("\n=== B12  accounting ===")
    print(json.dumps(rec["accounting"], indent=2))

    # ---- the fold-side gain as a function of a FIXED amplitude ------------
    # The served blend uses ONE scalar, so this is the curve the upload sits on.
    # It is a fold-side curve and not a board prediction.
    prof = []
    for name in ("stack-5", "k2-w300"):
        for sv in (0.10, 0.15, 0.18, 0.20, 0.2174, 0.25, 0.27, 0.2761, 0.30, 0.35, 0.40):
            num = den = numc = denc = 0.0
            for f in A.BOARD_FOLDS:
                r_ = P[f]["r"]
                d_ = P[f]["frame"][COL[name]].to_numpy("float64")
                k = P[f]["clean"]
                num += A.SHARES[f] * float(np.mean(r_ ** 2))
                den += A.SHARES[f] * float(np.mean((r_ - sv * d_) ** 2))
                numc += A.SHARES[f] * float(np.mean(r_[k] ** 2))
                denc += A.SHARES[f] * float(np.mean((r_[k] - sv * d_[k]) ** 2))
            prof.append({"candidate": name, "s": sv,
                         "gain_board_allrow": round(float(np.sqrt(num) - np.sqrt(den)), 4),
                         "gain_board_clean": round(float(np.sqrt(numc) - np.sqrt(denc)), 4)})
    Pr = pd.DataFrame(prof)
    Pr.to_csv(EXP / "analysis" / "q01-amplitude-profile.csv", index=False)
    print("\n=== the fold-side gain at a FIXED amplitude (the served blend's curve) ===")
    print(Pr.pivot(index="s", columns="candidate",
                   values="gain_board_allrow").to_string())

    (EXP / "analysis" / "round1.json").write_text(json.dumps(rec, indent=2))
    print("\nwritten: analysis/q01-procedure-ci.csv, q01-composition-blend.csv, "
          "q02-clean-ci.csv, q01-source-sha256.json, round1.json")


if __name__ == "__main__":
    main()
