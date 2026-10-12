"""A first look at why fold gains might not reach the board, from the record alone.

Three questions, none of which needs the ranking file:

**Q1 -- is the swap gain a clean gain or a rollover gain?**  The january fold's
raw RMSE is 339 and its clean RMSE is 220: two rows of 2025-01 carry 118 s of
it (`prc.data.UNREACHED_ROLLOVER`).  A swap that improves the raw january much
more than the clean january has bought most of its fold gain on a row class the
ranking months are believed not to contain, and that part of the gain cannot
transfer.  The board predictor is `clean + 4`, so this is measurable now.

**Q2 -- where in the fold does each swap's gain sit?**  Per fold and per error
class, so a gain concentrated in the tail (0.34 % of rows, 67 % of the squared
error) is separated from a gain on the bulk.

**Q3 -- how big is each swap's shift on the RANKING rows?**  With
`candidates/ranking-gates.parquet` present (`code/regime_table.py`, needs
`data/raw/ranking.parquet`) this is the EXACT served shift: the hybrid rule
picks the per-row weight vector, the floor is applied, and so is the residual
guard, which is the last thing `prc.blend.build_candidate` does.  Pass
`--gates <path>` or leave the default.

Without the gates table the shift is only PARTLY computable.  blend-5 uses one
weight vector inside `nm_late` and the same vector in the other three regimes,
so each row's served value is one of two numbers -- but which one is unknown
per row, and the residual guard cannot be applied at all, because it needs
`d_takeoff_aobt`.  The fallback therefore reports the two pure-vector
populations and labels them `unguarded`.  **They do not bracket the mixture.**
Every individual row's shift is one of the two computed values, so the SET of
per-row shifts is exact; an aggregate over the mixed population (rms, p99,
share over 60 s) mixes subsets of the two and can fall outside the range of the
two whole-population aggregates.  Review round 1 finding 3: the earlier version
of this file said "brackets", applied the floor twice and never applied the
guard.

Only the V3-allin swap can be done at all: `T-tailv2` and `L2-hub`'s ranking
files were `artifacts/`-only and are gone, so their served side cannot be
reconstructed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames as F  # noqa: E402
import ladder as L  # noqa: E402
import regime_table as RT  # noqa: E402
import rule as R  # noqa: E402
sys.path.insert(0, str(F.ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

OUT = F.EXP / "analysis"
#: exp-014's ranking regime census, the only per-regime ranking counts in git.
CENSUS = "experiments/exp-014-robustness/analysis/q3b_candidate_delta_by_regime.csv"


# --------------------------------------------------------- Q1 and Q2 -------
def fold_questions(cache: dict) -> pd.DataFrame:
    rows = []
    base = L.run(L.B4, cache)
    for label, names in [L.SWAP[0]] + L.SINGLE + [L.SWAP[-1]]:
        r = L.run(names, cache)
        for f in F.FOLDS:
            rows.append(dict(config=label, fold=f,
                             raw=r[f]["rmse"], clean=r[f]["rmse_clean"],
                             d_raw=r[f]["rmse"] - base[f]["rmse"],
                             d_clean=r[f]["rmse_clean"] - base[f]["rmse_clean"],
                             **{k: v for k, v in r["decomp"][f].items() if k != "rmse"}))
    T = pd.DataFrame(rows)
    for c in ("le3600", "gt3600", "no_nm", "noac"):
        T["d_" + c] = T[c] - T.merge(
            T[T.config == "blend-4"][["fold", c]], on="fold", suffixes=("", "_b"))[c + "_b"].values
    return T


# ------------------------------------------------------------------ Q3 -----
def _describe(d: np.ndarray) -> dict:
    return {"mean": float(d.mean()), "rms": float(np.sqrt((d ** 2).mean())),
            "max": float(d.max()), "min": float(d.min()),
            "p01": float(np.quantile(d, 0.01)), "p99": float(np.quantile(d, 0.99)),
            "share_over_60s": float((np.abs(d) > 60).mean())}


def _serve(cfg: dict, cols: dict, ids: np.ndarray, gates: pd.DataFrame | None,
           reg: str | None = None) -> np.ndarray:
    """One config's served ranking vector.

    With `gates`, this is the whole serve path: hybrid rule, floor, residual
    guard -- the same three steps, in the same order, as
    `prc.blend.build_candidate`.  Without it, `reg` selects one pure weight
    vector and the guard is skipped, which is NOT the served value.
    """
    names = list(cfg["ranking"])
    P = np.stack([cols[n].reindex(ids).to_numpy("float64") for n in names], axis=1)
    P = np.where(np.isfinite(P), P, cfg["floor"])
    if gates is None:
        # Only the floor is available without `d_takeoff_aobt`; this branch is
        # not the served value and does not claim to be.
        P = np.maximum(P, cfg["floor"])
        w = np.array([cfg["weights"][reg][n] for n in names], "float64")
        return np.maximum(P @ w, cfg["floor"])
    # Prepare each MEMBER first -- floor then residual guard, per member -- which
    # is what `build_candidate` and `rebuild_blend.py` do before any weight is
    # applied. Review round 2: this step was missing, so the shifts this file
    # called "exact served" were wrong on 359 rows by up to 14.3 s.
    a_prep = gates["d_takeoff_aobt"].to_numpy("float64")
    lo0, hi0 = cfg.get("guard", F.GUARD)
    for j in range(P.shape[1]):
        P[:, j] = np.maximum(P[:, j], cfg["floor"])
        P[:, j] = B.apply_residual_guard(P[:, j], a_prep, lo0, hi0, cfg["floor"])
    y = np.empty(len(ids), "float64")
    late = gates["regime"].to_numpy() == "nm_late"
    for key, m in (("nm_late", late), ("_all", ~late)):
        w = np.array([cfg["weights"][key][n] for n in names], "float64")
        y[m] = P[m] @ w
    y = np.maximum(y, cfg["floor"])
    lo, hi = cfg.get("guard", F.GUARD)
    return B.apply_residual_guard(y, gates["d_takeoff_aobt"].to_numpy("float64"),
                                  lo, hi, floor=cfg["floor"])


def ranking_shift(drop: str = "V3-allin", gates: pd.DataFrame | None = None) -> dict:
    """blend-5 minus blend-5-without-`drop` on the ranking rows.

    Exact served shift when `gates` is given; two unguarded pure-vector
    populations when it is not (see the module docstring: they do not bracket).
    """
    cfgd = F.EXP / "configs"
    a = json.loads((cfgd / "config-blend-5.json").read_text())
    b = json.loads((cfgd / f"config-b5-drop-{drop}.json").read_text())
    cols = {}
    for n, p in {**a["ranking"], **b["ranking"]}.items():
        d = pd.read_parquet(F.ROOT / p)
        cols[n] = pd.Series(d["y_pred"].to_numpy("float64"),
                            index=d["MVT_ID_mvt"].to_numpy("float64"))
    ids = cols["F1-fresh"].index.to_numpy()
    g = None
    if gates is not None:
        g = gates.set_index("MVT_ID_mvt").reindex(ids).reset_index()
        assert g["regime"].notna().all(), "gates table does not cover every member id"
    res: dict = {"exact": g is not None}
    if g is not None:
        d = _serve(a, cols, ids, g) - _serve(b, cols, ids, g)
        res["served"] = _describe(d)
        res["_delta_all"] = d
    else:
        label = {"_all": "bulk vector (unguarded)", "nm_late": "nm_late vector (unguarded)"}
        for reg in ("_all", "nm_late"):
            d = (_serve(a, cols, ids, None, reg) - _serve(b, cols, ids, None, reg))
            res[label[reg]] = _describe(d)
            if reg == "_all":
                res["_delta_all"] = d
    res["_ids"] = ids
    return res


#: The two regime populations that ARE named on the ranking side, by id.
#: `noac` is exp-002's 388-row head file; `nonm` is exp-013's `seg` flag minus
#: those ids. `nm_late` has no committed source, so `nm_bulk` cannot be
#: separated from it and neither is reported here.
NOAC_IDS = "experiments/exp-002-tail/noac_ranking_predictions.parquet"
NONM_SEG = "experiments/exp-013-nonm/oof/nonm-v1-ranking.parquet"


def by_named_regime(ids: np.ndarray, delta: np.ndarray) -> pd.DataFrame:
    """The shift on the ranking rows whose regime the record names.

    Neither population can be `nm_late` -- `noac` and `nonm` are disjoint from
    it by definition -- so both take blend-5's bulk weight vector, and the
    weight-vector choice is right for these rows even without the gates table.
    The residual guard is still missing in the fallback, so call those numbers
    unguarded, not exact; with `--gates` the caller passes the guarded delta and
    they are the served shift.  `noac` is the population §5a says carries three
    quarters of blend-5's fold gain, so its shift on the 388 served rows is the
    number that decides whether that gain can transfer.
    """
    noac = set(pd.read_parquet(F.ROOT / NOAC_IDS, columns=["MVT_ID_mvt"])
               ["MVT_ID_mvt"].astype("float64"))
    seg = pd.read_parquet(F.ROOT / NONM_SEG, columns=["MVT_ID_mvt", "seg"])
    seg["MVT_ID_mvt"] = seg["MVT_ID_mvt"].astype("float64")
    segset = set(seg.loc[seg["seg"].astype(bool), "MVT_ID_mvt"])
    m_noac = np.isin(ids, list(noac))
    m_nonm = np.isin(ids, list(segset - noac))
    rows = []
    for lab, m in (("noac (exp-002, 388 rows)", m_noac),
                   ("nonm (exp-013 seg, minus noac)", m_nonm),
                   ("everything else", ~(m_noac | m_nonm))):
        d = delta[m]
        rows.append(dict(group=lab, n=int(m.sum()), mean=float(d.mean()),
                         rms=float(np.sqrt((d ** 2).mean())),
                         max=float(np.abs(d).max()),
                         share_over_60s=float((np.abs(d) > 60).mean())))
    return pd.DataFrame(rows)


def by_month_airport(ids: np.ndarray, delta: np.ndarray,
                     gates: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """Split the shift by month and airport.

    With `gates` the airport comes from the ranking file itself, joined on
    `MVT_ID_mvt`, and there is nothing to verify.

    Without it the only per-airport source is
    `exp-007-winter/work/ranking_daystate.parquet`, which carries `month`, `apt`
    and `day` for 344,841 rows keyed by POSITION in the ranking file, not by
    `MVT_ID_mvt`.  Review round 1 finding 3: the earlier version checked the
    positional join on `month` alone, which two months of 152,719 and 192,122
    rows pass under ANY permutation inside a month, so it verified almost
    nothing.  There is no second id-keyed column in git to check `apt` or `day`
    against, so the honest fallback is to return None and print why.  The month
    agreement is still computed and printed: it is a necessary condition, and
    when it fails the positional reading is refuted outright.
    """
    if gates is not None:
        g = gates.set_index("MVT_ID_mvt").reindex(ids)
        assert g["airport"].notna().all()
        T = pd.DataFrame({"month": g["month"].to_numpy(),
                          "airport": g["airport"].to_numpy(), "d": delta})
        return _agg_month_airport(T)
    ds = pd.read_parquet(F.ROOT / "experiments/exp-007-winter/work/ranking_daystate.parquet",
                         columns=["month", "apt"]).sort_index()
    nm = pd.read_parquet(F.ROOT / "experiments/exp-013-nonm/oof/nonm-v1-ranking.parquet",
                         columns=["MVT_ID_mvt", "month"])
    nm["MVT_ID_mvt"] = nm["MVT_ID_mvt"].astype("float64")
    byid = nm.set_index("MVT_ID_mvt")["month"].reindex(ids)
    if len(ds) != len(ids) or byid.isna().any():
        return None
    agree = float((ds["month"].to_numpy() == byid.to_numpy()).mean())
    print(f"[transfer] positional month agreement {agree:.6f} over {len(ids)} rows "
          f"(a necessary condition only: any within-month permutation also passes it)")
    return None


def _agg_month_airport(T: pd.DataFrame) -> pd.DataFrame:
    g = T.groupby(["month", "airport"])["d"].agg(
        n="size", mean="mean", rms=lambda x: float(np.sqrt((x ** 2).mean())),
        p99=lambda x: float(np.quantile(x, 0.99))).reset_index()
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gates", type=Path, default=RT.OUT,
                    help="ranking-gates.parquet from code/regime_table.py; without it the "
                         "Q3 shifts are unguarded pure-vector populations, not served shifts")
    args = ap.parse_args()
    gates = pd.read_parquet(args.gates) if Path(args.gates).exists() else None
    print(f"[transfer] gates: {args.gates if gates is not None else 'absent -- Q3 is unguarded'}")
    cache: dict = {}
    T = fold_questions(cache)
    T.round(3).to_csv(OUT / "transfer_folds.csv", index=False)
    print("=== Q1: raw vs clean january, per swap (blend-4 is the reference) ===")
    jan = T[T.fold == "january"][["config", "raw", "clean", "d_raw", "d_clean"]]
    jan = jan.assign(rollover_share=(jan.d_raw - jan.d_clean).round(3))
    print(jan.round(3).to_string(index=False))
    print("\n=== Q2: gain per fold and per error class ===")
    print(T[["config", "fold", "d_clean", "d_le3600", "d_gt3600", "d_no_nm", "d_noac"]]
          .round(3).to_string(index=False))

    # The "exact served" claim, checked rather than asserted. When the two
    # candidates this shift is the difference of have been built and verified,
    # the shift must equal their difference on every row. Round 2 caught this
    # claim being false by 14.3 s on 359 rows; it is now self-verifying.
    if gates is not None:
        for member in ("V3-allin", "T-tailv4"):
            a = F.ROOT / "submissions/candidates/exp-026-blend-5.parquet"
            b = F.ROOT / f"submissions/candidates/exp-026-b5-drop-{member}.parquet"
            if not (a.exists() and b.exists()):
                continue
            r = ranking_shift(member, gates=gates)
            got = pd.Series(r["_delta_all"], index=r["_ids"])
            rd = lambda q: pd.Series(  # noqa: E731
                pd.read_parquet(q)[D.TARGET].to_numpy("float64"),
                index=pd.read_parquet(q)["MVT_ID_mvt"].to_numpy("float64"))
            truth = (rd(a) - rd(b)).reindex(got.index)
            e = float(np.abs(got.to_numpy() - truth.to_numpy()).max())
            print(f"[transfer] {member} shift == the verified candidates' difference: "
                  f"max {e:.3g} s")
            if e > 1e-9:
                raise SystemExit(f"{member}: the served shift is not the candidates' "
                                 f"difference (max {e:.6g} s); the 'exact served' claim "
                                 f"would be false")

    kind = "exact served" if gates is not None else "unguarded, one weight vector at a time"
    print(f"\n=== Q3: V3-allin's shift on the 344,841 ranking rows ({kind}) ===")
    rs = ranking_shift(gates=gates)
    for reg, s in rs.items():
        if reg.startswith("_") or reg == "exact":
            continue
        print(f"  {reg:22s}: mean {s['mean']:+8.3f}  rms {s['rms']:8.3f}  "
              f"[{s['min']:+9.1f}, {s['max']:+9.1f}]  |d|>60s on {s['share_over_60s']*100:.2f} %")
    census = pd.read_csv(F.ROOT / CENSUS)
    n_late = int(census[census.regime == "nm_late"]["n"].sum())
    print(f"  the nm_late vector applies to {n_late} rows ({n_late/344841*100:.2f} %), "
          f"exp-014's ranking regime census")

    print("\n=== every member's served shift, by the ranking regimes the record names ===")
    allrows = []
    for member in ("T-tailv4", "C2-cat", "L2-hub8725", "D1-winter", "F1-fresh",
                   "V3-allin", "N1-nmtail"):
        r = ranking_shift(member, gates=gates)
        t = by_named_regime(r["_ids"], r["_delta_all"])
        t.insert(0, "dropped", member)
        allrows.append(t)
    A = pd.concat(allrows, ignore_index=True)
    A.to_csv(OUT / "transfer_shift_all_members.csv", index=False)
    piv = A.pivot(index="dropped", columns="group", values="rms").round(2)
    piv.columns = [c.split(" (")[0] for c in piv.columns]
    print(piv.to_string())
    print("\n(rms seconds of blend-5 minus blend-5-without-that-member, bulk vector)")

    nr = by_named_regime(rs["_ids"], rs["_delta_all"])
    nr.to_csv(OUT / "transfer_shift_by_regime.csv", index=False)
    print("\n=== V3-allin's shift on the ranking regimes the record names ===")
    print(nr.round(3).to_string(index=False))

    g = by_month_airport(rs["_ids"], rs["_delta_all"], gates)
    if g is None:
        print("\n  per-airport split withheld: without the gates table there is no id-keyed "
              "airport column in git to verify the positional join against")
    else:
        g.to_csv(OUT / "transfer_shift_by_airport.csv", index=False)
        print("\n=== V3-allin's shift by month and airport (bulk vector) ===")
        print(g.round(2).to_string(index=False))
    (OUT / "transfer.json").write_text(json.dumps(
        {k: v for k, v in rs.items() if not k.startswith("_")}, indent=2))


if __name__ == "__main__":
    main()
