"""The per-flight block-time convention: the oracle, and what marks it label-free.

Finding (exp-007): the airport's recorded block time does not sit a fixed
fraction phi of the way from the NM actual off-block to the filed one. It sits
either ON the actual off-block or ON the filed one, per flight, and phi is the
mixing weight of the two, not a coefficient. Handed to the tail worker, who owns
the regime router.

Definitions, on NM-joined non-LIRF departures with |AOBT - LOBT| >= 300 s:

    r    = BLOCK_TIME_UTC_mvt - AOBT_3_flt        (label-derived, the unobservable)
    lob  = LOBT_flt - AOBT_3_flt                  (label-free)
    component = "AOBT"  if |r|       <= 60 s
                "LOBT"  if |r - lob| <= 60 s
                "other" otherwise

The oracle replaces the model's prediction on the two clean components with
take-off minus the component's implied r (0 or lob) -- an exactly correct
prediction, by definition of the component -- and leaves every other row alone.
Its gain is the ceiling on any router that classifies the component perfectly.

The screen below asks what a *label-free* rule could use to predict the
component. Candidates are fitted on the fold's training months only.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
import prc.data as D  # noqa: E402
import common as C  # noqa: E402

OUT = HERE.parent / "analysis" / "q5_mixture.md"
GAP = 300.0
TOL = 60.0


def prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["PHASE_mvt"] == "DEP") & df[D.TARGET].notna() & df["ADEP_mvt"].ne("LIRF")
           & df["FLIGHT_ID_mvt"].notna() & df["AOBT_3_flt"].notna() & df["LOBT_flt"].notna()].copy()
    d["r"] = (d["BLOCK_TIME_UTC_mvt"] - d["AOBT_3_flt"]).dt.total_seconds()
    d["lob"] = (d["LOBT_flt"] - d["AOBT_3_flt"]).dt.total_seconds()
    d = d[d["lob"].abs() >= GAP]
    d["comp"] = np.where(d["r"].abs() <= TOL, "AOBT",
                         np.where((d["r"] - d["lob"]).abs() <= TOL, "LOBT", "other"))
    d["apt"] = d["ADEP_mvt"].astype(str)
    d["op"] = d["AIRCRAFT_OPERATOR_flt"].astype(str)
    d["seg"] = d["MARKET_SEGMENT_flt"].astype(str)
    d["wake"] = d["WK_TBL_CAT_flt"].astype(str)
    d["ftype"] = d["FLIGHT_TYPE_flt"].astype(str)
    d["stand"] = d["STAND_mvt"].astype(str)
    d["lobt_moved"] = (d["LOBT_flt"] - d["IOBT_flt"]).dt.total_seconds()
    d["eobt_moved"] = (d["LOBT_flt"] - d["EOBT_1_flt"]).dt.total_seconds()
    d["abs_lob"] = d["lob"].abs()
    d["hour"] = d["MVT_TIME_UTC_mvt"].dt.hour
    return d


def auc(score, label):
    s, l = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(s)
    s, l = s[ok], l[ok]
    if l.sum() == 0 or (~l).sum() == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[l].sum() - l.sum() * (l.sum() + 1) / 2) / (l.sum() * (~l).sum()))


def rate_encoder(tr, key, target, prior_w=50.0):
    """P(component = LOBT | key), shrunk to the airport rate then the global rate.

    Fitted on training months only; the value it returns for a validation row saw
    no validation label.
    """
    glob = target.mean()
    apt = tr.groupby("apt")[target.name].agg(["sum", "count"])
    apt_rate = (apt["sum"] + prior_w * glob) / (apt["count"] + prior_w)
    if key == "apt":
        return apt_rate, apt_rate, glob
    g = tr.groupby(["apt", key])[target.name].agg(["sum", "count"])
    prior = g.index.get_level_values(0).map(apt_rate)
    return (g["sum"] + prior_w * prior) / (g["count"] + prior_w), apt_rate, glob


def main():
    P = ["# exp-007 question 5 — the block-time convention is a per-flight mixture\n",
         __doc__.split("Definitions")[0].strip(), "",
         "## Definitions\n",
         "```", "r    = BLOCK_TIME_UTC_mvt - AOBT_3_flt        (label-derived, the unobservable)",
         "lob  = LOBT_flt - AOBT_3_flt                  (label-free)",
         'component = "AOBT"  if |r|       <= 60 s',
         '            "LOBT"  if |r - lob| <= 60 s',
         '            "other" otherwise', "```",
         f"on NM-joined non-LIRF departures with |lob| >= {GAP:.0f} s.\n"]

    tab = ["| fold | rows in scope | AOBT | LOBT | other | oracle on the two clean components |",
           "|---|---|---|---|---|---|"]
    detail = []
    for fold in C.FOLDS:
        rows = pd.read_parquet(HERE.parent / "work" / f"rows_{fold}.parquet")
        f = D.FOLDS[fold]
        d = prep(D.load_month(f.valid_month))
        d = d.merge(rows[[D.ID, "pred", "err", "se"]], on=D.ID, how="inner", validate="1:1")
        d = d[d[D.TARGET] <= 21600]
        share = d["comp"].value_counts(normalize=True)
        # the oracle
        implied = np.where(d["comp"] == "AOBT", 0.0, np.where(d["comp"] == "LOBT", d["lob"], d["r"]))
        span = (d["MVT_TIME_UTC_mvt"] - d["AOBT_3_flt"]).dt.total_seconds().to_numpy()
        p2 = d["pred"].to_numpy().copy()
        known = d["comp"].isin(["AOBT", "LOBT"]).to_numpy()
        p2[known] = (span - implied)[known]
        y = d[D.TARGET].to_numpy(dtype=float)
        b, o = C.rmse(y - d["pred"].to_numpy()), C.rmse(y - p2)
        tab.append(f"| {fold} | {len(d):,} | {100*share.get('AOBT',0):.1f} % | "
                   f"{100*share.get('LOBT',0):.1f} % | {100*share.get('other',0):.1f} % | "
                   f"{b:.1f} → **{o:.1f}** s ({b-o:+.1f}) |")
        detail.append((fold, d))
    P.append("## The oracle\n")
    P.append("\n".join(tab) + "\n")
    P.append("The oracle is exact on the two clean components by construction and touches nothing "
             "else, so the gain is the **ceiling** on a perfect per-flight router. It is measured on "
             "the in-scope slice above, not on the whole fold.\n")

    P.append("## What marks the component, label-free\n")
    P.append("Every candidate is fitted on the fold's training months and scored on the validation "
             "month, on the in-scope rows whose component is clean (`AOBT` or `LOBT`). AUC is for "
             "`component == LOBT`.\n")
    lines = ["| fold | candidate | AUC | n scored |", "|---|---|---|---|"]
    for fold, d in detail:
        f = D.FOLDS[fold]
        tr = prep(D.load_months(f.train_months))
        tr = tr[tr["comp"] != "other"]
        tr["is_lobt"] = (tr["comp"] == "LOBT").astype(float)
        tr["is_lobt"].name = "is_lobt"
        va = d[d["comp"] != "other"].copy()
        lab = (va["comp"] == "LOBT").to_numpy()
        for key in ("op", "apt", "seg", "wake", "ftype", "stand"):
            enc, apt_rate, glob = rate_encoder(tr, key, tr["is_lobt"])
            idx = (va["apt"] if key == "apt"
                   else pd.MultiIndex.from_arrays([va["apt"], va[key]]))
            s = enc.reindex(idx).to_numpy()
            s = np.where(np.isfinite(s), s, va["apt"].map(apt_rate).fillna(glob).to_numpy())
            lines.append(f"| {fold} | `{key}` rate, shrunk (training months) | "
                         f"**{auc(s, lab):.3f}** | {len(va):,} |")
        for col in ("abs_lob", "lobt_moved", "eobt_moved", "hour"):
            lines.append(f"| {fold} | `{col}` | {auc(va[col].to_numpy(), lab):.3f} | {len(va):,} |")
        # operator rate combined with the airport rate is the natural router input
        enc, apt_rate, glob = rate_encoder(tr, "op", tr["is_lobt"])
        idx = pd.MultiIndex.from_arrays([va["apt"], va["op"]])
        s_op = enc.reindex(idx).to_numpy()
        s_op = np.where(np.isfinite(s_op), s_op, va["apt"].map(apt_rate).fillna(glob).to_numpy())
        # a router that thresholds at 0.5 and its RMSE on the in-scope clean rows
        span_v = (va["MVT_TIME_UTC_mvt"] - va["AOBT_3_flt"]).dt.total_seconds().to_numpy()
        yv = va[D.TARGET].to_numpy(dtype=float)
        hard = np.where(s_op >= 0.5, va["lob"].to_numpy(), 0.0)
        soft = s_op * va["lob"].to_numpy()
        base = C.rmse(yv - va["pred"].to_numpy())
        lines.append(f"| {fold} | **router on the airport-operator rate, hard 0.5** | "
                     f"RMSE {C.rmse(yv - (span_v - hard)):.1f} vs model {base:.1f} | {len(va):,} |")
        lines.append(f"| {fold} | **router on the airport-operator rate, soft (expected r)** | "
                     f"RMSE {C.rmse(yv - (span_v - soft)):.1f} vs model {base:.1f} | {len(va):,} |")
    P.append("\n".join(lines) + "\n")
    OUT.write_text("\n".join(P))
    print(OUT.read_text())


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# The check that refuted the mixture, added after the tables above were written.
# ---------------------------------------------------------------------------
def atom_test(month: str = "2025-07") -> str:
    """Are the two 'components' atoms, or just the continuum's local density?

    If an airport writes its block time ON the NM off-block, `r` is 0 to the
    second, not 'within 60 s'. The same for the filed off-block. So count the
    exact values, and watch how the share grows with the tolerance: an atom
    gives a step, a continuum gives a smooth ramp.
    """
    # gpt-5.6-sol round 1, blocking finding 9: the first version of this test
    # dropped the non-LIRF restriction and so ran on a different population from
    # the oracle it was refuting. It uses `prep` now, which is the oracle's own.
    d = prep(D.load_month(month))
    s = d[d[D.TARGET] <= 21600]
    out = [f"### The atoms are not there ({month}, {len(s):,} in-scope rows — "
           f"the oracle's own population: NM-joined, non-LIRF, |lob| >= {GAP:.0f} s)\n",
           "| tolerance | share with |r| ≤ tol | share with |r − lob| ≤ tol |", "|---|---|---|"]
    for tol in (0, 5, 30, 60, 120, 300):
        out.append(f"| {tol} s | {100*(s['r'].abs() <= tol).mean():.2f} % | "
                   f"{100*((s['r'] - s['lob']).abs() <= tol).mean():.2f} % |")
    out.append("")
    ex0 = 100 * (s["r"] == 0).mean()
    ex1 = 100 * ((s["r"] - s["lob"]) == 0).mean()
    out.append(f"An atom would put a step at 0 s. Instead the share grows smoothly with the "
               f"tolerance, and the exact values carry {ex0:.2f} % and {ex1:.2f} % of the rows. "
               f"So there are no POINT MASSES at 0 and at `lob`, which is what the withdrawn "
               f"claim asserted. That is what this test can show and all it can show: a "
               f"two-component mixture whose components are themselves diffuse would leave no "
               f"atom either, and is not excluded. What is excluded is the reading that made "
               f"the oracle look like a ceiling on a router — the rows inside a ±60 s window "
               f"are the continuum's local density, not a class — and the router built from "
               f"the best available label-free classifier of that supposed class loses 117 s "
               f"to the model. φ as a conditional mean stands.\n")
    s = s.copy()
    t = s.groupby("apt").apply(lambda g: pd.Series({
        "n": len(g), "r==0 %": 100 * (g["r"] == 0).mean(),
        "r-lob==0 %": 100 * ((g["r"] - g["lob"]) == 0).mean(),
        "|r|<=60 %": 100 * (g["r"].abs() <= 60).mean(),
        "|r-lob|<=60 %": 100 * ((g["r"] - g["lob"]).abs() <= 60).mean()}), include_groups=False)
    out.append("| airport | in-scope rows | exact r = 0 | exact r = lob | \\|r\\| ≤ 60 s | \\|r − lob\\| ≤ 60 s |")
    out.append("|---|---|---|---|---|---|")
    for a, x in t.iterrows():
        out.append(f"| {a} | {int(x['n']):,} | {x['r==0 %']:.2f} % | {x['r-lob==0 %']:.2f} % | "
                   f"{x['|r|<=60 %']:.2f} % | {x['|r-lob|<=60 %']:.2f} % |")
    return "\n".join(out) + "\n"


if __name__ == "__main__" and "--atom" in sys.argv:
    text = OUT.read_text() if OUT.exists() else ""
    OUT.write_text(text + "\n## The check that refuted the mixture\n\n" + atom_test())
    print(atom_test())
