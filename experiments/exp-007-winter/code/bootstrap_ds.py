"""Paired, airport-day-clustered bootstrap: day-state columns against the matched control.

Both arms come from `run_ds.py`, i.e. the same code path with the day-state step
switched off, so the difference is the columns and not a drift in the pipeline.

Clustering: departures at the same airport on the same day share the whole
surface state, so treating rows as independent understates the interval
(gpt-5.6-sol on exp-006). The assignment asks for airport-day clusters; the
calendar-day variant, which is coarser and therefore more conservative, is
reported beside it.

Slices, following COMMON.md and exp-006's "input-defined core":
  core        NM-joined, non-noac departures      (the rows a model can model)
  core nine   the same, excluding LIRF            (the assignment's slice)
  bulk        core with target <= 3600 s
  worst days  core nine on the worst 5 % of airport-days by control squared error
  disrupted   core nine on the 5 % of airport-days with the highest label-free
              day dispersion z -- the closest a 2025 fold gets to a January-2026 day
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
import prc.data as D  # noqa: E402
import prc.day_state_features as DS  # noqa: E402

# prc.data VERSION 2 adds july_loo: the other eleven months against July 2025.
# The submission model trains on all twelve, so July 2026 is predicted with July
# and August 2025 in training; the forward `july` fold has no same-season data and
# is pessimistic. (july_loo, january) is the desk's board-analogue pair.
FOLDM = {"july": "2025-07", "july_loo": "2025-07", "december": "2025-12", "january": "2025-01"}
_meta: dict = {}


def meta(month: str) -> pd.DataFrame:
    if month not in _meta:
        d = D.load_month(month, ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
                                 "FLIGHT_ID_mvt", "AIRCRAFT_TYPE_mvt", "BLOCK_TIME_UTC_mvt",
                                 "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])
        d = d[d["PHASE_mvt"].eq("DEP")].set_index("MVT_ID_mvt")
        ms = (d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
        _meta[month] = pd.DataFrame({
            "apt": d["ADEP_mvt"].astype(str),
            "nonm": d["FLIGHT_ID_mvt"].isna(),
            "noac": d["AIRCRAFT_TYPE_mvt"].isna(),
            "day": d["MVT_TIME_UTC_mvt"].dt.floor("D").astype("int64"),
            # the unreachable date-rollover rows of analysis/q1_floor.md
            "unreachable": (d["TAXITIME_SEC_mvt"].astype("float64") > 21600)
                           & d["AIRCRAFT_TYPE_mvt"].notna() & ms.abs().le(3600),
        })
    return _meta[month]


_dz: dict = {}


def disrupted_days(fold: str, ids: np.ndarray, frac: float = 0.05) -> np.ndarray:
    """The most disrupted 5 % of airport-days, chosen LABEL-FREE.

    The label-defined "worst days" slice answers a different question: it says
    where the error is. This one says where the *conditions* are, using the same
    column the ranking-month check leans on (`ds_span_std_z`, the day's dispersion
    of take-off minus AOBT against that airport's own 2025 reference). It is the
    closest thing a 2025 fold has to a January-2026 day, and it is fixed before
    any prediction is looked at.
    """
    if fold not in _dz:
        f = D.FOLDS[fold]
        train, masked, _ = D.fold_frames(f)
        z = DS.DayState().fit(train).transform(masked)
        dep = masked[masked["PHASE_mvt"].eq("DEP")]
        _dz[fold] = pd.Series(z["ds_span_std_z"].to_numpy(), index=dep[D.ID].to_numpy())
    z = _dz[fold].reindex(ids).to_numpy()
    thr = np.nanquantile(z, 1 - frac)
    return np.isfinite(z) & (z >= thr)


def load(name: str, fold: str) -> pd.DataFrame:
    return pd.read_parquet(EXP / "oof" / f"{name}-{fold}.parquet").sort_values(D.ID)


def paired(ea, ec, cluster, rng, n=2000):
    keys, inv = np.unique(cluster, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(keys)))
    ends = np.r_[starts[1:], len(order)]
    groups = [order[s:e] for s, e in zip(starts, ends)]
    d0 = np.sqrt(ea.mean()) - np.sqrt(ec.mean())
    boot = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        k = np.concatenate([groups[j] for j in pick])
        boot[i] = np.sqrt(ea[k].mean()) - np.sqrt(ec[k].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return d0, lo, hi, len(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="d1")
    ap.add_argument("--seeds", default="7")
    ap.add_argument("--folds", default="july,july_loo,december,january")
    ap.add_argument("--base-name", default=None,
                    help="pair the ds arm of --name against the base arm of another run, so a "
                         "column-subset arm can reuse an existing matched control")
    ap.add_argument("--out", default="analysis/q3_bootstrap.md")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    rng = np.random.default_rng(11)

    P = ["# exp-007 question 3 — day-state columns, paired and clustered\n",
         f"Arms: `{a.name}-ds-s<seed>` against `{a.name}-base-s<seed>`, seeds {seeds}, both",
         "produced by `code/run_ds.py` on retro-v1's `G1cv-s7` configuration. A negative",
         "number is the day-state arm winning.\n"]
    tab = ["## Fold numbers\n",
           "| fold | seed | arm | RMSE | bulk ≤3600 | tail >3600 | no-NM | noac | rounds | features |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    log = (EXP / "climb.log").read_text().splitlines() if (EXP / "climb.log").exists() else []
    for line in log:
        f = line.split("\t")
        if len(f) < 9 or not f[1].startswith(a.name):
            continue
        nm, fold = f[1], f[2]
        vals = {k.split("=")[0]: k.split("=")[1] for k in f[3:] if "=" in k}
        arm = "ds " if "-ds-" in nm else "base"
        tab.append(f"| {fold} | {vals.get('seed','')} | {arm} | {vals.get('rmse','')} | "
                   f"{vals.get('bulk','')} | {vals.get('tail','')} | {vals.get('no_nm','')} | "
                   f"{vals.get('noac','')} | {vals.get('rounds','')} | {vals.get('nfeat','')} |")
    P.append("\n".join(tab) + "\n")

    P.append("## Paired airport-day-clustered bootstrap, 2,000 draws\n")
    rows = ["| fold | slice | rows | clusters | Δ RMSE (ds − base) | 95 % interval | calendar-day interval |",
            "|---|---|---|---|---|---|---|"]
    for fold in [f for f in a.folds.split(",") if f in FOLDM]:
        month = FOLDM[fold]
        try:
            arms = {}
            for arm in ("ds", "base"):
                nm = a.base_name if (arm == "base" and a.base_name) else a.name
                ps = [load(f"{nm}-{arm}-s{s}", fold) for s in seeds]
                ids = ps[0][D.ID].to_numpy()
                assert all((p[D.ID].to_numpy() == ids).all() for p in ps)
                arms[arm] = (ids, ps[0]["y_true"].to_numpy(),
                             np.stack([p["y_pred"].to_numpy() for p in ps]).mean(0))
        except FileNotFoundError:
            continue
        assert (arms["ds"][0] == arms["base"][0]).all()
        ids, y = arms["ds"][0], arms["ds"][1]
        M = meta(month).reindex(ids)
        ea = (y - arms["ds"][2]) ** 2
        ec = (y - arms["base"][2]) ** 2
        core = (~M["nonm"].to_numpy()) & (~M["noac"].to_numpy())
        nine = M["apt"].ne("LIRF").to_numpy()
        reach = (~M["unreachable"].to_numpy())
        bulk = core & (y <= 3600)
        apt_day = M["apt"].to_numpy() + "|" + M["day"].astype(str).to_numpy()
        # worst 5 % of airport-days by the CONTROL's squared error, on the nine
        sse = pd.Series(ec[core & nine]).groupby(apt_day[core & nine]).sum()
        worst = set(sse.sort_values(ascending=False).head(max(1, int(round(0.05 * len(sse))))).index)
        wsel = core & nine & np.isin(apt_day, list(worst))
        dz = disrupted_days(fold, ids)
        for label, sel in (("whole fold (reachable)", reach),
                           ("core", core), ("core, other nine", core & nine),
                           ("bulk ≤3600, other nine", bulk & nine),
                           ("worst 5 % of airport-days (label-defined)", wsel),
                           ("most disrupted 5 % of days (label-free)", core & nine & dz)):
            d0, lo, hi, nc = paired(ea[sel], ec[sel], apt_day[sel], rng)
            _, lo2, hi2, _ = paired(ea[sel], ec[sel], M["day"].astype(str).to_numpy()[sel], rng)
            rows.append(f"| {fold} | {label} | {int(sel.sum()):,} | {nc} | **{d0:+.2f} s** | "
                        f"[{lo:+.2f}, {hi:+.2f}] | [{lo2:+.2f}, {hi2:+.2f}] |")
    P.append("\n".join(rows) + "\n")
    (EXP / a.out).write_text("\n".join(P))
    print((EXP / a.out).read_text())


if __name__ == "__main__":
    main()
