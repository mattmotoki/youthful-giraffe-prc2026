"""First deliverable: an estimate of the bulk ceiling, from three independent angles."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")
import common as C
import prc.data as d


def within_group_sigma(df, keys, col="blk_aobt", min_n=2):
    """RMS deviation of `col` from its group mean, pooled, with the df correction."""
    g = df.groupby(keys, observed=True)[col]
    n = g.transform("size")
    m = n >= min_n
    dev = (df.loc[m, col] - g.transform("mean")[m]).to_numpy()
    nn = n[m].to_numpy()
    ss = float(np.sum(dev ** 2))
    dof = float(np.sum(1 - 1 / nn))
    ngroups = int(df.loc[m].groupby(keys, observed=True).ngroups)
    return {"rows": int(m.sum()), "groups": ngroups,
            "sigma": np.sqrt(ss / dof) if dof > 0 else np.nan,
            "share_pct": 100 * m.mean()}


def main():
    dep = pd.read_parquet(d.ROOT / "artifacts/exp-003-anatomy/dep_all.parquet")
    dep["month"] = dep["month"].astype(str)
    dep["date"] = dep.MVT_TIME_UTC_mvt.dt.floor("D")
    dep["h"] = dep.hour.astype(int)
    dep["h2"] = (dep.hour // 2).astype(int)
    dep["slot30"] = dep.MVT_TIME_UTC_mvt.dt.floor("30min")
    dep["apt_rwy"] = dep.apt + "|" + dep.RUNWAY_mvt.astype(str)
    bulk = dep[(~dep.is_ranking) & dep.has_nm & dep.TAXITIME_SEC_mvt.le(3600)]
    out = []; W = out.append
    W("# Bulk ceiling: three estimates\n")
    W("The bulk is departures with an NM join and taxi time at or below 3600 s: "
      f"{len(bulk):,} of {len(dep[~dep.is_ranking]):,} 2025 departures "
      f"({100*len(bulk)/len(dep[~dep.is_ranking]):.2f} %).\n")

    W("## 1. Matched-group residual: how much of the offset survives perfect conditioning\n")
    W("Pooled within-group standard deviation of `BLOCK - AOBT_3` over 2025 bulk rows, "
      "for groups that share everything named. A model that conditions on exactly these "
      "columns and nothing else cannot beat the `sigma` column; a model that also uses "
      "take-off minus AOBT, neighbours and congestion can go below it. The row where "
      "sigma stops falling as the grouping gets finer is where the offset stops being a "
      "function of the flight's identity.\n")
    specs = [
        ("airport", ["apt"]),
        ("airport x hour", ["apt", "h"]),
        ("airport x runway x hour", ["apt_rwy", "h"]),
        ("stand", ["apt_stand"]),
        ("stand x runway", ["apt_stand", "RUNWAY_mvt"]),
        ("stand x runway x 2h", ["apt_stand", "RUNWAY_mvt", "h2"]),
        ("stand x runway x aircraft type", ["apt_stand", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt"]),
        ("stand x runway x operator", ["apt_stand", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt"]),
        ("stand x runway x type x 2h", ["apt_stand", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "h2"]),
        ("airport x same 30-min slot (same day)", ["apt", "slot30"]),
        ("airport x runway x same 30-min slot", ["apt_rwy", "slot30"]),
        ("stand x same day", ["apt_stand", "date"]),
        ("stand x runway x same day", ["apt_stand", "RUNWAY_mvt", "date"]),
        ("stand x runway x type x same day", ["apt_stand", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "date"]),
    ]
    rows = []
    for name, k in specs:
        r = within_group_sigma(bulk, k)
        r["grouping"] = name
        rows.append(r)
    W(C.md_table(pd.DataFrame(rows).set_index("grouping")[["rows", "groups", "share_pct", "sigma"]],
                 floatfmt="{:.1f}"))
    W("")
    W("Same instrument on the *taxi time itself* (what a model without the NM columns faces):\n")
    rows = []
    for name, k in specs[:9]:
        r = within_group_sigma(bulk, k, col="TAXITIME_SEC_mvt")
        r["grouping"] = name
        rows.append(r)
    W(C.md_table(pd.DataFrame(rows).set_index("grouping")[["rows", "groups", "share_pct", "sigma"]],
                 floatfmt="{:.1f}"))
    W("")

    W("## 2. Back-out from the leaderboard\n")
    rk = dep[dep.is_ranking]
    n_rk = len(rk)
    n_noac = int(rk.AIRCRAFT_TYPE_mvt.isna().sum())
    n_nonm = int((~rk.has_nm).sum())
    n_nonm_nonoac = n_nonm - n_noac
    # tail rate among NM rows, from 2025
    nmtr = dep[(~dep.is_ranking) & dep.has_nm]
    tail_rate = float((nmtr.TAXITIME_SEC_mvt > 3600).mean())
    n_nm = n_rk - n_nonm
    n_tail = tail_rate * n_nm
    n_bulk = n_nm - n_tail
    W(f"Hard upper bound on anyone's bulk RMSE: 277.3 * sqrt(N / n_bulk) = "
      f"{277.3 * np.sqrt(n_rk / n_bulk):.1f} s (achieved only by a team that scores "
      f"zero error on every non-bulk row).\n")
    W(f"Ranking population: {n_rk:,} departures = {n_noac} noac + {n_nonm_nonoac:,} other no-NM "
      f"+ {n_tail:,.0f} NM rows above 3600 s (at the 2025 rate of {100*tail_rate:.3f} %) "
      f"+ {n_bulk:,.0f} bulk rows.\n")
    W("Solving `N * 277.3^2 = n_noac*a^2 + n_other_noNM*b^2 + n_NMtail*c^2 + n_bulk*B^2` "
      "for the bulk RMSE `B`, over a grid of what the leaders might reach on the other "
      "three. **This is a conditional grid, not a bound.** The only unconditional "
      "statement the board supports is the hard upper bound above: a better total does "
      "not imply a better score on any particular segment, so our own segment numbers "
      "cannot be used to bound the leaders' segment numbers either (gpt-5.6-sol round 3).\n")
    rows = []
    for a in (3000, 4000, 4500, 5000):
        for b in (600, 900, 1400, 2000):
            for c in (1200, 1800, 2500):
                tot = n_rk * 277.3 ** 2
                rest = n_noac * a ** 2 + n_nonm_nonoac * b ** 2 + n_tail * c ** 2
                B2 = (tot - rest) / n_bulk
                rows.append({"noac RMSE": a, "other no-NM RMSE": b, "NM tail RMSE": c,
                             "implied bulk RMSE": np.sqrt(B2) if B2 > 0 else np.nan})
    W(C.md_table(pd.DataFrame(rows).set_index("noac RMSE"), floatfmt="{:.1f}"))
    W("")
    C.write("q0_ceiling.md", "\n".join(out))


if __name__ == "__main__":
    main()
