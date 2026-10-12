"""Q1(a): pooled within-group variance of the taxi time over near-duplicate groups.

Groups matched on observables. The pooled within-group standard deviation is the
same quantity as the within-pair disagreement divided by sqrt(2), computed
without materialising pairs. It is an **upper bound on the irreducible standard
deviation**: unmatched observables still vary inside a group.
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")   # for `features`
sys.path.insert(0, "experiments/exp-011-bulkfloor/code")  # must win for `common`
import prc.data as d
import common as C

FEAT = d.ROOT / "artifacts/exp-003-anatomy/feat"
COLS = ["MVT_ID_mvt", "apt", "stand_raw", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt",
        "AIRCRAFT_TYPE_mvt", "hour", "TAXITIME_SEC_mvt", "has_nm", "noac",
        "aobt_lobt", "nbr_apt_30m", "MVT_TIME_UTC_mvt"]


def pooled_sigma(df, keys, col="TAXITIME_SEC_mvt"):
    g = df.groupby(keys, observed=True)[col]
    n = g.transform("size")
    m = n >= 2
    if not m.any():
        return dict(rows=0, groups=0, share=0.0, sigma=np.nan)
    dev = (df.loc[m, col] - g.transform("mean")[m]).to_numpy(float)
    nn = n[m].to_numpy()
    dof = float(np.sum(1 - 1 / nn))
    return dict(rows=int(m.sum()), groups=int(df.loc[m].groupby(keys, observed=True).ngroups),
                share=100 * float(m.mean()), sigma=float(np.sqrt((dev ** 2).sum() / dof)))


def main():
    f = pd.concat([pd.read_parquet(FEAT / f"{m}.parquet", columns=COLS) for m in d.MONTHS],
                  ignore_index=True)
    b = f[f.has_nm & f.noac.ne(1) & f.TAXITIME_SEC_mvt.le(3600)].copy()
    b["date"] = b.MVT_TIME_UTC_mvt.dt.floor("D").astype(str)
    b["gap_bin"] = (b.aobt_lobt.fillna(0) / 120).round().astype(int)      # 2-minute bins
    b["q_bin"] = (b.nbr_apt_30m.fillna(0) / 4).round().astype(int)        # queue, 4-movement bins
    b["h2"] = (b.hour // 2).astype(int)
    b["month"] = b.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    b["week"] = b.MVT_TIME_UTC_mvt.dt.strftime("%Y-%W")
    out = []; W = out.append
    W("# Q1(a) — near-duplicate disagreement, all twelve 2025 months\n")
    W(f"Bulk = NM join, aircraft type present, taxi <= 3600 s: {len(b):,} of "
      f"{len(f):,} departures ({100*len(b)/len(f):.2f} %). `sigma` is the pooled "
      "within-group standard deviation of the taxi time, which equals the "
      "within-pair disagreement over sqrt(2). **It is an upper bound on the "
      "irreducible sigma**, not a floor: what a group does not match on still "
      "varies inside it. Rungs without a date or month key pool rows across the "
      "whole year, so they also carry the seasonal variation the model can "
      "condition on: the `same month`, `same week` and `same day` rungs are the "
      "honest ones, at the cost of coverage.\n")
    ladder = [
        ("airport", ["apt"]),
        ("airport x 2-hour block", ["apt", "h2"]),
        ("airport x runway x 2-hour block", ["apt", "RUNWAY_mvt", "h2"]),
        ("stand x runway", ["apt", "stand_raw", "RUNWAY_mvt"]),
        ("stand x runway x operator", ["apt", "stand_raw", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt"]),
        ("stand x runway x operator x 2h", ["apt", "stand_raw", "RUNWAY_mvt",
                                            "AIRCRAFT_OPERATOR_flt", "h2"]),
        ("... + same AOBT-LOBT 2-min bin", ["apt", "stand_raw", "RUNWAY_mvt",
                                            "AIRCRAFT_OPERATOR_flt", "h2", "gap_bin"]),
        ("... + same queue bin", ["apt", "stand_raw", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt",
                                  "h2", "gap_bin", "q_bin"]),
        ("... + same day (the tightest, and the thinnest)",
         ["apt", "stand_raw", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt", "h2", "gap_bin",
          "q_bin", "date"]),
        ("stand x runway x same day", ["apt", "stand_raw", "RUNWAY_mvt", "date"]),
        ("stand x runway x operator x same day",
         ["apt", "stand_raw", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt", "date"]),
        ("stand x runway x operator x 2h x same month",
         ["apt", "stand_raw", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt", "h2", "month"]),
        ("stand x runway x operator x 2h x same week",
         ["apt", "stand_raw", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt", "h2", "week"]),
    ]
    rows = [dict(match=name, **pooled_sigma(b, keys)) for name, keys in ladder]
    W(C.md_table(pd.DataFrame(rows).set_index("match"), floatfmt="{:.1f}"))
    W("")
    W("## The two tightest rungs, per airport\n")
    tight = ["apt", "stand_raw", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt", "h2", "gap_bin", "q_bin"]
    rows = []
    for a, g in b.groupby("apt", observed=True):
        r = {"airport": a, "bulk rows": len(g)}
        r |= {f"sigma, stand x runway x operator x 2h":
              pooled_sigma(g, ["stand_raw", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt", "h2"])["sigma"]}
        t = pooled_sigma(g, tight[1:])
        r |= {"sigma, + gap and queue bins": t["sigma"], "rows in a group of 2+": t["rows"],
              "share %": t["share"]}
        rows.append(r)
    W(C.md_table(pd.DataFrame(rows).set_index("airport"), floatfmt="{:.1f}"))
    W("")
    C.write("q1a_pairs.md", "\n".join(out))


if __name__ == "__main__":
    main()
