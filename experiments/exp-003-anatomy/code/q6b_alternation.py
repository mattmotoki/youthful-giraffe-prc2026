"""Q6 addendum: does a stand really hold one aircraft at a time, and is clipping worth it?"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "experiments/exp-003-anatomy/code")
import common as C
import prc.data as d
import prc.stand_features as sf
from features import ART

MONTHS = ["2025-01", "2025-07", "2025-12"]


def alternation(month):
    df = sf.add_stand_keys(d.load_month(month))
    df = df[df.BLOCK_TIME_UTC_mvt.notna()]
    rows = []
    for key in ["key_raw", "key_base"]:
        per = []
        for k, idx in df.groupby(key, observed=True).indices.items():
            sub = df.iloc[idx]
            isd = (sub.PHASE_mvt == "DEP").to_numpy()
            a = np.sort(sub.loc[~isd, "BLOCK_TIME_UTC_mvt"].dt.as_unit("ns").astype("int64").to_numpy())
            dp = np.sort(sub.loc[isd, "BLOCK_TIME_UTC_mvt"].dt.as_unit("ns").astype("int64").to_numpy())
            if len(a) < 2 or len(dp) == 0:
                continue
            per.append(np.searchsorted(dp, a[1:], "left") - np.searchsorted(dp, a[:-1], "right"))
        c = np.concatenate(per)
        rows.append({"month": month, "stand key": key, "arrival intervals": len(c),
                     "exactly 1 departure %": 100 * float((c == 1).mean()),
                     "0 departures %": 100 * float((c == 0).mean()),
                     "2+ departures %": 100 * float((c >= 2).mean())})
    return rows


def main():
    out = []; W = out.append
    W("\n## Does the stand really hold one aircraft at a time?\n")
    W("The bound above assumes occupancy alternates arrival, departure, arrival. Tested "
      "directly with the true block times: for every pair of consecutive arrivals at a "
      "stand, how many departure off-blocks fall between them?\n")
    rows = []
    for m in MONTHS:
        rows += alternation(m)
    W(C.md_table(pd.DataFrame(rows).set_index("month"), floatfmt="{:.2f}"))
    W("")
    W("**It alternates about 88 % of the time, not 100 %.** In 6 % of the gaps between two "
      "arrivals no aircraft departs (the aircraft is towed away, or stays overnight) and in "
      "5 % two or more depart (the stand name covers more than one physical position, or an "
      "aircraft is towed *on*). An 11 % failure rate rules out the counting construction as a "
      "hard clip: a bound built by counting arrivals between two departures is valid on only "
      "74 % of the rows it fires on (measured, `disp_lb_1`, July 2025), and its second-order "
      "version on 0.5 %. It is kept in `prc.stand_features.displacement_bounds` as a "
      "*feature*, never as a clip.\n")

    W("## Clipping: the occupancy bound against clipping at zero\n")
    W("Reference naive predictor, bulk RMSE (NM join, taxi <= 3600 s):\n")
    rows = []
    for fold, trm, vam in [("july", [f"2025-{m:02d}" for m in range(1, 7)], "2025-07"),
                           ("december", [f"2025-{m:02d}" for m in range(1, 12)], "2025-12"),
                           ("january", [f"2025-{m:02d}" for m in range(2, 13)], "2025-01")]:
        tr = pd.concat([pd.read_parquet(ART / f"{m}.parquet") for m in trm], ignore_index=True)
        va = pd.read_parquet(ART / f"{vam}.parquet")
        for f in (tr, va):
            f["off"] = (f.BLOCK_TIME_UTC_mvt - f.AOBT_3_flt).dt.total_seconds()
            f["y"] = f.TAXITIME_SEC_mvt.astype(float)
        mt = tr.has_nm & tr.y.le(3600)
        g = tr[mt].groupby("key_raw", observed=True)["off"].agg(["median", "size"])
        g = g[g["size"] >= 30]["median"]
        gm = tr[mt].groupby("apt", observed=True)["off"].median()
        off = va.key_raw.map(g).fillna(va.apt.map(gm)).fillna(tr.loc[mt, "off"].median())
        pred = (va.nm_taxi - off).to_numpy()
        ymed = tr[tr.y.notna()].groupby("apt", observed=True)["y"].median()
        pred = np.where(va.has_nm.to_numpy(), pred, va.apt.map(ymed).fillna(tr.y.median()).to_numpy())
        y = va.y.to_numpy(); bulk = (va.has_nm & va.y.le(3600)).to_numpy()
        lbnm, lbna = va["taxi_lb_nm"].to_numpy(), va["taxi_lb_na"].to_numpy()
        p0 = np.maximum(pred, 0)
        variants = {
            "no clip": pred,
            "clip at 0": p0,
            "clip at the next-arrival-after-take-off bound": np.where(np.isfinite(lbnm) & (pred < lbnm), lbnm, pred),
            "clip at 0, then at the next-arrival-after-AOBT bound":
                np.where(np.isfinite(lbna) & (p0 < lbna), lbna, p0),
        }
        r = {"fold": fold, "negative raw predictions": int((pred < 0).sum())}
        for k, p in variants.items():
            r[k] = d.rmse(y[bulk], p[bulk])
        rows.append(r)
    W(C.md_table(pd.DataFrame(rows).set_index("fold"), floatfmt="{:.2f}"))
    W("")
    W("Clipping at zero recovers 24 s of bulk RMSE on the july fold; the occupancy bound on "
      "top of that recovers a further 0.3 s. As a *clip* the bound is worth nothing that "
      "`max(pred, 0)` does not already give. Its value is as a **feature**, which the "
      "ablation table measures.\n")
    p = C.ANA / "q6_bounds.md"
    p.write_text(p.read_text() + "\n".join(out))
    print("appended to", p)


if __name__ == "__main__":
    main()
