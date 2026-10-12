"""Cache the stand re-occupation signals per month (and for the ranking file)."""
import sys
sys.path.insert(0, "experiments/exp-010-nmtail/code")
import pandas as pd
import prc.data as d
from prc.nmtail import reoccupation_signals

OUT = d.ROOT / "artifacts" / "exp-010-nmtail"
OUT.mkdir(parents=True, exist_ok=True)


def one(frame: pd.DataFrame) -> pd.DataFrame:
    masked, _ = d.mask_like_ranking(frame)
    R = reoccupation_signals(masked)
    R = R.reset_index(drop=True)
    R["MVT_ID_mvt"] = masked.loc[masked["PHASE_mvt"] == "DEP", "MVT_ID_mvt"].to_numpy()
    return R


def main():
    for m in d.MONTHS:
        p = OUT / f"reocc_{m}.parquet"
        if p.exists():
            print("skip", m, flush=True); continue
        one(d.load_month(m)).to_parquet(p, index=False)
        print("built", m, flush=True)
    p = OUT / "reocc_ranking.parquet"
    if not p.exists():
        rk = d.load_ranking()
        parts = [one(g) for _, g in rk.groupby(rk["month"], observed=True)]
        pd.concat(parts, ignore_index=True).to_parquet(p, index=False)
        print("built ranking from", len(parts), "monthly frames", flush=True)


if __name__ == "__main__":
    main()
