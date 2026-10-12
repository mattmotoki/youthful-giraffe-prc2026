"""The January 2026 flag and exceedance per RANKING ROW, from committed files.

The assignment asks whether `experiments/exp-007-winter/work/ranking_daystate.parquet`
carries what the envelope needs.  It carries half of it: `apt` and `day` for
every one of the 344,841 ranking rows, but no daily span mean and no
`MVT_ID_mvt`.  The other half is `experiments/exp-014-robustness/analysis/
q2_days_2026.csv`, which carries the daily span mean and the exceedance for all
620 ranking airport-days.  The join between them is POSITIONAL, and that is the
one assumption in this file, so it is checked three ways and the run aborts if
any check fails:

  1. `ds_n_dep` (the day-state block's own row count) equals the row count of
     each airport-day group -- exact on all 620 days.
  2. `q2_days_2026.csv`'s `n` equals the same -- exact on all 620 days.
  3. `has_nm` from `N1-nmtail-v4-ranking.parquet`, averaged inside each
     airport-day group under the positional join, reproduces `ds_nm_rate`.
     A shuffled control moves that discrepancy from 0.003 to 0.399, so the
     check has power.

`ranking_daystate.parquet` was written by `exp-007/code/q3b_ranking_check.py` as
`pd.concat` over month-sorted blocks with the ranking frame's index preserved,
so sorting by that index restores `prc.data.load_ranking()`'s row order, which
is the order every `*-ranking.parquet` member file is written in.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import envelope as E  # noqa: E402
import prc.data as D  # noqa: E402

DS = ROOT / "experiments/exp-007-winter/work/ranking_daystate.parquet"
IDSRC = ROOT / "experiments/exp-012-blend2/published/N1-nmtail-v4-ranking.parquet"
CHECK = ROOT / "experiments/exp-012-blend2/published/C2-cat-ranking.parquet"
OUT = EXP / "oof" / "ranking_flag.parquet"


def build(strict: bool = True) -> pd.DataFrame:
    ds = pd.read_parquet(DS, columns=["month", "apt", "day", "ds_nm_rate", "ds_n_dep"])
    if not ds.index.is_monotonic_increasing:
        ds = ds.sort_index()
    ds = ds.reset_index(drop=True)
    ids = pd.read_parquet(IDSRC, columns=["MVT_ID_mvt", "has_nm"])
    chk = pd.read_parquet(CHECK, columns=["MVT_ID_mvt"])
    if not np.array_equal(ids["MVT_ID_mvt"].to_numpy(), chk["MVT_ID_mvt"].to_numpy()):
        raise SystemExit("the two ranking member files are not in the same row order")
    if len(ds) != len(ids):
        raise SystemExit(f"{len(ds)} day-state rows against {len(ids)} member rows")
    a = ds.assign(MVT_ID_mvt=ids["MVT_ID_mvt"].to_numpy("float64"),
                  has_nm=ids["has_nm"].to_numpy())
    g = a.groupby(["apt", "day"], observed=True).agg(
        n=("MVT_ID_mvt", "size"), dsn=("ds_n_dep", "first"),
        rate=("has_nm", "mean"), dsr=("ds_nm_rate", "first")).reset_index()
    k = E.ranking_days()
    j = g.merge(k, on=["apt", "day"], how="outer", validate="one_to_one", indicator=True)
    j = j.rename(columns={"n_x": "rows_here", "n_y": "rows_q2"})
    checks = {
        "airport_days": int(len(j)),
        "unmatched": int((j["_merge"] != "both").sum()),
        "max_abs_n_minus_ds_n_dep": float((j["rows_here"] - j["dsn"]).abs().max()),
        "max_abs_n_minus_q2_n": float((j["rows_here"] - j["rows_q2"]).abs().max()),
        "max_abs_has_nm_minus_ds_nm_rate": float((j["rate"] - j["dsr"]).abs().max()),
    }
    rng = np.random.default_rng(0)
    b = a.assign(has_nm=rng.permutation(a["has_nm"].to_numpy()))
    gb = b.groupby(["apt", "day"], observed=True).agg(rate=("has_nm", "mean"),
                                                      dsr=("ds_nm_rate", "first"))
    checks["shuffled_control"] = float((gb["rate"] - gb["dsr"]).abs().max())
    print("[align] " + json.dumps(checks), flush=True)
    if strict and (checks["unmatched"] or checks["max_abs_n_minus_ds_n_dep"]
                   or checks["max_abs_n_minus_q2_n"]
                   or checks["max_abs_has_nm_minus_ds_nm_rate"] > 0.01):
        raise SystemExit("the positional join does not check out; refusing to write a flag")

    out = a[["MVT_ID_mvt", "month", "apt", "day"]].merge(
        k[["apt", "day", "day_span_mean", "envelope", "exceedance"]],
        on=["apt", "day"], how="left", validate="many_to_one")
    out["flagged"] = out["exceedance"] > 0
    if out["exceedance"].isna().any():
        raise SystemExit("some ranking rows have no airport-day exceedance")
    return out, checks


def main() -> None:
    out, checks = build()
    out.to_parquet(OUT, index=False)
    print(f"[write] {OUT.relative_to(ROOT)}  {len(out):,} rows  "
          f"sha256 {D.sha256(OUT)}", flush=True)
    jan = out[out["month"] == "2026-01"]
    par = json.loads((EXP / "analysis" / "arm_params.json").read_text())
    served = par["step|lofo|winsor"]["('january', None, None)"]
    x = jan["exceedance"].to_numpy("float64")
    c = np.where(x <= 0, 0.0, np.where(x <= E.CLEAR, served["lo"], served["hi"]))
    t = (jan.assign(c=c).groupby(["apt", "day"], observed=True)
         .agg(rows=("MVT_ID_mvt", "size"), exceedance=("exceedance", "first"),
              c=("c", "first")).reset_index())
    t = t[t["exceedance"] > 0].sort_values("exceedance", ascending=False)
    t["total_s"] = t["rows"] * t["c"]
    t.to_csv(EXP / "analysis" / "ranking_touched.csv", index=False)
    print("\n=== January 2026 airport-days the correction would touch "
          "(step|lofo|winsor, the january rule) ===")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.1f}"), flush=True)
    print(f"\ntouched rows {int(t['rows'].sum()):,} of {len(jan):,} "
          f"({t['rows'].sum()/len(jan):.2%}) across {len(t)} airport-days; "
          f"total added prediction {t['total_s'].sum():,.0f} s, "
          f"mean over the whole month {t['total_s'].sum()/len(jan):+.2f} s/row")
    K = E.ranking_days()
    jul = int((K.loc[K["month"] == "2026-07", "exceedance"] > 0).sum())
    print(f"July 2026 flagged airport-days: {jul}. The correction is not "
          f"January-only by construction -- it fires wherever the exceedance is "
          f"positive -- so July's flagged rows move too and are reported with it.")


if __name__ == "__main__":
    main()
