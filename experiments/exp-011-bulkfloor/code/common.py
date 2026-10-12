"""Shared loading for exp-010: the NM-joined tail."""
from pathlib import Path
import numpy as np, pandas as pd
import prc.data as d

FEAT = d.ROOT / "artifacts" / "exp-003-anatomy" / "feat"
ANA = d.ROOT / "experiments" / "exp-011-bulkfloor" / "analysis"
ANA.mkdir(parents=True, exist_ok=True)
VALID_MONTH = {"july": "2025-07", "july_loo": "2025-07",
               "december": "2025-12", "january": "2025-01"}

BASE_COLS = ["MVT_ID_mvt", "apt", "stand_raw", "stand_pfx", "RUNWAY_mvt", "hour",
             "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt", "MARKET_SEGMENT_flt",
             "TAXITIME_SEC_mvt", "has_nm", "noac",
             "nm_taxi", "mvt_lobt", "mvt_sched", "aobt_lobt", "lobt_sched",
             "aobt_sched", "lobt_eobt1", "mvt_eobt1", "taxi_ub_pa", "taxi_lb_na",
             "taxi_ub_pm", "arr_turnaround_pa", "n_arr_stand_month",
             "MVT_TIME_UTC_mvt", "AOBT_3_flt", "BLOCK_TIME_UTC_mvt"]


def month(m: str, columns=None) -> pd.DataFrame:
    return pd.read_parquet(FEAT / f"{m}.parquet", columns=columns or BASE_COLS)


def all_2025(columns=None) -> pd.DataFrame:
    return pd.concat([month(m, columns) for m in d.MONTHS], ignore_index=True)


def add_regime(f: pd.DataFrame) -> pd.DataFrame:
    """Anchor candidates and the oracle regime label.

    `phi = (BLOCK - AOBT) / (LOBT - AOBT)`: 0 is block-on-AOBT, 1 is
    block-on-LOBT. `gap = LOBT - AOBT` is observable; phi is not.
    """
    f = f.copy()
    f["gap"] = -f["aobt_lobt"]                      # LOBT - AOBT
    f["pred_aobt"] = f["nm_taxi"]                   # taxi if block sits on AOBT
    f["pred_lobt"] = f["mvt_lobt"]                  # taxi if block sits on LOBT
    y = f["TAXITIME_SEC_mvt"].astype(float)
    f["err_aobt"] = f["pred_aobt"] - y
    f["err_lobt"] = f["pred_lobt"] - y
    f["phi"] = np.where(f["gap"].abs() > 1e-9,
                        (y - f["nm_taxi"]) / (-f["gap"]) * -1.0, np.nan)
    # taxi = nm_taxi - phi*gap  =>  phi = (nm_taxi - taxi) / gap
    f["phi"] = np.where(f["gap"].abs() > 1e-9, (f["nm_taxi"] - y) / f["gap"], np.nan)
    f["anchor_true"] = np.where(f["err_lobt"].abs() < f["err_aobt"].abs(), "lobt", "aobt")
    f["oracle_pred"] = np.where(f["anchor_true"] == "lobt", f["pred_lobt"], f["pred_aobt"])
    return f


def md_table(df: pd.DataFrame, floatfmt="{:.1f}") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    head = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |"
            for i, r in zip(df.index, df.to_numpy())]
    return "\n".join([head, sep] + rows)


def write(name: str, text: str):
    p = ANA / name
    p.write_text(text)
    print("wrote", p, flush=True)
