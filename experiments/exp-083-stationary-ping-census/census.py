#!/usr/bin/env python3
"""Registered, label-free exp-083 census. It writes aggregates only."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
DENOMINATOR = 344_841
BOARD_RMSE = 245.0207
TOLERANCE_S = 60.0
SEED = 8_302_026
QUANTILES = [0, .25, .5, .75, .9, .95, .99, 1]
REFUSED = re.compile(r"block_time_utc_mvt|target|label|truth|residual|fold", re.I)
ALLOWED = {
    "ranking": ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt",
                "FLIGHT_mvt", "STAND_mvt", "RUNWAY_mvt"],
    "summary": ["icao", "callsign_modal", "callsigns_all", "kind", "seg_start", "seg_end",
                "first_motion", "obs_offblock", "ends_stationary", "stand_lat", "stand_lon",
                "n_ground_points"],
    "class": ["MVT_ID_mvt", "obs_has_clock", "noac"],
    "prediction": ["MVT_ID_mvt", "TAXITIME_SEC_mvt"],
}


def validate_columns(source: str, requested: list[str], returned: list[str] | None = None) -> None:
    if source not in ALLOWED or len(requested) != len(set(requested)):
        raise ValueError("unknown source or duplicate projection")
    if set(requested) - set(ALLOWED[source]):
        raise ValueError(f"column outside {source} allow-list")
    for column in requested:
        if REFUSED.search(column) or (column.lower() == "taxitime_sec_mvt" and source != "prediction"):
            raise ValueError(f"refused column: {column}")
    if returned is not None and set(returned) != set(requested):
        raise ValueError("loaded columns differ from the exact projection")


def read_projected(path: Path, source: str) -> pd.DataFrame:
    columns = ALLOWED[source]
    validate_columns(source, columns)
    frame = pd.read_parquet(path, columns=columns)
    validate_columns(source, columns, frame.columns.tolist())
    return frame


def normalize_callsign(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).upper()
    return "".join(char for char in value if "A" <= char <= "Z" or "0" <= char <= "9")


def utc_datetime(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        return pd.to_datetime(values, utc=True)
    return pd.to_datetime(values, unit="s", utc=True)


def epoch_seconds(values: pd.Series) -> pd.Series:
    times = utc_datetime(values)
    return times.astype("int64") / 1_000_000_000


def eligible_segments(frame: pd.DataFrame) -> pd.DataFrame:
    validate_columns("summary", ALLOWED["summary"], frame.columns.tolist())
    out = frame.copy()
    out["_modal"] = out["callsign_modal"].map(normalize_callsign)
    out["_tokens"] = out["callsigns_all"].map(
        lambda x: {normalize_callsign(v) for v in str(x).split("|") if normalize_callsign(v)}
        if pd.notna(x) else set())
    unambiguous = out.apply(lambda row: len(row["_tokens"]) == 1 and row["_modal"] in row["_tokens"], axis=1)
    finite = np.isfinite(pd.to_numeric(out["seg_start"], errors="coerce"))
    finite &= np.isfinite(pd.to_numeric(out["seg_end"], errors="coerce"))
    finite &= np.isfinite(pd.to_numeric(out["stand_lat"], errors="coerce"))
    finite &= np.isfinite(pd.to_numeric(out["stand_lon"], errors="coerce"))
    mask = (unambiguous & out["kind"].eq("other") & out["ends_stationary"].eq(True)
            & out["first_motion"].isna() & out["obs_offblock"].isna() & finite
            & (pd.to_numeric(out["seg_start"]) <= pd.to_numeric(out["seg_end"]))
            & (pd.to_numeric(out["n_ground_points"], errors="coerce") >= 1))
    out = out.loc[mask, ["icao", "_modal", "seg_end"]].copy()
    out["_segment_token"] = np.arange(len(out), dtype=np.int64)
    return out


def shuffled_callsigns(rows: pd.DataFrame) -> pd.Series:
    result = rows["_callsign"].copy()
    rng = np.random.default_rng(SEED)
    dates = utc_datetime(rows["MVT_TIME_UTC_mvt"]).dt.strftime("%Y-%m-%d")
    for _, idx in rows.groupby([rows["ADEP_mvt"], dates], sort=True).groups.items():
        idx = np.asarray(list(idx))
        order = sorted(idx, key=lambda i: hashlib.sha256(str(rows.at[i, "MVT_ID_mvt"]).encode()).digest())
        values = rows.loc[order, "_callsign"].to_numpy(copy=True)
        result.loc[order] = values[rng.permutation(len(values))]
    return result


def match(rows: pd.DataFrame, segments: pd.DataFrame, low_s: float, high_s: float,
          callsigns: pd.Series | None = None) -> pd.DataFrame:
    left = rows.copy()
    left["_match_callsign"] = left["_callsign"] if callsigns is None else callsigns
    pairs = left.merge(segments, left_on=["ADEP_mvt", "_match_callsign"],
                       right_on=["icao", "_modal"], how="inner")
    pairs["_delta"] = epoch_seconds(pairs["MVT_TIME_UTC_mvt"]) - pairs["seg_end"]
    pairs = pairs[pairs["_delta"].between(low_s, high_s, inclusive="both")]
    row_counts = pairs.groupby("_row_token")["_segment_token"].transform("size")
    seg_counts = pairs.groupby("_segment_token")["_row_token"].transform("size")
    pairs = pairs[(row_counts == 1) & (seg_counts == 1)].copy()
    pairs["U"] = pairs["_delta"] + TOLERANCE_S
    return pairs


def metric_block(accepted: pd.DataFrame) -> dict:
    if len(accepted):
        q = accepted["U"].quantile(QUANTILES)
        quantiles = {str(level): float(q.loc[level]) for level in QUANTILES}
    else:
        quantiles = {str(level): None for level in QUANTILES}
    margins = {}
    for margin in (60, 300):
        excess = np.maximum(accepted["prediction"].to_numpy() - accepted["U"].to_numpy() - margin, 0)
        gain = float(np.square(excess).sum() / DENOMINATOR)
        margins[str(margin)] = {"above_count": int((excess > 0).sum()), "G_lb_s2": gain,
                                "approx_board_seconds": gain / (2 * BOARD_RMSE)}
    return {"raw_support": int(len(accepted)), "accepted_rows": int(len(accepted)),
            "U_quantiles_s": quantiles, "margins_s": margins}


def aggregate(rows: pd.DataFrame, accepted: pd.DataFrame) -> dict:
    cells = {}
    groups = [("TOTAL", rows)]
    groups += [(f"{airport}|{month}", group) for (airport, month), group in
               rows.groupby(["ADEP_mvt", "month"], sort=True)]
    for name, population in groups:
        tokens = set(population["_row_token"])
        selected = accepted[accepted["_row_token"].isin(tokens)]
        cells[name] = {"complement_rows": int(len(population)), **metric_block(selected)}
    return cells


def verify_manifest(manifest: dict) -> None:
    records = manifest["fixed_sources"] + manifest["summary_collection"]["files"]
    for record in records:
        path = ROOT / record["path"]
        if path.stat().st_size != record["size_bytes"]:
            raise RuntimeError("registered source size changed")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise RuntimeError("registered source hash changed")


def run() -> dict:
    manifest = json.loads((EXP / "source-manifest.json").read_text())
    verify_manifest(manifest)
    ranking = read_projected(ROOT / "data/raw/ranking.parquet", "ranking")
    ranking = ranking[utc_datetime(ranking["MVT_TIME_UTC_mvt"]).dt.strftime("%Y-%m").isin(["2026-01", "2026-07"])]
    classes = pd.concat([read_projected(ROOT / f"artifacts/exp-046/class/2026-{month}-v22.parquet", "class")
                         for month in ("01", "07")], ignore_index=True)
    rows = ranking.merge(classes, on="MVT_ID_mvt", how="inner", validate="one_to_one")
    rows = rows[~rows["obs_has_clock"].astype(bool)].copy()
    predictions = read_projected(ROOT / "submissions/uploaded/youthful-giraffe_v22.parquet", "prediction")
    predictions = predictions.rename(columns={"TAXITIME_SEC_mvt": "prediction"})
    rows = rows.merge(predictions, on="MVT_ID_mvt", how="inner", validate="one_to_one")
    rows["_callsign"] = rows["CALLSIGN_flt"].map(normalize_callsign)
    rows["_row_token"] = np.arange(len(rows), dtype=np.int64)
    rows["month"] = utc_datetime(rows["MVT_TIME_UTC_mvt"]).dt.strftime("%Y-%m")
    if len(rows) != 148585:
        raise RuntimeError("registered complement count mismatch")
    summary_paths = [ROOT / record["path"] for record in manifest["summary_collection"]["files"]]
    summaries = pd.concat([read_projected(path, "summary") for path in summary_paths], ignore_index=True)
    segments = eligible_segments(summaries)
    treatments = {
        "main": match(rows, segments, 60, 3 * 3600),
        "shifted": match(rows, segments, 24 * 3600, 27 * 3600),
        "shuffled": match(rows, segments, 60, 3 * 3600, shuffled_callsigns(rows)),
    }
    output = {"experiment": "exp-083-stationary-ping-census", "denominator": DENOMINATOR,
              "registered_complement_rows": int((~classes["obs_has_clock"].astype(bool)).sum()),
              "matched_population_rows": int(len(rows)),
              "excluded_blank_callsign_rows": int(rows["_callsign"].eq("").sum()),
              "treatments": {name: aggregate(rows, accepted) for name, accepted in treatments.items()}}
    main = output["treatments"]["main"]["TOTAL"]["raw_support"]
    shifted = output["treatments"]["shifted"]["TOTAL"]["raw_support"]
    shuffled = output["treatments"]["shuffled"]["TOTAL"]["raw_support"]
    net = main - max(shifted, shuffled)
    output["net_support"] = {"main_minus_shifted": main - shifted,
                             "main_minus_shuffled": main - shuffled, "conservative": net}
    gain = output["treatments"]["main"]["TOTAL"]["margins_s"]["300"]["G_lb_s2"]
    output["gate"] = "ADVANCE" if net >= 1486 and gain >= 4.9 else "FAIL" if net < 298 or gain < .98 else "MAYBE"
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate": result["gate"], "net_support": result["net_support"]["conservative"],
                      "G_lb_300_s2": result["treatments"]["main"]["TOTAL"]["margins_s"]["300"]["G_lb_s2"]}))


if __name__ == "__main__":
    main()
