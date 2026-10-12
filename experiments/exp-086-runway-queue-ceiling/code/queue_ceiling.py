"""Label-blind census and frozen Phase-A primitives for exp-086 Phase A."""
from __future__ import annotations
import argparse, hashlib, json, os, re
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "analysis"
FORBIDDEN = frozenset({"BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt", "y_true", "y_pred"})
INPUT_COLUMNS = ("MVT_ID_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt",
    "PHASE_mvt", "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt",
    "LOBT_flt", "ADEP_flt", "ADES_flt", "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt", "IOBT_flt", "EOBT_1_flt")
PRIOR_MIN_SUPPORT = 20
PLAN_OFFSET_CLIP = (-7200.0, 14400.0)
UNIMPEDED_CLIP = (60.0, 1800.0)
SERVICE_CLIP = (20.0, 180.0)
EXCLUSION_PREFIX = "prc-exp068-excluded-v1:"
EXCLUSION_HASHES = frozenset({
    "f9969cc559d281ccda90bcb5532645d235e0ba3ff0a9915549ee9a86b31dade8",
    "1265dc65d6f5d4a5fd822029695938fa8e9c318056ea4e05edf7dacee1499798",
})
CONTROL_SEEDS = {"runway_permutation": 860217, "day_bootstrap": 860219}
FROZEN_CENSUS = {"rows": 854937, "cells": 182392, "supported_cells": 16196,
                 "supported_departures": 185581}
OUTPUT_COLUMNS = ("arm", "fold", "slice", "airport", "rows", "supported_rows",
                  "rmse_k2", "rmse_arm", "gain_seconds", "ci95_low", "ci95_high")
REGISTRATION_COMMIT = "3e802018"
REVIEW_SHA256 = "1c6c6de57dbc5988619dca232d8cdb2b2aacec8ca590da2e500ed4aa8c86da03"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def canonical_movement_id(value) -> str:
    if isinstance(value, (bool, np.bool_)): raise ValueError("invalid movement ID")
    if isinstance(value, (int, np.integer)):
        n = int(value)
    elif isinstance(value, (float, np.floating)):
        x = float(value)
        if not np.isfinite(x) or x < 0 or x > 2**53-1 or not x.is_integer(): raise ValueError("invalid movement ID")
        n = int(x)
    elif isinstance(value, str):
        if not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.0+)?", value): raise ValueError("invalid movement ID")
        n = int(value.split(".", 1)[0])
    else: raise ValueError("invalid movement ID")
    if n < 0: raise ValueError("invalid movement ID")
    return str(n)

def exclusion_digest(movement_id) -> str:
    return hashlib.sha256((EXCLUSION_PREFIX + canonical_movement_id(movement_id)).encode("utf-8")).hexdigest()

def exclusion_mask(values) -> np.ndarray:
    return np.fromiter((exclusion_digest(v) in EXCLUSION_HASHES for v in values), dtype=bool)

def canonical_frame_hash(frame: pd.DataFrame) -> str:
    """Hash names, dtypes, row order, nulls, and values of a permitted projection."""
    h = hashlib.sha256()
    schema = [(str(c), str(frame[c].dtype)) for c in frame.columns]
    h.update(json.dumps(schema, separators=(",", ":"), ensure_ascii=False).encode())
    h.update(pd.util.hash_pandas_object(frame, index=True, categorize=False).to_numpy("uint64").tobytes())
    return h.hexdigest()

def phase_a_mask(frame: pd.DataFrame) -> pd.DataFrame:
    remove = [c for c in frame if c in FORBIDDEN or c.startswith("label_derived__")]
    out = frame.drop(columns=remove, errors="ignore").copy()
    assert_no_label_lineage(out.columns)
    return out

def assert_no_label_lineage(columns) -> None:
    bad = {c for c in columns if c in FORBIDDEN or c.startswith("label_derived__")}
    if bad: raise AssertionError(f"Phase A contains label-side columns: {sorted(bad)}")

def planned_departures(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    d = phase_a_mask(raw).loc[lambda x: x["PHASE_mvt"].eq("DEP")].copy()
    for c in ("SCHED_TIME_UTC_mvt", "LOBT_flt", "EOBT_1_flt", "IOBT_flt"):
        d[c] = pd.to_datetime(d[c], utc=True, errors="coerce")
    d["plan_time"] = d["LOBT_flt"].combine_first(d["EOBT_1_flt"]).combine_first(d["IOBT_flt"]).combine_first(d["SCHED_TIME_UTC_mvt"])
    d["plan_source"] = np.select([d["LOBT_flt"].notna(), d["EOBT_1_flt"].notna(), d["IOBT_flt"].notna()], ["LOBT", "EOBT_1", "IOBT"], default="SCHED")
    delta = (d["plan_time"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    d["plan_missing"] = d["plan_time"].isna()
    d["plan_impossible"] = delta.abs().gt(86400)
    d["plan_late"] = delta.gt(7200) & ~d["plan_impossible"]
    d["nm_join_missing"] = d["FLIGHT_ID_mvt"].isna() | d["LOBT_flt"].isna()
    d["core_missing"] = d[["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt"]].isna().any(axis=1)
    d["slot_15m"] = d["plan_time"].dt.floor("15min")
    d["source"] = source
    return d

def add_stability_proxies(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy(); keys = ["ADEP_mvt", "AIRCRAFT_OPERATOR_flt", "FLIGHT_mvt", "ADES_mvt"]
    valid = out[keys].notna().all(axis=1)
    out["service_support"] = 0; out["runway_modal_share"] = np.nan; out["stand_modal_share"] = np.nan
    if valid.any():
        support = out.loc[valid].groupby(keys, dropna=False, sort=False)["MVT_ID_mvt"].transform("size")
        for value, result in (("RUNWAY_mvt", "runway_modal_share"), ("STAND_mvt", "stand_modal_share")):
            freq = out.loc[valid].groupby(keys + [value], dropna=False)["MVT_ID_mvt"].transform("size")
            peak = freq.groupby([out.loc[valid, c] for c in keys], dropna=False).transform("max")
            out.loc[valid, result] = (peak / support).to_numpy()
        out.loc[valid, "service_support"] = support.to_numpy()
    return out

def slot_census(d: pd.DataFrame) -> pd.DataFrame:
    keys = ["source", "ADEP_mvt", "RUNWAY_mvt", "slot_15m"]
    return (d.groupby(keys, dropna=False, sort=True).agg(planned_demand=("MVT_ID_mvt", "size"),
        nm_join_missing=("nm_join_missing", "sum"), plan_missing=("plan_missing", "sum"),
        plan_impossible=("plan_impossible", "sum"), plan_late=("plan_late", "sum"),
        core_complete=("core_missing", lambda x: int((~x).sum())), median_service_support=("service_support", "median"),
        runway_modal_share=("runway_modal_share", "mean"), stand_modal_share=("stand_modal_share", "mean")).reset_index()
        .assign(supported=lambda x: x.planned_demand.ge(10), coverage=lambda x: x.core_complete / x.planned_demand))

def self_excluded_slot_demand(d: pd.DataFrame) -> pd.Series:
    keys = ["source", "ADEP_mvt", "RUNWAY_mvt", "slot_15m"]
    return d.groupby(keys, dropna=False)["MVT_ID_mvt"].transform("size").sub(1).astype("int32")

def census_outputs(named_frames: list[tuple[str, pd.DataFrame]]) -> dict[str, bytes]:
    """Build every census output so one mutation test covers all output bytes."""
    parts, sources = [], []
    for name, raw in named_frames:
        projection = raw.loc[:, list(INPUT_COLUMNS)].copy()
        d = add_stability_proxies(planned_departures(projection, name)); parts.append(d)
        sources.append({"name": name, "projection_sha256": canonical_frame_hash(projection),
            "projection_columns": list(INPUT_COLUMNS), "rows": len(projection), "departure_rows": len(d)})
    all_dep = pd.concat(parts, ignore_index=True); cells = slot_census(all_dep)
    by_source = all_dep.groupby("source", sort=True).agg(rows=("MVT_ID_mvt", "size"), nm_join_missing=("nm_join_missing", "sum"), plan_missing=("plan_missing", "sum"), plan_impossible=("plan_impossible", "sum"), plan_late=("plan_late", "sum"), core_missing=("core_missing", "sum"), median_service_support=("service_support", "median"), mean_runway_modal_share=("runway_modal_share", "mean"), mean_stand_modal_share=("stand_modal_share", "mean")).reset_index()
    by_airport = all_dep.groupby(["source", "ADEP_mvt"], dropna=False, sort=True).agg(rows=("MVT_ID_mvt", "size"), nm_join_missing=("nm_join_missing", "sum"), plan_impossible=("plan_impossible", "sum"), plan_late=("plan_late", "sum"), core_missing=("core_missing", "sum"), mean_runway_modal_share=("runway_modal_share", "mean"), mean_stand_modal_share=("stand_modal_share", "mean")).reset_index()
    outputs = {"slot-census.csv": cells.to_csv(index=False).encode(), "source-summary.csv": by_source.to_csv(index=False).encode(), "airport-summary.csv": by_airport.to_csv(index=False).encode()}
    supported_departures = int(cells.loc[cells.supported, "planned_demand"].sum())
    payload = {"schema_version": 3, "hash_kind": "canonical-permitted-input-projection-v1", "slot_minutes": 15,
        "support_threshold": 10, "sources": sources, "rows": int(len(all_dep)), "cells": int(len(cells)),
        "supported_cells": int(cells.supported.sum()), "supported_departures": supported_departures,
        "outputs": {k: "analysis/" + k for k in outputs}}
    outputs["census.json"] = (json.dumps(payload, indent=2) + "\n").encode()
    return outputs

def rendered_census(payload: dict) -> bytes:
    share = 100.0 * payload["supported_departures"] / payload["rows"]
    return ("# exp-086 Phase-A census\n\n"
        f"The label-free census exactly reproduces the frozen population: {payload['rows']:,} departures in "
        f"{payload['cells']:,} airport-runway-plan-slot cells. There are {payload['supported_cells']:,} supported "
        f"cells containing {payload['supported_departures']:,} departures, or {share:.2f}% of the population.\n\n"
        "Phase A reads the registered 18-column projection only. The mutation proof covers every registered "
        "Phase-A CSV, JSON, hash, summary, output map, manifest, and rendered census byte. No label, prediction, "
        "residual, exposed movement ID, queue result, or score was read or written.\n").encode()

def rendered_source_manifest(payload: dict, template: dict) -> bytes:
    m = json.loads(json.dumps(template)); m["projection_hashes"]={x["name"]:x["projection_sha256"] for x in payload["sources"]}
    m["frozen_census"]={k:payload[k] for k in FROZEN_CENSUS}; m["output_columns"]=list(OUTPUT_COLUMNS)
    return (json.dumps(m,indent=2)+"\n").encode()

def assert_frozen_census(payload: dict) -> None:
    if {k:payload.get(k) for k in FROZEN_CENSUS} != FROZEN_CENSUS:
        raise AssertionError("frozen census mismatch")

def resolve_prior(history: pd.DataFrame, query: pd.Series, target: str, levels: list[list[str]],
                  scored_month: str, min_support: int = PRIOR_MIN_SUPPORT,
                  clip: tuple[float, float] | None = None) -> tuple[float, str, int]:
    """Resolve a strictly earlier-month, self-excluded median through fixed levels."""
    h = history.loc[history["month"].astype(str).lt(scored_month) & history[target].notna()].copy()
    if "row_id" in query and "row_id" in h: h = h.loc[~h["row_id"].eq(query["row_id"])]
    for cols in levels + [[]]:
        g = h
        for c in cols: g = g.loc[g[c].eq(query[c])]
        if len(g) >= min_support:
            value = float(g[target].median())
            if clip is not None: value = float(np.clip(value, *clip))
            return value, ("global" if not cols else "x".join(cols)), len(g)
    raise ValueError("no supported earlier-month fallback")

def historical_plan_offsets(history: pd.DataFrame, exclude_row_id=None) -> pd.DataFrame:
    """Create the registered block-minus-plan target for later earlier-month fitting."""
    h = history.copy()
    if exclude_row_id is not None: h = h.loc[~h["row_id"].eq(exclude_row_id)].copy()
    h["block_time"] = pd.to_datetime(h["block_time"], utc=True)
    h["plan_time"] = pd.to_datetime(h["plan_time"], utc=True)
    h["plan_offset_seconds"] = (h["block_time"] - h["plan_time"]).dt.total_seconds()
    return h

def freeze_unimpeded_inputs(history: pd.DataFrame) -> pd.DataFrame:
    """Create the only permitted categorical and time inputs for the U prior."""
    h = history.copy()
    wake = h["WK_TBL_CAT_flt"] if "WK_TBL_CAT_flt" in h else pd.Series(pd.NA, index=h.index)
    aircraft = h["AIRCRAFT_TYPE_mvt"] if "AIRCRAFT_TYPE_mvt" in h else pd.Series(pd.NA, index=h.index)
    h["wake_type"] = wake.combine_first(aircraft).fillna("__MISSING__").astype(str)
    t = pd.to_datetime(h["plan_time"], utc=True)
    h["quarter_hour_of_day"] = (t.dt.hour * 4 + t.dt.minute // 15).astype("Int64")
    h["supplied_stand"] = h.get("STAND_mvt", h.get("supplied_stand", pd.Series(pd.NA, index=h.index))).fillna("__MISSING__").astype(str)
    return h

def historical_unimpeded_events(history: pd.DataFrame, exclude_row_id=None) -> pd.DataFrame:
    """Rebuild low-demand U targets after removing a query from raw history."""
    h = history.copy()
    if exclude_row_id is not None:
        h = h.loc[~h["row_id"].eq(exclude_row_id)].copy()
    h = freeze_unimpeded_inputs(h)
    h["block_time"] = pd.to_datetime(h["block_time"], utc=True)
    h["event_time"] = pd.to_datetime(h["event_time"], utc=True)
    h["utc_day"] = h["block_time"].dt.floor("D")
    h = h.sort_values(["airport", "runway", "utc_day", "block_time", "row_id"], kind="mergesort")
    group = h.groupby(["airport", "runway", "utc_day"], dropna=False, sort=False)["block_time"]
    before = (h["block_time"] - group.shift()).dt.total_seconds()
    after = (group.shift(-1) - h["block_time"]).dt.total_seconds()
    h["low_demand"] = (before.isna() | before.ge(600.0)) & (after.isna() | after.ge(600.0))
    h["unimpeded_seconds"] = (h["event_time"] - h["block_time"]).dt.total_seconds()
    return h.loc[h["low_demand"]].copy()

def historical_service_events(history: pd.DataFrame, exclude_row_id=None) -> pd.DataFrame:
    """Rebuild positive same-server/day intervals after removing a query."""
    h = history.copy()
    if exclude_row_id is not None:
        h = h.loc[~h["row_id"].eq(exclude_row_id)].copy()
    h["event_time"] = pd.to_datetime(h["event_time"], utc=True); h["utc_day"] = h["event_time"].dt.floor("D")
    h = h.sort_values(["airport", "runway", "utc_day", "event_time", "row_id"], kind="mergesort")
    h["service_interval_seconds"] = h.groupby(["airport", "runway", "utc_day"], dropna=False)["event_time"].diff().dt.total_seconds()
    return h.loc[h["service_interval_seconds"].gt(0)].copy()

def resolve_unimpeded_prior(history: pd.DataFrame, query: pd.Series, scored_month: str,
                             min_support: int = PRIOR_MIN_SUPPORT) -> tuple[float, str, int]:
    targets = historical_unimpeded_events(history, query.get("row_id"))
    levels = [["airport", "supplied_stand", "runway", "wake_type", "quarter_hour_of_day"],
              ["airport", "runway", "wake_type"], ["airport", "runway"], ["airport"]]
    return resolve_prior(targets, query, "unimpeded_seconds", levels, scored_month, min_support, UNIMPEDED_CLIP)

def resolve_service_prior(history: pd.DataFrame, query: pd.Series, scored_month: str,
                          min_support: int = PRIOR_MIN_SUPPORT) -> tuple[float, str, int]:
    targets = historical_service_events(history, query.get("row_id"))
    return resolve_prior(targets, query, "service_interval_seconds",
                         [["airport", "runway"], ["airport"]], scored_month, min_support, SERVICE_CLIP)

def stable_queue_order(frame: pd.DataFrame) -> pd.DataFrame:
    if frame["row_id"].duplicated().any(): raise ValueError("row_id must be unique")
    return frame.sort_values(["queue_arrival_time", "row_id"], kind="mergesort").reset_index(drop=True)

def oracle_order_only_arrivals(frame: pd.DataFrame) -> pd.Series:
    """Assign deployable arrival spacings to identities in true-block rank per unit."""
    required = {"row_id", "airport", "runway", "utc_day", "deployable_arrival_time", "true_block_time"}
    if required.difference(frame.columns): raise ValueError("oracle frame lacks required columns")
    if frame["runway"].isna().any(): raise ValueError("missing runway is unsupported")
    result = pd.Series(index=frame.index, dtype="float64")
    for _, g in frame.groupby(["airport", "runway", "utc_day"], dropna=False, sort=False):
        rank = g.sort_values(["true_block_time", "row_id"], kind="mergesort").index
        result.loc[rank] = np.sort(g["deployable_arrival_time"].to_numpy("float64"), kind="mergesort")
    return result

def shifted_arrivals(frame: pd.DataFrame, seconds: float = 7200.0) -> pd.Series:
    return frame["deployable_arrival_time"].astype(float) + seconds

def permute_runways_within_airport_day(frame: pd.DataFrame, seed: int = CONTROL_SEEDS["runway_permutation"]) -> pd.Series:
    required = {"airport", "utc_day", "runway"}
    if required.difference(frame.columns): raise ValueError("control frame lacks required columns")
    if frame["runway"].isna().any(): raise ValueError("missing runway is unsupported")
    rng = np.random.default_rng(seed); out = frame["runway"].copy()
    for _, g in frame.groupby(["airport", "utc_day"], dropna=False, sort=True):
        out.loc[g.index] = rng.permutation(g["runway"].to_numpy())
    return out

def endpoint_status(gain: float, oracle_gain: float, july_lower: float, january_lower: float,
                    december_gain: float) -> str:
    if gain < 0.5 or (oracle_gain > 0 and gain <= 0): return "FAIL"
    if july_lower > 0 and january_lower > 0 and december_gain >= 0: return "ADVANCE"
    return "MAYBE"

def resolve_service_seconds(runway: str, airport: str, service: dict) -> tuple[float, str]:
    for key, level in (((airport, runway), "runway"), ((airport, "*"), "airport"), (("*", "*"), "global")):
        if key in service and service[key][1] >= service["min_support"]:
            return float(np.clip(service[key][0], *SERVICE_CLIP)), level
    raise ValueError("no supported service fallback")

def simulate_runway_queue(frame: pd.DataFrame, service: dict) -> pd.DataFrame:
    if {"row_id", "airport", "runway", "unimpeded_seconds"}.difference(frame.columns): raise ValueError("missing queue columns")
    if "queue_arrival_time" not in frame and "release_time" not in frame: raise ValueError("missing queue arrival or release")
    work = frame.copy()
    if work["runway"].isna().any(): raise ValueError("missing runway is unsupported")
    if "queue_arrival_time" not in work:
        work["queue_arrival_time"] = work["release_time"].astype(float) + work["unimpeded_seconds"].astype(float)
    else:
        work["queue_arrival_time"] = work["queue_arrival_time"].astype(float)
    free_at, rows = {}, []
    for r in stable_queue_order(work).itertuples(index=False):
        key = (r.airport, r.runway); seconds, fallback = resolve_service_seconds(r.runway, r.airport, service)
        start = max(float(r.queue_arrival_time), free_at.get(key, float("-inf"))); finish = start + seconds; free_at[key] = finish
        wait = start - float(r.queue_arrival_time)
        rows.append({"row_id": r.row_id, "airport": r.airport, "runway": r.runway,
            "queue_arrival_time": float(r.queue_arrival_time), "service_start": start, "service_finish": finish,
            "service_seconds": seconds, "queue_wait_seconds": wait, "unimpeded_seconds": float(r.unimpeded_seconds),
            "taxi_prediction_seconds": float(r.unimpeded_seconds) + wait, "service_fallback": fallback})
    return pd.DataFrame(rows)

def canonical_id_bytes(values) -> tuple[str, bytes]:
    a = np.asarray(values); header = json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)}, separators=(",", ":")).encode() + b"\0"
    return a.dtype.str, header + a.tobytes(order="C")

def assert_exact_row_alignment(reference_ids, candidate_ids) -> None:
    ad, ab = canonical_id_bytes(reference_ids); bd, bb = canonical_id_bytes(candidate_ids)
    if len(np.unique(reference_ids)) != len(reference_ids) or len(np.unique(candidate_ids)) != len(candidate_ids): raise AssertionError("row IDs must be unique")
    if ad != bd or ab != bb: raise AssertionError("row ID dtype, bytes, or order differ")

def build_census() -> dict:
    from prc.data import data_dir, month_file
    paths = [("ranking", data_dir() / "ranking.parquet")] + [(m, month_file(m)) for m in ("2025-01", "2025-07", "2025-12")]
    outputs = census_outputs([(name, pd.read_parquet(path, columns=list(INPUT_COLUMNS))) for name, path in paths])
    payload = json.loads(outputs["census.json"])
    assert_frozen_census(payload)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items(): (OUT / name).write_bytes(content)
    manifest_path=OUT.parent/"source-manifest.json"
    manifest_path.write_bytes(rendered_source_manifest(payload,json.loads(manifest_path.read_text())))
    (OUT.parent/"CENSUS.md").write_bytes(rendered_census(payload))
    return payload

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("command", choices=("census",)); ap.parse_args()
    print(json.dumps(build_census(), indent=2))
