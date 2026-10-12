#!/usr/bin/env python3
"""Frozen input-only aggregate screens for exp-090."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "exp-090-input-screens"
MANIFEST = EXP / "source-manifest.json"
OUT = EXP / "aggregate.json"
NM_FIELDS = [
    "CALLSIGN_flt", "ADEP_flt", "ADES_flt", "ADES_FILED_flt", "MARKET_SEGMENT_flt", "IOBT_flt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt",
    "AIRCRAFT_OPERATOR_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt",
]
FORBIDDEN_PROJECTED = {"BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "FLIGHT_ID_mvt", "STAND_mvt", "LOBT_flt"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def schema_pairs(path: Path) -> list[list[str]]:
    return [[f.name, str(f.type)] for f in pq.ParquetFile(path).schema_arrow]


def validate_source(root: Path, spec: dict[str, Any]) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != spec["size_bytes"] or sha256_file(path) != spec["sha256"]:
        raise ValueError(f"source bytes differ: {spec['path']}")
    pf = pq.ParquetFile(path)
    if pf.metadata.num_rows != spec["rows"] or schema_pairs(path) != spec["schema"]:
        raise ValueError(f"source schema differs: {spec['path']}")
    names = [x[0] for x in spec["schema"]]
    projected = spec["projected_columns"]
    if len(projected) != len(set(projected)) or any(c not in names for c in projected):
        raise ValueError("invalid projection")
    if spec["path"].endswith("ranking.parquet") and FORBIDDEN_PROJECTED.intersection(projected):
        raise ValueError("forbidden ranking field projected")
    if spec["path"].endswith("v22.parquet") and projected != ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]:
        raise ValueError("v22 projection differs")
    return path


def norm(v: Any) -> str | None:
    if v is None or pd.isna(v):
        return None
    x = unicodedata.normalize("NFKC", str(v)).strip().upper()
    return x or None


def normalized(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].map(norm)


def valid_ids(values: pd.Series, name: str, require_unique: bool = True) -> np.ndarray:
    a = values.to_numpy(dtype=float)
    if not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
        raise ValueError(f"non-finite or fractional {name}")
    if require_unique and pd.Series(a).duplicated().any():
        raise ValueError(f"duplicate {name}")
    return a


def collision_groups(raw: pd.Series, keys: list[pd.Series]) -> set[tuple[Any, ...]]:
    z = pd.DataFrame({f"k{i}": k for i, k in enumerate(keys)})
    z["raw"] = raw
    z["normal"] = raw.map(norm)
    z = z[z["normal"].notna()].copy()
    group_cols = [f"k{i}" for i in range(len(keys))] + ["normal"]
    n = z.groupby(group_cols, dropna=False)["raw"].agg(lambda s: len(set(str(x) for x in s if not pd.isna(x))))
    return {tuple(idx) for idx, count in n.items() if count > 1}


def digest_tie(row: pd.Series) -> str:
    text = "|".join("" if pd.isna(v) else str(v) for v in row)
    return hashlib.sha256(text.encode()).hexdigest()


def reciprocal_pairs(times: pd.Series, group: pd.Series, minimum: int, maximum: int) -> set[int]:
    """Indices of earlier rows with exactly one later, whose later has exactly one earlier."""
    z = pd.DataFrame({"time": pd.to_datetime(times, utc=True), "group": group}, index=times.index)
    z = z[z.time.notna() & z.group.notna()]
    supported: set[int] = set()
    for _, g in z.groupby("group", sort=False):
        t = g.time.astype("int64").to_numpy() // 1_000_000_000
        idx = g.index.to_numpy()
        delta = t[None, :] - t[:, None]
        edge = (delta > minimum) & (delta <= maximum)
        out_n = edge.sum(axis=1)
        in_n = edge.sum(axis=0)
        for i, j in zip(*np.where(edge)):
            if out_n[i] == 1 and in_n[j] == 1:
                supported.add(int(idx[i]))
    return supported


def permute_within(df: pd.DataFrame, key: pd.Series, group_cols: list[str]) -> pd.Series:
    out = key.copy()
    z = df[group_cols + ["MVT_TIME_UTC_mvt"]].copy()
    z["key"] = key
    z["tie"] = df.apply(lambda r: digest_tie(r[[*group_cols, "MVT_TIME_UTC_mvt"]]), axis=1)
    for _, g in z.groupby(group_cols, dropna=False, sort=False):
        order = g.sort_values(["MVT_TIME_UTC_mvt", "tie"], kind="mergesort").index
        vals = key.loc[order].tolist()
        if vals:
            out.loc[order] = vals[-1:] + vals[:-1]
    return out


def s2(dep: pd.DataFrame) -> dict[str, Any]:
    d = dep.copy()
    d["origin"] = normalized(d, "ADEP_mvt")
    d["date"] = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    lanes: dict[str, set[int]] = {}
    controls: dict[str, dict[str, int]] = {}
    collisions: dict[str, int] = {}
    for label, col in [("flight_key", "FLIGHT_mvt"), ("callsign_key", "CALLSIGN_flt")]:
        key = normalized(d, col)
        bad = collision_groups(d[col], [d.origin, d.date])
        ok = pd.Series([((o, dt, k) not in bad) for o, dt, k in zip(d.origin, d.date, key)], index=d.index)
        group = pd.Series(list(zip(d.origin, d.date, key)), index=d.index).where(ok & key.notna())
        real = reciprocal_pairs(d.MVT_TIME_UTC_mvt, group, 60, 86399)
        lanes[label] = real
        # Previous-day key: group by origin/key, allow only an event on the next date.
        dt = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True)
        z = pd.DataFrame({"time": dt, "origin": d.origin, "key": key}, index=d.index)
        prev: set[int] = set()
        for _, g in z.dropna().groupby(["origin", "key"], sort=False):
            for i, a in g.iterrows():
                cand = g[(g.time.dt.date == (a.time + pd.Timedelta(days=1)).date())]
                gaps = (cand.time - a.time).dt.total_seconds()
                cand = cand[(gaps > 60) & (gaps <= 172799)]
                if len(cand) == 1:
                    j = cand.index[0]
                    earlier = g[(g.time.dt.date == a.time.date())]
                    eg = (g.loc[j, "time"] - earlier.time).dt.total_seconds()
                    earlier = earlier[(eg > 60) & (eg <= 172799)]
                    if len(earlier) == 1:
                        prev.add(int(i))
        perm = permute_within(d, key, ["origin", "date"])
        pgroup = pd.Series(list(zip(d.origin, d.date, perm)), index=d.index).where(perm.notna())
        pc = reciprocal_pairs(d.MVT_TIME_UTC_mvt, pgroup, 60, 86399)
        controls[label] = {"previous_utc_day": len(prev), "within_airport_day_permutation": len(pc)}
        collisions[label] = len(bad)
    n = len(d)
    lane_result = {}
    passed = False
    for label, rows in lanes.items():
        count = len(rows)
        gate = count >= 100 and count / max(n, 1) >= 0.0002 and count > controls[label]["within_airport_day_permutation"]
        lane_result[label] = {"supported": count, "share": count / max(n, 1), "gate_pass": gate}
        passed |= gate
    return {"population": n, "lanes": lane_result, "intersection": len(lanes["flight_key"] & lanes["callsign_key"]),
            "controls": controls, "normalized_collision_groups": collisions, "gate_pass": passed}


def match_s3(dep: pd.DataFrame, arr: pd.DataFrame, shift_hours: int = 0, permute: bool = False) -> tuple[set[int], int]:
    d = dep.copy(); a = arr.copy()
    d["flight"] = normalized(d, "FLIGHT_mvt"); a["flight"] = normalized(a, "FLIGHT_mvt")
    d["origin"] = normalized(d, "ADEP_mvt"); d["dest"] = normalized(d, "ADES_mvt")
    a["origin"] = normalized(a, "ADEP_mvt"); a["dest"] = normalized(a, "ADES_mvt")
    a["time"] = pd.to_datetime(a.MVT_TIME_UTC_mvt, utc=True) + pd.Timedelta(hours=shift_hours)
    a["date"] = a.time.dt.strftime("%Y-%m-%d")
    if permute:
        a["MVT_TIME_UTC_mvt"] = a.time
        a["flight"] = permute_within(a, a.flight, ["dest", "date"])
    araw = arr["FLIGHT_mvt"]
    bad_a = collision_groups(araw, [a.origin, a.dest, a.date])
    ddate = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    bad_d = collision_groups(d["FLIGHT_mvt"], [d.origin, d.dest, ddate])
    edges: list[tuple[int, int]] = []
    ambiguous = 0
    for di, r in d.iterrows():
        if None in (r.flight, r.origin, r.dest) or pd.isna(r.MVT_TIME_UTC_mvt):
            continue
        t = pd.Timestamp(r.MVT_TIME_UTC_mvt)
        if (r.origin, r.dest, t.strftime("%Y-%m-%d"), r.flight) in bad_d:
            ambiguous += 1
            continue
        cand = a[(a.flight == r.flight) & (a.origin == r.origin) & (a.dest == r.dest)]
        gaps = (cand.time - t).dt.total_seconds()
        prev_date = (t - pd.Timedelta(days=1)).strftime("%Y-%m-%d"); same_date = t.strftime("%Y-%m-%d")
        window = ((cand.date == prev_date) & (gaps >= -64800) & (gaps <= -300)) | \
                 ((cand.date == same_date) & (gaps >= 300) & (gaps <= 64800))
        cand = cand[window]
        cand = cand[[((x.origin, x.dest, x.date, x.flight) not in bad_a) for _, x in cand.iterrows()]]
        if len(cand) == 1:
            edges.append((int(di), int(cand.index[0])))
        elif len(cand) > 1:
            ambiguous += 1
    dc = pd.Series([x[0] for x in edges]).value_counts(); ac = pd.Series([x[1] for x in edges]).value_counts()
    return {di for di, ai in edges if dc[di] == 1 and ac[ai] == 1}, ambiguous


def s3(all_rows: pd.DataFrame) -> dict[str, Any]:
    dep = all_rows[all_rows.PHASE_mvt == "DEP"].copy()
    arr = all_rows[all_rows.PHASE_mvt == "ARR"].copy()
    no_nm = dep[NM_FIELDS].isna().all(axis=1)
    mt = pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True); st = pd.to_datetime(dep.SCHED_TIME_UTC_mvt, utc=True)
    crossing = no_nm & mt.notna() & st.notna() & ((mt.dt.floor("D") - st.dt.floor("D")) == pd.Timedelta(days=1))
    c = dep[crossing]
    real, ambiguity = match_s3(c, arr)
    shifted, _ = match_s3(c, arr, shift_hours=24)
    permuted, _ = match_s3(c, arr, permute=True)
    split = {"D_minus_1": 0, "D": 0}
    # Reidentify the unique real arrival date for aggregate split.
    for di in real:
        r = c.loc[di]; t = pd.Timestamp(r.MVT_TIME_UTC_mvt)
        cand = arr[(normalized(arr, "FLIGHT_mvt") == norm(r.FLIGHT_mvt)) &
                   (normalized(arr, "ADEP_mvt") == norm(r.ADEP_mvt)) &
                   (normalized(arr, "ADES_mvt") == norm(r.ADES_mvt))]
        at = pd.to_datetime(cand.MVT_TIME_UTC_mvt, utc=True); gaps = (at - t).dt.total_seconds()
        prev_date = (t - pd.Timedelta(days=1)).strftime("%Y-%m-%d"); same_date = t.strftime("%Y-%m-%d")
        dates = at.dt.strftime("%Y-%m-%d")
        cand = cand[((dates == prev_date) & (gaps >= -64800) & (gaps <= -300)) |
                    ((dates == same_date) & (gaps >= 300) & (gaps <= 64800))]
        if len(cand) == 1:
            split["D" if pd.Timestamp(cand.iloc[0].MVT_TIME_UTC_mvt).date() == t.date() else "D_minus_1"] += 1
    n = len(c); count = len(real)
    gate = count >= 20 and count / max(n, 1) >= 0.005 and count > len(shifted) and count > len(permuted)
    return {"no_nm_population": int(no_nm.sum()), "crossing_population": n, "supported": count,
            "arrival_date_split": split, "ambiguity_rejections": ambiguity,
            "controls": {"arrival_plus_24h": len(shifted), "within_destination_day_identity_permutation": len(permuted)},
            "share": count / max(n, 1), "gate_pass": gate}


def s4(dep: pd.DataFrame, pred: pd.DataFrame) -> dict[str, Any]:
    did = valid_ids(dep.MVT_ID_mvt, "ranking departure ID")
    pid = valid_ids(pred.MVT_ID_mvt, "v22 ID")
    lookup = pd.Series(np.arange(len(dep)), index=did)
    loc = lookup.reindex(pid)
    unmatched = int(loc.isna().sum())
    identity_fail = int(unmatched or len(pid) != len(did) or set(pid) != set(did))
    if identity_fail:
        raise ValueError("v22 join identity failure")
    joined = dep.iloc[loc.astype(int).to_numpy()].copy().reset_index(drop=True)
    order_fail = int(not np.array_equal(joined.MVT_ID_mvt.to_numpy(float), pid))
    if order_fail:
        raise ValueError("v22 join order failure")
    p = pred.TAXITIME_SEC_mvt.to_numpy(float)
    finite = np.isfinite(p)
    joined["airport"] = normalized(joined, "ADEP_mvt").fillna("__MISSING__")
    joined["runway"] = normalized(joined, "RUNWAY_mvt").fillna("__MISSING__")
    grouped = []
    for (airport, runway), g in joined.assign(prediction=p).groupby(["airport", "runway"], sort=True):
        v = g.prediction.to_numpy(float); f = np.isfinite(v)
        grouped.append({"airport": airport, "runway": runway, "below_120": int((f & (v < 120)).sum()),
                        "below_180": int((f & (v < 180)).sum()), "below_240": int((f & (v < 240)).sum())})
    below = {str(t): int((finite & (p < t)).sum()) for t in (120, 180, 240)}
    refusal = {"negative": int((finite & (p < 0)).sum()), "below_60": int((finite & (p < 60)).sum()),
               "non_finite": int((~finite).sum()), "unmatched": unmatched, "duplicate_v22_ids": 0,
               "duplicate_ranking_departure_ids": 0, "join_order_failures": order_fail,
               "join_identity_failures": identity_fail}
    n = len(pred); gate = below["240"] >= 100 and below["240"] / max(n, 1) >= 0.0002
    return {"population": n, "below_seconds": below, "by_airport_runway": grouped,
            "refusal_checks": refusal, "below_240_share": below["240"] / max(n, 1), "gate_pass": gate}


def s5(dep: pd.DataFrame) -> dict[str, Any]:
    t = pd.to_datetime(dep.MVT_TIME_UTC_mvt, utc=True)
    airport = normalized(dep, "ADEP_mvt").fillna("__MISSING__")
    months = {}
    for name, start in [("january", pd.Timestamp("2026-01-01", tz="UTC")), ("july", pd.Timestamp("2026-07-01", tz="UTC"))]:
        end = start + pd.offsets.MonthBegin(1)
        mm = (t >= start) & (t < end)
        prefixes = {}
        for mins in (5, 15, 30, 60):
            mask = mm & (t < start + pd.Timedelta(minutes=mins))
            by = pd.Series(airport[mask]).value_counts().sort_index()
            prefixes[str(mins)] = {"total": int(mask.sum()), "by_airport": {str(k): int(v) for k, v in by.items()}}
        later = mm & (t.dt.day > 1) & (t.dt.hour == 0)
        days = int(t[later].dt.floor("D").nunique())
        months[name] = {"population": int(mm.sum()), "prefix_minutes": prefixes,
                        "later_day_first_hour_control": {"total": int(later.sum()), "days": days,
                        "per_day_mean": int(later.sum()) / max(days, 1)}}
    jan = months["january"]; count = jan["prefix_minutes"]["60"]["total"]
    gate = count >= 100 and count / max(jan["population"], 1) >= 0.0002
    return {"months": months, "january_first_hour_share": count / max(jan["population"], 1), "gate_pass": gate,
            "claim": "support measures only exposure to a month-isolated frame boundary"}


def privacy_check(value: Any, path: str = "") -> None:
    forbidden = ("mvt_id", "flight_id", "flight_mvt", "callsign", "row_record", "event_time")
    if isinstance(value, dict):
        for k, v in value.items():
            if any(x in str(k).lower() for x in forbidden):
                raise ValueError(f"private output key: {path}/{k}")
            privacy_check(v, f"{path}/{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value): privacy_check(v, f"{path}/{i}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-standard JSON number")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    privacy_check(value)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def run_real() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    paths = [validate_source(ROOT, s) for s in manifest["sources"]]
    rank = pd.read_parquet(paths[0], columns=manifest["sources"][0]["projected_columns"])
    pred = pd.read_parquet(paths[1], columns=manifest["sources"][1]["projected_columns"])
    phases = set(rank.PHASE_mvt.dropna().unique())
    if not phases.issubset({"DEP", "ARR"}): raise ValueError("unknown phase")
    dep = rank[rank.PHASE_mvt == "DEP"].copy()
    valid_ids(dep.MVT_ID_mvt, "ranking departure ID")
    result = {"format_version": 1, "claim_boundary": "input-only support; no prediction-value or physical-validity claim",
              "source_sha256": {s["path"]: s["sha256"] for s in manifest["sources"]},
              "screens": {"S2": s2(dep), "S3": s3(rank), "S4": s4(dep, pred), "S5": s5(dep)}}
    result["screens_clearing_gate"] = [k for k, v in result["screens"].items() if v["gate_pass"]]
    atomic_json(OUT, result)
    return result


if __name__ == "__main__":
    print(json.dumps(run_real()["screens_clearing_gate"]))
