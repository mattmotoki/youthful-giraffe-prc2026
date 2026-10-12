from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


EXP = Path(os.environ.get("EXP094_EXPERIMENT_ROOT", Path(__file__).resolve().parent)).resolve()
ROOT = Path(os.environ.get("EXP094_REPO_ROOT", EXP.parents[1])).resolve()
MANIFEST = EXP / "source-manifest.json"
RESULT = EXP / "result.json"
PHASE = EXP / "phase-receipt.json"
CANDIDATE = ROOT / "submissions/candidates/exp-094-repeat-pair-smoothing.parquet"
FOLDS = ["july_loo", "december", "january"]
FOLD_MONTH = {"july_loo": "2025-07", "december": "2025-12", "january": "2025-01"}
SHARES = {"july_loo": 192122 / 344841, "january": 152719 / 344841}
B = 2000
SEED = 93026


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    data = (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        os.fsync(directory)
        os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(temporary, index=False)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        os.fsync(directory)
        os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def norm(value: object) -> str | None:
    if pd.isna(value):
        return None
    normalized = unicodedata.normalize("NFKC", str(value)).strip().upper()
    return normalized or None


def validate_ids(values: pd.Series, name: str) -> np.ndarray:
    if not pd.api.types.is_integer_dtype(values.dtype):
        raise ValueError(name + " IDs are not exact integers")
    array = values.to_numpy()
    if pd.isna(array).any() or len(np.unique(array)) != len(array):
        raise ValueError(name + " invalid IDs")
    return array.astype("int64", copy=False)


def schema_fp(path: Path) -> str:
    return hashlib.sha256(str(pq.ParquetFile(path).schema_arrow).encode()).hexdigest()


def source_sets(manifest: dict) -> tuple[set[str], set[str]]:
    historical = set()
    candidate = set()
    for relative, source in manifest["sources"].items():
        kind = source[3]
        (historical if kind in {"training", "panel"} else candidate).add(relative)
    return historical, candidate


def verify_phase_sources(manifest: dict, phase: str) -> None:
    historical, candidate = source_sets(manifest)
    selected = historical if phase == "historical" else candidate
    for relative in sorted(selected):
        digest, size, rows, kind = manifest["sources"][relative]
        path = ROOT / relative
        if path.stat().st_size != size or sha(path) != digest:
            raise ValueError(relative + " byte refusal")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != rows or schema_fp(path) != manifest["schema_fingerprints"][kind]:
            raise ValueError(relative + " metadata refusal")
        projection = manifest["projections"]["raw" if kind in {"training", "ranking"} else kind]
        if len(projection) != len(set(projection)) or not set(projection) <= set(parquet.schema_arrow.names):
            raise ValueError(relative + " projection refusal")


def raw_path(month: str) -> Path:
    year, number = month.split("-")
    end = f"{year}-{int(number) + 1:02d}-01" if number != "12" else "2026-01-01"
    return ROOT / f"data/raw/training_{month}-01_{end}.parquet"


def collision_bad(raw: pd.Series, origin: pd.Series, date: pd.Series) -> set[tuple]:
    frame = pd.DataFrame({"raw": raw, "origin": origin, "date": date})
    frame["key"] = frame.raw.map(norm)
    frame = frame[frame.key.notna() & frame.origin.notna() & frame.date.notna()]
    counts = frame.groupby(["origin", "date", "key"], dropna=False).raw.agg(
        lambda values: len({str(value) for value in values if not pd.isna(value)})
    )
    return set(counts[counts > 1].index)


def prepared(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = raw.loc[raw.PHASE_mvt.eq("DEP")].reset_index(drop=True).copy()
    validate_ids(frame.MVT_ID_mvt, "departure")
    frame["time"] = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True)
    frame["date"] = frame.time.dt.strftime("%Y-%m-%d")
    frame["origin"] = frame.ADEP_mvt.map(norm)
    frame["service"] = frame.FLIGHT_mvt.map(norm)
    frame["radio"] = frame.CALLSIGN_flt.map(norm)
    origin_rows = pd.DataFrame({"raw": frame.ADEP_mvt, "date": frame.date, "key": frame.origin})
    origin_rows = origin_rows[origin_rows.key.notna() & origin_rows.date.notna()]
    origin_counts = origin_rows.groupby(["date", "key"], dropna=False).raw.agg(
        lambda values: len({str(value) for value in values if not pd.isna(value)})
    )
    bad_origin = set(origin_counts[origin_counts > 1].index)
    frame["origin_ok"] = [
        value is not None and date is not None and (date, value) not in bad_origin
        for date, value in zip(frame.date, frame.origin)
    ]
    bad_service = collision_bad(frame.FLIGHT_mvt, frame.origin, frame.date)
    bad_radio = collision_bad(frame.CALLSIGN_flt, frame.origin, frame.date)
    frame["service_ok"] = [
        value is not None and (origin, date, value) not in bad_service
        for origin, date, value in zip(frame.origin, frame.date, frame.service)
    ]
    frame["radio_ok"] = [
        value is not None and (origin, date, value) not in bad_radio
        for origin, date, value in zip(frame.origin, frame.date, frame.radio)
    ]
    return frame, {"origin": len(bad_origin), "service": len(bad_service), "radio": len(bad_radio)}


def reciprocal_pairs(
    ids: pd.Series,
    times: pd.Series,
    keys: pd.Series,
    min_gap_seconds: float = 60,
    max_gap_seconds: float = 86399,
    next_day: bool = False,
) -> set[tuple[int, int]]:
    rows = pd.DataFrame({"id": ids, "time": pd.to_datetime(times, utc=True), "key": keys})
    rows = rows[rows.time.notna() & rows.key.notna()]
    edges: set[tuple[int, int]] = set()
    lower = int(min_gap_seconds * 1_000_000_000)
    upper = int(max_gap_seconds * 1_000_000_000)
    for _, group in rows.groupby("key", sort=False):
        group = group.sort_values(["time", "id"], kind="mergesort")
        timestamps = group.time.astype("int64").to_numpy()
        row_ids = group.id.to_numpy(dtype="int64")
        outgoing_lo = np.searchsorted(timestamps, timestamps + lower, side="right")
        outgoing_hi = np.searchsorted(timestamps, timestamps + upper, side="right")
        incoming_lo = np.searchsorted(timestamps, timestamps - upper, side="left")
        incoming_hi = np.searchsorted(timestamps, timestamps - lower, side="left")
        for earlier in np.flatnonzero(outgoing_hi - outgoing_lo == 1):
            later = outgoing_lo[earlier]
            if later < len(group) and incoming_hi[later] - incoming_lo[later] == 1:
                if next_day and group.time.iloc[later].date() != (
                    group.time.iloc[earlier] + pd.Timedelta(days=1)
                ).date():
                    continue
                edges.add((int(row_ids[earlier]), int(row_ids[later])))
    return edges


def lane_pairs(
    frame: pd.DataFrame,
    identity: pd.Series,
    identity_ok: pd.Series,
    date_in_key: bool = True,
    max_gap_seconds: float = 86399,
    next_day: bool = False,
) -> set[tuple[int, int]]:
    valid = frame.origin_ok & identity_ok & identity.notna() & frame.time.notna()
    if date_in_key:
        keys = pd.Series(list(zip(frame.origin, frame.date, identity)), index=frame.index).where(valid)
    else:
        keys = pd.Series(list(zip(frame.origin, identity)), index=frame.index).where(valid)
    return reciprocal_pairs(frame.MVT_ID_mvt, frame.time, keys, 60, max_gap_seconds, next_day)


def disjoint(edges: set[tuple[int, int]]) -> list[tuple[int, int]]:
    degree = Counter(identifier for edge in edges for identifier in edge)
    return sorted(edge for edge in edges if degree[edge[0]] == 1 and degree[edge[1]] == 1)


def exact_pairs(
    frame: pd.DataFrame,
    service: pd.Series | None = None,
    radio: pd.Series | None = None,
    service_ok: pd.Series | None = None,
    radio_ok: pd.Series | None = None,
    date_in_key: bool = True,
    max_gap_seconds: float = 86399,
    next_day: bool = False,
) -> list[tuple[int, int]]:
    service = frame.service if service is None else service
    radio = frame.radio if radio is None else radio
    service_ok = frame.service_ok if service_ok is None else service_ok
    radio_ok = frame.radio_ok if radio_ok is None else radio_ok
    service_edges = lane_pairs(frame, service, service_ok, date_in_key, max_gap_seconds, next_day)
    radio_edges = lane_pairs(frame, radio, radio_ok, date_in_key, max_gap_seconds, next_day)
    return disjoint(service_edges & radio_edges)


def primary_pairs(frame: pd.DataFrame) -> list[tuple[int, int]]:
    return exact_pairs(frame)


def tie_digest(identifier: int) -> str:
    return hashlib.sha256(f"MVT_ID_mvt={int(identifier)}".encode()).hexdigest()


def rotation_pairs(frame: pd.DataFrame) -> list[tuple[int, int]]:
    valid = frame.origin_ok & frame.service_ok & frame.radio_ok & frame.service.notna() & frame.radio.notna()
    service = pd.Series(None, index=frame.index, dtype=object)
    radio = pd.Series(None, index=frame.index, dtype=object)
    rotated_ok = pd.Series(False, index=frame.index)
    rows = frame.loc[valid, ["origin", "date", "time", "MVT_ID_mvt", "service", "radio"]].copy()
    rows["tie"] = rows.MVT_ID_mvt.map(tie_digest)
    for _, group in rows.groupby(["origin", "date"], sort=False):
        indices = group.sort_values(["time", "tie"], kind="mergesort").index
        tuples = list(zip(frame.loc[indices, "service"], frame.loc[indices, "radio"]))
        rotated = tuples[-1:] + tuples[:-1]
        if rotated:
            service.loc[indices] = [value[0] for value in rotated]
            radio.loc[indices] = [value[1] for value in rotated]
            rotated_ok.loc[indices] = True
    return exact_pairs(frame, service, radio, rotated_ok, rotated_ok)


def previous_day_pairs(frame: pd.DataFrame) -> list[tuple[int, int]]:
    return exact_pairs(frame, date_in_key=False, max_gap_seconds=172799, next_day=True)


def smooth(base: np.ndarray, pairs: list[tuple[int, int]], ids: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    original = np.asarray(base, dtype="float64")
    output = original.copy()
    positions = {int(identifier): position for position, identifier in enumerate(ids)}
    mask = np.zeros(len(output), dtype=bool)
    for earlier_id, later_id in pairs:
        earlier, later = positions[earlier_id], positions[later_id]
        mean = (original[earlier] + original[later]) / 2.0
        output[earlier] = output[later] = mean
        mask[earlier] = mask[later] = True
    changed = mask & (output.view("uint64") != original.view("uint64"))
    if not np.array_equal(output[~mask].view("uint64"), original[~mask].view("uint64")):
        raise ValueError("unchanged-bit refusal")
    return output, mask, changed


def bootstrap(y: np.ndarray, base: np.ndarray, pred: np.ndarray, day: np.ndarray, seed: int) -> list[float]:
    grouped = pd.DataFrame({"day": day, "base": (y - base) ** 2, "pred": (y - pred) ** 2}).groupby(
        "day", sort=True
    ).agg(n=("base", "size"), base=("base", "sum"), pred=("pred", "sum")).to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(B)
    for draw in range(B):
        sampled = grouped[rng.integers(0, len(grouped), len(grouped))].sum(axis=0)
        values[draw] = math.sqrt(sampled[1] / sampled[0]) - math.sqrt(sampled[2] / sampled[0])
    return [float(value) for value in np.quantile(values, [0.025, 0.975], method="linear")]


def metrics(y, base, pred, pair_mask, changed, day, seed, no_nm, airport, pairs) -> dict:
    base_error = (y - base) ** 2
    pred_error = (y - pred) ** 2
    gain = base_error - pred_error

    def rmse(mask, values):
        return float(np.sqrt(np.mean(values[mask]))) if np.any(mask) else None

    airports = pd.DataFrame({"airport": airport, "gain": gain}).groupby(
        "airport", dropna=False, sort=True
    ).agg(rows=("gain", "size"), sse_gain=("gain", "sum")).reset_index()
    slices = {"le3600": y <= 3600, "gt3600": y > 3600, "no_nm": no_nm}
    all_rows = np.ones(len(y), dtype=bool)
    return {
        "pair_count": len(pairs),
        "paired_rows": int(pair_mask.sum()),
        "changed_rows": int(changed.sum()),
        "base_rmse": rmse(all_rows, base_error),
        "corrected_rmse": rmse(all_rows, pred_error),
        "sse_gain": float(gain.sum()),
        "rmse_gain": float(np.sqrt(base_error.mean()) - np.sqrt(pred_error.mean())),
        "paired_row_base_rmse": rmse(pair_mask, base_error),
        "paired_row_corrected_rmse": rmse(pair_mask, pred_error),
        "interval95": bootstrap(y, base, pred, day, seed),
        "airport_contributions": [
            {"airport": None if pd.isna(row.airport) else str(row.airport), "rows": int(row.rows), "sse_gain": float(row.sse_gain)}
            for row in airports.itertuples(index=False)
        ],
        "slices": {
            name: {"rows": int(mask.sum()), "base_rmse": rmse(mask, base_error), "corrected_rmse": rmse(mask, pred_error)}
            for name, mask in slices.items()
        },
    }


def join_panel(frame: pd.DataFrame, panel: pd.DataFrame, fold: str) -> None:
    raw_ids = validate_ids(frame.MVT_ID_mvt, fold + " raw")
    panel_ids = validate_ids(panel.MVT_ID_mvt, fold + " panel")
    if len(frame) != len(panel) or not np.array_equal(raw_ids, panel_ids):
        raise ValueError(fold + " identity/order refusal")
    if set(panel.fold.astype(str)) != {fold}:
        raise ValueError(fold + " fold refusal")
    if not np.array_equal(frame.date.astype(str).to_numpy(), panel.day.astype(str).to_numpy()):
        raise ValueError(fold + " day refusal")


def historical_phase(manifest: dict) -> tuple[dict, dict[str, dict], dict[str, dict]]:
    verify_phase_sources(manifest, "historical")
    output = {"registration": {"draws": B, "seed": SEED, "shares": SHARES, "source_manifest_sha256": sha(MANIFEST)}, "folds": {}}
    mses = {arm: {} for arm in ["base", "primary", "rotation"]}
    gains = {arm: {} for arm in ["primary", "rotation"]}
    for fold_offset, fold in enumerate(FOLDS):
        raw = pd.read_parquet(raw_path(FOLD_MONTH[fold]), columns=manifest["projections"]["raw"])
        frame, collisions = prepared(raw)
        panel_path = ROOT / f"experiments/exp-050-nmblind/work/panel-v22-{fold}.parquet"
        panel = pd.read_parquet(panel_path, columns=manifest["projections"]["panel"])
        join_panel(frame, panel, fold)
        y = panel.y_true.to_numpy(dtype="float64")
        base = panel.p_v22.to_numpy(dtype="float64")
        if not np.isfinite(y).all() or not np.isfinite(base).all():
            raise ValueError(fold + " nonfinite label/base")
        pairs_by_arm = {"primary": primary_pairs(frame), "rotation": rotation_pairs(frame)}
        fold_result = {
            "rows": len(frame),
            "collisions": collisions,
            "previous_day_diagnostic": {"pair_count": len(previous_day_pairs(frame)), "row_count": 2 * len(previous_day_pairs(frame))},
            "arms": {},
        }
        mses["base"][fold] = float(np.mean((y - base) ** 2))
        for arm_offset, arm in enumerate(["primary", "rotation"]):
            pred, pair_mask, changed = smooth(base, pairs_by_arm[arm], frame.MVT_ID_mvt)
            arm_metrics = metrics(
                y, base, pred, pair_mask, changed, frame.date.to_numpy(), SEED + fold_offset + 100 * arm_offset,
                panel.regime.astype(str).eq("nonm").to_numpy(), frame.origin.to_numpy(), pairs_by_arm[arm],
            )
            fold_result["arms"][arm] = arm_metrics
            mses[arm][fold] = float(np.mean((y - pred) ** 2))
            gains[arm][fold] = arm_metrics["rmse_gain"]
        output["folds"][fold] = fold_result
    output["board_share"] = {
        arm: float(math.sqrt(sum(SHARES[fold] * mses["base"][fold] for fold in SHARES)) - math.sqrt(sum(SHARES[fold] * mses[arm][fold] for fold in SHARES)))
        for arm in ["primary", "rotation"]
    }
    gates = {
        "nonnegative_all_folds": all(gains["primary"][fold] >= 0 for fold in FOLDS),
        "clear_july_january": all(output["folds"][fold]["arms"]["primary"]["interval95"][0] > 0 for fold in ["july_loo", "january"]),
        "beats_rotation_two_folds": sum(gains["primary"][fold] > gains["rotation"][fold] for fold in FOLDS) >= 2,
        "beats_rotation_board": output["board_share"]["primary"] > output["board_share"]["rotation"],
        "board_gain_at_least_0_10": output["board_share"]["primary"] >= 0.10,
    }
    output["historical_gates"] = gates
    return output, mses, gains


def candidate_phase(manifest: dict, output: dict) -> None:
    verify_phase_sources(manifest, "candidate")
    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=manifest["projections"]["raw"])
    ranking, collisions = prepared(raw)
    v22 = pd.read_parquet(ROOT / "submissions/uploaded/youthful-giraffe_v22.parquet", columns=manifest["projections"]["v22"])
    template = pd.read_parquet(ROOT / "data/raw/submitting.parquet", columns=manifest["projections"]["submitting"])
    ranking_ids = validate_ids(ranking.MVT_ID_mvt, "ranking")
    v22_ids = validate_ids(v22.MVT_ID_mvt, "v22")
    template_ids = validate_ids(template.MVT_ID_mvt, "template")
    if not np.array_equal(ranking_ids, v22_ids) or not np.array_equal(template_ids, v22_ids):
        raise ValueError("ranking/template/v22 identity-order refusal")
    base = v22.TAXITIME_SEC_mvt.to_numpy(dtype="float64")
    if not np.isfinite(base).all():
        raise ValueError("v22 nonfinite prediction")
    pairs = primary_pairs(ranking)
    pred, paired, changed = smooth(base, pairs, ranking.MVT_ID_mvt)
    output["ranking"] = {"rows": len(ranking), "pair_count": len(pairs), "paired_rows": int(paired.sum()), "changed_rows": int(changed.sum()), "collisions": collisions}
    output["gates"] = {**output["historical_gates"], "ranking_changed_at_least_100": int(changed.sum()) >= 100}
    if not output["gates"]["ranking_changed_at_least_100"]:
        output["decision"] = "FAIL"
        return
    candidate = template.copy()
    candidate["TAXITIME_SEC_mvt"] = pred
    atomic_parquet(CANDIDATE, candidate)
    written = pd.read_parquet(CANDIDATE, columns=manifest["projections"]["submitting"])
    written_ids = validate_ids(written.MVT_ID_mvt, "written candidate")
    written_pred = written.TAXITIME_SEC_mvt.to_numpy(dtype="float64")
    if len(written) != len(template) or not np.array_equal(written_ids, template_ids) or not np.isfinite(written_pred).all():
        raise ValueError("candidate identity/order/finiteness refusal")
    written_changed = written_pred.view("uint64") != base.view("uint64")
    if not np.array_equal(written_pred[~written_changed].view("uint64"), base[~written_changed].view("uint64")):
        raise ValueError("candidate unchanged-value refusal")
    if not np.array_equal(written_changed, changed):
        raise ValueError("candidate changed-mask refusal")
    output["decision"] = "ADVANCE"
    output["candidate"] = {
        "path": str(CANDIDATE.relative_to(ROOT)),
        "sha256": sha(CANDIDATE),
        "rows": int(len(written)),
        "exact_id_order": True,
        "unique_ids": bool(len(np.unique(written_ids)) == len(written_ids)),
        "finite_predictions": bool(np.isfinite(written_pred).all()),
        "changed_rows": int(written_changed.sum()),
        "unchanged_rows_bit_identical": int((~written_changed).sum()),
        "minimum": float(written_pred.min()),
        "maximum": float(written_pred.max()),
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    output, _, _ = historical_phase(manifest)
    historical_pass = all(output["historical_gates"].values())
    atomic_json(
        PHASE,
        {
            "historical_gates": output["historical_gates"],
            "historical_pass": historical_pass,
            "phase": "candidate_authorized" if historical_pass else "historical_fail",
            "time_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    if not historical_pass:
        output["gates"] = {**output["historical_gates"], "ranking_changed_at_least_100": "not_evaluated"}
        output["decision"] = "FAIL"
        output["candidate_phase_opened"] = False
        atomic_json(RESULT, output)
        return
    output["candidate_phase_opened"] = True
    candidate_phase(manifest, output)
    atomic_json(RESULT, output)


if __name__ == "__main__":
    main()
