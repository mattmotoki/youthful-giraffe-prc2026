"""Frozen label-free May-only contracts for the later registered experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

V, S, I = "2025-05", "2025-06", "2025-04"
A = ("2025-01", "2025-02", "2025-03", "2025-07", "2025-08", "2025-09",
     "2025-10", "2025-11", "2025-12")
DRAW_SEED = 660505
DRAW_COUNT = 2000
BLOCK_LENGTHS = (1, 2, 3)
SEED_SEARCH = range(1, 100001)


def assert_one_fold(folds):
    if tuple(folds) != (V,):
        raise ValueError("exactly one V fold (2025-05) is allowed")


def refuse_recombination(flag=False):
    if flag:
        raise ValueError("board or fold recombination is forbidden")


def _moving_blocks(n_dates, length, rng):
    rows = []
    for _ in range(DRAW_COUNT):
        draw = []
        while len(draw) < n_dates:
            start = int(rng.integers(0, n_dates))
            draw.extend((start + offset) % n_dates for offset in range(length))
        rows.append(draw[:n_dates])
    return np.asarray(rows, dtype="int32")


def freeze_draws(frame, output: Path):
    """Freeze draws from IDs and UTC dates only; never accepts a target-shaped column."""
    allowed = {"MVT_ID_mvt", "utc_date"}
    extra = set(frame.columns) - allowed
    if extra or set(frame.columns) != allowed:
        raise ValueError(f"feature-only draw input required; refused columns: {sorted(extra)}")
    if output.exists():
        raise FileExistsError(output)
    ids = np.asarray(frame["MVT_ID_mvt"].astype(str).tolist(), dtype="U")
    dates = np.asarray(frame["utc_date"].astype(str).tolist(), dtype="U")
    order = np.argsort(ids, kind="stable")
    unique = np.unique(dates[order])
    rng = np.random.default_rng(DRAW_SEED)
    arrays = {"schema_version": np.asarray([1], dtype="int32"),
              "ordered_ids": ids[order], "ordered_dates": dates[order],
              "unique_dates": unique}
    for length in BLOCK_LENGTHS:
        arrays[f"draw_l{length}"] = _moving_blocks(len(unique), length, rng)
    np.savez(output, **arrays)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def require_draws(path: Path, expected_digest, ids, dates):
    if not path.is_file():
        raise FileNotFoundError("premeasurement draw freeze is missing")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
        raise ValueError("draw digest mismatch")
    z = np.load(path)
    required = {"schema_version", "ordered_ids", "ordered_dates", "unique_dates",
                "draw_l1", "draw_l2", "draw_l3"}
    if set(z.files) != required or z["schema_version"].tolist() != [1]:
        raise ValueError("draw schema mismatch")
    order = np.argsort(np.asarray(ids).astype(str), kind="stable")
    if not np.array_equal(z["ordered_ids"], np.asarray(ids).astype(str)[order]):
        raise ValueError("ordered movement IDs mismatch")
    if not np.array_equal(z["ordered_dates"], np.asarray(dates).astype(str)[order]):
        raise ValueError("UTC dates mismatch")
    n_dates = len(z["unique_dates"])
    for length in BLOCK_LENGTHS:
        draw = z[f"draw_l{length}"]
        if draw.shape != (DRAW_COUNT, n_dates) or draw.min() < 0 or draw.max() >= n_dates:
            raise ValueError("draw dimensions or range mismatch")
        if length > 1:
            for row in draw[:20]:
                for start in range(0, n_dates, length):
                    chunk = row[start:min(start + length, n_dates)]
                    if any(chunk[k] != (chunk[0] + k) % n_dates for k in range(len(chunk))):
                        raise ValueError("not a moving block draw")


def search_matched_null(values, groups, supported, row_codes, c_weights, output: Path,
                        shams=10):
    if output.exists():
        raise FileExistsError(output)
    values = np.asarray(values, float); groups = np.asarray(groups)
    attempts = []
    selected = []
    for sham in range(shams):
        found = None
        for seed in SEED_SEARCH:
            perm = np.arange(len(values)); rng = np.random.default_rng([seed, sham])
            for group in np.unique(groups):
                rows = np.flatnonzero(groups == group); perm[rows] = rng.permutation(rows)
            moved = values[perm] != values
            reasons = []
            for group in np.unique(groups):
                if supported.get(str(group), False) and not moved[groups == group].any():
                    reasons.append(f"group:{group}")
            for month, codes in row_codes.items():
                codes = np.asarray(codes, int)
                for c_index, weight in enumerate(np.asarray(c_weights)):
                    for group in np.unique(groups):
                        if not supported.get(str(group), False): continue
                        rows = groups[codes] == group
                        before = values[codes[rows]] * weight[codes[rows]]
                        after = values[perm][codes[rows]] * weight[codes[rows]]
                        if not len(before) or not np.any(before != after):
                            reasons.append(f"stratum:{group}:{month}:c{c_index}")
            attempts.append({"sham": sham, "seed": seed, "accepted": not reasons,
                             "reasons": reasons})
            if not reasons: found = {"sham": sham, "seed": seed, "permutation": perm.tolist()}; break
        if found is None:
            output.write_text(json.dumps({"resolved": False, "attempts": attempts}, sort_keys=True)+"\n")
            raise RuntimeError("matched null incomplete; open (before label read)")
        selected.append(found)
    record = {"resolved": True, "attempts": attempts, "selected": selected}
    output.write_text(json.dumps(record, sort_keys=True)+"\n")
    return hashlib.sha256(output.read_bytes()).hexdigest()


def require_matched_null(path: Path, expected_digest):
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
        raise RuntimeError("matched null record missing or changed (before label read)")
    if not json.loads(path.read_text()).get("resolved"):
        raise RuntimeError("matched null incomplete; open (before label read)")
