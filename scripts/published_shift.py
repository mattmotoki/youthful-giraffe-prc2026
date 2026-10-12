"""Apply the frozen published-mean correction to a replayed base vector."""
import numpy as np


def apply(airports, months, base, eligible, parameters):
    airports, months = np.asarray(airports), np.asarray(months)
    base, eligible = np.asarray(base, dtype="float64"), np.asarray(eligible)
    if not (base.ndim == eligible.ndim == airports.ndim == months.ndim == 1 and
            len(base) == len(eligible) == len(airports) == len(months)):
        raise ValueError("Shift inputs must be aligned one-dimensional arrays")
    if not np.isfinite(base).all() or len(base) == 0 or base.min() < 60:
        raise ValueError("Invalid base predictions")
    if not np.isin(eligible, [0, 1]).all():
        raise ValueError("Admission values must be binary")
    eligible = eligible.astype(bool)
    out, covered, cells = base.copy(), np.zeros(len(base), bool), []
    seen = set()
    for p in parameters:
        if p["mode"] != "ranking_inspection":
            continue
        key = (p["airport"], p["month"])
        if key in seen:
            raise ValueError("Duplicate shift cell")
        seen.add(key)
        if not (0 <= p["lambda_value"] <= 1 and np.isfinite(p["corrected_mean_s"])):
            raise ValueError("Invalid frozen shrinkage parameter")
        keep = (airports == p["airport"]) & (months == p["month"])
        moving = keep & eligible
        n, nm = int(keep.sum()), int(moving.sum())
        if n == 0 or nm == 0:
            raise ValueError("Shift cell has no population or admitted rows")
        covered |= keep
        mean = float(base[keep].mean())
        wanted = mean + p["lambda_value"] * (p["corrected_mean_s"] - mean)
        target_moving = float(base[moving].mean()) + p["lambda_value"] * (p["corrected_mean_s"] - mean)
        target_moving = max(60, target_moving)
        lo = 60 - float(base[moving].max()) - 1
        hi = target_moving + float(base[moving].max()) + 1
        for _ in range(80):
            mid = (lo + hi) / 2
            if np.maximum(60, base[moving] + mid).mean() < target_moving:
                lo = mid
            else:
                hi = mid
        shift = (lo + hi) / 2
        out[moving] = np.maximum(60, base[moving] + shift)
        cells.append(dict(airport=p["airport"], month=p["month"], rows=n, moving_rows=nm,
                          held_rows=n-nm, base_mean_s=mean, proxy_mean_s=p["corrected_mean_s"],
                          lambda_value=p["lambda_value"], wanted_mean_s=wanted,
                          served_mean_s=float(out[keep].mean()), shift_s=shift,
                          floor_rows=int(np.sum(out[moving] == 60))))
    if not covered.all():
        raise ValueError("Frozen cells do not cover the input population")
    if not np.isfinite(out).all() or out.min() < 60 or not np.array_equal(out[~eligible], base[~eligible]):
        raise ValueError("Shift violates the serving floor or admission rule")
    return out, cells
