"""The T11b-L stage (target v64): the zabd entry plus a quarter move toward dark18.

    python scripts/build_entry.py --target v64 --data-root DIR --out-dir NEW --record NEW.json

Target v64 first replays target zabd (e7b04ffd), then applies this stage:

- P2 rows: 2026 rows with an aircraft type, a row class other than noac, and no NM
  flight join (the official flight fields are empty). No P2 row is in the dark18
  member's own scope.
- Move: blend() below. The dark18 value is clipped to the 029 bounds with the 60 s
  floor; the entry moves a quarter of the way toward it and is clipped again.
- Long-taxi airport-day: flags() below. Departures (PHASE_mvt DEP, ADES_mvt not
  ADEP_mvt) are grouped by (ADEP_mvt, UTC day of MVT_TIME_UTC_mvt); NM taxi is
  MVT minus AOBT_3. A day is long when it has 8 or more rows with an NM taxi of
  3,600 s or more, and these rows are 2 % or more of the rows with AOBT_3. Every
  row of a long-taxi day (row_keys() below) keeps the zabd value.

Inputs: the zabd output; four frames of release/ZABD-PACK.json (the dark18 2026
vector, the 029 bounds, the row class with the plan brackets, and the member
scope), each rebuilt and checked by digest; and the caller's official files,
checked by hash. No fit runs, and the stage reads no label. blend, move, flags
and row_keys are copies of the original build's functions; move calls blend by
its local name.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

import assembly_replay as R

ROOT = R.ROOT
ID, Y = R.ID, R.Y
FLOOR, ALPHA = 60.0, 0.25
COLS = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]
#: role -> ZABD-PACK frame (2026 split, template order)
FRAMES = {"member": "ranking/vector/dark18", "bounds": "ranking/exploit_frame",
          "bundle": "ranking/bundle", "scope": "ranking/primary_frame"}


# ---------------------------------------------------------------- original functions

def blend(base: np.ndarray, pred: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    p = np.maximum(FLOOR, np.clip(pred, lo, hi))
    return np.maximum(FLOOR, np.clip(base + ALPHA * (p - base), lo, hi))


def move(base, v, lo, hi, m):
    c = base.copy()
    assert np.isfinite(v[m]).all()
    c[m] = blend(base[m], v[m], lo[m], hi[m])
    return c


def flags(v: pd.DataFrame):
    """u13 prep(): (rows with ap and dstr, the airport-day table with the long-taxi flag)."""
    v = v[(v.PHASE_mvt == "DEP")].copy()
    v["ap"] = v.ADEP_mvt.astype(str)
    v = v[v.ADES_mvt != v.ap].copy()
    v["tn"] = (v.MVT_TIME_UTC_mvt - v.AOBT_3_flt).dt.total_seconds()
    v["dstr"] = v.MVT_TIME_UTC_mvt.dt.strftime("%m-%d")
    g = v.groupby(["ap", "dstr"]).agg(rows=("tn", "size"), a3=("tn", "count"), n_long=("tn", lambda s: (s >= 3600).sum())).reset_index()
    g["share"] = g.n_long / g.a3
    g["snow"] = (g.n_long >= 8) & (g.share >= 0.02)
    return v, g


def row_keys(ids: np.ndarray, frame: pd.DataFrame) -> pd.Series:
    """airport|day key of each id (departures; take-off UTC day); every id must be found."""
    dep = frame[frame.PHASE_mvt == "DEP"].drop_duplicates(ID).set_index(ID)
    idx = pd.Index(dep.index.to_numpy("float64").astype("int64"))
    dep.index = idx
    k = (dep.ADEP_mvt.astype(str) + "|" + dep.MVT_TIME_UTC_mvt.dt.strftime("%m-%d")).reindex(ids)
    assert k.notna().all()
    return k


# ---------------------------------------------------------------- the stage

def p2_rows(ranking: pd.DataFrame, cls: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Typed rows without an NM flight join; the join must match the finite plan brackets."""
    joined = ranking["LOBT_flt"].notna().to_numpy()
    if not np.array_equal(joined, np.isfinite(lo) & np.isfinite(hi)):
        raise SystemExit("the NM join of the official file differs from the saved plan brackets")
    typed = ranking["AIRCRAFT_TYPE_mvt"].notna().to_numpy() & (cls != "noac")
    return typed & ~joined


def long_day_rows(ids: np.ndarray, raw: pd.DataFrame, expected: dict) -> tuple[np.ndarray, list[str]]:
    """Rows on a long-taxi airport-day; the day count by airport must equal `expected`."""
    _, g = flags(raw)
    days = g[g.snow]
    counts = {str(a): int(n) for a, n in days.groupby("ap").size().items()}
    if counts != expected:
        raise SystemExit(f"long-taxi days by airport {counts} differ from the registered {expected}")
    keys = sorted(days.ap + "|" + days.dstr)
    return row_keys(ids, raw).isin(set(keys)).to_numpy(), keys


def apply(base: np.ndarray, member: np.ndarray, lo: np.ndarray, hi: np.ndarray,
          p2: np.ndarray, long_rows: np.ndarray) -> np.ndarray:
    """The move on the P2 rows; every row of a long-taxi day keeps the base value."""
    out = move(base, member, lo, hi, p2)
    out[long_rows] = base[long_rows]
    return out


def compose(spec: dict, base_path: Path, data_root: Path) -> dict:
    """Apply the stage to the replayed base entry; return the bytes, the table and the facts."""
    pack_path = ROOT / spec["pack"]
    if R.sha256_file(pack_path) != spec["pack_sha256"]:
        raise SystemExit("assembly pack hash mismatch")
    if R.sha256_file(base_path) != spec["base_sha256"]:
        raise SystemExit("base entry differs from the registered base bytes")
    pack = json.loads(pack_path.read_text())
    official = R.official_tables(data_root, pack["official_sha256"])
    ranking = official["ranking"]
    frames = {}
    for role, key in FRAMES.items():
        frame = R.rebuild_frame(pack["frames"][key], ranking, len(ranking))
        if R.frame_digest(frame) != pack["frames"][key]["digest"]:
            raise SystemExit(f"rebuilt frame differs from the original input: {key}")
        frames[role] = frame
    base = pd.read_parquet(base_path)
    ids = base[ID].to_numpy("float64")
    rid = ids.astype("int64")
    for frame in (ranking, *frames.values()):
        if not np.array_equal(frame[ID].to_numpy("float64"), ids):
            raise SystemExit("stage inputs are not in the template order")
    rb = base[Y].to_numpy("float64")
    bundle, bounds = frames["bundle"], frames["bounds"]
    blo, bhi = bundle["lo"].to_numpy(float), bundle["hi"].to_numpy(float)
    p2 = p2_rows(ranking, bundle["cls"].to_numpy(), blo, bhi)
    if int(p2.sum()) != spec["p2_rows"]:
        raise SystemExit(f"{int(p2.sum())} P2 rows, not the registered {spec['p2_rows']}")
    if (p2 & frames["scope"]["member_scope"].to_numpy(bool)).any():
        raise SystemExit("a P2 row is in the dark18 member's own scope")
    d = frames["member"]
    member = pd.Series(d["pred_seq170"].to_numpy("float64"),
                       index=d[ID].to_numpy("float64").astype("int64")).reindex(rid).to_numpy()
    lo, hi = bounds["lower"].to_numpy(float), bounds["upper"].to_numpy(float)
    raw = pd.read_parquet(Path(data_root) / R.OFFICIAL_FILES["ranking"], columns=COLS)
    long_rows, days = long_day_rows(rid, raw, spec["long_taxi_days"])
    out = apply(rb, member, lo, hi, p2, long_rows)
    table = pd.DataFrame({ID: ids, Y: out})
    buffer = io.BytesIO()
    table.to_parquet(buffer, index=False)
    fin = np.isfinite(blo) & np.isfinite(bhi)
    shift = out - rb
    changed = shift != 0
    checks = {
        "finite": bool(np.isfinite(out).all()),
        "min_60": bool(out.min() >= FLOOR),
        "bounds_029_hold": bool(np.array_equal(np.clip(out, lo, hi), out)),
        "plan_brackets_hold": bool(((out[fin] >= blo[fin]) & (out[fin] <= bhi[fin])).all()),
        "rows_outside_P2_unchanged": bool(np.array_equal(out[~p2], rb[~p2])),
        "long_day_rows_unchanged": bool(np.array_equal(out[long_rows], rb[long_rows])),
        "changed_rows_as_registered": int(changed.sum()) == spec["changed_rows"],
    }
    if not all(checks.values()):
        raise SystemExit(f"stage checks failed: {sorted(k for k, ok in checks.items() if not ok)}")
    facts = {"p2_rows": int(p2.sum()), "long_taxi_days": days, "rows_on_long_taxi_days": int(long_rows.sum()),
             "p2_rows_on_long_taxi_days": int((p2 & long_rows).sum()), "changed_rows": int(changed.sum()),
             "mean_shift_changed_s": float(shift[changed].mean()),
             "rms_shift_changed_s": float(np.sqrt((shift[changed] ** 2).mean())), "checks": checks}
    return {"body": buffer.getvalue(), "table": table, "facts": facts}
