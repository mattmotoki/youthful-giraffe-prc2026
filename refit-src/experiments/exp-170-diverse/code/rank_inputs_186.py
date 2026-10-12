"""exp-186: the 2026 rows of one saved exp-170 input view, for the ranking or the final 2026 file.

The saved views (root, runway-006, runway-gap-014, time-day-Z-052, row-time-day-Z-054,
runway-time-day-Z-059, coherent-dark-Z-annual-064) hold the 2,085,047 departures of 2025 and the 344,841
departures of ranking.parquet. Each builder of the chain pins its parent MANIFEST by SHA-256 and records its
own code hash, so the chain stays unchanged; this script is the final-mode caller. It rebuilds only the 2026
rows of one view: the 2026 departures of the mode's ranking file (prc.data), the train-only centres, scales
and vocabularies of the view MANIFEST, the view's row order (airport-day, or airport-day-runway) and its
column rules (same-runway gaps of prepare_gap.py; columns 57/116 zero in the Z form; the dark-day mask
touches no 2026 row). A 2026 airport-day holds no 2025 row, so each window over these rows equals the
window over the same rows in the saved view. No label is read and no encoder is fitted.

    PRC_RANKING_MODE=final python code/rank_inputs_186.py --view <saved view> --clock <frame12> \
        --track <track12> --out <dir>

Ranking mode: the view must record the SHA-256 of the ranking, clock and track files; the 2026 arrays are
compared with the view's 2026 rows and must be byte-equal (rows, order, airport-day bounds, numeric,
category, anchor). Final mode: rows = n_ranking(), months = rank_months(), departures per month =
ranking_spec().month_departures; every 2026 month must join the clock and track tables (at least half of
the mean Jan/Jul share); the Jan/Jul rows are compared with the view as a record only (the final file
moves some Jan/Jul values). --dry-run builds and checks and writes MANIFEST.json only.

--no-clock all|new gives rows the encoding of a row with no clock or track observation: every OBS and TRACK
value missing, so the 38 clock inputs (obs_, trk_, clock_, opdi_, nm_minus_obs) have value 0 and flag 0. It is
the encoding of a departure that frame12/track12 do not hold, and the dark-day mask of
prepare_coherent_dark_053.py writes the same zeros. `all`: every 2026 row, no clock or track table read. `new`:
the rows of the 2026 months outside the board months (Feb/Jun in the final mode); the board months join the
tables. In the ranking mode the view's own no-clock rows must be byte-equal, and every other row must be
byte-equal outside the 76 clock columns; view_equal.npy marks the rows whose inputs are byte-equal to the
view, for the strict compare of predict_rank_186.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2])]  # this folder's modules first, then the repo root's prc
import prc.data as D  # noqa: E402
from prepare import CATS, ID, OBS, TIME, TIMES, TRACK, numeric, sha  # noqa: E402

if not hasattr(D, "rank_months"):
    raise SystemExit("prc/data.py has no ranking modes (rank_months); update this checkout to exp-186 d351a474c or later")

WIDTH = 59
TIME_COLUMNS = [57, 116]  # seconds_in_day and its finite flag
RUNWAY_ORDER = ["ADEP_mvt", "date", "RUNWAY_mvt", TIME, ID]
CLOCK_PREFIXES = ("obs_", "trk_", "clock_", "opdi_")


def clock_columns(names):
    """The 38 inputs that come from the clock and track tables, and their 38 flags (the dark-day set of 053)."""
    values = [i for i, n in enumerate(names) if n.startswith(CLOCK_PREFIXES) or n == "nm_minus_obs"]
    assert len(values) == 38 and not set(values) & set(TIME_COLUMNS)
    return values + [i + WIDTH for i in values]


def view_form(rec):
    """Order, gap rule, time form and dark mask of a saved view, from its MANIFEST."""
    order = "airport-day-runway" if rec.get("group") == "airport-day-runway" else "airport-day"
    gaps = "gap_columns" in rec
    if gaps:
        assert order == "airport-day-runway" and set(rec["gap_columns"]) == {"gap_prev", "gap_next"}
    form = rec.get("time_day_form") or rec.get("fixed_time_form")
    if form not in (None, "Z"):
        raise SystemExit(f"time form {form!r}: this script builds the Z form (columns 57/116 zero) only")
    if form == "Z":
        assert rec["changed_numeric_columns"] == TIME_COLUMNS or rec.get("fixed57_116_exact")
    dark = "darkday_training_domain" in rec
    if dark:
        assert rec["nontraining_rows_exact"] and rec["fixed57_116_exact"]
    return {"order": order, "same_runway_gaps": gaps, "time_form": form, "dark_training_mask": dark}


def source_sha(rec, name):
    hits = [v for k, v in rec["sources"].items() if Path(k).name == name]
    assert len(hits) == 1, name
    return hits[0]


def rank_frame(training, ranking, clock, track, no_clock_months=()):
    """prepare.py main() steps on the 2026 departures; returns them in airport-day order. Rows of
    `no_clock_months` get every OBS and TRACK value missing; with no tables every row gets them."""
    cols = [ID, "PHASE_mvt", "TAXITIME_SEC_mvt"] + TIMES + CATS[:-1]
    probes = []
    for p in training:
        f = pd.read_parquet(p, columns=cols)
        f = f[f.PHASE_mvt == "DEP"].copy()
        for c in TIMES:
            f[c] = pd.to_datetime(f[c], utc=True)
        f["year"], f["month"] = 2025, f[TIME].dt.month
        # One row per 2025 part: pd.concat then gives the column types of the saved full frame.
        probes.append(f.head(1))
    f = pd.read_parquet(ranking, columns=[c for c in cols if c != "TAXITIME_SEC_mvt"])
    f = f[f.PHASE_mvt == "DEP"].copy()
    for c in TIMES:
        f[c] = pd.to_datetime(f[c], utc=True)
    f["TAXITIME_SEC_mvt"] = np.nan
    f["year"], f["month"] = 2026, f[TIME].dt.month
    f = pd.concat(probes + [f], ignore_index=True).iloc[len(probes):].reset_index(drop=True)
    f[ID] = f[ID].astype("int64")
    assert not f[ID].duplicated().any() and (f.year == 2026).all()
    share = {}
    for p, cs, name in ([] if clock is None else [(clock, OBS, "clock"), (track, TRACK, "track")]):
        t = pd.read_parquet(p, columns=[ID] + cs)
        t[ID] = t[ID].astype("int64")
        assert not t[ID].duplicated().any()
        share[name] = {int(m): float(v) for m, v in f[ID].isin(t[ID]).groupby(f.month).mean().round(4).items()}
        f = f.merge(t, on=ID, how="left", validate="one_to_one")
    if clock is None:
        for c in OBS + TRACK:
            f[c] = np.nan
    hit = f.month.isin(list(no_clock_months)).to_numpy()
    if hit.any():
        f[OBS + TRACK] = f[OBS + TRACK].astype(float)
        f.loc[hit, OBS + TRACK] = np.nan
    f["date"] = f[TIME].dt.strftime("%Y-%m-%d")
    f = f.sort_values(["ADEP_mvt", "date", TIME, ID], kind="stable").reset_index(drop=True)
    f["group"] = pd.factorize(pd.MultiIndex.from_frame(f[["ADEP_mvt", "date"]]), sort=False)[0]
    f["month_cat"] = f.month.astype(str)
    return f, share


def arrays(f, rec):
    """prepare.py encoding with the view's train-only centres, scales and vocabularies."""
    z = numeric(f)
    assert list(z) == rec["numeric"] and len(rec["numeric"]) == WIDTH and rec["category"] == CATS
    assert rec["numeric"][57] == "seconds_in_day"
    center, scale = np.array(rec["center"], dtype="float64"), np.array(rec["scale"], dtype="float64")
    val = np.sign(z.to_numpy(float)) * np.log1p(np.abs(z.to_numpy(float)))
    num = np.concatenate([np.nan_to_num((val - center) / scale, nan=0.).clip(-12, 12),
                          np.isfinite(val).astype(float)], axis=1).astype("float32")
    cat = []
    for c in CATS:
        s = f[c].astype("string").fillna("__MISSING__")
        cat.append(s.map({v: i + 1 for i, v in enumerate(rec["vocab"][c])}).fillna(0).to_numpy("int32"))
    cat = np.stack(cat, axis=1)
    nm = (f[TIME] - f.AOBT_3_flt).dt.total_seconds().to_numpy(float)
    anchor = np.where(np.isfinite(nm), np.clip(nm, 60., 43200.), 900.).astype("float32")
    labels = f.TAXITIME_SEC_mvt.to_numpy("float64")
    meta = f[[ID, "ADEP_mvt", TIME, "year", "month", "date", "group"]].copy()
    meta["aobt"], meta["aircraft"] = f.AOBT_3_flt.notna(), f.AIRCRAFT_TYPE_mvt.notna()
    meta["plan"] = f.LOBT_flt.notna()
    return meta, num, cat, anchor, labels


def runway_order(meta, arrays_, training, ranking):
    """prepare_runway.py: same values, airport-day-runway order and groups."""
    parts = []
    for p in list(training) + [ranking]:
        f = pd.read_parquet(p, columns=["MVT_ID_mvt", "RUNWAY_mvt", "PHASE_mvt"])
        f = f.loc[f.PHASE_mvt == "DEP", ["MVT_ID_mvt", "RUNWAY_mvt"]]
        parts.append(f if p == ranking else f.head(1))
    rw = pd.concat(parts).set_index("MVT_ID_mvt")
    rw = rw[rw.index.isin(meta.MVT_ID_mvt)]
    assert not rw.index.duplicated().any()
    meta = meta.copy()
    meta["RUNWAY_mvt"] = rw.RUNWAY_mvt.reindex(meta.MVT_ID_mvt).astype("string").fillna("__MISSING__").to_numpy()
    meta["original_row"] = np.arange(len(meta))  # row in this build's airport-day order (2026 rows only)
    meta = meta.sort_values(RUNWAY_ORDER, kind="stable").reset_index(drop=True)
    order = meta.original_row.to_numpy()
    assert np.array_equal(np.sort(order), np.arange(len(meta)))
    meta["group"] = pd.factorize(pd.MultiIndex.from_frame(meta[["ADEP_mvt", "date", "RUNWAY_mvt"]]), sort=False)[0]
    return meta, [x[order] for x in arrays_]


def same_runway_gaps(meta, num, rec):
    """prepare_gap.py: gap_prev/gap_next within the runway group, with the view's train-only scaling."""
    times = meta[TIME]
    gaps = {"gap_prev": times.groupby(meta.group, sort=False).diff().dt.total_seconds().clip(0, 86400),
            "gap_next": -times.groupby(meta.group, sort=False).diff(-1).dt.total_seconds().clip(-86400, 0)}
    for name, value in gaps.items():
        col = rec["numeric"].index(name)
        v = np.sign(value.to_numpy(float)) * np.log1p(np.abs(value.to_numpy(float)))
        center, scale = rec["gap_columns"][name]["center"], rec["gap_columns"][name]["scale"]
        assert center == rec["center"][col] and scale == rec["scale"][col]
        num[:, col] = np.nan_to_num((v - center) / scale, nan=0.).clip(-12, 12).astype("float32")
        num[:, col + WIDTH] = np.isfinite(v).astype("float32")
    if "timestamp_unit" in rec:
        assert str(times.dtype) == rec["timestamp_unit"], (str(times.dtype), rec["timestamp_unit"])


def bounds(groups):
    return np.flatnonzero(groups[1:] != groups[:-1]) + 1


def compare(view, rec, meta, num, cat, anchor, strict, no_clock=None):
    """The build against the view's 2026 rows: exact in the ranking mode, a record in the final mode. With
    `no_clock` (a row mask of this build) the ranking mode is exact on the view's no-clock rows and, on the
    other rows, outside the clock columns."""
    for name in ["rows.parquet", "numeric.npy", "category.npy", "anchor.npy"]:
        assert sha(view / name) == rec["files"][name], "view file differs from its MANIFEST: " + name
    old = pd.read_parquet(view / "rows.parquet", columns=[ID, "year", "group"])
    rows = np.flatnonzero(old.year.to_numpy() == 2026)
    ids = old[ID].to_numpy("int64")[rows]
    rec = {"view_2026_rows": int(len(rows)), "rows": int(len(meta))}
    if strict:
        assert np.array_equal(ids, meta[ID].to_numpy("int64")), "2026 row order differs from the view"
        assert np.array_equal(bounds(old.group.to_numpy()[rows]), bounds(meta.group.to_numpy())), "airport-day bounds differ"
        if no_clock is not None and no_clock.any():
            cols = clock_columns(rec_names(view))
            other = [c for c in range(2 * WIDTH) if c not in cols]
            ref = np.load(view / "numeric.npy", mmap_mode="r")[rows]
            assert (num[np.ix_(no_clock, cols)] == 0).all()
            dark = (ref[:, cols[38:]] == 0).all(axis=1)  # the view's no-clock rows: every clock flag 0
            assert np.array_equal(ref[dark & no_clock], num[dark & no_clock]), "no-clock rows differ from the view"
            assert np.array_equal(ref[~no_clock], num[~no_clock]), "clock rows differ from the view"
            assert np.array_equal(ref[:, other], num[:, other]), "non-clock columns differ from the view"
            equal = (ref == num).all(axis=1)
            rec["numeric"] = {"no_clock_rows": int(no_clock.sum()), "view_no_clock_rows": int((dark & no_clock).sum()),
                              "byte_equal_rows": int(equal.sum()),
                              "non_clock_columns": "byte-equal on every row"}
            names = [("category", cat), ("anchor", anchor)]
        else:
            equal = None
            names = [("numeric", num), ("category", cat), ("anchor", anchor)]
        for name, new in names:
            ref = np.load(view / (name + ".npy"), mmap_mode="r")
            assert ref.dtype == new.dtype and np.array_equal(ref[rows], new), name + " differs from the view"
            rec[name] = "byte-equal"
        return rec, equal
    pos = pd.Index(meta[ID].to_numpy("int64")).get_indexer(ids)
    hit = pos >= 0
    rec["view_rows_in_build"] = int(hit.sum())
    for name, new in [("numeric", num), ("category", cat), ("anchor", anchor)]:
        ref = np.load(view / (name + ".npy"), mmap_mode="r")[rows[hit]]
        mine = new[pos[hit]]
        equal = (ref == mine).reshape(len(ref), -1).all(axis=1)
        rec[name + "_rows_equal"] = int(equal.sum())
        if name == "numeric":
            rec["numeric_max_abs_diff"] = float(np.abs(ref.astype("float64") - mine).max()) if len(ref) else 0.
    return rec, None


def rec_names(view):
    return json.loads((view / "MANIFEST.json").read_text())["numeric"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", type=Path, required=True, help="saved exp-170 input view (MANIFEST.json and arrays)")
    ap.add_argument("--clock", type=Path, help="OBS table (prepare.py --clock); not read with --no-clock all")
    ap.add_argument("--track", type=Path, help="TRACK table (prepare.py --track); not read with --no-clock all")
    ap.add_argument("--no-clock", choices=["all", "new"], help="no-clock encoding for all 2026 rows, or for the"
                    " months outside the board months")
    ap.add_argument("--raw", type=Path, help="folder of the twelve 2025 training files (default prc.data.data_dir())")
    ap.add_argument("--ranking", type=Path, help="ranking mode only: a ranking file whose SHA-256 the view records")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="build and check; write MANIFEST.json only")
    a = ap.parse_args()
    if a.out.exists():
        raise RuntimeError("output directory already exists")
    mode = D.ranking_mode()
    spec = D.ranking_spec()
    view_sha = sha(a.view / "MANIFEST.json")
    rec = json.loads((a.view / "MANIFEST.json").read_text())
    form = view_form(rec)
    tables = a.no_clock != "all"
    if tables != (a.clock is not None and a.track is not None):
        raise SystemExit("give --clock and --track, except with --no-clock all (then give neither)")
    board = set(D.rank_month_numbers("ranking"))
    no_clock_months = set(range(1, 13)) if a.no_clock == "all" else \
        (set(range(1, 13)) - board if a.no_clock == "new" else set())
    raw = a.raw or D.data_dir()
    training = sorted(raw.glob("training_2025-*.parquet"))
    assert len(training) == 12
    for p in training:
        assert sha(p) == source_sha(rec, p.name), p.name
    if mode == "final":
        if a.ranking is not None:
            raise SystemExit("--ranking is for the ranking mode; the final mode reads prc.data.ranking_file()")
        ranking = D.ranking_file()
    else:
        ranking = a.ranking or D.ranking_file()
        assert sha(ranking) == source_sha(rec, "ranking.parquet"), "the view was built from another ranking file"
    ranking_sha = sha(ranking)
    mode_file = ranking_sha == spec.ranking_sha256
    side = list(rec["sources"].items())[-2:]  # prepare.py records the clock, then the track table
    if mode == "ranking" and tables:
        assert sha(a.clock) == side[0][1] and sha(a.track) == side[1][1], "clock/track tables differ from the view"

    f, share = rank_frame(training, ranking, a.clock, a.track, no_clock_months)
    no_clock_rows = f.month.isin(list(no_clock_months)).to_numpy()
    months = sorted(f.month.unique().tolist())
    by_month = {f"2026-{m:02d}": int(n) for m, n in f.groupby("month").size().items()}
    if mode == "final":
        assert len(f) == D.n_ranking(), (len(f), D.n_ranking())
        assert tuple(months) == D.rank_month_numbers(), months
        assert by_month == spec.month_departures, by_month
        for name, s in share.items():
            base = np.mean([s.get(1, 0.), s.get(7, 0.)])
            low = {m: v for m, v in s.items() if v < .5 * base and m not in no_clock_months}
            assert not low, f"{name} table misses 2026 months {low} (Jan/Jul mean share {base:.3f})"
    else:
        view_rank = rec["rows"] - rec["train_rows"] - rec["held_rows"]
        assert len(f) == view_rank, (len(f), view_rank)
        if mode_file:
            assert len(f) == D.n_ranking() and tuple(months) == D.rank_month_numbers()

    meta, num, cat, anchor, labels = arrays(f, rec)
    del f
    if form["order"] == "airport-day-runway":
        meta, (num, cat, anchor, labels, no_clock_rows) = runway_order(
            meta, [num, cat, anchor, labels, no_clock_rows], training, ranking)
    if form["same_runway_gaps"]:
        same_runway_gaps(meta, num, rec)
    if form["time_form"] == "Z":
        num[:, TIME_COLUMNS] = 0.
    assert np.isfinite(num).all() and np.isnan(labels).all()
    assert num.shape == (len(meta), 2 * WIDTH) and cat.shape == (len(meta), len(CATS))
    parity, equal = compare(a.view, rec, meta, num, cat, anchor, strict=mode == "ranking",
                            no_clock=no_clock_rows if a.no_clock else None) \
        if (a.view / "rows.parquet").exists() else (None, None)
    if mode == "ranking":
        assert parity is not None, "the ranking mode compares with the view's arrays"

    a.out.mkdir(parents=True)
    if not a.dry_run:
        meta.to_parquet(a.out / "rows.parquet", index=False)
        for name, x in [("numeric", num), ("category", cat), ("anchor", anchor), ("labels", labels)]:
            np.save(a.out / (name + ".npy"), x, allow_pickle=False)
        if equal is not None:  # ranking mode with --no-clock: rows whose inputs are byte-equal to the view
            np.save(a.out / "view_equal.npy", equal, allow_pickle=False)
    out = {"what": "2026 rows of one exp-170 input view; no label, no fitted encoder (exp-186 final mode)",
           "ranking_mode": mode, "dry_run": a.dry_run,
           "view": str(a.view.resolve()), "view_manifest_sha256": view_sha, "view_form": form,
           "group": form["order"], "numeric": rec["numeric"], "category": rec["category"],
           "vocab": rec["vocab"], "center": rec["center"], "scale": rec["scale"],
           "ranking_file": str(ranking), "ranking_file_sha256": ranking_sha, "ranking_file_is_mode_file": mode_file,
           "sources": {**{str(p): source_sha(rec, p.name) for p in training}, str(ranking): ranking_sha,
                       **({str(a.clock): sha(a.clock), str(a.track): sha(a.track)} if tables else {})},
           "rows": int(len(meta)), "rows_by_month": by_month, "join_share_by_month": share,
           "time_dtype": str(meta[TIME].dtype), "view_parity": parity,
           "no_clock": {"option": a.no_clock, "months": sorted(no_clock_months & set(months)),
                        "rows": int(no_clock_rows.sum()),
                        "rule": "OBS and TRACK values missing: 38 clock inputs value 0, flag 0"},
           "source_code_sha256": sha(__file__), "prepare_code_sha256": sha(HERE / "prepare.py"),
           "files": {p.name: sha(p) for p in sorted(a.out.iterdir())}}
    (a.out / "MANIFEST.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({"rows": out["rows"], "by_month": by_month, "view_parity": parity, "form": form}), flush=True)


if __name__ == "__main__":
    main()
