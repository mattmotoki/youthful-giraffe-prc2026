"""Part `head` of exp-190: the merged HB3u `jjrank` fit on the template departures of the ranking mode.

    EXP159_SET=12 $PY experiments/exp-190-final-clock/code/head_part.py --fits <fits dir> --out <head.parquet>

(run_chain.sh step `head` sets the environment.) One row per departure of the mode's template, in template
order (final mode: `final_submitting.parquet`, 670,790 rows; ranking mode: `submitting.parquet`, 344,841 rows):
`MVT_ID_mvt` (int64), `pred_HB3u` (float64: the fit's value on the frame's 2026 clock rows, NaN on every other
row, no floor) and `clock` (bool: a frame clock row). The released vector (ZABD-PACK `ranking/vector/head`)
came through exp-159 `pack.py`, which copies the fit's values to the L reference rows unchanged; this part
takes them from the fit directly (SPEC.md, design row "pack"). Checks: the fit passes `pilot.load`; its ids are
unique template departures; they are the frame's 2026 rows; every value is finite. Write-once: the part and
`<part>.json` next to it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
import pilot as P                     # noqa: E402  (path set in common)

STAGE = "HB3u"


def template_file() -> Path:
    if C.RANKING_MODE == "ranking":
        return C.RAW / "submitting.parquet"
    from prc import data as D         # the SOURCE tree's prc (ranking modes)
    return D.submitting_file(C.RANKING_MODE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rec_path = a.out.with_suffix(".json")
    C.refuse_existing(a.out, rec_path)
    if C.SET != "12":
        raise SystemExit("EXP159_SET=12: the HB3u frame")
    t0 = time.time()
    fit_path = a.fits / f"{STAGE}-jjrank.parquet"
    fit = P.load(fit_path, "jjrank")
    fit_rec = json.loads(fit_path.with_suffix(".json").read_text())
    col = f"pred_{STAGE}"
    tpl_path = template_file()
    tpl = pd.read_parquet(tpl_path, columns=[C.ID])
    ids = tpl[C.ID].to_numpy()
    if not np.array_equal(ids, np.round(ids)) or pd.Series(ids).duplicated().any():
        raise SystemExit("template ids: not whole numbers, or repeated")
    ids = ids.astype("int64")
    if C.N_DEPARTURES is not None and len(ids) != C.N_DEPARTURES:
        raise SystemExit(f"template rows {len(ids)}, expected {C.N_DEPARTURES}")
    fid = fit[C.ID].to_numpy("int64")
    if pd.Series(fid).duplicated().any():
        raise SystemExit("repeated fit ids")
    frame = pd.read_parquet(C.FRAME, columns=[C.ID, "month"])
    want = set(frame.loc[frame.month.isin(C.MONTHS_2026), C.ID].astype("int64"))
    if set(fid) != want:
        raise SystemExit(f"the fit scores {len(fid)} rows, the frame has {len(want)} 2026 clock rows")
    val = fit[col].to_numpy("float64")
    if not np.isfinite(val).all():
        raise SystemExit("a fit row has no value")
    pos = pd.Index(ids).get_indexer(fid)
    if (pos < 0).any():
        raise SystemExit(f"{int((pos < 0).sum())} fit ids are not template departures")
    out = pd.DataFrame({C.ID: ids, col: np.full(len(ids), np.nan), "clock": np.zeros(len(ids), bool)})
    out.loc[pos, col] = val
    out.loc[pos, "clock"] = True
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    rk = pd.read_parquet(C.RANKING, columns=[C.ID, C.TIME])
    rk[C.ID] = rk[C.ID].astype("int64")
    month = pd.to_datetime(rk.set_index(C.ID)[C.TIME].reindex(ids).to_numpy(), utc=True).strftime("%Y-%m")
    if pd.isna(month).any():
        raise SystemExit("a template departure is not in the ranking file")
    by = out.assign(month=month).groupby("month")
    rec = {"what": "exp-190 part head: merged HB3u jjrank fit on the template departures (NaN outside the clock rows)",
           "ranking_mode": C.RANKING_MODE, "rows": int(len(out)), "columns": list(out.columns),
           "clock_rows": int(out.clock.sum()), "clock_rows_by_month": by["clock"].sum().astype(int).to_dict(),
           "mean_by_month": by[col].mean().round(3).to_dict(),
           "template": {"path": str(tpl_path), "sha256": C.sha(tpl_path)},
           "fit": {"path": str(fit_path), "sha256": C.sha(fit_path), "output_sha256": fit_rec.get("output_sha256"),
                   "seeds": fit_rec["fit"]["seeds"], "merged_from": fit_rec.get("merged_from")},
           "frame_sha256": C.sha(C.FRAME), "output_sha256": C.sha(a.out), "seconds": round(time.time() - t0, 1),
           "code_sha256": {"head_part.py": C.sha(__file__), "common.py": C.sha(C.__file__)}}
    C.write_json(rec_path, rec)
    print(json.dumps({k: rec[k] for k in ("rows", "clock_rows", "clock_rows_by_month", "mean_by_month", "output_sha256")}))


if __name__ == "__main__":
    main()
