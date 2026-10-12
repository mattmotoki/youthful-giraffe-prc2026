"""Build the frozen inputs of protocol week-pair-1 (exp-146).

    PRC_SOURCE_ROOT=/path/to/competitions/prc-data-challenge-2026 \
        $PRC_SOURCE_ROOT/.venv/bin/python experiments/exp-146-validation/code/build_frozen.py

Writes experiments/exp-146-validation/frozen/:
  rows.parquet         one row per 2025 January, July and December departure: id, airport,
                       month, UTC date, pair, class (clock / ordinary / noac), aobt, clean
  cells.csv            ranking departures per month x airport x class (N), W = N / 344,841,
                       and the scored 2025 rows per cell (n)
  coverage.csv         clock coverage per airport-month, 2025 and 2026, and the removal rate
  clock_loss_days.csv  the frozen dark airport-days of each clock-loss scheme
  MANIFEST.json        sha256 of each file above, of every input, and of the resampling draws

Write-once: the run builds into a temporary directory. A file that already exists must come
out byte-identical (then the run is a reproduction check); a changed file stops the run,
because week-pair-1 is frozen and a new protocol needs a new frozen directory.

    --revise-manifest  lets MANIFEST.json alone change (the data files must still come out
                       byte-identical) and records the replaced manifest's sha256 in
                       MANIFEST_REVISIONS. Revision 1 (2026-09-30, exp-146 review): inner(k, j)
                       no longer scores fold k's buffer days, so the splits fingerprint changed.

Class rule (the served rule of v19..v32): noac = no AIRCRAFT_TYPE_mvt; clock = a known type
with an observed surface clock. For 2025 rows the clock is exp-046's ungated block with the
gated mode-4 matches kept exactly as chainfold.restrict keeps them (gap > 62.405 s,
|P - AOBT| <= 1,299 s, runway agrees); for 2026 rows it is obs_has_clock of the v25 class
export. Reads no label.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
SRC = Path(os.environ.get("PRC_SOURCE_ROOT", "/path/to/competitions/prc-data-challenge-2026"))
os.environ.setdefault("PRC_DATA_DIR", str(SRC / "data/raw"))

from prc import data as D          # noqa: E402
from prc import gscore as GS        # noqa: E402
from prc import weekpair as WP      # noqa: E402

OUT = EXP / "frozen"
UNGATED = {  # exp-046 chainfold's ungated blocks (experiments/exp-046-surfacefit/work/)
    "2025-01": "block-2025-01-cb4d6289cb421aa2-ungated.parquet",
    "2025-07": "block-2025-07-8179f74417c8cb99-ungated.parquet",
    "2025-12": "block-2025-12-d35b7171d7062341-ungated.parquet",
}
#: kept mode-4 rows per month, exp-046 analysis/chain-fold.json census_by_month
KEPT_MODE4 = {"2025-01": 742, "2025-07": 897, "2025-12": 858}
GAP_CUT_S, THR_S = 62.405, 1299.0
RANKING_CLASS = {"2026-01": "artifacts/exp-046/class/2026-01-v25.parquet",
                 "2026-07": "artifacts/exp-046/class/2026-07-v25.parquet"}
RANKING_TOTALS = {"clock": 196_256, "ordinary": 148_197, "noac": 388}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cls_of(noac: np.ndarray, clock: np.ndarray) -> np.ndarray:
    return np.where(noac, "noac", np.where(clock, "clock", "ordinary"))


def rows_2025(inputs: dict) -> pd.DataFrame:
    parts = []
    for m, fn in UNGATED.items():
        raw = D.load_month(m, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
                                       "AIRCRAFT_TYPE_mvt", "AOBT_3_flt"])
        inputs[str(D.month_file(m))] = sha(D.month_file(m))
        dep = raw[raw.PHASE_mvt == "DEP"].copy()
        p = SRC / "experiments/exp-046-surfacefit/work" / fn
        inputs[str(p)] = sha(p)
        b = pd.read_parquet(p, columns=["MVT_ID_mvt", "obs_has_clock", "match_mode", "chain_gap_s",
                                        "chain_abs_dt_offblock", "chain_runway_agrees"])
        m4 = (b.match_mode.fillna(-1) == 4).to_numpy()
        keep = (m4 & (b.chain_gap_s > GAP_CUT_S).to_numpy()
                & (b.chain_abs_dt_offblock <= THR_S).to_numpy()
                & (b.chain_runway_agrees == 1.0).to_numpy())
        if int(keep.sum()) != KEPT_MODE4[m]:
            raise SystemExit(f"{m}: {int(keep.sum())} kept mode-4 rows, recorded {KEPT_MODE4[m]}")
        b["clock"] = b.obs_has_clock.fillna(False).to_numpy(bool) & ~(m4 & ~keep)
        # cross-check: without mode 4, the class is exp-047's shipped block
        p47 = SRC / f"artifacts/exp-047/block/{m}.parquet"
        inputs[str(p47)] = sha(p47)
        b47 = pd.read_parquet(p47, columns=["MVT_ID_mvt", "obs_has_clock"]).set_index("MVT_ID_mvt")
        base_clock = pd.Series(b.clock.to_numpy() & ~(m4 & keep), index=b.MVT_ID_mvt)
        if not (base_clock.reindex(b47.index) == b47.obs_has_clock).all():
            raise SystemExit(f"{m}: the non-mode-4 clock differs from exp-047's block")
        d = dep.merge(b[["MVT_ID_mvt", "clock"]], on="MVT_ID_mvt", how="left", validate="1:1")
        if d.clock.isna().any():
            raise SystemExit(f"{m}: {int(d.clock.isna().sum())} departures have no block row")
        days = WP.day_of(d)
        noac = d.AIRCRAFT_TYPE_mvt.isna().to_numpy()
        parts.append(pd.DataFrame({
            "MVT_ID_mvt": d.MVT_ID_mvt.to_numpy("float64").astype("int64"),
            "airport": d.ADEP_mvt.astype(str).to_numpy(),
            "month": np.int8(int(m[-2:])),
            "date": days.astype(str),
            "pair": WP.pair_of(days) if m != "2025-12" else np.zeros(len(d), np.int8),
            "cls": cls_of(noac, d.clock.to_numpy(bool)),
            "aobt": d.AOBT_3_flt.notna().to_numpy(),
        }))
    rows = pd.concat(parts, ignore_index=True)
    ex = {int(i) for mm in D.UNREACHED_ROLLOVER for i in D.excluded_ids(mm)}
    rows["clean"] = ~rows.MVT_ID_mvt.isin(ex)
    if rows.MVT_ID_mvt.duplicated().any():
        raise SystemExit("duplicate ids")
    bad = rows[(rows.month != 12) & (rows.pair == 0)]
    if len(bad):
        raise SystemExit(f"{len(bad)} January/July rows fall outside every pair")
    return rows


def ranking_cells(inputs: dict) -> pd.DataFrame:
    rk = D.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
                                 "AIRCRAFT_TYPE_mvt"])
    inputs[str(D.data_dir() / "ranking.parquet")] = sha(D.data_dir() / "ranking.parquet")
    rk = rk[rk.PHASE_mvt == "DEP"]
    cl = []
    for m, f in RANKING_CLASS.items():
        inputs[str(SRC / f)] = sha(SRC / f)
        cl.append(pd.read_parquet(SRC / f, columns=["MVT_ID_mvt", "obs_has_clock"]))
    rk = rk.merge(pd.concat(cl), on="MVT_ID_mvt", how="left", validate="1:1")
    if rk.obs_has_clock.isna().any() or len(rk) != GS.N_RANKING:
        raise SystemExit("ranking departures and the v25 class export do not match")
    rk["cls"] = cls_of(rk.AIRCRAFT_TYPE_mvt.isna().to_numpy(), rk.obs_has_clock.to_numpy(bool))
    tot = rk.cls.value_counts().to_dict()
    if tot != RANKING_TOTALS:
        raise SystemExit(f"ranking class totals {tot} != {RANKING_TOTALS}")
    rk["month"] = rk.MVT_TIME_UTC_mvt.dt.month.astype(int)
    N = rk.groupby(["month", "ADEP_mvt", "cls"]).size().rename("N").reset_index()
    return N.rename(columns={"ADEP_mvt": "airport"})


FILES = ("rows.parquet", "cells.csv", "coverage.csv", "clock_loss_days.csv", "MANIFEST.json")
#: every manifest revision: the data files are unchanged, the manifest it replaces, and why
MANIFEST_REVISIONS = [
    {"revision": 1, "date": "2026-09-30",
     "replaces_manifest_sha256": "6f422b435272a71dc8b185e43ea43ad25d76c5af50b02f0a11d8ccad3df2605a",
     "replaces_commit": "c61f6c54",
     "why": "exp-146 review: inner(k, j).held drops fold k's buffer days, so every calibration "
            "row of fold k is a label row of outer(k); the splits fingerprint changed"},
]


def settle(stage: Path, revise_manifest: bool = False) -> None:
    """Copy the staged files into OUT; refuse any change to an existing frozen file.

    With `revise_manifest`, MANIFEST.json alone may change, and only to the manifest whose
    sha256 the last MANIFEST_REVISIONS entry says it replaces."""
    changed = [f for f in FILES if (OUT / f).exists() and sha(OUT / f) != sha(stage / f)]
    if revise_manifest and changed == ["MANIFEST.json"]:
        old = sha(OUT / "MANIFEST.json")
        if old != MANIFEST_REVISIONS[-1]["replaces_manifest_sha256"]:
            raise SystemExit(f"MANIFEST.json {old} is not the manifest revision "
                             f"{MANIFEST_REVISIONS[-1]['revision']} replaces")
        shutil.copy2(stage / "MANIFEST.json", OUT / "MANIFEST.json")
        print(f"MANIFEST.json: revised {old} -> {sha(OUT / 'MANIFEST.json')}")
        changed = []
    if changed:
        raise SystemExit(f"frozen files would change: {changed}. week-pair-1 is frozen; "
                         "a new protocol needs a new frozen directory")
    OUT.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if (OUT / f).exists():
            print(f"{f}: reproduced byte for byte")
        else:
            shutil.copy2(stage / f, OUT / f)
            print(f"{f}: written")


def main() -> None:
    revise = "--revise-manifest" in sys.argv[1:]
    stage = Path(tempfile.mkdtemp(prefix="exp146-frozen-"))
    inputs: dict = {}
    rows = rows_2025(inputs)
    N = ranking_cells(inputs)
    n = (rows[rows.month != 12].groupby(["month", "airport", "cls"]).size().rename("n").reset_index())
    cells = N.merge(n, on=["month", "airport", "cls"], how="outer").fillna(0)
    cells[["N", "n"]] = cells[["N", "n"]].astype(int)
    cells["W"] = cells.N / GS.N_RANKING
    cells["supported"] = cells.n >= GS.MIN_SUPPORT
    cells = cells.sort_values(["month", "airport", "cls"]).reset_index(drop=True)

    # clock coverage and the frozen dark airport-days
    cov = []
    for m in (1, 7):
        for a in GS.AIRPORTS:
            r25 = rows[(rows.month == m) & (rows.airport == a) & (rows.cls != "noac")]
            c26 = cells[(cells.month == m) & (cells.airport == a)]
            k26 = int(c26[c26.cls == "clock"].N.sum())
            n26 = int(c26[c26.cls.isin(["clock", "ordinary"])].N.sum())
            k25, n25 = int((r25.cls == "clock").sum()), len(r25)
            c25_, c26_ = (k25 / n25 if n25 else 0.0), (k26 / n26 if n26 else 0.0)
            rate = WP.removal_rate(c25_, c26_)
            cov.append({"month": m, "airport": a, "clock_2025": k25, "known_2025": n25,
                        "cov_2025": c25_, "clock_2026": k26, "known_2026": n26, "cov_2026": c26_,
                        "rate": rate, "n_dark_days": len(WP.dark_days(a, f"2025-{m:02d}", rate))})
    cov = pd.DataFrame(cov)
    dark = []
    for r in cov.itertuples():
        for day in WP.dark_days(r.airport, f"2025-{r.month:02d}", r.rate):
            dark.append({"scheme": "coverage", "airport": r.airport, "date": day})
    for day in WP.dark_days("LIRF", "2025-07", 1.0):
        dark.append({"scheme": "lirf_july", "airport": "LIRF", "date": day})
    dark = pd.DataFrame(dark)

    rows.to_parquet(stage / "rows.parquet", index=False, compression="zstd")
    cells.to_csv(stage / "cells.csv", index=False, float_format="%.10g")
    cov.to_csv(stage / "coverage.csv", index=False, float_format="%.10g")
    dark.to_csv(stage / "clock_loss_days.csv", index=False)
    man = {
        "protocol": WP.PROTOCOL, "splits_fingerprint": WP.fingerprint(),
        "files": {f: sha(stage / f) for f in ("rows.parquet", "cells.csv", "coverage.csv",
                                             "clock_loss_days.csv")},
        "draws": {k: hashlib.sha256(GS.draw_uniforms(k).tobytes()).hexdigest()
                  for k in ("airport_day", "common_day", "week_block")},
        "constants": {"N_RANKING": GS.N_RANKING, "MIN_SUPPORT": GS.MIN_SUPPORT, "SEED": GS.SEED,
                      "N_DRAWS": GS.N_DRAWS, "BLOCK_LEN": GS.BLOCK_LEN, "FLOOR": GS.FLOOR,
                      "PAIRS": WP.PAIRS, "BUFFER_DAYS": WP.BUFFER_DAYS,
                      "class_rule": {"gap_cut_s": GAP_CUT_S, "abs_dt_offblock_s": THR_S,
                                     "runway_agrees": 1.0, "kept_mode4": KEPT_MODE4}},
        "inputs": {k.replace(str(SRC) + "/", "O/"): v for k, v in sorted(inputs.items())},
        "numpy": np.__version__, "pandas": pd.__version__,
        "revisions": MANIFEST_REVISIONS,
    }
    (stage / "MANIFEST.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "cells": len(cells), "dark_days": len(dark),
                      "files": man["files"]}, indent=1))
    settle(stage, revise)
    shutil.rmtree(stage)


if __name__ == "__main__":
    main()
