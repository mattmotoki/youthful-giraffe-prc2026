"""Dry run of protocol week-pair-1 on 2025 data (exp-146, assignment item 3).

    PRC_SOURCE_ROOT=/path/to/competitions/prc-data-challenge-2026 \
        $PRC_SOURCE_ROOT/.venv/bin/python experiments/exp-146-validation/code/dry_run.py

1. Fold masks and row counts: every outer, inner and month-hold split over all 2025
   departures; held rows per pair x month x airport x class; per-pair cell support.
2. Clock-loss masks: made-dark and naturally dark rows per scheme, airport-month and pair.
3. A HISTORICAL REPLAY, not a decision test: the saved month-hold fold vectors of v22,
   v23 and v24 (no refit), composed by the served route and scored with prc.gscore.
   These predictions come from month holds with known label paths (REPORT.md section 4),
   so no gain is claimed from them.

Writes experiments/exp-146-validation/dryrun/ (or `--out NAME`: dryrun-r1 is the re-run after
the 2026-09-30 review fixes, with the new inner splits and the local R). Write-once: the run
stops before any compute, and again before each write, if an output already exists
(`--overwrite` lifts this; a correction goes to a new file name instead).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
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

OUT = EXP / (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "dryrun")
#: saved vectors, read from git objects at these commits (no checkout needed)
EXP134 = ("f7e7e9e427b110c5c0431e11dd4923f9256268f5",   # exp129-tail-markers
          "experiments/exp-134-proxies-in-learner/kaggle/run-fit-v3/exp134-output/predictions_{fold}.parquet")
EXP127 = ("5faa8df3b5656dafe562ff62db44f3b8ca74dc99",   # exp127-clock-precision
          "experiments/exp-127-clock-precision/push/fold_update_vectors-r1.npz")
PANEL = SRC / "experiments/exp-050-nmblind/work/panel-v22-{fold}.parquet"
W_SERVED = 0.4967252524893474          # v24's served blend weight (fitted on these months)
W_FOLD = {"january": 0.48438700652275296, "july_loo": 0.3692640950388034}   # exp-134 weights.csv
FOLDS = {"january": 1, "july_loo": 7}
inputs: dict = {}
OUTPUTS = ("splits.csv", "held_rows.csv", "support.csv", "clock_loss_rows.csv", "dryrun.json")
OVERWRITE = "--overwrite" in sys.argv


def refuse_existing(*names: str) -> None:
    hit = [n for n in names if (OUT / n).exists()]
    if hit and not OVERWRITE:
        raise SystemExit(f"refusing to overwrite {hit} in {OUT}; write a new name or pass --overwrite")


def git_blob(commit: str, path: str) -> bytes:
    b = subprocess.run(["git", "-C", str(SRC), "show", f"{commit}:{path}"], check=True,
                       capture_output=True).stdout
    inputs[f"{commit[:8]}:{path}"] = hashlib.sha256(b).hexdigest()
    return b


def all_departure_days() -> pd.DataFrame:
    parts = []
    for m in D.MONTHS:
        f = D.load_month(m, columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])
        parts.append(f[f.PHASE_mvt == "DEP"][["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]])
    return pd.concat(parts, ignore_index=True)


def fold_counts(rows: pd.DataFrame) -> dict:
    dep = all_departure_days()
    splits = list(WP.outer_splits())
    splits += [WP.inner(k, j) for k in WP.PAIRS for j in WP.inner_rotation(k)]
    splits += [WP.month_hold(m) for m in ("2025-01", "2025-07", "2025-12")]
    rec = []
    for s in splits:
        lab, sc, buf = s.label_mask(dep), s.score_mask(dep), s.buffer_mask(dep)
        assert not (lab & sc).any() and not (lab & buf).any()
        rec.append({"split": s.name, "token": s.token, "label_rows": int(lab.sum()),
                    "scored_rows": int(sc.sum()), "excluded_unscored_rows": int(buf.sum()),
                    "label_days": int(len(np.unique(WP.day_of(dep)[lab])))})
    refuse_existing("splits.csv")
    pd.DataFrame(rec).to_csv(OUT / "splits.csv", index=False)
    # each January / July departure is scored by exactly one outer fold, never trained on by it
    jj = rows[rows.month != 12].set_index("MVT_ID_mvt")
    dep_i = dep.set_index(dep.MVT_ID_mvt.astype("int64"))
    scored_by = np.zeros(len(dep), np.int8)
    for s in WP.outer_splits():
        m = s.score_mask(dep)
        assert (scored_by[m] == 0).all()
        scored_by[m] = int(s.name.split("-")[1])
    sb = pd.Series(scored_by, index=dep_i.index).reindex(jj.index)
    assert (sb.to_numpy() == jj.pair.to_numpy()).all(), "outer scoring disagrees with the frozen pair"
    for k in WP.PAIRS:                              # inner training sets nest inside the outer one
        lo = WP.outer(k).label_mask(dep)
        for j in WP.inner_rotation(k):
            assert not (WP.inner(k, j).label_mask(dep) & ~lo).any()
    held = (jj.reset_index().groupby(["pair", "month", "airport", "cls"]).size().rename("rows")
            .reset_index())
    refuse_existing("held_rows.csv")
    held.to_csv(OUT / "held_rows.csv", index=False)
    wide = held.pivot_table(index=["month", "airport", "cls"], columns="pair", values="rows",
                            fill_value=0)
    wide.columns = [f"n_pair{k}" for k in wide.columns]
    cells = GS.load_cells().set_index(["month", "airport", "cls"])
    sup = cells.join(wide, how="left").fillna(0)
    for c in wide.columns:
        sup[c] = sup[c].astype(int)
    sup["pairs_supported"] = (sup[wide.columns] >= GS.MIN_SUPPORT).sum(1)
    refuse_existing("support.csv")
    sup.reset_index().to_csv(OUT / "support.csv", index=False, float_format="%.8g")
    by_pair = held.groupby(["pair", "cls"]).rows.sum().unstack(fill_value=0)
    return {"splits": rec, "held_by_pair_class": by_pair.to_dict(orient="index"),
            "all_2025_departures": int(len(dep)),
            "held_airport_days": int(jj.groupby(["airport", "date"]).ngroups),
            "unsupported": sup[(sup.n < GS.MIN_SUPPORT) & (sup.N > 0)].reset_index()
            [["month", "airport", "cls", "N", "n"]].to_dict(orient="records"),
            "cells_not_testable_in_some_pair": sup[(sup.pairs_supported < 5) & (sup.N > 0)
                                                   & (sup.n >= GS.MIN_SUPPORT)].reset_index()
            [["month", "airport", "cls", "N", "n", "pairs_supported"]].to_dict(orient="records")}


def clock_loss(rows: pd.DataFrame) -> list:
    dd = GS.load_clock_loss_days()
    jj = rows[rows.month != 12]
    rec = []
    for scheme, g in dd.groupby("scheme"):
        key = set(zip(g.airport, g.date))
        on = np.array([(a, d) in key for a, d in zip(jj.airport, jj.date)])
        made = on & (jj.cls == "clock").to_numpy()
        t = jj.assign(made_dark=made, natural_dark=(jj.cls == "ordinary").to_numpy())
        s = t.groupby(["month", "airport", "pair"])[["made_dark", "natural_dark"]].sum()
        s = s[(s.made_dark > 0)].reset_index()
        s.insert(0, "scheme", scheme)
        rec.append(s)
    out = pd.concat(rec, ignore_index=True)
    refuse_existing("clock_loss_rows.csv")
    out.to_csv(OUT / "clock_loss_rows.csv", index=False)
    return out.groupby(["scheme", "month", "airport"])[["made_dark", "natural_dark"]].sum() \
        .reset_index().to_dict(orient="records")


def replay_vectors(rows: pd.DataFrame) -> dict:
    """v22, v23 and v24 fold vectors by the served route, on January and July."""
    z = np.load(io.BytesIO(git_blob(*EXP127)))
    cls = rows.set_index("MVT_ID_mvt")
    out = {}
    for fold, m in FOLDS.items():
        p = pd.read_parquet(Path(str(PANEL).format(fold=fold)))
        inputs[str(PANEL).format(fold=fold).replace(str(SRC) + "/", "O/")] = GS._sha256_file(
            Path(str(PANEL).format(fold=fold)))
        ids = p.MVT_ID_mvt.to_numpy("float64").astype("int64")
        v22 = pd.Series(p.p_v22.to_numpy("float64"), index=ids)
        panel_clock = pd.Series(p.obs_has_clock.to_numpy(bool), index=ids)
        frozen_clock = cls.cls.reindex(ids).to_numpy() == "clock"
        class_mismatch = int((panel_clock.to_numpy() != frozen_clock).sum())
        # v23: the repaired fold vector, LIRF held and the 60 s floor, as the served writer does
        dv = pd.Series(z[f"{fold}_delta"].astype("float64"), index=z[f"{fold}_MVT_ID_mvt"].astype("int64"))
        dv[cls.airport.reindex(dv.index).to_numpy() == "LIRF"] = 0.0
        v23 = v22.copy()
        v23.loc[dv.index] = np.maximum(60.0, v22.loc[dv.index] + dv)
        # v24: exp-134's learner on ordinary no-clock rows (the served class), blended from v23
        e = pd.read_parquet(io.BytesIO(git_blob(EXP134[0], EXP134[1].format(fold=fold))))
        e = e.set_index(e.MVT_ID_mvt.astype("int64"))
        served = cls.cls.reindex(e.index).to_numpy() == "ordinary"
        L = e.pred_proxy[served]
        vec = {"v22": v22, "v23": v23}
        for tag, w in (("v24_wfold", W_FOLD[fold]), ("v24_wserved", W_SERVED)):
            v = v23.copy()
            v.loc[L.index] = np.maximum(60.0, v23.loc[L.index] + w * (L - v23.loc[L.index]))
            vec[tag] = v
        out[fold] = {"vec": vec, "panel_vs_frozen_clock_mismatch": class_mismatch,
                     "n_v23_rows": int(len(dv)), "n_v23_lirf_held": int(
            (cls.airport.reindex(dv.index) == "LIRF").sum()), "n_v24_rows": int(served.sum()),
            "n_v24_skipped_mode4_clock": int((~served).sum()),
            "max_abs_saved_blend_vs_replay": float(np.abs(
                np.maximum(60.0, e.pred_blend_proxy[served]) - vec["v24_wfold"].loc[L.index]).max())}
    return out


def main() -> None:
    refuse_existing(*OUTPUTS)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = GS.load_rows()
    res = {"protocol": WP.PROTOCOL, "splits_fingerprint": WP.fingerprint(),
           "frozen_manifest_sha256": GS._sha256_file(GS.FROZEN / "MANIFEST.json")}
    res["folds"] = fold_counts(rows)
    res["clock_loss"] = clock_loss(rows)
    rv = replay_vectors(rows)
    cat = {tag: pd.concat([rv[f]["vec"][tag] for f in FOLDS]) for tag in rv["january"]["vec"]}
    ids = cat["v22"].index.to_numpy()
    for tag in cat:
        assert (cat[tag].index.to_numpy() == ids).all()
    res["replay_composition"] = {f: {k: v for k, v in rv[f].items() if k != "vec"} for f in FOLDS}
    comps = {"v23_vs_v22": ("v22", "v23"), "v24_vs_v23_wfold": ("v23", "v24_wfold"),
             "v24_vs_v23_wserved_insample": ("v23", "v24_wserved"), "v24_vs_v22_wfold": ("v22", "v24_wfold")}
    pair = rows.set_index("MVT_ID_mvt").pair.reindex(ids).to_numpy()
    scores = {}
    for name, (b, c) in comps.items():
        base, cand = cat[b].to_numpy(), cat[c].to_numpy()
        s = {"all": GS.score(ids, base, cand), "clean": GS.score(ids, base, cand, view="clean")}
        s["per_pair_all"] = {}
        for k in WP.PAIRS:
            mk = pair == k
            r = GS.score(ids[mk], base[mk], cand[mk], require_complete=False, resample=False)
            s["per_pair_all"][k] = {"G": r["G"], "gain_s": r["gain_s"], "n_rows": r["n_rows"],
                                    "unsupported_ranking_rows": r["unsupported_ranking_rows"]}
        s["lirf_july_mask_note"] = ("not scored: a clock-loss score needs predictions rebuilt "
                                    "from masked inputs, and these saved vectors are unmasked")
        scores[name] = s
        a = s["all"]
        print(f"{name:32s} G {a['G']:9.2f} s2/row  gain {a['gain_s']:+.3f} s at local R "
              f"{a['R_reference']:.2f} s  "
              f"95% [{a['interval']['lo95']:.1f}, {a['interval']['hi95']:.1f}]  "
              f"clean G {s['clean']['G']:9.2f}  per pair "
              + " ".join(f"{s['per_pair_all'][k]['G']:.0f}" for k in WP.PAIRS), flush=True)
    res["replay_scores"] = scores
    res["inputs"] = inputs
    refuse_existing("dryrun.json")
    (OUT / "dryrun.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + "\n")
    print(json.dumps(res["folds"]["held_by_pair_class"], indent=1))


if __name__ == "__main__":
    main()
