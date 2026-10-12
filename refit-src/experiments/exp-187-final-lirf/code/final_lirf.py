"""The label-free LIRF rules of v73 on a 2026 movement file: the value of every LIRF departure without an aircraft type.

The v73 chain on these rows (checked on January/July 2026, `stages_janjul.py`; every one is a no-plan row):
    1 no-plan v3cal   exp-167 M25/M25d form on the route value np_base            (`noplan_v3cal.py`)
    2 C73             LIRF route practice: max(60, v + w (max(60, MVT - SCHED) - v)) (exp-169 `c73.py`)
    3 C66             LIRF mirror rows: the point max(60, ds or 600)                (exp-169 `exploit029_rules.c66_rows`)
    4 bounds          clip to [lower, upper]: lower = max(60, C33, C64, C68), upper = min(C33, C40, C63)
                      (exp-169 `exploit029_rules.rule_frame`; links from exp-160 `exploit_links_014.unique_links`;
                      C55 and C71 cover EDDF, EGLL, LFPG and LTFM only)
    5 NOAC            LIRF no-type mixture rows: 900 + lam (v - 900), lam from the held 2025 fold (c5 `post_noac`)
    6 ROLL            LIRF no-type cross rows with 50,000 <= MVT - SCHED < 86,400 s: p (86,400 + offset) + (1 - p) ms
                      (c5 `post_roll`, `iso/roll_step.py`, logistic on the twelve 2025 months)
Every other v73 stage leaves these rows as they are (head, pair and primary blends, C39b practice, C55, C71, C95, C98,
C65, C58, T11b-L, EXT, LONG, LIRF, L2C, ORD-S): `stages_janjul.py` and REPORT.md.

The movement file comes from `prc/data.py` (mode ranking or final; the final files only by their SHA-256). np_base
is an input: parquet MVT_ID_mvt, np_base on every LIRF no-plan departure (`--parity` in ranking mode uses the exp159
entry of the original run).

    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-187-final-lirf/code/final_lirf.py --mode ranking --parity --out-dir <dir>
    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-187-final-lirf/code/final_lirf.py --mode final --np-base <parquet> --out-dir <dir>
    ... --np-r1-a 0.444: base weight of the no-plan r1 rows (design row, REPORT-V 4.2; the default is the served 0.75)
"""
from __future__ import annotations

import os
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXPORT = Path(os.environ.get("PRC_SOURCE_EXPORT", "/path/to/source-export"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp-169-stack-plan/code"))
sys.path.insert(0, str(EXPORT / "iso"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prc import data as D  # noqa: E402
import c73 as C73  # noqa: E402
import exploit029_rules as K  # noqa: E402
import roll_step as RS  # noqa: E402
import noplan_v3cal as NP  # noqa: E402

RS.RAW = ROOT / "data/raw"          # roll_step reads <export>/input-root/data/raw; the same 2025 files live here
ID = "MVT_ID_mvt"
AP = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
COLS = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "STAND_mvt", "RUNWAY_mvt",
        "TAXITIME_SEC_mvt"]                                  # exploit029_prepare.COLS
LINK_COLS = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt"]
LINKS_SRC = ROOT / "experiments/exp-160-feature-worker/code/exploit_links_014.py"
NOAC_R, NOAC_STEP = 900.0, 0.25                              # iso/candidate.py
STAGES = ROOT / "experiments/exp-187-final-lirf/out/stages-janjul.parquet"
HELD = ("training_2025-01-01_2025-02-01", "training_2025-07-01_2025-08-01")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pure(path: Path, name: str):
    """One function of a source file, run without its module (exploit029_prepare.pure)."""
    tree = ast.parse(path.read_text())
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(defs) == 1
    space = {"np": np, "pd": pd, "secs": K.secs, "epoch": K.epoch, "E": SimpleNamespace(AP=AP), "ID": ID, "AP": AP}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(path), "exec"), space)
    return space[name]


def training() -> pd.DataFrame:
    parts = []
    for p in sorted((ROOT / "data/raw").glob("training_2025-*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d["src"] = p.name[9:16]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------ rules on every departure of the file

def bounds(dep: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """C33, C40, C64, C68, C63 and the bound intersection, as `exploit029_rules.rule_frame` builds them."""
    f = pd.DataFrame({ID: dep[ID].to_numpy("int64")})
    plan = dep.LOBT_flt.notna().to_numpy()
    dl = K.secs(dep.MVT_TIME_UTC_mvt, dep.LOBT_flt).to_numpy(float)
    f["lo33"], f["hi33"] = dl - 3606, dl + 3606              # read_stacks.add_bracket
    link = links.set_index(ID).reindex(f[ID])
    ml, al, tn = link.dll.to_numpy(), link.a_l.to_numpy(), link.tnl.to_numpy()
    f["link_id"], f["dll"], f["tnl"], f["a_l"] = link[ID + "_a"].to_numpy(), ml, tn, al
    f["C40"] = np.isfinite(ml) & ~plan & (tn >= 60) & (tn <= 3600) & (al > 3606) & (ml >= 3600) & (ml - 3606 >= 60)
    f["C64"] = np.isfinite(ml) & ~plan & (ml < 3600)
    ap = dep.ADEP_mvt.astype(str).to_numpy()
    f["C63"] = np.isin(ap, ["LEBL", "LEMD"])
    f["C68"] = plan & (dl > 72000)
    lo33 = np.fmax(60, f.lo33.to_numpy())
    hi33 = np.where(np.isfinite(f.hi33), f.hi33, np.inf)
    lo64 = np.where(f.C64, ml + 3594, np.nan)
    lo68 = np.where(f.C68, 86400, np.nan)
    hi40 = np.where(f.C40, ml - 3606, np.nan)
    hi63 = np.where(f.C63, 3600, np.nan)
    f["lower"] = np.fmax.reduce([lo33, lo64, lo68])            # + C55, C71 at EDDF, EGLL, LFPG, LTFM (not LIRF)
    f["upper"] = np.fmin.reduce([hi33, hi40, hi63])
    f["lower_by"] = np.select([np.isfinite(lo68) & (lo68 >= np.fmax(lo33, lo64)), np.isfinite(lo64) & (lo64 > lo33)],
                              ["C68", "C64"], np.where(np.isfinite(f.lo33.to_numpy()) & (lo33 > 60), "C33", "floor60"))
    f["upper_by"] = np.select([np.isfinite(hi40) & (hi40 <= np.fmin(hi33, np.where(np.isfinite(hi63), hi63, np.inf))),
                               np.isfinite(hi63) & (hi63 < hi33)], ["C40", "C63"],
                              np.where(np.isfinite(f.hi33.to_numpy()), "C33", "none"))
    return f


def c73_rows(path: Path, ids: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """C73 on the 2026 rows of `path`; c73.raw_frame reads the twelve 2025 files and the 2026 file through `pin`."""
    pin = lambda p: path if Path(p).name == "ranking.parquet" else Path(p)  # noqa: E731
    raw = C73.raw_frame(pin)
    bundle = pd.DataFrame({ID: ids.astype("int64"), "practice": False})   # C39b needs a plan: no LIRF no-plan row
    b, d, rates, stats = C73.add_key(bundle, raw, "ranking")
    keep = b[[ID, "c73", "c73_weight", "c73_point", "c73_route", "c73_U_count"]].copy()
    return keep, stats


def c66_table(mov: pd.DataFrame) -> pd.DataFrame:
    z = K.c66_rows(mov)
    return pd.DataFrame({ID: z[ID].astype("int64").to_numpy(), "C66": z.scope.to_numpy(bool), "point66": z.point.to_numpy(),
                         "c66_ds": z.ds.to_numpy(), "c66_own": z.own.to_numpy(bool), "c66_arr": z[ID + "_arr"].astype("int64").to_numpy()})


# ------------------------------------------------------------------ c5 steps

def noac_lambda() -> dict:
    """lam of c5 `post_noac` for 2026: fitted on the held 2025 values of the no-type mixture rows (both months)."""
    s = pd.read_parquet(STAGES)
    f = s[s.split == "fold"].copy()
    parts = [pd.read_parquet(ROOT / f"data/raw/{h}.parquet", columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]) for h in HELD]
    t = pd.concat(parts, ignore_index=True)
    t = t[t.PHASE_mvt == "DEP"]
    ms = K.secs(t.MVT_TIME_UTC_mvt, t.SCHED_TIME_UTC_mvt)
    ms = pd.Series(ms.to_numpy(), index=t[ID].astype("int64").to_numpy()).reindex(f[ID].to_numpy()).to_numpy(float)
    v, y = f.final.to_numpy(float), f.y.to_numpy(float)
    mix = mixture(v, ms) & (y < 80000)
    x, tt = v[mix] - NOAC_R, y[mix] - NOAC_R
    lam_fit = float((x * tt).sum() / (x * x).sum())
    return {"rows": int(mix.sum()), "lam_fit": lam_fit, "lam": 1 - NOAC_STEP * (1 - lam_fit)}


def mixture(v: np.ndarray, ms: np.ndarray) -> np.ndarray:
    """c5 `_noac_mix` on rows that are already LIRF no-type rows."""
    den = np.where(np.abs(ms - NOAC_R) < 1, 1.0, ms - NOAC_R)
    p = (v - NOAC_R) / den
    return (p > 0.03) & (p < 0.97) & (ms > NOAC_R + 600)


def roll_model() -> dict:
    lr, off, t = RS.model(exclude_held=False)
    return {"lr": lr, "offset": off, "rows": int(len(t)), "rollovers": int(t.roll.sum()),
            "coef": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0])}


def cross(path: Path, ids: np.ndarray) -> np.ndarray:
    f = RS._frame([path], with_y=False)
    s = pd.Series(f.cross.to_numpy(), index=f[ID].astype("int64").to_numpy())
    return s.reindex(ids.astype("int64")).astype("boolean").fillna(False).to_numpy(bool)


# ------------------------------------------------------------------ the runner

def run(mode: str, np_base: pd.Series | None, parity: bool, out: Path, fill: str = "none", r1_a: float | None = None) -> dict:
    path = D.ranking_file(mode)
    tpl = pd.read_parquet(D.submitting_file(mode))
    mov = pd.read_parquet(path, columns=COLS + ["AIRCRAFT_TYPE_mvt"])
    dep = mov[mov.PHASE_mvt == "DEP"].set_index(ID)
    assert dep.index.is_unique and len(dep) == len(tpl) == D.n_ranking(mode)
    dep = dep.loc[tpl[ID].to_numpy()].reset_index()
    ids = dep[ID].to_numpy("int64")
    month = dep.MVT_TIME_UTC_mvt.dt.month.to_numpy()
    assert set(np.unique(month)) == set(D.rank_month_numbers(mode))
    ms = K.secs(dep.MVT_TIME_UTC_mvt, dep.SCHED_TIME_UTC_mvt).to_numpy(float)

    # bounds (links on the twelve 2025 files and this file, as exploit029_prepare)
    tr = training()
    allraw = pd.concat([tr[COLS], mov[COLS]], ignore_index=True)
    links, windows = pure(LINKS_SRC, "unique_links")(allraw[LINK_COLS])
    links = links.reset_index(drop=True)
    links[ID] = links[ID].astype("int64")
    bf = bounds(dep, links)
    c73, c73_stats = c73_rows(path, ids)
    c66 = c66_table(mov[COLS])
    rules = pd.DataFrame({ID: ids, "month": month, "ap": dep.ADEP_mvt.astype(str).to_numpy(), "ms": ms,
                          "notype": dep.AIRCRAFT_TYPE_mvt.isna().to_numpy(), "plan": dep.LOBT_flt.notna().to_numpy()})
    rules = rules.merge(bf, on=ID, how="left").merge(c73, on=ID, how="left").merge(c66, on=ID, how="left")
    rules["C66"] = rules.C66.fillna(False).astype(bool)
    rules["c73"] = rules.c73.fillna(False).astype(bool)
    assert np.array_equal(rules[ID].to_numpy(), ids) and (rules.lower <= rules.upper).all()
    rules["cross"] = cross(path, ids)

    # no-plan v3cal on the LIRF departures without a plan
    chosen, cell = NP.design()
    fit = NP.fit_2025()
    d26, te26 = NP.rows(mov_full(path), fit)
    p26, m26 = NP.classes(te26, fit, chosen)
    if parity:
        e = pd.read_parquet(NP.BASE_ENTRY)
        assert sha(NP.BASE_ENTRY) == NP.BASE_ENTRY_SHA
        np_base = pd.Series(e.TAXITIME_SEC_mvt.to_numpy("float64"), index=e[ID].to_numpy("float64").astype("int64"))
    base = np_base.reindex(d26[ID].to_numpy()).to_numpy(float)
    filled = ~np.isfinite(base)
    if fill == "mix":       # dry-run stand-in: the class-model value max(60, p ms + (1 - p) m_real)
        pm, mm = p26.reindex(d26[ID]).to_numpy(), m26.reindex(d26[ID]).to_numpy()
        base = np.where(filled, np.maximum(60, pm * d26.ms.to_numpy() + (1 - pm) * mm), base)
    assert np.isfinite(base).all(), "a no-plan row has no np_base value"
    npv = NP.value(d26, base, p26, m26, fit, cell)
    npv["base_filled"] = filled
    if r1_a is not None:    # design row (desk 2026-10-09 07:30 UTC, REPORT-V 4.2), the same form as frames_v.py --np-r1-a
        r1 = (npv.part == "r1").to_numpy()
        pm, nb = npv.p.to_numpy(float), npv.base.to_numpy(float)
        mix = np.maximum(NP.C.FLOOR, pm * npv.ms.to_numpy(float) + (1 - pm) * npv.m_real.to_numpy(float))
        assert np.array_equal(np.maximum(NP.C.FLOOR, (1 - NP.NF.W) * nb + NP.NF.W * mix)[r1], npv.cal.to_numpy(float)[r1])
        npv.loc[r1, "cal"] = np.maximum(NP.C.FLOOR, r1_a * nb[r1] + (1 - r1_a) * mix[r1])

    # the chain on the LIRF departures without an aircraft type
    tgt = rules[(rules.ap == "LIRF") & rules.notype].copy()
    n = npv.set_index(ID)
    assert set(tgt[ID]) <= set(n.index), "a LIRF no-type row has a plan"
    tgt = tgt.join(n[["part", "p", "m_real", "base", "base_filled", "candidate", "cal", "zone", "state", "rule39"]], on=ID)
    lo, hi = tgt.lower.to_numpy(float), tgt.upper.to_numpy(float)
    v = tgt.cal.to_numpy(float).copy()
    tgt["v_np"] = v.copy()
    q = tgt.c73.to_numpy(bool)
    v[q] = np.maximum(60, v[q] + tgt.c73_weight.to_numpy(float)[q] * (tgt.c73_point.to_numpy(float)[q] - v[q]))
    tgt["v_c73"] = v.copy()
    q = tgt.C66.to_numpy(bool)
    v[q] = tgt.point66.to_numpy(float)[q]
    tgt["v_c66"] = v.copy()
    v = np.clip(v, lo, hi)
    tgt["v_bounds"] = v.copy()
    lam = noac_lambda()
    tms = tgt.ms.to_numpy(float)
    mix = mixture(v, tms)
    v[mix] = NOAC_R + lam["lam"] * (v[mix] - NOAC_R)
    v[mix] = np.maximum(60, np.clip(v[mix], lo[mix], hi[mix]))
    tgt["noac_mix"], tgt["v_noac"] = mix, v.copy()
    rm = roll_model()
    roll = tgt.cross.to_numpy(bool) & (tms >= RS.ROLL_LO) & (tms < RS.DAY)
    pr, val = RS.value(rm["lr"], rm["offset"], tms[roll])
    v[roll] = np.maximum(60, np.clip(val, lo[roll], hi[roll]))
    tgt["roll"], tgt["roll_p"] = roll, np.nan
    tgt.loc[roll, "roll_p"] = pr
    tgt["v_final"] = v
    assert np.isfinite(v).all() and (v >= 60).all()
    tgt["set_by"] = set_by(tgt)

    out.mkdir(parents=True, exist_ok=True)
    rules.to_parquet(out / "rules-dep.parquet", index=False)
    npv[NP.KEEP + ["base_filled"]].to_parquet(out / "noplan-rows.parquet", index=False)
    tgt.to_parquet(out / "lirf-notype.parquet", index=False)
    links.to_parquet(out / "links.parquet", index=False)
    rec = {"mode": mode, "ranking_file": str(path.relative_to(ROOT)), "ranking_sha256": sha(path),
           "submitting_sha256": sha(D.submitting_file(mode)), "departures": int(len(ids)), "links": int(len(links)),
           "c73": c73_stats | {"selected_routes": [r["route"] for r in c73_stats["selected_routes"]]},
           "noplan_design": {"class_model": chosen, "c43_cell": cell, "r1_base_weight": 1 - NP.NF.W if r1_a is None else r1_a},
           "np_base_fill": fill,
           "np_base_filled_rows": {int(k): int(v) for k, v in pd.Series(filled).groupby(d26.month.to_numpy()).sum().items()}, "noac": lam,
           "roll": {k: rm[k] for k in ("offset", "rows", "rollovers", "coef", "intercept")},
           "counts": counts(rules, tgt, npv), "code_sha256": {p.name: sha(p) for p in (Path(__file__), Path(NP.__file__), LINKS_SRC,
                                                              Path(C73.__file__), Path(K.__file__), Path(RS.__file__))}}
    (out / "record.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    return rec


def mov_full(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def set_by(t: pd.DataFrame) -> np.ndarray:
    """The rule that sets the final value (the last stage that changed it; ties go to the later stage)."""
    order = [("v_c73", "C73"), ("v_c66", "C66"), ("v_bounds", None), ("v_noac", "NOAC"), ("v_final", "ROLL")]
    lab = ("noplan-" + t.part.astype(str)).to_numpy().astype(object)
    prev = t.v_np.to_numpy()
    for col, name in order:
        cur = t[col].to_numpy()
        ch = cur != prev
        if name is None:
            nm = np.where(cur < prev, t.upper_by.to_numpy().astype(object), t.lower_by.to_numpy().astype(object))
            lab = np.where(ch, nm, lab)
        else:
            lab = np.where(ch, name, lab)
        prev = cur
    return lab


def counts(rules: pd.DataFrame, tgt: pd.DataFrame, npv: pd.DataFrame) -> dict:
    by = lambda m, mo: {int(k): int(v) for k, v in pd.Series(m).groupby(mo).sum().items()}  # noqa: E731
    lirf = (rules.ap == "LIRF").to_numpy()
    out = {"departures": by(np.ones(len(rules), bool), rules.month.to_numpy()),
           "lirf_notype": by(np.ones(len(tgt), bool), tgt.month.to_numpy()),
           "lirf_noplan": by(np.ones(len(npv), bool), npv.month.to_numpy()),
           "lirf_noplan_typed": by(~npv.noac.to_numpy(bool), npv.month.to_numpy()),
           "C33_finite_all_airports": by(np.isfinite(rules.lo33.to_numpy()), rules.month.to_numpy()),
           "C40_all_airports": by(rules.C40.to_numpy(bool), rules.month.to_numpy()),
           "C64_all_airports": by(rules.C64.to_numpy(bool), rules.month.to_numpy()),
           "C68_all_airports": by(rules.C68.to_numpy(bool), rules.month.to_numpy()),
           "C73_lirf": by(rules.c73.to_numpy(bool) & lirf, rules.month.to_numpy()),
           "C66_lirf": by(rules.C66.to_numpy(bool), rules.month.to_numpy())}
    m = tgt.month.to_numpy()
    for col, name in (("v_c73", "C73"), ("v_c66", "C66"), ("v_bounds", "bounds"), ("v_noac", "NOAC"), ("v_final", "ROLL")):
        prevs = {"v_c73": "v_np", "v_c66": "v_c73", "v_bounds": "v_c66", "v_noac": "v_bounds", "v_final": "v_noac"}
        out[f"notype_moved_by_{name}"] = by((tgt[col] != tgt[prevs[col]]).to_numpy(), m)
    out["notype_noplan_parts"] = {int(k): g.part.value_counts().to_dict() for k, g in tgt.groupby("month")}
    out["notype_set_by"] = {int(k): g.set_by.value_counts().to_dict() for k, g in tgt.groupby("month")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["ranking", "final"])
    ap.add_argument("--np-base", type=Path, help="parquet MVT_ID_mvt, np_base on every LIRF no-plan departure")
    ap.add_argument("--parity", action="store_true", help="ranking mode: np_base from the exp159 entry of the original run")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--fill", default="none", choices=["none", "mix"], help="np_base for no-plan rows missing from --np-base")
    ap.add_argument("--np-r1-a", type=float, help="base weight a of the no-plan r1 rows, max(60, a base + (1 - a) mix); "
                                                   "default: the served 0.75 (design row a = 0.444, REPORT-V 4.2)")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    NP.threads(a.threads)
    assert a.parity == (a.np_base is None), "give --np-base, or --parity in ranking mode"
    assert not a.parity or a.mode == "ranking"
    base = None
    if a.np_base is not None:
        b = pd.read_parquet(a.np_base)
        base = pd.Series(b.np_base.to_numpy("float64"), index=b[ID].to_numpy("float64").astype("int64"))
    assert a.np_r1_a is None or not a.parity
    rec = run(a.mode, base, a.parity, a.out_dir, a.fill, a.np_r1_a)
    print(json.dumps(rec["counts"], indent=1, default=str))
    print("noac", rec["noac"], "roll", rec["roll"], "links", rec["links"])


if __name__ == "__main__":
    main()
