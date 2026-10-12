"""Path V: the rule and mask columns of the zabd ranking frames, built from official columns on a 2026 file.

Frames and columns (split agreed with finalplan, `experiments/exp-192-final-assembly/SPLIT.md`):
    bundle         lo, hi, rule, extension, noplan, np_v3cal, practice, practice_weight, practice_point, stand_floor,
                   cls, c73, c73_weight, c73_point                  (+ official airport, month)
    rule_frame     cls, t, scope, g, g71, c71                       (+ official ap, rw; F is finalroute's part)
    exploit_frame  C40, C58, C63, C64, C66, C68, C71, order_delta, point58, point66, lower, upper
Each column calls the original builder (exp-169 code, unchanged; a function is taken from its file by `ast` where the
module runs reads at import):
    lo, hi                 read_stacks.add_bracket                     MVT - LOBT -+ 3,606 s
    rule, extension        read_stacks.prepare (ranking lines)         clock rows outside the unsupported head cells; EXT
    noplan, np_v3cal       exp-167 no-plan form v3 + cal (code/noplan_v3cal.py; np_base is an input)
    practice*              prepare_unit2.add_practice + exp-167 practice_arm_r1.rows, twelve-month status table
    stand_floor            stand_floor.bounds (C55)
    cls                    prepare_stack3: noac without an aircraft type, clock where the clock head is finite, else ordinary
    c73*                   c73.raw_frame + add_key (C73)
    t, scope, g, g71, c71  stack9.ranking_frame + stand_inputs (u27_floor3.floors, u13_inbound.inbound, c71_parts)
    exploit frame          exploit029_rules.rule_frame on links (exp-160 unique_links), order (c65), C58 fit (twelve
                           2025 months) and the C71 parts above
Season map (COMMON): February takes January's month-keyed cells and June takes July's (head rule cells, EXT).
Inputs: --clock (MVT_ID_mvt, clock bool: the clock head is finite), --np-base (MVT_ID_mvt, np_base). With --parity
(ranking mode) both come from the saved v73-era inputs: HB1-jjrank (the original clock mask) and the exp159 entry.

    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-187-final-lirf/code/frames_v.py --mode ranking --parity --check
    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-187-final-lirf/code/frames_v.py --mode final --clock <p> --np-base <p> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
X = ROOT / "experiments/exp-169-stack-plan"
XC = X / "code"
S9 = X / "inputs/stack9-024"                      # stack9's staged copies of u27_floor3.py and u13_inbound.py
for p in (ROOT, XC, ROOT / "experiments/exp-167-target/code", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))
from prc import data as D  # noqa: E402
import c73 as C73  # noqa: E402
import exploit029_rules as K  # noqa: E402
import exploit_rules as R  # noqa: E402
import stand_floor as SF  # noqa: E402
import practice_arm_r1 as P  # noqa: E402
import noplan_v3cal as NP  # noqa: E402

ID = "MVT_ID_mvt"
AP = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
T0 = pd.Timestamp("2025-01-01", tz="UTC")
SEASON = {1: 1, 2: 1, 6: 7, 7: 7}                 # COMMON season map
UNSUPPORTED = {("LEMD", 1), ("LFPG", 1), ("LIRF", 1)}   # read_stacks.prepare: head rule cells without held support
COLS = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_ID_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "STAND_mvt", "RUNWAY_mvt",
        "TAXITIME_SEC_mvt"]                       # exploit029_prepare.COLS
LINK_COLS = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_ID_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt"]
RAWCOLS = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
           "LOBT_flt", "EOBT_1_flt", "AOBT_3_flt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_mvt"]  # stack9
PRACTICE_COLS = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                 "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "LOBT_flt", "AOBT_3_flt", "AIRCRAFT_TYPE_mvt"]  # prepare_unit2
LINKS_SRC = ROOT / "experiments/exp-160-feature-worker/code/exploit_links_014.py"
PRACTICE_REC = ROOT / "experiments/exp-167-target/results/practice-arm-r1-001.json"
C58_REC = X / "inputs/exploit-029/C58-statistics.json"
HB1_RANK = X / "inputs/HB1-jjrank.parquet"
FRAMES = {
    "bundle": ["lo", "hi", "rule", "extension", "noplan", "np_v3cal", "practice", "practice_weight", "practice_point",
               "stand_floor", "cls", "c73", "c73_weight", "c73_point"],
    "rule_frame": ["cls", "t", "scope", "g", "g71", "c71"],
    "exploit_frame": ["C40", "C58", "C63", "C64", "C66", "C68", "C71", "order_delta", "point58", "point66", "lower", "upper"],
}
DTYPES = {"bundle": {"lo": "float64", "hi": "float64", "rule": "bool", "extension": "bool", "noplan": "bool",
                     "np_v3cal": "float64", "practice": "bool", "practice_weight": "float64", "practice_point": "float64",
                     "stand_floor": "float64", "cls": "object", "c73": "bool", "c73_weight": "float64", "c73_point": "float64"},
          "rule_frame": {"cls": "object", "t": "float64", "scope": "bool", "g": "float64", "g71": "float64", "c71": "bool"},
          "exploit_frame": {k: "bool" for k in ["C40", "C58", "C63", "C64", "C66", "C68", "C71"]}
          | {k: "float64" for k in ["order_delta", "point58", "point66", "lower", "upper"]}}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def function(path: Path, name: str, ns: dict):
    """One top-level function of a source file, compiled into `ns` (the module is not imported)."""
    defs = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(defs) == 1, (path, name)
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


# the original functions, unchanged
NS9 = {"np": np, "pd": pd, "ast": ast, "ID": ID, "T0": T0, "R": R, "D": S9, "GS": None}
for _n in ("source_function", "rawframe", "stand_inputs"):
    function(XC / "stack9.py", _n, NS9)
add_bracket = function(XC / "read_stacks.py", "add_bracket", {"np": np, "pd": pd, "ID": ID})
add_practice = function(XC / "prepare_unit2.py", "add_practice", {"np": np, "pd": pd, "ID": ID, "P": P})
unique_links = function(LINKS_SRC, "unique_links", {"np": np, "pd": pd, "secs": K.secs, "epoch": K.epoch,
                                                    "E": SimpleNamespace(AP=AP), "ID": ID, "AP": AP})


def training() -> pd.DataFrame:
    """The twelve 2025 files in exploit029_prepare's form (file order, src, pos)."""
    parts = []
    for p in sorted((ROOT / "data/raw").glob("training_2025-*.parquet")):
        d = pd.read_parquet(p, columns=COLS)
        d["src"] = p.name[9:16]
        d["pos"] = np.arange(len(d))
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def season(month: np.ndarray) -> np.ndarray:
    return np.array([SEASON[int(m)] for m in month])


def build(mode: str, clock: pd.Series, np_base: pd.Series, fill: str = "none", r1_a: float | None = None) -> tuple[dict, dict]:
    path = D.ranking_file(mode)
    tpl = pd.read_parquet(D.submitting_file(mode), columns=[ID])
    ids = tpl[ID].to_numpy("int64")
    assert len(ids) == D.n_ranking(mode) and tpl[ID].is_unique
    raw = pd.read_parquet(path)
    assert raw.loc[raw.PHASE_mvt.eq("DEP"), "TAXITIME_SEC_mvt"].isna().all()
    dep = raw[raw.PHASE_mvt.eq("DEP")].set_index(ID)
    assert dep.index.is_unique and len(dep) == len(ids)
    dep = dep.loc[ids].reset_index()
    rec = {"mode": mode, "ranking_file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "ranking_sha256": sha(path),
           "submitting_sha256": sha(D.submitting_file(mode)), "departures": int(len(ids))}

    # ---------------- bundle
    b = pd.DataFrame({ID: ids})
    add_bracket(b, raw[[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt", "ADEP_mvt"]])
    assert set(np.unique(b.month)) == set(D.rank_month_numbers(mode))
    ck = clock.reindex(ids)
    assert ck.notna().all(), "the clock input misses a departure"
    ck = ck.to_numpy(bool)
    sm = season(b.month.to_numpy())
    b["rule"] = ck & np.array([(a, int(m)) not in UNSUPPORTED for a, m in zip(b.airport, sm)])
    b["extension"] = ck & b.airport.isin(["LEMD", "LFPG"]).to_numpy() & (sm == 1)
    # no-plan v3cal on the LIRF departures without a plan
    chosen, cell = NP.design()
    fit = NP.fit_2025()
    d26, te26 = NP.rows(raw, fit)
    p26, m26 = NP.classes(te26, fit, chosen)
    base = np_base.reindex(d26[ID].to_numpy()).to_numpy(float)
    filled = ~np.isfinite(base)
    if fill == "mix":       # dry-run stand-in (exp-187 REPORT section 5): max(60, p ms + (1 - p) m_real)
        pm, mm = p26.reindex(d26[ID]).to_numpy(), m26.reindex(d26[ID]).to_numpy()
        base = np.where(filled, np.maximum(60, pm * d26.ms.to_numpy() + (1 - pm) * mm), base)
    assert np.isfinite(base).all(), "a no-plan row has no np_base value"
    npv = NP.value(d26, base, p26, m26, fit, cell).set_index(ID)
    r1_rows = 0
    if r1_a is not None:    # design row (desk 2026-10-09 07:30 UTC, REPORT-V 4.2): r1 base weight refit on held 2025
        w = NP.NF.W
        r1 = (npv.part == "r1").to_numpy()
        pm, ms_, nb = npv.p.to_numpy(float), npv.ms.to_numpy(float), npv.base.to_numpy(float)
        mix = np.maximum(NP.C.FLOOR, pm * ms_ + (1 - pm) * npv.m_real.to_numpy(float))
        assert np.array_equal(np.maximum(NP.C.FLOOR, (1 - w) * nb + w * mix)[r1], npv.cal.to_numpy(float)[r1])
        npv.loc[r1, "cal"] = np.maximum(NP.C.FLOOR, r1_a * nb[r1] + (1 - r1_a) * mix[r1])
        r1_rows = int(r1.sum())
    b["noplan"] = b[ID].isin(npv.index)
    b["np_v3cal"] = npv.cal.reindex(b[ID]).to_numpy()
    rec["noplan"] = {"rows": int(b.noplan.sum()), "np_base_filled": int(filled.sum()), "fill": fill,
                     "class_model": chosen, "c43_cell": cell}
    if r1_a is not None:
        rec["noplan"] |= {"r1_base_weight": r1_a, "r1_rows": r1_rows}
    # C39b practice
    sc = json.loads(PRACTICE_REC.read_text())["rows_2026"]
    tab = pd.DataFrame(sc["status_table_twelve_months"]).set_index("pre")
    rec["practice"] = add_practice(b, raw[PRACTICE_COLS], tab, sc["rate_new_twelve_months"])
    # C55 stand floor
    st = SF.bounds(raw[SF.INPUT_COLUMNS]).set_index(ID)
    assert set(st.index) == set(ids)
    b["stand_floor"] = st.stand_floor.reindex(ids).to_numpy(float)
    # class
    b["cls"] = np.where(dep.AIRCRAFT_TYPE_mvt.isna(), "noac", np.where(ck, "clock", "ordinary")).astype(object)
    # C73
    pin = lambda p: path if Path(p).name == "ranking.parquet" else Path(p)  # noqa: E731
    b2, _, _, c73stats = C73.add_key(b[[ID, "practice"]], C73.raw_frame(pin), "ranking")
    for k in ("c73", "c73_weight", "c73_point"):
        b[k] = b2[k].to_numpy()
    rec["c73"] = c73stats | {"selected_routes": [r["route"] for r in c73stats["selected_routes"]]}
    assert not (b.noplan & (b.rule | b.extension)).any()

    # ---------------- rule frame (stack9.ranking_frame without F)
    rr = NS9["rawframe"](raw[RAWCOLS])
    f = rr[rr.PHASE_mvt.eq("DEP")].set_index(ID).loc[ids].reset_index()
    f["cls"] = b.cls.to_numpy()
    f["t"] = (f.MVT_TIME_UTC_mvt - T0).dt.total_seconds()
    f["scope"] = f.plan & f.ADES_mvt.ne(f.ap)
    g, g71, a, c, _ = NS9["stand_inputs"](raw[RAWCOLS], f[ID])
    f["g"], f["g71"], f["c71"] = g, g71, a | c
    assert not (f.c71.to_numpy() & b.stand_floor.notna().to_numpy()).any()

    # ---------------- exploit frame (exploit029_prepare.main, serving lines)
    tr = training()
    rank = raw[COLS].copy()
    rank["src"], rank["pos"] = "ranking", np.arange(len(rank))
    allraw = pd.concat([tr, rank], ignore_index=True)
    links, _ = unique_links(allraw[LINK_COLS])
    links = links.reset_index(drop=True)
    order = K.c65(allraw)
    u = K.c58_mask(tr)
    fit58 = K.c58_fit(u)
    saved58 = json.loads(C58_REC.read_text())["ranking"]
    # The fit reads only the twelve 2025 files, so it is the saved one. lstsq gives last-bit differences between LAPACK
    # builds (remote OpenBLAS); the saved coefficients keep point58 the same on every machine.
    assert fit58["training_digest"] == saved58["training_digest"]   # it covers the training IDs; the record holds no IDs
    c_fit, c_saved = np.asarray(fit58["coefficients"]), np.asarray(saved58["coefficients"])
    np.testing.assert_allclose(c_fit, c_saved, rtol=1e-9, atol=0)
    rec["c58_fit"] = {"coefficients": "saved ranking record", "bit_equal": bool((c_fit == c_saved).all()),
                      "max_abs_diff": float(np.abs(c_fit - c_saved).max())}
    fit58 = dict(fit58, coefficients=saved58["coefficients"])
    bundle = b.assign(bet=False)                  # bet feeds only the output column of the same name
    parent_rule = f[[ID, "c71", "g71"]]
    e = K.rule_frame(rank, ids, links, order, bundle, parent_rule, fit58, [])
    np.testing.assert_array_equal(e[ID].to_numpy("int64"), ids)
    rec["links"] = int(len(links))
    rec["c58_rows_in_file"] = int(len(K.c58_mask(rank)))

    frames = {"bundle": b, "rule_frame": f, "exploit_frame": e}
    out = {}
    for name, cols in FRAMES.items():
        x = frames[name]
        z = pd.DataFrame({ID: ids, "month": b.month.to_numpy("int64")})
        for k in cols:
            z[k] = x[k].to_numpy().astype(DTYPES[name][k]) if DTYPES[name][k] != "object" else x[k].to_numpy(object)
        out[name] = z
    rec["counts"] = counts(out)
    return out, rec


def counts(out: dict) -> dict:
    """Rows per month of every mask column and finite rows of every value column."""
    res = {}
    for name, z in out.items():
        for k in FRAMES[name]:
            v = z[k]
            if v.dtype == bool:
                m = v.to_numpy()
            elif v.dtype == object:
                for c in sorted(set(v)):
                    res[f"{name}.{k}={c}"] = {int(mo): int(n) for mo, n in (v == c).groupby(z.month).sum().items()}
                continue
            else:
                m = np.isfinite(v.to_numpy())
            res[f"{name}.{k}"] = {int(mo): int(n) for mo, n in pd.Series(m).groupby(z.month.to_numpy()).sum().items()}
    return res


def parity_inputs() -> tuple[pd.Series, pd.Series]:
    h = pd.read_parquet(HB1_RANK)                 # clock rows only; read_stacks.align gives NaN (not clock) elsewhere
    ids = pd.read_parquet(D.submitting_file("ranking"), columns=[ID])[ID].to_numpy("int64")
    hb1 = pd.Series(h.pred_HB1.to_numpy(float), index=h[ID].to_numpy("int64")).reindex(ids)
    clock = pd.Series(np.isfinite(hb1.to_numpy()), index=ids)
    e = pd.read_parquet(NP.BASE_ENTRY)
    assert sha(NP.BASE_ENTRY) == NP.BASE_ENTRY_SHA
    base = pd.Series(e.TAXITIME_SEC_mvt.to_numpy("float64"), index=e[ID].to_numpy("float64").astype("int64"))
    return clock, base


def column_sha(v: pd.Series) -> str:
    """SHA-256 of a column's dtype and values (NaN bits made one; text as JSON with null for a missing value)."""
    a = v.to_numpy()
    head = (str(v.dtype) + "\0").encode()
    if a.dtype.kind == "f":
        body = np.where(np.isnan(a), np.nan, a).astype("<f8").tobytes()
    elif a.dtype.kind in "biu":
        body = a.astype(a.dtype.newbyteorder("<")).tobytes()
    else:
        body = json.dumps([None if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x) for x in a]).encode()
    return hashlib.sha256(head + body).hexdigest()


def digests(out: dict) -> dict:
    return {f"{name}.{k}": column_sha(out[name][k]) for name in FRAMES for k in FRAMES[name]} | \
           {"ids": hashlib.sha256(out["bundle"][ID].to_numpy("<i8").tobytes()).hexdigest()}


def check(out: dict) -> dict:
    """Ranking mode: every built column against the pack's saved column (byte equality, NaN = NaN)."""
    import pack
    res = {}
    for name in FRAMES:
        ref = pack.frame(f"ranking/{name}")
        assert np.array_equal(ref[ID].to_numpy(), out[name][ID].to_numpy())
        for k in FRAMES[name]:
            a, r = out[name][k].to_numpy(), ref[k].to_numpy()
            if r.dtype == object:
                eq = np.array([x == y for x, y in zip(a, r)])
            else:
                a, r = a.astype(r.dtype), r
                eq = (a == r) | (pd.isna(a) & pd.isna(r)) if r.dtype.kind == "f" else (a == r)
            ok = bool(eq.all()) and str(out[name][k].dtype) == str(ref[k].dtype)
            entry = {"pass": ok, "differ": int((~eq).sum()), "dtype": str(out[name][k].dtype), "ref_dtype": str(ref[k].dtype)}
            if r.dtype.kind == "f" and not eq.all():
                d = np.abs(np.asarray(a, float) - np.asarray(r, float))
                entry["max_abs"] = float(np.nanmax(d)) if np.isfinite(d).any() else None
                entry["nan_mismatch"] = int((pd.isna(a) != pd.isna(r)).sum())
            res[f"{name}.{k}"] = entry
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["ranking", "final"])
    ap.add_argument("--parity", action="store_true", help="ranking mode: clock and np_base from the saved v73-era inputs")
    ap.add_argument("--clock", type=Path, help="parquet MVT_ID_mvt, clock (bool: the clock head is finite)")
    ap.add_argument("--np-base", type=Path, help="parquet MVT_ID_mvt, np_base on every LIRF no-plan departure")
    ap.add_argument("--fill", default="none", choices=["none", "mix"])
    ap.add_argument("--np-r1-a", type=float, help="base weight a of the no-plan r1 rows, max(60, a base + (1 - a) mix); "
                                                   "default: the served 0.75 (design row a = 0.444, REPORT-V 4.2)")
    ap.add_argument("--check", action="store_true", help="ranking mode: byte parity of every column against the pack")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, help="write <frame>.parquet and record.json here")
    ap.add_argument("--parts", action="store_true", help="write the parts to $PRC_FINAL_ROOT/finallirf/parts/ "
                                                         "(default $PRC_FINAL_ROOT): frames_<frame>[-tag].parquet")
    ap.add_argument("--tag", default="", help="part name suffix (a new input version gets a new name; parts are write-once)")
    ap.add_argument("--record", type=Path, help="write record.json (and parity) here when --out-dir is not given")
    ap.add_argument("--digest-ref", type=Path, help="ranking mode: compare column SHA-256 with this record "
                                                     "(out/v/ranking-pack-digests.json, the pack's columns)")
    a = ap.parse_args()
    NP.threads(a.threads)
    if a.parity:
        assert a.mode == "ranking" and a.clock is None and a.np_base is None
        clock, base = parity_inputs()
    else:
        c = pd.read_parquet(a.clock)
        clock = pd.Series(c.clock.to_numpy(bool), index=c[ID].to_numpy("float64").astype("int64"))
        n = pd.read_parquet(a.np_base)
        base = pd.Series(n.np_base.to_numpy("float64"), index=n[ID].to_numpy("float64").astype("int64"))
    out, rec = build(a.mode, clock, base, a.fill, a.np_r1_a)
    rec["inputs"] = {"parity": a.parity, "clock": str(a.clock) if a.clock else str(HB1_RANK.relative_to(ROOT)),
                     "clock_sha256": sha(a.clock) if a.clock else sha(HB1_RANK),
                     "np_base": str(a.np_base) if a.np_base else str(NP.BASE_ENTRY.relative_to(ROOT)),
                     "np_base_sha256": sha(a.np_base) if a.np_base else NP.BASE_ENTRY_SHA}
    rec["code_sha256"] = {p.name: sha(p) for p in (Path(__file__), Path(NP.__file__), LINKS_SRC, XC / "stack9.py",
                                                    XC / "read_stacks.py", XC / "prepare_unit2.py", Path(C73.__file__),
                                                    Path(K.__file__), Path(R.__file__), Path(SF.__file__), Path(P.__file__),
                                                    S9 / "u27_floor3.py", S9 / "u13_inbound.py")}
    rec["column_sha256"] = digests(out)
    if a.check:
        assert a.mode == "ranking"
        rec["parity"] = check(out)
        bad = {k: v for k, v in rec["parity"].items() if not v["pass"]}
        print(f"PARITY {len(rec['parity']) - len(bad)} of {len(rec['parity'])} columns byte-equal")
        for k, v in bad.items():
            print("  DIFFER", k, v)
    if a.digest_ref:
        assert a.mode == "ranking"
        ref = json.loads(a.digest_ref.read_text())
        same = {k: rec["column_sha256"][k] == ref[k] for k in ref}
        rec["digest_parity"] = {"ref": str(a.digest_ref), "equal": sum(same.values()), "of": len(same),
                                "differ": sorted(k for k, v in same.items() if not v)}
        print(f"DIGEST PARITY {rec['digest_parity']['equal']} of {len(same)} equal; differ {rec['digest_parity']['differ']}")
    if a.parts:
        root = Path(os.environ.get("PRC_FINAL_ROOT", os.environ.get("PRC_FINAL_ROOT", "/path/to/final"))) / "finallirf/parts"
        root.mkdir(parents=True, exist_ok=True)
        sfx = f"-{a.tag}" if a.tag else ""
        rec["parts"] = {}
        for name, z in out.items():
            dest = root / f"frames_{name}{sfx}.parquet"
            tmp = dest.with_suffix(".tmp")
            z.to_parquet(tmp, index=False)
            if dest.exists():
                assert sha(dest) == sha(tmp), f"write-once part {dest} exists with other bytes; use a new --tag"
                tmp.unlink()
            else:
                tmp.rename(dest)
            rec["parts"][name] = {"path": str(dest), "sha256": sha(dest), "rows": int(len(z)), "columns": list(z.columns)}
        (root / f"frames-record{sfx}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
        print(json.dumps(rec["parts"], indent=1))
    if a.out_dir:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        for name, z in out.items():
            z.to_parquet(a.out_dir / f"{name}.parquet", index=False)
        (a.out_dir / "record.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    elif a.record:
        a.record.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({k: rec[k] for k in ("departures", "links", "noplan", "c58_rows_in_file")}, default=str))


if __name__ == "__main__":
    main()
