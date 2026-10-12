"""exp-062 ownvalue-refine: the near-origin fits (SPEC sections 3 and 4), one at a time, after a
zero-fit plan stage.  It reads no target, residual, base, feature or fitted value to select a row,
and it computes no statistic of any vector beyond its SHA-256 and its moved-row count.

    vp_ctl.py --kind plan  --log logs/plan.log  -- .venv/bin/python .../vp_fit.py --plan
    vp_ctl.py --kind fit-stage1 --log logs/fits-stage1.log -- .venv/bin/python .../vp_fit.py --run
    vp_ctl.py --kind plan2 --log logs/plan2.log -- .venv/bin/python .../vp_fit.py --plan2
    vp_ctl.py --kind fit-stage2 --log logs/fits-stage2.log -- .venv/bin/python .../vp_fit.py --run2

--plan  recomputes the six matched counts n_a from min(n_writable(2025-02, a), n_writable(2025-06, a)),
        rebuilds the whole k/24 grid for k = 1..24 out of exp-058's frozen June order, and REFUSES
        unless the per-airport count vector, the count-vector SHA-256 and both ordered-id hashes of
        k = 1, 2, 3, 4, 8, 12 and 24 equal the values exp-059's committed plan.json registered, the
        23-row total-N sequence equals the registered one, and k = 24 is exp-058's matched June
        block.  It then builds every member panel and every fitting frame of the nine stage-1 fits
        and fits nothing: `chainfold.fit_arm` is called exactly as the fit will call it, with
        `arms.arm_lgb` replaced by a stub that hashes its inputs and stops the call before any
        LightGBM object exists.  Writes analysis/plan.json.
--plan2 does the same for the conditional June* arm at L1, L2 and L3, after analysis/probe.json
        fixes June*.  Its grid row must equal analysis/plan.json's.  Writes analysis/plan2.json.
--run / --run2 take the registered fits in order, one at a time.  Before LightGBM is called a
        wrapper requires the fitting-frame hash and `fit_arm`'s provenance to equal the plan's,
        every opened repository file to be inside the positive manifest, and the code hash to equal
        the plan's.  Every attempt -- completed, stopped, failed or guard-refused -- is one
        immutable line in analysis/launches.jsonl and counts against the 18-launch cap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
ROOT_S = str(ROOT) + os.sep
E46 = ROOT / "experiments/exp-046-surfacefit"
E51 = ROOT / "experiments/exp-051-transferadv"
E59 = ROOT / "experiments/exp-059-ownvalue"
ANALYSIS = EXP / "analysis"
WORK = ROOT / "artifacts/exp-062/work"
MANIFEST = ANALYSIS / "manifest.json"
PLAN1 = ANALYSIS / "plan.json"
PLAN2 = ANALYSIS / "plan2.json"
OWN = ("experiments/exp-062-ownvalue-refine/", "artifacts/exp-062/")
F = "july_loo"
LEARNER_SEED = 46
SALT = "origincontrast-5801"
MONTH = "2025-06"
FEB_MONTH = "2025-02"
SHARED_AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LSZH")
NA = {"EDDF": 13_481, "EDDM": 9_293, "EGLL": 1_066, "EHAM": 17_381, "LEBL": 11_187, "LSZH": 8_369}
GRID_N = (2_536, 5_068, 7_601, 10_131, 12_666, 15_198, 17_728, 20_261, 22_794, 25_329, 27_859,
          30_391, 32_924, 35_454, 37_989, 40_521, 43_055, 45_585, 48_117, 50_652, 53_182, 55_715,
          58_247)
E58_N = 60_777
STEP_EXTRA = {0: 0, 1: 99_709, 2: 290_037, 3: 386_574}
PANEL_ROWS = 172_783
JUNE_ROWS = 98_396
JUNE_IDS_SHA16 = "61d1ecb2e76a9583"
E46_BAG = {1: "onemonth-bag10.json", 2: "threemonth-bag10.json", 3: "seasonal-bag10.json"}
LGB_VERSION = "4.7.0"
DYLIB = ROOT / ".venv/lib/python3.11/site-packages/lightgbm/lib/lib_lightgbm.dylib"
DYLIB_SHA = "bc392db609d97730a9ed2acec7a56529b356c03dfad39bebf689523fed7dff18"
#: SPEC 4: the grid rows whose registered hashes the plan stage must reproduce
VERIFY_K = (1, 2, 3, 4, 8, 12, 24)

sys.path.insert(0, str(HERE))
import vp_ctl as CTL                                                  # noqa: E402

# ---------------------------------------------------------------- what the process opens --

OPENED: dict = {}


def _note(path, how: str) -> None:
    try:
        p = os.fsdecode(path)
    except TypeError:
        return
    if not os.path.isabs(p):
        p = os.path.join(os.getcwd(), p)
    p = os.path.normpath(p)
    if p.startswith(ROOT_S):
        rel = p[len(ROOT_S):]
        if rel.startswith((".venv/", ".git/")) or "__pycache__" in rel:
            return
        OPENED.setdefault(rel, how)


def _hook(event, args) -> None:
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _note(args[0], "open")


sys.addaudithook(_hook)


def _wrap_pyarrow() -> None:
    """pyarrow opens files in C++, outside the `open` audit event: record its sources too."""
    import pyarrow as pa
    import pyarrow.dataset as pds
    import pyarrow.parquet as pq

    def src(x) -> None:
        if isinstance(x, (list, tuple)):
            for y in x:
                src(y)
        elif isinstance(x, (str, bytes, os.PathLike)):
            _note(x, "pyarrow")
        else:
            n = getattr(x, "name", None)
            if isinstance(n, (str, bytes)):
                _note(n, "pyarrow-handle")

    def wrap(mod, name):
        orig = getattr(mod, name)

        def f(source, *a, **k):
            src(source)
            return orig(source, *a, **k)
        f.__wrapped__ = orig
        setattr(mod, name, f)

    for name in ("read_table", "read_schema", "read_metadata"):
        wrap(pq, name)
    wrap(pds, "dataset")
    wrap(pa, "memory_map")
    pf_init = pq.ParquetFile.__init__

    def init(self, source, *a, **k):
        src(source)
        return pf_init(self, source, *a, **k)
    pq.ParquetFile.__init__ = init


_wrap_pyarrow()

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(files: dict) -> str:
    return hashlib.sha256("".join(f"{k} {v}\n" for k, v in sorted(files.items())).encode()).hexdigest()


# ---------------------------------------------------------------- the positive source boundary --

MAN = json.loads(MANIFEST.read_text())
SOURCE = MAN["source_boundary"]["files"]
E59_CENSUS = {k: v["sha256"] for k, v in
              json.loads((E59 / "analysis/prehash.json").read_text())["files"].items()}
REG_GRID = MAN["exp059_sources"]["per_airport_prefix_counts"]
#: SPEC 2.4, amended: the effective pin of each code file is its manifest pin, amended by the
#: committed append-only `analysis/code-pins.jsonl`; vp_ctl.code_pins() is the one reader
CODE_PINS = CTL.code_pins()
for _rel, _sha in CODE_PINS["effective"].items():
    if sha_file(ROOT / _rel) != _sha:
        raise SystemExit(f"{_rel} differs from its effective code pin; nothing runs")
for rel in ("experiments/exp-051-transferadv/code/carrier.py",
            "experiments/exp-051-transferadv/code/common.py",
            "experiments/exp-051-transferadv/code/memberspread.py"):
    if sha_file(ROOT / rel) != SOURCE[rel]:
        raise SystemExit(f"{rel} differs from the manifest; nothing is imported")
sys.path.insert(0, str(E51 / "code"))
import carrier                                                        # noqa: E402

C = sys.modules["common"]
A, CF, NO = C.A, C.CF, C.NO
for name, want in (("carrier", E51 / "code/carrier.py"), ("common", E51 / "code/common.py"),
                   ("memberspread", E51 / "code/memberspread.py"),
                   ("arms", E46 / "code/arms.py"), ("chainfold", E46 / "code/chainfold.py"),
                   ("nooverwrite", E46 / "code/nooverwrite.py")):
    got = Path(sys.modules[name].__file__).resolve()
    if got != want:
        raise SystemExit(f"module {name} resolves to {got}, not {want} (module name collision)")
if A is not sys.modules["arms"] or CF is not sys.modules["chainfold"]:
    raise SystemExit("common's A / CF are not the arms / chainfold modules")

_CHECKED: dict = {}


def repo_modules() -> dict:
    out = {}
    for m in list(sys.modules.values()):
        fn = getattr(m, "__file__", None)
        if not fn:
            continue
        p = os.path.normpath(os.path.abspath(fn))
        if p.startswith(ROOT_S) and not p[len(ROOT_S):].startswith(".venv/"):
            out[p[len(ROOT_S):]] = sha_file(Path(p))
    return dict(sorted(out.items()))


def source_check(rels) -> dict:
    """SPEC 2.5: every repository file this process opened, outside this experiment's own paths,
    must be inside the positive manifest with those bytes.  A path that is not in the manifest but
    IS in exp-059's verified census, with the hash recorded there, is allowed and recorded
    separately (the assignment allows new unrelated paths outside the old manifest; this keeps the
    boundary positive either way).  Anything else refuses a fit."""
    inside, from_census, outside, differ = 0, [], [], []
    for rel in sorted(rels):
        if rel.startswith(OWN):
            continue
        p = ROOT / rel
        if not p.is_file():
            continue
        sha = _CHECKED.get(rel) or sha_file(p)
        _CHECKED[rel] = sha
        if rel in SOURCE:
            if SOURCE[rel] == sha:
                inside += 1
            else:
                differ.append({"path": rel, "manifest": SOURCE[rel], "now": sha})
        elif rel in E59_CENSUS:
            if E59_CENSUS[rel] == sha:
                from_census.append(rel)
            else:
                differ.append({"path": rel, "exp059_census": E59_CENSUS[rel], "now": sha})
        else:
            outside.append({"path": rel, "now": sha})
    return {"in_manifest": inside, "in_exp059_census_only": from_census,
            "outside_every_pin": outside, "differ": differ,
            "clean": not (differ or outside)}


def code_and_sources() -> tuple[dict, dict]:
    mods = repo_modules()
    for rel, sha in mods.items():
        if rel.startswith(OWN):
            continue
        want = SOURCE.get(rel) or E59_CENSUS.get(rel)
        if want != sha:
            raise SystemExit(f"imported module {rel} is not pinned with these bytes")
    src = {}
    for rel in sorted(OPENED):
        if rel.startswith(OWN) or rel.endswith(".py") or not (ROOT / rel).is_file():
            continue
        src[rel] = _CHECKED.get(rel) or sha_file(ROOT / rel)
    return mods, src


# ---------------------------------------------------------------- SPEC 3: the June grid --

def id_str(v: float) -> str:
    i = int(v)
    if float(i) != float(v) or not (1 <= i < (1 << 53)):
        raise SystemExit(f"MVT_ID_mvt {v!r} is not an integral id in [1, 2^53)")
    return str(i)


def ordered_sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


def ceil_k(k: int, n: int) -> int:
    e = -((-k * n) // 24)
    if e != math.ceil(k / 24 * n):
        raise SystemExit(f"ceil disagreement at k={k}, n={n}")
    return e


def month_writable(ctx: dict, month: str) -> pd.DataFrame:
    fr = C.ext_frame(ctx["scols"], [month], ctx["cache"])
    if fr is None or not len(fr):
        raise SystemExit(f"{month}: no writable rows")
    return fr


def counts_by_airport(fr: pd.DataFrame) -> dict:
    air = fr.airport.astype(str).to_numpy()
    return {a: int((air == a).sum()) for a in sorted(set(air))}


def june_order(fr: pd.DataFrame) -> tuple:
    """exp-058's frozen June order inside each shared airport.  Reads only the id, the constant
    month label and the airport: no target, residual, base, feature or fitted value."""
    ids = fr.MVT_ID_mvt.to_numpy("float64")
    air = fr.airport.astype(str).to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise SystemExit(f"{MONTH}: duplicate MVT_ID_mvt among the writable rows")
    out = {}
    for a in SHARED_AIRPORTS:
        pos = np.flatnonzero(air == a)
        if pos.size == 0:
            raise SystemExit(f"{MONTH}: airport {a} has no writable row (empty stratum)")
        keys = [hashlib.sha256(f"{SALT}|{MONTH}|{a}|{id_str(ids[i])}".encode("utf-8")).hexdigest()
                for i in pos]
        order = sorted(range(pos.size), key=lambda j: (keys[j], ids[pos[j]]))
        out[a] = {"positions": pos[np.array(order, dtype="int64")],
                  "keys": [keys[j] for j in order], "writable": int(pos.size)}
    return out, ids


def grid_row(order: dict, ids: np.ndarray, k: int) -> tuple:
    keep, per = [], {}
    for a in SHARED_AIRPORTS:
        n_a = NA[a]
        take_n = ceil_k(k, n_a)
        if take_n <= 0:
            raise SystemExit(f"k={k} {a}: empty stratum")
        if take_n > order[a]["positions"].size:
            raise SystemExit(f"k={k} {a}: {take_n} rows exceed the {order[a]['writable']} writable")
        take = order[a]["positions"][:take_n]
        keep.append(take)
        per[a] = {"writable": order[a]["writable"], "n_a": n_a, "selected": int(take.size),
                  "first_key": order[a]["keys"][0],
                  "last_selected_key": order[a]["keys"][take_n - 1],
                  "first_unselected_key": order[a]["keys"][take_n]
                  if take_n < order[a]["positions"].size else None}
        if take.size != take_n:
            raise SystemExit(f"k={k} {a}: selected {take.size} rows, not {take_n}")
    sel = np.sort(np.concatenate(keep))
    if len(np.unique(sel)) != len(sel):
        raise SystemExit(f"k={k}: a row was selected twice")
    sel_ids = ids[sel]
    if len(np.unique(sel_ids)) != len(sel_ids):
        raise SystemExit(f"k={k}: duplicate MVT_ID_mvt in the block")
    cv = [per[a]["selected"] for a in SHARED_AIRPORTS]
    rec = {"k": k, "f": f"{k}/24", "month": MONTH, "salt": SALT,
           "shared_airports": list(SHARED_AIRPORTS), "block_rows": int(sel.size),
           "by_airport": per, "count_vector": cv,
           "count_vector_sha256": ordered_sha(np.array(cv, "int64")),
           "ids_sha256_block_order": ordered_sha(sel_ids),
           "ids_sha256_sorted": ordered_sha(np.sort(sel_ids)),
           "ids_sha16": C.ids_sha16(sel_ids)}
    return sel, rec


def build_grid(ctx: dict) -> tuple:
    frF = month_writable(ctx, FEB_MONTH)
    frJ = month_writable(ctx, MONTH)
    cF, cJ = counts_by_airport(frF), counts_by_airport(frJ)
    missing = [a for a in SHARED_AIRPORTS if cF.get(a, 0) == 0 or cJ.get(a, 0) == 0]
    if missing:
        raise SystemExit(f"shared airports missing or empty: {missing}")
    n_a = {a: min(cF[a], cJ[a]) for a in SHARED_AIRPORTS}
    if n_a != NA:
        raise SystemExit(f"the matched counts {n_a} are not exp-058's {NA}")
    order, ids = june_order(frJ)
    rows, blocks = {}, {}
    for k in range(1, 25):
        sel, rec = grid_row(order, ids, k)
        rows[k], blocks[k] = rec, sel
    seq = tuple(rows[k]["block_rows"] for k in range(1, 24))
    if seq != GRID_N:
        raise SystemExit(f"the total-N sequence {seq} is not the registered one {GRID_N}")
    #: SPEC 4: every verified grid row must equal the values exp-059's plan.json registered
    ver = {}
    for k in VERIFY_K:
        want, got = REG_GRID[str(k)], rows[k]
        same = (got["block_rows"] == want["N"] and got["count_vector"] == want["count_vector"]
                and got["count_vector_sha256"] == want["count_vector_sha256"]
                and got["ids_sha256_block_order"] == want["ids_sha256_block_order"]
                and got["ids_sha256_sorted"] == want["ids_sha256_sorted"])
        ver[str(k)] = {"equals_exp059_registered_row": bool(same), "N": got["block_rows"],
                       "count_vector": got["count_vector"],
                       "count_vector_sha256": got["count_vector_sha256"],
                       "ids_sha256_block_order": got["ids_sha256_block_order"],
                       "ids_sha256_sorted": got["ids_sha256_sorted"]}
        if not same:
            raise SystemExit(f"k={k} does not reproduce exp-059's registered grid row: {ver[str(k)]}")
    if rows[24]["block_rows"] != E58_N or rows[24]["count_vector"] != [NA[a] for a in SHARED_AIRPORTS]:
        raise SystemExit("k = 24 is not exp-058's 60,777-row block by count")
    for k in range(1, 24):
        if not set(ids[blocks[k]].tolist()) <= set(ids[blocks[k + 1]].tolist()):
            raise SystemExit(f"k={k} is not a subset of k={k + 1}")
        if any(rows[k]["count_vector"][i] > rows[k + 1]["count_vector"][i] for i in range(6)):
            raise SystemExit(f"k={k} has a larger per-airport count than k={k + 1}")
    out = {"salt": SALT, "month": MONTH, "shared_airports": list(SHARED_AIRPORTS),
           "writable_by_airport": {FEB_MONTH: cF, MONTH: cJ},
           "matched_counts_n_a": {a: n_a[a] for a in SHARED_AIRPORTS},
           "matched_counts_equal_exp058": True,
           "june_writable_rows": int(len(frJ)), "february_writable_rows": int(len(frF)),
           "registered_N_sequence": list(GRID_N), "N_sequence_reproduced": True,
           "verified_against_exp059_registered_rows": ver,
           "k24_equals_exp058_matched_june_block": True, "prefix_nesting_checked": True,
           "target_reads_during_row_selection": 0,
           "rows": {str(k): rows[k] for k in range(1, 25)}}
    return out, blocks


def block_frame(ctx: dict, k: int, blocks: dict) -> pd.DataFrame:
    fr = month_writable(ctx, MONTH)
    return fr.iloc[blocks[k]].reset_index(drop=True)


def fitting_extra(ctx: dict, arm: str, level: int, blocks: dict) -> tuple:
    """exp-051 `carrier.fitting_extra`'s assembly with this experiment's prefix block first."""
    k = int(arm[1:])
    blk = block_frame(ctx, k, blocks)
    rows = {"arm": arm, "k": k, "f": f"{k}/24", "block_month": MONTH,
            "block_rows": int(len(blk)), "block_ids_sha16": C.ids_sha16(blk.MVT_ID_mvt)}
    parts = [blk]
    months = carrier.STEPS_JULY[level]
    ext = C.ext_frame(ctx["scols"], months, ctx["cache"]) if months else None
    if ext is not None:
        rows["ext_months"] = months
        rows["ext_rows"] = int(len(ext))
        miss = [c for c in parts[0].columns if c not in ext.columns]
        if miss:
            raise SystemExit(f"{arm}-L{level}: the step months lack {miss[:8]}")
        parts.append(ext.reindex(columns=parts[0].columns))
    extra = parts[0] if len(parts) == 1 else pd.concat(
        [parts[0]] + [p.reindex(columns=parts[0].columns) for p in parts[1:]], ignore_index=True)
    return extra, rows


# ---------------------------------------------------------------- the fitting-frame hash --

class _PlanStop(BaseException):
    def __init__(self, h: str):
        super().__init__(h)
        self.h = h


def frame_hash(args: tuple, kwargs: dict) -> str:
    """exp-057's, exp-058's and exp-059's definition, unchanged: what `arms.arm_lgb` reads."""
    if len(args) != 3:
        raise SystemExit(f"arm_lgb was called with {len(args)} positional arguments, not 3")
    fit, sco, cols = args
    w = kwargs.get("w")
    meta = {"cols": list(cols), "kwargs": {k: v for k, v in kwargs.items() if k != "w"},
            "n_fit": int(len(fit)), "n_sco": int(len(sco)),
            "w": None if w is None else int(len(w))}
    h = hashlib.sha256(json.dumps(meta, sort_keys=True, default=str).encode())
    h.update(np.ascontiguousarray(fit[list(cols)].to_numpy("float32")).tobytes())
    h.update(np.ascontiguousarray(fit["rbase"].to_numpy("float64")).tobytes())
    for c in ("day", "airport"):
        h.update(pd.util.hash_pandas_object(fit[c], index=False).to_numpy().tobytes())
    h.update(np.ascontiguousarray(sco["MVT_ID_mvt"].to_numpy("float64")).tobytes())
    h.update(np.ascontiguousarray(sco["usable"].to_numpy(bool)).tobytes())
    h.update(np.ascontiguousarray(sco[list(cols)].to_numpy("float32")).tobytes())
    h.update(pd.util.hash_pandas_object(sco["airport"], index=False).to_numpy().tobytes())
    if w is not None:
        h.update(np.asarray(w, dtype="float64").tobytes())
    return h.hexdigest()


def jround(x):
    return json.loads(json.dumps(x, sort_keys=True, default=float))


ARM_LGB, SETTING0, HOLD0 = A.arm_lgb, CF.SETTING, CF.HOLD


def call_fit_arm(ctx: dict, s: int, extra, arm_lgb, prov: dict) -> dict:
    CF.SETTING, CF.HOLD = {"num_leaves": C.LEAVES}, ()
    A.arm_lgb = arm_lgb
    try:
        return CF.fit_arm(ctx["pans"][s], prov=prov, extra=extra, folds=(F,))
    finally:
        A.arm_lgb, CF.SETTING, CF.HOLD = ARM_LGB, SETTING0, HOLD0


def restored() -> None:
    if A.arm_lgb is not ARM_LGB or CF.SETTING is not SETTING0 or CF.HOLD is not HOLD0:
        raise SystemExit("SETTING, HOLD or arm_lgb was not restored")


def plan_one(ctx: dict, arm: str, s: int, L: int, blocks: dict) -> dict:
    t0 = time.time()
    extra, rows = fitting_extra(ctx, arm, L, blocks)

    def stub(*a, **k):
        raise _PlanStop(frame_hash(a, k))
    prov: dict = {}
    try:
        call_fit_arm(ctx, s, extra, stub, prov)
        raise SystemExit(f"{arm}-s{s}-L{L}: the stub was not reached")
    except _PlanStop as e:
        h = e.h
    restored()
    return {"tag": f"{arm}-s{s}-L{L}", "arm": arm, "k": int(arm[1:]), "month": MONTH, "seed": s,
            "level": L, "extra_rows": int(len(extra)), "rows": jround(rows),
            "provenance": jround(prov[F]), "frame_sha256": h, "seconds": round(time.time() - t0, 1)}


def context(s: int, shared: dict) -> dict:
    t1 = time.time()
    ctx = carrier.panels([s])
    if "scols" in shared and shared["scols"] != ctx["scols"]:
        raise SystemExit(f"seed {s}: the shipped columns differ from the first seed's")
    shared.setdefault("scols", ctx["scols"])
    ctx["cache"] = shared.setdefault("cache", {})
    print(f"seed {s}: panel built in {time.time() - t1:.0f}s", flush=True)
    return ctx


def checks_for(e: dict, N: int) -> dict:
    arm, s, L = e["arm"], e["seed"], e["level"]
    r, p = e["rows"], e["provenance"]
    out = {"d_block": r["block_rows"] == N and r["block_month"] == MONTH,
           "d_counts": (e["extra_rows"] == N + STEP_EXTRA[L] and p["extra_rows"] == N + STEP_EXTRA[L]
                        and p["panel_rows"] == PANEL_ROWS
                        and p["total_fit_rows"] == PANEL_ROWS + N + STEP_EXTRA[L]
                        and r.get("ext_months", []) == carrier.STEPS_JULY[L]
                        and r.get("ext_rows", 0) == STEP_EXTRA[L])}
    if L >= 1 and (s in (2, 3) or L == 3):
        rec = json.loads((E46 / "analysis" / E46_BAG[L]).read_text())["members"][str(s)]
        out["e_exp046_member_rows"] = (
            p["panel_rows"] + (e["extra_rows"] - N) == rec["total_fit_rows"][F]
            and STEP_EXTRA[L] == rec["extra_rows"][F])
    bad = [k for k, v in out.items() if v is not True]
    if bad:
        raise SystemExit(f"{e['tag']}: plan checks fail: {bad}")
    return out


def check_f(ctx: dict, s: int, i2plan: dict) -> list:
    """exp-051's FULL June carrier frames of seed s, against its i2-plan.json."""
    by_tag = {x["tag"]: x for x in i2plan["fits"]}
    out = []
    other = int(sum(ctx["pans"][s][g].usable.sum() for g in A.FOLDS if g != F))
    for L in range(4):
        extra, rows = carrier.fitting_extra(ctx, s, "C", L, None)
        tag = "integrity-C0-s1" if (s == 1 and L == 0) else f"C-s{s}-L{L}"
        want = by_tag[tag]
        got = {"rows": jround(rows), "extra_rows": int(len(extra)), "member_other_fold_rows": other,
               "total_fit_rows": other + int(len(extra))}
        ok = (all(got[k] == want[k] for k in got)
              and rows["june_rows"] == JUNE_ROWS and rows["june_ids_sha16"] == JUNE_IDS_SHA16)
        out.append({"tag": tag, "level": L, "seed": s, "equal_to_exp051_plan": ok, **got})
        if not ok:
            raise SystemExit(f"{tag}: the plan code does not reproduce exp-051 i2-plan.json")
    return out


def check_h() -> dict:
    import lightgbm as lgb
    pre = json.loads((E46 / "analysis/predeclared.json").read_text())
    out = {"lightgbm_version": lgb.__version__, "dylib_sha256": sha_file(DYLIB),
           "lgb_equals_predeclared": A.LGB == pre["lgb"], "learner_seed": A.SEED,
           "leaves": C.LEAVES, "setting_default": CF.SETTING, "hold_default": list(CF.HOLD)}
    ok = (out["lightgbm_version"] == LGB_VERSION and out["dylib_sha256"] == DYLIB_SHA
          and out["lgb_equals_predeclared"] and A.SEED == LEARNER_SEED and C.LEAVES == 255)
    if not ok:
        raise SystemExit(f"the learner settings are not the registered ones: {out}")
    return out


def write_json(p: Path, obj: dict) -> None:
    NO.refuse_existing([p], overwrite=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


def june_star() -> dict:
    p = ANALYSIS / "probe.json"
    if not p.exists():
        raise SystemExit("analysis/probe.json does not exist: stage 2 runs only after the probe")
    pr = json.loads(p.read_text())
    js = pr.get("june_star")
    if not js:
        raise SystemExit(f"the probe ended '{pr.get('branch')}': there is no June* and no stage 2")
    return js


def fits_of(stage: int) -> list:
    if stage == 1:
        return list(CTL.STAGE1)
    return CTL.stage2(int(june_star()["k"]))


# ---------------------------------------------------------------- the plan --

def do_plan(stage: int) -> None:
    plan_path = PLAN1 if stage == 1 else PLAN2
    NO.refuse_existing([plan_path], overwrite=False)
    t0, c0 = time.time(), time.process_time()
    i2plan = json.loads((E51 / "analysis/i2-plan.json").read_text())
    h = check_h()
    spec = fits_of(stage)
    arms = sorted({a for a, _, _ in spec})
    ref = json.loads(PLAN1.read_text())["grid"] if stage == 2 else None
    fits, f_checks, shared = [], [], {}
    grid, blocks = None, None
    for s in (1, 2, 3):
        ctx = context(s, shared)
        if grid is None:
            grid, blocks = build_grid(ctx)
            if ref is not None and jround(grid) != ref:
                raise SystemExit("the grid differs from analysis/plan.json's")
            print("grid: " + ", ".join(
                f"k{int(a[1:])} N={grid['rows'][str(int(a[1:]))]['block_rows']:,}" for a in arms),
                flush=True)
        for arm, s2, L in spec:
            if s2 != s:
                continue
            e = plan_one(ctx, arm, s, L, blocks)
            e["checks"] = checks_for(e, grid["rows"][str(int(arm[1:]))]["block_rows"])
            fits.append(e)
            print(f"  {e['tag']:11} extra {e['extra_rows']:>7,} total "
                  f"{e['provenance']['total_fit_rows']:>7,} frame {e['frame_sha256'][:16]} "
                  f"({e['seconds']}s)", flush=True)
        f_checks += check_f(ctx, s, i2plan)
        print(f"  seed {s}: exp-051's full June carrier frames equal its i2-plan.json", flush=True)
        del ctx
    order = [f"{a}-s{x}-L{L}" for a, x, L in spec]
    fits.sort(key=lambda e: order.index(e["tag"]))
    sc = source_check(OPENED)
    mods, src = code_and_sources()
    rec = {"what": f"exp-062 plan stage {stage} (SPEC 3, 4): the near-origin grid, the fitting sets, "
                   f"the provenance and the fitting-frame hashes of {', '.join(arms)}; no fit",
           "stage": stage, "manifest_sha256": sha_file(MANIFEST),
           "manifest_new_code_digest": MAN["new_code"]["digest"],
           "spec_commit": MAN["registration"]["spec_commit"],
           "spec_sha256": MAN["registration"]["spec_sha256"],
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "arms": {a: {"k": int(a[1:]), "f": f"{int(a[1:])}/24",
                        "N": grid["rows"][str(int(a[1:]))]["block_rows"]} for a in arms},
           "levels": sorted({L for _, _, L in spec}),
           "grid": grid, "fit_order": order, "fits": fits,
           "check_exp051_full_june_frames": f_checks, "check_learner": h,
           "source_check": sc, "opened_by_how": dict(sorted(OPENED.items())),
           "code": {"sha256": digest(mods), "files": mods},
           "sources": {"sha256": digest(src), "files": src},
           "all_checks_pass": bool(sc["clean"]),
           "seconds": round(time.time() - t0, 1),
           "seconds_process_cpu": round(time.process_time() - c0, 1)}
    if stage == 2:
        rec["june_star"] = june_star()
    out = plan_path if rec["all_checks_pass"] else plan_path.with_name(f"{plan_path.stem}-source-fail.json")
    write_json(out, rec)
    print(f"plan stage {stage}: {len(fits)} fits over {arms}; source check: {sc['in_manifest']} files "
          f"in the manifest, {len(sc['in_exp059_census_only'])} in exp-059's census only, "
          f"{len(sc['outside_every_pin'])} outside every pin, {len(sc['differ'])} differ; "
          f"code {rec['code']['sha256'][:12]} ({len(mods)} files), sources "
          f"{rec['sources']['sha256'][:12]} ({len(src)} files); {rec['seconds']}s wall / "
          f"{rec['seconds_process_cpu']}s process cpu", flush=True)
    if not rec["all_checks_pass"]:
        raise SystemExit(f"the fit-input check fails: no fit launches. {json.dumps(sc, indent=1)}")


# ---------------------------------------------------------------- the fits --

def launch_line(obj: dict) -> None:
    CTL.append_line(CTL.LAUNCHES, obj)


def verify_done(tag: str, plan: dict) -> None:
    rec = json.loads((ANALYSIS / "fits" / f"{tag}.json").read_text())
    p = ROOT / rec["vector"]
    if rec["code_sha256"] != plan["code"]["sha256"] or not p.exists() \
            or sha_file(p) != rec["vector_sha256"]:
        raise SystemExit(f"{tag}: the existing record does not match its vector or the plan's code")
    want = {e["tag"]: e for e in plan["fits"]}[tag]
    if rec["frame_sha256"] != want["frame_sha256"] or rec["fit_arm_provenance"] != want["provenance"]:
        raise SystemExit(f"{tag}: the recorded fitting frame is not this plan's")


def do_run(stage: int) -> None:
    plan_path = PLAN1 if stage == 1 else PLAN2
    plan = json.loads(plan_path.read_text())
    spec = fits_of(stage)
    order = [f"{a}-s{s}-L{L}" for a, s, L in spec]
    if not plan["all_checks_pass"]:
        raise SystemExit(f"{plan_path.name} does not pass")
    if plan["fit_order"] != order:
        raise SystemExit("the fit order differs from the plan's")
    if plan["manifest_new_code_digest"] != MAN["new_code"]["digest"]:
        raise SystemExit("the plan was written against other code than the manifest now pins")
    by_tag = {e["tag"]: e for e in plan["fits"]}
    c12, s12 = plan["code"]["sha256"][:12], plan["sources"]["sha256"][:12]
    i2plan = json.loads((E51 / "analysis/i2-plan.json").read_text())
    check_h()
    t_job, cpu_job = time.time(), time.process_time()
    n_fitted, shared, blocks, grid = 0, {}, None, None
    ctx, cur_seed, checked_f = None, None, set()
    for arm, s, L in spec:
        tag = f"{arm}-s{s}-L{L}"
        if (ANALYSIS / "fits" / f"{tag}.json").exists():
            verify_done(tag, plan)
            print(f"skip {tag}: record and vector exist", flush=True)
            continue
        if cur_seed != s:
            ctx = None
            ctx = context(s, shared)
            cur_seed = s
            if grid is None:
                grid, blocks = build_grid(ctx)
                if jround(grid) != plan["grid"]:
                    raise SystemExit("the grid differs from the plan's")
                print("the k/24 grid reproduces the plan's", flush=True)
            if s not in checked_f:
                check_f(ctx, s, i2plan)
                checked_f.add(s)
        grow = plan["grid"]["rows"][str(int(arm[1:]))]
        N = grow["block_rows"]
        b12 = grow["ids_sha256_block_order"][:12]
        js = ANALYSIS / "fits" / f"{tag}.json"
        npz = WORK / f"{tag}__code-{c12}__src-{s12}__rows-{b12}.npz"
        stray = sorted(WORK.glob(f"{tag}__*.npz")) if WORK.exists() else []
        if stray:
            raise SystemExit(f"{tag}: a vector without a record exists: {stray}")
        NO.refuse_existing([npz, js], overwrite=False)
        done = CTL.done_fits()
        launches = CTL.fit_launches()
        est_w, est_c = CTL.fit_projection(arm, L, done)
        ch = CTL.charged(exclude=os.environ.get("VP_ATTEMPT_ID"))
        used_w = ch["fit_wall"] + (time.time() - t_job)
        used_c = ch["fit_cpu"] + (time.process_time() - cpu_job)
        #: SPEC 12: the 18-launch cap and the fit allocation, checked BEFORE the attempt is entered
        if launches + 1 > CTL.MAX_FIT_LAUNCHES:
            raise SystemExit(f"{tag}: {launches} fit attempts are recorded; the cap is "
                             f"{CTL.MAX_FIT_LAUNCHES}. No fit launches.")
        if used_w + est_w > CTL.ALLOC_FIT_WALL_S or used_c + est_c > CTL.ALLOC_FIT_CPU_S:
            raise SystemExit(f"budget gate: {tag} is projected at {est_w:.0f}s wall / {est_c:.0f}s "
                             f"process cpu on top of {used_w:.0f}s / {used_c:.0f}s charged to the fit "
                             f"allocation, which is projected OVER the {CTL.ALLOC_FIT_WALL_S:.0f}s / "
                             f"{CTL.ALLOC_FIT_CPU_S:.0f}s fit allocation; no fit launches")
        load = carrier.hold_for_load(tag)
        extra, rows = fitting_extra(ctx, arm, L, blocks)
        want = by_tag[tag]
        if jround(rows) != want["rows"] or int(len(extra)) != want["extra_rows"]:
            raise SystemExit(f"{tag}: the fitting set differs from the plan's")
        aid = uuid.uuid4().hex
        launch_line({"record": "attempt", "attempt_id": aid, "tag": tag, "stage": stage,
                     "arm": arm, "k": int(arm[1:]), "seed": s, "level": L, "N": N,
                     "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "load_1min": load, "projected_wall_s": round(est_w, 1),
                     "projected_process_cpu_s": round(est_c, 1),
                     "launches_before": launches,
                     "fit_allocation_charged_before": {"wall_s": round(used_w, 1),
                                                       "process_cpu_s": round(used_c, 1)},
                     "plan_file": str(plan_path.relative_to(ROOT)),
                     "code_sha256": plan["code"]["sha256"],
                     "frame_sha256": want["frame_sha256"]})
        box = {"ok": False}
        prov: dict = {}

        def wrapper(*a, **k):
            hh = frame_hash(a, k)
            if hh != want["frame_sha256"]:
                raise SystemExit(f"{tag}: fitting-frame hash {hh[:16]} is not the plan's "
                                 f"{want['frame_sha256'][:16]}; the fit does not run")
            if jround(prov[F]) != want["provenance"]:
                raise SystemExit(f"{tag}: fit_arm's provenance differs from the plan's")
            sc = source_check(OPENED)
            if not sc["clean"]:
                raise SystemExit(f"{tag}: the fit-input check fails before the fit: {sc}")
            mods, src = code_and_sources()
            if digest(mods) != plan["code"]["sha256"]:
                raise SystemExit(f"{tag}: the code hash {digest(mods)[:12]} is not the plan's {c12}")
            box.update(ok=True, sc=sc, mods=mods, src=src,
                       src_not_in_plan={r: v for r, v in src.items()
                                        if r not in plan["sources"]["files"]})
            return ARM_LGB(*a, **k)

        print(f"launch {tag} at load {load:.2f} ({time.strftime('%H:%M:%S', time.gmtime())} UTC); "
              f"fit allocation used {used_w:.0f}s wall / {used_c:.0f}s cpu, this fit projected "
              f"{est_w:.0f}s / {est_c:.0f}s; attempt {launches + 1} of {CTL.MAX_FIT_LAUNCHES}",
              flush=True)
        t1, c1 = time.time(), time.process_time()
        try:
            d = call_fit_arm(ctx, s, extra, wrapper, prov)
        except BaseException as exc:
            launch_line({"record": "outcome", "attempt_id": aid, "tag": tag, "status": "failed",
                         "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "seconds_wall": round(time.time() - t1, 1),
                         "seconds_process_cpu": round(time.process_time() - c1, 1),
                         "reached_lightgbm": bool(box["ok"]),
                         "error": f"{type(exc).__name__}: {exc}"[:400]})
            raise
        secs, cpu = time.time() - t1, time.process_time() - c1
        restored()
        if not box["ok"]:
            raise SystemExit(f"{tag}: the wrapper was not reached")
        if prov[F]["extra_rows"] != len(extra):
            raise SystemExit(f"{tag}: fit_arm carried {prov[F]['extra_rows']} extra rows")
        vec = d[F].astype("float64")
        WORK.mkdir(parents=True, exist_ok=True)
        NO.refuse_existing([npz, js], overwrite=False)
        np.savez_compressed(npz, **{F: vec})
        n_fitted += 1
        if digest(repo_modules()) != plan["code"]["sha256"]:
            raise SystemExit(f"{tag}: a repository module was imported during the fit")
        sc2 = source_check(OPENED)
        if not sc2["clean"]:
            raise SystemExit(f"{tag}: a file opened during the fit fails the source check: {sc2}")
        rec = {"tag": tag, "arm": arm, "k": int(arm[1:]), "f": f"{int(arm[1:])}/24", "month": MONTH,
               "seed": s, "level": L, "fold": F, "stage": stage, "attempt_id": aid,
               "learner_seed": A.SEED, "leaves": C.LEAVES, "hold": [],
               "manifest_sha256": sha_file(MANIFEST),
               "spec_commit": MAN["registration"]["spec_commit"],
               "plan_file": str(plan_path.relative_to(ROOT)),
               "block_rows": want["rows"]["block_rows"],
               "block_ids_sha16": want["rows"]["block_ids_sha16"],
               "block_count_vector": grow["count_vector"],
               "block_count_vector_sha256": grow["count_vector_sha256"],
               "block_ids_sha256_block_order": grow["ids_sha256_block_order"],
               "block_ids_sha256_sorted": grow["ids_sha256_sorted"],
               "code_sha256": plan["code"]["sha256"], "code_files": box["mods"],
               "source_sha256": plan["sources"]["sha256"],
               "sources_opened_by_this_fit_not_in_plan": box["src_not_in_plan"],
               "rows": want["rows"], "fit_arm_provenance": jround(prov[F]),
               "frame_sha256": want["frame_sha256"],
               "seconds_fit_wall": round(secs, 1), "seconds_fit_process_cpu": round(cpu, 1),
               "job_wall_s_at_end": round(time.time() - t_job, 1),
               "job_process_cpu_s_at_end": round(time.process_time() - cpu_job, 1),
               "projected_wall_s": round(est_w, 1), "projected_process_cpu_s": round(est_c, 1),
               "load_1min_at_launch": load, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
               "vector": str(npz.relative_to(ROOT)), "vector_sha256": sha_file(npz),
               "rows_moved": int((vec != 0).sum())}
        write_json(js, rec)
        launch_line({"record": "outcome", "attempt_id": aid, "tag": tag, "status": "fitted",
                     "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "seconds_wall": round(secs, 1), "seconds_process_cpu": round(cpu, 1),
                     "reached_lightgbm": True, "vector_sha256": rec["vector_sha256"],
                     "rows_moved": rec["rows_moved"]})
        print(f"  fitted {npz.name} in {secs:.0f}s wall / {cpu:.0f}s cpu: fit rows "
              f"{prov[F]['total_fit_rows']:,}, moved {rec['rows_moved']:,}, "
              f"sha {rec['vector_sha256'][:16]}", flush=True)
    print(f"all {len(spec)} fits of stage {stage} present; {n_fitted} fitted by this job in "
          f"{time.time() - t_job:.0f}s wall, {time.process_time() - cpu_job:.0f}s process cpu; "
          f"{CTL.fit_launches()} of {CTL.MAX_FIT_LAUNCHES} fit attempts recorded", flush=True)


def main() -> None:
    ps = argparse.ArgumentParser()
    g = ps.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--plan2", action="store_true")
    g.add_argument("--run2", action="store_true")
    a = ps.parse_args()
    if a.plan:
        do_plan(1)
    elif a.plan2:
        do_plan(2)
    elif a.run:
        do_run(1)
    else:
        do_run(2)


if __name__ == "__main__":
    main()
