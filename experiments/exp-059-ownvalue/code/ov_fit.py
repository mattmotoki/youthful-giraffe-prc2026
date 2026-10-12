"""exp-059 ownvalue: the June ladder's fits (SPEC sections 3-4), one at a time, after a zero-fit plan.

    .venv/bin/python experiments/exp-059-ownvalue/code/ov_fit.py --plan
    .venv/bin/python experiments/exp-059-ownvalue/code/ov_fit.py --run-check
    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-059-ownvalue/code/ov_fit.py --run
    .venv/bin/python experiments/exp-059-ownvalue/code/ov_fit.py --plan3 K     (SPEC 5.2 step 5)
    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-059-ownvalue/code/ov_fit.py --run3 K

--plan counts the writable rows of 2025-02 and 2025-06 by airport, reproduces exp-058's six matched
counts `n_a`, builds SPEC section 3's whole k/24 grid (k = 1..24: per-airport counts, count-vector
SHA-256 and both ordered id hashes, checked against the registered total-N sequence and against
exp-058's k = 24 block), then builds every member panel and every fitting frame of the two
registered arms and fits nothing: for each fit `chainfold.fit_arm` is called exactly as the fit
will call it, with `arms.arm_lgb` replaced for the call by a stub that hashes its inputs and stops
the call before any LightGBM object exists.  It runs SPEC 4.4 (a)-(i), checks every repository file
the process opened against the committed census (SPEC 2.4), and writes analysis/plan.json.

--plan3 K does the same for one further grid arm, the third ladder point that SPEC 5.2 step 5
selects, and writes analysis/plan3.json; its grid row must equal analysis/plan.json's.

--run and --run3 take the fits in the registered order.  Before LightGBM is called for a fit, a
wrapper requires the fitting-frame hash and `fit_arm`'s provenance to equal the plan's, every newly
opened repository file to match the census, and the code hash to equal the plan's.  Before each
launch the budget gate of SPEC 4.3 projects the fit's wall and process-CPU cost from exp-058's
refitted law and from this experiment's completed fits, and refuses a launch that would take the
experiment's cumulative wall or process CPU (analysis/cost.json, every job) past its cap.
--run-check does every pre-fit check for the first fit and stops before LightGBM; it writes nothing.

The driver prints and records hashes, row counts, times and the count of rows a vector moves.  It
computes no other statistic of any vector, and it reads no target, residual, base or fitted value
to select a row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
ROOT_S = str(ROOT) + os.sep
E46 = ROOT / "experiments/exp-046-surfacefit"
E51 = ROOT / "experiments/exp-051-transferadv"
E58 = ROOT / "experiments/exp-058-origincontrast"
WORK = ROOT / "artifacts/exp-059/work"
ANALYSIS = EXP / "analysis"
PLAN = ANALYSIS / "plan.json"
CENSUS = ANALYSIS / "prehash.json"
COST = ANALYSIS / "cost.json"            #: SPEC 12: append-only, one write-once row per job
OWN = ("experiments/exp-059-ownvalue/", "artifacts/exp-059/")
SPEC_COMMIT = "49db4157d8bc43edde6c2b8386e90b6f58929578"
F = "july_loo"
LEARNER_SEED = 46
#: SPEC 3: exp-058's salt, month, airports and matched counts n_a, unchanged
SALT = "origincontrast-5801"
MONTH = "2025-06"
FEB_MONTH = "2025-02"
SHARED_AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LSZH")
NA = {"EDDF": 13_481, "EDDM": 9_293, "EGLL": 1_066, "EHAM": 17_381, "LEBL": 11_187, "LSZH": 8_369}
#: SPEC 3: the registered total-N sequence for k = 1..23
GRID_N = (2_536, 5_068, 7_601, 10_131, 12_666, 15_198, 17_728, 20_261, 22_794, 25_329, 27_859,
          30_391, 32_924, 35_454, 37_989, 40_521, 43_055, 45_585, 48_117, 50_652, 53_182, 55_715,
          58_247)
#: SPEC 3.4: the k = 24 verification point is exp-058's matched June block
E58_N = 60_777
E58_COUNT_VECTOR_SHA = "ee89d40963ea3718e88f73757d285786f4bbaafa017f343547a6b82d6776547a"
E58_IDS_SHA_BLOCK = "bdfa7542b4f48eacca11aa5bfab414a04994e6fe26ef385d52c59afcd5b211b8"
E58_IDS_SHA_SORTED = "c9770096dccbbe84fe9f3ecd9f3eb925ae1e2d42fff2bf69668fdc7062baa927"
#: SPEC 4.1: the two registered arms (k = 8 -> N = 20,261 and k = 12 -> N = 30,391)
ARM_K = {"G08": 8, "G12": 12}
#: SPEC 4.1: the step months add these row counts at L1-L3 (exp-051 i2-plan.json)
STEP_EXTRA = {0: 0, 1: 99_709, 2: 290_037, 3: 386_574}
PANEL_ROWS = 172_783
JUNE_ROWS = 98_396
JUNE_IDS_SHA16 = "61d1ecb2e76a9583"
#: SPEC 4.4 (e): exp-046's recorded July fitting rows of a plain member at L1-L3
E46_BAG = {1: "onemonth-bag10.json", 2: "threemonth-bag10.json", 3: "seasonal-bag10.json"}
LGB_VERSION = "4.7.0"
DYLIB = ROOT / ".venv/lib/python3.11/site-packages/lightgbm/lib/lib_lightgbm.dylib"
DYLIB_SHA = "bc392db609d97730a9ed2acec7a56529b356c03dfad39bebf689523fed7dff18"
#: SPEC 12: the caps, and the measurement reserve that makes the fit gate
CAP_WALL_S = 3960
CAP_CPU_S = 9000
RESERVE_WALL_S = 260
RESERVE_CPU_S = 300
BUDGET_WALL_S = CAP_WALL_S - RESERVE_WALL_S
BUDGET_CPU_S = CAP_CPU_S - RESERVE_CPU_S
MAX_FITS = 36
#: SPEC 4.3: exp-058's least-squares cost law, refitted on its 24 fit records (costlaw-r1.json)
LAW_WALL = (47.5449478, 1.31849731e-4)
LAW_CPU = (89.5397667, 3.60042481e-4)


def fits_of(arms) -> list:
    """SPEC 4.1: arm-major, then member seed, then level -- the registered execution order."""
    return [(a, s, L) for a in arms for s in (1, 2, 3) for L in range(4)]


FITS = fits_of(("G08", "G12"))

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
    if event != "open":
        return
    if args and isinstance(args[0], (str, bytes, os.PathLike)):
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


CENSUS_FILES = json.loads(CENSUS.read_text())["files"]
#: SPEC 2.4: a file the plan stage opens that the census lacks is added by a dated, committed
#: amendment `analysis/prehash-a<k>.json` before any fit; an amendment never changes a hash
CENSUS_AMENDMENTS = sorted(str(p.relative_to(ROOT)) for p in ANALYSIS.glob("prehash-a*.json"))
for _am in CENSUS_AMENDMENTS:
    for _rel, _v in json.loads((ROOT / _am).read_text())["files"].items():
        if _rel in CENSUS_FILES and CENSUS_FILES[_rel]["sha256"] != _v["sha256"]:
            raise SystemExit(f"{_am} changes the census hash of {_rel}")
        CENSUS_FILES.setdefault(_rel, _v)
# SPEC 4.1: exp-051's construction is sha-checked before it is imported
for rel in ("experiments/exp-051-transferadv/code/carrier.py",
            "experiments/exp-051-transferadv/code/common.py",
            "experiments/exp-051-transferadv/code/memberspread.py"):
    if sha_file(ROOT / rel) != CENSUS_FILES[rel]["sha256"]:
        raise SystemExit(f"{rel} differs from the census; nothing is imported")
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


_CHECKED: dict = {}


def census_check(rels) -> dict:
    """Every recorded repository file outside this experiment must be in the census, same bytes."""
    missing, differ, absent, ok = [], [], [], 0
    for rel in sorted(rels):
        if rel.startswith(OWN):
            continue
        p = ROOT / rel
        if not p.exists():
            absent.append(rel)
            continue
        if not p.is_file():
            missing.append(rel + " (not a regular file)")
            continue
        sha = _CHECKED.get(rel) or sha_file(p)
        _CHECKED[rel] = sha
        if rel not in CENSUS_FILES:
            missing.append(rel)
        elif CENSUS_FILES[rel]["sha256"] != sha:
            differ.append({"path": rel, "census": CENSUS_FILES[rel]["sha256"], "now": sha})
        else:
            ok += 1
    return {"checked": ok, "missing": missing, "differ": differ, "absent_at_open": absent}


def code_and_sources() -> tuple[dict, dict]:
    mods = repo_modules()
    for rel, sha in mods.items():
        if not rel.startswith(OWN):
            if rel not in CENSUS_FILES or CENSUS_FILES[rel]["sha256"] != sha:
                raise SystemExit(f"imported module {rel} is not in the census with these bytes")
    src = {}
    for rel in sorted(OPENED):
        if rel.startswith(OWN) or rel.endswith(".py") or not (ROOT / rel).is_file():
            continue
        src[rel] = _CHECKED.get(rel) or sha_file(ROOT / rel)
    return mods, src


# ---------------------------------------------------------------- SPEC 3: the June ladder --

def id_str(v: float) -> str:
    """SPEC 3: the decimal integer form of a float64 MVT_ID_mvt, exactly integral."""
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
    """SPEC 3: ceil(k/24 * n) in exact integer arithmetic, asserted equal to the float form."""
    e = -((-k * n) // 24)
    if e != math.ceil(k / 24 * n):
        raise SystemExit(f"ceil disagreement at k={k}, n={n}")
    return e


def month_writable(ctx: dict, month: str) -> pd.DataFrame:
    """The month's writable rows, exactly as exp-051 `common.ext_frame` puts them in a fit."""
    fr = C.ext_frame(ctx["scols"], [month], ctx["cache"])
    if fr is None or not len(fr):
        raise SystemExit(f"{month}: no writable rows")
    return fr


def counts_by_airport(fr: pd.DataFrame) -> dict:
    air = fr.airport.astype(str).to_numpy()
    return {a: int((air == a).sum()) for a in sorted(set(air))}


def june_order(fr: pd.DataFrame) -> dict:
    """SPEC 3: exp-058's frozen June order inside each shared airport.  Reads only the id, the
    month label (a constant) and the airport -- no target, residual, base, feature or fitted
    value."""
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


def grid_row(order: dict, ids: np.ndarray, k: int) -> tuple[np.ndarray, dict]:
    """SPEC 3: the k/24 prefix block.  Returns the selected positions (ascending, the frame's own
    row order) and its registered record."""
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
                  "first_key": order[a]["keys"][0], "last_selected_key": order[a]["keys"][take_n - 1],
                  "first_unselected_key": order[a]["keys"][take_n] if take_n < order[a]["positions"].size else None}
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


def build_grid(ctx: dict) -> tuple[dict, dict]:
    """SPEC 3: the six counts, all 24 grid rows, the registered N sequence and exp-058's block."""
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
    if rows[24]["block_rows"] != E58_N or rows[24]["count_vector"] != [NA[a] for a in SHARED_AIRPORTS]:
        raise SystemExit("k = 24 is not exp-058's 60,777-row block by count")
    if (rows[24]["count_vector_sha256"] != E58_COUNT_VECTOR_SHA
            or rows[24]["ids_sha256_block_order"] != E58_IDS_SHA_BLOCK
            or rows[24]["ids_sha256_sorted"] != E58_IDS_SHA_SORTED):
        raise SystemExit("k = 24 does not reproduce exp-058's matched June block hashes")
    for k in range(1, 24):                       # SPEC 3.4: nested prefixes, counts non-decreasing
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
           "k24_equals_exp058_matched_june_block": True,
           "prefix_nesting_checked": True,
           "rows": {str(k): rows[k] for k in range(1, 25)}}
    return out, blocks


def block_frame(ctx: dict, arm: str, blocks: dict, arm_k: dict) -> pd.DataFrame:
    fr = month_writable(ctx, MONTH)
    return fr.iloc[blocks[arm_k[arm]]].reset_index(drop=True)


def fitting_extra(ctx: dict, arm: str, level: int, blocks: dict, arm_k: dict) -> tuple[pd.DataFrame, dict]:
    """exp-051 `carrier.fitting_extra`'s assembly with this experiment's block first."""
    blk = block_frame(ctx, arm, blocks, arm_k)
    rows = {"arm": arm, "k": arm_k[arm], "f": f"{arm_k[arm]}/24", "block_month": MONTH,
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


# ---------------------------------------------------------------- the fitting frame hash --

class _PlanStop(BaseException):
    def __init__(self, h: str):
        super().__init__(h)
        self.h = h


def frame_hash(args: tuple, kwargs: dict) -> str:
    """SPEC 4.4 (i), exp-057's definition: what `arms.arm_lgb(fit, sco, cols, ...)` reads."""
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
    """The value as it reads back from the JSON record."""
    return json.loads(json.dumps(x, sort_keys=True, default=float))


ARM_LGB, SETTING0, HOLD0 = A.arm_lgb, CF.SETTING, CF.HOLD


def call_fit_arm(ctx: dict, s: int, extra, arm_lgb, prov: dict) -> dict:
    """exp-051 `carrier.fit_one`'s call, with `arms.arm_lgb` replaced for the call."""
    CF.SETTING, CF.HOLD = {"num_leaves": C.LEAVES}, ()
    A.arm_lgb = arm_lgb
    try:
        return CF.fit_arm(ctx["pans"][s], prov=prov, extra=extra, folds=(F,))
    finally:
        A.arm_lgb, CF.SETTING, CF.HOLD = ARM_LGB, SETTING0, HOLD0


def restored() -> None:
    if A.arm_lgb is not ARM_LGB or CF.SETTING is not SETTING0 or CF.HOLD is not HOLD0:
        raise SystemExit("SETTING, HOLD or arm_lgb was not restored")


def plan_one(ctx: dict, arm: str, s: int, L: int, blocks: dict, arm_k: dict) -> dict:
    t0 = time.time()
    extra, rows = fitting_extra(ctx, arm, L, blocks, arm_k)

    def stub(*a, **k):
        raise _PlanStop(frame_hash(a, k))
    prov: dict = {}
    try:
        call_fit_arm(ctx, s, extra, stub, prov)
        raise SystemExit(f"{arm}-s{s}-L{L}: the stub was not reached")
    except _PlanStop as e:
        h = e.h
    restored()
    return {"tag": f"{arm}-s{s}-L{L}", "arm": arm, "k": arm_k[arm], "month": MONTH, "seed": s,
            "level": L, "extra_rows": int(len(extra)), "rows": jround(rows),
            "provenance": jround(prov[F]), "frame_sha256": h, "seconds": round(time.time() - t0, 1)}


def context(s: int, shared: dict) -> dict:
    """`carrier.panels([s])`, with one extension-month cache for the whole process."""
    t1 = time.time()
    ctx = carrier.panels([s])                                  # SPEC 4.4 (a): asserts the order
    if "scols" in shared and shared["scols"] != ctx["scols"]:
        raise SystemExit(f"seed {s}: the shipped columns differ from the first seed's")
    shared.setdefault("scols", ctx["scols"])
    ctx["cache"] = shared.setdefault("cache", {})
    print(f"seed {s}: panel built in {time.time() - t1:.0f}s", flush=True)
    return ctx


def checks_for(e: dict, N: int) -> dict:
    """SPEC 4.4 (d)-(e) for one planned fit; every failure is fatal."""
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
    """SPEC 4.4 (f): exp-051's FULL June carrier frames of seed s, against its i2-plan.json."""
    by_tag = {x["tag"]: x for x in i2plan["fits"]}
    out = []
    other = int(sum(ctx["pans"][s][g].usable.sum() for g in A.FOLDS if g != F))
    for L in range(4):
        extra, rows = carrier.fitting_extra(ctx, s, "C", L, None)
        # exp-051 tagged seed 1's level-0 carrier frame `integrity-C0-s1`, not `C-s1-L0`
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
        raise SystemExit(f"SPEC 4.4 (h) fails: {out}")
    return out


def write_json(p: Path, obj: dict) -> None:
    NO.refuse_existing([p], overwrite=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


# ---------------------------------------------------------------- the cost ledger (SPEC 12) --

def cost_rows() -> list:
    """analysis/cost.json is append-only: one write-once JSON row per compute job, one per line."""
    if not COST.exists():
        return []
    return [json.loads(ln) for ln in COST.read_text().splitlines() if ln.strip()]


def cost_used() -> tuple[float, float]:
    rows = cost_rows()
    return (sum(r["wall_s"] for r in rows), sum(r["process_cpu_s"] for r in rows))


def cost_append(job: str, wall: float, cpu: float, note: str = "") -> None:
    rows = cost_rows()
    if any(r["job"] == job for r in rows):
        raise SystemExit(f"cost.json already holds a row for job {job!r}; rows are write-once")
    COST.parent.mkdir(parents=True, exist_ok=True)
    with open(COST, "a") as fh:
        fh.write(json.dumps({"job": job, "wall_s": round(wall, 1), "process_cpu_s": round(cpu, 1),
                             "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "note": note}, sort_keys=True) + "\n")


# ---------------------------------------------------------------- the plan --

def do_plan(plan_path: Path, arms: tuple, grid_ref: dict | None, supersede: str | None = None) -> None:
    NO.refuse_existing([plan_path], overwrite=False)
    t0, c0 = time.time(), time.process_time()
    arm_k = {a: (ARM_K[a] if a in ARM_K else int(a[1:])) for a in arms}
    i2plan = json.loads((E51 / "analysis/i2-plan.json").read_text())
    h = check_h()
    fits_spec = fits_of(arms)
    fits, f_checks, shared = [], [], {}
    grid, blocks = None, None
    for s in (1, 2, 3):
        ctx = context(s, shared)
        if grid is None:
            grid, blocks = build_grid(ctx)
            if grid_ref is not None and grid != grid_ref:
                raise SystemExit("the grid differs from analysis/plan.json's")
            print("grid: N = " + ", ".join(f"k{k} {grid['rows'][str(k)]['block_rows']:,}"
                                           for k in sorted(arm_k.values())), flush=True)
        for arm, s2, L in fits_spec:
            if s2 != s:
                continue
            e = plan_one(ctx, arm, s, L, blocks, arm_k)
            e["checks"] = checks_for(e, grid["rows"][str(arm_k[arm])]["block_rows"])
            fits.append(e)
            print(f"  {e['tag']:11} extra {e['extra_rows']:>7,} total {e['provenance']['total_fit_rows']:>7,} "
                  f"frame {e['frame_sha256'][:16]} ({e['seconds']}s)", flush=True)
        f_checks += check_f(ctx, s, i2plan)
        print(f"  seed {s}: exp-051's full June carrier frames equal its i2-plan.json", flush=True)
        del ctx
    order = [f"{a}-s{x}-L{L}" for a, x, L in fits_spec]
    fits.sort(key=lambda e: order.index(e["tag"]))
    cc = census_check(OPENED)
    mods, src = code_and_sources()
    if cc["missing"] or cc["differ"]:
        print(json.dumps(cc, indent=1))
    rec = {"what": "exp-059 plan stage (SPEC 3, 4.4): the k/24 June grid, the fitting sets, the "
                   "provenance and the fitting-frame hashes of " + ", ".join(arms) + "; no fit",
           "spec_commit": SPEC_COMMIT, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "census_amendments": CENSUS_AMENDMENTS,
           "arms": {a: {"k": arm_k[a], "f": f"{arm_k[a]}/24",
                        "N": grid["rows"][str(arm_k[a])]["block_rows"]} for a in arms},
           "grid": grid,
           "fit_order": order, "fits": fits,
           "check_f_exp051_full_june_frames": f_checks, "check_h": h,
           "census_check": cc, "opened_by_how": dict(sorted(OPENED.items())),
           "code": {"sha256": digest(mods), "files": mods},
           "sources": {"sha256": digest(src), "files": src},
           "all_checks_pass": not (cc["missing"] or cc["differ"]),
           "seconds": round(time.time() - t0, 1),
           "seconds_process_cpu": round(time.process_time() - c0, 1)}
    if supersede is not None:
        old = json.loads((ROOT / supersede).read_text())
        same_grid = old["grid"] == rec["grid"]
        same_fits = ([{k_: e[k_] for k_ in ("tag", "extra_rows", "rows", "provenance", "frame_sha256")}
                      for e in old["fits"]]
                     == [{k_: e[k_] for k_ in ("tag", "extra_rows", "rows", "provenance", "frame_sha256")}
                         for e in rec["fits"]])
        changed = sorted(k_ for k_ in set(old["code"]["files"]) | set(rec["code"]["files"])
                         if old["code"]["files"].get(k_) != rec["code"]["files"].get(k_))
        rec["supersedes"] = [{"plan": supersede, "code_sha256": old["code"]["sha256"],
                              "source_sha256": old["sources"]["sha256"],
                              "same_grid": same_grid, "same_fit_inputs": same_fits,
                              "code_files_that_changed": changed}]
        if not (same_grid and same_fits and old["sources"]["sha256"] == rec["sources"]["sha256"]):
            raise SystemExit(f"{supersede}: the superseded plan does not carry the same fit inputs")
    out = plan_path if rec["all_checks_pass"] else plan_path.with_name(
        f"{plan_path.stem}-census-fail-a{len(CENSUS_AMENDMENTS)}.json")
    write_json(out, rec)
    cost_append(f"plan-{'-'.join(arms)}-{plan_path.stem}", time.time() - t0, time.process_time() - c0,
                "zero-fit plan stage")
    print(f"plan: {len(fits)} fits over {arms}; census check: {cc['checked']} files match, "
          f"{len(cc['missing'])} missing, {len(cc['differ'])} differ, {len(cc['absent_at_open'])} absent; "
          f"code {rec['code']['sha256'][:12]} ({len(mods)} files), sources {rec['sources']['sha256'][:12]} "
          f"({len(src)} files); {rec['seconds']}s wall / {rec['seconds_process_cpu']}s process cpu", flush=True)
    if not rec["all_checks_pass"]:
        raise SystemExit("the fit-input check fails: no fit launches (SPEC 2.4)")


# ---------------------------------------------------------------- the fits --

class _DryStop(BaseException):
    pass


def done_records() -> list:
    """Every fit record this experiment has written, whatever arm it belongs to."""
    d = ANALYSIS / "fits"
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []


def estimate(level: int, rows: int, done: list) -> tuple[float, float]:
    """SPEC 4.3: the larger of this experiment's completed same-level mean and exp-057's law."""
    law_w = LAW_WALL[0] + LAW_WALL[1] * rows
    law_c = LAW_CPU[0] + LAW_CPU[1] * rows
    same = [r for r in done if r["level"] == level]
    if same:
        law_w = max(law_w, sum(r["seconds_fit_wall"] for r in same) / len(same))
        law_c = max(law_c, sum(r["seconds_fit_process_cpu"] for r in same) / len(same))
    return law_w, law_c


def do_run(check_only: bool, plan_path: Path, arms: tuple) -> None:
    plan = json.loads(plan_path.read_text())
    fits_spec = fits_of(arms)
    order = [f"{a}-s{s}-L{L}" for a, s, L in fits_spec]
    if not plan["all_checks_pass"]:
        raise SystemExit(f"{plan_path.name} does not pass")
    if plan["fit_order"] != order:
        raise SystemExit("the fit order differs from the plan's")
    if plan["spec_commit"] != SPEC_COMMIT:
        raise SystemExit("the plan was written against another registration")
    arm_k = {a: plan["arms"][a]["k"] for a in arms}
    by_tag = {e["tag"]: e for e in plan["fits"]}
    c12, s12 = plan["code"]["sha256"][:12], plan["sources"]["sha256"][:12]
    i2plan = json.loads((E51 / "analysis/i2-plan.json").read_text())
    check_h()
    prior = done_records()
    if len(prior) + len(fits_spec) > MAX_FITS:
        raise SystemExit(f"{len(prior)} fits recorded plus {len(fits_spec)} planned passes {MAX_FITS}")
    used_w0, used_c0 = cost_used()
    print(f"{len(prior)} fits already recorded; the cost ledger carries {used_w0:.0f}s wall / "
          f"{used_c0:.0f}s process cpu into the budget", flush=True)
    t_job, cpu_job = time.time(), time.process_time()
    n_fitted, shared, blocks, grid = 0, {}, None, None
    ctx, cur_seed, checked_f = None, None, set()
    try:
        #: SPEC 4.1: the registered order, fit for fit; a seed's panel is built when that seed's
        #: next fit needs it and freed when the order moves on to another seed.
        for arm, s, L in fits_spec:
            tag = f"{arm}-s{s}-L{L}"
            if (ANALYSIS / "fits" / f"{tag}.json").exists():
                if not check_only:
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
                    print(f"  seed {s}: exp-051's full June carrier frames equal its i2-plan.json",
                          flush=True)
            grow = plan["grid"]["rows"][str(arm_k[arm])]
            N = grow["block_rows"]
            b12 = grow["ids_sha256_block_order"][:12]
            js = ANALYSIS / "fits" / f"{tag}.json"
            npz = WORK / f"{tag}__code-{c12}__src-{s12}__rows-{b12}.npz"
            stray = sorted(WORK.glob(f"{tag}__*.npz")) if WORK.exists() else []
            if stray:
                raise SystemExit(f"{tag}: a vector without a record exists: {stray}")
            NO.refuse_existing([npz, js], overwrite=False)
            done = done_records()
            if len(done) >= MAX_FITS:
                raise SystemExit(f"{MAX_FITS} fits reached")
            used_w = used_w0 + (time.time() - t_job)
            used_c = used_c0 + (time.process_time() - cpu_job)
            est_w, est_c = estimate(L, PANEL_ROWS + N + STEP_EXTRA[L], done)
            if used_w + est_w > BUDGET_WALL_S or used_c + est_c > BUDGET_CPU_S:
                raise SystemExit(
                    f"budget gate (SPEC 4.3): {tag} is projected at {est_w:.0f}s wall / {est_c:.0f}s "
                    f"process cpu on top of {used_w:.0f}s / {used_c:.0f}s used, which passes the fit "
                    f"cap {BUDGET_WALL_S}s / {BUDGET_CPU_S}s (the 1.1 wall-hour and 2.5 process-CPU-hour "
                    f"caps less the measurement reserve); no fit launches")
            load = carrier.hold_for_load(tag)
            extra, rows = fitting_extra(ctx, arm, L, blocks, arm_k)
            want = by_tag[tag]
            if jround(rows) != want["rows"] or int(len(extra)) != want["extra_rows"]:
                raise SystemExit(f"{tag}: the fitting set differs from the plan's")
            box = {"ok": False}
            prov: dict = {}

            def wrapper(*a, **k):
                hh = frame_hash(a, k)
                if hh != want["frame_sha256"]:
                    raise SystemExit(f"{tag}: fitting-frame hash {hh[:16]} is not the plan's "
                                     f"{want['frame_sha256'][:16]}; the fit does not run")
                if jround(prov[F]) != want["provenance"]:
                    raise SystemExit(f"{tag}: fit_arm's provenance differs from the plan's")
                cc = census_check(OPENED)
                if cc["missing"] or cc["differ"]:
                    raise SystemExit(f"{tag}: the fit-input check fails before the fit: {cc}")
                mods, src = code_and_sources()
                if digest(mods) != plan["code"]["sha256"]:
                    raise SystemExit(f"{tag}: the code hash {digest(mods)[:12]} is not the plan's {c12}")
                extra_src = {r: v for r, v in src.items() if r not in plan["sources"]["files"]}
                box.update(ok=True, census=cc, mods=mods, src=src, src_not_in_plan=extra_src)
                if check_only:
                    raise _DryStop()
                return ARM_LGB(*a, **k)

            print(f"launch {tag} at load {load:.2f} ({time.strftime('%H:%M:%S', time.gmtime())} UTC); "
                  f"budget used {used_w:.0f}s wall / {used_c:.0f}s cpu, this fit projected "
                  f"{est_w:.0f}s / {est_c:.0f}s", flush=True)
            t1, c1 = time.time(), time.process_time()
            try:
                d = call_fit_arm(ctx, s, extra, wrapper, prov)
            except _DryStop:
                restored()
                print(f"run-check: {tag} passed every pre-fit check (census {box['census']['checked']} "
                      f"files, code {c12}, {len(box['src_not_in_plan'])} sources not in the plan); "
                      f"LightGBM was not called; nothing written", flush=True)
                return
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
            mods_after = repo_modules()
            if digest(mods_after) != plan["code"]["sha256"]:
                raise SystemExit(f"{tag}: a repository module was imported during the fit")
            cc = census_check(OPENED)
            if cc["missing"] or cc["differ"]:
                raise SystemExit(f"{tag}: a file opened during the fit fails the census: {cc}")
            rec = {"tag": tag, "arm": arm, "k": arm_k[arm], "f": f"{arm_k[arm]}/24", "month": MONTH,
                   "seed": s, "level": L, "fold": F,
                   "learner_seed": A.SEED, "leaves": C.LEAVES, "hold": [], "spec_commit": SPEC_COMMIT,
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
                   "experiment_wall_s_at_end": round(used_w0 + time.time() - t_job, 1),
                   "experiment_process_cpu_s_at_end": round(used_c0 + time.process_time() - cpu_job, 1),
                   "projected_wall_s": round(est_w, 1), "projected_process_cpu_s": round(est_c, 1),
                   "load_1min_at_launch": load, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                   "vector": str(npz.relative_to(ROOT)), "vector_sha256": sha_file(npz),
                   "rows_moved": int((vec != 0).sum())}
            write_json(js, rec)
            print(f"  fitted {npz.name} in {secs:.0f}s wall / {cpu:.0f}s cpu: fit rows "
                  f"{prov[F]['total_fit_rows']:,}, moved {rec['rows_moved']:,}, sha {rec['vector_sha256'][:16]}",
                  flush=True)
    finally:
        if not check_only and n_fitted:
            cost_append(f"fits-{'-'.join(arms)}", time.time() - t_job, time.process_time() - cpu_job,
                        f"{n_fitted} fits")
    if check_only:
        raise SystemExit("run-check found no fit to check")
    print(f"all {len(fits_spec)} fits of {arms} present; {n_fitted} fitted by this job in "
          f"{time.time() - t_job:.0f}s wall, {time.process_time() - cpu_job:.0f}s process cpu "
          f"(experiment total {used_w0 + time.time() - t_job:.0f}s wall / "
          f"{used_c0 + time.process_time() - cpu_job:.0f}s cpu)", flush=True)


def verify_done(tag: str, plan: dict) -> None:
    """A recorded fit must carry this plan's code hash, or a code hash the plan supersedes with the
    same grid, the same fitting-frame hashes and the same provenance (see `supersedes`)."""
    rec = json.loads((ANALYSIS / "fits" / f"{tag}.json").read_text())
    ok_code = {plan["code"]["sha256"]} | {x["code_sha256"] for x in plan.get("supersedes", [])}
    p = ROOT / rec["vector"]
    if rec["code_sha256"] not in ok_code or not p.exists() or sha_file(p) != rec["vector_sha256"]:
        raise SystemExit(f"{tag}: the existing record does not match its vector or the plan's code")
    want = {e["tag"]: e for e in plan["fits"]}[tag]
    if rec["frame_sha256"] != want["frame_sha256"] or rec["fit_arm_provenance"] != want["provenance"]:
        raise SystemExit(f"{tag}: the recorded fitting frame is not this plan's")


def third_arm(k: int) -> str:
    if not (1 <= k <= 23) or k in ARM_K.values():
        raise SystemExit(f"k={k} is not an unfitted grid row in 1..23")
    return f"G{k:02d}"


def main() -> None:
    ps = argparse.ArgumentParser()
    g = ps.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--run-check", action="store_true")
    g.add_argument("--plan3", type=int, metavar="K", help="SPEC 5.2 step 5: plan the third arm k/24")
    g.add_argument("--run3", type=int, metavar="K", help="SPEC 5.2 step 5: fit the third arm k/24")
    ps.add_argument("--plan-file", default=None, help="the plan record to write or read (write-once)")
    ps.add_argument("--supersede", default=None,
                    help="an earlier plan record this one replaces; its grid, fitting-frame hashes, "
                         "provenance and source hash must be identical")
    a = ps.parse_args()
    if a.plan or a.run or a.run_check:
        arms, ref = ("G08", "G12"), None
        plan_path = (ROOT / a.plan_file) if a.plan_file else PLAN
    else:
        k = a.plan3 if a.plan3 is not None else a.run3
        arms = (third_arm(k),)
        plan_path = (ROOT / a.plan_file) if a.plan_file else ANALYSIS / "plan3.json"
        ref = json.loads(PLAN.read_text())["grid"]
        lad = json.loads((ANALYSIS / "ladder.json").read_text())
        if lad["branch"] != "step5_third_point" or lad["third_point"]["k"] != k:
            raise SystemExit(f"analysis/ladder.json does not select k={k} as the third point")
    if a.plan or a.plan3 is not None:
        do_plan(plan_path, arms, ref, a.supersede)
    else:
        do_run(a.run_check, plan_path, arms)


if __name__ == "__main__":
    main()
