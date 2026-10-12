"""exp-058 origincontrast: the 24 fits of SPEC section 4, one at a time, after a zero-fit plan.

    .venv/bin/python experiments/exp-058-origincontrast/code/oc_fit.py --plan
    .venv/bin/python experiments/exp-058-origincontrast/code/oc_fit.py --run-check
    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-058-origincontrast/code/oc_fit.py --run

--plan counts the writable rows of 2025-02 and 2025-06 by airport, applies SPEC section 3's fixed
per-airport match and SHA-256 selection, builds every member panel and every fitting frame, and
fits nothing: for each fit `chainfold.fit_arm` is called exactly as the fit will call it, with
`arms.arm_lgb` replaced for the call by a stub that hashes its inputs and stops the call before any
LightGBM object exists.  It runs SPEC 4.4 (a)-(i), checks every repository file the process opened
against the committed census (SPEC 2.4), and writes analysis/plan.json.

--run takes the fits in the SPEC order.  Before LightGBM is called for a fit, a wrapper requires
the fitting-frame hash and `fit_arm`'s provenance to equal the plan's, every newly opened
repository file to match the census, and the code hash to equal the plan's.  Before each launch the
budget gate of SPEC 4.3 projects the fit's wall and process-CPU cost and refuses a launch that
would take either cumulative total past its cap.  --run-check does every pre-fit check for the
first fit and stops before LightGBM; it writes nothing.

The driver prints and records hashes, row counts, times and the count of rows a vector moves.  It
computes no other statistic of any vector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
WORK = ROOT / "artifacts/exp-058/work"
ANALYSIS = EXP / "analysis"
PLAN = ANALYSIS / "plan.json"
CENSUS = ANALYSIS / "prehash.json"
OWN = ("experiments/exp-058-origincontrast/", "artifacts/exp-058/")
SPEC_COMMIT = "22e2a6b91f94040d1999d483ae4bf95565ca1f72"
F = "july_loo"
LEARNER_SEED = 46
#: SPEC 3
SALT = "origincontrast-5801"
SHARED_AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LSZH")
ARM_MONTH = {"F": "2025-02", "M": "2025-06"}
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
#: SPEC 12
BUDGET_WALL_S = 3600
BUDGET_CPU_S = 7200
MAX_FITS = 24
#: SPEC 4.3: exp-057's measured linear-in-fitting-rows law, registered before any fit
LAW_WALL = (43.6, 1.243e-4)
LAW_CPU = (82.4, 3.565e-4)
#: SPEC 4.1: February members 1-3 at L0-L3, then matched-June members 1-3 at L0-L3
FITS = [(arm, s, L) for arm in ("F", "M") for s in (1, 2, 3) for L in range(4)]

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


# ---------------------------------------------------------------- SPEC 3: the row match --

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


def month_writable(ctx: dict, month: str) -> pd.DataFrame:
    """The month's writable rows, exactly as exp-051 `common.ext_frame` puts them in a fit."""
    fr = C.ext_frame(ctx["scols"], [month], ctx["cache"])
    if fr is None or not len(fr):
        raise SystemExit(f"{month}: no writable rows")
    return fr


def select_block(fr: pd.DataFrame, month: str, n_by_airport: dict) -> tuple[np.ndarray, dict]:
    """SPEC 3.  Returns the selected positions (ascending, the frame's own row order) and the
    per-airport record.  Reads only MVT_ID_mvt, the month label and the airport."""
    ids = fr.MVT_ID_mvt.to_numpy("float64")
    air = fr.airport.astype(str).to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise SystemExit(f"{month}: duplicate MVT_ID_mvt among the writable rows")
    keep, per = [], {}
    for a in SHARED_AIRPORTS:
        pos = np.flatnonzero(air == a)
        if pos.size == 0:
            raise SystemExit(f"{month}: airport {a} has no writable row (empty stratum)")
        n_a = n_by_airport[a]
        if n_a <= 0:
            raise SystemExit(f"{month}: n_a = {n_a} at {a} (empty stratum)")
        if n_a > pos.size:
            raise SystemExit(f"{month}: n_a = {n_a} at {a} exceeds its {pos.size} writable rows")
        keys = [hashlib.sha256(f"{SALT}|{month}|{a}|{id_str(ids[i])}".encode("utf-8")).hexdigest()
                for i in pos]
        order = sorted(range(pos.size), key=lambda j: (keys[j], ids[pos[j]]))
        take = pos[np.array(order[:n_a], dtype="int64")]
        keep.append(take)
        per[a] = {"writable": int(pos.size), "n_a": int(n_a), "selected": int(take.size),
                  "rejected_inside_S": int(pos.size - n_a),
                  "first_key": keys[order[0]], "last_selected_key": keys[order[n_a - 1]],
                  "first_unselected_key": keys[order[n_a]] if n_a < pos.size else None}
        if take.size != n_a:
            raise SystemExit(f"{month} {a}: selected {take.size} rows, not {n_a}")
    sel = np.sort(np.concatenate(keep))
    if len(np.unique(sel)) != len(sel):
        raise SystemExit(f"{month}: a row was selected twice")
    sel_ids = ids[sel]
    if len(np.unique(sel_ids)) != len(sel_ids):
        raise SystemExit(f"{month}: duplicate MVT_ID_mvt in the block")
    if set(np.unique(air[sel])) != set(SHARED_AIRPORTS):
        raise SystemExit(f"{month}: the block's airport set is not S")
    rec = {"month": month, "salt": SALT, "shared_airports": list(SHARED_AIRPORTS),
           "block_rows": int(sel.size), "by_airport": per,
           "writable_rows_total_month": int(len(fr)),
           "rejected_outside_S": int(len(fr) - sum(v["writable"] for v in per.values())),
           "rejected_inside_S": int(sum(v["rejected_inside_S"] for v in per.values())),
           "ids_sha256_block_order": ordered_sha(sel_ids),
           "ids_sha256_sorted": ordered_sha(np.sort(sel_ids)),
           "ids_sha16": C.ids_sha16(sel_ids),
           "count_vector_sha256": ordered_sha(np.array([per[a]["n_a"] for a in SHARED_AIRPORTS], "int64")),
           "count_vector": [per[a]["n_a"] for a in SHARED_AIRPORTS]}
    return sel, rec


def build_match(ctx: dict) -> dict:
    """SPEC 3: counts, the per-airport minimum, both selections, and every refusal."""
    out = {"shared_airports": list(SHARED_AIRPORTS), "months": {}, "counts": {}}
    writable = {}
    for arm, month in ARM_MONTH.items():
        fr = month_writable(ctx, month)
        air = fr.airport.astype(str).to_numpy()
        cnt = {a: int((air == a).sum()) for a in sorted(set(air))}
        writable[month] = cnt
        out["months"][month] = {"arm": arm, "writable_rows": int(len(fr)),
                                "airports_present": sorted(cnt), "writable_by_airport": cnt}
        missing = [a for a in SHARED_AIRPORTS if cnt.get(a, 0) == 0]
        if missing:
            raise SystemExit(f"{month}: shared airports missing or empty: {missing}")
    n_by_airport = {a: min(writable["2025-02"][a], writable["2025-06"][a]) for a in SHARED_AIRPORTS}
    out["counts"] = {a: {"2025-02": writable["2025-02"][a], "2025-06": writable["2025-06"][a],
                         "n_a": n_by_airport[a],
                         "limited_by": "2025-02" if writable["2025-02"][a] <= writable["2025-06"][a] else "2025-06"}
                     for a in SHARED_AIRPORTS}
    out["N"] = int(sum(n_by_airport.values()))
    blocks = {}
    for arm, month in ARM_MONTH.items():
        fr = month_writable(ctx, month)
        sel, rec = select_block(fr, month, n_by_airport)
        blocks[arm] = sel
        out["months"][month]["selection"] = rec
        if rec["block_rows"] != out["N"]:
            raise SystemExit(f"{month}: the block has {rec['block_rows']} rows, not N = {out['N']}")
    cvF = out["months"]["2025-02"]["selection"]["count_vector"]
    cvM = out["months"]["2025-06"]["selection"]["count_vector"]
    if cvF != cvM:
        raise SystemExit(f"the arms' per-airport count vectors differ: {cvF} vs {cvM}")
    out["count_vectors_equal"] = True
    out["count_vector"] = cvF
    return out, blocks


def block_frame(ctx: dict, arm: str, blocks: dict) -> pd.DataFrame:
    month = ARM_MONTH[arm]
    fr = month_writable(ctx, month)
    return fr.iloc[blocks[arm]].reset_index(drop=True)


def fitting_extra(ctx: dict, arm: str, level: int, blocks: dict) -> tuple[pd.DataFrame, dict]:
    """exp-051 `carrier.fitting_extra`'s assembly with this experiment's block first."""
    blk = block_frame(ctx, arm, blocks)
    rows = {"arm": arm, "block_month": ARM_MONTH[arm], "block_rows": int(len(blk)),
            "block_ids_sha16": C.ids_sha16(blk.MVT_ID_mvt)}
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
    return {"tag": f"{arm}-s{s}-L{L}", "arm": arm, "month": ARM_MONTH[arm], "seed": s, "level": L,
            "extra_rows": int(len(extra)), "rows": jround(rows), "provenance": jround(prov[F]),
            "frame_sha256": h, "seconds": round(time.time() - t0, 1)}


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
    out = {"d_block": r["block_rows"] == N and r["block_month"] == ARM_MONTH[arm],
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


# ---------------------------------------------------------------- the plan --

def do_plan(plan_path: Path) -> None:
    NO.refuse_existing([plan_path], overwrite=False)
    t0 = time.time()
    i2plan = json.loads((E51 / "analysis/i2-plan.json").read_text())
    h = check_h()
    fits, f_checks, shared = [], [], {}
    match, blocks = None, None
    for s in (1, 2, 3):
        ctx = context(s, shared)
        if match is None:
            match, blocks = build_match(ctx)
            print(f"match: N = {match['N']:,}; per airport "
                  + ", ".join(f"{a} {match['counts'][a]['n_a']:,} ({match['counts'][a]['limited_by']})"
                              for a in SHARED_AIRPORTS), flush=True)
        for arm, s2, L in FITS:
            if s2 != s:
                continue
            e = plan_one(ctx, arm, s, L, blocks)
            e["checks"] = checks_for(e, match["N"])
            fits.append(e)
            print(f"  {e['tag']:9} extra {e['extra_rows']:>7,} total {e['provenance']['total_fit_rows']:>7,} "
                  f"frame {e['frame_sha256'][:16]} ({e['seconds']}s)", flush=True)
        f_checks += check_f(ctx, s, i2plan)
        print(f"  seed {s}: exp-051's full June carrier frames equal its i2-plan.json", flush=True)
        del ctx
    fits.sort(key=lambda e: [f"{a}-s{x}-L{L}" for a, x, L in FITS].index(e["tag"]))
    cc = census_check(OPENED)
    mods, src = code_and_sources()
    if cc["missing"] or cc["differ"]:
        print(json.dumps(cc, indent=1))
    rec = {"what": "exp-058 plan stage (SPEC 3, 4.4): the row match, fitting sets, provenance and "
                   "fitting-frame hashes; no fit",
           "spec_commit": SPEC_COMMIT, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "census_amendments": CENSUS_AMENDMENTS,
           "match": match,
           "fit_order": [f"{a}-s{s}-L{L}" for a, s, L in FITS], "fits": fits,
           "check_f_exp051_full_june_frames": f_checks, "check_h": h,
           "census_check": cc, "opened_by_how": dict(sorted(OPENED.items())),
           "code": {"sha256": digest(mods), "files": mods},
           "sources": {"sha256": digest(src), "files": src},
           "all_checks_pass": not (cc["missing"] or cc["differ"]),
           "seconds": round(time.time() - t0, 1)}
    out = plan_path if rec["all_checks_pass"] else plan_path.with_name(
        f"{plan_path.stem}-census-fail-a{len(CENSUS_AMENDMENTS)}.json")
    write_json(out, rec)
    print(f"plan: {len(fits)} fits, N = {match['N']:,}; census check: {cc['checked']} files match, "
          f"{len(cc['missing'])} missing, {len(cc['differ'])} differ, {len(cc['absent_at_open'])} absent; "
          f"code {rec['code']['sha256'][:12]} ({len(mods)} files), sources {rec['sources']['sha256'][:12]} "
          f"({len(src)} files); {rec['seconds']}s", flush=True)
    if not rec["all_checks_pass"]:
        raise SystemExit("the fit-input check fails: no fit launches (SPEC 2.4)")


# ---------------------------------------------------------------- the fits --

class _DryStop(BaseException):
    pass


def done_records() -> list:
    out = []
    for a, s, L in FITS:
        p = ANALYSIS / "fits" / f"{a}-s{s}-L{L}.json"
        if p.exists():
            out.append(json.loads(p.read_text()))
    return out


def estimate(level: int, rows: int, done: list) -> tuple[float, float]:
    """SPEC 4.3: the larger of this experiment's completed same-level mean and exp-057's law."""
    law_w = LAW_WALL[0] + LAW_WALL[1] * rows
    law_c = LAW_CPU[0] + LAW_CPU[1] * rows
    same = [r for r in done if r["level"] == level]
    if same:
        law_w = max(law_w, sum(r["seconds_fit_wall"] for r in same) / len(same))
        law_c = max(law_c, sum(r["seconds_fit_process_cpu"] for r in same) / len(same))
    return law_w, law_c


def do_run(check_only: bool, plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text())
    if not plan["all_checks_pass"]:
        raise SystemExit("plan.json does not pass")
    if plan["fit_order"] != [f"{a}-s{s}-L{L}" for a, s, L in FITS]:
        raise SystemExit("the fit order differs from the plan's")
    if plan["spec_commit"] != SPEC_COMMIT:
        raise SystemExit("the plan was written against another registration")
    by_tag = {e["tag"]: e for e in plan["fits"]}
    N = plan["match"]["N"]
    c12, s12 = plan["code"]["sha256"][:12], plan["sources"]["sha256"][:12]
    i2plan = json.loads((E51 / "analysis/i2-plan.json").read_text())
    check_h()
    prior = done_records()
    prior_wall = sum(r["seconds_fit_wall"] for r in prior)
    prior_cpu = sum(r["seconds_fit_process_cpu"] for r in prior)
    if prior:
        print(f"{len(prior)} fits already recorded: {prior_wall:.0f}s wall, {prior_cpu:.0f}s process cpu "
              f"carried into the budget", flush=True)
    t_job, cpu_job = time.time(), time.process_time()
    n_fitted, shared, blocks, match = 0, {}, None, None
    for s in (1, 2, 3):
        todo = [(a, s2, L) for a, s2, L in FITS if s2 == s]
        pending = [x for x in todo if not (ANALYSIS / "fits" / f"{x[0]}-s{s}-L{x[2]}.json").exists()]
        if not pending and not check_only:
            for a, _, L in todo:
                verify_done(f"{a}-s{s}-L{L}", plan)
            continue
        ctx = context(s, shared)
        if match is None:
            match, blocks = build_match(ctx)
            if jround(match) != plan["match"]:
                raise SystemExit("the row match differs from the plan's")
            print("the row match reproduces the plan's", flush=True)
        check_f(ctx, s, i2plan)
        for arm, _, L in todo:
            tag = f"{arm}-s{s}-L{L}"
            js = ANALYSIS / "fits" / f"{tag}.json"
            npz = WORK / f"{tag}__code-{c12}__src-{s12}.npz"
            if js.exists():
                verify_done(tag, plan)
                print(f"skip {tag}: record and vector exist", flush=True)
                continue
            stray = sorted(WORK.glob(f"{tag}__*.npz")) if WORK.exists() else []
            if stray:
                raise SystemExit(f"{tag}: a vector without a record exists: {stray}")
            NO.refuse_existing([npz, js], overwrite=False)
            done = done_records()
            if len(done) >= MAX_FITS:
                raise SystemExit(f"{MAX_FITS} fits reached")
            used_w = prior_wall + (time.time() - t_job)
            used_c = prior_cpu + (time.process_time() - cpu_job)
            est_w, est_c = estimate(L, PANEL_ROWS + N + STEP_EXTRA[L], done)
            if used_w + est_w > BUDGET_WALL_S or used_c + est_c > BUDGET_CPU_S:
                raise SystemExit(
                    f"budget gate (SPEC 4.3): {tag} is projected at {est_w:.0f}s wall / {est_c:.0f}s "
                    f"process cpu on top of {used_w:.0f}s / {used_c:.0f}s used, which passes the cap "
                    f"{BUDGET_WALL_S}s / {BUDGET_CPU_S}s; no fit launches")
            load = carrier.hold_for_load(tag)
            extra, rows = fitting_extra(ctx, arm, L, blocks)
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
            rec = {"tag": tag, "arm": arm, "month": ARM_MONTH[arm], "seed": s, "level": L, "fold": F,
                   "learner_seed": A.SEED, "leaves": C.LEAVES, "hold": [], "spec_commit": SPEC_COMMIT,
                   "block_rows": want["rows"]["block_rows"],
                   "block_ids_sha16": want["rows"]["block_ids_sha16"],
                   "block_ids_sha256_block_order": plan["match"]["months"][ARM_MONTH[arm]]["selection"]["ids_sha256_block_order"],
                   "code_sha256": plan["code"]["sha256"], "code_files": box["mods"],
                   "source_sha256": plan["sources"]["sha256"],
                   "sources_opened_by_this_fit_not_in_plan": box["src_not_in_plan"],
                   "rows": want["rows"], "fit_arm_provenance": jround(prov[F]),
                   "frame_sha256": want["frame_sha256"],
                   "seconds_fit_wall": round(secs, 1), "seconds_fit_process_cpu": round(cpu, 1),
                   "job_wall_s_at_end": round(time.time() - t_job, 1),
                   "job_process_cpu_s_at_end": round(time.process_time() - cpu_job, 1),
                   "budget_wall_s_at_end": round(prior_wall + time.time() - t_job, 1),
                   "budget_process_cpu_s_at_end": round(prior_cpu + time.process_time() - cpu_job, 1),
                   "projected_wall_s": round(est_w, 1), "projected_process_cpu_s": round(est_c, 1),
                   "load_1min_at_launch": load, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                   "vector": str(npz.relative_to(ROOT)), "vector_sha256": sha_file(npz),
                   "rows_moved": int((vec != 0).sum())}
            write_json(js, rec)
            print(f"  fitted {npz.name} in {secs:.0f}s wall / {cpu:.0f}s cpu: fit rows "
                  f"{prov[F]['total_fit_rows']:,}, moved {rec['rows_moved']:,}, sha {rec['vector_sha256'][:16]}",
                  flush=True)
        del ctx
    if check_only:
        raise SystemExit("run-check found no fit to check")
    print(f"all {len(FITS)} fits present; {n_fitted} fitted by this job in {time.time() - t_job:.0f}s wall, "
          f"{time.process_time() - cpu_job:.0f}s process cpu (budget total "
          f"{prior_wall + time.time() - t_job:.0f}s wall / {prior_cpu + time.process_time() - cpu_job:.0f}s cpu)",
          flush=True)


def verify_done(tag: str, plan: dict) -> None:
    rec = json.loads((ANALYSIS / "fits" / f"{tag}.json").read_text())
    p = ROOT / rec["vector"]
    if rec["code_sha256"] != plan["code"]["sha256"] or not p.exists() or sha_file(p) != rec["vector_sha256"]:
        raise SystemExit(f"{tag}: the existing record does not match its vector or the plan's code")


def main() -> None:
    ps = argparse.ArgumentParser()
    g = ps.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--run-check", action="store_true")
    ps.add_argument("--plan-file", default=str(PLAN.relative_to(ROOT)))
    a = ps.parse_args()
    plan_path = ROOT / a.plan_file
    if plan_path.parent != ANALYSIS or not plan_path.name.startswith("plan"):
        raise SystemExit("the plan record lives in analysis/ and is named plan*.json")
    if a.plan:
        do_plan(plan_path)
    else:
        do_run(a.run_check, plan_path)


if __name__ == "__main__":
    main()
