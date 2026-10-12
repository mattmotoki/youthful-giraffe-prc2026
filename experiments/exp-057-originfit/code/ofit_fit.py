"""exp-057 originfit: the 36 fits of SPEC section 3, one at a time, after a zero-fit plan stage.

    .venv/bin/python experiments/exp-057-originfit/code/ofit_fit.py --plan
    .venv/bin/python experiments/exp-057-originfit/code/ofit_fit.py --run-check
    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-057-originfit/code/ofit_fit.py --run

--plan builds every member panel and every fitting frame and fits nothing: for each fit,
`chainfold.fit_arm` is called exactly as the fit will call it, with `arms.arm_lgb` replaced for
the call by a stub that hashes its inputs and stops the call before any LightGBM object exists.
It checks SPEC 3.3 (a)-(i), checks every repository file the process opened against the
committed census (SPEC 2.3), and writes analysis/plan.json.

--run takes the fits in the SPEC order.  Before LightGBM is called for a fit, a wrapper around
`arms.arm_lgb` requires the fitting-frame hash and `fit_arm`'s provenance to equal the plan's,
every newly opened repository file to match the census, and the code hash to equal the plan's.
--run-check does all of that for the first fit and stops before LightGBM; it writes nothing.

The driver prints and records hashes, row counts, times and the count of rows a vector moves.
It computes no other statistic of any vector.
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
WORK = ROOT / "artifacts/exp-057/work"
ANALYSIS = EXP / "analysis"
PLAN = ANALYSIS / "plan.json"
CENSUS = ANALYSIS / "prehash.json"
OWN = ("experiments/exp-057-originfit/", "artifacts/exp-057/")
SPEC_COMMIT = "95d82b6611e2768e2a25fd596bd113f168eb307e"
F = "july_loo"
LEARNER_SEED = 46
SHAM_SEED = 5103
JUNE_IDS_SHA16 = "61d1ecb2e76a9583"
SHAM_RBASE_SHA16 = "8ccb1011f834c0e1"
SHAM_CORR = -0.0006138172009713599
EXTRA = {0: 98_396, 1: 198_105, 2: 388_433, 3: 484_970}
PANEL_ROWS = 172_783
E46_BAG = {1: "onemonth-bag10.json", 2: "threemonth-bag10.json", 3: "seasonal-bag10.json"}
LGB_VERSION = "4.7.0"
DYLIB = ROOT / ".venv/lib/python3.11/site-packages/lightgbm/lib/lib_lightgbm.dylib"
DYLIB_SHA = "bc392db609d97730a9ed2acec7a56529b356c03dfad39bebf689523fed7dff18"
BUDGET_WALL_S = 2 * 3600
MAX_FITS = 36
#: SPEC 3.1: the sham members 2 and 3, then the carrier members 4-10, each at L0-L3
FITS = ([("S", s, L) for s in (2, 3) for L in range(4)]
        + [("C", s, L) for s in range(4, 11) for L in range(4)])
#: the exp-051 records of the same kind and level (SPEC 3.3 d)
E51_SAME = {"S": "fits2/i2-S-s1-L{L}.json", "C": "fits2/i2-C-s2-L{L}.json"}

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
#: SPEC 2.3: a file the plan stage opens that the census lacks is added by a dated, committed
#: amendment `analysis/prehash-a<k>.json` before any fit; an amendment never changes a hash
CENSUS_AMENDMENTS = sorted(str(p.relative_to(ROOT)) for p in ANALYSIS.glob("prehash-a*.json"))
for _am in CENSUS_AMENDMENTS:
    for _rel, _v in json.loads((ROOT / _am).read_text())["files"].items():
        if _rel in CENSUS_FILES and CENSUS_FILES[_rel]["sha256"] != _v["sha256"]:
            raise SystemExit(f"{_am} changes the census hash of {_rel}")
        CENSUS_FILES.setdefault(_rel, _v)
# SPEC 3.1: exp-051's construction is sha-checked before it is imported
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


# ---------------------------------------------------------------- the fitting frame hash --

class _PlanStop(BaseException):
    def __init__(self, h: str):
        super().__init__(h)
        self.h = h


def frame_hash(args: tuple, kwargs: dict) -> str:
    """SPEC 3.3 (i): what `arms.arm_lgb(fit, sco, cols, seed=, params=, w=)` reads."""
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
    """exp-051 `carrier.fit_one`'s call, with `arms.arm_lgb` replaced for the call.  `fit_arm`
    fills `prov` before it calls `arm_lgb`."""
    CF.SETTING, CF.HOLD = {"num_leaves": C.LEAVES}, ()
    A.arm_lgb = arm_lgb
    try:
        return CF.fit_arm(ctx["pans"][s], prov=prov, extra=extra, folds=(F,))
    finally:
        A.arm_lgb, CF.SETTING, CF.HOLD = ARM_LGB, SETTING0, HOLD0


def restored() -> None:
    if A.arm_lgb is not ARM_LGB or CF.SETTING is not SETTING0 or CF.HOLD is not HOLD0:
        raise SystemExit("SETTING, HOLD or arm_lgb was not restored")


def plan_one(ctx: dict, s: int, kind: str, L: int) -> dict:
    t0 = time.time()
    extra, rows = carrier.fitting_extra(ctx, s, kind, L, None)

    def stub(*a, **k):
        raise _PlanStop(frame_hash(a, k))
    prov: dict = {}
    try:
        call_fit_arm(ctx, s, extra, stub, prov)
        raise SystemExit(f"{kind}-s{s}-L{L}: the stub was not reached")
    except _PlanStop as e:
        h = e.h
    restored()
    return {"tag": f"{kind}-s{s}-L{L}", "arm": kind, "seed": s, "level": L,
            "extra_rows": int(len(extra)), "rows": jround(rows), "provenance": jround(prov[F]),
            "frame_sha256": h, "seconds": round(time.time() - t0, 1)}


def context(s: int, shared: dict) -> dict:
    """`carrier.panels([s])`, with one extension-month cache for the whole process (exp-051's
    own run shared one cache across its seeds; the month frames do not depend on the seed)."""
    t1 = time.time()
    ctx = carrier.panels([s])                                  # SPEC 3.3 (a): asserts the order
    if "scols" in shared and shared["scols"] != ctx["scols"]:
        raise SystemExit(f"seed {s}: the shipped columns differ from the first seed's")
    shared.setdefault("scols", ctx["scols"])
    ctx["cache"] = shared.setdefault("cache", {})
    print(f"seed {s}: panel built in {time.time() - t1:.0f}s", flush=True)
    return ctx


def checks_for(e: dict, i2plan: dict) -> dict:
    """SPEC 3.3 (b)-(e) for one planned fit; every failure is fatal."""
    kind, s, L = e["arm"], e["seed"], e["level"]
    r, p = e["rows"], e["provenance"]
    out = {}
    out["b_june"] = r.get("june_rows") == 98_396 and r.get("june_ids_sha16") == JUNE_IDS_SHA16
    if kind == "S":
        out["c_sham"] = (r.get("sham_seed") == SHAM_SEED and r.get("sham_rbase_sha16") == SHAM_RBASE_SHA16
                         and r.get("sham_corr_with_true_rbase") == SHAM_CORR
                         and r.get("sham_permuted_within") == "airport")
    else:
        out["c_sham"] = "sham_seed" not in r
    ref = json.loads((E51 / "analysis" / E51_SAME[kind].format(L=L)).read_text())
    out["d_rows_equal_exp051"] = r == ref["rows"]
    out["d_provenance_equal_exp051"] = p == ref["fit_arm_provenance"]
    out["d_counts"] = (e["extra_rows"] == EXTRA[L] and p["extra_rows"] == EXTRA[L]
                       and p["panel_rows"] == PANEL_ROWS
                       and p["total_fit_rows"] == PANEL_ROWS + EXTRA[L]
                       and r.get("ext_months", []) == carrier.STEPS_JULY[L])
    if kind == "C" and L >= 1:
        rec = json.loads((E46 / "analysis" / E46_BAG[L]).read_text())["members"][str(s)]
        out["e_exp046_member_rows"] = (
            p["panel_rows"] + (EXTRA[L] - 98_396) == rec["total_fit_rows"][F]
            and EXTRA[L] - 98_396 == rec["extra_rows"][F])
    bad = [k for k, v in out.items() if v is not True]
    if bad:
        raise SystemExit(f"{e['tag']}: plan checks fail: {bad}")
    return out


def check_f(ctx: dict, s: int, i2plan: dict) -> list:
    """SPEC 3.3 (f): the carrier-kind frames of seeds 2 and 3 against exp-051's plan."""
    by_tag = {x["tag"]: x for x in i2plan["fits"]}
    out = []
    other = int(sum(ctx["pans"][s][g].usable.sum() for g in A.FOLDS if g != F))
    for L in range(4):
        extra, rows = carrier.fitting_extra(ctx, s, "C", L, None)
        want = by_tag[f"C-s{s}-L{L}"]
        got = {"rows": jround(rows), "extra_rows": int(len(extra)), "member_other_fold_rows": other,
               "total_fit_rows": other + int(len(extra))}
        ok = all(got[k] == want[k] for k in got)
        out.append({"tag": f"C-s{s}-L{L}", "equal_to_exp051_plan": ok, **got})
        if not ok:
            raise SystemExit(f"C-s{s}-L{L}: the plan code does not reproduce exp-051 i2-plan.json")
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
        raise SystemExit(f"SPEC 3.3 (h) fails: {out}")
    return out


def write_json(p: Path, obj: dict) -> None:
    NO.refuse_existing([p], overwrite=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


def seeds_in_order() -> list:
    out = []
    for _, s, _ in FITS:
        if s not in out:
            out.append(s)
    return out


# ---------------------------------------------------------------- the plan --

def do_plan(plan_path: Path) -> None:
    NO.refuse_existing([plan_path], overwrite=False)
    t0 = time.time()
    i2plan = json.loads((E51 / "analysis/i2-plan.json").read_text())
    h = check_h()
    fits, f_checks, shared = [], [], {}
    for s in seeds_in_order():
        ctx = context(s, shared)
        for kind, s2, L in FITS:
            if s2 != s:
                continue
            e = plan_one(ctx, s, kind, L)
            e["checks"] = checks_for(e, i2plan)
            fits.append(e)
            print(f"  {e['tag']:9} extra {e['extra_rows']:>7,} total {e['provenance']['total_fit_rows']:>7,} "
                  f"frame {e['frame_sha256'][:16]} ({e['seconds']}s)", flush=True)
        if s in (2, 3):
            f_checks += check_f(ctx, s, i2plan)
            print(f"  seed {s}: carrier-kind frames equal exp-051 i2-plan.json", flush=True)
        del ctx
    cc = census_check(OPENED)
    mods, src = code_and_sources()
    if cc["missing"] or cc["differ"]:
        print(json.dumps(cc, indent=1))
    rec = {"what": "exp-057 plan stage (SPEC 3.3): fitting sets, provenance and fitting-frame hashes; no fit",
           "spec_commit": SPEC_COMMIT, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "census_amendments": CENSUS_AMENDMENTS,
           "fit_order": [f"{k}-s{s}-L{L}" for k, s, L in FITS], "fits": fits,
           "check_f_carrier_frames_seeds_2_3": f_checks, "check_h": h,
           "census_check": cc, "opened_by_how": dict(sorted(OPENED.items())),
           "code": {"sha256": digest(mods), "files": mods},
           "sources": {"sha256": digest(src), "files": src},
           "all_checks_pass": not (cc["missing"] or cc["differ"]),
           "seconds": round(time.time() - t0, 1)}
    # a failed fit-input check is kept under its own name, so that the plan after an
    # amendment can still take plan.json
    out = plan_path if rec["all_checks_pass"] else plan_path.with_name(
        f"{plan_path.stem}-census-fail-a{len(CENSUS_AMENDMENTS)}.json")
    write_json(out, rec)
    print(f"plan: {len(fits)} fits, census check: {cc['checked']} files match, "
          f"{len(cc['missing'])} missing, {len(cc['differ'])} differ, {len(cc['absent_at_open'])} absent; "
          f"code {rec['code']['sha256'][:12]} ({len(mods)} files), sources {rec['sources']['sha256'][:12]} "
          f"({len(src)} files); {rec['seconds']}s", flush=True)
    if not rec["all_checks_pass"]:
        raise SystemExit("the fit-input check fails: no fit launches (SPEC 2.3)")


# ---------------------------------------------------------------- the fits --

class _DryStop(BaseException):
    pass


def do_run(check_only: bool, plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text())
    if not plan["all_checks_pass"]:
        raise SystemExit("plan.json does not pass")
    if plan["fit_order"] != [f"{k}-s{s}-L{L}" for k, s, L in FITS]:
        raise SystemExit("the fit order differs from the plan's")
    by_tag = {e["tag"]: e for e in plan["fits"]}
    c12, s12 = plan["code"]["sha256"][:12], plan["sources"]["sha256"][:12]
    i2plan = json.loads((E51 / "analysis/i2-plan.json").read_text())
    check_h()
    t_job, cpu_job = time.time(), time.process_time()
    n_fitted, shared = 0, {}
    for s in seeds_in_order():
        todo = [(k, s2, L) for k, s2, L in FITS if s2 == s]
        pending = [x for x in todo if not (ANALYSIS / "fits" / f"{x[0]}-s{s}-L{x[2]}.json").exists()]
        if not pending and not check_only:
            for k, _, L in todo:
                verify_done(f"{k}-s{s}-L{L}", plan)
            continue
        ctx = context(s, shared)
        if s in (2, 3):
            check_f(ctx, s, i2plan)
        for kind, _, L in todo:
            tag = f"{kind}-s{s}-L{L}"
            js = ANALYSIS / "fits" / f"{tag}.json"
            npz = WORK / f"{tag}__code-{c12}__src-{s12}.npz"
            if js.exists():
                verify_done(tag, plan)
                print(f"skip {tag}: record and vector exist", flush=True)
                continue
            stray = sorted(WORK.glob(f"{tag}__*.npz"))
            if stray:
                raise SystemExit(f"{tag}: a vector without a record exists: {stray}")
            NO.refuse_existing([npz, js], overwrite=False)
            if n_fitted >= MAX_FITS:
                raise SystemExit("36 fits reached")
            if time.time() - t_job > BUDGET_WALL_S:
                raise SystemExit(f"the fit job has used {time.time() - t_job:.0f}s of wall time; "
                                 f"no fit launches after 2 hours (SPEC 10)")
            load = carrier.hold_for_load(tag)
            extra, rows = carrier.fitting_extra(ctx, s, kind, L, None)
            want = by_tag[tag]
            if jround(rows) != want["rows"] or int(len(extra)) != want["extra_rows"]:
                raise SystemExit(f"{tag}: the fitting set differs from the plan's")
            box = {"ok": False}
            prov: dict = {}

            def wrapper(*a, **k):
                h = frame_hash(a, k)
                if h != want["frame_sha256"]:
                    raise SystemExit(f"{tag}: fitting-frame hash {h[:16]} is not the plan's "
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

            print(f"launch {tag} at load {load:.2f} ({time.strftime('%H:%M:%S', time.gmtime())} UTC)",
                  flush=True)
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
            rec = {"tag": tag, "arm": kind, "seed": s, "level": L, "fold": F,
                   "learner_seed": A.SEED, "sham_seed": SHAM_SEED if kind == "S" else None,
                   "leaves": C.LEAVES, "hold": [], "spec_commit": SPEC_COMMIT,
                   "code_sha256": plan["code"]["sha256"], "code_files": box["mods"],
                   "source_sha256": plan["sources"]["sha256"],
                   "sources_opened_by_this_fit_not_in_plan": box["src_not_in_plan"],
                   "rows": want["rows"], "fit_arm_provenance": jround(prov[F]),
                   "frame_sha256": want["frame_sha256"],
                   "seconds_fit_wall": round(secs, 1), "seconds_fit_process_cpu": round(cpu, 1),
                   "job_wall_s_at_end": round(time.time() - t_job, 1),
                   "job_process_cpu_s_at_end": round(time.process_time() - cpu_job, 1),
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
          f"{time.process_time() - cpu_job:.0f}s process cpu", flush=True)


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
    ps.add_argument("--plan-file", default=str(PLAN.relative_to(ROOT)),
                    help="the plan record (a revision after a code fix is plan-r<k>.json)")
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
