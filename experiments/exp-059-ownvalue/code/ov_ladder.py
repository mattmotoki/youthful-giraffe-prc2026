"""exp-059 ownvalue: SPEC section 5 -- the `V` read and the deterministic stopping rule.

ZERO FIT.  READS LABELS.  It computes **only** own values at L0 and their day-block intervals: no
`m`, no step, no loss identity, no subgroup and no decomposition (SPEC 5.3, 5.4).

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-059-ownvalue/code/ov_ladder.py --stage1
    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-059-ownvalue/code/ov_ladder.py --stage2

--stage1 reproduces exp-055's pins (R1), exp-055's draws (R5) and exp-058's five L0 own values
(R8 iv) before it opens a new vector, then reads `V` for the two registered arms, orders the four
points N = 0, 20,261, 30,391, 60,777 and applies SPEC 5.2 steps 2-5.  It writes analysis/ladder.json
with the points, the band tests, the branch, and -- when step 5 is reached -- the interpolated `N`
and the selected third grid row.

--stage2 runs only when step 5 was reached and the third arm's 12 fits exist.  It repeats every
reproduction, reads the third point, applies SPEC 5.2 step 6 and writes analysis/ladder2.json with
`June*` or with the non-monotone / unmatched stop and its residual mismatch.
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
E55 = ROOT / "experiments/exp-055-originmove"
E57 = ROOT / "experiments/exp-057-originfit"
E58 = ROOT / "experiments/exp-058-origincontrast"
ANALYSIS = EXP / "analysis"
COST = ANALYSIS / "cost.json"
OWN_PREFIX = ("experiments/exp-059-ownvalue/", "artifacts/exp-059/")

ap_ = argparse.ArgumentParser()
g_ = ap_.add_mutually_exclusive_group(required=True)
g_.add_argument("--stage1", action="store_true")
g_.add_argument("--stage2", action="store_true")
ARGS = ap_.parse_args()

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

#: SPEC 2.3: this process imports exp-055's closure only, and never exp-051's
sys.path.insert(0, str(E55 / "code"))
import common as C                                                    # noqa: E402  (exp-055)
import lossfals as LF                                                 # noqa: E402
import originvec as V                                                 # noqa: E402

for _name, _want in (("common", E55 / "code/common.py"), ("originvec", E55 / "code/originvec.py"),
                     ("lossfals", E55 / "code/lossfals.py")):
    if Path(sys.modules[_name].__file__).resolve() != _want:
        raise SystemExit(f"module {_name} resolves to {sys.modules[_name].__file__} (name collision)")
if "carrier" in sys.modules or "memberspread" in sys.modules:
    raise SystemExit("exp-051's closure is imported in the ladder process")

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

NO = C.NO
F = "july_loo"
B = 2000
BLOCKS = (1, 2, 3)
SEEDS3 = (1, 2, 3)
SEEDS10 = tuple(range(1, 11))
TOL_9 = 1e-9
SPEC_COMMIT = "49db4157d8bc43edde6c2b8386e90b6f58929578"
E58_SPEC_COMMIT = "22e2a6b91f94040d1999d483ae4bf95565ca1f72"
E58_CODE_SHA = "9a970bacf95c05dbf923d7a0e7260e93a69fa90c69553ceebe6e09928338dd11"
#: SPEC 5.1: the frozen constants
V_STAR = 168.58928050561724
BAND = (151.7303524550555, 185.44820855617897)
V_ANCHOR_TOP = 493.2541127299781
N_TOP = 60_777
MONO_TOL = 1e-9
#: SPEC 3: the registered total-N sequence, k = 1..23
GRID_N = (2_536, 5_068, 7_601, 10_131, 12_666, 15_198, 17_728, 20_261, 22_794, 25_329, 27_859,
          30_391, 32_924, 35_454, 37_989, 40_521, 43_055, 45_585, 48_117, 50_652, 53_182, 55_715,
          58_247)
FITTED_K = (8, 12)
#: exp-058's five recorded L0 own values (R8 iv)
E58_OWN = {"feb_K3": V_STAR, "june_matched_K3": V_ANCHOR_TOP, "carrier_K3": 582.2425054210122,
           "sham_K3": -4009.1762051589712, "carrier_K10": 571.4884463397937}

CENSUS = json.loads((EXP / "analysis/prehash.json").read_text())["files"]
for _am in sorted((EXP / "analysis").glob("prehash-a*.json")):
    for _r, _v in json.loads(_am.read_text())["files"].items():
        if _r in CENSUS and CENSUS[_r]["sha256"] != _v["sha256"]:
            raise SystemExit(f"{_am.name} changes the census hash of {_r}")
        CENSUS.setdefault(_r, _v)
def fit_inputs(plan: dict) -> list:
    """Every registered fit field except the frame-build wall time, which is not an input."""
    return [{k: v for k, v in f.items() if k != "seconds"} for f in plan["fits"]]


def load_plan(name: str) -> dict:
    """The newest registered plan of `name`: plan-rN.json when one supersedes it, with the
    superseded plan's grid, fit inputs and sources re-checked here (SPEC 4.1 note)."""
    base = json.loads((ANALYSIS / f"{name}.json").read_text())
    cur, seen = base, {name}
    for r1 in sorted(ANALYSIS.glob(f"{name}-r*.json")):
        cand = json.loads(r1.read_text())
        sup = cand.get("supersedes") or []
        if not any(str(ROOT / s["plan"]) == str(ANALYSIS / f"{name}.json") for s in sup):
            continue
        for s in sup:
            if not (s.get("same_grid") and s.get("same_fit_inputs")):
                raise SystemExit(f"{r1.name}: supersedes {s['plan']} without the same fit inputs")
            if s["source_sha256"] != cand["sources"]["sha256"]:
                raise SystemExit(f"{r1.name}: the superseded plan's sources differ")
        if cand["grid"] != base["grid"] or fit_inputs(cand) != fit_inputs(base):
            raise SystemExit(f"{r1.name}: the grid or the fit list is not {name}.json's")
        cur, seen = cand, seen | {r1.stem}
    cur["_plans_read"] = sorted(seen)
    return cur


def ok_codes(plan: dict) -> set:
    """The plan's code hash and every code hash it supersedes with the same fit inputs."""
    return {plan["code"]["sha256"]} | {x["code_sha256"] for x in plan.get("supersedes", [])}


PLAN = load_plan("plan")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check59(p: Path) -> str:
    """The file's bytes must carry this experiment's census hash for its path (SPEC 2.1)."""
    rel = str(Path(p).resolve().relative_to(ROOT))
    if rel not in CENSUS:
        raise SystemExit(f"{rel} is not in exp-059's census")
    got = sha_file(p)
    if got != CENSUS[rel]["sha256"]:
        raise SystemExit(f"{rel} is {got[:16]}, not the census {CENSUS[rel]['sha256'][:16]}")
    return got


def census_check() -> dict:
    opened = sorted(r for r in OPENED if not r.startswith(OWN_PREFIX) and (ROOT / r).is_file())
    miss = [r for r in opened if r not in CENSUS]
    diff = [r for r in opened if r in CENSUS and sha_file(ROOT / r) != CENSUS[r]["sha256"]]
    return {"opened": len(opened), "missing": miss, "differ": diff, "files": opened}


def cost_rows() -> list:
    return [json.loads(ln) for ln in COST.read_text().splitlines() if ln.strip()] if COST.exists() else []


def cost_append(job: str, wall: float, cpu: float, note: str = "") -> None:
    if any(r["job"] == job for r in cost_rows()):
        raise SystemExit(f"cost.json already holds a row for job {job!r}; rows are write-once")
    with open(COST, "a") as fh:
        fh.write(json.dumps({"job": job, "wall_s": round(wall, 1), "process_cpu_s": round(cpu, 1),
                             "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "note": note}, sort_keys=True) + "\n")


# ------------------------------------------------------------------------------ vectors ---------

def load_L0_exp058(vec: dict) -> dict:
    """exp-058's February and matched-June members at L0, sha-checked against its fit records."""
    info = {}
    for arm in ("F", "M"):
        for s in SEEDS3:
            tag = f"{arm}-s{s}-L0"
            rp = E58 / "analysis/fits" / f"{tag}.json"
            check59(rp)
            rec = json.loads(rp.read_text())
            if rec.get("spec_commit") != E58_SPEC_COMMIT or rec.get("code_sha256") != E58_CODE_SHA:
                raise SystemExit(f"exp-058 {tag}: not exp-058's registration or plan code")
            if rec["block_rows"] != N_TOP or rec["arm"] != arm or rec["level"] != 0 or rec["seed"] != s:
                raise SystemExit(f"exp-058 {tag}: the record does not identify this arm, seed and level")
            p = ROOT / rec["vector"]
            if check59(p) != rec["vector_sha256"]:
                raise SystemExit(f"exp-058 {tag}: the vector is not the recorded bytes")
            z = np.load(p)
            if list(z.keys()) != [F]:
                raise SystemExit(f"exp-058 {tag}: the vector holds {list(z.keys())}")
            vec[f"{arm}{s}_L0"] = z[F].astype("float64")
            info[tag] = {"vector": rec["vector"], "sha256": rec["vector_sha256"],
                         "block_ids_sha256_block_order": rec["block_ids_sha256_block_order"]}
    return info


def load_L0_exp057(vec: dict) -> dict:
    """exp-057's sham members 2-3 and carrier members 4-10 at L0 (for the five reproductions)."""
    info = {}
    for kind, seeds in (("S", (2, 3)), ("C", range(4, 11))):
        for s in seeds:
            tag = f"{kind}-s{s}-L0"
            rp = E57 / "analysis/fits" / f"{tag}.json"
            check59(rp)
            rec = json.loads(rp.read_text())
            p = ROOT / rec["vector"]
            if check59(p) != rec["vector_sha256"]:
                raise SystemExit(f"exp-057 {tag}: the vector is not the recorded bytes")
            vec[f"{kind}{s}_L0"] = np.load(p)[F].astype("float64")
            info[tag] = {"vector": rec["vector"], "sha256": rec["vector_sha256"]}
    return info


def arm_records(k: int) -> tuple[str, dict]:
    """This experiment's 12 fit records of the k/24 arm, checked against its plan."""
    arm = f"G{k:02d}"
    plan = PLAN if k in FITTED_K else load_plan("plan3")
    grow = PLAN["grid"]["rows"][str(k)]
    out = {}
    for s in SEEDS3:
        for L in range(4):
            tag = f"{arm}-s{s}-L{L}"
            rec = json.loads((ANALYSIS / "fits" / f"{tag}.json").read_text())
            if rec["spec_commit"] != SPEC_COMMIT or rec["code_sha256"] not in ok_codes(plan):
                raise SystemExit(f"{tag}: the record's registration or code is not the plan's")
            if (rec["k"] != k or rec["arm"] != arm or rec["seed"] != s or rec["level"] != L
                    or rec["block_rows"] != grow["block_rows"]
                    or rec["block_ids_sha256_block_order"] != grow["ids_sha256_block_order"]
                    or rec["block_count_vector_sha256"] != grow["count_vector_sha256"]):
                raise SystemExit(f"{tag}: the record does not identify this grid row")
            out[tag] = rec
    return arm, out


def load_L0_arm(vec: dict, k: int) -> dict:
    arm, recs = arm_records(k)
    info = {}
    for s in SEEDS3:
        rec = recs[f"{arm}-s{s}-L0"]
        p = ROOT / rec["vector"]
        if sha_file(p) != rec["vector_sha256"]:
            raise SystemExit(f"{arm}-s{s}-L0: the vector is not the recorded bytes")
        z = np.load(p)
        if list(z.keys()) != [F]:
            raise SystemExit(f"{arm}-s{s}-L0: the vector holds {list(z.keys())}")
        vec[f"{arm}{s}_L0"] = z[F].astype("float64")
        info[f"{arm}-s{s}-L0"] = {"vector": rec["vector"], "sha256": rec["vector_sha256"],
                                  "rows_moved": rec["rows_moved"]}
    return info


# ------------------------------------------------------------------------------ own value -------

class Own:
    """SPEC 5.1: V = [SSE(P3_L0) - SSE(X_L0)] / n on every July fold row, on exp-058's route.

    exp-058 sums (y - pbase - vector)^2 into (UTC day, airport) cells, adds the cells into days
    with exp-055 lossfals's day matrix, adds the days, and subtracts the two totals.  Each total is
    near 7.6e9 s2 while their difference is near 3e7 s2, so a route that subtracts row by row
    first is more accurate but does NOT reproduce the registered V* to the last digits.  This class
    walks exp-058's route column for column, so the five reproductions of SPEC 5.3 meet 1e-9.

    The day matrix is built per group of columns.  A column's day sum and total do not depend on
    the other columns in the frame, so R8's matrix (exp-058's vectors only, built before a new
    vector is opened) and each arm's matrix give the same numbers as one matrix over all columns.
    """

    def __init__(self, pan: dict, vec: dict, y: np.ndarray, days: np.ndarray):
        pf = pan[F]
        self.pf, self.vec, self.y, self.days = pf, vec, y, days
        self.n = len(y)
        self.cd, uq = pd.factorize(pd.MultiIndex.from_arrays([pf["day"], pf["ap"]]), sort=True)
        self.G = len(uq)
        self.cell_day = [a for a, _ in uq]
        self.cell_ap = [b for _, b in uq]
        self.nd = len(days)
        dn, _ = LF.day_matrix(pd.DataFrame({"day": self.cell_day, "airport": self.cell_ap,
                                            "n": np.bincount(self.cd, minlength=self.G).astype("int64")}),
                              days)
        self.day_n = dn[:, 0].astype("float64")
        if int(self.day_n.sum()) != self.n:
            raise SystemExit(f"the day counts sum to {self.day_n.sum()}, not {self.n}")
        self.sse = {}

    def bag(self, nm: str, seeds=SEEDS3) -> np.ndarray:
        pf = self.pf
        return C.bag({s: self.vec[f"{nm}{s}_L0"] for s in seeds}, seeds, pf["pbase"], pf["usable"],
                     pf["guard"])

    def cells(self, nm: str, v: np.ndarray) -> np.ndarray:
        """(day, airport) sums of (y - pbase - v)^2, by exp-058's `cell` route."""
        if nm in self.sse:
            raise SystemExit(f"the sse cells of {nm} are already registered")
        e = self.y - self.pf["pbase"] - v
        self.sse[nm] = np.bincount(self.cd, weights=e * e, minlength=self.G)
        return self.sse[nm]

    def totals(self, names) -> tuple:
        """exp-055 lossfals's day matrix of the named sse columns, and its column sums."""
        cells = pd.DataFrame({"day": self.cell_day, "airport": self.cell_ap,
                              **{f"sse_{nm}_L0": self.sse[nm] for nm in names}})
        Dm, ci = LF.day_matrix(cells, self.days)
        return Dm, ci, Dm.sum(0)

    def value(self, xn: str, yn: str, Dm: np.ndarray, ci: dict, P: np.ndarray) -> tuple:
        """[SSE(plain yn) - SSE(arm xn)] / n and its per-day contributions."""
        jx, jy = ci[f"sse_{xn}_L0"], ci[f"sse_{yn}_L0"]
        return float((P[jy] - P[jx]) / self.n), Dm[:, jy] - Dm[:, jx]


def band_test(v: float) -> bool:
    return bool(BAND[0] <= v <= BAND[1])


def nearest_unfitted(nhat: float, used: tuple) -> int:
    """SPEC 5.2 step 5: the unfitted k/24 grid row nearest N-hat; an integer-distance tie takes
    the larger N."""
    cand = [(abs(GRID_N[k - 1] - nhat), -GRID_N[k - 1], k) for k in range(1, 24) if k not in used]
    cand.sort()
    return cand[0][2]


# ------------------------------------------------------------------------------ main ------------

def main() -> None:
    t0, c0 = time.time(), time.process_time()
    stage = 1 if ARGS.stage1 else 2
    out = ANALYSIS / ("ladder.json" if stage == 1 else "ladder2.json")
    NO.refuse_existing([out], overwrite=False)
    if (ANALYSIS / "measure.json").exists():
        raise SystemExit("analysis/measure.json exists: the ladder read comes first (SPEC 5.4)")
    if stage == 2:
        lad1 = json.loads((ANALYSIS / "ladder.json").read_text())
        if lad1["branch"] != "step5_third_point":
            raise SystemExit("stage 2 runs only after SPEC 5.2 step 5")
        k3 = lad1["third_point"]["k"]
    rec = {"what": __doc__.strip().splitlines()[0], "stage": stage, "spec_commit": SPEC_COMMIT,
           "plans_read": PLAN["_plans_read"], "plan_code_sha256": PLAN["code"]["sha256"],
           "definition": "V = [SSE(P3_L0) - SSE(X3_L0)] / n on every July fold row (n = 190,713), "
                         "SSE over y - pbase - vector; positive means the arm's own L0 vector "
                         "predicts July better than the plain K = 3 bag; measured, never subtracted",
           "constants": {"V_star_february": V_STAR, "band": list(BAND), "band_rule": "V* +/- 10 %",
                         "monotone_tolerance_s2_row": MONO_TOL,
                         "anchor_V_at_0": 0.0, "anchor_V_at_60777": V_ANCHOR_TOP},
           "checks": {}}

    # ---- R1 ---------------------------------------------------------------------------------
    pan, vec, pin = V.load()
    lab = LF.load_labels(F, pan[F]["ids"])
    y = lab["y"]
    v = vec[F]
    rec["checks"]["R1_originvec_load"] = True
    print("R1 exp-055 pins verified; the labels equal exp-051's OOF y_true", flush=True)

    # ---- R5 (the draw generator, in exp-055's order) -----------------------------------------
    scr = json.loads((E55 / "analysis/screen.json").read_text())
    check59(E55 / "analysis/screen.json")
    days = np.sort(np.unique(pan[F]["day"]))
    if len(days) != 31:
        raise SystemExit("July does not have 31 fold days")
    W = {}
    for L in BLOCKS:
        rng = np.random.default_rng(5500 + L)
        st = {key: C.mbb_starts(rng, 31, L, B) for key in ("july_loo", "2026-07", "january", "2026-01",
                                                           "december")}
        sig = rng.integers(1, 4, size=(B, 3))
        want = scr["draw_sha256"][f"L{L}"]
        got = {k_: C.sha_arr(x_) for k_, x_ in st.items()} | {"seeds": C.sha_arr(sig)}
        if got != {k_: x_ for k_, x_ in want.items() if k_ != "multiset_counts"}:
            raise SystemExit(f"R5 L{L}: the regenerated draws are not exp-055's")
        W[L] = C.mbb_weights(st["july_loo"], 31, L)
    rec["checks"]["R5_draws_equal_exp055"] = True
    print("R5 the regenerated day draws equal exp-055's", flush=True)

    # ---- the plain members 4-10 and exp-057's L0 vectors, then exp-058's ----------------------
    for s in range(4, 11):
        p = C.member_path(0, s)
        check59(p)                      # this experiment's census, as exp-058 does; exp-055
        v[f"P{s}_L0"] = np.load(p)[F].astype("float64")   # never registered members 4-10
    rec["exp057_vectors_L0"] = load_L0_exp057(v)
    rec["exp058_vectors_L0"] = load_L0_exp058(v)

    OW = Own(pan, v, y, days)
    if OW.n != 190_713:
        raise SystemExit(f"July has {OW.n} fold rows, not 190,713")
    p3 = OW.bag("P")
    p10 = OW.bag("P", SEEDS10)
    if not np.array_equal(p3, v["P3bag_L0"]):
        raise SystemExit("R6: the rebuilt plain K = 3 bag is not exp-055's pinned bag")
    if not np.allclose(p10, v["T10_L0"], atol=0, rtol=0):
        d10 = float(np.max(np.abs(p10 - v["T10_L0"])))
        if d10 > TOL_9:
            raise SystemExit(f"R6: the rebuilt plain K = 10 bag differs from Q15's head by {d10:.3e} s")

    # ---- R8 (iv): exp-058's five L0 own values, before any new vector is opened ---------------
    E58_PAIRS = {"feb_K3": ("F3bag", "P3bag"), "june_matched_K3": ("M3bag", "P3bag"),
                 "carrier_K3": ("C3bag", "P3bag"), "sham_K3": ("S3bag", "P3bag"),
                 "carrier_K10": ("C10bag", "P10bag")}
    OW.cells("P3bag", p3)
    OW.cells("P10bag", p10)
    for X in ("F", "M", "C", "S"):
        OW.cells(f"{X}3bag", OW.bag(X))
    OW.cells("C10bag", OW.bag("C", SEEDS10))
    Dm8, ci8, P8 = OW.totals(sorted({n for pr in E58_PAIRS.values() for n in pr}))
    repro, worst = {}, 0.0
    for nm, (X, Y) in E58_PAIRS.items():
        val, dsum = OW.value(X, Y, Dm8, ci8, P8)
        repro[nm] = {"X": X, "Y": Y, "value": val, "exp058": E58_OWN[nm],
                     "abs_diff": abs(val - E58_OWN[nm]),
                     "intervals": {f"L{L}": LF.interval((W[L] @ dsum) / (W[L] @ OW.day_n))
                                   for L in BLOCKS}}
        worst = max(worst, repro[nm]["abs_diff"])
    rec["R8_exp058_own_values"] = {"by_arm": repro, "max_abs_diff_s2_row": worst}
    if worst > TOL_9:
        raise SystemExit(f"exp-058's five L0 own values are missed by {worst:.3e} s2/row")
    print(f"R8 exp-058's five L0 own values reproduced (max |diff| {worst:.2e} s2/row): "
          f"February {repro['feb_K3']['value']:+.4f}, matched June {repro['june_matched_K3']['value']:+.4f}",
          flush=True)
    if abs(repro["feb_K3"]["value"] - V_STAR) > TOL_9 or abs(repro["june_matched_K3"]["value"] - V_ANCHOR_TOP) > TOL_9:
        raise SystemExit("the frozen anchors do not reproduce")

    # ---- the fit-free anchors (SPEC 5.1) ------------------------------------------------------
    zero, _ = OW.value("P3bag", "P3bag", Dm8, ci8, P8)
    rec["checks"]["anchor_V_at_0_identity"] = {
        "value": zero,
        "why": "an empty block leaves the plain member's own L0 fit, so X3 = P3 and V = 0 by "
               "construction; the identity SSE(P3) - SSE(P3) = 0 is computed here, no fit exists"}
    if zero != 0.0:
        raise SystemExit("the V(0) identity is not exactly zero")

    # ---- the new arms -------------------------------------------------------------------------
    ks = list(FITTED_K) + ([k3] if stage == 2 else [])
    arms, new_info = {}, {}
    for k in ks:
        arm = f"G{k:02d}"
        new_info |= load_L0_arm(v, k)
        OW.cells(f"{arm}3bag", OW.bag(arm))
        for s in SEEDS3:
            OW.cells(f"{arm}{s}", v[f"{arm}{s}_L0"])
            if f"P{s}" not in OW.sse:
                OW.cells(f"P{s}", v[f"P{s}_L0"])
        Dk, cik, Pk = OW.totals(sorted([f"{arm}3bag", "P3bag"]
                                       + [f"{arm}{s}" for s in SEEDS3] + [f"P{s}" for s in SEEDS3]))
        val, dsum = OW.value(f"{arm}3bag", "P3bag", Dk, cik, Pk)
        members = {}
        for s in SEEDS3:
            mv, mds = OW.value(f"{arm}{s}", f"P{s}", Dk, cik, Pk)
            members[f"s{s}"] = {"value": mv,
                                "intervals": {f"L{L}": LF.interval((W[L] @ mds) / (W[L] @ OW.day_n))
                                              for L in BLOCKS}}
        arms[arm] = {"k": k, "f": f"{k}/24", "N": PLAN["grid"]["rows"][str(k)]["block_rows"],
                     "V": val, "in_band": band_test(val), "abs_V_minus_V_star": abs(val - V_STAR),
                     "members": members,
                     "intervals": {f"L{L}": LF.interval((W[L] @ dsum) / (W[L] @ OW.day_n))
                                   for L in BLOCKS}}
        mtxt = " ".join(f"{members['s%d' % s]['value']:+.1f}" for s in SEEDS3)
        print(f"{arm} (k={k}, N={arms[arm]['N']:,}): V = {val:+.4f} s2/row, in band {band_test(val)}, "
              f"members {mtxt}", flush=True)
    rec["new_vectors_L0"] = new_info
    rec["arms"] = arms

    # ---- the ladder points and SPEC 5.2 -------------------------------------------------------
    pts = [{"N": 0, "V": 0.0, "source": "identity (empty block)"}]
    for k in sorted(FITTED_K):
        a = arms[f"G{k:02d}"]
        pts.append({"N": a["N"], "V": a["V"], "source": f"fitted k={k}", "k": k})
    pts.append({"N": N_TOP, "V": V_ANCHOR_TOP, "source": "exp-058 matched June arm (k=24)"})
    if stage == 2:
        a = arms[f"G{k3:02d}"]
        pts.append({"N": a["N"], "V": a["V"], "source": f"fitted k={k3}", "k": k3})
        pts.sort(key=lambda d: d["N"])
    for p_ in pts:
        p_["in_band"] = band_test(p_["V"])
    rec["points"] = pts
    diffs = [pts[i + 1]["V"] - pts[i]["V"] for i in range(len(pts) - 1)]
    rec["adjacent_differences"] = diffs
    mono = all(d >= -MONO_TOL for d in diffs)
    rec["monotone"] = mono

    if stage == 1:
        if not mono:
            rec["branch"] = "step2_non_monotone"
            rec["june_star"] = None
            rec["reading"] = ("SPEC 5.2 step 2: a decrease larger than 1e-9 s2/row between adjacent V "
                              "values; the job ends at 24 fits as non-monotone and no curve reading "
                              "is applied (SPEC 5.5)")
        else:
            inb = [k for k in FITTED_K if arms[f"G{k:02d}"]["in_band"]]
            if len(inb) == 1:
                k_sel = inb[0]
                rec["branch"] = "step3_one_in_band"
            elif len(inb) == 2:
                a, b = (arms[f"G{k:02d}"] for k in FITTED_K)
                if a["abs_V_minus_V_star"] < b["abs_V_minus_V_star"]:
                    k_sel = FITTED_K[0]
                elif b["abs_V_minus_V_star"] < a["abs_V_minus_V_star"]:
                    k_sel = FITTED_K[1]
                else:
                    k_sel = FITTED_K[1]          # an exact stored-value tie selects the larger N
                rec["branch"] = "step4_both_in_band"
            else:
                k_sel = None
            if k_sel is not None:
                rec["june_star"] = {"k": k_sel, "arm": f"G{k_sel:02d}", "N": arms[f"G{k_sel:02d}"]["N"],
                                    "V": arms[f"G{k_sel:02d}"]["V"],
                                    "residual_abs_V_minus_V_star": arms[f"G{k_sel:02d}"]["abs_V_minus_V_star"],
                                    "fits_total": 24}
                rec["reading"] = f"SPEC 5.2 {rec['branch']}: June* is the k={k_sel} arm; the job ends at 24 fits"
            else:
                pair = None
                for i in range(len(pts) - 1):
                    if pts[i]["V"] <= V_STAR <= pts[i + 1]["V"]:
                        pair = i
                        break
                if pair is None:
                    raise SystemExit("no adjacent pair brackets V*, although the points are monotone")
                lo_, hi_ = pts[pair], pts[pair + 1]
                nhat = lo_["N"] + (V_STAR - lo_["V"]) / (hi_["V"] - lo_["V"]) * (hi_["N"] - lo_["N"])
                k_sel = nearest_unfitted(nhat, FITTED_K)
                rec["branch"] = "step5_third_point"
                rec["june_star"] = None
                rec["third_point"] = {"bracketing_pair": [lo_, hi_], "N_hat": float(nhat),
                                      "k": k_sel, "arm": f"G{k_sel:02d}", "N": GRID_N[k_sel - 1],
                                      "distance_to_N_hat": abs(GRID_N[k_sel - 1] - nhat),
                                      "fits_to_run": 12, "fits_total": 36}
                rec["reading"] = (f"SPEC 5.2 step 5: neither fitted point is in the band; the chord between "
                                  f"N = {lo_['N']:,} and N = {hi_['N']:,} reaches V* at N = {nhat:,.0f}, and "
                                  f"the nearest unfitted grid row is k = {k_sel} (N = {GRID_N[k_sel - 1]:,})")
    else:
        third = arms[f"G{k3:02d}"]
        i = [j for j, p_ in enumerate(pts) if p_.get("k") == k3][0]
        lo_, hi_ = pts[i - 1]["V"], pts[i + 1]["V"]
        outside = (third["V"] < lo_ - MONO_TOL) or (third["V"] > hi_ + MONO_TOL)
        rec["step6"] = {"inserted_at": i, "neighbours": [lo_, hi_], "outside_neighbours": bool(outside),
                        "in_band": third["in_band"],
                        "residual_abs_V_minus_V_star": third["abs_V_minus_V_star"]}
        if outside:
            rec["branch"] = "step6_non_monotone"
            rec["june_star"] = None
            rec["reading"] = ("SPEC 5.2 step 6: the third point's V lies outside its two measured "
                              "neighbours by more than 1e-9 s2/row; the ladder stops as non-monotone "
                              "and no curve reading is applied (SPEC 5.5)")
        elif not third["in_band"]:
            rec["branch"] = "step6_unmatched"
            rec["june_star"] = None
            rec["reading"] = (f"SPEC 5.2 step 6: the third point's V = {third['V']:+.4f} s2/row is outside "
                              f"the band {BAND}; the ladder stops as unmatched, the residual mismatch is "
                              f"{third['abs_V_minus_V_star']:.4f} s2/row, and no curve reading is applied "
                              f"(SPEC 5.5)")
        else:
            rec["branch"] = "step6_matched"
            rec["june_star"] = {"k": k3, "arm": f"G{k3:02d}", "N": third["N"], "V": third["V"],
                                "residual_abs_V_minus_V_star": third["abs_V_minus_V_star"],
                                "fits_total": 36}
            rec["reading"] = f"SPEC 5.2 step 6: the third point is inside the band; June* is the k={k3} arm"

    cc = census_check()
    rec["census_check"] = {k_: x_ for k_, x_ in cc.items() if k_ != "files"} | {"files_opened": cc["files"]}
    if cc["missing"] or cc["differ"]:
        raise SystemExit(f"census check fails: missing {cc['missing']}, differ {cc['differ']}")
    rec["seconds"] = round(time.time() - t0, 1)
    rec["seconds_process_cpu"] = round(time.process_time() - c0, 1)
    C.write_json(out, rec)
    cost_append(f"ladder-stage{stage}", time.time() - t0, time.process_time() - c0,
                "the V read and the stopping rule; zero fit")
    print(f"branch: {rec['branch']}", flush=True)
    print(f"reading: {rec['reading']}", flush=True)
    print(f"June*: {rec['june_star']}", flush=True)
    print(f"written {out.relative_to(ROOT)} ({rec['seconds']} s wall, {rec['seconds_process_cpu']} s cpu)",
          flush=True)


if __name__ == "__main__":
    main()
