"""exp-062 ownvalue-refine: the new near-origin own-value read and the deterministic selection.
ZERO FIT.  READS LABELS.  It computes own values at L0 and their day-block variation bands only:
no m, no loss identity, no subgroup, no step diagnostic (SPEC 8.4).

    vp_ctl.py --kind value --log logs/value.log -- .venv/bin/python .../vp_value.py

Order (SPEC 8):
  1  the nine L0 fit records of k = 1, 2 and 3 must all exist and carry the registered grid row,
     the plan's code hash and this experiment's registration, or the read refuses.  The assignment
     fits all nine members before a new own value is read and does not stop early on an apparent
     match, so a partial set is a refusal, not a smaller read.
  2  analysis/repro.json must exist and must record a clean R1-R8 and every separate reproduction
     (SPEC 6 and 7), because the new interpretation comes after them.
  3  the guard reproductions run again here, in this process, on this process's own matrix: exp-055's
     ordered pins and labels (R1), the day draws (R5), the plain K = 3 and K = 10 bags (R6), and
     exp-058's five L0 own values with February's V*, the k = 24 anchor and the V(0) identity
     (R8 iv).  Each is a refusal.
  4  the three new K = 3 bag own values and their nine members, on exp-059's exact
     cell-to-day-to-total route, with the one-, two- and three-day day-block variation bands.
  5  exp-059's k = 4 value, reproduced here from exp-059's own L0 vectors and required to equal its
     recorded +211.60409594675028 to 1e-12, so the measured order over k = 1, 2, 3, 4 is one
     matrix in one process.  This reproduces a recorded value; no prediction is built from it.
  6  the fixed band test on the exact stored values, the deterministic selection, and the freeze.

The band B = [151.7303524550555, 185.44820855617897] is frozen by the assignment.  It is not
widened, not replaced by a variation band, and the variation bands below play no part in the test.
"""
from __future__ import annotations

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
E59 = ROOT / "experiments/exp-059-ownvalue"
ANALYSIS = EXP / "analysis"
MANIFEST = ANALYSIS / "manifest.json"
PLAN = ANALYSIS / "plan.json"
REPRO = ANALYSIS / "repro.json"
OUT = ANALYSIS / "probe.json"
OWN_PREFIX = ("experiments/exp-062-ownvalue-refine/", "artifacts/exp-062/")
WORK_REL = "artifacts/exp-062/work"

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


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


MAN = json.loads(MANIFEST.read_text())
SOURCE = MAN["source_boundary"]["files"]
E59_CENSUS = {k: v["sha256"] for k, v in
              json.loads((E59 / "analysis/prehash.json").read_text())["files"].items()}


def _e59_vectors() -> dict:
    """SPEC 2.5, amended: the third pin tier.  exp-059's own 36 fitted vectors are in neither the
    positive manifest nor exp-059's census -- that census was taken before exp-059 fitted anything,
    and the manifest pinned exp-059's 36 fit records without the vector paths they name.  Their pin
    is therefore the hash each of those 36 frozen records carries, and the records' own bytes are
    verified against the manifest pin before a single hash is taken from them.  36 enumerated
    records, no live glob; this experiment writes none of these files."""
    out = {}
    for rel, v in MAN["exp059_sources"]["exp059_fit_records"]["files"].items():
        if SOURCE.get(rel) != v["sha256"] or sha_file(ROOT / rel) != v["sha256"]:
            raise SystemExit(f"{rel} differs from its manifest pin; no exp-059 vector is pinned")
        r = json.loads((ROOT / rel).read_text())
        out[r["vector"]] = r["vector_sha256"]
    return out


E59_VECTORS = _e59_vectors()
#: SPEC 2.4, amended: the effective pin of each code file is its manifest pin, amended by the
#: committed append-only `analysis/code-pins.jsonl`; vp_ctl.code_pins() is the one reader
sys.path.insert(0, str(HERE))
import vp_ctl as _CTL                                                 # noqa: E402
CODE_PINS = _CTL.code_pins()
for _rel, _sha in CODE_PINS["effective"].items():
    if sha_file(ROOT / _rel) != _sha:
        raise SystemExit(f"{_rel} differs from its effective code pin; nothing runs")

sys.path.insert(0, str(E55 / "code"))
import common as C                                                    # noqa: E402
import lossfals as LF                                                 # noqa: E402
import originvec as V                                                 # noqa: E402

for _name, _want in (("common", E55 / "code/common.py"), ("originvec", E55 / "code/originvec.py"),
                     ("lossfals", E55 / "code/lossfals.py")):
    if Path(sys.modules[_name].__file__).resolve() != _want:
        raise SystemExit(f"module {_name} resolves to {sys.modules[_name].__file__} (name collision)")
if "carrier" in sys.modules or "memberspread" in sys.modules:
    raise SystemExit("exp-051's closure is imported in the value process")

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

NO = C.NO
F = "july_loo"
B = 2000
BLOCKS = (1, 2, 3)
SEEDS3 = (1, 2, 3)
SEEDS10 = tuple(range(1, 11))
TOL_9, TOL_12 = 1e-9, 1e-12
#: SPEC 5: frozen by the assignment
V_STAR = 168.58928050561724
BAND = (151.7303524550555, 185.44820855617897)
V_TOP = 493.2541127299781
N_TOP = 60_777
PROBE_K = (1, 2, 3)
GRID_N = {1: 2_536, 2: 5_068, 3: 7_601}
#: exp-059's recorded k = 4 value, reproduced here only to order k = 1, 2, 3, 4
E59_K4 = 4
E59_V4 = 211.60409594675028
E58_OWN = {"feb_K3": V_STAR, "june_matched_K3": V_TOP, "carrier_K3": 582.2425054210122,
           "sham_K3": -4009.1762051589712, "carrier_K10": 571.4884463397937}
E58_PAIRS = {"feb_K3": ("F3bag", "P3bag"), "june_matched_K3": ("M3bag", "P3bag"),
             "carrier_K3": ("C3bag", "P3bag"), "sham_K3": ("S3bag", "P3bag"),
             "carrier_K10": ("C10bag", "P10bag")}
E57_SPEC_COMMIT = "95d82b6611e2768e2a25fd596bd113f168eb307e"
E58_SPEC_COMMIT = "22e2a6b91f94040d1999d483ae4bf95565ca1f72"
E58_CODE_SHA = "9a970bacf95c05dbf923d7a0e7260e93a69fa90c69553ceebe6e09928338dd11"
E59_SPEC_COMMIT = "49db4157d8bc43edde6c2b8386e90b6f58929578"
ARM_MONTH = {"F": "2025-02", "M": "2025-06"}


def check_pin(p: Path) -> str:
    rel = str(Path(p).resolve().relative_to(ROOT))
    want = SOURCE.get(rel) or E59_CENSUS.get(rel) or E59_VECTORS.get(rel)
    if want is None:
        raise SystemExit(f"{rel} is outside the positive manifest, exp-059's verified census and "
                         f"exp-059's 36 frozen fit records")
    got = sha_file(p)
    if got != want:
        raise SystemExit(f"{rel} is {got[:16]}, not the pinned {want[:16]}")
    return got


def source_check() -> dict:
    opened = sorted(r for r in OPENED if not r.startswith(OWN_PREFIX) and (ROOT / r).is_file())
    inside, census_only, e59_vec, outside, differ = 0, [], [], [], []
    for rel in opened:
        s = sha_file(ROOT / rel)
        if rel in SOURCE:
            if SOURCE[rel] == s:
                inside += 1
            else:
                differ.append(rel)
        elif rel in E59_CENSUS:
            (census_only if E59_CENSUS[rel] == s else differ).append(rel)
        elif rel in E59_VECTORS:
            (e59_vec if E59_VECTORS[rel] == s else differ).append(rel)
        else:
            outside.append(rel)
    return {"opened": len(opened), "in_manifest": inside, "in_exp059_census_only": census_only,
            "in_exp059_record_pins_only": e59_vec,
            "outside_every_pin": outside, "differ": differ, "clean": not (differ or outside),
            "files": opened}


class Own:
    """SPEC 5: V = [SSE(P3_L0) - SSE(X3_L0)] / n on every July fold row, on exp-058's route.

    exp-058 sums (y - pbase - vector)^2 into (UTC day, airport) cells, adds the cells into days with
    exp-055 lossfals's day matrix, adds the days, and subtracts the two totals.  Each total is near
    7.6e9 s2 while their difference is near 3e7 s2, so a route that subtracts row by row first is
    more accurate but does NOT reproduce the registered digits.  A column's day sum does not depend
    on the other columns of the frame, so one matrix over every column gives the same numbers as
    exp-059's per-arm matrices; the 1e-12 reproductions below prove it.
    """

    def __init__(self, pan: dict, vec: dict, y: np.ndarray, days: np.ndarray):
        pf = pan[F]
        self.pf, self.vec, self.y, self.days = pf, vec, y, days
        self.n = len(y)
        self.cd, uq = pd.factorize(pd.MultiIndex.from_arrays([pf["day"], pf["ap"]]), sort=True)
        self.G = len(uq)
        self.cell_day = [a for a, _ in uq]
        self.cell_ap = [b for _, b in uq]
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
        if nm in self.sse:
            raise SystemExit(f"the sse cells of {nm} are already registered")
        e = self.y - self.pf["pbase"] - v
        self.sse[nm] = np.bincount(self.cd, weights=e * e, minlength=self.G)
        return self.sse[nm]

    def totals(self, names) -> tuple:
        cells = pd.DataFrame({"day": self.cell_day, "airport": self.cell_ap,
                              **{f"sse_{nm}_L0": self.sse[nm] for nm in names}})
        Dm, ci = LF.day_matrix(cells, self.days)
        return Dm, ci, Dm.sum(0)

    def value(self, xn: str, yn: str, Dm: np.ndarray, ci: dict, P: np.ndarray) -> tuple:
        jx, jy = ci[f"sse_{xn}_L0"], ci[f"sse_{yn}_L0"]
        return float((P[jy] - P[jx]) / self.n), Dm[:, jy] - Dm[:, jx]


def band_test(v: float) -> bool:
    return bool(BAND[0] <= v <= BAND[1])


def distance_to_band(v: float) -> float:
    """0.0 inside B; otherwise the signed distance to the nearer edge (negative below, positive
    above), so the sign says which side the row falls on."""
    if v < BAND[0]:
        return v - BAND[0]
    if v > BAND[1]:
        return v - BAND[1]
    return 0.0


# ---------------------------------------------------------------------------- the vectors -------

def load_exp057_L0(vec: dict) -> dict:
    info = {}
    for kind, seeds in (("S", (2, 3)), ("C", range(4, 11))):
        for s in seeds:
            tag = f"{kind}-s{s}-L0"
            rp = E57 / "analysis/fits" / f"{tag}.json"
            check_pin(rp)
            rec = json.loads(rp.read_text())
            if rec.get("spec_commit") != E57_SPEC_COMMIT:
                raise SystemExit(f"exp-057 {tag}: not exp-057's registration")
            p = ROOT / rec["vector"]
            if check_pin(p) != rec["vector_sha256"]:
                raise SystemExit(f"exp-057 {tag}: the vector is not the recorded bytes")
            z = np.load(p)
            if list(z.keys()) != [F]:
                raise SystemExit(f"exp-057 {tag}: the vector holds {list(z.keys())}")
            vec[f"{kind}{s}_L0"] = z[F].astype("float64")
            info[tag] = {"vector": rec["vector"], "sha256": rec["vector_sha256"]}
    return info


def load_exp058_L0(vec: dict) -> dict:
    info = {}
    check_pin(E58 / "analysis/plan.json")
    e58plan = json.loads((E58 / "analysis/plan.json").read_text())
    for arm in ("F", "M"):
        bsha = e58plan["match"]["months"][ARM_MONTH[arm]]["selection"]["ids_sha256_block_order"]
        for s in SEEDS3:
            tag = f"{arm}-s{s}-L0"
            rp = E58 / "analysis/fits" / f"{tag}.json"
            check_pin(rp)
            rec = json.loads(rp.read_text())
            if rec.get("spec_commit") != E58_SPEC_COMMIT or rec.get("code_sha256") != E58_CODE_SHA:
                raise SystemExit(f"exp-058 {tag}: not exp-058's registration or plan code")
            if (rec["arm"] != arm or rec["seed"] != s or rec["level"] != 0
                    or rec["month"] != ARM_MONTH[arm] or rec["block_rows"] != N_TOP
                    or rec["block_ids_sha256_block_order"] != bsha):
                raise SystemExit(f"exp-058 {tag}: the record does not identify this arm")
            p = ROOT / rec["vector"]
            if check_pin(p) != rec["vector_sha256"]:
                raise SystemExit(f"exp-058 {tag}: the vector is not the recorded bytes")
            z = np.load(p)
            if list(z.keys()) != [F]:
                raise SystemExit(f"exp-058 {tag}: the vector holds {list(z.keys())}")
            vec[f"{arm}{s}_L0"] = z[F].astype("float64")
            info[tag] = {"vector": rec["vector"], "sha256": rec["vector_sha256"],
                         "rows_moved": rec["rows_moved"]}
    return info


def load_exp059_k4_L0(vec: dict, out_N: dict) -> dict:
    """exp-059's k = 4 arm at L0, read ONLY to reproduce its recorded V and order k = 1, 2, 3, 4."""
    check_pin(E59 / "analysis/plan.json")
    reg = json.loads((E59 / "analysis/plan.json").read_text())["grid"]["rows"][str(E59_K4)]
    out_N[E59_K4] = reg["block_rows"]
    info = {}
    for s in SEEDS3:
        tag = f"G{E59_K4:02d}-s{s}-L0"
        rp = E59 / "analysis/fits" / f"{tag}.json"
        check_pin(rp)
        rec = json.loads(rp.read_text())
        if rec.get("spec_commit") != E59_SPEC_COMMIT:
            raise SystemExit(f"exp-059 {tag}: not exp-059's registration")
        if (rec["k"] != E59_K4 or rec["seed"] != s or rec["level"] != 0
                or rec["month"] != "2025-06" or rec["block_rows"] != reg["block_rows"]
                or rec["block_ids_sha256_block_order"] != reg["ids_sha256_block_order"]
                or rec["block_count_vector_sha256"] != reg["count_vector_sha256"]):
            raise SystemExit(f"exp-059 {tag}: the record does not identify the k = 4 grid row")
        p = ROOT / rec["vector"]
        if check_pin(p) != rec["vector_sha256"]:
            raise SystemExit(f"exp-059 {tag}: the vector is not the recorded bytes")
        z = np.load(p)
        if list(z.keys()) != [F]:
            raise SystemExit(f"exp-059 {tag}: the vector holds {list(z.keys())}")
        vec[f"G{E59_K4:02d}{s}_L0"] = z[F].astype("float64")
        info[tag] = {"vector": rec["vector"], "sha256": rec["vector_sha256"],
                     "block_rows": rec["block_rows"]}
    return info


def load_new_L0(vec: dict, plan: dict) -> dict:
    """The nine new L0 vectors of k = 1, 2 and 3.  All nine must exist (SPEC 8.1)."""
    ok_code = {plan["code"]["sha256"]}
    spec_commit = plan["spec_commit"]
    if spec_commit != MAN["registration"]["spec_commit"]:
        raise SystemExit("the plan's registration commit is not the manifest's")
    info, missing = {}, []
    for k in PROBE_K:
        arm = f"G{k:02d}"
        grow = plan["grid"]["rows"][str(k)]
        if grow["block_rows"] != GRID_N[k]:
            raise SystemExit(f"the plan's k = {k} row has {grow['block_rows']} rows, not {GRID_N[k]}")
        for s in SEEDS3:
            tag = f"{arm}-s{s}-L0"
            rp = ANALYSIS / "fits" / f"{tag}.json"
            if not rp.exists():
                missing.append(tag)
                continue
            rec = json.loads(rp.read_text())
            if rec.get("code_sha256") not in ok_code or rec.get("spec_commit") != spec_commit:
                raise SystemExit(f"{tag}: the record's code or registration is not the plan's")
            if (rec["arm"] != arm or rec["k"] != k or rec["seed"] != s or rec["level"] != 0
                    or rec["month"] != "2025-06"):
                raise SystemExit(f"{tag}: the record does not identify this arm, seed and level")
            if (rec["block_rows"] != grow["block_rows"]
                    or rec["block_ids_sha256_block_order"] != grow["ids_sha256_block_order"]
                    or rec["block_count_vector_sha256"] != grow["count_vector_sha256"]):
                raise SystemExit(f"{tag}: the block is not the registered k = {k} grid row")
            p = ROOT / rec["vector"]
            if sha_file(p) != rec["vector_sha256"]:
                raise SystemExit(f"{tag}: the vector is not the recorded bytes")
            z = np.load(p)
            if list(z.keys()) != [F]:
                raise SystemExit(f"{tag}: the vector holds {list(z.keys())}")
            vec[f"{arm}{s}_L0"] = z[F].astype("float64")
            info[tag] = {"vector": rec["vector"], "sha256": rec["vector_sha256"], "k": k,
                         "block_rows": rec["block_rows"],
                         "block_ids_sha256_block_order": rec["block_ids_sha256_block_order"],
                         "rows_moved": rec["rows_moved"], "frame_sha256": rec["frame_sha256"],
                         "source_sha256": rec["source_sha256"],
                         "code_sha256": rec["code_sha256"],
                         "seconds_fit_wall": rec["seconds_fit_wall"],
                         "seconds_fit_process_cpu": rec["seconds_fit_process_cpu"]}
    if missing:
        raise SystemExit(f"the probe reads a new own value only after all nine L0 members are "
                         f"fitted (SPEC 8.1); these are missing: {missing}")
    return info


# ---------------------------------------------------------------------------- main --------------

def main() -> None:
    t0, c0 = time.time(), time.process_time()
    np.seterr(divide="ignore", invalid="ignore")
    NO.refuse_existing([OUT], overwrite=False)
    if (ANALYSIS / "measure.json").exists():
        raise SystemExit("analysis/measure.json exists: the own-value read comes first (SPEC 8)")
    plan = json.loads(PLAN.read_text())
    if not REPRO.exists():
        raise SystemExit("analysis/repro.json does not exist: SPEC 6 and 7 come before the new "
                         "interpretation")
    rp = json.loads(REPRO.read_text())
    if rp.get("source_check", {}).get("clean") is not True:
        raise SystemExit("analysis/repro.json does not record a clean source check")
    for need in ("R2_exp051_gains", "R3_R4_exp055_scripts", "R6", "R7_exp057", "R8_exp058",
                 "R8_vii_populations", "separate_reproductions"):
        if need not in rp:
            raise SystemExit(f"analysis/repro.json has no {need} block")
    if rp["separate_reproductions"]["B4_exp059_fitted_V"]["max_abs_diff_s2_row"] > TOL_12:
        raise SystemExit("repro.json did not reproduce exp-059's fitted V at 1e-12")

    rec = {"what": "exp-062 ownvalue-refine: the new near-origin own values at k = 1, 2 and 3, and "
                   "the deterministic selection.  Zero fit; own values and day-block variation "
                   "bands only.",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "manifest_sha256": sha_file(MANIFEST), "plan_sha256": sha_file(PLAN),
           "repro_sha256": sha_file(REPRO), "spec_commit": plan["spec_commit"],
           "plan_code_sha256": plan["code"]["sha256"],
           "definition": "V = [SSE(P3_L0) - SSE(X3_L0)] / n on every July fold row (n = 190,713), "
                         "SSE over y - pbase - vector, summed cell to day to total; positive means "
                         "the arm's own L0 vector predicts July better than the plain K = 3 bag; "
                         "measured, never subtracted",
           "constants": {"V_star_february": V_STAR, "band_B": list(BAND), "band_rule": "V* +/- 10 %",
                         "band_is_frozen": "the assignment fixes B; it is not widened and not "
                                           "replaced by a variation band, and the variation bands "
                                           "below play no part in the test",
                         "anchor_V_at_0": 0.0, "anchor_V_at_60777": V_TOP,
                         "registered_rows": {str(k): GRID_N[k] for k in PROBE_K}},
           "B": B, "blocks": list(BLOCKS), "generator": "np.random.default_rng(5500 + L)",
           "variation_bands": "2.5 % to 97.5 % of 2,000 moving-block day draws of the per-row own "
                              "value; empirical variation, not a confidence interval",
           "checks": {}}

    # ---- R1 ---------------------------------------------------------------------------------
    pan, vec, pin = V.load()
    lab = LF.load_labels(F, pan[F]["ids"])
    y = lab["y"]
    v = vec[F]
    rec["checks"]["R1_originvec_load"] = True
    print("R1 exp-055 pins verified; the labels equal exp-051's OOF y_true", flush=True)

    # ---- R5 ---------------------------------------------------------------------------------
    check_pin(E55 / "analysis/screen.json")
    scr = json.loads((E55 / "analysis/screen.json").read_text())
    days = np.sort(np.unique(pan[F]["day"]))
    if len(days) != 31:
        raise SystemExit("July does not have 31 fold days")
    W = {}
    for L in BLOCKS:
        rng = np.random.default_rng(5500 + L)
        st = {key: C.mbb_starts(rng, 31, L, B)
              for key in ("july_loo", "2026-07", "january", "2026-01", "december")}
        sig = rng.integers(1, 4, size=(B, 3))
        want = scr["draw_sha256"][f"L{L}"]
        got = {k_: C.sha_arr(x_) for k_, x_ in st.items()} | {"seeds": C.sha_arr(sig)}
        if got != {k_: x_ for k_, x_ in want.items() if k_ != "multiset_counts"}:
            raise SystemExit(f"R5 L{L}: the regenerated draws are not exp-055's")
        W[L] = C.mbb_weights(st["july_loo"], 31, L)
    rec["checks"]["R5_draws_equal_exp055"] = True
    print("R5 the regenerated day draws equal exp-055's", flush=True)

    # ---- R6 and the comparator vectors ------------------------------------------------------
    for s in range(4, 11):
        p = C.member_path(0, s)
        check_pin(p)
        v[f"P{s}_L0"] = np.load(p)[F].astype("float64")
    rec["exp057_vectors_L0"] = load_exp057_L0(v)
    rec["exp058_vectors_L0"] = load_exp058_L0(v)
    OW = Own(pan, v, y, days)
    if OW.n != 190_713:
        raise SystemExit(f"July has {OW.n} fold rows, not 190,713")
    p3 = OW.bag("P")
    p10 = OW.bag("P", SEEDS10)
    if not np.array_equal(p3, v["P3bag_L0"]):
        raise SystemExit("R6: the rebuilt plain K = 3 bag is not exp-055's pinned bag")
    d10 = float(np.max(np.abs(p10 - v["T10_L0"])))
    if d10 > TOL_9:
        raise SystemExit(f"R6: the rebuilt plain K = 10 bag differs from Q15's head by {d10:.3e} s")
    rec["checks"]["R6_plain_bags"] = {"K3_bit_identical": True, "K10_max_abs_diff_s": d10}
    print(f"R6 the plain K = 3 bag is exp-055's pinned bag; the K = 10 bag meets Q15's head to "
          f"{d10:.1e} s", flush=True)

    # ---- R8 (iv) and the fit-free anchors, before any new vector is opened -------------------
    OW.cells("P3bag", p3)
    OW.cells("P10bag", p10)
    for X in ("F", "M", "C", "S"):
        OW.cells(f"{X}3bag", OW.bag(X))
    OW.cells("C10bag", OW.bag("C", SEEDS10))
    for s in SEEDS3:
        OW.cells(f"P{s}", v[f"P{s}_L0"])
    Dm8, ci8, P8 = OW.totals(sorted({n for pr in E58_PAIRS.values() for n in pr}))
    repro58, worst = {}, 0.0
    for nm, (X, Y) in E58_PAIRS.items():
        val, dsum = OW.value(X, Y, Dm8, ci8, P8)
        repro58[nm] = {"X": X, "Y": Y, "value": val, "exp058": E58_OWN[nm],
                       "abs_diff": abs(val - E58_OWN[nm]),
                       "intervals": {f"L{L}": LF.interval((W[L] @ dsum) / (W[L] @ OW.day_n))
                                     for L in BLOCKS}}
        worst = max(worst, repro58[nm]["abs_diff"])
    rec["R8_iv_exp058_own_values"] = {"by_arm": repro58, "max_abs_diff_s2_row": worst}
    if worst > TOL_9:
        raise SystemExit(f"exp-058's five L0 own values are missed by {worst:.3e} s2/row")
    zero, _ = OW.value("P3bag", "P3bag", Dm8, ci8, P8)
    rec["checks"]["anchor_V_at_0_identity"] = {
        "value": zero,
        "why": "an empty block leaves the plain member's own L0 fit, so X3 = P3 and V = 0 by "
               "construction; the identity SSE(P3) - SSE(P3) = 0 is computed here, no fit exists"}
    if zero != 0.0:
        raise SystemExit("the V(0) identity is not exactly zero")
    if (abs(repro58["feb_K3"]["value"] - V_STAR) > TOL_9
            or abs(repro58["june_matched_K3"]["value"] - V_TOP) > TOL_9):
        raise SystemExit("the frozen anchors do not reproduce in this process")
    print(f"R8 (iv) exp-058's five L0 own values reproduced (max |diff| {worst:.2e} s2/row): "
          f"February {repro58['feb_K3']['value']:+.10f}, matched June "
          f"{repro58['june_matched_K3']['value']:+.10f}; V(0) = {zero}", flush=True)

    # ---- the nine new L0 vectors and exp-059's k = 4 arm -------------------------------------
    rec["new_vectors_L0"] = load_new_L0(v, plan)
    rowsN = dict(GRID_N)
    rec["exp059_k4_vectors_L0"] = load_exp059_k4_L0(v, rowsN)
    arms, names = {}, ["P3bag"] + [f"P{s}" for s in SEEDS3]
    for k in (*PROBE_K, E59_K4):
        arm = f"G{k:02d}"
        OW.cells(f"{arm}3bag", OW.bag(arm))
        names.append(f"{arm}3bag")
        for s in SEEDS3:
            OW.cells(f"{arm}{s}", v[f"{arm}{s}_L0"])
            names.append(f"{arm}{s}")
    Dk, cik, Pk = OW.totals(sorted(names))
    for k in (*PROBE_K, E59_K4):
        arm = f"G{k:02d}"
        val, dsum = OW.value(f"{arm}3bag", "P3bag", Dk, cik, Pk)
        members = {}
        for s in SEEDS3:
            mv, mds = OW.value(f"{arm}{s}", f"P{s}", Dk, cik, Pk)
            members[f"s{s}"] = {"value": mv,
                                "intervals": {f"L{L}": LF.interval((W[L] @ mds) / (W[L] @ OW.day_n))
                                              for L in BLOCKS}}
        arms[arm] = {"k": k, "f": f"{k}/24", "N": rowsN[k],
                     "V": val, "in_band": band_test(val),
                     "abs_V_minus_V_star": abs(val - V_STAR),
                     "distance_to_band_s2_row": distance_to_band(val),
                     "new_in_this_experiment": k in GRID_N, "members": members,
                     "intervals": {f"L{L}": LF.interval((W[L] @ dsum) / (W[L] @ OW.day_n))
                                   for L in BLOCKS}}
        mtxt = " ".join(f"{members['s%d' % s]['value']:+.1f}" for s in SEEDS3)
        print(f"{arm} (k={k}, N={arms[arm]['N']:,}): V = {val:+.10f} s2/row, in band "
              f"{arms[arm]['in_band']}, |V - V*| = {arms[arm]['abs_V_minus_V_star']:.4f}, "
              f"members {mtxt}", flush=True)
    rec["arms"] = arms
    #: exp-059's k = 4 value is a REPRODUCTION: it must equal its record, and it is used only to
    #: order k = 1, 2, 3, 4.  No prediction of this experiment comes from an exp-059 vector.
    d4 = abs(arms[f"G{E59_K4:02d}"]["V"] - E59_V4)
    rec["exp059_k4_reproduction"] = {
        "V_rebuilt": arms[f"G{E59_K4:02d}"]["V"], "V_exp059": E59_V4, "abs_diff": d4,
        "why": "exp-059's k = 4 arm is read here only to reproduce its recorded value, so the order "
               "over k = 1, 2, 3, 4 is one matrix in one process; this experiment builds no new "
               "prediction from an exp-059 vector"}
    if d4 > TOL_12:
        raise SystemExit(f"exp-059's k = 4 value is missed by {d4:.3e} s2/row")

    # ---- the measured order and the monotone read -------------------------------------------
    order = [{"k": 0, "N": 0, "V": 0.0, "source": "identity (empty block)"}] + [
        {"k": k, "N": arms[f"G{k:02d}"]["N"], "V": arms[f"G{k:02d}"]["V"],
         "source": ("fitted here" if k in GRID_N else "exp-059, reproduced here")}
        for k in (*PROBE_K, E59_K4)]
    rec["measured_order_k1_to_k4"] = {
        "points": order,
        "ascending_in_N": all(order[i]["V"] <= order[i + 1]["V"] + TOL_9
                              for i in range(len(order) - 1)),
        "adjacent_differences": [{"from_k": order[i]["k"], "to_k": order[i + 1]["k"],
                                  "delta_V": order[i + 1]["V"] - order[i]["V"],
                                  "delta_N": order[i + 1]["N"] - order[i]["N"]}
                                 for i in range(len(order) - 1)],
        "claim_boundary": "the order is a statement about these five measured points only; no "
                          "monotonicity is assumed for an unmeasured row"}

    # ---- the fixed band test and the deterministic selection --------------------------------
    inside = [k for k in PROBE_K if arms[f"G{k:02d}"]["in_band"]]
    rec["band_test"] = {"band_B": list(BAND), "rows_tested": list(PROBE_K),
                        "rows_in_band": inside,
                        "by_row": {str(k): {"V": arms[f"G{k:02d}"]["V"],
                                            "in_band": arms[f"G{k:02d}"]["in_band"],
                                            "distance_to_band_s2_row":
                                                arms[f"G{k:02d}"]["distance_to_band_s2_row"],
                                            "abs_V_minus_V_star":
                                                arms[f"G{k:02d}"]["abs_V_minus_V_star"],
                                            "three_day_day_block_band":
                                                [arms[f"G{k:02d}"]["intervals"]["L3"]["lo"],
                                                 arms[f"G{k:02d}"]["intervals"]["L3"]["hi"]]}
                                   for k in PROBE_K},
                        "rule": "the exact stored V of the K = 3 bag against the frozen B; the "
                                "variation bands take no part in the test"}
    if inside:
        cand = sorted((arms[f"G{k:02d}"]["abs_V_minus_V_star"], -arms[f"G{k:02d}"]["N"], k)
                      for k in inside)
        k_star = cand[0][2]
        tie = [c[2] for c in cand if c[0] == cand[0][0]]
        a = arms[f"G{k_star:02d}"]
        rec["branch"] = "near_origin_matched"
        rec["june_star"] = {"k": k_star, "arm": f"G{k_star:02d}", "N": a["N"], "V": a["V"],
                            "abs_V_minus_V_star": a["abs_V_minus_V_star"],
                            "selected_among": inside, "exact_stored_value_tie": tie,
                            "selection_rule": "the smaller exact stored abs(V - V*); an exact "
                                              "stored-value tie selects the larger N",
                            "fits_so_far": 9, "fits_to_come": 9,
                            "frozen_before_any_L1_L2_L3_read": True}
        rec["reading"] = (f"SPEC 8.6: {len(inside)} of the three new near-origin rows are inside B; "
                          f"June* is the k = {k_star} arm (N = {a['N']:,}, V = {a['V']:+.4f} s2/row, "
                          f"|V - V*| = {a['abs_V_minus_V_star']:.4f} s2/row).  The choice is frozen "
                          f"here, before any L1, L2 or L3 vector or movement-amplitude read.")
    else:
        rec["branch"] = "near_origin_unmatched"
        rec["june_star"] = None
        near = min(PROBE_K, key=lambda k: abs(arms[f"G{k:02d}"]["distance_to_band_s2_row"]))
        rec["reading"] = (
            f"SPEC 8.6: none of the three new near-origin rows is inside B = {list(BAND)}; the probe "
            f"stops as near-origin unmatched.  The nearest row is k = {near} at V = "
            f"{arms[f'G{near:02d}']['V']:+.4f} s2/row, "
            f"{abs(arms[f'G{near:02d}']['distance_to_band_s2_row']):.4f} s2/row from the band edge "
            f"and {arms[f'G{near:02d}']['abs_V_minus_V_star']:.4f} s2/row from V*.  No new m, loss, "
            f"subgroup or step diagnostic is computed, and no further fit is routed.  This result "
            f"covers only these three new grid rows: it cannot state that all 23 grid rows are "
            f"unmatched, and it assumes no monotonicity for an unmeasured row.")
        rec["stop"] = {"kind": "near-origin unmatched", "rows_covered": list(PROBE_K),
                       "nearest_row": near,
                       "does_not_cover": "the other 20 registered grid rows, any finer grid, and "
                                         "any unmeasured row between two measured rows",
                       "no_new_diagnostic": True, "routes_no_further_fit": True}
    print(f"branch: {rec['branch']}", flush=True)
    print(f"reading: {rec['reading']}", flush=True)
    print(f"June*: {rec['june_star']}", flush=True)

    # ---- the source check -------------------------------------------------------------------
    sc = source_check()
    rec["source_check"] = {k_: x_ for k_, x_ in sc.items() if k_ != "files"} | {
        "files_opened": sc["files"]}
    if not sc["clean"]:
        raise SystemExit(f"the source check fails: {json.dumps(sc, indent=1)}")
    rec["exp062_vectors_opened"] = sorted(r for r in OPENED if r.startswith(WORK_REL))
    if len(rec["exp062_vectors_opened"]) != 9:
        raise SystemExit(f"the read opened {len(rec['exp062_vectors_opened'])} new vectors, not 9")
    rec["seconds"] = round(time.time() - t0, 1)
    rec["seconds_process_cpu"] = round(time.process_time() - c0, 1)
    C.write_json(OUT, rec)
    print(f"written {OUT.relative_to(ROOT)} ({rec['seconds']} s wall, "
          f"{rec['seconds_process_cpu']} s process cpu)", flush=True)


if __name__ == "__main__":
    main()
