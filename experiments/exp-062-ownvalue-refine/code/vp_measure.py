"""exp-062 ownvalue-refine: the conditional completion (SPEC 9) and the fixed reading (SPEC 10).
ZERO FIT.  READS LABELS.  It runs only when analysis/probe.json selected a June*.

    vp_ctl.py --kind measure-standin --log logs/measure-standin.log -- \
        .venv/bin/python .../vp_measure.py --standin --scratch <session scratch dir>
    vp_ctl.py --kind measure --log logs/measure.log -- .venv/bin/python .../vp_measure.py

What it measures, on the same July H, days, member triples and plain K = 3 denominator exp-058
used, with exp-058's exact 2,000-draw one-, two- and three-day moving-block plus matched-member
method:

  every raw member and every K = 3 bag of February (exp-058's fixed F3 arm), June* (selected by
  the probe), the matched June k = 24 arm, exp-051's full June carrier and exp-057's sham; m, q,
  m - q, m / q, cosine, both RMS, the changed rows, the full loss identities (A, B, g_C, g_hat,
  g_plain, J, amplitude, alignment and the exact split), each arm's L0 own value, and the airport,
  NM-regime and serving-regime decompositions.  It saves the paired row contributions, the UTC-date
  sums, the three draw receipts and COMMON's full fold and serving tables.

Registered economies (SPEC 12.3), so the non-fit allocation holds:

  * R2, R3, R4, R7, R8 and the 2,343 / 1,287-leaf population comparison are CITED from
    analysis/repro.json, which ran them in full and refused on every mismatch.  This process
    verifies that record's bytes and that each block passed; it does not run them again.
  * the K = 10 comparator gets its point and its day-only interval here, and its day-plus-member
    interval is cited from exp-057, so the 2,000-draw ten-member loop does not run again.

`q` is Q15's pinned registered constant.  The point m - q and m / q use that constant; the
interval route uses exp-055's exact per-draw q from the two pinned receipts, which is exp-058's
registered interval method and not a rebuild of the pinned point.  This experiment makes no new
claim about q.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

P = argparse.ArgumentParser()
P.add_argument("--standin", action="store_true",
               help="the registered dry run: stand-in June* vectors, 50 draws, block length 3 only, "
                    "every output into --scratch, and no artifacts/exp-062/work file opened")
P.add_argument("--scratch", default=None, help="the dry run's output directory")
ARGS = P.parse_args()
if ARGS.standin and not ARGS.scratch:
    raise SystemExit("--standin needs --scratch")
if ARGS.scratch and not ARGS.standin:
    raise SystemExit("--scratch is the dry run's directory only")

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
REPRO = ANALYSIS / "repro.json"
PROBE = ANALYSIS / "probe.json"
STANDIN = ARGS.standin
SCRATCH = Path(ARGS.scratch).resolve() if ARGS.scratch else None
OUT = (SCRATCH / "measure-standin.json") if STANDIN else (ANALYSIS / "measure.json")
REC_DIR = (SCRATCH / "receipts") if STANDIN else (ANALYSIS / "receipts")
ART = ROOT / "artifacts/exp-062"
OOFD = (SCRATCH / "oof") if STANDIN else (ART / "oof")
ROWD = (SCRATCH / "receipts") if STANDIN else (ART / "receipts")
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
import screen as SC                                                   # noqa: E402

for _name, _want in (("common", E55 / "code/common.py"), ("originvec", E55 / "code/originvec.py"),
                     ("screen", E55 / "code/screen.py"), ("lossfals", E55 / "code/lossfals.py")):
    if Path(sys.modules[_name].__file__).resolve() != _want:
        raise SystemExit(f"module {_name} resolves to {sys.modules[_name].__file__} (name collision)")
if "carrier" in sys.modules or "memberspread" in sys.modules:
    raise SystemExit("exp-051's closure is imported in the measurement process")

import itertools                                                      # noqa: E402
import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

A = C.A
NO = C.NO
F = "july_loo"
B = 50 if STANDIN else 2000
BLOCKS = (3,) if STANDIN else (1, 2, 3)
KS = (1, 2, 3)
STEP = C.STEP
SP = [STEP[k] for k in KS]
READ_STEPS = ("v20", "v21")
MARGIN = 0.03
SEEDS3 = (1, 2, 3)
SEEDS10 = tuple(range(1, 11))
MULTI3 = list(itertools.combinations_with_replacement(SEEDS3, 3))
MS3 = ["".join(map(str, t)) for t in MULTI3]
STATS = ("ry", "yy", "rx", "xx", "xy")
TOL_9, TOL_12 = 1e-9, 1e-12
Q_POINT = {"v20": 0.7541214069685656, "v21": 0.7872337829612198, "v22": 0.5869686712745045}
R8_III_FIELDS = ("m", "q", "m_minus_q", "m_over_q", "I", "cos", "rms_H_carrier_s", "rms_H_plain_s",
                 "A", "B", "g_C", "g_hat", "g_plain", "J", "amplitude", "alignment")
E57_SPEC_COMMIT = "95d82b6611e2768e2a25fd596bd113f168eb307e"
E58_SPEC_COMMIT = "22e2a6b91f94040d1999d483ae4bf95565ca1f72"
E58_CODE_SHA = "9a970bacf95c05dbf923d7a0e7260e93a69fa90c69553ceebe6e09928338dd11"
ARM_MONTH = {"F": "2025-02", "M": "2025-06"}
V_STAR = 168.58928050561724
BAND = (151.7303524550555, 185.44820855617897)
V_TOP = 493.2541127299781
N_TOP = 60_777
E58_OWN = {"feb_K3": V_STAR, "june_matched_K3": V_TOP, "carrier_K3": 582.2425054210122,
           "sham_K3": -4009.1762051589712, "carrier_K10": 571.4884463397937}

#: the selected arm.  SPEC 9 runs only after the probe froze it.
if not PROBE.exists():
    raise SystemExit("analysis/probe.json does not exist: the probe comes first (SPEC 8)")
PR = json.loads(PROBE.read_text())
JS = PR.get("june_star")
if not JS:
    raise SystemExit(f"the probe stopped as {PR.get('branch')}: SPEC 9 routes no measurement "
                     f"(the assignment: a near-origin unmatched result routes no further fit)")
JK = int(JS["k"])
JARM = JS["arm"]
JN = int(JS["N"])
JV = float(JS["V"])

#: name -> (carrier level key, plain level key, K, role)
COMPS = {"feb_K3": ("F3bag", "P3bag", 3, "exp-058's fixed February 2025 arm, 60,777 rows"),
         "june_star_K3": (f"{JARM}3bag", "P3bag", 3,
                          f"June*: the selected near-origin June 2025 arm, k = {JK}/24, N = {JN:,}"),
         "june_matched_K3": ("M3bag", "P3bag", 3, "exp-058's matched June arm, k = 24/24, 60,777 rows"),
         "carrier_K3": ("C3bag", "P3bag", 3, "comparator: exp-051's full June carrier"),
         "sham_K3": ("S3bag", "P3bag", 3, "comparator: exp-057's K = 3 sham"),
         "carrier_K10": ("C10bag", "P10bag", 10, "comparator: exp-057's K = 10 June carrier")}
for _nm, _x in (("feb", "F"), ("june_star", JARM), ("june_matched", "M"), ("carrier", "C"),
                ("sham", "S")):
    for _s in SEEDS3:
        COMPS[f"{_nm}_s{_s}"] = (f"{_x}{_s}", f"P{_s}", 1, f"{_nm} member {_s}")
BAG_COMPS = ("feb_K3", "june_star_K3", "june_matched_K3", "carrier_K3", "sham_K3")
K3_DRAW = {"feb_K3": "F", "june_star_K3": JARM, "june_matched_K3": "M", "carrier_K3": "C",
           "sham_K3": "S"}
SSE_NAMES = (["P3bag", "P10bag", "C10bag", "F3bag", f"{JARM}3bag", "M3bag", "C3bag", "S3bag"]
             + [f"{nm}{s}" for nm in ("P", "F", JARM, "M", "C", "S") for s in SEEDS3])
DIFFS = {"feb_minus_june_star": ("feb_K3", "june_star_K3"),
         "feb_minus_june_matched": ("feb_K3", "june_matched_K3"),
         "feb_minus_full_june_K3": ("feb_K3", "carrier_K3")}
DIFF_MEMBER_X = {"feb_minus_june_star": ("feb", "june_star"),
                 "feb_minus_june_matched": ("feb", "june_matched"),
                 "feb_minus_full_june_K3": ("feb", "carrier")}


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


def fl(x):
    return float(x)


def interval(x) -> dict:
    return LF.interval(x)


def band_of(v, lo, hi) -> dict:
    return {"point": fl(v), "band": [lo, hi], "inside": bool(lo <= v <= hi)}


def quantities(n, nH, ry, yy, rx, xx, xy) -> dict:
    A_, Bq = ry / n, yy / n
    m = np.sqrt(xx / yy)
    ghat = 2 * m * A_ - m ** 2 * Bq
    gplain = 2 * A_ - Bq
    gC = (2 * rx - xx) / n
    cos = xy / np.sqrt(xx * yy)
    return {"A": A_, "B": Bq, "m": m, "g_hat": ghat, "g_plain": gplain, "g_C": gC,
            "J": np.abs(gplain - gC) - np.abs(ghat - gC), "cos": cos,
            "err_hat": np.abs(ghat - gC), "err_plain": np.abs(gplain - gC),
            "amplitude": (m - 1) * (2 * A_ - (m + 1) * Bq), "alignment": 2 * (rx / n - m * A_),
            "alignment_direction": 2 * m * (cos - 1) * A_,
            "alignment_orthogonal": 2 * (rx - (xy / yy) * ry) / n,
            "m_star": A_ / Bq, "best_amplitude_value": A_ ** 2 / Bq,
            "g_C_minus_best": gC - A_ ** 2 / Bq,
            "rms_H_carrier_s": np.sqrt(xx / nH), "rms_H_plain_s": np.sqrt(yy / nH)}


def q_cols(S, ci, c: str, s: str) -> dict:
    return quantities(S[..., ci["n"]], S[..., ci["n_H"]],
                      *(S[..., ci[f"{st}_{c}_{s}"]] for st in STATS))


def imp(m, q):
    return np.abs(1 - q) - np.abs(m - q)


# ------------------------------------------------------------------------- the vectors ----------

def load_exp057(vec: dict) -> dict:
    info = {}
    for kind, seeds in (("S", (2, 3)), ("C", range(4, 11))):
        for s in seeds:
            for L in range(4):
                tag = f"{kind}-s{s}-L{L}"
                rp = E57 / "analysis/fits" / f"{tag}.json"
                check_pin(rp)
                rec = json.loads(rp.read_text())
                if rec.get("spec_commit") != E57_SPEC_COMMIT:
                    raise SystemExit(f"exp-057 {tag}: not exp-057's registration")
                p = ROOT / rec["vector"]
                if check_pin(p) != rec["vector_sha256"]:
                    raise SystemExit(f"exp-057 {tag}: the vector is not the recorded bytes")
                vec[F][f"{kind}{s}_L{L}"] = np.load(p)[F].astype("float64")
                info[tag] = {"vector": rec["vector"], "sha256": rec["vector_sha256"]}
    return info


def load_exp058(vec: dict) -> dict:
    info = {}
    check_pin(E58 / "analysis/plan.json")
    e58plan = json.loads((E58 / "analysis/plan.json").read_text())
    for arm in ("F", "M"):
        bsha = e58plan["match"]["months"][ARM_MONTH[arm]]["selection"]["ids_sha256_block_order"]
        for s in SEEDS3:
            for L in range(4):
                tag = f"{arm}-s{s}-L{L}"
                rp = E58 / "analysis/fits" / f"{tag}.json"
                check_pin(rp)
                rec = json.loads(rp.read_text())
                if rec.get("spec_commit") != E58_SPEC_COMMIT \
                        or rec.get("code_sha256") != E58_CODE_SHA:
                    raise SystemExit(f"exp-058 {tag}: not exp-058's registration or plan code")
                if (rec["arm"] != arm or rec["seed"] != s or rec["level"] != L
                        or rec["month"] != ARM_MONTH[arm] or rec["block_rows"] != N_TOP
                        or rec["block_ids_sha256_block_order"] != bsha):
                    raise SystemExit(f"exp-058 {tag}: the record does not identify this arm")
                p = ROOT / rec["vector"]
                if check_pin(p) != rec["vector_sha256"]:
                    raise SystemExit(f"exp-058 {tag}: the vector is not the recorded bytes")
                vec[F][f"{arm}{s}_L{L}"] = np.load(p)[F].astype("float64")
                info[tag] = {"vector": rec["vector"], "sha256": rec["vector_sha256"],
                             "rows_moved": rec["rows_moved"]}
    return info


def load_june_star(vec: dict, plan1: dict, plan2: dict) -> dict:
    """June*'s twelve vectors: L0 from the stage-1 plan, L1 to L3 from the stage-2 plan.  In the dry
    run every one of them is exp-058's matched June arm instead, and no new vector is opened."""
    info = {}
    for s in SEEDS3:
        for L in range(4):
            tag = f"{JARM}-s{s}-L{L}"
            if STANDIN:
                vec[F][f"{JARM}{s}_L{L}"] = vec[F][f"M{s}_L{L}"].copy()
                info[tag] = {"standin_for": f"M{s}_L{L}"}
                continue
            plan = plan1 if L == 0 else plan2
            grow = plan["grid"]["rows"][str(JK)]
            rec = json.loads((ANALYSIS / "fits" / f"{tag}.json").read_text())
            if (rec.get("code_sha256") != plan["code"]["sha256"]
                    or rec.get("spec_commit") != MAN["registration"]["spec_commit"]):
                raise SystemExit(f"{tag}: the record's code or registration is not the plan's")
            if (rec["arm"] != JARM or rec["k"] != JK or rec["seed"] != s or rec["level"] != L
                    or rec["month"] != "2025-06" or rec["block_rows"] != grow["block_rows"]
                    or rec["block_ids_sha256_block_order"] != grow["ids_sha256_block_order"]
                    or rec["block_count_vector_sha256"] != grow["count_vector_sha256"]):
                raise SystemExit(f"{tag}: the record is not the registered k = {JK} grid row")
            p = ROOT / rec["vector"]
            if sha_file(p) != rec["vector_sha256"]:
                raise SystemExit(f"{tag}: the vector is not the recorded bytes")
            z = np.load(p)
            if list(z.keys()) != [F]:
                raise SystemExit(f"{tag}: the vector holds {list(z.keys())}")
            vec[F][f"{JARM}{s}_L{L}"] = z[F].astype("float64")
            info[tag] = {"vector": rec["vector"], "sha256": rec["vector_sha256"], "k": JK,
                         "level": L, "block_rows": rec["block_rows"],
                         "block_ids_sha256_block_order": rec["block_ids_sha256_block_order"],
                         "rows_moved": rec["rows_moved"], "frame_sha256": rec["frame_sha256"],
                         "seconds_fit_wall": rec["seconds_fit_wall"],
                         "seconds_fit_process_cpu": rec["seconds_fit_process_cpu"]}
    return info


def bag_of(pf: dict, v: dict, nm: str, seeds, L: int) -> np.ndarray:
    return C.bag({s: v[f"{nm}{s}_L{L}"] for s in seeds}, seeds, pf["pbase"], pf["usable"],
                 pf["guard"])


def levels(pan: dict, vec: dict) -> dict:
    """Level vectors by fold.  Every carrier fit is July-only (exp-051's folds=('july_loo',)), so on
    december and january each member and bag equals the plain member or P3bag of the same seed; the
    equality is asserted here by construction."""
    lev = {}
    for f in C.FOLDS:
        pf, v = pan[f], vec[f]
        lv = {nm: [v[f"{nm}_L{L}"] for L in range(4)]
              for nm in [f"P{s}" for s in SEEDS10] + ["P3bag", "T10"]}
        lv["P10bag"] = [bag_of(pf, v, "P", SEEDS10, L) for L in range(4)]
        if f == F:
            for nm in [f"C{s}" for s in SEEDS10] + ["S1", "S2", "S3", "C3bag"]:
                lv[nm] = [v[f"{nm}_L{L}"] for L in range(4)]
            lv["C10bag"] = [bag_of(pf, v, "C", SEEDS10, L) for L in range(4)]
            lv["S3bag"] = [bag_of(pf, v, "S", SEEDS3, L) for L in range(4)]
            for arm in ("F", "M", JARM):
                for s in SEEDS3:
                    lv[f"{arm}{s}"] = [v[f"{arm}{s}_L{L}"] for L in range(4)]
                lv[f"{arm}3bag"] = [bag_of(pf, v, arm, SEEDS3, L) for L in range(4)]
            for L in range(4):
                if not (np.array_equal(bag_of(pf, v, "C", SEEDS3, L), lv["C3bag"][L])
                        and np.array_equal(bag_of(pf, v, "P", SEEDS3, L), lv["P3bag"][L])):
                    raise SystemExit("the K = 3 bags do not rebuild exp-055's pinned bags")
        for nm, x in lv.items():
            if nm.endswith("bag") or nm == "T10":
                for k in KS:
                    if np.any((x[k] - x[k - 1])[~pf["H"]] != 0):
                        raise SystemExit(f"{f} {nm} {STEP[k]}: a bag step moves a row outside H")
        lev[f] = lv
    return lev


# ------------------------------------------------------------------------- cells and draws ------

def cells_july(pan: dict, lv: dict, y: np.ndarray) -> pd.DataFrame:
    pf = pan[F]
    H, pb = pf["H"], pf["pbase"]
    cd, uq = pd.factorize(pd.MultiIndex.from_arrays([pf["day"], pf["ap"]]), sort=True)
    G = len(uq)

    def cell(w):
        return np.bincount(cd, weights=w, minlength=G)
    cols = {"n": np.bincount(cd, minlength=G).astype("int64"),
            "n_H": np.bincount(cd, weights=H.astype("float64"), minlength=G).astype("int64")}
    for c, (X, Y, K, _) in COMPS.items():
        for k in KS:
            s = STEP[k]
            x0, x1, y0, y1 = lv[X][k - 1], lv[X][k], lv[Y][k - 1], lv[Y][k]
            r = y - pb - x0
            dx, dy = x1 - x0, y1 - y0
            for st, v in (("ry", r * dy), ("yy", dy * dy), ("rx", r * dx), ("xx", dx * dx),
                          ("xy", dx * dy)):
                cols[f"{st}_{c}_{s}"] = cell(np.where(H, v, 0.0))
            if K == 1:
                cols[f"ox_{c}_{s}"] = cell(np.where(~H, 2 * r * dx - dx * dx, 0.0))
                cols[f"oy_{c}_{s}"] = cell(np.where(~H, 2 * r * dy - dy * dy, 0.0))
    for k in KS:
        d = lv["T10"][k] - lv["T10"][k - 1]
        cols[f"ss_T10_{STEP[k]}"] = cell(np.where(H, d * d, 0.0))
    for nm in SSE_NAMES:
        for L in range(4):
            cols[f"sse_{nm}_L{L}"] = cell((y - pb - lv[nm][L]) ** 2)
    return pd.DataFrame({"day": [a for a, _ in uq], "airport": [b for _, b in uq], **cols})


class Draws:
    """Per-day H sums of one member draw of a bag comparison, the bags rebuilt by the bag rule."""

    def __init__(self, pan: dict, lv: dict, y: np.ndarray, days: np.ndarray, kinds):
        pf = pan[F]
        self.H = pf["H"]
        self.pbH = pf["pbase"][self.H]
        self.yH = y[self.H]
        self.dc = np.searchsorted(days, pf["day"][self.H])
        if not np.array_equal(days[self.dc], pf["day"][self.H]):
            raise SystemExit("an H row's day is not a July fold day")
        self.nd = len(days)
        self.M = {nm: [np.vstack([lv[f"{nm}{s}"][L][self.H] for s in SEEDS10 if f"{nm}{s}" in lv])
                       for L in range(4)] for nm in kinds}

    def bag(self, nm: str, idx, L: int) -> np.ndarray:
        return A.apply_floor(self.pbH, self.M[nm][L][np.asarray(idx) - 1].mean(0))

    def sums(self, X: str, Y: str, idx, levels=(0,)) -> dict:
        bx = [self.bag(X, idx, L) for L in range(4)]
        by = [self.bag(Y, idx, L) for L in range(4)]
        out = {}
        for k in KS:
            s = STEP[k]
            dx, dy = bx[k] - bx[k - 1], by[k] - by[k - 1]
            r = self.yH - self.pbH - bx[k - 1]
            for st, v in (("ry", r * dy), ("yy", dy * dy), ("rx", r * dx), ("xx", dx * dx),
                          ("xy", dx * dy)):
                out[f"{st}_{s}"] = np.bincount(self.dc, weights=v, minlength=self.nd)
        for L in levels:
            ey, ex = self.yH - self.pbH - by[L], self.yH - self.pbH - bx[L]
            out[f"dsse_L{L}"] = np.bincount(self.dc, weights=ey * ey - ex * ex, minlength=self.nd)
        return out


def draw_csv(cols: dict) -> str:
    return pd.DataFrame(cols).to_csv(index=False, float_format="%.17g")


def populations(pan: dict, lab: dict, lev: dict) -> dict:
    """COMMON's full fold and serving tables (SPEC 9.4)."""
    check_pin(C.E53 / "analysis/pin.json")
    qpin = json.loads((C.E53 / "analysis/pin.json").read_text())
    out = {}
    for f in C.FOLDS:
        ids, yy, pbf = pan[f]["ids"], lab[f]["y"], pan[f]["pbase"]
        nonm = LF.no_nm_mask(ids, f)
        if C.sha_arr(nonm) != qpin["folds"][f]["no_nm_sha256"]:
            raise SystemExit(f"{f}: the no-NM mask is not Q15's")
        masks = {"all": np.ones(len(ids), bool), "le3600": yy <= LF.CUT, "gt3600": yy > LF.CUT,
                 "no_nm": nonm, "clean": lab[f]["clean"]}
        masks |= {f"own_{g}": pan[f]["reg"] == g for g in C.REGIMES}
        masks |= {f"nm_{r_}": lab[f]["nmreg"] == r_ for r_ in sorted(set(lab[f]["nmreg"]))}
        preds = (["F3bag", f"{JARM}3bag", "M3bag"]
                 + [f"{nm}{i}" for nm in ("F", JARM, "M") for i in SEEDS3]
                 + ["P3bag", "C3bag", "S3bag", "P10bag", "C10bag"]) if f == F \
            else ["P3bag", "P10bag"]
        common5 = ("all", "le3600", "gt3600", "no_nm", "clean")
        out[f] = {"rows": {k: int(m.sum()) for k, m in masks.items()},
                  "rows_by_serving_regime": {g: {k: int((masks[k] & masks[f"own_{g}"]).sum())
                                                 for k in common5} for g in C.REGIMES},
                  "by_serving_regime": {}}
        for nm in preds:
            for Lv in range(4):
                e = yy - pbf - lev[f][nm][Lv]
                out[f][f"{nm}_L{Lv}"] = {k: fl(np.sqrt(np.mean(e[m] ** 2))) if m.any() else None
                                         for k, m in masks.items()}
                out[f]["by_serving_regime"][f"{nm}_L{Lv}"] = {
                    g: {k: fl(np.sqrt(np.mean(e[masks[k] & masks[f"own_{g}"]] ** 2)))
                        if (masks[k] & masks[f"own_{g}"]).any() else None for k in common5}
                    for g in C.REGIMES}
    return out


def cite_repro() -> dict:
    """SPEC 12.3: R2, R3, R4, R7, R8 and the leaf-complete population comparison are cited from
    analysis/repro.json, which ran them in full and refused on every mismatch."""
    if not REPRO.exists():
        raise SystemExit("analysis/repro.json does not exist; SPEC 6 and 7 come first")
    r = json.loads(REPRO.read_text())
    if r.get("source_check", {}).get("clean") is not True:
        raise SystemExit("analysis/repro.json does not record a clean source check")
    out = {"cited_record": str(REPRO.relative_to(ROOT)), "cited_record_sha256": sha_file(REPRO),
           "not_re_run_here": True,
           "why": "exp-055's two scripts, the 2,000-draw rebuilds of exp-057's and exp-058's draw "
                  "receipts and the 3,630-leaf population comparison dominate the non-fit "
                  "allocation; they passed at the tolerances below on the same bytes this "
                  "manifest pins, so they are carried over unchanged",
           "blocks": {}}
    checks = {
        "R2_exp051_gains": ("max_abs_diff", TOL_9),
        "R7_exp057": ("points_max_abs_diff_m", TOL_12),
        "R8_exp058": ("points_max_abs_diff_m_cos", TOL_12),
    }
    for blk, (key, tol) in checks.items():
        if blk not in r:
            raise SystemExit(f"repro.json has no {blk} block")
        if r[blk][key] > tol:
            raise SystemExit(f"repro.json {blk}.{key} is {r[blk][key]:.3e} > {tol:.0e}")
        out["blocks"][blk] = {key: r[blk][key], "tolerance": tol}
    r34 = r["R3_R4_exp055_scripts"]
    if not r34["registered_fields"]["all_within_tolerance"]:
        raise SystemExit("repro.json's R3/R4 registered fields did not pass")
    for sub, run in r34["runs"].items():
        w = run["whole_record"]
        if w["n_structural_diffs"] or w["float_fields_differing"] or not run["receipt_hashes_equal"]:
            raise SystemExit(f"repro.json's R3/R4 {sub} did not pass")
    out["blocks"]["R3_R4_exp055_scripts"] = {
        "registered_fields_n": r34["registered_fields"]["n"],
        "scripts": {k: {"whole_record_clean": True, "receipt_hashes_equal": True,
                        "oof_equal": v["oof_equal"]} for k, v in r34["runs"].items()}}
    out["blocks"]["R8_exp058"]["own_value_max_abs_diff"] = \
        r["R8_exp058"]["own_value_max_abs_diff"]
    out["blocks"]["R8_exp058"]["paired_intervals_max_abs_diff"] = \
        r["R8_exp058"]["paired_intervals_max_abs_diff"]
    out["blocks"]["R8_exp058"]["draw_receipts_bytes_identical"] = {
        k: v["bytes_identical"] for k, v in r["R8_exp058"]["draw_receipts"].items()}
    for k_, v_ in (("own_value_max_abs_diff", TOL_9),
                   ("paired_intervals_max_abs_diff", TOL_9)):
        if out["blocks"]["R8_exp058"][k_] > v_:
            raise SystemExit(f"repro.json R8 {k_} is {out['blocks']['R8_exp058'][k_]:.3e}")
    out["blocks"]["R8_vii_populations"] = r["R8_vii_populations"]
    out["blocks"]["separate_reproductions"] = {
        "B1_grid_rows_verified": list(r["separate_reproductions"]["B1_grid"]["verified_rows"]),
        "B2_anchors": {k: v for k, v in r["separate_reproductions"]["B2_anchors"].items()
                       if k != "why_V_at_0_is_an_identity"},
        "B3_february": {k: v for k, v in r["separate_reproductions"]["B3_february"].items()
                        if k != "band_rule"},
        "B4_max_abs_diff_s2_row":
            r["separate_reproductions"]["B4_exp059_fitted_V"]["max_abs_diff_s2_row"],
        "B5_day_cells": r["separate_reproductions"]["B5_day_cells"]}
    out["exp057_K10_day_plus_member_interval_cited"] = (
        "exp-057's K = 10 day-plus-member m intervals are cited from its measure.json; the "
        "2,000-draw ten-member loop is not run again here (SPEC 12.3)")
    return out


# ------------------------------------------------------------------------------ main ------------

def main() -> None:
    t0, c0 = time.time(), time.process_time()
    np.seterr(divide="ignore", invalid="ignore")
    outs = [OUT, REC_DIR / "daysums-july_loo.csv", REC_DIR / "days-july_loo.csv",
            *(REC_DIR / f"draws-L{L}.csv" for L in BLOCKS),
            *(OOFD / f"ownvalue-{f}.parquet" for f in C.FOLDS),
            OOFD / "members-july_loo.parquet", ROWD / "rows-july_loo.parquet"]
    NO.refuse_existing(outs, overwrite=False)
    plan1 = json.loads((ANALYSIS / "plan.json").read_text())
    plan2 = json.loads((ANALYSIS / "plan2.json").read_text()) if not STANDIN else plan1
    if not STANDIN and plan2.get("june_star", {}).get("k") != JK:
        raise SystemExit("analysis/plan2.json is not the plan of the selected June*")
    rec = {"what": "exp-062 ownvalue-refine: the conditional completion and the fixed reading",
           "standin": STANDIN, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "manifest_sha256": sha_file(MANIFEST),
           "spec_commit": MAN["registration"]["spec_commit"],
           "probe_sha256": sha_file(PROBE), "plan_sha256": sha_file(ANALYSIS / "plan.json"),
           "plan2_sha256": (None if STANDIN else sha_file(ANALYSIS / "plan2.json")),
           "june_star": JS, "B": B, "blocks": list(BLOCKS), "margin": MARGIN,
           "generator": "np.random.default_rng(5500 + L)",
           "draw_order": ["july_loo fold starts", "2026-07 ranking starts", "january fold starts",
                          "2026-01 ranking starts", "december fold starts",
                          "K = 3 member triples (B x 3)", "K = 10 member-index vectors (B x 10)"],
           "arms": {c: v[3] for c, v in COMPS.items()},
           "q_is_a_pinned_comparator":
               "q is Q15's pinned registered constant; the point m - q and m / q use that constant, "
               "and the interval route uses exp-055's exact per-draw q from the two pinned "
               "receipts, which is exp-058's registered interval method and not a rebuild of the "
               "pinned point.  This experiment makes no new claim about q.",
           "units": {"m": "ratio", "loss": "s2 per July fold row (n = 190,713)", "rmse": "s"},
           "checks": {}}
    if STANDIN:
        rec["standin_note"] = (
            "THE REGISTERED DRY RUN, NOT A RESULT.  June*'s twelve vectors are exp-058's matched "
            "June arm, the draw count is 50 instead of 2,000, only block length 3 runs, every "
            "output goes to the session scratch directory, and no artifacts/exp-062/work file is "
            "opened.  Every draw-count-dependent comparison is skipped and recorded as skipped.")
        rec["skipped_in_the_dry_run"] = [
            "the comparison of this run's comparator m intervals with exp-057's (2,000 draws)",
            "the comparison of the paired feb - june_matched intervals with exp-058's (2,000 draws)",
            "the interval width of every band (50 draws, not 2,000)"]

    # ---- the cited reproductions ------------------------------------------------------------
    rec["cited_reproductions"] = cite_repro()
    print(f"R2, R3, R4, R7, R8 and the 3,630-leaf population comparison cited from "
          f"{rec['cited_reproductions']['cited_record']} "
          f"({rec['cited_reproductions']['cited_record_sha256'][:16]})", flush=True)

    # ---- R1, R5, R6 -------------------------------------------------------------------------
    pan, vec, pin = V.load()
    lab = {f: LF.load_labels(f, pan[f]["ids"]) for f in C.FOLDS}
    rec["checks"]["R1_originvec_load"] = True
    check_pin(E55 / "analysis/screen.json")
    scr = json.loads((E55 / "analysis/screen.json").read_text())
    lfj = json.loads((E55 / "analysis/lossfals.json").read_text())
    check_pin(E55 / "analysis/lossfals.json")
    days = np.sort(np.unique(pan[F]["day"]))
    if len(days) != 31:
        raise SystemExit("July does not have 31 fold days")
    draws = {}
    for L in BLOCKS:
        rng = np.random.default_rng(5500 + L)
        st = {key: C.mbb_starts(rng, 31, L, B)
              for key in ("july_loo", "2026-07", "january", "2026-01", "december")}
        sig = rng.integers(1, 4, size=(B, 3))
        sig10 = rng.integers(1, 11, size=(B, 10))
        msi = np.array([MULTI3.index(tuple(sorted(t))) for t in sig.tolist()])
        if not STANDIN:
            want = scr["draw_sha256"][f"L{L}"]
            got = {k_: C.sha_arr(x_) for k_, x_ in st.items()} | {"seeds": C.sha_arr(sig)}
            if got != {k_: x_ for k_, x_ in want.items() if k_ != "multiset_counts"}:
                raise SystemExit(f"R5 L{L}: the regenerated draws are not exp-055's")
        draws[L] = {"st": st, "msi": msi, "sig10": np.sort(sig10, axis=1)}
    rec["checks"]["R5_draws_equal_exp055"] = not STANDIN
    for f in C.FOLDS:
        for s in range(4, 11):
            for L in range(4):
                p = C.member_path(L, s)
                check_pin(p)
                vec[f][f"P{s}_L{L}"] = np.load(p)[f].astype("float64")
    rec["exp057_vectors"] = load_exp057(vec)
    rec["exp058_vectors"] = load_exp058(vec)
    rec["june_star_vectors"] = load_june_star(vec, plan1, plan2)
    lev = levels(pan, vec)
    L7 = lev[F]
    worst10 = 0.0
    for f in C.FOLDS:
        for L in range(4):
            worst10 = max(worst10, float(np.max(np.abs(lev[f]["P10bag"][L] - vec[f][f"T10_L{L}"]))))
    if worst10 > TOL_9:
        raise SystemExit(f"R6: the plain K = 10 bag differs from Q15's heads by {worst10:.3e} s")
    rec["checks"]["R6_plain_K10_vs_heads_max_abs_diff_s"] = worst10
    rec["off_july"] = ("every carrier fit here is July-only (exp-051's design, "
                       "folds=('july_loo',)): on december and january each June*, February and "
                       "matched-June member equals the plain member of the same seed and every "
                       "K = 3 bag equals P3bag, so those folds are reported once as P3bag; the "
                       "equality is asserted by construction in levels()")
    print(f"R1, R5 and R6 pass; June*'s twelve vectors loaded "
          f"({'STAND-INS' if STANDIN else 'sha-checked against their fit records'})", flush=True)

    # ---- the receipt and the day matrix -----------------------------------------------------
    pf = pan[F]
    H, y, pb = pf["H"], lab[F]["y"], pf["pbase"]
    n_all, nH = len(y), int(H.sum())
    text, back = LF.receipt("july_loo", cells_july(pan, L7, y))
    Dm, ci = LF.day_matrix(back, days)
    P_ = Dm.sum(0)
    if int(P_[ci["n"]]) != n_all or int(P_[ci["n_H"]]) != nH or nH != C.H_ROWS[F]:
        raise SystemExit("the receipt does not cover every July row")
    dtext = pd.DataFrame(Dm, columns=list(ci)).assign(day=days)[["day"] + list(ci)].to_csv(
        index=False, float_format="%.17g")
    check_pin(E55 / "analysis/receipts/i2-daysums-july_loo.csv")
    lf_cells = LF.read_receipt(E55 / "analysis/receipts/i2-daysums-july_loo.csv")
    worst_cell = 0.0
    for s in SP:
        for st in STATS:
            a_ = back[f"{st}_carrier_K3_{s}"].to_numpy("float64")
            b_ = lf_cells[f"{st}_bag123_{s}"].to_numpy("float64")
            worst_cell = max(worst_cell, float(np.max(np.abs(a_ - b_))))
    rec["checks"]["carrier_K3_cells_vs_exp055_i2_receipt_max_abs_diff"] = worst_cell
    if worst_cell > TOL_9:
        raise SystemExit(f"the carrier K = 3 cells differ from exp-055's by {worst_cell:.3e}")
    rpath = E55 / "analysis/receipts/i1-daysums-rank-2026-07.csv"
    check_pin(rpath)
    Dr, cir = SC.day_matrix(SC.read_receipt(rpath),
                            np.array([f"2026-07-{d:02d}" for d in range(1, 32)]))
    for s in SP:
        qpt = float(np.sqrt((Dr.sum(0)[cir[f"ss_UP_{s}"]] / Dr.sum(0)[cir["n"]])
                            / (P_[ci[f"ss_T10_{s}"]] / P_[ci["n_H"]])))
        if abs(qpt - Q_POINT[s]) > TOL_12:
            raise SystemExit(f"{s}: q from the two receipts {qpt!r} is not Q15's {Q_POINT[s]!r}")
    rec["checks"]["q_from_two_receipts_equals_Q15_point"] = True

    # ---- the points -------------------------------------------------------------------------
    rec["points"] = {}
    w_direct, w_split, w_sse = 0.0, 0.0, 0.0
    for c, (X, Y, K, role) in COMPS.items():
        ent = {"carrier": X, "plain": Y, "K": K, "role": role, "steps": {}}
        Is, Js = [], []
        for k in KS:
            s = STEP[k]
            q = {a: fl(b) for a, b in q_cols(P_, ci, c, s).items()}
            dx, dy = L7[X][k] - L7[X][k - 1], L7[Y][k] - L7[Y][k - 1]
            direct_m = float(np.sqrt(np.sum(dx[H] ** 2) / np.sum(dy[H] ** 2)))
            w_direct = max(w_direct, abs(direct_m - q["m"]) / direct_m)
            r = y - pb - L7[X][k - 1]
            sse = float((np.sum(r ** 2) - np.sum((r - dx) ** 2)) / n_all)
            out_H = float(P_[ci[f"ox_{c}_{s}"]] / n_all) if K == 1 else 0.0
            w_sse = max(w_sse, abs(q["g_C"] + out_H - sse))
            w_split = max(w_split,
                          abs((q["g_C"] - q["g_plain"]) - (q["amplitude"] + q["alignment"])),
                          abs(q["alignment"] - (q["alignment_direction"]
                                                + q["alignment_orthogonal"])),
                          abs(q["g_hat"] - q["g_plain"] - q["amplitude"]))
            ry_ = y - pb - L7[Y][k - 1]
            q |= {"outside_H_g_C": out_H,
                  "outside_H_plain_step_gain": (float(P_[ci[f"oy_{c}_{s}"]] / n_all) if K == 1
                                                else 0.0),
                  "direct_sse_gain_all_rows": sse,
                  "g_P_own_p0_all_rows": float(np.sum(2 * ry_ * dy - dy * dy) / n_all)}
            q["g_P_minus_g_plain"] = q["g_P_own_p0_all_rows"] - q["g_plain"]
            q["J_positive"] = bool(q["J"] > 0)
            q["J_positive_band_for_m"] = [2 * q["A"] / q["B"] - 1, 1.0]
            q |= {"q": Q_POINT[s], "I": float(imp(q["m"], Q_POINT[s])),
                  "m_minus_q": q["m"] - Q_POINT[s], "m_over_q": q["m"] / Q_POINT[s],
                  "I_positive_band": [2 * Q_POINT[s] - 1, 1.0]}
            q["changed_rows"] = {nm: {g: int(((d != 0) & (pf["reg"] == g)).sum())
                                      for g in C.REGIMES}
                                 for nm, d in (("carrier", dx), ("plain", dy))}
            q["changed_rows_total"] = {nm: int((d != 0).sum())
                                       for nm, d in (("carrier", dx), ("plain", dy))}
            q["outside_H_share_of_step_mass"] = {
                nm: {g: float(np.sum(d[pf["reg"] == g] ** 2) / np.sum(d ** 2))
                     for g in ("G", "A", "C")} for nm, d in (("carrier", dx), ("plain", dy))}
            q["kish_days_mass"] = {"carrier": C.kish(Dm[:, ci[f"xx_{c}_{s}"]]),
                                   "plain": C.kish(Dm[:, ci[f"yy_{c}_{s}"]])}
            q["intervals"] = {}
            Is.append(q["I"])
            Js.append(q["J"])
            ent["steps"][s] = q
        ent["mean_I"] = {"point": float(np.mean(Is)), "intervals": {}}
        ent["mean_J"] = {"point": float(np.mean(Js)), "intervals": {}}
        rec["points"][c] = ent
    rec["points_fields_reported"] = list(R8_III_FIELDS)
    rec["checks"] |= {"m_receipt_vs_direct_max_rel": w_direct,
                      "split_identities_max_abs_diff_s2": w_split,
                      "direct_sse_reconciliation_max_abs_diff_s2": w_sse,
                      "kish_days_rows_H": C.kish(Dm[:, ci["n_H"]])}
    if w_direct > TOL_12 or w_split > TOL_9 or w_sse > 1e-6:
        raise SystemExit(f"an identity fails: m {w_direct:.2e}, split {w_split:.2e}, "
                         f"sse {w_sse:.2e}")
    #: the comparators this receipt shares with exp-055, exp-057 and exp-058 must come out equal
    wpt = 0.0
    for c, pt in (("carrier_K3", "bag123"), ("carrier_s1", "seed1"), ("carrier_s2", "seed2"),
                  ("carrier_s3", "seed3"), ("sham_s1", "sham")):
        for s in SP:
            for key in ("A", "B", "m", "g_C", "g_hat", "g_plain", "J"):
                wpt = max(wpt, abs(rec["points"][c]["steps"][s][key]
                                   - lfj["points"][F][pt]["steps"][s][key]))
    check_pin(E57 / "analysis/measure.json")
    e57 = json.loads((E57 / "analysis/measure.json").read_text())
    check_pin(E58 / "analysis/measure.json")
    e58 = json.loads((E58 / "analysis/measure.json").read_text())
    if e57["spec_commit"] != E57_SPEC_COMMIT or e57["standin"] \
            or e58["spec_commit"] != E58_SPEC_COMMIT or e58["standin"]:
        raise SystemExit("exp-057's or exp-058's measure.json is not its registered official record")
    w57, w58 = 0.0, 0.0
    for c in ("sham_K3", "carrier_K3", "carrier_K10", "sham_s1", "sham_s2", "sham_s3"):
        for s in SP:
            for key in ("A", "B", "m", "g_C", "g_hat", "g_plain", "J", "cos"):
                w57 = max(w57, abs(rec["points"][c]["steps"][s][key]
                                   - e57["points"][c]["steps"][s][key]))
    for c in list(BAG_COMPS) + [f"{nm}_s{i}" for nm in ("feb", "june_matched") for i in SEEDS3]:
        if c == "june_star_K3":
            continue
        for s in SP:
            for key in R8_III_FIELDS:
                w58 = max(w58, abs(rec["points"][c]["steps"][s][key]
                                   - e58["points"][c]["steps"][s][key]))
    rec["checks"] |= {"exp055_points_from_this_receipt_max_abs_diff": wpt,
                      "exp057_points_from_this_receipt_max_abs_diff": w57,
                      "exp058_points_from_this_receipt_max_abs_diff": w58}
    if max(wpt, w57, w58) > TOL_9:
        raise SystemExit(f"the receipt route misses a recorded point: exp-055 {wpt:.3e}, "
                         f"exp-057 {w57:.3e}, exp-058 {w58:.3e}")

    # ---- the paired differences -------------------------------------------------------------
    rec["paired_differences"] = {
        nm: {s: {"m_X": rec["points"][X]["steps"][s]["m"], "m_Y": rec["points"][Y]["steps"][s]["m"],
                 "difference": rec["points"][X]["steps"][s]["m"] - rec["points"][Y]["steps"][s]["m"],
                 "intervals": {}} for s in SP} for nm, (X, Y) in DIFFS.items()}
    rec["feb_minus_june_star"] = rec["paired_differences"]["feb_minus_june_star"]
    rec["difference_by_member"] = {
        nm: {f"s{i}": {s: rec["points"][f"{xa}_s{i}"]["steps"][s]["m"]
                          - rec["points"][f"{xb}_s{i}"]["steps"][s]["m"] for s in SP}
             for i in SEEDS3} for nm, (xa, xb) in DIFF_MEMBER_X.items()}
    rec["raw_members"] = {
        nm: {f"s{i}": {s: {k_: rec["points"][f"{nm}_s{i}"]["steps"][s][k_]
                           for k_ in R8_III_FIELDS} for s in SP} for i in SEEDS3}
        for nm in ("feb", "june_star", "june_matched", "carrier", "sham")}

    # ---- each arm's own value at L0 ---------------------------------------------------------
    own = {"definition": "[SSE(P3_L0) - SSE(X_L0)] / n on every July fold row (positive: the arm's "
                         "own L0 vector predicts July better than the plain K = 3 bag); measured, "
                         "never subtracted",
           "route": "cell to day to total, exp-058's route", "by_arm": {}}
    for nm, X in (("feb_K3", "F3bag"), ("june_star_K3", f"{JARM}3bag"),
                  ("june_matched_K3", "M3bag"), ("carrier_K3", "C3bag"), ("sham_K3", "S3bag"),
                  ("carrier_K10", "C10bag")):
        Y = "P10bag" if nm == "carrier_K10" else "P3bag"
        own["by_arm"][nm] = {"X": X, "Y": Y,
                             "value": fl((P_[ci[f"sse_{Y}_L0"]] - P_[ci[f"sse_{X}_L0"]]) / n_all),
                             "intervals": {}}
    own["by_arm"]["june_star_K3"] |= {"k": JK, "N": JN}
    for nm, X in (("feb", "F"), ("june_star", JARM), ("june_matched", "M"), ("carrier", "C"),
                  ("sham", "S")):
        for i in SEEDS3:
            own["by_arm"][f"{nm}_s{i}"] = {
                "X": f"{X}{i}", "Y": f"P{i}",
                "value": fl((P_[ci[f"sse_P{i}_L0"]] - P_[ci[f"sse_{X}{i}_L0"]]) / n_all),
                "intervals": {}}
    w_own = 0.0
    for nm, ref in E58_OWN.items():
        w_own = max(w_own, abs(own["by_arm"][nm]["value"] - ref))
    rec["checks"]["own_value_L0_vs_exp058_max_abs_diff"] = w_own
    if w_own > TOL_9:
        raise SystemExit(f"the L0 own values miss exp-058's by {w_own:.3e} s2/row")
    #: SPEC 9.2: June*'s own value here must be the value the probe's selection read.  SPEC 11's
    #: dry run substitutes exp-058's matched June arm for June*'s twelve vectors, so there the same
    #: check runs against that arm -- which the stand-in IS -- and the own-value route is still
    #: exercised end to end.
    if STANDIN:
        wj = abs(own["by_arm"]["june_star_K3"]["value"]
                 - own["by_arm"]["june_matched_K3"]["value"])
        for i in SEEDS3:
            wj = max(wj, abs(own["by_arm"][f"june_star_s{i}"]["value"]
                             - own["by_arm"][f"june_matched_s{i}"]["value"]))
        rec["checks"]["standin_june_star_own_value_vs_matched_june_max_abs_diff_s2_row"] = wj
        if wj > TOL_12:
            raise SystemExit(f"the dry run's stand-in June* differs from exp-058's matched June "
                             f"arm by {wj:.3e}: the stand-in is not that arm")
    else:
        wj = abs(own["by_arm"]["june_star_K3"]["value"] - JV)
        for i in SEEDS3:
            wj = max(wj, abs(own["by_arm"][f"june_star_s{i}"]["value"]
                             - PR["arms"][JARM]["members"][f"s{i}"]["value"]))
        rec["checks"]["june_star_own_value_vs_probe_max_abs_diff_s2_row"] = wj
        if wj > TOL_12:
            raise SystemExit(f"June*'s own value differs from analysis/probe.json by {wj:.3e}")
    own["band_B"] = list(BAND)
    own["V_star_february"] = V_STAR
    own["june_star_in_band"] = bool(BAND[0] <= own["by_arm"]["june_star_K3"]["value"] <= BAND[1])
    if STANDIN:
        #: the stand-in is exp-058's matched June arm (+493.2541), which is outside B by
        #: construction -- only the selected June* is inside it -- so the dry run records the flag
        #: and asserts the stand-in instead.  A stand-in that was silently the selected arm, or any
        #: other arm, is still caught here.
        own["standin_is_exp058_matched_june_outside_B"] = not own["june_star_in_band"]
        if own["june_star_in_band"] or abs(own["by_arm"]["june_star_K3"]["value"] - V_TOP) > TOL_9:
            raise SystemExit(f"the dry run's stand-in own value "
                             f"{own['by_arm']['june_star_K3']['value']:.10f} is not exp-058's "
                             f"matched June value {V_TOP:.10f} outside B")
    elif not own["june_star_in_band"]:
        raise SystemExit("June*'s own value is not inside the frozen band B")
    rec["own_value_L0"] = own

    # ---- the decompositions -----------------------------------------------------------------
    g_air = back.groupby("airport").sum(numeric_only=True)
    qa = scr["decompositions"]["airport_q"]["july_loo->2026-07"]
    dec = {"airport": {}, "nm_regime": {}, "serving_regime": {}}
    w_dec = 0.0
    for c, (X, Y, K, _) in COMPS.items():
        dec["airport"][c], dec["nm_regime"][c], dec["serving_regime"][c] = {}, {}, {}
        for k in KS:
            s = STEP[k]
            sx, sy = g_air[f"xx_{c}_{s}"], g_air[f"yy_{c}_{s}"]
            SX, SY = P_[ci[f"xx_{c}_{s}"]], P_[ci[f"yy_{c}_{s}"]]
            contrib = (sx - sy) / SY
            w_dec = max(w_dec, abs(contrib.sum() - (SX / SY - 1)))
            dec["airport"][c][s] = {
                "m2_minus_1": fl(SX / SY - 1), "sum_of_contributions": fl(contrib.sum()),
                "by_airport": {a: {"H_rows": int(g_air.loc[a, "n_H"]),
                                   "contribution_to_m2_minus_1": fl(contrib[a]),
                                   "plain_mass_share": fl(sy[a] / SY),
                                   "m_airport": (fl(np.sqrt(sx[a] / sy[a])) if sy[a] > 0 else None),
                                   "q_airport_exp055":
                                       qa[s]["by_airport"].get(a, {}).get("q_airport")}
                               for a in g_air.index}}
            dx, dy = L7[X][k] - L7[X][k - 1], L7[Y][k] - L7[Y][k - 1]
            by, tot = {}, 0.0
            for rg in C.NM_REGIMES:
                sel = H & (pf["nm"] == rg)
                a_, b_ = float(np.sum(dx[sel] ** 2)), float(np.sum(dy[sel] ** 2))
                by[rg] = {"H_rows": int(sel.sum()), "contribution_to_m2_minus_1": (a_ - b_) / SY,
                          "plain_mass_share": b_ / SY,
                          "m_regime": float(np.sqrt(a_ / b_)) if b_ > 0 else None}
                tot += (a_ - b_) / SY
            if sum(v_["H_rows"] for v_ in by.values()) != nH:
                raise SystemExit("the NM regimes do not cover H")
            w_dec = max(w_dec, abs(tot - (SX / SY - 1)))
            dec["nm_regime"][c][s] = {"m2_minus_1": fl(SX / SY - 1), "sum_of_contributions": tot,
                                      "by_regime": by}
            srv, tx, ty = {}, float(np.sum(dx ** 2)), float(np.sum(dy ** 2))
            for g in C.REGIMES:
                sel = pf["reg"] == g
                srv[g] = {"rows": int(sel.sum()),
                          "changed_rows_carrier": int((dx[sel] != 0).sum()),
                          "changed_rows_plain": int((dy[sel] != 0).sum()),
                          "carrier_step_mass_share": (float(np.sum(dx[sel] ** 2) / tx)
                                                      if tx > 0 else None),
                          "plain_step_mass_share": (float(np.sum(dy[sel] ** 2) / ty)
                                                    if ty > 0 else None)}
            dec["serving_regime"][c][s] = {
                "all_carrier_mass_in_H": bool(abs(srv["H"]["carrier_step_mass_share"] - 1) < 1e-12),
                "by_regime": srv}
    rec["checks"]["decomposition_max_reconciliation_error"] = w_dec
    if w_dec > TOL_9:
        raise SystemExit(f"a decomposition does not reconcile: {w_dec:.3e}")
    rec["decompositions"] = dec

    # ---- the fit-count record ---------------------------------------------------------------
    names = ([f"P{i}" for i in SEEDS10]
             + [f"{nm}{i}" for nm in ("F", JARM, "M", "C", "S") for i in SEEDS3]
             + [f"C{i}" for i in range(4, 11)]
             + ["P3bag", "F3bag", f"{JARM}3bag", "M3bag", "C3bag", "S3bag", "P10bag", "C10bag",
                "T10"])
    fc = {}
    for k in KS:
        s = STEP[k]
        stp = {nm: (L7[nm][k] - L7[nm][k - 1])[H] for nm in names}
        d10 = stp["P10bag"]
        fc[s] = {"rms_H_step_s": {nm: float(np.sqrt(np.sum(stp[nm] ** 2) / nH)) for nm in names},
                 "m": {"feb_K1": {f"s{i}": rec["points"][f"feb_s{i}"]["steps"][s]["m"]
                                  for i in SEEDS3},
                       "feb_K3": rec["points"]["feb_K3"]["steps"][s]["m"],
                       "june_star_K1": {f"s{i}": rec["points"][f"june_star_s{i}"]["steps"][s]["m"]
                                        for i in SEEDS3},
                       "june_star_K3": rec["points"]["june_star_K3"]["steps"][s]["m"],
                       "june_matched_K1": {f"s{i}":
                                           rec["points"][f"june_matched_s{i}"]["steps"][s]["m"]
                                           for i in SEEDS3},
                       "june_matched_K3": rec["points"]["june_matched_K3"]["steps"][s]["m"],
                       "full_june_K3": rec["points"]["carrier_K3"]["steps"][s]["m"],
                       "full_june_K10": rec["points"]["carrier_K10"]["steps"][s]["m"],
                       "sham_K3": rec["points"]["sham_K3"]["steps"][s]["m"]},
                 "cos_with_K10_plain_step": {
                     nm: float(np.sum(stp[nm] * d10)
                               / np.sqrt(np.sum(stp[nm] ** 2) * np.sum(d10 ** 2)))
                     for nm in names if nm != "P10bag"}}
    rec["fit_count_record_july_H"] = fc

    # ---- the intervals ----------------------------------------------------------------------
    DR = Draws(pan, L7, y, days, kinds=("P", "C", "S", "F", "M", JARM))
    for idx, X, want in (((1, 2, 3), "C", "C3bag"), ((1, 2, 3), "S", "S3bag"),
                         ((1, 2, 3), "F", "F3bag"), ((1, 2, 3), "M", "M3bag"),
                         ((1, 2, 3), JARM, f"{JARM}3bag"), (SEEDS10, "C", "C10bag"),
                         (SEEDS10, "P", "P10bag"), ((1, 2, 3), "P", "P3bag")):
        for L in range(4):
            if not np.array_equal(DR.bag(X, idx, L), L7[want][L][H]):
                raise SystemExit(f"the draw route's bag {want} L{L} is not the point bag on H")
    rec["checks"]["draw_route_bags_equal_point_bags_on_H"] = True
    k3 = {j: {c: DR.sums(K3_DRAW[c], "P", t, levels=range(4) if c == "sham_K3" else (0,))
              for c in K3_DRAW} for j, t in enumerate(MULTI3)}
    nday, nHday = Dm[:, ci["n"]], Dm[:, ci["n_H"]]
    IV = ("m", "J", "A", "B", "g_C", "g_hat", "g_plain", "amplitude", "alignment", "cos")
    dr_csv, t_draw = {}, time.time()
    for L in BLOCKS:
        d = draws[L]
        W = C.mbb_weights(d["st"]["july_loo"], 31, L)
        Wr = C.mbb_weights(d["st"]["2026-07"], 31, L)
        S_all = W @ Dm
        wn, wnH = W @ nday, W @ nHday
        qd = {s: np.sqrt(((Wr @ Dr[:, cir[f"ss_UP_{s}"]]) / (Wr @ Dr[:, cir["n"]]))
                         / ((W @ Dm[:, ci[f"ss_T10_{s}"]]) / wnH)) for s in SP}

        def put(c, mode, q):
            for s in SP:
                o = {k_: interval(q[s][k_]) for k_ in IV}
                o |= {"I": interval(imp(q[s]["m"], qd[s])),
                      "m_minus_q": interval(q[s]["m"] - qd[s]),
                      "m_over_q": interval(q[s]["m"] / qd[s])}
                rec["points"][c]["steps"][s]["intervals"].setdefault(f"L{L}", {})[mode] = o
            rec["points"][c]["mean_I"]["intervals"].setdefault(f"L{L}", {})[mode] = interval(
                np.mean([imp(q[s]["m"], qd[s]) for s in SP], 0))
            rec["points"][c]["mean_J"]["intervals"].setdefault(f"L{L}", {})[mode] = interval(
                np.mean([q[s]["J"] for s in SP], 0))
        for c in COMPS:
            put(c, "day", {s: q_cols(S_all, ci, c, s) for s in SP})
        for nm, e in own["by_arm"].items():
            e["intervals"].setdefault(f"L{L}", {})["day"] = interval(
                (S_all[:, ci[f"sse_{e['Y']}_L0"]] - S_all[:, ci[f"sse_{e['X']}_L0"]]) / wn)
        for nm, (X, Y) in DIFFS.items():
            for s in SP:
                rec["paired_differences"][nm][s]["intervals"].setdefault(f"L{L}", {})["day"] = \
                    interval(q_cols(S_all, ci, X, s)["m"] - q_cols(S_all, ci, Y, s)["m"])
        per = {c: {key: np.stack([W @ k3[j][c][key] for j in range(len(MULTI3))],
                                 axis=1)[np.arange(B), d["msi"]] for key in k3[0][c]}
               for c in K3_DRAW}
        cols = {"draw": np.arange(B), "k3_multiset": [MS3[j] for j in d["msi"]],
                "k10_indices": [" ".join(map(str, r_)) for r_ in d["sig10"]], "wn": wn,
                "wn_H": wnH}
        for s in SP:
            cols[f"q_{s}"] = qd[s]
        for c in BAG_COMPS:
            put(c, "day+member",
                {s: quantities(wn, wnH, *(per[c][f"{st}_{s}"] for st in STATS)) for s in SP})
            for s in SP:
                for st in STATS:
                    cols[f"w{st}_{c}_{s}"] = per[c][f"{st}_{s}"]
        for c in BAG_COMPS:
            for Lv in (range(4) if c == "sham_K3" else (0,)):
                cols[f"wdsse_{c}_L{Lv}"] = per[c][f"dsse_L{Lv}"]
                if Lv == 0:
                    own["by_arm"][c]["intervals"][f"L{L}"]["day+member"] = interval(
                        per[c]["dsse_L0"] / wn)
        for nm, (X, Y) in DIFFS.items():
            for s in SP:
                mX = np.sqrt(per[X][f"xx_{s}"] / per[X][f"yy_{s}"])
                mY = np.sqrt(per[Y][f"xx_{s}"] / per[Y][f"yy_{s}"])
                rec["paired_differences"][nm][s]["intervals"][f"L{L}"]["day+member"] = \
                    interval(mX - mY)
        dr_csv[L] = draw_csv(cols)
        print(f"intervals L{L} done ({time.time() - t_draw:.0f} s)", flush=True)
    rec["carrier_K10_day_plus_member_cited_from_exp057"] = {
        s: {f"L{L}": e57["points"]["carrier_K10"]["steps"][s]["intervals"][f"L{L}"]["day+member"]["m"]
            for L in (1, 2, 3)} for s in SP} if not STANDIN else None
    if not STANDIN:
        iw = 0.0
        for L in BLOCKS:
            for mode in ("day", "day+member"):
                for c in ("carrier_K3", "sham_K3"):
                    for s in SP:
                        a_ = rec["points"][c]["steps"][s]["intervals"][f"L{L}"][mode]["m"]
                        b_ = e57["points"][c]["steps"][s]["intervals"][f"L{L}"][mode]["m"]
                        iw = max(iw, abs(a_["lo"] - b_["lo"]), abs(a_["hi"] - b_["hi"]))
            for c in ("carrier_K10",):
                for s in SP:
                    a_ = rec["points"][c]["steps"][s]["intervals"][f"L{L}"]["day"]["m"]
                    b_ = e57["points"][c]["steps"][s]["intervals"][f"L{L}"]["day"]["m"]
                    iw = max(iw, abs(a_["lo"] - b_["lo"]), abs(a_["hi"] - b_["hi"]))
        rec["checks"]["comparator_m_intervals_vs_exp057_max_abs_diff"] = iw
        if iw > TOL_9:
            raise SystemExit(f"the comparators' m intervals miss exp-057's by {iw:.3e}")
        iw58 = 0.0
        for L in BLOCKS:
            for mode in ("day", "day+member"):
                for s in SP:
                    a_ = rec["paired_differences"]["feb_minus_june_matched"][s]["intervals"][
                        f"L{L}"][mode]
                    b_ = e58["feb_minus_june_matched"][s]["intervals"][f"L{L}"][mode]
                    iw58 = max(iw58, abs(a_["lo"] - b_["lo"]), abs(a_["hi"] - b_["hi"]))
        rec["checks"]["feb_minus_june_matched_intervals_vs_exp058_max_abs_diff"] = iw58
        if iw58 > TOL_9:
            raise SystemExit(f"the paired February - matched June intervals miss exp-058's "
                             f"by {iw58:.3e}")

    # ---- the fixed reading ------------------------------------------------------------------
    pF = rec["points"]["feb_K3"]["steps"]
    pJ = rec["points"]["june_star_K3"]["steps"]
    dif = rec["paired_differences"]["feb_minus_june_star"]
    bands = {s: dif[s]["intervals"]["L3"]["day+member"] for s in READ_STEPS}
    inside = {s: bool(-MARGIN <= bands[s]["lo"] and bands[s]["hi"] <= MARGIN) for s in READ_STEPS}
    above = {s: bool(bands[s]["lo"] > MARGIN) for s in READ_STEPS}
    if all(inside.values()):
        verdict = "curve-consistent"
        reading = (
            "curve-consistent: both complete three-day day-plus-member bands of m_Feb - m_June* at "
            "v20 and v21 lie inside [-0.03, +0.03].  This means only that the February point adds "
            "no visible amplitude evidence beyond the registered June block-size / own-value / "
            "amplitude curve, and it can close only this tested origin-screen form, provisionally.  "
            "It is not a causal equivalence test and it does not isolate season: information at "
            "L1-L3 and the per-airport composition of the two blocks remain unmatched.")
    elif all(above.values()):
        verdict = "residual positive separation"
        reading = (
            "residual positive separation: both lower endpoints of the three-day day-plus-member "
            "bands of m_Feb - m_June* at v20 and v21 are greater than +0.03.  This does not "
            "establish season, because later-level information and the per-airport composition "
            "remain unmatched; the difference can be any of them.  The tested origin-screen form "
            "stays open.")
    else:
        verdict = "inconclusive"
        reading = (
            "inconclusive: at v20 or v21 the three-day day-plus-member band of m_Feb - m_June* "
            "neither lies wholly inside [-0.03, +0.03] nor has its lower endpoint above +0.03.  "
            "The February point is neither shown to lie on the registered June curve nor shown to "
            "sit above it by more than the practical margin, and the tested origin-screen form "
            "stays open.")
    rec["registered_reading"] = {
        "statement": "the reading uses the paired 2.5 % to 97.5 % three-day day-plus-member "
                     "variation bands of m_Feb - m_June* at v20 and v21, with the practical "
                     "amplitude-equivalence margin 0.03: curve-consistent if both complete bands "
                     "lie inside [-0.03, +0.03]; residual positive separation if both lower "
                     "endpoints are greater than +0.03; inconclusive otherwise",
        "margin": MARGIN, "steps": list(READ_STEPS), "june_star": JS,
        "bands_3day_day_plus_member": bands,
        "bands_all_block_lengths": {s: {f"L{L}": dif[s]["intervals"][f"L{L}"]["day+member"]
                                        for L in BLOCKS} for s in SP},
        "both_bands_inside_margin": inside, "both_lower_endpoints_above_margin": above,
        "points": {s: {"m_Feb": pF[s]["m"], "m_June_star": pJ[s]["m"],
                       "difference": dif[s]["difference"]} for s in SP},
        "verdict": verdict, "reading": reading,
        "not_a_test": "these are empirical day-and-member variation bands, not confidence intervals "
                      "and not a causal equivalence test; the member draw holds about 2.13 distinct "
                      "members of three and is biased toward the K = 1 value",
        "no_substitute": "no loss field, no v22, no subgroup and no alternate threshold can replace "
                         "this reading",
        "meaning": "no outcome isolates season, validates 2026 transfer, moves the 0.6874-second "
                   "destination, reopens D32, creates a candidate, or approves a serving rule; "
                   "curve-consistent closes only this tested origin-screen form, provisionally, "
                   "and the desk sets the path status"}
    #: exp-058's rule O is reported unchanged as a reproduction, not re-run as a rule
    rec["rule_O_exp058_reproduced"] = {
        "m_Feb": {s: pF[s]["m"] for s in SP},
        "m_June_matched": {s: rec["points"]["june_matched_K3"]["steps"][s]["m"] for s in SP},
        "advances": bool(all(pF[s]["m"] >= 0.95 for s in READ_STEPS)
                         and all(rec["points"]["june_matched_K3"]["steps"][s]["m"] <= 0.85
                                 for s in READ_STEPS)),
        "note": "exp-058's rule O and its thresholds are not this experiment's rule; they are "
                "reproduced to show that nothing moved"}

    # ---- COMMON's populations ---------------------------------------------------------------
    rec["populations"] = populations(pan, lab, lev)
    pw = 0.0
    for f in C.FOLDS:
        for nm in (("P3bag", "C3bag") if f == F else ("P3bag",)):
            for Lv in range(4):
                for k in rec["populations"][f][f"{nm}_L{Lv}"]:
                    cell = lfj["populations"][f][f"{nm}_L{Lv}"].get(k)
                    ref = cell.get("rmse_s") if isinstance(cell, dict) else None
                    if ref is not None:
                        pw = max(pw, abs(ref - rec["populations"][f][f"{nm}_L{Lv}"][k]))
    pw58 = 0.0
    for f in C.FOLDS:
        for nm, cell in e58["populations"][f].items():
            if not isinstance(cell, dict) or nm.startswith("rows") \
                    or nm not in rec["populations"][f]:
                continue
            for k, ref in cell.items():
                if isinstance(ref, (int, float)) and not isinstance(ref, bool) \
                        and k in rec["populations"][f][nm]:
                    pw58 = max(pw58, abs(ref - rec["populations"][f][nm][k]))
    rec["checks"] |= {"populations_vs_exp055_max_abs_diff_s": pw,
                      "populations_vs_exp058_max_abs_diff_s": pw58}
    if max(pw, pw58) > TOL_9:
        raise SystemExit(f"a reproduced population differs by {max(pw, pw58):.3e} s")

    # ---- the OOF and row receipts -----------------------------------------------------------
    oofs = {}
    for f in C.FOLDS:
        pbf = pan[f]["pbase"]
        src = ({"F3": "F3bag", "J3": f"{JARM}3bag", "M3": "M3bag", "P3": "P3bag"} if f == F
               else {"F3": "P3bag", "J3": "P3bag", "M3": "P3bag", "P3": "P3bag"})
        o = pd.DataFrame({"MVT_ID_mvt": lab[f]["mvt_raw"], "fold": f, "y_true": lab[f]["y"],
                          "y_pred": pbf + lev[f][src["J3"]][3]})
        for nm, sn in src.items():
            for Lv in range(4):
                o[f"y_pred_{nm}_L{Lv}"] = pbf + lev[f][sn][Lv]
        oofs[f] = o
    mem = pd.DataFrame({"MVT_ID_mvt": lab[F]["mvt_raw"], "fold": F, "y_true": y})
    for s in SEEDS3:
        for Lv in range(4):
            mem[f"y_pred_{JARM}_s{s}_L{Lv}"] = pb + L7[f"{JARM}{s}"][Lv]
    rows = pd.DataFrame({"MVT_ID_mvt": lab[F]["mvt_raw"], "day": pf["day"], "airport": pf["ap"],
                         "H": H, "y_true": y, "pbase": pb})
    for c in BAG_COMPS:
        X, Y, _, _ = COMPS[c]
        for k in KS:
            s = STEP[k]
            rows[f"r_{c}_{s}"] = y - pb - L7[X][k - 1]
            rows[f"dX_{c}_{s}"] = L7[X][k] - L7[X][k - 1]
            rows[f"dY_{c}_{s}"] = L7[Y][k] - L7[Y][k - 1]

    # ---- the source check and the outputs ---------------------------------------------------
    sc = source_check()
    rec["source_check"] = {k_: x_ for k_, x_ in sc.items() if k_ != "files"} | {
        "files_opened": sc["files"]}
    if not sc["clean"]:
        raise SystemExit(f"the source check fails: {json.dumps(sc, indent=1)}")
    opened_work = sorted(r for r in OPENED if r.startswith(WORK_REL))
    rec["exp062_vectors_opened"] = opened_work
    if STANDIN and opened_work:
        raise SystemExit(f"the dry run opened {len(opened_work)} exp-062 vectors")
    if not STANDIN and len(opened_work) != 12:
        raise SystemExit(f"the measurement opened {len(opened_work)} June* vectors, not 12")

    def nm_(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    def put_text(p: Path, t: str) -> str:
        NO.refuse_existing([p], overwrite=False)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(t)
        return sha_file(p)

    def put_parquet(p: Path, df: pd.DataFrame) -> str:
        NO.refuse_existing([p], overwrite=False)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
        if not pd.read_parquet(p).equals(df):
            raise SystemExit(f"{p} does not read back")
        return sha_file(p)
    rec["receipts"] = {}
    rp_ = REC_DIR / "daysums-july_loo.csv"
    rec["receipts"][nm_(rp_)] = put_text(rp_, text)
    if not np.array_equal(LF.day_matrix(LF.read_receipt(rp_), days)[0], Dm):
        raise SystemExit("the day-by-airport receipt on disk does not rebuild the day sums")
    rp_ = REC_DIR / "days-july_loo.csv"
    rec["receipts"][nm_(rp_)] = put_text(rp_, dtext)
    if not np.array_equal(pd.read_csv(rp_, float_precision="round_trip")[list(ci)]
                          .to_numpy("float64"), Dm):
        raise SystemExit("the day receipt on disk does not read back")
    for L in BLOCKS:
        rec["receipts"][nm_(REC_DIR / f"draws-L{L}.csv")] = put_text(
            REC_DIR / f"draws-L{L}.csv", dr_csv[L])
    rec["oof"] = {}
    for f in C.FOLDS:
        p = OOFD / f"ownvalue-{f}.parquet"
        o = oofs[f]
        rec["oof"][nm_(p)] = {"sha256": put_parquet(p, o), "rows": int(len(o)),
                              "columns": list(o.columns),
                              "ordered_sha256": {c: C.sha_arr(pan[f]["ids"],
                                                              o[c].to_numpy("float64"))
                                                 for c in o.columns if c.startswith("y_pred")}}
    p = OOFD / "members-july_loo.parquet"
    rec["oof"][nm_(p)] = {"sha256": put_parquet(p, mem), "rows": int(len(mem)),
                          "columns": list(mem.columns)}
    p = ROWD / "rows-july_loo.parquet"
    rec["receipts"][nm_(p)] = put_parquet(p, rows)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["seconds_process_cpu"] = round(time.process_time() - c0, 1)
    C.write_json(OUT, rec)

    # ---- the log ----------------------------------------------------------------------------
    for c in BAG_COMPS + ("carrier_K10",):
        e = rec["points"][c]
        line = f"{c:18s} K={e['K']:<2d}"
        for s, q in e["steps"].items():
            line += f" | {s} m {q['m']:.4f} cos {q['cos']:.3f} J {q['J']:+8.2f}"
        print(line + f" | mean J {e['mean_J']['point']:+8.2f}", flush=True)
    for nm in DIFFS:
        for s in SP:
            d_ = rec["paired_differences"][nm][s]
            iv = d_["intervals"]["L3"]["day+member"]
            print(f"{nm} {s}: {d_['difference']:+.4f}, 3-day day+member "
                  f"[{iv['lo']:+.4f}, {iv['hi']:+.4f}]", flush=True)
    print(f"June* = {JARM} (k = {JK}, N = {JN:,}), V = "
          f"{own['by_arm']['june_star_K3']['value']:+.4f} s2/row, inside B "
          f"{own['june_star_in_band']}", flush=True)
    print(f"verdict: {verdict}", flush=True)
    print(f"reading: {reading}", flush=True)
    print("own value L0: " + ", ".join(
        f"{k_} {own['by_arm'][k_]['value']:+.1f}" for k_ in
        ("feb_K3", "june_star_K3", "june_matched_K3", "carrier_K3")) + " s2/row", flush=True)
    print(f"written {nm_(OUT)} ({rec['seconds']} s wall, {rec['seconds_process_cpu']} s process "
          f"cpu){' -- STAND-IN RUN, DISCARD' if STANDIN else ''}", flush=True)


if __name__ == "__main__":
    main()
