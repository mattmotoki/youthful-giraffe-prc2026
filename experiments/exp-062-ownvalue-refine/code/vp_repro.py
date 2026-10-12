"""exp-062 ownvalue-refine: every reproduction that must pass before a new interpretation
(SPEC sections 6 and 7).  ZERO FIT.  READS LABELS.  It opens no exp-062 vector and asserts it.

    vp_ctl.py --kind repro --log logs/repro.log -- .venv/bin/python .../vp_repro.py

Part A -- exp-059's R1-R8 with its FINAL CORRECTED scope (SPEC 6):
  R1  exp-055 `originvec.load()` re-verifies every exp-055 ordered pin; the labels equal exp-051's
      OOF `y_true` bit for bit on every fold.
  R2  exp-051's 36 stored July gains, with exp-051's arithmetic, through a reader that records every
      key it gives and refuses any other.
  R3/R4  exp-055's `screen.py` and `lossfals.py`, unchanged, run with `--scratch` into
      artifacts/exp-062/repro/.  The registered fields, the WHOLE records, the receipt hashes and
      the OOF comparison are each a REFUSAL in this driver, not a recorded observation
      (exp-059 review round 2, finding 3).
  R5  the regenerated draws carry exp-055's `draw_sha256` and `multiset_counts` and exp-057's
      unsorted K = 10 member draws.
  R6  `bag(P1..P10)` equals Q15's saved heads; `C3` and `P3` rebuild exp-055's pinned bags; S1
      carries exp-051's recorded hashes and S2, S3, Cx4..Cx10 exp-057's.
  R7  exp-057's rule-S and rule-Z points, its three draw receipts column for column, and its
      day-by-airport receipt cell for cell.
  R8  exp-058's 24 vectors by hash; its 17 points over ALL 16 registered fields, of which 15 are
      rebuilt and `q` is compared against Q15's pinned registered constant and is NOT rebuilt from
      Q15's ranks and fold receipts (review round 2, finding 1); its five L0 own values; its paired
      m_Feb - m_June intervals; its three draw receipts; its day and day-by-airport cells; and
      EVERY leaf of its recorded population block (2,343 values) and of exp-057's (1,287), with a
      missing path on either side a refusal (review round 2, finding 2).

Part B -- the separate reproductions the assignment names (SPEC 7), each a hard stop:
  B1  the k = 24 block hashes, the 23-row count sequence and the k = 1, 2, 3, 4, 8 and 12 rows,
      carried from this experiment's committed analysis/plan.json, which refuses any difference
      from the rows exp-059's plan.json registered.
  B2  the fit-free anchors: V(0) = 0 as an identity, and V(60,777) = +493.2541127299781.
  B3  February's value V* = +168.58928050561724 and the band B = V* +/- 10 %.
  B4  exp-059's fitted V at k = 4, 8 and 12 and their three members each, recomputed here from
      exp-059's own vectors on exp-058's cell-to-day-to-total summation route and required to equal
      exp-059's analysis/ladder2.json to 1e-12 s2/row.  Those vectors are read to REPRODUCE a
      recorded value; no prediction of this experiment is built from an exp-059 vector.
  B5  the day cells: exp-058's daysums and days receipts, exp-057's daysums receipt, and exp-055's
      i2 daysums receipt, cell for cell over every shared column.
"""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import os
import subprocess
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
OUT = ANALYSIS / "repro.json"
REC_DIR = ANALYSIS / "receipts"
REPRO_DIR = ROOT / "artifacts/exp-062/repro-r1"
OWN_PREFIX = ("experiments/exp-062-ownvalue-refine/", "artifacts/exp-062/")
E62_WORK = "artifacts/exp-062/work"
PY = str(ROOT / ".venv/bin/python")

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

#: SPEC 2.3: this process imports exp-055's closure only (exp-051's and exp-055's both hold a
#: module named `common`); every module of this experiment is named `vp_*`.
sys.path.insert(0, str(E55 / "code"))
import common as C                                                    # noqa: E402  (exp-055)
import lossfals as LF                                                 # noqa: E402
import originvec as V                                                 # noqa: E402
import screen as SC                                                   # noqa: E402

for _name, _want in (("common", E55 / "code/common.py"), ("originvec", E55 / "code/originvec.py"),
                     ("screen", E55 / "code/screen.py"), ("lossfals", E55 / "code/lossfals.py")):
    if Path(sys.modules[_name].__file__).resolve() != _want:
        raise SystemExit(f"module {_name} resolves to {sys.modules[_name].__file__} (name collision)")
if "carrier" in sys.modules or "memberspread" in sys.modules:
    raise SystemExit("exp-051's closure is imported in the reproduction process")

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

A = C.A
NO = C.NO
F = "july_loo"
B = 2000
BLOCKS = (1, 2, 3)
KS = (1, 2, 3)
STEP = C.STEP
SP = [STEP[k] for k in KS]
SEEDS3 = (1, 2, 3)
SEEDS10 = tuple(range(1, 11))
MULTI3 = list(itertools.combinations_with_replacement(SEEDS3, 3))
MS3 = ["".join(map(str, t)) for t in MULTI3]
CUT = 3600.0
STATS = ("ry", "yy", "rx", "xx", "xy")
TOL_12, TOL_9 = 1e-12, 1e-9
#: Q15's registered q, a PINNED COMPARATOR: this experiment does not rebuild q from Q15's ranks and
#: fold receipts and makes no new claim about it (exp-059 review round 2, finding 1)
Q_POINT = {"v20": 0.7541214069685656, "v21": 0.7872337829612198, "v22": 0.5869686712745045}
Q_L3 = {"v20": (0.7030750838597688, 0.7941267683291252),
        "v21": (0.7364214976003001, 0.8322645424736729),
        "v22": (0.555629780110832, 0.610847845536199)}
R8_III_FIELDS = ("m", "q", "m_minus_q", "m_over_q", "I", "cos", "rms_H_carrier_s", "rms_H_plain_s",
                 "A", "B", "g_C", "g_hat", "g_plain", "J", "amplitude", "alignment")
SCREEN_SHA = "081c39ed4789c5fc4851fe61808bf1cc28b13d5eee1e1e974a768faa17721ae9"
LOSSFALS_SHA = "1eecc0b52e517a5f7749c0a75ce6b9529cb268b708f163402945ce337e64a52f"
CARRIER_SHA = "dc49aa89816140365111aa601965ba6bcb6b4735340bf64f8f7aa4c92dbe25b4"
E57_SPEC_COMMIT = "95d82b6611e2768e2a25fd596bd113f168eb307e"
E58_SPEC_COMMIT = "22e2a6b91f94040d1999d483ae4bf95565ca1f72"
E58_CODE_SHA = "9a970bacf95c05dbf923d7a0e7260e93a69fa90c69553ceebe6e09928338dd11"
E59_SPEC_COMMIT = "49db4157d8bc43edde6c2b8386e90b6f58929578"
ARM_MONTH = {"F": "2025-02", "M": "2025-06"}
#: SPEC 5: the frozen own-value constants
V_STAR = 168.58928050561724
BAND = (151.7303524550555, 185.44820855617897)
V_TOP = 493.2541127299781
N_TOP = 60_777
E58_OWN = {"feb_K3": V_STAR, "june_matched_K3": V_TOP, "carrier_K3": 582.2425054210122,
           "sham_K3": -4009.1762051589712, "carrier_K10": 571.4884463397937}
#: SPEC 7 B4: exp-059's three fitted grid rows, reproduced here from its own vectors
E59_K = (4, 8, 12)

COMPS58 = {"feb_K3": ("F3bag", "P3bag", 3, "exp-058: the K = 3 February bag"),
           "june_matched_K3": ("M3bag", "P3bag", 3, "exp-058: the K = 3 matched June bag (N = 60,777)"),
           "carrier_K3": ("C3bag", "P3bag", 3, "comparator: exp-051's full June carrier"),
           "sham_K3": ("S3bag", "P3bag", 3, "comparator: exp-057's K = 3 sham"),
           "carrier_K10": ("C10bag", "P10bag", 10, "comparator: exp-057's K = 10 June carrier")}
COMPS58 |= {f"feb_s{s}": (f"F{s}", f"P{s}", 1, "February member") for s in SEEDS3}
COMPS58 |= {f"june_matched_s{s}": (f"M{s}", f"P{s}", 1, "matched June member") for s in SEEDS3}
COMPS58 |= {f"carrier_s{s}": (f"C{s}", f"P{s}", 1, "exp-051 June member") for s in SEEDS3}
COMPS58 |= {f"sham_s{s}": (f"S{s}", f"P{s}", 1, "exp-057 sham member") for s in SEEDS3}
BAG_COMPS58 = ("feb_K3", "june_matched_K3", "carrier_K3", "sham_K3", "carrier_K10")
K3_DRAW58 = {"feb_K3": "F", "june_matched_K3": "M", "carrier_K3": "C", "sham_K3": "S"}
SSE58 = (["P3bag", "F3bag", "M3bag", "C3bag", "S3bag", "P10bag", "C10bag"]
         + [f"{nm}{s}" for nm in ("P", "F", "M", "C", "S") for s in SEEDS3])
COMPS57 = {c: COMPS58[c] for c in ("sham_K3", "carrier_K3", "carrier_K10")}
SSE57 = ["P3bag", "S3bag", "C3bag", "P10bag", "C10bag", "S1", "S2", "S3", "P1", "P2", "P3"]


def check_pin(p: Path) -> str:
    """The file's bytes must carry the positive manifest's hash, or exp-059's verified census hash
    for a path outside the manifest (SPEC 2.5)."""
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
            inside += 1 if SOURCE[rel] == s else 0
            if SOURCE[rel] != s:
                differ.append(rel)
        elif rel in E59_CENSUS:
            (census_only if E59_CENSUS[rel] == s else differ).append(rel)
        elif rel in E59_VECTORS:
            (e59_vec if E59_VECTORS[rel] == s else differ).append(rel)
        else:
            outside.append(rel)
    return {"opened": len(opened), "in_manifest": inside, "in_exp059_census_only": census_only,
            "in_exp059_record_pins_only": e59_vec,
            "outside_every_pin": outside, "differ": differ,
            "clean": not (differ or outside), "files": opened}


def fl(x):
    return float(x)


def interval(x) -> dict:
    return LF.interval(x)


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
            "m_star": A_ / Bq, "best_amplitude_value": A_ ** 2 / Bq,
            "rms_H_carrier_s": np.sqrt(xx / nH), "rms_H_plain_s": np.sqrt(yy / nH)}


def q_cols(S, ci, c: str, s: str) -> dict:
    return quantities(S[..., ci["n"]], S[..., ci["n_H"]],
                      *(S[..., ci[f"{st}_{c}_{s}"]] for st in STATS))


def imp(m, q):
    return np.abs(1 - q) - np.abs(m - q)


# ------------------------------------------------------------------------------ R2 --------------

class Tracked:
    """A JSON object that records every key path it gives and refuses any other (R2)."""

    def __init__(self, obj, allowed: set, path: str = "", log: set | None = None):
        self.obj, self.allowed, self.path = obj, allowed, path
        self.log = set() if log is None else log

    def __getitem__(self, k):
        p = f"{self.path}.{k}" if self.path else str(k)
        if not any(a == p or a.startswith(p + ".") for a in self.allowed):
            raise SystemExit(f"carrier.json: key {p} is outside the registered read scope")
        v = self.obj[k]
        if p in self.allowed:
            if isinstance(v, dict):
                raise SystemExit(f"carrier.json: {p} is not a leaf")
            self.log.add(p)
            return v
        return Tracked(v, self.allowed, p, self.log)


def r2(pan: dict, vec: dict, lab: dict) -> dict:
    p = C.E51 / "analysis/carrier.json"
    if check_pin(p) != CARRIER_SHA:
        raise SystemExit("carrier.json is not the pinned record")
    allowed = set()
    for s in SP:
        allowed |= {f"steps.{s}.july_s2_per_row.C3", f"steps.{s}.july_s2_per_row.P3",
                    f"steps.{s}.sham_K1.july_s2.S", f"steps.{s}.sham_K1.july_s2.P1",
                    f"steps.{s}.january_s2_per_row_both_arms",
                    f"steps.{s}.december_s2_per_row_both_arms"}
        for m in SEEDS3:
            allowed |= {f"steps.{s}.members_record_only.{m}.july_s2.C",
                        f"steps.{s}.members_record_only.{m}.july_s2.P"}
    root = Tracked(json.loads(p.read_text()), allowed)
    car = root["steps"]
    out, worst = {}, 0.0
    for f in C.FOLDS:
        codes = pd.factorize(pan[f]["day"])[0]
        y, pb = lab[f]["y"], pan[f]["pbase"]
        v = vec[f]

        def g(nm, k):
            return (LF.mse51(y, pb, v[f"{nm}_L{k - 1}"], codes)
                    - LF.mse51(y, pb, v[f"{nm}_L{k}"], codes))
        for k in KS:
            s = STEP[k]
            if f == F:
                want = {"C3bag": car[s]["july_s2_per_row"]["C3"],
                        "P3bag": car[s]["july_s2_per_row"]["P3"],
                        "S1": car[s]["sham_K1"]["july_s2"]["S"],
                        "P1:sham": car[s]["sham_K1"]["july_s2"]["P1"]}
                for m in SEEDS3:
                    want[f"C{m}"] = car[s]["members_record_only"][str(m)]["july_s2"]["C"]
                    want[f"P{m}"] = car[s]["members_record_only"][str(m)]["july_s2"]["P"]
            else:
                want = {"P3bag": car[s][f"{f}_s2_per_row_both_arms"]}
            for key, ref in want.items():
                got = g(key.split(":")[0], k)
                d = abs(got - ref)
                worst = max(worst, d)
                out[f"{s}:{f}:{key}"] = {"exp051": ref, "reproduced": got, "abs_diff": d}
                if d > TOL_9:
                    raise SystemExit(f"R2 {s} {f} {key}: {got!r} is not exp-051's {ref!r}")
    if root.log != allowed:
        raise SystemExit(f"R2: {len(allowed - root.log)} registered keys were not read")
    return {"tolerance_s2_per_row": TOL_9, "values": out, "max_abs_diff": worst,
            "keys_read": sorted(root.log), "n_keys": len(root.log), "other_keys_read": 0}


# ------------------------------------------------------------------------------ R3 / R4 ---------

def jdiff(a, b, path="", skip=("seconds",)) -> tuple:
    if isinstance(a, dict) and isinstance(b, dict):
        diffs, wa, wr, nd = [], 0.0, 0.0, 0
        ka, kb = set(a) - set(skip), set(b) - set(skip)
        if ka != kb:
            diffs.append(f"{path}: keys {sorted(ka ^ kb)[:5]}")
        for k in sorted(ka & kb):
            d, x, y, z = jdiff(a[k], b[k], f"{path}.{k}", skip)
            diffs += d
            wa, wr, nd = max(wa, x), max(wr, y), nd + z
        return diffs, wa, wr, nd
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: list length {len(a)} vs {len(b)}"], 0.0, 0.0, 0
        diffs, wa, wr, nd = [], 0.0, 0.0, 0
        for i, (x_, y_) in enumerate(zip(a, b)):
            d, x, y, z = jdiff(x_, y_, f"{path}[{i}]", skip)
            diffs += d
            wa, wr, nd = max(wa, x), max(wr, y), nd + z
        return diffs, wa, wr, nd
    num = (int, float)
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, num) or not isinstance(b, num):
        return ([] if a == b else [f"{path}: {a!r} vs {b!r}"]), 0.0, 0.0, 0
    if a == b:
        return [], 0.0, 0.0, 0
    d = abs(float(a) - float(b))
    return [], d, d / max(abs(float(b)), 1e-300), 1


def r34() -> dict:
    """exp-055's screen.py and lossfals.py, unchanged, into a write-once scratch directory.  Every
    comparison here is a REFUSAL, not a recorded observation (exp-059 review round 2, finding 3)."""
    env = dict(os.environ, OMP_NUM_THREADS="3")
    runs = (("screen.py", "screen", "screen.json", E55 / "analysis/screen.json", SCREEN_SHA),
            ("lossfals.py", "lossfals", "analysis/lossfals.json", E55 / "analysis/lossfals.json",
             LOSSFALS_SHA))
    out, got = {"runs": {}}, {}
    for script, sub, rel_out, want_path, want_sha in runs:
        d = REPRO_DIR / sub
        if d.exists():
            raise SystemExit(f"{d} exists; the reproduction directory is write-once")
        d.mkdir(parents=True)
        t0 = time.time()
        res = subprocess.run([PY, str(E55 / "code" / script), "--scratch", str(d)], cwd=ROOT,
                             env=env, capture_output=True, text=True)
        (d / "run.log").write_text(res.stdout + "\n--- stderr ---\n" + res.stderr)
        if res.returncode != 0:
            raise SystemExit(f"R3/R4: exp-055 {script} failed:\n{res.stderr[-2000:]}")
        if check_pin(want_path) != want_sha:
            raise SystemExit(f"{want_path} is not the pinned record")
        g, w = json.loads((d / rel_out).read_text()), json.loads(want_path.read_text())
        got[sub] = (g, w)
        diffs, wa, wr, nd = jdiff(g, w, skip=("seconds", "receipts", "oof"))
        rsha = {Path(k).name: v for k, v in g["receipts"].items()}
        wsha = {Path(k).name: v for k, v in w["receipts"].items()}
        oofeq = ({k: g["oof"][k] == w["oof"][k]
                  for k in ("sha256", "rows", "columns", "ordered_sha256")} if "oof" in w else None)
        out["runs"][sub] = {"script": f"experiments/exp-055-originmove/code/{script}",
                            "scratch": str(d.relative_to(ROOT)), "seconds": round(time.time() - t0, 1),
                            "whole_record": {"structural_diffs": diffs[:20],
                                             "n_structural_diffs": len(diffs),
                                             "float_fields_differing": nd,
                                             "max_abs_float_diff": wa, "max_rel_float_diff": wr},
                            "receipt_hashes_equal": rsha == wsha, "oof_equal": oofeq,
                            "stdout_tail": res.stdout.strip().splitlines()[-2:]}
        #: the enforcement round 2 asked for: each of these refuses here
        if len(diffs) or nd or wa != 0.0 or wr != 0.0:
            raise SystemExit(f"R3/R4 {sub}: the whole record differs ({len(diffs)} structural, "
                             f"{nd} float fields, max abs {wa:.3e}, max rel {wr:.3e})")
        if rsha != wsha:
            raise SystemExit(f"R3/R4 {sub}: a receipt hash differs")
        if oofeq is not None and not all(bool(v) for v in oofeq.values()):
            raise SystemExit(f"R3/R4 {sub}: the OOF comparison is not all true ({oofeq})")
    sg, sw = got["screen"]
    lg, lw = got["lossfals"]
    reg = []
    for key in ("july_loo->2026-07", "january->2026-01"):
        for s in SP:
            reg.append((f"R3 q {key} {s}", sg["q"][key][s]["point"], sw["q"][key][s]["point"], TOL_12))
    for arm in ("june_bag", "june_seed1", "june_seed2", "june_seed3", "sham"):
        for s in SP:
            for k_ in ("m", "I"):
                reg.append((f"R3 {arm} {s} {k_}", sg["arms"][arm]["steps"][s][k_],
                            sw["arms"][arm]["steps"][s][k_], TOL_12))
    for L in BLOCKS:
        for mode in ("day", "day+seed"):
            for e in ("lo", "hi"):
                reg.append((f"R3 june_bag mean_I L{L} {mode} {e}",
                            sg["arms"]["june_bag"]["mean_I"]["intervals"][f"L{L}"][mode][e],
                            sw["arms"]["june_bag"]["mean_I"]["intervals"][f"L{L}"][mode][e], TOL_9))
                reg.append((f"R4 bag123 mean_J L{L} {mode} {e}",
                            lg["points"][F]["bag123"]["mean_J"]["intervals"][f"L{L}"][mode][e],
                            lw["points"][F]["bag123"]["mean_J"]["intervals"][f"L{L}"][mode][e], TOL_9))
        for s in SP:
            for e in ("lo", "hi"):
                reg.append((f"R3 q july {s} L{L} {e}",
                            sg["q"]["july_loo->2026-07"][s]["intervals"][f"L{L}"][e],
                            sw["q"]["july_loo->2026-07"][s]["intervals"][f"L{L}"][e], TOL_9))
    for pt in ("bag123", "seed1", "seed2", "seed3", "sham"):
        for s in SP:
            for k_ in ("A", "B", "m", "g_C", "g_hat", "g_plain", "J"):
                reg.append((f"R4 {pt} {s} {k_}", lg["points"][F][pt]["steps"][s][k_],
                            lw["points"][F][pt]["steps"][s][k_], TOL_9))
    worst = {TOL_12: 0.0, TOL_9: 0.0}
    for nm, a, b, tol in reg:
        d = abs(a - b)
        worst[tol] = max(worst[tol], d)
        if d > tol:
            raise SystemExit(f"{nm}: the scratch run gives {a!r}, the record {b!r} ({d:.3e})")
    out["registered_fields"] = {"n": len(reg), "max_abs_diff_fields_at_1e-12": worst[TOL_12],
                               "max_abs_diff_fields_at_1e-9": worst[TOL_9],
                               "all_within_tolerance": True}
    out["enforced_in_this_driver"] = ["registered fields", "whole record", "receipt hashes", "OOF"]
    return out


def own_points(pan: dict, vec: dict, lab: dict, scr: dict, lfj: dict) -> dict:
    """This script's own arithmetic on exp-055's July points, from the pinned vectors."""
    pf, v = pan[F], vec[F]
    H, y, pb = pf["H"], lab[F]["y"], pf["pbase"]
    n = len(y)
    pts = {"bag123": ("C3bag", "P3bag", "june_bag"), "seed1": ("C1", "P1", "june_seed1"),
           "seed2": ("C2", "P2", "june_seed2"), "seed3": ("C3", "P3", "june_seed3"),
           "sham": ("S1", "P1", "sham")}
    w9, w12 = 0.0, 0.0
    for pt, (X, Y, arm) in pts.items():
        for k in KS:
            s = STEP[k]
            dx = v[f"{X}_L{k}"] - v[f"{X}_L{k - 1}"]
            dy = v[f"{Y}_L{k}"] - v[f"{Y}_L{k - 1}"]
            r = y - pb - v[f"{X}_L{k - 1}"]
            q = quantities(n, H.sum(), np.sum(r[H] * dy[H]), np.sum(dy[H] ** 2),
                           np.sum(r[H] * dx[H]), np.sum(dx[H] ** 2), np.sum(dx[H] * dy[H]))
            ref = lfj["points"][F][pt]["steps"][s]
            for key in ("A", "B", "g_C", "g_hat", "g_plain", "J", "amplitude", "alignment"):
                w9 = max(w9, abs(float(q[key]) - ref[key]))
            sref = scr["arms"][arm]["steps"][s]
            w12 = max(w12, abs(float(q["m"]) - sref["m"]),
                      abs(float(imp(q["m"], Q_POINT[s])) - sref["I"]),
                      abs(float(q["cos"]) - sref["cos"]), abs(float(q["m"]) - ref["m"]))
    if w9 > TOL_9 or w12 > TOL_12:
        raise SystemExit(f"this script's arithmetic misses exp-055's points: loss {w9:.3e}, "
                         f"m/I/cos {w12:.3e}")
    return {"points": list(pts), "loss_max_abs_diff_s2": w9, "m_I_cos_max_abs_diff": w12}


# ------------------------------------------------------------------------------ vectors ---------

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
                z = np.load(p)
                if list(z.keys()) != [F]:
                    raise SystemExit(f"exp-057 {tag}: the vector holds {list(z.keys())}")
                vec[F][f"{kind}{s}_L{L}"] = z[F].astype("float64")
                info[tag] = {"vector": rec["vector"], "sha256": rec["vector_sha256"]}
    return info


def load_exp058(vec: dict) -> dict:
    """exp-058's 24 February and matched-June vectors, sha-checked against its fit records (R8 ii)."""
    info = {}
    e58plan = json.loads((E58 / "analysis/plan.json").read_text())
    check_pin(E58 / "analysis/plan.json")
    for arm in ("F", "M"):
        bsha = e58plan["match"]["months"][ARM_MONTH[arm]]["selection"]["ids_sha256_block_order"]
        for s in SEEDS3:
            for L in range(4):
                tag = f"{arm}-s{s}-L{L}"
                rp = E58 / "analysis/fits" / f"{tag}.json"
                check_pin(rp)
                rec = json.loads(rp.read_text())
                if rec.get("spec_commit") != E58_SPEC_COMMIT or rec.get("code_sha256") != E58_CODE_SHA:
                    raise SystemExit(f"exp-058 {tag}: not exp-058's registration or plan code")
                if (rec["arm"] != arm or rec["seed"] != s or rec["level"] != L
                        or rec["month"] != ARM_MONTH[arm] or rec["block_rows"] != N_TOP
                        or rec["block_ids_sha256_block_order"] != bsha):
                    raise SystemExit(f"exp-058 {tag}: the record does not identify this arm")
                pth = ROOT / rec["vector"]
                if check_pin(pth) != rec["vector_sha256"]:
                    raise SystemExit(f"exp-058 {tag}: the vector is not the recorded bytes")
                z = np.load(pth)
                if list(z.keys()) != [F]:
                    raise SystemExit(f"exp-058 {tag}: the vector holds {list(z.keys())}")
                vec[F][f"{arm}{s}_L{L}"] = z[F].astype("float64")
                info[tag] = {"vector": rec["vector"], "sha256": rec["vector_sha256"],
                             "rows_moved": rec["rows_moved"]}
    info["_february_block_ids_sha256_block_order"] = \
        e58plan["match"]["months"]["2025-02"]["selection"]["ids_sha256_block_order"]
    info["_june_block_ids_sha256_block_order"] = \
        e58plan["match"]["months"]["2025-06"]["selection"]["ids_sha256_block_order"]
    return info


def load_exp059_L0(vec: dict) -> dict:
    """SPEC 7 B4: exp-059's three fitted arms at L0 ONLY, read to reproduce its recorded V.  No
    prediction of this experiment is built from an exp-059 vector."""
    check_pin(E59 / "analysis/plan.json")
    reg = json.loads((E59 / "analysis/plan.json").read_text())["grid"]["rows"]
    info = {}
    for k in E59_K:
        arm = f"G{k:02d}"
        for s in SEEDS3:
            tag = f"{arm}-s{s}-L0"
            rp = E59 / "analysis/fits" / f"{tag}.json"
            check_pin(rp)
            rec = json.loads(rp.read_text())
            if rec.get("spec_commit") != E59_SPEC_COMMIT:
                raise SystemExit(f"exp-059 {tag}: not exp-059's registration")
            if (rec["k"] != k or rec["seed"] != s or rec["level"] != 0 or rec["month"] != "2025-06"
                    or rec["block_rows"] != reg[str(k)]["block_rows"]
                    or rec["block_ids_sha256_block_order"] != reg[str(k)]["ids_sha256_block_order"]
                    or rec["block_count_vector_sha256"] != reg[str(k)]["count_vector_sha256"]):
                raise SystemExit(f"exp-059 {tag}: the record does not identify this grid row")
            p = ROOT / rec["vector"]
            if check_pin(p) != rec["vector_sha256"]:
                raise SystemExit(f"exp-059 {tag}: the vector is not the recorded bytes")
            z = np.load(p)
            if list(z.keys()) != [F]:
                raise SystemExit(f"exp-059 {tag}: the vector holds {list(z.keys())}")
            vec[F][f"{arm}{s}_L0"] = z[F].astype("float64")
            info[tag] = {"vector": rec["vector"], "sha256": rec["vector_sha256"],
                         "k": k, "block_rows": rec["block_rows"], "rows_moved": rec["rows_moved"]}
    return info


def bag_of(pf: dict, v: dict, nm: str, seeds, L: int) -> np.ndarray:
    return C.bag({s: v[f"{nm}{s}_L{L}"] for s in seeds}, seeds, pf["pbase"], pf["usable"], pf["guard"])


# ------------------------------------------------------------------------------ own value -------

class Own:
    """SPEC 5: V = [SSE(P3_L0) - SSE(X_L0)] / n on every July fold row, on exp-058's route.

    exp-058 sums (y - pbase - vector)^2 into (UTC day, airport) cells, adds the cells into days with
    exp-055 lossfals's day matrix, adds the days, and subtracts the two totals.  Each total is near
    7.6e9 s2 while their difference is near 3e7 s2, so a route that subtracts row by row first is
    more accurate but does NOT reproduce the registered digits.  This walks exp-058's route column
    for column, which exp-059 verified reproduces its five own values bit for bit."""

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


# ------------------------------------------------------------------------------ draws -----------

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


def cmp_draw_csv(mine: str, path: Path) -> dict:
    check_pin(path)
    a = pd.read_csv(io.StringIO(mine), dtype={"k3_multiset": str, "k10_indices": str},
                    float_precision="round_trip")
    b = pd.read_csv(path, dtype={"k3_multiset": str, "k10_indices": str},
                    float_precision="round_trip")
    if list(a.columns) != list(b.columns):
        raise SystemExit(f"{path.name}: columns differ from the rebuild's")
    if not a[["draw", "k3_multiset", "k10_indices"]].equals(b[["draw", "k3_multiset", "k10_indices"]]):
        raise SystemExit(f"{path.name}: draw indices or member draws differ from the rebuild's")
    num = [c for c in a.columns if c not in ("k3_multiset", "k10_indices")]
    A_, B_ = a[num].to_numpy("float64"), b[num].to_numpy("float64")
    ad = np.abs(A_ - B_)
    rel = float(np.max(ad / np.maximum(np.abs(B_), 1e-300)))
    return {"columns": len(a.columns), "rows": int(len(a)), "max_abs_diff": float(ad.max()),
            "max_rel_diff": rel, "bytes_identical": bool(mine == path.read_text()),
            "within_1e-9_relative": bool(rel <= TOL_9)}


def cells_july(pan: dict, lv: dict, y: np.ndarray, comps: dict, sse_names: list) -> pd.DataFrame:
    pf = pan[F]
    H, pb = pf["H"], pf["pbase"]
    cd, uq = pd.factorize(pd.MultiIndex.from_arrays([pf["day"], pf["ap"]]), sort=True)
    G = len(uq)

    def cell(w):
        return np.bincount(cd, weights=w, minlength=G)
    cols = {"n": np.bincount(cd, minlength=G).astype("int64"),
            "n_H": np.bincount(cd, weights=H.astype("float64"), minlength=G).astype("int64")}
    for c, (X, Y, K, _) in comps.items():
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
    for nm in sse_names:
        for L in range(4):
            cols[f"sse_{nm}_L{L}"] = cell((y - pb - lv[nm][L]) ** 2)
    return pd.DataFrame({"day": [a for a, _ in uq], "airport": [b for _, b in uq], **cols})


def comparator_levels(pan: dict, vec: dict) -> dict:
    pf, v = pan[F], vec[F]
    lv = {nm: [v[f"{nm}_L{L}"] for L in range(4)]
          for nm in [f"P{s}" for s in SEEDS10] + [f"C{s}" for s in SEEDS10]
          + ["S1", "S2", "S3", "P3bag", "C3bag", "T10"]}
    lv["P10bag"] = [bag_of(pf, v, "P", SEEDS10, L) for L in range(4)]
    lv["C10bag"] = [bag_of(pf, v, "C", SEEDS10, L) for L in range(4)]
    lv["S3bag"] = [bag_of(pf, v, "S", SEEDS3, L) for L in range(4)]
    for L in range(4):
        if not (np.array_equal(bag_of(pf, v, "C", SEEDS3, L), lv["C3bag"][L])
                and np.array_equal(bag_of(pf, v, "P", SEEDS3, L), lv["P3bag"][L])):
            raise SystemExit("R6: the K = 3 bags do not rebuild exp-055's pinned bags")
    return lv


# ------------------------------------------------------------------------------ populations -----

def leaves(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, path + (k,))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, float(obj)


def compare_leaves(ref: dict, mine: dict, tag: str) -> dict:
    """Every leaf of the reference block must exist here and be equal; a missing path is a refusal
    (exp-059 review round 2, finding 2)."""
    n, worst, worst_at, missing = 0, 0.0, None, []
    for path, r in leaves(ref):
        cur = mine
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                cur = None
                break
            cur = cur[k]
        if cur is None or isinstance(cur, bool) or not isinstance(cur, (int, float)):
            missing.append("/".join(path))
            continue
        d = abs(float(cur) - r)
        n += 1
        if d > worst:
            worst, worst_at = d, "/".join(path)
    if missing:
        raise SystemExit(f"{tag}: {len(missing)} recorded values have no counterpart here, "
                         f"first {missing[:3]}")
    if worst > TOL_9:
        raise SystemExit(f"{tag}: a recorded population differs by {worst:.3e} at {worst_at}")
    return {"values_compared": n, "max_abs_diff": worst, "max_abs_diff_at": worst_at, "missing": []}


def populations(pan: dict, lab: dict, lev: dict) -> dict:
    """COMMON's populations over exp-058's recorded predictor set, on exp-058's masks."""
    check_pin(C.E53 / "analysis/pin.json")
    qpin = json.loads((C.E53 / "analysis/pin.json").read_text())
    out = {}
    for f in C.FOLDS:
        ids, yy, pbf = pan[f]["ids"], lab[f]["y"], pan[f]["pbase"]
        nonm = LF.no_nm_mask(ids, f)
        if C.sha_arr(nonm) != qpin["folds"][f]["no_nm_sha256"]:
            raise SystemExit(f"{f}: the no-NM mask is not Q15's")
        masks = {"all": np.ones(len(ids), bool), "le3600": yy <= CUT, "gt3600": yy > CUT,
                 "no_nm": nonm, "clean": lab[f]["clean"]}
        masks |= {f"own_{g}": pan[f]["reg"] == g for g in C.REGIMES}
        masks |= {f"nm_{r_}": lab[f]["nmreg"] == r_ for r_ in sorted(set(lab[f]["nmreg"]))}
        preds = (["F3bag", "M3bag"] + [f"{nm}{i}" for nm in ("F", "M") for i in SEEDS3]
                 + ["P3bag", "C3bag", "S3bag", "P10bag", "C10bag"]) if f == F else ["P3bag", "P10bag"]
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


# ------------------------------------------------------------------------------ main ------------

def main() -> None:
    t0, c0 = time.time(), time.process_time()
    np.seterr(divide="ignore", invalid="ignore")
    outs = [OUT, *(REC_DIR / f"exp057-draws-L{L}.csv" for L in BLOCKS),
            *(REC_DIR / f"exp058-draws-L{L}.csv" for L in BLOCKS),
            REC_DIR / "exp058-daysums-july_loo.csv"]
    NO.refuse_existing(outs, overwrite=False)
    rec = {"what": "exp-062 ownvalue-refine: exp-059's R1-R8 with its final corrected scope, and the "
                   "separate reproductions the assignment names.  Zero fit; no exp-062 vector opened.",
           "manifest_sha256": sha_file(MANIFEST),
           "plan_sha256": sha_file(ANALYSIS / "plan.json"),
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "B": B, "blocks": list(BLOCKS), "generator": "np.random.default_rng(5500 + L)",
           "q_is_a_pinned_comparator":
               "Q15's registered q is compared against its pinned constant and is NOT rebuilt from "
               "Q15's ranks and fold receipts; this experiment makes no new claim about q "
               "(exp-059 review round 2, finding 1)",
           "checks": {}}

    # ---- R1 ---------------------------------------------------------------------------------
    pan, vec, pin = V.load()
    lab = {f: LF.load_labels(f, pan[f]["ids"]) for f in C.FOLDS}
    rec["checks"]["R1_originvec_load"] = True
    print("R1 exp-055 pins verified; the labels equal exp-051's OOF y_true on every fold", flush=True)

    # ---- R2 ---------------------------------------------------------------------------------
    rec["R2_exp051_gains"] = r2(pan, vec, lab)
    print(f"R2 exp-051 gains reproduced: {len(rec['R2_exp051_gains']['values'])} values, max |diff| "
          f"{rec['R2_exp051_gains']['max_abs_diff']:.2e}; {rec['R2_exp051_gains']['n_keys']} keys "
          f"read, no other", flush=True)

    # ---- R3 / R4 ----------------------------------------------------------------------------
    rec["R3_R4_exp055_scripts"] = r34()
    for sub, r_ in rec["R3_R4_exp055_scripts"]["runs"].items():
        print(f"R3/R4 {sub}: registered fields equal; whole record "
              f"{r_['whole_record']['n_structural_diffs']} structural / "
              f"{r_['whole_record']['float_fields_differing']} float; receipts equal "
              f"{r_['receipt_hashes_equal']}; OOF {r_['oof_equal']}", flush=True)
    scr = json.loads((E55 / "analysis/screen.json").read_text())
    lfj = json.loads((E55 / "analysis/lossfals.json").read_text())
    for s in SP:
        got = scr["q"]["july_loo->2026-07"][s]
        if got["point"] != Q_POINT[s] or (got["intervals"]["L3"]["lo"],
                                          got["intervals"]["L3"]["hi"]) != Q_L3[s]:
            raise SystemExit(f"{s}: the registered q interval is not screen.json's")
    rec["checks"]["own_arithmetic_on_exp055_points"] = own_points(pan, vec, lab, scr, lfj)

    # ---- R5 ---------------------------------------------------------------------------------
    e57p = E57 / "analysis/measure.json"
    check_pin(e57p)
    e57 = json.loads(e57p.read_text())
    if e57["spec_commit"] != E57_SPEC_COMMIT or e57["standin"]:
        raise SystemExit("exp-057's measure.json is not its registered official record")
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
        want = scr["draw_sha256"][f"L{L}"]
        got = {k: C.sha_arr(v) for k, v in st.items()} | {"seeds": C.sha_arr(sig)}
        counts = {MS3[j]: int((msi == j).sum()) for j in range(len(MS3))}
        if got != {k: v for k, v in want.items() if k != "multiset_counts"} \
                or counts != want["multiset_counts"]:
            raise SystemExit(f"R5 L{L}: the regenerated draws are not exp-055's")
        if C.sha_arr(sig10) != e57["k10_member_draw_sha256_unsorted"][f"L{L}"]:
            raise SystemExit(f"R5 L{L}: the K = 10 member draws are not exp-057's")
        draws[L] = {"st": st, "msi": msi, "sig10": np.sort(sig10, axis=1),
                    "sig10_sha256": C.sha_arr(sig10)}
    rec["checks"]["R5_draws_equal_exp055_and_exp057"] = True
    rec["k10_member_draw_sha256_unsorted"] = {f"L{L}": draws[L]["sig10_sha256"] for L in BLOCKS}
    print("R5 the regenerated draws equal exp-055's and exp-057's", flush=True)

    # ---- R6 ---------------------------------------------------------------------------------
    for f in C.FOLDS:
        for s in range(4, 11):
            for L in range(4):
                p = C.member_path(L, s)
                check_pin(p)
                vec[f][f"P{s}_L{L}"] = np.load(p)[f].astype("float64")
    rec["exp057_vectors"] = load_exp057(vec)
    p10, worst10 = {}, 0.0
    for f in C.FOLDS:
        pf = pan[f]
        for L in range(4):
            b10 = C.bag({s: vec[f][f"P{s}_L{L}"] for s in SEEDS10}, SEEDS10, pf["pbase"],
                        pf["usable"], pf["guard"])
            d = float(np.max(np.abs(b10 - vec[f][f"T10_L{L}"])))
            worst10 = max(worst10, d)
            p10[f"{f}:L{L}"] = {"max_abs_diff_s": d,
                                "bit_identical": bool(np.array_equal(b10, vec[f][f"T10_L{L}"]))}
    if worst10 > TOL_9:
        raise SystemExit(f"R6: the rebuilt plain K = 10 bag differs from the heads by {worst10:.3e} s")
    s1 = {}
    for L in range(4):
        r51 = json.loads((C.E51 / f"analysis/fits2/i2-S-s1-L{L}.json").read_text())
        s1[f"L{L}"] = (sha_file(C.sham_path(L)) == r51["vector_sha256"]
                       and r51["vector"] == C.rel(C.sham_path(L)))
    if not all(s1.values()):
        raise SystemExit(f"R6: S1 is not exp-051's recorded vector: {s1}")
    rec["R6"] = {"plain_K10_vs_heads_max_abs_diff_s": worst10, "by_fold_level": p10,
                 "S1_equals_exp051_fit_records": s1,
                 "S2_S3_Cx4_Cx10_equal_exp057_fit_records": len(rec["exp057_vectors"]) == 36}
    print(f"R6 the plain K = 10 bag rebuilds the saved heads (max |diff| {worst10:.1e} s); S1 is "
          f"exp-051's; {len(rec['exp057_vectors'])} exp-057 vectors carry their recorded hashes",
          flush=True)

    # ---- the 2026-07 ranking receipt, for q in the draws -------------------------------------
    rpath = E55 / "analysis/receipts/i1-daysums-rank-2026-07.csv"
    check_pin(rpath)
    Dr, cir = SC.day_matrix(SC.read_receipt(rpath),
                            np.array([f"2026-07-{d:02d}" for d in range(1, 32)]))
    lvc = comparator_levels(pan, vec)

    # ---- R7 (no exp-058 vector yet) ----------------------------------------------------------
    pf = pan[F]
    H, y, pb = pf["H"], lab[F]["y"], pf["pbase"]
    n_all, nH = len(y), int(H.sum())
    cells57 = cells_july(pan, lvc, y, COMPS57, SSE57)
    _, back57 = LF.receipt("exp057-july_loo", cells57)
    Dm7, ci7 = LF.day_matrix(back57, days)
    P7 = Dm7.sum(0)
    if int(P7[ci7["n"]]) != n_all or int(P7[ci7["n_H"]]) != nH:
        raise SystemExit("R7: the comparator receipt does not cover every July row")
    w12, w9, pts57 = 0.0, 0.0, {}
    for c in COMPS57:
        pts57[c] = {}
        for s in SP:
            q = {a: fl(b) for a, b in q_cols(P7, ci7, c, s).items()}
            ref = e57["points"][c]["steps"][s]
            w12 = max(w12, abs(q["m"] - ref["m"]))
            for key in ("A", "B", "g_C", "g_hat", "g_plain", "J", "cos", "amplitude", "alignment"):
                w9 = max(w9, abs(q[key] - ref[key]))
            pts57[c][s] = {"m_rebuilt": q["m"], "m_exp057": ref["m"],
                           "abs_diff_m": abs(q["m"] - ref["m"])}
    if w12 > TOL_12 or w9 > TOL_9:
        raise SystemExit(f"R7 (i): exp-057's points are not reproduced (m {w12:.3e}, loss {w9:.3e})")
    DR57 = Draws(pan, lvc, y, days, kinds=("P", "C", "S"))
    for idx, X, want in (((1, 2, 3), "C", "C3bag"), ((1, 2, 3), "S", "S3bag"),
                         (SEEDS10, "C", "C10bag"), (SEEDS10, "P", "P10bag"),
                         ((1, 2, 3), "P", "P3bag")):
        for L in range(4):
            if not np.array_equal(DR57.bag(X, idx, L), lvc[want][L][H]):
                raise SystemExit(f"R7: the draw route's bag {want} L{L} is not the point bag on H")
    k3_57 = {j: {"sham": DR57.sums("S", "P", t, levels=range(4)), "carrier": DR57.sums("C", "P", t)}
             for j, t in enumerate(MULTI3)}
    nday7, nHday7 = Dm7[:, ci7["n"]], Dm7[:, ci7["n_H"]]
    out57, k10_keep, e57_text = {}, {}, {}
    for L in BLOCKS:
        d = draws[L]
        W = C.mbb_weights(d["st"]["july_loo"], 31, L)
        Wr = C.mbb_weights(d["st"]["2026-07"], 31, L)
        wn, wnH = W @ nday7, W @ nHday7
        qd = {s: np.sqrt(((Wr @ Dr[:, cir[f"ss_UP_{s}"]]) / (Wr @ Dr[:, cir["n"]]))
                         / ((W @ Dm7[:, ci7[f"ss_T10_{s}"]]) / wnH)) for s in SP}
        per = {arm: {key: np.stack([W @ k3_57[j][arm][key] for j in range(len(MULTI3))],
                                   axis=1)[np.arange(B), d["msi"]] for key in k3_57[0][arm]}
               for arm in ("sham", "carrier")}
        #: the K = 10 per-draw sums are computed ONCE here and reused by R8 (vi) and by the
        #: measurement: the same generator, the same vectors and the same bag rule give the same
        #: numbers, and the byte-identity of both rebuilt receipts proves it
        k10 = None
        for b in range(B):
            sm = DR57.sums("C", "P", tuple(d["sig10"][b]))
            if k10 is None:
                k10 = {key: np.empty(B) for key in sm}
            for key, v_ in sm.items():
                k10[key][b] = W[b] @ v_
        per["carrier10"] = k10
        k10_keep[L] = k10
        cols = {"draw": np.arange(B), "k3_multiset": [MS3[j] for j in d["msi"]],
                "k10_indices": [" ".join(map(str, r_)) for r_ in d["sig10"]], "wn": wn, "wn_H": wnH}
        for s in SP:
            cols[f"q_{s}"] = qd[s]
        for c, arm in (("sham_K3", "sham"), ("carrier_K3", "carrier"), ("carrier_K10", "carrier10")):
            for s in SP:
                for st in STATS:
                    cols[f"w{st}_{c}_{s}"] = per[arm][f"{st}_{s}"]
        for Lv in range(4):
            cols[f"wdsse_sham_K3_L{Lv}"] = per["sham"][f"dsse_L{Lv}"]
        cols["wdsse_carrier_K3_L0"] = per["carrier"]["dsse_L0"]
        cols["wdsse_carrier_K10_L0"] = per["carrier10"]["dsse_L0"]
        text = draw_csv(cols)
        e57_text[f"L{L}"] = text
        out57[f"L{L}"] = cmp_draw_csv(text, E57 / "analysis/receipts" / f"draws-L{L}.csv")
        if not out57[f"L{L}"]["within_1e-9_relative"]:
            raise SystemExit(f"R7 (ii): the rebuild of exp-057's draws-L{L}.csv misses it by "
                             f"{out57[f'L{L}']['max_rel_diff']:.3e} relative")
    p57 = E57 / "analysis/receipts/daysums-july_loo.csv"
    check_pin(p57)
    b57 = LF.read_receipt(p57)
    if not (b57["day"].equals(back57["day"]) and b57["airport"].equals(back57["airport"])):
        raise SystemExit("R7 (iii): exp-057's receipt is not on this receipt's (day, airport) rows")
    shared57 = [c for c in back57.columns if c in b57.columns and c not in ("day", "airport")]
    a_, bb = back57[shared57].to_numpy("float64"), b57[shared57].to_numpy("float64")
    rel57 = float(np.max(np.abs(a_ - bb) / np.maximum(np.abs(bb), 1e-300)))
    if rel57 > TOL_9:
        raise SystemExit(f"R7 (iii): exp-057's cells differ by {rel57:.3e} relative")
    rec["R7_exp057"] = {"points_max_abs_diff_m": w12, "points_max_abs_diff_loss": w9,
                        "points": pts57, "draw_receipts": out57,
                        "daysums_shared_columns": len(shared57), "daysums_max_rel_diff": rel57}
    print(f"R7 exp-057's points rebuilt (m {w12:.1e}, loss {w9:.1e}); its three draw receipts "
          f"rebuilt (bytes identical {[v['bytes_identical'] for v in out57.values()]}); its "
          f"day-by-airport cells agree to {rel57:.1e} relative", flush=True)

    # ---- R8 ---------------------------------------------------------------------------------
    rec["exp058_vectors"] = load_exp058(vec)
    e58p = E58 / "analysis/measure.json"
    check_pin(e58p)
    e58 = json.loads(e58p.read_text())
    if e58["spec_commit"] != E58_SPEC_COMMIT or e58["standin"]:
        raise SystemExit("exp-058's measure.json is not its registered official record")
    lv58 = dict(lvc)
    for arm in ("F", "M"):
        for s_ in SEEDS3:
            lv58[f"{arm}{s_}"] = [vec[F][f"{arm}{s_}_L{L}"] for L in range(4)]
        lv58[f"{arm}3bag"] = [bag_of(pf, vec[F], arm, SEEDS3, L) for L in range(4)]
    for nm in ("F3bag", "M3bag", "C3bag", "S3bag", "P3bag", "P10bag", "C10bag", "T10"):
        for k in KS:
            if np.any((lv58[nm][k] - lv58[nm][k - 1])[~H] != 0):
                raise SystemExit(f"{nm} {STEP[k]}: a bag step moves a row outside H")
    cells58 = cells_july(pan, lv58, y, COMPS58, SSE58)
    text58, back58 = LF.receipt("exp058-july_loo", cells58)
    Dm8, ci8 = LF.day_matrix(back58, days)
    P8 = Dm8.sum(0)
    if int(P8[ci8["n"]]) != n_all or int(P8[ci8["n_H"]]) != nH:
        raise SystemExit("R8: the exp-058 receipt does not cover every July row")
    w12, w9, pts58, f8 = 0.0, 0.0, {}, {}
    for c in COMPS58:
        pts58[c] = {}
        for s in SP:
            q = {a: fl(b) for a, b in q_cols(P8, ci8, c, s).items()}
            q |= {"q": Q_POINT[s], "I": fl(imp(q["m"], Q_POINT[s])),
                  "m_minus_q": q["m"] - Q_POINT[s], "m_over_q": q["m"] / Q_POINT[s]}
            ref = e58["points"][c]["steps"][s]
            w12 = max(w12, abs(q["m"] - ref["m"]), abs(q["cos"] - ref["cos"]),
                      abs(q["I"] - ref["I"]), abs(q["m_minus_q"] - ref["m_minus_q"]),
                      abs(q["m_over_q"] - ref["m_over_q"]), abs(q["q"] - ref["q"]))
            for key in ("A", "B", "g_C", "g_hat", "g_plain", "J", "amplitude", "alignment",
                        "rms_H_carrier_s", "rms_H_plain_s"):
                w9 = max(w9, abs(q[key] - ref[key]))
            for key in R8_III_FIELDS:
                f8[key] = max(f8.get(key, 0.0), abs(q[key] - ref[key]))
            pts58[c][s] = {k_: {"rebuilt": q[k_], "exp058": ref[k_],
                                "abs_diff": abs(q[k_] - ref[k_])} for k_ in R8_III_FIELDS}
    if w12 > TOL_12 or w9 > TOL_9:
        raise SystemExit(f"R8 (iii): exp-058's points are not reproduced (m/cos {w12:.3e}, "
                         f"loss {w9:.3e})")
    w_own, own58 = 0.0, {}
    for nm in BAG_COMPS58:
        X, Y = COMPS58[nm][0], COMPS58[nm][1]
        val = fl((P8[ci8[f"sse_{Y}_L0"]] - P8[ci8[f"sse_{X}_L0"]]) / n_all)
        ref = e58["own_value_L0"]["by_arm"][nm]["value"]
        w_own = max(w_own, abs(val - ref))
        own58[nm] = {"rebuilt": val, "exp058": ref, "abs_diff": abs(val - ref),
                     "expected": E58_OWN[nm]}
    if w_own > TOL_9:
        raise SystemExit(f"R8 (iv): exp-058's L0 own values are missed by {w_own:.3e} s2/row")
    DR58 = Draws(pan, lv58, y, days, kinds=("P", "C", "S", "F", "M"))
    k3_58 = {j: {c: DR58.sums(K3_DRAW58[c], "P", t,
                              levels=range(4) if c == "sham_K3" else (0,)) for c in K3_DRAW58}
             for j, t in enumerate(MULTI3)}
    nday8, nHday8 = Dm8[:, ci8["n"]], Dm8[:, ci8["n_H"]]
    out58, paired, e58_draw_text = {}, {s: {} for s in SP}, {}
    for L in BLOCKS:
        d = draws[L]
        W = C.mbb_weights(d["st"]["july_loo"], 31, L)
        Wr = C.mbb_weights(d["st"]["2026-07"], 31, L)
        wn, wnH = W @ nday8, W @ nHday8
        qd = {s: np.sqrt(((Wr @ Dr[:, cir[f"ss_UP_{s}"]]) / (Wr @ Dr[:, cir["n"]]))
                         / ((W @ Dm8[:, ci8[f"ss_T10_{s}"]]) / wnH)) for s in SP}
        per = {c: {key: np.stack([W @ k3_58[j][c][key] for j in range(len(MULTI3))],
                                 axis=1)[np.arange(B), d["msi"]] for key in k3_58[0][c]}
               for c in K3_DRAW58}
        per["carrier_K10"] = k10_keep[L]
        cols = {"draw": np.arange(B), "k3_multiset": [MS3[j] for j in d["msi"]],
                "k10_indices": [" ".join(map(str, r_)) for r_ in d["sig10"]], "wn": wn, "wn_H": wnH}
        for s in SP:
            cols[f"q_{s}"] = qd[s]
        for c in BAG_COMPS58:
            for s in SP:
                for st in STATS:
                    cols[f"w{st}_{c}_{s}"] = per[c][f"{st}_{s}"]
        for c in BAG_COMPS58:
            for Lv in (range(4) if c == "sham_K3" else (0,)):
                cols[f"wdsse_{c}_L{Lv}"] = per[c][f"dsse_L{Lv}"]
        txt = draw_csv(cols)
        e58_draw_text[f"L{L}"] = txt
        out58[f"L{L}"] = cmp_draw_csv(txt, E58 / "analysis/receipts" / f"draws-L{L}.csv")
        if not out58[f"L{L}"]["within_1e-9_relative"]:
            raise SystemExit(f"R8 (vi): the rebuild of exp-058's draws-L{L}.csv misses it by "
                             f"{out58[f'L{L}']['max_rel_diff']:.3e} relative")
        for s in SP:
            mF = np.sqrt(per["feb_K3"][f"xx_{s}"] / per["feb_K3"][f"yy_{s}"])
            mM = np.sqrt(per["june_matched_K3"][f"xx_{s}"] / per["june_matched_K3"][f"yy_{s}"])
            got = interval(mF - mM)
            ref = e58["feb_minus_june_matched"][s]["intervals"][f"L{L}"]["day+member"]
            gday = interval(q_cols(W @ Dm8, ci8, "feb_K3", s)["m"]
                            - q_cols(W @ Dm8, ci8, "june_matched_K3", s)["m"])
            rday = e58["feb_minus_june_matched"][s]["intervals"][f"L{L}"]["day"]
            paired[s][f"L{L}"] = {
                "day+member": {"rebuilt": [got["lo"], got["hi"]], "exp058": [ref["lo"], ref["hi"]],
                               "max_abs_diff": max(abs(got["lo"] - ref["lo"]),
                                                   abs(got["hi"] - ref["hi"]))},
                "day": {"rebuilt": [gday["lo"], gday["hi"]], "exp058": [rday["lo"], rday["hi"]],
                        "max_abs_diff": max(abs(gday["lo"] - rday["lo"]),
                                            abs(gday["hi"] - rday["hi"]))}}
    w_int = max(v_[m_]["max_abs_diff"] for s in SP for v_ in paired[s].values()
                for m_ in ("day", "day+member"))
    if w_int > TOL_9:
        raise SystemExit(f"R8 (v): exp-058's paired intervals are missed by {w_int:.3e}")
    rel58 = {}
    p58 = E58 / "analysis/receipts/daysums-july_loo.csv"
    check_pin(p58)
    b58 = LF.read_receipt(p58)
    if not (b58["day"].equals(back58["day"]) and b58["airport"].equals(back58["airport"])):
        raise SystemExit("R8 (vi): exp-058's receipt is not on this receipt's (day, airport) rows")
    shared = [c for c in back58.columns if c in b58.columns and c not in ("day", "airport")]
    a_, bb = back58[shared].to_numpy("float64"), b58[shared].to_numpy("float64")
    rel58["daysums"] = {"shared_columns": len(shared),
                        "max_rel_diff": float(np.max(np.abs(a_ - bb)
                                                     / np.maximum(np.abs(bb), 1e-300)))}
    pd58 = E58 / "analysis/receipts/days-july_loo.csv"
    check_pin(pd58)
    d58 = pd.read_csv(pd58, dtype={"day": str}, float_precision="round_trip")
    shared_d = [c for c in ci8 if c in d58.columns]
    a2 = np.stack([Dm8[:, ci8[c]] for c in shared_d], axis=1)
    b2 = d58[shared_d].to_numpy("float64")
    rel58["days"] = {"shared_columns": len(shared_d),
                     "max_rel_diff": float(np.max(np.abs(a2 - b2)
                                                  / np.maximum(np.abs(b2), 1e-300)))}
    for k_, v_ in rel58.items():
        if v_["max_rel_diff"] > TOL_9:
            raise SystemExit(f"R8 (vi): exp-058's {k_} cells differ by {v_['max_rel_diff']:.3e}")
    rec["R8_exp058"] = {"points_max_abs_diff_m_cos": w12, "points_max_abs_diff_loss": w9,
                        "points": pts58, "points_max_abs_diff_by_field": f8,
                        "points_fields_compared": list(R8_III_FIELDS),
                        "points_fields_rebuilt": [k for k in R8_III_FIELDS if k != "q"],
                        "q_not_rebuilt": "q is Q15's pinned registered constant Q_POINT; the "
                                         "comparison shows exp-058's stored q equals that constant "
                                         "and is not a rebuild of q from Q15's ranks and fold "
                                         "receipts", "own_value_L0": own58,
                        "own_value_max_abs_diff": w_own, "paired_intervals": paired,
                        "paired_intervals_max_abs_diff": w_int, "draw_receipts": out58,
                        "receipt_cells": rel58}
    print(f"R8 exp-058's points rebuilt (m/cos {w12:.1e}, loss {w9:.1e}); its five L0 own values to "
          f"{w_own:.1e}; its paired intervals to {w_int:.1e}; its draw receipts bytes identical "
          f"{[v['bytes_identical'] for v in out58.values()]}; its day cells to "
          f"{rel58['daysums']['max_rel_diff']:.1e} relative", flush=True)

    # ---- R8 (vii): every leaf of both recorded population blocks -----------------------------
    lev = {}
    for f in C.FOLDS:
        pff, v = pan[f], vec[f]
        lv = {nm: [v[f"{nm}_L{L}"] for L in range(4)]
              for nm in [f"P{s}" for s in SEEDS10] + ["P3bag", "T10"]}
        lv["P10bag"] = [bag_of(pff, v, "P", SEEDS10, L) for L in range(4)]
        if f == F:
            lv |= lv58
        for nm, x in lv.items():
            if nm.endswith("bag") or nm == "T10":
                for k in KS:
                    if np.any((x[k] - x[k - 1])[~pff["H"]] != 0):
                        raise SystemExit(f"{f} {nm} {STEP[k]}: a bag step moves a row outside H")
        lev[f] = lv
    pops = populations(pan, lab, lev)
    rec["populations"] = pops
    rec["R8_vii_populations"] = {}
    for tag, path in (("exp058", e58p), ("exp057", e57p)):
        ref = json.loads(path.read_text())["populations"]
        rec["R8_vii_populations"][tag] = compare_leaves(ref, pops, tag)
        print(f"R8 (vii) {tag}: {rec['R8_vii_populations'][tag]['values_compared']} recorded leaves, "
              f"max |diff| {rec['R8_vii_populations'][tag]['max_abs_diff']:.1e} s", flush=True)
    pw = 0.0
    for f in C.FOLDS:
        for nm in (("P3bag", "C3bag") if f == F else ("P3bag",)):
            for Lv in range(4):
                for k in pops[f][f"{nm}_L{Lv}"]:
                    cell = lfj["populations"][f][f"{nm}_L{Lv}"].get(k)
                    ref = cell.get("rmse_s") if isinstance(cell, dict) else None
                    if ref is not None:
                        pw = max(pw, abs(ref - pops[f][f"{nm}_L{Lv}"][k]))
    rec["checks"]["populations_vs_exp055_max_abs_diff_s"] = pw
    if pw > TOL_9:
        raise SystemExit(f"a reproduced exp-055 population differs by {pw:.3e} s")

    # ---- Part B: the separate reproductions --------------------------------------------------
    plan = json.loads((ANALYSIS / "plan.json").read_text())
    ver = plan["grid"]["verified_against_exp059_registered_rows"]
    if not all(v["equals_exp059_registered_row"] for v in ver.values()):
        raise SystemExit("analysis/plan.json does not carry the registered grid rows")
    if plan["grid"]["registered_N_sequence"] != list(MAN["reproduction_targets"]
                                                     ["grid_N_sequence_k1_to_k23"]):
        raise SystemExit("the 23-row N sequence in the plan is not the registered one")
    OW = Own(pan, vec[F], y, days)
    if OW.n != 190_713:
        raise SystemExit(f"July has {OW.n} fold rows, not 190,713")
    p3 = OW.bag("P")
    p10b = OW.bag("P", SEEDS10)
    if not np.array_equal(p3, vec[F]["P3bag_L0"]):
        raise SystemExit("the rebuilt plain K = 3 bag is not exp-055's pinned bag")
    if float(np.max(np.abs(p10b - vec[F]["T10_L0"]))) > TOL_9:
        raise SystemExit("the rebuilt plain K = 10 bag is not Q15's pinned head")
    OW.cells("P3bag", p3)
    OW.cells("P10bag", p10b)
    for X in ("F", "M", "C", "S"):
        OW.cells(f"{X}3bag", OW.bag(X))
    OW.cells("C10bag", OW.bag("C", SEEDS10))
    e59_info = load_exp059_L0(vec)
    for k in E59_K:
        arm = f"G{k:02d}"
        OW.cells(f"{arm}3bag", OW.bag(arm))
        for s in SEEDS3:
            OW.cells(f"{arm}{s}", vec[F][f"{arm}{s}_L0"])
    for s in SEEDS3:
        OW.cells(f"P{s}", vec[F][f"P{s}_L0"])
    names = ["P3bag", "P10bag", "F3bag", "M3bag", "C3bag", "S3bag", "C10bag"] \
        + [f"P{s}" for s in SEEDS3] \
        + [f"G{k:02d}3bag" for k in E59_K] + [f"G{k:02d}{s}" for k in E59_K for s in SEEDS3]
    Dmo, cio, Po = OW.totals(sorted(names))
    zero, _ = OW.value("P3bag", "P3bag", Dmo, cio, Po)
    if zero != 0.0:
        raise SystemExit("the V(0) identity is not exactly zero")
    vF, _ = OW.value("F3bag", "P3bag", Dmo, cio, Po)
    vM, _ = OW.value("M3bag", "P3bag", Dmo, cio, Po)
    if abs(vF - V_STAR) > TOL_9 or abs(vM - V_TOP) > TOL_9:
        raise SystemExit(f"the frozen anchors do not reproduce: February {vF!r}, matched June {vM!r}")
    lad2 = json.loads((E59 / "analysis/ladder2.json").read_text())
    e59V, worstV = {}, 0.0
    for k in E59_K:
        arm = f"G{k:02d}"
        val, dsum = OW.value(f"{arm}3bag", "P3bag", Dmo, cio, Po)
        ref = lad2["arms"][arm]["V"]
        worstV = max(worstV, abs(val - ref))
        mem = {}
        for s in SEEDS3:
            mv, mds = OW.value(f"{arm}{s}", f"P{s}", Dmo, cio, Po)
            mref = lad2["arms"][arm]["members"][f"s{s}"]["value"]
            worstV = max(worstV, abs(mv - mref))
            mem[f"s{s}"] = {"rebuilt": mv, "exp059": mref, "abs_diff": abs(mv - mref)}
        W3 = C.mbb_weights(draws[3]["st"]["july_loo"], 31, 3)
        iv = interval((W3 @ dsum) / (W3 @ OW.day_n))
        e59V[arm] = {"k": k, "N": lad2["arms"][arm]["N"], "rebuilt": val, "exp059": ref,
                     "abs_diff": abs(val - ref), "members": mem,
                     "three_day_day_interval_rebuilt": [iv["lo"], iv["hi"]],
                     "three_day_day_interval_exp059": [lad2["arms"][arm]["intervals"]["L3"]["lo"],
                                                       lad2["arms"][arm]["intervals"]["L3"]["hi"]]}
    if worstV > TOL_12:
        raise SystemExit(f"exp-059's fitted V at k = 4, 8, 12 are missed by {worstV:.3e} s2/row")
    lf_cells = LF.read_receipt(E55 / "analysis/receipts/i2-daysums-july_loo.csv")
    check_pin(E55 / "analysis/receipts/i2-daysums-july_loo.csv")
    worst_cell = 0.0
    for s in SP:
        for st in STATS:
            a_ = back58[f"{st}_carrier_K3_{s}"].to_numpy("float64")
            b_ = lf_cells[f"{st}_bag123_{s}"].to_numpy("float64")
            worst_cell = max(worst_cell, float(np.max(np.abs(a_ - b_))))
    if worst_cell > TOL_9:
        raise SystemExit(f"the carrier K = 3 cells differ from exp-055's i2 receipt by {worst_cell:.3e}")
    rec["separate_reproductions"] = {
        "B1_grid": {"from": "analysis/plan.json", "k24": ver["24"], "verified_rows": ver,
                    "N_sequence_reproduced": plan["grid"]["N_sequence_reproduced"]},
        "B2_anchors": {"V_at_0_identity": zero, "V_at_60777_rebuilt": vM, "V_at_60777_registered": V_TOP,
                       "abs_diff": abs(vM - V_TOP),
                       "why_V_at_0_is_an_identity": "an empty block leaves the plain member's own L0 "
                                                    "fit, so X3 = P3 and V = 0 by construction; "
                                                    "SSE(P3) - SSE(P3) = 0 is computed here, no fit "
                                                    "exists"},
        "B3_february": {"V_star_rebuilt": vF, "V_star_registered": V_STAR,
                        "abs_diff": abs(vF - V_STAR), "band_B": list(BAND),
                        "band_rule": "V* +/- 10 %",
                        "band_reproduced_from_V_star": [V_STAR * 0.9, V_STAR * 1.1],
                        "band_matches_registered": bool(abs(V_STAR * 0.9 - BAND[0]) <= 1e-12
                                                        and abs(V_STAR * 1.1 - BAND[1]) <= 1e-12)},
        "B4_exp059_fitted_V": {"arms": e59V, "max_abs_diff_s2_row": worstV,
                               "vectors": e59_info,
                               "why": "exp-059's fitted vectors are read here only to reproduce its "
                                      "recorded V; no prediction of this experiment is built from "
                                      "an exp-059 vector"},
        "B5_day_cells": {"exp058_daysums": rel58["daysums"], "exp058_days": rel58["days"],
                         "exp057_daysums": {"shared_columns": len(shared57),
                                            "max_rel_diff": rel57},
                         "exp055_i2_carrier_K3_cells_max_abs_diff": worst_cell},
    }
    if not rec["separate_reproductions"]["B3_february"]["band_matches_registered"]:
        raise SystemExit("the band B is not V* +/- 10 %")
    print(f"B1 the grid rows and the 23-row sequence carry exp-059's registered values; "
          f"B2 V(0) = {zero}, V(60,777) = {vM:+.10f}; B3 V* = {vF:+.10f}, band {BAND}; "
          f"B4 exp-059's fitted V max |diff| {worstV:.1e} s2/row; B5 day cells clean", flush=True)
    for k in E59_K:
        a = rec["separate_reproductions"]["B4_exp059_fitted_V"]["arms"][f"G{k:02d}"]
        print(f"   k={k} (N={a['N']:,}): V = {a['rebuilt']:+.4f} s2/row (exp-059 {a['exp059']:+.4f})",
              flush=True)

    # ---- the source check and the receipts ---------------------------------------------------
    sc = source_check()
    rec["source_check"] = {k: v for k, v in sc.items() if k != "files"} | {"files_opened": sc["files"]}
    if not sc["clean"]:
        raise SystemExit(f"the source check fails: {json.dumps(sc, indent=1)}")
    opened62 = sorted(r for r in OPENED if r.startswith(E62_WORK))
    rec["exp062_vectors_opened"] = opened62
    if opened62:
        raise SystemExit("the reproduction pass opened an exp-062 vector")
    rec["receipts"] = {}

    def put_text(p: Path, t: str) -> None:
        NO.refuse_existing([p], overwrite=False)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(t)
        rec["receipts"][str(p.relative_to(ROOT))] = sha_file(p)
    for L in BLOCKS:
        put_text(REC_DIR / f"exp057-draws-L{L}.csv", e57_text[f"L{L}"])
        put_text(REC_DIR / f"exp058-draws-L{L}.csv", e58_draw_text[f"L{L}"])
    put_text(REC_DIR / "exp058-daysums-july_loo.csv", text58)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["seconds_process_cpu"] = round(time.process_time() - c0, 1)
    C.write_json(OUT, rec)
    print(f"written {OUT.relative_to(ROOT)} ({rec['seconds']} s wall, "
          f"{rec['seconds_process_cpu']} s process cpu); no exp-062 vector was opened", flush=True)


if __name__ == "__main__":
    main()
