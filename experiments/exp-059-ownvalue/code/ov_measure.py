"""exp-059 ownvalue: every registered measurement (SPEC sections 6, 7, 9).  ZERO FIT.  READS LABELS.

Order (SPEC 6): the reproductions come first, and no exp-059 vector is opened until all of them pass.
  R1  exp-055 `originvec.load()` re-verifies every exp-055 ordered pin; the labels equal exp-051's
      OOF `y_true` bit for bit on every fold.
  R2  exp-051's stored July gains, with exp-051's arithmetic, from `carrier.json` read through a
      reader that records every key it gives and refuses any other.
  R3/R4  exp-055's `screen.py` and `lossfals.py`, unchanged, run with `--scratch` into
      `artifacts/exp-059/repro/`; their registered fields must equal the committed records, and the
      whole records, receipt hashes and OOF hashes are compared and reported.  This script's own
      arithmetic must reproduce exp-055's July points before anything new is opened.
  R5  the regenerated draws carry exp-055 `screen.json`'s `draw_sha256` and `multiset_counts` and
      exp-057 `measure.json`'s `k10_member_draw_sha256_unsorted`.
  R6  `P10 = bag(P1..P10)` equals the saved K = 10 heads; `C3` and `P3` rebuild exp-055's pinned
      bags; `S1` carries exp-051's recorded hashes and `S2`, `S3`, `Cx4..Cx10` exp-057's.
  R7  exp-057's rule-S and rule-Z points are rebuilt from the pinned vectors by this script's own
      arithmetic; exp-057's three draw receipts are rebuilt column for column; exp-057's
      day-by-airport receipt is read back and compared cell for cell.
  R8  exp-058's own record: its 24 vectors by hash, its February and matched-June points, its five
      L0 own values, its paired m_Feb - m_June intervals, its three draw receipts rebuilt over its
      own column set, its day-by-airport and day receipts cell for cell, and its populations.
Then the new June-ladder vectors (sha-checked against this experiment's fit records), their bags,
the label-free movement statistics, the loss equations as diagnostics, the moving-block intervals
with matched member draws, the paired `m_Feb - m_June*` interval, SPEC 8's registered reading,
COMMON's populations, the OOF files and the receipts.

The script refuses to run unless the ladder record names a `June*` whose vectors exist (SPEC 5.4).

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-059-ownvalue/code/ov_measure.py
    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-059-ownvalue/code/ov_measure.py --standin DIR
      (the SPEC 12 dry run: the ladder arms take existing vectors, no file under
       artifacts/exp-059/work may be opened, every output goes under DIR and is discarded)
"""
from __future__ import annotations

import argparse
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
E59_ART = "artifacts/exp-059/"
E59_EXP = "experiments/exp-059-ownvalue/"
WORK_REL = "artifacts/exp-059/work"
PY = str(ROOT / ".venv/bin/python")

ap_ = argparse.ArgumentParser()
ap_.add_argument("--standin", help="dry run: stand-in vectors, every output under this directory")
ap_.add_argument("--cite-r34", default=None, metavar="RECORD",
                 help="carry R3/R4 from an earlier reproduction record instead of running exp-055's "
                      "screen.py and lossfals.py again; only their subprocess time is saved and the "
                      "cited record must already hold the result")
ap_.add_argument("--repro-only", action="store_true",
                 help="SPEC 5.5: the ladder gave no June*, so run R1-R8 -- the reproductions of "
                      "exp-051's, exp-055's, exp-057's and exp-058's own recorded numbers -- and stop "
                      "before any exp-059 vector is opened. No new m, loss, subgroup or step "
                      "diagnostic of any new vector is computed (SPEC 5.4)")
ARGS = ap_.parse_args()
STANDIN = ARGS.standin is not None
REPRO_ONLY = ARGS.repro_only
if REPRO_ONLY and STANDIN:
    raise SystemExit("--repro-only opens no new vector, so it has no dry run")

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
        if STANDIN and rel.startswith(WORK_REL):
            raise PermissionError(f"stand-in run: {rel} may not be opened")
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

#: SPEC 2.3: the measurement imports only exp-055's closure (exp-051's and exp-055's both hold a
#: module named `common`), and every module of this experiment is named `oc_*`.
sys.path.insert(0, str(E55 / "code"))
import common as C                                                    # noqa: E402  (exp-055)
import lossfals as LF                                                 # noqa: E402
import originvec as V                                                 # noqa: E402
import screen as SC                                                   # noqa: E402

for _name, _want in (("common", E55 / "code/common.py"), ("originvec", E55 / "code/originvec.py"),
                     ("screen", E55 / "code/screen.py"), ("lossfals", E55 / "code/lossfals.py")):
    if Path(sys.modules[_name].__file__).resolve() != _want:
        raise SystemExit(f"module {_name} resolves to {sys.modules[_name].__file__} (module name collision)")
if "carrier" in sys.modules or "memberspread" in sys.modules:
    raise SystemExit("exp-051's closure is imported in the measurement process")

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
#: SPEC 7 (exp-055 screen.json q["july_loo->2026-07"][v]["point"] and ["intervals"]["L3"])
Q_POINT = {"v20": 0.7541214069685656, "v21": 0.7872337829612198, "v22": 0.5869686712745045}
#: SPEC 6 R8 (iii): every registered point field of exp-058's arms (review round 1, finding 1)
R8_III_FIELDS = ("m", "q", "m_minus_q", "m_over_q", "I", "cos", "rms_H_carrier_s", "rms_H_plain_s",
                 "A", "B", "g_C", "g_hat", "g_plain", "J", "amplitude", "alignment")
Q_L3 = {"v20": (0.7030750838597688, 0.7941267683291252), "v21": (0.7364214976003001, 0.8322645424736729),
        "v22": (0.555629780110832, 0.610847845536199)}
#: SPEC 8: the registered reading -- the practical amplitude-equivalence margin
MARGIN = 0.03
READ_STEPS = ("v20", "v21")
#: SPEC 5.1: the frozen own-value constants
V_STAR = 168.58928050561724
BAND = (151.7303524550555, 185.44820855617897)
SCREEN_SHA = "081c39ed4789c5fc4851fe61808bf1cc28b13d5eee1e1e974a768faa17721ae9"
LOSSFALS_SHA = "1eecc0b52e517a5f7749c0a75ce6b9529cb268b708f163402945ce337e64a52f"
CARRIER_SHA = "dc49aa89816140365111aa601965ba6bcb6b4735340bf64f8f7aa4c92dbe25b4"
SPEC_COMMIT = "49db4157d8bc43edde6c2b8386e90b6f58929578"
E57_SPEC_COMMIT = "95d82b6611e2768e2a25fd596bd113f168eb307e"
E58_SPEC_COMMIT = "22e2a6b91f94040d1999d483ae4bf95565ca1f72"
E58_CODE_SHA = "9a970bacf95c05dbf923d7a0e7260e93a69fa90c69553ceebe6e09928338dd11"
TOL_12, TOL_9 = 1e-12, 1e-9
ARM_MONTH = {"F": "2025-02", "M": "2025-06"}
ARM_NAME = {"F": "february", "M": "june_matched"}
#: SPEC 12: the stand-in members of the dry run (one existing triple per ladder arm)
STANDIN_POOL = ("C", "P", "S")

if STANDIN:
    BASE = Path(ARGS.standin).resolve()
    if str(BASE).startswith(ROOT_S):
        raise SystemExit("the stand-in directory must lie outside the repository")
    ANALYSIS, OOFD, ART = BASE / "analysis", BASE / "oof", BASE / "artifacts"
else:
    ANALYSIS, OOFD, ART = EXP / "analysis", EXP / "oof", ROOT / "artifacts/exp-059"
OUT = ANALYSIS / "measure.json"
#: a repeat reproduction pass writes its own record beside the first: records are write-once and a
#: correction is a new file, never an overwrite (COMMON.md)
REPRO_OUT = ANALYSIS / ("repro.json" if not (ANALYSIS / "repro.json").exists() else "repro-r1.json")
REC_DIR = ANALYSIS / "receipts"
REPRO = ART / "repro"

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
    base = json.loads((EXP / "analysis" / f"{name}.json").read_text())
    cur, seen = base, {name}
    for r1 in sorted((EXP / "analysis").glob(f"{name}-r*.json")):
        cand = json.loads(r1.read_text())
        sup = cand.get("supersedes") or []
        if not any(str(ROOT / s["plan"]) == str(EXP / "analysis" / f"{name}.json") for s in sup):
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
PLAN3 = load_plan("plan3") if (EXP / "analysis/plan3.json").exists() else None
LADDER = json.loads((EXP / "analysis/ladder2.json").read_text()) if (EXP / "analysis/ladder2.json").exists() \
    else json.loads((EXP / "analysis/ladder.json").read_text())
if not LADDER.get("june_star") and not REPRO_ONLY:
    raise SystemExit("the ladder record names no June*: SPEC 5.4 holds every new m behind it")
if REPRO_ONLY and LADDER.get("june_star"):
    raise SystemExit("--repro-only is the SPEC 5.5 route for a ladder without June*; this one has one")
JUNE_STAR = LADDER["june_star"]["arm"] if LADDER.get("june_star") else None
LADDER_ARMS = [f"G{k:02d}" for k in sorted(int(a[1:]) for a in LADDER["arms"])]
ARM_K = {a: LADDER["arms"][a]["k"] for a in LADDER_ARMS}
ARM_N = {a: LADDER["arms"][a]["N"] for a in LADDER_ARMS}
if JUNE_STAR is not None and JUNE_STAR not in LADDER_ARMS:
    raise SystemExit(f"June* {JUNE_STAR} has no ladder arm record")
STANDIN_SRC = {a: STANDIN_POOL[i % len(STANDIN_POOL)] for i, a in enumerate(LADDER_ARMS)}

# ------------------------------------------------------------------------------ the comparisons --

#: SPEC 7.1: name -> (carrier X, plain Y, K, role).  Every comparison uses the same ordered July H
#: rows and, at K = 3, the same plain K = 3 denominator P3.
#: exp-058's own comparison set, in exp-058's order (its receipts are rebuilt over exactly these)
COMPS58 = {"feb_K3": ("F3bag", "P3bag", 3, "exp-058: the K = 3 February bag"),
           "june_matched_K3": ("M3bag", "P3bag", 3, "exp-058: the K = 3 matched June bag (N = 60,777)"),
           "carrier_K3": ("C3bag", "P3bag", 3, "comparator: exp-051's full June carrier (exp-055's point)"),
           "sham_K3": ("S3bag", "P3bag", 3, "comparator: exp-057's K = 3 sham"),
           "carrier_K10": ("C10bag", "P10bag", 10, "comparator: exp-057's K = 10 June carrier")}
COMPS58 |= {f"feb_s{s}": (f"F{s}", f"P{s}", 1, "February member beside the K = 3 bag") for s in SEEDS3}
COMPS58 |= {f"june_matched_s{s}": (f"M{s}", f"P{s}", 1, "matched June member beside the bag") for s in SEEDS3}
COMPS58 |= {f"carrier_s{s}": (f"C{s}", f"P{s}", 1, "exp-051 June member (comparator)") for s in SEEDS3}
COMPS58 |= {f"sham_s{s}": (f"S{s}", f"P{s}", 1, "exp-057 sham member (comparator)") for s in SEEDS3}
BAG_COMPS58 = ("feb_K3", "june_matched_K3", "carrier_K3", "sham_K3", "carrier_K10")
K3_DRAW58 = {"feb_K3": "F", "june_matched_K3": "M", "carrier_K3": "C", "sham_K3": "S"}
SSE58 = (["P3bag", "F3bag", "M3bag", "C3bag", "S3bag", "P10bag", "C10bag"]
         + [f"{nm}{s}" for nm in ("P", "F", "M", "C", "S") for s in SEEDS3])

#: SPEC 7.1: this experiment's comparisons.  Every K = 3 comparison uses the same plain K = 3
#: denominator P3 on the same ordered July H rows.
LAD = {a: f"ladder_{a}_K3" for a in LADDER_ARMS}
COMPS = dict(COMPS58)
COMPS |= {LAD[a]: (f"{a}3bag", "P3bag", 3,
                   f"June ladder k = {ARM_K[a]} (N = {ARM_N[a]:,})"
                   + ("  -- June*" if a == JUNE_STAR else "")) for a in LADDER_ARMS}
COMPS |= {f"ladder_{a}_s{s}": (f"{a}{s}", f"P{s}", 1, f"ladder k = {ARM_K[a]} member {s}")
          for a in LADDER_ARMS for s in SEEDS3}
BAG_COMPS = tuple(BAG_COMPS58) + tuple(LAD[a] for a in LADDER_ARMS)
#: the K = 3 bag comparisons that share one member triple in a draw (so their difference is paired)
K3_DRAW = dict(K3_DRAW58) | {LAD[a]: a for a in LADDER_ARMS}
SSE_NAMES = SSE58 + [f"{a}3bag" for a in LADDER_ARMS] + [f"{a}{s}" for a in LADDER_ARMS for s in SEEDS3]
JS_COMP = LAD[JUNE_STAR] if JUNE_STAR is not None else None
#: exp-057's own comparison set and SSE names, for R7 (its receipts are rebuilt column for column)
COMPS57 = {c: COMPS[c] for c in ("sham_K3", "carrier_K3", "carrier_K10")}
SSE57 = ["P3bag", "S3bag", "C3bag", "P10bag", "C10bag", "S1", "S2", "S3", "P1", "P2", "P3"]


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


def interval(x) -> dict:
    return LF.interval(x)


def quantities(n, nH, ry, yy, rx, xx, xy) -> dict:
    """SPEC 7.1-7.2 (exp-055 `lossfals.quantities`, plus the best amplitude and the rms)."""
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
            "alignment_direction": 2 * m * (cos - 1) * A_, "alignment_orthogonal": 2 * (rx - (xy / yy) * ry) / n,
            "m_star": A_ / Bq, "best_amplitude_value": A_ ** 2 / Bq, "g_C_minus_best": gC - A_ ** 2 / Bq,
            "rms_H_carrier_s": np.sqrt(xx / nH), "rms_H_plain_s": np.sqrt(yy / nH)}


def q_cols(S, ci, c: str, s: str) -> dict:
    return quantities(S[..., ci["n"]], S[..., ci["n_H"]], *(S[..., ci[f"{st}_{c}_{s}"]] for st in STATS))


def imp(m, q):
    return np.abs(1 - q) - np.abs(m - q)


def fl(x):
    return float(x)


def band(v, lo, hi) -> dict:
    return {"point": fl(v), "band": [lo, hi], "inside": bool(lo <= v <= hi)}


# ------------------------------------------------------------------------------ reproductions --

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
    """exp-051's gains with exp-051's arithmetic (exp-055 SPEC 10.2's keys of carrier.json only)."""
    p = C.E51 / "analysis/carrier.json"
    if check59(p) != CARRIER_SHA:
        raise SystemExit("carrier.json is not the pinned record")
    allowed = set()
    for s in SP:
        allowed |= {f"steps.{s}.july_s2_per_row.C3", f"steps.{s}.july_s2_per_row.P3",
                    f"steps.{s}.sham_K1.july_s2.S", f"steps.{s}.sham_K1.july_s2.P1",
                    f"steps.{s}.january_s2_per_row_both_arms", f"steps.{s}.december_s2_per_row_both_arms"}
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
            return LF.mse51(y, pb, v[f"{nm}_L{k - 1}"], codes) - LF.mse51(y, pb, v[f"{nm}_L{k}"], codes)
        for k in KS:
            s = STEP[k]
            if f == F:
                want = {"C3bag": car[s]["july_s2_per_row"]["C3"], "P3bag": car[s]["july_s2_per_row"]["P3"],
                        "S1": car[s]["sham_K1"]["july_s2"]["S"], "P1:sham": car[s]["sham_K1"]["july_s2"]["P1"]}
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
    return {"tolerance_s2_per_row": TOL_9, "values": out, "max_abs_diff": worst, "keys_read": sorted(root.log),
            "n_keys": len(root.log), "other_keys_read": 0}


def jdiff(a, b, path="", skip=("seconds",)) -> tuple[list, float, float, int]:
    """Structural differences, the largest absolute and relative float differences, and the number
    of float fields that differ, between two JSON values."""
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
    """exp-055's screen.py and lossfals.py, unchanged, into a scratch directory; SPEC 6 R3 and R4."""
    env = dict(os.environ, OMP_NUM_THREADS="3")
    runs = (("screen.py", "screen", "screen.json", E55 / "analysis/screen.json", SCREEN_SHA),
            ("lossfals.py", "lossfals", "analysis/lossfals.json", E55 / "analysis/lossfals.json", LOSSFALS_SHA))
    out, got = {"runs": {}}, {}
    for script, sub, rel_out, want_path, want_sha in runs:
        d = REPRO / sub
        if d.exists():
            raise SystemExit(f"{d} exists; the reproduction directory is write-once")
        d.mkdir(parents=True)
        t0 = time.time()
        res = subprocess.run([PY, str(E55 / "code" / script), "--scratch", str(d)], cwd=ROOT, env=env,
                             capture_output=True, text=True)
        (d / "run.log").write_text(res.stdout + "\n--- stderr ---\n" + res.stderr)
        if res.returncode != 0:
            raise SystemExit(f"R3/R4: exp-055 {script} failed in the scratch run:\n{res.stderr[-2000:]}")
        if check59(want_path) != want_sha:
            raise SystemExit(f"{want_path} is not the pinned record")
        g, w = json.loads((d / rel_out).read_text()), json.loads(want_path.read_text())
        got[sub] = (g, w)
        diffs, wa, wr, nd = jdiff(g, w, skip=("seconds", "receipts", "oof"))
        rsha = {Path(k).name: v for k, v in g["receipts"].items()}
        wsha = {Path(k).name: v for k, v in w["receipts"].items()}
        oofeq = ({k: g["oof"][k] == w["oof"][k] for k in ("sha256", "rows", "columns", "ordered_sha256")}
                 if "oof" in w else None)
        out["runs"][sub] = {"script": f"experiments/exp-055-originmove/code/{script}", "scratch": str(d),
                            "seconds": round(time.time() - t0, 1),
                            "whole_record": {"structural_diffs": diffs[:20], "n_structural_diffs": len(diffs),
                                             "float_fields_differing": nd, "max_abs_float_diff": wa,
                                             "max_rel_float_diff": wr},
                            "receipt_hashes_equal": rsha == wsha, "oof_equal": oofeq,
                            "stdout_tail": res.stdout.strip().splitlines()[-2:]}
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
                reg.append((f"R3 q july {s} L{L} {e}", sg["q"]["july_loo->2026-07"][s]["intervals"][f"L{L}"][e],
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
            raise SystemExit(f"{nm}: the scratch run gives {a!r}, the committed record {b!r} (|diff| {d:.3e})")
    out["registered_fields"] = {"n": len(reg), "max_abs_diff_fields_at_1e-12": worst[TOL_12],
                                "max_abs_diff_fields_at_1e-9": worst[TOL_9], "all_within_tolerance": True}
    return out


def own_points(pan: dict, vec: dict, lab: dict, scr: dict, lfj: dict) -> dict:
    """This script's arithmetic on exp-055's July points, from the pinned vectors (no new vector)."""
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
            dx, dy = v[f"{X}_L{k}"] - v[f"{X}_L{k - 1}"], v[f"{Y}_L{k}"] - v[f"{Y}_L{k - 1}"]
            r = y - pb - v[f"{X}_L{k - 1}"]
            q = quantities(n, H.sum(), np.sum(r[H] * dy[H]), np.sum(dy[H] ** 2), np.sum(r[H] * dx[H]),
                           np.sum(dx[H] ** 2), np.sum(dx[H] * dy[H]))
            ref = lfj["points"][F][pt]["steps"][s]
            for key in ("A", "B", "g_C", "g_hat", "g_plain", "J", "amplitude", "alignment"):
                w9 = max(w9, abs(float(q[key]) - ref[key]))
            sref = scr["arms"][arm]["steps"][s]
            w12 = max(w12, abs(float(q["m"]) - sref["m"]), abs(float(imp(q["m"], Q_POINT[s])) - sref["I"]),
                      abs(float(q["cos"]) - sref["cos"]), abs(float(q["m"]) - ref["m"]))
    if w9 > TOL_9 or w12 > TOL_12:
        raise SystemExit(f"this script's arithmetic misses exp-055's points: loss {w9:.3e}, m/I/cos {w12:.3e}")
    return {"points": list(pts), "loss_max_abs_diff_s2": w9, "m_I_cos_max_abs_diff": w12}


# ------------------------------------------------------------------------------ vectors ---------

def load_exp057(vec: dict) -> dict:
    """exp-057's 36 July vectors: sham members 2 and 3, carrier members 4-10 (SPEC 5 `Cx`)."""
    info = {}
    for kind, seeds in (("S", (2, 3)), ("C", range(4, 11))):
        for s in seeds:
            for L in range(4):
                tag = f"{kind}-s{s}-L{L}"
                rp = E57 / "analysis/fits" / f"{tag}.json"
                check59(rp)
                rec = json.loads(rp.read_text())
                if rec.get("spec_commit") != E57_SPEC_COMMIT:
                    raise SystemExit(f"exp-057 {tag}: the record is not exp-057's registration")
                p = ROOT / rec["vector"]
                if check59(p) != rec["vector_sha256"]:
                    raise SystemExit(f"exp-057 {tag}: the vector is not the recorded bytes")
                z = np.load(p)
                if list(z.keys()) != [F]:
                    raise SystemExit(f"exp-057 {tag}: the vector holds {list(z.keys())}")
                vec[F][f"{kind}{s}_L{L}"] = z[F].astype("float64")
                info[tag] = {"vector": rec["vector"], "sha256": rec["vector_sha256"]}
    return info


def load_exp058(vec: dict) -> dict:
    """exp-058's 24 February and matched-June vectors, sha-checked against its fit records (R8 ii).
    These are frozen sources, not new vectors: exp-058 measured and published every statistic of
    them, and R8 reproduces those statistics before this experiment interprets anything."""
    info = {}
    e58plan = json.loads((E58 / "analysis/plan.json").read_text())
    check59(E58 / "analysis/plan.json")
    for arm in ("F", "M"):
        bsha = e58plan["match"]["months"][ARM_MONTH[arm]]["selection"]["ids_sha256_block_order"]
        for s in SEEDS3:
            for L in range(4):
                tag = f"{arm}-s{s}-L{L}"
                rp = E58 / "analysis/fits" / f"{tag}.json"
                check59(rp)
                rec = json.loads(rp.read_text())
                if rec.get("spec_commit") != E58_SPEC_COMMIT or rec.get("code_sha256") != E58_CODE_SHA:
                    raise SystemExit(f"exp-058 {tag}: not exp-058's registration or plan code")
                if (rec["arm"] != arm or rec["seed"] != s or rec["level"] != L
                        or rec["month"] != ARM_MONTH[arm] or rec["block_rows"] != 60_777
                        or rec["block_ids_sha256_block_order"] != bsha):
                    raise SystemExit(f"exp-058 {tag}: the record does not identify this arm, seed and level")
                pth = ROOT / rec["vector"]
                if check59(pth) != rec["vector_sha256"]:
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


def load_new(vec: dict) -> dict:
    """The new June-ladder vectors (or, in the dry run, their stand-ins).  SPEC 6, last item."""
    info = {}
    for arm in LADDER_ARMS:
        k = ARM_K[arm]
        plan = PLAN if PLAN3 is None or arm not in PLAN3["arms"] else PLAN3
        grow = PLAN["grid"]["rows"][str(k)]
        for s in SEEDS3:
            for L in range(4):
                tag = f"{arm}-s{s}-L{L}"
                if STANDIN:
                    src = f"{STANDIN_SRC[arm]}{s}_L{L}"
                    vec[F][f"{arm}{s}_L{L}"] = vec[F][src].copy()
                    info[tag] = {"standin_for": src}
                    continue
                rec = json.loads((EXP / "analysis/fits" / f"{tag}.json").read_text())
                if rec["code_sha256"] not in ok_codes(plan) or rec.get("spec_commit") != SPEC_COMMIT:
                    raise SystemExit(f"{tag}: the record's code or registration is not the plan's")
                if (rec["arm"] != arm or rec["k"] != k or rec["seed"] != s or rec["level"] != L
                        or rec["month"] != "2025-06"):
                    raise SystemExit(f"{tag}: the record does not identify this arm, seed and level")
                if (rec["block_rows"] != grow["block_rows"]
                        or rec["block_ids_sha256_block_order"] != grow["ids_sha256_block_order"]
                        or rec["block_count_vector_sha256"] != grow["count_vector_sha256"]):
                    raise SystemExit(f"{tag}: the block is not the registered k = {k} grid row")
                pth = ROOT / rec["vector"]
                if sha_file(pth) != rec["vector_sha256"]:
                    raise SystemExit(f"{tag}: the vector is not the recorded bytes")
                z = np.load(pth)
                if list(z.keys()) != [F]:
                    raise SystemExit(f"{tag}: the vector holds {list(z.keys())}")
                vec[F][f"{arm}{s}_L{L}"] = z[F].astype("float64")
                info[tag] = {"vector": rec["vector"], "sha256": rec["vector_sha256"],
                             "k": k, "block_rows": rec["block_rows"],
                             "block_ids_sha256_block_order": rec["block_ids_sha256_block_order"],
                             "rows_moved": rec["rows_moved"], "seconds_fit_wall": rec["seconds_fit_wall"],
                             "seconds_fit_process_cpu": rec["seconds_fit_process_cpu"]}
    return info


def bag_of(pf: dict, v: dict, nm: str, seeds, L: int) -> np.ndarray:
    return C.bag({s: v[f"{nm}{s}_L{L}"] for s in seeds}, seeds, pf["pbase"], pf["usable"], pf["guard"])


def comparator_levels(pan: dict, vec: dict) -> dict:
    """Level vectors of the reproduced comparators on July (no exp-058 vector)."""
    pf, v = pan[F], vec[F]
    lv = {nm: [v[f"{nm}_L{L}"] for L in range(4)]
          for nm in [f"P{s}" for s in SEEDS10] + [f"C{s}" for s in SEEDS10] + ["S1", "S2", "S3",
                                                                              "P3bag", "C3bag", "T10"]}
    lv["P10bag"] = [bag_of(pf, v, "P", SEEDS10, L) for L in range(4)]
    lv["C10bag"] = [bag_of(pf, v, "C", SEEDS10, L) for L in range(4)]
    lv["S3bag"] = [bag_of(pf, v, "S", SEEDS3, L) for L in range(4)]
    for L in range(4):
        if not (np.array_equal(bag_of(pf, v, "C", SEEDS3, L), lv["C3bag"][L])
                and np.array_equal(bag_of(pf, v, "P", SEEDS3, L), lv["P3bag"][L])):
            raise SystemExit("R6: the K = 3 bags do not rebuild exp-055's pinned bags")
    return lv


def bags(pan: dict, vec: dict) -> dict:
    """Level vectors by fold, the new arms included; the bag rule is exp-055 `common.bag` (SPEC 5)."""
    lev = {}
    for f in C.FOLDS:
        pf, v = pan[f], vec[f]
        lv = {nm: [v[f"{nm}_L{L}"] for L in range(4)] for nm in [f"P{s}" for s in SEEDS10] + ["P3bag", "T10"]}
        lv["P10bag"] = [bag_of(pf, v, "P", SEEDS10, L) for L in range(4)]
        if f == F:
            lv |= comparator_levels(pan, vec)
            for arm in (("F", "M") if REPRO_ONLY else ("F", "M", *LADDER_ARMS)):
                for s in SEEDS3:
                    lv[f"{arm}{s}"] = [v[f"{arm}{s}_L{L}"] for L in range(4)]
                lv[f"{arm}3bag"] = [bag_of(pf, v, arm, SEEDS3, L) for L in range(4)]
        for nm, x in lv.items():
            if nm.endswith("bag") or nm == "T10":
                for k in KS:
                    if np.any((x[k] - x[k - 1])[~pf["H"]] != 0):
                        raise SystemExit(f"{f} {nm} {STEP[k]}: a bag step moves a row outside H")
        lev[f] = lv
    return lev


# ------------------------------------------------------------------------------ statistics -----

def cells_july(pan: dict, lv: dict, y: np.ndarray, comps: dict, sse_names: list) -> pd.DataFrame:
    """Per (UTC day, airport) sums over every July row (SPEC 7.4), by exp-055 lossfals's route."""
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
            for st, v in (("ry", r * dy), ("yy", dy * dy), ("rx", r * dx), ("xx", dx * dx), ("xy", dx * dy)):
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


class Draws:
    """Per-day H sums of one member draw of a bag comparison, the bags rebuilt by the bag rule."""

    def __init__(self, pan: dict, lv: dict, y: np.ndarray, days: np.ndarray, kinds=("P", "C", "S")):
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
        """exp-055 `common.bag` on H (usable, not guard): the row mean in the order given, the floor."""
        return A.apply_floor(self.pbH, self.M[nm][L][np.asarray(idx) - 1].mean(0))

    def sums(self, X: str, Y: str, idx, levels=(0,)) -> dict:
        bx = [self.bag(X, idx, L) for L in range(4)]
        by = [self.bag(Y, idx, L) for L in range(4)]
        out = {}
        for k in KS:
            s = STEP[k]
            dx, dy = bx[k] - bx[k - 1], by[k] - by[k - 1]
            r = self.yH - self.pbH - bx[k - 1]
            for st, v in (("ry", r * dy), ("yy", dy * dy), ("rx", r * dx), ("xx", dx * dx), ("xy", dx * dy)):
                out[f"{st}_{s}"] = np.bincount(self.dc, weights=v, minlength=self.nd)
        for L in levels:             # SSE(Y_L) - SSE(X_L) on H; both bags are 0 outside H
            ey, ex = self.yH - self.pbH - by[L], self.yH - self.pbH - bx[L]
            out[f"dsse_L{L}"] = np.bincount(self.dc, weights=ey * ey - ex * ex, minlength=self.nd)
        return out


def draw_csv(cols: dict) -> str:
    return pd.DataFrame(cols).to_csv(index=False, float_format="%.17g")


def cmp_draw_csv(mine: str, path: Path) -> dict:
    """R7 (ii): this script's rebuild against exp-057's committed draw receipt, column for column."""
    check59(path)
    a = pd.read_csv(io.StringIO(mine), dtype={"k3_multiset": str, "k10_indices": str},
                    float_precision="round_trip")
    b = pd.read_csv(path, dtype={"k3_multiset": str, "k10_indices": str}, float_precision="round_trip")
    if list(a.columns) != list(b.columns):
        raise SystemExit(f"R7: {path.name} columns {list(b.columns)[:4]}... differ from the rebuild's")
    if not a[["draw", "k3_multiset", "k10_indices"]].equals(b[["draw", "k3_multiset", "k10_indices"]]):
        raise SystemExit(f"R7: {path.name} draw indices or member draws differ from the rebuild's")
    num = [c for c in a.columns if c not in ("k3_multiset", "k10_indices")]
    A_, B_ = a[num].to_numpy("float64"), b[num].to_numpy("float64")
    ad = np.abs(A_ - B_)
    rel = float(np.max(ad / np.maximum(np.abs(B_), 1e-300)))
    return {"columns": len(a.columns), "rows": int(len(a)), "max_abs_diff": float(ad.max()),
            "max_rel_diff": rel, "bytes_identical": bool(mine == path.read_text()),
            "within_1e-9_relative": bool(rel <= TOL_9)}


COST = EXP / "analysis/cost.json"


def cost_rows() -> list:
    return [json.loads(ln) for ln in COST.read_text().splitlines() if ln.strip()] if COST.exists() else []


def cost_append(job: str, wall: float, cpu: float, note: str = "") -> None:
    """SPEC 12: one write-once row per compute job of this experiment."""
    if any(r["job"] == job for r in cost_rows()):
        raise SystemExit(f"cost.json already holds a row for job {job!r}; rows are write-once")
    with open(COST, "a") as fh:
        fh.write(json.dumps({"job": job, "wall_s": round(wall, 1), "process_cpu_s": round(cpu, 1),
                             "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "note": note}, sort_keys=True) + "\n")


def census_check() -> dict:
    opened = sorted(r for r in OPENED if not r.startswith((E59_EXP, E59_ART)) and (ROOT / r).is_file())
    miss = [r for r in opened if r not in CENSUS]
    diff = [r for r in opened if r in CENSUS and sha_file(ROOT / r) != CENSUS[r]["sha256"]]
    return {"opened": len(opened), "missing": miss, "differ": diff, "files": opened}


# ------------------------------------------------------------------------------ R7 --------------

def repro057(pan: dict, lab: dict, lvc: dict, days: np.ndarray, draws: dict, Dr, cir,
             e57: dict) -> tuple[dict, dict, dict]:
    """SPEC 6 R7: exp-057's points and its three draw receipts, rebuilt before any new vector."""
    pf = pan[F]
    H, y, pb = pf["H"], lab[F]["y"], pf["pbase"]
    n_all, nH = len(y), int(H.sum())
    cells = cells_july(pan, lvc, y, COMPS57, SSE57)
    _, back = LF.receipt("exp057-july_loo", cells)
    Dm, ci = LF.day_matrix(back, days)
    P_ = Dm.sum(0)
    if int(P_[ci["n"]]) != n_all or int(P_[ci["n_H"]]) != nH:
        raise SystemExit("R7: the comparator receipt does not cover every July row")
    # (i) the rule-S and rule-Z points
    w12, w9, pts = 0.0, 0.0, {}
    for c in COMPS57:
        pts[c] = {}
        for s in SP:
            q = {a: fl(b) for a, b in q_cols(P_, ci, c, s).items()}
            ref = e57["points"][c]["steps"][s]
            w12 = max(w12, abs(q["m"] - ref["m"]))
            for key in ("A", "B", "g_C", "g_hat", "g_plain", "J", "cos", "amplitude", "alignment"):
                w9 = max(w9, abs(q[key] - ref[key]))
            pts[c][s] = {"m_rebuilt": q["m"], "m_exp057": ref["m"], "abs_diff_m": abs(q["m"] - ref["m"])}
    if w12 > TOL_12 or w9 > TOL_9:
        raise SystemExit(f"R7 (i): exp-057's points are not reproduced (m {w12:.3e}, loss {w9:.3e})")
    # (ii) exp-057's three draw receipts, column for column
    DR = Draws(pan, lvc, y, days, kinds=("P", "C", "S"))
    for idx, X, want in (((1, 2, 3), "C", "C3bag"), ((1, 2, 3), "S", "S3bag"), (SEEDS10, "C", "C10bag"),
                         (SEEDS10, "P", "P10bag"), ((1, 2, 3), "P", "P3bag")):
        for L in range(4):
            if not np.array_equal(DR.bag(X, idx, L), lvc[want][L][H]):
                raise SystemExit(f"R7: the draw route's bag {want} L{L} is not the point bag on H")
    k3 = {j: {"sham": DR.sums("S", "P", t, levels=range(4)), "carrier": DR.sums("C", "P", t)}
          for j, t in enumerate(MULTI3)}
    nday, nHday = Dm[:, ci["n"]], Dm[:, ci["n_H"]]
    out, k10_keep = {}, {}
    for L in BLOCKS:
        d = draws[L]
        W = C.mbb_weights(d["st"]["july_loo"], 31, L)
        Wr = C.mbb_weights(d["st"]["2026-07"], 31, L)
        wn, wnH = W @ nday, W @ nHday
        qd = {s: np.sqrt(((Wr @ Dr[:, cir[f"ss_UP_{s}"]]) / (Wr @ Dr[:, cir["n"]]))
                         / ((W @ Dm[:, ci[f"ss_T10_{s}"]]) / wnH)) for s in SP}
        per = {arm: {key: np.stack([W @ k3[j][arm][key] for j in range(len(MULTI3))], axis=1)[np.arange(B), d["msi"]]
                     for key in k3[0][arm]} for arm in ("sham", "carrier")}
        k10 = None
        for b in range(B):
            sm = DR.sums("C", "P", tuple(d["sig10"][b]))
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
        out[f"L{L}"] = cmp_draw_csv(text, E57 / "analysis/receipts" / f"draws-L{L}.csv") | {"text": text}
        if not out[f"L{L}"]["within_1e-9_relative"]:
            raise SystemExit(f"R7 (ii): the rebuild of exp-057's draws-L{L}.csv misses it by "
                             f"{out[f'L{L}']['max_rel_diff']:.3e} relative")
    # (iii) exp-057's day-by-airport receipt, cell for cell
    p57 = E57 / "analysis/receipts/daysums-july_loo.csv"
    check59(p57)
    b57 = LF.read_receipt(p57)
    if not (b57["day"].equals(back["day"]) and b57["airport"].equals(back["airport"])):
        raise SystemExit("R7 (iii): exp-057's receipt is not on this receipt's (day, airport) rows")
    shared = [c for c in back.columns if c in b57.columns and c not in ("day", "airport")]
    a_, bb = back[shared].to_numpy("float64"), b57[shared].to_numpy("float64")
    rel57 = float(np.max(np.abs(a_ - bb) / np.maximum(np.abs(bb), 1e-300)))
    if rel57 > TOL_9:
        raise SystemExit(f"R7 (iii): exp-057's cells differ by {rel57:.3e} relative")
    checks = {"points_max_abs_diff_m": w12, "points_max_abs_diff_loss": w9, "points": pts,
              "draw_receipts": {k: {a: b for a, b in v.items() if a != "text"} for k, v in out.items()},
              "daysums_shared_columns": len(shared), "daysums_max_rel_diff": rel57}
    return checks, {k: v["text"] for k, v in out.items()}, k10_keep


# ------------------------------------------------------------------------------ R8 --------------

def repro058(pan: dict, lab: dict, lv58: dict, days: np.ndarray, draws: dict, Dr, cir,
             e58: dict) -> tuple[dict, dict]:
    """SPEC 6 R8: exp-058's points, own values, paired intervals and receipts, rebuilt over exp-058's
    own comparison set and column order, before any new ladder vector is opened."""
    pf = pan[F]
    H, y, pb = pf["H"], lab[F]["y"], pf["pbase"]
    n_all, nH = len(y), int(H.sum())
    cells = cells_july(pan, lv58, y, COMPS58, SSE58)
    text58, back = LF.receipt("exp058-july_loo", cells)
    Dm, ci = LF.day_matrix(back, days)
    P_ = Dm.sum(0)
    if int(P_[ci["n"]]) != n_all or int(P_[ci["n_H"]]) != nH:
        raise SystemExit("R8: the exp-058 receipt does not cover every July row")
    # (iii) every point of exp-058's own comparison set
    w12, w9, pts, f8 = 0.0, 0.0, {}, {}
    for c in COMPS58:
        pts[c] = {}
        for s in SP:
            q = {a: fl(b) for a, b in q_cols(P_, ci, c, s).items()}
            #: review round 1, finding 1: SPEC 6 R8 (iii) names m - q, m / q and I as well. They
            #: are rebuilt here from this run's own m and exp-055's pinned q, and compared too.
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
            pts[c][s] = {k_: {"rebuilt": q[k_], "exp058": ref[k_], "abs_diff": abs(q[k_] - ref[k_])}
                         for k_ in R8_III_FIELDS}
    if w12 > TOL_12 or w9 > TOL_9:
        raise SystemExit(f"R8 (iii): exp-058's points are not reproduced (m/cos {w12:.3e}, loss {w9:.3e})")
    # (iv) exp-058's five L0 own values
    w_own, own58 = 0.0, {}
    for nm in BAG_COMPS58:
        X, Y = COMPS58[nm][0], COMPS58[nm][1]
        val = fl((P_[ci[f"sse_{Y}_L0"]] - P_[ci[f"sse_{X}_L0"]]) / n_all)
        ref = e58["own_value_L0"]["by_arm"][nm]["value"]
        w_own = max(w_own, abs(val - ref))
        own58[nm] = {"rebuilt": val, "exp058": ref, "abs_diff": abs(val - ref)}
    if w_own > TOL_9:
        raise SystemExit(f"R8 (iv): exp-058's L0 own values are missed by {w_own:.3e} s2/row")
    # (vi) exp-058's three draw receipts, over exp-058's own column set and order
    DR = Draws(pan, lv58, y, days, kinds=("P", "C", "S", "F", "M"))
    k3 = {j: {c: DR.sums(K3_DRAW58[c], "P", t, levels=range(4) if c == "sham_K3" else (0,))
              for c in K3_DRAW58} for j, t in enumerate(MULTI3)}
    nday, nHday = Dm[:, ci["n"]], Dm[:, ci["n_H"]]
    out, paired = {}, {s: {} for s in SP}
    for L in BLOCKS:
        d = draws[L]
        W = C.mbb_weights(d["st"]["july_loo"], 31, L)
        Wr = C.mbb_weights(d["st"]["2026-07"], 31, L)
        wn, wnH = W @ nday, W @ nHday
        qd = {s: np.sqrt(((Wr @ Dr[:, cir[f"ss_UP_{s}"]]) / (Wr @ Dr[:, cir["n"]]))
                         / ((W @ Dm[:, ci[f"ss_T10_{s}"]]) / wnH)) for s in SP}
        per = {c: {key: np.stack([W @ k3[j][c][key] for j in range(len(MULTI3))], axis=1)[np.arange(B), d["msi"]]
                   for key in k3[0][c]} for c in K3_DRAW58}
        k10 = None
        for b in range(B):
            sm = DR.sums("C", "P", tuple(d["sig10"][b]))
            if k10 is None:
                k10 = {key: np.empty(B) for key in sm}
            for key, v_ in sm.items():
                k10[key][b] = W[b] @ v_
        per["carrier_K10"] = k10
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
        out[f"L{L}"] = cmp_draw_csv(txt, E58 / "analysis/receipts" / f"draws-L{L}.csv") | {"text": txt}
        if not out[f"L{L}"]["within_1e-9_relative"]:
            raise SystemExit(f"R8 (vi): the rebuild of exp-058's draws-L{L}.csv misses it by "
                             f"{out[f'L{L}']['max_rel_diff']:.3e} relative")
        # (v) exp-058's paired m_Feb - m_June intervals
        for s in SP:
            mF = np.sqrt(per["feb_K3"][f"xx_{s}"] / per["feb_K3"][f"yy_{s}"])
            mM = np.sqrt(per["june_matched_K3"][f"xx_{s}"] / per["june_matched_K3"][f"yy_{s}"])
            got = interval(mF - mM)
            ref = e58["feb_minus_june_matched"][s]["intervals"][f"L{L}"]["day+member"]
            gday = interval(q_cols(W @ Dm, ci, "feb_K3", s)["m"] - q_cols(W @ Dm, ci, "june_matched_K3", s)["m"])
            rday = e58["feb_minus_june_matched"][s]["intervals"][f"L{L}"]["day"]
            paired[s][f"L{L}"] = {
                "day+member": {"rebuilt": [got["lo"], got["hi"]], "exp058": [ref["lo"], ref["hi"]],
                               "max_abs_diff": max(abs(got["lo"] - ref["lo"]), abs(got["hi"] - ref["hi"]))},
                "day": {"rebuilt": [gday["lo"], gday["hi"]], "exp058": [rday["lo"], rday["hi"]],
                        "max_abs_diff": max(abs(gday["lo"] - rday["lo"]), abs(gday["hi"] - rday["hi"]))}}
    w_int = max(v_[m_]["max_abs_diff"] for s in SP for v_ in paired[s].values() for m_ in ("day", "day+member"))
    if w_int > TOL_9:
        raise SystemExit(f"R8 (v): exp-058's paired intervals are missed by {w_int:.3e}")
    # (vi) exp-058's day-by-airport and day receipts, cell for cell
    rel58 = {}
    p58 = E58 / "analysis/receipts/daysums-july_loo.csv"
    check59(p58)
    b58 = LF.read_receipt(p58)
    if not (b58["day"].equals(back["day"]) and b58["airport"].equals(back["airport"])):
        raise SystemExit("R8 (vi): exp-058's receipt is not on this receipt's (day, airport) rows")
    shared = [c for c in back.columns if c in b58.columns and c not in ("day", "airport")]
    a_, bb = back[shared].to_numpy("float64"), b58[shared].to_numpy("float64")
    rel58["daysums"] = {"shared_columns": len(shared),
                        "max_rel_diff": float(np.max(np.abs(a_ - bb) / np.maximum(np.abs(bb), 1e-300)))}
    pd58 = E58 / "analysis/receipts/days-july_loo.csv"
    check59(pd58)
    d58 = pd.read_csv(pd58, dtype={"day": str}, float_precision="round_trip")
    shared_d = [c for c in ci if c in d58.columns]
    a2 = np.stack([Dm[:, ci[c]] for c in shared_d], axis=1)
    b2 = d58[shared_d].to_numpy("float64")
    rel58["days"] = {"shared_columns": len(shared_d),
                     "max_rel_diff": float(np.max(np.abs(a2 - b2) / np.maximum(np.abs(b2), 1e-300)))}
    for k_, v_ in rel58.items():
        if v_["max_rel_diff"] > TOL_9:
            raise SystemExit(f"R8 (vi): exp-058's {k_} cells differ by {v_['max_rel_diff']:.3e} relative")
    checks = {"points_max_abs_diff_m_cos": w12, "points_max_abs_diff_loss": w9, "points": pts,
              "points_max_abs_diff_by_field": f8, "points_fields_compared": list(R8_III_FIELDS),
              "own_value_L0": own58, "own_value_max_abs_diff": w_own,
              "paired_intervals": paired, "paired_intervals_max_abs_diff": w_int,
              "draw_receipts": {k_: {a: b for a, b in v_.items() if a != "text"} for k_, v_ in out.items()},
              "receipt_cells": rel58}
    return checks, {"draws": {k_: v_["text"] for k_, v_ in out.items()}, "daysums": text58}


def populations_block(rec: dict, pan: dict, lab: dict, lev: dict, lfj: dict, e57: dict,
                      e58: dict) -> None:
    """SPEC 9 and R8 (vii): COMMON's populations, and their reproduction of exp-055's, exp-057's
    and exp-058's recorded values.  Called by the measurement and, since review round 1 finding 2,
    by the --repro-only route as well, which builds exp-058's predictors only."""
    check59(C.E53 / "analysis/pin.json")
    qpin = json.loads((C.E53 / "analysis/pin.json").read_text())
    rec["populations"] = {}
    pw, pw57, pw58 = 0.0, 0.0, 0.0
    for f in C.FOLDS:
        ids, yy, pbf = pan[f]["ids"], lab[f]["y"], pan[f]["pbase"]
        nonm = LF.no_nm_mask(ids, f)
        if C.sha_arr(nonm) != qpin["folds"][f]["no_nm_sha256"]:
            raise SystemExit(f"{f}: the no-NM mask is not Q15's")
        masks = {"all": np.ones(len(ids), bool), "le3600": yy <= CUT, "gt3600": yy > CUT, "no_nm": nonm,
                 "clean": lab[f]["clean"]}
        masks |= {f"own_{g}": pan[f]["reg"] == g for g in C.REGIMES}
        masks |= {f"nm_{r_}": lab[f]["nmreg"] == r_ for r_ in sorted(set(lab[f]["nmreg"]))}
        lad_preds = ([] if REPRO_ONLY else
                     [f"{a}3bag" for a in LADDER_ARMS] + [f"{a}{i}" for a in LADDER_ARMS for i in SEEDS3])
        preds = (lad_preds + ["F3bag", "M3bag"] + [f"{nm}{i}" for nm in ("F", "M") for i in SEEDS3]
                 + ["P3bag", "C3bag", "S3bag", "P10bag", "C10bag"]) if f == F else ["P3bag", "P10bag"]
        common5 = ("all", "le3600", "gt3600", "no_nm", "clean")
        rec["populations"][f] = {"rows": {k: int(m.sum()) for k, m in masks.items()},
                                 "rows_by_serving_regime": {g: {k: int((masks[k] & masks[f"own_{g}"]).sum())
                                                                for k in common5} for g in C.REGIMES},
                                 "by_serving_regime": {}}
        for nm in preds:
            for Lv in range(4):
                e = yy - pbf - lev[f][nm][Lv]
                rec["populations"][f][f"{nm}_L{Lv}"] = {k: fl(np.sqrt(np.mean(e[m] ** 2))) if m.any() else None
                                                        for k, m in masks.items()}
                rec["populations"][f]["by_serving_regime"][f"{nm}_L{Lv}"] = {
                    g: {k: fl(np.sqrt(np.mean(e[masks[k] & masks[f"own_{g}"]] ** 2)))
                        if (masks[k] & masks[f"own_{g}"]).any() else None for k in common5} for g in C.REGIMES}
        for nm in (("P3bag", "C3bag") if f == F else ("P3bag",)):
            for Lv in range(4):
                for k in masks:
                    cell = lfj["populations"][f][f"{nm}_L{Lv}"].get(k)
                    ref = cell.get("rmse_s") if isinstance(cell, dict) else None
                    if ref is not None:
                        pw = max(pw, abs(ref - rec["populations"][f][f"{nm}_L{Lv}"][k]))
        for nm in (("P3bag", "C3bag", "S3bag", "P10bag", "C10bag") if f == F else ("P3bag", "P10bag")):
            for Lv in range(4):
                for k, ref in e57["populations"][f][f"{nm}_L{Lv}"].items():
                    if ref is not None and k in rec["populations"][f][f"{nm}_L{Lv}"]:
                        pw57 = max(pw57, abs(ref - rec["populations"][f][f"{nm}_L{Lv}"][k]))
        for nm, cell in e58["populations"][f].items():                      # R8 (vii)
            if not isinstance(cell, dict) or nm.startswith("rows"):
                continue
            if nm not in rec["populations"][f]:
                continue
            for k, ref in cell.items():
                if isinstance(ref, (int, float)) and k in rec["populations"][f][nm]:
                    pw58 = max(pw58, abs(ref - rec["populations"][f][nm][k]))
    rec["checks"]["populations_vs_exp055_max_abs_diff_s"] = pw
    rec["checks"]["populations_vs_exp057_max_abs_diff_s"] = pw57
    rec["checks"]["populations_vs_exp058_max_abs_diff_s"] = pw58
    if max(pw, pw57, pw58) > TOL_9:
        raise SystemExit(f"a reproduced population differs by {max(pw, pw57, pw58):.3e} s")



# ------------------------------------------------------------------------------ main ------------

def main() -> None:
    t0, c0 = time.time(), time.process_time()
    np.seterr(divide="ignore", invalid="ignore")
    #: the reproduction route guards its record here and its receipts in put_or_verify(), which
    #: keeps a receipt an earlier pass wrote and checks its bytes instead of replacing it
    outs = ([REPRO_OUT] if REPRO_ONLY else
            [OUT, REC_DIR / "daysums-july_loo.csv", REC_DIR / "days-july_loo.csv",
             *(REC_DIR / f"draws-L{L}.csv" for L in BLOCKS),
             *(REC_DIR / f"exp057-draws-L{L}.csv" for L in BLOCKS),
             *(REC_DIR / f"exp058-draws-L{L}.csv" for L in BLOCKS),
             REC_DIR / "exp058-daysums-july_loo.csv",
             *(OOFD / f"ownvalue-{f}.parquet" for f in C.FOLDS),
             ART / "receipts/rows-july_loo.parquet", ART / "oof/members-july_loo.parquet"])
    NO.refuse_existing(outs, overwrite=False)
    rec = {"what": __doc__.strip().splitlines()[0], "spec_commit": SPEC_COMMIT, "standin": STANDIN,
           "plan_code_sha256": PLAN["code"]["sha256"], "plans_read": PLAN["_plans_read"],
           "B": B, "blocks": list(BLOCKS),
           "generator": "np.random.default_rng(5500 + L)",
           "draw_order": ["july_loo fold starts", "2026-07 ranking starts", "january fold starts",
                          "2026-01 ranking starts", "december fold starts", "K = 3 member triples (B x 3)",
                          "K = 10 member-index vectors (B x 10), exp-057's"],
           "arms": {"F": "exp-058's February 2025 arm, true labels, 60,777 rows (the fixed comparator)",
                    "M": "exp-058's matched June 2025 arm, true labels, 60,777 rows (the ladder's k = 24 point)"}
                   | {a: f"June 2025 ladder arm k = {ARM_K[a]} (f = {ARM_K[a]}/24), N = {ARM_N[a]:,} rows, "
                        f"true labels" + ("  -- June*" if a == JUNE_STAR else "") for a in LADDER_ARMS},
           "june_star": LADDER["june_star"], "ladder_branch": LADDER["branch"],
           "ladder_record": "analysis/ladder2.json" if (EXP / "analysis/ladder2.json").exists()
                            else "analysis/ladder.json",
           "margin": MARGIN,
           "units": {"m": "ratio", "loss": "s2 per July fold row (n = 190,713)", "rmse": "s"}, "checks": {}}

    # ---- R1 -------------------------------------------------------------------------------------
    pan, vec, pin = V.load()
    lab = {f: LF.load_labels(f, pan[f]["ids"]) for f in C.FOLDS}
    rec["checks"]["R1_originvec_load"] = True
    print("R1 exp-055 pins verified; the labels equal exp-051's OOF y_true on every fold", flush=True)

    # ---- R2 -------------------------------------------------------------------------------------
    rec["R2_exp051_gains"] = r2(pan, vec, lab)
    print(f"R2 exp-051 gains reproduced: {len(rec['R2_exp051_gains']['values'])} values, max |diff| "
          f"{rec['R2_exp051_gains']['max_abs_diff']:.2e}; {rec['R2_exp051_gains']['n_keys']} keys read, no other",
          flush=True)

    # ---- R3, R4 ---------------------------------------------------------------------------------
    if ARGS.cite_r34:
        cited = json.loads((ROOT / ARGS.cite_r34).read_text())
        if not cited["R3_R4_exp055_scripts"]["registered_fields"]["all_within_tolerance"]:
            raise SystemExit(f"{ARGS.cite_r34}: its R3/R4 did not pass, so it cannot be cited")
        rec["R3_R4_exp055_scripts"] = dict(cited["R3_R4_exp055_scripts"]) | {
            "not_re_run_here": True, "cited_record": ARGS.cite_r34,
            "cited_record_sha256": sha_file(ROOT / ARGS.cite_r34),
            "why": "exp-055's screen.py and lossfals.py are run as subprocesses and dominate this "
                   "pass; they passed at zero difference in the cited record on the same bytes, "
                   "which this experiment's census pins, so they are carried over unchanged"}
        print(f"R3/R4 carried from {ARGS.cite_r34} (not re-run): "
              f"{rec['R3_R4_exp055_scripts']['registered_fields']['n']} registered fields equal",
              flush=True)
    else:
        rec["R3_R4_exp055_scripts"] = r34()
    for sub, r_ in rec["R3_R4_exp055_scripts"].get("runs", {}).items():
        print(f"R3/R4 {sub}: registered fields equal; whole record: {r_['whole_record']['n_structural_diffs']} "
              f"structural, {r_['whole_record']['float_fields_differing']} float fields differ "
              f"(max rel {r_['whole_record']['max_rel_float_diff']:.1e}); receipts equal {r_['receipt_hashes_equal']}",
              flush=True)
    scr = json.loads((E55 / "analysis/screen.json").read_text())
    lfj = json.loads((E55 / "analysis/lossfals.json").read_text())
    for s in SP:
        got = scr["q"]["july_loo->2026-07"][s]
        if got["point"] != Q_POINT[s] or (got["intervals"]["L3"]["lo"], got["intervals"]["L3"]["hi"]) != Q_L3[s]:
            raise SystemExit(f"{s}: the SPEC's q interval is not screen.json's")
    rec["checks"]["own_arithmetic_on_exp055_points"] = own_points(pan, vec, lab, scr, lfj)

    # ---- R5 -------------------------------------------------------------------------------------
    e57p = E57 / "analysis/measure.json"
    check59(e57p)
    e57 = json.loads(e57p.read_text())
    if e57["spec_commit"] != E57_SPEC_COMMIT or e57["standin"]:
        raise SystemExit("exp-057's measure.json is not its registered official record")
    days = np.sort(np.unique(pan[F]["day"]))
    if len(days) != 31:
        raise SystemExit("July does not have 31 fold days")
    draws = {}
    for L in BLOCKS:
        rng = np.random.default_rng(5500 + L)
        st = {key: C.mbb_starts(rng, 31, L, B) for key in ("july_loo", "2026-07", "january", "2026-01", "december")}
        sig = rng.integers(1, 4, size=(B, 3))
        sig10 = rng.integers(1, 11, size=(B, 10))
        msi = np.array([MULTI3.index(tuple(sorted(t))) for t in sig.tolist()])
        want = scr["draw_sha256"][f"L{L}"]
        got = {k: C.sha_arr(v) for k, v in st.items()} | {"seeds": C.sha_arr(sig)}
        counts = {MS3[j]: int((msi == j).sum()) for j in range(len(MS3))}
        if got != {k: v for k, v in want.items() if k != "multiset_counts"} or counts != want["multiset_counts"]:
            raise SystemExit(f"R5 L{L}: the regenerated draws are not exp-055's")
        if C.sha_arr(sig10) != e57["k10_member_draw_sha256_unsorted"][f"L{L}"]:
            raise SystemExit(f"R5 L{L}: the K = 10 member draws are not exp-057's")
        draws[L] = {"st": st, "msi": msi, "sig10": np.sort(sig10, axis=1), "sig10_sha256": C.sha_arr(sig10)}
    rec["checks"]["R5_draws_equal_exp055_and_exp057"] = True
    rec["k10_member_draw_sha256_unsorted"] = {f"L{L}": draws[L]["sig10_sha256"] for L in BLOCKS}
    print("R5 the regenerated draws equal exp-055's and exp-057's", flush=True)

    # ---- R6 -------------------------------------------------------------------------------------
    for f in C.FOLDS:
        for s in range(4, 11):
            for L in range(4):
                p = C.member_path(L, s)
                check59(p)
                vec[f][f"P{s}_L{L}"] = np.load(p)[f].astype("float64")
    rec["exp057_vectors"] = load_exp057(vec)
    p10, worst10 = {}, 0.0
    for f in C.FOLDS:
        pf = pan[f]
        for L in range(4):
            b10 = C.bag({s: vec[f][f"P{s}_L{L}"] for s in SEEDS10}, SEEDS10, pf["pbase"], pf["usable"], pf["guard"])
            d = float(np.max(np.abs(b10 - vec[f][f"T10_L{L}"])))
            worst10 = max(worst10, d)
            p10[f"{f}:L{L}"] = {"max_abs_diff_s": d, "bit_identical": bool(np.array_equal(b10, vec[f][f"T10_L{L}"]))}
    if worst10 > TOL_9:
        raise SystemExit(f"R6: the rebuilt plain K = 10 bag differs from the saved heads by {worst10:.3e} s")
    s1 = {}
    for L in range(4):
        r51 = json.loads((C.E51 / f"analysis/fits2/i2-S-s1-L{L}.json").read_text())
        s1[f"L{L}"] = sha_file(C.sham_path(L)) == r51["vector_sha256"] and r51["vector"] == C.rel(C.sham_path(L))
    if not all(s1.values()):
        raise SystemExit(f"R6: S1 is not exp-051's recorded vector: {s1}")
    rec["R6"] = {"plain_K10_vs_heads_max_abs_diff_s": worst10, "by_fold_level": p10,
                 "S1_equals_exp051_fit_records": s1,
                 "S2_S3_Cx4_Cx10_equal_exp057_fit_records": len(rec["exp057_vectors"]) == 36,
                 "C3_P3_are_exp055_pinned_bags": "checked in comparator_levels()"}
    print(f"R6 the plain K = 10 bag rebuilds the saved heads (max |diff| {worst10:.1e} s); S1 is exp-051's; "
          f"{len(rec['exp057_vectors'])} exp-057 vectors carry their recorded hashes", flush=True)

    # ---- R7 (still no exp-058 vector) -------------------------------------------------------------
    rpath = E55 / "analysis/receipts/i1-daysums-rank-2026-07.csv"
    check59(rpath)
    Dr, cir = SC.day_matrix(SC.read_receipt(rpath), np.array([f"2026-07-{d:02d}" for d in range(1, 32)]))
    lvc = comparator_levels(pan, vec)
    r7, e57_draw_text, k10_keep = repro057(pan, lab, lvc, days, draws, Dr, cir, e57)
    rec["R7_exp057"] = r7
    print(f"R7 exp-057's points rebuilt (max |diff| m {r7['points_max_abs_diff_m']:.1e}, loss "
          f"{r7['points_max_abs_diff_loss']:.1e}); its three draw receipts rebuilt (max rel "
          f"{max(v['max_rel_diff'] for v in r7['draw_receipts'].values()):.1e}, bytes identical "
          f"{[v['bytes_identical'] for v in r7['draw_receipts'].values()]}); its day-by-airport cells "
          f"agree to {r7['daysums_max_rel_diff']:.1e} relative", flush=True)

    # ---- R8: exp-058's own record, before any new ladder vector is opened ---------------------------
    rec["exp058_vectors"] = load_exp058(vec)
    e58p = E58 / "analysis/measure.json"
    check59(e58p)
    e58 = json.loads(e58p.read_text())
    if e58["spec_commit"] != E58_SPEC_COMMIT or e58["standin"]:
        raise SystemExit("exp-058's measure.json is not its registered official record")
    lv58 = dict(lvc)
    for arm in ("F", "M"):
        for s_ in SEEDS3:
            lv58[f"{arm}{s_}"] = [vec[F][f"{arm}{s_}_L{L}"] for L in range(4)]
        lv58[f"{arm}3bag"] = [bag_of(pan[F], vec[F], arm, SEEDS3, L) for L in range(4)]
    r8, e58_text = repro058(pan, lab, lv58, days, draws, Dr, cir, e58)
    rec["R8_exp058"] = r8
    print(f"R8 exp-058's points rebuilt (max |diff| m/cos {r8['points_max_abs_diff_m_cos']:.1e}, loss "
          f"{r8['points_max_abs_diff_loss']:.1e}); its five L0 own values to {r8['own_value_max_abs_diff']:.1e}; "
          f"its paired intervals to {r8['paired_intervals_max_abs_diff']:.1e}; its three draw receipts "
          f"rebuilt (bytes identical {[v['bytes_identical'] for v in r8['draw_receipts'].values()]}); its "
          f"day-by-airport cells to {r8['receipt_cells']['daysums']['max_rel_diff']:.1e} relative", flush=True)

    # ---- SPEC 5.5: the ladder gave no June*, so the run stops here, before any new vector --------
    if REPRO_ONLY:
        #: review round 1, finding 2: R8 (vii), exp-058's populations, runs here too. bags() and
        #: the predictor list drop the ladder arms under --repro-only, so no exp-059 vector is
        #: touched; every predictor below comes from exp-055's, exp-057's or exp-058's vectors.
        populations_block(rec, pan, lab, bags(pan, vec), lfj, e57, e58)
        print(f"R8 (vii) exp-058's populations reproduced to "
              f"{rec['checks']['populations_vs_exp058_max_abs_diff_s']:.1e} s "
              f"(exp-055 {rec['checks']['populations_vs_exp055_max_abs_diff_s']:.1e}, "
              f"exp-057 {rec['checks']['populations_vs_exp057_max_abs_diff_s']:.1e})", flush=True)
        cc = census_check()
        rec["census_check"] = {k: v_ for k, v_ in cc.items() if k != "files"} | {"files_opened": cc["files"]}
        if cc["differ"] or cc["missing"]:
            raise SystemExit(f"census check fails: missing {cc['missing']}, differ {cc['differ']}")
        rec["repro_only"] = {
            "why": "SPEC 5.2 step 6 ended the ladder unmatched, so there is no June*; SPEC 5.4 holds "
                   "every m, loss, subgroup and step diagnostic of every new vector, and SPEC 5.5 "
                   "reports exp-058's unchanged numbers instead. This run opened no exp-059 vector.",
            "ladder_branch": LADDER["branch"],
            "exp059_vectors_opened": sorted(r_ for r_ in OPENED if r_.startswith(WORK_REL)),
            "checks_run": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]}
        if rec["repro_only"]["exp059_vectors_opened"]:
            raise SystemExit("--repro-only opened an exp-059 vector")
        rec["receipts"] = {}

        def put_or_verify(pth: Path, txt: str) -> None:
            """Write the receipt, or, when an earlier pass already wrote it, check the bytes and
            keep the file: records are write-once and a rebuild must not replace one."""
            k_ = str(pth.relative_to(ROOT))
            if pth.exists():
                same = pth.read_text() == txt
                rec["receipts"][k_] = {"sha256": sha_file(pth), "written_by": "an earlier pass",
                                       "rebuilt_bytes_identical": same}
                if not same:
                    raise SystemExit(f"{k_} on disk is not the bytes this pass rebuilt")
                return
            pth.parent.mkdir(parents=True, exist_ok=True)
            pth.write_text(txt)
            rec["receipts"][k_] = {"sha256": sha_file(pth), "written_by": "this pass",
                                   "rebuilt_bytes_identical": True}
        for L in BLOCKS:
            put_or_verify(REC_DIR / f"exp057-draws-L{L}.csv", e57_draw_text[f"L{L}"])
            put_or_verify(REC_DIR / f"exp058-draws-L{L}.csv", e58_text["draws"][f"L{L}"])
        put_or_verify(REC_DIR / "exp058-daysums-july_loo.csv", e58_text["daysums"])
        rec["seconds"] = round(time.time() - t0, 1)
        rec["seconds_process_cpu"] = round(time.process_time() - c0, 1)
        C.write_json(REPRO_OUT, rec)
        cost_append(f"repro-only-{REPRO_OUT.stem}", time.time() - t0, time.process_time() - c0,
                    "R1-R8 with no exp-059 vector opened (SPEC 5.5)")
        print(f"written {REPRO_OUT.relative_to(ROOT)} ({rec['seconds']} s wall, "
              f"{rec['seconds_process_cpu']} s process cpu); no new vector was opened", flush=True)
        return

    # ---- the new vectors ---------------------------------------------------------------------------
    rec["new_vectors"] = load_new(vec)
    lev = bags(pan, vec)
    L7 = lev[F]
    rec["checks"]["new_vectors_sha_checked_against_fit_records"] = not STANDIN
    rec["checks"]["every_bag_step_zero_outside_H"] = True
    rec["off_july"] = ("every carrier fit here is July-only (exp-051's design, folds=('july_loo',)): on "
                       "december and january each ladder member, F_s and M_s equals the plain member P_s of "
                       "the same seed and every K = 3 bag equals P3, so those folds are reported once as "
                       "P3bag; the equality is asserted by construction in bags()")
    print(f"new vectors loaded ({'STAND-INS' if STANDIN else '24 fits, sha-checked'})", flush=True)

    # ---- receipts and the day matrix ---------------------------------------------------------------
    pf = pan[F]
    H, y, pb = pf["H"], lab[F]["y"], pf["pbase"]
    n_all, nH = len(y), int(H.sum())
    text, back = LF.receipt("july_loo", cells_july(pan, L7, y, COMPS, SSE_NAMES))
    Dm, ci = LF.day_matrix(back, days)
    P_ = Dm.sum(0)
    if int(P_[ci["n"]]) != n_all or int(P_[ci["n_H"]]) != nH or nH != C.H_ROWS[F]:
        raise SystemExit("the receipt does not cover every July row")
    dtext = pd.DataFrame(Dm, columns=list(ci)).assign(day=days)[["day"] + list(ci)].to_csv(
        index=False, float_format="%.17g")
    lf_cells = LF.read_receipt(E55 / "analysis/receipts/i2-daysums-july_loo.csv")
    check59(E55 / "analysis/receipts/i2-daysums-july_loo.csv")
    worst_cell = 0.0
    for s in SP:
        for st in STATS:
            a_ = back[f"{st}_carrier_K3_{s}"].to_numpy("float64")
            b_ = lf_cells[f"{st}_bag123_{s}"].to_numpy("float64")
            worst_cell = max(worst_cell, float(np.max(np.abs(a_ - b_))))
    rec["checks"]["carrier_K3_cells_vs_exp055_i2_receipt_max_abs_diff"] = worst_cell

    # ---- points (SPEC 7.1, 7.2) --------------------------------------------------------------------
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
            w_split = max(w_split, abs((q["g_C"] - q["g_plain"]) - (q["amplitude"] + q["alignment"])),
                          abs(q["alignment"] - (q["alignment_direction"] + q["alignment_orthogonal"])),
                          abs(q["g_hat"] - q["g_plain"] - q["amplitude"]))
            ry_ = y - pb - L7[Y][k - 1]
            q |= {"outside_H_g_C": out_H,
                  "outside_H_plain_step_gain": float(P_[ci[f"oy_{c}_{s}"]] / n_all) if K == 1 else 0.0,
                  "direct_sse_gain_all_rows": sse,
                  "g_P_own_p0_all_rows": float(np.sum(2 * ry_ * dy - dy * dy) / n_all)}
            q["g_P_minus_g_plain"] = q["g_P_own_p0_all_rows"] - q["g_plain"]
            q["J_positive"] = bool(q["J"] > 0)
            q["J_positive_band_for_m"] = [2 * q["A"] / q["B"] - 1, 1.0]
            q |= {"q": Q_POINT[s], "I": float(imp(q["m"], Q_POINT[s])), "m_minus_q": q["m"] - Q_POINT[s],
                  "m_over_q": q["m"] / Q_POINT[s], "I_positive_band": [2 * Q_POINT[s] - 1, 1.0]}
            q["changed_rows"] = {nm: {g: int(((d != 0) & (pf["reg"] == g)).sum()) for g in C.REGIMES}
                                 for nm, d in (("carrier", dx), ("plain", dy))}
            q["outside_H_share_of_step_mass"] = {
                nm: {g: float(np.sum(d[pf["reg"] == g] ** 2) / np.sum(d ** 2)) for g in ("G", "A", "C")}
                for nm, d in (("carrier", dx), ("plain", dy))}
            q["kish_days_mass"] = {"carrier": C.kish(Dm[:, ci[f"xx_{c}_{s}"]]),
                                   "plain": C.kish(Dm[:, ci[f"yy_{c}_{s}"]])}
            q["intervals"] = {}
            Is.append(q["I"])
            Js.append(q["J"])
            ent["steps"][s] = q
        ent["mean_I"] = {"point": float(np.mean(Is)), "intervals": {}}
        ent["mean_J"] = {"point": float(np.mean(Js)), "intervals": {}}
        rec["points"][c] = ent
    rec["checks"] |= {"m_receipt_vs_direct_max_rel": w_direct, "split_identities_max_abs_diff_s2": w_split,
                      "direct_sse_reconciliation_max_abs_diff_s2": w_sse, "kish_days_rows_H": C.kish(Dm[:, ci["n_H"]])}
    if w_direct > TOL_12 or w_split > TOL_9 or w_sse > 1e-6:
        raise SystemExit(f"an identity fails: m {w_direct:.2e}, split {w_split:.2e}, sse {w_sse:.2e}")
    wpt = 0.0
    for c, pt in (("carrier_K3", "bag123"), ("carrier_s1", "seed1"), ("carrier_s2", "seed2"),
                  ("carrier_s3", "seed3"), ("sham_s1", "sham")):
        for s in SP:
            for key in ("A", "B", "m", "g_C", "g_hat", "g_plain", "J"):
                wpt = max(wpt, abs(rec["points"][c]["steps"][s][key] - lfj["points"][F][pt]["steps"][s][key]))
    rec["checks"]["exp055_points_from_this_receipt_max_abs_diff"] = wpt
    if wpt > TOL_9:
        raise SystemExit(f"the receipt route misses exp-055's points by {wpt:.3e}")
    w57 = 0.0
    for c in ("sham_K3", "carrier_K3", "carrier_K10", "sham_s1", "sham_s2", "sham_s3"):
        for s in SP:
            for key in ("A", "B", "m", "g_C", "g_hat", "g_plain", "J", "cos"):
                w57 = max(w57, abs(rec["points"][c]["steps"][s][key] - e57["points"][c]["steps"][s][key]))
    rec["checks"]["exp057_points_from_this_receipt_max_abs_diff"] = w57
    if w57 > TOL_9:
        raise SystemExit(f"the receipt route misses exp-057's points by {w57:.3e}")

    # ---- the paired contrasts (SPEC 7.1) ------------------------------------------------------------
    #: name -> (X comparison, Y comparison); every pair is read on the same days and the same member
    #: triple in a draw, so each difference is paired.  `feb_minus_june_star` is SPEC 8's reading.
    DIFFS = {"feb_minus_june_star": ("feb_K3", JS_COMP),
             "feb_minus_june_matched": ("feb_K3", "june_matched_K3")}
    DIFFS |= {f"feb_minus_ladder_{a}": ("feb_K3", LAD[a]) for a in LADDER_ARMS if LAD[a] != JS_COMP}
    DIFFS["feb_minus_full_june_K3"] = ("feb_K3", "carrier_K3")
    rec["paired_differences"] = {
        nm: {s: {"m_X": rec["points"][X]["steps"][s]["m"], "m_Y": rec["points"][Y]["steps"][s]["m"],
                 "difference": rec["points"][X]["steps"][s]["m"] - rec["points"][Y]["steps"][s]["m"],
                 "intervals": {}} for s in SP} for nm, (X, Y) in DIFFS.items()}
    rec["feb_minus_june_matched"] = rec["paired_differences"]["feb_minus_june_matched"]
    rec["difference_by_member"] = {
        nm: {f"s{i}": {s: rec["points"][f"{X.replace('_K3', '')}_s{i}"]["steps"][s]["m"]
                          - rec["points"][f"{Y.replace('_K3', '')}_s{i}"]["steps"][s]["m"] for s in SP}
             for i in SEEDS3}
        for nm, (X, Y) in DIFFS.items() if X.endswith("_K3") and Y.endswith("_K3")}

    # ---- each arm's own value at L0 (SPEC 7.2) -------------------------------------------------------
    own = {"definition": "[SSE(P3_L0) - SSE(X_L0)] / n on every July fold row (positive: the arm's own "
                         "L0 vector predicts July better than the plain K = 3 bag); measured, never subtracted",
           "exp051_june_K3_recorded": 582.2425054210122, "by_arm": {}}
    for nm, X in (("feb_K3", "F3bag"), ("june_matched_K3", "M3bag"), ("carrier_K3", "C3bag"),
                  ("sham_K3", "S3bag"), ("carrier_K10", "C10bag")):
        Y = "P10bag" if nm == "carrier_K10" else "P3bag"
        own["by_arm"][nm] = {"X": X, "Y": Y, "value": fl((P_[ci[f"sse_{Y}_L0"]] - P_[ci[f"sse_{X}_L0"]]) / n_all),
                             "intervals": {}}
    for nm in LAD.values():
        arm = LADDER_ARMS[[LAD[a] for a in LADDER_ARMS].index(nm)]
        own["by_arm"][nm] = {"X": f"{arm}3bag", "Y": "P3bag",
                             "value": fl((P_[ci["sse_P3bag_L0"]] - P_[ci[f"sse_{arm}3bag_L0"]]) / n_all),
                             "k": ARM_K[arm], "N": ARM_N[arm], "intervals": {}}
    for nm, X in ([("feb", "F"), ("june_matched", "M")] + [(LAD[a], a) for a in LADDER_ARMS]):
        for i in SEEDS3:
            own["by_arm"][f"{nm}_s{i}"] = {
                "X": f"{X}{i}", "Y": f"P{i}",
                "value": fl((P_[ci[f"sse_P{i}_L0"]] - P_[ci[f"sse_{X}{i}_L0"]]) / n_all), "intervals": {}}
    # exp-057's damage D_0 = [SSE(plain_L0) - SSE(sham_L0)] / n is this own value with the same sign
    d57 = abs(own["by_arm"]["sham_K3"]["value"] - e57["damage"]["K3"]["L0"]["D"])
    d51 = abs(own["by_arm"]["carrier_K3"]["value"] - e57["damage"]["own_value_L0"]["K3"]["value"])
    rec["checks"]["own_value_L0_vs_exp057_max_abs_diff"] = max(d57, d51)
    if max(d57, d51) > TOL_9:
        raise SystemExit(f"the L0 own values miss exp-057's by {max(d57, d51):.3e}")
    #: SPEC 7.2: the ladder arms' own values here must be the ones the stopping rule read
    wl = 0.0
    for a in LADDER_ARMS:
        wl = max(wl, abs(own["by_arm"][LAD[a]]["value"] - LADDER["arms"][a]["V"]))
        for i in SEEDS3:
            wl = max(wl, abs(own["by_arm"][f"{LAD[a]}_s{i}"]["value"]
                             - LADDER["arms"][a]["members"][f"s{i}"]["value"]))
    rec["checks"]["ladder_own_values_max_abs_diff_s2_row"] = wl
    if wl > TOL_12:
        raise SystemExit(f"the ladder arms' own values differ from analysis/ladder.json by {wl:.3e}")
    rec["own_value_L0"] = own

    # ---- decompositions (SPEC 7.1) --------------------------------------------------------------------
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
            dec["airport"][c][s] = {"m2_minus_1": fl(SX / SY - 1), "sum_of_contributions": fl(contrib.sum()),
                                    "by_airport": {
                a: {"H_rows": int(g_air.loc[a, "n_H"]), "contribution_to_m2_minus_1": fl(contrib[a]),
                    "plain_mass_share": fl(sy[a] / SY), "m_airport": fl(np.sqrt(sx[a] / sy[a])) if sy[a] > 0 else None,
                    "q_airport_exp055": qa[s]["by_airport"].get(a, {}).get("q_airport")} for a in g_air.index}}
            dx, dy = L7[X][k] - L7[X][k - 1], L7[Y][k] - L7[Y][k - 1]
            by, tot = {}, 0.0
            for rg in C.NM_REGIMES:
                sel = H & (pf["nm"] == rg)
                a_, b_ = float(np.sum(dx[sel] ** 2)), float(np.sum(dy[sel] ** 2))
                by[rg] = {"H_rows": int(sel.sum()), "contribution_to_m2_minus_1": (a_ - b_) / SY,
                          "plain_mass_share": b_ / SY, "m_regime": float(np.sqrt(a_ / b_)) if b_ > 0 else None}
                tot += (a_ - b_) / SY
            if sum(v_["H_rows"] for v_ in by.values()) != nH:
                raise SystemExit("the NM regimes do not cover H")
            w_dec = max(w_dec, abs(tot - (SX / SY - 1)))
            dec["nm_regime"][c][s] = {"m2_minus_1": fl(SX / SY - 1), "sum_of_contributions": tot, "by_regime": by}
            srv, tx, ty = {}, float(np.sum(dx ** 2)), float(np.sum(dy ** 2))
            for g in C.REGIMES:
                sel = pf["reg"] == g
                srv[g] = {"rows": int(sel.sum()),
                          "changed_rows_carrier": int((dx[sel] != 0).sum()), "changed_rows_plain": int((dy[sel] != 0).sum()),
                          "carrier_step_mass_share": float(np.sum(dx[sel] ** 2) / tx) if tx > 0 else None,
                          "plain_step_mass_share": float(np.sum(dy[sel] ** 2) / ty) if ty > 0 else None}
            dec["serving_regime"][c][s] = {"all_carrier_mass_in_H": bool(abs(srv["H"]["carrier_step_mass_share"] - 1) < 1e-12),
                                           "by_regime": srv}
    rec["checks"]["decomposition_max_reconciliation_error"] = w_dec
    if w_dec > TOL_9:
        raise SystemExit(f"a decomposition does not reconcile: {w_dec:.3e}")
    rec["decompositions"] = dec

    # ---- the fit-count record (SPEC 7.1) ---------------------------------------------------------------
    fc = {}
    names = ([f"P{i}" for i in SEEDS10]
             + [f"{nm}{i}" for nm in ("F", "M", "C", "S", *LADDER_ARMS) for i in SEEDS3]
             + [f"C{i}" for i in range(4, 11)]
             + ["P3bag", "F3bag", "M3bag", "C3bag", "S3bag", "P10bag", "C10bag", "T10"]
             + [f"{a}3bag" for a in LADDER_ARMS])
    for k in KS:
        s = STEP[k]
        stp = {nm: (L7[nm][k] - L7[nm][k - 1])[H] for nm in names}
        d10 = stp["P10bag"]
        fc[s] = {"rms_H_step_s": {nm: float(np.sqrt(np.sum(stp[nm] ** 2) / nH)) for nm in names},
                 "m": {"feb_K1": {f"s{i}": rec["points"][f"feb_s{i}"]["steps"][s]["m"] for i in SEEDS3},
                       "feb_K3": rec["points"]["feb_K3"]["steps"][s]["m"],
                       "june_matched_K1": {f"s{i}": rec["points"][f"june_matched_s{i}"]["steps"][s]["m"]
                                           for i in SEEDS3},
                       "june_matched_K3": rec["points"]["june_matched_K3"]["steps"][s]["m"],
                       "ladder_K1": {a: {f"s{i}": rec["points"][f"ladder_{a}_s{i}"]["steps"][s]["m"]
                                         for i in SEEDS3} for a in LADDER_ARMS},
                       "ladder_K3": {a: rec["points"][LAD[a]]["steps"][s]["m"] for a in LADDER_ARMS},
                       "full_june_K3": rec["points"]["carrier_K3"]["steps"][s]["m"],
                       "full_june_K10": rec["points"]["carrier_K10"]["steps"][s]["m"],
                       "sham_K3": rec["points"]["sham_K3"]["steps"][s]["m"]},
                 "cos_with_K10_plain_step": {
                     nm: float(np.sum(stp[nm] * d10) / np.sqrt(np.sum(stp[nm] ** 2) * np.sum(d10 ** 2)))
                     for nm in names if nm != "P10bag"}}
    rec["fit_count_record_july_H"] = fc

    # ---- intervals (SPEC 7.3) ----------------------------------------------------------------------------
    for s in SP:
        qpt = float(np.sqrt((Dr.sum(0)[cir[f"ss_UP_{s}"]] / Dr.sum(0)[cir["n"]])
                            / (P_[ci[f"ss_T10_{s}"]] / P_[ci["n_H"]])))
        if abs(qpt - Q_POINT[s]) > TOL_12:
            raise SystemExit(f"{s}: q from the two receipts {qpt!r} is not {Q_POINT[s]!r}")
    DR = Draws(pan, L7, y, days, kinds=("P", "C", "S", "F", "M", *LADDER_ARMS))
    for idx, X, want in ([((1, 2, 3), "C", "C3bag"), ((1, 2, 3), "S", "S3bag"), ((1, 2, 3), "F", "F3bag"),
                          ((1, 2, 3), "M", "M3bag"), (SEEDS10, "C", "C10bag"), (SEEDS10, "P", "P10bag"),
                          ((1, 2, 3), "P", "P3bag")]
                         + [((1, 2, 3), a, f"{a}3bag") for a in LADDER_ARMS]):
        for L in range(4):
            if not np.array_equal(DR.bag(X, idx, L), L7[want][L][H]):
                raise SystemExit(f"the draw route's bag {want} L{L} is not the point bag on H")
    rec["checks"]["draw_route_bags_equal_point_bags_on_H"] = True
    k3 = {j: {c: DR.sums(K3_DRAW[c], "P", t, levels=range(4) if c == "sham_K3" else (0,)) for c in K3_DRAW}
          for j, t in enumerate(MULTI3)}
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
                o |= {"I": interval(imp(q[s]["m"], qd[s])), "m_minus_q": interval(q[s]["m"] - qd[s]),
                      "m_over_q": interval(q[s]["m"] / qd[s])}
                rec["points"][c]["steps"][s]["intervals"].setdefault(f"L{L}", {})[mode] = o
            rec["points"][c]["mean_I"]["intervals"].setdefault(f"L{L}", {})[mode] = interval(
                np.mean([imp(q[s]["m"], qd[s]) for s in SP], 0))
            rec["points"][c]["mean_J"]["intervals"].setdefault(f"L{L}", {})[mode] = interval(
                np.mean([q[s]["J"] for s in SP], 0))
        # day only: every comparison at its full member set
        for c in COMPS:
            put(c, "day", {s: q_cols(S_all, ci, c, s) for s in SP})
        for nm, e in own["by_arm"].items():
            e["intervals"].setdefault(f"L{L}", {})["day"] = interval(
                (S_all[:, ci[f"sse_{e['Y']}_L0"]] - S_all[:, ci[f"sse_{e['X']}_L0"]]) / wn)
        for nm, (X, Y) in DIFFS.items():
            for s in SP:
                rec["paired_differences"][nm][s]["intervals"].setdefault(f"L{L}", {})["day"] = interval(
                    q_cols(S_all, ci, X, s)["m"] - q_cols(S_all, ci, Y, s)["m"])
        # day plus member: the K = 3 arms on exp-055's triples (the same triple in a draw), K = 10 on
        # exp-057's ten-index draws (reused from R7: the same generator, the same bag rule)
        per = {c: {key: np.stack([W @ k3[j][c][key] for j in range(len(MULTI3))], axis=1)[np.arange(B), d["msi"]]
                   for key in k3[0][c]} for c in K3_DRAW}
        per["carrier_K10"] = {key: v_ for key, v_ in k10_keep[L].items()}
        cols = {"draw": np.arange(B), "k3_multiset": [MS3[j] for j in d["msi"]],
                "k10_indices": [" ".join(map(str, r_)) for r_ in d["sig10"]], "wn": wn, "wn_H": wnH}
        for s in SP:
            cols[f"q_{s}"] = qd[s]
        for c in BAG_COMPS:
            put(c, "day+member", {s: quantities(wn, wnH, *(per[c][f"{st}_{s}"] for st in STATS)) for s in SP})
            for s in SP:
                for st in STATS:
                    cols[f"w{st}_{c}_{s}"] = per[c][f"{st}_{s}"]
        for c in BAG_COMPS:
            for Lv in (range(4) if c == "sham_K3" else (0,)):
                cols[f"wdsse_{c}_L{Lv}"] = per[c][f"dsse_L{Lv}"]
                if Lv == 0:
                    own["by_arm"][c]["intervals"][f"L{L}"]["day+member"] = interval(per[c]["dsse_L0"] / wn)
        for nm, (X, Y) in DIFFS.items():
            for s in SP:
                mX = np.sqrt(per[X][f"xx_{s}"] / per[X][f"yy_{s}"])
                mY = np.sqrt(per[Y][f"xx_{s}"] / per[Y][f"yy_{s}"])
                rec["paired_differences"][nm][s]["intervals"][f"L{L}"]["day+member"] = interval(mX - mY)
        dr_csv[L] = draw_csv(cols)
        print(f"intervals L{L} done ({time.time() - t_draw:.0f} s)", flush=True)
    # exp-057's K = 3 carrier intervals, reproduced through this script's route
    iw = 0.0
    for L in BLOCKS:
        for mode in ("day", "day+member"):
            for c in ("carrier_K3", "sham_K3", "carrier_K10"):
                for s in SP:
                    a_ = rec["points"][c]["steps"][s]["intervals"][f"L{L}"][mode]["m"]
                    b_ = e57["points"][c]["steps"][s]["intervals"][f"L{L}"][mode]["m"]
                    iw = max(iw, abs(a_["lo"] - b_["lo"]), abs(a_["hi"] - b_["hi"]))
    rec["checks"]["comparator_m_intervals_vs_exp057_max_abs_diff"] = iw
    if iw > TOL_9:
        raise SystemExit(f"the comparators' m intervals miss exp-057's by {iw:.3e}")

    # ---- the registered reading (SPEC 8) ---------------------------------------------------------------
    pF = rec["points"]["feb_K3"]["steps"]
    pJ = rec["points"][JS_COMP]["steps"]
    dif = rec["paired_differences"]["feb_minus_june_star"]
    bands = {s: dif[s]["intervals"]["L3"]["day+member"] for s in READ_STEPS}
    inside = {s: bool(-MARGIN <= bands[s]["lo"] and bands[s]["hi"] <= MARGIN) for s in READ_STEPS}
    above = {s: bool(bands[s]["lo"] > MARGIN) for s in READ_STEPS}
    if all(inside.values()):
        verdict = "curve-consistent"
        reading = ("curve-consistent: both complete three-day day-plus-member bands of m_Feb - m_June* at "
                   "v20 and v21 lie inside [-0.03, +0.03]. This means only that the February point adds no "
                   "visible amplitude evidence beyond the registered June block-size / own-value / "
                   "amplitude curve. It is not a causal equivalence test and it does not isolate season: "
                   "row count, per-airport composition and information at L1-L3 remain unmatched.")
    elif all(above.values()):
        verdict = "residual positive separation"
        reading = ("residual positive separation: both lower endpoints of the three-day day-plus-member "
                   "bands of m_Feb - m_June* at v20 and v21 are greater than +0.03. This does not establish "
                   "season, because row count, per-airport composition and later-level information remain "
                   "unmatched; the difference can be any of them.")
    else:
        verdict = "inconclusive"
        reading = ("inconclusive: at v20 or v21 the three-day day-plus-member band of m_Feb - m_June* "
                   "neither lies wholly inside [-0.03, +0.03] nor has its lower endpoint above +0.03. "
                   "The February point is neither shown to lie on the registered June curve nor shown to "
                   "sit above it by more than the practical margin.")
    rec["rules"] = {"registered_reading": {
        "statement": "the reading uses the paired 2.5%-97.5% three-day day-plus-member variation bands of "
                     "m_Feb - m_June* at v20 and v21, with the practical amplitude-equivalence margin "
                     "0.03: curve-consistent if both complete bands lie inside [-0.03, +0.03]; residual "
                     "positive separation if both lower endpoints are greater than +0.03; inconclusive "
                     "otherwise",
        "margin": MARGIN, "steps": list(READ_STEPS), "june_star": LADDER["june_star"],
        "bands_3day_day_plus_member": bands,
        "bands_all_block_lengths": {s: {f"L{L}": dif[s]["intervals"][f"L{L}"]["day+member"]
                                        for L in BLOCKS} for s in SP},
        "both_bands_inside_margin": inside, "both_lower_endpoints_above_margin": above,
        "points": {s: {"m_Feb": pF[s]["m"], "m_June_star": pJ[s]["m"],
                       "difference": dif[s]["difference"]} for s in SP},
        "verdict": verdict, "reading": reading,
        "not_a_test": "these are empirical day-and-member variation bands, not confidence intervals and "
                      "not a causal equivalence test; the member draw holds about 2.13 distinct members "
                      "of three and is biased toward the K = 1 value",
        "meaning": "no outcome closes a destination or a mechanism, approves a fit rule, a candidate or a "
                   "serving rule, or reopens D32; rule O remains a non-advance (exp-058); the desk sets "
                   "the path status"}}
    #: exp-058's rule O is reported unchanged as a reproduction, not re-run as a rule
    rec["rules"]["O_exp058_reproduced"] = {
        "m_Feb": {s: pF[s]["m"] for s in SP},
        "m_June_matched": {s: rec["points"]["june_matched_K3"]["steps"][s]["m"] for s in SP},
        "advances": bool(all(pF[s]["m"] >= 0.95 for s in READ_STEPS)
                         and all(rec["points"]["june_matched_K3"]["steps"][s]["m"] <= 0.85
                                 for s in READ_STEPS)),
        "note": "exp-058's rule O and its thresholds are not this experiment's rule; they are reproduced "
                "to show that nothing moved"}

    # ---- COMMON's populations (SPEC 9) and R8 (vii) ---------------------------------
    populations_block(rec, pan, lab, lev, lfj, e57, e58)

    # ---- the registered predictions (SPEC 10), read against the record ------------------------------------
    fits = {t: v_ for t, v_ in rec["new_vectors"].items() if "seconds_fit_wall" in v_}
    lad = LADDER
    lad1 = json.loads((EXP / "analysis/ladder.json").read_text())
    vN = {a: own["by_arm"][LAD[a]]["value"] for a in LADDER_ARMS}
    mJ = {s: pJ[s]["m"] for s in SP}
    dJ = {s: dif[s]["difference"] for s in SP}
    cos_pred = {"v20": 0.60, "v21": 0.68}
    ms = sorted(((ARM_N[a], a) for a in LADDER_ARMS))
    pr = {"1_plan_checks": {"all_checks_pass": PLAN["all_checks_pass"],
                            "census_amendments": len(PLAN.get("census_amendments", [])),
                            "k24_equals_exp058": PLAN["grid"]["k24_equals_exp058_matched_june_block"]},
          "2_N_sequence_reproduced": PLAN["grid"]["N_sequence_reproduced"],
          "3_reproductions_pass": {
              "R2": rec["R2_exp051_gains"]["max_abs_diff"],
              "R7_points_m": rec["R7_exp057"]["points_max_abs_diff_m"],
              "R8_points_m_cos": r8["points_max_abs_diff_m_cos"],
              "R8_own_value": r8["own_value_max_abs_diff"],
              "R8_paired_intervals": r8["paired_intervals_max_abs_diff"],
              "R8_draws_bytes_identical": [v_["bytes_identical"] for v_ in r8["draw_receipts"].values()]},
          "4_V_20261": band(vN.get("G08"), 200, 360) if "G08" in vN else None,
          "5_V_30391": band(vN.get("G12"), 270, 425) if "G12" in vN else None,
          "6_monotone": lad1["monotone"] if lad.get("monotone") is None else lad["monotone"],
          "7_branch_is_step5": {"predicted": "step5_third_point", "branch": lad1["branch"],
                                "hit": lad1["branch"] == "step5_third_point"},
          "8_third_k_is_5": {"predicted_k": 5, "band_k": [3, 7],
                             "k": lad1.get("third_point", {}).get("k"),
                             "hit": lad1.get("third_point", {}).get("k") == 5,
                             "in_band": bool(3 <= lad1.get("third_point", {}).get("k", -1) <= 7)},
          "9_third_point_V_and_band": (
              {"V": band(vN[f"G{lad1['third_point']['k']:02d}"], 150, 260),
               "predicted_outside_band": True,
               "in_band": bool(BAND[0] <= vN[f"G{lad1['third_point']['k']:02d}"] <= BAND[1])}
              if lad1["branch"] == "step5_third_point" and f"G{lad1['third_point']['k']:02d}" in vN else None),
          "10_m_June_star": {s: band(mJ[s], 0.83, 0.93) for s in READ_STEPS},
          "11_m_Feb_minus_m_June_star": {s: band(dJ[s], -0.02, 0.08) for s in READ_STEPS},
          "12_reading_is_inconclusive": {"predicted": "inconclusive", "verdict": verdict,
                                         "hit": verdict == "inconclusive"},
          "13_ladder_m_monotone_decreasing_in_N": {
              s: {"by_N": [[n_, rec["points"][LAD[a]]["steps"][s]["m"]] for n_, a in ms]
                         + [[60777, rec["points"]["june_matched_K3"]["steps"][s]["m"]]],
                  "hit": bool(all(x[1] >= y[1] for x, y in zip(
                      [[n_, rec["points"][LAD[a]]["steps"][s]["m"]] for n_, a in ms]
                      + [[60777, rec["points"]["june_matched_K3"]["steps"][s]["m"]]],
                      ([[n_, rec["points"][LAD[a]]["steps"][s]["m"]] for n_, a in ms]
                       + [[60777, rec["points"]["june_matched_K3"]["steps"][s]["m"]]])[1:])))}
              for s in READ_STEPS},
          "14_cos_G08": ({s: band(rec["points"][LAD["G08"]]["steps"][s]["cos"], cos_pred[s] - 0.12,
                                  cos_pred[s] + 0.12) for s in READ_STEPS} if "G08" in LAD else None),
          "15_J_G08": ({"mean_J": band(rec["points"][LAD["G08"]]["mean_J"]["point"], 0, 50),
                        "positive_at_v20_v21": bool(all(rec["points"][LAD["G08"]]["steps"][s]["J"] > 0
                                                        for s in READ_STEPS)),
                        "by_step": {s: rec["points"][LAD["G08"]]["steps"][s]["J"] for s in SP}}
                       if "G08" in LAD else None),
          "16_fit_cost_24": {"wall_s": band(sum(v_["seconds_fit_wall"] for v_ in fits.values()
                                                if v_["k"] in (8, 12)), 2300, 2500),
                             "process_cpu_s": band(sum(v_["seconds_fit_process_cpu"] for v_ in fits.values()
                                                       if v_["k"] in (8, 12)), 5300, 5800)} if fits else None,
          "17_inside_the_caps": {"note": "read from analysis/cost.json in RESULTS; the fit gate refused "
                                         "no launch if every registered fit has a record"}}
    rec["registered_predictions"] = pr

    # ---- OOF and row receipts (SPEC 7.4, 9) ----------------------------------------------------------------
    oofs = {}
    for f in C.FOLDS:
        pbf = pan[f]["pbase"]
        src = ({"F3": "F3bag", "M3": "M3bag", "P3": "P3bag"} | {a: f"{a}3bag" for a in LADDER_ARMS}) \
            if f == F else ({"F3": "P3bag", "M3": "P3bag", "P3": "P3bag"}
                            | {a: "P3bag" for a in LADDER_ARMS})
        o = pd.DataFrame({"MVT_ID_mvt": lab[f]["mvt_raw"], "fold": f, "y_true": lab[f]["y"],
                          "y_pred": pbf + lev[f][src[JUNE_STAR]][3]})
        for nm, sn in src.items():
            for Lv in range(4):
                o[f"y_pred_{nm}_L{Lv}"] = pbf + lev[f][sn][Lv]
        oofs[f] = o
    mem = pd.DataFrame({"MVT_ID_mvt": lab[F]["mvt_raw"], "fold": F, "y_true": y})
    for arm in LADDER_ARMS:
        for s in SEEDS3:
            for Lv in range(4):
                mem[f"y_pred_{arm}_s{s}_L{Lv}"] = pb + L7[f"{arm}{s}"][Lv]
    rows = pd.DataFrame({"MVT_ID_mvt": lab[F]["mvt_raw"], "day": pf["day"], "airport": pf["ap"], "H": H,
                         "y_true": y, "pbase": pb})
    for c in BAG_COMPS:
        X, Y, _, _ = COMPS[c]
        for k in KS:
            s = STEP[k]
            rows[f"r_{c}_{s}"] = y - pb - L7[X][k - 1]
            rows[f"dX_{c}_{s}"] = L7[X][k] - L7[X][k - 1]
            rows[f"dY_{c}_{s}"] = L7[Y][k] - L7[Y][k - 1]

    # ---- the census of every file this process opened, then every output ------------------------------------
    cc = census_check()
    rec["census_check"] = {k: v_ for k, v_ in cc.items() if k != "files"} | {"files_opened": cc["files"]}
    print(f"census: {cc['opened']} repository files opened outside exp-058; {len(cc['missing'])} not in the "
          f"census, {len(cc['differ'])} differ", flush=True)
    if cc["differ"] or (cc["missing"] and not STANDIN):
        raise SystemExit(f"census check fails: missing {cc['missing']}, differ {cc['differ']}")

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
    rp = REC_DIR / "daysums-july_loo.csv"
    rec["receipts"][nm_(rp)] = put_text(rp, text)
    if not np.array_equal(LF.day_matrix(LF.read_receipt(rp), days)[0], Dm):
        raise SystemExit("the day-by-airport receipt on disk does not rebuild the day sums")
    rp = REC_DIR / "days-july_loo.csv"
    rec["receipts"][nm_(rp)] = put_text(rp, dtext)
    if not np.array_equal(pd.read_csv(rp, float_precision="round_trip")[list(ci)].to_numpy("float64"), Dm):
        raise SystemExit("the day receipt on disk does not read back")
    for L in BLOCKS:
        rec["receipts"][nm_(REC_DIR / f"draws-L{L}.csv")] = put_text(REC_DIR / f"draws-L{L}.csv", dr_csv[L])
        rec["receipts"][nm_(REC_DIR / f"exp057-draws-L{L}.csv")] = put_text(
            REC_DIR / f"exp057-draws-L{L}.csv", e57_draw_text[f"L{L}"])
        rec["receipts"][nm_(REC_DIR / f"exp058-draws-L{L}.csv")] = put_text(
            REC_DIR / f"exp058-draws-L{L}.csv", e58_text["draws"][f"L{L}"])
    rec["receipts"][nm_(REC_DIR / "exp058-daysums-july_loo.csv")] = put_text(
        REC_DIR / "exp058-daysums-july_loo.csv", e58_text["daysums"])
    rec["oof"] = {}
    for f in C.FOLDS:
        p = OOFD / f"ownvalue-{f}.parquet"
        o = oofs[f]
        rec["oof"][nm_(p)] = {"sha256": put_parquet(p, o), "rows": int(len(o)), "columns": list(o.columns),
                              "ordered_sha256": {c: C.sha_arr(pan[f]["ids"], o[c].to_numpy("float64"))
                                                 for c in o.columns if c.startswith("y_pred")}}
    p = ART / "oof/members-july_loo.parquet"
    rec["oof"][nm_(p)] = {"sha256": put_parquet(p, mem), "rows": int(len(mem)), "columns": list(mem.columns)}
    p = ART / "receipts/rows-july_loo.parquet"
    rec["receipts"][nm_(p)] = put_parquet(p, rows)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["seconds_process_cpu"] = round(time.process_time() - c0, 1)
    C.write_json(OUT, rec)
    cost_append("measurement-standin" if STANDIN else "measurement",
                time.time() - t0, time.process_time() - c0,
                "the dry run with stand-in vectors" if STANDIN else "the official measurement")

    # ---- log -----------------------------------------------------------------------------------------------
    for c in BAG_COMPS:
        e = rec["points"][c]
        line = f"{c:16s} K={e['K']:<2d}"
        for s, q in e["steps"].items():
            line += f" | {s} m {q['m']:.4f} cos {q['cos']:.3f} J {q['J']:+8.2f}"
        print(line + f" | mean J {e['mean_J']['point']:+8.2f}", flush=True)
    for nm in ("feb_minus_june_star", "feb_minus_june_matched"):
        for s in SP:
            d_ = rec["paired_differences"][nm][s]
            iv = d_["intervals"]["L3"]["day+member"]
            print(f"{nm} {s}: {d_['difference']:+.4f}, 3-day day+member [{iv['lo']:+.4f}, {iv['hi']:+.4f}]",
                  flush=True)
    print(f"June* = {JUNE_STAR} (k = {ARM_K[JUNE_STAR]}, N = {ARM_N[JUNE_STAR]:,}), "
          f"V = {own['by_arm'][JS_COMP]['value']:+.2f} s2/row", flush=True)
    print(f"verdict: {verdict}", flush=True)
    print(f"reading: {reading}", flush=True)
    print("own value L0: " + ", ".join(
        f"{k_} {own['by_arm'][k_]['value']:+.1f}" for k_ in
        ["feb_K3"] + [LAD[a] for a in LADDER_ARMS] + ["june_matched_K3", "carrier_K3"]) + " s2/row", flush=True)
    print(f"written {nm_(OUT)} ({rec['seconds']} s wall, {rec['seconds_process_cpu']} s process cpu)"
          f"{' -- STAND-IN RUN, DISCARD' if STANDIN else ''}", flush=True)


if __name__ == "__main__":
    main()
