"""exp-057 originfit: every registered measurement (SPEC sections 4-7).  ZERO FIT.  READS LABELS.

Order (SPEC section 5): the reproductions come first, and no new vector file is opened until all
of them pass.
  R1  exp-055 `originvec.load()` re-verifies every exp-055 ordered pin.
  R2  exp-051's stored gains, with exp-051's arithmetic, from `carrier.json` read through a
      reader that records every key it gives and refuses any other.
  R3/R4  exp-055's `screen.py` and `lossfals.py`, unchanged, run into a scratch directory; the
      registered fields of their records must equal the committed `screen.json` and
      `lossfals.json` (the whole records are compared and the differences reported).  This
      script's own arithmetic must reproduce the same points before any new vector is opened.
  R5  the regenerated draws carry `screen.json`'s `draw_sha256` and `multiset_counts`.
  R6  the plain K = 10 bag rebuilt from its ten members equals the saved heads; S1 carries
      exp-051's recorded hashes; C3 and P3 are exp-055's pinned bags.
Then the new vectors (sha-checked against their fit records), the bags S3 = bag(S1..S3) and
C10 = bag(C1..C10), the label-free movement statistics, the loss equations, the sham's damage,
the moving-block intervals with matched member draws, the rules S and Z, COMMON's populations,
and the receipts.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-057-originfit/code/ofit_measure.py
    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-057-originfit/code/ofit_measure.py --standin DIR
      (the SPEC 10 dry run: each new member is replaced by an existing vector, nothing under
       artifacts/exp-057 may be opened, every output goes under DIR and is discarded)
"""
from __future__ import annotations

import argparse
import hashlib
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
E57_ART = "artifacts/exp-057/"
E57_EXP = "experiments/exp-057-originfit/"
PY = str(ROOT / ".venv/bin/python")

ap_ = argparse.ArgumentParser()
ap_.add_argument("--standin", help="dry run: stand-in vectors, every output under this directory")
ARGS = ap_.parse_args()
STANDIN = ARGS.standin is not None

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
        if STANDIN and rel.startswith(E57_ART):
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

sys.path.insert(0, str(E55 / "code"))
import common as C                                                    # noqa: E402  (exp-055)
import lossfals as LF                                                 # noqa: E402
import originvec as V                                                 # noqa: E402
import screen as SC                                                   # noqa: E402

for _name, _want in (("common", E55 / "code/common.py"), ("originvec", E55 / "code/originvec.py"),
                     ("screen", E55 / "code/screen.py"), ("lossfals", E55 / "code/lossfals.py")):
    if Path(sys.modules[_name].__file__).resolve() != _want:
        raise SystemExit(f"module {_name} resolves to {sys.modules[_name].__file__} (module name collision)")

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
SEEDS10 = tuple(range(1, 11))
MULTI3 = list(itertools.combinations_with_replacement((1, 2, 3), 3))
MS3 = ["".join(map(str, t)) for t in MULTI3]
CUT = 3600.0
STATS = ("ry", "yy", "rx", "xx", "xy")
#: SPEC 6.7 (exp-055 screen.json q["july_loo->2026-07"][v]["point"] and ["intervals"]["L3"])
Q_POINT = {"v20": 0.7541214069685656, "v21": 0.7872337829612198, "v22": 0.5869686712745045}
Q_L3 = {"v20": (0.7030750838597688, 0.7941267683291252), "v21": (0.7364214976003001, 0.8322645424736729),
        "v22": (0.555629780110832, 0.610847845536199)}
RULE_S = {"m_S_min": 0.95, "m_C_max": 0.85, "steps": ("v20", "v21")}
SCREEN_SHA = "081c39ed4789c5fc4851fe61808bf1cc28b13d5eee1e1e974a768faa17721ae9"
LOSSFALS_SHA = "1eecc0b52e517a5f7749c0a75ce6b9529cb268b708f163402945ce337e64a52f"
CARRIER_SHA = "dc49aa89816140365111aa601965ba6bcb6b4735340bf64f8f7aa4c92dbe25b4"
SPEC_COMMIT = "95d82b6611e2768e2a25fd596bd113f168eb307e"
TOL_12, TOL_9 = 1e-12, 1e-9
#: SPEC 10: stand-in members for the dry run
STANDIN_SRC = {("S", 2): "S1", ("S", 3): "S1", ("C", 4): "C1", ("C", 5): "C2", ("C", 6): "C3", ("C", 7): "C1",
               ("C", 8): "C2", ("C", 9): "C3", ("C", 10): "C1"}

if STANDIN:
    BASE = Path(ARGS.standin).resolve()
    if str(BASE).startswith(ROOT_S):
        raise SystemExit("the stand-in directory must lie outside the repository")
    ANALYSIS, OOFD, ART = BASE / "analysis", BASE / "oof", BASE / "artifacts"
else:
    ANALYSIS, OOFD, ART = EXP / "analysis", EXP / "oof", ROOT / "artifacts/exp-057"
OUT = ANALYSIS / "measure.json"
REC_DIR = ANALYSIS / "receipts"
REPRO = ART / "repro"

CENSUS = json.loads((EXP / "analysis/prehash.json").read_text())["files"]
for _am in sorted((EXP / "analysis").glob("prehash-a*.json")):
    for _r, _v in json.loads(_am.read_text())["files"].items():
        if _r in CENSUS and CENSUS[_r]["sha256"] != _v["sha256"]:
            raise SystemExit(f"{_am.name} changes the census hash of {_r}")
        CENSUS.setdefault(_r, _v)
PLAN = json.loads((EXP / "analysis/plan.json").read_text())


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check57(p: Path) -> str:
    """The file's bytes must carry exp-057's census hash for its path."""
    rel = str(Path(p).resolve().relative_to(ROOT))
    if rel not in CENSUS:
        raise SystemExit(f"{rel} is not in exp-057's census")
    got = sha_file(p)
    if got != CENSUS[rel]["sha256"]:
        raise SystemExit(f"{rel} is {got[:16]}, not the census {CENSUS[rel]['sha256'][:16]}")
    return got


def interval(x) -> dict:
    return LF.interval(x)


def quantities(n, nH, ry, yy, rx, xx, xy) -> dict:
    """SPEC 6.1-6.2 (exp-055 lossfals.quantities, plus the best amplitude and the rms)."""
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
    if check57(p) != CARRIER_SHA:
        raise SystemExit("carrier.json is not the pinned record")
    allowed = set()
    for s in SP:
        allowed |= {f"steps.{s}.july_s2_per_row.C3", f"steps.{s}.july_s2_per_row.P3",
                    f"steps.{s}.sham_K1.july_s2.S", f"steps.{s}.sham_K1.july_s2.P1",
                    f"steps.{s}.january_s2_per_row_both_arms", f"steps.{s}.december_s2_per_row_both_arms"}
        for m in (1, 2, 3):
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
                for m in (1, 2, 3):
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
    """exp-055's screen.py and lossfals.py, unchanged, into a scratch directory; SPEC 5 R3 and R4."""
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
        if check57(want_path) != want_sha:
            raise SystemExit(f"{want_path} is not the pinned record")
        g, w = json.loads((d / rel_out).read_text()), json.loads(want_path.read_text())
        got[sub] = (g, w)
        diffs, wa, wr, nd = jdiff(g, w, skip=("seconds", "receipts", "oof"))
        rsha = {Path(k).name: v for k, v in g["receipts"].items()}
        wsha = {Path(k).name: v for k, v in w["receipts"].items()}
        oofeq = {k: g["oof"][k] == w["oof"][k] for k in ("sha256", "rows", "columns", "ordered_sha256")} if "oof" in w else None
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
                reg.append((f"R3 {arm} {s} {k_}", sg["arms"][arm]["steps"][s][k_], sw["arms"][arm]["steps"][s][k_], TOL_12))
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
                reg.append((f"R4 {pt} {s} {k_}", lg["points"][F][pt]["steps"][s][k_], lw["points"][F][pt]["steps"][s][k_], TOL_9))
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
           "seed2": ("C2", "P2", "june_seed2"), "seed3": ("C3", "P3", "june_seed3"), "sham": ("S1", "P1", "sham")}
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


# ------------------------------------------------------------------------------ vectors --------

def load_new(vec: dict) -> dict:
    """The 36 new July vectors (or, in the dry run, their stand-ins)."""
    info = {}
    for kind, seeds in (("S", (2, 3)), ("C", range(4, 11))):
        for s in seeds:
            for L in range(4):
                tag = f"{kind}-s{s}-L{L}"
                if STANDIN:
                    src = f"{STANDIN_SRC[(kind, s)]}_L{L}"
                    vec[F][f"{kind}{s}_L{L}"] = vec[F][src].copy()
                    info[tag] = {"standin_for": src}
                    continue
                rec = json.loads((EXP / "analysis/fits" / f"{tag}.json").read_text())
                if rec["code_sha256"] != PLAN["code"]["sha256"] or rec.get("spec_commit", SPEC_COMMIT) != SPEC_COMMIT:
                    raise SystemExit(f"{tag}: the record's code or registration is not the plan's")
                p = ROOT / rec["vector"]
                if sha_file(p) != rec["vector_sha256"]:
                    raise SystemExit(f"{tag}: the vector is not the recorded bytes")
                z = np.load(p)
                if list(z.keys()) != [F]:
                    raise SystemExit(f"{tag}: the vector holds {list(z.keys())}")
                vec[F][f"{kind}{s}_L{L}"] = z[F].astype("float64")
                info[tag] = {"vector": rec["vector"], "sha256": rec["vector_sha256"], "rows_moved": rec["rows_moved"],
                             "seconds_fit_wall": rec["seconds_fit_wall"],
                             "seconds_fit_process_cpu": rec["seconds_fit_process_cpu"]}
    return info


def bags(pan: dict, vec: dict) -> dict:
    """Level vectors by fold; the bag rule (exp-055 `common.bag`, members in ascending seed order)."""
    lev = {}
    for f in C.FOLDS:
        pf, v = pan[f], vec[f]

        def bag(nm, seeds, L):
            return C.bag({s: v[f"{nm}{s}_L{L}"] for s in seeds}, seeds, pf["pbase"], pf["usable"], pf["guard"])
        lv = {nm: [v[f"{nm}_L{L}"] for L in range(4)] for nm in [f"P{s}" for s in SEEDS10] + ["P3bag", "T10"]}
        lv["P10bag"] = [bag("P", SEEDS10, L) for L in range(4)]
        if f == F:
            for nm in [f"C{s}" for s in SEEDS10] + ["S1", "S2", "S3", "C3bag"]:
                lv[nm] = [v[f"{nm}_L{L}"] for L in range(4)]
            lv["S3bag"] = [bag("S", (1, 2, 3), L) for L in range(4)]
            lv["C10bag"] = [bag("C", SEEDS10, L) for L in range(4)]
            for L in range(4):
                if not (np.array_equal(bag("C", (1, 2, 3), L), lv["C3bag"][L])
                        and np.array_equal(bag("P", (1, 2, 3), L), lv["P3bag"][L])):
                    raise SystemExit("the K = 3 bags do not rebuild exp-055's pinned bags")
        for nm, x in lv.items():
            if nm.endswith("bag") or nm == "T10":
                for k in KS:
                    if np.any((x[k] - x[k - 1])[~pf["H"]] != 0):
                        raise SystemExit(f"{f} {nm} {STEP[k]}: a bag step moves a row outside H")
        lev[f] = lv
    return lev


# ------------------------------------------------------------------------------ statistics -----

#: SPEC 6.1 comparisons on July H: name -> (carrier X, plain Y, K, role)
COMPS = {"sham_K3": ("S3bag", "P3bag", 3, "rule S: the K = 3 sham"),
         "carrier_K3": ("C3bag", "P3bag", 3, "rule S: the K = 3 June carrier (exp-055's point, reproduced)"),
         "carrier_K10": ("C10bag", "P10bag", 10, "rule Z: the K = 10 June carrier")}
COMPS |= {f"sham_s{s}": (f"S{s}", f"P{s}", 1, "sham member beside the K = 3 sham bag") for s in (1, 2, 3)}
COMPS |= {f"carrier_s{s}": (f"C{s}", f"P{s}", 1, "carrier member beside the K = 10 carrier bag") for s in SEEDS10}
BAG_COMPS = ("sham_K3", "carrier_K3", "carrier_K10")
SSE_NAMES = ["P3bag", "S3bag", "C3bag", "P10bag", "C10bag", "S1", "S2", "S3", "P1", "P2", "P3"]


def cells_july(pan: dict, lv: dict, y: np.ndarray) -> pd.DataFrame:
    """Per (UTC day, airport) sums over every July row (SPEC 6.6), by exp-055 lossfals's route."""
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
            for st, v in (("ry", r * dy), ("yy", dy * dy), ("rx", r * dx), ("xx", dx * dx), ("xy", dx * dy)):
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

    def __init__(self, pan: dict, lv: dict, y: np.ndarray, days: np.ndarray):
        pf = pan[F]
        self.H = pf["H"]
        self.pbH = pf["pbase"][self.H]
        self.yH = y[self.H]
        self.dc = np.searchsorted(days, pf["day"][self.H])
        if not np.array_equal(days[self.dc], pf["day"][self.H]):
            raise SystemExit("an H row's day is not a July fold day")
        self.nd = len(days)
        self.M = {nm: [np.vstack([lv[f"{nm}{s}"][L][self.H] for s in SEEDS10 if f"{nm}{s}" in lv])
                       for L in range(4)] for nm in ("P", "C", "S")}

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


def census_check() -> dict:
    opened = sorted(r for r in OPENED if not r.startswith((E57_EXP, E57_ART)) and (ROOT / r).is_file())
    miss = [r for r in opened if r not in CENSUS]
    diff = [r for r in opened if r in CENSUS and sha_file(ROOT / r) != CENSUS[r]["sha256"]]
    return {"opened": len(opened), "missing": miss, "differ": diff, "files": opened}


def main() -> None:
    t0 = time.time()
    np.seterr(divide="ignore", invalid="ignore")
    outs = [OUT, REC_DIR / "daysums-july_loo.csv", REC_DIR / "days-july_loo.csv",
            *(REC_DIR / f"draws-L{L}.csv" for L in BLOCKS), *(OOFD / f"originfit-{f}.parquet" for f in C.FOLDS),
            ART / "receipts/rows-july_loo.parquet", ART / "oof/members-july_loo.parquet"]
    NO.refuse_existing(outs, overwrite=False)
    rec = {"what": __doc__.strip().splitlines()[0], "spec_commit": SPEC_COMMIT, "standin": STANDIN,
           "plan_code_sha256": PLAN["code"]["sha256"], "B": B, "blocks": list(BLOCKS),
           "generator": "np.random.default_rng(5500 + L)",
           "draw_order": ["july_loo fold starts", "2026-07 ranking starts", "january fold starts",
                          "2026-01 ranking starts", "december fold starts", "K = 3 member triples (B x 3)",
                          "K = 10 member-index vectors (B x 10), new here"],
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
    rec["R3_R4_exp055_scripts"] = r34()
    for sub, r_ in rec["R3_R4_exp055_scripts"]["runs"].items():
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
        draws[L] = {"st": st, "msi": msi, "sig10": np.sort(sig10, axis=1), "sig10_sha256": C.sha_arr(sig10)}
    rec["checks"]["R5_draws_equal_exp055"] = True
    rec["k10_member_draw_sha256_unsorted"] = {f"L{L}": draws[L]["sig10_sha256"] for L in BLOCKS}
    print("R5 the regenerated draws equal exp-055's", flush=True)

    # ---- R6 -------------------------------------------------------------------------------------
    for f in C.FOLDS:
        for s in range(4, 11):
            for L in range(4):
                p = C.member_path(L, s)
                check57(p)
                vec[f][f"P{s}_L{L}"] = np.load(p)[f].astype("float64")
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
                 "S1_equals_exp051_fit_records": s1, "C3_P3_are_exp055_pinned_bags": "checked in bags()"}
    print(f"R6 the plain K = 10 bag rebuilds the saved heads (max |diff| {worst10:.1e} s); S1 is exp-051's", flush=True)

    # ---- the new vectors --------------------------------------------------------------------------
    rec["new_vectors"] = load_new(vec)
    lev = bags(pan, vec)
    L7 = lev[F]
    rec["checks"]["new_vectors_sha_checked_against_fit_records"] = not STANDIN
    rec["checks"]["every_bag_step_zero_outside_H"] = True
    rec["off_july"] = ("the new fits are July-only (exp-051's design): on december and january S3 = P3 and "
                       "C10 = P10 by construction, and each is reported once as P3bag and P10bag")
    print(f"new vectors loaded ({'STAND-INS' if STANDIN else '36 fits, sha-checked'})", flush=True)

    # ---- receipts and the day matrix -----------------------------------------------------------------
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
    # carrier_K3's cells are exp-055's bag123 cells (the same route): its i2 receipt
    lf_cells = LF.read_receipt(E55 / "analysis/receipts/i2-daysums-july_loo.csv")
    check57(E55 / "analysis/receipts/i2-daysums-july_loo.csv")
    worst_cell = 0.0
    for s in SP:
        for st in STATS:
            a_ = back[f"{st}_carrier_K3_{s}"].to_numpy("float64")
            b_ = lf_cells[f"{st}_bag123_{s}"].to_numpy("float64")
            worst_cell = max(worst_cell, float(np.max(np.abs(a_ - b_))))
    rec["checks"]["carrier_K3_cells_vs_exp055_i2_receipt_max_abs_diff"] = worst_cell

    # ---- points --------------------------------------------------------------------------------------
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
                  "direct_sse_gain_all_rows": sse, "g_P_own_p0_all_rows": float(np.sum(2 * ry_ * dy - dy * dy) / n_all)}
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
            q["kish_days_mass"] = {"carrier": C.kish(Dm[:, ci[f"xx_{c}_{s}"]]), "plain": C.kish(Dm[:, ci[f"yy_{c}_{s}"]])}
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
    rec["sham_minus_carrier_K3"] = {s: {"m_S_minus_m_C": rec["points"]["sham_K3"]["steps"][s]["m"]
                                        - rec["points"]["carrier_K3"]["steps"][s]["m"], "intervals": {}} for s in SP}

    # ---- the sham's damage and the carrier's own value (SPEC 6.3) ---------------------------------------
    dmg = {"definition": "D_L = [SSE(plain_L) - SSE(sham_L)] / n on July (negative: the sham predictor is worse); "
                         "measured, never subtracted", "K3": {}, "members": {}, "own_value_L0": {}}
    for L in range(4):
        dmg["K3"][f"L{L}"] = {"D": fl((P_[ci[f"sse_P3bag_L{L}"]] - P_[ci[f"sse_S3bag_L{L}"]]) / n_all), "intervals": {}}
        for s in (1, 2, 3):
            dmg["members"].setdefault(f"s{s}", {})[f"L{L}"] = {
                "D": fl((P_[ci[f"sse_P{s}_L{L}"]] - P_[ci[f"sse_S{s}_L{L}"]]) / n_all), "intervals": {}}
    for kk, (X, Y) in (("K3", ("C3bag", "P3bag")), ("K10", ("C10bag", "P10bag"))):
        dmg["own_value_L0"][kk] = {"value": fl((P_[ci[f"sse_{Y}_L0"]] - P_[ci[f"sse_{X}_L0"]]) / n_all), "intervals": {}}
    rec["damage"] = dmg

    # ---- decompositions (SPEC 6.1) ---------------------------------------------------------------------
    g_air = back.groupby("airport").sum(numeric_only=True)
    qa = scr["decompositions"]["airport_q"]["july_loo->2026-07"]
    dec = {"airport": {}, "nm_regime": {}}
    w_dec = 0.0
    for c, (X, Y, K, _) in COMPS.items():
        dec["airport"][c], dec["nm_regime"][c] = {}, {}
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
    rec["checks"]["decomposition_max_reconciliation_error"] = w_dec
    if w_dec > TOL_9:
        raise SystemExit(f"a decomposition does not reconcile: {w_dec:.3e}")
    rec["decompositions"] = dec

    # ---- the fit-count record (SPEC 6.1) -----------------------------------------------------------------
    fc = {}
    names = ([f"P{i}" for i in SEEDS10] + [f"C{i}" for i in SEEDS10] + ["S1", "S2", "S3", "P3bag", "C3bag", "S3bag",
                                                                         "P10bag", "C10bag", "T10"])
    for k in KS:
        s = STEP[k]
        stp = {nm: (L7[nm][k] - L7[nm][k - 1])[H] for nm in names}
        d10 = stp["P10bag"]
        fc[s] = {"rms_H_step_s": {nm: float(np.sqrt(np.sum(stp[nm] ** 2) / nH)) for nm in names},
                 "m": {"K1": {f"s{i}": rec["points"][f"carrier_s{i}"]["steps"][s]["m"] for i in SEEDS10},
                       "K3": rec["points"]["carrier_K3"]["steps"][s]["m"],
                       "K10": rec["points"]["carrier_K10"]["steps"][s]["m"],
                       "sham_K1": {f"s{i}": rec["points"][f"sham_s{i}"]["steps"][s]["m"] for i in (1, 2, 3)},
                       "sham_K3": rec["points"]["sham_K3"]["steps"][s]["m"]},
                 "cos_with_K10_plain_step": {nm: float(np.sum(stp[nm] * d10) / np.sqrt(np.sum(stp[nm] ** 2) * np.sum(d10 ** 2)))
                                             for nm in names if nm != "P10bag"}}
    rec["fit_count_record_july_H"] = fc

    # ---- intervals (SPEC 6.5) ------------------------------------------------------------------------------
    rpath = E55 / "analysis/receipts/i1-daysums-rank-2026-07.csv"
    check57(rpath)
    Dr, cir = SC.day_matrix(SC.read_receipt(rpath), np.array([f"2026-07-{d:02d}" for d in range(1, 32)]))
    for s in SP:
        qpt = float(np.sqrt((Dr.sum(0)[cir[f"ss_UP_{s}"]] / Dr.sum(0)[cir["n"]])
                            / (P_[ci[f"ss_T10_{s}"]] / P_[ci["n_H"]])))
        if abs(qpt - Q_POINT[s]) > TOL_12:
            raise SystemExit(f"{s}: q from the two receipts {qpt!r} is not {Q_POINT[s]!r}")
    DR = Draws(pan, L7, y, days)
    for idx, X, want in (((1, 2, 3), "C", "C3bag"), ((1, 2, 3), "S", "S3bag"), (SEEDS10, "C", "C10bag"),
                         (SEEDS10, "P", "P10bag"), ((1, 2, 3), "P", "P3bag")):
        for L in range(4):
            if not np.array_equal(DR.bag(X, idx, L), L7[want][L][H]):
                raise SystemExit(f"the draw route's bag {want} L{L} is not the point bag on H")
    rec["checks"]["draw_route_bags_equal_point_bags_on_H"] = True
    k3 = {j: {"sham": DR.sums("S", "P", t, levels=range(4)), "carrier": DR.sums("C", "P", t)}
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
        for Lv in range(4):
            dmg["K3"][f"L{Lv}"]["intervals"].setdefault(f"L{L}", {})["day"] = interval(
                (S_all[:, ci[f"sse_P3bag_L{Lv}"]] - S_all[:, ci[f"sse_S3bag_L{Lv}"]]) / wn)
            for s in (1, 2, 3):
                dmg["members"][f"s{s}"][f"L{Lv}"]["intervals"].setdefault(f"L{L}", {})["day"] = interval(
                    (S_all[:, ci[f"sse_P{s}_L{Lv}"]] - S_all[:, ci[f"sse_S{s}_L{Lv}"]]) / wn)
        for kk, (X, Y) in (("K3", ("C3bag", "P3bag")), ("K10", ("C10bag", "P10bag"))):
            dmg["own_value_L0"][kk]["intervals"].setdefault(f"L{L}", {})["day"] = interval(
                (S_all[:, ci[f"sse_{Y}_L0"]] - S_all[:, ci[f"sse_{X}_L0"]]) / wn)
        for s in SP:
            rec["sham_minus_carrier_K3"][s]["intervals"].setdefault(f"L{L}", {})["day"] = interval(
                q_cols(S_all, ci, "sham_K3", s)["m"] - q_cols(S_all, ci, "carrier_K3", s)["m"])
        # day plus member: K = 3 on exp-055's triples (the same triple for the sham and the carrier)
        per = {}
        for arm in ("sham", "carrier"):
            per[arm] = {key: np.stack([W @ k3[j][arm][key] for j in range(len(MULTI3))], axis=1)[np.arange(B), d["msi"]]
                        for key in k3[0][arm]}
        # K = 10 on the new ten-index draws
        k10 = None
        for b in range(B):
            sm = DR.sums("C", "P", tuple(d["sig10"][b]))
            if k10 is None:
                k10 = {key: np.empty(B) for key in sm}
            for key, v_ in sm.items():
                k10[key][b] = W[b] @ v_
        per["carrier10"] = k10
        cols = {"draw": np.arange(B), "k3_multiset": [MS3[j] for j in d["msi"]],
                "k10_indices": [" ".join(map(str, r_)) for r_ in d["sig10"]], "wn": wn, "wn_H": wnH}
        for s in SP:
            cols[f"q_{s}"] = qd[s]
        for c, arm in (("sham_K3", "sham"), ("carrier_K3", "carrier"), ("carrier_K10", "carrier10")):
            pr_ = per[arm]
            put(c, "day+member", {s: quantities(wn, wnH, *(pr_[f"{st}_{s}"] for st in STATS)) for s in SP})
            for s in SP:
                for st in STATS:
                    cols[f"w{st}_{c}_{s}"] = pr_[f"{st}_{s}"]
        for s in SP:
            ms_ = np.sqrt(per["sham"][f"xx_{s}"] / per["sham"][f"yy_{s}"])
            mc_ = np.sqrt(per["carrier"][f"xx_{s}"] / per["carrier"][f"yy_{s}"])
            rec["sham_minus_carrier_K3"][s]["intervals"][f"L{L}"]["day+member"] = interval(ms_ - mc_)
        for Lv in range(4):
            dmg["K3"][f"L{Lv}"]["intervals"][f"L{L}"]["day+member"] = interval(per["sham"][f"dsse_L{Lv}"] / wn)
            cols[f"wdsse_sham_K3_L{Lv}"] = per["sham"][f"dsse_L{Lv}"]
        dmg["own_value_L0"]["K3"]["intervals"][f"L{L}"]["day+member"] = interval(per["carrier"]["dsse_L0"] / wn)
        dmg["own_value_L0"]["K10"]["intervals"][f"L{L}"]["day+member"] = interval(per["carrier10"]["dsse_L0"] / wn)
        cols["wdsse_carrier_K3_L0"] = per["carrier"]["dsse_L0"]
        cols["wdsse_carrier_K10_L0"] = per["carrier10"]["dsse_L0"]
        dr_csv[L] = pd.DataFrame(cols).to_csv(index=False, float_format="%.17g")
        print(f"intervals L{L} done ({time.time() - t_draw:.0f} s)", flush=True)
    iw = 0.0
    for L in BLOCKS:
        for mode, m55 in (("day", "day"), ("day+member", "day+seed")):
            a_ = rec["points"]["carrier_K3"]["mean_J"]["intervals"][f"L{L}"][mode]
            b_ = lfj["points"][F]["bag123"]["mean_J"]["intervals"][f"L{L}"][m55]
            c_ = rec["points"]["carrier_K3"]["mean_I"]["intervals"][f"L{L}"][mode]
            e_ = scr["arms"]["june_bag"]["mean_I"]["intervals"][f"L{L}"][m55]
            iw = max(iw, abs(a_["lo"] - b_["lo"]), abs(a_["hi"] - b_["hi"]), abs(c_["lo"] - e_["lo"]), abs(c_["hi"] - e_["hi"]))
    rec["checks"]["carrier_K3_intervals_vs_exp055_max_abs_diff"] = iw
    if iw > TOL_9:
        raise SystemExit(f"this script's K = 3 carrier intervals miss exp-055's by {iw:.3e}")

    # ---- the rules (SPEC 6.7) -------------------------------------------------------------------------------
    pS, pC, pZ = (rec["points"][c]["steps"] for c in BAG_COMPS)
    cond = {f"m_S_{s}_ge_0.95": bool(pS[s]["m"] >= RULE_S["m_S_min"]) for s in RULE_S["steps"]}
    cond |= {f"m_C_{s}_le_0.85": bool(pC[s]["m"] <= RULE_S["m_C_max"]) for s in RULE_S["steps"]}
    dm = {s: {f"L{L}": pS[s]["intervals"][f"L{L}"]["day+member"]["m"] for L in BLOCKS} for s in SP}
    cm = {s: {f"L{L}": pC[s]["intervals"][f"L{L}"]["day+member"]["m"] for L in BLOCKS} for s in SP}
    ruleS = {"statement": "ADVANCES only if m_S >= 0.95 at v20 and at v21 (the K = 3 sham bag points) and "
                          "m_C <= 0.85 at v20 and at v21 (the K = 3 carrier bag points); v22 reported, not in the rule",
             "conditions": cond,
             "points": {s: {"m_S": pS[s]["m"], "m_C": pC[s]["m"], "m_S_day_member_intervals": dm[s],
                            "m_C_day_member_intervals": cm[s]} for s in SP},
             "advances": bool(all(cond.values()))}
    condZ = {s: bool(Q_L3[s][0] <= pZ[s]["m"] <= Q_L3[s][1]) for s in SP}
    ruleZ = {"statement": "ADVANCES only if q_lo <= m_K10 <= q_hi at each of v20, v21 and v22 (exp-055's "
                          "3-day q intervals)",
             "q_interval_L3": {s: list(Q_L3[s]) for s in SP}, "inside": condZ,
             "points": {s: {"m": pZ[s]["m"], "q": Q_POINT[s], "m_minus_q": pZ[s]["m_minus_q"],
                            "m_over_q": pZ[s]["m_over_q"], "cos": pZ[s]["cos"],
                            "rms_H_carrier_s": pZ[s]["rms_H_carrier_s"], "rms_H_plain_s": pZ[s]["rms_H_plain_s"],
                            "m_day_member_intervals": {f"L{L}": pZ[s]["intervals"][f"L{L}"]["day+member"]["m"]
                                                       for L in BLOCKS}} for s in SP},
             "advances": bool(all(condZ.values()))}
    rec["rules"] = {"S": ruleS, "Z": ruleZ,
                    "meaning": "an advance supports only further study of the signature as a pre-loss-read diagnostic; "
                               "a non-advance closes no destination and no mechanism; the desk sets the path status"}

    # ---- COMMON's populations (SPEC 7) ----------------------------------------------------------------------
    check57(C.E53 / "analysis/pin.json")
    qpin = json.loads((C.E53 / "analysis/pin.json").read_text())
    rec["populations"] = {}
    pw = 0.0
    for f in C.FOLDS:
        ids, yy, pbf = pan[f]["ids"], lab[f]["y"], pan[f]["pbase"]
        nonm = LF.no_nm_mask(ids, f)
        if C.sha_arr(nonm) != qpin["folds"][f]["no_nm_sha256"]:
            raise SystemExit(f"{f}: the no-NM mask is not Q15's")
        masks = {"all": np.ones(len(ids), bool), "le3600": yy <= CUT, "gt3600": yy > CUT, "no_nm": nonm,
                 "clean": lab[f]["clean"]}
        masks |= {f"own_{g}": pan[f]["reg"] == g for g in C.REGIMES}
        masks |= {f"nm_{r_}": lab[f]["nmreg"] == r_ for r_ in sorted(set(lab[f]["nmreg"]))}
        preds = ["S3bag", "C10bag", "P10bag", "P3bag", "C3bag"] if f == F else ["P3bag", "P10bag"]
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
        for nm in ("P3bag", "C3bag") if f == F else ("P3bag",):
            for Lv in range(4):
                for k in masks:
                    ref = lfj["populations"][f][f"{nm}_L{Lv}"][k]["rmse_s"]
                    if ref is not None:
                        pw = max(pw, abs(ref - rec["populations"][f][f"{nm}_L{Lv}"][k]))
    rec["checks"]["populations_vs_exp055_max_abs_diff_s"] = pw
    if pw > TOL_9:
        raise SystemExit(f"P3 or C3 populations differ from exp-055's by {pw:.3e} s")

    # ---- registered predictions (SPEC 8), read against the record --------------------------------------------
    def band(v, lo, hi):
        return {"point": v, "band": [lo, hi], "inside": bool(lo <= v <= hi)}
    fits = {t: v_ for t, v_ in rec["new_vectors"].items() if "seconds_fit_wall" in v_}
    m1 = {s: rec["points"]["sham_s1"]["steps"][s]["m"] for s in SP}
    pr = {"2_P10_vs_heads_le_1e-12": {"max_abs_diff_s": worst10, "hit": bool(worst10 <= TOL_12)},
          "3_m_S": {s: band(pS[s]["m"], *b) for s, b in zip(SP, ([0.55, 0.85], [0.62, 0.90], [1.5, 2.6]))},
          "4_rule_S_does_not_advance": not ruleS["advances"],
          "5_m_S_minus_m_C": {s: band(rec["sham_minus_carrier_K3"][s]["m_S_minus_m_C"], c_ - 0.12, c_ + 0.12)
                              for s, c_ in (("v20", -0.10), ("v21", -0.04))},
          "6_sham_members_within_0.05_of_member_1": {
              f"s{i}": {s: {"m": rec["points"][f"sham_s{i}"]["steps"][s]["m"], "member_1": m1[s],
                            "inside": bool(abs(rec["points"][f"sham_s{i}"]["steps"][s]["m"] - m1[s]) <= 0.05)}
                        for s in SP} for i in (2, 3)},
          "7_m_K10": {s: band(pZ[s]["m"], *b) for s, b in zip(SP, ([0.67, 0.78], [0.68, 0.80], [0.71, 0.83]))},
          "8_rule_Z_does_not_advance": {"hit": not ruleZ["advances"], "v22_above_q_hi": bool(pZ["v22"]["m"] > Q_L3["v22"][1]),
                                        "v20_inside": condZ["v20"]},
          "9_cos_K10": {s: band(pZ[s]["cos"], c_ - 0.12, c_ + 0.12) for s, c_ in (("v20", 0.45), ("v21", 0.53), ("v22", 0.58))},
          "10_K10_J": {"mean": band(rec["points"]["carrier_K10"]["mean_J"]["point"], 5, 70),
                       "all_steps_positive": bool(all(pZ[s]["J"] > 0 for s in SP)), "by_step": {s: pZ[s]["J"] for s in SP}},
          "11_K10_alignment_larger_than_amplitude": {
              s: {"alignment": pZ[s]["alignment"], "amplitude": pZ[s]["amplitude"],
                  "abs_alignment_larger": bool(abs(pZ[s]["alignment"]) > abs(pZ[s]["amplitude"]))} for s in SP},
          "12_K3_sham_J": {"v21_positive": bool(pS["v21"]["J"] > 0), "v20": pS["v20"]["J"],
                           "v22": band(pS["v22"]["J"], -1000, -200)},
          "13_D0_K3": band(dmg["K3"]["L0"]["D"], -4600, -3200),
          "14_D0_members": {f"s{i}": band(dmg["members"][f"s{i}"]["L0"]["D"], -4700, -3300) for i in (2, 3)},
          "15_own_value_L0_K10": band(dmg["own_value_L0"]["K10"]["value"], 480, 720),
          "17_fit_wall_minutes": {"total": round(sum(v_["seconds_fit_wall"] for v_ in fits.values()) / 60, 1) if fits else None,
                                  "band": [60, 70]},
          "note": "items 1 (plan checks) and 16 (capacity) are read from analysis/plan.json and analysis/capacity.json"}
    rec["registered_predictions"] = pr

    # ---- OOF and row receipts (SPEC 6.6, 7) ---------------------------------------------------------------------
    oofs = {}
    for f in C.FOLDS:
        pbf = pan[f]["pbase"]
        src = {"S3": "S3bag", "C10": "C10bag", "P10": "P10bag"} if f == F else {"S3": "P3bag", "C10": "P10bag", "P10": "P10bag"}
        o = pd.DataFrame({"MVT_ID_mvt": lab[f]["mvt_raw"], "fold": f, "y_true": lab[f]["y"],
                          "y_pred": pbf + lev[f][src["C10"]][3]})
        for nm, sn in src.items():
            for Lv in range(4):
                o[f"y_pred_{nm}_L{Lv}"] = pbf + lev[f][sn][Lv]
        oofs[f] = o
    mem = pd.DataFrame({"MVT_ID_mvt": lab[F]["mvt_raw"], "fold": F, "y_true": y})
    for kind, seeds in (("S", (2, 3)), ("C", range(4, 11))):
        for s in seeds:
            for Lv in range(4):
                mem[f"y_pred_{kind}_s{s}_L{Lv}"] = pb + L7[f"{kind}{s}"][Lv]
    rows = pd.DataFrame({"MVT_ID_mvt": lab[F]["mvt_raw"], "day": pf["day"], "airport": pf["ap"], "H": H,
                         "y_true": y, "pbase": pb})
    for c in BAG_COMPS:
        X, Y, _, _ = COMPS[c]
        for k in KS:
            s = STEP[k]
            rows[f"r_{c}_{s}"] = y - pb - L7[X][k - 1]
            rows[f"dX_{c}_{s}"] = L7[X][k] - L7[X][k - 1]
            rows[f"dY_{c}_{s}"] = L7[Y][k] - L7[Y][k - 1]

    # ---- the census of every file this process opened, then every output -----------------------------------------
    cc = census_check()
    rec["census_check"] = {k: v_ for k, v_ in cc.items() if k != "files"} | {"files_opened": cc["files"]}
    print(f"census: {cc['opened']} repository files opened outside exp-057; {len(cc['missing'])} not in the census, "
          f"{len(cc['differ'])} differ", flush=True)
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
        rp = REC_DIR / f"draws-L{L}.csv"
        rec["receipts"][nm_(rp)] = put_text(rp, dr_csv[L])
    rec["oof"] = {}
    for f in C.FOLDS:
        p = OOFD / f"originfit-{f}.parquet"
        o = oofs[f]
        rec["oof"][nm_(p)] = {"sha256": put_parquet(p, o), "rows": int(len(o)), "columns": list(o.columns),
                              "ordered_sha256": {c: C.sha_arr(pan[f]["ids"], o[c].to_numpy("float64"))
                                                 for c in o.columns if c.startswith("y_pred")}}
    p = ART / "oof/members-july_loo.parquet"
    rec["oof"][nm_(p)] = {"sha256": put_parquet(p, mem), "rows": int(len(mem)), "columns": list(mem.columns)}
    p = ART / "receipts/rows-july_loo.parquet"
    rec["receipts"][nm_(p)] = put_parquet(p, rows)
    rec["seconds"] = round(time.time() - t0, 1)
    C.write_json(OUT, rec)

    # ---- log ----------------------------------------------------------------------------------------------------
    for c in COMPS:
        e = rec["points"][c]
        line = f"{c:12s} K={e['K']:<2d}"
        for s, q in e["steps"].items():
            line += f" | {s} m {q['m']:.4f} cos {q['cos']:.3f} J {q['J']:+8.2f}"
        print(line + f" | mean J {e['mean_J']['point']:+8.2f}", flush=True)
    print(f"rule S advances: {ruleS['advances']} {cond}", flush=True)
    print(f"rule Z advances: {ruleZ['advances']} {condZ}", flush=True)
    print(f"damage D0 K3 {dmg['K3']['L0']['D']:+.1f}; own value L0 K3 {dmg['own_value_L0']['K3']['value']:+.1f}, "
          f"K10 {dmg['own_value_L0']['K10']['value']:+.1f} s2/row", flush=True)
    print(f"written {nm_(OUT)} ({rec['seconds']} s){' -- STAND-IN RUN, DISCARD' if STANDIN else ''}", flush=True)


if __name__ == "__main__":
    main()
