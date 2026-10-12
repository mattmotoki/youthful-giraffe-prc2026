"""exp-055: the frozen label-free vectors of SPEC section 3.3, rebuilt exactly as `pin.py`
rebuilt them, and verified against the ordered (ids, vector) SHA-256 values that `pin.py`
wrote into `analysis/pin.json`.  A mismatch is fatal.  No label is read.

Returned per fold (d-space, float64, panel row order):
    P{s}_L{L}  plain member s (raw npz)          C{s}_L{L}  June carrier member s (July only)
    BL_L{L}    stitched half-month carrier        S1_L{L}    sham, member 1 (July only)
    T10_L{L}   K = 10 fold head (v19..v22)        P3bag_L{L} / C3bag_L{L}  the K = 3 bags
and the panel masks (ids, H, ownership regime, NM regime, airport, day, pbase, usable, guard).
"""
from __future__ import annotations

import json

import numpy as np

import common as C
import withinsplit as WS                                              # noqa: E402  (exp-046)

PIN = C.ANALYSIS / "pin.json"


def load() -> tuple[dict, dict, dict]:
    C.check(C.E53 / "analysis/pin.json")
    pin = json.loads(PIN.read_text())
    qpin = json.loads((C.E53 / "analysis/pin.json").read_text())       # hash keys only
    osha = pin["ordered_sha256"]
    heads = {}
    for lv, p in C.HEADS.items():
        C.check(p)
        z = np.load(p)
        heads[lv] = {f: z[f].astype("float64") for f in C.FOLDS}
    pan, vec = {}, {}
    for f in C.FOLDS:
        p = C.load_panel(f)
        if "y_true" in p.columns or "clean" in p.columns:
            raise SystemExit("a label column was read")
        ids = p.MVT_ID_mvt.to_numpy("float64")
        reg = C.fold_regime(p)
        if C.sha_arr(ids) != qpin["folds"][f]["ids_sha256"]:
            raise SystemExit(f"{f}: ids differ from Q15's pin")
        if C.sha_arr(C.regime_index(reg)) != qpin["folds"][f]["regime_sha256"]:
            raise SystemExit(f"{f}: regime differs from Q15's pin")
        pan[f] = {"ids": ids, "H": reg == "H", "reg": reg, "nm": p.regime.astype(str).to_numpy(),
                  "ap": p.airport.astype(str).to_numpy(), "day": p.day.astype(str).to_numpy(),
                  "pbase": p.pbase.to_numpy("float64"), "usable": p.usable.to_numpy(bool),
                  "guard": p.guard_held.to_numpy(bool), "hv": WS.halves(p.day)}
        v = {}
        for L in range(4):
            v[f"T10_L{L}"] = heads[C.LEVELS10[L]][f]
            for s in C.SEEDS:
                v[f"P{s}_L{L}"] = C.npz_fold(C.member_path(L, s), f)
            A_ = C.npz_fold(C.bl_path(f, "A", L), f)
            B_ = C.npz_fold(C.bl_path(f, "B", L), f)
            v[f"BL_L{L}"] = np.where(pan[f]["hv"] == 1, A_, B_)
            v[f"P3bag_L{L}"] = C.bag({s: v[f"P{s}_L{L}"] for s in C.SEEDS}, C.SEEDS, pan[f]["pbase"],
                                     pan[f]["usable"], pan[f]["guard"])
            if f == "july_loo":
                for s in C.SEEDS:
                    v[f"C{s}_L{L}"] = C.npz_fold(C.carrier_path(L, s), f)
                v[f"S1_L{L}"] = C.npz_fold(C.sham_path(L), f)
                v[f"C3bag_L{L}"] = C.bag({s: v[f"C{s}_L{L}"] for s in C.SEEDS}, C.SEEDS, pan[f]["pbase"],
                                         pan[f]["usable"], pan[f]["guard"])
        for name, x in v.items():
            if name.startswith("T10"):
                lv = C.LEVELS10[int(name[-1])]
                if C.sha_arr(x) != qpin["folds"][f]["head_vector_sha256"][lv]:
                    raise SystemExit(f"{f} {name}: the K = 10 head is not Q15's pinned vector")
                continue
            key = f"vector_{f}_{name}"
            if key not in osha:
                raise SystemExit(f"{key}: no ordered pin in pin.json")
            if C.sha_arr(ids, x) != osha[key]:
                raise SystemExit(f"{key}: the rebuilt vector is not the one pin.py pinned")
        vec[f] = v
    return pan, vec, pin
