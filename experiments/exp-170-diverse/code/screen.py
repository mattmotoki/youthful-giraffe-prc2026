"""Error diversity and opposite-month blend weights on the real F refit form."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID = "MVT_ID_mvt"
PINS = {"refs.parquet": "a07552f7357aa33266a5fe0138b6733c986ebd177b045b60f1b1cbce0c91fd45",
        "R4Fjjrefit-month-2025-01.parquet": "0095135f56b7f465c38d707ad278e8559f27528fca0cb44d6323b33657a18708",
        "R4Fjjrefit-month-2025-07.parquet": "42dee7eb7a81fc48e4853369f697dfcfa7dc97d4180218980b0d2b9a0a5fd531"}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def moments(y, base, member, w):
    if len(y) < 2 or w.sum() == 0:
        return {"n": len(y), "rho": None, "mse_ratio": None}
    w = w / w.sum()
    a, b = base - y, member - y
    mse_a, mse_b = float(w @ (a * a)), float(w @ (b * b))
    aa, bb = a - w @ a, b - w @ b
    den = np.sqrt((w @ (aa * aa)) * (w @ (bb * bb)))
    return {"n": len(y), "rho": float(w @ (aa * bb) / den) if den > 0 else None,
            "mse_ratio": mse_b / mse_a, "mse_F": mse_a, "mse_member": mse_b,
            "error_cosine": float(w @ (a * b) / np.sqrt(mse_a * mse_b)),
            "bias_F": float(w @ a), "bias_member": float(w @ b)}


def cross_blend(y, base, member, month, w, scopes):
    out = base.copy()
    choices = {}
    for name, on in scopes.items():
        choices[name] = {}
        for fit_month, apply_month in [(1, 7), (7, 1)]:
            fit = on & (month == fit_month)
            app = on & (month == apply_month)
            weights = np.arange(51) * .02
            loss = [float(w[fit] @ ((y[fit] - np.maximum(60., base[fit] + v *
                                                          (member[fit] - base[fit]))) ** 2)) for v in weights]
            v = float(weights[int(np.argmin(loss))]) if fit.any() and w[fit].sum() > 0 else 0.
            out[app] = np.maximum(60., base[app] + v * (member[app] - base[app]))
            choices[name][str(apply_month)] = {"weight": v, "fit_month": fit_month,
                                              "fit_rows": int(fit.sum()), "apply_rows": int(app.sum())}
    return out, choices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--refs", type=Path, required=True)
    ap.add_argument("--fold", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gate", type=float, help="keep the comparator where the original F is at least this")
    ap.add_argument("--head", type=Path, help="development comparison: HB1 at weight 0.5 on clock rows")
    a = ap.parse_args()
    if a.out.exists():
        raise RuntimeError("score directory already exists")
    a.out.mkdir(parents=True)
    sys.path.insert(0, str(a.source_root))
    os.environ["PRC_VALIDATION_DIR"] = str(a.source_root / "experiments/exp-146-validation/frozen")
    from prc import gscore as GS
    rows = GS.load_rows()
    rows = rows[rows.month.isin([1, 7])].copy().reset_index(drop=True)
    ids = rows[ID].to_numpy("int64")
    assert len(ids) == 344419
    member = pd.read_parquet(a.fit / "jjrefit.parquet").set_index(ID)
    assert not member.index.duplicated().any() and set(member.index) == set(ids)
    p = member.pred_seq170.reindex(ids).to_numpy(float)
    assert np.isfinite(p).all() and p.min() >= 60.
    refs = a.refs
    assert sha(refs) == PINS[refs.name]
    v32 = pd.read_parquet(refs).set_index(ID).v32.reindex(ids).to_numpy(float)
    preds, sources = [], {str(refs): sha(refs)}
    for m in ["01", "07"]:
        path = a.fold / ("R4Fjjrefit-month-2025-" + m + ".parquet")
        assert sha(path) == PINS[path.name]
        f = pd.read_parquet(path)
        preds.append(pd.Series(f.pred_Sb.to_numpy(float), index=f[ID].to_numpy("int64")))
        sources[str(path)] = sha(path)
    r4 = pd.concat(preds)
    assert not r4.index.duplicated().any() and set(r4.index) == set(ids)
    keep = (rows.cls == "noac").to_numpy() | ((rows.cls == "ordinary").to_numpy() & ~rows.aobt.to_numpy(bool))
    base = .5 * np.where(keep, v32, r4.reindex(ids).to_numpy(float)) + .5 * v32
    assert np.isfinite(base).all() and base.min() >= 60.
    anchor_F = base.copy()
    if a.head:
        assert sha(a.head) == "073030a5f7a3c8f72b5f150921129dd0e3fb443f617c9b26f7088babe6b5cee1"
        head = pd.read_parquet(a.head).set_index(ID).pred_HB1.reindex(ids).to_numpy(float)
        clock = (rows.cls == "clock").to_numpy()
        assert np.isfinite(head[clock]).all()
        base[clock] = np.maximum(60., .5 * base[clock] + .5 * np.maximum(60., head[clock]))
        sources[str(a.head)] = sha(a.head)
    y = GS.labels(ids)
    meta = pd.read_parquet(a.inputs / "rows.parquet").set_index(ID).loc[ids]
    all_labels = np.load(a.inputs / "labels.npy", mmap_mode="r")
    original = pd.read_parquet(a.inputs / "rows.parquet")
    yy = pd.Series(all_labels, index=original[ID].to_numpy("int64")).reindex(ids).to_numpy(float)
    assert np.array_equal(y, yy)
    cells = GS.load_cells().set_index(["month", "airport", "cls"]).N
    count = rows.groupby(["month", "airport", "cls"]).size()
    keys = list(zip(rows.month, rows.airport.astype(str), rows.cls.astype(str)))
    w = np.array([cells.get(k, 0.) / GS.N_RANKING / count[k] if count[k] >= GS.MIN_SUPPORT else 0. for k in keys])
    masks = {"all": np.ones(len(ids), bool), "clock": (rows.cls == "clock").to_numpy(),
             "body": (rows.cls == "ordinary").to_numpy(), "no_AOBT_3": ~rows.aobt.to_numpy(bool),
             "LIRF_no_plan": (rows.airport == "LIRF").to_numpy() & ~meta.plan.to_numpy(bool)}
    if a.gate is not None:
        masks["body_gate"] = masks["body"] & (anchor_F < a.gate)
        masks["clock_gate"] = masks["clock"] & (anchor_F < a.gate)
    largest = np.argsort(-((anchor_F - y) ** 2), kind="stable")[:100]
    body_trim = masks["body"].copy()
    body_trim[largest] = False
    errors = {}
    member_keep = (rows.cls == "noac").to_numpy() | ~rows.aobt.to_numpy(bool)
    if a.gate is not None:
        member_keep |= anchor_F >= a.gate
    served = np.where(member_keep, base, p)
    assert np.array_equal(served[member_keep], base[member_keep])
    for form, pp in [("raw", p), ("keep_F", served)]:
        errors[form] = {}
        for name, mask in masks.items():
            for trim in [False, True]:
                on = mask.copy()
                if trim:
                    on[largest] = False
                k = name + ("_without_100" if trim else "")
                errors[form][k] = {"row": moments(y[on], base[on], pp[on], np.ones(on.sum())),
                                   "ranking": moments(y[on], base[on], pp[on], w[on])}
    blends = {}
    for name, scopes in [("global", {"all": masks["all"]}),
                         ("by_class", {c: (rows.cls == c).to_numpy() for c in ["clock", "ordinary", "noac"]}),
                         ("body_only", {"body": masks["body"]}), ("clock_only", {"clock": masks["clock"]})]:
        cand, weights = cross_blend(y, base, served, rows.month.to_numpy(), w, scopes)
        score = GS.score(ids, base, cand, y=y, require_complete=True)
        independent = float(w @ ((y - base) ** 2 - (y - cand) ** 2))
        assert abs(independent - score["G"]) < 1e-7
        blends[name] = {"weights": weights, "score": score}
        pd.DataFrame({ID: ids, "pred_seq170": cand}).to_parquet(a.out / (name + ".parquet"), index=False)
    body = errors["raw"]["body"]["ranking"]
    clock = errors["raw"]["clock"]["ranking"]
    best_g = max(v["score"]["G"] for v in blends.values())
    correlation_pass = any(z["rho"] is not None and z["rho"] < .85 and z["mse_ratio"] <= 1.2 for z in [body, clock])
    result = {"error_views": errors, "blends": blends, "source_pins": sources,
              "model_fit_sha256": sha(a.fit / "FIT.json"), "member_sha256": sha(a.fit / "jjrefit.parquet"),
              "input_manifest_sha256": sha(a.inputs / "MANIFEST.json"), "score_code_sha256": sha(__file__),
              "largest_row_definition": "100 largest squared F errors; diagnostic only",
              "top100_ids": ids[largest].tolist(), "main_pass": bool(correlation_pass and best_g > 800.),
              "needs_repair": bool(best_g < 300. or (body["rho"] is not None and body["rho"] > .90 and body["mse_ratio"] >= 1.2)),
              "best_blend_G": best_g, "member_keep_rows": int(member_keep.sum()),
              "member_keep_rule": "comparator on noac or any row without AOBT_3; gate uses original F",
              "gate_F_less_than": a.gate, "comparison": "HB1-x local" if a.head else "F",
              "reviewer": "pending"}
    (a.out / "SCREEN.json").write_text(json.dumps(result, indent=1) + "\n")
    print("body", body, "clock", clock, flush=True)
    print("trimmed_body", errors["raw"]["body_without_100"]["ranking"],
          "trimmed_clock", errors["raw"]["clock_without_100"]["ranking"], flush=True)
    print("G", {k: v["score"]["G"] for k, v in blends.items()}, flush=True)
    print("main_pass", result["main_pass"], "needs_repair", result["needs_repair"], flush=True)


if __name__ == "__main__":
    main()
