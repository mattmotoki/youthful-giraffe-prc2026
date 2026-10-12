"""exp-186: 2026 predictions of saved exp-170 twelve-month fits on the inputs of rank_inputs_186.py.

Each fit folder holds FIT.json and model.pt of fit.py, fit_runway.py or fit_runway_fast_029.py (chosen by the
fit's code hash). The entry loads the model, builds the fit's windows over the 2026 rows (same halo, core and
batch size as the fit's own ranking2026 step) and writes one vector per fit and the equal mean of the fits
(the rule of average_rows_021.py and average_mixed_033.py). No label is read; no weight changes.

    PRC_RANKING_MODE=final python code/predict_rank_186.py --inputs <rank_inputs_186 output> \
        --fits <fit>... --out <dir> [--reference <saved member vector>] [--device cuda]

Each fit's vector is compared with the fit's own ranking2026.parquet on the shared IDs, and the mean with
--reference: in the ranking mode the largest difference must stay within --tol (default 1e-3 s, the replay
tolerance of exp-164 apply.py); in the final mode the difference is a record (the final file moves some
Jan/Jul inputs). Inputs with view_equal.npy (rank_inputs_186.py --no-clock in the ranking mode) hold rows
whose inputs differ from the view; there the strict compare covers only the rows whose prediction reads
view-equal inputs alone (row mode: the row; sequence modes: every row of the row's window), and the other
rows are a record. Every fit is recorded in PREDICT.json before a ranking-mode stop. --part writes the mean in
the template order of prc.data.load_submitting() (MVT_ID_mvt, pred_seq170 float64), the part contract of the
final file.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2])]  # this folder's modules first, then the repo root's prc
import prc.data as D  # noqa: E402
from prepare import sha  # noqa: E402

if not hasattr(D, "rank_months"):
    raise SystemExit("prc/data.py has no ranking modes (rank_months); update this checkout to exp-186 d351a474c or later")

ID = "MVT_ID_mvt"
CODES = ["fit.py", "fit_runway.py", "fit_runway_fast_029.py"]
TOL = 1e-3


def window_rule(module, checkpoint, fit):
    """Halo and core of the fit's own scoring windows."""
    if module == "fit":
        mode = checkpoint["mode"]
        return (48 if mode == "sequence-long" else (0 if mode == "row" else 16)), 32
    assert fit["window"] == {"group": "airport-day-runway", "halo": 4, "core": 4}
    return 4, 4


def predict(path, inputs, meta, num, cat, anchor, device, equal=None):
    fit = json.loads((path / "FIT.json").read_text())
    assert fit["args"]["final"], "a twelve-month fit (--final) serves the 2026 rows"
    assert sha(path / "model.pt") == fit["files"]["model.pt"]
    assert fit["inputs_manifest_sha256"] == inputs["view_manifest_sha256"], \
        "the inputs come from another view than the fit's"
    codes = {sha(HERE / n): n[:-3] for n in CODES}
    if fit["code_sha256"] not in codes:
        raise SystemExit(f"{path}: fit code {fit['code_sha256'][:8]} is not one of {CODES} in this folder")
    name = codes[fit["code_sha256"]]
    m = importlib.import_module(name)
    checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    sizes = [len(inputs["vocab"][c]) + 1 for c in inputs["category"]]
    assert checkpoint["numeric"] == num.shape[1] and checkpoint["vocab_sizes"] == sizes
    if name == "fit_runway_fast_029":
        model = m.Model(checkpoint["numeric"], checkpoint["vocab_sizes"], checkpoint["mode"],
                        checkpoint["width"], checkpoint["layers"])
    else:
        model = m.Model(checkpoint["numeric"], checkpoint["vocab_sizes"], checkpoint["mode"])
    model.load_state_dict(checkpoint["state"])
    model = model.to(device).eval()
    halo, core = window_rule(name, checkpoint, fit)
    chunks = m.windows(meta, np.ones(len(meta), dtype=bool), halo, core)
    size = fit["args"]["batch_size"]
    prediction = np.full(len(meta), np.nan, dtype="float64")
    if name == "fit_runway_fast_029":
        tensors = (torch.from_numpy(np.ascontiguousarray(num)).to(device),
                   torch.from_numpy(np.asarray(cat).astype("int64")).to(device),
                   torch.from_numpy(np.ascontiguousarray(anchor)).to(device))
        chunks_t = torch.tensor(chunks, dtype=torch.long, device=device)
    with torch.no_grad():
        for j in range(0, len(chunks), size):
            cs = chunks[j:j + size]
            if name == "fit_runway_fast_029":
                n, c, an, _, _, lengths = m.batch(chunks_t[j:j + size], *tensors, None, device)
            else:
                n, c, an, _, _, lengths = m.batch(cs, num, cat, anchor, None, device)
            pred = (an + model(n, c, lengths)).clamp(min=60.).cpu().numpy()
            for k, (lo, hi, central, end) in enumerate(cs):
                assert np.isnan(prediction[central:end]).all()
                prediction[central:end] = pred[k, central - lo:end - lo]
    assert np.isfinite(prediction).all()
    clean = None
    if equal is not None:  # rows whose prediction reads only view-equal inputs
        if checkpoint["mode"] == "row":
            clean = equal.copy()
        else:
            clean = np.zeros(len(meta), dtype=bool)
            for lo, hi, central, end in chunks:
                clean[central:end] = equal[lo:hi].all()
    return fit, name, prediction, {"halo": halo, "core": core, "batch_size": size, "windows": len(chunks)}, clean


def difference(new, old_path, ids, strict, what, tol=TOL, clean=None):
    old = pd.read_parquet(old_path)
    assert old[ID].is_unique
    s = old.set_index(ID).pred_seq170
    shared = np.isin(ids, s.index.to_numpy())
    d = new[shared] - s.reindex(ids[shared]).to_numpy(float)
    rec = {"shared_rows": int(shared.sum()), "reference_rows": int(len(s)),
           "max_abs_s": float(np.abs(d).max()) if len(d) else None,
           "rms_s": float(np.sqrt(np.mean(d ** 2))) if len(d) else None}
    if clean is not None:
        c = clean[shared]
        rec["view_equal_rows"] = int(c.sum())
        rec["view_equal_max_abs_s"] = float(np.abs(d[c]).max()) if c.any() else None
        rec["other_rows"] = int((~c).sum())
        rec["other_max_abs_s"] = float(np.abs(d[~c]).max()) if (~c).any() else None
        rec["other_rms_s"] = float(np.sqrt(np.mean(d[~c] ** 2))) if (~c).any() else None
    if strict:  # recorded here; main() stops after PREDICT.json is written
        worst = rec["max_abs_s"] if clean is None else rec["view_equal_max_abs_s"]
        if not shared.all() or len(s) != len(ids):
            rec["strict_failure"] = what + ": rows differ from the reference"
        elif worst is None or worst > tol:
            rec["strict_failure"] = f"{what}: replay difference {worst} s"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--fits", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", type=Path, help="saved member vector (MVT_ID_mvt, pred_seq170) for the mean")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tol", type=float, default=TOL, help="ranking mode: largest replay difference in s")
    ap.add_argument("--part", type=Path, help="write the mean as a part: MVT_ID_mvt, pred_seq170 (float64), one row per"
                    " template row of prc.data.load_submitting() in template order")
    a = ap.parse_args()
    if a.out.exists() or (a.part is not None and a.part.exists()):
        raise RuntimeError("output already exists")
    start = time.time()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(a.device)
    mode = D.ranking_mode()
    inputs = json.loads((a.inputs / "MANIFEST.json").read_text())
    assert inputs["ranking_mode"] == mode and not inputs["dry_run"]
    for name, digest in inputs["files"].items():
        assert sha(a.inputs / name) == digest, name
    meta = pd.read_parquet(a.inputs / "rows.parquet")
    num, cat, anchor = [np.load(a.inputs / (n + ".npy"), mmap_mode="r") for n in ["numeric", "category", "anchor"]]
    assert len(meta) == inputs["rows"] and (meta.year == 2026).all() and meta[ID].is_unique
    by_month = {f"2026-{m:02d}": int(n) for m, n in meta.groupby("month").size().items()}
    if inputs["ranking_file_is_mode_file"]:
        assert len(meta) == D.n_ranking() and by_month == D.ranking_spec().month_departures, by_month
    strict = mode == "ranking"
    equal = np.load(a.inputs / "view_equal.npy") if "view_equal.npy" in inputs["files"] else None
    if equal is not None:
        assert mode == "ranking" and equal.dtype == bool and equal.shape == (len(meta),)
    ids = meta[ID].to_numpy("int64")
    a.out.mkdir(parents=True)
    values, parents, cleans = [], [], []
    for path in a.fits:
        fit, code, prediction, window, clean = predict(path, inputs, meta, num, cat, anchor, device, equal)
        out = meta[[ID, "ADEP_mvt", "month", "date"]].copy()
        out["pred_seq170"] = prediction
        target = a.out / f"fit-{len(parents):02d}.parquet"
        out.to_parquet(target, index=False)
        saved = path / "ranking2026.parquet"
        assert sha(saved) == fit["files"]["ranking2026.parquet"]
        parents.append({"path": str(path.resolve()), "fit_sha256": sha(path / "FIT.json"),
                        "model_sha256": sha(path / "model.pt"), "fit_code": code + ".py",
                        "seed": fit["args"]["seed"], "window": window, "vector": target.name,
                        "against_saved_ranking2026": difference(prediction, saved, ids, strict, str(path),
                                                                a.tol, clean)})
        values.append(prediction)
        cleans.append(clean)
        print(json.dumps(parents[-1]), flush=True)
    assert len({p["fit_sha256"] for p in parents}) == len(parents)
    mean = meta[[ID, "ADEP_mvt", "month", "date"]].copy()
    mean["pred_seq170"] = np.mean(values, axis=0)
    mean.to_parquet(a.out / "ranking2026.parquet", index=False)
    part = None
    failures = [p["against_saved_ranking2026"]["strict_failure"] for p in parents
                if "strict_failure" in p["against_saved_ranking2026"]]
    if a.part is not None and not failures:
        template = D.load_submitting()[ID].to_numpy("int64")
        assert len(template) == D.n_ranking() and len(template) == len(ids) and np.isin(template, ids).all()
        value = pd.Series(mean.pred_seq170.to_numpy("float64"), index=ids).reindex(template).to_numpy("float64")
        assert np.isfinite(value).all()
        a.part.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({ID: template, "pred_seq170": value}).to_parquet(a.part, index=False)
        part = {"path": str(a.part), "sha256": sha(a.part), "rows": int(len(template)), "columns": [ID, "pred_seq170"]}
    reference = None
    if a.reference is not None:
        reference = {"path": str(a.reference), "sha256": sha(a.reference),
                     **difference(mean.pred_seq170.to_numpy(float), a.reference, ids, strict, "mean", a.tol,
                                  None if equal is None else np.logical_and.reduce(cleans))}
    rec = {"what": "2026 vectors of saved exp-170 fits and their equal mean (exp-186 final mode)",
           "ranking_mode": mode, "inputs": str(a.inputs.resolve()), "inputs_manifest_sha256": sha(a.inputs / "MANIFEST.json"),
           "view_manifest_sha256": inputs["view_manifest_sha256"], "rows": len(meta), "rows_by_month": by_month,
           "n_fits": len(parents), "parents": parents, "mean_against_reference": reference, "tol_s": a.tol,
           "part": part,
           "view_equal_rows": None if equal is None else int(equal.sum()),
           "device": a.device, "threads": a.threads, "torch": torch.__version__, "label_free": True,
           "code_sha256": sha(__file__), "seconds": time.time() - start,
           "files": {p.name: sha(p) for p in sorted(a.out.iterdir())}}
    (a.out / "PREDICT.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ["rows", "rows_by_month", "n_fits", "mean_against_reference"]}), flush=True)
    failures += [reference["strict_failure"]] if reference and "strict_failure" in reference else []
    if failures:
        raise SystemExit("ranking mode: " + "; ".join(failures))


if __name__ == "__main__":
    main()
