"""C53 inference-only mask; no labels are passed to the saved model."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fit import Model, batch, sha, windows

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", type=Path, required=True)
ap.add_argument("--fit", type=Path, required=True)
ap.add_argument("--airport", choices=["LIRF", "EDDF"], required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
assert not a.out.exists()
a.out.mkdir(parents=True)
start = time.time()
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
torch.cuda.set_per_process_memory_fraction(min(1., 10 * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory), 0)
fit = json.loads((a.fit / "FIT.json").read_text())
assert not fit["args"]["final"]
assert sha(a.inputs / "MANIFEST.json") == fit["inputs_manifest_sha256"]
assert sha(a.fit / "model.pt") == fit["files"]["model.pt"]
rec = json.loads((a.inputs / "MANIFEST.json").read_text())
for name, digest in rec["files"].items():
    assert sha(a.inputs / name) == digest
meta = pd.read_parquet(a.inputs / "rows.parquet")
num, cat, anchor = [np.load(a.inputs / (n + ".npy"), mmap_mode="r") for n in ["numeric", "category", "anchor"]]
names = rec["numeric"]
masked_names = [n for n in names if n.startswith(("obs_", "trk_", "clock_", "opdi_")) or n == "nm_minus_obs"]
assert len(masked_names) == 38
cols = [names.index(n) for n in masked_names]
cols += [i + len(names) for i in cols]
mask = (meta.ADEP_mvt == a.airport).to_numpy() & (meta.year == 2025).to_numpy() & (meta.month == 7).to_numpy()
held = (meta.year == 2025).to_numpy() & meta.month.isin([1, 7]).to_numpy()
checkpoint = torch.load(a.fit / "model.pt", map_location="cpu", weights_only=False)
model = Model(checkpoint["numeric"], checkpoint["vocab_sizes"], checkpoint["mode"]).cuda()
model.load_state_dict(checkpoint["state"])
model.eval()
halo = 48 if checkpoint["mode"] == "sequence-long" else (0 if checkpoint["mode"] == "row" else 16)
if checkpoint["mode"] == "sequence-runway":
    from fit_runway import windows as runway_windows
    chunks = runway_windows(meta, held, 4)
else:
    chunks = windows(meta, held, halo)
prediction = np.full(len(meta), np.nan)
with torch.no_grad():
    for j in range(0, len(chunks), fit["args"]["batch_size"]):
        cs = chunks[j:j + fit["args"]["batch_size"]]
        n, c, an, _, _, lengths = batch(cs, num, cat, anchor, None, "cuda")
        missing = np.zeros(n.shape[:2], dtype=bool)
        for k, (lo, hi, _, _) in enumerate(cs):
            missing[k, :hi - lo] = mask[lo:hi]
        rr, cc = np.nonzero(missing)
        if len(rr):
            n[torch.as_tensor(rr, device="cuda")[:, None], torch.as_tensor(cc, device="cuda")[:, None],
              torch.as_tensor(cols, device="cuda")[None, :]] = 0.
        p = (an + model(n, c, lengths)).clamp(min=60.).cpu().numpy()
        for k, (lo, hi, central, end) in enumerate(cs):
            assert np.isnan(prediction[central:end]).all()
            prediction[central:end] = p[k, central - lo:end - lo]
assert np.isfinite(prediction[held]).all()
out = meta.loc[held, ["MVT_ID_mvt", "ADEP_mvt", "month", "date"]].copy()
out["pred_seq170"] = prediction[held]
out.to_parquet(a.out / "jjrefit.parquet", index=False)
old = pd.read_parquet(a.fit / "jjrefit.parquet").set_index("MVT_ID_mvt").pred_seq170.reindex(out.MVT_ID_mvt).to_numpy(float)
unaffected = ~mask[held]
assert np.array_equal(old[unaffected], prediction[held][unaffected])
receipt = {"fit_sha256": sha(a.fit / "FIT.json"), "input_manifest_sha256": sha(a.inputs / "MANIFEST.json"),
           "model_sha256": sha(a.fit / "model.pt"), "code_sha256": sha(__file__),
           "masked_airport": a.airport, "masked_month": 7, "masked_rows": int(mask.sum()),
           "masked_numeric_columns": masked_names, "missing_rule": "normalized value zero and availability indicator zero",
           "label_free": True, "unaffected_predictions_exact": True,
           "seconds": time.time() - start, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
           "files": {p.name: sha(p) for p in a.out.iterdir()}}
(a.out / "PREDICT.json").write_text(json.dumps(receipt, indent=1) + "\n")
print(json.dumps(receipt), flush=True)
