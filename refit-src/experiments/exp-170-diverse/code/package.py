"""Check the twelve-month vector and publish a read-only member handoff."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--fit", type=Path, required=True)
ap.add_argument("--held-fit", type=Path, required=True)
ap.add_argument("--screen", type=Path, required=True)
ap.add_argument("--inputs", type=Path, required=True)
ap.add_argument("--template", type=Path, required=True)
ap.add_argument("--clock", type=Path, required=True)
ap.add_argument("--F", type=Path, required=True)
ap.add_argument("--validation-rows", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
assert not a.out.exists()
a.out.mkdir(parents=True)
fit = json.loads((a.fit / "FIT.json").read_text())
held = json.loads((a.held_fit / "FIT.json").read_text())
screen = json.loads(a.screen.read_text())
assert fit["train_months"] == list(range(1, 13)) and fit["args"]["final"]
assert held["train_months"] == [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
for k in ["mode", "loss", "epochs", "seed", "batch_size"]:
    assert fit["args"][k] == held["args"][k]
assert fit["inputs_manifest_sha256"] == held["inputs_manifest_sha256"]
assert fit["inputs_manifest_sha256"] == sha(a.inputs / "MANIFEST.json")
assert fit["code_sha256"] == held["code_sha256"] == sha(Path(__file__).with_name("fit.py"))
assert fit["files"]["ranking2026.parquet"] == sha(a.fit / "ranking2026.parquet")
assert held["files"]["jjrefit.parquet"] == sha(a.held_fit / "jjrefit.parquet")
assert screen["model_fit_sha256"] == sha(a.held_fit / "FIT.json")
assert screen["member_sha256"] == sha(a.held_fit / "jjrefit.parquet")
assert screen["input_manifest_sha256"] == sha(a.inputs / "MANIFEST.json")
assert screen["comparison"] == "HB1-x local"
assert screen["gate_F_less_than"] == 2400.
read = screen["blends"]["by_class"]["score"]
assert read["G"] > 300.
assert sum(x["contribution"] for x in read["top_airport_days"][:5]) < .5 * read["G"]
template = pd.read_parquet(a.template)
ids = template.MVT_ID_mvt.to_numpy("int64")
f = pd.read_parquet(a.fit / "ranking2026.parquet").set_index("MVT_ID_mvt")
assert len(ids) == 344841 and len(f) == len(ids) and set(f.index) == set(ids)
assert not f.index.duplicated().any()
prediction = f.pred_seq170.reindex(ids).to_numpy(float)
assert np.isfinite(prediction).all() and prediction.min() >= 60.
meta = pd.read_parquet(a.inputs / "rows.parquet").set_index("MVT_ID_mvt").loc[ids]
clock_ids = pd.read_parquet(a.clock, columns=["MVT_ID_mvt"]).MVT_ID_mvt.to_numpy("int64")
assert sha(a.F) == "7b29c560ab633254cb0879999aca959de983b6624f3f514f4c73840600a82db0"
anchor = pd.read_parquet(a.F).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(ids).to_numpy(float)
assert np.isfinite(anchor).all()
clock = np.isin(ids, clock_ids)
scope = (anchor < 2400.) & meta.aobt.to_numpy(bool) & meta.aircraft.to_numpy(bool)
validation = pd.read_parquet(a.validation_rows)
validation = validation[validation.month.isin([1, 7])]
counts = validation.groupby(["month", "airport", "cls"]).size()
cls = np.where(~meta.aircraft.to_numpy(bool), "noac", np.where(clock, "clock", "ordinary"))
supported = np.array([counts.get((int(m), str(apt), str(c)), 0) >= 2
                      for m, apt, c in zip(meta.month, meta.ADEP_mvt, cls)])
unsupported_removed = int((scope & ~supported).sum())
scope &= supported
out = pd.DataFrame({"MVT_ID_mvt": ids, "pred_seq170": prediction,
                    "member_scope": scope, "clock": clock, "month": meta.month.to_numpy()})
out.to_parquet(a.out / "Seq170-final.parquet", index=False)
weights = screen["blends"]["by_class"]["weights"]
manifest = {"member": "Seq170", "column": "pred_seq170", "ranking_rows": len(out),
            "scope_rows": int(scope.sum()), "clock_scope_rows": int((scope & clock).sum()),
            "ordinary_scope_rows": int((scope & ~clock).sum()),
            "unsupported_rows_kept": unsupported_removed,
            "ranking_vector_sha256": sha(a.out / "Seq170-final.parquet"),
            "held_vector": str(a.held_fit / "jjrefit.parquet"), "held_vector_sha256": sha(a.held_fit / "jjrefit.parquet"),
            "annual_fit": str(a.fit), "annual_fit_sha256": sha(a.fit / "FIT.json"),
            "screen_sha256": sha(a.screen), "template_sha256": sha(a.template), "clock_frame_sha256": sha(a.clock),
            "scope_rule": "original F < 2400, aircraft and AOBT_3 present, supported frozen cell; comparator elsewhere",
            "F_2026_sha256": sha(a.F), "comparison": screen["comparison"],
            "weights": weights, "weights_note": "each month uses the weight fitted on the other 2025 month; no board weight",
            "model_raw_on_other_rows": "provided for diagnostics; do not serve those values",
            "encoder_note": "same ten-month train-only encoders; model trained on all twelve 2025 months",
            "package_code_sha256": sha(__file__), "validation_rows_sha256": sha(a.validation_rows),
            "files": {p.name: sha(p) for p in a.out.iterdir()}}
(a.out / "PACK.json").write_text(json.dumps(manifest, indent=1) + "\n")
print(json.dumps(manifest), flush=True)
