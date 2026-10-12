"""Check matched annual sequence/control vectors for the two-weight route."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from screen import sha

ap = argparse.ArgumentParser()
for n in ["sequence-fit", "control-fit", "held-sequence", "held-control", "sequence-inputs", "control-inputs",
          "scope-pack", "read", "template", "out"]:
    ap.add_argument("--" + n, type=Path, required=True)
a = ap.parse_args()
assert not a.out.exists()
a.out.mkdir(parents=True)
read = json.loads(a.read.read_text())
assert read["gate_F_less_than"] == 2400.
assert read["comparison"] == "stack e9771b94 local"
step = read["paired_steps"]["grid_sequence"]
assert step["score"]["interval"]["lo95"] > 0 and step["G_minus_top5_point"] > 0
assert read["forms"]["paired_ls"]["score"]["G"] > 300
scope_rec = json.loads((a.scope_pack / "PACK.json").read_text())
scope_path = a.scope_pack / "Seq170-final.parquet"
assert sha(scope_path) == scope_rec["ranking_vector_sha256"]
assert sha(a.template) == scope_rec["template_sha256"]
template = pd.read_parquet(a.template)
ids = template.MVT_ID_mvt.to_numpy("int64")
scope = pd.read_parquet(scope_path)
assert len(ids) == 344841 and np.array_equal(ids, scope.MVT_ID_mvt.to_numpy("int64"))
out = scope[["MVT_ID_mvt", "member_scope", "clock", "month"]].copy()
records = {}
for name in ["sequence", "control"]:
    annual_path = getattr(a, name + "_fit")
    held_path = getattr(a, "held_" + name)
    inputs_path = getattr(a, name + "_inputs")
    annual = json.loads((annual_path / "FIT.json").read_text())
    held = json.loads((held_path / "FIT.json").read_text())
    assert annual["train_months"] == list(range(1, 13)) and annual["args"]["final"]
    assert held["train_months"] == [2, 3, 4, 5, 6, 8, 9, 10, 11, 12] and not held["args"]["final"]
    for k in ["mode", "loss", "epochs", "seed", "batch_size"]:
        assert annual["args"][k] == held["args"][k]
    assert annual["inputs_manifest_sha256"] == held["inputs_manifest_sha256"] == sha(inputs_path / "MANIFEST.json")
    code = "fit_runway.py" if annual["args"]["mode"] == "sequence-runway" else "fit.py"
    assert annual["code_sha256"] == held["code_sha256"] == sha(Path(__file__).with_name(code))
    assert sha(held_path / "FIT.json") == read["fits"][name]["fit_sha256"]
    assert sha(held_path / "jjrefit.parquet") == held["files"]["jjrefit.parquet"] == read["fits"][name]["vector_sha256"]
    assert sha(annual_path / "ranking2026.parquet") == annual["files"]["ranking2026.parquet"]
    vec = pd.read_parquet(annual_path / "ranking2026.parquet").set_index("MVT_ID_mvt")
    assert not vec.index.duplicated().any() and len(vec) == len(ids) and set(vec.index) == set(ids)
    prediction = vec.pred_seq170.reindex(ids).to_numpy(float)
    assert np.isfinite(prediction).all() and prediction.min() >= 60.
    out["pred_seq170" if name == "sequence" else "pred_row170"] = prediction
    records[name] = {"annual_fit_path": str(annual_path), "annual_fit_sha256": sha(annual_path / "FIT.json"),
                     "held_fit_path": str(held_path), "held_fit_sha256": sha(held_path / "FIT.json"),
                     "held_vector_sha256": sha(held_path / "jjrefit.parquet"),
                     "annual_raw_vector_sha256": sha(annual_path / "ranking2026.parquet"),
                     "input_manifest_sha256": annual["inputs_manifest_sha256"],
                     "train_rows": annual["train_rows"], "train_months": annual["train_months"]}
out["control_scope"] = out.member_scope & ~out.clock
out.to_parquet(a.out / "members-final.parquet", index=False)
manifest = {"member": "Seq170+Row170", "ranking_rows": len(out), "scope_rows": int(out.member_scope.sum()),
            "control_scope_rows": int(out.control_scope.sum()), "clock_scope_rows": int((out.member_scope & out.clock).sum()),
            "vector_sha256": sha(a.out / "members-final.parquet"), "records": records,
            "read_sha256": sha(a.read), "scope_pack_sha256": sha(a.scope_pack / "PACK.json"),
            "scope_rule": scope_rec["scope_rule"], "F_2026_sha256": scope_rec["F_2026_sha256"],
            "weights": read["forms"]["paired_ls"]["weights"],
            "serve_rule": "on ordinary scope: pre-base + w_sequence*(sequence-pre-base) + w_control*(control-pre-base); on clock scope: pre-base + w_sequence*(sequence-pre-base); comparator elsewhere; then practice and C33",
            "weights_rule": "other-month weights; negative control weight is part of the form",
            "encoder_note": "fixed ten-month train-only encoders; both models fitted on twelve months",
            "ranking_form": "annual final-retrain vectors; not a ten-month fold-form entry",
            "development_note": "gate and variants were chosen on these folds; intervals do not give independent confirmation",
            "template_sha256": sha(a.template), "code_sha256": sha(__file__)}
(a.out / "PACK.json").write_text(json.dumps(manifest, indent=1) + "\n")
print(json.dumps(manifest), flush=True)
