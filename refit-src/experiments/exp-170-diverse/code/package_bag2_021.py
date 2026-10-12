"""Annual pack of an n-seed sequence bag and an m-seed row-control bag (seqc unit 021).

The form of `package_bag_n.py`, with the same parent checks on the row control. Write-once output.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from screen import sha

HERE = Path(__file__).parent
RECIPE = {"sequence": {"kind": "equal n-seed prediction mean", "mean_code": "average_seeds_n.py", "fit_code": "fit_runway.py"},
          "control": {"kind": "equal n-seed row-control prediction mean", "mean_code": "average_rows_021.py", "fit_code": "fit.py"}}

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
records, seeds = {}, {}
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
    assert annual["kind"] == held["kind"] == RECIPE[name]["kind"]
    assert annual["code_sha256"] == held["code_sha256"] == sha(HERE / RECIPE[name]["mean_code"])
    seeds[name] = [p["seed"] for p in annual["parents"]]
    n = len(seeds[name])
    assert seeds[name] == [p["seed"] for p in held["parents"]] == list(range(17001, 17001 + n)) and n >= 2
    assert annual["n_seeds"] == held["n_seeds"] == n
    for annual_parent, held_parent in zip(annual["parents"], held["parents"]):
        for parent in [annual_parent, held_parent]:
            parent_path = Path(parent["path"])
            assert sha(parent_path / "FIT.json") == parent["fit_sha256"]
            assert json.loads((parent_path / "FIT.json").read_text()) == parent["fit"]
            assert sha(parent_path / "model.pt") == parent["model_sha256"]
            domain = "ranking2026" if parent["fit"]["args"]["final"] else "jjrefit"
            assert sha(parent_path / (domain + ".parquet")) == parent["vector_sha256"] == parent["fit"]["files"][domain + ".parquet"]
            assert parent["weight"] == 1 / n
            assert parent["fit"]["code_sha256"] == sha(HERE / RECIPE[name]["fit_code"])
            assert parent["fit"]["inputs_manifest_sha256"] == annual["inputs_manifest_sha256"]
        assert annual_parent["fit"]["args"]["final"] and not held_parent["fit"]["args"]["final"]
        for key in ["mode", "loss", "epochs", "seed", "batch_size"]:
            assert annual_parent["fit"]["args"][key] == held_parent["fit"]["args"][key]
        assert annual_parent["fit"]["train_months"] == list(range(1, 13))
        assert held_parent["fit"]["train_months"] == [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    assert sha(held_path / "FIT.json") == read["fits"][name]["fit_sha256"]
    assert sha(held_path / "jjrefit.parquet") == held["files"]["jjrefit.parquet"] == read["fits"][name]["vector_sha256"]
    assert sha(annual_path / "ranking2026.parquet") == annual["files"]["ranking2026.parquet"]
    vec = pd.read_parquet(annual_path / "ranking2026.parquet").set_index("MVT_ID_mvt")
    assert not vec.index.duplicated().any() and len(vec) == len(ids) and set(vec.index) == set(ids)
    parent_values = [pd.read_parquet(Path(p["path"]) / "ranking2026.parquet").set_index("MVT_ID_mvt").pred_seq170.reindex(ids).to_numpy(float)
                     for p in annual["parents"]]
    prediction = vec.pred_seq170.reindex(ids).to_numpy(float)
    assert np.array_equal(prediction, np.mean(parent_values, axis=0))
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
            "bag_seeds": seeds["sequence"], "bag_weights": [1 / len(seeds["sequence"])] * len(seeds["sequence"]),
            "control_bag_seeds": seeds["control"], "control_bag_weights": [1 / len(seeds["control"])] * len(seeds["control"]),
            "read_sha256": sha(a.read), "scope_pack_sha256": sha(a.scope_pack / "PACK.json"),
            "scope_rule": scope_rec["scope_rule"], "F_2026_sha256": scope_rec["F_2026_sha256"],
            "weights": read["forms"]["paired_ls"]["weights"],
            "serve_rule": "on ordinary scope: max(60, pre-base + w_sequence*(sequence-pre-base) + w_control*(control-pre-base)); on clock scope: max(60, pre-base + w_sequence*(sequence-pre-base)); comparator elsewhere; then practice and C33",
            "weights_rule": "other-month weights; negative control weight is part of the form",
            "encoder_note": "fixed ten-month train-only encoders; all models fitted on twelve months",
            "ranking_form": "equal n-seed annual final-retrain sequence plus equal m-seed annual row control",
            "development_note": "gate and variants were chosen on these folds; intervals do not give independent confirmation",
            "template_sha256": sha(a.template), "code_sha256": sha(__file__)}
(a.out / "PACK.json").write_text(json.dumps(manifest, indent=1) + "\n")
print(json.dumps(manifest), flush=True)
