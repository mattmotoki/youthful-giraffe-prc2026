"""Save stack e9771b94's checked fold base and rule fields in this lane."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from screen import sha

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
assert not a.out.exists()
assert sha(a.bundle) == "73274b9bb2ef13acc69a5d999bb862903003d2e69f56852bf50abaeb79c7f28f"
sys.path.insert(0, str(Path(__file__).with_name("stack_u8")))
from assembly import compose
b = pd.read_parquet(a.bundle)
base = compose(b, route="R4mk", noplan="v3cal", practice="selected")
pre = compose(b, route="R4mk", noplan="v3cal", practice="none", c33=False)
out = b[["MVT_ID_mvt", "lo", "hi", "noplan", "practice", "practice_weight", "practice_point"]].copy()
out["pred_base"], out["pred_pre"] = base, pre
assert out.MVT_ID_mvt.is_unique and len(out) == 344419 and np.isfinite(base).all()
out.to_parquet(a.out, index=False)
rec = {"source": "stack e9771b94; LEVERS C53-C54, 2735b71e",
       "bundle_sha256": sha(a.bundle), "base_sha256": sha(a.out),
       "code_sha256": {str(p): sha(p) for p in [Path(__file__), Path(__file__).with_name("stack_u8") / "assembly.py", Path(__file__).with_name("stack_u8") / "diverse_blend.py"]},
       "rule": "R4mk -> HB1-x -> no-plan v3cal -> selected practice -> C33"}
a.out.with_suffix(".json").write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps(rec), flush=True)
