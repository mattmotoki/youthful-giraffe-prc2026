"""The verification a served candidate here must pass, as a command.

Both files this experiment serves get the same checks, so the evidence for the
MAYBE reads exactly like the evidence for the candidate:

  * the submitting template's ID set, in the template's order;
  * every value finite, none below `prc.blend.FLOOR`;
  * the 388 `noac` rows and the 38,182 EHAM rows **bit identical** to v7 and to
    stack-5 -- these are the rows score-feedback rule and exp-030 arm 3 hold, and "bit
    identical" means a maximum absolute difference of exactly 0;
  * no row outside the served set moved at all;
  * a rebuild from the same record differing by exactly 0.0, which is the check
    that the serve is deterministic and the record complete.

The rebuild is not run here -- it is a second `serve.py` call, and its file is
passed in with `--rebuild` when there is one.

    python verify_served.py --name exp-027-k2-refitbulk [--rebuild <path>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-026-rebuild/code"))
import regime_table as RT  # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

DEST = ROOT / "submissions" / "candidates"
V7 = DEST / "exp-030-eham-out.parquet"
S5 = DEST / "exp-026-stack-5.parquet"


def col(p: Path) -> tuple[np.ndarray, np.ndarray]:
    d = pd.read_parquet(p)
    return d["MVT_ID_mvt"].to_numpy("float64"), d[D.TARGET].to_numpy("float64")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--rebuild", default=None)
    a = ap.parse_args()
    path = DEST / f"{a.name}.parquet"
    ids, v = col(path)
    _, b = col(V7)
    _, c = col(S5)
    tmpl = D.load_submitting()["MVT_ID_mvt"].to_numpy().astype("float64")
    g = RT.load().set_index("MVT_ID_mvt")
    reg = g["regime"].reindex(ids).to_numpy()
    apt = g["airport"].astype(str).reindex(ids).to_numpy()
    noac = reg == "noac"
    eham = apt == "EHAM"
    held = noac | eham

    checks, rec = [], {"name": a.name, "path": str(path.relative_to(ROOT)),
                       "sha256": D.sha256(path)}

    def chk(label, ok, got):
        checks.append((label, bool(ok), got))
        rec[label] = got

    chk("template_order", np.array_equal(ids, tmpl), int(len(ids)))
    chk("all_finite", np.isfinite(v).all(), int((~np.isfinite(v)).sum()))
    chk("none_below_floor", (v >= B.FLOOR).all(), int((v < B.FLOOR).sum()))
    chk("noac_identical_to_v7", np.abs(v[noac] - b[noac]).max() == 0.0,
        float(np.abs(v[noac] - b[noac]).max()))
    chk("noac_identical_to_stack5", np.abs(v[noac] - c[noac]).max() == 0.0,
        float(np.abs(v[noac] - c[noac]).max()))
    chk("eham_identical_to_v7", np.abs(v[eham] - b[eham]).max() == 0.0,
        float(np.abs(v[eham] - b[eham]).max()))
    chk("eham_identical_to_stack5", np.abs(v[eham] - c[eham]).max() == 0.0,
        float(np.abs(v[eham] - c[eham]).max()))
    chk("no_held_row_moved", int((v[held] != b[held]).sum()) == 0,
        int((v[held] != b[held]).sum()))
    if a.rebuild:
        _, r = col(Path(a.rebuild))
        chk("rebuild_max_abs_diff_zero", np.abs(v - r).max() == 0.0,
            float(np.abs(v - r).max()))
    rec.update({"n_noac": int(noac.sum()), "n_eham": int(eham.sum()),
                "n_served": int((~held).sum()),
                "n_moved_vs_v7": int((v != b).sum()),
                "rms_move_vs_v7": round(float(np.sqrt(((v - b) ** 2).mean())), 4),
                "min": float(v.min()), "max": float(v.max()),
                "mean": float(v.mean()),
                "n_above_3600": int((v > 3600).sum()),
                "v7_above_3600": int((b > 3600).sum())})
    for label, ok, got in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:28s} {got}")
    print(f"  ---- {rec['n_served']} served, {rec['n_moved_vs_v7']} moved against v7 at "
          f"{rec['rms_move_vs_v7']} s RMS; sha256 {rec['sha256']}")
    out = EXP / "analysis" / f"verify-{a.name}.json"
    rec["all_passed"] = all(ok for _, ok, _ in checks)
    out.write_text(json.dumps(rec, indent=2))
    print(f"  written: {out.relative_to(ROOT)}")
    if not rec["all_passed"]:
        raise SystemExit("verification FAILED")


if __name__ == "__main__":
    main()
