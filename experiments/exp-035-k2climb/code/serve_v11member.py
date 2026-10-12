"""tail rule: serve the climbed member in v11's own composition.

The desk, 13:50 UTC: the contract's advance clause is met on every fold at full
amplitude, the tail clause is met pooled at every airport and fails per fold only
on january by a mechanism §24 explains and which already sits inside january's
positive, clear interval, so the FULL form 2' ships and the clause failure is
recorded as the transfer risk.

The composition is v11's, with one thing changed -- the member:

    head    = the climbed member through exp-027's refit-bulk arm, served
              (`serve.py --arm-oof mod --rot refit-bulk-k2c-mod.json
                --arm k2-refit-bulk-eham-out`)
    masked  = head at exp-036's eight kept airports, v7 elsewhere
    served  = masked + 0.217 * (stack-5 - masked)

**The gate is the same arithmetic on K2's own head.**  Running these two lines on
`submissions/candidates/exp-027-k2-refitbulk.parquet` must return v10
(`ad962763...`) and then v11 (`42ca5ee6...`) at max abs difference 0, on the
bytes that were uploaded.  If it does not, this file refuses to write: the
composition would not be v11's.

Nothing is fitted here.  Four hashed columns, one airport list and one scalar.

    PRC_THREADS=3 python serve_v11member.py
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
NAME = "exp-035-k2climb-v11member"
S = 0.217
#: every input, pinned.  A changed byte anywhere here stops the run.
V7 = ("submissions/candidates/exp-030-eham-out.parquet",
      "67b70ee379e62a2a3bc9d91c9d771112baf3233efa3f74f4f95adbd7a3c03ef7")
S5 = ("submissions/candidates/exp-026-stack-5.parquet",
      "94f18e47a73265a6db66811ededfc3c935fb2a37fac04bf0cd34d45af708a2a1")
K2_RB = ("submissions/candidates/exp-027-k2-refitbulk.parquet",
         "3e6f3681025f60d7d8af9365c1c18c60d47df146c888c8b80351573211e2cf74")
V10 = ("submissions/uploaded/youthful-giraffe_v10.parquet",
       "ad962763236e9b40ace45a5dac12945568bbce441ed0547ff8e814d248c72c78")
V11 = ("submissions/candidates/exp-038-v10-stack5-blend.parquet",
       "42ca5ee6410938d56c8830e9ec3980d806633c3b2ff15963c0ffe2fe97c39f8c")
MOD_RB = ("submissions/candidates/exp-035-k2climb-mod-refitbulk.parquet", None)
GATES = ("experiments/exp-026-rebuild/candidates/ranking-gates.parquet",
         "7f14aeea2219eef62b918637b015268cafe15c905a2d4072b878f791e9e2dbfd")
MASK_JSON = "experiments/exp-036-mask/analysis/mask.json"
#: `analysis/price-vs-v11.json`, form 2' -- the fold-side update this serve must
#: be the same size as (exp-040's one-third rule).
FOLD_B = "form 2' (v11's recipe, member replaced)"


def column(rel: str, sha: str | None, tmpl: np.ndarray) -> np.ndarray:
    p = ROOT / rel
    got = D.sha256(p)
    if sha is not None and got != sha:
        raise SystemExit(f"{p.name}: sha256 {got[:8]} against the record's {sha[:8]}")
    d = pd.read_parquet(p)
    ids = d["MVT_ID_mvt"].to_numpy("float64")
    if np.array_equal(ids, tmpl):
        return d[D.TARGET].to_numpy("float64"), got
    v = d.set_index("MVT_ID_mvt")[D.TARGET].reindex(tmpl)
    if v.isna().any():
        raise SystemExit(f"{p.name}: {int(v.isna().sum())} template rows are missing")
    return v.to_numpy("float64"), got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=NAME)
    a = ap.parse_args()

    tmpl = D.load_submitting()["MVT_ID_mvt"].to_numpy().astype("float64")
    if D.sha256(ROOT / GATES[0]) != GATES[1]:
        raise SystemExit("the ranking gates table has changed")
    g = pd.read_parquet(ROOT / GATES[0]).set_index("MVT_ID_mvt")
    apt = g["airport"].astype(str).reindex(tmpl).to_numpy()
    keep = json.loads((ROOT / MASK_JSON).read_text())["keep"]
    on = np.isin(apt, keep)

    v7, _ = column(*V7, tmpl)
    s5, _ = column(*S5, tmpl)
    k2rb, _ = column(*K2_RB, tmpl)
    v10, _ = column(*V10, tmpl)
    v11, _ = column(*V11, tmpl)
    mod, mod_sha = column(*MOD_RB, tmpl)

    def compose(head: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = np.where(on, head, v7)
        return m, m + S * (s5 - m)

    # -- the gate: K2's own head through these two lines returns v10 and v11 ---
    m_k2, b_k2 = compose(k2rb)
    d10 = float(np.abs(m_k2 - v10).max())
    d11 = float(np.abs(b_k2 - v11).max())
    if d10 > 0.0 or d11 > 0.0:
        raise SystemExit(f"the composition is not v11's: it returns v10 at {d10:g} "
                         f"and v11 at {d11:g}, not 0")
    print(f"[gate] K2's own head through the same two lines returns v10 at {d10:g} "
          f"and v11 at {d11:g}", flush=True)

    masked, v = compose(mod)
    if not np.isfinite(v).all():
        raise SystemExit("the candidate is not finite")
    if v.min() < B.FLOOR:
        raise SystemExit(f"the candidate goes below the floor: {v.min()}")

    reg = g["regime"].astype(str).reindex(tmpl).to_numpy()
    noac, eham, eddf = reg == "noac", apt == "EHAM", apt == "EDDF"
    for label, m in (("noac", noac), ("EHAM", eham), ("EDDF", eddf)):
        off = float(np.abs(v[m] - v11[m]).max())
        if off > 0.0:
            raise SystemExit(f"the {label} rows are not v11's: max abs {off:g}")

    path = DEST / f"{a.name}.parquet"
    pd.DataFrame({"MVT_ID_mvt": tmpl, D.TARGET: v.astype("float64")}).to_parquet(
        path, index=False)

    fold = json.loads((EXP / "analysis" / "price-vs-v11.json").read_text())
    row = next(r for r in fold["rows"] if r["candidate"] == FOLD_B)
    fold_rms = {f: float(np.sqrt(row["folds"][f]["B"])) for f in row["folds"]}
    served_rms = float(np.sqrt(((v - v11) ** 2).mean()))
    ratio = served_rms / float(np.mean(list(fold_rms.values())))

    rec = {"name": a.name, "path": str(path.relative_to(ROOT)),
           "decision": "D20, desk 2026-09-08 13:50 UTC",
           "rule": "masked = where(airport in exp-036 keep, head, v7); "
                   "served = masked + s * (stack-5 - masked)",
           "s": S, "keep_airports": list(keep), "hold_at_v7": ["EDDF", "EHAM"],
           "head_file": MOD_RB[0], "head_sha256": mod_sha,
           "head": "the climbed member (phi,surface,deice) through exp-027's "
                   "refit-bulk arm; rotation analysis/refit-bulk-k2c-mod.json",
           "v7_file": V7[0], "v7_sha256": V7[1],
           "stack5_file": S5[0], "stack5_sha256": S5[1],
           "v11_file": V11[0], "v11_sha256": V11[1],
           "gates_table": GATES[0], "gates_table_sha256": GATES[1],
           "gate_k2_head_returns_v10": d10, "gate_k2_head_returns_v11": d11,
           "k2_head_file": K2_RB[0], "k2_head_sha256": K2_RB[1],
           "v10_file": V10[0], "v10_sha256": V10[1],
           "sha256": D.sha256(path), "n_rows": int(len(v)),
           "n_from_head": int(on.sum()), "n_from_v7": int((~on).sum()),
           "n_moved_vs_v11": int((v != v11).sum()),
           "rms_move_vs_v11": round(served_rms, 4),
           "max_abs_move_vs_v11": round(float(np.abs(v - v11).max()), 4),
           "fold_update_rms": {f: round(x, 4) for f, x in fold_rms.items()},
           "served_over_fold_rms_ratio": round(ratio, 4),
           "n_noac": int(noac.sum()), "n_eham": int(eham.sum()), "n_eddf": int(eddf.sum()),
           "min": float(v.min()), "median": float(np.median(v)),
           "p99": float(np.quantile(v, 0.99)), "max": float(v.max()),
           "measured_fold_side": {"gain_board_allrow": row["gain_board_allrow"],
                                  "ci_allrow_1day": row["ci_allrow_1day"],
                                  "gain_board_clean": row["gain_board_clean"],
                                  "ci_clean_1day": row["ci_clean_1day"],
                                  "per_fold": {f: {"gain_rmse": row["folds"][f]["gain_rmse"],
                                                   "ci": row["folds"][f]["ci_rmse"]}
                                               for f in row["folds"]},
                                  "tail_clause": row["tail_clause"]}}
    (EXP / "analysis" / f"served-{a.name}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: rec[k] for k in
                      ("sha256", "n_rows", "n_from_head", "n_from_v7", "n_moved_vs_v11",
                       "rms_move_vs_v11", "served_over_fold_rms_ratio", "min", "max")},
                     indent=1))
    print(f"  written: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
