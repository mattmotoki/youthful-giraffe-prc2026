"""Serve a masked candidate: one source file, one base file, an airport list.

    .venv/bin/python experiments/exp-036-mask/serve.py --arm refitbulk-mask

**No model runs here.**  The arm is `np.where(airport in keep, source, base)` over
two verified candidate files, so the served side is the fold side's own statement
and the rebuild that verifies it shares no model code with it — the record names
two parquet files, their SHA-256 values, the gates table's SHA-256, and a list of
airports, and nothing else.

The serving contract is taken from the **arm's own measured record**
(`analysis/arm-<arm>.json`, written by `code/mask.py`), not from the command line:
`--arm` names a record, and the record's `keep` list and source are what this
script serves.  A command therefore cannot attach an arm's score and gate to a
different mask or a different source file (exp-030 Sol round 1, blocker 1).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-026-rebuild/code"))
import regime_table as RT  # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

DEST = ROOT / "submissions" / "candidates"
#: The two files every arm here is made of, pinned by SHA-256.  `v7` is the base:
#: exp-030 arm 3, the file the board read at 263.4623.  The sources are the two
#: exp-027 candidates, K2 (v9, board 263.0827) and the refit-bulk MAYBE.
FILES = {
    "v7": ("submissions/candidates/exp-030-eham-out.parquet",
           "67b70ee379e62a2a3bc9d91c9d771112baf3233efa3f74f4f95adbd7a3c03ef7"),
    "K2": ("submissions/candidates/exp-027-k2.parquet",
           "a3a139dbe8255d5dfc17aa69cfd4c65a15df26c2e76feb9eeaf4dbad03ed6ba5"),
    "refit-bulk": ("submissions/candidates/exp-027-k2-refitbulk.parquet",
                   "3e6f3681025f60d7d8af9365c1c18c60d47df146c888c8b80351573211e2cf74"),
}


def checked(key: str) -> np.ndarray:
    """One source file's column, in the template's order, gated on its hash."""
    rel, want = FILES[key]
    p = ROOT / rel
    got = D.sha256(p)
    if got != want:
        raise SystemExit(f"{rel} has changed: {got[:8]} against {want[:8]}")
    d = pd.read_parquet(p)
    tmpl = D.load_submitting()["MVT_ID_mvt"].to_numpy().astype("float64")
    if not np.array_equal(d["MVT_ID_mvt"].to_numpy("float64"), tmpl):
        raise SystemExit(f"{rel} is not in the template's order")
    return d[D.TARGET].to_numpy("float64")


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="the measured arm to serve")
    ap.add_argument("--name", default=None, help="the candidate's file name")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    recp = HERE / "analysis" / f"arm-{a.arm}.json"
    if not recp.exists():
        raise SystemExit(f"{recp.relative_to(ROOT)} is absent; run code/mask.py first")
    m = json.loads(recp.read_text())
    src, keep = m.get("source"), m.get("keep")
    # Sol round 1, blocker 1.  The first version did `list(m["keep"])`, and the
    # control arm's record stores the STRING "all ten"; `list()` of a string is
    # its characters, so `--arm refitbulk-nomask --name exp-036-refitbulk-mask`
    # selected 0 source rows, served v7 on all 344,841, and still carried the
    # control's clean board and gain into the record -- and it would have written
    # that over the shipping filename before any verification ran.  The mask is
    # now type-checked and checked against the gates table's own airport
    # universe, and the file name is bound to the arm, so a name cannot be
    # detached from the bytes it was measured on.
    if src not in FILES:
        raise SystemExit(f"arm {a.arm!r} names no servable source: {src!r}. "
                         f"A derived arm's record names one of {sorted(FILES)}")
    if not isinstance(keep, list) or not keep or not all(isinstance(x, str) for x in keep):
        raise SystemExit(f"arm {a.arm!r} has no servable mask: keep = {keep!r}. "
                         f"A servable arm's `keep` is a non-empty list of airport codes")
    if not m.get("gate_mask"):
        raise SystemExit(f"arm {a.arm!r} did not pass its gate; it is not served")
    name = a.name or f"exp-036-{a.arm}"
    if name != f"exp-036-{a.arm}":
        raise SystemExit(f"--name {name!r} does not name arm {a.arm!r}; the candidate's "
                         f"file name is bound to the arm it was measured as")

    if not RT.OUT.exists():
        RT.build()
    gates = RT.load()
    tmpl = D.load_submitting()["MVT_ID_mvt"].to_numpy().astype("float64")
    g = gates.set_index("MVT_ID_mvt")
    apt = g["airport"].astype(str).reindex(tmpl).to_numpy()
    reg = g["regime"].astype(str).reindex(tmpl).to_numpy()
    if pd.isna(apt).any():
        raise SystemExit("some template rows have no airport in the gates table")
    universe = set(pd.unique(apt))
    unknown = sorted(set(keep) - universe)
    if unknown:
        raise SystemExit(f"arm {a.arm!r} keeps airports that are not in the gates "
                         f"table: {unknown}")

    base = checked("v7")
    source = checked(src)
    on = np.isin(apt, keep)
    pred = np.where(on, source, base)
    if not np.isfinite(pred).all():
        raise SystemExit("non-finite predictions")
    if float(np.abs(pred[~on] - base[~on]).max()) != 0.0:
        raise SystemExit("a held row differs from the base")

    rec = {"name": name, "arm": a.arm, "mode": "derived",
           "source": src, "source_file": FILES[src][0], "source_sha256": FILES[src][1],
           "base": "v7", "base_file": FILES["v7"][0], "base_sha256": FILES["v7"][1],
           "keep_airports": list(keep), "hold_at_base": list(m["hold_at_v7"]),
           "gates_table": str(RT.OUT.relative_to(ROOT)),
           "gates_table_sha256": D.sha256(RT.OUT),
           "rows_from_source": int(on.sum()), "rows_from_base": int((~on).sum()),
           "rows_moved_vs_base": int((pred != base).sum()),
           "n_noac": int((reg == "noac").sum()),
           "measured_clean": m["clean"], "measured_clean_board": m["clean_board"],
           "measured_gain_vs_K2": m["gain_board_vs_K2"],
           "predicted_board_vs_K2": m.get("predicted_board_vs_K2"),
           "wall_s": round(time.time() - t0, 1)}
    if a.dry_run:
        print(json.dumps(rec, indent=2))
        return
    out = DEST / f"{rec['name']}.parquet"
    ser = pd.Series(pred, index=tmpl)
    rec["sha256"] = D.write_submission(ser, str(out))
    (HERE / "analysis" / f"served-{rec['name']}.json").write_text(json.dumps(rec, indent=2))
    print(f"[serve] wrote {out.name}  sha256 {rec['sha256'][:8]}…  "
          f"{rec['rows_from_source']} from {src}, {rec['rows_from_base']} from v7",
          flush=True)


if __name__ == "__main__":
    main()
