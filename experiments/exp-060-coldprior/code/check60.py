"""exp-060-coldprior: the isolated checker (SPEC 3.7).

The assignment forbids opening exp-056's `pred-*` archives or row-loss artifacts inside table,
selection, correction or pricing code, and allows an isolated checker to compare a newly frozen
zero prediction with exp-056's zero OOF only AFTER all new zero choices exist.  This job is that
checker.  It writes nothing that any fit, selection or pricing step reads.

  --stage fit    after `fit60.py`: the new cold `zero` correction's array SHA-256 against the
                 value exp-056 recorded in `analysis/fit-<r>.json` (a hash in a record, not a
                 prediction archive).  Refuses a mismatch.
  --stage price  after `price60.py`: the new `zero` per-row V loss against the loss recomputed
                 from exp-056's `oof/zero-iso_<r>_v.parquet`.  Refuses a mismatch.

    .venv/bin/python experiments/exp-060-coldprior/code/check60.py --stage fit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e60 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402


def stage_fit() -> dict:
    out = {"stage": "fit", "utc": E.utc(),
           "what": ("the cold zero arm must reproduce exp-056 iteration 1's zero correction byte "
                    "for byte; the cold vocabulary differs from the warm one, and the label-free "
                    "probe in SPEC 1.4 is why that is attainable"),
           "rotations": {}}
    ok = True
    for r in E.ORDER:
        mine = E.load_json(E.ANALYSIS / f"fit60-{r}.json")
        theirs = E.load_json(E.E56 / f"analysis/fit-{r}.json")
        want = theirs["array_sha256"]["h_zero_main"]
        got = {lev: mine["arms"]["zero"]["levels"][lev]["h_sha256"] for lev in ("B", "A")}
        same = {lev: got[lev] == want for lev in got}
        out["rotations"][r] = {
            "exp056_h_zero_main_sha256": want, "exp060_h_zero_sha256": got,
            "byte_identical": same,
            "exp056_lam": theirs["arms"]["zero"]["lam"], "exp056_beta": theirs["arms"]["zero"]["beta"],
            "exp060_lam": mine["arms"]["zero"]["levels"]["B"]["lam"],
            "exp060_c": mine["arms"]["zero"]["levels"]["B"]["c"],
            "exp060_betas": mine["arms"]["zero"]["levels"]["B"]["betas_by_airport"]}
        ok &= all(same.values())
    out["all_byte_identical"] = bool(ok)
    return out


def stage_price() -> dict:
    out = {"stage": "price", "utc": E.utc(),
           "what": ("the cold zero arm's per-row V loss must reproduce exp-056 iteration 1's zero "
                    "V loss byte for byte; exp-056's OOF file is read here only, after every new "
                    "zero choice and the V verdict exist"),
           "rotations": {}}
    oos = E.load_json(E.ANALYSIS / "oos.json")
    rows = pd.read_parquet(E.ROOT / oos["artifacts"]["rows"]["path"],
                           columns=["rotation", "MVT_ID_mvt", "loss:zero|B"])
    ok = True
    for r in E.ORDER:
        mine = rows[rows["rotation"] == r].reset_index(drop=True)
        p = E.E56 / f"oof/zero-iso_{r}_v.parquet"
        th = pd.read_parquet(p)
        if not np.array_equal(mine["MVT_ID_mvt"].to_numpy("int64"),
                              th["MVT_ID_mvt"].to_numpy("int64")):
            raise SystemExit(f"{r}: exp-056's zero OOF is not in the same row order")
        their_loss = (th["y_true"].to_numpy("float64") - th["y_pred"].to_numpy("float64")) ** 2
        my_loss = mine["loss:zero|B"].to_numpy("float64")
        same = bool(np.array_equal(my_loss, their_loss))
        out["rotations"][r] = {
            "exp056_oof": str(p.relative_to(E.ROOT)), "exp056_oof_sha256": E.sha_file(p),
            "rows": int(len(my_loss)), "byte_identical": same,
            "max_abs_difference": float(np.max(np.abs(my_loss - their_loss))),
            "exp060_loss_sha256": E.sha_arr(my_loss), "exp056_loss_sha256": E.sha_arr(their_loss),
            "exp060_zero_rmse_s": float(np.sqrt(my_loss.mean())),
            "exp056_zero_rmse_s": float(np.sqrt(their_loss.mean()))}
        ok &= same
    out["all_byte_identical"] = bool(ok)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("fit", "price"), required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    p = E.ANALYSIS / f"check-{a.stage}.json"
    NO.refuse_existing([p], a.overwrite)
    t0 = time.time()
    rec = stage_fit() if a.stage == "fit" else stage_price()
    rec["code_sha256"] = E.code_shas()
    rec["seconds"] = round(time.time() - t0, 1)
    NO.refuse_existing([p], a.overwrite)
    p.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(json.dumps({r: v["byte_identical"] for r, v in rec["rotations"].items()}
                     | {"all_byte_identical": rec["all_byte_identical"]}), flush=True)
    if not rec["all_byte_identical"]:
        raise SystemExit(f"REFUSING: the cold zero arm does not reproduce exp-056 iteration 1 "
                         f"({a.stage} stage); see {p.relative_to(E.ROOT)}")
    print(f"written {p.relative_to(E.ROOT)}")


if __name__ == "__main__":
    main()
