"""The member pack for valid's R4 level 2 (R4-SPLIT-CONTRACT.md): one stage on every R4 split.

    $PY code/pack.py --stage HA2 --src out/pack-fits --out out/pack

Each fit file holds the clock-class rows of its split. The pack file of a split holds the exact
scored IDs of valid's L fit of that split, in its row order, with the head's prediction on the
clock-class rows and NaN on every other row (the head is a class member; the level 2 reads
`d_<stage>` as a missing value there). Checks before a file is written: the fit passes
`pilot.load`; its IDs are in the L reference and its context columns equal the reference's; every
frame clock row of the split's scored set that the reference holds has a prediction. After the
last file: the consumers q01, q07 and qfinal admit their splits, the combined training IDs are
unique, and PACK.json binds every file's sha256. Write-once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
import pilot as P                     # noqa: E402

#: SHA-256 of the int64 bytes of the one fit ID that the contract leaves out of jjx-03 (row IDs stay out of the
#: public source)
JJX03_LEFT_OUT = "6057550220ef15103ecab3bca462565ab7dfcd47c82b7a01f22c134bc18f5789"
JJX = ("jjx-12", "jjx-02", "jjx-06", "jjx-08", "jjx-03", "jjx-10", "jjx-11")
#: valid's L fit of each split (read only): the exact scored IDs of the contract
L_REF = {"jjx-12": "kaggle/r2-la/out", "jjx-02": "kaggle/r2-la/out", "jjx-06": "kaggle/r2-lb/out",
         "jjx-08": "kaggle/r2-lb/out", "jjx-03": "kaggle/r4-la/out", "jjx-10": "kaggle/r4-la/out",
         "jjx-11": "kaggle/r4-lb/out", "janjul": "kaggle/ws-2/out", "final": "kaggle/a/out",
         "jjrank": "kaggle/fj-L/out", "jjrefit": "kaggle/fj-L/out"}
ROWS = {"jjx-12": 165_588, "jjx-02": 143_673, "jjx-06": 182_953, "jjx-08": 190_983, "jjx-03": 164_389,
        "jjx-10": 185_556, "jjx-11": 162_277, "janjul": 344_017, "final": 344_453, "jjrank": None,
        "jjrefit": None}
CTX = ["month", "utc_day", "airport"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="*", default=list(L_REF))
    a = ap.parse_args()
    col = f"pred_{a.stage}"
    C.refuse_existing(a.out / "PACK.json")
    for n in a.splits:
        C.refuse_existing(a.out / f"{a.stage}-{n}.parquet", a.out / f"{a.stage}-{n}.json")
    a.out.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(C.FRAME, columns=[C.ID, C.TIME, "month"])
    files, summary = {}, {}
    for n in a.splits:
        s = P.split(n)
        if s.token != C.TOKENS[n]:
            raise SystemExit(f"{n}: token {s.token} is not the contract's {C.TOKENS[n]}")
        fit = P.load(a.src / f"{a.stage}-{n}.parquet", n)
        rec = json.loads((a.src / f"{a.stage}-{n}.json").read_text())
        ref_path = C.W148 / L_REF[n] / f"L-{n}.parquet"
        ref = pd.read_parquet(ref_path)
        if ROWS[n] is not None and len(ref) != ROWS[n]:
            raise SystemExit(f"{n}: the L reference has {len(ref)} rows, the contract says {ROWS[n]}")
        if ref[C.ID].duplicated().any() or fit[C.ID].duplicated().any():
            raise SystemExit(f"{n}: repeated ids")
        if not np.isfinite(fit[col].to_numpy()).all():
            raise SystemExit(f"{n}: a fit row has no prediction")
        # the clock rows this split scores, from the frame (not from the fit file)
        if n in P.RANKING_SPLITS:
            want = frame.loc[frame.month.isin(C.MONTHS_2026), C.ID]
        else:
            want = frame.loc[frame.month.isin(C.MONTHS_2025).to_numpy() & s.score_mask(frame[[C.TIME]]), C.ID]
        want = set(want.astype("int64"))
        have, refids = set(fit[C.ID].astype("int64")), set(ref[C.ID].astype("int64"))
        if have != want:
            raise SystemExit(f"{n}: the fit scores {len(have)} rows, the frame has {len(want)} clock rows")
        outside = sorted(have - refids)
        if n == "jjx-03" and [hashlib.sha256(np.int64(i).tobytes()).hexdigest() for i in outside] == [JJX03_LEFT_OUT]:
            outside = []                       # the contract leaves this id out of jjx-03
        elif outside:
            raise SystemExit(f"{n}: {len(outside)} fit ids are not in the L reference: {outside[:5]}")
        out = ref[[C.ID] + CTX].merge(fit[[C.ID] + CTX + [col]], on=C.ID, how="left",
                                      suffixes=("", "_fit"), validate="one_to_one")
        hit = out[col].notna().to_numpy()
        for c in CTX:
            if not (out.loc[hit, c].astype(str) == out.loc[hit, f"{c}_fit"].astype(str)).all():
                raise SystemExit(f"{n}: column {c} differs from the L reference")
        out = out[[C.ID] + CTX + [col]]
        if not (out[C.ID].to_numpy() == ref[C.ID].to_numpy()).all():
            raise SystemExit(f"{n}: the row order changed")
        keep = {k: rec[k] for k in ("arm", "target", "learner", "lightgbm", "numpy", "pandas", "fit", "features",
                                    "features_sha256", "train_rows", "train_rows_by_month", "train_split",
                                    "class", "inputs", "code_sha256", "source_commit", "domain") if k in rec}
        P.write(a.out, a.stage, s, n, out, dict(
            keep, member_class="clock", class_rows=int(hit.sum()), nan_rows=int((~hit).sum()),
            id_reference={"path": str(ref_path), "sha256": C.sha(ref_path), "rows": int(len(ref))},
            fit_artifact={"path": str(a.src / f"{a.stage}-{n}.parquet"), "sha256": rec["output_sha256"]},
            pack_code_sha256=C.sha(__file__)))
        files[n] = {"parquet": C.sha(a.out / f"{a.stage}-{n}.parquet"), "json": C.sha(a.out / f"{a.stage}-{n}.json")}
        summary[n] = {"rows": int(len(out)), "class_rows": int(hit.sum()), "token": s.token,
                      "train_rows": rec["train_rows"]}
        print(n, summary[n], flush=True)

    # consumers: the level 2 of January, of July and the served one
    checks = {}
    made = [n for n in a.splits]
    for cons, held in (("q01", "2025-01"), ("q07", "2025-07"), ("qfinal", None)):
        c = P.consumer(cons)
        ids = []
        for n in [x for x in JJX + ("janjul",) if x in made]:
            rec = json.loads((a.out / f"{a.stage}-{n}.json").read_text())
            c.assert_admits(rec["split"], what=f"{cons} <- {a.stage}-{n}")
            if n != "janjul":
                ids.append(pd.read_parquet(a.out / f"{a.stage}-{n}.parquet", columns=[C.ID])[C.ID])
        if cons == "qfinal" and "final" in made:
            c.assert_admits(json.loads((a.out / f"{a.stage}-final.json").read_text())["split"], what="qfinal <- final")
        allid = pd.concat(ids) if ids else pd.Series(dtype="int64")
        if allid.duplicated().any():
            raise SystemExit(f"{cons}: a training id is in two jjx splits")
        checks[cons] = {"admitted": [x for x in JJX + ("janjul",) if x in made], "training_ids": int(len(allid)),
                        "scores": held or "2026-01, 2026-07"}
    for n in ("jjrank", "jjrefit"):
        if n in made:
            P.load(a.out / f"{a.stage}-{n}.parquet", n)
    C.write_json(a.out / "PACK.json", {"what": "exp-159 clock-head member pack (R4 splits and the fold form)",
                                       "stage": a.stage, "column": col, "member_class": "clock",
                                       "outside_class": "NaN", "floor": "none applied; apply 60 s when scoring",
                                       "files": files, "splits": summary, "consumer_checks": checks,
                                       "splits_fingerprint": P.WP.fingerprint(),
                                       "code_sha256": {"pack.py": C.sha(__file__), "common.py": C.sha(C.__file__)}})
    print("PACK.json", C.sha(a.out / "PACK.json"))


if __name__ == "__main__":
    main()
