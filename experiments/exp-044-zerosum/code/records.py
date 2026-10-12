"""Sol round 1, blocking 8: the measurement records the contracts require.

  COMMON.md    "Every RESULTS table must show RMSE overall, RMSE on rows with
               target at or below 3600, RMSE on rows above, and RMSE on no-NM
               rows, for each fold."  Written here for v10 and for both ridge
               designs.
  QUEUE.md     "Save row and date contributions, counts, exact vector/input
               hashes and fit provenance."  Per-date CSVs and per-row parquets
               for the served arm, plus a provenance block whose code hashes are
               THIS experiment's -- `packet.code_state()` hashes exp-039's
               directory, because `packet.HERE` belongs to exp-039.

    OMP_NUM_THREADS=3 python records.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zs                                   # noqa: E402
import learner as L                         # noqa: E402
import packet as PK                         # noqa: E402

EXP = zs.EXP
ROOT = zs.ROOT
FOLDS = zs.FOLDS
A = EXP / "analysis"


def slices(y, p, nm_missing) -> dict:
    lo = y <= 3600.0
    return {"rmse_all": round(float(zs.D.rmse(y, p)), 4),
            "rmse_le3600": round(float(zs.D.rmse(y[lo], p[lo])), 4),
            "rmse_gt3600": round(float(zs.D.rmse(y[~lo], p[~lo])), 4),
            "rmse_nonm": round(float(zs.D.rmse(y[nm_missing], p[nm_missing])), 4),
            "n": int(len(y)), "n_le3600": int(lo.sum()), "n_gt3600": int((~lo).sum()),
            "n_nonm": int(nm_missing.sum())}


def main() -> None:
    pan = zs.V.load()
    S = {f: L.fold_state(f, pan[f]) for f in FOLDS}
    amps = json.load(open(A / "learner.json"))["amplitudes"]["ridge"]
    H = np.load(A / "h_vectors.npz")
    rowdir = A / "rows"
    rowdir.mkdir(exist_ok=True)
    (rowdir / ".gitignore").write_text("*.parquet\n")

    out = {"note": "the COMMON.md slice decomposition for v10 and EVERY arm on the "
                   "record (Sol round 2, blocking 4: the first version covered only the "
                   "ridge), and the QUEUE.md row and date contributions for the two "
                   "ridge designs the candidate is read from",
           "slices": {}, "dates": [], "rows": []}

    # every other arm's held-out h, from the three runs that keep them
    extra: dict[str, dict[str, np.ndarray]] = {f: {} for f in FOLDS}
    amp_of: dict[str, dict[str, float]] = {}
    LRN = json.loads((A / "learner.json").read_text())
    for a in LRN["arms"]:
        for cf in ("dayblock", "xmonth"):
            for f in FOLDS:
                extra[f][f"{a}|{cf}"] = H[f"{f}|{a}|{cf}"]
            amp_of[f"{a}|{cf}"] = LRN["amplitudes"][a]
    for name, ampsrc in (("h_controls.npz", None), ("h_repair.npz", "repair")):
        q = A / name
        if not q.exists():
            continue
        Z = np.load(q)
        R = (json.loads((A / "repair.json").read_text()).get("amplitudes")
             if ampsrc else None)
        for k in Z.files:
            f, key = k.split("|", 1)
            extra[f][key] = Z[k]
            arm = key.split("|")[0]
            amp_of[key] = (R[key] if R and key in R else LRN["amplitudes"][arm])

    for f in FOLDS:
        Sf = S[f]
        nm = pan[f]["nm_missing"].to_numpy().astype(bool)
        y, p10 = Sf["y"], Sf["p10"]
        out["slices"][f] = {"v10": slices(y, p10, nm)}
        m0 = Sf["m0"]
        cm0, ucm0 = pd.factorize(Sf["code"][m0])
        for key, hv in extra[f].items():
            d = np.zeros(len(m0))
            d[m0] = amp_of[key][f] * zs.project(hv[m0].astype("float64"), cm0, len(ucm0))
            out["slices"][f][key] = slices(y, p10 + d, nm)
        for cf in ("dayblock", "xmonth"):
            m0 = Sf["m0"]
            cm, ucm = pd.factorize(Sf["code"][m0])
            d = np.zeros(len(m0))
            d[m0] = amps[f] * zs.project(H[f"{f}|ridge|{cf}"][m0].astype("float64"),
                                         cm, len(ucm))
            p = p10 + d
            out["slices"][f][f"ridge|{cf}"] = slices(y, p, nm)   # recomputed here
            g = pd.DataFrame({"date": Sf["days"], "n": 1,
                              "sse_v10": (y - p10) ** 2, "sse_arm": (y - p) ** 2}
                             ).groupby("date").agg(
                n=("n", "sum"), sse_v10=("sse_v10", "sum"), sse_arm=("sse_arm", "sum")
            ).reset_index()
            g["delta_sse"] = g["sse_v10"] - g["sse_arm"]
            pd_ = A / f"dates-ridge-{cf}-{f}.csv"
            g.to_csv(pd_, index=False, float_format="%.6f")
            out["dates"].append(str(pd_.relative_to(ROOT)))
            r = pd.DataFrame({"MVT_ID_mvt": Sf["ids"], "date": Sf["days"], "y": y,
                              "p_v10": p10, "p_arm": p})
            r["sse_v10"] = (r["y"] - r["p_v10"]) ** 2
            r["sse_arm"] = (r["y"] - r["p_arm"]) ** 2
            r["delta_sse"] = r["sse_v10"] - r["sse_arm"]
            pr = rowdir / f"rows-ridge-{cf}-{f}.parquet"
            r.to_parquet(pr, index=False, compression="zstd")
            out["rows"].append(str(pr.relative_to(ROOT)))

    months = sorted({S[f]["month"] for f in FOLDS})
    out["provenance"] = {
        "code_sha256_exp044": {p.name: PK.sha_file(p) for p in sorted(HERE.glob("*.py"))},
        "code_sha256_exp039_imported": {p.name: PK.sha_file(p)
                                        for p in sorted(PK.HERE.glob("*.py"))},
        "v10_fold_vector_sha256": {f: PK.sha_bytes(
            pan[f]["MVT_ID_mvt"].to_numpy("float64"),
            pan[f]["p10"].to_numpy("float64")) for f in FOLDS},
        "uploaded_v10_sha256": zs.V.UPLOADED_SHA,
        "uploaded_v11_sha256": PK.sha_file(
            ROOT / "submissions/uploaded/youthful-giraffe_v11.parquet"),
        "validation_month_file_sha256": {m: PK.sha_file(zs.D.month_file(m)) for m in months},
        "ranking_file_sha256": PK.sha_file(zs.D.data_dir() / "ranking.parquet"),
        "shipped_cache_sha256": {m: PK.sha_file(zs.W_CACHE / f"{m}.parquet") for m in months},
        "git_commit": PK.git_commit(),
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                         capture_output=True, text=True).stdout.strip()),
        "shipped_columns": {"with_cache_vocabulary": int(zs.shipped(months[0]).shape[1]),
                            "without_cache_vocabulary": int(
                                zs.shipped(months[0], drop_cache_vocab=True).shape[1])},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__},
    }
    import lightgbm as lgb
    out["provenance"]["versions"]["lightgbm"] = lgb.__version__
    (A / "records.json").write_text(json.dumps(out, indent=2))
    for f in FOLDS:
        s = out["slices"][f]
        print(f"  [{f}] v10 all {s['v10']['rmse_all']:9.4f}  <=3600 "
              f"{s['v10']['rmse_le3600']:8.4f}  >3600 {s['v10']['rmse_gt3600']:9.2f}  "
              f"no-NM {s['v10']['rmse_nonm']:9.2f}")
        for cf in ("dayblock", "xmonth"):
            a = s[f"ridge|{cf}"]
            print(f"         ridge|{cf:9s} {a['rmse_all']:9.4f}  "
                  f"{a['rmse_le3600']:8.4f}  {a['rmse_gt3600']:9.2f}  {a['rmse_nonm']:9.2f}")
    print(f"  written {A / 'records.json'}")


if __name__ == "__main__":
    main()
