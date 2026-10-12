from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]

FOLDS = ("july_loo", "december", "january")
MARGINS = (0, 60, 300)
NAMES = tuple(f"nm_gt_next_{m}" for m in MARGINS)
RAW = {
    "july_loo": ROOT / "data/raw/training_2025-07-01_2025-08-01.parquet",
    "december": ROOT / "data/raw/training_2025-12-01_2026-01-01.parquet",
    "january": ROOT / "data/raw/training_2025-01-01_2025-02-01.parquet",
}
MANIFEST = EXP / "source_manifest.json"
THREADS = {"OMP_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": "3", "MKL_NUM_THREADS": "3"}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob(path: Path) -> str:
    rel = str(path.relative_to(ROOT))
    return subprocess.check_output(["git", "hash-object", rel], cwd=ROOT, text=True).strip()


def source_entry(path: Path, require_blob: bool = False) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing registered source {path}")
    entry = {"path": str(path.relative_to(ROOT)), "sha256": sha_file(path)}
    if require_blob:
        entry["git_blob"] = git_blob(path)
    return entry


def verify_sources() -> dict:
    if not MANIFEST.is_file():
        raise SystemExit("INCOMPLETE: source_manifest.json is missing")
    record = json.loads(MANIFEST.read_text())
    seen = set()
    for group in ("local_sources", "raw_months", "frozen_inputs"):
        for entry in record[group]:
            path = ROOT / entry["path"]
            if entry["path"] in seen or not path.is_file() or sha_file(path) != entry["sha256"]:
                raise SystemExit(f"INCOMPLETE: missing, duplicate, or changed source {entry['path']}")
            seen.add(entry["path"])
            if "git_blob" in entry and git_blob(path) != entry["git_blob"]:
                raise SystemExit(f"INCOMPLETE: Git identity changed for {entry['path']}")
            if "git_blob" in entry:
                committed = subprocess.run(["git", "rev-parse", f"HEAD:{entry['path']}"], cwd=ROOT,
                                           text=True, capture_output=True)
                if committed.returncode or committed.stdout.strip() != entry["git_blob"]:
                    raise SystemExit(f"INCOMPLETE: source is not the registered committed blob: {entry['path']}")
    return record


def register() -> None:
    if MANIFEST.exists():
        raise SystemExit(f"refuse existing {MANIFEST}")
    sys.path[:0] = [str(ROOT), str(ROOT / "experiments/exp-052-nmv22/code")]
    import rx_common  # noqa: F401
    import rx_price  # noqa: F401
    import rx_iter2  # noqa: F401
    local = {Path(__file__).resolve(), (EXP / "tests/test_nmwitness.py").resolve()}
    for module in tuple(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file:
            path = Path(file).resolve()
            if path.is_file() and ROOT in path.parents and ".venv" not in path.parts and path.suffix == ".py":
                local.add(path)
    i2p = ROOT / "experiments/exp-052-nmv22/analysis/iter2.json"
    trp = ROOT / "experiments/exp-052-nmv22/analysis/trunc.json"
    exp = ROOT / "experiments/exp-052-nmv22/analysis/trunc-exactness.json"
    i2, tr = json.loads(i2p.read_text()), json.loads(trp.read_text())
    frozen = [i2p, trp, exp]
    frozen += [ROOT / i2["selected"]["folds"][f]["rows_file"] for f in FOLDS]
    frozen += [ROOT / tr["runs"]["real"][f]["trunc"] for f in FOLDS]
    for fold in FOLDS:
        if sha_file(ROOT / i2["selected"]["folds"][fold]["rows_file"]) != i2["selected"]["folds"][fold]["rows_sha256"]:
            raise SystemExit(f"{fold}: iter2 manifest row hash mismatch")
        if sha_file(ROOT / tr["runs"]["real"][fold]["trunc"]) != tr["runs"]["real"][fold]["trunc_sha256"]:
            raise SystemExit(f"{fold}: trunc manifest hash mismatch")
    record = {
        "schema": "exp081-source-manifest-v1",
        "registered_family": dict(zip(NAMES, MARGINS)),
        "folds": list(FOLDS), "threads": THREADS,
        "local_sources": [source_entry(p, require_blob=True) for p in sorted(local)],
        "raw_months": [source_entry(RAW[f]) for f in FOLDS],
        "frozen_inputs": [source_entry(p, require_blob=(git_blob(p) == subprocess.run(
            ["git", "rev-parse", f"HEAD:{p.relative_to(ROOT)}"], cwd=ROOT, text=True,
            capture_output=True).stdout.strip())) for p in frozen],
    }
    MANIFEST.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(MANIFEST.relative_to(ROOT)), "sources": len(local),
                      "frozen": len(frozen)}, sort_keys=True))


def witness_membership(dep: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    d = dep.copy()
    a = arr.copy()
    d["airport"] = d["ADEP_mvt"].astype("string")
    a["airport"] = a["ADES_mvt"].astype("string")
    for x in (d, a):
        for col in ("MVT_TIME_UTC_mvt", "AOBT_3_flt", "BLOCK_TIME_UTC_mvt"):
            if col in x:
                x[col] = pd.to_datetime(x[col], utc=True)
        x["stand"] = x["STAND_mvt"].astype("string").str.strip()
        x["key"] = x["airport"] + "|" + x["stand"]
    valid_a = (a["key"].notna() & a["stand"].ne("") & a["BLOCK_TIME_UTC_mvt"].notna()
               & a["MVT_TIME_UTC_mvt"].notna()
               & (a["BLOCK_TIME_UTC_mvt"] >= a["MVT_TIME_UTC_mvt"]))
    a = a.loc[valid_a, ["key", "BLOCK_TIME_UTC_mvt"]].rename(
        columns={"BLOCK_TIME_UTC_mvt": "next_arrival"})
    a = a.sort_values("next_arrival", kind="mergesort").drop_duplicates(["key", "next_arrival"])
    good_d = d["key"].notna() & d["stand"].ne("") & d["MVT_TIME_UTC_mvt"].notna() & d["AOBT_3_flt"].notna()
    left = d.loc[good_d, ["MVT_ID_mvt", "key", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]].copy()
    left["_idx"] = left.index
    left = left.sort_values("MVT_TIME_UTC_mvt", kind="mergesort")
    a = a.sort_values("next_arrival", kind="mergesort")
    got = pd.merge_asof(left, a, left_on="MVT_TIME_UTC_mvt", right_on="next_arrival",
                        by="key", direction="forward", allow_exact_matches=False)
    out = pd.DataFrame({"MVT_ID_mvt": d["MVT_ID_mvt"].astype("float64")})
    for name, margin in zip(NAMES, MARGINS):
        vals = got["AOBT_3_flt"] > got["next_arrival"] + pd.to_timedelta(margin, unit="s")
        out[name] = False
        out.loc[got["_idx"].to_numpy(), name] = vals.fillna(False).to_numpy(bool)
    return out


def read_input_month(fold: str) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    path = RAW[fold]
    dc = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]
    ac = ["PHASE_mvt", "ADES_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"]
    dep = pd.read_parquet(path, columns=dc, filters=[("PHASE_mvt", "==", "DEP")])
    arr = pd.read_parquet(path, columns=ac, filters=[("PHASE_mvt", "!=", "DEP")])
    for frame in (dep, arr):
        for c in ("MVT_TIME_UTC_mvt", "AOBT_3_flt", "BLOCK_TIME_UTC_mvt"):
            if c in frame:
                frame[c] = pd.to_datetime(frame[c], utc=True)
    return dep.reset_index(drop=True), arr.reset_index(drop=True), path


def strict_membership(ids: np.ndarray, frame: pd.DataFrame) -> pd.DataFrame:
    mids = frame.MVT_ID_mvt.to_numpy("float64")
    if len(np.unique(mids)) != len(mids):
        raise ValueError("duplicate membership ID")
    if len(ids) != len(mids) or not np.array_equal(ids.astype("float64"), mids):
        raise ValueError("missing, extra, or reordered membership ID")
    if list(frame.columns) != ["MVT_ID_mvt", *NAMES] or any(frame[n].dtype != bool for n in NAMES):
        raise ValueError("membership schema mismatch")
    return frame.set_index("MVT_ID_mvt")


def render_outputs(results: dict, repo_root: Path = ROOT) -> dict:
    """Write every registered human-readable output and an exact receipt."""
    exp = repo_root / "experiments/exp-081-nmwitness"
    table_path = exp / "analysis/tables.md"
    results_path = exp / "RESULTS.md"
    climb_path = exp / "climb.log"
    ledger_path = repo_root / "docs/LEDGER.md"
    receipt_path = exp / "analysis/output-receipt.json"
    for path in (table_path, results_path, climb_path, receipt_path):
        if path.exists():
            raise SystemExit(f"refuse existing {path}")
    table_lines = ["# exp-081 registered tables", "", "| witness | fold | support | coefficients | held gains s2/row |",
                   "|---|---|---:|---|---|"]
    for witness, wr in results["witnesses"].items():
        for fold, fr in wr["folds"].items():
            table_lines.append(f"| {witness} | {fold} | {fr['support']} | {fr['coefficients']} | "
                               f"{fr['held_block_gain_s2_per_row']} |")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(table_lines) + "\n")
    primary = [w for w, rec in results["witnesses"].items() if rec["primary_positive"]]
    reading = results["reading"].upper()
    results_path.write_text(
        "# exp-081-nmwitness — RESULTS\n\n"
        f"## Registered reading\n\n**{reading}.** Primary-positive witnesses: {primary or 'none'}. "
        "This run changes no candidate, prediction, serving rule, or board expectation.\n\n"
        "## Deterministic records\n\nSee `analysis/results.json`, `analysis/tables.md`, and "
        "`analysis/output-receipt.json` for the complete registered result.\n")
    july = results["witnesses"][NAMES[0]]["folds"]["july_loo"]["five_table"]["rmse_new"]
    dec = results["witnesses"][NAMES[0]]["folds"]["december"]["five_table"]["rmse_new"]
    metric = results["witnesses"][NAMES[0]]["board"]["five_vs_four"]["all_row_gain_s"]
    climb = (f"exp-081 iteration=1 mechanism=fixed_nm_next_arrival july_rmse={july:.9f} "
             f"december_rmse={dec:.9f} decision_metric={metric:.12f} gap_fraction=0 remaining_gap=NA "
             f"reading={results['reading']}\n")
    climb_path.write_text(climb)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_row = (f"| 2026-09-14 | exp-081 | nmwitness | registered raw-stand next-arrival witness: "
                  f"{results['reading']} | {july:.3f} | {dec:.3f} | - | - | "
                  "experiments/exp-081-nmwitness/analysis/results.json; RESULTS.md; tables.md | "
                  "unreviewed by assignment | fixed exp-052 d22* contrast; no candidate or upload |\n")
    with ledger_path.open("a") as fh:
        fh.write(ledger_row)
    receipt = {"schema": "exp081-output-receipt-v1", "reading": results["reading"],
               "required_outputs": {str(p.relative_to(repo_root)): sha_file(p)
                                    for p in (table_path, results_path, climb_path, ledger_path)},
               "all_required_outputs_present": True}
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def membership() -> None:
    pre = verify_sources()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    start_wall, start_ru = time.time(), resource.getrusage(resource.RUSAGE_SELF)
    load = os.getloadavg()[0]
    if load > 10:
        raise SystemExit(f"load gate held: {load:.2f} > 10")
    rec = {"schema": "exp081-membership-v1", "registered_family": dict(zip(NAMES, MARGINS)), "folds": {}}
    for fold in FOLDS:
        dep, arr, src = read_input_month(fold)
        out = witness_membership(dep, arr)
        path = EXP / f"work/membership-{fold}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise SystemExit(f"refuse existing {path}")
        out.to_parquet(path, index=False)
        rec["folds"][fold] = {"source": str(src), "source_sha256": sha_file(src),
                              "path": str(path.relative_to(ROOT)), "sha256": sha_file(path),
                              "rows": len(out), "support": {n: int(out[n].sum()) for n in NAMES}}
    dest = EXP / "analysis/membership.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise SystemExit(f"refuse existing {dest}")
    end_ru = resource.getrusage(resource.RUSAGE_SELF)
    rec["receipt"] = {"start_utc": started, "end_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_seconds": round(time.time() - start_wall, 3),
        "process_cpu_seconds": round((end_ru.ru_utime + end_ru.ru_stime) - (start_ru.ru_utime + start_ru.ru_stime), 3),
        "peak_rss_mb": round(end_ru.ru_maxrss / 1024**2, 1), "prelaunch_load_1m": load,
        "threads": THREADS, "command": "nmwitness.py membership", "exit_status": 0}
    verify_sources()
    rec["receipt"]["experiment_size_bytes_before_output"] = sum(p.stat().st_size for p in EXP.rglob("*") if p.is_file())
    dest.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    rec["receipt"]["output_sha256"] = sha_file(dest)
    print(json.dumps(rec, indent=2))


def score() -> None:
    verify_sources()
    load = os.getloadavg()[0]
    if load > 10:
        raise SystemExit(f"load gate held: {load:.2f} > 10")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    start_wall, start_ru = time.time(), resource.getrusage(resource.RUSAGE_SELF)
    from scipy.stats import binomtest
    sys.path[:0] = [str(ROOT), str(ROOT / "experiments/exp-052-nmv22/code")]
    import rx_common as C
    import rx_price as RP
    import rx_iter2 as RI

    mempath = EXP / "analysis/membership.json"
    if not mempath.is_file():
        raise SystemExit("INCOMPLETE: membership table and receipt are required")
    memrec = json.loads(mempath.read_text())
    if memrec.get("schema") != "exp081-membership-v1" or memrec.get("receipt", {}).get("exit_status") != 0:
        raise SystemExit("INCOMPLETE: invalid membership receipt")
    if not any(memrec["folds"][f]["support"][n] for f in FOLDS for n in NAMES):
        raise SystemExit("registered no-support reading; labels were not opened")
    i2p = ROOT / "experiments/exp-052-nmv22/analysis/iter2.json"
    i2 = json.loads(i2p.read_text())
    ex = json.loads((ROOT / "experiments/exp-052-nmv22/analysis/trunc-exactness.json").read_text())
    if not ex["passes"]:
        raise SystemExit("exp-052 truncation exactness failed")
    trec = json.loads((ROOT / "experiments/exp-052-nmv22/analysis/trunc.json").read_text())
    results = {"sources": {"iter2": {"path": str(i2p.relative_to(ROOT)), "sha256": sha_file(i2p)}},
               "witnesses": {}, "runtime": {}}
    FD, old, members, no_nm = {}, {}, {}, {}
    for fold in FOLDS:
        F0 = RP.load(fold, False, None, False)
        tr = RI.load_trunc(fold, False, trec)
        ks = i2["selection"][fold]["k_by_block"]
        FD[fold] = RI.with_values(F0, tr, ks)
        rp = ROOT / i2["selected"]["folds"][fold]["rows_file"]
        if sha_file(rp) != i2["selected"]["folds"][fold]["rows_sha256"]:
            raise SystemExit(f"{fold}: exp-052 row source moved")
        old[fold] = pd.read_parquet(rp)
        mp = ROOT / memrec["folds"][fold]["path"]
        if sha_file(mp) != memrec["folds"][fold]["sha256"]:
            raise SystemExit(f"{fold}: membership moved")
        mm = pd.read_parquet(mp)
        try:
            members[fold] = strict_membership(FD[fold]["df"].MVT_ID_mvt.to_numpy("float64"), mm)
        except ValueError as exc:
            raise SystemExit(f"{fold}: {exc}") from exc
        import pyarrow.parquet as pq
        flt = [c for c in pq.ParquetFile(RAW[fold]).schema.names if c.endswith("_flt")]
        raw_nm = pd.read_parquet(RAW[fold], columns=["MVT_ID_mvt", "PHASE_mvt", *flt],
                                 filters=[("PHASE_mvt", "==", "DEP")]).reset_index(drop=True)
        if len(raw_nm) != len(FD[fold]["df"]) or not np.array_equal(
                raw_nm.MVT_ID_mvt.to_numpy("float64"), FD[fold]["df"].MVT_ID_mvt.to_numpy("float64")):
            raise SystemExit(f"{fold}: raw no-NM IDs are missing, extra, or reordered")
        no_nm[fold] = raw_nm[flt].isna().all(axis=1).to_numpy(bool)

    def coef(r, d, mask, cg):
        n = int(mask.sum())
        raw = float(np.dot(r[mask], d[mask]) / np.dot(d[mask], d[mask])) if n and np.dot(d[mask], d[mask]) else cg
        return float(np.clip(n / (n + 1000.0) * raw + 1000.0 / (n + 1000.0) * cg, 0, 1))

    for name in NAMES:
        U5, U4, UG, UA, OR = {}, {}, {}, {}, {}
        PLACE = [dict() for _ in C.SEEDS_PLACEBO]
        wr = {"folds": {}}
        for fold in FOLDS:
            F, rows = FD[fold], old[fold]
            df, P = F["df"], F["df"]["pop"].to_numpy(bool)
            ids = df.MVT_ID_mvt.to_numpy("float64")
            if not np.array_equal(ids, rows.MVT_ID_mvt.to_numpy("float64")):
                raise SystemExit(f"{fold}: row order changed")
            mi = members[fold]
            Wfull = mi[name].to_numpy(bool) & P
            WP = Wfull[P]
            blk = df.block.to_numpy(int)[P]
            ua, ub = F["O"]["aware"], F["O"]["blind"]
            d = ub - ua
            r = df.r.to_numpy("float64")[P] - ua
            u4 = rows.u_router.to_numpy("float64")
            ug = rows.u_global.to_numpy("float64")
            ua_full = rows.u_q_aware.to_numpy("float64")
            u5 = u4.copy()
            coefs = []
            for b in range(C.K):
                dev, sco = blk != b, blk == b
                di = F["I"]["blind"][b] - F["I"]["aware"][b]
                ri = df.r.to_numpy("float64")[P] - F["I"]["aware"][b]
                cg = C.NR._global(ri, di, dev)
                c = coef(ri, di, dev & WP, cg)
                coefs.append(c)
                ix = np.flatnonzero(P)[sco & WP]
                u5[ix] = ua_full[ix] + c * d[sco & WP]
            if not np.array_equal(u5[~Wfull].view("uint64"), u4[~Wfull].view("uint64")):
                raise SystemExit(f"{fold}: fifth cell changed a non-witness row")
            if not np.array_equal(u5[~P].view("uint64"), np.zeros((~P).sum()).view("uint64")):
                raise SystemExit(f"{fold}: a row outside P moved")
            oracle = ua_full.copy()
            cstar = np.where(d != 0, np.clip(r / d, 0, 1), 0)
            oracle[np.flatnonzero(P)[WP]] = ua[WP] + cstar[WP] * d[WP]
            agree = (np.sign(d[WP]) == np.sign(r[WP])) & (np.sign(d[WP]) != 0)
            n, k = int(WP.sum()), int(agree.sum())
            by_block_cell = {str(b): {str(c): int((Wfull & P & (df.block.to_numpy(int) == b)
                & (df.cell.to_numpy(int) == c)).sum()) for c in range(4)} for b in range(5)}
            # exp-052 contract: r = y - p_v22.  Held-block gain is therefore
            # exactly old residual-update loss minus fifth-cell residual-update loss.
            residual = df.y.to_numpy("float64") - df.p_v22.to_numpy("float64")
            if not np.array_equal(residual, df.r.to_numpy("float64")):
                raise SystemExit(f"{fold}: exp-052 residual contract changed")
            held_gain = {str(b): float(np.mean((residual[df.block.to_numpy(int) == b] -
                u4[df.block.to_numpy(int) == b]) ** 2) -
                np.mean((residual[df.block.to_numpy(int) == b] -
                u5[df.block.to_numpy(int) == b]) ** 2)) for b in range(5)}
            wr["folds"][fold] = {"support": n, "by_block": {str(b): int((WP & (blk == b)).sum()) for b in range(5)},
                "by_old_cell": {str(c): int((Wfull & (df.cell.to_numpy(int) == c)).sum()) for c in range(4)},
                "by_block_and_old_cell": by_block_cell, "held_block_gain_s2_per_row": held_gain,
                "sign_agree": k, "sign_share": k / n if n else None,
                "binom_p_one_sided": float(binomtest(k, n, 0.5, alternative="greater").pvalue) if n else None,
                "coefficients": coefs,
                "five_table": RP.table(df, u5),
                "five_vs_four": RP.paired(df, u4, u5), "five_vs_global": RP.paired(df, ug, u5)}
            yy = df.y.to_numpy("float64")
            pp = df.p_v22.to_numpy("float64") + u5
            clean = np.isfinite(yy) & np.isfinite(pp)
            def subset_rmse(mask):
                z = clean & mask
                return {"rows": int(z.sum()), "rmse": float(np.sqrt(np.mean((yy[z] - pp[z]) ** 2))) if z.any() else None}
            wr["folds"][fold]["common_rmse"] = {
                "overall": subset_rmse(np.ones(len(df), dtype=bool)),
                "target_at_most_3600": subset_rmse(yy <= 3600),
                "target_over_3600": subset_rmse(yy > 3600),
                "no_nm_all_flt_null": subset_rmse(no_nm[fold]),
                "clean": subset_rmse(clean),
            }
            U5[fold], U4[fold], UG[fold], UA[fold], OR[fold] = u5, u4, ug, ua_full, oracle
            key = (df.airport.astype(str) + "|" + df.day.astype(str)).to_numpy()[P]
            for j, seed in enumerate(C.SEEDS_PLACEBO):
                perm = C.airday_perm(key, np.random.default_rng(seed))
                Wp = WP[perm]
                up = u4.copy()
                for b in range(C.K):
                    dev, sco = blk != b, blk == b
                    di = F["I"]["blind"][b] - F["I"]["aware"][b]
                    ri = df.r.to_numpy("float64")[P] - F["I"]["aware"][b]
                    cg = C.NR._global(ri, di, dev)
                    c = coef(ri, di, dev & Wp, cg)
                    ix = np.flatnonzero(P)[sco & Wp]
                    up[ix] = ua_full[ix] + c * d[sco & Wp]
                PLACE[j][fold] = up
        wr["board"] = {"five_vs_four": RP.board({f: FD[f]["df"] for f in FOLDS}, U4, U5),
                       "five_vs_global": RP.board({f: FD[f]["df"] for f in FOLDS}, UG, U5),
                       "oracle_vs_aware": RP.board({f: FD[f]["df"] for f in FOLDS}, UA, OR)}
        real_placebo_stat = wr["board"]["five_vs_four"]["all_row_gain_s"]
        refs = [C.NR.board_point({f: FD[f]["df"] for f in FOLDS}, U4, p) for p in PLACE]
        wr["placebo_partition"] = RP.perm_summary(real_placebo_stat, refs)
        wr["passes"] = {
            "positive_coefficients": all(c > 0 for f in FOLDS for c in wr["folds"][f]["coefficients"]),
            "sign": all((wr["folds"][f]["sign_share"] or 0) > .5 and (wr["folds"][f]["binom_p_one_sided"] or 1) < .05/9 for f in FOLDS),
            "oracle_material": wr["board"]["oracle_vs_aware"]["all_row_gain_s"] >= .10,
            "five_vs_four": wr["board"]["five_vs_four"]["all_row_ci"]["ci_block_1d"][0] > 0 and all(wr["folds"][f]["five_vs_four"]["mse_s2_per_row"] >= 0 for f in FOLDS),
            "five_vs_global": wr["board"]["five_vs_global"]["all_row_ci"]["ci_block_1d"][0] > 0 and all(wr["folds"][f]["five_vs_global"]["mse_s2_per_row"] >= 0 for f in FOLDS)}
        wr["primary_positive"] = all(wr["passes"].values())
        results["witnesses"][name] = wr
    results["reading"] = "positive" if any(x["primary_positive"] for x in results["witnesses"].values()) else "negative"
    ru = resource.getrusage(resource.RUSAGE_SELF)
    results["runtime"] = {"start_utc": started, "end_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                          "wall_seconds": round(time.time() - start_wall, 2),
                          "process_cpu_seconds": round((ru.ru_utime + ru.ru_stime) - (start_ru.ru_utime + start_ru.ru_stime), 2),
                          "peak_rss_mb": round(ru.ru_maxrss / 1024**2, 1), "prelaunch_load_1m": load,
                          "threads": THREADS, "command": "nmwitness.py score", "exit_status": 0}
    out = EXP / "analysis/results.json"
    if out.exists():
        raise SystemExit(f"refuse existing {out}")
    verify_sources()
    out.write_text(json.dumps(results, indent=2, sort_keys=True, default=float) + "\n")
    render_outputs(results, ROOT)
    print(json.dumps({"reading": results["reading"], "runtime": results["runtime"], "primary": {k:v["primary_positive"] for k,v in results["witnesses"].items()}}, indent=2))


START = time.time()
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("register", "membership", "score"))
    args = ap.parse_args()
    register() if args.phase == "register" else membership() if args.phase == "membership" else score()
