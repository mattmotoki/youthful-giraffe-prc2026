from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/exp-052-nmv22/code")]
import prc.data as D  # noqa: E402

FOLDS = ("july_loo", "december", "january")
MARGINS = (0, 60, 300)
NAMES = tuple(f"nm_gt_next_{m}" for m in MARGINS)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def witness_membership(dep: pd.DataFrame, arr: pd.DataFrame) -> pd.DataFrame:
    d = dep.copy()
    a = arr.copy()
    d["airport"] = d["ADEP_mvt"].astype("string")
    a["airport"] = a["ADES_mvt"].astype("string")
    for x in (d, a):
        x["stand"] = x["STAND_mvt"].astype("string").str.strip()
        x["key"] = x["airport"] + "|" + x["stand"]
    valid_a = (a["key"].notna() & a["BLOCK_TIME_UTC_mvt"].notna()
               & a["MVT_TIME_UTC_mvt"].notna()
               & (a["BLOCK_TIME_UTC_mvt"] >= a["MVT_TIME_UTC_mvt"]))
    a = a.loc[valid_a, ["key", "BLOCK_TIME_UTC_mvt"]].rename(
        columns={"BLOCK_TIME_UTC_mvt": "next_arrival"})
    a = a.sort_values("next_arrival", kind="mergesort").drop_duplicates(["key", "next_arrival"])
    good_d = d["key"].notna() & d["MVT_TIME_UTC_mvt"].notna() & d["AOBT_3_flt"].notna()
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
    path = D.month_file(D.FOLDS[fold].valid_month)
    dc = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]
    ac = ["PHASE_mvt", "ADES_mvt", "STAND_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"]
    dep = pd.read_parquet(path, columns=dc, filters=[("PHASE_mvt", "==", "DEP")])
    arr = pd.read_parquet(path, columns=ac, filters=[("PHASE_mvt", "!=", "DEP")])
    for frame in (dep, arr):
        for c in ("MVT_TIME_UTC_mvt", "AOBT_3_flt", "BLOCK_TIME_UTC_mvt"):
            if c in frame:
                frame[c] = pd.to_datetime(frame[c], utc=True)
    return dep.reset_index(drop=True), arr.reset_index(drop=True), path


def membership() -> None:
    rec = {"registered_family": dict(zip(NAMES, MARGINS)), "folds": {}}
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
    dest.write_text(json.dumps(rec, indent=2, sort_keys=True))
    print(json.dumps(rec, indent=2))


def score() -> None:
    from scipy.stats import binomtest
    import rx_common as C
    import rx_price as RP
    import rx_iter2 as RI

    memrec = json.loads((EXP / "analysis/membership.json").read_text())
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
    FD, old, members = {}, {}, {}
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
        members[fold] = pd.read_parquet(mp).set_index("MVT_ID_mvt")

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
            Wfull = members[fold][name].reindex(ids).fillna(False).to_numpy(bool) & P
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
            wr["folds"][fold] = {"support": n, "by_block": {str(b): int((WP & (blk == b)).sum()) for b in range(5)},
                "by_old_cell": {str(c): int((Wfull & (df.cell.to_numpy(int) == c)).sum()) for c in range(4)},
                "sign_agree": k, "sign_share": k / n if n else None,
                "binom_p_one_sided": float(binomtest(k, n, 0.5, alternative="greater").pvalue) if n else None,
                "coefficients": coefs,
                "five_table": RP.table(df, u5),
                "five_vs_four": RP.paired(df, u4, u5), "five_vs_global": RP.paired(df, ug, u5)}
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
    results["runtime"] = {"seconds": round(time.time() - START, 2),
                          "process_cpu_seconds": round(ru.ru_utime + ru.ru_stime, 2),
                          "peak_rss_mb": round(ru.ru_maxrss / 1024**2, 1),
                          "load_1m": os.getloadavg()[0]}
    out = EXP / "analysis/results.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True, default=float))
    print(json.dumps({"reading": results["reading"], "runtime": results["runtime"], "primary": {k:v["primary_positive"] for k,v in results["witnesses"].items()}}, indent=2))


START = time.time()
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("membership", "score"))
    args = ap.parse_args()
    membership() if args.phase == "membership" else score()
