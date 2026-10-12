"""Export of the fold values of the v3 no-plan form and its cell variations (for `stack`, desk request 10-02).

    .venv/bin/python experiments/exp-167-target/code/noplan_form_v3_fold_export.py --tag 001

`noplan_form_v3.py` (M25) reads the form on the 397 LIRF rows without a plan of January and July
2025 and saves the reads, not the row values.  This script repeats its fold stages (2), (4) and (5)
with the same functions, tables and class-model inputs (every table and class model from the ten
other 2025 months) and writes one row per fold row:

    base      F on the row (v32 on these rows; the comparator of the fold reads), floor 60
    v2        the v2 form
    v3        the v3 form as registered (cell at the weight 0.1)
    cal       v3 with the cell weight of the probability bucket (0.1 up to p 0.3, 0.25 above)
    nocell    v3 without the cell (the cell rows at the r1 value)
    cellmix   v3 with the cell rows at the class model's mixture
    part      the part of v3 that gives the row its value

It holds no label.  The two class models (the v2 columns for the real point, the v3 columns for the
probability) are fitted again; the script stops unless every read equals the registered read of
`results/noplan-form-v3-001.json` and `results/noplan-form-v3-cal-001.json`, and unless the cell rows
equal `candidates/noplan-form-v3-cell-fold-001.parquet`.  Write-once.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

import common as C
import inbound_sign as I
import noac_candidate_r1 as R1
import noplan_form as NF
import noplan_form_v2 as V2
import noplan_form_v3 as V3
import noplan_form_v3_cal as CAL
import noplan_form_v3_variants as VV
import regime_signal as S

sys.path.insert(0, str(C.ROOT))
from prc import gscore as GS  # noqa: E402

TOL_G, TOL_ROW = 1e-6, 1e-6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out_rows, out_json = C.RESULTS / f"noplan-form-v3-fold-{a.tag}.parquet", C.RESULTS / f"noplan-form-v3-fold-{a.tag}.json"
    C.refuse_existing(out_rows, out_json)
    src_path, cal_path = C.RESULTS / "noplan-form-v3-001.json", C.RESULTS / "noplan-form-v3-cal-001.json"
    src, cal_rec = json.loads(src_path.read_text()), json.loads(cal_path.read_text())
    chosen, pass_cell = src["lines_read"]["class_model_of_v3"], bool(src["lines_read"]["c43_part_passes"])
    reg = src["two_fold"]

    t25 = C.raw_2025()
    share25 = V2.arrival_share(t25)
    ex25 = V3.extra_inputs(t25)
    pr10, all10 = V3.practice(t25, share25, R1.TEN)
    d25 = V2.add_v2(NF.classes(NF.noplan_rows(t25)), share25)
    d25_10 = V3.add_v3(d25, ex25, pr10, all10)
    x, ty = S.build()
    sg = I.inbound_sign(t25).set_index(C.ID)
    x["in_late"], x["in_lead"] = sg.in_late.reindex(x[C.ID]).to_numpy(), sg.in_lead.reindex(x[C.ID]).to_numpy()
    e25 = ex25.set_index(C.ID)
    for c in V3.C44 + V3.C43:
        x[c] = e25[c].reindex(x[C.ID]).to_numpy()

    # stage (2) of noplan_form_v3.py: tables and class models from the ten other months
    src10 = d25_10[d25_10.month.isin(R1.TEN)]
    tb10, rt10 = NF.tables(src10), V2.rule_tables(src10)
    tsrc = ty[ty.month.isin(R1.TEN)]
    tr = pd.concat([pd.concat([x[x.month == m], S.encodings(x[x.month == m], tsrc, m)], axis=1) for m in R1.TEN])
    te = x[x.month.isin((1, 7))]
    te = pd.concat([te, S.encodings(te, tsrc, 0)], axis=1)
    tid = te[C.ID].to_numpy()
    PF, MF = {}, None
    for k in dict.fromkeys(("v2", chosen)):
        p_, m_ = I.fit_score(tr, te, NF.COLS + V3.SETS[k])
        PF[k] = pd.Series(p_, index=tid)
        if k == "v2":
            MF = pd.Series(m_, index=tid)
    r = C.served_forms(C.fold_rows()).set_index(C.ID)
    ids = r.index.to_numpy("int64")
    y = GS.labels(ids)
    base = np.maximum(C.FLOOR, r.F.to_numpy("float64"))
    f = d25_10[d25_10[C.ID].isin(r.index)].reset_index(drop=True)
    pos = r.index.get_indexer(f[C.ID])
    bf = base[pos]

    def G(pred: np.ndarray, ref_rows: np.ndarray | None = None) -> float:
        b0, cand = base.copy(), base.copy()
        if ref_rows is not None:
            b0[pos] = ref_rows
        cand[pos] = pred
        return float(GS.score(ids, b0, cand, y=y)["G"])

    ms_te = te.set_index(C.ID).ms
    mix_v2 = pd.Series(np.maximum(C.FLOOR, PF["v2"].to_numpy() * ms_te.reindex(tid).to_numpy() + (1 - PF["v2"].to_numpy()) * MF.to_numpy()), index=tid)
    v2 = V2.form(f, bf, mix_v2, tb10, rt10, bet=False)[0]
    pf, mf = PF[chosen], MF
    v3, part = V3.form(f, bf, pf, mf, tb10, cell=pass_cell)
    nocell, _ = V3.form(f, bf, pf, mf, tb10, cell=False)
    ms, pp, mr = f.ms.to_numpy(), pf.reindex(f[C.ID]).to_numpy(), mf.reindex(f[C.ID]).to_numpy()
    cell = part == "c43"
    w = np.where(pp <= VV.P_SPLIT, CAL.W_LOW, CAL.W_HIGH)
    cal = np.where(cell, np.maximum(C.FLOOR, mr + w * (ms - mr)), v3)
    cellmix = np.where(cell, np.maximum(C.FLOOR, pp * ms + (1 - pp) * mr), v3)

    reads = {"v2_vs_F": G(v2), "v3_vs_F": G(v3), "v3_over_v2": G(v3, v2), "v3_without_cell_over_v2": G(nocell, v2), "cell_over_v3_without_cell": G(v3, nocell),
             "cal_vs_F": G(cal), "cal_over_v3_without_cell": G(cal, nocell), "nocell_vs_F": G(nocell), "cellmix_vs_F": G(cellmix), "cellmix_over_v3_without_cell": G(cellmix, nocell)}
    want = {k: reg[k]["G"] for k in ("v2_vs_F", "v3_vs_F", "v3_over_v2", "v3_without_cell_over_v2", "cell_over_v3_without_cell")}
    want["cal_over_v3_without_cell"] = cal_rec["fold_cell_rows"]["G_cal_over_form_without_cell"]
    bad = {k: (reads[k], v) for k, v in want.items() if abs(reads[k] - v) > TOL_G}
    parts = {p_: int((part == p_).sum()) for p_ in V3.PARTS}
    if bad or len(f) != reg["rows"] or parts != {k: v["rows"] for k, v in reg["by_part"].items()}:
        raise SystemExit(f"the rebuilt fold form is not the registered one: reads {bad}; rows {len(f)}; parts {parts}")
    cf = pd.read_parquet(CAL.FOLD_ROWS).set_index(C.ID)
    fi = pd.Index(f[C.ID].to_numpy("int64"))
    cpos = fi.get_indexer(cf.index.to_numpy("int64"))
    if (cpos < 0).any() or set(cpos) != set(np.flatnonzero(cell)):
        raise SystemExit("the cell rows are not the rows of the saved cell file")
    for col, val in (("p", pp), ("m_real", mr), ("r1_value", nocell), ("rule", v3), ("F", bf)):
        if not np.allclose(cf[col].to_numpy(), val[cpos], rtol=0, atol=TOL_ROW):
            raise SystemExit(f"the cell rows differ from the saved cell file in `{col}`")
    print({k: round(v, 4) for k, v in reads.items()}, "\nparts:", parts, flush=True)

    out = pd.DataFrame({C.ID: f[C.ID].to_numpy("int64"), "month": f.month.to_numpy(), "FLIGHT_mvt": f.FLIGHT_mvt.to_numpy(), "ms": ms, "p": pp, "m_real": mr, "part": part,
                        "base": bf, "v2": v2, "v3": v3, "cal": cal, "nocell": nocell, "cellmix": cellmix})
    val_cols = ["base", "v2", "v3", "cal", "nocell", "cellmix"]
    if out[C.ID].duplicated().any() or not np.isfinite(out[val_cols].to_numpy()).all() or (out[val_cols].to_numpy() < C.FLOOR).any():
        raise SystemExit("a value is not finite and at or above the floor, or an id repeats")
    C.refuse_existing(out_rows, out_json)
    out.to_parquet(out_rows, index=False)
    back = pd.read_parquet(out_rows)
    if not back.equals(out):
        raise SystemExit("the file does not read back as written")
    rec = {"what": "exp-167: the fold values of the v3 no-plan form and its cell variations on the 397 LIRF rows without a plan of January and July 2025 (no label in the file)",
           "source_results": {str(p.relative_to(C.ROOT)): C.sha(p) for p in (src_path, cal_path)}, "cell_fold_rows_sha256": C.sha(CAL.FOLD_ROWS), "class_model_of_v3": chosen, "cell": pass_cell,
           "rows": int(len(out)), "months": {int(k): int(v) for k, v in out.month.value_counts().sort_index().items()}, "parts": parts, "columns": list(out.columns),
           "G_vs_F_or_as_named": reads, "registered_reads_checked": want, "tolerance_G": TOL_G,
           "rows_differ": {k: int((out[k] != out.base).sum()) for k in val_cols[1:]}, "rows_differ_from_v3": {k: int((out[k] != out.v3).sum()) for k in ("cal", "nocell", "cellmix")},
           "rows_file": {"file": str(out_rows.relative_to(C.ROOT)), "sha256": C.sha(out_rows)},
           "code_sha256": {"noplan_form_v3_fold_export.py": C.sha(__file__), "noplan_form_v3.py": C.sha(V3.__file__), "noplan_form_v3_cal.py": C.sha(CAL.__file__), "common.py": C.sha(C.__file__)}}
    C.write_json(out_json, rec)
    print("rows file", rec["rows_file"], flush=True)


if __name__ == "__main__":
    main()
