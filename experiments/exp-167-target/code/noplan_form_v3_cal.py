"""M25d (unit 11): the C43 cell of the v3 form with the weight of the row's probability bucket.

    .venv/bin/python experiments/exp-167-target/code/noplan_form_v3_cal.py --tag 001
    .venv/bin/python experiments/exp-167-target/code/noplan_form_v3_cal.py --tag 002

`fresh`'s rule gives every cell row the weight 0.1 on the placeholder point.  In 2025 the
placeholder rate of the cell differs by the class model's probability p (`fresh`'s bucket edges,
`experiments/exp-168-fresh-look/checks/u4_held2.out`, and M25b):

    p at most 0.3            fold 0 of 32, ten other months 4 of 79 (v2 model); with v3's model the
                             fold has 0 of 31 and the ten months have 1 placeholder at or under 0.3
    p above 0.3, up to 0.7   3 of 14 (v2 model); 4 of about 16 (v3's model)

`cal` keeps 0.1 for the first bucket and takes W_HIGH = 0.25 for the second (the measured rate,
not a search).  The script reads the rows file of `noplan_form_v3_variants.py` and the v3 entry,
writes one more entry that differs from the v3 entry on the cell rows with p above 0.3, and
reads the same value on the fold cell rows of M25b (`candidates/noplan-form-v3-cell-fold-001.parquet`).
Write-once.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import common as C
import inbound_sign as I
import noac_candidate_r1 as R1
import noplan_form as NF
import noplan_form_v2 as V2
import noplan_form_v3 as V3
import noplan_form_v3_variants as VV
import regime_signal as S

W_LOW, W_HIGH = 0.10, 0.25
FOLD_ROWS = C.EXP / "candidates/noplan-form-v3-cell-fold-001.parquet"


def ten_months(chosen: str) -> dict:
    """The cell rows of the ten other months with the class model of each month held (as `noplan_form_v3.py` stage 3): buckets and SSE."""
    t25 = C.raw_2025()
    share25 = V2.arrival_share(t25)
    ex25 = V3.extra_inputs(t25)
    pr12, all12 = V3.practice(t25, share25, range(1, 13))
    d25 = V2.add_v2(NF.classes(NF.noplan_rows(t25)), share25)
    d10 = V3.add_v3(d25, ex25, pr12, all12)
    d10 = d10[d10.month.isin(R1.TEN)].reset_index(drop=True)
    tb12 = NF.tables(d25)
    x, ty = S.build()
    sg = I.inbound_sign(t25).set_index(C.ID)
    x["in_late"], x["in_lead"] = sg.in_late.reindex(x[C.ID]).to_numpy(), sg.in_lead.reindex(x[C.ID]).to_numpy()
    e25 = ex25.set_index(C.ID)
    for c in V3.C44 + V3.C43:
        x[c] = e25[c].reindex(x[C.ID]).to_numpy()
    xid = x[C.ID].to_numpy()
    P_l, M_l = np.full(len(x), np.nan), np.full(len(x), np.nan)
    for mth in R1.TEN:
        z = pd.concat([x, S.encodings(x, ty, mth)], axis=1)
        tr, te = z[z.month != mth], z[z.month == mth]
        p_, m_ = I.fit_score(tr, te, NF.COLS + V3.SETS[chosen])
        m2_ = I.fit_score(tr, te, NF.COLS)[1] if chosen != "v2" else m_
        P_l[(z.month == mth).to_numpy()], M_l[(z.month == mth).to_numpy()] = p_, m2_
        print("month", mth, "held", flush=True)
    pl, ml = pd.Series(P_l, index=xid), pd.Series(M_l, index=xid)
    _, part = V3.form(d10, np.zeros(len(d10)), pl, ml, tb12, cell=True)
    cl = d10[part == "c43"].copy()
    cl["p"], cl["mr"] = pl.reindex(cl[C.ID]).to_numpy(), ml.reindex(cl[C.ID]).to_numpy()
    n = len(C.fold_rows())
    w = np.where(cl.p <= VV.P_SPLIT, W_LOW, W_HIGH)
    val = {"rule": np.maximum(C.FLOOR, cl.mr + W_LOW * (cl.ms - cl.mr)), "cal": np.maximum(C.FLOOR, cl.mr + w * (cl.ms - cl.mr)), "mixture": np.maximum(C.FLOOR, cl.p * cl.ms + (1 - cl.p) * cl.mr)}
    sse = {k: float(((cl.y - v) ** 2).sum() / n) for k, v in val.items()}
    hi = cl.p > VV.P_SPLIT
    return {"rows": int(len(cl)), "placeholders": int((cl.cls == "P").sum()), "rows_with_p_at_most_split": int((~hi).sum()), "placeholders_with_p_at_most_split": int((~hi & (cl.cls == "P")).sum()),
            "rows_with_p_above_split": int(hi.sum()), "placeholders_with_p_above_split": int((hi & (cl.cls == "P")).sum()), "sse_per_fold_row": sse,
            "G_cal_over_rule": sse["rule"] - sse["cal"], "G_cal_over_mixture": sse["mixture"] - sse["cal"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--buckets", action="store_true", help="also count the cell rows of the ten other months by bucket, with each month held (ten fits)")
    a = ap.parse_args()
    out_json, out_entry = C.RESULTS / f"noplan-form-v3-cal-{a.tag}.json", C.SAVED / f"noplan-form-v3-{a.tag}-cal-entry.parquet"
    out_rows = C.EXP / f"candidates/noplan-form-v3-cal-{a.tag}.parquet"
    C.refuse_existing(out_json, out_entry, out_rows)
    src = json.loads((C.RESULTS / f"noplan-form-v3-{a.tag}.json").read_text())
    var = json.loads((C.RESULTS / f"noplan-form-v3-variants-{a.tag}.json").read_text())
    rows_path, entry_path = C.ROOT / var["rows_file"]["file"], C.ROOT / src["entry"]["file"]
    if C.sha(rows_path) != var["rows_file"]["sha256"] or C.sha(entry_path) != src["entry"]["sha256"]:
        raise SystemExit("the rows file or the entry is not the file of the result")
    d = pd.read_parquet(rows_path)
    cell = (d.part == "c43").to_numpy()
    w = np.where(d.p.to_numpy() <= VV.P_SPLIT, W_LOW, W_HIGH)
    cal = np.maximum(C.FLOOR, d.m_real + w * (d.ms - d.m_real)).to_numpy()
    d["cal"] = np.where(cell, cal, d.candidate)
    if not np.allclose(d.cal[cell & (d.p <= VV.P_SPLIT).to_numpy()], d.candidate[cell & (d.p <= VV.P_SPLIT).to_numpy()], rtol=0, atol=1e-6):
        raise SystemExit("the value of the first bucket is not the rule's value")

    # the fold cell rows of M25b
    f = pd.read_parquet(FOLD_ROWS)
    wf = np.where(f.p <= VV.P_SPLIT, W_LOW, W_HIGH)
    f["cal"] = np.maximum(C.FLOOR, f.m_real + wf * (f.ms - f.m_real))
    g = lambda v: ((f.y - f.r1_value) ** 2 - (f.y - v) ** 2) * f.w26  # noqa: E731
    fold = {"rows": int(len(f)), "rows_with_p_above_split": int((f.p > VV.P_SPLIT).sum()), "placeholders_with_p_above_split": int(((f.p > VV.P_SPLIT) & (f.cls == "P")).sum()),
            "placeholders_with_p_at_most_split": int(((f.p <= VV.P_SPLIT) & (f.cls == "P")).sum()), "G_rule_over_form_without_cell": float(g(f.rule).sum()), "G_cal_over_form_without_cell": float(g(f.cal).sum()),
            "by_month": {int(k): {"rule": float(g(f.rule)[f.month == k].sum()), "cal": float(g(f.cal)[f.month == k].sum())} for k in (1, 7)},
            "without_3_largest_gains": {"rule": float(np.sort(g(f.rule).to_numpy())[:-3].sum()), "cal": float(np.sort(g(f.cal).to_numpy())[:-3].sum())}}
    print("fold cell rows:", fold, flush=True)
    ten = ten_months(src["lines_read"]["class_model_of_v3"]) if a.buckets else None
    if ten is not None:
        print("ten other months, each held:", ten, flush=True)

    c = d[cell]
    P, R, pc, high = c.ms.to_numpy(), c.m_real.to_numpy(), c.p.to_numpy(), (c.p > VV.P_SPLIT).to_numpy()
    val = {"v3 (rule 0.1)": c.candidate.to_numpy(), "cal": c.cal.to_numpy(), "cellmix": c.cellmix.to_numpy(), "nocell": c.nocell.to_numpy()}

    def gain(old: np.ndarray, new: np.ndarray, pi: np.ndarray) -> float:
        return float((pi * ((P - old) ** 2 - (P - new) ** 2) + (1 - pi) * ((R - old) ** 2 - (R - new) ** 2)).sum() / C.N_RANKING)

    tab = {}
    for ref in ("v2", "base"):
        old = c[ref].to_numpy()
        t = pd.DataFrame({k: {s: gain(old, v, pc if q is None else np.where(high, q[1], q[0])) for s, q in VV.SCENARIOS.items()} for k, v in val.items()}).T
        t["all real"], t["all placeholders"] = [gain(old, v, np.zeros(len(c))) for v in val.values()], [gain(old, v, np.ones(len(c))) for v in val.values()]
        t[f"distance from {ref}"] = [float(((v - old) ** 2).sum() / C.N_RANKING) for v in val.values()]
        tab[ref] = t
        print(f"expected gain over {ref} on the {len(c)} cell rows:\n", t.round(1).to_string(), flush=True)
    e = pd.read_parquet(entry_path)
    ids, ev = e.MVT_ID_mvt.to_numpy("float64").astype("int64"), e.TAXITIME_SEC_mvt.to_numpy("float64")
    pos = pd.Index(ids).get_indexer(d[C.ID].to_numpy("int64"))
    if (pos < 0).any() or not np.allclose(ev[pos], d.candidate.to_numpy(), rtol=0, atol=1e-9):
        raise SystemExit("the v3 entry does not hold the rows file's values")
    new = ev.copy()
    new[pos] = d.cal.to_numpy()
    if not np.isfinite(new).all() or (new < C.FLOOR).any():
        raise SystemExit("the entry is not finite and above the floor")
    C.refuse_existing(out_json, out_entry, out_rows)
    d[[C.ID, "month", "FLIGHT_mvt", "ms", "p", "m_real", "part", "base", "v2", "candidate", "cal", "cellmix", "nocell"]].to_parquet(out_rows, index=False)
    pd.DataFrame({"MVT_ID_mvt": e.MVT_ID_mvt.to_numpy(), "TAXITIME_SEC_mvt": new}).to_parquet(out_entry, index=False)
    back = pd.read_parquet(out_entry)
    ch = back.TAXITIME_SEC_mvt.to_numpy("float64") != ev
    allowed = set(c[C.ID].to_numpy("int64")[high])
    if len(back) != C.N_RANKING or not np.array_equal(back.MVT_ID_mvt.to_numpy(), e.MVT_ID_mvt.to_numpy()) or set(ids[ch]) - allowed:
        raise SystemExit("the entry differs from the v3 entry outside the cell rows with p above the split")
    rec = {"what": "exp-167 M25d: the C43 cell with the weight of the row's probability bucket (0.1 up to p 0.3, 0.25 above)", "w_low": W_LOW, "w_high": W_HIGH, "p_split": VV.P_SPLIT,
           "base_entry": src["base_entry"], "base_sha256": src["base_sha256"], "v3_entry_sha256": src["entry"]["sha256"], "fold_cell_rows": fold, "fold_rows_sha256": C.sha(FOLD_ROWS), "ten_other_months_each_held": ten,
           "expected_gain": {ref: t.to_dict("index") for ref, t in tab.items()}, "rows_file": {"file": str(out_rows.relative_to(C.ROOT)), "sha256": C.sha(out_rows)},
           "entry": {"file": str(out_entry.relative_to(C.ROOT)), "sha256": C.sha(out_entry), "rows": int(len(back)), "rows_changed_from_the_v3_entry": int(ch.sum()),
                     "distance_from_the_v3_entry": float(((back.TAXITIME_SEC_mvt.to_numpy("float64") - ev) ** 2).sum() / C.N_RANKING),
                     "rows_changed_from_base": int((back.TAXITIME_SEC_mvt.to_numpy("float64")[pos] != d.base.to_numpy()).sum()),
                     "distance_from_base": float(((back.TAXITIME_SEC_mvt.to_numpy("float64")[pos] - d.base.to_numpy()) ** 2).sum() / C.N_RANKING)},
           "code_sha256": {"noplan_form_v3_cal.py": C.sha(__file__), "noplan_form_v3_variants.py": C.sha(VV.__file__), "common.py": C.sha(C.__file__)}}
    print("entry", rec["entry"], flush=True)
    C.write_json(out_json, rec)


if __name__ == "__main__":
    main()
