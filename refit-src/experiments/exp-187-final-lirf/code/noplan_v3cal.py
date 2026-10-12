"""No-plan v3cal (exp-167 M25 + M25d) on any 2026 movement file: the value of every LIRF departure without a plan.

This is step (7) of `experiments/exp-167-target/code/noplan_form_v3.py` and the bucket weight of
`noplan_form_v3_cal.py`, with the movement file and the base value as inputs (the original reads `ranking.parquet`
and a 344,841-row entry). The 2025 side (tables, typed encodings, the class-model training frame) and the design
read from the registered records (class model `c44`, the C43 cell in the form) are unchanged. The exp-167 functions
are imported, not copied.

    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-187-final-lirf/code/noplan_v3cal.py --mode ranking --parity
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/exp-167-target/code"))
import common as C  # noqa: E402
import inbound_sign as I  # noqa: E402
import noac_candidate as N  # noqa: E402
import noplan_form as NF  # noqa: E402
import noplan_form_v2 as V2  # noqa: E402
import noplan_form_v3 as V3  # noqa: E402
import noplan_form_v3_variants as VV  # noqa: E402
import regime_signal as S  # noqa: E402

ID = C.ID
W_LOW, W_HIGH = 0.10, 0.25                      # noplan_form_v3_cal.py
V3_RECORD = ROOT / "experiments/exp-167-target/results/noplan-form-v3-002.json"
CAL_ROWS = ROOT / "experiments/exp-167-target/candidates/noplan-form-v3-cal-002.parquet"
CAL_ROWS_SHA = "af4ff0c40ad5f7f734fc55a838b96bb85f120fba503a99c5d364c0fbb723aee8"
BASE_ENTRY = ROOT / "submissions/candidates/exp159-fold-F-HB1-w50-x.parquet"
BASE_ENTRY_SHA = "a5572156bf548dd154eaab1448bf7eea5d173fbb5e233232c9670b9a0cbf10bd"


def threads(n: int) -> None:
    """The Mac cap is 2 threads per job; the original class model ran at n_jobs 4."""
    S.PARAMS["n_jobs"] = n


def design() -> tuple[str, bool]:
    rec = json.loads(V3_RECORD.read_text())
    return rec["class_model_of_v3"], bool(rec["c43_part_in_the_form"])


def fit_2025() -> dict:
    """The 2025 side of step (7): prefix practice share, tables, typed rows and the twelve-month training frame."""
    t25 = C.raw_2025()
    share25 = V2.arrival_share(t25)
    ex25 = V3.extra_inputs(t25)
    pr12, all12 = V3.practice(t25, share25, range(1, 13))
    d25 = V2.add_v2(NF.classes(NF.noplan_rows(t25)), share25)
    tb12 = NF.tables(d25)
    x, ty = S.build()
    sg = I.inbound_sign(t25).set_index(ID)
    x["in_late"], x["in_lead"] = sg.in_late.reindex(x[ID]).to_numpy(), sg.in_lead.reindex(x[ID]).to_numpy()
    e25 = ex25.set_index(ID)
    for c in V3.C44 + V3.C43:
        x[c] = e25[c].reindex(x[ID]).to_numpy()
    tr12 = pd.concat([pd.concat([x[x.month == m], S.encodings(x[x.month == m], ty, m)], axis=1) for m in range(1, 13)])
    return {"pr12": pr12, "all12": all12, "tb12": tb12, "ty": ty, "tr12": tr12}


def rows(t26: pd.DataFrame, fit: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The label-free inputs of the no-plan rows of a 2026 movement file, and the class-model test frame."""
    t26 = t26.copy()
    t26["apt"] = np.where(t26.PHASE_mvt == "DEP", t26.ADEP_mvt, t26.ADES_mvt)
    share26 = V2.arrival_share(t26)
    ex26 = V3.extra_inputs(t26)
    d26 = V3.add_v3(V2.add_v2(NF.noplan_rows(t26), share26), ex26, fit["pr12"], fit["all12"])
    x26, _ = N.features(t26[t26.apt == "LIRF"].copy())
    s26 = I.inbound_sign(t26).set_index(ID)
    x26["in_late"], x26["in_lead"] = s26.in_late.reindex(x26[ID]).to_numpy(), s26.in_lead.reindex(x26[ID]).to_numpy()
    e26 = ex26.set_index(ID)
    for c in V3.C44 + V3.C43:
        x26[c] = e26[c].reindex(x26[ID]).to_numpy()
    te26 = pd.concat([x26, S.encodings(x26, fit["ty"], 0)], axis=1)
    return d26, te26


def classes(te26: pd.DataFrame, fit: dict, chosen: str) -> tuple[pd.Series, pd.Series]:
    """p (placeholder probability, the chosen class model) and m_real (the real point) of the rows without type."""
    p2_, m2_ = I.fit_score(fit["tr12"], te26, NF.COLS)
    p3_, _ = I.fit_score(fit["tr12"], te26, NF.COLS + V3.SETS[chosen]) if chosen != "v2" else (p2_, None)
    i26 = te26[ID].to_numpy()
    return pd.Series(p3_, index=i26), pd.Series(m2_, index=i26)


def value(d26: pd.DataFrame, base: np.ndarray, p26: pd.Series, m26: pd.Series, fit: dict, cell: bool) -> pd.DataFrame:
    """v3 and v3cal of every no-plan row; `base` is the route value np_base on the rows of d26."""
    d = d26.copy()
    d["base"] = base
    d["candidate"], d["part"] = V3.form(d, base, p26, m26, fit["tb12"], cell=cell)
    d["p"], d["m_real"] = p26.reindex(d[ID]).to_numpy(), m26.reindex(d[ID]).to_numpy()
    is43 = (d.part == "c43").to_numpy()
    w = np.where(d.p.to_numpy() <= VV.P_SPLIT, W_LOW, W_HIGH)
    d["cal"] = np.where(is43, np.maximum(C.FLOOR, d.m_real + w * (d.ms - d.m_real)), d.candidate)
    return d


KEEP = [ID, "month", "FLIGHT_mvt", "ADES_mvt", "noac", "ms", "ku", "later", "far3", "zone", "state", "rule2", "rule39",
        "band", "dest_held", "p", "m_real", "part", "base", "candidate", "cal"]


def main():
    sys.path.insert(0, str(ROOT))
    from prc import data as D
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ranking", choices=["ranking", "final"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--parity", action="store_true", help="ranking mode: base = the exp159 entry; compare with cal-002")
    ap.add_argument("--base", type=Path, help="parquet MVT_ID_mvt, np_base on every no-plan row")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    threads(a.threads)
    chosen, cell = design()
    fit = fit_2025()
    t26 = pd.read_parquet(D.ranking_file(a.mode))
    d26, te26 = rows(t26, fit)
    p26, m26 = classes(te26, fit, chosen)
    if a.parity:
        assert a.mode == "ranking" and C.sha(BASE_ENTRY) == BASE_ENTRY_SHA
        e = pd.read_parquet(BASE_ENTRY)
        bv = pd.Series(e.TAXITIME_SEC_mvt.to_numpy("float64"), index=e.MVT_ID_mvt.to_numpy("float64").astype("int64"))
    else:
        b = pd.read_parquet(a.base)
        bv = pd.Series(b.np_base.to_numpy("float64"), index=b[ID].to_numpy("float64").astype("int64"))
    base = bv.reindex(d26[ID]).to_numpy()
    assert np.isfinite(base).all(), "a no-plan row has no base value"
    d = value(d26, base, p26, m26, fit, cell)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    d[KEEP].to_parquet(a.out, index=False)
    print("rows", len(d), d.groupby(["month", "part"]).size().unstack(fill_value=0).to_string(), flush=True)
    if a.parity:
        assert C.sha(CAL_ROWS) == CAL_ROWS_SHA
        ref = pd.read_parquet(CAL_ROWS).set_index(ID)
        got = d.set_index(ID)
        assert set(ref.index) == set(got.index), "row sets differ"
        got = got.reindex(ref.index)
        for k in ("p", "m_real", "candidate", "cal"):
            eq = int((got[k].to_numpy() == ref[k].to_numpy()).sum())
            print(f"parity {k}: equal {eq} of {len(ref)}; max abs diff {float(np.nanmax(np.abs(got[k].to_numpy() - ref[k].to_numpy()))):.3g}", flush=True)
        print("parity part: equal", int((got.part.to_numpy() == ref.part.to_numpy()).sum()), "of", len(ref), flush=True)


if __name__ == "__main__":
    main()
