"""M25c (unit 11): two variations of the C43 cell of the v3 form, as entries, with a forecast that needs no label.

    .venv/bin/python experiments/exp-167-target/code/noplan_form_v3_variants.py --tag 001
    .venv/bin/python experiments/exp-167-target/code/noplan_form_v3_variants.py --tag 002

The v3 form gives a cell row the value real point + 0.1 x (MVT - SCHED - real point) (`fresh`'s
rule).  The bug check M25b shows that the fold gain of the rule sits on rows whose class
probability is near the cell's limit of 0.7.  This script reads the rows file and the entry of
`noplan_form_v3.py` (or `noplan_form_v3_base.py`) for the tag and writes two more entries that
differ from the v3 entry on the cell rows only:

    cellmix   the class model's mixture: p x (MVT - SCHED) + (1 - p) x real point (no fixed weight)
    nocell    the r1 value: max(60, 0.75 base + 0.25 mixture) (v3 without the cell = v2 + C44 + C45)

Forecast without a label: a cell row is a placeholder with chance pi, else it sits at the real
point of the class model.  The table gives the expected gain of each value over the v2 value
and over the base for several pi, with separate chances for the rows with p at most P_SPLIT and
above it.  Write-once.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import common as C
import noplan_form as NF

P_SPLIT = 0.3
# (chance of a placeholder for a cell row with p <= P_SPLIT, for a row above it); None: the class model's own p
SCENARIOS = {"low 0.04, high 0.13": (0.04, 0.13), "low 0.04, high 0.21": (0.04, 0.21), "low 0.04, high 0.35": (0.04, 0.35), "low 0.04, high 0.50": (0.04, 0.50),
             "low 0.10, high 0.50": (0.10, 0.50), "all 0.05": (0.05, 0.05), "all 0.25": (0.25, 0.25), "the class model's p": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out_json, out_md = C.RESULTS / f"noplan-form-v3-variants-{a.tag}.json", C.RESULTS / f"noplan-form-v3-variants-{a.tag}.md"
    out_rows = C.EXP / f"candidates/noplan-form-v3-variants-{a.tag}.parquet"
    out_entry = {k: C.SAVED / f"noplan-form-v3-{a.tag}-{k}-entry.parquet" for k in ("cellmix", "nocell")}
    C.refuse_existing(out_json, out_md, out_rows, *out_entry.values())
    src_path = C.RESULTS / f"noplan-form-v3-{a.tag}.json"
    src = json.loads(src_path.read_text())
    rows_path, entry_path = C.ROOT / src["rows_file"]["file"], C.ROOT / src["entry"]["file"]
    if C.sha(rows_path) != src["rows_file"]["sha256"] or C.sha(entry_path) != src["entry"]["sha256"]:
        raise SystemExit("the rows file or the entry is not the file of the result")
    d = pd.read_parquet(rows_path)
    e = pd.read_parquet(entry_path)
    ms, p, mr, base = d.ms.to_numpy(), d.p.to_numpy(), d.m_real.to_numpy(), d.base.to_numpy()
    mix = np.maximum(C.FLOOR, p * ms + (1 - p) * mr)
    r1v = np.maximum(C.FLOOR, (1 - NF.W) * base + NF.W * mix)
    is_r1, cell = (d.part == "r1").to_numpy(), (d.part == "c43").to_numpy()
    if not np.allclose(r1v[is_r1], d.candidate.to_numpy()[is_r1], rtol=0, atol=1e-6):
        raise SystemExit("the r1 value from the rows file's columns is not the form's value")
    rule = np.maximum(C.FLOOR, mr + 0.10 * (ms - mr))
    if not cell.any() or not np.allclose(rule[cell], d.candidate.to_numpy()[cell], rtol=0, atol=1e-6):
        raise SystemExit("the rule value from the rows file's columns is not the form's value")
    d["cellmix"], d["nocell"] = np.where(cell, mix, d.candidate), np.where(cell, r1v, d.candidate)
    c = d[cell]
    P, R, pc, high = c.ms.to_numpy(), c.m_real.to_numpy(), c.p.to_numpy(), (c.p > P_SPLIT).to_numpy()
    val = {"v3 (rule 0.1)": c.candidate.to_numpy(), "cellmix": c.cellmix.to_numpy(), "nocell": c.nocell.to_numpy()}

    def gain(old: np.ndarray, new: np.ndarray, pi: np.ndarray) -> float:
        return float((pi * ((P - old) ** 2 - (P - new) ** 2) + (1 - pi) * ((R - old) ** 2 - (R - new) ** 2)).sum() / C.N_RANKING)

    tab = {}
    for ref in ("v2", "base"):
        old = c[ref].to_numpy()
        t = pd.DataFrame({k: {s: gain(old, v, pc if q is None else np.where(high, q[1], q[0])) for s, q in SCENARIOS.items()} for k, v in val.items()}).T
        t["all real"], t["all placeholders"] = [gain(old, v, np.zeros(len(c))) for v in val.values()], [gain(old, v, np.ones(len(c))) for v in val.values()]
        t[f"distance from {ref}"] = [float(((v - old) ** 2).sum() / C.N_RANKING) for v in val.values()]
        tab[ref] = t
        print(f"expected gain over {ref} on the {len(c)} cell rows (rows with p above {P_SPLIT}: {int(high.sum())}):\n", t.round(1).to_string(), flush=True)
    rec: dict = {"what": "exp-167 M25c: the C43 cell of the v3 form at the class model's mixture (cellmix) and without the cell (nocell)", "v3_result": str(src_path.relative_to(C.ROOT)),
                 "base_entry": src["base_entry"], "base_sha256": src["base_sha256"], "v3_entry_sha256": src["entry"]["sha256"], "p_split": P_SPLIT, "scenarios": {k: v for k, v in SCENARIOS.items()},
                 "cell_rows": int(cell.sum()), "cell_rows_with_p_above_split": int(high.sum()), "expected_gain": {ref: t.to_dict("index") for ref, t in tab.items()}}
    C.refuse_existing(out_json, out_md, out_rows, *out_entry.values())
    d[[C.ID, "month", "FLIGHT_mvt", "ms", "p", "m_real", "part", "base", "v2", "candidate", "cellmix", "nocell"]].to_parquet(out_rows, index=False)
    rec["rows_file"] = {"file": str(out_rows.relative_to(C.ROOT)), "sha256": C.sha(out_rows)}
    ids = e.MVT_ID_mvt.to_numpy("float64").astype("int64")
    ev = e.TAXITIME_SEC_mvt.to_numpy("float64")
    pos = pd.Index(ids).get_indexer(d[C.ID].to_numpy("int64"))
    if (pos < 0).any() or not np.allclose(ev[pos], d.candidate.to_numpy(), rtol=0, atol=1e-9):
        raise SystemExit("the v3 entry does not hold the rows file's values")
    rec["entries"] = {}
    for k, path in out_entry.items():
        new = ev.copy()
        new[pos] = d[k].to_numpy()
        if not np.isfinite(new).all() or (new < C.FLOOR).any():
            raise SystemExit("an entry is not finite and above the floor")
        pd.DataFrame({"MVT_ID_mvt": e.MVT_ID_mvt.to_numpy(), "TAXITIME_SEC_mvt": new}).to_parquet(path, index=False)
        back = pd.read_parquet(path)
        ch = back.TAXITIME_SEC_mvt.to_numpy("float64") != ev
        if len(back) != C.N_RANKING or not np.array_equal(back.MVT_ID_mvt.to_numpy(), e.MVT_ID_mvt.to_numpy()) or set(ids[ch]) - set(c[C.ID].to_numpy("int64")):
            raise SystemExit("an entry differs from the v3 entry outside the cell rows")
        rec["entries"][k] = {"file": str(path.relative_to(C.ROOT)), "sha256": C.sha(path), "rows": int(len(back)), "rows_changed_from_the_v3_entry": int(ch.sum()),
                             "distance_from_the_v3_entry": float(((back.TAXITIME_SEC_mvt.to_numpy("float64") - ev) ** 2).sum() / C.N_RANKING)}
        print(k, rec["entries"][k], flush=True)
    rec["code_sha256"] = {"noplan_form_v3_variants.py": C.sha(__file__), "noplan_form.py": C.sha(NF.__file__), "common.py": C.sha(C.__file__)}
    out_md.write_text("\n".join([f"# M25c: the C43 cell at the mixture and without the cell, base `{src['base_entry']}`", "",
                                 f"Expected gain over the v2 value on the {len(c)} cell rows of 2026 (s² per ranking row; {int(high.sum())} rows have p above {P_SPLIT}):", "", C.md(tab["v2"]), "",
                                 "Over the base entry:", "", C.md(tab["base"]), ""]) + "\n")
    C.write_json(out_json, rec)


if __name__ == "__main__":
    main()
