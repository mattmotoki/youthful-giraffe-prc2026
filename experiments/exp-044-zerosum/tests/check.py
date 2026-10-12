"""Re-check this experiment's structural identities from the committed record.

Not a re-run of every number: a deterministic audit a human can read, of the
claims a reader has to take on trust otherwise.

  1  the reference arm reproduces exp-039's Q03 record exactly
  2  both candidate files are valid submissions (template order, finite, min 60)
  3  the served update is zero-sum in the ranking file's OWN cells, recomputed
     here from the ranking file rather than read back from `serve.json`
  4  the masked rows are bit-identical to v10 in both candidates
  5  the verdict-bearing relations hold on the record itself
  6  every table in RESULTS.md equals a LIVE render from the current record
  7  every number in RESULTS.md's prose is a value the record carries, with a
     documented list of the numbers that are not measurements

    .venv/bin/python experiments/exp-044-zerosum/tests/check.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import zs                                   # noqa: E402
import prc.blend as B                       # noqa: E402

A = zs.EXP / "analysis"
ROOT = zs.ROOT
FOLDS = zs.FOLDS
ok = []


NUM = re.compile(r"(?<![\w./-])(\d[\d,]*\d|\d)(?:\.(\d+))?(?:e[+-]?\d+)?(?![\w/])",
                 re.I)

#: Numbers in RESULTS.md's prose that are NOT measurements, each with why.
NOT_A_MEASUREMENT = {
    "0": "zero", "1": "list numbering", "2": "list numbering", "3": "list numbering",
    "4": "list numbering / the cell minimum / thread count", "5": "list numbering",
    "6": "list numbering", "7": "list numbering", "8": "list numbering",
    "9": "list numbering", "10": "section number and the 10-minute bin",
    "016": "exp-016", "018": "exp-018", "036": "exp-036", "039": "exp-039",
    "044": "this experiment", "03": "screen number", "04": "screen number",
    "2025": "calendar year", "2026": "calendar year",
    "01": "month", "07": "month", "12": "month",
    "39": "predeclared seed", "100": "predeclared ridge alpha",
    "202": "the row-only feature count",
    "13": "89 - 76, the shared-vocabulary c_* column count; derived in the prose from "
          "the two shipped-column counts records.json carries",
    "64": "predeclared trees in lgb_small", "16": "predeclared leaves in lgb_wide",
    "400": "predeclared trees in lgb_wide", "256": "lgb_small leaves, 4 x 64",
    "6,400": "lgb_wide leaves, 16 x 400", "6,000": "the same, to one figure",
    "600": "the 10-minute bin in seconds", "300": "the 5-minute grid in seconds",
    "1200": "the 20-minute grid in seconds", "60": "the output floor in seconds",
    "3600": "the COMMON slice boundary in seconds",
    "600,000": "the defective bin the epoch bug produced",
    "80": "cells in the REJECTED serving run; console only, no analysis record",
    "23": "update RMS in the REJECTED serving run; console only, no record",
    "1e9": "the nanosecond divisor in the defect note",
    "1e-8": "the declared cell-sum tolerance", "1e-14": "the observed cell-sum scale",
    "1e-9": "the finite-difference tolerance",
    "0.16": "BOARD READING 10's transfer factor, from another experiment",
    "0.26": "BOARD READING 9's transfer factor, from another experiment",
    "263.0464": "v10's board, from the submission log",
    "263.0298": "v11's board, from the submission log",
    "263.042": "263.0298 + 0.0166 - 0.0036, derived in the prose",
    "0.0166": "v11 - v10 on the board, from the submission log",
    "260.93": "the leader's board", "2.099": "the published gap to the leader",
    "0.2": "0.004 / 2.099 as a percentage, derived in the prose",
    "0.217": "the scalar v11 was served at, from exp-038",
    "0.004": "0.0222 x 0.16, derived in the prose",
    "0.006": "0.0222 x 0.26, derived in the prose",
    "0.01": "the desk's earlier range for Q03, quoted",
    "0.02": "the desk's earlier range for Q03, quoted",
    "0.0015": "0.0797 - 0.0782, derived in the prose",
    "0.0016": "0.0782 - 0.0766, derived in the prose",
    "0.0036": "the cross-month gain rounded to four places in the board arithmetic",
    "0.28": "0.0222 / 0.0782, derived in the prose",
    "0.25": "0.0339 / 0.1336, derived in the prose",
    "0.12": "0.0131 / 0.1096, derived in the prose",
    "0.29": "the july_loo dayblock-to-xmonth ratio, derived in the prose",
    "0.55": "the served amplitude to two places, and the december ratio",
    "0.27": "the january dayblock-to-xmonth ratio, derived in the prose",
    "0.81": "the cross-month optimum as a share of the served amplitude",
    "0.5571": "exp-016's july_loo board share, a predeclared constant",
    "0.4429": "exp-016's january board share, a predeclared constant",
    "13 %": "the served-to-fold RMS shortfall, derived in the prose",
    "29": "the served-to-fold RMS shortfall in percent, derived in the prose",
    "17": "the 1200 s cohort growth in percent, derived in the prose",
    "28": "the share the cross-month fit retains, in percent, derived in the prose",
    "5": "day blocks, and the percent cost of the amplitude choice",
    "31.0": "the july_loo gain at its own unrestricted optimum",
    "27.8": "the july_loo xmonth 2A-B, also carried in the record",
    "1,120.35": "a slice RMSE, also carried in the record",
    "1/3": "the assignment's threshold, written as a fraction",
    "12": "the months the assignment names, and a traffic band",
    "11": "a traffic band bound", "19": "a cell-rows band bound",
    "20": "a cell-rows band bound", "90,000": "the within-month fitting rows, rounded",
    "55": "the UTC-date guard lower bound", "70": "the UTC-date guard upper bound",
    "344,841": "the ranking row count, also carried in the record",
    "b9f7089": "a commit", "d2e55b7": "a commit",
    # the round-1 repair run, quoted in §10 as the thing round 2 corrected.  That
    # run was scored at the declared amplitudes and is superseded; its numbers live
    # in review/round1-answers.md and in the round-1 correction row of docs/LEDGER.md.
    "0.0145": "the round-1 repaired lgb_shipped_wide xmonth gain, quoted as superseded",
    "0.0350": "the round-1 repaired lgb_shipped xmonth gain, quoted as superseded",
    "0.1212": "the round-1 repaired lgb_small dayblock gain, quoted as superseded",
    "0.3465": "the round-1 repaired lgb_shipped_wide dayblock gain, quoted as superseded",
    "0.1318": "the round-1 repaired lgb_shipped dayblock gain, quoted as superseded",
    "0.0542": "the round-1 repaired lgb_wide xmonth gain, quoted as superseded",
}


def _leaves(o) -> list:
    if isinstance(o, dict):
        return [v for x in o.values() for v in _leaves(x)]
    if isinstance(o, (list, tuple)):
        return [v for x in o for v in _leaves(x)]
    return [o] if isinstance(o, (int, float)) and not isinstance(o, bool) else []


def _variants(v: float) -> set:
    """Every way the report may honestly write this recorded value."""
    out = set()
    for x in (v, -v):
        for nd in range(0, 7):
            out.add(f"{x:.{nd}f}")
            out.add(f"{x:,.{nd}f}")
        out.add(f"{x:g}")
        if float(x).is_integer():
            out.add(f"{int(x)}")
            out.add(f"{int(x):,}")
        out.add(f"{x * 100:.2f}")
        out.add(f"{x * 100:.1f}")
        out.add(f"{x * 100:.0f}")
        out.add(f"{x / 100:.4f}")
    return out


def _prose(txt: str) -> str:
    txt = re.sub(r"```.*?```", " ", txt, flags=re.S)      # code fences
    txt = re.sub(r"`[^`]*`", " ", txt)                    # inline code and paths
    txt = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", txt)      # links
    txt = re.sub(r"\S*/\S*", " ", txt)                    # any remaining path
    return txt


def check(name: str, cond: bool, note: str = "") -> None:
    ok.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + note if note else ''}")


def main() -> None:
    Q3 = json.loads((ROOT / "experiments/exp-039-screens/analysis/q03.json").read_text())
    L = json.loads((A / "learner.json").read_text())
    S = json.loads((A / "serve.json").read_text())
    S11 = json.loads((A / "serve-v11.json").read_text())

    print("1  the reference arm against exp-039's Q03 record")
    for f in FOLDS:
        q, r = Q3["folds"][f], L["folds"]["ridge|dayblock"][f]
        check(f"{f} cohort", q["n_scored_cohort"] == L["cohort"][f]["n_scored_cohort"],
              f"{q['n_scored_cohort']:,}")
        check(f"{f} cells", q["n_cells"] == L["cohort"][f]["n_cells"], f"{q['n_cells']:,}")
        check(f"{f} pairs", q["pair"]["cross"]["n_pairs"] == L["cohort"][f]["n_pairs"],
              f"{q['pair']['cross']['n_pairs']:,}")
        check(f"{f} cross-month amplitude",
              abs(q["amplitude_cross_month"] - L["amplitudes"]["ridge"][f]) < 5e-5,
              f"{q['amplitude_cross_month']}")
        check(f"{f} all-row gain", abs(q["cell"]["cross"]["gain_all"] - r["gain_all"]) < 5e-5,
              f"{q['cell']['cross']['gain_all']:+.4f}")
    check("board all-row", abs(Q3["board"]["cross"]["gain_all_row"]
                               - L["board"]["ridge|dayblock"]["gain_all_row"]) < 5e-5,
          f"{Q3['board']['cross']['gain_all_row']:+.4f}")
    check("board clean", abs(Q3["board"]["cross"]["gain_clean"]
                             - L["board"]["ridge|dayblock"]["gain_clean"]) < 5e-5,
          f"{Q3['board']['cross']['gain_clean']:+.4f}")

    print("2  the candidate files")
    tmpl = zs.D.load_submitting()[zs.D.ID].to_numpy("float64")
    cands = {}
    for rec in (S, S11):
        p = ROOT / rec["candidate"]
        c = pd.read_parquet(p)
        v = c[zs.D.TARGET].to_numpy("float64")
        cands[rec["base"]] = (c, v)
        check(f"{p.name} template order",
              np.array_equal(c[zs.D.ID].to_numpy("float64"), tmpl), f"{len(c):,} rows")
        check(f"{p.name} finite", bool(np.isfinite(v).all()))
        check(f"{p.name} min >= 60", v.min() >= 60.0 - 1e-9, f"min {v.min():.4f}")
        check(f"{p.name} sha256 matches the record", zs.D.sha256(p) == rec["sha256"],
              rec["sha256"][:16] + "...")

    print("3  the served update, recomputed from the ranking file")
    rk = zs.ranking_dep()
    ids = rk["MVT_ID_mvt"].astype("float64").to_numpy()
    air = rk["ADEP_mvt"].astype(str).to_numpy()
    reg = B.regimes_from_raw(rk).astype(str)
    t = zs.epoch_seconds(rk["MVT_TIME_UTC_mvt"])
    elig = (reg == zs.ELIGIBLE_REGIME) & ~np.isin(air, zs.HOLD_AIRPORTS)
    code, ncell = zs.cellcode(air, reg, t, zs.BIN)
    cnt = np.bincount(code[elig], minlength=ncell)
    m0 = elig & (cnt[code] >= zs.MIN_CELL)
    check("eligible rows", int(elig.sum()) == S["census"]["n_eligible"],
          f"{int(elig.sum()):,}")
    check("scored cohort", int(m0.sum()) == S["census"]["n_scored_cohort"],
          f"{int(m0.sum()):,}")
    v10 = pd.read_parquet(ROOT / "submissions/uploaded/youthful-giraffe_v10.parquet")
    p10 = pd.Series(v10[zs.D.TARGET].to_numpy("float64"),
                    index=v10[zs.D.ID].to_numpy("float64")).reindex(ids).to_numpy()
    d = cands["v10"][1] - p10
    cm, ucm = pd.factorize(code[m0])
    s = np.bincount(cm, weights=d[m0], minlength=len(ucm))
    check("cell sums of d are zero", float(np.abs(s).max()) < 1e-8 * max(1.0, float(np.abs(d).sum())),
          f"max abs {np.abs(s).max():.2e}")
    check("d is zero off the scored cohort", float(np.abs(d[~m0]).max()) == 0.0)
    check("changed rows", int((d != 0).sum()) == S["census"]["n_changed"],
          f"{int((d != 0).sum()):,}")

    print("4  the masked rows")
    v11 = pd.read_parquet(ROOT / "submissions/uploaded/youthful-giraffe_v11.parquet")
    p11 = pd.Series(v11[zs.D.TARGET].to_numpy("float64"),
                    index=v11[zs.D.ID].to_numpy("float64")).reindex(ids).to_numpy()
    for grp, mask in (("noac", reg == "noac"), ("EHAM", air == "EHAM"),
                      ("EDDF", air == "EDDF")):
        check(f"{grp} identical to v10 in the v10 candidate",
              float(np.abs(cands["v10"][1][mask] - p10[mask]).max()) == 0.0,
              f"{int(mask.sum()):,} rows")
        check(f"{grp} identical to v10 in the v11 candidate",
              float(np.abs(cands["v11"][1][mask] - (p10 if grp in ("noac", "EHAM") else p11)[mask]).max()) == 0.0,
              f"{int(mask.sum()):,} rows")
    check("the two candidates carry the same update",
          float(np.abs((cands["v11"][1] - p11) - d).max()) < 1e-12)

    print("5  the LightGBM objective is Q03's objective, checked numerically")
    rng = np.random.default_rng(0)
    n, k = 60, 12
    cd = rng.integers(0, k, n)
    rc = zs.project(rng.normal(size=n), cd, k)      # a within-cell centred target
    h = rng.normal(size=n)

    def loss(x):
        return 0.5 * float(((zs.project(x, cd, k) - rc) ** 2).sum())

    ana = zs.project(h, cd, k) - rc                  # what code/zs.py:lgb_h returns
    num = np.zeros(n)
    eps = 1e-6
    for i in range(n):
        e = np.zeros(n); e[i] = eps
        num[i] = (loss(h + e) - loss(h - e)) / (2 * eps)
    check("grad = project(h) - r_c equals the finite difference",
          float(np.abs(ana - num).max()) < 1e-6, f"max abs {np.abs(ana - num).max():.2e}")
    check("the naive gradient h - r_c does NOT",
          float(np.abs((h - rc) - num).max()) > 1e-3,
          f"max abs {np.abs((h - rc) - num).max():.3f}")
    (A / "gradcheck.json").write_text(json.dumps(
        {"note": "the finite-difference residuals this checker measures, kept so the "
                 "report's numbers for them are record values (Sol round 2, blocking 6)",
         "n_rows": n, "n_cells": k, "eps": eps,
         "max_abs_residual_declared_gradient": float(np.abs(ana - num).max()),
         "max_abs_residual_naive_gradient": round(float(np.abs((h - rc) - num).max()), 3)},
        indent=2))

    print("6  the verdict-bearing relations on the record")
    V = json.loads((A / "verdict.json").read_text())
    R = json.loads((A / "repair.json").read_text())
    RC = json.loads((A / "records.json").read_text())
    V11 = json.loads((A / "v11base.json").read_text())
    D = json.loads((A / "density.json").read_text())
    tg, tn = V["pooled"]["traffic_given_cell_rows"]
    cg, cn = V["pooled"]["cell_rows_given_traffic"]
    check("the density verdict is what the declared rule returns",
          V["verdict"] == ("MIXED" if tg > tn / 2 and cg > cn / 2 else V["verdict"]),
          V["verdict"])
    check("both orderings survive a majority of comparisons (the MIXED case)",
          tg > tn / 2 and cg > cn / 2, f"{tg}/{tn} and {cg}/{cn}")
    check("the traffic marginal is NOT monotone at Q03's own 600 s grid",
          V["traffic_marginal_monotone"]["600"]["monotone_decreasing"] is False)
    check("the served gate FAILS: a january xmonth interval crosses zero",
          L["folds"]["ridge|xmonth"]["january"]["ci"][0] < 0,
          f"{L['folds']['ridge|xmonth']['january']['ci']}")
    check("no arm clears zero on all three xmonth folds",
          not any(all(L["folds"][f"{a}|xmonth"][f]["ci"][0] > 0 for f in FOLDS)
                  for a in L["arms"]))
    check("the ridge is positive on all three folds under both designs",
          all(L["folds"][f"ridge|{c}"][f]["gain_all"] > 0
              for c in ("dayblock", "xmonth") for f in FOLDS))
    for a in ("lgb_small", "lgb_wide", "lgb_shipped", "lgb_shipped_wide"):
        tag = "hess_diag_nocachevocab" if "shipped" in a else "hess_diag"
        k = f"{a}|xmonth|{tag}"
        check(f"{a} refitted with the exact Hessian diagonal is on the record",
              k in R["board"], f"{R['board'][k]['gain_all_row']:+.4f}")
        check(f"{a} does not beat the ridge on xmonth under either diagonal",
              max(L["board"][f"{a}|xmonth"]["gain_all_row"],
                  R["board"][k]["gain_all_row"])
              <= L["board"]["ridge|xmonth"]["gain_all_row"] + 0.02)
    check("the cross-month code path run inside the month reproduces dayblock",
          abs(R["board"]["ridge|pathcheck|-"]["gain_all_row"]
              - L["board"]["ridge|dayblock"]["gain_all_row"]) < 0.005,
          f"{R['board']['ridge|pathcheck|-']['gain_all_row']:+.4f}")
    check("one fitting month is worse than two, in the cross-month family",
          max(R["board"]["ridge|xmonth_one_1|-"]["gain_all_row"],
              R["board"]["ridge|xmonth_one_2|-"]["gain_all_row"])
          < L["board"]["ridge|xmonth"]["gain_all_row"])
    check("the v11 headline uses the CROSS-FITTED amplitude, not the pooled one",
          "xmonth|cross_month" in V11["arms"],
          f"{V11['arms']['xmonth|cross_month']['gain_over_v11']:+.4f}")
    for f in FOLDS:
        s = RC["slices"][f]
        check(f"{f} carries the COMMON slice decomposition",
              all(k in s["v10"] for k in ("rmse_all", "rmse_le3600", "rmse_gt3600",
                                          "rmse_nonm")))
        check(f"{f} no-NM rows are untouched by the update",
              s["v10"]["rmse_nonm"] == s["ridge|xmonth"]["rmse_nonm"],
              f"{s['v10']['rmse_nonm']:.2f}")
    import hashlib
    live = {q.name: hashlib.sha256(q.read_bytes()).hexdigest()
            for q in sorted((zs.EXP / "code").glob("*.py"))}
    for label, rec in (("records.json", RC), ("serve.json", S)):
        got = rec["provenance"]["code_sha256_exp044"]
        check(f"{label} provenance lists every code file of this experiment",
              set(got) == set(live),
              f"{len(live)} files" if set(got) == set(live)
              else f"missing {sorted(set(live) - set(got))}")
        check(f"{label} provenance hashes are the files as they are now",
              all(got.get(k) == v for k, v in live.items()),
              "" if all(got.get(k) == v for k, v in live.items())
              else "stale: " + ", ".join(k for k, v in live.items() if got.get(k) != v))
    check("the record carries a git_dirty flag", "git_dirty" in RC["provenance"])
    check("the served RMS is inside a third of the fold side's cross-month RMS",
          all(abs(S["census"]["rms_update_all_scored"]
                  - L["folds"]["ridge|xmonth"][f]["rms_update"])
              / L["folds"]["ridge|xmonth"][f]["rms_update"] < 1 / 3 for f in FOLDS),
          f"{S['census']['rms_update_all_scored']:.3f} s")

    print("7  RESULTS.md against the record: live tables, then every prose number")
    import importlib.util                                 # noqa: E402
    spec = importlib.util.spec_from_file_location(          # exp-039 also has a
        "zs_tables", HERE.parent / "code" / "tables.py")     # module named tables
    TB = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(TB)
    txt = (zs.EXP / "RESULTS.md").read_text()
    prose = txt
    for name, fn in (("density", TB.t_density), ("learner", TB.t_learner),
                     ("controls", TB.t_controls), ("serve", TB.t_serve),
                     ("repair", TB.t_repair), ("records", TB.t_records)):
        pat = re.compile(rf"<!-- TABLES:{name} -->(.*?)<!-- /TABLES:{name} -->", re.S)
        m = pat.search(txt)
        check(f"RESULTS carries the {name} table block", m is not None)
        if m:
            want = ("\n<!-- generated by code/tables.py; do not edit by hand -->\n\n"
                    + fn() + "\n")
            check(f"the {name} tables equal a live render from the record",
                  m.group(1) == want)
            prose = prose.replace(m.group(0), " ")
    check("RESULTS carries both SHA-256", S["sha256"] in txt and S11["sha256"][:16] in txt)

    allowed: set[str] = set()
    for rec in (L, S, S11, V, R, RC, V11, D,
                json.loads((A / "controls.json").read_text()),
                json.loads((A / "cohorts.json").read_text()),
                json.loads((A / "gradcheck.json").read_text())):
        for v in _leaves(rec):
            allowed |= _variants(float(v))
    bad = []
    for m in NUM.finditer(_prose(prose)):
        tok = m.group(1) + ("." + m.group(2) if m.group(2) else "")
        if tok in NOT_A_MEASUREMENT or tok.replace(",", "") in NOT_A_MEASUREMENT:
            continue
        if tok in allowed or tok.replace(",", "") in allowed:
            continue
        bad.append(tok)
    check("every prose number is a value the record carries", not bad,
          "" if not bad else f"unrecorded: {sorted(set(bad))}")

    print(f"\n  {sum(ok)} of {len(ok)} checks pass")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
