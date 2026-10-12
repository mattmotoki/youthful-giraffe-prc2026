"""The packet's own gates, run from the committed packet and nothing else.

Every claim in `RESULTS.md` that is arithmetic rather than measurement is
checked here, so a reader can re-run one command instead of trusting the
table.  Each check prints its own tolerance and the worst value it saw.

    python checks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import align as A  # noqa: E402
import vectors as V  # noqa: E402
import prc.data as D  # noqa: E402

FOLDS = A.FOLDS
OK, BAD = [], []


def check(name: str, ok: bool, detail: str) -> None:
    (OK if ok else BAD).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main() -> None:
    prov = json.loads((EXP / "analysis" / "q01-provenance.json").read_text())
    cols = json.loads((EXP / "packet" / "columns.json").read_text())["d_columns"]
    T = pd.read_csv(EXP / "analysis" / "q01-table.csv").set_index("candidate")
    Dt = pd.read_csv(EXP / "analysis" / "q01-dates.csv")

    print("\n1. input hashes")
    f = prov["files"]["v10_served"]
    check("v10 parquet sha256", f["sha256"] == f["expected"],
          f"{f['sha256'][:16]} == {f['expected'][:16]}")
    m = prov["files"]["mask"]
    # the LIVE file is hashed and compared, not just counted: an eight-airport
    # list with different airports passed the old check (Sol round 1, blocking 12)
    live = D.sha256(ROOT / m["path"])
    check("mask json sha256 (live file against the record)", live == m["sha256"],
          f"{live[:16]} == {m['sha256'][:16]}")
    check("mask airport list", len(m["keep"]) == 8 and m["keep"] == sorted(m["keep"]),
          f"keep = {','.join(m['keep'])}")
    for fold, e in prov["manifest"].items():
        p = ROOT / e["path"]
        check(f"packet {fold} sha256", D.sha256(p) == e["sha256"],
              f"{e['sha256'][:16]}, {e['rows']} rows")

    print("\n2. the join, the masks and the cancellation")
    worst_cancel = worst_add = worst_ident = 0.0
    worst_eham = worst_noac = 0.0
    for fold in FOLDS:
        d0 = pd.read_parquet(EXP / "packet" / f"rows-{fold}.parquet")
        ids = d0["MVT_ID_mvt"].to_numpy("float64")
        assert len(np.unique(ids)) == len(ids)
        r = d0["r"].to_numpy("float64")
        eham = d0["airport"].to_numpy() == "EHAM"
        noac = d0["regime"].to_numpy() == "noac"
        for col, name in cols.items():
            d = d0[col].to_numpy("float64")
            dsse = d * d - 2.0 * r * d
            # (a) a row the candidate does not move contributes exactly zero
            worst_cancel = max(worst_cancel, float(np.abs(dsse[d == 0.0]).max(initial=0.0)))
            # (b) the A/B algebra equals the row sum
            A_, B_ = float(np.mean(r * d)), float(np.mean(d * d))
            lhs, rhs = float(dsse.mean()), B_ - 2.0 * A_
            worst_add = max(worst_add, abs(lhs - rhs))
            # (c) the per-date table adds back to the fold total
            sub = Dt[(Dt["candidate"] == name) & (Dt["fold"] == fold)]
            if len(sub):
                worst_ident = max(worst_ident,
                                  abs(float(sub["dsse"].sum()) - float(dsse.sum()))
                                  / max(abs(float(dsse.sum())), 1.0))
            if name.startswith(("K2 (v9)", "k2-mask", "k2-w300", "bulk-", "k3-hold",
                                "k3g-hold", "refit-bulk", "stack-5", "v10", "exp-037")):
                if "no-EHAM-mask" not in name:
                    worst_eham = max(worst_eham, float(np.abs(d[eham]).max(initial=0.0)))
            worst_noac = max(worst_noac, float(np.abs(d[noac]).max(initial=0.0))
                             if not name.startswith("exp-037") else 0.0)
    check("unchanged rows cancel exactly", worst_cancel == 0.0,
          f"max |paired SSE| on a row with d = 0 is {worst_cancel:g} (tolerance 0)")
    check("2A - B equals the row sum", worst_add < 1e-6,
          f"max |mean(dsse) - (B - 2A)| = {worst_add:.3g} (tolerance 1e-6)")
    check("the per-date table adds back", worst_ident < 1e-10,
          f"max relative gap = {worst_ident:.3g} (tolerance 1e-10)")
    check("every EHAM-masked candidate holds EHAM at v10", worst_eham == 0.0,
          f"max |d| on an EHAM row = {worst_eham:g} (tolerance 0)")
    check("no candidate but exp-037's moves a noac row", worst_noac == 0.0,
          f"max |d| on a noac row = {worst_noac:g} (tolerance 0)")

    print("\n3. the identity row")
    v10 = T.loc["v10 (exp-036 refitbulk-mask)"]
    check("v10 against itself is exactly zero",
          v10["gain_board_allrow"] == 0.0 and v10["n_changed_july"] == 0
          and v10["gain_board_clean"] == 0.0,
          f"gain {v10['gain_board_allrow']:g}, rows changed {int(v10['n_changed_july'])}")

    print("\n4. the clean endpoint reproduces the record")
    # exp-036 arm 2 measured +0.1404 s of clean board over K2 and +0.0353 over refit-bulk
    for name, want in (("K2 (v9)", -0.1404), ("refit-bulk", -0.0353)):
        got = float(T.loc[name, "gain_board_clean"])
        check(f"clean board gain over {name}", abs(got - want) < 5e-4,
              f"{got:+.4f} against the record's {want:+.4f}")

    print("\n5. Q02's reproduction gates, ROW BY ROW")
    # q02.py gates on the rounded clean RMSE within 0.001, which two different
    # vectors can share.  Here the rebuilt arm is compared to the registry's own
    # vector row by row (Sol round 1, blocking 12).
    GATED = {"gate-k2-hold": "K2 (v9)",
             "gate-bulk-add-hold": "bulk-add-hold",
             "unchanged (v10 replay)": "v10 (exp-036 refitbulk-mask)",
             "unchanged (avail rule)": "v10 (exp-036 refitbulk-mask)"}
    ref, vec, _prov, _files = V.load(verbose=False)
    for arm, reg in GATED.items():
        worst = 0.0
        for f in V.FOLDS:
            q = pd.read_parquet(EXP / "packet" / f"q02-{f}.parquet")
            if arm not in q.columns:
                worst = float("nan")
                break
            want = vec[reg][f]
            got = q[arm].to_numpy("float64")
            ids_q = q["MVT_ID_mvt"].to_numpy("float64")
            ids_r = ref[f]["MVT_ID_mvt"].to_numpy("float64")
            if not np.array_equal(ids_q, ids_r):
                worst = float("nan")
                break
            worst = max(worst, float(np.abs(got - want).max()))
        # the tolerance is double-precision noise, not a slack: a five-seed bag
        # summed in a different order differs in the last bits and nowhere else.
        check(f"{arm} equals {reg} row by row", worst <= V.VEC_TOL,
              f"max |difference| = {worst:g} (tolerance {V.VEC_TOL:g})")

    print(f"\n{len(OK)} passed, {len(BAD)} failed")
    if BAD:
        raise SystemExit("failed: " + "; ".join(BAD))


if __name__ == "__main__":
    main()
