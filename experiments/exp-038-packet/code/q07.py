"""Q07 stage 1 -- the cached partial-sensitivity audit on Q01's finalists.

Q07's first test, in its own words: "First use existing exp-031 baseline/
candidate vectors to compute the CHANGE in paired loss difference under partial
nesting, by date/group/month, and map every supervised ancestor.  Do not
transport a historical sensitivity number to a new candidate."

So nothing here is refitted and no historical 0.1119 s is reused.  The change is
recomputed, on this packet's own endpoint, for the five candidates exp-031
actually carries -- `stack-5`, `v6`, `v7`, `K2`, `refit-bulk` -- against a v10
rebuilt on each instrument by the SAME `np.where` mask arithmetic that builds it
on the published one.  Three instruments:

    published      the committed rotation Q01 is measured on
    nested         exp-031 `oof/exp-031-nested-{fold}`: the member channel closed
                   for three of seven members, shipped anchor
    nestedanchor   exp-031 `oof/exp-031-nestedanchor-nested-{fold}`: the same
                   with the anchor channel closed as well

`exp-031` RESULTS section 1 is the standing limit and is restated in the output:
the design confounds closing a channel with removing a month of training data,
the member shifts include negative values, and nothing here is monotone in the
direction of the clean score.  A number that survives is not proved clean; a
number that moves is not proved leaked.

    PRC_THREADS=4 python q07.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import align as A  # noqa: E402
import vectors as V  # noqa: E402

ROOT = V.ROOT
FOLDS = V.FOLDS
E31 = ROOT / "experiments/exp-031-nested/oof"
INSTR = {"nested": "exp-031-nested-{fold}.parquet",
         "nestedanchor": "exp-031-nestedanchor-nested-{fold}.parquet"}
COLS = {"stack-5": "y_pred_stack_5", "v6": "y_pred_v6", "v7": "y_pred_v7",
        "K2 (v9)": "y_pred_K2", "refit-bulk": "y_pred_refit_bulk"}

#: The supervised ancestors of every candidate in this table, and which of them
#: each instrument closes.  `exp-031` RESULTS sections 1 and 3.
ANCESTORS = {
    "members": {
        "T-tailv4": "CLOSED on nested and nestedanchor (exp-025 oof-nested/)",
        "L2-hub8725": "CLOSED on nested and nestedanchor (exp-025 oof-nested/)",
        "F1-fresh": "CLOSED on nested and nestedanchor (exp-025 oof-nested/). It is in "
                    "blend-5's seven, so it is a DIRECT stacker feature of stack-5, v6 "
                    "and v7 only -- K2-window holds its slot in K2, refit-bulk and v10. "
                    "It still REACHES all six: K2 and refit-bulk are anchored on the "
                    "blend-5 hybrid, which is fitted over NAMES5 and therefore over "
                    "F1-fresh (exp-031 RESULTS section 3), and v10 additionally serves "
                    "v7 -- an F1-fresh descendant -- on every airport outside the "
                    "eight-airport keep list",
        "C2-cat": "open on every instrument (never in exp-025's scope)",
        "D1-winter": "open on every instrument (never in exp-025's scope)",
        "N1-nmtail": "open on every instrument (never in exp-025's scope)",
        "K2-window": "open on every instrument (never in exp-025's scope); this is "
                     "the member K2/refit-bulk/v10 turn on",
        "V3-allin": "open on every instrument; it CANNOT be refit at all -- no ERA5 "
                    "snapshot on this machine (exp-031 section 10.3, exp-025 "
                    "oof-nested/README.md)",
    },
    "closed_count": {
        "_units": "DIRECT stacker feature columns, which is the only thing this count "
                  "counts. It is NOT a map of supervised ancestors: see F1-fresh above "
                  "(Sol round 1, blocking 6, which corrected an earlier version of this "
                  "record that read the direct count as the ancestor map)",
        "stack-5, v6, v7": "3 of their 7 direct member columns (T-tailv4, L2-hub8725, "
                           "F1-fresh)",
        "K2, refit-bulk, v10": "2 of their 7 direct member columns (T-tailv4, "
                               "L2-hub8725); F1-fresh still reaches them through the "
                               "blend-5 anchor, and v10 through v7 as well, so their "
                               "ANCESTOR closure is not smaller than stack-5's",
    },
    "not_audited": {
        "k2-w300": "a Q01 finalist with NO cached nested vector -- exp-031's OOF carries "
                   "stack_5, v6, v7, K2 and refit_bulk only. Q07 stage 1 cannot audit it "
                   "and K2 is not a substitute for it (Sol round 1, blocking 8)",
    },
    "identification": {
        "what the change contains": "published -> nested is NOT the nesting term alone. "
                                    "exp-031 splits published -> kaggle (the machine) "
                                    "from kaggle -> nested (the nesting), and the kaggle "
                                    "column is not in the committed OOF, so this stage "
                                    "cannot separate them. The change also carries the "
                                    "loss of one training month.",
        "what follows": "the gain changes little on two partial, confounded instruments. "
                        "These instruments do not identify a cause, and offsetting "
                        "machine, nesting and training-size terms can produce a small "
                        "net change (Sol round 1, blocking 7).",
    },
    "anchor": {
        "published": "fold g's training anchor is fitted on the other two folds, which "
                     "includes the scored fold f -- channel A OPEN",
        "nested": "channel A OPEN (shipped anchor)",
        "nestedanchor": "channel A CLOSED: fold g's training anchor is fitted on the "
                        "single remaining fold (exp-026 nested_anchor.py)",
    },
    "head": "the stacker itself is refitted per fold on the other two folds on every "
            "instrument; its own labels are never the scored fold's",
    "mask": "the eight-airport mask and the EHAM hold are SELECTED on the same three "
            "folds they are scored on (exp-036 section 3). No instrument here closes "
            "that channel, and it is the one an upload has already priced",
}
LIMIT = ("exp-031 RESULTS section 1: this design confounds closing a channel with "
         "removing one month of training data, the measured member shifts include "
         "negative values, and no column here is monotone in the direction of the "
         "clean score. Survival is not proof of cleanliness; movement is not proof "
         "of leakage.")


def endpoint(y: dict, p10: dict, cand: dict, P: dict) -> dict:
    """Q01's endpoints for one candidate on one instrument."""
    st, stc = {}, {}
    for f in FOLDS:
        pr = A.per_row(y[f], p10[f], cand[f])
        st[f] = A.fold_stats(pr["r"], pr["d"], np.ones(len(pr["d"]), bool))
        stc[f] = A.fold_stats(pr["r"], pr["d"], P[f]["clean"])
        st[f]["_r"], st[f]["_d"] = pr["r"], pr["d"]
    b = A.board_gain(st)
    loo = A.scale_loo({f: st[f]["A"] for f in FOLDS}, {f: st[f]["B"] for f in FOLDS},
                      {f: st[f]["n"] for f in FOLDS})
    cv = {f: A.scale_cv(st[f]["_r"], st[f]["_d"], P[f]["blk3"],
                        np.ones(len(st[f]["_d"]), bool)) for f in FOLDS}
    m10 = sum(A.SHARES[f] * st[f]["mse_v10"] for f in A.BOARD_FOLDS)
    out = {"gain_board_allrow": b["gain_board"],
           "gain_board_loo_amp": float(np.sqrt(m10) - np.sqrt(
               sum(A.SHARES[f] * (st[f]["mse_v10"] - loo[f]["gain_mse"])
                   for f in A.BOARD_FOLDS))),
           "gain_board_cv_amp": float(np.sqrt(m10) - np.sqrt(
               sum(A.SHARES[f] * (st[f]["mse_v10"] - cv[f]["gain_mse"])
                   for f in A.BOARD_FOLDS))),
           "board_v10": b["board_v10"]}
    # the CLEAN endpoint on every instrument, which Q07 stage 1 owes the same as
    # Q01 does (Sol round 1, blocking 8)
    looc = A.scale_loo({f: stc[f]["A"] for f in FOLDS}, {f: stc[f]["B"] for f in FOLDS},
                       {f: stc[f]["n"] for f in FOLDS})
    cvc = {f: A.scale_cv(st[f]["_r"], st[f]["_d"], P[f]["blk3"], P[f]["clean"])
           for f in FOLDS}
    m10c = sum(A.SHARES[f] * stc[f]["mse_v10"] for f in A.BOARD_FOLDS)
    out["gain_board_clean"] = A.board_gain(stc)["gain_board"]
    out["gain_board_loo_amp_clean"] = float(np.sqrt(m10c) - np.sqrt(
        sum(A.SHARES[f] * (stc[f]["mse_v10"] - looc[f]["gain_mse"])
            for f in A.BOARD_FOLDS)))
    out["gain_board_cv_amp_clean"] = float(np.sqrt(m10c) - np.sqrt(
        sum(A.SHARES[f] * (stc[f]["mse_v10"] - cvc[f]["gain_mse"])
            for f in A.BOARD_FOLDS)))
    for f in FOLDS:
        out[f"A_{f}"] = st[f]["A"]
        out[f"B_{f}"] = st[f]["B"]
        out[f"gainrmse_{f}"] = st[f]["gain_rmse"]
        out[f"scale_{f}"] = st[f]["scale"]
        out[f"amp_loo_{f}"] = loo[f]["s"]
    out["_st"] = st
    return out


def groups(st: dict, P: dict, label: str, instr: str) -> list:
    """Paired SSE by regime and by month, at full amplitude and at the LOO amplitude."""
    rows = []
    for f in FOLDS:
        r, d = st[f]["_r"], st[f]["_d"]
        e10 = r * r
        for reg in ("nm_bulk", "nm_late", "nonm", "noac"):
            m = P[f]["regime"] == reg
            if not m.any():
                continue
            rows.append({"candidate": label, "instrument": instr, "fold": f, "regime": reg,
                         "n": int(m.sum()),
                         "dsse_full": float((((r - d) ** 2)[m] - e10[m]).sum())})
    return rows


def dates(st: dict, P: dict, label: str, instr: str) -> list:
    """Per-DATE paired SSE on every instrument, which the shared contract asks
    for and an earlier version of this stage did not write (Sol round 1,
    blocking 8).  One row per candidate x instrument x fold x calendar day."""
    rows = []
    for f in FOLDS:
        r, d = st[f]["_r"], st[f]["_d"]
        t = pd.DataFrame({"day": P[f]["day"], "e10": r * r, "ec": (r - d) ** 2,
                          "d": d})
        g = t.groupby("day").agg(n=("d", "size"),
                                 n_changed=("d", lambda z: int((z != 0).sum())),
                                 sse_v10=("e10", "sum"), sse_cand=("ec", "sum")).reset_index()
        g.insert(0, "fold", f)
        g.insert(0, "instrument", instr)
        g.insert(0, "candidate", label)
        g["dsse"] = g["sse_cand"] - g["sse_v10"]
        rows.append(g)
    return rows


def main() -> None:
    ref, vec, prov, files = V.load(verbose=False)
    keep = files["mask"]["keep"]
    P = {f: {"regime": ref[f]["regime"].astype(str).to_numpy(),
             "airport": ref[f]["airport"].astype(str).to_numpy(),
             "blk3": A.blocks(A.day_of(ref[f]["t"].to_numpy("float64")), 3),
             "day": A.day_of(ref[f]["t"].to_numpy("float64")),
             "clean": ~ref[f]["MVT_ID_mvt"].isin(V.D.excluded_ids(f)).to_numpy()}
         for f in FOLDS}
    y = {f: ref[f]["y_true"].to_numpy("float64") for f in FOLDS}

    bank = {"published": {n: vec[n] for n in COLS}}
    bank["published"]["v10"] = vec["v10 (exp-036 refitbulk-mask)"]
    for instr, tmpl in INSTR.items():
        got = {}
        for name, col in COLS.items():
            v = {}
            for f in FOLDS:
                d = pd.read_parquet(E31 / tmpl.format(fold=f))
                if not np.array_equal(d["MVT_ID_mvt"].to_numpy("float64"),
                                      ref[f]["MVT_ID_mvt"].to_numpy("float64")):
                    raise SystemExit(f"{instr} {f}: not the reference's row order")
                if float(np.abs(d["y_true"].to_numpy("float64") - y[f]).max()) > 1e-9:
                    raise SystemExit(f"{instr} {f}: a different label vector")
                v[f] = d[col].to_numpy("float64")
            got[name] = v
        # v10 on this instrument: the SAME mask arithmetic, on this instrument's
        # refit-bulk and v7.  Nothing is refitted; the mask is exp-036's own list.
        got["v10"] = {f: np.where(np.isin(P[f]["airport"], keep),
                                  got["refit-bulk"][f], got["v7"][f]) for f in FOLDS}
        bank[instr] = got

    rows, grp, DT = [], [], []
    for instr, bk in bank.items():
        p10 = bk["v10"]
        for name in list(COLS) + ["v10"]:
            e = endpoint(y, p10, bk[name], P)
            st = e.pop("_st")
            rows.append({"instrument": instr, "candidate": name, **e})
            grp += groups(st, P, name, instr)
            DT += dates(st, P, name, instr)
    T = pd.DataFrame(rows)
    G = pd.DataFrame(grp)

    # the CHANGE, which is what Q07 asks for
    base = T[T["instrument"] == "published"].set_index("candidate")
    parts = []
    for instr in INSTR:
        d = T[T["instrument"] == instr].set_index("candidate")
        for k in ("gain_board_allrow", "gain_board_loo_amp", "gain_board_cv_amp",
                  "board_v10", "amp_loo_july_loo", "amp_loo_january",
                  "gainrmse_july_loo", "gainrmse_december", "gainrmse_january"):
            parts.append(pd.DataFrame({"instrument": instr, "quantity": k,
                                       "published": base[k], "instrument_value": d[k],
                                       "change": d[k] - base[k]}))
    C = pd.concat(parts).reset_index().rename(columns={"index": "candidate"})

    T.to_csv(EXP / "analysis" / "q07-instruments.csv", index=False)
    C.to_csv(EXP / "analysis" / "q07-change.csv", index=False)
    G.to_csv(EXP / "analysis" / "q07-groups.csv", index=False)
    if DT:
        pd.concat(DT, ignore_index=True).to_csv(
            EXP / "analysis" / "q07-dates.csv", index=False)
    (EXP / "analysis" / "q07-ancestors.json").write_text(json.dumps(
        {"ancestors": ANCESTORS, "limit": LIMIT,
         "instruments": {"published": "committed rotation (Q01's own)",
                         **{k: str(E31.relative_to(ROOT) / v) for k, v in INSTR.items()}},
         "note": "v10 is rebuilt on each instrument by exp-036's own np.where mask on "
                 "that instrument's refit-bulk and v7; the mask list is identical"},
        indent=2))

    pd.set_option("display.width", 220)
    print("=== the endpoint on each instrument (gain over that instrument's v10) ===")
    print(T[["instrument", "candidate", "board_v10", "gain_board_allrow",
             "gain_board_loo_amp", "gain_board_cv_amp",
             "amp_loo_july_loo", "amp_loo_january"]].round(4).to_string(index=False))
    print(f"\n=== the CHANGE against the published instrument ===")
    piv = C.pivot_table(index=["quantity", "candidate"], columns="instrument",
                        values="change")
    print(piv.round(4).to_string())
    print(f"\n{LIMIT}")


if __name__ == "__main__":
    main()
