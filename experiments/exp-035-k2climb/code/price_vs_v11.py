"""Section 20: the served candidate priced against **v11**, on the exp-038 contract.

The desk (2026-09-08 12:05 UTC) asked for the served candidate against v11, not
against v9, and on exp-038's own algebra rather than this experiment's rotation
tables, so that the number is comparable with `experiments/exp-038-packet/
analysis/q01-table.csv` row for row.  Nothing here is refitted: every vector is
either read back from a committed rotation, gated on the number its own
experiment published, or built by `np.where` / one scalar over two of them.

The contract, from `exp-038-packet/code/align.py` and `q01.py`:

    r = y - p_v11            the residual v11 leaves, fold side
    d = p_cand - p_v11       the update the candidate proposes
    A = mean(r*d)   B = mean(d*d)
    full-amplitude MSE gain = 2A - B ;  fitted scale = A/B, its gain A^2/B

reported on the **all-row** January/July board at the ranking month shares (the
primary), again **clean** (`prc.data.excluded_ids` dropped, the convention every
LEDGER row quotes), and december alone, which is gated but carries no share.
Intervals are the block bootstrap over CALENDAR DAYS shared across airports,
resampled inside each fold and recombined through the same board formula, with
the 3-day and 7-day widths as the sensitivity check.

**v11** is `submissions/candidates/exp-038-v10-stack5-blend.parquet`
(`docs/SUBMISSIONS.md` row 20, sha 42ca5ee6..., board 263.0298) =
`v10 + s*(stack-5 - v10)` at `s = 0.217`.  Its fold-side form is rebuilt here by
that same arithmetic on exp-038's own gated v10 and stack-5 vectors, and the
RULE is gated on the served bytes: the three uploaded/candidate parquet files
must satisfy the same identity to max abs difference 0.

Two forms of the candidate, as the desk named them:

  1. **the served candidate**, `submissions/candidates/exp-035-k2climb-mod.parquet`
     -- the `mod` head served as v7 serves K2 (EHAM and `noac` from stack-5).
     Its fold-side vector is `oof/cand-k2c-mod-k2-hold-{fold}.parquet:y_pred`,
     gated on section 17's published clean rotation.
  2. **the same head composed the way v11 is**: exp-036's mask arithmetic (the
     head at the eight kept airports of `exp-036-mask/analysis/mask.json`, v7
     elsewhere, so EDDF and EHAM stay at v7), then the same 0.217 toward
     stack-5.  This is the only form that is like-for-like with v11: it differs
     from v11 in the MEMBER and in nothing else.

Form 1 is what exists on disk and is what the desk would upload; form 2 is what
v11's own recipe would produce from this member.  Both are priced, because the
first answers "what does the file we have gain over what we submitted" and the
second answers "what does the member gain, with the composition held fixed".
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
P38 = ROOT / "experiments/exp-038-packet"
sys.path.insert(0, str(P38 / "code"))
import align as A  # noqa: E402
import vectors as V  # noqa: E402

FOLDS = V.FOLDS
BASE = "v10 (exp-036 refitbulk-mask)"
DRAWS = 2000
GATE = 1e-3
VEC_TOL = 1e-9

V11_FILE = "submissions/candidates/exp-038-v10-stack5-blend.parquet"
V11_SHA = "42ca5ee6410938d56c8830e9ec3980d806633c3b2ff15963c0ffe2fe97c39f8c"
V11_RECORD = P38 / "analysis" / "served-exp-038-v10-stack5-blend.json"
S5_FILE = "submissions/candidates/exp-026-stack-5.parquet"
S5_SHA = "94f18e47a73265a6db66811ededfc3c935fb2a37fac04bf0cd34d45af708a2a1"

#: exp-038's own published fold side for v11, `docs/LEDGER.md` (the v11 upload
#: row) and RESULTS §2c: clean 256.91 with july_loo 284.0467 and december
#: 204.7365, +0.1144 clean over v10.  The rebuilt vector is gated on it, so §20a
#: does not rest on the rule gate alone (round 3, non-blocking 4).
V11_PUBLISHED = {"july_loo": 284.0467, "december": 204.7365}
#: exp-030 `analysis/arm-eham-out.json`: the tail v7 leaves, the clause's base.
V7_TAIL = {"july_loo": 2406.961, "december": 1798.904, "january": 2088.426}
TAIL = 3600.0
V11_PUBLISHED_GAIN_CLEAN = 0.1144

CAND_OOF = EXP / "oof" / "cand-k2c-mod-k2-hold-{fold}.parquet"
#: Round 3, blocking 1: v11's head is the **refit-bulk** anchor, so the only
#: like-for-like form is the climbed member through that same arm.
#: `code/rotate_refit_bulk.py --arm mod`, 46 s on the laptop, 0 GPU-h.
CAND_RB_OOF = EXP / "oof" / "cand-k2c-mod-refit-bulk-{fold}.parquet"
CAND_RB_PUBLISHED = {"july_loo": 283.6627, "december": 204.2511, "january": 217.6985}
#: exp-027's own committed K2 head in the SAME hold anchor form 2 uses, so the
#: anchor's share of the difference can be read off instead of inferred.
K2_HOLD_OOF = (ROOT / "experiments/exp-027-step/oof/exp-027-k2-k2-hold-{fold}.parquet")
K2_HOLD_PUBLISHED = {"july_loo": 284.3195, "december": 205.2156, "january": 218.2582}
CAND_FILE = "submissions/candidates/exp-035-k2climb-mod.parquet"
CAND_SHA = "e6de36242cbe8edd4e5673854eaaef8f87cb5bdad296289cdd5c45c6b9e6eb0d"
#: section 17 / `analysis/candidate-k2c-mod.json`, arm `k2-hold-eham-out`.
CAND_PUBLISHED = {"july_loo": 283.9958, "december": 204.8691, "january": 217.9169}


def served_column(rel: str, sha: str) -> np.ndarray:
    """One served file's target column in the submitting template's order, hash checked."""
    p = ROOT / rel
    got = V.D.sha256(p)
    if got != sha:
        raise SystemExit(f"{p.name}: sha256 {got[:8]} against the record's {sha[:8]}")
    d = pd.read_parquet(p)
    tmpl = V.D.load_submitting()["MVT_ID_mvt"].to_numpy().astype("float64")
    ids = d["MVT_ID_mvt"].to_numpy("float64")
    if np.array_equal(ids, tmpl):
        return d[V.D.TARGET].to_numpy("float64")
    v = d.set_index("MVT_ID_mvt")[V.D.TARGET].reindex(tmpl)
    if v.isna().any():
        raise SystemExit(f"{p.name}: {int(v.isna().sum())} template rows are missing")
    return v.to_numpy("float64")


def gate_rule_on_served_bytes(s: float) -> dict:
    """v11's blend rule, checked on the bytes that were uploaded.

    The fold side cannot be gated against v11 row by row -- v11 is a ranking-file
    vector and has no fold rows -- so what is gated is the RULE the fold side
    applies: on the served files, `v11 == v10 + s*(stack5 - v10)` exactly.
    """
    v10 = served_column(V.V10_FILE, V.V10_SHA)
    s5 = served_column(S5_FILE, S5_SHA)
    v11 = served_column(V11_FILE, V11_SHA)
    rebuilt = v10 + s * (s5 - v10)
    off = float(np.abs(v11 - rebuilt).max())
    if off > VEC_TOL:
        raise SystemExit(f"v11 is not v10 + {s}*(stack-5 - v10) on the served bytes: {off:g}")
    return {"s": s, "n_rows": int(len(v11)), "max_abs_difference": off,
            "files": {"v10": V.V10_FILE, "stack5": S5_FILE, "v11": V11_FILE},
            "sha256": {"v10": V.V10_SHA, "stack5": S5_SHA, "v11": V11_SHA}}


def load_head(ref: dict, tpl, published: dict, what: str) -> dict:
    """One head's fold-side vector, row-order and clean-rotation gated."""
    out = {}
    for f in FOLDS:
        d = pd.read_parquet(str(tpl).format(fold=f))
        V._order(d, ref[f], what, f)
        out[f] = d["y_pred"].to_numpy("float64")
    got = V.clean_of(out, ref)
    off = {f: round(got[f] - published[f], 4) for f in FOLDS}
    if max(abs(x) for x in off.values()) > GATE:
        raise SystemExit(f"{what} does not reproduce its published rotation: {off}")
    return out


def price(name: str, cand: dict, p11: dict, panel: dict) -> dict:
    """One row of the table: the exp-038 algebra with v11 in the baseline's place."""
    st_all, st_cl, parts_all, parts_cl = {}, {}, {}, {}
    ci_fold, ci_fold_cl = {}, {}
    extra, rd, tail = {}, {}, {}
    for f in FOLDS:
        p = panel[f]
        pr = A.per_row(p["y"], p11[f], cand[f])
        r, d = pr["r"], pr["d"]
        allrows = np.ones(len(d), bool)
        st_all[f] = A.fold_stats(r, d, allrows)
        st_cl[f] = A.fold_stats(r, d, p["clean"])
        e11, ec = r * r, (r - d) ** 2
        parts_all[f] = (p["blk1"], e11, ec)
        parts_cl[f] = (p["blk1"][p["clean"]], e11[p["clean"]], ec[p["clean"]])
        ci_fold[f] = A.boot_fold(*parts_all[f], DRAWS)
        ci_fold_cl[f] = A.boot_fold(*parts_cl[f], DRAWS)
        rd[f] = (r, d)
        # the registered second kill clause, read on the composed vector: clean
        # rows with `y > 3600`, RMSE against v7 (and against v11 beside it).
        t = p["clean"] & (p["y"] > TAIL)
        yt = p["y"][t]
        tail[f] = {
            "n": int(t.sum()),
            "rmse_v7": round(float(np.sqrt(np.mean((yt - p["v7"][t]) ** 2))), 4),
            "rmse_v11": round(float(np.sqrt(np.mean((yt - p11[f][t]) ** 2))), 4),
            "rmse_cand": round(float(np.sqrt(np.mean((yt - cand[f][t]) ** 2))), 4)}
        tail[f]["vs_v7"] = round(tail[f]["rmse_v7"] - tail[f]["rmse_cand"], 4)
        tail[f]["vs_v11"] = round(tail[f]["rmse_v11"] - tail[f]["rmse_cand"], 4)
        tail[f]["clause_pass"] = bool(tail[f]["vs_v7"] > -1.0)
        extra[f] = {
            "d_max_on_eham": float(np.abs(d[p["airport"] == "EHAM"]).max()),
            "d_max_on_eddf": float(np.abs(d[p["airport"] == "EDDF"]).max()),
            "d_max_on_noac": float(np.abs(d[p["regime"] == "noac"]).max()),
        }
    b_all, b_cl = A.board_gain(st_all), A.board_gain(st_cl)
    # -- bounded amplitudes fitted AWAY from the rows they are scored on -----
    # `A/B` above is fitted and scored on the same rows, which exp-038's Q01
    # names as the thing not to believe; these two are its own repair, so the
    # "a blend would gain more" reading is quoted at a defensible amplitude.
    loo = A.scale_loo({f: st_all[f]["A"] for f in FOLDS}, {f: st_all[f]["B"] for f in FOLDS},
                      {f: st_all[f]["n"] for f in FOLDS})
    cv = {f: A.scale_cv(rd[f][0], rd[f][1], panel[f]["blk3"],
                        np.ones(len(rd[f][1]), bool)) for f in FOLDS}
    cv_c = {f: A.scale_cv(rd[f][0], rd[f][1], panel[f]["blk3"], panel[f]["clean"])
            for f in FOLDS}

    def board_at(st: dict, gain_mse: dict) -> float:
        b0 = np.sqrt(sum(A.SHARES[f] * st[f]["mse_v10"] for f in A.BOARD_FOLDS))
        b1 = np.sqrt(sum(A.SHARES[f] * (st[f]["mse_v10"] - gain_mse[f]) for f in A.BOARD_FOLDS))
        return float(b0 - b1)

    parts_loo = {f: (panel[f]["blk1"], rd[f][0] ** 2,
                     (rd[f][0] - loo[f]["s"] * rd[f][1]) ** 2) for f in A.BOARD_FOLDS}
    rec = {
        "candidate": name,
        "n_changed": {f: st_all[f]["n_changed"] for f in FOLDS},
        "max_abs_d": {f: st_all[f]["max_abs_d"] for f in FOLDS},
        "held_rows": extra,
        "tail_clause": tail,
        "tail_clause_pass": all(tail[f]["clause_pass"] for f in FOLDS),
        # -- the primary endpoint: ALL-ROW Jan/July at the ranking shares --
        "gain_board_allrow": b_all["gain_board"],
        "ci_allrow_1day": A.boot_board(parts_all, DRAWS),
        "ci_allrow_3day": A.boot_board({f: (panel[f]["blk3"], parts_all[f][1], parts_all[f][2])
                                        for f in A.BOARD_FOLDS}, DRAWS),
        "ci_allrow_7day": A.boot_board({f: (panel[f]["blk7"], parts_all[f][1], parts_all[f][2])
                                        for f in A.BOARD_FOLDS}, DRAWS),
        "gain_board_allrow_fitted": b_all["gain_board_fitted"],
        # -- clean, the convention the LEDGER quotes --
        "gain_board_clean": b_cl["gain_board"],
        "ci_clean_1day": A.boot_board(parts_cl, DRAWS),
        "gain_board_clean_fitted": b_cl["gain_board_fitted"],
        # -- the same shrink, at an amplitude fitted off the scored rows --
        "amp_loo": {f: loo[f]["s"] for f in FOLDS},
        "amp_cv_3day": {f: cv[f]["s"] for f in FOLDS},
        "gain_board_allrow_loo_amp": board_at(st_all, {f: loo[f]["gain_mse"] for f in FOLDS}),
        "ci_allrow_loo_amp": A.boot_board(parts_loo, DRAWS),
        "gain_board_allrow_cv_amp": board_at(st_all, {f: cv[f]["gain_mse"] for f in FOLDS}),
        "gain_board_clean_cv_amp": board_at(st_cl, {f: cv_c[f]["gain_mse"] for f in FOLDS}),
        "board_v11_allrow": b_all["board_v10"], "board_cand_allrow": b_all["board_cand"],
        "board_v11_clean": b_cl["board_v10"], "board_cand_clean": b_cl["board_cand"],
        # -- every fold, both conventions, with its own day-clustered interval --
        "folds": {f: {
            "n": st_all[f]["n"],
            "rmse_v11": st_all[f]["rmse_v10"], "rmse_cand": st_all[f]["rmse_cand"],
            "gain_rmse": st_all[f]["gain_rmse"], "ci_rmse": ci_fold[f],
            "A": st_all[f]["A"], "B": st_all[f]["B"],
            "gain_mse_2AB": st_all[f]["gain_mse_full"],
            "scale": st_all[f]["scale"], "gain_mse_fitted": st_all[f]["gain_mse_fitted"],
            "clean": {
                "n": st_cl[f]["n"],
                "rmse_v11": st_cl[f]["rmse_v10"], "rmse_cand": st_cl[f]["rmse_cand"],
                "gain_rmse": st_cl[f]["gain_rmse"], "ci_rmse": ci_fold_cl[f],
                "A": st_cl[f]["A"], "B": st_cl[f]["B"],
                "gain_mse_2AB": st_cl[f]["gain_mse_full"],
                "scale": st_cl[f]["scale"], "gain_mse_fitted": st_cl[f]["gain_mse_fitted"],
            },
        } for f in FOLDS},
    }
    return rec


def main() -> None:
    t0 = time.time()
    ref, vec, prov, files = V.load(verbose=True)
    keep = json.loads((ROOT / V.MASK_JSON).read_text())["keep"]

    rec38 = json.loads(V11_RECORD.read_text())
    if rec38["sha256"] != V11_SHA or rec38["rule"] != "p = v10 + s * (stack-5 - v10)":
        raise SystemExit(f"exp-038's served record does not describe v11: {rec38['sha256'][:8]}")
    S = float(rec38["s"])
    rule = gate_rule_on_served_bytes(S)
    print(f"[v11] rule gated on the served bytes at s={S}: max abs difference "
          f"{rule['max_abs_difference']:g}", flush=True)

    p10, s5, v7 = vec[BASE], vec["stack-5"], vec["v7"]
    p11 = {f: p10[f] + S * (s5[f] - p10[f]) for f in FOLDS}
    head = load_head(ref, CAND_OOF, CAND_PUBLISHED, "the mod head (hold anchor)")
    head_rb = load_head(ref, CAND_RB_OOF, CAND_RB_PUBLISHED,
                        "the mod head (refit-bulk anchor)")
    k2_hold = load_head(ref, K2_HOLD_OOF, K2_HOLD_PUBLISHED, "K2's own head (hold anchor)")

    def compose(h: dict) -> dict:
        """v11's own composition applied to one head: mask, then blend at s."""
        m = {f: np.where(np.isin(ref[f]["airport"].astype(str).to_numpy(), keep), h[f], v7[f])
             for f in FOLDS}
        return m, {f: m[f] + S * (s5[f] - m[f]) for f in FOLDS}

    #: The desk's step 2: form 2' where **v11's own prediction** is below 3600 s
    #: and v11 where it is not.  The selector is the served predictor, never the
    #: label, so it exists on the ranking file exactly as it does here.
    def tail_hold(cand: dict) -> dict:
        return {f: np.where(p11[f] < TAIL, cand[f], p11[f]) for f in FOLDS}

    masked, form2 = compose(head)
    _, form2p = compose(head_rb)     # like-for-like: v11 with its MEMBER replaced
    _, anchor_only = compose(k2_hold)  # like-for-like: v11 with its ANCHOR replaced
    #: The two axes that separate form 1 from form 2 are the exp-036 MASK and the
    #: 0.217 BLEND, and they move the answer in opposite directions, so the two
    #: remaining cells of the 2x2 are priced as well.  Both are `np.where` and one
    #: scalar over vectors already gated above -- no fit, no member, no GPU.
    blend1 = {f: head[f] + S * (s5[f] - head[f]) for f in FOLDS}

    panel = {}
    for f in FOLDS:
        d = ref[f]
        day = A.day_of(d["t"].to_numpy("float64"))
        panel[f] = {"y": d["y_true"].to_numpy("float64"), "v7": v7[f],
                    "airport": d["airport"].astype(str).to_numpy(),
                    "regime": d["regime"].astype(str).to_numpy(),
                    "clean": ~d["MVT_ID_mvt"].isin(V.D.excluded_ids(f)).to_numpy(),
                    "blk1": A.blocks(day, 1), "blk3": A.blocks(day, 3),
                    "blk7": A.blocks(day, 7)}

    # `align._cells` sums per block with `np.add.reduceat`, which returns the
    # element at the index instead of a zero for an EMPTY block, so a gap in the
    # block indexing would corrupt every interval quietly.  There is no gap:
    # every day of every fold carries rows, on all rows and on the clean subset.
    blocks_dense = {}
    for f in FOLDS:
        for w, key in ((1, "blk1"), (3, "blk3"), (7, "blk7")):
            b = panel[f][key]
            bc = b[panel[f]["clean"]]
            k, kc = int(b.max()) + 1, int(bc.max()) + 1
            ok = len(np.unique(b)) == k and len(np.unique(bc)) == kc
            if not ok:
                raise SystemExit(f"{f}: {w}-day blocks are not contiguous; the "
                                 f"bootstrap's reduceat would read an empty block")
            blocks_dense[f"{f}_{w}day"] = {"blocks": k, "clean_blocks": kc, "dense": True}

    #: The desk's 13:30 UTC arm: exp-036's own per-airport rule plus §24's clause,
    #: selected in `code/airport_hold.py` and read back here so the pricing
    #: instrument stays one file.  The selection is POST-HOC on these folds.
    ah_path = EXP / "analysis" / "airport-hold.json"
    ah = json.loads(ah_path.read_text()) if ah_path.exists() else None
    if ah is not None:
        sel = ah["selected"]
        form2p_ah = {f: np.where(np.isin(ref[f]["airport"].astype(str).to_numpy(), sel),
                                 form2p[f], p11[f]) for f in FOLDS}

    form2p_th = tail_hold(form2p)
    hold_stats = {}
    for f in FOLDS:
        held = p11[f] >= TAIL
        t = panel[f]["clean"] & (panel[f]["y"] > TAIL)
        hold_stats[f] = {
            "rows_held": int(held.sum()),
            "share_of_rows": round(float(held.mean()), 6),
            "tail_rows": int(t.sum()),
            "tail_rows_held": int((held & t).sum()),
            "tail_rows_held_share": round(float((held & t).sum() / max(t.sum(), 1)), 4),
            "non_tail_rows_held": int((held & ~t).sum())}
    print("[tail-hold] rows where v11 predicts >= 3600 s: "
          + ", ".join(f"{f} {hold_stats[f]['rows_held']} "
                      f"({hold_stats[f]['tail_rows_held']} of "
                      f"{hold_stats[f]['tail_rows']} tail rows)" for f in FOLDS), flush=True)

    c11, c10 = V.clean_of(p11, ref), V.clean_of(p10, ref)
    off11 = {f: round(c11[f] - V11_PUBLISHED[f], 4) for f in V11_PUBLISHED}
    gain11 = round(V.Sc.P.board(c10) - V.Sc.P.board(c11), 4)
    if (max(abs(x) for x in off11.values()) > GATE
            or abs(gain11 - V11_PUBLISHED_GAIN_CLEAN) > GATE):
        raise SystemExit(f"the rebuilt v11 does not reproduce exp-038's published fold "
                         f"side: per fold {off11}, clean gain over v10 {gain11}")
    print(f"[v11] fold side gated on exp-038's published numbers: per fold {off11}, "
          f"clean gain over v10 {gain11} against {V11_PUBLISHED_GAIN_CLEAN}", flush=True)

    clean = {n: {f: round(x, 4) for f, x in V.clean_of(v, ref).items()}
             for n, v in (("v11", p11), ("form 1 (served candidate)", head),
                          ("form 2' (member replaced)", form2p),
                          ("tail-hold form 2'", form2p_th),
                          *([("airport-hold form 2'", form2p_ah)] if ah is not None else []),
                          ("anchor only (anchor replaced)", anchor_only),
                          ("form 2 (confounded)", form2),
                          ("diagnostic: mask only, no blend", masked),
                          ("diagnostic: no mask, blend only", blend1),
                          ("v10 (exp-038 baseline)", p10))}
    for n, c in clean.items():
        print(f"  {n:38s} clean {c}  board {V.Sc.P.board(c):9.4f}", flush=True)

    rows = [price("v11 (identity)", p11, p11, panel)]
    if ah is not None:
        rows.append(price(f"airport-hold form 2' (serve {'+'.join(sel) or 'nothing'})",
                          form2p_ah, p11, panel))
    rows += [price("tail-hold form 2' (v11 predicts >= 3600 held)", form2p_th, p11, panel),
            price("form 1 (served candidate)", head, p11, panel),
            price("form 2' (v11's recipe, member replaced)", form2p, p11, panel),
            price("anchor only (v11's recipe, anchor replaced)", anchor_only, p11, panel),
            price("form 2 (confounded: member AND anchor)", form2, p11, panel),
            price("diagnostic: mask only, no blend", masked, p11, panel),
            price("diagnostic: no mask, blend only", blend1, p11, panel)]
    ident = rows[0]
    if (max(abs(ident["folds"][f]["A"]) for f in FOLDS) > 0.0
            or abs(ident["gain_board_allrow"]) > 1e-12):
        raise SystemExit("the identity row is not zero; the arithmetic above is wrong")

    out = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline": {"name": "v11", "file": V11_FILE, "sha256": V11_SHA,
                     "rule": rec38["rule"], "s": S,
                     "s_reason": rec38.get("s_reason", ""),
                     "board_scored": 263.0298,
                     "submissions_row": "docs/SUBMISSIONS.md row 20",
                     "fold_side": "exp-038 code/vectors.py v10 (exp-036 refitbulk-mask) "
                                  "and stack-5, both gated there, blended at s"},
        "rule_gate_on_served_bytes": rule,
        "candidate": {"file": CAND_FILE, "sha256": CAND_SHA,
                      "fold_side": str(Path(str(CAND_OOF).format(fold="{fold}"))
                                       .relative_to(ROOT)),
                      "published_clean": CAND_PUBLISHED,
                      "gate": "reproduces section 17's clean rotation within 1e-3"},
        "composition": {"rule": "np.where(airport in exp-036 keep, head, v7), "
                                "then + s*(stack-5 - it)",
                        "keep": list(keep), "hold_at_v7": ["EDDF", "EHAM"], "s": S,
                        "heads": {"form 2'": str(Path(str(CAND_RB_OOF)
                                                      .format(fold='{fold}')).relative_to(ROOT)),
                                  "anchor only": str(Path(str(K2_HOLD_OOF)
                                                          .format(fold='{fold}')).relative_to(ROOT)),
                                  "form 2": str(Path(str(CAND_OOF)
                                                     .format(fold='{fold}')).relative_to(ROOT))}},
        "airport_hold": ({"selected": ah["selected"], "held_at_v11": ah["held_at_v11"],
                          "rule": ah["rule"], "selection": ah["selection"],
                          "record": str(ah_path.relative_to(ROOT))} if ah is not None else None),
        "tail_hold": {"rule": "np.where(p_v11 < 3600, form 2', v11) -- the SERVED "
                              "predictor selects, never the label",
                      "threshold": TAIL, "stats": hold_stats},
        "v7_tail_published": V7_TAIL,
        "v11_fold_gate": {"published": V11_PUBLISHED, "offsets": off11,
                          "clean_gain_over_v10": gain11,
                          "published_clean_gain": V11_PUBLISHED_GAIN_CLEAN},
        "endpoint": ("paired all-row January/July MSE at the ranking month shares "
                     f"{A.SHARES}; clean and december reported separately"),
        "intervals": "block bootstrap over calendar days shared across airports, "
                     f"{DRAWS} draws, seed 7; 3-day and 7-day widths as sensitivity",
        "blocks_dense": blocks_dense,
        "clean_rotations": {n: {"per_fold": c, "board": round(V.Sc.P.board(c), 4)}
                            for n, c in clean.items()},
        "rows": rows,
        "wall_s": round(time.time() - t0, 1),
    }
    (EXP / "analysis").mkdir(exist_ok=True)
    (EXP / "analysis" / "price-vs-v11.json").write_text(json.dumps(out, indent=1, default=str))

    print("\n" + "=" * 78)
    for r in rows:
        print(f"{r['candidate']:38s} all-row {r['gain_board_allrow']:+8.4f} "
              f"[{r['ci_allrow_1day'][0]:+7.4f},{r['ci_allrow_1day'][1]:+7.4f}]  "
              f"clean {r['gain_board_clean']:+8.4f} "
              f"[{r['ci_clean_1day'][0]:+7.4f},{r['ci_clean_1day'][1]:+7.4f}]  "
              f"fitted {r['gain_board_allrow_fitted']:+7.4f} "
              f"(loo {r['gain_board_allrow_loo_amp']:+7.4f}, cv {r['gain_board_allrow_cv_amp']:+7.4f})  "
              f"dec {r['folds']['december']['gain_rmse']:+8.4f}  tail "
              + " ".join(f"{r['tail_clause'][f]['vs_v7']:+7.2f}" for f in FOLDS)
              + ("  CLAUSE PASS" if r["tail_clause_pass"] else "  CLAUSE FAIL"))
    print(f"\n[price-vs-v11] {time.time()-t0:.0f}s -> analysis/price-vs-v11.json", flush=True)


if __name__ == "__main__":
    main()
