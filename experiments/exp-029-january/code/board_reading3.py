"""Re-quote the measured arms under the desk's BOARD READING 3.

The desk (2026-09-07, after v3 = stack-5 scored 264.3861) set the convention:

    quote every candidate as clean-with-noac-at-the-anchor + 4.28

because bulk fold gains transfer to the board and `noac` fold gains do not
(v2 missed its face-value prediction by 2.8 s, of which 2.3 s was a
noac-concentrated fold gain; v3 missed it by 2.35 s the same way, while the
noac-discounted prediction was right within 0.2 s).

This file adds NO arm.  It re-prices the arms `correct.py` already measured,
holding the `noac` rows at the base -- which for stack-5 keep-noac IS blend-5's
anchor, so the base's own quote does not move.  The correction, however, does
touch `noac` rows: RESULTS.md section 5 found 19 of them on one July day
carrying 48.2 % of the flagged days' squared error.  Under the new convention
none of that may be counted.

Two quotes per arm:

  all rows            what section 3 reported (the old convention)
  noac at the anchor  BOARD READING 3: the correction credited on the bulk
                      rows only, `noac` left where stack-5 put it

The second is the number the desk should read.
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
sys.path.insert(0, str(ROOT / "experiments" / "exp-026-rebuild" / "code"))
import correct as C  # noqa: E402
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402
import frames as F26  # noqa: E402
import ladder as L26  # noqa: E402

OFFSET = 4.28          # the desk's board offset, BOARD READING 3
QUARTER = 0.25         # BOARD READING 5: noac fold gains transfer at ~1/4
ARMS = ("step|lofo|winsor", "step-half|lofo|winsor", "stress|lofo|winsor",
        "step|loo|winsor", "linear|lofo|winsor")


def blend4_noac() -> dict:
    """blend-4's rotated prediction, the substitute `noac_discount.py` uses.

    The desk's discounted currency is not "hold noac where stack-5 put it" but
    "put blend-4 back on the noac rows", because stack-5 serves its noac rows
    from blend-5's anchor and three quarters of blend-5's gain over blend-4
    sits there.  The lever's GAIN is the same under either substitute -- the
    same vector stands on both sides of the difference -- but the LEVEL is
    260.31, not 258.04, and the desk quotes the level.
    """
    cache: dict = {}
    b4 = L26.run(L26.B4, cache)
    out = {}
    for f in F26.FOLDS:
        m = F26.fold_frame(f, L26.B5)
        m4 = F26.fold_frame(f, L26.B4)
        if not np.array_equal(m["MVT_ID_mvt"].to_numpy(), m4["MVT_ID_mvt"].to_numpy()):
            raise SystemExit(f"{f}: member sets do not align")
        out[f] = pd.DataFrame({"MVT_ID_mvt": m["MVT_ID_mvt"].to_numpy("float64"),
                               "b4": b4[f]["pred"],
                               "noac": (m["regime"].to_numpy() == "noac")})
    return out


def ranking_noac() -> list:
    """The ranking side of the same question: 388 noac rows, how many flagged?

    If none, BOARD READING 3 does not move the lever's PRICE at all -- the
    correction's population and the discounted population do not intersect --
    and the only thing it moves is the july_loo harm on the 2025 fold.
    """
    rk = D.load_ranking()[["MVT_ID_mvt", "AIRCRAFT_TYPE_mvt"]].copy()
    rk["MVT_ID_mvt"] = rk["MVT_ID_mvt"].astype("float64")
    fl = pd.read_parquet(EXP / "oof" / "ranking_flag.parquet")
    fl["MVT_ID_mvt"] = fl["MVT_ID_mvt"].astype("float64")
    m = fl.merge(rk, on="MVT_ID_mvt", how="left", validate="one_to_one")
    m["noac"] = m["AIRCRAFT_TYPE_mvt"].isna()
    out = []
    for mo, g in m.groupby("month"):
        out.append(dict(month=str(mo), rows=int(len(g)), noac=int(g["noac"].sum()),
                        flagged=int(g["flagged"].sum()),
                        flagged_noac=int((g["flagged"] & g["noac"]).sum())))
    print("\n=== the ranking side: flagged rows against the 388 noac rows ===")
    print(pd.DataFrame(out).to_string(index=False))
    tot = sum(r["flagged_noac"] for r in out)
    print(f"  -> {tot} flagged noac ranking rows, so the discount changes the "
          f"lever's price by exactly nothing on the served population")
    return out


def main() -> None:
    R = pd.read_parquet(EXP / "analysis" / "fold_rows.parquet")
    K = pd.read_parquet(EXP / "analysis" / "corrections.parquet")
    keep = ["MVT_ID_mvt", "fold"] + [f"c::{a}" for a in ARMS]
    R = R.merge(K[keep], on=["MVT_ID_mvt", "fold"], how="left", validate="one_to_one")

    B4 = blend4_noac()
    R = R.merge(pd.concat([d.assign(fold=f) for f, d in B4.items()], ignore_index=True),
                on=["MVT_ID_mvt", "fold"], how="left", validate="one_to_one")
    noac = R["noac"].to_numpy()
    print(f"[rows] {len(R):,} fold rows, {noac.sum():,} noac "
          f"({100*noac.mean():.3f} %); blend-4 joined for the discount", flush=True)
    if (R["breg"].to_numpy() == "noac").sum() != noac.sum():
        raise SystemExit("the two noac definitions disagree")

    # ---- 1. how much of each arm's movement sits on noac rows --------------
    rows = []
    for a in ARMS:
        c = R[f"c::{a}"].to_numpy("float64")
        t = c != 0.0
        rows.append(dict(arm=a, touched=int(t.sum()), touched_noac=int((t & noac).sum()),
                         sum_c=float(c[t].sum()), sum_c_noac=float(c[t & noac].sum())))
    T = pd.DataFrame(rows)
    print("\n=== how much of the correction lands on noac rows ===")
    print(T.to_string(index=False))

    # ---- 2. the two quotes, per fold ---------------------------------------
    recs = []
    for f in C.E.FOLDS:
        g = R[R["fold"] == f]
        y = g["y_true"].to_numpy("float64")
        ids = g["MVT_ID_mvt"].to_numpy("float64")
        ref = g["y_pred"].to_numpy("float64")
        nz = g["noac"].to_numpy()
        b4 = g["b4"].to_numpy("float64")
        cl = C.clusters(g)
        #: the two currencies of the base: face value, and blend-4 on noac.
        base_face = ref
        base_disc = np.where(nz, b4, ref)
        for lab, p in (("face value", base_face), ("noac on blend-4", base_disc),
                       ("quarter (BOARD READING 5)", base_face)):
            recs.append(dict(fold=f, arm="stack-5 (base)", convention=lab,
                             clean=D.rmse_clean(y, p, ids, f), gain=0.0, lo=0.0, hi=0.0))
        for a in ARMS:
            c = g[f"c::{a}"].to_numpy("float64")
            p_face = np.maximum(ref + c, B.FLOOR)
            # BOARD READING 3: blend-4 stands on the noac rows on BOTH sides,
            # so the correction is credited on the bulk rows only.
            p_disc = np.where(nz, b4, p_face)
            # BOARD READING 5 (v4 +9.26 s, v5 +0.33 s): noac fold gains
            # transfer at about a QUARTER, not at zero and not at face value.
            # So the honest quote sits a quarter of the way from discounted to
            # face: the noac rows keep a quarter of the correction.
            p_qtr = np.where(nz, np.maximum(ref + 0.25 * c, B.FLOOR), p_face)
            for lab, p, r0 in (("face value", p_face, base_face),
                               ("noac on blend-4", p_disc, base_disc),
                               ("quarter (BOARD READING 5)", p_qtr, base_face)):
                d, lo, hi = C.boot_gain(y, p, r0, cl, ids, f)
                recs.append(dict(fold=f, arm=a, convention=lab,
                                 clean=D.rmse_clean(y, p, ids, f),
                                 gain=d, lo=lo, hi=hi))
    S = pd.DataFrame(recs)

    # ---- 3. the board quote -------------------------------------------------
    out = []
    for (a, conv), s in S.groupby(["arm", "convention"], sort=False):
        per = {f: float(s[s.fold == f]["clean"].iloc[0]) for f in F26.SHARES}
        b = F26.board_estimate(per)
        out.append(dict(arm=a, convention=conv, clean_board=b, quote=b + OFFSET,
                        **{f: float(s[s.fold == f]["clean"].iloc[0]) for f in C.E.FOLDS}))
    Q = pd.DataFrame(out)
    b0 = {c: float(Q[(Q.arm == "stack-5 (base)") & (Q.convention == c)]["clean_board"].iloc[0])
          for c in Q["convention"].unique()}
    Q["board gain"] = [b0[c] - v for c, v in zip(Q["convention"], Q["clean_board"])]

    print("\n=== BOARD READING 3: clean-with-noac-at-the-anchor + 4.28 ===")
    print(Q[["arm", "convention", "july_loo", "december", "january",
             "clean_board", "quote", "board gain"]].round(4).to_string(index=False))
    print("\n=== january, day-clustered 95 % on the gain ===")
    for r in S[S.fold == "january"].itertuples():
        print(f"  {r.arm:22s} {r.convention:20s} {r.gain:+7.4f} "
              f"[{r.lo:+7.4f}, {r.hi:+7.4f}]")

    # ---- 4. the ranking side: does the correction touch a noac row at all? --
    rank = ranking_noac()

    S.to_csv(EXP / "analysis" / "board_reading3_folds.csv", index=False)
    Q.to_csv(EXP / "analysis" / "board_reading3.csv", index=False)
    (EXP / "analysis" / "board_reading3.json").write_text(json.dumps(
        {"offset": OFFSET, "noac_rows": int(noac.sum()),
         "on_noac": T.to_dict("records"),
         "ranking": rank,
         "quote": Q.round(4).to_dict("records")}, indent=2))
    print(f"\n[out] analysis/board_reading3.csv, .json, _folds.csv")


if __name__ == "__main__":
    main()
