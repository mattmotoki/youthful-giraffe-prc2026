"""Price the exp-035 candidate against K2 (v9) like for like, arm by arm.

`rotate.py` prices every arm against v6 and v7, because those are the files the
board holds.  The desk's ship rule is written against **K2**:

    a climbed member ships only if the candidate's gain over K2 has every fold
    interval clear of zero; a gain inside the rotation floor is quoted at zero.

K2 is `k2-hold-eham-out` with exp-027's own committed member
(`experiments/exp-027-step/analysis/candidate-k2.json`), and this file reads
BOTH rotations' saved row vectors and prices the difference with the same
calendar-day x airport clustered paired bootstrap the rotation uses.  Nothing is
refitted here: the two vectors are the two rotations' outputs, so the only
difference between them is the member in K2's slot.

`y_pred` in a saved vector is the arm AFTER v7's EHAM mask, `y_pred_nomask`
before it; both are checked against the two rotations' published numbers before
anything is priced.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-030-window2/code"))
sys.path.insert(0, str(ROOT / "experiments/exp-028-window/code"))
sys.path.insert(0, str(ROOT / "experiments/exp-026-rebuild/code"))
import frames as F  # noqa: E402
import panel as P  # noqa: E402
import stack5 as S  # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

FOLDS = F.FOLDS
#: `--tag` picks which rotation is priced; `rotate.py --arm <x>` writes
#: `cand-k2c-<x>-*`, so arm 3 needs no edit here.
TAG = "seeds"
CAND = EXP / "oof" / "cand-k2c-{tag}-{arm}-{fold}.parquet"
#: K2's own rotation, rerun here on 2026-09-08 so that the two arms exp-027 did
#: not save vectors for can carry an interval.  Its `k2-hold` vectors reproduce
#: exp-027's committed ones at max abs difference 0.0 and all six arm numbers
#: match `analysis/candidate-k2.json`, so it IS K2 and not a rebuild of it.
BASE = EXP / "oof" / "cand-k2c-k2base-{arm}-{fold}.parquet"
COMMITTED = ROOT / "experiments/exp-027-step/oof/exp-027-k2-{arm}-{fold}.parquet"
#: `analysis/candidate-k2.json`, exp-027 RESULTS section 16.  The gate arm is the
#: first: it is the recipe the board holds as v9.
PUB = {
    "k2-hold-eham-out": {"july_loo": 284.3195, "december": 205.2156, "january": 218.2582},
    "k2-hold": {"july_loo": 284.3938, "december": 205.2990, "january": 218.4232},
    "k2-refit-eham-out": {"july_loo": 284.2415, "december": 205.0303, "january": 218.1002},
    "k2-refit": {"july_loo": 284.3521, "december": 205.1170, "january": 218.0424},
    "k2-nowin-eham-out": {"july_loo": 284.7171, "december": 205.1063, "january": 218.7306},
    "k2-nowin": {"july_loo": 284.6968, "december": 205.1054, "january": 218.6125},
}
#: exp-027 section 12: the rotation's own refit-to-refit half width.
FLOOR = 0.1025


def ref_frames() -> dict:
    return {f: F.fold_frame(f, S.B5) for f in FOLDS}


def price(arm: str, masked: bool, ref: dict) -> dict:
    col = "y_pred" if masked else "y_pred_nomask"
    name = arm + ("-eham-out" if masked else "")
    rec = {"arm": name, "k2": {}, "cand": {}, "gain": {}, "ci": {}, "ci_no_noac": {}}
    for f in FOLDS:
        a = pd.read_parquet(str(CAND).format(tag=TAG, arm=arm, fold=f))
        b = pd.read_parquet(str(BASE).format(arm=arm, fold=f))
        c = Path(str(COMMITTED).format(arm=arm, fold=f))
        if c.exists():
            k = pd.read_parquet(c)
            m = np.abs(b[col].to_numpy("float64") - k[col].to_numpy("float64")).max()
            if m != 0.0:
                raise SystemExit(f"{arm} {f}: the reread K2 rotation differs from "
                                 f"exp-027's committed vectors by {m:.3e}")
        d = ref[f]
        ids = d["MVT_ID_mvt"].to_numpy("float64")
        for got, who in ((a, "candidate"), (b, "K2")):
            if not np.array_equal(got["MVT_ID_mvt"].to_numpy("float64"), ids):
                raise SystemExit(f"{who} {name} {f}: rows do not match the frame")
            if not np.allclose(got["y_true"].to_numpy("float64"),
                               d["y_true"].to_numpy("float64"), rtol=0, atol=0):
                raise SystemExit(f"{who} {name} {f}: y_true disagrees with the frame")
        y = d["y_true"].to_numpy("float64")
        pc = a[col].to_numpy("float64")
        pk = b[col].to_numpy("float64")
        rec["cand"][f] = round(D.rmse_clean(y, pc, ids, f), 4)
        rec["k2"][f] = round(D.rmse_clean(y, pk, ids, f), 4)
        rec["gain"][f] = round(rec["k2"][f] - rec["cand"][f], 4)
        ok = ~d["MVT_ID_mvt"].isin(D.excluded_ids(f)).to_numpy()
        nn = ok & (d["regime"].to_numpy() != "noac")
        cl = P.clusters(d)
        rec["ci"][f] = [round(x, 4) for x in
                        P.boot_delta(y[ok], pc[ok], pk[ok], pd.factorize(cl[ok])[0])]
        rec["ci_no_noac"][f] = [round(x, 4) for x in
                                P.boot_delta(y[nn], pc[nn], pk[nn], pd.factorize(cl[nn])[0])]
    rec["k2_board"] = round(P.board(rec["k2"]), 4)
    rec["cand_board"] = round(P.board(rec["cand"]), 4)
    rec["gain_board"] = round(rec["k2_board"] - rec["cand_board"], 4)
    if name in PUB:
        bad = {f: (rec["k2"][f], PUB[name][f]) for f in FOLDS
               if abs(rec["k2"][f] - PUB[name][f]) > 1e-3}
        rec["published_check"] = "ok" if not bad else bad
        if bad:
            raise SystemExit(f"{name}: the baseline vectors do not reproduce "
                             f"exp-027's published numbers: {bad}")
    return rec


def show(rec: dict) -> None:
    print(f"=== {rec['arm']} ===")
    print("  K2 (v9)   " + "  ".join(f"{f} {rec['k2'][f]:9.4f}" for f in FOLDS)
          + f"   board {rec['k2_board']:9.4f}")
    print("  candidate " + "  ".join(f"{f} {rec['cand'][f]:9.4f}" for f in FOLDS)
          + f"   board {rec['cand_board']:9.4f}")
    print("  gain      " + "  ".join(f"{f} {rec['gain'][f]:+9.4f}" for f in FOLDS)
          + f"   board {rec['gain_board']:+9.4f}")
    for f in FOLDS:
        dl, lo, hi = rec["ci"][f]
        dn, lon, hin = rec["ci_no_noac"][f]
        clear = "clear of zero" if (-hi > 0 or -lo < 0) and (-hi) * (-lo) > 0 else "CROSSES ZERO"
        print(f"    {f:9s} vs K2 {-dl:+7.3f} [{-hi:+7.3f}, {-lo:+7.3f}]  {clear:13s}"
              f"   no-noac {-dn:+7.3f} [{-hin:+7.3f}, {-lon:+7.3f}]")
    q = 0.0 if abs(rec["gain_board"]) < FLOOR else rec["gain_board"]
    print(f"  quoted board gain {q:+.4f} (rotation floor {FLOOR}; measured "
          f"{rec['gain_board']:+.4f})")


#: v9 IS `k2-hold-eham-out` (docs/SUBMISSIONS.md rows 8-9, exp-027 RESULTS s16).
#: The desk's rule is written against v9, so an arm that is not `k2-hold` must be
#: priced against v9's OWN vectors and not against its own arm's baseline: Sol
#: round 1, finding 3.  `price()` above answers "what did the member change do
#: inside this arm"; this answers "what would shipping this file do to the board".
V9_ARM, V9_MASKED = "k2-hold", True


def price_vs_v9(arm: str, masked: bool, ref: dict) -> dict:
    """The candidate arm against v9's served vectors, with v9's own intervals."""
    col = "y_pred" if masked else "y_pred_nomask"
    name = arm + ("-eham-out" if masked else "")
    rec = {"arm": name, "base": "v9 = k2-hold-eham-out", "v9": {}, "cand": {},
           "gain": {}, "ci": {}, "ci_no_noac": {}}
    for f in FOLDS:
        a = pd.read_parquet(str(CAND).format(tag=TAG, arm=arm, fold=f))
        b = pd.read_parquet(str(BASE).format(arm=V9_ARM, fold=f))
        d = ref[f]
        ids = d["MVT_ID_mvt"].to_numpy("float64")
        for got, who in ((a, "candidate"), (b, "v9")):
            if not np.array_equal(got["MVT_ID_mvt"].to_numpy("float64"), ids):
                raise SystemExit(f"{who} {name} {f}: rows do not match the frame")
        y = d["y_true"].to_numpy("float64")
        pc = a[col].to_numpy("float64")
        pv = b["y_pred" if V9_MASKED else "y_pred_nomask"].to_numpy("float64")
        rec["cand"][f] = round(D.rmse_clean(y, pc, ids, f), 4)
        rec["v9"][f] = round(D.rmse_clean(y, pv, ids, f), 4)
        rec["gain"][f] = round(rec["v9"][f] - rec["cand"][f], 4)
        ok = ~d["MVT_ID_mvt"].isin(D.excluded_ids(f)).to_numpy()
        nn = ok & (d["regime"].to_numpy() != "noac")
        cl = P.clusters(d)
        rec["ci"][f] = [round(x, 4) for x in
                        P.boot_delta(y[ok], pc[ok], pv[ok], pd.factorize(cl[ok])[0])]
        rec["ci_no_noac"][f] = [round(x, 4) for x in
                                P.boot_delta(y[nn], pc[nn], pv[nn], pd.factorize(cl[nn])[0])]
    rec["v9_board"] = round(P.board(rec["v9"]), 4)
    rec["cand_board"] = round(P.board(rec["cand"]), 4)
    rec["gain_board"] = round(rec["v9_board"] - rec["cand_board"], 4)
    rec["every_fold_clear"] = all(rec["ci"][f][1] * rec["ci"][f][2] > 0 for f in FOLDS)
    return rec


def show_v9(rec: dict) -> None:
    print(f"=== {rec['arm']} priced against v9 (k2-hold-eham-out), the desk's base ===")
    print("  v9        " + "  ".join(f"{f} {rec['v9'][f]:9.4f}" for f in FOLDS)
          + f"   board {rec['v9_board']:9.4f}")
    print("  candidate " + "  ".join(f"{f} {rec['cand'][f]:9.4f}" for f in FOLDS)
          + f"   board {rec['cand_board']:9.4f}")
    print("  gain      " + "  ".join(f"{f} {rec['gain'][f]:+9.4f}" for f in FOLDS)
          + f"   board {rec['gain_board']:+9.4f}")
    for f in FOLDS:
        dl, lo, hi = rec["ci"][f]
        clear = "clear of zero" if lo * hi > 0 else "CROSSES ZERO"
        print(f"    {f:9s} vs v9 {-dl:+7.3f} [{-hi:+7.3f}, {-lo:+7.3f}]  {clear}")
    print(f"  the desk's rule (every fold interval clear of zero): "
          f"{'MET' if rec['every_fold_clear'] else 'NOT MET'}")


def changes_spec() -> dict:
    """The two member changes RESULTS s12a compares, each as
    (label, new member file, old member file, the arm, the head column it moved).

    Sol round 2, non-blocking 5: this used to be a module-level dict, so it froze
    the DEFAULT tag at import time and `--tag mod` silently priced the seed
    member's files under the mod head's name.  It is now built after `--tag` is
    read, and `main` passes the record it builds.
    """
    return {
        "arm": (f"the arm under test ({TAG})",
                EXP / "oof" / (TAG + "-{fold}.parquet"),
                ROOT / "experiments/exp-027-step/oof/k2-window-{fold}.parquet",
                "k2-hold", "y_pred_nomask"),
        "signal": ("the change that built K2: F1-fresh's member -> the window member",
                   ROOT / "experiments/exp-027-step/oof/k2-window-{fold}.parquet",
                   ROOT / "experiments/exp-012-blend2/published/F1-fresh-s1-{fold}.parquet",
                   "k2-hold", "y_pred_v6"),
    }


def passthrough(ref: dict) -> dict:
    """Every number RESULTS s12a quotes, for BOTH member changes, in one record.

    Sol round 1, finding 2: the signal side had no command of its own and its
    json carried only slope and correlation.  Both changes now run through the
    same function, which writes the source paths and their SHA-256 as well.

    `slope` is the member's effective weight in the arm -- the regression of the
    head's move on the member's move.  `align` is the correlation of the head's
    move with the head's own residual, and `dmse = 2 E[d r] + E[d^2]` is the
    identity that turns the two into the realised RMSE change.
    """
    out = {}
    for key, (note, newp, oldp, arm, headcol) in changes_spec().items():
        rec = {"note": note, "folds": {}, "files": {}}
        for f in FOLDS:
            pn, po = Path(str(newp).format(fold=f)), Path(str(oldp).format(fold=f))
            mn = pd.read_parquet(pn).set_index("MVT_ID_mvt")["y_pred"]
            mo = pd.read_parquet(po).set_index("MVT_ID_mvt")["y_pred"]
            dm = (mn - mo.reindex(mn.index)).dropna()
            a = pd.read_parquet(str(CAND).format(tag=TAG, arm=arm, fold=f)).set_index("MVT_ID_mvt")
            b = pd.read_parquet(str(BASE).format(arm=arm, fold=f)).set_index("MVT_ID_mvt")
            new_head = a["y_pred_nomask"] if key == "arm" else b["y_pred_nomask"]
            old_head = b[headcol] if key == "signal" else b["y_pred_nomask"]
            ds = (new_head - old_head).reindex(dm.index)
            r = (old_head - b["y_true"]).reindex(dm.index)
            # Sol round 2, blocking 3: the moments below and the RMSE gains
            # beside them must be on ONE population.  `rmse_clean` drops the
            # unreached-rollover rows, so they are dropped here as well and the
            # decomposition and the gains are like for like.
            m = (np.isfinite(ds.to_numpy()) & np.isfinite(dm.to_numpy())
                 & np.isfinite(r.to_numpy())
                 & ~pd.Index(dm.index).isin(D.excluded_ids(f)))
            x, yv, rr = dm.to_numpy()[m], ds.to_numpy()[m], r.to_numpy()[m]
            ids = np.asarray(dm.index)[m]
            yt = b["y_true"].reindex(dm.index).to_numpy()[m]
            # the rows are already clean (the mask above dropped the excluded
            # ids), and `rmse_clean` refuses a row set its exclusions are not in,
            # so the RMSE is taken directly on the population the moments use.
            def _rmse(pred: np.ndarray) -> float:
                return float(np.sqrt(np.mean((yt - pred) ** 2)))
            rmse_old = _rmse(old_head.reindex(dm.index).to_numpy()[m])
            rmse_new = _rmse(new_head.reindex(dm.index).to_numpy()[m])
            mo_r = _rmse(mo.reindex(dm.index).to_numpy()[m])
            mn_r = _rmse(mn.reindex(dm.index).to_numpy()[m])
            rec["folds"][f] = dict(
                n=int(m.sum()), population="clean (unreached-rollover rows dropped)",
                slope=round(float(np.dot(x, yv) / np.dot(x, x)), 4),
                corr_member_head=round(float(np.corrcoef(x, yv)[0, 1]), 4),
                sd_member=round(float(x.std()), 3), sd_head=round(float(yv.std()), 3),
                align_head_resid=round(float(np.mean(yv * rr) / (yv.std() * rr.std())), 4),
                E_d_r=round(float(np.mean(yv * rr)), 3),
                E_d2=round(float(np.mean(yv * yv)), 3),
                dmse=round(float(2 * np.mean(yv * rr) + np.mean(yv * yv)), 3),
                member_gain=round(mo_r - mn_r, 4), head_gain=round(rmse_old - rmse_new, 4),
                transfer=round((rmse_old - rmse_new) / (mo_r - mn_r), 4) if mo_r != mn_r else None)
            rec["files"][f] = {"new_member": [str(pn.relative_to(ROOT)), D.sha256(pn)],
                               "old_member": [str(po.relative_to(ROOT)), D.sha256(po)]}
        out[key] = rec
        print(f"\n  {note}")
        print(f"  {'fold':10s} {'slope':>7s} {'sd(mem)':>8s} {'sd(head)':>9s} "
              f"{'align':>7s} {'E[dr]':>9s} {'E[d^2]':>8s} {'member':>8s} {'head':>8s} {'transfer':>9s}")
        for f in FOLDS:
            v = rec["folds"][f]
            print(f"  {f:10s} {v['slope']:+7.4f} {v['sd_member']:8.2f} {v['sd_head']:9.2f} "
                  f"{v['align_head_resid']:+7.4f} {v['E_d_r']:+9.2f} {v['E_d2']:8.2f} "
                  f"{v['member_gain']:+8.4f} {v['head_gain']:+8.4f} "
                  f"{(str(round(100 * v['transfer'], 1)) + ' %') if v['transfer'] else '-':>9s}")
    return out


def member_passthrough(ref: dict) -> dict:
    """How much of the member's change reaches the stack, per row.

    The slope of `stack_new - stack_old` on `member_new - member_old` is the
    member's effective weight in the arm.  A pure variance reduction enters the
    stack's MSE at that weight SQUARED, which is the whole story of this arm.
    """
    out = {}
    for f in FOLDS:
        km = pd.read_parquet(ROOT / f"experiments/exp-027-step/oof/k2-window-{f}.parquet")
        sm = pd.read_parquet(EXP / f"oof/{TAG}-{f}.parquet")
        km = km.set_index("MVT_ID_mvt")["y_pred"]
        sm = sm.set_index("MVT_ID_mvt")["y_pred"]
        dm = (sm - km.reindex(sm.index)).dropna()
        a = pd.read_parquet(str(CAND).format(tag=TAG, arm="k2-hold", fold=f)).set_index("MVT_ID_mvt")
        b = pd.read_parquet(str(BASE).format(arm="k2-hold", fold=f)).set_index("MVT_ID_mvt")
        ds = (a["y_pred_nomask"] - b["y_pred_nomask"]).reindex(dm.index)
        m = np.isfinite(ds.to_numpy()) & np.isfinite(dm.to_numpy())
        x, yv = dm.to_numpy()[m], ds.to_numpy()[m]
        slope = float(np.dot(x, yv) / np.dot(x, x))
        out[f] = dict(n=int(m.sum()), slope=round(slope, 4),
                      slope_sq=round(slope ** 2, 4),
                      sd_member=round(float(x.std()), 3),
                      sd_stack=round(float(yv.std()), 3),
                      corr=round(float(np.corrcoef(x, yv)[0, 1]), 4))
        print(f"  {f:9s} n={out[f]['n']:7d} slope={slope:+.4f} slope^2={slope**2:.4f} "
              f"corr={out[f]['corr']:+.4f} sd(member)={x.std():.2f} sd(stack)={yv.std():.2f}")
    return out


def other_arms() -> dict:
    """The four arms exp-027 did not save row vectors for, from the two rotations'
    own jsons.  Scalars only: no interval can be built without the rows."""
    k = {a["arm"]: a for a in json.loads(
        (ROOT / "experiments/exp-027-step/analysis/candidate-k2.json").read_text())["arms"]}
    c = {a["arm"]: a for a in json.loads(
        (EXP / f"analysis/candidate-k2c-{TAG}.json").read_text())["arms"]}
    out = {}
    print("\n=== every arm, board clean, from the two rotations' jsons ===")
    for name in k:
        if name not in c:
            continue
        g = round(k[name]["clean_board"] - c[name]["clean_board"], 4)
        out[name] = dict(k2=k[name]["clean_board"], cand=c[name]["clean_board"], gain=g,
                         folds={f: round(k[name]["clean"][f] - c[name]["clean"][f], 4)
                                for f in FOLDS})
        mark = "  <- v9, the gate arm" if name == "k2-hold-eham-out" else ""
        print(f"  {name:20s} K2 {k[name]['clean_board']:9.4f}  cand {c[name]['clean_board']:9.4f}"
              f"  gain {g:+8.4f}   folds "
              + " ".join(f"{out[name]['folds'][f]:+7.4f}" for f in FOLDS) + mark)
    return out


def main() -> None:
    global TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="seeds",
                    help="the rotation to price: rotate.py --arm <tag>")
    TAG = ap.parse_args().tag
    print(f"[price] rotation tag {TAG}")
    ref = ref_frames()
    recs = [price("k2-hold", True, ref), price("k2-hold", False, ref),
            price("k2-refit", True, ref), price("k2-refit", False, ref),
            price("k2-nowin", True, ref)]
    for r in recs:
        show(r)
    scalars = other_arms()
    print("\n=== every arm priced against v9 itself, the desk's base ===")
    v9 = [price_vs_v9(a, True, ref) for a in ("k2-hold", "k2-refit", "k2-nowin")]
    for r in v9:
        show_v9(r)
    print("\n=== member pass-through (k2-hold, before the mask) ===")
    pt = member_passthrough(ref)
    print("\n=== the two member changes s12a compares, same instrument ===")
    ch = passthrough(ref)
    #: `seeds` keeps the tag-less name: two ledger rows and RESULTS s12 cite it,
    #: and the ledger is append-only, so its citations must not go dangling.
    out = EXP / "analysis" / ("price-vs-k2.json" if TAG == "seeds"
                              else f"price-vs-k2-{TAG}.json")
    out.write_text(json.dumps(dict(arms=recs, vs_v9=v9, other_arms=scalars,
                                   passthrough=pt, changes=ch, floor=FLOOR), indent=1))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
