"""Q02 -- the weather member replaced by the 64-column bulk member, against a sham refit.

    PRC_THREADS=4 python q02.py

**The reading taken, and the readings rejected.**  Q02's first test is "weather-
member to bulk64 substitution, which is unmeasured.  Compare unchanged refit,
addition and replacement on identical data with the same seed, search budget and
fully refitted blend/head; retain airport masks.  Evaluate all final vectors
against the v10 replay and sham control."

*The weather member is `V3-allin`.*  `docs/BACKLOG.md` line 15 names its frame's
weather family ("the v3 frame's weather family is built from the ERA5 event-hour
snapshot"), and it is the only member of the shipped seven whose inputs are
external weather.  `D1-winter` was checked and rejected as the referent: its
shipped `deice` arm carries **no external weather** (`docs/CLOUD-COMPUTE.md`
line 291), so substituting it would not answer Q02's question.

*"Fully refitted blend/head" is refit at BOTH stages, on the arm's own members.*
The blend rule (`exp-026/code/rule.fit_hybrid`) cannot carry a partial general
member -- `_split` treats only the `N1-nmtail` specialist's nulls -- and
`B1-bulk64` is partial by construction (there is no `nm_taxi` without an NM
join).  So an arm's ANCHOR is fitted on its full members and its HEAD on all of
them, and the replacement arm's anchor is therefore six members where v10's is
seven.  That confound is not hidden: `replacement-anchorheld` runs the same head
on v10's own anchor, and `drop-weather` runs the anchor change with no bulk
member at all, so the three numbers separate the head effect from the anchor
effect.

*The insertion point is v10's, not K2's.*  v10 = the mixed anchor of
`exp-027/code/refit_bulk.py` (blend-K off the NM tail, blend-5 on it) + the 600 s
`depx` stacker + `noac` held at blend-5's anchor + v7's EHAM mask + exp-036's
eight-airport mask.  Every arm here is that recipe with one thing changed, and
the mask list is exp-036's own, applied unchanged.  Two arms are also run on the
blend-5 "hold" anchor so that the gates against the committed K2 and against
exp-034's `bulk-add-hold` are exact.

*The sham refit control is the same member set at a different seed draw of the
same size and budget.*  Q02's advance clause is "a stable paired gain over the
sham refit"; a control that changes nothing at all measures nothing, so the
control re-rolls the only thing an insertion is allowed to re-roll.

Rejected: (a) selecting a new mask per arm -- the assignment says the masks are
retained; (b) refitting the members -- `assignments/packet.md` forbids base fits;
(c) treating exp-034's committed `bulk-add-hold` as the addition arm -- it sits
on the blend-5 anchor and is not matched to v10's, so it is run again here on
v10's recipe and the committed one is used as a gate instead.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from itertools import product

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp-034-bulk/code"))
import align as A  # noqa: E402
import candidate as C  # noqa: E402   (exp-034's harness: frames, anchors, rotate, masks)

F, R, S, SW, Sc = C.F, C.R, C.S, C.SW, C.Sc
B, D = C.B, C.D
FOLDS = C.FOLDS
THREADS = int(os.environ.get("PRC_THREADS", "4"))
SW.THREADS = THREADS

WEATHER = "V3-allin"
BULK = C.BULK                       # B1-bulk64
NAMES5 = C.NAMES5                   # blend-5's committed seven
NAMESK2 = C.NAMESK2                 # v10/K2's seven (K2-window in F1-fresh's slot)
NAMES_ADD = C.NAMESB                # the eight of exp-034's addition
NAMES_SUB = tuple(BULK if n == WEATHER else n for n in NAMESK2)
NAMES_DROP = tuple(n for n in NAMESK2 if n != WEATHER)
#: The sham refit's draws: five seeds each, same count and same 300 rounds as the
#: committed draw.  Q02's advance clause is "a STABLE paired gain over the sham
#: refit", and one control draw is one realisation and not a standard error, so
#: four are run and the head's seed spread is estimated rather than asserted.
SEEDS0 = S.SEEDS
SEEDS1 = (13, 113, 213, 313, 413)
SHAMS = {"sham refit (draw 2)": (23, 123, 223, 323, 423),
         "sham refit (draw 3)": (31, 131, 231, 331, 431),
         "sham refit (draw 4)": (41, 141, 241, 341, 441)}

PUB = {"K2": {"july_loo": 284.3195, "december": 205.2156, "january": 218.2582},
       "bulk-add-hold": {"july_loo": 284.3591, "december": 205.1995, "january": 218.2214},
       "refit-bulk": {"july_loo": 284.1774, "december": 204.9052, "january": 218.2115},
       "v10": {"july_loo": 284.1572, "december": 204.9218, "january": 218.1506}}
GATE = 1e-3
MASK = json.loads((ROOT / "experiments/exp-036-mask/analysis/mask.json").read_text())


def full_only(names) -> tuple:
    """The arm's members the blend rule can carry: the partial bulk columns out."""
    return tuple(n for n in names if n not in (C.BULK, C.BULK_ONLY))


PARTIAL = (C.BULK, C.BULK_ONLY)


def defined(m: pd.DataFrame, names) -> np.ndarray:
    """Rows where every PARTIAL member of `names` is finite."""
    part = [n for n in names if n in PARTIAL]
    if not part:
        return np.ones(len(m), bool)
    return np.all(np.isfinite(np.stack([m[n].to_numpy("float64") for n in part], 1)), 1)


def fit_avail(fit: pd.DataFrame, names) -> dict:
    """The shipped hybrid rule extended so a PARTIAL general member can enter.

    `rule.apply_hybrid` sends a null member's column to 0.0, which is right for
    the `nm_late` specialist (it is fitted only where it exists) and wrong for a
    general member defined on 98 % of rows: the other 2 % would be served a
    weight vector that is short by that member's whole contribution.  So the
    rule is fitted TWICE per regime block -- once on the rows where the partial
    members exist, with them in the member list, and once on the rows where they
    do not, without them -- and a row is served by the fit that matches its own
    availability.  With no partial member in `names` the two fits are the same
    NNLS on the same rows and this reduces to `rule.fit_hybrid` exactly.
    """
    names = list(names)
    keepable = list(full_only(names))
    ok = defined(fit, names)
    return {"with": R.fit_hybrid(fit[ok], names) if ok.any() else None,
            "without": R.fit_hybrid(fit[~ok] if (~ok).any() else fit, keepable),
            "names": names, "keepable": keepable}


def apply_avail(m: pd.DataFrame, W: dict) -> np.ndarray:
    ok = defined(m, W["names"])
    out = R.apply_hybrid(m, W["keepable"], W["without"])
    if W["with"] is not None and ok.any():
        out = np.where(ok, R.apply_hybrid(m, W["names"], W["with"]), out)
    return out


def anchor_for(data: dict, names, kind: str, a5: dict, anames=None) -> dict:
    """`hold` = blend-5 on the committed seven; `mixed` = v10's own construction;
    `avail` = v10's construction with the availability-aware rule, so a partial
    general member reaches the ANCHOR and not only the head (Q02's "fully
    refitted blend/head").  `anames` fits the anchor on a member list of its own,
    which is what the V3 factorial in section 4 needs."""
    if kind == "hold":
        return a5
    src_names = list(anames if anames is not None else names)
    nm = list(full_only(src_names))
    out = {}
    for g in FOLDS:
        src = pd.concat([data[h] for h in FOLDS if h != g], ignore_index=True)
        if kind == "avail":
            W5 = fit_avail(src, NAMES5)
            WK = fit_avail(src, src_names)
            aa5 = apply_avail(data[g], W5)
            aaK = apply_avail(data[g], WK)
        else:
            W5 = R.fit_hybrid(src, NAMES5)
            WK = R.fit_hybrid(src, nm)
            aa5 = R.apply_hybrid(data[g], NAMES5, W5)
            aaK = R.apply_hybrid(data[g], nm, WK)
        _, late = R._split(data[g], nm)
        out[g] = np.where(late, aa5, aaK)
    return out


def rotate_seeds(data, names, anchor, blk, noac_ref, seeds):
    """`exp-034 candidate.rotate` with the seed draw passed in."""
    out = {}
    for f in FOLDS:
        ap = data[f]
        fit = pd.concat([data[g] for g in FOLDS if g != f], ignore_index=True)
        a_fit = np.concatenate([anchor[g] for g in FOLDS if g != f])
        xf = pd.concat([blk[g] for g in FOLDS if g != f], ignore_index=True)
        bag = np.maximum(np.mean([SW.arm(fit, ap, s, a_fit, anchor[f], names, xf, blk[f])
                                  for s in seeds], axis=0), B.FLOOR)
        out[f] = np.where(ap["regime"].to_numpy() == "noac", noac_ref[f], bag)
    return out


def spread(recs, T) -> dict:
    """The head's own seed spread, as the MATCHED design it is.

    Five seed draws are shared by the seven-member head and the eight-member
    head: draw k differs between them only by `B1-bulk64`, so the five
    DIFFERENCES are the paired quantity and an unpaired two-sample test throws
    that pairing away (Sol round 1, blocking 11, which also caught the earlier
    version calling 25 cross-group comparisons "paired").  Both endpoints are
    reported, the contract's PRIMARY all-row one first.

    The uncertainty here is over SEEDS only.  It is a different question from
    the day-block interval beside each arm, and the two are never added.
    """
    def board_of(a):
        return next(r for r in recs if r["arm"] == a)["clean_board"]

    base = T[T.base == "unchanged (v10 replay)"].set_index("arm")

    def gain(a, col):
        return float(base.loc[a, col])

    seven = ["unchanged (v10 replay)", "sham refit (seed draw)"] + list(SHAMS)
    eight = ["addition", "addition (seed draw)"] + [n.replace("sham refit", "addition")
                                                    for n in SHAMS]
    seeds = [list(SEEDS0), list(SEEDS1)] + [list(v) for v in SHAMS.values()]
    b7 = [board_of(a) for a in seven]
    b8 = [board_of(a) for a in eight]

    out = {"draws": [{"seeds": sd, "seven_member_arm": a7, "eight_member_arm": a8,
                      "clean_board_7": x7, "clean_board_8": x8}
                     for sd, a7, a8, x7, x8 in zip(seeds, seven, eight, b7, b8)],
           "mean_clean_board_7": float(np.mean(b7)), "sd_clean_board_7": float(np.std(b7, ddof=1)),
           "mean_clean_board_8": float(np.mean(b8)), "sd_clean_board_8": float(np.std(b8, ddof=1))}

    for label, col in (("allrow_primary", "gain_board_allrow"),
                       ("clean", "gain_board_clean")):
        diff = np.array([gain(a8, col) - gain(a7, col) for a7, a8 in zip(seven, eight)])
        m = float(diff.mean())
        sd = float(diff.std(ddof=1))
        se = sd / np.sqrt(len(diff))
        # exact one-sided sign-flip test: every assignment of signs to the five
        # matched differences is equally likely under the null of no effect
        signs = np.array(list(product((1, -1), repeat=len(diff))))
        null = (signs * diff).mean(axis=1)
        pv = float((null >= m).mean())
        out[label] = {"matched_differences": [round(float(x), 6) for x in diff],
                      "mean": m, "sd": sd, "paired_se": float(se),
                      "t": m / se if se > 0 else None,
                      "wins": int((diff > 0).sum()), "n": int(len(diff)),
                      "exact_one_sided_sign_flip_p": pv}
    out["seeds"] = {"committed": list(SEEDS0), "draw 2": list(SEEDS1),
                    **{k: list(v) for k, v in SHAMS.items()}}
    out["note"] = ("five matched five-seed draws of each head, everything else identical "
                   "(v10's mixed anchor, 600 s depx block, noac held, EHAM mask, exp-036's "
                   "eight-airport mask). The sham refit control IS the seven-member "
                   "distribution; the addition's gain is the mean of the five PAIRED "
                   "differences and the standard error is the paired one. Seed uncertainty "
                   "only: it is not added to the day-block interval")
    (EXP / "analysis" / "q02-seed-spread.json").write_text(json.dumps(out, indent=2))
    print("\n=== five MATCHED five-seed draws of each head ===")
    for d in out["draws"]:
        print(f"  seeds {str(d['seeds']):26s} seven {d['clean_board_7']:9.4f}   "
              f"eight {d['clean_board_8']:9.4f}")
    for label in ("allrow_primary", "clean"):
        o = out[label]
        print(f"  {label:15s} mean {o['mean']:+.4f}  paired se {o['paired_se']:.4f}  "
              f"t {o['t']:+.2f}  wins {o['wins']}/{o['n']}  "
              f"sign-flip p {o['exact_one_sided_sign_flip_p']:.5f}")
    return out


def main() -> None:
    t0 = time.time()
    data = C.frames()
    ref = Sc.reference()
    a5 = C.anchors(data, NAMES5)
    for f in FOLDS:
        y = data[f]["y_true"].to_numpy("float64")
        d = float(np.sqrt(np.mean((y - a5[f]) ** 2))) - S.PUBLISHED[f]
        if abs(d) > GATE:
            raise SystemExit(f"gate A (blend-5) fails on {f}: {d:+.4f}")
    print("[q02] gate A: blend-5 reproduces on all three folds", flush=True)
    ref5 = {f: np.maximum(a5[f], B.FLOOR) for f in FOLDS}
    blk = {f: SW.block_for(f, SW.panel_of(data[f]), {"blocks": [C.BLOCK], "win": C.WIN})
           for f in FOLDS}
    v7 = C.eham_mask({f: ref[f]["y_pred_v6"].to_numpy("float64") for f in FOLDS}, ref)[0]
    apt = {f: ref[f]["airport"].astype(str).to_numpy() for f in FOLDS}
    keep = MASK["keep"]
    y = {f: ref[f]["y_true"].to_numpy("float64") for f in FOLDS}
    ids = {f: ref[f]["MVT_ID_mvt"].to_numpy("float64") for f in FOLDS}

    ARMS = [
        # name                     members      anchor   seeds    mask8  gate
        ("gate-k2-hold",           NAMESK2,     "hold",  SEEDS0,  False, "K2"),
        ("gate-bulk-add-hold",     NAMES_ADD,   "hold",  SEEDS0,  False, "bulk-add-hold"),
        ("unchanged (v10 replay)", NAMESK2,     "mixed", SEEDS0,  True,  "v10"),
        ("sham refit (seed draw)", NAMESK2,     "mixed", SEEDS1,  True,  None),
        ("addition",               NAMES_ADD,   "mixed", SEEDS0,  True,  None),
        ("replacement",            NAMES_SUB,   "mixed", SEEDS0,  True,  None),
        ("replacement (sham seed)", NAMES_SUB,  "mixed", SEEDS1,  True,  None),
        ("replacement-anchorheld", NAMES_SUB,   "v10anchor", SEEDS0, True, None),
        ("drop-weather",           NAMES_DROP,  "mixed", SEEDS0,  True,  None),
        ("addition-holdanchor",    NAMES_ADD,   "hold",  SEEDS0,  True,  None),
        ("replacement-holdanchor", NAMES_SUB,   "hold",  SEEDS0,  True,  None),
    ] + [(n, NAMESK2, "mixed", sd, True, None) for n, sd in SHAMS.items()] + [
        # The addition at the SAME five draws as the unchanged head, so the two
        # are compared as distributions and not as two single realisations.
        ("addition (seed draw)", NAMES_ADD, "mixed", SEEDS1, True, None),
    ] + [(n.replace("sham refit", "addition"), NAMES_ADD, "mixed", sd, True, None)
         for n, sd in SHAMS.items()] + [
        # Sol round 1, blocking 9: the contracted substitution needs the bulk
        # member in the ANCHOR too, which the shipped rule cannot carry.  These
        # three run on the availability-aware rule; the first is the gate, and
        # it must land on v10 because a seven-member list has no partial member.
        ("unchanged (avail rule)",     NAMESK2,    "avail", SEEDS0, True, "v10"),
        ("addition (avail anchor)",    NAMES_ADD,  "avail", SEEDS0, True, None),
        ("replacement (avail anchor)", NAMES_SUB,  "avail", SEEDS0, True, None),
        # Sol round 1, blocking 10: the V3 factorial.  `drop-weather` moved the
        # head AND the anchor together, so it cannot allocate its own loss.
        ("drop-weather, head only",    NAMES_DROP, "v10anchor", SEEDS0, True, None),
        ("drop-weather, anchor only",  NAMESK2,    "mixed", SEEDS0, True, None),
    ]
    v10_anchor = anchor_for(data, NAMESK2, "mixed", a5)
    #: the anchor of the "anchor only" arm is fitted on the six-member list while
    #: its head keeps v10's seven -- the other cell of blocking 10's factorial.
    ANCHOR_NAMES = {"drop-weather, anchor only": NAMES_DROP}
    vecs, recs = {}, []
    for name, names, akind, seeds, mask8, gate in ARMS:
        t = time.time()
        anc = (v10_anchor if akind == "v10anchor"
               else anchor_for(data, names, akind, a5, ANCHOR_NAMES.get(name)))
        p = C.align(rotate_seeds(data, names, anc, blk, ref5, seeds), data, ref)
        pe = C.eham_mask(p, ref)[0]
        if mask8:
            pe = {f: np.where(np.isin(apt[f], keep), pe[f], v7[f]) for f in FOLDS}
        clean = {f: D.rmse_clean(y[f], pe[f], ids[f], f) for f in FOLDS}
        rec = {"arm": name, "members": list(names), "n_members": len(names),
               "anchor": akind,
               "anchor_members": (list(NAMES5) if akind == "hold" else
                                  list(NAMESK2) if akind == "v10anchor" else
                                  list(ANCHOR_NAMES[name]) if name in ANCHOR_NAMES else
                                  list(names) if akind == "avail" else
                                  list(full_only(names))),
               "anchor_rule": ("blend-5" if akind == "hold" else
                               "v10's mixed anchor" if akind == "v10anchor" else
                               "mixed, availability-aware" if akind == "avail" else
                               "mixed, partial members dropped"),
               "seeds": list(seeds), "mask8": mask8,
               "clean": {f: round(clean[f], 4) for f in FOLDS},
               "clean_board": round(Sc.P.board(clean), 4),
               "wall_s": round(time.time() - t, 1)}
        if gate:
            off = {f: round(clean[f] - PUB[gate][f], 4) for f in FOLDS}
            rec["gate_against"] = gate
            rec["gate_offset"] = off
            rec["gate_pass"] = bool(max(abs(v) for v in off.values()) <= GATE)
            if not rec["gate_pass"]:
                raise SystemExit(f"{name} does not reproduce {gate}: {off}")
        vecs[name] = pe
        recs.append(rec)
        print(f"[{name:24s}] " + "  ".join(f"{f} {clean[f]:9.4f}" for f in FOLDS)
              + f"  board {rec['clean_board']:9.4f}  {rec['wall_s']:.0f}s"
              + (f"  GATE {gate} {off}" if gate else ""), flush=True)

    # ---- every arm priced against the v10 replay AND against the sham -------
    P = {f: {"blk1": A.blocks(A.day_of(ref[f]["t"].to_numpy("float64")), 1),
             "clean": ~ref[f]["MVT_ID_mvt"].isin(D.excluded_ids(f)).to_numpy()}
         for f in FOLDS}
    rows = []
    for base_name in ("unchanged (v10 replay)", "sham refit (seed draw)"):
        p10 = vecs[base_name]
        for name, p in vecs.items():
            st_all, st_cl, parts = {}, {}, {}
            for f in FOLDS:
                pr = A.per_row(y[f], p10[f], p[f])
                st_all[f] = A.fold_stats(pr["r"], pr["d"], np.ones(len(pr["d"]), bool))
                st_cl[f] = A.fold_stats(pr["r"], pr["d"], P[f]["clean"])
                e10 = pr["r"] ** 2
                parts[f] = (P[f]["blk1"], e10, (pr["r"] - pr["d"]) ** 2)
            b = A.board_gain(st_all)
            bc = A.board_gain(st_cl)
            ci = A.boot_board(parts, 2000)
            rows.append({"base": base_name, "arm": name,
                         "gain_board_allrow": b["gain_board"],
                         "ci_lo": ci[0], "ci_hi": ci[1],
                         "gain_board_clean": bc["gain_board"],
                         "gain_rmse_december": st_all["december"]["gain_rmse"],
                         **{f"gain_rmse_{f}": st_all[f]["gain_rmse"] for f in FOLDS},
                         "n_changed_july": st_all["july_loo"]["n_changed"],
                         "A_july": st_all["july_loo"]["A"], "B_july": st_all["july_loo"]["B"],
                         "scale_july": st_all["july_loo"]["scale"]})
    T = pd.DataFrame(rows)
    T.to_csv(EXP / "analysis" / "q02-table.csv", index=False)
    spread(recs, T)
    (EXP / "analysis" / "q02-arms.json").write_text(json.dumps(
        {"arms": recs, "weather_member": WEATHER, "bulk_member": BULK,
         "mask": MASK, "seeds_committed": list(SEEDS0), "seeds_sham": list(SEEDS1),
         "rounds": S.ROUNDS, "block": C.BLOCK, "win": C.WIN, "threads": THREADS,
         "wall_s": round(time.time() - t0, 1)}, indent=2))
    (EXP / "packet").mkdir(exist_ok=True)
    for f in FOLDS:
        out = pd.DataFrame({"MVT_ID_mvt": ids[f], "y_true": y[f]})
        for n, p in vecs.items():
            out[n] = p[f]
        out.to_parquet(EXP / "packet" / f"q02-{f}.parquet", index=False, compression="zstd")

    pd.set_option("display.width", 220)
    print("\n=== every arm against the v10 replay and against the sham refit ===")
    print(T.round(4).to_string(index=False))
    print(f"\n[q02] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    if "--spread-only" in sys.argv:
        # recompute the seed-spread block from the committed arms record; every
        # number in it is a clean board this file already wrote.
        spread(json.loads((EXP / "analysis" / "q02-arms.json").read_text())["arms"],
               pd.read_csv(EXP / "analysis" / "q02-table.csv"))
    else:
        main()
