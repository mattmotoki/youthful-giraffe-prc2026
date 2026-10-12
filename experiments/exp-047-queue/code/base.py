"""exp-047-queue: the fold-side v11 base, the contract, and the shared-component
estimator this lane reads its answer off.

Nothing is refitted anywhere in this experiment.  The base is arithmetic over two
gated vectors, exactly as `docs/SUBMISSIONS.md` row 11 describes the board head:

    p10 = exp-039 `v10.load()`   (exp-036 arm 2; gated on the uploaded file's
                                  SHA-256 and on 284.1572 / 204.9218 / 218.1506)
    p11 = p10 + S5_AMP * (stack-5 - p10),   S5_AMP = 0.217

`exp-046-surfacefit/code/base.py` does the same arithmetic.  This file re-derives it
rather than importing that one, for two reasons stated in `SPEC.md`: the sibling lane
is being edited by another worker while this one runs, and `assignments/queue.md`
forbids this lane from touching exp-046's files -- so a shared import would make this
lane's numbers depend on a file it may not fix.  The gate is the published clean board
256.91, so the two cannot silently disagree.

The contract is `exp-038-packet/RESULTS.md` §1, unchanged:

    r = y - p11,  d = p_arm - p11,  A = mean(r*d),  B = mean(d*d),  gain = 2A - B

The shared-component estimator is `exp-041-cell/code/anatomy.py`'s: the self-excluded
cross-product `sum_i r_i * (sum of the other rows' residuals in the cell) / sum_i
(their count)`, which is unbiased for `E[r_i r_j]` over the pairs a relation names and
carries no `Var(eps)/n_c` term.  It is imported from that experiment, not re-written.

    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-047-queue/code/base.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
from imports import load as _load                            # noqa: E402

V = _load("exp-039-screens", "v10")
C41 = _load("exp-041-cell", "base", ("exp-039-screens",))    # relation helpers
V38 = _load("exp-038-packet", "vectors")                     # the gated fold vectors
sys.path.insert(0, str(ROOT))
import prc.data as D                                         # noqa: E402

FOLDS = ["july_loo", "december", "january"]
SHARES = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621}
S5_AMP = 0.217                     # exp-038 §2c, the scalar v11 was served at
V11_BOARD_CLEAN = 256.91           # docs/SUBMISSIONS.md row 11
GATE = 0.01

# ---- v12, the board head from 2026-09-08 14:50 UTC (tail rule) ---------------------
#: v11 with ONLY its K2 member replaced by `exp-035-k2climb` arm 3b's climbed member,
#: served sha `48dd572d…`, board 262.7418.  Its fold side is `exp-035` §26a's own two
#: lines applied to the climbed head, which is what `exp-035/code/price_vs_v11.py`
#: calls form 2': mask at `exp-036-mask`'s eight kept airports with v7 elsewhere, then
#: the same 0.217 toward stack-5.  Nothing here is fitted; every input is a gated
#: vector on disk.  The gate is that the SAME two lines rebuild v11 itself from the
#: unclimbed base, and that the climbed head reads its own published fold RMSEs.
V12_HEAD_OOF = ROOT / "experiments/exp-035-k2climb/oof/cand-k2c-mod-refit-bulk-{fold}.parquet"
#: `exp-035/code/price_vs_v11.py` `CAND_RB_PUBLISHED`
V12_HEAD_RMSE = {"july_loo": 283.6627, "december": 204.2511}
#: `exp-035` §26c: +0.4460 clean board over v11, so 256.9104 - 0.4460
V12_BOARD_CLEAN = 256.4644
V38_BASE = "v10 (exp-036 refitbulk-mask)"

# ---- predeclared, before any number of this experiment was looked at ---------
BIN = 600                          # the cell's UTC window, seconds (Q03/Q04's grid)
BLOCKS = 5                         # contiguous day blocks for every cross fit
SEED = 47
BOOT = 2000
TREES, DEPTH, LR = 300, 3, 0.05    # exp-041 T7's learner, unchanged

MONTH_OF = {"july_loo": "2025-07", "december": "2025-12", "january": "2025-01"}

#: ONE PREFIX MAP IN BOTH LANES, AND THE REGISTER OWNS THE MONTH LIST (desk, 2026-09-09).
#: The airline prefix map turns a null `CALLSIGN_flt` into a reconstructed broadcast
#: callsign, so it is an INPUT to the matcher and it decides the class of the 1.5 % of
#: rows whose callsign is missing.  Both surface lanes must build it from the same rows
#: or their class vectors differ and their two updates overlap: the desk measured 30
#: ranking rows where the lanes' `obs_has_clock` disagreed, every one with a null
#: `CALLSIGN_flt`, because this lane took *the first three months of `D.MONTHS` other
#: than the built one, every phase* and `exp-046-surfacefit` took *the nine 2025 months
#: that are not fold months, departures only*.  Nine is right on its own merits (Sol
#: round 1, blocking 6): excluding only the built month left the map for a FITTING month
#: learned partly on the SCORED month's rows.
#:
#: The list is READ FROM THE REGISTER, `docs/EXTERNAL-DATA.md` (desk, 2026-09-09:
#: *"build your prefix frame from that line, not from D.MONTHS"*), and not derived here.
#: Deriving it is what let the two lanes drift: two derivations of "the months that are
#: not fold months" agreed only as long as both spellings agreed.  One written line, two
#: readers, and a reader that cannot find it stops.
REGISTER = ROOT / "docs/EXTERNAL-DATA.md"
PREFIX_CLAUSE = "airline prefix map learned on the nine 2025 months that are"


def _months_from_register() -> list[str]:
    """The nine prefix-map months, parsed out of `docs/EXTERNAL-DATA.md`.

    The register writes them as ranges -- "(2025-02 to 2025-06 and 2025-08 to 2025-11)"
    -- so the ranges are expanded and the result is checked three ways: nine months, all
    of them real training months, none of them a fold month.  Any other shape is a
    changed register and stops the lane rather than silently reverting to a derived list.
    """
    txt = REGISTER.read_text()
    if PREFIX_CLAUSE not in txt:
        raise SystemExit(f"base: {REGISTER} no longer carries the prefix-map clause "
                         f"'{PREFIX_CLAUSE}'; the register moved and this lane must be "
                         "pointed at the new line rather than guess the months")
    tail = txt[txt.index(PREFIX_CLAUSE):]
    span = tail[tail.index("(") + 1:tail.index(")")]
    got: list[str] = []
    for part in re.split(r"\band\b|,", span):
        ms = re.findall(r"20\d\d-\d\d", part)
        if len(ms) == 2 and " to " in part:
            i0, i1 = D.MONTHS.index(ms[0]), D.MONTHS.index(ms[1])
            got += list(D.MONTHS[i0:i1 + 1])
        else:
            got += ms
    folds = set(MONTH_OF.values())
    bad = [m for m in got if m not in D.MONTHS] + [m for m in got if m in folds]
    if len(got) != 9 or len(set(got)) != 9 or bad:
        raise SystemExit(f"base: the register's prefix-map months read as {got} "
                         f"({len(got)} months, {len(set(got))} distinct, bad {bad}); "
                         "expected nine DISTINCT training months, none of them a fold "
                         "month")
    return got


PREFIX_MONTHS = _months_from_register()

#: AND THE OTHER READER HAS TO AGREE (FALLBACK round 4, item 5).  "One written line, two
#: readers" had one reader: `exp-046-surfacefit/code/arms.py:prefix_months_for` still
#: derives its list from `D.MONTHS`, so an edit to the register would move this lane
#: alone and the class boundary would drift again.  Surfacefit publishes the list it
#: actually used in `analysis/class-manifest.json`, so this compares the two and stops
#: rather than serving a mask built from a different map.  The check is skipped only if
#: that manifest does not exist yet, which is the case before that lane's first publish.
CLASS_MANIFEST = ROOT / "experiments/exp-046-surfacefit/analysis/class-manifest.json"


def check_prefix_agrees(manifest: Path | None = None) -> dict:
    """`PREFIX_MONTHS` against the months surfacefit says it built its map from.

    `manifest` is the class manifest the CALLER is actually serving on.  It defaults to
    the untagged one, which is what the first serves read; a serve on a tagged vector
    must pass its own, or this checks a file the candidate does not use -- the same
    shape of weakness as the disjointness guard of §7t, where a named column that was
    not on the frame made a test read as a pass.
    """
    man = Path(manifest).resolve() if manifest else CLASS_MANIFEST
    if not man.exists():
        return {"checked": False, "reason": f"{man.name} does not exist"}
    theirs = json.loads(man.read_text()).get("prefix_months")
    if theirs is None:
        return {"checked": False, "reason": "the class manifest names no prefix_months"}
    if sorted(theirs) != sorted(PREFIX_MONTHS):
        raise SystemExit(
            f"base: this lane's prefix map is {PREFIX_MONTHS} and "
            f"exp-046-surfacefit's published map is {sorted(theirs)}; one map in both "
            "lanes (desk, 2026-09-09), so fix the register or that lane before serving")
    return {"checked": True, "months": list(PREFIX_MONTHS),
            "source": str(REGISTER.relative_to(ROOT)),
            "agrees_with": str(man.relative_to(ROOT))}


def prefix_frame() -> pd.DataFrame:
    """The rows the airline prefix map is learned on: `PREFIX_MONTHS`, departures only.

    Matches `exp-046-surfacefit/code/arms.py:month_block` line for line -- the same
    months, the same `PHASE_mvt == "DEP"` filter, the same two columns -- so
    `block.prefix_map` gets the identical frame in both lanes.  It is out of fold for
    every fold at once, so it does not depend on which month is being built and one
    frame serves every month, fold and ranking alike.
    """
    tr = pd.concat([pd.read_parquet(D.month_file(m),
                                    columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
                    for m in PREFIX_MONTHS], ignore_index=True)
    return tr[tr.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt").reset_index(drop=True)


def board(rmse_by_fold: dict[str, float]) -> float:
    return float(np.sqrt(sum(SHARES[f] * rmse_by_fold[f] ** 2 for f in SHARES)))


def v12_vectors() -> tuple[dict[str, pd.Series], dict]:
    """Fold-side v12 per fold, keyed by `MVT_ID_mvt`, with its own gate record.

    `exp-035` §26a, applied to the fold side:

        head    = the climbed member through exp-027's refit-bulk arm
        masked  = head at exp-036's eight kept airports, v7 elsewhere
        served  = masked + 0.217 * (stack-5 - masked)

    THE GATE is that those same two lines, given the UNCLIMBED base, return v11.  Both
    lines are run on it, not only the amplitude one (round 2 non-blocking D read the old
    code as running one and the docstring as claiming two): the mask line is a verified
    NO-OP there, because `exp-036`'s refit-bulk-mask base already equals v7 off the kept
    airports -- asserted below, `max|p10 - v7| = 0.0` on all three folds -- so the claim
    "v12 is v11 with its K2 member replaced" is a statement about these vectors and not
    about intent.
    """
    ref, vec, prov, files = V38.load(verbose=False)
    keep = json.loads((ROOT / V38.MASK_JSON).read_text())["keep"]
    p10, s5, v7 = vec[V38_BASE], vec["stack-5"], vec["v7"]
    rep = {"head": str(V12_HEAD_OOF), "keep": list(keep), "s": S5_AMP,
           "rule": "np.where(airport in exp-036 keep, head, v7), then + s*(stack-5 - it)"}
    out, head_rmse, clean11, clean12 = {}, {}, {}, {}
    for f in FOLDS:
        d = pd.read_parquet(str(V12_HEAD_OOF).format(fold=f))
        m = pd.Series(d["y_pred"].to_numpy("float64"),
                      index=d["MVT_ID_mvt"].to_numpy("float64"))
        ids = ref[f]["MVT_ID_mvt"].to_numpy("float64")
        h = pd.Series(ids).map(m).to_numpy("float64")
        if not np.isfinite(h).all():
            raise SystemExit(f"v12 {f}: the climbed head misses {int((~np.isfinite(h)).sum())} rows")
        y = ref[f]["y_true"].to_numpy("float64")
        head_rmse[f] = round(float(np.sqrt(np.mean((y - h) ** 2))), 4)
        apt = ref[f]["airport"].astype(str).to_numpy()
        mk = np.where(np.isin(apt, keep), h, v7[f])
        p12 = mk + S5_AMP * (s5[f] - mk)
        mk11 = np.where(np.isin(apt, keep), p10[f], v7[f])
        off = ~np.isin(apt, keep)
        gap = float(np.max(np.abs(p10[f][off] - v7[f][off]))) if off.any() else 0.0
        if gap > 1e-9:
            raise SystemExit(f"v12 gate: the mask line is not a no-op on the unclimbed "
                             f"base at {f} (max|p10 - v7| off keep = {gap})")
        p11 = mk11 + S5_AMP * (s5[f] - mk11)
        clean11[f] = D.rmse_clean(ref[f]["y_true"], p11, ref[f]["MVT_ID_mvt"], f)
        clean12[f] = D.rmse_clean(ref[f]["y_true"], p12, ref[f]["MVT_ID_mvt"], f)
        out[f] = pd.Series(p12, index=ids)
    for f, want in V12_HEAD_RMSE.items():
        if abs(head_rmse[f] - want) > 1e-3:
            raise SystemExit(f"v12: the climbed head reads {head_rmse[f]} on {f}, "
                             f"not the published {want}")
    b11, b12 = board(clean11), board(clean12)
    if abs(b11 - V11_BOARD_CLEAN) > GATE:
        raise SystemExit(f"v12 gate: the same two lines rebuild v11 at {b11:.4f}, "
                         f"not {V11_BOARD_CLEAN:.4f}")
    if abs(b12 - V12_BOARD_CLEAN) > GATE:
        raise SystemExit(f"fold-side v12 clean board {b12:.4f} against "
                         f"{V12_BOARD_CLEAN:.4f}")
    rep["head_rmse"] = head_rmse
    rep["clean_v11_v12_gap"] = {f: round(clean12[f] - clean11[f], 4) for f in FOLDS}
    rep["january_head_note"] = (
        "january's climbed head carries no published RMSE to gate against, so it is "
        "gated on the CLEAN board instead: its clean v12 is checked to sit within "
        "GATE of the published clean v12 board through board(clean12) above, and its "
        "all-row RMSE is recorded here rather than asserted.")
    rep["clean_v11"] = {f: round(v, 4) for f, v in clean11.items()}
    rep["clean_v12"] = {f: round(v, 4) for f, v in clean12.items()}
    rep["board_clean_v11"] = round(b11, 4)
    rep["board_clean_v12"] = round(b12, 4)
    rep["gain_clean_over_v11"] = round(b11 - b12, 4)
    return out, rep


def load(with_v12: bool = True) -> dict[str, pd.DataFrame]:
    """Per fold: the v10 panel with `p11`, `r11`, `p12`, `r12`, the cell and day keys.

    Step 1 of this experiment is written against `r11`, because it was measured before
    the desk made v12 the head at 14:50 UTC and re-basing a label-side diagnostic would
    not change what it diagnosed.  Step 2 and anything served are written against
    `r12`, which is what `assignments/queue.md` step 2's contract now means.
    """
    pan = V.load()
    _, vec, _ = V.Bs.load()
    out, clean = {}, {}
    v12, v12rep = (v12_vectors() if with_v12 else ({}, {}))
    for f in FOLDS:
        d = pan[f].copy()
        s5 = vec["stack-5"][f]
        if len(s5) != len(d):
            raise SystemExit(f"{f}: stack-5 has {len(s5)} rows, the panel {len(d)}")
        p10 = d["p10"].to_numpy("float64")
        d["p11"] = p10 + S5_AMP * (s5 - p10)
        d["r11"] = d["y_true"].to_numpy("float64") - d["p11"].to_numpy("float64")
        if with_v12:
            ids = d["MVT_ID_mvt"].to_numpy("float64")
            p12 = pd.Series(ids).map(v12[f]).to_numpy("float64")
            if not np.isfinite(p12).all():
                raise SystemExit(f"{f}: v12 misses rows the panel has")
            d["p12"] = p12
            d["r12"] = d["y_true"].to_numpy("float64") - p12
        d["day"] = V.day(d["t"].to_numpy("float64"))
        d["month"] = MONTH_OF[f]
        d["apt"] = d["airport"].astype(str).to_numpy()
        d["bin"] = np.floor(d["t"].to_numpy("float64") / BIN).astype("int64")
        out[f] = d.reset_index(drop=True)
        clean[f] = D.rmse_clean(d["y_true"], d["p11"], d["MVT_ID_mvt"], f)
    b = board(clean)
    if abs(b - V11_BOARD_CLEAN) > GATE:
        raise SystemExit(f"fold-side v11 clean board {b:.4f} against "
                         f"{V11_BOARD_CLEAN:.4f}")
    for f in FOLDS:
        out[f].attrs["rmse_clean"] = clean[f]
        out[f].attrs["v12"] = v12rep
    return out


# ---- the estimators, all borrowed --------------------------------------------
relation = None       # bound below from exp-041's anatomy, which needs its own base
blocks_of = C41.blocks_of
boot_ratio = C41.boot_ratio
group_sums = C41.group_sums


def self_excluded(code: np.ndarray, ncode: int, r: np.ndarray):
    """Sum and count of the OTHER rows in the same code, per row (exp-041)."""
    s, c = group_sums(code, ncode, r)
    return s - r, c - 1.0


def shared(r: np.ndarray, code: np.ndarray, ncode: int, days: np.ndarray, rng,
           keep: np.ndarray | None = None) -> dict:
    """exp-041's self-excluded cross-product on one cell code: `E[r_i r_j]`."""
    os_, oc_ = self_excluded(code, ncode, r)
    if keep is None:
        keep = np.ones(len(r), bool)
    num = np.where(keep, r * os_, 0.0)
    den = np.where(keep, oc_, 0.0)
    if den.sum() <= 0:
        return {"pairs": 0, "value": None, "ci": [None, None], "rows": 0}
    pt, lo, hi = boot_ratio(num, den, days, rng)
    return {"pairs": int(round(float(den.sum()) / 2.0)), "value": round(pt, 1),
            "ci": [round(lo, 1), round(hi, 1)],
            "rows": int((keep & (oc_ > 0)).sum())}


def cellcode(apt: np.ndarray, bins: np.ndarray, extra: np.ndarray | None = None):
    """A dense integer code for airport x 10-min bin (x an optional third key)."""
    parts = [pd.Series(apt).astype(str), pd.Series(bins).astype(str)]
    if extra is not None:
        parts.append(pd.Series(extra).astype(str))
    key = parts[0].str.cat(parts[1:], sep="|")
    code, uniq = pd.factorize(key)
    return code.astype("int64"), int(len(uniq))


def paired(r: np.ndarray, d: np.ndarray, days: np.ndarray, rng=None) -> dict:
    """`2A - B` per row of the whole fold, with a day-clustered interval."""
    v = 2 * r * d - d * d
    rec = {"rows": int(len(r)), "rows_moved": int((d != 0).sum()),
           "A_mean_rd": float(np.mean(r * d)), "B_mean_dd": float(np.mean(d * d)),
           "mse_gain_s2": float(np.mean(v))}
    if rng is not None:
        pt, lo, hi = boot_ratio(v, np.ones(len(v)), days, rng)
        rec["ci95"] = [float(lo), float(hi)]
    return rec


def main() -> None:
    pan = load()
    rows = []
    rng = np.random.default_rng(SEED)
    for f in FOLDS:
        d = pan[f]
        code, n = cellcode(d["apt"].to_numpy(), d["bin"].to_numpy())
        sh = shared(d["r11"].to_numpy("float64"), code, n, d["day"].to_numpy(), rng)
        rows.append({"fold": f, "rows": int(len(d)),
                     "rmse_clean_v11": round(float(d.attrs["rmse_clean"]), 4),
                     "cells": n,
                     "shared_component_s2": sh})
        print(f"{f:9s} rows {len(d):7,d} clean {d.attrs['rmse_clean']:.4f} "
              f"cells {n:6,d} shared {sh['value']} {sh['ci']}")
    b = board({f: pan[f].attrs["rmse_clean"] for f in SHARES})
    print(f"clean board {b:.4f} (published {V11_BOARD_CLEAN})")
    (EXP / "analysis/base.json").write_text(json.dumps(
        {"board_clean": round(b, 4), "published": V11_BOARD_CLEAN,
         "s5_amp": S5_AMP, "bin_s": BIN, "seed": SEED, "boot": BOOT,
         "folds": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
