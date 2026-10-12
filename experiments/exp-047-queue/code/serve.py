"""exp-047-queue step 3: the served candidate, on the NO-CLOCK class only.

The desk settled the division on 2026-09-08 at 14:10 UTC: this lane prices and serves
the rows with `obs_has_clock` **false**, `exp-046-surfacefit` serves the with-clock rows,
and the desk composes the two disjoint updates.  Since 2026-09-09 the boundary is ONE
published vector, surfacefit's `artifacts/exp-046/class/<month>.parquet`, so the two
updates are disjoint by construction rather than by the lanes agreeing; and the BASE is
**v13**, surfacefit's uploaded with-clock candidate, which is verified bit-identical to
v12 on every row this lane owns.  So this file changes **only** rows this lane owns and
may serve, and leaves every other byte of v13 exactly as it is.

    candidate = v13                                       everywhere
              = v13 + d   on `~obs_has_clock & ~noac` rows whose CELL the hold keeps

Held bit-identical to v13, by construction and then by assertion:

*   the 388 `noac` rows (`AIRCRAFT_TYPE_mvt` null) -- their error is the label's
    placeholder semantics, not a physical event (exp-037, `assignments/queue.md`);
*   every row `exp-046-surfacefit` serves, and every row its candidate moved;
*   every row in a cell the hold holds (see `cell_support`);
*   every row whose block carries no neighbourhood estimate at all.

The gate, before a byte is written: `analysis/arms-step2-unmatched-fitall-3f.json` must
show the `2A - B` interval clear of zero on **every** fold at full amplitude, on BOTH the
registered whole no-clock class and the cells actually served.  `--dry-run` writes
`work/dry-run.parquet` and the census instead, so the machinery can be checked without
producing a file that looks like a candidate.

    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-047-queue/code/serve.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(HERE))
import base as BS                                            # noqa: E402
import block as BK                                           # noqa: E402
import explain as EX                                         # noqa: E402
import arms as AR                                            # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D                                         # noqa: E402

#: THE BOARD HEAD, desk 2026-09-08 14:50 UTC (tail rule): v12 is v11 with only its K2 member
#: replaced by `exp-035-k2climb` arm 3b's climbed member, board 262.7418.  Its 388
#: `noac` rows and its EHAM and EDDF rows are v11's own, so a held row stays
#: bit-identical to v12 by the same argument that held it identical to v11.  The old
#: head is kept beside it so a report can say which one a number was read against.
#: THE BASE MOVED TO v13 (desk, 2026-09-09).  `exp-046-surfacefit`'s with-clock
#: candidate went up alone and scored **248.4818**, rank 1 by 12.449 s, a transfer of
#: 1.75x on its +8.128 s of fold gain.  The composition base is therefore v13 and not
#: v12.  This changes no value this lane writes: v13 is v12 plus surfacefit's update on
#: `moved_by_v13`, this lane's mask is disjoint from that set, and the two files are
#: verified BIT-IDENTICAL on all 154,358 rows this lane owns (`max|v13 - v12| = 0.0`,
#: `analysis/serve.json` `base.equals_v12_on_owned`).  So the correction is unchanged in
#: substance and what changes is the 188,483 bytes this lane holds rather than writes.
BASE = ROOT / "submissions/uploaded/youthful-giraffe_v13.parquet"
BASE_SHA = "79a301627e918140933bb4c1dcf9a9249a63bd81669f1793e252f821aa1dee51"
BASE_NAME = "v13 (surfacefit's with-clock candidate, board 248.4818)"
#: THE HEAD OF THE DAY, chosen with `--base` (desk, 2026-09-09).  `v15` is
#: surfacefit's next with-clock candidate, and it is built ON TOP of this lane's v14,
#: so on the rows this lane owns its bytes are v12 plus arm `N`'s update.  A serve on
#: it therefore REPLACES that update rather than stacking on it: the value written on
#: an owned row is `v12 + d`, never `base + d`, because the correction is priced
#: against v12's fold side.  The bytes being replaced are checked to be this lane's own
#: accepted candidate, so nothing of the other lane's is overwritten.
HEADS = {
    "v13": (BASE, BASE_SHA, BASE_NAME),
    "v15": (ROOT / "submissions/candidates/exp-046-surface-v2.parquet",
            "eabd824d925f15223db452557fe113e286ca8ce49302e8a97e79f060c0360ea1",
            "surfacefit's v15 candidate, built on v14 (goes up as v15)"),
    # THE BOARD FILE (desk pick, 2026-09-09): 246.7053, surfacefit's 255-leaf head and
    # tail guard on v15's bytes.  This lane's 63,862 rows are unchanged in it, so on an
    # owned row it carries v12 + arm N's v14 update and a serve REPLACES that update;
    # on a newly served cell it must equal v12, which `check_disjoint` proves.
    "v16": (ROOT / "submissions/uploaded/youthful-giraffe_v16.parquet",
            "6f8368da9b8da818a726c7cc6cf20de6ff7777196f1e78d41015a8cda441fdf7",
            "v16 (the board file, 246.7053)"),
}
#: this lane's accepted candidate, whose bytes a later base carries on the owned rows
PREV_CAND = ROOT / "submissions/candidates/exp-047-queue.parquet"
PREV_CAND_SHA = "519ca89f381056afb453ca62d64df622fcdbbdb14d8607cb0307c2be077676b2"
#: kept so the record can say the two agree where this lane writes
V12 = ROOT / "submissions/candidates/exp-035-k2climb-v11member.parquet"
V12_SHA = "48dd572d61ff2403f13614ce04bdc3a8fd0dbdb5175bef7eb143650ea520ec52"
PREV_HEAD = ROOT / "submissions/candidates/exp-038-v10-stack5-blend.parquet"
PREV_HEAD_SHA = "42ca5ee6410938d56c8830e9ec3980d806633c3b2ff15963c0ffe2fe97c39f8c"
OUT = ROOT / "submissions/candidates/exp-047-queue.parquet"
DRY = EXP / "work/dry-run.parquet"
FLOOR = 60.0


#: the fold months whose panels the served head is fitted on, and the ranking months it
#: is applied to.  The harvest delivered all five by 2026-09-08 23:39 UTC.
FIT_FOLDS = ["july_loo", "december", "january"]
RANK_MONTHS = ["2026-01", "2026-07"]
BLOCKS = ROOT / "artifacts/exp-047/block"
STEP2 = EXP / "analysis/arms-step2-unmatched-fitall-3f.json"


def gate(path: Path = STEP2) -> dict:
    """`assignments/queue.md`: every fold interval clear of zero at full amplitude, on
    the population it is served to -- which for this lane is the no-clock class.

    The endpoint is the PROJECTED one (desk, 2026-09-08 16:50 and 17:05 UTC): each
    airport's fold gain per served row, weighted by the rows that airport-month has in
    the ranking file.  The pooled fold gain is carried beside it and an arm has to clear
    zero on the projection of every fold; the pooled column is reported, never gated on,
    because it prices the 2025 receiver map's population and not the one served.
    """
    if not path.exists():
        return {"passed": False, "reason": f"{path.name} does not exist; step 2 has "
                                           "not been measured on the rotation"}
    a = json.loads(path.read_text())
    if a.get("rows") != "unmatched":
        return {"passed": False, "reason": f"{path.name} prices {a.get('rows')} rows, "
                                           "not the no-clock class"}
    bad, seen, pooled, servedonly = {}, {}, {}, {}
    for arm, per in a["arms"].items():
        proj = per.get("projected") or {}
        folds = [f for f in per if isinstance(per[f], dict) and f in proj]
        if not folds:
            bad.setdefault(arm, []).append("no fold was scored")
        for f in folds:
            lo = proj[f]["ci95_s2"][0]
            seen.setdefault(arm, {})[f] = proj[f]["ci95_s2"]
            pooled.setdefault(arm, {})[f] = (per[f].get("all") or {}).get("ci95")
            if lo <= 0:
                bad.setdefault(arm, []).append(f"{f}: {proj[f]['ci95_s2']}")
            # AND ON THE POPULATION ACTUALLY SERVED (desk, 2026-09-09).  The registered
            # endpoint is the whole no-clock class; the file written serves only the
            # cells the hold keeps, so the advance clause is read there as well.  An arm
            # passes only if BOTH are clear of zero on every fold.
            so = proj[f].get("served_cells_only")
            if so is None:
                bad.setdefault(arm, []).append(
                    f"{f}: the record predates the cell hold and carries no "
                    "served_cells_only reading; re-run code/arms.py")
                continue
            servedonly.setdefault(arm, {})[f] = so["ci95_s2"]
            if so["ci95_s2"][0] <= 0:
                bad.setdefault(arm, []).append(
                    f"{f} on the served cells: {so['ci95_s2']}")
    ok = [arm for arm in seen if arm not in bad]
    return {"passed": bool(ok), "arms_passing": sorted(ok),
            "record": str(path.relative_to(EXP)),
            "endpoint": "projected per airport-month AND per reach class, read both on "
                        "the whole no-clock class and on the cells the hold serves",
            "served_cells_intervals_s2": servedonly,
            "matcher_sha256": a.get("matcher_sha256"), "rule": a.get("rule"),
            "fit_rows": a.get("fit_rows"), "miss_indicator": a.get("miss_indicator"),
            "projected_intervals_s2": seen, "pooled_intervals_s2": pooled,
            "arms_failing": bad}


def fold_frame(rule: str | None, srcs: tuple[str, ...]) -> pd.DataFrame:
    """Every row of the three fold panels with its block, keyed on `r12`.

    The block is rebuilt here rather than read from `artifacts/exp-047/block/`, because
    the emitted fold blocks carry `*_n_res_*` against the fold-side **v11** while the
    ranking blocks carry them against the served **v12**.  Fitting on one convention and
    applying on the other would move the residual definition under the head.
    """
    pan = BS.load(with_v12=True)
    out = []
    for f in FIT_FOLDS:
        pred = pd.Series(pan[f]["p12"].to_numpy("float64"),
                         index=pan[f]["MVT_ID_mvt"].to_numpy("float64"))
        blk, _rep = EX.block_of(f, srcs, rule, pred=pred)
        m = pan[f].merge(blk, on="MVT_ID_mvt", how="left", validate="one_to_one")
        m["fold"] = f
        out.append(m)
    return pd.concat(out, ignore_index=True)


#: THE SHARED CLASS VECTOR (desk, 2026-09-09).  `exp-046-surfacefit` publishes the one
#: `obs_has_clock` both surface lanes read, with the `noac` hold and the rows its own
#: candidate moves, so the two updates are disjoint BY CONSTRUCTION rather than by two
#: lanes agreeing.  This lane's mask is `~obs_has_clock & ~noac`; surfacefit's candidate
#: moves `moved_by_v13`; the composition takes surfacefit's values there, this lane's
#: here and v12 everywhere else.  `_gate_class` refuses to serve if the two sets touch.
CLASS = ROOT / "artifacts/exp-046/class"
CLASS_MANIFEST = ROOT / "experiments/exp-046-surfacefit/analysis/class-manifest.json"


def class_basis(tag: str | None) -> tuple[Path, str]:
    """The shared class vector this serve reads, and the file suffix it carries.

    The desk (2026-09-09, after surfacefit's round 3): v15's candidate is built on the
    PER-MONTH class build, which moves 21 ranking rows against the published whole-file
    vector, and `artifacts/exp-046/class/<month>-v15.parquet` with its own manifest is
    the mask basis for every later serve in BOTH lanes.  `--class-tag v15` takes it;
    no tag keeps the vector v14 was served on.
    """
    if tag is None:
        return CLASS_MANIFEST, ""
    man = CLASS_MANIFEST.with_name(f"class-manifest-{tag}.json")
    if not man.exists():
        raise SystemExit(f"{man} does not exist; the {tag} class vector is not "
                         "published yet, or its manifest has another name")
    return man, f"-{tag}"


def moved_column(frame: pd.DataFrame, manifest: dict) -> str:
    """The class vector's column of rows the OTHER lane moves, whatever it is named.

    The manifest names it (`moved_column`); the sniff is the fallback for the first
    vector, which carried `moved_by_v13` and no such field.
    """
    named = manifest.get("moved_column")
    if named:
        if named not in frame.columns:
            raise SystemExit(f"the class manifest names {named} and the vector has "
                             f"{[c for c in frame.columns]}")
        return named
    got = [c for c in frame.columns
           if c.startswith("moved_by_") or c.startswith("changed_by_")]
    if len(got) != 1:
        raise SystemExit(f"the class vector carries {got or 'no'} moved/changed column; "
                         "exactly one is needed to check the two updates are disjoint")
    return got[0]


def rank_frame(rule: str | None, step2: Path | None = None,
               class_tag: str | None = None) -> pd.DataFrame:
    """The ranking months' blocks, as emitted, with the manifest gated on the matcher,
    and the SHARED class vector merged over this lane's own `obs_has_clock`."""
    man = json.loads((EXP / "analysis/interface.json").read_text())
    got = BK.sha_of(ROOT / "experiments/exp-046-surfacefit/code/block.py")
    if man.get("matcher_block_py_sha256") != got:
        raise SystemExit(
            f"the interface was written against matcher {man.get('matcher_block_py_sha256','')[:12]}… "
            f"and the matcher on disk is {got[:12]}…; re-emit before serving")
    # ROUND 3, C2: the GATE RECORD's matcher has to be the manifest's as well. The gate
    # is a measurement taken on one class boundary and the file is written on another;
    # if the matcher moved between them the projection was read on a population the
    # candidate does not serve, and nothing else here would notice.
    grec = json.loads((step2 or STEP2).read_text()).get("matcher_sha256")
    if grec != got:
        raise SystemExit(
            f"the step-2 gate was measured against matcher {str(grec)[:12]}… and the "
            f"matcher on disk is {got[:12]}…; re-run step 2 before serving")
    cmanp, suffix = class_basis(class_tag)
    cman = json.loads(cmanp.read_text())
    if cman.get("matcher_sha256") != got:
        raise SystemExit(
            f"the shared class vector was written against matcher "
            f"{str(cman.get('matcher_sha256'))[:12]}… and the matcher on disk is "
            f"{got[:12]}…")
    out = []
    for month in RANK_MONTHS:
        rec = man["months"].get(month)
        if rec is None:
            raise SystemExit(f"no block for {month} in the manifest")
        p = ROOT / rec["path"]
        if D.sha256(p) != rec["sha256"]:
            raise SystemExit(f"{p} does not hash to the manifest's {rec['sha256'][:12]}…")
        if rule is not None and rec["summaries_rule"] != rule:
            raise SystemExit(f"{month} was built under rule {rec['summaries_rule'][:8]}…, "
                             f"not {rule[:8]}…")
        b = pd.read_parquet(p)
        crec = cman["months"].get(month)
        cp = CLASS / f"{month}{suffix}.parquet"
        if crec is None or D.sha256(cp) != crec["sha256"]:
            raise SystemExit(f"{cp} does not hash to the class manifest's "
                             f"{str((crec or {}).get('sha256'))[:12]}…")
        cl = pd.read_parquet(cp)
        mvcol = moved_column(cl, cman)
        # THE OWNERSHIP COLUMN (desk, 2026-09-09, with the v16 vector).  The class
        # vector now separates three things and the guard must take the STRICTEST:
        #   occupied_by_exp_046  the rows that lane OWNS -- 189,009, of which 526 still
        #                        sit at v12 in v16 because it has not written them yet
        #   written_by_exp_046   the rows whose BYTES carry its value -- 188,483
        #   changed_by_v16       the rows its shipped candidate moves off its own base
        # A disjointness test on the written or moved set would let this lane claim the
        # 526 rows the other lane is about to write.  So the assertion is against
        # OWNERSHIP; the moved column stays for the record, and the desk verifies the
        # bytes against `written_by_exp_046`.
        # FALLBACK round 5, item 5: the ownership column is named by the MANIFEST's
        # `columns` block, not sniffed off the vector, and a vector at or above the v16
        # basis that declares no ownership column is refused rather than quietly falling
        # back to the moved set.
        declared = set(cman.get("columns") or {})
        own_named = "occupied_by_exp_046" if "occupied_by_exp_046" in declared else None
        if own_named is None and OWNERSHIP_REQUIRED_FROM <= str(cman.get("suffix") or ""):
            raise SystemExit(
                f"the class manifest {cmanp.name} declares no ownership column and its "
                f"basis {cman.get('suffix')!r} is at or above {OWNERSHIP_REQUIRED_FROM}; "
                "a disjointness test on the moved set alone would let this lane claim "
                "rows the other lane owns but has not written yet")
        if own_named and own_named not in cl.columns:
            raise SystemExit(f"the class manifest declares {own_named} and the vector "
                             f"for {month} does not carry it")
        extra = [c for c in (own_named, "written_by_exp_046")
                 if c and c in cl.columns]
        cl = cl[["MVT_ID_mvt", "obs_has_clock", "noac", mvcol] + extra].rename(
            columns={mvcol: "moved_by_head"})
        cl.attrs["moved_by"] = mvcol
        cl.attrs["ownership_column"] = own_named
        b = b.rename(columns={"obs_has_clock": "obs_has_clock_own"})
        b = b.merge(cl, on="MVT_ID_mvt", how="inner", validate="one_to_one")
        if len(b) != rec["rows"]:
            raise SystemExit(f"{month}: the class vector covers {len(b)} of "
                             f"{rec['rows']} block rows")
        b["month"] = month
        b.attrs["moved_by"] = mvcol
        b.attrs["ownership_column"] = cl.attrs.get("ownership_column")
        b.attrs["class_manifest"] = str(cmanp.relative_to(ROOT))
        b.attrs["class_files"] = str(CLASS.relative_to(ROOT)) + f"/<month>{suffix}.parquet"
        out.append(b)
    j = pd.concat(out, ignore_index=True)
    j.attrs.update(out[0].attrs)
    return j


def build_update(fit: pd.DataFrame, rank: pd.DataFrame, arm: str,
                 learner: str) -> tuple[pd.Series, dict]:
    """`d` for every ranking row this lane owns, fitted on the three 2025 fold panels.

    The head is fitted on EVERY row of the fitting panels, not only on the no-clock
    ones: `Nf_*` needs no observation and is defined identically in both classes, so a
    head fitted on every row estimates the same coefficients from twice the data.  It is
    then priced and applied only where this lane serves.  No missingness indicator, so
    an absent observation is never a feature (desk, 2026-09-08 16:50 UTC).
    """
    cols = sorted({c for v in AR.arm_columns(arm, fit).values() for c in v})
    missing = [c for c in cols if c not in rank.columns]
    if missing:
        raise SystemExit(f"the ranking blocks lack {missing[:3]}")
    # under the per-cell form (`AR.SPLIT_ARM`) there are two heads, one per reach class
    # of the RANKING rows, both fitted on the same fitting rows.  The reach class of a
    # ranking row is read from the emitted block's own `No_n_n_cell`, the same column
    # the fold side reads.
    d, arep = AR.arm_predict(arm, fit, rank, "r12", learner, pool=fit,
                             miss_indicator=False)
    rep = {"arm": arm, "columns": cols, "form": arep["form"], "learner": learner,
           "fit_rows": int(len(fit)), "fit_folds": FIT_FOLDS,
           "rank_rows": int(len(rank)), "amplitude": 1.0,
           "note": "full amplitude; no scale is fitted on anything the candidate is "
                   "scored on"}
    return pd.Series(d, index=rank["MVT_ID_mvt"].to_numpy("float64")), rep


def verify(cand: np.ndarray, base: np.ndarray, ids: np.ndarray,
           owned: np.ndarray, noac: np.ndarray, apt: np.ndarray) -> dict:
    """The desk's verification pattern, computed here so the report carries it."""
    d = cand - base
    moved = d != 0
    rep = {"rows": int(len(cand)), "finite": bool(np.isfinite(cand).all()),
           "min": float(np.min(cand)), "min_ok": bool(np.min(cand) >= FLOOR),
           "rows_moved": int(moved.sum()),
           "rms_moved_s": (round(float(np.sqrt(np.mean(d[moved] ** 2))), 4)
                           if moved.any() else 0.0),
           "max_abs_move_s": float(np.max(np.abs(d))) if len(d) else 0.0,
           "noac_rows": int(noac.sum()),
           "noac_bit_identical": bool(not moved[noac].any()),
           "held_rows_moved": int(moved[~owned].sum()),
           "owned_rows": int(owned.sum()),
           "by_airport": {}}
    for a in np.unique(apt):
        sel = apt == a
        m = sel & moved
        rep["by_airport"][str(a)] = {
            "rows": int(sel.sum()), "moved": int(m.sum()),
            "rms_s": round(float(np.sqrt(np.mean(d[m] ** 2))), 4) if m.any() else 0.0}
    return rep


def fold_currency(rule: str | None, arm: str, learner: str,
                  support: dict | None = None) -> dict:
    """The served correction priced in the BOARD's own units, cross-fitted by fold.

    The desk (2026-09-08, after the three-fold pass) asked for the gain in the contract's
    units so the two lanes compare: all-row seconds at the ranking shares
    0.5571 / 0.4429, clean and all-row, with december beside.  Each fold's `d` comes
    from a head fitted on the OTHER folds only, applied to that fold's SERVED rows and
    zero everywhere else, so nothing scores its own fit and the number is the one this
    candidate would earn if the fold months were the board.
    """
    pan = BS.load(with_v12=True)
    blocks = {}
    for f in FIT_FOLDS:
        pred = pd.Series(pan[f]["p12"].to_numpy("float64"),
                         index=pan[f]["MVT_ID_mvt"].to_numpy("float64"))
        blocks[f], _ = EX.block_of(f, SOURCES, rule, pred=pred)
    merged = {f: pan[f].merge(blocks[f], on="MVT_ID_mvt", how="left",
                              validate="one_to_one") for f in FIT_FOLDS}
    pool = merged[FIT_FOLDS[0]]
    cols = sorted({c for v in AR.arm_columns(arm, pool).values() for c in v})
    per, form = {}, None
    for f in FIT_FOLDS:
        fit = pd.concat([merged[g] for g in FIT_FOLDS if g != f], ignore_index=True)
        m = merged[f]
        d, arep = AR.arm_predict(arm, fit, m, "r12", learner, pool=pool,
                                 miss_indicator=False)
        form = arep["form"]
        y = m["y_true"].to_numpy("float64")
        p12 = m["p12"].to_numpy("float64")
        served = served_mask(m)
        # THE CELL HOLD, on the fold side too, so the currency is the currency of the
        # file that is actually written and not of a wider update
        if support is not None:
            cn = cell_of(m["apt"].astype(str).to_numpy(), AR.reached(m))
            served = served & np.array([bool(support.get(c, {}).get("served"))
                                        for c in cn])
        d = np.where(served, d, 0.0)
        ids = m["MVT_ID_mvt"]
        # THE ASSIGNMENT'S FOUR-WAY RMSE, on the rows this file actually moves (round 4,
        # item 2): T8's class-wide table is the whole no-clock class, which the candidate
        # no longer serves, so the split has to be re-read on the held update.
        nonm = m["nm_missing"].to_numpy(bool)
        per[f] = {
            "rows": int(len(m)), "served_rows": int(served.sum()),
            "rmse4_served_before": EX.rmse4(y[served], p12[served], nonm[served]),
            "rmse4_served_after": EX.rmse4(y[served], (p12 + d)[served], nonm[served]),
            "rmse4_fold_before": EX.rmse4(y, p12, nonm),
            "rmse4_fold_after": EX.rmse4(y, p12 + d, nonm),
            "rmse_all_before": round(float(np.sqrt(np.mean((y - p12) ** 2))), 4),
            "rmse_all_after": round(float(np.sqrt(np.mean((y - p12 - d) ** 2))), 4),
            "rmse_clean_before": round(float(D.rmse_clean(m["y_true"], p12, ids, f)), 4),
            "rmse_clean_after": round(float(D.rmse_clean(m["y_true"], p12 + d, ids, f)), 4),
            "rms_move_s": round(float(np.sqrt(np.mean(d[served] ** 2))), 4)
            if served.any() else 0.0,
            "max_abs_move_s": round(float(np.max(np.abs(d))), 4) if len(d) else 0.0}
    out = {"per_fold": per, "shares": BS.SHARES, "arm": arm, "columns": len(cols),
           "form": form,
           "note": "cross-fitted by fold: each fold's d comes from a head fitted on the "
                   "other two only. December is not a ranking-share fold and is carried "
                   "beside the board, never inside it."}
    for kind in ("all", "clean"):
        b0 = BS.board({f: per[f][f"rmse_{kind}_before"] for f in BS.SHARES})
        b1 = BS.board({f: per[f][f"rmse_{kind}_after"] for f in BS.SHARES})
        out[f"board_{kind}"] = {"before": round(b0, 4), "after": round(b1, 4),
                                "gain_s": round(b0 - b1, 4)}
    out["december"] = {k: per["december"][k] for k in
                       ("rmse_all_before", "rmse_all_after", "rmse_clean_before",
                        "rmse_clean_after", "served_rows", "rms_move_s")}
    return out


#: THE CELL HOLD (desk, 2026-09-09).  The rule and its threshold live in
#: `arms.cell_supported` / `arms.MIN_FOLD_ROWS`, so the record's `served_cells_only`
#: projection and the file written here are the same rule read once.  A cell is
#: (airport, reach class); a ranking MONTH enters through which of its rows are reached,
#: so the held set reads out per airport-month in the census below.
MIN_FOLD_ROWS = AR.MIN_FOLD_ROWS


def cell_support(arm: str, path: Path = STEP2) -> dict:
    """Per (airport, reach class): may this lane serve it, and what does holding cost?"""
    rec = json.loads(path.read_text())["arms"][arm]["projected"]
    folds = sorted(rec)
    cells = sorted({c for f in folds for c in rec[f]["by_airport_reach"]})
    out = {}
    for c in cells:
        per = {f: rec[f]["by_airport_reach"].get(c) for f in folds}
        ok, why = AR.cell_supported(per, airport=c.split("/")[0],
                                    lit_rule=AR.LIT_RULE,
                                    desk_window=AR.DESK_WINDOW)
        out[c] = {
            "served": ok,
            "lit_folds": AR.lit_folds(c.split("/")[0], folds,
                                      desk_window=AR.DESK_WINDOW),
            "by_fold": {f: {**{k: (per[f] or {}).get(k) for k in
                              ("fold_rows", "ranking_rows", "extrapolation",
                               "gain_per_row_s2", "gain_ci95_s2")},
                            "reason": why[f]} for f in folds},
            "projected_s2_forgone_if_held": (
                {} if ok else {f: (per[f] or {}).get("contribution_s2") for f in folds})}
    return out


def cell_of(apt: np.ndarray, reached: np.ndarray) -> np.ndarray:
    """The cell name of every row: `<airport>/reached` or `<airport>/unreached`."""
    return np.array([f"{a}/{'reached' if r else 'unreached'}"
                     for a, r in zip(apt, reached)])


def served_mask(m: pd.DataFrame) -> np.ndarray:
    """The rows this lane owns: no clock, and not `noac`.

    On a RANKING frame `obs_has_clock` is the SHARED vector `rank_frame` merged in, not
    this lane's own column, which it keeps as `obs_has_clock_own`: the desk's 2026-09-09
    rule is one boundary for both lanes, and the 20 ranking rows the two class RULES
    still read differently go to surfacefit's reading so that neither lane can move a
    row the other moves.  On a FOLD frame there is no shared vector and no `noac`, and
    the lane's own column is the only one there is.

    The population is exactly the one step 2 was gated on -- every no-clock row,
    including those the harvest never reached, whose correction is then the file-clock
    part plus the fit rows' mean.  Narrowing it to the reached rows would serve a
    population the gate was not read on; step 2 now prices the two reach classes apart
    and the projection weights each at the ranking months' own reached share, and the
    count and the RMS move on the unreached part are reported here as well.
    """
    # FALLBACK round 5, item 5: an ABSENT `obs_has_clock` used to read as "no row has a
    # clock", which serves the whole month, and an absent `noac` as "no row is noac".
    # A frame without the class column is a frame the class vector did not reach, and
    # that is a stop, not a wider serve.  `noac` genuinely does not exist on a FOLD
    # frame -- the fold panels carry no aircraft-type column -- so its absence stays
    # legal and is named as such.
    if "obs_has_clock" not in m.columns:
        raise SystemExit("served_mask: the frame carries no obs_has_clock; the class "
                         "vector was not merged and every row would read as no-clock")
    hc = m["obs_has_clock"].fillna(False).to_numpy(bool)
    noac = (m["noac"].to_numpy(bool) if "noac" in m.columns
            else np.zeros(len(m), bool))
    return (~hc) & (~noac)


def check_disjoint(rank: pd.DataFrame, mask: np.ndarray) -> dict:
    """This lane's rows and the sibling lane's moved set must not touch (desk,
    2026-09-09).  The column is whatever head the class vector names."""
    if "moved_by_head" not in rank.columns:
        raise SystemExit("the ranking frame carries no moved_by_* column; the shared "
                         "class vector was not merged")
    mv = rank["moved_by_head"].fillna(False).to_numpy(bool)
    # FAIL CLOSED, never fall back.  The first run of this guard reported the
    # ownership column's NAME and silently tested the moved column, because `j` is
    # built from an explicit column list and the ownership column was not on it.  A
    # disjointness test that quietly weakens itself is worse than none.
    own_col = rank.attrs.get("ownership_column")
    if own_col:
        if own_col not in rank.columns:
            raise SystemExit(
                f"the class manifest names {own_col} as the ownership column and the "
                "frame does not carry it; the disjointness test would silently fall "
                "back to the moved set and let this lane claim rows the other lane "
                "owns but has not written yet")
        own = rank[own_col].fillna(False).to_numpy(bool)
    else:
        own = mv
    both = int((mask & own).sum())
    if both:
        raise SystemExit(f"{both} rows are OWNED by the other lane and served by this "
                         "one; the two updates are not disjoint and the composition "
                         "would double-count them or overwrite bytes it is about to "
                         "write")
    moved_both = int((mask & mv).sum())
    if moved_both:
        raise SystemExit(f"{moved_both} rows are moved by BOTH lanes")
    rep_written = {}
    if "written_by_exp_046" in rank.columns:
        w = rank["written_by_exp_046"].fillna(False).to_numpy(bool)
        rep_written = {"written_by_the_other_lane": int(w.sum()),
                       "overlap_with_written": int((mask & w).sum()),
                       "owned_but_not_yet_written": int((own & ~w).sum())}
    return {"served_by_this_lane": int(mask.sum()),
            "owned_by_the_other_lane": int(own.sum()),
            "ownership_column": own_col or rank.attrs.get("moved_by", "moved_by_head"),
            "moved_by_the_other_lane": int(mv.sum()),
            "moved_by_column": rank.attrs.get("moved_by", "moved_by_head"),
            "overlap": both, "overlap_with_moved": moved_both, **rep_written,
            "held_at_the_base_by_both": int((~mask & ~own).sum()),
            "class_vector": rank.attrs.get(
                "class_files", str(CLASS.relative_to(ROOT)) + "/<month>.parquet"),
            "class_manifest": rank.attrs.get("class_manifest")}


#: the class-vector basis from which an ownership column is REQUIRED.  The first two
#: vectors (untagged and `-v15`) predate the three-way split and name only a moved
#: column; from `-v16` on, ownership is what the disjointness test must read.
OWNERSHIP_REQUIRED_FROM = "-v16"

SOURCES = ("F", "O", "H", "N", "geo")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="N")
    ap.add_argument("--learner", default="ridge")
    ap.add_argument("--rule", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--step2", default=str(STEP2),
                    help="the step-2 record the gate and the cell hold are read from")
    ap.add_argument("--base", default="v13", choices=sorted(HEADS),
                    help="the head of the day this candidate is composed onto; on the "
                         "rows this lane owns the value written is always v12 + d")
    ap.add_argument("--class-tag", default=None,
                    help="the shared class vector's basis, e.g. v15 for "
                         "artifacts/exp-046/class/<month>-v15.parquet")
    ap.add_argument("--tag", default=None,
                    help="write to submissions/candidates/exp-047-queue-<tag>.parquet "
                         "and analysis/serve-<tag>.json, so a second candidate never "
                         "overwrites one the desk has accepted")
    a = ap.parse_args()
    step2 = Path(a.step2)
    if not step2.is_absolute():
        step2 = (EXP / step2 if (EXP / step2).exists() else Path.cwd() / step2)
    out_path = (OUT if a.tag is None else
                OUT.with_name(f"{OUT.stem}-{a.tag}{OUT.suffix}"))
    g = gate(step2)
    print(json.dumps({k: v for k, v in g.items()
                      if k not in ("projected_intervals_s2", "pooled_intervals_s2")},
                     indent=2))
    if a.arm not in g.get("arms_passing", []):
        raise SystemExit(f"{a.arm} did not pass the step 2 gate: "
                         f"{g.get('arms_failing', {}).get(a.arm)}")
    # the prefix map is checked against the class manifest THIS serve reads, not the
    # untagged one (§7t: a check that reads a file the candidate does not use)
    cmanp, _ = class_basis(a.class_tag)
    prefix = BS.check_prefix_agrees(cmanp)
    print("prefix map:", json.dumps(prefix), flush=True)
    base_path, base_sha, base_name = HEADS[a.base]
    if D.sha256(base_path) != base_sha:
        raise SystemExit(f"{base_path.name} is not the head this lane serves on")
    if D.sha256(V12) != V12_SHA:
        raise SystemExit(f"{V12.name} is not v12")
    print(f"head gated: {base_name} {base_sha[:16]}", flush=True)

    support = cell_support(a.arm, step2)
    held_cells = {c: v for c, v in support.items() if not v["served"]}
    print(f"cells served: {sorted(c for c, v in support.items() if v['served'])}",
          flush=True)
    cur = fold_currency(a.rule, a.arm, a.learner, support)
    print("board (clean)", cur["board_clean"], "\nboard (all-row)", cur["board_all"],
          flush=True)

    fit = fold_frame(a.rule, SOURCES)
    rank = rank_frame(a.rule, step2, a.class_tag)
    d, urep = build_update(fit, rank, a.arm, a.learner)

    head = pd.read_parquet(base_path)
    base = pd.Series(head[D.TARGET].to_numpy("float64"),
                     index=head[D.ID].to_numpy("float64"))
    rk = D.load_ranking(columns=["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt",
                                 "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt"])
    rk = rk[rk.PHASE_mvt.eq("DEP")].reset_index(drop=True)
    rk["noac"] = rk["AIRCRAFT_TYPE_mvt"].isna()
    keep = ["MVT_ID_mvt", "obs_has_clock", "obs_has_clock_own", "reach_class",
            "moved_by_head", AR.REACH_COL]
    keep += [c for c in ("occupied_by_exp_046", "written_by_exp_046")
             if c in rank.columns]
    j = rk.merge(rank[keep], on="MVT_ID_mvt", how="left", validate="one_to_one")
    j.attrs.update(rank.attrs)
    in_class = served_mask(j)
    cn = cell_of(j["ADEP_mvt"].astype(str).to_numpy(), AR.reached(j))
    supported = np.array([bool(support.get(c, {}).get("served")) for c in cn])
    owned = in_class & supported
    census = {}
    for c in sorted(set(cn[in_class])):
        sel = in_class & (cn == c)
        rows_by_month = (j.loc[sel, "MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
                         .value_counts().sort_index().to_dict())
        census[c] = {"class_rows": int(sel.sum()), "by_month": rows_by_month,
                     "served": bool(support.get(c, {}).get("served")),
                     "reason": {f: v.get("reason")
                                for f, v in support.get(c, {}).get("by_fold", {}).items()},
                     "projected_s2_forgone": support.get(c, {})
                     .get("projected_s2_forgone_if_held", {})}
    print("cell census:", json.dumps({c: {"rows": v["class_rows"],
                                          "served": v["served"]}
                                      for c, v in census.items()}), flush=True)
    disj = check_disjoint(j, owned)
    disj["class_rows_before_the_cell_hold"] = int(in_class.sum())
    # the base moved from v12 to v13; on the rows this lane writes the two files must be
    # the same bytes, or the correction is being applied to a different head than the
    # one step 2 priced against
    v12 = pd.read_parquet(V12)
    v12s = pd.Series(v12[D.TARGET].to_numpy("float64"),
                     index=v12[D.ID].to_numpy("float64"))
    gap = np.abs(base.reindex(j["MVT_ID_mvt"].to_numpy("float64")).to_numpy("float64")
                 - v12s.reindex(j["MVT_ID_mvt"].to_numpy("float64")).to_numpy("float64"))
    disj["max_abs_base_minus_v12_on_owned"] = float(gap[owned].max())
    disj["rows_base_differs_from_v12"] = int((gap != 0).sum())
    # THE CORRECTION IS PRICED AGAINST v12'S FOLD SIDE, so an owned row is written as
    # `v12 + d` and never as `base + d`.  When the base already carries THIS LANE's
    # accepted update on those rows -- v15 is built on v14 -- the write REPLACES it,
    # and the bytes replaced are checked to be this lane's own and nobody else's.
    if disj["max_abs_base_minus_v12_on_owned"] != 0.0:
        prev = pd.read_parquet(PREV_CAND)
        if D.sha256(PREV_CAND) != PREV_CAND_SHA:
            raise SystemExit(f"{PREV_CAND.name} is not this lane's accepted candidate")
        pv = pd.Series(prev[D.TARGET].to_numpy("float64"),
                       index=prev[D.ID].to_numpy("float64"))
        pg = np.abs(base.reindex(j["MVT_ID_mvt"].to_numpy("float64")).to_numpy("float64")
                    - pv.reindex(j["MVT_ID_mvt"].to_numpy("float64")).to_numpy("float64"))
        disj["max_abs_base_minus_this_lanes_candidate_on_owned"] = float(pg[owned].max())
        disj["replaces"] = {"candidate": str(PREV_CAND.relative_to(ROOT)),
                            "sha256": PREV_CAND_SHA,
                            "rows": int(owned.sum()),
                            "rms_replaced_s": round(float(np.sqrt(np.mean(
                                gap[owned] ** 2))), 4)}
        if disj["max_abs_base_minus_this_lanes_candidate_on_owned"] != 0.0:
            raise SystemExit(
                "the base differs from v12 on an owned row by something that is NOT "
                "this lane's accepted update; the write would overwrite another lane's "
                "bytes")
    disj["own_class_would_serve"] = int(
        ((~j["obs_has_clock_own"].fillna(False).to_numpy(bool))
         & (~j["noac"].to_numpy(bool))).sum())
    print("disjoint:", json.dumps(disj), flush=True)
    ids = j["MVT_ID_mvt"].to_numpy("float64")
    dv = np.where(owned, d.reindex(ids).to_numpy("float64"), 0.0)
    if not np.isfinite(dv).all():
        raise SystemExit(f"{int((~np.isfinite(dv)).sum())} owned rows have no update")
    # an owned row is v12 + d (the fold side the correction was priced against); every
    # other row is the base's own byte
    cand = np.where(owned,
                    v12s.reindex(ids).to_numpy("float64") + dv,
                    base.reindex(ids).to_numpy("float64"))
    clipped = int((cand < FLOOR).sum())
    cand = np.maximum(cand, FLOOR)
    rep = verify(cand, base.reindex(ids).to_numpy("float64"), ids, owned,
                 j["noac"].to_numpy(bool), j["ADEP_mvt"].astype(str).to_numpy())
    rep["rows_clipped_at_floor"] = clipped
    rep["floor_s"] = FLOOR
    if not rep["noac_bit_identical"] or rep["held_rows_moved"]:
        raise SystemExit(f"a held row moved: {json.dumps(rep, indent=1)}")

    out = pd.Series(base.to_numpy("float64"), index=base.index.to_numpy("float64"))
    out.loc[ids] = cand
    path = DRY if a.dry_run else out_path
    sha = D.write_submission(out, path)
    rec = {"candidate": str(path.relative_to(ROOT)), "sha256": sha,
           "head": {"name": base_name, "file": str(base_path.relative_to(ROOT)),
                    "sha256": base_sha, "priced_against": "v12's fold side; an owned "
                    "row is written as v12 + d"},
           "gate": g, "update": urep, "verify": rep, "currency": cur,
           "class_boundary": disj, "prefix_map": prefix,
           "cell_hold": {"rule": "desk 2026-09-09, the LFPG rule at the reach grain: a "
                                 "cell whose own fold interval is not clear of zero "
                                 "above, or which is priced on fewer than "
                                 f"{MIN_FOLD_ROWS} rows in any fold, is held at the "
                                 "base",
                         "min_fold_rows": MIN_FOLD_ROWS,
                         "cells_served": sorted(c for c, v in support.items()
                                                if v["served"]),
                         "cells_held": sorted(held_cells),
                         "class_rows_in_class": int(in_class.sum()),
                         "class_rows_served": int(owned.sum()),
                         "class_rows_held": int((in_class & ~supported).sum()),
                         "census": census, "support": support},
           "dry_run": bool(a.dry_run),
           "reach_class_of_served": (j.loc[owned, "reach_class"].astype(str)
                                     .value_counts().to_dict())}
    stem = "serve-dry" if a.dry_run else "serve"
    if a.tag is not None:
        stem = f"{stem}-{a.tag}"
    (EXP / "analysis" / f"{stem}.json").write_text(
        json.dumps(rec, indent=2, sort_keys=True, default=str))
    print(json.dumps({k: rec[k] for k in ("candidate", "sha256", "dry_run")}, indent=1))
    print(json.dumps(rep, indent=1)[:1200])


if __name__ == "__main__":
    main()
