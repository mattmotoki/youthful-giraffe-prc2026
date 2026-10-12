"""exp-046-surfacefit step 2: the fold-side combination, on the contract.

Three arms, all of them a correction of **the base** on the rows that carry a clock of their
own, and on no other row (`SERVED_CLASS`):

    a_linear        pbase + w_air * (obs_taxi - offset_air - pbase)
    b_lgb           pbase + s_air * lgb(feature block -> rbase)
    c_lgb_shipped   arm b with the shipped feature block beside the observation

and all of them priced the same way: `r = y - pbase`, `d = p_arm - pbase`, `2A - B`
all-row at the ranking shares and clean, december alone, day-clustered intervals,
by airport and by reach class.  A row the arm does not move contributes exactly zero
and is checked to.

**The rotation.**  Score fold `f`, fit on the other two fold months.  The assignment
says "fit on the 2025 months other than the scored fold"; an out-of-fold base residual
exists on 2025-01, 2025-07 and 2025-12 only (the three fold panels), so those two are
what "the other 2025 months" can mean here.  The airport offset of arm (a) needs no
residual and could be fitted on any labelled month, but is fitted on the same two, so
that one rotation describes the whole arm.  Recorded as a deviation from the literal
wording, not as a design preference.

**The reach rule, fitted and not chosen.**  Every arm carries one weight per airport,
fitted on the fitting months and clipped to [0, 1].  An airport with fewer than
`MIN_AIRPORT_ROWS` matched fitting rows gets weight 0 and stays at the base.  That is how
LTFM (0 % receipt on the harvest's gate days) and LFPG (16 %) are excluded: by a
number the rotation produces, never by hand.

**The class boundary** (desk, 2026-09-08 13:40, from `exp-047-queue` step 1).  A head
that carries the row's own observed taxi must not span the two reach classes: priced on
the populations apart, one head fitted across both reads **+6,884 s2 on the rows with a
clock and -588 s2 on the rows without one**, because on a row where the feature is
missing the correction can only move the level.  The rows with no clock of their own are
`exp-047-queue`'s: its neighbourhood block -- the self-excluded cell mean of the OTHER
departures' observed taxi -- serves them **+583 s2**, and arrives as
`artifacts/exp-047/block/<month>.parquet` keyed by `MVT_ID_mvt`.  So every arm here fits
on `served_class()` rows and moves `served_class()` rows, the complement stays at the base for
that lane, and `handover()` counts what is left to it.  The boundary is the same rule on
both sides of it: `exp-047-queue/code/block.py` imports THIS file's `block.py` as "the
repo's one matcher" and keys its own class on the same `obs_has_clock` column.

Predeclared budget: `analysis/predeclared.json`, written before any arm ran.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/arms.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "code"))
import base as BA                                            # noqa: E402
import block as B                                            # noqa: E402
from imports import load as _load                            # noqa: E402

R = _load("exp-039-screens", "rowfeat")
sys.path.insert(0, str(ROOT))
import prc.data as D                                         # noqa: E402

PRE = json.loads((EXP / "analysis/predeclared.json").read_text())
FOLDS = BA.FOLDS
FOLD_MONTH = {"july_loo": "2025-07", "december": "2025-12", "january": "2025-01"}
MIN_AIRPORT_ROWS = PRE["MIN_AIRPORT_ROWS"]
LGB = PRE["lgb"]
BLOCKS = 5
SEED = PRE["bootstrap"]["seed"]
#: `a_linear` is the assignment's arm, one weight per airport.  `a_linear_reach` is the
#: same arm with the weight fitted per (airport x strict/wide), which is the smallest
#: instrument that lets the FOLD decide what a wide match is worth -- the linear arm is
#: the only one that cannot read the quality features (Sol round 1, blocking 8).  Both
#: are priced; neither replaces the other.
ARMS = ["a_linear", "a_linear_reach", "b_lgb", "b_lgb_nbr", "c_lgb_shipped",
        "c_lgb_shipped_nbr", "c_lgb_shipped_noobs",
        "b_lgb_placebo", "c_lgb_shipped_placebo"]
#: The PLACEBO arms (desk, 2026-09-08 17:20, after `exp-047-queue`'s both-clocks placebo).
#: The observation is permuted, as a whole row-block, among the rows of one
#: `(airport, take-off day)` cell that ALL carry a clock: every row keeps its class, its
#: base prediction and its label, and only the IDENTITY of the match is destroyed.  An arm
#: that still reads a gain is reading the cell, not the aircraft, and the whole table is a
#: day-and-airport effect the block already has.  The pair is `b_lgb` (observation alone)
#: and `c_lgb_shipped` (observation beside the shipped block).
PLACEBO_OF = {"b_lgb_placebo": "b_lgb", "c_lgb_shipped_placebo": "c_lgb_shipped"}
#: `c_lgb_shipped_noobs` is the CONTROL for `c_lgb_shipped`, and the arm the assignment
#: does not name.  It is the same LightGBM residual correction, on the same with-clock
#: rows, under the same rotation and the same per-airport amplitude, fitted on the shipped
#: feature block ALONE -- no surface observation among its inputs.  Whatever it reads is
#: the base's own underfit on those rows, which the surface features would otherwise be
#: credited with; the observation is worth the DIFFERENCE between the two arms, and that
#: is the number the candidate has to be gated on.
#: The twelve neighbourhood columns `exp-047-queue` measured as an increment ON the
#: with-clock rows -- +842 s2 over the own-clock feature on december (desk, 2026-09-08
#: 16:35).  They arrive as `artifacts/exp-047/block/<month>.parquet`, keyed on
#: `MVT_ID_mvt`, and enter this lane's head as inputs; the no-clock class stays that
#: lane's.  Every arm is priced with and without them, so the fold says what they add
#: here rather than that lane's estimate being carried over.
NBR_COLS = ["No_n_res_cell", "No_n_res_rwy_cell", "No_n_res_apt_30m",
            "No_n_res_rwy_30m", "No_n_res_rwy_60m", "No_n_anom_cell",
            "No_n_anom_rwy_cell", "No_n_anom_rwy_30m", "No_n_n_cell",
            "No_n_n_rwy_cell", "No_n_n_rwy_30m", "No_n_res_rwy_minus_apt"]
NBR_DIR = ROOT / "artifacts/exp-047/block"
#: what `panel_with_block()` found for each fold month, since pandas drops `attrs`
#: through a merge and the report has to say which months carried the columns
_NBR: dict[str, dict] = {}
WORK = EXP / "work"
#: The rows this lane may fit on and may move.  `obs_has_clock` is "the matcher found a
#: ground run AND it carries a 250 m stand-departure onset", so `obs_taxi` exists; the
#: `noac` hold is exp-033's (an observation of the physical event is not evidence about a
#: placeholder label).  Everything else -- unmatched, matched-but-no-clock, noac -- stays
#: at the base here and belongs to `exp-047-queue`'s neighbourhood block.
SERVED_CLASS = "obs_has_clock & obs_taxi.notna() & ~noac"
HANDOVER = "artifacts/exp-047/block/<month>.parquet, keyed by MVT_ID_mvt"
#: The two lanes cut the month with ONE matcher, pinned by content (desk, 2026-09-08
#: 14:10): this lane prices and serves the with-clock class, `exp-047-queue` prices and
#: serves the no-clock class, each on the base elsewhere, and the desk composes the two
#: disjoint updates.  `exp-047-queue/analysis/interface.json` records this sha as the
#: matcher it built its `reach_class` with; if `code/block.py` changes, the boundary
#: moves under both lanes at once and the desk and that lane have to be told.
#: This sha has moved three times and exp-047-queue was told each time: the EDDF-outage
#: hold (2026-09-08 14:45), the callsign-form repair (16:00, `analysis/identity.json`),
#: which DOES change the class of rows -- it matches 20 more rows on the pilot week and
#: 3,885 more on 2026-01 -- and the population pin (2026-09-09 00:28, commit `1d5f33b`).
#: **The pin does NOT change the matching function**: it adds `_PIN`, `pinned()`,
#: `pin_for_month()` and one guard clause in `_summary_path`, all inert when no
#: population is pinned, and the ranking census is identical across it (190,096 served,
#: 154,745 left, 188,484 occupied). So this move needs no interface rebuild in that lane,
#: unlike the callsign-form repair -- but the sha is what the other lane pins, so it moves
#: here and the lane is told (round 5 FALLBACK, finding 2).
#:
#: **`122c7d84` is the TURNAROUND PASS** (2026-09-09), the fifth move and the second
#: agreed with `exp-047-queue` before it was made: the spec is
#: `analysis/chain-spec.md`, queue read it and agreed (`690315b`), and the desk
#: relayed *go*.  It ADDS rows and changes none: 3,179 departures that no callsign
#: join can reach gain a clock through the arrival stand chain, and a 34-column sweep
#: over both ranking months found **0 values changed on the 341,662 rows the pass did
#: not place**.  2,730 of the 3,179 leave `exp-047-queue`'s class for this lane;
#: LFPG's 449 gain `obs_has_clock` and stay outside `occupied_by_exp_046`, so queue
#: keeps writing them and the LFPG hold is unchanged.  Both lanes re-pin and re-price
#: on this sha; nothing is served before that re-price.
MATCHER_SHA = "122c7d84571b53e02de26e53add8e783bc95be9f0eaea5b6ef6c155280aad780"
MATCHER_SHA_BEFORE_THE_TURNAROUND_PASS = (
    "6fdf39519eb0f500c06230e2972d3c8300aa6222ed8d9b003998ddf6e286093b")
MATCHER_SHA_BEFORE_THE_REBUILD = (
    "695af9737e5b082fd2f1017aed41dcf90d40ec7ba0904b14ba6c377a50543393")
MATCHER_SHA_BEFORE_THE_POPULATION_PIN = (
    "ba41935ed83a5f5c39245c8bbf308dbd5adca49372b267fdfa495f33ddade76c")
#: `eded54e5` was the pin itself; `695af973` adds `population_triples()` and
#: `population_key()` so the pin, the cache key and the serve's ranking-population record
#: share ONE hash (§13.5).  Neither move changes the matching function or any row's class,
#: and both were checked by reproducing all three pinned fold keys exactly.
#:
#: **`6fdf3951` is the COORDINATED REBUILD** (2026-09-09), the fourth move and the only one
#: agreed with `exp-047-queue` before it was made: the spec is
#: `analysis/blockpy-rebuild-spec.md`, queue read it and agreed, the desk relayed *go*, and
#: the desk sends ONE MATCHER_SHA line to both lanes now rather than one per edit.  Unlike
#: the pin, this one DOES change rows: the total tie-break of round 3 finding 2, the stale
#: `HELD_AIRPORT_DAYS` values with 2026-01-27 and 28 returned to service, and the
#: `published_short` hold.  Measured on a branch copy over all five months and all 34
#: emitted columns, **605 rows of 854,937 differ at all** -- 602 change class, 601 gaining,
#: and 3 keep their class and move only in a feature read off a neighbour's match.  Both
#: lanes rebuild, re-write their pins and re-price on this sha; nothing is served before
#: that re-price.
MATCHER_SHA_BEFORE_THE_OUTAGE_HOLD = (
    "4a1040e05a844bd69a30c622c1a6d55acd60dc302c15648593c4982f2ae5b0c9")
MIN_PRED = 60.0                 # the serving floor, applied to the fold vector too
#: exp-186: the 2026 months of the ranking mode ($PRC_RANKING_MODE; prc/data.py `rank_months`):
#: ["2026-01", "2026-07"] for ranking.parquet, the four final months for the final file
#: (serving_rule resolves the harvest rule over these months).
RANK_MONTHS = list(D.rank_months())
BOOT = PRE["bootstrap"]["draws"]


def month_arrivals(month: str) -> pd.DataFrame:
    """The month's ARR rows, masked exactly as the ranking file masks them.

    The turnaround pass of `code/block.py` reads the arrival side for identity, stand,
    operator, type and in-block.  `mask_like_ranking` blanks `BLOCK_TIME_UTC_mvt` and the
    target on **DEPARTURE** rows only, so every column this returns is an input; the
    assertion below is the guard that keeps it that way if the mask ever widens.

    `exp-039-screens`'s `rowfeat.departures()` is the sibling of this function, and it is
    that lane's file rather than this one's, so the ARR slice is taken here.
    """
    raw = D.load_month(month)
    masked, _ = D.mask_like_ranking(raw)
    del raw
    arr = masked[masked["PHASE_mvt"] == "ARR"].copy()
    del masked
    if not arr["BLOCK_TIME_UTC_mvt"].notna().any():
        raise SystemExit(f"{month}: the arrival in-block is blank; the pass cannot chain")
    return arr.reset_index(drop=True)


def served_class(d: pd.DataFrame) -> np.ndarray:
    """`SERVED_CLASS` as a mask.  One definition, used by the fold side and the serve."""
    return (d.obs_has_clock.to_numpy(bool) & d.obs_taxi.notna().to_numpy(bool)
            & ~d.noac.to_numpy(bool))


def handover(d: pd.DataFrame) -> dict:
    """What this lane leaves at the base for `exp-047-queue`, by reach class."""
    m = served_class(d)
    left = d.loc[~m, "reach_class"] if "reach_class" in d else pd.Series(dtype=object)
    now = hashlib.sha256((EXP / "code/block.py").read_bytes()).hexdigest()
    # what the OTHER lane actually built its interface with, read from its own manifest:
    # a stale sha there is not an error, it is a rebuild the desk has to schedule before
    # it composes, and this is where it becomes visible
    iface = ROOT / "experiments/exp-047-queue/analysis/interface.json"
    theirs = None
    if iface.exists():
        try:
            theirs = json.loads(iface.read_text()).get("months", {})
            theirs = {m: v.get("build", {}).get("match", {}).get("matcher_sha256")
                      for m, v in theirs.items()}
        except Exception:                                     # noqa: BLE001
            theirs = None
    return {"served_class": SERVED_CLASS,
            "boundary_matcher_sha256": now,
            "boundary_matcher_sha256_pinned": MATCHER_SHA,
            "boundary_matcher_matches_the_pin": now == MATCHER_SHA,
            "exp_047_interface_matcher_sha256": theirs,
            "exp_047_interface_is_stale": (
                None if not theirs else
                {m: (s != now) for m, s in theirs.items()}),
            "rows": int(len(d)),
            "served_rows": int(m.sum()),
            "left_at_base": int((~m).sum()),
            "left_at_base_by_reach_class": {str(k): int(v) for k, v in
                                           left.value_counts().items()},
            "interface": HANDOVER,
            "boundary_is_shared": (
                "exp-047-queue/code/block.py imports this lane's code/block.py as the "
                "repo's one matcher and keys its class on the same obs_has_clock column, "
                "so the two lanes' populations are complementary by construction and not "
                "by agreement. The noac hold is this lane's and any composition of the "
                "two candidates has to keep it."),
            "composition": (
                "desk, 2026-09-08 14:10: this lane prices and serves the with-clock class "
                "and the base elsewhere, exp-047-queue prices and serves the no-clock "
                "class and the base elsewhere, each on its own review round; the desk "
                "composes the two disjoint updates on the base and verifies the composed "
                "file, and either "
                "candidate may go up alone.")}


def neighbour_block(month: str) -> tuple[pd.DataFrame | None, dict]:
    """`exp-047-queue`'s neighbourhood columns for one month, or a stated absence."""
    f = NBR_DIR / f"{month}.parquet"
    if not f.exists():
        return None, {"available": False,
                      "why": f"{f.relative_to(ROOT)} does not exist yet"}
    have = set(pd.read_parquet(f, columns=[]).columns) or set(
        pq_cols(f))
    cols = [c for c in NBR_COLS if c in have]
    miss = [c for c in NBR_COLS if c not in have]
    d = pd.read_parquet(f, columns=["MVT_ID_mvt", *cols])
    return d, {"available": True, "file": str(f.relative_to(ROOT)),
               "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
               "columns": cols, "columns_missing": miss, "rows": int(len(d))}


def pq_cols(f: Path) -> list[str]:
    import pyarrow.parquet as pq                              # noqa: PLC0415
    return [c for c in pq.ParquetFile(f).schema.names]


def month_days(month: str) -> list[str]:
    """Every calendar day of `month`, as the rule resolver wants them."""
    a = pd.Timestamp(month + "-01")
    return pd.date_range(a, a + pd.offsets.MonthEnd(1), freq="D").strftime(
        "%Y-%m-%d").tolist()


_RULE: tuple[str | None, dict] | None = None


def serving_rule() -> tuple[str | None, dict]:
    """ONE `summaries_rule_sha256` for every fold month and every ranking month.

    Resolved across all of them at once, not per month (Sol round 1, blocking 2), so a
    fold table and the candidate it gates can never rest on two different rules.  Cached
    for the process.
    """
    global _RULE
    if _RULE is None:
        _RULE = B.one_rule([(B.AIRPORTS, month_days(m))
                            for m in list(FOLD_MONTH.values()) + RANK_MONTHS])
    return _RULE


_MSHA: str | None = None


def matcher_sha() -> str:
    """`code/block.py`'s own sha256, read once, checked against the declared constant.

    The constant is what the other lane pins and what every manifest quotes; this is what
    actually ran.  They diverged once already, silently (round 5 FALLBACK, finding 2), so
    they are compared here, at the one place every fold block passes through.
    """
    global _MSHA
    if _MSHA is None:
        _MSHA = hashlib.sha256((EXP / "code/block.py").read_bytes()).hexdigest()
        if _MSHA != MATCHER_SHA:
            raise SystemExit(
                f"code/block.py is {_MSHA[:12]}… but arms.MATCHER_SHA declares "
                f"{MATCHER_SHA[:12]}…; the two lanes pin the constant, so update it in "
                "the same commit as the matcher and tell the desk")
    return _MSHA


def block_cache_key(month: str, rule: str | None) -> str:
    """The cache key of a month's block: the rule, the files present, AND THEIR VERSIONS.

    Keyed on the rule alone, a month cached while the harvest had two of its days would
    stay cached when the other twenty-nine arrived (Sol round 1, blocking 3).  Keyed on
    the rule and the files, a month stays cached when only the MANIFEST changes -- and
    that is not hypothetical: the harvest landed 2025-01 with `summaries_rule_sha256:
    null` on every day, this loader failed closed on it and cached an empty block, and
    when the harvest stamped the same unchanged files the key did not move and the empty
    block was served again (2026-09-08 17:50).  The per-day version is what decides
    whether a day is read, so it belongs in the key.
    """
    # a PINNED month keys on its pinned population, not on what is on disk today: that
    # is what makes the key stable across a delivery (desk ruling, 2026-09-09).  The hash
    # itself lives in `block.population_key` so the pin, this key and the serve's
    # ranking-population record cannot drift apart.
    with B.pinned(B.pin_for_month(month)):
        tri = B.population_triples(B.AIRPORTS, month_days(month))
    return B.population_key(tri, rule)


# --------------------------------------------------------------- the blocks --


def month_block(month: str, prefix_months: list[str],
                rule: str | None = None) -> pd.DataFrame:
    """The feature block for one month, cached.  The prefix map excludes the fold months.

    `rule` overrides the resolved serving version.  Only a machinery check passes it
    (`code/smoke.py`, which works on the pilot week and its locally derived summaries);
    the fold table and the serve always take `serving_rule()`.
    """
    WORK.mkdir(exist_ok=True)
    overridden = rule is not None
    if rule is None:
        rule, rule_report = serving_rule()
    else:
        rule_report = {"chosen": rule, "note": "overridden by the caller"}
    # THE CACHE KEYS ON THE CODE AS WELL AS THE DATA (rebuild, 2026-09-09).
    # `block_cache_key` is a statement about the POPULATION -- the files, their bytes and
    # their rule version -- and it must stay that, because the fold pin records it and the
    # guard compares it.  But a cached block is a statement about the population AND the
    # matching function that read it, and the rebuild changes the function without moving
    # a single file: keyed on the population alone, every fold month would have returned
    # its pre-rebuild rows out of `work/` and the re-price would have priced the old
    # matcher.  The matcher's own sha therefore goes in the FILENAME, where it invalidates
    # the cache, and not in the key, where it would re-key both lanes' pins.
    matcher = matcher_sha()
    p = WORK / f"block-{month}-{block_cache_key(month, rule)}-{matcher[:8]}.parquet"
    # the record travels WITH the cache, under the same key, and is copied into
    # `analysis/` on every call.  Written only on a cache miss it could describe an
    # earlier run's block -- which is how the smoke override's record came to sit in the
    # fold table's file (round 2 FALLBACK, blocking 4)
    rec_src = p.with_suffix(".json")
    stem = f"block-{month}" if not overridden else f"block-{month}-override"
    if p.exists():
        if rec_src.exists():
            (EXP / f"analysis/{stem}.json").write_text(rec_src.read_text())
        return pd.read_parquet(p)
    dep = R.departures(month)                     # masked exactly like the ranking file
    # the ARR side of the same month, for the turnaround pass.  `mask_like_ranking` blanks
    # B and y on DEPARTURE rows only, so these rows are inputs and reach the block
    # untouched -- the same slice `code/serve.py` passes, so the fold side and the serve
    # compute one and the same function.  Without it the pass is a silent no-op.
    arr = month_arrivals(month)
    tr = pd.concat([pd.read_parquet(D.month_file(m),
                                    columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
                    for m in prefix_months], ignore_index=True)
    tr = tr[tr.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt")
    # the prefix map is built from MOVEMENT files and is outside the pin; the pin covers
    # the summaries this month reads, which is the population the block was priced on
    with B.pinned(B.pin_for_month(month)):
        blk = B.build(dep, tr, require_rule=rule, arrivals=arr)
    blk.to_parquet(p, index=False)
    rec = json.dumps(
        {"month": month, "rows": int(len(blk)), "rule_overridden": overridden,
         "block_cache": p.name,
         "reach": B.census(blk, B.cohort(dep, B.prefix_map(tr))),
         "rule_sha256": rule, "rule_cover": rule_report,
         "rules_seen": blk.attrs.get("rules_seen"),
         "days_wrong_rule": [f"{x[0]} {x[1]}" for x in
                             blk.attrs.get("days_wrong_rule", [])][:40],
         "days_present": blk.attrs.get("days_present"),
         "days_missing": [f"{a} {d}" for a, d in blk.attrs.get("days_missing", [])][:40]},
        indent=2, sort_keys=True, default=float)
    rec_src.write_text(rec)
    (EXP / f"analysis/{stem}.json").write_text(rec)
    return blk


def prefix_months_for(month: str) -> list[str]:
    """Every 2025 month that is NOT one of the three fold months.

    The prefix map reconstructs a broadcast callsign where `CALLSIGN_flt` is null, so it
    is an input to the feature block.  Excluding only the month being built left the map
    for a FITTING month learned partly on the SCORED month's rows (Sol round 1,
    blocking 6): the fitted weight then depends on the month it is scored against.
    Excluding all three fold months makes one map that is out of fold for every fold, at
    the cost of nine training months instead of eleven.  `month` is accepted so the
    signature still says what the map is for.
    """
    folds = set(FOLD_MONTH.values())
    return [m for m in D.MONTHS if m not in folds and m != month]


def panel_with_block(pan: dict[str, pd.DataFrame],
                     rule: str | None = None) -> dict[str, pd.DataFrame]:
    out = {}
    for f in FOLDS:
        m = FOLD_MONTH[f]
        blk = month_block(m, prefix_months_for(m), rule)
        d = pan[f].merge(blk, on="MVT_ID_mvt", how="left", validate="one_to_one")
        d["obs_matched"] = d.obs_matched.astype("boolean").fillna(False).astype(bool)
        d["obs_has_clock"] = d.obs_has_clock.astype("boolean").fillna(False).astype(bool)
        d["reach_class"] = d.reach_class.fillna("unmatched")
        # the noac hold, by construction: an observation of the physical event is not
        # evidence about a placeholder label (exp-033), so those rows can never move
        d.loc[d.noac.astype(bool), ["obs_matched", "obs_has_clock"]] = False
        d.loc[d.noac.astype(bool), "reach_class"] = "noac_hold"
        d["usable"] = served_class(d)
        # strict or wide, as one bit: the linear arm's only handle on match quality
        d["obs_wide"] = d.match_mode.fillna(0.0).gt(0.0)
        nbr, nrep = neighbour_block(m)
        _NBR[m] = nrep
        if nbr is not None:
            d = d.merge(nbr, on="MVT_ID_mvt", how="left", validate="one_to_one")
        out[f] = d
    return out


# ----------------------------------------------------------------- the arms --


def _dayblocks(days: pd.Series, k: int = BLOCKS) -> np.ndarray:
    """`k` contiguous calendar-day blocks inside each month of `days`."""
    out = np.zeros(len(days), dtype=int)
    dd = pd.to_datetime(days)
    for m, idx in dd.groupby(dd.dt.strftime("%Y-%m")).groups.items():
        pos = days.index.get_indexer(idx)
        u = np.sort(dd.loc[idx].dt.normalize().unique())
        code = {d: min(int(i * k / len(u)), k - 1) for i, d in enumerate(u)}
        out[pos] = dd.loc[idx].dt.normalize().map(code).to_numpy()
    return out


def _amplitude(r: np.ndarray, d: np.ndarray) -> float:
    Bv = float(np.mean(d * d))
    return float(np.clip(float(np.mean(r * d)) / Bv, 0.0, 1.0)) if Bv > 0 else 0.0


def _per_cell_amplitude(fit: pd.DataFrame, dcol: str,
                        keys: list[str]) -> tuple[dict, dict]:
    """One weight per cell of `keys`, fitted on the fitting rows, clipped to [0, 1].

    A cell with fewer than `MIN_AIRPORT_ROWS` fitting rows gets weight 0 and stays at
    the base.  That is how an airport or a match mode that does not carry is excluded: by a
    number the rotation produces, never by hand.
    """
    w, n = {}, {}
    for k, g in fit.groupby(keys if len(keys) > 1 else keys[0], observed=True):
        key = "|".join(str(x) for x in k) if isinstance(k, tuple) else str(k)
        n[key] = int(len(g))
        w[key] = (_amplitude(g.rbase.to_numpy("float64"), g[dcol].to_numpy("float64"))
                  if len(g) >= MIN_AIRPORT_ROWS else 0.0)
    return w, n


def _per_airport_amplitude(fit: pd.DataFrame, dcol: str) -> tuple[dict, dict]:
    """One weight per airport, fitted on the fitting rows, clipped to [0, 1]."""
    return _per_cell_amplitude(fit, dcol, ["airport"])


def _cell_key(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    if len(keys) == 1:
        return df[keys[0]].astype(str)
    out = df[keys[0]].astype(str)
    for k in keys[1:]:
        out = out + "|" + df[k].astype(str)
    return out


def arm_a(fit: pd.DataFrame, sco: pd.DataFrame,
          keys: list[str] | None = None) -> tuple[np.ndarray, dict]:
    """The linear blend of the base and the observation.

    `keys` names the cell the weight is fitted in: `["airport"]` is the assignment's
    arm; `["airport", "obs_wide"]` lets the fold price a wide match differently from a
    strict one, which is the only quality handle a linear arm has.  The OFFSET stays per
    airport in both: it is the observation's median disagreement with the label at that
    airport and does not depend on how the row was matched.
    """
    keys = keys or ["airport"]
    off = fit.groupby("airport").apply(
        lambda g: float((g.obs_taxi - g.y_true).median()), include_groups=False).to_dict()
    fit = fit.copy()
    fit["d_raw"] = fit.obs_taxi - fit.airport.map(off) - fit.pbase
    w, n = _per_cell_amplitude(fit, "d_raw", keys)
    d = np.zeros(len(sco))
    m = sco.usable.to_numpy(bool)
    raw = (sco.obs_taxi - sco.airport.map(off) - sco.pbase).to_numpy("float64")
    wv = _cell_key(sco, keys).map(w).fillna(0.0).to_numpy("float64")
    d[m] = np.nan_to_num(raw[m] * wv[m], nan=0.0)
    return d, {"offsets_s": off, "weights": w, "fit_rows_by_cell": n,
               "weight_cell": keys}


def _lgb_fit(X: np.ndarray, y: np.ndarray, seed: int, params: dict | None = None,
             w: np.ndarray | None = None):
    """`w` is the per-row sample weight of RESULTS section 47.9's `c_lgb_covmatch`.

    When it is None the call is `m.fit(X, y)` exactly as before, so every existing caller
    keeps its bytes; a weight vector is only ever supplied by that arm.
    """
    import lightgbm as lgb                                    # noqa: PLC0415
    p = dict(LGB)
    if params:                       # the capacity sweep, `code/capacity.py`; the arm
        p.update(params)             # set itself always runs on the predeclared LGB
    p["seed"] = seed
    m = lgb.LGBMRegressor(**{k: v for k, v in p.items() if k != "objective"},
                          objective=p["objective"])
    if w is None:
        m.fit(X, y)
    else:
        m.fit(X, y, sample_weight=np.asarray(w, dtype="float64"))
    return m


def arm_lgb(fit: pd.DataFrame, sco: pd.DataFrame, cols: list[str],
            seed: int = SEED,
            params: dict | None = None,
            w: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """A LightGBM correction of the base's residual, at a per-airport amplitude.

    The amplitude is fitted on CROSS-FITTED predictions: five contiguous day blocks
    inside each fitting month, a row's prediction always from a model that did not
    see its block.  The served prediction comes from a model fitted on every fitting
    row, and never on the scored month.
    """
    Xf = fit[cols].to_numpy("float32")
    rf = fit.rbase.to_numpy("float64")
    blk = _dayblocks(fit.day)
    oof = np.zeros(len(fit))
    n_weighted = 0
    for b in np.unique(blk):
        tr = blk != b
        if tr.sum() < 100 or (~tr).sum() == 0:
            continue
        wb = None if w is None else w[tr]
        n_weighted += int(wb is not None)
        oof[~tr] = _lgb_fit(Xf[tr], rf[tr], seed, params, wb).predict(Xf[~tr])
    fitc = fit.copy()
    fitc["d_raw"] = oof
    # NOTE the amplitude is fitted PER AIRPORT and `w` is constant within an airport, so a
    # composition weight cannot move it: `c_lgb_covmatch` changes the shared tree and
    # nothing else.  That is why section 47.9 predicts a small number.
    amp, n = _per_airport_amplitude(fitc, "d_raw")
    full = _lgb_fit(Xf, rf, seed, params, w)
    n_weighted += int(w is not None)
    d = np.zeros(len(sco))
    m = sco.usable.to_numpy(bool)
    if m.any():
        pred = full.predict(sco.loc[m, cols].to_numpy("float32"))
        wv = sco.loc[m, "airport"].map(amp).fillna(0.0).to_numpy("float64")
        d[m] = np.nan_to_num(pred * wv, nan=0.0)
    return d, {"weights": amp, "fit_rows_by_airport": n, "n_features": len(cols),
               "lgb_params_overridden": dict(params) if params else None,
               "sample_weighted": w is not None,
               "n_weighted_lgb_fits": n_weighted,
               "oof_rmse_vs_r11": float(np.sqrt(np.mean((rf - oof) ** 2))),
               "r11_rmse_on_fit_rows": float(np.sqrt(np.mean(rf ** 2)))}


def shipped_cols(pan: dict[str, pd.DataFrame]) -> tuple[dict, list[str]]:
    """The shipped feature block for the three fold months, vocabulary codes dropped."""
    zs = _load("exp-044-zerosum", "zs")
    out, cols = {}, None
    for f in FOLDS:
        w = zs.shipped(FOLD_MONTH[f], drop_cache_vocab=True)
        cols = list(w.columns) if cols is None else [c for c in cols if c in w.columns]
        out[f] = w
    return out, sorted(cols)


# ---------------------------------------------------------------- the table --


def _day_interval(v: np.ndarray, day: np.ndarray,
                  rng: np.random.Generator) -> list[float]:
    """A day-clustered 95 % interval on `mean(v)`, resampling whole calendar days."""
    codes, uniq = pd.factorize(day)
    k = len(uniq)
    if k < 2:
        return [float("nan"), float("nan")]
    ssum = np.bincount(codes, weights=v, minlength=k)
    cnt = np.bincount(codes, minlength=k).astype("float64")
    dr = rng.integers(0, k, size=(BOOT, k))
    b = ssum[dr].sum(1) / cnt[dr].sum(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return [float(lo), float(hi)]


def apply_floor(pbase: np.ndarray, d: np.ndarray) -> np.ndarray:
    """The serving floor, applied to the FOLD vector so both price the same thing.

    `serve.py` floors the served prediction at `MIN_PRED`; without the same floor here
    the fold evidence prices a vector the candidate never contains (Sol round 1,
    blocking 10).  Applied only where the arm already moves the row, so an unmatched or
    `noac` row stays bit-identical to the base whatever its prediction is.
    """
    moved = d != 0
    out = d.copy()
    out[moved] = np.maximum(pbase[moved] + d[moved], MIN_PRED) - pbase[moved]
    return out


def by_group(p: pd.DataFrame, d: np.ndarray, key: str,
             rng: np.random.Generator | None = None) -> dict:
    r = p.rbase.to_numpy("float64")
    v = 2 * r * d - d * d
    day = p.day.to_numpy()
    out = {}
    for g, idx in p.groupby(key, observed=True).indices.items():
        nmov = int((d[idx] != 0).sum())
        rec = {"rows": int(len(idx)), "rows_moved": nmov,
               "mse_gain_s2": float(np.mean(v[idx])),
               # the quantity the projection needs: what the arm is worth on a row it
               # actually moves, which does not depend on this month's reach
               "mse_gain_s2_per_moved_row": (float(v[idx][d[idx] != 0].mean())
                                             if nmov else 0.0),
               "rms_update_s": float(np.sqrt(np.mean(d[idx] ** 2))),
               "rmse_base": float(np.sqrt(np.mean(r[idx] ** 2))),
               "rmse_arm": float(np.sqrt(np.mean((r[idx] - d[idx]) ** 2)))}
        if rng is not None:
            rec["mse_gain_s2_ci95"] = _day_interval(v[idx], day[idx], rng)
        out[str(g)] = rec
    return out


def shuffle_obs(pan: dict[str, pd.DataFrame], cols: list[str],
                seed: int = SEED) -> tuple[dict[str, pd.DataFrame], dict]:
    """Permute the observation among the with-clock rows of one `(airport, day)` cell.

    `exp-047-queue`'s both-clocks placebo, on this lane's panel.  The permutation moves
    the whole observation block together, so the features stay mutually consistent and
    only the identification of a row with ITS aircraft's ground run is destroyed.  Rows
    outside the class, the base prediction, the label and the day are untouched, so the
    class boundary, the rotation and the amplitude machinery are all exactly the served
    ones.  A cell of one row cannot be permuted and is left alone; the share of rows that
    keep their own observation by chance is reported rather than engineered away, because
    a derangement is not what the null says.
    """
    rng = np.random.default_rng(seed)
    out, rep = {}, {}
    for f, p in pan.items():
        q = p.copy()
        n = len(q)
        idx = np.arange(n)
        u = q.usable.to_numpy(bool)
        cell = (q.airport.astype(str).to_numpy() + "|" + q.day.astype(str).to_numpy())
        perm = idx.copy()
        cells = 0
        for _, ii in pd.Series(idx[u]).groupby(cell[u]).indices.items():
            rows = idx[u][ii]
            if len(rows) > 1:
                perm[rows] = rng.permutation(rows)
                cells += 1
        for c in cols:                     # column by column, so no dtype is widened
            q[c] = q[c].to_numpy()[perm]
        kept = int((perm[u] == idx[u]).sum())
        out[f] = q
        rep[f] = {"cells_permuted": cells, "rows_in_class": int(u.sum()),
                  "rows_that_kept_their_own_observation": kept,
                  "share_kept": round(kept / max(int(u.sum()), 1), 4)}
    return out, rep


#: The REGISTERED tail clause (`assignments/k2climb.md` line 24, as the desk names it on
#: 2026-09-08 17:35): an arm that adds new columns must not make the rows with
#: `y > 3600 s` worse by more than one second of RMSE.  tail rule treats it as a SECONDARY
#: GUARD and not a gate -- v12 itself shipped with a -3.86 s january tail reading, recorded
#: as its transfer risk -- so it is reported per fold against the clause and it does not
#: block a review round.
TAIL_CLAUSE_S = 1.0
TAIL_Y_S = 3600.0


def tail_clauses(p: pd.DataFrame, d: np.ndarray) -> dict:
    """The registered tail clause, and the update tail beside it as diagnosis.

    The registered one is the k2climb clause above: the RMSE of the rows over an hour,
    arm against base, per fold.  The UPDATE tail is this arm's own and is diagnosis only:
    the largest 1 % of |d|, which is where a correction that is really an outlier detector
    would put all of its gain.  Neither is a gate.
    """
    r = p.rbase.to_numpy("float64")
    y = p.y_true.to_numpy("float64")
    v = 2 * r * d - d * d
    tot = float(v.sum())
    long = y > TAIL_Y_S
    rmse_b = float(np.sqrt(np.mean(r[long] ** 2))) if long.any() else float("nan")
    rmse_a = float(np.sqrt(np.mean((r[long] - d[long]) ** 2))) if long.any() else float("nan")
    ad = np.abs(d)
    k = max(int(round(0.01 * int((d != 0).sum()))), 1)
    cut = np.partition(ad, -k)[-k] if (d != 0).any() else np.inf
    top = ad >= cut
    return {"total_mse_gain_s2_sum": tot,
            "registered_clause_y_over_3600s": {
                "rows": int(long.sum()), "rows_moved": int((d[long] != 0).sum()),
                "rmse_base": rmse_b, "rmse_arm": rmse_a,
                # POSITIVE is better; the clause is "not worse by more than 1 s"
                "rmse_gain_s": rmse_b - rmse_a,
                "clause_s": TAIL_CLAUSE_S,
                "passes_the_clause": bool(rmse_b - rmse_a > -TAIL_CLAUSE_S),
                "share_of_the_gain": (float(v[long].sum() / tot) if tot else 0.0),
                "mse_gain_s2_per_row": float(v[long].mean()) if long.any() else 0.0},
            "population_bulk_under_3600s": {
                "rows": int((~long).sum()),
                "share_of_the_gain": (float(v[~long].sum() / tot) if tot else 0.0),
                "mse_gain_s2_per_row": float(v[~long].mean())},
            "update_tail_top_1pct_of_abs_d": {
                "rows": int(top.sum()), "abs_d_cut_s": float(cut),
                "share_of_the_gain": (float(v[top].sum() / tot) if tot else 0.0)},
            "gain_with_the_update_tail_removed_s2_per_row": (
                float(v[~top].sum() / len(v)) if len(v) else 0.0),
            "reading": ("the registered clause is assignments/k2climb.md line 24 -- rows "
                        "with y > 3600 s not worse by more than 1 s of RMSE -- and D20 "
                        "treats it as a secondary guard, not a gate: v12 shipped with a "
                        "-3.86 s january tail reading recorded as its transfer risk.  The "
                        "update tail beside it is diagnosis only.")}


def _gain_share(p: pd.DataFrame, d: np.ndarray) -> dict:
    """What fraction of the fold's whole MSE gain each airport carries."""
    v = 2 * p.rbase.to_numpy("float64") * d - d * d
    tot = float(v.sum())
    out = {}
    for a, idx in p.groupby("airport", observed=True).indices.items():
        out[str(a)] = float(v[idx].sum() / tot) if tot else 0.0
    return out


RANK_REACH = EXP / "analysis/ranking-reach.json"
BOARD_V12 = BA.V12_BOARD


def project_to_ranking(by_airport: dict[str, dict],
                       fold_board: float | None = None) -> dict:
    """The fold's per-MOVED-row gain per airport, projected onto each RANKING AIRPORT-MONTH.

    The desk's rule of 2026-09-08 16:50 and its correction of 17:05, and the pilot's
    method (`exp-043` §5).  The 2025 feeder is a different instrument from the 2026 one --
    EDDM at 0 % receipt for the first half of July 2025 and 99 % from the 19th, LEMD at 55
    against 99.4 in January, EGLL climbing 12 -> 65 against 93.1 -- so a fold-side board
    gain reads the 2025 network's reach and not the flights.  And the 2026 network is not
    one number per airport either: reach is 73.3 % over the January ranking rows and 64.5
    over the July ones, because LIRF is dark for all 31 days of July 2026 while it is
    reached in January.  So the unit of projection is the airport-MONTH.

    What transfers is the per-moved-row gain at an airport; what a ranking airport-month
    supplies is how many of ITS rows the matcher reaches.  The two are multiplied per
    airport-month and never mixed inside one fold number.

    Two readings of "per airport-month" are reported.  `by_airport_month` pools the
    per-moved-row gain over every fold that carries the airport, which is the desk's rule
    read on the reach alone.  `seasonal` is the sensitivity: the July ranking month takes
    the July fold's gain and the January ranking month takes December's and January's, so
    a seasonal difference in the correction itself would show as a difference between the
    two totals.

    `analysis/ranking-reach.json` is written by `serve.ranking_block()` on the real
    ranking file; without it there is no projection and the absence is reported.
    """
    if not RANK_REACH.exists():
        return {"available": False, "why": f"{RANK_REACH.name} has not been written yet; "
                                           "run serve.py --self-test to build it"}
    reach = json.loads(RANK_REACH.read_text())
    am = reach.get("by_airport_month")
    if not am:
        return {"available": False, "why": f"{RANK_REACH.name} carries no by_airport_month; "
                                           "re-run serve.py --self-test to rebuild it"}
    tot_rows = sum(v["rows"] for v in am.values())

    def pooled(a: str, folds: list[str]) -> tuple[float, int]:
        """The per-moved-row gain at one airport, over the named folds only."""
        gs = [by_airport[f][a]["mse_gain_s2_per_moved_row"] for f in folds
              if f in by_airport and a in by_airport[f]
              and by_airport[f][a]["rows_moved"]]
        ws = [by_airport[f][a]["rows_moved"] for f in folds
              if f in by_airport and a in by_airport[f]
              and by_airport[f][a]["rows_moved"]]
        if not gs:
            return 0.0, 0
        return float(np.average(gs, weights=ws)), int(sum(ws))

    every = list(by_airport)
    # the July ranking month is projected from the July fold, the January one from the two
    # winter folds; an airport with no fold rows on that side falls back to every fold
    SEASON = {"2026-07": ["july_loo"], "2026-01": ["december", "january"]}

    out = {"available": True, "basis": reach.get("basis"), "board_base": BOARD_V12,
           "unit": "airport-month", "by_airport_month": {}, "by_month": {},
           "seasonal": {"by_airport_month": {}, "by_month": {}}}
    dmse = 0.0
    dmse_season = 0.0
    per_month: dict[str, float] = {}
    per_month_season: dict[str, float] = {}
    for key, rv in sorted(am.items()):
        a, m = key.split(" ")
        g, moved = pooled(a, every)
        share = rv["with_clock"] / rv["rows"] if rv["rows"] else 0.0
        contrib = rv["with_clock"] * g / tot_rows if tot_rows else 0.0
        dmse += contrib
        per_month[m] = per_month.get(m, 0.0) + contrib
        out["by_airport_month"][key] = {"fold_gain_s2_per_moved_row": g,
                                        "fold_rows_moved": moved,
                                        "ranking_rows": rv["rows"],
                                        "ranking_with_clock": rv["with_clock"],
                                        "ranking_reach": round(share, 4),
                                        "mse_s2_per_ranking_row": contrib}
        gs_, moved_s = pooled(a, SEASON.get(m, every))
        if moved_s == 0:                       # no fold rows on that side of the year
            gs_, moved_s = g, moved
        c_s = rv["with_clock"] * gs_ / tot_rows if tot_rows else 0.0
        dmse_season += c_s
        per_month_season[m] = per_month_season.get(m, 0.0) + c_s
        out["seasonal"]["by_airport_month"][key] = {
            "folds": SEASON.get(m, every), "fold_gain_s2_per_moved_row": gs_,
            "fold_rows_moved": moved_s, "mse_s2_per_ranking_row": c_s}

    for m in sorted(per_month):
        rows_m = sum(v["rows"] for k, v in am.items() if k.endswith(m))
        clk_m = sum(v["with_clock"] for k, v in am.items() if k.endswith(m))
        out["by_month"][m] = {"ranking_rows": rows_m, "ranking_with_clock": clk_m,
                              "ranking_reach": round(clk_m / rows_m, 4) if rows_m else 0.0,
                              "mse_s2_per_ranking_row": per_month[m]}
        out["seasonal"]["by_month"][m] = {"mse_s2_per_ranking_row": per_month_season[m]}

    def board(d: float) -> tuple[float, float]:
        after = float(np.sqrt(max(BOARD_V12 ** 2 - d, 0.0)))
        return after, BOARD_V12 - after

    # THE TRANSFER THE PROJECTION CANNOT SEE.  The fold-side board and the leaderboard
    # board are not the same scale: v12's own fold-side board is about 308.6 s and its
    # board reading is 262.7418.  The paired gain is `2*mean(r*d) - mean(d*d)`, and its
    # first term is linear in the base's OWN residual, so a population where the base errs
    # less offers a smaller gain for the same update.  The conservative variant scales the
    # per-moved-row gain by the ratio of the two MSEs; the truth is between the two, and
    # both are reported rather than one being chosen.
    scale = (BOARD_V12 ** 2) / (fold_board ** 2) if fold_board else 1.0
    out["fold_side_board_base"] = fold_board
    out["conservative"] = {
        "scale_on_the_per_moved_row_gain": scale,
        "why": ("the fold-side board is the base's OOF error on 2025 and the leaderboard "
                "board is its error on the 2026 ranking file; the paired gain's first "
                "term is linear in the base's own residual, so a projection that carries "
                "the fold's per-moved-row gain unchanged assumes the base errs as much on "
                "the ranking rows as on the fold rows, which it does not"),
        "projected_mse_reduction_s2_per_ranking_row": dmse * scale}
    c_after, c_gain = board(dmse * scale)
    out["conservative"]["projected_board"] = c_after
    out["conservative"]["projected_board_gain_s"] = c_gain
    out["projected_mse_reduction_s2_per_ranking_row"] = dmse
    out["projected_board"], out["projected_board_gain_s"] = board(dmse)
    s_after, s_gain = board(dmse_season)
    out["seasonal"]["projected_mse_reduction_s2_per_ranking_row"] = dmse_season
    out["seasonal"]["projected_board"] = s_after
    out["seasonal"]["projected_board_gain_s"] = s_gain
    out["note"] = ("a projection, not a measured board move: it assumes the per-moved-row "
                   "gain measured on a 2025 fold holds on the 2026 rows the matcher "
                   "reaches at the same airport, and it is stated separately from every "
                   "fold-side number, which stays on the contract.  The board share of "
                   "the two ranking months is not used here: the projection is a mean "
                   "over the whole ranking file, which is what the RMSE of the file is.")
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    pan = panel_with_block(BA.load())
    ship, scols = (None, [])
    try:
        ship, scols = shipped_cols(pan)
    except Exception as e:                                    # noqa: BLE001
        print(f"[warn] the shipped block is unavailable: {e}")

    obs_cols = [c for c in B.FEATURES if c not in ("stand_lat", "stand_lon")]
    pan_shuf, shuf_report = shuffle_obs(pan, obs_cols)
    nbr_report = {FOLD_MONTH[f]: _NBR.get(FOLD_MONTH[f], {"available": False})
                  for f in FOLDS}
    nbr_ok = all(v.get("available") for v in nbr_report.values())
    rule, rule_report = serving_rule()
    rotation = {
        "scored": "one of july_loo (2025-07), december (2025-12), january (2025-01)",
        "fitted_on": "the other two of those three",
        "deviation_from_the_literal_wording": (
            "the assignment says 'fit on the 2025 months other than the scored fold', "
            "which reads as nine or eleven months. Only 2025-01, 2025-07 and 2025-12 "
            "carry an OUT-OF-FOLD base prediction: v12 is the same composition as v11 with "
            "the climbed member in the head's place, and "
            "its head, v7 and stack-5 vectors exist for the three fold panels only, "
            "so `rbase = y - pbase` -- the quantity every weight in every arm is fitted "
            "against -- does not exist on the other nine months. Fitting on them is not "
            "a choice this stage declined; it is a quantity the base does not have. The "
            "airport OFFSET of arm (a) needs no residual and could be fitted on any "
            "labelled month, and is deliberately fitted on the same two so that ONE "
            "rotation describes the whole arm rather than two. Sol round 1, blocking 5, "
            "is right that this is a deviation; it is recorded, not repaired, and the "
            "repair would be a fold-side base on all twelve months, which is exp-038's "
            "artifact and not this stage's."),
        "prefix_map_months": prefix_months_for(""),
        "prefix_map_note": (
            "the airline prefix map that reconstructs a missing broadcast callsign is "
            "learned on the nine 2025 months that are NOT fold months, so it is out of "
            "fold for every fold at once (Sol round 1, blocking 6)."),
    }
    rep = {"predeclared": PRE, "coverage": {}, "arms": {}, "rotation": rotation,
           "neighbour_block": nbr_report,
           "rule": {"sha256": rule, "cover": rule_report,
                    "months_resolved_together": list(FOLD_MONTH.values()) + RANK_MONTHS,
                    # the serve compares these against its own before it trusts this
                    # table's gate: the harvest can re-derive a month between the two
                    # runs, and then the serve would fit under one version and gate on a
                    # table priced under another (round 2 FALLBACK, blocking 1)
                    "block_cache_keys": {m: block_cache_key(m, rule)
                                         for m in FOLD_MONTH.values()}},
           }
    for f in FOLDS:
        p = pan[f]
        rep["coverage"][f] = {
            "rows": int(len(p)), "usable": int(p.usable.sum()),
            "usable_share": round(float(p.usable.mean()), 4),
            "by_airport": {a: round(float(g.usable.mean()), 4)
                           for a, g in p.groupby("airport", observed=True)},
            "by_reach_class": p.reach_class.value_counts().to_dict(),
            "handover_to_exp_047": handover(p)}

    for arm in ARMS:
        if arm.startswith("c_lgb_shipped") and ship is None:
            rep["arms"][arm] = {"skipped": "the shipped feature block is unavailable"}
            continue
        d_by_fold, detail = {}, {}
        # a placebo arm is the named arm, fitted and served on the permuted observation;
        # it is PRICED against the real panel, so `r`, the day and the class are the real
        # ones and only the identity of the match has been destroyed
        src = pan_shuf if arm in PLACEBO_OF else pan
        base_arm = PLACEBO_OF.get(arm, arm)
        for f in FOLDS:
            others = [g for g in FOLDS if g != f]
            fit = pd.concat([src[g][src[g].usable] for g in others], ignore_index=True)
            sco = src[f]
            if not len(fit) or not sco.usable.any():
                d_by_fold[f] = np.zeros(len(sco))
                detail[f] = {"skipped": "no fitting rows or no usable scored row",
                             "fit_rows": int(len(fit)),
                             "scored_usable": int(sco.usable.sum())}
                continue
            if base_arm == "a_linear":
                d, info = arm_a(fit, sco, ["airport"])
            elif base_arm == "a_linear_reach":
                d, info = arm_a(fit, sco, ["airport", "obs_wide"])
            else:
                # the control carries no observation feature at all
                cols = [] if base_arm == "c_lgb_shipped_noobs" else list(obs_cols)
                if base_arm.endswith("_nbr"):
                    if not nbr_ok:
                        rep["arms"][arm] = {"skipped": nbr_report}
                        d_by_fold = None
                        break
                    cols += [c for c in NBR_COLS if c in fit.columns]
                if base_arm.startswith("c_lgb_shipped"):
                    fit = fit.join(pd.concat([ship[g] for g in others])
                                   .reindex(fit.MVT_ID_mvt.to_numpy("float64"))
                                   .reset_index(drop=True))
                    sco = sco.join(ship[f].reindex(sco.MVT_ID_mvt.to_numpy("float64"))
                                   .reset_index(drop=True))
                    cols += scols
                d, info = arm_lgb(fit, sco, cols)
            d = apply_floor(pan[f].pbase.to_numpy("float64"), d)
            info["fit_rows"] = int(len(fit))
            info["scored_usable"] = int(sco.usable.sum())
            info["floored_rows"] = int(((pan[f].pbase.to_numpy("float64") + d) <= MIN_PRED
                                        ).sum())
            # the two structural checks, on every fold of every arm
            assert not np.any(d[pan[f].noac.to_numpy(bool)]), f"{arm} {f}: a noac row moved"
            assert not np.any(d[~pan[f].usable.to_numpy(bool)]), \
                f"{arm} {f}: a row outside the with-clock class moved"
            d_by_fold[f] = d
            detail[f] = info
        if d_by_fold is None:                # the neighbour block is not on disk yet
            continue
        rec = BA.price(pan, d_by_fold, rng)
        rec["detail"] = detail
        rec["by_airport"] = {f: by_group(pan[f], d_by_fold[f], "airport", rng)
                             for f in FOLDS}
        rec["by_reach_class"] = {f: by_group(pan[f], d_by_fold[f], "reach_class", rng)
                                 for f in FOLDS}
        # the desk's 17:20 line: the per-airport table on the fold's OWN with-clock rows,
        # which is the population the arm serves; `by_airport` above is every row of the
        # airport and so mixes the arm's gain with the airport's reach
        rec["by_airport_with_clock"] = {
            f: by_group(pan[f][pan[f].usable.to_numpy(bool)],
                        d_by_fold[f][pan[f].usable.to_numpy(bool)], "airport", rng)
            for f in FOLDS}
        rec["share_of_the_fold_gain_by_airport"] = {
            f: _gain_share(pan[f], d_by_fold[f]) for f in FOLDS}
        rec["tail_clauses"] = {f: tail_clauses(pan[f], d_by_fold[f]) for f in FOLDS}
        if arm in PLACEBO_OF:
            rec["placebo"] = {"of_arm": PLACEBO_OF[arm], "permutation": shuf_report,
                              "null": ("the observation permuted within (airport, day) "
                                       "among rows that all carry a clock")}
        rec["projection_to_the_ranking_months"] = project_to_ranking(
            rec["by_airport"], rec["board_all_row"]["base"])
        if d_by_fold is None:                # the neighbour block is not on disk yet
            continue
        rep["arms"][arm] = rec
        np.savez_compressed(WORK / f"d-{arm}.npz", **{f: d_by_fold[f] for f in FOLDS})
        proj = rec["projection_to_the_ranking_months"].get(
            "projected_board_gain_s", float("nan"))
        print(f"{arm}: board all-row gain {rec['board_all_row']['gain_s']:+.4f} s, "
              f"clean {rec['board_clean']['gain_s']:+.4f} s, "
              f"december {rec['december_rmse']['gain_s']:+.4f} s, "
              f"every fold clear {rec['every_fold_interval_clear']} "
              f"(moving folds {rec['every_moving_fold_interval_clear']}, "
              f"empty {rec['folds_with_no_moved_row']}), "
              f"projected {proj:+.4f} s")

    (EXP / "analysis/arms.json").write_text(json.dumps(rep, indent=2, sort_keys=True,
                                                       default=float))


if __name__ == "__main__":
    main()
