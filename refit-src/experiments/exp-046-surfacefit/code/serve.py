"""exp-046-surfacefit step 3: the served candidate.

    candidate = base  +  the arm's update on the ranking rows with a clock of their own

with the 388 `noac` rows, every unmatched row and every row the matcher found but which
carries no off-block instant of its own **bit-identical to the base**, template
order, finite, min >= 60.  **The base is v12** (desk, 2026-09-08 14:50, tail rule): v11 with
only its K2 member replaced by exp-035 arm 3b's climbed member, board 262.7418, the file
`submissions/candidates/exp-035-k2climb-v11member.parquet` (SHA-256 `48dd572d...`).
v12 is v11 exactly at EDDF and EHAM -- the airports this lane's observation reaches --
so the fold-side gains carry across the base change unchanged there.

**The class boundary** (desk, 2026-09-08 13:40, from `exp-047-queue` step 1).  The head
serves `arms.SERVED_CLASS` and nothing else: a head carrying the row's own observed taxi
reads **+6,884 s2 on the rows with a clock and -588 s2 on the rows without one** when it
spans both, so the rows with no clock stay at the base here and are `exp-047-queue`'s, whose
neighbourhood block serves them **+583 s2** and arrives as
`artifacts/exp-047/block/<month>.parquet` keyed by `MVT_ID_mvt`.  The two lanes share one
class boundary by construction: that lane imports this lane's `block.py` as the matcher
and keys on the same `obs_has_clock`.  `analysis/serve.json` carries the handover census.

**The serving fit.**  The arm is refitted on every fold month that carries an
observation (2025-01, 2025-07, 2025-12 -- the three months with an out-of-fold base
residual), and applied to the two ranking months.  No ranking-month label exists, so no
part of this fit can see the rows it serves; the amplitude is per airport, exactly as in
`code/arms.py`, and an airport whose fitting rows fall below `MIN_AIRPORT_ROWS` gets
weight 0 and is served the base unchanged.

**The gate the candidate has to pass before the desk uploads it** (`assignments/surfacefit.md`):
an all-row paired fold gain over the base with every fold interval clear of zero, and a served
update whose RMS per airport is within a third of the fold side's.  Both are checked here
and written into `analysis/serve.json`; a failure is reported, never worked around.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/serve.py [arm]
"""
from __future__ import annotations

import hashlib
import copy
import json
from datetime import datetime, timezone
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "code"))
import arms as A                                             # noqa: E402
import base as BA                                            # noqa: E402
import block as B                                            # noqa: E402
import guard as G                                            # noqa: E402
from imports import load as _load                            # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D                                         # noqa: E402

BASE_FILE = ROOT / "submissions/candidates/exp-035-k2climb-v11member.parquet"
BASE_SHA = BA.V12_SHA
OUT = ROOT / "submissions/candidates/exp-046-surface.parquet"
OUT_V2 = ROOT / "submissions/candidates/exp-046-surface-v2.parquet"
OUT_V3 = ROOT / "submissions/candidates/exp-046-surface-v3.parquet"
OUT_V4 = ROOT / "submissions/candidates/exp-046-surface-v4.parquet"
OUT_V5 = ROOT / "submissions/candidates/exp-046-surface-v5.parquet"
OUT_V6 = ROOT / "submissions/candidates/exp-046-surface-v6.parquet"
OUT_V7 = ROOT / "submissions/candidates/exp-046-surface-v7.parquet"
OUT_V8 = ROOT / "submissions/candidates/exp-046-surface-v8.parquet"
OUT_V9 = ROOT / "submissions/candidates/exp-046-surface-v9.parquet"
OUT_V10 = ROOT / "submissions/candidates/exp-046-surface-v10.parquet"
OUT_V11 = ROOT / "submissions/candidates/exp-046-surface-v11.parquet"
OUT_V12 = ROOT / "submissions/candidates/exp-046-surface-v12.parquet"

#: **THE UNGATED CHAIN ROWS, AT EXACTLY THE SECTION 28.2 GATE** (desk, 2026-09-09: "BUILD
#: THE CANDIDATE. Contents: v18's turnaround pass plus the gated ungated rows at exactly
#: the 28.2 gate (step 0, (i) 1,299 s, (ii) runway agreement)").
#:
#: `--ungated` routes BOTH sides -- the fold panel every gate is priced on and the ranking
#: block the file is written from -- through `analysis/chain-check/block_ungated.py`, which
#: admits every unique turnaround pair the +-60 s clause refuses as `match_mode` 4 and
#: carries the three check values out with the block, and then through
#: `chainfold.restrict`, which blanks every mode-4 row the gate refuses so it reads exactly
#: as unmatched.  One switch moves both sides, because a serve that moved only one of them
#: would gate a candidate on a panel it was not fitted on.
#:
#: **The three numbers are not chosen here and no fold chose them.**  Step 0 is the p95 of
#: `t_airborne - obs_wheels_off` on the SHIPPED ranking rows (27.1); 1,299 s is the shipped
#: ranking rows' p95 of |P - AOBT| (27.3); runway agreement has no threshold.  All three
#: were fixed on the ranking side before any fold was priced, and section 28.3's
#: cross-fitted pick equals the in-fold pick with a selection premium of zero -- which is
#: why the base set below records the update as UNSELECTED.
UNGATED_GATE = {
    "step_0_gap_cut_s": 62.405,
    "check_i_abs_dt_offblock_max_s": 1299.0,
    "check_ii_runway_agreement": True,
    "probe": "experiments/exp-046-surfacefit/analysis/chain-check/block_ungated.py",
    "restriction": "chainfold.restrict",
    "priced_in": "analysis/chain-bundle.json (RESULTS section 28.2)",
    "thresholds_were_fixed_on": (
        "the ranking side before any fold was priced: step 0 is the shipped rows' p95 of "
        "t_airborne - obs_wheels_off (27.1) and 1,299 s is the shipped rows' p95 of "
        "|P - AOBT| (27.3); the cross-fitted pick equals the in-fold pick and the "
        "selection premium is zero (28.3)"),
    "the_wrong_placements_are_inside_the_number": (
        "the gated subset's estimated precision is 0.8958, so about a tenth of the rows "
        "carry a fabricated clock; they are inside every fold interval section 28.2 "
        "quotes, which is what the desk means by pricing the trade"),
}
#: set by `main()` from `--ungated`, read by `ranking_block()`: one switch, both sides
_UNGATED = False
_UNGATED_MODULES: tuple | None = None

#: **A base has three roles and after v14 they are three different files.**  Serving on
#: v12 they coincide, which is why one `BASE_FILE` sufficed until now:
#:
#:   bytes   the file every row this lane does NOT serve is copied from, byte for byte.
#:           On v14 that carries `exp-047-queue`'s no-clock update, which must survive.
#:   price   the prediction the arm's update is ADDED to.  The arm is fitted against
#:           `rbase = y - p_v12`, so its update belongs on p_v12 and on nothing else.
#:           v14's with-clock rows already carry v13's update (measured: v14 == v13 on
#:           every `obs_has_clock` row, 0 differences), so adding a new update to v14's
#:           value on those rows would apply two heads to one class.
#:   head    the previously served candidate, for the INCREMENT gate: what this upload
#:           actually changes is `p_new - p_v13` on the with-clock rows, not `p_new - p_v12`.
#:
#: The three are named together in one predeclared set so a serve cannot mix them.
BASE_SETS = {
    "v12": {
        "bytes": BASE_FILE, "bytes_sha": BASE_SHA,
        "price": BASE_FILE, "price_sha": BASE_SHA,
        "head_file": None, "head_sha": None, "head_arm": None,
        "out": OUT,
        "what": "v12 in all three roles: the first serve of this class, nothing under it",
    },
    "v14": {
        "bytes": ROOT / "submissions/candidates/exp-047-queue.parquet",
        "bytes_sha": ("519ca89f381056afb453ca62d64df622"
                      "fcdbbdb14d8607cb0307c2be077676b2"),
        "price": BASE_FILE, "price_sha": BASE_SHA,
        "head_file": OUT,
        "head_sha": ("79a301627e918140933bb4c1dcf9a924"
                     "9a63bd81669f1793e252f821aa1dee51"),
        "head_arm": "c_lgb_shipped",
        # **The head's own occupancy**, which is NOT "differs from the price base": the
        # bytes file can carry another lane's update, and that is a difference from v12
        # this lane never wrote.  v13's occupancy is the `moved_by_v13` column of the
        # vector published for the composition.
        "head_class": {
            "column": "moved_by_v13",
            "months": {
                "2026-01": ("artifacts/exp-046/class/2026-01.parquet",
                            "62d06e22dee90c526ad507ee41ad3c9d"
                            "f136c714afffe0e66c7beb7e9e311fc0"),
                "2026-07": ("artifacts/exp-046/class/2026-07.parquet",
                            "273010410e1caedc85eb50cf38360a5a"
                            "503bab21c1bcfd1f31d112afcfeb0c6f")}},
        "out": OUT_V2,
        "what": ("v14's bytes for every row this lane does not serve (queue's no-clock "
                 "update stays), v12 as the prediction the update is added to, and v13 "
                 "as the head the increment is priced against"),
    },
    # v16: the tail-guarded arm on v15's bytes.  Here the BYTES and the HEAD are the same
    # file -- v15 is both the last thing uploaded and the last thing this lane served --
    # which is the ordinary case once a lane serves twice in a row.  Queue's no-clock
    # update rides inside v15's bytes and is copied forward untouched.
    "v15": {
        "bytes": OUT_V2,
        "bytes_sha": ("eabd824d925f15223db452557fe113e2"
                      "86ca8ce49302e8a97e79f060c0360ea1"),
        "price": BASE_FILE, "price_sha": BASE_SHA,
        "head_file": OUT_V2,
        "head_sha": ("eabd824d925f15223db452557fe113e2"
                     "86ca8ce49302e8a97e79f060c0360ea1"),
        "head_arm": "c_lgb_shipped@leaves_127x300_lr050",
        # here the bytes and the head are one file, so "differs from the price base" would
        # sweep in every one of `exp-047-queue`'s 63,862 no-clock rows -- they differ from
        # v12 because QUEUE moved them, not because this lane's head did.  The v15 vector's
        # `occupied_by_exp_046` column exists for exactly this question (round 4, finding 4).
        "head_class": {
            "column": "occupied_by_exp_046",
            "months": {
                "2026-01": ("artifacts/exp-046/class/2026-01-v15.parquet",
                            "48e9d3de571e04fbf470c3e83c0751c2"
                            "600db086e2ea1daa9a2d558a04d67958"),
                "2026-07": ("artifacts/exp-046/class/2026-07-v15.parquet",
                            "2def4fd8ec448e14b5584547a56fa6d5"
                            "34889744d75dec3fe928361fedd770c9")}},
        "out": OUT_V3,
        "what": ("v15's bytes for every row this lane does not serve, v12 as the "
                 "prediction the update is added to, and v15's own head "
                 "(127 leaves, unguarded) as the head the increment is priced against"),
    },
    # v17: the SAME arm as v16 on a new matcher.  `122c7d84` adds the turnaround pass, so
    # 3,179 ranking rows enter this lane's class and 2,730 of them enter its occupancy;
    # nothing about the arm, the rotation, the seed or the guard's quantile changes.
    "v17": {
        # v17's bytes: v16 with `exp-047-queue`'s lit-rule update written inside them.
        # Every row this lane does not write is copied from here untouched, queue's update
        # included -- the same carry-forward v15 and v16 did.
        "bytes": ROOT / "submissions/uploaded/youthful-giraffe_v17.parquet",
        "bytes_sha": ("4de323fc4fd9d4865b201b0dd0da1ddc"
                      "bf7de9e11f1d9bc8af07d293585001ea"),
        "price": BASE_FILE, "price_sha": BASE_SHA,
        # the head is what THIS lane last wrote, which is v16's candidate; queue's update
        # rides in the bytes above and is not this lane's head
        "head_file": OUT_V3,
        "head_sha": ("6f8368da9b8da818a726c7cc6cf20de6"
                     "ff7777196f1e78d41015a8cda441fdf7"),
        "head_arm": "c_lgb_shipped@leaves_255x300_lr050+guard995",
        # SAME ARM, different block -- so the head's fold vector cannot be `npz_path(arm)`,
        # which the re-price overwrote in place.  `code/blockmove.py` saved the served
        # arm's vector at 07:30 on 2026-09-09 (commit a671e65).
        #
        # **WHAT THAT VECTOR ACTUALLY IS** (review round 6, finding 2 -- this block
        # previously claimed otherwise and the claim was false).  It is the arm on matcher
        # `eded54e5`, fitted on the pins of that hour: july **287** airport-days and
        # december **288**.  It is NOT `6fdf3951`, and NOT the july `8179f744` /
        # january `cb4d6289` pins this serve reads.  So the increment it gives is NOT the
        # matcher's effect alone: +0.1182 = 9.29855 - 9.18036 decomposes into about
        # **+0.054 the matcher on identical pins** and **+0.064 the rebuild and the
        # population re-writes**.  The pass-only increment is priced separately, by
        # refitting the head on the surviving `6fdf3951` caches
        # (`work/block-2025-*-6fdf3951.parquet`), which carry the july and january pins
        # this serve reads; `code/passonly.py` does that and reports per-fold intervals.
        "head_npz": EXP / "work/d-blockmove-guard995.npz",
        "head_npz_sha": ("f587c51c56470ad28317a744b885e539"
                         "14578d83e9736da4532396aa7e3530f4"),
        # the v18 vector, emitted against v17's bytes at the cut that WROTE them (1864.37,
        # not this re-price's 1870.64).  `occupied_by_exp_046` is 191,739 and 3,256 of
        # those are rows the bytes leave at the price base -- 526 from the rebuild and
        # 2,730 the turnaround pass just added -- which this serve writes for the first
        # time.  The desk, 2026-09-09: the next serve writes EVERY occupied row.
        "head_class": {
            "column": "occupied_by_exp_046",
            "months": {
                "2026-01": ("artifacts/exp-046/class/2026-01-v18.parquet",
                            "0b4ee2dfacda4ac56d5658da23d2ff52"
                            "d664102d1937f8f7f16bafb98c1d8e2f"),
                "2026-07": ("artifacts/exp-046/class/2026-07-v18.parquet",
                            "793afae335f7b18066ec064e4b2e3894"
                            "c6aca6fafdfa12c2df32cf7e15fefad4")}},
        # AN UNSELECTED UPDATE: nothing here is chosen on fold evidence.  Same arm, same
        # rotation, seed, guard quantile and hold as the head; the only change is the
        # matcher, which adds rows and picks nothing.  Desk ruling 2026-09-09 -- see
        # `increment_gate`.
        "unselected": True,
        "out": OUT_V4,
        "what": ("v17's bytes for every row this lane does not serve, v12 as the "
                 "prediction the update is added to, and v16's head as the head the "
                 "increment is priced against. ROUND 6 FINDING 2: that head vector is "
                 "the arm on matcher eded54e5 at july 287 / december 288 airport-days, "
                 "NOT at this serve's pins, so its increment carries the rebuild and the "
                 "population re-writes as well as the turnaround pass; code/passonly.py "
                 "isolates the pass on the surviving 6fdf3951 caches and analysis/"
                 "pass-only.json carries that reading with per-fold intervals"),
    },
    # v19: THE BUNDLE.  v17's bytes again -- v18 was HELD as a MAYBE and never uploaded,
    # so the last file on the board is still v17 -- carrying v18's turnaround pass AND the
    # gated ungated chain rows, with the arm refitted on the class that holds both.  Run
    # with `--ungated`, which is what puts those rows in the class on BOTH sides.
    "v19": {
        "bytes": ROOT / "submissions/uploaded/youthful-giraffe_v17.parquet",
        "bytes_sha": ("4de323fc4fd9d4865b201b0dd0da1ddc"
                      "bf7de9e11f1d9bc8af07d293585001ea"),
        "price": BASE_FILE, "price_sha": BASE_SHA,
        # what this lane last WROTE is still v16's candidate: v18 never shipped
        "head_file": OUT_V3,
        "head_sha": ("6f8368da9b8da818a726c7cc6cf20de6"
                     "ff7777196f1e78d41015a8cda441fdf7"),
        "head_arm": "c_lgb_shipped@leaves_255x300_lr050+guard995",
        # **THE HEAD VECTOR IS THE SERVED HEAD ON THE PRE-PASS MATCHER**, refitted by
        # `code/passonly.py` on the surviving `6fdf3951` caches -- the repair round 6
        # finding 2 demanded.  Its pins, PER FOLD, because they are not all this serve's
        # (exp-047-queue's audit, section 7al, e5facc8):
        #
        #   2025-07  `8179f74417c8cb99`  296 airport-days -- IDENTICAL to this serve's
        #   2025-01  `cb4d6289cb421aa2`  218 airport-days -- IDENTICAL to this serve's
        #   2025-12  `34a155e40aac9ff6`  288 airport-days -- ONE POPULATION BEHIND this
        #            serve's `d35b7171d7062341` (297). The nine days between them are the
        #            nine `2025-11-30` airport-days, which are PREFIX rows feeding the
        #            arrival side and are not scored december rows; verified against
        #            `analysis/fold-population.json`, whose december day list carries
        #            exactly nine `2025-11-30` entries.
        #
        # So "the served head at this serve's own pins" is exact on july and january --
        # and `SHARES` is those two folds, so the BOARD number is at this serve's pins --
        # and approximate on december, which carries no board share. Queue's audit
        # reproduces section 28.2 to 0.0 per fold and every fold clears under ten seeds.
        "head_npz": EXP / "work/d-passonly-6fdf3951-guard995.npz",
        "head_npz_sha": ("e137939b74070180192544134b28a2ed"
                         "49011071aa29449598fb9de87953029a"),
        # and the CANDIDATE's own fold vector, which `npz_path` cannot name either: the
        # arm's name is unchanged and its saved vector is v18's fit on the shipped block.
        # This is the bundle's fit on the ungated class (`code/chainbundle.py`), the
        # vector section 28.2 was priced from.
        "arm_npz": EXP / "work/d-bundle-ungated1299-guard995.npz",
        "arm_npz_sha": ("5dbbcb8c7d14831fd75debde479071cd"
                        "2a2eff9b7ff9b7469b6625ce00238684"),
        # the head's occupancy is the v18 class vector, emitted against these same v17
        # bytes; the class only GROWS here (2,493 ranking rows), so nothing it occupies
        # leaves this serve's class and nothing is reverted
        "head_class": {
            "column": "occupied_by_exp_046",
            "months": {
                "2026-01": ("artifacts/exp-046/class/2026-01-v18.parquet",
                            "0b4ee2dfacda4ac56d5658da23d2ff52"
                            "d664102d1937f8f7f16bafb98c1d8e2f"),
                "2026-07": ("artifacts/exp-046/class/2026-07-v18.parquet",
                            "793afae335f7b18066ec064e4b2e3894"
                            "c6aca6fafdfa12c2df32cf7e15fefad4")}},
        # UNSELECTED, and here that is a measured claim rather than a description: the
        # arm, the rotation, the seed, the guard quantile and the hold are the head's, and
        # the two thresholds that define the added class were fixed on the RANKING side
        # before any fold was priced (27.1, 27.3).  Section 28.3 cross-fitted the pick and
        # got the in-fold pick back with a premium of zero.  It is recorded and never
        # applied: `increment_gate` gates on every fold interval whatever this says.
        "unselected": True,
        "out": OUT_V5,
        "what": ("v17's bytes for every row this lane does not serve, v12 as the "
                 "prediction the update is added to, and the SERVED HEAD AT THIS SERVE'S "
                 "PINS (code/passonly.py, work/d-passonly-6fdf3951-guard995.npz) as the "
                 "head the increment is priced against. The candidate is the bundle: the "
                 "turnaround pass plus the ungated chain rows at the section 28.2 gate, "
                 "the arm refitted on the class carrying both. Priced whole in "
                 "analysis/chain-bundle.json: +0.1918 [+0.1267, +0.2797] all-row, clean "
                 "+0.2341, every fold interval clear, smallest lower bound +42.49 s2/row "
                 "(december). The head vector's pins are july 8179f744 and january "
                 "cb4d6289, identical to this serve's, and december 34a155e4 (288 "
                 "airport-days) against this serve's d35b7171 (297); the nine days "
                 "between them are 2025-11-30 prefix rows and not scored rows"),
    },
    "v20": {
        # **v18's BYTES**, which are v5's byte for byte (both 4d75706b...): the file on
        # the board at reading 17, 246.3605, rank 1 of 83.
        "bytes": ROOT / "submissions/uploaded/youthful-giraffe_v18.parquet",
        "bytes_sha": ("4d75706bfcc13d88f7556265cd008b3d"
                      "3a312555955dca8eb528e4db5325fee8"),
        "price": BASE_FILE, "price_sha": BASE_SHA,
        # the head is what this lane last WROTE and what is on the board: v5's file
        "head_file": OUT_V5,
        "head_sha": ("4d75706bfcc13d88f7556265cd008b3d"
                     "3a312555955dca8eb528e4db5325fee8"),
        "head_arm": "c_lgb_shipped@leaves_255x300_lr050+guard995",
        "head_npz": EXP / "work/d-bundle-ungated1299-guard995.npz",
        "head_npz_sha": ("5dbbcb8c7d14831fd75debde479071cd"
                         "2a2eff9b7ff9b7469b6625ce00238684"),
        # this is true of THIS base set's head-and-arm pairing and of no other, so it
        # lives here and not in the file-keyed HEAD_VECTOR_PROVENANCE (round 9, N1)
        "head_npz_note": ("The rider this base set serves is that same fit with the LFPG "
                          "hold lifted, so the increment over it carries NO refit"),
        # **THE CANDIDATE'S VECTOR CARRIES NO REFIT.**  `work/d-rider-lfpg-on-v19.npz` is
        # the head's OWN fit with the LFPG hold lifted and spliced; `code/riderread.py`
        # checks, BEFORE it prices, that it equals the head vector off LFPG on 0 rows
        # differing on all three folds (`analysis/rider-read.json`).  That matters here
        # more than anywhere else: section 34.1 measured that any two independent fits of
        # this arm differ by ~18-21 s per row of churn orthogonal to the residual, worth
        # about -0.06 s of board, and three unrelated changes all read exactly that.  This
        # increment has none of it inside: rms |inc| 4.2 s on july and 6.5 s on december,
        # with 2A/B = 2.20 and 2.24 against a refit's 0.73-1.05.
        "arm_npz": EXP / "work/d-rider-lfpg-on-v19.npz",
        "arm_npz_sha": ("01620e927dd21c230a5896ccac7a1513"
                        "43aeb24bc9fcabe240cec96f2fd2c8c1"),
        # the head's occupancy is the v19 class vector, emitted against these same bytes.
        # The class only GROWS: LFPG's with-clock ranking rows become occupied and
        # written, and nothing the head occupied leaves.
        "head_class": {
            "column": "occupied_by_exp_046",
            "months": {
                "2026-01": ("artifacts/exp-046/class/2026-01-v19.parquet",
                            "42d97e38bb826d7b4a3bb84005966115"
                            "d44036c6edc0e7753d5ee70be83a83de"),
                "2026-07": ("artifacts/exp-046/class/2026-07-v19.parquet",
                            "2a76dbe3876d0463e4eb75ef5906f3ae"
                            "561b788f6f542f337478df01fd7c15d2")}},
        # **THE LIT RULE IS ASKED FOR HERE AND NOWHERE ELSE.**  The rider moves rows at
        # LFPG only, and LFPG january moves ZERO rows by instrument, so january is an
        # empty lit fold and `every_fold_interval_clear` is False by construction and can
        # never be True.  Section 26.5 is the registered ruling for exactly that cell, and
        # `increment_gate` checks every clause of it -- two lit folds, december among
        # them, every dark fold literally empty -- rather than taking this flag's word.
        "lit_rule": True,
        # the sibling assertion the desk required: v6 must differ from v5 ONLY at LFPG.
        # The serving fit is deterministic, so every other row must come back byte for
        # byte, and that is checked against the file rather than argued.
        "sibling": {"file": OUT_V5,
                    "sha": ("4d75706bfcc13d88f7556265cd008b3d"
                            "3a312555955dca8eb528e4db5325fee8"),
                    "may_differ_only_at_airports": ["LFPG"]},
        "unselected": True,
        "out": OUT_V6,
        "what": ("v18's bytes for every row this lane does not change, v12 as the "
                 "prediction the update is added to, and v18's own served vector as the "
                 "head. The candidate is v18 PLUS THE LFPG RIDER and nothing else: the "
                 "same fit with the airport hold lifted, so LFPG's with-clock rows stop "
                 "sitting at the price base and take this lane's value. Priced in "
                 "analysis/rider-read.json: +0.01940 [+0.00784, +0.03383] all-row, clean "
                 "+0.02369, july_loo +20.83 [+7.66, +35.17] on 552 rows and december "
                 "+51.91 [+28.93, +79.16] on 1,246, january dark by instrument. It is the "
                 "only increment on this lane's table that carries no refit"),
    },
    # ---------------------------------------------------------------- v22, the K = 10 bag
    # **The slot decision is by RULE, not by measurement.**  RESULTS section 43, committed
    # at 10:43 UTC on 2026-09-09 (d62c0a8) BEFORE the 511 read landed, quotes the three
    # rules: (a) section 13.1/15 "every fold clear stays the gate for an in-fold arm whose
    # form was not selected on the folds" -- K = 10's form was registered in section 40.2
    # before the read; (b) `arm_choice` "largest clean gain among arms with every fold
    # interval clear"; (c) the upload rule, against the last upload v18.  Both v7 and this
    # arm pass (a) and qualify under (c); (b) ranks this one higher on clean gain.  The
    # +0.0225 between them is BELOW the ~0.1 s upload floor, so no measurement separates
    # them -- and the K = 10 MINUS K = 5 increment is **NOT** clear on july (section 42.2:
    # +6.31 [-7.23, +19.80], gate FAILED, no v9 over v7).  Nothing here claims otherwise.
    "v22": {
        # v18's bytes and v18's own head H -- the same base and head v7 was served over.
        # v7 is a served, reviewed candidate that was never uploaded, as v6 was when v7
        # superseded it, so replacing it in the slot supersedes nothing off this laptop.
        "bytes": ROOT / "submissions/uploaded/youthful-giraffe_v18.parquet",
        "bytes_sha": ("4d75706bfcc13d88f7556265cd008b3d"
                          "3a312555955dca8eb528e4db5325fee8"),
        "price": BASE_FILE, "price_sha": BASE_SHA,
        "head_file": OUT_V5,
        "head_sha": ("4d75706bfcc13d88f7556265cd008b3d"
                          "3a312555955dca8eb528e4db5325fee8"),
        "head_arm": "c_lgb_shipped@leaves_255x300_lr050+guard995",
        "head_npz": EXP / "work/d-bundle-ungated1299-guard995.npz",
        "head_npz_sha": ("5dbbcb8c7d14831fd75debde479071cd"
                         "2a2eff9b7ff9b7469b6625ce00238684"),
        # **TEN ranking members, from the SAME ten seeds as the fold side** -- the same
        # 49-row draws, so the fold price describes the arm that ships.  Five recipes on
        # both sides was the K = 5 contract (section 37.3); at K = 10 it is ten on both
        # sides and any fewer is a different arm.  Registered in section 43.4 before this
        # ran.  `arm_npz` is the FOLD side only; the bag is a recipe, not a vector.
        "bag": {"K": 10, "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "leaves": 255,
                "cache": "work/d-rank-placebo{seed}-255.npz",
                "fold_members": {
                    "1": ("work/d-bundle-placebo1-255-free.npz",
                          "c951ad83b350897b818fb4214af39675"
                          "7baebec356f39a5b23170076ef2bbef0"),
                    "2": ("work/d-bundle-placebo2-255-free.npz",
                          "16ad546c70dfb28c34dd98d219d15782"
                          "5ea53fd9bdbe454876ca7782a774c626"),
                    "3": ("work/d-bundle-placebo3-255-free.npz",
                          "c94b66b4fe6c5a1ea5a2957bbbfa94f0"
                          "a6d51f28e73cfabcd7917ea8f3520ab7"),
                    "4": ("work/d-bundle-placebo4-255-free.npz",
                          "c97c34bbc625a8de88f292f85509f44a"
                          "122f40d6d2ca08ab0b4732d25ac2bd13"),
                    "5": ("work/d-bundle-placebo5-255-free.npz",
                          "31dd6a97c71d2294bcf6cf76153ea377"
                          "c9db8dbe71aaa517f3584d10e375958d"),
                    "6": ("work/d-bundle-placebo6-255-free.npz",
                          "5a8eba5756e33e4d95b3b09a10183e52"
                          "1939058fa85b49e76e78fb7660d0da48"),
                    "7": ("work/d-bundle-placebo7-255-free.npz",
                          "28cc1af0eeabb8ced85ddc98bb1e45e0"
                          "c6cc1436736e1937a2a03c8d7299e2c1"),
                    "8": ("work/d-bundle-placebo8-255-free.npz",
                          "22dc334b230ab437cd4fdb2515e02787"
                          "40e90f62f1afc77dc87ae9d3380aed1c"),
                    "9": ("work/d-bundle-placebo9-255-free.npz",
                          "273e5c60c8c578d86836be39fe939575"
                          "46deb9a1d226da44c28c38fca10bcce4"),
                    "10": ("work/d-bundle-placebo10-255-free.npz",
                          "49b3c8ac733cab7d21492b64c32087bd"
                          "c7f5760e071039ea557fa818add2d6d1")}},
        "head_npz_note": ("The arm this base set serves is the BAGGED 255 at K = 10 -- ten "
                          "fresh fits averaged -- so the increment over this head does "
                          "carry refit chaos; what makes it readable is that ten draws "
                          "divide that chaos, not that it is absent (RESULTS section 41.5: "
                          "B(K) = a + b/K with the b measured directly)"),
        "arm_npz": EXP / "work/d-bag10-255-free.npz",
        "arm_npz_sha": ("d32a993b48d18fa428cb6c0127de34a5"
                          "ccb0ce5ad09d2dd0a3e89db7a35cd26e"),
        "head_class": {
            "column": "occupied_by_exp_046",
            "months": {
                "2026-01": ("artifacts/exp-046/class/2026-01-v19.parquet",
                            "42d97e38bb826d7b4a3bb84005966115"
                            "d44036c6edc0e7753d5ee70be83a83de"),
                "2026-07": ("artifacts/exp-046/class/2026-07-v19.parquet",
                            "2a76dbe3876d0463e4eb75ef5906f3ae"
                            "561b788f6f542f337478df01fd7c15d2")}},
        "lit_rule": False,
        # the same sibling reading as v21: this arm replaces the head over the WHOLE class,
        # so "differs only at X" cannot be the claim -- it moves every with-clock airport,
        # LFPG included, and LFPG takes the bag's own value (this is bag-FREE at K = 10, so
        # LFPG is NOT held and NOT byte-identical to v7).  What must hold is that nothing
        # OUTSIDE the class moves: LTFM carries 47,581 ranking rows and not one with-clock
        # row, so LTFM must come back from v7 byte for byte.
        "sibling": {"file": OUT_V7,
                    "sha": ("128db7c58881959eaa15ba8990c48f08"
                          "3cbe018a195014f943f87de6b8e6b0fb"),
                    "may_differ_only_at_airports": [
                        "EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF",
                        "LSZH"],
                    "must_be_identical_at_airports": ["LTFM"]},
        # the bag is the unweighted mean of ALL TEN members; no member was fitted, read and
        # then kept or dropped, so there is no selection premium to charge
        "unselected": True,
        "out": OUT_V8,
        "what": ("v18's bytes for every row this lane does not change, v12 as the "
                 "prediction the update is added to, and v18's own served vector H as the "
                 "head. The candidate REPLACES that head with the BAGGED 255 at K = 10, "
                 "LFPG included: the unweighted mean of ten fits of the shipped recipe on "
                 "ten distinct 49-row class perturbations, restricted to the shipped class, "
                 "floored and guarded at the shipped panel's own cut. Priced at +0.24275 [+0.18095, +0.32601] all-row, clean +0.29645, july_loo +113.65 [+79.71, +146.61], december +256.66 [+210.10, +309.89], january +184.84 [+146.22, +229.58], every fold clear. "
                 "It supersedes v7 by RULE (b), not by measurement -- the +0.0225 between "
                 "them is below the upload floor and their increment fails its gate on "
                 "july (RESULTS sections 42.2, 43.2)."),
    },
    # v23: THE EXTENDED HEAD.  v22 with one change and one only -- every fit on BOTH sides
    # carries 2025-11's 99,709 writable rows in its fitting set on top of the fold rows.
    # The fold price is RESULTS section 47.20's K = 10 extended bag over v8's head; the
    # ranking side must be fitted the same way or the fold price describes an arm that does
    # not ship.  The extension's targets are base6 residuals and its `_n_res_` features come
    # from the same base6 (desk ruling 3ab2ef0: one base per row, base out-of-sample for the
    # row and in-sample for the scored fold, one rule uniformly).
    "v24": {
        # v10 = v9's base set EXACTLY, with the three-month head on both sides.  Same bytes,
        # same price base, same head H: the only difference from v9 is how many months the
        # head's fitting set carries.  v9 is the sibling this one must be readable against.
        "bytes": ROOT / "submissions/uploaded/youthful-giraffe_v18.parquet",
        "bytes_sha": ("4d75706bfcc13d88f7556265cd008b3d"
                          "3a312555955dca8eb528e4db5325fee8"),
        "price": BASE_FILE, "price_sha": BASE_SHA,
        "head_file": OUT_V5,
        "head_sha": ("4d75706bfcc13d88f7556265cd008b3d"
                          "3a312555955dca8eb528e4db5325fee8"),
        "head_arm": "c_lgb_shipped@leaves_255x300_lr050+guard995",
        "head_npz": EXP / "work/d-bundle-ungated1299-guard995.npz",
        "head_npz_sha": ("5dbbcb8c7d14831fd75debde479071cd"
                         "2a2eff9b7ff9b7469b6625ce00238684"),
        # **THE EXTENSION, now a STACK.**  Same one builder for both sides; each month's
        # base6 and neighbour block are checked by sha inside it before a row is used, and
        # each month's row count is the one section 47.32 measured.
        "extension": {
            "months": ["2025-11", "2025-10", "2025-03"], "rows": 290037,
            "month_rows": {"2025-11": 99709, "2025-10": 103344, "2025-03": 86984},
            "shas": {"2025-11": ("339ca16ed421a718", "3e231a8f7cd82d83"),
                     "2025-10": ("2bfc9a6959281d6e", "29932ab196080943"),
                     "2025-03": ("64c47603ebfdb0c7", "ec26128f3a5eaa74")},
            "what": ("the writable rows of all three months appended to the fitting set of "
                     "EVERY fit on both sides, after the shipped join. Fold-side multiples "
                     "2.679 / 2.821 / 2.514; the ranking side pools all three folds, so its "
                     "multiple is smaller and is recorded from the run"),
            "why_not_a_selection": ("all three months are section 47.12's registered order "
                                    "-- 'use the months that exist' -- and none was chosen "
                                    "on a fold reading. 2025-03 in particular was queued "
                                    "under the calendar order BEFORE the cyclic rule "
                                    "existed, which is what makes section 47.32 an "
                                    "out-of-sample test of that rule rather than a "
                                    "confirmation of it. Rule (a) applies (section 47.33.1); "
                                    "section 47.34 registers that the NEXT panel will not")},
        # a DISTINCT cache name again: a ranking member fitted on one month is a different
        # arm from one fitted on three, and the v23 key cannot tell them apart (section 45.2)
        "bag": {"K": 10, "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "leaves": 255,
                "cache": "work/d-rank-ext3-placebo{seed}-255.npz",
                "fold_members": {
                    "1": ("work/d-onemonth-twomonth-plus03-255-free.npz",
                          "5557544a1874b078625d1151f4314681"
                          "decdbc145bf4abef4b23f84b31bdbe65"),
                    "2": ("work/d-threemonth-m2-255-free.npz",
                          "7fd259a7413d906dc3175d1f9c9114d0"
                          "935d1a914f9a5d4672bd390857542a5f"),
                    "3": ("work/d-threemonth-m3-255-free.npz",
                          "4652793ec661b60243d5a4049e30d712"
                          "a8f62c904b51333ca7b485578025f4e5"),
                    "4": ("work/d-threemonth-m4-255-free.npz",
                          "aae0b0b155f1f57efb20855d9c8b058b"
                          "513a2d97ead0d8ce834813ef00fdc576"),
                    "5": ("work/d-threemonth-m5-255-free.npz",
                          "745ba8a14157721b9ff05ce939462a72"
                          "d3fcf506cf604e03bd907bc46b210ed9"),
                    "6": ("work/d-threemonth-m6-255-free.npz",
                          "f88a0dd53dc89448956263b2d8a3a036"
                          "27ccde84bd0eafa2e160454e1b2cbaae"),
                    "7": ("work/d-threemonth-m7-255-free.npz",
                          "a5668717de2882fd964fc4c6a629f497"
                          "5bbe0d41bbdad43d8e7131ec04b51be7"),
                    "8": ("work/d-threemonth-m8-255-free.npz",
                          "6a97113b0589caeb1a22a178e987359c"
                          "07b16e49574c50e0ef011ed18b0ad685"),
                    "9": ("work/d-threemonth-m9-255-free.npz",
                          "c574a987d6647421d25e8182fe957679"
                          "2135450bdd7eed7d9fb18e9c0e5473e9"),
                    "10": ("work/d-threemonth-m10-255-free.npz",
                          "fda814090405a04904604df252b4e811"
                          "b82d134d3738f5c5ce1d79bf03251c67")}},
        "arm_npz": EXP / "work/d-threemonth-bag10-255-free.npz",
        "arm_npz_sha": ("1e0d0085b6ecea726e8da603554bfff9"
                          "5eef85e1afc3bc17a703de59d7b627c6"),
        # the head is v18's H, unchanged from v23, so its class vector is unchanged too
        "head_class": {
            "column": "occupied_by_exp_046",
            "months": {
                "2026-01": ("artifacts/exp-046/class/2026-01-v19.parquet",
                            "42d97e38bb826d7b4a3bb84005966115"
                            "d44036c6edc0e7753d5ee70be83a83de"),
                "2026-07": ("artifacts/exp-046/class/2026-07-v19.parquet",
                            "2a76dbe3876d0463e4eb75ef5906f3ae"
                            "561b788f6f542f337478df01fd7c15d2")}},
        "lit_rule": False,
        # SECTION 47.22.4's defect, made impossible rather than remembered: v9's serve ran
        # --ungated and its class vector was first emitted WITHOUT it, giving a class short
        # by exactly the 2,456 mode-4 rows. The base set now DECLARES the setting and both
        # the serve and the class emission are checked against this field.
        "ungated": True,
        # ...and the ARM, for the same reason. v10 was first served with plain
        # "c_lgb_shipped" while this base set's head is the 255x300/lr050 fit under the
        # guard: the candidate came out with tail_guard null, a different composition from
        # v9. Nothing connected the CLI string to the base set. Now something does.
        "arm": "c_lgb_shipped@leaves_255x300_lr050+guard995",
        # the sibling is v9: same head, same class, more fitting months -- so every
        # with-clock airport moves and nothing outside the class may
        "sibling": {"file": OUT_V9,
                    "sha": ("8fd4b34166b61756a20746c77f97fc1d"
                          "06d06af28ee959ef7deac1de0a365d78"),
                    "may_differ_only_at_airports": [
                        "EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF",
                        "LSZH"],
                    "must_be_identical_at_airports": ["LTFM"]},
        "unselected": True,
        "out": OUT_V10,
        "what": ("v18's bytes for every row this lane does not change, v12 as the prediction "
                 "the update is added to, and v18's own served vector H as the head -- v23's "
                 "base set exactly. The candidate replaces that head with the K = 10 bag "
                 "fitted on the fold rows PLUS the 290,037 writable rows of 2025-11, 2025-10 "
                 "and 2025-03, on both the fold and the ranking side. Priced over V9'S HEAD; "
                 "the fold-side read and its intervals are filled from analysis/"
                 "threemonth-bag10.json when the bag lands. That figure passes the gate the "
                 "desk set; it is NOT the figure that decides the upload, which is this "
                 "serve's own increment over v18's head."),
    },
    "v25": {
        # SECTION 47.49 / recorded rule: the SEASONAL HEAD.  v10's base set exactly -- same bytes, same
        # price base, same head H -- with ONE change, and it is a change of SHAPE rather than
        # of degree: the candidate's head is composed BY SERVED MONTH.
        #
        #   2026-01 rows  <-  the January head, fitted on the folds + {Nov, Oct, Mar, FEB}
        #   2026-07 rows  <-  the July head,    fitted on the folds + {Nov, Oct, Mar, AUG}
        #
        # WHY, in one line, because a base set that does not say why is a base set nobody can
        # audit: the served months are ANTIPODAL (d(m,2026-01) + d(m,2026-07) = 6 for every
        # month), so no single fourth month helps both, and section 47.45.3 predicted -- before
        # either month existed -- that every all-served four-month panel fails a fold.  The
        # month per served month is the nearest BUILT month by served distance, Feb at d=1 and
        # Aug at d=1, both unique: parameter-free under monotonicity, chosen by no fold's
        # reading, so rule (a) applies and the premium is 0.
        "bytes": ROOT / "submissions/uploaded/youthful-giraffe_v18.parquet",
        "bytes_sha": ("4d75706bfcc13d88f7556265cd008b3d"
                          "3a312555955dca8eb528e4db5325fee8"),
        "price": BASE_FILE, "price_sha": BASE_SHA,
        "head_file": OUT_V5,
        "head_sha": ("4d75706bfcc13d88f7556265cd008b3d"
                          "3a312555955dca8eb528e4db5325fee8"),
        "head_arm": "c_lgb_shipped@leaves_255x300_lr050+guard995",
        "head_npz": EXP / "work/d-bundle-ungated1299-guard995.npz",
        "head_npz_sha": ("5dbbcb8c7d14831fd75debde479071cd"
                         "2a2eff9b7ff9b7469b6625ce00238684"),
        # **TWO extensions, one per served month.**  Each goes through `extension_rows`, the
        # same builder both sides use, so each month's base6 and neighbour block are checked
        # by sha before a row of it is fitted.  A served month missing from this map is FATAL:
        # its rows would be written by a head fitted on the OTHER served month's months.
        "extension_by_served_month": {
            "2026-01": {
                "months": ["2025-11", "2025-10", "2025-03", "2025-02"], "rows": 358159,
                "month_rows": {"2025-11": 99709, "2025-10": 103344, "2025-03": 86984,
                               "2025-02": 68122},
                "shas": {"2025-11": ("339ca16ed421a718", "3e231a8f7cd82d83"),
                         "2025-10": ("2bfc9a6959281d6e", "29932ab196080943"),
                         "2025-03": ("64c47603ebfdb0c7", "ec26128f3a5eaa74"),
                         "2025-02": ("36f7dd9f6d80451a", "32c9e25d05f6ad21")},
                "what": "the January head's fitting rows; 2025-02 is at served distance 1"},
            "2026-07": {
                "months": ["2025-11", "2025-10", "2025-03", "2025-08"], "rows": 386574,
                "month_rows": {"2025-11": 99709, "2025-10": 103344, "2025-03": 86984,
                               "2025-08": 96537},
                "shas": {"2025-11": ("339ca16ed421a718", "3e231a8f7cd82d83"),
                         "2025-10": ("2bfc9a6959281d6e", "29932ab196080943"),
                         "2025-03": ("64c47603ebfdb0c7", "ec26128f3a5eaa74"),
                         "2025-08": ("60c84140c3342390", "137a61f23de48d9d")},
                "what": "the July head's fitting rows; 2025-08 is at served distance 1. Queue "
                        "landed it 2026-09-10 (5859d1b) and BOTH shas were checked against the "
                        "files on disk before they were written here (section 47.51); "
                        "`month_rows` gains 2025-08 when a run HARVESTS its writable count, "
                        "and `arm_npz_sha` stays PENDING until the seasonal bag exists"},
        },
        # a distinct cache key PER SERVED MONTH.  The key is seed+leaves and cannot tell one
        # arm from another (section 45.2), and here there are TWO arms in one serve, so a
        # shared key would silently give both served months the same head -- which is exactly
        # the thing this base set exists to stop.
        "bag_by_served_month": {
            "2026-01": {"K": 10, "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "leaves": 255,
                        "cache": "work/d-rank-seasjan-placebo{seed}-255.npz"},
            "2026-07": {"K": 10, "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "leaves": 255,
                        "cache": "work/d-rank-seasjul-placebo{seed}-255.npz"},
        },
        # the fold side is ONE bag: each leave-one-out fit carries its own fold's month set
        # (january and december the Feb set, july the Aug set), so no fold is fitted twice.
        "arm_npz": EXP / "work/d-seasonal-bag10-255-free.npz",
        "arm_npz_sha": ("3edfa35fee55e1d03ec2df72192eeb59"
                        "6732e62f8164b162e9ef42597ecea6f5"),
        "head_class": {
            "column": "occupied_by_exp_046",
            "months": {
                "2026-01": ("artifacts/exp-046/class/2026-01-v19.parquet",
                            "42d97e38bb826d7b4a3bb84005966115"
                            "d44036c6edc0e7753d5ee70be83a83de"),
                "2026-07": ("artifacts/exp-046/class/2026-07-v19.parquet",
                            "2a76dbe3876d0463e4eb75ef5906f3ae"
                            "561b788f6f542f337478df01fd7c15d2")}},
        "lit_rule": False,
        "ungated": True,
        "arm": "c_lgb_shipped@leaves_255x300_lr050+guard995",
        # the sibling is v10: same bytes, same class, same guard -- the head is composed by
        # served month instead of one head everywhere, so every with-clock airport may move
        # and nothing outside the class may
        "sibling": {"file": OUT_V10,
                    "sha": ("7b0557a497f8c09e4f56a11352783b9d"
                          "da37fe1d0579345d4303e4f74d6e7aca"),
                    "may_differ_only_at_airports": [
                        "EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF",
                        "LSZH"],
                    "must_be_identical_at_airports": ["LTFM"]},
        "unselected": True,
        "out": OUT_V11,
        "what": ("v10's base set with the head composed BY SERVED MONTH: 2026-01 rows from a "
                 "head fitted on the folds + {2025-11, 2025-10, 2025-03, 2025-02}, 2026-07 "
                 "rows from a head fitted on the folds + {2025-11, 2025-10, 2025-03, "
                 "2025-08}. Priced over V10'S HEAD. The fold side is one K = 10 bag whose "
                 "every leave-one-out fit carries its own fold's month set."),
    },
    "v25plan": {
        # SECTION 47.49 (c): v25's MECHANICS, exercisable before 2025-08 exists.  It is
        # identical to v25 except that the July head's fourth month is ABSENT -- its set is
        # the three-month one -- so the two served months still get DIFFERENT month sets,
        # different row counts and different cache keys, and nothing here pretends a month
        # exists that does not.  `plan_only` makes it refuse to serve: it may be run with
        # --plan and in no other way, so a mechanics rehearsal can never become a candidate.
        "plan_only": True,
        "bytes": ROOT / "submissions/uploaded/youthful-giraffe_v18.parquet",
        "bytes_sha": ("4d75706bfcc13d88f7556265cd008b3d"
                          "3a312555955dca8eb528e4db5325fee8"),
        "price": BASE_FILE, "price_sha": BASE_SHA,
        "head_file": OUT_V5,
        "head_sha": ("4d75706bfcc13d88f7556265cd008b3d"
                          "3a312555955dca8eb528e4db5325fee8"),
        "head_arm": "c_lgb_shipped@leaves_255x300_lr050+guard995",
        "head_npz": EXP / "work/d-bundle-ungated1299-guard995.npz",
        "head_npz_sha": ("5dbbcb8c7d14831fd75debde479071cd"
                         "2a2eff9b7ff9b7469b6625ce00238684"),
        "extension_by_served_month": {
            "2026-01": {
                "months": ["2025-11", "2025-10", "2025-03", "2025-02"], "rows": 358159,
                "month_rows": {"2025-11": 99709, "2025-10": 103344, "2025-03": 86984,
                               "2025-02": 68122},
                "shas": {"2025-11": ("339ca16ed421a718", "3e231a8f7cd82d83"),
                         "2025-10": ("2bfc9a6959281d6e", "29932ab196080943"),
                         "2025-03": ("64c47603ebfdb0c7", "ec26128f3a5eaa74"),
                         "2025-02": ("36f7dd9f6d80451a", "32c9e25d05f6ad21")},
                "what": "the January head's fitting rows, exactly as v25 declares them"},
            "2026-07": {
                "months": ["2025-11", "2025-10", "2025-03"], "rows": 290037,
                "month_rows": {"2025-11": 99709, "2025-10": 103344, "2025-03": 86984},
                "shas": {"2025-11": ("339ca16ed421a718", "3e231a8f7cd82d83"),
                         "2025-10": ("2bfc9a6959281d6e", "29932ab196080943"),
                         "2025-03": ("64c47603ebfdb0c7", "ec26128f3a5eaa74")},
                "what": "PLACEHOLDER: the three-month set standing where 2025-08's set will "
                        "go, so the two served months differ without inventing a month"},
        },
        "bag_by_served_month": {
            "2026-01": {"K": 10, "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "leaves": 255,
                        "cache": "work/d-rank-seasjan-placebo{seed}-255.npz"},
            "2026-07": {"K": 10, "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "leaves": 255,
                        "cache": "work/d-rank-seasjul-placebo{seed}-255.npz"},
        },
        "arm_npz": EXP / "work/d-threemonth-bag10-255-free.npz",
        "arm_npz_sha": ("1e0d0085b6ecea726e8da603554bfff9"
                          "5eef85e1afc3bc17a703de59d7b627c6"),
        "head_class": {
            "column": "occupied_by_exp_046",
            "months": {
                "2026-01": ("artifacts/exp-046/class/2026-01-v19.parquet",
                            "42d97e38bb826d7b4a3bb84005966115"
                            "d44036c6edc0e7753d5ee70be83a83de"),
                "2026-07": ("artifacts/exp-046/class/2026-07-v19.parquet",
                            "2a76dbe3876d0463e4eb75ef5906f3ae"
                            "561b788f6f542f337478df01fd7c15d2")}},
        "lit_rule": False, "ungated": True,
        "arm": "c_lgb_shipped@leaves_255x300_lr050+guard995",
        "sibling": {"file": OUT_V10,
                    "sha": ("7b0557a497f8c09e4f56a11352783b9d"
                          "da37fe1d0579345d4303e4f74d6e7aca"),
                    "may_differ_only_at_airports": [
                        "EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF",
                        "LSZH"],
                    "must_be_identical_at_airports": ["LTFM"]},
        "unselected": True,
        # the dry-run path, never a candidate path: `DRY` is defined below this table, so
        # it is written out here rather than referenced
        "out": EXP / "work/dry-run.parquet",
        "what": ("PLAN ONLY. v25's mechanics with the three-month set standing in for the "
                 "July head's, so the composition census, the two cache keys and the "
                 "refusals can be exercised before 2025-08 exists. It cannot serve."),
    },
    "v23": {
        # v18's bytes and v18's own head H again -- the same base and head v8 was served
        # over, so the only difference between v8 and this candidate is the head's fitting
        # rows.  v8 is a served, reviewed candidate; it is NOT superseded here, it is the
        # sibling this one must be readable against.
        "bytes": ROOT / "submissions/uploaded/youthful-giraffe_v18.parquet",
        "bytes_sha": ("4d75706bfcc13d88f7556265cd008b3d"
                          "3a312555955dca8eb528e4db5325fee8"),
        "price": BASE_FILE, "price_sha": BASE_SHA,
        "head_file": OUT_V5,
        "head_sha": ("4d75706bfcc13d88f7556265cd008b3d"
                          "3a312555955dca8eb528e4db5325fee8"),
        "head_arm": "c_lgb_shipped@leaves_255x300_lr050+guard995",
        "head_npz": EXP / "work/d-bundle-ungated1299-guard995.npz",
        "head_npz_sha": ("5dbbcb8c7d14831fd75debde479071cd"
                         "2a2eff9b7ff9b7469b6625ce00238684"),
        # **THE EXTENSION.**  One builder for both sides (`serve.extension_rows` calls the
        # same `code/onemonth.py` the fold members were fitted through), and the row count
        # is the one section 47.20 registered its prediction against.  The two shas are
        # checked inside that builder before a row is used.
        "extension": {
            "month": "2025-11", "rows": 99709,
            "base6_sha16": "339ca16ed421a718",
            "neighbour_block_sha16": "3e231a8f7cd82d83",
            "what": ("2025-11's writable rows appended to the fitting set of EVERY fit on "
                     "both sides, after the shipped join. Multiples on the fold side "
                     "1.577 / 1.626 / 1.521; the ranking side pools all three folds, so "
                     "its multiple is smaller and is recorded from the run"),
            "why_not_a_selection": ("the month was fixed before any of its data was read "
                                    "(section 47.17, commit a89252a) and no airport, day "
                                    "or row of it was chosen on a reading")},
        # TEN ranking members on the same ten seeds as the fold side, as v22 -- and on a
        # DISTINCT cache name, because a member fitted without the extension is a different
        # arm and the v22 cache key (seed and leaves only) cannot tell them apart.  That is
        # the exact failure of 2026-09-09 recorded in section 45.2.
        "bag": {"K": 10, "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "leaves": 255,
                "cache": "work/d-rank-ext-placebo{seed}-255.npz",
                "fold_members": {
                    "1": ("work/d-onemonth-screen-255-free.npz",
                          "3228d8f6097f5ed93767f7bf351e263c"
                          "f19d22e11b54363dddde12ca186fc2ea"),
                    "2": ("work/d-onemonth-m2-255-free.npz",
                          "ed2245f0e304c3443dc92924921c34a6"
                          "2b828e2deb27c246fec41c8662bf4e7a"),
                    "3": ("work/d-onemonth-m3-255-free.npz",
                          "938bdcccebbcaac9694cabedc9e27c29"
                          "1e0ca74578025b5417a50fc988e163c4"),
                    "4": ("work/d-onemonth-m4-255-free.npz",
                          "839d177c2eb6332a8beb1921f220078d"
                          "2030257ec2d980292ddd00e1e9cf5ce2"),
                    "5": ("work/d-onemonth-m5-255-free.npz",
                          "7be2e6d234340c52abcf7c93f64542ac"
                          "0376fec8025a5930f6b4cd8ff16f311a"),
                    "6": ("work/d-onemonth-m6-255-free.npz",
                          "f3dfc876b931a5efabf4ed6fe7eb5572"
                          "6193408e575c15f433faa3d7f6a54038"),
                    "7": ("work/d-onemonth-m7-255-free.npz",
                          "f557891f16f6e40c40a790ee262a5db2"
                          "15b4d5c56bfe92746ef7486a61d8139d"),
                    "8": ("work/d-onemonth-m8-255-free.npz",
                          "2a3d90b5b4a0ecba82e8be7e27751314"
                          "1712c42babf6b19ec2f9ee83e3941d36"),
                    "9": ("work/d-onemonth-m9-255-free.npz",
                          "0fc43c96279b8e947c31b783e9c3f4f6"
                          "6d4f89295b1196b053999010b65b0651"),
                    "10": ("work/d-onemonth-m10-255-free.npz",
                          "f17615ce8996d3cdeb76a2bb880c4ac5"
                          "7847408f308fbcf7b1b186dbf35f4d52")}},
        "arm_npz": EXP / "work/d-onemonth-bag10-255-free.npz",
        "arm_npz_sha": ("e1af652c0e5b07a8786532005e185412"
                          "9470138a07f0999c38d002d6559a871e"),
        # the head is v18's H, unchanged from v22, so its class vector is unchanged too
        "head_class": {
            "column": "occupied_by_exp_046",
            "months": {
                "2026-01": ("artifacts/exp-046/class/2026-01-v19.parquet",
                            "42d97e38bb826d7b4a3bb84005966115"
                            "d44036c6edc0e7753d5ee70be83a83de"),
                "2026-07": ("artifacts/exp-046/class/2026-07-v19.parquet",
                            "2a76dbe3876d0463e4eb75ef5906f3ae"
                            "561b788f6f542f337478df01fd7c15d2")}},
        "lit_rule": False,
        # the sibling is v8, not v7: this candidate replaces the SAME head over the SAME
        # class with a head fitted on more rows, so every with-clock airport moves and
        # nothing outside the class may.
        "sibling": {"file": OUT_V8,
                    "sha": ("d03d132a5ff5d3fc70a72b9f60a1b418"
                          "a169e911ac32ef08b14378695932aa14"),
                    "may_differ_only_at_airports": [
                        "EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF",
                        "LSZH"],
                    "must_be_identical_at_airports": ["LTFM"]},
        # ten members, all ten kept: no member was fitted, read and then dropped
        "unselected": True,
        "out": OUT_V9,
        "what": ("v18's bytes for every row this lane does not change, v12 as the "
                 "prediction the update is added to, and v18's own served vector H as the "
                 "head -- v22's base set exactly. The candidate replaces that head with "
                 "the K = 10 bag fitted on the fold rows PLUS 2025-11's 99,709 writable "
                 "rows, on both the fold and the ranking side. Priced over V8'S HEAD at "
                 "+0.51347 [+0.39218, +0.68821] all-row, clean +0.62745, july_loo +290.27 "
                 "[+228.74, +357.17], december +919.98 [+850.94, +993.28], january +327.36 "
                 "[+253.43, +410.90], every fold clear, cross-fit identical to every digit "
                 "and selection premium 0.0 (RESULTS section 47.21). That figure passes the "
                 "gate the desk set; it is NOT the figure that decides the upload, which is "
                 "this serve's own increment over v18's head -- the quantity v8 was "
                 "measured on at +0.24275."),
    },
    "v21": {
        # **v18's BYTES**, the file on the board at reading 17 (246.3605), which are v5's
        # byte for byte.  v6 would give the SAME output file -- it differs from v18 only at
        # LFPG and every one of those rows is served here -- so the bytes are chosen to put
        # the gate record against the board (desk, 2026-09-09).
        "bytes": ROOT / "submissions/uploaded/youthful-giraffe_v18.parquet",
        "bytes_sha": ("4d75706bfcc13d88f7556265cd008b3d"
                      "3a312555955dca8eb528e4db5325fee8"),
        "price": BASE_FILE, "price_sha": BASE_SHA,
        # the head is v18's file and v18's own fold-side vector, H
        "head_file": OUT_V5,
        "head_sha": ("4d75706bfcc13d88f7556265cd008b3d"
                     "3a312555955dca8eb528e4db5325fee8"),
        "head_arm": "c_lgb_shipped@leaves_255x300_lr050+guard995",
        "head_npz": EXP / "work/d-bundle-ungated1299-guard995.npz",
        "head_npz_sha": ("5dbbcb8c7d14831fd75debde479071cd"
                         "2a2eff9b7ff9b7469b6625ce00238684"),
        # **THE BAG REPLACES THE HEAD; IT DOES NOT ADD TO IT.**  Every member is an update
        # over the PRICE BASE, exactly as H is, so `bagarm.py` prices the bag as `bag - H`.
        # A serve whose arm vector were `bag_held` alone would write v12 at LFPG and
        # silently REVERT v6's rider.  `code/bagcompose.py` composes the vector this base
        # set means -- the bag off LFPG, the rider's own values at LFPG -- and asserts
        # three things to the bit before writing it: the rider put back under the hold IS
        # the head (0 rows differing on all three folds), the two supports are disjoint
        # (0 rows in both), and `arm - rider == bag_held - head` exactly.
        # **THE BAG IS A RECIPE, NOT A VECTOR.**  `arm_npz` is the FOLD side only, so the
        # ranking side has to run the same recipe: five ranking fits on the same five
        # perturbed fit-panels, averaged.  Without this the serve would price the bag on
        # the folds and write the ordinary single fit -- which is exactly v6, and the
        # v21 dry run caught it at 0 rows differing from the sibling.
        "bag": {"K": 5, "seeds": [1, 2, 3, 4, 5], "leaves": 255,
                "cache": "work/d-rank-placebo{seed}-255.npz",
                # the FOLD members these ranking members must match, by sha, so "the same
                # five recipes on both sides" is checked and not asserted
                "fold_members": {
                    "1": ("work/d-bundle-placebo1-255-free.npz",
                          "c951ad83b350897b818fb4214af39675"
                          "7baebec356f39a5b23170076ef2bbef0"),
                    "2": ("work/d-bundle-placebo2-255-free.npz",
                          "16ad546c70dfb28c34dd98d219d15782"
                          "5ea53fd9bdbe454876ca7782a774c626"),
                    "3": ("work/d-bundle-placebo3-255-free.npz",
                          "c94b66b4fe6c5a1ea5a2957bbbfa94f0"
                          "a6d51f28e73cfabcd7917ea8f3520ab7"),
                    "4": ("work/d-bundle-placebo4-255-free.npz",
                          "c97c34bbc625a8de88f292f85509f44a"
                          "122f40d6d2ca08ab0b4732d25ac2bd13"),
                    "5": ("work/d-bundle-placebo5-255-free.npz",
                          "31dd6a97c71d2294bcf6cf76153ea377"
                          "c9db8dbe71aaa517f3584d10e375958d")}},
        # the opposite of v20's note, and the reason N1 was a finding: this arm is five
        # FRESH fits, so the increment does carry refit chaos -- what makes it readable is
        # that the chaos is averaged down, not absent (round 9, N1)
        "head_npz_note": ("The arm this base set serves is the BAGGED 255 -- five fresh "
                          "fits averaged -- so the increment over this head does carry "
                          "refit chaos; what makes it readable is that five draws divide "
                          "that chaos, not that it is absent (RESULTS section 37.2)"),
        # **bag-FREE**: the bag with the LFPG hold lifted, so LFPG carries the bag's own
        # value.  NOT `d-bag5-255-rider.npz` (bag off LFPG, rider at LFPG) -- see the
        # correction in RESULTS §36: "bag-free = the bag composed with the rider" was a
        # mislabel of mine.  bag-free is the better vector and the one this base set's
        # figures belong to: december reads +226.08 against bag-held's +176.32, and that
        # difference IS LFPG, so the bagged LFPG beats the rider's +51.91 there.
        "arm_npz": EXP / "work/d-bag5-255-free.npz",
        "arm_npz_sha": ("c6d1f5b6f366c7f885c3f3813ae37e46"
                        "b28997ee8f0d919ba61fa55000c854ac"),
        # **THE SIBLING VECTOR, and why this base set is not it.**  `d-bag5-255-rider.npz`
        # (`code/bagcompose.py`) is the bag off LFPG with the rider at LFPG -- the vector
        # that would leave v6's LFPG bytes untouched.  It is a SIBLING of this one, not a
        # rival: it prices at about +0.21877 over v18's bytes against bag-free's +0.22025,
        # and 0.0015 s sits far below the refit floor section 34.4 measured (~0.06 s), so
        # no measurement separates them.  bag-free is chosen by REGISTRATION AND PRINCIPLE
        # (desk, 2026-09-09): the registered read, every fold interval and queue's primary
        # expectation are bag-free's, and the bag replaces the head at LFPG exactly as it
        # does everywhere else.  RESULTS section 37.5.
        "head_class": {
            "column": "occupied_by_exp_046",
            "months": {
                "2026-01": ("artifacts/exp-046/class/2026-01-v19.parquet",
                            "42d97e38bb826d7b4a3bb84005966115"
                            "d44036c6edc0e7753d5ee70be83a83de"),
                "2026-07": ("artifacts/exp-046/class/2026-07-v19.parquet",
                            "2a76dbe3876d0463e4eb75ef5906f3ae"
                            "561b788f6f542f337478df01fd7c15d2")}},
        # every fold moves rows here, so the registered every-fold-clear gate applies as
        # written and the lit rule is NOT asked for
        "lit_rule": False,
        # **The sibling assertion, in the only reading that still does work here.**  v21
        # replaces the head over the WHOLE class, so "differs only at X" cannot be the
        # claim: v7 moves every with-clock airport, LFPG included.  What must be true is
        # that it moves nothing OUTSIDE the class -- and LTFM carries 47,581 ranking rows
        # and not one with-clock row, so LTFM must come back from v6 byte for byte.  That
        # is a real check on the serve mask, and it fails loudly if the class ever leaks.
        "sibling": {"file": OUT_V6,
                    "sha": ("54dec3df14a9aaea6938b05c0b8e4280"
                            "a958bce32f734b293ecb2deb32eb41fc"),
                    "may_differ_only_at_airports": [
                        "EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF",
                        "LSZH"],
                    "must_be_identical_at_airports": ["LTFM"]},
        # the bag is the unweighted mean of ALL FIVE members; no member was fitted, read
        # and then kept or dropped, so there is no selection premium to charge
        "unselected": True,
        "out": OUT_V7,
        "what": ("v18's bytes for every row this lane does not change, v12 as the "
                 "prediction the update is added to, and v18's own served vector H as the "
                 "head. The candidate REPLACES that head with the BAGGED 255, LFPG "
                 "included: the unweighted mean of five fits of the shipped recipe on five "
                 "distinct 49-row class perturbations, restricted to the shipped class, "
                 "floored and guarded at the shipped panel's own cut. Priced in "
                 "analysis/bag-arm.json: +0.22025 [+0.16206, +0.29728] all-row, clean "
                 "+0.26896, july_loo +107.35 [+73.92, +138.41], december +226.08 "
                 "[+175.43, +285.89], january +162.39 [+124.29, +207.63], every fold clear "
                 "and every fold far outside the placebo envelope. It supersedes v6 rather "
                 "than composing with it -- v6's LFPG rider is replaced by the bag's own "
                 "LFPG values, which read better (december +226.08 against the held bag's "
                 "+176.32, and that difference is LFPG). It carries no new information: it "
                 "removes the chaos energy H, being one draw, was carrying (section 37.2)"),
    },
}


# ---------------------------------------------------------------------------------------
# SECTION 47.58.  BASE SET v26 -- v25 with the FIVE-month July head.
#
# It is DERIVED from v25 rather than typed beside it, and the derivation asserts exactly
# which paths differ.  Four defects in this lane (47.43.4, 47.55) were a field describing
# one run inherited by another; a base set is the largest such field there is, and a
# hand-copied 60-line dict is the same defect waiting on a bigger surface.  What is written
# below is the DIFFERENCE from v25, and everything unnamed is inherited on purpose.
#
# The January head does NOT change: June is at served distance 5 from 2026-01, so it enters
# no January set (RESULTS 47.57).  Its ranking cache key is therefore left alone ON PURPOSE
# -- the same arm on the same month set is the same head, so v25's ten January members are
# exactly what a refit would produce and reusing them is correct.  July's key is new,
# because July's month set changed and 45.2's collision is precisely a changed arm under an
# unchanged key.
# ---------------------------------------------------------------------------------------
def _derive_v26() -> dict:
    v = copy.deepcopy(BASE_SETS["v25"])
    jul = v["extension_by_served_month"]["2026-07"]
    jul["months"] = ["2025-11", "2025-10", "2025-03", "2025-08", "2025-06"]
    jul["rows"] = None                     # June's writable count is HARVESTED, not predicted
    jul["shas"]["2025-06"] = ("0bff146d3bced007", "5e7e6e10898d2445")
    jul["what"] = ("the July head's fitting rows under 47.57's ruling: every built month at "
                   "the MINIMUM served distance, ties included, so 2025-08 AND 2025-06, both "
                   "at d = 1. Both shas were recomputed from the files on disk before they "
                   "were written here. `rows` and 2025-06's `month_rows` entry are filled "
                   "when a run harvests the writable count")
    v["bag_by_served_month"]["2026-07"]["cache"] = "work/d-rank-seas5jul-placebo{seed}-255.npz"
    v["arm_npz"] = EXP / "work/d-seasonal5-bag10-255-free.npz"
    v["arm_npz_sha"] = "PENDING"
    v["sibling"] = {"file": OUT_V11,
                    "sha": ("f11d34f88b62a43ec2a0cdbeec967a9e"
                            "e4b7d92e50a5508aa4d30e5a78ea5e1c"),
                    "may_differ_only_at_airports": v["sibling"]["may_differ_only_at_airports"],
                    "must_be_identical_at_airports": v["sibling"]["must_be_identical_at_airports"]}
    v["out"] = OUT_V12
    v["what"] = ("v25's base set with ONE change: the July head's month set gains 2025-06, "
                 "the second built month at served distance 1 from 2026-07 (47.57's ruling). "
                 "The January head, december's reading and every 2026-01 row are unchanged, "
                 "so this candidate may differ from v11 only where the July head writes.")
    # the positive-and-negative assertion: these paths differ from v25 and NOTHING else does
    changed = _paths_that_differ(BASE_SETS["v25"], v)
    want = {"arm_npz", "arm_npz_sha", "bag_by_served_month.2026-07.cache",
            "extension_by_served_month.2026-07.months",
            "extension_by_served_month.2026-07.rows",
            "extension_by_served_month.2026-07.shas.2025-06",
            "extension_by_served_month.2026-07.what",
            "out", "sibling.file", "sibling.sha", "what"}
    if changed != want:
        raise SystemExit(f"v26's derivation from v25 changes {sorted(changed)}, not "
                         f"{sorted(want)}. A base set is a description of a run and every "
                         f"difference from its sibling must be named (section 47.58)")
    if v["bag_by_served_month"]["2026-01"]["cache"] != \
            BASE_SETS["v25"]["bag_by_served_month"]["2026-01"]["cache"]:
        raise SystemExit("v26's January cache key moved; the January head did not")
    return v


def _paths_that_differ(a: dict, b: dict, prefix: str = "") -> set:
    out = set()
    for k in sorted(set(a) | set(b)):
        pa, pb = a.get(k, KeyError), b.get(k, KeyError)
        here = f"{prefix}{k}"
        if isinstance(pa, dict) and isinstance(pb, dict):
            out |= _paths_that_differ(pa, pb, here + ".")
        elif pa != pb:
            out.add(here)
    return out


BASE_SETS["v26"] = _derive_v26()
# the mechanics rehearsal: v26 with a STAND-IN arm, refusing everything but --plan.  The
# five-month set is REAL here (2025-06 exists and its shas are verified); the only thing that
# does not exist is the bag, so the stand-in is the arm and nothing else.
_v26plan = copy.deepcopy(BASE_SETS["v26"])
# 47.49's discipline, applied again: the rehearsal stands the FOUR-month July set where the
# five-month one will go, so the mechanics can be exercised without READING 2025-06. June
# exists and its shas are verified, but the desk's slot is design-only and a row count
# harvested here would be a June read. Nothing is invented: the two served months still get
# different sets, different row counts and different cache keys.
_v26plan["extension_by_served_month"]["2026-07"]["months"] = [
    "2025-11", "2025-10", "2025-03", "2025-08"]
_v26plan["extension_by_served_month"]["2026-07"]["rows"] = 386574
_v26plan["extension_by_served_month"]["2026-07"]["shas"].pop("2025-06")
BASE_SETS["v26plan"] = {**_v26plan,
                        "plan_only": True,
                        "arm_npz": EXP / "work/d-seasonal-bag10-255-free.npz",
                        "arm_npz_sha": ("3edfa35fee55e1d03ec2df72192eeb59"
                                        "6732e62f8164b162e9ef42597ecea6f5"),
                        "out": EXP / "work/dry-run.parquet",
                        "what": ("47.58's MECHANICS for v26: row counts, cache keys, the "
                                 "composition census and the refusals. The arm is v11's "
                                 "seasonal bag standing in for a bag that does not exist "
                                 "yet, and NO reading may be taken through it")}

#: **WHAT A NAMED HEAD VECTOR ACTUALLY CONTAINS**, keyed on the file (round 7, finding 3).
#: Round 6 finding 2 established that a head vector's contents are a fact about the FILE and
#: never about the arm's name, and the repair then put one sentence in `increment_gate` --
#: v18's -- which every later serve inherited.  A serve that cannot name its head's
#: provenance says so instead of borrowing the nearest one.
HEAD_VECTOR_PROVENANCE = {
    "d-blockmove-guard995.npz": (
        "the served arm on matcher eded54e5 at july 287 / december 288 airport-days, "
        "saved by code/blockmove.py at 07:30 on 2026-09-09 (a671e65). NOT at the v18 "
        "serve's pins, so the increment it prices carries the rebuild and the population "
        "re-writes as well as the turnaround pass; analysis/pass-only.json isolates the "
        "pass (round 6, finding 2)"),
    "d-passonly-6fdf3951-guard995.npz": (
        "the served arm refitted by code/passonly.py on the surviving 6fdf3951 caches: "
        "july 8179f74417c8cb99 (296 airport-days) and january cb4d6289cb421aa2 (218), "
        "IDENTICAL to the v19 serve's pins, and december 34a155e40aac9ff6 (288) against "
        "the serve's d35b7171d7062341 (297), the nine days being 2025-11-30 prefix rows "
        "and not scored rows. SHARES is july and january, so the board number is at this "
        "serve's pins exactly; december carries no board share (exp-047-queue audit 7al)"),
    # **ROUND 9, N2.**  This entry is keyed on the FILE (round 7, finding 3) and every
    # clause in it is a fact about the file: what fitted it, on what class, at what size,
    # under which guard and at which pins.  It used to end "...and the rider is that same
    # fit with the LFPG hold lifted -- so the increment carries NO refit", which is not a
    # fact about the file at all: it describes the file's RELATIONSHIP TO A PARTICULAR
    # ARM, and v21 names this same head with the bagged 255, five fresh fits, of which it
    # is exactly false.  Keying on the file was right; the clause simply did not belong in
    # a file-keyed string.  Anything about the head-and-arm pairing now lives in the base
    # set as `head_npz_note` and is appended per serve.
    "d-bundle-ungated1299-guard995.npz": (
        "the vector v18's BYTES ship: the bundle's fit (code/chainbundle.py) on the "
        "ungated class at the 28.2 gate, 255 leaves, the frozen p99.5 guard, the floor "
        "and LFPG held, at this serve's own pins july 8179f74417c8cb99 / december "
        "d35b7171d7062341 / january cb4d6289cb421aa2. It is the head whenever v18 is the "
        "base, because v18 is what is on the board (reading 17, 246.3605)"),
}
DRY = EXP / "work/dry-run.parquet"
#: exp-186: the 2026 months of the ranking mode ($PRC_RANKING_MODE; prc/data.py `rank_months`):
#: ["2026-01", "2026-07"] for ranking.parquet, the four final months for the final file
#: (ranking_block and the serve).
RANK_MONTHS = list(D.rank_months())
#: "within a third of the fold side's" (the assignment) is a third EITHER WAY, so the
#: admissible band on the ratio is [2/3, 4/3].  It was [1/3, 3] and that is three times
#: too wide (Sol round 1, blocking 4).
RMS_BAND = (2.0 / 3.0, 4.0 / 3.0)
MIN_PRED = A.MIN_PRED


def _ungated_modules():
    """`block_ungated` and `chainfold.restrict`, imported lazily and exactly once.

    `code/chainfold.py` imports THIS module, so the import happens here and not at the
    top of the file, and the running module is registered under its own name first --
    otherwise a serve started as `__main__` would execute `serve.py` a second time as
    `serve` and `chainfold` would hold a different copy of every constant.
    """
    global _UNGATED_MODULES
    if _UNGATED_MODULES is None:
        sys.modules.setdefault("serve", sys.modules[__name__])
        import chainfold as CF                                       # noqa: E402
        _UNGATED_MODULES = (CF.BU, CF.restrict, CF)
    return _UNGATED_MODULES


def bagged_ranking_update(spec: dict, rule: str, fit_on, dep_rows: int, usable):
    """The RANKING-side bag: K fits on the same K perturbed fit-panels, averaged.

    **Why this exists, and what the v21 dry run found.**  `arm_npz` is the FOLD-side
    vector; the ranking-side update is built by fitting the named arm once, right here.
    So a base set that names a bagged fold vector and nothing else prices the bag on the
    folds and then serves the ORDINARY SINGLE FIT -- which for v21 is exactly v6, and the
    sibling check caught it at 0 rows differing.  A bag is not a vector you can splice
    onto the ranking side from the fold side: it is a recipe, and the recipe has to be run
    on both.

    So the members here are the same five recipes as the fold members: the same seeds, the
    same `classplacebo.MOVED` counts, the same draw order (`arms.FOLDS`, so the random
    stream is consumed identically), the same 255-leaf setting.  Only the FIT PANEL is
    perturbed; the ranking side (`s2`) never moves, so a member differs from the shipped
    fit exactly by its fit-panel class and by nothing else.  The mean is returned raw --
    the caller's own guard, hold and floor then run on it unchanged, which is what keeps
    the bagged and the single path one path from here on.

    Each member is cached under its own name so the record can carry a sha per member.
    """
    CF = _ungated_modules()[2]
    sys.modules.setdefault("serve", sys.modules[__name__])
    import classplacebo as CP                                        # noqa: E402
    thr = float(UNGATED_GATE["check_i_abs_dt_offblock_max_s"])

    raw = {}
    for f in A.FOLDS:                       # this order fixes the random stream
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        n3 = int((b.match_mode.fillna(-1) == 3).sum())
        if n3 != CF.MODE3_REFERENCE[m]:
            raise SystemExit(f"{m}: mode-3 rows {n3} != {CF.MODE3_REFERENCE[m]} -- the "
                             "bag would be drawn on disturbed placements")
        raw[m] = b

    # the fold members this bag's ranking side must correspond to, checked by sha before
    # a single fit runs: if a fold member on disk is not the one the fold price was taken
    # on, "the same five recipes on both sides" is not true and nothing should be served
    fold_shas = {}
    for k, (rel, want) in sorted(spec.get("fold_members", {}).items()):
        got = D.sha256(EXP / rel)
        if got != want:
            raise SystemExit(f"fold member {k} ({rel}) is {got[:12]}..., not the "
                             f"{want[:12]}... this bag was priced on")
        fold_shas[k] = got

    members, census, shas = [], {}, {}
    real = A.month_block
    try:
        for seed in spec["seeds"]:
            cache = EXP / spec["cache"].format(seed=seed)
            if cache.exists():
                d_i = np.load(cache)["d"].astype("float64")
                census[str(seed)] = {"refitted_this_run": False}
            else:
                rng = np.random.default_rng(seed)
                blocks, cen = {}, {}
                for m, b in raw.items():
                    pb, c = CP.placebo_block(b, thr, CF.GAP_CUT_S, CP.MOVED[m], rng)
                    blocks[m] = pb
                    cen[m] = {k: v for k, v in c.items() if k != "rows_moved"}
                A.month_block = (lambda blocks: (
                    lambda month, prefix_months, rule=None: blocks[month]))(blocks)
                t0 = time.time()
                d_i, _ = fit_on(A.panel_with_block(BA.load()))
                np.savez_compressed(cache, d=np.asarray(d_i, dtype="float64"))
                census[str(seed)] = {"refitted_this_run": True, "census": cen,
                                     "seconds": round(time.time() - t0, 1)}
                print(f"ranking member {seed}: fitted ({time.time()-t0:.0f}s)", flush=True)
            if len(d_i) != dep_rows:
                raise SystemExit(f"ranking member {seed} has {len(d_i)} rows, not "
                                 f"{dep_rows}")
            # **A MEMBER MUST BELONG TO THE CLASS THIS RUN SERVES.**  The cache name keys
            # on seed and leaves only, so a member fitted under one gate setting is
            # silently reused under another: on 2026-09-09 members 1-5 came from the
            # `--ungated` v21 serve (class 196,256) and 6-10 from a run without the flag
            # (class 193,800), and the two averaged into a vector that moved 2,456 rows
            # outside the class.  The structural assert in `main` caught it only on the
            # MEAN, with no seed named.  Here it names the member and the flag.
            out = int(np.count_nonzero(np.asarray(d_i)[~usable]))
            if out:
                raise SystemExit(
                    f"ranking member {seed} ({cache.name}) moves {out} rows OUTSIDE the "
                    f"class this run serves ({int(usable.sum())} usable). It was fitted "
                    f"under a different gate setting -- most likely with/without "
                    f"--ungated. Delete it and refit under this run's flags; do NOT "
                    f"average members from two different classes.")
            shas[str(seed)] = D.sha256(cache)
            members.append(d_i)
    finally:
        A.month_block = real

    stack = np.vstack(members)
    d = stack.mean(0)
    info = {
        "bagged": True, "K": len(members), "seeds": list(spec["seeds"]),
        "leaves": spec.get("leaves"),
        "member_files": {k: str((EXP / spec["cache"].format(seed=int(k)))
                                .relative_to(ROOT)) for k in shas},
        "member_sha256": shas,
        "fold_member_sha256": fold_shas,
        "fold_members_checked_before_any_fit": bool(fold_shas),
        "member_census": census,
        "member_disagreement_rms_s": float(np.sqrt(np.mean(stack.var(0)))),
        "rows_moved_by_the_bag": int((d != 0).sum()),
        "what": ("the unweighted mean of K ranking-side fits of the shipped recipe on the "
                 "same K perturbed fit-panels the fold members use -- same seeds, same "
                 "MOVED counts, same draw order, so the fold price describes the arm that "
                 "ships. No member was selected"),
    }
    return d, info


def extension_rows(spec: dict, scols: list) -> tuple:
    """the extension month's writable rows, for the RANKING side of an extended head.

    One builder for both sides: `code/onemonth.py` derives the month's panel attributes,
    checks the base6 vector and the neighbour block by sha, and joins the shipped columns.
    The fold side (`code/onemonthbag.py`) calls the same function with the same argument,
    so the rows the arm ships on are the rows it was priced on and not a second
    derivation of them.  The row count is checked against the number the section 47.20
    registration named, because a silent change there changes the arm.
    """
    sys.modules.setdefault("serve", sys.modules[__name__])
    import onemonth as OM                                              # noqa: E402
    # a base set names either one month or a stack of them; a single-month spec goes
    # through the SAME path, so v9's extension is unchanged by the generalisation
    months = list(spec.get("months") or [spec["month"]])
    want_each = spec.get("month_rows") or {}
    ext = None
    per: dict = {}
    for mo in months:
        if mo not in OM.MONTHS:
            raise SystemExit(f"the base set extends on {mo}, which code/onemonth.py does "
                             f"not register; known: {sorted(OM.MONTHS)}")
        e = OM.nov_rows(list(scols), month=mo)
        e = e[e.usable.to_numpy(bool)].reset_index(drop=True)
        if mo in want_each and len(e) != int(want_each[mo]):
            raise SystemExit(f"{mo} gives {len(e)} writable rows, not the "
                             f"{int(want_each[mo])} this base set was priced on")
        per[mo] = int(len(e))
        if ext is None:
            ext = e
        else:
            miss = [c for c in ext.columns if c not in e.columns]
            if miss:
                raise SystemExit(f"{mo} lacks {len(miss)} column(s) the stack carries: "
                                 f"{miss[:8]} -- every month must be one frame")
            ext = pd.concat([ext, e.reindex(columns=ext.columns)], ignore_index=True)
    if len(ext) != int(spec["rows"]):
        raise SystemExit(f"the extension gives {len(ext)} writable rows, not the "
                         f"{int(spec['rows'])} this base set was priced on")
    if sum(per.values()) != len(ext):
        raise SystemExit("the stack did not concatenate to the sum of its months")
    print(f"extension {'+'.join(months)}: {len(ext):,} writable rows "
          f"({', '.join(f'{m} {n:,}' for m, n in per.items())})", flush=True)
    return ext, list(scols)


def install_ungated_fold_side(rule: str) -> dict:
    """Point `arms.month_block` at the ungated block, restricted to the 28.2 gate.

    Every fold-side reading in this file goes through `arms.panel_with_block` -- the
    serving fit, the increment gate, the guard's re-derivation and the per-airport RMS --
    so ONE patch moves all four together.  The blocks are the ones `code/chainfold.py`
    built and cached, at this serve's own fold pins, and the shipped mode-3 counts are
    checked against `chainfold.MODE3_REFERENCE` first: the ungated pass must ADD mode-4
    rows and must not have disturbed a single shipped placement.

    The patch is not restored.  A serve is one process with one panel, and a half-patched
    process is exactly the failure this replaces.
    """
    BU, restrict, CF = _ungated_modules()
    if CF.GAP_CUT_S != UNGATED_GATE["step_0_gap_cut_s"]:
        raise SystemExit(f"chainfold prices step 0 at {CF.GAP_CUT_S} s and this serve "
                         f"would write {UNGATED_GATE['step_0_gap_cut_s']} s")
    rblocks, census = {}, {}
    for f in A.FOLDS:
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        n3 = int((b.match_mode.fillna(-1) == 3).sum())
        if n3 != CF.MODE3_REFERENCE[m]:
            raise SystemExit(f"{m}: mode-3 rows {n3} != {CF.MODE3_REFERENCE[m]} -- the "
                             "ungated pass disturbed the shipped placements")
        r = restrict(b, UNGATED_GATE["check_i_abs_dt_offblock_max_s"],
                     UNGATED_GATE["step_0_gap_cut_s"],
                     UNGATED_GATE["check_ii_runway_agreement"])
        rblocks[m] = r
        census[m] = {"mode3_shipped": n3,
                     "mode4_admitted_by_the_probe":
                         int((b.match_mode.fillna(-1) == 4).sum()),
                     "mode4_kept_by_the_gate":
                         int((r.match_mode.fillna(-1) == 4).sum())}

    def month_block(month, prefix_months, rule=None):
        return rblocks[month]

    A.month_block = month_block
    return census


def ranking_block() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The ranking file's departure rows and their feature block.

    The prefix map is learned on the same NINE non-fold 2025 months the fold blocks use.
    None of the twelve is a month being served, so twelve would also be out of sample --
    but then the block would be a slightly different function on the fold side and the
    serving side, and the weights are fitted on one and applied to the other (round 2
    FALLBACK, non-blocking).  One map, one function.
    """
    rk = D.load_ranking()
    dep = rk[rk.PHASE_mvt == "DEP"].copy()
    if dep.BLOCK_TIME_UTC_mvt.notna().any() or dep[D.TARGET].notna().any():
        raise SystemExit("the ranking file carries a departure block time or target")
    tr = pd.concat([pd.read_parquet(D.month_file(m),
                                    columns=["PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt"])
                    for m in A.prefix_months_for("")], ignore_index=True)
    tr = tr[tr.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt")
    # the serve rests on one and the same rule version as the fold table (desk,
    # 2026-09-08), resolved ONCE across every fold month and every ranking month by
    # `arms.serving_rule()`; days written by another rule are refused, not mixed in
    rule, rule_report = A.serving_rule()
    # ONE BLOCK PER MONTH, as `arms.month_block` builds the fold side (round 3 FALLBACK,
    # finding 1).  Passing both ranking months to `B.build` in one call made
    # `_arrival_side` take the previous segment per (airport, icao) over the whole loaded
    # span, so an aircraft's first July segment took its last JANUARY segment as its
    # "previous" one -- a five-month gap the fold side, which builds a month at a time,
    # can never see.  It moved `arr_stand_occupancy_s` / `arr_gap_reception_s` on 1,024 of
    # the 190,095 served rows, worth about 0.05 s of board, and it is a train/serve skew:
    # the same row is a different row on the two sides.  Per month, the two sides compute
    # one and the same function.
    dep["_m"] = dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    parts, covers = [], {}
    for m in RANK_MONTHS:
        sel = dep._m == m
        if not sel.any():
            raise SystemExit(f"the ranking file carries no departure in {m}")
        # the ARR side of the SAME month, for the turnaround pass: the arrival is what
        # gives an unclaimed segment its stand chain.  Per month, so the serve and the
        # fold side (`code/arms.py`) compute one and the same function; without it the
        # pass is a silent no-op and the served block would lose 3,179 clocks.
        arr = rk[rk.PHASE_mvt.eq("ARR")
                 & rk.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m").eq(m)]
        if _UNGATED:
            # the SAME builder with the +-60 s clause's refusals admitted as mode 4, then
            # exactly the 28.2 gate over them.  A refused mode-4 row is blanked back to
            # unmatched, so this month's block is the shipped block plus the kept rows and
            # nothing else; `restrict` does not touch a mode-1/2/3 placement.
            BU, restrict, _ = _ungated_modules()
            BU.PROBE.clear()
            b = BU.build(dep[sel].drop(columns="_m").copy(), tr, require_rule=rule,
                         arrivals=arr)
            at = dict(b.attrs)
            b = restrict(b, UNGATED_GATE["check_i_abs_dt_offblock_max_s"],
                         UNGATED_GATE["step_0_gap_cut_s"],
                         UNGATED_GATE["check_ii_runway_agreement"])
            b.attrs.update(at)
        else:
            b = B.build(dep[sel].drop(columns="_m").copy(), tr, require_rule=rule,
                        arrivals=arr)
        covers[m] = dict(b.attrs)
        parts.append(b)
    dep = dep.drop(columns="_m")
    blk = pd.concat(parts, ignore_index=True)
    blk.attrs["rule_cover"] = rule_report
    blk.attrs["built_per_month"] = covers
    dep = dep.merge(blk, on="MVT_ID_mvt", how="left", validate="one_to_one")
    dep["obs_matched"] = dep.obs_matched.astype("boolean").fillna(False).astype(bool)
    dep["obs_has_clock"] = dep.obs_has_clock.astype("boolean").fillna(False).astype(bool)
    dep["noac"] = dep.AIRCRAFT_TYPE_mvt.isna()
    dep["airport"] = dep.ADEP_mvt.astype(str)
    dep["mvt_s"] = (dep.MVT_TIME_UTC_mvt - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    dep["usable"] = A.served_class(dep)          # one definition, with the fold side
    dep["obs_wide"] = dep.match_mode.fillna(0.0).gt(0.0)
    # the ranking side's own reach, for the fold table's projection (desk, 16:50): how
    # many rows each airport carries in the RANKING file and how many of them this lane
    # can move.  Written here because this is the one place the real ranking block exists.
    dep["rank_month"] = dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    reach = {"basis": "the ranking file under the serving rule, per airport",
             "rule_sha256": rule, "rows": int(len(dep)),
             "by_airport": {}, "by_airport_month": {}}
    for a, g in dep.groupby("airport"):
        reach["by_airport"][str(a)] = {"rows": int(len(g)),
                                       "with_clock": int(g.usable.sum())}
    for (a, m), g in dep.groupby(["airport", "rank_month"]):
        reach["by_airport_month"][f"{a} {m}"] = {"rows": int(len(g)),
                                                 "with_clock": int(g.usable.sum())}
    (EXP / "analysis/ranking-reach.json").write_text(
        json.dumps(reach, indent=2, sort_keys=True))
    return dep, blk


def fold_gate(arm: str, held: tuple[str, ...] | list[str] = (),
              npz: Path | None = None) -> dict:
    """The assignment's fold gate, read from the measured fold table -- never asserted.

    `analysis/arms.json` is written by `arms.py` on the contract.  A candidate may reach
    the candidate path only when that arm's all-row paired gain over the base is positive and
    every fold interval AND the board endpoint are clear of zero.  Before Sol round 1
    (blocking 4) this was a sentence in the report and serving proceeded regardless.

    **The gate is read on the VECTOR THAT SHIPS whenever one is named** -- `npz`, or
    `npz_path(arm)` when an airport is held -- and on the fold table only when neither
    says otherwise (round 8, blocking).  `held` decides which airports are zeroed before
    the re-price, and nothing else.

    **`held` gates the vector that ships, not the one that was priced** (round 3 FALLBACK,
    finding 4).  When an airport is held at the base on the ranking side, the fold table's
    number still contains it, so the gate was reading a different vector from the one being
    written -- `code/heldprice.py` re-priced it afterwards and the shipped candidate did
    pass, but by a second script.  With `held` non-empty this re-prices the SAVED fold-side
    update with the same airports zeroed -- the same `d` with a mask, no refit -- and gates
    on that.  The rule and cache-key check above still applies, because the saved `d`
    belongs to the same run as the table.
    """
    if not (EXP / "analysis/arms.json").exists():
        return {"passes": False, "reason": "analysis/arms.json does not exist yet, so "
                                           "no fold table has priced this arm"}
    table = json.loads((EXP / "analysis/arms.json").read_text())
    rec = priced_arms().get(arm)
    if not rec or "board_all_row" not in rec:
        return {"passes": False, "reason": f"no fold table prices an arm {arm!r}"}
    # the fold table and the serve are two processes, and the harvest can re-derive a
    # month between them.  The gate is only meaningful if the table was priced on the
    # same rule version and the same block caches this serve is about to use
    # (round 2 FALLBACK, blocking 1).
    rule_now, _ = A.serving_rule()
    keys_now = {m: A.block_cache_key(m, rule_now) for m in A.FOLD_MONTH.values()}
    rule_then = table.get("rule", {}).get("sha256")
    keys_then = table.get("rule", {}).get("block_cache_keys")
    if rule_then != rule_now or keys_then != keys_now:
        return {"passes": False, "rule_then": rule_then, "rule_now": rule_now,
                "block_cache_keys_then": keys_then, "block_cache_keys_now": keys_now,
                "reason": ("analysis/arms.json was priced under a different rule version "
                           "or different block caches than this serve resolves; re-run "
                           "arms.py before serving")}
    priced = {"board_all_row_gain_s": rec["board_all_row"]["gain_s"],
              "every_fold_interval_clear": bool(rec.get("every_fold_interval_clear")),
              "board_interval_ci95": rec["board_all_row"].get(
                  "interval", {}).get("ci95", [float("-inf"), 0.0])}
    read_on, extra = "the fold table as priced, with every airport in it", {}
    # **ROUND 8, BLOCKING.**  This re-price used to sit behind `if held:`, so a serve that
    # holds NO airport fell back to the fold table -- v18's own fit on the shipped block --
    # while `read_on` was free to claim otherwise.  That is the round-7 finding a third
    # time: the repair was applied to the vector-naming but not to the CONDITION, and v20
    # holds no airport, so `serve-v20.json` gated on +9.31283 (the fold table) when the
    # vector it ships prices at +9.46985.  Both pass, so no candidate was mis-released --
    # but the record quoted a number for a vector it was not describing, which is the
    # defect.  The rule now: **whenever a vector is NAMED, the gate is read on that
    # vector**, held or not; `held` only decides which airports are zeroed first, and an
    # empty hold zeroes nothing.
    if npz is not None or held:
        # `npz` names the vector when the arm's NAME cannot -- the same repair round 6
        # applied to `increment_gate` and `fold_update_rms` and NOT to this function, which
        # round 7 finding 1 failed the record on: v19 is the head's arm refitted on a bigger
        # class, so `npz_path(arm)` returns v18's fit on the shipped block and this gate
        # would read a vector the serve is not writing while `read_on` below claims it is
        # "the vector that ships".  One of the two releasing gates; it is named explicitly
        # now, and `vector` goes in the record so the claim can be checked against a file.
        z = Path(npz) if npz else npz_path(arm)
        if not z.exists():
            return {"passes": False, "reason":
                    f"{z.name} does not exist, so the gate cannot be read on the vector "
                    f"that ships with {sorted(held)} held at the base"}
        pan = A.panel_with_block(BA.load())
        zz = np.load(z)
        kept, zeroed = {}, {}
        for f in A.FOLDS:
            dv = zz[f].astype("float64").copy()
            m = np.isin(pan[f].airport.to_numpy().astype(str), list(held))
            zeroed[f] = int((dv[m] != 0).sum())
            dv[m] = 0.0
            kept[f] = dv
        pr = BA.price(pan, kept, np.random.default_rng(A.SEED))
        priced = {"board_all_row_gain_s": pr["board_all_row"]["gain_s"],
                  "every_fold_interval_clear": bool(pr["every_fold_interval_clear"]),
                  "board_interval_ci95": pr["board_all_row"]["interval"]["ci95"]}
        read_on = ("the vector that ships: the same saved fold-side update with "
                   f"{sorted(held)} zeroed, no refit") if held else (
            "the vector that ships: the saved fold-side update this base set names, "
            "no refit and no airport held")
        extra = {"airports_held": sorted(held),
                 "vector": z.name,
                 "vector_named_by": ("the base set's arm_npz" if npz
                                     else "npz_path(arm), from the arm's name"),
                 "rows_returned_to_the_base_by_fold": zeroed,
                 "as_priced_with_every_airport": {
                     "board_all_row_gain_s": rec["board_all_row"]["gain_s"],
                     "every_fold_interval_clear": bool(
                         rec.get("every_fold_interval_clear"))}}
    gain = priced["board_all_row_gain_s"]
    clear = priced["every_fold_interval_clear"]
    band = priced["board_interval_ci95"]
    ok = bool(gain > 0 and clear and band[0] > 0)
    return {"passes": ok, "board_all_row_gain_s": gain, "read_on": read_on,
            "every_fold_interval_clear": clear, "board_interval_ci95": band, **extra,
            "reason": "" if ok else
                      f"fold gate: gain {gain:+.4f} s, every fold interval clear "
                      f"{clear}, board interval {band}"}


def priced_arms() -> dict[str, dict]:
    """Every arm this lane has priced on the contract, from all three fold tables.

    `analysis/arms.json` is the arm set, `newarms.json` the columns-added arms
    (`code/newarms.py`) and `capacity.json` the same head at other LightGBM sizes
    (`code/capacity.py`), whose settings enter as `c_lgb_shipped@<setting>`.  They are one
    pool because they were priced on ONE rotation and are compared against each other, and
    the size of that pool is what makes the winner's interval post-selection.
    """
    out: dict[str, dict] = {}
    f = EXP / "analysis/arms.json"
    if f.exists():
        out.update(json.loads(f.read_text()).get("arms", {}))
    f = EXP / "analysis/newarms.json"
    if f.exists():
        out.update(json.loads(f.read_text()).get("arms", {}))
    f = EXP / "analysis/capacity.json"
    if f.exists():
        for k, v in json.loads(f.read_text()).get("settings", {}).items():
            out[f"c_lgb_shipped@{k}"] = v
    # the guarded settings (`code/guardarm.py`): the same fits with the frozen tail guard
    # masked over them.  They are part of the SAME pool -- one rotation, one contract --
    # and the count they add travels with the winner's gain (RESULTS §9.0).
    f = EXP / "analysis/guard-arms.json"
    if f.exists():
        for k, v in json.loads(f.read_text()).get("settings", {}).items():
            out[f"c_lgb_shipped@{k}"] = v
    return out


#: The two priced arms that read the `exp-047` neighbour block.  Named here because the
#: pool count has to know which readings depend on an interface that can be rebuilt under
#: it, and `arms.json` records only the sha, not which arm used it.
NBR_DEPENDENT = ("b_lgb_nbr", "c_lgb_shipped_nbr")


def superseded_neighbour_arms() -> dict:
    """The arms of `arms.json` whose neighbour block has been rebuilt since they were priced.

    `exp-047` rebuilt its five neighbour blocks at 17:14-17:15 UTC on 2026-09-08, in the
    middle of this lane's `_nbr` diagnosis (RESULTS §9.3).  An arm priced against the OLD
    block is a reading of a DIFFERENT INSTRUMENT, not a member of the pool the winner was
    chosen from, so it must neither inflate the selection count nor be eligible to serve
    (round 4 FALLBACK, finding 2).  Read off the files: the sha `arms.json` recorded
    against the sha on disk now, never asserted.
    """
    f = EXP / "analysis/arms.json"
    if not f.exists():
        return {"months_rebuilt": {}, "arms": []}
    rec = json.loads(f.read_text()).get("neighbour_block", {})
    moved = {}
    for m, r in rec.items():
        was, now = r.get("sha256"), None
        blk = ROOT / r.get("file", "")
        if blk.exists():
            now = D.sha256(blk)
        if was and now and was != now:
            moved[m] = {"priced_on": was, "on_disk_now": now}
    return {"months_rebuilt": moved,
            "arms": sorted(NBR_DEPENDENT) if moved else [],
            "what": ("an arm priced against a neighbour block that has since been "
                     "rebuilt is a reading of a different instrument")}


def _vector_fingerprint(arm: str) -> str | None:
    """A digest of an arm's three saved fold-side update vectors, or None if absent.

    Two table entries can be ONE arm: `capacity.json`'s `predeclared_31x300_lr050` is the
    same LightGBM as `arms.json`'s `c_lgb_shipped` (its parameter override is `None`), and
    counting both inflates the count the winner's interval is quoted with (round 4
    FALLBACK, finding 2).  Identity is MEASURED on the vectors, not inferred from names or
    from gains that could agree by chance.
    """
    z = npz_path(arm)
    if not z.exists():
        return None
    d = np.load(z)
    h = hashlib.sha256()
    for f in A.FOLDS:
        if f not in d:
            return None
        h.update(np.ascontiguousarray(d[f], dtype="float64").tobytes())
    return h.hexdigest()


def increment_gate(arm: str, head_arm: str, held: tuple[str, ...] | list[str] = (),
                   rng: np.random.Generator | None = None,
                   head_npz: Path | None = None,
                   unselected: bool = False,
                   new_npz: Path | None = None,
                   lit_rule: bool = False,
                   head_npz_note: str | None = None) -> dict:
    """The gate on what this upload CHANGES: the increment over the served head.

    The desk, 2026-09-08 19:30 UTC: "the gate for this upload is the INCREMENT over the
    served head, not over v12".  v13 already serves the with-clock class, so a second
    candidate on the same class is worth its slot only if `p_new - p_v13` is itself a gain,
    and a large gain over v12 says nothing about that -- most of it is already shipped.

    Built from the two saved fold-side update vectors with the same hold applied to both:
    the head's own base becomes `pbase + d_head`, the residual becomes `y - pbase - d_head`,
    and the increment `d_new - d_head` is priced against it with the same day-clustered
    intervals, the same clean split, december beside, and the tail clause on both.
    """
    rng = rng or np.random.default_rng(A.SEED)
    # `head_npz` names the head's fold vector explicitly, for the case `npz_path` cannot
    # express: the head and the candidate are THE SAME ARM on two different blocks.  A
    # matcher change re-prices the arm in place, so `npz_path(head_arm)` would return the
    # file the new fit just overwrote and the increment would be identically zero -- a
    # gate reading "no gain" for the arithmetic reason that it differenced a vector with
    # itself.  v18 is that case.  The head vector named by the v17 base set is the arm on
    # matcher `eded54e5` at july 287 / december 288 pins (round 6, finding 2), so the
    # increment it prices carries the rebuild and the population re-writes as well as the
    # pass; `code/passonly.py` isolates the pass on the surviving `6fdf3951` caches.
    # `new_npz` is the same repair on the candidate's side: `npz_path(arm)` names the
    # vector the arm's NAME implies, and the bundle is that arm refitted on a class its
    # name does not carry, so the gate would otherwise price v18's fit and call it v19's.
    zn = Path(new_npz) if new_npz else npz_path(arm)
    zh = Path(head_npz) if head_npz else npz_path(head_arm)
    for z in (zn, zh):
        if not z.exists():
            return {"passes": False, "reason": f"{z.name} does not exist, so the "
                                               "increment over the head cannot be priced"}
    pan = A.panel_with_block(BA.load())
    new_z, head_z = np.load(zn), np.load(zh)
    hm = {f: np.isin(pan[f].airport.to_numpy().astype(str), list(held))
          for f in A.FOLDS}
    d_new, d_head, inc, pan_head = {}, {}, {}, {}
    for f in A.FOLDS:
        a = new_z[f].astype("float64").copy()
        b = head_z[f].astype("float64").copy()
        a[hm[f]] = 0.0
        b[hm[f]] = 0.0
        d_new[f], d_head[f] = a, b
        inc[f] = a - b
        q = pan[f].copy()
        q["pbase"] = q["pbase"].to_numpy("float64") + b
        q["rbase"] = q["y_true"].to_numpy("float64") - q["pbase"].to_numpy("float64")
        pan_head[f] = q
    rec = BA.price(pan_head, inc, rng)
    clear = bool(rec["every_fold_interval_clear"])
    band = rec["board_all_row"]["interval"]["ci95"]
    gain = rec["board_all_row"]["gain_s"]
    selected_ok = bool(gain > 0 and clear and band[0] > 0)

    # **WHICH RULE GATES THIS INCREMENT** -- the registered one, and only that.
    #
    # History, because this line was wrong once and the record has to say so.  On
    # 2026-09-09 the desk ruled that an UNSELECTED update -- one choosing nothing on fold
    # evidence -- could be gated as the level currency gates an increment (pooled clear,
    # no fold clear below zero, no fold declining) rather than by every fold clearing,
    # and this function was written to switch on that.  **Review round 6 FAILED that
    # reading and the desk REVERSED it the same day**: the registered rule stands --
    # `every_fold_interval_clear` for an in-fold arm whose form was not selected (13.1,
    # section 15, recorded rule, recorded rule, and the v10 precedent at 0.26x).  The gate is therefore
    # ALWAYS `selected_ok`.
    #
    # `unselected` is now exactly what the comment beneath it always claimed and the code
    # did not do: a RECORD-ONLY property of the change.  It names which reading the desk
    # is being shown; it can never turn a failing gate into a passing one.  The second
    # verdict stays in `verdict_under_each_rule` as a diagnostic, so a MAYBE keeps the
    # evidence for its revival without that evidence being able to release it.
    per_fold = {f: rec["by_fold"][f]["all_row"] for f in A.FOLDS}
    none_clear_below = all(v["ci95"][1] >= 0 for v in per_fold.values())
    none_declining = all(v["mse_gain_s2"] > 0 for v in per_fold.values())
    unselected_ok = bool(gain > 0 and band[0] > 0 and none_clear_below and none_declining)
    ok = selected_ok

    # **THE LIT RULE (26.5), applied only where a base set asks for it.**  The registered
    # ruling is narrow and every clause of it is checked here rather than assumed:
    #
    #   "an EMPTY lit fold is not a crossing -- a cell dark by instrument is priced on its
    #    lit folds (>= 2, december among them for a winter ranking month)"
    #
    # A fold whose interval CROSSES zero is evidence against and still fails; a fold with
    # NO MOVED ROW AT ALL is no evidence, and `base.price` has carried the machinery since
    # it was written (`folds_with_no_moved_row`, `every_moving_fold_interval_clear`).  The
    # rule cannot be reached by accident: `lit_rule` is a property of the base set, and it
    # only ever fires when the increment leaves a fold literally empty.  Both readings go
    # in the record whatever happens, so a reader sees which gate released the candidate.
    dark = list(rec["folds_with_no_moved_row"])
    lit = [f for f in A.FOLDS if f not in dark]
    lit_clauses = {
        "a_fold_is_dark": bool(dark),
        "every_dark_fold_moves_zero_rows": all(
            int(np.count_nonzero(inc[f])) == 0 for f in dark),
        "at_least_two_lit_folds": bool(len(lit) >= 2),
        "december_is_lit": bool("december" in lit),
        "every_moving_fold_interval_clear": bool(rec["every_moving_fold_interval_clear"]),
        "pooled_gain_positive_and_clear": bool(gain > 0 and band[0] > 0),
    }
    lit_ok = bool(all(lit_clauses.values()))
    if lit_rule and lit_ok:
        ok = True
    return {
        "passes": ok,
        "rule_applied": ("the registered rule: every fold interval clear"
                         if not (lit_rule and lit_ok and not selected_ok) else
                         "the registered rule, read under the lit rule of section 26.5: "
                         "every MOVING fold interval clear, an empty lit fold being no "
                         "crossing (>= 2 lit folds, december among them)"),
        "lit_rule": {
            "asked_for_by_the_base_set": bool(lit_rule),
            "clauses": lit_clauses, "all_clauses_hold": lit_ok,
            "lit_folds": lit, "dark_folds": dark,
            "released_the_candidate": bool(lit_rule and lit_ok and not selected_ok),
            "ruling": ("desk, 2026-09-09, section 26.5: an EMPTY lit fold is not a "
                       "crossing -- a cell dark by instrument is priced on its lit folds "
                       "(>= 2, december among them for a winter ranking month). A fold "
                       "whose interval CROSSES zero is evidence against and still fails")},
        "arm_vector": str(zn.name),
        "head_vector": str(zh.name),
        # ROUND 7, FINDING 3: this string used to emit v18's sentence under any serve
        # that named a head vector, so v19's record described another serve's head.  It is
        # keyed on the file now, and a head vector with no entry says so rather than
        # borrowing the nearest description.
        "head_vector_provenance": (
            (HEAD_VECTOR_PROVENANCE.get(
                zh.name,
                f"{zh.name}: no registered provenance for this vector; round 6 finding 2 "
                "is why one is required -- what a named head vector contains is a fact "
                "about the file and never about the arm's name")
             + (f". {head_npz_note}" if head_npz_note else ""))
            if head_npz else
            "npz_path(head_arm), the vector the head arm was priced on"),
        "the_unselected_reading_is_record_only": (
            "desk, 2026-09-09, reversing its own 05:40 ruling after review round 6: the "
            "pooled/no-fold-declining reading is recorded beside the gate and never "
            "applied as one"),
        "unselected": bool(unselected),
        "verdict_under_each_rule": {
            "selected_variant_every_fold_clear": selected_ok,
            "unselected_update_pooled_and_no_fold_declining": unselected_ok,
            "pooled_interval_clear_of_zero": bool(gain > 0 and band[0] > 0),
            "no_fold_interval_clear_below_zero": bool(none_clear_below),
            "no_fold_declining_on_its_point_estimate": bool(none_declining),
            "fold_lower_bounds_s2": {f: v["ci95"][0] for f, v in per_fold.items()},
            "fold_point_estimates_s2": {f: v["mse_gain_s2"] for f, v in per_fold.items()},
        },
        "what": ("the increment over the served head: d = d_new - d_head on the with-clock "
                 "rows, priced against a base of pbase + d_head, with the airport hold "
                 "applied to BOTH vectors so the comparison is like for like"),
        "head_arm": head_arm, "arm": arm, "held": sorted(held),
        "board_all_row_gain_s": gain, "board_all_row_interval_ci95": band,
        "board_clean_gain_s": rec["board_clean"]["gain_s"],
        "board_clean_interval_ci95": rec["board_clean"]["interval"]["ci95"],
        "december_gain_s": rec["december_rmse"]["gain_s"],
        "every_fold_interval_clear": clear,
        "by_fold": {f: {"all_row": rec["by_fold"][f]["all_row"],
                        "rmse_head": rec["by_fold"][f]["rmse_base"],
                        "rmse_new": rec["by_fold"][f]["rmse_arm"]} for f in A.FOLDS},
        "rows_moved_by_the_increment": {f: int(np.count_nonzero(inc[f]))
                                        for f in A.FOLDS},
        "tail_clause_on_the_increment": {
            f: A.tail_clauses(pan_head[f], inc[f])["registered_clause_y_over_3600s"]
            for f in A.FOLDS},
        "tail_clause_on_the_whole_update": {
            f: A.tail_clauses(pan[f], d_new[f])["registered_clause_y_over_3600s"]
            for f in A.FOLDS},
        "reason": "" if ok else (
            f"increment gate: gain {gain:+.4f} s over the head, every fold interval "
            f"clear {clear}, board interval {band}. An increment whose interval crosses "
            "zero on any fold is inside the floor and does not take a slot."),
    }


def tail_clause(arm: str, held: tuple[str, ...] | list[str] = (),
                npz: Path | None = None) -> dict:
    """The registered k2climb tail clause per fold, read from the fold table.

    A SECONDARY GUARD, not a gate (tail rule): v12 shipped with a -3.86 s january tail reading
    recorded as its transfer risk.  It is reported on every serve and it is part of the
    desk's arm-choice rule of 2026-09-08 18:05, so it is computed here rather than read
    off a table by hand.

    **`npz` names the vector that SHIPS** (round 7 finding 1).  The fold table is keyed on
    the arm's name, and v19 is the head's arm refitted on a bigger class, so the table's
    clause describes v18's fit.  When a vector is named the clause is recomputed on it,
    with the same airport hold the serve applies, and both readings are returned -- the
    served one under `by_fold`, the table's beside it, so neither can be mistaken for the
    other.
    """
    rec = priced_arms().get(arm, {})
    tc = rec.get("tail_clauses")
    if not tc:
        return {"available": False}
    per = {k: {"rmse_gain_s": v["registered_clause_y_over_3600s"]["rmse_gain_s"],
               "passes": v["registered_clause_y_over_3600s"]["passes_the_clause"]}
           for k, v in tc.items()}
    as_priced, read_on = per, "the fold table, keyed on the arm's name"
    if npz is not None:
        z = Path(npz)
        if not z.exists():
            return {"available": False,
                    "reason": f"{z.name} does not exist, so the clause cannot be read on "
                              "the vector that ships"}
        pan = A.panel_with_block(BA.load())
        zz = np.load(z)
        per = {}
        for f in A.FOLDS:
            dv = zz[f].astype("float64").copy()
            if held:
                dv[np.isin(pan[f].airport.to_numpy().astype(str), list(held))] = 0.0
            c = A.tail_clauses(pan[f], dv)["registered_clause_y_over_3600s"]
            per[f] = {"rmse_gain_s": c["rmse_gain_s"], "passes": c["passes_the_clause"],
                      "rows": c["rows"], "rows_moved": c["rows_moved"]}
        read_on = (f"the vector that ships ({z.name}), with {sorted(held)} zeroed, "
                   "no refit")
    return {"available": True, "by_fold": per, "read_on": read_on,
            "as_priced_in_the_fold_table": as_priced,
            "passes_on_every_fold": all(v["passes"] for v in per.values()),
            "clause": "rows with y > 3600 s not worse by more than 1 s of RMSE "
                      "(assignments/k2climb.md line 24); D20 reads it as a secondary "
                      "guard and not a gate"}


#: The arms THIS FILE can build on the ranking side.  `main()` constructs the served
#: vector itself, so an arm the fold table prices but the serve cannot construct would be
#: gated on one feature set and served on another.  `choose_arm()` marks any other arm
#: ineligible and `main()` refuses it outright, rather than silently serving the nearest
#: thing it knows how to build.
SERVABLE = ("a_linear", "a_linear_reach", "b_lgb", "c_lgb_shipped")


RANK_POP = EXP / "analysis/ranking-population.json"


def ranking_population(rule: str, is_candidate: bool, sha: str) -> dict:
    """What the RANKING months read, and what changed since the previous serve.

    The desk, 2026-09-09: the fold months are pinned, the ranking months are not -- a
    served file should read everything the harvest has delivered.  The risk that leaves is
    a silent one: a 2026 delivery changes a served file with no key moving anywhere.  So
    the serve records the ranking population per month with the same bytes-and-rule key
    the fold pin hashes, and diffs it against the previous serve's.

    **A change is stated, never a stop.**  This function returns a report and raises
    nothing.  Only a real candidate advances `latest`; a dry run reads the comparison
    without moving it, so the next serve still diffs against the last file that shipped.
    """
    prev = json.loads(RANK_POP.read_text()) if RANK_POP.exists() else {}
    old = (prev.get("latest") or {}).get("months") or {}
    months, changed = {}, {}
    for m in RANK_MONTHS:
        # UNPINNED on purpose: this is the record of what was read, not a gate on it
        with B.pinned(None):
            tri = B.population_triples(B.AIRPORTS, A.month_days(m))
        months[m] = {"block_cache_key": B.population_key(tri, rule),
                     "airport_days": len(tri), "days": tri}
        o = old.get(m)
        if not o:
            changed[m] = {"first_record": True}
            continue
        was = {(x["airport"], x["day"]): x["bytes"] for x in o.get("days", [])}
        now = {(x["airport"], x["day"]): x["bytes"] for x in tri}
        added = sorted(f"{a} {d}" for a, d in now.keys() - was.keys())
        gone = sorted(f"{a} {d}" for a, d in was.keys() - now.keys())
        resized = sorted(f"{a} {d}" for k in now.keys() & was.keys()
                         if now[k] != was[k] for a, d in [k])
        changed[m] = {
            "key_moved": months[m]["block_cache_key"] != o.get("block_cache_key"),
            "key_was": o.get("block_cache_key"), "key_now": months[m]["block_cache_key"],
            "airport_days_was": o.get("airport_days"),
            "airport_days_now": len(tri),
            "days_added": added[:40], "days_added_count": len(added),
            "days_gone": gone[:40], "days_gone_count": len(gone),
            "days_resized": resized[:40], "days_resized_count": len(resized),
        }
    rec = {
        "what": ("the ranking months' population as this serve read it, and the diff "
                 "against the previous serve's; the ranking months are deliberately "
                 "UNPINNED (desk, 2026-09-09) so a delivery reaches the served file, and "
                 "this record is what keeps that visible rather than silent"),
        "a_change_is_stated_never_a_stop": True,
        "months": {m: {k: v for k, v in r.items() if k != "days"}
                   for m, r in months.items()},
        "changed_since_the_previous_serve": changed,
    }
    if is_candidate:
        hist = list(prev.get("history", []))
        if prev.get("latest"):
            hist.append({k: v for k, v in prev["latest"].items() if k != "months"}
                        | {"months": {m: {kk: vv for kk, vv in r.items() if kk != "days"}
                                      for m, r in (prev["latest"].get("months") or {}).items()}})
        RANK_POP.write_text(json.dumps(
            {"what": rec["what"], "written_by": "code/serve.py",
             "latest": {"when_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "candidate_sha256": sha, "months": months},
             "history": hist[-24:]}, indent=2, sort_keys=True))
        rec["recorded_as_latest"] = True
    else:
        rec["recorded_as_latest"] = False
        rec["why_not"] = "a dry run does not advance the record; only a candidate does"
    return rec


def npz_path(arm: str) -> Path:
    """The saved fold-side update vector for an arm, wherever the script that priced it
    chose to put it.  `code/capacity.py` writes `d-capacity-<setting>.npz`; every other
    script writes `d-<arm>.npz`.  One mapping, so a gate cannot read the predeclared
    arm's vector while the file is written from a capacity variant."""
    if "@" in arm:
        setting = arm.split("@", 1)[1]
        # a GUARDED setting is the same fit with the frozen tail guard masked over it
        # (`code/guard.py`, `code/guardarm.py`), and it keeps its own vector so a gate can
        # never read the unguarded one while the file is written from the guarded one
        if G.NAME in setting:
            return EXP / f"work/d-guard-{setting.replace('+', '-')}.npz"
        return EXP / f"work/d-capacity-{setting}.npz"
    return EXP / f"work/d-{arm}.npz"


def capacity_params(arm: str) -> dict | None:
    """`c_lgb_shipped@<setting>` -> that setting's LightGBM override, or None.

    A capacity setting is the SERVED HEAD AT A DIFFERENT SIZE: the same 103 columns, the
    same rotation, the same amplitude, the same floor.  So it needs no ranking-side
    construction of its own and this file can build it, which is exactly what the geometry
    and neighbour arms cannot claim -- they need columns `ranking_block()` does not
    compute, and `SERVABLE` refuses them for that reason (round 2 FALLBACK: the gate would
    pass on one feature set while the file was written from another).
    """
    if "@" not in arm:
        return None
    head, setting = arm.split("@", 1)
    # the guard changes no model parameter: it is a mask over the same fit, so the
    # LightGBM override is the underlying setting's
    setting = setting.split("+", 1)[0]
    if head != "c_lgb_shipped":
        raise SystemExit(f"{arm!r}: only c_lgb_shipped has a capacity variant")
    f = EXP / "analysis/capacity.json"
    if not f.exists():
        raise SystemExit(f"{arm!r}: analysis/capacity.json does not exist")
    table = json.loads(f.read_text())
    if setting not in table.get("settings", {}):
        raise SystemExit(f"{arm!r}: capacity.json prices no setting {setting!r}")
    # `None` is the predeclared setting, which is `c_lgb_shipped` itself
    return table["settings"][setting].get("lgb") or {}


def is_servable(arm: str) -> bool:
    """`SERVABLE`, extended to the capacity settings of a servable head."""
    if "@" in arm:
        head, setting = arm.split("@", 1)
        if head not in SERVABLE:
            return False
        # a guarded setting is servable exactly when its underlying setting is: the guard
        # reads `pbase` and `obs_taxi`, which every ranking row carries
        f = EXP / "analysis/capacity.json"
        return (f.exists() and setting.split("+", 1)[0]
                in json.loads(f.read_text()).get("settings", {}))
    return arm in SERVABLE


def choose_arm() -> tuple[str, dict]:
    """The desk's rule of 2026-09-08 18:05, as code rather than as a judgement.

    Serve the arm with the largest CLEAN gain among the arms whose every fold interval is
    clear AND whose registered tail clause passes on every fold.  Controls and placebos are
    excluded by name: they exist to be compared against, and an arm that reads a gain by
    construction must never be a candidate.
    """
    table = priced_arms()
    excluded = {a for a in table if a.endswith(("_noobs", "_placebo"))}
    # round 4 FALLBACK, finding 2: the pool the winner was chosen from is not the same
    # thing as the union of three JSON files.  Two entries can be one arm, and an entry
    # can be a reading of an instrument that no longer exists.  Both are measured.
    stale_rep = superseded_neighbour_arms()
    stale = {a for a in stale_rep["arms"] if a in table}
    prints, dupes = {}, {}
    for a in sorted(table):
        if a in excluded or "board_clean" not in table[a]:
            continue
        fp = _vector_fingerprint(a)
        if fp is None:
            continue
        if fp in prints:
            # keep the earlier name, which is the one the record already quotes
            dupes[a] = prints[fp]
        else:
            prints[fp] = a
    rows = []
    for a, rec in table.items():
        if a in excluded or a in dupes or a in stale or "board_clean" not in rec:
            continue
        tc = tail_clause(a)
        ok = bool(rec.get("every_fold_interval_clear")
                  and rec["board_all_row"]["gain_s"] > 0
                  and rec["board_all_row"]["interval"]["ci95"][0] > 0
                  and tc.get("passes_on_every_fold")
                  and is_servable(a))
        rows.append({"arm": a, "eligible": ok, "servable": is_servable(a),
                     "clean_gain_s": rec["board_clean"]["gain_s"],
                     "all_row_gain_s": rec["board_all_row"]["gain_s"],
                     "every_fold_interval_clear": bool(rec.get("every_fold_interval_clear")),
                     "tail_clause_passes": tc.get("passes_on_every_fold")})
    rows.sort(key=lambda r: -r["clean_gain_s"])
    chosen = next((r["arm"] for r in rows if r["eligible"]), None)
    passing = [r for r in rows if r["every_fold_interval_clear"]
               and r["tail_clause_passes"]]
    return chosen, {"rule": ("largest clean gain among arms with every fold interval "
                             "clear and the registered tail clause passing on every "
                             "fold; controls and placebos excluded by name"),
                    "excluded_by_name": sorted(excluded),
                    # measured, not asserted -- see `_vector_fingerprint` and
                    # `superseded_neighbour_arms`
                    "excluded_as_the_same_arm_under_a_second_name": dupes,
                    "excluded_as_a_superseded_instrument": {
                        "arms": sorted(stale), **stale_rep},
                    "considered": rows,
                    "chosen": chosen,
                    # the desk, 2026-09-08 18:40 UTC: the winner's interval is
                    # POST-SELECTION and the count that produced it travels with the gain
                    "selection": {
                        "entries_in_the_three_priced_tables": len(table),
                        "arms_priced_on_this_rotation": len(table) - len(dupes) - len(stale),
                        "arms_compared_after_the_name_exclusions": len(rows),
                        "arms_passing_every_clause": len(passing),
                        "servable_arms_passing_every_clause": len(
                            [r for r in passing if r["servable"]]),
                        "what_this_means": (
                            "the chosen arm's interval is the interval of the LARGEST of "
                            f"{len(passing)} readings that passed every clause on one and "
                            "the same three folds, not the interval of a single "
                            "pre-registered arm. It is quoted with this count beside it "
                            "and is expected to be optimistic by roughly the spread of "
                            "the pool. No correction is applied to it, because a "
                            "correction would be a second choice made on the same data; "
                            "the count is stated instead.")}}


def fold_update_rms(arm: str, pan: dict, npz: Path | None = None) -> dict:
    """The fold side's per-airport RMS of the update, for the serving comparison.

    `npz` names the vector when the arm's NAME cannot: the bundle is the same arm on a
    bigger class, so `npz_path(arm)` would return v18's fit on the shipped block and this
    ratio would compare the served update with a fold update it was not fitted as.
    """
    out: dict[str, list[float]] = {}
    f_npz = Path(npz) if npz else npz_path(arm)
    for f in A.FOLDS:
        dv = np.load(f_npz)[f] if f_npz.exists() else np.zeros(len(pan[f]))
        for a, idx in pan[f].groupby("airport", observed=True).indices.items():
            m = dv[idx] != 0
            if m.any():
                out.setdefault(str(a), []).append(
                    float(np.sqrt(np.mean(dv[idx][m] ** 2))))
    return {a: float(np.mean(v)) for a, v in out.items()}


def main(arm: str = "a_linear", dry_run: bool = False,
         accept_rms_breach: tuple[str, ...] = (),
         hold_airports: tuple[str, ...] = (),
         base_set: str = "v12", ungated: bool = False, plan: bool = False) -> None:
    global _UNGATED
    if D.ranking_mode() != "ranking":
        # exp-186: this serve writes the board candidate and analysis files under fixed names;
        # the final mode uses `ranking_block()` only (through exp-148 features.py)
        raise SystemExit("serve.main runs in the ranking mode only; unset PRC_RANKING_MODE")
    _UNGATED = bool(ungated)
    if not is_servable(arm):
        raise SystemExit(f"{arm!r} is priced on the fold side but this file cannot build "
                         f"it on the ranking side; servable arms are {SERVABLE} and "
                         f"their capacity variants c_lgb_shipped@<setting>")
    if base_set not in BASE_SETS:
        raise SystemExit(f"no base set {base_set!r}; known: {sorted(BASE_SETS)}")
    # A base set is written before its arm exists, so the shas it will verify are filled in
    # as PENDING.  No real sha is that string, so any comparison would fail anyway -- this
    # says WHY, because "sha mismatch" on a candidate serve reads like corruption.
    _pending = []
    _spec = BASE_SETS[base_set]
    if _spec.get("arm_npz_sha") == "PENDING":
        _pending.append("arm_npz_sha")
    for _s, (_f, _h) in (_spec.get("bag", {}).get("fold_members") or {}).items():
        if _h == "PENDING":
            _pending.append(f"bag.fold_members[{_s}]")
    # SECTION 47.49: a seasonal base set declares one extension PER SERVED MONTH, and a
    # month whose shas are not yet forwarded must refuse exactly as an unverified arm does
    for _sm, _es in (_spec.get("extension_by_served_month") or {}).items():
        if _es.get("rows") is None:
            _pending.append(f"extension_by_served_month[{_sm}].rows")
        for _mo, _pair in (_es.get("shas") or {}).items():
            if "PENDING" in _pair:
                _pending.append(f"extension_by_served_month[{_sm}].shas[{_mo}]")
        for _s2, (_f2, _h2) in (_es.get("fold_members") or {}).items():
            if _h2 == "PENDING":
                _pending.append(f"extension_by_served_month[{_sm}].fold_members[{_s2}]")
    if _spec.get("arm") is not None and arm != _spec["arm"]:
        raise SystemExit(
            f"base set {base_set!r} is served by arm {_spec['arm']!r} and this run was given "
            f"{arm!r}. The arm carries the capacity setting AND the tail guard, so serving "
            f"the wrong one silently produces a different composition -- v10's first serve "
            f"came out with tail_guard null and no rows returned to the base")
    if _spec.get("ungated") is not None and bool(_spec["ungated"]) != bool(ungated):
        raise SystemExit(
            f"base set {base_set!r} declares ungated={_spec['ungated']} and this run was "
            f"given ungated={ungated}. Section 47.22.4: the serve and its class vector must "
            f"agree on this flag, and the mismatch is silent in the output -- it shows up "
            f"only as a class short by the mode-4 rows")
    if _spec.get("plan_only") and not plan:
        raise SystemExit(
            f"base set {base_set!r} is PLAN ONLY (section 47.49 (c)): it stands a placeholder "
            f"month set where a real one will go, so a candidate built from it would carry a "
            f"head fitted on the wrong months. Run it with --plan, or use the real base set")
    if _pending:
        raise SystemExit(
            f"base set {base_set!r} still carries PENDING shas ({len(_pending)}: "
            f"{', '.join(_pending[:4])}{'...' if len(_pending) > 4 else ''}). The arm it "
            f"serves has not been fitted and verified yet; fill them from the bag's JSON "
            f"before serving. This is fail-closed on purpose: a candidate must never be "
            f"built against an unverified arm")
    bs = BASE_SETS[base_set]
    for role in ("bytes", "price", "head_file", "head_npz", "arm_npz"):
        f = bs.get(role)
        if f is None:
            continue
        want = bs[{"bytes": "bytes_sha", "price": "price_sha",
                   "head_file": "head_sha", "head_npz": "head_npz_sha",
                   "arm_npz": "arm_npz_sha"}[role]]
        got = D.sha256(f)
        if got != want:
            raise SystemExit(f"{f.name} is {got[:12]}…, not the {role} file base set "
                             f"{base_set!r} serves on ({want[:12]}…)")
    bytes_s = pd.read_parquet(bs["bytes"]).set_index("MVT_ID_mvt")[D.TARGET]
    base = pd.read_parquet(bs["price"]).set_index("MVT_ID_mvt")[D.TARGET]

    # BEFORE the first panel: the patch has to be in place for the serving fit, the
    # increment gate, the guard's re-derivation and the per-airport RMS alike
    ungated_folds = install_ungated_fold_side(A.serving_rule()[0]) if _UNGATED else None

    pan = A.panel_with_block(BA.load())
    fit = pd.concat([pan[f][pan[f].usable] for f in A.FOLDS], ignore_index=True)
    if not len(fit):
        raise SystemExit("no fitting row carries an observation; the harvest has not "
                         "delivered a fold month yet")

    dep, rblk = ranking_block()
    dep["pbase"] = dep.MVT_ID_mvt.map(base).to_numpy("float64")
    if not np.isfinite(dep.pbase).all():
        raise SystemExit("a ranking departure has no base prediction")

    obs_cols = [c for c in B.FEATURES if c not in ("stand_lat", "stand_lon")]
    # a capacity variant is the same head at a different size: strip the suffix, carry the
    # override, and everything below is unchanged
    # the FULL arm name is kept -- every gate, every table lookup and every saved vector
    # is keyed on it -- and only the construction dispatch uses the head
    lgb_params = capacity_params(arm)
    arm_head = arm.split("@", 1)[0]
    # **A DECLARED BAG MUST BE HONOURED OR THE RUN MUST STOP.**  The bag branch lives in
    # the `else` of the dispatch below, so a base set that declares `bag` and is served
    # with the DEFAULT arm silently writes an ordinary single fit -- the v22 dry run of
    # 2026-09-09 did exactly that, produced a vector 196,251 rows from v7, and looked like
    # a serve.  This is the third spec on this lane to be ignored by silence rather than
    # by error (the v20 guard flags, section 35.4; `fold_gate`'s `if held:`, section 35.5),
    # so it is fatal here rather than a comment somewhere.
    ext_prov: list = []
    ext = ext_scols = None
    ext_by_served: dict = {}
    if (bs.get("bag") or bs.get("bag_by_served_month")) \
            and arm_head in ("a_linear", "a_linear_reach", "b_lgb"):
        raise SystemExit(
            f"base set {base_set!r} declares a bag but the arm is "
            f"{arm!r}, whose branch cannot honour it -- the run would write an ordinary "
            f"single fit and call it the bag. Pass the shipped arm, e.g. "
            f"{bs.get('head_arm', 'c_lgb_shipped@...')!r}, or remove the bag spec.")
    if arm_head == "a_linear":
        d, info = A.arm_a(fit, dep, ["airport"])
    elif arm_head == "a_linear_reach":
        d, info = A.arm_a(fit, dep, ["airport", "obs_wide"])
    elif arm_head == "b_lgb":
        d, info = A.arm_lgb(fit, dep, obs_cols, params=lgb_params)
    else:
        zs = _load("exp-044-zerosum", "zs")
        Wr = pd.concat([zs.shipped(m, drop_cache_vocab=True) for m in RANK_MONTHS])
        s2 = dep.join(Wr.reindex(dep.MVT_ID_mvt.to_numpy("float64")).reset_index(drop=True))

        # the extension rows, built ONCE and before anything perturbs `A.month_block`:
        # `onemonth.nov_rows` builds its own month, and the bag's monkeypatch resolves the
        # three fold months only, so building it inside the member loop would raise
        if bs.get("extension_by_served_month"):
            # SECTION 47.49: one extension per served month. Both are built HERE, before
            # anything perturbs `A.month_block`, for the same reason the single one is.
            for _sm, _es in bs["extension_by_served_month"].items():
                ext_by_served[_sm] = extension_rows(_es, A.shipped_cols(pan)[1])
            if plan:
                _mo = dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m").to_numpy()
                _use = dep.usable.to_numpy(bool)
                _plan = {
                    "what": ("section 47.49 --plan: the seasonal serve's MECHANICS, with no "
                             "fit. Row counts, cache keys and the composition census, so the "
                             "path is exercised before the months exist"),
                    "base_set": base_set, "arm": arm, "ungated": bool(ungated),
                    "ranking_rows": int(len(dep)),
                    "by_served_month": {
                        s: {"months": bs["extension_by_served_month"][s]["months"],
                            "extension_rows_built": int(len(ext_by_served[s][0])),
                            "shipped_columns": len(ext_by_served[s][1]),
                            "ranking_cache_key":
                                bs["bag_by_served_month"][s]["cache"],
                            "ranking_cache_keys_that_would_be_written":
                                [bs["bag_by_served_month"][s]["cache"].format(seed=k)
                                 for k in bs["bag_by_served_month"][s]["seeds"]],
                            "rows_this_head_would_write": int((_mo == s).sum()),
                            "of_those_in_the_class": int(((_mo == s) & _use).sum())}
                        for s in bs["extension_by_served_month"]},
                }
                _plan["composition_covers_every_ranking_row"] = (
                    sum(v["rows_this_head_would_write"]
                        for v in _plan["by_served_month"].values()) == len(dep))
                _plan["cache_keys_are_distinct"] = len({
                    bs["bag_by_served_month"][s]["cache"]
                    for s in bs["bag_by_served_month"]}) == len(bs["bag_by_served_month"])
                _out = EXP / f"analysis/serve-{base_set}-plan.json"
                _out.write_text(json.dumps(_plan, indent=2, sort_keys=True, default=float))
                print(json.dumps(_plan, indent=2, sort_keys=True, default=float))
                print(f"\nwritten {_out.relative_to(ROOT)} -- PLAN ONLY, nothing was fitted "
                      f"and nothing was written to a candidate path", flush=True)
                return
        elif bs.get("extension"):
            ext, ext_scols = extension_rows(bs["extension"], A.shipped_cols(pan)[1])

        def fit_on(panel, ext=None, ext_scols=None, ext_prov=None):
            """the ranking-side arm fitted on `panel`'s usable rows.

            Lifted out of the line below UNCHANGED so the single-fit path is what it
            always was, and so a bag member is the same construction on a perturbed
            fit-panel rather than a second implementation of it.

            **The extension, when the base set declares one.**  A base set whose head was
            priced on the folds with an extension month in the fitting set must SHIP an
            arm fitted the same way: the fold price describes the arm that ships, or it
            describes nothing.  The extra rows are appended AFTER the shipped join, for
            the reason `chainfold.fit_arm` records -- appended before it, they carry NaN
            for all 76 shipped features and the tree learns the month instead of the
            signal.
            """
            ft = pd.concat([panel[f][panel[f].usable] for f in A.FOLDS], ignore_index=True)
            ship, scols = A.shipped_cols(panel)
            W = pd.concat([ship[f] for f in A.FOLDS])
            f2 = ft.join(W.reindex(ft.MVT_ID_mvt.to_numpy("float64"))
                         .reset_index(drop=True))
            if ext is not None:
                # the extension was built against the shipped columns of the UNPERTURBED
                # panel; if a perturbed panel resolves a different set, `reindex` below
                # would fill the difference with NaN in silence
                if list(scols) != list(ext_scols):
                    raise SystemExit(
                        f"the perturbed panel resolves {len(scols)} shipped columns, not "
                        f"the {len(ext_scols)} the extension rows were built against -- "
                        f"the extension would be fitted with NaN features")
                used = set(list(obs_cols) + list(scols)) | {"rbase", "day", "airport"}
                need = sorted(c for c in used if c not in ext.columns)
                if need:
                    raise SystemExit(
                        f"the extension rows are missing {len(need)} column(s) the fit "
                        f"USES: {need[:8]}")
                n_before = len(f2)
                f2 = pd.concat([f2, ext.reindex(columns=f2.columns)], ignore_index=True)
                if len(f2) <= n_before:
                    raise SystemExit("the extension added no row to the ranking fit")
                ext_prov.append({"fold_rows": int(n_before), "extra_rows": int(len(ext)),
                                 "total_fit_rows": int(len(f2)),
                                 "multiple": round(len(f2) / n_before, 4)})
            return A.arm_lgb(f2, s2, obs_cols + scols, params=lgb_params)

        if bs.get("bag_by_served_month"):
            # SECTION 47.49: TWO bags, composed by served month. Each served month's rows
            # are written by the head fitted on THAT month's month set, and the composition
            # is asserted to cover every ranking row exactly once.
            month_of = dep.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m").to_numpy()
            declared = list(bs["bag_by_served_month"])
            unserved = sorted(set(month_of.tolist()) - set(declared))
            if unserved:
                raise SystemExit(
                    f"the ranking set carries month(s) {unserved} that no head is declared "
                    f"for; their rows would be written by a head fitted on the OTHER served "
                    f"month's months, which is the whole thing this base set exists to stop")
            caches = [bs["bag_by_served_month"][s]["cache"] for s in declared]
            if len(set(caches)) != len(caches):
                raise SystemExit(
                    f"two served months share a ranking cache key {caches}. The key is "
                    f"seed+leaves and cannot tell one arm from another (section 45.2), so a "
                    f"shared key would give both served months the SAME head in silence")
            d = np.zeros(len(dep), dtype="float64")
            written = np.zeros(len(dep), dtype=bool)
            info = {"seasonal_by_served_month": {}, "composed_by": "served month"}
            _vecs = []
            for _sm in declared:
                _e, _es = ext_by_served[_sm]
                _prov: list = []
                d_sm, info_sm = bagged_ranking_update(
                    bs["bag_by_served_month"][_sm], A.serving_rule()[0],
                    lambda panel, _e=_e, _es=_es, _p=_prov: fit_on(panel, _e, _es, _p),
                    len(dep), dep.usable.to_numpy(bool))
                sel = (month_of == _sm)
                d_sm = np.asarray(d_sm, dtype="float64")
                _vecs.append(d_sm)
                d[sel] = d_sm[sel]
                written |= sel
                info_sm["extension_per_fit"] = _prov
                info_sm["months"] = bs["extension_by_served_month"][_sm]["months"]
                info_sm["rows_this_head_writes"] = int(sel.sum())
                info_sm["rows_this_head_writes_in_the_class"] = int(
                    (sel & dep.usable.to_numpy(bool)).sum())
                info["seasonal_by_served_month"][_sm] = info_sm
                ext_prov.extend(_prov)
            if not written.all():
                raise SystemExit(f"{int((~written).sum())} ranking rows were written by no "
                                 f"head -- the composition does not cover the set")
            if len(_vecs) > 1 and all(np.array_equal(_vecs[0], v) for v in _vecs[1:]):
                raise SystemExit(
                    "every served month's head is the SAME vector. A seasonal head that "
                    "composes one vector with itself is not a seasonal head, and the record "
                    "would claim one. Check that the two cache keys did not collide and that "
                    "the two month sets really differ")
            info["heads_differ_on_the_ranking_side"] = True
            info["head_disagreement_rms_s"] = float(
                np.sqrt(np.mean((_vecs[0] - _vecs[1]) ** 2))) if len(_vecs) == 2 else None
            # SECTION 47.54, queue's negative control.  "The heads differ" is a BINARY claim
            # and an auditor cannot reproduce a boolean.  What it can reproduce is the
            # separation ON THE ROWS EACH HEAD WROTE: rebuild the other set's mean, restrict
            # to this served month, and the distance must be this.  A collision of the two
            # caches, or two month sets that are secretly the same, both drive this to zero,
            # so the number is the control and the boolean above is only its shadow.
            for _i, _sm in enumerate(declared):
                sel = (month_of == _sm)
                _here = _vecs[_i][sel]
                _others = {declared[_j]: _vecs[_j][sel]
                           for _j in range(len(declared)) if _j != _i}
                info["seasonal_by_served_month"][_sm]["separation_on_its_own_rows"] = {
                    _o: {"rows": int(sel.sum()),
                         "rms_s": float(np.sqrt(np.mean((_here - _v) ** 2))),
                         "max_abs_s": float(np.max(np.abs(_here - _v))),
                         "rows_that_differ": int((_here != _v).sum()),
                         "what": (f"on the {int(sel.sum()):,} rows the {_sm} head wrote, the "
                                  f"distance to what the {_o} head would have written there. "
                                  f"This is queue's negative control as a NUMBER: rebuild "
                                  f"the {_o} set's mean from its ten members, restrict to "
                                  f"{_sm}, and the candidate must sit this far from it")}
                    for _o, _v in _others.items()}
        elif bs.get("bag"):
            d, info = bagged_ranking_update(bs["bag"], A.serving_rule()[0],
                                            lambda panel: fit_on(panel, ext, ext_scols,
                                                                 ext_prov),
                                            len(dep), dep.usable.to_numpy(bool))
        else:
            d, info = fit_on(pan, ext, ext_scols, ext_prov)

    # --- the structural holds, checked and not assumed ----------------------
    assert not np.any(d[dep.noac.to_numpy(bool)]), "a noac row moved"
    assert not np.any(d[~dep.usable.to_numpy(bool)]), \
        "a row outside the with-clock class moved"

    # the serving floor is applied ONLY where the arm moves the row, so an unmatched or
    # `noac` row stays bit-identical to the base whatever its own prediction is -- and it is
    # the same floor `arms.apply_floor` prices on the fold side (Sol round 1, blocking 10)
    # --- the airport hold, applied BEFORE the gates and before the floor --------
    # The assignment's repair clause: "if an airport's observation does not carry
    # (receipt, box, or clock offset), say which and exclude it by a rule fitted off the
    # scored rows, not by hand".  A held airport's rows stay bit-identical to the base.
    # The choice is made on LABEL-FREE evidence and on the fold side (`code/lfpgwhy.py`,
    # `analysis/lfpgwhy.json`), never on a board reading (score-feedback rule), and it is recorded here
    # with the fold-side cost of holding.
    # **The tail guard, applied to the served vector from the frozen definition.**  The
    # cut is the one `guardarm.py` registered from the FITTING months -- read from the
    # file, never recomputed here, so the vector that ships is masked by exactly the
    # number the fold side was priced with.
    guard_rows, guard_cut, guard_cut_in_file = 0, None, None
    guard_mask = np.zeros(len(dep), bool)
    if G.NAME in arm:
        gj = json.loads((EXP / "analysis/guard-arms.json").read_text())
        guard_cut = float(gj["guard"]["cut_s_for_the_ranking_serve"])
        if gj["guard"]["quantile"] != G.Q:
            raise SystemExit("analysis/guard-arms.json was priced at a different "
                             f"quantile ({gj['guard']['quantile']}) than code/guard.py "
                             f"freezes ({G.Q})")
        # RE-DERIVE IT (round 5 FALLBACK, finding 10).  `guard-arms.json` is not hashed
        # anywhere, so until now the only thing standing between the pricing and the serve
        # was that the quantile matched.  The pinned panel is already loaded, and the cut
        # is a quantile of it, so recompute and compare: nothing can move the number
        # between the two sides without this stopping.  The tolerance is the 0.01 s the
        # file itself is rounded to (`guardarm.py`).
        recut = G.cut(A.panel_with_block(BA.load()), list(A.FOLDS))
        guard_cut_in_file = guard_cut
        if _UNGATED:
            # **THE CUT IS A QUANTILE OF THE BLOCK'S OWN TAIL, so a class change moves
            # it.**  `analysis/guard-arms.json` holds the cut of the SHIPPED blocks
            # (1,870.64 s); the bundle's class carries 2,497 more fold rows and 2,493 more
            # ranking rows, and their gaps are in the quantile.  `code/chainfold.py`
            # re-derived it exactly this way when the bundle was priced (`fit_arm`:
            # `G.cut(pan, the other two folds)` per scored fold), so the SERVING cut is the
            # same quantile of the same panel over all three folds -- taken from the panel
            # that ships and never from a file that describes a different matcher.  The
            # quantile itself is still `code/guard.py`'s frozen 99.5 and is checked above;
            # what moves is the number that quantile lands on.  Both go in the record.
            guard_cut = recut
        elif abs(recut - guard_cut) > 0.01:
            raise SystemExit(
                f"the guard cut in analysis/guard-arms.json is {guard_cut:.2f} s but the "
                f"pinned panel gives {recut:.4f} s; the file and the folds disagree")
        gm = G.mask(dep.pbase.to_numpy("float64"), dep.obs_taxi, guard_cut)
        guard_mask = gm
        guard_rows = int((d[gm] != 0).sum())
        d = d.copy()
        d[gm] = 0.0

    held = sorted(hold_airports)
    hold_rows = 0
    if held:
        hm = np.isin(dep.airport.to_numpy().astype(str), held)
        hold_rows = int((d[hm] != 0).sum())
        d = d.copy()
        d[hm] = 0.0

    pbasev = dep.pbase.to_numpy("float64")
    d = A.apply_floor(pbasev, d)
    # **The served class takes the head's own prediction; every other row keeps the bytes
    # file.**  Serving on v12 the two are the same vector and this is `pbase + d`.  On v14
    # they are not: v14's with-clock rows already carry v13's update, so a served row takes
    # `p_v12 + d_new` -- one head on the class, not two -- and a row this lane does not
    # serve is copied from v14 byte for byte, which is what keeps queue's no-clock update
    # and the 388 `noac` rows intact.  A HELD airport is not served, so it keeps v14's
    # bytes too (and v14's LFPG with-clock rows are v12's, because v13 held them).
    bytesv = dep.MVT_ID_mvt.map(bytes_s).to_numpy("float64")
    if not np.isfinite(bytesv).all():
        raise SystemExit("a ranking departure has no prediction in the bytes file")
    serve_mask = dep.usable.to_numpy(bool) & ~dep.noac.to_numpy(bool)
    if held:
        serve_mask &= ~np.isin(dep.airport.to_numpy().astype(str), held)
    pred = bytesv.copy()
    pred[serve_mask] = (pbasev + d)[serve_mask]

    # **A row the head served, this serve does not, and the bytes file did not move
    # either, goes back to the price base.**  (The desk, 2026-09-09.)  Round 3 finding 1
    # changed the class: the whole-file block called some rows with-clock that the
    # per-month block does not.  On such a row v14's bytes are v13's OLD update, and no
    # new head lands there -- `p_v12 + d_new` is only written inside `serve_mask` -- so
    # the file would ship a stale head on a row that NEITHER lane serves.  The honest
    # value there is the price base.  Measured, never assumed: the head's class is read
    # off the head file (`headv != pbasev`) and what the bytes file changed on top of it
    # is read off the bytes file (`bytesv != headv`), so a row queue moved keeps queue's
    # update and only the untouched remainder reverts.  A HELD airport is in this serve's
    # class, so it is never reverted; it keeps the bytes.
    reverted = np.zeros(len(dep), bool)
    occupied_unwritten = {"rows": 0, "what": "no head class vector on this base set"}
    head_occupied = None
    if bs["head_file"] is not None:
        head_s = pd.read_parquet(bs["head_file"]).set_index("MVT_ID_mvt")[D.TARGET]
        headv = dep.MVT_ID_mvt.map(head_s).to_numpy("float64")
        if not np.isfinite(headv).all():
            raise SystemExit("a ranking departure has no prediction in the head file")
        in_class = dep.usable.to_numpy(bool) & ~dep.noac.to_numpy(bool)
        # the head's OWN class, read from the vector the head was published with and
        # checked against the sha the base set pins -- never inferred from the prediction
        spec = bs.get("head_class")
        if spec is None:
            raise SystemExit(f"base set {base_set!r} names a head but no head_class; the "
                             "revert rule cannot tell this lane's head from another "
                             "lane's update carried in the bytes")
        parts, gparts, no_guard_column = [], [], False
        for m, (rel, want) in sorted(spec["months"].items()):
            f = ROOT / rel
            got = D.sha256(f)
            if got != want:
                raise SystemExit(f"{rel} is {got[:12]}…, not the head class vector base "
                                 f"set {base_set!r} was composed on ({want[:12]}…)")
            parts.append(pd.read_parquet(f, columns=["MVT_ID_mvt", spec["column"]]))
            gcol0 = spec.get("guard_column", "guard_held_by_exp_046")
            try:
                gparts.append(pd.read_parquet(f, columns=["MVT_ID_mvt", gcol0]))
            except (KeyError, ValueError, OSError):
                # a vector written before the column existed: NO month gets an exemption,
                # so the assertion reads exactly as it did rather than half-applying
                no_guard_column = True
        hc = pd.concat(parts, ignore_index=True).set_index("MVT_ID_mvt")[spec["column"]]
        head_occupied = dep.MVT_ID_mvt.map(hc).to_numpy()
        if pd.isna(head_occupied).any():
            raise SystemExit("a ranking departure is absent from the head class vector")
        head_occupied = head_occupied.astype(bool)
        # A ROW THE HEAD OCCUPIED HAS THREE STATES, not two (round 5 FALLBACK, finding 6).
        # It normally differs from the price base, and a row that does not is a vector
        # disagreeing with its file.  But a GUARDED head occupies rows the tail guard
        # returned to the price base -- 1,481 of them on v16 -- and those are equal to it
        # by construction.  The exemption is DECLARED in the head's own class vector
        # (`guard_held_by_exp_046`, written by `classvec.py` from the guard's frozen cut
        # and checked there against the file in both directions), never inferred here from
        # the values, so the assertion keeps its force instead of being widened into a
        # hole.  A head vector written before the column existed carries no exemption and
        # reads exactly as it did.
        gcol = spec.get("guard_column", "guard_held_by_exp_046")
        head_guard_held = np.zeros(len(dep), bool)
        if gparts and not no_guard_column:
            hg = pd.concat(gparts, ignore_index=True).set_index("MVT_ID_mvt")[gcol]
            mapped = dep.MVT_ID_mvt.map(hg)
            if mapped.isna().any():
                raise SystemExit("a ranking departure is absent from the head guard column")
            head_guard_held = mapped.to_numpy().astype(bool)
        # **OCCUPIED, UNWRITTEN** (desk, 2026-09-09).  The third state the coordinated
        # rebuild put on the table: a row the CURRENT class occupies, that the head's bytes
        # leave at the price base because it was not in the class when they were written.
        # It is neither a guard hold nor a vector disagreeing with its file, and it is this
        # lane's to write at this serve.  Counted and reported, never a stop; the assertion
        # keeps its force over every row for which the head vector claims a written value.
        unwritten = head_occupied & (headv == pbasev) & ~head_guard_held
        occupied_unwritten = {
            "what": ("rows this lane's CURRENT class occupies that the head's bytes leave "
                     "at the price base: the class moved under bytes already written, so "
                     "this lane writes them here for the first time"),
            "rows": int(unwritten.sum()),
            "by_airport": {str(a): int(((dep.airport.to_numpy() == a) & unwritten).sum())
                           for a in sorted(set(dep.airport.to_numpy()[unwritten]))},
            "by_month": {str(m): int(((dep.rank_month.to_numpy() == m) & unwritten).sum())
                         for m in sorted(set(dep.rank_month.to_numpy()[unwritten]))},
            "column_read": spec["column"],
            "narrower_column_the_desk_verifies_on": "written_by_exp_046",
        }
        bad = int((unwritten & ~in_class).sum())
        if bad:
            raise SystemExit(f"{bad} rows the head class vector calls occupied are equal "
                             "to the price base in the head file, are not declared "
                             f"guard-held in {gcol}, and are NOT in this serve's class "
                             "either -- so no lane writes them and the composition would "
                             "strand them")
        # and the other direction: a declared guard-held row must actually BE at the base
        bad_g = int((head_guard_held & (headv != pbasev)).sum())
        if bad_g:
            raise SystemExit(f"{bad_g} rows the head class vector declares guard-held are "
                             "not equal to the price base in the head file")
        reverted = head_occupied & ~in_class & (bytesv == headv)
        pred[reverted] = pbasev[reverted]

    # what this upload CHANGES is measured against the bytes file, not against v12
    d_over_bytes = pred - bytesv

    # --- BOTH gates, computed from the update vector, BEFORE any write ------
    # The RMS gate used to be computed from the read-back of the written file, so an
    # exception between the write and the move could leave an ungated file on the
    # candidate path (round 2 FALLBACK, blocking 2).  `d` is the update and `pred` is
    # exactly what will be written, so nothing here needs a file to exist.
    fold = fold_gate(arm, held, npz=bs.get("arm_npz"))
    inc = ({"skipped": "this base set has no served head under it"}
           if not bs["head_arm"] else
           increment_gate(arm, bs["head_arm"], held, head_npz=bs.get("head_npz"),
                          unselected=bool(bs.get("unselected")),
                          new_npz=bs.get("arm_npz"),
                          lit_rule=bool(bs.get("lit_rule", False)),
                          head_npz_note=bs.get("head_npz_note")))
    fold_rms = fold_update_rms(arm, pan, npz=bs.get("arm_npz"))
    by_air = {}
    air = dep.airport.to_numpy()
    for ap in sorted(set(air)):
        sel = air == ap
        m = sel & (d != 0)
        served = float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else 0.0
        ref = fold_rms.get(str(ap))
        by_air[str(ap)] = {
            "rows": int(sel.sum()), "moved": int(m.sum()),
            "moved_share": round(float(m.sum() / max(sel.sum(), 1)), 4),
            "rms_update_s": served, "fold_rms_update_s": ref,
            "ratio_to_fold": (served / ref if ref else None),
            "within_a_third": (ref is not None and ref > 0
                               and RMS_BAND[0] <= served / ref <= RMS_BAND[1])}
    rms_ok = all(v["within_a_third"] for v in by_air.values() if v["moved"])
    # The assignment's clause is "within a third ... ELSE EXPLAIN before the desk uploads",
    # so a breach is an argued exception and not a silent widening of the band.  The band
    # is NOT relaxed: `rms_within_a_third_everywhere` still reads False and every airport's
    # ratio stays in the report.  What `--accept-rms-breach` does is record WHICH airports
    # the desk accepted, and it fails closed if any OTHER airport is out of band, so an
    # exception argued for one airport can never cover a second one that appears later.
    out_of_band = sorted(a for a, v in by_air.items()
                         if v["moved"] and not v["within_a_third"])
    accepted = sorted(accept_rms_breach)
    breach_covered = bool(accepted) and set(out_of_band) <= set(accepted)
    inc_ok = bool(inc.get("passes", True)) if bs["head_arm"] else True
    is_candidate = bool(fold["passes"] and inc_ok
                        and (rms_ok or breach_covered) and not dry_run)

    # --- one write, to the path the gates chose -----------------------------
    dest = bs["out"] if is_candidate else DRY
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = pd.Series(pred, index=dep.MVT_ID_mvt.to_numpy("float64"))

    # **THE SIBLING ASSERTION** (desk, 2026-09-09, for v20).  When a base set declares a
    # sibling, this candidate is the sibling PLUS one named change, and the claim is
    # checkable rather than argued: every row that is not at one of the named airports
    # must come back byte for byte.  It runs BEFORE the write and on a dry run too, so a
    # serve that quietly moved a row it did not mean to stops here with nothing written.
    # For v20 the sibling is v5 and the only airport allowed to move is LFPG: the serving
    # fit is deterministic, so lifting the airport hold can change LFPG and nothing else.
    # A base set may instead (or also) name airports that must be IDENTICAL, which is the
    # reading v21 needs: it replaces the head over the whole class, so the claim worth
    # checking is not what moved but that v6's LFPG rider SURVIVED the replacement.
    sib = bs.get("sibling")
    sibling_check = None
    if sib:
        sf = Path(sib["file"])
        got_sib = D.sha256(sf)
        if got_sib != sib["sha"]:
            raise SystemExit(f"the sibling {sf.name} is {got_sib[:12]}…, not the "
                             f"{sib['sha'][:12]}… this base set names")
        sv = pd.read_parquet(sf).set_index("MVT_ID_mvt")[D.TARGET]
        sv = sv.reindex(dep.MVT_ID_mvt.to_numpy("float64")).to_numpy("float64")
        if not np.isfinite(sv).all():
            raise SystemExit("a ranking departure has no value in the sibling file")
        differs = pred != sv
        allowed_ap = np.isin(dep.airport.to_numpy().astype(str),
                             list(sib["may_differ_only_at_airports"]))
        outside = int((differs & ~allowed_ap).sum())
        frozen_ap = np.isin(dep.airport.to_numpy().astype(str),
                            list(sib.get("must_be_identical_at_airports", [])))
        moved_frozen = int((differs & frozen_ap).sum())
        if moved_frozen:
            raise SystemExit(
                f"{moved_frozen} rows differ from the sibling {sf.name} at "
                f"{sib['must_be_identical_at_airports']}, which this base set says must "
                "come back byte for byte; nothing was written.")
        if outside:
            bad = sorted(set(dep.airport.to_numpy().astype(str)[differs & ~allowed_ap]))
            raise SystemExit(
                f"{outside} rows differ from the sibling {sf.name} outside "
                f"{sib['may_differ_only_at_airports']} (at {bad[:8]}); nothing was "
                "written. The serving fit is supposed to be deterministic, so this is a "
                "real difference and has to be understood before any bytes go out.")
        sibling_check = {
            "file": str(sf.relative_to(ROOT)), "sha256": got_sib,
            "may_differ_only_at_airports": list(sib["may_differ_only_at_airports"]),
            "rows_differing_from_the_sibling": int(differs.sum()),
            "rows_differing_outside_those_airports": outside,
            "rows_differing_at_those_airports": int((differs & allowed_ap).sum()),
            "rows_identical": int((~differs).sum()),
            "must_be_identical_at_airports": list(
                sib.get("must_be_identical_at_airports", [])),
            "rows_at_the_frozen_airports": int(frozen_ap.sum()),
            "rows_differing_at_the_frozen_airports": moved_frozen,
            "what": ("this candidate is the sibling plus one named change, checked row by "
                     "row before the write rather than described"),
        }
        print(f"sibling {sf.name}: {int(differs.sum())} rows differ, "
              f"{outside} of them outside "
              f"{sib['may_differ_only_at_airports']}", flush=True)

    # **A CANDIDATE PATH THAT ALREADY CARRIES BYTES IS NEVER OVERWRITTEN IN PLACE.**
    # Round 7 ordered a RE-EMIT of the record with the gates read on the right vector, and
    # a re-emit that quietly rewrote the reviewed file would put a second set of bytes
    # behind a reviewed sha.  So the write goes to a temporary sibling first and the file
    # on disk is replaced only when it does not exist yet; when it does, the two shas must
    # be equal and the existing bytes are left exactly as they are.  A difference is a stop
    # and a report, never a silent overwrite -- which is also the standing rule that keeps
    # v3 and v4 pinned.
    existing_sha = D.sha256(dest) if (dest != DRY and dest.exists()) else None
    if existing_sha is None:
        sha = D.write_submission(out, dest)
        bytes_action = "written; the path carried no bytes"
    else:
        tmp = dest.with_name(dest.name + ".reemit.tmp")
        sha = D.write_submission(out, tmp)
        if sha != existing_sha:
            tmp.unlink(missing_ok=True)
            raise SystemExit(
                f"{dest.name} on disk is {existing_sha[:12]}… and this run computes "
                f"{sha[:12]}…; the bytes are NOT the reviewed ones. Nothing was written "
                "and the file on disk is untouched. Report the difference; do not "
                "overwrite a candidate that has been reviewed or uploaded.")
        tmp.unlink(missing_ok=True)
        bytes_action = ("unchanged; the file on disk already carries these exact bytes "
                        "and was not touched")
    back = pd.read_parquet(dest).set_index("MVT_ID_mvt")[D.TARGET]
    round_trip = bool(np.array_equal(
        back.reindex(dep.MVT_ID_mvt.to_numpy("float64")).to_numpy("float64"), pred))

    moved = d_over_bytes != 0
    _m4 = (dep.match_mode.fillna(-1).to_numpy("float64") == 4.0) if _UNGATED else None
    if _UNGATED:
        # **THE UNGATED POOL AS THE RANKING BLOCK SEES IT**, row by row.  `restrict` blanks
        # a refused pair's features but leaves its three check values, so this frame carries
        # both the kept rows and the refused ones and the count can be reconciled against
        # the pool `code/chaincheck.py` reconstructed (2,493 at this gate) without a second
        # build of the ranking block.
        _pool = dep[np.isfinite(dep.chain_gap_s.to_numpy("float64"))
                    | np.isfinite(dep.chain_abs_dt_offblock.to_numpy("float64"))]
        _pool[["MVT_ID_mvt", "rank_month", "airport", "match_mode", "chain_gap_s",
               "chain_abs_dt_offblock", "chain_runway_agrees"]].reset_index(
            drop=True).to_parquet(EXP / f"work/ranking-ungated-pool-{base_set}.parquet",
                                  index=False)
    rep = {
        "arm": arm, "is_candidate": is_candidate, "base_set": base_set,
        # WHAT THE HEAD WAS FITTED ON.  The audit reads the bytes, the class and the
        # increment; it cannot see the fitting set, and on this base set the fitting set is
        # the entire difference from v8.  `per_fit` carries one entry per fit that actually
        # ran, so a run that served ten cached un-extended members writes an EMPTY list
        # here and says so instead of looking like an extended serve.
        "extension": (
            # SECTION 47.49: a seasonal serve has one extension PER SERVED MONTH, and the
            # report names which months each head was fitted on and how many ranking rows
            # that head wrote -- the two facts an audit of the BYTES cannot recover
            {"by_served_month": {
                _sm: {"declared": {k: v for k, v in
                                   bs["extension_by_served_month"][_sm].items()},
                      "rows_built": int(len(_e)),
                      "shipped_columns_at_build": len(_es),
                      "per_fit": info.get("seasonal_by_served_month", {})
                                     .get(_sm, {}).get("extension_per_fit", []),
                      "rows_this_head_writes":
                          info.get("seasonal_by_served_month", {})
                              .get(_sm, {}).get("rows_this_head_writes")}
                for _sm, (_e, _es) in ext_by_served.items()},
             "composed_by": "served month",
             "fits_that_carried_it": len(ext_prov),
             "note": ("a member read from cache did not refit, so it contributes no entry "
                      "to per_fit; its extension is the one it was fitted with, which the "
                      "cache NAME keys on")}
            if ext_by_served else
            None if ext is None else {
            "declared": {k: v for k, v in bs["extension"].items()},
            "rows_built": int(len(ext)),
            "shipped_columns_at_build": len(ext_scols),
            "per_fit": ext_prov,
            "fits_that_carried_it": len(ext_prov),
            "note": ("a member read from cache did not refit, so it contributes no entry "
                     "to per_fit; its extension is the one it was fitted with, which the "
                     "cache NAME keys on")}),
        "written_to": str(dest.relative_to(ROOT)), "sha256": sha,
        "bytes_action": bytes_action,
        "sibling_check": sibling_check,
        "bytes_already_on_disk_sha256": existing_sha,
        "read_back_identical_to_the_vector_priced": round_trip,
        "base": {
            "set": base_set, "what": bs["what"],
            "bytes_file": str(bs["bytes"].relative_to(ROOT)),
            "bytes_sha256": bs["bytes_sha"],
            "price_file": str(bs["price"].relative_to(ROOT)),
            "price_sha256": bs["price_sha"],
            "head_file": (str(bs["head_file"].relative_to(ROOT))
                          if bs["head_file"] else None),
            "head_sha256": bs["head_sha"], "head_arm": bs["head_arm"]},
        "rule": {"cover": rblk.attrs.get("rule_cover"),
                 "rules_seen": rblk.attrs.get("rules_seen"),
                 "days_wrong_rule": [f"{x[0]} {x[1]}" for x in
                                     rblk.attrs.get("days_wrong_rule", [])][:40],
                 "normalised_by_airborne_alt_ok":
                     rblk.attrs.get("normalised_by_airborne_alt_ok")},
        # **which population this file was fitted on** (desk ruling, 2026-09-09).  The
        # fold months are pinned to the airport-day set they were priced on, so a delivery
        # landing between the price and the serve cannot change what the candidate was
        # gated on; the ranking months are NOT pinned, because they are the serve's target
        # and not a priced fold table.  Stated in the report so the reviewer reads it off
        # the artifact rather than off the record.
        "population": {
            "fold_months_are_pinned_to": {
                m: {"block_cache_key": r.get("block_cache_key"),
                    "airport_days": r.get("airport_days_count"),
                    "dropped_days": r.get("dropped_days")}
                for m, r in sorted(B.pin_file().get("months", {}).items())},
            "pin_file": ("experiments/exp-046-surfacefit/analysis/fold-population.json"
                         if B.FOLD_POP.exists() else None),
            "policy": B.pin_file().get("policy"),
            "ranking_months_are_not_pinned": (
                "the 2026 blocks are rebuilt from whatever the harvest has delivered; "
                "no key gates them, which is the ruling's intent -- the served file "
                "should read everything delivered"),
            # `rule` is resolved here and not carried out of `ranking_block()`,
            # whose copy is local to it.  This line had never executed: the pin
            # ruling added `ranking_population` on 2026-09-09, AFTER the v16 serve,
            # so the v18 serve is the first to reach it and it raised NameError.
            "ranking": ranking_population(A.serving_rule()[0], is_candidate, sha),
        },
        "fit": {"rows": int(len(fit)),
                "months": sorted(set(fit.month)) if "month" in fit else None,
                "info": info},
        "census": {
            "rows": int(len(dep)),
            "moved": int(moved.sum()),
            "unchanged": int((~moved).sum()),
            "served_rows": int(serve_mask.sum()),
            "rows_the_bytes_file_keeps": int((~serve_mask).sum()),
            "every_unserved_row_bit_identical_to_the_bytes_file": bool(
                np.array_equal(pred[~serve_mask & ~reverted],
                               bytesv[~serve_mask & ~reverted])),
            "occupied_unwritten": occupied_unwritten,
            "reverted_to_the_price_base": {
                "what": ("OCCUPIED BY THE HEAD (read from the head's own class vector, "
                         "not inferred from the prediction), out of this serve's class, "
                         "and untouched by the bytes file: the stale head is dropped and "
                         "the row carries the price base"),
                "head_class_column": (bs.get("head_class") or {}).get("column"),
                "head_occupied_rows": (int(head_occupied.sum())
                                       if head_occupied is not None else None),
                "rows": int(reverted.sum()),
                "all_equal_the_price_base": bool(
                    np.array_equal(pred[reverted], pbasev[reverted])),
                "max_abs_change_off_the_bytes_s": (
                    float(np.abs(d_over_bytes[reverted]).max())
                    if reverted.any() else 0.0),
                "by_airport": {str(a): int(((air == a) & reverted).sum())
                               for a in sorted(set(air[reverted]))},
                "MVT_ID_mvt": sorted(
                    float(x) for x in dep.MVT_ID_mvt.to_numpy()[reverted]),
            },
            "rms_update_s": (float(np.sqrt(np.mean(d_over_bytes[moved] ** 2)))
                             if moved.any() else 0.0),
            "max_abs_update_s": (float(np.abs(d_over_bytes).max())
                                 if len(d_over_bytes) else 0.0),
            "rms_update_over_the_price_base_s": (
                float(np.sqrt(np.mean(d[d != 0] ** 2))) if np.any(d != 0) else 0.0),
            "min_prediction_s": float(pred.min()),
            "finite": bool(np.isfinite(pred).all()),
            "noac_rows": int(dep.noac.sum()),
            "noac_identical_to_the_bytes_file": bool(
                not np.any(d_over_bytes[dep.noac.to_numpy(bool)])),
            "outside_the_with_clock_class_identical_to_the_bytes_file": bool(
                not np.any(d_over_bytes[~dep.usable.to_numpy(bool) & ~reverted])),
            "held_airport_rows_identical_to_the_bytes_file": bool(
                not np.any(d_over_bytes[np.isin(dep.airport.to_numpy().astype(str),
                                                held)])) if held else None,
            "handover_to_exp_047": A.handover(dep),
            "floor_bound_rows": int((pred <= MIN_PRED).sum()),
            "by_airport": by_air,
        },
        # **THE ROWS THE CHAIN ADDS**, per ranking month, so the population record says
        # what the class gained and who held those rows before it (desk, 2026-09-09:
        # "the ranking population record (rows the chain adds per ranking month, 2,493
        # split, mode-4 counts)").  Every mode-4 row still in `dep` is one the gate KEPT:
        # `chainfold.restrict` blanked the refused ones back to unmatched before the merge.
        "ungated_chain_rows": ({
            "gate": UNGATED_GATE,
            "fold_side": ungated_folds,
            "ranking_side": {
                "rows_kept_by_the_gate": int(_m4.sum()),
                "of_those_in_this_lanes_class": int((_m4 & dep.usable.to_numpy(bool)).sum()),
                "of_those_served_by_this_file": int((_m4 & serve_mask).sum()),
                "of_those_noac": int((_m4 & dep.noac.to_numpy(bool)).sum()),
                "of_those_held_at_the_base": int(
                    (_m4 & np.isin(dep.airport.to_numpy().astype(str), held)).sum())
                    if held else 0,
                "of_those_the_guard_returned_to_the_base": int(
                    (_m4 & guard_mask & serve_mask).sum()),
                "by_month": {str(m): int((_m4 & (dep.rank_month.to_numpy() == m)).sum())
                             for m in sorted(set(dep.rank_month.to_numpy()[_m4]))},
                "by_airport": {str(a): int((_m4 & (air == a)).sum())
                               for a in sorted(set(air[_m4]))},
                # **THE CLASS BOUNDARY MOVES WITH THEM** (section 14 handshake).  Under
                # the shipped matcher `122c7d84` these rows carry no clock, so they are in
                # `exp-047-queue`'s complement and queue's update is what v17's bytes hold
                # on them.  This serve writes this lane's head over that value: one head
                # per row, which is the rule.  Queue's interface
                # (`artifacts/exp-047/block/<month>.parquet`) was built on the shipped
                # matcher and does not know it, so the count is reported here and the desk
                # relays it rather than the two lanes each deriving the boundary again.
                "carrying_another_lanes_value_in_the_bytes": int(
                    (_m4 & (bytesv != pbasev)).sum()),
                "which_lane": ("exp-047-queue: these rows have no clock under the shipped "
                               "matcher 122c7d84, so they are in queue's complement and "
                               "queue's interface was built without them"),
            },
        } if _UNGATED else None),
        "gate": {
            "rms_band_on_the_ratio": list(RMS_BAND),
            "rms_within_a_third_everywhere": rms_ok,
            "airports_out_of_band": out_of_band,
            "airports_held_at_the_base": held,
            "rows_the_hold_returned_to_the_base": hold_rows,
            "tail_guard": ({"quantile": G.Q, "cut_s": guard_cut,
                            "rows_returned_to_the_base": guard_rows,
                            "frozen_by": "code/guard.py",
                            "cut_read_from": ("the pinned panel this serve ships, "
                                              "re-derived at the frozen quantile"
                                              if _UNGATED else
                                              "analysis/guard-arms.json"),
                            "cut_s_in_guard_arms_json": guard_cut_in_file,
                            "why_the_two_differ": (
                                "the cut is a quantile of the block's own tail, and the "
                                "ungated class carries rows guard-arms.json's blocks do "
                                "not; the quantile is unchanged and frozen"
                                if _UNGATED else None)}
                           if G.NAME in arm else None),
            "rms_breach_accepted_for": accepted,
            "rms_breach_covered_by_the_acceptance": breach_covered,
            "fold": fold,
            "increment_over_the_head": inc,
            "all_checked_before_the_write": True,
            # a SECONDARY GUARD and not one of the two gates: it is reported on every
            # serve and it decides which arm is chosen, never whether a gated arm ships
            "tail_clause": tail_clause(arm, held, npz=bs.get("arm_npz")),
            "arm_choice": choose_arm()[1],
        },
    }
    if not is_candidate:
        rep["why_not_a_candidate"] = (
            ([] if fold["passes"] else [fold["reason"]])
            + ([] if inc_ok else [inc.get("reason", "the increment over the served "
                                          "head does not clear zero")])
            + ([] if rms_ok else ["the served update's RMS per airport is outside "
                                  f"{list(RMS_BAND)} of the fold side's"])
            + (["--dry-run was asked for"] if dry_run else []))
    if not round_trip:
        raise SystemExit(f"{dest.name} does not read back as the vector that was priced")
    # one report per BASE SET, so a second candidate cannot overwrite the record of the
    # one already uploaded (it did once, and only git had the v13 report afterwards)
    suffix = "" if base_set == "v12" else f"-{base_set}"
    name = (f"serve{suffix}.json" if is_candidate else f"serve{suffix}-dryrun.json")
    (EXP / f"analysis/{name}").write_text(json.dumps(rep, indent=2, sort_keys=True,
                                                     default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "fit"}, indent=2,
                     sort_keys=True, default=float)[:3000])


def self_test() -> None:
    """Exercise the write, the read-back and the holds with a ZERO update.

    Round 2 (FALLBACK) observed that `serve.py` has never run past its first exit since
    the repair, because no fold month carries an observation, so its write and read-back
    code is unexercised.  This runs that code on a vector the answer is known for: with
    `d = 0` on every row the candidate must be the base's own numbers, bit for bit, and the
    file must read back as what was written.  It needs no fold month and no arm, and it
    writes to the dry-run path only.
    """
    got = D.sha256(BASE_FILE)
    if got != BASE_SHA:
        raise SystemExit(f"{BASE_FILE.name} is {got[:12]}…, not the base this serves on")
    base = pd.read_parquet(BASE_FILE).set_index("MVT_ID_mvt")[D.TARGET]
    dep, rblk = ranking_block()
    pbasev = dep.MVT_ID_mvt.map(base).to_numpy("float64")
    if not np.isfinite(pbasev).all():
        raise SystemExit("a ranking departure has no base prediction")

    d = A.apply_floor(pbasev, np.zeros(len(dep)))
    assert not np.any(d), "apply_floor moved a row whose update was zero"
    pred = pbasev + d
    DRY.parent.mkdir(parents=True, exist_ok=True)
    out = pd.Series(pred, index=dep.MVT_ID_mvt.to_numpy("float64"))
    sha = D.write_submission(out, DRY)
    back = pd.read_parquet(DRY).set_index("MVT_ID_mvt")[D.TARGET]
    aligned = back.reindex(dep.MVT_ID_mvt.to_numpy("float64")).to_numpy("float64")
    rep = {
        "what": "a zero update through the whole serving path",
        "rows": int(len(dep)),
        "read_back_identical_to_the_vector_written": bool(np.array_equal(aligned, pred)),
        "identical_to_the_base_bit_for_bit": bool(np.array_equal(aligned, pbasev)),
        "sha256_of_the_zero_update_candidate": sha,
        "sha256_of_the_base": BASE_SHA,
        "sha_matches_the_base": sha == BASE_SHA,
        "noac_rows": int(dep.noac.sum()),
        "with_clock_rows": int(dep.usable.sum()),
        "handover_to_exp_047": A.handover(dep),
        "min_prediction_s": float(pred.min()),
        "written_to": str(DRY.relative_to(ROOT)),
        "rule": {"rules_seen": rblk.attrs.get("rules_seen"),
                 "cover": rblk.attrs.get("rule_cover")},
        "gate_on_this_arm": fold_gate("a_linear"),
    }
    assert rep["read_back_identical_to_the_vector_written"], rep
    assert rep["identical_to_the_base_bit_for_bit"], rep
    (EXP / "analysis/serve-selftest.json").write_text(
        json.dumps(rep, indent=2, sort_keys=True, default=float))
    print(json.dumps(rep, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--self-test" in sys.argv:
        self_test()
    else:
        acc = tuple(a.split("=", 1)[1].split(",")
                    for a in sys.argv[1:] if a.startswith("--accept-rms-breach="))
        hold = tuple(a.split("=", 1)[1].split(",")
                     for a in sys.argv[1:] if a.startswith("--hold-airport="))
        bset = [a.split("=", 1)[1] for a in sys.argv[1:]
                if a.startswith("--base-set=")]
        main(argv[0] if argv else "a_linear", dry_run="--dry-run" in sys.argv,
             plan="--plan" in sys.argv,
             accept_rms_breach=tuple(acc[0]) if acc else (),
             hold_airports=tuple(hold[0]) if hold else (),
             base_set=bset[0] if bset else "v12",
             ungated="--ungated" in sys.argv)
