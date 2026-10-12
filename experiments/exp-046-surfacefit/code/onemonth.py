"""RESULTS section 47.17's screen: the ONE-MONTH extension (2025-11), K = 1.

Fitting rows for every fold become the current panel PLUS 2025-11's writable rows.  Recipe
unchanged in every feature and hyperparameter.  Member 1's perturbation on the fold months, so
the reading is comparable to cached member 1 row for row; 2025-11 carries no perturbation
because it is not a placebo class -- it is new rows, and that is the whole point.

Two readings, both registered:
  (i)  over CACHED MEMBER 1 -- the gate, every fold clear, in-fold, rule (a);
  (ii) over v8's K = 10 head -- record only, and expected NEGATIVE, because a K = 1 fit is
       charged for a bag it does not have (section 47.16 measured that offset at -0.28613).

**THE PRECONDITION FAILED.**  Section 47.17 registered that if the rms of
`pbase_6 - pbase_true` exceeds 25 s on any fold, the screen is run but its result is NOT
evidence about rows.  It reads 39.154 / 30.095 / 30.109 (`analysis/precond-base6.json`).  That
was written before this file existed and it is not relitigated here: this run reports a number
and the number is not evidence about rows.  The companion, restricted to the airports that do
pass, is the reading that is.

The 2025-11 row attributes are derived, and the derivation was VALIDATED on july_loo against
the fold panel before it was trusted: `airport == ADEP_mvt`, `noac == AIRCRAFT_TYPE_mvt.isna()`
(339 of 339), `t == MVT_TIME_UTC_mvt`, all exact.  That check runs again here, every time.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/onemonth.py
    ... --airports=analysis/precond-base6.json   runs the section 47.17 companion
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))

import prc.data as D                                                  # noqa: E402
import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import chainfold as CF                                                # noqa: E402
import classplacebo as CP                                             # noqa: E402
import rung511 as R                                                   # noqa: E402
import bagarm as BAG                                                  # noqa: E402
import nooverwrite as NO                                              # noqa: E402
from imports import load as _load                                     # noqa: E402

SEED_MEMBER = 1
PLACEBO_SEED = 4719          # section 47.19 check 2, fixed and stated before the run
LEAVES = 255
NOV = "2025-11"
TEN = list(range(1, 11))
OOF = ROOT / "experiments/exp-047-queue/oof"
BASE6_NOV = OOF / f"base6--{NOV}.parquet"
BASE6_NOV_SHA = "339ca16ed421a718"
#: the extension months this file can build, each pinned by the sha of queue's base6 vector
#: and by the sha of the re-emitted neighbour block.  A month with no verified block can be
#: built with `with_neighbour=False` for an ID PRE-CHECK only -- never for a fit, because the
#: twelve neighbour columns are features the arm uses.
MONTHS = {
    "2025-11": {"base6_sha16": "339ca16ed421a718", "block_sha16": "3e231a8f7cd82d83",
                "rows": 99709},
    "2025-10": {"base6_sha16": "2bfc9a6959281d6e", "block_sha16": "29932ab196080943",
                "rows": 103344},
    # SECTION 47.31.  Queue is still building it; the shas below are PENDING and no sha starts
    # with that word, so every path through `nov_rows` REFUSES this month until the desk
    # forwards the real ones.  `rows` is None because no count is registered yet: the run
    # records what it harvests, and 47.31 restates its bands BEFORE the fit if the count is
    # more than 15 % from the ~100k its two siblings gave.
    "2025-03": {"base6_sha16": "64c47603ebfdb0c7", "block_sha16": "ec26128f3a5eaa74",
                "rows": None},
    # SECTION 47.46.  Queue's 2025-02, landed 2026-09-10 05:22 UTC (queue 1cf720e): 186 columns
    # over 143,731 rows, verify 6/6, the 13 head columns PRESENT.  Both shas were checked
    # against the files on disk before this line was written.  `rows` is None for the same
    # reason 2025-03's was: the writable count is what the run HARVESTS, not what I predict.
    "2025-02": {"base6_sha16": "36f7dd9f6d80451a", "block_sha16": "32c9e25d05f6ad21",
                "rows": None},
    # SECTION 47.51.  Queue's 2025-08, landed 2026-09-10 (queue 5859d1b): 186 columns over
    # 191,181 rows, verify 6/6, the 13 head columns PRESENT.  Both shas were checked against
    # the files on disk BEFORE this line was written -- base6 191,181 rows, neighbour block
    # 191,181 rows and 0 columns missing.  `rows` is None as every month's has been: the
    # writable count is what the run HARVESTS, not what I predict.
    "2025-08": {"base6_sha16": "60c84140c3342390", "block_sha16": "137a61f23de48d9d",
                "rows": None},
    # SECTION 47.58.  Queue's 2025-06, landed 2026-09-10 10:16 UTC (queue e34ae1d): 183,142
    # rows, verify 6/6.  The desk forwarded the shas TRUNCATED to eight hex, so both were
    # recomputed from the files on disk and checked to start with what the desk sent --
    # base6 183,142 rows, neighbour block 183,142 rows and 0 of the 13 head columns missing.
    # A truncated sha is not a registered sha: what is written here is what the file hashes to.
    "2025-06": {"base6_sha16": "0bff146d3bced007", "block_sha16": "5e7e6e10898d2445",
                "rows": None},
}
CACHED_M1 = EXP / "work/d-bundle-placebo1-255-free.npz"
HEAD10 = EXP / "work/d-bag10-255-free.npz"
OCT = "2025-10"
#: section 47.23.5's measured count, the figure the re-registered band was evaluated on
OCT_ROWS = 103344
PRECOND = EXP / "analysis/precond-base6.json"
THRESHOLD_S = 25.0


def nov_rows(scols: list[str], month: str = NOV,
             with_neighbour: bool = True) -> pd.DataFrame:
    """An extension month's writable rows, target from base6 and attributes derived.

    `month` and `with_neighbour` carry DEFAULTS that reproduce the 2025-11 path exactly, so
    every fit already taken through this function is unchanged.  `with_neighbour=False`
    stops before the neighbour block and is for an ID PRE-CHECK on a month whose block is
    not verified yet: the frame it returns is missing twelve columns the arm USES and must
    never reach a fit.  `chainfold.fit_arm`'s column check is what enforces that.
    """
    if month not in MONTHS:
        raise SystemExit(f"{month} is not a registered extension month; known: {sorted(MONTHS)}")
    want6 = MONTHS[month]["base6_sha16"]
    if "PENDING" in (want6, MONTHS[month]["block_sha16"]):
        raise SystemExit(
            f"{month} is registered with a PENDING sha and MUST NOT be fitted: the desk "
            f"forwards queue's base6 and block shas, and they go into MONTHS before any fit. "
            f"Fitting on an unverified base is the one thing this registry exists to stop")
    src = OOF / f"base6--{month}.parquet"
    if not src.exists():
        raise SystemExit(f"{src} does not exist yet -- queue has not landed {month}")
    got = hashlib.sha256(src.read_bytes()).hexdigest()
    if not got.startswith(want6):
        raise SystemExit(f"base6--{month}: sha {got[:16]} != registered {want6}")
    b6 = pd.read_parquet(src)

    rule = A.serving_rule()[0]
    blk = CF.restrict(CF.build_month(month, A.prefix_months_for(month), rule),
                      R.THR, CF.GAP_CUT_S)
    mv = D.load_month(month)
    mv["MVT_ID_mvt"] = mv["MVT_ID_mvt"].astype("float64")
    m = mv.set_index("MVT_ID_mvt")

    d = b6.rename(columns={"y_pred": "pbase"})[["MVT_ID_mvt", "y_true", "pbase"]].copy()
    d["MVT_ID_mvt"] = d["MVT_ID_mvt"].astype("float64")
    idx = d.MVT_ID_mvt.to_numpy("float64")
    d["airport"] = m.ADEP_mvt.reindex(idx).astype(str).to_numpy()
    d["noac"] = m.AIRCRAFT_TYPE_mvt.reindex(idx).isna().to_numpy()
    d["t"] = pd.to_datetime(m.MVT_TIME_UTC_mvt.reindex(idx).to_numpy(),
                            utc=True).astype("int64") // 10 ** 9
    d["day"] = pd.to_datetime(d["t"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    d["month"] = month
    d["fold"] = month
    d["rbase"] = d.y_true.to_numpy("float64") - d.pbase.to_numpy("float64")
    d = d.merge(blk, on="MVT_ID_mvt", how="left", validate="one_to_one")
    d["obs_matched"] = d.obs_matched.astype("boolean").fillna(False).astype(bool)
    d["obs_has_clock"] = d.obs_has_clock.astype("boolean").fillna(False).astype(bool)
    d["reach_class"] = d.reach_class.fillna("unmatched")
    d.loc[d.noac.astype(bool), ["obs_matched", "obs_has_clock"]] = False
    d["usable"] = A.served_class(d)
    d["obs_wide"] = d.match_mode.fillna(0.0).gt(0.0)

    # The neighbourhood block: TWELVE columns, SIX of them residual-derived (`No_n_res_*`).
    # For 2025-11 they come from queue's re-emitted block, which was rebuilt against BASE6 --
    # that is exactly why §7bk re-emitted it. So these rows carry a base6 target AND
    # base6-derived residual features: one base per row, which is the invariant the desk's
    # 3ab2ef0 ruling set.
    if with_neighbour:
        d = _join_neighbour(d, month)

    zs = _load("exp-044-zerosum", "zs")
    ship = zs.shipped(month, drop_cache_vocab=True)
    miss = [c for c in scols if c not in ship.columns]
    if miss:
        raise SystemExit(f"{month}: the shipped block lacks {miss[:8]}")
    d = d.join(ship[scols].reindex(d.MVT_ID_mvt.to_numpy("float64")).reset_index(drop=True))
    return d


def _join_neighbour(d: pd.DataFrame, month: str) -> pd.DataFrame:
    """The twelve neighbourhood columns, six of them residual-derived."""
    nbr, nrep = A.neighbour_block(month)
    if nbr is None:
        raise SystemExit(f"{month}: no neighbour block -- {nrep}")
    wantb = MONTHS[month]["block_sha16"]
    if wantb is None:
        raise SystemExit(f"{month}: no neighbour-block sha is registered, so no fit may use "
                         f"it; only with_neighbour=False (an id pre-check) is allowed")
    if not nrep["sha256"].startswith(wantb):
        raise SystemExit(f"{month}: neighbour block sha {nrep['sha256'][:16]} is not the "
                         f"re-emitted {wantb} the desk named")
    if nrep["columns_missing"]:
        raise SystemExit(f"{month}: neighbour block lacks {nrep['columns_missing']}")
    return d.merge(nbr, on="MVT_ID_mvt", how="left", validate="one_to_one")


def validate_derivation() -> dict:
    """The derivation is checked on a FOLD month, where the truth exists, every run."""
    pan = BA.load()
    p = pan["july_loo"]
    mv = D.load_month("2025-07")
    mv["MVT_ID_mvt"] = mv["MVT_ID_mvt"].astype("float64")
    m = mv.set_index("MVT_ID_mvt")
    idx = p.MVT_ID_mvt.to_numpy("float64")
    out = {
        "airport_eq_ADEP_mvt": bool((m.ADEP_mvt.reindex(idx).astype(str).to_numpy()
                                     == p.airport.astype(str).to_numpy()).all()),
        "noac_eq_AIRCRAFT_TYPE_isna": bool(
            (m.AIRCRAFT_TYPE_mvt.reindex(idx).isna().to_numpy() == p.noac.to_numpy(bool)).all()),
        "t_eq_MVT_TIME_UTC_mvt": bool(pd.Series(pd.to_datetime(
            m.MVT_TIME_UTC_mvt.reindex(idx).to_numpy(), utc=True)).eq(
            pd.Series(pd.to_datetime(p["t"], unit="s", utc=True).to_numpy())).all())}
    if not all(out.values()):
        raise SystemExit(f"the 2025-11 attribute derivation does NOT reproduce july_loo: "
                         f"{out} -- every extra row would carry a wrong airport, day or "
                         f"noac flag, and the screen would be measuring that instead")
    return out


def qualifying_airports() -> tuple[list[str], dict]:
    """Section 47.17's companion: airports under 25 s on EVERY fold."""
    pc = json.loads(PRECOND.read_text())
    per = {}
    for f, rec in pc["by_fold"].items():
        for a, v in rec["by_airport"].items():
            per.setdefault(a, {})[f] = v["rms_s"]
    keep = sorted(a for a, v in per.items()
                  if len(v) == len(A.FOLDS) and max(v.values()) <= THRESHOLD_S)
    return keep, per


def artifact_tag(args) -> str:
    """The artifact tag -- a function of the ARGUMENTS alone (47.71.6), so the
    overwrite guard can run before any compute.  Moved verbatim from main()."""
    tag = ("control" if args.control else "placebo" if args.placebo
           else "companion" if args.companion else
           "twomonth-w3" if (args.second_month and args.extra_share) else
           ("twomonth-plus" + "".join(m[5:] for m in args.add_month))
           if (args.second_month and args.add_month) else
           "twomonth" if args.second_month else
           f"only{args.only_month.replace('-', '')}" if args.only_month else "screen")
    # SECTION 47.71.  A --folds SUBSET must carry its folds in the tag.  47.69.5 added the
    # flag and NOT this, so two different reads of the same month wrote the same
    # analysis/onemonth-<tag>.json and work/d-onemonth-<tag>-255-free.npz: the 2025-02
    # july+january subset silently OVERWROTE 47.59.6's december-only artifact.  The
    # measurement survived in the RESULTS record; the durable address did not, which is
    # exactly what the address rule exists to stop.  A full run keeps its shipped name.
    if args.folds is not None:
        _short = {"july_loo": "jul", "december": "dec", "january": "jan"}
        _codes = [_short.get(x.strip(), x.strip())
                  for x in args.folds.split(",") if x.strip()]
        tag = tag + "-folds-" + "-".join(_codes)
    # SECTION 47.75 (recorded rule).  The matched extension is a different set of rows from the same
    # month, so it is a different read and must never share the full month's name.
    if getattr(args, "matched", None):
        tag = tag + "-matched" + args.matched
    return tag


def main() -> None:
    t0 = time.time()
    ps = argparse.ArgumentParser()
    ps.add_argument("--companion", action="store_true",
                    help="restrict the 2025-11 extension to the qualifying airports")
    ps.add_argument("--placebo", action="store_true",
                    help="section 47.19 check 2: permute 2025-11's TARGETS within airport x day")
    ps.add_argument("--only-month", default=None, metavar="YYYY-MM",
                    help="section 47.25 screen (1): use THIS month as the whole extension "
                         "instead of 2025-11, priced over the UNEXTENDED member 1 -- it "
                         "separates the month from the mismatched-row share")
    ps.add_argument("--extra-share", type=float, default=None, metavar="S",
                    help="section 47.28 screen (2): down-weight the extension rows so they "
                         "carry share S of each fold's fitting WEIGHT (fold rows weight 1)")
    ps.add_argument("--add-month", action="append", default=[], metavar="YYYY-MM",
                    help="section 47.31: stack a FURTHER month on top of 2025-11 and 2025-10. "
                         "Repeatable. Requires --second-month, so the stack is always built in "
                         "the order the record priced it")
    ps.add_argument("--second-month", action="store_true",
                    help="section 47.23: 2025-10 ON TOP OF 2025-11, priced over the "
                         "2025-11-extended member -- the increment of the SECOND month")
    ps.add_argument("--price-over", default=None, metavar="NPZ",
                    help="section 47.46: ALSO price the arm over this saved member, as "
                         "reading (iii). The stacked reads (i) and (ii) keep their bases "
                         "unchanged, so nothing already on the record moves; this adds the "
                         "MARGINAL increment of the newest month over the stack one month "
                         "shorter, which is the quantity 47.38.2 registered a prediction for")
    ps.add_argument("--folds", default=None,
                    help="SECTION 47.59.6: fit a COMMA-SEPARATED SUBSET of the folds and "
                         "carry the others from the cached member, so their increment is "
                         "EXACTLY zero by construction rather than to the ULP class (47.50). "
                         "Default None fits all three and reproduces the shipped path.")
    ps.add_argument("--control", action="store_true",
                    help="section 47.18's MISMATCH CONTROL: no new rows, the fitting months' "
                         "targets replaced by the rotated base6 residual")
    ps.add_argument("--matched", default=None, choices=("N2", "F2"),
                    help="SECTION 47.75 (D28): replace the --only-month extension by the "
                         "row-matched draw RESULTS 47.74 fitted for this arm on this fold -- "
                         "same frame, same withinsplit.draw, same per-airport budget, same "
                         "seed -- and price it as 47.69 did, over cached member 1.  Needs "
                         "--only-month equal to the arm's source and --folds naming one fold")
    ps.add_argument("--overwrite", action="store_true",
                    help="SECTION 47.71.6: allow this run to REPLACE an existing "
                         "analysis/ or work/ artifact.  Without it the run refuses, loudly, "
                         "before any compute -- a collision is a naming defect until shown "
                         "otherwise.")
    args = ps.parse_args()

    todo = tuple(A.FOLDS) if args.folds is None else tuple(
        x.strip() for x in args.folds.split(",") if x.strip())
    unknown = [f for f in todo if f not in A.FOLDS]
    if unknown:
        raise SystemExit(f"--folds names {unknown}, which are not folds; known {A.FOLDS}")
    if not todo:
        raise SystemExit("--folds was given but names no fold")

    # SECTION 47.71.6.  The artifact names depend on the arguments ONLY, so they are known
    # here -- before the derivation check, the panel build and the fit.  A run that would
    # overwrite a measured artifact is refused NOW, at no compute cost; the same check runs
    # again at each write.
    tag = artifact_tag(args)
    out_json = EXP / f"analysis/onemonth-{tag}.json"
    out_npz = EXP / f"work/d-onemonth-{tag}-255-free.npz"
    NO.refuse_existing([out_json, out_npz], overwrite=args.overwrite)
    print(f"artifact tag {tag}: no existing artifact at either path", flush=True)

    val = validate_derivation()
    print("derivation validated on july_loo:", val, flush=True)

    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:
        mth = A.FOLD_MONTH[f]
        b = CF.build_month(mth, A.prefix_months_for(mth), rule)
        R.assert_mode3(b, mth, "shipped")
        blk[mth] = b
    pan_ship = CF.panel_for(blk, R.THR, {mth: CF.GAP_CUT_S for mth in blk})
    _, scols = A.shipped_cols(pan_ship)

    d_m1 = {f: v.astype("float64") for f, v in np.load(CACHED_M1).items()}
    head_members = {s: {f: v.astype("float64") for f, v in
                        np.load(EXP / f"work/d-bundle-placebo{s}-255-free.npz").items()}
                    for s in TEN}
    bag10, _ = BAG.build_bag(head_members, pan_ship, TEN)
    if not all(np.array_equal(bag10[f], np.load(HEAD10)[f].astype("float64"))
               for f in A.FOLDS):
        raise SystemExit("the rebuilt K=10 head is not the vector v8 ships")
    print("v8's head rebuilds bit for bit; cached member 1 loaded", flush=True)

    keep, per_ap = qualifying_airports()
    if args.matched:
        import withinsplit as WS                   # noqa: PLC0415 -- it imports this module
        if len(todo) != 1 or todo[0] not in WS.DESIGN:
            raise SystemExit(f"--matched needs --folds naming ONE of {sorted(WS.DESIGN)}, "
                             f"got {todo}")
        want_src = WS.DESIGN[todo[0]][args.matched][0]
        if args.only_month != want_src:
            raise SystemExit(f"--matched {args.matched} on {todo[0]} draws from {want_src}; "
                             f"--only-month is {args.only_month}")
        if (args.second_month or args.control or args.placebo or args.companion
                or args.extra_share is not None or args.add_month or args.price_over):
            raise SystemExit("--matched is 47.69's one-month read with 47.74's rows; it takes "
                             "no other extension flag")
    if args.only_month and (args.second_month or args.control or args.placebo
                            or args.companion):
        raise SystemExit("--only-month replaces the extension; it cannot be combined with "
                         "--second-month, --control, --placebo or --companion")
    if args.extra_share is not None and not args.second_month:
        raise SystemExit("--extra-share weights the two-month extension; pass it with "
                         "--second-month")
    if args.add_month and not args.second_month:
        raise SystemExit("--add-month stacks ON TOP of the two-month extension; pass it with "
                         "--second-month so the stack is built in the priced order")
    if len(set(args.add_month)) != len(args.add_month):
        raise SystemExit(f"--add-month repeats a month: {args.add_month}")
    for extra_m in args.add_month:
        if extra_m in (NOV, OCT):
            raise SystemExit(f"{extra_m} is already in the two-month extension")
        if extra_m not in MONTHS:
            raise SystemExit(f"{extra_m} is not a registered extension month; "
                             f"known: {sorted(MONTHS)}")
    nov = None if args.control else nov_rows(scols, month=(args.only_month or NOV))
    nov_fit = None
    matched_prov = None
    added_months: dict[str, int] = {}
    if not args.control:
        if args.companion:
            before = int(nov.usable.sum())
            nov = nov[nov.airport.astype(str).isin(keep)]
            print(f"companion: {len(keep)} airports qualify, 2025-11 writable rows "
                  f"{int(nov.usable.sum()):,} of {before:,}", flush=True)
        nov_fit = nov[nov.usable.to_numpy(bool)].reset_index(drop=True)
        if args.only_month:
            want = {OCT: OCT_ROWS, NOV: 99709}.get(args.only_month)
            if want is not None and len(nov_fit) != want:
                raise SystemExit(f"{args.only_month} gives {len(nov_fit)} writable rows, not "
                                 f"the {want} the registration was written against")
        print(f"{args.only_month or NOV} writable rows used: {len(nov_fit):,}", flush=True)
        if args.matched:
            # SECTION 47.75 (recorded rule).  47.74's rows for this arm, rebuilt from the same frame by
            # the same function with the same budget and seed.  The frame above is exactly
            # withinsplit.source_rows's (nov_rows, usable subset, index reset), so the draw is
            # the one #4 fitted; its ids are hashed so the claim can be checked.
            ws = json.loads((EXP / f"analysis/withinsplit-{todo[0]}.json").read_text())
            budget = {a: int(v) for a, v in ws["plan"]["budget_per_airport"].items()}
            mseed = WS.SEED + 10 + list(WS.DESIGN[todo[0]]).index(args.matched)
            n_all = len(nov_fit)
            nov_fit = WS.draw(nov_fit, budget, mseed)
            if len(nov_fit) != ws["plan"]["fit_rows_per_set"]:
                raise SystemExit(f"the matched draw has {len(nov_fit)} rows, 47.74 fitted "
                                 f"{ws['plan']['fit_rows_per_set']}")
            got = nov_fit.airport.astype(str).value_counts().to_dict()
            if {a: int(got.get(a, 0)) for a in budget} != budget or set(got) - set(budget):
                raise SystemExit(f"the matched draw's airports {got} are not the budget {budget}")
            ids = np.sort(nov_fit.MVT_ID_mvt.to_numpy("float64"))
            matched_prov = {"arm": args.matched, "fold": todo[0], "source": args.only_month,
                            "seed": int(mseed), "budget_per_airport": budget,
                            "rows": int(len(nov_fit)), "drawn_from_writable_rows": int(n_all),
                            "mvt_id_sha16": hashlib.sha256(ids.tobytes()).hexdigest()[:16],
                            "budget_from": f"analysis/withinsplit-{todo[0]}.json"}
            print(f"matched draw ({args.matched}, seed {mseed}): {len(nov_fit):,} of {n_all:,} "
                  f"rows, ids sha16 {matched_prov['mvt_id_sha16']}", flush=True)
        if args.second_month:
            # SECTION 47.23.  The second month sits ON TOP of the first: the fit carries both,
            # and the increment is read over the member that carried 2025-11 alone.  Anything
            # else would price the two months together and call it the second one.
            octr = nov_rows(scols, month=OCT)
            oct_fit = octr[octr.usable.to_numpy(bool)].reset_index(drop=True)
            if len(oct_fit) != OCT_ROWS:
                raise SystemExit(f"2025-10 gives {len(oct_fit)} writable rows, not the "
                                 f"{OCT_ROWS} section 47.23.5 re-registered the band on")
            miss = [c for c in nov_fit.columns if c not in oct_fit.columns]
            if miss:
                raise SystemExit(f"2025-10 lacks {len(miss)} column(s) 2025-11 carries: "
                                 f"{miss[:8]} -- the two months must be one frame")
            n11 = len(nov_fit)
            nov_fit = pd.concat([nov_fit, oct_fit.reindex(columns=nov_fit.columns)],
                                ignore_index=True)
            if len(nov_fit) != n11 + len(oct_fit):
                raise SystemExit("the two months did not concatenate")
            print(f"2025-10 writable rows added: {len(oct_fit):,}  ->  extension total "
                  f"{len(nov_fit):,}", flush=True)
        for extra_m in args.add_month:
            # SECTION 47.31.  Each further month goes on the SAME frame under the SAME guards
            # the second month gets: registered sha, one frame of columns, a concat that is
            # proved by row count.  A month with no registered count is recorded, not gated --
            # 47.31 restates its bands before the fit, never after seeing the read.
            xr = nov_rows(scols, month=extra_m)
            x_fit = xr[xr.usable.to_numpy(bool)].reset_index(drop=True)
            want_rows = MONTHS[extra_m].get("rows")
            if want_rows is not None and len(x_fit) != want_rows:
                raise SystemExit(f"{extra_m} gives {len(x_fit)} writable rows, not the "
                                 f"{want_rows} registered for it")
            miss = [c for c in nov_fit.columns if c not in x_fit.columns]
            if miss:
                raise SystemExit(f"{extra_m} lacks {len(miss)} column(s) the extension carries: "
                                 f"{miss[:8]} -- every month must be one frame")
            n_before_m = len(nov_fit)
            nov_fit = pd.concat([nov_fit, x_fit.reindex(columns=nov_fit.columns)],
                                ignore_index=True)
            if len(nov_fit) != n_before_m + len(x_fit):
                raise SystemExit(f"{extra_m} did not concatenate")
            added_months[extra_m] = int(len(x_fit))
            print(f"{extra_m} writable rows added: {len(x_fit):,}  ->  extension total "
                  f"{len(nov_fit):,}"
                  + ("" if want_rows is not None else "   [count NOT registered: recorded only]"),
                  flush=True)
        if args.placebo:
            # SECTION 47.19 CHECK 2.  Permute the TARGET within airport x day.  Features are
            # untouched and every row is kept, so the extra rows keep every property except
            # the correspondence between a row's features and its label.  Whatever the extra
            # rows can still do here is NOT information.
            rb = nov_fit.rbase.to_numpy("float64").copy()
            before = rb.copy()
            key = (nov_fit.airport.astype(str) + "|" + nov_fit.day.astype(str)).to_numpy()
            order = np.argsort(key, kind="stable")
            ks = key[order]
            bounds = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
            rng = np.random.default_rng(PLACEBO_SEED)
            n_perm, n_single = 0, 0
            for a, b in zip(bounds[:-1], bounds[1:]):
                idx = order[a:b]
                if len(idx) < 2:
                    n_single += 1
                    continue
                rb[idx] = before[rng.permutation(idx)]
                n_perm += 1
            moved = int((rb != before).sum())
            if moved < 0.5 * len(rb):
                raise SystemExit(
                    f"the placebo permuted only {moved:,} of {len(rb):,} targets. A permutation "
                    f"that barely moves anything is NOT a placebo -- it would read like the real "
                    f"extension and 'the placebo is near zero' would be meaningless.")
            nov_fit = nov_fit.copy()
            nov_fit["rbase"] = rb
            print(f"placebo: {n_perm:,} airport x day cells permuted, {n_single:,} singletons "
                  f"left, {moved:,} of {len(rb):,} targets moved "
                  f"({100*moved/len(rb):.1f}%), rms unchanged "
                  f"{np.sqrt((before**2).mean()):.3f} -> {np.sqrt((rb**2).mean()):.3f}",
                  flush=True)

    rng = np.random.default_rng(SEED_MEMBER)
    blocks = {}
    for mth, b in blk.items():
        pb, _ = CP.placebo_block(b, R.THR, CF.GAP_CUT_S, CP.MOVED[mth], rng)
        blocks[mth] = pb
    real = A.month_block
    A.month_block = (lambda bs: (lambda month, prefix_months, rule=None: bs[month]))(blocks)
    try:
        pan = A.panel_with_block(BA.load())
    finally:
        A.month_block = real

    # SECTION 47.18's MISMATCH CONTROL.  No new rows.  Every fold's `rbase` -- the target a
    # FITTING row carries -- is replaced by the rotated base6 residual, while the SCORED fold
    # is priced against pbase_true on `pan_ship` exactly as always.  `arm_lgb` reads `rbase`
    # only on the fitting frame; scoring uses the features, `usable` and `airport`, so the
    # substitution cannot leak into the score.  It isolates what the 30-39 s target mismatch
    # costs with the row count held fixed.
    ctl_prov = {}
    if args.control:
        for f in A.FOLDS:
            b6 = pd.read_parquet(OOF / f"base6--{f}.parquet")
            got = hashlib.sha256((OOF / f"base6--{f}.parquet").read_bytes()).hexdigest()
            m6 = b6.set_index("MVT_ID_mvt")
            idx = pan[f].MVT_ID_mvt.to_numpy("float64")
            p6 = m6.y_pred.reindex(idx).to_numpy("float64")
            y6 = m6.y_true.reindex(idx).to_numpy("float64")
            y = pan[f].y_true.to_numpy("float64")
            ok = np.isfinite(p6)
            if int((np.abs(y6[ok] - y[ok]) > 1e-6).sum()):
                raise SystemExit(f"{f}: base6's y_true differs from the panel's -- bad join")
            if not ok.all():
                raise SystemExit(f"{f}: {int((~ok).sum())} panel rows have no base6 value")
            old_rb = pan[f].rbase.to_numpy("float64").copy()
            pan[f]["rbase"] = y - p6
            ctl_prov[f] = {
                "base6_sha16": got[:16],
                "rms_target_change_s": float(np.sqrt(np.mean((pan[f].rbase.to_numpy() - old_rb) ** 2))),
                "rows": int(len(pan[f]))}
            if ctl_prov[f]["rms_target_change_s"] < 1.0:
                raise SystemExit(f"{f}: the target moved by {ctl_prov[f]['rms_target_change_s']:.3f} s "
                                 f"-- that is what an unsubstituted fall-through looks like and "
                                 f"it would price as +0.00 and read as 'the mismatch is free'")
        print("control: fitting targets replaced by base6 residual, rms change "
              + " ".join(f"{f[:4]} {ctl_prov[f]['rms_target_change_s']:.2f}s"
                         for f in A.FOLDS), flush=True)

    setting_was, hold_was = CF.SETTING, CF.HOLD
    CF.SETTING, CF.HOLD = {"num_leaves": LEAVES}, ()
    prov: dict = {}
    t1 = time.time()
    try:
        d = CF.fit_arm(pan, prov=prov, extra=nov_fit,   # extra=None in the control
                       extra_share=args.extra_share, folds=todo)
    finally:
        CF.SETTING, CF.HOLD = setting_was, hold_was
    # SECTION 47.59.6.  A fold this call did not fit is carried from the cached member the
    # read is priced against, so its increment is EXACTLY 0.0 -- not "small".  A refit would
    # reproduce it only to the ULP class (47.50), which is the very difference that would
    # make an untouched fold read not-quite-zero and invite it to be called a reading.
    for f in A.FOLDS:
        if f not in todo:
            d[f] = d_m1[f].copy()
            prov.setdefault(f, {})["carried_from_the_cached_member_not_fitted"] = True
    if not args.control:
        miss = [f for f in todo if "extra_rows" not in prov.get(f, {})]
        if miss:
            raise SystemExit(f"no extra-row provenance for {miss} -- the fit ran on the plain "
                             f"panel while being labelled a one-month extension")
    print(f"fitted in {time.time()-t1:.0f}s  " + ("no new rows (control)" if args.control
          else "  ".join(f"{f[:4]} {prov[f]['total_fit_rows']:,} "
                         f"({prov[f]['extra_multiple']:.3f}x)" for f in todo)), flush=True)
    NO.refuse_existing([out_npz], overwrite=args.overwrite)
    np.savez_compressed(out_npz, **d)

    if args.second_month and args.extra_share:
        i = R.price(pan_ship, d, d_m1,
                    "(i) the WEIGHTED two-month extension over the unextended member 1")
    elif args.second_month:
        prev = EXP / "work/d-onemonth-screen-255-free.npz"
        if not prev.exists():
            raise SystemExit(f"{prev.name} is missing -- the second month is priced over the "
                             f"2025-11-extended member and there is nothing to price against")
        d_prev = {f: v.astype("float64") for f, v in np.load(prev).items()}
        i = R.price(pan_ship, d, d_prev,
                    "(i) the SECOND month over the 2025-11-extended member 1")
    else:
        i = R.price(pan_ship, d, d_m1, "(i) the one-month extension over cached member 1")
    ii = R.price(pan_ship, d, bag10, "(ii) the same over v8's K=10 head, record only")
    iii = None
    if args.price_over:
        base_npz = EXP / "work" / args.price_over
        if not base_npz.exists():
            raise SystemExit(f"{base_npz} does not exist -- --price-over names the member the "
                             f"MARGINAL increment is read against and it must already be built")
        d_base = {f: v.astype("float64") for f, v in np.load(base_npz).items()}
        iii = R.price(pan_ship, d, d_base,
                      f"(iii) the MARGINAL increment over {args.price_over}")
    pc = json.loads(PRECOND.read_text())
    rec = {"what": __doc__.strip().splitlines()[0], "which": tag,
           "member": SEED_MEMBER, "leaves": LEAVES,
           "derivation_validated_on_july_loo": val,
           "extra_row_provenance": prov,
           "nov_writable_rows_used": None if nov_fit is None else int(len(nov_fit)),
           "matched_draw": matched_prov,
           "extension_months": ([] if args.control else
                                [args.only_month or NOV]
                                + ([OCT] if args.second_month else [])
                                + list(args.add_month)),
           "added_month_rows": added_months,
           "control_target_substitution": ctl_prov or None,
           "companion_airports": keep if args.companion else None,
           "priced_over": i["label"],
           "marginal_priced_over": None if iii is None else args.price_over,
           "precondition": {"rms_by_fold": {f: pc["by_fold"][f]["rms_s_usable"]
                                            for f in A.FOLDS},
                            "threshold_s": THRESHOLD_S, "passes": pc["passes"]},
           "evidence_about_rows": bool(args.companion or pc["passes"]),
           "prices": ({"i_over_cached_member_1": i, "ii_over_v8_head_record_only": ii}
                      | ({} if iii is None
                         else {"iii_marginal_over_the_shorter_stack": iii}))}
    if not rec["evidence_about_rows"]:
        rec["MUST_BE_READ_WITH"] = (
            "The section 47.17 precondition FAILED at 39.154 / 30.095 / 30.109 s against a "
            "25 s threshold registered before any data. This number is NOT evidence about "
            "rows. The companion run, restricted to airports under 25 s on every fold, is.")
    g, ci = i["board_all_row"]["gain_s"], i["board_all_row"]["interval"]["ci95"]
    rec["verdict"] = {
        "gate_reading": "i_over_cached_member_1",
        "board_gain_s": g, "ci95": ci,
        "every_fold_interval_clear": bool(i["every_fold_interval_clear"]),
        "folds_clear": {f: bool(i["by_fold"][f]["ci95"][0] > 0) for f in A.FOLDS},
        "passes_the_gate": bool(i["every_fold_interval_clear"] and g > 0),
        "ii_board_gain_s": ii["board_all_row"]["gain_s"]}
    rec["seconds"] = round(time.time() - t0, 1)
    NO.refuse_existing([out_json], overwrite=args.overwrite)
    out_json.write_text(
        json.dumps(rec, indent=2, sort_keys=True, default=float))

    print()
    print(R.row(i)); print(R.fold_rows(i))
    v = rec["verdict"]
    # the base for (i) DEPENDS on the branch above; printing a fixed name for it misstated
    # the reading on every --second-month run (47.32.6), so the label is carried, not assumed
    print(f"\n  {i['label']}:\n      {g:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}]  "
          f"every-fold-clear {v['every_fold_interval_clear']}  gate "
          f"{v['passes_the_gate']}")
    print(f"  (ii) over v8's head, record only: {v['ii_board_gain_s']:+.5f}")
    if iii is not None:
        print(f"\n{R.row(iii)}\n{R.fold_rows(iii)}")
        gm = iii["board_all_row"]["gain_s"]
        cm = iii["board_all_row"]["interval"]["ci95"]
        print(f"\n  {iii['label']}:\n      {gm:+.5f} [{cm[0]:+.5f}, {cm[1]:+.5f}]  "
              f"every-fold-clear {iii['every_fold_interval_clear']}")
    if not rec["evidence_about_rows"]:
        print("\n  *** NOT EVIDENCE ABOUT ROWS -- the 47.17 precondition failed at "
              "39.154 / 30.095 / 30.109 s against a 25 s threshold registered before "
              "the data. ***")
    print("written to", (EXP / f'analysis/onemonth-{tag}.json').relative_to(ROOT),
          f"({rec['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
