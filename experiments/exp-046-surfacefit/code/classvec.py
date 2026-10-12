"""The ranking-month class vector, at a durable address, for the composition.

The desk, 2026-09-09: `exp-047-queue` adopts this lane's prefix map (the nine non-fold
2025 months, departures only -- `arms.prefix_months_for`), rebuilds its blocks and
re-serves on the COMPLEMENT of this lane's class.  For that the complement has to be a
file, not a re-derivation: two lanes deriving "the same" vector twice is exactly what put
87 ranking rows on both sides of the boundary yesterday (`analysis/boundary-diff-read.json`).

So this writes the vector once, from the same `serve.ranking_block()` the served candidate
was built from, to `artifacts/exp-046/class/<month>.parquet`:

  MVT_ID_mvt              the ranking key
  obs_has_clock           the class the desk named: the matcher found the row AND the
                          segment carries an off-block instant of its own
  served_by_exp_046       `arms.served_class` -- obs_has_clock & obs_taxi.notna() & ~noac.
                          NARROWER than obs_has_clock: it drops the 388 `noac` rows and
                          any row whose clock does not resolve to a taxi duration
  noac                    the `noac` hold (exp-033).  These rows move in NEITHER lane
  moved_by_v13            the rows the served candidate actually moves off v12.  Narrower
                          again: LFPG is held at the base (desk, 2026-09-09), so 1,612
                          served-class rows are at v12 in the shipped file

The complement to serve is `~obs_has_clock` for the class the desk named; a composition
that wants every row exactly once takes `~moved_by_v13 & ~noac` instead, and the manifest
carries both counts so the choice is made on numbers.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/classvec.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parent.parent

# `serve` FIRST: `arms` pulls in `imports.load`, which puts other experiments' code
# directories at the front of sys.path, and one of them also has a `serve.py`
import serve as S                                                     # noqa: E402
import arms as A                                                      # noqa: E402
import guard as G                                                     # noqa: E402

KEY = "MVT_ID_mvt"
#: the match mode whose rows entered this lane's class with the CURRENT matcher.
#: `122c7d84` adds the turnaround pass and nothing else, so mode 3 is exactly the
#: class this vector gains over the bytes it is emitted against.
#:
#: **It is a property of the CANDIDATE, not of the matcher**, and `--gained-modes` says
#: so.  "Gained" means "in this lane's class now and NOT written in these bytes"; for a
#: vector emitted against a file built before the pass that is mode 3, and for a vector
#: emitted against a file that serves every occupied row it is nothing at all.  v19 is the
#: second case -- it writes the turnaround pass AND the gated ungated rows (mode 4) -- so
#: it is emitted with `--gained-modes=` empty and `written_by_exp_046` equals
#: `occupied_by_exp_046` less whatever the guard returned to the base.
NEW_CLASS_MODE = 3
#: the ungated chain rows arrive as `match_mode` 4 (`serve.UNGATED_GATE`); counted in the
#: manifest whether or not they are "gained", because the composition needs the split
CHAIN_MODE = 4
TGT = S.D.TARGET

#: Defects found in an ALREADY EMITTED vector, carried into the manifest so a reviewer
#: meets them rather than finds them.  The vector at the address is the repaired one.
DEFECTS = [{
    "found": "2026-09-09, by this lane, before review round 8 read the commit",
    "affected": ("the first emission of the v20 vector, "
                 "artifacts/exp-046/class/2026-{01,07}-v20.parquet"),
    "what": ("guard_held_by_exp_046 was all-False on a GUARDED candidate, so "
             "written_by_exp_046 = occupied & ~gained & (~at_base | guard_held) dropped "
             "every occupied row the guard had returned to the price base: 194,683 "
             "written against 196,256 occupied (2026-01 90,074 of 91,433, 2026-07 "
             "104,609 of 104,823), and chain_mode_4_written read 2,451 of 2,456"),
    "cause": ("the run passed neither --arm nor --guarded=yes, so `guarded` was False and "
              "the whole guard branch -- including its two-way guard/price-base assertion "
              "-- was silently skipped"),
    "repair": ("re-emitted with --arm=c_lgb_shipped@leaves_255x300_lr050+guard995 and "
               "--guard-cut=1872.1075545150375, the gate.tail_guard.cut_s of "
               "analysis/serve-v20.json: guard_held 1,573, written == occupied == "
               "196,256, and the assertion ran"),
    "what_did_not_move": ("occupied_by_exp_046 is identical ROW FOR ROW across the two "
                          "emissions and only guard_held_by_exp_046 and "
                          "written_by_exp_046 changed, so the desk's verification of the "
                          "v6 bytes stands: analysis/v20-reemit-diff.json"),
    "closed_by": ("silence is now fatal -- the caller states --arm or --guarded=yes, or "
                  "affirms --unguarded=yes"),
    "bytes_unaffected": ("submissions/candidates/exp-046-surface-v6.parquet, sha256 "
                         "54dec3df14a9aaea6938b05c0b8e4280a958bce32f734b293ecb2deb32eb41f"
                         "c: the vector describes the file and never writes it"),
}]

OUTDIR = ROOT / "artifacts/exp-046/class"
CAND = ROOT / "submissions/candidates/exp-046-surface.parquet"
BASE = ROOT / "submissions/candidates/exp-035-k2climb-v11member.parquet"
COLS = ["MVT_ID_mvt", "obs_has_clock", "served_by_exp_046", "noac",
        "moved_by_v13", "occupied_by_exp_046", "guard_held_by_exp_046",
        "written_by_exp_046", "chain_mode_4"]
#: SECTION 47.49: on a SEASONAL candidate the head is composed BY SERVED MONTH, so a reader
#: of the class vector cannot tell which head wrote a row from the bytes.  The column is
#: added only when the serve report says the candidate is composed that way -- an ordinary
#: candidate's vector keeps exactly the columns above, so nothing already emitted changes.
HEAD_COL = "head_by_exp_046"

#: A second candidate needs its OWN vector and must not overwrite the one an upload was
#: already composed on.  `--candidate=<file> --against=<file> --suffix=<s> --moved-col=<c>`
#: writes `<month><suffix>.parquet` from the block THIS run builds, so the vector and the
#: file it describes come from one build.  v15's class comes from the per-month block
#: (round 3 finding 1) and the published whole-file vector differs from it on the 21 rows
#: round 3 finding 2 predicted; the two are kept side by side rather than one silently
#: replacing the other.


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = {a.split("=", 1)[0]: a.split("=", 1)[1]
            for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    # a path given on the command line may be relative to the working directory; the
    # manifest quotes it relative to the repository root, so resolve it first
    cand = Path(args.get("--candidate", CAND)).resolve()
    against = Path(args.get("--against", BASE)).resolve()
    suffix = args.get("--suffix", "")
    moved_col = args.get("--moved-col", "moved_by_v13")
    rule, _ = A.serving_rule()
    # **READ THE SERVE, DO NOT RETYPE IT** (desk, 2026-09-09).  Two emission defects in one
    # day came from a human transcribing a setting the serve report already carries: the
    # v20 guard flags (section 35.4) and the v22 `--gained-modes` default (section 45.3).
    # Both were single values sitting in `analysis/serve-<set>.json`.  `--from-serve=<path>`
    # takes the arm from the report, the guard cut from `gate.tail_guard.cut_s` and the
    # expected held count from `gate.tail_guard.rows_returned_to_the_base`, and the two
    # assertions at the end of this file check the emitted vector against them.  What the
    # repeat says about the structure is written in section 45.4: the operator should not
    # be a channel for a number that already exists in a file.
    from_serve = args.get("--from-serve")
    serve_rec = None
    if from_serve:
        serve_rec = json.loads(Path(from_serve).resolve().read_text())
        tg = serve_rec["gate"]["tail_guard"]
        args.setdefault("--arm", serve_rec["arm"])
        args.setdefault("--guard-cut", repr(float(tg["cut_s"])))
        if serve_rec.get("sha256") and args.get("--candidate"):
            got = sha256(Path(args["--candidate"]).resolve())
            if got != serve_rec["sha256"]:
                raise SystemExit(
                    f"--from-serve names {Path(from_serve).name}, whose candidate is "
                    f"{serve_rec['sha256'][:12]}..., but --candidate on disk is "
                    f"{got[:12]}... -- the vector would describe bytes the serve did not "
                    "write")
    guarded = (G.TAG in args.get("--arm", "")) or args.get("--guarded") == "yes"
    # THE 2026-09-09 DEFECT, and why silence is fatal here now.  `guarded` was inferred
    # from the flags alone, so an operator who passed neither got an UNGUARDED reading of
    # a GUARDED candidate: `guard_held_by_exp_046` all-False, the two-way assertion below
    # skipped, and `written_by_exp_046` short by every occupied row the guard had returned
    # to the price base -- 1,573 of them on the first v20 emission.  Nothing warned,
    # because "no guard" is the right answer for an unguarded candidate and the two cases
    # were indistinguishable from the outside.  A vector that describes SERVED BYTES may
    # not guess: the caller states which it is, or this stops.
    if not guarded and args.get("--unguarded") != "yes":
        raise SystemExit(
            "state the guard. Pass --arm=<the served arm> or --guarded=yes, with "
            "--guard-cut=<the cut that wrote the file> read off the serve report's "
            "gate.tail_guard.cut_s; or --unguarded=yes to affirm the candidate carries no "
            "guard. Inferring it from silence emitted a wrong v20 on 2026-09-09 (see "
            "DEFECTS in this file).")
    # `--ungated`: read the ranking block the v19 candidate was WRITTEN from, at exactly
    # the section 28.2 gate.  The vector describes a file, so it has to be built by the
    # same switch that built it -- a vector emitted on the shipped matcher would call
    # 2,493 rows this lane owns "queue's".
    # SECTION 47.22.4's durable repair, the one BACKLOG named: --ungated had to be given to
    # the serve and to this file SEPARATELY, and nothing but a lucky five-row overlap in the
    # guard count connected them. The base set now DECLARES the setting, so when the serve
    # report is being read anyway, the flag is checked against it instead of remembered.
    _ungated = "--ungated" in sys.argv
    _seasonal_months: dict = {}
    if "--from-serve" in args:
        _srv = json.loads(Path(args["--from-serve"]).read_text())
        _bs = _srv.get("base", {}).get("set")
        _ext = (_srv.get("extension") or {})
        if _ext.get("composed_by") == "served month":
            _seasonal_months = {m: v["declared"]["months"]
                                for m, v in _ext["by_served_month"].items()}
            print(f"the serve composed the head by served month: "
                  f"{ {m: '+'.join(v) for m, v in _seasonal_months.items()} }", flush=True)
        _want = S.BASE_SETS.get(_bs, {}).get("ungated")
        if _want is not None and bool(_want) != _ungated:
            raise SystemExit(
                f"base set {_bs!r} declares ungated={_want} and this emission was given "
                f"ungated={_ungated}. Section 47.22.4: the serve and its class vector must "
                f"agree on this flag, and the disagreement is silent -- it shows up only as "
                f"a class short by the mode-4 rows, caught last time by a five-row overlap")
        print(f"--ungated={_ungated} agrees with base set {_bs!r}", flush=True)
    S._UNGATED = _ungated
    dep, _ = S.ranking_block()

    # the shipped candidate against its base, row for row on the template key: the one
    # honest reading of "which rows this lane moves"
    cand_df = pd.read_parquet(cand)
    base = pd.read_parquet(against)
    if not cand_df[KEY].equals(base[KEY]):
        raise SystemExit("the candidate and the base are not in one and the same order")
    moved = pd.DataFrame({KEY: cand_df[KEY].astype("float64").to_numpy(),
                          moved_col: cand_df[TGT].to_numpy() != base[TGT].to_numpy()})

    out = dep[["MVT_ID_mvt", "obs_has_clock", "usable", "noac", "airport",
               "rank_month"]].rename(columns={"usable": "served_by_exp_046"})
    out[KEY] = out[KEY].astype("float64")
    out = out.merge(moved, on=KEY, how="left", validate="one_to_one")
    if out[moved_col].isna().any():
        raise SystemExit("a ranking departure row is absent from the candidate")
    for c in ("obs_has_clock", "served_by_exp_046", "noac", moved_col):
        out[c] = out[c].astype(bool)

    # **Which rows this lane OCCUPIES**, which is neither of the two columns above (round
    # 4 FALLBACK, finding 4).  `served_by_exp_046` counts a held airport, which this lane
    # leaves at the base; `changed_by_*` counts a REVERTED row, which this lane gives up.
    # A composition that complements either leaves rows unowned or owned twice.  The held
    # airports are read off the candidate itself -- an airport all of whose class rows are
    # at the price base is held -- so the column cannot drift from the file it describes.
    held = sorted(a for a, g in out.groupby("airport", observed=True)
                  if g.served_by_exp_046.any() and not g.loc[g.served_by_exp_046,
                                                             moved_col].any())
    out["occupied_by_exp_046"] = (out.served_by_exp_046
                                  & ~out.airport.astype(str).isin(held))

    # **Which occupied rows the TAIL GUARD returned to the price base** (round 5 FALLBACK,
    # finding 6).  A guarded candidate occupies these rows -- this lane decided their value
    # -- and the value it decided is the price base itself.  The revert rule's consistency
    # assertion in `serve.py` reads "an occupied row must differ from the price base", which
    # is true of every unguarded candidate and false of 1,481 rows of v16, so a v17 composed
    # on v16 would stop on a file that is correct.  The column is DECLARED here, computed
    # from the guard's own frozen cut and not inferred from the values, and `serve.py`
    # checks the file against it in BOTH directions -- so the assertion keeps its force
    # instead of being widened into a hole.
    # **THE ROWS THE CLASS GAINED SINCE THE CANDIDATE WAS WRITTEN.**  v17's bytes were
    # built on matcher `6fdf3951`, and the only change since is the turnaround pass, so the
    # rows in this lane's class now and not then are exactly the mode-3 rows.  They matter
    # to BOTH columns below and the old definitions get both wrong once the other lane
    # moves them:
    #
    #   * `guard_held_by_exp_046` claims "this lane decided this row's value and the value
    #     is the price base".  It cannot claim a row whose class the bytes never had.
    #   * `written_by_exp_046` claims "this lane wrote a value here in THESE bytes".  Its
    #     old test was "the value is not the price base", which was sound while every row
    #     outside this lane's class sat at the base -- and is wrong now: about 1,590 of the
    #     2,730 rows arriving with the pass are rows `exp-047-queue` MOVES today
    #     (queue's rider 2, `690315b`), so they carry a non-base value the OTHER lane wrote,
    #     and the old test would credit this lane with the other lane's bytes.
    #
    # Both columns therefore exclude them, and the count is reported rather than absorbed.
    gm = args.get("--gained-modes")
    # **STATE IT; DO NOT INHERIT IT.**  Whether these bytes carry the new-class rows is a
    # property of THE CANDIDATE, and the default silently made it a property of the day.
    # On 2026-09-09 the v22 emission omitted the flag, took the default [3.0], and
    # declared 3,179 occupied rows unwritten that v8 had in fact written -- v19, v20 and
    # v21 were all emitted with `--gained-modes=` empty.  Nothing warned, because both
    # readings are legitimate for SOME candidate.  This is the same defect class as the
    # guard flag above and as serve.py's arm and --ungated: a declared-or-inherited
    # setting that changes what the artifact MEANS, with no error on the silent branch.
    if gm is None:
        raise SystemExit(
            "state the class. Pass --gained-modes=<modes this candidate's bytes do NOT "
            "carry>, comma separated, or --gained-modes= (empty) to affirm the bytes "
            "carry every occupied row. Read it off the serve report: if "
            "gate/served_by_exp_046 equals occupied_by_exp_046, the answer is empty. "
            "Inheriting it emitted a wrong v22 on 2026-09-09 (see DEFECTS in this file).")
    gained_modes = [float(x) for x in gm.split(",") if x]
    gained = (np.isin(dep.match_mode.to_numpy("float64"), gained_modes)
              if ("match_mode" in dep and gained_modes) else np.zeros(len(out), bool))
    # SECTION 47.49: which head wrote this row, when the serve composed one per served month
    if _seasonal_months:
        miss = sorted(set(out.rank_month.astype(str)) - set(_seasonal_months))
        if miss:
            raise SystemExit(
                f"the serve report declares heads for {sorted(_seasonal_months)} but the "
                f"class carries rank_month(s) {miss}. Those rows were written by some head "
                f"and this vector could not say which -- the column would be a guess")
        out[HEAD_COL] = out.rank_month.astype(str).map(
            {m: f"{m}:{'+'.join(v)}" for m, v in _seasonal_months.items()}).astype(str)
        if out[HEAD_COL].nunique() < 2:
            raise SystemExit(
                f"{HEAD_COL} takes one value on this vector, so the candidate is not "
                f"composed by served month and the column would claim it is")
        if HEAD_COL not in COLS:
            COLS.append(HEAD_COL)
    out["chain_mode_4"] = (dep.match_mode.to_numpy("float64") == float(CHAIN_MODE)
                           if "match_mode" in dep else np.zeros(len(out), bool))
    gh = np.zeros(len(out), bool)
    guard_cut = None
    if guarded:
        gj = EXP / "analysis/guard-arms.json"
        if not gj.exists():
            raise SystemExit("a guarded candidate needs analysis/guard-arms.json")
        rec = json.loads(gj.read_text())
        if rec["guard"]["quantile"] != G.Q:
            raise SystemExit(f"guard-arms.json is at Q={rec['guard']['quantile']}, "
                             f"code/guard.py freezes {G.Q}")
        guard_cut = float(rec["guard"]["cut_s_for_the_ranking_serve"])
        # THE CUT THAT WROTE THE FILE, not the cut a later re-price computed.  The guard
        # cut is read off the block's own tail, so a matcher change moves it: the
        # turnaround pass moved it from 1864.37 to 1870.64.  `guard_held_by_exp_046`
        # describes the BYTES this vector is emitted against, so a vector re-emitted on a
        # new matcher over an older file must be given that file's cut -- otherwise the
        # assertion below fires on rows the new cut holds and the old bytes never held,
        # which is the machinery being right and the caller being wrong.  Recorded in the
        # manifest either way, so the vector says which cut its column means.
        cut_override = args.get("--guard-cut")
        if cut_override is not None:
            guard_cut = float(cut_override)
        # the guard is read against the PRICE base, which is v12 for every candidate this
        # lane has served; `serve.py` uses the same file for the same purpose
        pb = pd.read_parquet(Path(args.get("--price-base", BASE)).resolve())
        pmap = pd.Series(pb[TGT].to_numpy("float64"),
                         index=pb[KEY].to_numpy("float64"))
        pbase = out[KEY].map(pmap).to_numpy("float64")
        if not np.isfinite(pbase).all():
            raise SystemExit("a ranking departure is absent from the price base")
        m = G.mask(pbase, dep.obs_taxi, guard_cut)
        gh = m & out.occupied_by_exp_046.to_numpy(bool) & ~gained
        # the vector must agree with the file it describes, in BOTH directions: a row the
        # guard holds carries the price base, and an occupied row at the price base is one
        # the guard holds.  The one-sided reading is what round 5 finding 6 tripped over.
        candv = cand_df[TGT].to_numpy("float64")
        # EXACT equality, not `isclose`.  The assertion this column protects is
        # `serve.py`'s `headv == pbasev`, which is exact, and a tolerance here would
        # disagree with it in both directions: at the default `rtol` every row whose
        # update happens to be under about 0.01 s reads as "at the base" -- 540 of v16's
        # 188,484 occupied rows -- and the check would demand the guard explain rows the
        # guard never touched.  A returned row carries the base's bits, so `==` is the
        # question being asked.
        at_base = candv == pbase
        bad_held = int((gh & ~at_base).sum())
        if bad_held:
            raise SystemExit(f"{bad_held} rows the guard holds are not at the price base "
                             "in the candidate; the vector and the file disagree")
    out["guard_held_by_exp_046"] = gh

    # **THE CLASS CAN MOVE UNDER A FILE ALREADY WRITTEN**, and a vector re-emitted on a
    # newer matcher has to say so rather than fail.  v16's bytes were built by matcher
    # `695af973`; this vector is emitted on `6fdf3951`, where 526 more ranking rows carry
    # a clock -- 506 EDDF rows of 2026-01-27 and 28, returned to service by the corrected
    # outage table, and 20 rows the total tie-break gave a clock to.  They are in this
    # lane's class NOW, and v16 leaves them at the price base, because when v16 was
    # written they were not.  That is neither a guard hold nor a disagreement: it is a
    # third state, and the honest thing is to name it and count it.  `occupied_by_exp_046`
    # stays THE MASK -- what a later serve in either lane must complement -- and
    # `written_by_exp_046` is the narrower "this lane actually wrote a value here in THESE
    # bytes".  A composition that wants "who owns this row next" reads the first; one that
    # asks "what is in this file" reads the second.
    if guard_cut is not None or True:
        at_base_v = (cand_df[TGT].to_numpy("float64")
                     == out[KEY].map(pd.Series(
                         pd.read_parquet(Path(args.get("--price-base", BASE)).resolve())
                         [TGT].to_numpy("float64"),
                         index=pd.read_parquet(
                             Path(args.get("--price-base", BASE)).resolve())
                         [KEY].to_numpy("float64"))).to_numpy("float64"))
        occ = out.occupied_by_exp_046.to_numpy(bool)
        out["written_by_exp_046"] = (occ & ~gained
                                     & (~at_base_v | out.guard_held_by_exp_046.to_numpy(bool)))
        class_moved_under_the_file = int(
            (occ & ~gained & at_base_v
             & ~out.guard_held_by_exp_046.to_numpy(bool)).sum())
        class_gained_since_the_candidate = int((occ & gained).sum())
        gained_carrying_another_lanes_value = int((occ & gained & ~at_base_v).sum())
    else:                                                          # pragma: no cover
        out["written_by_exp_046"] = out.occupied_by_exp_046 & ~gained
        class_moved_under_the_file = 0
        class_gained_since_the_candidate = int((out.occupied_by_exp_046 & gained).sum())
        gained_carrying_another_lanes_value = 0
    # occupancy is "this lane writes its own head here", which is a statement about the
    # MASK and not about the value, so it is not defined as "the value changed"; that the
    # two agree is a check and is reported rather than assumed
    occupied_not_moved = int((out.occupied_by_exp_046 & ~out[moved_col]).sum())

    OUTDIR.mkdir(parents=True, exist_ok=True)
    man = {
        "what": ("this lane's class vector on the two ranking months, written once so "
                 "the composition reads a file instead of re-deriving it"),
        "written_by": "experiments/exp-046-surfacefit/code/classvec.py",
        "defects": DEFECTS,
        "from": ("serve.ranking_block() -- the same block the served candidate was "
                 "built from"),
        "matcher_sha256": A.MATCHER_SHA,
        "rule_sha256": rule,
        "prefix_months": list(A.prefix_months_for("")),
        "candidate": str(cand.relative_to(ROOT)), "candidate_sha256": sha256(cand),
        "against": str(against.relative_to(ROOT)), "against_sha256": sha256(against),
        "moved_column": moved_col, "suffix": suffix,
        "guard": {"quantile": G.Q if guard_cut is not None else None,
                  "cut_s_for_the_ranking_serve": guard_cut,
                  "cut_source": ("--guard-cut, the cut that wrote the candidate"
                                 if args.get("--guard-cut") else
                                 "analysis/guard-arms.json, this re-price's cut"),
                  "cut_in_guard_arms_json_now": (
                      float(json.loads((EXP / "analysis/guard-arms.json").read_text())
                            ["guard"]["cut_s_for_the_ranking_serve"])
                      if (EXP / "analysis/guard-arms.json").exists() else None),
                  "rows_held": int(out.guard_held_by_exp_046.sum())},
        "columns": {
            "MVT_ID_mvt": "the ranking key",
            "obs_has_clock": "the class the desk named; the complement to serve is ~obs_has_clock",
            "served_by_exp_046": "obs_has_clock & obs_taxi.notna() & ~noac (arms.served_class)",
            "noac": "the exp-033 hold; these rows move in neither lane",
            moved_col: ("the rows the shipped candidate actually moves off the file "
                        "it is composed against (--against); a held airport is NOT "
                        "moved, so this is narrower than served_by_exp_046, and on a "
                        "candidate composed on a previous head it also carries the rows "
                        "REVERTED to the price base, which this lane does not occupy"),
            "occupied_by_exp_046": (
                "the rows this lane actually writes its own head on: "
                "served_by_exp_046 AND not held at the base. THIS is the column a "
                "later serve in either lane must complement -- `moved` includes rows "
                "this lane gives up (a revert), and `served_by_exp_046` includes rows "
                "it holds, so neither leaves every row owned exactly once"),
            "guard_held_by_exp_046": (
                "occupied rows the tail guard returned to the price base, so this lane "
                "decided their value and the value is the base. Empty on an unguarded "
                "candidate. A later serve reads it to tell a guard-held row from a "
                "vector that disagrees with its file (round 5 FALLBACK, finding 6)"),
            "written_by_exp_046": (
                "occupied AND this lane actually wrote a value here in THESE bytes: "
                "occupied_by_exp_046 minus the rows the class gained AFTER the candidate "
                "was built. Read this to ask what is in the file; read "
                "occupied_by_exp_046 to ask who owns the row next"),
            "chain_mode_4": (
                "the row entered this lane's class through the ungated turnaround pass at "
                "the section 28.2 gate (match_mode 4). Always False on a vector emitted "
                "without --ungated"),
        },
        "gained_modes": gained_modes,
        "gained_modes_what": (
            "the match modes whose rows this lane's class holds and these bytes do NOT "
            "carry a value for. Empty when the candidate serves every occupied row"),
        "months": {},
        "totals": {},
    }
    cols = [c if c != "moved_by_v13" else moved_col for c in COLS]
    for m, g in out.groupby("rank_month"):
        p = OUTDIR / f"{m}{suffix}.parquet"
        g[cols].reset_index(drop=True).to_parquet(p, index=False)
        man["months"][str(m)] = {
            "path": str(p.relative_to(ROOT)),
            "sha256": sha256(p),
            "rows": int(len(g)),
            "obs_has_clock": int(g.obs_has_clock.sum()),
            "not_obs_has_clock": int((~g.obs_has_clock).sum()),
            "served_by_exp_046": int(g.served_by_exp_046.sum()),
            "noac": int(g.noac.sum()),
            moved_col: int(g[moved_col].sum()),
            "occupied_by_exp_046": int(g.occupied_by_exp_046.sum()),
            "guard_held_by_exp_046": int(g.guard_held_by_exp_046.sum()),
            "written_by_exp_046": int(g.written_by_exp_046.sum()),
            "chain_mode_4_rows": int(g.chain_mode_4.sum()),
            "chain_mode_4_occupied": int((g.chain_mode_4 & g.occupied_by_exp_046).sum()),
            "chain_mode_4_written": int((g.chain_mode_4 & g.written_by_exp_046).sum()),
            "by_airport": {str(a): {"rows": int(len(h)),
                                    "obs_has_clock": int(h.obs_has_clock.sum()),
                                    "served_by_exp_046": int(h.served_by_exp_046.sum()),
                                    moved_col: int(h[moved_col].sum())}
                           for a, h in g.groupby("airport")},
        }
    man["totals"] = {
        "rows": int(len(out)),
        "obs_has_clock": int(out.obs_has_clock.sum()),
        "not_obs_has_clock": int((~out.obs_has_clock).sum()),
        "served_by_exp_046": int(out.served_by_exp_046.sum()),
        "noac": int(out.noac.sum()),
        moved_col: int(out[moved_col].sum()),
        "occupied_by_exp_046": int(out.occupied_by_exp_046.sum()),
        "guard_held_by_exp_046": int(out.guard_held_by_exp_046.sum()),
        "written_by_exp_046": int(out.written_by_exp_046.sum()),
        "class_gained_since_the_candidate": class_gained_since_the_candidate,
        "class_gained_since_the_candidate_what": (
            "occupied rows this lane's class did not contain when the candidate was "
            "written: the turnaround pass's mode-3 rows. They are OCCUPIED -- this lane "
            "owns them next -- and NOT written_by_exp_046, because these bytes predate "
            "them. The next serve must write every one of them"),
        "class_gained_carrying_another_lanes_value": gained_carrying_another_lanes_value,
        "class_gained_carrying_another_lanes_value_what": (
            "of those, the rows whose bytes are NOT the price base, because exp-047-queue "
            "moves them today. The handover queue's rider 2 names: this lane owns the row "
            "next, the other lane's value is in the file until the next serve writes it"),
        "occupied_rows_at_the_price_base_the_guard_does_not_hold": (
            class_moved_under_the_file),
        "occupied_rows_at_the_price_base_why": (
            "the class moved under bytes already written: this vector is emitted on a "
            "newer matcher than the candidate was built with, so it carries rows the "
            "candidate could not have served"),
        "occupied_rows_whose_value_did_not_change": occupied_not_moved,
        # **THE ROWS THE CHAIN ADDS** (desk, 2026-09-09).  `match_mode` 4 is the ungated
        # turnaround pair the +-60 s clause refuses and the section 28.2 gate keeps; it
        # exists only in a vector emitted with `--ungated`, and is 0 in every other.  The
        # split matters to the composition: under the SHIPPED matcher `122c7d84` these
        # rows carry no clock, so `exp-047-queue` holds them today and the bytes this
        # candidate is composed over carry queue's value on most of them.
        "chain_mode_4_rows": int(out.chain_mode_4.sum()),
        "chain_mode_4_occupied": int((out.chain_mode_4 & out.occupied_by_exp_046).sum()),
        "chain_mode_4_written": int((out.chain_mode_4 & out.written_by_exp_046).sum()),
        "chain_mode_4_noac": int((out.chain_mode_4 & out.noac).sum()),
        "chain_mode_4_held_at_the_base": int(
            (out.chain_mode_4 & out.served_by_exp_046 & ~out.occupied_by_exp_046).sum()),
        "chain_mode_4_what": (
            "the ungated chain rows the section 28.2 gate keeps (step 0 at 62.405 s, "
            "|P - AOBT| <= 1,299 s, runway agreement). They are in THIS lane's class and "
            "in exp-047-queue's complement under the shipped matcher, which is the "
            "section 14 handshake this vector exists to settle"),
        "airports_held_at_the_base": held,
        "complement_of_the_named_class": int((~out.obs_has_clock).sum()),
        "complement_that_leaves_every_row_once": int((~out.occupied_by_exp_046).sum()),
        "complement_that_leaves_every_row_once_is": (
            "~occupied_by_exp_046 -- every row this lane does not write its own head on, "
            "including the held airport and any row reverted to the price base"),
        "superseded_reading_do_not_use": {
            "expression": "~%s & ~noac" % moved_col,
            "rows": int((~out[moved_col] & ~out.noac).sum()),
            "why": ("it strands every row this lane REVERTED to the price base: those "
                    "rows are changed and so are excluded from the complement, but no "
                    "lane owns them (round 4 FALLBACK, finding 4)"),
        },
    }
    # **THE TWO ASSERTIONS THE SERVE RECORD MAKES POSSIBLE** (desk, 2026-09-09).  A vector
    # emitted for a candidate THIS serve wrote must agree with the serve on two counts, and
    # both defects of 2026-09-09 broke exactly one of them: the v20 guard flags left
    # `guard_held` at 0 against the serve's 1,573, and the v22 `--gained-modes` default left
    # `written` at 193,077 against an occupancy of 196,256.  Neither raised anything.
    if serve_rec is not None:
        want_held = int(serve_rec["gate"]["tail_guard"]["rows_returned_to_the_base"])
        got_held = int(man["totals"]["guard_held_by_exp_046"])
        if got_held != want_held:
            raise SystemExit(
                f"guard_held_by_exp_046 is {got_held} but the serve returned {want_held} "
                f"rows to the base ({Path(from_serve).name}, gate.tail_guard."
                "rows_returned_to_the_base). The vector and the file disagree about the "
                "guard -- this is the section 35.4 defect.")
        occ_n = int(man["totals"]["occupied_by_exp_046"])
        wrt_n = int(man["totals"]["written_by_exp_046"])
        if wrt_n != occ_n:
            raise SystemExit(
                f"written_by_exp_046 is {wrt_n} but occupied_by_exp_046 is {occ_n} "
                f"({occ_n - wrt_n} occupied rows unwritten). A candidate this serve just "
                "wrote carries every occupied row, so --gained-modes is wrong for it "
                "(pass --gained-modes= empty). Emitting a vector for OLDER bytes whose "
                "class has since moved is legitimate -- do it without --from-serve, which "
                "is the flag that asserts 'these bytes are this serve's'.")
        man["asserted_against_the_serve"] = {
            "serve_record": str(Path(from_serve).resolve().relative_to(ROOT)),
            "candidate_sha256": serve_rec["sha256"],
            "guard_held_equals_rows_returned_to_the_base": want_held,
            "written_equals_occupied": occ_n,
        }

    (EXP / f"analysis/class-manifest{suffix}.json").write_text(
        json.dumps(man, indent=2, sort_keys=True))
    print(json.dumps(man["totals"], indent=2, sort_keys=True))
    for m, r in man["months"].items():
        print(m, r["path"], r["rows"], r["sha256"][:16])


if __name__ == "__main__":
    main()
