"""The five RANKING members' provenance, reconstructed after their fit log was lost.

Review round 9, non-blocking (N2).  The five ranking members were fitted during a v21 dry
run whose log was overwritten by the next dry run, so the per-member census and timings
that `bagged_ranking_update` printed are gone; the serve that shipped loaded all five from
cache and honestly records `refitted_this_run: false` for each.  What is NOT gone is
everything that matters, because every part of it is either deterministic or on disk:

  * the DRAWS are deterministic -- `np.random.default_rng(seed)` over `arms.FOLDS` order --
    so they are recomputed here and checked against `analysis/bag-provenance.json`, which
    recorded them for the FOLD members.  Same seeds, same MOVED counts, same order, so the
    two sides perturb the same rows; that is the claim, and it is checked rather than said.
  * the VECTORS are on disk, and `analysis/serve-v21.json` recorded a sha for each at the
    moment it served.  Those are re-hashed here.
  * the two numbers the serve derived from them -- 196,256 rows moved by the ranking bag
    and a member disagreement of 13.705272749779187 s -- are recomputed from the five files
    and compared to what the record claims.

That is a stronger reconstruction than the lost log: a log says what happened, these
assertions say the files still are what the record describes.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/rankprov.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))

import arms as A                                                      # noqa: E402
import chainfold as CF                                                # noqa: E402
import classplacebo as CP                                             # noqa: E402
import rung511 as R                                                   # noqa: E402

SEEDS = [1, 2, 3, 4, 5]
SERVE = EXP / "analysis/serve-v21.json"
#: the fold-side record that carries per-seed draw COUNTS for the seeds `bag-provenance`
#: predates -- members 6-10 were fitted by `bag10.py`, which recorded their census but not
#: their row ids, so counts are the independent check available for them
FOLDCOUNTS = EXP / "analysis/bag-10.json"
FOLDPROV = EXP / "analysis/bag-provenance.json"
OUT = EXP / "analysis/rank-provenance.json"


def sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = {a.split("=", 1)[0]: a.split("=", 1)[1]
            for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    seeds = ([int(x) for x in args["--seeds"].split(",")] if "--seeds" in args else SEEDS)
    serve_p = Path(args.get("--serve", SERVE)).resolve()
    out_p = Path(args.get("--out", OUT)).resolve()
    rule = A.serving_rule()[0]
    blk = {}
    for f in A.FOLDS:                       # this order fixes the random stream
        m = A.FOLD_MONTH[f]
        b = CF.build_month(m, A.prefix_months_for(m), rule)
        R.assert_mode3(b, m, "shipped")
        blk[m] = b

    # --- the draws, recomputed and checked against the fold side's record ---------------
    fold_rows = json.loads(FOLDPROV.read_text())["rows"]
    fold_counts = (json.loads(FOLDCOUNTS.read_text())["members"]
                   if FOLDCOUNTS.exists() else {})
    draws, agree, census = {}, {}, {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        cen, cnt = {}, {}
        for m, b in blk.items():
            _, c = CP.placebo_block(b, R.THR, CF.GAP_CUT_S, CP.MOVED[m], rng)
            cen[m] = {k: sorted(int(x) for x in v) for k, v in c["rows_moved"].items()}
            cnt[m] = {k: v for k, v in c.items() if k != "rows_moved"}
        draws[seed], census[str(seed)] = cen, cnt
        if str(seed) in fold_rows:
            # the strong check: the fold side recorded this draw's ROW IDS
            for m, c in cen.items():
                for side in ("forced_in", "forced_out"):
                    agree[f"seed{seed} {m} {side} rows"] = (
                        fold_rows[str(seed)][m][side] == c[side])
        elif str(seed) in fold_counts and fold_counts[str(seed)].get("census"):
            # members 6-10: `bag10.py` recorded COUNTS, not row ids, so that is the
            # independent check available.  Weaker than the row-id check above, and said
            # so rather than presented as equal.
            fc = fold_counts[str(seed)]["census"]
            for m, c in cnt.items():
                for side in ("forced_in", "forced_out", "kept", "in_pool", "out_pool"):
                    agree[f"seed{seed} {m} {side} count"] = (
                        int(fc[m][side]) == int(c[side]))
        else:
            agree[f"seed{seed} UNCHECKED"] = False

    # --- the vectors on disk -----------------------------------------------------------
    rec_serve = json.loads(serve_p.read_text())["fit"]["info"]
    want = rec_serve["member_sha256"]
    files, vectors, shas = {}, [], {}
    for seed in seeds:
        p = EXP / f"work/d-rank-placebo{seed}-255.npz"
        if not p.exists():
            raise SystemExit(f"ranking member {seed} is missing: {p}")
        got = sha256(p)
        if got != want[str(seed)]:
            raise SystemExit(f"ranking member {seed} is {got[:12]}..., not the "
                             f"{want[str(seed)][:12]}... the serve recorded")
        shas[str(seed)] = got
        files[str(seed)] = str(p.relative_to(ROOT))
        vectors.append(np.load(p)["d"].astype("float64"))

    stack = np.vstack(vectors)
    mean = stack.mean(0)
    moved = int((mean != 0).sum())
    disagreement = float(np.sqrt(np.mean(stack.var(0))))
    idx = {s_: i for i, s_ in enumerate(seeds)}
    pairwise_identical = [f"{a}^{b}" for a, b in itertools.combinations(seeds, 2)
                          if np.array_equal(vectors[idx[a]], vectors[idx[b]])]

    rec = {
        "what": __doc__.strip().splitlines()[0],
        "why": ("the fit log was overwritten by the next dry run; nothing else was lost, "
                "because the draws are deterministic and the vectors are on disk"),
        "seeds": seeds,
        "serve_record": str(serve_p.relative_to(ROOT)),
        "draw_row_ids": {str(k): v for k, v in draws.items()},
        "checked_against": {
            "row_ids_from": str(FOLDPROV.relative_to(ROOT)) + " (seeds it carries)",
            "counts_from": str(FOLDCOUNTS.relative_to(ROOT)) + " (seeds 6-10; COUNTS only, "
                           "because bag10.py did not record their row ids -- a weaker "
                           "check than the row-id one, stated as such)"},
        "draws_reproduce_the_fold_members": all(agree.values()),
        "draws_check": agree,
        "the_two_sides_perturb_the_same_rows": all(agree.values()),
        "member_files": files,
        "member_sha256": shas,
        "member_sha256_matches_the_serve_record": True,
        "member_census_recomputed": census,
        "rows_moved_by_the_ranking_bag": moved,
        "rows_moved_recorded_by_the_serve": rec_serve["rows_moved_by_the_bag"],
        "rows_moved_agrees": bool(moved == rec_serve["rows_moved_by_the_bag"]),
        "member_disagreement_rms_s": disagreement,
        "member_disagreement_recorded_by_the_serve":
            rec_serve["member_disagreement_rms_s"],
        "member_disagreement_agrees": bool(
            abs(disagreement - rec_serve["member_disagreement_rms_s"]) < 1e-9),
        "pairs_that_are_identical_vectors": pairwise_identical,
        "all_members_are_distinct": not pairwise_identical,
        "all_five_members_are_distinct": not pairwise_identical,
    }
    rec["every_assertion_holds"] = bool(
        rec["draws_reproduce_the_fold_members"] and rec["rows_moved_agrees"]
        and rec["member_disagreement_agrees"] and rec["all_members_are_distinct"])
    out_p.write_text(json.dumps(rec, indent=2, sort_keys=True))
    for k in ("draws_reproduce_the_fold_members", "member_sha256_matches_the_serve_record",
              "rows_moved_agrees", "member_disagreement_agrees",
              "all_members_are_distinct", "every_assertion_holds"):
        print(f"{k:48s} {rec[k]}")
    print(f"rows moved {moved} | disagreement {disagreement:.6f} s")
    print("written to", out_p.relative_to(ROOT))


if __name__ == "__main__":
    main()
