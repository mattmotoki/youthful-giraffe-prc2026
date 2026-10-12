"""The ablation ladder: fold numbers for every rung, configs for the buildable ones.

Two ladders, because they answer different questions and only one of them can
be uploaded.

**The swap ladder** walks blend-4 to blend-5 one swap at a time and reproduces
exp-012's composition claim (`RESULTS.md`, "blend-5's member set, and the three
swaps compose").  Every rung is computable, because every blend-4 out-of-fold
address survived in git.  None of the intermediate rungs can be SERVED: the
ranking files of `T-tailv2` (`e9d97994…`) and `L2-hub` (`29adfc27…`) lived only
in `artifacts/` and are gone, so any rung holding either is fold-only.

**The leave-one-out ladder** drops one member at a time from blend-5.  Every
rung is buildable, because blend-5's seven ranking files are all published.
These are the uploads that locate a member whose fold gain does not transfer.

Weights for a served rung are fitted on ALL THREE folds -- exp-012's rule, "the
rotation scores the rule, it does not choose the weights" -- and the rotation
beside them is the honest out-of-sample number.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames as F  # noqa: E402
import rule as R  # noqa: E402

B4 = ("T-tailv2", "C2-cat", "L2-hub", "D1-winter", "F1-fresh", "N1-nmtail")
B5 = ("T-tailv4", "C2-cat", "L2-hub8725", "D1-winter", "F1-fresh", "V3-allin", "N1-nmtail")

SWAP = [
    ("blend-4", B4),
    ("+V3-allin", ("T-tailv2", "C2-cat", "L2-hub", "D1-winter", "F1-fresh", "V3-allin",
                   "N1-nmtail")),
    ("+V3-allin +tail v4", ("T-tailv4", "C2-cat", "L2-hub", "D1-winter", "F1-fresh", "V3-allin",
                            "N1-nmtail")),
    ("+V3-allin +tail v4 +hub8725 (= blend-5)", B5),
]
#: One swap at a time off blend-4, so each gain is attributable on its own.
SINGLE = [
    ("blend-4 +V3-allin only", SWAP[1][1]),
    ("blend-4 +tail v4 only", ("T-tailv4", "C2-cat", "L2-hub", "D1-winter", "F1-fresh",
                               "N1-nmtail")),
    ("blend-4 +hub8725 only", ("T-tailv2", "C2-cat", "L2-hub8725", "D1-winter", "F1-fresh",
                               "N1-nmtail")),
]
LOO = [(f"blend-5 -{n}", tuple(x for x in B5 if x != n)) for n in B5]


def servable(names) -> bool:
    return all(n in F.RANKING for n in names)


def run(names, cache: dict) -> dict:
    key = tuple(names)
    if key not in cache:
        data = {f: F.fold_frame(f, names) for f in F.FOLDS}
        res = R.rotate(data, names)
        res["shipped_weights"] = R.fit_hybrid(
            pd.concat([data[f] for f in F.FOLDS], ignore_index=True), names)
        res["decomp"] = {f: R.decomp(data[f]["y_true"].to_numpy("float64"), res[f]["pred"],
                                     data[f]) for f in F.FOLDS}
        cache[key] = res
    return cache[key]


def table(rows, cache, ref=None) -> pd.DataFrame:
    recs = []
    for label, names in rows:
        r = run(names, cache)
        rec = {"config": label, "n_members": len(names), "servable": servable(names)}
        for f in F.FOLDS:
            rec[f] = round(r[f]["rmse_clean"], 2)
        rec["clean board"] = round(r["board_clean"], 3)
        rec["predict"] = round(r["predict"], 2)
        recs.append(rec)
    T = pd.DataFrame(recs)
    if ref is not None:
        T["vs " + ref] = (T["clean board"] - T.loc[T.config == ref, "clean board"].iloc[0]).round(3)
    return T


def main() -> None:
    cache: dict = {}
    out = {}
    swap = table(SWAP, cache, ref="blend-4")
    single = table(SINGLE + [SWAP[0]], cache, ref="blend-4")
    loo = table(LOO + [("blend-5", B5)], cache, ref="blend-5")

    print("=== swap ladder: blend-4 to blend-5, clean folds (exp-012 RESULTS table) ===")
    print(swap.to_string(index=False))
    print("\n=== one swap at a time off blend-4 ===")
    print(single.to_string(index=False))
    print("\n=== leave one out of blend-5 (every rung servable) ===")
    print(loo.to_string(index=False))

    for name, T in (("swap", swap), ("single", single), ("loo", loo)):
        T.to_csv(F.EXP / "analysis" / f"ladder_{name}.csv", index=False)

    # configs for every servable rung, weights fitted on all three folds
    cfgdir = F.EXP / "configs"
    cfgdir.mkdir(exist_ok=True)
    built = []
    for label, names in SWAP + SINGLE + LOO:
        if not servable(names):
            continue
        r = run(names, cache)
        slug = ("blend-5" if names == B5 else
                label.replace("blend-5 -", "b5-drop-").replace("blend-4 ", "b4-")
                     .replace("+", "plus-").replace(" ", "_"))
        cfg = {
            "note": f"exp-026 rebuild of `{label}` on published members only.",
            # the name `rebuild_blend.py --name` stages this config under, and the
            # key `finalise.py` matches a staged vector back to its config by.
            # Without it the end-to-end cross-check against
            # `prc.blend.build_candidate` silently never runs.
            "_stage_name": slug,
            "ranking": {n: F.RANKING[n] for n in names},
            "mode": "hybrid: nnls in nm_late, nnls elsewhere",
            "weights": r["shipped_weights"],
            "floor": 60.0, "guard_bounds": list(F.GUARD), "late_cut": 1800.0,
            "prepare_members": True, "blend_version": 5,
            "rotated": {f: round(r[f]["rmse"], 4) for f in F.FOLDS},
            "rotated_clean": {f: round(r[f]["rmse_clean"], 4) for f in F.FOLDS},
            "board_clean": round(r["board_clean"], 4), "predict": round(r["predict"], 4),
            "member_sha256": {n: F.sha(F.RANKING[n]) for n in names},
        }
        p = cfgdir / f"config-{slug}.json"
        p.write_text(json.dumps(cfg, indent=2))
        built.append((label, p.name, cfg["predict"]))
    print("\n=== servable configs written ===")
    for label, fn, pr in built:
        print(f"  {label:42s} {fn:34s} predict {pr:.2f}")

    # COMMON.md wants the decomposition on EVERY measured row, not only the two
    # ends of the ladder.  Review round 1 listed the intermediate and
    # leave-one-out rungs as missing; every rung is already in `cache`, so this
    # costs nothing but the writing.
    out["decomp"] = {lab: {f: run(nm, cache)["decomp"][f] for f in F.FOLDS}
                     for lab, nm in SWAP + SINGLE + LOO}
    (F.EXP / "analysis" / "ladder_decomp.json").write_text(json.dumps(out, indent=2))
    rec = []
    for lab in out["decomp"]:
        for f in F.FOLDS:
            rec.append(dict(config=lab, fold=f, **out["decomp"][lab][f]))
    pd.DataFrame(rec).round(4).to_csv(F.EXP / "analysis" / "ladder_decomp.csv", index=False)
    print(f"\n=== required decomposition, all {len(out['decomp'])} rungs "
          f"(analysis/ladder_decomp.csv) ===")
    print(pd.DataFrame(rec).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
