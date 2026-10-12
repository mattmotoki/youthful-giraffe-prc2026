"""exp-060-coldprior: every draw's board RMSE, winning sham and gain, for EVERY registered arm.

`code/banddraws60.py` covered `true`, the ten shams and `zero` only, because `bound60_r1.py`'s
band did.  Review round 2 finding 1 is right that the assignment asks for every arm.  This job
reads `bound60_r2.py`'s draws, which hold all seventeen, verifies every input artifact against the
hash its record carries (round 2, standards), and refuses unless it reproduces the band that
`analysis/bound-r2.json` records.

    .venv/bin/python experiments/exp-060-coldprior/code/banddraws60_r1.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e60 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402

BAND_ARMS = E.ARMS          # round 2 finding 1: every registered arm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    out = E.ANALYSIS / "banddraws60-r1.json"
    npz_p = E.ART / "band-r2-draws.npz"
    NO.refuse_existing([out, npz_p], a.overwrite)
    ref = E.load_json(E.ANALYSIS / "bound-r2.json")["winner_adjusted_three_day_band"]
    # round 2, standards: verify every input artifact against its recorded hash before use
    checked = {}
    for r in E.ORDER:
        rr = E.load_json(E.ANALYSIS / f"bound-r2-{r}.json")
        for key in ("band_artifact", "grid_artifact"):
            aa = rr[key]
            got = E.sha_file(E.ROOT / aa["path"])
            if got != aa["sha256"]:
                raise SystemExit(f"{aa['path']} is {got[:16]}, not the recorded {aa['sha256'][:16]}")
            checked[aa["path"]] = got
    d = {r: np.load(E.ART / f"bound-r2-draws-{r}.npz") for r in E.ORDER}
    rmse = {arm: np.sqrt(sum(s * d[r][f"{arm}_mse"] for r, s in E.BOARD.items()))
            for arm in BAND_ARMS}
    mat = np.stack([rmse[s] for s in E.SHAM_ARMS])
    win = mat.argmin(0)
    gain = mat.min(0) - rmse["true"]
    save = {f"board_rmse_{arm}": rmse[arm] for arm in BAND_ARMS}
    for r in E.ORDER:
        for arm in BAND_ARMS:
            save[f"{r}_{arm}_cell"] = d[r][f"{arm}_key"]
            save[f"{r}_{arm}_betas"] = d[r][f"{arm}_betas"]
        save[f"{r}_draw_dates"] = d[r]["draw_dates"]
    save["winning_sham_index"] = win
    save["winner_adjusted_gain_s"] = gain
    npz_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_p, **save)
    ci = E.ci(gain)
    rec = {"what": ("every draw's recombined board RMSE, selected cell and beta map, winning sham "
                    "and winner-adjusted gain, at block length 3 on the airport-adjusted level-B "
                    "bound (SPEC 6.3; review round 1 finding 3)"),
           "utc": E.utc(), "draws": E.BOOT, "block_length_days": 3, "arms": list(BAND_ARMS),
           "artifact": {"path": str(npz_p.relative_to(E.ROOT)), "sha256": E.sha_file(npz_p),
                        "arrays": sorted(save)},
           "reproduces_bound_r2": {
               "ci_block_3d": ci, "recorded": ref["ci_block_3d"],
               "equal": bool(ci == ref["ci_block_3d"]),
               "gain_positive_share": float((gain > 0).mean()),
               "recorded_positive_share": ref["gain_positive_share"],
               "gain_mean_s": float(gain.mean()), "recorded_mean_s": ref["gain_mean_s"]},
           "winning_sham_counts": {s: int(n) for s, n in
                                   zip(E.SHAM_ARMS, np.bincount(win, minlength=len(E.SHAM_ARMS)))},
           "verified_input_artifacts": checked, "code_sha256": E.code_shas()}
    ok = (rec["reproduces_bound_r2"]["equal"]
          and abs(rec["reproduces_bound_r2"]["gain_mean_s"] - ref["gain_mean_s"]) < 1e-12)
    rec["pass"] = bool(ok)
    NO.refuse_existing([out], a.overwrite)
    out.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(json.dumps({"pass": rec["pass"], "ci": ci, "recorded": ref["ci_block_3d"]}))
    if not ok:
        raise SystemExit("REFUSING: the per-draw board values do not reproduce the recorded band")
    print(f"written {out.relative_to(E.ROOT)} and {npz_p.relative_to(E.ROOT)}")


if __name__ == "__main__":
    main()
