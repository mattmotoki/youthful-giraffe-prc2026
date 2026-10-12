"""The PASS-ONLY increment: the turnaround pass alone, with per-fold intervals.

Review round 6, finding 2.  The v18 serve priced its increment against
`work/d-blockmove-guard995.npz` and the record claimed that vector was the served arm on
the previous matcher at this serve's pins, so that the increment was "the matcher's effect
and nothing else".  **That claim was false.**  The vector is the arm on matcher
`eded54e5` fitted at july 287 / december 288 airport-days (07:30, a671e65).  Its increment,
+0.1182 = 9.29855 - 9.18036, therefore carries the rebuild and the population re-writes as
well as the pass, and the desk's decomposition puts about +0.054 on the matcher at
identical pins and +0.064 on everything else.

This isolates the pass.  The head is REFITTED -- same arm, same rotation, seed, cross-fit,
floor, guard quantile and hold -- on the surviving `6fdf3951` block caches:

    work/block-2025-01-cb4d6289cb421aa2-6fdf3951.parquet     january  pin cb4d6289
    work/block-2025-07-8179f74417c8cb99-6fdf3951.parquet     july     pin 8179f744
    work/block-2025-12-34a155e40aac9ff6-6fdf3951.parquet     december pin 34a155e4

**July and january carry the SAME pin as the block this serve reads**, so on those two
folds the only thing that differs is the matching function, and the increment there is the
pass and nothing else.  December's cache is at the pin BEFORE the re-price added nine
airport-days (all 2025-11-30), so december mixes the pass with those days and is reported
as mixed.  Since `base.SHARES` is july_loo and january only, **the BOARD number is exactly
attributable to the pass**; december enters the fold table and not the board.

Nothing is cached, registered or served: the block caches are read, never written, and the
refitted vector goes to its own file that no gate reads.

    OMP_NUM_THREADS=3 .venv/bin/python experiments/exp-046-surfacefit/code/passonly.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent
ROOT = EXP.parent.parent

import serve as S                                                     # noqa: E402
import arms as A                                                      # noqa: E402
import base as BA                                                     # noqa: E402
import block as B                                                     # noqa: E402
import guard as G                                                     # noqa: E402

ARM = "c_lgb_shipped@leaves_255x300_lr050+guard995"
SETTING = {"num_leaves": 255}
HOLD = ("LFPG",)
HEAD_MATCHER = "6fdf3951"
CACHES = {"2025-01": "work/block-2025-01-cb4d6289cb421aa2-6fdf3951.parquet",
          "2025-07": "work/block-2025-07-8179f74417c8cb99-6fdf3951.parquet",
          "2025-12": "work/block-2025-12-34a155e40aac9ff6-6fdf3951.parquet"}
#: the pin each cache carries, against the pin this serve reads
PINS = {"2025-01": ("cb4d6289cb421aa2", "cb4d6289cb421aa2"),
        "2025-07": ("8179f74417c8cb99", "8179f74417c8cb99"),
        "2025-12": ("34a155e40aac9ff6", "d35b7171d7062341")}
HEAD_NPZ = EXP / "work/d-passonly-6fdf3951-guard995.npz"
OUT = EXP / "analysis/pass-only.json"


def head_panel() -> dict[str, pd.DataFrame]:
    """`panel_with_block` with the block read from the pinned 6fdf3951 caches.

    `month_block` is replaced rather than copied, so every derivation after the read --
    the noac hold, `usable`, `obs_wide`, the neighbour block -- is the SAME code the fold
    table and the serve run, and this file cannot drift from it.  The replacement refuses
    a month it has no cache for instead of falling through to a rebuild, which would
    silently re-derive the block with the CURRENT matcher and write it under the old
    matcher's name.
    """
    real = A.month_block

    def from_cache(month: str, prefix_months: list[str], rule: str | None = None):
        if month not in CACHES:
            raise SystemExit(f"passonly: no {HEAD_MATCHER} cache for {month}")
        p = EXP / CACHES[month]
        if not p.exists():
            raise SystemExit(f"passonly: {p.name} is gone; the pass-only head cannot be "
                             "refitted and no substitute is acceptable")
        return pd.read_parquet(p)

    A.month_block = from_cache
    try:
        return A.panel_with_block(BA.load())
    finally:
        A.month_block = real


def main() -> None:
    t0 = time.time()
    for m, c in CACHES.items():
        if not (EXP / c).exists():
            raise SystemExit(f"passonly: {c} is missing")
    out = {"what": __doc__.strip().splitlines()[0], "arm": ARM,
           "head_matcher": HEAD_MATCHER,
           "new_matcher": A.matcher_sha(),
           "caches": CACHES,
           "pins": {m: {"head_cache": a, "this_serve": b, "identical": a == b}
                    for m, (a, b) in PINS.items()},
           "board_folds": sorted(BA.SHARES),
           "attribution": ("july_loo and january carry identical pins, and base.SHARES is "
                           "those two folds only, so the BOARD increment is the pass "
                           "alone; december's cache predates nine airport-days added by "
                           "the re-price and its fold row is the pass PLUS those days")}

    pan_h = head_panel()
    out["head_panel"] = {
        "rows_by_fold": {f: int(len(pan_h[f])) for f in A.FOLDS},
        "usable_by_fold": {f: int(pan_h[f].usable.sum()) for f in A.FOLDS},
        "mode3_rows_by_fold": {f: int((pan_h[f].match_mode.fillna(-1) == 3).sum())
                               for f in A.FOLDS}}
    assert all(v == 0 for v in out["head_panel"]["mode3_rows_by_fold"].values()), \
        "the head block already contains turnaround rows: it is not the pre-pass matcher"
    print("head panel usable", out["head_panel"]["usable_by_fold"],
          "mode3", out["head_panel"]["mode3_rows_by_fold"], flush=True)

    # --- refit the served arm on the head panel, exactly as blockmove.py fitted it ---
    ship, scols = A.shipped_cols(pan_h)
    obs_cols = [c for c in B.FEATURES if c not in ("stand_lat", "stand_lon")]
    d_by_fold = {}
    for f in A.FOLDS:
        others = [g for g in A.FOLDS if g != f]
        fit = pd.concat([pan_h[g][pan_h[g].usable] for g in others], ignore_index=True)
        sco = pan_h[f]
        fit = fit.join(pd.concat([ship[g] for g in others])
                       .reindex(fit.MVT_ID_mvt.to_numpy("float64")).reset_index(drop=True))
        sco = sco.join(ship[f].reindex(sco.MVT_ID_mvt.to_numpy("float64"))
                       .reset_index(drop=True))
        d, _ = A.arm_lgb(fit, sco, list(obs_cols) + scols, params=SETTING)
        d = A.apply_floor(pan_h[f].pbase.to_numpy("float64"), d)
        assert not np.any(d[pan_h[f].noac.to_numpy(bool)]), f"{f}: a noac row moved"
        assert not np.any(d[~pan_h[f].usable.to_numpy(bool)]), \
            f"{f}: a row outside the with-clock class moved"
        d_by_fold[f] = d
        print(f"  fitted {f} ({time.time()-t0:.0f}s)", flush=True)

    # the guard: the QUANTILE is frozen, the cut is a quantile of the fitting months, so
    # the head's cut is derived on the head's own panel -- the same rule, its own block
    cuts = {f: G.cut(pan_h, [g for g in A.FOLDS if g != f]) for f in A.FOLDS}
    guarded = {}
    for f in A.FOLDS:
        v = d_by_fold[f].astype("float64").copy()
        v[G.mask(pan_h[f].pbase.to_numpy("float64"), pan_h[f].obs_taxi, cuts[f])] = 0.0
        guarded[f] = v
    np.savez_compressed(HEAD_NPZ, **guarded)
    out["head_guard_cut_s_by_scored_fold"] = {f: round(c, 2) for f, c in cuts.items()}
    out["head_npz"] = str(HEAD_NPZ.relative_to(ROOT))

    # the head's own fold gate, with the hold applied as the serve applies it
    rng = np.random.default_rng(A.SEED)
    held = {f: np.where(np.isin(pan_h[f].airport.to_numpy().astype(str), list(HOLD)),
                        0.0, guarded[f]) for f in A.FOLDS}
    rec_h = BA.price(pan_h, held, rng)
    out["head_fold_gate_on_its_own_block"] = {
        "all_row_gain_s": rec_h["board_all_row"]["gain_s"],
        "all_row_ci95": rec_h["board_all_row"]["interval"]["ci95"],
        "clean_gain_s": rec_h["board_clean"]["gain_s"],
        "every_fold_interval_clear": bool(rec_h["every_fold_interval_clear"]),
        "by_fold": {f: rec_h["by_fold"][f]["all_row"] for f in A.FOLDS}}
    print(f"head on its own block: {rec_h['board_all_row']['gain_s']:+.5f}", flush=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    # --- the increment, through the reviewed code path -----------------------
    inc = S.increment_gate(ARM, head_arm=ARM, held=HOLD, head_npz=HEAD_NPZ)
    out["pass_only_increment"] = inc
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))

    b = inc["board_all_row_gain_s"]
    ci = inc["board_all_row_interval_ci95"]
    print(f"\nPASS-ONLY increment  board all-row {b:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]  "
          f"passes {inc['passes']}  every fold clear {inc['every_fold_interval_clear']}")
    print(f"{'fold':10s} {'rows':>8} {'moved':>8} {'s2/row':>10} {'ci95':>26} "
          f"{'rmse head':>11} {'rmse new':>11}  pin")
    for f in A.FOLDS:
        r = inc["by_fold"][f]["all_row"]
        m = A.FOLD_MONTH[f]
        same = "identical" if PINS[m][0] == PINS[m][1] else "MIXED: +9 airport-days"
        print(f"{f:10s} {r['rows']:>8} {r['rows_moved']:>8} {r['mse_gain_s2']:>+10.2f} "
              f"[{r['ci95'][0]:>+10.2f},{r['ci95'][1]:>+10.2f}] "
              f"{inc['by_fold'][f]['rmse_head']:>11.4f} {inc['by_fold'][f]['rmse_new']:>11.4f}"
              f"  {same}")
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=float))
    print("written to", OUT.relative_to(ROOT), f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
