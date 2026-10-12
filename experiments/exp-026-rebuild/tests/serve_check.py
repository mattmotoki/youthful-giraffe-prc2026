"""End-to-end check of stack-5's SERVING path, against the rotation it claims.

Review round 1 finding 2 was "there is no stack-5 serving implementation".
`code/serve_stack5.py` is that implementation; this file is the evidence that it
does what section 4 says it does.  Six checks, each one a number a reader can
re-derive:

  1. shape          the candidate is the template's 344,841 rows, in the
                    template's id order, float64, finite, none under the floor
  2. anchor         the serve-time anchor is blend-5.  Checked against the
                    blend-5 vector that matches the SHIPPED v2 object to
                    1.5e-10 s, so the anchor is validated against a real
                    artifact and not against this experiment's own arithmetic
  3. keep-noac      the 388 `noac` rows are served from that anchor exactly
  4. stacker        the five seeds are fitted on all three folds (one row per
                    fold row) with the ROTATED anchor as the residual base, so
                    no fitting row's anchor saw its own labels
  5. guard          every served row satisfies the guard invariant, and the
                    floor holds after it
  6. fit/serve      the one convention difference between the rotation and the
                    serve path, measured rather than asserted: the rotation
                    floors its output and does not guard it, `build_candidate`
                    guards it.  This prices that difference on the fold rows.

    .venv/bin/python experiments/exp-026-rebuild/tests/serve_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import frames as F  # noqa: E402
import regime_table as RT  # noqa: E402
import rule as R  # noqa: E402
import stack5 as S  # noqa: E402
# The repository root goes at the END: `code/` must stay ahead of it so
# a module at the root cannot shadow one of this experiment's own
# (round 4 -- a root-level `verify.py` would have replaced the gate).
sys.path.append(str(F.ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

CAND = F.ROOT / "submissions/candidates/exp-026-stack-5.parquet"
BLEND5 = F.ROOT / "submissions/candidates/exp-026-blend-5.parquet"
SHIPPED = F.ROOT / "submissions/uploaded/youthful-giraffe_v2.parquet"
fails: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        fails.append(label)


def main() -> None:
    names = S.B5
    gates = RT.load()
    S5 = pd.read_parquet(CAND)
    y = S5[D.TARGET].to_numpy("float64")

    print("1. shape")
    T = D.load_submitting()
    check(len(S5) == len(T) == 344841, "row count is the template's", f"{len(S5):,}")
    check(bool((S5["MVT_ID_mvt"].to_numpy() == T["MVT_ID_mvt"].to_numpy("float64")).all()),
          "id order is the template's")
    check(y.dtype == np.float64 and bool(np.isfinite(y).all()), "float64 and finite")
    check(int((y < B.FLOOR - 1e-9).sum()) == 0, "nothing under the floor",
          f"min {y.min():.4f}")

    print("2. the serve-time anchor is blend-5")
    rk = __import__("serve_stack5").ranking_frame(names, gates)
    fit = pd.concat([F.fold_frame(f, names) for f in F.FOLDS], ignore_index=True)
    anchor_rk = R.apply_hybrid(rk, names, R.fit_hybrid(fit, names))
    a_served = B.apply_residual_guard(np.maximum(anchor_rk, B.FLOOR),
                                      rk["d_takeoff_aobt"].to_numpy("float64"),
                                      *F.GUARD, B.FLOOR)
    b5 = pd.read_parquet(BLEND5).set_index("MVT_ID_mvt")[D.TARGET].reindex(
        rk["MVT_ID_mvt"].to_numpy("float64")).to_numpy("float64")
    d = np.abs(a_served - b5).max()
    check(d < 1e-8, "anchor == the staged blend-5 vector", f"max abs diff {d:.3e} s")
    if SHIPPED.exists():
        sh = pd.read_parquet(SHIPPED).set_index("MVT_ID_mvt")[D.TARGET].reindex(
            rk["MVT_ID_mvt"].to_numpy("float64")).to_numpy("float64")
        ds = np.abs(a_served - sh).max()
        check(ds < 1e-6, "anchor == the SHIPPED youthful-giraffe_v2 object",
              f"max abs diff {ds:.3e} s over {len(sh):,} rows")

    print("3. keep-noac serves the anchor on the noac rows")
    reg = gates.set_index("MVT_ID_mvt").reindex(
        S5["MVT_ID_mvt"].to_numpy("float64"))["regime"].to_numpy()
    noac = reg == "noac"
    check(int(noac.sum()) == 388, "388 noac rows", f"{int(noac.sum())}")
    check(bool((y[noac] == b5[noac]).all()), "every noac row equals blend-5 exactly",
          f"{int((y[noac] != b5[noac]).sum())} differ")
    check(int((y[~noac] != b5[~noac]).sum()) > 300000, "the other rows do move",
          f"{int((y[~noac] != b5[~noac]).sum()):,} of {int((~noac).sum()):,}")

    print("4. the candidate rebuilds, and the stacker is fitted on all three folds")
    # Round 2: counting rows and seed constants is self-confirming. Rebuild the
    # served vector and compare it to the published file -- the same gate
    # `finalise.py` runs, called here so the test and the gate cannot drift.
    import verify as V
    try:
        vr = V.verify("stack-5", CAND)
        check(True, "the published file passes code/verify.py",
              f"rebuild max abs diff {vr['max_abs_diff']:.3g} s, "
              f"noac vs anchor {vr['noac_vs_anchor_max_abs_diff']:.3g} s, "
              f"{vr['non_noac_rows_moved']:,} non-noac rows moved")
    except V.Failed as e:
        check(False, "the published file passes code/verify.py", str(e))
    a_fit = np.concatenate([pd.read_parquet(F.EXP / "oof" / f"stack5-{f}.parquet")
                            ["y_pred_anchor"].to_numpy("float64") for f in F.FOLDS])
    check(len(a_fit) == len(fit), "one fitting row per fold row", f"{len(fit):,}")
    per = {f: len(F.fold_frame(f, names)) for f in F.FOLDS}
    check(sum(per.values()) == len(fit), "all three folds present", str(per))
    rot = {f: pd.read_parquet(F.EXP / "oof" / f"stack5-{f}.parquet") for f in F.FOLDS}
    own = max(float(np.abs(R.apply_hybrid(F.fold_frame(f, names), names,
                                          R.fit_hybrid(fit, names))
                           - rot[f]["y_pred_anchor"].to_numpy("float64")).max())
              for f in F.FOLDS)
    check(own > 1.0, "the fitting anchor is ROTATED, not the all-fold fit",
          f"they differ by up to {own:.1f} s, as a rotation must")
    check(len(S.SEEDS) == 5, "five seeds", str(S.SEEDS))

    print("5. the guard is applied, and its invariant holds")
    # Round 2: the invariant alone does not prove the guard ran. Build the same
    # vector WITHOUT the output guard and show it violates the bounds, so the
    # check has something to fail on.
    import serve_stack5 as SS
    b = SS.build(gates, "keep-noac")
    au = b["d_takeoff_aobt"]
    oku = np.isfinite(au)
    lo0, hi0 = F.GUARD
    ru = au[oku] - b["pred_unguarded"][oku]
    n_bind = int(((ru < lo0) | (ru > hi0)).sum())
    chg = np.abs(b["pred"] - b["pred_unguarded"])
    check(n_bind == 0, "the guard is inert on this candidate's output",
          f"0 rows bind; residual range before the guard "
          f"[{ru.min():.1f}, {ru.max():.1f}] sits inside [{lo0}, {hi0}]; "
          f"{int((chg > 0).sum())} values change, largest {chg.max():.2e} s, "
          f"which is `a - clip(a - y)` round-trip noise")

    # Round 3, stated against my own test: BECAUSE the guard is inert here,
    # nothing in this file can tell a build that calls the guard from one that
    # does not -- deleting the call would leave every check above passing. An
    # earlier version of this section claimed the control "can fail". It cannot,
    # on this candidate. What CAN be tested is that the guard implementation
    # works when it is needed, so that is what the positive control does: feed
    # it a vector that violates the bounds and require it to fix it.
    bad = b["pred_unguarded"].copy()
    idx = np.flatnonzero(oku)[:1000]
    bad[idx] = au[idx] - (hi0 + 5000.0)          # 5000 s past the upper bound
    fixed = B.apply_residual_guard(bad, au, lo0, hi0, B.FLOOR)
    rf = au[oku] - fixed[oku]
    check(int(((rf < lo0 - 1e-6) | (rf > hi0 + 1e-6)).sum()) == 0
          and not np.array_equal(fixed, bad),
          "positive control: the guard repairs a vector that violates the bounds",
          f"1000 rows planted 5000 s past the bound, "
          f"{int((np.abs(fixed - bad) > 1e-6).sum())} repaired, none left outside")
    a = gates.set_index("MVT_ID_mvt").reindex(
        S5["MVT_ID_mvt"].to_numpy("float64"))["d_takeoff_aobt"].to_numpy("float64")
    ok = np.isfinite(a)
    r = a[ok] - y[ok]
    lo, hi = F.GUARD
    check(bool(((r >= lo - 1e-6) & (r <= hi + 1e-6)).all()),
          "implied residual inside the guard bounds",
          f"[{r.min():.1f}, {r.max():.1f}] against [{lo}, {hi}]")

    print("6. the one fit/serve convention difference, priced AND gated")
    print("     the rotation floors its output; build_candidate also guards it.")
    for f in F.FOLDS:
        o = rot[f]
        yt = o["y_true"].to_numpy("float64")
        p = o["y_pred"].to_numpy("float64")
        g = F.fold_frame(f, names)
        af = g.set_index("MVT_ID_mvt").reindex(
            o["MVT_ID_mvt"].to_numpy("float64"))["d_takeoff_aobt"].to_numpy("float64")
        pg = B.apply_residual_guard(p, af, *F.GUARD, B.FLOOR)
        r0 = D.rmse_both(yt, p, o["MVT_ID_mvt"].to_numpy("float64"), f)["rmse_clean"]
        r1 = D.rmse_both(yt, pg, o["MVT_ID_mvt"].to_numpy("float64"), f)["rmse_clean"]
        print(f"     {f:9s} clean {r0:9.4f} -> {r1:9.4f}  ({r1-r0:+.4f} s, "
              f"{int((pg != p).sum())} rows touched)")
        # Round 2: this printed and never called check(), so it could not fail.
        check(abs(r1 - r0) < 5e-4, f"{f}: the convention difference is under 0.0005 s",
              f"{r1 - r0:+.6f} s")

    print("\n" + ("PASS" if not fails else f"FAIL: {fails}"))
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
