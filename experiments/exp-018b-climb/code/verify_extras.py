"""Are the borrowed modules invariant to the ranking mask?

exp-018 earned the right to its own features by rebuilding a month masked and
unmasked and diffing: bit-identical, so no departure's own BLOCK or TAXI can
reach its features. The four modules borrowed from the repo get the same test
rather than a reading of their leakage guards, because `extras.py` hands them
UNMASKED frames for the training months and a MASKED frame for the validation
month, and any column that is not mask-invariant would carry a train/serve skew
into every arm of step 2.

  PYTHONPATH=. .venv/bin/python experiments/exp-018b-climb/code/verify_extras.py [month]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "exp-018-fresh" / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prc import data, phi_features as PF, surface_features as SF, day_state_features as DS
import extras as X                                                        # noqa: E402

FIT_MONTHS = ("2025-01", "2025-02")


def _cmp(name: str, a: pd.DataFrame, b: pd.DataFrame, cols) -> int:
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    bad = []
    for c in cols:
        x = a[c].to_numpy(dtype="float64")
        y = b[c].to_numpy(dtype="float64")
        if not np.allclose(x, y, equal_nan=True):
            d = np.abs(np.nan_to_num(x) - np.nan_to_num(y))
            bad.append((c, float(d.max()), int((d > 0).sum())))
    print(f"{name:9s} {len(cols):3d} cols   "
          f"{'IDENTICAL masked vs unmasked' if not bad else 'DIFFERS: ' + str(bad[:6])}")
    return len(bad)


def main() -> None:
    month = sys.argv[1] if len(sys.argv) > 1 else "2025-03"
    raw = data.load_month(month)
    masked, _ = data.mask_like_ranking(raw)
    blanked = int((raw["BLOCK_TIME_UTC_mvt"].notna() & masked["BLOCK_TIME_UTC_mvt"].isna()).sum())
    print(f"month {month}: {len(raw)} rows; the mask blanks {blanked} BLOCK cells")
    if blanked == 0:
        raise SystemExit("the mask blanked nothing; this test would pass vacuously")
    fit_src = pd.concat([data.load_month(x) for x in FIT_MONTHS], ignore_index=True)

    bad = 0
    ds = DS.DayState().fit(fit_src)
    bad += _cmp("deice", ds.transform(raw), ds.transform(masked), list(DS.DEICE))

    sf = SF.SurfaceFeatures.v2().fit(fit_src)
    keep = [c for fam in X.SURFACE_FAMILIES for c in SF.FAMILY_COLUMNS[fam] if c not in X.SKIP]
    a = SF.build_features(raw.assign(month=month), sf)
    b = SF.build_features(masked.assign(month=month), sf)
    bad += _cmp("surface", a, b, [c for c in keep if c in a.columns])

    cv = PF.PhiCV(PF.PhiModel(weight="rows")).fit({m: data.load_month(m) for m in FIT_MONTHS})
    pa, pb = cv.transform_serve(raw, mode="avg"), cv.transform_serve(masked, mode="avg")
    bad += _cmp("phi", pa, pb, [c for c in pa.columns if c != "MVT_ID_mvt"])

    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
