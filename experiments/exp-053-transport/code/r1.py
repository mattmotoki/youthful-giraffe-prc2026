"""exp-053 revision r1, after Sol round 1: the stratum table with the registered support status,
and route B's registered row check.  ZERO FIT; no loss is recomputed.

B1 (Sol round 1).  SPEC §3 registers every G, A and C stratum as a structural zero, "supported by
construction", with a known zero step gain on both sides.  `instrument.py` set `supported` only
on head cells, so its table (`analysis/instrument-cells.csv`, write-once, left as run) marks the
65 structural-zero strata unsupported, and the four ranking-only structural-zero cells carry no
gain, RMSE or interval fields.  No estimate used those flags: S sums supported H cells, U counts
unsupported H mass, and the routing reads neither.  This script writes the corrected table
`analysis/instrument-cells-r1.csv`: status "structural zero" (supported by construction, gain 0,
RMSE gain 0, intervals [0, 0]) on every non-H stratum, "supported" / "unsupported" on H cells as
before.  It then checks, per (fold, target month), that the supported and unsupported target
mass summed from the per-cell status is 1 and equals `instrument.json`'s U.  The zeros rest on
`pin.json`: no fold head vector moves a row outside H, and no uploaded step changes a ranking
row outside H.

Route B (Sol round 1, note).  SPEC §8 (ii) also asks that each month of an identifiable pair
carry at least 2,000 writable rows at 5 or more common airports.  `instrument.py` stopped at
"already measured".  The check is computed here from `analysis/extcount.json` (input-only) for
the two pairs with a time gap of 6 or more.  It cannot change the route: (i) fails, and both
pairs were measured by #4 and recorded rule.

    .venv/bin/python experiments/exp-053-transport/code/r1.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C

CELLS = C.ANALYSIS / "instrument-cells.csv"
INSTR = C.ANALYSIS / "instrument.json"
PIN = C.ANALYSIS / "pin.json"
EXT = C.ANALYSIS / "extcount.json"
OUT_CSV = C.ANALYSIS / "instrument-cells-r1.csv"
OUT = C.ANALYSIS / "r1.json"
TARGETS = {"july_loo": ("2026-07",), "december": ("2026-01", "2026-07"), "january": ("2026-01",)}
STEP_NAMES = ("v20", "v21", "v22", "v22|v19")


def main() -> None:
    C.NO.refuse_existing([OUT_CSV, OUT], overwrite=False)
    ins = json.loads(INSTR.read_text())
    pin = json.loads(PIN.read_text())
    C.check(CELLS, ins["cells_csv"]["sha256"])
    # the zeros this table asserts, from the pin
    for f in C.FOLDS:
        if any(pin["folds"][f]["head_rows_moved_outside_H"].values()):
            raise SystemExit(f"{f}: a head vector moves a row outside H")
    for name in STEP_NAMES:
        by = pin["ranking"][f"upload_rows_changed_{name}"]
        if by["G"] or by["A"] or by["C"]:
            raise SystemExit(f"{name}: an uploaded step changes a ranking row outside H")

    c = pd.read_csv(CELLS)
    before = c.copy()
    sz = c.regime != "H"
    if not (c.structural_zero == sz).all():
        raise SystemExit("the structural_zero column disagrees with the regime")
    # every non-H value already present must be an exact zero
    gcols = [k for k in c.columns if k.startswith(("g_s2_", "rmse_gain_s_", "ci1_lo_", "ci1_hi_",
                                                   "ci2_lo_", "ci2_hi_", "ci3_lo_", "ci3_hi_"))]
    present = c.loc[sz, gcols]
    if (present.fillna(0) != 0).any().any():
        raise SystemExit("a structural-zero stratum carries a nonzero gain or interval")
    filled = int(present.isna().any(axis=1).sum())
    c.loc[sz, gcols] = 0.0
    for L in (1, 2, 3):
        c.loc[sz & (c.fold_rows == 0), f"ci{L}_empty_draws"] = 0
    c["status"] = np.where(sz, "structural zero", np.where(c.supported, "supported", "unsupported"))
    c["supported"] = sz | c.supported.astype(bool)
    # nothing on the head cells changes
    h = ~sz
    if not before.loc[h].drop(columns=["supported"]).equals(c.loc[h, before.columns].drop(columns=["supported"])) \
            or not (before.loc[h, "supported"].astype(bool) == c.loc[h, "supported"]).all():
        raise SystemExit("a head cell changed")

    # supported and unsupported target mass, from the per-cell status
    mass = {}
    for f, months in TARGETS.items():
        for m in months:
            col = f"rank_mass_{m}"
            sub = c[(c.fold == f) & c[col].notna()]
            tot = float(sub[col].sum())
            uns = float(sub.loc[sub.status == "unsupported", col].sum())
            want = ins["months"][f"{f}->{m}"]["unsupported_rank_H_mass"]
            if abs(tot - 1) > 1e-12 or abs(uns - want) > 1e-12:
                raise SystemExit(f"{f}->{m}: target mass {tot!r}, unsupported {uns!r} vs U {want!r}")
            mass[f"{f}->{m}"] = {
                "supported_target_mass": tot - uns, "unsupported_target_mass": uns,
                "of_which_structural_zero": float(sub.loc[sub.status == "structural zero", col].sum()),
                "of_which_supported_head": float(sub.loc[sub.status == "supported", col].sum()),
                "unsupported_cells": sorted(sub.loc[sub.status == "unsupported", "airport"])}

    # route B (ii), the registered row condition, on the pairs with a time gap >= 6
    ext = json.loads(EXT.read_text())["months"]
    rows_check = {}
    for fold, pair in (("december", ("2025-10", "2025-02")), ("january", ("2025-03", "2025-11"))):
        a, b = (ext[mo]["by_airport"] for mo in pair)
        common = sorted(set(a) & set(b))
        both_2000 = [ap for ap in common if a[ap] >= 2000 and b[ap] >= 2000]
        rows_check[f"{fold}:{pair[0]}/{pair[1]}"] = {
            "common_airports": common,
            "airports_with_2000_rows_in_each_month": both_2000,
            "rows_at_common_airports": {pair[0]: int(sum(a[x] for x in common)),
                                        pair[1]: int(sum(b[x] for x in common))},
            "passes_2000_rows_at_5_common_airports": len(both_2000) >= 5,
            "already_measured": "exp-046 #4 (section 47.74) and D28 (section 47.76)"}

    c.to_csv(OUT_CSV, index=False)
    rec = {"what": __doc__.strip().splitlines()[0],
           "inputs": {"cells": {"path": str(CELLS.relative_to(C.ROOT)), "sha256": ins["cells_csv"]["sha256"]},
                      "instrument": {"path": str(INSTR.relative_to(C.ROOT)), "sha256": C.sha_file(INSTR)},
                      "pin": {"path": str(PIN.relative_to(C.ROOT)), "sha256": C.sha_file(PIN)},
                      "extcount": {"path": str(EXT.relative_to(C.ROOT)), "sha256": C.sha_file(EXT)}},
           "cells_r1": {"path": str(OUT_CSV.relative_to(C.ROOT)), "sha256": C.sha_file(OUT_CSV),
                        "rows": int(len(c)), "structural_zero_strata": int(sz.sum()),
                        "structural_zero_strata_given_zero_fields": filled,
                        "head_cells": int(h.sum()),
                        "head_cells_unsupported": int((c.status == "unsupported").sum())},
           "target_mass": mass, "route_B_ii_rows": rows_check,
           "estimates_changed": "none: S, P, F, U and the routing never read the non-H support flag"}
    C.write_json(OUT, rec)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
