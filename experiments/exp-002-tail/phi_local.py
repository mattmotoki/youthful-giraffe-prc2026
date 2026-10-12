"""Local phi estimator, pending prc/phi_features.py from the lirf worker.

lirf's finding: the airport block time sits a stable fraction phi of the way
from the NM actual off-block toward the last filed off-block,

    BLOCK = AOBT_3 + phi * (LOBT - AOBT_3)

with phi stable per airport and per operator. That is exactly this experiment's
NM-head target `offset = BLOCK - AOBT_3` written as `phi * den`, where
`den = LOBT - AOBT_3` is observable. A tree given `den` as a feature has to
approximate the *product* of a learned per-group constant and a continuous
input, which is the same limitation that made the wrong anchor expensive in
finding 8 -- so handing it `phi_hat` and `phi_hat * den` directly should pay.

phi is estimated as the least-squares slope through the origin,
`phi = sum(num*den) / sum(den^2)`, not as a mean of ratios: the slope is what
minimises squared error in `num`, and it is stable when `den` is near zero,
where the ratio explodes. Groups are shrunk toward their parent
(airport x operator -> airport -> global) with a ridge in the same units.

Fitted on training months only. `num` reads BLOCK_TIME_UTC_mvt, so `fit` must
never see a validation or ranking frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RIDGE_ROWS = 40.0   # pseudo-rows of the parent mixed into each group


def _sec(a, b):
    return (a - b).dt.total_seconds()


def parts(frame: pd.DataFrame) -> pd.DataFrame:
    """`num` (BLOCK-AOBT, NaN when masked), `den` (LOBT-AOBT), and the group keys."""
    out = pd.DataFrame(index=frame.index)
    out["den"] = _sec(frame["LOBT_flt"], frame["AOBT_3_flt"])
    out["num"] = (_sec(frame["BLOCK_TIME_UTC_mvt"], frame["AOBT_3_flt"])
                  if "BLOCK_TIME_UTC_mvt" in frame else np.nan)
    out["apt"] = frame["ADEP_mvt"].astype(str)
    out["op"] = out["apt"] + "|" + frame["AIRCRAFT_OPERATOR_flt"].astype(str)
    return out


@dataclass
class PhiStats:
    glob: float = 0.0
    by_apt: pd.Series = field(default_factory=pd.Series)
    by_op: pd.Series = field(default_factory=pd.Series)
    n_op: pd.Series = field(default_factory=pd.Series)

    @staticmethod
    def _slope(g):
        sxx = float((g["den"] ** 2).sum())
        sxy = float((g["num"] * g["den"]).sum())
        return sxx, sxy

    def fit(self, train: pd.DataFrame) -> "PhiStats":
        p = parts(train)
        p = p[p["num"].notna() & p["den"].notna()]
        scale = float((p["den"] ** 2).mean())
        sxx_g, sxy_g = self._slope(p)
        self.glob = sxy_g / sxx_g if sxx_g > 0 else 0.0
        rows = []
        for key, parent_of in (("apt", None), ("op", "apt")):
            agg = p.groupby(key).apply(
                lambda g: pd.Series({"sxx": float((g["den"] ** 2).sum()),
                                     "sxy": float((g["num"] * g["den"]).sum()),
                                     "n": len(g)}), include_groups=False)
            rows.append((key, agg))
        apt = rows[0][1]
        ridge = RIDGE_ROWS * scale
        self.by_apt = ((apt["sxy"] + ridge * self.glob) / (apt["sxx"] + ridge))
        op = rows[1][1]
        parent = pd.Series(op.index.str.split("|").str[0], index=op.index).map(self.by_apt).fillna(self.glob)
        self.by_op = ((op["sxy"] + ridge * parent) / (op["sxx"] + ridge))
        self.n_op = op["n"]
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """phi and the quantities it implies, for every row of `frame`.

        Reads only LOBT_flt, AOBT_3_flt, MVT_TIME_UTC_mvt, ADEP_mvt and
        AIRCRAFT_OPERATOR_flt -- all present in the ranking file.
        """
        p = parts(frame)
        phi = p["op"].map(self.by_op)
        phi = phi.fillna(p["apt"].map(self.by_apt)).fillna(self.glob)
        out = pd.DataFrame(index=frame.index)
        out["phi_hat"] = phi.astype(float)
        out["phi_n"] = p["op"].map(self.n_op).astype(float)
        out["phi_den"] = p["den"]
        out["phi_offset"] = phi * p["den"]                    # implied BLOCK - AOBT
        implied_block = frame["AOBT_3_flt"] + pd.to_timedelta(out["phi_offset"], unit="s")
        out["phi_taxi"] = _sec(frame["MVT_TIME_UTC_mvt"], implied_block)
        return out
