"""Versioned feature-frame builder, for this experiment and for the blend owner.

`v2` is the frozen 54-column frame every blend member currently sits on. `v3` is
that frame plus every family the desk's method review found absent from all
members: the seven `prc.surface_features` families, the non-duplicated part of
`prc.stand_features`, `prc.weather_features.WINTER_CORE`, the de-icing column,
`phi` and the LIRF-gated block-clock signals.

    from frame import FrameBuilder
    fb = FrameBuilder("v3").fit({m: raw_month_frame for m in train_months})
    X  = fb.transform(one_month_frame)      # per month, never two together

Two contracts a caller must not break. **One month at a time**: the ranking file
holds January and July 2026 as isolated months and several families join on
nearest events, so transforming them together reaches across the six-month gap.
And **fit on training months only**: `phi`, the surface geometry family and the
day-state reference distribution all read labels.

Deduplication is measured, not declared. Four upstream modules compute the same
stand-occupancy quantities under different names — `dep_next_mvt_gap` and
`lf_stand_next_dep_gap` correlate at 1.000 — so after assembly any column whose
absolute correlation with an already-kept column exceeds `DEDUP_R` is dropped,
in a fixed source order, and the dropped pairs are recorded in `self.dropped`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import daystate_step
import exot_step
import features as F
import fresh_step
import lirf_step
import phi_step
import stand_step
import surface_step

DEDUP_R = 0.98
FRAME_VERSION = {"v2": 1, "v3": 1}
#: source order; earlier sources win a duplicate pair
#: FROZEN. The 171-column v3 frame the blend screened V3-allin on at +1.13 s.
#: Adding a source here changes what every downstream arm and every published
#: member is built from, which is exactly what happened when `fresh` was appended
#: mid-programme: V3-wins3600 silently gained four columns and stopped being
#: comparable with V3-allin. New families go in a new tuple and a new flag value.
SOURCES_V3 = ("base", "phi", "lirf", "surface", "stand", "weather", "deice", "exot")
#: v3 plus exp-018's three label-free families (175 columns after dedup)
SOURCES_V3F = SOURCES_V3 + ("fresh",)
SOURCE_SETS = {"v3": SOURCES_V3, "v3f": SOURCES_V3F}


class FrameBuilder:
    def __init__(self, version: str = "v3", sources: tuple = SOURCES_V3, dedup_r: float = DEDUP_R):
        self.version, self.sources, self.dedup_r = version, sources, dedup_r
        self.dropped: dict[str, str] = {}
        self.keep: list[str] | None = None

    # ---------------------------------------------------------------- fit --
    def fit(self, raw_by_month: dict) -> "FrameBuilder":
        self.months = list(raw_by_month)
        self.raw = raw_by_month
        if self.version == "v2":
            return self
        import prc.weather_features as W
        from prc.day_state_features import DayState
        from prc.phi_features import PhiCV, PhiModel
        from prc.surface_features import SurfaceFeatures

        self._hourly = W.load_hourly() if "weather" in self.sources else None
        if "phi" in self.sources:
            self._phi = PhiCV(PhiModel(**phi_step.PHI_SETTINGS)).fit(raw_by_month)
        if "surface" in self.sources:
            self._sf = {m: SurfaceFeatures(**surface_step.SURFACE_SETTINGS).fit(
                pd.concat([raw_by_month[o] for o in self.months if o != m], ignore_index=True))
                for m in self.months}
            self._sf_full = SurfaceFeatures(**surface_step.SURFACE_SETTINGS).fit(
                pd.concat(list(raw_by_month.values()), ignore_index=True))
        if "deice" in self.sources:
            self._ds = DayState().fit(pd.concat(list(raw_by_month.values()), ignore_index=True))
        return self

    # ---------------------------------------------------------- transform --
    def _parts(self, raw: pd.DataFrame, month: str | None) -> list[pd.DataFrame]:
        X = F.build_features(raw, "v2").reset_index(drop=True)
        out = [X]
        if self.version == "v2":
            return out
        train = month in getattr(self, "months", ())
        if "phi" in self.sources:
            p = (self._phi.transform_train(month, raw) if train
                 else self._phi.transform_serve(raw, mode="avg"))
            out.append(p.set_index("MVT_ID_mvt")[list(phi_step.COLUMNS)]
                       .reindex(X.MVT_ID_mvt).reset_index(drop=True))
        if "lirf" in self.sources:
            g = lirf_step._attach(X[["MVT_ID_mvt", "airport"]], lirf_step._one(raw), gate=True)
            out.append(g.drop(columns=["MVT_ID_mvt", "airport"]))
        if "surface" in self.sources:
            sf = self._sf[month] if train else self._sf_full
            out.append(surface_step._one(raw, sf).drop(columns=["MVT_ID_mvt"]).add_prefix("sf_"))
        if "stand" in self.sources:
            out.append(stand_step._one(raw).drop(columns=["MVT_ID_mvt"]).add_prefix("st_"))
        if "weather" in self.sources:
            import prc.weather_features as W
            w = W.transform(raw, self._hourly)
            dep = raw["PHASE_mvt"].eq("DEP").to_numpy()
            out.append(w.loc[dep, [c for c in W.WINTER_CORE if c in w.columns]].reset_index(drop=True))
        if "deice" in self.sources:
            out.append(daystate_step._one(raw, self._ds).drop(columns=["MVT_ID_mvt"]))
        if "exot" in self.sources:
            out.append(exot_step._one(raw).drop(columns=["MVT_ID_mvt"]))
        if "fresh" in self.sources:
            out.append(fresh_step._one(raw).drop(columns=["MVT_ID_mvt"]))
        return out

    def transform(self, raw: pd.DataFrame, month: str | None = None) -> pd.DataFrame:
        X = pd.concat(self._parts(raw, month), axis=1)
        X = X.loc[:, ~X.columns.duplicated()]
        if self.keep is not None:
            X = X[self.keep]
        return X

    # ------------------------------------------------------------- dedup --
    def learn_dedup(self, raw: pd.DataFrame, month: str) -> "FrameBuilder":
        """Measure the duplicate pairs on one month and freeze the kept column list."""
        X = self.transform(raw, month)
        num = [c for c in X.columns if c != "MVT_ID_mvt" and X[c].dtype.kind == "f"]
        keep, kept_arr = [], []
        A = X[num].to_numpy("float64")
        for j, c in enumerate(num):
            v = A[:, j]
            ok = np.isfinite(v)
            if ok.sum() < 1000 or v[ok].std() < 1e-9:
                keep.append(c); kept_arr.append(v); continue
            dup = None
            for c2, w in zip(keep, kept_arr):
                k = ok & np.isfinite(w)
                if k.sum() < 1000 or w[k].std() < 1e-9:
                    continue
                if abs(np.corrcoef(v[k], w[k])[0, 1]) >= self.dedup_r:
                    dup = c2; break
            if dup is None:
                keep.append(c); kept_arr.append(v)
            else:
                self.dropped[c] = dup
        cats = [c for c in X.columns if c not in num]
        self.keep = cats + keep
        return self
