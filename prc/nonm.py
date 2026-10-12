"""Taxi-out head for departures with no Network Manager join (exp-013).

The segment is departures whose `_flt` columns are all null (`LOBT_flt` null)
and whose `AIRCRAFT_TYPE_mvt` is present: the ordinary delayed flights, about
1.4 % of the ranking file. They have no `AOBT_3_flt`, so the model that serves
the rest of the fleet has nothing to anchor on.

What this head is
-----------------
`taxi = (MVT − SCHED) − (BLOCK − SCHED)` is an identity, so for this segment
the entire prediction problem is "when did it leave the stand". The only
evidence in the frame is what *other* flights — the ones that do have an
`AOBT_3_flt` — were doing at that airport and runway at that moment. The head
is therefore an ordinary gradient-boosted model over movement columns plus two
neighbour blocks built from other departures' off-block times.

Contract
--------
- **No `_flt` column is read anywhere**, in training or at prediction, so the
  model is fitted on the world it will be asked about. Training uses every
  departure, not only the segment: segment-only training scores 753 s against
  585 s on july (exp-013 RESULTS §4).
- Features for a month are computed from that month's frame alone, and the
  frame is masked with `prc.data.mask_like_ranking` first.
- `predict` returns a value for every departure row; the caller selects the
  segment with `segment_mask`.

Shared-module rule: no silent default changes; bump `VERSION` and keep the old
behaviour reachable by argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import prc.data as D
import prc.surface_features as S

VERSION = 1

#: movement-only columns; every one survives on a row with no NM join
BASE = ["mvt_sched", "hour", "sched_hour", "dow", "flight_len", "apt", "apt_stand",
        "apt_rwy", "ac_type", "ades", "flight_rule", "op3"]
CATS = ["apt", "apt_stand", "apt_rwy", "ac_type", "ades", "flight_rule", "op3"]
#: neighbour blocks, both built from other departures' AOBT
SURFACE = (["nbr_span_apt", "nbr_span_rwy", "nbr_taxiin_apt", "nbr_taxiin_rwy"]
           + [f"dep_{a}_{b}{w}" for w in (900, 1800, 3600)
              for a in ("rwy", "apt") for b in ("pre", "post")]
           + [f"arr_{a}_{b}1800" for a in ("rwy", "apt") for b in ("pre", "post")]
           + ["gap_prev_rwy", "gap_next_rwy", "n10_rwy", "rank10_rwy",
              "taxiing_at_off", "arr_taxiing_at_off"])
BRACKET = ["brk_prev_span", "brk_next_span", "brk_prev_gap", "brk_next_gap",
           "brk_span_mean", "brk_n"]
FEATURES = BASE + SURFACE + BRACKET

PARAMS = dict(n_estimators=600, learning_rate=0.05, num_leaves=63,
              min_child_samples=40, subsample=0.9, subsample_freq=1,
              colsample_bytree=0.9, verbose=-1)
SEEDS = (7, 17, 29)


def segment_mask(dep: pd.DataFrame) -> np.ndarray:
    """No NM join and an aircraft type, on a departures-only frame."""
    return (dep["LOBT_flt"].isna() & dep["AIRCRAFT_TYPE_mvt"].notna()).to_numpy()


def _op3(flight: pd.Series) -> np.ndarray:
    """ICAO operator prefix of the callsign.

    `AIRCRAFT_OPERATOR_flt` is null on precisely this segment, so the carrier
    is only reachable through `FLIGHT_mvt`, which is a movement column.
    """
    s = flight.astype(str).str.upper()
    return s.str.extract(r"^([A-Z]{1,3})", expand=False).fillna("NA").to_numpy().astype(str)


def bracket_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Spans of the nearest same-runway departures that do have an AOBT.

    Returned in taxi units. The previous side uses `searchsorted(..., "left")`
    and the next side `"right"`, so a row that is itself in the source set
    reads neither side as itself; with `"left"` on both, this column was a
    row's own span on every AOBT row and a neighbour's on the segment, which
    cost the segment 102 s (exp-013 RESULTS §5).
    """
    phase = frame["PHASE_mvt"].to_numpy()
    dep_i = np.flatnonzero(phase == "DEP")
    apt = frame["ADEP_mvt"].astype(str).to_numpy()[dep_i]
    rwy = frame["RUNWAY_mvt"].astype(str).to_numpy()[dep_i]
    key = np.char.add(np.char.add(apt, ":"), rwy)
    off = S.epoch(frame["MVT_TIME_UTC_mvt"])[dep_i]
    span = off - S.epoch(frame["AOBT_3_flt"])[dep_i]
    have = np.isfinite(span)
    n = len(dep_i)
    prev_s, next_s = np.full(n, np.nan), np.full(n, np.nan)
    prev_g, next_g = np.full(n, np.nan), np.full(n, np.nan)
    mean_s, nnb = np.full(n, np.nan), np.zeros(n)
    for _, idx in S._groups(key):
        src = idx[have[idx]]
        if len(src) == 0:
            continue
        order = np.argsort(off[src], kind="stable")
        st, sv = off[src][order], span[src][order]
        q = off[idx]
        lo = np.searchsorted(st, q, "left")
        hi = np.searchsorted(st, q, "right")
        ok_p, ok_n = lo > 0, hi < len(st)
        pi = np.clip(lo - 1, 0, len(st) - 1)
        ni = np.clip(hi, 0, len(st) - 1)
        pv = np.where(ok_p, sv[pi], np.nan)
        nv = np.where(ok_n, sv[ni], np.nan)
        prev_s[idx], next_s[idx] = pv, nv
        prev_g[idx] = np.where(ok_p, q - st[pi], np.nan)
        next_g[idx] = np.where(ok_n, st[ni] - q, np.nan)
        mean_s[idx] = np.nanmean(np.vstack([pv, nv]), axis=0)
        nnb[idx] = len(st)
    return pd.DataFrame({"brk_prev_span": prev_s, "brk_next_span": next_s,
                         "brk_prev_gap": prev_g, "brk_next_gap": next_g,
                         "brk_span_mean": mean_s, "brk_n": nnb},
                        index=frame.index[dep_i])


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Head features for the departure rows of one month frame.

    `frame` may be masked or not: no masked column is read.
    """
    dep = frame[frame["PHASE_mvt"] == "DEP"]
    mvt, sch = dep["MVT_TIME_UTC_mvt"], dep["SCHED_TIME_UTC_mvt"]
    apt = dep["ADEP_mvt"].astype(str)
    X = pd.DataFrame(index=dep.index)
    X["mvt_sched"] = (mvt - sch).dt.total_seconds().to_numpy()
    X["hour"] = mvt.dt.hour.to_numpy() + mvt.dt.minute.to_numpy() / 60.0
    X["sched_hour"] = sch.dt.hour.to_numpy() + sch.dt.minute.to_numpy() / 60.0
    X["dow"] = mvt.dt.dayofweek.to_numpy()
    X["flight_len"] = dep["FLIGHT_mvt"].astype(str).str.len().to_numpy()
    X["apt"] = apt.to_numpy()
    X["apt_stand"] = (apt + ":" + dep["STAND_mvt"].astype(str)).to_numpy()
    X["apt_rwy"] = (apt + ":" + dep["RUNWAY_mvt"].astype(str)).to_numpy()
    X["ac_type"] = dep["AIRCRAFT_TYPE_mvt"].astype(str).to_numpy()
    X["ades"] = dep["ADES_mvt"].astype(str).to_numpy()
    X["flight_rule"] = dep["FLIGHT_RULE_mvt"].astype(str).to_numpy()
    X["op3"] = _op3(dep["FLIGHT_mvt"])
    sur = S.build_features(frame, None).reindex(X.index)
    for c in SURFACE:
        X[c] = sur[c].to_numpy() if c in sur.columns else np.nan
    brk = bracket_features(frame).reindex(X.index)
    for c in BRACKET:
        X[c] = brk[c].to_numpy()
    return X[FEATURES]


def _align(Xtr: pd.DataFrame, Xva: pd.DataFrame):
    for c in CATS:
        a, b = Xtr[c].astype(str), Xva[c].astype(str)
        u = pd.Index(pd.unique(np.concatenate([a.to_numpy(), b.to_numpy()])))
        Xtr[c] = pd.Categorical(a, categories=u)
        Xva[c] = pd.Categorical(b, categories=u)
    return Xtr, Xva


@dataclass
class NoNMHead:
    """Seed-averaged LightGBM head. `n_jobs` follows docs/COMPUTE.md."""

    n_jobs: int = 6
    seeds: tuple = SEEDS
    params: dict = field(default_factory=lambda: dict(PARAMS))
    models_: list = field(default_factory=list)
    train_: pd.DataFrame | None = None

    def fit(self, train_frames, month_col: str = "month") -> "NoNMHead":
        import lightgbm as lgb
        Xs, ys = [], []
        for f in train_frames:
            masked, _ = D.mask_like_ranking(f)
            X = build_features(masked)
            Xs.append(X)
            ys.append(f.loc[X.index, D.TARGET].to_numpy(float))
        self.train_ = pd.concat(Xs, ignore_index=True)
        y = np.concatenate(ys)
        self.models_ = []
        for s in self.seeds:
            m = lgb.LGBMRegressor(**self.params, n_jobs=self.n_jobs, random_state=s)
            Xtr = self.train_.copy()
            for c in CATS:
                Xtr[c] = Xtr[c].astype("category")
            m.fit(Xtr, y, categorical_feature=CATS)
            self.models_.append(m)
        return self

    def predict(self, month_frame: pd.DataFrame) -> pd.Series:
        """Seed-averaged taxi prediction for every departure row of the frame."""
        masked, _ = D.mask_like_ranking(month_frame)
        Xva = build_features(masked)
        preds = []
        for m in self.models_:
            a, b = _align(self.train_.copy(), Xva.copy())
            preds.append(m.predict(b))
        return pd.Series(np.mean(preds, axis=0), index=Xva.index)
