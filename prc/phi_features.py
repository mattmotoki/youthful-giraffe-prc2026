"""The block-clock fraction phi: where the airport's block time sits between
Network Manager's actual off-block and the last filed off-block.

Discovered in exp-006 (`experiments/exp-006-lirf/RESULTS.md`, findings 2-4).

The airport movement record and the Network Manager flight record disagree about
when an aircraft went off-block, and the disagreement is not noise. Writing

    gap = LOBT_flt - AOBT_3_flt          (filed minus actual off-block, observable)
    r   = BLOCK_TIME_UTC_mvt - AOBT_3_flt (the unobservable we must predict)

then `r = phi * gap` with an airport-specific phi that is stable all year:

    EHAM -0.03   EDDF 0.06   LSZH 0.13   LEMD 0.15   LEBL 0.17
    LFPG 0.19    EDDM 0.23   EGLL 0.25   LIRF 0.38   LTFM 0.40

phi refines by operator, and that refinement is a *convention*, not a fit: over
the 58 operators with at least 60 LIRF departures in each half of 2025, phi
ranges from 0.09 to 1.21 and its correlation between the two halves is **0.948**.
Some handling agents' block time is recorded on the filed off-block, others' on
the actual one. Because `gap` grows with congestion (LIRF median 120 s in
January, 600 s in July) the same convention costs little in winter and a great
deal in summer, which is the whole of Rome's July difficulty.

Why the taxi task cares: the target is `take-off - BLOCK = (take-off - AOBT) - r`,
so `r` *is* the task once the known part is subtracted. A per-airport-and-operator
phi, fitted on the training months alone, predicts the NM-joined July 2025
departures at RMSE 392 s against 466 s for `take-off - AOBT` and, at LIRF alone,
551 s against a fitted LightGBM's 636 s.

Fold discipline: `fit` reads the departure block time, which exists only in
training months. Fit on the fold's training months, transform the validation or
ranking month. The estimator is a shrunk least-squares *ratio* over hundreds of
rows per group, so a row moves its own group's phi by O(1/n); it does not carry
the train-versus-serve noise mismatch that a small-group target mean does
(`experiments/exp-001-retro-v1/RESULTS.md`, "Target encodings").

    from prc.phi_features import PhiModel
    pm = PhiModel().fit([train_month_frames...])
    F = pm.transform(valid_month_frame)     # phi, block_hat, taxi_hat, support
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["PhiModel", "PhiCV", "DEFAULT_KEYS", "AIRPORT_PHI_2025", "VERSION"]

#: Module version. Bump on any change to a default or to the meaning of a
#: column, and name the consumers in the same report (desk rule, 2026-09-05).
#: Consumers pin the arguments they depend on rather than relying on a default.
#:
#: 3  2026-09-05  `transform(leave_one_out=True)` corrected: the coarsest prior
#:                now drops the row's own contribution too, and the reported
#:                support is decremented only for a row that entered the fit, so
#:                the result equals an exact delete-one refit
#:                (`experiments/exp-006-lirf/work/test_phi_loo.py`). The level
#:                gate still uses the fitted support by design. No default and no
#:                column meaning changed; `full`, `cv` and `cvfull` are untouched.
#:                Consumers: `experiments/exp-001-retro-v1/phi_step.py`,
#:                `experiments/exp-006-lirf/work/cache_phi.py`.
#: 2  2026-09-05  `weight` added; the DEFAULT CHANGED from the gap-squared
#:                least-squares coefficient to `"rows"`, the mean per-flight
#:                ratio shrunk by flight count, because the gap-squared weight
#:                gives one flight up to 36 % of its own group's coefficient
#:                (gpt-5.6-sol, `experiments/exp-006-lirf/review/sol-round1.md`).
#:                The old behaviour is `PhiModel(weight="gap2")`, unchanged.
#:                `PhiCV` and `PhiModel.transform(..., leave_one_out=True)` added;
#:                no existing column was renamed or redefined.
#:                Consumers at the time of the bump:
#:                `experiments/exp-001-retro-v1/phi_step.py`,
#:                `experiments/exp-006-lirf/work/cache_phi.py`.
#: 1  2026-09-05  first version: hierarchical shrunk least-squares phi.
VERSION = 3

#: The shrinkage chain, coarse to fine. Each level is shrunk toward the level
#: above it; a level with no support falls back to it exactly. A third level was
#: measured (`experiments/exp-006-lirf/work/q7_phimodel.py`) and is not worth its
#: parameters: on the NM-joined LIRF rows, airport 593.1 / airport+operator 553.1
#: / +operator-apron 551.6 / +operator-hour-band 552.4 / +operator-stand 554.8
#: on the july fold, with the same ordering on december. The shrinkage constant
#: is almost inert over 5e4 to 1e6 (under 0.01 s), because the groups that matter
#: all have hundreds of flights behind them.
DEFAULT_KEYS: tuple[tuple[str, ...], ...] = (
    ("airport",),
    ("airport", "operator"),
)

#: phi per airport over all twelve 2025 months, for reference and for a caller
#: that wants the prior without fitting.
AIRPORT_PHI_2025 = {
    "EHAM": -0.033, "EDDF": 0.062, "LSZH": 0.133, "LEMD": 0.153, "LEBL": 0.166,
    "LFPG": 0.195, "EDDM": 0.232, "EGLL": 0.250, "LIRF": 0.381, "LTFM": 0.399,
}

_APRON_RE = r"^(\d+)"


def _epoch(s: pd.Series) -> np.ndarray:
    v = s.astype("datetime64[ns, UTC]").astype("int64").to_numpy().astype("float64")
    v[s.isna().to_numpy()] = np.nan
    return v / 1e9


def _key_frame(dep: pd.DataFrame) -> pd.DataFrame:
    """The key columns every level can be built from, as plain strings."""
    stand = dep["STAND_mvt"].astype("string")
    apron = stand.str.extract(_APRON_RE)[0]
    apron = pd.to_numeric(apron, errors="coerce") // 100
    hour = dep["MVT_TIME_UTC_mvt"].dt.hour
    return pd.DataFrame({
        "airport": dep["ADEP_mvt"].astype("string").fillna("\x00NA").to_numpy(),
        "operator": dep["AIRCRAFT_OPERATOR_flt"].astype("string").fillna("\x00NA").to_numpy(),
        "apron": apron.astype("Float64").astype("string").fillna("\x00NA").to_numpy(),
        "stand": stand.fillna("\x00NA").to_numpy(),
        "hour_band": pd.cut(hour, [-1, 5, 9, 13, 17, 21, 24],
                            labels=["n", "em", "am", "md", "pm", "ev"]).astype("string")
                     .fillna("\x00NA").to_numpy(),
        "segment": dep["MARKET_SEGMENT_flt"].astype("string").fillna("\x00NA").to_numpy(),
    }, index=dep.index)


@dataclass
class PhiModel:
    """Hierarchical shrunk estimate of phi.

    Two estimators, because the obvious one has a leverage problem.

    - `weight="rows"` (the default) averages the per-flight ratio `r_i / g_i` over
      the flights whose `|gap|` is at least `min_abs_gap`, shrinking toward the
      parent level by row count: `phi = (sum(phi_i) + k*prior) / (n + k)` with `k`
      in flights. One flight can then own at most `1 / (n + k)` of its group's
      coefficient.
    - `weight="gap2"` is the least-squares coefficient
      `sum(r*g) / sum(g^2)`, shrunk with `k` in units of `sum(g^2)`. It fits the
      quantity the model actually needs, but its weight is `g^2`, so a single
      long-gap flight can own a large share of its own group's coefficient: on the
      july fold's training rows the largest own-row share is **36 %** and 163 rows
      exceed 10 % (gpt-5.6-sol, `review/sol-round1.md`). Kept for comparison and
      for the airport level, where n is large enough that leverage is irrelevant.

    `keys` is the chain, coarse to fine. `min_rows` refuses a level whose group has
    fewer rows than that.
    """

    keys: tuple[tuple[str, ...], ...] = DEFAULT_KEYS
    weight: str = "rows"
    shrink: float = 2.0e5          # used when weight == "gap2"
    shrink_rows: float = 150.0     # used when weight == "rows"
    min_abs_gap: float = 300.0     # flights below this do not identify a ratio
    min_rows: int = 100
    phi_global: float = 0.0
    num_global: float = 0.0
    den_global: float = 0.0
    shrink_global: float = 0.0
    tables: dict[tuple[str, ...], pd.DataFrame] = field(default_factory=dict)

    def _check_chain(self) -> None:
        """Each level must extend the one above it, so a child can look its parent up."""
        for i in range(1, len(self.keys)):
            if not set(self.keys[i - 1]).issubset(self.keys[i]):
                raise ValueError(f"level {self.keys[i]} does not extend {self.keys[i - 1]}")

    # ------------------------------------------------------------------ fit --
    @property
    def _k(self) -> float:
        return self.shrink_rows if self.weight == "rows" else self.shrink

    def fit(self, frames) -> "PhiModel":
        """Fit on labelled month frames (one frame or an iterable of them)."""
        self._check_chain()
        if isinstance(frames, pd.DataFrame):
            frames = [frames]
        parts = []
        for f in frames:
            dep = f[f["PHASE_mvt"].eq("DEP")]
            k = _key_frame(dep)
            k["gap"] = _epoch(dep["LOBT_flt"]) - _epoch(dep["AOBT_3_flt"])
            k["r"] = _epoch(dep["BLOCK_TIME_UTC_mvt"]) - _epoch(dep["AOBT_3_flt"])
            parts.append(k[np.isfinite(k.gap) & np.isfinite(k.r)])
        d = pd.concat(parts, ignore_index=True)
        if self.weight == "rows":
            d = d[d.gap.abs() >= self.min_abs_gap].copy()
            d["num"] = d.r / d.gap          # the per-flight ratio
            d["den"] = 1.0
        else:
            d["num"] = d.r * d.gap
            d["den"] = d.gap ** 2
        self.num_global = float(d.num.sum())
        self.den_global = float(d.den.sum())
        self.shrink_global = 0.0
        self.phi_global = self.num_global / max(self.den_global, 1e-9)
        self.tables = {}
        for level in self.keys:
            g = d.groupby(list(level), observed=True).agg(num=("num", "sum"),
                                                          den=("den", "sum"),
                                                          n=("r", "size"))
            prior = self._lookup(g.reset_index()[list(level)], parent_of(self.keys, level))
            g = g.reset_index()
            g["prior"] = prior
            g["phi"] = (g.num + self._k * g.prior) / (g.den + self._k)
            g.loc[g.n < self.min_rows, "phi"] = g.loc[g.n < self.min_rows, "prior"]
            self.tables[level] = g.set_index(list(level))[["phi", "n", "den", "num"]]
        return self

    # ------------------------------------------------------------ transform --
    def _lookup(self, keys_df: pd.DataFrame, level: tuple[str, ...] | None) -> np.ndarray:
        """phi at one level for the rows of `keys_df`, falling back to the global value."""
        if level is None or level not in self.tables:
            return np.full(len(keys_df), self.phi_global)
        t = self.tables[level]
        idx = pd.MultiIndex.from_frame(keys_df[list(level)]) if len(level) > 1 \
            else pd.Index(keys_df[level[0]])
        v = t.phi.reindex(idx).to_numpy("float64")
        return np.where(np.isfinite(v), v, self.phi_global)

    def _stat(self, keys_df: pd.DataFrame, level: tuple[str, ...], col: str) -> np.ndarray:
        """Raw sum(r*g) or sum(g^2) for each row's group, 0 where the group is unseen."""
        t = self.tables.get(level)
        if t is None or col not in t.columns:
            return np.zeros(len(keys_df))
        idx = pd.MultiIndex.from_frame(keys_df[list(level)]) if len(level) > 1 \
            else pd.Index(keys_df[level[0]])
        return np.nan_to_num(t[col].reindex(idx).to_numpy("float64"))

    def _support(self, keys_df: pd.DataFrame, level: tuple[str, ...]) -> np.ndarray:
        t = self.tables.get(level)
        if t is None:
            return np.zeros(len(keys_df))
        idx = pd.MultiIndex.from_frame(keys_df[list(level)]) if len(level) > 1 \
            else pd.Index(keys_df[level[0]])
        return np.nan_to_num(t.n.reindex(idx).to_numpy("float64"))

    def transform(self, frame: pd.DataFrame, leave_one_out: bool = False) -> pd.DataFrame:
        """Per-departure phi, the block time it implies, the taxi time it implies, support.

        `leave_one_out=True` removes each row's own contribution from every level
        of its own coefficient, analytically:

            phi_key^(-i) = (sum(r*g) - r_i*g_i + k*prior^(-i)) / (sum(g^2) - g_i^2 + k)

        It is only meaningful on a frame whose departure block times are present,
        i.e. a training month, and it is what a training row must see when the
        same fitted model serves a validation month. Unlike a leave-one-*month*-out
        construction the support changes by one row rather than by a sixth of the
        data, so the two sides of the model carry the same noise level. Without it
        a single row can own a large share of its group's coefficient: the weight
        is `g^2`, so on the july fold's training rows the largest own-row share of
        an airport-operator coefficient is 36 %, and 163 rows exceed 10 %.

        Returned columns, indexed like the departure rows of `frame`:

        ``phi``               the finest level with support, shrunk toward its parent
        ``phi_<level>``       phi at each level of the chain
        ``phi_support``       training rows behind the finest level used
        ``phi_support_<lvl>`` training rows at each level
        ``block_hat``         AOBT + phi * (LOBT - AOBT), the implied block time (epoch s)
        ``taxi_hat``          take-off - block_hat, the implied taxi time (s)
        ``taxi_hat_<level>``  the same at each level
        ``gap``               LOBT - AOBT, the spread phi multiplies

        `taxi_hat` is NaN where the NM join is missing; a caller needs its own
        fallback for those rows.
        """
        dep = frame[frame["PHASE_mvt"].eq("DEP")]
        k = _key_frame(dep)
        aobt = _epoch(dep["AOBT_3_flt"])
        lobt = _epoch(dep["LOBT_flt"])
        t = _epoch(dep["MVT_TIME_UTC_mvt"])
        gap = lobt - aobt
        if leave_one_out:
            r_own = _epoch(dep["BLOCK_TIME_UTC_mvt"]) - aobt
            ok_own = np.isfinite(r_own) & np.isfinite(gap)
            if self.weight == "rows":
                ok_own = ok_own & (np.abs(gap) >= self.min_abs_gap)
                own_num = np.where(ok_own, np.divide(r_own, gap, out=np.zeros_like(gap),
                                                     where=gap != 0), 0.0)
                own_den = np.where(ok_own, 1.0, 0.0)
            else:
                own_num = np.where(ok_own, r_own * gap, 0.0)
                own_den = np.where(ok_own, gap ** 2, 0.0)
        out = pd.DataFrame(index=dep.index)
        out["MVT_ID_mvt"] = dep["MVT_ID_mvt"].to_numpy()
        out["gap"] = gap
        if leave_one_out:
            # the global prior has to drop the row's own contribution too, or the
            # row's label reaches its own feature through the coarsest level
            # (gpt-5.6-sol round 2)
            g_prior = ((self.num_global - own_num) + self.shrink_global * 0.0) / \
                np.maximum(self.den_global - own_den, 1e-9)
        else:
            g_prior = np.full(len(dep), self.phi_global)
        best_phi = g_prior.copy()
        best_sup = np.zeros(len(dep))
        prior = g_prior.copy()
        for level in self.keys:
            tag = "_".join(level)
            s = self._support(k, level)
            if leave_one_out:
                num = self._stat(k, level, "num") - own_num
                den = self._stat(k, level, "den") - own_den
                p = (num + self._k * prior) / (den + self._k)
                # Two deliberate choices at the boundary, both documented because
                # they differ from a literal delete-one refit:
                #  * the level GATE uses the fitted support, not `s - 1`. Gating on
                #    `s - 1` would send a row in a group of exactly `min_rows` to
                #    the parent while an identical serving row used the group — a
                #    train/serve mismatch at the boundary, which is the one thing
                #    the leave-one-out form exists to avoid.
                #  * the reported SUPPORT is decremented only for a row that
                #    actually entered the fit, so it equals what a delete-one refit
                #    would report.
                p = np.where(s >= self.min_rows, p, prior)
                s = s - np.where(own_den > 0, 1.0, 0.0)
            else:
                p = self._lookup(k, level)
            prior = p
            out[f"phi_{tag}"] = p
            out[f"phi_support_{tag}"] = s
            out[f"taxi_hat_{tag}"] = (t - aobt) - p * gap
            take = s >= self.min_rows
            best_phi = np.where(take, p, best_phi)
            best_sup = np.where(take, s, best_sup)
        out["phi"] = best_phi
        out["phi_support"] = best_sup
        out["block_hat"] = aobt + best_phi * gap
        out["taxi_hat"] = t - out["block_hat"].to_numpy()
        return out


def parent_of(keys: tuple[tuple[str, ...], ...], level: tuple[str, ...]) -> tuple[str, ...] | None:
    """The level immediately coarser than `level` in the chain, or None."""
    i = keys.index(level)
    return keys[i - 1] if i else None


@dataclass
class PhiCV:
    """Cross-fitted phi: K month-wise fits plus the fit on every training month.

    A training row must see a phi that did not use its own label, and a serving row
    must see one built from the same amount of data — the two requirements that
    exp-001's `phi_step` traded against each other, at a cost of 9.5 s on the
    six-month july fold (`experiments/exp-006-lirf/RESULTS.md` finding 12).

    Construction: one fit per held-out training month. A training row in month `m`
    is transformed by the fit that excluded `m`; a serving row (validation or
    ranking) is transformed by the **average of the K fits**, so both sides carry a
    coefficient estimated from K-1 months. `mode="full"` serves the all-months fit
    instead, which is the un-matched alternative and is measured beside it.

    Every mode emits the same columns, so a comparison between two of them changes
    the construction and nothing else.

        cv = PhiCV(PhiModel(weight="rows")).fit({m: frame for m in train_months})
        Xtr_m = cv.transform_train(m, frame_m)
        Xva   = cv.transform_serve(valid_frame, mode="avg")
    """

    base: PhiModel = field(default_factory=PhiModel)
    models: dict[str, PhiModel] = field(default_factory=dict)
    full: PhiModel | None = None

    def _clone(self) -> PhiModel:
        return PhiModel(keys=self.base.keys, weight=self.base.weight,
                        shrink=self.base.shrink, shrink_rows=self.base.shrink_rows,
                        min_abs_gap=self.base.min_abs_gap, min_rows=self.base.min_rows)

    def fit(self, frames_by_month: dict) -> "PhiCV":
        months = list(frames_by_month)
        self.models = {}
        for m in months:
            rest = [frames_by_month[o] for o in months if o != m]
            self.models[m] = self._clone().fit(rest if rest else [frames_by_month[m]])
        self.full = self._clone().fit([frames_by_month[m] for m in months])
        return self

    def transform_train(self, month: str, frame: pd.DataFrame) -> pd.DataFrame:
        """phi for a training month, from the fit that excluded that month."""
        return self.models[month].transform(frame)

    def transform_serve(self, frame: pd.DataFrame, mode: str = "avg") -> pd.DataFrame:
        """phi for a validation or ranking month."""
        if mode == "full":
            return self.full.transform(frame)
        outs = [m.transform(frame) for m in self.models.values()]
        out = outs[0].copy()
        for c in out.columns:
            if c == "MVT_ID_mvt":
                continue
            out[c] = np.mean([o[c].to_numpy("float64") for o in outs], axis=0)
        return out
