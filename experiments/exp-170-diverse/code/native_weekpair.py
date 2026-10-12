"""Week-pair validation splitter (exp-146, protocol `week-pair-1`; docs/VALIDATION.md).

Every fitted stage imports this module to decide which rows may give it labels.

* Five outer folds. Fold k holds one January block and one July block of 2025
  (`PAIRS[k]`) at all airports, plus one buffer day on each side of each block.
  A row's day is the UTC calendar date of its take-off time `MVT_TIME_UTC_mvt`,
  the same field that splits the ranking file into its two months.
* Inner rotation. For outer fold k, the four other pairs j are held in turn
  (`inner(k, j)`) to fit calibration: those models exclude k and j, and their
  predictions on pair j are the only labels a calibration for fold k may read.
  `inner(k, j).held` leaves out fold k's buffer days (a block day of pair j next
  to pair k), so every calibration row is a label row of `outer(k)`.
* Label isolation. A fitted artifact records the split it was fitted under
  (`Split.manifest()`). A consumer may use it only if the artifact excluded at
  least every day the consumer excludes (`Split.admits`). An artifact fitted on
  all of 2025 (`FINAL`) is admissible only for the final fit.
* Context masking. Context builders read only frames whose departure
  BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt are blank (`context_frame`,
  `assert_context_masked`), for every date, as in the ranking file.
* Clock-loss masks. `dark_days` selects airport-days by a fixed hash at a
  given removal rate; the frozen evaluation rates live in
  `experiments/exp-146-validation/frozen/clock_loss_days.csv`.

Pure date arithmetic and row masks. Nothing here reads a label.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

PROTOCOL = "week-pair-1"
YEAR = 2025
#: (January first day, last day, July first day, last day) per pair, inclusive.
PAIRS: dict[int, tuple[int, int, int, int]] = {
    1: (1, 5, 1, 6),
    2: (6, 12, 7, 13),
    3: (13, 19, 14, 20),
    4: (20, 26, 21, 27),
    5: (27, 31, 28, 31),
}
BUFFER_DAYS = 1
TIME_COLUMN = "MVT_TIME_UTC_mvt"
LABEL_COLUMNS = ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")
CLOCK_LOSS_SALT = "prc-exp146-clockloss"


class LabelLeak(AssertionError):
    """A fitted stage would read labels its split must not see."""


def _d(month: int, day: int) -> _dt.date:
    return _dt.date(YEAR, month, day)


def _span(a: _dt.date, b: _dt.date) -> list[_dt.date]:
    return [a + _dt.timedelta(days=i) for i in range((b - a).days + 1)]


def blocks(k: int) -> tuple[tuple[_dt.date, _dt.date], tuple[_dt.date, _dt.date]]:
    """((January start, end), (July start, end)) of pair k, inclusive."""
    ja, jb, la, lb = PAIRS[k]
    return (_d(1, ja), _d(1, jb)), (_d(7, la), _d(7, lb))


def held_days(k: int) -> tuple[_dt.date, ...]:
    """The scored days of pair k: its January block and its July block."""
    (a1, b1), (a2, b2) = blocks(k)
    return tuple(_span(a1, b1) + _span(a2, b2))


def buffer_days(k: int) -> tuple[_dt.date, ...]:
    """One day before and after each block of pair k (2024-12-31 included for pair 1)."""
    out = []
    for a, b in blocks(k):
        for i in range(1, BUFFER_DAYS + 1):
            out += [a - _dt.timedelta(days=i), b + _dt.timedelta(days=i)]
    return tuple(sorted(set(out)))


def pair_of(days) -> np.ndarray:
    """Pair index (1..5) of each UTC date; 0 for a date outside every block."""
    days = np.asarray(days, dtype="datetime64[D]")
    out = np.zeros(days.shape, dtype=np.int8)
    for k in PAIRS:
        out[np.isin(days, np.array(held_days(k), dtype="datetime64[D]"))] = k
    return out


def day_of(frame: pd.DataFrame | pd.Series) -> np.ndarray:
    """UTC calendar date (datetime64[D]) of each row's take-off time."""
    s = frame[TIME_COLUMN] if isinstance(frame, pd.DataFrame) else frame
    s = pd.to_datetime(s, utc=True)
    return s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[D]")


def _is_dep(frame: pd.DataFrame) -> np.ndarray:
    return (frame["PHASE_mvt"] == "DEP").to_numpy() if "PHASE_mvt" in frame else np.ones(len(frame), bool)


@dataclass(frozen=True)
class Split:
    """One label partition of the 2025 rows.

    `held`: the days this split scores. `excluded`: the days whose departure
    labels no stage fitted under this split may read (held days, their buffers,
    and for an inner split the outer fold's days too).
    """

    name: str
    held: tuple[str, ...]
    excluded: tuple[str, ...]
    _held_np: np.ndarray = field(init=False, repr=False, compare=False)
    _excl_np: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_held_np", np.array(self.held, dtype="datetime64[D]"))
        object.__setattr__(self, "_excl_np", np.array(self.excluded, dtype="datetime64[D]"))
        if not set(self.held) <= set(self.excluded):
            raise ValueError(f"{self.name}: held days must be excluded days")

    # ---- row masks
    def label_mask(self, frame: pd.DataFrame) -> np.ndarray:
        """Departure rows whose labels a stage fitted under this split may read."""
        return _is_dep(frame) & ~np.isin(day_of(frame), self._excl_np)

    def score_mask(self, frame: pd.DataFrame) -> np.ndarray:
        """Departure rows this split scores."""
        return _is_dep(frame) & np.isin(day_of(frame), self._held_np)

    def buffer_mask(self, frame: pd.DataFrame) -> np.ndarray:
        """Departure rows excluded from labels but not scored (buffers, other held pairs)."""
        d = day_of(frame)
        return _is_dep(frame) & np.isin(d, self._excl_np) & ~np.isin(d, self._held_np)

    def assert_training_frame(self, frame: pd.DataFrame, target=None,
                              what: str = "training frame") -> None:
        """Raise if a training frame carries a target on an excluded day.

        `target`: the column name, or an array aligned with `frame`, that the stage
        fits. A departure row on an excluded day whose target is present raises.
        Without `target`, EVERY departure row on an excluded day raises, whatever its
        columns hold: the target may sit in any column (for example z = y - r), and
        `TAXITIME_SEC_mvt` may be blank because of context masking.
        """
        if target is None:
            lab = np.ones(len(frame), dtype=bool)
        else:
            t = frame[target] if isinstance(target, str) else pd.Series(np.asarray(target))
            if len(t) != len(frame):
                raise ValueError(f"{what}: target is not aligned with the frame")
            lab = pd.Series(t).notna().to_numpy()
        bad = _is_dep(frame) & lab & np.isin(day_of(frame), self._excl_np)
        if bad.any():
            days = sorted({str(x) for x in day_of(frame)[bad]})
            raise LabelLeak(f"{what}: {int(bad.sum())} training departures on days excluded by "
                            f"{self.name}: {days[:5]}{' ...' if len(days) > 5 else ''}")

    # ---- ancestry
    def admits(self, ancestor: "Split | Mapping") -> bool:
        """True if an artifact fitted under `ancestor` may feed a stage under this split."""
        anc = ancestor if isinstance(ancestor, Split) else Split.from_manifest(ancestor)
        return set(self.excluded) <= set(anc.excluded)

    def assert_admits(self, ancestor: "Split | Mapping", what: str = "ancestor") -> None:
        anc = ancestor if isinstance(ancestor, Split) else Split.from_manifest(ancestor)
        missing = sorted(set(self.excluded) - set(anc.excluded))
        if missing:
            raise LabelLeak(f"{what} fitted under {anc.name} read labels of {len(missing)} days "
                            f"that {self.name} excludes: {missing[:5]}"
                            f"{' ...' if len(missing) > 5 else ''}")

    # ---- identity
    @property
    def token(self) -> str:
        blob = json.dumps({"protocol": PROTOCOL, "name": self.name, "held": self.held,
                           "excluded": self.excluded}, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def manifest(self) -> dict:
        """What a fitted artifact records about its split."""
        return {"protocol": PROTOCOL, "name": self.name, "held": list(self.held),
                "excluded": list(self.excluded), "token": self.token}

    @staticmethod
    def from_manifest(m: Mapping) -> "Split":
        if m.get("protocol") != PROTOCOL:
            raise LabelLeak(f"artifact split protocol {m.get('protocol')!r} is not {PROTOCOL}")
        s = Split(m["name"], tuple(m["held"]), tuple(m["excluded"]))
        if "token" in m and m["token"] != s.token:
            raise LabelLeak(f"split manifest {m['name']}: token does not match its days")
        return s


def _iso(days: Iterable[_dt.date]) -> tuple[str, ...]:
    return tuple(sorted({d.isoformat() for d in days}))


def outer(k: int) -> Split:
    """Outer fold k: score pair k; exclude it and its buffers from every fitted stage."""
    return Split(f"outer-{k}", _iso(held_days(k)), _iso(held_days(k) + buffer_days(k)))


def inner(k: int, j: int) -> Split:
    """Inner calibration split of outer fold k: exclude k, j and both buffers.

    It scores pair j WITHOUT fold k's buffer days: when j is next to k, the block day
    of j beside k is a buffer of k, and a calibration for fold k must not read it.
    `inner(k, j)` and `inner(j, k)` exclude the same days, so one fit serves both.
    """
    if j == k or j not in PAIRS or k not in PAIRS:
        raise ValueError(f"inner({k}, {j}) is not a rotation of the five pairs")
    held = set(held_days(j)) - set(buffer_days(k))
    return Split(f"inner-{k}-{j}", _iso(held),
                 _iso(held_days(k) + buffer_days(k) + held_days(j) + buffer_days(j)))


def inner_rotation(k: int) -> tuple[int, ...]:
    return tuple(j for j in PAIRS if j != k)


def outer_splits() -> tuple[Split, ...]:
    return tuple(outer(k) for k in PAIRS)


def month_hold(month: str) -> Split:
    """Season check: hold a whole 2025 month ('2025-01', '2025-07', '2025-12') with buffers."""
    start = pd.Timestamp(month + "-01").date()
    end = (pd.Timestamp(month + "-01") + pd.offsets.MonthEnd(1)).date()
    days = _span(start, end)
    buf = [start - _dt.timedelta(days=1), end + _dt.timedelta(days=1)]
    return Split(f"month-{month}", _iso(days), _iso(days + buf))


#: The final fit: all 2025 labels. Admissible only as an ancestor of another final stage.
FINAL = Split("final", (), ())


def join_outer(preds: Mapping[int, pd.DataFrame], ids_days: pd.DataFrame,
               column: str = "pred") -> pd.Series:
    """Join the five outer predictions into one series indexed by MVT_ID_mvt.

    `preds[k]` holds columns MVT_ID_mvt and `column` for pair k's held rows.
    `ids_days` holds MVT_ID_mvt and `pair` for every January and July departure.
    Raises unless each held row is predicted exactly once, by its own fold.
    """
    want = ids_days.set_index("MVT_ID_mvt")["pair"]
    parts = []
    for k in PAIRS:
        p = preds[k]
        if p["MVT_ID_mvt"].duplicated().any():
            raise ValueError(f"outer-{k}: duplicate ids")
        own = want.reindex(p["MVT_ID_mvt"].to_numpy())
        if own.isna().any() or (own.to_numpy() != k).any():
            raise LabelLeak(f"outer-{k}: {int((own.to_numpy() != k).sum())} predicted rows are "
                            f"not held by pair {k}")
        n_want = int((want == k).sum())
        if len(p) != n_want:
            raise ValueError(f"outer-{k}: {len(p)} predictions for {n_want} held rows")
        parts.append(pd.Series(p[column].to_numpy("float64"), index=p["MVT_ID_mvt"].to_numpy()))
    out = pd.concat(parts)
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("non-finite outer predictions")
    return out


def context_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy with departure BLOCK_TIME_UTC_mvt and TAXITIME_SEC_mvt blank on every row."""
    out = frame.copy()
    dep = _is_dep(out)
    out.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    out["TAXITIME_SEC_mvt"] = out["TAXITIME_SEC_mvt"].astype("float64")
    out.loc[dep, "TAXITIME_SEC_mvt"] = np.nan
    return out


def assert_context_masked(frame: pd.DataFrame, what: str = "context frame") -> None:
    """Raise if a context builder is handed any departure label."""
    dep = _is_dep(frame)
    for c in LABEL_COLUMNS:
        if c in frame and frame.loc[dep, c].notna().any():
            raise LabelLeak(f"{what}: {int(frame.loc[dep, c].notna().sum())} departure rows carry {c}")


# ------------------------------------------------------------------ clock-loss masks
def _hash_u64(airport: str, day: _dt.date | str) -> int:
    s = f"{CLOCK_LOSS_SALT}|{airport}|{day if isinstance(day, str) else day.isoformat()}"
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big")


def dark_days(airport: str, month: str, rate: float) -> tuple[str, ...]:
    """The floor(rate * D + 0.5) days of one airport-month with the lowest fixed hash.

    `month` is 'YYYY-MM'; D is its number of days. The same airport-month and
    rate always give the same days, for every candidate and comparator.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate {rate} outside [0, 1]")
    start = pd.Timestamp(month + "-01")
    days = [(start + pd.Timedelta(days=i)).date().isoformat() for i in range(start.days_in_month)]
    n = int(np.floor(rate * len(days) + 0.5))
    return tuple(sorted(sorted(days, key=lambda d: _hash_u64(airport, d))[:n]))


def removal_rate(cov_2025: float, cov_2026: float) -> float:
    """Share of 2025 clock observations to remove so 2025 matches 2026 coverage; never adds."""
    if cov_2025 <= 0:
        return 0.0
    return float(min(1.0, max(0.0, 1.0 - cov_2026 / cov_2025)))


def training_rate_month(month: int) -> int:
    """Which frozen evaluation rate a masked TRAINING view uses: January's for Oct-Mar, July's for Apr-Sep."""
    return 1 if month in (10, 11, 12, 1, 2, 3) else 7


def fingerprint() -> str:
    """Hash of every split of the protocol; record it in each run manifest."""
    splits = [outer(k).manifest() for k in PAIRS]
    splits += [inner(k, j).manifest() for k in PAIRS for j in inner_rotation(k)]
    splits += [month_hold(m).manifest() for m in ("2025-01", "2025-07", "2025-12")]
    blob = json.dumps({"protocol": PROTOCOL, "splits": splits, "salt": CLOCK_LOSS_SALT},
                      sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()
