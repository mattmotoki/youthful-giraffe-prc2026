"""exp-046-surfacefit step 1: the surface-observation feature block.

One row per departure, computed from `ranking.parquet`-legal columns and the
adsb.lol ground summaries alone.  **No label enters this file.**  A DEPARTURE's
`BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` are never read here, not even on the
training side.  The turnaround pass reads an ARRIVAL's `BLOCK_TIME_UTC_mvt`, which
`mask_like_ranking` never blanks and which is an input on 100 % of ARR rows: the block
is a pure function of inputs, so the same code computes it for a masked validation
month and for the ranking file.

The block reads exactly one schema: `exp-045-harvest`'s per-airport, per-day segment
summary (`artifacts/<exp>/summaries/<airport>/<day>.parquet`).  The pilot's week is
re-expressed into that schema by `code/summaries.py`, so there is no second code path.

Identity, following `exp-043-pilot/code/match.py` §"Two readings of my own":

    the modal callsign of a ground segment that ends in a take-off, joined to the
    airport's departures, with `obs_wheels_off` inside +/- 15 min of `MVT_TIME_UTC_mvt`

with two changes the serving side forces, both recorded in `SPEC.md`:

*   the day key is the TAKE-OFF day, not the off-block day, because the off-block day
    is blanked in the ranking file.  Summaries for the take-off day and its two
    neighbours are loaded and the window discriminates.  On the pilot's week the two
    keys agree on every matched row.
*   the join is per `(airport, callsign)`, because ten airports share a callsign space.

Ambiguity is resolved by nearest wheels-off in both directions and counted, never
dropped silently.

    .venv/bin/python experiments/exp-046-surfacefit/code/block.py --self-test
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# PROBE COPY: anchored by NAME, because this file sits two directories deeper.
EXP = next(q for q in Path(__file__).resolve().parents
           if q.name == "exp-046-surfacefit")
ROOT = EXP.parents[1]

WINDOW = 900.0                      # s: the +/- 15 min identity window (pilot)
#: The wide rule (desk, 2026-09-08).  At LIRF, EDDF and EGLL the network loses the
#: aircraft before the runway, so the observed take-off is minutes early or missing --
#: but the OFF-BLOCK observation is at the stand and is unaffected.  A broken take-off
#: clock therefore need not discard the match.  A departure the strict rule leaves
#: unmatched is matched on (airport, callsign) plus an off-block instant that precedes
#: its take-off by a plausible taxi time, preferring a take-off inside the wider window
#: where one exists.  The clock-check status rides along as a quality feature and the
#: fold decides what such a row is worth; nothing about the strict rows changes.
WIDE_WINDOW = 3600.0                # s: the +/- 60 min take-off window of the wide rule
WIDE_MAX_TAXI = 7200.0              # s: the largest implied taxi a wide match may carry
WIDE_MIN_TAXI = 60.0                # s: the smallest; below it the segment is not a taxi
#: s: the ground run of a departure must END within this of the take-off.  The implied
#: taxi alone is not identity: a callsign flown twice in a day would lend the morning
#: pushback to the evening flight (Sol round 1, blocking 8).
#:
#: The value is MEASURED, not chosen (`analysis/segend-sweep.json`, round 2 FALLBACK,
#: blocking 3).  On real strict matches at EHAM the ground run ends within 300 s of the
#: take-off on 99.94 % of rows and within 900 s on all of them, but the wide rule exists
#: for airports where the network loses the aircraft BEFORE the runway, so the window
#: must be looser than that.  Sweeping it against reach and against a placebo cohort
#: whose take-offs are displaced by an hour:
#:
#:     window   EHAM real wide   placebo +1.0 h   2026-01 wide rows
#:      600 s        4                 1                 10
#:      900 s        4                 1                 10
#:     1800 s        5                 1                103
#:     3600 s        5                24                119
#:
#: 1800 s is the knee: it keeps every real wide match and 87 % of the ranking-side ones,
#: and holds the false-match rate at 1 row of 5,131 where 3600 s gives 24.  The last
#: 16 ranking rows are not worth twenty-four times the false matches.  No label is read
#: anywhere in this: `seg_end` and `MVT_TIME_UTC_mvt` are both inputs.
WIDE_SEG_END_WINDOW = 1800.0
PROBE: dict = {}

MODES = {"strict": 0, "wide_window": 1, "no_takeoff": 2, "turnaround": 3,
         "turnaround_ungated": 4}
MAX_TURNAROUND_S = 86400.0   # the chain: an airframe turning round inside a day
CHECK_S = 60.0               # the take-off agreement, the key's last clause
#: the repaired rule is the default everywhere; `exp-043-pilot/code/segments.py` owns it
REQUIRE_ALT = True
AIRPORTS = ["EHAM", "LFPG", "EGLL", "LTFM", "LEMD", "EDDF", "LEBL", "LIRF", "EDDM", "LSZH"]
#: where a day's summaries may live, in order of preference
SUMMARY_ROOTS = [ROOT / "artifacts/exp-045/summaries", ROOT / "artifacts/exp-046/summaries"]
BOXES = ROOT / "experiments/exp-045-harvest/analysis/boxes.json"
MANIFEST = ROOT / "experiments/exp-045-harvest/manifest.json"
LOCAL_RULE = ROOT / "artifacts/exp-046/summaries/rule.json"

#: **The pinned fold population** (desk ruling, 2026-09-09, standing policy for both
#: lanes).  A fold block is pinned to the airport-day population it was priced on, and
#: pricing, the gates, the dry run and the serve all read that pinned set, so a delivery
#: landing mid-block cannot change what a candidate was gated on.  A delivery enters a
#: fold only at a recorded re-price: a new population file, a new `arms.json`, the gate
#: re-read and a ledger row.  Written by `code/population.py`; `_summary_path` is the one
#: choke point every read and every cache key goes through, so pinning it pins both.
FOLD_POP = EXP / "analysis/fold-population.json"
_PIN: set[tuple[str, str]] | None = None


def pin_file() -> dict:
    """The population file, or `{}` when nothing is pinned."""
    return json.loads(FOLD_POP.read_text()) if FOLD_POP.exists() else {}


def pin_for_month(month: str) -> set[tuple[str, str]] | None:
    """The `(airport, day)` set a pinned month may read, or `None` for an unpinned one.

    An unpinned month reads whatever is on disk, which is right for the ranking months
    (their block is not a priced fold table) and for a diagnostic.
    """
    rec = pin_file().get("months", {}).get(month)
    if rec is None:
        return None
    return {(x["airport"], x["day"]) for x in rec["airport_days"]}


@contextlib.contextmanager
def pinned(pop: set[tuple[str, str]] | None):
    """Read only `pop` inside this block; a day outside it reads as absent."""
    global _PIN
    old = _PIN
    _PIN = pop
    try:
        yield
    finally:
        _PIN = old

#: the ranking-legal columns the block reads.  `BLOCK_TIME_UTC_mvt` and
#: `TAXITIME_SEC_mvt` are deliberately absent and must stay absent.
COHORT_COLS = ["MVT_ID_mvt", "PHASE_mvt", "FLIGHT_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt",
               "CALLSIGN_flt", "AOBT_3_flt", "STAND_mvt", "RUNWAY_mvt"]

#: every column the block produces, in order.  `_flag` columns are categorical.
FEATURES = [
    "obs_taxi", "obs_taxi_motion", "obs_seg_taxi", "dt_takeoff",
    "time_to_250m", "seen_before_move_s", "stand_stop_s", "turnaround_anchored",
    "seg_points", "n_ground_points", "seg_dt_median_s", "seg_gap_max_s", "seg_span_s",
    "callsign_n", "stand_lat", "stand_lon", "stand_dist_km", "stand_sin", "stand_cos",
    "arr_present", "arr_taxi_in_s", "arr_stand_occupancy_s", "arr_gap_reception_s",
    "obs_ambiguous", "obs_n_candidates",
    # the quality of the identity itself, so a fold can price a wide match separately
    "match_mode", "clock_ok", "has_wheels_off", "matched_on_modal",
]

#: Join the row's callsign against EVERY form the ground run broadcast, not only the
#: modal one (`analysis/identity.json`, 2026-01: the modal join loses 1,222 rows at EDDF,
#: 1,042 at EGLL, 893 at LIRF and 3,885 over the ten airports, because one ground run
#: carries the inbound and the outbound name and the modal can be either).  The form the
#: match used rides along as `matched_on_modal`, so a fold can price a non-modal match
#: separately, and `FORMS = False` restores the modal-only join exactly for the test.
FORMS = True
_ZEROS = re.compile(r"^([A-Z]+)0+(\d.*)$")
_KEEP = re.compile(r"[^A-Z0-9]")


def norm_callsign(s: pd.Series) -> pd.Series:
    """Callsign, comparable across the file and the transponder.

    Upper case, punctuation removed, and the leading zeros of the numeric run stripped:
    the file writes `BAW07A` where the transponder sends `BAW7A`.  The airline prefix is
    never rewritten -- that is `prefix_map()`'s job and a different claim.
    """
    u = s.fillna("").astype(str).str.upper().str.replace(_KEEP, "", regex=True)
    return u.str.replace(_ZEROS, r"\1\2", regex=True)


def broadcast_forms(dep: pd.DataFrame) -> pd.DataFrame:
    """One row per (departure segment, callsign form), `callsigns_all` being `|`-joined.

    The modal form is kept first so a tie between two forms of one segment resolves to
    the modal, which is what the modal-only join would have chosen.
    """
    d = dep.copy()
    d["form"] = d.callsigns_all.fillna(d.callsign_modal).astype(str).str.split("|")
    d = d.explode("form")
    d["form"] = d.form.str.strip()
    d = d[d.form.ne("") & ~d.form.str.fullmatch(r"[@?_]+", na=False)]
    d["join_key"] = norm_callsign(d.form)
    d = d[d.join_key.ne("")]                 # a key that normalises away matches nothing
    d["matched_on_modal"] = d.form.eq(d.callsign_modal.str.strip()).astype("float64")
    return d.sort_values("matched_on_modal", ascending=False).drop_duplicates(
        ["seg_key", "join_key"])


def arrival_keys(arrivals: pd.DataFrame | None) -> dict[str, set[str]]:
    """`airport -> the normalised callsigns ARRIVING there`.

    The identity a stale inbound broadcast carries.  `analysis/identity-read.json` found
    that 99.4 to 99.9 % of the clocked ground segments no departure claims are broadcasting
    a ranking ARRIVAL's callsign into the same airport: a departure run whose transponder
    was never switched off the inbound flight.

    Arrival callsigns are INPUTS.  Nothing about the target is read here.
    """
    if arrivals is None or not len(arrivals):
        return {}
    a = arrivals
    if "PHASE_mvt" in a.columns:
        a = a[a.PHASE_mvt == "ARR"]
    if "ADES_mvt" not in a.columns or "CALLSIGN_flt" not in a.columns:
        return {}
    a = a[["ADES_mvt", "CALLSIGN_flt"]].dropna()
    k = norm_callsign(a.CALLSIGN_flt)
    ok = k.ne("")
    return {str(ap): set(g) for ap, g in k[ok].groupby(a.ADES_mvt[ok])}


def arrival_legs(arrivals: pd.DataFrame | None) -> pd.DataFrame:
    """The inbound legs a chain may use: identity, stand, operator, type, in-block.

    Every column here is an INPUT.  `mask_like_ranking` blanks `BLOCK_TIME_UTC_mvt` on
    DEP rows only, so an arrival's in-block is untouched and present on 100 % of ARR rows;
    `STAND_mvt` is on 100 % of both phases.  No DEPARTURE in-block is read.
    """
    cols = ["ADES_mvt", "CALLSIGN_flt", "STAND_mvt", "AIRCRAFT_OPERATOR_flt",
            "AIRCRAFT_TYPE_mvt", "BLOCK_TIME_UTC_mvt"]
    if arrivals is None or not len(arrivals):
        return pd.DataFrame(columns=cols + ["key", "in_block"])
    a = arrivals
    if "PHASE_mvt" in a.columns:
        a = a[a.PHASE_mvt == "ARR"]
    miss = [c for c in cols if c not in a.columns]
    if miss:                       # never silently place on a column that is not there
        raise SystemExit(f"turnaround pass: arrivals lack {miss}")
    a = a[cols].copy()
    a["key"] = norm_callsign(a.CALLSIGN_flt)
    a["in_block"] = (a.BLOCK_TIME_UTC_mvt
                     - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    return a[a.key.ne("") & a.in_block.notna()]


# ---------------------------------------------------------------- the cohort


def prefix_map(train: pd.DataFrame) -> pd.Series:
    """`FLIGHT_mvt` letter prefix -> modal `CALLSIGN_flt` letter prefix.

    `train` must contain only months that are NOT being scored; the caller owns that.
    """
    t = train.dropna(subset=["FLIGHT_mvt", "CALLSIGN_flt"])
    fp = t.FLIGHT_mvt.str.extract(r"^([A-Z]+)")[0]
    cp = t.CALLSIGN_flt.str.extract(r"^([A-Z]+)")[0]
    ok = fp.notna() & cp.notna()
    if not ok.any():
        return pd.Series(dtype=object)
    return pd.DataFrame({"f": fp[ok], "c": cp[ok]}).groupby("f")["c"].agg(
        lambda s: s.mode().iat[0])


def cohort(frame: pd.DataFrame, pm: pd.Series) -> pd.DataFrame:
    """Departure rows with the broadcast callsign, given or reconstructed."""
    d = frame
    if "PHASE_mvt" in d.columns:
        d = d[d.PHASE_mvt == "DEP"]
    d = d[d.ADEP_mvt.isin(AIRPORTS)].copy()
    fpre = d.FLIGHT_mvt.str.extract(r"^([A-Z]+)")[0]
    fsuf = d.FLIGHT_mvt.str.replace(r"^[A-Z]+", "", regex=True)
    mapped = fpre.map(pm) if len(pm) else pd.Series(index=d.index, dtype=object)
    recon = (mapped + fsuf.fillna("")).where(mapped.notna())
    d["callsign"] = d.CALLSIGN_flt.fillna(recon).str.strip()
    d["callsign_source"] = np.where(d.CALLSIGN_flt.notna(), "flt",
                                    np.where(recon.notna(), "prefix", "none"))
    d["has_nm"] = d.AOBT_3_flt.notna()
    d["mvt_s"] = (d.MVT_TIME_UTC_mvt - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    d["aobt_s"] = (d.AOBT_3_flt - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    d["takeoff_day"] = d.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")
    return d


# ------------------------------------------------------------- the summaries


def rule_by_day() -> dict[tuple[str, str], str]:
    """`(summary root, day) -> summaries_rule_sha256`.

    The pilot's reviewer repaired the off-block rule on 2026-09-08 after four months had
    been harvested under the shipped one, and the harvest now backfills already-run
    months from their stored points (`exp-045-harvest/manifest.json`, `rule_note`).  A
    fold table or a served candidate must therefore rest on summaries written by ONE
    rule, and this is where the version comes from.

    The key carries the ROOT as well as the day (Sol round 1, blocking 2): the harvest's
    manifest speaks only for files under `artifacts/exp-045/summaries` and the local
    sidecar only for files under `artifacts/exp-046/summaries`, so a day present in both
    trees must not take the other tree's version.
    """
    out: dict[tuple[str, str], str] = {}
    harvest, local = str(SUMMARY_ROOTS[0]), str(SUMMARY_ROOTS[1])
    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text())
        for rec in man.get("months", {}).values():
            for day, d in rec.get("days", {}).items():
                if d.get("summaries_rule_sha256"):
                    out[(harvest, day)] = d["summaries_rule_sha256"]
    if LOCAL_RULE.exists():
        loc = json.loads(LOCAL_RULE.read_text())
        for day in loc.get("days", []):
            out[(local, day)] = loc["rule_sha256"]
    return out


def _want_days(days: list[str]) -> set[str]:
    """The summary days one run reads: each take-off day and the day either side."""
    want: set[str] = set()
    for d in days:
        ts = pd.Timestamp(d)
        want |= {(ts + pd.Timedelta(days=k)).strftime("%Y-%m-%d") for k in (-1, 0, 1)}
    return want


def _summary_path(airport: str, day: str) -> Path | None:
    # a pinned run reads only the airport-days its population names: a day delivered
    # after the pin reads as absent, exactly as it did when the block was priced
    if _PIN is not None and (airport, day) not in _PIN:
        return None
    for r in SUMMARY_ROOTS:
        p = r / airport / f"{day}.parquet"
        if p.exists():
            return p
    return None


def _rule_of(p: Path, day: str, rules: dict[tuple[str, str], str]) -> str | None:
    return rules.get((str(p.parents[1]), day))


def present_days(airports: list[str], days: list[str]) -> list[tuple[str, str, Path]]:
    """Every `(airport, day, path)` a run over `days` would read, that exists."""
    out = []
    for a in airports:
        for d in sorted(_want_days(days)):
            p = _summary_path(a, d)
            if p is not None:
                out.append((a, d, p))
    return out


def population_triples(airports: list[str], days: list[str]) -> list[dict]:
    """The airport-days a run over `days` reads, each with what identifies its content.

    ONE definition of a population, used by the fold pin (`code/population.py`), by the
    cache key (`arms.block_cache_key`) and by the ranking-population record the serve
    writes. Two implementations of "the same" hash is what put 87 ranking rows on both
    sides of a boundary once already, so there is one here.
    """
    rules = rule_by_day()
    return [{"airport": a, "day": d, "root": str(p.parents[1].relative_to(ROOT)),
             "bytes": int(p.stat().st_size),
             "summaries_rule_sha256": _rule_of(p, d, rules)}
            for a, d, p in present_days(airports, days)]


def population_key(triples: list[dict], rule: str | None) -> str:
    """The 16-hex key of a population: the rule, then each day's size and version."""
    h = hashlib.sha256()
    h.update((rule or "none").encode())
    for x in triples:
        h.update(f"{x['airport']}/{x['day']}/{x['bytes']}/"
                 f"{x['summaries_rule_sha256']}\n".encode())
    return h.hexdigest()[:16]


def one_rule(spans: list[tuple[list[str], list[str]]]) -> tuple[str | None, dict]:
    """ONE `summaries_rule_sha256` for every span a run will read, or `None`.

    The desk's instruction of 2026-09-08: "build the fold table and the serve only on
    summaries carrying one and the same version".  A span is one `(airports, days)`
    pair; the fold table and the serve pass every span they will read TOGETHER, so the
    version is resolved once across all fits, all folds and the ranking rows rather than
    per month (Sol round 1, blocking 2).  The harvest re-derives already-run months from
    stored points, so during a migration two versions sit on disk at once: this returns
    the one with the widest cover and reports the rest, and `load_summaries` then refuses
    the rest instead of mixing them in.  `None` means no file carries a known version,
    and the caller must fail closed.
    """
    rules = rule_by_day()
    seen: dict[tuple[str, str], str | None] = {}
    for airports, days in spans:
        for a, d, p in present_days(airports, days):
            # a day on the boundary of two spans is ONE file and is counted once
            # (round 2 FALLBACK, non-blocking)
            seen[(a, d)] = _rule_of(p, d, rules)
    counts: dict[str | None, int] = {}
    for k in seen.values():
        counts[k] = counts.get(k, 0) + 1
    if not counts:
        return None, {"airport_days_by_rule": {}, "n_rules": 0, "chosen": None,
                      "airport_days_rejected": 0, "airport_days_present": 0}
    known = {k: v for k, v in counts.items() if k is not None}
    best = max(known, key=lambda k: (known[k], k)) if known else None
    return best, {"airport_days_by_rule": {str(k): v for k, v in counts.items()},
                  "n_rules": len(counts), "chosen": best,
                  "airport_days_present": sum(counts.values()),
                  "airport_days_rejected": sum(v for k, v in counts.items() if k != best)}


def dominant_rule(airports: list[str], days: list[str]) -> tuple[str | None, dict]:
    """`one_rule` for a single span.  Kept for the diagnostic callers."""
    return one_rule([(airports, days)])


#: Airport-days whose summaries are ABSENT OR UNUSABLE, held out of the observation
#: entirely (desk, 2026-09-08 14:45 and 15:05).  EDDF's feeder went out over five days;
#: no row matches on them, no segment of theirs feeds a neighbour, and both lanes see the
#: same no-clock class.  **This is not a receipt threshold**: a matched row has its clock
#: whatever the day's receipt was, and a low-receipt day simply offers the no-clock lane
#: fewer neighbours, so a dip is never a reason to hold.  The rates below are the
#: harvest's (`exp-045-harvest/analysis/receipt.json`), recorded to identify the days;
#: the list itself is the harvest's diagnosis and nothing is held without a line.
HELD_AIRPORT_DAYS = {
    ("EDDF", "2026-01-29"): 0.1347,
    ("EDDF", "2026-01-30"): 0.0985,
    ("EDDF", "2026-01-31"): 0.1012,
}
HELD_REASON = ("EDDF feeder outage 2026-01-29..31 (desk 2026-09-08 14:45, corrected "
               "from the harvest's receipt.json at 49c953b7 on 2026-09-09)")

_SHORT: set[str] | None = None


def short_days() -> set[str]:
    """Days whose ARCHIVE is short of the size the source published for it.

    The harvest's `manifest.py` (91b64a5) records `published_bytes` and `published_short`
    on every day record.  A short archive is a partial one: its summaries are built from
    part of the day, so the day looks empty rather than absent, and no receipt rate can
    tell the two apart.  Such a day is held for EVERY airport, because the archive is the
    day, not one airport's feeder.

    This is a DIFFERENT criterion from `HELD_AIRPORT_DAYS`, which is one airport's outage
    diagnosed by hand; the two are recorded apart and neither is a receipt threshold.
    """
    global _SHORT
    if _SHORT is None:
        out: set[str] = set()
        if MANIFEST.exists():
            man = json.loads(MANIFEST.read_text())
            for rec in man.get("months", {}).values():
                for day, d in rec.get("days", {}).items():
                    if d.get("published_short"):
                        out.add(day)
        _SHORT = out
    return _SHORT


def load_summaries(airports: list[str], days: list[str],
                   require_rule: str | None = None,
                   allow_any_rule: bool = False) -> pd.DataFrame:
    """Every segment of `airports` on `days` and the day either side of each.

    `require_rule` keeps only the days whose `summaries_rule_sha256` equals it; days
    written by another rule are reported in `days_wrong_rule` and never mixed in.
    Where a summary carries `airborne_alt_ok` the rule is additionally NORMALISED here,
    so a file written before the repair and one written after give the same rows.

    **Fail closed** (Sol round 1, blocking 2): with no `require_rule` this refuses to
    read a file at all, rather than reading every version it finds.  A diagnostic that
    genuinely wants whatever is on disk says so with `allow_any_rule=True`; the fold
    table and the serve never do.
    """
    rules = rule_by_day()
    want = _want_days(days)
    frames, have, missing, wrong, held, short = [], [], [], [], [], []
    for a in airports:
        for d in sorted(want):
            p = _summary_path(a, d)
            if p is None:
                missing.append((a, d))
                continue
            if (a, d) in HELD_AIRPORT_DAYS:
                held.append((a, d, HELD_AIRPORT_DAYS[(a, d)]))
                continue
            if d in short_days():
                short.append((a, d))
                continue
            got = _rule_of(p, d, rules)
            if require_rule is None and not allow_any_rule:
                raise SystemExit(
                    f"load_summaries: {a} {d} exists under {p.parents[1]} but no rule "
                    "version was required. Resolve one with block.one_rule() and pass "
                    "it as require_rule, or pass allow_any_rule=True for a diagnostic.")
            if require_rule is not None and got != require_rule:
                wrong.append((a, d, got))
                continue
            s = pd.read_parquet(p)
            # normalise the repair where the file carries the evidence for it: a
            # `departure` whose first airborne point had a null altitude is not a
            # departure under the repaired rule.
            if REQUIRE_ALT and "airborne_alt_ok" in s.columns:
                bad = s.kind.eq("departure") & ~s.airborne_alt_ok.fillna(False).astype(bool)
                s.loc[bad, "kind"] = "other"
                s.loc[bad, ["obs_wheels_off", "t_airborne"]] = np.nan
            s["airport"] = a
            s["summary_day"] = d
            s["rule_sha256"] = got
            frames.append(s)
            have.append((a, d))
    if not frames:
        out = pd.DataFrame(columns=["airport", "summary_day"])
    else:
        out = pd.concat(frames, ignore_index=True)
    out.attrs["days_present"] = have
    out.attrs["days_missing"] = missing
    out.attrs["days_wrong_rule"] = wrong
    out.attrs["days_held"] = held
    out.attrs["days_held_reason"] = HELD_REASON if held else None
    out.attrs["days_held_short"] = short
    out.attrs["rules_seen"] = sorted({r for r in (out.rule_sha256.unique().tolist()
                                                  if "rule_sha256" in out.columns
                                                  else []) if isinstance(r, str)})
    out.attrs["normalised_by_airborne_alt_ok"] = bool(
        len(frames) and all("airborne_alt_ok" in f.columns for f in frames))
    return out


def _boxes() -> dict:
    b = json.loads(BOXES.read_text())
    return b["boxes"] if isinstance(b, dict) and "boxes" in b else b


def _centres() -> dict[str, tuple[float, float]]:
    out = {}
    for a, v in _boxes().items():
        if isinstance(v, dict) and "lat_min" in v:
            out[a] = ((v["lat_min"] + v["lat_max"]) / 2, (v["lon_min"] + v["lon_max"]) / 2)
        elif isinstance(v, dict) and "lat" in v:
            out[a] = (float(np.mean(v["lat"])), float(np.mean(v["lon"])))
    return out


# ------------------------------------------------------------------ the join


def _arrival_side(seg: pd.DataFrame) -> pd.DataFrame:
    """For every segment, the segment of the same aircraft that precedes it.

    Joined on `(airport, icao)` and time order, NOT on the callsign: an aircraft's
    inbound leg carries a different callsign from its outbound leg, so a callsign
    join would find almost nothing.  Yields the observed taxi-in of the inbound leg
    and the observed stand occupancy before pushback.

    What the three numbers ARE, precisely (round 2 FALLBACK, non-blocking: the names
    overstate them).  `arr_taxi_in_s` is the whole span of the preceding segment, so on
    an arrival it is the taxi-in PLUS whatever stand time was still being received before
    the trace stopped -- an upper bound on taxi-in, not taxi-in.  `arr_stand_occupancy_s`
    is the reception gap between the two segments, which is stand time only if the
    network kept hearing the aircraft on the stand.  `arr_gap_reception_s` is the same
    number under a name that says what it measures.  All three are inputs and the model
    is free to use them; no table should quote them as taxi-in or as turnaround.

    Keyed by the caller's own `seg_key`, and computed for EVERY segment, not only for
    the ones the strict rule calls a departure: the wide rule matches `other` segments
    too, and they need the same arrival side.  (Sol round 1, blocking 1: the previous
    version numbered the departure subset 0..N-1 and then joined that number to the
    whole frame's `seg_key`, so the feature reached exactly one row of the July block.)

    One aircraft's segments at one airport are a partition of its trace, so they are
    disjoint in time and the IMMEDIATELY preceding segment in `seg_start` order is the
    last preceding one.  That makes this a group shift rather than a self-merge, which
    a month of summaries could not afford.
    """
    cols = ["arr_present", "arr_taxi_in_s", "arr_stand_occupancy_s", "arr_gap_reception_s"]
    cur = seg[["airport", "icao", "seg_key", "kind", "seg_start", "seg_end"]].copy()
    if not len(cur):
        return cur.assign(**{c: (False if c == "arr_present" else np.nan) for c in cols})
    cur = cur.sort_values(["airport", "icao", "seg_start", "seg_end"])
    g = cur.groupby(["airport", "icao"], sort=False)
    prev_end = g.seg_end.shift(1)
    prev_start = g.seg_start.shift(1)
    prev_kind = g.kind.shift(1)
    ok = prev_end.notna() & (prev_end <= cur.seg_start)
    cur["arr_present"] = (ok & prev_kind.eq("arrival")).to_numpy(bool)
    cur["arr_taxi_in_s"] = np.where(cur.arr_present, prev_end - prev_start, np.nan)
    cur["arr_stand_occupancy_s"] = np.where(ok, cur.seg_start - prev_end, np.nan)
    cur["arr_gap_reception_s"] = cur["arr_stand_occupancy_s"]
    return cur[["seg_key", *cols]]


def build(frame: pd.DataFrame, train_for_prefix: pd.DataFrame,
          airports: list[str] | None = None, wide: bool = True,
          require_rule: str | None = None,
          allow_any_rule: bool = False,
          arrivals: pd.DataFrame | None = None,
          chain: bool = True) -> pd.DataFrame:
    """The feature block for every departure of `frame`.

    `train_for_prefix` supplies the airline prefix map and must exclude the months
    of `frame`.  Returns one row per departure of `frame`, in `frame`'s own order.

    `arrivals` turns on the THIRD pass (`analysis/chain-spec.md`): the ARR rows of the
    same ranking file.  Without them the pass cannot run, and `chain=True` with no
    arrivals is reported in `attrs["turnaround"]` rather than passed over in silence.
    """
    pm = prefix_map(train_for_prefix)
    coh = cohort(frame, pm)
    airports = airports or sorted(coh.ADEP_mvt.unique())
    days = sorted(coh.takeoff_day.unique())
    seg = load_summaries(airports, days, require_rule, allow_any_rule)
    # `seg` is re-bound by merges below and pandas does not carry `attrs` through a
    # merge, so the loader's report is captured here, once, and used from this dict.
    seg_attrs = dict(seg.attrs)
    n_days_present = len(seg_attrs.get("days_present", []))

    if not len(seg):
        out = coh[["MVT_ID_mvt"]].copy()
        out["obs_icao"] = None
        for c in ["obs_wheels_off", *FEATURES]:
            out[c] = np.nan
        out["obs_matched"] = False
        out["obs_has_clock"] = False
        out["reach_class"] = "unmatched"
        out["match_mode"] = np.nan
        out.attrs["days_present"] = 0
        out.attrs["days_present_list"] = []
        out.attrs["turnaround"] = {"ran": False, "why": "no segments", "placed": 0}
        out.attrs["days_missing"] = seg_attrs.get("days_missing", [])
        out.attrs["days_wrong_rule"] = seg_attrs.get("days_wrong_rule", [])
        out.attrs["days_held"] = seg_attrs.get("days_held", [])
        out.attrs["rules_seen"] = seg_attrs.get("rules_seen", [])
        return out

    seg = seg.copy()
    seg["seg_key"] = np.arange(len(seg))
    # the strict rule's candidates: a ground run that ends in a take-off
    dep = seg[(seg.kind == "departure") & seg.obs_wheels_off.notna()].copy()
    arr = _arrival_side(seg)
    seg = seg.merge(arr, on="seg_key", how="left", validate="one_to_one")
    seg["callsign_modal"] = seg.callsign_modal.str.strip()
    dep = seg[(seg.kind == "departure") & seg.obs_wheels_off.notna()].copy()

    keys = coh[["MVT_ID_mvt", "ADEP_mvt", "callsign", "mvt_s"]].copy()
    if FORMS:
        keys["join_key"] = norm_callsign(keys.callsign)
        keys = keys[keys.join_key.ne("")]    # a row with no callsign joins to nothing
        cand = broadcast_forms(dep)
        j = keys.merge(cand, left_on=["ADEP_mvt", "join_key"],
                       right_on=["airport", "join_key"], how="inner")
    else:
        j = keys.merge(dep, left_on=["ADEP_mvt", "callsign"],
                       right_on=["airport", "callsign_modal"], how="inner")
        j["matched_on_modal"] = 1.0
    j["dt_takeoff"] = j.obs_wheels_off - j.mvt_s
    j = j[j.dt_takeoff.abs() <= WINDOW].copy()
    # a tie between two segments resolves to the nearer take-off, and a tie between two
    # FORMS of one segment to the modal, which is what the modal-only join would choose
    j["absdt"] = j.dt_takeoff.abs()
    n_cand = j.groupby("MVT_ID_mvt").size().rename("obs_n_candidates")
    j = j.merge(n_cand, on="MVT_ID_mvt", how="left")
    n_multi_seg = int((n_cand > 1).sum())
    # REBUILD 2026-09-09, round 3 finding 2: each dedup re-sorted on `absdt` ALONE with
    # pandas' default (quicksort, unstable), so a tie resolved by the frame's row order --
    # the same matcher on the same summaries gave a different clock depending on whether it
    # was passed the whole file or one month.  The order is now TOTAL and the sort STABLE:
    # nearer take-off, then the modal form, then the row's own id, which no frame ordering
    # can permute.
    j = j.sort_values(["absdt", "matched_on_modal", "seg_key"],
                      ascending=[True, False, True], kind="mergesort")
    j = j.drop_duplicates("MVT_ID_mvt")
    n_multi_dep = int((j.groupby("seg_key").size() > 1).sum())
    j = j.sort_values(["absdt", "matched_on_modal", "MVT_ID_mvt"],
                      ascending=[True, False, True], kind="mergesort")
    j = j.drop_duplicates("seg_key")
    j["match_mode"] = float(MODES["strict"])

    # ---- the wide second pass -------------------------------------------------
    # It touches ONLY departures the strict rule left unmatched and ONLY segments the
    # strict rule did not claim, so every strict row is bit-identical with or without it.
    if wide:
        left = keys[~keys.MVT_ID_mvt.isin(j.MVT_ID_mvt)]
        # EVERY segment carrying an off-block instant, not only the ones that end in a
        # take-off.  At EGLL and EDDF the network loses the aircraft before the runway,
        # so its ground run has no airborne successor and is `other`, not `departure` --
        # but the pushback it saw at the stand is the observation this stage wants.
        # An ARRIVAL is excluded: its ground run begins where it left the runway, so its
        # `obs_offblock` is a landing roll-out and not a pushback (Sol round 1,
        # blocking 8).  `departure` and `other` are kept -- `other` is exactly the class
        # the strict rule cannot see at EGLL and EDDF.
        free = seg[seg.obs_offblock.notna() & ~seg.seg_key.isin(j.seg_key)
                   & ~seg.kind.eq("arrival")]
        if FORMS:
            free = broadcast_forms(free)
            left = left.copy()
            left["join_key"] = norm_callsign(left.callsign)
            left = left[left.join_key.ne("")]
            w = left.merge(free, left_on=["ADEP_mvt", "join_key"],
                           right_on=["airport", "join_key"], how="inner")
        else:
            w = left.merge(free, left_on=["ADEP_mvt", "callsign"],
                           right_on=["airport", "callsign_modal"], how="inner")
            w["matched_on_modal"] = 1.0
        n_wide_cand_raw = int(len(w))
        # an off-block that precedes the take-off by a plausible taxi time
        w["implied_taxi"] = w.mvt_s - w.obs_offblock
        w = w[w.implied_taxi.between(WIDE_MIN_TAXI, WIDE_MAX_TAXI)].copy()
        # ...and a ground run that ENDS near the take-off.  Without this a callsign flown
        # twice in a day, or another aircraft carrying it, can lend an earlier movement's
        # pushback to a later flight: the implied-taxi bound alone admits anything inside
        # the previous two hours (Sol round 1, blocking 8).
        n_wide_after_taxi = int(len(w))
        w = w[(w.seg_end - w.mvt_s).abs() <= WIDE_SEG_END_WINDOW].copy()
        n_wide_after_segend = int(len(w))
        w["dt_takeoff"] = w.obs_wheels_off - w.mvt_s
        w["absdt"] = w.dt_takeoff.abs()
        w["in_wide_window"] = w.absdt <= WIDE_WINDOW
        w["match_mode"] = np.where(w.in_wide_window, float(MODES["wide_window"]),
                                   float(MODES["no_takeoff"]))
        n_wide_cand = w.groupby("MVT_ID_mvt").size().rename("obs_n_candidates")
        w = w.drop(columns=["obs_n_candidates"], errors="ignore").merge(
            n_wide_cand, on="MVT_ID_mvt", how="left")
        # a take-off inside the wider window wins; failing that, the LATEST off-block
        # before the take-off, which is the aircraft's own pushback and not an earlier
        # movement of the day
        w = w.sort_values(["in_wide_window", "absdt", "implied_taxi", "seg_key"],
                          ascending=[False, True, True, True], kind="mergesort")
        w = w.drop_duplicates("MVT_ID_mvt")
        w = w.sort_values(["in_wide_window", "absdt", "implied_taxi", "MVT_ID_mvt"],
                          ascending=[False, True, True, True], kind="mergesort")
        w = w.drop_duplicates("seg_key")
        n_wide = int(len(w))
        if len(w):
            j = pd.concat([j, w.drop(columns=["implied_taxi", "in_wide_window"])],
                          ignore_index=True)
        wide_funnel = {"candidate_pairs": n_wide_cand_raw,
                       "after_implied_taxi": n_wide_after_taxi,
                       "after_segment_end_window": n_wide_after_segend,
                       "after_one_to_one": n_wide}
    else:
        n_wide = 0
        wide_funnel = {}

    # ---- the turnaround third pass ----------------------------------------------------
    # `analysis/chain-spec.md`.  It touches ONLY departures both earlier passes left and
    # ONLY segments neither claimed, and it is further restricted to segments whose
    # broadcast identity is a ranking ARRIVAL and NOT a ranking departure -- so it cannot
    # reach a segment the callsign join could have taken, and every strict and wide row is
    # bit-identical with the pass on or off.
    #
    # The key is the AIRFRAME, not the clock.  The refuted pass of `analysis/reach-spec.md`
    # placed on "exactly one unclaimed departure within +-60 s", which is a property of the
    # NEIGHBOURHOOD and not of the pair: at EDDF july that gate fires 31 % of the time by
    # arithmetic alone, and a placebo that moved the departures a day placed MORE than it
    # did.  Here the segment's inbound callsign names a ranking ARRIVAL, and the departure
    # is the one leaving THAT ARRIVAL'S STAND, by the same hashed operator and aircraft
    # type, after its in-block.  The +-60 s take-off agreement is the last clause of the
    # key.  Placebos on the same conjunction place 1.03 % (day) and 1.10 % (stand shuffle).
    ak = arrival_keys(arrivals) if chain else {}
    chain_report = {"ran": False, "why": ("no arrivals supplied" if chain
                                          else "chain=False"), "placed": 0}
    if chain and ak:
        dk = {}
        for ap, g in coh.groupby("ADEP_mvt"):
            k = norm_callsign(g.callsign)
            dk[str(ap)] = set(k[k.ne("")])
        _j12 = j.copy()                  # PROBE: passes 1 and 2 only
        PROBE["j12"] = _j12
        free3 = seg[seg.kind.eq("departure") & seg.obs_offblock.notna()
                    & seg.obs_wheels_off.notna() & ~seg.seg_key.isin(j.seg_key)]
        n_free3 = int(len(free3))
        n_elig = n_leg = n_pairs = n_one = n_gated = 0
        n_ungated_admitted = 0
        _m4 = []
        w3 = None
        if n_free3:
            c3 = broadcast_forms(free3)
            c3 = c3.assign(
                _dep=[k in dk.get(str(a), ()) for a, k in zip(c3.airport, c3.join_key)],
                _arr=[k in ak.get(str(a), ()) for a, k in zip(c3.airport, c3.join_key)])
            per = c3.groupby("seg_key")[["_dep", "_arr"]].any()
            elig = per.index[(~per._dep) & per._arr]
            n_elig = int(len(elig))
        if n_elig:
            # ---- the inbound leg: which ARRIVAL this segment is still broadcasting ------
            ex = c3.loc[c3._arr & c3.seg_key.isin(elig), ["seg_key", "join_key"]]
            sg = free3.loc[free3.seg_key.isin(elig),
                           ["seg_key", "airport", "obs_offblock", "obs_wheels_off"]]
            ex = ex.merge(sg, on="seg_key", how="inner")
            av = arrival_legs(arrivals)
            lg = ex.merge(av, left_on=["airport", "join_key"],
                          right_on=["ADES_mvt", "key"], how="inner")
            # the LATEST arrival of that callsign whose in-block precedes the ground run's
            # own off-block: the leg that actually brought this airframe to this stand
            lg = lg[lg.in_block <= lg.obs_offblock]
            lg = lg.sort_values("in_block", kind="mergesort")
            lg = lg.drop_duplicates("seg_key", keep="last")
            leg = lg[["seg_key", "airport", "STAND_mvt", "AIRCRAFT_OPERATOR_flt",
                      "AIRCRAFT_TYPE_mvt", "in_block", "obs_wheels_off"]].dropna()
            n_leg = int(len(leg))
        if n_leg:
            # ---- the departure that stand chain implies ---------------------------------
            # the unclaimed departures, from the COHORT: a departure neither earlier pass
            # matched.  `STAND_mvt` is on 100 % of both phases, so the chain is available
            # for every one of them.
            # CLAIMED means a pass MATCHED it, not "it carries a clock": a strict match
            # onto a segment with a take-off but no off-block is claimed and clockless.
            # `code/turnaround.py` first used the clock test, let those departures compete,
            # and two-way uniqueness then declined 78 real placements -- see
            # `analysis/chain-reconcile.json` and `analysis/chain-spec.md` section 9.1.
            dfree = coh.loc[~coh.MVT_ID_mvt.isin(set(j.MVT_ID_mvt)),
                            ["MVT_ID_mvt", "ADEP_mvt", "callsign", "mvt_s", "STAND_mvt",
                             "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt"]].dropna(
                                subset=["STAND_mvt", "AIRCRAFT_OPERATOR_flt",
                                        "AIRCRAFT_TYPE_mvt", "mvt_s"])
            m3 = leg.merge(dfree, left_on=["airport", "STAND_mvt",
                                           "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt"],
                           right_on=["ADEP_mvt", "STAND_mvt",
                                     "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt"],
                           how="inner")
            # the departure follows the arrival it turns round from, inside a day
            m3 = m3[(m3.mvt_s > m3.in_block)
                    & ((m3.mvt_s - m3.in_block) <= MAX_TURNAROUND_S)]
            if len(m3):
                # uniqueness INSIDE the chained set, and one-to-one the other way: a stand
                # that two segments each choose identifies neither
                sz = m3.groupby("seg_key").MVT_ID_mvt.transform("size")
                m3 = m3[sz == 1]
                m3 = m3[~m3.MVT_ID_mvt.duplicated(keep=False)]
                n_pairs = n_one = int(len(m3))
            if n_one:
                _m3_all = m3.copy()          # PROBE: the pairs before the clause
                m3 = m3[(m3.obs_wheels_off - m3.mvt_s).abs() <= CHECK_S]
                n_gated = int(len(m3))
                # PROBE: every unique pair the +-60 s clause refuses.  They are admitted
                # as match_mode 4 with the three check values beside them, and the
                # CONSUMER decides which of them count -- one build serves every
                # threshold, and no threshold is baked into the block.
                _m4 = _m3_all[~_m3_all.seg_key.isin(set(m3.seg_key))].copy()
            if n_gated:
                s3 = free3[free3.seg_key.isin(set(m3.seg_key))]
                w3 = (m3[["seg_key", "MVT_ID_mvt"]]
                      .merge(coh.loc[:, ["MVT_ID_mvt", "ADEP_mvt", "callsign", "mvt_s"]],
                             on="MVT_ID_mvt", how="inner", validate="1:1")
                      .merge(s3, on="seg_key", how="inner", validate="1:1"))
                w3["form"] = w3.seg_key.map(
                    lg.drop_duplicates("seg_key").set_index("seg_key").join_key)
                w3["join_key"] = w3["form"]
                w3["matched_on_modal"] = 0.0     # no callsign matched, by construction
                w3["dt_takeoff"] = w3.obs_wheels_off - w3.mvt_s
                w3["absdt"] = w3.dt_takeoff.abs()
                w3["obs_n_candidates"] = 1.0     # uniqueness IS part of the key
                w3["match_mode"] = float(MODES["turnaround"])
                missing = [c for c in j.columns if c not in w3.columns]
                if missing:                      # never silently NaN a column `j` needs
                    raise SystemExit(f"turnaround pass: {missing} absent from its rows")
                j = pd.concat([j, w3[j.columns]], ignore_index=True)
            # ---- PROBE: the ungated pairs as match_mode 4 --------------------------
            if len(_m4):
                s4 = free3[free3.seg_key.isin(set(_m4.seg_key))]
                w4 = (_m4[["seg_key", "MVT_ID_mvt"]]
                      .merge(coh.loc[:, ["MVT_ID_mvt", "ADEP_mvt", "callsign", "mvt_s"]],
                             on="MVT_ID_mvt", how="inner", validate="1:1")
                      .merge(s4, on="seg_key", how="inner", validate="1:1"))
                w4["form"] = w4.seg_key.map(
                    lg.drop_duplicates("seg_key").set_index("seg_key").join_key)
                w4["join_key"] = w4["form"]
                w4["matched_on_modal"] = 0.0
                w4["dt_takeoff"] = w4.obs_wheels_off - w4.mvt_s
                w4["absdt"] = w4.dt_takeoff.abs()
                w4["obs_n_candidates"] = 1.0
                w4["match_mode"] = 4.0
                # (0) how tightly the segment pins its own take-off
                _chk = pd.DataFrame({"MVT_ID_mvt": w4.MVT_ID_mvt.to_numpy()})
                _chk["chain_gap_s"] = (w4.t_airborne - w4.obs_wheels_off).to_numpy()
                # (i) the segment's off-block against the departure's AOBT
                _ao = coh.set_index("MVT_ID_mvt").aobt_s
                _chk["chain_abs_dt_offblock"] = np.abs(
                    w4.obs_offblock.to_numpy()
                    - w4.MVT_ID_mvt.map(_ao).to_numpy("float64"))
                # (ii) the runway the wheels-off position implies, learned from the
                # PASS-1/2 matched pairs of this same month: no database, nothing fitted
                # to the chain rows, and the map cannot see the pairs it judges.
                _mm = _j12[["seg_key", "MVT_ID_mvt"]].merge(
                    seg[["seg_key", "airport", "wheels_lat", "wheels_lon"]],
                    on="seg_key", how="left").merge(
                    coh[["MVT_ID_mvt", "RUNWAY_mvt"]], on="MVT_ID_mvt", how="left")
                _mm = _mm.dropna(subset=["airport", "RUNWAY_mvt",
                                         "wheels_lat", "wheels_lon"])
                _cent = {}
                for (_a, _r), _q in _mm.groupby(["airport", "RUNWAY_mvt"], observed=True):
                    if len(_q) >= 20:
                        _cent.setdefault(str(_a), {})[str(_r)] = (
                            float(_q.wheels_lat.median()), float(_q.wheels_lon.median()))
                _rw = coh.set_index("MVT_ID_mvt").RUNWAY_mvt
                _ag = []
                for _a, _la, _lo, _mid in zip(w4.airport.astype(str), w4.wheels_lat,
                                              w4.wheels_lon, w4.MVT_ID_mvt):
                    _c = _cent.get(_a)
                    _want = _rw.get(_mid)
                    if not _c or not np.isfinite(_la) or not np.isfinite(_lo) \
                            or _want is None or pd.isna(_want):
                        _ag.append(np.nan); continue
                    _n = list(_c)
                    _d = [(_la - _c[k][0]) ** 2 + (_lo - _c[k][1]) ** 2 for k in _n]
                    _ag.append(float(_n[int(np.argmin(_d))] == str(_want)))
                _chk["chain_runway_agrees"] = _ag
                PROBE["checks"] = _chk
                missing = [c for c in j.columns if c not in w4.columns]
                if missing:
                    raise SystemExit(f"ungated pass: {missing} absent from its rows")
                j = pd.concat([j, w4[j.columns]], ignore_index=True)
                n_ungated_admitted = int(len(w4))
        chain_report = {
            "ran": True, "why": None,
            "segments_left_by_both_passes": n_free3,
            "of_them_an_arrival_identity_and_not_a_departure_one": n_elig,
            "with_an_inbound_leg_before_their_own_off_block": n_leg,
            "after_uniqueness_both_ways": n_one,
            "after_the_take_off_agreement": n_gated,
            "placed": n_gated,
            "ungated_admitted_as_mode_4": n_ungated_admitted,
            "max_turnaround_s": MAX_TURNAROUND_S, "check_s": CHECK_S,
            "declined_no_leg": n_elig - n_leg,
            "declined_not_unique": max(n_leg - n_one, 0),
            "declined_no_take_off_agreement": n_one - n_gated,
            "by_airport": ({} if w3 is None else
                           {str(a): int(n) for a, n in
                            w3.airport.value_counts().sort_index().items()}),
        }

    cen = _centres()
    lat0 = j.airport.map(lambda a: cen.get(a, (np.nan, np.nan))[0])
    lon0 = j.airport.map(lambda a: cen.get(a, (np.nan, np.nan))[1])
    dx = (j.stand_lon - lon0) * 111.320 * np.cos(np.radians(lat0))
    dy = (j.stand_lat - lat0) * 110.540
    dist = np.hypot(dx, dy)

    f = pd.DataFrame({"MVT_ID_mvt": j.MVT_ID_mvt.to_numpy()})
    # identity columns, not features: which segment this row was matched to, so a
    # prediction can be traced back to the observation that moved it
    f["obs_icao"] = j.icao.to_numpy()
    f["obs_wheels_off"] = j.obs_wheels_off.to_numpy()
    f["obs_taxi"] = (j.mvt_s - j.obs_offblock).to_numpy()
    f["obs_taxi_motion"] = (j.mvt_s - j.first_motion).to_numpy()
    f["obs_seg_taxi"] = (j.obs_wheels_off - j.obs_offblock).to_numpy()
    f["dt_takeoff"] = j.dt_takeoff.to_numpy()
    f["time_to_250m"] = (j.obs_offblock_disp - j.obs_offblock_loose).to_numpy()
    f["seen_before_move_s"] = (j.obs_offblock - j.seg_start).to_numpy()
    for c in ("stand_stop_s", "seg_points", "n_ground_points", "seg_dt_median_s",
              "seg_gap_max_s", "seg_span_s", "callsign_n", "stand_lat", "stand_lon",
              "arr_taxi_in_s", "arr_stand_occupancy_s", "arr_gap_reception_s",
              "obs_n_candidates"):
        f[c] = j[c].to_numpy()
    f["turnaround_anchored"] = j.turnaround_anchored.astype("float64").to_numpy()
    f["arr_present"] = j.arr_present.astype("float64").to_numpy()
    f["stand_dist_km"] = dist.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        ang = np.arctan2(dy.to_numpy(), dx.to_numpy())
    f["stand_sin"], f["stand_cos"] = np.sin(ang), np.cos(ang)
    f["obs_ambiguous"] = (j.obs_n_candidates > 1).astype("float64").to_numpy()
    f["match_mode"] = j.match_mode.to_numpy("float64")
    f["clock_ok"] = (j.dt_takeoff.abs() <= 60).astype("float64").to_numpy()
    f["has_wheels_off"] = j.obs_wheels_off.notna().astype("float64").to_numpy()
    f["matched_on_modal"] = j.matched_on_modal.astype("float64").to_numpy()

    out = coh[["MVT_ID_mvt"]].merge(f, on="MVT_ID_mvt", how="left", validate="one_to_one")
    # `obs_matched` is "a ground segment was identified", which is the pilot's
    # `matched`.  `obs_has_clock` is the narrower "and it carries a 250 m
    # stand-departure onset", which is the pilot's `with_obs_offblock_disp` and the
    # only condition under which a row can be moved.  A wide match may have no
    # take-off point at all, so `obs_matched` cannot be read off `dt_takeoff`.
    out["obs_matched"] = out.match_mode.notna()
    out["obs_has_clock"] = out.obs_taxi.notna()
    _mode = {v: k for k, v in MODES.items()}
    out["reach_class"] = np.where(
        ~out.obs_matched, "unmatched",
        np.where(~out.obs_has_clock, "matched_no_clock",
                 np.where(out.obs_ambiguous > 0,
                          "ambiguous_" + out.match_mode.map(_mode).fillna("strict"),
                          out.match_mode.map(_mode).fillna("strict"))))
    out = out[["MVT_ID_mvt", "obs_icao", "obs_wheels_off", *FEATURES,
               "obs_matched", "obs_has_clock", "reach_class"]]
    # PROBE: the mode-4 rows' three check values ride out with the block, so the
    # consumer can apply any threshold without another build.
    _c = PROBE.get("checks")
    if _c is not None:
        out = out.merge(_c, on="MVT_ID_mvt", how="left", validate="one_to_one")
    else:
        for _n in ("chain_gap_s", "chain_abs_dt_offblock", "chain_runway_agrees"):
            out[_n] = np.nan
    out.attrs |= {"n_multi_seg": n_multi_seg, "n_multi_dep": n_multi_dep,
                  "n_segments": int(len(dep)), "days_present": n_days_present,
                  "days_present_list": seg_attrs.get("days_present", []),
                  "days_missing": seg_attrs.get("days_missing", []),
                  "days_wrong_rule": seg_attrs.get("days_wrong_rule", []),
                  "days_held": seg_attrs.get("days_held", []),
                  "days_held_reason": seg_attrs.get("days_held_reason"),
                  "rules_seen": seg_attrs.get("rules_seen", []),
                  "normalised_by_airborne_alt_ok":
                      seg_attrs.get("normalised_by_airborne_alt_ok"),
                  "n_wide_matches": n_wide, "wide_funnel": wide_funnel,
                  "turnaround": chain_report,
                  "callsign_source": coh.callsign_source.value_counts().to_dict()}
    return out


def census(block: pd.DataFrame, coh: pd.DataFrame) -> dict:
    """The reach numbers a report quotes, with no label in sight."""
    m = coh[["MVT_ID_mvt", "ADEP_mvt", "has_nm"]].merge(block, on="MVT_ID_mvt", how="left")
    matched = m.obs_matched.fillna(False)
    def share(s):
        return round(float(s.mean()), 4) if len(s) else None
    return {
        "departures": int(len(m)),
        "matched": int(matched.sum()),
        "matched_share": share(matched),
        "matched_nm": share(matched[m.has_nm]),
        "matched_no_nm": share(matched[~m.has_nm]),
        "no_nm_rows": int((~m.has_nm).sum()),
        # the class names carry the mode ("ambiguous_strict", "ambiguous_wide_window"),
        # so an equality test counts none of them (Sol round 1, non-blocking)
        "ambiguous": int(m.reach_class.astype(str).str.startswith("ambiguous").sum()),
        "by_airport": {a: {"departures": int(len(g)), "matched": int(g.obs_matched.fillna(False).sum()),
                           "rate": share(g.obs_matched.fillna(False))}
                       for a, g in m.groupby("ADEP_mvt")},
        "clock_check_wheels_off_minus_mvt_s": {
            "median": float(m.dt_takeoff.median()),
            "share_within_60s": float((m.dt_takeoff.abs() <= 60).mean(skipna=True)),
            "p01": float(m.dt_takeoff.quantile(.01)),
            "p99": float(m.dt_takeoff.quantile(.99))},
        "n_multi_seg": block.attrs.get("n_multi_seg"),
        "n_multi_dep": block.attrs.get("n_multi_dep"),
        "segments_observed": block.attrs.get("n_segments"),
    }


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.path.insert(0, str(EXP / "code"))
        import test_block                                    # noqa: PLC0415
        test_block.main()
