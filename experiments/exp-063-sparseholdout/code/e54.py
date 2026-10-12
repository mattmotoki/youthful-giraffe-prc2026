"""exp-054-arrivalrepr: shared inputs, checked by hash, and the registered constants (SPEC section 1).

Everything that fixes the instrument lives here, so that the audit, the representation job, the
departure fit job and the pricing job cannot disagree on a fold, a pin, the population, the day
blocks, the auxiliary outcome, a key, the support rule, the arrival factor model, the departure
learner, the one seed or a statistic.  exp-050's code is imported READ-ONLY after a sha check.

The auxiliary outcome is the arrival taxi-in time.  Its target clock is the arrival in-block time
`B_arr = BLOCK_TIME_UTC_mvt` on ARR rows; `TAXITIME_SEC_mvt` on ARR rows is `B_arr - T_arr` exactly
(`T_arr = MVT_TIME_UTC_mvt`, the landing).  No model input of this experiment reads `B_arr`, the
arrival `TAXITIME_SEC_mvt`, or any quantity computed from either (SPEC 1.3).  They are read in
one place only, `arrival_outcomes()`, which returns the auxiliary TARGET of the fitting months.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
E50 = ROOT / "experiments/exp-050-nmblind"
E46 = ROOT / "experiments/exp-046-surfacefit"
for p in (HERE, ROOT, E50 / "code", E46 / "code"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_arr(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype="float64")).tobytes())
    return h.hexdigest()


def sha_frame(df: pd.DataFrame) -> str:
    """sha256 over the columns in order: numeric as float64 bytes, other columns as UTF-8 text."""
    h = hashlib.sha256()
    for c in df.columns:
        h.update(str(c).encode())
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            h.update(np.ascontiguousarray(s.to_numpy("float64")).tobytes())
        else:
            h.update("\x1f".join(s.astype(str).tolist()).encode())
    return h.hexdigest()


# ---- pins (assignment "Leakage boundary"; exp-050 files as exp-052 pinned them) -----------
PANEL_PINS = {
    "july_loo": ("fc8700e9910f03e7f353624cc2d22bf252b81105032e905e6339bb5930b3630c",
                 "5bc2dbca6c1615745ff6a24daa9a3182215265f86494ba8c8c4ed233fcdd5d1e"),
    "december": ("8b3d855ca727f88e441063bf18eaa93213e5d081fda1bc7606c4ba06889e4846",
                 "50e0f61c093ada152d74172a36ff49d1a4702ef850d63415a95de51d9b99eab8"),
    "january": ("4672b7b3784b57f3903b5e58a6dd84ffa8ce67c86989874974d6fc59c8be3d62",
                "53ada01f6ecbe44e5e1282bd1f797f3a8de859eddd4a5aeb34e28448aafb26ad"),
}
PINNED = {
    E50 / "code/nb_router.py": "73a4486bd23f2ca7949e4c476e9abc461f2dd48711ed1b38ef2268e96a2c9be8",
    E50 / "code/nb_panel.py": "4f8da0a13bd2d94fe83b814d0f2029007a3d3a53900673135fc9656e40c14b03",
    E50 / "analysis/panel.json": "c091246f2fadac4feff277e5f76a7f1f71940cf38a27063687be1fd8a86e2f0f",
    E50 / "analysis/router-july_loo-december-january.json":
        "571f0510578ef104f5be067d42ce0363b309676ebfeaf8ad4a4a73d86ee89454",
    E46 / "code/nooverwrite.py": "4e480e6952121992343247f4804696697eecf8d4991af74695b4a333f4885d55",
}


def check_pinned() -> dict:
    out = {}
    for p, want in PINNED.items():
        got = sha_file(p)
        if got != want:
            raise SystemExit(f"{p.relative_to(ROOT)} is {got[:16]}, not the pinned {want[:16]}")
        out[str(p.relative_to(ROOT))] = got
    return out


check_pinned()
import nb_router as NR                                                # noqa: E402
import nooverwrite as NO                                              # noqa: E402
import prc.data as D                                                  # noqa: E402

FOLDS = ["july_loo", "december", "january"]
SHARES = dict(NR.SHARES)                # july_loo 0.5571321275602379, january 0.4428678724397621
K = 5                                   # five consecutive calendar-day blocks per fold month
SEED = 54                               # the one fixed seed (SPEC 1.7)
THREADS = 3
LOAD_LIMIT = 10.0
BOOT = 2000
MIN_PRED = 60.0


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def hold_load(what: str) -> dict:
    """The assignment: before every job over a few minutes, hold while the 1-minute load is
    above 10.  Called before every fit launch."""
    while True:
        l1 = os.getloadavg()[0]
        if l1 <= LOAD_LIMIT:
            break
        print(f"  {utc()} load {l1:.2f} > {LOAD_LIMIT:g}: holding {what}", flush=True)
        time.sleep(30)
    return {"launch_utc": utc(), "launch_load_1m": round(float(l1), 2)}


def fold_month(fold: str) -> str:
    return D.FOLDS[fold].valid_month


def fitting_months(fold: str) -> list[str]:
    """The outer fitting months: every 2025 month except the held one."""
    held = fold_month(fold)
    out = [m for m in D.MONTHS if m != held]
    assert held not in out and len(out) == 11
    return out


# ---- the frozen v22 panels (label-free reads by column) -----------------------------------
#: Read before the first label read.  `clean` is NOT here: it is set from label artefacts.
PANEL_LABEL_FREE = ["MVT_ID_mvt", "airport", "day", "regime", "guard_held", "p_v22"]


def panel_path(fold: str) -> Path:
    return E50 / f"work/panel-v22-{fold}.parquet"


def panel(fold: str, columns: list[str]) -> pd.DataFrame:
    """The frozen exp-050 v22 panel of one fold, only `columns` read.  Refuses to return
    anything unless the full-file sha AND the ordered id-then-p_v22 sha equal the pins."""
    p = panel_path(fold)
    want_file, want_vec = PANEL_PINS[fold]
    got = sha_file(p)
    if got != want_file:
        raise SystemExit(f"{p.name} is {got[:16]}, not the pinned {want_file[:16]}: refusing to start")
    iv = pd.read_parquet(p, columns=["MVT_ID_mvt", "p_v22"])
    gv = sha_arr(iv.MVT_ID_mvt, iv.p_v22)
    if gv != want_vec:
        raise SystemExit(f"{fold}: ordered id+p_v22 sha {gv[:16]} is not the pinned {want_vec[:16]}")
    out = pd.read_parquet(p, columns=columns)
    if "MVT_ID_mvt" in columns and not np.array_equal(out.MVT_ID_mvt.to_numpy("float64"),
                                                      iv.MVT_ID_mvt.to_numpy("float64")):
        raise SystemExit(f"{fold}: panel row order moved between reads")
    return out


def population_sha_q08() -> dict:
    rj = json.loads((E50 / "analysis/router-july_loo-december-january.json").read_text())
    return {f: rj["folds"][f]["population_ids_sha256"] for f in FOLDS}


def population(pan: pd.DataFrame, fold: str) -> np.ndarray:
    """P = nm_bulk and not guard-held (Q08's population), from label-free columns.  Its
    sorted-id sha must equal Q08's."""
    pop = (pan.regime.astype(str).to_numpy() == "nm_bulk") & ~pan.guard_held.to_numpy(bool)
    got = sha_arr(np.sort(pan.MVT_ID_mvt.to_numpy("float64")[pop]))
    want = population_sha_q08()[fold]
    if got != want:
        raise SystemExit(f"{fold}: the population sha {got[:16]} is not Q08's {want[:16]}")
    return pop


def blocks(day: np.ndarray) -> np.ndarray:
    """Five consecutive calendar-day blocks shared by every airport (exp-050 `dayblocks`)."""
    b = NR.dayblocks(np.asarray(day))
    assert set(np.unique(b)) == set(range(K))
    return b


# ---- the masked held month and the departure row features (SPEC 1.6) ---------------------
def _assert_masked(frame: pd.DataFrame) -> None:
    """A frame that carries the label columns must carry no departure value in them."""
    dep = frame["PHASE_mvt"].to_numpy() == "DEP"
    for c in ("BLOCK_TIME_UTC_mvt", D.TARGET):
        if c in frame.columns and frame.loc[dep, c].notna().any():
            raise SystemExit(f"{c} carries a departure value: the frame is not masked")


EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def ep(s: pd.Series) -> np.ndarray:
    return (s - EPOCH).dt.total_seconds().to_numpy("float64")


def _txt(s: pd.Series) -> np.ndarray:
    """Raw text key: missing and empty strings are missing (None); nothing else is changed."""
    v = s.astype("string")
    v = v.where(v.str.len() > 0)
    return np.array([None if pd.isna(x) else str(x) for x in v], dtype=object)


def _join(apt: np.ndarray, val: np.ndarray) -> np.ndarray:
    return np.array([None if (v is None or a is None) else f"{a}|{v}" for a, v in zip(apt, val)],
                    dtype=object)


#: Departure row-only inputs.  Every one is a field of the departure's OWN row in the masked
#: frame; no other row, no arrival field, no label, no p_v22 (SPEC 1.6).
ROW_CATS = ["apt", "apt_stand", "apt_rwy", "op", "wtc", "ac_type"]
ROW_NUM = ["hour", "dow", "to_aobt", "to_sched", "to_lobt", "to_eobt1", "aobt_lobt"]
ROW_FEATURES = ROW_CATS + ROW_NUM
#: The auxiliary representation: one arrival factor per shared key (SPEC 1.4).
KEYS = ["stand", "rwy", "op"]
REP_COLS = [f"f_{k}" for k in KEYS]
ALL_FEATURES = ROW_FEATURES + REP_COLS
#: Fields a departure row feature reads.  `row_features` cuts the frame to these first.
ROW_WHITELIST = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt",
                 "AIRCRAFT_OPERATOR_flt", "WK_TBL_CAT_flt", "AIRCRAFT_TYPE_mvt",
                 "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", "EOBT_1_flt"]
NM_FLT = ["LOBT_flt", "CALLSIGN_flt", "ADEP_flt", "ADES_flt", "ADES_FILED_flt",
          "MARKET_SEGMENT_flt", "IOBT_flt", "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt",
          "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt", "EOBT_1_flt",
          "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt"]


def departure_frame(month: str) -> pd.DataFrame:
    """The held month's movement rows with ONLY the columns a departure feature, key or no-NM
    flag reads.  Neither label column (`BLOCK_TIME_UTC_mvt`, `TAXITIME_SEC_mvt`) is in the read,
    so no departure label and no arrival in-block clock is ever loaded here: the mask is applied
    at the read, before feature building."""
    cols = list(dict.fromkeys(ROW_WHITELIST + NM_FLT))
    assert "BLOCK_TIME_UTC_mvt" not in cols and D.TARGET not in cols
    return D.load_month(month, columns=cols)


def departure_keys(dep: pd.DataFrame) -> pd.DataFrame:
    """The shared keys of departure rows: airport = ADEP_mvt."""
    apt = _txt(dep["ADEP_mvt"])
    return pd.DataFrame({
        "MVT_ID_mvt": dep["MVT_ID_mvt"].to_numpy("float64"),
        "apt": apt,
        "stand": _join(apt, _txt(dep["STAND_mvt"])),
        "rwy": _join(apt, _txt(dep["RUNWAY_mvt"])),
        "op": _join(apt, _txt(dep["AIRCRAFT_OPERATOR_flt"]))})


def row_features(masked: pd.DataFrame) -> pd.DataFrame:
    """Departure row-only features from the masked month, one row per departure, raw values
    (categoricals as text, codes are assigned later by `encode`)."""
    _assert_masked(masked)
    f = masked[ROW_WHITELIST]
    dep = f[f["PHASE_mvt"].to_numpy() == "DEP"].reset_index(drop=True)
    t = ep(dep["MVT_TIME_UTC_mvt"])
    k = departure_keys(dep)
    out = pd.DataFrame({
        "MVT_ID_mvt": dep["MVT_ID_mvt"].to_numpy("float64"),
        "apt": k["apt"], "apt_stand": k["stand"], "apt_rwy": k["rwy"],
        "op": _txt(dep["AIRCRAFT_OPERATOR_flt"]),
        "wtc": _txt(dep["WK_TBL_CAT_flt"]),
        "ac_type": _txt(dep["AIRCRAFT_TYPE_mvt"]),
        "hour": np.mod(t, 86400.0) / 3600.0,
        "dow": np.floor(t / 86400.0) % 7,
        "to_aobt": t - ep(dep["AOBT_3_flt"]),
        "to_sched": t - ep(dep["SCHED_TIME_UTC_mvt"]),
        "to_lobt": t - ep(dep["LOBT_flt"]),
        "to_eobt1": t - ep(dep["EOBT_1_flt"]),
        "aobt_lobt": ep(dep["AOBT_3_flt"]) - ep(dep["LOBT_flt"])})
    return out


def no_nm_mask(masked: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
    """COMMON.md's no-NM rows: departures whose every `_flt` column is null, in `ids` order."""
    dep = masked[masked["PHASE_mvt"].to_numpy() == "DEP"]
    m = dep[NM_FLT].isna().all(axis=1).to_numpy()
    s = pd.Series(m, index=dep["MVT_ID_mvt"].to_numpy("float64"))
    return s.reindex(ids).to_numpy(bool)


def encode(F: pd.DataFrame) -> pd.DataFrame:
    """Categorical text -> integer codes by sorted value within the month (missing -> -1 ->
    NaN for LightGBM).  Label-free and deterministic."""
    out = F.copy()
    for c in ROW_CATS:
        v = out[c].to_numpy(object)
        present = np.array([x is not None for x in v])
        codes = np.full(len(v), np.nan)
        if present.any():
            u = np.array(sorted(set(v[present].tolist())), dtype=object)
            lut = {x: i for i, x in enumerate(u)}
            codes[present] = [lut[x] for x in v[present]]
        out[c] = codes
    return out


# ---- the auxiliary outcome and the arrival keys (SPEC 1.3, 1.4) ---------------------------
Y_CLIP = (0.0, 3600.0)                  # auxiliary target clipped before any fit
ARR_KEY_COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt",
                "AIRCRAFT_OPERATOR_flt", "MVT_TIME_UTC_mvt"]
ARR_TARGET_COLS = ["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]


def arrival_keys(month: str) -> pd.DataFrame:
    """Arrival rows of one month: shared keys and the landing clock only (no in-block clock)."""
    raw = D.load_month(month, columns=ARR_KEY_COLS)
    a = raw[raw["PHASE_mvt"].to_numpy() == "ARR"].reset_index(drop=True)
    apt = _txt(a["ADES_mvt"])
    t = ep(a["MVT_TIME_UTC_mvt"])
    return pd.DataFrame({
        "MVT_ID_mvt": a["MVT_ID_mvt"].to_numpy("float64"), "month": month,
        "apt": apt,
        "stand": _join(apt, _txt(a["STAND_mvt"])),
        "rwy": _join(apt, _txt(a["RUNWAY_mvt"])),
        "op": _join(apt, _txt(a["AIRCRAFT_OPERATOR_flt"])),
        "t": t,
        "date": pd.to_datetime(t, unit="s", utc=True).strftime("%Y-%m-%d").to_numpy(object),
        "hour": np.floor(np.mod(t, 86400.0) / 3600.0).astype(int)})


def arrival_outcomes(month: str, ids: np.ndarray) -> np.ndarray:
    """THE auxiliary target of one month, in `ids` order: arrival taxi-in time `B_arr - T_arr`
    (asserted equal to TAXITIME_SEC_mvt on every ARR row, a missing value on both sides
    counting as equal), clipped to Y_CLIP; NaN where the arrival has no outcome.
    The only reader of the arrival target clock in this experiment."""
    raw = D.load_month(month, columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]
                       + ARR_TARGET_COLS)
    a = raw[raw["PHASE_mvt"].to_numpy() == "ARR"]
    alg = (a["BLOCK_TIME_UTC_mvt"] - a["MVT_TIME_UTC_mvt"]).dt.total_seconds().to_numpy("float64")
    tin = a["TAXITIME_SEC_mvt"].to_numpy("float64")
    same = (alg == tin) | (np.isnan(alg) & np.isnan(tin))
    if not same.all():
        raise SystemExit(f"{month}: TAXITIME_SEC_mvt != BLOCK - MVT on {int((~same).sum())} ARR rows")
    s = pd.Series(np.clip(tin, *Y_CLIP), index=a["MVT_ID_mvt"].to_numpy("float64"))
    if not s.index.is_unique:
        raise SystemExit(f"{month}: arrival ids are not unique")
    missing = ~np.isin(np.asarray(ids, dtype="float64"), s.index.to_numpy("float64"))
    if missing.any():
        raise SystemExit(f"{month}: {int(missing.sum())} requested ids are not arrivals of the month")
    return s.reindex(ids).to_numpy("float64")


# ---- the support rule (SPEC 1.5), registered before any fit -------------------------------
MIN_ROWS = 100                          # fitting-month arrivals per (airport, key value)
MIN_DAYS = 20                           # distinct fitting-month UTC dates per (airport, key value)


def support_table(A: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per key value: fitting-month arrival rows, distinct dates, supported flag."""
    k = A[key].to_numpy(object)
    m = np.array([x is not None for x in k])
    d = pd.DataFrame({"k": k[m], "date": A["date"].to_numpy(object)[m]})
    g = d.groupby("k")
    t = pd.DataFrame({"rows": g.size(), "days": g["date"].nunique()})
    t["supported"] = (t["rows"] >= MIN_ROWS) & (t["days"] >= MIN_DAYS)
    return t


# ---- the arrival factor model (SPEC 1.4) ----------------------------------------------------
BLOCKS_ARR = ["date", "hour", "stand", "rwy", "op"]
LAMBDA = {"date": 5.0, "hour": 5.0, "stand": 50.0, "rwy": 50.0, "op": 50.0}
SWEEPS = 60


def arrival_design(A: pd.DataFrame) -> dict:
    """Integer codes per block (-1 = no level: the row takes no effect in that block)."""
    apt = A["apt"].to_numpy(object)
    raw = {"date": _join(apt, A["date"].to_numpy(object)),
           "hour": _join(apt, A["hour"].astype(str).to_numpy(object)),
           "stand": A["stand"].to_numpy(object), "rwy": A["rwy"].to_numpy(object),
           "op": A["op"].to_numpy(object)}
    out = {}
    for b in BLOCKS_ARR:
        v = raw[b]
        present = np.array([x is not None for x in v])
        u = np.array(sorted(set(v[present].tolist())), dtype=object)
        lut = {x: i for i, x in enumerate(u)}
        codes = np.full(len(v), -1, dtype=np.int64)
        codes[present] = [lut[x] for x in v[present]]
        out[b] = (codes, u)
    return out


def fit_factors(y: np.ndarray, design: dict) -> dict:
    """Block ridge by backfitting: y = mu + sum_b e_b[code_b] + noise, each block's effect the
    shrunk mean of its partial residual, sum / (n + lambda_b).  SWEEPS fixed sweeps, blocks in
    BLOCKS_ARR order.  Deterministic; one trainable fit."""
    # a constant target takes its value exactly: np.mean of a constant array can miss it by an
    # ulp, which would leave 1e-15 s effects where the constant control must give exact zeros
    mu = float(y[0]) if np.all(y == y[0]) else float(np.mean(y))
    fitted = {b: np.zeros(len(y)) for b in BLOCKS_ARR}
    eff = {}
    trace = []
    for s in range(SWEEPS):
        delta = 0.0
        for b in BLOCKS_ARR:
            codes, u = design[b]
            m = codes >= 0
            part = y - mu - sum(fitted[c] for c in BLOCKS_ARR if c != b)
            num = np.bincount(codes[m], weights=part[m], minlength=len(u))
            cnt = np.bincount(codes[m], minlength=len(u)).astype("float64")
            e = num / (cnt + LAMBDA[b])
            new = np.zeros(len(y))
            new[m] = e[codes[m]]
            delta = max(delta, float(np.max(np.abs(new - fitted[b]))) if len(y) else 0.0)
            fitted[b] = new
            eff[b] = e
        trace.append(delta)
    resid = y - mu - sum(fitted.values())
    return {"mu": mu, "effects": {b: (design[b][1], eff[b]) for b in BLOCKS_ARR},
            "max_change_last_sweep_s": trace[-1], "trace": trace,
            "rmse_in_sample": float(np.sqrt(np.mean(resid ** 2)))}


# ---- the controls: auxiliary targets (SPEC 1.8) ---------------------------------------------
ARMS_REP = ["true", "const", "gmean", "donor", "shuffle"]
ARMS = ARMS_REP + ["rowonly"]


def cell_key(A: pd.DataFrame) -> np.ndarray:
    return _join(_join(A["apt"].to_numpy(object), A["date"].to_numpy(object)),
                 A["hour"].astype(str).to_numpy(object))


def within_perm(key: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A permutation that moves each row only inside its own key group (exp-052 airday_perm)."""
    codes, _ = pd.factorize(pd.Series(key), sort=True)
    base = np.argsort(codes, kind="stable")
    shuf = np.lexsort((rng.random(len(codes)), codes))
    perm = np.empty(len(codes), dtype=np.int64)
    perm[base] = shuf
    return perm


DONOR_SCAN = 200                        # neighbours scanned on each side in time order
DONOR_DIFFER = ("stand", "op")          # keys a donor must NOT share; the runway may coincide


def donor_index(A: pd.DataFrame, differ: tuple = DONOR_DIFFER) -> tuple[np.ndarray, dict]:
    """Per arrival: the time-nearest OTHER arrival at the same airport on the same UTC date whose
    stand and operator both differ from its own (a missing value never matches).  The runway is
    not restricted: it is part of the minute's airport state, and requiring it to differ breaks
    the time match wherever one arrival runway is in use (label-free audit, SPEC 1.8).  Ties go
    to the earlier landing.  -1 when none exists within DONOR_SCAN neighbours on either side.
    `differ` other than DONOR_DIFFER is used only by the audit, to record the rejected rule."""
    n = len(A)
    grp = _join(A["apt"].to_numpy(object), A["date"].to_numpy(object))
    gcode, _ = pd.factorize(pd.Series(grp), sort=True)
    t = A["t"].to_numpy("float64")
    order = np.lexsort((t, gcode))
    g_o, t_o = gcode[order], t[order]
    keys_o = {}
    for k in ("stand", "rwy", "op"):
        c, _ = pd.factorize(pd.Series(A[k].to_numpy(object)), use_na_sentinel=True)
        keys_o[k] = c[order]             # -1 for missing: never equal to a present code
    best = np.full(n, -1, dtype=np.int64)
    best_dt = np.full(n, np.inf)
    pos = np.arange(n)
    for off in range(1, DONOR_SCAN + 1):
        for sgn in (-1, 1):             # earlier first, so a tie keeps the earlier landing
            j = pos + sgn * off
            ok = (j >= 0) & (j < n)
            jj = np.where(ok, j, 0)
            ok &= g_o[jj] == g_o
            for k in differ:
                a, b = keys_o[k], keys_o[k][jj]
                ok &= ~((a == b) & (a >= 0))
            dt = np.abs(t_o[jj] - t_o)
            take = ok & (dt < best_dt)
            best[take] = jj[take]
            best_dt[take] = dt[take]
    out = np.full(n, -1, dtype=np.int64)
    out[order] = np.where(best >= 0, order[np.maximum(best, 0)], -1)
    gap = np.full(n, np.nan)
    gap[order] = np.where(best >= 0, best_dt, np.nan)
    f = gap[np.isfinite(gap)]
    has = out >= 0
    rw = A["rwy"].to_numpy(object)
    same_rwy = np.zeros(n, bool)
    same_rwy[has] = [(rw[i] is not None and rw[i] == rw[j]) for i, j in zip(np.flatnonzero(has), out[has])]
    stats = {"arrivals": int(n), "with_donor": int(has.sum()), "without_donor": int((~has).sum()),
             "donor_shares_runway": int(same_rwy.sum()),
             "gap_s_quantiles": {q: float(np.percentile(f, q)) for q in (50, 90, 99)} if len(f) else {}}
    return out, stats


def control_target(arm: str, y: np.ndarray, A: pd.DataFrame, donor: np.ndarray | None) -> tuple[np.ndarray, dict]:
    """The auxiliary target each representation arm is fitted on (fitting months only)."""
    if arm == "true":
        return y.copy(), {}
    if arm == "const":
        return np.full(len(y), float(np.mean(y))), {}
    ck = cell_key(A)
    codes, u = pd.factorize(pd.Series(ck), sort=True)
    cmean = (np.bincount(codes, weights=y, minlength=len(u))
             / np.bincount(codes, minlength=len(u)))[codes]
    if arm == "gmean":
        return cmean, {"cells": int(len(u))}
    if arm == "shuffle":
        perm = within_perm(ck, np.random.default_rng(SEED))
        return y[perm], {"cells": int(len(u)), "rows_moved": int((perm != np.arange(len(y))).sum())}
    if arm == "donor":
        out = np.where(donor >= 0, y[np.maximum(donor, 0)], cmean)
        return out, {"fallback_to_cell_mean": int((donor < 0).sum())}
    raise ValueError(arm)


# ---- the departure learner (SPEC 1.7) --------------------------------------------------------
LGB_PARAMS = dict(objective="regression", n_estimators=64, max_depth=2, num_leaves=4,
                  learning_rate=0.05, min_child_samples=200, subsample=1.0, subsample_freq=0,
                  colsample_bytree=1.0, reg_lambda=0.0, max_bin=255, cat_smooth=10.0,
                  min_data_per_group=100, max_cat_threshold=32, deterministic=True,
                  force_row_wise=True, random_state=SEED, n_jobs=THREADS, verbose=-1)


# ---- paths (write-once) ------------------------------------------------------------------------
def rep_path(fold: str, tag: str = "") -> Path:
    return EXP / f"work/rep{tag}-{fold}.parquet"


def rep_json(fold: str, tag: str = "") -> Path:
    return EXP / f"analysis/rep{tag}-{fold}.json"


def h_path(fold: str, tag: str = "") -> Path:
    return EXP / f"work/h{tag}-{fold}.parquet"


def fit_json(fold: str, tag: str = "") -> Path:
    return EXP / f"analysis/fit{tag}-{fold}.json"
