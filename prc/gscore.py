"""G statistic and airport-day resampling for the week-pair protocol (exp-146; docs/VALIDATION.md).

For each held departure i, d_i = (y_i - base_i)^2 - (y_i - cand_i)^2 (positive = the
candidate is better). With cells h = airport x month x class and ranking weights
W_h = N_h / 344,841 counted from the ranking inputs:

    G = sum over supported h of W_h * mean_h(d_i)        [s^2 per ranking row]

A cell is supported when it has at least `MIN_SUPPORT` scored 2025 rows; an
unsupported cell contributes zero (its ranking rows keep the comparator's route) and
its weight is NOT given to other cells. Rows the candidate leaves unchanged have
d_i = 0 and stay in their cell mean. Selection rule: G > 0 on the all-row view,
reported with its interval; there is no per-cell rule (positive-point rule).

Intervals: 2,000 draws, seed 601, of airport-days resampled with replacement within
each airport x month, all classes of that airport-day together. The draws are a
fixed array of uniforms, hash-checked, so every candidate sees the same draws.
Checks: common calendar days (all airports together) and circular seven-day blocks.
The intervals measure day-to-day variation in 2025 only, not the 2025 -> 2026 shift.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from prc import weekpair as WP

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path(os.environ.get("PRC_VALIDATION_DIR", ROOT / "experiments/exp-146-validation/frozen"))
N_RANKING = 344_841
MIN_SUPPORT = 2
SEED = 601
N_DRAWS = 2000
BLOCK_LEN = 7
FLOOR = 60.0
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
MONTHS = (1, 7, 12)
CLASSES = ("clock", "ordinary", "noac")
BANDS = ((-np.inf, 0.0, "y<=0"), (0.0, 3600.0, "0<y<=3600"), (3600.0, 14400.0, "3600<y<=14400"),
         (14400.0, np.inf, "y>14400"))


# ------------------------------------------------------------------ frozen inputs
def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest() -> dict:
    return json.loads((FROZEN / "MANIFEST.json").read_text())


def _checked(name: str) -> Path:
    p = FROZEN / name
    want = manifest()["files"][name]
    got = _sha256_file(p)
    if got != want:
        raise RuntimeError(f"{p}: sha256 {got[:12]} is not the frozen {want[:12]}")
    return p


def load_rows() -> pd.DataFrame:
    """Frozen per-row table of the 2025 January, July and December departures."""
    return pd.read_parquet(_checked("rows.parquet"))


def load_cells() -> pd.DataFrame:
    """Ranking cell counts N and weights W by (month, airport, cls)."""
    return pd.read_csv(_checked("cells.csv"))


def load_clock_loss_days() -> pd.DataFrame:
    return pd.read_csv(_checked("clock_loss_days.csv"))


def draw_uniforms(kind: str) -> np.ndarray:
    """The frozen uniforms for one resampling scheme, generated from seed 601.

    'airport_day': (2000, 30 strata = month x airport, 31 day slots)
    'common_day' : (2000, 3 months, 31 day slots)
    'week_block' : (2000, 3 months, 5 block starts)
    """
    shapes = {"airport_day": (N_DRAWS, len(MONTHS) * len(AIRPORTS), 31),
              "common_day": (N_DRAWS, len(MONTHS), 31),
              "week_block": (N_DRAWS, len(MONTHS), 5)}
    i = list(shapes).index(kind)
    ss = np.random.SeedSequence(SEED).spawn(len(shapes))[i]
    return np.random.Generator(np.random.PCG64(ss)).random(shapes[kind])


def _uniforms_checked(kind: str) -> np.ndarray:
    u = draw_uniforms(kind)
    got = hashlib.sha256(u.tobytes()).hexdigest()
    want = manifest()["draws"][kind]
    if got != want:
        raise RuntimeError(f"resampling draws {kind}: sha256 {got[:12]} is not the frozen {want[:12]}")
    return u


def labels(ids: np.ndarray) -> np.ndarray:
    """Departure taxi time for each 2025 id, read from the raw month files."""
    from prc import data as D
    want = pd.Index(ids)
    parts = []
    for m in ("2025-01", "2025-07", "2025-12"):
        f = D.load_month(m, columns=["MVT_ID_mvt", "PHASE_mvt", "TAXITIME_SEC_mvt"])
        f = f[f.PHASE_mvt == "DEP"]
        parts.append(pd.Series(f["TAXITIME_SEC_mvt"].to_numpy("float64"),
                               index=f["MVT_ID_mvt"].to_numpy("int64")))
    y = pd.concat(parts).reindex(want)
    if y.isna().any():
        raise ValueError(f"{int(y.isna().sum())} ids have no 2025 departure label")
    return y.to_numpy("float64")


# ------------------------------------------------------------------ core
def _counts(u: np.ndarray, present: np.ndarray, kind: str) -> np.ndarray:
    """(draws x 31) multiplicity of each day slot, resampling only the present slots."""
    B = u.shape[0]
    pos = np.flatnonzero(present)
    D = len(pos)
    out = np.zeros((B, 31), dtype=np.float64)
    if D == 0:
        return out
    if kind == "week_block":
        nb = int(np.ceil(D / BLOCK_LEN))
        starts = np.floor(u[:, :nb] * D).astype(int)                     # B x nb
        seq = (starts[:, :, None] + np.arange(BLOCK_LEN)[None, None, :]) % D
        seq = seq.reshape(B, -1)[:, :D]
    else:
        seq = np.floor(u[:, :D] * D).astype(int)                         # B x D
    for b in range(B):
        out[b, pos] = np.bincount(seq[b], minlength=D)
    return out


def _cell_means(S: np.ndarray, n: np.ndarray) -> np.ndarray:
    return np.where(n >= MIN_SUPPORT, S / np.maximum(n, 1), 0.0)


def score(ids, base, cand, *, y=None, view: str = "all", population: str = "janjul",
          clock_mask: str | None = None, floor: float | None = FLOOR,
          require_complete: bool = True, top_days: int = 5, R: float | None = None,
          groups: Mapping[str, np.ndarray] | None = None, resample: bool = True) -> dict:
    """Paired score of `cand` against `base` on 2025 held rows.

    ids, base, cand: aligned arrays (MVT_ID_mvt and served predictions, after floors
    and holds). view: 'all' (primary) or 'clean' (drops prc.data's unreached-rollover
    rows). population: 'janjul' (ranking-weighted G) or 'december' (control, row mean).
    clock_mask: None, 'coverage' or 'lirf_july': clock rows on the frozen dark
    airport-days count as ordinary, and the result splits naturally dark from made
    dark. The predictions must already come from the masked rebuild.
    R: the RMSE that converts G into seconds, R - sqrt(R^2 - G). Default: the
    comparator's own ranking-weighted RMSE on the supported cells of this score,
    sqrt(sum W_h mse_h / sum W_h). It is a unit only; no board number enters (score-feedback ban).
    """
    ids = np.asarray(ids, dtype="float64")
    if not np.all(ids == np.round(ids)):
        raise ValueError("ids must be integers")
    ids = ids.astype("int64")
    base = np.asarray(base, dtype="float64")
    cand = np.asarray(cand, dtype="float64")
    if not (len(ids) == len(base) == len(cand)):
        raise ValueError("ids, base and cand must be the same length")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("duplicate ids")
    if not (np.isfinite(base).all() and np.isfinite(cand).all()):
        raise ValueError("non-finite predictions")
    if floor is not None and (base.min() < floor - 1e-9 or cand.min() < floor - 1e-9):
        raise ValueError(f"a prediction is below the {floor} s floor "
                         f"(base min {base.min():.3f}, cand min {cand.min():.3f}); "
                         "score the served form")

    rows = load_rows().set_index("MVT_ID_mvt")
    months = (1, 7) if population == "janjul" else (12,)
    pop = rows[rows.month.isin(months)]
    missing = ~pd.Index(ids).isin(pop.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} ids are not {population} departures")
    if require_complete and len(ids) != len(pop):
        raise ValueError(f"{len(ids)} ids for {len(pop)} {population} departures; "
                         "pass require_complete=False for a partial screen")
    r = pop.loc[ids].copy()
    r["base"], r["cand"] = base, cand
    r["y"] = labels(ids) if y is None else np.asarray(y, dtype="float64")
    extra = {}
    for name, arr in (groups or {}).items():
        if len(arr) != len(ids):
            raise ValueError(f"group {name!r} is not aligned with ids")
        extra[name] = f"g_{name}"
        r[f"g_{name}"] = np.asarray(arr).astype(str)
    if view == "clean":
        from prc import data as D
        ex = {int(i) for m in D.UNREACHED_ROLLOVER for i in D.excluded_ids(m)}
        r = r[~r.index.isin(ex)]
    elif view != "all":
        raise ValueError(f"view {view!r}")

    cls = r["cls"].astype(str).to_numpy().copy()
    made_dark = np.zeros(len(r), bool)
    if clock_mask is not None:
        dd = load_clock_loss_days()
        dd = dd[dd.scheme == clock_mask]
        if dd.empty:
            raise ValueError(f"no frozen clock-loss scheme {clock_mask!r}")
        key = set(zip(dd.airport, dd.date))
        on = np.array([(a, str(d)) in key for a, d in zip(r.airport, r.date.astype(str))])
        made_dark = on & (cls == "clock")
        cls[made_dark] = "ordinary"
    r["cls_eff"] = cls

    d = (r.y - r.base) ** 2 - (r.y - r.cand) ** 2
    r["d"] = d.to_numpy()
    r["dom"] = pd.to_datetime(r.date).dt.day.to_numpy() - 1                  # 0..30

    # cells and weights
    cell_key = list(zip(r.month.astype(int).tolist(), r.airport.astype(str).tolist(),
                        r.cls_eff.tolist()))
    cells = sorted(set(cell_key))
    ci = {c: i for i, c in enumerate(cells)}
    h = np.array([ci[c] for c in cell_key])
    H = len(cells)
    if population == "janjul":
        cw = load_cells().set_index(["month", "airport", "cls"])["N"]
        N = np.array([float(cw.get(c, 0)) for c in cells])
        W = N / N_RANKING
    else:
        N = np.bincount(h, minlength=H).astype(float)
        W = N / N.sum()
    S = np.zeros((H, 31))
    n = np.zeros((H, 31))
    np.add.at(S, (h, r.dom.to_numpy()), r.d.to_numpy())
    np.add.at(n, (h, r.dom.to_numpy()), 1.0)
    n_h = n.sum(1)
    supported = n_h >= MIN_SUPPORT
    mean_h = _cell_means(S.sum(1), n_h)
    G = float((W * mean_h).sum())

    mse_base = float((W * _cell_means(np.bincount(h, (r.y - r.base) ** 2, H), n_h)).sum())
    w_sup = float(W[supported].sum())
    if R is None:
        R = float(np.sqrt(mse_base / w_sup)) if w_sup > 0 else float("nan")

    def contrib(mask: np.ndarray) -> float:
        Sg = np.bincount(h[mask], weights=r.d.to_numpy()[mask], minlength=H)
        return float((W * np.where(supported, Sg / np.maximum(n_h, 1), 0.0)).sum())

    out: dict = {
        "protocol": WP.PROTOCOL, "population": population, "view": view, "clock_mask": clock_mask,
        "n_rows": int(len(r)), "n_changed": int((r.base != r.cand).sum()),
        "G": G, "gain_s": _seconds(G, R), "gain_s_linear": G / (2 * R), "R_reference": R,
        "select_G_positive": bool(G > 0),
        "R_is_local": True, "weight_supported": w_sup,
        "weighted_mse_base": mse_base,
        "weighted_mse_cand": float((W * _cell_means(np.bincount(h, (r.y - r.cand) ** 2, H), n_h)).sum()),
        "rmse_base_rows": float(np.sqrt(np.mean((r.y - r.base) ** 2))),
        "rmse_cand_rows": float(np.sqrt(np.mean((r.y - r.cand) ** 2))),
    }
    unsup = [{"month": int(c[0]), "airport": c[1], "cls": c[2], "N_ranking": int(N[i]),
              "n_scored": int(n_h[i]), "changed": int((r.base != r.cand).to_numpy()[h == i].sum())}
             for i, c in enumerate(cells) if not supported[i] and N[i] > 0]
    if population == "janjul":                         # ranking cells with no scored row at all
        cw = load_cells()
        for m, a, c, Nc in cw[["month", "airport", "cls", "N"]].itertuples(index=False):
            if (m, a, c) not in ci and Nc > 0 and m in months:
                unsup.append({"month": int(m), "airport": a, "cls": c, "N_ranking": int(Nc),
                              "n_scored": 0, "changed": 0})
    out["unsupported_cells"] = unsup
    out["unsupported_ranking_rows"] = int(sum(u["N_ranking"] for u in unsup))

    # additive views: each partition's contributions sum to G
    views = {"class": r.cls_eff.to_numpy(), "airport": r.airport.to_numpy(),
             "month": r.month.astype(str).to_numpy(),
             "band": _bands(r.y.to_numpy()),
             "class_aobt": np.char.add(r.cls_eff.to_numpy().astype(str),
                                       np.where(r.aobt.to_numpy(), "+aobt", "-aobt"))}
    if clock_mask is not None:
        views["darkness"] = np.where(made_dark, "made_dark",
                                     np.where(r.cls_eff.to_numpy() == "ordinary", "natural_dark", "other"))
    for name, col in extra.items():
        views[name] = r[col].to_numpy()
    out["views"] = {name: {str(g): contrib(arr == g) for g in sorted(set(arr.tolist()))}
                    for name, arr in views.items()}

    # airport-days: contributions, largest, and G without the largest
    sa = [(m, a) for m in MONTHS for a in AIRPORTS]
    cell_stratum = np.array([sa.index((c[0], c[1])) for c in cells])
    per_day = (W / np.maximum(n_h, 1) * supported)[:, None] * S          # H x 31
    ad = {}
    for i in range(H):
        for t in np.flatnonzero(n[i]):
            k = (cells[i][1], int(cells[i][0]), int(t) + 1)
            ad[k] = ad.get(k, 0.0) + per_day[i, t]
    top = sorted(ad.items(), key=lambda kv: -kv[1])[:top_days]
    out["top_airport_days"] = [{"airport": k[0], "month": k[1], "day": k[2], "contribution": v}
                               for k, v in top]
    S2, n2 = S.copy(), n.copy()
    for (a, m, dday), _ in top:
        for i in range(H):
            if cells[i][0] == m and cells[i][1] == a:
                S2[i, dday - 1] = 0.0
                n2[i, dday - 1] = 0.0
    out["G_without_top_days"] = float((W * np.where(supported, S2.sum(1) / np.maximum(n2.sum(1), 1),
                                                     0.0)).sum())

    # analytic standard error: day-level residual sums within airport-month strata
    resid = S - mean_h[:, None] * n
    q = (W / np.maximum(n_h, 1) * supported)[:, None] * resid
    var = 0.0
    for s in set(cell_stratum.tolist()):
        qs = q[cell_stratum == s].sum(0)
        present = n[cell_stratum == s].sum(0) > 0
        K = int(present.sum())
        if K > 1:
            var += K / (K - 1) * float((qs[present] ** 2).sum())
    out["se_analytic"] = float(np.sqrt(var))

    if resample:
        out["resampling"] = {}
        for kind in ("airport_day", "common_day", "week_block"):
            u = _uniforms_checked(kind)
            Gb = _draw_G(kind, u, S, n, W, supported, cells, sa)
            lo, hi = np.percentile(Gb, [2.5, 97.5])
            out["resampling"][kind] = {
                "se": float(Gb.std(ddof=1)), "lo95": float(lo), "hi95": float(hi),
                "gain_s_lo95": _seconds(float(lo), R), "gain_s_hi95": _seconds(float(hi), R),
                "p_G_le_0": float((Gb <= 0).mean())}
        out["interval"] = out["resampling"]["airport_day"]
    return out


def _draw_G(kind, u, S, n, W, supported, cells, sa) -> np.ndarray:
    B = u.shape[0]
    H = len(cells)
    Sb = np.zeros((B, H))
    nb = np.zeros((B, H))
    if kind == "airport_day":
        for s, (m, a) in enumerate(sa):
            idx = [i for i, c in enumerate(cells) if c[0] == m and c[1] == a]
            if not idx:
                continue
            present = n[idx].sum(0) > 0
            C = _counts(u[:, s, :], present, kind)                         # B x 31
            Sb[:, idx] = C @ S[idx].T
            nb[:, idx] = C @ n[idx].T
    else:
        for mi, m in enumerate(MONTHS):
            idx = [i for i, c in enumerate(cells) if c[0] == m]
            if not idx:
                continue
            present = n[idx].sum(0) > 0
            C = _counts(u[:, mi, :], present, kind)
            Sb[:, idx] = C @ S[idx].T
            nb[:, idx] = C @ n[idx].T
    means = np.where(nb >= MIN_SUPPORT, Sb / np.maximum(nb, 1), 0.0) * supported[None, :]
    return means @ W


def _seconds(G: float, R: float) -> float:
    """Exact whole-entry RMSE gain R - sqrt(R^2 - G)."""
    return float(R - np.sqrt(max(R * R - G, 0.0)))


def _bands(y: np.ndarray) -> np.ndarray:
    out = np.empty(len(y), dtype=object)
    for lo, hi, name in BANDS:
        out[(y > lo) & (y <= hi)] = name
    return out.astype(str)
