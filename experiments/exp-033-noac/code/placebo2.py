"""The null for a matched-subset arm: permute the QUERY side, keep everything else.

Sol round 1, S6.  The first placebo drew a random 2025 subset of the same
per-band size.  That preserves neither the geometry of a nearest-neighbour match
nor its day structure nor its match propensity, and in the matched scope it also
moved the scoring population, so it did not price the sampling variance of the
actual match.

This one does, as far as 2025 allows.  The reference query set is built from
**calendar-day blocks with the season structure of the ranking months** (Sol
round 2, R2-3): 31 winter days from Dec/Jan/Feb 2025 and 31 summer days from
Jun/Jul/Aug 2025, the noac rows on those days, then capped per band at the 388's
own per-band counts.  The first version sampled individual rows across all twelve
months, so the fake queries occupied 189-216 calendar days against the real 62
and their day clustering, season and match propensity all differed from the real
queries'.  The identical pipeline then runs: the same within-band
k-nearest-neighbour rule in the same feature space, the same per-band head, the
same fold split, the same base.  A fake query may not take a neighbour from its
own calendar day, which is the year-disjointness the real 2026 queries face for
free; that ban costs only the fake side, so it moves the placebo's label effect
toward zero and makes the real effect slightly EASIER to see, not harder.

The structure match is not exact and the diagnostics say by how much: the number
of query rows, query days and matched rows per draw are written to
`analysis/placebo_reps_diag.csv` beside every replicate.

Two things are held fixed across real and placebo, so only the treatment moves:

  the scoring population   the REAL matched rows (and, for the second scope,
                           every noac row of the fold), never the placebo's own
  the fit months           each fold's training months

The statistic is an add-one permutation p-value, `(1 + #{placebo >= real}) /
(1 + n)`, which with `NREP` draws cannot go below `1/(NREP+1)`; the percentile
is reported beside it as a description, not as the test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prc.data as D  # noqa: E402
import direction as DIR  # noqa: E402
from describe import LAB  # noqa: E402
from direction import apply_head, fit_head, rmse  # noqa: E402

OOF = ROOT / "experiments" / "exp-026-rebuild" / "oof" / "stack5-{fold}.parquet"
NREP = 200
K = DIR.K


def query_match(a: pd.DataFrame, d_aa: np.ndarray, qi: np.ndarray, k: int = K) -> np.ndarray:
    """Matched mask when the QUERIES are 2025 rows `qi`; no same-day neighbour."""
    day = a.day.to_numpy()
    band = a.band.to_numpy()
    out = np.zeros(len(a), bool)
    for lab in LAB:
        q = qi[band[qi] == lab]
        p = np.flatnonzero(band == lab)
        if not len(q) or len(p) < 2:
            continue
        sub = d_aa[np.ix_(q, p)].copy()
        sub[day[q][:, None] == day[p][None, :]] = np.inf
        nb = np.argsort(sub, axis=1)[:, :min(k, len(p))]
        out[p[np.unique(nb.ravel())]] = True
    return out


WINTER = ("2025-12", "2025-01", "2025-02")
SUMMER = ("2025-06", "2025-07", "2025-08")


def fake_queries(a: pd.DataFrame, counts: dict, rng) -> np.ndarray:
    """A query set of 2025 rows with the ranking months' DAY-BLOCK structure.

    31 winter days and 31 summer days, the noac rows on them, capped per band at
    the 388's own counts.  Day clustering and the season split are preserved as
    far as 2025 supplies them; the residual mismatch is in the diagnostics.
    """
    day = a.day.to_numpy()
    pick = []
    for months, n in ((WINTER, 31), (SUMMER, 31)):
        d = np.unique(day[a.month.isin(months).to_numpy()])
        pick.append(rng.permutation(d)[:min(n, len(d))])
    keep = np.flatnonzero(pd.Series(day).isin(np.concatenate(pick)).to_numpy())
    band = a.band.to_numpy()
    out = []
    for lab, n in counts.items():
        c = keep[band[keep] == lab]
        out.append(rng.permutation(c)[:min(n, len(c))])
    return np.concatenate([o for o in out if len(o)])


def fold_frames(a: pd.DataFrame):
    keep = []
    for fold in ("july_loo", "december", "january"):
        vm = D.FOLDS[fold].valid_month
        oof = pd.read_parquet(str(OOF).format(fold=fold))[["MVT_ID_mvt", "y_pred"]]
        va = a[a.month == vm].merge(oof, on="MVT_ID_mvt", how="left")
        va["fold"] = fold
        keep.append(va)
    return pd.concat(keep, ignore_index=True)


def corrections(a: pd.DataFrame, mm: np.ndarray, v: pd.DataFrame) -> np.ndarray:
    """The per-row (matched-fit minus all-fit) correction on the fold rows."""
    aa = a.assign(_m=mm)
    out = np.zeros(len(v))
    for fold in ("july_loo", "december", "january"):
        fit = aa[aa.month.isin(D.FOLDS[fold].train_months)]
        m = (v.fold == fold).to_numpy()
        s = v[m]
        out[m] = apply_head(fit_head(fit[fit._m]), s) - apply_head(fit_head(fit), s)
    return out


def main() -> None:
    a = pd.read_parquet(EXP / "work" / "noac_2025_lab.parquet")
    b = pd.read_parquet(EXP / "work" / "noac_2026_d.parquet")
    d_aa = np.load(EXP / "work" / "d_aa_rowlevel.npy")
    real = DIR.matched_mask(a, b)
    v = fold_frames(a)
    v["_m_real"] = v.MVT_ID_mvt.isin(a.MVT_ID_mvt[real]).to_numpy()
    scopes = {"matched (real rows, fixed)": v._m_real.to_numpy(),
              "all noac rows of the fold": np.ones(len(v), bool)}

    c_real = corrections(a, real, v)
    band = v.band.to_numpy()
    y, base = v.y_true.to_numpy(float), v.y_pred.to_numpy(float)

    def gains(c):
        g = {}
        for sname, sel in scopes.items():
            r0 = rmse(y[sel], base[sel])
            for lab in LAB + ["ALL"]:
                m = np.ones(len(v), bool) if lab == "ALL" else (band == lab)
                arm = base + np.where(m, c, 0.0)
                g[(sname, lab)] = r0 - rmse(y[sel], arm[sel])
        return g

    g_real = gains(c_real)

    counts = {lab: int((b.band == lab).sum()) for lab in LAB}
    rng = np.random.default_rng(404)
    reps, diag = [], []
    for r in range(NREP):
        qi = fake_queries(a, counts, rng)
        mm = query_match(a, d_aa, qi)
        gp = gains(corrections(a, mm, v))
        for kk, vv in gp.items():
            reps.append({"rep": r, "scope": kk[0], "band": kk[1], "gain": vv})
        diag.append({"rep": r, "n_queries": len(qi),
                     "n_query_days": int(len(np.unique(a.day.to_numpy()[qi]))),
                     "n_matched": int(mm.sum())})
        if (r + 1) % 25 == 0:
            print(f"  {r + 1}/{NREP} placebo draws", flush=True)
    rp = pd.DataFrame(reps)
    rp.to_csv(EXP / "analysis" / "placebo_reps.csv", index=False)
    dg = pd.DataFrame(diag)
    dg.to_csv(EXP / "analysis" / "placebo_reps_diag.csv", index=False)
    print("\nstructure of the fake query sets against the real one "
          "(388 rows, %d query days, %d matched rows):"
          % (len(np.unique(pd.to_datetime(b.mvt_time, utc=True).dt.floor("D"))), real.sum()))
    print(dg[["n_queries", "n_query_days", "n_matched"]].describe()
          .loc[["min", "50%", "max"]].round(0).to_string())

    out = []
    for (sname, lab), g in g_real.items():
        col = rp[(rp.scope == sname) & (rp.band == lab)].gain.to_numpy()
        out.append({"scope": sname, "band": lab, "gain_real": g,
                    "n_draws": len(col), "placebo_mean": col.mean(),
                    "placebo_sd": col.std(), "placebo_p95": float(np.percentile(col, 95)),
                    "placebo_max": col.max(),
                    "percentile": float((col < g).mean()),
                    "add_one_p": float((1 + (col >= g).sum()) / (1 + len(col)))})
    t = pd.DataFrame(out)
    t.to_csv(EXP / "analysis" / "placebo.csv", index=False)
    for sname in scopes:
        print(f"\nscope = {sname}")
        print(t[t.scope == sname][["band", "gain_real", "placebo_mean", "placebo_sd",
                                   "placebo_p95", "percentile", "add_one_p"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
