"""Member OOF and ranking predictions for the stacker, pinned by hash.

Every member file is recorded in `MANIFEST.md` with its sha256 and re-verified
on every load, because exp-012's review round 1 found a 0.44-weight member
sourcing a file another lane was still editing, and this experiment consumes
five such files across four lanes.

Members are prepared exactly as `prc.blend.build_candidate` prepares them --
floor, then the residual guard on the fold's own training-month bounds -- so the
predictions the stacker is fitted on are the same estimators that get served.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.blend as B  # noqa: E402
import prc.data as D  # noqa: E402

#: The three folds every member covers. `july` is NOT a fourth rotation: it
#: validates the SAME MONTH as `july_loo` (2025-07, the same 190,713 rows and
#: the same labels), so fitting a stacker on a set containing one and scoring it
#: on the other trains and scores on the same rows. See SPEC.md.
FOLDS = ("july_loo", "december", "january")
SHARES = {"july_loo": 0.5571321275602379, "january": 0.4428678724397621}

OOF = {
    "T-tailv2":  "experiments/exp-009-noac2/oof/exp-009-noac2-oof-v2.parquet",
    "C2-cat":    "experiments/exp-012-blend2/pinned/C2-cat--C2-d8p-{fold}.parquet",
    "L2-hub":    "experiments/exp-012-blend2/pinned/L2-hub--L2-hub-{fold}.parquet",
    "D1-winter": "experiments/exp-007-winter/oof/d1-ds-s7-{fold}.parquet",
    "N1-nmtail": "experiments/exp-010-nmtail/oof/nmtail-phi_{fold}.parquet",
}
RANKING = {
    "T-tailv2":  "artifacts/exp-012/pred/T-tailv2-ranking.parquet",
    "C2-cat":    "artifacts/exp-008/pred/C2-d8p-ranking.parquet",
    "L2-hub":    "artifacts/exp-008/pred/L2-hub-ranking.parquet",
    "D1-winter": "artifacts/exp-012/pred/D1-winter-ranking.parquet",
    "N1-nmtail": "artifacts/exp-012/pred/N1-nmtail-ranking.parquet",
}
#: `N1-nmtail` predicts NM-joined departures only and is null elsewhere. It may
#: therefore carry weight ONLY where it is defined; `prc.blend` enforces the same
#: rule at candidate-build time.
PARTIAL = ("N1-nmtail",)
NAMES = tuple(OOF)
PINS = EXP / "pins.json"
#: Suffix for analysis outputs, so a run under a different member set cannot
#: overwrite the tables of the set it replaces. Empty for the default set.
TAG = ""


def use_config(path: str | Path) -> None:
    """Point this module at a different member set, e.g. `members-blend3.json`.

    The pin file moves with it, so a config swap cannot silently reuse the pins
    of the set it replaced. Every entry point takes `--config`; with none, the
    module-level addresses above are the shipped stack-1/stack-2 member set.
    """
    global PINS
    cfg = json.loads(Path(path).read_text())
    OOF.clear(); OOF.update(cfg["oof"])
    RANKING.clear(); RANKING.update(cfg["ranking"])
    globals()["NAMES"] = tuple(OOF)
    PINS = EXP / f"pins-{Path(path).stem}.json"
    globals()["TAG"] = "-" + Path(path).stem.replace("members-", "")


def config_from_argv(argv=None) -> None:
    """Apply `--config NAME.json` if present, and SAY which member set is live.

    `arm_choice.py` was run with `--config members-blend3.json` and silently
    used the default member set, because that entry point did not call this
    function -- the run produced numbers identical to the previous member set
    and nothing said why. Every run now prints its member addresses, so a config
    that did not take is visible in the log rather than inferred from a
    suspicious coincidence.
    """
    argv = list(sys.argv if argv is None else argv)
    if "--config" in argv:
        use_config(EXP / argv[argv.index("--config") + 1])
    elif any(a.endswith(".json") and "members" in a for a in argv):
        raise SystemExit(f"a members config was passed without --config: {argv}")
    print(f"[members] set={'default' if not TAG else TAG.lstrip('-')}  "
          f"pins={PINS.name}", flush=True)
    for n, a in OOF.items():
        print(f"[members]   {n:10s} {a}", flush=True)
V2 = ROOT / "artifacts" / "exp-008" / "frames"
DS = ROOT / "artifacts" / "exp-014" / "frames"


# ------------------------------------------------------------------ pins ----
def pin(write: bool = False) -> tuple[dict, list[str]]:
    """(sha256 of every declared member file, list of declared-but-absent ones)."""
    out, missing = {}, []
    for name, tmpl in OOF.items():
        for f in FOLDS:
            p = ROOT / tmpl.format(fold=f)
            key = f"oof:{name}:{f}" if "{fold}" in tmpl else f"oof:{name}"
            if key in out:
                continue
            if p.exists():
                out[key] = D.sha256(p)
            else:
                missing.append(f"absent:{key}:{p.relative_to(ROOT)}")
    for name, p in RANKING.items():
        q = ROOT / p
        if q.exists():
            out[f"rank:{name}"] = D.sha256(q)
        else:
            missing.append(f"absent:rank:{name}:{p}")
    if write:
        PINS.write_text(json.dumps(out, indent=1, sort_keys=True))
    return out, sorted(set(missing))


def verify(require_ranking: bool = False) -> list[str]:
    """Problems with the member set. Empty list is the good case.

    Reports bytes that MOVED since the pin and addresses that are DECLARED BUT
    ABSENT. The second kind matters because `members-blend3.json` names a member
    file exp-012's lane has not published yet, and a config that silently
    produced a working stacker without it would build stack-3 on the wrong member.
    Out-of-fold absences always count; ranking absences count only for a caller
    that serves a candidate, so the rotation can run before the member is
    servable -- which is exactly the state the deice member is in today.
    """
    now, missing = pin(write=False)
    problems = [m for m in missing
                if require_ranking or not m.startswith("absent:rank:")]
    if not PINS.exists():
        pin(write=True)
        return problems
    want = json.loads(PINS.read_text())
    problems += sorted(k for k in want if k in now and want[k] != now[k])
    problems += sorted(f"vanished:{k}" for k in want if k not in now)
    return problems


# -------------------------------------------------------------- features ----
GATE_NUM = ("d_takeoff_aobt", "d_aobt_lobt", "d_takeoff_lobt", "d_takeoff_sched",
            "local_hour", "local_minute_of_day", "queue_ahead_aobt",
            "dep_takeoffs_30m_before", "arr_landings_30m_before", "flight_len")
GATE_CAT = ("airport",)


def gates(fold_or_month: str, ranking: bool = False) -> pd.DataFrame:
    """Row-level gating features, all computable from one month's frame alone.

    Sources are caches another experiment already built, so this costs no fits:
    exp-008's frozen v2 month frames (built from the masked month alone, mask
    invariance verified there) and exp-014's day-state frames. `ds_deice_hr` is
    in `prc.day_state_features.ROW_COLUMNS`, computed by `DayState._deice` from
    the frame itself with no fitted reference, so it does not depend on which
    months `DayState` was fitted on and is legal on any fold.
    """
    if ranking:
        v = pd.read_parquet(V2 / "v2-ranking.parquet")
        ds = pd.read_parquet(DS / "ds-ranking.parquet", columns=["MVT_ID_mvt", "ds_deice_hr"])
    else:
        month = D.FOLDS[fold_or_month].valid_month if fold_or_month in D.FOLDS else fold_or_month
        v = pd.read_parquet(V2 / f"v2-{month}.parquet")
        ds = pd.read_parquet(DS / "ds-train.parquet",
                             columns=["MVT_ID_mvt", "ds_deice_hr", "month_str"])
        ds = ds[ds["month_str"] == month].drop(columns=["month_str"])
    keep = ["MVT_ID_mvt", "noac", "nm_missing"] + list(GATE_NUM) + list(GATE_CAT)
    if "t" in v.columns:
        keep.append("t")
    g = v[keep].copy()
    g["regime"] = B.regimes_from_features(v)
    g = g.merge(ds, on="MVT_ID_mvt", how="left")
    g["MVT_ID_mvt"] = g["MVT_ID_mvt"].astype("float64")
    return g


# --------------------------------------------------------------- members ----
def _guard_bounds(fold: str) -> tuple[float, float]:
    cache = json.loads((ROOT / "experiments" / "exp-008-diversity" / "guard_bounds.json").read_text())
    return tuple(cache[fold])


def fold_frame(fold: str, names=None) -> pd.DataFrame:
    """One fold: prepared member predictions, the label, and the gating features.

    `names` resolves at CALL time. A default of `names=NAMES` binds the tuple at
    DEFINITION time, so `use_config` could rebind the module-level NAMES and this
    function would keep reading the old member list -- silently, whenever the two
    sets share their names and differ only in addresses, which is exactly the
    blend-3 case. It failed loudly only when blend-4 added a sixth name.
    """
    names = NAMES if names is None else names
    base = None
    for n in names:
        p = ROOT / OOF[n].format(fold=fold)
        d = pd.read_parquet(p)
        if "fold" in d.columns:
            d = d[d["fold"] == fold]
        d = d[["MVT_ID_mvt", "y_true", "y_pred"]].rename(columns={"y_pred": n})
        d["MVT_ID_mvt"] = d["MVT_ID_mvt"].astype("float64")
        how = "left" if n in PARTIAL else "inner"
        base = d if base is None else base.merge(d[["MVT_ID_mvt", n]], on="MVT_ID_mvt", how=how)
    g = gates(fold)
    m = base.merge(g, on="MVT_ID_mvt", how="inner")
    if len(m) != len(base):
        raise ValueError(f"{fold}: gates cover {len(m)} of {len(base)} member rows")
    lo, hi = _guard_bounds(fold)
    a = m["d_takeoff_aobt"].to_numpy("float64")
    for n in names:
        v = m[n].to_numpy("float64")
        ok = np.isfinite(v)
        v[ok] = np.maximum(v[ok], B.FLOOR)
        v[ok] = B.apply_residual_guard(v[ok], a[ok], lo, hi, B.FLOOR)
        m[n] = v
    m["fold"] = fold
    return m


def ranking_frame(names=None) -> pd.DataFrame:
    names = NAMES if names is None else names
    base = None
    for n in names:
        d = pd.read_parquet(ROOT / RANKING[n])
        col = "y_pred" if "y_pred" in d.columns else D.TARGET
        d = d[["MVT_ID_mvt", col]].rename(columns={col: n})
        d["MVT_ID_mvt"] = d["MVT_ID_mvt"].astype("float64")
        how = "left" if n in PARTIAL else "inner"
        base = d if base is None else base.merge(d, on="MVT_ID_mvt", how=how)
    g = gates("", ranking=True)
    m = base.merge(g, on="MVT_ID_mvt", how="inner")
    lo, hi = _guard_bounds("_full12") if "_full12" in json.loads(
        (ROOT / "experiments" / "exp-008-diversity" / "guard_bounds.json").read_text()) else (-20657.0, 9069.0)
    a = m["d_takeoff_aobt"].to_numpy("float64")
    for n in names:
        v = m[n].to_numpy("float64")
        ok = np.isfinite(v)
        v[ok] = np.maximum(v[ok], B.FLOOR)
        v[ok] = B.apply_residual_guard(v[ok], a[ok], lo, hi, B.FLOOR)
        m[n] = v
    return m


def board_estimate(per_fold: dict) -> float:
    return float(np.sqrt(sum(SHARES[f] * per_fold[f] ** 2 for f in SHARES)))


if __name__ == "__main__":
    config_from_argv()
    bad = verify()
    print("pins:", "clean" if not bad else f"PROBLEMS {bad}")
    _, missing = pin(write=False)
    if missing:
        print("declared but absent (blocking only for a candidate build):")
        for m in missing:
            print("   ", m)
    for f in FOLDS:
        m = fold_frame(f)
        cov = {n: float(np.isfinite(m[n]).mean()) for n in NAMES}
        print(f"{f}: {m.shape}, coverage " + " ".join(f"{n}={c:.3f}" for n, c in cov.items()))
        print(f"   regimes {dict(pd.Series(m['regime']).value_counts())}")
