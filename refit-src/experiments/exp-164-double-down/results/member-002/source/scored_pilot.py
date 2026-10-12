"""exp-147 pilot: the splits of outer fold k and the record every fitted stage writes.

Outer-strict reading (REPORT.md section 2): every ancestor of an outer(k) prediction
excludes pair k and its buffers. A member is fitted six times for fold k:

  outer-k        scores pair k                      (the outer prediction)
  inner-k-j      scores inner(k, j).held, 4 j        (out-of-fold rows for the stages above)
  grouped-k      scores December and the five extension months (all days); excludes them,
                 pair k and its buffers              (out-of-fold rows for the head and queue)

`inner(k, j)` excludes the same days as `inner(j, k)`; a member fit is shared, but its
predictions are read only by the outer fold whose split names it (owner k here).

Every artifact is a parquet with `MVT_ID_mvt` and one prediction column per output, plus a
JSON record: `split` (Split.manifest()), `owner` (the outer fold), inputs and code with
sha256, fit settings and seconds. `load(path, owner)` refuses an artifact whose split
outer(owner) does not admit (`Split.assert_admits`) or whose owner is another fold. Under the
outer-strict reading this is the isolation check for every stage of fold k, inner stages
included: an inner(k, j) stage may read inner(k, i) artifacts, never one that saw pair k.
"""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import time
from pathlib import Path

import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from prc import weekpair as WP      # noqa: E402

ID = "MVT_ID_mvt"
MONTHS = tuple(f"2025-{m:02d}" for m in range(1, 13))
SCORED_MONTHS = ("2025-01", "2025-07")
#: December and the months the served surface head adds (Jan head: Nov Oct Mar Feb; Jul head:
#: Nov Oct Mar Aug; serve.py base set v25)
GROUPED_MONTHS = ("2025-02", "2025-03", "2025-08", "2025-10", "2025-11", "2025-12")


def _days(month: str) -> list[str]:
    return [d.date().isoformat() for d in pd.date_range(f"{month}-01", periods=pd.Period(month).days_in_month)]


def grouped(k: int) -> WP.Split:
    """Excludes outer(k)'s days and every day of GROUPED_MONTHS; scores those months."""
    held = sorted(d for m in GROUPED_MONTHS for d in _days(m))
    return WP.Split(f"grouped-{k}", tuple(held), tuple(sorted(set(WP.outer(k).excluded) | set(held))))


def janjul() -> WP.Split:
    """exp-148 whole-month check (desk scout prior, 2026-09-30): excludes every day of 2025-01
    and 2025-07 and the month-hold buffer days; scores both months. A stacker trained on one
    month's rows of this fit scores the other month (consumers `m01`, `m07`)."""
    held = sorted(d for m in SCORED_MONTHS for d in _days(m))
    excl = set(held)
    for m in SCORED_MONTHS:
        excl |= set(WP.month_hold(m).excluded)
    return WP.Split("janjul", tuple(held), tuple(sorted(excl)))


def jjrank() -> WP.Split:
    """exp-148 fold form (Matt, 10-01: the board entry is the mean of the two fold models): the
    training rows of `janjul` (January, July 2025 and their buffer days left out); scores the 2026
    ranking rows, as `final` does. Written with `jjrefit` by one fit."""
    return WP.Split("jjrank", (), janjul().excluded)


def jjrefit() -> WP.Split:
    """The January and July 2025 rows of the `jjrank` fit: the parity check against the saved
    `janjul` artifact, and the local fold read from the same models as the board entry."""
    j = janjul()
    return WP.Split("jjrefit", j.held, j.excluded)


#: splits whose fit scores the 344,841 ranking rows instead of 2025 rows
RANKING_SPLITS = ("final", "jjrank")


#: exp-148 SPEC-r2 (repair 2): the months whose out-of-fold rows train the level 2 of January
#: and July without any label of those two months.
R2_MONTHS = ("2025-12", "2025-02", "2025-06", "2025-08")
#: exp-148 SPEC-r4 (item M): the further months with exp-046 ungated blocks; the level 2 of R4
#: trains on these and R2's months (nine in all). April, May and September have no block.
R4_MONTHS = ("2025-03", "2025-10", "2025-11")


def jjx(month: str) -> WP.Split:
    """exp-148 SPEC-r2: excludes every day of 2025-01, 2025-07 and `month` and their month-hold
    buffer days; scores `month`. month_hold(2025-01) and month_hold(2025-07) admit it, so its
    rows can train the level 2 of either scored month; the served level 2 reads it too."""
    if month not in R2_MONTHS + R4_MONTHS:
        raise ValueError(f"jjx: {month} is not one of {R2_MONTHS + R4_MONTHS}")
    excl = set(janjul().excluded) | set(WP.month_hold(month).excluded)
    return WP.Split(f"jjx-{month[-2:]}", tuple(_days(month)), tuple(sorted(excl)))


def split(name: str) -> WP.Split:
    if name == "final":
        return WP.FINAL
    if name == "janjul":
        return janjul()
    if name == "jjrank":
        return jjrank()
    if name == "jjrefit":
        return jjrefit()
    kind, *ks = name.split("-")
    if kind == "jjx" and len(ks) == 1:
        return jjx(f"2025-{ks[0]}")
    if kind == "outer" and len(ks) == 1:
        return WP.outer(int(ks[0]))
    if kind == "inner" and len(ks) == 2:
        return WP.inner(int(ks[0]), int(ks[1]))
    if kind == "grouped" and len(ks) == 1:
        return grouped(int(ks[0]))
    raise ValueError(f"unknown split {name}")


def splits_for(k: int) -> list[str]:
    return [f"outer-{k}"] + [f"inner-{k}-{j}" for j in WP.inner_rotation(k)] + [f"grouped-{k}"]


def scored_months(s: WP.Split) -> tuple[str, ...]:
    """2025 months with scored rows; the final fit scores the 2026 ranking months instead."""
    return tuple(sorted({d[:7] for d in s.held}))


#: exp-148: the fits of one member. inner(a, b) and inner(b, a) exclude the same days, so one
#: fit per unordered pair writes both artifacts (owners a and b); `final` scores the ranking.
PAIRS = tuple((a, b) for a in range(1, 6) for b in range(a + 1, 6))
FITS = tuple([f"outer-{k}" for k in range(1, 6)] + [f"inner-{a}-{b}" for a, b in PAIRS] + ["final", "janjul"])
#: exp-148 SPEC-r2: one fit per month of R2_MONTHS, owner is the fit name.
FITS_R2 = tuple(f"jjx-{m[-2:]}" for m in R2_MONTHS)
#: exp-148 SPEC-r4: one fit per month of R4_MONTHS, owner is the fit name.
FITS_R4 = tuple(f"jjx-{m[-2:]}" for m in R4_MONTHS)
FITS = FITS + FITS_R2 + FITS_R4 + ("jjrank",)


def artifacts_of(fit: str) -> list[tuple[WP.Split, int | str]]:
    """(split, owner) of every artifact that one fit writes."""
    if fit == "final":
        return [(WP.FINAL, "final")]
    if fit == "janjul":
        return [(janjul(), "janjul")]
    if fit == "jjrank":
        return [(jjrank(), "jjrank"), (jjrefit(), "jjrefit")]
    if fit in FITS_R2 or fit in FITS_R4:
        return [(split(fit), fit)]
    kind, *ks = fit.split("-")
    if kind == "outer":
        return [(WP.outer(int(ks[0])), int(ks[0]))]
    if kind == "inner":
        a, b = int(ks[0]), int(ks[1])
        sa, sb = WP.inner(a, b), WP.inner(b, a)
        if set(sa.excluded) != set(sb.excluded):
            raise ValueError(f"inner({a},{b}) and inner({b},{a}) exclude different days")
        return [(sa, a), (sb, b)]
    raise ValueError(f"unknown fit {fit}")


def consumer(owner: int | str) -> WP.Split:
    named = {"final": lambda: WP.FINAL, "janjul": janjul, "jjrank": jjrank, "jjrefit": jjrefit,
             "m01": lambda: WP.month_hold("2025-01"), "m07": lambda: WP.month_hold("2025-07"),
             # exp-148 SPEC-r2: the level 2 of each scored month and the served level 2
             "r01": lambda: WP.month_hold("2025-01"), "r07": lambda: WP.month_hold("2025-07"),
             "rfinal": lambda: WP.FINAL,
             # exp-148 SPEC-r4: the same on nine months (R4)
             "q01": lambda: WP.month_hold("2025-01"), "q07": lambda: WP.month_hold("2025-07"),
             "qfinal": lambda: WP.FINAL}
    if owner in FITS_R2 or owner in FITS_R4:
        return split(owner)
    return named[owner]() if owner in named else WP.outer(int(owner))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(out: Path, stage: str, s: WP.Split, owner: int | str, pred: pd.DataFrame, record: dict) -> Path:
    """Write-once: `<out>/<stage>-<split>.parquet` and `.json`."""
    pq, js = out / f"{stage}-{s.name}.parquet", out / f"{stage}-{s.name}.json"
    if pq.exists() or js.exists():
        raise SystemExit(f"{pq} exists; write-once")
    if ID not in pred or pred[ID].duplicated().any():
        raise ValueError(f"{stage} {s.name}: ids missing or repeated")
    out.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(pq, index=False)
    rec = {"stage": stage, "split": s.manifest(), "owner": owner, "protocol": WP.PROTOCOL,
           "splits_fingerprint": WP.fingerprint(), "rows": int(len(pred)),
           "output_sha256": sha(pq), "host": socket.gethostname(),
           "python": platform.python_version(), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           **record}
    js.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    return pq


def load(path: Path, owner: int | str, what: str = "") -> pd.DataFrame:
    """Read an artifact after checking that outer(owner) admits its split, its owner and hash."""
    rec = json.loads(Path(path).with_suffix(".json").read_text())
    consumer(owner).assert_admits(rec["split"], what=what or Path(path).name)
    if rec.get("owner") != owner:
        raise WP.LabelLeak(f"{path} belongs to outer fold {rec.get('owner')}, not {owner}")
    if sha(path) != rec["output_sha256"]:
        raise RuntimeError(f"{path} changed since it was written")
    return pd.read_parquet(path)
