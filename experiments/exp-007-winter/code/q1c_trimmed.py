"""Question 1, part c: the january picture with and without the two unreachable rows.

exp-002 finding: a small class of departures carries a hidden block time on the
wrong calendar day. Every observable column on them is ordinary, so no feature
can separate them (exp-002 RESULTS, "How many are reachable"). Two of them land
in the january fold's validation month at LFPG. This file measures what they do
to every january number the desk is using, and re-runs the decomposition on the
rows that are left.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

OUT = HERE.parent / "analysis" / "q1_rollover.md"
# The unreachable rows, by identity rather than by a threshold, so the screen is
# stated and not fitted: hidden block time on an earlier calendar day than both
# the schedule and the take-off, while MVT - SCHED is an ordinary delay.
IMPOSSIBLE = 21600.0   # six hours of taxi; the physical screen, not a fitted one
ROLLOVER_RULE = ("TAXITIME_SEC_mvt > 21600 (physically impossible), and the observable "
                 "columns ordinary: aircraft type present and |MVT - SCHED| <= 3600 s")


def impossible_mask(d: pd.DataFrame) -> pd.Series:
    """S1: a recorded taxi time no aircraft can have produced."""
    return d["y"] > IMPOSSIBLE


def rollover_mask(d: pd.DataFrame) -> pd.Series:
    """S2 = S1 and nothing observable is out of the ordinary (exp-002's *unreachable* class).

    exp-002 separated the reachable members of the class -- `noac` rows, whose
    placeholder signature the noac sub-model keys on, and rows whose schedule is
    24 h stale and therefore visible -- from the rows where the corruption lives
    only in the hidden block time. Only the latter are counted here.
    """
    ms = (d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    return impossible_mask(d) & (~d["noac"]) & ms.abs().le(3600)


def block(d, name):
    nl = d[d["apt"] != "LIRF"]
    return dict(name=name, n=len(d), rmse_all=C.rmse(d["err"]), n_nl=len(nl),
                rmse_nl=C.rmse(nl["err"]),
                rmse_nl_bulk=C.rmse(nl[(~nl["no_nm"]) & (~nl["tail"])]["err"]),
                rmse_nl_tail=C.rmse(nl[(~nl["no_nm"]) & nl["tail"]]["err"]),
                rmse_nl_nonm=C.rmse(nl[nl["no_nm"]]["err"]),
                rmse_lirf=C.rmse(d[d["apt"] == "LIRF"]["err"]))


def main():
    parts = ["# exp-007 question 1c — the two unreachable rows and what they do to the january fold\n"]
    parts.append(f"Screen (stated, not fitted): `{ROLLOVER_RULE}`, i.e. exp-002's date-rollover class\n"
                 "restricted to the rows whose observable columns are ordinary.\n")
    tab = ["| fold | taxi > 6 h (S1) | of those, unreachable (S2) | S2 SSE share of the whole fold | of the non-LIRF slice |",
           "|---|---|---|---|---|"]
    keep = {}
    for fold in C.FOLDS:
        d = pd.read_parquet(HERE.parent / "work" / f"rows_{fold}.parquet")
        r = rollover_mask(d)
        imp = impossible_mask(d)
        nl = d["apt"] != "LIRF"
        s_all = d.loc[r, "se"].sum() / d["se"].sum()
        s_nl = d.loc[r & nl, "se"].sum() / d.loc[nl, "se"].sum()
        tab.append(f"| {fold} | {int(imp.sum())} ({int((imp & nl).sum())} non-LIRF) | "
                   f"{int(r.sum())} ({int((r & nl).sum())} non-LIRF) | "
                   f"**{100*s_all:.1f} %** | **{100*s_nl:.1f} %** |")
        keep[fold] = (d, r)
    parts.append("\n".join(tab) + "\n")

    parts.append("## The rows themselves\n")
    cols = ["fold", "apt", "day", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
            "SCHED_TIME_UTC_mvt", "y", "pred", "no_nm", "noac"]
    rr = pd.concat([d[r][cols] for d, r in keep.values()])
    lines = ["| fold | airport | flight | take-off | hidden block | schedule | y (s) | prediction | no-NM | noac | MVT−SCHED (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, x in rr.iterrows():
        ms = (x["MVT_TIME_UTC_mvt"] - x["SCHED_TIME_UTC_mvt"]).total_seconds()
        lines.append(f"| {x['fold']} | {x['apt']} | {x['FLIGHT_mvt']} | {x['MVT_TIME_UTC_mvt']:%Y-%m-%d %H:%M} | "
                     f"{x['BLOCK_TIME_UTC_mvt']:%Y-%m-%d %H:%M} | {x['SCHED_TIME_UTC_mvt']:%Y-%m-%d %H:%M} | "
                     f"{x['y']:,.0f} | {x['pred']:,.0f} | {x['no_nm']} | {x['noac']} | {ms:,.0f} |")
    parts.append("\n".join(lines) + "\n")

    parts.append("## Every fold number, with the class in and out\n")
    hdr = ["| fold | set | rows | RMSE all | RMSE non-LIRF | non-LIRF bulk | non-LIRF tail | non-LIRF no-NM | RMSE LIRF |",
           "|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for fold in C.FOLDS:
        d, r = keep[fold]
        for nm, sub in (("as scored", d), ("rollover removed", d[~r])):
            b = block(sub, nm)
            res[(fold, nm)] = b
            hdr.append(f"| {fold} | {nm} | {b['n']:,} | {b['rmse_all']:.1f} | **{b['rmse_nl']:.1f}** | "
                       f"{b['rmse_nl_bulk']:.1f} | {b['rmse_nl_tail']:.1f} | {b['rmse_nl_nonm']:.1f} | "
                       f"{b['rmse_lirf']:.1f} |")
    parts.append("\n".join(hdr) + "\n")

    # exp-006's board blend, recomputed
    parts.append("## The board blend of exp-006, recomputed\n")
    # exp-006's LIRF mix: 15,838 July and 11,061 January LIRF departures
    wl, wj = 15838 / 26899, 11061 / 26899
    # the other nine's OWN mix, which is what the other-nine blend needs:
    # (192,122 - 15,838) July and (152,719 - 11,061) January
    wl_all, wj_all = 176284 / 317942, 141658 / 317942
    lines = ["| set | other nine (jul/jan blend) | LIRF (jul/jan blend) | implied total |", "|---|---|---|---|"]
    for nm in ("as scored", "rollover removed"):
        j7, j1 = res[("july", nm)], res[("january", nm)]
        nine = np.sqrt(wl_all * j7["rmse_nl"] ** 2 + wj_all * j1["rmse_nl"] ** 2)
        lirf = np.sqrt(wl * j7["rmse_lirf"] ** 2 + wj * j1["rmse_lirf"] ** 2)
        tot = np.sqrt(0.922 * nine ** 2 + 0.078 * lirf ** 2)
        lines.append(f"| {nm} | **{nine:.0f}** | {lirf:.0f} | {tot:.0f} |")
    parts.append("\n".join(lines))
    parts.append(f"\n(The other-nine blend uses the other nine airports' OWN ranking mix, "
                 f"{100*wl_all:.1f} % july / {100*wj_all:.1f} % january — 176,284 July and 141,658 "
                 f"January non-LIRF ranking departures. exp-006's {100*wl:.1f}/{100*wj:.1f} is "
                 f"LIRF's own mix and is kept for the LIRF column. The implied total uses exp-006's "
                 f"7.8 % LIRF share, 26,899 of 344,841.)\n")

    OUT.write_text("\n".join(parts))
    print(OUT.read_text())


if __name__ == "__main__":
    main()
