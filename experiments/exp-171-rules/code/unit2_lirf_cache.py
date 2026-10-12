"""Unit 2: shared helpers. A LIRF-only frame, the turnaround link, the stack's vectors, a log.

lirf()    departures and arrivals at LIRF, 2025 and ranking; times in float seconds (load.py).
turn()    LIRF departures, each with the last arrival at its stand (the inbound aircraft).
vectors() B = the stack's fold base (form e9771b94), E = the 2026 entry of exp-169 (read only).
Log       a write-once log: the script stops if its log exists.

Caches are scratch files under load.CACHE. No fit here.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
import load as L  # noqa: E402

P = os.path.join(L.CACHE, "lirf_u2.pkl")
PT = os.path.join(L.CACHE, "lirf_turn_u2c.pkl")
PV = os.path.join(L.CACHE, "BE_u2.pkl")
TEN = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LTFM", "LSZH"]


def lirf(refresh=False):
    if os.path.exists(P) and not refresh:
        return pd.read_pickle(P)
    d = L.load()
    d = d[d["apt"] == "LIRF"].copy()
    for c in d.columns:
        if str(d[c].dtype) == "category":
            d[c] = d[c].astype(object)
    d["year"] = np.where(d["is_rank"], 2026, 2025)
    d["lab"] = d["taxi"]
    d["ms"] = d["mvt"] - d["sched"]
    d["present"] = d["lobt"].notna()
    d["ph"] = (d["block"] - d["sched"]).abs() <= 6  # NaN block -> False
    d.to_pickle(P)
    return d


def turn(refresh=False):
    """LIRF departures with derived columns and the inbound arrival at the same stand.

    jm, mvt0   visible offset of the take-off time and the time without it (minute grid)
    al         first three characters of the flight string (the airline)
    late       off-block lateness: AOBT_3 - SCHED with a plan, take-off - 15 min - SCHED without
    far        row with a plan whose LOBT is more than 3,612 s from SCHED (cannot be a placeholder)
    D          lateness for all rows (late with a plan, ms - 900 without)
    a_*        the last arrival at the stand that landed 20 min to 30 h before the take-off;
               linked = that arrival has the same airline. acls: R real in-block, S in-block =
               SCHED, T in-block = landing + 600 s.
    """
    if os.path.exists(PT) and not refresh:
        return pd.read_pickle(PT)
    d = lirf()
    d["jm"] = ((d.mvt + 30) % 60) - 30
    d["mvt0"] = d.mvt - d.jm
    d["al"] = d.FLIGHT_mvt.astype(str).str[:3]
    d["late"] = np.where(d.present, d.aobt - d.sched, d.mvt0 - 900 - d.sched)
    a = d[~d.dep].copy()
    a["tin"] = a.block - a.mvt
    a["acls"] = np.where((a.block - a.sched).abs() <= 6, "S", np.where((a.tin - 600).abs() <= 12, "T", "R"))
    a = a[a.STAND_mvt.notna()].sort_values("mvt")
    a = a[["mvt", "STAND_mvt", "al", "acls", "FLIGHT_mvt", "ADEP_mvt", "sched", "tin", "MVT_ID_mvt"]].rename(
        columns={"mvt": "a_mvt", "FLIGHT_mvt": "a_flight", "ADEP_mvt": "a_from", "sched": "a_sched",
                 "tin": "a_tin", "MVT_ID_mvt": "a_id", "al": "a_al"})
    x = d[d.dep].copy()
    x["key"] = x.mvt - 1200
    has = x.STAND_mvt.notna()
    m = pd.merge_asof(x[has].sort_values("key"), a, left_on="key", right_on="a_mvt", by="STAND_mvt",
                      direction="backward", tolerance=30 * 3600)
    m = pd.concat([m, x[~has]], ignore_index=True)
    m["linked"] = m.a_mvt.notna() & (m.a_al == m.al)
    m["gap"] = m.mvt - m.a_mvt
    m["mon"] = pd.to_datetime(m.mvt, unit="s").dt.month
    m["far"] = m.present & ((m.lobt - m.sched).abs() > 3612)
    m["D"] = np.where(m.present, m.late, m.ms - 900)
    m["id"] = m.MVT_ID_mvt.astype("int64")
    B, E = vectors()
    m["B"] = m.id.map(B)
    m["E"] = m.id.map(E)
    m.to_pickle(PT)
    return m


def vectors(refresh=False):
    if os.path.exists(PV) and not refresh:
        v = pd.read_pickle(PV)
        return v["B"], v["E"]
    stack = os.path.join(ROOT, "experiments/exp-169-stack-plan")
    sys.path.insert(0, os.path.join(stack, "code"))
    from assembly import compose  # noqa: E402

    sys.path.remove(os.path.join(stack, "code"))
    b = pd.read_parquet(os.path.join(stack, "inputs/unit2-ready/bundle-fold.parquet"))
    B = pd.Series(np.asarray(compose(b, route="R4mk", noplan="v3cal", practice="selected"), float),
                  index=b.MVT_ID_mvt.to_numpy("int64"))
    e = pd.read_parquet(os.path.join(stack, "exp169-stack4-mlp-bag6-ext-aobt-c55.parquet"))
    E = pd.Series(e.TAXITIME_SEC_mvt.to_numpy(float), index=e.MVT_ID_mvt.to_numpy("int64"))
    pd.to_pickle({"B": B, "E": E}, PV)
    return B, E


NF, NR = 344419, 344841  # fold rows (January and July 2025), ranking rows


class Log:
    def __init__(self, script):
        name = os.path.splitext(os.path.basename(script))[0] + ".log"
        self.path = os.path.join(HERE, "..", name)
        if os.path.exists(self.path):
            sys.exit(f"{self.path} exists; write-once. Use a -r1 name for a correction.")
        self.lines = []
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 60)
        pd.set_option("display.max_rows", 400)

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.lines.append(s)

    def close(self):
        with open(self.path, "x") as f:
            f.write("\n".join(self.lines) + "\n")


def day_boot(dl, day, n=2000, seed=0):
    """95 % interval of a summed gain, days drawn with replacement; s2 per fold row."""
    g = dl.groupby(day).sum().to_numpy()
    if len(g) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    bs = [g[rng.integers(0, len(g), len(g))].sum() / NF for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def ts(s):
    return pd.to_datetime(s, unit="s").dt.strftime("%m-%d %H:%M")


if __name__ == "__main__":
    m = turn(refresh="--refresh" in sys.argv)
    print(m.shape, m.groupby("year").linked.mean().round(3).to_dict())
