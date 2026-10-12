"""exp-171 step 2: the row order of the raw files follows the take-off or landing time before the offset of +-6 s.
At the grid airports the offset is visible in the time itself (W5). At the other airports it is not; there the row
order bounds it: a row that stands before a row with an earlier time has a high offset, and the reverse.
Test: (1) the order estimate against the visible offset at the grid airports; (2) the label against the order
estimate at the other airports (a mirrored offset gives slope 2). 2025 labels; 2026 is a label-free count. No fit
of a model; fold base B and entry E as in step2_kill_tests.py.
Run from the repo root: OMP_NUM_THREADS=4 .venv/bin/python experiments/exp-171-rules/code/step2_order_jitter.py
The script writes experiments/exp-171-rules/step2_order_jitter.log itself and refuses to overwrite it."""
import itertools
import os
import sys

_LOG = "experiments/exp-171-rules/step2_order_jitter.log"
try:
    _fh = open(_LOG, "x")
except FileExistsError:
    sys.exit(_LOG + " exists; write-once. Use a -r1 name for a correction.")


def say(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    _fh.write(s + "\n")
    _fh.flush()


sys.path.insert(0, "experiments/exp-171-rules/code")
import numpy as np
import pandas as pd

from load import AIRPORTS, load

GRID = ["EDDM", "EGLL", "LFPG", "LIRF", "LTFM"]
J = np.arange(-6, 7)

# posterior mean of the offset of a row given the two steps (this - previous, next - this), flat prior on the true gaps
# code 13 = no bound from that side (step of +13 s or more, or no neighbour)
tab = np.zeros((27, 27))
for a, b in itertools.product(range(-13, 14), repeat=2):
    num = den = 0.0
    for jp, j, jn in itertools.product(J, J, J):
        if (a < 13 and j - jp > a) or (b < 13 and jn - j > b):
            continue
        num += j
        den += 1
    tab[a + 13, b + 13] = num / den if den else np.nan

d = load()
d = d.sort_values(["src", "pos"]).copy()
same_p = (d.apt == d.apt.shift()) & (d.src == d.src.shift())
same_n = (d.apt == d.apt.shift(-1)) & (d.src == d.src.shift(-1))
dp = (d.mvt - d.mvt.shift()).where(same_p)
dn = (d.mvt.shift(-1) - d.mvt).where(same_n)
# a step back of more than 12 s is a chunk border: no bound
a = dp.where(dp >= -12).clip(upper=13).fillna(13).astype(int).to_numpy()
b = dn.where(dn >= -12).clip(upper=13).fillna(13).astype(int).to_numpy()
d["jhat"] = tab[a + 13, b + 13]
d["bind"] = (a < 0) | (b < 0)
r = d.mvt % 60
d["jm"] = np.where(r > 30, r - 60, r)
d["grid"] = d.apt.isin(GRID)

say("## 1. grid airports: the order estimate against the visible offset (all rows, both phases, 2025 and 2026)")
g = d[d.grid & (d.jm.abs() <= 6)]
for nm, m in (("all rows", g), ("rows with a step back on one side", g[g.bind])):
    sl = np.polyfit(m.jhat, m.jm, 1)[0]
    say(f"  {nm}: n {len(m)}, corr {np.corrcoef(m.jhat, m.jm)[0, 1]:.3f}, slope of the offset on the estimate {sl:.3f}, "
        f"share of the offset variance explained {1 - ((m.jm - m.jhat) ** 2).mean() / (m.jm ** 2).mean():.3f}")
say("  is the time without the offset in order inside a chunk (grid airports)? share of steps that go back:",
    round(float((((g.mvt - g.jm).diff() < 0) & (g.mvt.diff() > -1800) & same_p.reindex(g.index)).mean()), 5),
    "| with the offset:", round(float(((g.mvt.diff() < 0) & (g.mvt.diff() > -1800) & same_p.reindex(g.index)).mean()), 5))

say("\n## 2. other airports, 2025 departures: the label against the order estimate (slope 2 = a mirrored offset)")
dep = d[d.dep].copy()
dep["MVT_ID"] = dep.MVT_ID_mvt.astype("int64")
tr = dep[~dep.is_rank & ~dep.grid & (dep.taxi > 0) & (dep.taxi < 3600) & ((dep.block - dep.sched).abs() > 6)].copy()
tr["hr"] = (tr.mvt // 3600).astype(int)
tr["res"] = tr.taxi - tr.groupby(["apt", "hr"], observed=True).taxi.transform("mean")
n_c = tr.groupby(["apt", "hr"], observed=True).taxi.transform("size")
tr = tr[n_c >= 4]
for nm, m in [("all five", tr)] + [(a_, tr[tr.apt == a_]) for a_ in AIRPORTS if a_ not in GRID]:
    x = m.jhat - m.groupby(["apt", "hr"], observed=True).jhat.transform("mean")
    sl = float((x * m.res).sum() / (x ** 2).sum())
    se = float(np.sqrt(((m.res - sl * x) ** 2).mean() / (x ** 2).sum()))
    say(f"  {nm}: rows {len(m)}, rows with a step back {int(m.bind.sum())} ({m.bind.mean():.3f}), mean jhat^2 {float((m.jhat ** 2).mean()):.3f}, "
        f"slope {sl:.2f} (se {se:.2f})")
m = tr[tr.bind]
say("  rows with a step back, by the sign of the estimate: mean residual of the label",
    {k: (int(len(v)), round(float(v.res.mean()), 1), round(float(v.jhat.mean()), 2)) for k, v in m.groupby(np.sign(m.jhat.round(6)))})
say("  control, grid airports with the same code (the label holds 2 x the visible offset):")
tg = dep[~dep.is_rank & dep.grid & (dep.taxi > 0) & (dep.taxi < 3600) & ((dep.block - dep.sched).abs() > 6)].copy()
tg["hr"] = (tg.mvt // 3600).astype(int)
tg["res"] = tg.taxi - tg.groupby(["apt", "hr"], observed=True).taxi.transform("mean")
x = tg.jhat - tg.groupby(["apt", "hr"], observed=True).jhat.transform("mean")
sl = float((x * tg.res).sum() / (x ** 2).sum())
se = float(np.sqrt(((tg.res - sl * x) ** 2).mean() / (x ** 2).sum()))
say(f"    rows {len(tg)}, slope {sl:.2f} (se {se:.2f})")

say("\n## 3. the fold base B and the bound (other airports)")
STACK = "experiments/exp-169-stack-plan"
sys.path.insert(0, os.path.join(STACK, "code"))
from assembly import compose  # noqa: E402

sys.path.remove(os.path.join(STACK, "code"))
bb = pd.read_parquet(os.path.join(STACK, "inputs/unit2-ready/bundle-fold.parquet"))
B = pd.Series(np.asarray(compose(bb, route="R4mk", noplan="v3cal", practice="selected"), float), index=bb.MVT_ID_mvt.to_numpy("int64"))
dep["B"] = dep.MVT_ID.map(B)
fold = dep[dep.B.notna() & ~dep.is_rank]
N_F = len(fold)
for nm, m in (("other airports", fold[~fold.grid]), ("grid airports", fold[fold.grid])):
    m = m[(m.taxi < 3600) & ((m.block - m.sched).abs() > 6)]
    sl = float((m.jhat * (m.taxi - m.B)).sum() / (m.jhat ** 2).sum())
    se = float(np.sqrt(((m.taxi - m.B - sl * m.jhat) ** 2).mean() / (m.jhat ** 2).sum()))
    gain = float((((m.taxi - m.B) ** 2) - ((m.taxi - m.B - 2 * m.jhat) ** 2)).sum() / N_F)
    say(f"  {nm}: fold rows {len(m)}, slope of (label - B) on the estimate {sl:.2f} (se {se:.2f}); fold gain of B + 2 x estimate: {gain:+.2f} s2 per fold row; "
        f"the size of the step 4 x sum(jhat^2) / N: {float(4 * (m.jhat ** 2).sum() / N_F):.2f}")
rk = dep[dep.is_rank]
say(f"  2026: ranking rows at the other airports {int((~rk.grid).sum())} of {len(rk)}; with a step back {int((rk.bind & ~rk.grid).sum())}; "
    f"4 x sum(jhat^2) / N at the other airports {float(4 * (rk[~rk.grid].jhat ** 2).sum() / len(rk)):.2f}; at the grid airports {float(4 * (rk[rk.grid].jhat ** 2).sum() / len(rk)):.2f}")
_fh.close()
