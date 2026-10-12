"""exp-057 originfit: the unregistered arithmetic that RESULTS.md quotes, from analysis/measure.json and the
draw receipts (stdout only, no file, no label read).

    .venv/bin/python experiments/exp-057-originfit/code/ofit_arith.py

Fit-count law rms²(K) = μ² + σ²/K, solved from the mean K = 1 member rms² and one bag's rms²:
plain and carrier from members 1-10 and the K = 10 bag, the sham from members 1-3 and the K = 3 bag
(with the plain K = 3 bag for its m(∞)).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]
M = json.loads((EXP / "analysis/measure.json").read_text())
SP = ("v20", "v21", "v22")
FC = M["fit_count_record_july_H"]
Q = {s: M["rules"]["Z"]["points"][s]["q"] for s in SP}


def law(r: dict, prefix: str, members: range, bag: str, K: int) -> tuple[float, float]:
    k1 = sum(r[f"{prefix}{i}"] ** 2 for i in members) / len(members)
    kb = r[bag] ** 2
    s2 = (k1 - kb) / (1 - 1 / K)
    return kb - s2 / K, s2


print("| step | plain μ² | plain σ² | carrier μ² | carrier σ² | C3 rms² law | C3 rms² measured | carrier m(∞) "
      "| sham μ² | sham σ² | plain μ² (K = 3 fit) | sham m(∞) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for s in SP:
    r = FC[s]["rms_H_step_s"]
    mp, sp = law(r, "P", range(1, 11), "P10bag", 10)
    mc, sc = law(r, "C", range(1, 11), "C10bag", 10)
    ms, ss = law(r, "S", range(1, 4), "S3bag", 3)
    mp3, _ = law(r, "P", range(1, 4), "P3bag", 3)
    print(f"| {s} | {mp:.0f} | {sp:.0f} | {mc:.0f} | {sc:.0f} | {mc + sc / 3:.0f} | {r['C3bag'] ** 2:.0f} | "
          f"{(mc / mp) ** 0.5:.3f} | {ms:.0f} | {ss:.0f} | {mp3:.0f} | {(ms / mp3) ** 0.5:.3f} |")

print("\n| step | q | share of 1 − q, K = 3 | K = 10 | m drop K = 3 → 10 | carrier K = 1 mean m (s1-10) | C3 | "
      "sham K = 1 mean m (s1-3) | S3 |")
print("|---|---|---|---|---|---|---|---|---|")
for s in SP:
    m = FC[s]["m"]
    c1 = sum(m["K1"][f"s{i}"] for i in range(1, 11)) / 10
    s1 = sum(m["sham_K1"][f"s{i}"] for i in range(1, 4)) / 3
    print(f"| {s} | {Q[s]:.3f} | {(1 - m['K3']) / (1 - Q[s]):.3f} | {(1 - m['K10']) / (1 - Q[s]):.3f} | "
          f"{m['K3'] - m['K10']:.3f} | {c1:.3f} | {m['K3']:.3f} | {s1:.3f} | {m['sham_K3']:.3f} |")

print("\ndistinct members per interval draw (member indices drawn with replacement):")
for L in (1, 2, 3):
    d = pd.read_csv(EXP / f"analysis/receipts/draws-L{L}.csv", dtype={"k3_multiset": str, "k10_indices": str})
    k3 = d["k3_multiset"].map(lambda t: len(set(t))).mean()
    k10 = d["k10_indices"].map(lambda t: len(set(t.split()))).mean()
    print(f"  L{L}: K = 3 {k3:.3f} (expected {3 * (1 - (2 / 3) ** 3):.3f}); K = 10 {k10:.3f} "
          f"(expected {10 * (1 - 0.9 ** 10):.3f})")
