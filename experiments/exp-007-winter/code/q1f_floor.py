"""The floor an undetectable date-rollover row puts on EVERY team's score.

exp-002 established the class and its unreachability. This file does the board
arithmetic the desk asked for: how many such rows the 344,841 ranking departures
are expected to hold, what each one costs everyone, and what the top of the
board looks like if the cost is being paid.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import prc.data as D  # noqa: E402

N_RANK = 344_841
N_RANK_LIRF = 26_899
N_RANK_NINE = N_RANK - N_RANK_LIRF
# gpt-5.6-sol round 1, blocking finding 2: exp-002's convention treats every event
# as a clean one-day error, but the three observed events are 84,240, 58,206 and
# 87,341 s and `EJU42AY` is a 16.2-hour error. The coefficient below is therefore
# the ROOT MEAN SQUARE of the three observed residuals against a normal ~1,000 s
# prediction, not a single assumed magnitude. The one-day figure is kept beside it
# because exp-002's rows quote it.
ERR_ASSUMED = 86_400.0
OBSERVED_RESIDUALS = (83_335.0, 57_070.0, 86_401.0)   # EJU983W, EJU42AY, NJE389D
ERR = float(np.sqrt(np.mean(np.square(OBSERVED_RESIDUALS))))


def main():
    # base rate over the twelve training months, from the same screen as q1c
    tot = {"all": 0, "nine": 0, "lirf": 0}
    hit = {"all": 0, "nine": 0, "lirf": 0}
    rows = []
    for m in D.MONTHS:
        df = D.load_month(m)
        dep = df[(df["PHASE_mvt"] == "DEP") & df[D.TARGET].notna()]
        y = dep[D.TARGET].astype(float)
        ms = (dep["MVT_TIME_UTC_mvt"] - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
        unreach = (y > 21600) & dep["AIRCRAFT_TYPE_mvt"].notna() & ms.abs().le(3600)
        nine = dep["ADEP_mvt"].ne("LIRF")
        tot["all"] += len(dep); hit["all"] += int(unreach.sum())
        tot["nine"] += int(nine.sum()); hit["nine"] += int((unreach & nine).sum())
        tot["lirf"] += int((~nine).sum()); hit["lirf"] += int((unreach & ~nine).sum())
        rows.append((m, len(dep), int(unreach.sum()), int((unreach & nine).sum())))
    out = ["# exp-007 — the rollover floor on everybody's score\n",
           "## Base rate of the unreachable class in the twelve training months\n",
           "Screen: recorded taxi above 6 h (physically impossible), aircraft type present",
           "(so the `noac` sub-model cannot reach it) and |take-off − schedule| ≤ 1 h (so the",
           "corruption is invisible in every observable column). This is exp-002's",
           "*unreachable* sub-class; its reachable siblings are handled by the noac head.\n",
           "| slice | 2025 departures | unreachable rows | rate | expected in the ranking file |",
           "|---|---|---|---|---|"]
    exp = {}
    for k, n_rank in (("all", N_RANK), ("nine", N_RANK_NINE), ("lirf", N_RANK_LIRF)):
        rate = hit[k] / tot[k]
        exp[k] = rate * n_rank
        out.append(f"| {k} | {tot[k]:,} | {hit[k]} | {rate:.3e} | **{exp[k]:.2f}** |")
    lam = exp["all"]
    out.append("")
    # exact Poisson rate interval from 3 events, projected to ranking exposure
    from scipy.stats import chi2
    lo_rate = chi2.ppf(0.025, 2 * hit["all"]) / 2 / tot["all"]
    hi_rate = chi2.ppf(0.975, 2 * (hit["all"] + 1)) / 2 / tot["all"]
    out.append(f"The rate rests on **three events in twelve months**. The exact Poisson 95 % "
               f"interval on it, projected to the ranking file's exposure, is "
               f"**{lo_rate*N_RANK:.2f} to {hi_rate*N_RANK:.2f}** expected rows — and the events are "
               f"not independent draws from a homogeneous rate: two of the three are the same "
               f"operator at the same airport in the same month (gpt-5.6-sol round 1, blocking "
               f"finding 4). Restricting the class to genuine one-day errors leaves two events and "
               f"exp-002's mean of 0.33. Treat 0.50 as a point estimate with that interval, not as "
               f"a calibrated probability.\n")
    out.append(f"Poisson with mean {lam:.2f}: "
               + ", ".join(f"P({k} rows) = {np.exp(-lam)*lam**k/math.factorial(k):.3f}" for k in range(4))
               + ".\n")

    out.append("## What each one costs every team\n")
    out.append(f"An unpredicted row of this class adds its own squared residual to the total "
               f"squared error of *every* submission. The three observed events are not the same "
               f"size — {', '.join(f'{v:,.0f}' for v in OBSERVED_RESIDUALS)} s against a normal "
               f"prediction — so the coefficient is their root mean square, "
               f"**{ERR:,.0f} s**, not a single assumed day:\n")
    out.append(f"```\nfloor(k) = sqrt(k) * {ERR:,.0f} / sqrt({N_RANK:,}) = sqrt(k) * "
               f"{ERR/np.sqrt(N_RANK):.1f} s\n```\n")
    out.append(f"(exp-002's rows quote `sqrt(k) x {ERR_ASSUMED/np.sqrt(N_RANK):.1f} s`, which assumes "
               f"every event is a clean one-day error. Individually the two January events are worth "
               + " and ".join(f"{v/np.sqrt(N_RANK):.1f} s" for v in OBSERVED_RESIDUALS[:2])
               + ". gpt-5.6-sol round 1, blocking finding 2.)\n")
    out.append("| k | floor | a team whose score is 277.29 has skill | 278.37 | 278.48 | 291.06 |")
    out.append("|---|---|---|---|---|---|")
    for k in range(4):
        f = np.sqrt(k) * ERR / np.sqrt(N_RANK)
        cells = []
        for s in (277.2904, 278.3702, 278.4753, 291.0632):
            v = s ** 2 - f ** 2
            cells.append(f"{np.sqrt(v):.1f}" if v > 0 else "impossible")
        out.append(f"| {k} | {f:.1f} s | " + " | ".join(cells) + " |")
    out.append("")
    out.append("The three leaders sit at 277.29 / 278.37 / 278.48 — a spread of **1.19 s** — and the "
               "fourth is 12.6 s behind. A shared floor compresses skill differences: with "
               "`score = sqrt(skill² + floor²)`, `d(score)/d(skill) = skill/score < 1`, so the same "
               "three teams span 1.19 s of score but")
    for k in (0, 1, 2, 3):
        f = np.sqrt(k) * ERR / np.sqrt(N_RANK)
        a = np.sqrt(max(277.2904 ** 2 - f ** 2, 0)); b = np.sqrt(max(278.4753 ** 2 - f ** 2, 0))
        out.append(f"- k = {k}: {b - a:.2f} s of skill (floor {f:.0f} s)")
    out.append("")
    out.append("so the cluster at the top is what a shared floor looks like, and it is also what "
               "three equally good teams look like. The cluster is **consistent with** a floor; it "
               "is not evidence for one. What is not in doubt is the expected count above and the "
               "cost per row: the January half of the ranking file is a lottery ticket worth "
               f"{ERR/np.sqrt(N_RANK):.0f} s of RMSE per undetectable row, drawn identically for "
               "every team, and our own local folds draw from the same urn — the january fold drew "
               "two.\n")
    out.append("## What the board can and cannot say about k\n")
    out.append("Read the table the other way. At k = 3 the leaders' skill would be 160 s; at "
               "k = 2, 207 s. Our own NM-bulk RMSE — the easiest slice there is, 98 % of rows, an "
               "NM join present and a target under an hour — is 194 s on january and 227 s on "
               "july, and the bulk is where the irreducible noise of a taxi-out time lives. So a "
               "large k asks the leaders to beat us on the easy slice by a lot. **That is a "
               "plausibility argument and not a bound**: it compares a hypothetical overall skill "
               "against one pipeline's bulk RMSE, the two are different quantities, and nothing "
               "here forbids a team from holding 160 s (gpt-5.6-sol round 1, blocking finding 4). "
               "With the corrected coefficient the earlier version of this paragraph — which "
               "claimed the board *bounds* k <= 2 because k = 3 would need a 109 s bulk — no "
               "longer even has its arithmetic: 109 was computed with the one-day coefficient.\n")
    out.append("### What a submission would and would not settle\n")
    ours = 286.0
    out.append(f"A board reading of a pipeline whose cleaned fold skill is {ours:.0f} s would come "
               f"back at `sqrt(skill² + floor(k)²)`:\n")
    out.append("| k | predicted board score at a fold skill of 286 s |")
    out.append("|---|---|")
    for k in range(3):
        f = np.sqrt(k) * ERR / np.sqrt(N_RANK)
        out.append(f"| {k} | {np.sqrt(ours**2 + f**2):.0f} s |")
    out.append("\nThose are 28 and 27 seconds apart, so a reading does carry information about k. "
               "But it does **not** measure k, and the earlier claim that it does was too strong. "
               "A single score conflates at least five things: how many such rows the ranking file "
               "holds, how large each one is, whether any model responds to the identity axis of "
               "finding 1, how the folds transfer to 2026 at all, and the January-2026 shift that "
               "`docs/shift-2026.md` measured and no fold can price. What a reading gives is the "
               "SUM of those, which is still the single most informative number the desk could "
               "buy — it is just not a decomposition.\n")
    out.append("## Per-month history of the draw\n")
    out.append("| month | departures | unreachable | of those, non-LIRF |")
    out.append("|---|---|---|---|")
    for m, n, u, un in rows:
        out.append(f"| {m} | {n:,} | {u} | {un} |")
    p = HERE.parent / "analysis" / "q1_floor.md"
    p.write_text("\n".join(out))
    print(p.read_text())


if __name__ == "__main__":
    main()
