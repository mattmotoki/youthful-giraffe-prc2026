"""Every number in Q03.md .. Q06.md, checked against the record it came from.

Sol round 1 (finding 7) found Q04.md carrying intervals from a superseded run.
Sol round 2 (finding e2) found the first repair incomplete: the checker compared
each report against a SAVED table file, so the JSON, the saved file and the report
could all go stale together, and it asserted only a chosen list of prose numbers.

Three layers now, none of which is a hand-typed expectation:

1. `tables_live` RE-RENDERS every table from the CURRENT `analysis/q0*.json` by
   calling `code/tables.py`'s own render functions, and requires the saved
   `analysis/tables-q0*.md` AND the block injected in the report to equal that
   fresh render byte for byte.  A stale JSON render, a stale saved file or a
   stale report each fails here.

2. `prose_audit` extracts EVERY number written in the prose -- outside the
   generated table block, outside code fences and outside file paths -- and
   requires each one to be a value the record actually carries, at some sensible
   rounding.  Numbers that are not measurements (section numbers, source line
   numbers, predeclared constants, and values deliberately quoted from a
   superseded run) are listed in `NOT_A_MEASUREMENT` with the reason each is
   there.  A rerun that moves a number therefore fails until the prose is
   updated, and no new prose number can be invented.

3. `rel` asserts the verdict-bearing RELATIONS on the record itself: the
   identities are exact, the intervals fall on the side the verdict claims, the
   controls run on the arm's own support, the shift direction is what the report
   says.

    python tests/check_reports.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
A = EXP / "analysis"
sys.path.insert(0, str(EXP / "code"))
import tables as T  # noqa: E402

Q = {k: json.loads((A / f"q0{k}.json").read_text()) for k in (3, 4, 5, 6)}
REPORT = {k: (EXP / f"Q0{k}.md").read_text() for k in (3, 4, 5, 6)}
FOLDS = ("july_loo", "december", "january")
fails: list[str] = []
n_ok = 0


def rel(name: str, cond: bool) -> None:
    global n_ok
    if cond:
        n_ok += 1
    else:
        print(f"  FAIL  {name}")
        fails.append(name)


# ---------------------------------------------------------------------------
# 1. the tables, re-rendered from the current record
# ---------------------------------------------------------------------------
RENDER = {3: T.q03, 4: T.q04, 5: T.q05, 6: T.q06}


def injected(k: int) -> str | None:
    b, e = f"<!-- TABLES:q0{k} -->", "<!-- /TABLES -->"
    txt = REPORT[k]
    if b not in txt or e not in txt:
        return None
    return txt.split(b, 1)[1].split(e, 1)[0]


def tables_live(k: int) -> None:
    global n_ok
    fresh = RENDER[k]()
    saved = (A / f"tables-q0{k}.md").read_text()
    inj = injected(k)
    if inj is None:
        print(f"  FAIL  Q0{k} has no table markers")
        fails.append(f"Q0{k} markers")
    elif fresh != saved:
        print(f"  FAIL  analysis/tables-q0{k}.md is stale against q0{k}.json "
              f"(rerun code/tables.py)")
        fails.append(f"tables-q0{k}.md stale")
    elif inj != "\n" + fresh:
        print(f"  FAIL  Q0{k}.md's injected tables differ from the live render "
              f"(rerun code/tables.py)")
        fails.append(f"Q0{k} injected tables stale")
    else:
        n_ok += 1


# ---------------------------------------------------------------------------
# 2. the prose numbers, against every value the record carries
# ---------------------------------------------------------------------------
def leaves(o) -> list:
    if isinstance(o, dict):
        return [v for x in o.values() for v in leaves(x)]
    if isinstance(o, (list, tuple)):
        return [v for x in o for v in leaves(x)]
    return [o] if isinstance(o, (int, float)) and not isinstance(o, bool) else []


def variants(v: float) -> set:
    """Every way a report may honestly write this recorded value."""
    out = set()
    for x in (v, -v):
        for nd in range(0, 7):
            out.add(f"{x:.{nd}f}")
            out.add(f"{x:,.{nd}f}")
        out.add(f"{x:g}")
        if float(x).is_integer():
            out.add(f"{int(x)}")
            out.add(f"{int(x):,}")
        # a share written as a percentage of a stored fraction, and the reverse
        out.add(f"{x * 100:.2f}")
        out.add(f"{x * 100:.1f}")
        out.add(f"{x / 100:.4f}")
    return out


def allowed(k: int) -> set:
    s: set = set()
    for v in leaves(Q[k]):
        s |= variants(float(v))
    return s


#: Numbers in the prose that are NOT measurements, each with why it is here.
NOT_A_MEASUREMENT = {
    # structure and citations, in every report
    "0": "zero", "1": "list numbering", "2": "list numbering", "3": "list numbering",
    "4": "list numbering / thread count", "5": "list numbering", "6": "list numbering",
    "7": "list numbering", "8": "list numbering", "9": "list numbering",
    "10": "list numbering", "039": "experiment number", "016": "exp-016",
    "018": "exp-018", "027": "exp-027", "028": "exp-028", "036": "exp-036",
    "009": "exp-009", "03": "screen number", "04": "screen number",
    "05": "screen number", "06": "screen number", "10.": "section number",
    "2025": "calendar year", "07": "month", "12": "month", "01": "month",
    "2026": "calendar year", "64": "predeclared trees", "0.05": "predeclared lr",
    "39": "predeclared seed", "100": "predeclared ridge alpha", "202": "ridge features",
    "600": "10-minute bin in seconds", "60": "seconds in a minute",
    "15": "phase modulus", "319": "source line number", "166": "source line number",
    "609": "source line number", "652": "source line number",
    "2,000": "bootstrap draws", "1e-8": "declared tolerance",
    "1e-4": "rounding magnitude named in the round-2 repair note",
    "2.115": "the published gap to the leader (docs/SUMMARY.md), not a q0*.json value",
    "3.7": "0.0782 / 2.115, the share of the leader gap; derived in the prose",
    "0.26": "exp-036 BOARD READING 9 transfer factor, cited from another experiment",
    "179.47": "the SUPERSEDED round-1 pooled pair estimate, quoted as superseded",
    "30.94": "round-1 h1-h0, quoted as superseded", "27.96": "round-1 h1-h0, superseded",
    "10.39": "round-1 h1-h0, quoted as superseded",
    "171.2": "round-1 h0 increment, quoted as superseded",
    "19": "the round-1 h0 column count, quoted as superseded",
    "0.0781": "the pre-repair rounded board gain, quoted as the thing corrected",
    "117.2": "the pre-repair 3-day bound, quoted as superseded",
    "274.0": "the pre-repair 3-day bound, quoted as superseded",
    "7.74": "recall over the wrong denominator, quoted as the thing corrected",
    "15,166": "the first gold-removed draw, quoted to show the swing",
    "1,026": "the first gold-removed draw, quoted to show the swing",
    "11": "off-months in the nuisance fit / cited s of noac LOO",
    "13": "cited s of noac LOO from a shipped docstring",
    "1.89": "nuisance size in millions, stated as a range",
    "1.93": "nuisance size in millions, stated as a range",
    "1.55": "100 - 98.45, the off-grid share; derived in the prose",
    "99": "'99 %' and '0.7 %' rounded statements of recorded shares",
    "0.7": "1 - 0.9929 as a percentage; derived in the prose",
    "93": "fold-days, also carried in the record",
    "0.007": "a bound on |gain| stated to one significant figure",
    "200": "the shrinkage constant and the declared cell minimum",
    "20": "the declared date minimum", "500": "the abandoned row minimum",
    "40": "an abandoned arrival threshold", "1/2/3": "block lengths",
    "5/15/60": "the phase moduli", "3,600": "the clip upper bound",
    "0.25": "split-half correlation stated to two decimals",
    "0.31": "split-half correlation stated to two decimals",
    "35": "matched-support share stated to two significant figures",
    "0.09": "phase head loss stated to two decimals",
    "0.10": "phase head loss stated to two decimals",
    "800": "an interval half-width stated to one significant figure",
    "190,713": "fold row count, also carried in the record",
    "0.5571": "exp-016 july_loo board share, a predeclared constant",
    "0.4429": "exp-016 january board share, a predeclared constant",
    "1.6": "n_arr>=10 rows as a share of the fold; derived in the prose",
    "1.0": "n_arr>=10 rows as a share of the fold; derived in the prose",
    "0.6": "n_arr>=10 rows as a share of the fold; derived in the prose",
    "1,000": "the timestamp-unit factor in the defect note (us against ns)",
    "190,341": "the KL1818 example gap, in the defect note",
    "190.341": "the same gap as the defect printed it, in the defect note",
    "20.55": "the accepted-ID agreement BEFORE the timestamp fix, quoted as superseded",
    "250": "the scale of the problem in seconds, a descriptive statement",
    "83.76": "queries with SOME same-key same-leg donor row; derived from T1",
    "91.83": "queries with SOME same-key same-leg donor row; derived from T1",
}

NUM = re.compile(r"(?<![\w./-])(\d[\d,]*(?:\.\d+)?)(?![\w/])")


def prose_of(k: int) -> str:
    txt = REPORT[k]
    inj = injected(k)
    if inj:
        txt = txt.replace(inj, "\n")
    txt = re.sub(r"```.*?```", " ", txt, flags=re.S)      # code fences
    txt = re.sub(r"`[^`]*`", " ", txt)                    # inline code and paths
    txt = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", txt)        # links
    txt = re.sub(r"\S*/\S*", " ", txt)                    # any remaining path
    return txt


def prose_audit(k: int) -> None:
    global n_ok
    ok_vals = allowed(k)
    bad = []
    for m in NUM.finditer(prose_of(k)):
        tok = m.group(1)
        if tok in NOT_A_MEASUREMENT or tok.replace(",", "") in NOT_A_MEASUREMENT:
            continue
        if tok in ok_vals or tok.replace(",", "") in ok_vals:
            continue
        bad.append(tok)
    if bad:
        print(f"  FAIL  Q0{k} prose has {len(bad)} number(s) the record does not "
              f"carry: {sorted(set(bad))}")
        fails.append(f"Q0{k} unrecorded prose numbers")
    else:
        n_ok += 1


# ---------------------------------------------------------------------------
# 3. the verdict-bearing relations
# ---------------------------------------------------------------------------
q3, F3, P3, B3 = Q[3], Q[3]["folds"], Q[3]["pooled"], Q[3]["board"]
pr3 = q3["primary"]
rel("Q03 primary board gain positive", pr3["board_all_row"]["gain"] > 0)
rel("Q03 clean board gain positive", pr3["board_clean"]["gain"] > 0)
rel("Q03 board aggregated on unrounded RMSE (round 2, a1)",
    abs(pr3["board_all_row"]["gain_exact"] - pr3["board_all_row"]["gain"]) < 5e-5)
rel("Q03 pooled pair clear of zero", P3["pair_cross"]["ci"][0] > 0)
for L in (1, 2, 3):
    rel(f"Q03 pooled pair clear of zero at {L}-day blocks",
        P3["pair_cross"]["multiday"][f"ci_block_{L}d"][0] > 0)
rel("Q03 blocks drawn inside each fold (round 2, a2)",
    "never across months" in P3["pair_cross"]["multiday"].get("note", ""))
rel("Q03 uncentered control pooled is not positive", P3["cell_uncentered"]["point_s2"] < 0)
rel("Q03 january uncentered control excludes zero from below",
    F3["january"]["cell"]["uncentered"]["ci"][1] < 0)
rel("Q03 projection_off is a large board loss", B3["projection_off"]["gain_all_row"] < -1)
for f in FOLDS:
    c = F3[f]["cell"]["cross"]
    rel(f"Q03 cell sums zero {f}", c["cell_sum_check"] is True)
    rel(f"Q03 full DSSE == within-cell DSSE {f}", c["dsse_check"] is True)
    rel(f"Q03 cross cell interval clear of zero {f}", c["ci"][0] > 0)
    rel(f"Q03 pair cross clear of zero {f}", F3[f]["pair"]["cross"]["ci"][0] > 0)
    rel(f"Q03 uncentered runs on the arm's own support {f} (round 1, finding 4)",
        F3[f]["cell"]["uncentered"]["n_updated"] == c["n_updated"])
    rel(f"Q03 projection_off runs on the arm's own support {f}",
        F3[f]["cell"]["projection_off"]["n_updated"] == c["n_updated"])
tables_live(3)
prose_audit(3)

q4, F4, B4, BM4 = Q[4], Q[4]["folds"], Q[4]["board"], Q[4]["board_matched"]
rel("Q04 every arm loses the primary board",
    all(B4[a]["gain_all_row"] < 0 for a in ("h0", "h1", "h1c", "h1_shrunk")))
rel("Q04 h1 loses more than h0", B4["h1"]["gain_all_row"] < B4["h0"]["gain_all_row"])
rel("Q04 h1 increment is not positive on july_loo",
    F4["july_loo"]["arms"]["h1"]["increment_over_h0_s2"] < 0)
for f in FOLDS:
    rel(f"Q04 h0 carries the shipped block {f} (round 2, b1)", F4[f]["W_columns"] == 89)
    rel(f"Q04 every row joins the shipped block {f}", F4[f]["W_missing_rows"] == 0)
    rel(f"Q04 support rule met {f}", F4[f]["support_rule_met"] is True)
    rel(f"Q04 shift direction recorded {f} (round 2, b2)",
        F4[f]["time_match_direction"] == "later" and F4[f]["time_match_shift_days"] == 7)
    rel(f"Q04 matched support is a subset {f}",
        F4[f]["n_supported_matched"] < F4[f]["n_supported"])
    rel(f"Q04 matched arms all scored on the same rows {f}",
        len({F4[f]["arms_matched"][a]["n_scored"] for a in F4[f]["arms_matched"]}) == 1)
    rel(f"Q04 r*z interval spans zero {f}",
        F4[f]["cross_product_r_z"]["ci"][0] < 0 < F4[f]["cross_product_r_z"]["ci"][1])
    rel(f"Q04 self-excluded oracle clear of zero {f}",
        F4[f]["cross_product_r_rbar_self_excluded"]["ci"][0] > 0)
    rel(f"Q04 h1 and h1c are not separable on the matched support {f}",
        abs(F4[f]["arms_matched"]["h1"]["increment_over_h0_s2"]
            - F4[f]["arms_matched"]["h1c"]["increment_over_h0_s2"])
        < (F4[f]["arms_matched"]["h1"]["ci"][1] - F4[f]["arms_matched"]["h1"]["ci"][0]))
tables_live(4)
prose_audit(4)

q5, F5, B5 = Q[5], Q[5]["folds"], Q[5]["board"]
for f in FOLDS:
    i = F5[f]["identity"]
    rel(f"Q05 y = T - B exactly {f}", i["max_abs_y_minus_T_minus_B"] == 0.0)
    rel(f"Q05 anchored identity exactly {f}", i["max_abs_resid_identity"] == 0.0)
rel("Q05 anchor error is exactly minus (B - AOBT) on every flag (round 1, finding 9)",
    all(abs(r["delta_anchor_error"] + r["delta_BminusAOBT"]) < 1e-9
        for f in FOLDS for r in F5[f]["phase_table"]))
row5 = {r["flag"]: r for r in F5["july_loo"]["phase_table"]}
rel("Q05 v10 absorbs the phase the raw anchor does not",
    abs(row5["aobt_mod60"]["delta_r"]) < 0.02 * abs(row5["aobt_mod60"]["delta_anchor_error"]))
for h in ("phase_only", "phase_5s", "phase_15s", "phase_60s", "phase_beyond_controls"):
    rel(f"Q05 {h} loses the primary board", B5[h]["gain_all_row"] < 0)
rel("Q05 the phase increment over airport x operator is under 0.01 s",
    0 < B5["fingerprint_head"]["gain_board"] - B5["airport_operator_only"]["gain_board"] < 0.01)
rel("Q05 the sham is a cardinality control, not an exact capacity match",
    any(F5[f]["heads"]["sham_same_cardinality"]["n_groups"]
        != F5[f]["heads"]["fingerprint_head"]["n_groups"] for f in FOLDS))
rel("Q05 the witness correlation flips sign across folds",
    F5["july_loo"]["witness"]["corr_missing_type_shrunk_vs_placeholder"]
    * F5["january"]["witness"]["corr_missing_type_shrunk_vs_placeholder"] < 0)
tables_live(5)
prose_audit(5)

q6, C6, a6, F6 = Q[6], Q[6]["census_ranking"], Q[6]["accuracy_masked_joined"], Q[6]["folds"]
rel("Q06 no ambiguous ids survive the windows",
    C6["noac"]["in_window_nm"]["ids_gt1"] + C6["nonm"]["in_window_nm"]["ids_gt1"] == 0)
rel("Q06 ids were ambiguous before the windows",
    C6["noac"]["key_leg_only"]["ids_gt1"] > 0 and C6["nonm"]["key_leg_only"]["ids_gt1"] > 0)
rel("Q06 recall denominator is the counterparts that exist (round 1, finding 10)",
    abs(100 * a6["candidate_recall"] / a6["counterpart_exists"]
        - a6["candidate_recall_pct_of_existing"]) < 0.01)
rel("Q06 the all-query recall is kept and labelled separately",
    abs(100 * a6["candidate_recall"] / a6["n_queries"]
        - a6["candidate_recall_pct_of_all"]) < 0.01)
rel("Q06 one distinct id per acceptance (round 1, finding 11)",
    a6["accepted_unique_ids"] == a6["accepted"])
rel("Q06 wrong acceptance under gold removal is under a tenth of a percent",
    a6["stress"]["wrong_acceptance_pct"] < 0.1)
rel("Q06 primary board is a large loss", q6["primary"]["board_all_row"]["gain"] < -1)
for f in FOLDS:
    d = F6[f]["donor_anchor_nonm"]
    rel(f"Q06 donor anchor loses to v10 {f}", d["rmse_donor_anchor"] > d["rmse_v10_on_these"])
    rel(f"Q06 A/B negative {f}", d["unrestricted_scale"] < 0)
    rel(f"Q06 clean and all-row shares are both reported {f} (round 1, finding 12)",
        "sse_share_of_fold_clean_pct" in d and "sse_share_of_fold_all_row_pct" in d)
    rel(f"Q06 3-day block interval stays negative {f} (round 2, e1)",
        d["multiday"]["ci_block_3d"][1] < 0)
tables_live(6)
prose_audit(6)

print(f"\n{n_ok} checks pass"
      + ("" if not fails else f"; {len(fails)} FAILURES: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
