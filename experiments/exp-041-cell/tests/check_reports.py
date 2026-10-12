"""Every number RESULTS.md quotes, checked against the record it came from.

Three layers, following exp-039's repaired convention:

  1. the generated table block in RESULTS.md is byte-identical to
     `analysis/tables.md`, so a table cannot go stale without failing here;
  2. every interval bound and headline the PROSE quotes is asserted as a literal
     substring of the report, formatted from the record;
  3. the structural identities this experiment relies on -- the crossover's
     exact decomposition of the pooled statistic, and the fold-side v10 gate --
     are recomputed from the record;
  4. the REVERSE direction (Sol round 1, non-blocking): every numeric token in
     the prose must appear in one of the `analysis/*.json` records, or in the
     short declared list of numbers that come from OUTSIDE this experiment.
     Layer 2 proves the record reaches the report; only this one proves the
     report invents nothing.

    python tests/check_reports.py
"""
from __future__ import annotations

import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
A = EXP / "analysis"
REC = {n: json.loads((A / f"{n}.json").read_text())
       for n in ("anatomy", "order", "twofactor", "crossmonth", "slices",
                 "contrasts", "shared-partition")}
_RAW = (EXP / "RESULTS.md").read_text()


def norm(t: str) -> str:
    """Typography is not a number.  U+2212 becomes a hyphen, bold markers go,
    and every run of whitespace collapses, so a triple broken across a line
    still reads as one triple."""
    return " ".join(t.replace("\u2212", "-").replace("**", "").split())


REPORT = norm(_RAW)
FOLDS = ("july_loo", "december", "january")
fails: list[str] = []
n_ok = 0


def r0(v: float, nd: int = 0) -> str:
    """Half-UP rounding, the convention prose uses; `format` rounds half to even."""
    q = Decimal(1).scaleb(-nd)
    return f"{Decimal(repr(float(v))).quantize(q, rounding=ROUND_HALF_UP):,.{nd}f}"


def has(name: str, s: str) -> None:
    global n_ok
    if norm(s) in REPORT:
        n_ok += 1
    else:
        fails.append(f"{name}: report does not carry {s!r}")


def eq(name: str, got, want, tol: float = 1e-9) -> None:
    global n_ok
    ok = abs(float(got) - float(want)) <= tol
    if ok:
        n_ok += 1
    else:
        fails.append(f"{name}: {got} != {want}")


def pct(v: float, nd: int = 1) -> str:
    """A share written as a percentage.  `100 * 0.0865` is 8.649999... in binary,
    so the multiplication is done in decimal before rounding."""
    q = Decimal(1).scaleb(-nd)
    return f"{(Decimal(repr(float(v))) * 100).quantize(q, rounding=ROUND_HALF_UP):,.{nd}f}"


def triple(name: str, values, fmt: str, sep: str = " / ") -> None:
    if fmt == "pct":
        has(name, sep.join(pct(v) for v in values))
        return
    if fmt == ".3f":
        #  the tables are generated with `format`, so prose copied from a table
        #  must be checked the same way; `r0` is for hand-written prose
        has(name, sep.join(f"{v:.3f}" for v in values))
        return
    if fmt == ".2f":
        has(name, sep.join(r0(v, 2) for v in values))
        return
    if fmt == "+.4f":
        out = [("+" if v >= 0 else "-") + r0(abs(v), 4) for v in values]
        has(name, sep.join(out))
        return
    if fmt == "+,.1f":
        out = [("+" if v >= 0 else "-") + r0(abs(v), 1) for v in values]
        has(name, sep.join(out))
        return
    if fmt == "+.3f":
        out = [("+" if v >= 0 else "-") + r0(abs(v), 3) for v in values]
        has(name, sep.join(out))
        return
    if fmt == ".1f":
        has(name, sep.join(r0(v, 1) for v in values))
        return
    if fmt.endswith(",.0f"):
        out = [("+" if (fmt.startswith("+") and v >= 0) else "") + r0(v)
               for v in values]
    else:
        out = [format(v, fmt) for v in values]
    has(name, sep.join(out))


def pt_ci(name: str, d: dict, key: str = "delta_s2", nd: int = 1) -> None:
    has(name, f"{d[key]:+.{nd}f} [{d['ci'][0]:+.{nd}f}, {d['ci'][1]:+.{nd}f}]")


# ---- 1. the generated block --------------------------------------------------
gen = (A / "tables.md").read_text()
TBL_BEGIN, TBL_END = "<!-- TABLES:exp041 -->", "<!-- /TABLES -->"
b, e = TBL_BEGIN, TBL_END
if b in _RAW and e in _RAW:
    inj = _RAW.split(b, 1)[1].split(e, 1)[0]
    if inj.strip() == gen.strip():
        n_ok += 1
    else:
        fails.append("the injected table block is not analysis/tables.md")
else:
    fails.append("RESULTS.md has no table markers")   # raw text: byte identity

# ---- 2. the anatomy ----------------------------------------------------------
F = REC["anatomy"]["folds"]
vc = {k: [F[f]["variance_components_matched"][k] for f in FOLDS] for k in
      ("airport_window_s2", "runway_given_window_s2", "operator_given_window_s2",
       "additivity_residual_s2")}
triple("runway component (matched)", vc["runway_given_window_s2"], "+,.0f")
triple("window component (matched)", vc["airport_window_s2"], "+,.0f")
triple("operator component (matched)", vc["operator_given_window_s2"], "+,.0f")
#  1 dp: the reverse scan reads the record's own value (700.5), so the prose
#  carries the same digits rather than a rounding of them
triple("additivity residual (matched)", vc["additivity_residual_s2"], "+,.1f")
sh = F["july_loo"]["variance_components_matched"]["fractions_of_the_sum"]
has("runway share", f"{sh['runway_shared_pct']:.1f} %")
has("operator share", f"{sh['operator_shared_pct']:.1f} %")
sh2 = F["july_loo"]["variance_components"]["fractions_of_the_sum"]
has("runway share, all eligible", f"{sh2['runway_shared_pct']:.1f} %")
has("operator share, all eligible", f"{sh2['operator_shared_pct']:.1f} %")
triple("matched row counts",
       [F[f]["crossover_population"]["rows_with_all_four_relations_non_empty"]
        for f in FOLDS], ",.0f")
has("matched pct", " / ".join(
    str(F[f]["crossover_population"]["matched_pct_of_eligible"]) for f in FOLDS))
triple("runway-grain shared",
       [F[f]["runway_grain"]["shared"]["cell_ge_2"]["value"] for f in FOLDS], "+,.0f")
triple("plug-in variance",
       [F[f]["plugin_cell_mean"]["cell_ge_2"]["plugin_var_s2"] for f in FOLDS], "+,.0f")
triple("plug-in noise",
       [F[f]["plugin_cell_mean"]["cell_ge_2"]["noise_term_s2"] for f in FOLDS], "+,.1f")
triple("plug-in debiased",
       [F[f]["plugin_cell_mean"]["cell_ge_2"]["debiased_s2"] for f in FOLDS], "+,.0f")
triple("cross-fitted R2 of the cell mean",
       [F[f]["observables"]["cross_fitted_r2_cell_mean"] for f in FOLDS], ".3f")
triple("cross-fitted R2 of the shared part",
       [F[f]["observables"]["cross_fitted_r2_shared_part"] for f in FOLDS], ".3f")
triple("q04-support replication",
       [F[f]["headline"]["q04_support_dep6_arr6"]["value"] for f in FOLDS], "+,.1f")
triple("observables at runway grain",
       [F[f]["runway_grain"]["observables"]["mse_gain_s2"] for f in FOLDS], "+,.0f")
triple("observables at airport grain",
       [F[f]["observables"]["mse_gain_s2"] for f in FOLDS], "+,.0f")
for tag, get in (("airport", lambda f: F[f]["observables"]),
                 ("runway", lambda f: F[f]["runway_grain"]["observables"])):
    triple(f"fraction explained, {tag} grain",
           [get(f)["fraction_of_shared_explained"] for f in FOLDS], "pct")
for rel in ("same_rwy_diff_op", "diff_rwy_diff_op"):
    d = F["july_loo"]["crossover_matched"][rel]
    has(f"{rel} july interval (matched)",
        f"+{d['value']:,.1f} [+{d['ci'][0]:,.1f}, +{d['ci'][1]:,.1f}]")
#  A REAL identity (Sol round 2, non-blocking): the four relations partition every
#  other row of a query's cell, so their pair-count-weighted mean must equal the
#  pooled self-excluded cross-product read on the SAME rows.  The previous version
#  of this check compared one arrangement of a sum with another arrangement of the
#  same sum and could not fail.
for _f in FOLDS:
    _cx = F[_f]["crossover_matched"]
    _num = sum(_cx[k]["pairs"] * _cx[k]["value"] for k in _cx)
    _den = sum(_cx[k]["pairs"] for k in _cx)
    eq(f"crossover partitions the pooled component ({_f})", _num / _den,
       F[_f]["variance_components_matched"]["shared_component_on_these_rows_s2"],
       tol=0.05)
has("pooled july shared", f"{F['july_loo']['headline']['cell_ge_2']['value']:,.1f}")
#  T3c: the partition that answers the assignment's own sentence.  The four
#  pair-weighted contributions must add to the pooled cross-product exactly, and
#  the report must carry the runway share the fallback round asked for.
_P = REC["shared-partition"]["folds"]
for _f in FOLDS:
    eq(f"shared partition adds up ({_f})",
       sum(_P[_f][k] for k in ("window", "runway", "operator", "additivity")),
       _P[_f]["total"], tol=1e-6)
    eq(f"shared partition total is the cross-product ({_f})", _P[_f]["total"],
       F[_f]["variance_components_matched"]["shared_component_on_these_rows_s2"],
       tol=0.05)
for _k in ("runway", "operator", "window"):
    has(f"shared partition {_k} triple",
        " / ".join(f"{_P[f][f'{_k}_pct']:.1f}" for f in FOLDS))
for a in ("EGLL", "LEBL"):
    if any(a in F[f]["crossover_airports_supported"] for f in FOLDS):
        fails.append(f"{a} is reported unsupported but the record supports it")
    else:
        n_ok += 1
#  the autocorrelation is now a correlation of cell means on matched cell pairs,
#  reported on the plug-in mean and with the within-cell noise removed
for L, tag in ((1, "one bin"), (2, "two bins"), (6, "six bins")):
    for key in ("rho_plugin", "rho_debiased"):
        triple(f"acf lag {L} {key}, airport grain",
               [F[f]["autocorrelation"][f"lag_{L}"][key] for f in FOLDS], ".3f")
#  the queue's predeclared Q10 rule: zero cells at the 10-minute bin
for f in FOLDS:
    eq(f"q10 rule supports no cell at 10 min ({f})",
       F[f]["q10_rule_grain_scan"]["10min"]["cells_supported"], 0)
triple("q10 rule, runway component at 30 min",
       [F[f]["q10_rule_grain_scan"]["30min"]["components"]["runway_given_window_s2"]
        for f in FOLDS], "+,.0f")
triple("nm_late by regime", [F[f]["by_regime"]["nm_late"]["value"] for f in FOLDS],
       "+,.0f")
triple("nonm by regime", [F[f]["by_regime"]["nonm"]["value"] for f in FOLDS], "+,.0f")

# ---- 3. step 2 ---------------------------------------------------------------
O = REC["order"]["folds"]
for f in FOLDS:                       # cap B: point + interval per fold
    pt_ci(f"ord - unord {f} B", O[f]["arms"]["ord@B"]["vs"]["unord"])
for f in ("july_loo", "january"):
    pt_ci(f"ord - unord2 {f} B", O[f]["arms"]["ord@B"]["vs"]["unord2"])
triple("ord - h0, cap B",
       [O[f]["arms"]["ord@B"]["vs"]["h0"]["delta_s2"] for f in FOLDS], "+.1f")
triple("ord - ord_c, cap B",
       [O[f]["arms"]["ord@B"]["vs"]["ord_c"]["delta_s2"] for f in FOLDS], "+.1f")
for f in FOLDS:                       # cap A: ord adds exactly nothing
    eq(f"ord - h0 is zero at cap A on {f}",
       O[f]["arms"]["ord@A"]["vs"]["h0"]["delta_s2"], 0.0)
has("ord - unord2 july 3-day",
    "[+{:.1f}, +{:.1f}]".format(
        *O["july_loo"]["arms"]["ord@B"]["vs"]["unord2"]["multiday"]["ci_block_3d"]))
has("h0 board 3-day",
    "[+{:.4f}, +{:.4f}]".format(
        *REC["order"]["boards"]["h0"]["board_interval"]["ci_block_3d"]))
has("ord board 3-day",
    "[+{:.4f}, +{:.4f}]".format(
        *REC["order"]["boards"]["ord"]["board_interval"]["ci_block_3d"]))
for a in ("h0", "ord", "unord", "unord2", "ord_c"):
    has(f"{a} board", f"{REC['order']['boards'][a]['board_all_row']['gain']:+.4f}")
has("median gap 1", f"{O['july_loo']['gap1_median_s']:.0f} s")
has("order differs pct", f"{min(O[f]['order_differs_pct'] for f in FOLDS):.1f}")
has("order differs pct max", f"{max(O[f]['order_differs_pct'] for f in FOLDS):.1f}")
has("h0 board", f"+{REC['order']['boards']['h0']['board_all_row']['gain']:.4f} s")

has("base features", str(O["july_loo"]["n_base_features"]))
has("crossmonth base features", str(REC["crossmonth"]["n_base_features"]))
triple("observables features", [REC["anatomy"]["folds"]["july_loo"]
                                ["observables"]["n_features"]], ",.0f")

# ---- 4. step 3 ---------------------------------------------------------------
T = REC["twofactor"]["folds"]
for a2 in ("two_c", "g0", "one"):
    for f in FOLDS:
        pt_ci(f"two - {a2} {f} A", T[f]["arms"][f"two@A"]["vs"][a2])
for f in FOLDS:
    pt_ci(f"two - g0 {f} B", T[f]["arms"]["two@B"]["vs"]["g0"])
triple("two - one, cap B",
       [T[f]["arms"]["two@B"]["vs"]["one"]["delta_s2"] for f in FOLDS], "+.1f")
pt_ci("two_c - g0 december A", T["december"]["arms"]["two_c@A"]["vs"]["g0"])
for a in ("g0", "one", "two", "two_c"):
    has(f"twofactor {a} board",
        f"{REC['twofactor']['boards'][a]['board_all_row']['gain']:+.4f}")
#  Exact counts, not millions: a rounded token cannot be verified by the
#  reverse scan, which is where an unrecorded number hides.
has("nuisance departures low",
    f"{min(T[f]['nuisance_train_departures'] for f in FOLDS):,d}")
has("nuisance departures high",
    f"{max(T[f]['nuisance_train_departures'] for f in FOLDS):,d}")
for a in ("q04h1", "one_q04", "two_q04"):
    has(f"twofactor {a} board",
        f"{REC['twofactor']['boards'][a]['board_all_row']['gain']:+.4f}")
triple("two_q04 - one_q04",
       [T[f]["arms"]["two_q04@B"]["vs"]["one_q04"]["delta_s2"] for f in FOLDS], "+.1f")
triple("q04h1 - g0",
       [T[f]["arms"]["q04h1@B"]["vs"]["g0"]["delta_s2"] for f in FOLDS], "+.1f")
for f in FOLDS:
    pt_ci(f"two - two_c {f} B", T[f]["arms"]["two@B"]["vs"]["two_c"])
triple("same-rwy feature coverage",
       [T[f]["feature_coverage_pct"]["f_same_rwy_diff_op"] for f in FOLDS], ".1f")
triple("u sd", [T[f]["u_std_s"] for f in FOLDS], ".0f")
triple("same-op feature coverage",
       [T[f]["feature_coverage_pct"]["f_diff_rwy_same_op"] for f in FOLDS], ".1f")
for f in FOLDS:
    if T[f]["unsupported_airports"] != ["EGLL", "LEBL"]:
        fails.append(f"{f}: unsupported airports are {T[f]['unsupported_airports']}")
    else:
        n_ok += 1

# ---- 5. the cross-month control ---------------------------------------------
C = REC["crossmonth"]
for n in ("base_within", "base_cross", "base_cross_amp"):
    has(f"{n} board", f"{C['boards'][n]['board_all_row']['gain']:+.4f} s")
triple("base_cross per fold",
       [C["arms"]["base_cross"][f]["mse_gain_s2"] for f in FOLDS], "+,.1f")
triple("base_within per fold",
       [C["arms"]["base_within"][f]["mse_gain_s2"] for f in FOLDS], "+,.1f")
triple("base cross-month amplitudes",
       [C["amplitudes"][f"base:{f}"] for f in FOLDS], "+.4f")
D = C["diagnostics"]
triple("corr(r, h_cross)", [D[f"base:{f}"]["corr_r_cross"] for f in FOLDS], "+.3f")
triple("corr(r, h_within)", [D[f"base:{f}"]["corr_r_within"] for f in FOLDS], "+.3f")
triple("corr(h_within, h_cross)",
       [D[f"base:{f}"]["corr_within_cross"] for f in FOLDS], "+.3f")
triple("in-sample 2A-B", [D[f"base:{f}"]["insample_2A_B_s2"] for f in FOLDS], "+,.0f")

# ---- 6. the four-way RMSE and the zero-update identity -----------------------
S = REC["slices"]["folds"]
triple("no-NM RMSE", [S[f]["v10"]["no_nm"] for f in FOLDS], ".4f")
for f in FOLDS:
    for a in ("twofactor-g0", "twofactor-two", "crossmonth-base_within",
              "crossmonth-base_cross"):
        eq(f"no-NM unchanged {a} {f}", S[f][a]["no_nm"], S[f]["v10"]["no_nm"],
           tol=1e-4)
triple("gt3600 counts", [S[f]["counts"]["gt3600"] for f in FOLDS], ",.0f")

#  the paired contrasts on the primary endpoint: the tests the advance clauses name
CN = REC["contrasts"]["pairs"]
for _a, _b in (("order-ord", "order-unord"), ("order-ord", "order-unord2"),
               ("twofactor-two", "twofactor-one"),
               ("twofactor-two_q04", "twofactor-one_q04"),
               ("order-ord", "order-h0"), ("order-ord", "order-ord_c")):
    k = f"{_a} - {_b}"
    v = CN[k]
    has(f"paired contrast {k}", f"{v['gain_a_minus_b']:+.4f}")
    lo, hi = v["ci_block_3d"]
    has(f"paired contrast {k} 3-day", f"[{lo:+.4f}, {hi:+.4f}]")
    if v["clear_of_zero_3d"]:
        fails.append(f"{k}: the 3-day interval is clear of zero; the prose calls "
                     f"every step-2/step-3 first half null")
    else:
        n_ok += 1
#  the leak-free clean family: no base a scored fold uses may be that fold's month
CF = REC["crossmonth"]["clean_family"]
for _arm in ("clean_within", "clean_cross"):
    for _f in FOLDS:
        if CF["arms"][_arm][_f]["base_trained_on"] == _f:
            fails.append(f"clean family {_arm}/{_f}: the base is trained on the "
                         f"scored fold itself")
        else:
            n_ok += 1
for arm in ("clean_within", "clean_cross"):
    has(f"clean family {arm} board",
        f"{CF['boards'][arm]['gain']:+.4f}")
eq("clean family transfer ratio",
   round(CF["boards"]["clean_cross"]["gain"] / CF["boards"]["clean_within"]["gain"], 2),
   0.68)

# ---- layer 4: the reverse scan -----------------------------------------------
#: Numbers the prose legitimately carries that no exp-041 record holds, each with
#: the record outside this experiment that does.  Anything not here and not in a
#: record is the report inventing a number.
EXTERNAL = {
    "1,182": "docs/pro-reasoning-2026-09-08/score-budget.json, quoted by assignments/cell.md",
    "1,515": "1,182 divided by the eligible cohort's share of nm_bulk",
    "2.1154": "the gap to the leader, BOARD READING 10",
    "257.0248": "v10's published clean board (exp-036 arm 2)",
    "218.1506": "v10's published january clean RMSE (exp-036 arm 2)",
    "6,323": "assignments/cell.md's quoted shared component",
    "4,526": "assignments/cell.md's quoted shared component",
    "4,876": "assignments/cell.md's quoted shared component",
    "0.078": "exp-039 Q03's all-row board gain",
    "0.0782": "exp-039 Q03's all-row board gain",
    "202": "exp-039's row feature set, rowfeat.features",
    "0.55713": "exp-016's july ranking share",
    "0.44287": "exp-016's january ranking share",
    "2,000": "the bootstrap draw count, packet.BOOT",
}
_IGNORE = {"2025", "2026", "600", "3600", "7200", "041", "039", "036", "038",
           "044", "018", "027", "028", "016", "011", "010", "003", "004", "005",
           "1e-9", "256"}


def _record_numbers() -> set:
    seen = set()

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            v = float(x)
            for nd in range(0, 5):
                seen.add(f"{abs(v):,.{nd}f}")
                seen.add(f"{abs(v):.{nd}f}")
                seen.add(f"{abs(100 * v):,.{nd}f}")
                seen.add(f"{abs(100 * v):.{nd}f}")
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(k); walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            for tok in re.findall(r"\d[\d,]*(?:\.\d+)?", x):
                seen.add(tok)
    for rec in REC.values():
        walk(rec)
    return seen


def reverse_scan() -> None:
    """Every numeric token the PROSE carries must be in a record or declared."""
    global n_ok
    known = _record_numbers()
    text = _RAW
    if TBL_BEGIN not in text or TBL_END not in text:
        fails.append("reverse scan: the table markers are missing from RESULTS.md")
        return
    text = text.split(TBL_BEGIN, 1)[0] + text.split(TBL_END, 1)[1]   # prose only
    #  Code spans are stripped LINE BY LINE.  A document-wide `[^`]*` pairs an
    #  unbalanced backtick with one many lines away and silently swallows whole
    #  sections -- which is what happened here: 327 backticks in the prose, one
    #  of them a typo, and the scan was reading a fraction of the report.
    bad = []
    if text.count("`") % 2:
        fails.append("reverse scan: the prose has an odd number of backticks, so "
                     "a code span is unclosed")
    for line in text.splitlines():
        if line.startswith("|") or line.startswith("    "):
            continue
        line = re.sub(r"`[^`\n]*`", " ", line)
        for tok in re.findall(r"(?<![A-Za-z0-9./+-])[\u2212+-]?\d[\d,]*(?:\.\d+)?",
                              line):
            t = tok.lstrip("+-\u2212")
            if t in EXTERNAL or t in known or t in _IGNORE:
                continue
            plain = t.replace(",", "")
            try:
                x = float(plain)
            except ValueError:
                continue
            if "." not in plain and x <= 16:        # counts and list numbers
                continue
            #  Re-render at the TOKEN's own precision only.  The old form
            #  rounded every token to 0 dp as a fallback, and "0" is in every
            #  record, so any decimal token passed: 0.0709 and 12.3456 both
            #  slipped through a scan that was supposed to catch them.
            nd = len(plain.split(".")[1]) if "." in plain else 0
            if f"{x:,.{nd}f}" in known or f"{x:.{nd}f}" in known:
                continue
            bad.append((t, line.strip()[:100]))
    if bad:
        for t, line in bad[:25]:
            fails.append(f"reverse scan: {t} is in no record :: {line}")
    else:
        n_ok += 1


reverse_scan()

# ---- 7. report ---------------------------------------------------------------
print(f"{n_ok} checks passed, {len(fails)} failed")
for x in fails:
    print("  FAIL " + x)
sys.exit(1 if fails else 0)
