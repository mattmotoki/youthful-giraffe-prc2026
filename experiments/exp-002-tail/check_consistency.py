"""Tie every number in climb.log, RESULTS.md and the exp-002 LEDGER rows to an artifact.

    PYTHONPATH=. .venv/bin/python experiments/exp-002-tail/check_consistency.py [--selftest]

An earlier version of this file printed ALL CONSISTENT while three quoted
numbers were wrong, because it matched values as unbounded substrings and only
spot-checked a few lines. It now parses every data row of the climb table, looks
each one up in the CSV that produced it, and re-derives every `gap closed` value.
`--selftest` perturbs one artifact value in memory and asserts the check fails,
so a green run means something.

What it still cannot do: re-run the model. A dependency changing under the
artifacts is caught by `deliver.py --verify` and by pinning
`prc.tail_model.PHI_WEIGHT`, not here.
"""
from __future__ import annotations

import re
import sys

import pandas as pd

import prc.data as D

EXP = D.ROOT / "experiments" / "exp-002-tail"
BOARD = 277.3
FOLDS = ("july", "july_loo", "december", "january")

#: climb.log line tag -> (metrics csv, model name in it)
CLIMB = {
    "it0":  ("metrics_all", "it0-naive"),
    "it1":  ("metrics_all", "it1-bulk-target-y"),
    "it2":  ("metrics_all", "it2-bulk-target-offset"),
    "it3":  ("metrics_all", "it3-plus-tail-signals"),
    "it4":  ("metrics_all", "it4-plus-stand-occupancy"),
    "it5a": ("metrics_all", "it5-twostage-expectation"),
    "it5b": ("metrics_all", "it5-twostage-hardswitch"),
    "it6a": ("metrics_all", "it6-huber-bulk"),
    "it6b": ("metrics_all", "it6-twostage-shrunk-tail"),
    "it7":  ("metrics_all", "it7-separate-nonm-model"),
    "it8":  ("metrics_all", "it8-noac:mixture-expectation"),
    "it9":  ("metrics_seg", "it9-seg(nonm:y|noac:mixture-expectation)"),
    "it10": ("metrics_seg2", "it10-seg(nonm:allrows-noflt|noac:mixture-expectation)"),
    "it11": ("metrics_final", "it11-with-day|noac:mixture-expectation"),
    "it12": ("metrics_weather", "it12-day+weather|noac:mixture-expectation"),
    "it13": ("metrics_phi", "it13-baseline (use_phi=False)"),
    "it16": ("metrics_delivered", "TailModel (delivered)"),
}
#: unnamed ("--") climb rows: the label text -> (csv, model). Every numeric row
#: in climb.log must appear either in CLIMB or here, or the check fails.
ABLATION = {
    "it13, noac head = direct offset (pre-pin, no live artifact)": None,
    "it13, noac head = HARD SWITCH  (pre-pin, no live artifact)": None,
    "it13 + noac oracle             (pre-pin, no live artifact)": None,
    "it11 with a noac oracle (ceiling instrument)": ("metrics_final", "it11-with-day|noac:ORACLE"),
    "it10 with the noac rows sent to the ordinary head": ("metrics_seg2", "it10-seg(nonm:allrows-noflt|noac:same-head)"),
    "it13 baseline with use_phi=False (the phi comparison's base)": ("metrics_phi", "it13-baseline (use_phi=False)"),
    "it14 phi as features (== it16, the delivered model)": ("metrics_phi", "it14-phi-as-features"),
    "it15 phi as the ANCHOR instead of a feature": ("metrics_phi", "it15-phi-as-anchor"),
    "it15b raw phi taxi_hat, no model at all": ("metrics_phi", "it15b-raw taxi_hat, no model"),
    "it16, noac head = HARD SWITCH": ("metrics_delivered", "TailModel, noac head = hard switch"),
    "it16 with a noac oracle (ceiling instrument)": ("metrics_delivered", "TailModel, noac oracle (ceiling)"),
}
#: the noac LOMO block: label -> predictor in noac_lomo.csv
LOMO_ROWS = {
    "constant mean": "const-mean",
    "MVT-SCHED band mean": "band-mean",
    "LightGBM on y directly": "lgb-direct-y",
    "mixture, HARD SWITCH on argmax p": "mixture-hardswitch",
    "mixture, RMSE-OPTIMAL EXPECTATION": "mixture-expectation",
    "LightGBM on the offset (implicit mixture), 5 seeds": "lgb-direct-offset-5seed",
    "oracle placeholder flag (ceiling)": None,
}
ROW = re.compile(r"^(it\d+[ab]?|--)\s*\|\s*(.*?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([-\d.]+|-)\s*$")
NOAC_ROW = re.compile(r"^noac-([a-g])\s*\|\s*(.*?)\s*\|\s*([\d.]+)\s*\|")

_fail = []
#: rows whose artifact was superseded and cannot be re-derived. The check asserts
#: this set exactly, so a NEW unverifiable row is a failure rather than a silent
#: exemption -- which is how the pre-pin it13 numbers slipped through once.
EXPECTED_UNVERIFIABLE = {
    "ablation:it13, noac head = direct offset (pre-pin, no live artifact)",
    "ablation:it13, noac head = HARD SWITCH  (pre-pin, no live artifact)",
    "ablation:it13 + noac oracle             (pre-pin, no live artifact)",
    "lomo:oracle placeholder flag (ceiling)",
}
UNVERIFIABLE = set()


def chk(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (("  " + detail) if detail else ""))
    if not cond:
        _fail.append(name)


def load(csv):
    return pd.read_csv(EXP / "work" / f"{csv}.csv")


def climb_values(perturb=None):
    """Every numeric climb.log row.

    Returns (named, ablations, lomo, n_numeric): named keyed by `itN`,
    ablations keyed by their label text, lomo keyed by its label text, and the
    count of numeric rows seen so nothing can be silently skipped.
    """
    named, abl, lomo, n = {}, {}, {}, 0
    for line in (EXP / "climb.log").read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        # count anything shaped like a data row FIRST -- by its leading tag, not
        # by its numbers -- so a row this file cannot parse fails the count
        # rather than being skipped silently
        if re.match(r"^(it\S*|--|noac-\S*)\s*\|", line):
            n += 1
        m = ROW.match(line)
        if m:
            g = m.group(6)
            vals = (float(m.group(3)), float(m.group(4)), float(m.group(5)),
                    None if g == "-" else float(g))
            if m.group(1) == "--":
                abl[m.group(2)] = vals
            else:
                named[m.group(1)] = vals
            continue
        mn = NOAC_ROW.match(line)
        if mn:
            tail_tok = line.rsplit("|", 1)[1].strip() if line.count("|") >= 2 else ""
            lomo[mn.group(2)] = (float(mn.group(3)), tail_tok)
    if perturb:
        kind, key, i, v = perturb
        d = {"named": named, "abl": abl}[kind]
        if kind == "abl":
            cur = list(d[key]); cur[i] = v; d[key] = tuple(cur)
        else:
            cur = list(d[key]); cur[i] = v; d[key] = tuple(cur)
    return named, abl, lomo, n


def run(perturb=None, quiet=False):
    global _fail
    _fail = []
    _fail = []
    if quiet:
        globals()["chk"] = lambda n, c, d="": _fail.append(n) if not c else None
    res = (EXP / "RESULTS.md").read_text()
    led = (D.ROOT / "docs" / "LEDGER.md").read_text()
    UNVERIFIABLE.clear()
    named, abl, lomo, n_numeric = climb_values(perturb)

    def compare(label, got, csv, model, folds=("july", "december", "january")):
        try:
            d = load(csv)
        except FileNotFoundError:
            chk(f"{label}: source csv {csv}", False, "missing"); return
        sub = d[d.model == model].set_index("fold")
        for i, f in enumerate(folds):
            if f not in sub.index:
                chk(f"{label} {f}: present in {csv}", False); continue
            chk(f"{label} {f}", abs(got[i] - round(sub.loc[f, "rmse"], 1)) < 0.051,
                f"log {got[i]} vs artifact {sub.loc[f,'rmse']:.4f}")

    # 1. every NAMED climb row must exist and match
    for tag, (csv, model) in CLIMB.items():
        if tag not in named:
            chk(f"climb {tag}: row present in climb.log", False); continue
        compare(f"climb {tag}", named[tag], csv, model)
    for tag in named:
        chk(f"climb {tag}: known row", tag in CLIMB, "unmapped tag in climb.log")

    # 2. every UNNAMED ablation row must be mapped, and match where it has an artifact
    for label, vals in abl.items():
        if label not in ABLATION:
            chk(f"ablation '{label[:40]}': mapped", False, "add it to ABLATION"); continue
        src = ABLATION[label]
        if src is None:
            UNVERIFIABLE.add(f"ablation:{label}")   # no live artifact; see climb.log
            continue
        compare(f"ablation '{label[:40]}'", vals, *src)
    for label, src in ABLATION.items():
        chk(f"ablation '{label[:40]}': row present", label in abl)

    # 3. the noac LOMO block
    lo = load("noac_lomo").set_index("predictor")
    for label in lomo:
        chk(f"lomo '{label[:36]}': mapped", label in LOMO_ROWS, "add it to LOMO_ROWS")
    for label, pred in LOMO_ROWS.items():
        if label not in lomo:
            chk(f"lomo '{label[:36]}': row present", False); continue
        rmse, tail_tok = lomo[label]
        if pred is None:
            UNVERIFIABLE.add(f"lomo:{label}"); continue
        chk(f"lomo '{label[:36]}' rmse", abs(rmse - lo.loc[pred, "rmse"]) < 0.051,
            f"log {rmse} vs artifact {lo.loc[pred,'rmse']:.1f}")
        # the trailing cell must be a parseable "<number> s"; an unparseable
        # token is a failure, not a reason to skip the comparison
        m2 = re.fullmatch(r"(?:ranking contribution )?([\d.]+) s", tail_tok)
        chk(f"lomo '{label[:36]}' contribution parses", m2 is not None, repr(tail_tok))
        if m2:
            chk(f"lomo '{label[:36]}' ranking contribution",
                abs(float(m2.group(1)) - round(lo.loc[pred, "ranking_contrib_s"], 1)) < 0.051,
                f"log {m2.group(1)} vs artifact {lo.loc[pred,'ranking_contrib_s']:.1f}")

    chk("unverifiable rows are exactly the expected set",
        UNVERIFIABLE == EXPECTED_UNVERIFIABLE,
        f"unexpected {sorted(UNVERIFIABLE - EXPECTED_UNVERIFIABLE)} missing {sorted(EXPECTED_UNVERIFIABLE - UNVERIFIABLE)}")
    chk("every numeric climb.log row was checked",
        n_numeric == len(named) + len(abl) + len(lomo), f"{n_numeric} numeric rows seen")

    # 4. gap-closed arithmetic, exact to the printed 2 decimals
    order = [t for t in ["it0", "it1", "it2", "it3", "it4", "it5a", "it5b", "it6a", "it6b",
                         "it7", "it8", "it9", "it10", "it11", "it12", "it13", "it16"] if t in named]
    best = None
    for tag in order:
        july, gap = named[tag][0], named[tag][3]
        if best is not None and gap is not None:
            want = round((best - july) / (best - BOARD), 2)
            chk(f"gap closed {tag}", gap == want, f"{gap} vs {want}")
        best = july if best is None else min(best, july)

    # 5. the delivered artifact
    p = EXP / "oof" / "exp-002-tail-oof.parquet"
    oof = pd.read_parquet(p)
    chk("OOF sha recorded in RESULTS", D.sha256(p) in res, D.sha256(p)[:12])
    chk("OOF unique on (id, fold), no nulls",
        not oof.duplicated(["MVT_ID_mvt", "fold"]).any() and oof.notna().all().all(),
        f"{len(oof)} rows, {oof.fold.nunique()} folds")
    d = load("metrics_delivered")
    d = d[d.model == "TailModel (delivered)"].set_index("fold")
    for f, g in oof.groupby("fold"):
        v = D.rmse(g.y_true, g.y_pred)
        chk(f"OOF RMSE == metrics_delivered ({f})", abs(v - d.loc[f, "rmse"]) < 1e-6, f"{v:.6f}")
        chk(f"RESULTS quotes {v:.2f} for {f}", re.search(rf"(?<![\d.]){v:.2f}(?![\d])", res) is not None)
    ph = load("metrics_phi")
    it14 = ph[ph.model == "it14-phi-as-features"].set_index("fold")
    for f in ("july", "december", "january"):
        chk(f"phi it14 == delivered ({f})", abs(it14.loc[f, "rmse"] - d.loc[f, "rmse"]) < 1e-6)

    # 6. LOMO / ranking / phi screen quoted in RESULTS, bounded matches
    for _, row in load("noac_lomo").iterrows():
        pat = rf"(?<![\d.]){row.rmse:,.1f}(?![\d])".replace(",", "[,]?")
        chk(f"LOMO {row.predictor} quoted in RESULTS", re.search(pat, res) is not None, f"{row.rmse:.1f}")
    chk("LOMO prediction sha recorded", D.sha256(EXP / "work" / "noac_lomo_pred.parquet") in res)
    rk = pd.read_parquet(EXP / "noac_ranking_predictions.parquet")
    chk("388 ranking noac rows, all >= 60 s", len(rk) == 388 and (rk.y_pred >= 60).all())
    chk("ranking prediction sha recorded", D.sha256(EXP / "noac_ranking_predictions.parquet") in res)
    scr = EXP / "work" / "phi_screen.txt"
    if scr.exists():
        for line in scr.read_text().splitlines():
            for n in re.findall(r"(\d{3}\.\d)", line)[1:]:
                chk(f"phi screen {n} quoted", re.search(rf"(?<![\d.]){n}(?![\d])", res) is not None)

    # 7. EVERY exp-002 ledger row with numeric score columns must match an artifact
    def numeric(line):
        """(july, december) as (value, decimals) so the check honours the printed precision."""
        c = [x.strip() for x in line.split("|")]
        try:
            out = []
            for cell in (c[5], c[6]):
                out.append((float(cell), len(cell.split(".")[1]) if "." in cell else 0))
            return out
        except (ValueError, IndexError):
            return None
    # every exp-002 ledger row carrying numeric score cells must be mapped, and
    # the mapped ones compared on ALL four score columns (july, dec, july
    # tail>3600, july no-NM). Superseded rows whose artifact is gone are listed
    # explicitly rather than skipped, and the set is asserted.
    # keyed on a distinguishing substring of the row, because two rows share the
    # tag "exp-002 it11": mine, and one the desk wrote for the regime model that
    # climb.log calls it13. See the correction row in docs/LEDGER.md.
    LEDGER_SRC = {
        "airport-day and airport-hour aggregates": ("metrics_final", "it11-with-day|noac:mixture-expectation"),
        "regime router: NM rows on an AOBT_3 offset": ("metrics_seg2", "it10-seg(nonm:allrows-noflt|noac:mixture-expectation)"),
        "ERA5 weather (prc.weather_features) — measured and NOT adopted": ("metrics_weather", "it12-day+weather|noac:mixture-expectation"),
        "adds the july_loo fold": ("metrics_delivered", "TailModel (delivered)"),
        "REVIEW CLOSE: nine fresh-context Sol reviews": ("metrics_delivered", "TailModel (delivered)"),
        # superseded: their artifacts were overwritten by later reruns
        "regime segmentation (NM on AOBT anchor": None,
        "noac sub-model, leave-one-month-out over all 1507 rows": None,
        "regime router + tuned noac mixture + MVT-ARVT_1/3 signals": None,
        "prc.tail_model.PHI_WEIGHT pinned": None,
        "lirf's phi as NM-head features": None,
    }
    #: rows whose score cells are LOMO quantities: text -> {col: (predictor, column)}
    LEDGER_LOMO = {
        "noac sub-model LOMO over all 1507 rows": {
            7: ("lgb-direct-offset-5seed", "rmse_gt3600"),
            8: ("lgb-direct-offset-5seed", "rmse"),
        },
    }
    LEDGER_SUPERSEDED = {k for k, v in LEDGER_SRC.items() if v is None and k in led}
    COLS = {"july": 5, "december": 6, "rmse_gt3600": 7, "rmse_nonm": 8}
    from collections import Counter
    seen, superseded_seen = set(), Counter()
    for line in led.splitlines():
        if "| tail |" not in line or "exp-002" not in line:
            continue
        cells = [x.strip() for x in line.split("|")]
        if not any(re.match(r"[\d.]+", cells[i]) for i in COLS.values() if i < len(cells)):
            continue                      # a correction/analysis row: no scores
        lomo_tag = next((k for k in LEDGER_LOMO if k in line), None)
        if lomo_tag is not None:
            lo2 = load("noac_lomo").set_index("predictor")
            for idx, (pred, col) in LEDGER_LOMO[lomo_tag].items():
                m3 = re.match(r"([\d.]+)", cells[idx] if idx < len(cells) else "")
                chk(f"ledger LOMO '{lomo_tag[:30]}' col{idx} numeric", m3 is not None,
                    repr(cells[idx] if idx < len(cells) else ""))
                if m3:
                    dp = len(m3.group(1).split(".")[1]) if "." in m3.group(1) else 0
                    chk(f"ledger LOMO '{lomo_tag[:30]}' col{idx}",
                        float(m3.group(1)) == round(lo2.loc[pred, col], dp),
                        f"{m3.group(1)} vs {round(lo2.loc[pred, col], dp)}")
            seen.add(lomo_tag)
            continue
        tag = next((k for k in sorted(LEDGER_SRC, key=len, reverse=True) if k in line), None)
        chk(f"ledger row '{line.split('|')[1].strip()} {line.split('|')[2].strip()}' mapped",
            tag is not None, "add it to LEDGER_SRC")
        if tag is None:
            continue
        if LEDGER_SRC[tag] is None:
            superseded_seen[tag] += 1; continue
        seen.add(tag)
        csv, model = LEDGER_SRC[tag]
        sub = load(csv); sub = sub[sub.model == model].set_index("fold")
        for name, idx in COLS.items():
            cell = cells[idx] if idx < len(cells) else ""
            chk(f"ledger {tag} {name} is numeric", re.fullmatch(r"[\d.]+", cell) is not None, repr(cell))
            if not re.fullmatch(r"[\d.]+", cell):
                continue
            got = float(cell); dp = len(cell.split(".")[1]) if "." in cell else 0
            fold = "july" if name != "december" else "december"
            col = "rmse" if name in ("july", "december") else name
            chk(f"ledger {tag} {name}", got == round(sub.loc[fold, col], dp),
                f"{got} vs {round(sub.loc[fold, col], dp)} (at {dp} dp)")
    chk("ledger superseded rows are exactly the expected set",
        set(superseded_seen) == LEDGER_SUPERSEDED,
        f"unexpected {sorted(set(superseded_seen) - LEDGER_SUPERSEDED)} missing {sorted(LEDGER_SUPERSEDED - set(superseded_seen))}")
    # a superseded mechanism describes ONE historical row; a second row reusing
    # that text is a new claim hiding behind an exemption
    for tag, count in superseded_seen.items():
        chk(f"ledger superseded '{tag[:38]}' appears once", count == 1, f"{count} rows")
    for tag, src in LEDGER_SRC.items():
        if src is not None:
            chk(f"ledger row {tag} present", tag in seen)
    return not _fail


if __name__ == "__main__":
    ok = run()
    print("\nALL CONSISTENT" if ok else f"\n{len(_fail)} INCONSISTENCIES")
    if "--selftest" in sys.argv:
        print("\n--- selftest: each perturbation must be detected ---")
        cases = [("named", "it10", 2, 999.9), ("named", "it12", 0, 111.1),
                 ("abl", "it16, noac head = HARD SWITCH", 0, 111.1),
                 ("named", "it16", 3, 0.99)]
        allbad = True
        for c in cases:
            caught = not run(perturb=c, quiet=True)
            print(f"  {'detected' if caught else 'MISSED'}: {c[0]} {c[1]} -> {c[3]}")
            allbad = allbad and caught
        print("SELFTEST PASS" if allbad else "SELFTEST FAIL")
        sys.exit(0 if (ok and allbad) else 1)
    sys.exit(0 if ok else 1)
