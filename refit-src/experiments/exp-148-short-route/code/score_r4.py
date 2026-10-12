"""exp-148 SPEC-r4 items M.5 and L: form choice and local scores of a new level-2 stage, January and
July 2025 joined (344,419 rows), under R2's condition.

    python score_r4.py --stage R4 [--control R2g] --art DIR [--art DIR ...] --stack DIR \
        --refs refs.parquet --score score-r2.json --out DIR

Stages: R4 (owners q01, q07; nine months), R4s (the same with the season input), R2g (owners
r01, r07; R2's six months with CatBoost on the GPU), R2s, R2Lc and R2Lv (SPEC-r4n: the L pack's
control and variant arms; `--stage R2Lv --control R2Lc` gives the paired G as X against control). Form: SPEC-r2 section 3's rule on the
stage's two owners. Local vectors: X = the chosen form; X-kno = v32's saved route on noac rows
and ordinary rows without AOBT (SPEC-r3); X-B50 = 0.5 × X-kno + 0.5 × v32's saved route (SPEC-r4
section V, after v36). Reference: the local v35 (R2b-w100-kno, rebuilt; must give the saved G) and
the local B50 = 0.5 × v35 + 0.5 × v32. Candidate rule (SPEC-r4, the desk after v36): G of X-B50
against the local B50 above 0. Also: X against R2b, the control's form and the anchor, and the
LightGBM form (pred_S) of X against R2's (the same CPU learner on both). Write-once
`score-<stage>.json` and `report-tables-<stage>.md`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stacker as ST                 # noqa: E402  (sets sys.path for prc and pilot)
import score_r2 as SR                # noqa: E402
import report_tables as RT           # noqa: E402
from prc import gscore as GS         # noqa: E402
from prc import weekpair as WP       # noqa: E402

ID = ST.ID
OWNERS = {"R4": ("q01", "q07"), "R4s": ("q01", "q07"), "R2g": ("r01", "r07"), "R2s": ("r01", "r07"),
          "R2Lc": ("r01", "r07"), "R2Lv": ("r01", "r07"),      # SPEC-r4n: the L pack pair
          "R4q": ("q01", "q07"),                                 # R4 with the queue count columns
          "R4Lv": ("q01", "q07"),                                # R4 with the L and Lp bag members
          "R4d1m": ("q01", "q07"),                               # R4 with D1-winter's four-seed mean
          "R4LvO": ("q01", "q07"),                               # R4Lv plus the OPDI members
          "R4l2m": ("q01", "q07"),                               # R4 with L2-hub8725's four-seed mean
          "R4A3": ("q01", "q07"), "R4A3c": ("q01", "q07"),       # R4 with exp-159's head A3 (A3c)
          "R4c2m": ("q01", "q07"),                               # R4 with C2-cat's four-seed mean
          "R4mk": ("q01", "q07"), "R4mks": ("q01", "q07"),       # R4 with the level-2 loss mask
          "R4HB1": ("q01", "q07"),                               # R4 with exp-159's head HB1
          "R4F2": ("q01", "q07"),                                # R4 with exp-160's F2 member
          "R4HB1mk": ("q01", "q07"),                             # R4HB1 with the level-2 loss mask
          "R4F5": ("q01", "q07"),                                # R4 with features2's F5 member
          "R4SR": ("q01", "q07"),                                # R4 with exp-170's Seq170 and Row170
          "R4Mlp": ("q01", "q07"),                               # R4 with exp-164's MLP Mlp164
          "R4Mlpmk": ("q01", "q07"),                             # R4Mlp with the level-2 loss mask
          "R4mksD1s": ("q01", "q07"),                            # R4mks with D1s (D1-winter stop on eligible rows)
          "R4mksC": ("q01", "q07")}                              # R4mks with the round cap raised
#: stages whose anchor differs from R2's by design (a BASE member swapped); the check is recorded
ANCHOR_SWAP = ("R4d1m", "R4l2m", "R4c2m", "R4mksD1s")
#: stages whose form is set, not chosen: R4mksC changes only R4mks's rounds, and its CV (from probes)
#: has no mse_mean, so it keeps R4mks's form (Sb, score-R4mks.json); the stack reads pred_Sb
FIXED_FORM = {"R4mksC": "Sb"}
SPLITS = ("month-2025-01", "month-2025-07")
FORM_COLUMN = {"S": "pred_S", "Scat": "pred_Scat", "Sb": "pred_Sb"}


def form_choice(stack: Path, stage: str) -> tuple[str, list[str], dict]:
    sums = {k: 0.0 for k in ("mse_anchor", *RT.CV_FORM)}
    rows = []
    for owner, split in zip(OWNERS[stage], SPLITS):
        r = json.loads((stack / f"{stage}-{split}.json").read_text())
        for c in sums:
            sums[c] += r["cv_pooled"].get(c, np.nan) / 2
        rows.append(f"| {owner} | {r['train_rows']:,} | {r['rounds']} | {r['rounds_cat']} | "
                    + " | ".join(RT.fmt(r["cv_pooled"].get(c), 0) for c in ("mse_anchor", *RT.CV_FORM)) + " |")
    if stage in FIXED_FORM:
        best = next(c for c, f in RT.CV_FORM.items() if f == FIXED_FORM[stage])
    elif any(np.isnan(v) for v in sums.values()):
        raise SystemExit(f"{stage}: a CV form has no MSE")
    else:
        best = min(RT.CV_FORM, key=lambda c: sums[c])
    lines = ["| owner | training rows | LightGBM rounds | CatBoost rounds | MSE anchor | MSE LightGBM | "
             "MSE CatBoost | MSE mean |", "|---|---|---|---|---|---|---|---|", *rows,
             "| **mean** | | | | " + " | ".join(RT.fmt(sums[c], 0) for c in ("mse_anchor", *RT.CV_FORM)) + " |"]
    return RT.CV_FORM[best], lines, sums


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(OWNERS))
    ap.add_argument("--control", choices=sorted(OWNERS))
    ap.add_argument("--art", type=Path, action="append", required=True)
    ap.add_argument("--stack", type=Path, required=True)
    ap.add_argument("--refs", type=Path, required=True)
    ap.add_argument("--score", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    js, md = a.out / f"score-{a.stage}.json", a.out / f"report-tables-{a.stage}.md"
    if js.exists() or md.exists():
        raise SystemExit(f"{js} exists; write-once")
    form, form_lines, cv = form_choice(a.stack, a.stage)
    col = FORM_COLUMN[form]
    rows = GS.load_rows()
    ids_days = rows[rows.month.isin([1, 7])][[ID, "pair"]]
    ids = ids_days[ID].to_numpy("int64")
    rec: dict = {}
    for st in (a.stage, a.control):
        if st:
            SR.MONTH_OWNERS[st] = tuple(zip(OWNERS[st], SPLITS))
    vec = {"X": SR.joined(a.art, a.stage, col, ids, rec).to_numpy("float64"),
           "X_lgb": SR.joined(a.art, a.stage, "pred_S", ids, rec).to_numpy("float64"),
           "X_anchor": SR.joined(a.art, a.stage, "anchor_served", ids, rec).to_numpy("float64"),
           "R2b": SR.joined(a.art, "R2", "pred_Sb", ids, rec).to_numpy("float64"),
           "R2_lgb": SR.joined(a.art, "R2", "pred_S", ids, rec).to_numpy("float64"),
           "anchor_jj": SR.joined(a.art, "R2", "anchor_served", ids, rec).to_numpy("float64")}
    if a.control:
        cform, _, _ = form_choice(a.stack, a.control)
        vec["control"] = SR.joined(a.art, a.control, FORM_COLUMN[cform], ids, rec).to_numpy("float64")
        vec["control_lgb"] = SR.joined(a.art, a.control, "pred_S", ids, rec).to_numpy("float64")
    anchor_diff = float(np.max(np.abs(vec["X_anchor"] - vec["anchor_jj"])))
    if anchor_diff > 1e-6 and a.stage not in ANCHOR_SWAP:
        raise SystemExit(f"the stage's anchor differs from R2's by {anchor_diff}")
    v32 = pd.read_parquet(a.refs).set_index(ID).v32.reindex(ids).to_numpy("float64")
    r0 = rows.set_index(ID).reindex(ids)
    cls = r0.cls.astype(str).to_numpy()
    sets = {"noac": cls == "noac", "ordna": (cls == "ordinary") & ~r0.aobt.astype(bool).to_numpy()}
    kno = lambda v: SR.variants(v, vec["anchor_jj"], v32, sets, "x")["x-w100-kno"]   # noqa: E731
    vec["v35_local"] = kno(vec["R2b"])
    vec["B50_local"] = 0.5 * vec["v35_local"] + 0.5 * v32
    vec["X_kno"] = kno(vec["X"])
    vec["X_B50"] = 0.5 * vec["X_kno"] + 0.5 * v32
    vec["v32_ref"] = v32
    y = GS.labels(ids)
    pair = ids_days.set_index(ID)["pair"].reindex(ids).to_numpy()
    comps = [("X_B50", "B50_local"), ("X_kno", "v35_local"), ("X", "R2b"), ("X_lgb", "R2_lgb"),
             ("X", "anchor_jj"), ("X_B50", "v32_ref"), ("v35_local", "v32_ref"), ("B50_local", "v35_local")]
    if a.control:
        comps += [("X", "control"), ("X_lgb", "control_lgb"), ("control", "R2b"), ("control_lgb", "R2_lgb")]
    out = {"what": f"exp-148 SPEC-r4 local scores of stage {a.stage}", "stage": a.stage, "control": a.control,
           "form": form, "form_column": col, "cv_pooled_mean": cv,
           "spec_sha256": {n: RT.sha(HERE.parent / n) for n in ("SPEC-r2.md", "SPEC-r3.md", "SPEC-r4.md")},
           "protocol": WP.PROTOCOL, "frozen_manifest_sha256": GS._sha256_file(GS.FROZEN / "MANIFEST.json"),
           "code_sha256": {"score_r4.py": RT.sha(Path(__file__)), "score_r2.py": RT.sha(HERE / "score_r2.py"),
                           "stacker.py": RT.sha(HERE / "stacker.py")},
           "refs_sha256": RT.sha(a.refs), "artifacts": rec, "rows": int(len(ids)), "anchor_check_max_abs": anchor_diff,
           "rmse": {k: float(np.sqrt(np.mean((y - v) ** 2))) for k, v in vec.items()}, "scores": {}}
    for c, b in comps:
        res = {view: GS.score(ids, vec[b], vec[c], y=y, view=view, groups={"pair": pair}) for view in ("all", "clean")}
        out["scores"][f"{c}_vs_{b}"] = res
        print(json.dumps({f"{c}_vs_{b}": RT.g_row(f"{c} vs {b}", res)}), flush=True)
    saved = json.loads(a.score.read_text())["scores"]["R2b-w100-kno_vs_v32_ref"]["all"]["G"]
    got = out["scores"]["v35_local_vs_v32_ref"]["all"]["G"]
    if abs(got - saved) > 1e-6:
        raise SystemExit(f"rebuilt v35 local G {got} differs from score-r2.json {saved}")
    g = out["scores"]["X_B50_vs_B50_local"]["all"]
    name = a.stage + {"S": "", "Scat": "cat", "Sb": "b"}[form]
    out["candidate"] = {"variant": f"{name}-w100-kno-B50", "G_vs_B50_local": g["G"], "interval": g["interval"],
                        "rule": "G > 0 (SPEC-r4)", "candidate": bool(g["G"] > 0)}
    # read by serve_r2.py (--owner of the stage's final): X-kno, then serve_blend.py gives X-B50
    out["choice"] = {"form": name, "form_column": col, "variant": f"{name}-w100-kno", "w": 1.0, "keep": "kno",
                     "keep_sets": ["noac", "ordna"]}
    a.out.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    lines = [f"# exp-148 SPEC-r4 tables of stage {a.stage} (generated by code/score_r4.py)", "",
             f"SPEC-r4 sha256 {out['spec_sha256']['SPEC-r4.md'][:12]}…; rows {out['rows']:,}.", "",
             "## Form choice (training-month CV only)", "", *form_lines, "",
             f"Chosen form: **{form}** ({col}).", "", "## G (January and July 2025 joined)", "",
             *RT.G_HEAD, *[RT.g_row(k.replace("_vs_", " vs "), v) for k, v in out["scores"].items()], "",
             "RMSE (s): " + "; ".join(f"{k} {v:.2f}" for k, v in out["rmse"].items()), "",
             f"Candidate rule (G of X_B50 against the local B50 > 0): G {g['G']:,.1f}, candidate "
             f"{out['candidate']['candidate']}.", "", *RT.views(out["scores"], "X_B50_vs_B50_local"), "",
             *RT.views(out["scores"], "X_vs_R2b"), ""]
    md.write_text("\n".join(lines) + "\n")
    print(md.read_text())


if __name__ == "__main__":
    main()
