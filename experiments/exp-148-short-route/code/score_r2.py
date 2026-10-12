"""exp-148 SPEC-r2 sections 3 and 5: the form choice and the scores of the R2 level 2, January
and July 2025 joined (344,419 rows), and the tables for the R2 report.

    python score_r2.py --art DIR [--art DIR ...] --stack DIR --features features.parquet \
        --proxy proxy-features.parquet --refs refs.parquet --out DIR

Vectors: the R2 forms (owners r01 and r07, read through `pilot.load`), the janjul anchor (the
R2 records' `anchor_served`), the R22 forms (v22 inputs) when present, Sb of the whole-month
check (owners m01, m07), Sb of the serve read (the saved owner-k stackers re-predict pair k from
the janjul members, `quick_serve.py`'s path, rebuilt here), Sb of the week-pair owners, and the
saved v32 and v22 routes (reference). Form (SPEC-r2 section 3): the lowest `cv_pooled` MSE
averaged over r01 and r07. G by `prc.gscore.score` (all rows and clean, frozen interval,
views). SPEC-r3: the 16 served variants of the chosen form (blends with the anchor route, w in
1, 0.75, 0.5, 0.25; v32's value kept on noac rows, on ordinary rows without AOBT, on both), the
same 16 on the month-check Sb as a reference, and the selection with its cross-month check.
Write-once `score-r2.json` and `report-tables-r2.md`.
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
import clockloss as CL               # noqa: E402
import report_tables as RT           # noqa: E402
from prc import gscore as GS         # noqa: E402
from prc import weekpair as WP       # noqa: E402
import pilot as P                    # noqa: E402

ID = ST.ID
FORMS = {"": "pred_S", "cat": "pred_Scat", "b": "pred_Sb"}
NAME = {"": "", "cat": "cat", "b": "b"}
MONTH_OWNERS = {"R2": (("r01", "month-2025-01"), ("r07", "month-2025-07")),
                "R22": (("r01", "month-2025-01"), ("r07", "month-2025-07")),
                "S": (("m01", "month-2025-01"), ("m07", "month-2025-07"))}
#: SPEC-r3 section 1: the blend weights of the chosen form (the rest on the anchor route) and the
#: keep sets (v32's value), each list from simplest to least simple
W_GRID = (1.0, 0.75, 0.5, 0.25)
KEEPS = {"k0": (), "kn": ("noac",), "ko": ("ordna",), "kno": ("noac", "ordna")}
#: SPEC-r3 section 1: local row counts of the keep sets (frozen rows, January and July)
KEEP_ROWS = {"noac": 402, "ordna": 3709}


def one(arts: list[Path], name: str) -> Path:
    hits = [d / f"{name}.parquet" for d in arts if (d / f"{name}.parquet").exists()]
    if len(hits) != 1:
        raise SystemExit(f"{name}: {len(hits)} artifacts")
    return hits[0]


def joined(arts: list[Path], stage: str, column: str, ids: np.ndarray, rec: dict) -> pd.Series:
    parts = []
    for owner, split in MONTH_OWNERS[stage]:
        path = one(arts, f"{stage}-{split}")
        f = P.load(path, owner, what=f"{stage} {split}")
        parts.append(pd.Series(f[column].to_numpy("float64"), index=f[ID].astype("int64").to_numpy()))
        rec[f"{stage}-{split}"] = RT.sha(path)
    out = pd.concat(parts)
    if out.index.duplicated().any() or set(out.index) != set(ids):
        raise SystemExit(f"{stage}: the outputs do not cover each January and July row once")
    return out.reindex(ids)


def week(arts: list[Path], column: str, ids_days: pd.DataFrame, ids: np.ndarray, rec: dict) -> pd.Series:
    preds = {}
    for k in range(1, 6):
        path = one(arts, f"S-outer-{k}")
        f = P.load(path, k, what=f"S outer-{k}")
        preds[k] = f[[ID, column]].assign(**{ID: f[ID].astype("int64")})
        rec[f"S-outer-{k}"] = RT.sha(path)
    return WP.join_outer(preds, ids_days, column).reindex(ids)


def serve_read(arts: list[Path], feats: pd.DataFrame, proxy: pd.DataFrame, ids_days: pd.DataFrame,
               ids: np.ndarray) -> pd.Series:
    mem, _ = ST.members_for(arts, [("janjul", "janjul")], "janjul")
    df = ST.frame(mem, feats, proxy)
    pair = ids_days.set_index(ID).pair.reindex(df[ID].to_numpy("int64")).to_numpy()
    fx = json.loads(ST.find(arts, "S", "outer-1").with_suffix(".json").read_text())["features"]
    per = {k: CL.predict(arts, "S", k, df[pair == k], fx) for k in range(1, 6)}
    return WP.join_outer(per, ids_days, "pred_Sb").reindex(ids)


def variants(base: np.ndarray, anchor: np.ndarray, v32: np.ndarray, sets: dict, prefix: str) -> dict:
    """SPEC-r3 section 1: blend(w) = w * base + (1 - w) * anchor, then v32's value on the keep set."""
    out = {}
    for w in W_GRID:
        b = w * base + (1 - w) * anchor
        for k, members in KEEPS.items():
            m = np.zeros(len(b), bool)
            for name in members:
                m |= sets[name]
            out[f"{prefix}-w{int(round(w * 100))}-{k}"] = np.where(m, v32, b)
    return out


def parts(name: str) -> tuple[int, str]:
    """(w in percent, keep set) of a variant name `<prefix>-w<w>-<keep>`."""
    head, keep = name.rsplit("-", 1)
    return int(head.rsplit("-w", 1)[1]), keep


def select(scores: dict, names: list[str]) -> dict:
    """SPEC-r3 section 2: the highest joined G against v32's saved route; within 1 s² the simpler
    variant (larger w, then the smaller keep set); the cross-month check."""
    g = {n: scores[f"{n}_vs_v32_ref"]["all"]["G"] for n in names}
    top = max(g.values())
    simple = sorted(names, key=lambda n: (-parts(n)[0], list(KEEPS).index(parts(n)[1])))
    pick = next(n for n in simple if g[n] >= top - 1.0)
    month = {n: scores[f"{n}_vs_v32_ref"]["all"]["views"]["month"] for n in names}
    by_jan = max(names, key=lambda n: month[n]["1"])
    by_jul = max(names, key=lambda n: month[n]["7"])
    w, keep = parts(pick)
    return {"variant": pick, "w": w / 100, "keep": keep, "keep_sets": list(KEEPS[keep]),
            "G": g[pick], "top_G": top, "month_contributions": month[pick],
            "chosen_by_january": {"variant": by_jan, "january": month[by_jan]["1"], "july": month[by_jan]["7"]},
            "chosen_by_july": {"variant": by_jul, "july": month[by_jul]["7"], "january": month[by_jul]["1"]}}


def variant_rows(scores: dict, names: list[str], ref: str | None) -> list[str]:
    out = ["| variant | G vs v32 (all) | 95 % interval | P(G ≤ 0) | G clean | January | July | "
           "ordinary−aobt | noac−aobt | G vs w100-k0 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        a, c = scores[f"{n}_vs_v32_ref"]["all"], scores[f"{n}_vs_v32_ref"]["clean"]
        iv, mo, ca = a["interval"], a["views"]["month"], a["views"]["class_aobt"]
        r = scores.get(f"{n}_vs_{ref}", {}).get("all", {}).get("G") if ref else None
        out.append(f"| {n} | {RT.fmt(a['G'])} | [{RT.fmt(iv['lo95'])}, {RT.fmt(iv['hi95'])}] | "
                   f"{RT.fmt(iv['p_G_le_0'], 3)} | {RT.fmt(c['G'])} | {RT.fmt(mo.get('1'))} | {RT.fmt(mo.get('7'))} | "
                   f"{RT.fmt(ca.get('ordinary-aobt'))} | {RT.fmt(ca.get('noac-aobt'))} | {RT.fmt(r)} |")
    return out


def form_choice(stack: Path) -> tuple[str, list[str]]:
    rows, sums = [], {k: 0.0 for k in ("mse_anchor", *RT.CV_FORM)}
    for owner, split in MONTH_OWNERS["R2"]:
        r = json.loads((stack / f"R2-{split}.json").read_text())
        cv = r["cv_pooled"]
        for c in sums:
            sums[c] += cv[c] / 2
        rows.append(f"| {owner} | {r['train_rows']:,} | {r['rounds']} | {r['rounds_cat']} | "
                    + " | ".join(RT.fmt(cv[c], 0) for c in ("mse_anchor", *RT.CV_FORM)) + " |")
    best = min(RT.CV_FORM, key=lambda c: sums[c])
    form = {"S": "R2", "Scat": "R2cat", "Sb": "R2b"}[RT.CV_FORM[best]]
    out = ["| owner | training rows | LightGBM rounds | CatBoost rounds | MSE anchor | MSE LightGBM | "
           "MSE CatBoost | MSE mean |", "|---|---|---|---|---|---|---|---|", *rows,
           "| **mean** | | | | " + " | ".join(("**" + RT.fmt(sums[c], 0) + "**") for c in ("mse_anchor", *RT.CV_FORM)) + " |",
           "", f"Chosen form (SPEC-r2 section 3): **{form}** (lowest mean pooled CV MSE over r01 and r07, "
           f"{RT.fmt(sums[best], 0)} s²; leave one training month out)."]
    return form, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", type=Path, action="append", required=True)
    ap.add_argument("--stack", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--proxy", type=Path, required=True)
    ap.add_argument("--refs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    js, md = a.out / "score-r2.json", a.out / "report-tables-r2.md"
    if js.exists() or md.exists():
        raise SystemExit(f"{js} exists; write-once")
    form, form_lines = form_choice(a.stack)
    rows = GS.load_rows()
    ids_days = rows[rows.month.isin([1, 7])][[ID, "pair"]]
    ids = ids_days[ID].to_numpy("int64")
    vec, rec = {}, {}
    for f, col in FORMS.items():
        vec[f"R2{f}"] = joined(a.art, "R2", col, ids, rec)
    vec["anchor_jj"] = joined(a.art, "R2", "anchor_served", ids, rec)
    if all(any((d / f"R22-{s}.parquet").exists() for d in a.art) for _, s in MONTH_OWNERS["R22"]):
        for f, col in FORMS.items():
            vec[f"R22{f}"] = joined(a.art, "R22", col, ids, rec)
    vec["Sb_month"] = joined(a.art, "S", "pred_Sb", ids, rec)
    anchor_month = joined(a.art, "S", "anchor_served", ids, rec)
    anchor_check = float(np.max(np.abs(anchor_month.to_numpy() - vec["anchor_jj"].to_numpy())))
    if anchor_check > 1e-6:
        raise SystemExit(f"the R2 anchor differs from the month check's anchor by {anchor_check}")
    vec["Sb_week"] = week(a.art, "pred_Sb", ids_days, ids, rec)
    feats = pd.read_parquet(a.features)
    proxy = pd.read_parquet(a.proxy)
    proxy[ID] = proxy[ID].astype("int64")
    vec["Sb_serve"] = serve_read(a.art, feats, proxy, ids_days, ids)
    refs = pd.read_parquet(a.refs).set_index(ID)
    vec["v32_ref"] = refs.v32.reindex(ids)
    vec["v22_ref"] = refs.v22.reindex(ids)
    rec["refs"] = RT.sha(a.refs)
    y = GS.labels(ids)
    pair = ids_days.set_index(ID)["pair"].reindex(ids).to_numpy()
    r22 = {"R2": "R22", "R2cat": "R22cat", "R2b": "R22b"}
    pairs = [(form, b) for b in ("v32_ref", "v22_ref", "anchor_jj", r22[form], "Sb_month", "Sb_serve", "Sb_week")]
    pairs += [(r22[form], "anchor_jj"), ("Sb_serve", "v32_ref"), ("Sb_month", "v32_ref"), ("Sb_week", "v32_ref"),
              ("v32_ref", "anchor_jj"), ("v22_ref", "anchor_jj")]
    pairs += [(f, b) for f in ("R2", "R2cat", "R2b") if f != form for b in ("v32_ref", "anchor_jj", r22[f])]
    # SPEC-r3: the served variants of the chosen form and of the month-check Sb (reference)
    r0 = rows.set_index(ID).reindex(ids)
    sets = {"noac": (r0.cls == "noac").to_numpy(), "ordna": ((r0.cls == "ordinary") & ~r0.aobt.astype(bool)).to_numpy()}
    got = {k: int(v.sum()) for k, v in sets.items()}
    if got != KEEP_ROWS:
        raise SystemExit(f"keep-set rows {got}, SPEC-r3 says {KEEP_ROWS}")
    v32 = vec["v32_ref"].to_numpy("float64")
    anc = vec["anchor_jj"].to_numpy("float64")
    var = {**variants(vec[form].to_numpy("float64"), anc, v32, sets, form),
           **variants(vec["Sb_month"].to_numpy("float64"), anc, v32, sets, "Sb_month")}
    r2_names = [n for n in var if n.startswith(f"{form}-")]
    ref = f"{form}-w100-k0"
    out = {"what": "exp-148 SPEC-r2 scores (R2 level 2, January and July 2025) and SPEC-r3 variants",
           "spec_sha256": {n: RT.sha(HERE.parent / n) for n in ("SPEC.md", "SPEC-r1.md", "SPEC-r2.md", "SPEC-r3.md")},
           "protocol": WP.PROTOCOL, "splits_fingerprint": WP.fingerprint(),
           "frozen_manifest_sha256": GS._sha256_file(GS.FROZEN / "MANIFEST.json"),
           "form": form, "artifacts": rec, "rows": int(len(ids)), "anchor_check_max_abs": anchor_check,
           "code_sha256": {"score_r2.py": RT.sha(Path(__file__)), "stacker.py": RT.sha(HERE / "stacker.py"),
                           "clockloss.py": RT.sha(HERE / "clockloss.py")},
           "features_sha256": RT.sha(a.features), "proxy_sha256": RT.sha(a.proxy),
           "rmse_by_stage": {k: float(np.sqrt(np.mean((y - v.to_numpy("float64")) ** 2))) for k, v in vec.items()},
           "keep_rows": got, "variants": list(var), "scores": {}}
    for cand, base in pairs:
        if cand not in vec or base not in vec:
            continue
        c, b = vec[cand].to_numpy("float64"), vec[base].to_numpy("float64")
        res = {view: GS.score(ids, b, c, y=y, view=view, groups={"pair": pair}) for view in ("all", "clean")}
        out["scores"][f"{cand}_vs_{base}"] = res
        print(json.dumps({f"{cand}_vs_{base}": RT.g_row(f"{cand} vs {base}", res)}), flush=True)
    pair_keys = list(out["scores"])          # the SPEC-r2 comparisons; the variants get their own tables
    for n, c in var.items():
        res = {view: GS.score(ids, v32, c, y=y, view=view, groups={"pair": pair}) for view in ("all", "clean")}
        out["scores"][f"{n}_vs_v32_ref"] = res
        print(json.dumps({f"{n}_vs_v32_ref": RT.g_row(n, res)}), flush=True)
        if n in r2_names and n != ref:
            out["scores"][f"{n}_vs_{ref}"] = {"all": GS.score(ids, var[ref], c, y=y, view="all", groups={"pair": pair})}
    out["rmse_by_variant"] = {n: float(np.sqrt(np.mean((y - c) ** 2))) for n, c in var.items()}
    out["choice"] = {"form": form, "form_column": FORMS[{"R2": "", "R2cat": "cat", "R2b": "b"}[form]],
                     **select(out["scores"], r2_names)}
    print(json.dumps({"choice": out["choice"]}), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    lines = ["# exp-148 R2 tables (generated by code/score_r2.py)", "",
             f"Input: `score-r2.json` (written by the same run); SPEC-r2 sha256 {out['spec_sha256']['SPEC-r2.md'][:12]}…", "",
             "## Form choice (SPEC-r2 section 3, training-month CV only)", "", *form_lines, "",
             "## G, January and July 2025 joined (344,419 rows; members saw no day of January or July)", "",
             *RT.G_HEAD, *[RT.g_row(f"{k.replace('_vs_', ' vs ')}", out["scores"][k]) for k in pair_keys], "",
             "RMSE by stage (s): " + "; ".join(f"{k} {v:.2f}" for k, v in out["rmse_by_stage"].items()), "",
             *RT.views(out["scores"], f"{form}_vs_v32_ref"), "", *RT.views(out["scores"], f"{form}_vs_anchor_jj"), "",
             "## SPEC-r3 variants of the chosen form (selectable)", "",
             f"Keep-set rows (local): {got}. v32's value is the saved month-fold route (a reference, SPEC-r3 section 1).", "",
             *variant_rows(out["scores"], r2_names, ref), "",
             f"Chosen (SPEC-r3 section 2): **{out['choice']['variant']}**, G {RT.fmt(out['choice']['G'])} "
             f"(January {RT.fmt(out['choice']['month_contributions'].get('1'))}, July "
             f"{RT.fmt(out['choice']['month_contributions'].get('7'))}). Chosen by January alone: "
             f"{out['choice']['chosen_by_january']['variant']} (its July {RT.fmt(out['choice']['chosen_by_january']['july'])}); "
             f"by July alone: {out['choice']['chosen_by_july']['variant']} (its January "
             f"{RT.fmt(out['choice']['chosen_by_july']['january'])}).", "",
             "## SPEC-r3 variants of the month-check Sb (reference, not selectable)", "",
             *variant_rows(out["scores"], [n for n in var if n.startswith("Sb_month-")], None), ""]
    md.write_text("\n".join(lines) + "\n")
    print(json.dumps(out["rmse_by_stage"], indent=1))


if __name__ == "__main__":
    main()
