"""PLAN.md (d): label-free H8 checks of a final candidate file (exp-186 extension of the plan's checker).

Run: OMP_NUM_THREADS=2 .venv/bin/python -I check_final_candidate.py <candidate.parquet> <repo root>
         [--janjul-ref <v73.parquet>] [--stages <build-record.json>] [--parity <parity-spec.json>]
         [--setters <rows.parquet>] [--out <record.json>]
Prints one JSON record and exits 1 when a blocking check fails. Reads no label (the files have none).
The final files are read through `prc.data` in the final mode, so each is accepted only by its SHA-256.

Blocking checks (the organizers' ranking script rejects the file, or the value is not a taxi time):
  schema       two columns MVT_ID_mvt and TAXITIME_SEC_mvt, numeric
  ids          the same IDs as final_submitting.parquet, same order, no duplicate, 670,790 rows
  finite       no null, NaN or infinite value
  floor        every value >= 60 s (our floor) and < 172,800 s
  stages       (with --stages) every stage that changes board rows (January/July) also changes rows of each new
               month (February, June), unless the build record names it `board_only` with a reason; the
               month-loop trap of PLAN.md (b)
  parity       (with --parity) every `deterministic` part of the spec has equal values on the board rows
Information (for the reviewer):
  janjul       on the 344,841 January/July rows: equal to the reference (v73) or the size of the difference
  by_month     rows, mean, quantiles and share at the 60 s floor by month and airport, Feb/Jun against Jan/Jul
  parity       the refit parts of the spec: rms, bias and the largest difference by airport and class
  lirf_tail    the February/June LIRF departures without an aircraft type, and those of them whose take-off is
               20,000 s or more after the schedule, with the candidate value (and the rule that set it, from
               --setters: MVT_ID_mvt, rule)

Build record for --stages (written by the build):
  {"stages": [{"name": "C66", "changed_by_month": {"2026-01": 9, "2026-02": 7, "2026-06": 8, "2026-07": 10}},
              {"name": "EXT", "changed_by_month": {...}, "board_only": "scoped to January by design D<n>"}]}
Parity spec for --parity: [{"part": "route", "new": "<parquet>", "col": "pred", "kind": "refit"}, ...] (a
deterministic run across machines adds "tol": 1e-9) with the
reference table `refs` given once: {"refs": "<v73-parts.parquet>", "runs": [...]}.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
args = sys.argv[1:]


def opt(name):
    if name in args:
        i = args.index(name)
        v = Path(args[i + 1])
        del args[i:i + 2]
        return v
    return None


ref, stages_path, parity_path, setters_path, out_path = (opt("--janjul-ref"), opt("--stages"), opt("--parity"),
                                                         opt("--setters"), opt("--out"))
cand, root = Path(args[0]), Path(args[1]).resolve()
sys.path.insert(0, str(root))
from prc import data as D  # noqa: E402

spec = D.ranking_spec("final")
board_months = D.rank_months("ranking")
new_months = tuple(m for m in spec.months if m not in board_months)
board_numbers = tuple(int(m[5:]) for m in board_months)

rec = {"candidate": str(cand), "sha256": hashlib.sha256(cand.read_bytes()).hexdigest(), "bytes": cand.stat().st_size,
       "final_files": {"ranking": spec.ranking_sha256, "submitting": spec.submitting_sha256}}
fail = []
c = pd.read_parquet(cand)
tpl = D.load_submitting(mode="final")
fr = D.load_ranking(columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                             "AIRCRAFT_TYPE_mvt"], mode="final")
fr = fr[fr.PHASE_mvt.eq("DEP")].set_index(ID)

rec["schema"] = {k: str(v) for k, v in c.dtypes.items()}
if list(c.columns) != [ID, Y] or not all(np.issubdtype(c[k].dtype, np.number) for k in (ID, Y)):
    fail.append("schema")
rec["rows"] = len(c)
ids_ok = len(c) == len(tpl) == spec.departures and c[ID].is_unique and (c[ID].to_numpy() == tpl[ID].to_numpy()).all()
rec["ids_same_order_as_template"] = bool(ids_ok)
if not ids_ok:
    fail.append("ids")
v = c[Y].to_numpy(float)
rec["nonfinite"] = int((~np.isfinite(v)).sum())
if rec["nonfinite"]:
    fail.append("finite")
rec["min"], rec["max"] = float(np.nanmin(v)), float(np.nanmax(v))
rec["below_60"] = int((v < 60).sum())
rec["at_or_above_172800"] = int((v >= 172800).sum())
if rec["below_60"] or rec["at_or_above_172800"]:
    fail.append("floor")

if ids_ok:
    meta = fr.loc[c[ID].to_numpy()]
    month = meta.MVT_TIME_UTC_mvt.dt.month.to_numpy()
    airport = meta.ADEP_mvt.to_numpy()
    jj = np.isin(month, board_numbers)
    rec["rows_by_month"] = {int(m): int((month == m).sum()) for m in spec.month_numbers}
    assert rec["rows_by_month"] == {int(k[5:]): n for k, n in spec.month_departures.items()}, rec["rows_by_month"]
    if ref is not None:
        r = pd.read_parquet(ref).set_index(ID)[Y]
        rv = r.loc[c[ID].to_numpy()[jj]].to_numpy(float)
        d = v[jj] - rv
        rec["janjul"] = {"ref": str(ref), "rows": int(jj.sum()), "equal": int((d == 0).sum()),
                         "rms_diff_s": round(float(np.sqrt(np.mean(d ** 2))), 4), "max_abs_diff_s": float(np.abs(d).max())}
    tab = {}
    for m in spec.month_numbers:
        for a in sorted(set(airport)):
            s = (month == m) & (airport == a)
            x = v[s]
            tab[f"{m:02d} {a}"] = {"rows": int(s.sum()), "mean": round(float(x.mean()), 1),
                                   "q10": round(float(np.quantile(x, .1)), 1), "q50": round(float(np.quantile(x, .5)), 1),
                                   "q90": round(float(np.quantile(x, .9)), 1), "at_60": round(float((x == 60).mean()), 4)}
    rec["by_month_airport"] = tab
    fj = ~jj
    rec["mean_feb_jun_over_jan_jul"] = round(float(v[fj].mean() / v[jj].mean()), 4)

    # LIRF departures of the new months without an aircraft type (PLAN.md (c): 236 rows, 12 of them late)
    lirf = (airport == "LIRF") & ~jj & meta.AIRCRAFT_TYPE_mvt.isna().to_numpy()
    late = (meta.MVT_TIME_UTC_mvt - meta.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy() >= 20_000
    tail = pd.DataFrame({ID: c[ID].to_numpy()[lirf], "month": month[lirf], "value": v[lirf], "late": late[lirf]})
    if setters_path is not None:
        st = pd.read_parquet(setters_path)
        tail = tail.merge(st[[ID, "rule"]].astype({ID: "float64"}), on=ID, how="left")
    rec["lirf_tail"] = {"rows": int(len(tail)), "late_rows": int(tail.late.sum()),
                        "value_quantiles": {q: round(float(tail.value.quantile(q)), 1) for q in (.1, .5, .9)}
                        if len(tail) else {},
                        "late": tail[tail.late].assign(**{ID: lambda t: t[ID].astype("int64")}).to_dict("records")}
    if setters_path is not None:
        rec["lirf_tail"]["rule_counts"] = tail["rule"].fillna("none").value_counts().to_dict()

if stages_path is not None:
    st = json.loads(stages_path.read_text())["stages"]
    rows = []
    for s in st:
        ch = {m: int(s.get("changed_by_month", {}).get(m, 0)) for m in spec.months}
        on_board = sum(ch[m] for m in board_months)
        missing = [m for m in new_months if ch[m] == 0]
        bad = on_board > 0 and bool(missing) and not s.get("board_only")
        rows.append({"name": s["name"], "changed_by_month": ch, "board_only": s.get("board_only"),
                     "new_months_without_change": missing, "defect": bad})
    rec["stages"] = rows
    if any(r["defect"] for r in rows):
        fail.append("stages")

if parity_path is not None:
    ps = json.loads(parity_path.read_text())
    here = Path(__file__).resolve().parent
    rec["parity"] = []
    for i, run in enumerate(ps["runs"]):
        o = Path(ps.get("out_dir", parity_path.parent)) / f"parity-{i:02d}-{run['part'].replace('/', '_').replace(':', '_')}.json"
        cmd = [sys.executable, str(here / "parity.py"), "--refs", ps["refs"], "--new", run["new"], "--part",
               run["part"], "--kind", run.get("kind", "refit"), "--out", str(o)]
        if run.get("col"):
            cmd += ["--col", run["col"]]
        if run.get("tol"):
            cmd += ["--tol", str(run["tol"])]
        p = subprocess.run(cmd, capture_output=True, text=True)
        res = json.loads(o.read_text()) if o.exists() else {"error": p.stderr[-2000:]}
        rec["parity"].append({"part": run["part"], "kind": run.get("kind", "refit"), "exit": p.returncode,
                              "record": str(o), "verdict": res.get("verdict"),
                              "all": {k: v.get("all") for k, v in res.get("parts", {}).items()}})
        if run.get("kind") == "deterministic" and p.returncode != 0:
            fail.append(f"parity:{run['part']}")

rec["blocking_failures"] = fail
rec["verdict"] = "PASS" if not fail else "FAIL"
text = json.dumps(rec, indent=1, default=str)
if out_path is not None:
    out_path.write_text(text + "\n")
print(text)
sys.exit(1 if fail else 0)
