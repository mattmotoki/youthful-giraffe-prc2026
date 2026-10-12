"""Parity harness for the final file (PLAN.md block step 5): a rebuilt part against its saved v73-era value.

    OMP_NUM_THREADS=2 .venv/bin/python experiments/exp-186-final-core/code/parity.py \
        --refs <v73-parts.parquet> --new <rebuilt.parquet> --part route --col TAXITIME_SEC_mvt \
        --kind refit --out <parity-route.json>

`--new` is the rebuilt part's output: a parquet with `MVT_ID_mvt` and the value column(s), on the four final
months (final mode) or on the board rows (ranking mode). The harness compares its January/July rows with the
saved value of the same part on the 344,841 board rows (`refs.py`), and describes its February/June rows.

`--part` names a reference column of `v73-parts.parquet` (`<frame>:<column>` or `saved:<name>`) or an alias
(`--list` prints both). Several pairs go in one run: `--pair NEWCOL=REFCOL` (repeat).

Kinds:
* `deterministic`: a part computed from official columns, run on `ranking.parquet`. The check is equality of
  every value (float64 bits, NaN at the same rows; bool and text exact). Exit 1 when a value differs.
  `--tol T` (T > 0) is for a check across machines (the references hold Mac bytes, the final parts come from
  x86; COMMON 14:14 UTC: tolerance 1e-9 s): a number then counts as equal when both are NaN, or both are finite
  and differ by T or less; the record keeps the bit-equal count and gives the largest difference.
* `refit`: a part fitted again, or run on the final file (whose January/July times carry a new jitter draw,
  DATA_CARD). The record gives the rms difference, the bias (new minus saved), the largest difference, the
  rows that are finite in one and missing in the other, and the same by airport and by class (`bundle:cls`)
  and by airport x class. No pass line: the reviewer reads the numbers (AGENTS.md, Independent checks).

The record also gives, for the rows outside January/July (February/June), the rows, the finite share and the
quantiles of the new values by month and airport, next to the same numbers for January/July.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID = "MVT_ID_mvt"
ALIASES = {
    "route": "vector/route:TAXITIME_SEC_mvt", "head": "vector/head:pred_HB3u", "g": "vector/g:pred_Gmixed",
    "mlp": "primary_frame:mlp", "sequence021": "vector/sequence021:pred_seq170",
    "sequence039": "vector/sequence039:pred_seq170", "row6": "vector/row6:pred_seq170",
    "sequence011": "vector/sequence011:pred_seq170", "row011": "vector/row011:pred_seq170",
    "dark18": "vector/dark18:pred_seq170", "s7c": "saved:s7c", "zabd": "saved:zabd-final",
    "v73": "saved:v73-final", "v32": "saved:v32", "v22": "saved:v22", "cls": "bundle:cls",
    "lo": "bundle:lo", "hi": "bundle:hi", "np_v3cal": "bundle:np_v3cal", "noplan": "bundle:noplan",
    "stand_floor": "bundle:stand_floor", "lower": "exploit_frame:lower", "upper": "exploit_frame:upper",
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def r4(x) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def numeric_stats(new: np.ndarray, ref: np.ndarray) -> dict:
    fn, fr = np.isfinite(new), np.isfinite(ref)
    both = fn & fr
    d = new[both] - ref[both]
    out = {"rows": int(len(new)), "both_finite": int(both.sum()), "finite_ref_only": int((fr & ~fn).sum()),
           "finite_new_only": int((fn & ~fr).sum()), "equal_values": int((d == 0).sum())}
    if both.any():
        out |= {"rms_diff": r4(np.sqrt(np.mean(d ** 2))), "bias_new_minus_ref": r4(d.mean()),
                "max_abs_diff": r4(np.abs(d).max()), "p99_abs_diff": r4(np.quantile(np.abs(d), 0.99)),
                "ref_mean": r4(ref[both].mean()), "new_mean": r4(new[both].mean())}
        if both.sum() > 2 and np.std(ref[both]) > 0 and np.std(new[both]) > 0:
            out["corr"] = r4(np.corrcoef(new[both], ref[both])[0, 1])
    return out


def bits_equal(new: pd.Series, ref: pd.Series) -> np.ndarray:
    """Row-wise equality: float64 bits for numbers (NaN equals NaN), exact for bool and text."""
    if pd.api.types.is_numeric_dtype(ref) and not pd.api.types.is_bool_dtype(ref):
        a = new.to_numpy("float64")
        b = ref.to_numpy("float64")
        same = a.view("u8") == b.view("u8")
        return same | (np.isnan(a) & np.isnan(b))
    a, b = new.astype(object).to_numpy(), ref.astype(object).to_numpy()
    na, nb = pd.isna(a), pd.isna(b)
    eq = np.array([x == y for x, y in zip(a, b)], dtype=bool)
    return (eq & ~na & ~nb) | (na & nb)


def describe(v: pd.Series, month: pd.Series, airport: pd.Series) -> dict:
    out = {}
    num = pd.api.types.is_numeric_dtype(v) and not pd.api.types.is_bool_dtype(v)
    for (m, a), g in v.groupby([month, airport]):
        x = g.to_numpy("float64") if num else None
        rec = {"rows": int(len(g))}
        if num:
            f = np.isfinite(x)
            rec["finite_share"] = r4(f.mean())
            if f.any():
                rec |= {"mean": r4(x[f].mean()), "q10": r4(np.quantile(x[f], .1)),
                        "q50": r4(np.quantile(x[f], .5)), "q90": r4(np.quantile(x[f], .9))}
        else:
            rec["values"] = {str(k): int(n) for k, n in g.astype(object).value_counts(dropna=False).items()}
        out[f"{m} {a}"] = rec
    return out


def compare(new: pd.Series, ref: pd.Series, kind: str, airport: np.ndarray, cls: np.ndarray, ids: np.ndarray,
            tol: float = 0.0) -> dict:
    rec: dict = {}
    eq = bits_equal(new, ref)
    rec["rows_equal"] = int(eq.sum())
    if tol > 0 and pd.api.types.is_numeric_dtype(ref) and not pd.api.types.is_bool_dtype(ref):
        x, y = new.to_numpy("float64"), ref.to_numpy("float64")
        both = np.isfinite(x) & np.isfinite(y)
        d = np.where(both, np.abs(x - y), 0.0)
        eq = eq | (both & (d <= tol))
        rec |= {"tol": tol, "rows_bit_equal": rec.pop("rows_equal"), "rows_equal": int(eq.sum()),
                "max_abs_diff_both_finite": float(d.max()) if len(d) else 0.0}
    rec["rows_differ"] = int((~eq).sum())
    if (~eq).any():
        rec["first_differing_ids"] = [int(i) for i in ids[~eq][:10]]
    num = pd.api.types.is_numeric_dtype(ref) and not pd.api.types.is_bool_dtype(ref)
    if num:
        a, b = new.to_numpy("float64"), ref.to_numpy("float64")
        rec["all"] = numeric_stats(a, b)
        if kind == "refit":
            for name, key in (("by_airport", airport), ("by_class", cls),
                              ("by_airport_class", np.char.add(np.char.add(airport.astype(str), " "), cls.astype(str)))):
                rec[name] = {str(k): numeric_stats(a[key == k], b[key == k]) for k in sorted(set(key.tolist()))}
    else:
        x, y = new.astype(object).to_numpy(), ref.astype(object).to_numpy()
        tab = pd.crosstab(pd.Series(y, name="ref").astype(str), pd.Series(x, name="new").astype(str))
        rec["confusion_ref_by_new"] = {r: {c: int(tab.loc[r, c]) for c in tab.columns} for r in tab.index}
        if kind == "refit":
            rec["differ_by_airport"] = {str(k): int((~eq & (airport == k)).sum()) for k in sorted(set(airport.tolist()))}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", type=Path, required=True, help="v73-parts.parquet of refs.py")
    ap.add_argument("--refs-sha256", default=None, help="expected SHA-256 of --refs (default: the .json next to it)")
    ap.add_argument("--new", type=Path)
    ap.add_argument("--part", help="reference column or alias")
    ap.add_argument("--col", help="value column of --new (default: the only non-id column)")
    ap.add_argument("--pair", action="append", default=[], help="NEWCOL=REFCOL (repeat)")
    ap.add_argument("--kind", choices=("deterministic", "refit"), default="refit")
    ap.add_argument("--tol", type=float, default=0.0, help="deterministic across machines: equal within TOL")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    want = a.refs_sha256
    side = a.refs.with_suffix(".json")
    if want is None and side.exists():
        want = json.loads(side.read_text())["output_sha256"]
    got = sha(a.refs)
    if want is not None and got != want:
        raise SystemExit(f"{a.refs}: SHA-256 {got} is not {want}")
    refs = pd.read_parquet(a.refs)
    assert len(refs) == 344_841 and refs[ID].is_unique, len(refs)
    if a.list:
        print("aliases:", json.dumps(ALIASES, indent=1))
        print("columns:", ", ".join(c for c in refs.columns if ":" in c))
        return 0

    new = pd.read_parquet(a.new)
    assert ID in new.columns, f"{a.new} has no {ID}"
    ids = pd.to_numeric(new[ID], errors="raise").to_numpy("float64")
    assert np.all(ids == np.round(ids)), "non-integer ids"
    new[ID] = ids.astype("int64")
    assert new[ID].is_unique, "repeated ids in --new"
    pairs = []
    if a.part:
        col = a.col or [c for c in new.columns if c != ID][0]
        pairs.append((col, ALIASES.get(a.part, a.part)))
    for p in a.pair:
        n, r = p.split("=", 1)
        pairs.append((n, ALIASES.get(r, r)))
    assert pairs, "give --part or --pair"
    for n, r in pairs:
        assert n in new.columns, f"--new has no column {n}"
        assert r in refs.columns, f"no reference column {r} (see --list)"

    # month and airport of every new row: the board rows from the reference table, others from the final file
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from prc import data as D
    board = new[ID].isin(refs[ID])
    meta = refs.set_index(ID)[["airport", "month"]]
    rec: dict = {"refs": str(a.refs), "refs_sha256": got, "new": str(a.new), "new_sha256": sha(a.new),
                 "kind": a.kind, "rows_new": int(len(new)), "rows_new_on_board": int(board.sum()),
                 "board_rows_missing_in_new": int((~refs[ID].isin(new[ID])).sum()), "parts": {}}
    other = new.loc[~board, ID]
    if len(other):
        fr = D.load_ranking(columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"], mode="final")
        fr = fr[fr.PHASE_mvt == "DEP"]
        fr[ID] = fr[ID].astype("int64")
        fm = fr.set_index(ID).rename(columns={"ADEP_mvt": "airport"})[["airport", "month"]]
        miss = ~other.isin(fm.index)
        rec["new_rows_not_final_departures"] = int(miss.sum())
        meta = pd.concat([meta, fm.loc[other[~miss]]])
    j = refs.merge(new, on=ID, how="inner", suffixes=("", "__new"))
    airport = j["airport"].astype(str).to_numpy()
    cls = j["bundle:cls"].astype(str).to_numpy()
    for n, r in pairs:
        ncol = n if n not in refs.columns else f"{n}__new"
        res = compare(j[ncol], j[r], a.kind, airport, cls, j[ID].to_numpy(), a.tol)
        known = new[ID].isin(meta.index)
        m = meta.loc[new.loc[known, ID]]
        res["new_values_by_month_airport"] = describe(new.loc[known, n].reset_index(drop=True),
                                                      m["month"].reset_index(drop=True),
                                                      m["airport"].reset_index(drop=True))
        rec["parts"][f"{n} vs {r}"] = res
    fail = a.kind == "deterministic" and (rec["board_rows_missing_in_new"] > 0 or
                                          any(p["rows_differ"] for p in rec["parts"].values()))
    rec["verdict"] = ((f"deterministic: all board values equal within {a.tol:g}" if a.tol > 0 else
                       "deterministic: all board values equal") if a.kind == "deterministic" and not fail else
                      "deterministic: VALUES DIFFER" if fail else "refit: read the numbers")
    text = json.dumps(rec, indent=1, default=str) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    summary = {k: rec[k] for k in ("rows_new", "rows_new_on_board", "board_rows_missing_in_new", "verdict")}
    summary["parts"] = {k: {kk: v[kk] for kk in ("rows_equal", "rows_differ", "all") if kk in v}
                        for k, v in rec["parts"].items()}
    print(json.dumps(summary, indent=1, default=str))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
