"""Dump the v64 chain for the held 2025 months (fold) and for 2026 (ranking).

Runs the same steps as release/assembly_replay.assemble (zabd) and then the T11b-L stage
on both splits, and writes one parquet per split with the members, the stage values and
the frame columns. The fold file also holds the label y and the exposure weight w
(G = sum w_i d_i). Checks: the ranking final equals the v64 replay bytes.

    .venv/bin/python iso/dump_chain.py --data-root input-root --out work/chain
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "release"))
import assembly_replay as R  # noqa: E402
import finalmode as FM  # noqa: E402
import t11b_stage as S  # noqa: E402

ID, Y = R.ID, R.Y
V64_SHA = "9cd421ddc503ef528d7211505436549e44498d015594d37a9080df562050571b"
ZABD_SHA = "e7b04ffdd02a27d48dfcbf12fd3f682f7adf3fbef711dac5e34125296804a636"


def run(data_root: Path, out: Path):
    pack = json.loads((ROOT / "release/ZABD-PACK.json").read_text())
    official = R.official_tables(data_root, pack["official_sha256"])
    rows = {split: len(official[split]) for split in ("fold", "ranking")}
    frames = {}
    for key, spec in pack["frames"].items():
        frame = R.rebuild_frame(spec, official[spec["split"]], rows[spec["split"]])
        digest = R.frame_digest(frame)
        assert digest == spec["digest"], key
        frame.attrs["digest"] = digest
        frames[key] = frame
    A, T = R._assembly_modules()
    rd = R.make_reader(A, frames, pack["records"], official["template_path"])
    parent, extra = pack["manifest"]["parent"], pack["manifest"]["extra"]
    contexts = A.load(parent, rd)
    pw = A.fit_weights(parent, contexts, rd, {"refit": True})
    old = {}
    for split, c in contexts.items():
        old[split], _, _ = A.pre_stack(c, pw, split)
    A.rules(parent, contexts, old, rd)
    A.tail(parent, contexts, old, rd)
    extras = {split: rd.aligned(extra[split], c["b"], c["scope"], year=2025 if split == "fold" else 2026)
              for split, c in contexts.items()}
    rframe = rd.frame(parent["common"]["rule_frame"]["fold"])
    exposure = rd.frame(parent["common"]["exposure"])
    fw, pp = T.fit(contexts["fold"], extras["fold"], rframe.y.to_numpy(), exposure.weight.to_numpy())
    stages = {}
    for split, c in contexts.items():
        stages[split], _ = T.pre(c, extras[split], pw, fw, pp, split)
    A.rules(parent, contexts, stages, rd)
    rd.zero_order_delta = frozenset(parent["common"]["exploit_frame"][s]["path"] for s in ("fold", "ranking"))
    A.tail(parent, contexts, stages, rd)

    # raw member values (unblended) by role, as the loader aligned them
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    for split, c in contexts.items():
        b, f = c["b"], c["frame"]
        year = 2025 if split == "fold" else 2026
        raw = {}
        for role, rec in parent["vectors"][split].items():
            q = rd.frame(rec)
            raw[role] = q.set_index(rec.get("id_column", ID))[rec["column"]].reindex(b[ID]).to_numpy(float)
        ef = rd.frame(parent["common"]["exploit_frame"][split])
        d = pd.DataFrame({
            ID: b[ID].to_numpy("int64"), "airport": b.airport.to_numpy(), "month": b.month.to_numpy(),
            "cls": b.cls.to_numpy(), "scope": c["scope"], "ordinary": c["ordinary"],
            "rule": b.rule.to_numpy(bool), "extension": b.extension.to_numpy(bool),
            "lo": b.lo.to_numpy(float), "hi": b.hi.to_numpy(float),
            "lower": ef.lower.to_numpy(float), "upper": ef.upper.to_numpy(float),
            "noplan": b.noplan.to_numpy(bool), "practice": b.practice.to_numpy(bool), "c73": b.c73.to_numpy(bool),
            **{f"raw_{k}": v for k, v in raw.items()},
            "m_dark18": extras[split],
            **{f"st_{k}": np.asarray(v, float) for k, v in stages[split].items()},
        })
        result[split] = d

    # T11b-L on both splits
    raw_cols = S.COLS + ["LOBT_flt", "AIRCRAFT_TYPE_mvt"]
    for split in ("fold", "ranking"):
        d = result[split]
        off = official[split].set_index(ID).reindex(d[ID].to_numpy())
        if split == "ranking":
            raw = pd.read_parquet(Path(data_root) / R.OFFICIAL_FILES["ranking"], columns=S.COLS)
        else:
            raw = pd.concat([pd.read_parquet(Path(data_root) / R.OFFICIAL_FILES[k], columns=S.COLS)
                             for k in ("training_2025_01", "training_2025_07")], ignore_index=True)
        joined = off["LOBT_flt"].notna().to_numpy()
        assert np.array_equal(joined, np.isfinite(d.lo) & np.isfinite(d.hi)), split
        typed = off["AIRCRAFT_TYPE_mvt"].notna().to_numpy() & (d.cls.to_numpy() != "noac")
        p2 = typed & ~joined
        assert not (p2 & d.scope.to_numpy()).any()
        _, g = S.flags(raw)
        days = g[g.snow]
        keys = set(days.ap + "|" + days.dstr)
        long_rows = S.row_keys(d[ID].to_numpy(), raw).isin(keys).to_numpy()
        base = d["st_final"].to_numpy(float)
        v64 = S.apply(base, d["m_dark18"].to_numpy(float), d.lower.to_numpy(float), d.upper.to_numpy(float), p2, long_rows)
        d["p2"] = p2
        d["long_day"] = long_rows
        d["zabd"] = base
        d["v64"] = v64
        print(split, "p2", int(p2.sum()), "long days", len(keys), "changed", int((v64 != base).sum()))

    fold = result["fold"]
    fold["y"] = rframe.y.to_numpy(float)
    fold["w"] = exposure.weight.to_numpy(float)
    assert np.array_equal(rframe[ID].to_numpy("int64"), fold[ID].to_numpy())

    # byte checks on the ranking split; final mode (exp-191): the hashes of the final pack that finalplan's
    # assemble192.py wrote on the same machine, and a reference fallback (CHAIN_REF, the Z part) within 1e-9 s
    rk = result["ranking"]
    want_sha = {"zabd": ZABD_SHA, "v64": V64_SHA}
    tol = 1e-6
    if FM.is_final():
        want_sha = {"zabd": pack["candidate_sha256"], "v64": pack["final_build"]["v64_sha256"]}
        tol = 1e-9
        assert len(rk) == FM.n_rows()
    for name, want in want_sha.items():
        q = pd.DataFrame({ID: rk[ID].to_numpy("float64"), Y: rk[name].to_numpy("float64")})
        buf = io.BytesIO(); q.to_parquet(buf, index=False)
        got = hashlib.sha256(buf.getvalue()).hexdigest()
        print(name, got, "OK" if got == want else "MISMATCH")
        if got != want and os.environ.get("CHAIN_REF"):
            # another platform: floating-point order differs; accept values within 1e-6 s of the reference
            ref = pd.read_parquet(os.environ["CHAIN_REF"])
            assert np.array_equal(ref[ID].to_numpy("float64"), q[ID].to_numpy())
            diff = float(np.abs(ref[name].to_numpy(float) - q[Y].to_numpy(float)).max())
            print(name, "max abs diff to", os.environ["CHAIN_REF"], diff)
            assert diff < tol
            continue
        assert got == want
    for split, d in result.items():
        d.to_parquet(out / f"{split}.parquet", index=False)
    wcol = fold["w"].to_numpy(); dd = (fold.y - fold.zabd) ** 2 - (fold.y - fold.v64) ** 2
    print("held G v64 vs zabd", float((wcol * dd).sum()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.data_root.resolve(), a.out)
