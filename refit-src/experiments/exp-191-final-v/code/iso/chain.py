"""The v64 chain as a reusable function, with an optional replacement of the primary blend.

load() rebuilds the zabd contexts once (about 6 s). run(primary=None) returns, for the fold
and ranking splits, every stage up to v64. A replacement primary is a dict split -> array; the
downstream chain (no-plan form, practice, C73, C33, C55, the C71/C95/C98 refit on the fold
pre_exploit, C66, C58, the bounds, T11b-L) runs unchanged on it.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "release"))
import assembly_replay as R  # noqa: E402
import t11b_stage as S  # noqa: E402

ID = R.ID
DATA = ROOT / "input-root"
_STATE = {}


def load():
    if _STATE:
        return _STATE
    pack = json.loads((ROOT / "release/ZABD-PACK.json").read_text())
    official = R.official_tables(DATA, pack["official_sha256"])
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
    extras = {split: rd.aligned(extra[split], c["b"], c["scope"], year=2025 if split == "fold" else 2026)
              for split, c in contexts.items()}
    rframe = rd.frame(parent["common"]["rule_frame"]["fold"])
    exposure = rd.frame(parent["common"]["exposure"])
    fw, pp = T.fit(contexts["fold"], extras["fold"], rframe.y.to_numpy(), exposure.weight.to_numpy())
    # T11b-L inputs
    t11 = {}
    for split in ("fold", "ranking"):
        b = contexts[split]["b"]
        off = official[split].set_index(ID).reindex(b[ID].to_numpy())
        if split == "ranking":
            raw = pd.read_parquet(DATA / R.OFFICIAL_FILES["ranking"], columns=S.COLS)
        else:
            raw = pd.concat([pd.read_parquet(DATA / R.OFFICIAL_FILES[k], columns=S.COLS)
                             for k in ("training_2025_01", "training_2025_07")], ignore_index=True)
        joined = off["LOBT_flt"].notna().to_numpy()
        typed = off["AIRCRAFT_TYPE_mvt"].notna().to_numpy() & (b.cls.to_numpy() != "noac")
        _, g = S.flags(raw)
        days = g[g.snow]
        keys = set(days.ap + "|" + days.dstr)
        t11[split] = {"p2": typed & ~joined, "long": S.row_keys(b[ID].to_numpy(), raw).isin(keys).to_numpy()}
    _STATE.update(A=A, T=T, rd=rd, parent=parent, contexts=contexts, pw=pw, extras=extras, fw=fw, pp=pp,
                  rframe=rframe, exposure=exposure, t11=t11)
    return _STATE


def run(primary: dict | None = None, after_primary=None) -> dict:
    """Return {split: dict of stage arrays}; stage 'v64' is the final value."""
    st = load()
    A, T, rd, parent, contexts = st["A"], st["T"], st["rd"], st["parent"], st["contexts"]
    import stack7  # noqa: E402  (on sys.path through the assembly modules)
    rd.zero_order_delta = frozenset()
    stages = {}
    for split, c in contexts.items():
        stages[split], _ = T.pre(c, st["extras"][split], st["pw"], st["fw"], st["pp"], split)
        if primary is not None:
            new = np.asarray(primary[split], float).copy()
            stages[split] = copy.copy(stages[split])
            stages[split]["primary"] = new
            stages[split]["pre_exploit"] = stack7.independent_tail(c["b"], new)
    A.rules(parent, contexts, stages, rd)
    rd.zero_order_delta = frozenset(parent["common"]["exploit_frame"][s]["path"] for s in ("fold", "ranking"))
    A.tail(parent, contexts, stages, rd)
    out = {}
    for split, c in contexts.items():
        ef = rd.frame(parent["common"]["exploit_frame"][split])
        lower, upper = ef.lower.to_numpy(float), ef.upper.to_numpy(float)
        base = np.asarray(stages[split]["final"], float)
        t = st["t11"][split]
        v64 = S.apply(base, st["extras"][split], lower, upper, t["p2"], t["long"])
        res = {k: np.asarray(v, float) for k, v in stages[split].items()}
        res["v64"] = v64
        res["lower"], res["upper"] = lower, upper
        out[split] = res
    return out
