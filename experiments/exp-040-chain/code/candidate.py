"""The candidate the desk named: `v11 + (H' - v10)`, and everything that gates it.

    .venv/bin/python experiments/exp-040-chain/code/candidate.py --arm published  # self-test
    .venv/bin/python experiments/exp-040-chain/code/candidate.py --arm chain

`H'` is the **regime-held head**: the chain member's served head on `nm_bulk` and
`nm_late`, v10 everywhere else.  `code/regime_gate.py` measured it and it is the
only arm in this experiment that meets the assignment's advance clause; the hold
was named after reading section 4b's slice table, so it is a post-hoc rule and
the record says so beside december's `+0.0025` lower bound.

Adding `H' - v10` to v11 rather than serving `H'` itself keeps v11's own increment
(`v10 + 0.217*(stack-5 - v10)`, board 263.0298, the best file we hold) and moves
only the rows the chain member moves.  Every row where `H' = v10` -- `nonm`,
`noac`, EHAM, and EDDF, which is outside exp-036's eight-airport mask -- is
therefore v11 byte for byte, and that is checked here at tolerance 0, not argued.

The heads come from `code/serve_arm.py`, whose `--arm published` gate reproduces
v10's SHA-256 exactly, so a head is the member change and nothing else.

`--arm published` runs this file's own arithmetic on that gated head, where
`H' = v10` on every row by construction, so the result must be **v11 byte for
byte**.  That is the self-test for everything below the head: the regime join, the
`np.where`, the floor and the writer.  It writes into the experiment, never into
`submissions/candidates`.

Writes `submissions/candidates/exp-040-chain.parquet` and `analysis/candidate.json`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-026-rebuild/code"))
import regime_table as RT                                     # noqa: E402
sys.path.insert(0, str(ROOT / "experiments/exp-038-packet/code"))
import align as A                                             # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D                                          # noqa: E402

FOLDS = A.FOLDS
#: The hold `code/regime_gate.py` chose, on every fold, both in sample and out.
HELD = ("nm_bulk", "nm_late")
FLOOR = 60.0
V10 = (ROOT / "submissions/uploaded/youthful-giraffe_v10.parquet",
       "ad962763236e9b40ace45a5dac12945568bbce441ed0547ff8e814d248c72c78")
V11 = (ROOT / "submissions/uploaded/youthful-giraffe_v11.parquet",
       "42ca5ee6410938d56c8830e9ec3980d806633c3b2ff15963c0ffe2fe97c39f8c")
PUB_RANK_SHA = "dd0e8710000ab29b930ec72d8b058b00e6ed86ba3240cad01402c13813527882"
ARM_RANK = "experiments/exp-040-chain/oof/{arm}--ranking.parquet"
PUBLISHED_RANK = "experiments/exp-012-blend2/published/L2-hub8725-ranking.parquet"
OUT = ROOT / "submissions/candidates/exp-040-chain.parquet"
SELFTEST = EXP / "candidates" / "selftest-published.parquet"


def column(path: Path, want: str | None, tmpl: np.ndarray) -> np.ndarray:
    got = D.sha256(path)
    if want and got != want:
        raise SystemExit(f"{path.name} has changed: {got[:8]} against {want[:8]}")
    d = pd.read_parquet(path)
    if not np.array_equal(d["MVT_ID_mvt"].to_numpy("float64"), tmpl):
        raise SystemExit(f"{path.name} is not in the template's order")
    return d[D.TARGET].to_numpy("float64")


def by(key: np.ndarray, dif: np.ndarray) -> dict:
    """Changed rows and their RMS, per value of `key`."""
    out = {}
    for k in sorted(set(key.tolist())):
        m = key == k
        out[str(k)] = {"rows": int(m.sum()), "changed": int((dif[m] != 0).sum()),
                       "rms": float(np.sqrt((dif[m] ** 2).mean())),
                       "max_abs": float(np.abs(dif[m]).max())}
    return out


def fold_update_rms(arm: str) -> dict:
    """The same update on the fold side, for THIS arm (Sol round 2, blocking 3)."""
    out = {}
    col = f"{arm} d11-noac"
    for f in FOLDS:
        d = pd.read_parquet(EXP / "packet" / f"q09-{f}.parquet")
        if col not in d.columns:
            raise SystemExit(f"packet/q09-{f}.parquet has no column {col!r}")
        on = d["regime"].isin(list(HELD)).to_numpy()
        u = np.where(on, d[col].to_numpy("float64")
                     - d["v10 replay"].to_numpy("float64"), 0.0)
        out[f] = {"rows_changed": int((u != 0).sum()),
                  "rms_changed": float(np.sqrt((u[u != 0] ** 2).mean())),
                  "rms_all": float(np.sqrt((u ** 2).mean()))}
    return out


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="chain", choices=["chain", "derived", "published"])
    a = ap.parse_args()
    # Sol round 2, blocking 3.  `--arm derived` was accepted although
    # `code/regime_gate.py` scored only the chain arm, and the one-third test
    # always used the chain fold update -- so a derived candidate could have been
    # written under a rotation that never saw it.  The arm the regime gate
    # actually scored is read from its record and is the only arm that may ship.
    scored = None
    if a.arm != "published":
        rgate = EXP / "analysis" / "regime-gate.json"
        if not rgate.exists():
            raise SystemExit("analysis/regime-gate.json is absent; run code/regime_gate.py")
        rec_g = json.loads(rgate.read_text())
        scored = str(rec_g["arm"]).split()[0]          # e.g. "chain d11-noac" -> "chain"
        if scored != a.arm:
            raise SystemExit(f"the regime gate scored {scored!r}, not {a.arm!r}; no "
                             f"candidate may be written for an arm no rotation priced")
        if list(rec_g["regimes"]) != ["nm_bulk", "nm_late", "nonm"] or \
                rec_g["held_out_choice"]["pick_july_loo"] != "+".join(sorted(HELD)):
            raise SystemExit(f"the regime gate's held-out choice is not {HELD}: "
                             f"{rec_g['held_out_choice']['pick_july_loo']}")
    out = SELFTEST if a.arm == "published" else OUT
    head_rec = EXP / "analysis" / f"head-{a.arm}.json"
    if not head_rec.exists():
        raise SystemExit(f"analysis/head-{a.arm}.json is absent; run "
                         f"code/serve_arm.py --arm {a.arm} first")
    gate = json.loads((EXP / "analysis" / "head-published.json").read_text())
    if not gate.get("gate_pass"):
        raise SystemExit("the serving-path gate has not passed; no head is valid")
    # Sol round 1, blocking 4.  The first version recorded these and wrote the
    # file anyway, so a later run could have emitted a candidate whose ranking
    # member was not the machine term but a different fit.  They are gates now.
    rg_path = EXP / "analysis" / "rank-gate.json"
    if not rg_path.exists():
        raise SystemExit("analysis/rank-gate.json is absent; run code/rank_gate.py first")
    rg = json.loads(rg_path.read_text())
    if not rg.get("gate", {}).get("explained"):
        raise SystemExit(f"the ranking member's offset is not explained by the machine "
                         f"term: {rg.get('gate')}")
    if rg["committed"]["sha256"] != PUB_RANK_SHA:
        raise SystemExit("rank-gate.json was written against a different committed member")
    # Sol round 2, blocking 4.  The checks above read the RECORD.  These hash the
    # bytes on disk now, so a file changed under the same path since the gate ran
    # cannot pass.
    now = D.sha256(ROOT / PUBLISHED_RANK)
    if now != PUB_RANK_SHA:
        raise SystemExit(f"the committed ranking member on disk is {now[:8]}, not "
                         f"{PUB_RANK_SHA[:8]}")
    if a.arm != "published":
        arm_rank = ROOT / ARM_RANK.replace("{arm}", a.arm)
        got = D.sha256(arm_rank)
        want = rg["machine_term_ranking"].get(a.arm, {}).get("sha256")
        if want is None:
            raise SystemExit(f"rank-gate.json does not cover the {a.arm} ranking member")
        if got != want:
            raise SystemExit(f"{arm_rank.name} is {got[:8]} but the ranking gate was run "
                             f"on {want[:8]}; rerun code/rank_gate.py")
    hrec = json.loads(head_rec.read_text())
    want_rank = (PUBLISHED_RANK if a.arm == "published"
                 else ARM_RANK.replace("{arm}", a.arm))
    if hrec["member_ranking"] != want_rank:
        raise SystemExit(f"head-{a.arm}.json was built from {hrec['member_ranking']}, "
                         f"not {want_rank}")

    tmpl = D.load_submitting()["MVT_ID_mvt"].to_numpy().astype("float64")
    v10 = column(*V10, tmpl)
    v11 = column(*V11, tmpl)
    head = column(EXP / "candidates" / f"head-{a.arm}.parquet", hrec["sha256"], tmpl)

    if not RT.OUT.exists():
        RT.build()
    gates_sha = D.sha256(RT.OUT)
    g = RT.load().set_index("MVT_ID_mvt").reindex(tmpl)
    reg = g["regime"].astype(str).to_numpy()
    apt = g["airport"].astype(str).to_numpy()
    if pd.isna(g["regime"]).any() or pd.isna(g["airport"]).any():
        raise SystemExit("a ranking row is missing from the gates table")

    on = np.isin(reg, list(HELD))
    #: Sol round 1, blocking 4.  The record's three file hashes do not determine
    #: the file on their own: which rows are served comes from the regime table,
    #: which is a build artefact and was not pinned.  Both the table's SHA-256 and
    #: a hash of the served mask itself go into the record, and the rebuild below
    #: reloads the table rather than reusing this array.
    mask_sha = hashlib.sha256(np.packbits(on).tobytes()).hexdigest()
    hp = np.where(on, head, v10)                       # H', the regime-held head
    upd = hp - v10
    raw = v11 + upd
    cand = np.maximum(raw, FLOOR)
    bound = int((cand != raw).sum())

    dif = cand - v11
    held = ~ (dif != 0)
    rec = {
        "arm": a.arm,
        "rule": f"v11 + (H' - v10), H' = {a.arm} head on " + "+".join(HELD) + ", v10 elsewhere",
        "held_regimes": list(HELD), "floor": FLOOR, "floor_bound_rows": bound,
        "regime_table": {"path": str(RT.OUT.relative_to(ROOT)), "sha256": gates_sha},
        "served_mask_sha256": mask_sha, "served_mask_rows": int(on.sum()),
        "rank_gate": rg["gate"],
        "sources": {"v10": {"path": str(V10[0].relative_to(ROOT)), "sha256": V10[1]},
                    "v11": {"path": str(V11[0].relative_to(ROOT)), "sha256": V11[1]},
                    "head": {"path": hrec["head"], "sha256": hrec["sha256"],
                             "member_ranking": hrec["member_ranking"]}},
        "serving_path_gate": {"published_head_sha256": gate["sha256"],
                              "v10_sha256": gate["gate_v10_sha256"],
                              "pass": gate["gate_pass"]},
        "rows": int(len(tmpl)), "rows_changed_vs_v11": int((dif != 0).sum()),
        "rows_identical_to_v11": int(held.sum()),
        "rms_vs_v11": float(np.sqrt((dif ** 2).mean())),
        "rms_changed_vs_v11": (float(np.sqrt((dif[dif != 0] ** 2).mean()))
                              if (dif != 0).any() else 0.0),
        "max_abs_vs_v11": float(np.abs(dif).max()),
        "min": float(cand.min()), "max": float(cand.max()),
        "finite": bool(np.isfinite(cand).all()),
        "by_regime": by(reg, dif), "by_airport": by(apt, dif),
        "fold_side_update": fold_update_rms(a.arm if a.arm != "published" else "none"),
        "regime_gate_arm": scored,
    }
    # the held set, stated as a set and not as a count
    for nm, m in (("nonm", reg == "nonm"), ("noac", reg == "noac"),
                  ("EHAM", apt == "EHAM"), ("EDDF", apt == "EDDF")):
        rec[f"held_{nm}_rows"] = int(m.sum())
        rec[f"held_{nm}_max_abs_vs_v11"] = float(np.abs(dif[m]).max()) if m.any() else 0.0
    off = ~on
    if float(np.abs(hp[off] - v10[off]).max()) != 0.0:
        raise SystemExit("H' differs from v10 on a regime it holds")
    # the served update against the fold side's, the desk's one-third rule
    fs = rec["fold_side_update"]
    served = rec["rms_changed_vs_v11"]
    folds = [fs[f]["rms_changed"] for f in FOLDS]
    ref = float(np.mean(folds))
    rec["update_rms_served"] = served
    rec["update_rms_fold_range"] = [min(folds), max(folds)]
    rec["update_rms_fold_mean"] = ref
    #: the desk's rule: the served update must not be a different size from the
    #: one that was priced.  Within a third of it, either way.
    rec["update_rms_ratio"] = served / ref
    rec["update_rms_within_third"] = bool(abs(served - ref) <= ref / 3.0)

    if a.arm != "published" and not rec["update_rms_within_third"]:
        raise SystemExit(f"the served update's RMS {served:.4f} is not within a third of "
                         f"the fold side's {ref:.4f}; no candidate is written")
    rec["sha256"] = D.write_submission(pd.Series(cand, index=tmpl), out)
    rec["path"] = str(out.relative_to(ROOT))
    if a.arm == "published":
        rec["selftest_v11_sha256"] = V11[1]
        rec["selftest_pass"] = bool(rec["sha256"] == V11[1])
        print(f"[selftest] {rec['sha256'][:8]}… against v11 {V11[1][:8]}…  "
              f"{'PASS' if rec['selftest_pass'] else 'FAIL'}", flush=True)
    rec["wall_s"] = round(time.time() - t0, 1)

    # An independent rebuild from the record's three hashes, at tolerance 0.
    # The parentheses are part of the rule, not a detail: `v11 + (H' - v10)`
    # makes the update EXACTLY 0.0 wherever `H' = v10`, so those rows come back
    # as v11's own bits.  `(v11 + H') - v10` is the same real number and not the
    # same float, and it would move every held row by an ulp.
    _g = RT.load(RT.OUT)
    if D.sha256(RT.OUT) != rec["regime_table"]["sha256"]:
        raise SystemExit("the regime table changed under the rebuild")
    _reg = _g.set_index("MVT_ID_mvt").reindex(tmpl)["regime"].astype(str).to_numpy()
    _on = np.isin(_reg, list(HELD))
    if hashlib.sha256(np.packbits(_on).tobytes()).hexdigest() != rec["served_mask_sha256"]:
        raise SystemExit("the served mask does not rebuild from the recorded table")
    _v11 = column(V11[0], rec["sources"]["v11"]["sha256"], tmpl)
    _v10 = column(V10[0], rec["sources"]["v10"]["sha256"], tmpl)
    _h = np.where(_on, column(EXP / "candidates" / f"head-{a.arm}.parquet",
                              rec["sources"]["head"]["sha256"], tmpl), _v10)
    back = np.maximum(_v11 + (_h - _v10), FLOOR)
    rec["rebuild_max_abs"] = float(np.abs(back - cand).max())
    if rec["rebuild_max_abs"] != 0.0:
        raise SystemExit("the file does not rebuild from its own record")

    (EXP / "analysis" / f"candidate-{a.arm}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: v for k, v in rec.items()
                      if k not in ("by_airport", "by_regime")}, indent=2))
    print("\nby regime:  " + "  ".join(
        f"{k} {v['changed']}/{v['rows']} rms {v['rms']:.3f}" for k, v in rec["by_regime"].items()))
    print("by airport: " + "  ".join(
        f"{k} {v['changed']}/{v['rows']} rms {v['rms']:.3f}" for k, v in rec["by_airport"].items()))
    print(f"\nwrote {out.relative_to(ROOT)}  sha256 {rec['sha256']}")
    if a.arm == "published" and not rec["selftest_pass"]:
        raise SystemExit("the arithmetic below the head does not return v11")


if __name__ == "__main__":
    main()
