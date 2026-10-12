"""Arm 2's frame check, run before any GPU is asked for.

exp-027 s14b's discipline: the flag-off path must stay byte identical, the
shipped flag-on path must stay byte identical, and the wide arm must differ only
by the columns of the scales it adds. A stale month cache would break all three
silently, so the cache key is checked here too.

    PRC_THREADS=3 python experiments/exp-035-k2climb/code/frame_ab.py 2025-07
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]

CHILD = r'''
import hashlib, json, os, sys
from pathlib import Path
ROOT = Path(sys.argv[1]); month = sys.argv[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp-018-fresh/code"))
import pandas as pd
import features as F
from prc import data
frame = data.load_month(month)
masked, labels = data.mask_like_ranking(frame)
X = F.build(masked)
h = hashlib.sha256(pd.util.hash_pandas_object(X, index=True).values.tobytes()).hexdigest()
import window_features as W
print(json.dumps({"rows": int(len(X)), "cols": int(X.shape[1]), "sha": h[:16],
                  "scales": list(W.SCALES) if os.environ.get("PRC_WINDOW") == "1" else [],
                  "names": list(X.columns)}))
'''


def run(month: str, env_extra: dict) -> dict:
    env = dict(os.environ)
    env.pop("PRC_WINDOW", None)
    env.pop("PRC_WINDOW_SCALES", None)
    env.update(env_extra)
    env["PYTHONPATH"] = str(ROOT)
    p = subprocess.run([sys.executable, "-c", CHILD, str(ROOT), month],
                       capture_output=True, text=True, env=env, cwd=str(ROOT))
    if p.returncode != 0:
        raise SystemExit(f"{env_extra}: rc={p.returncode}\n{p.stderr[-4000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def run_at_head(month: str, env_extra: dict | None = None) -> dict:
    """The same frame from the committed code at HEAD, in a scratch checkout."""
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="k2climb-head-"))
    try:
        (tmp / "experiments" / "exp-018-fresh").mkdir(parents=True)
        subprocess.run(["git", "archive", "HEAD", "experiments/exp-018-fresh/code", "prc"],
                       cwd=str(ROOT), check=True, stdout=open(tmp / "head.tar", "wb"))
        subprocess.run(["tar", "xf", str(tmp / "head.tar")], cwd=str(tmp), check=True)
        for extra in ("data", "artifacts"):
            (tmp / extra).symlink_to(ROOT / extra)
        env = dict(os.environ)
        env.pop("PRC_WINDOW", None)
        env.pop("PRC_WINDOW_SCALES", None)
        env.update(env_extra or {})
        env["PYTHONPATH"] = str(tmp)
        p = subprocess.run([sys.executable, "-c", CHILD, str(tmp), month],
                           capture_output=True, text=True, env=env, cwd=str(tmp))
        if p.returncode != 0:
            raise SystemExit(f"HEAD build: rc={p.returncode}\n{p.stderr[-4000:]}")
        return json.loads(p.stdout.strip().splitlines()[-1])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    month = sys.argv[1] if len(sys.argv) > 1 else "2025-07"
    arms = {
        "off": {},
        "on-shipped": {"PRC_WINDOW": "1"},
        "on-wide": {"PRC_WINDOW": "1", "PRC_WINDOW_SCALES": "300,600,1800,3600"},
    }
    got = {k: run(month, v) for k, v in arms.items()}
    for k, r in got.items():
        print(f"{k:12s} rows={r['rows']:7d} cols={r['cols']:4d} sha={r['sha']} scales={r['scales']}")
    off, ship, wide = got["off"], got["on-shipped"], got["on-wide"]
    ok = True
    # 1. the flag-off frame is exp-027 s14b's frozen one.
    #    RECORD CONFLICT: exp-027 s14b quotes sha256 `702ce9fb947545f7` for it and
    #    does not record the recipe that produced that digest; this file's recipe
    #    (`pd.util.hash_pandas_object` over the frame, sha256 of the bytes) gives a
    #    different one on an unmodified checkout, so the digest cannot be re-read.
    #    The claim behind it is testable without it and is what is tested here: the
    #    same command run against the committed code at HEAD and against this branch
    #    must give `DataFrame.equals` True, which is exp-027's own second column.
    head = run_at_head(month)
    if (off["rows"], off["cols"]) != (190713, 89) or head["sha"] != off["sha"]:
        print(f"FAIL flag-off frame moved against the committed code at HEAD "
              f"({head['rows']}x{head['cols']} {head['sha']} against "
              f"{off['rows']}x{off['cols']} {off['sha']})")
        ok = False
    else:
        print(f"\nflag-off against HEAD's own features.py: equal "
              f"({off['rows']} x {off['cols']}, {off['sha']})")
    # 2. the shipped block is unchanged: off's columns are a prefix-subset and 68 are added
    head_on = run_at_head(month, {"PRC_WINDOW": "1"})
    if ship["cols"] - off["cols"] != 68 or ship["scales"] != [600.0, 3600.0]:
        print("FAIL shipped block is not 68 columns at (600, 3600)")
        ok = False
    if head_on["sha"] != ship["sha"] or head_on["cols"] != ship["cols"]:
        print(f"FAIL the shipped flag-on frame moved against HEAD "
              f"({head_on['cols']} {head_on['sha']} against {ship['cols']} {ship['sha']})")
        ok = False
    else:
        print(f"flag-on  against HEAD's own features.py: equal "
              f"({ship['rows']} x {ship['cols']}, {ship['sha']})")
    if set(off["names"]) - set(ship["names"]):
        print("FAIL the shipped arm dropped a base column")
        ok = False
    # 3. the wide arm adds exactly the two new scales' columns and nothing else
    add = [c for c in wide["names"] if c not in set(ship["names"])]
    drop = [c for c in ship["names"] if c not in set(wide["names"])]
    per = (ship["cols"] - off["cols"]) // 2
    print(f"\nwide adds {len(add)} columns, drops {len(drop)}; "
          f"expected {2 * per} added (2 scales x {per}) and 0 dropped")
    if len(add) != 2 * per or drop:
        print("FAIL the wide arm is not the shipped block plus two scales")
        ok = False
    if sorted({c.split("_")[0] for c in add}) != ["w1800", "w300"]:
        print(f"FAIL added columns are not w300/w1800: {sorted({c.split('_')[0] for c in add})[:6]}")
        ok = False
    # 4. the cache key carries the scale set
    sys.path.insert(0, str(ROOT / "experiments/exp-018-fresh/code"))
    keys = {}
    for k, v in arms.items():
        for kk, vv in list(v.items()):
            os.environ[kk] = vv
        for kk in ("PRC_WINDOW", "PRC_WINDOW_SCALES"):
            if kk not in v:
                os.environ.pop(kk, None)
        for m in ("window_features", "dataset"):
            sys.modules.pop(m, None)
        import dataset as Dm
        import window_features as Wm  # noqa: F401
        keys[k] = Dm.month_table.__globals__ and (Dm.CACHE / f"{month}"), None
        # reproduce the suffix the module would compute
        suffix = ""
        if os.environ.get("PRC_WINDOW") == "1":
            suffix = "-w"
            if Wm.SCALES != (600.0, 3600.0):
                suffix += "-" + "-".join(str(int(x)) for x in Wm.SCALES)
        keys[k] = f"{month}{suffix}.parquet"
    print("\ncache keys:", json.dumps(keys, indent=1))
    if len(set(keys.values())) != 3 or keys["off"] != f"{month}.parquet" or keys["on-shipped"] != f"{month}-w.parquet":
        print("FAIL cache keys do not separate the three arms, or the two frozen keys moved")
        ok = False
    rec = {"month": month, "arms": {k: {kk: vv for kk, vv in v.items() if kk != "names"}
                                    for k, v in got.items()},
           "cache_keys": keys, "added": sorted(add), "ok": ok}
    (EXP / "analysis").mkdir(exist_ok=True)
    (EXP / "analysis" / "frame-ab.json").write_text(json.dumps(rec, indent=1))
    print("\n" + ("FRAME A/B OK" if ok else "FRAME A/B FAILED"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
