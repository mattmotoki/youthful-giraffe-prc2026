"""exp-055, before the registration: the byte-hash census of every frozen source.  NO VALUE READ.

The assignment (`assignments/origin.md`, "Frozen objects") allows schemas and byte hashes
before the iteration-1 registration, and forbids reading `y_true` or parsing a stored loss
field.  This script therefore only

  * hashes files as bytes (SHA-256, streamed);
  * parses the exp-051 fit receipts (`analysis/fits/*.json`, `analysis/fits2/*.json`), the
    integrity receipts, `plan.json` and `populations-rerun-check.json`, which carry rows,
    provenance and hashes but no loss; and reads their recorded hash fields only;
  * reads parquet schemas and row counts from the file footers (no column values).

`analysis/boardlike.json`, `analysis/carrier.json` and `analysis/populations.json` of exp-051
carry loss fields: they are hashed as bytes and NOT parsed.  Q15's `pin.json` and
`routec.json` are hashed as bytes here; their hash keys are compared later by `pin.py`.

Output: `analysis/prehash.json` (write-once).  A recorded hash that the bytes do not match is
fatal.

    .venv/bin/python experiments/exp-055-originmove/code/prehash.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
E46 = ROOT / "experiments/exp-046-surfacefit"
E50 = ROOT / "experiments/exp-050-nmblind"
E51 = ROOT / "experiments/exp-051-transferadv"
E53 = ROOT / "experiments/exp-053-transport"
W51 = ROOT / "artifacts/exp-051/work"
sys.path.insert(0, str(E46 / "code"))
import nooverwrite as NO                                              # noqa: E402

OUT = EXP / "analysis/prehash.json"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    return str(Path(p).relative_to(ROOT))


def member_path(L: int, s: int) -> Path:
    """exp-051 `memberspread.member_path`, copied so that no exp-051 module is imported."""
    if L == 0:
        return E46 / f"work/d-bundle-placebo{s}-255-free.npz"
    if s == 1 and L in (1, 2):
        return {1: E46 / "work/d-onemonth-screen-255-free.npz",
                2: E46 / "work/d-onemonth-twomonth-plus03-255-free.npz"}[L]
    return E46 / "work" / {1: f"d-onemonth-m{s}-255-free.npz", 2: f"d-threemonth-m{s}-255-free.npz",
                           3: f"d-seasonal-m{s}-255-free.npz"}[L]


def main() -> None:
    NO.refuse_existing([OUT], overwrite=False)
    t0 = time.time()
    rec = {"what": __doc__.strip().splitlines()[0], "files": {}, "mismatches": []}

    def put(key: str, p: Path, recorded: str | None, source: str | None) -> None:
        got = sha(p)
        ent = {"path": rel(p), "sha256": got, "bytes": Path(p).stat().st_size}
        if recorded is not None:
            ent["recorded_sha256"] = recorded
            ent["recorded_in"] = source
            ent["matches_record"] = got == recorded
            if got != recorded:
                rec["mismatches"].append(key)
        rec["files"][key] = ent

    # ---- exp-051 OOF receipts: recorded in populations-rerun-check.json (hashes only)
    chk = json.loads((E51 / "analysis/populations-rerun-check.json").read_text())
    for name, v in chk["oof"].items():
        put(f"e51_oof_{name}", E51 / "oof" / name, v["committed_sha256"],
            "experiments/exp-051-transferadv/analysis/populations-rerun-check.json")
        ent = rec["files"][f"e51_oof_{name}"]
        md = pq.read_metadata(E51 / "oof" / name)
        ent["rows"] = md.num_rows
        ent["columns"] = pq.read_schema(E51 / "oof" / name).names
    # ---- exp-051 loss records: bytes only, never parsed before registration 2
    for name in ("boardlike.json", "carrier.json", "populations.json", "populations-rerun.json"):
        put(f"e51_{name}", E51 / "analysis" / name, None, None)
    # ---- exp-051 fit receipts and the vectors they cite
    for d in ("fits", "fits2"):
        for js in sorted(glob.glob(str(E51 / "analysis" / d / "*.json"))):
            r = json.loads(Path(js).read_text())
            put(f"e51_receipt_{Path(js).stem}", Path(js), None, None)
            put(f"e51_vector_{Path(js).stem}", ROOT / r["vector"], r["vector_sha256"], rel(Path(js)))
    for tag in ("integrity-C0-s1", "integrity-P1-s2"):
        js = E51 / f"analysis/i2-{tag}.json"
        r = json.loads(js.read_text())
        put(f"e51_integrity_receipt_{tag}", js, None, None)
        put(f"e51_integrity_refit_{tag}", W51 / f"i2-{tag}.npz", r["refit_sha256"], rel(js))
        put(f"e51_integrity_reference_{tag}", ROOT / r["reference"], r["reference_sha256"], rel(js))
    # ---- plain members s = 1..3, levels 0..3 (member 1 recorded in exp-051 plan.json)
    plan = json.loads((E51 / "analysis/plan.json").read_text())
    for s in (1, 2, 3):
        for L in range(4):
            p = member_path(L, s)
            recd = plan["plain"][str(L)]["sha256"] if s == 1 else None
            put(f"plain_member_s{s}_L{L}", p, recd,
                "experiments/exp-051-transferadv/analysis/plan.json" if s == 1 else None)
    put("e51_plan.json", E51 / "analysis/plan.json", None, None)
    put("e51_populations-rerun-check.json", E51 / "analysis/populations-rerun-check.json", None, None)
    # ---- Q15 (exp-053) sources, SPEC section 2 values
    q15 = {
        "panel_july_loo": (E50 / "work/panel-v22-july_loo.parquet",
                           "fc8700e9910f03e7f353624cc2d22bf252b81105032e905e6339bb5930b3630c"),
        "panel_december": (E50 / "work/panel-v22-december.parquet",
                           "8b3d855ca727f88e441063bf18eaa93213e5d081fda1bc7606c4ba06889e4846"),
        "panel_january": (E50 / "work/panel-v22-january.parquet",
                          "4672b7b3784b57f3903b5e58a6dd84ffa8ce67c86989874974d6fc59c8be3d62"),
        "head_v19": (E46 / "work/d-bag10-255-free.npz",
                     "d32a993b48d18fa428cb6c0127de34a5ccb0ce5ad09d2dd0a3e89db7a35cd26e"),
        "head_v20": (E46 / "work/d-onemonth-bag10-255-free.npz",
                     "e1af652c0e5b07a8786532005e1854129470138a07f0999c38d002d6559a871e"),
        "head_v21": (E46 / "work/d-threemonth-bag10-255-free.npz",
                     "1e0d0085b6ecea726e8da603554bfff95eef85e1afc3bc17a703de59d7b627c6"),
        "head_v22": (E46 / "work/d-seasonal-bag10-255-free.npz",
                     "3edfa35fee55e1d03ec2df72192eeb596732e62f8164b162e9ef42597ecea6f5"),
        "ranking": (ROOT / "data/raw/ranking.parquet",
                    "9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86"),
        "upload_v19": (ROOT / "submissions/uploaded/youthful-giraffe_v19.parquet",
                       "d03d132a5ff5d3fc70a72b9f60a1b418a169e911ac32ef08b14378695932aa14"),
        "upload_v20": (ROOT / "submissions/uploaded/youthful-giraffe_v20.parquet",
                       "8fd4b34166b61756a20746c77f97fc1d06d06af28ee959ef7deac1de0a365d78"),
        "upload_v21": (ROOT / "submissions/uploaded/youthful-giraffe_v21.parquet",
                       "7b0557a497f8c09e4f56a11352783b9dda37fe1d0579345d4303e4f74d6e7aca"),
        "upload_v22": (ROOT / "submissions/uploaded/youthful-giraffe_v22.parquet",
                       "f11d34f88b62a43ec2a0cdbeec967a9ee4b7d92e50a5508aa4d30e5a78ea5e1c"),
    }
    cls = {"v22": ("4c9a91b6cd760277445ab78d80dc131287feec9ffd6e0a72c7a71c22d556f97f",
                   "a3a4aa3be298240d7b93bc0d723f83e5d902f25f146340d1acdc799fd28869be"),
           "v23": ("8cef2f2d8175be9c846650dc494439b1cda2f2c94d154d5bcf269addd6e23cb2",
                   "e69faf3a46e8ea61ad71dc6cb3b74e74ab3add1720b84524015eb45f8a2bca3f"),
           "v24": ("e6be6c0fdedeb5807f7ea99c57a3037e36a2ba66b4fbaeb1db191f14df801465",
                   "0db0de12d33cb39b9d40351a094fe8a6838a78341902a2abc9a9696292809b15"),
           "v25": ("e69f78b4f7d17fa3c0c95151fc171f2c7cd9022efd79489add1bbce119494697",
                   "4eee477d219d32fa2ebb31346dbb105af0736e37ceaf786ec0eb0d0906be13fe")}
    for v, (s01, s07) in cls.items():
        q15[f"class_{v}_2026-01"] = (ROOT / f"artifacts/exp-046/class/2026-01-{v}.parquet", s01)
        q15[f"class_{v}_2026-07"] = (ROOT / f"artifacts/exp-046/class/2026-07-{v}.parquet", s07)
    for k, (p, want) in q15.items():
        put(f"q15_{k}", p, want, "experiments/exp-053-transport/SPEC.md section 2 / analysis/pin.json")
    for k in ("panel_july_loo", "panel_december", "panel_january", "ranking"):
        p = q15[k][0]
        rec["files"][f"q15_{k}"]["rows"] = pq.read_metadata(p).num_rows
        rec["files"][f"q15_{k}"]["columns"] = pq.read_schema(p).names
    # Q15 records, bytes (RESULTS.md prints 92f70c73... and dd554a70...)
    put("q15_pin.json", E53 / "analysis/pin.json", None, None)
    put("q15_routec.json", E53 / "analysis/routec.json", None, None)
    for k, pref in (("q15_pin.json", "92f70c73"), ("q15_routec.json", "dd554a70")):
        rec["files"][k]["results_md_prefix"] = pref
        rec["files"][k]["matches_results_md_prefix"] = rec["files"][k]["sha256"].startswith(pref)
        if not rec["files"][k]["matches_results_md_prefix"]:
            rec["mismatches"].append(k)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["n_files"] = len(rec["files"])
    NO.refuse_existing([OUT], overwrite=False)
    OUT.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(f"{rec['n_files']} files hashed; mismatches: {rec['mismatches'] or 'none'} "
          f"({rec['seconds']} s) -> {rel(OUT)}")
    if rec["mismatches"]:
        raise SystemExit(f"recorded hash mismatch: {rec['mismatches']}")


if __name__ == "__main__":
    main()
