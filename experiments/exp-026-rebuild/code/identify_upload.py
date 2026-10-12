"""Which candidate is the object on the board?

`youthful-giraffe_v2` scored 265.9631 and nobody logged what was uploaded
(`docs/SUBMISSIONS.md` row 2, `internal decision record` recorded rule).  Given the downloaded
object, a SHA-256 match against `KNOWN` names the file outright and settles it.

**Everything short of a hash match is evidence, not proof.**  When no hash
matches, this reports the max and RMS absolute difference against every
candidate held, plus the recorded fingerprints.  Those narrow the field, and
the scale of a difference is informative -- a rebuilt-versus-shipped pair of the
same object differs by RMS ~8 s while a blend and a stack differ by ~29 s
(`docs/SUBMISSIONS.md`, rebuildability note) -- but a fingerprint is four
summary statistics of 344,841 numbers, so two different vectors can carry the
same one.  Review round 1: an earlier draft said a fingerprint match "settles"
the identity.  It does not; only the hash does.

    python identify_upload.py <downloaded.parquet> [more.parquet ...]

With no argument it compares everything under `submissions/uploaded/`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frames as F  # noqa: E402
sys.path.insert(0, str(F.ROOT))
import prc.data as D  # noqa: E402

#: Hashes recorded before the workstation was lost, so a match names the file
#: even though the file itself is gone.
KNOWN = {
    "5d6a99610907bc432e2e46fb9ea11fc8103feb8aa9efdd3f7c86f1188866a4cb":
        "exp-012-blend-5 (blend-config-5.json; D6 ordered it as v2)",
    "3b7e558a0b542690673f5ad01ade7a0c210a998905928e5e95434c752549337d":
        "exp-012-blend-4 (blend-config-4.json)",
    "48472e441288f0dfe33d35d5a18d31df786b1c4abe7ee06f9ef171ace2f0a6f0":
        "exp-016-stack-4 (uploaded as v1, board 265.7612)",
    "b6b864a21d73ba4ebc6d2293282eb1b1c898e501866726a7ee4c3b09cd261001": "exp-012-blend-3b",
    "26c40e69e0a8f3b018d2eb9b3c8d2c64e8ce7df5167d7aa194a038b997f9a231": "exp-008-blend3",
    "ee8fa94dfec2c06f6f2cc545c1cff1595981e418582c7c00d807baeb568c71d3": "exp-001-retro-v1-c9",
    "fa6195ffb21682e10f784d02af94e43e6236c9520e387335f2ae3dcbcda68201": "exp-001-retro-v1",
}

#: Objects whose full hash was never written down -- `docs/SUBMISSIONS.md` kept
#: only the prefix.  Review round 1: these were in KNOWN, where a full-hash
#: lookup could never reach them, so a real exp-018b upload would have been
#: reported as "none of the recorded hashes".
KNOWN_PREFIX = {
    "6be12ce6": "exp-018b-s1",
    "e71b297f": "exp-018-fresh",
}


#: Recorded fingerprints. These identify an object with NO candidate file on
#: disk, which matters because every candidate died with the workstation:
#: blend-5 and stack-4 differ by 5.6 s of median and 43 s of p99.
FINGERPRINT = {
    "exp-012-blend-5": dict(n=344841, min=60.0, median=962.0, p99=2507.3,
                            src="blend-config-5.json"),
    "exp-016-stack-4": dict(n=344841, min=60.0, median=956.415285929554,
                            p99=2550.363017392313, max=115010.35015942814,
                            src="exp-016-stack/candidate-4.json"),
}


def fingerprint(v: pd.Series) -> dict:
    return {"n": int(len(v)), "min": float(v.min()), "median": float(v.median()),
            "p99": float(v.quantile(0.99)), "max": float(v.max())}


def match_fingerprint(fp: dict) -> list[str]:
    """Names whose recorded fingerprint agrees with this one to 0.05 s."""
    out = []
    for name, want in FINGERPRINT.items():
        keys = [k for k in ("n", "min", "median", "p99", "max") if k in want]
        if all(abs(float(want[k]) - float(fp[k])) <= 0.05 for k in keys):
            out.append(f"{name} ({want['src']})")
    return out


def values(path: Path) -> pd.Series:
    d = pd.read_parquet(path)
    col = D.TARGET if D.TARGET in d.columns else "y_pred"
    return pd.Series(d[col].to_numpy("float64"), index=d[D.ID].to_numpy("float64"))


def references() -> dict[str, Path]:
    out = {}
    for root, tag in ((F.EXP / "candidates", "staged"),
                      (F.ROOT / "submissions" / "candidates", "candidate"),
                      (F.ROOT / "submissions" / "uploaded", "uploaded")):
        if root.exists():
            for p in sorted(root.glob("*.parquet")):
                if p.stem != "ranking-gates":
                    out[f"{tag}:{p.stem}"] = p
    return out


def compare(target: Path, refs: dict[str, Path]) -> None:
    sha = D.sha256(target)
    print(f"\n=== {target} ===")
    print(f"  sha256 {sha}")
    named = KNOWN.get(sha)
    if named is None:
        pre = KNOWN_PREFIX.get(sha[:8])
        named = f"{pre} (matched on the 8-char prefix only)" if pre else None
    print(f"  hash match: {named if named else 'none of the recorded hashes'}")
    t = values(target)
    fp = fingerprint(t)
    print(f"  n {fp['n']}  min {fp['min']:.4f}  median {fp['median']:.4f}  "
          f"p99 {fp['p99']:.4f}  max {fp['max']:.4f}")
    hit = match_fingerprint(fp)
    print(f"  fingerprint match: {', '.join(hit) if hit else 'none of the recorded fingerprints'}")
    rows = []
    for name, p in refs.items():
        if p.resolve() == target.resolve():
            continue
        try:
            r = values(p)
        except Exception as e:                                    # noqa: BLE001
            rows.append((name, float("nan"), float("nan"), 0, str(e)))
            continue
        common = t.index.intersection(r.index)
        d = t.reindex(common).to_numpy() - r.reindex(common).to_numpy()
        rows.append((name, float(np.abs(d).max()), float(np.sqrt((d ** 2).mean())),
                     int(len(common)), "" if len(common) == len(t) else "PARTIAL id overlap"))
    rows.sort(key=lambda r: (np.isnan(r[2]), r[2]))
    print(f"  {'reference':44s} {'max|d|':>12s} {'rms|d|':>12s} {'n':>8s}")
    for name, mx, rms, n, note in rows:
        print(f"  {name:44s} {mx:12.4f} {rms:12.4f} {n:8d}  {note}")
    if rows and rows[0][2] == 0.0:
        print(f"  IDENTICAL VALUES to {rows[0][0]}")


def main() -> None:
    args = [Path(a) for a in sys.argv[1:]]
    if not args:
        up = F.ROOT / "submissions" / "uploaded"
        args = sorted(up.glob("*.parquet")) if up.exists() else []
    if not args:
        raise SystemExit("no parquet given and submissions/uploaded/ is empty or absent")
    refs = references()
    if not refs:
        print("warning: no reference candidates on disk; the hash test still runs")
    for a in args:
        compare(a, refs)


if __name__ == "__main__":
    main()
