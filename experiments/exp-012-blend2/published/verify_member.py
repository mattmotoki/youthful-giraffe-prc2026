"""Verify a published member against its own interface document.

    python published/verify_member.py [MEMBER-*.md]

A published member is a promise to a consumer: these bytes, these columns, these
scores. This reads the promise out of the markdown table in the interface doc and
checks it against the files, so a consumer (exp-016, or a later session of this
lane) can tell in one command whether what they are reading is what was
published. It is also the check that catches the interface going stale when a
member is rebuilt and the doc is not.
"""
from __future__ import annotations
import hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402


def claims(doc: Path) -> dict:
    """path -> {sha, fold}. A four-column row also declares the fold the member
    was VALIDATED on, which is the claim a filename cannot carry: retro-v1 shipped
    a file whose name said july and whose fold was july, into a slot expecting
    july_loo, with identical ids, identical y_true and a different model."""
    out = {}
    for line in doc.read_text().splitlines():
        if not line.startswith("|"):
            continue
        p = [c.strip().strip("`") for c in line.split("|")[1:-1]]
        sha = next((x for x in p if len(x) == 64 and all(c in "0123456789abcdef" for c in x)), None)
        path = next((x for x in p if "/" in x and x.endswith(".parquet")), None)
        if sha and path:
            fold = next((x for x in p if x in ("july", "july_loo", "december", "january", "ranking")), None)
            out[path] = {"sha": sha, "fold": fold}
    return out


def frame_contract(p: Path) -> tuple[str, bool]:
    """A member's frame identity, if the producer emitted it.

    Neither of retro-v1's two bugs is visible in the parquet: the 50-column
    build_submission would have written the right rows and the right ids from the
    wrong frame. The consumer can only assert what the producer states, so a
    member declares its frame in <artifact>.meta.json and this reports loudly when
    that file is absent rather than passing silently."""
    m = p.with_suffix(".meta.json")
    if not m.exists():
        return "no .meta.json - frame UNVERIFIABLE", False
    import json
    d = json.loads(m.read_text())
    return (f"frame {d.get('n_frame_columns','?')} cols "
            f"sha {str(d.get('column_sha',''))[:12]} threads {d.get('threads','?')}"), True


def main(doc_name: str | None = None) -> int:
    docs = [HERE / doc_name] if doc_name else sorted(HERE.glob("MEMBER-*.md"))
    bad = 0
    for doc in docs:
        c = claims(doc)
        print(f"{doc.name}: {len(c)} hashed artifact(s)")
        for rel, claim in c.items():
            sha = claim["sha"]
            p = ROOT / rel if rel.startswith("artifacts") else HERE.parent / rel
            if not p.exists():
                print(f"  MISSING {rel}"); bad += 1; continue
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            d = pd.read_parquet(p)
            miss = {"MVT_ID_mvt", "y_pred"} - set(d.columns)
            note = ""
            if "fold" in d.columns and "y_true" in d.columns:
                f = d["fold"].iloc[0]
                n = len(D.load_month(D.FOLDS[f].valid_month,
                                     columns=["MVT_ID_mvt", "PHASE_mvt"]).query("PHASE_mvt=='DEP'"))
                note = (f"rows {len(d):,}/{n:,}  "
                        f"rmse {np.sqrt(((d.y_true - d.y_pred) ** 2).mean()):.2f}")
            else:
                note = f"rows {len(d):,}  nulls {int(d['y_pred'].isna().sum()):,}"
            fold_ok = True
            if claim["fold"] and claim["fold"] != "ranking":
                actual = str(d["fold"].iloc[0]) if "fold" in d.columns else "(none)"
                if actual != claim["fold"]:
                    note += f"  FOLD MISMATCH declared {claim['fold']} actual {actual}"
                    fold_ok = False
            fc, has_meta = frame_contract(p)
            note += f"  [{fc}]"
            ok = got == sha and not miss and fold_ok
            bad += 0 if ok else 1
            print(f"  {'ok ' if ok else 'BAD'} {rel.split('/')[-1]:44s} {note}"
                  + (f"  missing columns {miss}" if miss else ""))
    print("\nINTERFACE OK" if not bad else f"\n{bad} PROBLEM(S)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:2] or [None])))
