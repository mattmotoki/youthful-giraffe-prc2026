"""Does the candidate fit see the same feature set the folds were scored on?

The failure shape this rules out has no symptom: a serving path that builds a
different frame from the fold path still produces the right row count, the
right ids and a file that joins cleanly, and is simply a different model. It
bit retro-v1 today (build_submission on a 50-column frame while the folds ran
on 171) and only surfaced because a KeyError happened to fire.

`model2.fit_predict` derives its features from the TRAINING frame's columns, so
the check is whether the twelve-month training frame and each fold's training
frame yield the same `base_feats`, and whether the ranking frame carries every
one of them.

  PYTHONPATH=. .venv/bin/python experiments/exp-018b-climb/code/verify_serve_frame.py
"""
from __future__ import annotations
import hashlib, sys
from pathlib import Path
import pandas as pd

_F = Path(__file__).resolve().parents[2] / "exp-018-fresh" / "code"
sys.path.insert(0, str(_F)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as D            # noqa: E402
import model as M              # noqa: E402
from prc import data           # noqa: E402


def feats_of(train: pd.DataFrame) -> list[str]:
    return [c for c in train.columns if c not in M.DROP + M.HIGH_CARD]


def sig(cols) -> str:
    return hashlib.sha256("\n".join(cols).encode()).hexdigest()[:12]


def main() -> None:
    rank_tr = D.load(data.MONTHS)
    rank_va = D.load(["2026-01", "2026-07"])
    rank_tr, (rank_va,) = D.add_encodings(rank_tr, [rank_va])
    serve = feats_of(rank_tr)
    print(f"serve  (12 months -> ranking) {len(serve):3d} feats  sig {sig(serve)}")
    missing = [c for c in serve if c not in rank_va.columns]
    print(f"       columns the ranking frame lacks: {missing if missing else 'none'}")
    bad = bool(missing)
    del rank_tr, rank_va

    for f in ("july", "july_loo", "december", "january"):
        fold = data.FOLDS[f]
        tr = D.load(fold.train_months)
        va = D.load([fold.valid_month])
        tr, (va,) = D.add_encodings(tr, [va])
        ff = feats_of(tr)
        same = ff == serve
        extra = sorted(set(ff) - set(serve))
        lack = sorted(set(serve) - set(ff))
        print(f"fold {f:10s} {len(ff):3d} feats  sig {sig(ff)}  "
              f"{'IDENTICAL to serve' if same else 'DIFFERS'}"
              + ("" if same else f"  only-in-fold {extra}  only-in-serve {lack}"))
        bad |= not same
        del tr, va
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
