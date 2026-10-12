"""One line per submission candidate in the repo, so the desk can order uploads.

Board estimate is the quadrature combination of the two ranking-analogue folds
weighted by ranking-month share. A candidate whose experiment recorded no such
estimate is listed with the score it did record and a note, never with a guess.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

#: What each candidate's own experiment recorded. Board estimate = quadrature of
#: july_loo and january; "-" where the experiment never measured those folds.
KNOWN = {
    "exp-001-retro-v1.parquet":      ("exp-001 v1, LightGBM raw seconds", None, "july 531.4 / dec 287.0; no july_loo"),
    "exp-001-retro-v1-c6.parquet":   ("exp-001 c6, residual + linear leaves + noac head", None, "july 343.6 / dec 222.4 / jan 349.0"),
    "exp-001-retro-v1-c7.parquet":   ("exp-001 c6 blended 0.5/0.5 with prc.tail_model", None, "july 323.5 / dec 218.1 / jan 347.1"),
    "exp-001-retro-v1-c8.parquet":   ("exp-001 c8", None, "see exp-001 RESULTS"),
    "exp-001-retro-v1-c9.parquet":   ("exp-001 c9", None, "see exp-001 RESULTS"),
    "exp-008-blend3.parquet":        ("exp-008, NNLS of tail v1 + Huber LGB + linear-leaf LGB", 320.37, "july_loo 300.0 / dec 216.1 / jan 344.3"),
    # exp-016 (worker `robustness`, additive entries only; nothing else in this
    # file changed). A learned stacker over the five member OOFs, fitted on OOF
    # only, five-seed bag, `noac` rows on the flat four-member average. stack-1
    # and stack-2 sit on the PRE-REPAIR block winter member and so are compared
    # against a blend-2 that uses it; stack-3 sits on the published deice member
    # and is compared against a blend-2 that uses that, board 317.661. Board
    # estimates are therefore NOT comparable across the two bases -- see the note
    # column and experiments/exp-016-stack/RESULTS.md.
    # NOT REBUILDABLE (2026-09-05): all three consume artifacts/exp-008/pred/
    # C2-d8p-ranking.parquet, whose bytes moved from 48acb7e7 to 56f95373 when
    # exp-012's reproducibility rebuild refit CatBoost and did not preserve the
    # original. The FOLD numbers below are unaffected -- the stacker is fitted and
    # scored on OOF, which did not move -- but the served files cannot be
    # regenerated. Superseded by exp-016-stack-3b once exp-012 publishes a final
    # member set; rebuilding stack-3 on the current C2-cat moves it by RMS 8.0 s.
    "exp-016-stack-1.parquet":       ("exp-016 stacker, members-only, block winter member — NOT REBUILDABLE", 317.28, "july_loo 295.53 / dec 213.93 / jan 342.69; base blend-2 318.02"),
    "exp-016-stack-2.parquet":       ("exp-016 stacker, members+gates, block winter member — NOT REBUILDABLE", 317.53, "july_loo 295.95 / dec 213.17 / jan 342.75; base blend-2 318.02"),
    "exp-016-stack-3.parquet":       ("exp-016 stacker, members-only, deice winter member — NOT REBUILDABLE", 317.22, "july_loo 295.88 / dec 214.37 / jan 342.17; base blend-2 317.66"),
    # stack-4: rebuildable, on blend-4's pinned member set and blend-4's own
    # ranking inputs. Anchored on blend-4's NNLS rather than on a flat average --
    # under this member set flat is 1.92 s WORSE than the blend, so a flat-anchored
    # stacker loses by 4.11 s and the same stacker anchored on the blend wins by
    # 0.50 s. Beats blend-4 on all three folds; two of three intervals exclude zero.
    "exp-016-stack-4.parquet":       ("exp-016 stacker on blend-4, anchored on blend-4's NNLS", 312.68, "july_loo 288.85 / dec 207.27 / jan 340.30; base blend-4 313.17, gain 0.50"),
}


def main() -> None:
    d = ROOT / "submissions" / "candidates"
    rows = []
    for p in sorted(d.glob("*.parquet")):
        name = p.name
        desc, be, note = KNOWN.get(name, (None, None, ""))
        cfg = ROOT / "experiments" / "exp-012-blend2" / f"blend-config-{name.split('-')[-1].split('.')[0]}.json"
        if name.startswith("exp-012-blend-") and cfg.exists():
            c = json.loads(cfg.read_text())
            desc = "exp-012 standing blend: " + ", ".join(c["weights"]["_all"])
            be = c["board_estimate"]
            note = " / ".join(f"{k} {v}" for k, v in c["rotated"].items())
        rows.append((name, desc or "(unrecorded)", be, note, D.sha256(p)[:12]))
    rows.sort(key=lambda r: (r[2] is None, r[2] if r[2] is not None else 0))
    print("| candidate | what it is | board estimate | folds | sha256 |")
    print("|---|---|---|---|---|")
    for n, desc, be, note, sha in rows:
        b = f"**{be:.2f}**" if be is not None else "–"
        print(f"| `{n}` | {desc} | {b} | {note} | `{sha}…` |")


if __name__ == "__main__":
    main()
