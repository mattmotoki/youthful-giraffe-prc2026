# exp-016 handover — running the stacker on a new blend

For the blend owner, who runs this lane's code when blend-5 is issued.
Written 2026-09-06 as the `robustness` session closed. Everything below is
committed; nothing depends on that session.

## The one command sequence

```
cd experiments/exp-016-stack
# 1. write members-blend5.json: copy members-blend4.json, point "oof" and
#    "ranking" at blend-5's addresses. N1-nmtail must be listed in "oof" even
#    though members-N.json omits it -- it enters through the nm_late split.
python code/members.py     --config members-blend5.json   # pins + prints the live set
python code/blend4.py      --config members-blend5.json   # the rotation and the gain
python code/submit4.py     --config members-blend5.json   # the candidate
```

`code/blend4.py` is named for blend-4 but is not specific to it: it fits
**per-regime NNLS, all members inside `nm_late` and all but `N1-nmtail`
outside**, which is blend-4's rule. **If blend-5's rule differs, change
`fit_blend4` to match it and re-check the reproduction line first** (below).

## Three things that will bite if skipped

**1. Check the baseline reproduces before believing any gain.** `blend4.py`
prints, per fold, `blend-4 <mine> (published <yours>, delta …)`. On blend-4 the
delta was `+0.0000` on all three folds. **If it is not ~0, the comparison is
against a re-implementation and the gain is meaningless** — fix `fit_blend4`
before reading anything else.

**2. The anchor is worth more than the stacker.** The residual is taken from an
anchor and added back. On blend-4, anchoring on the **flat average** lost 4.11 s
and anchoring on **blend-4's own NNLS** won 0.50 s — a 4.6 s swing from one line.
`submit4.py` anchors on the blend. **Whatever rule blend-5 ships, anchor on
that**, not on flat: a stacker anchored on a rule that cannot reallocate
inherits that rule's handicap. `code/submit.py` (stack-1..3) still anchors on
flat and is kept only to reproduce those older candidates.

**3. The anchor must be out of sample on the stacker's training rows.**
`blend4.py` builds it as the *rotated* blend — for fold g, fitted on the other
two. Using an in-sample anchor would have the stacker correcting a blend that
has already seen those labels.

## What the pins do

`members.verify(require_ranking=True)` refuses **by name** on any member whose
bytes moved or whose file is absent, e.g.

```
member set is not servable:
  absent:rank:D1-winter:artifacts/exp-012/pred/D1w-deice-ranking.parquet
```

Pins are per-member-set (`pins-members-blend5.json`), so a config swap cannot
reuse the previous set's pins. Every entry point prints the live member
addresses on startup — if a `--config` silently fails to take, that line is how
you see it.

## Reporting

Report **both** numbers, per `prc.data.rmse_both` (VERSION 4). The board
predictor calibrated on reading 1 is **board ≈ clean + 4.3 s**; it is one point,
dominated by January, so a candidate that improves January specifically should
see a smaller correction. `docs/board-reading.md` has the derivation.

## Known open items

- blend-4 serves an `N1-nmtail` ranking file built 01:15 while its weights are
  fitted on the v4 OOF built 03:06. If those are different `prc.nmtail`
  versions, blend-4 and stack-4 carry the same inherited mismatch. Unresolved.
- `stack-1`, `stack-2`, `stack-3` are **not rebuildable** (their CatBoost
  ranking input was destroyed). They stay on disk annotated; `stack-4` is.
