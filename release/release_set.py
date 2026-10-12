"""The rule that defines the public GPLv3 release set of this repository.

One rule, used by three consumers: `release/build_manifest.py` (writes
`release/MANIFEST.tsv`), `release/check_release.py` (verifies a tree against
that manifest) and `scripts/check_licenses.py` (licence and attribution
checks on the same set).  Changing the release set means changing this file
and regenerating the manifest, which is the point: the set cannot drift by
accident.

The release is a SOURCE release.  It carries the code that produced the
submission, the documentation that explains it, and the small JSON and CSV
records the documentation quotes.  It carries no competition rows, labels, trained model artifact, log or
internal task description. Saved replay arrays contain model prediction or admission values; IDs
come from the separately supplied official template. Open geometry retains
its data licences, separate from the GPLv3 source.

Each exclusion below states its source or privacy boundary.
"""
from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The refit sources: the source files of the final chain that the export lacks or holds in other bytes, at their
#: repo paths under refit-src/, and the versions of one held step under refit-pins/<unit>/ (release/make_refit_tree.py
#: lays them over the export source). The list is exact: such a path ships only when release/REFIT-SOURCES.json holds
#: it, so the exploration code of the same folders stays out. The rules below still apply to each listed path.
REFIT_PREFIX: tuple[str, ...] = ("refit-src/", "refit-pins/")
REFIT_SOURCES = "release/REFIT-SOURCES.json"


@lru_cache(maxsize=None)
def refit_listed(root: Path = ROOT) -> frozenset[str]:
    """The published paths of release/REFIT-SOURCES.json (empty when the file is absent)."""
    path = root / REFIT_SOURCES
    if not path.exists():
        return frozenset()
    doc = json.loads(path.read_text())
    return frozenset(f["published"] for f in doc["files"])

#: Whole trees kept out of the public release.
EXCLUDE_PREFIX: tuple[str, ...] = (
    # Another challenge's data is rendered into 19 notebook cells here, and
    # nothing in the pipeline imports it (the source-only boundary).
    "references/nasa-airport-throughput-prediction-challenge/",
    # Internal worker task descriptions: not part of the method.
    "assignments/",
    # CatBoost training scratch.
    "catboost_info/",
    # Captured public-leaderboard snapshots: ranking data, and no input to
    # the model (docs/input-rule.md rule 1).
    "docs/board/",
    "docs/status/",
    "docs/scout/",
    "docs/archive/",
    ".desk/",
    "scripts/prc-desk-queue/",
    "experiments/exp-145-reproduce-v32/",
    "experiments/exp-149-release-v32/",
    "experiments/exp-149-release-v35/",
    "experiments/exp-149-release-v37/",
    "experiments/exp-149-release-v40/",
    "experiments/exp-149-release-zabd/",
    "experiments/exp-151-published-means/results/step-d/",
    "experiments/exp-151-published-means/source/",
    # Two third-party documents copied in whole as research notes: the
    # EUROCONTROL DPI Implementation Guide 2.700 and a Journal of Air
    # Transport Management article.  Neither is ours to redistribute.
    "experiments/exp-019-research/notes/",
    # Round 1 finding 3: verbatim transcripts of consultations with a third
    # party's model, each headed by an ACCOUNT-BOUND conversation URL.  Neither
    # the transcript nor the URL is ours to publish.
    "docs/pro-reasoning-2026-09-08/",
    "docs/research/",
    # Post-v22 outside-in research is useful to the private competition desk,
    # but it neither produced nor explains the released model.
    "experiments/exp-067-operational-research/",
    # Generated exp-068 census tables summarize organiser records.  The
    # reusable implementation and synthetic tests remain public.
    "experiments/exp-068-runway-queue/analysis/",
    # Release construction evidence is private process material.  Keeping the
    # whole record out also lets a later evidence-only commit prove exact R.
    "experiments/exp-069-release-refresh/",
    "experiments/exp-070-release-hardening/",
    "experiments/exp-071-release-route-correction/",
    "experiments/exp-072-release-row-correction/",
    "experiments/exp-073-release-id-normalization/",
    "experiments/exp-074-release-boundary-binding/",
    "experiments/exp-075-release-stream-final/",
    "experiments/exp-076-release-stateful-scanner/",
    "experiments/exp-077-release-import-isolation/",
    "experiments/exp-078-release-isolation-proof/",
    "experiments/exp-064-release/",
    "docs/assignments/",
    # Post-v22 transport evidence names private Kaggle datasets, kernels,
    # account ownership and remote run identities.  Keep the reusable source
    # and tests; withhold the account-bound evidence and process receipts.
    "experiments/exp-065-sparseholdout-retry/analysis/kaggle-dataset/",
    "experiments/exp-065-sparseholdout-retry/receipts/",
    "experiments/exp-065-sparseholdout-retry/source/",
    "experiments/exp-066-sparseholdout-preflight/evidence/",
    "experiments/exp-066-sparseholdout-preflight/receipts/",
    "experiments/exp-066-sparseholdout-preflight/source/",
)

#: Files that the rules below would withhold but that the v22 build path NEEDS.
#: Round 1 finding 5: the extension panel of the shipped seasonal head is built
#: by these, and without them the release has no source for a step that
#: materially changes v22.  Each line says what it produces.
INCLUDE_EXACT: frozenset[str] = frozenset({
    # Public lineage proof; the rest of exp-064 is internal release evidence.
    "experiments/exp-064-release/code/lineage.py",
    "experiments/exp-064-release/analysis/lineage.json",
    # the per-month orchestrator: one extension month, end to end
    "experiments/exp-047-queue/work/month_chain.sh",
    # the members it fits, in the order month_chain.sh calls them
    "experiments/exp-047-queue/work/member_folds.py",
    "experiments/exp-047-queue/work/nmtail_fold.py",
    "experiments/exp-047-queue/work/c2cat_hedge.sh",
    "experiments/exp-047-queue/work/l2hub_hedge.sh",
    "experiments/exp-047-queue/work/f1_local.sh",
    "experiments/exp-047-queue/work/promote_hedge.py",
    # base6--<month>.parquet, which exp-046's onemonth.py reads
    "experiments/exp-047-queue/work/base6.py",
    # the block re-emission check that gates the month
    "experiments/exp-047-queue/work/block_verify.py",
    # Round 2 finding 4: every producer above imports this one, so releasing
    # them without it released nine scripts that cannot start.
    "experiments/exp-047-queue/work/lane.py",
    "experiments/exp-047-queue/work/census_v22.py",
})

#: A path with any of these directory names anywhere in it is excluded:
#: scratch, process records and out-of-fold prediction files.  Out-of-fold
#: files carry `y_true`, which is the organisers' 2025 label column.
EXCLUDE_DIR_SEGMENT: frozenset[str] = frozenset({
    # Round 1 finding 3: the rule excluded "review" and released nine reviewer
    # transcripts under docs/reviews/, contradicting the source-only boundary.
    "reviews",
    "logs", "work", "review", "oof", "__pycache__", ".ipynb_checkpoints",
})

#: Data and machine output.  `.parquet`, `.npz` and `.npy` hold competition
#: rows, labels or model predictions; the rest are run records.
EXCLUDE_SUFFIX: tuple[str, ...] = (
    ".xlsx", ".xls",
    ".parquet", ".npz", ".npy",
    ".log", ".pid", ".out", ".err", ".tmp", ".tfevents", ".sha256", ".time",
)

#: File NAMES excluded wherever they appear.  `kernel-metadata.json` carries
#: one Kaggle account's kernel slug, its `is_private` flag and the slugs of
#: private datasets holding the organisers' files; it is account metadata, not
#: method, and a reproducer writes their own (`docs/REPRODUCE.md` §5).
EXCLUDE_NAME: frozenset[str] = frozenset({
    "kernel-metadata.json",
})

#: Exact exclusions cover private records, real movement rows and contamination
#: evidence. Source and synthetic tests are released only after content checks.
EXCLUDE_EXACT: frozenset[str] = frozenset({
    # Internal records: real organiser rows or private Kaggle custody details.
    "docs/RELEASE-CHECKLIST.md",
    "docs/LEDGER.md",
    "docs/SUBMISSIONS.md",
    "docs/DESK_CURRENT.md",
    "docs/PAUSE-2026-09-17.md",
    "docs/RETURN-2026-09-17.md",
    "experiments/exp-087-q09-half-amplitude/code/publish_package.sh",
    "experiments/exp-097-pkc-motion-transition/code/ws_run.sh",
    "experiments/exp-097-pkc-motion-transition/code/ws_watch.sh",
    "experiments/exp-098-q09-serving-repair/code/launch.sh",
    "experiments/exp-098-q09-serving-repair/code/watch.sh",
    "docs/board-reading.md",
    "docs/BACKLOG.md",
    "docs/COORDINATION.md",
    "docs/DECISIONS.md",
    "docs/QUEUE-2026-09-07.md",
    "docs/SUMMARY.md",
    "experiments/exp-009-noac2/RESULTS.md",
    "experiments/exp-046-surfacefit/RESULTS.md",
    # Private clean-score row values and a 131-row organiser-ID analysis.
    "prc/_private_clean_score_rows.py",
    "experiments/exp-046-surfacefit/analysis/class-placebo.json",
    "experiments/exp-063-sparseholdout/RESULTS.md",
    "experiments/exp-065-sparseholdout-retry/RESULTS.md",
    "experiments/exp-066-sparseholdout-preflight/PREFLIGHT.md",
    # A private post-v22 contamination ledger, not model evidence.
    "experiments/exp-068-runway-queue/CONTAMINATION.md",
    "experiments/exp-046-surfacefit/analysis/boundary-diff-read.json",        # 49
    "experiments/exp-046-surfacefit/analysis/class-diff-permonth.json",       # 21
    "experiments/exp-047-queue/analysis/boundary-diff.json",                  # 49
    "experiments/exp-047-queue/analysis/leverA-allrows.json",                 # 2
    "experiments/exp-047-queue/analysis/leverB-screen.json",                  # 16
    "experiments/exp-047-queue/analysis/nonm-anatomy.json",                   # 90
    "experiments/exp-047-queue/analysis/pre-handshake/prefix-classdiff.json", # 31
    "experiments/exp-047-queue/analysis/prefix-classdiff.json",               # 3,730
    "experiments/exp-047-queue/analysis/trimmed-12h-registration.json",       # 16
    "experiments/exp-054-arrivalrepr/analysis/audit-r1.json",                 # 12
    "experiments/exp-054-arrivalrepr/analysis/audit-r2.json",                 # 9
    "experiments/exp-054-arrivalrepr/analysis/audit-r3.json",                 # 9
    "experiments/exp-054-arrivalrepr/analysis/audit.json",                    # 12
})


#: Round 1 finding 2: released records that carry ROW-LEVEL organiser
#: movements -- a movement id, a flight callsign and stand, or the 2025 target
#: under an alias -- in a shape the old JSON-only gate could not see.
ROW_LEVEL_RECORDS: frozenset[str] = frozenset({
    # MVT ids with flight, stand, runway, type and our ranking predictions
    "experiments/exp-014-robustness/analysis/q3_member_disagreement.csv",
    "experiments/exp-014-robustness/analysis/q3_predictions_above_20000.csv",
    # MVT ids with two candidate predictions on ranking rows
    "experiments/exp-014-robustness/analysis/r1_no_aobt_with_nm.csv",
    # MVT ids, flight details AND the 2025 target under the alias `y`
    "experiments/exp-020-cleanboard/analysis/rollover_census.csv",
    # MVT ids with callsigns and stands
    "experiments/exp-043-pilot/analysis/nonm_unmatched.csv",
    # MVT ids in JSON LISTS, which the `"MVT_ID_mvt": <number>` gate missed.
    # class-diff-v15 carries 69 across five lists; serve-v14 carries the roll
    # of ranking rows reverted to the price base.
    "experiments/exp-046-surfacefit/analysis/class-diff-v15.json",
    "experiments/exp-046-surfacefit/analysis/serve-v14.json",
    # Round 2 finding 2: bulk lists of the organisers' row ids under keys the
    # gate did not read -- `draw_row_ids`, `forced_in`, `forced_out`.  341, 208
    # and 208 distinct movement ids.  They serve no reader.
    "experiments/exp-046-surfacefit/analysis/rank-provenance-v22.json",
    "experiments/exp-046-surfacefit/analysis/rank-provenance.json",
    "experiments/exp-046-surfacefit/analysis/bag-provenance.json",
    "experiments/exp-056-q12advocate/analysis/artifacts-r1.json",
    "experiments/exp-056-q12advocate/analysis/artifacts-r2.json",
    "experiments/exp-056-q12advocate/analysis/dag.json",
    "experiments/exp-056-q12advocate/analysis/dag-r2.json",
})

#: Private prose, dataset and kernel receipts found by the full blob inventory.
PRIVATE_RECORDS: frozenset[str] = frozenset({
    "experiments/exp-001-retro-v1/RESULTS.md",
    "experiments/exp-054-arrivalrepr/SPEC.md",
    "experiments/exp-054-arrivalrepr/analysis/bound-support-census.md",
    "experiments/exp-058-origincontrast/SPEC.md",
    "docs/SUBMISSIONS.md",
    "references/source-lock.json",
    'docs/CLOUD-COMPUTE.md',
    'docs/COMPUTE.md',
    'docs/HANDOVER.md',
    'experiments/exp-008-diversity/blend-config-exp-008-blend3.json',
    'experiments/exp-012-blend2/blend-config-1.json',
    'experiments/exp-012-blend2/blend-config-2.json',
    'experiments/exp-012-blend2/blend-config-3.json',
    'experiments/exp-012-blend2/blend-config-3b.json',
    'experiments/exp-012-blend2/blend-config-4.json',
    'experiments/exp-012-blend2/blend-config-5.json',
    'experiments/exp-025-cloud/RESULTS.md',
    'experiments/exp-025-cloud/SPEC.md',
    'experiments/exp-025-cloud/oof-nested/README.md',
    'experiments/exp-025-cloud/oof-synthetic/README.md',
    'experiments/exp-025-cloud/results/f1-smoke-port.json',
    'experiments/exp-025-cloud/results/nested-f1-fresh-a.json',
    'experiments/exp-025-cloud/results/nested-f1-fresh-b.json',
    'experiments/exp-025-cloud/results/nested-f1-fresh-base.json',
    'experiments/exp-025-cloud/results/nested-f1-fresh-c.json',
    'experiments/exp-025-cloud/results/nested-l2-hub8725-a.json',
    'experiments/exp-025-cloud/results/nested-l2-hub8725-b.json',
    'experiments/exp-025-cloud/results/nested-l2-hub8725-base.json',
    'experiments/exp-025-cloud/results/nested-l2-hub8725-c.json',
    'experiments/exp-025-cloud/results/nested-t-tailv4-all.json',
    'experiments/exp-025-cloud/results/nested-t-tailv4-base.json',
    'experiments/exp-025-cloud/results/port.json',
    'experiments/exp-027-step/RESULTS.md',
    'experiments/exp-027-step/SPEC.md',
    'experiments/exp-027-step/results/port.json',
    'experiments/exp-035-k2climb/NEXT.md',
    'experiments/exp-035-k2climb/RESULTS.md',
    'experiments/exp-035-k2climb/SPEC.md',
    'experiments/exp-035-k2climb/kernels/mod-dj/port-mod-dj.json',
    'experiments/exp-035-k2climb/kernels/mod-july/port-mod-july.json',
    'experiments/exp-035-k2climb/kernels/mod-rank1/port-mod-rank1.json',
    'experiments/exp-035-k2climb/kernels/mod-rank2/port-mod-rank2.json',
    'experiments/exp-035-k2climb/kernels/phi-dj/port-phi-dj.json',
    'experiments/exp-035-k2climb/kernels/phi-july/port-phi-july.json',
    'experiments/exp-035-k2climb/kernels/seed1-dj/port-seed1-dj.json',
    'experiments/exp-035-k2climb/kernels/seed1-july/port-seed1-july.json',
    'experiments/exp-035-k2climb/kernels/seed2-dj/port-seed2-dj.json',
    'experiments/exp-035-k2climb/kernels/seed2-july/port-seed2-july.json',
    'experiments/exp-035-k2climb/kernels/seed2-rank/port-seed2-rank.json',
    'experiments/exp-035-k2climb/kernels/wide-july/port-wide-july.json',
    'experiments/exp-035-k2climb/results/port-mod-dj.json',
    'experiments/exp-035-k2climb/results/port-mod-july.json',
    'experiments/exp-035-k2climb/results/port-mod-rank1.json',
    'experiments/exp-035-k2climb/results/port-mod-rank2.json',
    'experiments/exp-035-k2climb/results/port-phi-dj.json',
    'experiments/exp-035-k2climb/results/port-phi-july.json',
    'experiments/exp-035-k2climb/results/port-seed1-dj.json',
    'experiments/exp-035-k2climb/results/port-seed1-july.json',
    'experiments/exp-035-k2climb/results/port-seed2-dj.json',
    'experiments/exp-035-k2climb/results/port-seed2-july.json',
    'experiments/exp-035-k2climb/results/port-seed2-rank.json',
    'experiments/exp-035-k2climb/results/port-wide-july.json',
    'experiments/exp-040-chain/kernels/c2-december/out/port.json',
    'experiments/exp-040-chain/kernels/c2-january/out/port.json',
    'experiments/exp-040-chain/kernels/c2/out/port.json',
    'experiments/exp-040-chain/results/port-c2-december.json',
    'experiments/exp-040-chain/results/port-c2-january.json',
    'experiments/exp-040-chain/results/port-c2-july_loo.json',
    'experiments/exp-040-chain/results/port-december.json',
    'experiments/exp-040-chain/results/port-january.json',
    'experiments/exp-040-chain/results/port-july_loo.json',
    'experiments/exp-040-chain/results/port-ranking.json',
    'experiments/exp-042-surface/REPORT.md',
    'experiments/exp-043-pilot/analysis/provenance.json',
    'experiments/exp-045-harvest/SPEC.md',
    'experiments/exp-045-harvest/analysis/machine_check.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-01.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-02.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-03.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-04.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-05.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-06.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-07.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-08.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-09.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-10.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-11.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-12.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-13.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-14.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-15.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-16.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-17.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-18.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-19.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-20.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-21.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-22.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-23.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-24.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-25.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-26.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-27.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-28.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-29.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-30.json',
    'experiments/exp-045-harvest/analysis/rederived/2025-12-31.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-01.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-02.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-03.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-04.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-05.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-06.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-07.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-08.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-09.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-10.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-11.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-12.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-13.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-14.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-15.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-16.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-17.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-18.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-19.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-20.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-21.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-22.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-23.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-24.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-25.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-26.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-27.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-28.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-29.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-30.json',
    'experiments/exp-045-harvest/analysis/rederived/2026-01-31.json',
    'experiments/exp-045-harvest/manifest.json',
    'experiments/exp-047-queue/analysis/refit-cost.json',
    'experiments/exp-047-queue/analysis/refit-port.json',
    'experiments/exp-056-q12advocate/RESULTS.md',
    'experiments/exp-056-q12advocate/RUNBOOK.md',
    'experiments/exp-056-q12advocate/SPEC.md',
    'experiments/exp-056-q12advocate/analysis/dag-r1.json',
    'experiments/exp-056-q12advocate/analysis/kernel-receipts.json',
    'experiments/exp-056-q12advocate/analysis/kernel-receipts.md',
})


PUBLIC_STAGE_INPUTS = frozenset({
    "experiments/exp-148-short-route/SPEC.md",
    "experiments/exp-151-published-means/frozen-parameters.json",
    "experiments/exp-127-clock-precision/stand_geometry.csv",
    "experiments/exp-127-clock-precision/rederive/SPEC.md",
    "experiments/exp-130-ranking-exposure/census.csv",
    "experiments/exp-130-ranking-exposure/CENSUS.md",
    "experiments/exp-130-ranking-exposure/SPEC.md",
    "experiments/exp-134-proxies-in-learner/SPEC.md",
    "experiments/exp-136-wider-proxies/SPEC.md",
    "experiments/exp-137-withclock-proxies/SPEC.md",
})


def is_released(path: str) -> bool:
    """True when `path` (a repo-relative POSIX path) belongs in the release."""
    if path.startswith(REFIT_PREFIX) and path not in refit_listed():
        return False
    if path in INCLUDE_EXACT:
        return True
    # Experiments after the v22 fitting path retain source, not private process evidence.
    if path.startswith("experiments/"):
        import re
        match = re.match(r"experiments/exp-(\d+)[^/]*/", path)
        if match and int(match[1]) >= 48:
            if not path.endswith((".py", ".sh")) and path not in PUBLIC_STAGE_INPUTS:
                return False
    if path in EXCLUDE_EXACT or path in ROW_LEVEL_RECORDS or path in PRIVATE_RECORDS:
        return False
    if path.startswith(EXCLUDE_PREFIX):
        return False
    parts = path.split("/")
    if parts[-1] in EXCLUDE_NAME:
        return False
    if EXCLUDE_DIR_SEGMENT.intersection(parts[:-1]):
        return False
    if path.endswith(EXCLUDE_SUFFIX):
        return False
    return True


def tracked(root: Path = ROOT, rev: str | None = None) -> list[str]:
    """Every tracked path, NUL-separated so a path with a space survives.

    With `rev`, the paths are the ones THAT COMMIT has.  Round 2 finding 1: the
    manifest builder hashed blobs from `--rev` but chose which paths to hash from
    the current index, so building a manifest for an older commit could list a
    file that commit never had.
    """
    cmd = (["git", "ls-tree", "-r", "-z", "--name-only", rev] if rev
           else ["git", "ls-files", "-z"])
    r = subprocess.run(cmd, cwd=root, capture_output=True, check=True)
    return [p for p in r.stdout.decode().split("\0") if p]


def release_set(root: Path = ROOT, rev: str | None = None) -> list[str]:
    """Every tracked path that ships, the manifest itself included."""
    return sorted(p for p in tracked(root, rev) if is_released(p))


#: The manifest ships, but it cannot record its own digest: hashing it is part
#: of writing it.  It is therefore the one released path the manifest does not
#: list, and `check_release.py` gate G knows about exactly this exception.
MANIFEST_PATH = "release/MANIFEST.tsv"


def manifest_set(root: Path = ROOT, rev: str | None = None) -> list[str]:
    """The paths `release/MANIFEST.tsv` lists: the release set minus itself."""
    return [p for p in release_set(root, rev) if p != MANIFEST_PATH]


if __name__ == "__main__":
    import sys
    rel = release_set()
    if "--excluded" in sys.argv:
        rel = sorted(set(tracked()) - set(rel))
    print("\n".join(rel))
