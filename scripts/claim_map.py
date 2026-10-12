"""Write `docs/CLAIM-MAP.md`: every public method claim -> code, record, digest.

    .venv/bin/python scripts/claim_map.py            # write the map
    .venv/bin/python scripts/claim_map.py --check    # write nothing; exit 1 if stale

Each public-method claim needs its exact code/config/record path and hash.
The earlier gap was that `scripts/build_v22.py` mapped the eighteen output
stages and nothing below them, so a reader could not get from a sentence in the write-up to the code
that implements it and the record that measures it.

Every path here is hashed as this checkout has it, so a claim whose evidence moved
shows up as a missing path rather than as a sentence nobody can check.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/CLAIM-MAP.md"

#: (write-up section, the claim, code that implements it, record that measures it)
CLAIMS = (
    ("M4.11", "Release target final-v-dh-r9-ld, the final file: V15A-R9-LD plus the DH step on 14,587 EDDF rows from saved values; no fit; the same bytes on every platform (final-v-dh-r9, final-v-dh, final-iii-dh-r9-ld, final-iii-dh-r9, final-iii-dh, final-ii-dh and final-dh: the same step on V15A-R9, V15A, V9A-R9-LD, V9A-R9, V9A, V9 and V)", "release/dh_stage.py", "release/ENTRIES.json"),
    ("M4.11", "Target final-iii-r9-ld (V9A-R9-LD): V9A-R9 with LD, ORD-S also on the ordinary member-scope rows of long-taxi airport-days (flag --long-days), its fit without these days; exact bytes on x86-64 Linux", "w2/ords_ld.py", "release/ENTRIES.json"),
    ("M4.11", "Targets final-v, final-v-r9 and final-v-r9-ld (V15A, V15A-R9, V15A-R9-LD): the chains of final-iii, final-iii-r9 and final-iii-r9-ld with the s7 member as the mean of 15 fits in each mode; exact bytes on x86-64 Linux", "release/saved-vectors/s7c15/final-ranking.json", "release/ENTRIES.json"),
    ("M4.11", "Target final-iii-r9 (V9A-R9): V9A on the R9 assembly, the 12-fit MLP values on every member-scope row of the fold and 2026 primary frames; the chain zabd-final-r9, v64-final-r9 and final-iii-r9 in final mode; exact bytes on x86-64 Linux", "release/v73_stage.py", "release/ZABD-PACK-final-r9.json"),
    ("M4.11", "Target final-iii (V9A): the model of v73 on the final files with the 9-fit s7 member and the s7 input of L2C, the chain zabd-final, v64-final and final-iii in final mode; exact bytes on x86-64 Linux", "release/v73_stage.py", "release/ENTRIES.json"),
    ("M4.11", "L2C of V9A: the s7 deviation from c5 as one more input, 9 LightGBM seeds", "w2/l2c_candidate_a.py", "docs/LINEAGE-ZABD.md"),
    ("M4.11", "s7 member of V9 and V9A: the mean of 9 fits in each mode; its values are saved", "w2/seq/fit.py", "release/saved-vectors/s7c9/final-ranking.json"),
    ("M4.11", "Targets final-ii (V9) and final (V): the same chain with the L2C of v73, on the 9-fit and the 3-fit s7 member", "scripts/build_entry.py", "release/ENTRIES.json"),
    ("M4.11", "Held rows keep their labels and saved member values; their exposure weights follow the four-month mix, and the assembly refits its blend weights with them", "release/assembly_replay.py", "release/ZABD-PACK-final.json"),
    ("M4.11", "Final pack: member and frame values of the final build; the LIRF rows without a plan read the R4mks base with r1 weight 0.444", "release/assembly_replay.py", "release/ZABD-PACK-final.json"),
    ("M4.11", "Final mode of the v73 steps: the two final files and the row order of the chain", "iso/finalmode.py", "docs/LINEAGE-ZABD.md"),
    ("M4.11", "Board check: upload v76 of the January and July rows of V, 223.8069 s, inside the forecast range", "-", "docs/PUBLIC-SUBMISSIONS.md"),
    ("M4.10", "Final entry v73: the c5 steps, the L2C level 2 and the ORD-S column of the saved s7 member on the v64 chain; exact bytes on x86-64 Linux", "release/v73_stage.py", "release/ENTRIES.json"),
    ("M4.10", "c5 steps EXT, LONG, LIRF, NOAC and ROLL on the v64 primary blend, then the v64 rule chain", "iso/candidate.py", "docs/LINEAGE-ZABD.md"),
    ("M4.10", "L2C: LightGBM correction of the residual on clock rows with an NM join, half step, 3 seeds", "w2/l2c.py", "docs/LINEAGE-ZABD.md"),
    ("M4.10", "ORD-S: the s7 sequence member as a column on ordinary stack rows, one coefficient per airport floored at 0", "w2/ords.py", "release/saved-vectors/s7c/ranking.json"),
    ("M4.10", "s7 member inputs and GPU fit commands; its values are saved", "w2/seq/fit.py", "w2/rebuild.sh"),
    ("M4.9", "Release target v64: T11b-L stage on zabd (P2 rows, quarter move toward dark18, long-taxi days kept); exact replay", "release/t11b_stage.py", "release/ENTRIES.json"),
    ("M4.9", "Release target zabd: population, file bytes and digest", "scripts/build_entry.py", "release/ENTRIES.json"),
    ("M4", "zabd assembly order; refit stack weights and C95/C98 statistics; exact replay from saved member values", "release/assembly_replay.py", "release/ZABD-PACK-ranking.json"),
    ("M4", "zabd members, rule stages, label use and gaps", "experiments/exp-169-stack-plan/code/assemble_final_032.py", "docs/LINEAGE-ZABD.md"),
    ("M2.2", "zabd external sources, licences and the OPDI declaration", "-", "docs/DATA.md"),
    ("M7", "final-v-dh-r9-ld, final-v-r9-ld, final-v-dh-r9, final-v-r9, final-v-dh, final-v, final-iii-dh-r9-ld, final-iii-r9-ld, final-iii-dh-r9, final-iii-r9, final-iii-dh, final-iii, final-ii-dh, final-ii, final-dh, final, v73, v64 and zabd replay inputs, commands and measured cost", "scripts/build_entry.py", "docs/REPRODUCE.md"),
    ("2", "R4 nine-month members, scored-month isolation and CPU stacker", "experiments/exp-148-short-route/code/stacker.py", "docs/PUBLIC-MODEL.json"),
    ("2", "R4 keep sets, fixed half blend and a single frozen shift", "scripts/build_entry.py", "release/V40-CHOICE.json"),
    ("3", "R4 development gain against local shifted B50; full route unmeasured", "experiments/exp-148-short-route/code/score_r4.py", "docs/PUBLIC-MODEL.json"),
    ("2", "Frozen coverage offset, shrinkage and bias prior", "experiments/exp-151-published-means/code/step_d.py", "experiments/exp-151-published-means/frozen-parameters.json"),
    ("2", "Published-mean shift, floor and unsupported-row guard", "scripts/published_shift.py", "docs/PUBLIC-MODEL.json"),
    ("3", "Published aggregate held-label path and optimistic development gain", "experiments/exp-151-published-means/code/step_d.py", "docs/PUBLIC-MODEL.json"),
    ("4", "EUROCONTROL monthly source, attribution and separate terms", "experiments/exp-151-published-means/code/extract_source.py", "docs/EUROCONTROL-DATA.md"),
    ("2", "Fixed 0.5 v35/v32 blend, template alignment and bytes", "experiments/exp-148-short-route/code/serve_blend.py", "docs/PUBLIC-MODEL-v36.json"),
    ("3", "B50 local gain and interval; optimistic old reference", "experiments/exp-148-short-route/code/serve_blend.py", "docs/PUBLIC-MODEL-v36.json"),
    ("2", "Eight short-route members, split ownership and admission", "experiments/exp-147-foldserve/code/pilot.py", "docs/PUBLIC-MODEL-v35.json"),
    ("2", "R2 member split masks and proxy learners", "experiments/exp-147-foldserve/code/learner_wp.py", "docs/PUBLIC-MODEL-v35.json"),
    ("2", "R2 stacker form, training rows, features and CV rounds", "experiments/exp-148-short-route/code/stacker.py", "docs/PUBLIC-MODEL-v35.json"),
    ("2", "Frozen weight and keep sets; selected versus equal fallback row counts", "experiments/exp-148-short-route/code/serve_r2.py", "release/V35-CHOICE.json"),
    ("3", "Month-hold gain, interval and optimistic v32 reference", "experiments/exp-148-short-route/code/score_r2.py", "docs/PUBLIC-MODEL-v35.json"),
    ("5", "Target parameter, component routes and v40 exact replay", "scripts/build_entry.py", "release/ENTRIES.json"),
    ("5", "Caller-owned boundary exceptions and unchanged missing-row guard", "scripts/exp148_missing_features.py", "docs/PUBLIC-MODEL-v35.json"),
    ("0", "Accepted target: score, population, file bytes and digest", "scripts/build_entry.py", "docs/PUBLIC-MODEL.json"),
    ("1", "Task target and ranking input contract", "prc/data.py", "docs/input-rule.md"),
    ("1", "Open off-block observation and match rule", "experiments/exp-046-surfacefit/code/block.py", "docs/EXTERNAL-DATA.md"),
    ("2", "v22 base composition and source chain", "scripts/build_v22.py", "experiments/exp-064-release/analysis/lineage.json"),
    ("2", "v22, v23, v24 and v32 score facts", "-", "docs/PUBLIC-SUBMISSIONS.md"),
    ("2", "Stage counts, floors, hold count and output digests", "scripts/build_v32.py", "release/V32-REPLAY.json"),
    ("2", "v23 fitted implied-clock rule", "experiments/exp-127-clock-precision/rederive/kaggle/fit-kernel/fit_rederived_clock.py", "docs/PUBLIC-MODEL-v32.json"),
    ("2", "v24 full-precision 2025-fitted weight", "experiments/exp-134-proxies-in-learner/code/fit_exp134.py", "docs/PUBLIC-MODEL-v32.json"),
    ("2", "v32 fitted and served weight, training rows, features and rounds", "experiments/exp-137-withclock-proxies/kaggle/run-v4/final/kernel.py", "docs/PUBLIC-MODEL-v32.json"),
    ("2", "Stage source pins and exp136 input role", "experiments/exp-137-withclock-proxies/code/prepare_inputs.py", "docs/PUBLIC-MODEL-v32.json"),
    ("3", "Inherited validation and fold/serve faults", "experiments/exp-134-proxies-in-learner/code/fit_exp134.py", "docs/PUBLIC-MODEL-v32.json"),
    ("3", "exp137 frozen gain gate failed; positive-point rule allowed the supported-cell final fit", "experiments/exp-137-withclock-proxies/kaggle/run-v4/final/kernel.py", "docs/PUBLIC-MODEL-v32.json"),
    ("3", "Probe history; no board-derived released parameter", "-", "docs/PUBLIC-SUBMISSIONS.md"),
    ("4", "Source GPLv3 and external-data attribution", "scripts/check_licenses.py", "docs/EXTERNAL-DATA.md"),
    ("4", "ERA5 products, transformations and exact separate CC BY bundle", "release/era5_data.py", "release/ERA5-DATA-MANIFEST.json"),
    ("4", "Gateway and OSM stand sources and separate data notices", "experiments/exp-127-clock-precision/build_stand_geometry.py", "docs/GEOMETRY-DATA.md"),
    ("4", "Recorded organizer post-operations answer", "-", "docs/external-data-rules.md"),
    ("5", "Replay arrays contain only finite model values in official template order", "scripts/build_v32.py", "release/V32-REPLAY.json"),
    ("5", "Original input checks: seven versions, 67 matched files, three unrecorded files; about 21 CPU hours, no new fit", "-", "docs/PUBLIC-MODEL-v32.json"),
    ("5", "Complete v23 fold vector regeneration", "experiments/exp-127-clock-precision/push/build_oof_fold_updates.py", "release/V23-FOLD-BASE.json"),
    ("5", "Private clean-score sidecar not required or packaged", "experiments/exp-130-ranking-exposure/code/prepare.py", "docs/REPRODUCE.md"),
)


def digest(rel: str) -> str:
    if rel == "-":
        return "—"
    p = ROOT / rel
    if not p.is_file():
        return "**MISSING**"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16] + "…"


def stages() -> list[dict]:
    spec = importlib.util.spec_from_file_location("bv22", ROOT / "scripts/build_v22.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return list(m.STAGES)


def render() -> str:
    L = ["# Claim map — every public method claim, to its code, its record and a digest",
         "",
         "Generated by `scripts/claim_map.py`; do not edit by hand.",
         "Digests are of this checkout, truncated to 16 hex characters.",
         "The write-up is [`WRITEUP.md`](WRITEUP.md); the build chain is",
         "[`REPRODUCE.md`](REPRODUCE.md) §3.",
         "Rows marked M cite sections of the method write-up for the release target",
         "final-v-dh-r9-ld (the model of v73 on the final files with four changes, then the DH step), [`writeup/METHOD.md`](writeup/METHOD.md).", "",
         "## 1. The method claims", "",
         "| §  | claim | code | digest | record | digest |",
         "|---|---|---|---|---|---|"]
    for sec, claim, code, rec in CLAIMS:
        L.append(f"| {sec} | {claim} | `{code}` | {digest(code)} | `{rec}` | {digest(rec)} |")
    L += ["", "## 2. Composition and inputs", "",
          "The target registry is `release/ENTRIES.json`; the fallback composition is `release/V32-REPLAY.json`.",
          "The older base plan is `scripts/build_v22.py --plan`.",
          "The seven derived-package rebuild routes are in `docs/REPRODUCE.md`.", "",
          "## 3. Limits", "",
          "Saved-array replay is checked. No fresh fit has reproduced those arrays.",
          "The zabd replay refits only the level-2 weights and two rule statistics; no member is refit.",
          "The v64 stage fits nothing and reads no label.",
          "The v73 steps refit the step models from the 2025 labels; the s7 member values are saved, not refit.",
          "The v73 bytes are recorded on x86-64 Linux; on macOS arm64 the L2C trees differ.",
          "The final V bytes are recorded on x86-64 Linux, as for v73. The DH entries fit nothing.",
          "The old folds have inherited label paths and scope differences.",
          "Public publication and entry eligibility are separate from local checks.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = render()
    missing = text.count("**MISSING**")
    if a.check:
        if OUT.exists() and OUT.read_text() == text:
            print(f"CLAIM-MAP.md is current ({len(CLAIMS)} claims, {missing} missing paths)")
            return 1 if missing else 0
        print("CLAIM-MAP.md is OUT OF DATE; run scripts/claim_map.py")
        return 1
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(CLAIMS)} claims")
    if missing:
        print(f"FAIL — {missing} cited path(s) do not exist")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
