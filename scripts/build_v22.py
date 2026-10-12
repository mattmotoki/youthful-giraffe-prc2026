"""The chain that produces the uploaded submission `youthful-giraffe_v22.parquet`.

    .venv/bin/python scripts/build_v22.py --plan      # the ordered chain, with commands
    .venv/bin/python scripts/build_v22.py --verify    # hash what is on disk against it
    .venv/bin/python scripts/build_v22.py --json      # the same, machine-readable

**This script fits nothing and writes no prediction.**  The shipped model is a
chain of fifteen steps that spans two Kaggle GPU lanes and about a hundred
CPU-hours, and no single process runs it; `scripts/build_candidate.py` covers
the exp-015 release blends and has no stacker, no surface head and no config
that names this chain (`docs/REPRODUCE.md` §3).  What a reader needs and did
not have is ONE place that says, in order, which command produces which file
from which inputs, and what that file's SHA-256 must be.  That is this file.

The later entries v23, v24 and v32 add saved ranking vectors to the v22 bytes;
`scripts/build_v32.py` composes them and checks each hash.

`--verify` is the executable half: for every step whose output is present it
compares the bytes against the recorded digest and reports `ok`, `absent` or
`MISMATCH`.  On a clean clone every output is `absent`, which is correct —
`submissions/` and `artifacts/` are not distributed. On the machine that built
the submission every present step must be `ok`;
a `MISMATCH` means the checkout and the record disagree and exits non-zero.

Naming, because two numbering schemes cross here and confusing them is the
easiest mistake to make with this record:

* `v<N>` in `docs/PUBLIC-SUBMISSIONS.md` and below is the BOARD upload number.
* `--base-set=v<M>` inside `experiments/exp-046-surfacefit/code/serve.py` is
  that lane's own base-set version.  They are different sequences: base set
  `v25` produces candidate `exp-046-surface-v11.parquet`, which went to the
  board as upload **v22**.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if not (ROOT / "LICENSE").is_file() or not (ROOT / "prc" / "data.py").is_file():
    raise SystemExit(
        f"build_v22.py: {ROOT} is not this repository.  Run the script from its\n"
        f"place in the tree; a copy elsewhere resolves every output to a missing\n"
        f"path and would report a vacuous pass.")

#: One step of the chain.
#:   id        short name used in the plan
#:   board     the board upload this file became, or None if it was never uploaded
#:   what      one sentence: what the step does
#:   produces  repo-relative path of the file the step writes
#:   sha256    the digest the experiment record states for that file
#:   command   the command as the experiment record states it
#:   record    the write-once record that states the digest and the gate
#:   cost      "fit" (a model is trained), "compose" (bytes only), "fetch"
STAGES: tuple[dict, ...] = (
    dict(id="data", board=None, cost="fetch",
         what="the organisers' twelve 2025 monthly files plus ranking.parquet and "
              "submitting.parquet, placed in data/raw/ (not redistributed here)",
         produces="data/raw/ranking.parquet", sha256=None,
         command="download from the organisers' bucket; see docs/REPRODUCE.md §2",
         record="docs/input-rule.md"),
    dict(id="surface", board=None, cost="fetch",
         what="the adsb.lol ground-trace slice of the ten airport boxes for 427 days, "
              "reduced to one summary row per aircraft ground segment",
         produces="artifacts/exp-045/summaries", sha256=None,
         command="bash experiments/exp-045-harvest/code/push_kernel.sh <YYYY-MM> "
                 "\"$(python3 experiments/exp-045-harvest/code/days.py <YYYY-MM>)\" "
                 "&& bash experiments/exp-045-harvest/code/pull_kernel.sh <YYYY-MM>",
         record="docs/EXTERNAL-DATA.md"),
    dict(id="members", board=None, cost="fit",
         what="the seven blend members. `month_chain.sh` below fits them for ONE "
              "month and is how each extension month of the shipped head was "
              "built; the twelve-month ranking fits behind base v12 predate it "
              "and their per-member producers are in docs/REPRODUCE.md §5. "
              "`V3-allin` cannot be refitted here at all -- see the BLOCKED note "
              "in docs/REPRODUCE.md §3",
         produces="experiments/exp-012-blend2/published", sha256=None,
         command="PRC_LANE_MONTH=<YYYY-MM> PRC_LANE_FOLD=<fold> bash "
                 "experiments/exp-047-queue/work/month_chain.sh   # per month, in order: "
                 "work/member_folds.py (T-tailv2, D1-winter) -> work/nmtail_fold.py -> "
                 "work/l2hub_hedge.sh -> work/c2cat_hedge.sh -> work/f1_local.sh -> "
                 "work/promote_hedge.py -> work/base6.py -> code/emit.py -> "
                 "work/block_verify.py",
         record="experiments/exp-012-blend2/RESULTS.md"),
    dict(id="stack-5", board=3, cost="fit",
         what="blend-5's two per-regime non-negative least squares vectors as the anchor, "
              "and the LightGBM residual stacker over the seven members, fitted by rotation",
         produces="submissions/candidates/exp-026-stack-5.parquet",
         sha256="94f18e47a73265a6db66811ededfc3c935fb2a37fac04bf0cd34d45af708a2a1",
         command=".venv/bin/python experiments/exp-026-rebuild/code/serve_stack5.py "
                 "--variant keep-noac --name stack-5 && "
                 ".venv/bin/python experiments/exp-026-rebuild/finalise.py",
         record="experiments/exp-026-rebuild/analysis/stack5.json"),
    dict(id="v6", board=6, cost="fit",
         what="the 50-column 600 s transient-window block added inside the stacker",
         produces="submissions/candidates/exp-028-window.parquet",
         sha256="d55b1de9ce8fe7577c8c566c6efc57301d18484242aaa06ce808a111d38dabb0",
         command=".venv/bin/python experiments/exp-028-window/serve.py",
         record="experiments/exp-028-window/RESULTS.md §4"),
    dict(id="v7", board=7, cost="compose",
         what="EHAM's 38,182 departures served from stack-5 instead of from the window block",
         produces="submissions/candidates/exp-030-eham-out.parquet",
         sha256="67b70ee379e62a2a3bc9d91c9d771112baf3233efa3f74f4f95adbd7a3c03ef7",
         command=".venv/bin/python experiments/exp-030-window2/serve.py --arm eham-out "
                 "--name exp-030-eham-out "
                 "--from submissions/candidates/exp-028-window.parquet "
                 "--exclude-airports EHAM",
         record="experiments/exp-030-window2/RESULTS.md §4"),
    dict(id="K2", board=9, cost="fit",
         what="the fresh lineage refitted with its own 68-column window block in "
              "F1-fresh's stacker slot; the anchor stays on the committed seven",
         produces="submissions/candidates/exp-027-k2.parquet",
         sha256="a3a139dbe8255d5dfc17aa69cfd4c65a15df26c2e76feb9eeaf4dbad03ed6ba5",
         command=".venv/bin/python experiments/exp-027-step/serve.py",
         record="experiments/exp-027-step/analysis/served-exp-027-k2.json"),
    dict(id="v10", board=10, cost="compose",
         what="refit-bulk (the anchor refitted with K2 off the NM tail only) served at "
              "eight airports, EDDF and EHAM held at v7",
         produces="submissions/candidates/exp-036-refitbulk-mask.parquet",
         sha256="ad962763236e9b40ace45a5dac12945568bbce441ed0547ff8e814d248c72c78",
         command=".venv/bin/python experiments/exp-036-mask/serve.py --arm refitbulk-mask",
         record="experiments/exp-036-mask/analysis/arm-refitbulk-mask.json"),
    dict(id="v11", board=11, cost="compose",
         what="p = v10 + 0.217 * (stack-5 - v10) on every row outside EHAM and noac; "
              "the amplitude is stated in the record before the file is written",
         produces="submissions/candidates/exp-038-v10-stack5-blend.parquet",
         sha256="42ca5ee6410938d56c8830e9ec3980d806633c3b2ff15963c0ffe2fe97c39f8c",
         command=".venv/bin/python experiments/exp-038-packet/code/serve.py --s 0.217 "
                 "--name exp-038-v10-stack5-blend "
                 "--why exp-038-packet/RESULTS.md-2a",
         record="experiments/exp-038-packet/RESULTS.md §2a"),
    dict(id="v12", board=12, cost="fit",
         what="v11's composition with only its K2 member replaced by the climbed member "
              "(phi, surface and de-icing extras, three seeds averaged)",
         produces="submissions/candidates/exp-035-k2climb-v11member.parquet",
         sha256="48dd572d61ff2403f13614ce04bdc3a8fd0dbdb5175bef7eb143650ea520ec52",
         command=".venv/bin/python experiments/exp-035-k2climb/code/serve_v11member.py",
         record="experiments/exp-035-k2climb/analysis/served-exp-035-k2climb-v11member.json"),
    dict(id="v13", board=13, cost="fit",
         what="the first surface head: a LightGBM residual on the adsb.lol observation "
              "block, served on the with-clock class, LFPG held",
         produces="submissions/candidates/exp-046-surface.parquet",
         sha256="79a301627e918140933bb4c1dcf9a9249a63bd81669f1793e252f821aa1dee51",
         command="OMP_NUM_THREADS=3 .venv/bin/python "
                 "experiments/exp-046-surfacefit/code/serve.py c_lgb_shipped "
                 "--hold-airport=LFPG",
         record="experiments/exp-046-surfacefit/analysis/serve.json"),
    dict(id="v14", board=14, cost="fit",
         what="the queue lane's runway-neighbourhood arm on two no-clock cells, "
              "composed on v13's bytes",
         produces="submissions/candidates/exp-047-queue.parquet",
         sha256="519ca89f381056afb453ca62d64df622fcdbbdb14d8607cb0307c2be077676b2",
         command="OMP_NUM_THREADS=2 .venv/bin/python "
                 "experiments/exp-047-queue/code/serve.py --arm N",
         record="experiments/exp-047-queue/analysis/serve.json"),
    dict(id="v15", board=15, cost="fit",
         what="the with-clock class re-served by the 127-leaf head, on v14's bytes",
         produces="submissions/candidates/exp-046-surface-v2.parquet",
         sha256="eabd824d925f15223db452557fe113e286ca8ce49302e8a97e79f060c0360ea1",
         command="OMP_NUM_THREADS=3 .venv/bin/python "
                 "experiments/exp-046-surfacefit/code/serve.py "
                 "'c_lgb_shipped@leaves_127x300_lr050' --base-set=v14",
         record="docs/PUBLIC-SUBMISSIONS.md row 15"),
    dict(id="v16", board=16, cost="fit",
         what="the 255-leaf head plus the frozen label-free p99.5 tail guard, on v15's bytes",
         produces="submissions/candidates/exp-046-surface-v3.parquet",
         sha256="6f8368da9b8da818a726c7cc6cf20de6ff7777196f1e78d41015a8cda441fdf7",
         command="OMP_NUM_THREADS=3 .venv/bin/python "
                 "experiments/exp-046-surfacefit/code/serve.py "
                 "'c_lgb_shipped@leaves_255x300_lr050+guard995' --hold-airport=LFPG "
                 "--base-set=v15",
         record="experiments/exp-046-surfacefit/analysis/serve-v15.json"),
    dict(id="v17", board=17, cost="fit",
         what="the queue arm re-priced on three no-clock cells under the lit rule, "
              "composed on v16's bytes",
         produces="submissions/candidates/exp-047-queue-lit.parquet",
         sha256="4de323fc4fd9d4865b201b0dd0da1ddcbf7de9e11f1d9bc8af07d293585001ea",
         command="OMP_NUM_THREADS=2 .venv/bin/python "
                 "experiments/exp-047-queue/code/serve.py --arm N --base v16",
         record="experiments/exp-047-queue/analysis/serve-lit.json"),
    dict(id="v18", board=18, cost="fit",
         what="the chain bundle: the with-clock arm refitted on the class carrying the "
              "turnaround pass, plus 2,456 rows admitted by the ungated pass and kept "
              "by the label-free gate; composed on v17's bytes",
         produces="submissions/candidates/exp-046-surface-v5.parquet",
         sha256="4d75706bfcc13d88f7556265cd008b3d3a312555955dca8eb528e4db5325fee8",
         command="OMP_NUM_THREADS=3 .venv/bin/python "
                 "experiments/exp-046-surfacefit/code/serve.py --ungated --base-set=v19 "
                 "--hold-airport=LFPG",
         record="experiments/exp-046-surfacefit/analysis/serve-v19.json"),
    dict(id="bag", board=None, cost="fit",
         what="the seasonal K = 10 bag itself: ten fits per served month on ten "
              "distinct 49-row class perturbations, averaged.  Round 1 finding 5: "
              "the chain named the serve and not the fit that the serve reads, so "
              "this step was missing from the route",
         produces="experiments/exp-046-surfacefit/work/d-seasonal-bag10-255-free.npz",
         sha256=None,
         command="OMP_NUM_THREADS=3 .venv/bin/python "
                 "experiments/exp-046-surfacefit/code/onemonthbag.py --stack seasonal",
         record="experiments/exp-046-surfacefit/analysis/seasonal-bag10.json"),
    dict(id="v22", board=22, cost="fit",
         what="THE SHIPPED FILE: the seasonal head — two K = 10 bag heads by served "
              "month, each fitted on the three folds plus four 2025 extension months — "
              "serving the with-clock class on v18's bytes",
         produces="submissions/candidates/exp-046-surface-v11.parquet",
         sha256="f11d34f88b62a43ec2a0cdbeec967a9ee4b7d92e50a5508aa4d30e5a78ea5e1c",
         command="OMP_NUM_THREADS=3 .venv/bin/python "
                 "experiments/exp-046-surfacefit/code/serve.py "
                 "'c_lgb_shipped@leaves_255x300_lr050+guard995' --base-set=v25 --ungated",
         record="experiments/exp-046-surfacefit/analysis/serve-v25.json"),
    dict(id="upload", board=22, cost="compose",
         what="the same bytes, uploaded as youthful-giraffe_v22.parquet; board 245.0207",
         produces="submissions/uploaded/youthful-giraffe_v22.parquet",
         sha256="f11d34f88b62a43ec2a0cdbeec967a9ee4b7d92e50a5508aa4d30e5a78ea5e1c",
         command="bash scripts/upload_candidate.sh "
                 "submissions/candidates/exp-046-surface-v11.parquet 22",
         record="docs/PUBLIC-SUBMISSIONS.md row 22"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def status(stage: dict) -> dict:
    p = ROOT / stage["produces"]
    if stage["sha256"] is None:
        return {"state": "not-hashed", "note": "an input or a directory, not one file"}
    if not p.is_file():
        return {"state": "absent"}
    got = sha256(p)
    return {"state": "ok" if got == stage["sha256"] else "MISMATCH", "sha256": got}


def plan() -> None:
    print("The chain that produces youthful-giraffe_v22.parquet, in order.\n")
    print(f"{'#':>2}  {'step':8s} {'board':>5s}  {'kind':8s} what it produces")
    for i, s in enumerate(STAGES, 1):
        b = f"v{s['board']}" if s["board"] else "-"
        print(f"{i:2d}  {s['id']:8s} {b:>5s}  {s['cost']:8s} {s['produces']}")
    print("\nEach step in full:\n")
    for i, s in enumerate(STAGES, 1):
        print(f"--- {i}. {s['id']}"
              + (f"  (board upload v{s['board']})" if s["board"] else ""))
        print(f"    {s['what']}")
        print(f"    command : {s['command']}")
        print(f"    produces: {s['produces']}")
        if s["sha256"]:
            print(f"    sha256  : {s['sha256']}")
        print(f"    record  : {s['record']}")
        print()
    fits = sum(1 for s in STAGES if s["cost"] == "fit")
    print(f"{fits} of the {len(STAGES)} steps train a model.  None of them runs here.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="print the ordered chain")
    ap.add_argument("--verify", action="store_true",
                    help="hash every present output against the record")
    ap.add_argument("--json", action="store_true", help="print a JSON record")
    ap.add_argument("--require-present", action="store_true",
                    help="fail when a hashed output is absent, not only when it differs")
    a = ap.parse_args()
    if not (a.plan or a.verify or a.json):
        a.plan = True

    if a.plan:
        plan()

    if not (a.verify or a.json):
        return 0

    # v22 consumes the exact, separately licensed ERA5 bundle. Verification is
    # not complete when prediction files happen to match but this input is absent.
    from release.era5_data import verify_bundle
    era5_spec = json.loads((ROOT / "release/ERA5-DATA-MANIFEST.json").read_text())
    era5_bundle = ROOT / "external-data" / era5_spec["bundle"]
    era5_error = None
    try:
        verify_bundle(era5_bundle, era5_spec)
    except ValueError as error:
        era5_error = str(error)

    rows = [dict(stage=s["id"], board=s["board"], produces=s["produces"],
                 expected=s["sha256"], **status(s)) for s in STAGES]
    bad = [r for r in rows if r["state"] == "MISMATCH"]
    if a.json:
        print(json.dumps({"stages": rows, "era5": "ok" if era5_error is None else era5_error,
                          "mismatches": [r["stage"] for r in bad]},
                         indent=1))
    else:
        print(f"\n{'step':8s} {'state':10s} produces")
        for r in rows:
            print(f"{r['stage']:8s} {r['state']:10s} {r['produces']}")
        n_ok = sum(1 for r in rows if r["state"] == "ok")
        n_abs = sum(1 for r in rows if r["state"] == "absent")
        print(f"\n{n_ok} verified, {n_abs} absent, {len(bad)} MISMATCH")
        if era5_error:
            print(f"ERA5 BLOCK — {era5_error}")
        if bad:
            print("FAIL — the checkout and the record disagree on:")
            for r in bad:
                print(f"  {r['produces']}\n    record {r['expected']}\n    disk   {r['sha256']}")
        elif n_ok == 0:
            # Round 1 finding 5: the old text said PASS here, which is what a
            # clean clone always saw -- an unverifiable claim dressed as a check.
            print("NOT VERIFIED — no output of the chain is present on this machine.\n"
                  "  That is expected on a clean clone: submissions/ and artifacts/ are\n"
                  "  git-ignored and are not distributed.  Nothing was checked.")
        else:
            print(f"PASS — the {n_ok} output(s) present on this machine match the record")
    missing = [r for r in rows if r["state"] == "absent"]
    if a.require_present and missing:
        return 1
    return 1 if bad or era5_error else 0


if __name__ == "__main__":
    sys.exit(main())
