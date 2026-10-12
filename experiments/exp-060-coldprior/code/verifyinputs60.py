"""exp-060-coldprior: verify every input artifact of the audit jobs against its recorded hash.

Review round 2, standards: `identity60.py`, `ceiling60_r1.py` and `banddraws60.py` read frozen
artifacts without first checking them against the hash the producing record carries.  Repairing
the three scripts and re-running them would change no number, because the artifacts are the ones
this job now checks.  So the repair here is the check itself, made once, over every artifact those
jobs read, against the record that produced it.  `banddraws60_r1.py` carries the check inline.

    .venv/bin/python experiments/exp-060-coldprior/code/verifyinputs60.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e60 as E                                                       # noqa: E402

sys.path.insert(0, str(E.ROOT / "experiments/exp-046-surfacefit/code"))
import nooverwrite as NO                                              # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    p = E.ANALYSIS / "verifyinputs60.json"
    NO.refuse_existing([p], a.overwrite)
    items, bad = {}, []

    def check(path, want, why):
        got = E.sha_file(E.ROOT / path)
        items[str(path)] = {"recorded": want, "measured": got, "equal": got == want, "read_by": why}
        if got != want:
            bad.append(str(path))

    oos = E.load_json(E.ANALYSIS / "oos.json")
    for r in E.ORDER:
        inp = E.load_json(E.ANALYSIS / f"inputs60-{r}.json")
        check(inp["npz"], inp["npz_sha256"], ["identity60", "ceiling60_r1", "align60", "grid60"])
        cold = E.load_json(E.ANALYSIS / f"cold-{r}.json")
        check(cold["table"]["path"], cold["table"]["sha256"], ["freeze60", "answers60"])
        fit = E.load_json(E.ANALYSIS / f"fit60-{r}.json")
        check(fit["pred_npz"], fit["pred_npz_sha256"], ["price60", "bound60"])
        for arm in ("base", "zero", "true"):
            rel = f"experiments/exp-060-coldprior/oof/{arm}-iso_{r}_v.parquet"
            check(rel, oos["artifacts"]["oof"][rel], ["align60", "ceiling60_r1", "check60"])
        for key in ("band_artifact", "grid_artifact"):
            rr = E.load_json(E.ANALYSIS / f"bound-r2-{r}.json")
            check(rr[key]["path"], rr[key]["sha256"], ["banddraws60_r1"])
        rb = E.load_json(E.ANALYSIS / f"bound-{r}.json")
        check(rb["band_artifact"]["path"], rb["band_artifact"]["sha256"], ["banddraws60"])
        r1 = E.load_json(E.ANALYSIS / f"bound-r1-{r}.json")
        check(r1["band_artifact"]["path"], r1["band_artifact"]["sha256"], ["banddraws60"])
        g = E.load_json(E.ANALYSIS / f"grid60-{r}.json")
        check(g["row_losses"]["path"], g["row_losses"]["sha256"], ["the record itself"])
    for key in ("rows", "draws"):
        aa = oos["artifacts"][key]
        check(aa["path"], aa["sha256"], ["bound60", "check60"])

    rec = {"what": ("every input artifact of the audit jobs, checked against the hash its "
                    "producing record carries (review round 2, standards)"),
           "utc": E.utc(), "items": items, "count": len(items), "all_equal": not bad,
           "mismatches": bad, "code_sha256": E.code_shas()}
    NO.refuse_existing([p], a.overwrite)
    p.write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(json.dumps({"checked": len(items), "all_equal": rec["all_equal"], "mismatches": bad}))
    if bad:
        raise SystemExit(f"REFUSING: {len(bad)} artifacts do not match their recorded hash")
    print(f"written {p.relative_to(E.ROOT)}")


if __name__ == "__main__":
    main()
