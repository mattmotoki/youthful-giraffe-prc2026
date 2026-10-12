"""exp-061-sparseprior: the label-free threshold census disclosed in SPEC 1.6.

It reads the hash-pinned exp-060 input archive's `nA_*` and `prior_true_*` arrays and the
vocabulary, and counts, for every rotation, family and registered threshold: the columns below the
threshold, the (family, airport) groups, the groups with fewer than two columns (which the primary
test excludes), and how many of the columns below the threshold the cold table gives a non-zero
value.  No label, no loss, no selection.  It exists because SPEC reading (a) turns on it.

    .venv/bin/python experiments/exp-061-sparseprior/code/census61.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e61 as E                                                       # noqa: E402


def main() -> None:
    rows = []
    for r in E.ORDER:
        rec = E.load_json(E.guard_open(E.E60EXP / f"analysis/inputs60-{r}.json", "census source"))
        npz = E.guard_open(E.ROOT / rec["npz"], "census source")
        if E.sha_file(npz) != rec["npz_sha256"]:
            raise SystemExit(f"{npz.name} is not exp-060's frozen input archive")
        z = np.load(npz)
        vocab = json.loads(Path(E.ROOT / rec["vocab_path"]).read_text())
        for f in E.PRIOR_FAMS:
            nA = z[f"nA_{f}"]
            pr = z[f"prior_true_{f}"]
            apt = np.array([str(v).split("|")[0] for v in vocab[f]])
            for ns in E.THRESHOLDS:
                m = nA < ns
                g = collections.Counter(apt[m])
                rows.append({"rotation": r, "family": f, "n_star": ns,
                             "columns_below": int(m.sum()), "groups": len(g),
                             "singleton_groups": sum(1 for v in g.values() if v < 2),
                             "columns_in_kept_groups": sum(v for v in g.values() if v >= 2),
                             "nonzero_true_below": int((pr[m] != 0).sum())})
        z.close()
    for x in rows:
        print("{rotation:4s} {family:5s} n*<{n_star:<4d} columns {columns_below:5d}  groups {groups:3d}"
              "  singleton {singleton_groups:2d}  kept {columns_in_kept_groups:5d}"
              "  nonzero true {nonzero_true_below:4d}".format(**x), flush=True)
    print(json.dumps({"what": "SPEC 1.6 label-free threshold census", "label_reads": [],
                      "opened": E.OPENED, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
