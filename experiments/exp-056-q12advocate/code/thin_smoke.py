"""exp-056-q12advocate: local smoke test of `kernels/iso/prc-iso-thin.py`'s producer patches.

For every alias and member it takes the exact `-c` program the kernel would run, points its paths
at this Mac, replaces the final producer call with a check, stubs the heavy step (feature build or
fit), and asserts that the patched entry point thins the training rows to `n_keep`, the rows kept
are the kernel's hash selection, and the validation month is not thinned. Label-free: it reads
`MVT_ID_mvt` and `PHASE_mvt` only. No model is fitted.

    .venv/bin/python experiments/exp-056-q12advocate/code/thin_smoke.py
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e56 as E                                                       # noqa: E402

KSRC = E.EXP / "kernels/iso/prc-iso-thin.py"
src = KSRC.read_text()
ns: dict = {"__name__": "thin_defs"}
exec(compile(src[:src.index('banner("inputs")')], str(KSRC), "exec"), ns)   # definitions only
THIN, SALT = ns["THIN"], ns["SALT"]
TMP = Path(tempfile.mkdtemp(prefix="thin-smoke-"))


def dep_ids(month):
    t = pd.read_parquet(E.D.month_file(month), columns=["MVT_ID_mvt", "PHASE_mvt"])
    return t.loc[t["PHASE_mvt"].to_numpy() == "DEP", "MVT_ID_mvt"].to_numpy("int64")


STUB = {
    "F1-fresh": ("runpy.run_path(str(pathlib.Path('experiments/exp-018-fresh/code/run.py').resolve()),\n"
                 "               run_name='__main__')",
                 "def _mt(m, rebuild=False):\n"
                 "    t = pd.read_parquet(PD.month_file(m), columns=['MVT_ID_mvt', 'PHASE_mvt'])\n"
                 "    t = t[t['PHASE_mvt'].to_numpy() == 'DEP']\n"
                 "    return pd.DataFrame({'MVT_ID_mvt': t['MVT_ID_mvt'].to_numpy('int64')})\n"
                 "_D.month_table = _mt\n"
                 "tr = _D.load(PD.FOLDS[_ALIAS].train_months)\n"
                 "va = _D.load([PD.FOLDS[_ALIAS].valid_month])\n"),
    "L2-hub8725": ("runpy.run_path(str(pathlib.Path('run.py').resolve()), run_name='__main__')",
                   "def _lc(m):\n"
                   "    t = pd.read_parquet(PD.month_file(m), columns=['MVT_ID_mvt', 'PHASE_mvt'])\n"
                   "    t = t[t['PHASE_mvt'].to_numpy() == 'DEP']\n"
                   "    return pd.DataFrame({'MVT_ID_mvt': t['MVT_ID_mvt'].to_numpy('int64')})\n"
                   "frames.load_cached = _lc\n"
                   "tr, va = frames.fold_frames(_ALIAS)\n"
                   "assert frames.FAMILY[_ALIAS] == 'm11'\n"),
    "T-tailv4": ("deliver.main(out_name='_iso-{f}.parquet')",
                 "_got = {}\n"
                 "TM._got = _got          # the swapped code runs with prc.tail_model's globals\n"
                 "def _stub(self, train_frames, *a, **k):\n"
                 "    _got['keep'] = k['keep']\n"
                 "    return self\n"
                 "_fit.__code__ = _stub.__code__\n"
                 "trf = pd.concat([pd.read_parquet(PD.month_file(m), columns=['MVT_ID_mvt', 'PHASE_mvt'])\n"
                 "                 for m in PD.FOLDS[_ALIAS].train_months], ignore_index=True)\n"
                 "TM.TailModel().fit(trf, n_jobs=1, noac_head='v4')\n"
                 "kp = _got['keep']\n"
                 "dep = trf['PHASE_mvt'].to_numpy() == 'DEP'\n"
                 "tr = trf.loc[dep & kp.reindex(trf.index).to_numpy()]\n"
                 "assert deliver.FOLDS == (_ALIAS,)\n"
                 "va = pd.DataFrame({'MVT_ID_mvt': PD.load_month(PD.FOLDS[_ALIAS].valid_month, "
                 "columns=['MVT_ID_mvt', 'PHASE_mvt']).query('PHASE_mvt == \"DEP\"')['MVT_ID_mvt']})\n"),
}
CHECK = ("import json as _j\n"
         "pathlib.Path(_OUT).write_text(_j.dumps({'train_rows': int(len(tr)), 'valid_rows': int(len(va)),\n"
         "    'train_ids_sha256': __import__('hashlib').sha256(np.ascontiguousarray(\n"
         "        np.sort(tr['MVT_ID_mvt'].to_numpy('int64')).astype('float64')).tobytes()).hexdigest()}))\n")

out = {}
for f, spec in THIN.items():
    fo = E.D.FOLDS[spec["base"]]
    ids = np.concatenate([dep_ids(m) for m in fo.train_months])
    assert len(ids) == spec["train_departures"], (f, len(ids))
    hv = np.array([int(hashlib.sha256(f"{SALT}|{int(i)}".encode()).hexdigest()[:15], 16) for i in ids],
                  dtype="int64")
    keep = np.sort(ids[np.argsort(hv, kind="stable")[:spec["n_keep"]]])
    np.save(TMP / f"keep-{f}.npy", keep)
    want = hashlib.sha256(np.ascontiguousarray(keep.astype("float64")).tobytes()).hexdigest()
    n_valid = len(dep_ids(fo.valid_month))
    for member, m in ns["MEMBERS"].items():
        code = m["cmd"](f, 1)[1]
        last, stub = STUB[member]
        last = last.format(f=f)
        assert code.count(last) == 1, (member, "the final producer call was not found")
        code = code.replace(last, stub + CHECK)
        code = code.replace("/tmp/repo", str(E.ROOT)).replace("/tmp/keep-", str(TMP) + "/keep-") \
                   .replace("/tmp/thin-", str(TMP) + "/thin-")
        res = TMP / f"res-{member}-{f}.json"
        head = f"_ALIAS = {f!r}\n_OUT = {str(res)!r}\n"
        # the stubs need names the prelude defines, so they are appended after it
        p = subprocess.run([sys.executable, "-c", head + code], cwd=str(E.ROOT / m["cwd"]),
                           capture_output=True, text=True,
                           env={"PYTHONPATH": str(E.ROOT), "PATH": "/usr/bin:/bin"})
        if p.returncode != 0:
            raise SystemExit(f"{member} {f}: rc={p.returncode}\n{p.stderr[-3000:]}")
        r = json.loads(res.read_text())
        rec = json.loads((TMP / f"thin-{f}.json").read_text())
        ok = (r["train_rows"] == spec["n_keep"] and r["train_ids_sha256"] == want
              and r["valid_rows"] == n_valid)
        out[f"{member} {f}"] = {**r, "thin_record": rec, "kept_ids_sorted_sha256": want,
                                "valid_departures": n_valid, "ok": ok}
        print(f"{member:11s} {f}: train {r['train_rows']} (want {spec['n_keep']}), valid "
              f"{r['valid_rows']} (want {n_valid}), ids {'match' if r['train_ids_sha256'] == want else 'DIFFER'}"
              f", record {rec}", flush=True)
        (TMP / f"thin-{f}.json").unlink()
bad = [k for k, v in out.items() if not v["ok"]]
(E.ANALYSIS / "thin-smoke.json").write_text(json.dumps(
    {"what": __doc__.strip().splitlines()[0], "utc": E.utc(), "kernel_sha256": E.sha_file(KSRC),
     "code_sha256": E.code_shas(), "results": out, "all_ok": not bad}, indent=1, sort_keys=True))
if bad:
    raise SystemExit(f"FAILED: {bad}")
print("all patches thin exactly as the kernel will")
