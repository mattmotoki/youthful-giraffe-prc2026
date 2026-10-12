"""exp-185: the exp-018 month feature table (`artifacts/exp-018-fresh/cache/<month>.parquet`, built by
exp-018 `dataset.month_table`) for 2026 months, written to a folder of this worker with its own copy
of the append-only category vocabulary.

    PRC_RANKING_MODE=final .venv/bin/python experiments/exp-185-final-data/code/exp018_cache.py \
        <out_dir> <vocab.json to copy> <month> [<month> ...]

The table is built by the unchanged exp-018 code of $PRC_SRC (a `git archive` of a main commit; default
this checkout); the wrapper only points its CACHE and VOCAB at <out_dir>, so the shared vocabulary is
never written. Set PRC_DATA_DIR to the raw data folder. New category values of a month get codes
appended after the existing ones, in the order of the months given.
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(os.environ.get("PRC_SRC", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp-018-fresh/code"))
import dataset as D18  # noqa: E402
from prc import data  # noqa: E402

out, vocab_src, months = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
out.mkdir(parents=True, exist_ok=True)
if not (out / "vocab.json").exists():
    shutil.copy2(vocab_src, out / "vocab.json")
D18.CACHE, D18.VOCAB = out, out / "vocab.json"
print("mode", data.ranking_mode(), "file", data.ranking_file(), flush=True)
for m in months:
    assert m in data.rank_months(), (m, data.rank_months())
    X = D18.month_table(m, rebuild=True)
    assert X["MVT_ID_mvt"].is_unique and X["month"].eq(m).all()
    print(m, X.shape, flush=True)
