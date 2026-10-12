"""Pre-fit gates for the future registered producer; this module never fits."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

THREADS = "3"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "PRC_THREADS"):
    os.environ[_name] = THREADS

RAW_SHA256 = {
 "training_2025-01-01_2025-02-01.parquet":"a4dc8f152c082c650cd7eb745fe9c641089f34f74df127e6944a188990a0ed94",
 "training_2025-02-01_2025-03-01.parquet":"b6f9ff2cdb1cfe07c5442f4e1f11bbcdbeafe52852dcb0b8175f62ab84c6a066",
 "training_2025-03-01_2025-04-01.parquet":"58fde7e5f8b539546af63b41d7f6ec0f2884fdc3d3dd9b6bfca7c59a420dcdd3",
 "training_2025-04-01_2025-05-01.parquet":"9ffe0cf5741c4863799c41be81475dfaecedabe0ba687b8f8e4eacba992f9390",
 "training_2025-05-01_2025-06-01.parquet":"5481c0642a8fbb4dc395f93bc713c57f4cc16854fb361d404967077e1fce2ff9",
 "training_2025-06-01_2025-07-01.parquet":"c3ef5080f2cc636c3416baf1555f7d0ee3519f9046e8043c1c0c8f017a244025",
 "training_2025-07-01_2025-08-01.parquet":"34d1c5bba7c8596a1c20c09edd3cb2f63277b5a684fa801911660fe0eaf63c9d",
 "training_2025-08-01_2025-09-01.parquet":"6c8d2aa7a1b6c9dff9a0305a37eb02de1e87a68d07557264f41b77738272c540",
 "training_2025-09-01_2025-10-01.parquet":"955308a4000947e32a00586d3ee13928949d2231b67f84f6dfb0631893f3754a",
 "training_2025-10-01_2025-11-01.parquet":"4a31a35601c1bdcd74b8cd6379231ba516a48324e4c2478bf1c9360cd23e687f",
 "training_2025-11-01_2025-12-01.parquet":"ac36b17db7a537e3baeb682fa4763dacaa1fcc5d9bfa6bf32b3ee5d9f91300b0",
 "training_2025-12-01_2026-01-01.parquet":"2d357f2f403a86778a69e93f72d8754effe77f1ac0942399b3d1b7728241f724",
}
FORBIDDEN_ROOTS = (
 "experiments/exp-056-q12advocate", "experiments/exp-060-sparsecold",
 "experiments/exp-061-sparseprior", "experiments/exp-063-sparseholdout",
 "experiments/exp-065-sparseholdout-retry",
)
FORBIDDEN_WORDS = ("prediction", "loss", "setting", "bound", "result", "cold table")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_raw(data_dir):
    data_dir = Path(data_dir)
    actual = {name: sha256(data_dir / name) for name in RAW_SHA256}
    if actual != RAW_SHA256:
        raise RuntimeError("raw-month hash mismatch before fit")
    return actual


def reject_forbidden(paths):
    for item in map(str, paths):
        low = item.lower()
        if any(root in item for root in FORBIDDEN_ROOTS) or any(word in low for word in FORBIDDEN_WORDS):
            raise ValueError(f"forbidden producing input: {item}")
