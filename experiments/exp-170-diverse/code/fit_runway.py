"""C50 fixed-round runway sequence: four to seven neighbours each side."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def windows(meta, mask, halo, core=4):
    groups = meta.group.to_numpy()
    bounds = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(meta)]
    out = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if not mask[lo:hi].any():
            continue
        if not mask[lo:hi].all():
            raise RuntimeError("an airport-day crosses a fit domain")
        for c in range(lo, hi, core):
            out.append((max(lo, c - halo), min(hi, c + core + halo), c, min(hi, c + core)))
    return out


def batch(chunks, numeric, category, anchor, labels, device):
    lengths = [b - a for a, b, _, _ in chunks]
    width = max(lengths)
    n = np.zeros((len(chunks), width, numeric.shape[1]), dtype="float32")
    c = np.zeros((len(chunks), width, category.shape[1]), dtype="int64")
    a = np.zeros((len(chunks), width), dtype="float32")
    y = np.zeros_like(a)
    mask = np.zeros_like(a, dtype=bool)
    for k, (start, stop, central, end) in enumerate(chunks):
        z = stop - start
        n[k, :z], c[k, :z], a[k, :z] = numeric[start:stop], category[start:stop], anchor[start:stop]
        if labels is not None:
            y[k, :z] = labels[start:stop]
        mask[k, central - start:end - start] = True
    return (torch.from_numpy(n).to(device), torch.from_numpy(c).to(device),
            torch.from_numpy(a).to(device), torch.from_numpy(y).to(device),
            torch.from_numpy(mask).to(device), torch.tensor(lengths, dtype=torch.long))


class Model(nn.Module):
    def __init__(self, n_numeric, vocab_sizes, mode):
        super().__init__()
        dims = [min(16, max(3, int(np.ceil(np.sqrt(v))))) for v in vocab_sizes]
        self.embeds = nn.ModuleList([nn.Embedding(v, d, padding_idx=0) for v, d in zip(vocab_sizes, dims)])
        self.encoder = nn.Sequential(nn.Linear(n_numeric + sum(dims), 128), nn.LayerNorm(128),
                                     nn.SiLU(), nn.Dropout(.05), nn.Linear(128, 64), nn.SiLU())
        self.mode = mode
        if mode != "row":
            self.gru = nn.GRU(64, 64, num_layers=2, batch_first=True, bidirectional=True, dropout=.05)
        self.head = nn.Sequential(nn.Linear(64 if mode == "row" else 192, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, n, c, lengths):
        h = self.encoder(torch.cat([n] + [e(c[:, :, i]) for i, e in enumerate(self.embeds)], dim=-1))
        if self.mode != "row":
            packed = pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
            ctx, _ = self.gru(packed)
            ctx, _ = pad_packed_sequence(ctx, batch_first=True, total_length=n.shape[1])
            h = torch.cat([h, ctx], dim=-1)
        return self.head(h).squeeze(-1) * 300.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["sequence-runway"], required=True)
    ap.add_argument("--loss", choices=["huber", "mse"], default="huber")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=17001)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.out.exists():
        raise RuntimeError("fit directory already exists")
    a.out.mkdir(parents=True)
    start = time.time()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(a.device)
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., 10 * 1024 ** 3 / total), 0)
    rec = json.loads((a.inputs / "MANIFEST.json").read_text())
    for name, expected in rec["files"].items():
        if sha(a.inputs / name) != expected:
            raise RuntimeError("input hash changed: " + name)
    meta = pd.read_parquet(a.inputs / "rows.parquet")
    num, cat, anchor, labels = [np.load(a.inputs / (n + ".npy"), mmap_mode="r")
                                 for n in ["numeric", "category", "anchor", "labels"]]
    train = (meta.year.to_numpy() == 2025)
    if not a.final:
        train &= ~meta.month.isin([1, 7]).to_numpy()
    held = (meta.year.to_numpy() == 2025) & meta.month.isin([1, 7]).to_numpy()
    rank = meta.year.to_numpy() == 2026
    months = sorted(meta.loc[train, "month"].unique().tolist())
    assert months == (list(range(1, 13)) if a.final else [2, 3, 4, 5, 6, 8, 9, 10, 11, 12])
    assert np.isfinite(labels[train]).all()
    if not a.final:
        assert set(meta.loc[train, "group"]).isdisjoint(meta.loc[held, "group"])
    halo = 4
    chunks = windows(meta, train, halo)
    # Each supervised row appears once, while context rows can appear in more windows.
    assert sum(e - c for _, _, c, e in chunks) == int(train.sum())
    sizes = [len(rec["vocab"][c]) + 1 for c in rec["category"]]
    model = Model(num.shape[1], sizes, a.mode).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, a.epochs, eta_min=.0002)
    rng = np.random.default_rng(a.seed)
    history = []
    print("start", vars(a), "train", int(train.sum()), "chunks", len(chunks), "params",
          sum(p.numel() for p in model.parameters()), flush=True)
    for epoch in range(a.epochs):
        model.train()
        order = rng.permutation(len(chunks))
        loss_sum, count = 0., 0
        t = time.time()
        for j in range(0, len(order), a.batch_size):
            cs = [chunks[k] for k in order[j:j + a.batch_size]]
            n, c, an, y, on, lengths = batch(cs, num, cat, anchor, labels, device)
            optim.zero_grad(set_to_none=True)
            delta = model(n, c, lengths)[on] / 300.
            target = (y[on] - an[on]) / 300.
            if a.loss == "huber":
                loss = nn.functional.smooth_l1_loss(delta, target, beta=2.)
            else:
                loss = nn.functional.mse_loss(delta, target.clamp(-24, 24))
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optim.step()
            k = int(on.sum())
            loss_sum += loss.item() * k
            count += k
        scheduler.step()
        row = {"epoch": epoch + 1, "train_loss": loss_sum / count, "seconds": time.time() - t}
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save({"state": model.state_dict(), "numeric": num.shape[1], "vocab_sizes": sizes,
                    "mode": a.mode, "epoch": epoch + 1}, a.out / "model.pt")
    model.eval()
    for domain, mask in [("jjrefit", held)] + ([("ranking2026", rank)] if a.final else []):
        if domain == "jjrefit" and a.final:
            continue
        score_chunks = windows(meta, mask, halo)
        prediction = np.full(len(meta), np.nan, dtype="float64")
        with torch.no_grad():
            for j in range(0, len(score_chunks), a.batch_size):
                cs = score_chunks[j:j + a.batch_size]
                n, c, an, _, _, lengths = batch(cs, num, cat, anchor, None, device)
                pred = (an + model(n, c, lengths)).clamp(min=60.).cpu().numpy()
                for k, (lo, hi, central, end) in enumerate(cs):
                    assert np.isnan(prediction[central:end]).all()
                    prediction[central:end] = pred[k, central - lo:end - lo]
        assert np.isfinite(prediction[mask]).all()
        out = meta.loc[mask, ["MVT_ID_mvt", "ADEP_mvt", "month", "date"]].copy()
        out["pred_seq170"] = prediction[mask]
        out.to_parquet(a.out / (domain + ".parquet"), index=False)
        print(domain, len(out), float(out.pred_seq170.mean()), flush=True)
    result = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              "train_months": months, "train_rows": int(train.sum()), "history": history,
              "window": {"group": "airport-day-runway", "halo": 4, "core": 4},
              "inputs_manifest_sha256": sha(a.inputs / "MANIFEST.json"),
              "code_sha256": sha(__file__), "seconds": time.time() - start,
              "train_ids_sha256": hashlib.sha256(meta.loc[train, "MVT_ID_mvt"].to_numpy("int64").tobytes()).hexdigest(),
              "files": {p.name: sha(p) for p in a.out.iterdir()}}
    if device.type == "cuda":
        result["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
    (a.out / "FIT.json").write_text(json.dumps(result, indent=1) + "\n")
    print("done", result["seconds"], flush=True)


if __name__ == "__main__":
    main()
