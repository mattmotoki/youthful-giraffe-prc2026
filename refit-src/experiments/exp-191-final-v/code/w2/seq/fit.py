"""Fit the airport-context sequence member s1 on one GPU.

Token = one movement at the same airport within 40 min (12 each side + the departure). Transformer over
the tokens; the departure token's output plus the center features give a correction to the anchor
clip(NM taxi, 60, 7200), in 300 s units, smooth-L1. Joined typed departures with labels below 20,000 s.

    python3 w2/seq/fit.py --mode held --seed 1 --out work/seq/s1-held-1.npy     (no Jan, Jul 2025, no buffer days)
    python3 w2/seq/fit.py --mode full --seed 1 --out work/seq/s1-full-1.npy     (all twelve months)
The output holds a prediction for every departure row of work/seq (NaN where no anchor).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "work/seq"
BUFFER = {(12, 31), (2, 1), (6, 30), (8, 1)}


class Net(nn.Module):
    def __init__(self, n_tok, tok_sizes, n_c, c_sizes, d=128, layers=3):
        super().__init__()
        self.tok_emb = nn.ModuleList([nn.Embedding(s, 16) for s in tok_sizes])
        self.tok_in = nn.Linear(n_tok + 16 * len(tok_sizes) + 3, d)
        enc = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.c_emb = nn.ModuleList([nn.Embedding(s, min(32, max(4, int(s ** 0.35) * 2))) for s in c_sizes])
        c_dim = n_c + sum(e.embedding_dim for e in self.c_emb)
        self.c_mlp = nn.Sequential(nn.Linear(c_dim, 256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, d))
        self.head = nn.Sequential(nn.Linear(2 * d, 256), nn.SiLU(), nn.Linear(256, 1))
        self.center = None

    def forward(self, tnum, tcat, dt, same, pad, cnum, ccat):
        e = torch.cat([emb(tcat[..., j]) for j, emb in enumerate(self.tok_emb)], -1)
        center_flag = torch.zeros_like(dt); center_flag[:, dt.shape[1] // 2] = 1
        h = self.tok_in(torch.cat([tnum, e, dt[..., None], same[..., None], center_flag[..., None]], -1))
        h = self.enc(h, src_key_padding_mask=pad)
        hc = h[:, dt.shape[1] // 2]
        ce = torch.cat([cnum] + [emb(ccat[:, j]) for j, emb in enumerate(self.c_emb)], -1)
        return self.head(torch.cat([hc, self.c_mlp(ce)], -1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["held", "full"], required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data", type=Path, default=Path("work/seq"))
    ap.add_argument("--amp", action="store_true", help="float16 autocast with a gradient scaler")
    ap.add_argument("--bench", type=int, default=0, help="time this many steps, then stop")
    ap.add_argument("--hb", type=float, default=1.0, help="smooth-L1 threshold in 300 s units")
    a = ap.parse_args()
    global D
    D = a.data if a.data.is_absolute() else ROOT / a.data
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")
    meta = json.loads((D / "meta.json").read_text())
    tok_num = torch.from_numpy(np.load(D / "tok_num.npy")).to(dev)
    tok_cat = torch.from_numpy(np.load(D / "tok_cat.npy").astype(np.int64)).to(dev)
    nbr = np.load(D / "nbr.npy"); dt = np.load(D / "dt.npy"); same = np.load(D / "same_rw.npy")
    c_num = np.load(D / "c_num.npy"); c_cat = np.load(D / "c_cat.npy").astype(np.int64)
    y = np.load(D / "y.npy").astype(np.float64); nm = np.load(D / "nm_raw.npy").astype(np.float64)
    month = np.load(D / "month.npy"); dom = np.load(D / "dom.npy"); is26 = np.load(D / "is2026.npy")
    anchor = np.clip(nm, 60, 7200)
    has = np.isfinite(nm) & (c_cat[:, meta["c_cat"].index("actype")] > 0)
    lab = has & ~is26 & np.isfinite(y) & (y < 20000)
    if a.mode == "held":
        buf = np.array([(int(m), int(x)) in BUFFER for m, x in zip(month, dom)])
        lab &= ~np.isin(month, [1, 7]) & ~buf
    tr = np.flatnonzero(lab)
    rng = np.random.default_rng(a.seed)
    target = np.clip((y - anchor) / 300.0, -10, 20)
    net = Net(tok_num.shape[1], meta["tok_cat_sizes"], c_num.shape[1], meta["c_cat_sizes"], d=a.d, layers=a.layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    lossf = nn.SmoothL1Loss(beta=a.hb)
    nbr_t = torch.from_numpy(nbr.astype(np.int64)); dt_t = torch.from_numpy(dt); same_t = torch.from_numpy(same.astype(np.float32))
    cnum_t = torch.from_numpy(c_num); ccat_t = torch.from_numpy(c_cat); tgt_t = torch.from_numpy(target.astype(np.float32))

    def batch(ix):
        ix = torch.from_numpy(ix)
        nb = nbr_t[ix].to(dev, non_blocking=True)
        pad = nb < 0; safe = nb.clamp(min=0)
        return (tok_num[safe] * (~pad)[..., None], tok_cat[safe] * (~pad)[..., None], dt_t[ix].to(dev),
                same_t[ix].to(dev), pad, cnum_t[ix].to(dev), ccat_t[ix].to(dev))

    scaler = torch.amp.GradScaler("cuda", enabled=a.amp)
    t0 = time.time(); step = 0
    for ep in range(a.epochs):
        net.train(); perm = rng.permutation(tr); tot = 0.0; n = 0
        for b in range(0, len(perm) - a.batch + 1, a.batch):
            ix = perm[b:b + a.batch]
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
                out = net(*batch(ix))
            loss = lossf(out.float(), tgt_t[ix].to(dev))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            tot += float(loss.detach()) * len(ix); n += len(ix)
            if a.bench and step == 20:
                torch.cuda.synchronize(); tb = time.time()
            if a.bench and step == 20 + a.bench:
                torch.cuda.synchronize()
                print(json.dumps({"bench_steps": a.bench, "sec_per_step": (time.time() - tb) / a.bench,
                                  "steps_per_epoch": len(tr) // a.batch, "loss": tot / n}))
                return
        print(json.dumps({"epoch": ep + 1, "train_loss": tot / n, "sec": round(time.time() - t0)}), flush=True)
    net.eval(); pred = np.full(len(y), np.nan)
    allix = np.flatnonzero(has)
    with torch.no_grad():
        for b in range(0, len(allix), 8192):
            ix = allix[b:b + 8192]
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
                o = net(*batch(ix))
            pred[ix] = o.float().cpu().numpy() * 300.0 + anchor[ix]
    pred = np.maximum(60, pred)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, pred.astype(np.float32))
    print(json.dumps({"done": str(a.out), "train_rows": int(len(tr)), "sec": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
