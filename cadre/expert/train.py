"""Train the expert model on the pool (multi-label binary cross-entropy over the 150 statements) and export its logits
on the memory and test records.

Training: AdamW (learning rate 2e-3, weight decay 0.01), linear warm-up and cosine decay, batch 256 per GPU, 8 epochs,
bf16 autocast, gradient clipping at 1.0. 2% of the pool patients (CRC32 of the patient id modulo 50 equal to 0) are
held out for model selection by macro AUROC.

  torchrun --standalone --nproc_per_node 4 scripts/03_train_expert.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
import zlib

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from cadre import config
from cadre.expert.model import ResNet1D


def index(name):
    with open(config.work("expert", f"{name}_index.csv")) as handle:
        return list(csv.DictReader(handle))


class Signals(torch.utils.data.Dataset):
    def __init__(self, name, rows, labels=None):
        n = len(index(name))
        self.x = np.memmap(config.work("expert", f"{name}_signals.int16"), dtype=np.int16, mode="r", shape=(n, 12, 5000))
        self.rows, self.y = rows, labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        i = self.rows[k]
        x = torch.from_numpy(np.asarray(self.x[i], dtype=np.float32) / 200.0)
        y = torch.from_numpy(self.y[i].astype(np.float32)) if self.y is not None else torch.zeros(0)
        return x, y


def predict(model, dataset, device, batch=512):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=8)
    out = []
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in loader:
            out.append(torch.sigmoid(model(x.to(device, non_blocking=True)).float()).cpu().numpy())
    return np.concatenate(out)


def macro_auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(np.mean([roc_auc_score(y[:, k], p[:, k]) for k in range(y.shape[1]) if 0 < y[:, k].sum() < len(y)]))


def main(epochs=8, batch=256, lr=2e-3):
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.manual_seed(27)
    pool = index("pool")
    Y = np.load(config.work("expert", "pool_labels.npy"))
    vocab = json.loads(config.work("expert", "vocab.json").read_text())
    held = lambda s: zlib.crc32(s.encode()) % 50 == 0  # noqa: E731
    train_rows = [int(r["row"]) for r in pool if r["ok"] == "1" and not held(r["subject_id"])]
    val_rows = [int(r["row"]) for r in pool if r["ok"] == "1" and held(r["subject_id"])]
    train_set, val_set = Signals("pool", train_rows, Y), Signals("pool", val_rows, Y)
    sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True, seed=27)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch, sampler=sampler, num_workers=12, pin_memory=True,
                                         persistent_workers=True, drop_last=True)
    model = nn.parallel.DistributedDataParallel(ResNet1D(len(vocab)).to(device), device_ids=[local])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * len(loader)
    warm = min(1000, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    best, ckpt = -1.0, config.work("expert", "best.pt")
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        tick, total, n = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.binary_cross_entropy_with_logits(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total, n = total + float(loss), n + 1
        if rank == 0:
            auc = macro_auc(Y[val_rows], predict(model.module, val_set, device))
            print(json.dumps({"epoch": epoch, "train_loss": total / n, "val_macro_auroc": auc,
                              "minutes": (time.time() - tick) / 60}), flush=True)
            if auc > best:
                best = auc
                torch.save(model.module.state_dict(), ckpt)
        dist.barrier()
    if rank == 0:
        model.module.load_state_dict(torch.load(ckpt, map_location=device))
        for name in ("memory", "test"):
            rows = index(name)
            p = np.clip(predict(model.module, Signals(name, [int(r["row"]) for r in rows]), device).astype(np.float64),
                        1e-6, 1 - 1e-6)
            np.save(config.work("expert", f"logits_{name}.npy"), np.log(p / (1 - p)).astype(np.float32))
            config.work("expert", f"ids_{name}.json").write_text(json.dumps([r["study_id"] for r in rows]))
        print(json.dumps({"best_val_macro_auroc": best}), flush=True)
    dist.barrier()
    dist.destroy_process_group()
