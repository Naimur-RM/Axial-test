"""Pretraining on freshly simulated worlds.

Each step draws a batch of worlds (one shared row count, any mix of kinds) and
fits three heads: pair membership, pair role, and boundary size. The lattice
family is held out. Checkpoints are written atomically so a run on a cloud
notebook can resume after a disconnect. ``best.pt`` is chosen on the synthetic
validation slices only.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .batch import collate, world_item
from .config import make_config
from .model import build_model
from .prior import FAMILIES, sample_world


class WorldStream(torch.utils.data.IterableDataset):
    """Endless batches. Each worker gets its own seed stream."""

    def __init__(self, cfg, families, seed):
        self.cfg = cfg
        self.families = families
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = 0 if info is None else info.id
        rng = np.random.default_rng([self.seed, wid])
        cfg = self.cfg
        while True:
            n = int(rng.integers(cfg["N_min"], cfg["N_max"] + 1))
            worlds = [sample_world(rng, cfg, families=self.families, n=n) for _ in range(cfg["batch"])]
            items = [world_item(w, rng) for w in worlds]
            labels = [{k: w[k] for k in ("member", "role", "pair_mask", "count", "count_mask")} for w in worlds]
            yield items, labels


def _identity(x):
    return x


def compute_loss(out, target, cfg):
    valid = out["valid"] & target["pair_mask"]
    pos_weight = torch.tensor(cfg["pos_weight"], device=out["member"].device)
    bce = nn.functional.binary_cross_entropy_with_logits(
        out["member"], target["member"], pos_weight=pos_weight, reduction="none",
    )
    vf = valid.float()
    l_pair = (bce * vf).sum() / vf.sum().clamp(min=1.0)
    ce_role = nn.functional.cross_entropy(
        out["role"].reshape(-1, 3), target["role"].reshape(-1), reduction="none",
    ).view_as(vf)
    l_role = (ce_role * vf).sum() / vf.sum().clamp(min=1.0)
    cm = target["count_mask"] & out["valid"].any(-1)
    logits = out["count"][cm]
    counts = target["count"][cm]
    l_count = nn.functional.cross_entropy(logits, counts) if cm.any() else out["count"].sum() * 0
    prob = torch.softmax(out["count"], -1)
    ks = torch.arange(prob.shape[-1], device=prob.device, dtype=prob.dtype)
    expected = (prob.nan_to_num(0.0) * ks).sum(-1)
    soft = (torch.sigmoid(out["member"]) * vf).sum(-1)
    l_cons = ((expected - soft).abs() * cm.float()).sum() / cm.float().sum().clamp(min=1.0)
    total = (
        l_pair
        + cfg["role_weight"] * l_role
        + cfg["count_weight"] * l_count
        + cfg["consistency_weight"] * l_cons
    )
    parts = dict(pair=l_pair.item(), role=l_role.item(), count=l_count.item(), cons=l_cons.item())
    return total, parts


def _amp(device):
    if not str(device).startswith("cuda"):
        return contextlib.nullcontext(), None
    if torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16), None
    return torch.autocast("cuda", dtype=torch.float16), torch.amp.GradScaler("cuda")


def _schedule(cfg):
    warm = max(1, cfg["warmup"])
    total = cfg["steps"]

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        frac = (step - warm) / max(1, total - warm)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))

    return lr_lambda


def _atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(path, model, cfg, extra=None):
    payload = {"model": model.state_dict(), "cfg": cfg, "version": 2}
    if extra:
        payload.update(extra)
    _atomic_save(payload, path)
    return Path(path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("version") != 2:
        raise ValueError(f"{path} is not an AxialMB v2 checkpoint")
    cfg = ckpt["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def train(cfg, out_dir, device="cpu", seed=7, max_minutes=None):
    from .validation import quick_validation

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    torch.manual_seed(seed)
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _schedule(cfg))
    autocast, scaler = _amp(device)
    hist, start, best = [], 1, -1.0
    if last_path.is_file():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        hist, start, best = ck.get("hist", []), ck["step"] + 1, ck.get("best", -1.0)
        print(f"resumed from {last_path} at step {start}", flush=True)

    families = [f for f in FAMILIES if f != cfg["holdout_family"]]
    stream = WorldStream(cfg, families, seed=[seed, start])
    loader = torch.utils.data.DataLoader(
        stream, batch_size=None, num_workers=cfg.get("workers", 0), collate_fn=_identity,
        persistent_workers=cfg.get("workers", 0) > 0,
    )
    batches = iter(loader)
    accum = max(1, cfg.get("accum", 1))
    t0 = time.time()
    model.train()
    step = start - 1
    for step in range(start, cfg["steps"] + 1):
        opt.zero_grad(set_to_none=True)
        parts_sum = {}
        loss_sum = 0.0
        for _ in range(accum):
            items, labels = next(batches)
            batch, target = collate(items, device, labels=labels)
            with autocast:
                out = model(**batch)
            loss, parts = compute_loss(out, target, cfg)
            if scaler is not None:
                scaler.scale(loss / accum).backward()
            else:
                (loss / accum).backward()
            loss_sum += loss.item() / accum
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v / accum
        if scaler is not None:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        sched.step()

        if step % cfg["log_every"] == 0 or step == start:
            elapsed = time.time() - t0
            rate = elapsed / max(1, step - start + 1)
            peak = torch.cuda.max_memory_allocated() / 1e9 if str(device).startswith("cuda") else 0.0
            row = dict(step=step, loss=loss_sum, **parts_sum, secs=elapsed, peak_gb=peak,
                       lr=sched.get_last_lr()[0])
            hist.append(row)
            print(
                f"step {step:6d}/{cfg['steps']} | loss {loss_sum:.3f} | pair {parts_sum['pair']:.3f} "
                f"| count {parts_sum['count']:.3f} | role {parts_sum['role']:.3f} | {rate:.2f}s/step "
                f"| ETA {(cfg['steps'] - step) * rate / 60:.0f}m | peak {peak:.1f}GB",
                flush=True,
            )
        stop = max_minutes is not None and (time.time() - t0) / 60 >= max_minutes
        if step % cfg["val_every"] == 0 or step == cfg["steps"] or stop:
            model.eval()
            val = quick_validation(model, cfg, device)
            model.train()
            hist.append(dict(step=step, val=val))
            print(f"  validation at step {step}: " + json.dumps(val), flush=True)
            if val["mean_f1"] > best:
                best = val["mean_f1"]
                save_checkpoint(best_path, model, cfg, {"step": step, "val": val})
                print(f"  new best {best:.3f} -> {best_path}", flush=True)
        if step % cfg["ckpt_every"] == 0 or step == cfg["steps"] or stop:
            _atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                              step=step, hist=hist, best=best, cfg=cfg), last_path)
        if stop:
            print(f"stopped at step {step} after {max_minutes} min", flush=True)
            break
    summary = dict(tier=cfg["tier"], steps_done=step, best_val_f1=best, device=str(device),
                   best=str(best_path), last=str(last_path))
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pretrain AxialMB v2")
    parser.add_argument("--tier", default="smoke", choices=["smoke", "A", "C"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="")
    parser.add_argument("--out", default="output/axialmb_v2/checkpoints")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--accum", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    overrides = {k: v for k, v in dict(
        steps=args.steps, lr=args.lr, batch=args.batch, accum=args.accum, workers=args.workers,
    ).items() if v is not None}
    cfg = make_config(args.tier, **overrides)
    print(f"tier {cfg['tier']} | device {device} | d={cfg['d']} blocks={cfg['blocks']} "
          f"| p {cfg['p_min']}-{cfg['p_max']} | N {cfg['N_min']}-{cfg['N_max']} | held out {cfg['holdout_family']}",
          flush=True)
    return train(cfg, args.out, device=device, seed=args.seed, max_minutes=args.max_minutes)


if __name__ == "__main__":
    main()
