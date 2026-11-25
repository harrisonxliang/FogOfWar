#!/usr/bin/env python3
"""Train the masked-diffusion belief model.

Works on CPU for smoke tests; pass --device cuda on Colab for the real run.
Validation reports the full-masking (t=1) metrics — hidden-square log-loss and
accuracy — which is exactly the "model v1" quality measure, plus a
predict-most-common baseline for reference.

Usage:
    .venv/Scripts/python boardgen/train.py --data data/dataset.npz --epochs 10 --out checkpoints/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoding import UNKNOWN, build_inputs
from model import BeliefNet, parameter_count


def load_split(path, val_fraction_games=10):
    data = np.load(path)
    arrays = {key: data[key] for key in data.files}
    val = arrays['game'] % val_fraction_games == 0
    train_idx, val_idx = np.where(~val)[0], np.where(val)[0]
    print(f'{len(train_idx)} train / {len(val_idx)} val examples '
          f'(split by game, {arrays["game"].max() + 1} games)')
    return arrays, train_idx, val_idx


def make_batch(arrays, indices, rng, t=None):
    """Build (planes, target, loss_mask) for a batch, applying random masking."""
    obs = arrays['obs'][indices]
    target = arrays['target'][indices]
    n = len(indices)
    hidden = obs == UNKNOWN
    if t is None:
        t = rng.uniform(0.05, 1.0, size=(n, 1))
    else:
        t = np.full((n, 1), t)
    masked = hidden & (rng.uniform(size=(n, 64)) < t)
    # Canvas: true contents everywhere except masked hidden squares. On visible
    # squares this equals the observation; on kept hidden squares it reveals truth.
    canvas = target.copy()
    canvas[masked] = UNKNOWN
    planes = build_inputs(obs, arrays['last_seen'][indices], arrays['since'][indices],
                          arrays['ply'][indices], arrays['to_move'][indices], canvas)
    return (torch.from_numpy(planes), torch.from_numpy(target.astype(np.int64)),
            torch.from_numpy(masked))


def evaluate(model, arrays, val_idx, device, batch_size=512):
    """Full-masking (t=1) hidden-square log-loss and accuracy."""
    rng = np.random.default_rng(0)
    model.eval()
    total_loss, total_correct, total_squares = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.inference_mode():
        for start in range(0, len(val_idx), batch_size):
            batch_idx = val_idx[start:start + batch_size]
            planes, target, masked = make_batch(arrays, batch_idx, rng, t=1.0)
            planes, target, masked = planes.to(device), target.to(device), masked.to(device)
            logits = model(planes)
            total_loss += criterion(logits.permute(0, 2, 1)[masked], target[masked]).item()
            total_correct += (logits.argmax(dim=1)[masked] == target[masked]).sum().item()
            total_squares += masked.sum().item()
    return total_loss / total_squares, total_correct / total_squares


def baseline_accuracy(arrays, val_idx):
    """Accuracy of predicting the most common class (empty) on hidden squares."""
    obs = arrays['obs'][val_idx]
    target = arrays['target'][val_idx]
    hidden = obs == UNKNOWN
    values, counts = np.unique(target[hidden], return_counts=True)
    best = values[counts.argmax()]
    return (target[hidden] == best).mean(), int(best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--channels', type=int, default=96)
    parser.add_argument('--blocks', type=int, default=6)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', type=Path, default=Path('checkpoints'))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    arrays, train_idx, val_idx = load_split(args.data)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    model = BeliefNet(args.channels, args.blocks).to(args.device)
    print(f'{parameter_count(model):,} parameters, device {args.device}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, len(train_idx) // args.batch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch)
    criterion = nn.CrossEntropyLoss()

    base_acc, base_class = baseline_accuracy(arrays, val_idx)
    print(f'baseline (always predict class {base_class}): {base_acc:.3f} accuracy')
    args.out.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(train_idx)
        epoch_loss, batches = 0.0, 0
        started = time.monotonic()
        for start in range(0, len(order) - args.batch + 1, args.batch):
            planes, target, masked = make_batch(arrays, order[start:start + args.batch], rng)
            planes, target, masked = (planes.to(args.device), target.to(args.device),
                                      masked.to(args.device))
            logits = model(planes).permute(0, 2, 1)  # (N, 64, 13)
            loss = criterion(logits[masked], target[masked])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            batches += 1
        val_loss, val_acc = evaluate(model, arrays, val_idx, args.device)
        print(f'epoch {epoch + 1}/{args.epochs}: train loss {epoch_loss / batches:.4f}, '
              f'val t=1 loss {val_loss:.4f}, val acc {val_acc:.3f} '
              f'({time.monotonic() - started:.0f}s)', flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'state_dict': model.state_dict(),
                        'config': {'channels': args.channels, 'blocks': args.blocks}},
                       args.out / 'belief.pt')
    (args.out / 'belief.json').write_text(json.dumps({
        'parameters': parameter_count(model), 'channels': args.channels,
        'blocks': args.blocks, 'val_loss_t1': best_loss, 'val_acc_t1': val_acc,
        'baseline_acc': float(base_acc)}) + '\n')
    print(f'best val loss {best_loss:.4f}; saved to {args.out / "belief.pt"}')


if __name__ == '__main__':
    main()
