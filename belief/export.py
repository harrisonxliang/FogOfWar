"""Export a trained checkpoint to ONNX and check parity with PyTorch.

Inputs: int64 cur, last, age, xt [B,64]; inv [B,7]; ply [B]; to_move [B]
(dynamic batch). Output: logits [B,64,7] float32. Writes the .onnx file and a
JSON sidecar (model config, vocab sizes, training commit, run metrics).

Parity: (1) max |logit diff| between PyTorch and ONNX Runtime on 1,000 random
validation inputs (random diffusion times) must be < 1e-4; (2) a full
sample() with the same seed through both backends must give the same boards.

    .venv/bin/python -m belief.export --run runs/arena3k_s0 --out belief/B.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from belief.dataset import load_shards
from belief.eval import load_checkpoint
from belief.features import view_from_arrays
from belief.model import MASK, N_AGE, N_CUR, N_LAST, N_OUT, N_XT, VIS
from belief.sampler import INPUT_NAMES, DiffusionBelief, SamplerConfig, make_session

OPSET = 17
MAX_IR_VERSION = 8      # IR 8 covers opset 17 and loads on older ONNX Runtime builds (arena: 1.24)


def export(model, path: Path) -> None:
    model = model.cpu().eval()
    B = 3
    args = (torch.zeros(B, 64, dtype=torch.long), torch.zeros(B, 64, dtype=torch.long),
            torch.full((B, 64), 5, dtype=torch.long), torch.full((B, 64), MASK, dtype=torch.long),
            torch.ones(B, 7, dtype=torch.long), torch.zeros(B, dtype=torch.long), torch.ones(B, dtype=torch.long))
    dyn = {name: {0: 'B'} for name in INPUT_NAMES}
    dyn['logits'] = {0: 'B'}
    torch.onnx.export(model, args, str(path), input_names=list(INPUT_NAMES), output_names=['logits'],
                      dynamic_axes=dyn, opset_version=OPSET, dynamo=False, do_constant_folding=True)
    import onnx
    m = onnx.load(str(path))
    m.ir_version = min(m.ir_version, MAX_IR_VERSION)
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def random_inputs(val: dict, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    idx = rng.choice(len(val['t']), size=n, replace=False)
    x = {k: val[k][idx].astype(np.int64) for k in ('cur', 'last', 'age', 'inv', 'ply', 'to_move')}
    tgt = val['tgt'][idx].astype(np.int64)
    unseen = x['cur'] == 0
    t = rng.uniform(1e-3, 1, size=(n, 1))
    masked = unseen & (rng.random((n, 64)) < t)
    x['xt'] = np.where(unseen, np.where(masked, MASK, np.maximum(tgt, 0)), VIS)
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--ckpt', default='best.pt')
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'B.onnx')
    ap.add_argument('--data', type=Path, default=None)
    ap.add_argument('--n-parity', type=int, default=1000)
    args = ap.parse_args()

    model, ckpt = load_checkpoint(args.run / args.ckpt, 'cpu')
    export(model, args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f'wrote {args.out} ({size_mb:.2f} MB, opset {OPSET})')

    data_dir = args.data or Path(ckpt['train_config']['data'])
    val = load_shards(data_dir, 'val', 200_000)
    rng = np.random.default_rng(0)
    x = random_inputs(val, args.n_parity, rng)
    sess = make_session(str(args.out), threads=2)
    with torch.no_grad():
        ref = np.concatenate([model(*[torch.from_numpy(x[k][s:s + 250]) for k in INPUT_NAMES]).numpy()
                              for s in range(0, args.n_parity, 250)])
    got = np.concatenate([sess.run(['logits'], {k: x[k][s:s + 250] for k in INPUT_NAMES})[0]
                          for s in range(0, args.n_parity, 250)])
    max_diff = float(np.abs(ref - got).max())
    logit_ok = max_diff < 1e-4
    print(f'logit parity on {args.n_parity} val inputs: max |diff| = {max_diff:.2e} -> {"PASS" if logit_ok else "FAIL"}')

    # full sample() through both backends, same seed
    def torch_forward(inputs):
        with torch.no_grad():
            return model(*[torch.from_numpy(inputs[k]) for k in INPUT_NAMES]).numpy()
    idx = rng.choice(len(val['t']), size=16, replace=False)
    views = [view_from_arrays(val['cur'][i], val['last'][i], val['age'][i], val['inv'][i], val['ply'][i],
                              val['to_move'][i], val['viewer'][i] == 0, val['t'][i]) for i in idx]
    b_ort = DiffusionBelief(str(args.out), threads=2).sample(views, 4, 4, np.random.default_rng(7))
    b_pt = DiffusionBelief(forward=torch_forward).sample(views, 4, 4, np.random.default_rng(7))
    same = sum(a == b for va, vb in zip(b_ort, b_pt) for a, b in zip(va, vb))
    total = sum(len(v) for v in b_ort)
    sample_ok = same == total
    print(f'sample() parity: {same}/{total} boards identical -> {"PASS" if sample_ok else "FAIL"}')

    sidecar = {
        'onnx': args.out.name, 'opset': OPSET, 'size_bytes': args.out.stat().st_size,
        'inputs': {k: ['B', 64] for k in ('cur', 'last', 'age', 'xt')} | {'inv': ['B', 7], 'ply': ['B'], 'to_move': ['B']},
        'output': {'logits': ['B', 64, N_OUT]},
        'vocab': {'cur': N_CUR, 'last': N_LAST, 'age': N_AGE, 'xt': N_XT, 'out': N_OUT, 'MASK': MASK, 'VIS': VIS},
        'model_config': ckpt['model_config'], 'train_config': ckpt['train_config'], 'step': ckpt['step'],
        'training_git': ckpt.get('git'), 'val': ckpt.get('val'), 'run': str(args.run),
        'parity': {'max_abs_logit_diff': max_diff, 'n': args.n_parity, 'logits_pass': logit_ok,
                   'sample_boards_identical': f'{same}/{total}', 'sample_pass': sample_ok},
        'sampler_defaults': SamplerConfig().__dict__,
    }
    args.out.with_suffix('.json').write_text(json.dumps(sidecar, indent=1))
    print(f'wrote {args.out.with_suffix(".json")}')
    sys.exit(0 if logit_ok and sample_ok else 1)


if __name__ == '__main__':
    main()
