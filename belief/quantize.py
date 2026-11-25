"""Optional int8: ONNX Runtime dynamic quantization of B.onnx, with the accuracy check.

Keep the int8 model only if the validation one-step cross-entropy worsens by
less than 1% relative AND latency improves (latency: run belief/bench.py on
both files in the arena-like container).

Accuracy is measured with ONNX Runtime on the same validation examples for both
models: one-step CE (all unseen squares masked), an ELBO estimate with shared
random masks, and argmax agreement between the two models.

    .venv/bin/python -m belief.quantize --onnx belief/B.onnx --shards data/shards/arena3k
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from belief.sampler import INPUT_NAMES, MASK, VIS, _log_softmax, make_session

LN2 = math.log(2)


def quantize(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process
    pre = dst.with_suffix('.pre.onnx')
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=False)
    quantize_dynamic(str(pre), str(dst), weight_type=QuantType.QInt8, per_channel=True)
    pre.unlink()


def load_val(shards: Path, n: int, seed: int) -> dict[str, np.ndarray]:
    parts = {}
    for f in sorted((shards / 'val').glob('shard_*.npz')):
        with np.load(f) as z:
            for k in ('cur', 'last', 'age', 'inv', 'ply', 'to_move', 'tgt'):
                parts.setdefault(k, []).append(z[k])
    data = {k: np.concatenate(v) for k, v in parts.items()}
    idx = np.sort(np.random.default_rng(seed).choice(len(data['tgt']), min(n, len(data['tgt'])), replace=False))
    return {k: v[idx] for k, v in data.items()}


def scores(sess, val, masks_t, batch=2048) -> dict:
    """Sum of one-step NLL, ELBO-weighted NLL and hidden counts; argmax predictions."""
    one, elbo, hidden, preds = 0.0, 0.0, 0, []
    rng = np.random.default_rng(123)                  # same masks for every model
    for s in range(0, len(val['tgt']), batch):
        x = {k: val[k][s:s + batch].astype(np.int64) for k in ('cur', 'last', 'age', 'inv', 'ply', 'to_move')}
        tgt = val['tgt'][s:s + batch].astype(np.int64)
        unseen = x['cur'] == 0
        safe = np.maximum(tgt, 0)
        x['xt'] = np.where(unseen, MASK, VIS)
        lp = _log_softmax(sess.run(['logits'], {k: x[k] for k in INPUT_NAMES})[0].astype(np.float64))
        nll = -np.take_along_axis(lp, safe[..., None], -1)[..., 0]
        one += float((nll * unseen).sum())
        hidden += int(unseen.sum())
        preds.append(lp.argmax(-1)[unseen])
        t = masks_t[s:s + batch, None]
        masked = unseen & (rng.random(unseen.shape) < t)
        x['xt'] = np.where(unseen, np.where(masked, MASK, safe), VIS)
        lp = _log_softmax(sess.run(['logits'], {k: x[k] for k in INPUT_NAMES})[0].astype(np.float64))
        nll = -np.take_along_axis(lp, safe[..., None], -1)[..., 0]
        elbo += float(((nll * masked).sum(1) / t[:, 0]).sum())
    return {'onestep_bits': one / hidden / LN2, 'elbo_bits': elbo / hidden / LN2, 'preds': np.concatenate(preds)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--onnx', type=Path, default=Path('belief/B.onnx'))
    ap.add_argument('--out', type=Path, default=None, help='default: <onnx stem>_int8.onnx')
    ap.add_argument('--shards', type=Path, default=Path('data/shards/arena3k'))
    ap.add_argument('--n', type=int, default=50_000, help='validation examples')
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    out = args.out or args.onnx.with_name(args.onnx.stem + '_int8.onnx')
    quantize(args.onnx, out)
    print(f'{args.onnx}: {args.onnx.stat().st_size / 1e6:.2f} MB -> {out}: {out.stat().st_size / 1e6:.2f} MB')

    val = load_val(args.shards, args.n, seed=0)
    t = 1e-3 + (1 - 1e-3) * np.random.default_rng(7).random(len(val['tgt']))
    res = {}
    for name, path in (('fp32', args.onnx), ('int8', out)):
        res[name] = scores(make_session(str(path), args.threads), val, t)
        print(f'{name}: one-step {res[name]["onestep_bits"]:.4f} bits, ELBO {res[name]["elbo_bits"]:.4f} bits')
    rel = res['int8']['onestep_bits'] / res['fp32']['onestep_bits'] - 1
    agree = float((res['int8']['preds'] == res['fp32']['preds']).mean())
    verdict = rel < 0.01
    print(f'one-step CE change: {rel:+.2%} relative (keep only if < +1.00%) -> '
          f'{"accuracy OK" if verdict else "accuracy REJECT"}; argmax agreement {agree:.2%}')
    report = {'fp32': str(args.onnx), 'int8': str(out), 'n_val_examples': len(val['tgt']), 'shards': str(args.shards),
              'fp32_onestep_bits': res['fp32']['onestep_bits'], 'int8_onestep_bits': res['int8']['onestep_bits'],
              'fp32_elbo_bits': res['fp32']['elbo_bits'], 'int8_elbo_bits': res['int8']['elbo_bits'],
              'onestep_rel_change': rel, 'argmax_agreement': agree, 'accuracy_ok': verdict,
              'fp32_bytes': args.onnx.stat().st_size, 'int8_bytes': out.stat().st_size}
    res_dir = Path(__file__).resolve().parent / 'results'
    res_dir.mkdir(exist_ok=True)
    (res_dir / 'int8_accuracy.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
