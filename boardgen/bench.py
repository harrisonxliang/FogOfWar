#!/usr/bin/env python3
"""Export the belief model to ONNX (with numerical check, like the neural
starter) and benchmark against the arena budget on 2 CPU threads:
score 2,000 boards or generate 256 boards in < 0.5 s.

Latency does not depend on weight values, so this is valid with any checkpoint.

Usage:
    .venv/Scripts/python boardgen/bench.py --checkpoint checkpoints/smoke/belief.pt --out checkpoints/smoke/belief.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoding import PLANES, UNKNOWN
from model import BeliefNet
from sample import BeliefState, OnnxPredictor, classes_to_fen_board, generate, score

INITIAL_WHITE_OBS = 'RNBQKBNRPPPPPPPP----------------????????????????????????????????'


def export(checkpoint_path, onnx_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = BeliefNet(**checkpoint['config']).eval()
    model.load_state_dict(checkpoint['state_dict'])
    torch.onnx.export(model, (torch.zeros(1, PLANES, 8, 8),), str(onnx_path),
                      input_names=['planes'], output_names=['logits'],
                      opset_version=17, dynamo=False,
                      dynamic_axes={'planes': {0: 'batch'}, 'logits': {0: 'batch'}})
    predictor = OnnxPredictor(onnx_path)
    for batch in (1, 64):
        example = torch.randn(batch, PLANES, 8, 8)
        with torch.inference_mode():
            expected = model(example).numpy()
        np.testing.assert_allclose(expected, predictor(example.numpy()), rtol=1e-3, atol=1e-4)
    print(f'exported {onnx_path} and verified against torch (batch 1 and 64)')
    return predictor


def bench(predictor):
    state = BeliefState(True)
    state.update(INITIAL_WHITE_OBS, 0, True)
    rng = np.random.default_rng(0)

    # score(): 2,000 random-ish candidate boards.
    hidden = np.where(state.obs == UNKNOWN)[0]
    fens = []
    for _ in range(2000):
        board = state.obs.copy()
        board[hidden] = rng.integers(0, 13, size=len(hidden))
        fens.append(classes_to_fen_board(board, True))
    for _ in range(2):  # warm-up then measure
        started = time.perf_counter()
        weights = score(predictor, state, fens)
        score_seconds = time.perf_counter() - started
    assert len(weights) == 2000
    print(f'score 2000 boards: {score_seconds * 1000:.0f} ms '
          f'({"PASS" if score_seconds < 0.5 else "FAIL"} vs 500 ms budget)')

    for steps in (4, 8):
        for _ in range(2):
            started = time.perf_counter()
            boards, _ = generate(predictor, state, n=256, steps=steps, rng=rng)
            generate_seconds = time.perf_counter() - started
        assert len(boards) == 256
        print(f'generate 256 boards, {steps} steps: {generate_seconds * 1000:.0f} ms '
              f'({"PASS" if generate_seconds < 0.5 else "FAIL"} vs 500 ms budget)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    predictor = export(args.checkpoint, args.out)
    bench(predictor)


if __name__ == '__main__':
    main()
