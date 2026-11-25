"""Inference for the belief model: score() for B's weight function, generate()
for board generation via iterative unmasking (MaskGIT-style).

Both consume a BeliefState built from the observer's raw observation stream,
so the arena bot and offline analysis share one code path. Works from a torch
checkpoint or an exported ONNX session (see bench.py).

score():  ONE forward pass regardless of how many boards are scored — all
          candidates share the same conditioning, so scoring 2,000 FENs is a
          numpy gather over the per-square log-probs.
generate(): batch of n boards refined over --steps forward passes; pass the
          survivors through Misty's BeliefHardFacts before use (filter_fn hook).
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoding import (EMPTY, NUM_CLASSES, PIECES, UNKNOWN, HistoryTracker,
                      build_inputs, canonicalize)


class BeliefState:
    """Observation history for one observer, ready to condition the model."""

    def __init__(self, is_white):
        self.is_white = is_white
        self.tracker = HistoryTracker(is_white)
        self.obs = None
        self.last_seen = None
        self.since = None
        self.ply = 0

    def update(self, observation_text, ply, to_move=True):
        raw = self.tracker.update(observation_text)
        self.obs, self.last_seen, self.since = self.tracker.snapshot(raw)
        self.ply = ply
        self.to_move = to_move

    def conditioning(self, n, canvas):
        """(n, 45, 8, 8) input planes for n copies of this state with given canvases."""
        rep = lambda a: np.repeat(a[None], n, axis=0)
        return build_inputs(rep(self.obs), rep(self.last_seen), rep(self.since),
                            np.full(n, self.ply), np.full(n, float(self.to_move)), canvas)


class TorchPredictor:
    def __init__(self, checkpoint_path, device='cpu'):
        import torch
        from model import BeliefNet
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model = BeliefNet(**checkpoint['config']).to(device).eval()
        self.model.load_state_dict(checkpoint['state_dict'])
        self.device = device
        self.torch = torch

    def __call__(self, planes):
        with self.torch.inference_mode():
            logits = self.model(self.torch.from_numpy(planes).to(self.device))
        return logits.cpu().numpy()


class OnnxPredictor:
    def __init__(self, onnx_path, threads=2):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx_path), options,
                                            providers=['CPUExecutionProvider'])

    def __call__(self, planes):
        return self.session.run(None, {'planes': planes})[0]


def log_softmax(logits, axis):
    shifted = logits - logits.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def fen_board_to_classes(fen_board, is_white):
    """FEN board field -> canonicalized (64,) class array (a1..h8 before flip)."""
    rows = fen_board.split('/')
    squares = np.empty(64, dtype=np.uint8)
    for rank_from_top, row in enumerate(rows):
        rank = 7 - rank_from_top
        file = 0
        for symbol in row:
            if symbol.isdigit():
                for _ in range(int(symbol)):
                    squares[rank * 8 + file] = EMPTY
                    file += 1
            else:
                squares[rank * 8 + file] = PIECES.index(symbol)
                file += 1
    return canonicalize(squares[None], is_white)[0]


def classes_to_fen_board(classes, is_white):
    """Canonicalized (64,) class array -> FEN board field in true orientation."""
    squares = canonicalize(classes[None], is_white)[0]  # involution: undo the flip
    rows = []
    for rank in range(7, -1, -1):
        row, run = '', 0
        for file in range(8):
            piece = squares[rank * 8 + file]
            if piece == EMPTY:
                run += 1
            else:
                row += (str(run) if run else '') + PIECES[piece]
                run = 0
        rows.append(row + (str(run) if run else ''))
    return '/'.join(rows)


def hidden_log_probs(predictor, state):
    """One fully-masked pass -> (64, 13) per-square log-probs (v1 distribution)."""
    canvas = state.obs.copy()  # hidden squares are already UNKNOWN == MASK
    logits = predictor(state.conditioning(1, canvas[None]))[0]  # (13, 64)
    return log_softmax(logits, axis=0).T  # (64, 13)


def score(predictor, state, fen_boards):
    """B's weight function: log p(hidden squares of each board | observation).

    fen_boards: list of FEN board fields (first FEN token). Returns (N,) float64
    log-weights. One forward pass total.
    """
    per_square = hidden_log_probs(predictor, state)
    hidden = np.where(state.obs == UNKNOWN)[0]
    boards = np.stack([fen_board_to_classes(fen, state.is_white) for fen in fen_boards])
    return per_square[hidden, boards[:, hidden]].sum(axis=1)


def generate(predictor, state, n=256, steps=8, temperature=1.0, rng=None,
             filter_fn=None):
    """Sample n boards by iterative unmasking; returns FEN board fields.

    filter_fn(fen_board) -> bool applies Misty's BeliefHardFacts; survivors only.
    """
    rng = rng or np.random.default_rng()
    hidden = np.where(state.obs == UNKNOWN)[0]
    canvas = np.repeat(state.obs[None], n, axis=0)  # hidden squares start MASKed
    for step in range(steps):
        masked = canvas == UNKNOWN
        if not masked.any():
            break
        if step == 0:
            # Every canvas is identical on the first step (all hidden squares
            # masked): one forward pass, broadcast, sample n independent draws.
            logits = np.repeat(predictor(state.conditioning(1, canvas[:1])), n, axis=0)
        else:
            logits = predictor(state.conditioning(n, canvas))  # (n, 13, 64)
        log_probs = log_softmax(logits / max(temperature, 1e-6), axis=1)
        probs = np.exp(log_probs)
        # Sample a candidate piece for every still-masked square.
        cumulative = probs.cumsum(axis=1)
        draws = rng.uniform(size=(n, 1, 64))
        sampled = (draws < cumulative).argmax(axis=1)  # (n, 64)
        confidence = np.take_along_axis(probs, sampled[:, None, :], axis=1)[:, 0, :]
        confidence = np.where(masked, confidence, -np.inf)
        # Cosine schedule: commit the most confident squares, re-mask the rest.
        remaining_after = int(len(hidden) * np.cos((step + 1) / steps * np.pi / 2))
        for i in range(n):
            squares = np.where(masked[i])[0]
            order = squares[np.argsort(-confidence[i, squares])]
            if step == steps - 1 or remaining_after <= 0:
                chosen = order  # final step: commit everything
            else:
                chosen = order[:max(1, len(squares) - remaining_after)]
            canvas[i, chosen] = sampled[i, chosen]
    boards = [classes_to_fen_board(canvas[i], state.is_white) for i in range(n)]
    if filter_fn is not None:
        survivors = [fen for fen in boards if filter_fn(fen)]
        return survivors, len(survivors) / len(boards)
    return boards, 1.0
