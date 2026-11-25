"""Shared encodings for the belief model (dataset building, training, inference).

Class indices for a square's content:
    0..11  pieces in PNBRQKpnbrqk order (uppercase = observer's side after
           canonicalization; boards are flipped so the observer always plays
           "white, moving up")
    12     empty
    13     unknown ('?') in observations / never-seen in last_seen / MASK in
           the diffusion canvas

Torch-free so the arena bot can reuse it with ONNX Runtime + numpy only.
"""
from __future__ import annotations

import numpy as np

PIECES = 'PNBRQKpnbrqk'
EMPTY = 12
UNKNOWN = 13  # '?', never-seen, and the diffusion MASK token
NUM_CLASSES = 13  # targets: 12 pieces + empty (UNKNOWN is never a target)
PLANES = 45  # obs 14 + last_seen 14 + since 1 + ply 1 + to_move 1 + canvas 14

SYMBOL_TO_CLASS = {symbol: index for index, symbol in enumerate(PIECES)}
SYMBOL_TO_CLASS['-'] = EMPTY
SYMBOL_TO_CLASS['?'] = UNKNOWN
SWAP_CASE = np.array([SYMBOL_TO_CLASS[symbol.swapcase()] for symbol in PIECES]
                     + [EMPTY, UNKNOWN], dtype=np.uint8)


def board_string_to_classes(text):
    """64-char board/observation string (a1..h8) -> (64,) uint8 class array."""
    return np.array([SYMBOL_TO_CLASS[symbol] for symbol in text], dtype=np.uint8)


def canonicalize(classes, is_white):
    """Flip a (…, 64) class array so the observer plays white moving up."""
    if is_white:
        return classes
    # Mirror ranks (square -> square ^ 56) and swap piece colors.
    mirrored = classes.reshape(*classes.shape[:-1], 8, 8)[..., ::-1, :].reshape(*classes.shape[:-1], 64)
    return SWAP_CASE[mirrored]


def classes_to_planes(classes, num_channels=14):
    """(N, 64) class array -> (N, C, 8, 8) one-hot planes."""
    n = classes.shape[0]
    planes = np.zeros((n, num_channels, 64), dtype=np.float32)
    rows = np.arange(n)[:, None]
    planes[rows, classes, np.arange(64)[None, :]] = 1.0
    return planes.reshape(n, num_channels, 8, 8)


def build_inputs(obs, last_seen, since, ply, to_move, canvas):
    """Assemble the (N, 45, 8, 8) model input from compact per-square arrays.

    obs, last_seen, canvas: (N, 64) uint8 class arrays (canonicalized).
    since: (N, 64) plies since each square was last seen (255 = never).
    ply: (N,) move count; to_move: (N,) 1.0 if the observer is to move.
    """
    n = obs.shape[0]
    parts = [
        classes_to_planes(obs),
        classes_to_planes(last_seen),
        (np.minimum(since, 50) / 50.0).astype(np.float32).reshape(n, 1, 8, 8),
        np.broadcast_to((np.minimum(ply, 200) / 200.0).astype(np.float32).reshape(n, 1, 1, 1), (n, 1, 8, 8)).copy(),
        np.broadcast_to(np.asarray(to_move, dtype=np.float32).reshape(n, 1, 1, 1), (n, 1, 8, 8)).copy(),
        classes_to_planes(canvas),
    ]
    return np.concatenate(parts, axis=1)


class HistoryTracker:
    """Maintains last-seen piece and plies-since-seen from a stream of observations.

    Feed raw (uncanonicalized) 64-char observation strings in ply order for one
    observer; canonicalization happens in snapshot(). The arena bot runs the
    same class against live stdin observations.
    """

    def __init__(self, is_white):
        self.is_white = is_white
        self.last_seen = np.full(64, UNKNOWN, dtype=np.uint8)
        self.since = np.full(64, 255, dtype=np.uint8)

    def update(self, observation_text):
        obs = board_string_to_classes(observation_text)
        seen = obs != UNKNOWN
        self.last_seen[seen] = obs[seen]
        self.since[~seen] = np.minimum(self.since[~seen].astype(np.int32) + 1, 255).astype(np.uint8)
        self.since[seen] = 0
        return obs

    def snapshot(self, obs):
        """Canonicalized (obs, last_seen, since) for the current position."""
        flip = lambda a: canonicalize(a[None], self.is_white)[0]
        since = self.since if self.is_white else canonicalize_since(self.since)
        return flip(obs), flip(self.last_seen), since


def canonicalize_since(since):
    """Mirror the (64,) since array for a black observer (no piece relabeling)."""
    return since.reshape(8, 8)[::-1, :].reshape(64).copy()
