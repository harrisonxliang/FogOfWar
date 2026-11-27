"""Viewer-side features for the belief model. Play-time code: stdlib + NumPy only.

Golden rule: everything here is computed from the viewer's own observations and
own moves, never from the true board.

Canonical orientation: for a Black viewer, square s maps to s ^ 56 (ranks
mirrored) and colors are swapped, so the viewer's pieces are always "own"
(uppercase in canonical strings) and start on ranks 1-2. Bishop square color
is parity in canonical coordinates: LIGHT[sq] = (file + rank) % 2 == 1.

Per-square codes (uint8, canonical square order a1..h8):
  cur   14: 0 unseen, 1 visible empty, 2-7 own P N B R Q K, 8-13 visible enemy P N B R Q K
  last   8: 0 never seen, 1 seen with no enemy piece, 2-7 enemy P N B R Q K
  age    6: 0 visible now, 1, 2, 3 (3-5 plies), 4 (6+), 5 never seen
Globals:
  inv    7: enemy P, N, light B, dark B, R, Q, K = enemy start set minus the
            enemy pieces the viewer has captured (unseen promotions not reflected)
  ply    min(t // 8, 31) where t = frame index (plies played so far)
  to_move 1 if the viewer moves next
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PIECES = 'PNBRQK'                      # piece type k (1..6) is PIECES[k-1]
N_CUR, N_LAST, N_AGE, N_INV, N_PLY = 14, 8, 6, 7, 32
UNSEEN, EMPTY = 0, 1
OWN0, ENEMY0 = 2, 8                    # cur code of own pawn / enemy pawn
INV_NAMES = ('P', 'N', 'B_light', 'B_dark', 'R', 'Q', 'K')
INV_START = np.array([8, 2, 1, 1, 2, 1, 1], dtype=np.int16)
SQ = np.arange(64)
MIRROR = SQ ^ 56                       # canonical <-> original for a Black viewer (involution)
LIGHT = ((SQ % 8 + SQ // 8) % 2 == 1)  # canonical square colour
RANK = SQ // 8                         # canonical rank 0..7

# Observation char -> cur code, in canonical colours. For a White viewer
# uppercase is own; for a Black viewer lowercase is own.
_LUT_W = np.full(256, 255, dtype=np.uint8)
_LUT_B = np.full(256, 255, dtype=np.uint8)
for _lut, _own, _enemy in ((_LUT_W, 'PNBRQK', 'pnbrqk'), (_LUT_B, 'pnbrqk', 'PNBRQK')):
    _lut[ord('?')] = UNSEEN
    _lut[ord('-')] = EMPTY
    for _k in range(6):
        _lut[ord(_own[_k])] = OWN0 + _k
        _lut[ord(_enemy[_k])] = ENEMY0 + _k
# cur code -> last code (visible squares only)
CUR_TO_LAST = np.array([0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)
# plies since last visible -> age bucket
_AGE_BUCKET = np.array([0, 1, 2, 3, 3, 3], dtype=np.uint8)


def age_bucket(plies: np.ndarray) -> np.ndarray:
    """plies since last visible (-1 = never) -> age code."""
    out = np.where(plies >= 6, 4, _AGE_BUCKET[np.clip(plies, 0, 5)]).astype(np.uint8)
    out[plies < 0] = 5
    return out


def canonical_cur(obs64: str, viewer_is_white: bool) -> np.ndarray:
    """64-char observation (original orientation) -> cur codes [64] in canonical order."""
    raw = np.frombuffer(obs64.encode('ascii'), dtype=np.uint8)
    if raw.shape != (64,):
        raise ValueError('observation must be 64 characters')
    codes = (_LUT_W if viewer_is_white else _LUT_B)[raw if viewer_is_white else raw[MIRROR]]
    if (codes == 255).any():
        raise ValueError(f'bad observation characters: {obs64!r}')
    return codes


def to_canonical_square(sq: int, viewer_is_white: bool) -> int:
    return sq if viewer_is_white else sq ^ 56


def inv_slot(piece_type: int, canonical_sq: int) -> int:
    """Piece type 1..6 (P..K) on a canonical square -> inventory index 0..6."""
    if piece_type == 3:
        return 2 if LIGHT[canonical_sq] else 3
    return (0, 1, None, 4, 5, 6)[piece_type - 1]


def _square_index(name: str) -> int:
    f, r = ord(name[0]) - 97, ord(name[1]) - 49
    if not (0 <= f < 8 and 0 <= r < 8):
        raise ValueError(f'bad square {name!r}')
    return 8 * r + f


@dataclass
class ViewFeatures:
    """One viewer's features at one frame. Arrays are canonical-orientation codes."""
    cur: np.ndarray        # uint8 [64]
    last: np.ndarray       # uint8 [64]
    age: np.ndarray        # uint8 [64]
    inv: np.ndarray        # uint8 [7]
    ply: int               # 0..31
    to_move: int           # 0/1
    viewer_is_white: bool
    obs: str               # the current observation, original orientation
    t: int                 # frame index (plies played)


def stack_features(views: list[ViewFeatures]) -> dict[str, np.ndarray]:
    """Batch views into the model's int64 input dict (minus xt)."""
    return {
        'cur': np.stack([v.cur for v in views]).astype(np.int64),
        'last': np.stack([v.last for v in views]).astype(np.int64),
        'age': np.stack([v.age for v in views]).astype(np.int64),
        'inv': np.stack([v.inv for v in views]).astype(np.int64),
        'ply': np.array([v.ply for v in views], dtype=np.int64),
        'to_move': np.array([v.to_move for v in views], dtype=np.int64),
    }


class ViewTracker:
    """Incremental per-viewer feature state.

    Call observe() with every observation the viewer receives (one per ply,
    starting with the initial position), and own_move() with each move the
    viewer plays, after the observation it was played from and before the next.
    """

    def __init__(self, viewer_is_white: bool):
        self.white = bool(viewer_is_white)
        self.t = -1
        self.cur = np.zeros(64, dtype=np.uint8)
        self.last = np.zeros(64, dtype=np.uint8)
        self.last_seen = np.full(64, -1, dtype=np.int32)   # frame index when last visible
        self.inv = INV_START.copy()
        self.obs: str | None = None
        self.captures: list[tuple[int, int, int]] = []     # (frame t, piece type, canonical sq)
        self._moved_since_observe = False

    def observe(self, obs64: str) -> None:
        cur = canonical_cur(obs64, self.white)
        self.t += 1
        self.cur = cur
        self.obs = obs64
        vis = cur != UNSEEN
        self.last[vis] = CUR_TO_LAST[cur[vis]]
        self.last_seen[vis] = self.t
        self._moved_since_observe = False

    def own_move(self, uci: str) -> None:
        if self.obs is None:
            raise ValueError('own_move() before any observation')
        if self._moved_since_observe:
            raise ValueError('two own moves without an observation in between')
        self._moved_since_observe = True
        frm = to_canonical_square(_square_index(uci[0:2]), self.white)
        to = to_canonical_square(_square_index(uci[2:4]), self.white)
        mover = int(self.cur[frm])
        if not OWN0 <= mover < ENEMY0:
            raise ValueError(f'own_move {uci}: no own piece on the from-square')
        target = int(self.cur[to])
        captured = None
        if target >= ENEMY0:
            # Ordinary capture: the destination showed an enemy piece. Moves only
            # land on visible squares, so this is always known.
            captured, sq = target - ENEMY0 + 1, to
        elif mover == OWN0 and (frm % 8) != (to % 8) and target == EMPTY:
            # En passant: a pawn moves diagonally onto a visibly empty square.
            # The captured pawn stood behind the destination (canonical rank - 1).
            captured, sq = 1, to - 8
        if captured is not None:
            slot = inv_slot(captured, to)
            if self.inv[slot] > 0:
                self.inv[slot] -= 1
            elif self.inv[0] > 0:
                # More pieces of this type than the start set: the enemy promoted
                # a pawn earlier, so the capture removes one pawn unit instead.
                self.inv[0] -= 1
            self.captures.append((self.t, captured, sq))

    def features(self) -> ViewFeatures:
        if self.obs is None:
            raise ValueError('features() before any observation')
        age = age_bucket(np.where(self.last_seen >= 0, self.t - self.last_seen, -1))
        to_move = int((self.t % 2 == 0) == self.white)
        return ViewFeatures(cur=self.cur.copy(), last=self.last.copy(), age=age,
                            inv=self.inv.astype(np.uint8), ply=min(self.t // 8, 31),
                            to_move=to_move, viewer_is_white=self.white, obs=self.obs, t=self.t)


_CUR_CHARS = np.frombuffer(b'?-PNBRQKpnbrqk', dtype=np.uint8)
_SWAP = np.arange(256, dtype=np.uint8)
for _a, _b in zip(b'PNBRQK', b'pnbrqk'):
    _SWAP[_a], _SWAP[_b] = _b, _a


def view_from_arrays(cur, last, age, inv, ply, to_move, viewer_is_white: bool, t: int = -1) -> ViewFeatures:
    """Rebuild a ViewFeatures from stored dataset arrays (the observation string
    is recovered from cur). For benchmarks and offline evaluation."""
    cur = np.asarray(cur, dtype=np.uint8)
    canon = _CUR_CHARS[cur]
    raw = canon if viewer_is_white else _SWAP[canon][MIRROR]
    return ViewFeatures(cur=cur.copy(), last=np.asarray(last, np.uint8).copy(), age=np.asarray(age, np.uint8).copy(),
                        inv=np.asarray(inv, np.uint8).copy(), ply=int(ply), to_move=int(to_move),
                        viewer_is_white=bool(viewer_is_white), obs=raw.tobytes().decode('ascii'), t=int(t))


def board_from_canonical(values: np.ndarray, view: ViewFeatures) -> str:
    """Canonical per-square fill (0 empty, 1-6 enemy P..K; only read on unseen
    squares) -> 64-char full board in the original orientation and colours.
    Visible squares are copied from the viewer's observation."""
    enemy = 'pnbrqk' if view.viewer_is_white else 'PNBRQK'
    canon = ['-'] * 64
    for sq in np.flatnonzero(view.cur == UNSEEN):
        v = int(values[sq])
        canon[sq] = enemy[v - 1] if v > 0 else '-'
    out = list(view.obs)
    for sq in range(64):
        osq = sq if view.viewer_is_white else sq ^ 56
        if out[osq] == '?':
            out[osq] = canon[sq]
    return ''.join(out)


def canonical_target(board64: str, view: ViewFeatures) -> np.ndarray:
    """Full true/believed board (original orientation) -> canonical codes on
    every square: 0 empty, 1-6 enemy P..K, 7-12 own P..K. Used offline only."""
    raw = canonical_cur(board64.replace('?', '-'), view.viewer_is_white)
    out = np.zeros(64, dtype=np.int16)
    own = (raw >= OWN0) & (raw < ENEMY0)
    enemy = raw >= ENEMY0
    out[enemy] = raw[enemy] - ENEMY0 + 1
    out[own] = raw[own] - OWN0 + 7
    return out
