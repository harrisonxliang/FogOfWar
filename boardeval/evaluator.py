"""Position evaluators: given a fully-specified board, how good is it for `side`?

Centipawn scale, positive = good for `side`. Mode registry (NOTES.md §0):

- handcrafted:  material_eval everywhere; fog terms on the truth side.
- sf-hybrid:    material_eval for reply scoring ("their head" — smoothed by
                tau/beta/averaging, errors wash out), Stockfish depth-1 for
                true-board scoring ("our ledger" — unsmoothed, picks the move).
                Depth 1 includes qsearch, so hanging pieces are resolved.
- sf-full:      Stockfish everywhere. Offline ablation only — too slow live.

Guard order is a hard rule (NOTES.md §1): terminal_score (king capture) runs
BEFORE anything else, including before any Stockfish call. FoW positions are
routinely illegal under orthodox rules; Stockfish never sees them.

Fog-term semantics (NOTES.md §1): the evaluator receives a FULL-INFORMATION
board, so both fog terms are exact per-world computations — never worst-case
assumptions. Uncertainty lives above this layer (K-world sampling, delta-mass
veto), not inside it.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess

from referee import observation

WIN = 100_000.0
PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 310,
                chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
# Pawn advancement bonus by rank from the pawn owner's perspective.
PAWN_RANK_BONUS = [0, 0, 2, 5, 10, 20, 40, 0]

# Minor-piece centralization (tiny PST): bonus by Chebyshev ring from center.
_RING_BONUS = (12, 8, 4, 0)  # center 2x2, next ring, next, rim
CENTRALITY = [
    _RING_BONUS[int(max(abs(chess.square_file(sq) - 3.5),
                        abs(chess.square_rank(sq) - 3.5)) - 0.5)]
    for sq in range(64)
]

# Fog-term weights (cp). Hidden attacker on our king is the FoW death mode;
# a key piece inside their view is plannable-against (the hiding-move signal).
HIDDEN_ATTACKER_CP = 40
SEEN_KEY_PIECE_CP = 8
KEY_PIECES = {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING}


def terminal_score(board: chess.Board, side: chess.Color) -> float | None:
    """+-WIN if the position is already decided by king capture/absence, else None.

    MUST be the first check in every evaluator (precedence is unit-tested).
    King capture dynamics: whoever moves next can take a hanging king.
    """
    our_king = board.king(side)
    their_king = board.king(not side)
    if our_king is None:
        return -WIN
    if their_king is None:
        return WIN
    mover = board.turn
    if mover == side and board.is_attacked_by(side, their_king):
        return WIN
    if mover != side and board.is_attacked_by(not side, our_king):
        return -WIN
    return None


def fog_terms(board: chess.Board, side: chess.Color) -> float:
    """The two FoW terms, exact per world, symmetric in the two colors.

    - hidden attackers: enemy pieces attacking a king from squares OUTSIDE the
      king owner's vision (threats they would not see coming in this world);
    - visibility: key pieces standing inside the enemy's vision mask.
    Costs two observation() calls; only the truth-side evaluator pays it.
    """
    views = {color: observation(board, color) for color in (True, False)}
    total = 0.0
    for color, sign in ((side, 1.0), (not side, -1.0)):
        own_view = views[color]
        enemy_view = views[not color]
        king = board.king(color)
        hidden = sum(1 for sq in board.attackers(not color, king)
                     if own_view[sq] == '?')
        seen = sum(1 for sq, piece in board.piece_map().items()
                   if piece.color == color and piece.piece_type in KEY_PIECES
                   and enemy_view[sq] != '?')
        total += sign * (-HIDDEN_ATTACKER_CP * hidden - SEEN_KEY_PIECE_CP * seen)
    return total


def material_eval(board: chess.Board, side: chess.Color, fog: bool = False) -> float:
    """Material + PST + king safety (+ optional fog terms), from `side`'s view.

    fog=False is the reply-side default (microseconds, called ~2,400x/move);
    fog=True is the truth-side handcrafted mode (pays two observation() calls).
    """
    terminal = terminal_score(board, side)
    if terminal is not None:
        return terminal
    score = 0.0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        if piece.piece_type == chess.PAWN:
            rank = chess.square_rank(square)
            value += PAWN_RANK_BONUS[rank if piece.color else 7 - rank]
        elif piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            value += CENTRALITY[square]
        score += value if piece.color == side else -value
    # Light king safety: shelter pawns in front of the king.
    for color, sign in ((side, 1), (not side, -1)):
        king = board.king(color)
        shield = 0
        rank = chess.square_rank(king)
        step = 1 if color else -1
        if 0 <= rank + step <= 7:
            for df in (-1, 0, 1):
                file = chess.square_file(king) + df
                if 0 <= file <= 7:
                    piece = board.piece_at(chess.square(file, rank + step))
                    if piece == chess.Piece(chess.PAWN, color):
                        shield += 1
        score += sign * 12 * shield
    # Latent danger: enemy attackers bearing on our king's square (and ours on theirs).
    score -= 25 * len(board.attackers(not side, board.king(side)))
    score += 25 * len(board.attackers(side, board.king(not side)))
    if fog:
        score += fog_terms(board, side)
    return score


def tanh_eval(board: chess.Board, side: chess.Color, scale: float = 600.0,
              fog: bool = True) -> float:
    """Normalized [-1, 1] evaluator per the proposal: +-1 on king capture,
    otherwise tanh(cp / scale). Sweep scale over {300, 600, 1200} (NOTES.md §1).
    """
    terminal = terminal_score(board, side)
    if terminal is not None:
        return 1.0 if terminal > 0 else -1.0
    return math.tanh(material_eval(board, side, fog=fog) / scale)


class StockfishEvaluator:
    """Depth-1 Stockfish evaluation with the NOTES.md §1 guards, in order:

    1. terminal_score first — Stockfish never sees a decided/king-hanging state;
    2. orthodox-illegal or engine error -> material_eval fallback, cost one eval;
    3. engine process death -> permanent mode flip to material_eval (self.dead),
       the game continues;
    4. per-turn FEN cache so anytime-loop re-visits are free (call new_turn()).
    """

    def __init__(self, engine_path='stockfish'):
        import chess.engine
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.limit = chess.engine.Limit(depth=1)
        self.cache: dict = {}
        self.dead = False
        self.fallbacks = 0
        self.calls = 0

    def new_turn(self):
        """Clear the per-turn eval cache (planner calls this on each clock line)."""
        self.cache.clear()

    def evaluate(self, board: chess.Board, side: chess.Color) -> float:
        import chess.engine
        self.calls += 1
        terminal = terminal_score(board, side)  # guard 1: before the engine, always
        if terminal is not None:
            return terminal
        if self.dead:
            self.fallbacks += 1
            return material_eval(board, side)
        key = (board.fen(), side)
        if key in self.cache:
            return self.cache[key]
        if not board.is_valid():  # guard 2: orthodox-illegal FoW states
            self.fallbacks += 1
            return material_eval(board, side)
        try:
            info = self.engine.analyse(board, self.limit)
            value = float(info['score'].pov(side).score(mate_score=int(WIN)))
        except chess.engine.EngineTerminatedError:
            self.dead = True  # guard 3: mode flip, never crash the move
            self.fallbacks += 1
            value = material_eval(board, side)
        except chess.engine.EngineError:
            self.fallbacks += 1
            value = material_eval(board, side)
        self.cache[key] = value
        return value

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass  # a dead engine has nothing to quit


def get_evaluator(kind='material', engine_path='stockfish'):
    """Returns (evaluate_fn, close_fn). Single-evaluator interface (X1 etc.)."""
    if kind == 'material':
        return material_eval, lambda: None
    if kind == 'stockfish':
        sf = StockfishEvaluator(engine_path)
        return sf.evaluate, sf.close
    raise ValueError(f'unknown evaluator: {kind}')


def get_eval_config(mode='handcrafted', engine_path='stockfish'):
    """The two-sided evaluator seam for the planner (mode registry, NOTES.md §0).

    Returns {'mode', 'reply', 'truth', 'new_turn', 'close'}:
      reply(board, side) — scores opponent replies on believed boards;
      truth(board, side) — scores V(x, a) outcomes on the true board.
    The time manager may rebuild this with mode='handcrafted' under clock
    pressure (sf-hybrid -> handcrafted is a degradation rung).
    """
    if mode == 'handcrafted':
        return {'mode': mode,
                'reply': material_eval,
                'truth': lambda b, s: material_eval(b, s, fog=True),
                'new_turn': lambda: None,
                'close': lambda: None}
    if mode in ('sf-hybrid', 'sf-full'):
        sf = StockfishEvaluator(engine_path)
        return {'mode': mode,
                'reply': material_eval if mode == 'sf-hybrid' else sf.evaluate,
                'truth': sf.evaluate,
                'new_turn': sf.new_turn,
                'close': sf.close}
    raise ValueError(f'unknown eval mode: {mode}')
