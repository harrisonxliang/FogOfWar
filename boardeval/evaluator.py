"""Position evaluators: given a fully-specified board, how good is it for `side`?

Two implementations behind one call signature `evaluate(board, side) -> float`
(centipawn-ish scale, positive = good for `side`):

- material_eval: material + FoW king danger + light positional terms. Pure
  Python, microseconds, no dependencies. The live-bot default.
- StockfishEvaluator: Stockfish static-ish eval (depth-1 search). FoW positions
  are frequently illegal under orthodox rules (the side not to move may be "in
  check"), so every call is guarded and falls back to material_eval.

FoW scoring convention used everywhere: a side whose king is already capturable
on the opponent's turn is lost; a missing king is lost.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess

WIN = 100_000.0
PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 310,
                chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
# Pawn advancement bonus by rank from the pawn owner's perspective.
PAWN_RANK_BONUS = [0, 0, 2, 5, 10, 20, 40, 0]


def material_eval(board: chess.Board, side: chess.Color) -> float:
    """Material + FoW king danger, from `side`'s perspective."""
    our_king = board.king(side)
    their_king = board.king(not side)
    if our_king is None:
        return -WIN
    if their_king is None:
        return WIN
    # King capture dynamics: whoever moves next can take a hanging king.
    mover = board.turn
    if mover == side and board.is_attacked_by(side, their_king):
        return WIN  # we capture their king on the spot
    if mover != side and board.is_attacked_by(not side, our_king):
        return -WIN  # they capture ours
    score = 0.0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        if piece.piece_type == chess.PAWN:
            rank = chess.square_rank(square)
            value += PAWN_RANK_BONUS[rank if piece.color else 7 - rank]
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
    score -= 25 * len(board.attackers(not side, our_king))
    score += 25 * len(board.attackers(side, their_king))
    return score


class StockfishEvaluator:
    """Depth-1 Stockfish evaluation with orthodox-legality guards.

    Usage: ev = StockfishEvaluator('path/to/stockfish'); ev.evaluate(board, side).
    Positions Stockfish cannot represent (opponent already in "check", missing
    king, etc.) fall back to material_eval. Call close() when done.
    """

    def __init__(self, engine_path='stockfish'):
        import chess.engine
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.limit = chess.engine.Limit(depth=1)
        self.fallbacks = 0
        self.calls = 0

    def evaluate(self, board: chess.Board, side: chess.Color) -> float:
        import chess.engine
        self.calls += 1
        # Orthodox-illegal states (side not to move in check, no kings, etc.)
        if board.king(True) is None or board.king(False) is None or not board.is_valid():
            self.fallbacks += 1
            return material_eval(board, side)
        try:
            info = self.engine.analyse(board, self.limit)
            return float(info['score'].pov(side).score(mate_score=int(WIN)))
        except chess.engine.EngineError:
            self.fallbacks += 1
            return material_eval(board, side)

    def close(self):
        self.engine.quit()


def get_evaluator(kind='material', engine_path='stockfish'):
    """Returns (evaluate_fn, close_fn)."""
    if kind == 'material':
        return material_eval, lambda: None
    if kind == 'stockfish':
        sf = StockfishEvaluator(engine_path)
        return sf.evaluate, sf.close
    raise ValueError(f'unknown evaluator: {kind}')
