"""Single-world stub tracker — the planner's world source until P2's
enumerating tracker lands (guide: "develop against a stub tracker").

Maintains, from our own observations only:
  - exact legal moves (uniform_random.ObservationTracker: castling rights +
    en-passant inference — reused, not rewritten);
  - one reconstructed world via the last-seen rule (shared with the naive X1
    contestant: boardeval.reply.last_seen_board);
  - an approximate danger flag (any enemy piece in the reconstruction
    attacking our king).

The real tracker replaces worlds() with the enumerated set S and danger()
with the exact any-world flag; the planner's call sites don't change.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uniform_random import ObservationTracker
from boardgen.encoding import HistoryTracker
from boardeval.reply import board_from_squares, last_seen_board


class StubTracker:
    def __init__(self, side: bool):
        self.side = side
        self.obs = ObservationTracker(side)
        self.history = HistoryTracker(side)
        self.current = None

    def update(self, observation: str) -> None:
        self.obs.observe(observation)
        self.history.update(observation)
        self.current = observation

    def legal_moves(self):
        return self.obs.legal_moves()

    def record_move(self, move) -> None:
        self.obs.record_move(move)

    def worlds(self):
        """One last-seen world, us to move, with BOTH kings guaranteed.

        Kings hide for most of a FoW game, so the enemy king is frequently
        absent from a pure last-seen reconstruction. A world with no enemy
        king evaluates to +WIN for every move — the veto goes blind and the
        planner plays arbitrary moves (observed: the bot walked its king onto
        a remembered bishop's diagonal). If the enemy king is missing, place
        it on a plausible unseen square instead of leaving it off the board.
        """
        board = last_seen_board(self.current, self.history, self.side)
        if board is None:
            board = board_from_squares(
                [ch if ch not in '?' else '-' for ch in self.current], self.side)
        if board.king(not self.side) is None:
            square = self._guess_enemy_king_square(board)
            if square is not None:
                import chess
                board.set_piece_at(square, chess.Piece(chess.KING, not self.side))
        return [board]

    def _guess_enemy_king_square(self, board):
        """A plausible square for the unseen enemy king: unknown in the
        current observation, empty in the reconstruction, not attacked by us
        (a king we instantly 'capture' would make every move look +WIN),
        scanning from their home rank toward us."""
        ranks = range(7, -1, -1) if self.side else range(8)
        fallback = None
        for rank in ranks:
            for file in range(8):
                square = rank * 8 + file
                if self.current[square] != '?' or board.piece_at(square):
                    continue
                if fallback is None:
                    fallback = square
                if not board.attackers(self.side, square):
                    return square
        return fallback

    def danger(self) -> bool:
        """Approximate: does the reconstruction let them capture our king?"""
        world = self.worlds()[0]
        king = world.king(self.side)
        return king is not None and bool(world.attackers(not self.side, king))
