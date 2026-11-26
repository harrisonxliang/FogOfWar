"""Candidate-move shortlist: which of our legal moves get the expensive
K-worlds x diffusion x reply-distribution treatment.

Deterministic, never sampled: every legal move gets the cheap score (1-ply
eval-after-their-best-reply on one world — same evaluator as the deep
analysis, just without the belief machinery), the top M survive, PLUS the
four guaranteed slots. The slots exist because the cheap scorer is blind to
exactly the fog-value moves the thesis is about (proposal step 4):

  hiding    — removes a key piece from the opponent's view
  scouting  — reveals the most enemy squares to us
  captures  — captures of visible pieces (in FoW every capture target is
              visible by definition of vision; kept as its own slot so strong
              captures survive a bad cheap rank)
  king_safety — most reduces the attacker count on our king

Costs ~(len(legal) x ~30 reply evals + 2 observation calls per move): a few
tens of ms with the handcrafted evaluator. Stockfish never runs here.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess

from referee import fow_moves, observation
from boardeval.evaluator import KEY_PIECES, PIECE_VALUES, terminal_score

CAPTURE_SLOTS = 4  # at most this many capture moves force-included, by victim value


def eval_after_best_reply(x2, evaluate, side):
    """Our score after the opponent's best reply on this full board.

    The evaluator is side-symmetric (tested), so their best reply is the one
    minimizing our eval. Terminal positions short-circuit. This one function
    is the cheap scorer, the PIMC fallback's core, and the planner's beta=1
    (clairvoyant) fast path — one definition, three budgets.
    """
    terminal = terminal_score(x2, side)
    if terminal is not None:
        return terminal
    best = None
    for reply in fow_moves(x2):
        x2.push(reply)
        value = evaluate(x2, side)
        x2.pop()
        if best is None or value < best:
            best = value
    return best if best is not None else evaluate(x2, side)


def _count_seen_key_pieces(board, side, enemy_view):
    return sum(1 for sq, piece in board.piece_map().items()
               if piece.color == side and piece.piece_type in KEY_PIECES
               and enemy_view[sq] != '?')


def build_shortlist(world, side, legal_moves, reply_eval, m):
    """Returns (moves, diag): the deduped candidate list and which move won
    each slot. `legal_moves` must be the EXACT legal set (from the observation
    tracker) — the world is only used for scoring and views.
    """
    their_view_before = observation(world, not side)
    our_view_before = observation(world, side)
    seen_before = _count_seen_key_pieces(world, side, their_view_before)
    visible_before = sum(1 for ch in our_view_before if ch != '?')
    attackers_before = len(world.attackers(not side, world.king(side)))

    cheap, hiding, scouting, safety, captures = {}, [], [], [], []
    for move in legal_moves:
        if not world.is_pseudo_legal(move):
            # World disagrees with reality on a visible square — shouldn't
            # happen for consistent worlds; skip rather than crash.
            continue
        victim = world.piece_at(move.to_square)
        world.push(move)
        cheap[move] = eval_after_best_reply(world, reply_eval, side)
        their_view_after = observation(world, not side)
        our_view_after = observation(world, side)
        hide_gain = seen_before - _count_seen_key_pieces(world, side, their_view_after)
        scout_gain = sum(1 for ch in our_view_after if ch != '?') - visible_before
        our_king = world.king(side)
        safety_gain = (attackers_before
                       - (len(world.attackers(not side, our_king)) if our_king else 99))
        world.pop()
        if hide_gain > 0:
            hiding.append((hide_gain, move))
        if scout_gain > 0:
            scouting.append((scout_gain, move))
        if safety_gain > 0:
            safety.append((safety_gain, move))
        if victim is not None:
            captures.append((PIECE_VALUES[victim.piece_type], move))

    ranked = sorted(cheap, key=lambda mv: -cheap[mv])
    best = lambda pairs: max(pairs, key=lambda p: p[0])[1]  # noqa: E731 — gain only; Moves don't order
    slots = {
        'hiding': best(hiding) if hiding else None,
        'scouting': best(scouting) if scouting else None,
        'king_safety': best(safety) if safety else None,
        'captures': [mv for _, mv in sorted(captures, key=lambda p: -p[0])[:CAPTURE_SLOTS]],
    }

    chosen, seen = [], set()
    for move in ranked[:m] + [slots['hiding'], slots['scouting'],
                              slots['king_safety']] + slots['captures']:
        if move is not None and move not in seen:
            seen.add(move)
            chosen.append(move)
    diag = {'cheap': cheap, 'slots': slots}
    return chosen, diag
