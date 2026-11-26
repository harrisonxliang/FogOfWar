"""The planner: the one component that sees a whole turn.

Per-move flow (proposal steps 2-7; NOTES.md §5 anytime rules):

  1. Compute the PIMC fallback FIRST — a move is always ready from here on.
  2. Anytime rounds over the widening schedule: sample worlds (weighting),
     shortlist moves, score every (world, move) pair, veto + aggregate.
  3. The incumbent move is replaced only at COMPLETED-round boundaries; a
     deadline mid-round discards the partial round (a partial V matrix
     compares moves over different world sets — biased). Scores are never
     compared ACROSS rounds; the widest completed round supersedes.

Scoring one (world x, move a) pair:
  x2 = x + a. Terminal -> +-WIN. beta >= 1 is the clairvoyant fast path
  (method A: their best reply on x2, no belief machinery). Otherwise:
  believed boards from the belief source, replies enumerated on x2 (they
  always know their own moves), scored on the believed boards from THEIR
  side, softmax at tau + smoothing floor, beta-mix, then
  V = sum p(r) * truth_eval(x2 + r) — predicted on what they believe,
  settled on what's true.
"""
from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess

from referee import fow_moves
from boardeval.decide import choose
from boardeval.evaluator import get_eval_config, terminal_score
from boardeval.reply import mix_with_clairvoyant, rank_moves, to_distribution
from boardeval.weighting import sample_worlds
from bot.belief_sources import get_belief_source
from bot.shortlist import build_shortlist, eval_after_best_reply


def score_pair(world, move, side, cfg, source, reply_eval, truth_eval, rng):
    """V(x, a) for one world and one candidate move."""
    x2 = world.copy(stack=False)
    x2.push(move)
    terminal = terminal_score(x2, side)
    if terminal is not None:
        return terminal

    if cfg.beta >= 1.0:
        # Clairvoyant opponent (method A / fallback): their best reply on the
        # true x2. No belief source, no distribution — one minimization.
        return eval_after_best_reply(x2, truth_eval, side)

    replies = fow_moves(x2)  # they always know their own moves (on this world)
    if not replies:
        return truth_eval(x2, side)
    legal_ucis = [reply.uci() for reply in replies]

    # Rank replies on the boards THEY might believe in, from THEIR side.
    boards = source.boards_for_reply(x2, side, cfg.J, rng)
    totals, counts = {}, {}
    for believed in boards:
        for reply in replies:
            if not believed.is_pseudo_legal(reply):
                continue  # impossible under this belief; the floor keeps it alive
            believed.push(reply)
            value = reply_eval(believed, not side)
            believed.pop()
            uci = reply.uci()
            totals[uci] = totals.get(uci, 0.0) + value
            counts[uci] = counts.get(uci, 0) + 1
    ranked = sorted(((uci, totals[uci] / counts[uci]) for uci in totals),
                    key=lambda pair: -pair[1])

    p = to_distribution(ranked, legal_ucis, tau=cfg.tau, floor=cfg.floor)
    if cfg.beta > 0.0:
        clair = rank_moves(x2, reply_eval, not side)
        p = mix_with_clairvoyant(p, clair, cfg.beta)

    # Settle on the true board.
    value = 0.0
    for reply in replies:
        prob = p.get(reply.uci(), 0.0)
        if prob <= 0.0:
            continue
        x2.push(reply)
        value += prob * truth_eval(x2, side)
        x2.pop()
    return value


def pimc_fallback(worlds, legal_moves, side, cfg, truth_eval, rng):
    """Proposal step 2: the always-ready safe move. A handful of uniform
    worlds x every legal move x eval-after-their-best-reply, with the veto."""
    picked = worlds if len(worlds) <= cfg.fallback_worlds else \
        [worlds[int(i)] for i in rng.choice(len(worlds), size=cfg.fallback_worlds,
                                            replace=False)]
    values = np.zeros((len(legal_moves), len(picked)))
    for wi, world in enumerate(picked):
        for mi, move in enumerate(legal_moves):
            x2 = world.copy(stack=False)
            if not world.is_pseudo_legal(move):
                values[mi, wi] = 0.0
                continue
            x2.push(move)
            values[mi, wi] = eval_after_best_reply(x2, truth_eval, side)
    move, _ = choose([mv.uci() for mv in legal_moves], values,
                     lam=cfg.lam, veto_threshold=cfg.veto_delta)
    return move


def plan_move(worlds, legal_moves, side, cfg, budget_s, rng,
              predictor=None, state=None, danger_index=None,
              eval_config=None, belief_source=None, now=time.monotonic):
    """Run one turn. Returns (uci, diagnostics). Never raises for lack of time.

    worlds: candidate world boards (the tracker's enumerated set, or the stub's
    single reconstruction). legal_moves: the EXACT legal set from the
    observation tracker — worlds are for scoring, never for legality.
    `now` is injectable for the anytime tests.
    """
    if cfg.own_worlds != 'tracker':
        raise NotImplementedError(
            f"method {cfg.method}: own_worlds='{cfg.own_worlds}' needs P1's "
            "own_worlds(view, K) (Dec 3-5); only tracker worlds run today.")
    deadline = now() + budget_s
    evals = eval_config or get_eval_config(cfg.eval_mode)
    evals['new_turn']()
    source = belief_source or get_belief_source(cfg, predictor=predictor)

    fallback = pimc_fallback(worlds, legal_moves, side, cfg, evals['truth'], rng)
    incumbent = fallback
    diag = {'fallback': fallback, 'completed_rounds': 0, 'rounds': []}

    for k, m in cfg.widening:
        if now() >= deadline:
            break
        indices, agg_weights, sample_diag = sample_worlds(
            worlds, k, rng, predictor=predictor, state=state,
            epsilon=cfg.epsilon, mode=cfg.selection_mode,
            danger_index=danger_index)
        picked = [worlds[i] for i in indices]

        if m is None or m >= len(legal_moves):
            candidates, slots = list(legal_moves), None
        else:
            candidates, short_diag = build_shortlist(
                picked[0], side, legal_moves, evals['reply'], m)
            slots = short_diag['slots']
        if not candidates:
            break

        values = np.zeros((len(candidates), len(picked)))
        aborted = False
        for wi, world in enumerate(picked):
            for mi, move in enumerate(candidates):
                if now() >= deadline:
                    aborted = True
                    break
                values[mi, wi] = score_pair(world, move, side, cfg, source,
                                            evals['reply'], evals['truth'], rng)
            if aborted:
                break
        if aborted:
            break  # NOTES §5: a partial round is discarded, never adopted

        move, choice_diag = choose([mv.uci() for mv in candidates], values,
                                   weights=agg_weights, lam=cfg.lam,
                                   veto_threshold=cfg.veto_delta)
        incumbent = move
        diag['completed_rounds'] += 1
        diag['rounds'].append({
            'k': len(picked), 'm': len(candidates), 'move': move,
            'vetoed': choice_diag['vetoed'],
            'captured_mass': sample_diag['captured_mass'],
            'slots': {name: mv.uci() if isinstance(mv, chess.Move) else
                      [c.uci() for c in mv] if isinstance(mv, list) else mv
                      for name, mv in (slots or {}).items()} if slots else None,
        })

    return incumbent, diag
