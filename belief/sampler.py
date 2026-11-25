"""DiffusionBelief: constrained masked-diffusion sampler. Play-time code: stdlib,
NumPy and onnxruntime only.

All views x J samples are batched into one ONNX call per denoising step.
Per sequence:
  1. U = unseen squares; budget need[slot] = inv - visible enemy pieces (bishops
     split by canonical square colour), negatives clipped to 0 and counted
     (they mean an unseen promotion). xt = VIS on visible squares, MASK on U.
  2. For step s = 1..T: one forward pass; unmask n = ceil(masked / (T - s + 1))
     squares, ordered by confidence (log max constrained probability + Gumbel
     noise); commit them one at a time, updating the budget after each commit.
  3. Constraints before each commit: no piece type whose budget is exhausted;
     a bishop only on a square colour with budget left; no pawns on canonical
     ranks 1/8; no "empty" when every remaining masked square must hold a piece.
     If nothing is allowed, place empty and set the violated flag.
  4. Map back to the original orientation: a full 64-char board with the
     viewer's observation on visible squares and enemy letters or '-' elsewhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from belief.features import (ENEMY0, LIGHT, MIRROR, RANK, UNSEEN, ViewFeatures, canonical_target,
                             stack_features)

MASK, VIS = 7, 8
N_CLS = 7
INPUT_NAMES = ('cur', 'last', 'age', 'xt', 'inv', 'ply', 'to_move')
_CLASS_SLOT = np.array([-1, 0, 1, -1, 4, 5, 6])      # bishops (class 3) resolved by square colour
_ENEMY_CHARS = {True: np.frombuffer(b'-pnbrqk', np.uint8), False: np.frombuffer(b'-PNBRQK', np.uint8)}


@dataclass
class SamplerConfig:
    hard_inventory: bool = True     # enforce the inventory budget
    rank_rules: bool = True         # no pawns on canonical ranks 1 and 8
    allow_promotion: bool = False   # a pawn budget unit may become N/B/R/Q ...
    promotion_prior: float = 0.05   # ... with its probability scaled by this prior
    temperature: float = 1.0


def make_session(onnx_path: str, threads: int = 2):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def budget(cur: np.ndarray, inv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """need [B,7] = inv minus visible enemy pieces (clipped at 0), deficit [B] = clipped amount."""
    counts = np.zeros(inv.shape, dtype=np.int64)
    cls = np.where(cur >= ENEMY0, cur.astype(np.int64) - ENEMY0 + 1, 0)
    for c in (1, 2, 4, 5, 6):
        counts[:, _CLASS_SLOT[c]] = (cls == c).sum(1)
    counts[:, 2] = ((cls == 3) & LIGHT).sum(1)
    counts[:, 3] = ((cls == 3) & ~LIGHT).sum(1)
    need = inv.astype(np.int64) - counts
    return np.maximum(need, 0), np.maximum(-need, 0).sum(1)


class DiffusionBelief:
    def __init__(self, onnx_path: str | None = None, threads: int = 2, config: SamplerConfig | None = None,
                 forward=None):
        """forward: optional callable(dict of int64 arrays) -> float32 logits [B,64,7]
        replacing ONNX Runtime (used by the export parity test)."""
        self.config = config or SamplerConfig()
        if forward is None:
            sess = make_session(onnx_path, threads)
            names = [i.name for i in sess.get_inputs()]
            forward = lambda inputs: sess.run(['logits'], {k: inputs[k] for k in names})[0]
        self._forward = forward
        self.last_info: dict = {}

    # --- public API ----------------------------------------------------------------------

    def warmup(self, batch: int = 64, T: int = 1) -> None:
        cur = np.zeros(64, np.uint8)
        cur[:16] = 2                                  # a dummy view: own pieces on ranks 1-2
        v = ViewFeatures(cur=cur, last=np.zeros(64, np.uint8), age=np.full(64, 5, np.uint8),
                         inv=np.array([8, 2, 1, 1, 2, 1, 1], np.uint8), ply=0, to_move=1,
                         viewer_is_white=True, obs='P' * 16 + '?' * 48, t=0)
        self.sample([v], batch, T, np.random.default_rng(0))

    def sample(self, views: list[ViewFeatures], J: int, T: int, rng: np.random.Generator) -> list[list[str]]:
        """J believed full boards (64-char, original orientation) per view."""
        boards, info = self.sample_with_info(views, J, T, rng)
        self.last_info = info
        return boards

    def marginals(self, view: ViewFeatures) -> np.ndarray:
        """[64, 7] probabilities over {empty, enemy P N B R Q K}, indexed by ORIGINAL
        square, from one pass with all unseen squares masked. Visible squares are
        one-hot on what is observed (own pieces count as 'empty' of enemy pieces)."""
        x = stack_features([view])
        x['xt'] = np.where(x['cur'] == UNSEEN, MASK, VIS).astype(np.int64)
        p = np.exp(_log_softmax(self._forward(x)[0].astype(np.float64) / self.config.temperature))
        vis = view.cur != UNSEEN
        onehot = np.where(view.cur >= ENEMY0, view.cur.astype(np.int64) - ENEMY0 + 1, 0)
        p[vis] = np.eye(N_CLS)[onehot[vis]]
        return p if view.viewer_is_white else p[MIRROR]

    def log_prob(self, view: ViewFeatures, board: str) -> float:
        """Exact chain-rule log p(board | view) under the raw (unconstrained) model:
        unmask one unseen square per pass in ascending canonical-square order.
        The |U| passes run as one batch. -inf if the board contradicts the view."""
        truth = canonical_target(board, view)
        unseen = np.flatnonzero(view.cur == UNSEEN)
        vis = view.cur != UNSEEN
        seen = np.where(view.cur >= ENEMY0, view.cur.astype(np.int64) - ENEMY0 + 1,
                        np.where(view.cur >= 2, view.cur.astype(np.int64) - 2 + 7, 0))
        if not np.array_equal(truth[vis], seen[vis]) or (truth[unseen] > 6).any():
            return float('-inf')
        n = len(unseen)
        if n == 0:
            return 0.0
        x = {k: np.repeat(v, n, axis=0) for k, v in stack_features([view]).items()}
        xt = np.where(x['cur'] == UNSEEN, MASK, VIS).astype(np.int64)
        prefix = np.tril(np.ones((n, n), dtype=bool), -1)          # row k reveals unseen[:k]
        xt[:, unseen] = np.where(prefix, truth[unseen][None, :], MASK)
        x['xt'] = xt
        logp = _log_softmax(self._forward(x).astype(np.float64))
        return float(logp[np.arange(n), unseen, truth[unseen]].sum())

    # --- sampler -------------------------------------------------------------------------

    def _allowed(self, rows, sq, need, m_rem):
        """Allowed classes [n,7] and which of them consume a pawn unit (promotion)."""
        cfg, n = self.config, len(rows)
        ok = np.ones((n, N_CLS), dtype=bool)
        promo = np.zeros((n, N_CLS), dtype=bool)
        if cfg.hard_inventory:
            nd = need[rows]
            bslot = np.where(LIGHT[sq], 2, 3)
            ok[:, 1] = nd[:, 0] > 0
            ok[:, 2] = nd[:, 1] > 0
            ok[:, 3] = nd[np.arange(n), bslot] > 0
            ok[:, 4:7] = nd[:, 4:7] > 0
            if cfg.allow_promotion:
                promo[:, 2:6] = ~ok[:, 2:6] & (nd[:, 0] > 0)[:, None]
                ok |= promo
            ok[:, 0] = m_rem[rows] > nd.sum(1)                     # every remaining square must hold a piece
        if cfg.rank_rules:
            back = (RANK[sq] == 0) | (RANK[sq] == 7)
            ok[back, 1] = False
        return ok, promo

    def sample_with_info(self, views, J, T, rng):
        cfg = self.config
        x = {k: np.repeat(v, J, axis=0) for k, v in stack_features(views).items()}
        B = len(x['ply'])
        unseen = x['cur'] == UNSEEN
        need, deficit = budget(x['cur'], x['inv'])
        xt = np.where(unseen, MASK, VIS).astype(np.int64)
        m_rem = unseen.sum(1)
        violated = np.zeros(B, dtype=bool)
        promoted = np.zeros(B, dtype=np.int64)
        for s in range(1, T + 1):
            x['xt'] = xt
            logp = _log_softmax(self._forward(x).astype(np.float64) / cfg.temperature)   # [B,64,7]
            masked = xt == MASK
            n_s = np.where(s == T, m_rem, np.ceil(m_rem / (T - s + 1)).astype(np.int64))
            if n_s.max() == 0:
                break
            # confidence order: log max constrained probability + Gumbel noise
            rows, sqs = np.nonzero(masked)
            ok, promo = self._allowed(rows, sqs, need, m_rem)
            lp = np.where(ok, logp[rows, sqs], -np.inf)
            lp = np.where(promo, lp + math.log(cfg.promotion_prior), lp)
            conf = np.full((B, 64), -np.inf)
            conf[rows, sqs] = lp.max(1) + rng.gumbel(size=len(rows))
            conf[rows, sqs] = np.where(np.isfinite(conf[rows, sqs]), conf[rows, sqs], -1e30)
            order = np.argsort(-conf, axis=1, kind='stable')
            for k in range(int(n_s.max())):
                r = np.flatnonzero(k < n_s)
                sq = order[r, k]
                ok, promo = self._allowed(r, sq, need, m_rem)
                p = np.exp(logp[r, sq]) * ok
                p = np.where(promo, p * cfg.promotion_prior, p)
                tot = p.sum(1)
                bad = tot <= 0
                violated[r[bad]] = True
                p[bad] = 0
                p[bad, 0] = 1
                cdf = np.cumsum(p / p.sum(1, keepdims=True), 1)
                c = np.minimum((cdf < rng.random(len(r))[:, None]).sum(1), N_CLS - 1)
                xt[r, sq] = c
                m_rem[r] -= 1
                used_promo = promo[np.arange(len(r)), c]
                promoted[r] += used_promo
                slot = np.where(c == 3, np.where(LIGHT[sq], 2, 3), _CLASS_SLOT[c])
                slot = np.where(used_promo, 0, slot)
                pieces = c > 0
                need[r[pieces], slot[pieces]] = np.maximum(need[r[pieces], slot[pieces]] - 1, 0)
        values = np.where(unseen, xt, 0)
        boards = self._boards(views, values, J)
        info = {'violated': violated, 'promo_deficit': deficit, 'promoted': promoted,
                'unplaced': need.sum(1) if cfg.hard_inventory else None, 'values': values}
        return boards, info

    @staticmethod
    def _boards(views, values, J):
        out = []
        for i, v in enumerate(views):
            vals = values[i * J:(i + 1) * J]
            if not v.viewer_is_white:
                vals = vals[:, MIRROR]                             # canonical -> original square order
            obs = np.frombuffer(v.obs.encode('ascii'), np.uint8)
            fill = _ENEMY_CHARS[v.viewer_is_white][vals]
            full = np.where(obs[None, :] == ord('?'), fill, obs[None, :])
            out.append([row.tobytes().decode('ascii') for row in full])
        return out


def check_board(view: ViewFeatures, board: str) -> dict:
    """Validity of one believed board against a view (offline checks)."""
    truth = canonical_target(board, view)
    u = view.cur == UNSEEN
    enemy = (truth >= 1) & (truth <= 6)
    counts = np.zeros(7, dtype=np.int64)
    for c in (1, 2, 4, 5, 6):
        counts[_CLASS_SLOT[c]] = ((truth == c) & enemy).sum()
    counts[2] = ((truth == 3) & LIGHT).sum()
    counts[3] = ((truth == 3) & ~LIGHT).sum()
    pawns_back = bool(((truth == 1) & ((RANK == 0) | (RANK == 7))).any())
    obs_ok = all(board[i] == ch for i, ch in enumerate(view.obs) if ch != '?')
    return {'one_king': counts[6] == 1, 'within_inv': bool((counts <= view.inv).all()),
            'no_back_pawns': not pawns_back, 'obs_consistent': obs_ok,
            'unseen_only_enemy': not ((truth > 6) & u).any()}
