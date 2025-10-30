""" 
Leduc Hold'em minimal environment for Behavior Cloning (BC)
- Two-player limit poker, 2 betting rounds (pre-turn, post-turn)
- Deck: ranks {J, Q, K}, each duplicated (6 cards total)
- Bets: round1 bet size = 1, round2 bet size = 2
- Max 2 bets/raises per round (cap), standard Leduc
- Terminal on fold or after showdown

This file provides:
 1) LeducEnv: a gym-like environment (reset/step) with legal action masking
 2) ExpertPolicy: a simple rule-based expert to generate BC trajectories
 3) rollout() / make_bc_dataset(): utilities to create (s, a) pairs for BC
 4) Example main block to generate dataset.npz for train_bc.py

State encoding (default):
  - private card one-hot (3) -> [J,Q,K]
  - public card one-hot (4) -> [none,J,Q,K]
  - player position (1) -> [is_button] {0/1}
  - betting history (4) -> [bets_this_round (0..2) one-hot size 3] + [opponent_acted (0/1)]
  - pot size normalized (1) -> pot / 10 (simple scale)
  - round index one-hot (2) -> [round0, round1]
Total dim: 3 + 4 + 1 + 4 + 1 + 2 = 15

Actions:
 0 = FOLD, 1 = CHECK/CALL, 2 = BET/RAISE (if legal; otherwise mapped to CALL)

NOTE: This is a minimal, self-contained simulator for quick iteration and BC.
It is not a full Texas Hold'em engine. For viz/real-world use, map your vision outputs
into a similar discrete-state abstraction and keep the BC pipeline identical.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

FOLD, CALL, RAISE = 0, 1, 2

RANKS = ['J', 'Q', 'K']
RANK_TO_IDX = {r:i for i,r in enumerate(RANKS)}

@dataclass
class Obs:
    state: np.ndarray  # (15,)
    legal_mask: np.ndarray  # (3,) binary mask for [FOLD,CALL,RAISE]

from typing import Optional, Tuple, Dict, Any

class LeducEnv:
    def __init__(self, seed: Optional[int]=None):
        self.rng = random.Random(seed)
        self.n_players = 2
        self.bet_sizes = [1, 2]
        self.max_raises_per_round = 2  # classic cap
        self.reset()
        self.pot: float = 0.0              # ← pot 用 float
        

    def _fresh_deck(self):
        deck = RANKS * 2  # two of each rank
        self.rng.shuffle(deck)
        return deck

    def reset(self, button: Optional[int]=None):
        self.deck = self._fresh_deck()
        # Private cards
        self.private = [self.deck.pop(), self.deck.pop()]
        # No public yet
        self.public: Optional[str] = None
        # Blinds/Antes: use simple forced bet from button as ante to seed pot
        self.button = self.rng.randrange(2) if button is None else button
        self.to_act = self.button  # button acts first pre-turn in Leduc
        self.pot = 0
        self.stacks: list[float] = [100.0, 100.0]
        self.pot = 0.0                      # ← float
       

        # Betting state
        self.round_idx = 0  # 0=pre-turn, 1=post-turn
        self.bets_this_round = 0
        self.raises_used = 0  # total raises executed in current round
        self.last_aggressive = None  # who last bet/raise
        self.opponent_acted = [False, False]
        self.terminated = False
        self.winner: Optional[int] = None
        self.folded = [False, False]

        # Each round starts with zero to call. CHECK allowed initially.
        return self._make_obs()

    def _deal_public(self):
        # Burn one (optional), then deal public rank
        self.public = self.deck.pop()

    def _legal_actions(self) -> np.ndarray:
        mask = np.ones(3, dtype=np.float32)
        # FOLD always legal; CALL always legal (CHECK if no bet)
        # RAISE legal only if raises cap not reached
        if self.raises_used >= self.max_raises_per_round:
            mask[RAISE] = 0.0
        return mask

    def _encode_state(self, player: int) -> np.ndarray:
        # private one-hot (3)
        priv = np.zeros(3, dtype=np.float32)
        priv[RANK_TO_IDX[self.private[player]]] = 1.0
        # public one-hot (4)
        pub = np.zeros(4, dtype=np.float32)
        if self.public is None:
            pub[0] = 1.0
        else:
            pub[1 + RANK_TO_IDX[self.public]] = 1.0
        # position (1)
        pos = np.array([1.0 if player == self.button else 0.0], dtype=np.float32)
        # betting features (4): bets_this_round one-hot (3) + opponent_acted flag
        btr = np.zeros(3, dtype=np.float32)
        btr[min(self.bets_this_round,2)] = 1.0
        opp_flag = np.array([1.0 if self.opponent_acted[1-player] else 0.0], dtype=np.float32)
        # pot normalized (1)
        potn = np.array([self.pot/10.0], dtype=np.float32)
        # round one-hot (2)
        rnd = np.zeros(2, dtype=np.float32)
        rnd[self.round_idx] = 1.0
        return np.concatenate([priv, pub, pos, btr, opp_flag, potn, rnd], axis=0)

    def _make_obs(self) -> Obs:
        s = self._encode_state(self.to_act)
        mask = self._legal_actions()
        return Obs(state=s, legal_mask=mask)

    def _hand_strength(self, player: int) -> tuple[int, int]:
        """
        Simple ranking for showdown:
        - Pair with public (i.e., same rank) > high card
        - If both pair, higher rank wins (K>Q>J)
        - If high card only, compare private rank
        - Ties split pot
        This is the standard Leduc evaluation.
        """
        priv_rank = RANK_TO_IDX[self.private[player]]
        pub_rank = -1 if self.public is None else RANK_TO_IDX[self.public]
        pair = (pub_rank == priv_rank)
        return (2 if pair else 1, priv_rank)  # (pair flag 2/1, rank index)


    def _resolve_showdown(self):
        s0 = self._hand_strength(0)
        s1 = self._hand_strength(1)
        if s0 > s1:
            self.winner = 0
        elif s1 > s0:
            self.winner = 1
        else:
            self.winner = None  # split

    def _award_pot(self):
        if self.winner is None:
            # split
            self.stacks[0] += self.pot/2
            self.stacks[1] += self.pot/2
        else:
            self.stacks[self.winner] += self.pot
        self.pot = 0.0

    def _end_round_or_deal(self):
        # Move to next betting round or showdown
        if self.round_idx == 0:
            self.round_idx = 1
            self.bets_this_round = 0
            self.raises_used = 0
            self.last_aggressive = None
            self.opponent_acted = [False, False]
            self._deal_public()
            # Next round: first to act is button again in Leduc
            self.to_act = self.button
        else:
            # showdown
            self._resolve_showdown()
            self._award_pot()
            self.terminated = True

    def step(self, action: int) -> Tuple[Obs, float, bool, Dict[str, Any]]:
        assert not self.terminated, "Call reset() before step after terminal."

        reward: float = 0.0
        done: bool = False
        info: Dict[str, Any] = {}

        player = self.to_act
        opp = 1 - player
        mask = self._legal_actions()
        if mask[action] == 0 and action == RAISE:
            action = CALL  # map illegal raise to call

        bet_size = self.bet_sizes[self.round_idx]
        info = {}

        if action == FOLD:
            self.folded[player] = True
            self.winner = opp
            self._award_pot()
            self.terminated = True
            reward = -1.0 * bet_size  # optional shaping
            done = True
            term_mask = np.zeros_like(mask)
            return Obs(self._encode_state(self.to_act), term_mask), reward, done, info


        elif action == CALL:
            self.opponent_acted[player] = True

            if self.last_aggressive is None:
                # 没有过往加注：这次是“CHECK”
                if self.opponent_acted[opp]:
                    # 对手也已经行动过（双方都CHECK）→ 结束本轮下注
                    self._end_round_or_deal()
                    done = self.terminated
                    reward = 0.0
                else:
                    # 轮到对手行动
                    self.to_act = opp
                    reward = 0.0
                    done = False
            else:
                # 面对对手的加注：本次是“CALL”，匹配下注并结束本轮
                self.pot += bet_size
                self._end_round_or_deal()
                done = self.terminated
                reward = 0.0


        elif action == RAISE:
            # Aggressive action
            self.raises_used += 1
            self.bets_this_round += 1
            self.last_aggressive = player
            self.opponent_acted[player] = True
            self.pot += float(bet_size)
            self.to_act = opp
            reward = 0.0
            done = False

        # 统一出口
        if self.terminated:
            term_mask = np.zeros_like(mask)
            obs = Obs(self._encode_state(self.to_act), term_mask)
        else:
            obs = self._make_obs()

        return obs, reward, done, info
    

# ---------------- Expert Policy (rule-based) ----------------
class ExpertPolicy:
    """
    A simple heuristic expert for Leduc:
    - Pre-turn (no public):
        * Raise with K often, Q sometimes, J rarely
        * If opponent raised: fold J, call Q, 3-bet K if raises remain
    - Post-turn (with public):
        * If paired (private == public): raise/cap
        * If unpaired: check/call with K, mix call/fold with Q, fold J to aggression
    This is *not* game-theoretic optimal, but gives reasonable structure for BC.
    """
    def __init__(self, rng: Optional[random.Random]=None):
        self.rng = rng or random.Random()

    def act(self, env: LeducEnv, player: int) -> int:
        priv = env.private[player]
        priv_i = RANK_TO_IDX[priv]
        pub = env.public
        btr = env.bets_this_round
        raises_left = env.max_raises_per_round - env.raises_used
        last_agg = env.last_aggressive

        def can_raise():
            return raises_left > 0

        # Round 0: no public card
        if env.round_idx == 0:
            if last_agg is None:
                if priv == 'K' and can_raise():
                    return RAISE
                if priv == 'Q' and can_raise() and self.rng.random() < 0.4:
                    return RAISE
                return CALL  # check
            else:
                # Facing aggression
                if priv == 'K':
                    if can_raise() and self.rng.random() < 0.5:
                        return RAISE
                    return CALL
                if priv == 'Q':
                    return CALL
                # priv == 'J'
                return FOLD if self.rng.random() < 0.8 else CALL

        # Round 1: public card present
        paired = (pub == priv)
        if paired:
            if last_agg is None:
                return RAISE if can_raise() else CALL
            else:
                return RAISE if can_raise() else CALL
        else:
            if priv == 'K':
                if last_agg is None:
                    return CALL  # check
                else:
                    return CALL
            if priv == 'Q':
                if last_agg is None:
                    return CALL
                else:
                    return FOLD if self.rng.random() < 0.3 else CALL
            if priv == 'J':
                if last_agg is None:
                    return CALL  # check
                else:
                    return FOLD
            return CALL

# ---------------- Rollout & Dataset ----------------

def play_hand(env: LeducEnv, expert: ExpertPolicy) -> List[Dict]:
    """Play one hand with expert on both sides, return trajectory for the button player.
    Records (s, a) with legal mask for Behavior Cloning.
    """
    traj = []
    obs = env.reset()
    while True:
        p = env.to_act
        a = expert.act(env, p)
        traj.append({
            'state': obs.state.copy(),
            'legal': obs.legal_mask.copy(),
            'action': a,
            'round': env.round_idx,
        })
        obs, _, done, _ = env.step(a)
        if done:
            break
    return traj

def rollout(n_hands: int, seed: int=42) -> List[Dict]:
    env = LeducEnv(seed)
    expert = ExpertPolicy(random.Random(seed))
    data: List[Dict] = []
    for _ in range(n_hands):
        traj = play_hand(env, expert)
        data.extend(traj)
    return data

def make_bc_dataset(n_hands: int=5000, out_path: str="dataset.npz", seed: int=42):
    data = rollout(n_hands=n_hands, seed=seed)
    states = np.stack([d['state'] for d in data], axis=0)
    legal = np.stack([d['legal'] for d in data], axis=0)
    actions = np.array([d['action'] for d in data], dtype=np.int64)
    np.savez(out_path, states=states, legal=legal, actions=actions)
    print(f"Saved BC dataset to {out_path} :: states={states.shape}, actions={actions.shape}")

if __name__ == "__main__":
    # Example: generate 10k hands for BC
    make_bc_dataset(n_hands=10000, out_path="dataset.npz", seed=7)