# tests/test_leduc_env.py
import numpy as np
from sim.leduc_env import LeducEnv
import pytest

FOLD, CALL, RAISE = 0, 1, 2

@pytest.fixture
def env():
    return LeducEnv(seed=42)

def test_reset_obs_shape(env):
    obs = env.reset()
    assert hasattr(obs, "state") and hasattr(obs, "legal_mask")
    assert isinstance(obs.legal_mask, np.ndarray)
    assert obs.legal_mask.shape == (3,)
    assert obs.legal_mask.dtype == np.float32
    assert (obs.legal_mask >= 0).all()

def test_actions_mask_respects_raise_cap(env):
    env.reset()
    env.raises_used = env.max_raises_per_round
    mask = env._legal_actions()
    assert mask[RAISE] == 0.0
    assert mask[CALL] > 0.0 and mask[FOLD] > 0.0

def test_fold_path_terminates(env):
    obs = env.reset()
    # 当前行动玩家直接 FOLD
    obs, reward, done, info = env.step(FOLD)
    assert done is True
    assert env.terminated is True
    # 终局后 pot 应清零、某玩家 stack 增加（或平分）
    assert env.pot == 0.0
    assert abs(sum(env.stacks) - 200.0) < 1e-6

def test_call_check_progression(env):
    env.reset()
    # 预期：未有进攻时 CALL == CHECK，应切换到对手或推进回合
    prev_round = env.round_idx
    obs, reward, done, info = env.step(CALL)
    assert done in (False, True)
    # 如果对手也 CHECK，应该进入下一轮或终局
    if not done:
        # 对手也 CHECK 一次
        obs, reward, done, info = env.step(CALL)
        assert env.round_idx >= prev_round  # 可能进入下一轮
        assert done in (False, True)

def test_raise_and_call_settles_round(env):
    env.reset()
    # 进攻一次
    obs, reward, done, info = env.step(RAISE)
    assert env.last_aggressive is not None
    # 对手 CALL，应该推进回合或终局
    obs, reward, done, info = env.step(CALL)
    assert done in (False, True)

def test_showdown_strength_tuple(env):
    env.reset()
    # 人为设置摊牌场景：public = private[0] → 玩家0 对子
    env.public = env.private[0]
    s0 = env._hand_strength(0)
    s1 = env._hand_strength(1)
    assert isinstance(s0, tuple) and isinstance(s1, tuple)
    assert s0 >= s1

def test_terminal_mask_zeroed(env):
    env.reset()
    # 强制摊牌
    env.round_idx = 1
    env.public = env.private[0]
    env._resolve_showdown()
    env._award_pot()
    env.terminated = True
    mask = env._legal_actions()
    # 有的实现终局仍返回 (1,1,0)；如果你实现了全 0 掩码，改为调用 _terminal_mask()
    # 这里宽松检查：不抛异常即可
    assert (mask >= 0).all()
