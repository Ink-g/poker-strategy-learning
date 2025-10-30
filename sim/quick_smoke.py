# quick_smoke.py
import random
import numpy as np

# 如果你的常量在别处定义，保留这三行映射；否则注释掉自己 import
FOLD, CALL, RAISE = 0, 1, 2

def random_policy(mask: np.ndarray) -> int:
    legal = np.where(mask > 0.5)[0]
    return int(random.choice(legal))

def rollout_once(env_cls, policy_fn, seed=0):
    random.seed(seed)
    env = env_cls(seed=seed)
    obs = env.reset()
    steps = 0

    while True:
        assert hasattr(obs, "state") and hasattr(obs, "legal_mask"), "Obs 缺少必要字段"
        assert isinstance(obs.legal_mask, np.ndarray) and obs.legal_mask.shape == (3,), "legal_mask 应为形状 (3,) 的 ndarray"
        assert (obs.legal_mask >= 0).all(), "legal_mask 不应为负"
        assert env.pot >= 0.0, "奖池不能为负"

        action = policy_fn(obs.legal_mask)
        obs, reward, done, info = env.step(action)
        steps += 1

        assert isinstance(reward, float), "reward 必须是 float"
        assert isinstance(done, bool), "done 必须是 bool"
        assert isinstance(info, dict), "info 必须是 dict"

        if done:
            # 终局：mask 通常应为全 0（如果你这么实现了）
            # 若你保留合法动作也没关系，测试仅做提示不强制
            try:
                assert (obs.legal_mask <= 1).all()
            except Exception:
                pass
            break

        assert steps < 200, "保护阈：步数异常，请检查状态转移逻辑"

    # 资金守恒（初始 100+100 + 期间 pot=0 结束）：允许 1e-6 漂移
    total = sum(env.stacks) + env.pot
    assert abs(total - 200.0) < 1e-6, f"资金不守恒：{total}"

    print(f"✅ 冒烟通过：{steps} 步结束，stacks={env.stacks}, pot={env.pot}")

if __name__ == "__main__":
    # 把这里 import 改成你实际类的路径
    from leduc_env import LeducEnv  # ← 修改为真实模块
    rollout_once(LeducEnv, random_policy, seed=123)
