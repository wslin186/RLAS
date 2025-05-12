#!/usr/bin/env python
# scripts/evaluate.py

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from data.loader import DataLoader
from env.a_share_env import AShareEnv


def load_env_and_model():
    # 1. 加载配置
    base_dir = os.path.dirname(os.path.dirname(__file__))
    cfg_path = os.path.join(base_dir, 'config', 'default.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 2. 数据预处理
    loader = DataLoader(
        token=cfg['tushare_token'],
        start_date=cfg['data']['start_date'],
        end_date=cfg['data']['end_date'],
        tickers=cfg['data']['tickers'],
        time_frame=cfg['data']['time_frame']
    )
    raw_df = loader.fetch()
    df = loader.feature_engineer(raw_df)

    # 3. 划分测试集
    test_ratio = cfg['data'].get('test_ratio', 0.2)
    split_idx = int(len(df) * (1 - test_ratio))
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    print(f"Using test data rows: {len(df_test)}")

    # 4. 构造测试环境
    def make_env():
        base_env = AShareEnv(df_test, cfg['env'])
        env = Monitor(base_env)
        return env
    vec_env = DummyVecEnv([make_env])

    # 5. 模型加载
    model_path = os.path.join(base_dir, 'models', 'final_model.zip')
    model = DDPG.load(model_path, env=vec_env)
    return vec_env, model


def evaluate(env, model, n_eval=10, use_noise=False):
    episode_rewards = []
    episode_lengths = []
    equity_curves = []

    for ep in range(n_eval):
        obs = env.reset()
        done = False
        total_reward = 0.0
        curve = []
        # 重置噪声
        if use_noise and hasattr(model, 'action_noise'):
            model.action_noise.reset()

        while not done:
            # 总是用确定性策略输出基线动作
            action, _ = model.predict(obs, deterministic=True)
            # 若启用噪声，手动添加
            if use_noise and hasattr(model, 'action_noise'):
                noise = model.action_noise()  # shape (action_dim,)
                action = action + noise
                # 限幅
                low, high = env.action_space.low, env.action_space.high
                action = np.clip(action, low, high)

            # 执行一步
            obs, rewards, dones, infos = env.step(action)
            reward = rewards[0]
            done = dones[0]
            info = infos[0]

            total_reward += reward
            # 从底层 env 获取资产数据
            curve.append(env.envs[0].env.total_value)

            if done:
                ep_len = info.get('episode', {}).get('l', len(curve))
                episode_lengths.append(ep_len)
        episode_rewards.append(total_reward)
        equity_curves.append(curve)

    # 打印统计信息
    mode = 'with_noise' if use_noise else 'deterministic'
    print(f"Evaluation over {n_eval} episodes ({mode}):")
    print(f"  Average return: {np.mean(episode_rewards):.4f}")
    print(f"  Std of return:  {np.std(episode_rewards):.4f}")
    print(f"  Max return:     {np.max(episode_rewards):.4f}")
    print(f"  Min return:     {np.min(episode_rewards):.4f}")
    print(f"  Avg episode length: {np.mean(episode_lengths):.1f}")

    # 绘制第一个 episode 的资产曲线
    plt.figure(figsize=(8, 4))
    plt.plot(equity_curves[0])
    plt.title(f'Episode curve ({mode})')
    plt.xlabel('Step')
    plt.ylabel('Total Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    env, model = load_env_and_model()
    # 分别跑确定性和带噪声评估
    evaluate(env, model, n_eval=5, use_noise=False)
    evaluate(env, model, n_eval=5, use_noise=True)
