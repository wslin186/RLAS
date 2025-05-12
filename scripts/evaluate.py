#!/usr/bin/env python
# scripts/evaluate.py

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG

from data.loader import DataLoader
from env.a_share_env import AShareEnv


def load_data_and_model():
    # 1. 加载配置和数据
    base_dir = os.path.dirname(os.path.dirname(__file__))
    cfg_path = os.path.join(base_dir, 'config', 'default.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    loader = DataLoader(
        token=cfg['tushare_token'],
        start_date=cfg['data']['start_date'],
        end_date=cfg['data']['end_date'],
        tickers=cfg['data']['tickers'],
        time_frame=cfg['data']['time_frame']
    )
    raw_df = loader.fetch()
    df = loader.feature_engineer(raw_df)

    # 2. 划分测试集
    test_ratio = cfg['data'].get('test_ratio', 0.2)
    split_idx = int(len(df) * (1 - test_ratio))
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    print(f"Using test data rows: {len(df_test)}")

    # 3. 加载模型
    model_path = os.path.join(base_dir, 'models', 'final_model.zip')
    model = DDPG.load(model_path)
    return df_test, model, cfg


def evaluate_random(df, model, cfg, n_eval=5, use_noise=False):
    env = AShareEnv(df, cfg['env'])
    returns = []

    for _ in range(n_eval):
        # 随机起点重置
        obs, _ = env.reset(random_start=True)
        if use_noise and hasattr(model, 'action_noise'):
            model.action_noise.reset()

        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if use_noise and hasattr(model, 'action_noise'):
                noise = model.action_noise()
                action = np.clip(action + noise,
                                 env.action_space.low,
                                 env.action_space.high)
            # 兼容 Gymnasium 5-返回值
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)

    mode = 'with_noise' if use_noise else 'deterministic'
    print(f"Random start evaluation ({mode}) over {n_eval} runs:")
    print(f"  Mean: {np.mean(returns):.4f}, Std: {np.std(returns):.4f}, Min: {np.min(returns):.4f}, Max: {np.max(returns):.4f}")


def evaluate_sliding(df, model, cfg, window_size=None, stride=None):
    if window_size is None:
        window_size = len(df)
    if stride is None:
        stride = window_size

    segments = []
    returns = []

    for start in range(0, len(df) - window_size + 1, stride):
        segment = df.iloc[start:start + window_size].reset_index(drop=True)
        env = AShareEnv(segment, cfg['env'])
        obs, _ = env.reset(random_start=False)
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # step 返回 5 项
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)
        segments.append((start, start + window_size))

    print(f"Sliding window evaluation over {len(returns)} segments:")
    print(f"  Mean: {np.mean(returns):.4f}, Std: {np.std(returns):.4f}, Min: {np.min(returns):.4f}, Max: {np.max(returns):.4f}")
    for (s, e), r in zip(segments, returns):
        print(f"  Segment {s}-{e}: Return={r:.4f}")


def plot_example_curve(df, model, cfg, random=False):
    env = AShareEnv(df, cfg['env'])
    obs, _ = env.reset(random_start=random)
    curve = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # step 返回 5 项
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        curve.append(env.total_value)
    plt.figure(figsize=(8, 4))
    title = 'Random Start' if random else 'Fixed Start'
    plt.plot(curve)
    plt.title(f'Example Equity Curve ({title})')
    plt.xlabel('Step')
    plt.ylabel('Total Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df_test, model, cfg = load_data_and_model()
    evaluate_random(df_test, model, cfg, n_eval=5, use_noise=False)
    evaluate_random(df_test, model, cfg, n_eval=5, use_noise=True)
    evaluate_sliding(df_test, model, cfg, window_size=cfg['data'].get('eval_window', len(df_test)), stride=cfg['data'].get('eval_stride', cfg['data'].get('eval_window', len(df_test))))
    plot_example_curve(df_test, model, cfg, random=False)
    plot_example_curve(df_test, model, cfg, random=True)