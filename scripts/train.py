#!/usr/bin/env python
# scripts/train.py

import os
import random
import yaml
import numpy as np
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from data.loader import DataLoader
from env.a_share_env import AShareEnv
from utils.logger import setup_logger

class NaNRewardStopper(BaseCallback):
    """
    Callback to stop training if a NaN or Inf reward is encountered across any environment.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        rewards = self.locals.get('rewards', None)
        if rewards is None:
            return True
        rewards_arr = np.array(rewards)
        if np.any(np.isnan(rewards_arr)) or np.any(np.isinf(rewards_arr)):
            if self.verbose > 0:
                print(f"[NaNRewardStopper] Stopping training: encountered rewards={rewards_arr}")
            return False
        return True


def main():
    # 1. 加载配置
    base_dir = os.path.dirname(os.path.dirname(__file__))
    cfg_path = os.path.join(base_dir, 'config', 'default.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 2. 设置随机种子
    seed = cfg['train'].get('seed', 0)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 初始化 Logger（只调用一次）
    logger = setup_logger('train', cfg['train']['log_file'])  # initialize logger once
    logger.info(f"Random seed set to {seed}")
    logger.info('Configuration loaded from %s', cfg_path)
    logger.info(f"Random seed set to {seed}")
    logger.info('Configuration loaded from %s', cfg_path)

    # 3. 数据加载与特征工程
    loader = DataLoader(
        token=cfg['tushare_token'],
        start_date=cfg['data']['start_date'],
        end_date=cfg['data']['end_date'],
        tickers=cfg['data']['tickers'],
        time_frame=cfg['data']['time_frame']
    )
    raw_df = loader.fetch()
    logger.info('Raw data rows: %d', len(raw_df))
    df = loader.feature_engineer(raw_df)
    logger.info('Processed data rows: %d', len(df))

    # 4. 划分训练/测试集
    test_ratio = cfg['data'].get('test_ratio', 0.2)
    split_idx = int(len(df) * (1 - test_ratio))
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    logger.info('Train rows: %d, Test rows: %d', len(df_train), len(df_test))

    # 5. 并行环境数量
    n_envs = cfg['train'].get('n_envs', 4)
    logger.info('Using %d parallel environments', n_envs)

    # 6. 构造并行训练环境
    def make_env(rank):
        def _init():
            env = AShareEnv(df_train, cfg['env'])
            return Monitor(env)
        return _init
    envs = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # 7. 构造动作噪声
    noise = NormalActionNoise(
        mean=np.zeros(envs.action_space.shape),
        sigma=cfg['model']['action_noise_sigma'] * np.ones(envs.action_space.shape)
    )

    # 8. 初始化模型
    model = DDPG(
        'MlpPolicy',
        envs,
        gamma=cfg['model']['gamma'],
        learning_rate=cfg['model']['learning_rate'],
        buffer_size=cfg['model']['buffer_size'],
        batch_size=cfg['model']['batch_size'],
        tau=cfg['model']['tau'],
        action_noise=noise,
        verbose=1
    )

    # 确保目录存在
    os.makedirs(cfg['train']['checkpoint_dir'], exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)

    # 9. 回调配置
    checkpoint_cb = CheckpointCallback(
        save_freq=cfg['train']['save_freq'],
        save_path=cfg['train']['checkpoint_dir'],
        name_prefix='rl'
    )
    nan_cb = NaNRewardStopper(verbose=1)

    # 10. 开始训练
    logger.info('Starting training for %d timesteps', cfg['train']['total_timesteps'])
    model.learn(
        total_timesteps=cfg['train']['total_timesteps'],
        callback=[checkpoint_cb, nan_cb],
        log_interval=4
    )

    # 11. 保存最终模型
    final_path = os.path.join(base_dir, 'models', 'final_model.zip')
    model.save(final_path)
    logger.info('Training completed, model saved to %s', final_path)

if __name__ == '__main__':
    main()
