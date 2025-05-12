#!/usr/bin/env python
# scripts/train.py

import os
import yaml
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from data.loader import DataLoader
from env.a_share_env import AShareEnv
from utils.logger import setup_logger

class NaNRewardStopper(BaseCallback):
    """
    Callback to stop training if a NaN or Inf reward is encountered.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Check last reward
        reward = self.locals.get('rewards') or []
        if reward and (np.isnan(reward[-1]) or np.isinf(reward[-1])):
            if self.verbose > 0:
                print(f"[NaNRewardStopper] Stopping training: encountered reward={reward[-1]}")
            return False
        return True


def main():
    # 1. 加载配置
    base_dir = os.path.dirname(os.path.dirname(__file__))
    cfg_path = os.path.join(base_dir, 'config', 'default.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 2. 日志初始化
    logger = setup_logger('train', cfg['train']['log_file'])
    logger.info('Configuration loaded from %s', cfg_path)

    # 3. 数据加载与预处理
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

    # 4. 构建环境并包装
    env = AShareEnv(df, cfg['env'])
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # 5. 构造动作噪声
    noise = NormalActionNoise(
        mean=np.zeros(env.action_space.shape),
        sigma=cfg['model']['action_noise_sigma'] * np.ones(env.action_space.shape)
    )

    # 6. 初始化模型
    model = DDPG(
        'MlpPolicy',
        env,
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

    # 7. 检查点回调
    checkpoint_cb = CheckpointCallback(
        save_freq=cfg['train']['save_freq'],
        save_path=cfg['train']['checkpoint_dir'],
        name_prefix='rl'
    )

    # 8. 配置 NaN 停止回调
    nan_cb = NaNRewardStopper(verbose=1)

    # 9. 开始训练，log_interval=1 每个 episode 打印日志
    logger.info('Starting training')
    model.learn(
        total_timesteps=cfg['train']['total_timesteps'],
        callback=[checkpoint_cb, nan_cb],
        log_interval=1
    )

    # 10. 保存最终模型
    final_path = os.path.join(base_dir, 'models', 'final_model.zip')
    model.save(final_path)
    logger.info('Training completed, model saved to %s', final_path)

if __name__ == '__main__':
    main()
