# test_env.py
import yaml, os
import numpy as np
from data.loader import DataLoader
from env.a_share_env import AShareEnv

# 加载配置
cfg = yaml.safe_load(open(os.path.join("config","default.yaml"), "r", encoding="utf-8"))

# 只取一只票、一点步数
loader = DataLoader(
    token=cfg['tushare_token'],
    start_date=cfg['data']['start_date'],
    end_date=cfg['data']['end_date'],
    tickers=cfg['data']['tickers'],
    time_frame=cfg['data']['time_frame']
)
df = loader.feature_engineer(loader.fetch())
env = AShareEnv(df, cfg['env'])

obs = env.reset()
done = False
step = 0
while not done:
    obs, rew, done, info = env.step([0.0])
    step += 1
    if done:
        print(f"✅ Done at step {step}, episode info = {info['episode']}")
        break
