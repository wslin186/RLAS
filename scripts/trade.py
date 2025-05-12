import os
import time
import yaml
from stable_baselines3 import DDPG

from data.loader import DataLoader
from env.a_share_env import AShareEnv
from utils.broker_api import BrokerClient


def main():
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'prod.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    loader = DataLoader(
        token=cfg['tushare_token'],
        start_date=cfg['data']['start_date'],
        end_date=cfg['data']['end_date'],
        tickers=cfg['data']['tickers'],
        time_frame=cfg['data']['time_frame']
    )
    df = loader.feature_engineer(loader.fetch())
    env = AShareEnv(df, cfg['env'])
    model = DDPG.load(os.path.join('models', 'final_model.zip'))

    client = BrokerClient(cfg['broker'])
    client.connect()
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        volume = int(abs(action[0]) * cfg['trade']['max_volume'])
        side = 'buy' if action[0] > 0 else 'sell'
        client.order(symbol=cfg['data']['tickers'][0], volume=volume, side=side)
        obs, _, done, _ = env.step(action)
        time.sleep(cfg['trade']['interval_seconds'])
        if done:
            break

if __name__ == '__main__':
    main()