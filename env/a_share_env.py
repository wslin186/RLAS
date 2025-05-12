import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class AShareEnv(gym.Env):
    """
    Gymnasium environment for A-share trading with:
      - Z-score feature normalization
      - No-short, no-leverage trading constraints
      - Commission & slippage costs
      - Episode reward logging for Monitor compatibility
      - Support random start for robust evaluation
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data: pd.DataFrame, cfg: dict):
        super().__init__()
        # 1) 保留数值列
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        raw = data[numeric_cols].reset_index(drop=True)

        # 2) Z-score 归一化参数
        self.feature_means = raw.mean()
        self.feature_stds = raw.std().replace(0, 1.0)

        # 3) 存原始 raw_data，用于 price 和归一化
        self.raw_data = raw.copy()

        # 环境参数
        self.window_size = cfg['window_size']
        self.initial_cash = float(cfg['initial_cash'])
        self.commission = cfg['commission']
        self.slippage = cfg['slippage']

        # 动作与观测空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        n_feats = raw.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, n_feats + 2),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None, random_start=False):
        """
        重置环境状态。
        如果 random_start=True，则随机选择起始点；
        否则从固定窗口后位置开始。
        返回 obs, info（Gymnasium API）。
        """
        super().reset(seed=seed)
        # 随机起点或固定起点
        if random_start:
            self.current_step = np.random.randint(self.window_size, len(self.raw_data))
        else:
            self.current_step = self.window_size
        # 初始化资金和持仓
        self.cash = self.initial_cash
        self.shares = 0.0
        self.total_value = self.cash
        self._ep_reward = 0.0

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        w = self.raw_data.iloc[
            self.current_step - self.window_size: self.current_step
        ]
        norm = (w - self.feature_means) / self.feature_stds
        obs_vals = norm.values.astype(np.float32)

        cash_pct = self.cash / self.initial_cash
        price = float(self.raw_data.loc[self.current_step - 1, 'close'])
        pos_pct = (self.shares * price) / self.initial_cash

        cash_col = np.full((self.window_size, 1), cash_pct, dtype=np.float32)
        pos_col = np.full((self.window_size, 1), pos_pct, dtype=np.float32)

        return np.hstack([obs_vals, cash_col, pos_col])

    def step(self, action):
        """
        执行动作，更新状态，返回 obs, reward, terminated, truncated, info
        """
        # 1) 解包并限制 action
        pct = float(action[0])
        pct = np.clip(pct, -0.999, 0.999)

        # 2) 当前价格与持仓价值
        price = float(self.raw_data.loc[self.current_step, 'close'])
        prev_total = self.total_value
        current_pos_value = self.shares * price

        # 3) 目标持仓变化
        target_value = prev_total * pct
        delta_value = target_value - current_pos_value
        shares_delta = delta_value / price

        # 4) 无做空、无杠杆限制
        if shares_delta > 0:
            max_buy = self.cash / (price * (1 + self.commission + self.slippage))
            shares_delta = min(shares_delta, max_buy)
        else:
            shares_delta = max(shares_delta, -self.shares)

        # 5) 执行交易
        trade_amount = shares_delta * price
        cost = abs(trade_amount) * (self.commission + self.slippage)
        self.cash -= trade_amount + cost
        self.shares += shares_delta

        # 6) 计算 reward
        self.total_value = self.cash + self.shares * price
        reward = (self.total_value - prev_total) / (prev_total + 1e-8)
        self._ep_reward += reward

        # 7) 推进
        self.current_step += 1
        done = self.current_step >= len(self.raw_data)
        terminated = done
        truncated = False

        # 8) 生成 obs
        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)

        # 9) episode 信息
        info = {}
        if done:
            ep_len = self.current_step - self.window_size
            info = {'episode': {'r': float(self._ep_reward), 'l': ep_len}}

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f"Step {self.current_step}: Total Value {self.total_value:.2f}")
