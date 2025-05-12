import pandas as pd
import tushare as ts

class DataLoader:
    """
    负责从 Tushare 获取指定 A 股列表的数据，并进行分组特征工程。
    """
    def __init__(self, token: str, start_date: str, end_date: str, tickers: list, time_frame: str = 'D'):
        self.pro = ts.pro_api(token)
        self.start_date = start_date
        self.end_date = end_date
        self.time_frame = time_frame
        self.tickers = tickers

    def fetch(self) -> pd.DataFrame:
        """
        获取 config 中 tickers 列表的日线数据，返回 DataFrame，列：['ts_code','date','open','high','low','close','volume']。
        """
        dfs = []
        for ts_code in self.tickers:
            df = self.pro.daily(ts_code=ts_code, start_date=self.start_date, end_date=self.end_date)
            if df is None or df.empty:
                continue
            df = df[['ts_code','trade_date','open','high','low','close','vol']]
            dfs.append(df)
        data = pd.concat(dfs, ignore_index=True)
        data.rename(columns={'vol':'volume','trade_date':'date'}, inplace=True)
        data['date'] = pd.to_datetime(data['date'])
        data.sort_values(['ts_code','date'], inplace=True)
        return data

    def feature_engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        按 ts_code 分组计算技术指标：MA、STD、Upper、Lower。
        返回带特征的 DataFrame。
        """
        def _compute(group):
            window = 20
            group['MA'] = group['close'].rolling(window).mean()
            group['STD'] = group['close'].rolling(window).std()
            group['Upper'] = group['MA'] + 2 * group['STD']
            group['Lower'] = group['MA'] - 2 * group['STD']
            return group.dropna()

        feat = data.groupby('ts_code').apply(_compute).reset_index(drop=True)
        return feat