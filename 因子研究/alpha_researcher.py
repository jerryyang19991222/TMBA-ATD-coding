import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from IPython.display import display

from alive_progress import alive_bar
import time
total = 100

class Handler(dict):
    def __init__(self, path: str = 'Handler_cache_max'):
        self.base_path = path
        self.raw_path = os.path.join(self.base_path, 'raw')  # 原始資料
        self.factor_path = os.path.join(self.base_path, 'factor')  # 底層因子
        
        os.makedirs(self.raw_path, exist_ok=True)
        os.makedirs(self.factor_path, exist_ok=True)

    def _get_file_path(self, key, category):
        """ 根據 key 和 category (raw/factor) 取得檔案路徑 """
        if category not in ['raw', 'factor']:
            raise ValueError("Category must be 'raw' or 'factor'")
        folder = self.raw_path if category == 'raw' else self.factor_path
        return os.path.join(folder, f'{key}.pkl')

    def __getitem__(self, key):
        """ 允許 'raw:key' 或 'factor:key' 來取得不同類型的資料 """
        if ':' not in key:
            raise KeyError(f"Invalid key format. Use 'raw:key' or 'factor:key'.")

        category, key_name = key.split(':', 1)
        file_path = self._get_file_path(key_name, category)

        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            except (pickle.PickleError, EOFError) as e:
                raise KeyError(f"Failed to load key '{key}': {e}")
        else:
            raise KeyError(f"Key '{key}' not found in '{category}'.")

    def __setitem__(self, key, value):
        """ 允許 'raw:key' 或 'factor:key' 來存取不同類型的資料 """
        if ':' not in key:
            raise KeyError(f"Invalid key format. Use 'raw:key' or 'factor:key'.")

        category, key_name = key.split(':', 1)
        file_path = self._get_file_path(key_name, category)

        with open(file_path, 'wb') as f:
            pickle.dump(value, f)

    def cash_list(self, category=None):
        """ 
        根据指定类别 ('raw' 或 'factor') 返回文件列表，若不指定，则返回全部类别的数据。
        """
        if category not in [None, 'raw', 'factor']:
            raise ValueError("Category must be 'raw' or 'factor'")

        if category is None:
            return {
                'raw': self.cash_list('raw'),
                'factor': self.cash_list('factor')
            }
        
        folder = self.raw_path if category == 'raw' else self.factor_path
        files = set(filter(lambda X: X.endswith(".pkl"), os.listdir(folder)))
        return list(map(lambda X: X[:-4], files))
    
    def combine_data(self, keys, category="raw"):
        data_list = []
        for key in keys:
            try:
                df = self[f"{category}:{key}"]
            except KeyError:
                raise KeyError(f"Key '{category}:{key}' 不存在。")
            data_list.append(df)
        combined_df = pd.concat(data_list, axis=1, keys=keys)
        return combined_df

    @property
    def info(self):
        """ 返回 raw 和 factor 的快取資料數量 """
        cache_list = self.cash_list()
        return {
            'cache_path': self.base_path,
            'raw_data_count': len(cache_list['raw']),
            'factor_data_count': len(cache_list['factor']),
        }

    def _repr_html_(self):
        """ 在 Jupyter Notebook 中顯示漂亮的 HTML 資訊 """
        html = "<table>"
        html += "<tr><th>Key</th><th>Value</th></tr>"
        for key, value in self.info.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        html += "</table>"
        return html

with alive_bar(total, manual=True, title='Handler', bar='blocks', spinner='dots') as bar:
    bar(1)  


class factor_generator:
    def __init__(self,period):
        self.period = period

    def income_momentum(self,Data_df, 方法):
        if 方法 == 'no_operating':
            return Data_df

        def seasonal_additive_trend(df):
            return df.apply(lambda x: seasonal_decompose(x, model='additive', period=self.period, extrapolate_trend='freq').trend if x.notna().sum() > self.period else x)

        def seasonal_multiplicative_trend(df):
            return df.apply(lambda x: seasonal_decompose(x, model='multiplicative', period=self.period, extrapolate_trend='freq').trend if x.notna().sum() > self.period else x)

        def holt_winters_additive_trend(df):
            return df.apply(lambda x: ExponentialSmoothing(x, trend='add', seasonal=None).fit().fittedvalues if x.notna().sum() > self.period else x)


        def holt_winters_multiplicative_trend(df):
            return df.apply(lambda x: ExponentialSmoothing(x + abs(x.min()) + 1, trend='mul', seasonal=None).fit().fittedvalues - (abs(x.min()) + 1) if x.notna().sum() > self.period else x)

        def arima_trend(df):
            return df.apply(lambda x: ARIMA(x, order=(1,1,1)).fit().fittedvalues if x.notna().sum() > self.period else x)

        operations = {
            'sue': lambda df: (df-df.shift(52))/(df-df.shift(52)).rolling(window=self.period).std(),
            'sue_flow': lambda df: ((df-df.shift(52))-((df-df.shift(52)).rolling(window=self.period).mean()))/(df-df.shift(52)).rolling(window=self.period).std(),
            'sue_ewm': lambda df: ((df-df.shift(52))-((df-df.shift(52)).ewm(span=self.period, adjust=False).mean()))/(df-df.shift(52)).rolling(window=self.period).std(),
            'sue_half': lambda df: ((df - df.shift(52)) - ((df - df.shift(52)).ewm(halflife=self.period/2, adjust=False).mean())) / (df-df.shift(52)).rolling(window=self.period).std(),
            'sue_quantile': lambda df: (df - df.shift(52)).apply(lambda x: (x - x.median()) / (x.quantile(0.75) - x.quantile(0.25))),
            '均線交叉': lambda df: (df.rolling(int(self.period)).mean()/df.rolling(int(self.period)*2).mean())-1,
            '新高因子': lambda df: (df/df.rolling(int(self.period)).max())-1,
            'sue_二次移動平均': lambda df: ((df - df.shift(52)) - df.rolling(self.period).mean().rolling(self.period).mean()) / (df-df.shift(52)).rolling(self.period).std(),
            'sue_二次指數移動平均': lambda df: ((df - df.shift(52)) - df.ewm(span=self.period, adjust=False).mean().ewm(span=self.period, adjust=False).mean()) / (df-df.shift(52)).rolling(self.period).std(),
            'sue_季節性加法': lambda df: ((df - df.shift(52)) - seasonal_additive_trend(df)) / (df-df.shift(52)).rolling(self.period).std(),
            'sue_季節性乘法': lambda df: ((df - df.shift(52)) - seasonal_multiplicative_trend(df)) / (df-df.shift(52)).rolling(self.period).std(),
            'sue_HoltWinter加法': lambda df: ((df - df.shift(52)) - holt_winters_additive_trend(df)) / (df-df.shift(52)).rolling(self.period).std(),
            'sue_HoltWinter乘法': lambda df: ((df - df.shift(52)) - holt_winters_multiplicative_trend(df)) / (df-df.shift(52)).rolling(self.period).std(),
            'sue_ARIMA': lambda df: ((df - df.shift(52)) - arima_trend(df)) / (df-df.shift(52)).rolling(self.period).std(),
        }
        
        return operations.get(方法, lambda df: df)(Data_df)
    
    def price_momentum(self,Data_df,方法):
        operations = {
            '傳統動能_ted': lambda df: (df/df.shift(int(self.period)))-1,
            '傳統動能_mean': lambda df: df.rolling(int(self.period)).mean(),
            '傳統動能_rank': lambda df: df.rolling(window=int(self.period)).rank(pct=True),
            '一年最高價比值': lambda df: df/df.rolling(window=int(self.period)).max(),
        }
        
        return operations.get(方法, lambda df: df)(Data_df)
    
    def growth_factor(self,Data_df,方法):

        def arima_trend(df):
            return df.apply(lambda x: ARIMA(x, order=(1,1,1)).fit().fittedvalues if x.notna().sum() > self.period else x)

        operations = {
            '均線成長': lambda df: (df.rolling(int(self.period)).mean()/df.shift(int(self.period)).rolling(int(self.period)).mean())-1,
            'YOY': lambda df: ((df-df.shift(int(self.period)))/abs(df.shift(int(self.period)))),
            'YOY_rolling_mean': lambda df: ((df-df.shift(int(self.period)))/abs(df.shift(int(self.period)))).rolling(int(self.period)).mean(),
            'YOY_rolling_ewm': lambda df: ((df-df.shift(int(self.period)))/abs(df.shift(int(self.period)))).ewm(span=self.period, adjust=False).mean(),
            'YOY_rolling_half': lambda df: ((df-df.shift(int(self.period)))/abs(df.shift(int(self.period)))).ewm(halflife=self.period/2, adjust=False).mean(),
            'YOY_rolling_arima': lambda df: arima_trend((df-df.shift(int(self.period)))/abs(df.shift(int(self.period)))),
        }
        
        return operations.get(方法, lambda df: df)(Data_df)

    def valuation_factor(self, 分子, 分母, 方法):###先化為每股xx

        def arima_trend(df):
            return df.apply(lambda x: ARIMA(x, order=(1,1,1)).fit().fittedvalues if x.notna().sum() > self.period else x)

        operations = {
            'PE': lambda x, y: x / y,
            'PE_rolling_mean': lambda x, y: x.rolling(int(self.period)).mean() / y,
            'PE_rolling_ewm': lambda x, y: x.ewm(span=self.period, adjust=False).mean() / y,
            'PE_rolling_half': lambda x, y: x.ewm(halflife=self.period/2, adjust=False).mean()/y,
            'PE_rolling_rank': lambda x, y: x.rolling(window=int(self.period)).rank(pct=True) / y,
            'PE_predict_arima': lambda x, y: arima_trend(x) / y,
            'PEG': lambda x, y: (x / y) / ((((x - x.shift(int(self.period))) / abs(x.shift(int(self.period)))) * 100) ),
        }
        return operations.get(方法, lambda x, y: None)(分子, 分母)

with alive_bar(total, manual=True, title='factor_generator', bar='blocks', spinner='dots') as bar:
    bar(1) 



class super_factor_generator:
    def __init__(self,period):
        self.period = period 

    def CTA(self,Data_df, 方法):
        if 方法 == 'no_operating':
            return Data_df
        def compute_rsi(df, window):
            delta = df.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        def compute_atr(df, window):
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            return true_range.rolling(window=window).mean()

        def compute_cci(df, window):
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            mean_tp = typical_price.rolling(window=window).mean()
            mean_dev = (typical_price - mean_tp).abs().rolling(window=window).mean()
            return (typical_price - mean_tp) / (0.015 * mean_dev)
        
        operations = {
            'sma': lambda df: df.rolling(window=int(self.period)).mean(),
            'ema': lambda df: df.ewm(span=int(self.period), adjust=False).mean(),
            'momentum': lambda df: df - df.shift(int(self.period)),
            'rsi': lambda df: compute_rsi(df, int(self.period)),
            'bollinger_upper': lambda df: df.rolling(window=int(self.period)).mean() + 2 * df.rolling(window=int(self.period)).std(),
            'bollinger_lower': lambda df: df.rolling(window=int(self.period)).mean() - 2 * df.rolling(window=int(self.period)).std(),
            'macd': lambda df: df.ewm(span=12, adjust=False).mean() - df.ewm(span=26, adjust=False).mean(),
            'stoch_k': lambda df: ((df - df.rolling(window=int(self.period)).min()) / (df.rolling(window=int(self.period)).max() - df.rolling(window=int(self.period)).min())) * 100,
            'williams_r': lambda df: (df.rolling(window=int(self.period)).max() - df) / (df.rolling(window=int(self.period)).max() - df.rolling(window=int(self.period)).min()) * -100,
            'roc': lambda df: (df - df.shift(int(self.period))) / df.shift(int(self.period)) * 100,
            'atr': lambda df: compute_atr(df, int(self.period)),
            'cci': lambda df: compute_cci(df, int(self.period)),
            'rank': lambda df: df.rank(axis=1, pct=True)
        }
        
        return operations.get(方法, lambda df: df)(Data_df)

    def time_seriers(self ,Data_df, 方法):
        if 方法 == 'no_operating':
            return Data_df
        operations = {
            'ts_mean': lambda df: df.rolling(window=int(self.period)).mean(),
            'ts_rank': lambda df: df.rolling(window=int(self.period)).rank(pct=True),
            'ts_sharpe': lambda df: df.rolling(window=int(self.period)).mean() / df.rolling(window=int(self.period)).std(),
            'ts_skewness': lambda df: df.rolling(window=int(self.period)).skew(),
            'ts_kurtosis': lambda df: df.rolling(window=int(self.period)).kurt(),
            'ts_delay': lambda df: df.shift(int(self.period)),
            'ts_delta': lambda df: df.diff(int(self.period)),
            'ts_min': lambda df: df.rolling(window=int(self.period)).min(),
            'ts_max': lambda df: df.rolling(window=int(self.period)).max(),
            'ts_argmin': lambda df: df.rolling(window=int(self.period)).apply(np.nanargmin, raw=True) + 1,
            'ts_argmax': lambda df: df.rolling(window=int(self.period)).apply(np.nanargmax, raw=True) + 1,
            'ts_decay_linear': lambda df: df.rolling(window=int(self.period)).apply(lambda x: np.sum(x * np.arange(1, int(self.period)+1) / np.sum(np.arange(1, int(self.period)+1))), raw=True),
            'ts_sum': lambda df: df.rolling(window=int(self.period)).sum(),
            'ts_std': lambda df: df.rolling(window=int(self.period)).std(),
            'ts_product': lambda df: df.rolling(window=int(self.period)).apply(np.prod, raw=True),
        }
        return operations.get(方法, lambda df: df)(Data_df)
    
    def cross_section(self,Data_df, 方法 ):
        if 方法 == 'no_operating':
            return Data_df
        
        operations = {
            'scale': lambda df: df / df.abs().sum() * 2,
            'cs_rank': lambda df: df.rank(axis=1),
            'cs_normalization': lambda df: (df.sub(df.min(axis=1), axis=0)).div(df.max(axis=1) - df.min(axis=1), axis=0),
            'cs_standardization': lambda df: (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0),
            'cs_scale': lambda df: df.mul(1).div(df.abs().sum(axis=1), axis='index'),
            'cs_zscore': lambda df: (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0),
            'cs_quantile': lambda df: df.rank(axis=1, pct=True),  
            'cs_clip': lambda df: df.clip(lower=df.quantile(0.01, axis=1), upper=df.quantile(0.99, axis=1), axis=0),  
            'cs_mad': lambda df: (df.sub(df.median(axis=1), axis=0)).div((df.sub(df.median(axis=1), axis=0)).abs().mean(axis=1), axis=0),  
            'cs_log': lambda df: np.log1p(df.abs()).mul(np.sign(df)),  
            'cs_minmax_by_column': lambda df: (df.sub(df.min(axis=0), axis=1)).div(df.max(axis=0) - df.min(axis=0), axis=1),  
            'cs_softmax': lambda df: np.exp(df).div(np.exp(df).sum(axis=1), axis=0), 
            'signed_power': lambda df: df ** 2,
            'abs': lambda df: df.abs(),  
            'log': lambda df: np.log1p(df),  
            'sign': lambda df: np.sign(df),
        }
        
        return operations.get(方法, lambda df: df)(Data_df)
    

    def single_factor(self,Data_df, 方法):
        if 方法 == 'no_operating':
            return Data_df
        operations = {
            'ts_mean': lambda df: df.rolling(window=int(self.period)).mean(),
            'ts_rank': lambda df: df.rolling(window=int(self.period)).rank(pct=True),
            'ts_sharpe': lambda df: df.rolling(window=int(self.period)).mean() / df.rolling(window=int(self.period)).std(),
            'ts_skewness': lambda df: df.rolling(window=int(self.period)).skew(),
            'ts_kurtosis': lambda df: df.rolling(window=int(self.period)).kurt(),
            'ts_delay': lambda df: df.shift(int(self.period)),
            'ts_delta': lambda df: df.diff(int(self.period)),
            'ts_min': lambda df: df.rolling(window=int(self.period)).min(),
            'ts_max': lambda df: df.rolling(window=int(self.period)).max(),
            'ts_argmin': lambda df: df.rolling(window=int(self.period)).apply(np.nanargmin, raw=True) + 1,
            'ts_argmax': lambda df: df.rolling(window=int(self.period)).apply(np.nanargmax, raw=True) + 1,
            'ts_decay_linear': lambda df: df.rolling(window=int(self.period)).apply(lambda x: np.sum(x * np.arange(1, int(self.period)+1) / np.sum(np.arange(1, int(self.period)+1))), raw=True),
            'ts_sum': lambda df: df.rolling(window=int(self.period)).sum(),
            'ts_std': lambda df: df.rolling(window=int(self.period)).std(),
            'ts_product': lambda df: df.rolling(window=int(self.period)).apply(np.prod, raw=True),
            'scale': lambda df: df / df.abs().sum() * 2,
            'cs_rank': lambda df: df.rank(axis=1),
            'cs_normalization': lambda df: (df.sub(df.min(axis=1), axis=0)).div(df.max(axis=1) - df.min(axis=1), axis=0),
            'cs_standardization': lambda df: (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0),
            'cs_scale': lambda df: df.mul(1).div(df.abs().sum(axis=1), axis='index'),
            'cs_zscore': lambda df: (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0),
            'cs_quantile': lambda df: df.rank(axis=1, pct=True),  
            'cs_clip': lambda df: df.clip(lower=df.quantile(0.01, axis=1), upper=df.quantile(0.99, axis=1), axis=0),  
            'cs_mad': lambda df: (df.sub(df.median(axis=1), axis=0)).div((df.sub(df.median(axis=1), axis=0)).abs().mean(axis=1), axis=0),  
            'cs_log': lambda df: np.log1p(df.abs()).mul(np.sign(df)),  
            'cs_minmax_by_column': lambda df: (df.sub(df.min(axis=0), axis=1)).div(df.max(axis=0) - df.min(axis=0), axis=1),  
            'cs_softmax': lambda df: np.exp(df).div(np.exp(df).sum(axis=1), axis=0), 
            'signed_power': lambda df: df ** 2,
            'abs': lambda df: df.abs(),  
            'log': lambda df: np.log1p(df),  
            'sign': lambda df: np.sign(df),  
        }
        return operations.get(方法, lambda df: df)(Data_df)
    
    
    def two_factor(self ,factor1: pd.DataFrame, factor2: pd.DataFrame, method: str) -> pd.DataFrame:
        if method == 'add':
            return factor1 + factor2
        elif method == 'factor1':
            return factor1 
        elif method == 'subtract':
            return factor1 - factor2
        elif method == 'multiply':
            return factor1 * factor2
        elif method == 'divide':
            return factor1.div(factor2.replace(0, np.nan))
        elif method == 'mean':
            return (factor1 + factor2) / 2
        elif method == 'weighted':
            weights1 = factor1.abs() / (factor1.abs() + factor2.abs())
            weights2 = 1 - weights1
            return factor1 * weights1 + factor2 * weights2
        elif method == 'cov':  
            return factor1.rolling(window=int(self.period)).cov(factor2)
        elif method == 'corr':  
            return factor1.rolling(window=int(self.period)).corr(factor2)
        else:
            raise ValueError("Unsupported dual-factor processing method.")

with alive_bar(total, manual=True, title='super_factor_generator', bar='blocks', spinner='dots') as bar:
    bar(1)        


class factor_prehandle:

    def __init__(self,clip):
        self.clip = clip
    
    def extreme_mad(self ,df: pd.DataFrame):
        median = df.median(axis=1)
        mad = (df.T - median).T.abs().median(axis=1)
        upper_limit = median + int(self.clip) * mad
        lower_limit = median - int(self.clip) * mad
        return df.clip(lower=lower_limit, upper=upper_limit, axis=0)
    
    def winsorize_method(self,df: pd.DataFrame, ub=None, lb=None, method='clip', quantiles=(0.01, 0.99)):
        if ub is None:
            ub = df.quantile(quantiles[1], axis=0)
        if lb is None:
            lb = df.quantile(quantiles[0], axis=0)
        if method == 'clip':
            df_filtered = df.clip(lower=lb, upper=ub, axis=0)
        elif method == 'mask':
            mask = (df < lb) | (df > ub)
            df_filtered = df.mask(mask)
        else:
            raise ValueError("Invalid method. Please use 'clip' or 'mask'")
        return df_filtered
    
    def Neutralization(self,Y: pd.DataFrame, *X_list: pd.DataFrame):
        def get_rsid(X: np.ndarray, Y: np.ndarray):
            def get_beta(X: np.ndarray, Y: np.ndarray):
                # 计算回归系数
                coefficients = np.linalg.pinv(X.T @ X) @ X.T @ Y
                return coefficients

            coefficients = get_beta(X, Y)
            predicted_Y = X @ coefficients
            rsid = Y - predicted_Y
            return rsid

        Y_stack = Y.stack()
        Y_stack.index.names = ['date', 'order_book_id']
        X = pd.concat({i: X_list[i] for i in range(len(X_list))}, axis=1).stack(dropna=False)
        X.index.names = ['date', 'order_book_id']
        
        neutralized_df = (
            pd.concat([Y_stack, X], axis=1)
            .dropna()
            .groupby('date')
            .apply(lambda data: pd.Series(get_rsid(data.iloc[:, 1:].values, data.iloc[:, 0].values), 
                                        index=data.index.get_level_values('order_book_id')))
        )
        neutralized_df.index.names = [None, 'order_book_id']
        return neutralized_df.unstack().reindex_like(Y)

    def get_industry_dummies(self,industry):
        industry_codes = industry.stack().unique().tolist()
        industry_codes = [code for code in industry_codes if isinstance(code, str)]
        encoded_industries = {code: (industry == code).astype(int) for code in industry_codes}
        industry_dummies = [encoded_industries[ind] for ind in industry_codes]
        return industry_dummies

    def Zscore(self,df: pd.DataFrame) -> pd.DataFrame:
        return (df - df.mean(axis=1, keepdims=True)) / df.std(axis=1, keepdims=True)

    def rank_pct(self,df: pd.DataFrame) -> pd.DataFrame:
        return df.rank(axis=1, pct=True)   
    
with alive_bar(total, manual=True, title='factor_prehandle', bar='blocks', spinner='dots') as bar:
    bar(1) 



class factor_merge_valuation:
    def __init__(self,factor_merge,return_df,period):
        self.factor_merge = factor_merge
        self.return_df = return_df
        self.period=period
    
    def calculate_demean_ls_returns(self , filter_condition=None,only_long:bool=False):
        columns = self.factor_merge.columns.get_level_values(0).unique().tolist()
        return_series = {}
        if filter_condition is not None:
            self.return_df = self.return_df[filter_condition]
        for column in columns:      
            demeanded = self.factor_merge[column] - np.nanmean(self.factor_merge[column],axis=1)[:,None]
            weights=demeanded/np.nansum(np.abs(demeanded),axis=1)[:,None]
            if only_long:
                weights[weights<0] = 0
                weights*=2
            return_series[column] = (weights * self.return_df).sum(axis=1)
        return_series_df = pd.DataFrame(return_series)
        return return_series_df

    def calculate_ic_series(self, filter_condition=None):
        columns = self.factor_merge.columns.get_level_values(0).unique().tolist()
        return_series = {}
        if filter_condition is not None:
            self.return_df = self.return_df[filter_condition]
        for column in columns:      
            return_series[column] = self.factor_merge[column].corrwith(self.return_df,method='spearman',axis = 1)
        return_series_df = pd.DataFrame(return_series)
        return return_series_df

    def analyze_negative_factors(self,dataframe, keyword):
        analysis_columns = dataframe.loc[:, dataframe.columns.str.contains(keyword)]
        analysis_sum = analysis_columns.sum(axis=0)
        negative_values = analysis_sum[analysis_sum < 0]
        print("Negative Values in Analysis Sum:")
        print(negative_values)
        plt.figure(figsize=(20, 10))
        analysis_sum.plot(kind='bar', color='skyblue')
        plt.title('Sum of Factors Containing: ' + keyword, fontsize=16)
        plt.xlabel('Factors', fontsize=14)
        plt.ylabel('Sum Value', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout() 
        plt.show()
        return negative_values.index.tolist()
    
    def calculate_metrics_two( self , returns , ic_series):
        if isinstance(ic_series, pd.DataFrame) and ic_series.shape[1] == 1:
            ic_series = ic_series.squeeze()
        if isinstance(returns, pd.DataFrame):
            return returns.apply(lambda col: self.calculate_metrics_week(col, ic_series[col.name] if col.name in ic_series else ic_series), axis=0)
        annual_return = round(((1 + returns.mean()) ** int(self.period) - 1) * 100, 2)
        annual_sharpe = round(np.sqrt(int(self.period)) * returns.mean() / returns.std(), 2)
        max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
        calmar_ratio = round(((annual_return / 100) / abs(max_drawdown)), 2)
        mdd = round(max_drawdown * 100, 2)
        drawdown = returns.cumsum() - returns.cumsum().cummax()
        longest_drawdown = drawdown[drawdown < 0].groupby((drawdown >= 0).cumsum()).count().max()
        longest_drawdown = round(longest_drawdown, 0)
        win_rate = round((len(returns[returns > 0]) / len(returns[returns != 0])) * 100, 2)
        ic_mean = round(ic_series.mean(), 4) if isinstance(ic_series, pd.Series) else np.nan
        ic_ir = round(ic_series.mean() / ic_series.std(), 4) if isinstance(ic_series, pd.Series) else np.nan
        return pd.Series({'年化收益率(%)': annual_return,'年化夏普比率': annual_sharpe,'卡玛比率': calmar_ratio,'MDD(%)': mdd,'最长回撤周数': longest_drawdown,'周胜率(%)': win_rate,'IC_MEAN': ic_mean,'IC_IR': ic_ir})
    

with alive_bar(total, manual=True, title='factor_merge_valuation', bar='blocks', spinner='dots') as bar:
    bar(1) 



class factor_merger:
    def __init__(self,factor_merge,demean_ls,ic_se):
        self.factor_merge = factor_merge
        self.demean_ls = demean_ls
        self.ic_se = ic_se

    def cum_icir_weighting(self):
        評分項目一 = (self.ic_se.rolling(window=52).apply(lambda x: (x > 0.03).sum() >= 26).fillna(0).astype(int))
        評分項目二 = (self.demean_ls.rolling(52).mean()>=0.003).fillna(0).astype(int)
        評分項目三 = ((self.ic_se.rolling(52).mean())/(self.ic_se.rolling(12).std())>=0.3).fillna(0).astype(int)
        評分項目四 = ((self.ic_se.rolling(52).mean())>=0.03).fillna(0).astype(int)
        綜合評分 =評分項目一 + 評分項目二 + 評分項目三 + 評分項目四 
        選定因子 = 綜合評分.rank(axis=1, ascending=False) <= 4
        hrp_weights_over_year=(((self.ic_se.rolling(52).mean())/(self.ic_se.rolling(52).std())[選定因子]).fillna(0)).rank(axis=1, pct=True)
        return hrp_weights_over_year

    def calculate_combine_factor_rolling(self, weights):
        if not isinstance(self.factor_merge.columns, pd.MultiIndex):
            raise ValueError("df必須有MultiIndex列結構")
        result = pd.DataFrame(index=self.factor_merge.index, columns=self.factor_merge.columns.get_level_values(1).unique())
        for column in result.columns:
            symbol = self.factor_merge.xs(column, axis=1, level=1, drop_level=True)
            symbol_bool = symbol.notna()
            if not weights.index.equals(symbol.index):
                weights = weights.reindex(symbol.index)
            if not weights.columns.equals(symbol.columns):
                weights = weights.reindex(columns=symbol.columns)
            symbol_weights = weights.where(symbol_bool)
            symbol_weights = symbol_weights.div(symbol_weights.sum(axis=1), axis=0)
            result_score = symbol.mul(symbol_weights)
            result[column] = result_score.sum(axis=1)
        return result.replace(0, np.nan)
    
    def get_fama_macbeth_weighting(self,data):
        t_values_dict = {}
        for date, group in data.groupby(data.index.names[0]):
            if len(group) > 1:
                X = group.iloc[:, 1:]  
                y = group.iloc[:, 0]   
                X = sm.add_constant(X) 
                mask = ~X.isin([np.nan, np.inf, -np.inf]).any(axis=1) & ~y.isin([np.nan, np.inf, -np.inf])
                X, y = X[mask], y[mask]
                if len(X) <= 1:
                    continue
                model = sm.OLS(y, X).fit()  
                t_values_dict[date] = {factor: t_value for factor, t_value in model.tvalues.items() if factor != "const"}
        t_values_df = pd.DataFrame.from_dict(t_values_dict, orient='index')
        t_values_df = t_values_df.sort_index()
        return t_values_df

with alive_bar(total, manual=True, title='factor_merger', bar='blocks', spinner='dots') as bar:
    bar(1)



class factor_valuation:
    def __init__(self,factor_df,return_df,period):
        self.factor_df = factor_df
        self.return_df = return_df
        self.period = period     

    def get_IC_Se(self):
        IC_se = self.factor_df.corrwith(self.return_df,method='spearman',axis = 1)
        return IC_se
    
    def get_ic_ir(self, periods=[1, 5, 10, 22]):
        results = {}
        for period in periods:
            if period == 1:
                future_returns = self.return_df
            else:
                future_returns = self.return_df.rolling(window=period).sum().shift(-period + 1)
            aligned_expreturns = future_returns.loc[self.factor_df.index]
            ic_values = aligned_expreturns.corrwith(self.factor_df,axis=1)
            ic_mean = ic_values.mean()
            ic_std = ic_values.std()
            ir = ic_mean / ic_std if ic_std != 0 else 0
            results[f'{period}_days'] = {'IC': ic_mean, 'IR': ir}
        return results

    def factor_to_weight(self,only_long:bool=False):
        demeanded = self.factor_df - np.nanmean(self.factor_df,axis=1)[:,None]
        weights=demeanded/np.nansum(np.abs(demeanded),axis=1)[:,None]
        if only_long:
            weights[weights<0] = 0
            weights*=2
        return weights
    
    def get_strategy_return(self,weighting, buy_fee=0.04/100, sell_fee=0.04/100,intraday = False):
        expreturns = self.return_df.loc[weighting.index[0]:weighting.index[-1]]
        returns_df = weighting * expreturns
        returns_series = returns_df.sum(axis=1)
        if intraday:
            fee_df = weighting.abs()*(buy_fee+sell_fee)
        else:
            delta_weight = weighting.diff()
            buy_fees = delta_weight[delta_weight > 0]*(buy_fee)
            buy_fees = buy_fees.fillna(0)
            sell_fees = delta_weight[delta_weight < 0].abs()*(sell_fee)
            sell_fees = sell_fees.fillna(0)
            fee_df = buy_fees + sell_fees
        all_fee_series = fee_df.sum(axis=1)
        performance_series = returns_series - all_fee_series
        return performance_series
    
    def get_quantile_return_ted(self,rank_range_n=10):
        return_series = {}
        factor = self.factor_df.dropna(axis=1,how='all').dropna(axis=0,how='all')
        expreturns = self.return_df.loc[factor.index[0]:factor.index[-1]]
        for quantile in (range(1, rank_range_n + 1)):
            signal = factor.rank(axis=1, pct=True, ascending=True)
            is_quantile = signal.rank(axis=1, pct=True) <= quantile / 10
            is_quantile &= signal.rank(axis=1, pct=True) > (quantile - 1) / 10
            qunatile_signal = signal[is_quantile].notna().astype(int)
            quantile_ew = qunatile_signal.div(qunatile_signal.sum(axis=1),axis=0)
            quantile_returns_series = self.get_strategy_return(quantile_ew, expreturns, 0, 0)
            return_series[quantile] = (quantile_returns_series-expreturns.mean(axis = 1))
        return_series_df = pd.DataFrame(return_series)
        return return_series_df
    
    def quantile_weighting(self, num_quantiles=10): 
        quantile_weights = []
        for i in range(num_quantiles):
            lower = i / num_quantiles 
            upper = (i + 1) / num_quantiles  
            threshold_lower = (self.factor_df.notna().sum(axis=1) * lower).round()
            threshold_upper = (self.factor_df.notna().sum(axis=1) * upper).round()     
            position = self.factor_df.rank(axis=1, ascending=True).gt(threshold_lower, axis=0) & \
                    self.factor_df.rank(axis=1, ascending=True).le(threshold_upper, axis=0)
            weighting = position.astype(float)
            weighting.replace({0: np.nan}, inplace=True)
            weighting = weighting.div(weighting.sum(axis=1), axis=0)
            quantile_weights.append(weighting)
        result = pd.concat(quantile_weights, keys=[f'quantile_{i}' for i in range(num_quantiles)], axis=1)
        return result
    
    def plot_quantile_distribution(self,quantile_weighting, cap,name):
        columns = quantile_weighting.columns.get_level_values(0).unique().tolist()
        return_series = {}
        for column in columns:  
            屬性值 = quantile_weighting[column] * cap
            return_series[column] = 屬性值.mean().mean()  # 计算均值
        return_series = pd.Series(return_series)
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 6))
        return_series.plot(kind='bar', color='gray', edgecolor='white')
        plt.xlabel("quantile", color='white')
        plt.ylabel("mean", color='white')
        plt.title(name, color='white')
        plt.xticks(rotation=45, color='white')
        plt.grid(axis='y', linestyle='--', alpha=0.7, color='white')
        plt.show()

    def combined_factor_returns(self, num_quantiles=10, filter=None):  
        self.return_df = self.return_df[filter if filter is not None else slice(None)]
        def calculate_taxed_return(weighting, return_data):
            ret_df = weighting * return_data  
            delta_weighting = weighting.diff()  
            buy_fee = delta_weighting[delta_weighting > 0].abs() * (0.001425 * 0.3 + 0.003)
            buy_fee = buy_fee.fillna(0)  
            sell_fee = delta_weighting[delta_weighting < 0].abs() * (0.001425 * 0.3)
            sell_fee = sell_fee.fillna(0)  
            taxed_return = ret_df - buy_fee - sell_fee
            return taxed_return.sum(axis=1) 
        quantile_returns = {}
        for i in range(num_quantiles):
            lower = i / num_quantiles  
            upper = (i + 1) / num_quantiles  
            threshold_lower = (self.factor_df.notna().sum(axis=1) * lower).round() 
            threshold_upper = (self.factor_df.notna().sum(axis=1) * upper).round() 
            position = self.factor_df.rank(axis=1, ascending=True).gt(threshold_lower, axis=0) & \
                    self.factor_df.rank(axis=1, ascending=True).le(threshold_upper, axis=0)
            weighting = position.astype(float) 
            weighting.replace({0: np.nan}, inplace=True) 
            weighting = weighting.div(weighting.sum(axis=1), axis=0) 
            taxed_return = calculate_taxed_return(weighting, self.return_df)  
            quantile_returns[f'Q{i+1}'] = taxed_return  
        long_threshold = (self.factor_df.notna().sum(axis=1) * 0.3).round() 
        short_threshold = (self.factor_df.notna().sum(axis=1) * 0.7).round() 
        long_position = self.factor_df.rank(axis=1, ascending=True).le(long_threshold, axis=0)
        short_position = self.factor_df.rank(axis=1, ascending=True).gt(short_threshold, axis=0)
        weighting = long_position.astype(float) + short_position.astype(float)  
        weighting.replace({0: np.nan}, inplace=True) 
        weighting = weighting.div(weighting.sum(axis=1), axis=0) 
        weighting[long_position] *= -1  
        fama_ls = (weighting * self.return_df).sum(axis=1)  
        demean = self.factor_df.sub(self.factor_df.mean(axis=1), axis=0)  
        weighting = demean.div(demean.abs().sum(axis=1), axis=0)  
        demean_ls = (weighting * self.return_df).sum(axis=1)  
        all_returns = pd.DataFrame(quantile_returns)  
        all_returns['30%_LS'] = fama_ls  
        all_returns['Demean_LS'] = demean_ls  
        return all_returns  

    def plot_quantile_pnl(self,quantile_return):
        plt.style.use("grayscale")
        fig, ax = plt.subplots(figsize=(16, 5))
        color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        quantile_return.cumsum().plot(ax=ax, color=color_list)
        plt.legend()
        plt.show()   
    
    def quantile_return_his(self,quantiled_return):
        compound_returns = (quantiled_return + 1).prod() -1
        days = len(quantiled_return)
        average_returns  = (compound_returns + 1) ** (252 / days) - 1
        plt.figure(figsize=(12, 6))
        average_returns.plot(kind='bar', color='grey', edgecolor='black') 
        plt.title('Quantile Average Annualized Return', fontsize=16, color='black')
        plt.xlabel('Quantile', fontsize=14, color='black')
        plt.ylabel('Mean Return (%)', fontsize=14, color='black')
        plt.xticks(rotation=45, fontsize=12, color='black')
        plt.yticks(fontsize=12, color='black')
        plt.grid(axis='y', linestyle='--', alpha=0.7, color='grey')
        plt.tight_layout()
        plt.show() 

    def quantile_sharpe_his(self,quantiled_return):
        average_returns = round(np.sqrt(int(self.period)) * quantiled_return.mean() / quantiled_return.std(), 2)
        plt.figure(figsize=(12, 6))
        average_returns.plot(kind='bar', color='grey', edgecolor='black')  
        plt.title('Quantile Sharpe Ratio', fontsize=16, color='black')
        plt.xlabel('Quantile', fontsize=14, color='black')
        plt.ylabel('Sharpe Ratio', fontsize=14, color='black')
        plt.xticks(rotation=45, fontsize=12, color='black')
        plt.yticks(fontsize=12, color='black')
        plt.grid(axis='y', linestyle='--', alpha=0.7, color='grey')
        plt.tight_layout()
        plt.show()

    def quantile_volatility_his(self,quantiled_return):
        volatility = np.sqrt(int(self.period))*quantiled_return.std()  
        plt.figure(figsize=(12, 6))
        volatility.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('quantile_volatility_his', fontsize=16)
        plt.xlabel('quantile', fontsize=14)
        plt.ylabel('volatility', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def calculate_metrics(self,returns):
        annual_return = round(((1 + returns.mean()) ** int(self.period) - 1)*100, 2)
        annual_sharpe = round(np.sqrt(int(self.period)) * returns.mean() / returns.std(), 2)
        max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
        calmar_ratio = round(((annual_return/100 )/ abs(max_drawdown)), 2)
        time_in_market = round((len(returns[returns != 0]) / len(returns))*100, 2)
        mdd = round(max_drawdown*100, 2)
        drawdown = returns.cumsum() - returns.cumsum().cummax()
        longest_drawdown = drawdown[drawdown < 0].groupby((drawdown >= 0).cumsum()).count().max()
        longest_drawdown = round(longest_drawdown,0)
        profit_factor = round(abs(returns[returns > 0].sum() / returns[returns < 0].sum()), 2)
        win_rate = round((len(returns[returns > 0]) / len(returns[returns != 0]))*100, 2)
        return pd.Series({'年化收益率(%)': annual_return,'年化夏普比率': annual_sharpe,'卡玛比率': calmar_ratio,'在市场上时间(%)': time_in_market,'MDD(%)': mdd,'最长回撤月数': longest_drawdown,'Profit Factor': profit_factor,'胜率(%)': win_rate})
    
    
    def calculate_metrics_ted(self,returns, demean_ls_w):
        cumprod_return=round(((1+returns).prod()-1)*100,2)
        cumulative_return =  round(returns.sum()*100,2)
        compound_returns = (returns + 1).prod() -1
        days = len(returns)
        annual_return  = (compound_returns + 1) ** (int(self.period)/ days) - 1
        annual_sharpe = round(np.sqrt(int(self.period)) * returns.mean() / returns.std(), 2)
        max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
        calmar_ratio = round((annual_return / 100) / abs(max_drawdown), 2)
        time_in_market = round((len(returns[returns != 0]) / len(returns)) * 100, 2)
        mdd = round(max_drawdown * 100, 2)
        drawdown = returns.cumsum() - returns.cumsum().cummax()
        longest_drawdown = drawdown[drawdown < 0].groupby((drawdown >= 0).cumsum()).count().max()
        longest_drawdown = round(longest_drawdown, 0)
        profit_factor = round(abs(returns[returns > 0].sum() / returns[returns < 0].sum()) , 2)
        win_rate = round((len(returns[returns > 0]) / len(returns[returns != 0])) * 100, 2)
        volatility = round(returns.std() * np.sqrt(int(self.period)) * 100, 2)  
        std = round(returns.std() * 100, 2) 
        demean_ls_w_delta=demean_ls_w.fillna(0).diff()
        demean_ls_w_delta=abs(demean_ls_w_delta).sum(axis=1)
        avg_turnover=demean_ls_w_delta.sum() / len(demean_ls_w_delta)
        performance_metrics = {
            'Prod Ret': [f"{cumprod_return:.2f} %"],
            'Sum Ret': [f"{cumulative_return:.2f} %"],
            'CAGR': [f"{annual_return:.2f} %"],
            'Sharpe': [f"{annual_sharpe:.2f}"],
            'Carmar':[f"{calmar_ratio:.2f}"],
            'MDD': [f"{mdd:.2f} %"],
            'Max MDD period': [f"{longest_drawdown:.2f}"],
            'Volatility': [f"{volatility:.2f} %"],
            'STD': [f"{std:.2f} %"],
            'Win Rate': [f"{win_rate:.2f} %"],
            'Profit factor': [f"{profit_factor:.2f} "],
            'Turnover': [f"{avg_turnover*100:.2f} %"],
            'Time in market': [f"{time_in_market:.2f} %"],
            }
        print(f'StartTime: {returns.index[0]}, EndTime: {returns.index[-1]}')
        performance_df = pd.DataFrame(performance_metrics, index=['Perform'])
        display(performance_df)
        return performance_df


    def plot_quantile_return(combined_factor_returns):
        plt.style.use("grayscale")
        fig, ax = plt.subplots(figsize=(9, 5))
        color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        combined_factor_returns.loc['2016-02-1':'2024-11-26'].cumsum().plot(ax=ax, color=color_list)
        plt.legend()
        plt.show()


    def operate_fama_macbeth(self,data):
        t_values_dict = {}
        for date, group in data.groupby(data.index.names[0]):
            if len(group) > 1:
                X = group.iloc[:, 1:]  
                y = group.iloc[:, 0]   
                X = sm.add_constant(X) 
                mask = ~X.isin([np.nan, np.inf, -np.inf]).any(axis=1) & ~y.isin([np.nan, np.inf, -np.inf])
                X, y = X[mask], y[mask]
                if len(X) <= 1:
                    continue
                model = sm.OLS(y, X).fit()  
                for factor, t_value in model.tvalues.items():
                    if factor != "const":  
                        if factor not in t_values_dict:
                            t_values_dict[factor] = []
                        t_values_dict[factor].append(t_value)
        t_mean_dict = {factor: sum(t_list) / len(t_list) for factor, t_list in t_values_dict.items() if len(t_list) > 0}
        print("各因子的 T 值平均：")
        for factor, t_mean in t_mean_dict.items():
            print(f"{factor}: {t_mean:.4f}")
    

    def get_quantile_return_every_period(self,adj_open_df,rank_range_n=10,period_list=['D','M','W']):
        def get_strategy_return(weighting, buy_fee=0.04/100, sell_fee=0.04/100,intraday = False):
            expreturns = return_df.loc[weighting.index[0]:weighting.index[-1]]
            returns_df = weighting * expreturns
            returns_series = returns_df.sum(axis=1)
            if intraday:
                fee_df = weighting.abs()*(buy_fee+sell_fee)
            else:
                delta_weight = weighting.diff()
                buy_fees = delta_weight[delta_weight > 0]*(buy_fee)
                buy_fees = buy_fees.fillna(0)
                sell_fees = delta_weight[delta_weight < 0].abs()*(sell_fee)
                sell_fees = sell_fees.fillna(0)
                fee_df = buy_fees + sell_fees
            all_fee_series = fee_df.sum(axis=1)
            performance_series = returns_series - all_fee_series
            return performance_series
        def resample_to_period(df, period):
            df['datetime'] = df.index
            df = df.groupby(df.index.to_period(period)).apply(lambda x: x.iloc[-1])
            df.index = df['datetime']
            df = df.drop(columns=['datetime'])
            return df
        period_results = {}
        for period in period_list:
            adj_open_df_shift=adj_open_df.shift(-1)
            adj_open_df_resample=resample_to_period(adj_open_df_shift, period=period)
            return_df=(adj_open_df_resample.shift(-1)/adj_open_df_resample)-1
            factor_df_resample=resample_to_period(self.factor_df , period=period)
            return_series = {}
            factor = factor_df_resample.dropna(axis=1,how='all').dropna(axis=0,how='all')
            expreturns = return_df.loc[factor.index[0]:factor.index[-1]]
            for quantile in (range(1, rank_range_n + 1)):
                signal = factor.rank(axis=1, pct=True, ascending=True)
                is_quantile = signal.rank(axis=1, pct=True) <= quantile / 10
                is_quantile &= signal.rank(axis=1, pct=True) > (quantile - 1) / 10
                qunatile_signal = signal[is_quantile].notna().astype(int)
                quantile_ew = qunatile_signal.div(qunatile_signal.sum(axis=1),axis=0)
                quantile_returns_series = get_strategy_return(quantile_ew, expreturns,0.001425*0.25,0.001425*0.25+0.003)
                return_series[quantile] = quantile_returns_series
            return_series_df = pd.DataFrame(return_series)
            period_results[period] = return_series_df

        return period_results




with alive_bar(total, manual=True, title='factor_valuation', bar='blocks', spinner='dots') as bar:
    bar(1) 



class macro_economic:
    def __init__(self,fred_list):
        self.fred_list=fred_list

    def get_fred_data(self,指標='CPIAUCSL'):###用手機app查詢
        from fredapi import Fred
        fred = Fred(api_key='0dbf8e62c3b8642348ae1295c1d9e210')  
        data = fred.get_series('CPIAUCSL')
        return data
    
    def get_fred_data_multiple(self):
        dfs = []
        for item in self.fred_list:
            fred_data = self.get_fred_data(指標=item)
            fred_data.name = item
            dfs.append(fred_data)
        final_df = pd.concat(dfs, axis=1, join='outer')
        final_df.fillna(method='ffill', inplace=True)

        return final_df
    
    def get_yfinance_data(self):
        import yfinance as yf
        data_frames = []
        for ticker in self.fred_list:
            stock = yf.Ticker(ticker)
            data = stock.history(period='max')
            data.columns = pd.MultiIndex.from_product([data.columns, [ticker]])
            data_frames.append(data)
        combined_data = pd.concat(data_frames, axis=1)
        return combined_data
    
    def get_wbdata(self,indicators,countries):
        import wbdata
        ###資料庫來源：https://www.shihang.org/ext/zh/home
        ###使用說明書：https://datahelpdesk.worldbank.org/knowledgebase/articles/898599-indicator-api-queries
        df = wbdata.get_dataframe(indicators, country=countries)
        return df
        

with alive_bar(total, manual=True, title='macro_economic', bar='blocks', spinner='dots') as bar:
    bar(1) 