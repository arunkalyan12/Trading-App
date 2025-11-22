import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data by replacing infinities with NaN and dropping rows with NaN."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    return df


def generate_labels(df: pd.DataFrame, horizon=5, atr_mult=1.5, threshold = 0.2) -> pd.DataFrame:
    df = df.copy()
    df['future_close'] = df['Close'].shift(-horizon)
    df['future_ret'] = (df['future_close'] - df['Close']) / df['Close']

    # Consensus of indicators
    consensus = (
        0.25 * np.sign(df.get('ema_diff_9_21', 0)) +
        0.20 * np.sign(df.get('macd_hist', 0)) +
        0.15 * np.sign(df.get('rsi_norm', 0)) +
        0.15 * df.get('supertrend_dir', 0) +
        0.10 * np.sign(df.get('cci_norm', 0)) +
        0.10 * np.sign(df.get('obv_slope', 0)) +
        0.05 * np.sign(df.get('mfi_norm', 0))
    )
    df['consensus_score'] = consensus.clip(-1,1)

    atr_threshold = df['atr_pct'].rolling(20).mean() * atr_mult
    df['label_prob'] = np.tanh(df['future_ret'] / (atr_threshold + 1e-6)) * df['consensus_score']

    buy_cond = df['label_prob'] > threshold
    sell_cond = df['label_prob'] < -threshold
    df['label'] = 0
    df.loc[buy_cond, 'label'] = 1
    df.loc[sell_cond, 'label'] = -1

    df.dropna(subset=['label_prob'], inplace=True)
    return df



def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: pandas DataFrame with columns ['open','high','low','close','volume']
    Returns: df with features for Core 12 indicators ready for ML/backtesting
    """

    df = df.copy()

    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    # # --------------------
    # # TREND INDICATORS
    # # --------------------
    # # EMA
    # df['ema9'] = ta.ema(df['Close'], length=9)
    # df['ema21'] = ta.ema(df['Close'], length=21)
    # df['ema50'] = ta.ema(df['Close'], length=50)
    # df['ema_diff_21_50'] = df['ema21'] - df['ema50']          # feature: trend slope
    # df['ema_diff_9_21'] = df['ema9'] - df['ema21']
    #
    # # HMA
    # df['hma14'] = ta.hma(df['Close'], length=14)
    # df['hma_slope'] = df['hma14'].diff()
    #
    # # SuperTrend
    # st = ta.supertrend(df['High'], df['Low'], df['Close'], length=14, multiplier=3)
    # df['supertrend'] = st[st.columns[0]]
    # df['supertrend_dir'] = st[st.columns[1]]
    # df['supertrend_dist'] = df['Close'] - df['supertrend']
    #
    # # --------------------
    # # MOMENTUM / OSCILLATORS
    # # --------------------
    # # RSI
    # df['rsi14'] = ta.rsi(df['Close'], length=14)
    # df['rsi_norm'] = (df['rsi14'] - 50)/50                      # normalize around 0
    #
    # # Stochastic RSI
    # stoch_rsi = ta.stochrsi(df['Close'], length=14, rsi_length=14, k=3, d=3)
    # df['stoch_rsi_k'] = stoch_rsi['STOCHRSIk_14_14_3_3']
    # df['stoch_rsi_d'] = stoch_rsi['STOCHRSId_14_14_3_3']
    # df['stoch_rsi_signal'] = df['stoch_rsi_k'].apply(lambda x: 1 if x < 0.2 else -1 if x > 0.8 else 0)
    #
    # # MACD
    # macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    # df['macd'] = macd['MACD_12_26_9']
    # df['macd_signal'] = macd['MACDs_12_26_9']
    # df['macd_hist'] = df['macd'] - df['macd_signal']
    # df['macd_cross'] = df['macd_hist'].apply(lambda x: 1 if x>0 else -1)
    #
    # # CCI
    # df['cci20'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
    # df['cci_norm'] = df['cci20']/100
    #
    # # --------------------
    # # VOLATILITY / RANGE
    # # --------------------
    # # Bollinger Bands
    # bb = ta.bbands(df['Close'], length=20, std=2)
    # df['bb_upper'] = bb['BBU_20_2.0']
    # df['bb_lower'] = bb['BBL_20_2.0']
    # df['bb_pos'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    #
    # # ATR
    # df['atr14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    # df['atr_pct'] = df['atr14']/df['Close']
    #
    # # --------------------
    # # VOLUME / FLOW
    # # --------------------
    # # OBV
    # df['obv'] = ta.obv(df['Close'], df['Volume'])
    # df['obv_slope'] = df['obv'].diff()
    #
    # # MFI
    # df['mfi14'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    # df['mfi_norm'] = (df['mfi14'] - 50)/50
    #
    # # --------------------
    # # MARKET REGIME FILTER
    # # --------------------
    # df['adx14'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    # df['adx_scaled'] = df['adx14']/100
    #
    # # --------------------
    # # Label
    # # --------------------
    #
    # df = generate_labels(df, horizon=25, atr_mult=1.55, threshold=0.400)
    #
    # # --------------------
    # # CLEAN UP
    # # --------------------
    # # Drop raw columns if you only want engineered features
    # feature_cols = ["Open", "Close", 'ema_diff_21_50', 'ema_diff_9_21', 'hma_slope', 'supertrend_dir', 'supertrend_dist',
    #                 'rsi_norm', 'stoch_rsi_signal', 'stoch_rsi_k', 'stoch_rsi_d',
    #                 'macd_hist', 'macd_cross', 'macd_signal', 'cci_norm',
    #                 'bb_pos', 'atr_pct', 'obv_slope', 'mfi_norm', 'adx_scaled',
    #                 'consensus_score', 'label_prob', 'label']

    df = clean_data(df)
    return df