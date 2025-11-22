import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data by replacing infinities with NaN and dropping rows with NaN."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    return df

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI)."""
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(df: pd.DataFrame) -> pd.Series:
    """Calculate the Moving Average Convergence Divergence (MACD)."""
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    return ema_12 - ema_26

def bollinger_bands(df: pd.DataFrame, window: int = 20):
    """Calculate Bollinger Bands."""
    sma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

def stochastic_oscillator(df: pd.DataFrame, period: int = 14):
    """Calculate the Stochastic Oscillator."""
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    return 100 * (df['Close'] - low_min) / (high_max - low_min)

def atr(df: pd.DataFrame, period: int = 14):
    """Calculate the Average True Range (ATR)."""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = high_low.combine(high_close, max).combine(low_close, max)
    return true_range.rolling(window=period).mean()

def is_doji(df: pd.DataFrame) -> pd.Series:
    """Identify Doji candlestick pattern."""
    high_low_range = df['High'] - df['Low']
    return abs(df['Open'] - df['Close']) / high_low_range < 0.1

def is_hammer(df: pd.DataFrame) -> pd.Series:
    """Identify Hammer candlestick pattern."""
    high_low_range = df['High'] - df['Low']
    return ((df['Close'] > df['Open']) &
            ((df['High'] - df['Close']) > 2 * (df['Close'] - df['Open'])))

def is_shooting_star(df: pd.DataFrame) -> pd.Series:
    """Identify Shooting Star candlestick pattern."""
    return ((df['Close'] < df['Open']) &
            ((df['Open'] - df['Close']) > 2 * (df['High'] - df['Open'])))

def is_harami(df: pd.DataFrame) -> pd.Series:
    """Identify Harami candlestick pattern."""
    return ((df['Close'] < df['Open']) &
            (df['Close'].shift(1) > df['Open'].shift(1)) &
            (df['Open'] > df['Close'].shift(1)) &
            (df['Close'] < df['Open'].shift(1)))

def is_engulfing(df: pd.DataFrame) -> pd.Series:
    """Identify Engulfing candlestick pattern."""
    return ((df['Close'] > df['Open']) &
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Close'] > df['Open'].shift(1)) &
            (df['Open'] < df['Close'].shift(1)))


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering to generate technical indicators and candlestick patterns."""

    # Ensure 'Close' and other columns are numeric for calculations
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    # Moving Averages, Volatility, RSI, and returns
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Volatility'] = df['Close'].rolling(window=50).std()
    df['returns'] = df['Close'].pct_change()
    df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
    df['RSI'] = rsi(df['Close'])

    # Drop rows with NaN values (especially from rolling windows)
    df.dropna(inplace=True)

    # Technical Indicators
    df['MACD'] = macd(df)
    df['ATR'] = atr(df)
    df['Bollinger_Upper'], df['Bollinger_Lower'] = bollinger_bands(df)
    df['Stochastic_Oscillator'] = stochastic_oscillator(df)

    # Candlestick Patterns
    df['Doji_Pattern'] = is_doji(df).astype(int)
    df['Hammer_Pattern'] = is_hammer(df).astype(int)
    df['Engulfing_Pattern'] = is_engulfing(df).astype(int)
    df['Shooting_Star_Pattern'] = is_shooting_star(df).astype(int)
    df['Harami_Pattern'] = is_harami(df).astype(int)

    # OBV (On-Balance Volume)
    df['OBV'] = ta.obv(df['Close'], df['Volume'])

    # Candle metrics
    df['Body_Size'] = abs(df['Close'] - df['Open'])
    df['Upper_Wick_Size'] = df['High'] - np.maximum(df['Close'], df['Open'])
    df['Lower_Wick_Size'] = np.minimum(df['Close'], df['Open']) - df['Low']

    # Ensure numeric columns and drop rows with NaN values again
    df['Body_Size'] = pd.to_numeric(df['Body_Size'], errors='coerce')
    df['Upper_Wick_Size'] = pd.to_numeric(df['Upper_Wick_Size'], errors='coerce')
    df['Lower_Wick_Size'] = pd.to_numeric(df['Lower_Wick_Size'], errors='coerce')

    # Drop any NaN rows before further calculations
    df.dropna(inplace=True)

    # Calculate Body_to_Wick_Ratio after ensuring numeric types
    df['Body_to_Wick_Ratio'] = df['Body_Size'] / (df['Upper_Wick_Size'] + df['Lower_Wick_Size'])

    print(df.info())

    return df


def generate_labels(df: pd.DataFrame, future_period: int = 1) -> pd.Series:
    """Generate labels based on future price movement."""
    future_close = df['Close'].shift(-future_period)
    return (future_close > df['Close']).astype(int)

def scale_data(df: pd.DataFrame, columns: list, feature_range: tuple = (0, 1)) -> pd.DataFrame:
    """Scale specified columns using Min-Max scaling."""
    scaler = MinMaxScaler(feature_range=feature_range)
    df[columns] = scaler.fit_transform(df[columns])
    return df
