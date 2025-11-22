import pandas as pd
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.preprocessing_helper import clean_data, rsi, macd, bollinger_bands, atr, stochastic_oscillator, \
    feature_engineering
from Config.config_loader import load_config


def generate_labels(data, window_size=120, alpha=0.01, beta=0.02):
    """
    Generate trading labels based on forward and backward price movements.

    Parameters:
    - data (pd.DataFrame): DataFrame containing 'Close' prices and 'Timestamp'.
    - window_size (int): Time window size in seconds to look forward and backward.
    - alpha (float): Threshold for buying (price above moving average).
    - beta (float): Threshold for selling (price below moving average).

    Returns:
    - pd.Series: Series containing trading labels.
    """
    labels = []

    for i in range(len(data)):
        if i < window_size or i + window_size >= len(data):
            labels.append(0)  # Not enough data for the forward/backward window
            continue

        current_price = data['Close'].iloc[i]

        # Look backward within the window size
        backward_prices = data['Close'].iloc[i - window_size:i]
        # Look forward within the window size
        forward_prices = data['Close'].iloc[i + 1:i + window_size + 1]

        # Calculate the average prices in the backward and forward windows
        backward_avg = backward_prices.mean()
        forward_avg = forward_prices.mean()

        # Label generation based on thresholds
        if current_price > backward_avg * (1 + alpha) and current_price > forward_avg * (1 + alpha):
            labels.append(1)  # Buy
        elif current_price < backward_avg * (1 - beta) and current_price < forward_avg * (1 - beta):
            labels.append(-1)  # Sell
        else:
            labels.append(0)  # Hold

    return pd.Series(labels, index=data.index)


def preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Clean the data
    df = clean_data(df)

    # Convert 'Open Time' and 'Close Time' to datetime
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')

    # Set the 'Open Time' as the index
    df.set_index('Open Time', inplace=True)

    # Feature engineering
    df = feature_engineering(df)

    # Generate labels
    df['Labels'] = generate_labels(df, window_size=3600, alpha=0.01, beta=0.02)

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN values generated during calculations
    df.dropna(inplace=True)

    # Drop unwanted columns and update the DataFrame
    df.drop(['Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Volume', 'Taker Buy Quote Volume', 'Ignore'],
            axis='columns', inplace=True)

    return df


config = load_config()
processed_df = preprocess_data(config['data']['raw_data'])
print("Raw data has been loaded")
print(processed_df.info())
processed_df.to_csv(r"C:/Users/arunm/Documents/Projects/Trading-App/Data/Preprocessed/Preprocessed.csv", index=False)
