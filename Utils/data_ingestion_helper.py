import pandas as pd
import time
import requests


def ingest_data(symbol, interval, start_time, end_time, save_path):
    """
    Ingest OHLCV data from the Binance API.

    Parameters:
    - symbol: Trading pair symbol (e.g., 'BTCUSDT').
    - interval: Time interval for the data (e.g., '1m' for 1 minute).
    - start_time: Start time in milliseconds since epoch.
    - end_time: End time in milliseconds since epoch.
    - save_path: Path to save the CSV file containing the fetched data.

    Returns:
    - DataFrame containing OHLCV data.
    """

    def fetch_ohlcv(symbol, interval, start_time, end_time=None, limit=1000):
        """
        Fetch OHLCV data from the Binance API.
        """
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        try:
            response = requests.get(url, params=params)
            if response.status_code == 429:
                print("Rate limit exceeded. Sleeping for 30 seconds...")
                time.sleep(30)
                return fetch_ohlcv(symbol, interval, start_time, end_time, limit)
            response.raise_for_status()
            data = response.json()
            if not data:
                return pd.DataFrame()
            return pd.DataFrame(data, columns=[
                'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
                'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Volume',
                'Taker Buy Quote Volume', 'Ignore'
            ])
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return pd.DataFrame()

    # Prepare the loop to fetch data in chunks
    all_data = []
    while start_time < end_time:
        df = fetch_ohlcv(symbol, interval, start_time, end_time)
        if df.empty:
            print("No data fetched, ending loop.")
            break
        all_data.append(df)
        start_time = int(df.iloc[-1]['Close Time']) + 1
        print(f"Fetched {len(df)} records. New start time: {start_time}")
        time.sleep(1)

    # Combine and save the data
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(save_path, index=False)
        print(f"Data successfully saved to {save_path}.")
        return final_df
    else:
        print("No data fetched. CSV file not created.")
        return pd.DataFrame()
