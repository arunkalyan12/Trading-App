import requests
import pandas as pd
import time

def fetch_ohlcv(symbol, interval, start_time, end_time=None, limit=1000):
    """
    Fetch OHLCV data from the Binance API.

    Parameters:
    - symbol: Trading pair symbol (e.g., 'BTCUSDT').
    - interval: Time interval for the data (e.g., '1m' for 1 minute).
    - start_time: Start time in milliseconds since epoch.
    - end_time: End time in milliseconds since epoch (optional).
    - limit: Number of data points to retrieve per request (default is 1000).

    Returns:
    - DataFrame containing OHLCV data.
    """
    url = "https://api.binance.com/api/v3/klines"  # API endpoint for fetching OHLCV data
    params = {
        'symbol': symbol,  # Trading pair symbol
        'interval': interval,  # Time interval for the data
        'startTime': start_time,  # Start time in milliseconds
        'endTime': end_time,  # End time in milliseconds (optional)
        'limit': limit  # Number of data points to retrieve
    }
    try:
        response = requests.get(url, params=params)  # Send GET request to the API
        if response.status_code == 429:
            # Handle rate limit exceeded (HTTP 429 Too Many Requests)
            print("Rate limit exceeded. Sleeping for 30 seconds...")
            time.sleep(30)  # Sleep for 30 seconds before retrying
            return fetch_ohlcv(symbol, interval, start_time, end_time, limit)  # Retry the request
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        data = response.json()  # Parse JSON response
        if not data:
            return pd.DataFrame()  # Return an empty DataFrame if no data is returned
        # Convert the response data into a DataFrame
        return pd.DataFrame(data, columns=[
            'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
            'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Volume',
            'Taker Buy Quote Volume', 'Ignore'
        ])
    except requests.exceptions.RequestException as e:
        # Handle any issues with the request (e.g., network problems, invalid responses)
        print(f"Request failed: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

symbol = 'BTCUSDT'  # Trading pair symbol
interval = '1m'  # Time interval (1 minute)
start_time = int(pd.Timestamp('2023-01-01').timestamp() * 1000)  # Convert start date to milliseconds
end_time = int(pd.Timestamp('2024-09-12').timestamp() * 1000)  # Convert end date to milliseconds
all_data = []  # List to store dataframes

while start_time < end_time:
    df = fetch_ohlcv(symbol, interval, start_time, end_time)  # Fetch OHLCV data
    if df.empty:
        # Break the loop if no data is returned or if the API response is empty
        print("No data fetched, ending loop.")
        break
    all_data.append(df)  # Append the dataframe to the list
    # Update the start_time with the last 'Close Time' in the data to fetch the next chunk
    start_time = int(df.iloc[-1]['Close Time']) + 1
    print(f"Fetched {len(df)} records. New start time: {start_time}")
    time.sleep(1)  # Sleep to avoid hitting API rate limits

# Combine all data into one DataFrame if any data was fetched
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)  # Concatenate all dataframes into a single DataFrame
    print(final_df)

    # Save the combined DataFrame to a CSV file
    final_df.to_csv(r'C:/Users/arunm/Documents/Projects/Trading-App/Data/Raw/Rawbtc_ohlcv_jan2023_to_sep2024.csv', index=False)
    print("Data successfully saved to CSV.")
else:
    # Print a message if no data was fetched
    print("No data fetched. CSV file not created.")
