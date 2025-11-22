import pandas as pd
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.preprocessing_helper import clean_data, feature_engineering
from Config.config_loader import load_config

def preprocess_data(file_path):
    #Load the dataset
    df = pd.read_csv(file_path)
    
    #Clean Data
    df = clean_data(df)

    #Convert ''Open Time' and 'Close Time' to datetime
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')

    # Set the 'Open Time' as the index
    df.set_index('Open Time', inplace=True)

    # Feature engineering and label generation
    df = feature_engineering(df)

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN values generated during calculations
    df.dropna(inplace=True)

    return df

config = load_config()
processed_df = preprocess_data(config['data']['raw_data'])
print("Raw data has been loaded")
print(processed_df.info())
processed_df.to_csv(r"C:/Users/arunm/Documents/Projects/Trading-App/Data/Preprocessed/Preprocessed.csv", index=False)