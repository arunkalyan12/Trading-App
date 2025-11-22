import os
import sys
import pandas as pd
import numpy as np
import joblib

# Import components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.data_ingestion_helper import ingest_data
from old.preprocessing_helper import feature_engineering
from Components.preprocessing import generate_labels
from Config.config_loader import load_config
from Logging.logging_config import setup_logging
from old.backtest_helper import backtest


def main():
    logger = None

    try:
        # Load configuration
        print('Config is being loaded...')
        config = load_config()
        print('Config loaded successfully.')

        # Setup logging
        print('Setting up logging...')
        logger = setup_logging(config['logging']['log_file_path'])
        logger.info("Pipeline execution started.")
        print('Logging setup complete.')

        # Step 1: Data Ingestion
        print('Starting data ingestion...')
        symbol = config['data']['symbol']
        interval = config['data']['interval']
        start_time = int(pd.Timestamp(config['backtesting']['start_date']).timestamp() * 1000)
        end_time = int(pd.Timestamp(config['backtesting']['end_date']).timestamp() * 1000)
        save_path = r'C:/Users/arunm/Documents/Projects/Trading-App/Data/Raw/Pipeline_raw.csv'

        # Fetch OHLCV data using the ingest_data function
        df = ingest_data(symbol, interval, start_time, end_time, save_path)

        # Convert 'Open Time' to datetime format
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')

        # Log the actual date range in the ingested data
        logger.info(f"Data date range: {df['Open Time'].min()} to {df['Open Time'].max()}")
        logger.info("Data ingestion completed successfully.")

        # Step 2: Preprocessing
        preprocessed_df = feature_engineering(df)
        preprocessed_df.dropna(inplace=True)
        logger.info("Data preprocessing completed successfully.")

        # Replace infinite values with NaN
        preprocessed_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Generate labels
        preprocessed_df['Labels'] = generate_labels(preprocessed_df, window_size=3600, alpha=0.01, beta=0.02)   #3600
        preprocessed_df.dropna(inplace=True)

        # Step 3: Load trained model
        model_path = config['model']['save_path']
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Step 4: Prepare test data for prediction
        print('Preparing test data...')
        features = config['data']['features']

        # Select only the features from the config file
        X_test = preprocessed_df[features]

        # Drop 'Open' and 'Close Time' columns if they exist
        X_test = X_test.drop(columns=['Open', 'Close Time'], errors='ignore')

        # Set up the target variable (Labels)
        y_test = preprocessed_df[config['data']['label']]
        print('Test data preparation complete.')

        # Step 5: Get predictions from the model
        y_pred = model.predict(X_test)
        logger.info("Model prediction completed successfully.")
        print('Prediction complete.')

        # Step 6: Backtesting - Evaluate the model's performance using actual trading metrics
        if config['backtesting']['enabled']:
            # Check if there is data available for backtesting
            if not preprocessed_df.empty:
                final_balance = backtest(preprocessed_df, y_pred, config, logger)
                logger.info(f"Backtesting completed successfully. Final balance: {final_balance}")
            else:
                logger.error("No data available for backtesting.")
                print("No data available for backtesting.")

        # Step 7: Evaluate general performance
        evaluate_performance(y_test, y_pred, logger)

    except Exception as e:
        if logger:
            logger.error(f"Error occurred during pipeline execution: {e}")
        else:
            print(f"An error occurred (logger not available): {e}")


def evaluate_performance(y_true, y_pred, logger):
    """
    Evaluate model performance.
    Logs the accuracy and other relevant metrics.
    """
    accuracy = (y_true == y_pred).mean()
    logger.info(f"Accuracy: {accuracy:.4f}")
    # Add more metrics like precision, recall, etc., as needed


if __name__ == "__main__":
    main()
