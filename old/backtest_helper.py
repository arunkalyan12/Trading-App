import pandas as pd
import numpy as np
import logging


def initialize_logger(log_path='backtest.log'):
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger()


def calculate_position_size(balance, risk_percentage):
    """
    Calculate position size based on balance and risk percentage.
    """
    return balance * (risk_percentage / 100)


def apply_stop_loss_take_profit(entry_price, current_price, stop_loss, take_profit):
    """
    Check if the stop loss or take profit levels are hit.
    Returns True if stop loss or take profit are triggered.
    """
    if current_price <= entry_price * (1 - stop_loss / 100):
        return 'stop_loss'
    elif current_price >= entry_price * (1 + take_profit / 100):
        return 'take_profit'
    return None


def backtest(df, y_pred, config, logger):
    """
    Perform backtesting using predicted signals and returns the final balance.
    - Buy when prediction = 1
    - Sell when prediction = -1
    - Hold when prediction = 0
    """

    initial_balance = config['backtesting']['initial_balance']
    balance = initial_balance
    position = 0  # No position at the start
    entry_price = 0
    risk_percentage = config['risk_management']['risk_percentage']
    stop_loss = config['risk_management']['stop_loss']
    take_profit = config['risk_management']['take_profit']

    logger.info(f"Starting backtest from {config['backtesting']['start_date']} to {config['backtesting']['end_date']}")

    # Add predicted signals to the DataFrame
    df['Predicted_Signal'] = y_pred

    # Loop through each record and apply trading logic
    for idx, row in df.iterrows():
        signal = row['Predicted_Signal']
        current_price = row['Close']

        if position == 0:  # If no active position
            if signal == 1:  # Buy signal
                position_size = calculate_position_size(balance, risk_percentage)
                position = position_size / current_price  # Amount of the asset bought
                balance -= position_size
                entry_price = current_price
                logger.info(f"BUY: Bought {position} units at {current_price}, balance: {balance}")
            elif signal == -1:
                logger.info(f"HOLD: No active position and signal is SELL, waiting for buy signal.")

        elif position > 0:  # If in a long position
            # Check stop-loss and take-profit
            sl_tp_trigger = apply_stop_loss_take_profit(entry_price, current_price, stop_loss, take_profit)

            if signal == -1 or sl_tp_trigger == 'take_profit' or sl_tp_trigger == 'stop_loss':
                balance += position * current_price
                profit_loss = (current_price - entry_price) * position
                logger.info(f"SELL: Sold at {current_price}, PnL: {profit_loss}, balance: {balance}")
                position = 0  # Close position
            else:
                logger.info(f"HOLD: Holding position, current price: {current_price}, entry price: {entry_price}")

    final_balance = balance + position * df.iloc[-1]['Close'] if position > 0 else balance
    logger.info(f"Backtesting completed. Final balance: {final_balance}")

    return final_balance
