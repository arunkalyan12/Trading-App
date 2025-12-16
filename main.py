import os
import sys
import logging

# Extend import path to access Pipeline and Config modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Pipeline')))
from Pipeline.pipeline import main as pipeline_main
from Config.config_loader import load_config
from Logging.logging_config import setup_logging

# Set up a fallback logger in case config-based logging fails
fallback_logger = logging.getLogger("fallback_logger")
fallback_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
fallback_logger.addHandler(console_handler)

def main():
    """
    Main entry point for the trading system.
    This function triggers the pipeline execution.
    """
    # Start with fallback
    logger = fallback_logger

    try:
        print('Loading config...')
        config = load_config()

        # print('Setting up logging...')
        # if 'logging' in config and 'log_file_path' in config['logging']:
        #     logger = setup_logging(config['logging']['log_file_path'])
        # else:
        #     fallback_logger.warning("Logging config path not found in config. Using fallback logger.")

        # logger.info("Trading system pipeline is starting...")

        # Run the trading pipeline
        pipeline_main()

        # logger.info("Trading system pipeline completed successfully.")
        print("Trading system pipeline completed successfully.")

    except Exception as e:
        fallback_logger.error(f"An error occurred while running the trading system: {e}")
        print(f"An error occurred while running the trading system: {e}")


if __name__ == "__main__":
    main()
