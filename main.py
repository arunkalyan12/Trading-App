import os
import sys
import logging

# Import the pipeline module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Pipeline')))
from Pipeline.pipeline import main as pipeline_main
from Config.config_loader import load_config
from Logging.logging_config import setup_logging

# Set up a basic fallback logger to capture errors during logging setup
fallback_logger = logging.getLogger("fallback_logger")
fallback_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
fallback_logger.addHandler(console_handler)


def main():
    """
    Main entry point for the trading system.
    This function triggers the pipeline execution.
    """
    logger = fallback_logger  # Use the fallback logger initially
    try:
        # Load configuration
        print('Loading config...')
        config = load_config()

        # Setup logging
        print('Setting up logging...')
        if 'logging' in config and 'config_path' in config['logging']:
            logger = setup_logging(config['logging']['config_path'])
        else:
            fallback_logger.warning("Logging config path not found in config. Using fallback logger.")

        logger.info("Trading system pipeline is starting...")

        # Execute the pipeline
        pipeline_main()  # Run the pipeline
        logger.info("Trading system pipeline completed successfully.")
        print("Trading system pipeline completed successfully.")

    except Exception as e:
        fallback_logger.error(f"An error occurred while running the trading system: {e}")
        print(f"An error occurred while running the trading system: {e}")


if __name__ == "__main__":
    main()
