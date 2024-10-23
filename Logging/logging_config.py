import logging
import logging.config
import yaml
import os


def setup_logging(config_path='logging.yaml'):
    """
    Set up logging configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the log directory exists
    config_dir = os.path.dirname(config_path)

    # Load YAML configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract log directory from config if specified
    log_file_path = config['handlers']['file']['filename']
    log_dir = os.path.dirname(log_file_path)

    # Create log directory if it does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure logging
    logging.config.dictConfig(config)

    # Return the root logger
    return logging.getLogger()


# Example usage of the logger
if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Logging setup complete.")
