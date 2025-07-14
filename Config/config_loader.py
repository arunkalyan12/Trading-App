import yaml
import os


def load_config(config_path=r'C:/Users/arunm/Documents/Projects/Trading-App/Config/config.yaml'):
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config
