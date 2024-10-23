# tests/test_config_loader.py

import unittest
import os
from Config.config_loader import load_config

class TestConfigLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a sample config file for testing."""
        cls.test_config_path = 'config/test_config.yaml'
        cls.sample_config = {
            'data': {
                'data_path': 'data/ohlcv_data.csv',
                'frequency': '1min',
                'validation_split': 0.2
            },
            'keras': {
                'model': {
                    'cnn': {
                        'num_filters': [16, 32, 64],
                        'kernel_size': [3, 3],
                        'pool_size': [2, 2],
                        'dropout_rate': 0.3,
                        'activation': 'relu',
                        'output_activation': 'softmax'
                    }
                },
                'training': {
                    'learning_rate': 0.001,
                    'batch_size': 16,
                    'epochs': 30,
                    'optimizer': 'adam',
                    'loss_function': 'categorical_crossentropy'
                }
            }
        }

        # Write sample config to a file
        with open(cls.test_config_path, 'w') as file:
            import yaml
            yaml.dump(cls.sample_config, file)

    @classmethod
    def tearDownClass(cls):
        """Remove the test config file after tests are done."""
        if os.path.exists(cls.test_config_path):
            os.remove(cls.test_config_path)

    def test_load_config(self):
        """Test if the configuration is loaded correctly."""
        config = load_config(config_path=self.test_config_path)
        self.assertEqual(config['data']['data_path'], 'data/ohlcv_data.csv')
        self.assertEqual(config['keras']['model']['cnn']['activation'], 'relu')
        self.assertEqual(config['keras']['training']['epochs'], 30)

if __name__ == '__main__':
    unittest.main()
