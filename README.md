# Trading Strategy Project

This project aims to develop and test a trading strategy using a 2D Convolutional Neural Network (CNN) to analyze 1-minute OHLCV (Open, High, Low, Close, Volume) data from Coinbase. The goal is to predict profitable trading signals and make informed buy/sell decisions.

## Folder Structure

```
Trading_Strategy
├── Components
│   ├── checkpoints
│   │   └── model_checkpoint.keras
│   ├── path
│   │   └── to
│   │       └── logfile.log
│   ├── __pycache__
│   │   ├── model_builder.cpython-39.pyc
│   │   ├── predict.cpython-39.pyc
│   │   └── preprocessing.cpython-39.pyc
│   ├── candlestick_model.keras
│   ├── data_ingestion.py
│   ├── model_builder.py
│   ├── model_evaluation.py
│   ├── predict.py
│   ├── predictions.csv
│   ├── preprocessing.py
│   ├── train.py
│   └── trained_model.keras
├── Config
│   ├── __pycache__
│   │   └── config_loader.cpython-39.pyc
│   ├── config.yaml
│   └── config_loader.py
├── Constants
├── Data
│   ├── Preprocessed
│   │   └── Preprocessed.csv
│   └── Raw
│       └── Rawbtc_ohlcv_jan2023_to_sep2024.csv
├── Entity
├── Logging
│   ├── logging
│   ├── logs
│   │   └── project.log
│   ├── logging.yaml
│   └── logging_config.py
├── Notebooks
│   ├── .ipynb_checkpoints
│   │   ├── Exploratory Data Analysis-checkpoint.ipynb
│   │   ├── model_experimentation-checkpoint.ipynb
│   │   └── Untitled-checkpoint.ipynb
│   ├── Exploratory Data Analysis.ipynb
│   ├── model_experimentation.ipynb
│   └── Untitled.ipynb
├── Pipeline
├── Tests
│   ├── __pycache__
│   │   └── test_config_loader.cpython-39.pyc
│   └── test_config_loader.py
├── Utils
│   ├── data_preprocessing.py
│   └── model_helpers.py
├── .gitignore
├── main.py
├── README.md
└── requirements.txt

```

### **Directory Descriptions**

- **Components/**: Contains modules for specific tasks like data loading, preprocessing, model training, and evaluation.
- **Config/**: Configuration files including `config.yaml` for project settings and hyperparameters.
- **Constants/**: Files for storing constant values such as fixed hyperparameters and thresholds.
- **Entity/**: Data classes or schemas for defining the structure of important objects.
- **Logging/**: Custom logging configurations and scripts for tracking execution and metrics.
- **Pipeline/**: Scripts for end-to-end data processing, model training, and evaluation pipelines.
- **Tests/**: Unit tests to ensure the correctness of various components.
- **Utils/**: General utility functions and helper scripts.
- **.gitignore**: Specifies files and directories to be ignored by Git.
- **README.md**: This file.
- **requirements.txt**: List of project dependencies.

### **Configuration**

The configuration settings are stored in `Config/config.yaml`. This file includes paths, model parameters, and training configurations.

#### Example `config.yaml`

```yaml
data:
  data_path: 'data/ohlcv_data.csv'
  frequency: '1min'
  validation_split: 0.2

keras:
  model:
    cnn:
      num_filters: [16, 32, 64]
      kernel_size: [3, 3]
      pool_size: [2, 2]
      dropout_rate: 0.3
      activation: 'relu'
      output_activation: 'softmax'
  training:
    learning_rate: 0.001
    batch_size: 16
    epochs: 30
    optimizer: 'adam'
    loss_function: 'categorical_crossentropy'
```
