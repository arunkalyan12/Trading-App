# Trading Strategy Project

This trading system is designed to streamline and automate the analysis and execution of trading strategies on financial markets. Built with advanced machine learning and deep learning components, it incorporates a robust pipeline for data ingestion, preprocessing, model training, backtesting, and evaluation. The project is modular and scalable, allowing easy adaptation for various financial assets and strategies. With support for logging, configuration management, and model checkpoints, the system is both maintainable and production-ready.

## Folder Structure

```
Trading_System
├── checkpoints
│   ├── bidirectional_lstm_checkpoint.keras
│   ├── cnn_checkpoint.keras
│   ├── cnn_lstm_checkpoint.keras
│   ├── cnn_lstm_ensemble_checkpoint.keras
│   ├── cnn_only_checkpoint.keras
│   ├── deep_cnn_checkpoint.keras
│   ├── gru_only_checkpoint.keras
│   ├── lstm_only_checkpoint.keras
│   ├── simple_cnn_checkpoint.keras
│   └── stacked_lstm_checkpoint.keras
├── Components
│   ├── Model Testing
│   │   ├── Logs
│   │   │   ├── model_performance.txt
│   │   │   └── model_testing.log
│   │   ├── __pycache__
│   │   │   └── model_experiment.cpython-39.pyc
│   │   ├── model_builder (ann version).py
│   │   ├── model_performance_colab.txt
│   │   └── model_tester (ml version).py
│   ├── path
│   │   └── to
│   │       └── logfile.log
│   ├── __pycache__
│   │   ├── data_loader.cpython-39.pyc
│   │   ├── model_builder.cpython-39.pyc
│   │   └── preprocessing.cpython-39.pyc
│   ├── best_extra_trees_params.txt
│   ├── data_ingestion.py
│   ├── model_builder.py
│   ├── preprocessing.py
│   └── train.py
├── Config
│   ├── __pycache__
│   │   └── config_loader.cpython-39.pyc
│   ├── config.yaml
│   └── config_loader.py
├── Data
│   ├── Preprocessed
│   │   └── Preprocessed.csv
│   └── Raw
│       ├── Pipeline_raw.csv
│       └── Rawbtc_ohlcv_jan2023_to_sep2024.csv
├── Logging
│   ├── logs
│   │   └── project.log
│   ├── __pycache__
│   │   └── logging_config.cpython-39.pyc
│   ├── logging.yaml
│   └── logging_config.py
├── Logs
│   ├── modeling.log
│   ├── model_performance.txt
│   └── project.log
├── Models
│   └── trained_model.pkl
├── Notebooks
│   ├── .ipynb_checkpoints
│   │   ├── Exploratory Data Analysis-checkpoint.ipynb
│   │   ├── model_experimentation-checkpoint.ipynb
│   │   └── Preprocessed_EDA-checkpoint.ipynb
│   ├── Exploratory Data Analysis.ipynb
│   ├── model_experimentation.ipynb
│   ├── model_optimization.ipynb
│   └── Untitled.ipynb
├── Pipeline
│   ├── logs
│   │   └── project.log
│   ├── __pycache__
│   │   └── pipeline.cpython-39.pyc
│   └── pipeline.py
├── Tests
│   ├── __pycache__
│   │   └── test_config_loader.cpython-39.pyc
│   └── test_config_loader.py
├── Utils
│   ├── __pycache__
│   │   ├── backtest_helper.cpython-39.pyc
│   │   ├── data_ingestion_helper.cpython-39.pyc
│   │   ├── data_loader.cpython-39.pyc
│   │   ├── model_helper.cpython-39.pyc
│   │   └── preprocessing_helper.cpython-39.pyc
│   ├── backtest_helper.py
│   ├── data_ingestion_helper.py
│   ├── model_helper.py
│   └── preprocessing_helper.py
├── .gitignore
├── main.py
├── README.md
└── requirements.txt

```

### Root Directory
- **main.py**: Entry point for executing the entire pipeline.
- **requirements.txt**: Lists all Python packages required to run the project.
- **README.md**: This document.
- **.gitignore**: Specifies files and directories to ignore in version control.

---

#### 1. `checkpoints`
Contains saved checkpoints for various models. These files store intermediate training states, enabling resuming or ensemble modeling.
  - `bidirectional_lstm_checkpoint.keras`
  - `cnn_checkpoint.keras`
  - `cnn_lstm_checkpoint.keras`
  - ... (and more)

---

#### 2. `Components`
Holds scripts for model building, training, and testing, including helper utilities for data ingestion and preprocessing.

- **Model Testing**
  - `Logs`: Stores logs and performance metrics for model testing.
    - `model_performance.txt`
    - `model_testing.log`
  - `model_builder (ann version).py`: Alternative model builder script (for ANN models).
  - `model_performance_colab.txt`: Model performance output from Google Colab.
  - `model_tester (ml version).py`: Machine learning model tester script.
  - Additional files:
    - `data_ingestion.py`: Script for data ingestion and loading.
    - `model_builder.py`: Main model-building script.
    - `preprocessing.py`: Script for data preprocessing tasks.
    - `train.py`: Model training script.

---

#### 3. `Config`
Contains configuration files and loaders, specifying parameters and paths for different modules.

- `config.yaml`: Main configuration file for setting parameters.
- `config_loader.py`: Script to load and handle configuration data.

---

#### 4. `Data`
Stores raw and preprocessed datasets used for training and testing models.

- **Preprocessed**
  - `Preprocessed.csv`: Final processed dataset.
- **Raw**
  - `Pipeline_raw.csv`: Raw dataset for pipeline input.
  - `Rawbtc_ohlcv_jan2023_to_sep2024.csv`: Original dataset for analysis.

---

#### 5. `Logging`
Handles configuration and storage of logs.

- `logging.yaml`: Logging configuration file.
- `logging_config.py`: Python script to initialize logging as per configuration.

---

#### 6. `Logs`
Central storage for logs generated during modeling and project execution.

- `modeling.log`: Log for the modeling process.
- `model_performance.txt`: Documented performance metrics.
- `project.log`: General project logs.

---

#### 7. `Models`
Holds saved models after training.

- `trained_model.pkl`: Serialized trained model.

---

#### 8. `Notebooks`
Contains Jupyter Notebooks for exploratory data analysis, experimentation, and optimization.

- `Exploratory Data Analysis.ipynb`: Initial data exploration and visualization.
- `model_experimentation.ipynb`: Notebook for testing and experimenting with models.
- `model_optimization.ipynb`: Optimizing model hyperparameters and structure.
- Checkpoints:
  - `.ipynb_checkpoints`: Contains notebook checkpoints for autosaved progress.

---

#### 9. `Pipeline`
Scripts and logs for orchestrating the end-to-end pipeline execution.

- `pipeline.py`: Main pipeline script that manages ingestion, preprocessing, and model execution.
- `logs`: Stores logs generated during pipeline execution.

---

#### 10. `Tests`
Unit tests for ensuring modules function correctly.

- `test_config_loader.py`: Tests for configuration loading functions.

---

#### 11. `Utils`
Helper functions for various stages in the pipeline, including data ingestion, preprocessing, and backtesting.

- `backtest_helper.py`: Helper functions for backtesting model predictions.
- `data_ingestion_helper.py`: Utilities for fetching and processing data.
- `model_helper.py`: Support functions for model-related tasks.
- `preprocessing_helper.py`: Helper functions for data preprocessing.

---

## Usage Instructions

### Setup:
- Clone the repository and navigate to the project directory.
- Install dependencies using:

    ```bash
    pip install -r requirements.txt
    ```

### Run the Pipeline:
- Execute the following command to start the main pipeline:

    ```bash
    python main.py
    ```

### Configuration:
- Update the `Config/config.yaml` file to adjust model parameters and other settings.

