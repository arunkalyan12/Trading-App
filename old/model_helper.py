import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_data(file_path, label_column='Labels', test_size=0.2, random_state=42):
    """
    Load data from a CSV file, split into features (X) and labels (y),
    and split into training and test sets.

    Parameters:
    - file_path (str): The path to the CSV file.
    - label_column (str): The column containing the target labels.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - X_train, X_test, y_train, y_test: Training and test sets for features and labels.
    """
    try:
        # Load the dataset into a DataFrame
        df = pd.read_csv(file_path)

        # Debug: Print DataFrame shape and columns
        print("DataFrame shape:", df.shape)
        print("DataFrame columns:", df.columns.tolist())

        # Ensure the label_column exists
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in DataFrame.")

        # Separate features (X) and labels (y)
        X = df.drop(columns=[label_column])  # All columns except the label
        y = df[label_column]  # Target label column

        # Remove unwanted columns
        X = X.loc[:, ~X.columns.isin(['Open', 'Close Time'])]

        # Debug: Print the shapes of X and y
        print("Features shape:", X.shape)
        print("Labels shape:", y.shape)

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Debug: Print the shapes of the training and test sets
        print("Training set shape:", X_train.shape, y_train.shape)
        print("Test set shape:", X_test.shape, y_test.shape)

        return X_train, X_test, y_train, y_test

    except FileNotFoundError:
        print(f"File not found at path: {file_path}")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None, None, None, None

# def load_data(file_path, label_column='Labels', test_size=0.2, random_state=42):
#     """
#     Load data from a CSV file, split into features (X) and labels (y),
#     and split into training and test sets.
#
#     Parameters:
#     - file_path (str): The path to the CSV file.
#     - label_column (str): The column containing the target labels.
#     - test_size (float): Proportion of the dataset to include in the test split.
#     - random_state (int): Random seed for reproducibility.
#
#     Returns:
#     - X_train, X_test, y_train, y_test: Training and test sets for features and labels.
#     """
#     try:
#         # Load the dataset into a DataFrame
#         df = pd.read_csv(file_path)
#
#         # Set 'Open Time' as the index if present
#         if 'Open Time' in df.columns:
#             df.set_index('Open Time', inplace=True)
#
#         # Separate features (X) and labels (y)
#         X = df.drop(columns=[label_column])  # All columns except the label
#         y = df[label_column]  # Target label column
#
#         # Remove unwanted columns
#         X = X.loc[:, ~X.columns.isin(['Open', 'Close Time'])]
#
#         # Split into training and test sets
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
#
#         return X_train, X_test, y_train, y_test
#
#     except FileNotFoundError:
#         print(f"File not found at path: {file_path}")
#         return None, None, None, None
#     except Exception as e:
#         print(f"An error occurred while loading data: {e}")
#         return None, None, None, None


def build_model(best_params):
    """
    Build and return the model with the given best hyperparameters.

    Parameters:
    - best_params (dict): The best hyperparameters for the model.

    Returns:
    - model: The initialized model.
    """
    model = ExtraTreesClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        bootstrap=best_params['bootstrap']
    )
    return model


def train_model(model, X_train, y_train):
    """
    Train the model using the training dataset.

    Parameters:
    - model: The machine learning model to be trained.
    - X_train (pd.DataFrame): Features for training.
    - y_train (pd.Series): Labels for training.

    Returns:
    - model: The trained model.
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using the test dataset.

    Parameters:
    - model: The trained model to evaluate.
    - X_test: Features for the test dataset.
    - y_test: True labels for the test dataset.

    Returns:
    - accuracy: Accuracy of the model on the test dataset.
    - report: Classification report of the model's performance.
    """

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate classification report
    report = classification_report(y_test, y_pred)

    return accuracy, report

def save_model(model, config):
    """
    Save the trained model to a specified path in the config.

    Parameters:
    - model: The trained model to save.
    - config: The configuration dictionary containing model saving path.
    """
    try:
        model_path = config['model']['save_path']
        os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Create directories if they don't exist
        joblib.dump(model, model_path)
        print(f'Model saved to {model_path}')
    except Exception as e:
        print(f'Error saving model: {e}')

