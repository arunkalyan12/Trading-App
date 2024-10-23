#This code was run on colab refer to the notebooks folder

import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron, \
    PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
import os
from google.colab import files

# Load configuration from YAML file
def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load and preprocess data
def main(config_path: str):
    config = load_config(config_path)

    # Load the preprocessed data
    df = pd.read_csv('/content/drive/MyDrive/projects/Prerpcessing2.csv')
    df.dropna(inplace = True)

    # Define features and labels based on the config
    X = df[config['data']['features']]  # Features based on the config
    y = df[config['data']['label']]  # Labels based on the config
    X = X.loc[:, ~X.columns.isin(['Open', 'Close Time'])]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # List of models to test
    models = {
        "Logistic Regression": LogisticRegression(),
        "KNeighbors Classifier": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Extra Trees": ExtraTreesClassifier(),
        "Bagging": BaggingClassifier(),
        "MLP Classifier": MLPClassifier(),
        "SVC": SVC(),
        "Linear SVC": LinearSVC(),
        "Ridge Classifier": RidgeClassifier(),
        "SGD Classifier": SGDClassifier(),
        "GaussianNB": GaussianNB(),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
        "Perceptron": Perceptron(),
        "Passive Aggressive": PassiveAggressiveClassifier(),
        "Nearest Centroid": NearestCentroid(),
        "XGBoost Classifier": XGBClassifier()
    }

    # Evaluate each model
    for name, model in models.items():
        if name == "XGBoost Classifier":
            # Remap the labels from [-1, 0, 1] to [2, 0, 1] for compatibility with XGBoost
            y_train_mapped = y_train.map({-1: 2, 0: 0, 1: 1})
            y_test_mapped = y_test.map({-1: 2, 0: 0, 1: 1})
            model.fit(X_train, y_train_mapped)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test_mapped, y_pred)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

        with open('/content/drive/MyDrive/projects/model_performance_colab.txt', 'a') as txt:
            txt.write(f'{name} Accuracy: {accuracy:.4f}\n')

# Replace with your Google Drive path to the config file
if __name__ == "__main__":
    main('/content/drive/MyDrive/projects/config.yaml')
