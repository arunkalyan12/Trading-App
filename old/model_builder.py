import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier, BaggingClassifier

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Config.config_loader import load_config


def main():
    try:
        # Load configuration
        config = load_config()  # Specify the path to your config file

        # Load preprocessed data directly from CSV file
        df = pd.read_csv(config['data']['path'])

        # Drop rows with missing values
        df.dropna(inplace=True)

        # Define features and labels based on the config
        X = df[config['data']['features']]
        X = X.loc[:, ~X.columns.isin(['Open', 'Close Time'])]
        y = df[config['data']['label']]

        # Hyperparameter space for Extra Trees
        param_space = {
            'n_estimators': [50, 100, 200, 500],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        # Set up K-Fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        best_accuracy = 0
        best_params = None
        results = []

        # Randomized search with cross-validation
        for i in range(15):  # 15 iterations
            model = ExtraTreesClassifier()

            # RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_space,
                n_iter=10,  # Number of parameter settings sampled
                cv=kf,
                scoring='accuracy',
                random_state=42,
                n_jobs=-1,
                verbose=1
            )

            # Fit and evaluate
            random_search.fit(X, y)
            accuracy = random_search.best_score_
            params = random_search.best_params_

            results.append((accuracy, params))

            # Check if this is the best accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params

        # Log best parameters to a file
        with open('best_extra_trees_params.txt', 'w') as f:
            f.write(f'Best Accuracy: {best_accuracy:.4f}\n')
            f.write(f'Best Hyperparameters: {best_params}\n')

    except Exception as e:
        print(f'Error occurred: {e}')


if __name__ == "__main__":
    main()
