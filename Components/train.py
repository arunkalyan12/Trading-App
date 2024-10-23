import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Config.config_loader import load_config
from Utils.model_helper import load_data, build_model, train_model, evaluate_model, save_model


def main():
    config = load_config()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = load_data(config['data']['path'], config['data']['label'])

    # Build the model
    best_params = config.get('model', {}).get('best_params', {})
    model = build_model(best_params)

    # Train the model
    trained_model = train_model(model, X_train, y_train)

    # Evaluate the model
    test_accuracy, test_report = evaluate_model(trained_model, X_test, y_test)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(test_report)

    # Save the model
    save_model(trained_model, config)

if __name__ == "__main__":
    main()