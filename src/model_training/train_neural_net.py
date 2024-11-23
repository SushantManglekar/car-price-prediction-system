import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from src.utils.custom_logger import create_logger  # Assuming logger is in utils

# Initialize logger
logger = create_logger()

def build_neural_network(input_dim):
    """
    Builds a simple neural network model for regression.

    Args:
        input_dim (int): The number of input features.

    Returns:
        keras.Model: The compiled Keras model.
    """
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))  # First hidden layer
    model.add(Dense(32, activation='relu'))  # Second hidden layer
    model.add(Dense(1))  # Output layer with one neuron (for regression)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def train_neural_network(train_path: str, test_path: str, model_output_path: str):
    """
    Train a Neural Network for regression and log metrics to MLflow.

    Args:
        train_path (str): Path to the training dataset.
        test_path (str): Path to the testing dataset.
        model_output_path (str): Directory where the model will be saved.
    """
    logger.log_info("Starting Neural Network training...")

    # Load the train and test data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Split features and target
    X_train = train_data.drop(columns=["price"]).values
    y_train = train_data["price"].values
    X_test = test_data.drop(columns=["price"]).values
    y_test = test_data["price"].values

    # Convert X_train and X_test to float32
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)

    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    # Build the neural network model
    model = build_neural_network(X_train.shape[1])

    # Start an MLflow run
    with mlflow.start_run():
        # Train the model
        logger.log_info("Training Neural Network model...")
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

        # Evaluate the model
        logger.log_info("Evaluating model on test data...")
        y_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)

        # Log metrics to MLflow
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("test_mse", test_mse)

        logger.log_info(f"R² score on test data: {test_r2}")
        logger.log_info(f"Mean Squared Error on test data: {test_mse}")

        # Save the trained model
        logger.log_info("Saving the trained Neural Network model...")
        os.makedirs(model_output_path, exist_ok=True)
        model_file = os.path.join(model_output_path, "neural_network_model.h5")
        model.save(model_file)
        mlflow.keras.log_model(model, "neural_network_model")
        logger.log_info(f"Model saved at {model_file}")

    logger.log_info("Neural Network training completed.")

if __name__ == "__main__":
    # Define file paths
    train_file_path = "./data/train_data.csv"
    test_file_path = "./data/test_data.csv"
    model_save_path = "./models"

    # Call the training function
    train_neural_network(train_file_path, test_file_path, model_save_path)
