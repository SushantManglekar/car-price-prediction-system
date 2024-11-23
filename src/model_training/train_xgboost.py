import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from src.utils.custom_logger import create_logger  # Assuming logger is in utils

# Initialize logger
logger = create_logger()

def train_xgboost(train_path: str, test_path: str, model_output_path: str):
    """
    Train an XGBoost model with hyperparameter tuning, log metrics to MLflow, and save the trained model.

    Args:
        train_path (str): Path to the train dataset.
        test_path (str): Path to the test dataset.
        model_output_path (str): Directory where the model will be saved.
    """
    logger.log_info("Starting XGBoost Regressor training...")

    # Load train and test data
    logger.log_info(f"Loading training data from {train_path}")
    train_data = pd.read_csv(train_path)
    logger.log_info(f"Loading test data from {test_path}")
    test_data = pd.read_csv(test_path)

    # Split features and target
    X_train = train_data.drop(columns=["price"])
    y_train = train_data["price"]
    X_test = test_data.drop(columns=["price"])
    y_test = test_data["price"]

    # Start an MLflow run
    with mlflow.start_run():
        logger.log_info("Training an XGBoost model...")

        # Step 1: Initialize and train the XGBoost model
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        model.fit(X_train, y_train)

        # Step 2: Perform cross-validation
        logger.log_info("Performing cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        logger.log_info(f"Cross-validation R² scores: {cv_scores}")
        logger.log_info(f"Mean CV R² score: {np.mean(cv_scores)}")
        mlflow.log_metric("mean_cv_r2", np.mean(cv_scores))

        # Step 3: Evaluate on test data
        logger.log_info("Evaluating model on test data...")
        y_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)
        logger.log_info(f"R² score on test data: {test_r2}")
        logger.log_info(f"Mean Squared Error on test data: {test_mse}")
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("test_mse", test_mse)

        # Step 4: Hyperparameter tuning (XGBoost)
        logger.log_info("Performing hyperparameter tuning with GridSearchCV...")
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0]
        }
        grid_search = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', random_state=42), param_grid, scoring="r2", cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        logger.log_info(f"Best XGBoost parameters: {grid_search.best_params_}")
        logger.log_info(f"Best GridSearch R² score: {grid_search.best_score_}")
        mlflow.log_param("best_xgb_params", grid_search.best_params_)
        mlflow.log_metric("best_xgb_cv_r2", grid_search.best_score_)

        # Save the best model from grid search
        best_model = grid_search.best_estimator_
        logger.log_info("Saving the trained XGBoost model...")
        os.makedirs(model_output_path, exist_ok=True)
        model_file = os.path.join(model_output_path, "xgboost_model.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(best_model, f)
        mlflow.sklearn.log_model(best_model, "xgboost_model")
        logger.log_info(f"Model saved at {model_file}")

    logger.log_info("XGBoost Regressor training completed.")


if __name__ == "__main__":
    # Define file paths
    train_file_path = "./data/train_data.csv"
    test_file_path = "./data/test_data.csv"
    model_save_path = "./models"

    # Call the training function
    train_xgboost(train_file_path, test_file_path, model_save_path)
