import os
import pickle
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from src.utils.custom_logger import create_logger  # Assuming logger is in utils

# Initialize logger
logger = create_logger()

def train_ridge_regression(train_path: str, test_path: str, model_output_path: str):
    logger.log_info("Starting Ridge Regression training...")

    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    X_train, y_train = train_data.drop(columns=["price"]), train_data["price"]
    X_test, y_test = test_data.drop(columns=["price"]), test_data["price"]

    with mlflow.start_run():
        # Hyperparameter tuning
        ridge = Ridge()
        param_grid = {"alpha": [0.1, 1.0, 10.0, 100.0]}
        grid_search = GridSearchCV(ridge, param_grid, scoring="r2", cv=5)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Evaluate the model
        y_pred = best_model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_param("best_alpha", grid_search.best_params_["alpha"])

        # Save the model
        os.makedirs(model_output_path, exist_ok=True)
        model_path = os.path.join(model_output_path, "ridge_regression.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        mlflow.sklearn.log_model(best_model, "ridge_regression_model")

    logger.log_info("Ridge Regression training completed.")


if __name__ == "__main__":
    train_ridge_regression("./data/train_data.csv", "./data/test_data.csv", "./models")
