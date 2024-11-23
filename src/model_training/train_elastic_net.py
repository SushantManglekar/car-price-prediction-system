import os
import pickle
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from src.utils.custom_logger import create_logger  # Assuming logger is in utils

# Initialize logger
logger = create_logger()

def train_elastic_net(train_path: str, test_path: str, model_output_path: str):
    logger.log_info("Starting Elastic Net training...")

    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    X_train, y_train = train_data.drop(columns=["price"]), train_data["price"]
    X_test, y_test = test_data.drop(columns=["price"]), test_data["price"]

    with mlflow.start_run():
        # Hyperparameter tuning
        elastic_net = ElasticNet()
        param_grid = {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.2, 0.5, 0.8]}
        grid_search = GridSearchCV(elastic_net, param_grid, scoring="r2", cv=5)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Evaluate the model
        y_pred = best_model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_params(grid_search.best_params_)

        # Save the model
        os.makedirs(model_output_path, exist_ok=True)
        model_path = os.path.join(model_output_path, "elastic_net.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        mlflow.sklearn.log_model(best_model, "elastic_net_model")

    logger.log_info("Elastic Net training completed.")


if __name__ == "__main__":
    train_elastic_net("./data/train_data.csv", "./data/test_data.csv", "./models")
