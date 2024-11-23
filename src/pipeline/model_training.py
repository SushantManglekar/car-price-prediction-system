import os
import pickle
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import mlflow
from mlflow.tracking import MlflowClient
from src.utils.custom_logger import create_logger
from src.model_training.train_linear_regression import train_linear_regression
from src.model_training.train_ridge_regression import train_ridge_regression
from src.model_training.train_elastic_net import train_elastic_net
from src.model_training.train_random_forest import train_random_forest
from src.model_training.train_xgboost import train_xgboost
from src.model_training.train_neural_net import train_neural_network
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize logger
logger = create_logger()

def evaluate_model(model_path, X_test, y_test):
    """
    Load a model, make predictions on the test set, and calculate evaluation metrics.

    Args:
        model_path (str): Path to the saved model file.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target values.

    Returns:
        dict: Dictionary containing R² score and Mean Squared Error (MSE).
    """
    # Check if the model is a Keras model (.h5 file)
    if model_path.endswith(".h5"):
        model = load_model(model_path)  # Load Keras model
    # Check if the model is an MLflow model
    elif model_path.startswith("runs:/"):  # MLflow model URI format
        model = mlflow.keras.load_model(model_path)  # Adjust based on the actual MLflow model type
    # For other models (Scikit-learn, XGBoost, etc.)
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)  # Load Scikit-learn model

    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate and return metrics
    return {
        "r2_score": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
    }

def register_model_in_mlflow(model_name, model_path, metrics):
    """
    Register the best model in MLflow Model Registry and transition it to the 'Staging' stage.

    Args:
        model_name (str): Name of the model to register.
        model_path (str): Path to the model file.
        metrics (dict): Dictionary of evaluation metrics.
    """
    logger.log_info(f"Registering model {model_name} to MLflow...")

    # Set the MLflow tracking URI (local or remote)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Update with your MLflow server URI if remote

    # Initialize MLflow client
    client = MlflowClient()

    try:
        # Create an MLflow run to log the model
        with mlflow.start_run():
            # Log metrics
            mlflow.log_metric("r2_score", metrics["r2_score"])
            mlflow.log_metric("mse", metrics["mse"])

            # Log the model
            if model_path.endswith(".h5"):
                mlflow.keras.log_model(
                    tf.keras.models.load_model(model_path),
                    artifact_path="model",
                    registered_model_name=model_name
                )
            else:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=model_name
                )

        logger.log_info(f"Model {model_name} registered successfully in MLflow.")

        # Fetch the latest registered model version
        latest_versions = client.get_latest_versions(name=model_name, stages=["None"])
        if not latest_versions:
            raise ValueError(f"No registered version found for the model {model_name}.")

        latest_version = latest_versions[0].version
        logger.log_info(f"Latest version {latest_version} of model {model_name} retrieved.")

        # Transition the latest model version to the 'Staging' stage
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Staging"
        )
        logger.log_info(f"Model {model_name} version {latest_version} transitioned to 'Staging' stage.")

    except Exception as e:
        logger.log_error(f"An error occurred while registering and staging the model: {str(e)}")
        raise e


def run_model_training_pipeline(train_path: str, test_path: str, models_dir: str, evaluation_output_path: str):
    """
    Run the pipeline to train multiple models, evaluate them, and select the best-performing model.

    Args:
        train_path (str): Path to the training dataset.
        test_path (str): Path to the testing dataset.
        models_dir (str): Directory to save trained models.
        evaluation_output_path (str): File path to save the evaluation results.
    """
    logger.log_info("Starting the model training pipeline...")

    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Train and save models
    logger.log_info("Training Linear Regression...")
    train_linear_regression(train_path, test_path, models_dir)

    logger.log_info("Training Ridge Regression...")
    train_ridge_regression(train_path, test_path, models_dir)

    logger.log_info("Training Elastic Net...")
    train_elastic_net(train_path, test_path, models_dir)

    logger.log_info("Training Random forest...")
    train_random_forest(train_path, test_path, models_dir)

    logger.log_info("Training XGBoost...")
    train_xgboost(train_path, test_path, models_dir)

    logger.log_info("Training Neural Net...")
    train_neural_network(train_path, test_path, models_dir)

    # Load test data
    logger.log_info("Loading test data...")
    test_data = pd.read_csv(test_path)
    X_test = test_data.drop(columns=["price"])
    y_test = test_data["price"]

    # Evaluate models
    logger.log_info("Evaluating models...")
    evaluation_results = []
    model_paths = {
        "Linear Regression": os.path.join(models_dir, "linear_regression.pkl"),
        "Ridge Regression": os.path.join(models_dir, "ridge_regression.pkl"),
        "Elastic Net": os.path.join(models_dir, "elastic_net.pkl"),
        "Random Forest": os.path.join(models_dir, "random_forest.pkl"),
        "XGBoost": os.path.join(models_dir, "xgboost_model.pkl"),
        "Neural Net": os.path.join(models_dir, "neural_network_model.h5"),
    }

    for model_name, model_path in model_paths.items():
        metrics = evaluate_model(model_path, X_test, y_test)
        evaluation_results.append({
            "model": model_name,
            "r2_score": metrics["r2_score"],
            "mse": metrics["mse"],
        })
        logger.log_info(f"{model_name} - R²: {metrics['r2_score']}, MSE: {metrics['mse']}")

    # Save evaluation results
    logger.log_info("Saving evaluation results...")
    results_df = pd.DataFrame(evaluation_results)
    results_df.to_csv(evaluation_output_path, index=False)

    # Select the best model
    best_model = results_df.sort_values(by="r2_score", ascending=False).iloc[0]
    logger.log_info(f"Best model: {best_model['model']} with R²: {best_model['r2_score']}, MSE: {best_model['mse']}")

    # Register the best model in MLflow
    best_model_name = best_model['model']
    best_model_path = model_paths[best_model_name]
    register_model_in_mlflow("BestCarPricePredictionModel", best_model_path, best_model)

    logger.log_info("Model training pipeline completed.")


def promote_model_to_production(model_name):
    """
    Promote the latest staged version of a model to the Production stage.

    Args:
        model_name (str): Name of the model in the MLflow Model Registry.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Replace with your MLflow server URI
    client = MlflowClient()

    try:
        # Fetch the latest version in 'Staging'
        staged_versions = client.get_latest_versions(name=model_name, stages=["Staging"])
        if not staged_versions:
            raise ValueError(f"No staged version found for model {model_name}.")

        staged_version = staged_versions[0].version
        print(f"Staged version {staged_version} of model {model_name} retrieved.")

        # Transition the staged model to Production
        client.transition_model_version_stage(
            name=model_name,
            version=staged_version,
            stage="Production"
        )
        print(f"Model {model_name} version {staged_version} transitioned to 'Production' stage.")

    except Exception as e:
        print(f"An error occurred while promoting the model to production: {str(e)}")
        raise e

if __name__ == "__main__":
    # Define file paths
    train_file_path = "./data/train_data.csv"
    test_file_path = "./data/test_data.csv"
    models_directory = "./models"
    evaluation_results_path = "./models/evaluation_results.csv"

    # Run the pipeline
    run_model_training_pipeline(train_file_path, test_file_path, models_directory, evaluation_results_path)

    promote_model_to_production("BestCarPricePredictionModel")