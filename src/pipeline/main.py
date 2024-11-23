import os
import pandas as pd
from src.pipeline.data_preprocessing import data_preprocessing_pipeline
from src.pipeline.model_training import run_model_training_pipeline
from src.utils.custom_logger import create_logger

# Initialize logger
logger = create_logger()

def run_full_pipeline(raw_data_path: str, train_data_path: str, test_data_path: str, models_dir: str, evaluation_output_path: str):
    """
    Run the full pipeline: Data Preprocessing followed by Model Training.

    Args:
        raw_data_path (str): Path to the raw dataset.
        train_data_path (str): Path to save the processed training dataset.
        test_data_path (str): Path to save the processed testing dataset.
        models_dir (str): Directory to save trained models.
        evaluation_output_path (str): Path to save evaluation results.
    """
    try:
        # Step 1: Data Preprocessing Pipeline
        logger.log_info("Starting data preprocessing pipeline...")
        data_preprocessing_pipeline(raw_data_path)

        # Step 2: Model Training Pipeline
        logger.log_info("Starting model training pipeline...")
        run_model_training_pipeline(train_data_path, test_data_path, models_dir, evaluation_output_path)
        
        
    except Exception as e:
        logger.log_error(f"An error occurred during the full pipeline execution: {str(e)}")
        raise  # Reraise the exception to stop the pipeline if something fails

if __name__ == "__main__":
    # Define paths
    raw_data_path = './data/raw_data.csv'
    train_data_path = './data/train_data.csv'
    test_data_path = './data/test_data.csv'
    models_directory = './models'
    evaluation_results_path = './models/evaluation_results.csv'

    # Ensure the necessary directories exist
    os.makedirs(models_directory, exist_ok=True)

    # Run the full pipeline
    run_full_pipeline(raw_data_path, train_data_path, test_data_path, models_directory, evaluation_results_path)
