import pandas as pd
from src.utils.custom_logger import create_logger
from src.components.eda import perform_eda  # Assuming the EDA function is in the 'eda' module
from src.components.feature_engineering import perform_feature_engineering  # Assuming the feature engineering function is in 'feature_engineering' module

# Initialize logger
logger = create_logger()

def data_preprocessing_pipeline(file_path: str):
    """
    Executes the full data preprocessing pipeline: EDA followed by Feature Engineering.
    This pipeline takes a file path, loads the data, performs EDA, applies feature engineering,
    and saves the processed train and test datasets.

    Args:
        file_path (str): The file path to the dataset.
    """
    try:
        logger.log_info("Starting data preprocessing pipeline...")
        # Step 1: Load the dataset from the given file path
        logger.log_info(f"Loading data from: {file_path}")
        data = pd.read_csv(file_path)
        
        # Step 2: Perform EDA
        perform_eda(data)  # Perform EDA on the data
        
        # Step 3: Perform Feature Engineering
        perform_feature_engineering(data)  # Perform feature engineering

        logger.log_info("Processed data saved: train_data.csv and test_data.csv")
    
    except Exception as e:
        logger.log_error(f"An error occurred in the data preprocessing pipeline: {str(e)}")
        raise  # Reraise the exception to ensure the process stops

if __name__ == "__main__":
    # File path to the raw dataset
    file_path = 'https://dagshub.com/SushantManglekar/car-price-prediction-system/src/master/data/raw_data.csv'

    # Execute the data preprocessing pipeline
    data_preprocessing_pipeline(file_path)
