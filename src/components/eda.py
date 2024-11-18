import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import mlflow
from io import StringIO
from src.utils.custom_logger import create_logger  # Assuming logger is in utils
from src.utils.data_explorer import DataExplorer
from src.utils.bivariate_analyzer import BivariateAnalyzer
from src.utils.univariate_analyzer import UnivariateAnalyzer
from src.utils.outlier_detector import OutlierDetector

# Initialize logger
logger = create_logger()

def perform_eda(data: pd.DataFrame):
    """
    Perform EDA, save outputs to the specified directory, log artifacts to MLflow, and log insights.

    Args:
        data (pd.DataFrame): The dataset to analyze.
    """
    try:
        logger.log_info("Starting EDA...")

        # Step 1. Dataset exploration
        logger.log_info("Exploring Dataset...")

        # Initialize DataExplorer with the dataframe
        data_explorer = DataExplorer()

        # Start MLflow run and perform exploration
        data_explorer.explore_and_log_data(data=data)
        
        # Step 2. Univariate analysis
        logger.log_info("Starting Univariate analysis...")

        univariate_analyzer = UnivariateAnalyzer()

        numerical_columns = ['Price','Cylinders']
        categorical_columns = ['Category','Fuel type','Gear box type', 'Wheel']
        univariate_analyzer.analyze_and_log(data=data, numerical_features=numerical_columns, categorical_features=categorical_columns)

        # Step 3. Bivariate analysis
        logger.log_info("Starting Bivariate analysis...")

        # Initialize BivariateAnalyzer
        bivariate_analyzer = BivariateAnalyzer()

        # Specify numerical and categorical features
        numerical_features = ["Price", "Airbags"]
        categorical_features = ["Fuel type", "Gear box type","Drive wheels"]

        # Perform bivariate analysis and log results
        bivariate_analyzer.analyze_and_log(data, numerical_features, categorical_features)

        # Step 4. Outlier analysis
        logger.log_info("Starting Outlier analysis...")

        # Specify the numerical columns to check
        numerical_columns = ["Price", "Cylinders"]

        # Initialize OutlierDetector
        detector = OutlierDetector()

        # Detect and visualize outliers
        detector.detect_and_visualize(data, numerical_columns, method="IQR")

        logger.log_info("EDA completed. Results logged to MLflow.")

    except Exception as e:
        logger.log_error(f"An error occurred during EDA: {str(e)}")
        raise  # Reraise the exception to ensure the process stops


if __name__ == "__main__":
    
    try:
        # Load the data into a DataFrame
        data = pd.read_csv('./data/raw_data.csv')

        # Call the perform_eda function
        perform_eda(data=data)

    except Exception as e:
        logger.log_error(f"An error occurred while running the EDA script: {str(e)}")
