import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile
import mlflow
from src.utils.custom_logger import create_logger


class OutlierDetector:
    def __init__(self):
        self.logger = create_logger()

    def detect_outliers(self, data: pd.Series, method="IQR"):
        """
        Detect outliers in a numerical column using the specified method.
        
        Args:
            data (pd.Series): The column to analyze.
            method (str): Method to detect outliers ("IQR" or "Z-Score").
        
        Returns:
            pd.Series: Boolean mask where True indicates an outlier.
        """
        try:
            if method == "IQR":
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return (data < lower_bound) | (data > upper_bound)

            elif method == "Z-Score":
                mean = data.mean()
                std_dev = data.std()
                z_scores = (data - mean) / std_dev
                return abs(z_scores) > 3

            else:
                raise ValueError("Invalid method specified. Use 'IQR' or 'Z-Score'.")
            
        except Exception as e:
            self.logger.log_error(f"Error detecting outliers in column: {e}")
            raise

    def visualize_outliers(self, data: pd.DataFrame, column: str):
        """
        Visualize outliers in a column using scatter plot and box plot.
        
        Args:
            data (pd.DataFrame): The DataFrame containing the column.
            column (str): The name of the column to visualize.
        """
        try:
            # Create figure with two subplots (scatter and box plot)
            plt.figure(figsize=(10, 6))

            # Plot Histogram and KDE (Density Plot)
            plt.subplot(1, 2, 1)
            sns.histplot(data[column], kde=True, bins=30)
            plt.title(f'Distribution of {column}')
            
            # Plot Boxplot to visualize spread and outliers
            plt.subplot(1, 2, 2)
            sns.boxplot(x=data[column])
            plt.title(f'Boxplot of {column}')

            plt.tight_layout()
           

            # Save plot to temporary file and log as artifact
            with NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                plt.savefig(tmpfile.name)
                mlflow.log_artifact(tmpfile.name, f"outlier_visualizations/{column}_outliers.png")

            plt.close()
            self.logger.log_info(f"Outlier visualization for '{column}' logged successfully.")

        except Exception as e:
            self.logger.log_error(f"Error while visualizing outliers for column '{column}': {e}")
            mlflow.log_param(f"{column}_outlier_visualization_error", str(e))

    def detect_and_visualize(self, data: pd.DataFrame, columns: list, method="IQR"):
        """
        Detect and visualize outliers for a list of columns.
        
        Args:
            data (pd.DataFrame): The DataFrame to analyze.
            columns (list): List of numerical columns to check for outliers.
            method (str): Method to detect outliers ("IQR" or "Z-Score").
        """
        try:
            with mlflow.start_run():
                for column in columns:
                    if data[column].dtype in [np.float64, np.int64, np.float32, np.int32]:
                        # Detect outliers
                        outlier_mask = self.detect_outliers(data[column], method=method)
                        outlier_count = outlier_mask.sum()

                        # Log outlier count
                        mlflow.log_param(f"{column}_outlier_count", outlier_count)
                        self.logger.log_info(f"{column}: {outlier_count} outliers detected.")

                        # If outliers exist, visualize them
                        if outlier_count > 0:
                            self.visualize_outliers(data, column)
                        else:
                            self.logger.log_info(f"{column}: No outliers detected.")
                    else:
                        self.logger.log_warning(f"{column} is not a numerical column and will be skipped.")

        except Exception as e:
            self.logger.log_error(f"Error while detecting and visualizing outliers: {e}")
            mlflow.log_param("outlier_detection_error", str(e))


# This block ensures the code below only runs if this script is executed directly
if __name__ == "__main__":
    # Load the DataFrame (your raw data)
    data = pd.read_csv('./data/raw_data.csv')

    # Specify the numerical columns to check
    numerical_columns = ["Price", "Airbags"]

    # Initialize OutlierDetector
    detector = OutlierDetector()

    # Detect and visualize outliers
    detector.detect_and_visualize(data, numerical_columns, method="IQR")
