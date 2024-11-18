import pandas as pd
import mlflow
from io import StringIO
from src.utils.custom_logger import create_logger


class DataExplorer:

    def __init__(self):
        self.logger = create_logger()

    def log_info(self, data):
        """Helper function to log basic data info."""
        try:
            # Capture DataFrame info as a string
            buf = StringIO()
            data.info(buf=buf)
            df_info_str = buf.getvalue()
            
            # Log info string as parameter
            mlflow.log_param("data_info", df_info_str)
            self.logger.log_info("Data information logged successfully.")

        except Exception as e:
            self.logger.log_error(f"Error while logging data info: {e}")
            mlflow.log_param("data_info_error", str(e))

    def log_summary(self, data):
        """Helper function to log summary statistics."""
        try:
            # Capture describe summary as a string
            df_summary_str = data.describe().to_string()
            
            # Log summary string as parameter
            mlflow.log_param("data_summary", df_summary_str)
            self.logger.log_info("Data summary statistics logged successfully.")

        except Exception as e:
            self.logger.log_error(f"Error while logging data summary: {e}")
            mlflow.log_param("data_summary_error", str(e))

    def log_missing_data(self, data):
        """Log missing data count."""
        try:
            missing_data = data.isnull().sum().to_dict()
            missing_data_str = str(missing_data)
            
            # Log missing data as string
            mlflow.log_param("missing_data", missing_data_str)
            self.logger.log_info("Missing data logged successfully.")

        except Exception as e:
            self.logger.log_error(f"Error while logging missing data: {e}")
            mlflow.log_param("missing_data_error", str(e))

    def log_duplicates(self, data):
        """Log duplicated rows count."""
        try:
            duplicates = data.duplicated().sum()
            
            # Log duplicates count as string
            mlflow.log_param("duplicates", str(duplicates))
            self.logger.log_info("Duplicate rows logged successfully.")

        except Exception as e:
            self.logger.log_error(f"Error while logging duplicates: {e}")
            mlflow.log_param("duplicates_error", str(e))

    def explore_and_log_data(self, data: pd.DataFrame):
        """Explores the data and logs to MLFlow. Starts and ends the MLFlow run automatically."""
        try:
            with mlflow.start_run():
                self.logger.log_info("Starting data exploration and logging process.")

                # Log Data Info
                self.log_info(data)

                # Log Describe Summary
                self.log_summary(data)

                # Log Missing Values
                self.log_missing_data(data)

                # Log Duplicate Rows
                self.log_duplicates(data)

                self.logger.log_info("Data exploration and logging process completed successfully.")

        except Exception as e:
            self.logger.log_error(f"Error while exploring data: {e}")
            mlflow.log_param("exploration_error", str(e))


# This block ensures the code below only runs if this script is executed directly
if __name__ == "__main__":
    # Load the dataframe (your raw data)
    df = pd.read_csv('./data/raw_data.csv')

    # Initialize DataExplorer
    data_explorer = DataExplorer()

    # Perform data exploration and log results
    data_explorer.explore_and_log_data(df)
