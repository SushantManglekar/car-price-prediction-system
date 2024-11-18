import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
from tempfile import NamedTemporaryFile
from src.utils.custom_logger import create_logger


class UnivariateAnalyzer:

    def __init__(self):
        self.logger = create_logger()

    def log_numerical_feature(self, data: pd.DataFrame, feature: str):
        """Logs univariate analysis for a numerical feature."""
        try:
            self.logger.log_info(f"Starting univariate analysis for numerical feature: {feature}")
            
            plt.figure(figsize=(10, 6))

            # Plot Histogram and KDE (Density Plot)
            plt.subplot(1, 2, 1)
            sns.histplot(data[feature], kde=True, bins=30)
            plt.title(f'Distribution of {feature}')
            
            # Plot Boxplot to visualize spread and outliers
            plt.subplot(1, 2, 2)
            sns.boxplot(x=data[feature])
            plt.title(f'Boxplot of {feature}')

            plt.tight_layout()

            # Save plot to a temporary file and log as artifact
            with NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                plt.savefig(tmpfile.name)
                mlflow.log_artifact(tmpfile.name, f"plots/{feature}_distribution.png")
            plt.close()

            self.logger.log_info(f"Logged histogram plot for numerical feature: {feature}")

            # Log basic statistics
            summary_stats = data[feature].describe().to_dict()
            mlflow.log_param(f"{feature}_stats", summary_stats)
            self.logger.log_info(f"Logged summary statistics for numerical feature: {feature}")

        except Exception as e:
            self.logger.log_error(f"Error while logging numerical feature {feature}: {e}")
            mlflow.log_param(f"{feature}_error", str(e))

    def log_categorical_feature(self, data: pd.DataFrame, feature: str):
        """Logs univariate analysis for a categorical feature."""
        try:
            self.logger.log_info(f"Starting univariate analysis for categorical feature: {feature}")
            
            # Generate bar plot
            plt.figure(figsize=(8, 4))
            sns.countplot(data=data, x=feature, order=data[feature].value_counts().index)
            plt.title(f"Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.xticks(rotation=45)

            # Save plot to a temporary file and log as artifact
            with NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                plt.savefig(tmpfile.name)
                mlflow.log_artifact(tmpfile.name, f"plots/{feature}_distribution.png")
            plt.close()

            self.logger.log_info(f"Logged bar plot for categorical feature: {feature}")

            # Log value counts
            value_counts = data[feature].value_counts().to_dict()
            mlflow.log_param(f"{feature}_value_counts", value_counts)
            self.logger.log_info(f"Logged value counts for categorical feature: {feature}")

        except Exception as e:
            self.logger.log_error(f"Error while logging categorical feature {feature}: {e}")
            mlflow.log_param(f"{feature}_error", str(e))

    def analyze_and_log(self, data: pd.DataFrame, numerical_features: list, categorical_features: list):
        """
        Perform univariate analysis and log results to MLFlow.
        
        Args:
            data (pd.DataFrame): Input data.
            numerical_features (list): List of numerical features to analyze.
            categorical_features (list): List of categorical features to analyze.
        """
        try:
            self.logger.log_info("Starting univariate analysis for all features.")

            with mlflow.start_run():
                # Log numerical features
                for feature in numerical_features:
                    self.log_numerical_feature(data, feature)

                # Log categorical features
                for feature in categorical_features:
                    self.log_categorical_feature(data, feature)

            self.logger.log_info("Completed univariate analysis for all features.")

        except Exception as e:
            self.logger.log_error(f"Error while performing univariate analysis: {e}")
            mlflow.log_param("univariate_analysis_error", str(e))


# This block ensures the code below only runs if this script is executed directly
if __name__ == "__main__":
    # Load the dataframe (your raw data)
    df = pd.read_csv('./data/raw_data.csv')

    # Initialize UnivariateAnalyzer
    analyzer = UnivariateAnalyzer()

    # Specify numerical and categorical features
    numerical_features = ["Price", "Levy"]
    categorical_features = ["Manufacturer", "Fuel type", "Gear box type"]

    # Perform univariate analysis and log results
    analyzer.analyze_and_log(df, numerical_features, categorical_features)
