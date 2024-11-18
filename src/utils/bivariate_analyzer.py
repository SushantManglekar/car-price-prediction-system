import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
from tempfile import NamedTemporaryFile
from src.utils.custom_logger import create_logger


class BivariateAnalyzer:

    def __init__(self):
        self.logger = create_logger()

    def log_heatmap(self, data: pd.DataFrame, numerical_features: list):
        """Logs a heatmap showing correlations between numerical features."""
        try:
            self.logger.log_info(f"Starting heatmap generation for all numerical features...")
            
            # Drop rows/columns with NaN values to avoid issues in correlation
            cleaned_data = data[data.select_dtypes(include=['int64','float64']).columns.to_list()].dropna()

            # Calculate correlation matrix
            correlation_matrix = cleaned_data.corr()

            # Generate heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm",linewidths=0.5)
            plt.title("Correlation Heatmap")

            # Save plot to a temporary file and log as artifact
            with NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                plt.savefig(tmpfile.name)
                mlflow.log_artifact(tmpfile.name, "plots/correlation_heatmap.png")
            plt.close()

            self.logger.log_info("Logged heatmap showing correlation between numerical features.")

        except Exception as e:
            self.logger.log_error(f"Error while logging heatmap: {e}")
            mlflow.log_param("heatmap_error", str(e))

    def log_scatterplot(self, data: pd.DataFrame, feature_x: str, feature_y: str):
        """Logs scatterplots for two numerical features."""
        try:
            self.logger.log_info(f"Starting scatterplot generation for {feature_x} vs {feature_y}")
            
            # Generate scatterplot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=data[feature_x], y=data[feature_y])
            plt.title(f"Scatter Plot: {feature_x} vs {feature_y}")
            plt.xlabel(feature_x)
            plt.ylabel(feature_y)

            # Save plot to a temporary file and log as artifact
            with NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                plt.savefig(tmpfile.name)
                mlflow.log_artifact(tmpfile.name, f"plots/{feature_x}_vs_{feature_y}_scatter.png")
            plt.close()

            self.logger.log_info(f"Logged scatterplot for {feature_x} vs {feature_y}")

        except Exception as e:
            self.logger.log_error(f"Error while logging scatterplot {feature_x} vs {feature_y}: {e}")
            mlflow.log_param(f"{feature_x}_vs_{feature_y}_scatter_error", str(e))

    def log_boxplot(self, data: pd.DataFrame, numerical_feature: str, categorical_feature: str):
        """Logs boxplots for a numerical feature grouped by a categorical feature."""
        try:
            self.logger.log_info(f"Starting boxplot generation for {numerical_feature} by {categorical_feature}")
            
            # Generate boxplot
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=data[categorical_feature], y=data[numerical_feature])
            plt.title(f"Box Plot: {numerical_feature} by {categorical_feature}")
            plt.xlabel(categorical_feature)
            plt.ylabel(numerical_feature)
            plt.xticks(rotation=45)

            # Save plot to a temporary file and log as artifact
            with NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                plt.savefig(tmpfile.name)
                mlflow.log_artifact(tmpfile.name, f"plots/{numerical_feature}_by_{categorical_feature}_boxplot.png")
            plt.close()

            self.logger.log_info(f"Logged boxplot for {numerical_feature} by {categorical_feature}")

        except Exception as e:
            self.logger.log_error(f"Error while logging boxplot {numerical_feature} by {categorical_feature}: {e}")
            mlflow.log_param(f"{numerical_feature}_by_{categorical_feature}_boxplot_error", str(e))

    def log_countplot(self, data: pd.DataFrame, feature_x: str, feature_y: str):
        """Logs countplots for two categorical features."""
        try:
            self.logger.log_info(f"Starting countplot generation for {feature_x} vs {feature_y}")
            
            # Generate countplot
            plt.figure(figsize=(12, 6))
            sns.countplot(x=feature_x, hue=feature_y, data=data)
            plt.title(f"Count Plot: {feature_x} vs {feature_y}")
            plt.xlabel(feature_x)
            plt.ylabel("Count")
            plt.xticks(rotation=45)

            # Save plot to a temporary file and log as artifact
            with NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                plt.savefig(tmpfile.name)
                mlflow.log_artifact(tmpfile.name, f"plots/{feature_x}_vs_{feature_y}_countplot.png")
            plt.close()

            self.logger.log_info(f"Logged countplot for {feature_x} vs {feature_y}")

        except Exception as e:
            self.logger.log_error(f"Error while logging countplot {feature_x} vs {feature_y}: {e}")
            mlflow.log_param(f"{feature_x}_vs_{feature_y}_countplot_error", str(e))

    def analyze_and_log(self, data: pd.DataFrame, numerical_features: list, categorical_features: list):
        """
        Perform bivariate analysis and log results to MLFlow.
        
        Args:
            data (pd.DataFrame): Input data.
            numerical_features (list): List of numerical features to analyze.
            categorical_features (list): List of categorical features to analyze.
        """
        try:
            self.logger.log_info("Starting bivariate analysis for all features.")

            with mlflow.start_run():
                # Log heatmap for all numerical features
                self.log_heatmap(data, numerical_features)

                # Numerical vs Numerical Scatterplots
                for i in range(len(numerical_features)):
                    for j in range(i + 1, len(numerical_features)):
                        self.log_scatterplot(data, numerical_features[i], numerical_features[j])

                # Numerical vs Categorical Boxplots
                for num_feature in numerical_features:
                    for cat_feature in categorical_features:
                        self.log_boxplot(data, num_feature, cat_feature)

                # Categorical vs Categorical Countplots
                for i in range(len(categorical_features)):
                    for j in range(i + 1, len(categorical_features)):
                        self.log_countplot(data, categorical_features[i], categorical_features[j])

            self.logger.log_info("Completed bivariate analysis for all features.")

        except Exception as e:
            self.logger.log_error(f"Error while performing bivariate analysis: {e}")
            mlflow.log_param("bivariate_analysis_error", str(e))


# This block ensures the code below only runs if this script is executed directly
if __name__ == "__main__":
    # Load the dataframe (your raw data)
    df = pd.read_csv('./data/raw_data.csv')

    # Initialize BivariateAnalyzer
    analyzer = BivariateAnalyzer()

    # Specify numerical and categorical features
    numerical_features = ["Price", "Levy"]
    categorical_features = ["Manufacturer", "Fuel type", "Gear box type"]

    # Perform bivariate analysis and log results
    analyzer.analyze_and_log(df, numerical_features, categorical_features)
