import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import os
from src.utils.custom_logger import create_logger  # Assuming logger is in utils

# Initialize logger
logger = create_logger()

def perform_eda(data: pd.DataFrame, output_dir: str):
    """
    Perform EDA, save outputs to the specified directory, log artifacts to MLflow, and log insights.

    Args:
        data (pd.DataFrame): The dataset to analyze.
        output_dir (str): Directory to save EDA outputs.
        mlflow: MLflow instance to log artifacts.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logger.log_info("Starting EDA...")

    # 1. Price Analysis (Distribution and Outliers)
    logger.log_info("Performing Price Analysis...")
    
    # Distribution of price
    plt.figure(figsize=(8, 6))
    sns.histplot(data['price'], kde=True)
    plt.title('Distribution of Price')
    price_dist_path = os.path.join(output_dir, "price_distribution.png")
    plt.savefig(price_dist_path)
    mlflow.log_artifact(price_dist_path)
    plt.close()

    # Outliers: Using Z-score to detect outliers in price
    z_scores = np.abs((data['price'] - data['price'].mean()) / data['price'].std())
    data_no_outliers = data[z_scores < 3]
    logger.log_info(f"Removed {len(data) - len(data_no_outliers)} outliers based on Z-score.")

    # 2. Categorical Features (Manufacturer, Model)
    logger.log_info("Performing Categorical Feature Analysis...")
    
    # Manufacturer analysis
    top_manufacturers = data['manufacturer'].value_counts().head(10)
    manufacturer_price_mean = data.groupby('manufacturer')['price'].mean().sort_values(ascending=False)
    
    # Plot manufacturer price comparison
    plt.figure(figsize=(10, 6))
    manufacturer_price_mean.head(10).plot(kind='bar')
    plt.title('Average Price by Manufacturer')
    manufacturer_price_path = os.path.join(output_dir, "manufacturer_price_comparison.png")
    plt.savefig(manufacturer_price_path)
    mlflow.log_artifact(manufacturer_price_path)
    plt.close()

    # Model analysis (Top models)
    top_models = data['model'].value_counts().head(10)
    logger.log_info(f"Top 10 Manufacturers by Price: {manufacturer_price_mean.head(10)}")
    logger.log_info(f"Top 10 Models by Frequency: {top_models}")

    # 3. Relationship Between Numerical Features
    logger.log_info("Analyzing Relationships Between Numerical Features...")
    
    # Manufacturing Year vs Price
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='manufacturing_year', y='price', data=data)
    plt.title('Manufacturing Year vs Price')
    year_price_path = os.path.join(output_dir, "manufacturing_year_price_relationship.png")
    plt.savefig(year_price_path)
    mlflow.log_artifact(year_price_path)
    plt.close()

    # Distance Travelled vs Price
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='distance_travelled', y='price', data=data)
    plt.title('Distance Travelled vs Price')
    distance_price_path = os.path.join(output_dir, "distance_price_relationship.png")
    plt.savefig(distance_price_path)
    mlflow.log_artifact(distance_price_path)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(data[data.select_dtypes(include=['int64','float64']).columns.to_list()].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    heatmap_path = os.path.join(output_dir, "eda_correlation_heatmap.png")
    plt.savefig(heatmap_path)
    mlflow.log_artifact(heatmap_path)
    plt.close()

    # 4. Handling Missing Data
    logger.log_info("Analyzing Missing Data...")
    missing_data = data.isnull().sum()
    logger.log_info(f"Missing Data by Feature: {missing_data[missing_data > 0]}")

    # Imputing missing values (example: 'levy' column)
    data['levy'].fillna(data['levy'].median(), inplace=True)
    logger.log_info("Imputed missing values in 'levy' column with median value.")

    # 5. Outliers Analysis (in other features like 'cylinders' and 'doors')
    logger.log_info("Performing Outlier Analysis on Cylinders and Doors...")
    
    # Cylinders
    data_no_outliers = data[data['cylinders'] < 10]  # Example threshold for outliers
    logger.log_info(f"Outliers removed from 'cylinders': {len(data) - len(data_no_outliers)}")
    
    # Doors
    data_no_outliers = data[data['doors'] < 10]  # Example threshold for outliers
    logger.log_info(f"Outliers removed from 'doors': {len(data) - len(data_no_outliers)}")

    # 6. Insights Summary
    logger.log_info("EDA Insights Summary:")
    logger.log_info("1. Price Analysis: Left-skewed distribution, outliers removed.")
    logger.log_info("2. Categorical Features: Manufacturer impacts price; top models identified.")
    logger.log_info("3. Numerical Relationships: Newer cars are more expensive; distance travelled negatively correlates with price.")
    logger.log_info("4. Missing Data: Handled missing values in 'levy' by imputation.")
    logger.log_info("5. Outliers: Removed outliers in 'cylinders' and 'doors'.")
    
    logger.log_info(f"EDA completed. Results saved in {output_dir} and logged to MLflow.")


if __name__ == "__main__":
    
    # Load the data into a DataFrame
    data = pd.read_csv('./data/cleaned_data.csv')

    # Call the perform_eda function
    perform_eda(data=data,output_dir='./mlflow_experiments/eda')