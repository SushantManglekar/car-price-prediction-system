import os
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

from src.utils.custom_logger import create_logger  # Assuming logger is in utils

# Initialize logger
logger = create_logger()

def perform_feature_encoding(data: pd.DataFrame):
    """
    Perform feature encoding on the dataset, save encoders using MLflow, and log artifacts.

    Args:
        data (pd.DataFrame): The dataset to process.
    """
    logger.log_info("Starting feature encoding...")

    with mlflow.start_run(nested=True):

        # Step 1: Split Data
        logger.log_info("Step 1: Splitting data into train and test sets...")
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        logger.log_info(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
        mlflow.log_param("train_size", len(train_data))
        mlflow.log_param("test_size", len(test_data))

        # Step 2: Distance Travelled Binning
        logger.log_info("Step 2: Binning 'distance_travelled'...")
        labels = ['low', 'medium', 'high']
        train_data['distance_travelled_group'] = pd.qcut(
            train_data['distance_travelled'], q=3, labels=labels, duplicates='drop'
        )
        test_data['distance_travelled_group'] = pd.qcut(
            test_data['distance_travelled'], q=3, labels=labels, duplicates='drop'
        ) 
        distance_travelled_encoder = LabelEncoder()
        train_data['distance_travelled_group'] = distance_travelled_encoder.fit_transform(
            train_data['distance_travelled_group']
        )
        test_data['distance_travelled_group'] = distance_travelled_encoder.transform(
            test_data['distance_travelled_group']
        )
        logger.log_info("Encoded 'distance_travelled_group'.")
        mlflow.log_artifact(pickle_encoder(distance_travelled_encoder, "distance_travelled_encoder.pkl"))

        # Step 3: Manufacturer Encoding
        logger.log_info("Step 3: Encoding 'manufacturer' using target-based encoding...")
        manufacturer_price_mean = train_data.groupby('manufacturer')['price'].mean()
        train_data['manufacturer_encoded'] = train_data['manufacturer'].map(manufacturer_price_mean)
        test_data['manufacturer_encoded'] = test_data['manufacturer'].map(manufacturer_price_mean)

        # Step 4: Model Encoding
        logger.log_info("Step 4: Encoding 'model' using target-based encoding...")
        model_price_mean = train_data.groupby('model')['price'].mean()
        train_data['model_encoded'] = train_data['model'].map(model_price_mean)
        test_data['model_encoded'] = test_data['model'].map(model_price_mean)

        # Step 5: Group Rare Categories in 'Category'
        logger.log_info("Step 5: Grouping rare categories in 'category'...")
        category_threshold = 300
        train_data['category_grouped'] = train_data['category'].apply(
            lambda x: x if train_data['category'].value_counts()[x] >= category_threshold else 'Other'
        )
        test_data['category_grouped'] = test_data['category'].apply(
            lambda x: x if train_data['category'].value_counts()[x] >= category_threshold else 'Other'
        )
        category_price_mean = train_data.groupby('category_grouped')['price'].mean()
        train_data['category_encoded'] = train_data['category_grouped'].map(category_price_mean)
        test_data['category_encoded'] = test_data['category_grouped'].map(category_price_mean)

        # Step 6: Encode 'leather_interior'
        logger.log_info("Step 6: Encoding 'leather_interior' as binary...")
        train_data['leather_interior_encoded'] = train_data['leather_interior'].map({'Yes': 1, 'No': 0})
        test_data['leather_interior_encoded'] = test_data['leather_interior'].map({'Yes': 1, 'No': 0})

        # Step 7: Group Rare Categories in 'fuel_type'
        logger.log_info("Step 7: Grouping rare categories in 'fuel_type'...")
        threshold = 100
        fuel_counts = train_data['fuel_type'].value_counts()
        train_data['fuel_type_grouped'] = train_data['fuel_type'].apply(
            lambda x: x if fuel_counts[x] >= threshold else 'Other'
        )
        test_data['fuel_type_grouped'] = test_data['fuel_type'].apply(
            lambda x: x if x in fuel_counts[fuel_counts >= threshold] else 'Other'
        )
        fuel_type_encoder = LabelEncoder()
        train_data['fuel_type_encoded'] = fuel_type_encoder.fit_transform(train_data['fuel_type_grouped'])
        test_data['fuel_type_encoded'] = fuel_type_encoder.transform(test_data['fuel_type_grouped'])
        mlflow.log_artifact(pickle_encoder(fuel_type_encoder, "fuel_type_encoder.pkl"))

        # Step 8: One-Hot Encode 'drive_wheels' and 'drive_type'
        logger.log_info("Step 8: One-hot encoding 'drive_wheels' and 'drive_type'...")
        train_data = pd.get_dummies(train_data, columns=['drive_wheels', 'drive_type'], drop_first=True)
        test_data = pd.get_dummies(test_data, columns=['drive_wheels', 'drive_type'], drop_first=True)

        # Step 9: Label Encode 'gear_box_type'
        logger.log_info("Step 9: Label encoding 'gear_box_type'...")
        gear_box_encoder = LabelEncoder()
        train_data['gear_box_type_encoded'] = gear_box_encoder.fit_transform(train_data['gear_box_type'])
        test_data['gear_box_type_encoded'] = gear_box_encoder.transform(test_data['gear_box_type'])
        mlflow.log_artifact(pickle_encoder(gear_box_encoder, "gear_box_encoder.pkl"))

        # Step 10: Group Rare Colors and Frequency Encode
        logger.log_info("Step 10: Frequency encoding 'color'...")
        threshold = 0.01  # Frequency threshold for grouping
        color_counts = train_data['color'].value_counts(normalize=True)
        rare_colors = color_counts[color_counts < threshold].index
        train_data['color'] = train_data['color'].replace(rare_colors, 'Other')
        color_frequency = train_data['color'].value_counts(normalize=True)
        train_data['color_encoded'] = train_data['color'].map(color_frequency)
        test_data['color'] = test_data['color'].replace(rare_colors, 'Other')
        test_data['color_encoded'] = test_data['color'].map(color_frequency)

        # Return train and test data
        return train_data,test_data
        

def pickle_encoder(encoder, filename):
    """
    Save an encoder to a pickle file.

    Args:
        encoder: The encoder to save.
        filename (str): The file path to save the encoder.

    Returns:
        str: The file path of the saved encoder.
    """
    filepath = f"./models/{filename}"
    os.makedirs("./models", exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(encoder, f)
    return filepath

if __name__ == "__main__":
    # Load the data into a DataFrame
    data = pd.read_csv('./data/raw_data.csv')

    # Perform feature encoding
    perform_feature_encoding(data)
