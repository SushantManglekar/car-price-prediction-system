import os
import pandas as pd
import numpy as np
import mlflow
from src.utils.custom_logger import create_logger  # Assuming logger is in utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from src.utils.feature_encoder import perform_feature_encoding, pickle_encoder
from datetime import datetime

# Initialize logger
logger = create_logger()


def perform_feature_engineering(data: pd.DataFrame):
    """
    Perform feature engineering, log artifacts to MLflow, and log insights.

    Args:
        data (pd.DataFrame): The dataset to process.
    """
    logger.log_info("Starting feature engineering...")

    try:
        with mlflow.start_run(nested=True):

            # Step 1. Data Cleaning
            logger.log_info("Step 1: Data Cleaning...")
            cleaned_data = data.dropna()  # Example cleaning step
            logger.log_info(f"{len(data) - len(cleaned_data)} Null values removed.")
            mlflow.log_param("Null values removed", len(data) - len(cleaned_data))

            # Remove duplicates
            duplicate_rows = cleaned_data.duplicated().sum()
            cleaned_data.drop_duplicates(inplace=True)
            logger.log_info(f"Duplicate rows removed: {duplicate_rows}")
            mlflow.log_param("Duplicate rows removed", duplicate_rows)

            # Standardize Feature Names
            cleaned_data.rename(
                columns={'Prod. year': 'manufacturing_year', 'Mileage': 'distance_travelled', 'Wheel': 'drive_type'},
                inplace=True,
            )
            cleaned_data.columns = [column.replace(" ", "_").lower() for column in cleaned_data.columns]

            # Replace '-' with an empty string, then convert to numeric
            cleaned_data['levy'] = cleaned_data['levy'].astype(str).replace('-', None)
            cleaned_data['levy'] = pd.to_numeric(cleaned_data['levy'], errors='coerce')
            cleaned_data['levy'] = cleaned_data['levy'].fillna(cleaned_data['levy'].median()).astype('int64')

            # Convert manufacturing_year to DateTime
            cleaned_data['manufacturing_year'] = pd.to_datetime(cleaned_data['manufacturing_year'], format='%Y')

            # Replacing values and converting dtype to int64
            cleaned_data['doors'] = cleaned_data['doors'].replace({'04-May': '4', '02-Mar': '2', '>5': '6'})
            cleaned_data['doors'] = cleaned_data['doors'].astype(np.int64)

            # Remove 'km' and convert to int64
            cleaned_data['distance_travelled'] = cleaned_data['distance_travelled'].str.replace(' km', '', regex=False).astype('int64')

            # Step 2. Split Data
            logger.log_info("Step 2: Splitting data into train and test sets...")
            train_data, test_data = train_test_split(cleaned_data, test_size=0.2, random_state=42)
            logger.log_info(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
            mlflow.log_param("train_size", len(train_data))
            mlflow.log_param("test_size", len(test_data))

            # Step 3. Handle Outliers
            logger.log_info("Step 3: Handling outliers...")
            Q1 = cleaned_data['price'].quantile(0.25)
            Q3 = cleaned_data['price'].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_removed = len(cleaned_data) - len(
                cleaned_data[(cleaned_data['price'] >= lower_bound) & (cleaned_data['price'] <= upper_bound)]
            )
            cleaned_data = cleaned_data[(cleaned_data['price'] >= lower_bound) & (cleaned_data['price'] <= upper_bound)]

            logger.log_info(f"{outliers_removed} outliers removed from price.")
            mlflow.log_param("price_outliers_removed", outliers_removed)

            # Step 4. Feature Creation
            logger.log_info("Step 4: Creating new features...")
            cleaned_data['turbo'] = cleaned_data['engine_volume'].str.contains('Turbo', case=False, na=False).astype(int)
            cleaned_data['engine_volume'] = cleaned_data['engine_volume'].str.replace('Turbo', '', case=False).astype(float)

            current_year = datetime.now().year
            cleaned_data['car_age'] = current_year - cleaned_data['manufacturing_year'].dt.year
            logger.log_info("Feature 'turbo' and 'car_age' created.")
            mlflow.log_param("features_created", ["car_age", "turbo"])

            # Step 5. Feature Encoding
            logger.log_info("Step 5: Encoding categorical features...")
            train_data, test_data = perform_feature_encoding(cleaned_data)
            logger.log_info("Categorical features encoded.")

            # Step 6. Feature Transformation
            logger.log_info("Step 6: Transforming features...")
            scaler = StandardScaler()
            columns_to_scale = [
                'engine_volume', 'levy', 'cylinders', 'doors', 'airbags', 'car_age',
                'manufacturer_encoded', 'model_encoded', 'category_encoded'
            ]
            train_data[columns_to_scale] = scaler.fit_transform(train_data[columns_to_scale])
            test_data[columns_to_scale] = scaler.transform(test_data[columns_to_scale])
            logger.log_info("Numerical features standardized.")
            mlflow.log_artifact(pickle_encoder(scaler, "scaler.pkl"))

            # Drop unnecessary columns
            columns_to_drop = [
                'id', 'manufacturer', 'model', 'manufacturing_year', 'category', 'leather_interior',
                'fuel_type', 'distance_travelled', 'gear_box_type', 'color', 'category_grouped', 'fuel_type_grouped'
            ]
            train_data.drop(columns=columns_to_drop, inplace=True)
            test_data.drop(columns=columns_to_drop, inplace=True)
            
            # # Step 7. Feature Selection
            # logger.log_info("Step 7: Selecting features")
            train_data.dropna(inplace=True)
            test_data.dropna(inplace=True)
            
            # X_train = train_data.drop(columns=["price"])
            # y_train = train_data["price"]
            # selector = SelectKBest(score_func=f_regression, k=10)
            # X_train_selected = selector.fit_transform(X_train, y_train)
            # selected_features = X_train.columns[selector.get_support()].tolist()

            # X_test = test_data.drop(columns=["price"])
            # X_test_selected = selector.transform(X_test)
            # test_data = test_data[selected_features + ["price"]]
            # train_data = train_data[selected_features + ["price"]]

            # logger.log_info(f"Selected features: {selected_features}")
            # mlflow.log_param("selected_features", selected_features)

            # Save processed data
            train_data.to_csv("./data/train_data.csv", index=False)
            test_data.to_csv("./data/test_data.csv", index=False)

        logger.log_info("Feature engineering completed. Results logged to MLflow and data saved.")

    except Exception as e:
        logger.log_error(f"Error during feature engineering: {str(e)}")
        mlflow.log_param("error", str(e))
        raise e  # Re-raise the error to propagate it

if __name__ == "__main__":
    try:
        data = pd.read_csv('./data/raw_data.csv')
        perform_feature_engineering(data)
    except Exception as e:
        logger.log_error(f"Error in main execution: {str(e)}")
