from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import joblib
import mlflow.pyfunc
import subprocess

app = Flask(__name__)

# Load models and encoders
data = pd.read_csv("./data/raw_data.csv")
distance_travelled_encoder = joblib.load('./models/distance_travelled_encoder.pkl')
scaler = joblib.load('./models/scaler.pkl')  # Replace with your scaler path

model = joblib.load('./models/xgboost_model.pkl')
# Configuration
MODEL_NAME = "BestCarPricePredictionModel"  # Name of your model in MLflow
STAGE = "Production"  # Stage to load the model from

# Load the latest production model from MLflow
def load_model():
    """
    Load the latest production model from MLflow Model Registry.
    """
    global model
    try:
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{STAGE}")
    except Exception as e:
     
        raise RuntimeError(f"Failed to load model: {e}")
    
# Trigger training pipeline (optional)
@app.route("/train", methods=["POST"])
def train_model():
    """
    Trigger the model training pipeline.
    """
    try:
        # Run the training pipeline script
        subprocess.run(["python", "src/pipeline/model_training.py"],
               check=True)
        return jsonify({"message": "Model training pipeline triggered successfully!"})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Training failed: {e}"}), 500


# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    """
    Health check for the API.
    """
    return jsonify({"status": "OK"}), 200

def preprocess_input(user_input):
    """
    Preprocess the user input data and prepare it for prediction.
    """
    # Encode distance travelled
 
    distance_travelled = ''
    if user_input["distance_travelled"] <= 50000:
        distance_travelled = 'low'
    elif user_input["distance_travelled"] <= 100000:
        distance_travelled = 'medium'
    else:
        distance_travelled = 'high'

    distance_travelled_group = distance_travelled_encoder.transform(
        [distance_travelled]
    )[0]

    # Encode manufacturer
    manufacturer_price_mean = data.groupby('Manufacturer')['Price'].mean()
    manufacturer_encoded = manufacturer_price_mean.get(
        user_input["manufacturer"], manufacturer_price_mean.mean()
    )
    manufacturer_encoded = manufacturer_encoded
    
    # Encode model
    model_price_mean = data.groupby('Model')['Price'].mean()
    model_encoded = model_price_mean.get(user_input['model'],model_price_mean)

    # Combine all inputs
    processed_input = {
        "levy": user_input.get("levy", 0),  # Default levy to 0 if not provided
        "engine_volume": user_input["engine_volume"],
        "cylinders": user_input["cylinders"],
        "doors": user_input["doors"],
        "airbags": user_input["airbags"],
        "turbo": user_input["turbo"],
        "car_age": user_input["car_age"],
        "distance_travelled_group": distance_travelled_group,
        "manufacturer_encoded": manufacturer_encoded,
        "model_encoded": model_encoded,  # Assuming categorical encoding is done
        "category_encoded": 1.257 ,  # Assuming categorical encoding is done
        "leather_interior_encoded": user_input["leather_interior"],
        "fuel_type_encoded": 1,
        "drive_wheels_Front": int(user_input["drive_wheels"] == "Front"),
        "drive_wheels_Rear": int(user_input["drive_wheels"] == "Rear"),
        "drive_type_Right-hand drive": int(user_input.get("drive_type", "Left-hand drive") == "Right-hand drive"),
        "gear_box_type_encoded": user_input["gear_box_type"],  # Assuming categorical encoding is done
        "color_encoded": 0.025  # Assuming categorical encoding is done
    }

    return pd.DataFrame([processed_input])

@app.route('/predict', methods=['POST'])
def predict_price():
    """
    Predict car price based on user input.
    """
    # Get user input from the POST request
    user_input = request.json

    # Preprocess user input
    input_df = preprocess_input(user_input)

    # Predict the price
    predicted_price = model.predict(input_df)

    # Return the predicted price
    return jsonify({"predicted_price": f'${round(predicted_price[0])}'})


if __name__ == '__main__':
    load_model()
        
    app.run(host="0.0.0.0", port=5000)
