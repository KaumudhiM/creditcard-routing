from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the trained model and preprocessing objects
model = joblib.load('models/accurate_model.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Define the feature columns used during training
feature_columns = [
    'amount', '3D_secured', 'transaction_fee', 'hour_of_day', 'day_of_week',
    'retry_count', 'is_retry', 'is_first_attempt', 'time_since_first_attempt',
    'country_Germany', 'country_Switzerland', 'card_Master', 'card_Visa'
]

# Define the fee structure
FEE_STRUCTURE = {
    'Moneycard': {'success': 5, 'fail': 2},
    'Goldcard': {'success': 10, 'fail': 5},
    'UK_Card': {'success': 3, 'fail': 1},
    'Simplecard': {'success': 1, 'fail': 0.5}
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Extract features from the request
    country = data['country']
    amount = data['amount']
    secured_3d = data['3D_secured']
    card = data['card']

    # Get current timestamp
    current_timestamp = datetime.now()
    hour_of_day = current_timestamp.hour
    day_of_week = current_timestamp.weekday()

    # Calculate transaction fee based on card type
    transaction_fee = FEE_STRUCTURE.get(card, {'success': 0})['success']

    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'country': [country],
        'amount': [amount],
        '3D_secured': [secured_3d],
        'card': [card],
        'transaction_fee': [transaction_fee],
        'hour_of_day': [hour_of_day],
        'day_of_week': [day_of_week],
        'retry_count': [0],  # Assuming a default value
        'is_retry': [0],  # Assuming a default value
        'is_first_attempt': [1],  # Assuming a default value
        'time_since_first_attempt': [0]  # Assuming a default value
    })

    # One-hot encode categorical variables
    input_data_encoded = pd.get_dummies(input_data, columns=['country', 'card'], drop_first=True)

    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0

    # Reorder columns to match the training data
    input_data_encoded = input_data_encoded[feature_columns]

    # Scale the input data
    input_data_scaled = scaler.transform(input_data_encoded)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    predicted_psp = label_encoder.inverse_transform(prediction)

    # Return the prediction as a JSON response
    return jsonify({'recommended_PSP': predicted_psp[0]})

if __name__ == '__main__':
    app.run(debug=True)