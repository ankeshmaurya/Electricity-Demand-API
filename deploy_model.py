from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib

# Load trained model and scaler
model = tf.keras.models.load_model("lstm_electricity_model.h5")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        temp = float(data['temperature'])
        humidity = float(data['humidity'])
        hour = int(data['hour'])
        weekday = int(data['weekday'])
        
        input_data = np.array([[temp, humidity, hour, weekday]])
        input_scaled = scaler.transform(input_data)

        input_reshaped = np.reshape(input_scaled, (1, 1, 4))  # Reshape for LSTM
        prediction = model.predict(input_reshaped)

        predicted_demand = scaler.inverse_transform(
            np.hstack((input_scaled[:, :-1], prediction))
        )[:, -1][0]

        return jsonify({'predicted_electricity_demand': round(predicted_demand, 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
