import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib  # For saving scalers

# Generate & Save Data (Same as previous script)
np.random.seed(42)
tf.random.set_seed(42)

date_rng = pd.date_range(start="2024-01-01", end="2024-12-31", freq='H')

base_demand = np.sin(np.linspace(0, 12*np.pi, len(date_rng))) * 50 + 200
random_noise = np.random.normal(0, 15, len(date_rng))
electricity_demand = base_demand + random_noise

temperature = np.sin(np.linspace(0, 8*np.pi, len(date_rng))) * 10 + 25 + np.random.normal(0, 2, len(date_rng))
humidity = np.cos(np.linspace(0, 8*np.pi, len(date_rng))) * 10 + 50 + np.random.normal(0, 5, len(date_rng))

df = pd.DataFrame({
    'timestamp': date_rng,
    'electricity_demand': electricity_demand,
    'temperature': temperature,
    'humidity': humidity
})

df['hour'] = df['timestamp'].dt.hour
df['weekday'] = df['timestamp'].dt.weekday

df.to_csv("synthetic_electricity_demand.csv", index=False)

# Load & Normalize Data
df = pd.read_csv("synthetic_electricity_demand.csv", parse_dates=['timestamp'])
features = ['temperature', 'humidity', 'hour', 'weekday']
target = 'electricity_demand'

scaler = MinMaxScaler()
df[features + [target]] = scaler.fit_transform(df[features + [target]])

joblib.dump(scaler, "scaler.pkl")  # Save the scaler

def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, -1])  
    return np.array(X), np.array(y)

data_array = df[features + [target]].values
X, y = create_sequences(data_array)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation="relu"),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=1)

model.save("lstm_electricity_model.h5")  # Save model

# Make Predictions
y_pred = model.predict(X_test)

y_test_rescaled = scaler.inverse_transform(np.hstack((X_test[:, -1, :-1], y_test.reshape(-1, 1))))[:, -1]
y_pred_rescaled = scaler.inverse_transform(np.hstack((X_test[:, -1, :-1], y_pred.reshape(-1, 1))))[:, -1]

plt.figure(figsize=(12, 5))
plt.plot(y_test_rescaled, label="Actual Demand", color="blue", alpha=0.6)
plt.plot(y_pred_rescaled, label="Predicted Demand", color="red", linestyle="dashed", alpha=0.7)
plt.xlabel("Time Steps")
plt.ylabel("Electricity Demand (kWh)")
plt.title("Actual vs Predicted Electricity Demand")
plt.legend()
plt.show()
