import React, { useState } from "react";
import axios from "axios";

const PredictionForm = () => {
  const [formData, setFormData] = useState({
    temperature: "",
    humidity: "",
    hour: "",
    weekday: "",
  });

  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData);
      setPrediction(response.data.predicted_electricity_demand);
    } catch (err) {
      setError("Error fetching prediction. Ensure Flask API is running.");
    }
  };

  return (
    <div style={{ maxWidth: "400px", margin: "auto", textAlign: "center" }}>
      <h2>Electricity Demand Prediction</h2>
      <form onSubmit={handleSubmit}>
        <input type="number" name="temperature" placeholder="Temperature (°C)" onChange={handleChange} required /><br />
        <input type="number" name="humidity" placeholder="Humidity (%)" onChange={handleChange} required /><br />
        <input type="number" name="hour" placeholder="Hour (0-23)" onChange={handleChange} required /><br />
        <input type="number" name="weekday" placeholder="Weekday (0=Monday, 6=Sunday)" onChange={handleChange} required /><br />
        <button type="submit">Predict</button>
      </form>
      {prediction !== null && <h3>Predicted Demand: {prediction} kWh</h3>}
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
};

export default PredictionForm;
