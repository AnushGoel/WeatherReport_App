import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor

# ---------- CONFIG ----------
API_KEY = "9c6f06d4d8af4a52e743d4cd5a39425c"  # Replace with your OpenWeather API Key
GEO_URL = "http://api.openweathermap.org/geo/1.0/direct"
ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall"

# ---------- HELPER FUNCTIONS ----------

# Get city coordinates using Geopy
def get_coordinates(city):
    geolocator = Nominatim(user_agent="weatherApp")
    location = geolocator.geocode(city)
    if location:
        return location.latitude, location.longitude
    return None, None

# Fetch live weather and alerts
def fetch_weather_data(lat, lon, days=7):
    params = {
        "lat": lat, "lon": lon, "appid": API_KEY,
        "units": "metric", "exclude": "minutely,hourly,alerts"
    }
    response = requests.get(ONECALL_URL, params=params)
    data = response.json()

    weather_records = []
    if 'daily' in data:
        for day in data['daily'][:days]:
            weather_records.append({
                "date": datetime.fromtimestamp(day["dt"]).date(),
                "temp": day["temp"]["day"],
                "humidity": day["humidity"],
                "wind_speed": day["wind_speed"],
                "pressure": day["pressure"],
                "precipitation": day["pop"]
            })

    return weather_records

# Fetch AI-powered weather summary (for today and tomorrow)
def fetch_ai_summary(lat, lon):
    forecast_url = f"http://api.openweathermap.org/data/2.5/forecast"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric",
        "cnt": 2  # Only get the first two forecasts (today and tomorrow)
    }
    response = requests.get(forecast_url, params=params)
    if response.status_code == 200:
        data = response.json()
        today_summary = data['list'][0]['weather'][0]['description']
        tomorrow_summary = data['list'][1]['weather'][0]['description']
        return {"today": today_summary, "tomorrow": tomorrow_summary}
    else:
        return {"today": "Data unavailable", "tomorrow": "Data unavailable"}

# XGBoost model for weather prediction
def create_xgboost_model(data, window_size=5):
    data = np.array(data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i-window_size:i, 0])
        y.append(data_scaled[i, 0])

    X, y = np.array(X), np.array(y)

    # Reshaping X to ensure it's a 2D array for XGBoost
    X = X.reshape(X.shape[0], X.shape[1])

    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, scaler

# LSTM model for weather prediction
def create_lstm_model(data, window_size=5):
    data = np.array(data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i-window_size:i, 0])
        y.append(data_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # LSTM expects 3D input (samples, time steps, features)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))  # LSTM Layer 1
    model.add(LSTM(units=50, return_sequences=False))  # LSTM Layer 2
    model.add(Dense(units=1))  # Output layer (single unit for prediction)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32)

    return model, scaler

# Predict next 'n' days using ML model
def predict_with_model(model, data, scaler, days=7, window_size=5, is_lstm=False):
    data = np.array(data)
    data_scaled = scaler.transform(data.reshape(-1, 1))

    if is_lstm:
        inputs = data_scaled[-window_size:].reshape(1, window_size, 1)  # Reshape for LSTM
    else:
        inputs = data_scaled[-window_size:].reshape(1, window_size)

    predictions = []
    for _ in range(days):
        prediction = model.predict(inputs)
        predictions.append(prediction[0][0])
        inputs = np.append(inputs[:, 1:, :], prediction.reshape(1, 1, 1), axis=1) if is_lstm else np.append(inputs[:, 1:], prediction.reshape(1, 1), axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# ---------- STREAMLIT APP LAYOUT ----------

st.set_page_config(page_title="Professional Weather App", layout="wide")
st.title("🌍 Professional Weather Forecast with ML and Interactive Features")

# User input: City, number of days to predict, temperature units
city = st.text_input("Enter the city name:", value="Toronto")
days_to_predict = st.slider("Select number of days to predict:", min_value=1, max_value=14, value=7)
temp_unit = st.radio("Select temperature unit:", ("Celsius (°C)", "Fahrenheit (°F)"))

# Fetch coordinates for the city
lat, lon = get_coordinates(city)

if lat and lon:
    st.success(f"📍 Found location: {city} (Lat: {lat}, Lon: {lon})")

    # Get weather data for the city
    weather_data = fetch_weather_data(lat, lon, days=days_to_predict)

    # Train models for the city
    df = pd.DataFrame(weather_data)
    model_lstm, scaler_lstm = create_lstm_model(df['temp'].values)
    model_xgb, scaler_xgb = create_xgboost_model(df['temp'].values)

    # Predict next 'n' days for the city
    future_preds_lstm = predict_with_model(model_lstm, df['temp'].values, scaler_lstm, days=days_to_predict, is_lstm=True)
    future_preds_xgb = predict_with_model(model_xgb, df['temp'].values, scaler_xgb, days=days_to_predict, is_lstm=False)

    # Convert to Fahrenheit if needed
    if temp_unit == "Fahrenheit (°F)":
        future_preds_lstm = future_preds_lstm * 9/5 + 32
        future_preds_xgb = future_preds_xgb * 9/5 + 32

    # Display table of the city's forecasted temperatures
    forecast_table = pd.DataFrame({
        "City": [city] * days_to_predict,
        "Date": df["date"].values[:days_to_predict],
        "LSTM Predicted Temperature": future_preds_lstm.flatten(),
        "XGBoost Predicted Temperature": future_preds_xgb.flatten()
    })
    st.subheader("📊 Forecasted Temperatures")
    st.table(forecast_table)

    # Plot weather data for the city
    if st.button("🔄 Display Temperature Comparison"):
        with st.spinner("Fetching data and training model..."):
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df["date"], df["temp"], label=f"{city} - Actual", color="blue")
            ax.plot(df["date"][:days_to_predict], future_preds_lstm.flatten(), label="LSTM Predicted Temp", color="red", linestyle="--")
            ax.plot(df["date"][:days_to_predict], future_preds_xgb.flatten(), label="XGBoost Predicted Temp", color="green", linestyle="--")
            ax.set_ylabel("Temperature (°C)")
            ax.set_title(f"Temperature Forecast for {city}")
            ax.legend()
            st.pyplot(fig)

    # AI Summary for Today and Tomorrow
    ai_data = fetch_ai_summary(lat, lon)

    st.subheader(f"🧠 AI-Powered Weather Summary for {city}")
    st.write(f"**{city} Today**: {ai_data['today']}")
    st.write(f"**{city} Tomorrow**: {ai_data['tomorrow']}")

    # Display Weather Alerts
    weather_data_alerts = fetch_weather_data(lat, lon)
    if 'alerts' in weather_data_alerts:
        st.subheader(f"⚠️ Weather Alerts for {city}")
        for alert in weather_data_alerts['alerts']:
            st.warning(f"{alert['event']} in {alert['sender_name']}")
else:
    st.error("❌ City not found. Please check the city name and try again.")
