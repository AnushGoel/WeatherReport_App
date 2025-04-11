import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM

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

# Machine Learning: LSTM Model for Weather Prediction
def create_lstm_model(data, window_size=5):
    data = np.array(data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i-window_size:i, 0])
        y.append(data_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # LSTM expects 3D input

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32)

    return model, scaler

# Predict next 'n' days using LSTM model
def predict_with_lstm(model, data, scaler, days=7, window_size=5):
    data = np.array(data)
    data_scaled = scaler.transform(data.reshape(-1, 1))

    inputs = data_scaled[-window_size:].reshape(1, window_size, 1)
    predictions = []
    for _ in range(days):
        prediction = model.predict(inputs)
        predictions.append(prediction[0][0])
        inputs = np.append(inputs[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# ---------- STREAMLIT APP LAYOUT ----------

st.set_page_config(page_title="Professional Weather App", layout="wide")
st.title("🌍 Professional Weather Forecast with ML and Interactive Features")

# User input: Cities, number of days to predict, temperature units
city1 = st.text_input("Enter the first city name:", value="Toronto")
city2 = st.text_input("Enter the second city name:", value="Vancouver")
days_to_predict = st.slider("Select number of days to predict:", min_value=1, max_value=14, value=7)
temp_unit = st.radio("Select temperature unit:", ("Celsius (°C)", "Fahrenheit (°F)"))

# Fetch coordinates for both cities
lat1, lon1 = get_coordinates(city1)
lat2, lon2 = get_coordinates(city2)

if lat1 and lon1 and lat2 and lon2:
    st.success(f"📍 Found locations: {city1} (Lat: {lat1}, Lon: {lon1}), {city2} (Lat: {lat2}, Lon: {lon2})")

    # Get weather data for both cities
    weather_data1 = fetch_weather_data(lat1, lon1, days=days_to_predict)
    weather_data2 = fetch_weather_data(lat2, lon2, days=days_to_predict)

    # Train LSTM model for both cities
    df1 = pd.DataFrame(weather_data1)
    df2 = pd.DataFrame(weather_data2)
    model1, scaler1 = create_lstm_model(df1['temp'].values)
    model2, scaler2 = create_lstm_model(df2['temp'].values)

    # Predict next 'n' days for both cities
    future_preds1 = predict_with_lstm(model1, df1['temp'].values, scaler1, days=days_to_predict)
    future_preds2 = predict_with_lstm(model2, df2['temp'].values, scaler2, days=days_to_predict)

    # Convert to Fahrenheit if needed
    if temp_unit == "Fahrenheit (°F)":
        future_preds1 = future_preds1 * 9/5 + 32
        future_preds2 = future_preds2 * 9/5 + 32

    # Display table of cities and their forecasted temperatures
    forecast_table = pd.DataFrame({
        "City": [city1] * days_to_predict + [city2] * days_to_predict,
        "Date": np.concatenate([df1["date"].values, df2["date"].values]),
        "Predicted Temperature": np.concatenate([future_preds1.flatten(), future_preds2.flatten()])
    })
    st.subheader("📊 Forecasted Temperatures")
    st.table(forecast_table)

    # Plot weather data for both cities
    if st.button("🔄 Compare Weather Data"):
        with st.spinner("Fetching weather data and training model..."):
            # Plot comparison graph
            plot_temperature_comparison(df1, df2, city1, city2, df1['date'], future_preds1)

    # AI Summary for Today and Tomorrow
    ai_data1 = fetch_ai_summary(lat1, lon1)
    ai_data2 = fetch_ai_summary(lat2, lon2)

    st.subheader(f"🧠 AI-Powered Weather Summary for {city1} and {city2}")
    st.write(f"**{city1} Today**: {ai_data1['today']}")
    st.write(f"**{city1} Tomorrow**: {ai_data1['tomorrow']}")
    st.write(f"**{city2} Today**: {ai_data2['today']}")
    st.write(f"**{city2} Tomorrow**: {ai_data2['tomorrow']}")

    # Display Weather Alerts
    weather_data1_alerts = fetch_weather_data(lat1, lon1)
    weather_data2_alerts = fetch_weather_data(lat2, lon2)
    if 'alerts' in weather_data1_alerts:
        st.subheader(f"⚠️ Weather Alerts for {city1}")
        for alert in weather_data1_alerts['alerts']:
            st.warning(f"{alert['event']} in {alert['sender_name']}")

    if 'alerts' in weather_data2_alerts:
        st.subheader(f"⚠️ Weather Alerts for {city2}")
        for alert in weather_data2_alerts['alerts']:
            st.warning(f"{alert['event']} in {alert['sender_name']}")
else:
    st.error("❌ One or both cities not found. Please check the city names and try again.")
