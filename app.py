import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
API_KEY = "9c6f06d4d8af4a52e743d4cd5a39425c"  # Replace with your OpenWeather API Key
ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall"

# ---------- HELPER FUNCTIONS ----------

# Get city coordinates using Geopy
def get_coordinates(city):
    geolocator = Nominatim(user_agent="weatherApp")
    location = geolocator.geocode(city)
    if location:
        return location.latitude, location.longitude
    return None, None

# Fetch live weather data for the given city
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
            if 'temp' in day:
                weather_records.append({
                    "date": datetime.fromtimestamp(day["dt"]).date(),
                    "temp": round(day["temp"]["day"], 2),
                    "humidity": round(day.get("humidity", "Data unavailable"), 2),
                    "wind_speed": round(day.get("wind_speed", "Data unavailable"), 2),
                    "pressure": round(day.get("pressure", "Data unavailable"), 2),
                    "precipitation": round(day.get("pop", "Data unavailable"), 2)
                })
    return weather_records

# Train multiple models and select the best one based on Mean Squared Error (MSE)
def train_models(df):
    # Prepare data
    X = df[['humidity', 'wind_speed', 'pressure', 'precipitation']].values
    y = df['temp'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42),
        "LinearRegression": LinearRegression()
    }

    best_model = None
    best_mse = float('inf')

    # Train and evaluate models
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"{model_name} MSE: {mse:.4f}")
        
        if mse < best_mse:
            best_mse = mse
            best_model = model
    
    st.write(f"Best Model: {best_model.__class__.__name__} with MSE: {best_mse:.4f}")
    return best_model

# Predict future values
# Predict future values using ML model
def predict_with_model(model, data, scaler, days=7, window_size=5):
    data = np.array(data)
    data_scaled = scaler.transform(data.reshape(-1, 1))  # Scale the data

    # Ensure inputs are reshaped correctly (1 sample with window_size features)
    inputs = data_scaled[-window_size:].reshape(1, -1)  # Correct reshaping for model input

    predictions = []
    for _ in range(days):
        prediction = model.predict(inputs)  # Make a prediction
        predictions.append(prediction[0])  # Get the predicted value
        inputs = np.append(inputs[:, 1:], prediction.reshape(1, 1), axis=1)  # Update input for next prediction

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))  # Rescale the predictions back
    return predictions

# ---------- STREAMLIT APP LAYOUT ----------

st.set_page_config(page_title="Advanced Weather Forecasting", layout="wide")
st.title("🌍 Advanced Weather Forecast with ML and Multiple Algorithms")

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

    # Ensure weather data is available before proceeding
    if not weather_data:
        st.error("No weather data available for the selected city.")
    else:
        # Prepare the data
        df = pd.DataFrame(weather_data)

        # Train multiple models and select the best one
        best_model = train_models(df)

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = scaler.fit_transform(df[['temp']].values)

        # Predict future temperatures using the best model
        future_preds = predict_with_model(best_model, df['temp'].values, scaler, days=days_to_predict)

        # Convert to Fahrenheit if needed
        if temp_unit == "Fahrenheit (°F)":
            future_preds = future_preds * 9/5 + 32

        # Display forecasted temperatures
        forecast_table = pd.DataFrame({
            "City": [city] * days_to_predict,
            "Date": df["date"].values[:days_to_predict],
            "Predicted Temperature": [round(temp, 2) for temp in future_preds.flatten()]
        })
        st.subheader("📊 Forecasted Temperatures")
        st.table(forecast_table)

        # Plot weather data for the city
        if st.button("🔄 Display Temperature Comparison"):
            with st.spinner("Fetching data and training model..."):
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df["date"], df["temp"], label=f"{city} - Actual", color="blue")
                ax.plot(df["date"][:days_to_predict], future_preds.flatten(), label="Predicted Temp", color="red", linestyle="--")
                ax.set_ylabel("Temperature (°C)")
                ax.set_title(f"Temperature Forecast for {city}")
                ax.legend()
                st.pyplot(fig)

            # AI Summary for Today and Tomorrow
            ai_data = fetch_ai_summary(lat, lon)

            st.subheader(f"🧠 AI-Powered Weather Summary for {city}")
            st.write(f"**{city} Today**: {ai_data['today']}")
            st.write(f"**{city} Tomorrow**: {ai_data['tomorrow']}")

else:
    st.error("❌ City not found. Please check the city name and try again.")
