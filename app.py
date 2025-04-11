import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import numpy as np

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

# Train the ML model to predict next 'n' days based on historical data
def train_model(df):
    df['day_of_year'] = pd.to_datetime(df['date']).dt.dayofyear
    df['year'] = pd.to_datetime(df['date']).dt.year
    X = df[['day_of_year', 'year']]
    y = df['temp']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Predict next 'n' days
def predict_next_days(model, days):
    today = datetime.now().date()
    future_dates = [today + timedelta(days=i) for i in range(1, days+1)]
    df_future = pd.DataFrame({
        "day_of_year": [d.timetuple().tm_yday for d in future_dates],
        "year": [d.year for d in future_dates]
    })
    preds = model.predict(df_future)
    return future_dates, preds

# Plot a graph comparing actual and predicted temperatures for two cities
def plot_temperature_comparison(df1, df2, city1, city2, future_dates, future_preds):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df1["date"], df1["temp"], label=f"{city1} - Actual", color="blue")
    ax.plot(df2["date"], df2["temp"], label=f"{city2} - Actual", color="green")
    ax.plot(future_dates, future_preds, label="Predicted Temp", color="red", linestyle="--")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Temperature Forecast Comparison: {city1} vs {city2}")
    ax.legend()
    st.pyplot(fig)

# Plot weather data on a map (interactive map)
def plot_weather_map():
    map_data = px.data.gapminder()  # Placeholder map data (can replace with city data later)
    fig = px.scatter_geo(map_data, locations="iso_alpha", hover_name="country", size="pop", projection="natural earth")
    fig.update_layout(title="Weather by Location")
    st.plotly_chart(fig)

# ---------- STREAMLIT APP LAYOUT ----------

st.set_page_config(page_title="Complex Weather App", layout="wide")
st.title("🌍 Professional Weather Forecast with ML and Interactive Features")

# User input: Cities, number of days to predict
city1 = st.text_input("Enter the first city name:", value="Toronto")
city2 = st.text_input("Enter the second city name:", value="Vancouver")
days_to_predict = st.slider("Select number of days to predict:", min_value=1, max_value=14, value=7)

# World map to display locations
plot_weather_map()

# Fetch coordinates for both cities
lat1, lon1 = get_coordinates(city1)
lat2, lon2 = get_coordinates(city2)

if lat1 and lon1 and lat2 and lon2:
    st.success(f"📍 Found locations: {city1} (Lat: {lat1}, Lon: {lon1}), {city2} (Lat: {lat2}, Lon: {lon2})")

    # Get weather data for both cities
    weather_data1 = fetch_weather_data(lat1, lon1, days=days_to_predict)
    weather_data2 = fetch_weather_data(lat2, lon2, days=days_to_predict)

    # Plot weather data for both cities
    if st.button("🔄 Compare Weather Data"):
        with st.spinner("Fetching weather data and training model..."):
            # Train models for both cities
            df1 = pd.DataFrame(weather_data1)
            df2 = pd.DataFrame(weather_data2)
            model1 = train_model(df1)
            model2 = train_model(df2)

            # Predict next days for both cities
            future_dates1, future_preds1 = predict_next_days(model1, days_to_predict)
            future_dates2, future_preds2 = predict_next_days(model2, days_to_predict)

            # Plot comparison graph
            plot_temperature_comparison(df1, df2, city1, city2, future_dates1, future_preds1)

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
