import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim

# ---------- CONFIG ----------
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your OpenWeather API Key
GEO_URL = "http://api.openweathermap.org/geo/1.0/direct"
ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall"

# ---------- FUNCTIONS ----------
# Get city coordinates
def get_coordinates(city):
    geolocator = Nominatim(user_agent="weatherApp")
    location = geolocator.geocode(city)
    if location:
        return location.latitude, location.longitude
    return None, None

# Fetch live weather, alerts, and AI summary
def fetch_weather_data(lat, lon):
    params = {
        "lat": lat, "lon": lon, "appid": API_KEY,
        "units": "metric", "exclude": "minutely,hourly,alerts"
    }
    response = requests.get(ONECALL_URL, params=params)
    data = response.json()

    # Extract data for each day's weather
    weather_records = []
    if 'daily' in data:
        for day in data['daily']:
            weather_records.append({
                "date": datetime.fromtimestamp(day["dt"]).date(),
                "temp": day["temp"]["day"],
                "humidity": day["humidity"],
                "wind_speed": day["wind_speed"],
                "pressure": day["pressure"],
                "precipitation": day["pop"]
            })

    return weather_records

# Train ML model to predict next 7 days
def train_model(df):
    df['day_of_year'] = pd.to_datetime(df['date']).dt.dayofyear
    df['year'] = pd.to_datetime(df['date']).dt.year
    X = df[['day_of_year', 'year']]
    y = df['temp']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Predict next 7 days
def predict_next_7_days(model):
    today = datetime.now().date()
    future_dates = [today + timedelta(days=i) for i in range(1, 8)]
    df_future = pd.DataFrame({
        "day_of_year": [d.timetuple().tm_yday for d in future_dates],
        "year": [d.year for d in future_dates]
    })
    preds = model.predict(df_future)
    return future_dates, preds

# Interactive map display with Plotly
def plot_weather_map():
    map_data = px.data.gapminder()  # Placeholder map data (could replace with city data later)
    fig = px.scatter_geo(map_data, locations="iso_alpha", hover_name="country", size="pop", projection="natural earth")
    fig.update_layout(title="Click on a country for detailed weather")
    st.plotly_chart(fig)

# ---------- STREAMLIT APP LAYOUT ----------
st.set_page_config(page_title="Complex Weather App", layout="wide")
st.title("🌍 Weather Forecast with ML and Interactive Features")

city = st.text_input("Enter city name", value="Toronto")

# ---------- World Map Display --------
plot_weather_map()

if city:
    lat, lon = get_coordinates(city)
    if lat and lon:
        st.success(f"📍 Found location: {city} (Lat: {lat}, Lon: {lon})")

        # Live Weather Data
        st.subheader("📡 Current Weather")
        weather_data = fetch_weather_data(lat, lon)
        st.metric("Temperature", f"{weather_data[0]['temp']}°C")  # Assuming the first entry has today's data
        st.metric("Humidity", f"{weather_data[0]['humidity']}%")
        st.metric("Wind Speed", f"{weather_data[0]['wind_speed']} m/s")
        
        # AI Summary (Today and Tomorrow)
        ai_data = fetch_ai_summary(lat, lon)
        st.subheader("🧠 AI-Powered Weather Summary")
        st.write(f"**Today**: {ai_data['list'][0]['weather'][0]['description']}")
        st.write(f"**Tomorrow**: {ai_data['list'][1]['weather'][0]['description']}")
        
        # Alerts (if any)
        if 'alerts' in weather_data:
            st.subheader("⚠️ Weather Alerts")
            for alert in weather_data['alerts']:
                st.warning(f"{alert['event']} in {alert['sender_name']}")

        # Train ML model & plot predictions
        if st.button("🔄 Train ML Model & Predict Next 7 Days"):
            with st.spinner("Fetching data and training model..."):
                df = pd.DataFrame(weather_data)  # Now the data is correctly structured for the DataFrame
                model = train_model(df)
                future_dates, future_preds = predict_next_7_days(model)
                st.success("✅ Prediction Complete")

                # Plot prediction comparison
                st.subheader("📈 Temperature Forecast Comparison")
                fig = plt.figure(figsize=(10, 5))
                plt.plot(future_dates, future_preds, label="ML Predicted Temp", color="red", linestyle="--")
                plt.ylabel("Temperature (°C)")
                plt.title("Forecast vs. ML Prediction")
                plt.legend()
                st.pyplot(fig)
    else:
        st.error("❌ City not found. Try another.")
