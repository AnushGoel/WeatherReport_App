import streamlit as st
import requests
from datetime import datetime
import pandas as pd

# ---------- CONFIG ----------
API_KEY = "9c6f06d4d8af4a52e743d4cd5a39425c"  # Replace with your OpenWeather API Key
ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall/timemachine"  # URL for historical data

# ---------- HELPER FUNCTIONS ----------

# Get city coordinates using Geopy
def get_coordinates(city):
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="weatherApp")
    location = geolocator.geocode(city)
    if location:
        return location.latitude, location.longitude
    return None, None

# Fetch historical weather data from OpenWeather API (for one specific day)
def fetch_historical_weather_data(lat, lon, dt):
    params = {
        "lat": lat, "lon": lon, "dt": dt, "appid": API_KEY,
        "units": "metric"
    }
    response = requests.get(ONECALL_URL, params=params)
    data = response.json()
    
    if 'current' in data:
        weather_record = {
            "date": datetime.utcfromtimestamp(data["current"]["dt"]).date(),
            "temp": round(data["current"]["temp"], 2),
            "humidity": round(data["current"]["humidity"], 2),
            "wind_speed": round(data["current"]["wind_speed"], 2),
            "pressure": round(data["current"]["pressure"], 2),
            "precipitation": round(data["current"].get("precipitation", 0), 2)
        }
        return weather_record
    return None

# ---------- STREAMLIT APP LAYOUT ----------

st.set_page_config(page_title="Historical Weather Data", layout="wide")
st.title("🌍 Fetch Historical Weather Data")

# User input: City and specific date
city = st.text_input("Enter the city name:", value="Toronto")
date_str = st.text_input("Enter the date (YYYY-MM-DD):", value="2025-01-01")

# Convert date to Unix timestamp (for historical data API)
try:
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    timestamp = int(date_obj.timestamp())
except ValueError:
    timestamp = None
    st.error("Invalid date format. Please enter the date in YYYY-MM-DD format.")

# Fetch coordinates for the city
if city and timestamp:
    lat, lon = get_coordinates(city)

    if lat and lon:
        st.success(f"📍 Found location: {city} (Lat: {lat}, Lon: {lon})")

        # Get historical weather data for the specific date
        weather_data = fetch_historical_weather_data(lat, lon, timestamp)

        if weather_data:
            # Display the weather data
            st.subheader(f"Historical Weather Data for {city} on {date_str}")
            st.write(f"Temperature: {weather_data['temp']} °C")
            st.write(f"Humidity: {weather_data['humidity']} %")
            st.write(f"Wind Speed: {weather_data['wind_speed']} m/s")
            st.write(f"Pressure: {weather_data['pressure']} hPa")
            st.write(f"Precipitation: {weather_data['precipitation']} mm")
        else:
            st.error("No weather data available for the selected date.")
    else:
        st.error("❌ City not found. Please check the city name and try again.")
