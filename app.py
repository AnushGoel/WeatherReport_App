import streamlit as st
import requests
from datetime import datetime

API_KEY = "9c6f06d4d8af4a52e743d4cd5a39425c"  # Replace with your One Call API 3.0 key
BASE_URL = "https://api.openweathermap.org/data/3.0/onecall"

# API Endpoints
GEOCODE_URL = "http://api.openweathermap.org/geo/1.0/direct"
ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall"

# Function to get coordinates of city
def get_coordinates(city):
    params = {
        "q": city,
        "limit": 1,
        "appid": API_KEY
    }
    response = requests.get(GEOCODE_URL, params=params)
    return response.json()

# Function to get weather data using lat/lon
def get_weather(lat, lon):
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric",
        "exclude": "minutely,hourly,alerts"
    }
    response = requests.get(ONECALL_URL, params=params)
    return response.json()

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Weather Forecast", page_icon="🌦", layout="centered")
st.title("🌤 Weather Forecast App")

city = st.text_input("Enter a city name:", value="Toronto")

if city:
    with st.spinner("🔎 Searching for city and fetching forecast..."):
        geo_data = get_coordinates(city)

    if geo_data and isinstance(geo_data, list) and len(geo_data) > 0:
        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]
        weather_data = get_weather(lat, lon)

        st.success(f"✅ Found {geo_data[0]['name']}, {geo_data[0].get('country', '')}")
        st.subheader("📅 7-Day Forecast")

        for day in weather_data["daily"]:
            date = datetime.fromtimestamp(day["dt"]).strftime("%A %d %B")
            temp = day["temp"]["day"]
            desc = day["weather"][0]["description"].capitalize()
            icon = day["weather"][0]["icon"]
            icon_url = f"http://openweathermap.org/img/wn/{icon}.png"

            with st.container():
                st.markdown(f"### {date}")
                col1, col2 = st.columns([1, 5])
                with col1:
                    st.image(icon_url)
                with col2:
                    st.write(f"**Temperature:** {temp}°C")
                    st.write(f"**Condition:** {desc}")
                st.markdown("---")
    else:
        st.error("❌ City not found. Please enter a valid city name.")
