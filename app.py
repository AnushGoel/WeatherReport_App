import streamlit as st
import requests
from datetime import datetime

API_KEY = "9c6f06d4d8af4a52e743d4cd5a39425c"  # Replace with your One Call API 3.0 key
BASE_URL = "https://api.openweathermap.org/data/3.0/onecall"

def get_weather(lat, lon):
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric",
        "exclude": "minutely,hourly,alerts"
    }
    response = requests.get(BASE_URL, params=params)
    return response.json()

st.title("🌤 Weather Forecast App")

if city:
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
    geo_data = requests.get(geo_url).json()

    if geo_data and isinstance(geo_data, list) and len(geo_data) > 0:
        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]
        weather_data = get_weather(lat, lon)

        st.subheader(f"7-Day Forecast for {city}")
        for day in weather_data["daily"]:
            dt = datetime.fromtimestamp(day["dt"]).strftime("%A %d %B")
            temp = day["temp"]["day"]
            desc = day["weather"][0]["description"]
            st.write(f"**{dt}**: {temp}°C, {desc.capitalize()}")
    else:
        st.error("❌ Could not find location. Please check the city name and try again.")
