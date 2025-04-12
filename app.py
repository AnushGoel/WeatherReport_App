import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from api_key import API_KEY  # Import the OpenWeather API key

# Predefined cities and their coordinates in Canada
cities = {
    "Toronto": {"lat": 43.7, "lon": -79.42},
    "Vancouver": {"lat": 49.28, "lon": -123.12},
    "Montreal": {"lat": 45.50, "lon": -73.58},
    "Calgary": {"lat": 51.05, "lon": -114.07},
    "Ottawa": {"lat": 45.42, "lon": -75.70}
}

# Function to get weather data from OpenWeather One Call API
def get_weather_data(api_key, lat, lon, forecast_type="daily", units="metric", days=60):
    base_url = "https://api.openweathermap.org/data/2.5/onecall"
    params = {
        "lat": lat,
        "lon": lon,
        "exclude": "minutely,alerts",
        "units": units,
        "appid": api_key
    }
    
    if forecast_type == "daily":
        params["cnt"] = days
    elif forecast_type == "hourly":
        params["cnt"] = 48  # 48 hours forecast
    
    response = requests.get(base_url, params=params)
    data = response.json()
    
    # Check if the response contains the "daily" or "hourly" key
    if "daily" not in data:
        st.error(f"Error fetching daily weather data. Response: {data}")
        return None
    
    return data

# Streamlit UI for the application
def main():
    st.title("Weather Forecast and Prediction App")
    
    # User input for selecting a city from the predefined list
    city_name = st.sidebar.selectbox("Select City", list(cities.keys()))
    lat, lon = cities[city_name]["lat"], cities[city_name]["lon"]
    
    # User input for temperature unit (Celsius or Fahrenheit)
    units = st.sidebar.selectbox("Select Temperature Units", ["metric", "imperial"])
    units_label = "Celsius" if units == "metric" else "Fahrenheit"
    
    # User input for forecast type
    forecast_type = st.sidebar.radio("Select Forecast Type", ["daily", "hourly"])
    
    # Display forecast data options
    days = st.sidebar.slider("Select Number of Days for Forecast", 1, 60, 7)
    
    # Get forecast data
    weather_data = get_weather_data(API_KEY, lat, lon, forecast_type, units, days)
    
    if weather_data:
        if forecast_type == "daily":
            # Show Daily Data
            st.subheader(f"Daily Forecast for the next {days} Days")
            daily_data = weather_data["daily"]
            df_daily = pd.DataFrame(daily_data)
            df_daily["date"] = pd.to_datetime(df_daily["dt"], unit="s")
            st.write(df_daily)
            
            # Graph for daily temperature forecast
            st.subheader("Daily Temperature Forecast")
            plt.plot(df_daily["date"], df_daily["temp"].apply(lambda x: x['day']))
            plt.title("Daily Temperature Forecast")
            plt.xlabel("Date")
            plt.ylabel(f"Temperature ({units_label})")
            st.pyplot()

        elif forecast_type == "hourly":
            # Show Hourly Data
            st.subheader("Hourly Forecast for the next 48 Hours")
            hourly_data = weather_data["hourly"]
            df_hourly = pd.DataFrame(hourly_data)
            df_hourly["date"] = pd.to_datetime(df_hourly["dt"], unit="s")
            st.write(df_hourly)
            
            # Graph for hourly temperature forecast
            st.subheader("Hourly Temperature Forecast")
            plt.plot(df_hourly["date"], df_hourly["temp"])
            plt.title("Hourly Temperature Forecast")
            plt.xlabel("Hour")
            plt.ylabel(f"Temperature ({units_label})")
            st.pyplot()

if __name__ == "__main__":
    main()
