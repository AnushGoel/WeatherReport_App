import time
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from api_key import API_KEY  # Import the API key
from geopy.geocoders import Nominatim, GeocoderTimedOut, GeocoderServiceError  # Import geocoding exceptions

# Function to get coordinates of a city using Geopy with retry mechanism
def get_coordinates(city_name, retries=3, delay=5):
    geolocator = Nominatim(user_agent="weather_forecast_app")
    
    for attempt in range(retries):
        try:
            location = geolocator.geocode(city_name)
            if location:
                return location.latitude, location.longitude
            else:
                st.error(f"Could not find coordinates for the city '{city_name}'. Please enter a valid city.")
                return None, None
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            st.warning(f"Geocoding service unavailable. Retrying... (Attempt {attempt + 1} of {retries})")
            time.sleep(delay)  # Wait before retrying
        except Exception as e:
            st.error(f"An error occurred while geocoding: {e}")
            return None, None
    
    st.error("Unable to get coordinates after multiple retries. Please try again later.")
    return None, None

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

# Function to process historical data for the last 15 years
def get_historical_data(api_key, lat, lon, units="metric"):
    historical_data = []
    base_url = "https://api.openweathermap.org/data/2.5/onecall/timemachine"
    
    # Get 15 years of historical data
    for year in range(2006, 2021):
        for month in range(1, 13):
            for day in range(1, 28):  # Avoid days that don't exist in all months
                timestamp = int(pd.Timestamp(f"{year}-{month:02d}-{day:02d}").timestamp())
                params = {
                    "lat": lat,
                    "lon": lon,
                    "dt": timestamp,
                    "units": units,
                    "appid": api_key
                }
                response = requests.get(base_url, params=params)
                data = response.json()
                if "current" in data:
                    historical_data.append(data["current"])
    return pd.DataFrame(historical_data)

# Function to predict the weather for the next 60 days using a simple linear regression
def predict_weather(data, days=60):
    temperatures = np.array(data['temp']).reshape(-1, 1)  # Temperature data
    days = np.array(range(len(data))).reshape(-1, 1)  # Day indices
    
    model = LinearRegression()
    model.fit(days, temperatures)
    
    future_days = np.array(range(len(data), len(data) + days)).reshape(-1, 1)
    future_temps = model.predict(future_days)
    return future_temps

# Streamlit UI for the application
def main():
    st.title("Weather Forecast and Prediction App")
    
    # User inputs for city name
    city_name = st.sidebar.text_input("Enter City Name", "San Francisco")
    
    # Get coordinates from city name
    lat, lon = get_coordinates(city_name)
    if lat is None or lon is None:
        return  # Stop execution if coordinates are not available
    
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

        # Predicting weather for the next 60 days
        st.subheader("Weather Prediction for the Next 60 Days")
        historical_data = get_historical_data(API_KEY, lat, lon, units)
        prediction = predict_weather(historical_data, days=60)
        
        # Plot Prediction
        st.subheader("Predicted Temperature for the Next 60 Days")
        plt.plot(range(60), prediction)
        plt.title("Predicted Temperature (Next 60 Days)")
        plt.xlabel("Day")
        plt.ylabel(f"Temperature ({units_label})")
        st.pyplot()

if __name__ == "__main__":
    main()
