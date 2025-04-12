import time
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from api_key import API_KEY  # Import the API key
from geopy.geocoders import Nominatim  # Corrected import for geocoder
from geopy.exc import GeocoderTimedOut, GeocoderServiceError  # Correct exception imports

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

# Streamlit UI for the application
def main():
    st.title("Weather Forecast and Prediction App")
    
    # User inputs for city name
    city_name = st.sidebar.text_input("Enter City Name", "San Francisco")
    
    # Get coordinates from city name
    lat, lon = get_coordinates(city_name)
    if lat is None or lon is None:
        return  # Stop execution if coordinates are not available
    
    # Additional code for getting weather data and displaying the app...
    
if __name__ == "__main__":
    main()
