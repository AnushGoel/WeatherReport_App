 
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ------------------- CONFIG -------------------
API_KEY = "9c6f06d4d8af4a52e743d4cd5a39425c"
WEATHER_URL = "https://api.openweathermap.org/data/3.0/onecall"
GEOCODE_URL = "http://api.openweathermap.org/geo/1.0/direct"

# ------------------- APP SETUP -------------------
st.set_page_config(page_title="🧠 ML Weather Forecast", layout="wide")
st.title("🌤️ Weather Forecast & ML Prediction App")

# ------------------- USER INPUT -------------------
city = st.text_input("Enter city name", value="Toronto")
uploaded_file = st.file_uploader("📂 Upload historical weather CSV (6+ years)", type="csv")

# ------------------- LIVE WEATHER -------------------
def get_live_weather(city):
    geo_req = requests.get(GEOCODE_URL, params={"q": city, "limit": 1, "appid": API_KEY})
    geo_data = geo_req.json()
    if not geo_data:
        return None
    lat, lon = geo_data[0]["lat"], geo_data[0]["lon"]
    weather_req = requests.get(WEATHER_URL, params={
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric",
        "exclude": "minutely,hourly,alerts"
    })
    return weather_req.json()

# ------------------- ML MODEL -------------------
def train_model(df):
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_year'] = df['date'].dt.dayofyear
    df['year'] = df['date'].dt.year
    X = df[['day_of_year', 'year']]
    y = df['temp']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_next_week(model, start_date):
    future_dates = [start_date + timedelta(days=i) for i in range(7)]
    future_df = pd.DataFrame({
        "day_of_year": [d.timetuple().tm_yday for d in future_dates],
        "year": [d.year for d in future_dates]
    })
    preds = model.predict(future_df)
    return future_dates, preds

# ------------------- PLOT FUNCTION -------------------
def plot_prediction(past_dates, past_temps, future_dates, future_preds):
    plt.figure(figsize=(12, 6))
    plt.plot(past_dates, past_temps, label="Actual Temp (Past)", color="blue")
    plt.plot(future_dates, future_preds, label="Predicted Temp (Next 7 Days)", color="red", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title("Actual vs Predicted Temperature")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

# ------------------- MAIN EXECUTION -------------------
if city:
    st.subheader(f"📍 Real-Time Weather in {city}")
    weather_data = get_live_weather(city)
    if weather_data:
        st.success(f"Current Temperature: {weather_data['current']['temp']} °C")
        st.markdown("---")
    else:
        st.error("Could not find the city. Check spelling or try another location.")

# ------------------- HISTORICAL + ML SECTION -------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'date' not in df.columns or 'temp' not in df.columns:
        st.error("CSV must contain at least 'date' and 'temp' columns.")
    else:
        st.subheader("📈 ML-Based Temperature Prediction")
        st.info("Training model... please wait ⌛")
        model = train_model(df)

        # Extract past 30 days for plot
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        past_30 = df[df['date'] > df['date'].max() - pd.Timedelta(days=30)]

        # Predict next 7 days
        future_dates, future_preds = predict_next_week(model, df['date'].max() + timedelta(days=1))

        plot_prediction(past_30['date'], past_30['temp'], future_dates, future_preds)

        st.success("✅ Forecast generated using machine learning model.")
        st.markdown("You can retrain the model anytime by uploading a new dataset.")
else:
    st.warning("📂 Please upload at least 6 years of historical weather data to enable ML forecasting.")
