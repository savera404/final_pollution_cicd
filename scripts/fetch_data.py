import requests
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# ✅ Load .env from root folder
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ✅ Coordinates for Karachi only
KARACHI_COORDS = {"lat": 24.8607, "lon": 67.0011}

def fetch_air_quality():
    """Fetch air quality data for Karachi from OpenWeather API."""
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={KARACHI_COORDS['lat']}&lon={KARACHI_COORDS['lon']}&appid={API_KEY}"
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Failed to fetch data: {response.status_code}, {response.text}")
        return None
    
    data = response.json()
    if "list" not in data:
        print(f"⚠️ No data in response: {data}")
        return None
    
    components = data["list"][0]["components"]
    aqi = data["list"][0]["main"]["aqi"]
    timestamp = datetime.utcnow().strftime("%m/%d/%Y %H:%M")
    
    return {
        "datetime_utc": timestamp,
        "aqi": aqi,
        "co": components.get("co", 0),
        "no": components.get("no", 0),
        "no2": components.get("no2", 0),
        "o3": components.get("o3", 0),
        "so2": components.get("so2", 0),
        "pm2_5": components.get("pm2_5", 0),
        "pm10": components.get("pm10", 0),
        "nh3": components.get("nh3", 0),
    }

def main():
    """Fetch data and save to CSV file."""
    record = fetch_air_quality()
    
    if not record:
        print("❌ No data fetched.")
        return
    
    # Create DataFrame with single record
    df = pd.DataFrame([record])
    print(df)
    
    # ✅ Save or append to CSV
    csv_path = "data/pollution_data.csv"
    
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_csv(csv_path, index=False)
        print("✅ Data appended to existing CSV file.")
    else:
        df.to_csv(csv_path, index=False)
        print("✅ New CSV file created and data saved.")

if __name__ == "__main__":
    main()
