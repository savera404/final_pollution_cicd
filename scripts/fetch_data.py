import requests
import pandas as pd
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
import time

# ✅ Load .env from root folder
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# ✅ API key from .env
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ✅ Coordinates for Karachi only
KARACHI_COORDS = {"lat": 24.8607, "lon": 67.0011}

def fetch_historical_air_quality(start_timestamp, end_timestamp):
    """Fetch historical air quality data for Karachi from OpenWeather API."""
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={KARACHI_COORDS['lat']}&lon={KARACHI_COORDS['lon']}&start={start_timestamp}&end={end_timestamp}&appid={API_KEY}"
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Failed to fetch historical data: {response.status_code}, {response.text}")
        return []
    
    data = response.json()
    if "list" not in data:
        print(f"⚠️ No data in response: {data}")
        return []
    
    records = []
    for entry in data["list"]:
        components = entry["components"]
        aqi = entry["main"]["aqi"]
        timestamp = datetime.fromtimestamp(entry["dt"], tz=timezone.utc).strftime("%m/%d/%Y %H:%M")
        
        records.append({
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
        })
    
    return records

def fetch_current_air_quality():
    """Fetch current air quality data for Karachi from OpenWeather API."""
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

def fetch_all_historical_data():
    """Fetch historical data from January 1, 2024 to now in chunks."""
    # Start date: January 1, 2024
    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc)
    
    all_records = []
    current_start = start_date
    
    # Fetch in 30-day chunks to avoid API limits
    while current_start < end_date:
        current_end = min(
            datetime(current_start.year, current_start.month, current_start.day, tzinfo=timezone.utc).timestamp() + (30 * 24 * 60 * 60),
            end_date.timestamp()
        )
        
        start_timestamp = int(current_start.timestamp())
        end_timestamp = int(current_end)
        
        print(f"📥 Fetching data from {datetime.fromtimestamp(start_timestamp, tz=timezone.utc)} to {datetime.fromtimestamp(end_timestamp, tz=timezone.utc)}")
        
        records = fetch_historical_air_quality(start_timestamp, end_timestamp)
        all_records.extend(records)
        
        # Move to next chunk
        current_start = datetime.fromtimestamp(current_end, tz=timezone.utc)
        
        # Sleep to avoid rate limiting
        time.sleep(1)
    
    return all_records

def filter_to_3hour_intervals(df):
    """Filter data to keep only records at 3-hour intervals (00:00, 03:00, 06:00, etc.)"""
    df['datetime_parsed'] = pd.to_datetime(df['datetime_utc'], format='%m/%d/%Y %H:%M')
    
    # Keep only records where hour is divisible by 3
    df_filtered = df[df['datetime_parsed'].dt.hour % 3 == 0].copy()
    
    # Drop the temporary column
    df_filtered = df_filtered.drop('datetime_parsed', axis=1)
    
    return df_filtered

def main():
    """Fetch data and save to CSV file."""
    csv_path = "data/pollution_data.csv"
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print("🔄 No existing CSV found. Fetching historical data from January 2024...")
        
        # Fetch all historical data
        historical_records = fetch_all_historical_data()
        
        if not historical_records:
            print("❌ No historical data fetched.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(historical_records)
        
        # Filter to 3-hour intervals
        df = filter_to_3hour_intervals(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['datetime_utc'], keep='first')
        
        # Sort by datetime
        df['datetime_parsed'] = pd.to_datetime(df['datetime_utc'], format='%m/%d/%Y %H:%M')
        df = df.sort_values('datetime_parsed').drop('datetime_parsed', axis=1)
        
        print(f"✅ Fetched {len(df)} historical records")
        
        # Save to CSV
        os.makedirs("data", exist_ok=True)
        df.to_csv(csv_path, index=False)
        print("✅ Historical data saved to CSV file.")
        
    else:
        print("📊 CSV file exists. Fetching current data...")
        
        # Fetch current data
        record = fetch_current_air_quality()
        
        if not record:
            print("❌ No current data fetched.")
            return
        
        # Create DataFrame with single record
        df_new = pd.DataFrame([record])
        
        # Load existing data
        existing_df = pd.read_csv(csv_path)
        
        # Check if this timestamp already exists
        if record['datetime_utc'] in existing_df['datetime_utc'].values:
            print("ℹ️ Data for this timestamp already exists. Skipping.")
            return
        
        # Append new data
        updated_df = pd.concat([existing_df, df_new], ignore_index=True)
        
        # Remove duplicates
        updated_df = updated_df.drop_duplicates(subset=['datetime_utc'], keep='last')
        
        # Sort by datetime
        updated_df['datetime_parsed'] = pd.to_datetime(updated_df['datetime_utc'], format='%m/%d/%Y %H:%M')
        updated_df = updated_df.sort_values('datetime_parsed').drop('datetime_parsed', axis=1)
        
        # Save updated data
        updated_df.to_csv(csv_path, index=False)
        print("✅ Data appended to existing CSV file.")
        print(df_new)

if __name__ == "__main__":
    main()
