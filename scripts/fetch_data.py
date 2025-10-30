import requests
import pandas as pd
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
import time
import sys

# ✅ Load .env from root folder
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# ✅ API key from .env
API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not API_KEY:
    print("❌ ERROR: OPENWEATHER_API_KEY not found in environment variables!")
    sys.exit(1)

# ✅ Coordinates for Karachi only
KARACHI_COORDS = {"lat": 24.8607, "lon": 67.0011}

def fetch_historical_air_quality(start_timestamp, end_timestamp):
    """Fetch historical air quality data for Karachi from OpenWeather API."""
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={KARACHI_COORDS['lat']}&lon={KARACHI_COORDS['lon']}&start={start_timestamp}&end={end_timestamp}&appid={API_KEY}"
    
    print(f"🌐 Requesting: {url[:100]}...")  # Debug output
    
    try:
        response = requests.get(url, timeout=30)
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 401:
            print("⚠️ 401 Unauthorized - API key may be invalid or historical data requires paid subscription")
            return None
        
        if response.status_code != 200:
            print(f"❌ Failed to fetch historical data: {response.status_code}, {response.text}")
            return None
        
        data = response.json()
        
        if "list" not in data:
            print(f"⚠️ No 'list' in response: {data}")
            return None
        
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
        
        print(f"✅ Fetched {len(records)} records")
        return records
    
    except Exception as e:
        print(f"❌ Exception while fetching historical data: {str(e)}")
        return None

def fetch_current_air_quality():
    """Fetch current air quality data for Karachi from OpenWeather API."""
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={KARACHI_COORDS['lat']}&lon={KARACHI_COORDS['lon']}&appid={API_KEY}"
    
    print(f"🌐 Requesting current data...")
    
    try:
        response = requests.get(url, timeout=30)
        print(f"📡 Response status: {response.status_code}")
        
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
        
        record = {
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
        
        print(f"✅ Current data fetched: {record}")
        return record
    
    except Exception as e:
        print(f"❌ Exception while fetching current data: {str(e)}")
        return None

def fetch_all_historical_data():
    """Fetch historical data from January 1, 2024 to now in chunks."""
    # Start date: January 1, 2024
    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc)
    
    print(f"📅 Fetching historical data from {start_date} to {end_date}")
    
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
        
        print(f"📥 Fetching chunk: {datetime.fromtimestamp(start_timestamp, tz=timezone.utc)} to {datetime.fromtimestamp(end_timestamp, tz=timezone.utc)}")
        
        records = fetch_historical_air_quality(start_timestamp, end_timestamp)
        
        if records is None:
            print("⚠️ Historical data fetch returned None - API may require paid subscription")
            return None
        
        if records:
            all_records.extend(records)
            print(f"   Added {len(records)} records, total: {len(all_records)}")
        
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
    
    print(f"🔍 Filtered from {len(df)} to {len(df_filtered)} records (3-hour intervals)")
    
    return df_filtered

def main():
    """Fetch data and save to CSV file."""
    print("=" * 60)
    print("🚀 Starting pollution data fetch script")
    print("=" * 60)
    
    csv_path = "data/pollution_data.csv"
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    print(f"📁 Data directory ensured at: {os.path.abspath('data')}")
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print("🔄 No existing CSV found. Attempting to fetch historical data from January 2024...")
        
        # Try to fetch all historical data
        historical_records = fetch_all_historical_data()
        
        # If historical fetch fails or returns None, fall back to current data
        if not historical_records:
            print("⚠️ Historical data not available (may require paid API subscription)")
            print("📊 Falling back to current data only...")
            
            current_record = fetch_current_air_quality()
            if not current_record:
                print("❌ Failed to fetch current data as well. Exiting.")
                sys.exit(1)
            
            # Create CSV with just current data
            df = pd.DataFrame([current_record])
            df.to_csv(csv_path, index=False)
            print(f"✅ CSV created with current data at: {os.path.abspath(csv_path)}")
            print(df)
            return
        
        # Create DataFrame from historical data
        df = pd.DataFrame(historical_records)
        print(f"📊 Created DataFrame with {len(df)} records")
        
        # Filter to 3-hour intervals
        df = filter_to_3hour_intervals(df)
        
        # Remove duplicates
        original_len = len(df)
        df = df.drop_duplicates(subset=['datetime_utc'], keep='first')
        if original_len != len(df):
            print(f"🔄 Removed {original_len - len(df)} duplicate records")
        
        # Sort by datetime
        df['datetime_parsed'] = pd.to_datetime(df['datetime_utc'], format='%m/%d/%Y %H:%M')
        df = df.sort_values('datetime_parsed').drop('datetime_parsed', axis=1)
        
        print(f"✅ Final dataset has {len(df)} historical records")
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"✅ Historical data saved to: {os.path.abspath(csv_path)}")
        print(f"📋 First few records:\n{df.head()}")
        
    else:
        print(f"📊 CSV file exists at: {os.path.abspath(csv_path)}")
        print("📥 Fetching current data...")
        
        # Fetch current data
        record = fetch_current_air_quality()
        
        if not record:
            print("❌ No current data fetched.")
            return
        
        # Create DataFrame with single record
        df_new = pd.DataFrame([record])
        
        # Load existing data
        existing_df = pd.read_csv(csv_path)
        print(f"📖 Loaded {len(existing_df)} existing records")
        
        # Check if this timestamp already exists
        if record['datetime_utc'] in existing_df['datetime_utc'].values:
            print(f"ℹ️ Data for {record['datetime_utc']} already exists. Skipping.")
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
        print(f"✅ Data appended. Total records: {len(updated_df)}")
        print(f"📋 New record:\n{df_new}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"💥 FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
