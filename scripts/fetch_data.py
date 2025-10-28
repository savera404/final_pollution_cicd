import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# === Karachi coordinates ===
LAT = 24.8607
LON = 67.0011

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "pollution_data.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# === Fetch Historical Data (2 years) ===
def fetch_historical_data():
    """Fetch 2 years of air pollution history for Karachi in 4 chunks (6 months each)."""
    print("📘 Fetching 2 years of historical data for Karachi...")

    end_time = int(time.time())
    part3_end = int((datetime.now() - timedelta(days=180)).timestamp())
    part2_end = int((datetime.now() - timedelta(days=365)).timestamp())
    part1_end = int((datetime.now() - timedelta(days=545)).timestamp())
    start_time = int((datetime.now() - timedelta(days=730)).timestamp())

    def fetch_chunk(start, end):
        url = (
            f"http://api.openweathermap.org/data/2.5/air_pollution/history"
            f"?lat={LAT}&lon={LON}&start={start}&end={end}&appid={API_KEY}"
        )
        response = requests.get(url)
        if response.status_code != 200:
            print(f"❌ Error {response.status_code} ({datetime.utcfromtimestamp(start)} → {datetime.utcfromtimestamp(end)})")
            return pd.DataFrame()

        data = response.json()
        records = []
        for item in data.get("list", []):
            dt = datetime.utcfromtimestamp(item["dt"])
            records.append({
                "datetime_utc": dt.strftime("%m/%d/%Y %H:%M"),
                "aqi": item["main"]["aqi"],
                "co": item["components"].get("co", 0),
                "no": item["components"].get("no", 0),
                "no2": item["components"].get("no2", 0),
                "o3": item["components"].get("o3", 0),
                "so2": item["components"].get("so2", 0),
                "pm2_5": item["components"].get("pm2_5", 0),
                "pm10": item["components"].get("pm10", 0),
                "nh3": item["components"].get("nh3", 0),
            })
        return pd.DataFrame(records)

    # Combine 4 chunks (6 months each)
    df = pd.concat([
        fetch_chunk(start_time, part1_end),
        fetch_chunk(part1_end, part2_end),
        fetch_chunk(part2_end, part3_end),
        fetch_chunk(part3_end, end_time)
    ], ignore_index=True)

    df.drop_duplicates(subset=["datetime_utc"], inplace=True)
    df.sort_values("datetime_utc", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"✅ Total {len(df)} records fetched for Karachi.")
    return df


# === Fetch Current Data (for next runs) ===
def fetch_current_data():
    """Fetch the current air quality reading for Karachi."""
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"❌ Failed to fetch current data: {response.status_code}")
        return None

    data = response.json()
    if "list" not in data:
        print("⚠️ No data in API response.")
        return None

    item = data["list"][0]
    record = {
        "datetime_utc": datetime.utcnow().strftime("%m/%d/%Y %H:%M"),
        "aqi": item["main"]["aqi"],
        "co": item["components"].get("co", 0),
        "no": item["components"].get("no", 0),
        "no2": item["components"].get("no2", 0),
        "o3": item["components"].get("o3", 0),
        "so2": item["components"].get("so2", 0),
        "pm2_5": item["components"].get("pm2_5", 0),
        "pm10": item["components"].get("pm10", 0),
        "nh3": item["components"].get("nh3", 0),
    }
    return record


# === MAIN ===
def main():
    if not os.path.exists(CSV_PATH):
        print("🕐 No existing data found → Fetching 2-year historical dataset...")
        df = fetch_historical_data()
        df.to_csv(CSV_PATH, index=False)
        print(f"✅ Historical data saved to {CSV_PATH}")
    else:
        print("⏰ CSV found → Fetching current air quality data...")
        record = fetch_current_data()
        if record:
            df_new = pd.DataFrame([record])
            df_old = pd.read_csv(CSV_PATH)
            df_updated = pd.concat([df_old, df_new], ignore_index=True)
            df_updated.to_csv(CSV_PATH, index=False)
            print(f"✅ Appended latest record at {record['datetime_utc']} to {CSV_PATH}")
        else:
            print("⚠️ No new record added.")


if __name__ == "__main__":
    main()
