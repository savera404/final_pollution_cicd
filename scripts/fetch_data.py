import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# === Load API key ===
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Karachi coordinates
LAT, LON = 24.8607, 67.0011

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

CSV_PATH = "data/pollution_data.csv"


def fetch_data(start, end):
    """Fetch air pollution data between two UNIX timestamps"""
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start}&end={end}&appid={API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Error {response.status_code} for {datetime.utcfromtimestamp(start)} → {datetime.utcfromtimestamp(end)}")
        return pd.DataFrame()

    data = response.json()
    records = []
    for item in data.get("list", []):
        dt = datetime.utcfromtimestamp(item["dt"])
        records.append({
            "datetime_utc": dt,
            "aqi": item["main"]["aqi"],
            **item["components"]
        })
    return pd.DataFrame(records)


def main():
    if not os.path.exists(CSV_PATH):
        # === FIRST RUN: Fetch full 2-year (2023–2025) history ===
        start_time = int(datetime(2023, 1, 1).timestamp())
        end_time = int(time.time())

        print("📘 Fetching 2023–present historical data for Karachi...")

        df_parts = []
        step_days = 180
        current_start = start_time

        while current_start < end_time:
            current_end = int((datetime.utcfromtimestamp(current_start) + timedelta(days=step_days)).timestamp())
            if current_end > end_time:
                current_end = end_time

            df_part = fetch_data(current_start, current_end)
            df_parts.append(df_part)
            current_start = current_end
            time.sleep(1)  # avoid rate limit

        df = pd.concat(df_parts, ignore_index=True)
        df.drop_duplicates(subset="datetime_utc", inplace=True)
        df.sort_values("datetime_utc", inplace=True)
        df.reset_index(drop=True, inplace=True)

        df.to_csv(CSV_PATH, index=False, date_format="%d/%m/%Y %H:%M")
        print(f"✅ Created new file: {CSV_PATH} with {len(df)} records")

    else:
        # === SUBSEQUENT RUNS: Fetch latest data only ===
        existing = pd.read_csv(CSV_PATH, parse_dates=["datetime_utc"], dayfirst=True)
        last_dt = existing["datetime_utc"].max()
        last_timestamp = int(pd.Timestamp(last_dt).timestamp())

        new_start = last_timestamp + 1
        new_end = int(time.time())

        print(f"📈 Updating data from {last_dt} to {datetime.utcfromtimestamp(new_end)}...")

        df_new = fetch_data(new_start, new_end)

        if df_new.empty:
            print("⚠️ No new data found.")
            return

        combined = pd.concat([existing, df_new], ignore_index=True)
        combined.drop_duplicates(subset="datetime_utc", inplace=True)
        combined.sort_values("datetime_utc", inplace=True)
        combined.reset_index(drop=True, inplace=True)

        combined.to_csv(CSV_PATH, index=False, date_format="%d/%m/%Y %H:%M")
        print(f"✅ Appended {len(df_new)} new records to {CSV_PATH}")


if __name__ == "__main__":
    main()
