import os
import requests
import pandas as pd
import time
from datetime import datetime, timedelta

LAT = 24.8607   # Karachi latitude
LON = 67.0011   # Karachi longitude
API_KEY = '485983fec1ac52f97cf59566b45a6c16'

# === Ensure output folder exists ===
os.makedirs("../data", exist_ok=True)

def fetch_data(start, end):
    """Fetch air pollution data between two UNIX timestamps."""
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start}&end={end}&appid={API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Error {response.status_code} fetching data for {datetime.utcfromtimestamp(start)} → {datetime.utcfromtimestamp(end)}")
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

# === Define time range: from Jan 2023 to now (covers 2023–2025) ===
start_time = int(datetime(2023, 1, 1).timestamp())
end_time = int(time.time())  # current time

# Split into 6-month chunks to avoid API limits
chunk_size = 180  # days
chunks = []
current_start = start_time

print("📘 Fetching historical air pollution data for Karachi (2023–2025)...")

while current_start < end_time:
    current_end = int((datetime.utcfromtimestamp(current_start) + timedelta(days=chunk_size)).timestamp())
    if current_end > end_time:
        current_end = end_time
    df_chunk = fetch_data(current_start, current_end)
    chunks.append(df_chunk)
    current_start = current_end
    time.sleep(1)  # avoid hitting API rate limit

# Combine all chunks
df = pd.concat(chunks, ignore_index=True)
df.drop_duplicates(subset="datetime_utc", inplace=True)

# ✅ Correctly sort by datetime
df = df.sort_values("datetime_utc").reset_index(drop=True)

print(f"📊 Total records collected: {len(df)}")

# === Save to CSV ===
file_path = "../data/pollution_data.csv"
df.to_csv(file_path, index=False, date_format="%d/%m/%Y %H:%M")
print(f"✅ Data saved to {file_path}")
