 🌍 Automated Air Quality Index (AQI) Prediction Pipeline

# Project Overview
This project is a fully automated MLOps pipeline for Air Quality Index (AQI) prediction using real-time pollution and weather data.  
It fetches live data every few hours, cleans and preprocesses it, computes AQI manually, performs EDA and feature engineering, and retrains machine learning models — all automated with GitHub Actions and Hopsworks.

The system ensures that the AQI predictions remain continuously updated and accurate without manual intervention.


### 🔁 Automation Workflows (GitHub Actions)

| Workflow | Schedule | Description |
|-----------|-----------|-------------|
| `fetch_data.yml` | Every 3 hours | Fetch raw pollutant data from OpenWeather API (no AQI included) |
| `eda.yml` | 15 minutes after fetch | Clean, preprocess, compute AQI, perform EDA, and upload features to Hopsworks |
| `training.yml` | Every 6 hours | Retrain models using updated features and register new model versions |



## 🗂️ Repository Structure

├── .github/workflows/
│ ├── fetch_data.yml # Fetch data automation
│ ├── eda.yml # Preprocessing + EDA automation
│ ├── training.yml # Model training automation
│
├── data/
│ └── 2_years.csv # Collected pollution data (auto-updated)
│
├── scripts/
│ ├── fetch_data.py # Fetch raw data from OpenWeather
│ ├── eda.py # Clean data, compute AQI, upload to Hopsworks
│ ├── model_loading.py # Train and upload model to Hopswork
│ ├── final_frontend.py
├
├── requirements.txt # Dependencies
└── README.md # Project documentation
