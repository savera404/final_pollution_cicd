



import pandas as pd
import numpy as np
from datetime import datetime

# STEP 1: US EPA AQI CONVERSION FUNCTIONS (1-500 Scale)

#this function converts a pollutant’s raw concentration (µg/m³) into AQI value based on US EPA standards.
def calculate_aqi_single_pollutant(concentration, pollutant):
   
    # US EPA AQI Breakpoints [C_low, C_high, AQI_low, AQI_high]
    breakpoints = {
        'pm2_5': [  
            (0.0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500)
        ],
        'pm10': [  
            (0, 54, 0, 50),
            (55, 154, 51, 100),
            (155, 254, 101, 150),
            (255, 354, 151, 200),
            (355, 424, 201, 300),
            (425, 504, 301, 400),
            (505, 604, 401, 500)
        ],
        'o3': [ 
            (0, 54, 0, 50),
            (55, 70, 51, 100),
            (71, 85, 101, 150),
            (86, 105, 151, 200),
            (106, 200, 201, 300),
            (201, 404, 301, 400),
            (405, 604, 401, 500)
        ],
        'no2': [  
            (0, 53, 0, 50),
            (54, 100, 51, 100),
            (101, 360, 101, 150),
            (361, 649, 151, 200),
            (650, 1249, 201, 300),
            (1250, 1649, 301, 400),
            (1650, 2049, 401, 500)
        ],
        'so2': [ 
            (0, 35, 0, 50),
            (36, 75, 51, 100),
            (76, 185, 101, 150),
            (186, 304, 151, 200),
            (305, 604, 201, 300),
            (605, 804, 301, 400),
            (805, 1004, 401, 500)
        ],
        'co': [  
            (0.0, 4.4, 0, 50),
            (4.5, 9.4, 51, 100),
            (9.5, 12.4, 101, 150),
            (12.5, 15.4, 151, 200),
            (15.5, 30.4, 201, 300),
            (30.5, 40.4, 301, 400),
            (40.5, 50.4, 401, 500)
        ]
    }

    if pollutant not in breakpoints:
        return np.nan

    # convert concentrations for specific pollutants
    if pollutant == 'o3':
     
        concentration = (concentration * 24.45) / 48
    elif pollutant == 'no2':
        
        concentration = (concentration * 24.45) / 46
    elif pollutant == 'so2':
       
        concentration = (concentration * 24.45) / 64
    elif pollutant == 'co':
       
        concentration = (concentration / 1000) * 0.873

    
    for c_low, c_high, aqi_low, aqi_high in breakpoints[pollutant]:
        if c_low <= concentration <= c_high:
            # linear interpolation formula
            aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (concentration - c_low) + aqi_low
            return round(aqi, 2)

    # If concentration exceeds all breakpoints, return 500 (hazardous)
    return 500.0

#now that each pollutant’s AQI can be computed, we can calculate an overall AQI for a given timestamp.
def calculate_overall_aqi(row):
 
    pollutants = {
        'pm2_5': row.get('pm2_5', np.nan),
        'pm10': row.get('pm10', np.nan),
        'o3': row.get('o3', np.nan),
        'no2': row.get('no2', np.nan),
        'so2': row.get('so2', np.nan),
        'co': row.get('co', np.nan)
    }

    aqi_values = {}
    for pollutant, concentration in pollutants.items():
        if pd.notna(concentration) and concentration >= 0:
            aqi_values[pollutant] = calculate_aqi_single_pollutant(concentration, pollutant)

    if not aqi_values:
        return {'aqi': np.nan, 'dominant_pollutant': None}

    # overall AQI is the maximum of all pollutant AQIs
    dominant_pollutant = max(aqi_values, key=aqi_values.get)
    overall_aqi = aqi_values[dominant_pollutant]

    return {
        'aqi': overall_aqi,
        'dominant_pollutant': dominant_pollutant,
        **{f'aqi_{k}': v for k, v in aqi_values.items()}
    }


# STEP 2: DATA CLEANING & PREPROCESSING
# ==========================================================================
#this function takes one row from your dataset (a single timestamp’s readings)
def clean_air_quality_data(df):


    df_clean = df.copy()
    initial_rows = len(df_clean)
    print(f"\n Initial dataset size: {initial_rows} rows, {len(df_clean.columns)} columns")

    # handle datetime
    if 'datetime_utc' in df_clean.columns:
        print("\n⏰ Converting datetime column...")
        df_clean['datetime_utc'] = pd.to_datetime(df_clean['datetime_utc'], errors='coerce')
        df_clean = df_clean.dropna(subset=['datetime_utc'])
        print(f"   ✓ Removed {initial_rows - len(df_clean)} rows with invalid datetime")

    # remove duplicate timestamps
    if 'datetime_utc' in df_clean.columns:
        before_dedup = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['datetime_utc'], keep='first')
        print(f"    Removed {before_dedup - len(df_clean)} duplicate timestamps")

    #  handle negative values (pollutants can't be negative)
    pollutant_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    print(f"\n🔍 Checking for negative values in pollutant columns...")

    for col in pollutant_cols:
        if col in df_clean.columns:
            negative_count = (df_clean[col] < 0).sum()
            if negative_count > 0:
                print(f"   ⚠️  {col}: Found {negative_count} negative values, setting to 0")
                df_clean[col] = df_clean[col].clip(lower=0)

    # handle extreme outliers using IQR method (per pollutant)
    print(f"\n📈 Handling extreme outliers (values beyond 3×IQR)...")
    for col in pollutant_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  
            upper_bound = Q3 + 3 * IQR

            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"   ⚠️  {col}: Capping {outliers} extreme outliers")
                df_clean[col] = df_clean[col].clip(lower=max(0, lower_bound), upper=upper_bound)

    # Handle missing values
    print(f"\n Handling missing values...")
    missing_before = df_clean.isnull().sum().sum()

    # For pollutants: Use forward fill then backward fill (reasonable for time series)
    for col in pollutant_cols:
        if col in df_clean.columns:
            missing = df_clean[col].isnull().sum()
            if missing > 0:
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
                # If still missing, fill with median
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                print(f"   ✓ {col}: Filled {missing} missing values")

    missing_after = df_clean.isnull().sum().sum()
    print(f"   ✓ Total missing values reduced from {missing_before} to {missing_after}")

    # remove AQI column if present (we'll recalculate it)
    if 'aqi' in df_clean.columns:
        print(f"\n🗑️  Removing existing 'aqi' column (will be recalculated)")
        df_clean = df_clean.drop(columns=['aqi'])

    print(f"\n✅ CLEANING COMPLETE")
    print(f"   Final dataset size: {len(df_clean)} rows ({initial_rows - len(df_clean)} rows removed)")
    print("=" * 70)

    return df_clean



# STEP 3: FEATURE ENGINEERING
# ============================================================================

def create_features(df):

    df_features = df.copy()

    # Temporal features
    if 'datetime_utc' in df_features.columns:
        print("\n🕐 Creating temporal features...")
        df_features['hour'] = df_features['datetime_utc'].dt.hour
        df_features['day_of_week'] = df_features['datetime_utc'].dt.dayofweek
        df_features['day_of_month'] = df_features['datetime_utc'].dt.day
        df_features['month'] = df_features['datetime_utc'].dt.month
        df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)

        # season 
        df_features['season'] = df_features['month'].apply(
            lambda x: 'winter' if x in [12, 1, 2] else
                     'spring' if x in [3, 4, 5] else
                     'summer' if x in [6, 7, 8] else 'fall'
        )

        # time of day
        df_features['time_of_day'] = df_features['hour'].apply(
            lambda x: 'night' if x < 6 or x >= 22 else
                     'morning' if x < 12 else
                     'afternoon' if x < 18 else 'evening'
        )

        print(f"   ✓ Created: hour, day_of_week, month, season, time_of_day, is_weekend")

    # pollutant ratios and interaction, these composite features give the model a broader view of overall air quality.
    print("\n Creating pollutant interaction features...")

    # PM ratio (indicates particle size distribution)
    if 'pm2_5' in df_features.columns and 'pm10' in df_features.columns:
        df_features['pm_ratio'] = df_features['pm2_5'] / (df_features['pm10'] + 1e-6)
        print(f"   ✓ pm_ratio: PM2.5/PM10 ratio")

    # nitrogen oxides ratio
    if 'no2' in df_features.columns and 'no' in df_features.columns:
        df_features['nox_ratio'] = df_features['no2'] / (df_features['no'] + df_features['no2'] + 1e-6)
        print(f"   ✓ nox_ratio: NO2/(NO+NO2) ratio")

    # total particulate matter
    if 'pm2_5' in df_features.columns and 'pm10' in df_features.columns:
        df_features['total_pm'] = df_features['pm2_5'] + df_features['pm10']
        print(f"   ✓ total_pm: Sum of PM2.5 and PM10")

    # total gaseous pollutants
    gas_cols = ['co', 'no', 'no2', 'o3', 'so2']
    available_gas = [col for col in gas_cols if col in df_features.columns]
    if available_gas:
        df_features['total_gases'] = df_features[available_gas].sum(axis=1)
        print(f"    total_gases: Sum of gaseous pollutants")

    # rolling averages Smooths short-term noise and highlights trends.
    print("\n Creating rolling average features (window=3)...")
    pollutant_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10']

    for col in pollutant_cols:
        if col in df_features.columns:
            df_features[f'{col}_rolling_3'] = df_features[col].rolling(window=3, min_periods=1).mean()
            print(f"   ✓ {col}_rolling_3: 3-period rolling average")

    # pollution intensity indicators: Creates binary flags (0/1) for whether pollution exceeds safe limits.
    print("\n Creating pollution intensity indicators...")

    if 'pm2_5' in df_features.columns:
        df_features['pm2_5_high'] = (df_features['pm2_5'] > 35.4).astype(int)  # Above Good level
        print(f"    pm2_5_high: Binary indicator (>35.4 µg/m³)")

    if 'pm10' in df_features.columns:
        df_features['pm10_high'] = (df_features['pm10'] > 154).astype(int)  # Above Good level
        print(f"   ✓ pm10_high: Binary indicator (>154 µg/m³)")

    print("\n FEATURE ENGINEERING COMPLETE")
    print(f"   Total features: {len(df_features.columns)}")
    print("=" * 70)

    return df_features


# STEP 4: CALCULATE AQI AND PREPARE FINAL DATASET
# ============================================================================

def prepare_final_dataset(df):


    df_final = df.copy()

    # calculate individual pollutant AQIs and overall AQI For each row (time record), it calls your previously defined function calculate_overall_aqi()
    print("\n Calculating AQI for all pollutants...")
    aqi_results = df_final.apply(calculate_overall_aqi, axis=1)
    aqi_df = pd.DataFrame(aqi_results.tolist())

    # merge results
    df_final = pd.concat([df_final, aqi_df], axis=1)

    # Shows a quick summary of AQI distribution
    print(f"\n📈 AQI Statistics:")
    print(f"   Mean AQI: {df_final['aqi'].mean():.2f}")
    print(f"   Median AQI: {df_final['aqi'].median():.2f}")
    print(f"   Min AQI: {df_final['aqi'].min():.2f}")
    print(f"   Max AQI: {df_final['aqi'].max():.2f}")

    # Classify AQI into categories
   
    df_final['aqi_category'] = pd.cut(
        df_final['aqi'],
        bins=[0, 50, 100, 150, 200, 300, 400, 500],
        labels=['Good', 'Moderate', 'Unhealthy for Sensitive', 'Unhealthy',
                'Very Unhealthy', 'Hazardous', 'Beyond Hazardous']
    )

    print(f"\n AQI Category Distribution:")
    print(df_final['aqi_category'].value_counts().sort_index())

    print("\n AQI CALCULATION COMPLETE")
    print("=" * 70)

    return df_final



# MAIN PIPELINE FUNCTION
# ============================================================================
#this function ties together all parts of pipeline — it’s the master workflow.
def preprocess_air_quality_pipeline(df):
 
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    #  Clean data
    df_clean = clean_air_quality_data(df)

    # Feature engineering
    df_features = create_features(df_clean)

    # Calculate AQI
    df_final = prepare_final_dataset(df_features)

    print(f"\n🎉 PIPELINE COMPLETE!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Final dataset shape: {df_final.shape}")
    print("="*70 + "\n")

    return df_final


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Load your data
    # df = pd.read_csv('your_air_quality_data.csv')

    # Or use the sample data from the image

    sample_data = pd.read_csv("data/2_years.csv")
    df = pd.DataFrame(sample_data)

    # Run pipeline
    df_processed = preprocess_air_quality_pipeline(df)

    # Display results
    print("\nSample of processed data:")
    print(df_processed[['datetime_utc', 'aqi', 'aqi_category', 'dominant_pollutant',
                        'pm2_5', 'pm10', 'total_pm']].head(10))

    # Save processed data
    # df_processed.to_csv('processed_air_quality_data.csv', index=False)
    print("\n Data is ready for Hopsworks upload!")




#EDA AND FEATURE SELECTION



#Part 2: EDA & Hopsworks Integration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')


# EXPLORATORY DATA ANALYSIS (EDA)


def perform_comprehensive_eda(df, target='aqi'):


    insights = {}

    # Basic Statistics
    print("\n📊 1. BASIC STATISTICS")
    print("-" * 70)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"\nTarget variable '{target}' statistics:")
    print(df[target].describe())

    insights['numeric_features'] = numeric_cols
    insights['target_stats'] = df[target].describe().to_dict()

    # Distribution Analysis, Measures skewness — how asymmetric the data distribution is.
    print("\n 2. DISTRIBUTION ANALYSIS")
    print("-" * 70)

    # Check skewness
    skewness = {}
    for col in numeric_cols[:10]:  # Check first 10 numeric columns
        skew = df[col].skew()
        skewness[col] = skew
        if abs(skew) > 1:
            print(f"   ⚠️  {col}: Highly skewed (skewness={skew:.2f})")

    insights['skewness'] = skewness

    # Target variable distribution
    print(f"\n   Target '{target}' distribution:")
    print(f"   - Skewness: {df[target].skew():.2f}")
    print(f"   - Kurtosis: {df[target].kurtosis():.2f}")

    # Missing Values Analysis
    print("\n 3. MISSING VALUES ANALYSIS")
    print("-" * 70)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✓ No missing values found!")
    else:
        print("   Missing values per column:")
        for col in missing[missing > 0].index:
            print(f"   - {col}: {missing[col]} ({missing[col]/len(df)*100:.2f}%)")

    insights['missing_values'] = missing.to_dict()

    # Correlation Analysis
    print("\n 4. CORRELATION ANALYSIS")
    print("-" * 70)

    # Correlation with target
    correlations = df[numeric_cols].corrwith(df[target]).sort_values(ascending=False)

    print(f"\n   Top 10 features correlated with '{target}':")
    top_corr = correlations.head(11)[1:]  # Exclude target itself
    for feat, corr in top_corr.items():
        print(f"   {feat:30s}: {corr:6.3f}")

    insights['top_correlations'] = top_corr.to_dict()

    # Identify multicollinearity
    print("\n  Checking for multicollinearity (correlation > 0.9):")
    corr_matrix = df[numeric_cols].corr()
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })

    if high_corr_pairs:
        for pair in high_corr_pairs:
            print(f"   ⚠️  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
        print(f"\n   💡 Recommendation: Consider removing one feature from highly correlated pairs")
    else:
        print("    No severe multicollinearity detected")

    insights['high_correlation_pairs'] = high_corr_pairs

    #Outlier Detection
    print("\n 5. OUTLIER DETECTION (IQR Method)")
    print("-" * 70)

    outlier_counts = {}
    for col in [target] + ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_counts[col] = outliers
            if outliers > 0:
                print(f"   {col:15s}: {outliers:4d} outliers ({outliers/len(df)*100:.1f}%)")

    insights['outlier_counts'] = outlier_counts

    # Feature Importance (Statistical Tests)
    print("\n 6. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 70)

    # Prepare features for analysis
    feature_cols = [col for col in numeric_cols if col != target and col not in
                    ['aqi_pm2_5', 'aqi_pm10', 'aqi_co', 'aqi_no2', 'aqi_o3', 'aqi_so2']]

    X = df[feature_cols].fillna(0)
    y = df[target]

    # F-statistic based selection, Measures linear relationship strength between each feature and target.
    print("\n  F-statistic based feature importance:")
    selector_f = SelectKBest(score_func=f_regression, k='all')
    selector_f.fit(X, y)

    f_scores = pd.DataFrame({
        'feature': feature_cols,
        'f_score': selector_f.scores_
    }).sort_values('f_score', ascending=False)

    print("\n   Top 15 features by F-score:")
    for idx, row in f_scores.head(15).iterrows():
        print(f"   {row['feature']:30s}: {row['f_score']:10.2f}")

    # mutual Information based selection, Measures nonlinear dependency between each feature and target.
    print("\n   🔍 Mutual Information based feature importance:")
    mi_scores = mutual_info_regression(X, y, random_state=42)

    mi_scores_df = pd.DataFrame({
        'feature': feature_cols,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)

    print("\n   Top 15 features by Mutual Information:")
    for idx, row in mi_scores_df.head(15).iterrows():
        print(f"   {row['feature']:30s}: {row['mi_score']:10.4f}")

    insights['f_scores'] = f_scores.to_dict('records')
    insights['mi_scores'] = mi_scores_df.to_dict('records')

    print("\n EDA COMPLETE")

    return insights


# FEATURE SELECTION
# ============================================================================

def select_best_features(df, target='aqi', n_features=20):


    print("FEATURE SELECTION")


    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude target and individual pollutant AQI columns
    exclude_cols = [target, 'aqi_pm2_5', 'aqi_pm10', 'aqi_co', 'aqi_no2', 'aqi_o3', 'aqi_so2']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    print(f"\n Total available features: {len(feature_cols)}")
    print(f"  Selecting top {n_features} features...\n")

    X = df[feature_cols].fillna(0)
    y = df[target]
#Run three selection methods
    #  F-statistic
    selector_f = SelectKBest(score_func=f_regression, k=min(n_features, len(feature_cols)))
    selector_f.fit(X, y)
    f_features = [feature_cols[i] for i in selector_f.get_support(indices=True)]

    #  Mutual Information
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_features = pd.Series(mi_scores, index=feature_cols).nlargest(n_features).index.tolist()

    #  Correlation
    correlations = X.corrwith(y).abs().nlargest(n_features).index.tolist()

    # Combine all results
    selected_features = list(set(f_features + mi_features + correlations))

    print(" Feature Selection Methods:")
    print(f"   • F-statistic top features: {len(f_features)}")
    print(f"   • Mutual Information top features: {len(mi_features)}")
    print(f"   • Correlation top features: {len(correlations)}")
    print(f"\n Final selected features: {len(selected_features)}")

    # Ensure critical pollutant features are included
    critical_features = ['pm2_5', 'pm10', 'no2', 'co', 'o3', 'so2']
    for feat in critical_features:
        if feat in df.columns and feat not in selected_features:
            selected_features.append(feat)
            print(f"  Added critical feature: {feat}")

    print(f"\n Final feature list ({len(selected_features)} features):")
    for i, feat in enumerate(sorted(selected_features), 1):
        print(f"   {i:2d}. {feat}")



    return selected_features



# DATA VALIDATION FOR HOPSWORKS
# ============================================================================
#Before uploading data to Hopsworks, this function cleans and standardizes the DataFrame to avoid upload errors.
def validate_dataframe_for_hopsworks(df):


    df_clean = df.copy()
    issues_fixed = []

    # Check for MultiIndex
    if isinstance(df_clean.index, pd.MultiIndex):
        print("⚠️  Issue: MultiIndex detected")
        df_clean = df_clean.reset_index(drop=True)
        issues_fixed.append("Reset MultiIndex")

    # Check for MultiIndex columns
    if isinstance(df_clean.columns, pd.MultiIndex):
        print("⚠️  Issue: MultiIndex columns detected")
        df_clean.columns = ['_'.join(map(str, col)).strip() for col in df_clean.columns.values]
        issues_fixed.append("Flattened MultiIndex columns")

    # Check for duplicate column names
    if df_clean.columns.duplicated().any():
        print("⚠️  Issue: Duplicate column names detected")
        df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]
        issues_fixed.append("Removed duplicate columns")

    # Check for mixed types in columns
    print("\n🔍 Checking column data types...")
    for col in df_clean.columns:
        # Get unique types in column
        types_in_col = df_clean[col].apply(type).unique()
        if len(types_in_col) > 2:  # More than 2 types (including NaN)
            print(f"⚠️  Issue: Mixed types in column '{col}': {types_in_col}")
            # Try to convert to most common type
            if df_clean[col].dtype == 'object':
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    issues_fixed.append(f"Converted '{col}' to numeric")
                except:
                    df_clean[col] = df_clean[col].astype(str)
                    issues_fixed.append(f"Converted '{col}' to string")

    # Handle datetime columns
    datetime_cols = df_clean.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) > 0:
        print(f"\n Processing {len(datetime_cols)} datetime columns...")
        for col in datetime_cols:
            # Convert to Unix timestamp (milliseconds since epoch)
            df_clean[col] = df_clean[col].astype('int64') // 10**6
            issues_fixed.append(f"Converted '{col}' to timestamp")

    #Handle categorical columns
    categorical_cols = df_clean.select_dtypes(include=['category']).columns
    if len(categorical_cols) > 0:
        print(f"\n  Processing {len(categorical_cols)} categorical columns...")
        for col in categorical_cols:
            df_clean[col] = df_clean[col].astype(str)
            issues_fixed.append(f"Converted '{col}' category to string")

    # Handle object columns (ensure they are all strings)
    object_cols = df_clean.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"\n📝 Processing {len(object_cols)} object columns...")
        for col in object_cols:
            df_clean[col] = df_clean[col].fillna('unknown').astype(str)

    #Ensure numeric columns are float64 or int64
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].dtype not in ['int64', 'float64']:
            if df_clean[col].apply(lambda x: x == int(x) if pd.notna(x) else True).all():
                df_clean[col] = df_clean[col].astype('int64')
            else:
                df_clean[col] = df_clean[col].astype('float64')

    # Handle NaN values
    nan_count = df_clean.isnull().sum().sum()
    if nan_count > 0:
        print(f"\n Found {nan_count} NaN values, filling with defaults...")
        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                df_clean[col] = df_clean[col].fillna(0)
            else:
                df_clean[col] = df_clean[col].fillna('unknown')
        issues_fixed.append("Filled NaN values")

    # Check for infinity values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(df_clean[col]).sum()
        if inf_count > 0:
            print(f" Found {inf_count} infinity values in '{col}'")
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            issues_fixed.append(f"Replaced infinity in '{col}'")

    # Validate column names (no special characters except underscore)
    invalid_cols = [col for col in df_clean.columns if not col.replace('_', '').isalnum()]
    if invalid_cols:
        print(f"\n Invalid column names detected: {invalid_cols}")
        for col in invalid_cols:
            new_col = ''.join(c if c.isalnum() or c == '_' else '_' for c in col)
            df_clean = df_clean.rename(columns={col: new_col})
        issues_fixed.append("Sanitized column names")

    # Final validation
    print(f"\nVALIDATION COMPLETE")
    if issues_fixed:
        print(f"\n🔧 Issues fixed ({len(issues_fixed)}):")
        for i, issue in enumerate(issues_fixed, 1):
            print(f"   {i}. {issue}")
    else:
        print("   No issues found - DataFrame is ready!")

    print(f"\n Final DataFrame Summary:")
    print(f"   • Shape: {df_clean.shape}")
    print(f"   • Columns: {len(df_clean.columns)}")
    print(f"   • Data types:")
    dtype_counts = df_clean.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"      - {dtype}: {count} columns")
    print(f"   • Missing values: {df_clean.isnull().sum().sum()}")
    print(f"   • Duplicate rows: {df_clean.duplicated().sum()}")

 

    return df_clean


# Update the upload function to use validation
#This function uploads your processed and validated air quality dataset (after feature engineering and AQI calculation) to the Hopsworks Feature Store, where it can be reused for training
def upload_to_hopsworks(df, feature_group_name="karachi_air_quality_features", version=1):


    try:
        import hopsworks
        import os

        print("\n Connecting to Hopsworks...")

        # # Login to Hopsworks
        # project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        project = hopsworks.login(
            project="pollution_cicd",
            api_key_value=os.getenv("HOPSWORKS_API_KEY")
         )
        print(f"    Connected to project: {project.name}")

        # Get feature store
        fs = project.get_feature_store()
        print(f"   ✓ Accessed feature store")

        #  Validate and prepare dataframe before uploading
        print(f"\n Validating DataFrame before upload...")
        df_upload = validate_dataframe_for_hopsworks(df)

      
        # Convert datetime to timestamp (int64) for better compatibility
        if 'datetime_utc' in df_upload.columns:

            if not np.issubdtype(df_upload['datetime_utc'].dtype, np.number):
                df_upload['datetime_utc'] = pd.to_datetime(df_upload['datetime_utc'], errors='coerce')
                df_upload = df_upload.dropna(subset=['datetime_utc'])
                df_upload['datetime_utc'] = (df_upload['datetime_utc'].astype('int64') // 10**6).astype('int64')
                print(f"    Converted datetime_utc to timestamp")
            else:
                print(f"    datetime_utc already numeric, skipping reconversion")
    
                print(f"    Converted datetime to timestamp")

        # Handle categorical columns - convert to string explicitly
        categorical_cols = df_upload.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            df_upload[col] = df_upload[col].fillna('unknown').astype(str)
            print(f"    Converted categorical column: {col}")

        # Ensure all numeric columns are proper numeric types
        numeric_cols = df_upload.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col != 'id' and col != 'datetime_utc':
                # Convert to float64 to avoid dtype issues
                df_upload[col] = pd.to_numeric(df_upload[col], errors='coerce').astype('float64')

        # Fill any remaining NaN values
        df_upload = df_upload.fillna(0)

        # Verify dataframe integrity
        print(f"\n🔍 Verifying data integrity...")
        print(f"   • Data types check: {df_upload.dtypes.nunique()} unique types")
        print(f"   • Missing values: {df_upload.isnull().sum().sum()}")
        print(f"   • Duplicate rows: {df_upload.duplicated().sum()}")

        # Remove duplicates if any
        if df_upload.duplicated().sum() > 0:
            df_upload = df_upload.drop_duplicates()
            print(f"   ✓ Removed duplicate rows")

        print(f"\n Uploading data...")
        print(f"   • Feature group name: {feature_group_name}")
        print(f"   • Version: {version}")
        print(f"   • Rows: {len(df_upload)}")
        print(f"   • Columns: {len(df_upload.columns)}")
        print(f"   • Columns: {', '.join(df_upload.columns.tolist()[:10])}...")

        # Create or get feature group (without event_time if datetime was converted)
        fg_params = {
            "name": feature_group_name,
            "version": version,
            "primary_key": ["datetime_utc"],
            "description": "Air Quality data with calculated AQI (US EPA 1-500 scale) and engineered features",
            "online_enabled": False
        }

        # Only add event_time if datetime column exists
        if 'datetime_utc' in df_upload.columns:
            fg_params["event_time"] = "datetime_utc"

        feature_group = fs.get_or_create_feature_group(**fg_params)

        # Insert data with proper options
        print(f"\n Inserting data (this may take a few moments)...")
        feature_group.insert(df_upload, write_options={"wait_for_job": False})

        print(f"\n DATA SUCCESSFULLY UPLOADED TO HOPSWORKS!")
        print(f"   Feature group: {feature_group_name} (v{version})")
        print(f"   Note: Data ingestion job is running in background")
        print("="*70)

        return feature_group

    except ImportError:
        print("\n ERROR: hopsworks package not installed")
        print("   Install with: pip install hopsworks")
        return None

    except Exception as e:
        import traceback
        print(f"\n ERROR uploading to Hopsworks: {str(e)}")
        print("\n Detailed error traceback:")
        print(traceback.format_exc())
        print("\n Troubleshooting tips:")
        print("   1. Check if your API key is valid")
        print("   2. Ensure all column dtypes are compatible (int, float, str)")
        print("   3. Check for special characters in column names")
        print("   4. Verify no mixed-type columns exist")
        return None

#Retrieves your uploaded dataset from Hopsworks Feature Sto
def fetch_from_hopsworks(feature_group_name="karachi_air_quality_features", version=1):


    try:
        import hopsworks
        import os

        print("\n Connecting to Hopsworks...")

        # Login to Hopsworks
        # project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        project = hopsworks.login(
            project="pollution_cicd",
            api_key_value=os.getenv("HOPSWORKS_API_KEY")
         )
        print(f"   ✓ Connected to project: {project.name}")

        # Get feature store
        fs = project.get_feature_store()

        print(f"\n Fetching feature group...")
        print(f"   • Feature group name: {feature_group_name}")
        print(f"   • Version: {version}")

        # Get feature group
        feature_group = fs.get_feature_group(
            name=feature_group_name,
            version=version
        )

        # Read data
        df = feature_group.read()

        print(f"\n DATA SUCCESSFULLY RETRIEVED!")
        print(f"   • Rows: {len(df)}")
        print(f"   • Columns: {len(df.columns)}")
        print("="*70)

        return df

    except ImportError:
        print("\n ERROR: hopsworks package not installed")
        print("   Install with: pip install hopsworks")
        return None

    except Exception as e:
        print(f"\n ERROR fetching from Hopsworks: {str(e)}")
        return None


# ============================================================================
# VISUALIZATION FUNCTIONS (Optional but helpful)
# ============================================================================

# def create_eda_visualizations(df, target='aqi', save_plots=False):
#     """
#     Create comprehensive EDA visualizations

#     Parameters:
#     -----------
#     df : pandas DataFrame
#         Processed data
#     target : str
#         Target variable name
#     save_plots : bool
#         Whether to save plots to files
#     """
#     print("\n📊 Creating EDA visualizations...")

#     # Set style
#     sns.set_style("whitegrid")
#     plt.rcParams['figure.figsize'] = (15, 10)

#     # 1. Target Distribution
#     fig, axes = plt.subplots(2, 2, figsize=(15, 10))

#     # Histogram
#     axes[0, 0].hist(df[target], bins=50, edgecolor='black', alpha=0.7)
#     axes[0, 0].set_title(f'{target.upper()} Distribution', fontsize=14, fontweight='bold')
#     axes[0, 0].set_xlabel('AQI Value')
#     axes[0, 0].set_ylabel('Frequency')

#     # Box plot
#     axes[0, 1].boxplot(df[target])
#     axes[0, 1].set_title(f'{target.upper()} Box Plot', fontsize=14, fontweight='bold')
#     axes[0, 1].set_ylabel('AQI Value')

#     # Q-Q plot
#     stats.probplot(df[target], dist="norm", plot=axes[1, 0])
#     axes[1, 0].set_title('Q-Q Plot', fontsize=14, fontweight='bold')

#     # AQI Category Distribution
#     if 'aqi_category' in df.columns:
#         df['aqi_category'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 1])
#         axes[1, 1].set_title('AQI Category Distribution', fontsize=14, fontweight='bold')
#         axes[1, 1].set_xlabel('Category')
#         axes[1, 1].set_ylabel('Count')
#         axes[1, 1].tick_params(axis='x', rotation=45)

#     plt.tight_layout()
#     if save_plots:
#         plt.savefig('eda_target_distribution.png', dpi=300, bbox_inches='tight')
#     plt.show()

#     # 2. Correlation Heatmap
#     numeric_cols = df.select_dtypes(include=[np.number]).columns[:15]  # Top 15 features
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm',
#                 center=0, square=True, linewidths=1)
#     plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
#     plt.tight_layout()
#     if save_plots:
#         plt.savefig('eda_correlation_heatmap.png', dpi=300, bbox_inches='tight')
#     plt.show()

#     print("   ✓ Visualizations created!")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def complete_eda_and_upload_pipeline(df_processed, selected_features_count=20):

    # Step 1: Comprehensive EDA
    insights = perform_comprehensive_eda(df_processed, target='aqi')

    # Step 2: Feature Selection
    selected_features = select_best_features(
        df_processed,
        target='aqi',
        n_features=selected_features_count
    )

    # Step 3: Prepare final dataset with selected features

    # Prepare final columns (no need for 'id' anymore)
    final_cols = ['datetime_utc'] if 'datetime_utc' in df_processed.columns else []
    final_cols += selected_features + ['aqi', 'aqi_category', 'dominant_pollutant']
    
    df_final = df_processed[final_cols].copy()


    print(f"\n Final dataset prepared with {len(df_final.columns)} columns")

    # Step 4: Upload to Hopsworks
    feature_group = upload_to_hopsworks(
        df_final,
        feature_group_name="karachi_air_quality_features",
        version=1
    )

    return selected_features, feature_group



# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":


    # Run complete EDA and upload pipeline
    selected_features, feature_group = complete_eda_and_upload_pipeline(
        df_processed,
        selected_features_count=20
    )

    print("\n" + "="*70)
    print("PART 2 COMPLETE - Ready for Model Training!")
    print("="*70)
    print("\nNext steps:")
    print("1.  Data preprocessed and cleaned")
    print("2.  Comprehensive EDA performed")
    print("3.  Best features selected")
    print("4.  Data uploaded to Hopsworks Feature Store")
    print("5.   Ready for Part 3: Model Training & Deployment")

    print("="*70)








