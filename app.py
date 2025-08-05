import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
from datetime import datetime
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches

warnings.filterwarnings('ignore')

# Set up matplotlib for Streamlit
plt.style.use('default')
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 10

# Prophet model (optional)
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Configuration for different sectors
SECTORS = {
    "Makro-Mikro İqtisadi Göstəricilər": {
        "models": {
            "İqtisadi Sahələr üzrə ÜDM": {
                "file": "Kend_Teserrufati_Saheleruzre.xlsx",
                "data_type": "excel",
                "date_column": "Year",
                "date_range": (2000, 2023),
                "categories": [
                    "Cəmi",
                    "sənaye",
                    "kənd təsərrüfatı, meşə təsərrüfatı və balıqçılıq",
                    "tikinti",
                    "nəqliyyat və rabitə",
                    "xalis vergilər",
                    "digər sahələr"
                ]
            },
            "Neft və Enerji Göstəriciləri": {  # NEW MODEL
                "file": "OilAze.csv",
                "data_type": "csv",
                "date_column": "observation_date",
                "date_range": (2000, 2025),
                "categories": [
                    "Breakeven Fiscal Oil Price for Azerbaijan",
                    "Crude Oil Exports for Azerbaijan"
                ]
            },
            "Ümumi İqtisadi Göstəricilər": {  # NEW MODEL
                "file": "data.csv",
                "data_type": "csv",
                "date_column": "Year",
                "date_range": (1995, 2024),
                "categories": [
                    "Ümumi daxili məhsul",
                    "Sənaye məhsulu",
                    "Əsas kapitala yönəldilən vəsaitlər",
                    "Kənd təsərrüfatı məhsulu",
                    "Pərakəndə əmtəə dövriyyəsi",
                    "Əhaliyə göstərilən ödənişli xidmətlər",
                    "İnformasiya və rabitə xidmətləri",
                    "Nəqliyyat sektorunda yük daşınması",
                    "Orta aylıq nominal əmək haqqı"
                ]
            }
        }
    },
    "Kənd Təsərrüfatı": {
        "models": {
            "Əhali və Ərazi Göstəriciləri": {
                "file": "Ehali_ve_Erazi.xlsx",
                "data_type": "excel",
                "date_column": "Year",
                "date_range": (1990, 2023),
                "categories": [
                    "1 km² əraziyə adam düşür (nəfər)",
                    "100 ha-ya adam düşür (nəfər)",
                    "Adambaşına kənd təsərrüfatına yararlı torpaq (ha)",
                    "Əkin yeri (min ha)",
                    "Adambaşına əkin yeri (ha)",
                    "Dincə qoyulmuş torpaqlar (min ha)",
                    "Çoxillik əkmələr (min ha)",
                    "Biçənək və örüş-otlaq sahələri (min ha)",
                    "Meşə ilə örtülü sahələr (min ha)"
                ]
            },
            "Mineral Gübrələrin Verilməsi": {  # NEW MODEL
                "file": "Mineral gübrələrin verilməsi.csv",
                "data_type": "csv",
                "date_column": "Il",
                "date_range": (1990, 2018),
                "categories": [
                    "Cəmi, min ton",
                    "Gübrələnmiş sahənin ümumi əkində xüsusi çəkisi %",
                    "Hər hektar əkinə, kq",
                    "Kartof, kq",
                    "Meyvə bağları, kq",
                    "Pambıq, kq",
                    "Taxıl, kq",
                    "Tütün, kq",
                    "Tərəvəz və bostan bitkiləri, kq",
                    "Üzümlüklər, kq",
                    "Yem bitkiləri, kq"
                ]
            }
        }
    },
    "Səhiyyə": {
        "models": {
            "Həkimlərin Sayı": {
                "file": "output_cleaned.csv",
                "data_type": "csv",
                "date_column": "Tarix",
                "date_range": (2000, 2023),
                "categories": [
                    "Həkimlərin sayı - cəmi",
                    "terapevtlər",
                    "cərrahlar",
                    "pediatrlar",
                    "stomatoloq və diş həkimləri"
                ]
            },
            "İnfeksiya və Parazit Xəstəlikləri": {
                "file": "infeksion_və_parazit_xəstəlikləri.csv",
                "data_type": "csv",
                "date_column": "Illər",
                "date_range": (2000, 2023),
                "categories": [
                    "Göyöskürək",
                    "Pedikulyoz",
                    "Qarayara",
                    "Qrip və yuxarı tənəffüs yollarının kəskin infeksiyası",
                    "Qızılça",
                    "Suçiçəyi",
                    "Viruslu hepatitlər",
                    "Ümumi kəskin bağırsaq infeksiyaları"
                ]
            },
            "Vərəm Xəstəliyi": {
                "file": "vərəm_xəstəliyi.csv",
                "data_type": "csv",
                "date_column": "Illər",
                "date_range": (2000, 2023),
                "categories": [
                    "0-13 yaşlı - cəmi",
                    "14-17 yaşlı - cəmi",
                    "18-29 yaşlı - cəmi",
                    "30-44 yaşlı - cəmi",
                    "45-64 yaşlı - cəmi",
                    "65 və yuxarı yaşda - cəmi",
                    "kişilər",
                    "qadınlar",
                    "İlk dəfə qoyulmuş diaqnozla qeydə alınmış xəstələrin sayı- cəmi, nəfər",
                    "Əhalinin hər 100 000 nəfərinə -cəmi (müvafiq cins və yaş qruplarına görə)"
                ]
            }
        }
    }
}


def load_comprehensive_macro_data():
    """Load comprehensive macroeconomic data from CSV file - NEW FUNCTION"""
    try:
        # Read the comprehensive macro data CSV
        df_raw = pd.read_csv('data.csv')

        # The data structure: first column has indicator names, subsequent columns are years
        # We need to transpose this data to make years the index

        # Get the indicator names from the first column
        indicators = df_raw.iloc[:, 0].tolist()

        # Get year columns (skip the first column which has indicator names)
        year_columns = df_raw.columns[1:].tolist()

        # Create a dictionary to store the transposed data
        transposed_data = {}

        # Add a Year column
        transposed_data['Year'] = [int(year) for year in year_columns]

        # For each indicator, extract its values across years
        for idx, indicator in enumerate(indicators):
            values = []
            for year_col in year_columns:
                try:
                    # Get the value for this indicator and year
                    raw_value = df_raw.iloc[idx][year_col]

                    # Handle different data formats (comma as decimal separator, etc.)
                    if isinstance(raw_value, str):
                        # Replace comma with dot for decimal conversion
                        cleaned_value = raw_value.replace(',', '.')
                        value = float(cleaned_value)
                    else:
                        value = float(raw_value) if pd.notna(raw_value) else np.nan

                    values.append(value)
                except (ValueError, TypeError):
                    values.append(np.nan)

            transposed_data[indicator] = values

        # Create DataFrame from transposed data
        df = pd.DataFrame(transposed_data)

        # Convert Year to datetime and set as index
        df['Date'] = pd.to_datetime(df['Year'], format='%Y')
        df.set_index('Date', inplace=True)
        df.drop('Year', axis=1, inplace=True)

        # Clean column names and ensure they match our categories
        expected_categories = [
            "Ümumi daxili məhsul",
            "Sənaye məhsulu",
            "Əsas kapitala yönəldilən vəsaitlər",
            "Kənd təsərrüfatı məhsulu",
            "Pərakəndə əmtəə dövriyyəsi",
            "Əhaliyə göstərilən ödənişli xidmətlər",
            "İnformasiya və rabitə xidmətləri",
            "Nəqliyyat sektorunda yük daşınması",
            "Orta aylıq nominal əmək haqqı"
        ]

        # Keep only expected categories that exist in the data
        available_categories = [cat for cat in expected_categories if cat in df.columns]
        df = df[available_categories]

        # Handle any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')

        # Sort by date
        df = df.sort_index()

        return df

    except Exception as e:
        st.error(f"Ümumi iqtisadi məlumatlar yüklənə bilmədi: {e}")
        import traceback
        st.error(f"Trace: {traceback.format_exc()}")
        return None


# Data loading functions
def load_gdp_data():
    """Load GDP data from Excel file (for Makro-Mikro İqtisadi Göstəricilər)"""
    try:
        df_raw = pd.read_excel('Kend_Teserrufati_Saheleruzre.xlsx', header=None)

        data = {
            'Year': [],
            'Cəmi': [],
            'sənaye': [],
            'kənd təsərrüfatı, meşə təsərrüfatı və balıqçılıq': [],
            'tikinti': [],
            'nəqliyyat və rabitə': [],
            'xalis vergilər': [],
            'digər sahələr': []
        }

        # Extract million manat data (rows 5-28)
        for i in range(5, 29):  # 2000-2023 data
            try:
                row = df_raw.iloc[i]
                if pd.notna(row[1]):
                    year_str = str(row[1]).replace('*', '').strip()
                    year = int(year_str)
                    data['Year'].append(year)
                    data['Cəmi'].append(float(row[2]) if pd.notna(row[2]) else np.nan)
                    data['sənaye'].append(float(row[3]) if pd.notna(row[3]) else np.nan)
                    data['kənd təsərrüfatı, meşə təsərrüfatı və balıqçılıq'].append(
                        float(row[4]) if pd.notna(row[4]) else np.nan)
                    data['tikinti'].append(float(row[5]) if pd.notna(row[5]) else np.nan)
                    data['nəqliyyat və rabitə'].append(float(row[6]) if pd.notna(row[6]) else np.nan)
                    data['xalis vergilər'].append(float(row[7]) if pd.notna(row[7]) else np.nan)
                    data['digər sahələr'].append(float(row[8]) if pd.notna(row[8]) else np.nan)
            except (ValueError, TypeError):
                continue

        df = pd.DataFrame(data)
        df['Tarix'] = pd.to_datetime(df['Year'], format='%Y')
        df.set_index('Tarix', inplace=True)
        df.drop('Year', axis=1, inplace=True)

        return df
    except Exception as e:
        st.error(f"İqtisadi göstəricilər məlumatları yüklənə bilmədi: {e}")
        return None


def load_mineral_fertilizer_data():
    """Load mineral fertilizer data from CSV file"""
    try:
        df = pd.read_csv('Mineral gübrələrin verilməsi.csv')

        # The CSV has a complex structure, let's handle it properly
        # Skip the empty first column and clean headers
        proper_columns = [
            'Il',  # Year
            'Cəmi, min ton',  # Total, thousand tons
            'Gübrələnmiş sahənin ümumi əkində xüsusi çəkisi %',  # Fertilized area percentage
            'Hər hektar əkinə, kq',  # Per hectare, kg
            'Kartof, kq',  # Potato, kg
            'Meyvə bağları, kq',  # Fruit gardens, kg
            'Pambıq, kq',  # Cotton, kg
            'Taxıl, kq',  # Grain, kg
            'Tütün, kq',  # Tobacco, kg
            'Tərəvəz və bostan bitkiləri, kq',  # Vegetables and garden plants, kg
            'Üzümlüklər, kq',  # Vineyards, kg
            'Yem bitkiləri, kq'  # Feed plants, kg
        ]

        # Read the CSV properly - it has 13 columns including the empty first one
        df_raw = pd.read_csv('Mineral gübrələrin verilməsi.csv', header=0)

        # Create a clean dataframe
        data = {}

        # Get the column indices (skip first empty column)
        col_indices = list(range(1, len(proper_columns) + 1))

        for i, col_name in enumerate(proper_columns):
            if i == 0:  # Year column
                data[col_name] = df_raw.iloc[:, 1].astype(int)
            else:
                data[col_name] = pd.to_numeric(df_raw.iloc[:, i + 1], errors='coerce')

        df_clean = pd.DataFrame(data)

        # Filter out invalid years and convert to datetime index
        df_clean = df_clean[df_clean['Il'].between(1990, 2018)]
        df_clean['Date'] = pd.to_datetime(df_clean['Il'], format='%Y')
        df_clean.set_index('Date', inplace=True)
        df_clean.drop('Il', axis=1, inplace=True)

        # Handle any remaining NaN values
        df_clean = df_clean.fillna(0)

        return df_clean

    except Exception as e:
        st.error(f"Mineral gübrə məlumatları yüklənə bilmədi: {e}")
        import traceback
        st.error(f"Trace: {traceback.format_exc()}")
        return None


# NEW FUNCTION: Load oil data
def load_oil_data():
    """Load oil and energy data from CSV file - NEW FUNCTION"""
    try:
        # Read the oil data CSV
        df = pd.read_csv('OilAze.csv')

        # Clean column names - remove extra spaces and make consistent
        df.columns = df.columns.str.strip()

        # Convert observation_date to datetime
        df['observation_date'] = pd.to_datetime(df['observation_date'])

        # Set date as index
        df.set_index('observation_date', inplace=True)

        # Clean and convert numeric columns
        numeric_columns = ['Breakeven Fiscal Oil Price for Azerbaijan', 'Crude Oil Exports for Azerbaijan']

        for col in numeric_columns:
            if col in df.columns:
                # Convert to numeric, handling any non-numeric values
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove any rows where all numeric data is NaN
        df = df.dropna(how='all', subset=numeric_columns)

        # Sort by date
        df = df.sort_index()

        return df

    except Exception as e:
        st.error(f"Neft məlumatları yüklənə bilmədi: {e}")
        import traceback
        st.error(f"Trace: {traceback.format_exc()}")
        return None

def load_population_territory_data():
    """Load population and territory data from CSV file for Streamlit"""
    try:
        # Read the CSV file
        df_raw = pd.read_csv('Ehali_Ve_Erazi.csv', header=None)

        # Based on the CSV structure, data starts at row 7 (0-indexed)
        data_start_row = 7

        # Extract only the data rows (from 1990 to 2023)
        data_rows = []

        for i in range(data_start_row, len(df_raw)):
            row = df_raw.iloc[i]

            # Check if this row has year data (column 1 should have the year)
            if pd.notna(row.iloc[1]):
                try:
                    year_val = int(float(str(row.iloc[1]).strip()))
                    if 1990 <= year_val <= 2023:
                        data_rows.append(row)
                    elif year_val < 1990:
                        # Stop when we hit years before 1990
                        break
                except:
                    # If we can't parse the year, this might be the end of data
                    break
            else:
                # Empty year column means end of data
                break

        if not data_rows:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(data_rows)
        df.reset_index(drop=True, inplace=True)

        # Set up proper column names based on the CSV structure
        # Only include columns that have meaningful data
        column_names = [
            'Empty',  # Column 0 is empty - will be dropped
            'Year',  # Column 1 has years
            'Əhali (min nəfər)',  # Column 2
            'Ərazi (min km²)',  # Column 3
            'Kənd təsərrüfatına yararlı torpaqlar (min ha)',  # Column 4
            '1 km² əraziyə adam düşür (nəfər)',  # Column 5
            '100 ha-ya adam düşür (nəfər)',  # Column 6
            'Adambaşına kənd təsərrüfatına yararlı torpaq (ha)',  # Column 7
            'Əkin yeri (min ha)',  # Column 8
            'Adambaşına əkin yeri (ha)',  # Column 9
            'Dincə qoyulmuş torpaqlar (min ha)',  # Column 10
            'Çoxillik əkmələr (min ha)',  # Column 11
            'Biçənək və örüş-otlaq sahələri (min ha)',  # Column 12
            'Meşə ilə örtülü sahələr (min ha)'  # Column 13
        ]

        # Adjust column names to match actual columns
        if len(column_names) > df.shape[1]:
            column_names = column_names[:df.shape[1]]
        elif len(column_names) < df.shape[1]:
            for i in range(len(column_names), df.shape[1]):
                column_names.append(f'Column_{i}')

        df.columns = column_names

        # Drop the empty first column
        if 'Empty' in df.columns:
            df = df.drop('Empty', axis=1)

        # Convert Year column to proper format
        df['Year'] = df['Year'].astype(int)

        # Convert Year to datetime and set as index
        df['Date'] = pd.to_datetime(df['Year'], format='%Y')
        df.set_index('Date', inplace=True)
        df.drop('Year', axis=1, inplace=True)

        # Convert all numeric columns to float, ensuring proper handling of small decimals
        for col in df.columns:
            # Convert to numeric, handling any string formatting and ensuring decimals are preserved
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

        # Fill any NaN values with interpolation
        df = df.interpolate(method='linear')

        # Remove any rows that are still all NaN
        df = df.dropna(how='all')

        # DEBUG: Print actual values for the problematic columns before filtering
        debug_cols = ['Adambaşına kənd təsərrüfatına yararlı torpaq (ha)', 'Adambaşına əkin yeri (ha)']
        for debug_col in debug_cols:
            if debug_col in df.columns:
                print(f"DEBUG - {debug_col}:")
                print(f"  Sample values: {df[debug_col].head().tolist()}")
                print(f"  Max value: {df[debug_col].max()}")
                print(f"  Min value: {df[debug_col].min()}")
                print(f"  All values: {df[debug_col].tolist()}")

        # Filter out columns that are problematic or have no valid category
        # Remove the first 3 columns as specified and keep only meaningful indicators
        valid_columns = []
        for col in df.columns:
            # Skip the problematic columns mentioned by user
            if col in ['Əhali (min nəfər)', 'Ərazi (min km²)', 'Kənd təsərrüfatına yararlı torpaqlar (min ha)']:
                continue

            # Check if column has meaningful variation (not all zeros or very small values)
            col_data = df[col].dropna()
            if len(col_data) > 0:
                # ALWAYS include the two specific columns that have small decimal values
                if col in ['Adambaşına kənd təsərrüfatına yararlı torpaq (ha)', 'Adambaşına əkin yeri (ha)']:
                    print(f"FORCING INCLUSION of {col} - Max value: {col_data.abs().max()}")
                    valid_columns.append(col)
                else:
                    # For other columns, use the original threshold
                    if col_data.abs().max() > 0.001:
                        valid_columns.append(col)

        # Keep only valid columns
        df = df[valid_columns]

        # Final debug print
        print(f"Final columns included: {df.columns.tolist()}")
        for debug_col in debug_cols:
            if debug_col in df.columns:
                print(f"Final {debug_col} values: {df[debug_col].tolist()}")

        return df

    except Exception as e:
        st.error(f"CSV faylını yükləməkdə xəta: {e}")
        return None

# UPDATED load_data function
@st.cache_data
def load_data(file_path, data_type, date_column, clear_cache=False):
    """Load and prepare data based on type - UPDATED WITH COMPREHENSIVE MACRO DATA"""
    try:
        if data_type == "excel":
            if "Kend_Teserrufati" in file_path:
                return load_gdp_data()
            elif "Ehali_ve_Erazi" in file_path:
                return load_population_territory_data()
        else:
            # Handle CSV files
            if "Mineral gübrələrin verilməsi" in file_path:
                return load_mineral_fertilizer_data()
            elif "OilAze" in file_path:
                return load_oil_data()
            elif "data.csv" in file_path:  # NEW CONDITION
                return load_comprehensive_macro_data()
            else:
                # Existing CSV handling for health data
                df = pd.read_csv(file_path)
                df[date_column] = pd.to_datetime(df[date_column], format='%Y')
                df.set_index(date_column, inplace=True)
                return df
    except Exception as e:
        st.error(f"Məlumat yüklənə bilmədi: {e}")
        import traceback
        st.error(f"Trace: {traceback.format_exc()}")
        return None

# Forecasting functions
def linear_trend_forecast(series, periods=5):
    """Linear trend forecasting"""
    if len(series) < 2:
        return np.array([series.iloc[-1]] * periods), None, None

    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(series), len(series) + periods).reshape(-1, 1)
    forecast = model.predict(future_X)

    residuals = y - model.predict(X)
    std_error = np.std(residuals)
    conf_int = np.column_stack([forecast - 1.96 * std_error, forecast + 1.96 * std_error])

    return forecast, conf_int, model


def arima_forecast(series, periods=5, order=(1, 1, 1)):
    """ARIMA forecasting"""
    if len(series) < 3:
        return linear_trend_forecast(series, periods)

    try:
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=periods)
        conf_int = fitted_model.get_forecast(steps=periods).conf_int()
        return forecast.values, conf_int.values, fitted_model
    except:
        return linear_trend_forecast(series, periods)


def prophet_forecast(series, periods=5):
    """Prophet forecasting if available"""
    if not PROPHET_AVAILABLE or len(series) < 3:
        return linear_trend_forecast(series, periods)

    try:
        df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })

        model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        model.fit(df)

        future = model.make_future_dataframe(periods=periods, freq='Y')
        forecast = model.predict(future)

        forecast_values = forecast.tail(periods)['yhat'].values
        conf_int = forecast.tail(periods)[['yhat_lower', 'yhat_upper']].values

        return forecast_values, conf_int, model
    except:
        return linear_trend_forecast(series, periods)


def random_forest_forecast(series, periods=5):
    """Random Forest forecasting"""
    if len(series) < 5:
        return linear_trend_forecast(series, periods)

    n_lags = min(5, len(series) // 2)
    X, y = [], []

    for i in range(n_lags, len(series)):
        X.append(series.values[i - n_lags:i])
        y.append(series.values[i])

    X, y = np.array(X), np.array(y)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    forecasts = []
    last_values = series.values[-n_lags:]

    for _ in range(periods):
        pred = model.predict([last_values])[0]
        forecasts.append(pred)
        last_values = np.append(last_values[1:], pred)

    return np.array(forecasts), None, model

# Also add this modified version of create_forecast_plot with debug info:

# Add this diagnostic function to your app.py to debug the issue
def debug_data_flow(sector, model_name, category, data, start_year, end_year):
    """Debug function to trace where the zeros are coming from"""
    
    print(f"=== DEBUG DATA FLOW ===")
    print(f"Sector: {sector}")
    print(f"Model: {model_name}")
    print(f"Category: {category}")
    print(f"Period: {start_year}-{end_year}")
    
    if data is None:
        print("ERROR: Data is None!")
        return
    
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")
    print(f"Data index range: {data.index.min()} to {data.index.max()}")
    
    if category not in data.columns:
        print(f"ERROR: Category '{category}' not found in columns!")
        print(f"Available columns: {data.columns.tolist()}")
        return
    
    print(f"\nRaw data for {category}:")
    print(data[category].head(10))
    print(f"Data type: {data[category].dtype}")
    print(f"Non-null count: {data[category].count()}")
    print(f"Min: {data[category].min()}")
    print(f"Max: {data[category].max()}")
    
    # Special handling for Əhali və Ərazi Göstəriciləri
    if model_name == "Əhali və Ərazi Göstəriciləri":
        effective_start_year = max(start_year, 1990)
        print(f"Effective start year: {effective_start_year}")
    else:
        effective_start_year = start_year
    
    # Filter data by selected period
    print(f"\nFiltering data from {effective_start_year} to {end_year}")
    filtered_data = data[data.index.year >= effective_start_year]
    filtered_data = filtered_data[filtered_data.index.year <= end_year]
    
    print(f"Filtered data shape: {filtered_data.shape}")
    print(f"Filtered data for {category}:")
    print(filtered_data[category])
    
    series = filtered_data[category]
    print(f"\nSeries before dropna:")
    print(f"Length: {len(series)}")
    print(f"Values: {series.values}")
    
    # Drop any NaN values
    series = series.dropna()
    print(f"\nSeries after dropna:")
    print(f"Length: {len(series)}")
    print(f"Values: {series.values}")
    
    if len(series) < 2:
        print("ERROR: Not enough data points after filtering!")
        return
    
    print(f"\nFinal series stats:")
    print(f"Min: {series.min()}")
    print(f"Max: {series.max()}")
    print(f"Mean: {series.mean()}")
    print(f"First value: {series.iloc[0]}")
    print(f"Last value: {series.iloc[-1]}")
    
    return series


def create_forecast_plot(sector, model_name, category, data, start_year, end_year, forecast_periods=5):
    """Create forecast visualization - UPDATED WITH ALL DATA TYPES"""

    # Special handling for different models
    if model_name == "Əhali və Ərazi Göstəriciləri":
        effective_start_year = max(start_year, 1990)
        max_end_year = 2023
        if start_year < 1990:
            st.warning(
                f"Əhali və Ərazi məlumatları 1990-ci ildən başlayır. Analiz {effective_start_year}-ci ildən başlayacaq.")
    elif model_name == "Mineral Gübrələrin Verilməsi":
        effective_start_year = max(start_year, 1990)
        max_end_year = 2018
        if start_year < 1990:
            st.warning(
                f"Mineral gübrə məlumatları 1990-ci ildən başlayır. Analiz {effective_start_year}-ci ildən başlayacaq.")
        if end_year > 2018:
            end_year = 2018
            st.warning(f"Mineral gübrə məlumatları 2018-ci ilə qədərdir. Analiz {end_year}-ci ilə qədər aparılacaq.")
    elif model_name == "Neft və Enerji Göstəriciləri":
        effective_start_year = max(start_year, 2000)
        max_end_year = 2025
        if start_year < 2000:
            st.warning(
                f"Neft və enerji məlumatları 2000-ci ildən başlayır. Analiz {effective_start_year}-ci ildən başlayacaq.")
        if end_year > 2025:
            end_year = 2025
            st.warning(f"Neft və enerji məlumatları 2025-ci ilə qədərdir. Analiz {end_year}-ci ilə qədər aparılacaq.")
    elif model_name == "Ümumi İqtisadi Göstəricilər":  # NEW CONDITION
        effective_start_year = max(start_year, 1995)
        max_end_year = 2024
        if start_year < 1995:
            st.warning(
                f"Ümumi iqtisadi məlumatlar 1995-ci ildən başlayır. Analiz {effective_start_year}-ci ildən başlayacaq.")
        if end_year > 2024:
            end_year = 2024
            st.warning(f"Ümumi iqtisadi məlumatlar 2024-cü ilə qədərdir. Analiz {end_year}-ci ilə qədər aparılacaq.")
    else:
        effective_start_year = start_year
        max_end_year = 2023

    # Filter data by selected period
    filtered_data = data[data.index.year >= effective_start_year]
    filtered_data = filtered_data[filtered_data.index.year <= end_year]

    if category not in filtered_data.columns:
        st.error(f"Kateqoriya '{category}' məlumatlarda tapılmadı!")
        return None

    series = filtered_data[category].dropna()

    if len(series) < 2:
        st.error("Analiz üçün kifayət qədər məlumat yoxdur!")
        return None

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot historical data
    ax.plot(series.index, series.values, 'o-', label='Tarixi məlumatlar',
            color='blue', linewidth=2, markersize=6)

    # Special highlighting for different data types
    if model_name == "Ümumi İqtisadi Göstəricilər":  # NEW HIGHLIGHTING
        # Highlight key milestone years
        milestone_years = [1995, 2000, 2005, 2010, 2015, 2020, 2024]

        for year in milestone_years:
            if year >= effective_start_year and year <= end_year:
                idx = series.index.year == year
                if any(idx):
                    point_idx = series.index[idx][0]
                    point_value = series[point_idx]
                    ax.plot(point_idx, point_value, 'o', color='red',
                            markersize=8, markeredgecolor='black')

                    # Add label with appropriate formatting
                    if "daxili məhsul" in category or "məhsulu" in category or "dövriyyəsi" in category or "vəsaitlər" in category:
                        if point_value >= 1000000:
                            label_text = f'{point_value / 1000000:.1f}M'
                        elif point_value >= 1000:
                            label_text = f'{point_value / 1000:.0f}K'
                        else:
                            label_text = f'{int(point_value)}'
                    elif "xidmətlər" in category:
                        if point_value >= 1000:
                            label_text = f'{point_value / 1000:.0f}K'
                        else:
                            label_text = f'{int(point_value)}'
                    elif "daşınması" in category:
                        if point_value >= 1000:
                            label_text = f'{point_value / 1000:.1f}K'
                        else:
                            label_text = f'{int(point_value)}'
                    elif "əmək haqqı" in category:
                        label_text = f'{int(point_value)}'
                    else:
                        label_text = f'{point_value:.0f}'

                    ax.annotate(label_text,
                                xy=(point_idx, point_value),
                                xytext=(0, 10),
                                textcoords='offset points',
                                fontsize=9,
                                ha='center',
                                va='bottom',
                                bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor='yellow', alpha=0.7))
    elif model_name == "Neft və Enerji Göstəriciləri":
        # Highlight key milestone years for oil data
        milestone_years = [2000, 2005, 2010, 2015, 2020, 2025]

        for year in milestone_years:
            if year >= effective_start_year and year <= end_year:
                idx = series.index.year == year
                if any(idx):
                    point_idx = series.index[idx][0]
                    point_value = series[point_idx]
                    ax.plot(point_idx, point_value, 'o', color='red',
                            markersize=8, markeredgecolor='black')

                    # Add label with appropriate formatting
                    if "Price" in category:
                        label_text = f'${int(point_value)}'
                    elif "Export" in category:
                        label_text = f'{int(point_value / 1000)}K'
                    else:
                        label_text = f'{point_value:.1f}'

                    ax.annotate(label_text,
                                xy=(point_idx, point_value),
                                xytext=(0, 10),
                                textcoords='offset points',
                                fontsize=9,
                                ha='center',
                                va='bottom',
                                bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor='yellow', alpha=0.7))
    elif model_name == "Mineral Gübrələrin Verilməsi":
        # Highlight key milestone years for mineral fertilizer data
        milestone_years = [1990, 1995, 2000, 2005, 2010, 2015, 2018]

        for year in milestone_years:
            if year >= effective_start_year and year <= end_year:
                idx = series.index.year == year
                if any(idx):
                    point_idx = series.index[idx][0]
                    point_value = series[point_idx]
                    ax.plot(point_idx, point_value, 'o', color='red',
                            markersize=8, markeredgecolor='black')

                    # Add label with appropriate formatting
                    if "%" in category:
                        label_text = f'{int(point_value)}%'
                    elif "min ton" in category:
                        label_text = f'{point_value:.1f}'
                    elif "kq" in category:
                        label_text = f'{int(point_value)}'
                    else:
                        label_text = f'{point_value:.1f}'

                    ax.annotate(label_text,
                                xy=(point_idx, point_value),
                                xytext=(0, 10),
                                textcoords='offset points',
                                fontsize=9,
                                ha='center',
                                va='bottom',
                                bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor='yellow', alpha=0.7))

    # Method configurations - UPDATED WITH COMPREHENSIVE MACRO DATA
    if sector == "Makro-Mikro İqtisadi Göstəricilər" and model_name == "İqtisadi Sahələr üzrə ÜDM":
        methods_config = {
            'Cəmi': ['arima', 'linear'],
            'sənaye': ['arima', 'rf', 'prophet'],
            'kənd təsərrüfatı, meşə təsərrüfatı və balıqçılıq': ['arima', 'rf', 'linear'],
            'tikinti': ['arima', 'prophet'],
            'nəqliyyat və rabitə': ['arima', 'linear'],
            'xalis vergilər': ['arima', 'prophet'],
            'digər sahələr': ['arima', 'prophet']
        }
    elif sector == "Makro-Mikro İqtisadi Göstəricilər" and model_name == "Neft və Enerji Göstəriciləri":
        methods_config = {
            'Breakeven Fiscal Oil Price for Azerbaijan': ['arima', 'prophet', 'rf'],
            'Crude Oil Exports for Azerbaijan': ['arima', 'prophet', 'rf']
        }
    elif sector == "Makro-Mikro İqtisadi Göstəricilər" and model_name == "Ümumi İqtisadi Göstəricilər":  # NEW CONFIG
        methods_config = {
            'Ümumi daxili məhsul': ['arima', 'prophet', 'rf'],
            'Sənaye məhsulu': ['arima', 'prophet', 'rf'],
            'Əsas kapitala yönəldilən vəsaitlər': ['arima', 'prophet', 'rf'],
            'Kənd təsərrüfatı məhsulu': ['arima', 'prophet', 'rf'],
            'Pərakəndə əmtəə dövriyyəsi': ['arima', 'prophet', 'rf'],
            'Əhaliyə göstərilən ödənişli xidmətlər': ['arima', 'prophet', 'rf'],
            'İnformasiya və rabitə xidmətləri': ['arima', 'prophet', 'rf'],
            'Nəqliyyat sektorunda yük daşınması': ['arima', 'prophet', 'rf'],
            'Orta aylıq nominal əmək haqqı': ['arima', 'prophet', 'linear']
        }
    elif sector == "Kənd Təsərrüfatı" and model_name == "Əhali və Ərazi Göstəriciləri":
        methods_config = {
            '1 km² əraziyə adam düşür (nəfər)': ['arima', 'prophet'],
            '100 ha-ya adam düşür (nəfər)': ['arima', 'prophet'],
            'Adambaşına kənd təsərrüfatına yararlı torpaq (ha)': ['arima', 'linear'],
            'Əkin yeri (min ha)': ['arima', 'rf'],
            'Adambaşına əkin yeri (ha)': ['arima', 'linear'],
            'Dincə qoyulmuş torpaqlar (min ha)': ['arima', 'prophet'],
            'Çoxillik əkmələr (min ha)': ['arima', 'linear'],
            'Biçənək və örüş-otlaq sahələri (min ha)': ['arima', 'rf'],
            'Meşə ilə örtülü sahələr (min ha)': ['arima', 'linear']
        }
    elif sector == "Kənd Təsərrüfatı" and model_name == "Mineral Gübrələrin Verilməsi":
        methods_config = {
            'Cəmi, min ton': ['arima', 'prophet'],
            'Gübrələnmiş sahənin ümumi əkində xüsusi çəkisi %': ['arima', 'linear'],
            'Hər hektar əkinə, kq': ['arima', 'prophet', 'rf'],
            'Kartof, kq': ['arima', 'prophet'],
            'Meyvə bağları, kq': ['arima', 'prophet'],
            'Pambıq, kq': ['arima', 'prophet', 'linear'],
            'Taxıl, kq': ['arima', 'prophet'],
            'Tütün, kq': ['arima', 'prophet', 'linear'],
            'Tərəvəz və bostan bitkiləri, kq': ['arima', 'prophet', 'rf'],
            'Üzümlüklər, kq': ['arima', 'prophet', 'linear'],
            'Yem bitkiləri, kq': ['arima', 'prophet']
        }
    else:
        # Default healthcare configurations
        methods_config = {
            "Həkimlərin sayı - cəmi": ['arima', 'linear'],
            "terapevtlər": ['arima', 'rf', 'prophet'],
            "cərrahlar": ['arima', 'rf', 'linear'],
            "pediatrlar": ['arima', 'linear'],
            "stomatoloq və diş həkimləri": ['arima', 'prophet']
        }

    # Only show forecasts if analyzing current data
    if end_year == max_end_year:
        # Future years
        future_years = pd.date_range(start=f'{series.index[-1].year + 1}', periods=forecast_periods, freq='Y')

        # Method info
        method_info = {
            'arima': {'name': 'ARIMA', 'color': 'red'},
            'prophet': {'name': 'Prophet', 'color': 'green'},
            'rf': {'name': 'Random Forest', 'color': 'orange'},
            'linear': {'name': 'Xətti Trend', 'color': 'purple'}
        }

        # Get methods to plot
        methods_to_plot = methods_config.get(category, ['arima', 'linear'])

        # Perform forecasting and plot
        for method in methods_to_plot:
            try:
                if method == 'arima':
                    forecast, conf_int, _ = arima_forecast(series, forecast_periods)
                elif method == 'prophet':
                    forecast, conf_int, _ = prophet_forecast(series, forecast_periods)
                elif method == 'rf':
                    forecast, conf_int, _ = random_forest_forecast(series, forecast_periods)
                elif method == 'linear':
                    forecast, conf_int, _ = linear_trend_forecast(series, forecast_periods)

                info = method_info[method]

                # Plot forecast line
                ax.plot(future_years, forecast, 'o--',
                        label=f'{info["name"]} proqnozu',
                        color=info['color'], linewidth=2, markersize=6)

                # Add confidence intervals
                if conf_int is not None:
                    try:
                        if hasattr(conf_int, 'values'):
                            conf_int = conf_int.values
                        if conf_int.ndim == 2 and conf_int.shape[1] >= 2:
                            ax.fill_between(future_years, conf_int[:, 0], conf_int[:, 1],
                                            alpha=0.2, color=info['color'])
                    except Exception as e:
                        print(f"Warning: Could not plot confidence interval for {method}: {e}")

            except Exception as e:
                print(f"Error in {method} forecasting: {e}")
                continue

    # Add trend line
    from scipy import stats
    years_numeric = np.arange(len(series))
    slope, intercept, r_value, p_value, std_err = stats.linregress(years_numeric, series.values)

    if end_year == max_end_year:
        all_years_numeric = np.arange(len(series) + forecast_periods)
        future_years = pd.date_range(start=f'{series.index[-1].year + 1}', periods=forecast_periods, freq='Y')
        all_years = list(series.index) + list(future_years)
    else:
        all_years_numeric = np.arange(len(series))
        all_years = list(series.index)

    trend_line = slope * all_years_numeric + intercept

    ax.plot(all_years, trend_line, '--', color='gray', alpha=0.7,
            label=f'Trend (R²={r_value ** 2:.3f})')

    # Styling
    title = f'{category} - Zaman Seriyası Analizi ({effective_start_year}-{end_year})'
    if end_year == max_end_year:
        title += ' və Proqnoz'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set appropriate Y-axis label based on category type - UPDATED WITH ALL UNITS
    if "Price" in category:
        ax.set_ylabel('ABŞ Dolları ($)', fontsize=12)
    elif "Export" in category:
        ax.set_ylabel('Barel / Gün', fontsize=12)
    elif "daxili məhsul" in category or "məhsulu" in category or "dövriyyəsi" in category or "vəsaitlər" in category:
        ax.set_ylabel('Min Manat', fontsize=12)
    elif "xidmətlər" in category:
        ax.set_ylabel('Min Manat', fontsize=12)
    elif "daşınması" in category:
        ax.set_ylabel('Min Ton', fontsize=12)
    elif "əmək haqqı" in category:
        ax.set_ylabel('Manat', fontsize=12)
    elif "min ton" in category:
        ax.set_ylabel('Min Ton', fontsize=12)
    elif "%" in category:
        ax.set_ylabel('Faiz (%)', fontsize=12)
    elif "kq" in category:
        ax.set_ylabel('Kiloqram', fontsize=12)
    elif sector == "Makro-Mikro İqtisadi Göstəricilər" and model_name == "İqtisadi Sahələr üzrə ÜDM":
        ax.set_ylabel('Milyon Manat', fontsize=12)
    else:
        ax.set_ylabel('Dəyər', fontsize=12)

    ax.set_xlabel('İl', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add statistics box - UPDATED WITH ALL FORMATTING
    last_value = series.values[-1]
    mean_value = series.mean()

    # Format values based on category type
    if "Price" in category:
        last_value_str = f"${int(last_value)}"
        mean_value_str = f"${int(mean_value)}"
    elif "Export" in category:
        last_value_str = f"{int(last_value / 1000)}K barel/gün"
        mean_value_str = f"{int(mean_value / 1000)}K barel/gün"
    elif "daxili məhsul" in category or "məhsulu" in category or "dövriyyəsi" in category or "vəsaitlər" in category:
        if last_value >= 1000000:
            last_value_str = f"{last_value / 1000000:.1f}M"
            mean_value_str = f"{mean_value / 1000000:.1f}M"
        elif last_value >= 1000:
            last_value_str = f"{last_value / 1000:.0f}K"
            mean_value_str = f"{mean_value / 1000:.0f}K"
        else:
            last_value_str = f"{int(last_value)}"
            mean_value_str = f"{int(mean_value)}"
    elif "xidmətlər" in category:
        if last_value >= 1000:
            last_value_str = f"{last_value / 1000:.0f}K"
            mean_value_str = f"{mean_value / 1000:.0f}K"
        else:
            last_value_str = f"{int(last_value)}"
            mean_value_str = f"{int(mean_value)}"
    elif "daşınması" in category:
        if last_value >= 1000:
            last_value_str = f"{last_value / 1000:.1f}K"
            mean_value_str = f"{mean_value / 1000:.1f}K"
        else:
            last_value_str = f"{int(last_value)}"
            mean_value_str = f"{int(mean_value)}"
    elif "əmək haqqı" in category:
        last_value_str = f"{int(last_value)}"
        mean_value_str = f"{int(mean_value)}"
    elif "%" in category:
        last_value_str = f"{int(last_value)}%"
        mean_value_str = f"{int(mean_value)}%"
    elif "min ton" in category:
        last_value_str = f"{last_value:.1f}"
        mean_value_str = f"{mean_value:.1f}"
    elif "kq" in category:
        last_value_str = f"{int(last_value)}"
        mean_value_str = f"{int(mean_value)}"
    elif abs(last_value) < 1:
        last_value_str = f"{last_value:.3f}"
        mean_value_str = f"{mean_value:.3f}"
    elif abs(last_value) < 100:
        last_value_str = f"{last_value:.1f}"
        mean_value_str = f"{mean_value:.1f}"
    else:
        last_value_str = f"{int(last_value)}"
        mean_value_str = f"{int(mean_value)}"

    stats_text = f"Son dəyər: {last_value_str}\n"
    stats_text += f"Orta: {mean_value_str}\n"
    stats_text += f"Trend: {slope:.4f}/il\n"
    stats_text += f"Dövr: {effective_start_year}-{end_year}"

    # Add special notes for different models
    if model_name == "Mineral Gübrələrin Verilməsi":
        stats_text += f"\nMəlumat dövrü: 1990-2018"
    elif model_name == "Neft və Enerji Göstəriciləri":
        stats_text += f"\nMəlumat dövrü: 2000-2025"
    elif model_name == "Ümumi İqtisadi Göstəricilər":
        stats_text += f"\nMəlumat dövrü: 1995-2024"

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    return fig

# UPDATED main function PDF mapping section
def get_pdf_file(selected_sector, selected_model, model_config):
    """Get appropriate PDF file for the selected model - CORRECTED VERSION"""
    pdf_file = None

    # Check based on sector and model combination
    if "Makro" in selected_sector or "İqtisadi" in selected_sector:
        if "İqtisadi" in selected_model or "UDM" in selected_model:
            pdf_file = "Birləşdirilmiş_Hesabat_Agriculture.pdf"
        elif "Neft" in selected_model or "Enerji" in selected_model:
            pdf_file = "Enerji_Makroiqtisadi_Birlesmis.pdf"
        elif "Ümumi" in selected_model or "Göstəricilər" in selected_model:  # NEW CONDITION
            pdf_file = "Makroiqtisadi_Hesabat.pdf"
    elif "Kənd" in selected_sector or "Təsərrüfat" in selected_sector:
        if "Əhali" in selected_model or "Ərazi" in selected_model:
            pdf_file = "Birləşdirilmiş_Hesabat_Demoqrafiya.pdf"
        elif "Mineral" in selected_model or "Gübrə" in selected_model:
            pdf_file = "Mineral_Gübrələr_üzrə_Detallı_Hesabat.pdf"
    elif "Səhiyyə" in selected_sector:
        if "Həkim" in selected_model or "həkim" in selected_model:
            pdf_file = "Birləşdirilmiş_Hesabat_Hekimler.pdf"
        elif "İnfeksiya" in selected_model or "infeksiya" in selected_model or "parazit" in selected_model:
            pdf_file = "Səhiyyə_İnfeksiya_və_Parazit_Xəstəlikləri_Hesabat.pdf"
        elif "Vərəm" in selected_model or "vərəm" in selected_model:
            pdf_file = "Vərəm_Xəstəliyi_Hesabat.pdf"

    # Alternative approach - check by file patterns if above doesn't work
    if not pdf_file:
        if "Ehali_Ve_Erazi" in model_config.get("file", ""):
            pdf_file = "Birləşdirilmiş_Hesabat_Demoqrafiya.pdf"
        elif "Kend_Teserrufati" in model_config.get("file", ""):
            pdf_file = "Birləşdirilmiş_Hesabat_Agriculture.pdf"
        elif "Mineral gübrələrin verilməsi" in model_config.get("file", ""):
            pdf_file = "Mineral_Gübrələr_üzrə_Detallı_Hesabat.pdf"
        elif "OilAze" in model_config.get("file", ""):
            pdf_file = "Enerji_Makroiqtisadi_Birlesmis.pdf"
        elif "data.csv" in model_config.get("file", ""):  # NEW CONDITION
            pdf_file = "Makroiqtisadi_Hesabat.pdf"
        elif "infeksion" in model_config.get("file", ""):
            pdf_file = "Səhiyyə_İnfeksiya_və_Parazit_Xəstəlikləri_Hesabat.pdf"
        elif "vərəm" in model_config.get("file", ""):
            pdf_file = "Vərəm_Xəstəliyi_Hesabat.pdf"

    return pdf_file


def main():
    st.set_page_config(page_title="Azərbaycan Analitik Sistem", layout="wide")

    st.title("📊 Azərbaycan Analitik Sistem")
    st.markdown("---")

    # Debug and cache options
    with st.sidebar:
        debug_mode = st.checkbox("Debug rejimi", value=False)
        clear_cache = st.checkbox("Keşi təmizlə", value=False)
        if clear_cache:
            st.cache_data.clear()
            st.success("Keş təmizləndi!")

    # Main sector selection
    st.sidebar.header("🏢 Sektor Seçimi")
    selected_sector = st.sidebar.selectbox(
        "Əsas Sektor:",
        list(SECTORS.keys())
    )

    # Model selection within sector
    st.sidebar.header("🔬 Model Seçimi")
    available_models = list(SECTORS[selected_sector]["models"].keys())
    selected_model = st.sidebar.selectbox(
        "Model:",
        available_models
    )

    # Get model configuration
    model_config = SECTORS[selected_sector]["models"][selected_model]

    # Time period selection with special handling for different models
    st.sidebar.header("📅 Zaman Dövrü Seçimi")
    min_year, max_year = model_config["date_range"]

    # Special handling for different data types
    if selected_model == "Mineral Gübrələrin Verilməsi":
        if max_year > 2018:
            max_year = 2018
            st.sidebar.info("ℹ️ Mineral gübrə məlumatları 2018-ci ilə qədərdir")
    elif selected_model == "Neft və Enerji Göstəriciləri":
        if max_year > 2025:
            max_year = 2025
            st.sidebar.info("ℹ️ Neft və enerji məlumatları 2025-ci ilə qədərdir")
    elif selected_model == "Ümumi İqtisadi Göstəricilər":  # NEW CONDITION
        if max_year > 2024:
            max_year = 2024
            st.sidebar.info("ℹ️ Ümumi iqtisadi məlumatlar 2024-cü ilə qədərdir")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_year = st.selectbox(
            "Başlanğıc İl:",
            options=list(range(min_year, max_year + 1)),
            index=0
        )

    with col2:
        end_year = st.selectbox(
            "Son İl:",
            options=list(range(start_year, max_year + 1)),
            index=len(list(range(start_year, max_year + 1))) - 1
        )

    # Category selection
    st.sidebar.header("📋 Kateqoriya Seçimi")
    selected_category = st.sidebar.selectbox(
        "Kateqoriya:",
        model_config["categories"]
    )

    # PDF Report Generation Section - UPDATED WITH ALL MODELS
    st.sidebar.header("📄 Hesabat Generasiyası")

    # Get PDF file using corrected helper function
    pdf_file = get_pdf_file(selected_sector, selected_model, model_config)

    # Debug information (remove after testing)
    if debug_mode:
        st.sidebar.write(f"Debug - Selected Sector: '{selected_sector}'")
        st.sidebar.write(f"Debug - Selected Model: '{selected_model}'")
        st.sidebar.write(f"Debug - Model Config File: '{model_config.get('file', 'N/A')}'")
        st.sidebar.write(f"Debug - Determined PDF: '{pdf_file}'")

    if pdf_file:
        try:
            with open(pdf_file, "rb") as file:
                pdf_data = file.read()

            st.sidebar.download_button(
                label="📥 PDF Hesabatı Yüklə",
                data=pdf_data,
                file_name=f"{selected_model}_Hesabat.pdf",
                mime="application/pdf",
                help=f"{selected_model} üçün tam hesabat PDF faylı"
            )

            st.sidebar.success("✅ PDF hesabat mövcuddur!")

        except FileNotFoundError:
            st.sidebar.error(f"❌ PDF faylı tapılmadı: {pdf_file}")
        except Exception as e:
            st.sidebar.error(f"❌ PDF yükləmə xətası: {str(e)}")
    else:
        st.sidebar.info("ℹ️ Bu seçim üçün PDF hesabat mövcud deyil")
        # Show current selection for debugging
        if debug_mode:
            st.sidebar.write("Available PDFs:")
            st.sidebar.write("- Birləşdirilmiş_Hesabat_Agriculture.pdf")
            st.sidebar.write("- Birləşdirilmiş_Hesabat_Demoqrafiya.pdf")
            st.sidebar.write("- Birləşdirilmiş_Hesabat_Hekimler.pdf")
            st.sidebar.write("- Mineral_Gübrələr_üzrə_Detallı_Hesabat.pdf")
            st.sidebar.write("- Enerji_Makroiqtisadi_Birlesmis.pdf")  # CORRECTED
            st.sidebar.write("- Səhiyyə_İnfeksiya_və_Parazit_Xəstəlikləri_Hesabat.pdf")
            st.sidebar.write("- Vərəm_Xəstəliyi_Hesabat.pdf")
            st.sidebar.write("- Makroiqtisadi_Hesabat.pdf")

    # Forecast settings - UPDATED FOR ALL DATA TYPES
    if selected_model == "Mineral Gübrələrin Verilməsi":
        forecast_max_year = 2018
    elif selected_model == "Neft və Enerji Göstəriciləri":
        forecast_max_year = 2025
    elif selected_model == "Ümumi İqtisadi Göstəricilər":  # NEW CONDITION
        forecast_max_year = 2024
    else:
        forecast_max_year = 2023

    if end_year == forecast_max_year:
        st.sidebar.header("🔮 Proqnoz Parametrləri")
        forecast_years = st.sidebar.selectbox(
            "Proqnoz müddəti (il):",
            options=[1, 2, 3, 4, 5],
            index=2
        )

        if selected_model == "Mineral Gübrələrin Verilməsi":
            st.sidebar.info("💡 Proqnozlar 2018-ci ilə qədər olan mineral gübrə məlumatları üçün göstərilir")
        elif selected_model == "Neft və Enerji Göstəriciləri":
            st.sidebar.info("💡 Proqnozlar 2025-ci ilə qədər olan neft və enerji məlumatları üçün göstərilir")
        elif selected_model == "Ümumi İqtisadi Göstəricilər":  # NEW INFO
            st.sidebar.info("💡 Proqnozlar 2024-cü ilə qədər olan ümumi iqtisadi məlumatlar üçün göstərilir")
        else:
            st.sidebar.info("💡 Proqnozlar yalnız 2023-cü ilə qədər olan məlumatlar üçün göstərilir")
    else:
        forecast_years = 3
        if selected_model == "Mineral Gübrələrin Verilməsi":
            st.sidebar.info("ℹ️ Proqnozları görmək üçün son il 2018 seçin")
        elif selected_model == "Neft və Enerji Göstəriciləri":
            st.sidebar.info("ℹ️ Proqnozları görmək üçün son il 2025 seçin")
        elif selected_model == "Ümumi İqtisadi Göstəricilər":  # NEW INFO
            st.sidebar.info("ℹ️ Proqnozları görmək üçün son il 2024 seçin")
        else:
            st.sidebar.info("ℹ️ Proqnozları görmək üçün son il 2023 seçin")

    # Main content area
    col1, col2 = st.columns([3, 1])

    with col1:
        period_text = f"({start_year}-{end_year})"
        if end_year == forecast_max_year:
            period_text += " + Proqnoz"

        st.header(f"📈 {selected_sector} - {selected_model}")
        st.subheader(f"Analiz: {selected_category} {period_text}")

        # Add special info boxes for different models
        if selected_model == "Mineral Gübrələrin Verilməsi":
            st.info("🌱 Bu model 1990-2018 dövrü üçün mineral gübrələrin istifadəsi haqqında məlumatları əhatə edir. "
                    "Məlumatlar kənd təsərrüfatı məhsullarının gübrə tələbatı və istifadə dinamikasını göstərir.")
        elif selected_model == "Neft və Enerji Göstəriciləri":
            st.info("🛢️ Bu model 2000-2025 dövrü üçün Azərbaycanın neft və enerji sektoruna aid əsas makroiqtisadi "
                    "göstəriciləri əhatə edir. Fiskal tarazlıq üçün neft qiyməti və xam neft ixracı göstəriciləri daxildir.")
        elif selected_model == "Ümumi İqtisadi Göstəricilər":  # NEW INFO BOX
            st.info("📊 Bu model 1995-2024 dövrü üçün Azərbaycanın ümumi makroiqtisadi göstəricilərini əhatə edir. "
                    "ÜDM, sənaye məhsulu, investisiyalar, pərakəndə satış və digər əsas iqtisadi göstəricilər daxildir.")

        # Load and display data
        data = load_data(
            model_config["file"],
            model_config["data_type"],
            model_config["date_column"]
        )

        if debug_mode and data is not None:
            st.write("Debug - Data structure:")
            st.write(f"Shape: {data.shape}")
            st.write(f"Index: {data.index[:5]}")
            st.write(f"Columns: {data.columns.tolist()}")

            if selected_category in data.columns:
                # Apply special filtering based on model type
                if selected_model == "Mineral Gübrələrin Verilməsi":
                    effective_start_year = max(start_year, 1990)
                    effective_end_year = min(end_year, 2018)
                elif selected_model == "Neft və Enerji Göstəriciləri":
                    effective_start_year = max(start_year, 2000)
                    effective_end_year = min(end_year, 2025)
                elif selected_model == "Ümumi İqtisadi Göstəricilər":  # NEW DEBUG
                    effective_start_year = max(start_year, 1995)
                    effective_end_year = min(end_year, 2024)
                else:
                    effective_start_year = start_year
                    effective_end_year = end_year

                filtered_data = data[data.index.year >= effective_start_year]
                filtered_data = filtered_data[filtered_data.index.year <= effective_end_year]
                st.write(f"Filtered data for {selected_category} ({effective_start_year}-{effective_end_year}):")
                st.write(filtered_data[selected_category].sort_index())

        if data is not None:
            # Create and display plot
            fig = create_forecast_plot(
                selected_sector,
                selected_model,
                selected_category,
                data,
                start_year,
                end_year,
                forecast_years
            )

            if fig:
                st.pyplot(fig)

                # Display statistics for selected period - UPDATED FOR ALL MODELS
                if selected_category in data.columns:
                    # Apply special filtering based on model type
                    if selected_model == "Mineral Gübrələrin Verilməsi":
                        effective_start_year = max(start_year, 1990)
                        effective_end_year = min(end_year, 2018)
                        if start_year < 1990 or end_year > 2018:
                            st.info(
                                f"📊 Statistikalar əlçatan məlumat dövrü üçün hesablanıb: {effective_start_year}-{effective_end_year}")
                    elif selected_model == "Neft və Enerji Göstəriciləri":
                        effective_start_year = max(start_year, 2000)
                        effective_end_year = min(end_year, 2025)
                        if start_year < 2000 or end_year > 2025:
                            st.info(
                                f"📊 Statistikalar əlçatan məlumat dövrü üçün hesablanıb: {effective_start_year}-{effective_end_year}")
                    elif selected_model == "Ümumi İqtisadi Göstəricilər":  # NEW STATS HANDLING
                        effective_start_year = max(start_year, 1995)
                        effective_end_year = min(end_year, 2024)
                        if start_year < 1995 or end_year > 2024:
                            st.info(
                                f"📊 Statistikalar əlçatan məlumat dövrü üçün hesablanıb: {effective_start_year}-{effective_end_year}")
                    else:
                        effective_start_year = start_year
                        effective_end_year = end_year

                    # Filter data by effective period
                    filtered_data = data[data.index.year >= effective_start_year]
                    filtered_data = filtered_data[filtered_data.index.year <= effective_end_year]
                    series = filtered_data[selected_category].dropna()

                    st.markdown("### 📊 Seçilmiş Dövrün Statistikaları")

                    col1_stats, col2_stats, col3_stats, col4_stats = st.columns(4)

                    with col1_stats:
                        # Format values based on category type - UPDATED WITH ALL UNITS
                        last_val = series.iloc[-1]
                        mean_val = series.mean()

                        if "Price" in selected_category:
                            st.metric("Son Dəyər", f"${int(last_val)}")
                            st.metric("Orta Dəyər", f"${int(mean_val)}")
                        elif "Export" in selected_category:
                            st.metric("Son Dəyər", f"{int(last_val / 1000)}K barel/gün")
                            st.metric("Orta Dəyər", f"{int(mean_val / 1000)}K barel/gün")
                        elif "daxili məhsul" in selected_category or "məhsulu" in selected_category or "dövriyyəsi" in selected_category or "vəsaitlər" in selected_category:
                            if last_val >= 1000000:
                                st.metric("Son Dəyər", f"{last_val / 1000000:.1f}M manat")
                                st.metric("Orta Dəyər", f"{mean_val / 1000000:.1f}M manat")
                            elif last_val >= 1000:
                                st.metric("Son Dəyər", f"{last_val / 1000:.0f}K manat")
                                st.metric("Orta Dəyər", f"{mean_val / 1000:.0f}K manat")
                            else:
                                st.metric("Son Dəyər", f"{int(last_val)} manat")
                                st.metric("Orta Dəyər", f"{int(mean_val)} manat")
                        elif "xidmətlər" in selected_category:
                            if last_val >= 1000:
                                st.metric("Son Dəyər", f"{last_val / 1000:.0f}K manat")
                                st.metric("Orta Dəyər", f"{mean_val / 1000:.0f}K manat")
                            else:
                                st.metric("Son Dəyər", f"{int(last_val)} manat")
                                st.metric("Orta Dəyər", f"{int(mean_val)} manat")
                        elif "daşınması" in selected_category:
                            if last_val >= 1000:
                                st.metric("Son Dəyər", f"{last_val / 1000:.1f}K ton")
                                st.metric("Orta Dəyər", f"{mean_val / 1000:.1f}K ton")
                            else:
                                st.metric("Son Dəyər", f"{int(last_val)} ton")
                                st.metric("Orta Dəyər", f"{int(mean_val)} ton")
                        elif "əmək haqqı" in selected_category:
                            st.metric("Son Dəyər", f"{int(last_val)} manat")
                            st.metric("Orta Dəyər", f"{int(mean_val)} manat")
                        elif "%" in selected_category:
                            st.metric("Son Dəyər", f"{int(last_val)}%")
                            st.metric("Orta Dəyər", f"{int(mean_val)}%")
                        elif "min ton" in selected_category:
                            st.metric("Son Dəyər", f"{last_val:.1f} min ton")
                            st.metric("Orta Dəyər", f"{mean_val:.1f} min ton")
                        elif "kq" in selected_category:
                            st.metric("Son Dəyər", f"{int(last_val)} kq")
                            st.metric("Orta Dəyər", f"{int(mean_val)} kq")
                        else:
                            st.metric("Son Dəyər", f"{last_val:.1f}")
                            st.metric("Orta Dəyər", f"{mean_val:.1f}")

                    with col2_stats:
                        max_val = series.max()
                        min_val = series.min()

                        # Apply same formatting logic for max/min values
                        if "Price" in selected_category:
                            st.metric("Maksimum", f"${int(max_val)}")
                            st.metric("Minimum", f"${int(min_val)}")
                        elif "Export" in selected_category:
                            st.metric("Maksimum", f"{int(max_val / 1000)}K barel/gün")
                            st.metric("Minimum", f"{int(min_val / 1000)}K barel/gün")
                        elif "daxili məhsul" in selected_category or "məhsulu" in selected_category or "dövriyyəsi" in selected_category or "vəsaitlər" in selected_category:
                            if max_val >= 1000000:
                                st.metric("Maksimum", f"{max_val / 1000000:.1f}M")
                                st.metric("Minimum", f"{min_val / 1000000:.1f}M")
                            elif max_val >= 1000:
                                st.metric("Maksimum", f"{max_val / 1000:.0f}K")
                                st.metric("Minimum", f"{min_val / 1000:.0f}K")
                            else:
                                st.metric("Maksimum", f"{int(max_val)}")
                                st.metric("Minimum", f"{int(min_val)}")
                        elif "%" in selected_category:
                            st.metric("Maksimum", f"{int(max_val)}%")
                            st.metric("Minimum", f"{int(min_val)}%")
                        elif "min ton" in selected_category:
                            st.metric("Maksimum", f"{max_val:.1f} min ton")
                            st.metric("Minimum", f"{min_val:.1f} min ton")
                        elif "kq" in selected_category:
                            st.metric("Maksimum", f"{int(max_val)} kq")
                            st.metric("Minimum", f"{int(min_val)} kq")
                        else:
                            st.metric("Maksimum", f"{max_val:.1f}")
                            st.metric("Minimum", f"{min_val:.1f}")

                    with col3_stats:
                        change = series.iloc[-1] - series.iloc[0]
                        std_val = series.std()

                        # Apply same formatting logic for change/std values
                        if "Price" in selected_category:
                            st.metric("Dövr Dəyişikliyi", f"${int(change)}")
                            st.metric("Standart Sapma", f"${int(std_val)}")
                        elif "Export" in selected_category:
                            st.metric("Dövr Dəyişikliyi", f"{int(change / 1000)}K")
                            st.metric("Standart Sapma", f"{int(std_val / 1000)}K")
                        elif "daxili məhsul" in selected_category or "məhsulu" in selected_category or "dövriyyəsi" in selected_category or "vəsaitlər" in selected_category:
                            if abs(change) >= 1000000:
                                st.metric("Dövr Dəyişikliyi", f"{change / 1000000:.1f}M")
                                st.metric("Standart Sapma", f"{std_val / 1000000:.1f}M")
                            elif abs(change) >= 1000:
                                st.metric("Dövr Dəyişikliyi", f"{change / 1000:.0f}K")
                                st.metric("Standart Sapma", f"{std_val / 1000:.0f}K")
                            else:
                                st.metric("Dövr Dəyişikliyi", f"{int(change)}")
                                st.metric("Standart Sapma", f"{int(std_val)}")
                        elif "%" in selected_category:
                            st.metric("Dövr Dəyişikliyi", f"{int(change)}%")
                            st.metric("Standart Sapma", f"{int(std_val)}%")
                        elif "min ton" in selected_category:
                            st.metric("Dövr Dəyişikliyi", f"{change:.1f}")
                            st.metric("Standart Sapma", f"{std_val:.1f}")
                        elif "kq" in selected_category:
                            st.metric("Dövr Dəyişikliyi", f"{int(change)} kq")
                            st.metric("Standart Sapma", f"{int(std_val)} kq")
                        else:
                            st.metric("Dövr Dəyişikliyi", f"{change:.1f}")
                            st.metric("Standart Sapma", f"{std_val:.1f}")

                    with col4_stats:
                        from scipy import stats
                        slope, _, r_value, _, _ = stats.linregress(np.arange(len(series)), series.values)
                        trend_direction = "↗ Artan" if slope > 0 else "↘ Azalan"
                        st.metric("Trend", trend_direction)
                        st.metric("R² Dəyəri", f"{r_value ** 2:.3f}")

                    # Data table with enhanced formatting for all models
                    if st.checkbox("Tarixi məlumatları göstər"):
                        st.markdown("### 📋 Seçilmiş Dövrün Məlumatları")
                        display_data = series.reset_index()
                        display_data.columns = ['İl', 'Dəyər']
                        display_data['İl'] = display_data['İl'].dt.year

                        # Format the value column based on category type - UPDATED WITH ALL FORMATTING
                        if "Price" in selected_category:
                            display_data['Dəyər'] = display_data['Dəyər'].apply(lambda x: f"${int(x)}")
                        elif "Export" in selected_category:
                            display_data['Dəyər'] = display_data['Dəyər'].apply(lambda x: f"{int(x / 1000)}K barel/gün")
                        elif "daxili məhsul" in selected_category or "məhsulu" in selected_category or "dövriyyəsi" in selected_category or "vəsaitlər" in selected_category:
                            display_data['Dəyər'] = display_data['Dəyər'].apply(lambda x:
                                                                                f"{x / 1000000:.1f}M" if x >= 1000000 else
                                                                                f"{x / 1000:.0f}K" if x >= 1000 else f"{int(x)}")
                        elif "xidmətlər" in selected_category:
                            display_data['Dəyər'] = display_data['Dəyər'].apply(lambda x:
                                                                                f"{x / 1000:.0f}K" if x >= 1000 else f"{int(x)}")
                        elif "daşınması" in selected_category:
                            display_data['Dəyər'] = display_data['Dəyər'].apply(lambda x:
                                                                                f"{x / 1000:.1f}K ton" if x >= 1000 else f"{int(x)} ton")
                        elif "əmək haqqı" in selected_category:
                            display_data['Dəyər'] = display_data['Dəyər'].apply(lambda x: f"{int(x)} manat")
                        elif "%" in selected_category:
                            display_data['Dəyər'] = display_data['Dəyər'].apply(lambda x: f"{int(x)}%")
                        elif "min ton" in selected_category:
                            display_data['Dəyər'] = display_data['Dəyər'].apply(lambda x: f"{x:.1f} min ton")
                        elif "kq" in selected_category:
                            display_data['Dəyər'] = display_data['Dəyər'].apply(lambda x: f"{int(x)} kq")
                        else:
                            display_data['Dəyər'] = display_data['Dəyər'].apply(lambda x: f"{x:.1f}")

                        st.dataframe(display_data, use_container_width=True)

                        # Add model-specific summary information
                        if selected_model == "Ümumi İqtisadi Göstəricilər":  # NEW SUMMARY
                            st.markdown("#### 📈 Əsas Məlumatlar")
                            if "daxili məhsul" in selected_category:
                                st.write("• **1995-2021**: Daimi artım trendi")
                                st.write("• **2020-2023**: Pandemiya və neft qiymətlərinin təsiri")
                                st.write("• **Strateji Əhəmiyyət**: Ölkənin ümumi iqtisadi göstəricisi")
                            elif "məhsulu" in selected_category:
                                st.write("• **Sənaye**: Neft və qeyri-neft sahələrinin birgə inkişafı")
                                st.write("• **Kənd Təsərrüfatı**: Davamlı artım tendensiyası")
                            elif "dövriyyəsi" in selected_category:
                                st.write("• **Pərakəndə Satış**: İstehlak artımının göstəricisi")
                                st.write("• **İqtisadi Canlanma**: Daxili tələbatın artması")
                            elif "əmək haqqı" in selected_category:
                                st.write("• **Sosial Rifah**: Əhalinin gəlir səviyyəsi")
                                st.write("• **İnflyasiya Təsiri**: Nominal artımın real dəyəri")
                        elif selected_model == "Mineral Gübrələrin Verilməsi":
                            st.markdown("#### 📈 Əsas Məlumatlar")
                            if "Cəmi" in selected_category:
                                st.write("• **1990-ci il**: Ən yüksək gübrə istifadəsi dövrü")
                                st.write("• **1990-2000**: Kəskin azalma dövrü")
                                st.write("• **2007-ci ildən**: Tədrici bərpa və artım")
                            elif "%" in selected_category:
                                st.write("• Gübrələnmiş sahələrin payının dinamikası")
                                st.write("• Kənd təsərrüfatının intensivləşməsi göstəricisi")
                        elif selected_model == "Neft və Enerji Göstəriciləri":
                            st.markdown("#### 📈 Əsas Məlumatlar")
                            if "Price" in selected_category:
                                st.write("• **Fiskal Tarazlıq**: Büdcə balansı üçün tələb olunan neft qiyməti")
                                st.write("• **Qiymət Dinamikası**: Dünya bazarı və daxili fiskal tələbat")
                                st.write("• **2010-2015**: Yüksək fiskal tələbat dövrü (70-90$)")
                                st.write("• **2020+**: Daha sabit fiskal göstəricilər")
                            elif "Export" in selected_category:
                                st.write("• **İxrac Həcmi**: Gündəlik xam neft ixracı (barel/gün)")
                                st.write("• **2005-2010**: İxrac artımı dövrü")
                                st.write("• **2010+**: Tədricən azalma tendensiyası")
                                st.write("• **Strateji Əhəmiyyət**: ÜDM-də yüksək pay")
        else:
            st.error("Məlumatlar yüklənə bilmədi")

    with col2:
        st.header("ℹ️ Məlumat")

        # Sector information - UPDATED WITH COMPREHENSIVE MACRO INFO
        sector_descriptions = {
            "Makro-Mikro İqtisadi Göstəricilər": "Makro və mikro iqtisadi göstəricilərin, o cümlədən neft və enerji sektoru, ümumi iqtisadi göstəricilər və sektorlar üzrə analiz və proqnozlaşdırılması.",
            "Kənd Təsərrüfatı": "Kənd təsərrüfatı sahəsində əhali, ərazi və mineral gübrə göstəricilərinin analizi.",
            "Səhiyyə": "Səhiyyə sektoruna aid müxtəlif göstəricilərin statistik təhlilini təqdim edir."
        }

        st.info(sector_descriptions.get(selected_sector, "Sektor məlumatı"))

        # Model information - UPDATED WITH ALL MODELS
        st.markdown("### 🔬 Seçilmiş Parametrlər")
        st.markdown(f"**Sektor:** {selected_sector}")
        st.markdown(f"**Model:** {selected_model}")
        st.markdown(f"**Kateqoriya:** {selected_category}")
        st.markdown(f"**Zaman dövrü:** {start_year} - {end_year}")

        # Model-specific information
        if selected_model == "Mineral Gübrələrin Verilməsi":
            st.markdown("**Məlumat dövrü:** 1990-2018")
            st.markdown("**Vahid:** Müxtəlif ölçü vahidləri")
        elif selected_model == "Neft və Enerji Göstəriciləri":
            st.markdown("**Məlumat dövrü:** 2000-2025")
            st.markdown("**Vahid:** ABŞ dolları və barel/gün")
        elif selected_model == "Ümumi İqtisadi Göstəricilər":  # NEW MODEL INFO
            st.markdown("**Məlumat dövrü:** 1995-2024")
            st.markdown("**Vahid:** Manat, min ton, faiz")

        if end_year == forecast_max_year:
            st.markdown(f"**Proqnoz müddəti:** {forecast_years} il")
        else:
            if selected_model == "Mineral Gübrələrin Verilməsi":
                st.markdown("**Proqnoz:** Deaktiv (2018 seçin)")
            elif selected_model == "Neft və Enerji Göstəriciləri":
                st.markdown("**Proqnoz:** Deaktiv (2025 seçin)")
            elif selected_model == "Ümumi İqtisadi Göstəricilər":
                st.markdown("**Proqnoz:** Deaktiv (2024 seçin)")
            else:
                st.markdown("**Proqnoz:** Deaktiv (2023 seçin)")

        # Available categories
        st.markdown("### 📋 Mövcud Kateqoriyalar")
        for i, cat in enumerate(model_config["categories"], 1):
            if cat == selected_category:
                st.markdown(f"**{i}. {cat[:30]}...** ✅" if len(cat) > 30 else f"**{i}. {cat}** ✅")
            else:
                st.markdown(f"{i}. {cat[:30]}..." if len(cat) > 30 else f"{i}. {cat}")

        # Method information - UPDATED WITH ALL METHODS
        if end_year == forecast_max_year:
            st.markdown("### 🧮 Proqnoz Metodları")
            if selected_model == "Ümumi İqtisadi Göstəricilər":  # NEW METHOD INFO
                st.markdown("""
                - **ARIMA**: Avtoreqressiv inteqrallı model
                - **Prophet**: Facebook tərəfindən hazırlanmış
                - **Random Forest**: Maşın öyrənməsi metodu

                **Xüsusi qeyd:** Ümumi iqtisadi məlumatlar üçün optimizasiya edilib.
                """)
            elif selected_model == "Neft və Enerji Göstəriciləri":
                st.markdown("""
                - **ARIMA**: Avtoreqressiv inteqrallı model
                - **Prophet**: Facebook tərəfindən hazırlanmış
                - **Random Forest**: Maşın öyrənməsi metodu

                **Xüsusi qeyd:** Neft və enerji məlumatları üçün optimizasiya edilib.
                """)
            elif selected_model == "Mineral Gübrələrin Verilməsi":
                st.markdown("""
                - **ARIMA**: Avtoreqressiv inteqrallı model
                - **Prophet**: Facebook tərəfindən hazırlanmış
                - **Random Forest**: Maşın öyrənməsi
                - **Linear Trend**: Xətti trend analizi

                **Xüsusi qeyd:** Mineral gübrə məlumatları üçün optimizasiya edilib.
                """)
            else:
                st.markdown("""
                - **ARIMA**: Avtoreqressiv model
                - **Prophet**: Facebook tərəfindən hazırlanmış
                - **Random Forest**: Maşın öyrənməsi
                - **Linear Trend**: Xətti trend
                """)

        # Usage guide - UPDATED WITH ALL GUIDANCE
        st.markdown("### 📖 İstifadə Qaydaları")
        if selected_model == "Ümumi İqtisadi Göstəricilər":  # NEW USAGE GUIDE
            st.markdown("""
            1. **Sektor**: Makro-Mikro İqtisadi Göstəricilər seçin
            2. **Model**: Ümumi İqtisadi Göstəricilər seçin
            3. **Zaman dövrü**: 1995-2024 arası təyin edin
            4. **Kateqoriya**: Analiz ediləcək göstərici seçin
            5. Proqnoz üçün son il **2024** seçin
            6. **PDF Hesabat** yükləyin
            7. Nəticələri təhlil edin

            **Qeyd:** Bu model 1995-2024 dövrünü əhatə edir.
            """)
        elif selected_model == "Neft və Enerji Göstəriciləri":
            st.markdown("""
            1. **Sektor**: Makro-Mikro İqtisadi Göstəricilər seçin
            2. **Model**: Neft və Enerji Göstəriciləri seçin
            3. **Zaman dövrü**: 2000-2025 arası təyin edin
            4. **Kateqoriya**: Analiz ediləcək göstərici seçin
            5. Proqnoz üçün son il **2025** seçin
            6. **PDF Hesabat** yükləyin
            7. Nəticələri təhlil edin

            **Qeyd:** Bu model 2000-2025 dövrünü əhatə edir.
            """)
        elif selected_model == "Mineral Gübrələrin Verilməsi":
            st.markdown("""
            1. **Sektor**: Kənd Təsərrüfatı seçin
            2. **Model**: Mineral Gübrələrin Verilməsi seçin
            3. **Zaman dövrü**: 1990-2018 arası təyin edin
            4. **Kateqoriya**: Analiz ediləcək göstərici seçin
            5. Proqnoz üçün son il **2018** seçin
            6. **PDF Hesabat** yükləyin
            7. Nəticələri təhlil edin

            **Qeyd:** Bu model yalnız 1990-2018 dövrünü əhatə edir.
            """)
        else:
            st.markdown("""
            1. **Sektor** seçin
            2. **Model** seçin
            3. **Zaman dövrü** təyin edin
            4. **Kateqoriya** seçin
            5. Proqnoz üçün son il **2023** seçin
            6. **PDF Hesabat** yükləyin
            7. Nəticələri təhlil edin
            """)

        # Additional info for specific models
        if selected_model == "Ümumi İqtisadi Göstəricilər":  # NEW INFO SECTION
            st.markdown("### 📊 Ümumi İqtisadi Göstəricilər Haqqında")
            st.markdown("""
            **Əsas Kateqoriyalar:**
            - Ümumi daxili məhsul (ÜDM)
            - Sənaye və kənd təsərrüfatı məhsulu
            - İnvestisiya və kapital xərcləri
            - Pərakəndə əmtəə dövriyyəsi
            - Xidmətlər sektoru
            - Nəqliyyat və əmək haqqı

            **Tarixi Mərhələlər:**
            - 1995-2005: İlkin iqtisadi inkişaf
            - 2005-2015: Sürətli artım dövrü
            - 2015-2020: Sabitləşmə mərhələsi
            - 2020+: Post-pandemiya bərpası

            **Makroiqtisadi Əhəmiyyət:**
            - Ölkənin ümumi iqtisadi vəziyyəti
            - Diversifikasiya səylərinin nəticələri
            - Qeyri-neft sektorunun inkişafı
            """)
        elif selected_model == "Neft və Enerji Göstəriciləri":
            st.markdown("### 🛢️ Neft və Enerji Haqqında")
            st.markdown("""
            **Əsas Kateqoriyalar:**
            - Fiskal tarazlıq neft qiyməti ($)
            - Xam neft ixracı (barel/gün)

            **Tarixi Mərhələlər:**
            - 2000-2005: İxrac artımı başlanğıcı
            - 2005-2010: Sürətli iqtisadi inkişaf
            - 2010-2015: Yüksək qiymət dövrü
            - 2015+: Diversifikasiya çalışmaları

            **Makroiqtisadi Əhəmiyyət:**
            - Dövlət büdcəsinin əsas mənbəyi
            - İxrac gəlirlərinin əsas hissəsi
            - Valyuta ehtiyatlarının formalaşması
            """)
        elif selected_model == "Mineral Gübrələrin Verilməsi":
            st.markdown("### 🌱 Mineral Gübrə Haqqında")
            st.markdown("""
            **Əsas Kateqoriyalar:**
            - Ümumi istifadə (min ton)
            - Gübrələnmiş sahələrin payı (%)
            - Hektara düşən məlumat (kq)
            - Məhsul növləri üzrə bölgü

            **Tarixi Mərhələlər:**
            - 1990: Ən yüksək göstəricilər
            - 1990-2000: Azalma dövrü
            - 2007+: Bərpa mərhələsi
            """)


if __name__ == "__main__":
    main()