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
@st.cache_data
def load_data(file_path, data_type, date_column, clear_cache=False):
    """Load and prepare data based on type"""
    try:
        if data_type == "excel":
            if "Kend_Teserrufati" in file_path:
                return load_gdp_data()
            elif "Ehali_ve_Erazi" in file_path:
                return load_population_territory_data()
        else:
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


def create_forecast_plot(sector, model_name, category, data, start_year, end_year, forecast_periods=5):
    """Create forecast visualization with period selection"""

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
    """Create forecast visualization with period selection and full debugging"""
    
    # Debug the data flow first
    series = debug_data_flow(sector, model_name, category, data, start_year, end_year)
    
    if series is None or len(series) < 2:
        st.error("Debugging shows insufficient data!")
        return None
    
    print(f"About to plot series with values: {series.values}")
    
    # Special handling for effective start year
    if model_name == "Əhali və Ərazi Göstəriciləri":
        effective_start_year = max(start_year, 1990)
        if start_year < 1990:
            st.warning(f"Əhali və Ərazi məlumatları 1990-ci ildən başlayır. Analiz {effective_start_year}-ci ildən başlayacaq.")
    else:
        effective_start_year = start_year
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot historical data
    ax.plot(series.index, series.values, 'o-', label='Tarixi məlumatlar',
            color='blue', linewidth=2, markersize=6)
    
    print(f"Plotted historical data - X: {series.index.tolist()}, Y: {series.values.tolist()}")
    
    # Special handling for Ehali ve Erazi data
    if model_name == "Əhali və Ərazi Göstəriciləri":
        # Since data now starts from 1990, emphasize 5-year interval points from 1990 onwards
        years_5yr = [year for year in range(1990, end_year + 1, 5)
                     if year >= effective_start_year and year <= end_year]

        # Find these points in the data
        for year in years_5yr:
            idx = series.index.year == year
            if any(idx):
                point_idx = series.index[idx][0]
                point_value = series[point_idx]
                ax.plot(point_idx, point_value, 'o', color='red',
                        markersize=8, markeredgecolor='black')

                # Add label - handle small decimal values properly
                if abs(point_value) < 1:
                    label_text = f'{point_value:.3f}'
                else:
                    label_text = f'{int(point_value)}'
                    
                ax.annotate(label_text,
                            xy=(point_idx, point_value),
                            xytext=(0, 10),
                            textcoords='offset points',
                            fontsize=9,
                            ha='center',
                            va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='yellow', alpha=0.7))

    # Add regular value labels (not for all points to avoid clutter)
    label_step = max(1, len(series) // 8)  # Adjust based on data density

    # Add first and last point labels always
    if len(series) > 0:
        first_val = series.iloc[0]
        last_val = series.iloc[-1]
        
        # Handle small decimal values
        first_label = f'{first_val:.3f}' if abs(first_val) < 1 else f'{int(first_val)}'
        last_label = f'{last_val:.3f}' if abs(last_val) < 1 else f'{int(last_val)}'
        
        ax.annotate(first_label,
                    xy=(series.index[0], first_val),
                    xytext=(0, 10),
                    textcoords='offset points',
                    fontsize=8,
                    ha='center',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

        ax.annotate(last_label,
                    xy=(series.index[-1], last_val),
                    xytext=(0, 10),
                    textcoords='offset points',
                    fontsize=8,
                    ha='center',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

        # Add some intermediate labels
        for i in range(label_step, len(series) - 1, label_step):
            if series.index[i].year % 5 == 0:  # Prefer years divisible by 5
                intermediate_val = series.iloc[i]
                intermediate_label = f'{intermediate_val:.3f}' if abs(intermediate_val) < 1 else f'{int(intermediate_val)}'
                
                ax.annotate(intermediate_label,
                            xy=(series.index[i], intermediate_val),
                            xytext=(0, 10),
                            textcoords='offset points',
                            fontsize=8,
                            ha='center',
                            va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    # Only show forecasts if end year is 2023
    if end_year == 2023:
        # Future years
        future_years = pd.date_range(start=f'{series.index[-1].year + 1}', periods=forecast_periods, freq='Y')

        # Method configurations based on sector and model
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
        elif sector == "Kənd Təsərrüfatı" and model_name == "Əhali və Ərazi Göstəriciləri":
            methods_config = {
                'Əhali (min nəfər)': ['arima', 'linear'],
                'Ərazi (min km²)': ['linear'],
                'Kənd təsərrüfatına yararlı torpaqlar (min ha)': ['arima', 'linear'],
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
        else:
            # Default healthcare configurations
            methods_config = {
                "Həkimlərin sayı - cəmi": ['arima', 'linear'],
                "terapevtlər": ['arima', 'rf', 'prophet'],
                "cərrahlar": ['arima', 'rf', 'linear'],
                "pediatrlar": ['arima', 'linear'],
                "stomatoloq və diş həkimləri": ['arima', 'prophet']
            }

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

    if end_year == 2023:
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
    if end_year == 2023:
        title += ' və Proqnoz'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set appropriate Y-axis label based on sector
    if sector == "Makro-Mikro İqtisadi Göstəricilər":
        ax.set_ylabel('Milyon Manat', fontsize=12)
    elif sector == "Kənd Təsərrüfatı":
        ax.set_ylabel('Dəyər', fontsize=12)
    else:
        ax.set_ylabel('Dəyər', fontsize=12)

    ax.set_xlabel('İl', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set x-ticks to show years at appropriate intervals
    # For Ehali ve Erazi model, emphasize 5-year intervals starting from 1990
    if model_name == "Əhali və Ərazi Göstəriciləri":
        # Create a list of years to show as ticks
        years_to_show = []

        # For 1990 onwards, show every 5 years
        milestone_years = [year for year in range(1990, end_year + 1, 5)
                          if year >= effective_start_year and year <= end_year]
        years_to_show.extend(milestone_years)

        # Add some intermediate years for better visualization
        if end_year > effective_start_year:
            step = max(1, (end_year - effective_start_year) // 6)
            intermediate_years = [year for year in range(effective_start_year, end_year + 1, step)
                                 if year not in years_to_show]
            if end_year not in years_to_show:
                intermediate_years.append(end_year)
            years_to_show.extend(intermediate_years)

        # Sort the years
        years_to_show = sorted(list(set(years_to_show)))

        # Convert years to datetime for setting ticks
        x_ticks = [pd.Timestamp(f"{year}-01-01") for year in years_to_show]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(year) for year in years_to_show], rotation=45)

    # Add statistics box - FIXED VERSION
    last_value = series.values[-1]
    mean_value = series.mean()
    
    # Format values properly
    last_value_str = f"{last_value:.3f}" if abs(last_value) < 1 else f"{int(last_value)}"
    mean_value_str = f"{mean_value:.3f}" if abs(mean_value) < 1 else f"{int(mean_value)}"
    
    stats_text = f"Son dəyər: {last_value_str}\n"
    stats_text += f"Orta: {mean_value_str}\n"
    stats_text += f"Trend: {slope:.4f}/il\n"
    stats_text += f"Dövr: {effective_start_year}-{end_year}"
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    
    print(f"Plot created successfully with {len(series)} data points")
    return fig

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

    # Time period selection
    st.sidebar.header("📅 Zaman Dövrü Seçimi")
    min_year, max_year = model_config["date_range"]

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

    # PDF Report Generation Section
    st.sidebar.header("📄 Hesabat Generasiyası")

    # Define PDF mapping based on sector and model - Updated to match exact names
    pdf_mapping = {
        # Add debug to see actual values
        selected_sector: {
            selected_model: None  # Will be determined below
        }
    }

    # Determine PDF file based on selections
    pdf_file = None

    # Check based on sector and model combination
    if "Makro" in selected_sector or "İqtisadi" in selected_sector:
        if "İqtisadi" in selected_model or "UDM" in selected_model:
            pdf_file = "Birləşdirilmiş_Hesabat_Agriculture.pdf"
    elif "Kənd" in selected_sector or "Təsərrüfat" in selected_sector:
        if "Əhali" in selected_model or "Ərazi" in selected_model:
            pdf_file = "Birləşdirilmiş_Hesabat_Demoqrafiya.pdf"
    elif "Səhiyyə" in selected_sector:
        if "Həkim" in selected_model or "həkim" in selected_model:
            pdf_file = "Birləşdirilmiş_Hesabat_Hekimler.pdf"
        elif "İnfeksiya" in selected_model or "infeksiya" in selected_model or "parazit" in selected_model:
            pdf_file = "Səhiyyə_İnfeksiya_və_Parazit_Xəstəlikləri_Hesabat.pdf"
        elif "Vərəm" in selected_model or "vərəm" in selected_model:
            pdf_file = "Vərəm_Xəstəliyi_Hesabat.pdf"

    # Alternative approach - check by file patterns if above doesn't work
    if not pdf_file:
        # Check if any of the data files match common patterns
        if "Ehali_Ve_Erazi.csv" in model_config.get("file", ""):
            pdf_file = "Birləşdirilmiş_Hesabat_Demoqrafiya.pdf"
        elif "Kend_Teserrufati" in model_config.get("file", ""):
            pdf_file = "Birləşdirilmiş_Hesabat_Agriculture.pdf"
        elif "infeksion" in model_config.get("file", ""):
            pdf_file = "Səhiyyə_İnfeksiya_və_Parazit_Xəstəlikləri_Hesabat.pdf"
        elif "vərəm" in model_config.get("file", ""):
            pdf_file = "Vərəm_Xəstəliyi_Hesabat.pdf"

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
            st.sidebar.write("- Səhiyyə_İnfeksiya_və_Parazit_Xəstəlikləri_Hesabat.pdf")
            st.sidebar.write("- Vərəm_Xəstəliyi_Hesabat.pdf")

    # Forecast settings (only shown if end year is 2023)
    if end_year == 2023:
        st.sidebar.header("🔮 Proqnoz Parametrləri")
        forecast_years = st.sidebar.selectbox(
            "Proqnoz müddəti (il):",
            options=[1, 2, 3, 4, 5],
            index=2
        )

        st.sidebar.info("💡 Proqnozlar yalnız 2023-cü ilə qədər olan məlumatlar üçün göstərilir")
    else:
        forecast_years = 3
        st.sidebar.info("ℹ️ Proqnozları görmək üçün son il 2023 seçin")

    # Main content area
    col1, col2 = st.columns([3, 1])

    with col1:
        period_text = f"({start_year}-{end_year})"
        if end_year == 2023:
            period_text += " + Proqnoz"

        st.header(f"📈 {selected_sector} - {selected_model}")
        st.subheader(f"Analiz: {selected_category} {period_text}")

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
                filtered_data = data[data.index.year >= start_year]
                filtered_data = filtered_data[filtered_data.index.year <= end_year]
                st.write(f"Filtered data for {selected_category} ({start_year}-{end_year}):")
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

                # Display statistics for selected period
                if selected_category in data.columns:
                    # Filter data by selected period
                    filtered_data = data[data.index.year >= start_year]
                    filtered_data = filtered_data[filtered_data.index.year <= end_year]
                    series = filtered_data[selected_category].dropna()

                    st.markdown("### 📊 Seçilmiş Dövrün Statistikaları")

                    col1_stats, col2_stats, col3_stats, col4_stats = st.columns(4)

                    with col1_stats:
                        st.metric("Son Dəyər", f"{series.iloc[-1]:.1f}")
                        st.metric("Orta Dəyər", f"{series.mean():.1f}")

                    with col2_stats:
                        st.metric("Maksimum", f"{series.max():.1f}")
                        st.metric("Minimum", f"{series.min():.1f}")

                    with col3_stats:
                        change = series.iloc[-1] - series.iloc[0]
                        st.metric("Dövr Dəyişikliyi", f"{change:.1f}")
                        st.metric("Standart Sapma", f"{series.std():.1f}")

                    with col4_stats:
                        from scipy import stats
                        slope, _, r_value, _, _ = stats.linregress(np.arange(len(series)), series.values)
                        trend_direction = "↗ Artan" if slope > 0 else "↘ Azalan"
                        st.metric("Trend", trend_direction)
                        st.metric("R² Dəyəri", f"{r_value ** 2:.3f}")

                    # Data table
                    if st.checkbox("Tarixi məlumatları göstər"):
                        st.markdown("### 📋 Seçilmiş Dövrün Məlumatları")
                        display_data = series.reset_index()
                        display_data.columns = ['İl', 'Dəyər']
                        display_data['İl'] = display_data['İl'].dt.year
                        st.dataframe(display_data, use_container_width=True)
        else:
            st.error("Məlumatlar yüklənə bilmədi")

    with col2:
        st.header("ℹ️ Məlumat")

        # Sector information
        sector_descriptions = {
            "Makro-Mikro İqtisadi Göstəricilər": "Makro və mikro iqtisadi göstəricilərin analizi və proqnozlaşdırılması.",
            "Kənd Təsərrüfatı": "Kənd təsərrüfatı sahəsində əhali və ərazi göstəricilərinin analizi.",
            "Səhiyyə": "Səhiyyə sektoruna aid müxtəlif göstəricilərin statistik təhlilini təqdim edir."
        }

        st.info(sector_descriptions.get(selected_sector, "Sektor məlumatı"))

        # Model information
        st.markdown("### 🔬 Seçilmiş Parametrlər")
        st.markdown(f"**Sektor:** {selected_sector}")
        st.markdown(f"**Model:** {selected_model}")
        st.markdown(f"**Kateqoriya:** {selected_category}")
        st.markdown(f"**Zaman dövrü:** {start_year} - {end_year}")
        if end_year == 2023:
            st.markdown(f"**Proqnoz müddəti:** {forecast_years} il")
        else:
            st.markdown("**Proqnoz:** Deaktiv (2023 seçin)")

        # Available categories
        st.markdown("### 📋 Mövcud Kateqoriyalar")
        for i, cat in enumerate(model_config["categories"], 1):
            if cat == selected_category:
                st.markdown(f"**{i}. {cat[:30]}...** ✅" if len(cat) > 30 else f"**{i}. {cat}** ✅")
            else:
                st.markdown(f"{i}. {cat[:30]}..." if len(cat) > 30 else f"{i}. {cat}")

        # Method information
        if end_year == 2023:
            st.markdown("### 🧮 Proqnoz Metodları")
            st.markdown("""
            - **ARIMA**: Avtoreqressiv model
            - **Prophet**: Facebook tərəfindən hazırlanmış
            - **Random Forest**: Maşın öyrənməsi
            - **Linear Trend**: Xətti trend
            """)

        # Usage guide
        st.markdown("### 📖 İstifadə Qaydaları")
        st.markdown("""
        1. **Sektor** seçin
        2. **Model** seçin
        3. **Zaman dövrü** təyin edin
        4. **Kateqoriya** seçin
        5. Proqnoz üçün son il **2023** seçin
        6. **PDF Hesabat** yükləyin
        7. Nəticələri təhlil edin
        """)
if __name__ == "__main__":
    main()