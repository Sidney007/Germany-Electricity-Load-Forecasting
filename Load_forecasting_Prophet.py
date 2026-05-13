# -*- coding: utf-8 -*-
"""
Goal: Time series forecasting using a Prophet model.
The data is imported using Energy Charts API (Fraunhofer ISE) for 2024 (training)
and 2025 (test).

After the model fitting metrics such as MAE, RMSE and MAPE are calculated to test the accuracy
of forecasting.

Plots used for visualizing the following: raw load pattern, seasonal time series decomposition,
components of the forecast and comparison of the forecast and test data.

"""


import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time


warnings.filterwarnings("ignore")

MAX_HOURLY_ROWS = 8900

def fetch_load(year: int) -> pd.DataFrame:
    """
    Fetch hourly electricity load (MW) for Germany.
    Uses the /total_power endpoint which includes a
    'Load (incl. self-consumption)' series.

    Returns a DataFrame with columns:
        timestamp  (datetime64, UTC)
        load_mw    (float, positive values only)
    """
    url    = "https://api.energy-charts.info/total_power"
    params = {
        "country": "de",
        "start":   f"{year}-01-01",
        "end":     f"{year + 1}-01-01", ##due to API behavior since API does 
        ##not involve last day of the year otherwise
    }
    r = requests.get(url, params=params, timeout=60) 
    r.raise_for_status()
    data = r.json()

    if "unix_seconds" not in data or "production_types" not in data:
        raise ValueError(
            f"Unexpected load API response for {year}. "
            f"Keys: {list(data.keys())}"
        )

    timestamps       = pd.to_datetime(data["unix_seconds"], unit="s", utc=True)
    production_types = data.get("production_types", []) ## retrieve all possible load/power types from API response

    # API returns 'Load (incl. self-consumption)' — match by startswith
    ### look for dataset that starts with load
    load_series = next(
        (t for t in production_types
         if t.get("name", "").lower().startswith("load")),
        None,
    )

    if load_series is None: ## if no load series is found 
        available = [t.get("name") for t in production_types] ## collect available dataset names
        raise ValueError(
            f"'Load' series not found for year {year}. "
            f"Available: {available}"
        ) ##detailed error to show which datasets actually returned

    df = pd.DataFrame({"timestamp": timestamps, "load_mw": load_series["data"]})

    # Remove non-positive load values before resampling
    n_before = len(df)
    df = df[df["load_mw"] > 0].copy() ## keep only positive load values
    if len(df) < n_before:
        print(f"  WARNING {year}: dropped {n_before - len(df)} non-positive load rows.")
        ## show how many dropped

    # Resample to hourly whenever the API returns sub-hourly data.
    # Using mean() across the four 15-min readings per hour.
    # _to_hourly also drops any hours that became NaN after filtering above.
    if len(df) > MAX_HOURLY_ROWS:
        df = _to_hourly(df, "load_mw", agg="mean")
        print(f"  {year}: resampled load to hourly    → {len(df)} rows")
    else:
        print(f"  {year}: {len(df)} load rows fetched")



    return df

   

def _to_hourly(df: pd.DataFrame, value_col: str, agg: str) -> pd.DataFrame:
 ## Converting timestamped data from 15-min to hourly format and aggregate values
 ## load value aggregation done using mean value

    
    resampled = (
        df.set_index("timestamp") ## timestamp column as index to perform time-based resampling
        .resample("h")[value_col]
        .agg(agg)
        .reset_index() ## converting timestamp index back into column
        .dropna(subset=[value_col])   # drop hours where all slots were NaN/filtered
    )
    return resampled

def main ():
    
    #training subset
    df_train= fetch_load(2024).dropna(subset=["load_mw"])
    df_train = df_train[df_train['timestamp'].dt.year == 2024] ##due to extra hours of 1 day of the following year
    print(f"  {2024}:  Finally resampled load to hourly    → {len(df_train)} rows")
    
    time.sleep(15) ## to ensure no rate limiting
    
    #test subset
    df_test = fetch_load(2025).dropna(subset=["load_mw"])
    df_test = df_test[df_test['timestamp'].dt.year == 2025] ##due to extra hours of 1 day of the following year
    print(f"  {2025}:  Finally resampled load to hourly    → {len(df_test)} rows")
    print(df_test.head(6))
    
    ###timezone removed since decompose did not work with it
    df_train['timestamp'] = df_train['timestamp'].dt.tz_localize(None)
    df_test['timestamp'] = df_test['timestamp'].dt.tz_localize(None)
    print(df_train.dtypes)
    print(df_test.dtypes)
    
    #weekly rhythm training data
    sample = df_train.head(24 * 7 * 4)  # first 4 weeks
    plt.figure(figsize=(14, 4))
    plt.plot(sample['timestamp'], sample['load_mw'], linewidth=0.8)
    plt.title('Germany hourly load: first 4 weeks of 2024')
    plt.ylabel('MW')
    plt.tight_layout()
    plt.show()
    
    # Investigating the weekly pattern with period = 168 (24x7)
    decompose = seasonal_decompose( 
        df_train.load_mw, model='additive',
        extrapolate_trend='freq',
        period=168)     
    decompose.plot()
    plt.title('Decompose into trend, seasonality, residual')
    plt.show()
    

    
    #  Model fitting with Prophet
    df_train_prophet = df_train.copy()

    # date variable needs to be named "ds" for prophet
    df_train_prophet = df_train_prophet.rename(columns={"timestamp": "ds"})

    # target variable needs to be named "y" for prophet
    df_train_prophet = df_train_prophet.rename(columns={"load_mw": "y"})
    
    
    # Modell fitting for prediction
    
    model_prophet  = Prophet()
    model_prophet.fit(df_train_prophet)
    
    periods_test=len(df_test)
    ##creating the future dataframe with same data as the train+test
    df_future = model_prophet.make_future_dataframe(periods=periods_test, freq= 'h')
    print(df_future.head(6))
    forecast = model_prophet.predict(df_future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round().tail())
    
    # components of forecast
    model_prophet.plot_components(forecast)
    plt.title('Components of forecast')
    plt.show()
    
    # align forecast with test period only
    forecast_test = forecast.tail(periods_test) # since forecast has length of train+test
    # calculate certain metrics
    mae  = mean_absolute_error(df_test['load_mw'].values, forecast_test['yhat'].values) # mean absolute error
    rmse = np.sqrt(mean_squared_error(df_test['load_mw'].values, forecast_test['yhat'].values)) # root mean squared error
    mape = np.mean(np.abs((df_test['load_mw'].values - forecast_test['yhat'].values) 
                       / df_test['load_mw'].values)) * 100 # mean absolute percentage error


    print(f"MAE:  {mae:.0f} MW")
    print(f"RMSE: {rmse:.0f} MW")
    print(f"MAPE: {mape:.1f}%")
    
    ## Plotting
    forecast_plot = model_prophet.plot(forecast)
    
    # adding vertical line at end of training period
    axes = forecast_plot.gca()
    last_training_date = forecast['ds'].iloc[-(periods_test)-1]
    axes.axvline(x=last_training_date, color='red', linestyle='--', label='Training End')

    # plot true test data for the period after the red line
    df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
    plt.plot(df_test['timestamp'], df_test['load_mw'],'ro', markersize=3, label='True Test Data')
    
    plt.legend()
    plt.title('Forecast vs actual')
    plt.show()
    
if __name__ == "__main__":
    main()
