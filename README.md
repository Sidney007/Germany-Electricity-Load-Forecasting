# Germany-Electricity-Load-Forecasting

This repository presents a project for electricity load forecasting in Germany using a Meta Prophet model. Forecasting is an important activity in the power system and the electricity market for maintaining power system security.

# Data source
Energy-Charts API by Fraunhofer ISE (DE-LU bidding zone) https://api.energy-charts.info

# Flow of the project
1. Fetching Germany load data for 2024 and 2025 using API
2. Data cleaning and resampling
3. Exploring seasonality in the data
4. Training a forecasting model
5. Calculation of metrics to evaluate the accuracy of predictions (mean absolute error, root mean square error, mean absolute percentage error)

# Results
The results of the metrics are as follows:
- Mean absolute error (MAE): 10941 MW
- Root mean square error (RMSE): 12372 MW
- Mean absolute percentage error (MAPE): 20.8%

Germany's load demand is between 40000 and 80000 MW. The MAE of ~11000 MW and the MAPE of 20.8% is indicative of a Prophet model trained on one year of data with no additional regressors (such as temperature and public holidays). The RMSE of 12372 MW is higher than the MAE indicating that some of the spikes in the difference between forecasting and test data have a strong effect on the metric's value.

The uncertainty bands of the forecast get wider over time. This is expected of long range probablistic forecasting and the model is being honest rather than being overconfident.

# Possible improvements
- Using public holiday data using Prophets built in holiday support
- Adding temperature as a regressor
- Comparing against other time series forecasting models

# How to run
```
pip install prophet pandas matplotlib seaborn scikit-learn statsmodels requests
python Load_forecasting_Prophet.py
```

# Helpful links
Time series decomposition: https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/

Guide to time series with Prophet: https://medium.com/data-science/getting-started-predicting-time-series-data-with-facebook-prophet-c74ad3040525


# Disclaimer
The Python code in this repository was generated through iterative prompting and debugging with Claude (Anthropic). This includes the data pipeline, API integrations, metric calculations and data visualisation.

The conceptualization of the project idea was done with help of Claude (Anthropic).

The analytical framework, interpretation of results and conclusions are entirely my own. AI was used as an assistant and coding tool and not as a thinking tool.
