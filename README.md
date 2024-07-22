# Electricity price prediction
Fiddling with time series data and models for predicting next day electricity price. I had been thinking about doing some time series modelleing related to the energy sector and recently read a Medium article on time series forecasting that inspired me to get started (see https://towardsdatascience.com/forecasting-in-the-age-of-foundation-models-8cd4eea0079d).

## About this repository

This repository is a toy project to collect and prepare some datasets that could be useful for predicting next day electricity price in Sweden. Since there is likely some prediction power in seasonality of the electricity price, we start with just price as a (univariate) time series and try to predict the next day price. We then expand the training data to also include next-day temperature forecast as a second variable and see if we can improve the model performance a bit. The idea being that temperature will likely affect energy consumption and therefore affect the electricity price.

### Collect datasets
#### Download historic electricity price data for one region of Sweden.

Hourly day ahead price is used as a proxy for the actual price. The day ahead market is describes as:
"In the day-ahead market customers can sell or buy energy for the next 24 hours in a closed auction. Orders are matched to maximize social welfare while taking network constraints provided by transmission system operators into consideration."
https://www.nordpoolgroup.com/en/market-data12/Dayahead/Area-Prices/SE/Hourly/

A couple of years hourly day ahead price (EUR) for Sweden in electricity price zone S04 (southern Sweden, where price volatility is high due to e.g. locality and energy infrastructure bottlenecks) was manually downloaded from Nordpool website https://www.nordpoolgroup.com/en/market-data12/Dayahead/Area-Prices/ALL1/Hourly/?dd=SE4&view=table

These files, containing xml tables, are processed (impute missing values etc) and merged into a single csv file before training.
See `src/data_preparation/process_raw_electricity_data.py` for the processing script, and `data/processed/processed_day_ahead_prices.csv` for the result csv file.

For those interested, the below image shows the location of the SE04 price zone (https://www.energyprices.eu/electricity/sweden-south)
![SE04](/data/images/se04-sweden-south-electricity-price-zone.png)

#### Download historic temperature data for that region.

Hourly mean temperature in the largest city in the region was downloaded from https://www.temperatur.nu/nobeltorget_malmo.

This csv file is processed (impute missing values etc) before training. See `src/data_preparation/process_raw_temperature_data.py` for the processing script, and `data/processed/temperature.csv` for the result csv file.

### Initial data sanity checks

A first look at the processed data with some basic data sanity checks can be found here `notebooks/data_checks.ipynb`

### Baseline: "Same price" predictor
We start with a simple baseline model which for a given day just predicts the same price for the next day.

### Single variable model: price time series
We train a couple of other (univariate) models and see if we can beat the baseline. Options here are classic ML models for forecasting (e.g. XGBoost) as well a deep learning ones (Lag-Llama).

### Going multivariate: adding temperature forecast
We go on to include the next day temperature forecast and see if we can get any better.