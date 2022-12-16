# Personal Project: Predicting BitCoin Volume
# Project Description
A trader wants insight on drivers of volume to reach actionable solutions.

# Project Goal
- Investigate drivers of volume in BitCoin Market Data from January 2018 to December 2022.

- Construct a ML time-series model that accurately predicts volume of Bitcoin.

# Initial Thoughts
My initial hypothesis is that drivers of volume will likely be the price of BitCoin and the time of the year or month.

# Plan

- Acquire data from messari.io api

- Prepare data by dropping unnecessary columns, filling nulls, renaming columns, and optimizing data types if necessary.

- Explore data in search of drivers of volume and answer the following:

> Does volume have seasonality over time?


> Do opening price and volume have a relationship?


> Do closing price and volume have a relationship?


> Do highest price and volume have a relationship?


> Do lowest price and volume have a relationship?

- Develop a model to predict volume

> Use drivers identified in explore to build predictive models

> Evaluate models on train and validate data

> Select best model based on lowest RMSE

> Evaluation of best model on test data

- Draw conlcusions

# Data Dictionary

| Feature | Definition |
| :- | :- |
| btc_open | Decimal value, opening price of BitCoin for the day. |
| btc_close | Decimal value, closing price of BitCoin for the day. |
| btc_high | Decimal value, highest price of BitCoin for the day. |
| btc_low | Decimal value, lowest price of BitCoin for the day. |
| btc_volume | Decimal value, number of shares traded in BitCoin for the day. |

# Steps to Reproduce
1. Clone this repo
2. Acquire the data from SQL database
3. Place data in file containing the cloned repo
4. Run notebook

# Takewaways and Conclusions

- Price seems to not share a relationship with volume.


- Volume seems to not have seasonality.


- Volume lacks seasonality and the data from 2022 is significantly different than 2018-2021.

# Recommendations

- Approach the project differently with more sophisticated data, exploration, and models to use.


- Acquire data from another api that gives more frequent intervals of data in a day.


- Explore the effects of real-world events that have affected BitCoin in the past.


- Learn more sophisticated modeling with regards to stock market data and cryptocurrencies.
