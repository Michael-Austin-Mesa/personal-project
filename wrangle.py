#!/usr/bin/env python
# coding: utf-8

# In[23]:


import requests
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit 


# In[24]:


def get_crypto_price(symbol, start, end):
    api_url = f'https://data.messari.io/api/v1/markets/binance-{symbol}-usdt/metrics/price/time-series?start={start}&end={end}&interval=1d'
    raw = requests.get(api_url).json()
    df = pd.DataFrame(raw['data']['values'])
    df = df.rename(columns = {0:'date',1:f'{symbol}_open',2:f'{symbol}_high',3:f'{symbol}_low',4:f'{symbol}_close',5:f'{symbol}_volume'})
    df['date'] = pd.to_datetime(df['date'], unit = 'ms')
    df = df.set_index('date')
    return df

