#!/usr/bin/env python
# coding: utf-8

# In[25]:


import requests
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit 


# In[26]:


def get_crypto_price(symbol, start, end):
    
    api_url = f'https://data.messari.io/api/v1/markets/binance-{symbol}-usdt/metrics/price/time-series?start={start}&end={end}&interval=1d'
    raw = requests.get(api_url).json()
    df = pd.DataFrame(raw['data']['values'])
    df = df.rename(columns = {0:'date',1:f'{symbol}_open',2:f'{symbol}_high',3:f'{symbol}_low',4:f'{symbol}_close',5:f'{symbol}_volume'})
    df['date'] = pd.to_datetime(df['date'], unit = 'ms')
    df = df.set_index('date')
    
    return df


# In[27]:


def clean_btc_data_2022(btc):
    
    df = btc['2022']
    df = df.reset_index()
    df = df.append(pd.DataFrame({'date': pd.date_range(start=df.date.iloc[-1], periods=20, freq='D', closed='right')}))
    df = df.set_index('date')
    df = df.groupby(df.index.day).ffill()
    btc = pd.concat([btc[:'2021'], df], ignore_index=False)
    
    return btc


# In[28]:


def clean_btc_data_2021(btc):
    
    df = btc['2021'].resample('D').mean()
    df = df.groupby(df.index.day).bfill()
    df2 = pd.concat([btc[:'2020'], df], ignore_index=False)
    btc = pd.concat([df2, btc['2022']], ignore_index=False)
    
    return btc


# In[29]:


def remove_leap_day(btc):
    
    btc = btc[btc.index != '2020-02-29']
    
    return btc


# In[30]:


def clean_data(btc):
    
    btc = remove_leap_day(btc)
    btc = clean_btc_data_2021(btc)
    btc = clean_btc_data_2022(btc)
    
    return btc

