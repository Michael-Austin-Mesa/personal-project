#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")


import requests

import pandas as pd
import numpy as np
from scipy import stats
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.tsa.api import Holt

from datetime import datetime

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit 


import wrangle as w


# In[2]:


btc = w.get_crypto_price('btc', '2018-01-01', '2022-12-12')


# In[3]:


btc = w.clean_data(btc)


# In[4]:


train = btc[:'2020']
validate = btc['2021']
test = btc['2022']


# In[5]:


y = train.btc_volume


# In[6]:


def get_avg_vol_monthly():
    ''' This function returns a plot of the
        average monthly volume of BitCoin 
        in the train data
    '''
    sns.set(rc={'figure.figsize': (10, 6)})
    sns.set_theme(style="whitegrid")
    ax = y.groupby([y.index.year, y.index.month]).mean().unstack(0).plot()
    ax.set(title="Average Volume by Month", xlabel='Month', ylabel='Volume')
    ax.legend(title='Year')
    plt.show()


# In[7]:


def get_vol_by_date():
    ''' This function returns a plot of BitCoin
        volume by date day
    '''
    sns.set(rc={'figure.figsize': (10, 6)})
    sns.set_theme(style="whitegrid")
    y.groupby([y.index.year]).plot(title= 'Volume by Date', xlabel='Date (Day)', ylabel = 'Volume', legend=True)
    plt.show()


# In[8]:


def plot_price_vol():
    ''' This function returns scatter plots of
        the price features against volume
    '''
    sns.set(rc={'figure.figsize': (10, 6)})
    sns.set_theme(style="whitegrid")
    plot_df = train[['btc_open', 'btc_close','btc_high','btc_low']]
    for col in plot_df.columns:
        sns.scatterplot(x=plot_df[col], y=y)
        plt.show()

