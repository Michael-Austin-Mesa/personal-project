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


# In[11]:


btc = w.get_crypto_price('btc', '2018-01-01', '2022-12-12')

btc = w.clean_data(btc)


# In[12]:


# human splitting by year
train = btc[:'2020']
validate = btc['2021']
test = btc['2022']

col = 'btc_volume'


# In[13]:


# Create the empty dataframe
eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'rmse'])

# Initialize volume, yhat_df, and period for modeling
volume = 0 #train['btc_volume'][-1:][0]

yhat_df = pd.DataFrame({'btc_volume': [volume]}, 
                       index = validate.index)

period = 0


# In[2]:


# evaluation function to compute rmse
def evaluate(target_var, validate, yhat_df):
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 0)
    return rmse


# In[3]:


# plot and evaluate
def plot_and_eval(target_var, train, validate, yhat_df):
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label = 'Train', linewidth = 1)
    plt.plot(validate[target_var], label = 'Validate', linewidth = 1)
    plt.plot(yhat_df[target_var])
    plt.title(target_var)
    rmse = evaluate(target_var, validate, yhat_df)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.show()


# In[4]:


# function to store rmse for comparison purposes
def append_eval_df(model_type, target_var, validate, yhat_df):
    rmse = evaluate(target_var, validate, yhat_df)
    d = {'model_type': [model_type], 'target_var': [target_var], 'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)


# In[5]:


def test_evaluate(target_var, test, yhat_df):
    rmse = round(sqrt(mean_squared_error(test[target_var], yhat_df[target_var])), 0)
    return rmse


# In[6]:


def plot_and_eval_test(target_var, train, validate, test, yhat_df):
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label = 'Train', linewidth = 1)
    plt.plot(validate[target_var], label = 'Validate', linewidth = 1)
    plt.plot(test[target_var], label = 'Test', linewidth = 1)
    plt.plot(yhat_df[target_var], alpha = .5, color="red")
    rmse = test_evaluate(target_var, test, yhat_df)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.title(target_var)
    plt.legend()
    plt.show()


# In[7]:


def get_btc_last_observed(train, validate, volume, yhat_df):
    
    volume = train['btc_volume'][-1:][0]

    yhat_df = pd.DataFrame({'btc_volume': [volume]}, 
                           index = validate.index)
    

    return volume, yhat_df


# In[8]:


def get_btc_simple_average(train, validate, volume, yhat_df):
    
    # getting the average of btc_volume in train
    volume = round(train['btc_volume'].mean(), 2)


    yhat_df = pd.DataFrame({'btc_volume': [volume]}, 
                           index = validate.index)

    
    return volume, yhat_df


# In[16]:


def get_btc_30d_average(train, validate, volume, yhat_df):
    
    period = 30
    volume = round(train['btc_volume'].rolling(period).mean().iloc[-1], 2)

    yhat_df = pd.DataFrame({'btc_volume': [volume]}, 
                           index = validate.index)

    
    return volume, yhat_df


# In[17]:


def get_btc_7d_average(train, validate, volume, yhat_df):
    
    period = 7
    volume = round(train['btc_volume'].rolling(period).mean().iloc[-1], 2)

    yhat_df = pd.DataFrame({'btc_volume': [volume]}, 
                           index = validate.index)

    
    return volume, yhat_df


# In[18]:


def get_btc_14d_average(train, validate, volume, yhat_df):
    
    period = 14
    volume = round(train['btc_volume'].rolling(period).mean().iloc[-1], 2)

    yhat_df = pd.DataFrame({'btc_volume': [volume]}, 
                           index = validate.index)

    
    return volume, yhat_df


# In[19]:


def get_btc_21d_average(train, validate, volume, yhat_df):
    
    period = 21
    volume = round(train['btc_volume'].rolling(period).mean().iloc[-1], 2)

    yhat_df = pd.DataFrame({'btc_volume': [volume]}, 
                           index = validate.index)

    
    return volume, yhat_df


# In[20]:


def get_btc_28d_average(train, validate, volume, yhat_df):
    
    period = 28
    volume = round(train['btc_volume'].rolling(period).mean().iloc[-1], 2)

    yhat_df = pd.DataFrame({'btc_volume': [volume]}, 
                           index = validate.index)

    
    return volume, yhat_df


# In[21]:


def get_btc_120d_average(train, validate, volume, yhat_df):
    
    period = 120
    volume = round(train['btc_volume'].rolling(period).mean().iloc[-1], 2)

    yhat_df = pd.DataFrame({'btc_volume': [volume]}, 
                           index = validate.index)

    
    return volume, yhat_df


# In[10]:


def get_btc_previous_cycle(train, validate):
    
    yhat_df = train['2020'] + train.diff(365).mean()
    yhat_df.index = validate.index
    
    return yhat_df


# In[15]:


def get_test_btc_21d_average(train, test, volume, yhat_df):
    
    period = 21
    volume = round(train['btc_volume'].rolling(period).mean().iloc[-1], 2)

    yhat_df = pd.DataFrame({'btc_volume': [volume]}, 
                           index = test.index)

    
    return volume, yhat_df

