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
    ''' This function performs root-mean-squared-error calculations
        on validation data and yhat_df, yhat_df being the dataframe
        where predictions from model are stored.
    '''
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 0)
    return rmse


# In[3]:


# plot and evaluate
def plot_and_eval(target_var, train, validate, yhat_df):
    ''' This function prints the rmse and plot of a given models
        performance on the validate data.
    '''
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
    ''' This function is supposed to update an evaluation
        dataframe but was not working as intended when called
        in the final report notebook. Need to refer back to
        final report notebook to determine what would be a better
        way of writing this function.
    '''
    rmse = evaluate(target_var, validate, yhat_df)
    d = {'model_type': [model_type], 'target_var': [target_var], 'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)


# In[5]:


def test_evaluate(target_var, test, yhat_df):
    ''' This function performs the rmse evaluation on test
        data instead of validate data.'''
    rmse = round(sqrt(mean_squared_error(test[target_var], yhat_df[target_var])), 0)
    return rmse


# In[6]:


def plot_and_eval_test(target_var, train, validate, test, yhat_df):
    ''' This function performs the similar to plot_and_eval,
        however this function also plots the test data set 
        and plots performance of the model over test data
        instead of validate.
    '''
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
    ''' This function prints results of last observed model. '''
    
    volume = train['btc_volume'][-1:][0]

    yhat_df = pd.DataFrame({'btc_volume': [volume]}, 
                           index = validate.index)
    

    return volume, yhat_df


# In[8]:


def get_btc_simple_average(train, validate, volume, yhat_df):
    ''' This function prints results of simple average model. '''
    
    # getting the average of btc_volume in train
    volume = round(train['btc_volume'].mean(), 2)


    yhat_df = pd.DataFrame({'btc_volume': [volume]}, 
                           index = validate.index)

    
    return volume, yhat_df


# In[16]:


def get_btc_30d_average(train, validate, volume, yhat_df):
    ''' This function prints results of 30d moving average model. '''
    
    period = 30
    volume = round(train['btc_volume'].rolling(period).mean().iloc[-1], 2)

    yhat_df = pd.DataFrame({'btc_volume': [volume]}, 
                           index = validate.index)

    
    return volume, yhat_df


# In[17]:


def get_btc_7d_average(train, validate, volume, yhat_df):
    ''' This function prints results of 7d moving average model. '''
    
    period = 7
    volume = round(train['btc_volume'].rolling(period).mean().iloc[-1], 2)

    yhat_df = pd.DataFrame({'btc_volume': [volume]}, 
                           index = validate.index)

    
    return volume, yhat_df


# In[18]:


def get_btc_14d_average(train, validate, volume, yhat_df):
    ''' This function prints results of 14d moving average model. '''
    
    period = 14
    volume = round(train['btc_volume'].rolling(period).mean().iloc[-1], 2)

    yhat_df = pd.DataFrame({'btc_volume': [volume]}, 
                           index = validate.index)

    
    return volume, yhat_df


# In[19]:


def get_btc_21d_average(train, validate, volume, yhat_df):
    ''' This function prints results of 21d moving average model. '''
    
    period = 21
    volume = round(train['btc_volume'].rolling(period).mean().iloc[-1], 2)

    yhat_df = pd.DataFrame({'btc_volume': [volume]}, 
                           index = validate.index)

    
    return volume, yhat_df


# In[20]:


def get_btc_28d_average(train, validate, volume, yhat_df):
    ''' This function prints results of 28d moving average model. '''
    
    period = 28
    volume = round(train['btc_volume'].rolling(period).mean().iloc[-1], 2)

    yhat_df = pd.DataFrame({'btc_volume': [volume]}, 
                           index = validate.index)

    
    return volume, yhat_df


# In[21]:


def get_btc_120d_average(train, validate, volume, yhat_df):
    ''' This function prints results of 120d moving average model. '''
    
    period = 120
    volume = round(train['btc_volume'].rolling(period).mean().iloc[-1], 2)

    yhat_df = pd.DataFrame({'btc_volume': [volume]}, 
                           index = validate.index)

    
    return volume, yhat_df


# In[10]:


def get_btc_previous_cycle(train, validate):
    ''' This function prints results of previous cycle model. '''
    
    yhat_df = train['2020'] + train.diff(365).mean()
    yhat_df.index = validate.index
    
    return yhat_df


# In[15]:


def get_test_btc_21d_average(train, test, volume, yhat_df):
    ''' This function prints results of 21d moving average model on test data. '''
    
    period = 21
    volume = round(train['btc_volume'].rolling(period).mean().iloc[-1], 2)

    yhat_df = pd.DataFrame({'btc_volume': [volume]}, 
                           index = test.index)

    
    return volume, yhat_df

