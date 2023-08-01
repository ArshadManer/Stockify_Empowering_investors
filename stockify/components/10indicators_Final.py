#!/usr/bin/env python
# coding: utf-8

# 1. Simple Moving Average (SMA)
# 2. Exponential Moving Average (EMA)
# 3. Average True Range (ATR)
# 4. RSI (Relative Strength Index)
# 5. High/Low of previous Session 
# 6. Standard Deviation
# 7. Bollinger Bands
# 8. Moving Average Convergence/Divergence (MACD)
# 9. SMA Crossover
# 10. Stochastic Oscillator

# In[1]:


import pandas as pd  # for data analysis and calculation of technical indicators
import plotly.express as px  # for data visualization
from IPython.display import display, Markdown, Latex  # to display results in Jupyter Notebook


# In[2]:


df = pd.read_csv("C:/Users/prxsh/OneDrive/Documents/Notebooks/Stockify/DataSets/TCS_data.csv") 
df.head()


# In[3]:


df['Datetime'] = pd.to_datetime(df['Datetime']).dt.date  
df.head()


# ---
# 
#  Close Prices 

# In[4]:


fig = px.line(df, x='Datetime', y='Close', title='TATAINR - Close Prices')
display(fig)  


# ---
# 
# Simple Moving Average (SMA)
# 
# 
# 

# In[5]:


sma_period = 10  # using 10 close prices for the average calculation.
df['sma_10'] = df['Close'].rolling(sma_period).mean()
display(df[['Datetime', 'Close', 'sma_10']])
fig_sma = px.line(df, x='Datetime', y=['Close', 'sma_10'], title='SMA Indicator') 
display(fig_sma)


# ---
# 
# 2. Exponential Moving Average (EMA)
# 

# In[6]:


ema_period = 10  
df['ema_10'] = df['Close'].ewm(span=ema_period, min_periods=ema_period).mean()
display(df[['Datetime', 'Close', 'ema_10']])

fig_ema = px.line(df, x='Datetime', y=['Close', 'ema_10'], title='EMA Indicator')  # to plot EMA, add it to the y parameter
display(fig_ema)


#  Comparison between SMA and EMA- so that the person visiting our website can see which he wants to use for the stock prediction as the EMA is more sensitive towards the closing price(green line) as compared to SMA(red line) i.e. for faster signal.

# In[ ]:


fig_sma_ema_compare = px.line(df, x='Datetime', y=['Close', 'sma_10', 'ema_10'], title='Comparison SMA vs EMA')
display(fig_sma_ema_compare)


# ---
# 
# 3. Average True Range (ATR)
# 

# In[ ]:


atr_period = 14  # same as ema and sma
df['Range'] = df['High'] - df['Low']
df['atr_14'] = df['Range'].rolling(atr_period).mean()
display(df[['Datetime', 'atr_14']])

# plotting atr
fig_atr = px.line(df, x='Datetime', y='atr_14', title='ATR Indicator')
display(fig_atr)


# ---
# 
#  4. Relative Strength Index (RSI)
# 

# In[ ]:


# setting the rsi period
rsi_period = 14

df['gain'] = (df['Close'] - df['Open']).apply(lambda x: x if x > 0 else 0)
df['loss'] = (df['Close'] - df['Open']).apply(lambda x: -x if x < 0 else 0)

# expo moving average calculation done here.
df['ema_gain'] = df['gain'].ewm(span=rsi_period, min_periods=rsi_period).mean()
df['ema_loss'] = df['loss'].ewm(span=rsi_period, min_periods=rsi_period).mean()

df['rs'] = df['ema_gain'] / df['ema_loss']
df['rsi_14'] = 100 - (100 / (df['rs'] + 1))
display(df[['Datetime', 'rsi_14', 'rs', 'ema_gain', 'ema_loss']])
fig_rsi = px.line(df, x='Datetime', y='rsi_14', title='RSI Indicator')
overbought_level = 70
orversold_level = 30

# adding the lines so that the user can see at which point the tata stock or any other stock was being overly sold or overly bought
fig_rsi.add_hline(y=overbought_level, opacity=0.5)
fig_rsi.add_hline(y=orversold_level, opacity=0.5)

display(fig_rsi)


# ---
# 
#  5. High/Low of previous Session 
# 
# for intraday traders and used to compare the sessions between yesterdays high low with the current ones.

# In[ ]:


# using shift for shifting values by 1
df['prev_high'] = df['High'].shift(1)
df['prev_low'] = df['Low'].shift(1)

display(df[['Close', 'High', 'prev_high', 'Low', 'prev_low']])

fig_prev_hl = px.line(df, x='Datetime', y=['Close', 'prev_high', 'prev_low'])
display(fig_prev_hl)


# ---
# 
#  6. Standard Deviation
# 
# if the standard deviation is high, it means the market is ,more volatile
# 
# 
# period 20 calculation

# In[ ]:


deviation_period = 20
 
df['std_20'] = df['Close'].rolling(20).std()
display(df[['Datetime', 'Close', 'std_20']])

fig_std = px.line(df, x='Datetime', y='std_20', title="Standard Deviation")
display(fig)
display(fig_std)


# ---
# 
#  7. Bollinger Bands
# 
# Simple Moving Average (Period 20), a Lower Band, and an Upper Band. The Bands are usually 2 Standard Deviations away from the Moving Average
# 
# Bollinger Bands are often used for Mean Reversion Strategies but can also indicate breakouts.

# In[ ]:


sma_period = 20

# calculating individual components of Bollinger Bands
df['sma_20'] = df['Close'].rolling(sma_period).mean()
df['upper_band_20'] = df['sma_20'] + 2 * df['std_20']
df['lower_band_20'] = df['sma_20'] - 2 * df['std_20']

display(df[['Datetime', 'Close', 'sma_20', 'upper_band_20', 'lower_band_20']])

# plotting Bollinger Bands
fig_bollinger = px.line(df, x='Datetime', y=['Close', 'sma_20', 'upper_band_20', 'lower_band_20'], title='Bollinger Bands')
display(fig_bollinger)


# ---
# 
#  8. Moving Average Convergence/Divergence (MACD)
# 
# MACD is a trend indicator trying to predict trends and reversals at an early stage. That is done by looking at the relationship of a **fast EMA (period 12)** and a **slow EMA (period 26)**.
# 
# **The MACD Indicator is the difference obtained by subtracting EMA26 - EMA12**.
# 
# **Calculating the EMA of the MACD (period 9) generates a Signal Line**. The crossover between the MACD and the Signal Line can be an indication of a Trend Reversal
# 
# red line crosses the blue line means its the shift from uptrend to downtrend and vise versa(reversal in trend)

# In[ ]:


# setting the EMA periods
fast_ema_period = 12
slow_ema_period = 26

# calculating EMAs
df['ema_12'] = df['Close'].ewm(span=fast_ema_period, min_periods=fast_ema_period).mean()
df['ema_26'] = df['Close'].ewm(span=slow_ema_period, min_periods=slow_ema_period).mean()

# calculating MACD by subtracting the EMAs
df['macd'] = df['ema_26'] - df['ema_12']

# calculating to Signal Line by taking the EMA of the MACD
signal_period = 9
df['macd_signal'] = df['macd'].ewm(span=signal_period, min_periods=signal_period).mean()

display(df[['Datetime', 'Close', 'macd', 'macd_signal']])

# Plotting
fig_macd = px.line(df, x='Datetime', y=[df['macd'], df['macd_signal']])
display(fig_macd)


# ---
# 
#  9. SMA Crossover
# 
# SMA Crossover are indicators to determine change in trends. **They consist of a fast Moving Average and a slow Moving Average**.
# 
# I have used period 10 and period 20 for the calculation of the Simple Moving Averages for general stocks as we have only 5 stocks currently.

# In[ ]:


# setting the SMA Periods
fast_sma_period = 10
slow_sma_period = 20

# calculating fast SMA
df['sma_10'] = df['Close'].rolling(fast_sma_period).mean()

# To find crossovers, previous SMA value is necessary using shift()
df['prev_sma_10'] = df['sma_10'].shift(1)

# calculating slow SMA
df['sma_20'] = df['Close'].rolling(slow_sma_period).mean()

# function to find crossovers
def sma_cross(row):
    
    bullish_crossover = row['sma_10'] >= row['sma_20'] and row['prev_sma_10'] < row['sma_20']
    bearish_crossover = row['sma_10'] <= row['sma_20'] and row['prev_sma_10'] > row['sma_20']
    
    if bullish_crossover or bearish_crossover:
        return True

# applying function to dataframe
df['crossover'] = df.apply(sma_cross, axis=1)

# plotting moving averages
fig_crossover = px.line(df, x='Datetime', y=['Close', 'sma_10', 'sma_20'], title='SMA Crossover')

# plotting crossovers
for i, row in df[df['crossover'] == True].iterrows():
    fig_crossover.add_vline(x=row['Datetime'], opacity=0.2)

display(fig_crossover)


# ---
# 
#  10. Stochastic Oscillator
# 
# The Stochastic Oscillator is similar to RSI but uses High/Low values for a specific period for the calculation. It helps you determine overbought and oversold levels.
# 
# Stochastic Oscillator with the Period 14

# In[ ]:


stochastic_period = 14

#maximum high and min low 
df['14_period_low'] = df['Low'].rolling(stochastic_period).min()
df['14_period_high'] = df['High'].rolling(stochastic_period).max()

# formula for stochastic oscillator
df['stoch_osc'] = (df['Close'] - df['14_period_low']) / (df['14_period_high'] - df['14_period_low'])
display(df[['Datetime', 'stoch_osc']])

fig_stoch_osc = px.line(df, x='Datetime', y='stoch_osc', title='Stochastic Oscillator')
display(fig_stoch_osc)


# In[ ]:




