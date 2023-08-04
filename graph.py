import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from copy import copy
from scipy import stats
from ta.trend import MACD
from plotly.subplots import make_subplots

from ta.momentum import RSIIndicator
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import matplotlib.dates as mdates
import mplcursors
import pandas_ta as pta

import warnings
warnings.filterwarnings('ignore')

class TechnicalAnalysis:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.dataframes = {}
        self.load_data()

    def load_data(self):
        for file_path in self.file_paths:
            df = pd.read_csv(file_path)
            df = pd.DataFrame(df)
            self.dataframes[file_path] = df
            
    def open_close(self, file_path):
        df = self.dataframes.get(file_path)
        if df is None:
            raise ValueError("File path not found.")
        
        fig = go.Figure(data=[go.Candlestick(x=df['Datetime'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'])])

        fig.update_layout(xaxis_title='Date',yaxis_title='Price',
            xaxis_rangeslider_visible=False)
        
       
        fig.update_layout({
        'plot_bgcolor': '#e5f2ff',  # transparent background
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        return fig
    
    def calculate_sma_signals_plot(self, file_path, short_window=20, long_window=50):
        df = self.dataframes.get(file_path)
        if df is None:
            raise ValueError("File path not found.")
        df['Short_SMA'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
        # Create long_window days simple moving average column
        df['Long_SMA'] = df['Close'].rolling(window=long_window, min_periods=1).mean()
        # Create 'Signal', 'Position', and 'Buy'/'Sell' columns
        df['Signal'] = np.where(df['Short_SMA'] > df['Long_SMA'], 1.0, 0.0)
        df['Position'] = df['Signal'].diff()

        # Create the Plotly figure
        fig = go.Figure()

        # Plot close price
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Close'], mode='lines', name='Close Price'))

        # Plot short_window-day SMA
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Short_SMA'], mode='lines', name=f'{short_window}-day SMA'))

        # Plot long_window-day SMA
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Long_SMA'], mode='lines', name=f'{long_window}-day SMA'))

        # Plot 'Buy' signals
        buy_signals = df[df['Position'] == 1]
        fig.add_trace(go.Scatter(x=buy_signals['Datetime'], y=buy_signals['Short_SMA'], mode='markers', marker=dict(color='green', size=10), name='Buy'))

        # Plot 'Sell' signals
        sell_signals = df[df['Position'] == -1]
        fig.add_trace(go.Scatter(x=sell_signals['Datetime'], y=sell_signals['Long_SMA'], mode='markers', marker=dict(color='red', size=10), name='Sell'))

        # Update layout
        fig.update_layout(xaxis_title='Datetime', yaxis_title='Price in Rupees', showlegend=True)
        return fig
    
    

    def calculate_ema_signals_plot(self, file_path, short_window=50, long_window=200):
        df = self.dataframes.get(file_path)
        if df is None:
            raise ValueError("File path not found.")
        
        df[f'{short_window}_EMA'] = df['Close'].ewm(span=short_window, adjust=False).mean()
        df[f'{long_window}_EMA'] = df['Close'].ewm(span=long_window, adjust=False).mean()
        df['Signal'] = np.where(df[f'{short_window}_EMA'] > df[f'{long_window}_EMA'], 1.0, 0.0)
        df['Position'] = df['Signal'].diff()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Close'], mode='lines', name='Close Price'))
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df[f'{short_window}_EMA'], mode='lines', name=f'{short_window}-day EMA'))
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df[f'{long_window}_EMA'], mode='lines', name=f'{long_window}-day EMA'))
        buy_signals = df[df['Position'] == 1]
        fig.add_trace(go.Scatter(x=buy_signals['Datetime'], y=buy_signals[f'{short_window}_EMA'], mode='markers', marker=dict(color='green', size=10), name='Buy'))
        sell_signals = df[df['Position'] == -1]
        fig.add_trace(go.Scatter(x=sell_signals['Datetime'], y=sell_signals[f'{long_window}_EMA'], mode='markers', marker=dict(color='red', size=10), name='Sell'))
        fig.update_layout(xaxis_title='Datetime', yaxis_title='Price in Rupees', showlegend=True)
        return fig
    

        
    def generate_buy_sell_signals(self, condition_buy, condition_sell, dataframe, strategy):
        last_signal = None
        indicators = []
        buy = []
        sell = []
        for i in range(0, len(dataframe)):
            if condition_buy(i, dataframe) and last_signal != 'Buy':
                last_signal = 'Buy'
                indicators.append(last_signal)
                buy.append(dataframe['Close'].iloc[i])
                sell.append(np.nan)
            elif condition_sell(i, dataframe) and last_signal == 'Buy':
                last_signal = 'Sell'
                indicators.append(last_signal)
                buy.append(np.nan)
                sell.append(dataframe['Close'].iloc[i])
            else:
                indicators.append(last_signal)
                buy.append(np.nan)
                sell.append(np.nan)

        dataframe[f"{strategy}_Last_Signal"] = np.array(last_signal)
        dataframe[f"{strategy}_Indicator"] = np.array(indicators)
        dataframe[f"{strategy}_Buy"] = np.array(buy)
        dataframe[f"{strategy}_Sell"] = np.array(sell)

    def get_rsi(self, company):
        close_prices = company['Close']
        rsi_time_period = 20

        rsi_indicator = RSIIndicator(close_prices, rsi_time_period)
        company['RSI'] = rsi_indicator.rsi()

        low_rsi = 40
        high_rsi = 70

        self.generate_buy_sell_signals(
            lambda x, company: company['RSI'].values[x] < low_rsi,
            lambda x, company: company['RSI'].values[x] > high_rsi,
            company, 'RSI'
        )
        return company
    

    def plot_rsi(self, company):
        rsi = company.iloc[-504:]  # Plot the last 504 rows
        low_rsi = 40
        high_rsi = 70

        fig = make_subplots(rows=2, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

        fig.add_trace(go.Candlestick(x=rsi['Datetime'], open=rsi['Open'], high=rsi['High'], low=rsi['Low'], close=rsi['Close'], increasing_line_color='white', decreasing_line_color='black', name='Candlestick'), row=1, col=1)

        buy_signals = rsi[rsi['RSI_Indicator'] == 'Buy']
        sell_signals = rsi[rsi['RSI_Indicator'] == 'Sell']
        
        fig.add_trace(go.Scatter(x=buy_signals['Datetime'], y=buy_signals['Close'], mode='markers', marker=dict(color='green', symbol='triangle-up'), name='Buy Signal'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_signals['Datetime'], y=sell_signals['Close'], mode='markers', marker=dict(color='red', symbol='triangle-down'), name='Sell Signal'), row=1, col=1)

        fig.add_trace(go.Scatter(x=rsi['Datetime'], y=rsi['RSI'], mode='lines', line=dict(color='blue'), name='RSI'), row=2, col=1)

        fig.add_trace(go.Scatter(x=rsi['Datetime'], y=[low_rsi]*len(rsi), fill='tozeroy', fillcolor='rgba(173, 204, 255, 0.3)', line=dict(color='rgba(0, 0, 0, 0)'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=rsi['Datetime'], y=[high_rsi]*len(rsi), fill='tozeroy', fillcolor='rgba(173, 204, 255, 0.3)', line=dict(color='rgba(0, 0, 0, 0)'), name='RSI Range (40-70)'), row=2, col=1)

        fig.update_layout(height=800, showlegend=False)
        fig.update_xaxes(showline=True, linewidth=2, linecolor='white', mirror=True, row=1, col=1)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='white', mirror=True, row=1, col=1)
        fig.update_xaxes(showline=True, linewidth=2, linecolor='white', mirror=True, row=2, col=1)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='white', mirror=True, row=2, col=1)

        return fig
    
    def analyze_rsi(self, file_path):
        df = self.dataframes.get(file_path)
        if df is None:
            raise ValueError("File path not found.")
        company_with_rsi = self.get_rsi(df)
        fig = self.plot_rsi(company_with_rsi)
        return fig
    
    def ichimoku_graph(self, file_path, TS=12, KS=24, SS=120, CS=24, OS=0):
        dataframe = self.dataframes.get(file_path)
        if dataframe is None:
            raise ValueError("File path not found.")
    
        dataframe['Datetime'] = pd.to_datetime(dataframe['Datetime']).dt.date
        dataframe['TenkanSan'] = pta.ichimoku(high=dataframe['High'], low=dataframe['Low'], close=dataframe['Close'], tenkan=TS, kijun=KS, senkou=SS, include_chikou=True, offset=OS)[0]['ITS_12']
        dataframe['Kijun'] = pta.ichimoku(high=dataframe['High'], low=dataframe['Low'], close=dataframe['Close'], tenkan=TS, kijun=KS, senkou=SS, include_chikou=True, offset=OS)[0]['IKS_24']
        dataframe['SenkanA'] = pta.ichimoku(high=dataframe['High'], low=dataframe['Low'], close=dataframe['Close'], tenkan=TS, kijun=KS, senkou=SS, include_chikou=True, offset=OS)[0]['ISA_12']
        dataframe['SenkanB'] = pta.ichimoku(high=dataframe['High'], low=dataframe['Low'], close=dataframe['Close'], tenkan=TS, kijun=KS, senkou=SS, include_chikou=True, offset=OS)[0]['ISB_24']
        dataframe['Chinkou'] = pta.ichimoku(high=dataframe['High'], low=dataframe['Low'], close=dataframe['Close'], tenkan=TS, kijun=KS, senkou=SS, include_chikou=True, offset=OS)[0]['ICS_24']
        fig = go.Figure()
        days = 500
        dates = dataframe['Datetime'].tail(days)
        candlestick = go.Candlestick(x=dates,open=dataframe['Open'].tail(days),high=dataframe['High'].tail(days),low=dataframe['Low'].tail(days),close=dataframe['Close'].tail(days),name='Candlestick')
        tenkan_sen = go.Scatter(x=dates, y=dataframe['TenkanSan'].tail(days),mode='lines',name='Tenkan Sen',line=dict(color='blue'))
        kijun_sen = go.Scatter(x=dates,y=dataframe['Kijun'].tail(days),mode='lines',name='Kijun Sen',line=dict(color='orange'))
        senkan_a = go.Scatter(x=dates,y=dataframe['SenkanA'].tail(days),mode='lines',name='Senkan A',line=dict(color='purple'))
        senkan_b = go.Scatter(x=dates,y=dataframe['SenkanB'].tail(days),mode='lines',name='Senkan B',line=dict(color='green'))
        Chinkou = go.Scatter(x=dates,y=dataframe['Chinkou'].tail(days),mode='lines',name='Chinkou',line=dict(color='grey'))
        
        fig.add_trace(candlestick)
        fig.add_trace(tenkan_sen)
        fig.add_trace(kijun_sen)
        fig.add_trace(senkan_a)
        fig.add_trace(senkan_b)
        fig.add_trace(Chinkou)
        fig.update_layout(title='Ichimoku Indicator', xaxis_title='Date',yaxis_title='Price',xaxis_rangeslider_visible=False)

        return fig