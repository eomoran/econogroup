import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

def load_data():
    """Load precious metals and market indicator data"""
    
    tickers = {'gold': 'GC=F', 'silver': 'SI=F', 'platinum': 'PL=F', 'palladium': 'PA=F'}
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=10*365)
    
    try:
        # Download metals data
        data = yf.download(
            list(tickers.values()),
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        )
        
        # Handle MultiIndex columns for multiple tickers
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close'].copy()
        else:
            # Single ticker case
            prices = pd.DataFrame(data['Close'])
            prices.columns = ['Close']
        
        inverse_tickers = {v: k for k, v in tickers.items()}
        prices = prices.rename(columns=inverse_tickers)
        prices = prices.dropna(how='all')
        
        # Download market indicators
        other_tickers = {
            'vix': '^VIX',
            'us2y_yield': '^IRX',
            'usd_index': 'DX-Y.NYB',
            'us10y_yield': '^TNX',
            'wti_oil': 'CL=F'
        }
        
        other_data = yf.download(
            list(other_tickers.values()),
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False
        )
        
        # Handle MultiIndex columns
        if isinstance(other_data.columns, pd.MultiIndex):
            df1 = other_data['Close'].copy()
        else:
            df1 = pd.DataFrame(other_data['Close'])
            df1.columns = ['Close']
        
        inverse_other = {v: k for k, v in other_tickers.items()}
        df1 = df1.rename(columns=inverse_other)
        df1 = df1.dropna(how='all')
        
        # Combine data with outer join first to see what we have
        df = prices.join(df1, how='outer')
        
        # Forward fill missing values (common for some indices) - up to 5 days
        df = df.fillna(method='ffill', limit=5).fillna(method='bfill', limit=5)
        
        # Now drop any rows that are still completely empty
        df = df.dropna(how='all')
        
        # For metals specifically, don't forward fill - only keep actual trading days
        # But for indices, we already forward filled above
        
        # Calculate log returns
        metals = ['gold', 'silver', 'platinum', 'palladium']
        for metal in metals:
            if metal in df.columns:
                df[f'{metal}_lr'] = np.log(df[metal] / df[metal].shift(1))
        
        if 'vix' in df.columns:
            df['vix_lr'] = np.log(df['vix'] / df['vix'].shift(1))
        if 'usd_index' in df.columns:
            df['usd_index_lr'] = np.log(df['usd_index'] / df['usd_index'].shift(1))
        if 'wti_oil' in df.columns:
            df['wti_oil_lr'] = np.log(df['wti_oil'] / df['wti_oil'].shift(1))
        if 'us10y_yield' in df.columns:
            df['us10y_yield_change'] = df['us10y_yield'] - df['us10y_yield'].shift(1)
        if 'us2y_yield' in df.columns:
            df['us2y_yield_change'] = df['us2y_yield'] - df['us2y_yield'].shift(1)
        
        # Drop infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df, prices
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Return empty dataframes to prevent crashes
        return pd.DataFrame(), pd.DataFrame()