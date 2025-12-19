# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

from data.data_loading import load_data
from models.ols import display_ols_analysis
from models.arima import display_arima_forecast
from models.garch import display_garch_volatility

# Page config
st.set_page_config(
    page_title="Metals Forecasting Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.markdown("# 💰 Precious Metals Forecasting Dashboard")
st.markdown("### *Advanced Time Series Analysis with OLS, ARIMA & GARCH*")
st.markdown("---")

# Data loading with caching
@st.cache_data(ttl=3600)
def load_data_cached():
    return load_data()

# Load data
with st.spinner("🔄 Loading market data..."):
    df, prices = load_data_cached()

# Check if data loaded successfully
if df.empty:
    st.error("Failed to load data from Yahoo Finance. Please try again later or check your internet connection.")
    st.stop()

# Sidebar
st.sidebar.markdown("## 🎮 Control Panel")
st.sidebar.success(f"✅ Data loaded: {len(df)} trading days")
st.sidebar.info(f"📅 **Period:** {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
st.sidebar.markdown("---")

# Metal selection
metal_choice = st.sidebar.selectbox(
    "🎯 Select Precious Metal:",
    ['gold', 'silver', 'platinum', 'palladium'],
    format_func=lambda x: f"{x.title()} {'🥇' if x=='gold' else '🥈' if x=='silver' else '⚪' if x=='platinum' else '⚫'}"
)

# Model selection
st.sidebar.markdown("---")
model_choice = st.sidebar.radio(
    "🔧 Select Model:",
    ['OLS Regression', 'ARIMA Forecasting', 'GARCH Volatility']
)

# Main content routing
if model_choice == 'OLS Regression':
    display_ols_analysis(st, df, metal_choice)
elif model_choice == 'ARIMA Forecasting':
    display_arima_forecast(st, df, metal_choice)
else:  # GARCH
    display_garch_volatility(st, df, metal_choice)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white;'>
        <h4>Financial Econometrics Project - FIN41660</h4>
        <p>Built using Streamlit | Data from Yahoo Finance</p>
        <p><em>For educational purposes only - Not financial advice</em></p>
    </div>
""", unsafe_allow_html=True)
