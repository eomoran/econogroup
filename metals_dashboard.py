"""
🏆 Precious Metals vs Market Volatility Dashboard
Interactive Financial Econometrics Application
FIN41660 - Group Project
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Page config
st.set_page_config(
    page_title="💰 Metals Forecasting Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for fun styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        background-color: #FFD700;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        border: 3px solid #FFA500;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FFA500;
        transform: scale(1.05);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    h1 {
        color: #FFD700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    h2, h3 {
        color: #FFA500 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title with emojis
st.markdown("# 💰 Precious Metals Forecasting Lab 📈")
st.markdown("### 🔬 *Analyze • Forecast • Predict Market Volatility*")

# Sidebar
st.sidebar.markdown("## 🎮 Control Panel")
st.sidebar.markdown("---")

# Data loading with caching
@st.cache_data(ttl=3600)
def load_data():
    """Load precious metals and market indicator data"""
    
    # Define tickers
    tickers = {'gold': 'GC=F', 'silver': 'SI=F', 'platinum': 'PL=F', 'palladium': 'PA=F'}
    
    # Date range
    end_date = datetime.today()
    start_date = end_date - timedelta(days=10*365)
    
    # Download metals data
    data = yf.download(
        list(tickers.values()),
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        progress=False
    )
    
    inverse_tickers = {v: k for k, v in tickers.items()}
    prices = data['Close'].copy()
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
    
    inverse_other = {v: k for k, v in other_tickers.items()}
    df1 = other_data['Close'].copy()
    df1 = df1.rename(columns=inverse_other)
    df1 = df1.dropna(how='all')
    
    # Combine data
    df = prices.join(df1, how='inner')
    
    # Calculate log returns
    metals = ['gold', 'silver', 'platinum', 'palladium']
    for metal in metals:
        df[f'{metal}_lr'] = np.log(df[metal] / df[metal].shift(1))
    
    df['vix_lr'] = np.log(df['vix'] / df['vix'].shift(1))
    df['usd_index_lr'] = np.log(df['usd_index'] / df['usd_index'].shift(1))
    df['wti_oil_lr'] = np.log(df['wti_oil'] / df['wti_oil'].shift(1))
    df['us10y_yield_change'] = df['us10y_yield'] - df['us10y_yield'].shift(1)
    df['us2y_yield_change'] = df['us2y_yield'] - df['us2y_yield'].shift(1)
    
    return df, prices

# Load data with spinner
with st.spinner("🔄 Loading market data from Yahoo Finance..."):
    df, prices = load_data()

st.sidebar.success(f"✅ Data loaded: {len(df)} trading days")
st.sidebar.markdown(f"📅 **Period:** {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

# Metal selection
st.sidebar.markdown("---")
st.sidebar.markdown("## 🎯 Select Your Metal")
metal_choice = st.sidebar.selectbox(
    "Choose a precious metal:",
    ['gold', 'silver', 'platinum', 'palladium'],
    format_func=lambda x: f"{'🥇' if x=='gold' else '🥈' if x=='silver' else '⚪' if x=='platinum' else '⚫'} {x.title()}"
)

# Model selection
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔧 Choose Your Model")
model_choice = st.sidebar.radio(
    "Select forecasting method:",
    ['📊 OLS Regression', '📈 ARIMA Forecasting', '📉 GARCH Volatility'],
    help="Each model reveals different market insights!"
)

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["📊 Model Analysis", "📈 Interactive Charts", "🎲 Quick Stats"])

with tab1:
    st.markdown(f"## 🔍 Analyzing: **{metal_choice.upper()}** {' 🥇' if metal_choice=='gold' else '🥈' if metal_choice=='silver' else '⚪' if metal_choice=='platinum' else '⚫'}")
    
    if model_choice == '📊 OLS Regression':
        st.markdown("### 🎯 OLS: Safe Haven Analysis")
        st.info("💡 **What does this tell us?** How does this metal respond to market stress indicators like VIX, USD strength, and oil prices?")
        
        # Prepare data
        data_metal = df[[f'{metal_choice}_lr', 'vix_lr', 'usd_index_lr', 'wti_oil_lr', 
                         'us10y_yield_change', 'us2y_yield_change']].dropna()
        
        # Fit OLS model
        formula = f'{metal_choice}_lr ~ vix_lr + usd_index_lr + wti_oil_lr + us10y_yield_change + us2y_yield_change'
        model_ols = smf.ols(formula=formula, data=data_metal).fit()
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 R-squared", f"{model_ols.rsquared:.4f}", 
                     help="How much variance is explained by the model")
        with col2:
            st.metric("🎯 Adj R-squared", f"{model_ols.rsquared_adj:.4f}")
        with col3:
            st.metric("📉 AIC", f"{model_ols.aic:.2f}")
        
        # Coefficients
        st.markdown("#### 🔢 Regression Coefficients")
        coef_df = pd.DataFrame({
            'Variable': model_ols.params.index,
            'Coefficient': model_ols.params.values,
            'P-value': model_ols.pvalues.values,
            'Significant': ['✅' if p < 0.05 else '❌' for p in model_ols.pvalues.values]
        })
        
        st.dataframe(coef_df.style.highlight_max(subset=['Coefficient'], color='lightgreen')
                    .highlight_min(subset=['Coefficient'], color='lightcoral'), use_container_width=True)
        
        # Interpretation
        st.markdown("#### 🧠 Key Insights:")
        vix_coef = model_ols.params['vix_lr']
        if vix_coef > 0:
            st.success(f"✅ **Safe Haven Confirmed!** When VIX rises by 1%, {metal_choice} returns increase by {vix_coef:.4f}%")
        else:
            st.warning(f"⚠️ **Not a Safe Haven!** When VIX rises by 1%, {metal_choice} returns decrease by {abs(vix_coef):.4f}%")
        
        # Residuals plot
        st.markdown("#### 📉 Model Diagnostics")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Residuals over time
        ax1.plot(model_ols.resid.index, model_ols.resid.values, alpha=0.7, linewidth=0.5)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_title('Residuals Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Residuals')
        ax1.grid(True, alpha=0.3)
        
        # QQ plot
        sm.qqplot(model_ols.resid, line='s', ax=ax2)
        ax2.set_title('Q-Q Plot')
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
    elif model_choice == '📈 ARIMA Forecasting':
        st.markdown("### 🔮 ARIMA: Price Return Forecasting")
        st.info("💡 **What does this tell us?** Predicts future returns based on past patterns (autoregressive + moving average)")
        
        # Prepare data
        returns = df[f'{metal_choice}_lr'].dropna()
        returns = returns.asfreq('B')  # Business day frequency
        
        # Allow user to select forecast horizon
        forecast_days = st.slider("📅 Forecast Horizon (business days):", 1, 30, 10)
        
        # ARIMA model selection
        with st.spinner("🤖 Finding best ARIMA model..."):
            arma_results = []
            for p in range(0, 3):
                for q in range(0, 3):
                    try:
                        model = ARIMA(returns, order=(p, 0, q))
                        fitted = model.fit()
                        arma_results.append({
                            'AR(p)': p,
                            'MA(q)': q,
                            'AIC': fitted.aic,
                            'BIC': fitted.bic
                        })
                    except:
                        pass
            
            arma_df = pd.DataFrame(arma_results).sort_values('AIC')
        
        st.markdown("#### 🏆 Best ARIMA Models (by AIC)")
        st.dataframe(arma_df.head(5), use_container_width=True)
        
        # Fit best model
        best_p = int(arma_df.iloc[0]['AR(p)'])
        best_q = int(arma_df.iloc[0]['MA(q)'])
        
        best_model = ARIMA(returns, order=(best_p, 0, best_q))
        best_fitted = best_model.fit()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Best Model", f"ARMA({best_p}, {best_q})")
        with col2:
            st.metric("📊 AIC Score", f"{best_fitted.aic:.2f}")
        
        # Generate forecast
        forecast = best_fitted.forecast(steps=forecast_days)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Historical returns (last 252 days)
        ax.plot(returns.index[-252:], returns.values[-252:], label='Historical Returns', color='steelblue', linewidth=2)
        
        # Forecast
        future_dates = pd.date_range(returns.index[-1] + timedelta(days=1), periods=forecast_days, freq='B')
        ax.plot(future_dates, forecast.values, label='Forecast', color='orange', linewidth=2, linestyle='--')
        ax.axhline(y=0, color='red', linestyle=':', alpha=0.5)
        
        ax.set_title(f'{metal_choice.title()} Returns Forecast - ARMA({best_p},{best_q})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Log Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Forecast summary
        st.markdown("#### 🎲 Forecast Summary")
        forecast_df = pd.DataFrame({
            'Date': future_dates.strftime('%Y-%m-%d'),
            'Predicted Return (%)': (forecast.values * 100).round(4)
        })
        st.dataframe(forecast_df, use_container_width=True)
        
        avg_forecast = forecast.mean() * 100
        if avg_forecast > 0:
            st.success(f"📈 **Bullish Signal!** Average predicted return: +{avg_forecast:.3f}%")
        else:
            st.warning(f"📉 **Bearish Signal!** Average predicted return: {avg_forecast:.3f}%")
        
    else:  # GARCH
        st.markdown("### ⚡ GARCH: Volatility Forecasting")
        st.info("💡 **What does this tell us?** Models volatility clustering - periods of high volatility tend to follow high volatility")
        
        # Prepare data
        returns = df[f'{metal_choice}_lr'].dropna() * 100  # Scale to percentage
        
        # Forecast horizon
        forecast_days = st.slider("📅 Volatility Forecast Horizon:", 1, 30, 10)
        
        # Fit GARCH(1,1)
        with st.spinner("⚙️ Fitting GARCH(1,1) model..."):
            garch_model = arch_model(returns, vol='GARCH', p=1, q=1)
            garch_fitted = garch_model.fit(disp='off')
        
        # Display model summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Model", "GARCH(1,1)")
        with col2:
            st.metric("🎯 Log-Likelihood", f"{garch_fitted.loglikelihood:.2f}")
        with col3:
            st.metric("📉 AIC", f"{garch_fitted.aic:.2f}")
        
        # Parameters
        st.markdown("#### 🔢 Model Parameters")
        params_df = pd.DataFrame({
            'Parameter': garch_fitted.params.index,
            'Value': garch_fitted.params.values,
            'Std Error': garch_fitted.std_err.values
        })
        st.dataframe(params_df, use_container_width=True)
        
        # Generate forecast
        forecast_vol = garch_fitted.forecast(horizon=forecast_days)
        variance_forecast = forecast_vol.variance.values[-1]
        
        # Plot volatility
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Conditional volatility
        cond_vol = garch_fitted.conditional_volatility
        ax1.plot(cond_vol.index[-252:], cond_vol.values[-252:], label='Conditional Volatility', color='purple', linewidth=2)
        ax1.set_title(f'{metal_choice.title()} Conditional Volatility (Last Year)', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Volatility (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volatility forecast
        future_dates = pd.date_range(returns.index[-1] + timedelta(days=1), periods=forecast_days, freq='D')
        ax2.plot(future_dates, np.sqrt(variance_forecast), marker='o', linestyle='--', 
                color='orange', linewidth=2, markersize=6)
        ax2.set_title(f'Volatility Forecast (Next {forecast_days} Days)', fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volatility (%)')
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Volatility stats
        st.markdown("#### 📊 Volatility Statistics")
        current_vol = cond_vol.iloc[-1]
        avg_forecast_vol = np.sqrt(variance_forecast).mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Volatility", f"{current_vol:.3f}%")
        with col2:
            st.metric("Avg Forecast Vol", f"{avg_forecast_vol:.3f}%")
        with col3:
            change = ((avg_forecast_vol - current_vol) / current_vol) * 100
            st.metric("Expected Change", f"{change:+.1f}%")
        
        if avg_forecast_vol > current_vol:
            st.warning("⚠️ **Volatility Expected to INCREASE** - Market uncertainty rising!")
        else:
            st.success("✅ **Volatility Expected to DECREASE** - Market calming down!")

with tab2:
    st.markdown("## 📊 Interactive Price Charts")
    
    # Time period selector
    period = st.selectbox("Select Time Period:", 
                         ['1 Month', '3 Months', '6 Months', '1 Year', '5 Years', 'All Data'])
    
    period_map = {
        '1 Month': 30,
        '3 Months': 90,
        '6 Months': 180,
        '1 Year': 252,
        '5 Years': 252*5,
        'All Data': len(df)
    }
    
    days = period_map[period]
    
    # Price chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index[-days:], df[metal_choice].iloc[-days:], linewidth=2, color='gold' if metal_choice=='gold' else 'silver')
    ax.set_title(f'{metal_choice.title()} Price - Last {period}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Returns distribution
    st.markdown("### 📊 Returns Distribution")
    fig, ax = plt.subplots(figsize=(12, 4))
    returns_subset = df[f'{metal_choice}_lr'].dropna().iloc[-days:]
    ax.hist(returns_subset * 100, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=returns_subset.mean() * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {returns_subset.mean()*100:.3f}%')
    ax.set_title(f'{metal_choice.title()} Daily Returns Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab3:
    st.markdown("## 🎲 Quick Statistics Dashboard")
    
    # Calculate statistics
    returns = df[f'{metal_choice}_lr'].dropna() * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Mean Return", f"{returns.mean():.3f}%")
    with col2:
        st.metric("📉 Volatility", f"{returns.std():.3f}%")
    with col3:
        st.metric("📈 Max Return", f"{returns.max():.3f}%")
    with col4:
        st.metric("📉 Min Return", f"{returns.min():.3f}%")
    
    # Correlation heatmap
    st.markdown("### 🔥 Correlation Heatmap")
    corr_cols = ['gold_lr', 'silver_lr', 'platinum_lr', 'palladium_lr', 'vix_lr', 'usd_index_lr']
    corr_data = df[corr_cols].dropna()
    corr_matrix = corr_data.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Asset Correlation Matrix', fontsize=14, fontweight='bold')
    st.pyplot(fig)
    
    # Recent performance
    st.markdown("### 📅 Recent Performance (Last 30 Days)")
    recent_prices = df[metal_choice].iloc[-30:]
    perf = ((recent_prices.iloc[-1] / recent_prices.iloc[0]) - 1) * 100
    
    if perf > 0:
        st.success(f"📈 {metal_choice.title()} is UP {perf:.2f}% over the last 30 days!")
    else:
        st.error(f"📉 {metal_choice.title()} is DOWN {abs(perf):.2f}% over the last 30 days!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white;'>
        <h4>🎓 Financial Econometrics Project - FIN41660</h4>
        <p>Built with ❤️ using Streamlit | Data from Yahoo Finance</p>
        <p><em>⚠️ For educational purposes only - Not financial advice!</em></p>
    </div>
""", unsafe_allow_html=True)
