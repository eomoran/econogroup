# models/arima.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
from utils.formatting import color_forecast_returns

def display_arima_forecast(st, df, metal_choice):
    st.markdown(f"## 🔮 ARIMA Forecasting: {metal_choice.upper()}")
    st.info("💡 **Objective:** Forecast future returns using AutoRegressive Integrated Moving Average models")
    
    # Prepare data
    returns = df[f'{metal_choice}_lr'].dropna()
    
    # Check if we have data
    if len(returns) < 100:
        st.error(f"Not enough data for {metal_choice}. Need at least 100 observations, have {len(returns)}.")
        st.stop()
    
    try:
        returns = returns.asfreq('B')
    except Exception as e:
        st.error(f"Error setting business day frequency: {str(e)}")
        st.info("Attempting to continue without frequency setting...")
        pass
    
    # Forecast settings
    col1, col2 = st.columns([3, 1])
    with col1:
        forecast_days = st.slider("📅 Forecast Horizon (business days):", 5, 30, 10)
    with col2:
        st.metric("📊 Data Points", f"{len(returns):,}")
    
    st.markdown("---")
    
    # Model selection
    with st.spinner("🤖 Testing ARIMA models..."):
        arma_results = []
        for p in range(0, 4):
            for q in range(0, 4):
                try:
                    model = ARIMA(returns, order=(p, 0, q))
                    fitted = model.fit()
                    arma_results.append({
                        'AR(p)': p,
                        'MA(q)': q,
                        'AIC': fitted.aic,
                        'BIC': fitted.bic,
                        'Log-Likelihood': fitted.llf
                    })
                except:
                    pass
        
        arma_df = pd.DataFrame(arma_results).sort_values('AIC')
    
    # Display model comparison
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🏆 ARIMA Model Comparison")
        st.markdown("*Models ranked by AIC (lower is better)*")
        
        # Style the table with color gradient (using a simple inline function for now)
        display_df = arma_df.head(10).copy()
        display_df['AIC'] = display_df['AIC'].apply(lambda x: f"{x:.2f}")
        display_df['BIC'] = display_df['BIC'].apply(lambda x: f"{x:.2f}")
        display_df['Log-Likelihood'] = display_df['Log-Likelihood'].apply(lambda x: f"{x:.2f}")
        
        # Add rank column
        display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
        
        # Highlight best model
        def highlight_best(row):
            if row['Rank'] == 1:
                return ['background-color: #90EE90; font-weight: bold'] * len(row)
            elif row['Rank'] <= 3:
                return ['background-color: #E8F5E9'] * len(row)
            else:
                return [''] * len(row)
        
        styled_arma = display_df.style.apply(highlight_best, axis=1)
        st.dataframe(styled_arma, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 🎯 Best Model")
        best_p = int(arma_df.iloc[0]['AR(p)'])
        best_q = int(arma_df.iloc[0]['MA(q)'])
        
        st.success(f"**ARMA({best_p}, {best_q})**")
        st.metric("AIC", f"{arma_df.iloc[0]['AIC']:.2f}")
        st.metric("BIC", f"{arma_df.iloc[0]['BIC']:.2f}")
        
        st.markdown("---")
        st.info(f"**AR({best_p})**: Uses {best_p} lag(s)\n\n**MA({best_q})**: Uses {best_q} error term(s)")
    
    # Fit best model and forecast
    best_model = ARIMA(returns, order=(best_p, 0, best_q))
    best_fitted = best_model.fit()
    forecast = best_fitted.forecast(steps=forecast_days)
    
    st.markdown("---")
    
    # Forecast visualisation
    st.markdown("### 📊 Forecast Visualisation")
    
    fig = go.Figure()
    
    # Historical data (last 252 days)
    historical = returns.iloc[-252:]
    fig.add_trace(go.Scatter(
        x=historical.index,
        y=historical.values * 100,  # Convert to percentage
        mode='lines',
        name='Historical Returns',
        line=dict(color='steelblue', width=2),
        hovertemplate='Date: %{x}<br>Return: %{y:.3f}%<extra></extra>'
    ))
    
    # Forecast
    future_dates = pd.date_range(returns.index[-1] + timedelta(days=1), periods=forecast_days, freq='B')
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast.values * 100,  # Convert to percentage
        mode='lines+markers',
        name='Forecast',
        line=dict(color='orange', width=3, dash='dash'),
        marker=dict(size=8, color='orange', symbol='diamond'),
        hovertemplate='Date: %{x}<br>Forecast: %{y:.3f}%<extra></extra>'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dot", line_color="red", opacity=0.5)
    
    fig.update_layout(
        title=dict(text=f'{metal_choice.title()} Returns Forecast - ARMA({best_p},{best_q})', font=dict(size=18, color='#333')),
        xaxis_title='Date',
        yaxis_title='Return (%)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridcolor='LightGray', zeroline=True)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast table
    st.markdown("### 📋 Detailed Forecast")
    
    forecast_df = pd.DataFrame({
        'Date': future_dates.strftime('%Y-%m-%d'),
        'Day': range(1, forecast_days + 1),
        'Predicted Return (%)': (forecast.values * 100).round(4)
    })
    
    # Color code the returns (using shared function from utils)
    styled_forecast = forecast_df.style.applymap(color_forecast_returns, subset=['Predicted Return (%)'])
    st.dataframe(styled_forecast, use_container_width=True, hide_index=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    avg_forecast = forecast.mean() * 100
    with col1:
        st.metric("📊 Average Forecast", f"{avg_forecast:.3f}%")
    with col2:
        st.metric("📈 Max Forecast", f"{(forecast.max() * 100):.3f}%")
    with col3:
        st.metric("📉 Min Forecast", f"{(forecast.min() * 100):.3f}%")
    with col4:
        cumulative = ((1 + forecast).prod() - 1) * 100
        st.metric("🎯 Cumulative Return", f"{cumulative:.3f}%")
    
    if avg_forecast > 0:
        st.success(f"📈 **Bullish Signal!** Model predicts positive returns averaging {avg_forecast:.3f}% per day")
    else:
        st.warning(f"📉 **Bearish Signal!** Model predicts negative returns averaging {avg_forecast:.3f}% per day")
