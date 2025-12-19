# models/garch.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
from arch import arch_model

def display_garch_volatility(st, df, metal_choice):
    st.markdown(f"## ⚡ GARCH Volatility Analysis: {metal_choice.upper()}")
    st.info("💡 **Objective:** Model and forecast volatility clustering using GARCH(1,1)")
    
    # Prepare data
    returns = df[f'{metal_choice}_lr'].dropna() * 100
    
    # Check if we have enough data
    if len(returns) < 100:
        st.error(f"Not enough data for {metal_choice}. Need at least 100 observations, have {len(returns)}.")
        st.stop()
    
    # Forecast settings
    col1, col2 = st.columns([3, 1])
    with col1:
        forecast_days = st.slider("📅 Volatility Forecast Horizon:", 5, 30, 10)
    with col2:
        st.metric("📊 Data Points", f"{len(returns):,}")
    
    st.markdown("---")
    
    # Fit GARCH
    try:
        with st.spinner("⚙️ Fitting GARCH(1,1) model..."):
            garch_model = arch_model(returns, vol='GARCH', p=1, q=1)
            garch_fitted = garch_model.fit(disp='off')
    except Exception as e:
        st.error(f"Error fitting GARCH model: {str(e)}")
        st.info(f"This can happen with certain data patterns. Try a different metal or check your data.")
        st.stop()
    
    # Model summary
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 GARCH Parameters")
        
        params_df = pd.DataFrame({
            'Parameter': garch_fitted.params.index,
            'Value': garch_fitted.params.values,
            'Std Error': garch_fitted.std_err.values,
            'T-Statistic': garch_fitted.tvalues.values
        })
        
        params_df['Value'] = params_df['Value'].apply(lambda x: f"{x:.6f}")
        params_df['Std Error'] = params_df['Std Error'].apply(lambda x: f"{x:.6f}")
        params_df['T-Statistic'] = params_df['T-Statistic'].apply(lambda x: f"{x:.3f}")
        
        styled_params = params_df.style.set_properties(**{
            'background-color': '#f0f2f6',
            'border': '1px solid #ddd',
            'font-weight': 'bold'
        })
        
        st.dataframe(styled_params, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 📊 Model Fit")
        st.metric("Log-Likelihood", f"{garch_fitted.loglikelihood:.2f}")
        st.metric("AIC", f"{garch_fitted.aic:.2f}")
        st.metric("BIC", f"{garch_fitted.bic:.2f}")
        
        st.markdown("---")
        st.info("**GARCH(1,1)**\n\nCaptures volatility clustering and persistence")
    
    # Generate forecast
    forecast_vol = garch_fitted.forecast(horizon=forecast_days)
    variance_forecast = forecast_vol.variance.values[-1]
    
    st.markdown("---")
    
    # Volatility visualisation
    st.markdown("### 📈 Volatility Analysis")
    
    cond_vol = garch_fitted.conditional_volatility
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Historical Conditional Volatility', f'Volatility Forecast ({forecast_days} Days)'),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4]
    )
    
    # Historical volatility
    fig.add_trace(
        go.Scatter(
            x=cond_vol.index[-252:],
            y=cond_vol.values[-252:],
            mode='lines',
            name='Conditional Volatility',
            line=dict(color='purple', width=2),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.2)',
            hovertemplate='Date: %{x}<br>Volatility: %{y:.3f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Forecast
    future_dates = pd.date_range(returns.index[-1] + timedelta(days=1), periods=forecast_days, freq='D')
    forecast_vol_vals = np.sqrt(variance_forecast)
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=forecast_vol_vals,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='orange', width=3, dash='dash'),
            marker=dict(size=10, color='orange', symbol='diamond'),
            hovertemplate='Date: %{x}<br>Forecast: %{y:.3f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1, showgrid=True, gridcolor='LightGray')
    fig.update_xaxes(title_text="Date", row=2, col=1, showgrid=True, gridcolor='LightGray')
    fig.update_yaxes(title_text="Volatility (%)", row=1, col=1, showgrid=True, gridcolor='LightGray')
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1, showgrid=True, gridcolor='LightGray')
    
    fig.update_layout(
        height=700,
        template='plotly_white',
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volatility metrics
    st.markdown("### 📊 Volatility Statistics")
    
    current_vol = cond_vol.iloc[-1]
    avg_forecast_vol = np.sqrt(variance_forecast).mean()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Volatility", f"{current_vol:.3f}%")
    with col2:
        st.metric("Avg Forecast Vol", f"{avg_forecast_vol:.3f}%")
    with col3:
        change = ((avg_forecast_vol - current_vol) / current_vol) * 100
        st.metric("Expected Change", f"{change:+.1f}%")
    with col4:
        st.metric("Annual Vol (252d)", f"{current_vol * np.sqrt(252):.2f}%")
    
    if avg_forecast_vol > current_vol:
        st.warning("⚠️ **Volatility Expected to INCREASE** - Higher risk ahead!")
    else:
        st.success("✅ **Volatility Expected to DECREASE** - Market calming down!")
