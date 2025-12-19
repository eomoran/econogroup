import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from utils.plotting import create_diagnostic_plots
from utils.formatting import format_coefficients_table, highlight_significant

def display_ols_analysis(st, df, metal_choice):
    st.markdown(f"## 📊 OLS Regression Analysis: {metal_choice.upper()}")
    st.info("💡 **Objective:** Analyse how this metal responds to market stress indicators (VIX, USD, Oil, Yields)")
    
    # Prepare data
    data_metal = df[[f'{metal_choice}_lr', 'vix_lr', 'usd_index_lr', 'wti_oil_lr', 
                     'us10y_yield_change', 'us2y_yield_change']].dropna()
    
    # Check if we have enough data
    if len(data_metal) < 50:
        st.error(f"Not enough data points for {metal_choice}. Need at least 50 observations, have {len(data_metal)}.")
        with st.expander("🔍 Debug Information"):
            st.write(f"**Total rows in main dataframe:** {len(df)}")
            st.write(f"**Rows with {metal_choice} returns:** {df[f'{metal_choice}_lr'].notna().sum()}")
            missing = data_metal.isnull().sum()
            st.dataframe(missing)
            st.info("💡 Try selecting a different metal (Gold or Silver typically have the most data)")
        st.stop()
    
    # Fit OLS model
    try:
        formula = f'{metal_choice}_lr ~ vix_lr + usd_index_lr + wti_oil_lr + us10y_yield_change + us2y_yield_change'
        model_ols = smf.ols(formula=formula, data=data_metal).fit()
    except Exception as e:
        st.error(f"Error fitting OLS model: {str(e)}")
        st.stop()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📊 R-squared", f"{model_ols.rsquared:.4f}")
    with col2: st.metric("🎯 Adj R-squared", f"{model_ols.rsquared_adj:.4f}")
    with col3: st.metric("📉 AIC", f"{model_ols.aic:.2f}")
    with col4: st.metric("📈 F-statistic", f"{model_ols.fvalue:.2f}")
    
    st.markdown("---")
    
    # Coefficients table
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📋 Regression Coefficients")
        coef_df = format_coefficients_table(model_ols)
        styled_df = coef_df.style.apply(highlight_significant, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 🧠 Key Insights")
        vix_coef = model_ols.params['vix_lr']
        vix_pval = model_ols.pvalues['vix_lr']
        
        if vix_pval < 0.05:
            # Calculate 10% VIX impact
            impact_10pct = vix_coef * 10
            
            # Determine magnitude
            abs_coef = abs(vix_coef)
            if abs_coef < 0.02:
                strength = "weak"
            elif abs_coef < 0.08:
                strength = "moderate"
            else:
                strength = "strong"
            
            if vix_coef > 0:
                st.success(f"**Positive Response to Volatility** ✅\n\n{metal_choice.title()} shows a **{strength}** positive relationship with market volatility.\n\n**VIX Coefficient:** {vix_coef:.4f}\n\n**Interpretation:** When VIX rises by 10%, {metal_choice} returns typically change by {impact_10pct:+.3f}%.")
            else:
                st.warning(f"**Negative Response to Volatility** ⚠️\n\n{metal_choice.title()} shows a **{strength}** negative relationship with market volatility.\n\n**VIX Coefficient:** {vix_coef:.4f}\n\n**Interpretation:** When VIX rises by 10%, {metal_choice} returns typically change by {impact_10pct:+.3f}%.")
        else:
            st.info(f"**No Significant Response to Volatility**\n\n{metal_choice.title()} does not show a statistically significant relationship with market volatility.\n\n**VIX Coefficient:** {vix_coef:.4f}\n**P-value:** {vix_pval:.4f}\n\n*Changes in VIX do not significantly explain {metal_choice} returns.*")
        
        st.metric("🎲 Durbin-Watson", f"{sm.stats.stattools.durbin_watson(model_ols.resid):.3f}", 
                 help="Tests for autocorrelation. ~2.0 is ideal")
    
    # Diagnostic plots
    st.markdown("### 📈 Model Diagnostics")
    fig = create_diagnostic_plots(model_ols)
    st.plotly_chart(fig, use_container_width=True)
