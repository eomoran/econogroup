# utils/formatting.py
import pandas as pd
import streamlit as st

def format_coefficients_table(model_ols):
    variable_names = {
            'Intercept': 'Intercept',
            'vix_lr': 'VIX Returns',
            'usd_index_lr': 'USD Index Returns',
            'wti_oil_lr': 'WTI Oil Returns',
            'us10y_yield_change': 'Change in 10Y Yield',
            'us2y_yield_change': 'Change in 2Y Yield'
        }
    coef_df = pd.DataFrame({
        'Variable': [variable_names.get(var, var) for var in model_ols.params.index],
        'Coefficient': model_ols.params.values,
        'Std Error': model_ols.bse.values,
        'P-value': model_ols.pvalues.values,
        'Significant': ['✅ Yes' if p < 0.05 else '❌ No' for p in model_ols.pvalues.values]
    })
    coef_df['Coefficient'] = coef_df['Coefficient'].apply(lambda x: f"{x:.6f}")
    coef_df['Std Error'] = coef_df['Std Error'].apply(lambda x: f"{x:.6f}")
    coef_df['P-value'] = coef_df['P-value'].apply(lambda x: f"{x:.4f}")
    return coef_df

def highlight_significant(row):
            # Detect current theme
            theme = st.get_option("theme.base")
            
            if '✅' in row['Significant']:
                if theme == "dark":
                    return ['background-color: #1e5631'] * len(row)  # Dark green
                else:
                    return ['background-color: #d4edda'] * len(row)  # Light green
            else:
                if theme == "dark":
                    return ['background-color: #5a1f1f'] * len(row)  # Dark red
                else:
                    return ['background-color: #f8d7da'] * len(row)  # Light red

def color_forecast_returns(val):
        if isinstance(val, (int, float)):
            color = '#d4edda' if val > 0 else '#f8d7da' if val < 0 else 'white'
            return f'background-color: {color}; font-weight: bold'
        return ''
