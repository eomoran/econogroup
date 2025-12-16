# 💰 Precious Metals Forecasting Dashboard

Interactive Financial Econometrics Dashboard for analyzing precious metals (Gold, Silver, Platinum, Palladium) using OLS, ARIMA, and GARCH models.

## 🚀 Quick Start

### Installation

1. Clone this repository or download the files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Dashboard

```bash
streamlit run metals_dashboard.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## 📊 Features

### Three Econometric Models:
- **OLS Regression**: Analyzes safe-haven properties of metals against VIX, USD Index, WTI Oil, and yields
- **ARIMA Forecasting**: Predicts future price returns based on historical patterns
- **GARCH Volatility**: Models and forecasts volatility clustering in precious metals

### Interactive Controls:
- Select from 4 precious metals (Gold, Silver, Platinum, Palladium)
- Choose between OLS, ARIMA, or GARCH analysis
- Adjust forecast horizons dynamically
- View interactive charts and statistics

### Data Tabs:
1. **Model Analysis**: Deep dive into each econometric model with diagnostics
2. **Interactive Charts**: Historical price charts and returns distributions
3. **Quick Stats**: Correlation heatmaps and performance metrics

## 📦 What's Included

- `metals_dashboard.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `README.md` - This file

## 🎓 Academic Context

This dashboard was built for FIN41660 Financial Econometrics at University College Dublin.

**Models Implemented:**
- Ordinary Least Squares (OLS) with HAC-robust standard errors
- ARIMA for time series forecasting with automatic model selection (AIC/BIC)
- GARCH(1,1) for volatility modeling and forecasting

**Data Sources:**
- Real-time data from Yahoo Finance
- 10 years of historical daily data
- Precious metals futures prices
- Market indicators (VIX, USD Index, yields, WTI oil)

## ⚠️ Notes

- Data is fetched in real-time from Yahoo Finance (requires internet connection)
- First load may take 30-60 seconds to download data
- Data is cached for 1 hour to improve performance
- For educational purposes only - not financial advice

## 🐛 Troubleshooting

If you encounter errors:

1. **Import errors**: Make sure all packages are installed
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **Data loading issues**: Check your internet connection

3. **Matplotlib backend issues**: Try setting the backend
   ```python
   import matplotlib
   matplotlib.use('Agg')
   ```

## 📝 Assignment Requirements Met

✅ OLS regression with multiple regressors  
✅ ARIMA model with automatic selection  
✅ GARCH volatility modeling  
✅ Interactive user interface  
✅ Real-time data loading  
✅ Forecasting with visualization  
✅ Model diagnostics and evaluation  
✅ Professional documentation  

---

Built with ❤️ using Python, Streamlit, and Financial Econometrics
