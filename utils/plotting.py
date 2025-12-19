# utils/plotting.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

def create_diagnostic_plots(model_ols):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Residuals Over Time', 'Q-Q Plot'), horizontal_spacing=0.1)
    
    # Residuals over time
    fig.add_trace(go.Scatter(x=model_ols.resid.index, y=model_ols.resid.values, mode='lines', line=dict(color='steelblue', width=1), name='Residuals'), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2, row=1, col=1)
    
    # Q-Q plot
    (osm, osr), (slope, intercept, r) = stats.probplot(model_ols.resid, dist="norm")
    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', marker=dict(color='steelblue', size=5, opacity=0.6), name='Sample Quantiles'), row=1, col=2)
    fig.add_trace(go.Scatter(x=osm, y=slope * osm + intercept, mode='lines', line=dict(color='red', dash='dash', width=2), name='Reference Line', showlegend=False), row=1, col=2)
    
    fig.update_layout(height=400, showlegend=True, template='plotly_white')
    return fig
