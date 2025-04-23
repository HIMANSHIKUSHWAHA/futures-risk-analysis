import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from arch import arch_model
from scipy.stats import norm

# Set page configuration
st.set_page_config(
    page_title="Futures Liquidity Risk Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0f52ba;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
    border-radius: 5px;
    padding: 1.5rem;
    background-color: #f8f9fa;
    color: black;  /* <-- Add this line */
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 1rem;
}
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3d59;
    }
    .metric-delta {
        font-size: 1rem;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .insight-card {
        background-color: #edf2fb;
        border-left: 4px solid #0f52ba;
        padding: 1rem;
        margin-bottom: 1.5rem;
        color: black;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin-bottom: 1.5rem;
        color: black;
    }
    .alert-card {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin-bottom: 1.5rem;
        color: black;
    }
            
    .insight-card h4,
    .warning-card h4,
    .alert-card h4 {
        color: black;
    }
    .card h4, .card p {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)

# SYMBOL_NAME_MAP = {
#     'GC=F': 'Gold Futures',
#     'CL=F': 'Crude Oil Futures',
#     'ES=F': 'S&P 500 Futures'
# }


# Custom components
def metric_card(title, value, delta=None, delta_color="normal"):
    delta_html = f"<span class='metric-delta' style='color:{'green' if delta_color=='positive' else 'red' if delta_color=='negative' else 'gray'};'>{delta}</span>" if delta else ""
    
    st.markdown(f"""
    <div class="card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value} {delta_html}</div>
    </div>
    """, unsafe_allow_html=True)

def insight_card(title, content, type="insight"):
    class_name = {"insight": "insight-card", "warning": "warning-card", "alert": "alert-card"}.get(type, "insight-card")
    icon = {"insight": "💡", "warning": "⚠️", "alert": "🔴"}.get(type, "💡")
    
    st.markdown(f"""
    <div class="{class_name}">
        <h4>{icon} {title}</h4>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

# Data loading functions
@st.cache_data(ttl=3600)
def load_futures_data():
    conn = sqlite3.connect('data/futures_data.db')
    df = pd.read_sql("""
        SELECT * FROM futures_data
        ORDER BY date DESC
    """, conn, parse_dates=['date'])
    conn.close()
    return df

@st.cache_data(ttl=3600)
def load_liquidity_metrics():
    # In a real scenario, this would be loaded from a database
    # Here we'll simulate it based on the futures data
    df = load_futures_data()

    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
    df.dropna(subset=['close', 'volume'], inplace=True)
    
    # Calculate bid-ask spread (simulated)
    df['bid'] = df['close'] * (1 - np.random.uniform(0.0001, 0.002, len(df)))
    df['ask'] = df['close'] * (1 + np.random.uniform(0.0001, 0.002, len(df)))
    df['bid_ask_spread'] = (df['ask'] - df['bid']) / df['close'] * 10000  # In basis points
    
    # Volume metrics
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df['dollar_volume'] = df['volume'] * df['close']
    
    # Calculate market depth (simulated)
    df['market_depth'] = df['volume'] * np.random.uniform(0.5, 1.5, len(df))
    
    # Calculate Amihud illiquidity ratio
    df['daily_return'] = df.groupby('symbol')['close'].pct_change()
    df['amihud_illiquidity'] = abs(df['daily_return']) / df['dollar_volume'] * 10**6
    
    # Order imbalance (simulated)
    df['buy_volume'] = df['volume'] * np.random.uniform(0.4, 0.6, len(df))
    df['sell_volume'] = df['volume'] - df['buy_volume']
    df['order_imbalance'] = (df['buy_volume'] - df['sell_volume']) / df['volume']
    
    # Calculate trading costs (simulated)
    df['implementation_shortfall'] = df['bid_ask_spread'] * np.random.uniform(0.3, 0.7, len(df))
    
    # Market resilience (simulated recovery time after volume shock)
    df['market_resilience'] = np.random.uniform(1, 5, len(df))  # hours
    
    # Volatility based on GARCH
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].sort_values('date')
        if len(symbol_data) > 30:  # Need sufficient data for GARCH
            returns = np.log(symbol_data['close'] / symbol_data['close'].shift(1)).dropna() * 100
            if len(returns) > 30:
                try:
                    model = arch_model(returns, vol='Garch', p=1, q=1)
                    result = model.fit(disp='off')
                    vol = result.conditional_volatility
                    # Map back to original dataframe
                    idx = symbol_data.dropna().index[-len(vol):]
                    df.loc[idx, 'garch_volatility'] = vol.values
                except:
                    pass
    
    return df

# VaR calculation function
@st.cache_data(ttl=3600)
def calculate_var(df, confidence_level=0.95, position_size=1000000):
    df = df.copy()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)
    if df.empty or df['log_return'].isna().all():
        return {
            "historical_var": None,
            "garch_var": None,
            "ewma_var": None,
            "dataframe": df
        }
    
    # Calculate VaR using historical simulation
    historical_var = np.percentile(df['log_return'], (1-confidence_level)*100) * position_size
    
    # Calculate GARCH-based VaR if volatility is available
    if 'garch_volatility' in df.columns and df['garch_volatility'].notna().any():
        z_score = norm.ppf(1-confidence_level)
        df['garch_var'] = -z_score * df['garch_volatility'] / 100 * position_size
        garch_var = df['garch_var'].iloc[-1]
    else:
        garch_var = None

    # Calculate EWMA-based VaR
    if len(df) >= 20:
        # Calculate EWMA volatility (lambda = 0.94)
        lambda_param = 0.94
        returns = df['log_return'].values
        volatility = np.zeros(len(returns))
        volatility[0] = returns[0]**2
        
        for t in range(1, len(returns)):
            volatility[t] = lambda_param * volatility[t-1] + (1 - lambda_param) * returns[t-1]**2
        
        volatility = np.sqrt(volatility)
        df['ewma_volatility'] = volatility
        
        z_score = norm.ppf(1-confidence_level)
        df['ewma_var'] = -z_score * df['ewma_volatility'] * position_size
        ewma_var = df['ewma_var'].iloc[-1]
    else:
        ewma_var = None
        
    return {
        "historical_var": historical_var,
        "garch_var": garch_var,
        "ewma_var": ewma_var,
        "dataframe": df
    }

# Liquidity cost calculation
@st.cache_data(ttl=3600)
def calculate_liquidity_cost(df, trade_size):
    """Calculate the cost of liquidating a position quickly vs. over time"""
    df = df.copy()
    
    # Avg daily volume
    avg_volume = df['volume'].mean()
    
    # Immediate liquidation cost (estimated)
    price_impact = 0.1 * (trade_size / avg_volume) ** 0.5  # Square root model
    immediate_cost = trade_size * price_impact
    
    # Optimal execution cost (estimated)
    days_to_liquidate = min(5, (trade_size / avg_volume) ** 0.5)  # Cap at 5 days
    optimal_cost = immediate_cost * (1 / days_to_liquidate) ** 0.5
    
    # Risk-adjusted cost (includes market risk during liquidation)
    daily_volatility = df['close'].pct_change().std()
    risk_cost = trade_size * daily_volatility * np.sqrt(days_to_liquidate / 2)
    
    return {
        "price_impact_bps": price_impact * 10000,
        "immediate_cost": immediate_cost,
        "optimal_cost": optimal_cost,
        "risk_adjusted_cost": optimal_cost + risk_cost,
        "days_to_liquidate": days_to_liquidate
    }

# Stress test function
@st.cache_data(ttl=3600)
def stress_test_liquidity(df, stress_factor=2.0):
    """Simulate a liquidity crisis by increasing spreads and reducing volume"""
    df = df.copy()
    
    # Increase bid-ask spreads
    df['stressed_spread'] = df['bid_ask_spread'] * stress_factor
    
    # Reduce volume
    df['stressed_volume'] = df['volume'] / stress_factor
    
    # Increase Amihud illiquidity
    df['stressed_amihud'] = df['amihud_illiquidity'] * stress_factor
    
    # Increase market resilience time
    df['stressed_resilience'] = df['market_resilience'] * stress_factor
    
    return df

# Load data
df = load_futures_data()
liquidity_df = load_liquidity_metrics()
SYMBOL_NAME_MAP = {
    'GC=F': 'Gold Futures',
    'CL=F': 'Crude Oil Futures',
    'ES=F': 'S&P 500 Futures'
}
LABEL_TO_SYMBOL = {v: k for k, v in SYMBOL_NAME_MAP.items()}
# Sidebar
st.sidebar.markdown("# Controls")
st.sidebar.markdown("## Futures Selection")
selected_labels = st.sidebar.multiselect(
    "Select Futures Contracts",
    options=list(SYMBOL_NAME_MAP.values()),
    default=list(SYMBOL_NAME_MAP.values())[:3]
)
selected_symbols = [LABEL_TO_SYMBOL[label] for label in selected_labels]
date_range = st.sidebar.date_input(
    "Date Range",
    value=(
        datetime.now() - timedelta(days=90),
        datetime.now()
    ),
    max_value=datetime.now()
)

st.sidebar.markdown("## Risk Parameters")
confidence_level = st.sidebar.slider(
    "Confidence Level for VaR",
    min_value=0.90,
    max_value=0.99,
    value=0.95,
    step=0.01
)

position_size = st.sidebar.number_input(
    "Position Size ($)",
    min_value=100000,
    max_value=10000000,
    value=1000000,
    step=100000
)

trade_pct = st.sidebar.slider(
    "Trade Size (% of Position)",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.1
)

trade_size = position_size * trade_pct

st.sidebar.markdown("## Stress Test")
stress_factor = st.sidebar.slider(
    "Stress Factor",
    min_value=1.0,
    max_value=5.0,
    value=2.0,
    step=0.5
)

stress_test = st.sidebar.checkbox("Apply Stress Test", value=False)

filtered_df = liquidity_df[
    (liquidity_df['symbol'].isin(selected_symbols)) &
    (liquidity_df['date'] >= pd.Timestamp(date_range[0])) &
    (liquidity_df['date'] <= pd.Timestamp(date_range[1]))
]

# Main content
st.markdown('<h1 class="main-header">Futures Market Liquidity Risk Analysis</h1>', unsafe_allow_html=True)

# Dashboard overview
col1, col2, col3 = st.columns(3)

with col1:
    # Average Bid-Ask Spread
    avg_spread = filtered_df['bid_ask_spread'].mean()
    prev_spread = filtered_df[filtered_df['date'] < pd.Timestamp(date_range[0]) - timedelta(days=1)]['bid_ask_spread'].mean()
    spread_delta = f"{((avg_spread / prev_spread) - 1) * 100:.1f}%" if not pd.isna(prev_spread) and prev_spread != 0 else " "
    delta_color = "negative" if avg_spread > prev_spread else "positive"
    metric_card("Avg Bid-Ask Spread (bps)", f"{avg_spread:.2f}", spread_delta, delta_color)

with col2:
    # Average Daily Volume
    avg_volume = filtered_df['volume'].mean()
    prev_volume = filtered_df[filtered_df['date'] < pd.Timestamp(date_range[0]) - timedelta(days=1)]['volume'].mean()
    volume_delta = f"{((avg_volume / prev_volume) - 1) * 100:.1f}%" if not pd.isna(prev_volume) and prev_volume != 0 else " "
    delta_color = "positive" if avg_volume > prev_volume else "negative"
    metric_card("Avg Daily Volume", f"{avg_volume:,.0f}", volume_delta, delta_color)

with col3:
    # Average Amihud Illiquidity
    avg_amihud = filtered_df['amihud_illiquidity'].mean()
    prev_amihud = filtered_df[filtered_df['date'] < pd.Timestamp(date_range[0]) - timedelta(days=1)]['amihud_illiquidity'].mean()
    amihud_delta = f"{((avg_amihud / prev_amihud) - 1) * 100:.1f}%" if not pd.isna(prev_amihud) and prev_amihud != 0 else " "
    delta_color = "negative" if avg_amihud > prev_amihud else "positive"
    metric_card("Avg Amihud Illiquidity", f"{avg_amihud:.4f}", amihud_delta, delta_color)

# Liquidity Risk Summary
st.markdown('<h2 class="sub-header">Liquidity Risk Summary</h2>', unsafe_allow_html=True)

# VaR metrics for each symbol
var_data = []
for symbol in selected_symbols:
    symbol_df = filtered_df[filtered_df['symbol'] == symbol].sort_values('date')
    if len(symbol_df) > 0:
        var_results = calculate_var(symbol_df, confidence_level, position_size)
        
        # Calculate liquidity cost
        liq_cost = calculate_liquidity_cost(symbol_df, trade_size)
        
        var_data.append({
            "symbol": symbol,
            "historical_var": var_results["historical_var"],
            "garch_var": var_results["garch_var"] if var_results["garch_var"] is not None else np.nan,
            "ewma_var": var_results["ewma_var"] if var_results["ewma_var"] is not None else np.nan,
            "avg_spread_bps": symbol_df['bid_ask_spread'].mean(),
            "liquidity_cost": liq_cost["risk_adjusted_cost"],
            "days_to_liquidate": liq_cost["days_to_liquidate"],
            "price_impact_bps": liq_cost["price_impact_bps"],
        })

var_df = pd.DataFrame(var_data)
# var_df = var_df[var_df['historical_var'].notnull()]

if not var_df.empty:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Liquidity-adjusted VaR chart
        fig = go.Figure()
        
        for i, row in var_df.iterrows():
            symbol = row['symbol']
            if row['historical_var'] is None:
                continue  # Skip rows with missing historical_var
            hist_var = abs(row['historical_var'])
            liq_cost = row['liquidity_cost']

            
            fig.add_trace(go.Bar(
                x=[symbol],
                y=[hist_var],
                name='Market Risk (VaR)',
                marker_color='#3182bd'
            ))
            
            fig.add_trace(go.Bar(
                x=[symbol],
                y=[liq_cost],
                name='Liquidity Cost',
                marker_color='#e6550d'
            ))
        
        fig.update_layout(
            title='Liquidity-Adjusted VaR by Instrument',
            barmode='stack',
            xaxis_title='Instrument',
            yaxis_title=f'Value at Risk (${confidence_level*100:.0f}% Confidence)',
            legend_title='Risk Component',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Summary table
        var_df_display = var_df.copy()
        var_df_display["historical_var"] = var_df_display["historical_var"].apply(lambda x: f"${abs(x):,.0f}")
        var_df_display["liquidity_cost"] = var_df_display["liquidity_cost"].apply(lambda x: f"${x:,.0f}")
        var_df_display["days_to_liquidate"] = var_df_display["days_to_liquidate"].apply(lambda x: f"{x:.1f}")
        var_df_display["price_impact_bps"] = var_df_display["price_impact_bps"].apply(lambda x: f"{x:.1f}")
        var_df_display["avg_spread_bps"] = var_df_display["avg_spread_bps"].apply(lambda x: f"{x:.1f}")
        
        var_df_display = var_df_display.rename(columns={
            "symbol": "Instrument",
            "historical_var": "VaR (Historical)",
            "liquidity_cost": "Liquidity Cost",
            "days_to_liquidate": "Days to Liquidate",
            "price_impact_bps": "Price Impact (bps)",
            "avg_spread_bps": "Avg Spread (bps)"
        })
        
        st.dataframe(var_df_display[["Instrument", "VaR (Historical)", "Liquidity Cost", "Days to Liquidate", "Price Impact (bps)"]], use_container_width=True)

else:
    st.warning("Not enough data to calculate VaR for the selected symbols and date range.")

# Key Insights
st.markdown('<h2 class="sub-header">Key Liquidity Insights</h2>', unsafe_allow_html=True)

# Generate insights based on data
if not var_df.empty:
    most_liquid_symbol = var_df.loc[var_df['avg_spread_bps'].idxmin()]['symbol']
    least_liquid_symbol = var_df.loc[var_df['avg_spread_bps'].idxmax()]['symbol']
    
    highest_impact_symbol = var_df.loc[var_df['price_impact_bps'].idxmax()]['symbol']
    highest_impact_value = var_df.loc[var_df['price_impact_bps'].idxmax()]['price_impact_bps']
    
    longest_liquidation = var_df.loc[var_df['days_to_liquidate'].idxmax()]['symbol']
    longest_liquidation_days = var_df.loc[var_df['days_to_liquidate'].idxmax()]['days_to_liquidate']
    
    insight_card(
        "Market Liquidity Assessment",
        f"{most_liquid_symbol} shows the best liquidity profile with the tightest bid-ask spreads, while {least_liquid_symbol} exhibits the widest spreads. This indicates potential challenges in executing large trades in {least_liquid_symbol} without significant market impact."
    )
    
    insight_card(
        "Price Impact Analysis",
        f"Trading {trade_pct*100:.0f}% of your position in {highest_impact_symbol} could result in a price impact of {highest_impact_value:.1f} basis points, substantially increasing transaction costs. Consider implementing a more gradual execution strategy for this instrument."
    )
    
    if longest_liquidation_days > 3:
        insight_card(
        "Position Liquidation Warning",
        f"Liquidating your position in {longest_liquidation} would require approximately {longest_liquidation_days:.1f} days to minimize market impact. This extended liquidation timeline exposes the position to additional market risk during the execution period.",
        type="warning"
    )


# Historical Liquidity Trends
st.markdown('<h2 class="sub-header">Historical Liquidity Trends</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Bid-Ask Spread Trend
    if not filtered_df.empty:
        spread_df = filtered_df.groupby(['date', 'symbol'])['bid_ask_spread'].mean().reset_index()
        
        fig = px.line(
            spread_df,
            x='date',
            y='bid_ask_spread',
            color='symbol',
            title='Bid-Ask Spread Trend (bps)',
            labels={'bid_ask_spread': 'Spread (bps)', 'date': 'Date'},
            template='plotly_white'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data to display bid-ask spread trends.")

with col2:
    # Volume Trend
    if not filtered_df.empty:
        volume_df = filtered_df.groupby(['date', 'symbol'])['volume'].sum().reset_index()
        
        fig = px.line(
            volume_df,
            x='date',
            y='volume',
            color='symbol',
            title='Daily Trading Volume',
            labels={'volume': 'Volume', 'date': 'Date'},
            template='plotly_white'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data to display volume trends.")

# Liquidity Risk Matrix
st.markdown('<h2 class="sub-header">Liquidity Risk Matrix</h2>', unsafe_allow_html=True)

if not filtered_df.empty and not var_df.empty:
    # Calculate normalized metrics for each symbol
    symbol_metrics = []
    
    for symbol in selected_symbols:
        symbol_df = filtered_df[filtered_df['symbol'] == symbol]
        if len(symbol_df) > 0:
            avg_spread = symbol_df['bid_ask_spread'].mean()
            avg_volume = symbol_df['volume'].mean()
            avg_amihud = symbol_df['amihud_illiquidity'].mean()
            
            # Get VaR and liquidity cost
            symbol_var_row = var_df[var_df['symbol'] == symbol]
            if not symbol_var_row.empty:
                var_value = abs(symbol_var_row['historical_var'].values[0])
                liq_cost = symbol_var_row['liquidity_cost'].values[0]
                
                symbol_metrics.append({
                    'symbol': symbol,
                    'avg_spread': avg_spread,
                    'avg_volume': avg_volume,
                    'avg_amihud': avg_amihud,
                    'var_value': var_value,
                    'liq_cost': liq_cost,
                    'liq_var_ratio': liq_cost / var_value if var_value != 0 else np.nan
                })
    
    metrics_df = pd.DataFrame(symbol_metrics)
    
    if not metrics_df.empty:
        # Create bubble chart with Amihud illiquidity vs Spread, size by volume
        fig = px.scatter(
            metrics_df,
            x='avg_spread',
            y='avg_amihud',
            size='avg_volume',
            color='liq_var_ratio',
            hover_name='symbol',
            size_max=50,
            title='Liquidity Risk Matrix: Spread vs Illiquidity',
            labels={
                'avg_spread': 'Bid-Ask Spread (bps)',
                'avg_amihud': 'Amihud Illiquidity Ratio',
                'avg_volume': 'Average Volume',
                'liq_var_ratio': 'Liquidity Cost / VaR Ratio'
            },
            color_continuous_scale='Viridis',
            template='plotly_white'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation of the chart
        st.markdown("""
        **How to interpret this chart:**
        - **X-axis**: Higher bid-ask spreads indicate higher trading costs and potential liquidity issues
        - **Y-axis**: Higher Amihud illiquidity ratio indicates that prices move more per dollar of trading volume
        - **Bubble size**: Larger bubbles represent higher trading volumes (more liquid markets)
        - **Color**: Darker colors indicate higher ratios of liquidity cost to market risk (VaR)
        
        Instruments in the upper right quadrant have the highest liquidity risk, with wide spreads and high price impact per unit of volume.
        """)
    else:
        st.warning("Insufficient data to create the Liquidity Risk Matrix.")
else:
    st.warning("Not enough data to generate the Liquidity Risk Matrix.")

# Stress Test Section
if stress_test:
    st.markdown('<h2 class="sub-header">Liquidity Stress Test Analysis</h2>', unsafe_allow_html=True)
    
    # Apply stress test
    stressed_df = stress_test_liquidity(filtered_df, stress_factor)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Compare normal vs stressed spreads
        spread_comparison = pd.DataFrame({
            'symbol': filtered_df['symbol'].unique()
        })
        
        for symbol in spread_comparison['symbol']:
            symbol_df = filtered_df[filtered_df['symbol'] == symbol]
            symbol_stressed = stressed_df[stressed_df['symbol'] == symbol]
            
            spread_comparison.loc[spread_comparison['symbol'] == symbol, 'normal_spread'] = symbol_df['bid_ask_spread'].mean()
            spread_comparison.loc[spread_comparison['symbol'] == symbol, 'stressed_spread'] = symbol_stressed['stressed_spread'].mean()
        
        fig = go.Figure()
        
        for symbol in spread_comparison['symbol']:
            normal = spread_comparison.loc[spread_comparison['symbol'] == symbol, 'normal_spread'].values[0]
            stressed = spread_comparison.loc[spread_comparison['symbol'] == symbol, 'stressed_spread'].values[0]
            
            fig.add_trace(go.Bar(
                x=[symbol],
                y=[normal],
                name='Normal',
                marker_color='#3182bd'
            ))
            
            fig.add_trace(go.Bar(
                x=[symbol],
                y=[stressed],
                name='Stressed',
                marker_color='#e6550d'
            ))
        
        fig.update_layout(
            title=f'Bid-Ask Spread Under Stress (Factor: {stress_factor}x)',
            barmode='group',
            xaxis_title='Symbol',
            yaxis_title='Spread (bps)',
            legend_title='Scenario',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Recalculate liquidity costs under stress
        stress_costs = []
        
        for symbol in selected_symbols:
            symbol_df = filtered_df[filtered_df['symbol'] == symbol]
            symbol_stressed = stressed_df[stressed_df['symbol'] == symbol]
            
            if len(symbol_df) > 0:
                normal_cost = calculate_liquidity_cost(symbol_df, trade_size)
                stressed_cost = calculate_liquidity_cost(symbol_stressed, trade_size)
                
                stress_costs.append({
                    'symbol': symbol,
                    'normal_cost': normal_cost['risk_adjusted_cost'],
                    'stressed_cost': stressed_cost['risk_adjusted_cost'],
                    'normal_days': normal_cost['days_to_liquidate'],
                    'stressed_days': stressed_cost['days_to_liquidate']
                })
        
        costs_df = pd.DataFrame(stress_costs)
        
        if not costs_df.empty:
            fig = go.Figure()
            
            for symbol in costs_df['symbol']:
                normal = costs_df.loc[costs_df['symbol'] == symbol, 'normal_cost'].values[0]
                stressed = costs_df.loc[costs_df['symbol'] == symbol, 'stressed_cost'].values[0]
                
                fig.add_trace(go.Bar(
                    x=[symbol],
                    y=[normal],
                    name='Normal',
                    marker_color='#3182bd'
                ))
                
                fig.add_trace(go.Bar(
                    x=[symbol],
                    y=[stressed],
                    name='Stressed',
                    marker_color='#e6550d'
                ))
            
            fig.update_layout(
                title=f'Liquidity Cost Under Stress (Factor: {stress_factor}x)',
                barmode='group',
                xaxis_title='Symbol',
                yaxis_title='Cost ($)',
                legend_title='Scenario',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # # Stress test insights
    # max_increase
    # Stress test insights
    max_increase_symbol = costs_df.loc[(costs_df['stressed_cost'] / costs_df['normal_cost']).idxmax()]['symbol']
    max_increase_pct = ((costs_df.loc[(costs_df['stressed_cost'] / costs_df['normal_cost']).idxmax()]['stressed_cost'] / 
                        costs_df.loc[(costs_df['stressed_cost'] / costs_df['normal_cost']).idxmax()]['normal_cost']) - 1) * 100
    
    max_days_symbol = costs_df.loc[costs_df['stressed_days'].idxmax()]['symbol']
    max_days = costs_df.loc[costs_df['stressed_days'].idxmax()]['stressed_days']
    
    insight_card(
        "Stress Test Analysis",
        f"Under a {stress_factor}x liquidity stress scenario, {max_increase_symbol} shows the highest vulnerability with a {max_increase_pct:.1f}% increase in liquidity costs. This indicates potential fragility in market depth and resilience during crisis periods.",
        type="warning"
    )
    
    if max_days > 5:
        insight_card(
            "Execution Risk Alert",
            f"In a stressed market scenario, liquidating your position in {max_days_symbol} could take up to {max_days:.1f} days to minimize market impact. This substantially increases exposure to adverse price movements during the liquidation period.",
            type="alert"
        )

# Order Book Analysis (simulated)
st.markdown('<h2 class="sub-header">Order Book Depth Analysis</h2>', unsafe_allow_html=True)

# Create simulated order book depth for selected symbols
if not filtered_df.empty and len(selected_symbols) > 0:
    # Use the most recent data point for each selected symbol
    recent_data = []
    
    for symbol in selected_symbols:
        symbol_df = filtered_df[filtered_df['symbol'] == symbol].sort_values('date', ascending=False)
        if len(symbol_df) > 0:
            recent_data.append(symbol_df.iloc[0])
    
    if recent_data:
        recent_df = pd.DataFrame(recent_data)
        
        # Simulate order book data
        order_book_data = []
        
        for _, row in recent_df.iterrows():
            symbol = row['symbol']
            price = row['close']
            spread = row['bid_ask_spread'] / 10000 * price  # Convert bps to price
            
            # Best bid and ask
            best_bid = price - spread/2
            best_ask = price + spread/2
            
            # Create price levels
            price_levels = 10
            
            for i in range(price_levels):
                # Decay factor for volume at each level
                volume_decay = 0.8 ** i
                price_decay = (1 + 0.0005 * i)
                
                # Bid side
                bid_price = best_bid * (1 - 0.0005 * i)
                bid_volume = row['volume'] * 0.05 * volume_decay  # 5% of daily volume at best bid
                
                # Ask side
                ask_price = best_ask * price_decay
                ask_volume = row['volume'] * 0.05 * volume_decay  # 5% of daily volume at best ask
                
                order_book_data.append({
                    'symbol': symbol,
                    'side': 'Bid',
                    'level': i+1,
                    'price': bid_price,
                    'volume': bid_volume,
                    'distance_bps': (price - bid_price) / price * 10000
                })
                
                order_book_data.append({
                    'symbol': symbol,
                    'side': 'Ask',
                    'level': i+1,
                    'price': ask_price,
                    'volume': ask_volume,
                    'distance_bps': (ask_price - price) / price * 10000
                })
        
        order_book_df = pd.DataFrame(order_book_data)
        
        # Let user select a symbol for order book visualization
        selected_ob_symbol = st.selectbox("Select Symbol for Order Book Analysis", selected_symbols)
        
        if selected_ob_symbol:
            symbol_ob = order_book_df[order_book_df['symbol'] == selected_ob_symbol]
            
            # Create order book visualization
            bid_data = symbol_ob[symbol_ob['side'] == 'Bid'].sort_values('level')
            ask_data = symbol_ob[symbol_ob['side'] == 'Ask'].sort_values('level')
            
            fig = go.Figure()
            
            # Add bid side
            fig.add_trace(go.Bar(
                x=bid_data['distance_bps'],
                y=bid_data['volume'],
                name='Bid',
                marker_color='green',
                opacity=0.7
            ))
            
            # Add ask side
            fig.add_trace(go.Bar(
                x=ask_data['distance_bps'],
                y=ask_data['volume'],
                name='Ask',
                marker_color='red',
                opacity=0.7
            ))
            
            fig.update_layout(
                title=f'Order Book Depth: {selected_ob_symbol}',
                xaxis_title='Distance from Mid Price (bps)',
                yaxis_title='Volume',
                template='plotly_white',
                barmode='relative',
                height=500
            )
            
            # Add mid price line
            fig.add_shape(
                type="line",
                x0=0, y0=0,
                x1=0, y1=bid_data['volume'].max() * 1.1,
                line=dict(
                    color="Black",
                    width=2,
                    dash="dash",
                )
            )
            
            fig.add_annotation(
                x=0, y=bid_data['volume'].max() * 1.05,
                text="Mid Price",
                showarrow=False,
                font=dict(
                    size=12,
                    color="black"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Order book liquidity analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Calculate total volume at different levels
                bid_volume_5bp = bid_data[bid_data['distance_bps'] <= 5]['volume'].sum()
                bid_volume_10bp = bid_data[bid_data['distance_bps'] <= 10]['volume'].sum()
                
                ask_volume_5bp = ask_data[ask_data['distance_bps'] <= 5]['volume'].sum()
                ask_volume_10bp = ask_data[ask_data['distance_bps'] <= 10]['volume'].sum()
                
                # Create summary table
                depth_summary = pd.DataFrame({
                    'Distance': ['Within 5 bps', 'Within 10 bps'],
                    'Bid Volume': [f"{bid_volume_5bp:,.0f}", f"{bid_volume_10bp:,.0f}"],
                    'Ask Volume': [f"{ask_volume_5bp:,.0f}", f"{ask_volume_10bp:,.0f}"],
                    'Total Volume': [f"{(bid_volume_5bp + ask_volume_5bp):,.0f}", f"{(bid_volume_10bp + ask_volume_10bp):,.0f}"]
                })
                
                st.markdown("#### Order Book Depth Summary")
                st.dataframe(depth_summary, use_container_width=True)
            
            with col2:
                # Calculate market impact for various trade sizes
                trade_sizes = [0.2, 0.5, 0.8, 1.0]
                impact_data = []
                
                for size_pct in trade_sizes:
                    trade_amount = position_size * size_pct
                    
                    # Calculate how much volume we consume and the resulting price impact
                    remaining = trade_amount
                    max_level = 0
                    cumulative_volume = 0
                    
                    for _, row in ask_data.sort_values('level').iterrows():
                        cumulative_volume += row['price'] * row['volume']
                        max_level = row['level']
                        
                        if cumulative_volume >= trade_amount:
                            break
                    
                    # Estimate price impact based on how deep into the book we go
                    impact_bps = ask_data[ask_data['level'] <= max_level]['distance_bps'].max()
                    
                    impact_data.append({
                        'trade_size_pct': size_pct * 100,
                        'trade_amount': trade_amount,
                        'price_impact_bps': impact_bps
                    })
                
                impact_df = pd.DataFrame(impact_data)
                impact_df['trade_amount'] = impact_df['trade_amount'].apply(lambda x: f"${x:,.0f}")
                impact_df['price_impact_bps'] = impact_df['price_impact_bps'].apply(lambda x: f"{x:.2f}")
                
                impact_df = impact_df.rename(columns={
                    'trade_size_pct': 'Position %',
                    'trade_amount': 'Trade Amount',
                    'price_impact_bps': 'Est. Price Impact (bps)'
                })
                
                st.markdown("#### Estimated Price Impact")
                st.dataframe(impact_df, use_container_width=True)

# Market Correlation Analysis
st.markdown('<h2 class="sub-header">Liquidity Correlation Analysis</h2>', unsafe_allow_html=True)

# Create correlation matrix for liquidity metrics across instruments
if not filtered_df.empty and len(selected_symbols) >= 2:
    # Prepare data for correlation analysis
    liquidity_corr_data = []
    
    for date in sorted(filtered_df['date'].unique()):
        date_data = {'date': date}
        
        for symbol in selected_symbols:
            symbol_data = filtered_df[(filtered_df['date'] == date) & (filtered_df['symbol'] == symbol)]
            
            if len(symbol_data) > 0:
                date_data[f'{symbol}_spread'] = symbol_data['bid_ask_spread'].mean()
                date_data[f'{symbol}_volume'] = symbol_data['volume'].mean()
                date_data[f'{symbol}_amihud'] = symbol_data['amihud_illiquidity'].mean()
        
        liquidity_corr_data.append(date_data)
    
    liquidity_time_df = pd.DataFrame(liquidity_corr_data)
    
    # Create separate dataframes for each metric
    spread_cols = [col for col in liquidity_time_df.columns if 'spread' in col]
    volume_cols = [col for col in liquidity_time_df.columns if 'volume' in col]
    amihud_cols = [col for col in liquidity_time_df.columns if 'amihud' in col]
    
    # Calculate correlation matrices
    if len(spread_cols) >= 2:
        spread_corr = liquidity_time_df[spread_cols].corr()
        spread_corr.columns = [col.split('_')[0] for col in spread_corr.columns]
        spread_corr.index = [col.split('_')[0] for col in spread_corr.index]

        # Create heatmap
        fig = px.imshow(
            spread_corr,
            text_auto=True,
            title="Bid-Ask Spread Correlation Across Instruments",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation
        st.markdown("""
        **Correlation Analysis Insights:**
        
        High positive correlations between instruments' bid-ask spreads indicate that liquidity conditions tend to deteriorate simultaneously across these markets. This suggests potential **systemic liquidity risk** where market stress in one instrument could quickly spread to correlated instruments.
        
        In portfolio risk management, these correlations should inform:
        - Diversification strategies (selecting instruments with lower liquidity correlation)
        - Stress testing scenarios (assuming coordinated liquidity deterioration)
        - Execution timing (avoiding simultaneous trades in highly correlated instruments)
        """)

# Liquidity Risk Management Recommendations
st.markdown('<h2 class="sub-header">Liquidity Risk Management Recommendations</h2>', unsafe_allow_html=True)

# Generate recommendations based on the data
if not filtered_df.empty and not var_df.empty and 'liquidity_cost' in var_df.columns:
    # Only proceed if the required data is available
    try:
        highest_cost_symbol = var_df.loc[var_df['liquidity_cost'].idxmax()]['symbol']
        highest_cost_value = var_df.loc[var_df['liquidity_cost'].idxmax()]['liquidity_cost']
        highest_cost_pct = (highest_cost_value / position_size) * 100
        
        # Recommendations code continues
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        st.info("Please select different data parameters and try again.")
    
    recommendations = [
        {
            "title": "Optimize Trade Execution",
            "content": f"Implement a time-weighted average price (TWAP) execution strategy for {highest_cost_symbol}, spreading trades over multiple days to minimize market impact. This could reduce liquidity costs by up to 40% compared to immediate execution.",
        },
        {
            "title": "Diversify Across Futures Contracts",
            "content": "Distribute position exposure across multiple correlated futures contracts to reduce concentration risk in any single instrument's liquidity profile. This diversification can provide alternative liquidation paths during stress scenarios.",
        },
        {
            "title": "Establish Liquidity Buffers",
            "content": f"Maintain cash reserves of at least {highest_cost_pct:.1f}% of position value to account for potential liquidation costs during stress scenarios. This buffer ensures adequate resources for managing forced liquidations without distressed selling.",
        },
        {
            "title": "Implement Early Warning Indicators",
            "content": "Monitor daily changes in bid-ask spreads and Amihud illiquidity measures, with alerts triggered by 25%+ deterioration. These early warnings can signal declining market quality before significant price movements occur.",
        },
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        for i in range(0, len(recommendations), 2):
            st.markdown(f"""
            <div class="card">
                <h4>{recommendations[i]['title']}</h4>
                <p>{recommendations[i]['content']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        for i in range(1, len(recommendations), 2):
            st.markdown(f"""
            <div class="card">
                <h4>{recommendations[i]['title']}</h4>
                <p>{recommendations[i]['content']}</p>
            </div>
            """, unsafe_allow_html=True)

# Footer with disclaimer
st.markdown("""
---
### Disclaimer

This dashboard is for demonstration purposes only. The liquidity metrics, risk calculations, and recommendations are based on simulated data and should not be the sole basis for investment decisions. In actual trading environments, additional factors must be considered and all metrics should be validated with real market data.

© 2025 Futures Liquidity Risk Analysis Tool
""")