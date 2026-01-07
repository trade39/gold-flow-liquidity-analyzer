import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Configuration ---
st.set_page_config(
    page_title="Gold Flow & Liquidity Analyzer",
    page_icon="⚱️",
    layout="wide"
)

# --- Custom CSS for aesthetic tweaks ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stPlotlyChart {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Data Ingestion (Cached) ---
@st.cache_data
def fetch_market_data(ticker_symbol, period, interval):
    """
    Fetches historical market data from Yahoo Finance.
    """
    try:
        data = yf.download(ticker_symbol, period=period, interval=interval, progress=False)
        if data.empty:
            return None
        # Clean up MultiIndex columns if present (common in new yfinance versions)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# --- 2. Computation Engine ---
def compute_flow_liquidity_metrics(df, flow_source_col='Volume', window=20):
    """
    Computes Returns, Flow Proxies, Liquidity Parameters, and Implied Moves.
    """
    df = df.copy()
    
    # A. Returns (ΔS)
    df['Return'] = df['Close'].pct_change()
    df['Price_Change'] = df['Close'].diff()
    
    # B. Flow Proxy (ΔQ)
    # Logic: Signed Volume. If Close > Open, we assume 'Buying Pressure', else 'Selling Pressure'.
    # Note: A more advanced version would use tick data, but OHLCv allows this heuristic.
    df['Direction'] = np.where(df['Close'] >= df['Open'], 1, -1)
    
    # Normalized Flow: We use Volume * Direction. 
    # To make it comparable across time, we can normalize by a moving average of volume if needed,
    # but raw Signed Volume preserves the magnitude of "surges".
    df['Flow_Raw'] = df[flow_source_col] * df['Direction']
    
    # C. Liquidity / Impact Parameter (λ)
    # We use the Amihud Illiquidity Ratio proxy: |Return| / (Price * Volume)
    # However, to get a cleaner "Impact Slope", we can use: |Price Change| / Volume
    # λ = How much price moves ($) per unit of volume.
    # We smooth this with a rolling window to estimate the current "Regime".
    
    # Avoid division by zero
    safe_volume = df[flow_source_col].replace(0, np.nan)
    
    # Raw Amihud (Daily Impact)
    df['Amihud_Raw'] = df['Price_Change'].abs() / safe_volume
    
    # Smoothed Liquidity Parameter (λ_t)
    # High λ = Fragile Liquidity (Price moves easily on low volume)
    # Low λ = Deep Liquidity (Price absorbs volume well)
    df['Lambda_Liquidity'] = df['Amihud_Raw'].rolling(window=window).mean()
    
    # D. Flow-Implied Price Move (ΔS_hat)
    # Model: ΔS_hat = λ_t * Flow_t
    # We use the rolling Lambda to estimate what the price move "should" have been
    # given the volume flow and the current liquidity regime.
    df['Implied_Price_Change'] = df['Lambda_Liquidity'] * df['Flow_Raw']
    
    # Reconstruct Implied Price Level (Indexed to start of window)
    # We start the index at the first valid data point
    valid_idx = df['Lambda_Liquidity'].first_valid_index()
    
    if valid_idx:
        start_price = df.loc[valid_idx, 'Close']
        # Cumulative sum of implied changes added to the starting price
        df['Implied_Price'] = start_price + df['Implied_Price_Change'].cumsum()
        
        # Align the actual price for the chart comparison (starting from same point)
        df['Actual_Price_Indexed'] = df['Close']
    else:
        df['Implied_Price'] = np.nan
        df['Actual_Price_Indexed'] = np.nan

    # E. Divergence (Residual)
    # Positive = Price is higher than flows would justify (Premium/Markup)
    # Negative = Price is lower than flows would justify (Discount/Markdown)
    df['Divergence'] = df['Close'] - df['Implied_Price']

    return df.dropna()

# --- 3. Visualization Helpers ---
def plot_charts(df, ticker):
    # Create Subplots: Main Price, Flow, Liquidity
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            f"Price Action: Actual vs. Flow-Implied ({ticker})", 
            "Net Flow Pressure (Signed Volume)", 
            "Liquidity Regime (λ - Price Impact Parameter)"
        )
    )

    # --- Row 1: Price vs Implied ---
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], 
        name="Actual Price", 
        line=dict(color='gold', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Implied_Price'], 
        name="Flow-Implied Price", 
        line=dict(color='purple', width=2, dash='dot')
    ), row=1, col=1)
    
    # Highlight Divergence Zones
    # We can fill area between the two lines
    # Usually requires split traces for different colors, simplifying here:
    # fig.add_trace(go.Scatter(
    #    x=df.index, y=df['Implied_Price'], fill='tonexty', fillcolor='rgba(128,0,128,0.1)', line=dict(width=0), showlegend=False
    # ), row=1, col=1)

    # --- Row 2: Flow Pressure ---
    # Color bars based on direction
    colors = ['#00cc96' if v > 0 else '#EF553B' for v in df['Flow_Raw']]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Flow_Raw'],
        name="Net Flow",
        marker_color=colors
    ), row=2, col=1)

    # --- Row 3: Liquidity Regime (Lambda) ---
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Lambda_Liquidity'],
        name="Liquidity Impact (λ)",
        line=dict(color='#19d3f3', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(25, 211, 243, 0.1)'
    ), row=3, col=1)

    # Annotations for Liquidity
    # Add a horizontal line for average liquidity
    avg_lambda = df['Lambda_Liquidity'].mean()
    fig.add_hline(y=avg_lambda, line_dash="dash", line_color="gray", annotation_text="Avg Fragility", row=3, col=1)

    fig.update_layout(
        height=800, 
        hovermode="x unified", 
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

# --- Main Application Logic ---
def main():
    # --- Sidebar Controls ---
    st.sidebar.title("⚙️ Configuration")
    
    # Data Selection
    asset_option = st.sidebar.selectbox(
        "Select Asset", 
        options=["GC=F (Gold Futures)", "GLD (SPDR Gold Shares)", "XAUUSD=X (Spot Gold)"],
        index=0
    )
    ticker_map = {
        "GC=F (Gold Futures)": "GC=F", 
        "GLD (SPDR Gold Shares)": "GLD",
        "XAUUSD=X (Spot Gold)": "XAUUSD=X"
    }
    ticker = ticker_map[asset_option]

    # Timeframe
    period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    interval = st.sidebar.selectbox("Interval", ["1d", "1h", "90m"], index=0)

    # Model Parameters
    st.sidebar.markdown("### 🧮 Model Parameters")
    window = st.sidebar.slider(
        "Rolling Window (Days/Bars)", 
        min_value=5, max_value=50, value=20, 
        help="Window size to estimate the Liquidity Parameter (λ). A smaller window makes the model more sensitive to recent volatility."
    )
    
    st.sidebar.info(
        """
        **How to read this:**
        1. **Price vs Implied:** When 'Actual' deviates from 'Implied', price is moving without volume support (or against it).
        2. **Flow:** Green bars = Net Buying Pressure, Red = Net Selling.
        3. **Liquidity (λ):** Higher values = **Fragile**. Price moves easily on low volume. Low values = **Deep**. Hard to move price.
        """
    )

    # --- Main Content ---
    st.title("⚱️ Gold Price Action: Flow × Liquidity Framework")
    st.markdown(
        f"""
        This dashboard deconstructs **{asset_option}** price movements into two components:
        1. **Flow ($\Delta Q$):** The pressure from buying/selling volume.
        2. **Liquidity ($\lambda$):** The market's capacity to absorb that flow (Price Impact).
        
        $$ \text{{Implied Move}} \approx \text{{Flow}} \times \text{{Liquidity Parameter}} $$
        """
    )

    # Fetch Data
    with st.spinner('Fetching market data...'):
        raw_df = fetch_market_data(ticker, period, interval)

    if raw_df is None:
        st.warning("No data found. Try a different ticker or timeframe.")
        return

    # Compute Metrics
    processed_df = compute_flow_liquidity_metrics(raw_df, flow_source_col='Volume', window=window)

    if processed_df.empty:
        st.error("Not enough data to calculate metrics. Try increasing the period or decreasing the rolling window.")
        return

    # --- Dashboard Layout ---
    
    # 1. Summary Statistics
    last_row = processed_df.iloc[-1]
    prev_row = processed_df.iloc[-2]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Price", 
            f"${last_row['Close']:,.2f}",
            f"{last_row['Return']:.2%}"
        )
    
    with col2:
        flow_delta = last_row['Flow_Raw']
        st.metric(
            "Net Flow Pressure", 
            f"{flow_delta:,.0f} Vol", 
            delta_color="normal" if flow_delta > 0 else "inverse"
        )
        
    with col3:
        # Liquidity State
        curr_liq = last_row['Lambda_Liquidity']
        avg_liq = processed_df['Lambda_Liquidity'].mean()
        liq_status = "Fragile (High Impact)" if curr_liq > avg_liq else "Deep (Low Impact)"
        st.metric(
            "Liquidity Regime", 
            liq_status,
            f"λ: {curr_liq:.2e}"
        )

    with col4:
        # Correlation
        corr = processed_df['Close'].corr(processed_df['Implied_Price'])
        st.metric(
            "Model Fit (Correlation)",
            f"{corr:.2f}",
            help="Correlation between Actual Price and Flow-Implied Price. High correlation means flows explain price well."
        )

    st.markdown("---")

    # 2. Charts
    chart_fig = plot_charts(processed_df, ticker)
    st.plotly_chart(chart_fig, use_container_width=True)

    # 3. Data Table (Expandable)
    with st.expander("🔍 View Detailed Data Logic"):
        st.markdown("""
        **Methodology:**
        * **Flow Proxy ($\Delta Q$):** Calculated as `Volume * Direction`, where Direction is +1 if Close > Open, else -1.
        * **Liquidity Parameter ($\lambda$):** Based on the **Amihud Illiquidity Ratio**. It measures the absolute price change per unit of volume over the rolling window.
            * Formula: $\lambda_t = \text{Avg}(\frac{|\Delta Price|}{\text{Volume}})$ over $N$ periods.
        * **Implied Price:** Constructed by applying the historical impact parameter ($\lambda$) to the current flows.
        """)
        st.dataframe(processed_df[['Close', 'Volume', 'Flow_Raw', 'Lambda_Liquidity', 'Implied_Price', 'Divergence']].sort_index(ascending=False).head(50))

if __name__ == "__main__":
    main()
