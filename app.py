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

# --- Custom CSS for aesthetic tweaks (Updated for Dark Mode) ---
st.markdown("""
<style>
    /* Using Streamlit theme variables for dark/light mode compatibility */
    .metric-card {
        background-color: var(--secondary-background-color);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        border: 1px solid rgba(128, 128, 128, 0.1);
    }
    .stPlotlyChart {
        /* Keep charts on a slightly lighter background in dark mode for contrast */
        background-color: var(--secondary-background-color);
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        border: 1px solid rgba(128, 128, 128, 0.1);
    }
    .insight-box {
        padding: 15px;
        border-left: 5px solid #FF4B4B;
        /* Use theme-aware background color */
        background-color: var(--secondary-background-color);
        border-radius: 5px;
        margin-bottom: 10px;
        /* Ensure text adapts to theme */
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.1);
        height: 100%; /* accurate alignment */
    }
    .summary-box {
        padding: 20px;
        border-left: 5px solid #00cc96; /* Green accent for summary */
        background-color: var(--secondary-background-color);
        border-radius: 5px;
        margin-bottom: 25px;
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.1);
    }
    .insight-header {
        font-weight: bold;
        /* Use theme-aware text color */
        color: var(--text-color);
        font-size: 1.1em;
        margin-bottom: 8px;
    }
    /* Ensure paragraph text inside insights adapts */
    .insight-box p, .summary-box p {
        color: var(--text-color);
        opacity: 0.9; /* Slightly softer contrast for body text */
        margin-bottom: 0;
        line-height: 1.5;
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
    df['Direction'] = np.where(df['Close'] >= df['Open'], 1, -1)
    
    # Normalized Flow: We use Volume * Direction. 
    df['Flow_Raw'] = df[flow_source_col] * df['Direction']
    
    # C. Liquidity / Impact Parameter (λ)
    # We use the Amihud Illiquidity Ratio proxy: |Price Change| / Volume
    safe_volume = df[flow_source_col].replace(0, np.nan)
    
    # Raw Amihud (Daily Impact)
    df['Amihud_Raw'] = df['Price_Change'].abs() / safe_volume
    
    # Smoothed Liquidity Parameter (λ_t)
    # High λ = Fragile Liquidity (Price moves easily on low volume)
    # Low λ = Deep Liquidity (Price absorbs volume well)
    df['Lambda_Liquidity'] = df['Amihud_Raw'].rolling(window=window).mean()
    
    # D. Flow-Implied Price Move (ΔS_hat)
    # Model: ΔS_hat = λ_t * Flow_t
    df['Implied_Price_Change'] = df['Lambda_Liquidity'] * df['Flow_Raw']
    
    # Reconstruct Implied Price Level (Indexed to start of window)
    valid_idx = df['Lambda_Liquidity'].first_valid_index()
    
    if valid_idx:
        start_price = df.loc[valid_idx, 'Close']
        df['Implied_Price'] = start_price + df['Implied_Price_Change'].cumsum()
        df['Actual_Price_Indexed'] = df['Close']
    else:
        df['Implied_Price'] = np.nan
        df['Actual_Price_Indexed'] = np.nan

    # E. Divergence (Residual)
    df['Divergence'] = df['Close'] - df['Implied_Price']

    return df.dropna()

# --- 3. Insights Generation ---
def generate_insights(df):
    """
    Generates structured text insights based on data statistics.
    """
    insights = []
    
    last_row = df.iloc[-1]
    avg_lambda = df['Lambda_Liquidity'].mean()
    curr_lambda = last_row['Lambda_Liquidity']
    
    # 1. Divergence Analysis
    div_val = last_row['Divergence']
    close_price = last_row['Close']
    div_pct = (div_val / close_price) * 100
    
    if div_pct > 2.0:
        insights.append({
            "title": "⚠️ Significant Premium (Overbought)",
            "text": f"The actual price is **{div_pct:.1f}% higher** than what the flow model predicts. This suggests price is rising without significant volume support, often a sign of exhaustion or speculative markup."
        })
    elif div_pct < -2.0:
        insights.append({
            "title": "⚠️ Significant Discount (Oversold)",
            "text": f"The actual price is **{abs(div_pct):.1f}% lower** than the flow model predicts. Price has dropped aggressively despite volume flows that would suggest a higher value, potentially indicating a capitulation bottom."
        })
    else:
        insights.append({
            "title": "✅ Fair Value Alignment",
            "text": "The price is tracking closely with the volume flow model, indicating the current trend is well-supported by actual market activity."
        })

    # 2. Liquidity Regime Analysis
    lambda_ratio = curr_lambda / avg_lambda
    if lambda_ratio > 1.2:
        insights.append({
            "title": "🌊 Fragile Liquidity Environment",
            "text": f"Current liquidity impact ($\lambda$) is **{lambda_ratio:.1f}x higher** than average. The order book is thin; expect higher volatility and slippage. Small orders are moving price significantly."
        })
    elif lambda_ratio < 0.8:
        insights.append({
            "title": "🛡️ Deep Liquidity Environment",
            "text": "Current liquidity impact is lower than average. The market is absorbing volume well. It will take significant buying/selling pressure to create large price moves."
        })
    else:
        insights.append({
            "title": "⚖️ Normal Liquidity Conditions",
            "text": "Market depth is near historical averages. Volatility is expected to be standard."
        })

    # 3. Flow Trend
    recent_flows = df['Flow_Raw'].tail(5).sum()
    if recent_flows > 0:
        insights.append({
            "title": "📈 Net Buying Pressure",
            "text": "Over the last 5 periods, cumulative volume flow has been positive, supporting bullish price action."
        })
    else:
        insights.append({
            "title": "📉 Net Selling Pressure",
            "text": "Over the last 5 periods, cumulative volume flow has been negative, exerting downward pressure on price."
        })

    return insights

def generate_executive_summary(df):
    """
    Generates a cohesive executive summary paragraph.
    """
    last_row = df.iloc[-1]
    
    # Determine Trend
    price_trend = "upward" if last_row['Close'] > df['Close'].iloc[-5] else "downward"
    
    # Determine Flow Context
    recent_net_flow = df['Flow_Raw'].tail(5).sum()
    flow_desc = "strong net buying" if recent_net_flow > 0 else "strong net selling"
    
    # Determine Liquidity Context
    avg_lambda = df['Lambda_Liquidity'].mean()
    curr_lambda = last_row['Lambda_Liquidity']
    if curr_lambda > avg_lambda * 1.2:
        liq_desc = "fragile liquidity conditions, meaning volatility is elevated"
    elif curr_lambda < avg_lambda * 0.8:
        liq_desc = "deep liquidity, meaning the market is absorbing orders efficiently"
    else:
        liq_desc = "stable liquidity conditions"

    # Determine Divergence
    div_val = last_row['Divergence']
    close_price = last_row['Close']
    div_pct = (div_val / close_price) * 100
    
    if div_pct > 2.0:
        div_desc = "trading at a premium relative to flows, warning of potential exhaustion"
    elif div_pct < -2.0:
        div_desc = "trading at a discount relative to flows, suggesting a potential bounce"
    else:
        div_desc = "trading at fair value consistent with volume flows"

    summary_text = (
        f"**Market Executive Summary:** Gold is currently trending **{price_trend}** supported by **{flow_desc}**. "
        f"The market is operating under **{liq_desc}**. "
        f"Critically, the price is currently **{div_desc}**. "
        "Traders should monitor the flow pressure gauge for reversals in this regime."
    )
    return summary_text

# --- 4. Visualization Helpers ---
def plot_charts(df, ticker):
    # Create Subplots: Main Price, Flow, Liquidity
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
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
    
    # Highlight Divergence Zones (Fill)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Implied_Price'],
        fill='tonexty', # Fills to the trace before it (Actual Price)
        fillcolor='rgba(128,0,128,0.05)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=1)

    # --- Row 2: Flow Pressure ---
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
    avg_lambda = df['Lambda_Liquidity'].mean()
    fig.add_hline(y=avg_lambda, line_dash="dash", line_color="gray", annotation_text="Avg Fragility", row=3, col=1)

    # Update layout for theme compatibility
    fig.update_layout(
        height=850, 
        hovermode="x unified", 
        # template="plotly_white", # Removed to allow Plotly to adapt to Streamlit theme automatically
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)', # Transparent background for plots
        paper_bgcolor='rgba(0,0,0,0)', # Transparent background for the whole figure
    )
    # Ensure gridlines and text look okay in both modes if template is removed
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

# --- Main Application Logic ---
def main():
    # --- Sidebar Controls ---
    st.sidebar.title("⚙️ Configuration")
    
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

    period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    interval = st.sidebar.selectbox("Interval", ["1d", "1h", "90m"], index=0)

    st.sidebar.markdown("### 🧮 Model Parameters")
    window = st.sidebar.slider(
        "Rolling Window (Days/Bars)", 
        min_value=5, max_value=50, value=20, 
        help="Window size to estimate the Liquidity Parameter (λ)."
    )
    
    st.sidebar.info(
        """
        **How to read this:**
        1. **Price vs Implied:** When 'Actual' deviates from 'Implied', price is moving without volume support.
        2. **Flow:** Green bars = Net Buying Pressure, Red = Net Selling.
        3. **Liquidity (λ):** Higher values = **Fragile**. Price moves easily on low volume.
        """
    )

    # --- Main Content ---
    st.title("⚱️ Gold Price Action: Flow × Liquidity Framework")
    st.markdown(
        f"""
        This dashboard deconstructs **{asset_option}** price movements into two components:
        1. **Flow ($\Delta Q$):** The pressure from buying/selling volume.
        2. **Liquidity ($\lambda$):** The market's capacity to absorb that flow.
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
        st.error("Not enough data to calculate metrics.")
        return

    # --- Dashboard Layout ---
    
    # 1. Summary Metrics
    last_row = processed_df.iloc[-1]
    
    # Use custom CSS class for metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Current Price", f"${last_row['Close']:,.2f}", f"{last_row['Return']:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        flow_delta = last_row['Flow_Raw']
        st.metric("Net Flow Pressure", f"{flow_delta:,.0f} Vol", delta_color="normal" if flow_delta > 0 else "inverse")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        curr_liq = last_row['Lambda_Liquidity']
        avg_liq = processed_df['Lambda_Liquidity'].mean()
        liq_status = "Fragile (High Impact)" if curr_liq > avg_liq else "Deep (Low Impact)"
        st.metric("Liquidity Regime", liq_status, f"λ: {curr_liq:.2e}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        corr = processed_df['Close'].corr(processed_df['Implied_Price'])
        st.metric("Model Fit (Correlation)", f"{corr:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # 2. Charts
    chart_fig = plot_charts(processed_df, ticker)
    st.plotly_chart(chart_fig, use_container_width=True)

    # 3. Automated Analysis Section
    st.subheader("📝 Automated Chart Analysis")
    
    # Generate and display Executive Summary (New Feature)
    summary_text = generate_executive_summary(processed_df)
    st.markdown(f"<div class='summary-box'>{summary_text}</div>", unsafe_allow_html=True)

    # Generate insights based on data
    insights = generate_insights(processed_df)
    
    # Display insights in columns for readability
    icol1, icol2, icol3 = st.columns(3)
    
    # Map the three generated insights to columns
    if len(insights) >= 3:
        with icol1:
            st.markdown(f"<div class='insight-box'><div class='insight-header'>{insights[0]['title']}</div><p>{insights[0]['text']}</p></div>", unsafe_allow_html=True)
        with icol2:
            st.markdown(f"<div class='insight-box'><div class='insight-header'>{insights[1]['title']}</div><p>{insights[1]['text']}</p></div>", unsafe_allow_html=True)
        with icol3:
            st.markdown(f"<div class='insight-box'><div class='insight-header'>{insights[2]['title']}</div><p>{insights[2]['text']}</p></div>", unsafe_allow_html=True)

    # 4. Data Logic Expander
    with st.expander("🔍 View Detailed Data Logic"):
        st.markdown("""
        **Methodology:**
        * **Flow Proxy ($\Delta Q$):** Calculated as `Volume * Direction`.
        * **Liquidity Parameter ($\lambda$):** Based on the **Amihud Illiquidity Ratio**.
        * **Implied Price:** Constructed by applying historical impact ($\lambda$) to current flows.
        """)
        st.dataframe(processed_df.tail(50).sort_index(ascending=False))

if __name__ == "__main__":
    main()
