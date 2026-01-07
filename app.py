import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# NEW IMPORT FOR FRED
from fredapi import Fred

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

@st.cache_data
def fetch_fred_data(api_key, start_date):
    """
    Fetches Macroeconomic data from FRED API.
    1. DFII10: 10-Year Real Interest Rate (Inflation Indexed)
    2. DTWEXBGS: Trade Weighted U.S. Dollar Index (Broad)
    """
    if not api_key:
        return None
        
    try:
        fred = Fred(api_key=api_key)
        
        # FRED needs dates in string YYYY-MM-DD
        start_str = start_date.strftime('%Y-%m-%d')
        
        # Fetch Data
        real_yield = fred.get_series('DFII10', observation_start=start_str)
        dxy = fred.get_series('DTWEXBGS', observation_start=start_str)
        
        # If Broad Dollar Index is empty/lagging, try standard DXY proxy (limited on FRED, using Broad as default)
        # Note: DTWEXBGS is daily but sometimes lags. 
        
        macro_df = pd.DataFrame({
            'Real_Yield_10Y': real_yield,
            'Dollar_Index': dxy
        })
        
        # Forward fill to handle weekends/holidays in macro data before merging
        macro_df = macro_df.fillna(method='ffill')
        return macro_df
        
    except Exception as e:
        st.warning(f"Could not fetch FRED data. Check API Key. Error: {e}")
        return None

# --- 2. Computation Engine ---
def compute_flow_liquidity_metrics(df, macro_df=None, flow_source_col='Volume', window=20):
    """
    Computes Returns, Flow Proxies, Liquidity Parameters, Implied Moves, and merges Macro Data.
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
    df['Lambda_Liquidity'] = df['Amihud_Raw'].rolling(window=window).mean()
    
    # D. Flow-Implied Price Move (ΔS_hat)
    # Model: ΔS_hat = λ_t * Flow_t
    df['Implied_Price_Change'] = df['Lambda_Liquidity'] * df['Flow_Raw']
    
    # Reconstruct Implied Price Level
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
    
    # F. Statistical Analysis (Z-Scores)
    lambda_mean = df['Lambda_Liquidity'].rolling(window=window*2).mean()
    lambda_std = df['Lambda_Liquidity'].rolling(window=window*2).std()
    df['Liquidity_Z'] = (df['Lambda_Liquidity'] - lambda_mean) / lambda_std
    
    df['Div_Pct'] = (df['Divergence'] / df['Close']) * 100

    # G. Merge Macro Data (If available)
    if macro_df is not None:
        # Ensure timezone compatibility (remove tz from Yahoo if present to match FRED)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Join: Left join on Yahoo dates
        # We assume df (intraday/daily) is the master index.
        # FRED data is daily. We forward fill the macro data for intraday bars.
        df = df.join(macro_df, how='left')
        
        # Forward fill macro data for intraday gaps
        df[['Real_Yield_10Y', 'Dollar_Index']] = df[['Real_Yield_10Y', 'Dollar_Index']].fillna(method='ffill')

    return df.dropna()

# --- 3. Insights Generation ---
def generate_insights(df):
    """
    Generates structured text insights based on data statistics.
    """
    insights = []
    
    last_row = df.iloc[-1]
    
    # 1. Divergence Analysis
    div_pct = last_row['Div_Pct']
    
    if div_pct > 2.0:
        insights.append({
            "title": "⚠️ Significant Premium (Overbought)",
            "text": f"The actual price is **{div_pct:.1f}% higher** than model predictions. This is a statistical 'Premium' zone, often preceding mean reversion."
        })
    elif div_pct < -2.0:
        insights.append({
            "title": "⚠️ Significant Discount (Oversold)",
            "text": f"The actual price is **{abs(div_pct):.1f}% lower** than model predictions. This is a statistical 'Discount' zone, potentially indicating a value buying opportunity."
        })
    else:
        insights.append({
            "title": "✅ Fair Value Alignment",
            "text": "The price is tracking closely with the volume flow model, indicating the current trend is well-supported by actual market activity."
        })

    # 2. Liquidity Regime Analysis
    liq_z = last_row['Liquidity_Z']
    
    if liq_z > 2.0:
        insights.append({
            "title": "🌊 Critical Fragility (3-Sigma)",
            "text": f"Liquidity is extremely thin (Z-Score: {liq_z:.1f}). Price impact is statistically abnormal. Expect slippage and sharp volatility."
        })
    elif liq_z < -1.0:
        insights.append({
            "title": "🛡️ Deep Liquidity",
            "text": "The market is absorbing volume exceptionally well. Large orders are required to move price significantly in this regime."
        })
    else:
        insights.append({
            "title": "⚖️ Normal Liquidity Conditions",
            "text": "Market depth is within standard statistical deviations. Volatility is expected to be normal."
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
    liq_z = last_row['Liquidity_Z']
    if liq_z > 1.5:
        liq_desc = "fragile liquidity conditions (High Impact)"
    elif liq_z < -0.5:
        liq_desc = "deep liquidity (Low Impact)"
    else:
        liq_desc = "stable liquidity conditions"

    # Determine Divergence
    div_pct = last_row['Div_Pct']
    
    if div_pct > 2.0:
        div_desc = "trading at a statistical premium (Potential Top)"
    elif div_pct < -2.0:
        div_desc = "trading at a statistical discount (Potential Bottom)"
    else:
        div_desc = "trading at fair value consistent with volume flows"
    
    # Macro Add-on (If data exists)
    macro_text = ""
    if 'Real_Yield_10Y' in df.columns:
        # Check simple 5 day change in Yields
        yield_delta = df['Real_Yield_10Y'].diff(5).iloc[-1]
        if not pd.isna(yield_delta):
            yield_trend = "rising" if yield_delta > 0 else "falling"
            macro_text = f" On the macro front, Real Yields are {yield_trend}, adding {'headwinds' if yield_trend == 'rising' else 'tailwinds'} to Gold."

    summary_text = (
        f"Market Executive Summary: Gold is currently trending {price_trend} supported by {flow_desc}. "
        f"The market is operating under {liq_desc}. "
        f"Critically, the price is currently {div_desc}.{macro_text} "
        "Traders should monitor the signal markers on the chart for reversals."
    )
    return summary_text

# --- 4. Visualization Helpers ---
def plot_micro_charts(df, ticker):
    # Create Subplots: Main Price, Flow, Liquidity
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            f"Price Action: Actual vs. Flow-Implied ({ticker})", 
            "Net Flow Pressure (Signed Volume)", 
            "Liquidity Regime (Z-Score & Impact)"
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
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Implied_Price'],
        fill='tonexty',
        fillcolor='rgba(128,0,128,0.05)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=1)

    # Signal Markers
    bear_div = df[df['Div_Pct'] > 2.0]
    if not bear_div.empty:
        fig.add_trace(go.Scatter(
            x=bear_div.index, y=bear_div['Close'] * 1.005,
            mode='markers',
            name='Premium (Bearish)',
            marker=dict(symbol='triangle-down', size=10, color='red')
        ), row=1, col=1)

    bull_div = df[df['Div_Pct'] < -2.0]
    if not bull_div.empty:
        fig.add_trace(go.Scatter(
            x=bull_div.index, y=bull_div['Close'] * 0.995,
            mode='markers',
            name='Discount (Bullish)',
            marker=dict(symbol='triangle-up', size=10, color='#00cc96')
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
    std_lambda = df['Lambda_Liquidity'].std()
    
    fig.add_hline(y=avg_lambda, line_dash="dash", line_color="gray", annotation_text="Avg", row=3, col=1)
    fig.add_hline(
        y=avg_lambda + (2 * std_lambda), 
        line_dash="dot", 
        line_color="#FF4B4B", 
        annotation_text="Extreme Fragility (2σ)", 
        annotation_position="top left",
        row=3, col=1
    )

    fig.update_layout(
        height=850, 
        hovermode="x unified", 
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def plot_macro_charts(df, ticker):
    """
    Plots Gold Price vs FRED Macro Data.
    """
    if 'Real_Yield_10Y' not in df.columns or df['Real_Yield_10Y'].isnull().all():
        return go.Figure().add_annotation(text="Macro Data Not Available (Check API Key or Date Range)", showarrow=False)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Gold Price (Left Axis)
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name=f"{ticker} Price", line=dict(color='gold', width=2)),
        secondary_y=False,
    )

    # 2. Real Yields (Right Axis)
    # We invert Real Yields because Gold usually moves opposite to yields.
    # High Yields = Bad for Gold. Low Yields = Good.
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['Real_Yield_10Y'], 
            name="10Y Real Yield (Inv)", 
            line=dict(color='#FF4B4B', width=2, dash='dash'),
            opacity=0.7
        ),
        secondary_y=True,
    )
    
    # 3. Dollar Index (Right Axis - Optional or Separate, combining for now)
    # If we want to show both, we might clutter, but let's add DXY as a faint line
    if 'Dollar_Index' in df.columns:
         fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['Dollar_Index'], 
                name="Dollar Index (DXY)", 
                line=dict(color='green', width=1, dash='dot'),
                opacity=0.5
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title="Macro Regime: Gold vs. Real Yields (Inverted) & DXY",
        height=600,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    # Invert the Secondary Y Axis for Yields/DXY to show correlation better
    # Standard correlation: Gold UP, Yields DOWN. 
    # By inverting Yields axis, the lines should move together if correlation holds.
    fig.update_yaxes(title_text="Gold Price", secondary_y=False, showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Yields % / DXY", secondary_y=True, showgrid=False, autorange="reversed") 

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
    
    # FRED API Key Handling
    # User puts key in secrets or environment variable
    fred_api_key = st.secrets.get("FRED_API_KEY", None)
    
    st.sidebar.info(
        """
        **How to read this:**
        1. **Price vs Implied:** Look for **Red Triangles** (Sell Signals) or **Green Triangles** (Buy Signals).
        2. **Flow:** Green bars = Net Buying Pressure.
        3. **Liquidity:** If the blue line hits the **Red Dotted Line**, the market is Extremely Fragile.
        """
    )
    
    if fred_api_key:
        st.sidebar.success("✅ FRED API Connected")
    else:
        st.sidebar.warning("⚠️ FRED API Key not found in secrets. Macro charts will be disabled.")

    # --- Main Content ---
    st.title("⚱️ Gold Flow × Liquidity × Macro")
    st.markdown(
        f"""
        This dashboard deconstructs **{asset_option}** price movements into two components:
        1. **Microstructure:** Flow Pressure & Liquidity Fragility.
        2. **Macro Regime:** Real Yields & Dollar Strength.
        """
    )

    # Fetch Data
    with st.spinner('Fetching market data...'):
        raw_df = fetch_market_data(ticker, period, interval)

    if raw_df is None:
        st.warning("No data found. Try a different ticker or timeframe.")
        return
        
    # Fetch Macro Data (if key exists)
    macro_df = None
    if fred_api_key:
        with st.spinner('Fetching Macro data (FRED)...'):
            # Calculate start date from raw_df
            start_date = raw_df.index[0]
            macro_df = fetch_fred_data(fred_api_key, start_date)

    # Compute Metrics (Pass macro_df if available)
    processed_df = compute_flow_liquidity_metrics(raw_df, macro_df=macro_df, flow_source_col='Volume', window=window)

    if processed_df.empty:
        st.error("Not enough data to calculate metrics.")
        return

    # --- Dashboard Layout ---
    
    # 1. Summary Metrics
    last_row = processed_df.iloc[-1]
    
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
        z_score = last_row['Liquidity_Z']
        liq_status = "Fragile (2σ)" if z_score > 2 else "Normal"
        st.metric("Liquidity Stress", liq_status, f"Z: {z_score:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        # Show Real Yield if available, else Model Correlation
        if 'Real_Yield_10Y' in processed_df.columns and not pd.isna(last_row['Real_Yield_10Y']):
             st.metric("10Y Real Yield", f"{last_row['Real_Yield_10Y']:.2f}%")
        else:
            corr = processed_df['Close'].corr(processed_df['Implied_Price'])
            st.metric("Model Fit (Correlation)", f"{corr:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # 2. Charts (Tabs)
    tab_micro, tab_macro = st.tabs(["🔬 Microstructure (Flow & Liquidity)", "🌍 Macro (Yields & DXY)"])
    
    with tab_micro:
        chart_fig = plot_micro_charts(processed_df, ticker)
        st.plotly_chart(chart_fig, use_container_width=True)
        
    with tab_macro:
        if fred_api_key:
            macro_fig = plot_macro_charts(processed_df, ticker)
            st.plotly_chart(macro_fig, use_container_width=True)
        else:
            st.info("Add FRED_API_KEY to secrets to unlock Macro analysis.")

    # 3. Automated Analysis Section
    st.subheader("📝 Automated Chart Analysis")
    
    # Generate and display Executive Summary
    summary_text = generate_executive_summary(processed_df)
    st.markdown(f"<div class='summary-box'>{summary_text}</div>", unsafe_allow_html=True)

    # Generate insights based on data
    insights = generate_insights(processed_df)
    
    # Display insights in columns
    icol1, icol2, icol3 = st.columns(3)
    
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
        * **Fragility (Z-Score):** Measures how many standard deviations current liquidity is from the mean.
        * **Macro:** Real Yields (FRED: DFII10) and Dollar Index (FRED: DTWEXBGS).
        """)
        st.dataframe(processed_df.tail(50).sort_index(ascending=False))

if __name__ == "__main__":
    main()
