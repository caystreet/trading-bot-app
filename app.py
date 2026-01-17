import streamlit as st
import pandas as pd
import numpy as np
from tiingo import TiingoClient
from fredapi import Fred
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Bot Tester",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 Trading Bot Tester")
st.markdown("Machine Learning-powered trading signal generator for stocks and crypto")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# API Keys
with st.sidebar.expander("🔑 API Keys", expanded=True):
    tiingo_key = st.text_input(
        "Tiingo API Key",
        value="e30bc8b0b855970d930743b116e517e36b72cc8f",
        type="password"
    )
    fred_key = st.text_input(
        "FRED API Key",
        value="1c1c04541bfefa93e9cb0db265da9c19",
        type="password"
    )

# Trading Parameters
with st.sidebar.expander("📊 Trading Parameters", expanded=True):
    target_asset = st.selectbox(
        "Target Asset",
        ["BTCUSD", "ETHUSD", "SOLUSD", "SPY", "GLD", "QQQ", "IWM"],
        index=0
    )

    market_context = st.multiselect(
        "Market Context",
        ["MSTR", "SPY", "GLD", "QQQ", "IWM", "IEA", "TLT", "VIX"],
        default=["MSTR", "SPY", "GLD", "QQQ"]
    )

    macro_context = st.multiselect(
        "Macro Context (FRED)",
        ["FEDFUNDS", "CPIAUCSL", "PCE", "PCEPI", "DGS10", "UNRATE", "M2SL"],
        default=["FEDFUNDS", "CPIAUCSL", "PCE"]
    )

# Model selection
with st.sidebar.expander("🧠 Model Selection", expanded=True):
    use_rf = st.checkbox("Random Forest", value=True)
    use_knn = st.checkbox("K-Nearest Neighbors", value=True)

    if use_rf:
        n_estimators = st.slider("RF: Number of Trees", 50, 200, 100)
        max_depth = st.slider("RF: Max Depth", 5, 20, 10)

    if use_knn:
        n_neighbors = st.slider("KNN: Number of Neighbors", 3, 15, 5)

# --- DATA FUNCTIONS ---
@st.cache_data(ttl=3600)
def get_consolidated_data(tiingo_key, fred_key, target_asset, market_context, macro_context):
    """Fetch and consolidate market and macro data"""
    try:
        client = TiingoClient({'api_key': tiingo_key})
        fred = Fred(api_key=fred_key)

        # Fetch all market symbols
        all_market = [target_asset] + market_context
        df_market = client.get_dataframe(all_market, metric_name='close', startDate='2018-01-01')

        # Fetch economic data from FRED
        df_econ = pd.DataFrame({s: fred.get_series(s) for s in macro_context})

        # Merge and fill
        df_combined = df_market.join(df_econ)
        df_combined = df_combined.ffill().bfill()

        # Fill macro data NaNs
        for col in macro_context:
            if col in df_combined.columns:
                df_combined[col] = df_combined[col].fillna(0)

        return df_combined.dropna(subset=[target_asset])
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def build_lab_features(df, target):
    """Build features and labels for training"""
    # Momentum
    df['Mom_30'] = (df[target] - df[target].shift(30)) / df[target].shift(30)
    df['Mom_14'] = (df[target] - df[target].shift(14)) / df[target].shift(14)

    # Smoothing
    df['Smoothed_30'] = df['Mom_30'].rolling(window=30).mean()
    df['Smoothed_10'] = df['Mom_30'].rolling(window=10).mean()

    # Define Label: 1 if 8.5% TP hit within 40 candles
    df['Signal'] = (df[target].shift(-40) > df[target] * 1.085).astype(int)

    return df.dropna()

def calculate_metrics(results_series):
    """Calculate trading metrics"""
    total_signals = len(results_series)
    if total_signals == 0:
        return {"total": 0, "win_rate": 0}

    win_rate = round((results_series == 'TP').mean() * 100, 2)
    return {"total": total_signals, "win_rate": win_rate}

def run_backtest(df, preds, target):
    """Run backtest and return results"""
    signals_indices = df.index[preds == 1]
    trades = []
    trades_index = []

    for idx in signals_indices:
        entry = df.loc[idx, target]
        window = df.loc[idx:].iloc[1:41][target]
        result = 'Expired'

        if not window.empty:
            for price in window:
                if price <= entry * 0.97:  # 3% Stop Loss
                    result = 'SL'
                    break
                elif price >= entry * 1.085:  # 8.5% Take Profit
                    result = 'TP'
                    break

        trades.append(result)
        trades_index.append(idx)

    results_series = pd.Series(trades, index=trades_index)
    df_end_date = df.index[-1]

    # Calculate metrics for different periods
    overall = calculate_metrics(results_series)

    ninety_days_ago = df_end_date - pd.Timedelta(days=90)
    results_90d = results_series[results_series.index >= ninety_days_ago]
    last_90d = calculate_metrics(results_90d)

    sixty_days_ago = df_end_date - pd.Timedelta(days=60)
    results_60d = results_series[results_series.index >= sixty_days_ago]
    last_60d = calculate_metrics(results_60d)

    thirty_days_ago = df_end_date - pd.Timedelta(days=30)
    results_30d = results_series[results_series.index >= thirty_days_ago]
    last_30d = calculate_metrics(results_30d)

    return {
        "overall": overall,
        "90d": last_90d,
        "60d": last_60d,
        "30d": last_30d
    }

def get_live_prediction(model, df, target):
    """Get live prediction from the model"""
    # Apply feature engineering
    df['Mom_30'] = (df[target] - df[target].shift(30)) / df[target].shift(30)
    df['Mom_14'] = (df[target] - df[target].shift(14)) / df[target].shift(14)
    df['Smoothed_30'] = df['Mom_30'].rolling(window=30).mean()
    df['Smoothed_10'] = df['Mom_30'].rolling(window=10).mean()

    df_features = df.dropna()
    last_row = df_features.iloc[[-1]]

    current_price = last_row[target].values[0]
    current_date = last_row.index[0].strftime('%Y-%m-%d')

    prediction = model.predict(last_row)

    try:
        probability = model.predict_proba(last_row)[0][1]
    except:
        probability = 0.5

    return {
        "prediction": prediction[0],
        "probability": probability,
        "price": current_price,
        "date": current_date
    }

# --- MAIN APP ---
if st.sidebar.button("🚀 Run Analysis", type="primary"):
    if not tiingo_key or not fred_key:
        st.error("Please provide both API keys")
    else:
        with st.spinner("Fetching data..."):
            raw_data = get_consolidated_data(
                tiingo_key, fred_key, target_asset,
                market_context, macro_context
            )

        if raw_data is not None:
            with st.spinner("Building features..."):
                processed_data = build_lab_features(raw_data.copy(), target_asset)

                X = processed_data.drop(columns=['Signal'])
                y = processed_data['Signal']

            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📊 Live Signal", "📈 Backtest Results", "📉 Data Visualization"])

            with tab1:
                st.header("🤖 Live Trading Signals")

                col1, col2 = st.columns(2)

                # Train and predict with Random Forest
                if use_rf:
                    with col1:
                        with st.spinner("Training Random Forest..."):
                            rf_model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                criterion='entropy',
                                max_depth=max_depth
                            )
                            rf_model.fit(X, y)

                            rf_pred = get_live_prediction(rf_model, raw_data.copy(), target_asset)

                        st.subheader("🌲 Random Forest Model")
                        st.metric("Current Price", f"${rf_pred['price']:,.2f}")
                        st.metric("Date", rf_pred['date'])

                        if rf_pred['prediction'] == 1:
                            st.success("✅ BUY SIGNAL DETECTED")
                            st.metric("Confidence", f"{rf_pred['probability']*100:.2f}%")
                        else:
                            st.warning("⛔ NO SIGNAL (WAIT)")
                            st.metric("Wait Confidence", f"{(1-rf_pred['probability'])*100:.2f}%")

                # Train and predict with KNN
                if use_knn:
                    with col2:
                        with st.spinner("Training K-Nearest Neighbors..."):
                            knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
                            knn_model.fit(X, y)

                            knn_pred = get_live_prediction(knn_model, raw_data.copy(), target_asset)

                        st.subheader("🔵 K-Nearest Neighbors Model")
                        st.metric("Current Price", f"${knn_pred['price']:,.2f}")
                        st.metric("Date", knn_pred['date'])

                        if knn_pred['prediction'] == 1:
                            st.success("✅ BUY SIGNAL DETECTED")
                            st.metric("Confidence", f"{knn_pred['probability']*100:.2f}%")
                        else:
                            st.warning("⛔ NO SIGNAL (WAIT)")
                            st.metric("Wait Confidence", f"{(1-knn_pred['probability'])*100:.2f}%")

            with tab2:
                st.header("📈 Historical Backtest Results")

                # Run backtests
                if use_rf:
                    with st.spinner("Running Random Forest backtest..."):
                        rf_results = run_backtest(processed_data, rf_model.predict(X), target_asset)

                    st.subheader("🌲 Random Forest Backtest")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Overall Signals", rf_results['overall']['total'])
                        st.metric("Overall Win Rate", f"{rf_results['overall']['win_rate']}%")

                    with col2:
                        st.metric("90D Signals", rf_results['90d']['total'])
                        st.metric("90D Win Rate", f"{rf_results['90d']['win_rate']}%")

                    with col3:
                        st.metric("60D Signals", rf_results['60d']['total'])
                        st.metric("60D Win Rate", f"{rf_results['60d']['win_rate']}%")

                    with col4:
                        st.metric("30D Signals", rf_results['30d']['total'])
                        st.metric("30D Win Rate", f"{rf_results['30d']['win_rate']}%")

                    st.divider()

                if use_knn:
                    with st.spinner("Running KNN backtest..."):
                        knn_results = run_backtest(processed_data, knn_model.predict(X), target_asset)

                    st.subheader("🔵 K-Nearest Neighbors Backtest")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Overall Signals", knn_results['overall']['total'])
                        st.metric("Overall Win Rate", f"{knn_results['overall']['win_rate']}%")

                    with col2:
                        st.metric("90D Signals", knn_results['90d']['total'])
                        st.metric("90D Win Rate", f"{knn_results['90d']['win_rate']}%")

                    with col3:
                        st.metric("60D Signals", knn_results['60d']['total'])
                        st.metric("60D Win Rate", f"{knn_results['60d']['win_rate']}%")

                    with col4:
                        st.metric("30D Signals", knn_results['30d']['total'])
                        st.metric("30D Win Rate", f"{knn_results['30d']['win_rate']}%")

            with tab3:
                st.header("📉 Price Action Visualization")

                # Plot last 30 days
                fig, ax = plt.subplots(figsize=(12, 6))
                recent_data = raw_data[target_asset].tail(30)
                ax.plot(recent_data.index, recent_data.values, color='skyblue', linewidth=2, label='Price')

                # Highlight last point based on prediction
                if use_rf:
                    color = 'green' if rf_pred['prediction'] == 1 else 'red'
                    ax.scatter(recent_data.index[-1], recent_data.values[-1],
                             color=color, s=100, zorder=5, label='Current Position')

                ax.set_title(f"{target_asset} - 30 Day Price Action")
                ax.set_xlabel('Date')
                ax.set_ylabel('Price (USD)')
                ax.grid(True, alpha=0.3)
                ax.legend()

                st.pyplot(fig)

                # Show raw data
                with st.expander("📊 View Raw Data"):
                    st.dataframe(raw_data.tail(50))

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit")
st.sidebar.markdown("Data: Tiingo & FRED")
