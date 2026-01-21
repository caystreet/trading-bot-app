import streamlit as st
import pandas as pd
import numpy as np
from tiingo import TiingoClient
from fredapi import Fred
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import warnings
import json
import pickle
from datetime import datetime
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

    st.markdown("**Date Range**")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=pd.to_datetime("2018-01-01"),
            min_value=pd.to_datetime("2015-01-01"),
            max_value=pd.to_datetime("today")
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=pd.to_datetime("today"),
            min_value=pd.to_datetime("2015-01-01"),
            max_value=pd.to_datetime("today")
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

# Technical Indicators
with st.sidebar.expander("📈 Technical Indicators", expanded=True):
    st.markdown("**Simple Moving Averages**")
    use_sma_10 = st.checkbox("10-day SMA", value=True)
    use_sma_50 = st.checkbox("50-day SMA", value=True)
    use_sma_100 = st.checkbox("100-day SMA", value=False)
    use_sma_200 = st.checkbox("200-day SMA", value=False)

    st.markdown("**Momentum Indicators**")
    use_mom_14 = st.checkbox("14-day Momentum", value=True)
    use_mom_30 = st.checkbox("30-day Momentum", value=True)
    use_mom_60 = st.checkbox("60-day Momentum", value=False)

    st.markdown("**Smoothed Momentum**")
    use_smoothed_10 = st.checkbox("10-day Smoothed Mom", value=True)
    use_smoothed_30 = st.checkbox("30-day Smoothed Mom", value=True)

    st.markdown("**Volume Indicators**")
    use_volume = st.checkbox("Volume", value=True)
    use_volume_sma = st.checkbox("Volume SMA (20-day)", value=True)
    use_volume_ratio = st.checkbox("Volume Ratio (vs 20d avg)", value=True)

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
def get_consolidated_data(tiingo_key, fred_key, target_asset, market_context, macro_context, start_date, end_date):
    """Fetch and consolidate market and macro data"""
    try:
        client = TiingoClient({'api_key': tiingo_key})
        fred = Fred(api_key=fred_key)

        # Fetch all market symbols (close prices)
        all_market = [target_asset] + market_context
        df_market = client.get_dataframe(
            all_market,
            metric_name='close',
            startDate=start_date.strftime('%Y-%m-%d'),
            endDate=end_date.strftime('%Y-%m-%d')
        )

        # Fetch volume data for target asset
        try:
            df_volume = client.get_dataframe(
                [target_asset],
                metric_name='volume',
                startDate=start_date.strftime('%Y-%m-%d'),
                endDate=end_date.strftime('%Y-%m-%d')
            )
            df_volume.columns = [f"{target_asset}_volume"]
            df_market = df_market.join(df_volume)
        except:
            # If volume data not available, create a dummy column
            df_market[f"{target_asset}_volume"] = 0

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

def build_lab_features(df, target, indicator_config):
    """Build features and labels for training"""
    # Simple Moving Averages
    if indicator_config.get('use_sma_10'):
        df['SMA_10'] = df[target].rolling(window=10).mean()
    if indicator_config.get('use_sma_50'):
        df['SMA_50'] = df[target].rolling(window=50).mean()
    if indicator_config.get('use_sma_100'):
        df['SMA_100'] = df[target].rolling(window=100).mean()
    if indicator_config.get('use_sma_200'):
        df['SMA_200'] = df[target].rolling(window=200).mean()

    # Momentum Indicators
    if indicator_config.get('use_mom_14'):
        df['Mom_14'] = (df[target] - df[target].shift(14)) / df[target].shift(14)
    if indicator_config.get('use_mom_30'):
        df['Mom_30'] = (df[target] - df[target].shift(30)) / df[target].shift(30)
    if indicator_config.get('use_mom_60'):
        df['Mom_60'] = (df[target] - df[target].shift(60)) / df[target].shift(60)

    # Smoothed Momentum
    if indicator_config.get('use_smoothed_10') and 'Mom_30' in df.columns:
        df['Smoothed_10'] = df['Mom_30'].rolling(window=10).mean()
    if indicator_config.get('use_smoothed_30') and 'Mom_30' in df.columns:
        df['Smoothed_30'] = df['Mom_30'].rolling(window=30).mean()

    # Volume Indicators
    volume_col = f"{target}_volume"
    if volume_col in df.columns:
        if indicator_config.get('use_volume'):
            df['Volume'] = df[volume_col]
        if indicator_config.get('use_volume_sma'):
            df['Volume_SMA_20'] = df[volume_col].rolling(window=20).mean()
        if indicator_config.get('use_volume_ratio'):
            volume_avg = df[volume_col].rolling(window=20).mean()
            df['Volume_Ratio'] = df[volume_col] / volume_avg

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
    df_start_date = df.index[0]
    df_end_date = df.index[-1]

    # Get last signal date
    last_signal_date = results_series.index[-1] if len(results_series) > 0 else None

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

    # Calculate trades per month
    total_days = (df_end_date - df_start_date).days
    total_months = total_days / 30.44  # Average days per month
    trades_per_month = round(len(results_series) / total_months, 2) if total_months > 0 else 0

    return {
        "overall": overall,
        "90d": last_90d,
        "60d": last_60d,
        "30d": last_30d,
        "start_date": df_start_date.strftime('%Y-%m-%d'),
        "end_date": df_end_date.strftime('%Y-%m-%d'),
        "last_signal_date": last_signal_date.strftime('%Y-%m-%d') if last_signal_date else "No signals",
        "trades_per_month": trades_per_month
    }

def get_live_prediction(model, df, target, indicator_config):
    """Get live prediction from the model"""
    # Apply feature engineering (same as build_lab_features but without Signal)
    # Simple Moving Averages
    if indicator_config.get('use_sma_10'):
        df['SMA_10'] = df[target].rolling(window=10).mean()
    if indicator_config.get('use_sma_50'):
        df['SMA_50'] = df[target].rolling(window=50).mean()
    if indicator_config.get('use_sma_100'):
        df['SMA_100'] = df[target].rolling(window=100).mean()
    if indicator_config.get('use_sma_200'):
        df['SMA_200'] = df[target].rolling(window=200).mean()

    # Momentum Indicators
    if indicator_config.get('use_mom_14'):
        df['Mom_14'] = (df[target] - df[target].shift(14)) / df[target].shift(14)
    if indicator_config.get('use_mom_30'):
        df['Mom_30'] = (df[target] - df[target].shift(30)) / df[target].shift(30)
    if indicator_config.get('use_mom_60'):
        df['Mom_60'] = (df[target] - df[target].shift(60)) / df[target].shift(60)

    # Smoothed Momentum
    if indicator_config.get('use_smoothed_10') and 'Mom_30' in df.columns:
        df['Smoothed_10'] = df['Mom_30'].rolling(window=10).mean()
    if indicator_config.get('use_smoothed_30') and 'Mom_30' in df.columns:
        df['Smoothed_30'] = df['Mom_30'].rolling(window=30).mean()

    # Volume Indicators
    volume_col = f"{target}_volume"
    if volume_col in df.columns:
        if indicator_config.get('use_volume'):
            df['Volume'] = df[volume_col]
        if indicator_config.get('use_volume_sma'):
            df['Volume_SMA_20'] = df[volume_col].rolling(window=20).mean()
        if indicator_config.get('use_volume_ratio'):
            volume_avg = df[volume_col].rolling(window=20).mean()
            df['Volume_Ratio'] = df[volume_col] / volume_avg

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
        # Create indicator configuration
        indicator_config = {
            'use_sma_10': use_sma_10,
            'use_sma_50': use_sma_50,
            'use_sma_100': use_sma_100,
            'use_sma_200': use_sma_200,
            'use_mom_14': use_mom_14,
            'use_mom_30': use_mom_30,
            'use_mom_60': use_mom_60,
            'use_smoothed_10': use_smoothed_10,
            'use_smoothed_30': use_smoothed_30,
            'use_volume': use_volume,
            'use_volume_sma': use_volume_sma,
            'use_volume_ratio': use_volume_ratio
        }

        with st.spinner("Fetching data..."):
            raw_data = get_consolidated_data(
                tiingo_key, fred_key, target_asset,
                market_context, macro_context, start_date, end_date
            )

        if raw_data is not None:
            with st.spinner("Building features..."):
                processed_data = build_lab_features(raw_data.copy(), target_asset, indicator_config)

                X = processed_data.drop(columns=['Signal'])
                y = processed_data['Signal']

            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Signal", "📈 Backtest Results", "📉 Data Visualization", "🤖 Deploy Bot"])

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

                            rf_pred = get_live_prediction(rf_model, raw_data.copy(), target_asset, indicator_config)

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

                            knn_pred = get_live_prediction(knn_model, raw_data.copy(), target_asset, indicator_config)

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

                    # Date range info
                    st.info(f"📅 **Backtest Period:** {rf_results['start_date']} to {rf_results['end_date']} | 📍 **Last Signal:** {rf_results['last_signal_date']} | 📊 **Avg Trades/Month:** {rf_results['trades_per_month']}")

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

                    # Date range info
                    st.info(f"📅 **Backtest Period:** {knn_results['start_date']} to {knn_results['end_date']} | 📍 **Last Signal:** {knn_results['last_signal_date']} | 📊 **Avg Trades/Month:** {knn_results['trades_per_month']}")

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

            with tab4:
                st.header("🤖 Deploy & Save Bot Configuration")

                st.markdown("""
                Save your current bot configuration to deploy it for daily scanning.
                The bot will use the trained model and selected indicators to generate signals.
                """)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📋 Current Configuration")
                    bot_config = {
                        "target_asset": target_asset,
                        "start_date": start_date.strftime('%Y-%m-%d'),
                        "end_date": end_date.strftime('%Y-%m-%d'),
                        "market_context": market_context,
                        "macro_context": macro_context,
                        "indicators": indicator_config,
                        "model_type": "Random Forest" if use_rf else "K-Nearest Neighbors",
                        "model_params": {
                            "n_estimators": n_estimators if use_rf else None,
                            "max_depth": max_depth if use_rf else None,
                            "n_neighbors": n_neighbors if use_knn else None
                        },
                        "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }

                    st.json(bot_config)

                    # Bot performance summary
                    st.subheader("📊 Bot Performance")
                    if use_rf:
                        st.metric("Win Rate", f"{rf_results['overall']['win_rate']}%")
                        st.metric("Total Signals", rf_results['overall']['total'])
                        st.metric("Trades/Month", rf_results['trades_per_month'])

                with col2:
                    st.subheader("💾 Save Configuration")

                    bot_name = st.text_input("Bot Name", value=f"{target_asset}_bot_{datetime.now().strftime('%Y%m%d')}")

                    if st.button("💾 Save Bot Configuration", type="primary"):
                        # Save configuration as JSON
                        config_json = json.dumps(bot_config, indent=2)

                        st.download_button(
                            label="📥 Download Bot Config (JSON)",
                            data=config_json,
                            file_name=f"{bot_name}_config.json",
                            mime="application/json"
                        )

                        st.success(f"✅ Bot configuration '{bot_name}' ready for download!")

                    st.markdown("---")
                    st.subheader("📅 Daily Scanning Setup")
                    st.info("""
                    **To enable daily scanning:**

                    1. Download the bot configuration above
                    2. Use the config file with a scheduled task or cron job
                    3. The bot will scan daily and alert on BUY signals

                    **Example Python script for daily scanning:**
                    ```python
                    # Load config and run daily scan
                    import json
                    with open('bot_config.json', 'r') as f:
                        config = json.load(f)

                    # Your scanning logic here
                    # Send alerts via email/SMS when signal detected
                    ```
                    """)

                    st.markdown("---")
                    st.warning("⚠️ **Disclaimer:** Automated trading carries significant risk. Always review signals manually before trading.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit")
st.sidebar.markdown("Data: Tiingo & FRED")
