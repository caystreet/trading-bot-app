import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, timedelta
from tiingo import TiingoClient
from fredapi import Fred
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bot Monitor",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Trading Bot Monitor")
st.markdown("Monitor and scan your deployed trading bots")

# Session state for bots
if 'monitor_bots' not in st.session_state:
    st.session_state.monitor_bots = []
if 'bot_alerts' not in st.session_state:
    st.session_state.bot_alerts = []

# Sidebar - API Keys
st.sidebar.header("⚙️ Configuration")

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

# Import bots section
st.sidebar.markdown("---")
st.sidebar.header("📥 Import Bots")

uploaded_file = st.sidebar.file_uploader("Upload Bot File", type=['pkl'])
if uploaded_file is not None:
    try:
        import_data = pickle.loads(uploaded_file.read())
        imported_bots = import_data.get('bots', [])

        if st.sidebar.button("✅ Load Bots", type="primary"):
            st.session_state.monitor_bots = imported_bots
            st.session_state.bot_alerts = import_data.get('alerts', [])
            st.sidebar.success(f"Loaded {len(imported_bots)} bot(s)!")
            st.rerun()
    except Exception as e:
        st.sidebar.error(f"Import failed: {str(e)}")

# Helper functions
def build_lab_features(df, target, indicator_config):
    """Build features for prediction"""
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

    return df.dropna()

@st.cache_data(ttl=3600)
def get_consolidated_data(tiingo_key, fred_key, target_asset, market_context, macro_context, start_date, end_date):
    """Fetch market and macro data"""
    try:
        client = TiingoClient({'api_key': tiingo_key})
        fred = Fred(api_key=fred_key)

        # Fetch market data
        all_market = [target_asset] + market_context
        df_market = client.get_dataframe(
            all_market,
            metric_name='close',
            startDate=start_date.strftime('%Y-%m-%d'),
            endDate=end_date.strftime('%Y-%m-%d')
        )

        # Fetch volume data
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
            df_market[f"{target_asset}_volume"] = 0

        # Fetch macro data
        for macro in macro_context:
            try:
                macro_series = fred.get_series(macro, start_date, end_date)
                macro_series = macro_series.resample('D').ffill()
                df_market = df_market.join(macro_series.rename(macro), how='left')
                df_market[macro] = df_market[macro].ffill()
            except:
                pass

        return df_market.dropna()

    except Exception as e:
        st.error(f"Data fetch error: {str(e)}")
        return None

# Main content
if len(st.session_state.monitor_bots) == 0:
    st.info("👆 Import bots from the sidebar to get started")
    st.markdown("""
    ### How to use:
    1. Export bots from the main Trading Bot Tester app
    2. Upload the exported file in the sidebar
    3. Click "Load Bots" to import
    4. Scan and monitor your bots here
    """)
else:
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Bots", len(st.session_state.monitor_bots))

    with col2:
        active_bots = len([b for b in st.session_state.monitor_bots if b['status'] == 'active'])
        st.metric("Active Bots", active_bots)

    with col3:
        buy_signals = len([a for a in st.session_state.bot_alerts if a['signal'] == 'BUY'])
        st.metric("BUY Signals", buy_signals)

    with col4:
        recent_scans = len([b for b in st.session_state.monitor_bots if b['last_scan']])
        st.metric("Scanned Bots", recent_scans)

    st.markdown("---")

    # Recent alerts
    if st.session_state.bot_alerts:
        st.subheader("🔔 Recent Alerts")
        recent_alerts = st.session_state.bot_alerts[-10:][::-1]  # Last 10, reversed

        for alert in recent_alerts:
            alert_color = "🟢" if alert['signal'] == 'BUY' else "⚪"
            st.info(f"{alert_color} **{alert['bot_name']}** - {alert['signal']} signal on {alert['asset']} at {alert['timestamp']}")

    st.markdown("---")

    # Bot list with scan functionality
    st.subheader("🤖 Bot Dashboard")

    # Scan All button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔍 Scan All Bots", type="primary"):
            with st.spinner("Scanning all bots..."):
                for bot in st.session_state.monitor_bots:
                    if bot['status'] == 'active':
                        try:
                            scan_data = get_consolidated_data(
                                tiingo_key,
                                fred_key,
                                bot['config']['target_asset'],
                                bot['config']['market_context'],
                                bot['config']['macro_context'],
                                datetime.now() - timedelta(days=1825),  # 5 years for all indicators
                                datetime.now()
                            )

                            if scan_data is not None:
                                scan_df = build_lab_features(scan_data, bot['config']['target_asset'], bot['config']['indicators'])

                                if len(scan_df) > 0:
                                    # Get feature columns (exclude target asset price column)
                                    feature_cols = [col for col in scan_df.columns if col != bot['config']['target_asset']]
                                    X_latest = scan_df[feature_cols].iloc[-1:]
                                    prediction = bot['model'].predict(X_latest)[0]
                                    signal = "BUY" if prediction == 1 else "WAIT"

                                    bot['last_scan'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    bot['last_signal'] = signal

                                    if signal == "BUY":
                                        alert = {
                                            "bot_name": bot['name'],
                                            "signal": signal,
                                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                            "asset": bot['config']['target_asset']
                                        }
                                        st.session_state.bot_alerts.append(alert)
                        except:
                            pass

                st.success("Scan complete!")
                st.rerun()

    st.markdown("---")

    # Individual bot cards
    for idx, bot in enumerate(st.session_state.monitor_bots):
        with st.expander(f"🤖 {bot['name']} ({bot['config']['target_asset']}) - {bot['model_type']}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.write(f"**Status:** {bot['status'].upper()}")
                st.write(f"**Deployed:** {bot['deployed_at']}")

            with col2:
                st.write(f"**Last Scan:** {bot['last_scan'] or 'Never'}")
                st.write(f"**Last Signal:** {bot['last_signal'] or 'None'}")

            with col3:
                # Individual scan button
                if st.button(f"🔍 Scan", key=f"scan_{idx}"):
                    with st.spinner(f"Scanning {bot['name']}..."):
                        try:
                            scan_data = get_consolidated_data(
                                tiingo_key,
                                fred_key,
                                bot['config']['target_asset'],
                                bot['config']['market_context'],
                                bot['config']['macro_context'],
                                datetime.now() - timedelta(days=1825),  # 5 years for all indicators
                                datetime.now()
                            )

                            if scan_data is not None:
                                st.write(f"Debug: Fetched {len(scan_data)} rows of raw data")
                                scan_df = build_lab_features(scan_data, bot['config']['target_asset'], bot['config']['indicators'])
                                st.write(f"Debug: After features {len(scan_df)} rows, Columns: {list(scan_df.columns)}")

                                if len(scan_df) > 0:
                                    # Get feature columns (exclude target asset price column)
                                    feature_cols = [col for col in scan_df.columns if col != bot['config']['target_asset']]
                                    st.write(f"Debug: Feature columns: {feature_cols}")
                                    X_latest = scan_df[feature_cols].iloc[-1:]
                                    st.write(f"Debug: X_latest shape: {X_latest.shape}")
                                    prediction = bot['model'].predict(X_latest)[0]
                                    signal = "BUY" if prediction == 1 else "WAIT"

                                    bot['last_scan'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    bot['last_signal'] = signal

                                    if signal == "BUY":
                                        alert = {
                                            "bot_name": bot['name'],
                                            "signal": signal,
                                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                            "asset": bot['config']['target_asset']
                                        }
                                        st.session_state.bot_alerts.append(alert)

                                    st.success(f"Signal: **{signal}**")
                                else:
                                    st.error(f"Insufficient data after applying indicators. Raw data had {len(scan_data)} rows but after dropna() got 0 rows.")
                            else:
                                st.error("Data fetch failed")
                        except Exception as e:
                            st.error(f"Scan failed: {str(e)}")

            with col4:
                # Controls
                if bot['status'] == 'active':
                    if st.button("⏸️ Pause", key=f"pause_{idx}"):
                        bot['status'] = 'paused'
                        st.rerun()
                else:
                    if st.button("▶️ Resume", key=f"resume_{idx}"):
                        bot['status'] = 'active'
                        st.rerun()

            # Show config
            with st.expander("📋 Configuration"):
                st.json(bot['config'])

    # Export updated bots
    st.markdown("---")
    st.subheader("💾 Export Updated Bots")

    export_data = {
        'bots': st.session_state.monitor_bots,
        'alerts': st.session_state.bot_alerts,
        'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    bot_data = pickle.dumps(export_data)
    st.download_button(
        label="📥 Download Updated Bots",
        data=bot_data,
        file_name=f"bots_monitored_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
        mime="application/octet-stream",
        help="Save bots with updated scan results"
    )
