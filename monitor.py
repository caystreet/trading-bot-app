import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, timedelta
from tiingo import TiingoClient
from fredapi import Fred
import warnings
import requests
warnings.filterwarnings('ignore')

# Import the persistent caching module
from data_cache import (
    get_cached_consolidated_data,
    get_cached_alphavantage_data,
    get_cache_stats,
    clear_expired_cache,
    clear_all_cache
)

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
    data_source = st.radio("Data Source", ["Tiingo", "Alpha Vantage"], index=0)

    if data_source == "Tiingo":
        tiingo_key = st.text_input(
            "Tiingo API Key",
            value="e30bc8b0b855970d930743b116e517e36b72cc8f",
            type="password"
        )
        alphavantage_key = None
    else:
        alphavantage_key = st.text_input(
            "Alpha Vantage API Key",
            value="8IBMQ7XLHZ8CVWPC",
            type="password"
        )
        tiingo_key = None

    fred_key = st.text_input(
        "FRED API Key",
        value="1c1c04541bfefa93e9cb0db265da9c19",
        type="password"
    )

# Cache Management
with st.sidebar.expander("💾 Cache Settings", expanded=False):
    force_refresh = st.checkbox("Force refresh data", value=False,
                                help="Bypass cache and fetch fresh data from APIs")

    try:
        cache_stats = get_cache_stats()
        st.caption(f"📦 Cache: {cache_stats['valid_entries']} entries ({cache_stats['cache_size_mb']} MB)")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear Expired", use_container_width=True):
                deleted = clear_expired_cache()
                st.success(f"Cleared {deleted} expired entries")
        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                deleted = clear_all_cache()
                st.success(f"Cleared {deleted} entries")
    except Exception as e:
        st.caption(f"Cache not initialized: {e}")

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
def build_lab_features(df, target, indicator_config, trading_params=None):
    """Build features for prediction

    Args:
        df: DataFrame with price data
        target: Target asset column name
        indicator_config: Dict of indicator settings
        trading_params: Dict with take_profit_pct, stop_loss_pct, lookback_candles (optional)
    """
    # Get trading params with defaults
    if trading_params is None:
        trading_params = {}
    take_profit_pct = trading_params.get('take_profit_pct', 8.5)
    lookback_candles = trading_params.get('lookback_candles', 40)

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

    # Fill remaining NaNs in macro/market columns (from forward/backward fill gaps)
    # Only drop rows where the target asset or indicator columns are NaN
    st.write(f"Debug: Before dropna in build_lab_features: {len(df)} rows, NaN counts: {df.isna().sum().to_dict()}")

    # Fill any remaining macro NaNs with 0 (these are typically at edges of date range)
    macro_cols = [col for col in df.columns if col not in [target, volume_col] and not any(x in col for x in ['SMA', 'Mom', 'Smooth', 'Volume'])]
    for col in macro_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    result = df.dropna()
    st.write(f"Debug: After dropna in build_lab_features: {len(result)} rows")
    st.write(f"Debug: Using TP={take_profit_pct}%, Lookback={lookback_candles} candles")
    return result

# Alpha Vantage Helper Functions
def fetch_alphavantage_daily(symbol, api_key, outputsize='compact'):
    """Fetch daily price data from Alpha Vantage"""
    # Check if it's a crypto symbol
    crypto_pairs = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'BTC', 'ETH', 'SOL']
    is_crypto = any(crypto in symbol.upper() for crypto in crypto_pairs)

    if is_crypto:
        # For crypto, use DIGITAL_CURRENCY_DAILY
        if 'USD' in symbol:
            base = symbol.replace('USD', '')
            market = 'USD'
        else:
            base = symbol
            market = 'USD'

        url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={base}&market={market}&apikey={api_key}'
        response = requests.get(url)
        data = response.json()

        # Debug: Show what we got from API
        st.info(f"Debug - API Response Keys for {symbol}: {list(data.keys())}")

        # Check for API errors
        if 'Error Message' in data:
            st.error(f"Alpha Vantage Error for {symbol}: {data['Error Message']}")
            return None
        if 'Note' in data:
            st.warning(f"Alpha Vantage Note for {symbol}: {data['Note']}")
            return None
        if 'Information' in data:
            st.warning(f"Alpha Vantage Information for {symbol}: {data['Information']}")
            return None

        if 'Time Series (Digital Currency Daily)' in data:
            df = pd.DataFrame.from_dict(data['Time Series (Digital Currency Daily)'], orient='index')
            df.index = pd.to_datetime(df.index)
            # Crypto data has different column format - find the close and volume columns
            st.info(f"Debug - Available columns: {list(df.columns)}")

            # Find close column (may have market in parentheses or not)
            close_cols = [col for col in df.columns if 'close' in col.lower()]
            if close_cols:
                close_col = close_cols[0]
            else:
                st.error(f"No close column found in {list(df.columns)}")
                return None

            # Find volume column
            volume_cols = [col for col in df.columns if 'volume' in col.lower()]
            if volume_cols:
                volume_col = volume_cols[0]
            else:
                st.error(f"No volume column found in {list(df.columns)}")
                return None

            df = df[[close_col, volume_col]].copy()
            df.columns = ['close', 'volume']
            df = df.astype(float)
            df = df.sort_index()
            return df
        else:
            st.error(f"Unexpected response for {symbol}: {list(data.keys())}")
            return None
    else:
        # For stocks, use TIME_SERIES_DAILY
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={outputsize}&apikey={api_key}'
        response = requests.get(url)
        data = response.json()

        # Debug: Show what we got from API
        st.info(f"Debug - API Response Keys for {symbol}: {list(data.keys())}")

        # Check for API errors
        if 'Error Message' in data:
            st.error(f"Alpha Vantage Error for {symbol}: {data['Error Message']}")
            return None
        if 'Note' in data:
            st.warning(f"Alpha Vantage Note for {symbol}: {data['Note']}")
            return None
        if 'Information' in data:
            st.warning(f"Alpha Vantage Information for {symbol}: {data['Information']}")
            return None

        if 'Time Series (Daily)' in data:
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.sort_index()
            return df

    return None

def fetch_alphavantage_indicator(symbol, api_key, function, **params):
    """Fetch technical indicator from Alpha Vantage"""
    param_str = '&'.join([f"{k}={v}" for k, v in params.items()])
    url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&{param_str}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()

    for key in data.keys():
        if 'Technical Analysis' in key or key.startswith('Technical'):
            df = pd.DataFrame.from_dict(data[key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df = df.sort_index()
            return df
    return None

@st.cache_data(ttl=3600)
def get_alphavantage_data(av_key, fred_key, target_asset, market_context, macro_context, start_date, end_date, av_indicators):
    """Fetch data from Alpha Vantage with technical indicators"""
    try:
        fred = Fred(api_key=fred_key)

        # Fetch target asset price data
        df_target = fetch_alphavantage_daily(target_asset, av_key)
        if df_target is None:
            return None

        # Keep only close price and volume for target
        df_market = df_target[['close', 'volume']].copy()
        df_market.columns = [target_asset, f"{target_asset}_volume"]

        # Fetch market context assets (skip if same as target to avoid duplicates)
        for asset in market_context:
            if asset == target_asset:
                continue  # Skip duplicate of target asset
            df_context = fetch_alphavantage_daily(asset, av_key)
            if df_context is not None:
                df_market = df_market.join(df_context[['close']].rename(columns={'close': asset}), how='outer')

        # Fetch Alpha Vantage technical indicators if specified
        if av_indicators:
            if av_indicators.get('use_av_rsi'):
                df_rsi = fetch_alphavantage_indicator(target_asset, av_key, 'RSI', interval='daily', time_period=14, series_type='close')
                if df_rsi is not None:
                    df_market = df_market.join(df_rsi.rename(columns={'RSI': 'AV_RSI'}), how='left')

            if av_indicators.get('use_av_macd'):
                df_macd = fetch_alphavantage_indicator(target_asset, av_key, 'MACD', interval='daily', series_type='close')
                if df_macd is not None:
                    df_market = df_market.join(df_macd.add_prefix('AV_'), how='left')

            if av_indicators.get('use_av_adx'):
                df_adx = fetch_alphavantage_indicator(target_asset, av_key, 'ADX', interval='daily', time_period=14)
                if df_adx is not None:
                    df_market = df_market.join(df_adx.rename(columns={'ADX': 'AV_ADX'}), how='left')

            if av_indicators.get('use_av_cci'):
                df_cci = fetch_alphavantage_indicator(target_asset, av_key, 'CCI', interval='daily', time_period=20)
                if df_cci is not None:
                    df_market = df_market.join(df_cci.rename(columns={'CCI': 'AV_CCI'}), how='left')

            if av_indicators.get('use_av_stoch'):
                df_stoch = fetch_alphavantage_indicator(target_asset, av_key, 'STOCH', interval='daily')
                if df_stoch is not None:
                    df_market = df_market.join(df_stoch.add_prefix('AV_'), how='left')

            if av_indicators.get('use_av_bbands'):
                df_bbands = fetch_alphavantage_indicator(target_asset, av_key, 'BBANDS', interval='daily', time_period=20, series_type='close')
                if df_bbands is not None:
                    df_market = df_market.join(df_bbands.add_prefix('AV_'), how='left')

        # Fetch macro data from FRED
        for macro in macro_context:
            try:
                macro_series = fred.get_series(macro, start_date, end_date)
                macro_series = macro_series.resample('D').ffill()
                df_market = df_market.join(macro_series.rename(macro), how='left')
                df_market[macro] = df_market[macro].ffill()
            except:
                pass

        # Filter by date range
        df_market = df_market.loc[start_date:end_date]

        return df_market.dropna()
    except Exception as e:
        st.error(f"Alpha Vantage data fetch error: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def get_consolidated_data(tiingo_key, fred_key, target_asset, market_context, macro_context, start_date, end_date):
    """Fetch market and macro data"""
    try:
        client = TiingoClient({'api_key': tiingo_key})
        fred = Fred(api_key=fred_key)

        # Fetch market data - filter out target_asset from market_context to avoid duplicates
        filtered_context = [asset for asset in market_context if asset != target_asset]
        all_market = [target_asset] + filtered_context
        st.write(f"Debug: Fetching {all_market} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        df_market = client.get_dataframe(
            all_market,
            metric_name='close',
            startDate=start_date.strftime('%Y-%m-%d'),
            endDate=end_date.strftime('%Y-%m-%d')
        )
        st.write(f"Debug: Market data fetched: {len(df_market)} rows")

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

        st.write(f"Debug: Before dropna: {len(df_market)} rows")
        result = df_market.dropna(subset=[target_asset])
        st.write(f"Debug: After dropna(subset=[target_asset]): {len(result)} rows")
        return result

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
    st.subheader("🔔 Recent Alerts")
    if st.session_state.bot_alerts:
        recent_alerts = st.session_state.bot_alerts[-10:][::-1]  # Last 10, reversed

        for alert in recent_alerts:
            alert_color = "🟢" if alert['signal'] == 'BUY' else "⚪"
            st.info(f"{alert_color} **{alert['bot_name']}** - {alert['signal']} signal on {alert['asset']} at {alert['timestamp']}")
    else:
        st.info("No alerts yet. Scan your bots to detect BUY signals!")

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
                            # Get AV indicators config from bot if it exists
                            av_indicators = bot['config'].get('av_indicators', {})

                            # Use appropriate data source with caching
                            if data_source == "Alpha Vantage" and alphavantage_key:
                                scan_data = get_cached_alphavantage_data(
                                    alphavantage_key,
                                    fred_key,
                                    bot['config']['target_asset'],
                                    list(bot['config']['market_context']),
                                    list(bot['config']['macro_context']),
                                    datetime.now() - timedelta(days=800),
                                    datetime.now(),
                                    av_indicators,
                                    force_refresh=force_refresh
                                )
                            else:
                                scan_data = get_cached_consolidated_data(
                                    tiingo_key,
                                    fred_key,
                                    bot['config']['target_asset'],
                                    list(bot['config']['market_context']),
                                    list(bot['config']['macro_context']),
                                    datetime.now() - timedelta(days=800),
                                    datetime.now(),
                                    force_refresh=force_refresh
                                )

                            if scan_data is not None:
                                # Get trading params from bot config (with defaults for older bots)
                                trading_params = bot['config'].get('trading_params', {
                                    'take_profit_pct': 8.5,
                                    'stop_loss_pct': 3.0,
                                    'lookback_candles': 40
                                })
                                scan_df = build_lab_features(
                                    scan_data, bot['config']['target_asset'],
                                    bot['config']['indicators'], trading_params
                                )

                                if len(scan_df) > 0:
                                    # Get all columns for features (model was trained with all columns including target price)
                                    X_latest = scan_df.iloc[-1:]
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
        # Get trading params for display
        trading_params = bot['config'].get('trading_params', {'take_profit_pct': 8.5, 'stop_loss_pct': 3.0, 'lookback_candles': 40})
        tp_display = trading_params.get('take_profit_pct', 8.5)
        sl_display = trading_params.get('stop_loss_pct', 3.0)

        with st.expander(f"🤖 {bot['name']} ({bot['config']['target_asset']}) - TP:{tp_display}% SL:{sl_display}%", expanded=False):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.write(f"**Status:** {bot['status'].upper()}")
                st.write(f"**Deployed:** {bot['deployed_at']}")

            with col2:
                st.write(f"**Last Scan:** {bot['last_scan'] or 'Never'}")
                st.write(f"**Last Signal:** {bot['last_signal'] or 'None'}")
                st.write(f"**TP/SL:** {tp_display}% / {sl_display}%")

            with col3:
                # Individual scan button
                if st.button(f"🔍 Scan", key=f"scan_{idx}"):
                    with st.spinner(f"Scanning {bot['name']}..."):
                        try:
                            start_dt = datetime.now() - timedelta(days=800)
                            end_dt = datetime.now()
                            st.write(f"Debug: Scanning with dates {start_dt.date()} to {end_dt.date()}")

                            # Get AV indicators config from bot if it exists
                            av_indicators = bot['config'].get('av_indicators', {})

                            # Use appropriate data source with caching
                            if data_source == "Alpha Vantage" and alphavantage_key:
                                scan_data = get_cached_alphavantage_data(
                                    alphavantage_key,
                                    fred_key,
                                    bot['config']['target_asset'],
                                    list(bot['config']['market_context']),
                                    list(bot['config']['macro_context']),
                                    start_dt,
                                    end_dt,
                                    av_indicators,
                                    force_refresh=force_refresh
                                )
                            else:
                                scan_data = get_cached_consolidated_data(
                                    tiingo_key,
                                    fred_key,
                                    bot['config']['target_asset'],
                                    list(bot['config']['market_context']),
                                    list(bot['config']['macro_context']),
                                    start_dt,
                                    end_dt,
                                    force_refresh=force_refresh
                                )

                            if scan_data is not None:
                                st.write(f"Debug: Fetched {len(scan_data)} rows of raw data")
                                st.write(f"Debug: Indicator config: {bot['config']['indicators']}")
                                # Get trading params from bot config (with defaults for older bots)
                                trading_params = bot['config'].get('trading_params', {
                                    'take_profit_pct': 8.5,
                                    'stop_loss_pct': 3.0,
                                    'lookback_candles': 40
                                })
                                st.write(f"Debug: Trading params: TP={trading_params.get('take_profit_pct')}%, SL={trading_params.get('stop_loss_pct')}%, Lookback={trading_params.get('lookback_candles')}")
                                scan_df = build_lab_features(
                                    scan_data, bot['config']['target_asset'],
                                    bot['config']['indicators'], trading_params
                                )
                                st.write(f"Debug: After features {len(scan_df)} rows, Columns: {list(scan_df.columns)}")

                                if len(scan_df) > 0:
                                    # Get all columns for features (model was trained with all columns including target price)
                                    st.write(f"Debug: Feature columns: {list(scan_df.columns)}")
                                    X_latest = scan_df.iloc[-1:]
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
