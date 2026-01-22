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
from datetime import datetime, timedelta
import os
import requests
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Bot Tester",
    page_icon="🤖",
    layout="wide"
)

# Bot persistence - NOTE: Streamlit Cloud has ephemeral filesystem
# Bots persist during the session but will be cleared on app restart
# Users can export/import bot collections for long-term storage

# Initialize session state for deployed bots
if 'deployed_bots' not in st.session_state:
    st.session_state.deployed_bots = []
if 'bot_alerts' not in st.session_state:
    st.session_state.bot_alerts = []
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'scanning_bot' not in st.session_state:
    st.session_state.scanning_bot = None

def export_all_bots():
    """Export all bots as a downloadable file"""
    export_data = {
        'bots': st.session_state.deployed_bots,
        'alerts': st.session_state.bot_alerts,
        'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    return pickle.dumps(export_data)

def import_bots(uploaded_file):
    """Import bots from uploaded file"""
    try:
        import_data = pickle.loads(uploaded_file.read())
        st.session_state.deployed_bots = import_data.get('bots', [])
        st.session_state.bot_alerts = import_data.get('alerts', [])
        return True, len(import_data.get('bots', []))
    except Exception as e:
        return False, str(e)

# Title
st.title("🤖 Trading Bot Tester")
st.markdown("Machine Learning-powered trading signal generator for stocks and crypto")

# Debug: Show app version to verify deployment
st.sidebar.caption("App version: 2.0.1 (Bot Persistence)")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# API Keys
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

    if data_source == "Alpha Vantage":
        st.markdown("**Alpha Vantage Technical Indicators**")
        use_av_rsi = st.checkbox("RSI (14)", value=False)
        use_av_macd = st.checkbox("MACD", value=False)
        use_av_adx = st.checkbox("ADX (14)", value=False)
        use_av_cci = st.checkbox("CCI (20)", value=False)
        use_av_stoch = st.checkbox("Stochastic Oscillator", value=False)
        use_av_bbands = st.checkbox("Bollinger Bands", value=False)
    else:
        use_av_rsi = use_av_macd = use_av_adx = use_av_cci = use_av_stoch = use_av_bbands = False

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

# Alpha Vantage Helper Functions
def fetch_alphavantage_daily(symbol, api_key, outputsize='full'):
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

        if 'Time Series (Digital Currency Daily)' in data:
            df = pd.DataFrame.from_dict(data['Time Series (Digital Currency Daily)'], orient='index')
            df.index = pd.to_datetime(df.index)
            # Crypto data has different column format
            df = df[[f'4a. close ({market})', f'5. volume']].copy()
            df.columns = ['close', 'volume']
            df = df.astype(float)
            df = df.sort_index()
            return df
    else:
        # For stocks, use TIME_SERIES_DAILY
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={outputsize}&apikey={api_key}'
        response = requests.get(url)
        data = response.json()

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

    # Different indicators have different keys
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
            st.error(f"Failed to fetch {target_asset} from Alpha Vantage")
            return None

        # Keep only close price and volume for target
        df_market = df_target[['close', 'volume']].copy()
        df_market.columns = [target_asset, f"{target_asset}_volume"]

        # Fetch market context assets
        for asset in market_context:
            df_context = fetch_alphavantage_daily(asset, av_key)
            if df_context is not None:
                df_market = df_market.join(df_context[['close']].rename(columns={'close': asset}), how='outer')

        # Fetch Alpha Vantage technical indicators
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
    st.session_state.run_analysis_clicked = True

if st.session_state.get('run_analysis_clicked', False):
    # Validate API keys based on data source
    if data_source == "Tiingo" and (not tiingo_key or not fred_key):
        st.error("Please provide Tiingo and FRED API keys")
    elif data_source == "Alpha Vantage" and (not alphavantage_key or not fred_key):
        st.error("Please provide Alpha Vantage and FRED API keys")
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

        # Alpha Vantage indicators config
        av_indicators_config = {
            'use_av_rsi': use_av_rsi,
            'use_av_macd': use_av_macd,
            'use_av_adx': use_av_adx,
            'use_av_cci': use_av_cci,
            'use_av_stoch': use_av_stoch,
            'use_av_bbands': use_av_bbands
        }

        with st.spinner("Fetching data..."):
            if data_source == "Alpha Vantage":
                raw_data = get_alphavantage_data(
                    alphavantage_key, fred_key, target_asset,
                    market_context, macro_context, start_date, end_date, av_indicators_config
                )
            else:
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

                            # Store in session state for bot deployment
                            st.session_state.rf_model = rf_model

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

                            # Store in session state for bot deployment
                            st.session_state.knn_model = knn_model

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
                        # Store in session state
                        st.session_state.rf_results = rf_results

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
                st.header("🤖 Deploy & Manage Bots")

                st.markdown("""
                Deploy bots in-app for monitoring or download configurations for external scanning.
                Deployed bots will use trained models to monitor for signals.
                """)

                # Create two sub-tabs
                deploy_tab1, deploy_tab2 = st.tabs(["🚀 Deploy New Bot", "📡 Active Bots"])

                with deploy_tab1:
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
                            "av_indicators": av_indicators_config if data_source == "Alpha Vantage" else {},
                            "data_source": data_source,
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
                        st.subheader("🚀 Deploy Options")

                        bot_name = st.text_input("Bot Name", value=f"{target_asset}_bot_{datetime.now().strftime('%Y%m%d')}")

                        # In-App Deployment - Show separate buttons if both models trained
                        if use_rf and use_knn and 'rf_model' in st.session_state and 'knn_model' in st.session_state:
                            st.write("**Deploy which model?**")
                            col_rf, col_knn = st.columns(2)

                            with col_rf:
                                if st.button("🌲 Deploy RF", use_container_width=True):
                                    deployed_bot = {
                                        "name": f"{bot_name}_RF",
                                        "config": bot_config.copy(),
                                        "model": st.session_state.rf_model,
                                        "model_type": "Random Forest",
                                        "deployed_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        "status": "active",
                                        "last_scan": None,
                                        "last_signal": None
                                    }
                                    deployed_bot["config"]["model_type"] = "Random Forest"
                                    st.session_state.deployed_bots.append(deployed_bot)
                                    st.success(f"✅ RF Bot deployed! Total bots: {len(st.session_state.deployed_bots)}")
                                    st.info("👉 Check sidebar for bot")

                            with col_knn:
                                if st.button("🔵 Deploy KNN", use_container_width=True):
                                    deployed_bot = {
                                        "name": f"{bot_name}_KNN",
                                        "config": bot_config.copy(),
                                        "model": st.session_state.knn_model,
                                        "model_type": "KNN",
                                        "deployed_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        "status": "active",
                                        "last_scan": None,
                                        "last_signal": None
                                    }
                                    deployed_bot["config"]["model_type"] = "KNN"
                                    st.session_state.deployed_bots.append(deployed_bot)
                                    st.success(f"✅ KNN Bot deployed! Total bots: {len(st.session_state.deployed_bots)}")
                                    st.info("👉 Check sidebar for bot")

                        # Single model deployment
                        elif use_rf and 'rf_model' in st.session_state:
                            if st.button("🚀 Deploy RF Bot In-App", type="primary", use_container_width=True):
                                deployed_bot = {
                                    "name": bot_name,
                                    "config": bot_config.copy(),
                                    "model": st.session_state.rf_model,
                                    "model_type": "Random Forest",
                                    "deployed_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    "status": "active",
                                    "last_scan": None,
                                    "last_signal": None
                                }
                                st.session_state.deployed_bots.append(deployed_bot)
                                st.success(f"✅ Bot '{bot_name}' deployed! Total bots: {len(st.session_state.deployed_bots)}")
                                st.info("👉 Check sidebar for bot")

                        elif use_knn and 'knn_model' in st.session_state:
                            if st.button("🚀 Deploy KNN Bot In-App", type="primary", use_container_width=True):
                                deployed_bot = {
                                    "name": bot_name,
                                    "config": bot_config.copy(),
                                    "model": st.session_state.knn_model,
                                    "model_type": "KNN",
                                    "deployed_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    "status": "active",
                                    "last_scan": None,
                                    "last_signal": None
                                }
                                st.session_state.deployed_bots.append(deployed_bot)
                                st.success(f"✅ Bot '{bot_name}' deployed! Total bots: {len(st.session_state.deployed_bots)}")
                                st.info("👉 Check sidebar for bot")

                        else:
                            st.warning("⚠️ Please run analysis first to train models")

                        st.info("👉 Check sidebar 'Active Bots' section")

                        st.markdown("---")

                        # Download Configuration
                        st.subheader("💾 Download Configuration")
                        config_json = json.dumps(bot_config, indent=2)

                        st.download_button(
                            label="📥 Download Bot Config (JSON)",
                            data=config_json,
                            file_name=f"{bot_name}_config.json",
                            mime="application/json",
                            use_container_width=True
                        )

                        st.caption("Download for external scheduling/scanning")

                with deploy_tab2:
                    st.subheader("📡 Active Bots")

                    if not st.session_state.deployed_bots:
                        st.info("No bots deployed yet. Deploy a bot in the 'Deploy New Bot' tab.")
                    else:
                        # Display alerts first
                        if st.session_state.bot_alerts:
                            st.subheader("🔔 Recent Alerts")
                            for alert in st.session_state.bot_alerts[-5:]:  # Show last 5 alerts
                                alert_type = "🟢" if alert['signal'] == 'BUY' else "⚪"
                                st.info(f"{alert_type} **{alert['bot_name']}** - {alert['signal']} signal detected at {alert['timestamp']}")

                        st.markdown("---")

                        # Display each deployed bot
                        for idx, bot in enumerate(st.session_state.deployed_bots):
                            with st.expander(f"🤖 {bot['name']} ({bot['model_type']}) - Status: {bot['status'].upper()}", expanded=True):
                                col1, col2, col3 = st.columns([2, 2, 1])

                                with col1:
                                    st.write(f"**Target:** {bot['config']['target_asset']}")
                                    st.write(f"**Deployed:** {bot['deployed_at']}")
                                    st.write(f"**Last Scan:** {bot['last_scan'] or 'Never'}")

                                with col2:
                                    st.write(f"**Last Signal:** {bot['last_signal'] or 'None'}")
                                    st.write(f"**Model:** {bot['model_type']}")

                                with col3:
                                    # Scan button - prevent multiple simultaneous scans
                                    scan_disabled = st.session_state.scanning_bot is not None
                                    if st.button("🔍 Scan Now", key=f"scan_{idx}", disabled=scan_disabled):
                                        st.session_state.scanning_bot = bot['name']
                                        st.rerun()

                                # Perform scan if this bot is marked for scanning (only once per button click)
                                if st.session_state.scanning_bot == bot['name'] and not st.session_state.get('scan_in_progress', False):
                                    # Set in-progress flag immediately to prevent re-execution
                                    st.session_state.scan_in_progress = True

                                    with st.spinner(f"Scanning {bot['name']}..."):
                                        try:
                                            # Get fresh data for scanning
                                            scan_data = get_consolidated_data(
                                                tiingo_key,
                                                fred_key,
                                                bot['config']['target_asset'],
                                                bot['config']['market_context'],
                                                bot['config']['macro_context'],
                                                datetime.now() - timedelta(days=365),  # Last year of data
                                                datetime.now()
                                            )

                                            # Build features using bot's indicator config
                                            scan_df = build_lab_features(scan_data, bot['config']['target_asset'], bot['config']['indicators'])
                                            scan_df = scan_df.dropna()

                                            if len(scan_df) > 0:
                                                # Get prediction
                                                X_latest = scan_df.iloc[-1:][scan_df.columns.drop([f"{bot['config']['target_asset']}_target"])]
                                                prediction = bot['model'].predict(X_latest)[0]
                                                signal = "BUY" if prediction == 1 else "WAIT"

                                                # Update bot status
                                                bot['last_scan'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                                bot['last_signal'] = signal

                                                # If BUY signal, add to alerts
                                                if signal == "BUY":
                                                    alert = {
                                                        "bot_name": bot['name'],
                                                        "signal": signal,
                                                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                        "asset": bot['config']['target_asset']
                                                    }
                                                    st.session_state.bot_alerts.append(alert)

                                                st.success(f"Scan complete! Signal: **{signal}**")
                                            else:
                                                st.error("Insufficient data for scanning")

                                        except Exception as e:
                                            st.error(f"Scan failed: {str(e)}")

                                        finally:
                                            # Clear both flags
                                            st.session_state.scanning_bot = None
                                            st.session_state.scan_in_progress = False

                                # Bot controls
                                col1, col2 = st.columns(2)

                                with col1:
                                    # Pause/Resume button
                                    if bot['status'] == 'active':
                                        if st.button("⏸️ Pause", key=f"pause_{idx}", use_container_width=True):
                                            bot['status'] = 'paused'
                                            st.rerun()
                                    else:
                                        if st.button("▶️ Resume", key=f"resume_{idx}", use_container_width=True):
                                            bot['status'] = 'active'
                                            st.rerun()

                                with col2:
                                    # Delete button
                                    if st.button("🗑️ Delete", key=f"delete_{idx}", use_container_width=True):
                                        st.session_state.deployed_bots.pop(idx)
                                        st.success(f"Bot '{bot['name']}' deleted")
                                        st.rerun()

                st.markdown("---")
                st.warning("⚠️ **Disclaimer:** Automated trading carries significant risk. Always review signals manually before trading.")

# Active Bots Quick Access
st.sidebar.markdown("---")
st.sidebar.header("🤖 Active Bots")

# Debug info
bot_count = len(st.session_state.deployed_bots)
st.sidebar.caption(f"Debug: {bot_count} bot(s) in memory")

if bot_count == 0:
    st.sidebar.info("No active bots")
else:
    st.sidebar.metric("Active Bots", bot_count)

    # Show recent alerts
    if st.session_state.bot_alerts:
        recent_buy_alerts = [a for a in st.session_state.bot_alerts if a['signal'] == 'BUY'][-3:]
        if recent_buy_alerts:
            st.sidebar.success(f"🔔 {len(recent_buy_alerts)} recent BUY signal(s)")

    # Quick view of each bot
    for idx, bot in enumerate(st.session_state.deployed_bots):
        with st.sidebar.expander(f"🤖 {bot['name']}", expanded=False):
            st.write(f"**Asset:** {bot['config']['target_asset']}")
            st.write(f"**Status:** {bot['status'].upper()}")
            st.write(f"**Last Signal:** {bot['last_signal'] or 'None'}")

            if st.button("🔍 Scan", key=f"sidebar_scan_{idx}"):
                st.info("Go to 'Deploy Bot' tab to scan")

# Export/Import bots (always visible)
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Save/Load Bots")

col1, col2 = st.sidebar.columns(2)

with col1:
    if len(st.session_state.deployed_bots) > 0:
        bot_data = export_all_bots()
        st.download_button(
            label="💾 Export",
            data=bot_data,
            file_name=f"bots_backup_{datetime.now().strftime('%Y%m%d')}.pkl",
            mime="application/octet-stream",
            use_container_width=True,
            help="Save your bots to reload later"
        )
    else:
        st.button("💾 Export", disabled=True, use_container_width=True, help="No bots to export")

with col2:
    uploaded_file = st.file_uploader("Import", type=['pkl'], label_visibility="collapsed", key="bot_import")
    if uploaded_file is not None:
        success, result = import_bots(uploaded_file)
        if success:
            st.success(f"✅ Loaded {result} bot(s)")
            st.rerun()
        else:
            st.error(f"❌ Import failed: {result}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit")
st.sidebar.markdown("Data: Tiingo & FRED")

