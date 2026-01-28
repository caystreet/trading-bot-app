"""
Persistent Data Cache Module for Trading Bot

This module provides caching for API responses to reduce API calls and avoid rate limits.

Environment Detection:
- Local: Uses SQLite for persistent storage across restarts
- Streamlit Cloud: Uses st.session_state (in-memory) since filesystem is ephemeral

Features:
- Configurable TTL (time-to-live) for cached data
- Separate caches for different data types (prices, indicators, macro)
- Smart cache invalidation - only fetches new data when needed
- Automatic environment detection
"""

import pandas as pd
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import os

# Detect if running on Streamlit Cloud
IS_STREAMLIT_CLOUD = os.environ.get('STREAMLIT_SHARING_MODE') or os.environ.get('STREAMLIT_SERVER_HEADLESS')

# Default cache directory (for local use)
CACHE_DIR = Path.home() / '.trading_bot_cache'
CACHE_DB = CACHE_DIR / 'api_cache.db'

# Default TTL values (in seconds)
DEFAULT_TTL = {
    'price_data': 3600,      # 1 hour for price data
    'indicator': 3600,        # 1 hour for technical indicators
    'macro_data': 86400,      # 24 hours for macro data (updates less frequently)
    'fred_data': 86400,       # 24 hours for FRED data
}


# ============================================
# Session State Cache (for Streamlit Cloud)
# ============================================

def _get_session_cache():
    """Get or initialize the session state cache"""
    import streamlit as st
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = {}
    return st.session_state.data_cache


def _session_get(cache_key: str):
    """Get data from session cache if not expired"""
    cache = _get_session_cache()
    if cache_key in cache:
        entry = cache[cache_key]
        if datetime.now() < entry['expires_at']:
            return entry['data']
        else:
            # Expired, remove it
            del cache[cache_key]
    return None


def _session_set(cache_key: str, data, data_type: str, ttl_seconds: int = None):
    """Store data in session cache"""
    cache = _get_session_cache()
    if ttl_seconds is None:
        ttl_seconds = DEFAULT_TTL.get(data_type, 3600)

    cache[cache_key] = {
        'data': data,
        'data_type': data_type,
        'created_at': datetime.now(),
        'expires_at': datetime.now() + timedelta(seconds=ttl_seconds)
    }


def _session_clear_expired():
    """Remove expired entries from session cache"""
    cache = _get_session_cache()
    now = datetime.now()
    expired_keys = [k for k, v in cache.items() if now >= v['expires_at']]
    for k in expired_keys:
        del cache[k]
    return len(expired_keys)


def _session_clear_all():
    """Clear all session cache"""
    import streamlit as st
    count = len(st.session_state.get('data_cache', {}))
    st.session_state.data_cache = {}
    return count


def _session_stats():
    """Get session cache statistics"""
    cache = _get_session_cache()
    now = datetime.now()

    valid = sum(1 for v in cache.values() if now < v['expires_at'])
    expired = len(cache) - valid

    by_type = {}
    for v in cache.values():
        if now < v['expires_at']:
            dt = v['data_type']
            by_type[dt] = by_type.get(dt, 0) + 1

    return {
        'total_entries': len(cache),
        'valid_entries': valid,
        'expired_entries': expired,
        'by_type': by_type,
        'cache_size_mb': 0,  # Can't easily measure in-memory size
        'storage': 'session_state'
    }


# ============================================
# SQLite Cache (for Local Development)
# ============================================

def _init_sqlite_db():
    """Initialize the SQLite cache database"""
    import sqlite3
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(CACHE_DB))
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_cache (
            cache_key TEXT PRIMARY KEY,
            data_type TEXT NOT NULL,
            symbol TEXT,
            data BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            metadata TEXT
        )
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_cache_type_symbol
        ON api_cache(data_type, symbol)
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_cache_expires
        ON api_cache(expires_at)
    ''')

    conn.commit()
    conn.close()


def _sqlite_get(cache_key: str):
    """Get data from SQLite cache if not expired"""
    import sqlite3
    _init_sqlite_db()

    conn = sqlite3.connect(str(CACHE_DB))
    cursor = conn.cursor()

    cursor.execute('''
        SELECT data, expires_at FROM api_cache
        WHERE cache_key = ? AND expires_at > datetime('now')
    ''', (cache_key,))

    result = cursor.fetchone()
    conn.close()

    if result:
        try:
            return pickle.loads(result[0])
        except Exception:
            return None
    return None


def _sqlite_set(cache_key: str, data, data_type: str, symbol: str = None,
                ttl_seconds: int = None, metadata: dict = None):
    """Store data in SQLite cache"""
    import sqlite3
    _init_sqlite_db()

    if ttl_seconds is None:
        ttl_seconds = DEFAULT_TTL.get(data_type, 3600)

    expires_at = datetime.now() + timedelta(seconds=ttl_seconds)

    conn = sqlite3.connect(str(CACHE_DB))
    cursor = conn.cursor()

    cursor.execute('''
        INSERT OR REPLACE INTO api_cache
        (cache_key, data_type, symbol, data, expires_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        cache_key,
        data_type,
        symbol,
        pickle.dumps(data),
        expires_at.isoformat(),
        json.dumps(metadata) if metadata else None
    ))

    conn.commit()
    conn.close()


def _sqlite_clear_expired():
    """Remove expired entries from SQLite cache"""
    import sqlite3
    _init_sqlite_db()

    conn = sqlite3.connect(str(CACHE_DB))
    cursor = conn.cursor()

    cursor.execute("DELETE FROM api_cache WHERE expires_at < datetime('now')")
    deleted = cursor.rowcount

    conn.commit()
    conn.close()

    return deleted


def _sqlite_clear_all():
    """Clear all SQLite cache"""
    import sqlite3
    _init_sqlite_db()

    conn = sqlite3.connect(str(CACHE_DB))
    cursor = conn.cursor()

    cursor.execute("DELETE FROM api_cache")
    deleted = cursor.rowcount

    conn.commit()
    conn.close()

    return deleted


def _sqlite_stats():
    """Get SQLite cache statistics"""
    import sqlite3
    _init_sqlite_db()

    conn = sqlite3.connect(str(CACHE_DB))
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM api_cache")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM api_cache WHERE expires_at < datetime('now')")
    expired = cursor.fetchone()[0]

    cursor.execute('''
        SELECT data_type, COUNT(*)
        FROM api_cache
        WHERE expires_at > datetime('now')
        GROUP BY data_type
    ''')
    by_type = dict(cursor.fetchall())

    cache_size = CACHE_DB.stat().st_size if CACHE_DB.exists() else 0

    conn.close()

    return {
        'total_entries': total,
        'valid_entries': total - expired,
        'expired_entries': expired,
        'by_type': by_type,
        'cache_size_mb': round(cache_size / (1024 * 1024), 2),
        'storage': 'sqlite'
    }


# ============================================
# Unified Cache Interface
# ============================================

def generate_cache_key(data_type: str, **params) -> str:
    """Generate a unique cache key based on data type and parameters"""
    sorted_params = sorted(params.items())
    key_string = f"{data_type}:{json.dumps(sorted_params, sort_keys=True, default=str)}"
    return hashlib.md5(key_string.encode()).hexdigest()


def get_cached_data(cache_key: str):
    """Retrieve cached data if it exists and hasn't expired"""
    if IS_STREAMLIT_CLOUD:
        return _session_get(cache_key)
    else:
        return _sqlite_get(cache_key)


def set_cached_data(cache_key: str, data, data_type: str,
                    symbol: str = None, ttl_seconds: int = None, metadata: dict = None):
    """Store data in the cache"""
    if IS_STREAMLIT_CLOUD:
        _session_set(cache_key, data, data_type, ttl_seconds)
    else:
        _sqlite_set(cache_key, data, data_type, symbol, ttl_seconds, metadata)


def clear_expired_cache():
    """Remove expired entries from the cache"""
    if IS_STREAMLIT_CLOUD:
        return _session_clear_expired()
    else:
        return _sqlite_clear_expired()


def clear_all_cache():
    """Clear all cached data"""
    if IS_STREAMLIT_CLOUD:
        return _session_clear_all()
    else:
        return _sqlite_clear_all()


def get_cache_stats() -> dict:
    """Get statistics about the cache"""
    if IS_STREAMLIT_CLOUD:
        return _session_stats()
    else:
        return _sqlite_stats()


# ============================================
# Cached API Functions
# ============================================

def cached_tiingo_price(client, symbols: list, start_date, end_date,
                        metric_name: str = 'close', force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch price data from Tiingo with caching.
    """
    import streamlit as st

    cache_key = generate_cache_key(
        'tiingo_price',
        symbols=tuple(sorted(symbols)),
        metric=metric_name,
        start=str(start_date),
        end=str(end_date)
    )

    if not force_refresh:
        cached = get_cached_data(cache_key)
        if cached is not None:
            st.info(f"📦 Using cached Tiingo data for {symbols} ({len(cached)} rows)")
            return cached

    st.info(f"🌐 Fetching fresh Tiingo data for {symbols}...")
    df = client.get_dataframe(
        symbols,
        metric_name=metric_name,
        startDate=start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else start_date,
        endDate=end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else end_date
    )

    set_cached_data(
        cache_key, df, 'price_data',
        symbol=','.join(symbols),
        metadata={'symbols': symbols, 'metric': metric_name}
    )

    return df


def cached_alphavantage_daily(symbol: str, api_key: str, outputsize: str = 'compact',
                               force_refresh: bool = False) -> pd.DataFrame | None:
    """
    Fetch daily price data from Alpha Vantage with caching.
    """
    import streamlit as st
    import requests

    cache_key = generate_cache_key(
        'alphavantage_daily',
        symbol=symbol,
        outputsize=outputsize
    )

    if not force_refresh:
        cached = get_cached_data(cache_key)
        if cached is not None:
            st.info(f"📦 Using cached Alpha Vantage daily data for {symbol}")
            return cached

    crypto_pairs = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'BTC', 'ETH', 'SOL']
    is_crypto = any(crypto in symbol.upper() for crypto in crypto_pairs)

    if is_crypto:
        if 'USD' in symbol:
            base = symbol.replace('USD', '')
            market = 'USD'
        else:
            base = symbol
            market = 'USD'

        url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={base}&market={market}&apikey={api_key}'
    else:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={outputsize}&apikey={api_key}'

    st.info(f"🌐 Fetching fresh Alpha Vantage data for {symbol}...")
    response = requests.get(url)
    data = response.json()

    if 'Error Message' in data:
        st.error(f"Alpha Vantage Error for {symbol}: {data['Error Message']}")
        return None
    if 'Note' in data:
        st.warning(f"Alpha Vantage Rate Limit: {data['Note']}")
        return None
    if 'Information' in data:
        st.warning(f"Alpha Vantage Info: {data['Information']}")
        return None

    df = None
    if is_crypto and 'Time Series (Digital Currency Daily)' in data:
        df = pd.DataFrame.from_dict(data['Time Series (Digital Currency Daily)'], orient='index')
        df.index = pd.to_datetime(df.index)

        close_cols = [col for col in df.columns if 'close' in col.lower()]
        volume_cols = [col for col in df.columns if 'volume' in col.lower()]

        if close_cols and volume_cols:
            df = df[[close_cols[0], volume_cols[0]]].copy()
            df.columns = ['close', 'volume']
            df = df.astype(float)
            df = df.sort_index()
    elif 'Time Series (Daily)' in data:
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df = df.sort_index()

    if df is not None:
        set_cached_data(cache_key, df, 'price_data', symbol=symbol)

    return df


def cached_alphavantage_indicator(symbol: str, api_key: str, function: str,
                                   force_refresh: bool = False, **params) -> pd.DataFrame | None:
    """
    Fetch technical indicator from Alpha Vantage with caching.
    """
    import streamlit as st
    import requests

    cache_key = generate_cache_key(
        'alphavantage_indicator',
        symbol=symbol,
        function=function,
        **params
    )

    if not force_refresh:
        cached = get_cached_data(cache_key)
        if cached is not None:
            st.info(f"📦 Using cached {function} indicator for {symbol}")
            return cached

    param_str = '&'.join([f"{k}={v}" for k, v in params.items()])
    url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&{param_str}&apikey={api_key}'

    st.info(f"🌐 Fetching {function} indicator for {symbol}...")
    response = requests.get(url)
    data = response.json()

    df = None
    for key in data.keys():
        if 'Technical Analysis' in key or key.startswith('Technical'):
            df = pd.DataFrame.from_dict(data[key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df = df.sort_index()
            break

    if df is not None:
        set_cached_data(cache_key, df, 'indicator', symbol=symbol,
                       metadata={'function': function})

    return df


def cached_fred_series(fred, series_id: str, start_date, end_date,
                       force_refresh: bool = False) -> pd.Series | None:
    """
    Fetch FRED economic data with caching.
    """
    import streamlit as st

    cache_key = generate_cache_key(
        'fred_series',
        series_id=series_id,
        start=str(start_date),
        end=str(end_date)
    )

    if not force_refresh:
        cached = get_cached_data(cache_key)
        if cached is not None:
            st.info(f"📦 Using cached FRED data for {series_id}")
            return cached.iloc[:, 0] if isinstance(cached, pd.DataFrame) else cached

    st.info(f"🌐 Fetching FRED data for {series_id}...")
    try:
        series = fred.get_series(series_id, start_date, end_date)

        df = pd.DataFrame({series_id: series})
        set_cached_data(cache_key, df, 'fred_data', symbol=series_id,
                       ttl_seconds=DEFAULT_TTL['fred_data'])

        return series
    except Exception as e:
        st.warning(f"Failed to fetch FRED {series_id}: {e}")
        return None


# ============================================
# High-level cached data fetchers
# ============================================

def get_cached_consolidated_data(tiingo_key: str, fred_key: str, target_asset: str,
                                  market_context: list, macro_context: list,
                                  start_date, end_date, force_refresh: bool = False) -> pd.DataFrame | None:
    """
    Fetch and consolidate market and macro data with intelligent caching.
    """
    import streamlit as st
    from tiingo import TiingoClient
    from fredapi import Fred

    cache_key = generate_cache_key(
        'consolidated_tiingo',
        target=target_asset,
        market=tuple(sorted(market_context)),
        macro=tuple(sorted(macro_context)),
        start=str(start_date),
        end=str(end_date)
    )

    if not force_refresh:
        cached = get_cached_data(cache_key)
        if cached is not None:
            st.success(f"📦 Using cached consolidated data ({len(cached)} rows)")
            return cached

    try:
        client = TiingoClient({'api_key': tiingo_key})
        fred = Fred(api_key=fred_key)

        all_market = [target_asset] + market_context
        df_market = cached_tiingo_price(
            client, all_market, start_date, end_date,
            metric_name='close', force_refresh=force_refresh
        )

        try:
            df_volume = cached_tiingo_price(
                client, [target_asset], start_date, end_date,
                metric_name='volume', force_refresh=force_refresh
            )
            df_volume.columns = [f"{target_asset}_volume"]
            df_market = df_market.join(df_volume)
        except Exception:
            df_market[f"{target_asset}_volume"] = 0

        for macro in macro_context:
            try:
                macro_series = cached_fred_series(
                    fred, macro, start_date, end_date,
                    force_refresh=force_refresh
                )
                if macro_series is not None:
                    macro_series = macro_series.resample('D').ffill()
                    df_market = df_market.join(macro_series.rename(macro), how='left')
                    df_market[macro] = df_market[macro].ffill()
            except Exception:
                pass

        result = df_market.dropna(subset=[target_asset])

        set_cached_data(
            cache_key, result, 'price_data',
            symbol=target_asset,
            metadata={
                'target': target_asset,
                'market_context': market_context,
                'macro_context': macro_context
            }
        )

        st.success(f"✅ Data fetched and cached ({len(result)} rows)")
        return result

    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None


def get_cached_alphavantage_data(av_key: str, fred_key: str, target_asset: str,
                                  market_context: list, macro_context: list,
                                  start_date, end_date, av_indicators: dict,
                                  force_refresh: bool = False) -> pd.DataFrame | None:
    """
    Fetch Alpha Vantage data with intelligent caching.
    """
    import streamlit as st
    from fredapi import Fred

    cache_key = generate_cache_key(
        'consolidated_alphavantage',
        target=target_asset,
        market=tuple(sorted(market_context)),
        macro=tuple(sorted(macro_context)),
        indicators=json.dumps(av_indicators, sort_keys=True),
        start=str(start_date),
        end=str(end_date)
    )

    if not force_refresh:
        cached = get_cached_data(cache_key)
        if cached is not None:
            st.success(f"📦 Using cached Alpha Vantage data ({len(cached)} rows)")
            return cached

    try:
        fred = Fred(api_key=fred_key)

        df_target = cached_alphavantage_daily(
            target_asset, av_key, force_refresh=force_refresh
        )
        if df_target is None:
            st.error(f"Failed to fetch {target_asset} from Alpha Vantage")
            return None

        df_market = df_target[['close', 'volume']].copy()
        df_market.columns = [target_asset, f"{target_asset}_volume"]

        for asset in market_context:
            df_context = cached_alphavantage_daily(
                asset, av_key, force_refresh=force_refresh
            )
            if df_context is not None:
                df_market = df_market.join(
                    df_context[['close']].rename(columns={'close': asset}),
                    how='outer'
                )

        if av_indicators:
            if av_indicators.get('use_av_rsi'):
                df_rsi = cached_alphavantage_indicator(
                    target_asset, av_key, 'RSI',
                    interval='daily', time_period=14, series_type='close',
                    force_refresh=force_refresh
                )
                if df_rsi is not None:
                    df_market = df_market.join(df_rsi.rename(columns={'RSI': 'AV_RSI'}), how='left')

            if av_indicators.get('use_av_macd'):
                df_macd = cached_alphavantage_indicator(
                    target_asset, av_key, 'MACD',
                    interval='daily', series_type='close',
                    force_refresh=force_refresh
                )
                if df_macd is not None:
                    df_market = df_market.join(df_macd.add_prefix('AV_'), how='left')

            if av_indicators.get('use_av_adx'):
                df_adx = cached_alphavantage_indicator(
                    target_asset, av_key, 'ADX',
                    interval='daily', time_period=14,
                    force_refresh=force_refresh
                )
                if df_adx is not None:
                    df_market = df_market.join(df_adx.rename(columns={'ADX': 'AV_ADX'}), how='left')

            if av_indicators.get('use_av_cci'):
                df_cci = cached_alphavantage_indicator(
                    target_asset, av_key, 'CCI',
                    interval='daily', time_period=20,
                    force_refresh=force_refresh
                )
                if df_cci is not None:
                    df_market = df_market.join(df_cci.rename(columns={'CCI': 'AV_CCI'}), how='left')

            if av_indicators.get('use_av_stoch'):
                df_stoch = cached_alphavantage_indicator(
                    target_asset, av_key, 'STOCH',
                    interval='daily',
                    force_refresh=force_refresh
                )
                if df_stoch is not None:
                    df_market = df_market.join(df_stoch.add_prefix('AV_'), how='left')

            if av_indicators.get('use_av_bbands'):
                df_bbands = cached_alphavantage_indicator(
                    target_asset, av_key, 'BBANDS',
                    interval='daily', time_period=20, series_type='close',
                    force_refresh=force_refresh
                )
                if df_bbands is not None:
                    df_market = df_market.join(df_bbands.add_prefix('AV_'), how='left')

        for macro in macro_context:
            try:
                macro_series = cached_fred_series(
                    fred, macro, start_date, end_date,
                    force_refresh=force_refresh
                )
                if macro_series is not None:
                    macro_series = macro_series.resample('D').ffill()
                    df_market = df_market.join(macro_series.rename(macro), how='left')
                    df_market[macro] = df_market[macro].ffill()
            except Exception:
                pass

        result = df_market.dropna()

        set_cached_data(
            cache_key, result, 'price_data',
            symbol=target_asset,
            metadata={
                'target': target_asset,
                'market_context': market_context,
                'indicators': av_indicators
            }
        )

        st.success(f"✅ Alpha Vantage data fetched and cached ({len(result)} rows)")
        return result

    except Exception as e:
        st.error(f"Alpha Vantage data fetch error: {str(e)}")
        return None
