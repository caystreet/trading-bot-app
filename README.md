# 🤖 Crypto Trading Bot Tester

A machine learning-powered web application for generating cryptocurrency trading signals using Random Forest and K-Nearest Neighbors models.

## Features

- **Live Trading Signals**: Get real-time BUY/WAIT signals for cryptocurrencies
- **Dual Model Analysis**: Compare predictions from Random Forest and KNN models
- **Historical Backtesting**: Evaluate model performance over 30, 60, 90 days and overall
- **Interactive Visualization**: View 30-day price action charts
- **Customizable Parameters**: Adjust model hyperparameters and market context
- **Multiple Data Sources**: Integrates Tiingo (market data) and FRED (macro indicators)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Tiingo API key (free at https://www.tiingo.com/)
- FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)

### Setup Steps

1. **Clone or download this repository**

2. **Navigate to the project directory**
```bash
cd trading_bot_app
```

3. **Create a virtual environment (recommended)**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

1. **Start the Streamlit app**
```bash
streamlit run app.py
```

2. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to that URL

### Using the Application

1. **Configure API Keys** (sidebar)
   - Enter your Tiingo API key
   - Enter your FRED API key
   - Default keys are provided but you should use your own

2. **Set Trading Parameters**
   - Choose target asset (BTCUSD, ETHUSD, SOLUSD)
   - Select market context symbols (MSTR, SPY, IEA, TLT)
   - Select macro indicators (FEDFUNDS, CPIAUCSL, DGS10)

3. **Configure Models**
   - Enable/disable Random Forest and KNN models
   - Adjust hyperparameters:
     - Random Forest: number of trees, max depth
     - KNN: number of neighbors

4. **Run Analysis**
   - Click the "🚀 Run Analysis" button
   - Wait for data fetching and model training
   - View results in three tabs:
     - **Live Signal**: Current trading recommendations
     - **Backtest Results**: Historical performance metrics
     - **Data Visualization**: Price charts and raw data

## Understanding the Results

### Signal Types

- **✅ BUY SIGNAL**: Model predicts 8.5%+ gain within 40 candles (days)
- **⛔ NO SIGNAL (WAIT)**: Model recommends waiting for better entry

### Confidence Scores

- Higher confidence (>70%) = stronger signal
- Lower confidence (<60%) = uncertain, use caution

### Backtest Metrics

- **Total Signals**: Number of buy signals generated
- **Win Rate**: Percentage of trades that hit 8.5% take profit
- Target: 8.5% take profit, 3% stop loss
- Time horizon: 40 candles maximum per trade

### Performance Periods

- **Overall**: All historical data from 2018
- **90D/60D/30D**: Recent performance (may indicate model drift)

## Configuration Options

### Supported Assets

- BTCUSD (Bitcoin)
- ETHUSD (Ethereum)
- SOLUSD (Solana)

### Market Context Symbols

- MSTR (MicroStrategy - Bitcoin proxy)
- SPY (S&P 500 ETF)
- IEA (iShares U.S. Energy)
- TLT (Treasury Bonds)

### Macro Indicators (FRED)

- FEDFUNDS (Federal Funds Rate)
- CPIAUCSL (Consumer Price Index)
- DGS10 (10-Year Treasury Rate)

## Technical Details

### Models

**Random Forest Classifier**
- Ensemble learning method
- Uses multiple decision trees
- Better for non-linear patterns
- Provides probability estimates

**K-Nearest Neighbors (KNN)**
- Instance-based learning
- Looks at similar historical patterns
- Simpler, more interpretable
- Good for pattern recognition

### Features

The models use these engineered features:
- `Mom_30`: 30-day momentum indicator
- `Mom_14`: 14-day momentum indicator
- `Smoothed_30`: 30-day rolling average of momentum
- `Smoothed_10`: 10-day rolling average of momentum
- Market context prices (MSTR, SPY, etc.)
- Macro indicators (Fed Funds, CPI, etc.)

### Trading Logic

- **Entry**: Model predicts BUY (class 1)
- **Take Profit**: +8.5% gain
- **Stop Loss**: -3% loss
- **Time Limit**: 40 candles (days)

## Troubleshooting

### API Key Errors

**Error**: "Invalid API key"
- Solution: Get free keys from Tiingo and FRED websites
- Make sure keys are copied correctly (no spaces)

### Data Fetching Issues

**Error**: "Error fetching data"
- Check your internet connection
- Verify API keys are valid and have quota remaining
- Some assets may not be available on Tiingo

### Module Not Found

**Error**: "No module named 'streamlit'"
- Solution: Run `pip install -r requirements.txt`
- Make sure virtual environment is activated

### Slow Performance

- First run is slower (fetching historical data)
- Data is cached for 1 hour
- Reduce date range if needed
- Use fewer market context symbols

## API Rate Limits

- **Tiingo Free**: 500 requests/hour, 50k requests/month
- **FRED Free**: Unlimited for personal use
- Data is cached to minimize requests

## Disclaimer

**IMPORTANT**: This tool is for educational and research purposes only.

- Not financial advice
- Past performance doesn't guarantee future results
- Cryptocurrency trading involves substantial risk
- Always do your own research (DYOR)
- Never invest more than you can afford to lose
- Backtest results may not reflect real trading conditions

## License

MIT License - Free to use and modify

## Support

For issues or questions:
- Check the troubleshooting section
- Review API documentation (Tiingo, FRED)
- Verify all dependencies are installed

## Future Enhancements

Potential improvements:
- Add more ML models (XGBoost, LSTM)
- Support for more cryptocurrencies
- Real-time price updates
- Trading simulation with paper trading
- Email/SMS alerts for signals
- Advanced risk management parameters
- Portfolio optimization

## Credits

- Data: Tiingo & FRED APIs
- Framework: Streamlit
- ML: scikit-learn
- Original concept: Colab notebook

---

**Happy Trading! 🚀📈**

Remember: Use responsibly and always manage your risk!
