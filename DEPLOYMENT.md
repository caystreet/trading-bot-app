# 🚀 Deployment Guide

## Your app is ready! Here's how to deploy it:

### 📁 Files Location
All your files are in: `/Users/keithryan/Documents/claude/trading-bot-app/`

---

## Option 1: Deploy to Streamlit Cloud (Recommended - FREE!)

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Fill in:
   - Repository name: `trading-bot-app`
   - Description: `ML-powered crypto trading bot with Streamlit`
   - Make it **Public**
   - **Don't** add README, .gitignore, or license (we already have them)
3. Click **Create repository**

### Step 2: Push Your Code to GitHub

Open Terminal and run these commands:

```bash
cd ~/Documents/claude/trading-bot-app

# Add the GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/trading-bot-app.git

# Push your code
git push -u origin main
```

**Note**: If it asks for credentials, you may need to:
- Create a Personal Access Token at https://github.com/settings/tokens
- Use the token as your password when pushing

### Step 3: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click **"New app"**
3. Connect your GitHub account (if not already connected)
4. Select:
   - Repository: `YOUR_USERNAME/trading-bot-app`
   - Branch: `main`
   - Main file path: `app.py`
5. Click **"Deploy"**

**That's it!** Your app will be live in 2-3 minutes at a URL like:
`https://YOUR_USERNAME-trading-bot-app.streamlit.app`

---

## Option 2: Run Locally (Quick Test)

Want to test it first? Run locally:

```bash
cd ~/Documents/claude/trading-bot-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

It will open at `http://localhost:8501`

---

## Option 3: Deploy to Other Platforms

### Heroku
1. Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```
2. Deploy:
```bash
heroku create trading-bot-app
git push heroku main
```

### Render.com
1. Connect your GitHub repo
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `streamlit run app.py`

### Railway.app
1. Connect GitHub repo
2. Add environment variables (if needed)
3. Deploy automatically

---

## 🔑 Important: API Keys

The default API keys in the app are:
- Tiingo: `e30bc8b0b855970d930743b116e517e36b72cc8f`
- FRED: `1c1c04541bfefa93e9cb0db265da9c19`

**For production**, you should:
1. Get your own FREE API keys:
   - Tiingo: https://www.tiingo.com/
   - FRED: https://fred.stlouisfed.org/docs/api/api_key.html

2. Remove default keys from `app.py` and use Streamlit secrets instead

3. In Streamlit Cloud:
   - Go to App settings → Secrets
   - Add:
```toml
TIINGO_KEY = "your_key_here"
FRED_KEY = "your_key_here"
```

4. Update `app.py` to use secrets:
```python
tiingo_key = st.secrets.get("TIINGO_KEY", "")
fred_key = st.secrets.get("FRED_KEY", "")
```

---

## 🐛 Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### Git push fails
- Create a Personal Access Token: https://github.com/settings/tokens
- Use token as password when pushing

### Streamlit Cloud build fails
- Check requirements.txt has all dependencies
- Make sure app.py is in the root directory
- Check Streamlit Cloud logs for specific errors

### API rate limit errors
- Get your own API keys (free)
- Reduce the number of market context symbols
- Use shorter date ranges

---

## 📊 What Next?

Once deployed, your app will:
- ✅ Fetch live crypto data
- ✅ Train ML models on historical data
- ✅ Generate trading signals
- ✅ Show backtest results
- ✅ Display interactive charts

Share your live URL with anyone!

---

## 🎯 Quick Commands Summary

```bash
# Navigate to app
cd ~/Documents/claude/trading-bot-app

# Test locally
streamlit run app.py

# Deploy to GitHub
git remote add origin https://github.com/YOUR_USERNAME/trading-bot-app.git
git push -u origin main

# Then deploy on https://share.streamlit.io/
```

---

Need help? Check:
- Streamlit Docs: https://docs.streamlit.io/
- Tiingo API: https://www.tiingo.com/documentation/general/overview
- FRED API: https://fred.stlouisfed.org/docs/api/fred/
