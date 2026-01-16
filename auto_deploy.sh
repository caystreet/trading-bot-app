#!/bin/bash

# Automated deployment script for Trading Bot App

echo "🤖 Trading Bot App - Automated Deployment"
echo "=========================================="
echo ""

# Navigate to project directory
cd ~/Documents/claude/trading-bot-app || exit 1

# Step 1: Clean up any git locks
echo "🧹 Cleaning up git locks..."
rm -f .git/index.lock .git/HEAD.lock .git/refs/heads/main.lock 2>/dev/null
rm -rf .git/objects/*/tmp_* 2>/dev/null
echo "✅ Cleaned"
echo ""

# Step 2: Add all files
echo "📦 Adding all files to git..."
git add .
echo "✅ Files added"
echo ""

# Step 3: Commit
echo "💾 Committing changes..."
git commit -m "Complete trading bot app with Streamlit UI

- Streamlit web interface
- Random Forest and KNN models
- Live trading signals
- Historical backtesting
- Interactive charts
- Full documentation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>" 2>/dev/null || echo "No changes to commit"
echo "✅ Committed"
echo ""

# Step 4: Push to GitHub
echo "⬆️  Pushing to GitHub..."
echo ""
echo "🔑 IMPORTANT: When prompted for password:"
echo "   1. Go to: https://github.com/settings/tokens/new"
echo "   2. Click 'Generate new token (classic)'"
echo "   3. Name it: 'Trading Bot App'"
echo "   4. Check the 'repo' box"
echo "   5. Click 'Generate token'"
echo "   6. Copy the token and paste it as your password"
echo ""
echo "Username: caystreet"
echo ""

git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS! Code is on GitHub!"
    echo "🔗 View at: https://github.com/caystreet/trading-bot-app"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📱 FINAL STEP: Deploy to Streamlit Cloud"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1. Open: https://share.streamlit.io/"
    echo "2. Click: 'New app'"
    echo "3. Fill in:"
    echo "   Repository: caystreet/trading-bot-app"
    echo "   Branch: main"
    echo "   File: app.py"
    echo "4. Click: 'Deploy!'"
    echo ""
    echo "Your app will be live at:"
    echo "🚀 https://caystreet-trading-bot-app.streamlit.app"
    echo ""

    # Try to open Streamlit Cloud in browser
    if command -v open &> /dev/null; then
        echo "Opening Streamlit Cloud in your browser..."
        open https://share.streamlit.io/
    elif command -v xdg-open &> /dev/null; then
        xdg-open https://share.streamlit.io/
    fi
else
    echo ""
    echo "❌ Push failed!"
    echo ""
    echo "📋 Manual steps:"
    echo "1. Create token: https://github.com/settings/tokens/new"
    echo "2. Check 'repo' scope"
    echo "3. Run: git push -u origin main"
    echo "4. Use token as password"
fi
