#!/bin/bash

# Script to push your trading bot app to GitHub

echo "🚀 Pushing Trading Bot App to GitHub..."
echo ""

cd ~/Documents/claude/trading-bot-app

# Check if we're in a git repo
if [ ! -d .git ]; then
    echo "❌ Error: Not a git repository"
    exit 1
fi

# Add remote if not already added
if ! git remote | grep -q "origin"; then
    echo "📡 Adding GitHub remote..."
    git remote add origin https://github.com/caystreet/trading-bot-app.git
else
    echo "✅ GitHub remote already exists"
fi

# Push to GitHub
echo "⬆️  Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS! Code pushed to GitHub!"
    echo "🔗 View at: https://github.com/caystreet/trading-bot-app"
    echo ""
    echo "Next step: Deploy to Streamlit Cloud"
    echo "👉 Go to: https://share.streamlit.io/"
else
    echo ""
    echo "❌ Push failed. You may need to:"
    echo "   1. Set up GitHub authentication"
    echo "   2. Create a Personal Access Token at: https://github.com/settings/tokens"
    echo "   3. Use the token as your password when git asks"
fi
