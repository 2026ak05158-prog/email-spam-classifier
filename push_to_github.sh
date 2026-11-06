#!/usr/bin/env bash
# Helper script to link and push your assignment repo to GitHub

if [ -z "$1" ]; then
    echo "Usage: ./push_to_github.sh <YOUR_GITHUB_REPO_URL>"
    echo "Example: ./push_to_github.sh https://github.com/your-username/email-spam-classifier-bits.git"
    exit 1
fi

REPO_URL=$1
echo "Setting remote origin to $REPO_URL..."
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"
git branch -M main

echo "Pushing code to main branch..."
git push -u origin main

echo ""
echo "================================================================"
echo "Successfully pushed! Next step: Deploy on Streamlit Cloud"
echo "1. Open https://streamlit.io/cloud"
echo "2. Click 'New App'"
echo "3. Select this repository and 'app.py', then click 'Deploy'"
echo "================================================================"
