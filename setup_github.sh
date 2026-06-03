#!/usr/bin/env bash
# =============================================================================
# setup_github.sh — One-time GitHub setup for BSE Volatility Forecasting
#
# Run from inside the project folder:
#   cd stock-market-volatility-forecasting
#   bash setup_github.sh
#
# Prerequisites:
#   1. Create an empty GitHub repo at https://github.com/new
#      (public or private, no README / .gitignore — we push our own)
#   2. Have your GitHub username and a Personal Access Token (PAT) ready.
#      Create a PAT at: https://github.com/settings/tokens
#      (Scopes needed: repo — full control of private repositories)
# =============================================================================

set -e  # exit on error

# ── 1. Initialise git ────────────────────────────────────────────────────────
if [ ! -d ".git" ]; then
    git init
    echo "✓ Git initialised"
else
    echo "✓ Git already initialised"
fi

# ── 2. Set identity (update with your details) ───────────────────────────────
# git config user.name  "Your Name"
# git config user.email "you@example.com"

# ── 3. Create .gitignore additions specific to this project ──────────────────
echo "✓ .gitignore already set up"

# ── 4. Stage all project files ───────────────────────────────────────────────
git add .
git status --short

# ── 5. Initial commit ────────────────────────────────────────────────────────
git commit -m "Initial commit: BSE volatility forecasting project structure"
echo "✓ Initial commit created"

# ── 6. Add GitHub remote ─────────────────────────────────────────────────────
# Replace the URL below with your actual GitHub repo URL
# Format: https://github.com/<your-username>/<your-repo-name>.git

echo ""
echo "Next: paste the URL of your GitHub repo and run:"
echo ""
echo "  git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPO>.git"
echo "  git branch -M main"
echo "  git push -u origin main"
echo ""
echo "When prompted for password, use your PAT (not your GitHub password)."
