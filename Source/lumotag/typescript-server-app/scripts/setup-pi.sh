#!/bin/bash
# Simple Pi setup - installs Node.js, clones repo, builds app
# Run once: bash scripts/setup-pi.sh https://github.com/your-username/repo.git

echo "🚀 Simple Pi Setup"
echo ""

REPO_URL=${1:-""}
APP_DIR="/home/pi/typescript-app"

if [ -z "$REPO_URL" ]; then
    echo "Usage: bash scripts/setup-pi.sh <git-repo-url>"
    echo "Example: bash scripts/setup-pi.sh https://github.com/username/typescript-server-app.git"
    exit 1
fi

echo "📦 Installing Node.js and dependencies..."
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs build-essential
sudo npm install -g pnpm@8

echo "📡 Cloning repository..."
git clone "$REPO_URL" "$APP_DIR"

echo "🔨 Building application..."
cd "$APP_DIR"
pnpm install --frozen-lockfile
pnpm run build

echo ""
echo "✅ Setup complete!"
echo "🌐 Start app: cd $APP_DIR && pnpm run start:prod"
echo "🔄 Update app: bash scripts/update-and-run.sh"
