#!/bin/bash
# Simple update script - pulls latest code and rebuilds
# Run on Pi: bash scripts/deploy-pi.sh

echo "🔄 Updating application..."

cd /home/pi/typescript-app

# Check current commit
OLD_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

# Pull latest
echo "📡 Pulling from git..."
git fetch origin
git reset --hard origin/main

# Check if anything changed
NEW_COMMIT=$(git rev-parse HEAD)
if [ "$OLD_COMMIT" = "$NEW_COMMIT" ]; then
    echo "✅ No changes detected"
    exit 0
fi

echo "📦 Changes detected, rebuilding..."
echo "📋 Installing dependencies..."
pnpm install --frozen-lockfile

echo "🔨 Building..."
pnpm run build

echo "✅ Update complete!"
echo "💡 Start with: pnpm run start:prod"
