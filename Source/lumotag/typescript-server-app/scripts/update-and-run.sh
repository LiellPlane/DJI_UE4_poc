#!/bin/bash
# Update from git and run the app
# Perfect for power cycle updates - run this on Pi startup

echo "🚀 Update and Run Script"

# Update the app
bash scripts/deploy-pi.sh

# If update was successful, start the app
if [ $? -eq 0 ]; then
    echo "🌐 Starting application..."
    pnpm run start:prod
else
    echo "❌ Update failed, not starting app"
    exit 1
fi
