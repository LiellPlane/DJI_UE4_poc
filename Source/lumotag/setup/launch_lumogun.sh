#!/bin/bash

# Launcher script for lumogun application
# This handles environment setup and error logging

LOG_FILE="/home/lumotag/app_launch.log"
APP_DIR="/home/lumotag/DJI_UE4_poc/Source/lumotag"
VENV_PATH="$APP_DIR/lumotagvenv"

# Log function
log() {
    echo "$(date): $1" >> "$LOG_FILE"
}

log "Starting lumogun launcher..."

# Check if directory exists
if [ ! -d "$APP_DIR" ]; then
    log "ERROR: Application directory does not exist: $APP_DIR"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    log "ERROR: Virtual environment does not exist: $VENV_PATH"
    exit 1
fi

# Check if lumogun.py exists
if [ ! -f "$APP_DIR/lumogun.py" ]; then
    log "ERROR: lumogun.py does not exist in $APP_DIR"
    exit 1
fi

# Change to application directory
cd "$APP_DIR" || {
    log "ERROR: Failed to change to directory $APP_DIR"
    exit 1
}

# Activate virtual environment
source "$VENV_PATH/bin/activate" || {
    log "ERROR: Failed to activate virtual environment"
    exit 1
}

log "Environment setup complete, starting application..."

# Run the application
python lumogun.py >> "$LOG_FILE" 2>&1
