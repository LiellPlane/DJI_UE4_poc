#!/bin/bash

# Launcher script for lumogun application
# This handles environment setup and error logging

LOG_FILE="/home/lumotag/app_launch.log"
APP_DIR="/home/lumotag/DJI_UE4_poc/Source/lumotag"
VENV_PATH="$APP_DIR/lumotagvenv"

# Rotate log file if it gets too large (>10MB)
rotate_log() {
    if [ -f "$LOG_FILE" ] && [ $(stat -c%s "$LOG_FILE") -gt 10485760 ]; then
        # Remove oldest backup
        rm -f "${LOG_FILE}.old.1" 2>/dev/null
        # Move previous backup to oldest
        [ -f "${LOG_FILE}.old" ] && mv "${LOG_FILE}.old" "${LOG_FILE}.old.1"
        # Move current log to previous backup
        mv "$LOG_FILE" "${LOG_FILE}.old"
    fi
}

# Log function
log() {
    echo "$(date): $1" >> "$LOG_FILE"
    echo "$1"  # Also print to stdout
}

# Rotate log if needed
rotate_log

log "Starting lumogun launcher..."

# Run update and setup script first
UPDATE_SCRIPT="/home/lumotag/update_and_setup.sh"
if [ -f "$UPDATE_SCRIPT" ]; then
    log "Running update and setup script..."
    if bash "$UPDATE_SCRIPT"; then
        log "Update and setup completed successfully"
    else
        log "ERROR: Update and setup script failed, cannot continue"
        exit 1
    fi
else
    log "ERROR: Update script not found at $UPDATE_SCRIPT"
    exit 1
fi

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
log "Launching lumogun.py..."
if python lumogun.py >> "$LOG_FILE" 2>&1; then
    log "Application exited successfully"
else
    log "ERROR: Application exited with error code $?"
    exit 1
fi

# Deactivate virtual environment
deactivate

log "Lumogun launcher completed"