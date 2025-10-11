#!/bin/bash

# Update and setup script for lumogun application
# This handles internet check, git operations, and dependency installation

LOG_FILE="/home/lumotag/setup_update.log"
CONFIG_FILE="/home/lumotag/MY_INFO.txt"
BASE_DIR="/home/lumotag"

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

# Internet connectivity check (translated from bootstrap.py)
check_internet() {
    local websites=("http://www.google.com" "http://www.amazon.com" "http://www.microsoft.com" "http://www.apple.com" "http://www.cloudflare.com")
    
    for site in "${websites[@]}"; do
        if curl -s --connect-timeout 1 "$site" > /dev/null 2>&1; then
            log "Internet connection verified via $site"
            return 0
        fi
    done
    return 1
}

# Read configuration from MY_INFO.txt (like bootstrap.py)
read_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        log "ERROR: Configuration file not found: $CONFIG_FILE"
        return 1
    fi
    
    # Parse JSON config file
    MY_ID=$(python3 -c "import json; data=json.load(open('$CONFIG_FILE')); print(data['MY_ID'])" 2>/dev/null)
    HQ_URL=$(python3 -c "import json; data=json.load(open('$CONFIG_FILE')); print(data['HQ'])" 2>/dev/null)
    REPO_URL=$(python3 -c "import json; data=json.load(open('$CONFIG_FILE')); print(data['REPO'])" 2>/dev/null)
    CODE_PATH=$(python3 -c "import json; data=json.load(open('$CONFIG_FILE')); print(data['CODEPATH'])" 2>/dev/null)
    
    if [ -z "$MY_ID" ] || [ -z "$REPO_URL" ] || [ -z "$CODE_PATH" ]; then
        log "ERROR: Failed to parse configuration file"
        return 1
    fi
    
    log "Configuration loaded - MY_ID: $MY_ID, REPO: $REPO_URL"
    return 0
}

# Rotate log if needed
rotate_log

log "Starting update and setup process..."

# Check internet connection with limited retries
# Each check tries 5 sites with 1s timeout = max 5s per check
# 2 retries: (5s check + 5s sleep) + (5s check) = ~15 seconds max
log "Checking internet connection..."
MAX_RETRIES=2
RETRY_COUNT=0
HAS_INTERNET=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if check_internet; then
        HAS_INTERNET=true
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        log "No internet connection. Retry $RETRY_COUNT/$MAX_RETRIES in 5 seconds..."
        sleep 5
    fi
done

if [ "$HAS_INTERNET" = false ]; then
    log "WARNING: No internet connection after $MAX_RETRIES attempts"
fi

# Read configuration
if ! read_config; then
    log "ERROR: Failed to read configuration"
    exit 1
fi

# Check if we can proceed without internet
APP_DIR="$CODE_PATH/Source/lumotag"
VENV_PATH="$APP_DIR/lumotagvenv"

if [ "$HAS_INTERNET" = false ]; then
    # No internet - check if we have existing setup
    if [ ! -d "$CODE_PATH" ] || [ ! -d "$APP_DIR" ] || [ ! -d "$VENV_PATH" ]; then
        log "ERROR: No internet and missing required files. Cannot proceed with first-time setup."
        exit 1
    fi
    log "No internet but existing installation found. Skipping all updates."
    exit 0
fi

# Has internet - proceed with git updates
log "Internet available. Proceeding with updates..."
cd "$BASE_DIR" || {
    log "ERROR: Failed to change to base directory: $BASE_DIR"
    exit 1
}

if [ -d "$CODE_PATH" ]; then
    log "Repository exists, attempting to pull updates..."
    cd "$CODE_PATH" || {
        log "ERROR: Failed to change to repository directory: $CODE_PATH"
        exit 1
    }
    
    if git pull --ff-only >> "$LOG_FILE" 2>&1; then
        log "Successfully pulled updates from repository"
    else
        log "WARNING: Git pull failed, but continuing..."
    fi
else
    log "Repository does not exist, cloning from $REPO_URL..."
    if git clone "$REPO_URL" >> "$LOG_FILE" 2>&1; then
        log "Successfully cloned repository"
        git config --global --add safe.directory "$CODE_PATH" >> "$LOG_FILE" 2>&1
    else
        log "ERROR: Failed to clone repository"
        exit 1
    fi
fi

# Install dependencies
if [ ! -d "$APP_DIR" ]; then
    log "ERROR: Application directory does not exist: $APP_DIR"
    exit 1
fi

cd "$APP_DIR" || {
    log "ERROR: Failed to change to application directory: $APP_DIR"
    exit 1
}

log "Setting up virtual environment and dependencies..."

# Create virtual environment with system site packages
if ! uv venv lumotagvenv --system-site-packages --clear >> "$LOG_FILE" 2>&1; then
    log "ERROR: Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment and install dependencies
source "$VENV_PATH/bin/activate" || {
    log "ERROR: Failed to activate virtual environment"
    exit 1
}

if uv pip install -r pyproject.toml >> "$LOG_FILE" 2>&1; then
    log "Successfully installed dependencies"
else
    log "ERROR: Failed to install dependencies"
    deactivate
    exit 1
fi

deactivate
log "Setup and update process completed successfully"
