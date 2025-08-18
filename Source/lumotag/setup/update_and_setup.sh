#!/bin/bash

# Update and setup script for lumogun application
# This handles internet check, git operations, and dependency installation

LOG_FILE="/home/lumotag/setup_update.log"
CONFIG_FILE="/home/lumotag/MY_INFO.txt"
BASE_DIR="/home/lumotag"

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

log "Starting update and setup process..."

# Check internet connection with retry logic
log "Checking internet connection..."
while ! check_internet; do
    log "No internet connection. Retrying in 5 seconds..."
    sleep 5
done

# Read configuration
if ! read_config; then
    log "ERROR: Failed to read configuration"
    exit 1
fi

# Handle git operations (pull or clone)
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
    
    if sudo git pull --ff-only; then
        log "Successfully pulled updates from repository"
    else
        log "WARNING: Git pull failed, but continuing..."
    fi
else
    log "Repository does not exist, cloning from $REPO_URL..."
    if sudo git clone "$REPO_URL"; then
        log "Successfully cloned repository"
        # Set safe directory
        sudo git config --global --add safe.directory "$CODE_PATH"
    else
        log "ERROR: Failed to clone repository"
        exit 1
    fi
fi

# Install dependencies (from experimental_setup.sh pattern)
APP_DIR="$CODE_PATH/Source/lumotag"
VENV_PATH="$APP_DIR/lumotagvenv"

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
if ! uv venv lumotagvenv --system-site-packages --clear ; then
    log "ERROR: Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment and install dependencies
source "$VENV_PATH/bin/activate" || {
    log "ERROR: Failed to activate virtual environment"
    exit 1
}

if uv pip install -r pyproject.toml; then
    log "Successfully installed dependencies"
else
    log "ERROR: Failed to install dependencies"
    deactivate
    exit 1
fi

deactivate
log "Setup and update process completed successfully"
