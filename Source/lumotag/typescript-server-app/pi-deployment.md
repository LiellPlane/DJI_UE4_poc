# Raspberry Pi 5 Simple Deployment Guide

## Overview
Simple manual deployment approach with good visibility. Perfect for hobby projects where you want to see what's happening.

## One-Time Setup on Raspberry Pi

### 1. Initial Setup (Run Once)
```bash
# SSH to your Pi
ssh pi@your-pi-ip

# Run setup script
bash scripts/setup-pi.sh https://github.com/your-username/typescript-server-app.git
```

This installs Node.js, clones your repo, and builds the app.

## Your Simple Development Workflow

### 1. Develop on Your Mac (Normal)
```bash
# On your Mac - normal development
git add .
git commit -m "Add feature"
git push origin main
```

### 2. Update Pi (Manual & Visible)
```bash
# Power cycle approach
ssh pi@your-pi-ip "cd /home/pi/typescript-app && bash scripts/update-and-run.sh"

# Or step by step
ssh pi@your-pi-ip
cd /home/pi/typescript-app
bash scripts/deploy-pi.sh  # Pull & build
pnpm run start:prod        # Start app
```

### 3. Power Cycle Updates
When you restart the Pi, just run:
```bash
bash scripts/update-and-run.sh
```

This pulls latest code, builds, and starts the app.

## Simple Management

### Basic Commands
```bash
# Update and run (main command)
bash scripts/update-and-run.sh

# Just update (don't start)
bash scripts/deploy-pi.sh

# Start app manually
pnpm run start:prod

# Check if app is responding
curl http://localhost:3000/api/health
```

### Monitoring
```bash
# Check system resources
htop

# Check Pi temperature
vcgencmd measure_temp

# Check disk space
df -h

# Check memory usage
free -h
```

### Troubleshooting
```bash
# Check git status
cd /home/pi/typescript-app
git status
git log --oneline -3

# Verify build exists
ls -la packages/server/dist/

# Manual rebuild if needed
pnpm install --frozen-lockfile
pnpm run build

# Test manual start
node packages/server/dist/app.js
```

## Tips for Pi 5

- **Memory**: Pi 5 has plenty of RAM for Node.js apps
- **Storage**: Use good quality SD card (Class 10+) or SSD
- **Temperature**: Monitor with `vcgencmd measure_temp`
- **Network**: App runs on port 3000 by default

## Future Automation Options

When you want to add automation later:
- **Power-on auto-start**: Add to Pi startup scripts
- **Periodic updates**: Cron job every few minutes  
- **Instant deploys**: Git webhooks
- **Service management**: systemd service (placeholder ready)
