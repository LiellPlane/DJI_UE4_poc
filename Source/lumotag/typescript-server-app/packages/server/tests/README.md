# Destruction Testing with k6

## Quick Start

1. **Install k6:**
   ```bash
   winget install k6  # Windows
   ```

2. **Start your server:**
   ```bash
   npm run dev  # Make sure it's running on localhost:8080
   ```

3. **Run destruction test:**
   ```bash
   cd tests
   k6 run gamestate-destruction-test.js
   ```

## Get Graphs

### Option 1: Grafana Cloud k6 (Free)
1. **Sign up** at https://grafana.com/products/cloud/k6/
2. **Get API token** from your dashboard
3. **Login and run:**
   ```bash
   k6 cloud login --token YOUR_TOKEN
   k6 cloud run gamestate-destruction-test.js
   ```

### Option 2: Local Console Output
Just run the basic test - it shows response times, error rates, and RPS in the console output.

## What It Does

- Ramps up from 5 → 1000 concurrent users
- Hammers `/api/v1/gamestate` endpoint
- Shows where your server breaks
- Free tier: 50 tests/month

## Expected Results

- **200-300 users**: Performance degradation starts
- **500+ users**: Significant slowdown/errors  
- **1000 users**: Likely complete failure

## Files

- `gamestate-destruction-test.js` - Main destruction test script