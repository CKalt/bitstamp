# Auto Trading with Diagnostic Tracking - User Guide

## Overview

This guide explains how to run the auto-trading system with comprehensive diagnostic tracking that helps debug issues and provides detailed logs for troubleshooting.

## Starting Auto Trading

### Basic Commands

```bash
# Start auto trading with your balance
auto_trade 2.47btc long      # If you have BTC and want to trade long
auto_trade 234462usd short    # If you have USD and want to trade short

# Check status
status                        # Quick position summary
status long                   # Detailed status with all metrics

# Stop trading
stop_auto_trade
```

### What Happens When You Start

1. **Strategy Selection**: Loads `best_strategy.json` configuration
2. **Diagnostic File Creation**: Creates `diagnostics_ADAPTIVE_MULTI_YYYYMMDD_HHMMSS.json`
3. **Position Initialization**: Sets up your starting position (long/short)
4. **Real-time Monitoring**: Begins watching for signals and market changes

## Diagnostic System

### Automatic Logging

The system automatically creates a JSON diagnostic file that tracks:

- **Session Events**: Start/end times, parameters used
- **Signal Evaluations**: Every trading signal considered and why it was/wasn't executed
- **Trades**: All executed trades with position changes and P&L
- **Market Regime Changes**: When strategy switches between trending/ranging/volatile
- **Position Anomalies**: Unusual situations like entry price > 1.5x current price
- **Errors**: Any exceptions or problems encountered
- **Periodic Snapshots**: System