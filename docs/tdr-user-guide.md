# TDR (Trading Data Relay) User Guide

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Client-Server Model](#client-server-model)
5. [Interactive vs Command System](#interactive-vs-command-system)
6. [Historical Data Processing](#historical-data-processing)
7. [Backtesting System](#backtesting-system)
8. [Trading Strategies](#trading-strategies)
9. [AdaptiveMultiStrategy System](#adaptivemultistrategy-system)
10. [Dynamic Pivot Protection](#dynamic-pivot-protection)
11. [Command Reference](#command-reference)
12. [Troubleshooting](#troubleshooting)
13. [Best Practices](#best-practices)

## Overview

TDR is a sophisticated Bitcoin trading system that operates in a client-server architecture with real-time data processing, adaptive strategy selection, and comprehensive backtesting capabilities. The system is designed for algorithmic trading with full position reversals (100% LONG ↔ 100% SHORT) and advanced risk management.

### Key Features
- **Client-Server Architecture**: Separate client and server processes for reliability
- **Adaptive Strategy Selection**: Automatically switches between TRENDING, RANGING, and VOLATILE strategies
- **Dynamic Pivot Protection**: Advanced profit protection and quick re-entry system
- **Real-time Data Processing**: Continuous data feed with WebSocket integration
- **Comprehensive Backtesting**: Historical simulation with multiple timeframes
- **Claude Integration**: AI-powered monitoring and analysis via command files

## System Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   TDR Client    │◄──►│   TDR Server    │◄──►│  Bitstamp API   │
│  (Local)        │    │  (chriskoin)    │    │   (WebSocket)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Command Files   │    │ Trading Engine  │    │ Historical Data │
│ (Claude Interface)│   │ (Strategies)    │    │ (btcusd.log)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Charting System │
                       │ (Dash/Plotly)   │
                       └─────────────────┘
```

### Process Flow
1. **Data Collection**: `websock-ticker2.py` continuously writes price data to `btcusd.log`
2. **Server Processing**: TDR server loads historical data and processes live WebSocket feeds
3. **Strategy Execution**: AdaptiveMultiStrategy analyzes market conditions and executes trades
4. **Client Interface**: TDR client provides command interface and Claude integration
5. **Risk Management**: Dynamic pivot protection monitors positions for profit protection

## Installation & Setup

### Prerequisites
- Python 3.8+
- Virtual environment setup
- SSH access to remote server (chriskoin)
- Bitstamp API credentials

### Local Machine Setup
```bash
cd /Users/chris/projects/python/btc
source env/bin/activate
pip install -r requirements.txt
```

### Remote Server Setup (chriskoin)
```bash
ssh chriskoin
cd /home/chris/projects/bitstamp/
source env/bin/activate
# Ensure websock-ticker2.py is running
python src/websock-ticker2.py &
```

### SSH Tunneling
Essential for client-server communication:
```bash
# Basic tunnel
ssh -L 4000:localhost:4000 chriskoin

# Full tunnel with charting
ssh -L 4000:localhost:4000 -L 8050:localhost:8050 chriskoin
```

## Client-Server Model

### Server (chriskoin)
**Location**: `/home/chris/projects/bitstamp/`

**Responsibilities**:
- Historical data management (`btcusd.log`)
- Real-time WebSocket data processing
- Trading strategy execution
- Order placement and management
- Risk management and pivot protection
- Charting system hosting

**Starting the Server**:
```bash
ssh chriskoin
cd /home/chris/projects/bitstamp/
source env/bin/activate
python src/tdr.py --server
```

### Client (Local)
**Location**: `/Users/chris/projects/python/btc/`

**Responsibilities**:
- User interface and command processing
- Claude integration via command files
- Status monitoring and reporting
- Configuration management
- Log viewing and analysis

**Starting the Client**:
```bash
cd /Users/chris/projects/python/btc
source env/bin/activate
python src/tdr.py  # Client mode is default
tdr> enable_commands  # Enable Claude integration
```

### Communication Protocol
- **Port**: 4000 (tunneled via SSH)
- **Protocol**: HTTP REST API
- **Data Format**: JSON
- **Command Flow**: Client → Server → Response
- **Claude Integration**: JSON files in `commands/pending/` and `commands/processed/`

## Interactive vs Command System

### Interactive Mode
Direct command-line interface for real-time interaction:

```bash
tdr> status long          # Detailed position status
tdr> trades               # Recent trading activity
tdr> logs                 # Server log viewing
tdr> chart                # Launch charting interface
tdr> strategy_diagnostics # Detailed strategy analysis
```

**Features**:
- Tab completion for commands
- Real-time command execution
- Direct server communication
- Immediate feedback

### Command File System (Claude Integration)
Asynchronous command processing via JSON files:

**Command Creation** (`commands/pending/`):
```json
{
  "timestamp": "2025-01-15T20:00:00Z",
  "command": "status long",
  "source": "claude_health_check",
  "args": ""
}
```

**Response Processing** (`commands/processed/`):
```json
{
  "timestamp": "2025-01-15T20:00:00Z",
  "command": "status long",
  "result": {
    "success": true,
    "output": "...",
    "position": {...}
  }
}
```

**Workflow**:
1. Claude writes command to `commands/pending/`
2. Client monitors directory and processes commands
3. Server executes command and returns result
4. Client writes response to `commands/processed/`
5. Claude reads processed file for analysis

## Historical Data Processing

### Data Sources
1. **Historical File**: `btcusd.log`
   - Continuously updated by `websock-ticker2.py`
   - Preserves all historical trades
   - Never lost even if TDR server crashes

2. **Live WebSocket**: Real-time price updates
   - Bitstamp WebSocket feed
   - Integrated with historical data
   - Seamless data continuity

### Data Format
```
timestamp,price,volume
2025-07-15T16:00:00Z,117182.50,0.5
```

### Loading Process
1. **Server Startup**: Loads complete `btcusd.log` file
2. **Memory Management**: Configurable historical window
3. **Live Integration**: WebSocket data appended to memory
4. **Chart Data**: Combined historical + live for complete picture

### Data Manager Architecture
```python
class DataManager:
    def __init__(self):
        self.historical_data = []  # From btcusd.log
        self.live_data = []        # From WebSocket
    
    def get_combined_data(self):
        # Returns merged historical + live data
        return self.historical_data + self.live_data
```

## Backtesting System

### Overview
The backtesting system simulates trading strategies against historical data to evaluate performance before live deployment.

### Backtesting Architecture
```
Historical Data → Strategy Logic → Simulated Trades → Performance Analysis
```

### Key Components

#### 1. Data Preparation
- Load historical price data from `btcusd.log`
- Apply timeframe aggregation (1H, 45T, 15T, etc.)
- Calculate technical indicators

#### 2. Strategy Simulation
- Execute strategy logic on historical data
- Simulate order placement and execution
- Account for trading fees and slippage
- Track position changes and P&L

#### 3. Performance Metrics
- **Total Return**: Percentage gain/loss
- **Sharpe Ratio**: Risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Trade**: Mean profit/loss per trade

### Backtesting Commands
```bash
# Basic backtest
python src/backtest.py --strategy MA --start 2024-01-01 --end 2024-12-31

# Enhanced backtesting with multiple strategies
python src/backtesting/backtester.py --config best_strategy.json

# Strategy optimization
python src/optimization/optimizer.py --strategy adaptive --optimize-params
```

### Configuration Example
```json
{
  "strategy": "AdaptiveMultiStrategy",
  "timeframe": "1H",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "initial_capital": 100000,
  "fee_rate": 0.005,
  "parameters": {
    "ma_fast": 10,
    "ma_slow": 46,
    "rsi_period": 14,
    "confidence_threshold": 0.8
  }
}
```

### Integration with Live Trading
1. **Strategy Validation**: Backtest before deployment
2. **Parameter Optimization**: Use historical data to tune parameters
3. **Risk Assessment**: Evaluate maximum drawdown and volatility
4. **Performance Baseline**: Compare live results to backtested expectations

## Trading Strategies

### Core Strategy Types

#### 1. MA (Moving Average) Strategy
- **Logic**: Buy when fast MA crosses above slow MA, sell when below
- **Parameters**: Fast period, slow period
- **Best For**: Trending markets

#### 2. RSI (Relative Strength Index) Strategy
- **Logic**: Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought)
- **Parameters**: RSI period, overbought/oversold levels
- **Best For**: Ranging markets

#### 3. RAMM (Range-Aware Moving Mean) Strategy
- **Logic**: Adaptive strategy based on price ranges and volatility
- **Parameters**: Range periods, volatility thresholds
- **Best For**: Mixed market conditions

### Strategy Configuration
Strategies are configured in `best_strategy.json`:
```json
{
  "strategy_type": "AdaptiveMultiStrategy",
  "symbol": "btcusd",
  "timeframe": "1H",
  "ma_fast": 10,
  "ma_slow": 46,
  "rsi_period": 14,
  "confidence_threshold": 0.8,
  "signal_confirmation_bars": 2,
  "min_trade_gap_minutes": 15
}
```

## AdaptiveMultiStrategy System

### Overview
The AdaptiveMultiStrategy automatically selects the optimal trading strategy based on current market conditions by classifying market regimes as TRENDING, RANGING, or VOLATILE.

### Market Regime Detection

#### Regime Classification
1. **TRENDING**: Clear directional movement with sustained momentum
   - Uses MA Crossover strategy
   - Optimized for trend following

2. **RANGING**: Sideways price action within defined boundaries
   - Uses RSI or range-based strategies
   - Optimized for mean reversion

3. **VOLATILE**: High volatility with rapid price swings
   - Uses specialized volatile market strategies
   - Optimized for quick entries/exits

#### Confidence Threshold
- **Requirement**: 80% confidence to switch strategies
- **Purpose**: Prevents excessive strategy switching
- **Calculation**: Based on statistical analysis of price patterns

### Strategy Switching Logic
```python
def update_strategy(self):
    regime = self.detect_market_regime()
    confidence = self.calculate_confidence(regime)
    
    if confidence >= 0.8 and regime != self.current_regime:
        self.switch_strategy(regime)
        self.log_strategy_switch(regime, confidence)
```

### Implementation Details

#### Regime Detection Factors
- Price volatility (standard deviation)
- Trend strength (directional movement)
- Range boundaries (support/resistance levels)
- Volume patterns
- Time-based momentum

#### Strategy Mapping
- **TRENDING** → MA Crossover Strategy
- **RANGING** → RSI Strategy  
- **VOLATILE** → Adaptive RAMM Strategy

#### Performance Tracking
- Individual strategy performance by regime
- Regime classification accuracy
- Strategy switch timing analysis
- Confidence threshold optimization

## Dynamic Pivot Protection

### Overview
Dynamic Pivot Protection is an advanced risk management system that provides profit protection and quick re-entry capabilities based on recent price action.

### Key Features

#### 1. Sticky Support/Resistance Levels
- **Problem Solved**: Prevents levels from "chasing" price down
- **Implementation**: Levels lock in place once established
- **Reset Trigger**: Only after successful position flip

#### 2. Market-Based Level Calculation
- **Lookback Period**: Default 2 hours of price data
- **Buffer Zone**: $100 total buffer to prevent whipsaws
- **Level Calculation**:
  - Support = Recent Low - $50
  - Resistance = Recent High + $50

#### 3. Instant Execution
- **No Confirmation Required**: Immediate position flip when levels break
- **Full Position Reversal**: 100% LONG ↔ 100% SHORT
- **Quick Re-entry**: Enables trend continuation capture

### Configuration Parameters
```json
{
  "enable_pivot_protection": true,
  "pivot_buffer": 100,
  "pivot_lookback_hours": 2
}
```

### Status Display Example
```
🎯 Dynamic Pivot Protection (Profit Lock & Quick Re-entry):
   📊 Calculation Details:
     • Looking at last 2 hours of price data
     • Recent High: $117199
     • Recent Low: $116237
     • Buffer Zone: $100 (prevents whipsaws)
     • Level Status: 🔒 LOCKED (Sticky)

   📈 LONG Position Protection:
   • TAKE PROFIT Level: $115728
     → If price drops below, immediately flip to SHORT
   • Current Price: $116498
   • Distance to Profit Lock: $770 (0.7%)
   • RE-ENTRY Level: $117246
     → After SHORT flip, if price rises back above $117246
     → Will flip back to LONG for trend continuation
```

### Trading Examples
- **Entry**: LONG @ $117,182
- **Support Lock**: $115,728 (protects against further losses)
- **Trigger**: Price drops below $115,728 → Immediate flip to SHORT
- **Re-entry**: If price rises back above $117,246 → Flip back to LONG

## Command Reference

### Core Commands

#### Status Commands
```bash
status                    # Basic system status
status long              # Detailed status with pivot protection
trades                   # Recent trading activity
history_status           # Historical data status
strategy_diagnostics     # Detailed strategy analysis
```

#### Trading Commands
```bash
resume_auto_trade [btc] [direction] [entry_price]  # Resume trading
stop_auto_trade                                    # Stop trading
```

#### Data Commands
```bash
logs                     # View server logs
read_server_file <path> [lines]  # Read files on server
chart [symbol] [timeframe] [port]  # Launch charting
```

#### Analysis Commands
```bash
analyze_performance <start> <end>  # Performance analysis
analyze_strategy <name> <timeframe>  # Strategy analysis
analyze_pivots <count>             # Pivot analysis
analyze_regime_switches            # Regime analysis
```

### Claude Integration Commands
Commands are processed via JSON files in `commands/pending/`:

```json
{
  "timestamp": "2025-01-15T20:00:00Z",
  "command": "strategy_diagnostics",
  "source": "claude_analysis",
  "args": ""
}
```

### Server File Access
The `read_server_file` command allows reading files on the remote server:
```bash
read_server_file trades.json              # Read entire file
read_server_file trades.json 50           # Last 50 lines
read_server_file trades.json 100 150      # Lines 100-150
```

## Troubleshooting

### Common Issues

#### 1. Connection Problems
**Symptoms**: Commands fail, client can't reach server
**Solutions**:
- Verify SSH tunnel: `ssh -L 4000:localhost:4000 chriskoin`
- Check server status: `ps aux | grep tdr_server`
- Restart server if needed

#### 2. Data Loading Issues
**Symptoms**: Historical data missing, charts empty
**Solutions**:
- Check `btcusd.log` exists and is recent
- Verify `websock-ticker2.py` is running
- Check server logs for data loading errors

#### 3. Trading Not Executing
**Symptoms**: Signals generated but no trades
**Solutions**:
- Check if auto-trader is running: `status`
- Verify daily trade limits not exceeded
- Check signal confirmation requirements
- Review trade gap constraints

#### 4. Pivot Protection Not Triggering
**Symptoms**: Price breaks levels but no position flip
**Solutions**:
- Verify pivot protection is enabled
- Check if levels are properly locked
- Review buffer zone settings
- Check recent trade history for triggers

### Diagnostic Commands
```bash
# System health check
status long

# Strategy analysis
strategy_diagnostics

# Recent activity
trades

# Server logs
logs

# Historical data status
history_status
```

### Log Files
- **Client Logs**: `logs/tdr_client.log`
- **Server Logs**: Via `logs` command or `read_server_file`
- **Trade History**: `trades.json`
- **Resume State**: `resume-auto-trade.json`

## Best Practices

### Deployment
1. **Always Test First**: Run backtests before live deployment
2. **Gradual Rollout**: Start with small position sizes
3. **Monitor Closely**: Watch first few trades carefully
4. **Have Exit Plan**: Know how to stop trading if needed

### Configuration Management
1. **Backup Configs**: Save working configurations
2. **Version Control**: Use git for all changes
3. **Document Changes**: Note what and why changes were made
4. **Test Parameters**: Backtest before applying new parameters

### Risk Management
1. **Position Sizing**: Don't risk more than you can afford to lose
2. **Daily Limits**: Set maximum trades per day
3. **Drawdown Limits**: Define maximum acceptable losses
4. **Regular Review**: Analyze performance regularly

### Monitoring
1. **Daily Health Checks**: Verify system status daily
2. **Performance Tracking**: Monitor win rates and P&L
3. **Error Monitoring**: Watch for connection or execution errors
4. **Strategy Performance**: Track regime detection accuracy

### Git Workflow
1. **Commit Changes**: Always commit code modifications
2. **Push to Remote**: Ensure changes are backed up
3. **Clear Messages**: Use descriptive commit messages
4. **Branch Management**: Use feature branches for development

### Claude Integration
1. **Enable Commands**: Always run `enable_commands` in client
2. **Monitor Processing**: Check `commands/processed/` for responses
3. **Clear Pending**: Remove old commands from `commands/pending/`
4. **Regular Updates**: Update session knowledge with new learnings

---

## Appendix

### File Structure
```
/Users/chris/projects/python/btc/          # Local client
├── src/tdr.py                            # Main entry point
├── src/tdr_core/strategies.py            # Strategy implementation
├── commands/pending/                     # Claude command input
├── commands/processed/                   # Claude command output
├── best_strategy.json                    # Strategy configuration
└── prompts/resume-claude-session.txt     # Session knowledge

/home/chris/projects/bitstamp/             # Remote server
├── src/tdr_server.py                     # Server implementation
├── btcusd.log                           # Historical data
├── trades.json                          # Trade history
└── resume-auto-trade.json               # Position state
```

### Key Configuration Files
- `best_strategy.json`: Strategy parameters and settings
- `websock-ticker-config.json`: WebSocket configuration
- `resume-auto-trade.json`: Saved trading state

### External Dependencies
- **Bitstamp API**: Live trading and data
- **WebSocket**: Real-time price feeds
- **SSH Tunneling**: Client-server communication
- **Dash/Plotly**: Charting system

### Version History
- **2025-01-13**: Entry price calculation fix
- **2025-01-14**: Dynamic pivot protection implementation
- **2025-01-15**: Trade reason preservation and strategy optimization

This guide provides comprehensive coverage of the TDR trading system. For additional support or questions, refer to the session knowledge in `prompts/resume-claude-session.txt` or create specific command files for Claude analysis.