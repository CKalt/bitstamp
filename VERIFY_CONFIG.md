# Configuration Verification Steps

## Before Deployment - Capture Remote State

### On Remote Server (before stopping):
```bash
screen -r tdr
(crypto) status
# Screenshot or copy the full output showing:
# - Strategy parameters
# - Position details
# - Entry price

(crypto) stop_auto_trade
(crypto) quit

# Save current config
cd /home/chris/projects/bitstamp
cp best_strategy.json best_strategy.json.remote_backup
cat best_strategy.json > ~/remote_config_backup.txt
```

## After Server Starts - Verify Configuration

### 1. Server Initialization Log Check
When the server starts and receives config from client, it will log:
```
[TDRServer] - INFO - Initializing server with client configuration...
[TDRServer] - INFO - Server initialization response:
  - Live Trading: True
  - Strategy: MA
  - WebSocket: Enabled
```

### 2. Client-Side Verification
In the client after connecting:
```bash
tdr> status
# Should show the same position and parameters

tdr> show_config
# Custom command to display active configuration
```

### 3. Server-Side Config Endpoint
We can add a verification endpoint to check what config the server is using:
```bash
# From another terminal while server is running
curl http://localhost:4000/api/status | jq '.'
```

## Configuration Comparison Checklist

| Parameter | Remote Value | Local Value | Server Active |
|-----------|--------------|-------------|---------------|
| Strategy | MA | MA | ? |
| Short_Window | 10 | 10 | ? |
| Long_Window | 46 | 46 | ? |
| do_live_trades | true | true | ? |
| Position | LONG 1.52275326 | (from resume) | ? |
| Entry Price | 108234 | 108234 | ? |
| regime_switch_threshold | 0.40 | 0.40 | ? |
| signal_confirmation_bars | 2 | 2 | ? |
| min_trade_gap_minutes | 15 | 15 | ? |

## Adding Config Verification Command

We should add this to the server to show active configuration:
```python
@app.route('/api/config', methods=['GET'])
def get_config():
    """Get active server configuration"""
    if not initialization_complete:
        return jsonify({'error': 'Server not initialized'}), 503
    
    return jsonify({
        'server_config': server_config,
        'live_trading': order_placer.do_live_trades if order_placer else False,
        'strategy_config': shell.config if shell else {},
        'data_manager_symbols': data_manager.symbols if data_manager else []
    }), 200
```

## Verification During Deployment

### Phase 1: Before stopping remote
```bash
# Capture current state
(crypto) status > ~/current_position.txt
(crypto) show_config > ~/current_config.txt  # if available
```

### Phase 2: After client connects
```bash
# In client
tdr> status
# Compare with ~/current_position.txt

# Via curl
curl http://localhost:4000/api/config | jq '.' > ~/server_config.txt
diff ~/remote_config_backup.txt ~/server_config.txt
```

### Phase 3: Before resuming auto-trade
```bash
# Verify all parameters match before executing:
tdr> resume_auto_trade 1.52275326btc long 108234
```

## Red Flags to Stop Deployment

STOP if you see any of these:
1. do_live_trades = false when it should be true
2. Different MA window values
3. Position doesn't match (should be LONG 1.52275326 BTC)
4. Entry price doesn't match (should be 108234)
5. Server shows different strategy than "MA"