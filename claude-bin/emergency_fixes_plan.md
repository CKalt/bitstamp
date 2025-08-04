# Emergency Fixes Plan

## QUICK FIXES FOR LIVE (Minimal Risk)

### 1. Add Strategy Loop Heartbeat (5 min fix)
```python
# In strategies.py run_strategy_loop():
while self.running:
    evaluation_count += 1
    current_time = datetime.now()
    
    # ADD THIS - Heartbeat every evaluation
    self.logger.info(f"💓 HEARTBEAT: Strategy loop alive at {current_time.strftime('%H:%M:%S')}")
    
    try:
        # ... existing evaluation code ...
    except Exception as e:
        self.logger.error(f"❌ Strategy loop error: {e}", exc_info=True)
        # DON'T CRASH - Continue looping
        time.sleep(30)
        continue
```

### 2. Add Watchdog Script (10 min fix)
```bash
#!/bin/bash
# claude-bin/monitor_strategy.sh
# Run via: screen -dmS watchdog bash claude-bin/monitor_strategy.sh

while true; do
    # Check if strategy evaluated in last 5 minutes
    LAST_EVAL=$(ssh ck "grep 'HEARTBEAT' ~/projects/bitstamp/logs/tdr_server.log | tail -1 | awk '{print \$1, \$2}'")
    LAST_TIMESTAMP=$(date -d "$LAST_EVAL" +%s 2>/dev/null || echo 0)
    NOW=$(date +%s)
    DIFF=$((NOW - LAST_TIMESTAMP))
    
    if [ $DIFF -gt 300 ]; then
        echo "$(date): WARNING - No heartbeat for $DIFF seconds!"
        # Alert you somehow (email, text, etc)
        curl -X POST http://localhost:4000/api/command \
            -H "Content-Type: application/json" \
            -d '{"command": "force_signal_check"}'
    fi
    
    sleep 60
done
```

### 3. Manual Monitoring Command (5 min fix)
```python
# Add to shell.py:
def do_check_strategy_health(self, arg):
    """Check if strategy loop is running"""
    if hasattr(self, 'auto_trader') and self.auto_trader:
        last_eval = getattr(self.auto_trader, '_last_evaluation', None)
        if last_eval:
            seconds_ago = (datetime.now() - last_eval).total_seconds()
            if seconds_ago > 120:
                print(f"⚠️ WARNING: Last evaluation was {seconds_ago:.0f} seconds ago!")
            else:
                print(f"✅ Strategy healthy - last eval {seconds_ago:.0f}s ago")
        else:
            print("❌ No evaluation timestamp found!")
    else:
        print("❌ No auto trader running!")
```

## BETTER FIXES FOR TEST BRANCH

### 1. Strategy Loop Supervisor
```python
class StrategyLoopSupervisor:
    """Monitors and restarts strategy loop if it dies"""
    
    def __init__(self, strategy):
        self.strategy = strategy
        self.last_heartbeat = datetime.now()
        self.restart_count = 0
        
    def monitor(self):
        while True:
            if (datetime.now() - self.last_heartbeat).total_seconds() > 120:
                self.logger.error("Strategy loop appears dead, restarting...")
                self.restart_strategy_loop()
            time.sleep(30)
    
    def restart_strategy_loop(self):
        try:
            self.strategy.running = False
            time.sleep(5)
            self.strategy.running = True
            self.strategy.start()  # Restart the thread
            self.restart_count += 1
        except Exception as e:
            self.logger.error(f"Failed to restart: {e}")
```

### 2. Separate Signal Checker Thread
Instead of one big loop, have dedicated threads:
- Price collection thread
- Signal evaluation thread  
- Trade execution thread
- Health monitor thread

### 3. Circuit Breaker Pattern
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=3, timeout=300):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure = None
        self.is_open = False
    
    def call(self, func, *args, **kwargs):
        if self.is_open:
            if (datetime.now() - self.last_failure).total_seconds() > self.timeout:
                self.is_open = False  # Try again
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self.failure_count = 0  # Reset on success
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure = datetime.now()
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
            raise
```

### 4. Event-Driven Architecture
Instead of polling every 30 seconds:
```python
class PriceEventHandler:
    def on_new_candle(self, candle):
        # Only evaluate when new hourly candle forms
        if candle.timeframe == '1H':
            self.strategy.evaluate_signal(candle)
```

## DEPLOYMENT PLAN

### For Live (TODAY):
1. Add heartbeat logging (5 min)
2. Add try/except to prevent crashes (5 min)
3. Deploy and monitor closely
4. Use manual checks every hour

### For Test (THIS WEEK):
1. Implement supervisor pattern
2. Add comprehensive error handling
3. Test for 48 hours on test server
4. Monitor reliability metrics

## Manual Commands for Now:
```bash
# Check strategy health
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "check_strategy_health"}'

# Force evaluation
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "force_signal_check"}'

# Watch for heartbeats
ssh ck "tail -f ~/projects/bitstamp/logs/tdr_server.log | grep -E '(HEARTBEAT|SIGNAL_EVAL|ERROR)'"
```