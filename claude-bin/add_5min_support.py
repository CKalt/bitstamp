#!/usr/bin/env python3
"""
Add 5-minute candle support for testing
"""

print("""
MODIFICATIONS NEEDED FOR 5-MINUTE TESTING:

1. In strategies.py, replace the hourly candle check with:

# CANDLE INTERVAL CHECK - Configurable for testing
candle_interval = self.config.get('candle_interval', '1h')  # Default hourly

if candle_interval == '5min':
    # 5-minute candles for rapid testing
    current_candle = signal_time.replace(second=0, microsecond=0)
    current_candle = current_candle.replace(minute=(current_candle.minute // 5) * 5)
    
    if not hasattr(self, '_last_candle_check'):
        self._last_candle_check = current_candle
        self.logger.info(f"🕐 Initial 5-min candle: {current_candle}")
    
    should_evaluate = current_candle > self._last_candle_check
    
    if should_evaluate:
        self.logger.info(f"🕐 NEW 5-MIN CANDLE: {current_candle}")
        self._last_candle_check = current_candle
    else:
        seconds_until_next = 300 - (datetime.now().minute % 5) * 60 - datetime.now().second
        self.logger.info(f"⏳ Next 5-min candle in {seconds_until_next}s")
        time.sleep(5)  # Check more frequently
        continue
        
elif candle_interval == '1h':
    # Original hourly logic
    current_hour = signal_time.replace(minute=0, second=0, microsecond=0)
    # ... rest of hourly code

2. In __init__, add:
    self.config = config if config else {}

This allows testing with 5-minute bars while keeping hourly as default.
""")