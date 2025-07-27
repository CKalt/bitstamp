#!/usr/bin/env python3
"""
Patch to add configuration commands to TDR shell
Add these methods to the TDRShell class in src/tdr_core/shell.py
"""

# Add these imports at the top of shell.py if not already present:
# import requests
# import json

# Add these methods to the TDRShell class:

config_commands = '''
    def do_show_strategy(self, arg):
        """
        Show current strategy configuration from server's best_strategy.json
        Usage: show_strategy
        """
        if not self.check_server_connection():
            return
            
        try:
            response = requests.get(f"{self.server_url}/api/best_strategy")
            if response.ok:
                data = response.json()
                if data.get('success'):
                    strategy = data.get('best_strategy', {})
                    self.poutput("\n📊 Current Strategy Configuration:")
                    self.poutput("-" * 50)
                    self.poutput(f"Strategy Type: {strategy.get('Strategy', 'N/A')}")
                    self.poutput(f"Short MA Window: {strategy.get('Short_Window', 'N/A')}")
                    self.poutput(f"Long MA Window: {strategy.get('Long_Window', 'N/A')}")
                    self.poutput(f"Timeframe: {strategy.get('Frequency', 'N/A')}")
                    self.poutput(f"Live Trading: {strategy.get('do_live_trades', False)}")
                    self.poutput(f"\nPerformance Metrics:")
                    self.poutput(f"  Average Trades/Day: {strategy.get('Average_Trades_Per_Day', 0):.2f}")
                    self.poutput(f"  Total Return: {strategy.get('Total_Return', 0):.2f}%")
                    self.poutput(f"  Profit Factor: {strategy.get('Profit_Factor', 0):.2f}")
                    self.poutput(f"\nRisk Management:")
                    self.poutput(f"  Max Trades/Day: {strategy.get('max_trades_per_day', 5)}")
                    self.poutput(f"  Min Time Between Trades: {strategy.get('min_time_between_trades_minutes', 120)} minutes")
                else:
                    self.perror(f"Failed to get strategy: {data.get('error')}")
            else:
                self.perror(f"Server error: {response.status_code}")
        except Exception as e:
            self.perror(f"Error: {str(e)}")
    
    def do_update_ma(self, arg):
        """
        Update MA (Moving Average) parameters on the server
        Usage: update_ma <short_window> <long_window>
        
        Example:
            update_ma 4 20    # Set short MA to 4, long MA to 20
        """
        if not self.check_server_connection():
            return
            
        try:
            parts = arg.split()
            if len(parts) != 2:
                self.perror("Usage: update_ma <short_window> <long_window>")
                return
            
            short_window = int(parts[0])
            long_window = int(parts[1])
            
            if short_window >= long_window:
                self.perror("Short window must be less than long window")
                return
            
            # First get current strategy
            response = requests.get(f"{self.server_url}/api/best_strategy")
            if not response.ok:
                self.perror("Failed to get current strategy")
                return
                
            data = response.json()
            if not data.get('success'):
                self.perror(f"Failed to get strategy: {data.get('error')}")
                return
                
            strategy = data.get('best_strategy', {})
            old_short = strategy.get('Short_Window', 'N/A')
            old_long = strategy.get('Long_Window', 'N/A')
            
            # Update parameters
            strategy['Short_Window'] = short_window
            strategy['Long_Window'] = long_window
            
            # Send update to server
            response = requests.post(
                f"{self.server_url}/api/best_strategy",
                json=strategy,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.ok:
                result = response.json()
                if result.get('success'):
                    self.poutput(f"\n✅ MA parameters updated successfully!")
                    self.poutput(f"  Short MA: {old_short} → {short_window}")
                    self.poutput(f"  Long MA: {old_long} → {long_window}")
                    self.poutput("\nThe server will use these parameters immediately.")
                else:
                    self.perror(f"Failed to update: {result.get('error')}")
            else:
                self.perror(f"Server error: {response.status_code}")
                
        except ValueError:
            self.perror("MA windows must be integers")
        except Exception as e:
            self.perror(f"Error: {str(e)}")
    
    def do_update_strategy_param(self, arg):
        """
        Update any parameter in best_strategy.json
        Usage: update_strategy_param <parameter> <value>
        
        Examples:
            update_strategy_param max_trades_per_day 10
            update_strategy_param min_time_between_trades_minutes 60
            update_strategy_param do_live_trades true
        """
        if not self.check_server_connection():
            return
            
        try:
            parts = arg.split(None, 1)
            if len(parts) != 2:
                self.perror("Usage: update_strategy_param <parameter> <value>")
                return
            
            param_name = parts[0]
            param_value = parts[1]
            
            # Get current strategy
            response = requests.get(f"{self.server_url}/api/best_strategy")
            if not response.ok:
                self.perror("Failed to get current strategy")
                return
                
            data = response.json()
            if not data.get('success'):
                self.perror(f"Failed to get strategy: {data.get('error')}")
                return
                
            strategy = data.get('best_strategy', {})
            old_value = strategy.get(param_name, 'N/A')
            
            # Parse the value
            if param_value.lower() == 'true':
                param_value = True
            elif param_value.lower() == 'false':
                param_value = False
            elif param_value.replace('.', '').replace('-', '').isdigit():
                param_value = float(param_value) if '.' in param_value else int(param_value)
            
            # Update parameter
            strategy[param_name] = param_value
            
            # Send update to server
            response = requests.post(
                f"{self.server_url}/api/best_strategy",
                json=strategy,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.ok:
                result = response.json()
                if result.get('success'):
                    self.poutput(f"\n✅ Parameter updated successfully!")
                    self.poutput(f"  {param_name}: {old_value} → {param_value}")
                    if result.get('preserved_fields'):
                        self.poutput("\nPreserved server-managed fields:")
                        for field, value in result['preserved_fields'].items():
                            self.poutput(f"  {field}: {value}")
                else:
                    self.perror(f"Failed to update: {result.get('error')}")
            else:
                self.perror(f"Server error: {response.status_code}")
                
        except Exception as e:
            self.perror(f"Error: {str(e)}")
    
    def check_server_connection(self):
        """Check if we can connect to the server"""
        try:
            response = requests.get(f"{self.server_url}/api/ping", timeout=2)
            return response.ok
        except:
            self.perror("Cannot connect to server. Make sure:")
            self.perror("  1. SSH tunnel is running: ssh -L 4000:localhost:4000 chriskoin")
            self.perror("  2. TDR server is running on the remote machine")
            return False
'''

print("Configuration commands ready to add to shell.py")
print("\nThese commands provide:")
print("1. show_strategy - View current strategy configuration")
print("2. update_ma <short> <long> - Update MA windows")
print("3. update_strategy_param <param> <value> - Update any parameter")
print("\nTo implement:")
print("1. Add 'import requests' to imports in shell.py if not present")
print("2. Add these methods to the TDRShell class")
print("3. Commit and push the changes")
print("4. Pull on server and restart")