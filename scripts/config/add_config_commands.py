#!/usr/bin/env python3
"""
Add configuration viewing and editing commands to TDR shell
This script shows the code to add to shell.py
"""

# Add these methods to the TDRShell class in src/tdr_core/shell.py

SHELL_COMMANDS = '''
    def do_show_config(self, arg):
        """
        Show current server configuration
        Usage: show_config [section]
        
        Examples:
            show_config              # Show all configuration
            show_config strategy     # Show only strategy configuration
            show_config ma           # Show MA parameters
        """
        try:
            response = self.send_command(f"show_config {arg}")
            if response.get('success'):
                self.print_response(response['output'])
            else:
                self.print_error(f"Failed to get configuration: {response.get('error')}")
        except Exception as e:
            self.print_error(f"Error getting configuration: {str(e)}")
    
    def do_update_ma(self, arg):
        """
        Update MA (Moving Average) parameters
        Usage: update_ma <short_window> <long_window>
        
        Example:
            update_ma 4 20    # Set short MA to 4, long MA to 20
        """
        try:
            parts = arg.split()
            if len(parts) != 2:
                self.print_error("Usage: update_ma <short_window> <long_window>")
                return
            
            short_window = int(parts[0])
            long_window = int(parts[1])
            
            if short_window >= long_window:
                self.print_error("Short window must be less than long window")
                return
            
            response = self.send_command(f"update_ma {short_window} {long_window}")
            if response.get('success'):
                self.print_response(response['output'])
            else:
                self.print_error(f"Failed to update MA parameters: {response.get('error')}")
        except ValueError:
            self.print_error("MA windows must be integers")
        except Exception as e:
            self.print_error(f"Error updating MA parameters: {str(e)}")
    
    def do_view_strategy(self, arg):
        """
        View current strategy configuration from best_strategy.json
        Usage: view_strategy
        """
        try:
            response = self.send_command("view_strategy")
            if response.get('success'):
                self.print_response(response['output'])
            else:
                self.print_error(f"Failed to view strategy: {response.get('error')}")
        except Exception as e:
            self.print_error(f"Error viewing strategy: {str(e)}")
    
    def do_reload_config(self, arg):
        """
        Reload configuration from best_strategy.json
        Usage: reload_config
        
        This will reload the configuration without restarting the server
        """
        try:
            response = self.send_command("reload_config")
            if response.get('success'):
                self.print_response(response['output'])
            else:
                self.print_error(f"Failed to reload configuration: {response.get('error')}")
        except Exception as e:
            self.print_error(f"Error reloading configuration: {str(e)}")
'''

# Add these command handlers to the server's handle_command function

SERVER_HANDLERS = '''
def handle_show_config_command(args):
    """Show current configuration"""
    try:
        if not args or args[0] == "all":
            # Show full configuration
            config = {
                'best_strategy': server_config.get('best_strategy', {}),
                'position': {
                    'side': 'LONG' if server_config.get('position', 0) == 1 else 'SHORT',
                    'amount': server_config.get('btc_amount', 0),
                    'entry_price': server_config.get('entry_price', 0)
                },
                'auto_trader': {
                    'status': 'Running' if auto_trader and auto_trader.is_running() else 'Stopped',
                    'strategy': server_config.get('best_strategy', {}).get('Strategy', 'MA')
                }
            }
            output = json.dumps(config, indent=2)
        elif args[0].lower() in ['strategy', 'ma']:
            # Show strategy configuration
            strategy = server_config.get('best_strategy', {})
            output = f"Current Strategy Configuration:\\n"
            output += f"  Strategy Type: {strategy.get('Strategy', 'N/A')}\\n"
            output += f"  Short MA Window: {strategy.get('Short_Window', 'N/A')}\\n"
            output += f"  Long MA Window: {strategy.get('Long_Window', 'N/A')}\\n"
            output += f"  Frequency: {strategy.get('Frequency', 'N/A')}\\n"
            output += f"  Live Trading: {strategy.get('do_live_trades', False)}\\n"
            output += f"  Average Trades/Day: {strategy.get('Average_Trades_Per_Day', 0):.2f}\\n"
            output += f"  Max Trades/Day: {strategy.get('max_trades_per_day', 5)}"
        else:
            return {'success': False, 'error': f'Unknown section: {args[0]}'}
        
        return {'success': True, 'output': output}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def handle_update_ma_command(args):
    """Update MA parameters"""
    try:
        if len(args) != 2:
            return {'success': False, 'error': 'Usage: update_ma <short_window> <long_window>'}
        
        short_window = int(args[0])
        long_window = int(args[1])
        
        if short_window >= long_window:
            return {'success': False, 'error': 'Short window must be less than long window'}
        
        # Load current strategy
        best_strategy_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_strategy.json')
        with open(best_strategy_file, 'r') as f:
            strategy = json.load(f)
        
        # Update MA parameters
        old_short = strategy.get('Short_Window', 'N/A')
        old_long = strategy.get('Long_Window', 'N/A')
        
        strategy['Short_Window'] = short_window
        strategy['Long_Window'] = long_window
        
        # Save updated strategy
        with open(best_strategy_file, 'w') as f:
            json.dump(strategy, f, indent=2)
        
        # Update server config
        server_config['best_strategy'] = strategy
        
        # Update auto trader if running
        if auto_trader and hasattr(auto_trader, 'strategy'):
            auto_trader.strategy.short_window = short_window
            auto_trader.strategy.long_window = long_window
            logger.info(f"Updated running strategy MA parameters: {short_window}/{long_window}")
        
        output = f"✅ MA parameters updated:\\n"
        output += f"  Short MA: {old_short} → {short_window}\\n"
        output += f"  Long MA: {old_long} → {long_window}\\n"
        output += f"\\nChanges take effect immediately for new signals."
        
        return {'success': True, 'output': output}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def handle_view_strategy_command(args):
    """View current strategy configuration"""
    try:
        best_strategy_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_strategy.json')
        with open(best_strategy_file, 'r') as f:
            strategy = json.load(f)
        
        output = "📊 Current Strategy Configuration (best_strategy.json):\\n"
        output += "-" * 50 + "\\n"
        
        # Key parameters
        output += f"Strategy Type: {strategy.get('Strategy', 'N/A')}\\n"
        output += f"Short MA Window: {strategy.get('Short_Window', 'N/A')}\\n"
        output += f"Long MA Window: {strategy.get('Long_Window', 'N/A')}\\n"
        output += f"Timeframe: {strategy.get('Frequency', 'N/A')}\\n"
        output += f"\\nPerformance Metrics:\\n"
        output += f"  Total Return: {strategy.get('Total_Return', 0):.2f}%\\n"
        output += f"  Average Trades/Day: {strategy.get('Average_Trades_Per_Day', 0):.2f}\\n"
        output += f"  Profit Factor: {strategy.get('Profit_Factor', 0):.2f}\\n"
        output += f"  Sharpe Ratio: {strategy.get('Sharpe_Ratio', 0):.2f}\\n"
        output += f"\\nTrading Settings:\\n"
        output += f"  Live Trading: {strategy.get('do_live_trades', False)}\\n"
        output += f"  Max Trades/Day: {strategy.get('max_trades_per_day', 5)}\\n"
        output += f"  Min Time Between Trades: {strategy.get('min_time_between_trades_minutes', 120)} minutes\\n"
        
        return {'success': True, 'output': output}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def handle_reload_config_command(args):
    """Reload configuration from file"""
    try:
        best_strategy_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_strategy.json')
        with open(best_strategy_file, 'r') as f:
            new_strategy = json.load(f)
        
        # Update server config
        old_short = server_config.get('best_strategy', {}).get('Short_Window', 'N/A')
        old_long = server_config.get('best_strategy', {}).get('Long_Window', 'N/A')
        
        server_config['best_strategy'] = new_strategy
        
        # Update auto trader if running
        if auto_trader and hasattr(auto_trader, 'strategy'):
            auto_trader.strategy.short_window = new_strategy.get('Short_Window', 6)
            auto_trader.strategy.long_window = new_strategy.get('Long_Window', 34)
            logger.info(f"Reloaded strategy configuration")
        
        output = "✅ Configuration reloaded from best_strategy.json\\n"
        if old_short != new_strategy.get('Short_Window') or old_long != new_strategy.get('Long_Window'):
            output += f"\\nMA parameters changed:\\n"
            output += f"  Short MA: {old_short} → {new_strategy.get('Short_Window')}\\n"
            output += f"  Long MA: {old_long} → {new_strategy.get('Long_Window')}"
        
        return {'success': True, 'output': output}
    except Exception as e:
        return {'success': False, 'error': str(e)}
'''

print("Configuration commands implementation ready!")
print("\nTo add these commands:")
print("1. Add the SHELL_COMMANDS methods to TDRShell class in src/tdr_core/shell.py")
print("2. Add the SERVER_HANDLERS to the command handling in src/tdr_server.py")
print("\nThese commands will provide:")
print("- show_config: View current configuration")
print("- update_ma: Update MA parameters on the fly")
print("- view_strategy: View strategy details from best_strategy.json")
print("- reload_config: Reload configuration without restart")