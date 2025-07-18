#!/usr/bin/env python
# src/tdr_client.py
# TDR client that connects to remote server

import sys
import os
import cmd
import json
import requests
import argparse
import threading
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import shutil

# Enable tab completion
try:
    import readline
except ImportError:
    # readline not available on Windows
    pass
else:
    # Enable tab completion
    readline.parse_and_bind("tab: complete")
    # Optional: Add history file support
    import atexit
    histfile = os.path.expanduser("~/.tdr_history")
    try:
        readline.read_history_file(histfile)
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass
    atexit.register(readline.write_history_file, histfile)

# Configuration
DEFAULT_SERVER_URL = "http://localhost:4000"
REQUEST_TIMEOUT = 30  # 30 seconds default, history loads separately

# Setup logging
def setup_client_logging(verbose=False):
    """Configure client logging"""
    logger = logging.getLogger("TDRClient")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    
    # Enhanced formatter
    formatter = logging.Formatter('%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler('logs/tdr_client.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_client_logging()

class LocalCommandInterface:
    """Monitors local command files and forwards them to remote server"""
    
    def __init__(self, client, base_dir="commands"):
        self.client = client
        self.base_dir = Path(base_dir)
        self.pending_dir = self.base_dir / "pending"
        self.processed_dir = self.base_dir / "processed"
        self.failed_dir = self.base_dir / "failed"
        self.logger = logging.getLogger("TDRClient.CommandInterface")
        
        # Create directory structure
        for dir_path in [self.pending_dir, self.processed_dir, self.failed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.running = False
        self.monitor_thread = None
        
    def start(self):
        """Start monitoring for local command files"""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"✅ Local command interface enabled")
        print(f"   Monitoring: {self.pending_dir}")
        print(f"   Processed: {self.processed_dir}")
        print(f"   Failed: {self.failed_dir}")
        
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        print("Command interface stopped")
        
    def _monitor_loop(self):
        """Monitor for new command files and forward to server"""
        while self.running:
            try:
                # Check for new command files
                for cmd_file in sorted(self.pending_dir.glob("*.json")):
                    self._process_command_file(cmd_file)
            except Exception as e:
                print(f"Command monitor error: {e}")
            
            time.sleep(0.5)  # Check twice per second
            
    def _process_command_file(self, cmd_file):
        """Process a single command file by sending to server"""
        try:
            # Read command
            with open(cmd_file, 'r') as f:
                cmd_data = json.load(f)
            
            # Extract command and args
            command = cmd_data.get('command', '')
            args = cmd_data.get('args', '')
            source = cmd_data.get('source', 'claude')
            full_command = f"{command} {args}".strip()
            
            self.logger.info(f"[CLAUDE-CMD] Processing from {cmd_file.name} | Source: {source} | Command: {full_command}")
            print(f"Processing command from {cmd_file.name}: {full_command}")
            
            # Send to server with source info
            response = self.client.send_command(full_command, source=source)
            
            # Add result to command data
            cmd_data['result'] = response
            cmd_data['processed_at'] = datetime.now().isoformat()
            
            if response.get('success', True):
                # Save to processed directory
                processed_file = self.processed_dir / cmd_file.name
                with open(processed_file, 'w') as f:
                    json.dump(cmd_data, f, indent=2)
                print(f"Command processed successfully: {cmd_file.name}")
            else:
                # Save to failed directory
                self._move_to_failed(cmd_file, response.get('error', 'Unknown error'))
                return
                
            # Remove from pending
            cmd_file.unlink()
            
        except Exception as e:
            print(f"Failed to process command {cmd_file.name}: {e}")
            self._move_to_failed(cmd_file, str(e))
            
    def _move_to_failed(self, cmd_file, reason):
        """Move command to failed directory with error reason"""
        try:
            failed_file = self.failed_dir / cmd_file.name
            shutil.move(str(cmd_file), str(failed_file))
            
            # Add failure info
            error_file = failed_file.with_suffix('.error')
            with open(error_file, 'w') as f:
                json.dump({
                    'failed_at': datetime.now().isoformat(),
                    'reason': reason
                }, f, indent=2)
        except Exception as e:
            print(f"Error moving command to failed directory: {e}")

class RemoteTDRClient(cmd.Cmd):
    """Enhanced TDR client that initializes server with local config"""
    
    intro = """
================================================================================
                        TDR Trading Client (Remote)
================================================================================
Initializing connection to remote server...
================================================================================
"""
    prompt = 'tdr> '
    
    def __init__(self, server_url: str, config_file: str = 'best_strategy.json', verbose: bool = False):
        super().__init__()
        self.server_url = server_url.rstrip('/')
        self.verbose = verbose
        self.config_file = config_file
        self.last_status = None
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        # Enable command completion
        self.use_rawinput = True
        self.completekey = 'tab'
        self.command_interface = None
        self.initialized = False
        self.logger = logging.getLogger("TDRClient")
        
        # Load local configuration
        self.config = self.load_configuration()
        
        # Check if server is already initialized
        if self.check_server_initialized():
            print(f"✅ Server at {self.server_url} is already initialized")
            self.initialized = True
            # Wait for server to be fully ready
            self.wait_for_server_ready()
            self.update_status()
            # Automatically enable command interface
            self.do_enable_commands("")
        else:
            # Initialize server with configuration
            print(f"Server not initialized, sending configuration...")
            if self.initialize_server():
                print(f"✅ Successfully initialized TDR server at {self.server_url}")
                self.initialized = True
                # Wait for server to be fully ready
                self.wait_for_server_ready()
                self.update_status()
                # Automatically enable command interface
                self.do_enable_commands("")
            else:
                print(f"❌ Failed to initialize server at {self.server_url}")
                print("Some commands may not work properly.")
    
    def emptyline(self):
        """Do nothing on empty line"""
        pass
    
    def completenames(self, text, *ignored):
        """Override to provide command name completion"""
        dotext = 'do_' + text
        return [a[3:] for a in self.get_names() if a.startswith(dotext)]
    
    def load_configuration(self, sync_from_server=True) -> Dict[str, Any]:
        """Load configuration from local files
        
        Args:
            sync_from_server: If True, offers to sync from server for non-backtesting operations
        """
        config = {
            'best_strategy': {},
            'verbose': self.verbose,
            'enable_websocket': True
        }
        
        # Check if we should sync from server (skip for backtesting)
        is_backtest = hasattr(self, 'args') and hasattr(self.args, 'backtest') and self.args.backtest
        if sync_from_server and not is_backtest and self.test_connection():
            try:
                # Check if server is initialized first
                if self.check_server_initialized():
                    response = requests.get(f"{self.server_url}/api/best_strategy", timeout=5)
                    if response.status_code == 200:
                        server_data = response.json()
                        if server_data.get('success') and server_data.get('best_strategy'):
                            server_strategy = server_data['best_strategy']
                            
                            # Compare with local if it exists
                            local_strategy = None
                            if os.path.exists(self.config_file):
                                with open(self.config_file, 'r') as f:
                                    local_strategy = json.load(f)
                            
                            # Check if they differ in key fields
                            if local_strategy:
                                # Categorize fields by priority
                                server_priority_fields = ['Last_Trade_Price', 'Last_Trade_Timestamp', 'Last_Signal_Action', 
                                                         'Last_Signal_Timestamp']  # Server owns these
                                client_priority_fields = ['auto_resume', 'max_trades_per_day', 'Short_Window', 'Long_Window', 
                                                        'Strategy', 'regime_switch_threshold', 'signal_confirmation_bars', 
                                                        'min_trade_gap_minutes', 'emergency_loss_threshold', 'do_live_trades',
                                                        'auto_align_position', 'emergency_override_enabled']  # Client owns these
                                
                                # Find differences
                                from_server = []  # Fields to update from server
                                to_server = []    # Fields to push to server
                                
                                # Check server-priority fields (always take server's value)
                                for field in server_priority_fields:
                                    if local_strategy.get(field) != server_strategy.get(field):
                                        from_server.append((field, local_strategy.get(field), server_strategy.get(field)))
                                
                                # Check client-priority fields (push non-null client values to server)
                                for field in client_priority_fields:
                                    local_val = local_strategy.get(field)
                                    server_val = server_strategy.get(field)
                                    if local_val != server_val:
                                        # If server has null and client has value, push to server
                                        if server_val is None and local_val is not None:
                                            to_server.append((field, server_val, local_val))
                                        # If both have values but differ, ask user
                                        elif local_val is not None and server_val is not None:
                                            to_server.append((field, server_val, local_val))
                                
                                # Execute two-way sync
                                if from_server or to_server:
                                    print("\n🔄 Syncing best_strategy.json...")
                                    
                                    # First, apply server priority fields (no confirmation needed)
                                    if from_server:
                                        print("\n📥 Updating from server:")
                                        for field, local_val, server_val in from_server:
                                            local_strategy[field] = server_val
                                            print(f"  ✅ {field}: {local_val} → {server_val}")
                                    
                                    # Then handle client priority fields that need to go to server
                                    if to_server:
                                        print("\n📤 Client has updates for server:")
                                        for field, server_val, local_val in to_server:
                                            if server_val is None:
                                                print(f"  • {field}: server=null → client={local_val}")
                                            else:
                                                print(f"  • {field}: server={server_val} → client={local_val}")
                                        
                                        response = input("\nPush these client values to server? (yes/no) [yes]: ").strip().lower()
                                        
                                        if response in ['', 'yes', 'y']:
                                            # Push to server
                                            print("📤 Pushing to server...")
                                            # Apply client values to the strategy we'll send
                                            push_strategy = server_strategy.copy()
                                            for field, _, local_val in to_server:
                                                push_strategy[field] = local_val
                                            
                                            try:
                                                push_response = requests.post(
                                                    f"{self.server_url}/api/best_strategy",
                                                    json=push_strategy,
                                                    timeout=10
                                                )
                                                if push_response.status_code == 200:
                                                    print("✅ Successfully synced to server")
                                                    # Update our local copy with the merged result
                                                    config['best_strategy'] = local_strategy
                                                else:
                                                    print(f"❌ Failed to push: {push_response.text}")
                                            except Exception as e:
                                                print(f"❌ Error pushing: {e}")
                                        else:
                                            print("⏭️  Skipping server update")
                                    
                                    # Save the updated local strategy
                                    with open(self.config_file, 'w') as f:
                                        json.dump(local_strategy, f, indent=2)
                                    config['best_strategy'] = local_strategy
                                    
                                else:
                                    # No differences
                                    config['best_strategy'] = local_strategy
                            else:
                                # No local file, use server's
                                with open(self.config_file, 'w') as f:
                                    json.dump(server_strategy, f, indent=2)
                                print("✅ Downloaded best_strategy.json from server")
                                config['best_strategy'] = server_strategy
                                
            except Exception as e:
                logger.debug(f"Could not sync strategy from server: {e}")
        
        # If we didn't load from server sync, load from local file
        if not config['best_strategy'] and os.path.exists(self.config_file):
            print(f"Loading configuration from {self.config_file}")
            with open(self.config_file, 'r') as f:
                config['best_strategy'] = json.load(f)
        elif not config['best_strategy']:
            print(f"Warning: {self.config_file} not found, using defaults")
            config['best_strategy'] = {
                'Strategy': 'MA',
                'Short_Window': 10,
                'Long_Window': 46,
                'do_live_trades': False,
                'start_window_days_back': 30,
                'end_window_days_back': 0
            }
        
        # Load position from resume file if exists
        resume_file = 'resume-auto-trade.json'
        if os.path.exists(resume_file):
            print(f"Loading saved position from {resume_file}")
            with open(resume_file, 'r') as f:
                resume_data = json.load(f)
                # Handle both old and new resume file formats
                position_str = resume_data.get('position', 'long').lower()
                position_value = 1 if position_str == 'long' else -1
                
                # Get amounts based on position type
                btc_amount = resume_data.get('amount', resume_data.get('btc_amount', 0))
                usd_amount = resume_data.get('usd_amount', 0)
                
                # For new format, 'amount' field depends on position type
                if 'unit' in resume_data:
                    if resume_data['unit'] == 'btc':
                        btc_amount = resume_data['amount']
                        usd_amount = 0
                    else:
                        btc_amount = 0
                        usd_amount = resume_data['amount']
                
                config['initial_position'] = {
                    'btc_balance': btc_amount,
                    'usd_balance': usd_amount,
                    'position': position_value,
                    'position_size': btc_amount if position_value == 1 else -btc_amount,
                    'position_cost_basis': resume_data.get('entry_price', 0) * abs(btc_amount if position_value == 1 else usd_amount / resume_data.get('entry_price', 1))
                }
        
        return config
    
    def initialize_server(self) -> bool:
        """Initialize remote server with local configuration"""
        try:
            print("Sending configuration to server...")
            response = requests.post(
                f"{self.server_url}/api/initialize",
                json=self.config,
                timeout=30  # Longer timeout for initialization
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Server initialization successful!")
                summary = result.get('config_summary', {})
                print(f"  - Live Trading: {summary.get('do_live_trades', False)}")
                print(f"  - Strategy: {summary.get('strategy', 'Unknown')}")
                print(f"  - WebSocket: {'Enabled' if summary.get('websocket', False) else 'Disabled'}")
                
                if summary.get('history_loading', False):
                    print(f"  - Historical Data: Loading in background...")
                    print("\n⚠️  IMPORTANT: Historical data is loading in the background.")
                    print("  - Use 'history_status' to check loading progress")
                    print("  - Trading commands will be blocked until loading completes")
                else:
                    print(f"  - Historical Data: {'Loaded' if summary.get('historical_data_loaded', False) else 'Not loaded'}")
                
                return True
            else:
                print(f"Server initialization failed: {response.status_code}")
                if response.text:
                    print(f"Error: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print("Initialization request timed out")
            return False
        except requests.exceptions.ConnectionError:
            print(f"Cannot connect to server at {self.server_url}")
            return False
        except Exception as e:
            print(f"Initialization error: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test connection to server"""
        try:
            response = requests.get(f"{self.server_url}/api/ping", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def check_server_initialized(self) -> bool:
        """Check if server is already initialized"""
        try:
            response = requests.get(f"{self.server_url}/api/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Server is initialized if it returns a valid status
                return data.get('initialized', False)
            elif response.status_code == 503:
                # 503 means server not initialized
                return False
            return False
        except:
            return False
    
    def wait_for_server_ready(self):
        """Wait for server to be fully ready (strategy initialized, history loaded if needed)"""
        print("\n⏳ Waiting for server to be ready...")
        
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        spinner_idx = 0
        last_status = ""
        
        while True:
            try:
                response = requests.get(f"{self.server_url}/api/status", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check if auto trader is initialized
                    auto_trader = data.get('auto_trader', {})
                    if not auto_trader.get('active', False):
                        # Check if we should auto_resume
                        if data.get('auto_resume', False) and not data.get('history_loading', False):
                            print("\r📊 Executing auto_resume..." + " " * 30)
                            response = self.send_command("auto_resume")
                            if response and response.get('success'):
                                print("\r✅ Auto-resume executed successfully" + " " * 30)
                                time.sleep(1)  # Give server a moment to initialize
                                continue
                        status = "Waiting for trading strategy to initialize"
                    # Check if history is still loading
                    elif data.get('history_loading', False):
                        history_status = data.get('history_status', 'Loading historical data')
                        status = f"Loading historical data: {history_status}"
                    # Check if we're waiting for history to load
                    elif not data.get('history_loaded', False) and data.get('auto_resume', False):
                        status = "Waiting for historical data to load"
                    else:
                        # Server is ready
                        print("\r✅ Server is ready for commands!" + " " * 50)
                        
                        # Note about commands that don't need history
                        if data.get('history_loading', False) or not data.get('history_loaded', False):
                            print("\n📌 Note: Some commands work without historical data:")
                            print("   • whipsaw_stats, show_trade_sequence (use trades.json)")
                            print("   • status, trades, positions (use current state)")
                            print("   • Trading commands will wait for history to complete")
                        return
                    
                    # Update status display
                    if status != last_status:
                        print(f"\r{spinner[spinner_idx]} {status}" + " " * 20, end='', flush=True)
                        last_status = status
                    else:
                        print(f"\r{spinner[spinner_idx]} {status}", end='', flush=True)
                    
                    spinner_idx = (spinner_idx + 1) % len(spinner)
                else:
                    print(f"\r⚠️  Server returned status {response.status_code}", end='', flush=True)
                    
            except Exception as e:
                print(f"\r⚠️  Error checking server status: {e}", end='', flush=True)
            
            time.sleep(0.5)
    
    def update_status(self):
        """Update cached status from server"""
        try:
            response = requests.get(f"{self.server_url}/api/status", timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                self.last_status = response.json()
                return True
        except Exception as e:
            if self.verbose:
                print(f"Error updating status: {e}")
        return False
    
    def send_command(self, command: str, source: str = "interactive") -> Dict[str, Any]:
        """Send command to server and return response"""
        try:
            self.logger.info(f"[CMD-SEND] Source: {source} | Command: {command}")
            response = requests.post(
                f"{self.server_url}/api/command",
                json={'command': command, 'source': source},
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'success': False,
                    'error': f"Server returned {response.status_code}: {response.text}"
                }
        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'Request timed out'}
        except requests.exceptions.ConnectionError:
            return {'success': False, 'error': 'Cannot connect to server'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def print_response(self, response: Dict[str, Any]):
        """Print formatted response from server"""
        if response.get('success', True):
            # Check if this is a status command and add server state
            if response.get('command') == 'status' or response.get('command') == 'status long':
                # Get current server status
                try:
                    status_response = requests.get(f"{self.server_url}/api/status", timeout=5)
                    if status_response.status_code == 200:
                        server_status = status_response.json()
                        
                        # Show server state header
                        print("\n📊 Server Status:")
                        print("━" * 50)
                        
                        # History loading state
                        if server_status.get('history_loading'):
                            status = server_status.get('history_status', 'Loading...')
                            print(f"⏳ History: {status}")
                        elif server_status.get('history_loaded'):
                            record_count = server_status.get('position', {}).get('history_record_count', 0)
                            print(f"✅ History: Loaded")
                        else:
                            print(f"❌ History: Not loaded")
                            
                        # Auto-trader state
                        if server_status.get('auto_trader', {}).get('active'):
                            at = server_status['auto_trader']
                            print(f"🤖 Auto-Trader: Active ({at.get('strategy', 'Unknown')})")
                            print(f"   Trades Today: {at.get('trades_today', 0)}")
                        else:
                            print(f"🔴 Auto-Trader: Not running")
                            
                        # Connection state
                        print(f"🌐 WebSocket: {server_status.get('websocket', 'unknown')}")
                        print(f"💰 Live Trading: {'Enabled' if server_status.get('live_trading') else 'Disabled'}")
                        print("━" * 50)
                except:
                    pass  # Don't fail if we can't get server status
            
            # Print special message for auto-resume response
            if response.get('auto_resume') and response.get('message'):
                print(f"\n✅ {response['message']}")
                if response.get('history_progress') is not None:
                    print(f"⏳ History loading progress: {response['history_progress']:.1f}%")
                return
            
            # Print command output
            output = response.get('output', '')
            if output:
                print(output.rstrip())
            
            # Print position update if present
            if 'position' in response:
                pos = response['position']
                print(f"\nPosition Update:")
                
                # Determine and show position direction
                if 'position' in pos:
                    position_val = pos['position']
                    if position_val == 1:
                        print(f"  Direction: LONG")
                    elif position_val == -1:
                        print(f"  Direction: SHORT")
                    else:
                        print(f"  Direction: NEUTRAL (Error - should not happen)")
                
                print(f"  BTC: {pos['btc_balance']:.8f}")
                print(f"  USD: ${pos['usd_balance']:.2f}")
                if pos.get('entry_price') and pos['entry_price'] > 0:
                    print(f"  Entry Price: ${pos['entry_price']:.2f}")
        else:
            print(f"Error: {response.get('error', 'Unknown error')}")
    
    # Override default to send all commands to server
    def default(self, line):
        """Handle all commands by sending to server"""
        if not line.strip():
            return
        
        if not self.initialized:
            print("Server not initialized. Some commands may fail.")
        
        # Send command to server
        response = self.send_command(line)
        self.print_response(response)
        
        # Update status after certain commands
        if any(cmd in line for cmd in ['buy', 'sell', 'auto_trade', 'stop_auto_trade']):
            self.update_status()
    
    # Special client-side commands
    
    def do_logs(self, args):
        """View server logs
        Usage: logs [lines] [type] [search]
        Types: server, trading, diagnostic
        Example: logs 50 server ERROR"""
        parts = args.split()
        lines = int(parts[0]) if parts else 100
        log_type = parts[1] if len(parts) > 1 else 'server'
        search = ' '.join(parts[2:]) if len(parts) > 2 else ''
        
        try:
            params = {
                'lines': lines,
                'type': log_type,
                'search': search
            }
            response = requests.get(
                f"{self.server_url}/api/logs",
                params=params,
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                data = response.json()
                print(f"\n=== {data['type'].upper()} LOGS ({data['file']}) ===")
                if data.get('lines'):
                    for line in data['lines']:
                        if line.strip():
                            print(line)
                print(f"\nShowing {data['total_lines']} lines")
            else:
                print(f"Error getting logs: {response.text}")
        except Exception as e:
            print(f"Error: {e}")
    
    def do_show_diagnostics(self, args):
        """Show diagnostic events from server
        Usage: show_diagnostics [event_type] [count]
        Event types: ALL, SIGNAL_EVAL, TRADE_EXECUTED, REGIME_CHANGE, etc."""
        parts = args.split()
        event_type = parts[0] if parts else 'ALL'
        count = int(parts[1]) if len(parts) > 1 else 20
        
        try:
            params = {
                'event_type': event_type,
                'count': count
            }
            response = requests.get(
                f"{self.server_url}/api/diagnostics",
                params=params,
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                data = response.json()
                events = data.get('events', [])
                print(f"\n=== DIAGNOSTIC EVENTS ({event_type}) ===")
                for event in events:
                    timestamp = event.get('timestamp', 'N/A')
                    event_type = event.get('event_type', 'UNKNOWN')
                    print(f"\n[{timestamp}] {event_type}")
                    
                    # Print event data based on type
                    event_data = event.get('data', {})
                    if isinstance(event_data, dict):
                        for key, value in event_data.items():
                            if key not in ['timestamp', 'event_type']:
                                print(f"  {key}: {value}")
                print(f"\nShowing {len(events)} of {data['total_events']} events")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error: {e}")
    
    def do_resume_auto_trade(self, args):
        """Resume auto-trading from saved state or with manual parameters
        
        Usage: 
          resume_auto_trade                              # Use saved resume-auto-trade.json
          resume_auto_trade 1.55612586btc long 107374   # Manual LONG position
          resume_auto_trade 167500usd short 107263      # Manual SHORT position
        """
        # Forward to server
        response = self.send_command(f"resume_auto_trade {args}")
        self.print_response(response)
        self.update_status()
    
    def do_stop_auto_trade(self, args):
        """Stop auto-trading if running"""
        # Forward to server
        response = self.send_command("stop_auto_trade")
        self.print_response(response)
        self.update_status()
    
    def do_fix_entry_price(self, args):
        """Fix entry price by recalculating from trades.json
        
        This command:
        - Calculates the correct average entry price from actual trades
        - Updates resume-auto-trade.json with the correct entry price
        - Updates best_strategy.json with the correct Last_Trade_Price
        - Fixes the position tracking in the running auto-trader
        """
        try:
            print("Fixing entry price from trades.json...")
            response = requests.post(f"{self.server_url}/api/fix_entry_price", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"\n✅ Entry price fixed successfully!")
                print(f"  Position: {data['current_position']}")
                print(f"  Total BTC: {data['total_btc']:.8f}")
                print(f"  Correct Entry Price: ${data['average_entry_price']:.2f}")
                print(f"  Based on: {data['trades_count']} trades")
                
                # Sync best_strategy.json from server
                print("\nSyncing best_strategy.json from server...")
                self.config = self.load_configuration(sync_from_server=True)
                
                # Update status to show new values
                self.update_status()
                self.do_status("")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Error fixing entry price: {e}")
    
    def do_trades(self, args):
        """Show recent trades from server
        Usage: trades [limit]"""
        try:
            limit = int(args.strip()) if args.strip() else 10
        except ValueError:
            limit = 10
        
        try:
            response = requests.get(
                f"{self.server_url}/api/trades",
                params={'limit': limit},
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                data = response.json()
                trades = data.get('trades', [])
                if trades:
                    print(f"\n=== RECENT TRADES (showing {len(trades)}) ===")
                    for trade in trades:
                        # Handle timestamp - could be string or number
                        timestamp_val = trade.get('timestamp', 0)
                        if isinstance(timestamp_val, str):
                            try:
                                # Try parsing as ISO format first
                                timestamp = datetime.fromisoformat(timestamp_val.replace('Z', '+00:00'))
                            except ValueError:
                                try:
                                    # Try parsing as float string
                                    timestamp = datetime.fromtimestamp(float(timestamp_val))
                                except (ValueError, TypeError):
                                    timestamp = datetime.now()
                        else:
                            try:
                                timestamp = datetime.fromtimestamp(timestamp_val)
                            except (ValueError, TypeError):
                                timestamp = datetime.now()
                        # Check for trade type - could be 'action' or 'type' field
                        action = trade.get('action')
                        if action is None:
                            # Look for 'type' field and convert to BUY/SELL
                            trade_type = trade.get('type', 'UNKNOWN')
                            if trade_type == 'buy':
                                action = 'BUY'
                            elif trade_type == 'sell':
                                action = 'SELL'
                            else:
                                action = 'UNKNOWN'
                        
                        amount = trade.get('amount', 0)
                        price = trade.get('price', 0)
                        total = trade.get('total_usd', amount * price)
                        print(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {action} {amount:.8f} BTC @ ${price:.2f} = ${total:.2f}")
                else:
                    print("No trades found")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error: {e}")
    
    def do_reconnect(self, arg):
        """Reconnect to the server and sync configuration"""
        print(f"Reconnecting to {self.server_url}...")
        
        # First test if server is reachable
        if not self.test_connection():
            print("❌ Cannot connect to server")
            return
        
        # Check if server is already initialized
        if self.check_server_initialized():
            print("✅ Server is already initialized, syncing configuration...")
            self.config = self.load_configuration()
            self.initialized = True
            self.update_status()
            print("✅ Reconnection successful!")
        else:
            # Server needs initialization
            print("Server not initialized, sending configuration...")
            self.config = self.load_configuration()
            if self.initialize_server():
                print("✅ Reconnection and initialization successful!")
                self.initialized = True
                self.update_status()
            else:
                print("❌ Reconnection failed")
    
    def do_load_history(self, arg):
        """Start loading historical data on the server"""
        try:
            response = requests.post(f"{self.server_url}/api/load_history", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"Status: {data['status']}")
                print(f"Message: {data['message']}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error loading history: {e}")
    
    def do_history_status(self, arg):
        """Check historical data loading status"""
        try:
            response = requests.get(f"{self.server_url}/api/history_status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print("\n=== History Loading Status ===")
                if data['history_loading']:
                    status = data.get('history_status', 'Loading...')
                    phase = data.get('current_phase', 'unknown')
                    
                    print(f"⏳ {status}")
                    if phase and phase != 'unknown':
                        phase_display = phase.replace('_', ' ').title()
                        print(f"   Phase: {phase_display}")
                elif data['history_loaded']:
                    print(f"✅ Complete: {data.get('record_count', 0):,} records loaded")
                else:
                    print("❌ Not loaded")
                    
                if data.get('history_error'):
                    print(f"❌ Error: {data['history_error']}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error checking status: {e}")
    
    def do_server(self, arg):
        """Show server information"""
        print(f"Server URL: {self.server_url}")
        print(f"Connection: {'OK' if self.test_connection() else 'FAILED'}")
        print(f"Initialized: {self.initialized}")
        if self.last_status:
            print(f"Live Trading: {self.last_status.get('live_trading', False)}")
            print(f"WebSocket: {self.last_status.get('websocket', 'Unknown')}")
    
    def do_poll_server(self, arg):
        """Poll server for important events and display notifications
        Usage: poll_server [interval_seconds]
        
        Monitors:
        - History loading progress
        - Trading events from auto-trader
        - Position changes
        - Connection status changes
        
        Press 'q' + Enter or Ctrl+C to stop polling
        """
        interval = float(arg) if arg else 10.0
        
        if hasattr(self, 'polling_thread') and self.polling_thread and self.polling_thread.is_alive():
            print("Already polling server. Use 'q' + Enter to stop.")
            return
        
        print(f"\n🔄 Starting server event polling (checking every {interval}s)")
        print("📊 Monitoring: History loading, trades, position changes")
        print("⏹️  To stop: Type 'q' and press Enter (or press Ctrl+C)\n")
        print("─" * 60)
        
        self.stop_polling = threading.Event()
        self.last_trade_count = None
        self.last_history_status = None
        self.last_position = None
        self.history_complete_notified = False
        
        def poll_loop():
            while not self.stop_polling.is_set():
                try:
                    # Get current status
                    response = requests.get(f"{self.server_url}/api/status", timeout=5)
                    if response.status_code == 200:
                        status = response.json()
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        
                        # Check history loading status
                        if status.get('history_loading'):
                            history_status = status.get('history_status', 'Loading...')
                            if history_status != self.last_history_status:
                                print(f"[{timestamp}] 📚 History: {history_status}")
                                self.last_history_status = history_status
                        elif status.get('history_loaded') and not self.history_complete_notified:
                            record_count = status.get('position', {}).get('history_record_count', 0)
                            print(f"[{timestamp}] ✅ History loading complete! ({record_count:,} records)")
                            self.history_complete_notified = True
                        
                        # Check for auto-trader events
                        if status.get('auto_trader', {}).get('active'):
                            at = status['auto_trader']
                            trades_today = at.get('trades_today', 0)
                            
                            # Check for new trades
                            if self.last_trade_count is not None and trades_today > self.last_trade_count:
                                # Get recent trades to show the new one
                                trades_response = requests.get(f"{self.server_url}/api/trades", params={'limit': 1}, timeout=5)
                                if trades_response.status_code == 200:
                                    trades_data = trades_response.json()
                                    if trades_data.get('trades'):
                                        trade = trades_data['trades'][0]
                                        action = trade.get('action', 'UNKNOWN')
                                        amount = trade.get('amount', 0)
                                        price = trade.get('price', 0)
                                        print(f"[{timestamp}] 💰 NEW TRADE: {action} {amount:.8f} BTC @ ${price:.2f}")
                                        
                                        # Play alert sound if available
                                        try:
                                            print('\a')  # Terminal bell
                                        except:
                                            pass
                            
                            self.last_trade_count = trades_today
                        
                        # Check for position changes
                        if 'position' in status:
                            pos = status['position']
                            current_position = {
                                'btc': pos['btc_balance'],
                                'usd': pos['usd_balance'],
                                'side': pos['position']
                            }
                            
                            if self.last_position and current_position != self.last_position:
                                # Position changed
                                btc_change = current_position['btc'] - self.last_position['btc']
                                usd_change = current_position['usd'] - self.last_position['usd']
                                
                                if abs(btc_change) > 0.00000001 or abs(usd_change) > 0.01:
                                    side = 'LONG' if current_position['side'] > 0 else 'SHORT'
                                    print(f"[{timestamp}] 📍 Position changed to {side}")
                                    print(f"              BTC: {current_position['btc']:.8f} ({btc_change:+.8f})")
                                    print(f"              USD: ${current_position['usd']:.2f} ({usd_change:+.2f})")
                            
                            self.last_position = current_position
                        
                        # Check WebSocket connection
                        ws_status = status.get('websocket', 'unknown')
                        if hasattr(self, 'last_ws_status') and ws_status != self.last_ws_status:
                            if ws_status == 'connected':
                                print(f"[{timestamp}] 🟢 WebSocket connected")
                            else:
                                print(f"[{timestamp}] 🔴 WebSocket disconnected")
                        self.last_ws_status = ws_status
                        
                except KeyboardInterrupt:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ⏹️  Stopped polling (Ctrl+C)")
                    break
                except Exception as e:
                    if not self.stop_polling.is_set():
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Poll error: {e}")
                
                # Wait for interval or stop signal
                self.stop_polling.wait(interval)
            
            print("─" * 60)
            print("Server polling stopped.")
        
        # Start polling thread
        self.polling_thread = threading.Thread(target=poll_loop, daemon=True)
        self.polling_thread.start()
        
        # Start input monitor thread
        def input_monitor():
            while self.polling_thread.is_alive():
                try:
                    user_input = input()
                    if user_input.lower() == 'q':
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏹️  Stopping polling...")
                        self.stop_polling.set()
                        break
                except:
                    break
        
        input_thread = threading.Thread(target=input_monitor, daemon=True)
        input_thread.start()
    
    def do_monitor(self, arg):
        """Start real-time monitoring of prices and positions
        Usage: monitor [interval_seconds]"""
        interval = float(arg) if arg else 5.0
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            print("Monitoring already active. Use 'stop_monitor' first.")
            return
        
        print(f"Starting real-time monitoring (update every {interval}s)")
        print("Press Ctrl+C or use 'stop_monitor' to stop")
        
        self.stop_monitoring.clear()
        
        def monitor_loop():
            while not self.stop_monitoring.is_set():
                try:
                    response = requests.get(f"{self.server_url}/api/status", timeout=5)
                    if response.status_code == 200:
                        status = response.json()
                        
                        # Clear screen and show status
                        os.system('clear' if os.name == 'posix' else 'cls')
                        print("=== TDR Real-Time Monitor ===")
                        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"Server: {status['server']}")
                        print(f"WebSocket: {status['websocket']}")
                        
                        if 'position' in status:
                            pos = status['position']
                            print(f"\nPosition:")
                            print(f"  BTC: {pos['btc_balance']:.8f}")
                            print(f"  USD: ${pos['usd_balance']:.2f}")
                            print(f"  Side: {'LONG' if pos['position'] > 0 else 'SHORT' if pos['position'] < 0 else 'NEUTRAL'}")
                            if pos.get('entry_price') and pos['entry_price'] > 0:
                                print(f"  Entry: ${pos['entry_price']:.2f}")
                        
                        # Get current price
                        price_response = requests.get(f"{self.server_url}/api/price/btcusd", timeout=5)
                        if price_response.status_code == 200:
                            price_data = price_response.json()
                            print(f"\nBTC/USD: ${price_data['price']:.2f}")
                        
                        if 'auto_trader' in status and status['auto_trader']:
                            at = status['auto_trader']
                            print(f"\nAuto Trader:")
                            print(f"  Active: {at['active']}")
                            print(f"  Strategy: {at['strategy']}")
                            print(f"  Trades Today: {at['trades_today']}")
                        
                        print("\n(Press Ctrl+C or use 'stop_monitor' to stop)")
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Monitor error: {e}")
                
                # Wait for interval or stop signal
                self.stop_monitoring.wait(interval)
            
            print("\nMonitoring stopped.")
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
    
    def do_stop_monitor(self, arg):
        """Stop real-time monitoring"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=2)
            print("Monitoring stopped.")
        else:
            print("No active monitoring to stop.")
    
    def do_get_price(self, symbol):
        """Get current price for a symbol
        Usage: get_price [symbol]"""
        symbol = symbol or 'btcusd'
        try:
            response = requests.get(f"{self.server_url}/api/price/{symbol}", timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                print(f"{data['symbol'].upper()}: ${data['price']:.2f}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error getting price: {e}")
    
    def do_get_data(self, args):
        """Get recent price data
        Usage: get_data [symbol] [limit] [frequency]
        Example: get_data btcusd 50 1H"""
        parts = args.split()
        symbol = parts[0] if parts else 'btcusd'
        limit = parts[1] if len(parts) > 1 else '100'
        frequency = parts[2] if len(parts) > 2 else '1H'
        
        try:
            response = requests.get(
                f"{self.server_url}/api/data/{symbol}",
                params={'limit': limit, 'frequency': frequency},
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                data = response.json()
                print(f"Recent {frequency} data for {symbol.upper()} (last {limit} bars):")
                if data.get('data'):
                    # Show last few entries
                    for entry in data['data'][-5:]:
                        timestamp = entry.get('time', entry.get('timestamp', 'N/A'))
                        print(f"  {timestamp}: O:{entry.get('open', 0):.2f} "
                              f"H:{entry.get('high', 0):.2f} "
                              f"L:{entry.get('low', 0):.2f} "
                              f"C:{entry.get('close', 0):.2f}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error getting data: {e}")
    
    def do_quick_buy(self, amount):
        """Quick buy BTC
        Usage: quick_buy <amount>"""
        if not amount:
            print("Usage: quick_buy <amount>")
            return
        
        try:
            response = requests.post(
                f"{self.server_url}/api/buy",
                json={'symbol': 'btcusd', 'amount': amount},
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                self.print_response(response.json())
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error placing buy order: {e}")
    
    def do_quick_sell(self, amount):
        """Quick sell BTC
        Usage: quick_sell <amount>"""
        if not amount:
            print("Usage: quick_sell <amount>")
            return
        
        try:
            response = requests.post(
                f"{self.server_url}/api/sell",
                json={'symbol': 'btcusd', 'amount': amount},
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                self.print_response(response.json())
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error placing sell order: {e}")
    
    def do_start_auto(self, args):
        """Start auto trading
        Usage: start_auto <amount> <position> [strategy]
        Example: start_auto 1.5btc long adaptive"""
        parts = args.split()
        if len(parts) < 2:
            print("Usage: start_auto <amount> <position> [strategy]")
            return
        
        amount = parts[0]
        position = parts[1]
        strategy = parts[2] if len(parts) > 2 else 'adaptive'
        
        try:
            response = requests.post(
                f"{self.server_url}/api/strategy/start",
                json={'amount': amount, 'position': position, 'strategy': strategy},
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                self.print_response(response.json())
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error starting auto trade: {e}")
    
    # Removed duplicate do_stop_auto - use do_stop_auto_trade instead
    
    def do_diagnostics(self, arg):
        """Get strategy diagnostics"""
        try:
            response = requests.get(f"{self.server_url}/api/strategy/diagnostics", timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                self.print_response(response.json())
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error getting diagnostics: {e}")
    
    def do_enable_commands(self, arg):
        """Enable local command interface for Claude Code
        This monitors the local commands/pending/ directory and forwards commands to the server"""
        if self.command_interface is None:
            self.command_interface = LocalCommandInterface(self)
            self.command_interface.start()
            print("\n✅ Local command interface enabled for Claude Code")
            print("\nClaude can now create JSON command files in:")
            print(f"   {self.command_interface.pending_dir}")
            print("\nExample command file content:")
            print(json.dumps({
                "timestamp": "2025-01-05T12:00:00Z",
                "command": "status",
                "source": "claude_check",
                "args": ""
            }, indent=2))
        else:
            print("Command interface is already enabled")
    
    def do_disable_commands(self, arg):
        """Disable local command interface"""
        if self.command_interface:
            self.command_interface.stop()
            self.command_interface = None
            print("✅ Command interface disabled")
        else:
            print("Command interface is not enabled")
    
    def do_recalc_pivots(self, arg):
        """Force recalculation of pivot levels"""
        response = self.send_command("recalc_pivots")
        self.print_response(response)
        
    def do_server_restart(self, arg):
        """Restart server with latest code changes
        
        Executes bin/quick_restart.sh on the server to:
        - Stop the server gracefully
        - Pull latest changes from git
        - Restart with data caching for fast startup
        """
        response = self.send_command("server_restart")
        if response and 'data' in response:
            data = response['data']
            if isinstance(data, str):
                print(data)
            else:
                print(json.dumps(data, indent=2))
                
    def do_whipsaw_stats(self, arg):
        """Show whipsaw statistics and analysis
        
        Displays:
        - Total whipsaws detected
        - Whipsaw losses
        - Average whipsaw cost
        - Recent whipsaw patterns
        - Whipsaw rate
        """
        response = self.send_command("whipsaw_stats")
        self.print_response(response)
    
    def do_show_trade_sequence(self, arg):
        """Show recent trade sequence for whipsaw analysis
        Usage: show_trade_sequence [hours]
        Default: 24 hours
        """
        response = self.send_command(f"show_trade_sequence {arg}")
        self.print_response(response)
    
    def do_help(self, arg):
        """Show available commands"""
        if arg:
            super().do_help(arg)
        else:
            print("""
Available Commands:
==================

All standard TDR commands are forwarded to the server:
  status, buy, sell, auto_trade, stop_auto_trade, positions, whipsaw_stats, etc.

Local Client Commands:
  logs [n] [type] [search]  - View server logs
  show_diagnostics [type] [n] - View diagnostic events  
  trades [n]                - Show recent trades
  load_history              - Start loading historical data
  history_status            - Check history loading progress
  poll_server [interval]    - Monitor server events (q+Enter or Ctrl+C to stop)
  monitor [interval]        - Real-time price/position display
  stop_monitor              - Stop real-time monitoring
  enable_commands           - Enable Claude command interface
  disable_commands          - Disable command interface
  reconnect                 - Reinitialize server connection
  server                    - Show server info
  help [command]            - Show help
  quit/exit                 - Exit client
""")
    
    def do_quit(self, arg):
        """Exit the client"""
        print("Disconnecting from TDR server...")
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        if hasattr(self, 'stop_polling'):
            self.stop_polling.set()
        if hasattr(self, 'polling_thread') and self.polling_thread:
            self.polling_thread.join(timeout=2)
        if self.command_interface:
            self.command_interface.stop()
        return True
    
    def do_exit(self, arg):
        """Exit the client"""
        return self.do_quit(arg)
    
    def do_EOF(self, arg):
        """Handle Ctrl+D"""
        print()  # New line
        return self.do_quit(arg)

def main():
    """Main entry point for TDR client"""
    parser = argparse.ArgumentParser(description='TDR Trading Client')
    parser.add_argument('--server', type=str, help='Server URL (default: http://localhost:4000)')
    parser.add_argument('--config', type=str, default='best_strategy.json', help='Configuration file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--command', type=str, help='Execute single command and exit')
    
    args = parser.parse_args()
    
    # Reinitialize logger with verbose setting
    global logger
    logger = setup_client_logging(verbose=args.verbose)
    
    # Determine server URL
    server_url = args.server or os.environ.get('TDR_SERVER_URL', DEFAULT_SERVER_URL)
    
    logger.info(f"Starting TDR Client - Server: {server_url}, Config: {args.config}")
    print(f"TDR Client - Connecting to server")
    print(f"Server: {server_url}")
    print(f"Config: {args.config} (will only be sent if server needs initialization)")
    
    # Create client
    client = RemoteTDRClient(server_url, config_file=args.config, verbose=args.verbose)
    
    # Execute single command if provided
    if args.command:
        response = client.send_command(args.command)
        client.print_response(response)
        sys.exit(0 if response.get('success', True) else 1)
    
    # Start interactive shell
    try:
        client.cmdloop()
    except KeyboardInterrupt:
        print("\nExiting...")
        client.do_quit('')

if __name__ == '__main__':
    main()