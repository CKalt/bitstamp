#!/usr/bin/env python3
"""
Command Interface for External Control of TDR
Allows safe, auditable command submission to running trading system
"""

import os
import json
import time
import threading
import logging
from datetime import datetime
from pathlib import Path
import shutil


class CommandInterface:
    """File-based command queue for external control of trading system."""
    
    def __init__(self, shell, logger, base_dir="commands"):
        self.shell = shell
        self.logger = logger
        self.base_dir = Path(base_dir)
        self.pending_dir = self.base_dir / "pending"
        self.processed_dir = self.base_dir / "processed"
        self.failed_dir = self.base_dir / "failed"
        
        # Create directory structure
        for dir_path in [self.pending_dir, self.processed_dir, self.failed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.running = False
        self.monitor_thread = None
        self.command_counter = 0
        
    def start(self):
        """Start monitoring for commands."""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"Command interface started, monitoring {self.pending_dir}")
        
    def stop(self):
        """Stop monitoring for commands."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        self.logger.info("Command interface stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Check for new command files
                for cmd_file in sorted(self.pending_dir.glob("*.json")):
                    self._process_command_file(cmd_file)
                    
            except Exception as e:
                self.logger.error(f"Command monitor error: {e}")
                
            time.sleep(0.5)  # Check twice per second
            
    def _process_command_file(self, cmd_file):
        """Process a single command file."""
        try:
            # Read command
            with open(cmd_file, 'r') as f:
                cmd_data = json.load(f)
                
            # Validate command structure
            if not self._validate_command(cmd_data):
                self._move_to_failed(cmd_file, "Invalid command structure")
                return
                
            # Log command receipt
            self.logger.info(f"Processing command from {cmd_file.name}: {cmd_data.get('command')}")
            
            # Execute command
            result = self._execute_command(cmd_data)
            
            # Add result to command data
            cmd_data['result'] = result
            cmd_data['processed_at'] = datetime.now().isoformat()
            
            # Save to processed directory
            processed_file = self.processed_dir / cmd_file.name
            with open(processed_file, 'w') as f:
                json.dump(cmd_data, f, indent=2)
                
            # Remove from pending
            cmd_file.unlink()
            
            self.logger.info(f"Command processed successfully: {cmd_file.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to process command {cmd_file.name}: {e}")
            self._move_to_failed(cmd_file, str(e))
            
    def _validate_command(self, cmd_data):
        """Validate command structure."""
        required_fields = ['timestamp', 'command', 'source']
        return all(field in cmd_data for field in required_fields)
        
    def _execute_command(self, cmd_data):
        """Execute command in shell context."""
        command = cmd_data.get('command', '')
        args = cmd_data.get('args', '')
        
        # Build full command string
        full_command = f"{command} {args}".strip()
        
        # Check if command exists
        method_name = f"do_{command}"
        if not hasattr(self.shell, method_name):
            return {
                'success': False,
                'error': f"Unknown command: {command}",
                'output': None
            }
            
        try:
            # Capture output by temporarily redirecting
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            # Execute command
            self.shell.onecmd(full_command)
            
            # Get output
            output = buffer.getvalue()
            sys.stdout = old_stdout
            
            return {
                'success': True,
                'error': None,
                'output': output
            }
            
        except Exception as e:
            sys.stdout = old_stdout
            return {
                'success': False,
                'error': str(e),
                'output': None
            }
            
    def _move_to_failed(self, cmd_file, reason):
        """Move command to failed directory with error reason."""
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
            self.logger.error(f"Failed to move command to failed directory: {e}")
            
    def submit_command(self, command, args="", source="manual"):
        """Submit a command to the queue (for testing)."""
        self.command_counter += 1
        timestamp = datetime.now()
        
        cmd_data = {
            'timestamp': timestamp.isoformat(),
            'command': command,
            'args': args,
            'source': source,
            'session_id': f"{source}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            'sequence': self.command_counter
        }
        
        # Write to pending directory
        filename = f"cmd_{timestamp.strftime('%Y%m%d_%H%M%S')}_{self.command_counter:04d}.json"
        cmd_file = self.pending_dir / filename
        
        with open(cmd_file, 'w') as f:
            json.dump(cmd_data, f, indent=2)
            
        return filename