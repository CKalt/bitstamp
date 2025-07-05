#!/usr/bin/env python
"""
Example script showing how Claude Code can interact with TDR client-server system
This demonstrates creating command files that the client will forward to the server
"""

import json
import os
from datetime import datetime
from pathlib import Path
import time

def send_tdr_command(command, args="", source="claude"):
    """
    Create a command file for TDR client to process and forward to server
    
    Args:
        command: The TDR command to execute (e.g., "status", "strategy_diagnostics")
        args: Arguments for the command
        source: Source identifier for tracking
    
    Returns:
        Path to the created command file
    """
    # Create command structure
    cmd_data = {
        "timestamp": datetime.now().isoformat(),
        "command": command,
        "args": args,
        "source": source,
        "session_id": f"{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    # Create filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"claude_{command}_{timestamp}.json"
    
    # Ensure commands directory exists
    commands_dir = Path("commands/pending")
    commands_dir.mkdir(parents=True, exist_ok=True)
    
    # Write command file
    cmd_path = commands_dir / filename
    with open(cmd_path, 'w') as f:
        json.dump(cmd_data, f, indent=2)
    
    print(f"Created command file: {cmd_path}")
    return cmd_path

def wait_for_result(command_file, timeout=30):
    """
    Wait for command to be processed and return result
    
    Args:
        command_file: Path to the command file
        timeout: Maximum seconds to wait
    
    Returns:
        Result data if found, None if timeout
    """
    processed_dir = Path("commands/processed")
    failed_dir = Path("commands/failed")
    
    start_time = time.time()
    filename = command_file.name
    
    while time.time() - start_time < timeout:
        # Check processed directory
        processed_file = processed_dir / filename
        if processed_file.exists():
            with open(processed_file, 'r') as f:
                data = json.load(f)
            print(f"Command processed successfully")
            return data
        
        # Check failed directory
        failed_file = failed_dir / filename
        if failed_file.exists():
            with open(failed_file, 'r') as f:
                data = json.load(f)
            error_file = failed_file.with_suffix('.error')
            if error_file.exists():
                with open(error_file, 'r') as f:
                    error_data = json.load(f)
                    print(f"Command failed: {error_data.get('reason', 'Unknown error')}")
            return data
        
        time.sleep(0.5)
    
    print(f"Timeout waiting for result after {timeout} seconds")
    return None

# Example usage for Claude Code
if __name__ == "__main__":
    print("TDR Client-Server Command Interface Example")
    print("=" * 50)
    print("\nMake sure:")
    print("1. TDR server is running on remote machine")
    print("2. TDR client is running locally with 'enable_commands' executed")
    print("3. This script is run in the same directory as the client")
    print("\n" + "=" * 50)
    
    # Example 1: Get status
    print("\n1. Getting system status...")
    cmd_file = send_tdr_command("status")
    result = wait_for_result(cmd_file)
    if result and result.get('result', {}).get('success'):
        output = result['result'].get('output', '')
        print("Status output:")
        print(output)
    
    # Example 2: Get strategy diagnostics
    print("\n2. Getting strategy diagnostics...")
    cmd_file = send_tdr_command("strategy_diagnostics")
    result = wait_for_result(cmd_file)
    if result and result.get('result', {}).get('success'):
        output = result['result'].get('output', '')
        # Show first few lines of diagnostics
        lines = output.split('\n')[:10]
        print("Diagnostics output (first 10 lines):")
        for line in lines:
            if line.strip():
                print(line)
    
    # Example 3: Get recent trades
    print("\n3. Getting recent trades...")
    cmd_file = send_tdr_command("trades", "5")
    result = wait_for_result(cmd_file)
    if result and result.get('result', {}).get('success'):
        output = result['result'].get('output', '')
        print("Recent trades:")
        print(output)
    
    print("\n" + "=" * 50)
    print("Example complete!")
    print("\nYou can use the send_tdr_command() function to execute any TDR command")
    print("and wait_for_result() to get the response from the remote server.")