#!/usr/bin/env python3
"""
Send commands to running TDR system
Safe for use by Claude Code or other external systems
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def send_command(command, args="", source="claude_code"):
    """Send a command to the TDR command queue."""
    
    # Setup paths
    commands_dir = Path("commands/pending")
    commands_dir.mkdir(parents=True, exist_ok=True)
    
    # Create command data
    timestamp = datetime.now()
    cmd_data = {
        'timestamp': timestamp.isoformat(),
        'command': command,
        'args': args,
        'source': source,
        'session_id': f"{source}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    }
    
    # Generate filename
    filename = f"cmd_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.json"
    cmd_file = commands_dir / filename
    
    # Write command file
    with open(cmd_file, 'w') as f:
        json.dump(cmd_data, f, indent=2)
    
    print(f"Command submitted: {filename}")
    print(f"Command: {command} {args}")
    return str(cmd_file)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python send_tdr_command.py <command> [args]")
        print("Example: python send_tdr_command.py status")
        print("Example: python send_tdr_command.py strategy_diagnostics")
        sys.exit(1)
    
    command = sys.argv[1]
    args = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
    
    send_command(command, args)