#!/usr/bin/env python3
"""
Safe integration of Early Warning Monitor.
This adds monitoring WITHOUT affecting trading logic.

SAFETY FEATURES:
1. Creates backup of strategies.py first
2. Only adds monitor calls, doesn't change trade logic  
3. Can be reverted with one command
4. Includes feature flag to disable
"""

import shutil
import os
from datetime import datetime

def create_backup():
    """Create timestamped backup of strategies.py"""
    source = "src/tdr_core/strategies.py"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"src/tdr_core/strategies.py.backup_{timestamp}"
    
    print(f"Creating backup: {backup}")
    shutil.copy2(source, backup)
    return backup

def add_early_warning_import():
    """Add import statement for early warning monitor."""
    import_line = "from .early_warning_monitor import EarlyWarningMonitor\n"
    
    with open("src/tdr_core/strategies.py", "r") as f:
        content = f.read()
    
    # Check if already imported
    if "early_warning_monitor" in content:
        print("Early warning already imported")
        return False
        
    # Add import after other imports
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("from datetime import"):
            lines.insert(i + 1, import_line)
            break
    
    with open("src/tdr_core/strategies.py", "w") as f:
        f.write("\n".join(lines))
    
    print("Added early warning import")
    return True

def add_monitor_initialization():
    """Add monitor initialization in __init__ method."""
    init_code = """
        # Early Warning Monitor (optional, read-only)
        self.enable_early_warning = kwargs.get('enable_early_warning', False)
        self.early_warning_monitor = None
        if self.enable_early_warning:
            try:
                self.early_warning_monitor = EarlyWarningMonitor(self.data_manager, self.logger)
                self.logger.info("Early Warning Monitor enabled")
            except Exception as e:
                self.logger.error(f"Failed to initialize Early Warning Monitor: {e}")
                self.early_warning_monitor = None
"""
    
    with open("src/tdr_core/strategies.py", "r") as f:
        content = f.read()
    
    # Find the end of __init__ method
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "def __init__" in line:
            # Find the end of __init__
            indent_count = len(line) - len(line.lstrip())
            for j in range(i + 1, len(lines)):
                if lines[j].strip() and not lines[j].startswith(" " * (indent_count + 4)):
                    # Found end of __init__, insert before
                    lines.insert(j - 1, init_code)
                    break
            break
    
    with open("src/tdr_core/strategies.py", "w") as f:
        f.write("\n".join(lines))
    
    print("Added monitor initialization")
    return True

def add_monitor_call():
    """Add monitor call in run_strategy loop."""
    monitor_code = """
            # Early Warning Monitor (optional, read-only)
            if self.early_warning_monitor and self.position != 0:
                try:
                    self.early_warning_monitor.monitor_step(self.position)
                except Exception as e:
                    self.logger.debug(f"Early warning check failed: {e}")
"""
    
    with open("src/tdr_core/strategies.py", "r") as f:
        content = f.read()
    
    # Find the main strategy loop
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "SIGNAL_EVAL v2:" in line:
            # Add monitor call after signal evaluation
            for j in range(i + 1, min(i + 20, len(lines))):
                if "self.diagnostic_logger.log_event" in lines[j]:
                    # Insert after diagnostic logging
                    indent = "                        "  # Match indentation
                    monitor_lines = [indent + l for l in monitor_code.strip().split("\n")]
                    for k, ml in enumerate(monitor_lines):
                        lines.insert(j + k + 1, ml)
                    break
            break
    
    with open("src/tdr_core/strategies.py", "w") as f:
        f.write("\n".join(lines))
    
    print("Added monitor call to strategy loop")
    return True

def create_config_update():
    """Create script to enable/disable early warning."""
    config_script = '''#!/usr/bin/env python3
"""
Enable or disable Early Warning Monitor
"""

import json
import sys

def update_config(enable=True):
    """Update configuration to enable/disable early warning."""
    try:
        with open("best_strategy.json", "r") as f:
            config = json.load(f)
        
        config["enable_early_warning"] = enable
        
        with open("best_strategy.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Early warning {'enabled' if enable else 'disabled'} in config")
        return True
    except Exception as e:
        print(f"Error updating config: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        enable = sys.argv[1].lower() in ['true', 'enable', 'on', '1']
        update_config(enable)
    else:
        print("Usage: python toggle_early_warning.py [true|false]")
'''
    
    with open("toggle_early_warning.py", "w") as f:
        f.write(config_script)
    
    os.chmod("toggle_early_warning.py", 0o755)
    print("Created toggle_early_warning.py script")

def verify_integration():
    """Verify the integration was successful."""
    with open("src/tdr_core/strategies.py", "r") as f:
        content = f.read()
    
    checks = [
        ("Import added", "from .early_warning_monitor import" in content),
        ("Initialization added", "self.early_warning_monitor" in content),
        ("Monitor call added", "monitor_step" in content),
    ]
    
    print("\nVerification:")
    all_good = True
    for name, check in checks:
        status = "✅" if check else "❌"
        print(f"  {status} {name}")
        all_good = all_good and check
    
    return all_good

def main():
    """Run the integration process."""
    print("Early Warning Monitor Integration")
    print("=" * 50)
    
    # Check if monitor exists
    if not os.path.exists("src/tdr_core/early_warning_monitor.py"):
        print("❌ Error: early_warning_monitor.py not found!")
        return False
    
    # Create backup
    backup_file = create_backup()
    print(f"Backup created: {backup_file}")
    
    try:
        # Add components
        add_early_warning_import()
        add_monitor_initialization()
        add_monitor_call()
        create_config_update()
        
        # Verify
        if verify_integration():
            print("\n✅ Integration successful!")
            print("\nNext steps:")
            print("1. Run tests: python tests/test_early_warning.py")
            print("2. Enable in config: python toggle_early_warning.py true")
            print("3. Restart server and monitor logs")
            print(f"\nTo revert: cp {backup_file} src/tdr_core/strategies.py")
            return True
        else:
            print("\n❌ Integration verification failed!")
            print(f"Revert with: cp {backup_file} src/tdr_core/strategies.py")
            return False
            
    except Exception as e:
        print(f"\n❌ Integration failed: {e}")
        print(f"Revert with: cp {backup_file} src/tdr_core/strategies.py")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)