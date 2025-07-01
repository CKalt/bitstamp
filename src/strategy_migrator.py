# src/strategy_migrator.py
# Safe strategy configuration migration tool

import json
import os
import shutil
from datetime import datetime
import argparse
from typing import Dict, Any

class StrategyMigrator:
    """Safely migrate strategy configurations with validation and rollback."""
    
    def __init__(self, source='recommended_strategy.json', 
                 target='best_strategy.json',
                 backup_dir='strategy_backups'):
        self.source = source
        self.target = target
        self.backup_dir = backup_dir
        
        # Create backup directory if it doesn't exist
        os.makedirs(backup_dir, exist_ok=True)
        
    def create_backup(self) -> str:
        """Create timestamped backup of current configuration."""
        if not os.path.exists(self.target):
            print(f"No existing {self.target} to backup")
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{self.backup_dir}/best_strategy_{timestamp}.json"
        
        shutil.copy2(self.target, backup_name)
        print(f"✅ Created backup: {backup_name}")
        return backup_name
    
    def load_configurations(self) -> tuple[Dict, Dict]:
        """Load source and target configurations."""
        # Load recommended config
        with open(self.source, 'r') as f:
            recommended = json.load(f)
            
        # Load current config if exists
        current = {}
        if os.path.exists(self.target):
            with open(self.target, 'r') as f:
                current = json.load(f)
                
        return recommended, current
    
    def merge_configurations(self, recommended: Dict, current: Dict, 
                           preserve_live_settings: bool = True) -> Dict:
        """
        Merge recommended configuration with current, preserving critical settings.
        
        This ensures backward compatibility and preserves live trading settings.
        """
        # Start with the recommended configuration's optimal parameters
        merged = {
            "Frequency": recommended.get("optimal_parameters", {}).get("frequency", "1H"),
            "Strategy": recommended.get("optimal_parameters", {}).get("strategy", "MA"),
            "Short_Window": recommended.get("optimal_parameters", {}).get("short_window", 10),
            "Long_Window": recommended.get("optimal_parameters", {}).get("long_window", 46)
        }
        
        # Add performance metrics from backtest
        perf_metrics = recommended.get("performance_metrics", {})
        merged.update({
            "Final_Balance": 10000 * (1 + perf_metrics.get("total_return_pct", 0) / 100),
            "Total_Return": perf_metrics.get("total_return_pct", 0),
            "Total_Trades": perf_metrics.get("total_trades", 0),
            "Average_Trades_Per_Day": perf_metrics.get("avg_trades_per_day", 0),
            "Profit_Factor": perf_metrics.get("profit_factor", 1.0),
            "Sharpe_Ratio": perf_metrics.get("sharpe_ratio", 0)
        })
        
        # Preserve critical live trading settings if requested
        if preserve_live_settings and current:
            # Always preserve these from current config
            preserved_fields = [
                "do_live_trades",
                "Last_Signal_Timestamp",
                "Last_Signal_Action", 
                "Last_Trade_Timestamp",
                "Last_Trade_Price"
            ]
            
            for field in preserved_fields:
                if field in current:
                    merged[field] = current[field]
        
        # Set safe defaults for live trading
        if "do_live_trades" not in merged:
            merged["do_live_trades"] = False
            
        # Add adaptive strategy specific parameters if applicable
        if merged["Strategy"] == "AdaptiveMulti":
            adaptive_params = recommended.get("optimal_parameters", {})
            merged.update({
                "regime_switch_threshold": adaptive_params.get("regime_switch_threshold", 0.40),
                "signal_confirmation_bars": adaptive_params.get("signal_confirmation_bars", 2), 
                "min_trade_gap_minutes": adaptive_params.get("min_trade_gap_minutes", 15),
                "whipsaw_threshold": adaptive_params.get("whipsaw_threshold", 8.0),
                "auto_align_position": current.get("auto_align_position", False),
                "emergency_loss_threshold": adaptive_params.get("emergency_loss_threshold", -2000),
                "max_trades_per_day": adaptive_params.get("max_trades_per_day", 5)
            })
            
        # Add metadata about the migration
        merged["_migration_metadata"] = {
            "migrated_at": datetime.now().isoformat(),
            "source_file": self.source,
            "backtest_metadata": recommended.get("backtest_metadata", {}),
            "validation_status": recommended.get("validation_status", {})
        }
        
        return merged
    
    def validate_merged_config(self, config: Dict) -> bool:
        """Validate that merged configuration has all required fields."""
        required_fields = [
            "Frequency", "Strategy", "Short_Window", "Long_Window",
            "do_live_trades"
        ]
        
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
            return False
            
        # Validate field types and values
        validations = [
            ("Short_Window", lambda x: isinstance(x, int) and 1 <= x <= 100),
            ("Long_Window", lambda x: isinstance(x, int) and 1 <= x <= 200),
            ("do_live_trades", lambda x: isinstance(x, bool)),
            ("Short_Window", lambda x: config["Long_Window"] > x)  # Short < Long
        ]
        
        for field, validator in validations:
            if field in config and not validator(config[field]):
                print(f"❌ Invalid value for {field}: {config[field]}")
                return False
                
        print("✅ Configuration validation passed")
        return True
    
    def show_diff(self, current: Dict, new: Dict):
        """Display differences between configurations."""
        print("\n" + "="*60)
        print("CONFIGURATION CHANGES")
        print("="*60)
        
        # Find all keys
        all_keys = set(current.keys()) | set(new.keys())
        
        changes = []
        additions = []
        removals = []
        
        for key in sorted(all_keys):
            if key.startswith("_"):  # Skip metadata fields
                continue
                
            if key in current and key in new:
                if current[key] != new[key]:
                    changes.append((key, current[key], new[key]))
            elif key in new:
                additions.append((key, new[key]))
            else:
                removals.append((key, current[key]))
                
        if changes:
            print("\n📝 Modified Fields:")
            for key, old_val, new_val in changes:
                print(f"  {key}: {old_val} → {new_val}")
                
        if additions:
            print("\n➕ New Fields:")
            for key, val in additions:
                print(f"  {key}: {val}")
                
        if removals:
            print("\n➖ Removed Fields:")
            for key, val in removals:
                print(f"  {key}: {val}")
                
        if not changes and not additions and not removals:
            print("\n✅ No changes detected")
            
    def migrate(self, dry_run: bool = False, force: bool = False) -> bool:
        """
        Perform the migration.
        
        Args:
            dry_run: If True, show what would happen without making changes
            force: If True, proceed even if validation warns
            
        Returns:
            True if migration successful, False otherwise
        """
        print("="*60)
        print("STRATEGY CONFIGURATION MIGRATION")
        print("="*60)
        
        # Load configurations
        try:
            recommended, current = self.load_configurations()
        except Exception as e:
            print(f"❌ Error loading configurations: {e}")
            return False
            
        # Create merged configuration
        merged = self.merge_configurations(recommended, current)
        
        # Show changes
        if current:
            self.show_diff(current, merged)
        else:
            print("\n🆕 Creating new configuration (no existing config found)")
            
        # Validate
        if not self.validate_merged_config(merged):
            if not force:
                print("\n❌ Validation failed. Use --force to proceed anyway.")
                return False
            else:
                print("\n⚠️  Proceeding despite validation warnings (--force)")
                
        # Show performance expectations
        print("\n" + "="*60)
        print("EXPECTED PERFORMANCE")
        print("="*60)
        
        if current:
            current_return = current.get("Total_Return", 0)
            new_return = merged.get("Total_Return", 0)
            improvement = new_return - current_return
            
            print(f"Current Return: {current_return:.2f}%")
            print(f"New Return: {new_return:.2f}%")
            print(f"Expected Improvement: {improvement:+.2f}%")
        else:
            print(f"Expected Return: {merged.get('Total_Return', 0):.2f}%")
            
        print(f"Strategy: {merged.get('Strategy', 'Unknown')}")
        print(f"Parameters: MA({merged.get('Short_Window')}, {merged.get('Long_Window')})")
        
        if dry_run:
            print("\n" + "="*60)
            print("DRY RUN COMPLETE - No changes made")
            print("="*60)
            print("\nTo apply changes, run without --dry-run flag")
            
            # Save dry run output for review
            dry_run_file = "migration_dry_run.json"
            with open(dry_run_file, 'w') as f:
                json.dump(merged, f, indent=4)
            print(f"\nDry run configuration saved to: {dry_run_file}")
            
            return True
            
        # Perform actual migration
        print("\n" + "="*60)
        print("APPLYING MIGRATION")
        print("="*60)
        
        # Create backup
        backup_path = self.create_backup()
        
        # Write new configuration
        try:
            with open(self.target, 'w') as f:
                json.dump(merged, f, indent=4)
            print(f"✅ Successfully migrated to {self.target}")
            
            # Save migration report
            report = {
                "migration_timestamp": datetime.now().isoformat(),
                "backup_path": backup_path,
                "source_file": self.source,
                "target_file": self.target,
                "changes": {
                    "modified": dict(changes) if 'changes' in locals() else {},
                    "added": dict(additions) if 'additions' in locals() else {},
                    "removed": dict(removals) if 'removals' in locals() else {}
                }
            }
            
            report_file = f"{self.backup_dir}/migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=4)
            print(f"✅ Migration report saved to: {report_file}")
            
            print("\n" + "="*60)
            print("NEXT STEPS")
            print("="*60)
            print("1. Review the migrated configuration")
            print("2. Monitor the system in paper trading mode")
            print("3. Enable live trading when confident")
            print(f"\nRollback command: cp {backup_path} {self.target}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during migration: {e}")
            if backup_path:
                print(f"Backup available at: {backup_path}")
            return False

def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description="Safely migrate trading strategy configurations"
    )
    parser.add_argument('--source', type=str, default='recommended_strategy.json',
                        help='Source configuration file')
    parser.add_argument('--target', type=str, default='best_strategy.json',
                        help='Target configuration file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without making changes')
    parser.add_argument('--force', action='store_true',
                        help='Proceed even if validation warnings occur')
    parser.add_argument('--no-preserve', action='store_true',
                        help='Do not preserve live trading settings from current config')
    
    args = parser.parse_args()
    
    # Create migrator
    migrator = StrategyMigrator(args.source, args.target)
    
    # Run migration
    success = migrator.migrate(
        dry_run=args.dry_run,
        force=args.force
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())