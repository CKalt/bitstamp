# TDR Trading System Scripts

This directory contains organized scripts for managing, analyzing, and deploying the TDR trading system.

## Directory Structure

### setup/
Environment setup scripts for initializing development and test environments.
- `setup-dev-environment.sh` - Main development environment setup
- `setup-ggmap-dev.sh` - Configure ggmap shortcuts for easy navigation
- `setup-local-test-env.sh` - Local test environment setup
- `setup-local-tst.sh` - Quick local test setup
- `setup-test-branches.sh` - Git branch setup for test/dev

### deploy/
Deployment and synchronization scripts.
- `deploy-autoresume-fix.sh` - Deploy auto-resume fixes
- `fix-and-deploy.sh` - Fix issues and deploy in one step
- `fix-git-and-deploy.sh` - Fix git issues and deploy
- `push-autoresume-fix.sh` - Push auto-resume fixes to server
- `reset-test-branch.sh` - Reset test branch to stable

### manage/
Scripts for managing the dual-version (live/test) setup.
- `manage-dual-versions.sh` - Main dual-version management script

### analyze/
Analysis and debugging tools for understanding system behavior.
- `analyze_ma_performance.py` - Analyze MA crossover performance
- `analyze_signal_delays.py` - Check signal timing and delays
- `check_activity.py` - Monitor trading activity
- `check_ma_distance.py` - Check MA separation distances
- `check_ma_status.py` - Current MA status
- `check_signal_evaluation.py` - Verify signal evaluation logic
- `diagnose_live_system.py` - Comprehensive system diagnosis
- `get_ma_status.py` - Quick MA status check
- `get_ma_values.py` - Get current MA values
- `ma_flip_analysis.py` - Analyze MA flip patterns
- `trace_ma_decision_flow.py` - Trace decision-making flow
- `verify_trading_enabled.py` - Verify trading is enabled
- `test_dry_run_trading.py` - Test trading in dry run mode
- `calculate_ma_from_log.py` - Calculate MAs from price log

### config/
Configuration management scripts.
- `add_config_commands.py` - Add new configuration commands
- `config_commands_patch.py` - Patch configuration commands
- `configure_pure_ma.py` - Configure pure MA strategy
- `create_resume_file.py` - Create resume position file
- `enable_enhanced_logging.py` - Enable detailed logging
- `integrate_early_warning.py` - Integrate early warning system
- `resume_short_position.py` - Resume with SHORT position
- `update_server_config.py` - Update server configuration
- `view_server_config.py` - View current configuration

### backtest/
Backtesting tools for strategy validation.
- `backtest_30days_custom.py` - 30-day custom backtest
- `backtest_efficient.py` - Efficient backtesting implementation

### archive/
Old backups and temporary files (not tracked in git).

## Usage Examples

### Setting up a new test environment:
```bash
cd scripts/setup
./setup-dev-environment.sh
./setup-ggmap-dev.sh
```

### Deploying a fix:
```bash
cd scripts/deploy
./fix-and-deploy.sh
```

### Analyzing system behavior:
```bash
cd scripts/analyze
python check_ma_status.py
python diagnose_live_system.py
```

### Managing dual versions:
```bash
cd scripts/manage
./manage-dual-versions.sh status
./manage-dual-versions.sh switch test
```

## Notes

- Always run Python scripts with the virtual environment activated: `source source-venv.sh`
- Shell scripts may need to be run with appropriate permissions
- Some scripts require SSH access to the server (`ssh ck`)
- Configuration scripts modify live system settings - use with caution