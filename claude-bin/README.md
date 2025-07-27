# Claude Helper Scripts

This directory contains helper scripts specifically created for Claude Code operations and monitoring.

## Scripts

### Monitoring Scripts
- `check_ma_status.sh` - Quick MA status check via shell
- `check_status_api.py` - Check system status via API
- `monitor_crossover_live.py` - Live monitoring of MA crossovers
- `monitor_trigger_zone.sh` - Monitor when MAs are in trigger zone
- `proximity_check.sh` - Check MA proximity percentage
- `quick_diagnostic.sh` - Quick system diagnostic
- `verify_system_behavior.py` - Verify expected system behavior
- `verify-auto-trading.sh` - Verify auto-trading is active

### Testing Scripts
- `test_signal_evaluation.py` - Test signal evaluation logic

### Documentation
- `signal_evaluation_flow.md` - Signal evaluation flow documentation

### screen-monitoring/
Subdirectory for screen session monitoring tools.
- `clean-status.sh` - Clean status display for screen sessions

## Usage

These scripts are designed to be run from the project root:
```bash
./claude-bin/check_ma_status.sh
python claude-bin/monitor_crossover_live.py
```

Most monitoring scripts provide real-time or near real-time system status updates.