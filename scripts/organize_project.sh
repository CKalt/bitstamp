#!/bin/bash
# Script to organize all project files into a clean structure

echo "======================================"
echo "ORGANIZING PROJECT STRUCTURE"
echo "======================================"

# Create directory structure
echo "Creating directory structure..."
mkdir -p scripts/{setup,deploy,manage,analyze,config,backtest,archive}
mkdir -p docs
mkdir -p prompts
mkdir -p claude-bin/screen-monitoring
mkdir -p examples
mkdir -p tests

# Move setup scripts
echo "Moving setup scripts..."
mv -f setup-*.sh scripts/setup/ 2>/dev/null || true

# Move deployment scripts
echo "Moving deployment scripts..."
mv -f deploy-*.sh push-*.sh fix-and-deploy.sh fix-git-and-deploy.sh reset-test-branch.sh scripts/deploy/ 2>/dev/null || true

# Move management scripts
echo "Moving management scripts..."
mv -f manage-*.sh scripts/manage/ 2>/dev/null || true

# Move analysis scripts
echo "Moving analysis scripts..."
mv -f analyze_*.py check_*.py diagnose_*.py get_*.py trace_*.py verify_*.py ma_flip_analysis.py scripts/analyze/ 2>/dev/null || true
mv -f test_dry_run_trading.py scripts/analyze/ 2>/dev/null || true

# Move configuration scripts
echo "Moving configuration scripts..."
mv -f *config*.py create_resume_file.py enable_enhanced_logging.py integrate_early_warning.py resume_short_position.py scripts/config/ 2>/dev/null || true

# Move backtest scripts
echo "Moving backtest scripts..."
mv -f backtest_*.py scripts/backtest/ 2>/dev/null || true

# Move calculation scripts to analyze
echo "Moving calculation scripts..."
mv -f calculate_*.py scripts/analyze/ 2>/dev/null || true

# Move old backups to archive
echo "Moving old backups to archive..."
mv -f *.backup_* scripts/archive/ 2>/dev/null || true
mv -f calculation_comparisons.jsonl scripts/archive/ 2>/dev/null || true

# Keep claude-bin scripts in place (already organized)
echo "Claude-bin scripts are already organized"

# Keep docs in place (already organized)
echo "Documentation files are already in docs/"

# Keep prompts in place
echo "Prompt files are already in prompts/"

# List remaining untracked files
echo ""
echo "======================================"
echo "ORGANIZATION COMPLETE"
echo "======================================"
echo ""
echo "Remaining untracked files:"
git status --porcelain | grep "^??" | cut -d' ' -f2 | grep -v "^scripts/" | grep -v "^docs/" | grep -v "^claude-bin/" | grep -v "^prompts/" | sort

echo ""
echo "Directory structure created:"
echo "scripts/"
echo "├── setup/       # Environment setup scripts"
echo "├── deploy/      # Deployment and sync scripts"
echo "├── manage/      # Dual-version management"
echo "├── analyze/     # Analysis and debugging tools"
echo "├── config/      # Configuration management"
echo "├── backtest/    # Backtesting tools"
echo "└── archive/     # Old backups and temp files"
echo ""
echo "claude-bin/      # Claude-specific helper scripts"
echo "claude-temp-fixes/  # Temporary fix scripts"
echo "docs/            # Documentation"
echo "prompts/         # System prompts and context"