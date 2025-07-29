# Documentation Index

## Primary Documents

### 📋 [SYSTEM_DOCUMENTATION.md](SYSTEM_DOCUMENTATION.md)
**Main consolidated reference & USK** - Start here!
- System architecture and component locations
- Current trading positions and status  
- Development environment setup
- Critical operations and commands
- Recent work and future plans
- **USK (Update Session Knowledge) section at bottom**

### 🏗️ [ARCHITECTURE.md](ARCHITECTURE.md)
Technical architecture details
- Data flow between components
- Network configuration
- Directory structures
- Process architecture

### 📊 [LIVE_VS_BACKTEST_COMPARISON_PLAN.md](../btc-testing/docs/LIVE_VS_BACKTEST_COMPARISON_PLAN.md)
Plan for verifying backtest accuracy
- Comparison logging infrastructure
- Daily verification workflow
- Success criteria

## Legacy/Detailed Documents

These have been consolidated into SYSTEM_DOCUMENTATION.md but retained for reference:

### 🚨 `prompts/in-case-I-crash-details.md`
Emergency recovery information (now in SYSTEM_DOCUMENTATION.md)

### 🛠️ `prompts/dev-plan.md`  
Development process details (now in SYSTEM_DOCUMENTATION.md Section 5)

### 📝 `prompts/resume-claude-session.md`
Detailed session history (very long, kept for reference)

## Quick Reference

### Need to...
- **Check system status?** → See [SYSTEM_DOCUMENTATION.md#current-trading-status](SYSTEM_DOCUMENTATION.md#current-trading-status)
- **Start/stop servers?** → See [SYSTEM_DOCUMENTATION.md#critical-operations](SYSTEM_DOCUMENTATION.md#critical-operations)
- **Deploy changes?** → See [SYSTEM_DOCUMENTATION.md#git-workflow](SYSTEM_DOCUMENTATION.md#git-workflow)
- **Understand architecture?** → See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Run backtest comparison?** → See [LIVE_VS_BACKTEST_COMPARISON_PLAN.md](../btc-testing/docs/LIVE_VS_BACKTEST_COMPARISON_PLAN.md)

## Updates

When making significant changes:
1. Update `SYSTEM_DOCUMENTATION.md` first
2. Update specialized docs if needed
3. Note the date at bottom of updated files

*Last Updated: 2025-07-28*