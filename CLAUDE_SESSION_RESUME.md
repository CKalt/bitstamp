# How to Resume Claude Sessions

## Quick Resume Command

```bash
claude --resume
```

This automatically continues from where we left off.

## If --resume Doesn't Work or After Mac Crash

### 1. Start Fresh Session with Context
```bash
# Option A: Point to the crash doc
claude "Please read prompts/in-case-I-crash-details.md and continue where we left off"

# Option B: With specific task context
claude "Read prompts/in-case-I-crash-details.md. We were working on [specific task]"
```

### 2. Key Files Claude Should Check

When resuming, I should immediately read:
1. `README_CLAUDE.md` - Critical architecture reminder
2. `prompts/in-case-I-crash-details.md` - Current system state & todo list
3. `.claude_critical_context.md` - Quick architecture reference
4. Check `screen -ls` to see what's running locally
5. Check `ssh ck "screen -ls"` to see what's running on server

### 3. Verify System State

```bash
# Check local screens
screen -ls

# Check server status
ssh ck "screen -ls"
ssh ck "cd /home/chris/projects/bitstamp && tail -20 logs/tdr_server.log | grep -E 'SIGNAL_EVAL|position'"

# Check current position
curl -s http://localhost:4000/api/status | jq '.position'
```

### 4. Common Resume Scenarios

#### After Normal Session End
```bash
claude --resume
# Claude automatically has context from previous session
```

#### After Mac Crash/Reboot
```bash
# Reattach to local screen if it survived
screen -r claude-tdr

# Or start fresh
claude "Read prompts/in-case-I-crash-details.md and check current trading status"
```

#### After Long Break (days/weeks)
```bash
claude "Read prompts/in-case-I-crash-details.md. Check current market position and catch me up on any issues"
```

## What Claude Retains vs Loses

### Retains with --resume:
- Previous conversation context
- Recent file edits
- Task progress
- Understanding of system architecture (if reminded)

### Loses and Needs Reminding:
- Which screens are running
- Current market position
- Recent trades
- Server status
- TODO list progress

## Best Practice for Ending Sessions

Before ending a session, ask Claude to:
1. Update `prompts/in-case-I-crash-details.md` with current status
2. Commit any pending changes
3. Document any half-finished tasks
4. Update the TODO list

Example:
```
"Please update the crash doc with current status and commit any changes before we end"
```

## Emergency Resume After Trading Issue

If trading system has issues while Claude is offline:
```bash
claude "URGENT: Read prompts/in-case-I-crash-details.md and check trading status. System may have issues."
```

## Pro Tip: Create Session Bookmarks

After complex work, create a bookmark:
```bash
# Ask Claude to create a session summary
"Create docs/session-$(date +%Y%m%d).md with everything we did today"
```

Then resume with:
```bash
claude "Read docs/session-20250728.md and continue"
```