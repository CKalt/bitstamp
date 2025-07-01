# SECURITY NOTICE - API CREDENTIALS

## Critical Files - NEVER SHARE OR UPLOAD:
- `.bitstamp` - Contains API keys and secrets
- `.influxdb` - Database credentials
- `.ses_smtp_config.json` - Email credentials

## Security Measures Implemented:

### 1. Files Protected by .gitignore:
✅ `.bitstamp` - Already in .gitignore
✅ `.influxdb` - Already in .gitignore  
✅ `.ses_smtp_config.json` - Already in .gitignore

### 2. For Claude Code / AI Assistant Work:
**NEVER:**
- Run `cat .bitstamp` or read credential files
- Share output that might contain API keys
- Upload these files to any service

**ALWAYS:**
- Work only with code logic, not credentials
- Use dummy/placeholder values in examples
- Keep credential files local only

### 3. Recommended Additional Security:

1. **Move to Environment Variables:**
   ```bash
   export BITSTAMP_API_KEY="your_key"
   export BITSTAMP_API_SECRET="your_secret"
   export BITSTAMP_CUSTOMER_ID="your_id"
   ```

2. **Set File Permissions:**
   ```bash
   chmod 600 .bitstamp
   chmod 600 .influxdb
   ```

3. **Create .claude-ignore file:**
   List files that should never be accessed by AI assistants

## Verification Commands:
```bash
# Check file permissions
ls -la .bitstamp

# Verify not tracked by git
git status --ignored

# Ensure credentials are not in git history
git log --all --full-history -- .bitstamp
```