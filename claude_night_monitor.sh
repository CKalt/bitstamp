#!/bin/bash
# Claude Night Monitor - Keeps Claude informed of system status while you sleep

# Configuration
MONITOR_INTERVAL=300  # Check every 5 minutes
COMMANDS_DIR="/Users/chris/projects/python/btc/commands"
LOG_FILE="/Users/chris/projects/python/btc/claude_monitor.log"

echo "🌙 Claude Night Monitor Started at $(date)" | tee -a "$LOG_FILE"
echo "Monitoring interval: ${MONITOR_INTERVAL} seconds" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"

# Function to send command and wait for response
send_command() {
    local cmd_name=$1
    local cmd_json=$2
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Write command
    echo "$cmd_json" > "${COMMANDS_DIR}/pending/claude_monitor_${cmd_name}.json"
    
    # Wait for processing
    sleep 5
    
    # Check for result
    if [ -f "${COMMANDS_DIR}/processed/claude_monitor_${cmd_name}.json" ]; then
        echo "✅ Command completed: $cmd_name at $timestamp"
        cat "${COMMANDS_DIR}/processed/claude_monitor_${cmd_name}.json" | jq -r '.result.output' 2>/dev/null || cat "${COMMANDS_DIR}/processed/claude_monitor_${cmd_name}.json"
        rm -f "${COMMANDS_DIR}/processed/claude_monitor_${cmd_name}.json"
    else
        echo "⚠️ Command failed: $cmd_name at $timestamp"
    fi
}

# Main monitoring loop
while true; do
    echo -e "\n🕐 Check at $(date):" | tee -a "$LOG_FILE"
    
    # 1. Get detailed status
    echo -e "\n📊 System Status:" | tee -a "$LOG_FILE"
    send_command "status" '{
        "timestamp": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'",
        "command": "status long",
        "source": "claude_night_monitor",
        "args": ""
    }' | tee -a "$LOG_FILE"
    
    # 2. Check for any new trades
    echo -e "\n💹 Recent Trades:" | tee -a "$LOG_FILE"
    send_command "trades" '{
        "timestamp": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'",
        "command": "trades",
        "source": "claude_night_monitor",
        "args": ""
    }' | tee -a "$LOG_FILE"
    
    # 3. Check strategy diagnostics every 30 minutes
    if [ $(($(date +%M) % 30)) -lt 5 ]; then
        echo -e "\n🔍 Strategy Diagnostics:" | tee -a "$LOG_FILE"
        send_command "diag" '{
            "timestamp": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'",
            "command": "strategy_diagnostics",
            "source": "claude_night_monitor",
            "args": ""
        }' | tee -a "$LOG_FILE"
    fi
    
    # 4. Alert on significant events
    # Parse the status output and check for important changes
    if grep -q "PIVOT PROTECTION TRIGGERED" "$LOG_FILE"; then
        echo "🚨 ALERT: Pivot protection triggered!" | tee -a "$LOG_FILE"
        echo "System has flipped position due to pivot break" | tee -a "$LOG_FILE"
    fi
    
    echo -e "\n------- End of check -------\n" | tee -a "$LOG_FILE"
    
    # Show summary to Claude
    echo "📝 Monitoring Summary:"
    echo "- Monitoring active for $(ps -p $$ -o etime= | xargs)"
    echo "- Next check in ${MONITOR_INTERVAL} seconds"
    echo "- Log file: $LOG_FILE"
    echo ""
    echo "🛡️ Your position is protected by:"
    echo "- Pivot support at \$116,187"
    echo "- Automatic position flips"
    echo "- 10 trade daily limit"
    echo ""
    echo "💤 Sleep well - I'm watching the system!"
    
    sleep $MONITOR_INTERVAL
done