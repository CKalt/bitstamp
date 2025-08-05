#!/bin/bash
# Live signal monitoring with clear formatting

echo "📊 MA 4/20 Signal Monitor - Live Updates"
echo "========================================"
echo ""

# Tail and format the logs
ssh ck "tail -f /home/chris/projects/bitstamp/logs/tdr_server.log" | while read line; do
    # Heartbeat - shows system is alive
    if [[ $line =~ "HEARTBEAT" ]]; then
        echo -e "\n💓 $(echo $line | grep -o 'at [0-9:]*')"
    
    # Signal evaluation - the key info
    elif [[ $line =~ "SIGNAL_EVAL v2" ]]; then
        # Extract values
        MA4=$(echo $line | grep -o 'MA4=[0-9]*' | cut -d= -f2)
        MA20=$(echo $line | grep -o 'MA20=[0-9]*' | cut -d= -f2)
        DIFF=$(echo $line | grep -o 'Diff=[-0-9]*' | cut -d= -f2)
        PROX=$(echo $line | grep -o 'Prox=[0-9.]*' | cut -d= -f2)
        SIG=$(echo $line | grep -o 'Sig=[-0-9]*' | cut -d= -f2)
        POS=$(echo $line | grep -o 'Pos=[-0-9]*' | cut -d= -f2)
        ACTION=$(echo $line | grep -o 'Action=[A-Z_]*' | cut -d= -f2)
        
        # Format signal name
        if [ "$SIG" = "1" ]; then
            SIG_NAME="LONG"
            SIG_ICON="📈"
        else
            SIG_NAME="SHORT"
            SIG_ICON="📉"
        fi
        
        # Format position name
        if [ "$POS" = "1" ]; then
            POS_NAME="LONG"
        elif [ "$POS" = "-1" ]; then
            POS_NAME="SHORT"
        else
            POS_NAME="NONE"
        fi
        
        # Determine if close to flip
        if (( $(echo "$PROX < 0.5" | bc -l) )); then
            ALERT="🚨 CLOSE TO FLIP!"
        elif (( $(echo "$PROX < 1.0" | bc -l) )); then
            ALERT="⚠️  Approaching flip"
        else
            ALERT="✅ Stable"
        fi
        
        echo ""
        echo "📊 Signal Check @ $(date +%H:%M:%S)"
        echo "   MA4:  \$$MA4"
        echo "   MA20: \$$MA20"
        echo "   Diff: \$$DIFF (${PROX}%)"
        echo "   Signal: $SIG_ICON $SIG_NAME | Position: $POS_NAME"
        echo "   Action: $ACTION"
        echo "   Status: $ALERT"
        
        # Predict next move
        if [ "$SIG" != "$POS" ] && [ "$ACTION" = "NO_TRADE" ]; then
            echo "   ⏳ Waiting for confirmation..."
        fi
    
    # Trade execution
    elif [[ $line =~ "Executing trade" ]] || [[ $line =~ "TRADE EXECUTED" ]]; then
        echo ""
        echo "🎯 ====== TRADE ALERT ======"
        echo "$line" | grep -o 'Executing.*'
        echo "==========================="
    
    # Errors
    elif [[ $line =~ "ERROR" ]] || [[ $line =~ "error" ]]; then
        echo "❌ ERROR: $(echo $line | grep -o 'ERROR.*')"
    fi
done