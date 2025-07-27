#!/bin/bash
# Local management script for dual version development

# Configuration
SERVER="ck"
LIVE_DIR="/home/chris/projects/bitstamp"
DEV_DIR="/home/chris/projects/bitstamp-dev"
LOCAL_DIR=$(pwd)

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

function show_menu() {
    echo -e "${BLUE}=== Dual Version Management ===${NC}"
    echo "1) Check both version status"
    echo "2) View live logs"
    echo "3) View dev logs"
    echo "4) Deploy changes to dev"
    echo "5) Compare versions"
    echo "6) Promote dev to live"
    echo "7) Setup dev environment"
    echo "8) Emergency stop dev"
    echo "9) Restart dev server"
    echo "0) Exit"
    echo
}

function check_status() {
    echo -e "${BLUE}Checking both versions...${NC}"
    ssh $SERVER "cd $DEV_DIR && ./check-both-versions.sh 2>/dev/null" || \
    ssh $SERVER "cd $LIVE_DIR && curl -s http://localhost:4000/api/command -H 'Content-Type: application/json' -d '{\"command\": \"status\"}' | grep -E 'Position:|Entry' && echo '---' && curl -s http://localhost:4001/api/command -H 'Content-Type: application/json' -d '{\"command\": \"status\"}' | grep -E 'Position:|Entry'"
}

function view_live_logs() {
    echo -e "${GREEN}=== LIVE LOGS ===${NC}"
    ssh $SERVER "tail -f $LIVE_DIR/logs/tdr_server.log | grep -E 'SIGNAL_EVAL|TRADE DECISION|ERROR|Executing trade'"
}

function view_dev_logs() {
    echo -e "${YELLOW}=== DEV LOGS ===${NC}"
    ssh $SERVER "tail -f $DEV_DIR/logs/tdr_server_dev.log | grep -E 'SIGNAL_EVAL|TRADE DECISION|ERROR|Executing trade|DEV:'"
}

function deploy_to_dev() {
    echo -e "${BLUE}Deploying current changes to dev...${NC}"
    
    # Commit local changes
    echo "Committing local changes..."
    git add -A
    read -p "Commit message: " commit_msg
    git commit -m "DEV: $commit_msg" || {
        echo -e "${RED}No changes to commit${NC}"
        return
    }
    
    # Push to dev
    git push origin development || git push origin HEAD:development
    
    # Pull on server
    echo "Updating dev server..."
    ssh $SERVER "cd $DEV_DIR && git fetch && git checkout development && git pull origin development"
    
    echo -e "${GREEN}✅ Deployed to dev!${NC}"
    read -p "Restart dev server? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        restart_dev
    fi
}

function compare_versions() {
    echo -e "${BLUE}Comparing versions...${NC}"
    
    # Compare recent trades
    echo -e "\n${GREEN}=== RECENT TRADES ===${NC}"
    echo "LIVE:"
    ssh $SERVER "grep 'Executing trade' $LIVE_DIR/logs/tdr_server.log | tail -3"
    echo -e "\nDEV:"
    ssh $SERVER "grep 'Executing trade' $DEV_DIR/logs/tdr_server_dev.log | tail -3"
    
    # Compare positions
    echo -e "\n${GREEN}=== POSITION COMPARISON ===${NC}"
    ssh $SERVER "echo 'LIVE:' && grep 'position_tracking' $LIVE_DIR/logs/tdr_server.log | tail -1"
    ssh $SERVER "echo 'DEV:' && grep 'position_tracking' $DEV_DIR/logs/tdr_server_dev.log | tail -1"
    
    # Compare errors
    echo -e "\n${RED}=== RECENT ERRORS ===${NC}"
    echo "LIVE errors (last 24h):"
    ssh $SERVER "grep -c ERROR $LIVE_DIR/logs/tdr_server.log"
    echo "DEV errors (last 24h):"
    ssh $SERVER "grep -c ERROR $DEV_DIR/logs/tdr_server_dev.log"
}

function promote_to_live() {
    echo -e "${RED}⚠️  WARNING: This will promote dev code to live!${NC}"
    read -p "Are you sure? (yes/no) " confirm
    
    if [[ $confirm != "yes" ]]; then
        echo "Cancelled"
        return
    fi
    
    # Create backup tag
    backup_tag="backup-$(date +%Y%m%d-%H%M%S)"
    echo "Creating backup tag: $backup_tag"
    git tag $backup_tag
    git push origin $backup_tag
    
    # Merge development to main
    git checkout main
    git merge development -m "Promote dev to live after testing"
    git push origin main
    
    # Update live server
    ssh $SERVER "cd $LIVE_DIR && git pull origin main"
    
    echo -e "${GREEN}✅ Dev promoted to live!${NC}"
    echo "Backup tag created: $backup_tag"
    echo "To rollback: git checkout $backup_tag"
}

function setup_dev() {
    echo -e "${BLUE}Setting up dev environment on server...${NC}"
    
    # Copy setup script
    scp setup-dev-environment.sh $SERVER:/tmp/
    
    # Run setup
    ssh $SERVER "cd /home/chris/projects && bash /tmp/setup-dev-environment.sh"
    
    echo -e "${GREEN}✅ Dev environment setup complete!${NC}"
}

function emergency_stop() {
    echo -e "${RED}Emergency stopping dev trading...${NC}"
    
    # Stop auto trading
    ssh $SERVER "curl -X POST http://localhost:4001/api/command -H 'Content-Type: application/json' -d '{\"command\": \"stop_auto_trade\"}'"
    
    # Kill server
    ssh $SERVER "screen -S dev-trading -X quit 2>/dev/null || pkill -f 'python.*4001'"
    
    echo -e "${GREEN}✅ Dev trading stopped${NC}"
}

function restart_dev() {
    echo -e "${BLUE}Restarting dev server...${NC}"
    
    # Kill existing
    ssh $SERVER "screen -S dev-trading -X quit 2>/dev/null"
    sleep 2
    
    # Start new
    ssh $SERVER "cd $DEV_DIR && screen -dmS dev-trading ./start-dev-trading.sh"
    
    echo "Waiting for server to start..."
    sleep 10
    
    # Check if running
    ssh $SERVER "curl -s http://localhost:4001/api/command -H 'Content-Type: application/json' -d '{\"command\": \"status\"}' >/dev/null 2>&1" && \
        echo -e "${GREEN}✅ Dev server running${NC}" || \
        echo -e "${RED}❌ Dev server failed to start${NC}"
}

# Main loop
while true; do
    show_menu
    read -p "Select option: " choice
    
    case $choice in
        1) check_status ;;
        2) view_live_logs ;;
        3) view_dev_logs ;;
        4) deploy_to_dev ;;
        5) compare_versions ;;
        6) promote_to_live ;;
        7) setup_dev ;;
        8) emergency_stop ;;
        9) restart_dev ;;
        0) exit 0 ;;
        *) echo -e "${RED}Invalid option${NC}" ;;
    esac
    
    echo
    read -p "Press Enter to continue..."
    clear
done