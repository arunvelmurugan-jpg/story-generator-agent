#!/bin/bash

###############################################################################
# PHTN.AI Sub-Agent Framework - Status Script
###############################################################################

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$FRAMEWORK_DIR/.sub-agent.pid"
LOG_FILE="$FRAMEWORK_DIR/logs/sub-agent.log"
PORT=8000

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}║          📊  PHTN.AI Sub-Agent Framework Status  📊           ║${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if PID file exists
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Status: RUNNING${NC}"
        echo -e "${GREEN}   PID: $PID${NC}"
        
        # Get process info
        CPU_MEM=$(ps -p "$PID" -o %cpu,%mem | tail -1)
        echo -e "${BLUE}   CPU/Memory: $CPU_MEM${NC}"
        
        # Get uptime
        START_TIME=$(ps -p "$PID" -o lstart= 2>/dev/null)
        if [ -n "$START_TIME" ]; then
            echo -e "${BLUE}   Started: $START_TIME${NC}"
        fi
    else
        echo -e "${RED}❌ Status: NOT RUNNING${NC}"
        echo -e "${YELLOW}   (Stale PID file found: $PID)${NC}"
        exit 1
    fi
else
    # Check by port
    PORT_PID=$(lsof -ti:$PORT 2>/dev/null)
    if [ -n "$PORT_PID" ]; then
        echo -e "${YELLOW}⚠️  Status: RUNNING (no PID file)${NC}"
        echo -e "${YELLOW}   PID: $PORT_PID${NC}"
        echo -e "${YELLOW}   Note: Process not started by start.sh script${NC}"
    else
        echo -e "${RED}❌ Status: NOT RUNNING${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${BLUE}🌐 Server Information:${NC}"
echo -e "   Port: $PORT"
echo -e "   URL: http://localhost:$PORT"
echo -e "   Dashboard: http://localhost:$PORT/dashboard"
echo -e "   API Docs: http://localhost:$PORT/docs"
echo ""

# Try to get agent info from API
if command -v curl >/dev/null 2>&1; then
    AGENT_CARD=$(curl -s http://localhost:$PORT/.well-known/agent-card.json 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$AGENT_CARD" ]; then
        echo -e "${GREEN}✅ API: RESPONDING${NC}"
        echo ""
        
        # Parse agent card
        AGENT_NAME=$(echo "$AGENT_CARD" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('name', 'N/A'))" 2>/dev/null)
        AGENT_VERSION=$(echo "$AGENT_CARD" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('version', 'N/A'))" 2>/dev/null)
        AGENT_DESC=$(echo "$AGENT_CARD" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('description', 'N/A')[:60])" 2>/dev/null)
        SKILLS_COUNT=$(echo "$AGENT_CARD" | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('skills', [])))" 2>/dev/null)
        EXEC_PATTERN=$(echo "$AGENT_CARD" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('extensions', {}).get('phtnai', {}).get('executionPattern', 'N/A'))" 2>/dev/null)
        AGENT_STATUS=$(echo "$AGENT_CARD" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('extensions', {}).get('phtnai', {}).get('status', 'N/A'))" 2>/dev/null)
        
        echo -e "${BLUE}🤖 Agent Configuration:${NC}"
        echo -e "   Name: $AGENT_NAME"
        echo -e "   Version: $AGENT_VERSION"
        echo -e "   Description: $AGENT_DESC..."
        echo -e "   Skills/Tools: $SKILLS_COUNT"
        echo -e "   Execution Pattern: $EXEC_PATTERN"
        echo -e "   Status: $AGENT_STATUS"
    else
        echo -e "${YELLOW}⚠️  API: NOT RESPONDING${NC}"
        echo -e "${YELLOW}   Server may still be starting up...${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  curl not available, skipping API check${NC}"
fi

echo ""
echo -e "${BLUE}📁 Files:${NC}"
echo -e "   PID File: $PID_FILE"
echo -e "   Log File: $LOG_FILE"
echo -e "   Config: $FRAMEWORK_DIR/.phtnai/PHTN-AGENT.json"

if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
    echo -e "   Log Size: $LOG_SIZE"
fi

echo ""
echo -e "${BLUE}🎯 Quick Commands:${NC}"
echo -e "   ${GREEN}View logs:${NC}        tail -f $LOG_FILE"
echo -e "   ${GREEN}Stop server:${NC}      ./scripts/stop.sh"
echo -e "   ${GREEN}Restart:${NC}          ./scripts/restart.sh"
echo -e "   ${GREEN}Agent card:${NC}       curl http://localhost:$PORT/.well-known/agent-card.json | python3 -m json.tool"
echo -e "   ${GREEN}Dashboard:${NC}        open http://localhost:$PORT/dashboard"
echo -e "   ${GREEN}Health check:${NC}     curl http://localhost:$PORT/health"
