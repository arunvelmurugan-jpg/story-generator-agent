#!/bin/bash

###############################################################################
# PHTN.AI Sub-Agent Framework - Stop Script
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
PORT=8000

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}║          🛑  Stopping PHTN.AI Sub-Agent Framework  🛑          ║${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo -e "${YELLOW}⚠️  No PID file found${NC}"
    
    # Try to find process by port
    PORT_PID=$(lsof -ti:$PORT 2>/dev/null)
    if [ -n "$PORT_PID" ]; then
        echo -e "${YELLOW}   Found process on port $PORT (PID: $PORT_PID)${NC}"
        echo -e "${BLUE}   Attempting to stop...${NC}"
        kill -TERM "$PORT_PID" 2>/dev/null
        sleep 2
        
        # Force kill if still running
        if ps -p "$PORT_PID" > /dev/null 2>&1; then
            echo -e "${YELLOW}   Process still running, force killing...${NC}"
            kill -9 "$PORT_PID" 2>/dev/null
        fi
        
        if ! ps -p "$PORT_PID" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Sub-agent stopped successfully${NC}"
        else
            echo -e "${RED}❌ Failed to stop sub-agent${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}   No sub-agent process found${NC}"
    fi
    exit 0
fi

# Read PID from file
PID=$(cat "$PID_FILE")

# Check if process is running
if ! ps -p "$PID" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Sub-agent is not running (stale PID: $PID)${NC}"
    rm -f "$PID_FILE"
    exit 0
fi

echo -e "${BLUE}🛑 Stopping sub-agent (PID: $PID)...${NC}"

# Try graceful shutdown first
kill -TERM "$PID" 2>/dev/null

# Wait for process to stop (max 5 seconds)
for i in {1..5}; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Sub-agent stopped gracefully${NC}"
        rm -f "$PID_FILE"
        echo ""
        echo -e "${BLUE}📊 To start again: ./scripts/start.sh${NC}"
        exit 0
    fi
    sleep 1
    echo -e "${YELLOW}   Waiting... ($i/5)${NC}"
done

# Force kill if still running
echo -e "${YELLOW}⚠️  Graceful shutdown timeout, force killing...${NC}"
kill -9 "$PID" 2>/dev/null
sleep 1

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Sub-agent stopped (forced)${NC}"
    rm -f "$PID_FILE"
    echo ""
    echo -e "${BLUE}📊 To start again: ./scripts/start.sh${NC}"
else
    echo -e "${RED}❌ Failed to stop sub-agent${NC}"
    echo -e "${RED}   You may need to manually kill the process: kill -9 $PID${NC}"
    exit 1
fi
