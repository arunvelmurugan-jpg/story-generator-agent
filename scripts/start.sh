#!/bin/bash

###############################################################################
# PHTN.AI Sub-Agent Framework - Start Script
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
SUBAGENT_ROOT="$(dirname "$(dirname "$(dirname "$FRAMEWORK_DIR")")")"
PID_FILE="$FRAMEWORK_DIR/.sub-agent.pid"
LOG_FILE="$FRAMEWORK_DIR/logs/sub-agent.log"
PORT=8000

# Load environment variables from .env file if it exists
ENV_FILE="$FRAMEWORK_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    echo -e "${BLUE}📄 Loading environment from .env file${NC}"
    set -a
    source "$ENV_FILE"
    set +a
fi

# Environment variable defaults (can be overridden)
# OPENAI_API_KEY - Set this in .env or export before running
# ANTHROPIC_API_KEY - Set this in .env or export before running
# PHTN_MOCK_LLM_ENABLED - Set to "true" to enable mock LLM fallback (default: false)

# Create logs directory if it doesn't exist
mkdir -p "$FRAMEWORK_DIR/logs"

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}║          🤖  Starting PHTN.AI Sub-Agent Framework  🤖          ║${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Sub-agent is already running (PID: $PID)${NC}"
        echo -e "${YELLOW}   Use './scripts/stop.sh' to stop it first${NC}"
        exit 1
    else
        echo -e "${YELLOW}⚠️  Stale PID file found, removing...${NC}"
        rm -f "$PID_FILE"
    fi
fi

# Check if port is available
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}❌ Port $PORT is already in use${NC}"
    echo -e "${RED}   Please stop the process using this port${NC}"
    echo -e "${YELLOW}   Tip: Run 'lsof -ti:$PORT | xargs kill -9' to force stop${NC}"
    exit 1
fi

# Check if configuration exists
if [ ! -f "$FRAMEWORK_DIR/.phtnai/PHTN-AGENT.json" ]; then
    echo -e "${RED}❌ Configuration file not found: .phtnai/PHTN-AGENT.json${NC}"
    echo -e "${RED}   Please ensure the agent configuration file exists${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Pre-flight checks passed${NC}"
echo ""

# Change to subagent root directory (where run_agent.py is)
cd "$SUBAGENT_ROOT" || exit 1

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${BLUE}📦 Python Version: $PYTHON_VERSION${NC}"

# Check if virtual environment is activated
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${GREEN}✅ Virtual environment active: $VIRTUAL_ENV${NC}"
else
    echo -e "${YELLOW}⚠️  No virtual environment detected${NC}"
    echo -e "${YELLOW}   Consider using: python3 -m venv venv && source venv/bin/activate${NC}"
fi
echo ""

# Parse agent configuration
AGENT_NAME=$(python3 -c "import json; data=json.load(open('$FRAMEWORK_DIR/.phtnai/PHTN-AGENT.json')); print(data.get('name', 'Sub-Agent'))" 2>/dev/null)
AGENT_ID=$(python3 -c "import json; data=json.load(open('$FRAMEWORK_DIR/.phtnai/PHTN-AGENT.json')); print(data.get('agent_id', 'unknown'))" 2>/dev/null)
AGENT_VERSION=$(python3 -c "import json; data=json.load(open('$FRAMEWORK_DIR/.phtnai/PHTN-AGENT.json')); print(data.get('version', '1.0.0'))" 2>/dev/null)

echo -e "${BLUE}🤖 Agent Configuration:${NC}"
echo -e "   Name: $AGENT_NAME"
echo -e "   ID: $AGENT_ID"
echo -e "   Version: $AGENT_VERSION"
echo ""

# Display LLM configuration
echo -e "${BLUE}🧠 LLM Configuration:${NC}"
if [ -n "$OPENAI_API_KEY" ]; then
    echo -e "   ${GREEN}✅ OPENAI_API_KEY: configured${NC}"
else
    echo -e "   ${YELLOW}⚠️  OPENAI_API_KEY: not set${NC}"
fi
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo -e "   ${GREEN}✅ ANTHROPIC_API_KEY: configured${NC}"
else
    echo -e "   ${YELLOW}⚠️  ANTHROPIC_API_KEY: not set${NC}"
fi
if [ "$PHTN_MOCK_LLM_ENABLED" = "true" ]; then
    echo -e "   ${YELLOW}⚠️  Mock LLM: ENABLED (demo mode)${NC}"
else
    echo -e "   ${BLUE}   Mock LLM: disabled${NC}"
fi
echo ""

# Start the server
echo -e "${BLUE}🚀 Starting sub-agent server...${NC}"
echo -e "${BLUE}   Port: $PORT${NC}"
echo -e "${BLUE}   Log file: $LOG_FILE${NC}"
echo ""

# Start run_agent.py in background
nohup python3 run_agent.py > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Save PID
echo $SERVER_PID > "$PID_FILE"

# Wait a moment for server to start
sleep 3

# Check if server is running
if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Sub-agent started successfully!${NC}"
    echo -e "${GREEN}   PID: $SERVER_PID${NC}"
    echo -e "${GREEN}   URL: http://localhost:$PORT${NC}"
    echo ""
    
    # Try to get health check
    if command -v curl >/dev/null 2>&1; then
        sleep 2
        HEALTH_RESPONSE=$(curl -s http://localhost:$PORT/health 2>/dev/null)
        if [ $? -eq 0 ] && [ -n "$HEALTH_RESPONSE" ]; then
            echo -e "${GREEN}✅ Health check: PASSED${NC}"
        else
            echo -e "${YELLOW}⚠️  Health check: Server still starting...${NC}"
        fi
    fi
    
    echo ""
    echo -e "${GREEN}🎯 Quick Commands:${NC}"
    echo -e "   ${BLUE}View logs:${NC}        tail -f $LOG_FILE"
    echo -e "   ${BLUE}Stop server:${NC}      ./scripts/stop.sh"
    echo -e "   ${BLUE}Check status:${NC}     ./scripts/status.sh"
    echo -e "   ${BLUE}Agent card:${NC}       curl http://localhost:$PORT/.well-known/agent-card.json"
    echo -e "   ${BLUE}Dashboard:${NC}        open http://localhost:$PORT/dashboard"
    echo -e "   ${BLUE}API docs:${NC}         open http://localhost:$PORT/docs"
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║    Sub-Agent Framework is ready! Happy agent building! 🎊      ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
else
    echo -e "${RED}❌ Failed to start sub-agent${NC}"
    echo -e "${RED}   Check logs: $LOG_FILE${NC}"
    rm -f "$PID_FILE"
    exit 1
fi
