#!/bin/bash

###############################################################################
# PHTN.AI Sub-Agent Framework - Restart Script
###############################################################################

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}║         🔄  Restarting PHTN.AI Sub-Agent Framework  🔄         ║${NC}"
echo -e "${CYAN}║                                                                ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Stop the server
"$SCRIPT_DIR/stop.sh"

# Wait a moment
sleep 2

echo ""

# Start the server
"$SCRIPT_DIR/start.sh"
