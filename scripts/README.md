# PHTN.AI Sub-Agent Framework - Management Scripts

This directory contains shell scripts for managing the PHTN.AI Sub-Agent Framework.

## 📋 Available Scripts

### 🚀 `start.sh` - Start the Sub-Agent

Starts the sub-agent framework server in the background.

```bash
./scripts/start.sh
```

**Features:**
- ✅ Pre-flight checks (port availability, configuration file)
- ✅ Starts server in background with nohup
- ✅ Saves PID to `.sub-agent.pid`
- ✅ Logs output to `logs/sub-agent.log`
- ✅ Displays agent configuration
- ✅ Shows quick command reference

**Output:**
- Server URL: http://localhost:8000
- Dashboard: http://localhost:8000/dashboard
- API Docs: http://localhost:8000/docs
- PID file: `.sub-agent.pid`
- Log file: `logs/sub-agent.log`

---

### 🛑 `stop.sh` - Stop the Sub-Agent

Gracefully stops the running sub-agent server.

```bash
./scripts/stop.sh
```

**Features:**
- ✅ Graceful shutdown (SIGTERM)
- ✅ Force kill after 5 seconds if needed
- ✅ Cleans up PID file
- ✅ Handles stale PID files
- ✅ Can stop by port if PID file missing

**Exit Codes:**
- 0: Successfully stopped
- 1: Failed to stop

---

### 📊 `status.sh` - Check Server Status

Displays the current status of the sub-agent server.

```bash
./scripts/status.sh
```

**Information Displayed:**
- ✅ Running status (RUNNING/NOT RUNNING)
- ✅ Process ID (PID)
- ✅ CPU and memory usage
- ✅ Start time
- ✅ Server URL and port
- ✅ API health check
- ✅ Agent configuration (name, version, skills)
- ✅ Execution pattern
- ✅ File locations and sizes

**Exit Codes:**
- 0: Server is running
- 1: Server is not running

---

### 🔄 `restart.sh` - Restart the Sub-Agent

Stops and then starts the sub-agent server.

```bash
./scripts/restart.sh
```

**Features:**
- ✅ Calls `stop.sh` followed by `start.sh`
- ✅ Waits 2 seconds between stop and start
- ✅ Useful for applying configuration changes

---

## 🎯 Common Usage Patterns

### Start the Server
```bash
cd /path/to/framework
./scripts/start.sh
```

### Check if Running
```bash
./scripts/status.sh
```

### View Logs in Real-Time
```bash
tail -f logs/sub-agent.log
```

### Stop the Server
```bash
./scripts/stop.sh
```

### Restart After Config Changes
```bash
./scripts/restart.sh
```

### Quick Test
```bash
# Start
./scripts/start.sh

# Check status
./scripts/status.sh

# Test API
curl http://localhost:8000/health

# View agent card
curl http://localhost:8000/.well-known/agent-card.json | python3 -m json.tool

# Open dashboard
open http://localhost:8000/dashboard

# Stop
./scripts/stop.sh
```

---

## 📁 Generated Files

The scripts create and manage the following files:

| File | Purpose | Location |
|------|---------|----------|
| `.sub-agent.pid` | Process ID of running server | Framework root |
| `logs/sub-agent.log` | Server output and logs | `logs/` directory |

---

## 🔧 Configuration

The scripts use the following default configuration:

- **Port**: 8000
- **Host**: 0.0.0.0 (listens on all interfaces)
- **Log Directory**: `logs/`
- **PID File**: `.sub-agent.pid`
- **Config File**: `.phtnai/PHTN-AGENT.json`

---

## 🌐 Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Health check |
| `/ready` | Readiness check |
| `/dashboard` | Execution monitor dashboard |
| `/docs` | Swagger API documentation |
| `/redoc` | ReDoc API documentation |
| `/.well-known/agent-card.json` | A2A Agent Card |
| `/api/v2/` | JSON-RPC 2.0 endpoint |
| `/api/v2/execution/stream` | SSE execution stream |

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find process using port 8000
lsof -ti:8000

# Kill the process
kill -9 $(lsof -ti:8000)

# Or use the stop script
./scripts/stop.sh
```

### Stale PID File
```bash
# Remove stale PID file
rm -f .sub-agent.pid

# Or let stop.sh handle it
./scripts/stop.sh
```

### Server Won't Start
```bash
# Check configuration
ls -la .phtnai/PHTN-AGENT.json

# Check logs
cat logs/sub-agent.log

# Verify Python dependencies
pip3 list | grep -E "fastapi|uvicorn|pydantic|openai|anthropic"
```

### Server Not Responding
```bash
# Check if process is running
./scripts/status.sh

# Check logs for errors
tail -50 logs/sub-agent.log

# Restart the server
./scripts/restart.sh
```

### Authentication Issues on Dashboard
If you see "No valid authentication provided" when accessing `/dashboard`:
1. Ensure the server has been restarted after code changes
2. The dashboard endpoint should be exempt from authentication

---

## 🔒 Permissions

The scripts require execute permissions. If you get a "Permission denied" error:

```bash
chmod +x scripts/*.sh
```

---

## 📊 Monitoring

### Real-Time Logs
```bash
tail -f logs/sub-agent.log
```

### Server Status
```bash
watch -n 5 './scripts/status.sh'
```

### API Health Check
```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

### Execution Stream (SSE)
```bash
curl -N http://localhost:8000/api/v2/execution/stream
```

### Process Monitoring
```bash
ps aux | grep "run_agent.py"
```

---

## 🚀 Production Deployment

For production, consider:

1. **Use a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Use a Process Manager** (systemd, supervisor, PM2)
   ```bash
   # Example systemd service
   sudo systemctl start phtn-sub-agent
   sudo systemctl enable phtn-sub-agent
   ```

3. **Set Up Log Rotation**
   ```bash
   # /etc/logrotate.d/phtn-sub-agent
   /path/to/logs/sub-agent.log {
       daily
       rotate 7
       compress
       missingok
       notifempty
   }
   ```

4. **Configure Monitoring** (Prometheus, Grafana, ELK)

5. **Use Helm Charts** (see `phtnai-iac-framework/helm-charts`)

---

## 📚 Related Documentation

- **PHTN-AGENT.json**: Agent configuration file
- **PHTN-AGENT-SCHEMA_v2.json**: Configuration schema
- **API Documentation**: http://localhost:8000/docs

---

## 🤝 Contributing

To add new management scripts:

1. Create a new `.sh` file in this directory
2. Add execute permissions: `chmod +x scripts/your-script.sh`
3. Follow the existing script structure (colors, error handling)
4. Update this README with documentation

---

**Happy Agent Building! 🎊**
