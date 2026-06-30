# Story Generator Agent

Generates INVEST-compliant user stories with acceptance criteria from epics and features. Built on the PHTN.AI sub-agent framework with governance scoring and story refinement.

## Repository

**GitHub:** https://github.com/arunvelmurugan-jpg/story_generator_agent

## Quick Start (Local)

```bash
# 1. Clone
git clone https://github.com/arunvelmurugan-jpg/story_generator_agent.git
cd story_generator_agent

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# 4. Run
python run_agent.py
```

Server starts on **http://localhost:8080** (or `PORT` from `.env`).

## API Endpoints

Base path: `/sub-ba-story-generate`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/sub-ba-story-generate/health` | Health check |
| `GET` | `/sub-ba-story-generate/livez` | Liveness probe |
| `GET` | `/sub-ba-story-generate/readyz` | Readiness probe |
| `GET` | `/sub-ba-story-generate/docs` | Swagger UI |
| `GET` | `/sub-ba-story-generate/.well-known/agent-card.json` | A2A agent discovery |
| `POST` | `/sub-ba-story-generate/run` | Generate user stories |
| `GET` | `/sub-ba-story-generate/agent/info` | Agent metadata |

### Generate Stories (`POST /sub-ba-story-generate/run`)

**Request:**
```json
{
  "epics": [
    {
      "id": "EPIC-1",
      "title": "User Authentication",
      "description": "Allow users to sign in securely"
    }
  ],
  "capabilities": [],
  "domain": "e-commerce",
  "grounded": true,
  "fe_only": false
}
```

**Response:**
```json
{
  "stories": [...],
  "us_governance": {...},
  "input_tokens": 1200,
  "output_tokens": 800
}
```

### Example curl

```bash
curl -X POST http://localhost:8080/sub-ba-story-generate/run \
  -H "Content-Type: application/json" \
  -d '{
    "epics": [{"id": "EPIC-1", "title": "Login", "description": "User login flow"}],
    "domain": "retail"
  }'
```

## Deploy (Live Endpoint)

### Option A — Render (recommended)

1. Push this repo to GitHub.
2. Go to [Render Dashboard](https://dashboard.render.com/) → **New** → **Blueprint**.
3. Connect `arunvelmurugan-jpg/story_generator_agent`.
4. Set `OPENAI_API_KEY` when prompted.
5. After deploy, your live base URL will be:
   ```
   https://story-generator-agent.onrender.com/sub-ba-story-generate
   ```
6. Set `AGENT_URL` in Render env vars to that full base URL.

### Option B — Docker

```bash
docker build -t story-generator-agent .
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=your-key \
  -e AGENT_URL=http://localhost:8080/sub-ba-story-generate \
  story-generator-agent
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes* | — | OpenAI API key |
| `LLM_PROVIDER` | No | `openai` | LLM provider |
| `OPENAI_MODEL` | No | `gpt-4o` | OpenAI model |
| `PORT` / `AGENT_PORT` | No | `8080` | Server port |
| `AGENT_URL` | No | `http://localhost:8080` | Public agent URL for discovery |

\* Required when `LLM_PROVIDER=openai`

## Project Structure

```
story_generator_agent/
├── run_agent.py          # Entry point (FastAPI + /run endpoint)
├── .phtnai/PHTN-AGENT.json
├── shared/engines/story_generator.py
├── api/                  # FastAPI app factory
├── core/                 # Agent engine
├── Dockerfile
└── render.yaml           # Render deployment blueprint
```
